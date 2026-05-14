# SPDX-License-Identifier: MIT
"""Lane 12 NeRV clean-inflate dependency closure checks.

These tests cover the narrow boundary between the `.nrv` mask payload and the
contest inflate/package path.  They intentionally avoid editing the shared
renderer while proving what is already wired and what still blocks promotion.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.nerv_mask_codec import NeRVMaskCodec, encode_nerv_codec
from tac.submission_archive import (
    RENDERER_NRV_MANIFEST,
    build_submission_archive,
    detect_pose_manifest,
    validate_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_nerv_mask_codec_top_level_import_closure_is_clean() -> None:
    """Inflate-time codec import needs only stdlib, numpy, and torch."""
    path = REPO_ROOT / "src" / "tac" / "nerv_mask_codec.py"
    tree = ast.parse(path.read_text())

    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                imports.add("<relative>")
            else:
                imports.add((node.module or "").split(".")[0])

    assert imports <= {"__future__", "dataclasses", "io", "numpy", "struct", "torch"}
    assert "tac" not in imports, (
        "tac.nerv_mask_codec must not import the wider tac stack at module import; "
        "inflate only needs decode_nerv_codec/render_mask_argmax."
    )


def _load_inflate_renderer_module():
    path = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
    spec = importlib.util.spec_from_file_location("_lane12_inflate_renderer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_contest_auth_eval_module():
    path = REPO_ROOT / "experiments" / "contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location("_lane12_contest_auth_eval", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_train_nerv_mask_module():
    path = REPO_ROOT / "experiments" / "train_nerv_mask.py"
    spec = importlib.util.spec_from_file_location("_lane12_train_nerv_mask", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_inflate_renderer_decodes_tiny_nrv_payload(tmp_path: Path) -> None:
    """`masks.nrv` can inflate through the renderer helper with only tac+torch."""
    codec = NeRVMaskCodec(num_freqs=1, hidden_dim=4, num_classes=5, depth=2)
    nrv_path = tmp_path / "masks.nrv"
    nrv_path.write_bytes(encode_nerv_codec(codec, weight_dtype="fp16"))

    inflate_mod = _load_inflate_renderer_module()
    masks = inflate_mod._load_masks_from_nrv(
        nrv_path,
        expected_frames=2,
        height=3,
        width=4,
    )

    assert isinstance(masks, torch.Tensor)
    assert masks.shape == (2, 3, 4)
    assert masks.dtype == torch.long
    assert int(masks.min()) >= 0
    assert int(masks.max()) < 5


def test_train_nerv_mask_segnet_source_uses_preprocess_input(
    monkeypatch, tmp_path: Path
) -> None:
    """Compress-time mask extraction must use SegNet's scorer preprocessing."""
    train_mod = _load_train_nerv_mask_module()

    class _FakeSegNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.preprocess_calls = 0

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            assert x.ndim == 5
            assert x.shape[1] == 1
            assert x.shape[2] == 3
            self.preprocess_calls += 1
            return x[:, -1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.ndim == 4
            b, _c, h, w = x.shape
            logits = torch.zeros(b, 5, h, w, device=x.device)
            logits[:, 2] = 1.0
            return logits

    fake_segnet = _FakeSegNet()
    fake_scorer = types.ModuleType("tac.scorer")
    fake_scorer.load_differentiable_scorers = lambda upstream_dir, device: (
        object(),
        fake_segnet,
    )
    monkeypatch.setitem(sys.modules, "tac.scorer", fake_scorer)

    video = tmp_path / "videos" / "0.mkv"
    video.parent.mkdir()
    video.write_bytes(b"stub")

    def fake_run(cmd, **_kwargs):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="8,4\n", stderr="")
        if cmd[0] == "ffmpeg":
            frames = np.zeros((2, 4, 8, 3), dtype=np.uint8)
            return subprocess.CompletedProcess(cmd, 0, stdout=frames.tobytes(), stderr=b"")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    masks = train_mod._load_segnet_argmax_masks(tmp_path, device="cpu", num_frames=2)

    assert tuple(masks.shape) == (2, 384, 512)
    assert fake_segnet.preprocess_calls == 1
    assert set(masks.unique().tolist()) == {2}


def _make_alpha_primitive_contract(
    train_mod,
    masks: torch.Tensor,
    *,
    source_path: Path,
    decoded_sha256: str | None = None,
    decoded_shape: list[int] | None = None,
) -> dict:
    decoded_shape = decoded_shape or [int(v) for v in masks.shape]
    return {
        "schema_version": 1,
        "diagnostic": "alpha_geo_primitive_contract_v1",
        "score_evidence_grade": "empirical",
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_eval_claim": False,
        "source": {
            "baseline": {
                "path": str(source_path),
                "archive_sha256": train_mod._sha256_file(source_path),
                "archive_size_bytes": int(source_path.stat().st_size),
                "decoded_mask_sha256": decoded_sha256
                or train_mod._mask_tensor_sha256(masks),
                "decoded_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
                "decoded_mask_shape": decoded_shape,
                "decoded_mask_dtype": str(masks.dtype),
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            },
            "failed_candidate": {
                "provenance_only": True,
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            },
        },
        "shape": {
            "frames": decoded_shape[0],
            "height": decoded_shape[1],
            "width": decoded_shape[2],
        },
        "protected_classes": [
            {"class_id": 1, "class_name": "class_1"},
            {"class_id": 2, "class_name": "class_2"},
        ],
        "ranked_critical_boxes": [
            {
                "rank": 1,
                "frames": [1],
                "class_id": 1,
                "box_xyxy": [1, 1, 4, 3],
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            }
        ],
        "decoded_baseline_boundary_recipes": [
            {
                "radius_px": 1,
                "source": "decoded_baseline_mask",
                "source_decoded_mask_sha256": decoded_sha256
                or train_mod._mask_tensor_sha256(masks),
                "protected_classes": [1, 2],
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            }
        ],
        "worst_transition_pairs": [
            {
                "rank": 1,
                "pair_index": 0,
                "frames": [0, 1],
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            }
        ],
        "threshold_gates": {
            "exploratory_retrain_gate": {
                "passed": True,
                "thresholds": {},
                "observed": {},
                "blockers": [],
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            },
            "exact_eval_spend_gate": {
                "passed": False,
                "thresholds": {},
                "observed": {},
                "blockers": ["not requested by trainer"],
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
            },
        },
    }


def test_train_nerv_mask_decoded_baseline_zip_source_records_custody(
    monkeypatch, tmp_path: Path
) -> None:
    """Alpha-Geo-1 target mode decodes baseline archive masks, not SegNet."""
    train_mod = _load_train_nerv_mask_module()
    import tac.mask_codec as mask_codec

    expected = torch.zeros(2, 4, 5, dtype=torch.long)
    expected[1, :, 2:] = 3
    payload = b"baseline-mask-video"
    archive = tmp_path / "baseline.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", payload)

    seen: dict[str, object] = {}

    def fake_decode_masks(path: Path, expected_frames: int | None = None) -> torch.Tensor:
        seen["path_name"] = Path(path).name
        seen["expected_frames"] = expected_frames
        seen["payload"] = Path(path).read_bytes()
        return expected

    monkeypatch.setattr(mask_codec, "decode_masks", fake_decode_masks)

    masks, metadata = train_mod._load_decoded_baseline_masks(
        archive,
        archive_member=None,
        expected_frames=2,
    )

    assert torch.equal(masks, expected)
    assert seen == {
        "path_name": "masks.mkv",
        "expected_frames": 2,
        "payload": payload,
    }
    assert metadata["archive_member_resolved"] == "masks.mkv"
    assert metadata["archive_member_sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["decoded_mask_sha256"] == train_mod._mask_tensor_sha256(expected)
    assert metadata["decoded_mask_shape"] == [2, 4, 5]


def test_train_nerv_mask_alpha_contract_rejects_decoded_sha_mismatch(
    tmp_path: Path,
) -> None:
    train_mod = _load_train_nerv_mask_module()
    masks = torch.zeros(2, 4, 5, dtype=torch.long)
    source_path = tmp_path / "baseline_masks.pt"
    torch.save(masks, source_path)
    contract = _make_alpha_primitive_contract(
        train_mod,
        masks,
        source_path=source_path,
        decoded_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="decoded_mask_sha256_match"):
        train_mod._validate_alpha_primitive_contract(
            contract,
            contract_metadata={
                "path": str(tmp_path / "contract.json"),
                "sha256": "c" * 64,
            },
            masks_THW=masks,
            decoded_metadata={"source_sha256": train_mod._sha256_file(source_path)},
        )


def test_train_nerv_mask_alpha_contract_sampling_pool_consumes_all_sources(
    tmp_path: Path,
) -> None:
    train_mod = _load_train_nerv_mask_module()
    masks = torch.zeros(2, 4, 5, dtype=torch.long)
    masks[1, :, 2:] = 1
    source_path = tmp_path / "baseline_masks.pt"
    torch.save(masks, source_path)
    contract = _make_alpha_primitive_contract(
        train_mod,
        masks,
        source_path=source_path,
    )
    contract_metadata = {
        "path": str(tmp_path / "contract.json"),
        "sha256": "d" * 64,
    }

    gates = train_mod._validate_alpha_primitive_contract(
        contract,
        contract_metadata=contract_metadata,
        masks_THW=masks,
        decoded_metadata={"source_sha256": train_mod._sha256_file(source_path)},
    )
    pool, sampling = train_mod._build_alpha_primitive_sampling_pool(
        contract,
        masks,
        seed=12,
        contract_sha256=contract_metadata["sha256"],
    )

    assert gates["overall_passed"] is True
    assert sampling["contract_sha256"] == contract_metadata["sha256"]
    assert sampling["applied_to_training"] is True
    assert sampling["score_claim_eligible"] is False
    component_records = {row["name"]: row for row in sampling["components"]}
    assert set(component_records) == {
        "uniform_base",
        "critical_boxes",
        "boundary_bands",
        "transition_endpoints",
    }
    assert component_records["critical_boxes"]["active"] is True
    assert component_records["boundary_bands"]["active"] is True
    assert component_records["transition_endpoints"]["active"] is True
    assert {component.name for component in pool.components} == set(component_records)


def test_train_nerv_mask_decoded_baseline_shape_gate_is_fail_closed() -> None:
    train_mod = _load_train_nerv_mask_module()

    with pytest.raises(ValueError, match="decoded-baseline masks must match"):
        train_mod._validate_decoded_baseline_target_shape(
            torch.zeros(1, 4, 5, dtype=torch.long),
            expected_frames=2,
            expected_height=4,
            expected_width=5,
        )


def test_train_nerv_mask_decoded_baseline_cli_smoke_records_non_promotable_provenance(
    tmp_path: Path,
) -> None:
    """CPU smoke covers Alpha-Geo-1 decoded-baseline custody without score claims."""
    train_mod = _load_train_nerv_mask_module()
    baseline = torch.zeros(2, 4, 5, dtype=torch.long)
    baseline[1, :, 2:] = 1
    baseline_path = tmp_path / "baseline_masks.pt"
    contract_path = tmp_path / "alpha_contract.json"
    output_dir = tmp_path / "nerv_smoke"
    torch.save(baseline, baseline_path)
    contract = _make_alpha_primitive_contract(
        train_mod,
        baseline,
        source_path=baseline_path,
    )
    contract_path.write_text(json.dumps(contract, sort_keys=True))

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "train_nerv_mask.py"),
            "--profile",
            "nerv_mask_lane_g_v3",
            "--device",
            "cpu",
            "--gt-masks-source",
            "decoded-baseline",
            "--decoded-baseline-path",
            str(baseline_path),
            "--alpha-primitive-contract",
            str(contract_path),
            "--output-dir",
            str(output_dir),
            "--num-frames",
            "2",
            "--mask-height",
            "4",
            "--mask-width",
            "5",
            "--steps",
            "1",
            "--eval-every",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "RESULT_JSON" in proc.stdout
    assert (output_dir / "masks.nrv").is_file()
    provenance = json.loads((output_dir / "provenance.json").read_text())
    target_metadata = provenance["target_mask_metadata"]
    expected_mask_sha = train_mod._mask_tensor_sha256(baseline)

    assert provenance["trainer_artifact_evidence_grade"] == "empirical"
    assert provenance["trainer_score_claim_eligible"] is False
    assert provenance["trainer_exact_eval_requested"] is False
    assert provenance["trainer_smoke_run"] is True
    assert "contest_auth_eval.py --device cuda" in provenance["trainer_score_claim_source_required"]
    assert any(
        "training device is cpu" in reason
        for reason in provenance["trainer_non_promotable_reasons"]
    )
    assert any(
        "not the full contest scorer geometry" in reason
        for reason in provenance["trainer_non_promotable_reasons"]
    )
    assert any(
        "steps=1 differs" in reason
        for reason in provenance["trainer_non_promotable_reasons"]
    )
    assert provenance["gt_masks_source"] == "decoded-baseline"
    assert provenance["requested_mask_shape"] == [2, 4, 5]
    assert provenance["target_mask_shape"] == [2, 4, 5]
    assert provenance["target_mask_sha256"] == expected_mask_sha
    assert target_metadata["source"] == "decoded-baseline"
    assert target_metadata["path"] == str(baseline_path)
    assert target_metadata["decoded_mask_sha256"] == expected_mask_sha
    contract_provenance = provenance["alpha_primitive_contract"]
    sampling = provenance["alpha_sampling_pool"]
    assert contract_provenance["path"] == str(contract_path)
    assert contract_provenance["sha256"] == hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    assert contract_provenance["consumption_gates"]["overall_passed"] is True
    assert sampling["mode"] == "alpha_primitive_contract_weighted"
    assert sampling["contract_sha256"] == contract_provenance["sha256"]
    assert sampling["score_claim_eligible"] is False
    components = {row["name"]: row for row in sampling["components"]}
    assert components["uniform_base"]["active"] is True
    assert components["critical_boxes"]["active"] is True
    assert components["boundary_bands"]["active"] is True
    assert components["transition_endpoints"]["active"] is True


def test_train_nerv_mask_segnet_cli_fails_closed_without_forensic_flag(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "train_nerv_mask.py"),
            "--profile",
            "nerv_mask_lane_g_v3",
            "--device",
            "cpu",
            "--gt-masks-source",
            "segnet",
            "--output-dir",
            str(tmp_path / "segnet_forbidden"),
            "--num-frames",
            "1",
            "--steps",
            "1",
            "--eval-every",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "--allow-forensic-segnet-target" in (proc.stderr + proc.stdout)


def test_train_nerv_mask_production_decoded_baseline_requires_alpha_contract(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "train_nerv_mask.py"),
            "--profile",
            "nerv_mask_lane_g_v3",
            "--device",
            "cuda",
            "--gt-masks-source",
            "decoded-baseline",
            "--decoded-baseline-path",
            str(tmp_path / "baseline_masks.pt"),
            "--output-dir",
            str(tmp_path / "nerv_production_no_contract"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    combined = proc.stderr + proc.stdout
    assert proc.returncode != 0
    assert "decoded-baseline production retraining requires --alpha-primitive-contract" in combined
    assert "torch.cuda.is_available" not in combined


def test_canonical_archive_validation_accepts_masks_nrv(tmp_path: Path) -> None:
    """Canonical manifest detection now recognizes Lane 12 masks.nrv."""
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"R" * 10_001)
        zf.writestr("masks.nrv", b"NRV1unit-test-payload")
        zf.writestr("optimized_poses.pt", b"P" * 2_000)

    manifest = detect_pose_manifest(archive)
    result = validate_archive(archive, manifest=manifest, strict=True)

    assert result.valid, result.summary()
    assert result.files_missing == []
    assert result.files_unexpected == []


def test_contest_auth_eval_allows_nrv_archive_member() -> None:
    """The exact archive eval harness does not reject Lane 12 .nrv payloads."""
    contest_auth_eval = _load_contest_auth_eval_module()

    contest_auth_eval._validate_archive_members(
        ["renderer.bin", "masks.nrv", "optimized_poses.pt"]
    )


def test_build_submission_archive_supports_masks_nrv_manifest(tmp_path: Path) -> None:
    renderer = tmp_path / "renderer.bin"
    masks = tmp_path / "masks.nrv"
    poses = tmp_path / "optimized_poses.pt"
    archive = tmp_path / "archive.zip"
    renderer.write_bytes(b"R" * 10_001)
    masks.write_bytes(b"NRV1unit-test-payload")
    poses.write_bytes(b"P" * 2_000)

    result = build_submission_archive(
        archive,
        renderer_bin=renderer,
        masks_nrv=masks,
        optimized_poses_pt=poses,
        manifest=RENDERER_NRV_MANIFEST,
        validate=True,
    )

    assert result.valid, result.summary()
    with zipfile.ZipFile(archive, "r") as zf:
        assert zf.namelist() == ["renderer.bin", "masks.nrv", "optimized_poses.pt"]
        assert zf.getinfo("masks.nrv").date_time == (1980, 1, 1, 0, 0, 0)


def test_remote_lane_nerv_defaults_to_alpha_geo_build_only_guardrail() -> None:
    """Remote Lane 12 defaults to canonical decoded-baseline build-only mode."""
    script = (REPO_ROOT / "scripts" / "remote_lane_nerv.sh").read_text()

    assert '"$PYBIN" -m pip install -e .' in script
    assert 'LANE_G_V3_BASE_ARCHIVE_REL="${LANE_G_V3_BASE_ARCHIVE_REL:-experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip}"' in script
    assert 'BASE_ARCHIVE="${BASE_ARCHIVE:-$LANE_G_V3_BASE_ARCHIVE}"' in script
    assert 'GT_MASKS_SOURCE="${GT_MASKS_SOURCE:-decoded-baseline}"' in script
    assert 'RUN_AUTH_EVAL="${RUN_AUTH_EVAL:-0}"' in script
    assert 'ALPHA_PRIMITIVE_CONTRACT="${ALPHA_PRIMITIVE_CONTRACT:-}"' in script
    assert "alpha_geo_primitive_contract_v1" in script
    assert "requires ALPHA_PRIMITIVE_CONTRACT" in script
    assert 'if [ -z "$ALPHA_PRIMITIVE_CONTRACT" ]; then' in script
    assert 'if [ ! -f "$ALPHA_PRIMITIVE_CONTRACT" ]; then' in script
    assert '"alpha_primitive_contract": optional_json_file_meta' in script
    assert "command -v nvidia-smi" in script
    assert 'GPU_NAME="nvidia-smi unavailable"' in script
    assert 'L2_CLEARANCE_PATH="${L2_CLEARANCE_PATH:-$WORKSPACE/.omx/state/lane12_nerv_l2_clearance.json}"' in script
    assert "No new NeRV retraining is allowed until this packet is valid" in script
    assert "cleared_for_retraining_unblock must be true" in script
    assert "grand_council_clean_passes must be an integer >= 3" in script
    assert "retired jsonfix40 target path" in script
    assert "requires POSE_REGEN_PROVENANCE" in script
    assert "ALLOW_STALE_POSE_AUTH_EVAL" not in script
    assert "requires ALPHA_GEO_PROVENANCE" in script
    assert "Alpha-Geo geometry gate did not pass" in script
    assert "no CUDA auth eval by guardrail" in script
    assert "duplicate BASE_ARCHIVE member" in script
    assert "unsafe BASE_ARCHIVE member path" in script
    assert "hidden/system BASE_ARCHIVE member" in script
    assert "unexpected BASE_ARCHIVE member" in script
    assert "validate_archive(dst, manifest=detect_pose_manifest(dst), strict=True)" in script
    assert "experiments/contest_auth_eval.py" in script
    assert "scripts/adjudicate_contest_auth_eval.py" in script
    assert 'if [ "$GT_MASKS_SOURCE" = "decoded-baseline" ]; then' in script
    assert '--decoded-baseline-path "$DECODED_BASELINE_PATH"' in script
    assert '--decoded-baseline-member "$DECODED_BASELINE_MEMBER"' in script
    assert '--alpha-primitive-contract "$ALPHA_PRIMITIVE_CONTRACT"' in script
    assert 'NERV_STEPS="${NERV_STEPS:-}"' in script
    assert 'NERV_EVAL_EVERY="${NERV_EVAL_EVERY:-}"' in script
    assert 'NERV_WEIGHT_DTYPE="${NERV_WEIGHT_DTYPE:-}"' in script
    assert "NERV_TRAINING_ARGS=()" in script
    assert 'NERV_TRAINING_ARGS+=(--steps "$NERV_STEPS")' in script
    assert 'NERV_TRAINING_ARGS+=(--eval-every "$NERV_EVAL_EVERY")' in script
    assert 'NERV_TRAINING_ARGS+=(--weight-dtype "$NERV_WEIGHT_DTYPE")' in script
    assert '"${NERV_TRAINING_ARGS[@]}"' in script
    assert '"nerv_steps_override": os.environ["NERV_STEPS"]' in script
    assert '"nerv_eval_every_override": os.environ["NERV_EVAL_EVERY"]' in script
    assert '"nerv_weight_dtype_override": os.environ["NERV_WEIGHT_DTYPE"]' in script


def test_wave_omega_2_nerv_wrapper_forwards_full_cuda_overrides() -> None:
    """Wave-Omega wrapper must not silently fall back to profile step defaults."""
    script = (REPO_ROOT / "scripts" / "wave_omega_2_nerv_full_cuda.sh").read_text()

    assert 'NERV_STEPS="${NERV_STEPS:-60000}"' in script
    assert 'NERV_EVAL_EVERY="${NERV_EVAL_EVERY:-1000}"' in script
    assert 'NERV_WEIGHT_DTYPE="${NERV_WEIGHT_DTYPE:-fp16}"' in script
    assert "remote_lane_nerv.sh is missing NERV_TRAINING_ARGS forwarding" in script
    assert "PROFILE_STEPS=" not in script
    assert 'export NERV_STEPS NERV_EVAL_EVERY NERV_WEIGHT_DTYPE' in script


def test_remote_lane_nerv_retains_gated_exact_cuda_archive_eval() -> None:
    """Exact eval path remains canonical but is not the default dispatch mode."""
    script = (REPO_ROOT / "scripts" / "remote_lane_nerv.sh").read_text()

    assert 'if [ "$RUN_AUTH_EVAL" != "1" ]; then' in script
    assert '--archive "$ARCHIVE"' in script
    assert "--inflate-sh submissions/robust_current/inflate.sh" in script
    assert "--device cuda" in script
    assert "AUTH_EVAL_DEVICE" not in script
    assert "--keep-work-dir" in script
    assert '--work-dir "$EVAL_WORK_DIR"' in script
    assert 'CONTEST_JSON="$EVAL_WORK_DIR/contest_auth_eval.json"' in script
    assert '--contest-json "$CONTEST_JSON"' in script
    assert '--result-copy "$RESULT_JSON"' in script
    assert 'score_delta_vs_lane_g_v3' in script
    assert "--max-sane-score 100.0" in script
    assert "refusing log JSON scrape" in script
    assert "auth_eval_renderer.py" not in script
