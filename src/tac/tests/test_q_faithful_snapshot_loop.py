# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "q_faithful_snapshot_loop.py"
REPACK = REPO / "experiments" / "repack_quantizr_faithful_qzs3_archive.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("q_faithful_snapshot_loop_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_repack():
    submissions_pkg = types.ModuleType("submissions")
    submissions_pkg.__path__ = [str(REPO / "submissions")]  # type: ignore[attr-defined]
    robust_pkg = types.ModuleType("submissions.robust_current")
    robust_pkg.__path__ = [str(REPO / "submissions" / "robust_current")]  # type: ignore[attr-defined]
    sys.modules["submissions"] = submissions_pkg
    sys.modules["submissions.robust_current"] = robust_pkg
    spec = importlib.util.spec_from_file_location("q_faithful_repack_test", REPACK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _declared_flags(path: Path) -> set[str]:
    text = path.read_text()
    return set(re.findall(r'add_argument\(\s*["\']--([a-z][a-z0-9-]+)', text))


def test_repack_command_uses_real_argparse_flags() -> None:
    loop = _load_script()

    cmd = loop.build_repack_command(
        python_bin="/opt/conda/bin/python",
        workspace=REPO,
        source_archive=Path("/tmp/raw/archive.zip"),
        output_dir=Path("/tmp/qzs3"),
        output_archive=Path("/tmp/qzs3/archive.zip"),
        renderer_codec="qzs3",
        qzs3_block_size=32,
        submission_layout="pr64_mask_first_single_blob",
        pose_codec="pose_qp1_v1",
        pose_residual_topk=0,
        brotli_quality=11,
    )

    used = {arg[2:] for arg in cmd if arg.startswith("--")}
    assert used <= _declared_flags(REPACK)
    assert "--source-archive" in cmd
    assert "--output-archive" in cmd
    assert "--renderer-codec" in cmd
    assert "pr64_mask_first_single_blob" in cmd
    assert "pose_qp1_v1" in cmd


def test_eval_invocation_carries_custody_and_source_sha_preflight() -> None:
    loop = _load_script()
    source_shas = {
        "scripts/q_faithful_snapshot_loop.py": "a" * 64,
        "submissions/robust_current/inflate.sh": "b" * 64,
    }

    cmd, env = loop.build_eval_invocation(
        workspace=REPO,
        archive_path=Path("/tmp/qzs3/archive.zip"),
        archive_label="qfaithful_snapshot_ckpt",
        log_dir=Path("/tmp/eval"),
        predicted_low=0.0,
        predicted_high=9.99,
        controlled_baseline="non-claiming snapshot",
        source_shas=source_shas,
        eval_script=REPO / "scripts" / "remote_archive_only_eval.sh",
    )

    assert cmd == ["bash", str(REPO / "scripts" / "remote_archive_only_eval.sh")]
    assert env["ARCHIVE_PATH"] == "/tmp/qzs3/archive.zip"
    assert env["ARCHIVE_LABEL"] == "qfaithful_snapshot_ckpt"
    assert env["LOG_DIR"] == "/tmp/eval"
    assert env["REQUIRED_SOURCE_SHA256S"].splitlines() == [
        "scripts/q_faithful_snapshot_loop.py=" + "a" * 64,
        "submissions/robust_current/inflate.sh=" + "b" * 64,
    ]


def test_dry_run_writes_non_claiming_manifest(tmp_path: Path) -> None:
    loop = _load_script()
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "renderer_epoch_001.pt"
    checkpoint.write_bytes(b"not a torch checkpoint; dry run only")
    masks = tmp_path / "masks.mkv"
    poses = tmp_path / "optimized_poses.pt"
    masks.write_bytes(b"mask")
    poses.write_bytes(b"pose")
    output_root = tmp_path / "snapshots"

    rc = loop.main(
        [
            "--workspace",
            str(REPO),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--min-checkpoint-age-seconds",
            "0",
            "--masks-mkv",
            str(masks),
            "--poses-pt",
            str(poses),
            "--output-root",
            str(output_root),
            "--dry-run",
            "--eval-mode",
            "command",
        ]
    )

    assert rc == 0
    manifests = list(output_root.glob("*/snapshot_manifest.json"))
    assert len(manifests) == 1
    text = manifests[0].read_text()
    assert '"status": "dry_run"' in text
    assert '"score_claim": false' in text
    assert '"exact_cuda_json_required": true' in text
    assert '"checkpoint"' in text
    assert '"export_command"' in text
    assert '"repack_command"' in text
    assert '"source_runtime_sha256s"' in text
    assert '"eval_roundtrip": true' in text
    assert '"snapshot_screen_contract"' in text
    assert '"promotable_exact_screen": false' in text
    assert "mask_frame_contract_unknown" in text
    assert "pose_tensor_unreadable" in text


def test_eval_run_fails_closed_without_dispatch_claim(tmp_path: Path) -> None:
    loop = _load_script()
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    masks = tmp_path / "masks.mkv"
    poses = tmp_path / "optimized_poses.pt"
    masks.write_bytes(b"mask")
    poses.write_bytes(b"pose")

    args = Namespace(
        checkpoint_dir=checkpoint_dir,
        masks_mkv=masks,
        poses_pt=poses,
        eval_mode="run",
        dispatch_claim_mode="none",
        existing_dispatch_claim_id=None,
    )

    with pytest.raises(loop.SnapshotError) as exc:
        loop.validate_static_inputs(args)
    assert exc.value.failure_class == "dispatch_claim_required"


def test_full_frame_snapshot_contract_preserves_promotable_screen(tmp_path: Path) -> None:
    loop = _load_script()
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"synthetic; operator declares full-frame")
    args = Namespace(
        masks_mkv=masks,
        mask_frame_contract="full",
        zoom_warp_path=None,
        submission_layout="pr64_mask_first_single_blob",
        renderer_codec="qzs3",
    )

    contract = loop.build_snapshot_screen_contract(args)

    assert contract["mask_frame_contract"]["contract"] == "full"
    assert contract["promotable_exact_screen"] is True
    assert contract["non_promotable_reasons"] == []
    assert contract["full_frame_archives_preserve_legacy_behavior"] is True
    loop.enforce_exact_screen_contract(contract, eval_mode="run")


def test_half_frame_without_zoom_warp_is_non_promotable_and_fails_run(tmp_path: Path) -> None:
    loop = _load_script()
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"synthetic; operator declares half-frame")
    args = Namespace(
        masks_mkv=masks,
        mask_frame_contract="half",
        zoom_warp_path=None,
        submission_layout="pr64_mask_first_single_blob",
        renderer_codec="qzs3",
    )

    contract = loop.build_snapshot_screen_contract(args)

    assert contract["mask_frame_contract"]["contract"] == "half"
    assert contract["promotable_exact_screen"] is False
    assert contract["non_promotable_reasons"] == [
        "half_frame_masks_without_zoom_warp_geometry"
    ]
    with pytest.raises(loop.SnapshotError) as exc:
        loop.enforce_exact_screen_contract(contract, eval_mode="run")
    assert exc.value.failure_class == "non_promotable_runtime_contract"


def test_half_frame_with_charged_zoom_warp_is_promotable_via_runtime_mask_expansion(
    tmp_path: Path,
) -> None:
    loop = _load_script()
    masks = tmp_path / "masks.mkv"
    zoom = tmp_path / "zoom_scalars.bin"
    masks.write_bytes(b"synthetic; operator declares half-frame")
    zoom.write_bytes(b"zoom geometry")
    args = Namespace(
        masks_mkv=masks,
        mask_frame_contract="half",
        zoom_warp_path=zoom,
        submission_layout="pr64_mask_first_single_blob",
        renderer_codec="qzs3",
    )

    contract = loop.build_snapshot_screen_contract(args)

    assert contract["zoom_warp_geometry"]["source"]["present"] is True
    assert contract["zoom_warp_geometry"]["archive_member_name"] == "zoom_scalars.bin"
    assert contract["zoom_warp_geometry"]["packed_in_repacked_archive"] is True
    assert contract["zoom_warp_geometry"]["required_renderer_consumption"] is False
    assert contract["zoom_warp_geometry"]["required_runtime_mask_expansion_consumption"] is True
    assert contract["renderer_zoom_contract"]["renderer_consumes_ego_flow"] is False
    assert contract["renderer_zoom_contract"]["runtime_consumes_zoom_warp_for_mask_expansion"] is True
    assert contract["promotable_exact_screen"] is True
    assert contract["non_promotable_reasons"] == []
    loop.enforce_exact_screen_contract(contract, eval_mode="run")


def test_raw_archive_charges_zoom_warp_geometry_member(tmp_path: Path) -> None:
    loop = _load_script()
    renderer = tmp_path / "renderer.bin"
    masks = tmp_path / "masks.mkv"
    poses = tmp_path / "optimized_poses.bin"
    zoom = tmp_path / "zoom_scalars.bin"
    archive = tmp_path / "archive.zip"
    renderer.write_bytes(b"renderer")
    masks.write_bytes(b"masks")
    poses.write_bytes(b"poses")
    zoom.write_bytes(b"zoom geometry")

    meta = loop.build_raw_archive(
        renderer_bin=renderer,
        masks_mkv=masks,
        poses_pt=poses,
        output_archive=archive,
        zoom_warp_path=zoom,
    )

    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == [
            "renderer.bin",
            "masks.mkv",
            "optimized_poses.bin",
            "zoom_scalars.bin",
        ]
        assert zf.read("zoom_scalars.bin") == b"zoom geometry"
    assert meta["members"]["zoom_scalars.bin"]["member_name"] == "zoom_scalars.bin"
    assert meta["members"]["zoom_scalars.bin"]["bytes"] == len(b"zoom geometry")
    assert len(meta["members"]["zoom_scalars.bin"]["sha256"]) == 64


def test_pr64_repack_preserves_zoom_warp_as_charged_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _load_script()
    repack = _load_repack()
    source = tmp_path / "source.zip"
    output = tmp_path / "out.zip"
    zoom_path = tmp_path / "zoom_scalars.bin"
    zoom_bytes = b"zoom geometry"
    zoom_path.write_bytes(zoom_bytes)
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", b"masks")
        zf.writestr("optimized_poses.bin", b"poses")
        zf.writestr("zoom_scalars.bin", zoom_bytes)

    monkeypatch.setattr(
        repack,
        "_renderer_bytes",
        lambda renderer_raw, renderer_codec, **_kwargs: (
            b"qzs3-renderer",
            {
                "renderer_codec": renderer_codec,
                "source_renderer_format": "test",
                "action": "passthrough_for_test",
            },
        ),
    )

    meta = repack.build_submission_archive(
        source,
        output,
        renderer_codec=repack.RENDERER_CODEC_QZS3,
        submission_layout=repack.SUBMISSION_LAYOUT_PR64_MASK_FIRST_SINGLE_BLOB,
        pose_codec="raw",
    )

    with zipfile.ZipFile(output) as zf:
        assert zf.namelist() == ["p", "zoom_scalars.bin"]
        assert zf.read("zoom_scalars.bin") == zoom_bytes
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"synthetic; operator declares half-frame")
    contract = loop.build_snapshot_screen_contract(
        Namespace(
            masks_mkv=masks,
            mask_frame_contract="half",
            zoom_warp_path=zoom_path,
            submission_layout="pr64_mask_first_single_blob",
            renderer_codec="qzs3",
        )
    )
    loop.validate_repacked_geometry_contract(
        screen_contract=contract,
        repacked_archive_meta=loop.archive_metadata(output),
    )
    assert meta["geometry_preservation"]["preserved"] is True
    assert meta["geometry_preservation"]["output_geometry_members"]["zoom_scalars.bin"][
        "bytes"
    ] == len(zoom_bytes)
    assert len(
        meta["geometry_preservation"]["output_geometry_members"]["zoom_scalars.bin"][
            "sha256"
        ]
    ) == 64


def test_qfai_export_metadata_carries_runtime_contract(tmp_path: Path) -> None:
    loop = _load_script()
    checkpoint = tmp_path / "renderer.pt"
    masks = tmp_path / "masks.mkv"
    checkpoint.write_bytes(b"checkpoint")
    masks.write_bytes(b"masks")
    args = Namespace(
        masks_mkv=masks,
        mask_frame_contract="full",
        zoom_warp_path=None,
        submission_layout="multi_member",
        renderer_codec="qzs3",
    )
    contract = loop.build_snapshot_screen_contract(args)

    meta = loop.qfai_export_contract_metadata(
        checkpoint=checkpoint,
        checkpoint_sha="c" * 64,
        profile="q_faithful_dilated_88k",
        screen_contract=contract,
        training_pose_contract={"training_pose_contract_promotable": True},
    )

    assert set(loop.QFAITHFUL_EXPORT_CONTRACT_REQUIRED_KEYS) <= set(meta)
    assert json.loads(json.dumps(meta)) == meta
    assert meta["runtime_contract_version"] == loop.SNAPSHOT_RUNTIME_CONTRACT_VERSION
    assert meta["mask_frame_contract"] == "full"
    assert meta["renderer_zoom_contract"]["architecture"] == (
        "quantizr_faithful_joint_frame_generator"
    )
    assert meta["promotable_exact_screen"] is True
    assert meta["packed_from_ema_shadow"] is False
    assert "pose_tensor_contract" in meta
    assert "training_pose_contract" in meta


def test_qfai_export_writes_raw_renderer_bin_for_qzs3_repack(tmp_path: Path) -> None:
    loop = _load_script()
    repack = _load_repack()
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

    checkpoint = tmp_path / "renderer.pt"
    model = build_quantizr_faithful_renderer()
    torch.save({"ema_shadow": model.state_dict()}, checkpoint)
    renderer_bin = tmp_path / "export" / "renderer.bin"

    meta = loop.export_qfai_renderer(
        checkpoint,
        renderer_bin,
        state_source="ema_shadow",
        brotli_quality=5,
        extra_meta={"test_contract": True},
    )

    raw_payload = renderer_bin.read_bytes()
    assert raw_payload[:4] == b"QFAI"
    assert Path(meta["raw_qfai_path"]).read_bytes() == raw_payload
    assert meta["renderer_bin_wire_format"] == "QFAI"
    assert meta["renderer_bin_brotli_compressed"] is False
    assert Path(meta["compressed_qfai_sidecar_path"]).is_file()
    qzs3_payload, qzs3_meta = repack._qzs3_renderer_bytes(raw_payload, block_size=32)
    assert qzs3_payload[:4] == b"QZS3"
    assert qzs3_meta["source_renderer_format"] == "QFAI"


def test_qfaithful_profile_is_roundtrip_proof() -> None:
    loop = _load_script()

    proof = loop.verify_eval_roundtrip_profile(REPO, "q_faithful_dilated_88k")

    assert proof == {
        "profile": "q_faithful_dilated_88k",
        "eval_roundtrip": True,
        "source": "tac.profiles.get_profile",
    }


def test_pose_tensor_contract_rejects_all_zero_raw_pose_file(tmp_path: Path) -> None:
    loop = _load_script()
    poses = tmp_path / "optimized_poses.bin"
    poses.write_bytes(b"\x00\x00" * 600 * 6)

    contract = loop.inspect_pose_tensor_contract(poses)

    assert contract["failure_class"] == "pose_tensor_all_zero"
    assert contract["all_zero"] is True
    assert contract["promotable_pose_contract"] is False
    with pytest.raises(loop.SnapshotError) as exc:
        loop.enforce_pose_tensor_contract(contract, allow_unproven=False)
    assert exc.value.failure_class == "pose_tensor_all_zero"


def test_pose_tensor_contract_accepts_nonzero_torch_pose_file(tmp_path: Path) -> None:
    loop = _load_script()
    poses = tmp_path / "optimized_poses.pt"
    pose_tensor = torch.zeros(600, 6)
    pose_tensor[:, 0] = 1.0
    torch.save({"optimized_poses": pose_tensor}, poses)

    contract = loop.inspect_pose_tensor_contract(poses)

    assert contract["source_key"] == "optimized_poses"
    assert contract["shape"] == [600, 6]
    assert contract["nonzero_elements"] == 600
    assert contract["promotable_pose_contract"] is True


def test_eval_run_fails_before_dispatch_for_all_zero_pose_tensor(tmp_path: Path) -> None:
    loop = _load_script()
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "renderer_epoch_001.pt"
    checkpoint.write_bytes(b"not loaded because pose contract fails first")
    masks = tmp_path / "masks.mkv"
    poses = tmp_path / "optimized_poses.bin"
    masks.write_bytes(b"mask")
    poses.write_bytes(b"\x00\x00" * 600 * 6)

    with pytest.raises(loop.SnapshotError) as exc:
        loop.main(
            [
                "--workspace",
                str(REPO),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--min-checkpoint-age-seconds",
                "0",
                "--masks-mkv",
                str(masks),
                "--mask-frame-contract",
                "full",
                "--poses-pt",
                str(poses),
                "--output-root",
                str(tmp_path / "snapshots"),
                "--eval-mode",
                "run",
                "--dispatch-claim-mode",
                "already-claimed",
                "--existing-dispatch-claim-id",
                "claimed-by-test",
            ]
        )
    assert exc.value.failure_class == "pose_tensor_all_zero"


def test_training_pose_contract_accepts_exact_deployed_pose_sha(tmp_path: Path) -> None:
    loop = _load_script()
    poses = tmp_path / "optimized_poses.pt"
    pose_tensor = torch.zeros(600, 6)
    pose_tensor[:, 0] = 1.0
    torch.save({"optimized_poses": pose_tensor}, poses)
    pose_contract = loop.inspect_pose_tensor_contract(poses)
    checkpoint = tmp_path / "renderer.pt"
    torch.save(
        {
            "__meta__": {
                "qfaithful_training_pose_contract": {
                    "pose_dim": 6,
                    "pose_sha256": pose_contract["sha256"],
                    "training_uses_nonzero_pose_stream": True,
                    "zero_pose_fallback_allowed": False,
                }
            }
        },
        checkpoint,
    )

    contract = loop.inspect_checkpoint_training_pose_contract(
        checkpoint,
        deployed_pose_contract=pose_contract,
        profile="q_faithful_dilated_88k",
    )

    assert contract["training_pose_contract_promotable"] is True
    assert contract["failure_class"] is None
    loop.enforce_checkpoint_training_pose_contract(contract)


def test_eval_run_fails_before_export_without_training_pose_contract(
    tmp_path: Path,
) -> None:
    loop = _load_script()
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "renderer_epoch_001.pt"
    torch.save({"model_state_dict": {"w": torch.ones(1)}}, checkpoint)
    masks = tmp_path / "masks.mkv"
    poses = tmp_path / "optimized_poses.pt"
    masks.write_bytes(b"synthetic; operator declares full-frame")
    pose_tensor = torch.zeros(600, 6)
    pose_tensor[:, 0] = 1.0
    torch.save({"optimized_poses": pose_tensor}, poses)

    with pytest.raises(loop.SnapshotError) as exc:
        loop.main(
            [
                "--workspace",
                str(REPO),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--min-checkpoint-age-seconds",
                "0",
                "--masks-mkv",
                str(masks),
                "--mask-frame-contract",
                "full",
                "--poses-pt",
                str(poses),
                "--output-root",
                str(tmp_path / "snapshots"),
                "--eval-mode",
                "run",
                "--dispatch-claim-mode",
                "already-claimed",
                "--existing-dispatch-claim-id",
                "claimed-by-test",
            ]
        )
    assert exc.value.failure_class == "qfaithful_training_pose_contract_missing"


def test_auto_state_dict_prefers_ema_shadow_and_rejects_live_export() -> None:
    loop = _load_script()
    state, label = loop.select_state_dict(
        {
            "model_state_dict": {"w": object()},
            "ema_shadow": {"w": object()},
        },
        "auto",
    )

    assert label == "ema_shadow"
    assert state == {"w": state["w"]}
    with pytest.raises(loop.SnapshotError) as exc:
        loop.enforce_ema_export_contract(
            {"checkpoint_state_source": "model_state_dict", "packed_from_ema_shadow": False},
            allow_live_weight_export=False,
        )
    assert exc.value.failure_class == "ema_shadow_export_missing"
