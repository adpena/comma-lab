# SPDX-License-Identifier: MIT
"""Tests for `experiments/line_search_pose_refinement.py` (Wave-1.5).

Verifies the R(D)-joint coordinate-descent objective + the deterministic
re-encode + the metadata-driven slice path. Uses a tiny 6-frame random
renderer archive at (16, 24) output canvas so smoke tests run on CPU in
<60s without any GPU.

Per CLAUDE.md MPS-PoseNet-23x rule: device=cuda paths are gated by
``torch.cuda.is_available()`` and skip when unavailable. The MockPoseNet
keeps the rest of the suite deterministic on CPU.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import pytest
import torch

import experiments.line_search_pose_refinement as line_search_pose_refinement

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.build_qpose_archive import build_qpose_archive
from experiments.line_search_pose_refinement import (
    ARCHIVE_MEMBER_NAME,
    ORIGINAL_SIZE,
    _MockPoseNet,
    coordinate_descent,
    col0_from_pose_payload,
    col0_to_pose_array,
    compute_joint_objective,
    encode_col0_to_pose_br,
    gradient_guided_delta_matrix,
    load_archive_blob,
    load_metadata,
    load_renderer,
    main,
    parse_basis_delta_sets,
    parse_delta_sets,
    parse_magnitude_sets,
    patch_posenet_for_differentiable_search,
    pose_atom_selection_summary,
    assert_metadata_matches_archive,
    assert_dali_runtime_dependency_available,
    assert_scorer_runtime_dependencies_available,
    slice_blob,
    temporal_basis_matrix,
    write_refined_archive,
)
from tac.qp1_pose_codec import (
    VELOCITY_OFFSET,
    VELOCITY_SCALE,
    decode_qp1,
    encode_qp1,
)
from tac.preflight import (
    check_line_search_scorer_runtime_preflight,
    check_posenet_gradient_preprocess_patch,
)
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer


# Tiny target size for CPU-tractable forward passes (real CUDA path uses 874x1164).
SMOKE_TARGET_H = 16
SMOKE_TARGET_W = 24


# -----------------------------------------------------------------------------
# Fixtures: build a tiny smoke archive that the line-search tool can refine
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_archive(tmp_path_factory) -> dict:
    """Build a tiny smoke archive with random renderer + non-trivial col0.

    Module-scoped fixture: each renderer build + QZS3 encode costs ~1-2s on
    CPU; share it across tests to keep the suite under 60s.
    """
    n = 6
    tmp_path = tmp_path_factory.mktemp("smoke_archive")
    torch.manual_seed(0)
    template = build_quantizr_faithful_renderer().eval()
    state = template.state_dict()

    # Construct a non-constant pose array to give the line search work to do.
    rng = np.random.default_rng(13)
    poses = np.zeros((n, 6), dtype=np.float32)
    poses[:, 0] = 30.0 + rng.normal(0, 0.5, size=n).astype(np.float32)

    archive_path = tmp_path / "archive.zip"
    meta = build_qpose_archive(
        renderer_state_dict=state,
        pose_array=poses,
        mask_obu_br_bytes=b"\x00" * 1024,  # tiny mask placeholder for fast IO
        output_archive=archive_path,
    )
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    col0 = col0_from_pose_payload(
        load_archive_blob(archive_path)[meta["mask_br_bytes"] + meta["model_br_bytes"]:]
    )
    return {
        "archive_path": archive_path,
        "metadata_path": meta_path,
        "col0": col0,
        "n": n,
        "meta": meta,
    }


def _build_smoke_generator(model_br: bytes) -> torch.nn.Module:
    """Decode QZS3 weights + shrink output canvas for CPU-tractable smoke."""
    qzs3_payload = brotli.decompress(model_br)
    generator = load_renderer(qzs3_payload, torch.device("cpu"))
    # Shrink renderer's output canvas — only mutates coord-grid + final
    # upsample target, not weights.
    generator.out_h, generator.out_w = SMOKE_TARGET_H, SMOKE_TARGET_W
    return generator


def _smoke_cd_kwargs(mask_br_n: int, model_br_n: int) -> dict:
    """Common coordinate_descent kwargs for smoke tests (small batch+chunk)."""
    return dict(
        device=torch.device("cpu"),
        batch_size=4,
        candidate_chunk=4,
        max_candidate_items=64,
        mask_br_bytes=mask_br_n,
        model_br_bytes=model_br_n,
        archive_overhead=100,
        target_h=SMOKE_TARGET_H,
        target_w=SMOKE_TARGET_W,
    )


def _smoke_cli_args(
    smoke_archive: dict,
    out_zip: Path,
    out_meta: Path,
    *,
    radii: str = "1",
    passes: int = 1,
) -> list[str]:
    """Common main() CLI args for smoke tests (cpu + mock-posenet + tiny canvas)."""
    return [
        "--archive-path",
        str(smoke_archive["archive_path"]),
        "--metadata-path",
        str(smoke_archive["metadata_path"]),
        "--output-path",
        str(out_zip),
        "--output-metadata",
        str(out_meta),
        "--device",
        "cpu",
        "--mock-posenet",
        "--no-gt",
        "--radii",
        radii,
        "--passes",
        str(passes),
        "--batch-size",
        "4",
        "--candidate-chunk",
        "4",
        "--mask-shape",
        f"{smoke_archive['n']},{SMOKE_TARGET_H},{SMOKE_TARGET_W}",
        "--target-h",
        str(SMOKE_TARGET_H),
        "--target-w",
        str(SMOKE_TARGET_W),
    ]


# -----------------------------------------------------------------------------
# 1. QP1 round-trip: encode -> decode -> col0 matches
# -----------------------------------------------------------------------------


def test_qp1_col0_round_trip_exact() -> None:
    """encode_qp1 -> decode -> recover col0 byte-identical."""
    rng = np.random.default_rng(7)
    col0 = rng.integers(5_120, 25_600, size=200, dtype=np.int64)
    poses = col0_to_pose_array(col0)
    payload = encode_qp1(poses)
    decoded = decode_qp1(payload)
    recovered = np.rint(
        (decoded[:, 0].astype(np.float64) - VELOCITY_OFFSET) * VELOCITY_SCALE
    ).astype(np.int64)
    assert np.array_equal(recovered, col0), (
        f"col0 round-trip drift: first diff at "
        f"{int(np.argmax(recovered != col0))}"
    )


def test_encode_col0_to_pose_br_brotli_roundtrip(smoke_archive) -> None:
    """The brotli-wrapped pose stream re-decodes to the same col0."""
    col0 = smoke_archive["col0"]
    pose_br = encode_col0_to_pose_br(col0)
    raw = brotli.decompress(pose_br)
    decoded = decode_qp1(raw)
    recovered = np.rint(
        (decoded[:, 0].astype(np.float64) - VELOCITY_OFFSET) * VELOCITY_SCALE
    ).astype(np.int64)
    assert np.array_equal(recovered, col0)


# -----------------------------------------------------------------------------
# 2. Joint-objective monotone descent: best_obj never exceeds baseline
# -----------------------------------------------------------------------------


def test_coordinate_descent_monotone_descent(smoke_archive) -> None:
    """coordinate_descent's best_obj must never exceed baseline objective."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)

    generator = _build_smoke_generator(model_br)
    posenet = _MockPoseNet().eval()
    col0 = smoke_archive["col0"]
    n = col0.shape[0]
    masks = torch.zeros((n, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long)
    target = torch.zeros((n, 6), dtype=torch.float32)

    accepted_objs: list[float] = []

    def cb(s: str) -> None:
        idx = s.find("obj=")
        if idx >= 0:
            tok = s[idx + 4 :].split()[0]
            accepted_objs.append(float(tok))

    refined_col0, stats = coordinate_descent(
        col0_init=col0,
        masks=masks,
        target=target,
        generator=generator,
        posenet=posenet,
        radii=[1, 2],
        passes=1,
        progress_cb=cb,
        **_smoke_cd_kwargs(len(mask_br), len(model_br)),
    )
    # Best objective never exceeds baseline.
    assert stats["best_obj"] <= accepted_objs[0] + 1e-12, (
        f"best_obj={stats['best_obj']} > baseline={accepted_objs[0]}"
    )
    # The refined col0 reproduces the same archive size when re-encoded.
    final_pose_br = encode_col0_to_pose_br(refined_col0)
    expected_size = len(mask_br) + len(model_br) + len(final_pose_br) + 100
    assert stats["best_archive_size"] == expected_size


# -----------------------------------------------------------------------------
# 3. Determinism: same inputs+seed -> same refined bytes
# -----------------------------------------------------------------------------


def test_refinement_deterministic(smoke_archive) -> None:
    """Two identical runs produce byte-identical refined col0."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)

    def run_once(seed: int) -> bytes:
        torch.manual_seed(seed)
        generator = _build_smoke_generator(model_br)
        torch.manual_seed(seed)
        posenet = _MockPoseNet().eval()
        col0 = smoke_archive["col0"].copy()
        n = col0.shape[0]
        masks = torch.zeros((n, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long)
        target = torch.zeros((n, 6), dtype=torch.float32)
        refined, _stats = coordinate_descent(
            col0_init=col0,
            masks=masks,
            target=target,
            generator=generator,
            posenet=posenet,
            radii=[1],
            passes=1,
            progress_cb=None,
            **_smoke_cd_kwargs(len(mask_br), len(model_br)),
        )
        return refined.tobytes()

    a = run_once(42)
    b = run_once(42)
    assert a == b, "two seeded runs produced different refined col0"


# -----------------------------------------------------------------------------
# 4. End-to-end smoke: tiny archive completes via main() CLI dispatch
# -----------------------------------------------------------------------------


def test_main_smoke_completes(smoke_archive, tmp_path: Path) -> None:
    """main() returns 0 on a smoke run with mock posenet + no GT + radius=1."""
    out = tmp_path / "refined.zip"
    out_meta = tmp_path / "refined.json"
    rc = main(_smoke_cli_args(smoke_archive, out, out_meta))
    assert rc == 0
    assert out.exists()
    assert out_meta.exists()
    assert out.with_suffix(".accepted_latest.zip").exists()
    assert out.with_suffix(".accepted_latest.json").exists()
    out_meta_dict = json.loads(out_meta.read_text())
    assert "refinement" in out_meta_dict
    assert out_meta_dict["refinement"]["device"] == "cpu"
    assert out_meta_dict["archive_path"] == str(out)
    assert out_meta_dict["archive_bytes"] == out.stat().st_size
    assert out_meta_dict["archive_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert out_meta_dict["source_archive_path"] == str(smoke_archive["archive_path"])
    assert (
        out_meta_dict["source_archive_sha256"]
        == hashlib.sha256(smoke_archive["archive_path"].read_bytes()).hexdigest()
    )
    checkpoint_meta = json.loads(out.with_suffix(".accepted_latest.json").read_text())
    checkpoint_zip = out.with_suffix(".accepted_latest.zip")
    assert checkpoint_meta["archive_path"] == str(checkpoint_zip)
    assert checkpoint_meta["archive_sha256"] == hashlib.sha256(
        checkpoint_zip.read_bytes()
    ).hexdigest()


def test_metadata_archive_custody_guard_rejects_stale_sha(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"current archive bytes")
    metadata = {
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": "0" * 64,
    }

    with pytest.raises(RuntimeError, match="metadata/archive custody mismatch"):
        assert_metadata_matches_archive(metadata, archive)

    metadata["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert_metadata_matches_archive(metadata, archive)


def test_scorer_runtime_dependency_preflight_reports_missing(monkeypatch) -> None:
    """Remote line-search dispatches must fail before opaque upstream imports."""
    original_find_spec = line_search_pose_refinement.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "timm":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(
        line_search_pose_refinement.importlib.util,
        "find_spec",
        fake_find_spec,
    )
    with pytest.raises(RuntimeError, match="missing: timm"):
        assert_scorer_runtime_dependencies_available(("timm",))


def test_scorer_runtime_dependency_preflight_requires_dali(monkeypatch) -> None:
    """GT-backed line search needs DALI before paid scorer/profile work starts."""
    original_find_spec = line_search_pose_refinement.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "nvidia.dali":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(
        line_search_pose_refinement.importlib.util,
        "find_spec",
        fake_find_spec,
    )
    with pytest.raises(RuntimeError, match="missing: nvidia.dali"):
        assert_scorer_runtime_dependencies_available()


def test_scorer_runtime_dependency_preflight_accepts_available_module() -> None:
    """The preflight is reusable for narrow smoke checks."""
    assert_scorer_runtime_dependencies_available(("json",))


def test_dali_runtime_dependency_preflight_reports_missing(monkeypatch) -> None:
    """GT-video line-search paths must fail before opaque DALI imports."""
    original_find_spec = line_search_pose_refinement.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "nvidia.dali":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(
        line_search_pose_refinement.importlib.util,
        "find_spec",
        fake_find_spec,
    )
    with pytest.raises(RuntimeError, match="nvidia.dali"):
        assert_dali_runtime_dependency_available()


# -----------------------------------------------------------------------------
# 5. Baseline preservation: refined archive's mask + model bytes IDENTICAL
# -----------------------------------------------------------------------------


def test_refined_archive_preserves_mask_and_model_bytes(smoke_archive, tmp_path: Path) -> None:
    """Only the pose stream changes; mask + model bytes stay identical."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)

    out = tmp_path / "refined.zip"
    out_meta = tmp_path / "refined.json"
    rc = main(_smoke_cli_args(smoke_archive, out, out_meta))
    assert rc == 0

    refined_blob = load_archive_blob(out)
    out_meta_dict = json.loads(out_meta.read_text())
    refined_mask, refined_model, _refined_pose = slice_blob(refined_blob, out_meta_dict)
    assert refined_mask == mask_br, "mask bytes mutated by refinement"
    assert refined_model == model_br, "model bytes mutated by refinement"


# -----------------------------------------------------------------------------
# 6. Edge cases: zero-radius (no-op) + col0 at boundaries
# -----------------------------------------------------------------------------


def test_zero_radius_is_no_op(smoke_archive) -> None:
    """radii=[] should immediately return the input col0 untouched."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)
    generator = _build_smoke_generator(model_br)
    posenet = _MockPoseNet().eval()
    col0 = smoke_archive["col0"]
    n = col0.shape[0]
    refined, _stats = coordinate_descent(
        col0_init=col0,
        masks=torch.zeros((n, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long),
        target=torch.zeros((n, 6), dtype=torch.float32),
        generator=generator,
        posenet=posenet,
        radii=[],
        passes=2,
        progress_cb=None,
        **_smoke_cd_kwargs(len(mask_br), len(model_br)),
    )
    assert np.array_equal(refined, col0)


def test_col0_boundary_clamping(smoke_archive) -> None:
    """col0 values near 0 / 65535 must clamp without overflow."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)
    generator = _build_smoke_generator(model_br)
    posenet = _MockPoseNet().eval()
    col0_at_max = np.full((4,), 65535, dtype=np.int64)
    col0_at_min = np.full((4,), 0, dtype=np.int64)
    masks = torch.zeros((4, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long)
    target = torch.zeros((4, 6), dtype=torch.float32)
    for c in (col0_at_max, col0_at_min):
        refined, _stats = coordinate_descent(
            col0_init=c,
            masks=masks,
            target=target,
            generator=generator,
            posenet=posenet,
            radii=[3],
            passes=1,
            progress_cb=None,
            **_smoke_cd_kwargs(len(mask_br), len(model_br)),
        )
        assert refined.min() >= 0 and refined.max() <= 65535


def test_candidate_batch_cap_splits_large_radius(smoke_archive, monkeypatch) -> None:
    """Full-res runs must cap candidate batches before PyTorch index limits."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)
    generator = _build_smoke_generator(model_br)
    posenet = _MockPoseNet().eval()
    col0 = smoke_archive["col0"]
    n = col0.shape[0]
    progress: list[str] = []
    observed_forward_items: list[int] = []
    original_make_frames = line_search_pose_refinement.make_frames

    def record_make_frames(*args, **kwargs):
        observed_forward_items.append(int(args[1].shape[0]))
        return original_make_frames(*args, **kwargs)

    monkeypatch.setattr(line_search_pose_refinement, "make_frames", record_make_frames)
    refined, stats = coordinate_descent(
        col0_init=col0,
        masks=torch.zeros((n, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long),
        target=torch.zeros((n, 6), dtype=torch.float32),
        generator=generator,
        posenet=posenet,
        radii=[3],
        passes=1,
        progress_cb=progress.append,
        device=torch.device("cpu"),
        batch_size=4,
        candidate_chunk=99,
        max_candidate_items=5,
        mask_br_bytes=len(mask_br),
        model_br_bytes=len(model_br),
        archive_overhead=100,
        target_h=SMOKE_TARGET_H,
        target_w=SMOKE_TARGET_W,
    )
    assert refined.shape == col0.shape
    assert stats["best_archive_size"] > 0
    assert any("candidate_batch_cap" in msg for msg in progress)
    assert max(observed_forward_items) <= 5


def test_directional_delta_sets_allow_sparse_asymmetric_search(smoke_archive) -> None:
    """Sparse/asymmetric delta stages support directional pose-manifold probes."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)
    generator = _build_smoke_generator(model_br)
    posenet = _MockPoseNet().eval()
    col0 = smoke_archive["col0"]
    n = col0.shape[0]
    progress: list[str] = []
    accepted_payloads: list[dict] = []

    refined, stats = coordinate_descent(
        col0_init=col0,
        masks=torch.zeros((n, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long),
        target=torch.zeros((n, 6), dtype=torch.float32),
        generator=generator,
        posenet=posenet,
        radii=[99],  # ignored when delta_sets is provided
        passes=1,
        progress_cb=progress.append,
        accepted_cb=lambda _col0, payload: accepted_payloads.append(payload),
        delta_sets=[[-5, -3, -1, 0, 1], [-1, 0, 2, 5, 8]],
        **_smoke_cd_kwargs(len(mask_br), len(model_br)),
    )
    assert refined.shape == col0.shape
    assert stats["best_archive_size"] > 0
    assert any("delta_set=1" in msg for msg in progress)
    for payload in accepted_payloads:
        assert payload["search_stage_kind"] == "delta_set"
        assert payload["delta_count"] >= 2


def test_parse_delta_sets_includes_noop_and_rejects_empty_stage() -> None:
    """Directional CLI parser keeps no-op available and fails closed."""
    assert parse_delta_sets("-3,-1,1;2,5") == [[-3, -1, 0, 1], [0, 2, 5]]
    with pytest.raises(ValueError, match="empty"):
        parse_delta_sets("-1,0; ;1")


def test_gradient_guided_delta_matrix_orients_by_descent_direction() -> None:
    """Differentiable proposal deltas follow -sign(gradient) with backtracking."""
    matrix = gradient_guided_delta_matrix(
        np.array([2.0, -3.0, 0.0], dtype=np.float32),
        magnitudes=[1, 5],
        backtrack_magnitudes=[2],
    )
    assert matrix.tolist() == [
        [0, -1, -5, 2],
        [0, 1, 5, -2],
        [0, 1, 5, -2],
    ]


def test_gradient_delta_sets_run_differentiable_proposal_stage(smoke_archive) -> None:
    """Gradient-guided search stays archive-compatible and records stage metadata."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)
    generator = _build_smoke_generator(model_br)
    posenet = _MockPoseNet().eval()
    col0 = smoke_archive["col0"]
    n = col0.shape[0]
    progress: list[str] = []
    accepted_payloads: list[dict] = []

    refined, stats = coordinate_descent(
        col0_init=col0,
        masks=torch.zeros((n, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long),
        target=torch.zeros((n, 6), dtype=torch.float32),
        generator=generator,
        posenet=posenet,
        radii=[99],  # ignored when gradient_delta_sets is provided
        passes=1,
        progress_cb=progress.append,
        accepted_cb=lambda _col0, payload: accepted_payloads.append(payload),
        gradient_delta_sets=[[1, 3]],
        gradient_backtrack_magnitudes=[1],
        **_smoke_cd_kwargs(len(mask_br), len(model_br)),
    )
    assert refined.shape == col0.shape
    assert stats["best_archive_size"] > 0
    assert any("gradient_delta_set=1" in msg for msg in progress)
    for payload in accepted_payloads:
        assert payload["search_stage_kind"] == "gradient_delta_set"
        assert payload["gradient_candidate_count"] == 4


def test_parse_magnitude_sets_normalizes_positive_magnitudes() -> None:
    """Gradient stage parser treats signs as magnitudes and rejects empty stages."""
    assert parse_magnitude_sets("-3,1,3;2") == [[1, 3], [2]]
    with pytest.raises(ValueError, match="empty"):
        parse_magnitude_sets("1; ;2")


def test_parse_basis_delta_sets_normalizes_vector_stages() -> None:
    """Temporal/vector proposal parser keeps kind and positive magnitudes."""
    assert parse_basis_delta_sets("dct:-3,1,3;pair_window:2") == [
        {"kind": "dct", "magnitudes": [1, 3], "basis_delta_set_index": 1},
        {"kind": "pair_window", "magnitudes": [2], "basis_delta_set_index": 2},
    ]
    with pytest.raises(ValueError, match="kind:magnitudes"):
        parse_basis_delta_sets("dct")


def test_temporal_basis_matrix_builds_dct_and_pair_windows() -> None:
    """Basis atoms cover smooth DCT directions and pair-index windows."""
    dct = temporal_basis_matrix(
        8, kind="dct", modes=[0, 1, 2], pair_indices=[], window_radius=0
    )
    assert dct.shape == (3, 8)
    assert np.allclose(np.max(np.abs(dct), axis=1), 1.0)

    windows = temporal_basis_matrix(
        10, kind="pair_window", modes=[], pair_indices=[2], window_radius=1
    )
    assert windows.shape == (1, 10)
    assert windows[0, 4] == pytest.approx(1.0)
    assert windows[0, 5] == pytest.approx(1.0)
    assert windows[0, 3] > 0
    assert windows[0, 6] > 0


def test_basis_delta_sets_run_vector_proposal_stage(smoke_archive) -> None:
    """Vector basis search stays archive-compatible and records metadata."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)
    generator = _build_smoke_generator(model_br)
    posenet = _MockPoseNet().eval()
    col0 = smoke_archive["col0"]
    n = col0.shape[0]
    progress: list[str] = []
    accepted_payloads: list[dict] = []

    refined, stats = coordinate_descent(
        col0_init=col0,
        masks=torch.zeros((n, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long),
        target=torch.zeros((n, 6), dtype=torch.float32),
        generator=generator,
        posenet=posenet,
        radii=[99],  # ignored when basis_delta_sets is provided
        passes=1,
        progress_cb=progress.append,
        accepted_cb=lambda _col0, payload: accepted_payloads.append(payload),
        basis_delta_sets=[{"kind": "dct", "magnitudes": [1, 3], "basis_delta_set_index": 1}],
        basis_modes=[0, 1],
        **_smoke_cd_kwargs(len(mask_br), len(model_br)),
    )
    assert refined.shape == col0.shape
    assert stats["best_archive_size"] > 0
    assert any("basis_delta_set=1 kind=dct" in msg for msg in progress)
    assert any("basis_candidate_start" in msg for msg in progress)
    for payload in accepted_payloads:
        assert payload["search_stage_kind"] == "basis_delta_set"
        assert payload["basis_kind"] == "dct"


def test_patch_posenet_for_differentiable_search_repairs_no_grad_preprocess() -> None:
    """Gradient proposal must not inherit upstream PoseNet's no-grad barrier."""

    class NoGradPoseNet(torch.nn.Module):
        @torch.no_grad()
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x * 1.0

    posenet = NoGradPoseNet()
    x = torch.randn(1, 2, 3, 16, 16, requires_grad=True)
    assert not posenet.preprocess_input(x).requires_grad

    patch_posenet_for_differentiable_search(posenet)
    patched = posenet.preprocess_input(x)
    assert patched.requires_grad
    patched.mean().backward()
    assert x.grad is not None


def test_preflight_catches_unpatched_posenet_gradient_proposal(tmp_path: Path) -> None:
    """The gradient proposal bug class is pinned in static preflight."""
    script = tmp_path / "experiments" / "line_search_pose_refinement.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
def estimate_col0_gradient(posenet, x):
    # gradient-guided proposal path
    pred = posenet(posenet.preprocess_input(x))
    loss = pred.pow(2).mean()
    loss.backward()
""".lstrip()
    )
    violations = check_posenet_gradient_preprocess_patch(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 2
    assert "does not patch" in violations[0]
    assert "loss.requires_grad" in violations[1]


def test_preflight_allows_current_posenet_gradient_proposal() -> None:
    """The live line-search tool has the differentiable patch and guard."""
    violations = check_posenet_gradient_preprocess_patch(
        repo_root=_REPO_ROOT, strict=False, verbose=False
    )
    assert violations == []


def test_preflight_catches_line_search_missing_dali_runtime_guard(tmp_path: Path) -> None:
    """GT-backed line search must pin DALI in its runtime dependency guard."""
    script = tmp_path / "experiments" / "line_search_pose_refinement.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
SCORER_RUNTIME_MODULES = ("timm",)

def assert_scorer_runtime_dependencies_available():
    pass

def load_posenet():
    from upstream.modules import PoseNet

def target():
    return DaliVideoDataset()
""".lstrip()
    )
    violations = check_line_search_scorer_runtime_preflight(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any('"nvidia.dali"' in item for item in violations)
    assert any("bootstrap_dali_hash_pinned.py" in item for item in violations)
    assert any("before importing" in item for item in violations)


def test_preflight_allows_current_line_search_scorer_runtime_guard() -> None:
    """The live line-search tool fail-closes on scorer deps and DALI."""
    violations = check_line_search_scorer_runtime_preflight(
        repo_root=_REPO_ROOT, strict=False, verbose=False
    )
    assert violations == []


# -----------------------------------------------------------------------------
# 7. Joint objective formula (mirrors pr67_line_search.py:140 byte-for-byte)
# -----------------------------------------------------------------------------


def test_compute_joint_objective_matches_pr67_formula() -> None:
    """obj = sqrt(10 * pose_mse) + 25 * archive_size / ORIGINAL_SIZE."""
    pose_mse = 0.0107
    mask_br = 219_472
    model_br = 56_093
    pose_br = 899
    overhead = 100
    expected = (
        (10.0 * pose_mse) ** 0.5
        + 25.0 * (mask_br + model_br + pose_br + overhead) / ORIGINAL_SIZE
    )
    got = compute_joint_objective(
        pose_mse,
        mask_br_bytes=mask_br,
        model_br_bytes=model_br,
        pose_br_bytes=pose_br,
        archive_overhead=overhead,
    )
    assert abs(got - expected) < 1e-12


# -----------------------------------------------------------------------------
# 8. CUDA-only test (gated): verifies the cuda dispatch path imports cleanly
# -----------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda unavailable")
def test_cuda_dispatch_smoke(smoke_archive, tmp_path: Path) -> None:
    """CUDA dispatch with mock-posenet completes — verifies device routing.

    Per CLAUDE.md MPS-PoseNet-23x rule, real strategic decisions on this
    tool MUST come from a CUDA run. This test only verifies the dispatch
    path imports + runs cleanly when cuda is present.
    """
    out = tmp_path / "refined_cuda.zip"
    out_meta = tmp_path / "refined_cuda.json"
    args = _smoke_cli_args(smoke_archive, out, out_meta)
    # Replace --device cpu with --device cuda:0
    di = args.index("--device")
    args[di + 1] = "cuda:0"
    rc = main(args)
    assert rc == 0
    assert out.exists()


# -----------------------------------------------------------------------------
# 9. Metadata-driven slicing rejects mismatched payload
# -----------------------------------------------------------------------------


def test_slice_blob_rejects_length_mismatch(smoke_archive) -> None:
    """slice_blob must hard-error if blob length disagrees with metadata."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    bad_meta = dict(meta)
    bad_meta["pose_br_bytes"] = bad_meta["pose_br_bytes"] + 1
    with pytest.raises(ValueError, match="blob length"):
        slice_blob(blob, bad_meta)


# -----------------------------------------------------------------------------
# 10. Write-then-read symmetry on the refined archive (deterministic stored zip)
# -----------------------------------------------------------------------------


def test_write_refined_archive_is_deterministic_stored_zip(tmp_path: Path) -> None:
    """write_refined_archive emits a stored ZIP with a single 'p' member."""
    mask_br = b"\x01" * 100
    model_br = b"\x02" * 200
    pose_br = b"\x03" * 50
    out = tmp_path / "out.zip"
    n_bytes = write_refined_archive(out, mask_br, model_br, pose_br)
    assert n_bytes > 350
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == [ARCHIVE_MEMBER_NAME]
        info = zf.getinfo(ARCHIVE_MEMBER_NAME)
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.file_size == 350
        # Deterministic timestamp (1980 epoch, ZIP date_time field)
        assert info.date_time == (1980, 1, 1, 0, 0, 0)


# -----------------------------------------------------------------------------
# 11. Single-pass single-radius is well-defined (edge case)
# -----------------------------------------------------------------------------


def test_single_pass_single_radius(smoke_archive) -> None:
    """passes=1, radii=[1] is a valid minimal-effort refinement run."""
    blob = load_archive_blob(smoke_archive["archive_path"])
    meta = load_metadata(smoke_archive["metadata_path"])
    mask_br, model_br, _pose_br = slice_blob(blob, meta)
    generator = _build_smoke_generator(model_br)
    posenet = _MockPoseNet().eval()
    col0 = smoke_archive["col0"]
    n = col0.shape[0]
    refined, stats = coordinate_descent(
        col0_init=col0,
        masks=torch.zeros((n, SMOKE_TARGET_H, SMOKE_TARGET_W), dtype=torch.long),
        target=torch.zeros((n, 6), dtype=torch.float32),
        generator=generator,
        posenet=posenet,
        radii=[1],
        passes=1,
        progress_cb=None,
        **_smoke_cd_kwargs(len(mask_br), len(model_br)),
    )
    assert refined.shape == col0.shape
    assert stats["best_archive_size"] > 0


def test_pose_atom_selection_summary_records_charged_qp1_atoms() -> None:
    source = np.array([10, 20, 30, 40], dtype=np.int64)
    refined = np.array([10, 23, 30, 35], dtype=np.int64)
    summary = pose_atom_selection_summary(
        source_col0=source,
        refined_col0=refined,
        pose_br_bytes=901,
        archive_bytes=276_427,
        policy="test_policy",
    )
    assert summary["pose_codec"] == "pose_qp1_v1"
    assert summary["charged_accounting"]["pose_br_bytes"] == 901
    assert summary["charged_accounting"]["sidecar_bytes"] == 0
    assert summary["selection_hooks"]["supports_bandit_or_rl_proposal_ordering"] is True
    assert summary["atom_count"] == 2
    assert summary["atoms"] == [
        {
            "frame_index": 1,
            "pair_index": 0,
            "frame_in_pair": 1,
            "delta_q": 3,
            "source_q": 20,
            "refined_q": 23,
        },
        {
            "frame_index": 3,
            "pair_index": 1,
            "frame_in_pair": 1,
            "delta_q": -5,
            "source_q": 40,
            "refined_q": 35,
        },
    ]
