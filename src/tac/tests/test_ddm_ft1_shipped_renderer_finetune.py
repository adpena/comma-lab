"""Behaviour tests for the DDM-FT1 identity gate and candidate verdict.

Every test here is written so that replacing the function under test with a
constant-returning stub FAILS it.  That is deliberate: the NO-FAKE
"tests-verify-constants-not-behavior" class is exactly what a receipt-shaped
module invites, so the assertions are on transformed values, not on schema keys.
"""

from __future__ import annotations

import importlib
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

gate = importlib.import_module("experiments.ddm_ft1_identity_gate_and_caches")
verdict = importlib.import_module("experiments.ddm_ft1_verdict_bhw_pose")


# --------------------------------------------------------------------------
# B/H/W law -- the decomposition every prior arm on this object reported.
# --------------------------------------------------------------------------


def test_classify_pool_separates_benefit_harm_and_wash() -> None:
    incumbent = np.array([0, 1, 2, 3, 4, 0], dtype=np.uint8)
    candidate = np.array([1, 1, 3, 0, 2, 2], dtype=np.uint8)
    gt = np.array([1, 1, 4, 3, 1, 0], dtype=np.uint8)
    pool = verdict.classify_pool(incumbent, candidate, gt)
    # index 0: 0->1, gt 1  => benefit
    # index 1: unchanged   => outside the population entirely
    # index 2: 2->3, gt 4  => wash (both wrong)
    # index 3: 3->0, gt 3  => harm
    # index 4: 4->2, gt 1  => wash
    # index 5: 0->2, gt 0  => harm
    assert pool["B_benefit"] == 1
    assert pool["H_harm"] == 2
    assert pool["W_wash"] == 2
    assert pool["changed"] == 5


def test_wash_is_not_the_untouched_count() -> None:
    """W must count only CHANGED positions, never agreement."""

    incumbent = np.array([3, 3, 3, 3], dtype=np.uint8)
    candidate = np.array([3, 3, 3, 3], dtype=np.uint8)
    gt = np.zeros(4, dtype=np.uint8)  # every position wrong in both fields
    pool = verdict.classify_pool(incumbent, candidate, gt)
    assert pool["changed"] == 0
    assert pool["W_wash"] == 0, "unchanged-but-wrong is outside B/H/W"


def test_bhw_reproduces_the_delta_dseg_identity() -> None:
    """B - H must equal the reduction in wrong positions, exactly."""

    rng = np.random.default_rng(20260903)
    gt = rng.integers(0, 5, size=4096).astype(np.uint8)
    incumbent = gt.copy()
    candidate = gt.copy()
    incumbent[rng.choice(4096, 300, replace=False)] = 4
    candidate[rng.choice(4096, 300, replace=False)] = 2
    pool = verdict.classify_pool(incumbent, candidate, gt)
    wrong_before = int((incumbent != gt).sum())
    wrong_after = int((candidate != gt).sum())
    assert pool["B_benefit"] - pool["H_harm"] == wrong_before - wrong_after


def test_per_class_pool_partitions_the_field_by_gt_class() -> None:
    gt = np.array([0, 0, 1, 1, 4], dtype=np.uint8)
    incumbent = np.array([1, 0, 0, 1, 0], dtype=np.uint8)
    candidate = np.array([0, 1, 1, 1, 4], dtype=np.uint8)
    per_class = verdict.per_class_pool(incumbent, candidate, gt)
    assert per_class["Road"]["B_benefit"] == 1  # index 0: 1->0 with gt 0
    assert per_class["Road"]["H_harm"] == 1  # index 1: 0->1 with gt 0
    assert per_class["Lane"]["B_benefit"] == 1  # index 2: 0->1 with gt 1
    assert per_class["MyCar"]["B_benefit"] == 1  # index 4: 0->4 with gt 4
    assert sum(v["gt_pixels"] for v in per_class.values()) == gt.size


# --------------------------------------------------------------------------
# Score arithmetic -- must reproduce the frontier receipt from components.
# --------------------------------------------------------------------------


def test_score_components_reproduce_the_frontier_receipt() -> None:
    """AFR1: d_seg 0.00020139, d_pose 6.37e-6, 180,002 B -> S 0.147976..."""

    parts = verdict.score_components(0.00020139, 6.37e-06, 180_002)
    assert parts["S"] == pytest.approx(0.14797617125559104, rel=1e-6)
    assert parts["seg_term"] == pytest.approx(0.020139, rel=1e-9)
    assert parts["rate_term"] == pytest.approx(25 * 180_002 / 37_545_489, rel=1e-12)


def test_score_components_pose_term_is_the_square_root_law() -> None:
    """A 4x rise in d_pose must double the pose term, not quadruple it."""

    low = verdict.score_components(0.0, 1e-06, 0)["pose_term"]
    high = verdict.score_components(0.0, 4e-06, 0)["pose_term"]
    assert high == pytest.approx(2.0 * low, rel=1e-12)


def test_score_components_rate_scales_with_bytes() -> None:
    single = verdict.score_components(0.0, 0.0, 1)["rate_term"]
    assert verdict.score_components(0.0, 0.0, 1000)["rate_term"] == pytest.approx(
        1000 * single, rel=1e-12
    )


# --------------------------------------------------------------------------
# SM3R header recovery -- the export path's only source of truth.
# --------------------------------------------------------------------------


def _sm3r_blob(*, version: int, mode: int, keep: int, reserved: int, names: int) -> bytes:
    payload = bytearray(b"SM3R")
    payload.extend(bytes([version, mode, keep, reserved]))
    payload.extend(struct.pack("<H", 0))
    payload.extend(bytes((names + 1) // 2))
    return bytes(payload)


def test_recover_sm3r_allocation_refuses_a_foreign_magic() -> None:
    with pytest.raises(gate.IdentityGateError, match="not an SM3R payload"):
        gate.recover_sm3r_allocation(b"WANS" + bytes(16), {})


def test_recover_sm3r_allocation_refuses_the_wrong_mode() -> None:
    template = {"a.weight": torch.zeros(2, 2)}
    blob = _sm3r_blob(version=1, mode=5, keep=1, reserved=0, names=1)
    with pytest.raises(gate.IdentityGateError, match="unsupported SM3R header"):
        gate.recover_sm3r_allocation(blob, template)


def test_recover_sm3r_allocation_refuses_a_nonzero_reserved_byte() -> None:
    template = {"a.weight": torch.zeros(2, 2)}
    blob = _sm3r_blob(version=1, mode=6, keep=1, reserved=7, names=1)
    with pytest.raises(gate.IdentityGateError, match="unsupported SM3R header"):
        gate.recover_sm3r_allocation(blob, template)


@pytest.mark.skipif(
    not gate.FRONTIER_ARCHIVE.exists(), reason="frontier archive not mounted"
)
def test_recover_sm3r_allocation_reads_keep_percent_and_depths() -> None:
    """The depth nibbles must round-trip through the deployed packer's writer."""

    sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")
    shipped = gate.load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(gate.SEMANTIC_WIDTH).state_dict()
    names = sd1.quantized_names(template)
    depths = [3 if index % 2 else 4 for index in range(len(names))]
    payload = bytearray(b"SM3R")
    payload.extend(bytes([1, 6, 7, 0]))
    sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")
    payload.extend(struct.pack("<H", sm3.mask_for_names(names, sm3.PRUNE_NAMES)))
    payload.extend(sd1._pack_depth_nibbles(depths))
    keep, allocation = gate.recover_sm3r_allocation(bytes(payload), template)
    assert keep == 7
    assert allocation == dict(zip(names, depths, strict=True))


@pytest.mark.skipif(
    not gate.FRONTIER_ARCHIVE.exists(), reason="frontier archive not mounted"
)
def test_recover_sm3r_allocation_refuses_a_foreign_prune_mask() -> None:
    """A section that prunes a different tensor set must not be re-encoded."""

    sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")
    shipped = gate.load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(gate.SEMANTIC_WIDTH).state_dict()
    names = sd1.quantized_names(template)
    payload = bytearray(b"SM3R")
    payload.extend(bytes([1, 6, 1, 0]))
    payload.extend(struct.pack("<H", 1))  # not the deployed mask
    payload.extend(sd1._pack_depth_nibbles([4] * len(names)))
    with pytest.raises(gate.IdentityGateError, match="prune mask"):
        gate.recover_sm3r_allocation(bytes(payload), template)


def test_recover_sm3r_allocation_refuses_a_template_without_pruned_tensors() -> None:
    sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")
    template = {"token_embed.weight": torch.zeros(5, 4)}
    names = sd1.quantized_names(template)
    payload = bytearray(b"SM3R")
    payload.extend(bytes([1, 6, 1, 0]))
    payload.extend(struct.pack("<H", 0))
    payload.extend(sd1._pack_depth_nibbles([4] * len(names)))
    with pytest.raises(gate.IdentityGateError, match="pruned tensors"):
        gate.recover_sm3r_allocation(bytes(payload), template)


# --------------------------------------------------------------------------
# GT lineage -- the fork that decides what the fine-tune is aimed at.
# --------------------------------------------------------------------------


def test_gt_lineage_fork_counts_only_real_disagreements() -> None:
    dali = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    pyav = np.array([[0, 4], [2, 4]], dtype=np.uint8)
    fork = gate.gt_lineage_fork(dali, pyav)
    assert fork["argmax_disagreements"] == 2
    assert fork["positions"] == 4
    assert fork["rate"] == pytest.approx(0.5)


def test_gt_lineage_fork_is_zero_for_identical_tables() -> None:
    table = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    assert gate.gt_lineage_fork(table, table.copy())["argmax_disagreements"] == 0


def test_load_gt_tables_reads_the_dali_container(tmp_path: Path) -> None:
    seg = torch.randint(0, 5, (3, 2, 2), dtype=torch.uint8)
    pose = torch.randn(3, 6)
    path = tmp_path / "gt_cache_dali.pt"
    torch.save({"seg": seg, "pose": pose}, path)
    labels, poses = verdict.load_gt_tables(path)
    assert np.array_equal(labels, seg.numpy())
    assert poses.dtype == np.float64
    assert np.allclose(poses, pose.numpy().astype(np.float64))


def test_load_gt_tables_reads_the_pyav_npz_container(tmp_path: Path) -> None:
    lstars = np.random.default_rng(7).integers(0, 5, (3, 2, 2)).astype(np.int64)
    gt_poses = np.random.default_rng(8).normal(size=(3, 6))
    path = tmp_path / "gt.npz"
    np.savez(path, lstars=lstars, gt_poses=gt_poses)
    labels, poses = verdict.load_gt_tables(path)
    assert np.array_equal(labels, lstars.astype(np.uint8))
    assert np.allclose(poses, gt_poses)


def test_load_gt_seg_agrees_across_both_containers(tmp_path: Path) -> None:
    seg = np.random.default_rng(11).integers(0, 5, (2, 3, 4)).astype(np.uint8)
    npz_path = tmp_path / "a.npz"
    pt_path = tmp_path / "b.pt"
    np.savez(npz_path, lstars=seg.astype(np.int64))
    torch.save({"seg": torch.from_numpy(seg)}, pt_path)
    assert np.array_equal(gate.load_gt_seg(npz_path), gate.load_gt_seg(pt_path))


# --------------------------------------------------------------------------
# Cache writer -- refuses a field the trainer would silently misread.
# --------------------------------------------------------------------------


def test_write_token_cache_refuses_a_wrong_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="token field must be"):
        gate.write_token_cache(np.zeros((4, 4, 4), dtype=np.uint8), tmp_path / "x.pt")


def test_write_token_cache_refuses_out_of_range_classes(tmp_path: Path) -> None:
    field = np.zeros((gate.N_PAIRS, gate.EVAL_H, gate.EVAL_W), dtype=np.uint8)
    field[0, 0, 0] = 9
    with pytest.raises(ValueError, match="class indices"):
        gate.write_token_cache(field, tmp_path / "x.pt")


def test_write_token_cache_round_trips_through_the_trainer_read(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260903)
    field = rng.integers(0, 5, (gate.N_PAIRS, 4, 4)).astype(np.uint8)
    # Shape guard is on the full field; exercise the writer via a monkeypatched
    # geometry so the test stays fast but still writes and reads real bytes.
    original = (gate.EVAL_H, gate.EVAL_W)
    gate.EVAL_H, gate.EVAL_W = 4, 4
    try:
        path = tmp_path / "cache.pt"
        record = gate.write_token_cache(field, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)["seg"].long()
    finally:
        gate.EVAL_H, gate.EVAL_W = original
    assert torch.equal(loaded, torch.from_numpy(field.astype(np.int64)))
    assert record["bytes"] == path.stat().st_size
    assert len(record["sha256"]) == 64


# --------------------------------------------------------------------------
# Frame composition -- the two frames must come from different operators.
# --------------------------------------------------------------------------


def test_master_frames_upsamples_to_camera_and_quantizes() -> None:
    class _Renderer(torch.nn.Module):
        def forward(self, tokens, idx):
            return torch.full((tokens.shape[0], 3, 8, 8), 12.4)

    out = verdict.master_frames(
        _Renderer(), torch.zeros(2, 8, 8, dtype=torch.long), torch.tensor([0, 1])
    )
    assert out.shape == (2, 3, verdict.CAMERA_H, verdict.CAMERA_W)
    assert torch.equal(out, torch.round(out)), "camera frame must be integral"
    assert out.max() <= 255.0 and out.min() >= 0.0


def test_master_frames_clamps_an_out_of_range_render() -> None:
    class _Hot(torch.nn.Module):
        def forward(self, tokens, idx):
            return torch.full((tokens.shape[0], 3, 4, 4), 900.0)

    out = verdict.master_frames(
        _Hot(), torch.zeros(1, 4, 4, dtype=torch.long), torch.tensor([0])
    )
    assert out.max() == 255.0


def test_exact_r_returns_eval_resolution() -> None:
    qat = importlib.import_module(
        "tac.pr130_lift.train_semantic_quantized_resumable"
    )._load_lifted_qat()
    frame = torch.rand(1, 3, gate.EVAL_H, gate.EVAL_W) * 255.0
    out = gate.exact_r(frame, qat)
    assert out.shape == (1, 3, gate.EVAL_H, gate.EVAL_W)
    assert not torch.equal(out, frame), "R must not be the identity on a random frame"


# --------------------------------------------------------------------------
# Live-artifact regression: the shipped section must still round-trip.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not gate.FRONTIER_ARCHIVE.exists(), reason="frontier archive not mounted"
)
def test_shipped_semantic_section_round_trips_byte_identically() -> None:
    sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")
    blob = gate.read_semantic_section(gate.FRONTIER_ARCHIVE)
    shipped = gate.load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(gate.SEMANTIC_WIDTH).state_dict()
    state = shipped.unpack_variant_semantic_or_none(blob, template)
    assert state is not None
    keep, allocation = gate.recover_sm3r_allocation(blob, template)
    encoded, _expected, _meta = sm3.pack_prune_mixed_candidate(
        state, keep_percent=keep, depths=allocation
    )
    assert encoded == bytes(blob)
    assert len(encoded) == 36_130


@pytest.mark.skipif(
    not gate.FRONTIER_ARCHIVE.exists(), reason="frontier archive not mounted"
)
def test_sm3r_export_size_is_independent_of_the_weight_values() -> None:
    """The whole same-bytes premise of the arm rests on this."""

    sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")
    blob = gate.read_semantic_section(gate.FRONTIER_ARCHIVE)
    shipped = gate.load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(gate.SEMANTIC_WIDTH).state_dict()
    state = shipped.unpack_variant_semantic_or_none(blob, template)
    keep, allocation = gate.recover_sm3r_allocation(blob, template)
    generator = torch.Generator().manual_seed(20260903)
    perturbed = {
        name: value + 0.01 * torch.randn(value.shape, generator=generator)
        for name, value in state.items()
    }
    encoded, _expected, _meta = sm3.pack_prune_mixed_candidate(
        perturbed, keep_percent=keep, depths=allocation
    )
    assert len(encoded) == len(blob)
    assert encoded != bytes(blob), "perturbed weights must change the payload content"


def test_carrier_frames_are_camera_sized_and_integral() -> None:
    """Frame 2p must come out of the carrier basis, quantized at camera size."""

    class _Renderer:
        CARRIER_DIM = 4
        CARRIER_AMPLITUDE = 64.0

    basis = torch.zeros(4, 3, 6, 8)
    basis[0, :, :, :] = 1.0
    coefficients = torch.tensor([[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    out = verdict.carrier_frames(
        basis, coefficients, _Renderer(), torch.tensor([0, 1])
    )
    assert out.shape == (2, 3, verdict.CAMERA_H, verdict.CAMERA_W)
    assert torch.equal(out, torch.round(out))
    # pair 1 has zero coefficients, so it must be the flat 127.5 -> 128 plane
    assert out[1].min() == out[1].max() == 128.0
    # pair 0 carries signal, so it must differ from the flat plane
    assert out[0].mean() != out[1].mean()


# --------------------------------------------------------------------------
# Realization: the verdict must score the bytes that ship, not the trained
# weights.  export_section returns both the record and the parsed-back state.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not gate.FRONTIER_ARCHIVE.exists(), reason="frontier archive not mounted"
)
def test_export_section_returns_the_receiver_parsed_state() -> None:
    shipped = gate.load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(gate.SEMANTIC_WIDTH).state_dict()
    blob = gate.read_semantic_section(gate.FRONTIER_ARCHIVE)
    state = shipped.unpack_variant_semantic_or_none(blob, template)
    record, realized = verdict.export_section(state, gate.FRONTIER_ARCHIVE)
    assert record["size_preserved"] is True
    assert record["parse_back_max_abs_delta"] == 0.0
    # the shipped weights are already SM3R-quantized, so export is a fixed point
    assert record["trained_vs_realized_max_abs_delta"] == 0.0
    assert set(realized) == set(state)
    for name in state:
        assert torch.equal(realized[name], state[name])


@pytest.mark.skipif(
    not gate.FRONTIER_ARCHIVE.exists(), reason="frontier archive not mounted"
)
def test_export_section_realizes_a_perturbed_state_differently() -> None:
    """A trained state must NOT be assumed equal to what the receiver loads."""

    shipped = gate.load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(gate.SEMANTIC_WIDTH).state_dict()
    blob = gate.read_semantic_section(gate.FRONTIER_ARCHIVE)
    state = shipped.unpack_variant_semantic_or_none(blob, template)
    generator = torch.Generator().manual_seed(20260903)
    perturbed = {
        name: value + 0.05 * torch.randn(value.shape, generator=generator)
        for name, value in state.items()
    }
    record, realized = verdict.export_section(perturbed, gate.FRONTIER_ARCHIVE)
    assert record["size_preserved"] is True
    assert record["parse_back_max_abs_delta"] == 0.0
    assert record["trained_vs_realized_max_abs_delta"] > 0.0, (
        "the deployed encoder quantizes and prunes, so realized != trained"
    )
    pruned = "blocks.1.film.weight"
    rows = realized[pruned].reshape(realized[pruned].shape[0], -1)
    nonzero_rows = int((rows.abs().sum(dim=1) > 0).sum())
    assert nonzero_rows == record["kept_rows"][pruned], (
        "the export must keep exactly keep_percent of the pruned rows"
    )
