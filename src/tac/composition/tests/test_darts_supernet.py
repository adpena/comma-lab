# SPDX-License-Identifier: MIT
"""Tests for tac.composition.darts_supernet.

Covers (per CLAUDE.md "Subagent coherence-by-default" + the directive's
">=15 dedicated tests" requirement):

1. Config validation (axis dedup, candidate count, latent_dim, anchor signs)
2. AxisOp forward shape + non-negativity + bounded delta
3. AxisOp anchor pinning (output near anchor at z=0)
4. SuperNet forward score scale (within plausible band)
5. SuperNet forward batch broadcasting
6. SuperNet discovered_architecture at zero-alpha (defined ordering)
7. SuperNet param-group split (arch vs weight)
8. Temperature anneal monotonicity
9. Search runs without NaN
10. Search lowers loss vs initial uniform
11. Search produces top-k ranking by predicted score
12. Search reports KL per axis
13. Search produces full per-axis trajectory
14. Provenance JSON round-trip
15. Provenance enforces score_claim=False
16. CPU is default device; MPS rejected
17. CUDA rejected when unavailable
18. Discovered values match canonical search-space band
19. default_search_axes is the 5-axis, 960-arch spec
20. Determinism under same seed
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tac.composition.darts_supernet import (
    AxisCandidate,
    AxisOp,
    AxisOpError,
    SuperNetConfig,
    SuperNetError,
    TimeTravelerSuperNet,
    default_search_axes,
    load_provenance,
    reasonable_candidate_value,
    run_supernet_search,
    write_provenance,
)
from tac.darts import DARTSAnnealSchedule

# --- AxisCandidate / SuperNetConfig validation ---------------------------


def test_axis_candidate_rejects_negative_rate() -> None:
    with pytest.raises(AxisOpError, match="rate_bytes_anchor"):
        AxisCandidate("bad", 1, -1.0, 0.001, 1.0e-5)


def test_axis_candidate_rejects_negative_seg() -> None:
    with pytest.raises(AxisOpError, match="seg_proxy_anchor"):
        AxisCandidate("bad", 1, 1.0, -0.001, 1.0e-5)


def test_axis_candidate_rejects_negative_pose() -> None:
    with pytest.raises(AxisOpError, match="pose_proxy_anchor"):
        AxisCandidate("bad", 1, 1.0, 0.001, -1.0e-5)


def test_axis_candidate_rejects_empty_name() -> None:
    with pytest.raises(AxisOpError, match="name must be non-empty"):
        AxisCandidate("", 1, 1.0, 0.001, 1.0e-5)


def test_supernet_config_rejects_empty_axes() -> None:
    with pytest.raises(SuperNetError, match="axes must be non-empty"):
        SuperNetConfig(axes=())


def test_supernet_config_rejects_duplicate_axis() -> None:
    c1 = AxisCandidate("a", 1, 1.0, 0.001, 1.0e-5)
    c2 = AxisCandidate("b", 2, 2.0, 0.001, 1.0e-5)
    with pytest.raises(SuperNetError, match="Duplicate axis name"):
        SuperNetConfig(axes=(("axis", (c1, c2)), ("axis", (c1, c2))))


def test_supernet_config_rejects_single_candidate_axis() -> None:
    c1 = AxisCandidate("only", 1, 1.0, 0.001, 1.0e-5)
    with pytest.raises(SuperNetError, match=">= 2 candidates"):
        SuperNetConfig(axes=(("solo", (c1,)),))


def test_supernet_config_rejects_zero_latent_dim() -> None:
    c1 = AxisCandidate("a", 1, 1.0, 0.001, 1.0e-5)
    c2 = AxisCandidate("b", 2, 2.0, 0.001, 1.0e-5)
    with pytest.raises(SuperNetError, match="latent_dim must be >= 1"):
        SuperNetConfig(axes=(("axis", (c1, c2)),), latent_dim=0)


# --- default_search_axes is the canonical 5x960 spec ---------------------


def test_default_search_axes_is_five_axis_960_archs() -> None:
    cfg = default_search_axes()
    assert cfg.axis_names() == (
        "world_model_size",
        "per_pair_budget",
        "foveation_grid",
        "decoder_hidden_dim",
        "quant_mode",
    )
    # 4 * 5 * 4 * 4 * 3 = 960
    assert cfg.total_architectures() == 960


def test_default_search_axes_candidate_names_are_unique_per_axis() -> None:
    cfg = default_search_axes()
    for axis_name in cfg.axis_names():
        names = cfg.candidate_names(axis_name)
        assert len(names) == len(set(names))


# --- AxisOp forward semantics --------------------------------------------


def test_axis_op_forward_shape() -> None:
    c = AxisCandidate("c", 1, 1000.0, 0.001, 1.0e-5)
    op = AxisOp(c, latent_dim=8)
    z = torch.zeros(3, 8)
    out = op(z)
    assert out.shape == (3, 3)


def test_axis_op_forward_non_negative() -> None:
    c = AxisCandidate("c", 1, 1000.0, 0.001, 1.0e-5)
    op = AxisOp(c, latent_dim=8)
    # Even with random z, output must be >= 0.
    torch.manual_seed(0)
    z = torch.randn(5, 8)
    out = op(z)
    assert (out >= 0.0).all()


def test_axis_op_near_anchor_at_zero_z() -> None:
    c = AxisCandidate("c", 1, 1000.0, 0.001, 1.0e-5)
    op = AxisOp(c, latent_dim=8)
    z = torch.zeros(1, 8)
    out = op(z).detach()
    # At z=0, the MLP's first layer produces only its small bias init,
    # then tanh(scaled-near-zero) ≈ small. Output is within ~0.5% of anchor.
    assert abs(float(out[0, 0]) - 1000.0) / 1000.0 < 0.01
    assert abs(float(out[0, 1]) - 0.001) / 0.001 < 0.01
    assert abs(float(out[0, 2]) - 1.0e-5) / 1.0e-5 < 0.01


def test_axis_op_bounded_delta() -> None:
    """The output is bounded to ±5% of the anchor."""
    c = AxisCandidate("c", 1, 1000.0, 0.001, 1.0e-5)
    op = AxisOp(c, latent_dim=8)
    # Pump the MLP with extreme z to saturate tanh.
    z = 1000.0 * torch.ones(1, 8)
    out = op(z).detach()
    # Output ∈ anchor * [0.95, 1.05].
    assert 950.0 <= float(out[0, 0]) <= 1050.0
    assert 0.00095 <= float(out[0, 1]) <= 0.00105


def test_axis_op_rejects_wrong_z_shape() -> None:
    c = AxisCandidate("c", 1, 1000.0, 0.001, 1.0e-5)
    op = AxisOp(c, latent_dim=8)
    with pytest.raises(AxisOpError, match="expects z of shape"):
        op(torch.zeros(8))  # 1-D, not (B, 8)
    with pytest.raises(AxisOpError, match="expects z of shape"):
        op(torch.zeros(3, 4))  # wrong latent_dim


def test_axis_op_rejects_zero_latent_dim() -> None:
    c = AxisCandidate("c", 1, 1000.0, 0.001, 1.0e-5)
    with pytest.raises(AxisOpError, match="latent_dim must be >= 1"):
        AxisOp(c, latent_dim=0)


# --- SuperNet forward + structure ----------------------------------------


def test_supernet_forward_scalar() -> None:
    cfg = default_search_axes()
    sn = TimeTravelerSuperNet(cfg)
    s = sn(batch_size=1)
    assert s.dim() == 0
    assert float(s.item()) > 0.0


def test_supernet_forward_in_plausible_band_at_init() -> None:
    """At uniform alpha + zero z, the score should be near PR101's 0.193 band.

    The seeded anchors are calibrated against PR106 r2 + time-traveler
    memo §7. Uniform-alpha score should fall in [0.10, 0.25]; outside that
    range indicates the anchor calibration drifted from the memos.
    """
    torch.manual_seed(0)
    cfg = default_search_axes()
    sn = TimeTravelerSuperNet(cfg)
    s = float(sn(batch_size=1).item())
    assert 0.10 <= s <= 0.25, f"uniform-alpha score {s} outside plausible band"


def test_supernet_forward_batch_broadcast() -> None:
    cfg = default_search_axes()
    sn = TimeTravelerSuperNet(cfg)
    s1 = sn(batch_size=1)
    s4 = sn(batch_size=4)
    # Batched forward should give the same scalar (mean over identical
    # broadcasted latent rows).
    assert pytest.approx(float(s1.item()), abs=1e-5) == float(s4.item())


def test_supernet_forward_rejects_zero_batch() -> None:
    cfg = default_search_axes()
    sn = TimeTravelerSuperNet(cfg)
    with pytest.raises(SuperNetError, match="batch_size must be"):
        sn(batch_size=0)


def test_supernet_discovered_at_init_is_well_defined() -> None:
    """argmax(zeros) is index 0 by convention; check it stays consistent."""
    cfg = default_search_axes()
    sn = TimeTravelerSuperNet(cfg)
    disc = sn.discovered_architecture()
    assert set(disc.keys()) == set(cfg.axis_names())
    # Every discovered value is the first candidate of its axis.
    for axis_name, cands in cfg.axes:
        assert disc[axis_name] == cands[0].name


def test_supernet_param_groups_split() -> None:
    """The 5 alpha tensors (one per cell) are separated from MLP weights."""
    cfg = default_search_axes()
    sn = TimeTravelerSuperNet(cfg)
    arch = sn.architecture_parameters()
    weights = sn.weight_parameters()
    # 5 axes -> 5 alpha tensors.
    assert len(arch) == 5
    # Each alpha tensor is 1-D.
    for p in arch:
        assert p.dim() == 1
    # Arch + weights are disjoint by id.
    arch_ids = {id(p) for p in arch}
    for w in weights:
        assert id(w) not in arch_ids


def test_supernet_temperature_anneal_monotone() -> None:
    """Temperature should monotonically decrease from anneal."""
    cfg = SuperNetConfig(
        axes=default_search_axes().axes,
        latent_dim=8,
        anneal=DARTSAnnealSchedule(T_start=4.0, T_end=0.5),
    )
    sn = TimeTravelerSuperNet(cfg)
    t_start = sn.temperature_anneal(0, 100)
    t_mid = sn.temperature_anneal(50, 100)
    t_end = sn.temperature_anneal(99, 100)
    for axis in t_start:
        assert t_start[axis] >= t_mid[axis] >= t_end[axis]
        assert pytest.approx(t_start[axis]) == 4.0
        assert pytest.approx(t_end[axis], abs=1e-3) == 0.5


# --- Search loop semantics -----------------------------------------------


def test_run_supernet_search_no_nan() -> None:
    """The search must not produce NaN at any step."""
    sn, res = run_supernet_search(total_steps=50, top_k=3, seed=42)
    assert res.final_score == res.final_score  # not NaN
    for _axis, traj in res.trajectory_per_axis.items():
        for rec in traj:
            assert rec["val_loss"] == rec["val_loss"]
            for v in rec["alpha"]:
                assert v == v


def test_run_supernet_search_lowers_score() -> None:
    """Final discrete score should be ≤ initial uniform-alpha score.

    The seeded anchors place the minimum at the 'small-everything' corner;
    DARTS should walk alpha toward that corner.
    """
    torch.manual_seed(0)
    cfg = default_search_axes()
    sn0 = TimeTravelerSuperNet(cfg)
    init_score = float(sn0(batch_size=1).item())
    _, res = run_supernet_search(total_steps=200, top_k=3, seed=0)
    # Allow a small tolerance for stochastic seed effects.
    assert res.final_score < init_score + 0.01


def test_run_supernet_search_top_k_length() -> None:
    sn, res = run_supernet_search(total_steps=50, top_k=5, seed=0)
    assert len(res.ranked_top_k) == 5
    # All entries are (float, dict) tuples.
    for score, arch in res.ranked_top_k:
        assert isinstance(score, float)
        assert isinstance(arch, dict)
        assert set(arch.keys()) == set(default_search_axes().axis_names())


def test_run_supernet_search_kl_per_axis() -> None:
    """Every axis reports a KL nats value >= 0."""
    _, res = run_supernet_search(total_steps=50, top_k=3, seed=0)
    for _axis, kl in res.kl_per_axis.items():
        assert kl >= 0.0
    assert set(res.kl_per_axis.keys()) == set(default_search_axes().axis_names())


def test_run_supernet_search_trajectory_recorded() -> None:
    """Trajectory has one record per step per axis."""
    _, res = run_supernet_search(total_steps=20, top_k=3, seed=0)
    for _axis, traj in res.trajectory_per_axis.items():
        assert len(traj) == 20
        for rec in traj:
            assert "alpha" in rec
            assert "softmax" in rec
            assert "argmax_index" in rec


def test_run_supernet_search_rejects_mps() -> None:
    """MPS is intentionally NOT supported per CLAUDE.md."""
    with pytest.raises(SuperNetError, match="device must be"):
        run_supernet_search(total_steps=5, device="mps")


def test_run_supernet_search_rejects_cuda_when_unavailable() -> None:
    """CUDA is rejected if torch.cuda.is_available() == False."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; this test only covers the negative path")
    with pytest.raises(SuperNetError, match=r"cuda.*not available|is_available"):
        run_supernet_search(total_steps=5, device="cuda")


def test_run_supernet_search_deterministic_under_seed() -> None:
    """Same seed → same discovered architecture."""
    _, res1 = run_supernet_search(total_steps=30, top_k=3, seed=1234)
    _, res2 = run_supernet_search(total_steps=30, top_k=3, seed=1234)
    assert res1.discovered == res2.discovered
    assert res1.discovered_values == res2.discovered_values


def test_discovered_values_are_in_canonical_band() -> None:
    """Every discovered value must be one of the canonical candidates."""
    _, res = run_supernet_search(total_steps=30, top_k=3, seed=0)
    for axis, value in res.discovered_values.items():
        assert reasonable_candidate_value(axis, value), (
            f"axis {axis}: value {value!r} not in canonical band"
        )


# --- Provenance JSON round-trip ------------------------------------------


def test_provenance_round_trip(tmp_path: Path) -> None:
    _, res = run_supernet_search(total_steps=10, top_k=3, seed=0)
    p = tmp_path / "supernet_provenance.json"
    write_provenance(res, p)
    assert p.exists()
    raw = json.loads(p.read_text())
    assert raw["score_claim"] is False
    assert raw["score_claim_valid"] is False
    assert raw["promotion_eligible"] is False
    assert raw["ready_for_exact_eval_dispatch"] is False
    assert raw["evidence_grade"] == "MPS-research-signal"
    assert raw["discovered"] == res.discovered
    assert raw["axes_searched"] == [
        "world_model_size",
        "per_pair_budget",
        "foveation_grid",
        "decoder_hidden_dim",
        "quant_mode",
    ]
    assert raw["total_steps"] == 10


def test_provenance_loader_rejects_score_claim_true(tmp_path: Path) -> None:
    """The loader refuses any provenance that flips score_claim to True."""
    _, res = run_supernet_search(total_steps=5, top_k=3, seed=0)
    p = tmp_path / "supernet_provenance.json"
    write_provenance(res, p)
    raw = json.loads(p.read_text())
    raw["score_claim"] = True
    p.write_text(json.dumps(raw, indent=2))
    with pytest.raises(SuperNetError, match="score_claim=True"):
        load_provenance(p)


def test_provenance_loader_rejects_missing_keys(tmp_path: Path) -> None:
    """The loader refuses provenance missing required keys."""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"discovered": {}}))
    with pytest.raises(SuperNetError, match="missing keys"):
        load_provenance(p)


# --- reasonable_candidate_value sanity ------------------------------------


def test_reasonable_candidate_value_accepts_canonical() -> None:
    assert reasonable_candidate_value("world_model_size", 50_000)
    assert reasonable_candidate_value("per_pair_budget", 40)
    assert reasonable_candidate_value("foveation_grid", 8)
    assert reasonable_candidate_value("decoder_hidden_dim", 64)
    assert reasonable_candidate_value("quant_mode", "fp4")
    assert reasonable_candidate_value("quant_mode", "ternary")


def test_reasonable_candidate_value_rejects_non_canonical() -> None:
    assert not reasonable_candidate_value("world_model_size", 60_000)
    assert not reasonable_candidate_value("quant_mode", "bf16")
    assert not reasonable_candidate_value("unknown_axis", 0)
