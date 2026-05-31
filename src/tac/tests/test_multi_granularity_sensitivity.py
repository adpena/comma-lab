# SPDX-License-Identifier: MIT
"""Tests for tac.multi_granularity_sensitivity.

NO FAKE behavioral assertions: every test verifies the ACTUAL contest-score math
(argmax-flip-concentrated boundary band, out//2 pose window, 100·d_seg / sqrt(10·d_pose)
per-pair contribution, marginal-weighted byte-axis share) — not constants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tac import multi_granularity_sensitivity as mgs

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Provenance non-promotable contract (Catalog #341 / #192 / #127 / #323).
# ---------------------------------------------------------------------------
def test_non_promotable_provenance_is_never_promotable() -> None:
    prov = mgs.non_promotable_provenance_dict(
        model_id="x", inputs_sha256="a" * 64
    )
    assert prov["promotion_eligible"] is False
    assert prov["score_claim_valid"] is False
    assert prov["measurement_axis"] == "[predicted]"


# ---------------------------------------------------------------------------
# Boundary-band weight map (the optimal seg-distillation spatial weight).
# ---------------------------------------------------------------------------
def _two_class_boundary_logits() -> torch.Tensor:
    logits = torch.zeros(1, 5, 8, 8)
    logits[0, 0, :, :4] = 10.0  # confident class 0 on the left
    logits[0, 1, :, 4:] = 10.0  # confident class 1 on the right
    logits[0, 0, :, 3] = 1.0  # blur the boundary column → small margin
    logits[0, 1, :, 3] = 0.9
    return logits


def test_boundary_band_weight_near_one_on_boundary_near_zero_interior() -> None:
    w, _ = mgs.segnet_boundary_band_weights(_two_class_boundary_logits(), tau=1.0)
    # confident interior (col 0): margin ~10 → w ~ exp(-10) ≈ 0
    assert float(w[0, 0]) < 1e-3
    # small-margin boundary (col 3): margin 0.1 → w ~ exp(-0.1) ≈ 0.905
    assert float(w[0, 3]) > 0.5


def test_boundary_band_fraction_is_small_for_sharp_boundary() -> None:
    _, st = mgs.segnet_boundary_band_weights(_two_class_boundary_logits(), tau=1.0)
    # only the single blurred column is in the band → ~1/8 of pixels
    assert 0.0 < st.boundary_band_fraction < 0.3
    # weight is tightly concentrated (high gini)
    assert st.weight_gini > 0.5
    assert st.n_pixels == 64


def test_boundary_band_accepts_chw_and_numpy() -> None:
    logits_chw = torch.zeros(5, 4, 4)
    logits_chw[0] = 5.0
    logits_chw[1, :, 2:] = 5.0  # boundary
    w1, _ = mgs.segnet_boundary_band_weights(logits_chw, tau=1.0)
    assert tuple(w1.shape) == (4, 4)
    w2, _ = mgs.segnet_boundary_band_weights(
        logits_chw.numpy(), tau=1.0
    )
    assert tuple(w2.shape) == (4, 4)
    assert np.allclose(w1.numpy(), w2.numpy(), atol=1e-5)


def test_boundary_band_hard_margin_threshold() -> None:
    _, st = mgs.segnet_boundary_band_weights(
        _two_class_boundary_logits(), tau=1.0, margin_threshold=0.5
    )
    # hard band: pixels with margin < 0.5 (only the 0.1-margin column)
    assert st.margin_threshold == 0.5
    assert 0.0 < st.boundary_band_fraction < 0.3


def test_boundary_band_smaller_tau_tightens_band() -> None:
    # a graded-margin field; smaller tau concentrates weight more (higher gini)
    logits = torch.zeros(1, 5, 1, 16)
    logits[0, 0, 0, :] = torch.linspace(0.0, 8.0, 16)  # class-0 logit ramps
    logits[0, 1, 0, :] = 4.0  # class-1 constant → margin sweeps through 0
    _, st_small = mgs.segnet_boundary_band_weights(logits, tau=0.3)
    _, st_large = mgs.segnet_boundary_band_weights(logits, tau=3.0)
    assert st_small.weight_gini >= st_large.weight_gini


def test_boundary_band_rejects_bad_inputs() -> None:
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.segnet_boundary_band_weights(torch.zeros(1, 1, 4, 4))  # <2 classes
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.segnet_boundary_band_weights(torch.zeros(5, 4), tau=1.0)  # bad ndim
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.segnet_boundary_band_weights(_two_class_boundary_logits(), tau=0.0)
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.segnet_boundary_band_weights([[1, 2], [3, 4]])  # not tensor/array


# ---------------------------------------------------------------------------
# Per-pose-dimension score contribution (the out//2 window + AIL weight source).
# ---------------------------------------------------------------------------
def test_pose_dims_outside_contest_window_contribute_zero() -> None:
    # dims 3-5 have huge error but are OUTSIDE the out//2=3 contest window
    student = np.array([34.5, 0.1, 0.02, 99.0, 99.0, 99.0])
    teacher = np.array([34.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dims = mgs.per_pose_dim_score_contribution(
        student, teacher, contest_window_dims=3
    )
    for d in dims:
        if d.dim_index >= 3:
            assert d.score_contribution == 0.0
            assert d.in_contest_window is False
            assert d.delta_sq > 1000.0  # error is large but score-irrelevant
        else:
            assert d.in_contest_window is True


def test_pose_dim_score_contribution_uses_hyperbolic_marginal() -> None:
    student = np.array([1.0, 0.0])
    teacher = np.array([0.0, 0.0])
    dims = mgs.per_pose_dim_score_contribution(
        student, teacher, contest_window_dims=1, d_pose_running=0.01
    )
    expected_marginal = 5.0 / math.sqrt(10.0 * 0.01)
    # dim0 delta_sq=1.0 → contribution = 1.0 * marginal
    assert dims[0].score_contribution == pytest.approx(expected_marginal, rel=1e-6)


def test_pose_dim_default_window_is_half() -> None:
    student = np.array([1.0, 1.0, 1.0, 1.0])
    teacher = np.zeros(4)
    dims = mgs.per_pose_dim_score_contribution(student, teacher)  # window defaults to 2
    in_window = [d for d in dims if d.in_contest_window]
    assert len(in_window) == 2


def test_pose_dim_shape_mismatch_raises() -> None:
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.per_pose_dim_score_contribution([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Per-pair axis contribution (the 600-pair native granularity).
# ---------------------------------------------------------------------------
def test_per_pair_seg_contribution_is_100_times_d_seg() -> None:
    pairs = mgs.per_pair_axis_score_contribution([0.01, 0.5], [0.001, 0.001])
    assert pairs[0].seg_score_contribution == pytest.approx(1.0)  # 100*0.01
    assert pairs[1].seg_score_contribution == pytest.approx(50.0)  # 100*0.5


def test_per_pair_pose_contribution_is_sqrt_10_d_pose() -> None:
    pairs = mgs.per_pair_axis_score_contribution([0.0], [0.4])
    assert pairs[0].pose_score_contribution == pytest.approx(math.sqrt(10.0 * 0.4))


def test_per_pair_dominant_axis_classification() -> None:
    # pair with high d_seg → seg-dominant; pair with high d_pose → pose-dominant
    pairs = mgs.per_pair_axis_score_contribution([0.5, 0.0001], [0.0001, 0.02])
    assert pairs[0].dominant_axis == "seg"
    assert pairs[1].dominant_axis == "pose"


def test_per_pair_mismatched_lengths_raise() -> None:
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.per_pair_axis_score_contribution([0.1, 0.2], [0.1])


# ---------------------------------------------------------------------------
# REAL-measured byte-axis sensitivity from the master-gradient ledger ($0).
# ---------------------------------------------------------------------------
def _write_synthetic_anchor(tmp_path: Path) -> tuple[str, Path]:
    """Write a synthetic (n_bytes,3) array + a single-row ledger pointing at it."""
    arr = np.zeros((100, 3), dtype=np.float32)
    arr[:10, 0] = 1.0  # 10 seg-dominant bytes
    arr[10:15, 1] = 0.5  # 5 pose-dominant bytes
    # rate column stays 0 (rate gradient is structurally 0 — byte count only)
    arr_path = tmp_path / "grad.npy"
    np.save(arr_path, arr)
    sha = "f" * 64
    ledger = tmp_path / "ledger.jsonl"
    row = {
        "schema_version": 1,
        "archive_sha256": sha,
        "measurement_axis": "[macOS-CPU advisory]",
        "measurement_hardware": "macos_arm64",
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "gradient_array_path": str(arr_path),
        "gradient_subject_sha256": "a" * 64,
        "n_bytes": 100,
        "operating_point": {
            "d_seg": 0.001,
            "d_pose": 0.0038,
            "rate": 0.0047,
            "score": 0.4175,
        },
        "written_at_utc": "2026-05-31T00:00:00Z",
    }
    ledger.write_text(json.dumps(row) + "\n")
    return sha, ledger


def test_byte_axis_sensitivity_real_from_synthetic_ledger(tmp_path: Path) -> None:
    sha, ledger = _write_synthetic_anchor(tmp_path)
    rep = mgs.byte_axis_sensitivity_from_master_gradient(sha, ledger_path=ledger)
    assert rep.n_bytes == 100
    by_axis = {a.axis: a for a in rep.per_axis}
    # seg has 10 nonzero bytes; pose has 5; rate has 0
    assert by_axis["seg"].dominant_byte_count == 10
    assert by_axis["pose"].dominant_byte_count == 5
    assert by_axis["rate"].dominant_byte_count == 0
    # seg share dominates (seg marginal=100 >> pose marginal, and more mass)
    assert by_axis["seg"].share > by_axis["pose"].share
    # provenance is non-promotable
    assert rep.provenance["promotion_eligible"] is False


def test_byte_axis_sensitivity_gini_distinguishes_concentration(tmp_path: Path) -> None:
    sha, ledger = _write_synthetic_anchor(tmp_path)
    rep = mgs.byte_axis_sensitivity_from_master_gradient(sha, ledger_path=ledger)
    by_axis = {a.axis: a for a in rep.per_axis}
    # 10/100 bytes carry all seg signal → high gini (concentrated)
    assert by_axis["seg"].gini > 0.5
    # rate is all-zero → gini 0
    assert by_axis["rate"].gini == 0.0
    # top decile (10 bytes) holds ~all of seg's mass
    assert by_axis["seg"].top_decile_mass > 0.9


def test_byte_axis_sensitivity_as_dict_roundtrips_to_json(tmp_path: Path) -> None:
    sha, ledger = _write_synthetic_anchor(tmp_path)
    rep = mgs.byte_axis_sensitivity_from_master_gradient(sha, ledger_path=ledger)
    blob = json.dumps(rep.as_dict())
    parsed = json.loads(blob)
    assert parsed["schema"] == "multi_granularity_byte_axis_sensitivity_v1"
    assert len(parsed["per_axis"]) == 3


def test_byte_axis_sensitivity_missing_anchor_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.byte_axis_sensitivity_from_master_gradient("0" * 64, ledger_path=empty)


def test_byte_axis_sensitivity_missing_array_fails_closed(tmp_path: Path) -> None:
    sha = "e" * 64
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "archive_sha256": sha,
                "measurement_axis": "[macOS-CPU advisory]",
                "gradient_array_path": str(tmp_path / "nope.npy"),
                "operating_point": {
                    "d_seg": 0.001,
                    "d_pose": 0.004,
                    "rate": 0.0,
                    "score": 0.4,
                },
            }
        )
        + "\n"
    )
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.byte_axis_sensitivity_from_master_gradient(sha, ledger_path=ledger)


def test_byte_axis_sensitivity_wrong_shape_fails_closed(tmp_path: Path) -> None:
    sha = "d" * 64
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((10, 2)))  # not (n,3)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "archive_sha256": sha,
                "measurement_axis": "[macOS-CPU advisory]",
                "gradient_array_path": str(bad),
                "operating_point": {
                    "d_seg": 0.001,
                    "d_pose": 0.004,
                    "rate": 0.0,
                    "score": 0.4,
                },
            }
        )
        + "\n"
    )
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.byte_axis_sensitivity_from_master_gradient(sha, ledger_path=ledger)


# ---------------------------------------------------------------------------
# REAL-LEDGER regression guard: the actual fec6 contest-CUDA anchor on disk.
# ---------------------------------------------------------------------------
def test_byte_axis_sensitivity_real_fec6_anchor_seg_dominant_if_present() -> None:
    sha = "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
    try:
        rep = mgs.byte_axis_sensitivity_from_master_gradient(sha, axis="[contest-CUDA]")
    except mgs.MultiGranularitySensitivityError:
        pytest.skip("fec6 contest-CUDA anchor not present in this checkout")
    by_axis = {a.axis: a for a in rep.per_axis}
    # the real frontier archive is seg-dominant (verified empirically ~0.89 share)
    assert by_axis["seg"].share > by_axis["pose"].share
    assert rep.n_bytes > 100_000
    assert rep.provenance["promotion_eligible"] is False


# ---------------------------------------------------------------------------
# Pending-measurement design (no fabricated numbers; Catalog #307).
# ---------------------------------------------------------------------------
def test_pending_measurement_boundary_seg_is_research_only_with_recipe() -> None:
    pm = mgs.design_input_domain_sensitivity_measurement("boundary", "seg")
    assert pm.research_only is True
    assert pm.requires_forward_pass is True
    assert len(pm.measurement_recipe) >= 4
    assert pm.provenance["promotion_eligible"] is False
    # NO fabricated numbers: the recipe is text + reactivation criteria only
    blob = json.dumps(pm.as_dict())
    assert "reactivation_criteria" in blob


def test_pending_measurement_frame_and_pair_granularities() -> None:
    for gran in ("frame", "pair"):
        for axis in ("seg", "pose", "rate"):
            pm = mgs.design_input_domain_sensitivity_measurement(gran, axis)
            assert pm.granularity == gran
            assert pm.score_axis == axis
            assert pm.requires_forward_pass is True


def test_pending_measurement_rejects_bad_granularity_or_axis() -> None:
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.design_input_domain_sensitivity_measurement("nope", "seg")
    with pytest.raises(mgs.MultiGranularitySensitivityError):
        mgs.design_input_domain_sensitivity_measurement("boundary", "nope")


# ---------------------------------------------------------------------------
# Module surface.
# ---------------------------------------------------------------------------
def test_canonical_axes_and_granularities() -> None:
    assert mgs.SCORE_AXES == ("seg", "pose", "rate")
    assert mgs.GRANULARITIES[0] == "byte"
    assert "boundary" in mgs.GRANULARITIES
    assert "pair" in mgs.GRANULARITIES
