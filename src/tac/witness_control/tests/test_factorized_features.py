# SPDX-License-Identifier: MIT
"""Tests for the shared exact-factorization SENSE core (organ upgrades A/B/C).

NO-FAKE: every test verifies BEHAVIOR against the REAL operator — the resize taps are
checked numerically against live ``torch.nn.functional.interpolate`` (the exact op the
frozen scorer applies), the ker(A) mask is checked by PERTURBING blind pixels and
asserting the true operator output is bitwise unchanged, and the blind fraction is
checked against the canonical MEASURED constant.  No test would pass on a stub."""
from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS
from tac.witness_control.factorized_features import (
    CAMERA_HW,
    MARGIN_HIST_EDGES,
    SCORER_HW,
    MarginSnapshot,
    default_pair_sample,
    ker_a_zero_weight_mask,
    load_frozen_segnet_cpu,
    load_witness_ema,
    oriented_key,
    pair_key,
    pair_norm_for_oriented,
    parse_oriented_key,
    resize_taps,
    touched_1d,
    verify_ker_mask_against_canonical,
    visible_energy_split,
)

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


def _apply_taps_2d(x: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    i0h, i1h, w0h, w1h = resize_taps(out_hw[0], x.shape[0])
    i0w, i1w, w0w, w1w = resize_taps(out_hw[1], x.shape[1])
    rows = x[i0h] * w0h[:, None] + x[i1h] * w1h[:, None]
    return rows[:, i0w] * w0w[None, :] + rows[:, i1w] * w1w[None, :]


def test_resize_taps_match_torch_at_real_scorer_sizes():
    rng = np.random.default_rng(1)
    x = rng.random(CAMERA_HW).astype(np.float64)
    ref = F.interpolate(torch.from_numpy(x)[None, None], size=SCORER_HW, mode="bilinear").numpy()[0, 0]
    ours = _apply_taps_2d(x, SCORER_HW)
    assert float(np.abs(ours - ref).max()) < 1e-10


@pytest.mark.parametrize(("n_out", "n_in"), [(37, 100), (7, 8), (5, 23), (384, 874), (512, 1164)])
def test_resize_taps_match_torch_1d_general_sizes(n_out, n_in):
    rng = np.random.default_rng(2)
    x = rng.random((n_in, 3)).astype(np.float64)
    ref = F.interpolate(torch.from_numpy(x)[None, None], size=(n_out, 3), mode="bilinear").numpy()[0, 0]
    i0, i1, w0, w1 = resize_taps(n_out, n_in)
    ours = x[i0] * w0[:, None] + x[i1] * w1[:, None]
    # width axis of the ref is identity here (3 -> 3 keeps values on the same grid)
    assert float(np.abs(ours - ref).max()) < 1e-10


def test_resize_taps_rejects_nonpositive_sizes():
    with pytest.raises(ValueError):
        resize_taps(0, 10)
    with pytest.raises(ValueError):
        resize_taps(10, 0)


def test_ker_mask_fraction_matches_canonical_measured_constant():
    out = verify_ker_mask_against_canonical()
    assert out["abs_diff"] < 2e-3
    assert abs(out["closed_form_zero_weight_frac"] - 0.226969) < 1e-4


def test_blind_pixel_perturbation_has_exactly_zero_operator_effect():
    """The zero-marginal THEOREM, tested on the live operator: adding arbitrary values on
    ker(A) support leaves the true torch resize output bitwise unchanged."""
    rng = np.random.default_rng(3)
    x = rng.random(CAMERA_HW)
    k = ker_a_zero_weight_mask()
    x2 = x.copy()
    x2[k] += 1e6 * rng.random(int(k.sum()))
    r1 = F.interpolate(torch.from_numpy(x)[None, None], size=SCORER_HW, mode="bilinear")
    r2 = F.interpolate(torch.from_numpy(x2)[None, None], size=SCORER_HW, mode="bilinear")
    assert torch.equal(r1, r2)


def test_touched_pixel_perturbation_changes_operator_output():
    rng = np.random.default_rng(4)
    x = rng.random(CAMERA_HW)
    k = ker_a_zero_weight_mask()
    ys, xs = np.nonzero(~k)
    x2 = x.copy()
    x2[ys[0], xs[0]] += 5.0
    r1 = F.interpolate(torch.from_numpy(x)[None, None], size=SCORER_HW, mode="bilinear")
    r2 = F.interpolate(torch.from_numpy(x2)[None, None], size=SCORER_HW, mode="bilinear")
    assert not torch.equal(r1, r2)


def test_touched_1d_covers_all_outputs_taps():
    t = touched_1d(SCORER_HW[0], CAMERA_HW[0])
    i0, i1, w0, w1 = resize_taps(SCORER_HW[0], CAMERA_HW[0])
    assert t[i0[w0 > 0]].all() and t[i1[w1 > 0]].all()
    assert 0.0 < t.mean() < 1.0  # some rows genuinely untouched


def test_visible_energy_split_pure_blind_and_pure_visible():
    k = ker_a_zero_weight_mask()
    m = np.zeros(CAMERA_HW)
    m[k] = 3.0
    s = visible_energy_split(m, k)
    assert s["visible_frac"] == 0.0 and s["energy_blind"] > 0
    m2 = np.zeros(CAMERA_HW)
    m2[~k] = 2.0
    s2 = visible_energy_split(m2, k)
    assert s2["visible_frac"] == 1.0 and s2["energy_blind"] == 0.0
    s3 = visible_energy_split(np.zeros(CAMERA_HW), k)
    assert s3["visible_frac"] is None


def test_visible_energy_split_rejects_wrong_shape():
    with pytest.raises(ValueError):
        visible_energy_split(np.zeros((10, 10)))


def test_pair_keys_and_norms_roundtrip():
    assert pair_key(1, 0) == "Road-Lane" == pair_key(0, 1)
    assert oriented_key(0, 1) == "Road->Lane"
    assert parse_oriented_key("Lane->Road") == (1, 0)
    assert pair_norm_for_oriented("Road->Lane") == HEAD_PAIR_NORMS["Road-Lane"]
    assert pair_norm_for_oriented("Lane->Road") == HEAD_PAIR_NORMS["Road-Lane"]  # orientation-free


def _synthetic_snapshot(margins_by: dict[str, list[float]], total_px: int = 2 * 384 * 512) -> MarginSnapshot:
    wrongs, gts, ms = [], [], []
    for key, vals in margins_by.items():
        w, g = parse_oriented_key(key)
        for v in vals:
            wrongs.append(w)
            gts.append(g)
            ms.append(v)
    n = len(ms)
    return MarginSnapshot(
        run_ref="synthetic", ema_epoch=1, generated_at="t", pair_indices=(0, 1),
        scorer_hw=SCORER_HW, total_px=total_px,
        d_seg_sample=n / total_px,
        flip_pair_idx=np.zeros(n, np.int32), flip_y=np.zeros(n, np.int32),
        flip_x=np.arange(n, dtype=np.int32),
        flip_wrong=np.asarray(wrongs, np.int8), flip_gt=np.asarray(gts, np.int8),
        flip_margin=np.asarray(ms, np.float64),
    )


def test_margins_by_oriented_pair_groups_and_sorts():
    snap = _synthetic_snapshot({"Road->Lane": [0.3, 0.1, 0.2], "Lane->Road": [0.5]})
    mb = snap.margins_by_oriented_pair()
    assert set(mb) == {"Road->Lane", "Lane->Road"}
    assert np.allclose(mb["Road->Lane"], [0.1, 0.2, 0.3])  # sorted


def test_flipdist_feature_space_uses_canonical_pair_norm():
    snap = _synthetic_snapshot({"Road->Lane": [0.3953]})
    fd = snap.flipdist_feature_space_by_oriented_pair()["Road->Lane"]
    assert np.allclose(fd, 0.3953 / HEAD_PAIR_NORMS["Road-Lane"])


def test_summary_row_histogram_accounts_for_every_flip():
    vals = [*np.geomspace(2e-3, 10.0, 40), 1e-4, 50.0]  # includes under+overflow
    snap = _synthetic_snapshot({"Road->Lane": vals})
    row = snap.summary_row()
    d = row["by_oriented_pair"]["Road->Lane"]
    assert d["n"] == len(vals)
    assert sum(d["margin_hist"]) + d["margin_underflow"] + d["margin_overflow"] == len(vals)
    assert row["margin_hist_edges"] == [float(e) for e in MARGIN_HIST_EDGES]
    assert row["score_claim"] is False and row["schema"] == "witness_factorized_snapshot.v1"


def test_default_pair_sample_stride():
    s = default_pair_sample(600, 24)
    assert s == list(range(0, 600, 25)) and len(s) == 24
    assert default_pair_sample(10, 100) == list(range(10))


def test_load_witness_ema_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_witness_ema(tmp_path / "nope.npz")


def test_load_frozen_segnet_fails_closed_on_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frozen_segnet_cpu(upstream_dir=tmp_path)


def test_load_frozen_segnet_fails_closed_on_hash_mismatch(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "segnet.safetensors").write_bytes(b"not the frozen weights")
    with pytest.raises(AssertionError, match="sha256"):
        load_frozen_segnet_cpu(upstream_dir=tmp_path)
