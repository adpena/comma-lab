# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the NUANCED survival-selective Lever-D.

The crude Lever-D coded ALL flips → NO-GO (mean σ=0.46 < σ*=0.77). The nuanced
module (``tac.torch_vehicle.lever_d_selective``) codes ONLY the survival-robust
sub-population and waterfills by leverage-per-byte. These tests prove the
load-bearing claims (each, if wrong, makes the selective coder a FAKE — either it
silently codes all flips regardless of survival, or it asserts a savings the
economics do not support):

  1. ECONOMICS — the σ* break-even is the exact ``b/WATERLINE`` (re-derived two ways).
  2. SELECTION IS REAL — selective-vs-all A/B: a select-all stub collapses σ_eff to
     the population mean; real selection lifts it. The leverage waterfill admits
     ONLY net-negative-S flips (a constant-σ-0 input admits nothing).
  3. ROUND-TRIP — the selective sidecar bit-exactly round-trips the admitted
     survivors; an empty selection serializes ZERO bytes (default-OFF / NO-GO).
  4. NO-OP / NO-FAKE GUARD — the "no-op fails the savings assertion" contract:
     coding all flips at the basin's measured σ is NOT GO; only survival-selection
     that genuinely lifts σ_eff past σ* is GO.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.margin_conditional_residual import WATERLINE_BYTES_PER_FLIP
from tac.torch_vehicle.lever_d_selective import (
    SCORE_PER_BYTE,
    SEG_VALUE_PER_FLIP,
    build_selection,
    decode_selective_sidecar,
    effective_sigma_of_subset,
    encode_selective_sidecar,
    net_delta_s_seg_sidecar,
    select_survivors,
    survival_break_even_sigma,
    waterfill_by_leverage,
)

_N_A = 37_545_489
_N_SCORED_TOTAL = 600 * 384 * 512


# ── 1. ECONOMICS: the σ* break-even ─────────────────────────────────────────
def test_survival_break_even_is_b_over_waterline():
    """σ* = b/WATERLINE — the canonical closed form."""
    for b in (0.5, 0.985, 1.0, 1.273108, 2.0):
        assert survival_break_even_sigma(b) == pytest.approx(b / WATERLINE_BYTES_PER_FLIP)


def test_survival_break_even_independent_derivation():
    """Re-derive σ* the LONG way (from the score formula) and confirm it matches
    the module's closed form — the two paths must agree or the economics are wrong."""
    b = 0.985
    # net_ΔS = −100·σ·N/N_scored + 25·b·N/N_a < 0  ⟹  σ > 25·b·N_scored/(100·N_a)
    sigma_star_long = (25.0 * b * _N_SCORED_TOTAL) / (100.0 * _N_A)
    assert survival_break_even_sigma(b) == pytest.approx(sigma_star_long, rel=1e-12)
    # numeric anchor: ≈ 0.7737
    assert survival_break_even_sigma(b) == pytest.approx(0.773697, abs=1e-5)


def test_crude_all_flips_at_basin_sigma_is_not_go():
    """The crude probe coded all flips at mean σ≈0.46 and b≈0.985 → net ΔS > 0.
    This is the structural NO-GO the nuance must overcome."""
    net = net_delta_s_seg_sidecar(n_flips_coded=530_000, sigma_effective=0.464, bytes_per_flip=0.985)
    assert net > 0.0, "all-flips at basin σ must RAISE S (the crude NO-GO)"


def test_net_delta_s_sign_flips_at_break_even():
    """At exactly σ*, net ΔS == 0; above it, net < 0 (GO); below, net > 0."""
    b = 0.985
    sigma_star = survival_break_even_sigma(b)
    n = 100_000
    assert net_delta_s_seg_sidecar(n, sigma_star, b) == pytest.approx(0.0, abs=1e-9)
    assert net_delta_s_seg_sidecar(n, sigma_star + 0.05, b) < 0.0
    assert net_delta_s_seg_sidecar(n, sigma_star - 0.05, b) > 0.0


def test_net_delta_s_flip_count_factors_out_of_sign():
    """The SIGN of net ΔS is independent of the flip count N (N factors out)."""
    b, sigma = 0.985, 0.9
    signs = {np.sign(net_delta_s_seg_sidecar(n, sigma, b)) for n in (10, 1000, 530_000)}
    assert len(signs) == 1, "the GO/NO-GO sign must not depend on flip count"


def test_per_flip_score_constants_consistent():
    """SEG_VALUE_PER_FLIP and SCORE_PER_BYTE are the exact contest constants."""
    assert pytest.approx(100.0 / _N_SCORED_TOTAL) == SEG_VALUE_PER_FLIP
    assert pytest.approx(25.0 / _N_A) == SCORE_PER_BYTE
    # WATERLINE is the ratio (bytes per flip at score-neutral survival=1)
    assert pytest.approx(WATERLINE_BYTES_PER_FLIP) == SEG_VALUE_PER_FLIP / SCORE_PER_BYTE


# ── 2. SELECTION IS REAL (the selective-vs-all A/B discrimination) ───────────
def test_effective_sigma_of_select_all_is_population_mean():
    """NO FAKE: a select-all mask collapses σ_eff to the population mean — the crude
    coder's 0.46. Selection only helps if the SELECTED subset's mean exceeds it."""
    rng = np.random.default_rng(0)
    sf = (rng.random(1000) < 0.46).astype(np.float64)  # ~0.46 survival
    select_all = np.ones(1000, dtype=bool)
    assert effective_sigma_of_subset(sf, select_all) == pytest.approx(sf.mean())


def test_oracle_selection_lifts_effective_sigma_to_one():
    """The oracle (select measured-survivors) has σ_eff == 1.0 by construction —
    the UPPER BOUND. The real question (tested elsewhere) is the SURVIVOR COUNT."""
    rng = np.random.default_rng(1)
    sf = (rng.random(2000) < 0.46).astype(np.float64)
    mask = select_survivors(survival_flags=sf, bytes_per_flip=0.985)
    assert mask.sum() == int(sf.sum())  # exactly the measured survivors
    assert effective_sigma_of_subset(sf, mask) == pytest.approx(1.0)


def test_predictor_selection_with_real_structure_beats_population():
    """When a decoder-FREE predictor CORRELATES with survival, thresholding it gives a
    selected subset with σ_eff ABOVE the population mean — the deployable nuance."""
    rng = np.random.default_rng(2)
    n = 5000
    predictor = rng.random(n)  # decoder-free feature in [0,1]
    # survival probability INCREASES with the predictor (real structure)
    p_survive = 0.2 + 0.7 * predictor
    sf = (rng.random(n) < p_survive).astype(np.float64)
    pop_mean = sf.mean()
    mask = select_survivors(
        survival_flags=sf, bytes_per_flip=0.985,
        survival_predictor=predictor, predictor_threshold=0.85,
    )
    assert mask.sum() > 0
    sel_sigma = effective_sigma_of_subset(sf, mask)
    assert sel_sigma > pop_mean + 0.1, "predictor selection must lift σ_eff materially"


def test_predictor_selection_with_no_structure_does_not_beat_population():
    """NO FAKE: if the predictor is INDEPENDENT of survival, thresholding gives a
    subset whose σ_eff ≈ the population mean — selection cannot manufacture survival."""
    rng = np.random.default_rng(3)
    n = 8000
    predictor = rng.random(n)
    sf = (rng.random(n) < 0.46).astype(np.float64)  # survival independent of predictor
    pop_mean = sf.mean()
    mask = select_survivors(
        survival_flags=sf, bytes_per_flip=0.985,
        survival_predictor=predictor, predictor_threshold=0.85,
    )
    sel_sigma = effective_sigma_of_subset(sf, mask)
    assert abs(sel_sigma - pop_mean) < 0.08, "no-structure predictor must NOT beat population"


def test_waterfill_admits_only_net_negative_flips():
    """The leverage waterfill admits a flip iff σ̂_i > b_i/WATERLINE (its own
    break-even). Flips below their break-even RAISE S and must NOT be admitted."""
    # 3 high-predicted-survival cheap flips (GO) + 3 low-survival expensive flips (NO-GO)
    shat = np.array([1.0, 1.0, 0.9, 0.1, 0.2, 0.0])
    bpf = np.array([0.5, 0.5, 0.5, 0.985, 0.985, 0.985])
    admitted = waterfill_by_leverage(predicted_survival=shat, per_flip_bytes=bpf)
    # each admitted flip must individually clear σ̂_i > b_i/WATERLINE
    for i in np.flatnonzero(admitted):
        assert shat[i] > bpf[i] / WATERLINE_BYTES_PER_FLIP
    # the low-survival flips must be rejected
    assert not admitted[3] and not admitted[4] and not admitted[5]
    assert admitted[0] and admitted[1] and admitted[2]


def test_waterfill_constant_zero_survival_admits_nothing():
    """NO FAKE: zero-predicted-survival flips have positive marginal ΔS — the waterfill
    admits NOTHING (a select-all stub would FAIL this)."""
    shat = np.zeros(50)
    bpf = np.full(50, 0.985)
    admitted = waterfill_by_leverage(predicted_survival=shat, per_flip_bytes=bpf)
    assert admitted.sum() == 0


def test_build_selection_go_when_survivors_clear_break_even():
    """End-to-end (DEPLOYABLE, the REALISTIC basin regime): a population whose mean
    survival is BELOW σ* (so all-flips is NO-GO, like the crude probe) but whose
    decoder-free predictor CORRELATES with ground-truth survival → survival-selection
    FLIPS it to GO. This is the headline nuance: NO-GO → GO purely by selection."""
    rng = np.random.default_rng(4)
    n = 4000
    predictor = rng.random(n)  # decoder-free survival estimate in [0,1]
    # ground-truth survival correlates with predictor; population mean ~0.46 (< σ*=0.774)
    p_survive = 0.05 + 0.82 * predictor
    sf = (rng.random(n) < p_survive).astype(np.float64)
    bpf = np.full(n, 0.985)  # the REALISTIC basin per-flip cost (σ* = 0.774)
    sel = build_selection(
        survival_flags=sf, per_flip_bytes=bpf,
        survival_predictor=predictor, predictor_threshold=0.92,
    )
    # all-flips at the population mean is NO-GO (the crude probe); selection is GO
    assert sel.net_delta_s_all_flips > 0.0, "all-flips must be NO-GO in the basin regime"
    assert sel.n_survivors_selected > 0
    assert sel.go is True
    assert sel.net_delta_s_selected < 0.0
    # the ground-truth σ_eff of the predictor-selected set clears the break-even σ*
    assert sel.sigma_effective_selected > sel.sigma_break_even
    assert sel.sigma_effective_selected > sel.sigma_population_mean + 0.1


def test_build_selection_no_go_when_population_unselectable():
    """NO FAKE: when the predictor is INDEPENDENT of ground-truth survival AND survival
    is low, the predictor-selected set's ground-truth σ_eff stays at the (low) population
    mean → net ΔS > 0 → NO-GO. The verdict is honest, not asserted."""
    rng = np.random.default_rng(5)
    n = 4000
    predictor = rng.random(n)
    sf = (rng.random(n) < 0.30).astype(np.float64)  # low survival, INDEPENDENT of predictor
    bpf = np.full(n, 0.6)  # predictor high → flips admitted, but ground-truth σ low
    sel = build_selection(
        survival_flags=sf, per_flip_bytes=bpf,
        survival_predictor=predictor, predictor_threshold=0.9,
    )
    # flips ARE admitted (predicted to survive) but the GROUND-TRUTH σ_eff (~0.30) is
    # below σ* at b=0.6 (=0.47) → net ΔS positive → honest NO-GO
    assert sel.sigma_effective_selected < sel.sigma_break_even
    assert sel.net_delta_s_selected >= 0.0
    assert sel.go is False


def test_select_all_baseline_matches_crude_probe_no_go():
    """The all-flips NET baseline (every flip coded at the population σ, b=0.985)
    reproduces the crude probe's NO-GO: net ΔS > 0 (RAISES S)."""
    rng = np.random.default_rng(6)
    n = 3000
    sf = (rng.random(n) < 0.46).astype(np.float64)
    bpf = np.full(n, 0.985)
    # uniform predictor at 0.99 → admits all (all "predicted to survive"), but the
    # GROUND-TRUTH σ_eff is the population mean ~0.46 < σ*(0.985)=0.774 → NO-GO
    sel = build_selection(
        survival_flags=sf, per_flip_bytes=bpf,
        survival_predictor=np.full(n, 0.99), predictor_threshold=0.5,
    )
    assert sel.net_delta_s_all_flips > 0.0  # crude all-flips RAISES S
    assert sel.go is False  # admitting all (no real selection) stays NO-GO


# ── 3. ROUND-TRIP: the bit-exact selective sidecar ──────────────────────────
def test_selective_sidecar_round_trips_bit_exact():
    """The admitted (pixel, class) survivor set serializes + parses back exactly."""
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(196_608 * 600, size=500, replace=False)).astype(np.int64)
    cls = rng.integers(0, 5, size=500).astype(np.int64)
    blob = encode_selective_sidecar(idx, cls, n_seg_classes=5)
    didx, dcls = decode_selective_sidecar(blob)
    assert np.array_equal(didx, idx)
    assert np.array_equal(dcls, cls)


def test_empty_selection_serializes_zero_bytes():
    """NO-GO / default-OFF: an empty admitted set adds ZERO archive bytes."""
    blob = encode_selective_sidecar(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert blob == b""
    didx, dcls = decode_selective_sidecar(b"")
    assert didx.size == 0 and dcls.size == 0


def test_selective_sidecar_byte_length_is_the_real_cost():
    """The sidecar byte length is the REAL coded cost (5 B/entry: u32 idx + u8 class) —
    the economics must price THIS, not a fictional smaller number."""
    n = 123
    idx = np.arange(n, dtype=np.int64)
    cls = (idx % 5).astype(np.int64)
    blob = encode_selective_sidecar(idx, cls)
    assert len(blob) == 12 + n * 5  # 12-byte header + 5 bytes per entry


def test_selective_sidecar_fail_closed_on_bad_magic():
    """A malformed/wrong-magic blob raises (no silent wrong-correction)."""
    with pytest.raises(ValueError):
        decode_selective_sidecar(b"XXXX" + b"\x00" * 8)


def test_selective_sidecar_rejects_out_of_range_class():
    with pytest.raises(ValueError):
        encode_selective_sidecar(np.array([0], dtype=np.int64), np.array([9], dtype=np.int64), n_seg_classes=5)


# ── 4. NO-OP / NO-FAKE GUARD ────────────────────────────────────────────────
def test_no_op_coder_fails_the_savings_assertion():
    """THE no-op guard: a coder that admits ALL predicted flips (select-all) has a LOWER
    ground-truth σ_eff than a survival-selective coder (high threshold) — the savings
    assertion (higher σ_eff, better per-flip net) holds ONLY for genuine selection. A
    fake that ignores selection is structurally caught here."""
    rng = np.random.default_rng(8)
    n = 5000
    # genuine selectable structure: ground-truth survival correlates with predictor
    predictor = rng.random(n)
    p = 0.05 + 0.9 * predictor
    sf = (rng.random(n) < p).astype(np.float64)
    bpf = np.full(n, 0.6)
    # FAKE: select-all (threshold below min predictor) — admits low-predictor flips too
    fake = build_selection(
        survival_flags=sf, per_flip_bytes=bpf,
        survival_predictor=predictor, predictor_threshold=0.0,
    )
    # REAL: survival-selective (high threshold) — admits only high-predictor flips
    real = build_selection(
        survival_flags=sf, per_flip_bytes=bpf,
        survival_predictor=predictor, predictor_threshold=0.9,
    )
    # the real selective coder has a STRICTLY HIGHER ground-truth effective σ
    assert real.sigma_effective_selected > fake.sigma_effective_selected
    # and a better (more-negative) net ΔS PER admitted flip
    assert real.n_survivors_selected > 0 and fake.n_survivors_selected > 0
    assert (real.net_delta_s_selected / real.n_survivors_selected) < (
        fake.net_delta_s_selected / fake.n_survivors_selected
    )
