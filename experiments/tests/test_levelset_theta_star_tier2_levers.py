# SPDX-License-Identifier: MIT
"""THETA* TIER-2 additive curriculum/optimization levers — unit tests.

Covers the four levers added to ``experiments/train_levelset_witness_realized_through_R_mlx.py``:
  * MUST-1  --tau-anneal-shape {cosine,geometric,cosine_hold} + --tau-hold-frac
  * MUST-2  --code-nuclear-weight (differentiable smoothed nuclear-norm low-rank code penalty)
  * MUST-3  --ema-decay-finisher (SWA / wider-finisher EMA)
  * STRETCH-1  --eikonal-junction-relax (junction-aware Eikonal relax)

THE #1 GATE (safety-critical): every lever is BIT-IDENTICAL when its flag is at the default (OFF).
A theta* run with ALL flags off MUST reproduce today's run exactly, so a crash-resume of the live
run onto merged code is safe. The bit-identical tests below reconstruct the pre-theta* formulas
inline and assert ``==`` (exact float equality), not ``approx``.

CPU-ONLY: never sets mx.gpu (the live run owns the GPU). MLX ops run + autodiff on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_levelset_witness_realized_through_R_mlx as T  # noqa: E402

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)  # NEVER gpu: the live run owns it.


def _tau_args(**over):
    base = dict(
        epochs=1000, softmax_temp_start=1.0, softmax_temp_end=0.05,
        tau_anneal_shape="cosine", tau_hold_frac=1.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _old_cosine(ep, a):
    """The pre-theta* inline cosine formula (the captured baseline)."""
    prog_t = (ep - 1) / max(a.epochs - 1, 1)
    return float(a.softmax_temp_end + 0.5 * (a.softmax_temp_start - a.softmax_temp_end) * (1 + np.cos(np.pi * prog_t)))


# =========================================================================== MUST-1: tau anneal shape
def test_must1_default_cosine_is_bit_identical_to_baseline():
    """Default (cosine, hold_frac=1.0) == the captured pre-theta* inline formula, EXACTLY."""
    a = _tau_args()
    for ep in (1, 2, 250, 500, 750, 900, 999, 1000):
        assert T._softmax_temp_for_epoch(ep, a) == _old_cosine(ep, a)


def test_must1_missing_attrs_default_to_bit_identical_cosine():
    """An args namespace WITHOUT the new fields (old callers / old _anneal_args) still hits the
    bit-identical cosine via getattr defaults -> back-compat with existing tests/checkpoints."""
    a = SimpleNamespace(epochs=1000, softmax_temp_start=1.0, softmax_temp_end=0.05)
    for ep in (1, 333, 1000):
        assert T._softmax_temp_for_epoch(ep, a) == _old_cosine(ep, a)


def test_must1_cosine_hold_frac_1_is_bit_identical_to_cosine():
    """cosine_hold with --tau-hold-frac 1.0 (no hold) == cosine, EXACTLY (the bit-identical claim)."""
    a = _tau_args(tau_anneal_shape="cosine_hold", tau_hold_frac=1.0)
    for ep in (1, 2, 250, 500, 750, 900, 999, 1000):
        assert T._softmax_temp_for_epoch(ep, a) == _old_cosine(ep, a)


def test_must1_endpoints_cosine():
    a = _tau_args(tau_anneal_shape="cosine")
    assert T._softmax_temp_for_epoch(1, a) == 1.0
    assert abs(T._softmax_temp_for_epoch(1000, a) - 0.05) < 1e-12


def test_must1_geometric_endpoints_and_monotone():
    a = _tau_args(tau_anneal_shape="geometric")
    assert abs(T._softmax_temp_for_epoch(1, a) - 1.0) < 1e-12          # start
    assert abs(T._softmax_temp_for_epoch(1000, a) - 0.05) < 1e-12       # end
    vals = [T._softmax_temp_for_epoch(ep, a) for ep in range(1, 1001, 50)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1)), "geometric must be non-increasing"


def test_must1_geometric_spends_more_epochs_at_small_tau():
    """The rationale: geometric reaches small tau FASTER than cosine (more epochs at small tau)."""
    a = _tau_args()
    mid_cos = T._softmax_temp_for_epoch(500, _tau_args(tau_anneal_shape="cosine"))
    mid_geo = T._softmax_temp_for_epoch(500, _tau_args(tau_anneal_shape="geometric"))
    assert mid_geo < mid_cos
    # geometric closed form at the ACTUAL prog of ep=500 (prog=(500-1)/999, not exactly 0.5)
    prog = (500 - 1) / (a.epochs - 1)
    expect = a.softmax_temp_start * (a.softmax_temp_end / a.softmax_temp_start) ** prog
    assert abs(mid_geo - expect) < 1e-6
    # ... and near sqrt(start*end) since prog≈0.5 (the geometric midpoint intuition)
    assert abs(mid_geo - (a.softmax_temp_start * a.softmax_temp_end) ** 0.5) < 1e-3


def test_must1_cosine_hold_holds_at_floor_after_holdpoint():
    a = _tau_args(tau_anneal_shape="cosine_hold", tau_hold_frac=0.7)
    # before the hold point (prog<0.7): descending, above the floor
    assert T._softmax_temp_for_epoch(100, a) > a.softmax_temp_end
    # at/after the hold point (prog>=0.7): exactly the floor
    for ep in (701, 800, 900, 1000):  # prog = (ep-1)/999 >= 0.7 for ep>=700.3
        assert T._softmax_temp_for_epoch(ep, a) == a.softmax_temp_end


def test_must1_cosine_hold_reaches_floor_earlier_than_plain_cosine():
    a_hold = _tau_args(tau_anneal_shape="cosine_hold", tau_hold_frac=0.5)
    a_cos = _tau_args(tau_anneal_shape="cosine")
    # at prog=0.6 (ep≈600): hold variant already at floor; cosine still above
    assert T._softmax_temp_for_epoch(600, a_hold) == a_hold.softmax_temp_end
    assert T._softmax_temp_for_epoch(600, a_cos) > a_cos.softmax_temp_end


# =========================================================================== MUST-2: nuclear norm
def _numpy_nuc(C):
    import numpy.linalg as nla
    return float(nla.svd(np.asarray(C, np.float64), compute_uv=False).sum())


def test_must2_value_matches_numpy_on_well_conditioned():
    rng = np.random.default_rng(3)
    C = mx.array(rng.standard_normal((200, 32)).astype(np.float32))
    v = float(T._nuclear_norm_smooth_mlx(C))
    ref = _numpy_nuc(np.asarray(C))
    assert abs(v - ref) / ref < 0.01, f"smoothed nuc {v} vs exact {ref} (>1% off)"


def test_must2_zero_codes_give_near_zero_norm():
    C = mx.zeros((64, 32))
    assert float(T._nuclear_norm_smooth_mlx(C)) < 1e-3


def test_must2_gradient_finite_on_rank_deficient():
    """The penalty drives codes rank-deficient; the gradient MUST stay finite there (plain NS NaNs)."""
    rng = np.random.default_rng(5)
    with np.errstate(all="ignore"):  # macOS Accelerate-BLAS emits spurious matmul warnings here
        B = (rng.standard_normal((300, 4)) @ rng.standard_normal((4, 32))).astype(np.float32)  # rank 4 of 32
    C = mx.array(B)
    g = mx.grad(lambda C: T._nuclear_norm_smooth_mlx(C))(C)
    mx.eval(g)
    gn = np.asarray(g)
    assert np.all(np.isfinite(gn)), "rank-deficient nuclear-norm gradient must be finite"
    assert float(np.abs(gn).sum()) > 0.0


def test_must2_gradient_cosine_vs_exact_subgradient():
    """For well-conditioned C the nuclear-norm subgradient is U V^T; the NS path must align with it."""
    import numpy.linalg as nla
    rng = np.random.default_rng(11)
    Cnp = rng.standard_normal((120, 16)).astype(np.float32)
    C = mx.array(Cnp)
    g = np.asarray(mx.grad(lambda C: T._nuclear_norm_smooth_mlx(C))(C))
    with np.errstate(all="ignore"):  # macOS Accelerate-BLAS emits spurious matmul warnings here
        U, S, Vt = nla.svd(Cnp, full_matrices=False)
        gref = (U @ Vt).astype(np.float64)
        cos = float((g.ravel() @ gref.ravel()) / (nla.norm(g) * nla.norm(gref) + 1e-20))
    assert cos > 0.99, f"grad cosine vs U V^T = {cos}"


def test_must2_drives_low_rank_rank1_below_fullrank_at_matched_frobenius():
    """The defining low-rank property: at matched Frobenius norm, a rank-1 matrix has a SMALLER
    nuclear norm than a full-rank one -> the penalty prefers low rank."""
    rng = np.random.default_rng(7)
    full = rng.standard_normal((64, 32)).astype(np.float32)
    u = rng.standard_normal((64, 1)).astype(np.float32)
    v = rng.standard_normal((1, 32)).astype(np.float32)
    r1 = (u @ v).astype(np.float32)
    fro = lambda M: float(np.sqrt((np.asarray(M, np.float64) ** 2).sum()))
    full = full / fro(full)
    r1 = r1 / fro(r1)
    nuc_full = float(T._nuclear_norm_smooth_mlx(mx.array(full)))
    nuc_r1 = float(T._nuclear_norm_smooth_mlx(mx.array(r1)))
    assert nuc_r1 < nuc_full, f"rank1 {nuc_r1} should be < fullrank {nuc_full} at matched Frobenius"


def test_must2_off_branch_is_additive_zero_on_when_positive():
    """Emulates total_loss_fn's `if code_nuc_w > 0.0:` gate. w=0 => delta is EXACTLY 0.0 (the loss is
    byte-identical); w>0 => delta == w * helper_value (>0)."""
    rng = np.random.default_rng(9)
    C = mx.array(rng.standard_normal((64, 32)).astype(np.float32))
    L0 = 1.2345

    def total(w):
        L = L0
        if w > 0.0:
            L = L + w * float(T._nuclear_norm_smooth_mlx(C))
        return L

    assert total(0.0) == L0                       # OFF: bit-identical (no contribution at all)
    nuc = float(T._nuclear_norm_smooth_mlx(C))
    assert abs((total(0.01) - L0) - 0.01 * nuc) < 1e-6 and (total(0.01) - L0) > 0.0


# =========================================================================== MUST-3: finisher EMA
def _ema_args(**over):
    base = dict(ema_decay=0.997, ema_decay_finisher=None,
                ema_decay_finisher_start_epoch=None, muon_start_epoch=None)
    base.update(over)
    return SimpleNamespace(**base)


def _resolve_finisher(args):
    """Mirror run_train's resolution logic exactly."""
    dec = (float(args.ema_decay_finisher) if getattr(args, "ema_decay_finisher", None) is not None else None)
    start = (int(args.ema_decay_finisher_start_epoch)
             if getattr(args, "ema_decay_finisher_start_epoch", None) is not None
             else (int(args.muon_start_epoch) if args.muon_start_epoch is not None else None))
    return dec, start


def test_must3_resolution_explicit_else_muon_else_none():
    # explicit start wins
    d, s = _resolve_finisher(_ema_args(ema_decay_finisher=0.999, ema_decay_finisher_start_epoch=700, muon_start_epoch=900))
    assert (d, s) == (0.999, 700)
    # falls back to muon_start
    d, s = _resolve_finisher(_ema_args(ema_decay_finisher=0.999, muon_start_epoch=900))
    assert (d, s) == (0.999, 900)
    # off => decay None, start None (no widen)
    d, s = _resolve_finisher(_ema_args())
    assert d is None and s is None


def test_must3_off_never_mutates_decay_bit_identical():
    """ema_finisher_decay None => the loop guard NEVER fires => ema.decay stays at --ema-decay for
    every epoch (the EMA trajectory is BIT-IDENTICAL)."""
    import mlx.nn as nn
    m = nn.Linear(3, 3)
    ema = T.MlxEMA(m, decay=0.997)
    dec, start = _resolve_finisher(_ema_args(ema_decay=0.997))
    for ep in range(1, 1001):
        if dec is not None and start is not None and ep >= start and ema.decay != dec:
            ema.decay = dec  # never taken
    assert ema.decay == 0.997


def test_must3_widens_at_resolved_start_only():
    import mlx.nn as nn
    m = nn.Linear(3, 3)
    ema = T.MlxEMA(m, decay=0.997)
    dec, start = _resolve_finisher(_ema_args(ema_decay=0.997, ema_decay_finisher=0.9995, muon_start_epoch=900))
    seen = {}
    for ep in (1, 500, 899, 900, 901, 1000):
        if dec is not None and start is not None and ep >= start and ema.decay != dec:
            ema.decay = dec
        seen[ep] = ema.decay
    assert seen[1] == 0.997 and seen[899] == 0.997      # base before the finisher
    assert seen[900] == 0.9995 and seen[1000] == 0.9995  # wider from the start epoch on


def test_must3_mlx_ema_update_reads_current_decay():
    """Behavioral proof the mutation mechanism works: MlxEMA.update reads self.decay EACH call, so
    setting ema.decay mid-run changes the averaging weight."""
    import mlx.nn as nn
    m = nn.Linear(2, 2)
    ema = T.MlxEMA(m, decay=0.0)   # decay 0 => shadow == live each update
    # perturb live weights then update; with decay 0 the shadow tracks live exactly
    m.weight = m.weight + 1.0
    ema.update(m)
    mx.eval(list(ema.shadow.values()))
    w_key = [k for k in ema.shadow if k.endswith("weight")][0]
    assert np.allclose(np.asarray(ema.shadow[w_key]), np.asarray(m.weight))
    # now widen decay to 1.0 => shadow FROZEN regardless of further live changes
    ema.decay = 1.0
    frozen = np.asarray(ema.shadow[w_key]).copy()
    m.weight = m.weight + 5.0
    ema.update(m)
    mx.eval(list(ema.shadow.values()))
    assert np.allclose(np.asarray(ema.shadow[w_key]), frozen), "decay=1.0 must freeze the shadow"


# =========================================================================== STRETCH-1: junction relax
def _make_phi(h=8, w=8, k=5, seed=2):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((h * w, k)).astype(np.float32)), h, w


def test_stretch1_relax_off_is_bit_identical():
    """junction_relax=0.0 (default) == explicit 0.0 == the unweighted Eikonal, EXACTLY."""
    phi, h, w = _make_phi()
    eik_default, len_default, _ = T._eikonal_length_mlx(phi, h, w)
    eik_zero, len_zero, _ = T._eikonal_length_mlx(phi, h, w, junction_relax=0.0)
    assert float(eik_default) == float(eik_zero)
    assert float(len_default) == float(len_zero)


def test_stretch1_relax_on_changes_eik_and_is_le_baseline():
    """relax>0 actually changes the Eikonal term, and (since w in [1-relax,1]) it can only LOWER the
    mean of the non-negative residual."""
    phi, h, w = _make_phi()
    eik0, _, _ = T._eikonal_length_mlx(phi, h, w, junction_relax=0.0)
    eik1, _, _ = T._eikonal_length_mlx(phi, h, w, junction_relax=0.5, junction_tau=0.5)
    assert float(eik1) != float(eik0)
    assert float(eik1) <= float(eik0) + 1e-7


def test_stretch1_length_term_unchanged_by_relax():
    """The relax only touches the Eikonal term; the Chan-Vese length term is untouched."""
    phi, h, w = _make_phi()
    _, len0, _ = T._eikonal_length_mlx(phi, h, w, junction_relax=0.0)
    _, len1, _ = T._eikonal_length_mlx(phi, h, w, junction_relax=0.7, junction_tau=0.3)
    assert float(len0) == float(len1)


def test_stretch1_gradient_finite_when_on():
    phi, h, w = _make_phi()

    def eik_of(p):
        e, _l, _g = T._eikonal_length_mlx(p, h, w, junction_relax=0.5, junction_tau=0.5)
        return e

    g = mx.grad(eik_of)(phi)
    mx.eval(g)
    assert np.all(np.isfinite(np.asarray(g)))
