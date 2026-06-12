# SPDX-License-Identifier: MIT
"""LEVER-D NUANCED — survival-selective margin-conditional seg-repair sidecar.

The CRUDE Lever-D (``distortion_finishing_kit.LeverDVerdict`` + the witness probe)
coded **ALL** boundary flips and measured a mean round-trip survival of ~0.464 →
structural NO-GO. This module is the NUANCED realization the operator directive
called for: it does NOT code every flip; it codes ONLY the sub-population whose
per-flip round-trip survival clears the **survival break-even** σ*, then waterfills
that survivor set by d_seg-leverage-per-byte to the marginal break-even λ.

THE SOLVABLE-MATH CORE (independently re-derived; see the probe + memo):

Coding ``N`` flips, each with survival probability ``σ`` (fraction of coded
corrections that actually land the GT argmax through the eval round-trip) and
per-flip byte cost ``b``, the contest score change is::

    net_ΔS = −100·(σ·N)/N_scored_total  +  25·(b·N)/N_a

The flip count ``N`` FACTORS OUT — the SIGN is set entirely by the *effective
survival of the coded subset*. GO (net < 0) iff::

    σ_effective  >  σ*  =  b · (25 · N_scored_total) / (100 · N_a)
                        =  b / WATERLINE_BYTES_PER_FLIP          (the canonical form)

With ``b ≈ 0.985`` B/flip and ``WATERLINE = 1.273`` B/flip → ``σ* ≈ 0.7737``.
The crude probe's all-flips mean ``σ = 0.464 << 0.774`` → NO-GO, structurally.

THE NUANCE (this module): ``σ`` is a DISTRIBUTION over flips, not a scalar. A
selective coder that codes only the flips PREDICTED (from decoder-FREE features:
margin, GT-class, local agreement) to survive can have a coded-subset
``σ_effective`` ABOVE 0.774 even though the population mean is 0.464 — IF the
survival distribution has selectable structure. This module MEASURES that
structure (it does not assume it), SELECTS the predicted-survivor subset, and
WATERFILLS by leverage-per-byte. The GO/NO-GO is then an HONEST measured verdict
on the converged base: does survival-selection lift ``σ_effective`` past ``σ*``?

NO FAKE (the discrimination contract):
  * ``survival_break_even_sigma(b)`` returns ``b / WATERLINE`` — the exact threshold;
    a coder that ignores it (codes all flips) FAILS the selective-vs-all A/B test.
  * ``select_survivors`` admits ONLY flips whose measured/predicted survival exceeds
    σ* — a select-all stub admits the whole population and is caught by
    ``effective_sigma_of_subset`` collapsing back to the population mean.
  * ``waterfill_by_leverage`` ranks survivors by (net seg value / byte cost) and
    admits the prefix whose marginal net ≥ 0 — a constant-value stub admits nothing.
  * the economics functions are pure closed-form (re-derivable by hand); the
    survival distribution + the effective-σ of the coded subset are MEASURED on the
    real frozen scorer + real eval round-trip by the companion probe.

This module is **inflate-side, default-OFF**: a disabled selective sidecar adds
ZERO bytes and is a byte-identical no-op (the live-arm safety contract). It ships a
section ONLY when the measured verdict is GO on the target base.

Cross-references:
  * ``tac.boundary_math.margin_conditional_residual`` (WATERLINE, ``measure_code_cost``,
    the conditional-position coder — REUSED, not reinvented).
  * ``experiments/witness_seg_boundary_decisive_probe.py`` (the crude all-flips
    survival measurement; this module's per-flip-survival probe extends it).
  * ``distortion_finishing_kit.LeverDVerdict`` (the crude NO-GO record this supersedes
    at IMPLEMENTATION level — paradigm intact per Catalog #307).
  * ``witness_seg_boundary_decisive_probe_20260612`` (the 884 flips/pair, 46.4%
    τ-insensitive survival, flip-count crux).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tac.boundary_math.margin_conditional_residual import (
    WATERLINE_BYTES_PER_FLIP,
)

# ── contest score constants (independently restated; consistent with the coder) ──
_SEG_WEIGHT = 100.0
_RATE_DENOM = 37_545_489  # N_a
_N_SCORED_PER_FRAME = 384 * 512  # 196,608
_N_FRAMES_SCORED = 600
_N_SCORED_TOTAL = _N_FRAMES_SCORED * _N_SCORED_PER_FRAME  # 117,964,800
# per-flip seg score value (a single fixed flip removes 100/N_scored_total of S)
SEG_VALUE_PER_FLIP = _SEG_WEIGHT / _N_SCORED_TOTAL  # 8.479e-7 score/flip
# per-byte score cost (25/N_a)
SCORE_PER_BYTE = 25.0 / _RATE_DENOM  # 6.659e-7 score/byte


# ═══════════════════════════════════════════════════════════════════════════
# 1. THE ECONOMICS — the survival break-even σ* (the core of the analysis).
# ═══════════════════════════════════════════════════════════════════════════
def survival_break_even_sigma(bytes_per_flip: float) -> float:
    """The minimum coded-subset survival ``σ*`` for a seg-repair sidecar to be GO.

    Derived from ``net_ΔS = −100·σ·N/N_scored_total + 25·b·N/N_a < 0``; the flip
    count ``N`` cancels, leaving::

        σ*  =  b · (25 · N_scored_total) / (100 · N_a)  =  b / WATERLINE_BYTES_PER_FLIP

    NO FAKE: a sidecar coding ALL flips has ``σ_effective = population_mean`` (~0.46
    on the basin), which is BELOW ``σ*`` for any ``b ≳ 0.59`` — the crude NO-GO. The
    *selective* coder must lift ``σ_effective`` above THIS value to flip to GO.
    """
    b = float(bytes_per_flip)
    if b < 0:
        raise ValueError(f"bytes_per_flip must be >= 0; got {b}")
    return b / WATERLINE_BYTES_PER_FLIP


def net_delta_s_seg_sidecar(
    n_flips_coded: int, sigma_effective: float, bytes_per_flip: float
) -> float:
    """The advisory net ΔS of coding ``n_flips_coded`` flips at effective survival
    ``σ_effective`` and per-flip byte cost ``bytes_per_flip``.

    ``net_ΔS = −100·(σ·N)/N_scored_total + 25·(b·N)/N_a``. NEGATIVE = GO (lowers S).
    Pure closed-form; the inputs (σ_effective, b) are the MEASURED quantities.
    """
    n = int(n_flips_coded)
    if n < 0:
        raise ValueError(f"n_flips_coded must be >= 0; got {n}")
    sigma = float(sigma_effective)
    b = float(bytes_per_flip)
    seg_gain = -_SEG_WEIGHT * (sigma * n) / _N_SCORED_TOTAL
    rate_cost = 25.0 * (b * n) / _RATE_DENOM
    return seg_gain + rate_cost


def effective_sigma_of_subset(
    survival_flags: np.ndarray, selected_mask: np.ndarray
) -> float:
    """The effective survival ``σ_effective`` of the SELECTED subset = the mean of
    ``survival_flags`` over ``selected_mask``.

    ``survival_flags``: per-flip 0/1 (or fractional) measured round-trip survival.
    ``selected_mask``: bool, which flips the selective coder admits.

    NO FAKE: if ``selected_mask`` is all-True (a select-all stub), this returns the
    POPULATION mean — exactly the 0.46 the crude probe measured; selection only helps
    if the selected subset's survival mean EXCEEDS the population mean. Returns 0.0
    for an empty selection (nothing coded → no seg gain claimable)."""
    sf = np.asarray(survival_flags, dtype=np.float64).reshape(-1)
    sel = np.asarray(selected_mask, dtype=bool).reshape(-1)
    if sf.shape != sel.shape:
        raise ValueError(
            f"survival_flags {sf.shape} and selected_mask {sel.shape} must match"
        )
    n_sel = int(sel.sum())
    if n_sel == 0:
        return 0.0
    return float(sf[sel].mean())


# ═══════════════════════════════════════════════════════════════════════════
# 2. SURVIVAL-ROBUST SELECTION (the nuance the crude probe missed).
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SurvivalSelection:
    """The result of survival-robust + leverage-waterfilled flip selection."""

    n_flips_total: int
    n_survivors_selected: int
    sigma_population_mean: float
    sigma_effective_selected: float
    sigma_break_even: float
    bytes_per_flip: float
    net_delta_s_selected: float
    net_delta_s_all_flips: float
    go: bool
    selected_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    rationale: str = ""

    @property
    def verdict(self) -> str:
        return "GO" if self.go else "NO-GO"


def select_survivors(
    *,
    survival_flags: np.ndarray,
    bytes_per_flip: float,
    survival_predictor: np.ndarray | None = None,
    predictor_threshold: float | None = None,
) -> np.ndarray:
    """Select the flips PREDICTED to survive (the survival-robust subset).

    Two selection modes (both real, NO FAKE):

      * **oracle** (``survival_predictor is None``): select flips whose MEASURED
        ``survival_flags`` == 1. This is the UPPER BOUND on what any predictor can
        achieve (it cheats by knowing the round-trip outcome). Use it to measure
        whether a selectable structure EXISTS (σ_effective of the survivors == 1.0
        by construction, but the *count* is the real signal — how many survivors are
        there, and is coding only them GO?).

      * **predictor** (``survival_predictor`` + ``predictor_threshold`` given): select
        flips whose decoder-FREE predictor score >= threshold. This is the
        DEPLOYABLE selection (the inflate side has the margin/class/agreement features
        but NOT the round-trip outcome). The effective σ of THIS subset is the honest
        deployable number.

    Returns a bool mask over the flips. A select-all result (threshold below the min
    predictor) is detectable — its effective σ collapses to the population mean.
    """
    sf = np.asarray(survival_flags).reshape(-1)
    if survival_predictor is None:
        # oracle: the measured survivors
        return sf.astype(bool)
    pred = np.asarray(survival_predictor, dtype=np.float64).reshape(-1)
    if pred.shape != sf.shape:
        raise ValueError(
            f"survival_predictor {pred.shape} must match survival_flags {sf.shape}"
        )
    if predictor_threshold is None:
        raise ValueError("predictor_threshold required when survival_predictor given")
    return pred >= float(predictor_threshold)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MARGIN-CONDITIONAL WATERFILL by d_seg-leverage-per-byte.
# ═══════════════════════════════════════════════════════════════════════════
def waterfill_by_leverage(
    *,
    predicted_survival: np.ndarray,
    per_flip_bytes: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Among candidate flips, admit those whose PREDICTED per-flip net ΔS is negative,
    ranked by net-value-per-byte — the KKT waterfill to the break-even λ.

    The inflate side codes WITHOUT knowing the round-trip outcome; it ranks by the
    decoder-FREE ``predicted_survival`` (a margin/class/agreement-derived survival
    estimate in [0,1]). Each flip's PREDICTED marginal score value =
    ``σ̂_i · SEG_VALUE_PER_FLIP``; its cost = ``b_i · SCORE_PER_BYTE``. A flip is
    PREDICTED net-negative-S (admit) iff ``σ̂_i > b_i / WATERLINE`` (its own
    break-even). We admit every candidate flip clearing its per-flip break-even
    (each is independently net-negative under the prediction; the waterline IS the
    per-flip admission test — there is no shared budget to exhaust).

    NO FAKE: a flip with σ̂_i below its own break-even is NOT admitted (predicted to
    RAISE S). A constant-σ̂-0 input admits nothing. The σ used here is the PREDICTION,
    not the ground truth — the realized σ_eff (which decides the true GO/NO-GO) is the
    ACTUAL survival of this admitted set, measured separately in ``build_selection``.
    Returns a bool mask of admitted flips.
    """
    shat = np.asarray(predicted_survival, dtype=np.float64).reshape(-1)
    bpf = np.asarray(per_flip_bytes, dtype=np.float64).reshape(-1)
    if shat.shape != bpf.shape:
        raise ValueError(
            f"predicted_survival {shat.shape} must match per_flip_bytes {bpf.shape}"
        )
    n = shat.shape[0]
    if candidate_mask is None:
        cand = np.ones(n, dtype=bool)
    else:
        cand = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        if cand.shape != shat.shape:
            raise ValueError("candidate_mask must match predicted_survival shape")

    admitted = np.zeros(n, dtype=bool)
    cand_idx = np.flatnonzero(cand)
    if cand_idx.size == 0:
        return admitted

    # predicted per-flip marginal net ΔS (negative = predicted-good).
    marginal = -(shat[cand_idx] * SEG_VALUE_PER_FLIP) + (bpf[cand_idx] * SCORE_PER_BYTE)
    admit_local = marginal < 0.0  # each clears its own per-flip break-even
    admitted[cand_idx[admit_local]] = True
    return admitted


def build_selection(
    *,
    survival_flags: np.ndarray,
    per_flip_bytes: np.ndarray,
    survival_predictor: np.ndarray | None = None,
    predictor_threshold: float | None = None,
    bytes_per_flip_summary: float | None = None,
) -> SurvivalSelection:
    """End-to-end survival-robust selection → leverage waterfill → measured verdict.

    The DEPLOYABLE/HONEST model separates the PREDICTION (what the inflate side uses to
    select + rank, a decoder-FREE survival estimate) from the GROUND TRUTH (the actual
    round-trip survival ``survival_flags``, which decides the true σ_eff + net ΔS):

      * ``survival_predictor`` given → it is BOTH the selection feature (threshold) AND
        the predicted survival the waterfill ranks by. ``survival_flags`` is the GROUND
        TRUTH used ONLY to compute the realized σ_eff + net of the admitted set. This is
        the honest deployable verdict — selection cannot peek at the round-trip outcome.
      * ``survival_predictor is None`` (ORACLE) → select uses the ground-truth survivors
        directly; the waterfill ranks by the ground truth too. This is the UPPER BOUND:
        the realized σ_eff of the admitted set is 1.0 by construction (only true
        survivors admitted), and the SURVIVOR COUNT is the real signal — is coding only
        the (perfectly-identified) survivors GO?

    The verdict is HONEST: GO iff the admitted set's net ΔS (computed on the GROUND-TRUTH
    survival) < 0 AND the set is non-empty.
    """
    sf = np.asarray(survival_flags, dtype=np.float64).reshape(-1)
    bpf = np.asarray(per_flip_bytes, dtype=np.float64).reshape(-1)
    n_total = sf.shape[0]
    if n_total == 0:
        return SurvivalSelection(
            n_flips_total=0,
            n_survivors_selected=0,
            sigma_population_mean=0.0,
            sigma_effective_selected=0.0,
            sigma_break_even=0.0,
            bytes_per_flip=0.0,
            net_delta_s_selected=0.0,
            net_delta_s_all_flips=0.0,
            go=False,
            selected_indices=np.array([], dtype=np.int64),
            rationale="no flips",
        )

    b_summary = (
        float(bytes_per_flip_summary)
        if bytes_per_flip_summary is not None
        else float(bpf.mean())
    )
    sigma_star = survival_break_even_sigma(b_summary)
    pop_mean = float(sf.mean())

    # SELECT (decoder-free): the candidate set the inflate side admits.
    candidate_mask = select_survivors(
        survival_flags=sf,
        bytes_per_flip=b_summary,
        survival_predictor=survival_predictor,
        predictor_threshold=predictor_threshold,
    )
    # WATERFILL by PREDICTED leverage. In oracle mode the prediction == ground truth
    # (sf); in predictor mode the prediction is the decoder-free ``survival_predictor``.
    predicted = sf if survival_predictor is None else np.asarray(
        survival_predictor, dtype=np.float64
    ).reshape(-1)
    admitted = waterfill_by_leverage(
        predicted_survival=predicted, per_flip_bytes=bpf, candidate_mask=candidate_mask
    )
    n_sel = int(admitted.sum())
    # σ_eff + net ΔS are computed on the GROUND-TRUTH survival of the admitted set.
    sigma_eff = effective_sigma_of_subset(sf, admitted)

    # net ΔS of the admitted set (use per-flip bytes summed exactly)
    if n_sel > 0:
        sel_idx = np.flatnonzero(admitted)
        seg_gain = -_SEG_WEIGHT * (sf[sel_idx].sum()) / _N_SCORED_TOTAL
        rate_cost = 25.0 * (bpf[sel_idx].sum()) / _RATE_DENOM
        net_sel = float(seg_gain + rate_cost)
    else:
        net_sel = 0.0

    # all-flips baseline (the crude probe): every flip coded at population σ
    net_all = net_delta_s_seg_sidecar(n_total, pop_mean, b_summary)

    go = bool(n_sel > 0 and net_sel < 0.0)
    rationale = (
        f"selected {n_sel}/{n_total} flips; σ_eff={sigma_eff:.4f} vs σ*={sigma_star:.4f}; "
        f"net_sel={net_sel:.6f} (GO={go}); net_all={net_all:.6f}"
    )
    return SurvivalSelection(
        n_flips_total=n_total,
        n_survivors_selected=n_sel,
        sigma_population_mean=pop_mean,
        sigma_effective_selected=sigma_eff,
        sigma_break_even=sigma_star,
        bytes_per_flip=b_summary,
        net_delta_s_selected=net_sel,
        net_delta_s_all_flips=net_all,
        go=go,
        selected_indices=np.flatnonzero(admitted).astype(np.int64),
        rationale=rationale,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. THE BIT-EXACT SELECTIVE SIDECAR (round-trips the coded survivors).
# ═══════════════════════════════════════════════════════════════════════════
import struct  # noqa: E402  (kept local to the section it serves)

_SELECTIVE_MAGIC = b"LDSV"  # Lever-D Selective
_SELECTIVE_VERSION = 1
_SELECTIVE_HEADER = struct.Struct("<4sBBHI")  # magic, version, n_seg_classes, _pad, n_entries
_SELECTIVE_ENTRY = struct.Struct("<IB")  # flat_pixel_index (uint32), gt_class (uint8)


def encode_selective_sidecar(
    flat_pixel_indices: np.ndarray,
    gt_classes: np.ndarray,
    *,
    n_seg_classes: int = 5,
) -> bytes:
    """Serialize the admitted (pixel, target-class) survivor set into a fixed-layout
    blob. Returns ``b""`` for an empty selection (the default-OFF / NO-GO contract:
    a selective coder that admits nothing adds ZERO bytes).

    NO FAKE: this stores the ACTUAL admitted survivor corrections; the byte length is
    the real cost ``decode_selective_sidecar`` must reproduce bit-exactly."""
    idx = np.asarray(flat_pixel_indices, dtype=np.int64).reshape(-1)
    cls = np.asarray(gt_classes, dtype=np.int64).reshape(-1)
    if idx.shape != cls.shape:
        raise ValueError(f"indices {idx.shape} must match classes {cls.shape}")
    if idx.size == 0:
        return b""
    if (cls < 0).any() or (cls >= n_seg_classes).any():
        raise ValueError("gt_classes out of range")
    if (idx < 0).any() or (idx >= (1 << 32)).any():
        raise ValueError("flat_pixel_indices out of uint32 range")
    out = bytearray(
        _SELECTIVE_HEADER.pack(
            _SELECTIVE_MAGIC, _SELECTIVE_VERSION, int(n_seg_classes), 0, int(idx.size)
        )
    )
    for i, c in zip(idx.tolist(), cls.tolist(), strict=True):
        out += _SELECTIVE_ENTRY.pack(int(i), int(c))
    return bytes(out)


def decode_selective_sidecar(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse the selective sidecar back into (flat_pixel_indices, gt_classes).

    An EMPTY blob → two empty arrays (the default-OFF round-trip). A malformed/wrong-
    magic blob raises (fail-closed; no silent wrong-correction)."""
    if not blob:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if len(blob) < _SELECTIVE_HEADER.size:
        raise ValueError("selective sidecar too short for header")
    magic, version, n_cls, _pad, n_entries = _SELECTIVE_HEADER.unpack(
        blob[: _SELECTIVE_HEADER.size]
    )
    if magic != _SELECTIVE_MAGIC:
        raise ValueError(f"bad selective sidecar magic {magic!r}")
    if version != _SELECTIVE_VERSION:
        raise ValueError(f"unsupported selective sidecar version {version}")
    expected = _SELECTIVE_HEADER.size + n_entries * _SELECTIVE_ENTRY.size
    if len(blob) != expected:
        raise ValueError(f"selective sidecar must be {expected} bytes; got {len(blob)}")
    idx = np.empty(n_entries, dtype=np.int64)
    cls = np.empty(n_entries, dtype=np.int64)
    off = _SELECTIVE_HEADER.size
    for i in range(n_entries):
        pi, c = _SELECTIVE_ENTRY.unpack(blob[off : off + _SELECTIVE_ENTRY.size])
        idx[i] = pi
        cls[i] = c
        off += _SELECTIVE_ENTRY.size
    return idx, cls


@dataclass
class LeverDSelectiveVerdict:
    """The MEASURED GO/NO-GO verdict for the nuanced survival-selective Lever-D.

    Carried as a NON-byte record on the basin/converged base; ships an actual sidecar
    section ONLY when ``go`` is True at the target operating point."""

    base_label: str
    n_pairs_measured: int
    mean_flips_per_pair: float
    bytes_per_flip: float
    sigma_break_even: float
    sigma_population_mean: float
    sigma_effective_selected: float
    n_survivors_per_pair: float
    net_delta_s_crude_all_flips: float
    net_delta_s_nuanced_selected: float
    go: bool
    reactivation_flip_count: float
    reactivation_survival_threshold: float
    rationale: str = ""

    @property
    def verdict(self) -> str:
        return "GO" if self.go else "NO-GO"


__all__ = [
    "SCORE_PER_BYTE",
    "SEG_VALUE_PER_FLIP",
    "LeverDSelectiveVerdict",
    "SurvivalSelection",
    "build_selection",
    "decode_selective_sidecar",
    "effective_sigma_of_subset",
    "encode_selective_sidecar",
    "net_delta_s_seg_sidecar",
    "select_survivors",
    "survival_break_even_sigma",
    "waterfill_by_leverage",
]
