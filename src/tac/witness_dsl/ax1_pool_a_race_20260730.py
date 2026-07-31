"""ddm_pa1b — the Pool-A RACE HARNESS: the hull-curvature instrument (gc10 op-routable 3, #793).

gc10 re-chartered the Pool-A race as THE HULL-CURVATURE INSTRUMENT (sequence step-2): the gc9
"stuck-hull" framing rests on TWO on-contour points (QA24 c=0.08815, B c=0.08835) at iso-c ~0.088
— the Assumption-Adversary's binding dissent: *"two points define a line, not a hull, and certainly
not its curvature."*  The race adds the missing points {rowband ∥ margin-quant ∥ delta-sparsity at
matched SMEVR bytes} and MEASURES whether any matched-bytes point sits strictly INSIDE the iso-c
contour (c < 0.088 = the hull moved) or slides ALONG it (c ≈ 0.088 = the 3-move negative record
extends to the class).

This module is SCORER-FREE by construction (gc10 §7 / task #793): it builds the race PLAN, the $0
band-edge THEOREM (exhaust theorem #2), and the curvature ANALYZER.  The d_seg column of each
receipt is MEASURED by MAIN when the race fires (scorer slot after nv1); the analyzer CONSUMES the
receipts and emits the geometric verdict.  Nothing here launches a scorer job.

Three tools:
  1. ``enumerate_band_edge_theorem`` — the COMPLETE band-placement enumeration (exhaust theorem #2),
     deriving the true count from the RowBandGrammar constraint set (the memo's C(24,2)=276 is
     corrected to the full C(25,2)=300 boundary space; the enumeration is the SUPERSET so no optimal
     placement is missed) + the $0 geometric-rate-proxy Pareto frontier (min counted cells subject to
     the measured flip-mass coverage) → the PROVABLY-optimal-rate band, not a sampled one.
  2. ``seal_matched_bytes_race`` — the seal-time byte-matcher: price each arm's SMEVR estimate on a
     supplied parent token field via the SHIPPED coder, enforce ±1% matching (refuse mismatch = the
     burn-2 tuning-step signal), and record each arm's sealed-ticket argv-diff vs the control (the
     pre-fire diff law from the QA86c #517-twin incident).
  3. ``HullCurvatureAnalyzer`` — the typed race-receipt schema + the hull-curvature verdict (does any
     matched-bytes point sit strictly inside the iso-c contour?).

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED.  Plan/theorem/analyzer only; score_claim=False;
measured d_seg columns are [macOS-CPU advisory] until byte-closed on the authority axis by MAIN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb

import numpy as np

from tac.canonical_equations.ddm_gc9_seg_rate_product_law_20260730 import product_c
from tac.witness_dsl.qa84_rowband_grammar_20260731 import RowBandGrammar

#: The gc9 token-family hull iso-c (MEASURED QA24 anchor c=0.08815; B slid to 0.08835) — the
#: contour the race probes.  Sourced from the registered product-law anchor, not a magic number.
HULL_ISO_C: float = 0.08815

#: SMEVR byte-match tolerance (±1%); dw1 precedent + gc10 "matched-bytes constraint is
#: SMEVR-priced, ±1% tolerance per dw1".
MATCHED_BYTES_TOL: float = 0.01


# ===========================================================================
# 1. Exhaust theorem #2 — the complete band-edge enumeration.
# ===========================================================================
@dataclass(frozen=True)
class BandEdgeConfig:
    """One band placement: D8 fine-grid rows [lo, hi) kept FREE, bulk tied coarse_factor²."""

    band_row_lo: int          # D8 fine-grid row (coarse-aligned)
    band_row_hi: int
    independent_cells: int     # counted DOF (rate proxy)
    band_spec_bytes: int
    flip_mass_covered: float   # fraction of the measured per-row flip mass inside [lo,hi)

    def rate_proxy_cells(self) -> int:
        return self.independent_cells


@dataclass(frozen=True)
class BandEdgeTheorem:
    """The complete band-edge enumeration result (exhaust theorem #2)."""

    fine_gh: int
    fine_gw: int
    coarse_factor: int
    total_configs: int
    memo_claimed_count: int
    count_derivation: str
    configs: tuple[BandEdgeConfig, ...]
    pareto_frontier: tuple[BandEdgeConfig, ...]     # (min cells, max coverage) non-dominated set
    optimal_at_coverage: dict[str, BandEdgeConfig]  # min-cells band achieving each coverage target

    def verdict(self) -> dict[str, object]:
        return {
            "grid_fine": [self.fine_gh, self.fine_gw], "coarse_factor": self.coarse_factor,
            "total_configs_DERIVED": self.total_configs,
            "memo_claimed_count": self.memo_claimed_count,
            "count_matches_memo": self.total_configs == self.memo_claimed_count,
            "count_derivation": self.count_derivation,
            "pareto_frontier_size": len(self.pareto_frontier),
            "optimal_at_coverage": {
                k: {"band_rows_D8": [c.band_row_lo, c.band_row_hi],
                    "independent_cells": c.independent_cells,
                    "flip_mass_covered": round(c.flip_mass_covered, 4)}
                for k, c in self.optimal_at_coverage.items()},
        }


def per_row_flip_mass_from_field(cell_flip_mass: np.ndarray) -> np.ndarray:
    """Reduce a per-cell flip-mass field (gh,gw) to a per-fine-row measure (gh,) by summing
    columns — the measured measure the band-edge theorem optimizes coverage against."""
    return np.asarray(cell_flip_mass, dtype=np.float64).sum(axis=1)


def enumerate_band_edge_theorem(
    per_row_flip_mass: np.ndarray, *, fine_gw: int = 64, coarse_factor: int = 2,
    code_width: int = 4, coverage_targets: tuple[float, ...] = (0.5, 0.721, 0.9),
) -> BandEdgeTheorem:
    """EXHAUST THEOREM #2: enumerate the COMPLETE coarse-aligned band-placement space and return
    the geometric-rate Pareto frontier + the min-rate band at each flip-mass coverage target.

    COUNT DERIVATION (corrects the memo's C(24,2)=276): the RowBandGrammar constraint set is
    ``0 <= lo < hi <= fine_gh`` with ``lo, hi`` multiples of ``coarse_factor``.  On the D8 grid
    (fine_gh=len(per_row_flip_mass)=48, coarse_factor=2) the coarse-aligned boundaries are
    {0,2,...,48} = ``fine_gh/coarse_factor + 1`` = 25 boundaries; a non-empty band picks 2 of
    them (lo<hi) => **C(25,2) = 300** placements.  The memo's C(24,2)=276 counts pairs of the 24
    *D16 rows* (equivalently, excludes the 24 single-D16-row bands from the 300) — an
    off-by-one (boundaries vs rows).  We enumerate the full 300 SUPERSET so no optimal placement
    is missed: the arm becomes a THEOREM (provably-optimal-rate band), not a sample.

    ``per_row_flip_mass`` (fine_gh,) is the MEASURED per-fine-row flip mass (from the QA80 field
    via ``per_row_flip_mass_from_field``); coverage of a band = the fraction inside [lo,hi)."""
    prm = np.asarray(per_row_flip_mass, dtype=np.float64)
    fine_gh = int(prm.size)
    if fine_gh % coarse_factor:
        raise ValueError(f"fine_gh {fine_gh} not a multiple of coarse_factor {coarse_factor}")
    total_mass = float(prm.sum())
    n_boundaries = fine_gh // coarse_factor + 1
    boundaries = [i * coarse_factor for i in range(n_boundaries)]
    derived_total = comb(n_boundaries, 2)
    memo_count = comb(fine_gh // coarse_factor, 2)  # C(24,2)=276 for the standard D8/D16 grid

    cfgs: list[BandEdgeConfig] = []
    for i in range(n_boundaries):
        for j in range(i + 1, n_boundaries):
            lo, hi = boundaries[i], boundaries[j]
            g = RowBandGrammar(fine_gh, fine_gw, lo, hi, coarse_factor, code_width)
            covered = float(prm[lo:hi].sum()) / total_mass if total_mass > 0 else 0.0
            cfgs.append(BandEdgeConfig(
                band_row_lo=lo, band_row_hi=hi, independent_cells=g.independent_cells(),
                band_spec_bytes=g.band_spec_bytes(), flip_mass_covered=covered))
    if len(cfgs) != derived_total:
        raise AssertionError(f"enumerated {len(cfgs)} != derived {derived_total}")

    # Pareto frontier: non-dominated in (minimize cells, maximize coverage).
    by_cells = sorted(cfgs, key=lambda c: (c.independent_cells, -c.flip_mass_covered))
    pareto: list[BandEdgeConfig] = []
    best_cov = -1.0
    for c in by_cells:
        if c.flip_mass_covered > best_cov + 1e-12:
            pareto.append(c)
            best_cov = c.flip_mass_covered
    # min-cells band achieving each coverage target (provably optimal by complete enumeration).
    optimal: dict[str, BandEdgeConfig] = {}
    for tgt in coverage_targets:
        feas = [c for c in cfgs if c.flip_mass_covered >= tgt - 1e-9]
        if feas:
            optimal[f"cov>={tgt:g}"] = min(
                feas, key=lambda c: (c.independent_cells, c.band_row_lo))
    return BandEdgeTheorem(
        fine_gh=fine_gh, fine_gw=fine_gw, coarse_factor=coarse_factor,
        total_configs=derived_total, memo_claimed_count=memo_count,
        count_derivation=(
            f"coarse-aligned boundaries={n_boundaries} (fine_gh/{coarse_factor}+1); "
            f"non-empty bands = C({n_boundaries},2) = {derived_total}; memo C(24,2)="
            f"{memo_count} excludes the {derived_total - memo_count} single-D16-row bands "
            "(boundaries-vs-rows off-by-one). Enumeration = full superset (no placement missed)."),
        configs=tuple(cfgs), pareto_frontier=tuple(pareto), optimal_at_coverage=optimal)


# ===========================================================================
# 2. Matched-SMEVR-bytes race seal (byte-matcher + argv-diff verification).
# ===========================================================================
@dataclass(frozen=True)
class ArmByteRecord:
    arm: str
    smevr_bytes: int
    delta_vs_control_bytes: int
    within_tol: bool
    argv_diff_vs_control: dict[str, object]
    ticket_hash: str


@dataclass(frozen=True)
class SealVerdict:
    control_arm: str
    control_bytes: int
    tol: float
    arms: tuple[ArmByteRecord, ...]
    matched: bool
    refusal_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "control_arm": self.control_arm, "control_bytes": self.control_bytes,
            "tol": self.tol, "matched": self.matched, "refusal_reason": self.refusal_reason,
            "arms": [{"arm": a.arm, "smevr_bytes": a.smevr_bytes,
                      "delta_vs_control_bytes": a.delta_vs_control_bytes,
                      "within_tol": a.within_tol, "ticket_hash": a.ticket_hash,
                      "argv_diff_vs_control": a.argv_diff_vs_control} for a in self.arms],
        }


def smevr_bytes_of_field(codes_u8: np.ndarray, levels: int) -> int:
    """SMEVR byte estimate of a quantized token field via the SHIPPED r7 coder (the real
    coder that byte-closes the archive; the matched-bytes currency).  Deterministic, lossless."""
    from experiments.ddm_r7_token_coder import encode_token_codes

    return len(encode_token_codes(np.ascontiguousarray(codes_u8, dtype=np.uint8),
                                  levels=int(levels), codec="smevr"))


def _argv_diff(control_argv: list[str], arm_argv: list[str]) -> dict[str, object]:
    """The pre-fire argv-diff law (QA86c #517-twin incident: a mid-run resume drifted 2
    module-default flags; MAIN's pre-fire diff caught it).  Returns {added, removed, changed}
    flag→value maps between the control and arm sealed argv."""
    def _to_map(argv: list[str]) -> dict[str, str]:
        m: dict[str, str] = {}
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok.startswith("--"):
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    m[tok] = argv[i + 1]
                    i += 2
                else:
                    m[tok] = ""      # store_true
                    i += 1
            else:
                i += 1
        return m
    cm, am = _to_map(control_argv), _to_map(arm_argv)
    added = {k: am[k] for k in am if k not in cm}
    removed = {k: cm[k] for k in cm if k not in am}
    changed = {k: {"control": cm[k], "arm": am[k]} for k in am if k in cm and am[k] != cm[k]}
    return {"added": added, "removed": removed, "changed": changed}


def seal_matched_bytes_race(
    control_argv: list[str], control_codes_u8: np.ndarray,
    arms: dict[str, tuple[list[str], np.ndarray]], *, levels: int,
    control_ticket_hash: str = "", arm_ticket_hashes: dict[str, str] | None = None,
    tol: float = MATCHED_BYTES_TOL,
) -> SealVerdict:
    """Seal a matched-SMEVR-bytes race: price the control + each arm's token field via the
    shipped coder, enforce ±``tol`` byte matching, and record each arm's argv-diff vs the control.

    ``control_codes_u8`` / each arm's codes are the parent-derived quantized token fields
    (P,gh,gw,c) uint8 (parent resolved at FIRE time — ps1's composed best or B; NEVER hardcoded).
    REFUSES (matched=False + reason) if any arm is outside tol — the burn-2 tuning-step signal
    (adjust code_width/grid/level-map to re-match) per the non-additive-pools law (the race, not a
    per-lever claim, is the composition adjudicator)."""
    arm_ticket_hashes = arm_ticket_hashes or {}
    ctrl_bytes = smevr_bytes_of_field(control_codes_u8, levels)
    recs: list[ArmByteRecord] = []
    off: list[str] = []
    for name in sorted(arms):
        arm_argv, codes = arms[name]
        b = smevr_bytes_of_field(codes, levels)
        within = abs(b - ctrl_bytes) <= tol * max(ctrl_bytes, 1)
        if not within:
            off.append(f"{name}({b}B, {100.0 * (b - ctrl_bytes) / max(ctrl_bytes, 1):+.2f}%)")
        recs.append(ArmByteRecord(
            arm=name, smevr_bytes=b, delta_vs_control_bytes=b - ctrl_bytes, within_tol=within,
            argv_diff_vs_control=_argv_diff(control_argv, arm_argv),
            ticket_hash=arm_ticket_hashes.get(name, "")))
    matched = not off
    reason = None if matched else (
        f"arms outside ±{tol:.0%} of control {ctrl_bytes}B: " + ", ".join(off)
        + " — re-match (code_width/grid/level-map tuning step) before fire")
    return SealVerdict(control_arm="control", control_bytes=ctrl_bytes, tol=tol,
                       arms=tuple(recs), matched=matched, refusal_reason=reason)


# ===========================================================================
# 3. Hull-curvature verdict schema + analyzer.
# ===========================================================================
@dataclass(frozen=True)
class RaceReceipt:
    """One matched-bytes race point (the d_seg column is MEASURED by MAIN at fire time)."""

    lever: str                 # "control" | "rowband" | "margin_quant" | "delta_sparsity" | joint
    window: str                # the sealed window id
    d_seg: float               # MEASURED realized argmax d_seg (scorer, [macOS-CPU advisory])
    counted_bytes: int         # SMEVR byte-closed counted bytes
    delta_d_seg_vs_control: float
    delta_bytes_vs_control: int
    within_matched_tol: bool

    def c(self) -> float:
        return product_c(self.d_seg, self.counted_bytes)


@dataclass(frozen=True)
class HullCurvatureVerdict:
    iso_c: float
    control_c: float
    n_points: int
    inside_contour: tuple[str, ...]      # levers with c strictly < iso_c (hull MOVED there)
    on_contour: tuple[str, ...]          # levers with c ≈ iso_c (slid ALONG)
    outside_contour: tuple[str, ...]     # levers with c > iso_c (worse)
    hull_moved: bool                     # ANY matched-bytes point strictly inside
    best_lever: str | None
    best_c: float | None
    curvature_note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "iso_c": self.iso_c, "control_c": round(self.control_c, 6), "n_points": self.n_points,
            "hull_moved": self.hull_moved, "best_lever": self.best_lever,
            "best_c": None if self.best_c is None else round(self.best_c, 6),
            "inside_contour": list(self.inside_contour), "on_contour": list(self.on_contour),
            "outside_contour": list(self.outside_contour), "curvature_note": self.curvature_note,
        }


class HullCurvatureAnalyzer:
    """Consume matched-bytes race receipts → the hull-curvature verdict.

    The question (gc10, Assumption-Adversary): does ANY matched-bytes point sit strictly INSIDE
    the iso-c contour (c < iso_c ⇒ the hull MOVED) or do they all slide ALONG it (c ≈ iso_c ⇒
    the 3-move negative record extends to the class)?  ``on_contour`` uses a relative band so a
    point within ``rel_band`` of iso_c is "along the line," not a move."""

    def __init__(self, iso_c: float = HULL_ISO_C, *, rel_band: float = 0.02):
        self.iso_c = float(iso_c)
        self.rel_band = float(rel_band)

    def analyze(self, receipts: list[RaceReceipt]) -> HullCurvatureVerdict:
        if not receipts:
            raise ValueError("no receipts")
        ctrl = next((r for r in receipts if r.lever == "control"), None)
        control_c = ctrl.c() if ctrl is not None else self.iso_c
        band = self.rel_band * self.iso_c
        inside, on, outside = [], [], []
        matched = [r for r in receipts if r.lever != "control" and r.within_matched_tol]
        for r in matched:
            cval = r.c()
            if cval < self.iso_c - band:
                inside.append(r.lever)
            elif cval > self.iso_c + band:
                outside.append(r.lever)
            else:
                on.append(r.lever)
        best = min(matched, key=lambda r: r.c()) if matched else None
        note = self._curvature_note(matched, control_c)
        return HullCurvatureVerdict(
            iso_c=self.iso_c, control_c=control_c, n_points=len(matched),
            inside_contour=tuple(sorted(inside)), on_contour=tuple(sorted(on)),
            outside_contour=tuple(sorted(outside)), hull_moved=bool(inside),
            best_lever=None if best is None else best.lever,
            best_c=None if best is None else best.c(), curvature_note=note)

    def _curvature_note(self, matched: list[RaceReceipt], control_c: float) -> str:
        if not matched:
            return "no matched-bytes arms — curvature UNMEASURED (Assumption-Adversary line stands)"
        cs = [r.c() for r in matched]
        spread = max(cs) - min(cs)
        if min(cs) < self.iso_c * (1 - self.rel_band):
            return (f"hull MOVED: {min(cs):.5f} < iso_c {self.iso_c:.5f}; the contour is a "
                    "true convex hull with interior points, not a line")
        if spread < self.rel_band * self.iso_c:
            return (f"all {len(matched)} matched points within ±{self.rel_band:.0%} of iso_c — "
                    "the 3-move negative record EXTENDS to the class (hull is a line here)")
        return (f"matched points span {spread:.5f} around iso_c but none strictly inside — "
                "mixed; the min point is on/above the contour")


def analyze_race_json(receipts_json: str, *, iso_c: float = HULL_ISO_C) -> dict[str, object]:
    """Analyzer entry for a JSON list of receipt dicts (MAIN pipes the measured race here)."""
    rows = json.loads(receipts_json)
    receipts = [RaceReceipt(
        lever=r["lever"], window=r.get("window", ""), d_seg=float(r["d_seg"]),
        counted_bytes=int(r["counted_bytes"]),
        delta_d_seg_vs_control=float(r.get("delta_d_seg_vs_control", 0.0)),
        delta_bytes_vs_control=int(r.get("delta_bytes_vs_control", 0)),
        within_matched_tol=bool(r.get("within_matched_tol", True))) for r in rows]
    return HullCurvatureAnalyzer(iso_c).analyze(receipts).to_dict()


__all__ = [
    "HULL_ISO_C",
    "MATCHED_BYTES_TOL",
    "ArmByteRecord",
    "BandEdgeConfig",
    "BandEdgeTheorem",
    "HullCurvatureAnalyzer",
    "HullCurvatureVerdict",
    "RaceReceipt",
    "SealVerdict",
    "analyze_race_json",
    "enumerate_band_edge_theorem",
    "per_row_flip_mass_from_field",
    "seal_matched_bytes_race",
    "smevr_bytes_of_field",
]
