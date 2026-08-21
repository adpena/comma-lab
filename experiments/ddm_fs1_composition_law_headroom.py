"""ddm_fs1 -- price the composition law's remaining headroom on the LIVE rc2 body.

Two scorer-free, $0 questions, both answered from RETAINED n600 payloads:

1. POSE-ACTUATOR HEADROOM.  ``ddm_jg5`` measured, at n600 on the live vehicle,
   that re-solving the pose carrier against a pair's OWN edited render is a pose
   CREDIT for most edited pairs (the "edits are a pose actuator" law).  145 of
   600 pairs never banked that credit: 27 were never edited and 118 were edited
   but dropped by the joint waterfill.  This module derives, from jg5's retained
   per-pair d_pose arrays, the per-pair break-even BYTE BUDGET an edit must fit
   inside to pay for itself on the pose axis alone, and compares it against every
   edit-encoding cost this vehicle has actually measured.

2. JS6B COMPENSATED RE-SCREEN (``ddm_na10`` reopened row 5).  The ``ddm_js6b``
   200-row bank was closed at FORMULATION scope against an UNCOMPENSATED pose
   envelope -- measured before a carrier re-solve existed.  This module re-screens
   the same sealed rows under a compensation factor DERIVED from jg5's per-pair
   distribution, plus the real ``qs2`` rate cost that js6b optimistically set to
   zero, and reports each row's break-even compensation factor.

Both are advisory arithmetic over retained arrays.  No scorer forward, no Modal
dispatch, no archive is built here.  A re-screen can only REOPEN a closure; it can
never ADMIT a candidate -- admission needs a real byte-closed measurement.

Axis: ``[macOS-CPU advisory, scorer-free retained-array arithmetic]``.
``score_claim=false``  ``promotion_eligible=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --- contest constants (CLAUDE.md, upstream/evaluate.py:63) -------------------
RATE_DENOMINATOR_BYTES = 37_545_489
RATE_WEIGHT = 25.0
S_PER_BYTE = RATE_WEIGHT / RATE_DENOMINATOR_BYTES
N_PAIRS = 600

# --- the live pointer body (ddm_rc2 sixteenth move) ---------------------------
LIVE_ARCHIVE_SHA256 = "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080"
LIVE_ARCHIVE_BYTES = 180_456

# --- retained inputs, content-pinned ------------------------------------------
JG5_FINAL = Path("/Volumes/APDataStore/pact/ddm_jg5/retained/final")
JS6B_STORE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js6b_pose_screened_compile_20260813"
)

#: Every retained input is content-hashed AT READ TIME and the measured digest is
#: recorded in the result JSON (see ``load_jg5_per_pair``).  No expected-digest
#: table is hardcoded here: a hand-copied hash is a second authority that drifts
#: from the store it claims to pin, and the store is the authority.

#: Edit-encoding costs MEASURED on this vehicle, in archive bytes per edited pair.
#: Each is a real receipt, not a model.
MEASURED_EDIT_ENCODING_COSTS_B_PER_PAIR: dict[str, float] = {
    # ddm_jg5 §6.2: br1 pointer body 176,429 B -> jg5 subset body 180,580 B
    # carrying 455 admitted edits.  (180580 - 176429) / 455.
    "jg5_seg_edits_live_vehicle": (180_580 - 176_429) / 455.0,
    # ddm_na10 line 562: "It paid 12.83 B/pair; qs2 got 5.667; jg1's re-solve
    # moves 9-12".  qs2 is the cheapest encoding this vehicle has measured.
    "qs2_cheapest_measured": 5.667,
    "jg1_resolve_midpoint": 10.5,
    "rc4_rung4_reference": 12.83,
}


def sha256_of(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hex, byte_length)`` for ``path``."""
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def marginal_dS_per_pair_d_pose(mean_d_pose: float) -> float:
    """Exact derivative of the contest pose leg w.r.t. ONE pair's ``d_pose``.

    The contest pose term is ``sqrt(10 * mean(d_i))`` over ``N_PAIRS`` pairs, so

        d/d(d_i) sqrt(10 * (1/N) * sum d_j) = 10 / (2 * N * sqrt(10 * mean))

    This is ``ddm_jg5`` §4's ``dS/dd_i``, re-derived here at whatever operating
    point the caller is actually standing on rather than transferred as a
    constant -- the cross-regime-constant-transfer genus is why jg5 wrote it out.
    """
    if mean_d_pose <= 0.0:
        raise ValueError(f"mean_d_pose must be positive, got {mean_d_pose!r}")
    return 10.0 / (2.0 * N_PAIRS * np.sqrt(10.0 * mean_d_pose))


def pose_leg(mean_d_pose: float) -> float:
    """The contest score's pose contribution ``sqrt(10 * d_pose)``."""
    return float(np.sqrt(10.0 * mean_d_pose))


@dataclass(frozen=True)
class Jg5PerPair:
    """jg5's retained n600 per-pair d_pose arrays, with their content hashes."""

    base: np.ndarray
    """Per-pair d_pose with the BASE (unedited) odd frame and br1's carrier."""
    candidate: np.ndarray
    """Per-pair d_pose with the EDITED odd frame and br1's STALE carrier."""
    refined: np.ndarray
    """Per-pair d_pose with the EDITED odd frame and this arm's RE-SOLVED carrier."""
    kept_pairs: tuple[int, ...]
    """The 455 pair indices the jg5 joint waterfill admitted."""
    input_digests: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def edited(self) -> np.ndarray:
        """Boolean mask of pairs jg3 proposed an edit for (573 of 600)."""
        return ~np.isclose(self.candidate, self.base, rtol=0.0, atol=0.0)

    @property
    def shipped(self) -> np.ndarray:
        """Per-pair d_pose actually shipped: kept pairs re-solved, dropped at base.

        ``ddm_jg5`` §2 proved by decode that a dropped pair ships exactly the base
        frame 1, so pricing it at its base pose value is a measurement.
        """
        keep = np.isin(np.arange(len(self.base)), np.asarray(self.kept_pairs))
        return np.where(keep, self.refined, self.base)

    @property
    def dropped_pairs(self) -> tuple[int, ...]:
        """Edited pairs the waterfill rejected (118)."""
        edited = set(np.nonzero(self.edited)[0].tolist())
        return tuple(sorted(edited - set(self.kept_pairs)))

    @property
    def unedited_pairs(self) -> tuple[int, ...]:
        """Pairs jg3 never proposed an edit for (27)."""
        edited = set(np.nonzero(self.edited)[0].tolist())
        return tuple(sorted(set(range(len(self.base))) - edited))


def load_jg5_per_pair(root: Path = JG5_FINAL) -> Jg5PerPair:
    """Load and content-hash jg5's retained per-pair arrays."""
    names = {
        "base": "d_pose_per_pair_base_odd_frames.npy",
        "candidate": "d_pose_per_pair_candidate.npy",
        "refined": "d_pose_per_pair_refined_matched.npy",
    }
    arrays: dict[str, np.ndarray] = {}
    digests: dict[str, dict[str, Any]] = {}
    for key, name in names.items():
        path = root / name
        digest, nbytes = sha256_of(path)
        arrays[key] = np.load(path)
        digests[name] = {"path": str(path), "bytes": nbytes, "sha256": digest}
    kept_path = root / "kept_pairs.json"
    kept_digest, kept_bytes = sha256_of(kept_path)
    digests["kept_pairs.json"] = {
        "path": str(kept_path),
        "bytes": kept_bytes,
        "sha256": kept_digest,
    }
    kept = tuple(int(p) for p in json.loads(kept_path.read_text()))
    return Jg5PerPair(
        base=arrays["base"],
        candidate=arrays["candidate"],
        refined=arrays["refined"],
        kept_pairs=kept,
        input_digests=digests,
    )


def compensation_distribution(jg5: Jg5PerPair) -> dict[str, Any]:
    """Derive the per-pair carrier-compensation factor from jg5's own n600 rows.

    For each edited pair, ``c_i = (candidate_i - base_i) / (refined_i - base_i)``:
    the uncompensated edit-induced pose damage divided by what survives the
    carrier re-solve.  Pairs whose re-solved value lands at or BELOW base carry
    non-positive compensated damage -- the edit is a net pose CREDIT and no finite
    factor describes it, so they are counted separately rather than folded into a
    percentile that they would silently dominate.
    """
    edited = jg5.edited
    uncompensated = jg5.candidate[edited] - jg5.base[edited]
    compensated = jg5.refined[edited] - jg5.base[edited]
    positive = compensated > 0.0
    factors = uncompensated[positive] / compensated[positive]
    pct = np.percentile(factors, [5, 25, 50, 75, 95]) if factors.size else np.zeros(5)
    return {
        "edited_pairs": int(edited.sum()),
        "pairs_landing_at_or_below_base": int((~positive).sum()),
        "fraction_at_or_below_base": float((~positive).sum() / max(edited.sum(), 1)),
        "pairs_with_residual_damage": int(positive.sum()),
        "aggregate_factor_sum_over_sum": (
            float(uncompensated.sum() / compensated.sum())
            if compensated.sum() != 0
            else None
        ),
        "residual_damage_factor_p5": float(pct[0]),
        "residual_damage_factor_p25": float(pct[1]),
        "residual_damage_factor_median": float(pct[2]),
        "residual_damage_factor_p75": float(pct[3]),
        "residual_damage_factor_p95": float(pct[4]),
        "note": (
            "Cross-regime bracket, NOT a transferable constant. Measured on "
            "jg3-derived seg edits on the br1/rc2 body. Using it on a different "
            "edit family is the cross-regime-constant-transfer genus; it is "
            "reported to test whether a closure's MARGIN survives, never to "
            "admit a candidate."
        ),
    }


def pose_actuator_break_even(jg5: Jg5PerPair) -> dict[str, Any]:
    """Derive the byte budget a pose-targeted edit must fit inside to pay off.

    An edit that buys a per-pair pose credit ``k`` is worth ``k * dS/dd_i`` in
    score units and costs ``B * S_PER_BYTE``.  It breaks even at
    ``B = k * (dS/dd_i) / S_PER_BYTE``.  Comparing that budget against the edit
    encodings this vehicle has actually measured decides the family at $0.
    """
    shipped = jg5.shipped
    mean_shipped = float(shipped.mean())
    slope = marginal_dS_per_pair_d_pose(mean_shipped)

    credits = (jg5.base - jg5.refined)[np.asarray(jg5.kept_pairs, dtype=np.intp)]
    credits = credits[credits > 0.0]
    if credits.size == 0:
        # Percentiles of an empty set are NaN, and a NaN byte budget reads like a
        # measurement. Refuse instead of emitting one.
        raise ValueError(
            "no kept pair carries a positive pose credit; the break-even budget "
            "is undefined and must not be reported as NaN"
        )

    budgets: dict[str, Any] = {}
    for label, value in (
        ("p25", float(np.percentile(credits, 25))),
        ("median", float(np.median(credits))),
        ("mean", float(credits.mean())),
        ("p75", float(np.percentile(credits, 75))),
        ("p90", float(np.percentile(credits, 90))),
    ):
        budgets[label] = {
            "per_pair_pose_credit": value,
            "break_even_bytes_per_pair": float(value * slope / S_PER_BYTE),
        }

    encodings: dict[str, Any] = {}
    for label, bytes_per_pair in MEASURED_EDIT_ENCODING_COSTS_B_PER_PAIR.items():
        required = bytes_per_pair * S_PER_BYTE / slope
        qualifying = int((credits >= required).sum())
        encodings[label] = {
            "bytes_per_pair": bytes_per_pair,
            "required_per_pair_pose_credit": float(required),
            "qualifying_credit_pairs": qualifying,
            "credit_pair_population": int(credits.size),
            "qualifying_fraction": float(qualifying / max(credits.size, 1)),
            "shortfall_vs_median_budget": float(
                bytes_per_pair / budgets["median"]["break_even_bytes_per_pair"]
            ),
        }

    return {
        "live_shipped_d_pose": mean_shipped,
        "live_pose_leg": pose_leg(mean_shipped),
        "dS_per_pair_d_pose": float(slope),
        "s_per_archive_byte": S_PER_BYTE,
        "credit_pair_population": int(credits.size),
        "break_even_budgets": budgets,
        "measured_encodings": encodings,
        "unbanked_pairs": {
            "unedited": len(jg5.unedited_pairs),
            "dropped": len(jg5.dropped_pairs),
            "total": len(jg5.unedited_pairs) + len(jg5.dropped_pairs),
        },
    }


def unbanked_population_validity(jg5: Jg5PerPair) -> dict[str, Any]:
    """Test whether the kept-pair credit distribution is a legal prior for the unbanked.

    The tempting move is to price a pose-targeted edit on the 145 unbanked pairs
    using the credit distribution of the 355 pairs the waterfill kept.  That prior
    is drawn from a population **selected for being credits**, so this function
    measures the unbanked population directly and reports whether the transfer is
    admissible.  It is the ``measured_object_vs_named_object`` check applied to
    this module's own arithmetic: an estimate that names "145 unbanked pairs"
    while measuring "355 selected credit pairs" is the dominant false-verdict
    genus, and it must fail loudly rather than round to a headline.
    """
    dropped = np.asarray(jg5.dropped_pairs, dtype=np.intp)
    unedited = np.asarray(jg5.unedited_pairs, dtype=np.intp)
    measured_credit = jg5.base - jg5.refined

    dropped_credit = measured_credit[dropped]
    unedited_move = np.abs(jg5.refined[unedited] - jg5.base[unedited])

    all_dropped_are_costs = bool((dropped_credit <= 0.0).all())
    # An empty unedited set is "nothing unmeasured", not "movement of zero".
    unedited_unmeasured = float(unedited_move.max()) if unedited_move.size else 0.0

    return {
        "unbanked_total": int(dropped.size + unedited.size),
        "dropped_with_measured_pose_cost": int((dropped_credit <= 0.0).sum()),
        "dropped_population": int(dropped.size),
        "all_dropped_pairs_measure_as_pose_cost": all_dropped_are_costs,
        "dropped_median_measured_cost": float(np.median(-dropped_credit)),
        "unedited_population": int(unedited.size),
        "unedited_max_abs_resolve_movement": unedited_unmeasured,
        "unedited_have_no_edit_measurement": bool(unedited_unmeasured < 1e-9),
        "kept_pair_prior_is_admissible_for_unbanked": False,
        "verdict": (
            "REFUSED. The kept-pair credit distribution may NOT be transferred to "
            "the unbanked population. Of the 145 unbanked pairs, the 118 dropped "
            "ones each MEASURE a pose cost under jg3's edit family -- they are not "
            "unknowns, they are measured negatives -- and the 27 unedited ones "
            "carry no edit and so no re-solve measurement at all. Any per-pair "
            "credit assumed for these pairs is a HYPOTHESIS about a different, "
            "milder edit, and must be labelled as one."
        ),
    }


def pose_actuator_ceiling(jg5: Jg5PerPair) -> dict[str, Any]:
    """Upper bound on what a pose-targeted edit could buy on the 145 unbanked pairs.

    Strictly a CEILING (the `ra3` price-the-ceiling law): it assumes a hypothetical
    edit on an unbanked pair reproduces the credit distribution of the pairs the
    waterfill already kept.  For the 118 dropped pairs that assumption is
    contradicted by their own measured row -- under jg3's edits every one of them
    is a pose COST -- so the ceiling requires a DIFFERENT, milder edit to exist.
    For the 27 unedited pairs there is no measurement at all.  Rate is excluded
    here on purpose; ``pose_actuator_break_even`` is where rate binds.
    """
    shipped = jg5.shipped
    baseline_leg = pose_leg(float(shipped.mean()))
    credits = (jg5.base - jg5.refined)[np.asarray(jg5.kept_pairs, dtype=np.intp)]
    credits = credits[credits > 0.0]
    if credits.size == 0:
        # Percentiles of an empty set are NaN, and a NaN byte budget reads like a
        # measurement. Refuse instead of emitting one.
        raise ValueError(
            "no kept pair carries a positive pose credit; the break-even budget "
            "is undefined and must not be reported as NaN"
        )

    groups = {
        "unedited_27": np.asarray(jg5.unedited_pairs, dtype=np.intp),
        "dropped_118": np.asarray(jg5.dropped_pairs, dtype=np.intp),
        "both_145": np.asarray(jg5.unedited_pairs + jg5.dropped_pairs, dtype=np.intp),
    }
    stats = {"median": float(np.median(credits)), "mean": float(credits.mean())}

    out: dict[str, Any] = {
        "baseline_pose_leg": baseline_leg,
        "gross_of_rate": True,
        "groups": {},
    }
    for gname, idx in groups.items():
        row: dict[str, Any] = {"pair_count": int(idx.size)}
        for sname, credit in stats.items():
            trial = shipped.copy()
            trial[idx] = np.maximum(trial[idx] - credit, 0.0)
            row[sname] = {
                "assumed_per_pair_credit": credit,
                "d_pose": float(trial.mean()),
                "delta_S_pose": float(pose_leg(float(trial.mean())) - baseline_leg),
            }
        out["groups"][gname] = row

    floor = np.minimum(np.minimum(jg5.base, jg5.refined), shipped)
    out["absolute_floor_every_pair_at_min_base_or_refined"] = {
        "d_pose": float(floor.mean()),
        "delta_S_pose": float(pose_leg(float(floor.mean())) - baseline_leg),
    }
    return out


def js6b_compensated_rescreen(
    screen_rows_path: Path,
    bytes_per_pair: float,
    compensation_factors: tuple[float, ...],
) -> dict[str, Any]:
    """Re-screen the sealed js6b bank under a compensated pose envelope + real rate.

    ``ddm_js6b`` screened each row as ``-seg + pose_risk + 0`` and held all 200.
    Two of those three terms were wrong in a KNOWN direction: the pose envelope was
    measured with a stale carrier (too high), and the rate was set to zero (too
    low).  This recomputes ``-seg + pose_risk / c + rate`` and, for every row,
    solves for the compensation factor at which it would first admit.

    A row admitting here is NOT a candidate.  js6b's seg term is itself an
    optimistic upper bound (every target pixel credited as a successful flip), so
    this screen can only refuse a refusal -- it reopens, it never admits.
    """
    digest, nbytes = sha256_of(screen_rows_path)
    rows = [json.loads(line) for line in screen_rows_path.read_text().splitlines() if line]
    rate_cost_s = bytes_per_pair * S_PER_BYTE

    per_row: list[dict[str, Any]] = []
    for row in rows:
        screen = row["screen"]
        seg = float(screen["optimistic_seg_value_s"])
        headroom = seg - rate_cost_s
        entry: dict[str, Any] = {
            "proposal_id": row["proposal_id"],
            "pair": row["pair"],
            "semantic_cell_count": row["semantic_cell_count"],
            "optimistic_seg_value_s": seg,
            "rate_cost_s": rate_cost_s,
            "seg_headroom_after_rate_s": headroom,
        }
        for bound in ("lower", "upper"):
            risk = float(screen[f"measured_pose_risk_{bound}_s"])
            entry[f"pose_risk_{bound}_s"] = risk
            # break-even c solves  risk / c == headroom.
            entry[f"break_even_compensation_{bound}"] = (
                float(risk / headroom) if headroom > 0.0 else None
            )
        per_row.append(entry)

    distinct_pairs = len({r["pair"] for r in rows})
    ceiling_seg = sum(r["optimistic_seg_value_s"] for r in per_row)
    ceiling_rate = distinct_pairs * rate_cost_s

    sweep: dict[str, Any] = {}
    # Deduplicate: two equal factors would format to the same key and the second
    # would silently overwrite the first, losing a sweep row without a warning.
    seen: set[str] = set()
    for c in compensation_factors:
        if f"c={c:g}" in seen:
            continue
        seen.add(f"c={c:g}")
        admitted = {"lower": [], "upper": []}
        for entry in per_row:
            for bound in ("lower", "upper"):
                net = (
                    -entry["optimistic_seg_value_s"]
                    + entry[f"pose_risk_{bound}_s"] / c
                    + rate_cost_s
                )
                if net < 0.0:
                    admitted[bound].append((entry["proposal_id"], net))
        sweep[f"c={c:g}"] = {
            bound: {
                "admitted_rows": len(admitted[bound]),
                "best_net_delta_s": (
                    min(n for _, n in admitted[bound]) if admitted[bound] else None
                ),
                "summed_net_delta_s_independent_rows": (
                    float(sum(n for _, n in admitted[bound]))
                    if admitted[bound]
                    else 0.0
                ),
            }
            for bound in ("lower", "upper")
        }

    return {
        "input": {
            "path": str(screen_rows_path),
            "bytes": nbytes,
            "sha256": digest,
            "row_count": len(rows),
        },
        "rate_model": {
            "bytes_per_pair": bytes_per_pair,
            "rate_cost_s_per_pair": rate_cost_s,
            "provenance": "ddm_na10 line 562 -- qs2's measured 5.667 B/pair",
        },
        "bank_ceiling": {
            "summed_optimistic_seg_credit_s": ceiling_seg,
            "distinct_pairs": distinct_pairs,
            "rate_cost_all_pairs_s": ceiling_rate,
            "ceiling_net_s_zero_pose": ceiling_seg - ceiling_rate,
            "note": (
                "Optimistic on BOTH the seg term (every target pixel credited as a "
                "flip) and the pose term (zero). A ceiling on a ceiling; it bounds "
                "the family, it does not predict it."
            ),
        },
        "compensation_sweep": sweep,
        "per_row": per_row,
        "admission_boundary": (
            "This screen can REOPEN a closure. It cannot ADMIT a candidate: the seg "
            "term is an optimistic upper bound and no archive, receiver round-trip, "
            "or scorer forward is measured here."
        ),
    }


def build_result(
    jg5: Jg5PerPair,
    js6b_rows: Path | None,
    bytes_per_pair: float,
) -> dict[str, Any]:
    """Assemble the full advisory result."""
    comp = compensation_distribution(jg5)
    factors = (
        1.0,
        comp["aggregate_factor_sum_over_sum"] or 1.0,
        comp["residual_damage_factor_median"],
    )
    result: dict[str, Any] = {
        "schema": "ddm_fs1_composition_law_headroom.v1",
        "arm": "ddm_fs1",
        "axis": "[macOS-CPU advisory, scorer-free retained-array arithmetic]",
        "score_claim": False,
        "promotion_eligible": False,
        "scorer_run": False,
        "modal_dispatch": False,
        "live_pointer": {
            "archive_sha256": LIVE_ARCHIVE_SHA256,
            "archive_bytes": LIVE_ARCHIVE_BYTES,
            "note": "ddm_rc2 sixteenth move; S recomputed from components per Catalog #877.",
        },
        "inputs": jg5.input_digests,
        "carrier_compensation_distribution": comp,
        "pose_actuator_break_even": pose_actuator_break_even(jg5),
        "unbanked_population_validity": unbanked_population_validity(jg5),
        "pose_actuator_ceiling": pose_actuator_ceiling(jg5),
    }
    if js6b_rows is not None:
        result["js6b_compensated_rescreen"] = js6b_compensated_rescreen(
            js6b_rows, bytes_per_pair, factors
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jg5-final",
        type=Path,
        default=JG5_FINAL,
        help="jg5 retained final directory holding the per-pair d_pose arrays.",
    )
    parser.add_argument(
        "--js6b-screen-rows",
        type=Path,
        default=JS6B_STORE / "POSE_SCREEN.jsonl",
        help="Sealed js6b POSE_SCREEN.jsonl. Pass an absent path to skip leg 2.",
    )
    parser.add_argument(
        "--bytes-per-pair",
        type=float,
        default=MEASURED_EDIT_ENCODING_COSTS_B_PER_PAIR["qs2_cheapest_measured"],
        help="Archive bytes charged per edited pair in the js6b re-screen.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_fs1/retained"),
        help="Retention directory. Every materialized array is persisted here.",
    )
    args = parser.parse_args(argv)

    jg5 = load_jg5_per_pair(args.jg5_final)
    rows_path = (
        args.js6b_screen_rows if args.js6b_screen_rows.exists() else None
    )
    result = build_result(jg5, rows_path, args.bytes_per_pair)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ALWAYS KEEP THE PAYLOAD: persist every array this run derived, not just the
    # scalars computed from them.
    derived = {
        "shipped_d_pose_per_pair.npy": jg5.shipped,
        "carrier_credit_per_pair.npy": jg5.base - jg5.refined,
        "uncompensated_damage_per_pair.npy": jg5.candidate - jg5.base,
        "compensated_damage_per_pair.npy": jg5.refined - jg5.base,
    }
    retained: dict[str, Any] = {}
    for name, array in derived.items():
        path = args.out_dir / name
        np.save(path, array)
        digest, nbytes = sha256_of(path)
        retained[name] = {"path": str(path), "bytes": nbytes, "sha256": digest}
    result["retained_payloads"] = retained
    result["unbanked_pair_indices"] = {
        "unedited": list(jg5.unedited_pairs),
        "dropped": list(jg5.dropped_pairs),
    }

    out_path = args.out_dir / "FS1_COMPOSITION_LAW_HEADROOM.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    digest, nbytes = sha256_of(out_path)
    print(f"wrote {out_path} ({nbytes} B, sha256 {digest})")

    be = result["pose_actuator_break_even"]
    print(
        "\nPOSE-ACTUATOR BREAK-EVEN: median credit needs an edit encoding of "
        f"{be['break_even_budgets']['median']['break_even_bytes_per_pair']:.3f} B/pair"
    )
    for label, row in be["measured_encodings"].items():
        print(
            f"  {label:34s} {row['bytes_per_pair']:6.3f} B/pair -> "
            f"{row['qualifying_fraction'] * 100:5.1f}% of credit pairs pay "
            f"({row['shortfall_vs_median_budget']:.2f}x over the median budget)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
