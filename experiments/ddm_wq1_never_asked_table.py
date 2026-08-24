"""ddm_wq1 — the machine-readable never-asked table.

Deterministic, scorer-free, $0. Emits the probe-shape characterization, the
per-direction status enumeration, and the ranked unasked table to the SSD tier
with sha256 + bytes, per the ALWAYS-KEEP-THE-PAYLOAD non-negotiable.

Every number here is either (a) recomputed from the dx2 authority components,
(b) quoted from a cited receipt with that receipt's own axis label, or
(c) exact arithmetic over those, labelled DERIVED.

Exchange rate 6.658590e-07 S/B is CITED from ddm_tx1_toolbox_crosswalk_20260819.md
section 0, never re-derived (#1207).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

# ---------------------------------------------------------------- dx2 authority
DX2_ARCHIVE_SHA = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
DX2_BYTES = 180_368
DX2_D_SEG = 0.00020139
DX2_D_POSE = 0.00000637
RATE_DENOM = 37_545_489
EXCHANGE_S_PER_BYTE = 6.658590e-07  # CITED tx1 section 0 — NOT re-derived
TARGET_S = 0.12

OUT_DIR = Path("/Volumes/APDataStore/pact/ddm_wq1_never_asked")


def dx2_components() -> dict:
    """S recomputed FROM COMPONENTS, never from a rounded display (#877)."""
    rate = 25.0 * DX2_BYTES / RATE_DENOM
    seg = 100.0 * DX2_D_SEG
    pose = math.sqrt(10.0 * DX2_D_POSE)
    s = rate + seg + pose
    return {
        "archive_bytes": DX2_BYTES,
        "archive_sha256": DX2_ARCHIVE_SHA,
        "d_seg": DX2_D_SEG,
        "d_pose": DX2_D_POSE,
        "rate_term": rate,
        "seg_term": seg,
        "pose_term": pose,
        "S": s,
        "gap_to_0_12": s - TARGET_S,
        "share_rate": rate / s,
        "share_seg": seg / s,
        "share_pose": pose / s,
        "share_not_in_training_objective": (rate + pose) / s,
        "axis": "[contest-CUDA T4, n600]",
    }


def probe_shape() -> dict:
    """The shape ~20 arms share, stated as a constraint on the reachable
    hypothesis space. Each leg cites the evidence that establishes it."""
    return {
        "statement": (
            "Take the CONVERGED dx2 object; hold its TRAINING OBJECTIVE fixed; "
            "perturb one section, parameter, ordering or address POST-HOC; "
            "byte-close; measure. The four arms that did change the object were "
            "still required to reproduce dx2's exact 117,964,800-position "
            "categorical field, so the exact-reproduction residual dominated them."
        ),
        "legs": [
            {
                "leg": "POST-HOC",
                "claim": (
                    "No arm re-trains under a changed objective. The live trainer "
                    "src/tac/pr130_lift/train_semantic_quantized_resumable.py has a "
                    "distortion-only loss: cross-entropy / band-weighted CE, with "
                    "fixed-lattice int4 QAT. Zero rate-penalty sites; no PoseNet "
                    "term. Quantization is rate-AWARE only in the sense that the "
                    "lattice is fixed; packed size is MEASURED post-hoc, never "
                    "differentiated."
                ),
                "evidence_label": "MEASURED (source read)",
                "evidence": "src/tac/pr130_lift/train_semantic_quantized_resumable.py:951,1240-1259,1446",
            },
            {
                "leg": "SINGLE-OBJECT",
                "claim": (
                    "The comparison is dx2 vs dx2-perturbed. The whole-body "
                    "replacements (rc1/ri1/nr1/ni1) failed on DISTORTION "
                    "(S=17.31 / 27.80), a different failure mode from the "
                    "250-800x rate-vs-distortion refusals."
                ),
                "evidence_label": "MEASURED",
                "evidence": "ddm_sy2_composition_synergy_deep_pass_20260823.md closed-family inventory",
            },
            {
                "leg": "EXACT-FIELD-REPRODUCTION",
                "claim": (
                    "Every alternative-representation arm priced EXACT reproduction "
                    "of the dx2 categorical field. The common killer is the "
                    "unique-home categorical residual (359,280 B in hg1, 362,473 B "
                    "in hr3), not the generator. The contest requires "
                    "evaluator-cell equivalence, not token identity."
                ),
                "evidence_label": "MEASURED",
                "evidence": "ddm_hg1(460,408 B) / ddm_et1(535,761 B) / ddm_ws1(918,904 B) / ddm_hr3(463,601 B)",
            },
        ],
        "why_it_produces_250_to_800x": {
            "derivation": (
                "dx2 sits at a stationary point of a loss containing ONLY the seg "
                "term. Rate is 81.03% of shipped S and pose is 5.38%; together "
                "86.41% of the score is produced by terms the optimizer never saw. "
                "A post-hoc perturbation that buys rate moves off the seg minimum, "
                "and pose -- which nothing protected -- absorbs the damage."
            ),
            "predicted_signature": "damage should be pose-dominated",
            "measured_signature": [
                {"arm": "dg2 k060000", "pose_share_of_damage": 0.933, "ratio_to_credit": 687.3},
                {"arm": "dg2 k040000", "pose_share_of_damage": 0.934, "ratio_to_credit": 791.7},
                {"arm": "w72 renderer", "pose_share_of_damage": 0.653, "d_pose_multiplier": 303989},
                {"arm": "ap1 carrier L1-L3", "pose_share_of_damage": 1.0,
                 "note": "SegNet-inert at all 3 levels: 0 flips in all 5 classes"},
            ],
            "label": "DERIVED prediction, MEASURED signature — they agree",
        },
        "marginal_asymmetry": None,  # filled in below
    }


def marginals(c: dict) -> dict:
    """d(seg term)/d(d_seg) is constant at 100; d(pose term)/d(d_pose) diverges
    as pose improves. At dx2's operating point pose is the expensive axis."""
    d_seg_marginal = 100.0
    d_pose_marginal = 5.0 / math.sqrt(10.0 * DX2_D_POSE)
    crossover_d_pose = 2.5e-4  # where the two marginals are equal (100 = 5/sqrt(10 d))
    breakeven_slope = EXCHANGE_S_PER_BYTE / d_pose_marginal
    # ap1 carrier level 1 secant: -2,742 B costs +0.00987277665396 d_pose
    ap1_secant = 0.00987277665396 / 2742.0
    return {
        "d_seg_term_per_unit_d_seg": d_seg_marginal,
        "d_pose_term_per_unit_d_pose": d_pose_marginal,
        "pose_over_seg_marginal_ratio": d_pose_marginal / d_seg_marginal,
        "crossover_d_pose": crossover_d_pose,
        "dx2_is_below_crossover_by": crossover_d_pose / DX2_D_POSE,
        "carrier_breakeven_d_pose_per_byte": breakeven_slope,
        "ap1_carrier_L1_secant_d_pose_per_byte": ap1_secant,
        "ap1_secant_over_breakeven": ap1_secant / breakeven_slope,
        "caveat": (
            "The ap1 figure is a SECANT over 2,742 B on a convex response "
            "(L1/L2/L3 S-per-byte 1.117e-4 / 1.245e-4 / 4.348e-4). A secant on a "
            "convex curve OVERSTATES the local slope, so 'ap1_secant_over_breakeven' "
            "is an UPPER bound on how far the shipped carrier sits from its own "
            "rate-distortion optimum. Label: DERIVED-FROM-SECANT."
        ),
        "max_possible_pose_saving_bytes": c["pose_term"] / EXCHANGE_S_PER_BYTE,
    }


def cross_regime_correction(c: dict) -> dict:
    """ddm_tc1 (2026-08-17) marked --w-rate and --rate-model smevr_surrogate
    DOMINATED via its section 1.2. That derivation was computed on TR1 rows whose
    best-ever d_seg is 0.00389011. Applying tc1's own method to dx2 gives a
    completely different answer."""
    tr1_best_d_seg = 0.00389011
    tc1_budget = 0.15959729 - 0.0082946  # tc1's own sub-frontier seg+rate budget
    tc1_best_row_segrate = 0.57767
    dx2_budget = TARGET_S - c["pose_term"]
    dx2_segrate = c["seg_term"] + c["rate_term"]
    return {
        "tc1_verdict_quoted": "--w-rate NEVER A/B'd ... DOMINATED (section 1.2)",
        "tc1_source": "ddm_tc1_tr1_lifecycle_spec_20260817.md:196,197,264",
        "tc1_object_best_d_seg": tr1_best_d_seg,
        "dx2_d_seg": DX2_D_SEG,
        "d_seg_ratio_tc1_object_over_dx2": tr1_best_d_seg / DX2_D_SEG,
        "tc1_x_budget_best_row": tc1_best_row_segrate / tc1_budget,
        "dx2_x_budget": dx2_segrate / dx2_budget,
        "tc1_free_archive_check": "S = 0.397306 = 2.489x its own frontier",
        "dx2_free_archive_check_S": c["seg_term"] + c["pose_term"],
        "dx2_free_archive_x_of_0_12": (c["seg_term"] + c["pose_term"]) / TARGET_S,
        "verdict": (
            "tc1's dominance derivation does not transfer. Its binding premise -- "
            "'a zero-byte archive does not save TR1 while d_seg sits at 3.89e-3' -- "
            "is TRUE for TR1 and FALSE for dx2, whose zero-byte archive lands at "
            "0.2343x of the 0.12 target. Carrying the DOMINATED label onto dx2 is "
            "the cross-regime constant-transfer genus."
        ),
        "label": "DERIVED (exact arithmetic over two cited measured rows)",
        "caveat": (
            "TR1 is additionally a RETIRED vehicle (tc1 point 5) and the live "
            "trainer is pr130_lift. So --w-rate is not a flag flip on the live "
            "vehicle; rate-in-loss there is a BUILD, not a config change."
        ),
    }


def directions(c: dict) -> list[dict]:
    demand = DX2_BYTES - 137_986
    return [
        {
            "id": "D1",
            "name": "Rate absent from the LIVE training objective",
            "status": "UNASKED on the live vehicle; NAMED-NEVER-RACED on the retired one",
            "reach_bytes_upper": demand,
            "reach_basis": (
                "Rate is 81.03% of shipped S. Putting it in the loss changes the "
                "object, which is the leg sy2 requires and fb1's >=2-axis "
                "arithmetic demands. Ceiling is the whole demand because the "
                "objective moves rate AND distortion jointly."
            ),
            "evidence_live_vehicle": (
                "src/tac/pr130_lift/train_semantic_quantized_resumable.py -- full grep "
                "for w_rate|rate_loss|rate_term|lambda_rate|entropy|rate_model returns "
                "ZERO rate-penalty sites. Loss is F.cross_entropy / band-weighted CE. "
                "--bits hard-refused off 4 (int4-only packer); packed_size is MEASURED."
            ),
            "evidence_retired_vehicle": (
                "TR1 train_tr1_partition_renderer_mlx.py:5256-5257 adds "
                "w_rate*token_rate_term to the loss, w_rate=0.05 across the whole "
                "lineage. rsf1 MEASURED the live 'entropy' surrogate ANTI-correlated "
                "with shipped bytes (rho=-0.7235, CI [-0.943,-0.227]); permutation "
                "moves the surrogate <=4.8e-07 bits and real bytes +16,062..+18,339 B. "
                "rg5 MEASURED smevr_surrogate 1.5-3.6x stronger byte-descent with "
                "cos(entropy,smevr) in [-0.066,+0.020] -- NEARLY ORTHOGONAL. "
                "rate_model='entropy' in 29/29 landed configs; smevr_surrogate NEVER "
                "FIRED; the sum arm does not exist."
            ),
            "why_not_already_closed": (
                "tc1 marked it DOMINATED on a 19.316x-worse-d_seg object -- see the "
                "cross_regime_correction block."
            ),
            "cost_to_falsify": "one training run on the live trainer + byte-close + one advisory n600 row",
            "labels": {"live_vehicle_absence": "MEASURED", "reach": "DERIVED"},
        },
        {
            "id": "D2",
            "name": "Pose absent from the LIVE training objective",
            "status": "UNASKED on the live vehicle; TR1's 15 jd1 pose flags NEVER FIRED",
            "reach_bytes_upper": int(c["pose_term"] / EXCHANGE_S_PER_BYTE),
            "reach_basis": (
                "Pose is only 5.38% of S so it cannot close alone (fb1 section 4: "
                "pose->0 still leaves +0.020239). But pose is 93.3%/93.4% of dg2's "
                "measured damage and 65.3% of w72's. Protecting it in the loss is "
                "what would make every OTHER lever's refusal ratio fall."
            ),
            "evidence": (
                "Live trainer: no posenet/pose_loss/d_pose/w_pose. TR1: w_pose=0.0, "
                "compute_pose=False at train_tr1...:1851-1853 (rg5 re-derived at "
                "file:line). tc1:203,264 -- 'jd1 pose family: all 15 flags never "
                "fired ... this is why raw TR1 ships an inert 83 B pose stub'."
            ),
            "cost_to_falsify": "rides free on D1's run as a second loss term",
            "labels": {"absence": "MEASURED", "reach": "DERIVED"},
        },
        {
            "id": "D3",
            "name": "Exact-field reproduction is a self-imposed constraint the contest does not require",
            "status": "NAMED (db1 L120, vf1), NEVER MEASURED -- 0/117,964,800 positions classified",
            "reach_bytes_upper": 113_777,
            "reach_basis": (
                "The whole token stream. db1: 'scorer-cell equivalence is a weaker "
                "requirement than exact token identity; the 42,382 B demand fits "
                "inside the token member's mass'. vf1 measured the classification "
                "denominator at 0/117,964,800 positions and 0/113,777 coded bytes -- "
                "so 0 B of 42,382 B required = 0% measured evaluator-visible credit."
            ),
            "evidence_that_the_constraint_is_load_bearing": (
                "All four alternative-representation arms were killed by the "
                "exact-reproduction residual, not by their generator: hg1 359,280 B "
                "residual inside a 460,408 B container; hr3 362,473 B residual inside "
                "463,601 B; et1 535,761 B; ws1 918,904 B. Adjacent slack evidence: "
                "msr1 measured 63.4% of manufactured pixels had a 17x17 token window "
                "EXACTLY equal to GT, which EXONERATES the token field at those sites."
            ),
            "counter_evidence_to_respect": (
                "ld1 measured six lossy Lane field edits on dx2 and EVERY ONE made the "
                "archive BIGGER (+196/+279/+824/+1,528/+598/+21 B); mf1's explicit "
                "addressing cost +35,969 B. So tolerance does NOT pay when the changed "
                "subset must be NAMED. The open question is a representation whose "
                "coding units are already cells, so no address is ever emitted."
            ),
            "cost_to_falsify": (
                "CHEAP FORM: the tolerance curve. Reassign k random token positions "
                "for k in a log ladder, measure d_seg through the real receiver. "
                "~5 advisory n600 rows, $0 local, no Modal. Directly fills vf1's "
                "empty denominator."
            ),
            "labels": {"denominator": "MEASURED at 0%", "reach": "DERIVED"},
        },
        {
            "id": "D4",
            "name": "Decode-time compute (~1,300 s of the 1,800 s budget unspent)",
            "status": "ALREADY MEASURED and INSUFFICIENT (family A); family B BLOCKED on a named missing artifact",
            "reach_bytes_upper": 2_693,
            "reach_basis": (
                "dc1 family A (fixed-grid combinatorial hash sieve) is a REAL solver: "
                "510/510 blocks reproduced exactly, group-0 instance +4.488810 B "
                "ideal-body equivalent. But the five-group bounded sample LOSES "
                "-4.907404 B, and dc1's own best-case projection of the favourable "
                "group-0 row over all 600 frames is 2,693.286 B = 6.34% of the "
                "42,470 B ask. It CANNOT reach 42,382 B."
            ),
            "family_b": (
                "Constraint-shipped task-cell solve: gross ceiling is the whole "
                "113,777 B stream, but the explicit constraint is 44,244,000 B and "
                "P(C) is unmeasured. BLOCKED on one named artifact: a "
                "receiver-checkable certificate that EVERY candidate the scorer-free "
                "decoder may return lies inside the desired Seg/Pose cell, without "
                "loading the forbidden scorers. Family C (bits-back/REC) inherits the "
                "same missing quotient and refunds 0 bits on a deterministic field."
            ),
            "honest_adjudication": (
                "The ~1,300 s is real and unspent, but decode-time compute as "
                "currently formulated cannot buy 42,382 B. Family B is not a compute "
                "problem at all -- it is the SAME missing object as D3."
            ),
            "cost_to_falsify": "n/a -- already adjudicated",
            "labels": {"family_a": "MEASURED", "projection": "dc1's own, explicitly not a full-video result"},
        },
        {
            "id": "D5",
            "name": "sy2's rank-3 born-small edge/topology carrier -- its two gates have both since closed",
            "status": "UNASKED: the queue was never drained after its gates fell",
            "reach_bytes_upper": demand,
            "reach_basis": (
                "sy2 named it 'the only composition with enough object mass to target "
                "sub-0.12' and dispositioned it QUEUED-BEHIND-JF1/W96. Both gates have "
                "since been measured and refused: the jf1 diagonal closed at "
                "FORMULATION scope (dg2, 687x and 792x, monotone -- the smaller move is "
                "worse), and w72 refused 46.3x. Nobody re-ran sy2's queue afterwards."
            ),
            "cost_to_falsify": "one born-small training/materialization + one n600 row; dollar cost unestimated (sy2's own words)",
            "labels": {"gate_closures": "MEASURED", "queue_state": "DERIVED from sy2's own ranking"},
        },
        {
            "id": "D6",
            "name": "Buy pose WITH bytes -- every carrier rung ever measured REMOVES bytes, none ADDS",
            "status": "UNASKED, and bounded",
            "reach_bytes_upper": int(c["pose_term"] / EXCHANGE_S_PER_BYTE),
            "reach_basis": (
                "ap1 sampled the carrier at three COARSENING levels only (-2,742 / "
                "-5,875 / -9,035 B), all net-positive. The + direction is unsampled. "
                "Ceiling is pose->0 = 11,986 B = 28.28% of demand, so it is a second "
                "axis at best, never a close."
            ),
            "cost_to_falsify": "one carrier re-solve at a larger byte budget + one advisory row",
            "labels": {"unsampled_direction": "MEASURED (ap1 sampled 3 rungs, all negative-B)",
                       "ceiling": "DERIVED"},
        },
        {
            "id": "D7",
            "name": "Per-pair / temporal byte allocation (rate is global, distortion is a mean over 600 pairs)",
            "status": "WEAKLY ALREADY MEASURED -- low prior",
            "reach_bytes_upper": None,
            "reach_basis": (
                "hr3 measured per-FRAME mismatch Gini = 0.211095, 'much less "
                "concentrated across frames than BL1's per-position cost field' "
                "(Gini 0.99516). The concentration is SPATIAL, not TEMPORAL, so a "
                "per-pair reallocation has little to equalize."
            ),
            "cost_to_falsify": "n/a -- downranked on hr3's measured Gini",
            "labels": {"evidence": "MEASURED (hr3 line 65)"},
        },
        {
            "id": "D8",
            "name": "Seed / run-to-run variance of the shipped S",
            "status": "UNASKED",
            "reach_bytes_upper": None,
            "reach_basis": (
                "dx2 is ONE point of ONE run. If between-run S variance is "
                "comparable to the deltas the ~20 arms measured, the 'sharp optimum' "
                "reading is partly an instance property. SEARCH SCOPE for this "
                "absence: .omx/research/ddm_*2026082*.md -- every run-to-run figure "
                "found (cd1, rr6, rr8) is WALL-CLOCK variance, not S variance."
            ),
            "cost_to_falsify": "one extra seed; rides free on any retrain arm (D1/D2/D5)",
            "labels": {"absence": "NOT FOUND within the stated scope"},
        },
    ]


def ranked(dirs: list[dict]) -> list[dict]:
    """Rank by (plausible reach) / (cost to falsify). Cost is an ordinal, stated.

    Fail-closed: every enumerated direction id must appear in the ranking, so a
    direction can never be silently dropped between the enumeration and the table.
    """
    order = [
        ("D1+D2", 1, "42,382 B in play (object change, >=2 axes, 86.41% of S enters the objective)",
         "MEDIUM: one live-trainer run + byte-close + one advisory row"),
        ("D3", 2, "113,777 B ceiling; cheap discriminator exists",
         "LOW for the tolerance curve (~5 advisory rows, $0, no Modal)"),
        ("D5", 3, "42,382 B; sy2's own rank-1 object-mass candidate",
         "HIGH: a born-small build + materialization + n600 row"),
        ("D6", 4, "11,986 B ceiling (28.28% of demand) -- second axis only",
         "LOW-MEDIUM: one carrier re-solve"),
        ("D8", 5, "unquantified; recalibrates every prior verdict",
         "FREE if it rides a retrain; otherwise one full run"),
        ("D4", 6, "2,693 B measured ceiling -- CANNOT close",
         "n/a, adjudicated"),
        ("D7", 7, "downranked on hr3's per-frame Gini 0.211",
         "n/a, downranked"),
    ]
    rows = [
        {"rank": r, "direction": d, "reach": reach, "cost_to_falsify": cost}
        for d, r, reach, cost in order
    ]
    enumerated = {d["id"] for d in dirs}
    ranked_ids = {part for row in rows for part in row["direction"].split("+")}
    missing = enumerated - ranked_ids
    if missing:
        raise AssertionError(f"directions dropped between enumeration and ranking: {sorted(missing)}")
    unknown = ranked_ids - enumerated
    if unknown:
        raise AssertionError(f"ranking cites directions that were never enumerated: {sorted(unknown)}")
    return rows


def main() -> None:
    c = dx2_components()
    shape = probe_shape()
    shape["marginal_asymmetry"] = marginals(c)
    dirs = directions(c)
    doc = {
        "arm": "ddm_wq1_what_was_never_asked",
        "date_utc": "2026-08-24",
        "type": "DERIVED synthesis over MEASURED receipts; no scorer, no Modal, no Metal, $0",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "exchange_rate_S_per_byte": EXCHANGE_S_PER_BYTE,
        "exchange_rate_source": "ddm_tx1_toolbox_crosswalk_20260819.md section 0 — CITED, not re-derived",
        "dx2": c,
        "demand": {
            "fixed_distortion_bytes": DX2_BYTES - 137_986,
            "zero_distortion_bytes": DX2_BYTES - 180_218,
            "archive_ceiling_fixed_distortion": 137_986,
            "archive_ceiling_zero_distortion": 180_218,
        },
        "probe_shape": shape,
        "cross_regime_correction": cross_regime_correction(c),
        "directions": dirs,
        "ranked": ranked(dirs),
        "loop_until_dry_rounds": 6,
        "prior_law_prediction": {
            "predicted": ">=2 genuinely-unasked directions survive the corpus check, "
                         "at least one about the OBJECTIVE or the REPRESENTATION CLASS",
            "adjudication": "CONFIRMED",
            "count": 5,
            "surviving_ids": ["D1", "D2", "D3", "D5", "D6"],
            "objective_or_representation_among_them": True,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    p = OUT_DIR / "ddm_wq1_never_asked_table.json"
    p.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    files[p.name] = p

    # flat CSV of the ranked table for eyeball review
    lines = ["rank,direction,reach,cost_to_falsify"]
    for row in doc["ranked"]:
        r = row["reach"].replace('"', "'")
        ct = row["cost_to_falsify"].replace('"', "'")
        lines.append(f'{row["rank"]},{row["direction"]},"{r}","{ct}"')
    q = OUT_DIR / "ddm_wq1_ranked.csv"
    q.write_text("\n".join(lines) + "\n")
    files[q.name] = q

    manifest = {}
    for name, path in files.items():
        b = path.read_bytes()
        manifest[name] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    m = OUT_DIR / "MANIFEST.json"
    m.write_text(json.dumps(manifest, indent=2) + "\n")
    mb = m.read_bytes()
    manifest["MANIFEST.json"] = {"bytes": len(mb), "sha256": hashlib.sha256(mb).hexdigest()}

    for name, meta in manifest.items():
        print(f'{name}  {meta["bytes"]} B  sha256 {meta["sha256"]}')


if __name__ == "__main__":
    main()
