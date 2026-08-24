"""ddm_tri1 - exact arithmetic over MEASURED inputs: the 56 axis TRIPLES on the dx2 object,
plus the adversarial adjudication of tac1's one surviving PAIR (tokens x HPAC) at demanded scale.

This script MEASURES NOTHING. It fires no scorer, no Modal job, no Metal job. It reads no
upstream/ file. Every input below is a MEASURED value quoted from a named receipt; the arithmetic
is exact (Fraction for every rate-derived quantity) and deterministic.

Run:  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_tri1_triple_composition.py

Emits ddm_tri1_triple_table.{json,csv} + MANIFEST.json under the SSD artifact tier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from pathlib import Path

# --------------------------------------------------------------------------------------------
# MEASURED INPUTS - every one cited. Nothing here is derived by this script.
# --------------------------------------------------------------------------------------------

ARCHIVE_SHA = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
ARCHIVE_BYTES = 180_368  # dx2 authority row
D_SEG = Fraction(20139, 100_000_000)  # 0.00020139, contest-CUDA T4 n600
D_POSE_NUM = 637  # 0.00000637 = 637e-8, contest-CUDA T4 n600
D_POSE = Fraction(D_POSE_NUM, 100_000_000)
SCORE_DENOM = 37_545_489  # upstream/evaluate.py rate denominator
TARGET = Fraction(12, 100)

# Exchange rate: CITED from ddm_tx1_toolbox_crosswalk_20260819.md line 31, NOT re-derived.
RATE_PER_BYTE = Fraction(25, SCORE_DENOM)  # 6.658590e-07 S/B
TX1_CITATION = "ddm_tx1_toolbox_crosswalk_20260819.md:31 (CITED, not re-derived)"

# ar1b residue census - zero remainder, sums to ARCHIVE_BYTES exactly.
CENSUS = {
    "tokens": 113_777,
    "renderer": 30_856,
    "carrier": 22_010,
    "hpac": 13_515,
    "framing": 114,
    "residual": 96,
}

N_PIXELS = 117_964_800  # ms9 / mst1 / tba1
LANE_BOTH_CORRECT = 688_847  # ld1:62 - positions where GT and shipped token are both Lane

# ms9 partition of the 23,757 final seg flips (exact integers, ms9:13-16)
MS9_MANUFACTURED_FLIPS = 21_493  # 90.4702%, actuator = renderer (mst1: 78.7093% at native render)
MS9_TRANSMITTED_FLIPS = 2_264  # 9.5298%, actuator = tokens
MS9_TOTAL_FLIPS = 23_757

# dg2 measured rungs (macOS-CPU advisory, n600, matched CPU base) - the k-ladder is NESTED (ld1:9,63).
DG2_CPU_BASE = {"bytes": 180_368, "d_seg": 0.0003474, "d_pose": 0.00014701}
DG2_RUNGS = {
    "k040000": {"edits": 40_000, "bytes": 179_426, "d_seg": 0.0006754, "d_pose": 0.02521067},
    "k060000": {"edits": 60_000, "bytes": 178_792, "d_seg": 0.00083204, "d_pose": 0.05056445},
}

# tba1 D3 - the ONE untested direction whose ceiling is not arithmetically excluded (tba1:248).
TBA1_D3_CEILING_B = 38_649.8
TBA1_D3_CEILING_S = 0.025735
TBA1_D1_BEST_NET_B = 9.45  # explicit position-selected treatment, best over all thresholds
TBA1_D1_COLEX_B = 9.90  # PR101 colex-rank counter-attack, tba1's most generous token figure

# Two whole-body lossy objects at the FAR end of the same direction. These are DIFFERENT OBJECTS
# (RC1/NR1 temporal-program codebooks), NOT rungs of dg2's k-ladder. They are used only to test
# whether dg2's two-point power law survives contact with a large-byte measurement.
FAR_OBJECTS = {
    "ri1": {"bytes": 113_006, "S": 17.306291, "d_seg": 0.01605413, "d_pose": 24.41603851,
            "axis": "[env-mismatch advisory, n600]"},
    "ni1": {"bytes": 122_250, "S": 27.7984, "d_seg": 0.07583781, "d_pose": 40.53479004,
            "axis": "[contest-CUDA T4, n600]"},
}

# Per-axis MEASURED-OR-CANDIDATE movable ceiling at ACCEPTABLE distortion, each with its memo.
# "Acceptable" = the move does not cost more S in distortion than it credits in rate.
MOVABLE_CEILING_B = {
    "tokens": (TBA1_D1_COLEX_B, "ddm_tba1 D1 + colex-rank counter-attack; best net over all thresholds"),
    "renderer": (1078.0, "ddm_rj1 film_amortized_flat_w96, UNMEASURED - a generous upper bound (w72 sec4)"),
    "carrier": (0.0, "ddm_ap1 carrier_l1/l2/l3 all net-positive dS (worse); cheapest 167.8x"),
    "hpac": (0.0, "ddm_ap1 hpac_l1/l2/l3 at 81,588x / 21,952x / 14,497x the going rate"),
    "framing": (0.0, "ddm_ar1b receiver-required structure; measured-movable ceiling 0 B"),
    "residual": (0.0, "ddm_ap1 residual_l1/l2/l3 each returned 0 archive bytes"),
    "seg": (1.27, "ddm_msr1 oracle over 260 (interface x direction x delta) nets +1 pixel = 1.27 B"),
    "pose": (0.0, "no measured reducing mechanism on this object (dx2 row / sy2 / w72)"),
}

# Deepest MEASURED coarsening rung per axis (bytes), used for the adversarial sensitivity where
# every rung is granted its full byte credit at ZERO distortion - contradicting its own measurement.
DEEPEST_MEASURED_RUNG_B = {
    "tokens": TBA1_D3_CEILING_B,  # ddm_tba1 D3 ceiling
    "renderer": 10_879.0,  # ddm_w72 nested_group_dense_w72
    "carrier": 9_035.0,  # ddm_ap1 carrier_l3_fixed_coder
    "hpac": 6_026.0,  # ddm_ap1 hpac_l3
    "framing": 0.0,
    "residual": 0.0,  # ddm_ap1 three lattice levels each returned 0 archive bytes
}

# The four ACTUATORS (axes with a rate handle) and the two READOUTS, from tac1 sec5's matrix,
# which is built from ap1 (per-class SegNet response), ms9/mst1 (stage attribution), w72
# (bit-identical tokens under a renderer move) and ar1b (zero-remainder census).
ACTUATORS = ("tokens", "renderer", "carrier", "hpac")
READOUTS = ("seg", "pose")
DEAD_LEGS = ("framing", "residual")  # measured-movable ceiling 0 B
READOUT_ACTUATORS = {
    "seg": ("tokens", "renderer"),  # tokens = the 9.5298% transmitted part; renderer = the 90.4702%
    "pose": ("renderer", "carrier"),  # w72 (renderer dominant); ap1 (carrier load-bearing)
}

# Measured pairwise sy2 relations. QUOTED rows come from tac1's table; DERIVED rows extend tac1's
# own rule (stated in its sec5) to the pairs tac1 left blank because they were arithmetically
# IMPOSSIBLE as pairs but appear inside feasible triples.
PAIR_SY2 = {
    frozenset(("tokens", "hpac")): ("OBJECT-CHANGE", "tac1", "the model is fitted to the field"),
    frozenset(("renderer", "seg")): ("OBJECT-CHANGE", "tac1", "mst1: 78.7093% of manufactured seg is born at the native render"),
    frozenset(("renderer", "carrier")): ("OBJECT-CHANGE", "tac1", "sy2 rank-2: a moved renderer changes the object the carrier is solved against"),
    frozenset(("tokens", "renderer")): ("STACK", "tac1", "w72 decoded_token_sha256 cc10a7b0... bit-identical under a renderer move"),
    frozenset(("tokens", "carrier")): ("STACK", "tac1", "measured scope only; the carrier re-solve escape is named"),
    frozenset(("renderer", "hpac")): ("STACK", "tac1", "the model is priced on the token field, which a renderer move leaves bit-identical"),
    frozenset(("carrier", "seg")): ("STACK", "tac1", "ap1: 0 flips in all 5 classes at all 3 depths"),
    frozenset(("tokens", "framing")): ("STACK", "tac1", "framing ceiling 0 B"),
    frozenset(("tokens", "residual")): ("STACK", "tac1", "residual ceiling 0 B"),
    frozenset(("tokens", "seg")): ("DEGENERATE", "tac1", "one actuator asked to be smaller and more accurate"),
    frozenset(("renderer", "pose")): ("DEGENERATE", "tac1", "one actuator read twice"),
    frozenset(("tokens", "pose")): ("DEGENERATE", "tac1", "no pose actuator inside the pair"),
    frozenset(("hpac", "seg")): ("DEGENERATE", "tac1", "hpac has no non-destructive seg actuator"),
    # DERIVED by this arm, applying tac1's own sec5 rule to pairs tac1 left blank:
    frozenset(("carrier", "hpac")): ("STACK", "tri1-DERIVED", "ar1b census: disjoint sections; the carrier does not change the token field the model is priced on"),
    frozenset(("carrier", "pose")): ("DEGENERATE", "tri1-DERIVED", "carrier IS a pose actuator (ap1 d_pose +0.00987/+0.05466/+1.54927): one actuator read twice"),
    frozenset(("hpac", "pose")): ("DEGENERATE", "tri1-DERIVED", "hpac has no non-destructive pose actuator"),
    frozenset(("seg", "pose")): ("DEGENERATE", "tri1-DERIVED", "both legs are readouts: no actuator in the pair at all"),
    frozenset(("renderer", "framing")): ("STACK", "tri1-DERIVED", "framing ceiling 0 B"),
    frozenset(("renderer", "residual")): ("STACK", "tri1-DERIVED", "residual ceiling 0 B"),
    frozenset(("carrier", "framing")): ("STACK", "tri1-DERIVED", "framing ceiling 0 B"),
    frozenset(("carrier", "residual")): ("STACK", "tri1-DERIVED", "residual ceiling 0 B"),
    frozenset(("hpac", "framing")): ("STACK", "tri1-DERIVED", "framing ceiling 0 B"),
    frozenset(("hpac", "residual")): ("STACK", "tri1-DERIVED", "residual ceiling 0 B"),
    frozenset(("framing", "seg")): ("DEGENERATE", "tri1-DERIVED", "framing ceiling 0 B and no seg actuator"),
    frozenset(("framing", "pose")): ("DEGENERATE", "tri1-DERIVED", "framing ceiling 0 B and no pose actuator"),
    frozenset(("residual", "seg")): ("DEGENERATE", "tri1-DERIVED", "residual ceiling 0 B and no seg actuator"),
    frozenset(("residual", "pose")): ("DEGENERATE", "tri1-DERIVED", "residual ceiling 0 B and no pose actuator"),
    frozenset(("framing", "residual")): ("STACK", "tri1-DERIVED", "both ceilings 0 B"),
}


class Tri1Error(RuntimeError):
    """Raised when an internal invariant fails. Fail closed - never emit an unchecked table."""


# --------------------------------------------------------------------------------------------
# Exact base arithmetic
# --------------------------------------------------------------------------------------------


def pose_term(d_pose: Fraction) -> float:
    """sqrt(10*d_pose). The one irrational term; everything else stays exact."""
    return math.sqrt(10.0 * float(d_pose))


@dataclass(frozen=True)
class Base:
    rate_S: Fraction
    seg_S: Fraction
    pose_S: float
    S: float
    gap_S: float
    gap_B: int
    ceiling_B: int


def compute_base() -> Base:
    rate_S = Fraction(25 * ARCHIVE_BYTES, SCORE_DENOM)
    seg_S = 100 * D_SEG
    pose_S = pose_term(D_POSE)
    S = float(rate_S) + float(seg_S) + pose_S
    gap_S = S - float(TARGET)
    # Strict archive ceiling at fixed distortion: the largest integer B with S(B) < 0.12.
    ceiling_B = ARCHIVE_BYTES
    while float(Fraction(25 * ceiling_B, SCORE_DENOM)) + float(seg_S) + pose_S >= float(TARGET):
        ceiling_B -= 1
    return Base(rate_S, seg_S, pose_S, S, gap_S, ARCHIVE_BYTES - ceiling_B, ceiling_B)


def byte_equiv(s_value: float) -> float:
    return s_value / float(RATE_PER_BYTE)


# --------------------------------------------------------------------------------------------
# Axis inventory
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    name: str
    kind: str  # actuator | readout | dead
    nominal: str
    max_saving_S: float  # IDEALISED limit (pool -> 0 bytes, distortion term -> 0)
    max_saving_B: float
    structural_ceiling_S: float  # the measured structural partition, where one exists
    movable_ceiling_B: float
    movable_memo: str


def build_axes(base: Base) -> dict[str, Axis]:
    axes: dict[str, Axis] = {}
    for name, nbytes in CENSUS.items():
        kind = "dead" if name in DEAD_LEGS else "actuator"
        s = float(nbytes * RATE_PER_BYTE)
        ceil_b, memo = MOVABLE_CEILING_B[name]
        axes[name] = Axis(name, kind, f"{nbytes} B", s, float(nbytes), s, ceil_b, memo)

    # seg: structural ceiling is the MANUFACTURED part only (ms9). The transmitted 9.5298% is
    # removable only through the TOKEN axis, so it is not reachable when tokens is absent.
    seg_s = float(base.seg_S)
    seg_manufactured_s = seg_s * MS9_MANUFACTURED_FLIPS / MS9_TOTAL_FLIPS
    ceil_b, memo = MOVABLE_CEILING_B["seg"]
    axes["seg"] = Axis("seg", "readout", f"d_seg {float(D_SEG)}", seg_s, byte_equiv(seg_s),
                       seg_manufactured_s, ceil_b, memo)

    ceil_b, memo = MOVABLE_CEILING_B["pose"]
    axes["pose"] = Axis("pose", "readout", f"d_pose {float(D_POSE)}", base.pose_S,
                        byte_equiv(base.pose_S), base.pose_S, ceil_b, memo)
    return axes


# --------------------------------------------------------------------------------------------
# The triple surface + the cuts
# --------------------------------------------------------------------------------------------


@dataclass
class Triple:
    axes: tuple[str, str, str]
    joint_max_S: float
    joint_max_B: float
    each_pct: float | None
    per_axis_demand_B: dict[str, float]
    n_actuators: int
    n_readouts: int
    n_dead: int
    pair_classes: dict[str, str]
    sy2: str
    cut: str
    evidence: str
    reduces_to: str | None
    supply_B: float
    shortfall_x: float | None
    notes: list[str] = field(default_factory=list)


def pair_class(a: str, b: str) -> tuple[str, str, str]:
    key = frozenset((a, b))
    if key not in PAIR_SY2:
        raise Tri1Error(f"no sy2 class registered for pair {sorted(key)}")
    return PAIR_SY2[key]


def classify(triple: tuple[str, str, str], axes: dict[str, Axis], base: Base) -> Triple:
    names = tuple(sorted(triple))
    members = [axes[n] for n in names]
    joint_max_S = sum(m.max_saving_S for m in members)
    joint_max_B = sum(m.max_saving_B for m in members)

    feasible = joint_max_S >= base.gap_S
    each_pct = (base.gap_S / joint_max_S * 100.0) if feasible else None
    per_axis = ({m.name: m.max_saving_B * base.gap_S / joint_max_S for m in members}
                if feasible else {})

    n_act = sum(1 for n in names if n in ACTUATORS)
    n_read = sum(1 for n in names if n in READOUTS)
    n_dead = sum(1 for n in names if n in DEAD_LEGS)

    pcs: dict[str, str] = {}
    classes: list[str] = []
    for a, b in combinations(names, 2):
        cls, src, why = pair_class(a, b)
        pcs["x".join(sorted((a, b)))] = f"{cls} [{src}: {why}]"
        classes.append(cls)

    if "OBJECT-CHANGE" in classes:
        sy2 = "OBJECT-CHANGE"
    elif n_dead:
        sy2 = "DEGENERATE"
    else:
        sy2 = "STACK"

    # Supply = sum of per-axis measured-or-candidate movable ceilings at acceptable distortion.
    # Valid as a bound ONLY under the STACK reading (each closure transfers). Reported for every
    # row; the OBJECT-CHANGE rows are adjudicated individually in the memo, not by this number.
    supply = sum(m.movable_ceiling_B for m in members)
    shortfall = (base.gap_B / supply) if supply > 0 else None

    cut, evidence, reduces_to = decide_cut(names, axes, base, feasible, n_dead, n_read, classes)

    return Triple(names, joint_max_S, joint_max_B, each_pct, per_axis, n_act, n_read, n_dead,
                  pcs, sy2, cut, evidence, reduces_to, supply, shortfall)


def decide_cut(names, axes, base, feasible, n_dead, n_read, classes):
    """The falling-rule cut ladder. Order matters: the most structural cut fires first."""
    live = tuple(n for n in names if n not in DEAD_LEGS)

    if not feasible:
        short_B = base.gap_B - sum(axes[n].max_saving_B for n in names)
        return ("IMPOSSIBLE",
                f"sum of idealised maxima {sum(axes[n].max_saving_B for n in names):,.0f} B-eq "
                f"< the {base.gap_B:,} B demand, short by {short_B:,.0f} B-eq",
                None)

    if n_dead:
        return ("DEAD-LEG",
                f"{n_dead} leg(s) with MEASURED-MOVABLE CEILING 0 B (ar1b framing = "
                f"receiver-required structure; ap1 residual = 0 archive bytes at three lattice "
                f"levels). The triple reduces to its live legs, already adjudicated by tac1.",
                " x ".join(live) if len(live) > 1 else (live[0] if live else "nothing"))

    # A readout leg whose actuators are ALL absent has no handle inside the triple: the row is a
    # four-axis demand wearing three labels (tac1 sec5 point 1, lifted).
    orphans = [r for r in READOUTS if r in names
               and not set(READOUT_ACTUATORS[r]) & set(names)]
    if orphans:
        return ("DISGUISED-QUADRUPLE",
                f"readout leg(s) {orphans} have no actuator inside the triple "
                f"(seg actuators = {READOUT_ACTUATORS['seg']}, pose actuators = "
                f"{READOUT_ACTUATORS['pose']}); this is a 4-axis demand with 3 labels",
                None)

    if "OBJECT-CHANGE" not in classes:
        return ("STACK",
                "no leg changes the object any other leg was priced on; sy2's law says every "
                "measured closure transfers intact. sy2 disposition NO_BANKABLE_COMPOSITE_YET "
                "closes direct stacking on this converged object.",
                None)

    return ("OBJECT-CHANGE-CANDIDATE",
            "carries >=1 measured object-change relation; adjudicated individually in the memo",
            None)


# --------------------------------------------------------------------------------------------
# Part 1 - the last pair, adversarially, at demanded scale
# --------------------------------------------------------------------------------------------


def adjudicate_pair(base: Base) -> dict:
    """tokens x HPAC. Decomposed into realization classes; each priced from measured rows."""
    r = float(RATE_PER_BYTE)
    out: dict = {"pair": "tokens x hpac",
                 "pool_B": CENSUS["tokens"] + CENSUS["hpac"],
                 "demand_B": base.gap_B}
    pool_S = float((CENSUS["tokens"] + CENSUS["hpac"]) * RATE_PER_BYTE)
    out["each_pct"] = base.gap_S / pool_S * 100.0  # S-ratio, matching the triple table

    # --- dg2's measured lossy rungs -----------------------------------------------------------
    rungs = {}
    for tag, row in DG2_RUNGS.items():
        d_b = DG2_CPU_BASE["bytes"] - row["bytes"]
        credit = d_b * r
        d_seg = row["d_seg"] - DG2_CPU_BASE["d_seg"]
        dmg_seg = 100.0 * d_seg
        dmg_pose = math.sqrt(10.0 * row["d_pose"]) - math.sqrt(10.0 * DG2_CPU_BASE["d_pose"])
        damage = dmg_seg + dmg_pose
        rungs[tag] = {
            "edits": row["edits"], "bytes_shed": d_b, "credit_S": credit,
            "damage_seg_S": dmg_seg, "damage_pose_S": dmg_pose, "damage_S": damage,
            "net_delta_S": damage - credit, "ratio": damage / credit,
            "pose_share_of_damage": dmg_pose / damage,
            "final_flips_added": d_seg * N_PIXELS,
            "flips_per_edit": d_seg * N_PIXELS / row["edits"],
        }
    out["dg2_rungs"] = rungs

    lo, hi = rungs["k040000"], rungs["k060000"]
    br = hi["bytes_shed"] / lo["bytes_shed"]
    out["two_point_fit"] = {
        "bracket_B": [lo["bytes_shed"], hi["bytes_shed"]],
        "byte_ratio": br,
        "damage_exponent": math.log(hi["damage_S"] / lo["damage_S"]) / math.log(br),
        "credit_exponent": math.log(hi["credit_S"] / lo["credit_S"]) / math.log(br),
        "degrees_of_freedom": 0,
        "bracket_log_span_nats": math.log(br),
        "extrapolation_log_span_nats": math.log(base.gap_B / hi["bytes_shed"]),
    }
    out["two_point_fit"]["extrapolation_in_bracket_lengths"] = (
        out["two_point_fit"]["extrapolation_log_span_nats"]
        / out["two_point_fit"]["bracket_log_span_nats"])
    e = out["two_point_fit"]["damage_exponent"]
    lever = base.gap_B / hi["bytes_shed"]
    out["point_estimate_at_demand"] = hi["ratio"] * lever ** (e - 1.0)

    # --- the FIT-FREE bound: holds for ANY damage exponent >= 0 -------------------------------
    credit_at_demand = base.gap_B * r
    out["fit_free_bound"] = {
        "assumption": "damage is non-decreasing in bytes shed within the family; ld1:9/:63 settle "
                      "that the k-ladder edit sets are NESTED (a single descending cost ranking "
                      "sliced at increasing k), and every edit converts a CORRECT label to an "
                      "incorrect one, so the transmitted error count is monotone by construction",
        "damage_floor_S": hi["damage_S"],
        "credit_at_demand_S": credit_at_demand,
        "ratio_lower_bound": hi["damage_S"] / credit_at_demand,
        "required_exponent_to_clear": math.log(credit_at_demand / hi["damage_S"]) / math.log(lever),
        "measured_exponent": e,
    }

    # --- tba1 D3: the alphabet-collapse extreme of the SAME family ----------------------------
    d3_credit = TBA1_D3_CEILING_B * r
    d3_flips_measured_rate = LANE_BOTH_CORRECT * hi["flips_per_edit"]
    d3_flips_structural = float(LANE_BOTH_CORRECT)  # every correct Lane label destroyed
    out["tba1_D3"] = {
        "ceiling_B": TBA1_D3_CEILING_B,
        "ceiling_S_tba1": TBA1_D3_CEILING_S,
        "credit_S_recomputed": d3_credit,
        "pct_of_demand": TBA1_D3_CEILING_B / base.gap_B * 100.0,
        "lane_positions_destroyed": LANE_BOTH_CORRECT,
        "seg_damage_S_at_dg2_measured_flip_rate": 100.0 * d3_flips_measured_rate / N_PIXELS,
        "seg_damage_S_at_structural_floor": 100.0 * d3_flips_structural / N_PIXELS,
        "ratio_seg_leg_only_measured_rate": (100.0 * d3_flips_measured_rate / N_PIXELS) / d3_credit,
        "ratio_seg_leg_only_structural": (100.0 * d3_flips_structural / N_PIXELS) / d3_credit,
        "pose_leg": "UNBOUNDED and uncounted; dg2 measured pose at 93.3-93.4% of the damage on "
                    "rungs 11.5x SMALLER than this one",
    }

    # --- adversarial test of the fit: two large-byte objects in the SAME direction -------------
    # If the power law is honest, it should not badly mis-predict a real measurement 37-43x
    # further out. These are DIFFERENT OBJECTS, so this is corroboration by direction, never
    # family membership. Convention matches dg2's exactly: damage / credit, damage = net + credit.
    far = {}
    for tag, row in FAR_OBJECTS.items():
        shed = ARCHIVE_BYTES - row["bytes"]
        credit = shed * r
        net = row["S"] - base.S
        damage = net + credit
        predicted = hi["ratio"] * (shed / hi["bytes_shed"]) ** (e - 1.0)
        far[tag] = {
            "axis": row["axis"], "bytes_shed": shed, "credit_S": credit,
            "net_delta_S": net, "damage_S": damage, "measured_ratio": damage / credit,
            "ratio_predicted_by_dg2_power_law": predicted,
            "law_underpredicts_by_x": (damage / credit) / predicted,
            "different_object": True,
        }
    out["far_object_cross_check"] = far
    out["fit_direction_verdict"] = (
        "The two-point law UNDER-predicts the real cost at large byte savings on both nearby "
        "large-byte objects. My attempt to break the extrapolation found evidence in the OPPOSITE "
        "direction to the one that would rescue the pair."
    )

    # --- the realization family --------------------------------------------------------------
    out["realizations"] = [
        {"id": "R1", "name": "lossy field edit (k-threshold) + model refit",
         "status": "CLOSED at FORMULATION in BOTH directions",
         "evidence": "dg2 two rungs, 791.7x and 687.3x; damage exponent 0.7252 vs credit 1.0000 "
                     "so shrinking RAISES the ratio; fit-free bound >= 25.6x at demanded scale"},
        {"id": "R2", "name": "alphabet collapse (class merge to 4 symbols) + model refit = tba1 D3",
         "status": "CLOSED by this arm on the seg leg alone",
         "evidence": "tba1's own 38,649.8 B ceiling priced against dg2's MEASURED 0.9528 "
                     "final-flips-per-edit: >= 21.6x, and >= 22.7x at the structural floor. Pose "
                     "is uncounted on top and is 93% of the damage in dg2"},
        {"id": "R3", "name": "lossless traversal reorder + model REFIT",
         "status": "UNMEASURED - the last live cell",
         "evidence": "to2 tested 9 orders but REPLACED the learned model with generic coders "
                     "(tba1:236 classifies it field FIXED / model REPLACED / order CHANGED), so "
                     "its +196.07% is NOT a verdict on refit. ad2 likewise used Brotli q11 on both "
                     "sides. Zero occurrences of refit/retrain in either memo."},
        {"id": "R4", "name": "better model of the UNCHANGED field",
         "status": "not a pair - this is the hpac axis alone, inside the sharp optimum",
         "evidence": "oe1/ae1/ef1/xs1; ad2:96 records RB1's 0 B measured recode headroom on the "
                     "fixed dx2 representation"},
    ]
    return out


# --------------------------------------------------------------------------------------------
# The exhaustion arithmetic
# --------------------------------------------------------------------------------------------


def adjudicate_competing_explanations(base: Base) -> dict:
    """E1 (actuator count, #1233/tac1) vs E2' (POSE-specific objective gap).

    CORRECTION-20260824: an earlier routing named
    `src/tac/pr130_lift/train_semantic_quantized_resumable.py` and put the unoptimized share at
    86.41275%. That is a SISTER trainer, imported only for EMA helpers; it did not produce this
    lineage. The real trainer is `tools/train_ddm_cl1_hpac_capacity.py`, re-verified at source
    here (:1320-1322): `task_loss = F.cross_entropy(logits, target)`;
    `rate_loss = args.rate_lambda * math.log(2) * variable_weight_bits(model, deployed=False)
    / pixels`; `loss = task_loss + rate_loss`. `--rate-lambda` defaults to 1.0 and `<= 0.0` RAISES
    (:964-965), so the rate penalty is MANDATORY. `grep -c pose` on that file returns 0.

    So SEG and RATE are both differentiated. POSE alone is not - 5.38472% of S, at 6.2647x seg's
    marginal. The claim narrows by ~16x and becomes POSE-specific, which makes it testable.

    The naive discriminator "pose dominates the damage" does NOT separate E1 from E2': pose's
    marginal is 6.2647x seg's by the score function's own arithmetic, so pose dominance is
    over-determined. Two sharp tests are used instead: the RESPONSE ASYMMETRY, and the
    SEG-vs-POSE DOMINATION SPLIT over every measured ap1 family (which bounds E2's coverage).
    """
    pose_marginal = 5.0 / base.pose_S  # d/d(d_pose) of sqrt(10*d_pose)
    d_pose_control = 6.365873831275037e-06  # ap1's own control
    seg_budget = float(D_SEG)

    indirect = {}
    for tag, d_pose_delta, d_s_dist, nbytes in (
        ("ap1_carrier_l1", 0.00987277665396, 0.306332390066, 2742),
        ("ap1_carrier_l2", 0.0546586253751, 0.731379127655, 5875),
        ("ap1_carrier_l3", 1.54927024096, 3.92810647691, 9035),
    ):
        indirect[tag] = {
            "d_seg_delta": 0.0,
            "d_seg_delta_note": "EXACTLY ZERO in all five classes at all three levels (ap1:132-138)",
            "d_pose_amplification_x": d_pose_delta / d_pose_control,
            "delta_S_dist": d_s_dist,
            "multiples_of_entire_dx2_seg_budget": (d_s_dist / 100.0) / seg_budget,
            "bytes": nbytes,
        }

    hi = DG2_RUNGS["k060000"]
    direct = {
        "probe": "dg2 k060000 - 60,000 DELIBERATE correct-label corruptions of the token field",
        "term": "seg (the ONE term the objective differentiates)",
        "d_seg_ratio_to_base": hi["d_seg"] / DG2_CPU_BASE["d_seg"],
        "final_flips_per_edit": (hi["d_seg"] - DG2_CPU_BASE["d_seg"]) * N_PIXELS / hi["edits"],
        "amplification": "NONE - essentially one-for-one",
    }

    # Which measured families are SEG-dominated? This BOUNDS E2's coverage: a seg-dominated
    # refusal cannot be explained by an unshaped term, because seg IS differentiated.
    split = {}
    for tag, d_seg_delta, d_s_dist in (
        ("ap1_semantic_l1", 0.00871830410428, 8.54699549584),
        ("ap1_semantic_l2", 0.0571536509196, 17.1962839791),
        ("ap1_semantic_l3", 0.0610589599609, 17.8672642239),
        ("ap1_carrier_l1", 0.0, 0.306332390066),
        ("ap1_carrier_l2", 0.0, 0.731379127655),
        ("ap1_carrier_l3", 0.0, 3.92810647691),
        ("ap1_hpac_l1", 0.670700480143, 103.870699820),
        ("ap1_hpac_l2", 0.488460515340, 85.7877961051),
        ("ap1_hpac_l3", 0.503762986925, 87.3595347294),
        ("ap1_residual_l1", 0.406644278632, 78.7139712043),
    ):
        seg_s = 100.0 * d_seg_delta
        pose_s = d_s_dist - seg_s
        split[tag] = {"seg_S": seg_s, "pose_S": pose_s,
                      "pose_share_of_damage_pct": pose_s / d_s_dist * 100.0,
                      "dominated_by": "POSE" if pose_s > seg_s else "SEG"}
    seg_dominated = sorted(k for k, v in split.items() if v["dominated_by"] == "SEG")

    worst = max(v["d_pose_amplification_x"] for v in indirect.values())
    return {
        "E1_actuator_count": {
            "claim": "four movable actuators, three score terms, many-to-one; a third leg adds no handle",
            "status": "NOT REFUTED - independently measured by this arm in the triple table "
                      "(seg has exactly 2 actuators, pose 2, framing/residual 0)",
            "explains": "the STRUCTURE of the failure - why no combination of dx2's axes opens a route",
            "predicts": "only a representation with more independent actuators opens routes; says "
                        "NOTHING about retraining the same representation",
        },
        "E2prime_pose_specific_objective_gap": {
            "claim": "the ONE score term nobody differentiates is the one that kills every "
                     "POSE-DOMINATED perturbation, at 6.2647x the marginal of the protected term",
            "source": "ddm_wq1 @ 1cc670031c as CORRECTED by MAIN; trainer re-verified at source "
                      "by this arm at tools/train_ddm_cl1_hpac_capacity.py:1320-1322",
            "seg_differentiated": True,
            "seg_mechanism": "F.cross_entropy(logits, target)",
            "rate_differentiated": True,
            "rate_mechanism": "rate_lambda * log(2) * variable_weight_bits(...) / pixels; "
                              "--rate-lambda default 1.0, <=0 RAISES (:964-965) => MANDATORY",
            "pose_differentiated": False,
            "pose_mechanism": "grep -c pose on the real trainer returns 0",
            "unoptimized_share_of_S_pct": base.pose_S / base.S * 100.0,
            "rate_share_pct": float(base.rate_S) / base.S * 100.0,
            "seg_share_pct": float(base.seg_S) / base.S * 100.0,
            "pose_marginal": pose_marginal,
            "pose_marginal_vs_seg_x": pose_marginal / 100.0,
            "superseded_claim": "an earlier routing put the unoptimized share at 86.41275% on a "
                                "SISTER trainer; corrected to 5.38472%, a ~16x scope reduction",
            "explains": "the MAGNITUDE of the POSE-DOMINATED refusals only",
            "does_not_explain": seg_dominated,
            "predicts": "adding POSE to the objective (a one-term addition - rate is already "
                        "there) cheapens the pose-dominated refusals and does NOTHING for the "
                        "seg-dominated ones",
        },
        "domination_split": split,
        "coverage_bound": {
            "pose_dominated": sorted(k for k, v in split.items() if v["dominated_by"] == "POSE"),
            "seg_dominated": seg_dominated,
            "note": "E2' covers the pose-dominated families ONLY. The seg-dominated ones are the "
                    "HPAC and residual coarsenings - and ap1 records their seg damage as broken "
                    "EXACT TOKEN RECONSTRUCTION, i.e. a DECODE-INTEGRITY failure, not a "
                    "shaped-vs-unshaped one. That is a THIRD mechanism, not a counter-example.",
        },
        "three_mechanism_model": {
            "rate_slack_exhausted_by_optimization": "rate WAS differentiated under a mandatory "
                "positive lambda, so the object already spent its rate budget. This - not the "
                "actuator count - is why tier-1 movable mass is only 1,089.17 B.",
            "pose_fragility_unshaped_term": "pose carries 6.2647x seg's marginal and has no "
                "gradient, so pose-dominated perturbations detonate (up to 243,371x base).",
            "decode_integrity": "HPAC/residual coarsening breaks exact token reconstruction, "
                "which is why those refusals are SEG-dominated despite seg being differentiated.",
        },
        "naive_discriminator_rejected": (
            "'pose dominates the damage' does NOT separate E1 from E2': pose's marginal is 6.2647x "
            "seg's by the score function's own arithmetic, so pose dominance is over-determined."),
        "sharp_discriminator": {
            "direct_attack_on_optimized_term": direct,
            "indirect_perturbation_on_unoptimized_term": indirect,
            "asymmetry_x": worst / direct["d_seg_ratio_to_base"],
            "verdict": "The OPTIMIZED term responds ~1:1 to a DIRECT attack (0.9528 flips/edit). "
                       "The UNOPTIMIZED term amplifies up to 243,371x against a perturbation that "
                       "leaves seg EXACTLY unchanged, costing 195x the ENTIRE seg budget. That is "
                       "~5 orders of magnitude beyond the 6.2647x the score arithmetic explains - "
                       "but ONLY on the pose-dominated families. E2' is supported there and is "
                       "silent elsewhere; E1 survives as a structural fact about a DIFFERENT "
                       "question.",
        },
        "both_true": "E1 explains STRUCTURE (why a third axis adds no handle). E2' explains the "
                     "MAGNITUDE of the pose-dominated refusals. Decode integrity explains the "
                     "seg-dominated ones. Three mechanisms, cleanly separated by the domination "
                     "split; no single one accounts for the record.",
        "discriminating_experiment": {
            "fire": "add POSE to the objective and retrain the EXISTING representation - a "
                    "ONE-TERM addition, since rate is already differentiated under a mandatory "
                    "positive lambda",
            "measure": "the realized distortion cost of an ALREADY-MEASURED pose-dominated "
                       "perturbation - ap1 carrier_l1, today 0.306332 S at EXACTLY ZERO seg cost "
                       "for 2,742 B (100% pose-dominated, so it is the cleanest probe available)",
            "E2prime_predicts": "that cost falls materially on a pose-trained body",
            "E1_predicts": "it does not - the actuator structure is unchanged by retraining",
            "falsifier_for_E2prime": "no material reduction => E2' refuted and the new-object "
                                     "route stands on E1 alone",
            "scope_limit": "E2' predicts NOTHING for the seg-dominated families (HPAC, residual) "
                           "or for the D3 closure, which is priced on the seg leg alone",
        },
    }


def exhaustion(axes: dict[str, Axis], base: Base) -> dict:
    supply = sum(a.movable_ceiling_B for a in axes.values())
    seg_transmitted_B = byte_equiv(float(base.seg_S) * MS9_TRANSMITTED_FLIPS / MS9_TOTAL_FLIPS)
    generous = sum(DEEPEST_MEASURED_RUNG_B[n] for n in CENSUS)
    return {
        "demand_B": base.gap_B,
        "tier1_measured_or_candidate_supply_B": supply,
        "tier1_pct_of_demand": supply / base.gap_B * 100.0,
        "tier1_shortfall_x": base.gap_B / supply,
        "tier1_per_axis": {n: {"B": a.movable_ceiling_B, "memo": a.movable_memo}
                           for n, a in axes.items()},
        "tier2_plus_free_transmitted_seg_B": supply + seg_transmitted_B,
        "tier2_pct_of_demand": (supply + seg_transmitted_B) / base.gap_B * 100.0,
        "tier2_shortfall_x": base.gap_B / (supply + seg_transmitted_B),
        "tier3_every_rate_rung_free_of_distortion_B": generous,
        "tier3_pct_of_demand": generous / base.gap_B * 100.0,
        "tier3_supply_over_demand_x": generous / base.gap_B,
        "tier3_note": "grants every measured coarsening rung its FULL byte credit at ZERO "
                      "distortion - contradicting the very measurements that produced them. "
                      "It OVERSHOOTS the demand at 152%. The object is therefore NOT "
                      "byte-starved; it is DISTORTION-starved. Every byte the campaign can name "
                      "is reachable, and every one is measured load-bearing on a scored term.",
        "tier3_per_axis": {n: DEEPEST_MEASURED_RUNG_B[n] for n in CENSUS},
    }


# --------------------------------------------------------------------------------------------
# Assembly + self-checks
# --------------------------------------------------------------------------------------------


def cross_checks(base: Base, axes: dict[str, Axis]) -> list[dict]:
    checks = []

    def add(name, mine, prior, tol, src):
        ok = abs(float(mine) - float(prior)) <= tol
        checks.append({"check": name, "this_arm": float(mine), "prior_receipt": float(prior),
                       "agrees": ok, "source": src})
        if not ok:
            raise Tri1Error(f"cross-check FAILED: {name} {mine} vs {prior}")

    add("census sums to archive", sum(CENSUS.values()), ARCHIVE_BYTES, 0, "ar1b (zero remainder)")
    add("S from components", base.S, 0.14821987563243377, 1e-15, "dx2 authority row")
    add("gap", base.gap_S, 0.028219875632433777, 1e-15, "fb1 / tl1 / tac1")
    add("fixed-distortion demand B", base.gap_B, 42382, 0, "fb1 / tl1 / ar1b / sy2")
    add("strict archive ceiling B", base.ceiling_B, 137986, 0, "fb1")
    add("exchange rate S/B", float(RATE_PER_BYTE), 6.658590e-07, 5e-13, TX1_CITATION)
    add("ms9 flip partition", MS9_MANUFACTURED_FLIPS + MS9_TRANSMITTED_FLIPS, MS9_TOTAL_FLIPS, 0, "ms9:38")
    add("d_seg from ms9 numerator", MS9_TOTAL_FLIPS / N_PIXELS, 0.00020139058430989585, 1e-17, "ms9:23")
    add("ms9 manufactured pct", MS9_MANUFACTURED_FLIPS / MS9_TOTAL_FLIPS * 100, 90.47017721, 1e-6, "ld1:51")
    add("seg byte-equivalent", axes["seg"].max_saving_B, 30245.2, 0.1, "ms9:16")
    add("seg manufactured byte-eq", byte_equiv(axes["seg"].structural_ceiling_S), 27362.9, 0.2, "ms9:15")
    add("tba1 D3 ceiling in S", TBA1_D3_CEILING_B * float(RATE_PER_BYTE), TBA1_D3_CEILING_S, 5e-7, "tba1:248")
    # Independent reproduction of tac1's OWN pair row from this arm's axis inventory.
    add("tac1 tokens x hpac pool B", axes["tokens"].max_saving_B + axes["hpac"].max_saving_B,
        127292, 0, "tac1 sec7 / jf1 bar")
    add("tac1 tokens x hpac each %",
        base.gap_S / (axes["tokens"].max_saving_S + axes["hpac"].max_saving_S) * 100, 33.29, 0.01,
        "tac1 sec4")
    # fb1's zero-distortion residual, reproduced from the axis table rather than a scenario row.
    add("both distortions -> 0 residual B",
        base.gap_B - (axes["seg"].max_saving_B + axes["pose"].max_saving_B), 150, 0.5, "fb1 / tl1")
    # ap1's damage/credit column reproduced as a multiple of the going rate (tac1 quoted these).
    add("ap1 carrier_l1 multiple of going rate", 0.000111718595939 / float(RATE_PER_BYTE),
        167.8, 0.1, "ap1:105 damage/B column")
    add("ap1 hpac_l1 multiple of going rate", 0.0543256798219 / float(RATE_PER_BYTE),
        81587.0, 2.0, "ap1:108 damage/B column")
    return checks


def build() -> dict:
    base = compute_base()
    axes = build_axes(base)
    checks = cross_checks(base, axes)

    names = sorted(axes)
    triples = [classify(t, axes, base) for t in combinations(names, 3)]
    if len(triples) != 56:
        raise Tri1Error(f"expected 56 triples, built {len(triples)}")

    counts: dict[str, int] = {}
    for t in triples:
        counts[t.cut] = counts.get(t.cut, 0) + 1

    survivors = [t for t in triples if t.cut == "OBJECT-CHANGE-CANDIDATE"]

    # Falsifier: >=1 triple OPEN with >=2 object-change relations AND under 40% of each axis.
    two_oc = []
    for t in triples:
        n_oc = sum(1 for v in t.pair_classes.values() if v.startswith("OBJECT-CHANGE"))
        if n_oc >= 2:
            two_oc.append({"triple": list(t.axes), "n_object_change": n_oc,
                           "each_pct": t.each_pct, "cut": t.cut})
    under40 = [t for t in triples if t.each_pct is not None and t.each_pct < 40.0]

    return {
        "arm": "ddm_tri1",
        "type": "DERIVED (exact arithmetic over MEASURED inputs); measures nothing",
        "verdict_scope": "INSTANCE:DX2_OBJECT_TRIPLEWISE_ISO_0_12_SURFACE",
        "object": {"archive_sha256": ARCHIVE_SHA, "archive_bytes": ARCHIVE_BYTES,
                   "d_seg": float(D_SEG), "d_pose": float(D_POSE),
                   "S": base.S, "axis": "[contest-CUDA T4, n600]"},
        "exchange_rate_S_per_B": float(RATE_PER_BYTE),
        "exchange_rate_citation": TX1_CITATION,
        "base": {"rate_S": float(base.rate_S), "seg_S": float(base.seg_S), "pose_S": base.pose_S,
                 "S": base.S, "gap_S": base.gap_S, "gap_B": base.gap_B,
                 "strict_archive_ceiling_B": base.ceiling_B},
        "cross_checks": checks,
        "axes": {n: {"kind": a.kind, "nominal": a.nominal, "max_saving_S": a.max_saving_S,
                     "max_saving_B_equiv": a.max_saving_B,
                     "structural_ceiling_S": a.structural_ceiling_S,
                     "measured_movable_ceiling_B": a.movable_ceiling_B,
                     "movable_memo": a.movable_memo} for n, a in axes.items()},
        "pair_adjudication": adjudicate_pair(base),
        "triples": [{"axes": list(t.axes), "joint_max_S": t.joint_max_S,
                     "joint_max_B_equiv": t.joint_max_B, "each_pct": t.each_pct,
                     "per_axis_demand_B": t.per_axis_demand_B, "n_actuators": t.n_actuators,
                     "n_readouts": t.n_readouts, "n_dead_legs": t.n_dead,
                     "pair_sy2_classes": t.pair_classes, "sy2": t.sy2, "cut": t.cut,
                     "evidence": t.evidence, "reduces_to": t.reduces_to,
                     "measured_supply_B": t.supply_B, "shortfall_x": t.shortfall_x}
                    for t in triples],
        "counts": counts,
        "supply_bound_caveat": (
            "measured_supply_B sums the per-axis measured-or-candidate movable ceilings. It is a "
            "valid BOUND only under the STACK reading, where every measured closure transfers. On "
            "an OBJECT-CHANGE row the voided leg's ceiling does NOT bind, so those ten rows are "
            "adjudicated individually in the memo and NOT by this number."),
        "tba1_D1_best_net_B_no_colex": TBA1_D1_BEST_NET_B,
        "object_change_candidates": [list(t.axes) for t in survivors],
        "falsifier": {
            "clause_1_two_object_change_relations": two_oc,
            "clause_2_under_40_pct_each": [list(t.axes) for t in under40],
            "intersection": [list(t.axes) for t in triples
                             if t.each_pct is not None and t.each_pct < 40.0
                             and sum(1 for v in t.pair_classes.values()
                                     if v.startswith("OBJECT-CHANGE")) >= 2],
        },
        "exhaustion": exhaustion(axes, base),
        "competing_explanations": adjudicate_competing_explanations(base),
        "NOT_CLAIMED": [
            "Every fraction is a REQUIREMENT, not an achieved value. No arm has achieved any.",
            "This does not prove sub-0.12 unreachable. It prices the triples on THIS object.",
            "Axis maxima are IDEALISED limits, several physically degenerate.",
            "Quadruples and above are out of scope (declared SCOPE reduction).",
            "A NEW BASIS is out of scope: a changed alphabet leaves the dx2 residue map.",
            "score_claim: false; promotion_eligible: false; pointer_moved: false",
        ],
    }


def emit(payload: dict, outdir: Path) -> list[tuple[str, int, str]]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int, str]] = []

    jpath = outdir / "ddm_tri1_triple_table.json"
    jpath.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cpath = outdir / "ddm_tri1_triple_table.csv"
    cols = ["axes", "joint_max_S", "joint_max_B_equiv", "each_pct", "n_actuators", "n_readouts",
            "n_dead_legs", "sy2", "cut", "reduces_to", "measured_supply_B", "shortfall_x",
            "evidence"]
    with cpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in payload["triples"]:
            w.writerow([" x ".join(row["axes"])] + [row[c] for c in cols[1:]])

    for p in (jpath, cpath):
        data = p.read_bytes()
        written.append((p.name, len(data), hashlib.sha256(data).hexdigest()))

    man = {"arm": "ddm_tri1", "object_sha256": ARCHIVE_SHA,
           "files": [{"name": n, "bytes": b, "sha256": s} for n, b, s in written]}
    mpath = outdir / "MANIFEST.json"
    mpath.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mdata = mpath.read_bytes()
    written.append((mpath.name, len(mdata), hashlib.sha256(mdata).hexdigest()))
    return written


def main() -> None:
    payload = build()
    outdir = Path("/Volumes/APDataStore/pact/ddm_tri1_triple_composition")
    written = emit(payload, outdir)

    b = payload["base"]
    print(f"S={b['S']!r}  gap={b['gap_S']!r}  demand={b['gap_B']:,} B  "
          f"ceiling={b['strict_archive_ceiling_B']:,} B")
    print(f"cross-checks: {len(payload['cross_checks'])} ALL PASS")
    print("\ncuts:", json.dumps(payload["counts"], sort_keys=True))
    print("object-change candidates:", payload["object_change_candidates"])
    print("falsifier intersection:", payload["falsifier"]["intersection"])

    pa = payload["pair_adjudication"]
    print(f"\nPAIR tokens x hpac: pool {pa['pool_B']:,} B, each {pa['each_pct']:.2f}%")
    for tag, r in sorted(pa["dg2_rungs"].items()):
        print(f"  {tag}: shed {r['bytes_shed']:,} B  net dS +{r['net_delta_S']:.6f}  "
              f"{r['ratio']:.1f}x  flips/edit {r['flips_per_edit']:.4f}")
    f = pa["two_point_fit"]
    print(f"  fit: damage exp {f['damage_exponent']:.4f}, credit exp {f['credit_exponent']:.4f}, "
          f"dof {f['degrees_of_freedom']}, extrapolation {f['extrapolation_in_bracket_lengths']:.2f} "
          f"bracket-lengths")
    print(f"  point estimate at demand: {pa['point_estimate_at_demand']:.1f}x")
    fb = pa["fit_free_bound"]
    print(f"  FIT-FREE bound: >= {fb['ratio_lower_bound']:.2f}x   "
          f"required exponent to clear {fb['required_exponent_to_clear']:.4f} "
          f"(measured {fb['measured_exponent']:.4f})")
    d3 = pa["tba1_D3"]
    print(f"  tba1 D3: {d3['ceiling_B']:,.1f} B = {d3['pct_of_demand']:.1f}% of demand; "
          f"seg leg alone {d3['ratio_seg_leg_only_measured_rate']:.2f}x "
          f"({d3['ratio_seg_leg_only_structural']:.2f}x structural)")

    ex = payload["exhaustion"]
    print(f"\nEXHAUSTION: supply {ex['tier1_measured_or_candidate_supply_B']:,.2f} B = "
          f"{ex['tier1_pct_of_demand']:.4f}% of demand, {ex['tier1_shortfall_x']:.2f}x short")
    print(f"  +free transmitted seg: {ex['tier2_pct_of_demand']:.2f}%, "
          f"{ex['tier2_shortfall_x']:.2f}x short")
    print(f"  every rung free of distortion: {ex['tier3_pct_of_demand']:.2f}% of demand "
          f"-> NOT byte-starved, DISTORTION-starved")

    print("\nfar-object cross-check (DIFFERENT objects, same direction):")
    for tag, row in sorted(pa["far_object_cross_check"].items()):
        print(f"  {tag} {row['axis']}: shed {row['bytes_shed']:,} B  measured "
              f"{row['measured_ratio']:.1f}x  vs law-predicted "
              f"{row['ratio_predicted_by_dg2_power_law']:.1f}x  -> law under-predicts "
              f"{row['law_underpredicts_by_x']:.2f}x")

    print("\nobject-change candidates, individually:")
    for row in payload["triples"]:
        if row["cut"] != "OBJECT-CHANGE-CANDIDATE":
            continue
        print(f"  {' x '.join(row['axes']):32s} each {row['each_pct']:5.2f}%  "
              f"supply {row['measured_supply_B']:8,.2f} B  "
              f"{row['shortfall_x']:7.2f}x short")

    ce = payload["competing_explanations"]
    e2 = ce["E2prime_pose_specific_objective_gap"]
    print(f"\nE1 vs E2': seg differentiated={e2['seg_differentiated']} "
          f"rate differentiated={e2['rate_differentiated']} "
          f"pose differentiated={e2['pose_differentiated']}")
    print(f"  unoptimized share of S = {e2['unoptimized_share_of_S_pct']:.5f}% (pose only)  "
          f"pose marginal {e2['pose_marginal']:.2f} ({e2['pose_marginal_vs_seg_x']:.4f}x seg)")
    print(f"  E2' does NOT cover: {', '.join(e2['does_not_explain'])}")
    d = ce["sharp_discriminator"]["direct_attack_on_optimized_term"]
    print(f"  DIRECT attack on OPTIMIZED seg: {d['d_seg_ratio_to_base']:.3f}x base, "
          f"{d['final_flips_per_edit']:.4f} flips/edit - no amplification")
    for tag, v in sorted(ce["sharp_discriminator"]["indirect_perturbation_on_unoptimized_term"].items()):
        print(f"  INDIRECT {tag}: d_seg EXACTLY 0, d_pose x{v['d_pose_amplification_x']:,.0f}, "
              f"costs {v['multiples_of_entire_dx2_seg_budget']:.2f}x the whole seg budget")

    print("\npayload:")
    for n, nb, s in written:
        print(f"  {n}  {nb:,} B  {s}")


if __name__ == "__main__":
    main()
