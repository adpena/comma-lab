#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-RA3: the ONE legal way to price an advisory-axis pose measurement, as code.

THE RULE (operator/MAIN binding 2026-08-16, from ``pi2``'s measurement).  The advisory pose
instrument is neither broken nor multiplicatively biased.  It carries a FIXED ADDITIVE FLOOR of
``d_pose = 1.4061e-04``, essentially all of it GT-decode-path drift (PyAV/``yuv420_to_rgb`` GT vs
DALI/nvdec GT, same frozen scorer, n600).  Scorer-forward CPU/CUDA drift is ``3.57e-12`` and is
falsified as the cause by nine orders of magnitude.  Closure, re-derived here at import time:

    6.88e-06 (contest-CUDA authority baseline) + 1.4061e-04 (floor) = 1.474900e-04
                                                       measured advisory base = 1.4747e-04
                                                                        error = 0.014%

Because the floor is additive IN ``d_pose``, it cancels in the ABSOLUTE DELTA and in NOTHING else:

    delta_d_pose_abs = d_pose_advisory(candidate) - d_pose_advisory(base)
    dS_pose          = sqrt(10 * (T4_BASE + delta_d_pose_abs)) - sqrt(10 * T4_BASE)

FORBIDDEN, and this module refuses to produce them:
  * an advisory ``d_pose`` RATIO quoted as a score or compared against a ratio bar.  A ratio of
    floored quantities understates the contest-axis ratio by up to F = advisory_base/T4_base.
    Ratios remain valid ONLY for comparing candidates measured on the SAME instrument, because
    the floor is then common -- that is what makes ra2's 2x2 internally sound.
  * rescaling a ``dS_pose`` computed on the advisory axis.

WHY THIS IS A MODULE AND NOT A PARAGRAPH.  The 11.5x "bar" that three arms carried into their
verdicts was a ratio of floored quantities compared against a ratio bar -- two level errors
compounding in the OPTIMISTIC direction.  Encoding the rule is the only way it stops recurring.

Axis: [macOS-CPU advisory] in, contest-CUDA-referenced dS out. score_claim=false, promotable=false.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

#: MEASURED by pi2 (2026-08-16): the GT-decode-path floor on the advisory pose axis.
ADVISORY_POSE_FLOOR = 1.4061e-04
#: MEASURED contest-CUDA authority baseline at the hv1 frontier (pose term 0.0082945765).
T4_BASE_D_POSE = 6.88e-06
#: MEASURED advisory base on the jc1/ra2/ra3 instrument.
ADVISORY_BASE_D_POSE = 1.4747e-04
#: The contest rate term's marginal value.
S_PER_BYTE = 25.0 / 37_545_489.0

_CLOSURE = abs(T4_BASE_D_POSE + ADVISORY_POSE_FLOOR - ADVISORY_BASE_D_POSE) / ADVISORY_BASE_D_POSE
if _CLOSURE > 1e-3:
    raise RuntimeError(
        f"floor model does not close on its own constants ({_CLOSURE:.5f} relative); "
        "refusing to price anything"
    )


class PricingRefusal(RuntimeError):
    """Fail-closed refusal: an illegal pricing was requested."""


@dataclass(frozen=True)
class PosePrice:
    """A floor-free price for one advisory-axis pose measurement."""

    label: str
    advisory_d_pose: float
    delta_d_pose_abs: float
    dS_pose: float
    bytes_returned: float
    dS_rate_credit: float
    net_dS: float
    pose_cost_over_rate_credit: float
    breaks_even: bool


def break_even_delta_d_pose(bytes_returned: float) -> float:
    """The largest absolute pose damage a byte credit can pay for."""
    credit = bytes_returned * S_PER_BYTE
    return ((math.sqrt(10.0 * T4_BASE_D_POSE) + credit) ** 2) / 10.0 - T4_BASE_D_POSE


def price(label: str, advisory_d_pose: float, bytes_returned: float) -> PosePrice:
    """Price an advisory pose measurement the only legal way: through the absolute delta."""
    if advisory_d_pose < ADVISORY_POSE_FLOOR:
        raise PricingRefusal(
            f"{label}: advisory d_pose {advisory_d_pose:.6e} is BELOW the measured floor "
            f"{ADVISORY_POSE_FLOOR:.6e}; that is not physically reachable on this instrument "
            "and indicates a wrong base, a wrong instrument, or a mis-parsed receipt"
        )
    delta = advisory_d_pose - ADVISORY_BASE_D_POSE
    d_pose_term = math.sqrt(10.0 * (T4_BASE_D_POSE + delta)) - math.sqrt(10.0 * T4_BASE_D_POSE)
    credit = bytes_returned * S_PER_BYTE
    return PosePrice(
        label=label,
        advisory_d_pose=advisory_d_pose,
        delta_d_pose_abs=delta,
        dS_pose=d_pose_term,
        bytes_returned=bytes_returned,
        dS_rate_credit=credit,
        net_dS=d_pose_term - credit,
        pose_cost_over_rate_credit=(d_pose_term / credit if credit > 0 else math.inf),
        breaks_even=bool(d_pose_term <= credit),
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row", action="append", default=[], metavar="LABEL=D_POSE[,BYTES]",
        help="price one advisory measurement; BYTES defaults to the r=11 credit 1846.75",
    )
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    rows = args.row or [
        "ra2_pose_subspace_r11=0.0022432311489210995",
        "ra2_euclid_subspace_r11=0.004220052711028264",
        "ra2_pose_subspace_r8=0.025799909424459113,7387.0",
        "jc1_euclid_keepset_r11=0.03469834",
    ]
    priced = []
    for spec in rows:
        label, _, rhs = spec.partition("=")
        parts = rhs.split(",")
        value = float(parts[0])
        returned = float(parts[1]) if len(parts) > 1 else 22161.0 / 12.0
        priced.append(price(label, value, returned))

    print(f"break-even delta_d_pose_abs at 1846.75 B = "
          f"{break_even_delta_d_pose(22161.0 / 12.0):.6e}")
    print(f"{'candidate':>28s} {'adv d_pose':>12s} {'delta_abs':>12s} "
          f"{'dS_pose':>10s} {'net dS':>10s} {'cost/credit':>12s}")
    for row in priced:
        print(f"{row.label:>28s} {row.advisory_d_pose:12.8f} {row.delta_d_pose_abs:12.4e} "
              f"{row.dS_pose:+10.6f} {row.net_dS:+10.6f} "
              f"{row.pose_cost_over_rate_credit:11.1f}x")

    if args.json_out:
        payload = {
            "schema": "ddm_ra3_advisory_pose_pricing.v1",
            "rule": "delta_d_pose_abs only; the GT-decode floor cancels there and nowhere else",
            "advisory_pose_floor": ADVISORY_POSE_FLOOR,
            "t4_base_d_pose": T4_BASE_D_POSE,
            "advisory_base_d_pose": ADVISORY_BASE_D_POSE,
            "floor_model_closure_relative_error": _CLOSURE,
            "score_claim": False,
            "promotable": False,
            "rows": [row.__dict__ for row in priced],
        }
        with open(args.json_out, "w") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
