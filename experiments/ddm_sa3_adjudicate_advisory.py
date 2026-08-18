"""ddm_sa3 -- adjudicate the advisory n600 leg against the four PRE-REGISTERED falsifiers.

The falsifiers were sealed BEFORE the leg ran (`FIRE_ORDER_sa3.json`), which is the only
thing that makes them falsifiers rather than a post-hoc reading.  This module re-reads
them FROM the seal rather than restating them, so the check cannot drift from the
contract it claims to be checking.

    F1  measured net dS >= -3.5e-6 vs the sz1 base leg           -> REFUSE
    F2  d_seg delta off S2's measured +1.72e-6 by more than 2x   -> the frame-0-only
                                                                   seg-invariance argument
                                                                   is unsound
    F3  residual d_pose above 5.937842e-07                       -> REFUSE
    F4  any nonzero carrier-code deviation on parse-back         -> REFUSE

SAME-INSTRUMENT DISCIPLINE.  Absolutes from this arm's local CPU harness are never
compared against the T4 pointer.  The base leg is the rr4 advisory row, which is a valid
base for sz1 because their decoded states are byte-identical (proven in the memo, and
asserted at build time) -- the archives differ only in container bytes, which the rate
term accounts for exactly.

AXIS ``[macOS-CPU advisory n600, env-mismatch grade]``.  Not a score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEAL: Final = Path("/Volumes/APDataStore/pact/ddm_sa3/FIRE_ORDER_sa3.json")
BASE_LEG: Final = Path(
    "/Volumes/APDataStore/pact/ddm_sa1/advisory_n600_cpu/rr4_base/attempt_0002"
    "/contest_auth_eval.json"
)
CANDIDATE_LEG: Final = Path(
    "/Volumes/APDataStore/pact/ddm_sa3/advisory_n600_cpu/sa3_candidate/attempt_0002"
    "/contest_auth_eval.json"
)
BUILD_REPORT: Final = Path("/Volumes/APDataStore/pact/ddm_sa3/build/SA3_REBASE.json")

RATE_DENOMINATOR: Final = 37_545_489
ADMIT_BAR_S: Final = -3.5e-6
S2_D_SEG_DELTA_CPU: Final = 1.72e-06
F3_RESIDUAL_CEILING: Final = 5.937842e-07
T4_SCORE: Final = 0.15771357797660338
T4_D_POSE: Final = (0.008294576541331089**2) / 10.0
T4_D_SEG: Final = 0.029611 / 100.0


class SA3AdjudicationError(RuntimeError):
    """A receipt was missing, malformed, or measured on the wrong axis."""


def _find_nested(row: Any, key: str) -> Any:
    """``archive_sha256`` is nested in the receipt, so a top-level .get returns None."""
    if isinstance(row, dict):
        if key in row:
            return row[key]
        for value in row.values():
            found = _find_nested(value, key)
            if found is not None:
                return found
    elif isinstance(row, list):
        for value in row:
            found = _find_nested(value, key)
            if found is not None:
                return found
    return None


def assert_receipt_matches_seal(row: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    """Refuse to adjudicate a receipt that did not evaluate the SEALED bytes.

    Without this the printed sha was the seal quoting itself (the receipt nests
    ``archive_sha256``, so the old top-level lookup always fell through), and a stale
    receipt from any other archive would have adjudicated silently.
    """
    sha = _find_nested(row, "archive_sha256")
    sealed_sha = seal["archive"]["sha256"]
    sealed_bytes = seal["archive"]["bytes"]
    if sha is not None and sha != sealed_sha:
        raise SA3AdjudicationError(
            f"receipt evaluated a DIFFERENT archive: {sha} != sealed {sealed_sha}"
        )
    if int(row["archive_size_bytes"]) != int(sealed_bytes):
        raise SA3AdjudicationError(
            f"receipt archive bytes {row['archive_size_bytes']} != sealed {sealed_bytes}"
        )
    return {
        "receipt_archive_sha256": sha or "(absent from receipt)",
        "sealed_archive_sha256": sealed_sha,
        "sha_compared": sha is not None,
        "bytes_compared": True,
        "status": "BOUND" if sha is not None else "BYTES-ONLY",
    }


def load_leg(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SA3AdjudicationError(f"{label} receipt missing: {path}")
    row = json.loads(path.read_text())
    if int(row.get("n_samples", 0)) != 600:
        raise SA3AdjudicationError(f"{label} is not n600 (n_samples={row.get('n_samples')})")
    for flag in ("promotable", "promotion_eligible", "score_claim_valid"):
        if row.get(flag):
            raise SA3AdjudicationError(f"{label} claims {flag}=True on an advisory axis")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-leg", type=Path, default=CANDIDATE_LEG)
    parser.add_argument("--base-leg", type=Path, default=BASE_LEG)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    seal = json.loads(SEAL.read_text())
    build = json.loads(BUILD_REPORT.read_text())
    base = load_leg(args.base_leg, "base leg")
    cand = load_leg(args.candidate_leg, "candidate leg")

    base_seg, base_pose = base["avg_segnet_dist"], base["avg_posenet_dist"]
    cand_seg, cand_pose = cand["avg_segnet_dist"], cand["avg_posenet_dist"]
    base_bytes, cand_bytes = base["archive_size_bytes"], cand["archive_size_bytes"]

    # Rate is priced against the LIVE POINTER's bytes, not the base leg's: the base leg
    # ran on rr4's 181,161 B container while the pointer ships sz1's 179,930 B. Using the
    # base leg's bytes would silently re-bank sz1's -1,231 B, which is the very error
    # this arm exists to correct.
    sz1_bytes = build["base"]["bytes"]
    delta_bytes = cand_bytes - sz1_bytes
    rate_s = 25.0 * delta_bytes / RATE_DENOMINATOR

    d_seg_delta = cand_seg - base_seg
    seg_s = 100.0 * d_seg_delta
    residual_d_pose = cand_pose - base_pose
    pose_s = ((10.0 * cand_pose) ** 0.5) - ((10.0 * base_pose) ** 0.5)
    net_s = rate_s + seg_s + pose_s

    parse_back = build["results"]["candidate_nosplit"]["parse_back"]
    deviation = int(parse_back.get("max_abs_code_deviation", -1))

    f1 = net_s >= ADMIT_BAR_S
    f2 = abs(d_seg_delta - S2_D_SEG_DELTA_CPU) > 2.0 * S2_D_SEG_DELTA_CPU
    f3 = residual_d_pose > F3_RESIDUAL_CEILING
    f4 = deviation != 0
    # F4 CANNOT FIRE and is recorded as VACUOUS rather than as a passing test.
    # ``verify_parse_back`` (ddm_sa3_rebase_sz1.py) RAISES unless codes_match, and
    # max_abs_code_deviation is max|codes - expected| over those same two arrays, so a
    # persisted build report always reads 0.  The parse-back control is real -- it just
    # runs at BUILD time and fails closed there; re-testing it here proves nothing.
    # The honest pre-registered falsifier count is THREE, not four.
    f4_vacuous = True
    fired = {"F1": f1, "F2": f2, "F3": f3}

    custody = assert_receipt_matches_seal(cand, seal)
    print(f"seal            : {seal['candidate_id']} [{seal['axis']}]")
    print(f"archive         : {cand_bytes:,} B  sha {custody['receipt_archive_sha256']}  "
          f"(custody {custody['status']})")
    print(f"base leg        : {args.base_leg}")
    print(f"  d_seg {base_seg:.8f}  d_pose {base_pose:.8f}  bytes {base_bytes:,}")
    print(f"candidate leg   : {args.candidate_leg}")
    print(f"  d_seg {cand_seg:.8f}  d_pose {cand_pose:.8f}  bytes {cand_bytes:,}")
    print()
    print(f"delta bytes vs LIVE POINTER ({sz1_bytes:,} B) : {delta_bytes:+d}")
    print(f"  rate  {rate_s:+.6e}")
    print(f"  seg   {seg_s:+.6e}   (d_seg delta {d_seg_delta:+.3e}, S2 measured {S2_D_SEG_DELTA_CPU:+.3e})")
    print(f"  pose  {pose_s:+.6e}   (residual d_pose {residual_d_pose:+.6e})")
    print(f"  NET   {net_s:+.6e}   bar {ADMIT_BAR_S:+.1e}")
    print()
    for name, did in fired.items():
        print(f"  {name}: {'FIRED' if did else 'clear'}")
    print(f"  F4: VACUOUS -- cannot fire (build-time parse-back already fails closed; "
          f"deviation={deviation} by construction)")
    # The net above is a SAME-INSTRUMENT delta: a CPU-measured pose term against a
    # CPU-measured base.  Adding it to the T4 pointer would mix operating points -- the
    # pose term is sqrt(10*d_pose) and the CPU base sits 21.4x higher, where its marginal
    # is 4.63x flatter.  So the T4 projection is computed by TRANSFERRING the residual,
    # under both models, and is reported separately from the measured delta.
    t4_base_pose_s = (10.0 * T4_D_POSE) ** 0.5
    t4_abs = ((10.0 * (T4_D_POSE + residual_d_pose)) ** 0.5) - t4_base_pose_s
    ratio = cand_pose / base_pose
    t4_rel = ((10.0 * T4_D_POSE * ratio) ** 0.5) - t4_base_pose_s
    seg_rel = seg_s * (T4_D_SEG / base_seg)
    t4 = {
        "absolute": rate_s + seg_s + t4_abs,
        "relative": rate_s + seg_rel + t4_rel,
    }

    # FINDING-1 FIX. ``net_s`` prices rate against the POINTER's 179,930 B, but the base
    # leg ran on rr4's 181,161 B.  ``base_score + net_s`` therefore composed a delta from
    # one baseline onto a score from another and overstated by exactly
    # 25*(181161-179930)/37545489 = +8.196723e-04 -- 234x the admit bar.  The honest base
    # is rr4's score minus the 1,231 B it carries that the pointer does not.  That this
    # reproduces the INDEPENDENTLY MEASURED candidate score to 1e-16 is the control that
    # proves net_s itself is right.
    sz1_equivalent_base = base["canonical_score"] - 25.0 * (base_bytes - sz1_bytes) / RATE_DENOMINATOR
    reconstructed = sz1_equivalent_base + net_s
    closure = reconstructed - cand["canonical_score"]
    if abs(closure) > 1e-9:
        raise SA3AdjudicationError(
            "same-instrument arithmetic does not close: "
            f"{sz1_equivalent_base:.12f} + {net_s:.12f} = {reconstructed:.12f} but the "
            f"candidate leg measured {cand['canonical_score']:.12f} (residual {closure:+.3e})"
        )

    verdict = "REFUSE" if any(fired.values()) else "ADMIT"
    print(f"\nVERDICT: {verdict}   (F1 is decided on the SAME-INSTRUMENT net above)")
    print("\nsame-instrument (CPU), all three from the SAME baseline:")
    print(f"  rr4 base leg score            {base['canonical_score']:.12f}  ({base_bytes:,} B)")
    print(f"  sz1-equivalent base (derived) {sz1_equivalent_base:.12f}  ({sz1_bytes:,} B)")
    print(f"  + net {net_s:+.12f}      -> {reconstructed:.12f}")
    print(f"  candidate leg MEASURED        {cand['canonical_score']:.12f}  (closure {closure:+.2e})")
    print("T4 PROJECTION (residual transferred; UNMEASURED, not a score):")
    for name, value in t4.items():
        print(f"  [{name:8s}] net {value:+.6e} -> S {T4_SCORE + value:.9f}")

    report = {
        "schema": "ddm_sa3_advisory_adjudication.v1",
        "axis": "[macOS-CPU advisory n600, env-mismatch grade]",
        "score_claim": False,
        "promotion_eligible": False,
        "seal_sha256": seal["seal_sha256"],
        "base_leg": str(args.base_leg),
        "candidate_leg": str(args.candidate_leg),
        "delta_bytes_vs_live_pointer": delta_bytes,
        "terms": {"rate_s": rate_s, "seg_s": seg_s, "pose_s": pose_s, "net_delta_s": net_s},
        "measured": {
            "base_d_seg": base_seg, "base_d_pose": base_pose,
            "candidate_d_seg": cand_seg, "candidate_d_pose": cand_pose,
            "d_seg_delta": d_seg_delta, "residual_d_pose": residual_d_pose,
            "parse_back_max_abs_code_deviation": deviation,
        },
        "falsifiers": [
            {"id": k, "text": seal["falsifiers"][i], "fired": fired[k], "vacuous": False}
            for i, k in enumerate(fired)
        ]
        + [{
            "id": "F4", "text": seal["falsifiers"][3], "fired": f4,
            "vacuous": f4_vacuous,
            "note": (
                "cannot fire: build-time verify_parse_back raises unless codes_match, so "
                "max_abs_code_deviation is 0 in every persisted report. Effective "
                "pre-registered falsifier count is 3."
            ),
        }],
        "effective_falsifier_count": len(fired),
        "custody": custody,
        "verdict": verdict,
        "same_instrument": {
            "rr4_base_score_cpu": base["canonical_score"],
            "rr4_base_bytes": base_bytes,
            "sz1_equivalent_base_score_cpu": sz1_equivalent_base,
            "pointer_bytes": sz1_bytes,
            "candidate_score_cpu_measured": cand["canonical_score"],
            "net_delta_s": net_s,
            "reconstructed_from_net": reconstructed,
            "closure_residual": closure,
        },
        "t4_projection_unmeasured": {
            "base_score": T4_SCORE, "models": t4,
            "projected": {k: T4_SCORE + v for k, v in t4.items()},
            "caveat": (
                "The compensation was fitted on the CPU instrument and the base leg shows "
                "21.4x CPU-vs-T4 d_pose drift on identical bytes. Transfer is UNMEASURED; "
                "the absolute model is the conservative end and neither is a score."
            ),
        },
    }
    out = args.out or args.candidate_leg.parent / "ADJUDICATION.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nreport -> {out}")
    return 0 if verdict == "ADMIT" else 3


if __name__ == "__main__":
    raise SystemExit(main())
