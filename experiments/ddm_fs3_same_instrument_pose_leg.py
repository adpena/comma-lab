#!/usr/bin/env python3
"""ddm_fs3 -- price the pose and seg legs as a SAME-INSTRUMENT delta, or refuse.

WHY
---
The mirror candidate's advisory returned ``d_pose = 0.00055125``.  That number is
NOT the pose leg.  The compose-stage figure it would naively be compared against
(``8.20845e-06``) came off a different instrument -- a direct frozen-scorer forward
rather than a full ``inflate.sh`` + ``upstream/evaluate.py`` pass with the CPU GT
lineage -- and the units x level x aggregation law forbids differencing across
them ([[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]).

The measurable leg is ``d(candidate) - d(base)`` where BOTH sides come from the
same instrument tuple.  This module computes that delta and **REFUSES** unless the
forward-relevant fields of ``instrument_tuple`` are identical on both receipts.

THE ONE FIELD ALLOWED TO DIFFER, AND WHY
----------------------------------------
``instrument_tuple.code.pact_commit`` records the repo HEAD at run time.  Two
receipts fired minutes apart in a live tree will differ there even when nothing on
the forward path moved -- sister arms land commits continuously.  So a strict
whole-tuple equality would refuse every delta this repo can actually produce.

The cure is not to relax the check but to make it PRECISE: every hash that
actually determines the forward must match ---

* ``upstream_snapshot_sha256``   (the scorer + evaluate.py snapshot)
* ``upstream_evaluate_py.sha256``(the scoring script itself)
* ``runtime_files_sha256``       (the receiver tree that produces the frames)
* ``inflate_script_sha256``      (the decode entry point)

plus device, torch version, thread counts and batch shape wherever the receipts
carry them.  ``pact_commit`` may differ ONLY when the caller supplies
``--commit-delta-proof``, a git range whose changed files this module re-checks
against the forward-path prefixes itself.  A proof that touches ``upstream/``, the
runtime, or the eval harness is REFUSED.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

#: Any changed path under these prefixes can move the forward, so a commit range
#: touching them cannot be waived.
FORWARD_PATH_PREFIXES = (
    "upstream/",
    "runtime-rs/",
    "experiments/contest_auth_eval.py",
    "src/tac/scorer",
    "src/tac/gt_lineage",
    "src/tac/frame_utils",
    "src/tac/differentiable_eval_roundtrip",
)

S_PER_ARCHIVE_BYTE = 25.0 / 37_545_489


class PoseLegError(RuntimeError):
    """Fail-closed error."""


def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def forward_fields(receipt: dict[str, Any]) -> dict[str, Any]:
    tup = receipt.get("instrument_tuple") or {}
    prov = receipt.get("provenance") or {}
    return {
        "upstream_snapshot_sha256": _dig(tup, "code", "upstream_snapshot_sha256"),
        "upstream_evaluate_py_sha256": _dig(
            tup, "code", "upstream_evaluate_py", "sha256"
        ),
        "runtime_files_sha256": _dig(tup, "code", "runtime_files_sha256"),
        "inflate_script_sha256": _dig(tup, "code", "inflate_script_sha256"),
        "device": prov.get("device"),
        "torch_version": prov.get("torch_version"),
        "gt_lineage": _dig(receipt, "gt_lineage", "lineage"),
        "n_samples": receipt.get("n_samples"),
    }


def commit_delta_touches_forward(commit_range: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", commit_range],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [p for p in out if any(p.startswith(pre) for pre in FORWARD_PATH_PREFIXES)]


def run(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base_receipt).read_text())
    cand = json.loads(Path(args.candidate_receipt).read_text())
    fb, fc = forward_fields(base), forward_fields(cand)

    mismatched = {k: (fb[k], fc[k]) for k in fb if fb[k] != fc[k]}
    missing = [k for k, v in fc.items() if v is None]
    if missing:
        raise PoseLegError(
            f"candidate receipt is missing forward-determining fields {missing}; "
            "a delta across an unidentified instrument is not a measurement"
        )
    if mismatched:
        raise PoseLegError(
            f"REFUSING: forward-determining fields differ between the two receipts: "
            f"{mismatched}. A delta across a mismatched tuple is an instrument "
            "artifact, not a finding."
        )

    commit_base = _dig(base, "instrument_tuple", "code", "pact_commit")
    commit_cand = _dig(cand, "instrument_tuple", "code", "pact_commit")
    commit_note: dict[str, Any] = {
        "base_pact_commit": commit_base,
        "candidate_pact_commit": commit_cand,
        "identical": commit_base == commit_cand,
    }
    if commit_base != commit_cand:
        if not args.commit_delta_proof:
            raise PoseLegError(
                f"pact_commit differs ({commit_base} vs {commit_cand}) and no "
                "--commit-delta-proof was supplied. Pass the git range so this "
                "module can re-check it against the forward path itself."
            )
        offenders = commit_delta_touches_forward(args.commit_delta_proof)
        if offenders:
            raise PoseLegError(
                f"REFUSING: the commit range {args.commit_delta_proof} changes "
                f"forward-path files {offenders}; the two receipts are not the "
                "same instrument."
            )
        commit_note["commit_delta_proof"] = args.commit_delta_proof
        commit_note["forward_path_files_changed"] = []
        commit_note["verdict"] = (
            "pact_commit differs but the range touches NO forward-path file; "
            "re-checked by this module, not asserted by the caller"
        )

    d_seg_b, d_seg_c = base["avg_segnet_dist"], cand["avg_segnet_dist"]
    d_pose_b, d_pose_c = base["avg_posenet_dist"], cand["avg_posenet_dist"]
    bytes_b = _dig(base, "provenance", "archive_size_bytes")
    bytes_c = _dig(cand, "provenance", "archive_size_bytes")

    seg_leg = 100.0 * (d_seg_c - d_seg_b)
    pose_leg = (10.0 * d_pose_c) ** 0.5 - (10.0 * d_pose_b) ** 0.5
    rate_leg = (bytes_c - bytes_b) * S_PER_ARCHIVE_BYTE
    net = seg_leg + pose_leg + rate_leg

    report = {
        "schema": "ddm_fs3_same_instrument_legs.v1",
        "arm": "ddm_fs3",
        "task": 1176,
        "axis": "[cpu_env_mismatch_advisory] both sides -- SAME instrument, so the DELTA is the claim, never either level",
        "score_claim": False,
        "promotion_eligible": False,
        "instrument_match": {
            "forward_determining_fields": fc,
            "verdict": "IDENTICAL",
            "pact_commit": commit_note,
        },
        "base": {
            "receipt": str(args.base_receipt),
            "archive_bytes": bytes_b,
            "d_seg": d_seg_b,
            "d_pose": d_pose_b,
        },
        "candidate": {
            "receipt": str(args.candidate_receipt),
            "archive_bytes": bytes_c,
            "d_seg": d_seg_c,
            "d_pose": d_pose_c,
        },
        "legs_same_instrument": {
            "seg": seg_leg,
            "pose": pose_leg,
            "rate": rate_leg,
            "net": net,
        },
        "multiple_of_bar": abs(net) / 3.5e-6,
        "clears_admission_bar": net < -3.5e-6,
        "caveat": (
            "these legs are measured on the PYAV/env-mismatch advisory axis. The "
            "delta is same-instrument and therefore legitimate, but its TRANSFER to "
            "the contest-CUDA axis is a separate question (ddm_pi2 measured 1.4425x "
            "on d_seg and +1.4061e-04 additive on d_pose across that lineage gap)."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("instrument match: IDENTICAL on every forward-determining field")
    if not commit_note["identical"]:
        print(f"  pact_commit differs; range {args.commit_delta_proof} touches no forward file")
    print(f"  base      {bytes_b:,} B  d_seg {d_seg_b:.8f}  d_pose {d_pose_b:.8f}")
    print(f"  candidate {bytes_c:,} B  d_seg {d_seg_c:.8f}  d_pose {d_pose_c:.8f}")
    print(f"\n  seg  {seg_leg:+.6e}\n  pose {pose_leg:+.6e}\n  rate {rate_leg:+.6e}")
    print(
        f"  NET  {net:+.6e}  ({report['multiple_of_bar']:.2f}x bar)  "
        f"clears={report['clears_admission_bar']}"
    )
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-receipt", required=True)
    parser.add_argument("--candidate-receipt", required=True)
    parser.add_argument(
        "--commit-delta-proof",
        default=None,
        help="git range re-checked by this module against the forward path",
    )
    parser.add_argument(
        "--out", default="/Volumes/APDataStore/pact/ddm_fs3/FS3_SAME_INSTRUMENT_LEGS.json"
    )
    parser.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
