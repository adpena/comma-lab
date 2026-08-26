#!/usr/bin/env python3
"""make_candidate_seal.py — freeze a candidate into one checkable sealed object.

Operator 2026-08-18: *"Things can be better frozen and constrained through engineering."*
This is the producer half of the candidate-seal contract (task #1115); the consumer half is
``tools/fire_modal_auth_eval.py --seal``, which refuses to spend a paid Modal call unless
every pin written here still holds against disk.

WHAT IT REFUSES TO DO.  It has **no flag for a hand-typed sha**.  Every hash in the seal is
computed from the bytes on disk at seal time, because a producer that accepted a typed digest
would reproduce the exact failure the seal exists to stop (``rr2``: fired bytes were never
the proved bytes).  ``--verify-archive-sha`` exists for the caller who already holds an
expected value — it is CHECKED against the measured bytes and refuses on mismatch, and it
never becomes the stored value.

THE BAR CARRIES ITS OWN DERIVATION.  ``--admit-bar-net-ds`` is stored beside the frontier
pointer score and candidate it was derived against, so the validator can re-derive it later
and refuse when the ground has moved (``qs4``: a compensation constant carried onto a
different object cost +2.4e-4 S).  ``--bar-tolerance`` declares how far the baseline may move
before the bar is stale; the honest default is 0.0 — any movement refuses.

IT VALIDATES ITS OWN OUTPUT.  After writing, the seal is re-read and re-validated through the
same ``validate_seal`` the fire path uses.  A seal that cannot pass its own consumer's gate
is deleted rather than left on disk to be discovered at fire time.

Example (the ddm_sa1 rank-1 candidate):

    .venv/bin/python tools/make_candidate_seal.py \\
        --candidate-id sm3r_keep01 \\
        --runtime-dir /Volumes/APDataStore/pact/ddm_sa1/candidate_runtime \\
        --axis contest_cuda \\
        --admit-bar-net-ds -3.5e-6 \\
        --retained-path /Volumes/APDataStore/pact/ddm_sa1/retained/sm3r_keep01 \\
        --falsifier "net dS >= -3.5e-6 at n600 refutes the rate credit" \\
        --out /Volumes/APDataStore/pact/ddm_sa1/SEAL_sm3r_keep01.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.candidate_seal import (  # noqa: E402
    SEAL_AXES,
    AdmitBar,
    SealContractError,
    build_seal,
    check_pin_consistency,
    measure_archive_identity,
    read_pointer_state,
    validate_seal,
    write_seal,
)

DEFAULT_ADMIT_RULE = (
    "net dS = dS_rate + 100*(d_seg_new - d_seg_base) "
    "+ (sqrt(10*d_pose_new) - sqrt(10*d_pose_base)) < threshold"
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Every admission bar in this project is a small NEGATIVE number in scientific notation
    # (-3.5e-6 is the live sa1 bar). argparse's built-in negative-number matcher understands
    # "-1" and "-1.5" but not "-3.5e-6", so it classifies the value as an unknown option and
    # dies with "expected one argument". Widening the matcher is safe here because no option
    # string of this parser begins with a digit, which is the only case argparse's narrower
    # default protects. Caught by the producer control in test_candidate_seal.py — the
    # operator would otherwise have met it at seal time, by hand, which is the hazard.
    ap._negative_number_matcher = re.compile(r"^-\d+$|^-\d*\.\d+$|^-\d*\.?\d+[eE][+-]?\d+$")
    ap.add_argument("--candidate-id", required=True, help="the candidate's name; a placeholder refuses")
    ap.add_argument("--runtime-dir", required=True, help="the staged runtime tree that will be fired")
    ap.add_argument("--archive", default=None, help="default: <runtime-dir>/archive.zip")
    ap.add_argument("--axis", required=True, choices=list(SEAL_AXES))
    ap.add_argument("--out", required=True, help="where to write the seal JSON")
    ap.add_argument(
        "--receiver",
        action="append",
        default=None,
        help="runtime-relative path to pin per-file (repeatable; default: inflate.py and inflate.sh)",
    )
    ap.add_argument("--archive-member", default="", help="pin one member INSIDE the archive, e.g. 0.bin")
    ap.add_argument("--retained-path", action="append", default=[], help="retained payload custody (repeatable)")
    ap.add_argument("--falsifier", action="append", default=[], help="pre-registered falsifier (repeatable)")
    ap.add_argument("--admit-bar-net-ds", required=True, type=float, help="the net dS threshold to admit")
    ap.add_argument("--admit-bar-rule", default=DEFAULT_ADMIT_RULE)
    ap.add_argument("--pointer-axis", default="contest_cuda", choices=("contest_cuda", "contest_cpu", "effective"))
    ap.add_argument(
        "--bar-tolerance",
        type=float,
        default=0.0,
        help="how far the frontier pointer score may move before the bar is stale (default 0.0: any move refuses)",
    )
    ap.add_argument(
        "--allow-pointer-candidate-change",
        action="store_true",
        help="do NOT refuse when the frontier later points at a different candidate at a similar score "
        "(off by default: a delta is unanchored without its baseline)",
    )
    ap.add_argument(
        "--verify-archive-sha",
        default="",
        help="an expected archive sha you already hold; CHECKED against the measured bytes and refuses "
        "on mismatch. It is never stored as the value — the disk is.",
    )
    ap.add_argument(
        "--bound-base-receipt",
        default="",
        help="path to the BASE row's auth-eval receipt (MODAL_REMOTE_RESULT.json or the inner "
        "contest_auth_eval.json). When given, the seal pre-registers a COMPUTED report-8dp "
        "bound falsifier instead of a hand-typed one. Never pass a bound as a number: there "
        "is deliberately no flag for that.",
    )
    ap.add_argument("--sealed-by", default="MAIN")
    ap.add_argument("--notes", default="")
    ap.add_argument("--json", action="store_true", help="print the seal document to stdout")
    return ap


def compose_bound_falsifier(base_receipt_path: str) -> str:
    """Pre-register the report-8dp bound falsifier with COMPUTED numbers.

    WHY (rv13 F3 + F9, round-12 F1 + rv13 F2). Every bound that reached a seal or
    a memo by hand was wrong in one of three ways: divided by ONE row's bound
    when bounds ADD for a delta (2.00x overstatement, twice); listed one row's
    AXIS addends under a two-row total so they summed to half; or re-derived the
    pose bound from the rounded d_pose with the linearized form, disagreeing
    with the harness's own published field in the 4th significant figure.

    At seal time only the BASE row exists, so only its half can be stated. The
    sentence therefore states the base half, names the rule, and says plainly
    that the candidate's own bound ADDS once measured -- which is the fact whose
    absence caused all three defects. ``tools/report_8dp_delta_bound.py``
    composes the full two-row sentence at adjudication time.
    """
    from tac.report_8dp_bounds import row_bound_from_result

    path = Path(base_receipt_path)
    if not path.is_file():
        raise SealContractError(f"--bound-base-receipt is not a file: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealContractError(f"--bound-base-receipt is not readable JSON: {exc}") from exc
    row = row_bound_from_result(receipt, label="base")
    return (
        f"report-8dp bound (COMPUTED, source={row.source}): the base row's bound is "
        f"{row.total:.6e} = seg {row.seg:.6e} + pose {row.pose:.6e} (d_pose {row.d_pose:.6e}). "
        f"Bounds ADD for a DELTA, so the admissible margin must be judged against "
        f"base + candidate, NOT against this row alone -- dividing by one row's bound "
        f"overstates the margin by exactly 2.00x when the two rows' bounds are equal. "
        f"Compose the two-row sentence with tools/report_8dp_delta_bound.py once the "
        f"candidate row lands. Base receipt: {path}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    runtime_dir = Path(args.runtime_dir)
    archive_path = Path(args.archive) if args.archive else runtime_dir / "archive.zip"
    out_path = Path(args.out)

    try:
        pin = check_pin_consistency(runtime_dir, archive_path=archive_path)
        if not pin.ok:
            raise SealContractError(
                "candidate runtime pin-consistency preflight refused before seal creation: "
                f"{pin.summary()}"
            )
        measured = measure_archive_identity(archive_path)
        if args.verify_archive_sha and args.verify_archive_sha.strip().lower() != measured.sha256:
            print(
                "FATAL: --verify-archive-sha does not match the bytes on disk "
                f"(expected {args.verify_archive_sha.strip()[:16]}…, measured {measured.sha256[:16]}…). "
                "Sealing the measured bytes anyway would freeze the wrong candidate.",
                file=sys.stderr,
            )
            return 4

        pointer = read_pointer_state(axis=args.pointer_axis)
        bar = AdmitBar(
            rule=args.admit_bar_rule,
            net_dS_threshold=args.admit_bar_net_ds,
            pointer_axis=args.pointer_axis,
            pointer_score_at_seal=float(pointer["pointer_score"]),
            pointer_archive_sha256_at_seal=str(pointer["pointer_archive_sha256"]),
            pointer_tolerance_abs=args.bar_tolerance,
            require_pointer_archive_identity=not args.allow_pointer_candidate_change,
        )
        receivers = tuple(args.receiver) if args.receiver else ("inflate.py", "inflate.sh")
        falsifiers = tuple(args.falsifier)
        if args.bound_base_receipt:
            falsifiers = (*falsifiers, compose_bound_falsifier(args.bound_base_receipt))
        document = build_seal(
            candidate_id=args.candidate_id,
            runtime_dir=runtime_dir,
            archive_path=archive_path,
            axis=args.axis,
            admit_bar=bar,
            receiver_relative_paths=receivers,
            archive_member_name=args.archive_member,
            retained_payload_paths=tuple(args.retained_path),
            falsifiers=falsifiers,
            sealed_by=args.sealed_by,
            notes=args.notes,
        )
    except SealContractError as exc:
        print(f"FATAL: cannot seal this candidate: {exc}", file=sys.stderr)
        return 3

    write_seal(document, out_path)

    # The producer does not get to declare its own output good. It runs the CONSUMER's gate.
    verdict = validate_seal(out_path)
    if not verdict.ok:
        out_path.unlink(missing_ok=True)
        print(f"FATAL: the seal just written does not pass its own validator: {verdict.summary()}", file=sys.stderr)
        print("The seal was deleted; a seal that cannot be consumed is worse than none.", file=sys.stderr)
        return 5

    print(f"SEALED: {out_path}")
    print(f"  candidate   {document['candidate_id']} [{document['axis']}]")
    print(f"  archive     {document['archive']['bytes']:,} B sha {document['archive']['sha256'][:16]}…")
    print(
        f"  runtime     {document['runtime']['file_count']} files, "
        f"{document['runtime']['total_bytes']:,} B digest {document['runtime']['sha256'][:16]}…"
    )
    print(f"  receivers   {', '.join(pin['relative_path'] for pin in document['receiver_pins'])}")
    print(
        f"  admit bar   net dS < {bar.net_dS_threshold} vs {bar.pointer_axis} "
        f"{bar.pointer_score_at_seal:.8f} (tolerance {bar.pointer_tolerance_abs:g})"
    )
    print(f"  seal sha    {document['seal_sha256']}")
    print(f"  VALIDATED   {verdict.verdict}")
    print(f"\nFire it with:\n  .venv/bin/python tools/fire_modal_auth_eval.py --seal {out_path} \\")
    print("      --output-dir <dir> --lane-id <lane> --instance-job-id <job>")
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
