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
from tac.decode_wall_clock import (  # noqa: E402
    inherit_decode_wall_clock,
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
    ap.add_argument("--candidate-id", help="the candidate's name; a placeholder refuses")
    ap.add_argument("--runtime-dir", help="the staged runtime tree that will be fired")
    ap.add_argument("--archive", default=None, help="default: <runtime-dir>/archive.zip")
    ap.add_argument("--axis", choices=list(SEAL_AXES))
    ap.add_argument("--out", required=True, help="where to write the seal JSON")
    ap.add_argument(
        "--receiver",
        action="append",
        default=None,
        help="runtime-relative path to pin per-file (repeatable; default: inflate.py and inflate.sh)",
    )
    ap.add_argument("--archive-member", default="", help="pin one member INSIDE the archive, e.g. 0.bin")
    ap.add_argument(
        "--public-entrypoint-smoke",
        help="JSON receipt block (or JSON document containing that block) with candidate/frontier "
        "public_path_probes and inflate_sh_smokes",
    )
    timing = ap.add_mutually_exclusive_group(required=True)
    timing.add_argument(
        "--decode-wall-clock", help="JSON containing the measured decode_wall_clock leg",
    )
    timing.add_argument(
        "--inherit-decode-wall-clock",
        help="pointer's measured leg; revalidates identical receiver code and timing before inheritance",
    )
    timing.add_argument("--first-fire-intent", action="store_true")
    timing.add_argument("--complete-first-fire-intent")
    for flag in ("candidate-manifest", "manifest-validation", "twin-encode-receipt",
                 "archive-parseback-receipt", "raw-identity-receipt", "literal-census",
                 "retention-manifest", "timing-risk-evidence", "first-measurement-authorization",
                 "candidate-t4-receipt"):
        ap.add_argument("--" + flag)
    ap.add_argument("--retained-path", action="append", default=[], help="retained payload custody (repeatable)")
    ap.add_argument("--falsifier", action="append", default=[], help="pre-registered falsifier (repeatable)")
    ap.add_argument("--admit-bar-net-ds", type=float, help="the net dS threshold to admit")
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


def _prefire_main(args: argparse.Namespace, supplied: list[str]) -> int:
    from tac.candidate_seal import (
        PrefireRefusal,
        build_prefire_intent,
        complete_first_fire_intent,
        write_prefire_refusal,
    )
    paths = [Path(args.out)]
    output_dir = None
    try:
        flags = {value.split("=", 1)[0] for value in supplied if value.startswith("--")}
        if args.complete_first_fire_intent:
            paths.append(Path(args.complete_first_fire_intent))
            allowed = {"--complete-first-fire-intent", "--first-measurement-authorization", "--candidate-t4-receipt", "--out"}
            if flags - allowed or not args.first_measurement_authorization or not args.candidate_t4_receipt:
                raise PrefireRefusal("FIRST_MEASUREMENT_ARGUMENT_REFUSED", "completion accepts only intent, authorization, receipt, and new output")
            paths.append(Path(args.first_measurement_authorization))
            try:
                from tac.candidate_seal import _pf_output
                auth = json.loads(paths[2].read_text())
                output_dir = _pf_output(auth.get("output_dir"))
            except (OSError, ValueError, AttributeError, PrefireRefusal):
                pass  # The authoritative completion validator below emits the typed refusal.
            document = complete_first_fire_intent(intent_path=paths[1], authorization_path=paths[2],
                receipt_path=Path(args.candidate_t4_receipt), out_path=paths[0])
        else:
            mapping = {"candidate_manifest": "candidate_manifest", "manifest_validation": "manifest_validation",
                "twin_encode": "twin_encode_receipt", "archive_parseback": "archive_parseback_receipt",
                "raw_identity_n600": "raw_identity_receipt", "literal_census": "literal_census",
                "retention_manifest": "retention_manifest", "timing_risk": "timing_risk_evidence"}
            allowed = {"--first-fire-intent", "--candidate-id", "--runtime-dir", "--axis", "--public-entrypoint-smoke",
                "--admit-bar-net-ds", "--pointer-axis", "--bar-tolerance", "--retained-path", "--falsifier", "--out",
                *("--" + name.replace("_", "-") for name in mapping.values())}
            if flags - allowed or args.axis != "contest_cuda" or args.pointer_axis != "contest_cuda" or args.bar_tolerance != 0:
                raise PrefireRefusal("FIRST_MEASUREMENT_ARGUMENT_REFUSED", "intent accepts only frozen CUDA contract flags")
            if any(getattr(args, name) is None for name in (*mapping.values(), "candidate_id", "runtime_dir",
                    "public_entrypoint_smoke", "admit_bar_net_ds")):
                raise PrefireRefusal("PREFIRE_NON_TIMING_GATE_REFUSED", "all non-timing evidence flags required")
            smoke = json.loads(Path(args.public_entrypoint_smoke).read_text())
            document = build_prefire_intent(candidate_id=args.candidate_id, runtime_dir=Path(args.runtime_dir),
                evidence_paths={key: Path(getattr(args, name)) for key, name in mapping.items()},
                public_entrypoint_smoke=smoke.get("public_entrypoint_smoke", smoke), net_ds_threshold=args.admit_bar_net_ds,
                retained_paths=[str(Path(p).resolve()) for p in args.retained_path], falsifiers=args.falsifier, out_path=paths[0])
        print(f"CREATED: {args.out} ({document['schema']})")
        return 0
    except (PrefireRefusal, OSError, ValueError, KeyError, TypeError, AttributeError, SealContractError) as exc:
        code = "FIRST_MEASUREMENT_RESULT_REFUSED" if args.complete_first_fire_intent else "PREFIRE_NON_TIMING_GATE_REFUSED"
        refusal = exc if isinstance(exc, PrefireRefusal) else PrefireRefusal(code, str(exc))
        write_prefire_refusal(refusal, paths=tuple(paths), output_dir=output_dir)
        return 3


def main(argv: list[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    is_prefire = any(s.split("=", 1)[0] in {"--first-fire-intent", "--complete-first-fire-intent"} for s in supplied)
    if is_prefire:
        from tac.candidate_seal import PrefireRefusal, write_prefire_refusal
        def argument_refusal(message):
            raise PrefireRefusal("FIRST_MEASUREMENT_ARGUMENT_REFUSED", message)
        parser.error = argument_refusal
        try:
            args = parser.parse_args(supplied)
        except PrefireRefusal as exc:
            paths = tuple(Path(supplied[i + 1]) for i, flag in enumerate(supplied[:-1])
                          if flag in {"--out", "--complete-first-fire-intent", "--first-measurement-authorization"}
                          and not supplied[i + 1].startswith("--"))
            write_prefire_refusal(exc, paths=paths)
            return 3
    else:
        args = parser.parse_args(supplied)
    if args.first_fire_intent or args.complete_first_fire_intent:
        return _prefire_main(args, supplied)
    for name in ("candidate_id", "runtime_dir", "axis", "public_entrypoint_smoke", "admit_bar_net_ds"):
        if getattr(args, name) is None:
            build_parser().error("--" + name.replace("_", "-") + " is required for a normal seal")

    runtime_dir = Path(args.runtime_dir)
    archive_path = Path(args.archive) if args.archive else runtime_dir / "archive.zip"
    out_path = Path(args.out)

    try:
        smoke_path = Path(args.public_entrypoint_smoke)
        if not smoke_path.is_file():
            raise SealContractError(f"--public-entrypoint-smoke is not a file: {smoke_path}")
        try:
            smoke_payload = json.loads(smoke_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SealContractError(f"--public-entrypoint-smoke is not readable JSON: {exc}") from exc
        if not isinstance(smoke_payload, dict):
            raise SealContractError("--public-entrypoint-smoke JSON must be an object")
        public_entrypoint_smoke = smoke_payload.get("public_entrypoint_smoke", smoke_payload)
        if not isinstance(public_entrypoint_smoke, dict):
            raise SealContractError("public_entrypoint_smoke must be an object")

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
        if args.inherit_decode_wall_clock:
            decode_wall_clock = inherit_decode_wall_clock(
                source_leg_path=Path(args.inherit_decode_wall_clock),
                runtime_dir=runtime_dir, archive_path=archive_path,
                pointer_archive_sha256=str(pointer["pointer_archive_sha256"]),
            )
        else:
            try:
                timing_payload = json.loads(Path(args.decode_wall_clock).read_text())
            except (OSError, ValueError) as exc:
                raise SealContractError(f"cannot read --decode-wall-clock: {exc}") from exc
            if not isinstance(timing_payload, dict):
                raise SealContractError("--decode-wall-clock JSON must be an object")
            decode_wall_clock = timing_payload.get("decode_wall_clock", timing_payload)
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
            public_entrypoint_smoke=public_entrypoint_smoke,
            decode_wall_clock=decode_wall_clock,
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
    verdict = validate_seal(out_path, require_decode_wall_clock=True)
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
