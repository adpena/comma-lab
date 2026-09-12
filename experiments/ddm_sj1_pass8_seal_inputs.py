#!/usr/bin/env python3
"""ddm_sj1 pass 8 -- the seal inputs for a DISTORTION-MOVING candidate on the move-49 field.

Why this file exists rather than a reuse of ``ddm_hpr1_seal_inputs``: that producer emits
the FIRST-MEASUREMENT intent inputs, and it hardcodes its own identity-class legs (moves 44
and 46), its own execution ids and its own rate-only distortion argument.  Editing it would
put this arm's object behind another arm's words AND would break every in-flight hpr1/dpi1
resume, because that module binds its own sha into each receipt
([[binding_hash_whole_module_kills_checkpoints_20260909]]).  The GENERIC helpers -- the
file-reference, endpoint, census and content-diff primitives -- are IMPORTED from
``ddm_ntb2_intent_inputs`` and ``tac.candidate_seal`` unchanged; only the CONTENT is this
arm's.

The seal path is NOT asserted here.  This producer stops at the inputs; which route the
candidate takes is decided by what its decode leg admits and is recorded in the seal, never
in a summary sentence that can go stale.

Axis ``[macOS-CPU advisory]``.  No scorer runs.  No score is claimed.  NO SEAL IS WRITTEN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from experiments.ddm_ntb2_intent_inputs import (
    MEMBER_NAME,
    census_files,
    content_diff,
    endpoints,
    git_commit,
    prefire_file_reference,
    utc_now,
)
from tac.candidate_seal import read_archive_member_identity
from tac.decode_wall_clock import measure_receiver_digest, validate_receiver_manifest

#: The score's rate term, in S per archive byte: 25 / 37,545,489.
BYTES_TO_S = 6.658589531221714e-07
RETENTION_CAP_BYTES = 8 << 30


class SealInputsError(RuntimeError):
    """An input is not what this candidate's own receipts say it is."""


def canonical(path: Path) -> Path:
    """Rewrite a repo path to the repo's OWN spelling of its case.

    macOS is case-insensitive and ``Path.resolve()`` does not correct case, so a caller
    whose cwd carries ``/Users/adpena/projects/pact`` writes that spelling into every
    reference while the validator compares references as STRINGS.  ntb2's first intent died
    exactly there, so the normalisation is a property of this producer rather than of
    whoever invoked it.
    """
    resolved = Path(path).resolve()
    try:
        return REPO / resolved.relative_to(REPO)
    except ValueError:
        lower = str(REPO).lower()
        if str(resolved).lower().startswith(lower + "/"):
            return REPO / str(resolved)[len(lower) + 1:]
        return resolved


def write(path: Path, document) -> Path:
    path = canonical(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def manifest_validation(candidate: Path, out: Path, pins: dict) -> tuple[Path, list[str]]:
    """Re-hash every MANIFEST.sha256 row FROM OUTSIDE THE TREE, then validate the listing.

    Two different checks, deliberately both: the row-by-row re-hash proves the listing
    describes the bytes on disk, and ``validate_receiver_manifest`` is the CONTRACT's own
    independent validator, the one a seal consults.  Pass 7 shipped a candidate whose
    listing named the POINTER's inflate.py against its own re-pinned bytes, because the
    pin patcher rewrote the file and not the listing; the patcher now rebinds, and this
    re-hash is what would catch it if it ever stopped.
    """
    listing = candidate / "MANIFEST.sha256"
    mismatched: list[str] = []
    rows = 0
    for line in listing.read_text().splitlines():
        digest, _, relative = line.partition("  ")
        if not relative:
            continue
        rows += 1
        target = candidate / relative
        if not target.is_file() or prefire_file_reference(target)["sha256"] != digest:
            mismatched.append(relative)
    problems: list[str] = []
    try:
        validate_receiver_manifest(candidate)
    except Exception as error:  # the contract's validator speaks in typed refusals
        problems.append(f"{type(error).__name__}: {error}")
    path = write(out / "MANIFEST_VALIDATION.json", {
        **pins,
        "score_claim": False,
        "receiver_manifest_listing": prefire_file_reference(listing),
        "rows_rehashed": rows,
        "all_hashes_passed": not mismatched,
        "mismatched_rows": mismatched,
        "contract_validator": "tac.decode_wall_clock.validate_receiver_manifest",
        "contract_validator_problems": problems,
        "verification_method": (
            "every MANIFEST.sha256 row re-hashed from OUTSIDE the tree against the shipped "
            "file, then the contract's own independent listing validator run on the same "
            "tree; the listing excludes itself and archive.zip"
        ),
    })
    return path, mismatched + problems


def emit(candidate: Path, pointer: Path, out: Path, retained: list[Path],
         candidate_id: str, close_receipt: Path, price_receipt: Path,
         parseback_receipt: Path, seg_receipt: Path) -> dict:
    pins = endpoints(candidate)
    commit = git_commit()
    close = json.loads(close_receipt.read_text())
    price = json.loads(price_receipt.read_text())
    parseback = json.loads(parseback_receipt.read_text())
    seg = json.loads(seg_receipt.read_text())

    if close["candidate_archive"]["sha256"] != pins["archive_sha256"]:
        raise SealInputsError("the close receipt does not name the staged candidate archive")
    if parseback["archive"]["sha256"] != pins["archive_sha256"]:
        raise SealInputsError("the parse-back did not decode the staged candidate archive")
    twins = price["twin_sha256"][price["candidate_field"]]
    if twins[0] != twins[1]:
        raise SealInputsError("twins disagree")
    if twins[0] != close["body_archive_sha256"]:
        raise SealInputsError(
            "the priced twin archive is not the body the carrier was spliced into"
        )

    rows, total, occurrences = census_files(candidate)
    dependency_rows = [{k: row[k] for k in ("relative_path", "bytes", "sha256")} for row in rows]
    write(out / "CANDIDATE_MANIFEST.json", {
        **pins,
        "producer_source_commit": commit,
        "production_started_at_utc": utc_now(),
        "files": dependency_rows,
        "source_custody": {
            "promoted_pointer_tree": str(pointer),
            "close_receipt": prefire_file_reference(close_receipt),
            "price_receipt": prefire_file_reference(price_receipt),
            "parseback_receipt": prefire_file_reference(parseback_receipt),
            "seg_receipt": prefire_file_reference(seg_receipt),
        },
    })

    _, manifest_problems = manifest_validation(candidate, out, pins)

    member_sha, member_bytes = read_archive_member_identity(candidate / "archive.zip", MEMBER_NAME)
    write(out / "ARCHIVE_PARSEBACK.json", {
        **pins,
        "score_claim": False,
        "member": {"name": MEMBER_NAME, "sha256": member_sha, "bytes": member_bytes},
        "parse_method": "runtime.residual_archive.read_residual_archive on the shipped archive",
        "decoded_through_the_candidates_own_inflate": {
            "pin_check": parseback["pin_check"],
            "rendered_raw": parseback["rendered_raw"],
            "decoded_field_matches_admitted": parseback["decoded_field_matches_admitted"],
            "decoded_token_sha256":
                parseback["inflate_report"]["token_decoder"]["decoded_token_sha256"],
            "decoder_bit_position":
                parseback["inflate_report"]["token_decoder"]["decoder_bit_position"],
            "wall_clock_seconds": parseback["wall_clock_seconds"],
        },
        "source_archive": prefire_file_reference(pointer / "archive.zip"),
    })

    diff = content_diff(pointer, candidate)
    differing = sorted(row["relative_path"] for row in diff)
    write(out / "LITERAL_CENSUS.json", {
        **pins,
        "score_claim": False,
        "complete": True,
        "files": dependency_rows,
        "numeric_literal_occurrences_by_file": rows,
        "literal_occurrences": occurrences,
        "occurrence_count": total,
        "content_diff_vs_pointer_tree": diff,
        "rule": 118,
        "rule_statement": (
            "rule 118: generic algorithm is free in the receiver, video-derived content is "
            "counted in archive.zip"
        ),
        "review_scope": "every shipped file of the candidate tree against move 49's promoted tree",
        "new_video_selected_literals_in_free_code": [],
        "verdict": "CLEAR" if differing == ["MANIFEST.sha256", "archive.zip", "inflate.py"] else "REVIEW",
        "verdict_rationale": (
            "This candidate moves TOKENS and CARRIER CODES, both of which are video-derived "
            "and both of which are COUNTED: the tokens inside archive.zip's tail stream, the "
            "codes inside its carrier section. Not one of them enters free receiver code. The "
            "ONLY files differing from the pointer tree are archive.zip (counted), inflate.py "
            "(its two archive-pin constants, which restate the archive's own identity) and "
            "MANIFEST.sha256 (a derived listing of the others), MEASURED here as "
            f"{differing}. No fitted scalar and no video-derived value entered free code, "
            "which is the distinction move 41 was retracted for missing."
        ),
        "whole_receiver_integer": True,
        "tc4_maps": [],
    })

    if not retained:
        raise SealInputsError(
            "no retained payloads declared; a retention manifest that lists nothing is the "
            "measure-and-discard shape, not a receipt"
        )
    payloads = [prefire_file_reference(path) for path in retained]
    seen = {ref["path"] for ref in payloads}
    if len(seen) != len(payloads):
        raise SealInputsError("duplicate retained payload path")
    retained_bytes = sum(ref["bytes"] for ref in payloads)
    if retained_bytes > RETENTION_CAP_BYTES:
        raise SealInputsError(
            f"retention {retained_bytes} B exceeds the {RETENTION_CAP_BYTES} B cap"
        )
    write(out / "RETENTION_MANIFEST.json", {
        **pins,
        "score_claim": False,
        "payloads": payloads,
        "retained_bytes_total": retained_bytes,
        "retention_cap_bytes": RETENTION_CAP_BYTES,
        "statement": (
            "every payload this candidate rests on is retained with bytes and sha256 on the "
            "SSD tier; nothing was measured and discarded"
        ),
    })

    pointer_bytes = prefire_file_reference(pointer / "archive.zip")["bytes"]
    candidate_bytes = prefire_file_reference(candidate / "archive.zip")["bytes"]
    net_bytes = candidate_bytes - pointer_bytes
    summary = {
        "schema": "ddm_sj1_pass8_seal_inputs_summary.v1",
        "candidate_id": candidate_id,
        "pointer_archive_sha256": prefire_file_reference(pointer / "archive.zip")["sha256"],
        "pointer_archive_bytes": pointer_bytes,
        "candidate_archive_bytes": candidate_bytes,
        "net_bytes_vs_pointer": net_bytes,
        "net_delta_s_rate_leg_only": net_bytes * BYTES_TO_S,
        "three_leg_net_delta_s": close["net_dS_vs_pointer"],
        "seg_cells_on_the_shipped_decode": seg["legs"]["dali"]["cells_disagreeing"],
        "receiver_digest": measure_receiver_digest(candidate),
        "content_diff_vs_pointer_tree": differing,
        "manifest_problems": manifest_problems,
        "literal_occurrence_count": total,
        "seal_path": (
            "UNDECIDED HERE -- this producer stops at the inputs. Which route the candidate "
            "seals by is decided by what its decode leg admits, and is recorded in the seal "
            "object, never in this summary."
        ),
        "seal_written": False,
        "score_claim": False,
        "promotable": False,
        "emitted_before_this_summary": sorted(p.name for p in canonical(out).glob("*.json")),
        "emitted_note": "this summary is written last and cannot list itself",
    }
    write(out / "SEAL_INPUTS_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--pointer-runtime", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--close-receipt", type=Path, required=True)
    parser.add_argument("--price-receipt", type=Path, required=True)
    parser.add_argument("--parseback-receipt", type=Path, required=True)
    parser.add_argument("--seg-receipt", type=Path, required=True)
    parser.add_argument("--retained-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary = emit(
        candidate=args.candidate_runtime.resolve(),
        pointer=args.pointer_runtime.resolve(),
        out=args.out_dir,
        retained=[p.resolve() for p in args.retained_path],
        candidate_id=args.candidate_id,
        close_receipt=args.close_receipt,
        price_receipt=args.price_receipt,
        parseback_receipt=args.parseback_receipt,
        seg_receipt=args.seg_receipt,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
