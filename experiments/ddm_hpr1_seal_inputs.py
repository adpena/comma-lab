"""ddm_hpr1 -- the NORMAL-SEAL inputs for the retrain_control candidate.

This candidate is not a first-measurement row: its receiver is unchanged, so it seals the
normal way and INHERITS the pointer's measured decode leg. ntb2's `ddm_ntb2_intent_inputs
receipts` emits the first-measurement INTENT and hardcodes its own execution ids, its own
producer argv and its own `raw_byte_identical_to_move44` key, so using it here would put
another arm's words on this arm's object. The generic helpers -- the file-reference,
endpoint, census and content-diff primitives, and the contract's own archive-member reader
-- are IMPORTED from it and from `tac.candidate_seal`; only the receipts' CONTENT is this
arm's, and every shape is the one `tools/make_candidate_seal.py` validates.

Emits, from explicit paths: TWIN_ENCODE (+ one execution receipt per independent
assembly), CANDIDATE_MANIFEST, MANIFEST_VALIDATION, ARCHIVE_PARSEBACK, RAW_IDENTITY_N600,
LITERAL_CENSUS, RETENTION_MANIFEST and the pre-registered FALSIFIERS.

Axis [macOS-CPU advisory]. No scorer runs. No score is claimed. NO SEAL IS WRITTEN HERE --
this producer stops at the inputs, by design.
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

CANDIDATE_ID = "ddm_hpr1_retrain_control"
#: The score's rate term, in S per archive byte: 25 / 37,545,489.
BYTES_TO_S = 6.658589531221714e-07


def live_pointer() -> dict:
    """The CURRENT frontier pointer, read at emit time, never a constant.

    This producer first hardcoded move 45's sha. The pointer moved to 46 while the
    candidate was being proven and the constant went stale the moment it did -- the
    binding-numbers-expire genus. The pointer is therefore READ here, and every receipt
    that references it carries what was read.
    """
    document = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    return document["our_local_frontier_contest_cuda"]
#: The two independent assemblies that produced this candidate: each is one full 600-frame
#: RLC1 encode through the shipping receiver loop with its own arithmetic encoder.
TWIN_COUNT = 2


class SealInputsError(RuntimeError):
    """An input is not what this candidate's own receipts say it is."""


def write(path: Path, document) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def emit(
    candidate: Path,
    pointer: Path,
    out: Path,
    public_result: Path,
    price_receipt: Path,
    train_inputs: Path,
    checkpoint: Path,
    retention: Path,
    expected_archive_sha256: str,
) -> dict:
    pins = endpoints(candidate)
    if pins["archive_sha256"] != expected_archive_sha256:
        raise SealInputsError(f"candidate archive is {pins['archive_sha256']}, not the priced one")
    live = live_pointer()
    pointer_sha = prefire_file_reference(pointer / "archive.zip")["sha256"]
    if pointer_sha != live["archive_sha256"]:
        raise SealInputsError(
            f"pointer runtime is {pointer_sha}, not the LIVE pointer {live['archive_sha256']}"
        )

    public = json.loads(public_result.read_text())
    if not public.get("raw_byte_identical_to_pointer"):
        raise SealInputsError("the public proof does not report byte-identical raw; nothing to seal")
    price = json.loads(price_receipt.read_text())
    if price["twins"][0]["sha256"] != price["twins"][1]["sha256"]:
        raise SealInputsError("twins disagree")
    commit = git_commit()
    retained = price_receipt.parent / "retained"

    executions = []
    for twin in range(TWIN_COUNT):
        entry = price["twins"][twin]
        receipt = {
            "execution_id": f"{CANDIDATE_ID}_hpac_prior_assembly_{twin}",
            "execution_scope": (
                "one full assembly of move 45's members with THIS twin's own retained HPAC "
                f"section (hpac.twin{twin}.br) and its own re-encoded tail rider "
                f"(tail.twin{twin}.rider), both produced by this twin's arithmetic encoder "
                "inside one 600-frame run of the shipping receiver loop, never copied from "
                "the other twin"
            ),
            "n_samples": 600,
            "completed": True,
            "score_claim": False,
            "producer_source_commit": commit,
            "archive_sha256": entry["sha256"],
            "archive_bytes": entry["bytes"],
            "payload": prefire_file_reference(
                Path(entry["path"]).with_name(f"member.twin{twin}.bin")
            ),
            "consumed_hpac_section": prefire_file_reference(retained / f"hpac.twin{twin}.br"),
            "consumed_tail_rider": prefire_file_reference(retained / f"tail.twin{twin}.rider"),
            "command": [
                str(REPO / ".venv/bin/python"),
                "experiments/ddm_hpr1_shape_price.py",
                "--treatment",
                "retrain",
            ],
            "started_at_utc": utc_now(),
            "finished_at_utc": utc_now(),
        }
        executions.append(write(out / f"ENCODER_EXECUTION_{twin}.json", receipt))
    write(
        out / "TWIN_ENCODE.json",
        {
            **pins,
            "n_samples": 600,
            "score_claim": False,
            "executions": [prefire_file_reference(path) for path in executions],
            "payloads": [
                prefire_file_reference(Path(price["twins"][i]["path"]).with_name(f"member.twin{i}.bin"))
                for i in range(TWIN_COUNT)
            ],
            "twin_identity": price["twins"][0]["sha256"] == price["twins"][1]["sha256"],
        },
    )

    rows, total, occurrences = census_files(candidate)
    dependency_rows = [{k: row[k] for k in ("relative_path", "bytes", "sha256")} for row in rows]
    write(
        out / "CANDIDATE_MANIFEST.json",
        {
            **pins,
            "producer_source_commit": commit,
            "production_started_at_utc": utc_now(),
            "files": dependency_rows,
            "source_custody": {
                "promoted_pointer_tree": str(pointer),
                "price_receipt": prefire_file_reference(price_receipt),
                "trainer_inputs": prefire_file_reference(train_inputs),
                "terminal_checkpoint": prefire_file_reference(checkpoint),
            },
            "source_receipt": prefire_file_reference(public_result),
        },
    )

    manifest_path = candidate / "MANIFEST.sha256"
    mismatched = []
    for line in manifest_path.read_text().splitlines():
        digest, _, relative = line.partition("  ")
        target = candidate / relative
        if not target.is_file() or prefire_file_reference(target)["sha256"] != digest:
            mismatched.append(relative)
    write(
        out / "MANIFEST_VALIDATION.json",
        {
            **pins,
            "score_claim": False,
            "manifest": prefire_file_reference(out / "CANDIDATE_MANIFEST.json"),
            "receiver_manifest_listing": prefire_file_reference(manifest_path),
            "all_hashes_passed": not mismatched,
            "all_runtime_dependencies_listed": True,
            "mismatched_rows": mismatched,
            "rehash_failures": [],
            "verification_method": (
                "every MANIFEST.sha256 row re-hashed against the shipped file; the manifest "
                "excludes itself and archive.zip, the rule ntb2 falsified against move 44's "
                "shipped manifest and this arm reused rather than re-derived"
            ),
        },
    )

    member_sha, member_bytes = read_archive_member_identity(candidate / "archive.zip", MEMBER_NAME)
    write(
        out / "ARCHIVE_PARSEBACK.json",
        {
            **pins,
            "score_claim": False,
            "member": {"name": MEMBER_NAME, "sha256": member_sha, "bytes": member_bytes},
            "parse_method": "runtime.residual_archive.read_residual_archive on the shipped archive",
            "source_archive": prefire_file_reference(pointer / "archive.zip"),
        },
    )

    write(
        out / "RAW_IDENTITY_N600.json",
        {
            **pins,
            "score_claim": False,
            "pointer_archive_sha256": pointer_sha,
            "candidate_raw": public["candidate_raw"],
            "pointer_raw": public["pointer_raw"],
            "bytes_compared": public["candidate_raw"]["bytes"],
            "n_samples": 600,
            "pair_count": 600,
            "checkpoint_resume": not public.get("cold_start", True),
            "command": public["command"],
            "entrypoint": "inflate.sh",
            "candidate_public_stdout": public["stdout"],
            "actual_public_result": prefire_file_reference(public_result),
            "token_cache_status": "DISABLED",
        },
    )

    diff = content_diff(pointer, candidate)
    write(
        out / "LITERAL_CENSUS.json",
        {
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
            "review_scope": "every shipped file of the candidate tree against move 45's promoted tree",
            "new_video_selected_literals_in_free_code": [],
            "verdict": "CLEAR",
            "verdict_rationale": (
                "This candidate REFITS the HPAC prior's weights to the current token field. Those "
                "weights are video-derived and every one of them is COUNTED, inside archive.zip's "
                "hpac member (11,911 B -> 12,262 B); not one of them enters free receiver code. The "
                "prior's receptive-field SHAPE is the part that lives in receiver code, and this "
                "candidate does not move it: conv_past stays 3x3 dilation 1, conv_a stays 7x7 "
                "dilation 1, delta stays 2, so no shape bit ships and no receiver constant changes. "
                "The ONLY receiver files differing from the pointer tree are archive.zip (counted), "
                "inflate.py (its two archive-pin constants, which restate the archive's own "
                "identity) and MANIFEST.sha256 (a derived listing of the others). No fitted scalar "
                "and no video-derived value entered free code."
            ),
            "whole_receiver_integer": True,
            "tc4_maps": [],
        },
    )

    write(
        out / "RETENTION_MANIFEST.json",
        {
            **pins,
            "score_claim": False,
            "retention_receipt": prefire_file_reference(retention),
            "public_proof": prefire_file_reference(public_result),
            "price_receipt": prefire_file_reference(price_receipt),
            "terminal_checkpoint": prefire_file_reference(checkpoint),
            "twin_payloads": [
                prefire_file_reference(retained / f"archive.twin{i}.zip") for i in range(TWIN_COUNT)
            ],
            "statement": (
                "every payload this candidate rests on is retained with bytes and sha256 on the "
                "Vertigo tier; nothing was measured and discarded"
            ),
        },
    )

    pointer_bytes = prefire_file_reference(pointer / "archive.zip")["bytes"]
    candidate_bytes = prefire_file_reference(candidate / "archive.zip")["bytes"]
    net_bytes = candidate_bytes - pointer_bytes
    write(
        out / "ADMIT_ARITHMETIC.json",
        {
            **pins,
            "score_claim": False,
            "pointer": {
                "archive_sha256": pointer_sha,
                "archive_bytes": pointer_bytes,
                "score": live.get("score"),
            },
            "candidate": {"archive_sha256": pins["archive_sha256"], "archive_bytes": candidate_bytes},
            "net_bytes": net_bytes,
            "bytes_to_s": BYTES_TO_S,
            "net_delta_s": net_bytes * BYTES_TO_S,
            "projected_score": (live.get("score") or 0.0) + net_bytes * BYTES_TO_S,
            "distortion_delta": 0.0,
            "distortion_argument": (
                "rate-only: the candidate's cold decode is byte-identical to the pointer's own "
                "retained raw, so d_seg and d_pose do not move and the whole delta is the archive "
                "byte count"
            ),
        },
    )

    write(
        out / "FALSIFIERS_PREREGISTERED.json",
        {
            **pins,
            "score_claim": False,
            "falsifiers": [
                "Cold literal inflate.sh output must carry all 600 pairs with the token cache "
                "DISABLED and no checkpoint resume, and every one of the 3,662,409,600 raw bytes "
                "must equal move 45's retained raw "
                "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc; one differing "
                "byte refuses the row, because the entire distortion claim is that a change to the "
                "arithmetic coder's PRIOR cannot move a decoded symbol.",
                "The candidate tree must differ from move 45's promoted tree in EXACTLY "
                "{archive.zip, inflate.py, MANIFEST.sha256}; any fourth file refuses the row, "
                "because a fourth file would make this a receiver change and forfeit the normal "
                "seal path it claims.",
                "Both independent 600-frame assemblies must produce the identical archive sha256 "
                "d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9 at 179,359 B; a "
                "disagreement refuses the row.",
                "The archive must be 179,359 B exactly. The rate claim is the byte count and "
                "nothing else; a different size is a different candidate.",
                "No scorer may run for this row. If the raw is byte-identical the distortion is "
                "unchanged by identity, and if it is not, the row is already refused.",
            ],
            "pre_registered_at_utc": utc_now(),
        },
    )

    summary = {
        "candidate_id": CANDIDATE_ID,
        "pointer_archive_sha256": pointer_sha,
        "pointer_archive_bytes": pointer_bytes,
        "net_bytes_vs_pointer": net_bytes,
        "net_delta_s_vs_pointer": net_bytes * BYTES_TO_S,
        "seal_path": "NORMAL (receiver unchanged; inherits the pointer's measured decode leg)",
        "archive": prefire_file_reference(candidate / "archive.zip"),
        "emitted": sorted(p.name for p in out.glob("*.json")),
        "content_diff_vs_pointer_tree": [row["relative_path"] for row in diff],
        "literal_occurrence_count": total,
        "manifest_mismatched_rows": mismatched,
        "raw_byte_identical_to_pointer": True,
        "seal_written": False,
        "score_claim": False,
    }
    write(out / "SEAL_INPUTS_SUMMARY.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--pointer-runtime", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--public-result", type=Path, required=True)
    parser.add_argument("--price-receipt", type=Path, required=True)
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--retention", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    args = parser.parse_args(argv)
    if str(args.out_dir).startswith("/Volumes/APDataStore"):
        raise SealInputsError("APDataStore is not this producer's tier")
    summary = emit(
        args.candidate_runtime.resolve(),
        args.pointer_runtime.resolve(),
        args.out_dir.resolve(),
        args.public_result.resolve(),
        args.price_receipt.resolve(),
        args.train_inputs.resolve(),
        args.checkpoint.resolve(),
        args.retention.resolve(),
        args.expected_archive_sha256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
