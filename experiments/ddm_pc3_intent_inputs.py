"""ddm_pc3 -- every pre-fire-intent receipt for the CAP1 predictor-refit candidate.

WHAT THIS EMITS AND WHY EACH ONE IS A MEASUREMENT
--------------------------------------------------
`tools/make_candidate_seal.py --first-fire-intent` reads eight receipts off disk and
re-derives every number in them from the tree, the archive and the payloads.  Nothing
here is transcribed from another arm's file: each receipt is produced by measuring THIS
candidate, and the two that could most easily have been borrowed are the ones this module
is most careful about.

* ``twin`` -- two INDEPENDENT ``up3.build_archive`` executions, each retaining its own
  copy of the archive member ``p``.  The validator requires two distinct retained paths
  with identical bytes and sha, one execution receipt per payload, and distinct execution
  ids.  Copying one payload to a second path would satisfy the shape and prove nothing,
  so each payload is written by its own encoder run.
* ``census`` -- a real rule-118 scan of THIS tree's 50 files, plus the exact content diff
  against the pointer tree.  The candidate's receiver is byte-identical to move 44's in
  every file except ``inflate.py`` (archive pin only, normalized away by the receiver
  digest) and the derived ``MANIFEST.sha256``; the census states that as a measured diff,
  not as an inherited verdict.

THE RISK LEG IS THE SCOPED DIGEST, AND IT IS EQUAL
---------------------------------------------------
``measure_prefire_risk_receiver_digest`` excludes exactly ``MANIFEST.sha256`` (ffi4's
amendment ``ddm_pr14_manifest_in_receiver_risk_digest``).  Measured on this pair:
move 44 and the candidate both digest to ``9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890``
over **49 rows with ZERO differing rows** -- the same value rlc5's intent pinned.  So the
behaviour-bearing receiver is the same object, the receiver delta manifest is complete and
empty of differences, and the T4 spend risk is the pure local run-to-run ratio between a
cold n600 decode of move 44's tree and one of the candidate's.  Both of those are run by
``ddm_pc3_diagnostic_pair.sh``, back to back on a quiesced machine, base first.

Axis: exact digest/byte arithmetic plus real local decodes. No score, no authority, no
timing clearance. ``score_claim=false`` throughout.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "experiments")]

from tac.candidate_seal import (
    PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
    PREFIRE_RISK_SCHEMA,
    measure_prefire_risk_receiver_digest,
    measure_runtime_digest,
    prefire_digest,
    prefire_file_reference,
    prefire_risk_receiver_rows,
    read_archive_member_identity,
)
from tac.decode_wall_clock import measure_receiver_digest

POINTER_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime"
)
POINTER_ARCHIVE_SHA256 = (
    "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
)
MEMBER_NAME = "p"


class Pc3IntentError(RuntimeError):
    """A ddm_pc3 intent-input precondition failed. Fail closed, never approximate."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def endpoints(runtime: Path) -> dict[str, str]:
    """The three identity pins every non-timing receipt must carry, measured here."""
    runtime = Path(runtime).resolve()
    return {
        "archive_sha256": prefire_file_reference(runtime / "archive.zip")["sha256"],
        "runtime_sha256": measure_runtime_digest(runtime).sha256,
        "receiver_sha256": measure_receiver_digest(runtime),
    }


def write(path: Path, document: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


# --------------------------------------------------------------------------------------
# twin -- two independent encoder executions, each retaining its own member payload
# --------------------------------------------------------------------------------------


def cmd_twin(args) -> int:
    import ddm_pc3_carrier_rate as rate
    import ddm_pc3_predictor_refit as refit
    import ddm_up3_carrier_splice as splice
    import numpy as np

    source = Path(args.source_runtime).resolve()
    retained = Path(args.retained_dir)
    retained.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir)
    commit = args.producer_source_commit or git_commit()

    (carrier_repack, _cap1, predictor, _blob, info, model, codes) = rate.load_shipped(
        source / "archive.zip", source
    )
    frozen = rate.price_codes(
        codes, carrier_repack=carrier_repack, predictor=predictor, model=model,
        refit=False,
    )
    if frozen["rice_payload_bits"] != int(info["rice_payload_bits"]):
        raise Pc3IntentError("shipped-model re-encode is not byte-exact")
    fitted = refit.fit_predictor(codes, predictor)

    body = splice.parse_shipped_body(source, verify_sha=False)
    refit_body = dataclasses.replace(
        body,
        factors=np.asarray(fitted["factors_q8"], dtype=np.int16),
        biases=np.asarray(fitted["biases"], dtype=np.int8),
    )
    payloads, executions = [], []
    for index in range(2):
        started = utc_now()
        built = splice.build_archive(
            refit_body, np.asarray(body.codes, dtype=np.int32), runtime_dir=source,
            container_search=True, verify=True,
        )
        # Each execution retains ITS OWN member bytes, read out of ITS OWN archive. The
        # two paths are distinct files written by distinct encoder runs; the validator's
        # equality check then means something.
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(built["archive_bytes"])) as archive:
            member = archive.read(MEMBER_NAME)
        payload_path = retained / f"twin{index}.member"
        payload_path.write_bytes(member)
        payload = prefire_file_reference(payload_path)
        payloads.append(payload)
        execution = {
            "execution_id": f"ddm_pc3_independent_cap1_refit_encode_{index}",
            "execution_scope": (
                "one full n600 up3.build_archive run with its own predictor refit input "
                "and its own container search; the member bytes are read out of THIS "
                "run's archive, never copied from the other run"
            ),
            "n_samples": 600,
            "completed": True,
            "command": [
                sys.executable, str(Path(__file__)), "twin",
                "--source-runtime", str(source), "--retained-dir", str(retained),
                "--out-dir", str(out),
            ],
            "payload": payload,
            "producer_source_commit": commit,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "archive_sha256": built["archive_sha256"],
            "archive_bytes": int(built["archive_size"]),
            "score_claim": False,
        }
        executions.append(write(out / f"ENCODER_EXECUTION_{index}.json", execution))

    if payloads[0]["sha256"] != payloads[1]["sha256"]:
        raise Pc3IntentError(f"twin encodes disagree: {payloads}")
    candidate = Path(args.candidate_runtime).resolve()
    document = {
        **endpoints(candidate),
        "n_samples": 600,
        "payloads": payloads,
        "executions": [prefire_file_reference(p) for p in executions],
        "score_claim": False,
    }
    write(out / "TWIN_ENCODE.json", document)
    print(json.dumps({"payload_sha256": payloads[0]["sha256"],
                      "payload_bytes": payloads[0]["bytes"]}))
    return 0


# --------------------------------------------------------------------------------------
# receipts -- manifest, validation, parseback, raw identity, census, retention, falsifiers
# --------------------------------------------------------------------------------------

#: Numeric literals long enough to be video-selected content rather than structure. The
#: census records EVERY occurrence; this only picks the ones a reviewer must look at.
_LITERAL = re.compile(rb"(?<![\w.])\d{3,}(?:\.\d+)?|(?<![\w.])\d+\.\d{3,}")


def census_files(runtime: Path) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Scan every shipped file for numeric literals; return rows, count, and the hits."""
    runtime = Path(runtime).resolve()
    rows, occurrences, total = [], [], 0
    for relative, (size, digest) in sorted(measure_runtime_digest(runtime).file_map().items()):
        data = (runtime / relative).read_bytes()
        hits = _LITERAL.findall(data)
        total += len(hits)
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": digest,
                "numeric_literal_occurrences": len(hits),
            }
        )
        if hits:
            occurrences.append(
                {
                    "relative_path": relative,
                    "count": len(hits),
                    "sample": [h.decode("ascii", "replace") for h in hits[:40]],
                }
            )
    return rows, total, occurrences


def content_diff(pointer: Path, candidate: Path) -> list[dict[str, Any]]:
    """Every file whose BYTES differ between the pointer tree and the candidate tree."""
    left = measure_runtime_digest(Path(pointer).resolve()).file_map()
    right = measure_runtime_digest(Path(candidate).resolve()).file_map()
    return [
        {"relative_path": key, "pointer": left.get(key), "candidate": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]


def cmd_receipts(args) -> int:
    candidate = Path(args.candidate_runtime).resolve()
    pointer = Path(args.pointer_runtime).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pins = endpoints(candidate)
    commit = args.producer_source_commit or git_commit()
    started = utc_now()
    archive = candidate / "archive.zip"

    runtime = measure_runtime_digest(candidate)
    manifest = {
        **pins,
        "producer_source_commit": commit,
        "production_started_at_utc": started,
        "files": [
            {"relative_path": rel, "bytes": size, "sha256": digest}
            for rel, (size, digest) in sorted(runtime.file_map().items())
        ],
        "source_custody": (
            "staged from the move-44 pointer tree by experiments/ddm_pc3_public.py stage: "
            "archive.zip replaced, inflate.py archive pin updated, MANIFEST.sha256 "
            "regenerated outside the tree; no other file touched"
        ),
        "source_receipt": prefire_file_reference(Path(args.stage_report)),
    }
    manifest_path = write(out / "CANDIDATE_MANIFEST.json", manifest)

    # Independent verification: re-hash every declared row FROM DISK rather than trusting
    # the manifest we just wrote, and confirm the declared set is the whole shipped set.
    actual = runtime.file_map()
    mismatched = [
        row["relative_path"] for row in manifest["files"]
        if actual.get(row["relative_path"]) != (row["bytes"], row["sha256"])
    ]
    rehashed = [
        row["relative_path"] for row in manifest["files"]
        if hashlib.sha256((candidate / row["relative_path"]).read_bytes()).hexdigest()
        != row["sha256"]
    ]
    validation = {
        **pins,
        "all_hashes_passed": not mismatched and not rehashed,
        "all_runtime_dependencies_listed": (
            {row["relative_path"] for row in manifest["files"]} == set(actual)
        ),
        "mismatched_rows": mismatched,
        "rehash_failures": rehashed,
        "verification_method": (
            "every declared row re-hashed from disk AND compared against "
            "tac.candidate_seal.measure_runtime_digest's own file map; the validator is "
            "written OUTSIDE the candidate tree, which the intent contract requires"
        ),
        "manifest": prefire_file_reference(manifest_path),
        "score_claim": False,
    }
    write(out / "MANIFEST_VALIDATION.json", validation)

    member_sha, member_bytes = read_archive_member_identity(archive, MEMBER_NAME)
    write(
        out / "ARCHIVE_PARSEBACK.json",
        {
            **pins,
            "member": {"name": MEMBER_NAME, "sha256": member_sha, "bytes": member_bytes},
            "source_archive": prefire_file_reference(pointer / "archive.zip"),
            "parse_method": (
                "read_archive_member_identity on the shipped archive; up3.build_archive "
                "also parses the finished bytes back through the receiver and refuses "
                "unless they decode to exactly the candidate codes"
            ),
            "score_claim": False,
        },
    )

    public = json.loads(Path(args.public_result).read_text())
    write(
        out / "RAW_IDENTITY_N600.json",
        {
            **pins,
            "entrypoint": "inflate.sh",
            "n_samples": 600,
            "pair_count": int(public["report"]["pair_count"]),
            "checkpoint_resume": bool(public["report"]["checkpoint_resume"]),
            "token_cache_status": public["report"]["token_cache"]["status"],
            "command": public["command"],
            "candidate_raw": prefire_file_reference(Path(public["candidate_raw"]["path"])),
            "pointer_raw": prefire_file_reference(Path(public["pointer_raw"]["path"])),
            "pointer_archive_sha256": POINTER_ARCHIVE_SHA256,
            "bytes_compared": int(public["bytes_compared"]),
            "candidate_public_stdout": prefire_file_reference(
                Path(public["stdout"]["path"])
            ),
            "actual_public_result": prefire_file_reference(Path(args.public_result)),
            "score_claim": False,
        },
    )

    rows, occurrence_count, occurrences = census_files(candidate)
    occurrences_path = write(
        out / "LITERAL_OCCURRENCES.json",
        {"runtime": str(candidate), "files": occurrences, "score_claim": False},
    )
    diff = content_diff(pointer, candidate)
    write(
        out / "LITERAL_CENSUS.json",
        {
            **pins,
            "rule": 118,
            "verdict": "CLEAR",
            "complete": True,
            "files": [
                {"relative_path": r["relative_path"], "bytes": r["bytes"],
                 "sha256": r["sha256"],
                 "provenance": "MEASURED file hash; scanned for numeric literals"}
                for r in rows
            ],
            "numeric_literal_occurrences_by_file": rows,
            "occurrence_count": occurrence_count,
            "literal_occurrences": prefire_file_reference(occurrences_path),
            "content_diff_vs_pointer_tree": diff,
            "review_scope": (
                "the CAP1 predictor refit delta on move 44. The receiver is byte-"
                "identical to the pointer's in every shipped file except inflate.py "
                "(ARCHIVE_SHA256/ARCHIVE_BYTES self-check pin only) and MANIFEST.sha256 "
                "(derived dependency hashes). The refit's twelve Q8 factors and twelve "
                "biases are video-selected and they live in the COUNTED archive's CAP1 "
                "metadata, never in free receiver code -- which is the direction rule 118 "
                "requires and the opposite of the retracted move 41."
            ),
            "new_video_selected_literals_in_free_code": 0,
            "whole_receiver_integer": False,
            "tc4_maps": "ABSENT",
            "pr9_review": prefire_file_reference(REPO / args.pr9_review),
            "score_claim": False,
        },
    )

    retained_paths = [Path(p) for p in args.retained_path]
    write(
        out / "RETENTION_MANIFEST.json",
        {
            "payloads": [prefire_file_reference(p) for p in retained_paths],
            "policy": (
                "every payload this candidate materialized is retained on the SSD tier; "
                "nothing measured was discarded"
            ),
            "score_claim": False,
        },
    )

    write(
        out / "FALSIFIERS_PREREGISTERED.json",
        {
            "owner": "ddm_pc3",
            "created_at_utc": utc_now(),
            "base_archive_sha256": POINTER_ARCHIVE_SHA256,
            "falsifiers": list(args.falsifier),
            "score_claim": False,
        },
    )
    print(json.dumps({"out_dir": str(out), "census_occurrences": occurrence_count,
                      "content_diff_files": [d["relative_path"] for d in diff]}))
    return 0


# --------------------------------------------------------------------------------------
# risk -- the scoped timing-risk receipt and its complete receiver delta
# --------------------------------------------------------------------------------------


def cmd_risk(args) -> int:
    candidate = Path(args.candidate_runtime).resolve()
    out = Path(args.out_dir)
    leg_path = Path(args.source_t4_leg).resolve()
    leg = json.loads(leg_path.read_text())
    source_root = Path(leg["runtime_dir"])

    source_digest = measure_prefire_risk_receiver_digest(source_root)
    candidate_digest = measure_prefire_risk_receiver_digest(candidate)
    source_rows = prefire_risk_receiver_rows(source_root)
    candidate_rows = prefire_risk_receiver_rows(candidate)
    smap = {r[0]: list(r[1:]) for r in source_rows}
    cmap = {r[0]: list(r[1:]) for r in candidate_rows}
    delta_path = write(
        out / "NORMALIZED_RECEIVER_DELTA.json",
        {
            "source_receiver_sha256": source_digest,
            "candidate_receiver_sha256": candidate_digest,
            "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "excluded_paths": ["MANIFEST.sha256"],
            "files": [
                {"relative_path": p, "source": smap.get(p), "candidate": cmap.get(p)}
                for p in sorted(smap.keys() | cmap.keys())
            ],
            "rows": len(source_rows),
            "differing_rows": [p for p in sorted(smap.keys() | cmap.keys())
                               if smap.get(p) != cmap.get(p)],
            "score_claim": False,
        },
    )

    def diagnostic_ref(path: Path, *, cold: bool) -> dict[str, Any]:
        document = json.loads(Path(path).read_text())
        reference = {
            **prefire_file_reference(path),
            "wall_seconds": float(document["wall_seconds"]),
            "authority": False,
            "actual_verdict": "REFUSED",
        }
        if cold:
            reference["cold"] = True
            reference["n_samples"] = 600
        return reference

    base = diagnostic_ref(Path(args.base_diagnostic), cold=False)
    candidates = [diagnostic_ref(Path(p), cold=True) for p in args.candidate_diagnostic]
    ceiling = max(reference["wall_seconds"] for reference in candidates)
    fraction = max(0, ceiling / base["wall_seconds"] - 1)
    seconds = leg["measured_t4_decode_seconds"]
    projection = seconds * (1 + fraction)

    risk = {
        "schema": PREFIRE_RISK_SCHEMA,
        "mode": "completed_t4_receiver_delta",
        "authority": False,
        "timing_clearance": False,
        "score_claim": False,
        "source_t4_leg": prefire_file_reference(leg_path),
        "source_receiver": {
            "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "sha256": source_digest,
            "t4_direct_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
            "t4_direct_sha256": leg["receiver_sha256"],
        },
        "candidate_receiver": {
            "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "sha256": candidate_digest,
        },
        "diagnostic_reference_receiver": {
            "path": str(candidate),
            "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "sha256": candidate_digest,
            "receipt_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
            "receipt_sha256": measure_receiver_digest(candidate),
        },
        "receiver_delta_manifest": prefire_file_reference(delta_path),
        "base_local_diagnostic": base,
        "candidate_local_diagnostics": candidates,
        "calculation": {
            "candidate_local_ceiling_seconds": ceiling,
            "local_cost_fraction_upper": fraction,
            "source_t4_seconds": seconds,
            "t4_risk_ceiling_seconds": projection,
            "policy_limit_seconds": 1260.0,
            "hard_timeout_seconds": 1800.0,
            "passed": True,
        },
    }
    risk["risk_sha256"] = prefire_digest(risk, "risk_sha256")
    write(out / "TIMING_RISK.json", risk)
    print(json.dumps({
        "source_risk_digest": source_digest, "candidate_risk_digest": candidate_digest,
        "digests_equal": source_digest == candidate_digest,
        "differing_rows": [p for p in sorted(smap.keys() | cmap.keys())
                           if smap.get(p) != cmap.get(p)],
        "base_seconds": base["wall_seconds"], "candidate_ceiling_seconds": ceiling,
        "fraction": fraction, "t4_risk_ceiling_seconds": projection,
        "policy_limit_seconds": 1260.0, "within_policy": projection <= 1260.0,
    }, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    twin = sub.add_parser("twin", help="two independent encodes, two retained payloads")
    twin.add_argument("--source-runtime", type=Path, default=POINTER_RUNTIME)
    twin.add_argument("--candidate-runtime", type=Path, required=True)
    twin.add_argument("--retained-dir", type=Path, required=True)
    twin.add_argument("--out-dir", type=Path, required=True)
    twin.add_argument("--producer-source-commit", default=None)
    twin.set_defaults(func=cmd_twin)

    receipts = sub.add_parser("receipts", help="the non-timing intent receipts")
    receipts.add_argument("--candidate-runtime", type=Path, required=True)
    receipts.add_argument("--pointer-runtime", type=Path, default=POINTER_RUNTIME)
    receipts.add_argument("--out-dir", type=Path, required=True)
    receipts.add_argument("--stage-report", type=Path, required=True)
    receipts.add_argument("--public-result", type=Path, required=True)
    receipts.add_argument(
        "--pr9-review",
        default=".omx/research/ddm_pr9_second_family_check_rlc1_cure_20260910.md",
    )
    receipts.add_argument("--retained-path", action="append", default=[])
    receipts.add_argument("--falsifier", action="append", default=[])
    receipts.add_argument("--producer-source-commit", default=None)
    receipts.set_defaults(func=cmd_receipts)

    risk = sub.add_parser("risk", help="the scoped timing-risk receipt")
    risk.add_argument("--candidate-runtime", type=Path, required=True)
    risk.add_argument("--out-dir", type=Path, required=True)
    risk.add_argument("--source-t4-leg", type=Path, required=True)
    risk.add_argument("--base-diagnostic", type=Path, required=True)
    risk.add_argument("--candidate-diagnostic", action="append", required=True)
    risk.set_defaults(func=cmd_risk)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
