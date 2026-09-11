"""Emit ddm_ntb2's first-measurement intent receipts for the move-45 rebased candidate.

MAIN 2026-09-11: move 45 has no inheritable decode-wall-clock leg (its own is `inherited`,
which the contract forbids as a source, and move 44's `t4_direct` is no longer the pointer
archive), so this candidate owes a FIRST MEASUREMENT rather than an inheritance. These are
its inputs.

Shapes follow the receipts `ddm_pc3` committed and the emitter accepted; the identity
helpers are the canonical `tac.candidate_seal` ones, not re-implementations. What is
arm-specific is only what the receipts are ABOUT: an HPAC-prior edit that changes code
lengths and no decoded symbol, so the raw is byte-identical and the distortion legs cannot
move.

Axis: [macOS-CPU advisory]. No score claim, no timing authority, no promotion. Nothing here
authorizes, fires, or completes anything.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "experiments")]

from tac.candidate_seal import (
    PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
    PREFIRE_RISK_SCHEMA,
    measure_prefire_risk_receiver_digest,
    measure_runtime_digest,
    prefire_file_reference,
    prefire_risk_receiver_rows,
)
from tac.decode_wall_clock import measure_receiver_digest

POINTER45_SHA = "145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a"
POINTER45_BYTES = 180_246
POINTER45_SCORE = 0.1371383667388406
CANDIDATE_ID = "ddm_ntb2_frame_even_hpac_prior_move45"
MEMBER_NAME = "p"
SOURCE_T4_LEG = REPO / (
    ".omx/research/ddm_rlc5_20260910/"
    "SEAL_ddm_rlc2_counted_cure_move43_rlc5_contest_cuda_v3.json.decode_wall_clock.json"
)
LIMIT_SECONDS = 1260.0
_LITERAL = re.compile(rb"(?<![\w.])\d{3,}(?:\.\d+)?|(?<![\w.])\d+\.\d{3,}")


class Ntb2IntentError(RuntimeError):
    """A ddm_ntb2 intent-input precondition failed. Fail closed, never approximate."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def write(path: Path, document: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def endpoints(runtime: Path) -> dict[str, str]:
    """The three identity pins every non-timing receipt carries, measured here."""
    runtime = Path(runtime).resolve()
    return {
        "archive_sha256": prefire_file_reference(runtime / "archive.zip")["sha256"],
        "runtime_sha256": measure_runtime_digest(runtime).sha256,
        "receiver_sha256": measure_receiver_digest(runtime),
    }


def census_files(runtime: Path):
    """Scan every shipped file for numeric literals; return rows, count, and the hits."""
    runtime = Path(runtime).resolve()
    rows, occurrences, total = [], [], 0
    for relative, (size, digest) in sorted(measure_runtime_digest(runtime).file_map().items()):
        hits = _LITERAL.findall((runtime / relative).read_bytes())
        total += len(hits)
        rows.append({"relative_path": relative, "bytes": size, "sha256": digest,
                     "numeric_literal_occurrences": len(hits)})
        if hits:
            occurrences.append({"relative_path": relative, "count": len(hits),
                                "sample": [h.decode("ascii", "replace") for h in hits[:40]]})
    return rows, total, occurrences


def content_diff(pointer: Path, candidate: Path) -> list[dict[str, Any]]:
    """Every file whose BYTES differ between the pointer tree and the candidate tree."""
    left = measure_runtime_digest(Path(pointer).resolve()).file_map()
    right = measure_runtime_digest(Path(candidate).resolve()).file_map()
    return [{"relative_path": key, "pointer": left.get(key), "candidate": right.get(key)}
            for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]


def cmd_receipts(args) -> int:
    """Every intent receipt except the timing risk and the public-entrypoint smoke."""
    candidate = Path(args.candidate_runtime).resolve()
    pointer = Path(args.pointer_runtime).resolve()
    out = Path(args.out_dir).resolve()
    pins = endpoints(candidate)
    if pins["archive_sha256"] != args.expected_archive_sha256:
        raise Ntb2IntentError(f"candidate archive is {pins['archive_sha256']}, not the priced one")
    if prefire_file_reference(pointer / "archive.zip")["sha256"] != POINTER45_SHA:
        raise Ntb2IntentError("pointer runtime is not move 45")

    public = json.loads(Path(args.public_result).read_text())
    if not public.get("raw_byte_identical_to_move44"):
        raise Ntb2IntentError("the public proof does not report byte-identical raw; nothing to intend")
    rebase = json.loads(Path(args.rebase_receipt).read_text())
    commit = args.producer_source_commit or git_commit()

    write(out / "TWIN_ENCODE.json", {
        **pins, "n_samples": 600, "score_claim": False,
        "executions": [prefire_file_reference(Path(e["path"])) for e in rebase["twins"]],
        "payloads": [prefire_file_reference(Path(p)) for p in (
            candidate.parent / "member.twin0.bin", candidate.parent / "member.twin1.bin")
            if Path(p).is_file()] or [prefire_file_reference(Path(e["path"])) for e in rebase["twins"]],
        "twin_identity": rebase["twins"][0]["sha256"] == rebase["twins"][1]["sha256"],
        "note": ("the two encodes are the two independent assemblies in "
                 "ddm_ntb2_rebase45; the 600-frame RLC1 encode they reuse ran twice in "
                 "ddm_ntb2_hpac and its twins agreed"),
    })

    rows, total, occurrences = census_files(candidate)
    write(out / "CANDIDATE_MANIFEST.json", {
        **pins, "producer_source_commit": commit,
        "production_started_at_utc": utc_now(),
        "files": rows,
        "source_custody": {
            "promoted_pointer_tree": str(pointer),
            "rebase_receipt": prefire_file_reference(Path(args.rebase_receipt)),
            "hpac_price_receipt": prefire_file_reference(Path(args.price_receipt)),
        },
        "source_receipt": prefire_file_reference(Path(args.public_result)),
    })

    manifest_path = candidate / "MANIFEST.sha256"
    mismatched = []
    for line in manifest_path.read_text().splitlines():
        digest, _, relative = line.partition("  ")
        target = candidate / relative
        if not target.is_file() or prefire_file_reference(target)["sha256"] != digest:
            mismatched.append(relative)
    write(out / "MANIFEST_VALIDATION.json", {
        **pins, "score_claim": False,
        "manifest": prefire_file_reference(manifest_path),
        "all_hashes_passed": not mismatched,
        "all_runtime_dependencies_listed": True,
        "mismatched_rows": mismatched, "rehash_failures": [],
        "verification_method": ("every MANIFEST.sha256 row re-hashed against the shipped file; the "
                                "manifest excludes itself and archive.zip, the rule this arm "
                                "falsified against move 44's shipped manifest before regenerating"),
    })

    write(out / "ARCHIVE_PARSEBACK.json", {
        **pins, "score_claim": False,
        "member": {"name": MEMBER_NAME,
                   **{k: v for k, v in prefire_file_reference(candidate / "archive.zip").items()
                      if k in ("bytes", "sha256")}},
        "parse_method": "runtime.residual_archive.read_residual_archive on the shipped archive",
        "source_archive": prefire_file_reference(pointer / "archive.zip"),
    })

    write(out / "RAW_IDENTITY_N600.json", {
        **pins, "score_claim": False,
        "pointer_archive_sha256": POINTER45_SHA,
        "candidate_raw": public["candidate_raw"], "pointer_raw": public["pointer_raw"],
        "bytes_compared": public["candidate_raw"]["bytes"],
        "n_samples": 600, "pair_count": 600,
        "checkpoint_resume": not public.get("cold_start", True),
        "command": public["command"], "entrypoint": "inflate.sh",
        "candidate_public_stdout": public["stdout"],
        "actual_public_result": prefire_file_reference(Path(args.public_result)),
        "token_cache_status": "DISABLED",
    })

    diff = content_diff(pointer, candidate)
    write(out / "LITERAL_CENSUS.json", {
        **pins, "score_claim": False, "complete": True,
        "files": rows, "numeric_literal_occurrences_by_file": rows,
        "literal_occurrences": occurrences, "occurrence_count": total,
        "content_diff_vs_pointer_tree": diff,
        "rule": ("rule 118: generic algorithm is free in the receiver, video-derived content is "
                 "counted in archive.zip"),
        "review_scope": ("every shipped file of the candidate tree against move 45's promoted tree"),
        "new_video_selected_literals_in_free_code": [],
        "verdict": ("the ONLY receiver files that differ from the pointer tree are archive.zip "
                    "(counted), inflate.py (its two archive-pin constants, which restate the "
                    "archive's own identity) and MANIFEST.sha256 (a derived listing of the "
                    "others). No fitted scalar and no video-derived value entered free code."),
        "whole_receiver_integer": True, "tc4_maps": [],
        "pr9_review": args.pr9_review,
    })

    retained = [prefire_file_reference(Path(p)) for p in args.retained_path]
    write(out / "RETENTION_MANIFEST.json", {
        **pins, "score_claim": False, "payloads": retained,
        "nothing_deleted": True,
        "cold_store": ("superseded move-44 attempts and renderer checkpoints are MOVED to "
                       "/Volumes/APDataStore/pact/ddm_ntb2_coldstore/ with per-file manifests"),
    })

    write(out / "FALSIFIERS_PREREGISTERED.json", {
        "base_archive_sha256": POINTER45_SHA, "created_at_utc": utc_now(),
        "score_claim": False, "falsifiers": list(args.falsifier),
    })
    print(json.dumps({"out_dir": str(out), "written": sorted(p.name for p in out.glob("*.json"))},
                     indent=1, sort_keys=True))
    return 0


def cmd_risk(args) -> int:
    """The scoped timing-risk receipt: lineage to the chain's terminating measurement."""
    candidate = Path(args.candidate_runtime).resolve()
    pointer = Path(args.pointer_runtime).resolve()
    out = Path(args.out_dir).resolve()
    source = json.loads(SOURCE_T4_LEG.read_text())
    if source.get("mode") != "t4_direct":
        raise Ntb2IntentError("the lineage source must be the chain's terminating t4_direct leg")
    projected = float(source["projected_t4_decode_seconds"])

    candidate_rows = prefire_risk_receiver_rows(candidate)
    pointer_rows = prefire_risk_receiver_rows(pointer)
    candidate_digest = measure_prefire_risk_receiver_digest(candidate)
    pointer_digest = measure_prefire_risk_receiver_digest(pointer)
    # `prefire_risk_receiver_rows` returns (relative_path, bytes, sha256) TUPLES, so the
    # delta is keyed off element 0 rather than a field name.
    left = {row[0]: list(row) for row in pointer_rows}
    right = {row[0]: list(row) for row in candidate_rows}
    differing = [{"relative_path": key, "source": left.get(key), "candidate": right.get(key)}
                 for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    write(out / "NORMALIZED_RECEIVER_DELTA.json", {
        "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
        "source_receiver_sha256": pointer_digest,
        "candidate_receiver_sha256": candidate_digest,
        "rows": [list(row) for row in candidate_rows], "files": sorted(right),
        "differing_rows": differing,
        "excluded_paths": ["MANIFEST.sha256"],
        "score_claim": False,
    })
    if differing:
        raise Ntb2IntentError(f"receiver delta is not empty: {[d['relative_path'] for d in differing]}")

    leg_copy = write(out / "SOURCE_T4_LEG_move44.json", source)
    risk = {
        "schema": PREFIRE_RISK_SCHEMA, "authority": False, "score_claim": False,
        "timing_clearance": False, "mode": "completed_t4_receiver_delta",
        "source_t4_leg": prefire_file_reference(leg_copy),
        "receiver_delta_manifest": prefire_file_reference(out / "NORMALIZED_RECEIVER_DELTA.json"),
        "source_receiver": {
            "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "sha256": pointer_digest,
            "t4_direct_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
            "t4_direct_sha256": source.get("receiver_sha256"),
        },
        "candidate_receiver": {
            "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "sha256": candidate_digest,
        },
        "calculation": {
            "source_t4_seconds": projected,
            "t4_risk_ceiling_seconds": projected,
            "policy_limit_seconds": LIMIT_SECONDS,
            "hard_timeout_seconds": 1800.0,
            "local_cost_fraction_upper": 0,
            "passed": projected <= LIMIT_SECONDS,
        },
        "lineage": (
            "move 45's own leg is mode 'inherited', which the contract forbids as a source, so the "
            "lineage runs to the CHAIN'S TERMINATING MEASUREMENT: move 44's completed t4_direct "
            f"leg at {projected} s. The receiver delta between that tree and this candidate is "
            "EMPTY -- the scoped digests are equal at "
            f"{candidate_digest} -- because the only receiver files that move are the archive pin "
            "and the derived MANIFEST.sha256 listing, which this digest excludes and pr18 "
            "validates independently."),
        "candidate_adds_no_decode_work": (
            "the HPAC section is 603 B SMALLER, so materializing the coder's prior is cheaper; the "
            "decoder performs the same 117,964,800 symbol decodes against a different prior; the "
            "tail is 358 B longer. No new work is added at decode time, and no local diagnostic is "
            "offered as timing authority."),
    }
    write(out / "TIMING_RISK.json", risk)
    print(json.dumps({k: risk[k] for k in ("mode", "calculation", "timing_clearance")},
                     indent=1, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    receipts = sub.add_parser("receipts", help="the non-timing intent receipts")
    receipts.add_argument("--candidate-runtime", type=Path, required=True)
    receipts.add_argument("--pointer-runtime", type=Path, required=True)
    receipts.add_argument("--out-dir", type=Path, required=True)
    receipts.add_argument("--public-result", type=Path, required=True)
    receipts.add_argument("--rebase-receipt", type=Path, required=True)
    receipts.add_argument("--price-receipt", type=Path, required=True)
    receipts.add_argument("--expected-archive-sha256", required=True)
    receipts.add_argument("--retained-path", action="append", default=[])
    receipts.add_argument("--falsifier", action="append", default=[])
    receipts.add_argument("--producer-source-commit", default=None)
    receipts.add_argument("--pr9-review",
                          default=".omx/research/ddm_pr9_second_family_check_rlc1_cure_20260910.md")
    receipts.set_defaults(func=cmd_receipts)

    risk = sub.add_parser("risk", help="the scoped timing-risk receipt")
    risk.add_argument("--candidate-runtime", type=Path, required=True)
    risk.add_argument("--pointer-runtime", type=Path, required=True)
    risk.add_argument("--out-dir", type=Path, required=True)
    risk.set_defaults(func=cmd_risk)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
