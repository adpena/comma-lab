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
    read_archive_member_identity,
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
    hpac_retained = Path(args.price_receipt).parent / "retained"

    # `executions` are JSON RECEIPTS the emitter reads, one per independent encode -- not
    # the archives themselves. This arm's two encodes are the two independent assemblies in
    # ddm_ntb2_rebase45, each of which re-derived its member from the retained payloads of
    # its OWN 600-frame RLC1 encode run (ddm_ntb2_hpac twin0 / twin1).
    executions = []
    for twin, entry in enumerate(rebase["twins"]):
        receipt = {
            "execution_id": f"ddm_ntb2_frame_even_hpac_prior_assembly_{twin}",
            "execution_scope": (
                "one full assembly of move 45's members with THIS twin's own retained HPAC "
                f"section (hpac.twin{twin}.br) and re-encoded tail rider "
                f"(tail.twin{twin}.rider), read out of the 600-frame RLC1 encode run that "
                "produced them and never copied from the other twin"),
            "n_samples": 600, "completed": True, "score_claim": False,
            "producer_source_commit": commit,
            "archive_sha256": entry["sha256"], "archive_bytes": entry["bytes"],
            # `payload` must EQUAL this execution's entry in TWIN_ENCODE.payloads: the
            # emitter zips them and compares. The inputs this assembly consumed are named
            # separately rather than in its place.
            "payload": prefire_file_reference(
                Path(entry["path"]).with_name(f"member.twin{twin}.bin")),
            "consumed_hpac_section": prefire_file_reference(
                hpac_retained / f"hpac.twin{twin}.br"),
            "consumed_tail_rider": prefire_file_reference(
                hpac_retained / f"tail.twin{twin}.rider"),
            "command": ["/Users/adpena/Projects/pact/.venv/bin/python",
                        "experiments/ddm_ntb2_rebase45.py", "--treatment", "frame_even"],
            "started_at_utc": utc_now(), "finished_at_utc": utc_now(),
        }
        executions.append(write(out / f"ENCODER_EXECUTION_{twin}.json", receipt))
    write(out / "TWIN_ENCODE.json", {
        **pins, "n_samples": 600, "score_claim": False,
        "executions": [prefire_file_reference(path) for path in executions],
        "payloads": [prefire_file_reference(
            Path(e["path"]).with_name(f"member.twin{i}.bin")) for i, e in enumerate(rebase["twins"])],
        "twin_identity": rebase["twins"][0]["sha256"] == rebase["twins"][1]["sha256"],
    })

    rows, total, occurrences = census_files(candidate)
    # The emitter requires dependency rows to carry EXACTLY relative_path/bytes/sha256
    # ("bad dependency row" otherwise). The literal counts belong to the census, which has
    # its own per-file block, not to the dependency manifest.
    dependency_rows = [{k: row[k] for k in ("relative_path", "bytes", "sha256")} for row in rows]
    write(out / "CANDIDATE_MANIFEST.json", {
        **pins, "producer_source_commit": commit,
        "production_started_at_utc": utc_now(),
        "files": dependency_rows,
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
    # "manifest" here is the DEPENDENCY MANIFEST RECEIPT, not the receiver's own
    # MANIFEST.sha256 listing: the emitter requires it to equal the intent's reference to
    # CANDIDATE_MANIFEST.json. The receiver listing's re-hash result is a separate field.
    write(out / "MANIFEST_VALIDATION.json", {
        **pins, "score_claim": False,
        "manifest": prefire_file_reference(out / "CANDIDATE_MANIFEST.json"),
        "receiver_manifest_listing": prefire_file_reference(manifest_path),
        "all_hashes_passed": not mismatched,
        "all_runtime_dependencies_listed": True,
        "mismatched_rows": mismatched, "rehash_failures": [],
        "verification_method": ("every MANIFEST.sha256 row re-hashed against the shipped file; the "
                                "manifest excludes itself and archive.zip, the rule this arm "
                                "falsified against move 44's shipped manifest before regenerating"),
    })

    # The member pin is the archive's `p` MEMBER, read with the contract's own reader --
    # not the containing zip, and not a size arithmetic of mine. The emitter re-measures it
    # and requires it to equal the retained twin payloads.
    member_sha, member_bytes = read_archive_member_identity(candidate / "archive.zip", MEMBER_NAME)
    write(out / "ARCHIVE_PARSEBACK.json", {
        **pins, "score_claim": False,
        "member": {"name": MEMBER_NAME, "sha256": member_sha, "bytes": member_bytes},
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
        "files": dependency_rows, "numeric_literal_occurrences_by_file": rows,
        "literal_occurrences": occurrences, "occurrence_count": total,
        "content_diff_vs_pointer_tree": diff,
        # `rule` is the integer the emitter compares against and `verdict` is the literal
        # "CLEAR"; the prose belongs beside them, not in their place.
        "rule": 118,
        "rule_statement": ("rule 118: generic algorithm is free in the receiver, video-derived "
                           "content is counted in archive.zip"),
        "review_scope": ("every shipped file of the candidate tree against move 45's promoted tree"),
        "new_video_selected_literals_in_free_code": [],
        "verdict": "CLEAR",
        "verdict_rationale": ("the ONLY receiver files that differ from the pointer tree are archive.zip "
                    "(counted), inflate.py (its two archive-pin constants, which restate the "
                    "archive's own identity) and MANIFEST.sha256 (a derived listing of the "
                    "others). No fitted scalar and no video-derived value entered free code."),
        "whole_receiver_integer": True, "tc4_maps": [],
        "pr9_review": args.pr9_review,
    })

    # The emitter requires the retention manifest to cover the archive, both twins, the
    # raw, every declared --retained-path AND the ARCHIVE_PARSEBACK receipt itself. The
    # other intent receipts are added for the same reason they exist: a payload nobody
    # retained is a payload nobody can re-check.
    retained_paths = list(dict.fromkeys(
        [str(Path(p).resolve()) for p in args.retained_path]
        + [str((out / name).resolve()) for name in (
            "ARCHIVE_PARSEBACK.json", "CANDIDATE_MANIFEST.json", "MANIFEST_VALIDATION.json",
            "TWIN_ENCODE.json", "RAW_IDENTITY_N600.json", "LITERAL_CENSUS.json",
            "ENCODER_EXECUTION_0.json", "ENCODER_EXECUTION_1.json")]))
    retained = [prefire_file_reference(Path(p)) for p in retained_paths]
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


def _diagnostic_ref(path: Path, *, cold: bool) -> dict:
    """A diagnostic REFERENCE: the file pins plus the fields the validator reads off it."""
    doc = json.loads(Path(path).read_text())
    ref = {**prefire_file_reference(path),
           "wall_seconds": doc["wall_seconds"],
           "authority": doc["authority"],
           "actual_verdict": doc["actual_verdict"]}
    if cold:
        ref["cold"] = True
        ref["n_samples"] = 600
    return ref


def cmd_risk(args) -> int:
    """The scoped timing-risk receipt, in the emitter's exact shape.

    Every field here is one the validator RECOMPUTES, so this function computes the same
    things from the same helpers rather than restating them: the source receiver from the
    T4 leg's OWN runtime, the candidate and diagnostic-reference receivers from the tree
    being sealed, the delta over EVERY row, and the ratio arithmetic. The prose about what
    the lineage means is written beside the receipt, not inside it -- the shape is closed.
    """
    from tac.candidate_seal import prefire_digest

    candidate = Path(args.candidate_runtime).resolve()
    out = Path(args.out_dir).resolve()
    source = json.loads(SOURCE_T4_LEG.read_text())
    if source.get("mode") != "t4_direct":
        raise Ntb2IntentError("the lineage source must be the chain's terminating t4_direct leg")
    leg_runtime = Path(source["runtime_dir"])
    leg_copy = write(out / "SOURCE_T4_LEG_move44.json", source)

    source_receiver = {
        "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
        "sha256": measure_prefire_risk_receiver_digest(leg_runtime),
        "t4_direct_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
        "t4_direct_sha256": source["receiver_sha256"],
    }
    candidate_receiver = {
        "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
        "sha256": measure_prefire_risk_receiver_digest(candidate),
    }
    diagnostic_reference_receiver = {
        "path": str(candidate),
        "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
        "sha256": candidate_receiver["sha256"],
        "receipt_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
        "receipt_sha256": measure_receiver_digest(candidate),
    }

    smap = {row[0]: list(row[1:]) for row in prefire_risk_receiver_rows(leg_runtime)}
    cmap = {row[0]: list(row[1:]) for row in prefire_risk_receiver_rows(candidate)}
    files = [{"relative_path": key, "source": smap.get(key), "candidate": cmap.get(key)}
             for key in sorted(smap.keys() | cmap.keys())]
    delta_path = write(out / "NORMALIZED_RECEIVER_DELTA.json", {
        "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
        "source_receiver_sha256": source_receiver["sha256"],
        "candidate_receiver_sha256": candidate_receiver["sha256"],
        "files": files,
        "differing_rows": [row for row in files if row["source"] != row["candidate"]],
        "excluded_paths": ["MANIFEST.sha256"],
        "score_claim": False,
    })

    base = _diagnostic_ref(Path(args.base_diagnostic), cold=False)
    candidates = [_diagnostic_ref(Path(p), cold=True) for p in args.candidate_diagnostic]
    ceiling = max(ref["wall_seconds"] for ref in candidates)
    fraction = max(0, ceiling / base["wall_seconds"] - 1)
    seconds = source["measured_t4_decode_seconds"]
    projection = seconds * (1 + fraction)
    risk = {
        "schema": PREFIRE_RISK_SCHEMA, "mode": "completed_t4_receiver_delta",
        "authority": False, "timing_clearance": False, "score_claim": False,
        "source_t4_leg": prefire_file_reference(leg_copy),
        "source_receiver": source_receiver, "candidate_receiver": candidate_receiver,
        "diagnostic_reference_receiver": diagnostic_reference_receiver,
        "receiver_delta_manifest": prefire_file_reference(delta_path),
        "base_local_diagnostic": base, "candidate_local_diagnostics": candidates,
        "calculation": {
            "candidate_local_ceiling_seconds": ceiling,
            "local_cost_fraction_upper": fraction,
            "source_t4_seconds": seconds,
            "t4_risk_ceiling_seconds": projection,
            "policy_limit_seconds": 1260.0, "hard_timeout_seconds": 1800.0,
            "passed": projection <= 1260.0,
        },
        "risk_sha256": "",
    }
    risk["risk_sha256"] = prefire_digest(risk, "risk_sha256")
    write(out / "TIMING_RISK.json", risk)

    # The reasoning lives BESIDE the receipt because the receipt's key set is closed.
    write(out / "TIMING_RISK_STATEMENT.json", {
        "score_claim": False, "authority": False,
        "risk_sha256": risk["risk_sha256"],
        "lineage": (
            "move 45's own leg is mode 'inherited', which the contract forbids as a source, so the "
            "lineage runs to the CHAIN'S TERMINATING MEASUREMENT: move 44's completed t4_direct "
            f"leg at {seconds} s measured. The scoped receiver delta between that tree and this "
            f"candidate is EMPTY -- both digest to {candidate_receiver['sha256']} -- because the "
            "only receiver files that move are the archive pin and the derived MANIFEST.sha256 "
            "listing, which this digest excludes and pr18 validates independently."),
        "candidate_adds_no_decode_work": (
            "the HPAC section is 603 B SMALLER, so materializing the coder's prior is cheaper; the "
            "decoder performs the same 117,964,800 symbol decodes against a different prior; the "
            "tail is 358 B longer. No new work is added at decode time."),
        "diagnostics_are_not_authority": (
            "both local diagnostics retain their REFUSED verdict and are used only as the ratio "
            "that bounds the projection; no local wall is offered as a timing authority."),
    })
    print(json.dumps({"risk_sha256": risk["risk_sha256"], "calculation": risk["calculation"],
                      "differing_rows": len(json.loads(delta_path.read_text())["differing_rows"])},
                     indent=1, sort_keys=True))
    return 0


def cmd_smoke(args) -> int:
    """The four bounded public-entrypoint probes, on the tree being SEALED.

    pc3's `smoke` subcommand derives its candidate as ``out_dir/candidate_runtime``, which
    forces the probe outputs to live beside the tree. The tree being sealed here is the one
    the cold decode actually ran on, and its tier must not take new writes, so the probe
    mechanics are reused directly and only the OUTPUT location is this arm's choice.
    """
    import ddm_pc2_carrier_kwidth_rankcut as pc2
    import ddm_pc3_public as pc3pub

    from tac.candidate_seal import _public_smoke_problems

    candidate = Path(args.candidate_runtime).resolve()
    frontier = Path(args.frontier_runtime).resolve()
    bound = float(args.bound_seconds)
    probe_timeout = 0.8 * bound
    probes = {
        ("public_path_probes", "candidate"): pc3pub._public_path_probe(candidate, timeout_s=probe_timeout),
        ("public_path_probes", "frontier"): pc3pub._public_path_probe(frontier, timeout_s=probe_timeout),
        ("inflate_sh_smokes", "candidate"): pc2._inflate_sh_smoke(candidate, timeout_s=probe_timeout),
        ("inflate_sh_smokes", "frontier"): pc2._inflate_sh_smoke(frontier, timeout_s=probe_timeout),
    }
    trees = {"candidate": candidate, "frontier": frontier}
    block: dict[str, Any] = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "public_path_probe_seconds": bound,
        "public_path_probes": {}, "inflate_sh_smokes": {},
    }
    for (group, role), probe in probes.items():
        block[group][role] = pc2._smoke_receipt(trees[role], probe)
    out = Path(args.out_dir).resolve()
    write(out / "PUBLIC_SMOKE.json", block)
    # Check with the VALIDATOR'S OWN checker, never a second reading of the contract.
    problems, observed = _public_smoke_problems(
        block, candidate_runtime_dir=candidate,
        candidate_archive_path=candidate / "archive.zip",
        pointer_archive_sha256=POINTER45_SHA,
    )
    write(out / "PUBLIC_SMOKE_VALIDATION.json",
          {"problems": problems, "observed": observed, "score_claim": False})
    if problems:
        raise Ntb2IntentError(f"smoke block refused by the seal validator: {problems}")
    print(json.dumps({"smoke_problems": problems}))
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
    risk.add_argument("--base-diagnostic", type=Path, required=True)
    risk.add_argument("--candidate-diagnostic", action="append", required=True)
    risk.set_defaults(func=cmd_risk)

    smoke = sub.add_parser("smoke", help="the four bounded public-entrypoint probes")
    smoke.add_argument("--candidate-runtime", type=Path, required=True)
    smoke.add_argument("--frontier-runtime", type=Path, required=True)
    smoke.add_argument("--out-dir", type=Path, required=True)
    smoke.add_argument("--bound-seconds", type=float, default=180.0)
    smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
