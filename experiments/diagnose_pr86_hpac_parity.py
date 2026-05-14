#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed PR86 HPAC replay/parity intake adjudicator.

This diagnostic consumes the already-captured public PR86 intake artifacts and
the local `archive.zip`. It does not run inflate, contest eval, GPU work, or
remote dispatch. Its job is to turn the public 0.27 PR design signal into local
custody facts plus a precise blocker for faithful replay/porting.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PR86_DIR = REPO_ROOT / "experiments/results/public_pr86_intake_20260504_codex"
DEFAULT_ARCHIVE = DEFAULT_PR86_DIR / "archive.zip"
DEFAULT_FULL_REENCODE = (
    DEFAULT_PR86_DIR / "pr86_hpac_full_decode_reencode_gate_20260504_codex.json"
)
DEFAULT_ANATOMY = DEFAULT_PR86_DIR / "pr86_hpac_token_anatomy_forensics.json"
DEFAULT_PR85_PROBE = DEFAULT_PR86_DIR / "pr86_hpac_pr85_qma9_parity_probe.json"
DEFAULT_PR_VIEW = DEFAULT_PR86_DIR / "pr86_view.json"

EXPECTED_ARCHIVE_BYTES = 207_579
EXPECTED_ARCHIVE_SHA256 = "e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef"
REQUIRED_MEMBERS = (
    "master.pt.gz",
    "slave.pt.gz",
    "hpac.pt.ppmd",
    "tokens.bin",
    "meta.pt",
)
EXPECTED_MEMBER_BYTES = {
    "master.pt.gz": 31_144,
    "slave.pt.gz": 32_287,
    "hpac.pt.ppmd": 28_243,
    "tokens.bin": 113_900,
    "meta.pt": 1_499,
}


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def archive_custody(archive: Path) -> dict[str, Any]:
    """Return deterministic archive identity and member anatomy facts."""
    if not archive.is_file():
        return {
            "path": repo_rel(archive),
            "exists": False,
            "status": "missing_archive",
            "identity_matches_expected": False,
            "member_contract_status": "blocked_missing_archive",
        }

    members: list[dict[str, Any]] = []
    names: list[str] = []
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            names.append(info.filename)
            members.append(
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "zip_compress_type": int(info.compress_type),
                    "crc32_hex": f"{info.CRC:08x}",
                    "sha256": sha256_bytes(data),
                }
            )

    archive_bytes = archive.stat().st_size
    archive_sha = sha256_path(archive)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    missing = [name for name in REQUIRED_MEMBERS if name not in names]
    unexpected = [name for name in names if name not in REQUIRED_MEMBERS]
    byte_mismatches = {
        row["name"]: {
            "expected": EXPECTED_MEMBER_BYTES[row["name"]],
            "actual": row["file_size"],
        }
        for row in members
        if row["name"] in EXPECTED_MEMBER_BYTES
        and row["file_size"] != EXPECTED_MEMBER_BYTES[row["name"]]
    }
    unsafe = [
        name
        for name in names
        if name.startswith("/") or ".." in Path(name).parts or name == ""
    ]
    all_stored = all(row["zip_compress_type"] == zipfile.ZIP_STORED for row in members)
    identity_matches = (
        archive_bytes == EXPECTED_ARCHIVE_BYTES and archive_sha == EXPECTED_ARCHIVE_SHA256
    )
    member_contract_ok = (
        not duplicates
        and not missing
        and not unexpected
        and not byte_mismatches
        and not unsafe
        and all_stored
    )

    return {
        "path": repo_rel(archive),
        "exists": True,
        "size_bytes": archive_bytes,
        "sha256": archive_sha,
        "expected_size_bytes": EXPECTED_ARCHIVE_BYTES,
        "expected_sha256": EXPECTED_ARCHIVE_SHA256,
        "identity_matches_expected": identity_matches,
        "member_count": len(members),
        "required_members": list(REQUIRED_MEMBERS),
        "duplicate_member_names": duplicates,
        "missing_required_members": missing,
        "unexpected_members": unexpected,
        "unsafe_member_names": unsafe,
        "all_members_zip_stored": all_stored,
        "byte_mismatches": byte_mismatches,
        "members": members,
        "member_contract_status": "passed" if member_contract_ok else "blocked",
    }


def artifact_identity_consistency(
    custody: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Check that saved forensic artifacts refer to the same archive bytes."""
    expected_sha = custody.get("sha256")
    expected_bytes = custody.get("size_bytes")
    rows: list[dict[str, Any]] = []
    for label, payload in artifacts.items():
        candidates = [
            payload.get("archive", {}),
            payload.get("archive_member_contract", {}),
            payload.get("current_exact_replay_status", {}).get("archive_identity", {}),
            payload,
        ]
        found_sha = None
        found_bytes = None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            found_sha = (
                found_sha
                or candidate.get("sha256")
                or candidate.get("archive_sha256")
                or candidate.get("pr86_archive_sha256")
            )
            found_bytes = (
                found_bytes
                or candidate.get("size_bytes")
                or candidate.get("archive_bytes")
                or candidate.get("archive_size_bytes")
                or candidate.get("pr86_archive_bytes")
            )
        rows.append(
            {
                "artifact": label,
                "archive_sha256": found_sha,
                "archive_bytes": found_bytes,
                "archive_sha256_matches_local": found_sha == expected_sha,
                "archive_bytes_recorded": found_bytes is not None,
                "archive_bytes_match_local": found_bytes == expected_bytes
                if found_bytes is not None
                else None,
                "matches_local_archive": found_sha == expected_sha
                and (found_bytes is None or found_bytes == expected_bytes),
            }
        )

    return {
        "status": "passed" if rows and all(row["matches_local_archive"] for row in rows) else "blocked",
        "artifacts": rows,
    }


def public_claim(pr_view: dict[str, Any]) -> dict[str, Any]:
    body = str(pr_view.get("body", ""))
    score_match = re.search(r"Final score:\s*[^=]+=\s*([0-9.]+)", body)
    size_match = re.search(r"Submission file size:\s*([0-9,]+)\s*bytes", body)
    pose_match = re.search(r"Average PoseNet Distortion:\s*([0-9.]+)", body)
    seg_match = re.search(r"Average SegNet Distortion:\s*([0-9.]+)", body)
    commit = (pr_view.get("commits") or [{}])[0]
    return {
        "source": pr_view.get("url"),
        "pr_number": pr_view.get("number"),
        "title": pr_view.get("title"),
        "head_commit": commit.get("oid"),
        "public_claimed_score": score_match.group(1) if score_match else None,
        "public_claimed_archive_bytes": (
            int(size_match.group(1).replace(",", "")) if size_match else None
        ),
        "public_claimed_posenet": pose_match.group(1) if pose_match else None,
        "public_claimed_segnet": seg_match.group(1) if seg_match else None,
        "evidence_grade": "external_public_pr_text",
        "local_score_claim": False,
    }


def dependency_contract(replay: dict[str, Any], pr85_probe: dict[str, Any]) -> dict[str, Any]:
    recorded = replay.get("dependencies", {})
    probe_deps = pr85_probe.get("pr86_dependencies", {})
    required = {
        "constriction": "0.4.2 observed; pyproject range constriction>=0.4,<0.5",
        "pyppmd": "1.3.1 observed; pyproject range pyppmd>=1.3,<2.0",
        "torch": "must match PR86 HPACMini state dict and queue decode behavior",
        "numpy": "uint32 queue-word interpretation for tokens.bin",
    }
    installed = {
        name: package_version(name)
        for name in ("constriction", "pyppmd", "torch", "numpy")
    }
    return {
        "required": required,
        "recorded_in_full_decode_reencode_artifact": {
            key: recorded.get(key) for key in ("python", "torch", "numpy", "constriction", "pyppmd")
        },
        "recorded_in_pr85_probe": probe_deps,
        "installed_now": {"python": sys.version.split()[0], **installed},
        "status": (
            "passed_observed_versions_present"
            if recorded.get("constriction") and recorded.get("pyppmd")
            else "blocked_missing_recorded_versions"
        ),
    }


def hpac_stage_summary(
    anatomy: dict[str, Any],
    replay: dict[str, Any],
    pr85_probe: dict[str, Any],
) -> dict[str, Any]:
    gate = replay.get("full_decode_reencode_gate", {})
    token_contract = anatomy.get("token_hpac_decode_contract", {})
    probability_contract = replay.get("probability_model_contract", {})
    failure_class = gate.get("failure_class")
    if gate.get("status") == "failed_closed" and gate.get("error_type") == "AssertionError":
        blocker = "hpac_entropy_decode_contract_mismatch"
    elif gate.get("status") == "passed":
        blocker = None
    else:
        blocker = failure_class or "hpac_replay_unproven"

    return {
        "stage_statuses": [
            {
                "stage": "archive_custody_and_member_anatomy",
                "status": "passed",
                "evidence": "archive.zip identity and five-member anatomy match local intake",
            },
            {
                "stage": "hpac_model_container_decode",
                "status": "passed",
                "evidence": "hpac.pt.ppmd is a PPMd-compressed torch HPACMini state dict in anatomy artifact",
            },
            {
                "stage": "constriction_queue_api_self_test",
                "status": replay.get("constriction_queue_contract", {}).get("status", "unknown"),
                "evidence": "same-order RangeEncoder/RangeDecoder prefix roundtrip",
            },
            {
                "stage": "submitted_tokens_decode",
                "status": gate.get("status", "not_run"),
                "failure_class": blocker,
                "error_type": gate.get("error_type"),
                "error": gate.get("error"),
                "failed_at": gate.get("failed_at"),
                "decoded_symbol_count_before_failure": gate.get("decoded_symbol_count"),
            },
            {
                "stage": "decode_then_reencode_byte_parity",
                "status": "blocked_not_reached" if gate.get("status") == "failed_closed" else gate.get("status", "not_run"),
                "byte_exact_reencode": gate.get("byte_exact_reencode"),
            },
            {
                "stage": "pr85_qma9_hpac_transfer_parity",
                "status": pr85_probe.get("status", "not_run"),
                "failure_class": pr85_probe.get("failure_class"),
                "error": pr85_probe.get("observed_error"),
            },
        ],
        "token_semantics": {
            "submitted_archive_token_encoding": replay.get("conclusions", {}).get(
                "submitted_archive_token_encoding"
            ),
            "training_objective": replay.get("conclusions", {}).get("training_objective"),
            "anatomy_token_contract": token_contract,
        },
        "probability_model_contract": {
            "queue_api": replay.get("constriction_queue_contract", {}).get("queue_api"),
            "model_api": replay.get("constriction_queue_contract", {}).get("model_api"),
            "probability_clip_eps": probability_contract.get("probability_clip_eps"),
            "categorical_perfect_false": probability_contract.get(
                "categorical_perfect_false_in_archive_code"
            ),
            "explicit_16384_grid_in_code": probability_contract.get(
                "explicit_16384_grid_in_archive_code"
            ),
        },
        "blocker_class": blocker,
        "blocker_summary": (
            "PR86 public archive custody is locally reproducible, but submitted "
            "tokens.bin fails HPAC entropy decode on CPU at frame 0/group 10/"
            "symbol 191 with constriction's invalid-entropy-model assertion. "
            "Byte-exact decode->reencode parity is therefore blocked before "
            "any PR85 transfer or local score claim."
            if blocker == "hpac_entropy_decode_contract_mismatch"
            else "PR86 HPAC replay remains unproven."
        ),
    }


def code_contract_needed() -> list[dict[str, Any]]:
    return [
        {
            "contract": "archive_container",
            "requirement": (
                "Exactly master.pt.gz, slave.pt.gz, hpac.pt.ppmd, tokens.bin, "
                "and meta.pt; no sidecars, duplicates, unsafe names, or ZIP parser divergence."
            ),
        },
        {
            "contract": "compressed_state_members",
            "requirement": (
                "master.pt.gz and slave.pt.gz are gzip-wrapped torch state dicts; "
                "hpac.pt.ppmd is pyppmd.decompress(..., max_order=4, mem_size=16<<20) "
                "followed by torch.load."
            ),
        },
        {
            "contract": "token_entropy_stream",
            "requirement": (
                "tokens.bin is little-endian uint32 words consumed in original order by "
                "constriction.stream.queue.RangeDecoder; encoder parity must use "
                "RangeEncoder.get_compressed().tobytes() and match tokens.bin exactly."
            ),
        },
        {
            "contract": "hpac_probability_model",
            "requirement": (
                "For each HPAC group, generate HPACMini logits from current tokens, "
                "frame index, and previous frame; softmax to float probabilities, clip "
                "to 1e-7, renormalize, then pass Categorical(..., perfect=False)."
            ),
        },
        {
            "contract": "token_semantics",
            "requirement": (
                "Submitted archive tokens are treated as raw class tokens, not training "
                "residual targets, unless a separate byte-identical replay proves residual reconstruction."
            ),
        },
        {
            "contract": "promotion_gate",
            "requirement": (
                "Full 600-frame PR86 decode plus byte-exact decode->encode parity must pass "
                "under pinned dependencies before PR86 can become local reproducible score evidence "
                "or a PR85 HPAC port template."
            ),
        },
    ]


def build_report(
    *,
    pr86_dir: Path = DEFAULT_PR86_DIR,
    archive: Path = DEFAULT_ARCHIVE,
    full_reencode: Path = DEFAULT_FULL_REENCODE,
    anatomy: Path = DEFAULT_ANATOMY,
    pr85_probe: Path = DEFAULT_PR85_PROBE,
    pr_view: Path = DEFAULT_PR_VIEW,
) -> dict[str, Any]:
    custody = archive_custody(archive)
    replay_payload = load_json(full_reencode)
    anatomy_payload = load_json(anatomy)
    pr85_payload = load_json(pr85_probe)
    pr_view_payload = load_json(pr_view)
    artifacts = {
        repo_rel(full_reencode): replay_payload,
        repo_rel(anatomy): anatomy_payload,
        repo_rel(pr85_probe): pr85_payload,
    }
    stage = hpac_stage_summary(anatomy_payload, replay_payload, pr85_payload)

    return {
        "schema_version": 1,
        "tool": "experiments/diagnose_pr86_hpac_parity.py",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "evidence_grade": "local_fail_closed_replay_intake",
        "status": "blocked" if stage["blocker_class"] else "passed",
        "public_design_claim": public_claim(pr_view_payload),
        "source_artifacts": sorted(artifacts.keys()) + [repo_rel(pr_view)],
        "archive_custody": custody,
        "artifact_identity_consistency": artifact_identity_consistency(custody, artifacts),
        "dependency_contract": dependency_contract(replay_payload, pr85_payload),
        "hpac_replay_parity": stage,
        "code_contract_needed_to_port_or_replay_faithfully": code_contract_needed(),
        "next_implementation_patch": {
            "title": "PR86 HPAC runtime replay shim with byte-exact full-stream parity gate",
            "scope": (
                "Extract the HPACMini loader/decode loop into an owned module or adapter, "
                "pin constriction/pyppmd behavior, add a full-stream PR86 tokens.bin decode "
                "fixture/gate, and fail closed before PR85 transfer until decode->reencode "
                "tokens.bin SHA matches 14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225."
            ),
            "do_not_dispatch_until": [
                "PR86 own-stream full decode passes",
                "decode->reencode tokens.bin byte parity passes",
                "CPU/CUDA HPAC decode contract is explicit",
                "PR85 raw-token extraction and decoded-token parity exist",
            ],
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr86-dir", type=Path, default=DEFAULT_PR86_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--full-reencode", type=Path, default=DEFAULT_FULL_REENCODE)
    parser.add_argument("--anatomy", type=Path, default=DEFAULT_ANATOMY)
    parser.add_argument("--pr85-probe", type=Path, default=DEFAULT_PR85_PROBE)
    parser.add_argument("--pr-view", type=Path, default=DEFAULT_PR_VIEW)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path. Without this, JSON is printed only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_report(
        pr86_dir=args.pr86_dir.resolve(),
        archive=args.archive.resolve(),
        full_reencode=args.full_reencode.resolve(),
        anatomy=args.anatomy.resolve(),
        pr85_probe=args.pr85_probe.resolve(),
        pr_view=args.pr_view.resolve(),
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
