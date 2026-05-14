#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate an existing PR101 Brotli ``lgwin=18`` archive candidate.

This is a local byte-custody and decoder-parity guard. It compares a canonical
PR101 source archive against a supplied candidate archive, verifies the
candidate is not a no-op, and checks that the candidate decoder bytes decode to
the same PR101 state as the source decoder bytes.

The tool never imports contest scorers, runs CUDA, dispatches GPU work, or
emits SegNet/PoseNet/score estimates. A passing report is still blocked on
exact runtime custody plus the lane-dispatch claim gate before any CUDA auth
eval can run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tac.submission_packet_compiler import TARGET_PROFILE_POLICIES  # noqa: E402

TOOL_NAME = "tools/validate_pr101_lgwin18_candidate.py"
SCHEMA_VERSION = "pr101_lgwin18_candidate_validation.v2"
TARGET_PROFILES = tuple(TARGET_PROFILE_POLICIES)
DEFAULT_BROTLI_QUALITY = 11
DEFAULT_BROTLI_LGWIN = 18

PASS_VERDICT = "LOCAL_RUNTIME_PARITY_PASS_EXACT_EVAL_BLOCKED_NO_SCORE_CLAIM"
FAIL_VERDICT = "FAIL_CLOSED_LOCAL_VALIDATION_BLOCKED_NO_SCORE_CLAIM"


@dataclass(frozen=True)
class ArchiveSections:
    path: Path
    archive_size_bytes: int
    archive_sha256: str
    inner_member_name: str
    inner_member_bytes: int
    inner_member_sha256: str
    decoder_blob: bytes
    decoder_blob_sha256: str
    latent_blob: bytes
    latent_blob_sha256: str
    sidecar_blob: bytes
    sidecar_blob_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "path": _display_path(self.path),
            "archive_size_bytes": self.archive_size_bytes,
            "archive_sha256": self.archive_sha256,
            "inner_member": {
                "name": self.inner_member_name,
                "bytes": self.inner_member_bytes,
                "sha256": self.inner_member_sha256,
            },
            "decoder_blob": {
                "bytes": len(self.decoder_blob),
                "sha256": self.decoder_blob_sha256,
            },
            "latent_blob": {
                "bytes": len(self.latent_blob),
                "sha256": self.latent_blob_sha256,
            },
            "sidecar_blob": {
                "bytes": len(self.sidecar_blob),
                "sha256": self.sidecar_blob_sha256,
            },
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_archive_sections(path: Path) -> ArchiveSections:
    from pr101_archive_substitution_surgery import (
        PR101_INNER_MEMBER_NAME,
        _read_inner_blob,
        _split_pr101_inner_blob,
    )

    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {_display_path(path)}")
    inner = _read_inner_blob(path)
    decoder_blob, latent_blob, sidecar_blob = _split_pr101_inner_blob(inner)
    return ArchiveSections(
        path=path,
        archive_size_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        inner_member_name=PR101_INNER_MEMBER_NAME,
        inner_member_bytes=len(inner),
        inner_member_sha256=_sha256_bytes(inner),
        decoder_blob=decoder_blob,
        decoder_blob_sha256=_sha256_bytes(decoder_blob),
        latent_blob=latent_blob,
        latent_blob_sha256=_sha256_bytes(latent_blob),
        sidecar_blob=sidecar_blob,
        sidecar_blob_sha256=_sha256_bytes(sidecar_blob),
    )


def _compare_state_dicts(
    source_state: dict[str, Any],
    candidate_state: dict[str, Any],
) -> dict[str, Any]:
    import torch

    failures: list[str] = []
    source_keys = set(source_state)
    candidate_keys = set(candidate_state)
    for name in sorted(source_keys - candidate_keys):
        failures.append(f"candidate missing tensor {name}")
    for name in sorted(candidate_keys - source_keys):
        failures.append(f"candidate has unexpected tensor {name}")
    compared = 0
    for name in sorted(source_keys & candidate_keys):
        source_tensor = source_state[name]
        candidate_tensor = candidate_state[name]
        if tuple(candidate_tensor.shape) != tuple(source_tensor.shape):
            failures.append(
                f"{name}: shape mismatch {tuple(candidate_tensor.shape)} "
                f"!= {tuple(source_tensor.shape)}"
            )
            continue
        if candidate_tensor.dtype != source_tensor.dtype:
            failures.append(
                f"{name}: dtype mismatch {candidate_tensor.dtype} "
                f"!= {source_tensor.dtype}"
            )
            continue
        if not torch.equal(candidate_tensor, source_tensor):
            failures.append(f"{name}: decoded tensor bytes differ")
            continue
        compared += 1
    return {
        "passed": not failures,
        "compared_tensors": compared,
        "failures": failures,
    }


def _decoder_parity_checks(
    source: ArchiveSections,
    candidate: ArchiveSections,
    *,
    brotli_quality: int,
    brotli_lgwin: int,
) -> dict[str, Any]:
    from tac.pr101_split_brotli_codec import (
        decode_decoder_compact,
        encode_decoder_compact,
    )

    source_state = decode_decoder_compact(source.decoder_blob)
    candidate_state = decode_decoder_compact(candidate.decoder_blob)
    state_parity = _compare_state_dicts(source_state, candidate_state)

    source_default = encode_decoder_compact(
        source_state,
        brotli_quality=brotli_quality,
    )
    candidate_default = encode_decoder_compact(
        candidate_state,
        brotli_quality=brotli_quality,
    )
    source_lgwin = encode_decoder_compact(
        source_state,
        brotli_quality=brotli_quality,
        brotli_lgwin=brotli_lgwin,
    )

    return {
        "decoder_state_dict_parity": state_parity,
        "source_default_reencode": {
            "sha256": _sha256_bytes(source_default),
            "matches_source_decoder_blob": source_default == source.decoder_blob,
        },
        "candidate_default_reencode": {
            "sha256": _sha256_bytes(candidate_default),
            "matches_source_decoder_blob": candidate_default == source.decoder_blob,
        },
        "source_lgwin_reencode": {
            "brotli_quality": brotli_quality,
            "brotli_lgwin": brotli_lgwin,
            "sha256": _sha256_bytes(source_lgwin),
            "matches_candidate_decoder_blob": source_lgwin == candidate.decoder_blob,
        },
    }


def _substitution_report_checks(
    substitution_report: Path | None,
    *,
    source: ArchiveSections,
    candidate: ArchiveSections,
) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {
        "provided": substitution_report is not None,
        "path": _display_path(substitution_report) if substitution_report else None,
        "checks": {},
        "errors": [],
    }
    blockers: list[str] = []
    if substitution_report is None:
        return summary, blockers
    if not substitution_report.is_file():
        msg = "substitution_report_missing"
        summary["errors"].append(msg)
        blockers.append(msg)
        return summary, blockers
    try:
        data = json.loads(substitution_report.read_text())
    except json.JSONDecodeError as exc:
        msg = f"substitution_report_invalid_json:{exc.msg}"
        summary["errors"].append(msg)
        blockers.append("substitution_report_invalid_json")
        return summary, blockers

    expected_fields = {
        "sha256_input_archive": source.archive_sha256,
        "sha256_output_archive": candidate.archive_sha256,
        "sha256_input_inner_member": source.inner_member_sha256,
        "sha256_output_inner_member": candidate.inner_member_sha256,
        "sha256_input_decoder_blob": source.decoder_blob_sha256,
        "sha256_replacement_decoder_blob": candidate.decoder_blob_sha256,
        "sha256_input_latent_blob": source.latent_blob_sha256,
        "sha256_output_latent_blob": candidate.latent_blob_sha256,
        "sha256_input_sidecar_blob": source.sidecar_blob_sha256,
        "sha256_output_sidecar_blob": candidate.sidecar_blob_sha256,
        "input_size_bytes": source.archive_size_bytes,
        "output_size_bytes": candidate.archive_size_bytes,
        "decoder_blob_input_len": len(source.decoder_blob),
        "decoder_blob_replacement_len": len(candidate.decoder_blob),
        "latent_blob_len": len(candidate.latent_blob),
        "sidecar_blob_len": len(candidate.sidecar_blob),
    }
    for key, expected in expected_fields.items():
        actual = data.get(key)
        passed = actual == expected
        summary["checks"][key] = {
            "expected": expected,
            "actual": actual,
            "passed": passed,
        }
        if not passed:
            blockers.append(f"substitution_report_{key}_mismatch")
    return summary, blockers


def _append_failed_check(
    blockers: list[str],
    checks: dict[str, Any],
    key: str,
    reason: str,
) -> None:
    if checks.get(key) is False:
        blockers.append(reason)


def validate_candidate(
    *,
    source_archive: Path,
    candidate_archive: Path,
    expected_candidate_sha256: str | None = None,
    expected_candidate_size_bytes: int | None = None,
    substitution_report: Path | None = None,
    target_profile: str = "contest_one_video_replay",
    brotli_quality: int = DEFAULT_BROTLI_QUALITY,
    brotli_lgwin: int = DEFAULT_BROTLI_LGWIN,
) -> dict[str, Any]:
    """Return a deterministic validation report for a PR101 candidate archive."""
    validation_blockers: list[str] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "target_profile": target_profile,
        "target_profile_context": {
            "primary": target_profile,
            "policy": TARGET_PROFILE_POLICIES.get(target_profile),
            "contest_one_video_replay": (
                "PR101 is a fixed contest-video replay packet; local byte and "
                "decoder parity are relevant preconditions for exact auth eval."
            ),
            "production_generalized": (
                "not claimed by this validation; no cross-video or openpilot "
                "generalization evidence is produced"
            ),
        },
        "score_claim": False,
        "evidence_grade": "empirical",
        "evidence_limits": [
            "local byte custody only",
            "local PR101 decoder parity only",
            "no contest scorer import",
            "no CUDA auth eval",
            "no SegNet/PoseNet/component estimates",
            "no GPU dispatch",
        ],
        "dispatch": {
            "gpu_dispatch_performed": False,
            "gpu_dispatch_authorized_by_this_tool": False,
        },
    }

    if target_profile not in TARGET_PROFILES:
        validation_blockers.append(f"unknown_target_profile:{target_profile}")
    if target_profile in TARGET_PROFILE_POLICIES and target_profile != "contest_one_video_replay":
        validation_blockers.append(
            f"{target_profile}_not_validated_by_pr101_one_video_tool"
        )

    try:
        source = _read_archive_sections(source_archive)
        candidate = _read_archive_sections(candidate_archive)
    except Exception as exc:
        report.update(
            {
                "source_archive": {"path": _display_path(source_archive)},
                "candidate_archive": {"path": _display_path(candidate_archive)},
                "checks": {},
                "validation_blockers": [
                    *validation_blockers,
                    "archive_layout_or_read_failed",
                ],
                "failure_detail": f"{type(exc).__name__}: {exc}",
                "verdict": FAIL_VERDICT,
                "exact_eval_blockers": _exact_eval_blockers(),
            }
        )
        return report

    checks: dict[str, Any] = {
        "candidate_archive_byte_different": (
            source.archive_sha256 != candidate.archive_sha256
        ),
        "candidate_inner_member_byte_different": (
            source.inner_member_sha256 != candidate.inner_member_sha256
        ),
        "candidate_decoder_blob_byte_different": (
            source.decoder_blob_sha256 != candidate.decoder_blob_sha256
        ),
        "decoder_blob_length_preserved": (
            len(source.decoder_blob) == len(candidate.decoder_blob)
        ),
        "latent_blob_preserved": source.latent_blob_sha256
        == candidate.latent_blob_sha256,
        "sidecar_blob_preserved": source.sidecar_blob_sha256
        == candidate.sidecar_blob_sha256,
        "archive_size_preserved": source.archive_size_bytes
        == candidate.archive_size_bytes,
        "expected_candidate_sha256": expected_candidate_sha256,
        "candidate_sha256_matches_expected": (
            None
            if expected_candidate_sha256 is None
            else candidate.archive_sha256 == expected_candidate_sha256
        ),
        "expected_candidate_size_bytes": expected_candidate_size_bytes,
        "candidate_size_matches_expected": (
            None
            if expected_candidate_size_bytes is None
            else candidate.archive_size_bytes == expected_candidate_size_bytes
        ),
        "charged_bits_changed": source.inner_member_sha256
        != candidate.inner_member_sha256,
        "score_affecting_payload_changed": source.inner_member_sha256
        != candidate.inner_member_sha256,
    }

    _append_failed_check(
        validation_blockers,
        checks,
        "candidate_archive_byte_different",
        "candidate_archive_not_byte_different_against_source",
    )
    _append_failed_check(
        validation_blockers,
        checks,
        "candidate_inner_member_byte_different",
        "candidate_inner_member_noop_against_source",
    )
    _append_failed_check(
        validation_blockers,
        checks,
        "candidate_decoder_blob_byte_different",
        "candidate_decoder_blob_noop_against_source",
    )
    _append_failed_check(
        validation_blockers,
        checks,
        "decoder_blob_length_preserved",
        "decoder_blob_length_changed",
    )
    _append_failed_check(
        validation_blockers,
        checks,
        "latent_blob_preserved",
        "latent_blob_changed",
    )
    _append_failed_check(
        validation_blockers,
        checks,
        "sidecar_blob_preserved",
        "sidecar_blob_changed",
    )
    if checks["candidate_sha256_matches_expected"] is False:
        validation_blockers.append("candidate_sha256_mismatch")
    if checks["candidate_size_matches_expected"] is False:
        validation_blockers.append("candidate_size_mismatch")

    try:
        decoder_checks = _decoder_parity_checks(
            source,
            candidate,
            brotli_quality=brotli_quality,
            brotli_lgwin=brotli_lgwin,
        )
    except Exception as exc:
        decoder_checks = {
            "decoder_check_failed": True,
            "failure_detail": f"{type(exc).__name__}: {exc}",
        }
        validation_blockers.append("local_decoder_parity_check_failed_to_run")
    else:
        if not decoder_checks["decoder_state_dict_parity"]["passed"]:
            validation_blockers.append("local_decoder_state_dict_parity_failed")
        if not decoder_checks["source_default_reencode"][
            "matches_source_decoder_blob"
        ]:
            validation_blockers.append(
                "source_default_reencode_does_not_match_source_decoder"
            )
        if not decoder_checks["candidate_default_reencode"][
            "matches_source_decoder_blob"
        ]:
            validation_blockers.append(
                "candidate_default_reencode_does_not_match_source_decoder"
            )
        if not decoder_checks["source_lgwin_reencode"][
            "matches_candidate_decoder_blob"
        ]:
            validation_blockers.append(
                "source_lgwin_reencode_does_not_match_candidate_decoder"
            )

    substitution_summary, substitution_blockers = _substitution_report_checks(
        substitution_report,
        source=source,
        candidate=candidate,
    )
    validation_blockers.extend(substitution_blockers)

    report.update(
        {
            "source_archive": source.summary(),
            "candidate_archive": candidate.summary(),
            "checks": checks,
            "decoder_parity_checks": decoder_checks,
            "substitution_report": substitution_summary,
            "validation_blockers": validation_blockers,
            "exact_eval_blockers": _exact_eval_blockers(),
            "verdict": PASS_VERDICT if not validation_blockers else FAIL_VERDICT,
        }
    )
    return report


def _exact_eval_blockers() -> list[str]:
    return [
        "exact runtime parity/runtime-tree custody is not established by this CPU validator",
        "matching active lane dispatch claim is required before any GPU dispatch",
        "CUDA auth eval on archive.zip -> inflate.sh -> upstream/evaluate.py has not been run",
    ]


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--substitution-report", type=Path, default=None)
    parser.add_argument("--expected-candidate-sha256", default=None)
    parser.add_argument("--expected-candidate-size-bytes", type=int, default=None)
    parser.add_argument(
        "--target-profile",
        choices=TARGET_PROFILES,
        default="contest_one_video_replay",
    )
    parser.add_argument(
        "--brotli-quality",
        type=int,
        default=DEFAULT_BROTLI_QUALITY,
    )
    parser.add_argument(
        "--brotli-lgwin",
        type=int,
        default=DEFAULT_BROTLI_LGWIN,
    )
    args = parser.parse_args(argv)

    report = validate_candidate(
        source_archive=args.source_archive,
        candidate_archive=args.candidate_archive,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_candidate_size_bytes=args.expected_candidate_size_bytes,
        substitution_report=args.substitution_report,
        target_profile=args.target_profile,
        brotli_quality=args.brotli_quality,
        brotli_lgwin=args.brotli_lgwin,
    )
    if args.report is not None:
        write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
