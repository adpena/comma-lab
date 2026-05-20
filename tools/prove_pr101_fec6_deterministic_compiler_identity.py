#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove PR101/FEC6 archive identity through the deterministic compiler."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.deterministic_compiler import (  # noqa: E402
    MANIFEST_NAME,
    compile_packet,
)
from tac.repo_io import json_text, read_json, sha256_file  # noqa: E402

DEFAULT_SUBMISSION_DIR = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir"
)
DEFAULT_WORK_DIR = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "deterministic_identity_closure_20260519_codex"
)
DEFAULT_OUTPUT_MANIFEST = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "deterministic_packet_compiler_manifest.json"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / ".omx/research/"
    "pr101_fec6_deterministic_compiler_identity_20260519_codex.md"
)
DEFAULT_ARCHIVE_SHA256 = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)
SANITIZED_FILE_RELATIVE_PATHS = (
    "archive.zip",
    "inflate.py",
    "inflate.sh",
    "src/codec.py",
    "src/codec_sidecar.py",
    "src/frame_selector.py",
    "src/model.py",
)


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _prepare_sanitized_input(
    *,
    submission_dir: Path,
    input_dir: Path,
    allow_existing_work_dir: bool,
) -> list[str]:
    if input_dir.exists():
        if not allow_existing_work_dir:
            raise FileExistsError(
                f"sanitized input already exists: {input_dir}; "
                "pass --allow-existing-work-dir to refresh"
            )
        shutil.rmtree(input_dir)
    copied: list[str] = []
    for rel in SANITIZED_FILE_RELATIVE_PATHS:
        source = submission_dir / rel
        if not source.is_file():
            raise FileNotFoundError(f"submission file missing: {source}")
        target = input_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel)
    return copied


def _render_markdown(manifest: dict[str, object], *, output_manifest: Path) -> str:
    blockers = manifest.get("blockers")
    blocker_list = blockers if isinstance(blockers, list) else []
    lines = [
        "# PR101/FEC6 deterministic compiler identity",
        "",
        "This is a byte-custody artifact only. It does not claim score, "
        "promotion eligibility, or dispatch readiness.",
        "",
        f"- Manifest: `{_repo_relative(output_manifest)}`",
        f"- Schema: `{manifest.get('schema_version')}`",
        f"- Mode: `{manifest.get('mode')}`",
        f"- Target profile: `{manifest.get('target_profile')}`",
        f"- Archive SHA-256: `{manifest.get('archive_sha256')}`",
        f"- Archive bytes: `{manifest.get('archive_size_bytes')}`",
        f"- Runtime tree SHA-256: `{manifest.get('runtime_tree_sha256')}`",
        f"- No-op detector passed: `{(manifest.get('no_op_proof') or {}).get('no_op_detector_passed') if isinstance(manifest.get('no_op_proof'), dict) else None}`",
        f"- Score claim: `{manifest.get('score_claim')}`",
        f"- Promotion eligible: `{manifest.get('promotion_eligible')}`",
        f"- Ready for exact eval dispatch: `{manifest.get('ready_for_exact_eval_dispatch')}`",
        "",
        "## Blockers",
        "",
    ]
    if blocker_list:
        for blocker in blocker_list:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--expected-archive-sha256",
        default=DEFAULT_ARCHIVE_SHA256,
    )
    parser.add_argument(
        "--allow-existing-work-dir",
        action="store_true",
        help="Refresh the generated deterministic identity work directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.work_dir / "input_packet"
    output_dir = args.work_dir / "compiled_packet"
    copied = _prepare_sanitized_input(
        submission_dir=args.submission_dir,
        input_dir=input_dir,
        allow_existing_work_dir=args.allow_existing_work_dir,
    )
    result = compile_packet(
        input_packet=input_dir,
        output_dir=output_dir,
        mode="identity",
        target_profile="contest_one_video_replay",
        baseline_archive_sha256=args.expected_archive_sha256,
        allow_existing_output_dir=True,
    )
    manifest_path = output_dir / MANIFEST_NAME
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, args.output_manifest)
    manifest = read_json(args.output_manifest)
    if not isinstance(manifest, dict):
        raise TypeError(f"compiler manifest is not a JSON object: {args.output_manifest}")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        _render_markdown(manifest, output_manifest=args.output_manifest),
        encoding="utf-8",
    )
    summary = {
        "schema": manifest.get("schema_version"),
        "mode": manifest.get("mode"),
        "target_profile": manifest.get("target_profile"),
        "archive_sha256": manifest.get("archive_sha256"),
        "archive_size_bytes": manifest.get("archive_size_bytes"),
        "runtime_tree_sha256": manifest.get("runtime_tree_sha256"),
        "no_op_detector_passed": (
            manifest.get("no_op_proof", {}).get("no_op_detector_passed")
            if isinstance(manifest.get("no_op_proof"), dict)
            else None
        ),
        "score_claim": manifest.get("score_claim"),
        "promotion_eligible": manifest.get("promotion_eligible"),
        "ready_for_exact_eval_dispatch": manifest.get(
            "ready_for_exact_eval_dispatch"
        ),
        "blockers": manifest.get("blockers"),
        "copied_inputs": copied,
        "outputs": {
            "manifest": _repo_relative(args.output_manifest),
            "manifest_sha256": sha256_file(args.output_manifest),
            "markdown": _repo_relative(args.output_md),
            "markdown_sha256": sha256_file(args.output_md),
        },
    }
    sys.stdout.write(json_text(summary))
    return 0 if not result.blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
