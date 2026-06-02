#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize measured PVQ section cuts as a byte-closed archive candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.archive_bound_candidate_runtime_bridge import (  # noqa: E402
    run_generated_inflate_receiver_proof,
)
from tac.repo_io import write_json  # noqa: E402
from tac.substrates._shared.inflate_runtime import CAMERA_HW  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tac.substrates.pact_nerv_vq.section_value import (  # noqa: E402
    PVQ_SUPPORTED_SECTION_NAMES,
    neutralize_pvq_section,
    pvq_layout_report,
)
from tools.profile_pact_nerv_selector_v3_mlx_section_value import (  # noqa: E402
    _extract_submission,
    _read_archive_member,
)
from tools.profile_pact_nerv_selector_v4_mlx_section_value import (  # noqa: E402
    _sha256_file,
    _write_zip_replacing_member,
)

OWNED_MARKER = ".pact_nerv_vq_section_cut_owned.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--run-receiver-proof", action="store_true")
    parser.add_argument("--retain-receiver-proof-output", action="store_true")
    parser.add_argument("--receiver-proof-timeout-seconds", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    repo_root = _resolve(args.repo_root, base=REPO_ROOT)
    archive = _resolve(args.archive, base=repo_root)
    output_dir = _resolve(args.output_dir, base=repo_root)
    profile = None if args.profile is None else _resolve(args.profile, base=repo_root)
    _prepare_owned_dir(output_dir, force=bool(args.force))

    requested = _normalize_sections(args.sections)
    started = time.time()
    baseline_blob = _read_archive_member(archive, "0.bin")
    layout = pvq_layout_report(blob=baseline_blob)
    mutated_blob = baseline_blob
    section_rows = []
    for section in requested:
        before = mutated_blob
        mutated_blob = neutralize_pvq_section(mutated_blob, section)
        section_rows.append(
            {
                "section": section,
                "before_bytes": len(before),
                "before_sha256": _sha256_bytes(before),
                "after_bytes": len(mutated_blob),
                "after_sha256": _sha256_bytes(mutated_blob),
            }
        )

    candidate_dir = output_dir / "candidate"
    archive_zip = candidate_dir / "archive.zip"
    replacement_report = _write_zip_replacing_member(
        source_archive=archive,
        output_archive=archive_zip,
        member_name="0.bin",
        replacement_bytes=mutated_blob,
        allow_overwrite=True,
    )
    submission_dir = candidate_dir / "submission"
    _extract_submission(archive_zip, submission_dir)
    _ensure_inflate_executable(submission_dir)
    archive_sha = _sha256_file(archive_zip)
    archive_bytes = archive_zip.stat().st_size
    proof = None
    if args.run_receiver_proof:
        proof = run_generated_inflate_receiver_proof(
            archive_zip_path=archive_zip,
            archive_sha256=archive_sha,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=candidate_dir / "receiver_proof",
            repo_root=repo_root,
            candidate_label="pact_nerv_vq_section_cut",
            expected_receiver_output_name="0.raw",
            expected_receiver_output_bytes=_expected_receiver_output_bytes(layout),
            timeout_seconds=int(args.receiver_proof_timeout_seconds),
            retain_receiver_output=bool(args.retain_receiver_proof_output),
        )

    report = {
        "schema": "pact_nerv_vq_section_cut_candidate.v1",
        "tool": Path(__file__).name,
        "tool_argv": [sys.executable, str(Path(__file__).resolve()), *raw_argv],
        "repo_root": repo_root.as_posix(),
        "source_archive": _file_row(archive),
        "profile_path": None if profile is None else profile.as_posix(),
        "profile_sha256": None if profile is None else _sha256_file(profile),
        "sections_cut": requested,
        "layout": layout,
        "section_mutation_rows": section_rows,
        "zip_replacement_report": replacement_report,
        "candidate_archive": _file_row(archive_zip),
        "candidate_submission_dir": submission_dir.as_posix(),
        "receiver_proof": proof,
        "elapsed_seconds": time.time() - started,
        "blockers": [
            "contest_cpu_cuda_exact_eval_not_executed",
            *(
                []
                if proof is not None and proof.get("runtime_consumption_proof_passed")
                else ["receiver_proof_not_executed_or_not_passed"]
            ),
        ],
        **FALSE_AUTHORITY,
    }
    report_path = output_dir / "pact_nerv_vq_section_cut_candidate.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "report": report_path.as_posix(),
                "archive": archive_zip.as_posix(),
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha,
                "sections_cut": requested,
                "receiver_proof_passed": (
                    None
                    if proof is None
                    else proof.get("runtime_consumption_proof_passed")
                ),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _normalize_sections(raw_sections: list[str]) -> list[str]:
    sections: list[str] = []
    for raw in raw_sections:
        section = str(raw).strip().lower()
        if section not in PVQ_SUPPORTED_SECTION_NAMES:
            raise SystemExit(
                f"unsupported section {section!r}; "
                f"valid={tuple(sorted(PVQ_SUPPORTED_SECTION_NAMES))}"
            )
        if section == "receiver_state":
            raise SystemExit("refusing unsafe measured-cut section: receiver_state")
        if section not in sections:
            sections.append(section)
    if not sections:
        raise SystemExit("at least one section is required")
    return sections


def _expected_receiver_output_bytes(layout: dict[str, Any]) -> int:
    num_pairs = int(layout.get("num_pairs") or 0)
    return num_pairs * 2 * int(CAMERA_HW[0]) * int(CAMERA_HW[1]) * 3


def _ensure_inflate_executable(submission_dir: Path) -> None:
    inflate_sh = submission_dir / "inflate.sh"
    if inflate_sh.is_file():
        inflate_sh.chmod(inflate_sh.stat().st_mode | 0o111)


def _prepare_owned_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        marker = path / OWNED_MARKER
        if not force and any(path.iterdir()):
            raise SystemExit(f"output dir exists; pass --force: {path}")
        if force:
            if not marker.exists() and any(path.iterdir()):
                raise SystemExit(f"refusing --force on non-owned output dir: {path}")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / OWNED_MARKER).write_text(
        json.dumps({"schema": "owned_directory_marker.v1", "tool": Path(__file__).name})
        + "\n",
        encoding="utf-8",
    )


def _resolve(path: Path, *, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve(strict=False)


def _file_row(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
