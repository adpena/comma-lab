#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the research-only G25 population-global same-state recode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.witness_dsl.ep725_lossless_xcodec_recode import inspect_source_zip, parse_ep725_lvls1
from tac.witness_dsl.ep725_population_global_recode_v2 import (
    G20_CONTROL_ARCHIVE_SHA256,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_MEMBER_SHA256,
    SearchResumeStateV2,
    config_from_json,
    search_population_global_recode_v2,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/archive.zip"
)
DEFAULT_SOURCE_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py"
)
DEFAULT_G20_CONTROL = ROOT / (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "ep725_lossless_xcodec_recode_20260726/ep725_lossless_xcodec_recode.not_a_candidate.zip"
)
DEFAULT_OUTPUT_DIR = ROOT / (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_population_global_recode_v2_20260726_r2"
)
EXPECTED_RUNTIME_SHA256 = "4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224"


class MaterializePopulationGlobalRecodeError(RuntimeError):
    """Materializer custody, resume, or durable-write invariant failed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise MaterializePopulationGlobalRecodeError(
            f"durable no-replace artifact already exists with different bytes: {path}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _path_receipt(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _load_resume(output_dir: Path, parsed: Any) -> SearchResumeStateV2 | None:
    checkpoints = sorted((output_dir / "checkpoints").glob("stage_cycle*_*.json"))
    if not checkpoints:
        return None
    try:
        value = json.loads(checkpoints[-1].read_bytes())
        state = value["next_resume_state"]
        return SearchResumeStateV2(
            config=config_from_json(state["config"], parsed),
            cycle_index=int(state["cycle_index"]),
            next_coordinate_index=int(state["next_coordinate_index"]),
            cycle_start_archive_sha256=state["cycle_start_archive_sha256"],
            points_measured=int(state["points_measured"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MaterializePopulationGlobalRecodeError(
            f"latest resume checkpoint is malformed: {checkpoints[-1]}"
        ) from exc


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute_reviewed:
        raise MaterializePopulationGlobalRecodeError("refusing execution without --execute-reviewed")
    source_archive_path = args.source_archive.resolve()
    source_runtime_path = args.source_runtime.resolve()
    g20_control_path = args.g20_control.resolve()
    output_dir = (args.resume_from or args.output_dir).resolve()
    if args.resume_from is None and output_dir.exists() and any(output_dir.iterdir()):
        raise MaterializePopulationGlobalRecodeError(f"nonempty run directory requires --resume-from: {output_dir}")
    for path, label in (
        (source_archive_path, "source archive"),
        (source_runtime_path, "source runtime"),
        (g20_control_path, "G20 control"),
    ):
        if not path.is_file():
            raise MaterializePopulationGlobalRecodeError(f"{label} is missing: {path}")
    if not str(source_archive_path).startswith(("/Volumes/VertigoDataTier/pact/", "/Volumes/APDataStore/pact/")):
        raise MaterializePopulationGlobalRecodeError(
            "real n600 source archive must remain on the preferred SSD waterfall"
        )

    source_archive = source_archive_path.read_bytes()
    source_runtime = source_runtime_path.read_bytes()
    g20_control = g20_control_path.read_bytes()
    if _sha256(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise MaterializePopulationGlobalRecodeError("source archive SHA-256 drifted")
    if _sha256(source_runtime) != EXPECTED_RUNTIME_SHA256:
        raise MaterializePopulationGlobalRecodeError("source runtime SHA-256 drifted")
    if _sha256(g20_control) != G20_CONTROL_ARCHIVE_SHA256:
        raise MaterializePopulationGlobalRecodeError("G20 control archive SHA-256 drifted")
    source_profile = inspect_source_zip(
        source_archive,
        expected_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_member_sha256=SOURCE_MEMBER_SHA256,
    )
    parsed = parse_ep725_lvls1(source_profile.member_bytes, require_source_form=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_once(
        checkpoint_dir / "stage000_preflight.json",
        _canonical_json(
            {
                "schema": "tac.ep725_population_global_recode_preflight.v2",
                "source_archive": _path_receipt(source_archive_path),
                "source_runtime": _path_receipt(source_runtime_path),
                "g20_control_archive": _path_receipt(g20_control_path),
                "population_shape": [600, 2, 32],
                "source_state_arrays": len(parsed.base_order) + 1,
                "storage": {
                    "source_on_preferred_ssd": True,
                    "bulk_scratch_created": False,
                    "scratch_strategy": "memory_resident_complete_archive_points",
                    "cleanup_required": False,
                },
                "truth": {
                    "research_only": True,
                    "scorer_or_eval_invoked": False,
                    "candidate_claim": False,
                },
            }
        ),
    )
    resume = _load_resume(output_dir, parsed) if args.resume_from is not None else None

    stage_serial = len(list(checkpoint_dir.glob("stage_cycle*_*.json")))

    def checkpoint(stage: dict[str, Any]) -> None:
        nonlocal stage_serial
        path = checkpoint_dir / f"stage_cycle{stage_serial:04d}_{stage['coordinate']}.json"
        _write_once(path, _canonical_json(stage))
        stage_serial += 1

    result = search_population_global_recode_v2(
        source_archive,
        g20_control,
        resume=resume,
        checkpoint_callback=checkpoint,
    )
    if result.selected_control_name != "g25_v2":
        blocker = {
            "schema": "tac.ep725_population_global_recode_blocker.v2",
            "blocker": "SEALED_V2_MENU_DID_NOT_BEAT_COMPLETE_OBJECT_CONTROL",
            "selected_control_name": result.selected_control_name,
            "source_archive_bytes": len(source_archive),
            "g20_archive_bytes": len(g20_control),
            "v2_archive_bytes": result.selected_v2.archive_nbytes,
            "verdict_scope": "this exact lossless population-global transform/coder/container menu",
        }
        _write_once(output_dir / "blocker.json", _canonical_json(blocker))
        raise MaterializePopulationGlobalRecodeError(blocker["blocker"])

    archive_path = output_dir / "ep725_population_global_recode_v2.not_a_candidate.zip"
    _write_once(archive_path, result.selected_v2.archive_bytes)
    receipt = result.structural_receipt()
    receipt.update(
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "git_head": _git_head(),
            "artifact": {
                "path": str(archive_path.relative_to(ROOT)),
                "classification": "not_a_candidate",
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256(archive_path.read_bytes()),
            },
            "source_paths": {
                "archive": str(source_archive_path),
                "runtime": str(source_runtime_path),
                "g20_control": str(g20_control_path),
            },
            "runtime": {
                "path": str(source_runtime_path),
                "bytes": len(source_runtime),
                "sha256": _sha256(source_runtime),
                "v2_state_receiver": (
                    "tac.witness_dsl.ep725_population_global_recode_v2.parse_population_global_member"
                ),
                "full_n600_output_replay_owed": True,
            },
            "reproduction": {
                "argv": sys.argv,
                "python": sys.version,
                "implementation": {
                    "module": _path_receipt(ROOT / "src/tac/witness_dsl/ep725_population_global_recode_v2.py"),
                    "tool": _path_receipt(Path(__file__).resolve()),
                    "spec": _path_receipt(
                        ROOT
                        / (
                            ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
                            "SPEC_g25_population_global_same_solution_recode_20260726.md"
                        )
                    ),
                },
                "resume_from": None if args.resume_from is None else str(output_dir),
            },
            "cleanup_certificate": {
                "schema": "tac.disk_hygiene_certificate.v1",
                "bulk_scratch_created": False,
                "scratch_strategy": "memory_resident_complete_archive_points",
                "temporary_paths_remaining": [],
                "durable_artifacts": [
                    str(archive_path.relative_to(ROOT)),
                    str((output_dir / "receipt.json").relative_to(ROOT)),
                    str(checkpoint_dir.relative_to(ROOT)),
                ],
                "destructive_cleanup_performed": False,
            },
        }
    )
    receipt_path = output_dir / "receipt.json"
    _write_once(receipt_path, _canonical_json(receipt))
    _write_once(
        checkpoint_dir / "stage999_final.json",
        _canonical_json(
            {
                "schema": "tac.ep725_population_global_recode_final.v2",
                "archive": _path_receipt(archive_path),
                "receipt": _path_receipt(receipt_path),
                "points_measured": result.points_measured,
                "converged": result.converged,
            }
        ),
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--source-runtime", type=Path, default=DEFAULT_SOURCE_RUNTIME)
    parser.add_argument("--g20-control", type=Path, default=DEFAULT_G20_CONTROL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="resume the exact source-bound search from the last complete stage in this run dir",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        receipt = materialize(args)
    except (MaterializePopulationGlobalRecodeError, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
