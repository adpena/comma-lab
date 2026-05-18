#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""No-spend exact-eval handoff doctor for Z7 recurrent-vs-static packets.

This tool classifies whether the latest Z7 score-aware packet can enter paired
CPU/CUDA exact eval. It does not open lane claims, dispatch provider work, or
make score/promotion claims. Its job is to preserve the same-byte
recurrent-vs-static signal while failing closed on the current one-pair smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATS_JSON = (
    "experiments/results/z7_gru_score_aware_smoke_codex_20260518T1255Z/"
    "z7_gru_prebuild_full_main_export_stats.json"
)
DEFAULT_OUTPUT_DIR = ".omx/state/z7_exact_eval_handoff"
LANE_ID = "lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517"
SUBSTRATE_ID = "time_traveler_l5_z7_lstm_predictive_coding"
PAIR_GROUP_ID = "z7_temporal_coherence_vs_static_capacity_same_bytes"
REQUIRED_PAIR_COUNT = 600
HEX_DIGITS = set("0123456789abcdefABCDEF")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _short_sha(value: str, *, label: str) -> str:
    raw = str(value or "").strip()
    if len(raw) >= 12 and all(char in HEX_DIGITS for char in raw[:12]):
        return raw[:12].lower()
    return f"<{label}_archive_sha256_prefix>"


def _zip_member_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for info in [row for row in zf.infolist() if not row.is_dir()]:
                blob = zf.read(info.filename)
                rows.append(
                    {
                        "name": info.filename,
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                        "crc": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(blob).hexdigest(),
                        "compression_method": info.compress_type,
                    }
                )
    except (OSError, zipfile.BadZipFile) as exc:
        blockers.append(f"zip_unreadable:{type(exc).__name__}")
    return rows, blockers


def _archive_status(
    *,
    repo_root: Path,
    mode: str,
    path_value: str,
    expected_bytes: Any,
    expected_sha256: Any,
) -> dict[str, Any]:
    path = _resolve_repo_path(repo_root, path_value)
    blockers: list[str] = []
    exists = path.is_file()
    zip_rows: list[dict[str, Any]] = []
    actual_bytes: int | None = None
    actual_sha: str | None = None
    if not exists:
        blockers.append(f"z7_exact_handoff_{mode}_archive_zip_missing")
    else:
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        zip_rows, zip_blockers = _zip_member_rows(path)
        blockers.extend(f"z7_exact_handoff_{mode}_{b}" for b in zip_blockers)
        if len(zip_rows) != 1:
            blockers.append(f"z7_exact_handoff_{mode}_archive_zip_not_single_member")
        elif zip_rows[0].get("name") != "0.bin":
            blockers.append(f"z7_exact_handoff_{mode}_archive_zip_member_not_0_bin")
        if isinstance(expected_bytes, int) and actual_bytes != expected_bytes:
            blockers.append(f"z7_exact_handoff_{mode}_archive_bytes_mismatch")
        if expected_sha256 and str(expected_sha256).lower() != str(actual_sha).lower():
            blockers.append(f"z7_exact_handoff_{mode}_archive_sha256_mismatch")
    return {
        "mode": mode,
        "zip_path": path_value,
        "zip_exists": exists,
        "zip_bytes": actual_bytes,
        "zip_sha256": actual_sha or expected_sha256,
        "expected_zip_bytes": expected_bytes,
        "expected_zip_sha256": expected_sha256,
        "zip_member_rows": zip_rows,
        "blockers": blockers,
    }


def _runtime_custody(repo_root: Path, runtime_dir_value: str) -> dict[str, Any]:
    path = _resolve_repo_path(repo_root, runtime_dir_value)
    blockers: list[str] = []
    records: list[dict[str, Any]] = []
    if not path.is_dir():
        return {
            "path": runtime_dir_value,
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "aggregate_sha256": None,
            "inflate_sh_executable": False,
            "blockers": ["z7_exact_handoff_submission_runtime_dir_missing"],
        }
    inflate_sh = path / "inflate.sh"
    if not inflate_sh.is_file():
        blockers.append("z7_exact_handoff_submission_runtime_inflate_sh_missing")
    elif not (inflate_sh.stat().st_mode & stat.S_IXUSR):
        blockers.append("z7_exact_handoff_submission_runtime_inflate_sh_not_executable")
    for cur in sorted(p for p in path.rglob("*") if p.is_file()):
        if "__pycache__" in cur.parts:
            continue
        rel = cur.relative_to(path).as_posix()
        records.append(
            {
                "path": rel,
                "bytes": cur.stat().st_size,
                "sha256": _sha256(cur),
            }
        )
    agg = hashlib.sha256()
    total_bytes = 0
    for record in records:
        total_bytes += int(record["bytes"])
        agg.update(record["path"].encode("utf-8"))
        agg.update(b"\0")
        agg.update(str(record["sha256"]).encode("ascii"))
        agg.update(b"\0")
    return {
        "path": runtime_dir_value,
        "exists": True,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "aggregate_sha256": agg.hexdigest(),
        "inflate_sh_executable": inflate_sh.is_file()
        and bool(inflate_sh.stat().st_mode & stat.S_IXUSR),
        "records": records,
        "blockers": blockers,
    }


def _paired_dispatch_command(
    *,
    archive_zip: str,
    archive_sha: str,
    mode: str,
    submission_dir: str,
    execute: bool,
) -> str:
    short = _short_sha(archive_sha, label=mode)
    safe_mode = mode.replace("_", "-")
    lane_base = f"lane_z7_lstm_predictive_coding_20260518_{safe_mode}"
    cmd = [
        ".venv/bin/python",
        "tools/dispatch_modal_paired_auth_eval.py",
        "--archive",
        archive_zip,
        "--expected-archive-sha256",
        archive_sha,
        "--submission-dir",
        submission_dir,
        "--inflate-sh",
        "inflate.sh",
        "--label",
        f"z7_{safe_mode}",
        "--run-id",
        f"z7_{safe_mode}_{short}",
        "--pair-group-id",
        f"{PAIR_GROUP_ID}_{safe_mode}_{short}",
        "--lane-id-base",
        lane_base,
        "--output-root",
        "experiments/results",
        "--modal-bin",
        ".venv/bin/modal",
        "--gpu",
        "T4",
        "--claim-agent",
        "codex:z7_exact_eval_handoff",
        "--claim-notes",
        (
            "Z7 recurrent-vs-static paired exact-eval handoff; "
            f"mode={mode}; archive_sha={archive_sha}; "
            f"pair_group_id={PAIR_GROUP_ID}; score_claim=false"
        ),
        "--expected-runtime-tree-sha256",
        "auto",
        "--skip-axis-if-promotable-anchor-exists",
    ]
    if execute:
        cmd.append("--execute")
    return shlex.join(cmd)


def _command_templates(*, execute: bool) -> dict[str, str]:
    return {
        "recurrent_paired_contest_cpu_cuda": _paired_dispatch_command(
            archive_zip="<ratified_z7_run_dir>/archive.zip",
            archive_sha="<recurrent_archive_zip_sha256>",
            mode="recurrent",
            submission_dir="<ratified_z7_run_dir>/submission_runtime",
            execute=execute,
        ),
        "static_control_paired_contest_cpu_cuda": _paired_dispatch_command(
            archive_zip="<ratified_z7_run_dir>/static_capacity_control/archive.zip",
            archive_sha="<static_control_archive_zip_sha256>",
            mode="static_control",
            submission_dir="<ratified_z7_run_dir>/submission_runtime",
            execute=execute,
        ),
    }


def build_packet(
    *,
    repo_root: Path = REPO_ROOT,
    stats_json: Path | None = None,
    required_pair_count: int = REQUIRED_PAIR_COUNT,
) -> dict[str, Any]:
    stats_path = stats_json or repo_root / DEFAULT_STATS_JSON
    if not stats_path.is_absolute():
        stats_path = repo_root / stats_path
    blockers: list[str] = []
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        stats = {}
        blockers.append(f"z7_exact_handoff_stats_json_unreadable:{type(exc).__name__}")
    if not isinstance(stats, Mapping):
        stats = {}
        blockers.append("z7_exact_handoff_stats_json_not_object")

    static = stats.get("static_capacity_control")
    if not isinstance(static, Mapping):
        static = {}
        blockers.append("z7_exact_handoff_static_capacity_control_missing")

    config = stats.get("config")
    if not isinstance(config, Mapping):
        config = {}
        blockers.append("z7_exact_handoff_config_missing")

    for field in ("score_claim", "promotion_eligible", "ready_for_paid_dispatch"):
        if stats.get(field) is not False:
            blockers.append(f"z7_exact_handoff_stats_{field}_not_false")
        if static and static.get(field) is not False:
            blockers.append(f"z7_exact_handoff_static_control_{field}_not_false")

    if stats.get("loss_mode") != "score_aware":
        blockers.append("z7_exact_handoff_loss_mode_not_score_aware")
    if stats.get("score_aware_scorer_loss_used") is not True:
        blockers.append("z7_exact_handoff_score_aware_scorer_loss_not_used")

    recurrent = _archive_status(
        repo_root=repo_root,
        mode="recurrent",
        path_value=str(stats.get("archive_zip_path") or ""),
        expected_bytes=stats.get("archive_zip_bytes"),
        expected_sha256=stats.get("archive_zip_sha256"),
    )
    static_row = _archive_status(
        repo_root=repo_root,
        mode="static_control",
        path_value=str(static.get("archive_zip_path") or ""),
        expected_bytes=static.get("archive_zip_bytes"),
        expected_sha256=static.get("archive_zip_sha256"),
    )
    for row in (recurrent, static_row):
        blockers.extend(row.get("blockers") or [])

    if recurrent.get("zip_bytes") != static_row.get("zip_bytes"):
        blockers.append("z7_exact_handoff_recurrent_static_archive_byte_count_mismatch")
    if static.get("same_archive_zip_bytes_as_recurrent") is not True:
        blockers.append("z7_exact_handoff_static_control_same_archive_bytes_flag_not_true")
    if static.get("runtime_output_changed_vs_recurrent") is not True:
        blockers.append("z7_exact_handoff_static_control_runtime_output_not_changed")
    byte_diff = static.get("runtime_output_byte_differences_vs_recurrent")
    if not isinstance(byte_diff, int) or byte_diff <= 0:
        blockers.append("z7_exact_handoff_static_control_byte_differences_not_positive")

    num_pairs = config.get("num_pairs")
    if not isinstance(num_pairs, int):
        blockers.append("z7_exact_handoff_num_pairs_missing_or_not_int")
    elif num_pairs != int(required_pair_count):
        blockers.append(f"z7_exact_handoff_current_packet_not_{required_pair_count}_pairs")

    runtime = _runtime_custody(
        repo_root,
        str(stats.get("submission_runtime_dir") or ""),
    )
    blockers.extend(runtime.get("blockers") or [])

    submission_dir = str(stats.get("submission_runtime_dir") or "")
    plan_command_allowed_blockers = {
        f"z7_exact_handoff_current_packet_not_{required_pair_count}_pairs",
    }
    valid_archive_pair = all(
        blocker in plan_command_allowed_blockers for blocker in blockers
    )
    plan_commands: dict[str, str] = {}
    execute_commands: dict[str, str] = {}
    if valid_archive_pair and submission_dir:
        plan_commands = {
            "recurrent_paired_contest_cpu_cuda": _paired_dispatch_command(
                archive_zip=str(stats.get("archive_zip_path") or ""),
                archive_sha=str(stats.get("archive_zip_sha256") or ""),
                mode="recurrent",
                submission_dir=submission_dir,
                execute=False,
            ),
            "static_control_paired_contest_cpu_cuda": _paired_dispatch_command(
                archive_zip=str(static.get("archive_zip_path") or ""),
                archive_sha=str(static.get("archive_zip_sha256") or ""),
                mode="static_control",
                submission_dir=submission_dir,
                execute=False,
            ),
        }
    ready = not blockers
    if ready:
        execute_commands = {
            key: command + " --execute" for key, command in plan_commands.items()
        }

    return {
        "schema": "z7_exact_eval_handoff_v1",
        "generated_utc": _utc_now(),
        "lane_id": LANE_ID,
        "substrate_id": SUBSTRATE_ID,
        "stats_json": _safe_rel(repo_root, stats_path),
        "required_pair_count": int(required_pair_count),
        "current_pair_count": num_pairs if isinstance(num_pairs, int) else None,
        "ready_for_exact_eval_handoff": ready,
        "provider_dispatch_attempted": False,
        "lane_claim_opened": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "z7_exact_eval_handoff_no_spend",
        "pair_group_id": PAIR_GROUP_ID,
        "axis_plan": [
            {"axis": "contest-CUDA", "mode": "recurrent"},
            {"axis": "contest-CPU", "mode": "recurrent"},
            {"axis": "contest-CUDA", "mode": "static_control"},
            {"axis": "contest-CPU", "mode": "static_control"},
        ],
        "source_archive_rows": [recurrent, static_row],
        "same_archive_zip_bytes": recurrent.get("zip_bytes") == static_row.get("zip_bytes"),
        "runtime_output_changed_vs_recurrent": static.get(
            "runtime_output_changed_vs_recurrent"
        ),
        "runtime_output_byte_differences_vs_recurrent": byte_diff,
        "runtime_custody": runtime,
        "modal_plan_commands_for_current_packet": plan_commands,
        "modal_execute_commands_after_ratified_full_packet": execute_commands,
        "modal_command_templates_after_ratified_full_packet": _command_templates(
            execute=True
        ),
        "result_review_blockers": [] if ready else blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-json", type=Path, default=Path(DEFAULT_STATS_JSON))
    parser.add_argument("--required-pair-count", type=int, default=REQUIRED_PAIR_COUNT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    payload = build_packet(
        repo_root=REPO_ROOT,
        stats_json=args.stats_json,
        required_pair_count=args.required_pair_count,
    )
    if args.write_artifact:
        out_dir = args.output_dir
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"z7_exact_eval_handoff_{_utc_now_compact()}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["artifact_path"] = _safe_rel(REPO_ROOT, path)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json:
        print(text)
    else:
        status = "READY" if payload["ready_for_exact_eval_handoff"] else "BLOCKED"
        print(f"Z7 exact-eval handoff: {status}")
        for blocker in payload.get("result_review_blockers", []):
            print(f"- {blocker}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
