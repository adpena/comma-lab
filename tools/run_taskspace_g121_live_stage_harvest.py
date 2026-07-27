#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Watch one governed G111 run and harvest preserved stages through G121.

The monitor invokes the incremental, non-exhaustive G121 entrypoint whenever a
new preserved stage alias or complete periodic native-v3 triplet becomes
visible. It invokes the exhaustive entrypoint only after
``levelset_train_result.json`` physically binds the current final fresh-lineage
tip. A single output directory is permanently bound to one producer path; each
cold/resume launch epoch is appended only after reopening its externally
supplied manifest SHA. This prevents reuse of another run's payloads while
preserving one exact checkpoint-keyed ledger across resumes.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.witness_control import (  # noqa: E402
    g120_governed_clean_dry_run_gate_v1 as g120_gate,
)
from tac.witness_control import (  # noqa: E402
    taskspace_g121_resumable_stage_harvest_v1 as g121,
)

MONITOR_BINDING_SCHEMA = "tac.g121_live_stage_harvest_monitor_binding.v1"
MONITOR_BINDING_BASENAME = "g121_live_monitor_binding.json"
MONITOR_STATUS_BASENAME = "g121_live_monitor_status.json"
MONITOR_LAUNCH_EPOCHS_BASENAME = "g121_live_monitor_launch_epochs.jsonl"
MONITOR_PROCESS_LOCK_BASENAME = "g121_live_monitor.process.lock"


def _absolute(value: str, *, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path


def _jsonable_dataclass(value: object) -> dict[str, object]:
    encoded = asdict(value)
    for key, item in tuple(encoded.items()):
        if isinstance(item, Path):
            encoded[key] = str(item)
    return encoded


def _preserved_signature(producer: Path) -> tuple[tuple[str, int, int], ...]:
    rows: list[tuple[str, int, int]] = []
    for path in sorted(producer.glob("levelset_ckpt_stage*_ep*.npz")):
        try:
            stat_result = path.stat(follow_symlinks=False)
        except OSError:
            continue
        rows.append((path.name, stat_result.st_size, stat_result.st_mtime_ns))
        resume = producer / path.name.replace(
            "levelset_ckpt_", "levelset_resume_", 1
        )
        try:
            resume_stat = resume.stat(follow_symlinks=False)
        except OSError:
            continue
        rows.append(
            (resume.name, resume_stat.st_size, resume_stat.st_mtime_ns)
        )
        native = producer / path.name.replace(
            "levelset_ckpt_", "levelset_g111_native_", 1
        )
        try:
            native_stat = native.stat(follow_symlinks=False)
        except OSError:
            continue
        rows.append(
            (native.name, native_stat.st_size, native_stat.st_mtime_ns)
        )
    for _tag, _epoch, deploy, resume, native in (
        g121._complete_periodic_alias_triplets(producer)
    ):
        triplet_rows: list[tuple[str, int, int]] = []
        try:
            for path in (deploy, resume, native):
                path_stat = path.stat(follow_symlinks=False)
                triplet_rows.append(
                    (path.name, path_stat.st_size, path_stat.st_mtime_ns)
                )
        except OSError:
            continue
        rows.extend(triplet_rows)
    return tuple(rows)


def _write_monitor_status(
    output: Path,
    *,
    status: str,
    detail: str,
    preserved_signature: tuple[tuple[str, int, int], ...],
) -> None:
    payload = g121._canonical_json(
        {
            "schema": "tac.g121_live_stage_harvest_monitor_status.v1",
            "status": status,
            "detail": detail,
            "preserved_file_count": len(preserved_signature),
            "research_only": True,
            "score_claim": False,
            "evaluation_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        }
    )
    g121._atomic_replace(output / MONITOR_STATUS_BASENAME, payload)


def _bind_monitor(
    *,
    producer: Path,
    output: Path,
    progress: Path,
) -> None:
    binding_path = output / MONITOR_BINDING_BASENAME
    if not binding_path.exists() and any(output.iterdir()):
        raise g121.G121StageHarvestError(
            "unbound G121 monitor output directory is not empty; refusing "
            "possible old-producer payload reuse"
        )
    binding = {
        "schema": MONITOR_BINDING_SCHEMA,
        "producer_run_dir": str(producer),
        "output_dir": str(output),
        "progress_dir": str(progress),
        "incremental_entrypoint": "harvest_g111_available_stages_v1",
        "terminal_entrypoint": "harvest_g111_stages_v1",
        "old_producer_payload_reuse": False,
        "terminal_only_exhaustive_publication": True,
        "research_only": True,
        "score_claim": False,
        "evaluation_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    g121._atomic_write_once(
        binding_path,
        g121._canonical_json(binding),
    )


def _authorize_launch_epoch(
    *,
    producer: Path,
    expected_launch_manifest_sha256: str,
    g120_dry_run_receipt: dict[str, object],
    output: Path,
) -> dict[str, object]:
    _payload, launch_binding = g121._stable_regular_file(
        producer / "launch_manifest.json",
        name="G121 monitor governed launch manifest",
        expected_sha256=expected_launch_manifest_sha256,
    )
    row = {
        "schema": "tac.g121_live_stage_harvest_launch_epoch.v1",
        "producer_run_dir": str(producer),
        "launch_manifest": launch_binding,
        "g120_governed_clean_dry_run_receipt": g120_dry_run_receipt,
        "externally_sha_bound": True,
        "research_only": True,
        "score_claim": False,
        "evaluation_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    payload = g121._canonical_json(row)
    path = output / MONITOR_LAUNCH_EPOCHS_BASENAME
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = path.read_bytes().splitlines(keepends=True) if path.exists() else []
        if payload in existing:
            return launch_binding
        if existing and any(
            (output / basename).exists()
            for basename in (
                g121.COMPLETION_RECEIPT_BASENAME,
                g121.RETAINED_PREPOSE_BASENAME,
                g121.SCHEDULING_HINT_BASENAME,
            )
        ):
            raise g121.G121StageHarvestError(
                "a new launch epoch cannot reuse an output directory that already "
                "contains terminal G121 reductions; choose fresh output/progress paths"
            )
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return launch_binding


def _latest_authorized_launch_sha(
    *,
    producer: Path,
    output: Path,
) -> str:
    path = output / MONITOR_LAUNCH_EPOCHS_BASENAME
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise g121.G121StageHarvestError(
            "G121 monitor launch-epoch ledger is absent"
        ) from exc
    if not lines:
        raise g121.G121StageHarvestError(
            "G121 monitor launch-epoch ledger is empty"
        )
    try:
        row = json.loads(lines[-1].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise g121.G121StageHarvestError(
            "latest G121 monitor launch epoch is corrupt"
        ) from exc
    if (
        type(row) is not dict
        or row.get("schema")
        != "tac.g121_live_stage_harvest_launch_epoch.v1"
        or row.get("producer_run_dir") != str(producer)
        or row.get("externally_sha_bound") is not True
        or type(row.get("launch_manifest")) is not dict
        or type(
            row.get("g120_governed_clean_dry_run_receipt")
        )
        is not dict
    ):
        raise g121.G121StageHarvestError(
            "latest G121 monitor launch epoch has wrong custody"
        )
    launch_binding = row["launch_manifest"]
    launch_sha = g121._require_sha256(
        launch_binding.get("sha256"),
        name="authorized launch-manifest SHA-256",
    )
    _payload, reopened = g121._stable_regular_file(
        producer / "launch_manifest.json",
        name="currently authorized launch manifest",
        expected_sha256=launch_sha,
    )
    if reopened != launch_binding:
        raise g121.G121StageHarvestError(
            "current launch manifest differs from the latest externally authorized epoch"
        )
    dry_run_binding = row[
        "g120_governed_clean_dry_run_receipt"
    ]
    _payload, reopened_dry_run = g121._stable_regular_file(
        Path(str(dry_run_binding.get("path"))),
        name="authorized G120 governed clean dry-run receipt",
        expected_sha256=str(dry_run_binding.get("sha256")),
    )
    if reopened_dry_run != dry_run_binding:
        raise g121.G121StageHarvestError(
            "authorized G120 governed clean dry-run receipt changed"
        )
    return launch_sha


def run_monitor(
    *,
    producer_run_dir: Path,
    expected_launch_manifest_sha256: str,
    g120_dry_run_receipt: Path,
    expected_g120_dry_run_receipt_sha256: str,
    output_dir: Path,
    progress_dir: Path,
    poll_seconds: float,
    once: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    producer = g121._durable_directory(
        producer_run_dir,
        name="producer_run_dir",
    )
    output = g121._durable_directory(output_dir, name="output_dir")
    progress = g121._durable_directory(progress_dir, name="progress_dir")
    if len({producer, output, progress}) != 3:
        raise g121.G121StageHarvestError(
            "producer/output/progress directories must be distinct"
        )
    if not isinstance(poll_seconds, (int, float)) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    launch_sha = g121._require_sha256(
        expected_launch_manifest_sha256,
        name="expected launch manifest SHA-256",
    )
    dry_run_sha = g121._require_sha256(
        expected_g120_dry_run_receipt_sha256,
        name="expected G120 dry-run receipt SHA-256",
    )
    measurement_cache = g121._durable_directory(
        producer / "g121_measurement_cache",
        name="G121 measurement cache",
    )
    try:
        g120_gate.open_g120_governed_clean_dry_run_v1(
            g120_dry_run_receipt,
            expected_sha256=dry_run_sha,
            repo_root=REPO.resolve(),
            producer_run_dir=producer,
            expected_launch_manifest_sha256=launch_sha,
            monitor_output_dir=output,
            monitor_progress_dir=progress,
            measurement_cache_dir=measurement_cache,
        )
    except g120_gate.G120GovernedDryRunError as exc:
        raise g121.G121StageHarvestError(
            f"G120 governed clean dry-run gate refused monitor launch: {exc}"
        ) from exc
    _dry_run_payload, dry_run_binding = g121._stable_regular_file(
        g120_dry_run_receipt,
        name="G120 governed clean dry-run receipt",
        expected_sha256=dry_run_sha,
    )
    _bind_monitor(
        producer=producer,
        output=output,
        progress=progress,
    )
    _authorize_launch_epoch(
        producer=producer,
        expected_launch_manifest_sha256=launch_sha,
        g120_dry_run_receipt=dry_run_binding,
        output=output,
    )
    process_lock = (output / MONITOR_PROCESS_LOCK_BASENAME).open("a+b")
    try:
        try:
            fcntl.flock(
                process_lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            if once:
                raise g121.G121StageHarvestError(
                    "another G121 live monitor already owns this output"
                ) from None
            print(
                json.dumps(
                    {
                        "event": "G121_LAUNCH_EPOCH_REGISTERED_WITH_LIVE_MONITOR",
                        "launch_manifest_sha256": launch_sha,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        last_completed_signature: tuple[tuple[str, int, int], ...] | None = None
        while True:
            signature = _preserved_signature(producer)
            try:
                active_launch_sha = _latest_authorized_launch_sha(
                    producer=producer,
                    output=output,
                )
                if signature and signature != last_completed_signature:
                    incremental = g121.harvest_g111_available_stages_v1(
                        producer_run_dir=producer,
                        expected_launch_manifest_sha256=active_launch_sha,
                        output_dir=output,
                        progress_dir=progress,
                    )
                    last_completed_signature = signature
                    print(
                        json.dumps(
                            {
                                "event": "G121_INCREMENTAL_STAGE_HARVEST",
                                **_jsonable_dataclass(incremental),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if (producer / "levelset_train_result.json").is_file():
                    final = g121.harvest_g111_stages_v1(
                        producer_run_dir=producer,
                        expected_launch_manifest_sha256=active_launch_sha,
                        output_dir=output,
                        progress_dir=progress,
                    )
                    _write_monitor_status(
                        output,
                        status="EXHAUSTIVE_STAGE_HARVEST_COMPLETE",
                        detail=str(final.completion_receipt_path),
                        preserved_signature=signature,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "G121_EXHAUSTIVE_STAGE_HARVEST_COMPLETE",
                                **_jsonable_dataclass(final),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    return 0
                if once:
                    if not signature:
                        raise g121.G121StageHarvestError(
                            "no preserved G111 stage is currently eligible"
                        )
                    return 0
                _write_monitor_status(
                    output,
                    status="WAITING_FOR_NEXT_PRESERVED_STAGE",
                    detail="incremental ledger durable; exhaustive outputs withheld",
                    preserved_signature=signature,
                )
            except g121.G121StageHarvestError as exc:
                _write_monitor_status(
                    output,
                    status="RETRYABLE_FAIL_CLOSED",
                    detail=f"{type(exc).__name__}: {exc}",
                    preserved_signature=signature,
                )
                if once:
                    raise
                print(
                    json.dumps(
                        {
                            "event": "G121_RETRYABLE_FAIL_CLOSED",
                            "detail": str(exc),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            sleep(float(poll_seconds))
    finally:
        process_lock.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-run-dir", required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--g120-dry-run-receipt", required=True)
    parser.add_argument(
        "--expected-g120-dry-run-receipt-sha256",
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="harvest the current preserved-stage snapshot and exit without waiting",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_monitor(
        producer_run_dir=_absolute(
            args.producer_run_dir,
            name="producer_run_dir",
        ),
        expected_launch_manifest_sha256=args.expected_launch_manifest_sha256,
        g120_dry_run_receipt=_absolute(
            args.g120_dry_run_receipt,
            name="g120_dry_run_receipt",
        ),
        expected_g120_dry_run_receipt_sha256=(
            args.expected_g120_dry_run_receipt_sha256
        ),
        output_dir=_absolute(args.output_dir, name="output_dir"),
        progress_dir=_absolute(args.progress_dir, name="progress_dir"),
        poll_seconds=args.poll_seconds,
        once=args.once,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
