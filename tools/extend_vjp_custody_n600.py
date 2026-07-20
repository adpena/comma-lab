#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumably extend immutable VJP custody to the real n600 surface.

The underlying producer intentionally accepts at most twelve pairs.  This
driver schedules deterministic chunks, validates every existing manifest and
sidecar before skipping it, resumes an interrupted prefix, and writes an
atomic campaign receipt after every chunk.  Pair-specific refusals are
preserved as scoped blockers while independent later chunks continue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for import_path in (REPO, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tac.optimization.vjp_custody import atomic_json, sha256_file  # noqa: E402
from tools.produce_vjp_custody import (  # noqa: E402
    BYTES_PER_PAIR_PREFLIGHT,
    DEFAULT_OUTPUT,
    FIXED_FREE_SPACE_RESERVE,
    MAX_PAIRS,
    REFUSAL_VERDICT_SCOPE,
    _load_composition_source,
    _validate_refusal_row,
)

SCHEMA = "vjp_custody_n600_extension.v1"
N_PAIRS = 600
DEFAULT_CAMPAIGN_ROOT = DEFAULT_OUTPUT / "extension_n600_20260720"
DEFAULT_SOURCES = (
    DEFAULT_OUTPUT / "chunk_000_011/manifest.json",
    DEFAULT_OUTPUT / "chunk_012_023/manifest.json",
    DEFAULT_OUTPUT / "replacement_pair_0024/manifest.json",
    DEFAULT_OUTPUT / "replacement_pair_0025/manifest.json",
    DEFAULT_OUTPUT / "extension_20260720_timing_026_028/manifest.json",
)
KNOWN_ISOLATED_PAIRS = (11, 245, 277, 482, 514, 532, 574)
KNOWN_NATIVE_REFRESH_PAIRS = KNOWN_ISOLATED_PAIRS
RunCommand = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class ExtensionError(RuntimeError):
    """Fail-closed campaign configuration or custody error."""


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _chunks(missing: Sequence[int], isolated: Sequence[int]) -> list[list[int]]:
    unique_missing = sorted({int(value) for value in missing})
    if len(unique_missing) != len(missing) or any(value < 0 or value >= N_PAIRS for value in unique_missing):
        raise ExtensionError("missing pair ids must be unique members of [0,600)")
    isolated_set = {int(value) for value in isolated} & set(unique_missing)
    ordinary = [value for value in unique_missing if value not in isolated_set]
    result = [[value] for value in sorted(isolated_set)]
    result.extend(ordinary[index : index + MAX_PAIRS] for index in range(0, len(ordinary), MAX_PAIRS))
    return result


def _recovery_work(
    pair_ids: Sequence[int], completed: set[int], refused: set[int]
) -> list[tuple[list[int], str]]:
    """Split one refused cached chunk into native retries plus untouched tails."""

    declared = set(pair_ids)
    if not refused or not refused <= declared or not completed <= declared:
        raise ExtensionError("refused-chunk recovery sets are inconsistent")
    untouched = sorted(declared - completed - refused)
    work = [([pair_id], "fresh-native") for pair_id in sorted(refused)]
    work.extend((chunk, "cached-verified") for chunk in _chunks(untouched, []))
    return work


def _manifest_rows(path: Path) -> tuple[dict[str, Any], set[int], set[int]]:
    _, manifest, _ = _load_composition_source(path)
    completed = {int(row["pair_id"]) for row in manifest["sidecars"]}
    refused: set[int] = set()
    for row in manifest.get("refusals", []):
        refused.add(_validate_refusal_row(row, manifest))
    return manifest, completed, refused


def _source_coverage(paths: Sequence[Path]) -> tuple[set[int], list[dict[str, Any]]]:
    coverage: set[int] = set()
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        manifest, completed, refused = _manifest_rows(path)
        duplicate = coverage & completed
        if duplicate:
            raise ExtensionError(f"source manifests duplicate completed pairs: {sorted(duplicate)}")
        coverage.update(completed)
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "completed_pair_ids": sorted(completed),
                "refused_pair_ids": sorted(refused),
                "manifest_content_sha256": manifest.get("manifest_content_sha256"),
            }
        )
    return coverage, records


def _prior_manifest_paths(state_path: Path) -> list[Path]:
    if not state_path.is_file():
        return []
    try:
        campaign = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtensionError(f"cannot resume prior campaign receipt: {state_path}") from exc
    if campaign.get("schema") != SCHEMA:
        raise ExtensionError(f"prior campaign schema mismatch: {state_path}")
    allowed = DEFAULT_OUTPUT.resolve()
    paths: list[Path] = []
    for row in campaign.get("source_manifests", []):
        if isinstance(row, dict):
            paths.append(Path(str(row.get("path", ""))))
    for row in campaign.get("chunks", []):
        if isinstance(row, dict):
            paths.append(Path(str(row.get("path", ""))) / "manifest.json")
    resolved: list[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve(strict=True)
        if not (candidate.parent == allowed or candidate.parent.is_relative_to(allowed)):
            raise ExtensionError(f"prior campaign manifest escaped output root: {candidate}")
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def _chunk_dir(root: Path, pair_ids: Sequence[int], *, winner_policy: str = "cached-verified") -> Path:
    identity = f"{winner_policy}:" + ",".join(str(value) for value in pair_ids)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:8]
    suffix = "_fresh_native" if winner_policy == "fresh-native" else ""
    return root / f"chunk_{pair_ids[0]:04d}_{pair_ids[-1]:04d}_{digest}{suffix}"


def _classify_chunk(path: Path, expected_ids: Sequence[int]) -> tuple[str, set[int], set[int]]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return "absent", set(), set()
    manifest, completed, refused = _manifest_rows(manifest_path)
    if manifest.get("pair_ids") != list(expected_ids):
        raise ExtensionError(f"chunk pair-id drift at {manifest_path}")
    if refused:
        return "refused", completed, refused
    if completed == set(expected_ids) and manifest.get("completed_at_utc"):
        return "complete", completed, refused
    return "partial", completed, refused


def _preflight(root: Path, missing_count: int) -> dict[str, int]:
    resolved = root.expanduser().resolve(strict=False)
    allowed = DEFAULT_OUTPUT.resolve()
    if not (resolved == allowed or resolved.is_relative_to(allowed)):
        raise ExtensionError(f"campaign output must remain below {allowed}")
    resolved.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(resolved)
    required = FIXED_FREE_SPACE_RESERVE + missing_count * BYTES_PER_PAIR_PREFLIGHT
    if usage.free < required:
        raise ExtensionError(f"SSD preflight refused campaign: free={usage.free}, required={required}")
    return {"free_bytes": usage.free, "required_bytes": required}


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=REPO, text=True, check=False)


def execute(args: argparse.Namespace, *, runner: RunCommand = _default_runner) -> int:
    root = args.output_root.expanduser().resolve(strict=False)
    state_path = root / "campaign_receipt.json"
    source_paths = [Path(value) for value in args.source_manifest]
    if args.resume_campaign:
        source_paths.extend(_prior_manifest_paths(state_path))
    source_paths = list(dict.fromkeys(path.expanduser().resolve(strict=True) for path in source_paths))
    source_coverage, source_records = _source_coverage(source_paths)
    missing = sorted(set(range(N_PAIRS)) - source_coverage)
    native_refresh = {int(value) for value in args.refresh_native_pair} & set(missing)
    schedule = _chunks(missing, [*args.isolate_pair, *native_refresh])
    storage = _preflight(root, len(missing))
    campaign: dict[str, Any] = {
        "schema": SCHEMA,
        "updated_at_utc": _utc_now(),
        "git_head": _git_head(),
        "authority": {
            "axis": "[macOS-arm64 CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        },
        "resumability": {
            "unit": "immutable per-pair NPZ plus atomic prefix manifest",
            "restart": "validate and skip complete/refused chunks; --resume partial chunks",
            "checkpoint_interval": "every pair in producer; campaign receipt after every chunk",
        },
        "source_manifests": source_records,
        "source_completed_pair_ids": sorted(source_coverage),
        "initial_missing_pair_ids": missing,
        "native_refresh_pair_ids": sorted(native_refresh),
        "schedule": schedule,
        "storage_preflight": storage,
        "chunks": [],
    }
    if args.plan_only:
        campaign["status"] = "PLAN_ONLY"
        atomic_json(state_path, campaign)
        print(json.dumps({"receipt": str(state_path), "status": campaign["status"], "missing": len(missing)}))
        return 0

    executed = 0
    work_queue = [
        (
            pair_ids,
            "fresh-native"
            if len(pair_ids) == 1 and pair_ids[0] in native_refresh
            else "cached-verified",
        )
        for pair_ids in schedule
    ]
    queued = {(tuple(pair_ids), winner_policy) for pair_ids, winner_policy in work_queue}
    cursor = 0
    while cursor < len(work_queue):
        pair_ids, winner_policy = work_queue[cursor]
        cursor += 1
        chunk_dir = _chunk_dir(root, pair_ids, winner_policy=winner_policy)
        before, completed, refused = _classify_chunk(chunk_dir, pair_ids)
        argv = [
            str(args.python),
            str(REPO / "tools/produce_vjp_custody.py"),
            "--pair-indices",
            *(str(value) for value in pair_ids),
            "--output-dir",
            str(chunk_dir),
            "--cpu-threads",
            str(args.cpu_threads),
            "--winner-policy",
            winner_policy,
        ]
        invoked = False
        returncode: int | None = None
        if before not in {"complete", "refused"}:
            if args.stop_after_chunks is not None and executed >= args.stop_after_chunks:
                break
            if before == "partial":
                argv.append("--resume")
            invoked = True
            returncode = runner(argv).returncode
            executed += 1
        after, completed, refused = _classify_chunk(chunk_dir, pair_ids)
        campaign["chunks"].append(
            {
                "pair_ids": pair_ids,
                "path": str(chunk_dir),
                "status_before": before,
                "status_after": after,
                "invoked": invoked,
                "returncode": returncode,
                "winner_policy": winner_policy,
                "completed_pair_ids": sorted(completed),
                "refused_pair_ids": sorted(refused),
                "verdict_scope": REFUSAL_VERDICT_SCOPE if refused else None,
                "manifest_sha256": sha256_file(chunk_dir / "manifest.json")
                if (chunk_dir / "manifest.json").is_file()
                else None,
            }
        )
        if refused and winner_policy == "cached-verified":
            recovery = _recovery_work(pair_ids, completed, refused)
            native_refresh.update(refused)
            campaign["native_refresh_pair_ids"] = sorted(native_refresh)
            for item in recovery:
                identity = (tuple(item[0]), item[1])
                if identity not in queued:
                    work_queue.append(item)
                    queued.add(identity)
            campaign["dynamic_recovery_work"] = [
                {"pair_ids": ids, "winner_policy": policy}
                for ids, policy in work_queue[len(schedule) :]
            ]
        campaign["updated_at_utc"] = _utc_now()
        atomic_json(state_path, campaign)

    extension_completed = {
        int(pair_id)
        for row in campaign["chunks"]
        for pair_id in row["completed_pair_ids"]
    }
    extension_refused = {
        int(pair_id)
        for row in campaign["chunks"]
        for pair_id in row["refused_pair_ids"]
    }
    final_coverage = source_coverage | extension_completed
    still_missing = sorted(set(range(N_PAIRS)) - final_coverage)
    source_refused = {
        int(pair_id)
        for record in source_records
        for pair_id in record["refused_pair_ids"]
    }
    effective_refused = sorted((source_refused | extension_refused) - final_coverage)
    campaign.update(
        {
            "updated_at_utc": _utc_now(),
            "extension_completed_pair_ids": sorted(extension_completed),
            "refused_pair_ids": effective_refused,
            "final_completed_pair_ids": sorted(final_coverage),
            "final_completed_count": len(final_coverage),
            "still_missing_pair_ids": still_missing,
            "status": (
                "COMPLETE_N600"
                if not still_missing and not effective_refused
                else "IN_PROGRESS_OR_SCOPED_BLOCKED"
            ),
        }
    )
    atomic_json(state_path, campaign)
    print(
        json.dumps(
            {
                "receipt": str(state_path),
                "status": campaign["status"],
                "completed": len(final_coverage),
                "missing": len(still_missing),
                "refused": effective_refused,
            },
            sort_keys=True,
        )
    )
    return 0 if not still_missing else 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument(
        "--source-manifest",
        action="append",
        default=[str(path) for path in DEFAULT_SOURCES],
        help="Validated existing manifest; repeat to add sources.",
    )
    parser.add_argument("--isolate-pair", action="append", type=int, default=list(KNOWN_ISOLATED_PAIRS))
    parser.add_argument(
        "--refresh-native-pair",
        action="append",
        type=int,
        default=list(KNOWN_NATIVE_REFRESH_PAIRS),
    )
    parser.add_argument(
        "--no-resume-campaign",
        dest="resume_campaign",
        action="store_false",
        help="Ignore prior campaign manifests and rebuild the schedule from explicit sources only.",
    )
    parser.set_defaults(resume_campaign=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--stop-after-chunks", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.cpu_threads <= 0 or (args.stop_after_chunks is not None and args.stop_after_chunks < 0):
        raise SystemExit("cpu threads must be positive and stop-after-chunks nonnegative")
    raise SystemExit(execute(args))


if __name__ == "__main__":
    main()
