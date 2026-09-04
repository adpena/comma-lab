#!/usr/bin/env python3
"""Measured-peak ledger: what a cell family ACTUALLY costs, never what someone typed (ddm_gov2).

THE DEFECT THIS CURES (MEASURED 2026-09-04).  Every governed cell on this box was launched with a
HAND-TYPED memory declaration.  ng2 and ng3 ran under ``--measured-peak-rss-gib 2.3959503173828125``
-- a number carried forward from an unrelated run -- and ng4 inherited it "for comparability".  Two
of those cells then ran concurrently and the machine went to a VM-compressor space shortage at
17:11Z; jetsam killed background daemons.  MEASURED from ``.omx/state/memory_blackbox.jsonl``: the
compressor peaked at **76.978 GiB** with **72.0 GiB** of swap on a 128 GiB box, while the admission
gate had been told each cell would cost 2.396 GiB.

TWO INDEPENDENT NUMBERS, because RSS alone is a lie on Apple Silicon:

* ``peak_rss_gib`` -- what ``safe_run`` actually watched (``peak_rss_observed``/``peak_rss_mib`` are
  already in every status receipt, including those written by SEALED-SOURCE launcher copies).  For
  a Metal cell this is a FLOOR, not the cost: MEASURED on live ng4 the peak RSS was 1.746 GiB while
  its declared footprint was 45.0 GiB, because ``ps rss`` cannot see the Metal allocator.
* ``system_availability_delta_gib`` -- system availability before the launch minus the minimum
  observed while it ran.  This is the number that describes a Metal cell, and it comes from the
  memory blackbox this repo has been sampling all along rather than from a new instrument.

The ledger is APPEND-ONLY under an exclusive lock, and every row is labelled with the grade of its
attribution.  It is read by ``tools/cell_queue_driver.py fire``, which REFUSES a hand-typed peak for
a family that already has a measured row.

    measured_peaks harvest --root /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep
    measured_peaks record  --status-receipt <dir>/resource_safe_run_status.json
    measured_peaks lookup  --family ddm_qbr1_born_fairform_burn_prep --json

Non-promotable: no row here is a score claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]

MEASURED_PEAK_SCHEMA = "measured_peak.v1"
MEASURED_PEAKS_LEDGER = _REPO / ".omx" / "state" / "measured_peaks.jsonl"
MEMORY_BLACKBOX = _REPO / ".omx" / "state" / "memory_blackbox.jsonl"

#: Attribution grades, worst to best.  A consumer that cannot tell these apart will read a
#: confounded number as a clean one, which is how the 2.396 GiB fiction survived three launches.
GRADE_NO_PRE_LAUNCH = "UNAVAILABLE_NO_PRE_LAUNCH_READING"
GRADE_NO_LIVE_SAMPLES = "UNAVAILABLE_NO_LIVE_SAMPLES"
GRADE_CONFOUNDED = "CONFOUNDED_OVERLAPPING_CELL"
GRADE_SOLE_CELL = "SOLE_CELL_INFERRED_FROM_LEDGER"

_STATUS_RECEIPT_NAME = "resource_safe_run_status.json"
_WALK_PRUNE_NAMES = frozenset(
    {
        ".git",
        "retained",
        "runs",
        "checkpoints",
        "frames",
        "node_modules",
        "__pycache__",
        ".venv",
        "site-packages",
    }
)
_WALK_PRUNE_PREFIXES = ("sealed_source_", "stage_", "step_", "shard_")
_WALK_MAX_DEPTH = 6
_GIB = 1024.0**3


def utc_text(value: dt.datetime | None = None) -> str:
    return (value or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(text: str | None) -> dt.datetime | None:
    if not text:
        return None
    raw = str(text).strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.UTC)


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


# ── the row ─────────────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class MeasuredPeak:
    """One governed exit (or one live observation) of one cell family."""

    family: str
    cell_id: str | None
    peak_rss_gib: float
    peak_rss_observed: bool
    declared_peak_gib: float | None
    system_availability_delta_gib: float | None
    attribution_grade: str
    status: str | None
    exit_code: int | None
    elapsed_s: float | None
    start_utc: str | None
    status_receipt_path: str
    manifest_path: str | None
    config_path: str | None
    config_sha256: str | None
    pre_launch_available_gib: float | None
    min_available_while_live_gib: float | None
    blackbox_samples: int
    artifact_gib: float | None = None
    artifact_bytes_truncated: bool = False
    recorded_utc: str = dataclasses.field(default_factory=utc_text)

    @property
    def governed_peak_gib(self) -> float:
        """The number a launcher should declare: the LARGER of the two measurements.

        RSS is a floor for a Metal cell and the availability delta is a floor for a CPU-only one
        (the machine may have been busy for other reasons).  Charging the larger is the only
        fail-closed reading of two partial instruments.
        """
        candidates = [self.peak_rss_gib]
        if self.system_availability_delta_gib is not None:
            candidates.append(self.system_availability_delta_gib)
        return max(candidates)

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["schema"] = MEASURED_PEAK_SCHEMA
        payload["governed_peak_gib"] = round(self.governed_peak_gib, 4)
        payload["score_claim"] = False
        payload["promotable"] = False
        return payload


# ── the memory blackbox as the pre-launch / while-live instrument ───────────────────────────────


def blackbox_window(
    start: dt.datetime,
    end: dt.datetime | None = None,
    *,
    blackbox_path: Path | None = None,
    pre_window_s: float = 120.0,
) -> tuple[float | None, float | None, int]:
    """``(available just before ``start``, min available in [start, end], sample count)``.

    Reuses ``.omx/state/memory_blackbox.jsonl`` -- the sampler this repo already runs -- rather
    than forking a second memory instrument.  The pre-launch reading is the LAST sample strictly
    before ``start`` within ``pre_window_s``; a launch whose blackbox coverage has a hole gets
    ``None`` and the row is graded ``UNAVAILABLE_NO_PRE_LAUNCH_READING`` instead of guessing.
    """
    path = MEMORY_BLACKBOX if blackbox_path is None else Path(blackbox_path)
    if not path.is_file():
        return None, None, 0
    finish = end or dt.datetime.now(dt.UTC)
    pre_floor = start - dt.timedelta(seconds=pre_window_s)
    pre: float | None = None
    pre_ts: dt.datetime | None = None
    low: float | None = None
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"available_gib"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stamp = _parse_utc(row.get("ts_iso"))
                if stamp is None:
                    continue
                available = row.get("available_gib")
                if not isinstance(available, (int, float)):
                    continue
                if pre_floor <= stamp < start and (pre_ts is None or stamp > pre_ts):
                    pre, pre_ts = float(available), stamp
                elif start <= stamp <= finish:
                    count += 1
                    low = float(available) if low is None else min(low, float(available))
    except OSError:
        return None, None, 0
    return pre, low, count


# ── ledger I/O ──────────────────────────────────────────────────────────────────────────────────


def ledger_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    override = os.environ.get("TAC_MEASURED_PEAKS_LEDGER")
    return Path(override) if override else MEASURED_PEAKS_LEDGER


def append_row(row: Mapping[str, Any], path: Path | None = None) -> Path:
    """Append one row under ``LOCK_EX``.  Append-only: rows are never rewritten."""
    target = ledger_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(row), sort_keys=True, default=str)
    with target.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return target


def read_rows(path: Path | None = None) -> list[dict[str, Any]]:
    target = ledger_path(path)
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("schema") == MEASURED_PEAK_SCHEMA:
            rows.append(row)
    return rows


def lookup_family(family: str, *, path: Path | None = None) -> dict[str, Any] | None:
    """The governing peak for ``family``: the MAXIMUM measured across every recorded run.

    Not the mean and not the latest -- a memory reservation that holds only for the average run is
    a reservation that fails on the run that matters.
    """
    rows = [row for row in read_rows(path) if row.get("family") == family]
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("governed_peak_gib") or 0.0))
    return {
        "family": family,
        "row_count": len(rows),
        "governed_peak_gib": float(best.get("governed_peak_gib") or 0.0),
        "peak_rss_gib": best.get("peak_rss_gib"),
        "system_availability_delta_gib": best.get("system_availability_delta_gib"),
        "attribution_grade": best.get("attribution_grade"),
        "artifact_gib": max(
            (float(row.get("artifact_gib") or 0.0) for row in rows),
            default=0.0,
        ),
        "from_row_recorded_utc": best.get("recorded_utc"),
        "status_receipt_path": best.get("status_receipt_path"),
    }


# ── building a row from a status receipt ────────────────────────────────────────────────────────


def family_of(receipt: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> str:
    """The family a memory profile belongs to: the TRAINER ENTRY POINT, not the seed.

    Three cells (ng2 area_cap, ng3 tau_band, ng4 continuous_objective) are three configs of ONE
    trainer, and they share a memory profile because the trainer is what allocates.  Keying on the
    cell_id would give every seed its own family and no family would ever have a measured row.
    """
    argv = receipt.get("argv") if isinstance(receipt.get("argv"), Sequence) else []
    for part in [str(item) for item in argv]:
        if part.endswith(".py"):
            return Path(part).stem
    if manifest:
        m_argv = manifest.get("argv") if isinstance(manifest.get("argv"), Sequence) else []
        for part in [str(item) for item in m_argv]:
            if part.endswith(".py"):
                return Path(part).stem
    label = receipt.get("label")
    return str(label) if label else "unknown"


def _config_path_from_argv(argv: Sequence[Any]) -> Path | None:
    parts = [str(item) for item in argv]
    for index, item in enumerate(parts):
        if item == "run-config" and index + 1 < len(parts):
            return Path(parts[index + 1])
    return None


def _overlaps_another_cell(start: dt.datetime, end: dt.datetime, rows: Iterable[Mapping[str, Any]]) -> bool:
    for row in rows:
        other_start = _parse_utc(row.get("start_utc"))
        if other_start is None:
            continue
        elapsed = row.get("elapsed_s")
        other_end = other_start + dt.timedelta(seconds=float(elapsed or 0.0))
        if other_start < end and start < other_end:
            return True
    return False


def row_from_status_receipt(
    receipt_path: Path,
    *,
    manifest_path: Path | None = None,
    family: str | None = None,
    blackbox_path: Path | None = None,
    ledger: Path | None = None,
) -> MeasuredPeak | None:
    """Build a ``measured_peak.v1`` row from a ``safe_run`` status receipt.

    Works for SEALED-SOURCE launcher copies: the receipt schema predates this tool, so nothing has
    to have been instrumented at launch time for the peak to be recoverable.
    """
    receipt = _read_json(receipt_path)
    if receipt is None or receipt.get("schema") != "safe_run_status_receipt.v1":
        return None
    if manifest_path is None:
        sibling = receipt_path.parent / "launch_manifest.json"
        manifest_path = sibling if sibling.is_file() else None
    manifest = _read_json(manifest_path)

    peak_mib = receipt.get("peak_rss_mib")
    peak_gib = float(peak_mib) / 1024.0 if isinstance(peak_mib, (int, float)) else 0.0

    declared: float | None = None
    if isinstance(manifest, dict):
        budget = manifest.get("resource_budget")
        if isinstance(budget, Mapping):
            try:
                declared = float(budget.get("measured_peak_rss_gib"))
            except (TypeError, ValueError):
                declared = None

    start = _parse_utc(receipt.get("start_utc"))
    elapsed = receipt.get("elapsed_s")
    end = start + dt.timedelta(seconds=float(elapsed or 0.0)) if start is not None else None
    pre = low = None
    samples = 0
    if start is not None:
        pre, low, samples = blackbox_window(start, end, blackbox_path=blackbox_path)

    delta: float | None = None
    if pre is None:
        grade = GRADE_NO_PRE_LAUNCH
    elif low is None or samples == 0:
        grade = GRADE_NO_LIVE_SAMPLES
    else:
        delta = max(0.0, pre - low)
        others = [row for row in read_rows(ledger) if row.get("status_receipt_path") != str(receipt_path)]
        confounded = start is not None and end is not None and _overlaps_another_cell(start, end, others)
        grade = GRADE_CONFOUNDED if confounded else GRADE_SOLE_CELL

    argv = receipt.get("argv") if isinstance(receipt.get("argv"), Sequence) else []
    config_path = _config_path_from_argv(argv)
    config = _read_json(config_path)

    # DISK, measured the same way: what did this family's run actually write? The launcher's
    # storage waterfall defaults its artifact budget from this rather than from a typed guess.
    artifact: float | None = None
    truncated = False
    output_dir: Path | None = None
    if isinstance(config, dict) and isinstance(config.get("output"), str) and config["output"]:
        output_dir = Path(config["output"])
    elif isinstance(manifest, dict) and isinstance(manifest.get("output_dir"), str):
        output_dir = Path(manifest["output_dir"])
    if output_dir is not None and output_dir.is_dir():
        artifact, truncated = directory_gib(output_dir)
    return MeasuredPeak(
        family=family or family_of(receipt, manifest),
        cell_id=(str(config.get("cell_id")) if isinstance(config, dict) and config.get("cell_id") else None),
        peak_rss_gib=round(peak_gib, 6),
        peak_rss_observed=bool(receipt.get("peak_rss_observed")),
        declared_peak_gib=declared,
        system_availability_delta_gib=None if delta is None else round(delta, 4),
        attribution_grade=grade,
        status=(str(receipt["status"]) if receipt.get("status") else None),
        exit_code=(int(receipt["exit"]) if isinstance(receipt.get("exit"), int) else None),
        elapsed_s=(float(elapsed) if isinstance(elapsed, (int, float)) else None),
        start_utc=(None if start is None else start.strftime("%Y-%m-%dT%H:%M:%SZ")),
        status_receipt_path=str(receipt_path),
        manifest_path=None if manifest_path is None else str(manifest_path),
        config_path=None if config_path is None else str(config_path),
        config_sha256=_sha256(config_path),
        pre_launch_available_gib=None if pre is None else round(pre, 4),
        min_available_while_live_gib=None if low is None else round(low, 4),
        blackbox_samples=samples,
        artifact_gib=None if artifact is None else round(artifact, 4),
        artifact_bytes_truncated=truncated,
    )


def directory_gib(root: Path, *, file_cap: int = 200_000) -> tuple[float, bool]:
    """``(size GiB, truncated)`` for one run directory.

    Bounded by ``file_cap`` so a launcher preflight can never turn into an unbounded walk of a
    retained-payload tree.  Truncation is REPORTED, never silently rounded away -- a truncated
    reading is a floor, and a floor is still a usable fail-closed budget.
    """
    total = 0
    seen = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    seen += 1
                    total += entry.stat().st_size
                    if seen >= file_cap:
                        return total / _GIB, True
            except OSError:
                continue
    return total / _GIB, False


def find_status_receipts(root: Path, max_depth: int = _WALK_MAX_DEPTH) -> list[Path]:
    """Status receipts under ``root``, with the same bulk pruning the governor's walk uses."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue
            try:
                if entry.is_file() and name == _STATUS_RECEIPT_NAME:
                    found.append(Path(entry.path))
                elif (
                    entry.is_dir()
                    and depth < max_depth
                    and name not in _WALK_PRUNE_NAMES
                    and not name.startswith(_WALK_PRUNE_PREFIXES)
                ):
                    stack.append((Path(entry.path), depth + 1))
            except OSError:
                continue
    return sorted(found)


def already_recorded(receipt_path: Path, *, ledger: Path | None = None) -> bool:
    """True when this receipt has a row with the same elapsed time (the run has not moved on)."""
    receipt = _read_json(receipt_path)
    elapsed = None if receipt is None else receipt.get("elapsed_s")
    for row in read_rows(ledger):
        if row.get("status_receipt_path") != str(receipt_path):
            continue
        if elapsed is None or row.get("elapsed_s") == elapsed:
            return True
    return False


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────


def _cmd_record(args: argparse.Namespace) -> int:
    row = row_from_status_receipt(
        args.status_receipt,
        manifest_path=args.manifest,
        family=args.family,
        blackbox_path=args.blackbox,
        ledger=args.ledger,
    )
    if row is None:
        print(f"measured_peaks: not a safe_run status receipt: {args.status_receipt}", file=sys.stderr)
        return 2
    payload = row.as_dict()
    if not args.dry_run:
        append_row(payload, args.ledger)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_harvest(args: argparse.Namespace) -> int:
    written: list[dict[str, Any]] = []
    skipped = 0
    for root in args.root:
        for receipt_path in find_status_receipts(Path(root), args.max_depth):
            if not args.force and already_recorded(receipt_path, ledger=args.ledger):
                skipped += 1
                continue
            row = row_from_status_receipt(
                receipt_path, family=args.family, blackbox_path=args.blackbox, ledger=args.ledger
            )
            if row is None:
                continue
            payload = row.as_dict()
            if not args.dry_run:
                append_row(payload, args.ledger)
            written.append(payload)
    print(
        json.dumps(
            {
                "schema": "measured_peak_harvest.v1",
                "harvested_utc": utc_text(),
                "roots": [str(root) for root in args.root],
                "recorded": len(written),
                "skipped_already_recorded": skipped,
                "dry_run": bool(args.dry_run),
                "rows": written,
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_lookup(args: argparse.Namespace) -> int:
    found = lookup_family(args.family, path=args.ledger)
    if found is None:
        print(json.dumps({"family": args.family, "found": False}, indent=2))
        return 2
    print(json.dumps({**found, "found": True}, indent=2, sort_keys=True))
    return 0


def _cmd_families(args: argparse.Namespace) -> int:
    families = sorted({str(row.get("family")) for row in read_rows(args.ledger)})
    payload = [lookup_family(name, path=args.ledger) for name in families]
    print(json.dumps({"families": payload, "count": len(families)}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common_ledger: dict[str, Any] = {"type": Path, "default": None, "help": "ledger path override"}

    record = sub.add_parser("record", help="append one row from a safe_run status receipt")
    record.add_argument("--status-receipt", type=Path, required=True)
    record.add_argument("--manifest", type=Path, default=None)
    record.add_argument("--family", default=None)
    record.add_argument("--blackbox", type=Path, default=None)
    record.add_argument("--ledger", **common_ledger)
    record.add_argument("--dry-run", action="store_true")
    record.set_defaults(func=_cmd_record)

    harvest = sub.add_parser("harvest", help="record every status receipt under one or more roots")
    harvest.add_argument("--root", type=Path, action="append", required=True)
    harvest.add_argument("--max-depth", type=int, default=_WALK_MAX_DEPTH)
    harvest.add_argument("--family", default=None)
    harvest.add_argument("--blackbox", type=Path, default=None)
    harvest.add_argument("--ledger", **common_ledger)
    harvest.add_argument("--force", action="store_true", help="re-record receipts already in the ledger")
    harvest.add_argument("--dry-run", action="store_true")
    harvest.set_defaults(func=_cmd_harvest)

    lookup = sub.add_parser("lookup", help="the governing measured peak for one family (rc=2 when absent)")
    lookup.add_argument("--family", required=True)
    lookup.add_argument("--ledger", **common_ledger)
    lookup.set_defaults(func=_cmd_lookup)

    families = sub.add_parser("families", help="every family with a measured row")
    families.add_argument("--ledger", **common_ledger)
    families.set_defaults(func=_cmd_families)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
