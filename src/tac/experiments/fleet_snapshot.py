#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_DIRECTORY = Path("/private/tmp/pact-mine/experiments/postfilter_weights")
META_SUFFIX = "_best_meta.json"
META_PREFIX = "postfilter_"
LOG_PREFIX = "train_"
FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
SPECIAL_LOG_ALIAS_RE = re.compile(r"^(?P<base>.+_h\d+)_long\d+$")
LOG_ROW_RE = re.compile(
    rf"^\s*(?P<epoch>\d+)\s+"
    rf"(?P<total>{FLOAT_RE})\s+"
    rf"(?P<scorer>{FLOAT_RE})\s+"
    rf"(?P<pose>{FLOAT_RE})\s+"
    rf"(?P<seg>{FLOAT_RE})\b"
)


def _slug_from_meta_path(path: Path) -> str:
    name = path.name
    if name.startswith(META_PREFIX) and name.endswith(META_SUFFIX):
        return name[len(META_PREFIX) : -len(META_SUFFIX)]
    if name.endswith(META_SUFFIX):
        return name[: -len(META_SUFFIX)]
    return path.stem


def _slug_from_log_path(path: Path) -> str:
    name = path.name
    if name.startswith(LOG_PREFIX) and name.endswith(".log"):
        return name[len(LOG_PREFIX) : -len(".log")]
    if name.endswith(".log"):
        return path.stem
    return path.stem


def _candidate_log_aliases(meta_slug: str) -> list[str]:
    aliases = [meta_slug]
    match = SPECIAL_LOG_ALIAS_RE.match(meta_slug)
    if match is not None:
        aliases.append(match.group("base"))
    return aliases


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def parse_best_meta(meta_path: Path) -> dict:
    payload = json.loads(meta_path.read_text())
    best: dict[str, object] = {}

    epoch = payload.get("epoch", payload.get("iteration"))
    scorer = payload.get("scorer", payload.get("score"))
    int8_size = payload.get("int8_size")

    epoch_value = _int_or_none(epoch)
    if epoch_value is not None:
        best["epoch"] = epoch_value

    scorer_value = _float_or_none(scorer)
    if scorer_value is not None:
        best["scorer"] = scorer_value

    int8_size_value = _int_or_none(int8_size)
    if int8_size_value is not None:
        best["int8_size"] = int8_size_value

    record = {"slug": _slug_from_meta_path(meta_path)}
    if best:
        record["best"] = best
    return record


def parse_training_log(log_path: Path) -> dict:
    latest: dict[str, object] | None = None
    for line in log_path.read_text(errors="ignore").splitlines():
        match = LOG_ROW_RE.match(line)
        if not match:
            continue
        latest = {
            "epoch": int(match.group("epoch")),
            "scorer": float(match.group("scorer")),
            "pose": float(match.group("pose")),
            "seg": float(match.group("seg")),
        }

    record = {"slug": _slug_from_log_path(log_path), "active": latest is not None}
    if latest is not None:
        record["latest"] = latest
    return record


def build_snapshot(directory: Path) -> dict:
    directory = Path(directory)
    lanes: dict[str, dict] = {}
    log_alias_to_slug: dict[str, str] = {}

    for meta_path in sorted(directory.glob(f"*{META_SUFFIX}")):
        record = parse_best_meta(meta_path)
        lane = lanes.setdefault(record["slug"], {"slug": record["slug"], "active": False})
        best = record.get("best")
        if best:
            lane["best"] = best
        for alias in _candidate_log_aliases(record["slug"]):
            log_alias_to_slug.setdefault(alias, record["slug"])

    for log_path in sorted(directory.glob(f"{LOG_PREFIX}*.log")):
        record = parse_training_log(log_path)
        canonical_slug = log_alias_to_slug.get(record["slug"], record["slug"])
        lane = lanes.setdefault(canonical_slug, {"slug": canonical_slug, "active": False})
        lane["active"] = bool(record.get("active"))
        latest = record.get("latest")
        if latest:
            lane["latest"] = latest

    return {
        "directory": str(directory),
        "lanes": [lanes[slug] for slug in sorted(lanes)],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit a concise JSON snapshot of live postfilter lanes.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEFAULT_DIRECTORY,
        type=Path,
        help="Directory containing best-meta files and train_*.log files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    snapshot = build_snapshot(args.directory)
    json.dump(snapshot, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
