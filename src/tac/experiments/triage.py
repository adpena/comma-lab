#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _canonical_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve(strict=False))


def _slug_from_meta_path(path: Path) -> str:
    name = path.name
    suffix = "_best_meta.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def scan_best_meta(directory: Path) -> list[dict]:
    records: list[dict] = []
    for meta_path in sorted(directory.glob("postfilter_*best_meta.json")):
        payload = json.loads(meta_path.read_text())
        scorer_value = payload.get("scorer", payload.get("score"))
        if scorer_value is None:
            continue
        record = {
            "slug": _slug_from_meta_path(meta_path),
            "meta_path": str(meta_path),
            "epoch": payload.get("epoch"),
            "scorer": float(scorer_value),
            "int8_size": int(payload.get("int8_size", 0)),
            "fp32_path": payload.get("fp32_path"),
            "int8_path": payload.get("int8_path"),
            "meta": payload.get("meta", {}),
        }
        records.append(record)
    records.sort(key=lambda item: (item["scorer"], item["slug"]))
    return records


def scan_proxy_logs(directory: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for log_path in sorted(directory.glob("proxy*.log")):
        text = log_path.read_text(errors="ignore")
        weights_match = re.search(r'"weights":\s*"([^"]+)"', text)
        if weights_match:
            key = _canonical_path(weights_match.group(1))
            if key is not None:
                resolved[key] = str(log_path)
            continue
        line_match = re.search(r"^\[proxy-faithful\]\s+weights:\s+(.+)$", text, flags=re.MULTILINE)
        if line_match:
            key = _canonical_path(line_match.group(1).strip())
            if key is not None:
                resolved[key] = str(log_path)
    return resolved


def _deploy_blockers(record: dict) -> list[str]:
    blockers: list[str] = []
    if record["slug"] == "postfilter_dilated_h64_long1000" and record.get("meta", {}).get("variant") == "saliency_weighted":
        blockers.append("variant mismatch for special lane slug: postfilter_dilated_h64_long1000 uses saliency_weighted")
    return blockers


def build_triage_report(
    records: list[dict],
    *,
    promoted_slug: str | None = None,
    proxy_gap_threshold: float = 0.20,
    resolved_proxy_logs: dict[str, str] | None = None,
) -> dict:
    if not records:
        return {
            "reference": None,
            "proxy_gap_threshold": proxy_gap_threshold,
            "ranked": [],
        }

    reference = None
    if promoted_slug is not None:
        reference = next((item for item in records if item["slug"] == promoted_slug), None)
    if reference is None:
        reference = min(records, key=lambda item: item["scorer"])

    resolved_proxy_logs = resolved_proxy_logs or {}
    ranked: list[dict] = []
    for idx, record in enumerate(records, start=1):
        delta = float(record["scorer"]) - float(reference["scorer"])
        int8_path = _canonical_path(record.get("int8_path"))
        proxy_already_run = bool(int8_path and int8_path in resolved_proxy_logs)
        deploy_blockers = _deploy_blockers(record)
        deploy_ready = not deploy_blockers
        ranked.append(
            {
                **record,
                "rank": idx,
                "delta_vs_reference": delta,
                "proxy_already_run": proxy_already_run,
                "proxy_log": resolved_proxy_logs.get(int8_path) if int8_path else None,
                "deploy_ready": deploy_ready,
                "deploy_blockers": deploy_blockers,
                "proxy_ready": (
                    record["slug"] != reference["slug"]
                    and delta <= proxy_gap_threshold
                    and not proxy_already_run
                    and deploy_ready
                ),
            }
        )

    return {
        "reference": reference,
        "proxy_gap_threshold": proxy_gap_threshold,
        "ranked": ranked,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank saved best-meta artifacts for proxy-gate triage.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=Path("/private/tmp/pact-mine/experiments/postfilter_weights"),
        type=Path,
        help="Directory containing postfilter_*best_meta.json files.",
    )
    parser.add_argument(
        "--promoted-slug",
        default="postfilter_long1000_h64",
        help="Slug to use as the promoted reference artifact.",
    )
    parser.add_argument(
        "--proxy-gap-threshold",
        type=float,
        default=0.20,
        help="Maximum local-score gap from the promoted reference to flag a candidate as proxy-ready.",
    )
    parser.add_argument(
        "--proxy-log-dir",
        type=Path,
        default=None,
        help="Optional directory containing proxy*.log files used to mark already-resolved candidates.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = scan_best_meta(args.directory)
    resolved = scan_proxy_logs(args.proxy_log_dir) if args.proxy_log_dir is not None else {}
    report = build_triage_report(
        records,
        promoted_slug=args.promoted_slug,
        proxy_gap_threshold=args.proxy_gap_threshold,
        resolved_proxy_logs=resolved,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
