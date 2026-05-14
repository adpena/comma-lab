#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare packed HNeRV payload sections for a source/candidate archive pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.hnerv_payload_diff import build_hnerv_payload_diff, dump_json  # noqa: E402


def _load_manifest(path: Path | None, label: str) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    manifests = payload.get("payload_section_manifests")
    if isinstance(manifests, list):
        for manifest in manifests:
            if isinstance(manifest, dict) and str(manifest.get("label")) == label:
                return manifest
        raise SystemExit(f"missing payload_section_manifests label: {label}")
    if isinstance(payload.get("sections"), list):
        return payload
    raise SystemExit(f"{path} is not a payload section manifest or scorecard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--source-label", default="source")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--source-manifest-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-no-section-change", action="store_true")
    args = parser.parse_args()

    source_manifest = _load_manifest(args.source_manifest_json, args.source_label)
    payload = build_hnerv_payload_diff(
        args.source_archive,
        args.candidate_archive,
        source_label=args.source_label,
        candidate_label=args.candidate_label,
        source_manifest=source_manifest,
    )
    text = dump_json(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    else:
        print(text, end="")
    if args.fail_if_no_section_change and not payload["changed_section_count"]:
        raise SystemExit("no HNeRV payload section changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
