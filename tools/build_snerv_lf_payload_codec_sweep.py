#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority SNeRV LF payload codec sweep from receiver bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_lf_payload_codec_sweep import (  # noqa: E402
    DEFAULT_LF_CODEC_MODES,
    SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA,
    build_snerv_lf_payload_codec_sweep,
    render_snerv_lf_payload_codec_sweep_markdown,
)
from tac.repo_io import write_json  # noqa: E402
from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    unpack_snerv_archive,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--packet",
        type=Path,
        help="SNAR1 receiver packet whose LF planes should be decoded and swept.",
    )
    source.add_argument(
        "--lf-planes-npz",
        type=Path,
        help="NPZ containing integer LF planes to sweep in stored key order.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Codec mode to include. Defaults to the canonical portfolio.",
    )
    parser.add_argument("--baseline-mode", default="int64_lzma")
    args = parser.parse_args(argv)

    planes, source_meta = _load_planes(args.packet, args.lf_planes_npz)
    report = build_snerv_lf_payload_codec_sweep(
        planes,
        modes=tuple(args.mode or DEFAULT_LF_CODEC_MODES),
        baseline_mode=str(args.baseline_mode),
    )
    report["source"] = source_meta
    output = args.output_json.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(
            render_snerv_lf_payload_codec_sweep_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _load_planes(
    packet_path: Path | None,
    npz_path: Path | None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if packet_path is not None:
        path = packet_path.expanduser().resolve(strict=False)
        packet = path.read_bytes()
        decoded = unpack_snerv_archive(packet)
        return decoded.decode_lf_quant_planes(), {
            "kind": "snar1_packet",
            "path": path.as_posix(),
            "bytes": len(packet),
            "sha256": hashlib.sha256(packet).hexdigest(),
            "packet_sha256": decoded.packet_sha256,
            "metadata": decoded.metadata,
        }
    assert npz_path is not None
    path = npz_path.expanduser().resolve(strict=False)
    with np.load(path, allow_pickle=False) as data:
        planes = [np.asarray(data[key], dtype=np.int64) for key in data.files]
        keys = list(data.files)
    blob = path.read_bytes()
    return planes, {
        "kind": "lf_planes_npz",
        "path": path.as_posix(),
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "keys": keys,
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA,
        "report_path": report.get("report_path"),
        "source_kind": report.get("source", {}).get("kind"),
        "plane_count": report["plane_count"],
        "baseline_mode": report["baseline_mode"],
        "baseline_payload_bytes": report["baseline_payload_bytes"],
        "selected_mode": report["selected_rate_only_row"]["mode"],
        "selected_payload_bytes": report["selected_rate_only_row"]["payload_bytes"],
        "byte_price_decision_counts": report["byte_price_plan"]["decision_counts"],
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
