#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Map OP3-V3 top-K byte targets to FEC6 payload sections/tensors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.fec6_byte_targets import (  # noqa: E402
    ByteRange,
    decoder_tensor_ranges,
    parse_fec6_sections,
    section_overlaps,
    summarize_tensor_shortlist,
    tensor_overlaps,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_codec(runtime_dir: Path) -> ModuleType:
    src_dir = runtime_dir / "src"
    codec_py = src_dir / "codec.py"
    if not codec_py.is_file():
        raise SystemExit(f"codec.py not found under runtime: {codec_py}")
    sys.path.insert(0, str(src_dir))
    spec = importlib.util.spec_from_file_location("_op3v3_fec6_codec", codec_py)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load codec module: {codec_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("_op3v3_fec6_codec", None)
    spec.loader.exec_module(module)
    return module


def _topk_summary(topk: dict[str, Any], k_top: int) -> dict[str, Any]:
    for row in topk.get("topk_summaries", []):
        if isinstance(row, dict) and int(row.get("k_top", -1)) == int(k_top):
            return row
    raise SystemExit(f"k_top={k_top} not found in top-K artifact")


def _axis_share(axis_score: dict[str, float]) -> dict[str, float]:
    total = sum(float(axis_score.get(axis, 0.0)) for axis in ("seg", "pose", "rate"))
    if total == 0.0:
        return {"seg": 0.0, "pose": 0.0, "rate": 0.0}
    return {axis: float(axis_score.get(axis, 0.0)) / total for axis in ("seg", "pose", "rate")}


def _records_by_run(summary: dict[str, Any]) -> list[dict[str, Any]]:
    records_by_byte = {
        int(record["byte_index"]): record
        for record in summary.get("top_records", [])
        if isinstance(record, dict) and "byte_index" in record
    }
    rows = []
    for run in summary.get("top_byte_runs", []):
        start = int(run["start"])
        end_inclusive = int(run["end"])
        axis_sum = {"seg": 0.0, "pose": 0.0, "rate": 0.0}
        score_sum = 0.0
        records = []
        for byte_index in range(start, end_inclusive + 1):
            record = records_by_byte.get(byte_index)
            if record is None:
                continue
            records.append(record)
            score_sum += float(record.get("score_impact_abs_sum", 0.0))
            axis_score = record.get("axis_score_impact")
            if isinstance(axis_score, dict):
                for axis in axis_sum:
                    axis_sum[axis] += float(axis_score.get(axis, 0.0))
        rows.append(
            {
                "range": {"start": start, "end": end_inclusive + 1, "length": end_inclusive - start + 1},
                "top_record_count_in_artifact": len(records),
                "score_impact_abs_sum_in_record_limit": score_sum,
                "axis_score_impact_abs_sum_in_record_limit": axis_sum,
                "axis_share_in_record_limit": _axis_share(axis_sum),
            }
        )
    return rows


def build_map(args: argparse.Namespace) -> dict[str, Any]:
    topk_path = args.topk_targets.resolve()
    archive_bin = args.archive_bin.resolve()
    runtime_dir = args.runtime_dir.resolve()
    topk = _read_json(topk_path)
    anchor = topk.get("anchor")
    if not isinstance(anchor, dict):
        raise SystemExit("top-K artifact missing anchor")
    expected_member_sha = anchor.get("gradient_subject_sha256")
    member_sha = _sha256_file(archive_bin)
    if args.expected_member_sha256 and member_sha != args.expected_member_sha256:
        raise SystemExit(
            f"archive member SHA mismatch: expected {args.expected_member_sha256}, got {member_sha}"
        )
    if expected_member_sha and member_sha != expected_member_sha:
        raise SystemExit(f"top-K anchor subject SHA mismatch: expected {expected_member_sha}, got {member_sha}")

    codec = _load_codec(runtime_dir)
    member_bytes = archive_bin.read_bytes()
    sections = parse_fec6_sections(
        member_bytes,
        decoder_blob_len=int(codec.DECODER_BLOB_LEN),
        latent_blob_len=int(codec.LATENT_BLOB_LEN),
    )
    tensors = decoder_tensor_ranges(codec, member_bytes)
    summary = _topk_summary(topk, args.k_top)
    run_rows = _records_by_run(summary)
    for row in run_rows:
        byte_range = ByteRange(int(row["range"]["start"]), int(row["range"]["end"]))
        row["section_overlaps"] = section_overlaps(byte_range, sections)
        row["decoder_tensor_overlaps"] = tensor_overlaps(byte_range, tensors)
        row["mutation_policy"] = (
            "codec_grammar_rebuild_required; raw byte flips are diagnostic-only "
            "and not contest-promotion evidence"
        )

    tensor_shortlist = summarize_tensor_shortlist(summary.get("top_records", []), tensors)
    for row in tensor_shortlist:
        total = float(row["score_impact_abs_sum"])
        axis_sum = row["axis_score_impact_abs_sum"]
        row["axis_share"] = _axis_share(axis_sum) if total else {"seg": 0.0, "pose": 0.0, "rate": 0.0}

    return {
        "schema": "op3v3_fec6_byte_target_layout_map_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/map_op3v3_byte_targets_to_fec6_layout.py",
        "inputs": {
            "topk_targets_path": str(topk_path),
            "archive_bin_path": str(archive_bin),
            "archive_bin_sha256": member_sha,
            "runtime_dir": str(runtime_dir),
        },
        "anchor": anchor,
        "k_top": int(summary["k_top"]),
        "top_byte_indices": summary.get("top_byte_indices"),
        "sections": [section.as_dict() for section in sections],
        "target_runs": run_rows,
        "decoder_tensor_shortlist": tensor_shortlist,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "mapping_authority": (
                "FEC6 section offsets are exact; decoder tensor compressed ranges are "
                "the same approximate uniform decompressed-to-compressed mapping used by "
                "the OP3-V3 gradient projector."
            ),
            "required_before_candidate_claim": [
                "grammar-aware tensor-domain mutation or legal sidecar/selector mutation",
                "re-encoded archive member and rebuilt archive.zip",
                "official inflate raw hash/locality/no-op control",
                "advisory component-response measurement",
                "exact contest eval for any promotion claim",
            ],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk-targets", type=Path, required=True)
    parser.add_argument("--archive-bin", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--expected-member-sha256")
    parser.add_argument("--k-top", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_map(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "k_top": payload["k_top"],
                "target_run_count": len(payload["target_runs"]),
                "tensor_shortlist_count": len(payload["decoder_tensor_shortlist"]),
                "top_tensor": payload["decoder_tensor_shortlist"][0] if payload["decoder_tensor_shortlist"] else None,
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
