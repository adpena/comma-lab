#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize PR106 component-moving score-table cells into candidate archives.

The input plan is produced by ``tools/plan_pr106_component_moving_cells.py``.
It is a compress-time marginal table, not a score claim.  This tool converts
selected latent-sidecar cells into byte-closed PR106 sidecar archives so the
next step can be paired exact CPU/CUDA eval on real packets.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    apply_proxy_evidence_boundary,
    ordered_unique,
    validate_proxy_candidate,
)
from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_LATENT_N_PAIRS,
    PR106_NO_OP_DIM,
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106SidecarPacket,
    decode_brotli_dim_delta_sidecar_payload,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    encode_brotli_dim_delta_sidecar_payload,
    parse_pr106_sidecar_packet,
    pr106_sidecar_consumed_byte_proof,
    pr106_sidecar_manifest,
    read_single_stored_member_archive,
    sha256_hex,
)
from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

SCHEMA = "pr106_component_moving_cell_materialization_v1"
SUMMARY_SCHEMA = "pr106_component_moving_cell_materialization_summary_v1"
PLAN_SCHEMA = "pr106_component_moving_cell_plan_v1"
DEFAULT_PLAN = REPO_ROOT / ".omx/research/pr106_component_moving_cells_20260515_codex.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/results"
DISPATCH_BLOCKERS = (
    "component_cells_are_proxy_score_table_evidence",
    "requires_lane_dispatch_claim_before_exact_eval",
    "requires_paired_contest_cuda_auth_eval_on_materialized_archive",
    "requires_paired_contest_cpu_auth_eval_on_materialized_archive",
    "requires_adjudicated_component_recompute_before_score_claim",
)


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / (
        "pr106_component_moving_cell_candidates_"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": repo_relative(path, REPO_ROOT),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _load_plan(plan_path: Path) -> dict[str, Any]:
    payload = read_json(plan_path)
    if not isinstance(payload, dict):
        raise ValueError(f"component plan must be a JSON object: {plan_path}")
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"component plan schema mismatch: {payload.get('schema')!r}")
    if payload.get("kind") != "latent_sidecar":
        raise ValueError("only latent_sidecar component-cell plans are supported")
    if payload.get("score_claim") is not False:
        raise ValueError("component plan must keep score_claim=false")
    cells = payload.get("top_cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("component plan must contain non-empty top_cells")
    return payload


def _resolve_source_archive(plan: Mapping[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit
    else:
        custody = plan.get("source_custody")
        if not isinstance(custody, Mapping):
            raise ValueError("component plan source_custody missing")
        local = custody.get("local_source_archive")
        if not isinstance(local, Mapping):
            raise ValueError("component plan local_source_archive missing")
        local_path = local.get("path")
        if not isinstance(local_path, str) or not local_path:
            raise ValueError("component plan local_source_archive.path missing")
        path = Path(local_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"source archive not found: {path}")
    return path


def _validate_source_archive(plan: Mapping[str, Any], source_archive: Path) -> dict[str, object]:
    member = read_single_stored_member_archive(source_archive.read_bytes())
    payload_sha256 = sha256_hex(member.payload)
    custody = plan.get("source_custody")
    blockers: list[str] = []
    warnings: list[str] = []
    expected_payload_sha: str | None = None
    expected_archive_sha: str | None = None
    expected_member_name: str | None = None
    if isinstance(custody, Mapping):
        local = custody.get("local_source_archive")
        if isinstance(local, Mapping):
            expected_payload_sha = (
                local.get("member_sha256") if isinstance(local.get("member_sha256"), str) else None
            )
            expected_archive_sha = (
                local.get("sha256") if isinstance(local.get("sha256"), str) else None
            )
            expected_member_name = (
                local.get("member_name") if isinstance(local.get("member_name"), str) else None
            )
        manifest_zero = custody.get("manifest_source_zero_bin_sha256")
        if expected_payload_sha is None and isinstance(manifest_zero, str):
            expected_payload_sha = manifest_zero
    if expected_payload_sha is not None and payload_sha256 != expected_payload_sha:
        blockers.append("source_payload_sha256_mismatch")
    if expected_member_name is not None and member.name != expected_member_name:
        blockers.append("source_member_name_mismatch")
    archive_sha = sha256_file(source_archive)
    if expected_archive_sha is not None and archive_sha != expected_archive_sha:
        warnings.append("source_archive_zip_sha256_differs_from_plan_local")
    return {
        "archive": _artifact(source_archive),
        "member_name": member.name,
        "member_bytes": len(member.payload),
        "member_sha256": payload_sha256,
        "expected_member_sha256": expected_payload_sha,
        "expected_member_name": expected_member_name,
        "blockers": ordered_unique(blockers),
        "warnings": ordered_unique(warnings),
    }


def _cell_candidate(cell: Mapping[str, Any]) -> tuple[int, int, int]:
    pair_idx = int(cell.get("pair_idx"))
    candidate = cell.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError(f"cell {cell.get('cell_id')} has no candidate object")
    dim = int(candidate.get("dim"))
    delta_q = int(candidate.get("delta_q"))
    if pair_idx < 0 or pair_idx >= PR106_LATENT_N_PAIRS:
        raise ValueError(f"pair_idx out of range for cell {cell.get('cell_id')}: {pair_idx}")
    if dim < 0 or dim >= 28:
        raise ValueError(f"latent dim out of range for cell {cell.get('cell_id')}: {dim}")
    if delta_q == 0 or delta_q < -127 or delta_q > 127:
        raise ValueError(f"delta_q out of range for cell {cell.get('cell_id')}: {delta_q}")
    return pair_idx, dim, delta_q


def _arrays_for_cells(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]], list[str]]:
    dims = np.full(PR106_LATENT_N_PAIRS, PR106_NO_OP_DIM, dtype=np.uint8)
    deltas = np.zeros(PR106_LATENT_N_PAIRS, dtype=np.int8)
    applied: list[Mapping[str, Any]] = []
    skipped: list[str] = []
    for cell in cells:
        pair_idx, dim, delta_q = _cell_candidate(cell)
        if dims[pair_idx] != PR106_NO_OP_DIM:
            skipped.append(str(cell.get("cell_id") or f"pair{pair_idx}"))
            continue
        dims[pair_idx] = np.uint8(dim)
        deltas[pair_idx] = np.int8(delta_q)
        applied.append(cell)
    return dims, deltas, applied, skipped


def _archive_for_arrays(
    *,
    source_archive: Path,
    dims: np.ndarray,
    deltas: np.ndarray,
) -> tuple[bytes, dict[str, object]]:
    source_member = read_single_stored_member_archive(source_archive.read_bytes())
    sidecar_payload = encode_brotli_dim_delta_sidecar_payload(dims, deltas, quality=11)
    packet = PR106SidecarPacket(
        format_id=PR106_SIDECAR_FORMAT_BROTLI,
        pr106_bytes=source_member.payload,
        sidecar_payload=sidecar_payload,
        framing_meta=None,
    )
    packet_payload = emit_pr106_sidecar_packet(packet)
    reparsed = parse_pr106_sidecar_packet(packet_payload)
    reparsed_dims, reparsed_deltas = decode_brotli_dim_delta_sidecar_payload(
        reparsed.sidecar_payload
    )
    if not (np.array_equal(reparsed_dims, dims) and np.array_equal(reparsed_deltas, deltas)):
        raise ValueError("component-cell sidecar packet failed semantic roundtrip")
    candidate_member = type(source_member)(
        name=source_member.name,
        payload=packet_payload,
        date_time=source_member.date_time,
        external_attr=source_member.external_attr,
        create_system=source_member.create_system,
        flag_bits=source_member.flag_bits,
        comment=source_member.comment,
        extra=source_member.extra,
        archive_comment=source_member.archive_comment,
    )
    diagnostics = {
        "source_member_name": source_member.name,
        "source_member_sha256": sha256_hex(source_member.payload),
        "packet_payload_sha256": sha256_hex(packet_payload),
        "sidecar_payload_bytes": len(sidecar_payload),
        "sidecar_payload_sha256": sha256_hex(sidecar_payload),
        "nonzero_correction_count": int(np.count_nonzero(dims != PR106_NO_OP_DIM)),
        "dim_sha256": sha256_hex(dims.astype(np.uint8).tobytes()),
        "delta_q_sha256": sha256_hex(deltas.astype(np.int8).tobytes()),
        "packet_ir": {
            "candidate": pr106_sidecar_manifest(packet),
            "candidate_consumed_byte_proof": pr106_sidecar_consumed_byte_proof(packet),
        },
    }
    return emit_single_stored_member_archive(candidate_member), diagnostics


def _candidate_id(mode: str, index: int, cells: Sequence[Mapping[str, Any]]) -> str:
    if mode == "single":
        cell = cells[0]
        return str(cell.get("cell_id") or f"single_{index}").replace(":", "_")
    return f"prefix_top_{index}"


def _materialize_one(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    source_archive: Path,
    output_dir: Path,
    mode: str,
    index: int,
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    candidate_id = _candidate_id(mode, index, cells)
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    dims, deltas, applied_cells, skipped_duplicates = _arrays_for_cells(cells)
    archive_bytes, diagnostics = _archive_for_arrays(
        source_archive=source_archive,
        dims=dims,
        deltas=deltas,
    )
    archive_path = candidate_dir / "archive.zip"
    archive_path.write_bytes(archive_bytes)
    source_bytes = source_archive.stat().st_size
    archive_delta = archive_path.stat().st_size - source_bytes
    rate_delta = 25.0 * archive_delta / 37_545_489.0
    cell_component_delta = float(
        sum(float(cell.get("component_score_delta_no_rate", 0.0)) for cell in cells)
    )
    cell_net_delta = float(sum(float(cell.get("net_score_delta_charged", 0.0)) for cell in cells))
    manifest = apply_proxy_evidence_boundary(
        {
            "schema": SCHEMA,
            "candidate_id": candidate_id,
            "lane_id": "lane_pr106_latent_sidecar_component_cell",
            "source_plan": _artifact(plan_path),
            "source_plan_label": plan.get("label"),
            "source_score_axis_label": plan.get("axis_labels", {}).get(
                "source_score_axis_label"
            ),
            "mode": mode,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "source_archive": _artifact(source_archive),
            "archive": _artifact(archive_path),
            "archive_delta_bytes_vs_source": archive_delta,
            "rate_component_score_delta_vs_source": rate_delta,
            "proxy_component_delta_no_rate_sum": cell_component_delta,
            "proxy_net_delta_charged_sum_from_plan": cell_net_delta,
            "requested_cells": [dict(cell) for cell in cells],
            "applied_cells": [dict(cell) for cell in applied_cells],
            "skipped_duplicate_pair_cells": skipped_duplicates,
            "semantic_packet": diagnostics,
            "score_claim_blockers": list(DISPATCH_BLOCKERS),
            "exact_eval_dispatch_blockers": list(DISPATCH_BLOCKERS),
            "promotion_requires": (
                "paired [contest-CUDA] and [contest-CPU] auth eval on this "
                "archive plus component recomputation"
            ),
        },
        dispatch_blockers=DISPATCH_BLOCKERS,
    )
    violations = validate_proxy_candidate(manifest)
    if violations:
        raise RuntimeError(
            "component-cell materialization leaked score authority: "
            + ", ".join(violations)
        )
    manifest_path = candidate_dir / "candidate_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest"] = _artifact(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def materialize_from_plan(
    *,
    plan_path: Path,
    source_archive: Path | None,
    output_dir: Path,
    singles: int,
    prefixes: Sequence[int],
) -> dict[str, object]:
    plan = _load_plan(plan_path)
    resolved_source = _resolve_source_archive(plan, source_archive)
    source_audit = _validate_source_archive(plan, resolved_source)
    if source_audit["blockers"]:
        raise ValueError(
            "source archive does not match component plan custody: "
            + ", ".join(str(item) for item in source_audit["blockers"])
        )
    top_cells = [cell for cell in plan["top_cells"] if isinstance(cell, Mapping)]
    if not top_cells:
        raise ValueError("component plan top_cells has no object rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []
    for idx, cell in enumerate(top_cells[: max(0, singles)], start=1):
        manifests.append(
            _materialize_one(
                plan=plan,
                plan_path=plan_path,
                source_archive=resolved_source,
                output_dir=output_dir,
                mode="single",
                index=idx,
                cells=[cell],
            )
        )
    for prefix in sorted({int(v) for v in prefixes if int(v) > 0}):
        cells = top_cells[:prefix]
        manifests.append(
            _materialize_one(
                plan=plan,
                plan_path=plan_path,
                source_archive=resolved_source,
                output_dir=output_dir,
                mode="prefix",
                index=prefix,
                cells=cells,
            )
        )
    summary = apply_proxy_evidence_boundary(
        {
            "schema": SUMMARY_SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "source_plan": _artifact(plan_path),
            "source_archive_audit": source_audit,
            "candidate_count": len(manifests),
            "candidates": [
                {
                    "candidate_id": manifest["candidate_id"],
                    "archive": manifest["archive"],
                    "manifest": manifest["manifest"],
                    "archive_delta_bytes_vs_source": manifest[
                        "archive_delta_bytes_vs_source"
                    ],
                    "proxy_net_delta_charged_sum_from_plan": manifest[
                        "proxy_net_delta_charged_sum_from_plan"
                    ],
                }
                for manifest in manifests
            ],
            "dispatch_blockers": list(DISPATCH_BLOCKERS),
            "next_step": (
                "Claim one or more candidate archives and run paired exact "
                "[contest-CUDA] / [contest-CPU] auth eval before score language."
            ),
        },
        dispatch_blockers=DISPATCH_BLOCKERS,
    )
    violations = validate_proxy_candidate(summary)
    if violations:
        raise RuntimeError(
            "component-cell summary leaked score authority: " + ", ".join(violations)
        )
    summary_path = output_dir / "materialization_summary.json"
    write_json(summary_path, summary)
    summary["summary"] = _artifact(summary_path)
    write_json(summary_path, summary)
    return summary


def _parse_prefixes(value: str) -> list[int]:
    if not value.strip():
        return []
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("prefix values must be positive")
        out.append(parsed)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--singles", type=int, default=3)
    parser.add_argument("--prefixes", type=_parse_prefixes, default=[1, 4, 16])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or default_output_dir()
    summary = materialize_from_plan(
        plan_path=args.plan_json,
        source_archive=args.source_archive,
        output_dir=output_dir,
        singles=args.singles,
        prefixes=args.prefixes,
    )
    print(json_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
