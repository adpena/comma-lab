#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan top-K byte targets from a score-weighted master-gradient anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.cathedral.consumer_contract import AxisDecomposition  # noqa: E402
from tac.cathedral_consumers.top_k_byte_sensitivity_consumer import (  # noqa: E402
    consume_candidate as consume_top_k_candidate,
)
from tac.optimization.byte_score_impact import (  # noqa: E402
    AXIS_ORDER,
    parse_k_values,
    rank_bytes_by_score_impact,
    score_impact_matrix,
    summarize_topk_score_impact,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _select_anchor(args: argparse.Namespace) -> dict[str, Any]:
    if args.anchor_json is not None:
        anchor = json.loads(args.anchor_json.read_text(encoding="utf-8"))
        if not isinstance(anchor, dict):
            raise SystemExit("--anchor-json must contain a JSON object")
        return anchor
    rows = _read_jsonl(args.anchor_jsonl)
    selected = []
    for row in rows:
        if args.measurement_axis and row.get("measurement_axis") != args.measurement_axis:
            continue
        if args.archive_sha256 and row.get("archive_sha256") != args.archive_sha256:
            continue
        if args.gradient_sha256:
            gradient_path = row.get("gradient_array_path")
            if not isinstance(gradient_path, str):
                continue
            path = _resolve_repo_path(gradient_path)
            if not path.is_file() or _sha256_file(path) != args.gradient_sha256:
                continue
        selected.append(row)
    if not selected:
        raise SystemExit("no matching master-gradient anchor found")
    return selected[-1]


def _marginal_coefficients(anchor: dict[str, Any]) -> dict[str, float]:
    dominance = anchor.get("score_axis_dominance")
    if isinstance(dominance, dict):
        marginals = dominance.get("marginal_coefficients")
        if isinstance(marginals, dict):
            return {
                "seg": float(marginals["seg"]),
                "pose": float(marginals["pose"]),
                "rate": float(marginals["rate"]),
            }
    raise SystemExit("anchor missing score_axis_dominance.marginal_coefficients")


def _top_axis_lists(m_archive: Any, marginals: dict[str, float], k_axis: int) -> dict[str, list[int]]:
    return {
        axis: rank_bytes_by_score_impact(
            m_archive,
            marginals,
            k_top=k_axis,
            axis=axis,
        )
        for axis in AXIS_ORDER
    }


def _catalog356_observability(
    *,
    k_top: int,
    top_byte_indices: list[int],
    gradient_sha256: str,
    selected_axis_raw_abs_sum: dict[str, float],
) -> dict[str, Any]:
    consumer_output = consume_top_k_candidate(
        {
            "top_k_byte_indices": top_byte_indices,
            "k_top": k_top,
            "m_archive_array_sha256": gradient_sha256,
            "per_byte_sensitivity_sums": {
                "seg_axis_sum": selected_axis_raw_abs_sum["seg"],
                "pose_axis_sum": selected_axis_raw_abs_sum["pose"],
            },
            "archive_bytes_protected_delta": 0,
        }
    )
    decomp_raw = consumer_output.get("predicted_axis_decomposition")
    decomp = AxisDecomposition.from_dict(decomp_raw)
    return {
        "catalog": 356,
        "validated": True,
        "axis_decomposition": decomp.as_dict(),
        "consumer_signal_kind": consumer_output.get("consumer_signal_kind"),
        "rationale": consumer_output.get("rationale"),
        "observability_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("numpy required") from exc

    anchor = _select_anchor(args)
    gradient_path = _resolve_repo_path(str(anchor["gradient_array_path"]))
    gradient_sha256 = _sha256_file(gradient_path)
    if args.gradient_sha256 and gradient_sha256 != args.gradient_sha256:
        raise SystemExit(
            f"gradient sha mismatch: expected {args.gradient_sha256}, got {gradient_sha256}"
        )
    m_archive = np.load(gradient_path)
    n_bytes = int(m_archive.shape[0])
    if tuple(m_archive.shape) != (int(anchor.get("n_bytes")), 3):
        raise SystemExit(f"gradient shape {m_archive.shape} does not match anchor n_bytes")
    marginals = _marginal_coefficients(anchor)
    impact = score_impact_matrix(m_archive, marginals)
    raw_abs = np.abs(np.asarray(m_archive, dtype=np.float64))
    k_values = parse_k_values(args.k_values)
    summaries = []
    for k_top in k_values:
        summary = summarize_topk_score_impact(
            m_archive,
            marginals,
            k_top=k_top,
            top_record_limit=args.top_record_limit,
        )
        top_indices = summary["top_byte_indices"]
        selected_raw = raw_abs[top_indices].sum(axis=0) if top_indices else np.zeros(3)
        selected_axis_raw_abs_sum = {
            axis: float(selected_raw[i]) for i, axis in enumerate(AXIS_ORDER)
        }
        summary["selected_axis_raw_abs_gradient_sum"] = selected_axis_raw_abs_sum
        summary["catalog356_axis_decomposition_validation"] = _catalog356_observability(
            k_top=int(summary["k_top"]),
            top_byte_indices=top_indices,
            gradient_sha256=gradient_sha256,
            selected_axis_raw_abs_sum=selected_axis_raw_abs_sum,
        )
        summaries.append(summary)

    total_axis = impact.sum(axis=0)
    total_mass = float(total_axis.sum())
    plan = {
        "schema": "topk_byte_score_impact_targets_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/plan_topk_byte_score_impact_targets.py",
        "anchor": {
            "archive_sha256": anchor.get("archive_sha256"),
            "gradient_array_path": str(gradient_path),
            "gradient_array_sha256": gradient_sha256,
            "gradient_subject_sha256": anchor.get("gradient_subject_sha256"),
            "gradient_subject_bytes": anchor.get("gradient_subject_bytes"),
            "measurement_axis": anchor.get("measurement_axis"),
            "measurement_hardware": anchor.get("measurement_hardware"),
            "measurement_call_id": anchor.get("measurement_call_id"),
            "measurement_method": anchor.get("measurement_method"),
            "measurement_utc": anchor.get("measurement_utc"),
            "n_bytes": n_bytes,
            "n_pairs_total": anchor.get("n_pairs_total"),
            "n_pairs_used": anchor.get("n_pairs_used"),
            "operating_point": anchor.get("operating_point"),
            "score_axis_dominance": anchor.get("score_axis_dominance"),
        },
        "marginal_coefficients": marginals,
        "dS_d_byte": marginals["rate"],
        "total_score_impact_abs_sum": total_mass,
        "total_axis_score_impact_abs_sum": {
            axis: float(total_axis[i]) for i, axis in enumerate(AXIS_ORDER)
        },
        "total_axis_share": {
            axis: (float(total_axis[i]) / total_mass) if total_mass else 0.0
            for i, axis in enumerate(AXIS_ORDER)
        },
        "top_axis_byte_indices": _top_axis_lists(m_archive, marginals, args.top_axis_k),
        "topk_summaries": summaries,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "note": (
                "Diagnostic byte-target prior only. Packet-valid mutations still require "
                "grammar-aware candidate specs, raw/inflate controls, and exact eval."
            ),
        },
    }
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-jsonl", type=Path, default=Path(".omx/state/master_gradient_anchors.jsonl"))
    parser.add_argument("--anchor-json", type=Path)
    parser.add_argument("--measurement-axis", default="[contest-CUDA]")
    parser.add_argument("--archive-sha256", default="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf")
    parser.add_argument("--gradient-sha256")
    parser.add_argument("--k-values", default="32,64,128,256,512,1024")
    parser.add_argument("--top-axis-k", type=int, default=32)
    parser.add_argument("--top-record-limit", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    first = plan["topk_summaries"][0]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gradient_array_sha256": plan["anchor"]["gradient_array_sha256"],
                "measurement_axis": plan["anchor"]["measurement_axis"],
                "dS_d_byte": plan["dS_d_byte"],
                "first_k": first["k_top"],
                "first_top5": first["top_byte_indices"][:5],
                "first_selected_axis_share": first["selected_axis_share_within_topk"],
                "catalog356_validated": first["catalog356_axis_decomposition_validation"]["validated"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
