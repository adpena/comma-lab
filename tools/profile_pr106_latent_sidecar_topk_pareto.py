#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile PR106 latent sidecar top-k selector Pareto points.

This is a planning-only tool. It reads an existing PR106 latent score table,
selects the top-k measured per-pair corrections under the canonical reducer,
profiles their sidecar byte cost through PacketIR grammars, and emits a
non-promotable report. It never builds an archive and never claims score
movement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr106_latent_sidecar_selection import (  # noqa: E402
    build_latent_candidate_grid,
    profile_latent_sidecar_topk_pareto,
    validate_score_table_manifest,
)
from tac.packet_compiler.pr106_sidecar_packet import sha256_hex  # noqa: E402

TOOL = "tools/profile_pr106_latent_sidecar_topk_pareto.py"
SCHEMA = "pr106_latent_sidecar_topk_pareto_report_v1"


def _parse_top_k_values(text: str) -> list[int]:
    values: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 0:
            raise argparse.ArgumentTypeError(f"top-k value must be non-negative: {value}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one top-k value is required")
    return values


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    score_table = np.load(args.score_table_npy, allow_pickle=False)
    if not isinstance(score_table, np.ndarray):
        raise TypeError(
            f"--score-table-npy must load a numpy ndarray, got {type(score_table).__name__}"
        )
    candidates = build_latent_candidate_grid(
        latent_dim=args.latent_dim,
        delta_radius=args.delta_radius,
    )
    if score_table.shape != (args.n_pairs, len(candidates)):
        raise ValueError(
            "score table shape mismatch: expected "
            f"({args.n_pairs}, {len(candidates)}), got {score_table.shape}"
        )

    manifest: dict[str, object] | None = None
    if args.score_table_manifest is not None:
        if args.source_archive is None:
            raise ValueError("--source-archive is required with --score-table-manifest")
        manifest = validate_score_table_manifest(
            args.score_table_manifest,
            score_table_npy=args.score_table_npy,
            source_archive=args.source_archive,
            n_pairs=args.n_pairs,
            latent_dim=args.latent_dim,
            delta_radius=args.delta_radius,
            candidate_count=len(candidates),
        )

    profile = profile_latent_sidecar_topk_pareto(
        score_table,
        candidates,
        top_k_values=args.top_k_values,
        require_improvement=True,
    )
    source: dict[str, Any] = {
        "score_table_npy_path": str(args.score_table_npy),
        "score_table_npy_bytes": int(args.score_table_npy.stat().st_size),
        "score_table_npy_sha256": sha256_hex(args.score_table_npy.read_bytes()),
        "score_table_manifest_path": str(args.score_table_manifest)
        if args.score_table_manifest is not None
        else None,
        "score_table_manifest_sha256": sha256_hex(args.score_table_manifest.read_bytes())
        if args.score_table_manifest is not None
        else None,
        "score_table_manifest_validated": manifest is not None,
        "source_archive": str(args.source_archive) if args.source_archive is not None else None,
        "source_archive_sha256": sha256_hex(args.source_archive.read_bytes())
        if args.source_archive is not None
        else None,
        "n_pairs": int(args.n_pairs),
        "latent_dim": int(args.latent_dim),
        "delta_radius": int(args.delta_radius),
    }
    if manifest is not None:
        source["validated_source_archive_sha256_match"] = manifest.get(
            "validated_source_archive_sha256_match"
        )
        source["validated_source_zero_bin_sha256_match"] = manifest.get(
            "validated_source_zero_bin_sha256_match"
        )

    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "created_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "source": source,
        "profile": profile,
    }


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source"]
    profile = report["profile"]
    lines = [
        "# PR106 Latent Sidecar Top-k Pareto Profile",
        "",
        f"- score_claim: `{str(report['score_claim']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(report['ready_for_exact_eval_dispatch']).lower()}`",
        f"- score_table_npy: `{source['score_table_npy_path']}`",
        f"- score_table_npy_sha256: `{source['score_table_npy_sha256']}`",
        f"- score_table_shape: `{profile['score_table_shape']}`",
        f"- strict_improvement_pair_count: `{profile['strict_improvement_pair_count']}`",
        f"- best_raw_improvement_sum: `{profile['best_raw_improvement_sum']}`",
        "",
        "## Pareto Frontier",
        "",
        (
            "| top_k | corrections | selector improvement sum | "
            "runtime sidecar bytes | rate delta vs top_k=0 |"
        ),
        "|---:|---:|---:|---:|---:|",
    ]
    for row in profile["pareto_frontier"]:
        lines.append(
            f"| {row['top_k_cap']} | {row['n_corrections']} | "
            f"{row['selector_improvement_sum']:.12g} | "
            f"{row['best_runtime_consumed_charged_bytes']} | "
            f"{row['rate_score_delta_vs_topk0_sidecar_if_runtime_consumed']:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Evaluated Rows",
            "",
            (
                "| top_k | corrections | retained vs full | runtime sidecar bytes | "
                "best runtime grammar |"
            ),
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in profile["rows"]:
        best = row["best_runtime_consumed_sidecar"] or {}
        retained = row["selector_improvement_retained_fraction_vs_full"]
        retained_text = "n/a" if retained is None else f"{retained:.6g}"
        lines.append(
            f"| {row['top_k_cap']} | {row['n_corrections']} | {retained_text} | "
            f"{row['best_runtime_consumed_charged_bytes']} | "
            f"`{best.get('name')}` |"
        )
    claim = profile["adversarial_claim_check"]
    lines.extend(
        [
            "",
            "## Adversarial Claim Check",
            "",
            f"- verdict: `{claim['verdict']}`",
            "",
            claim["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table-npy", type=Path, required=True)
    parser.add_argument("--score-table-manifest", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument("--delta-radius", type=int, default=1)
    parser.add_argument(
        "--top-k-values",
        type=_parse_top_k_values,
        default=_parse_top_k_values("0,1,2,4,8,16,32,64,96,128,192,256,384,512,600"),
        help="Comma-separated non-negative top-k caps. 0 and full-positive are always added.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.json_out}")
    if args.md_out is not None:
        print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
