#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority SNeRV LF/HF receiver-runtime binding proof."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_lf_hf_runtime_binding import (  # noqa: E402
    SCHEMA,
    build_snerv_lf_hf_runtime_binding_proof,
    load_json_with_source_identity,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-residual-receiver-payload-proof",
        action="append",
        default=[],
        type=Path,
        help="snerv_lf_conditioned_hf_residual_receiver_proof.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--joint-codebook-receiver-payload-proof",
        action="append",
        default=[],
        type=Path,
        help="snerv_joint_lf_hf_factorized_codebook_receiver_proof.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--temporal-lf-predictor-receiver-payload-proof",
        action="append",
        default=[],
        type=Path,
        help="snerv_temporal_lf_predictor_receiver_proof.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--lf-super-resolution-receiver-payload-proof",
        action="append",
        default=[],
        type=Path,
        help="snerv_lf_super_resolution_tiny_anchor_receiver_proof.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--spectral-band-allocator-receiver-payload-proof",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_score_tethered_spectral_band_allocator_receiver_proof.v1 "
            "JSON. Repeatable."
        ),
    )
    parser.add_argument(
        "--lf-latent-hyperprior-receiver-payload-proof",
        action="append",
        default=[],
        type=Path,
        help="snerv_lf_latent_hyperprior_receiver_proof.v1 JSON. Repeatable.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_root
        or Path("/Volumes/VertigoDataTier/pact")
        / f"snerv_lf_hf_runtime_binding_{stamp}"
    )
    output_json = args.output_json or output_root / "snerv_lf_hf_runtime_binding_proof.json"
    proof_paths = [
        *args.hf_residual_receiver_payload_proof,
        *args.joint_codebook_receiver_payload_proof,
        *args.temporal_lf_predictor_receiver_payload_proof,
        *args.lf_super_resolution_receiver_payload_proof,
        *args.spectral_band_allocator_receiver_payload_proof,
        *args.lf_latent_hyperprior_receiver_payload_proof,
    ]
    report = build_snerv_lf_hf_runtime_binding_proof(
        [load_json_with_source_identity(path) for path in proof_paths]
    )
    result = write_json_artifact(output_json, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_json": result.path,
                "output_json_sha256": result.sha256,
                "runtime_binding_row_count": report["runtime_binding_row_count"],
                "closed_campaign_blockers": report["closed_campaign_blockers"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
