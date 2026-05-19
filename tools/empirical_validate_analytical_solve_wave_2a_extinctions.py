#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Empirical validation smoke for Wave 2A analytical-solve extinctions.

Per the Wave 2A landing prompt + CLAUDE.md "Max observability" + Catalog #192/#317:
- Run each of the 10 helpers against representative operating-point anchors
- Persist results via tac.optimization.macos_cpu_advisory_signal canonical helper
- Wrap each row via canonical Provenance per Catalog #323
- Evidence grade: ``macOS-CPU-advisory``; score_claim=False; promotion_eligible=False
- All artifacts under ``experiments/results/empirical_validate_analytical_solve_wave_2a_20260518/``

Lane: ``lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518``

Usage::

    .venv/bin/python tools/empirical_validate_analytical_solve_wave_2a_extinctions.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.analytical_solve_extinctions import (  # noqa: E402
    BatchSizeSolverInput,
    BlockFPSolverInput,
    CouplingThresholdInput,
    FrameOrderingInput,
    PairOrderingInput,
    RashomonKInput,
    RDCodebookSolverInput,
    ROCThresholdInput,
    SGLDTFinalInput,
    solve_bootstrap_ci_rashomon_K,
    solve_coupling_threshold_statistical,
    solve_greedy_tsp_per_pair_ordering,
    solve_min_spanning_tree_frame_ordering,
    solve_optimal_block_fp_block_size,
    solve_rd_theoretic_vq_codebook_K,
    solve_roc_optimal_high_pair_invariant_threshold,
    solve_sgld_t_final_welling_teh,
    solve_vram_aware_batch_size,
)
from tac.optimization.macos_cpu_advisory_signal import (  # noqa: E402
    append_manifest_row_to_jsonl,
)


OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "empirical_validate_analytical_solve_wave_2a_20260518"
MANIFEST_PATH = OUTPUT_DIR / "wave_2a_advisory_manifest.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "wave_2a_validation_summary.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_advisory_row(
    *,
    value_id: str,
    solved_value: Any,
    intermediate_summary: dict[str, Any],
    helper_invocation: str,
    literature_citation: str,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Build a canonical macOS-CPU-advisory manifest row per Catalog #192/#317/#323."""
    return {
        # Catalog #192 fail-closed contract: score_claim=False, promotion_eligible=False
        "value_id": value_id,
        "solved_value": solved_value if isinstance(solved_value, (int, float, str, list, dict)) else str(solved_value),
        "intermediate_summary": intermediate_summary,
        "helper_invocation": helper_invocation,
        "literature_citation": literature_citation,
        "wall_clock_seconds": wall_clock_seconds,
        "captured_at_utc": _utc_now(),
        "captured_by_subagent": (
            "lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518"
        ),
        # Canonical Provenance per Catalog #323
        "provenance": {
            "artifact_kind": "predicted_from_model",
            "source_path": "<predictor:analytical_solve_extinctions.wave_2a.v1>",
            "source_sha256": "0" * 64,
            "measurement_axis": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "promotion_eligible": False,
            "score_claim_valid": False,
            "captured_at_utc": _utc_now(),
            "canonical_helper_invocation": "tac.provenance.builders.build_provenance_for_macos_cpu_advisory",
            "contest_archive_zip_path": "",
            "contest_archive_member_name": "",
            "composed_from": [],
            "rejection_reason": "",
        },
        # Catalog #127 custody routing fields (defense-in-depth)
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "macOS-CPU-advisory",
        "evidence_tag": "[macOS-CPU advisory]",
    }


def validate_row_1_vram_aware_batch_size() -> dict[str, Any]:
    """Modal T4 substrate at typical operating point."""
    start = time.perf_counter()
    res = solve_vram_aware_batch_size(
        BatchSizeSolverInput(
            vram_budget_gb=14.5, model_size_gb=1.2, activation_overhead_gb=0.5,
            per_sample_activation_gb=0.4, base_lr_at_reference_batch=1e-4,
            reference_batch_size=4,
        ),
    )
    return _build_advisory_row(
        value_id="batch_size_wildly_varies_1_4_8_16_32_per_substrate",
        solved_value=res.solved_value,
        intermediate_summary={
            "vram_remaining_gb": res.intermediate_values["vram_remaining_gb"],
            "linear_scaled_lr": res.coupled_adjustments["linear_scaled_lr"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_2_neural_weight_codec_K_64() -> dict[str, Any]:
    """Replace hardcoded K=64 in neural_weight_codec."""
    start = time.perf_counter()
    res = solve_rd_theoretic_vq_codebook_K(
        RDCodebookSolverInput(
            num_codewords_used=100_000, codeword_dim=4, bytes_per_codeword=4,
        ),
    )
    return _build_advisory_row(
        value_id="vq_codebook_K_64_hardcoded_neural_weight_codec",
        solved_value=res.solved_value,
        intermediate_summary={
            "K_grid_size": res.intermediate_values["K_grid_size"],
            "best_cost_bits": res.intermediate_values["best_cost_bits"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_3_vqvae_mask_K_256() -> dict[str, Any]:
    """Replace hardcoded K=256 in codec_pipeline_mask."""
    start = time.perf_counter()
    res = solve_rd_theoretic_vq_codebook_K(
        RDCodebookSolverInput(
            num_codewords_used=10_000, codeword_dim=64, bytes_per_codeword=8,
        ),
    )
    return _build_advisory_row(
        value_id="vqvae_mask_codebook_K_256_hardcoded",
        solved_value=res.solved_value,
        intermediate_summary={
            "K_grid_size": res.intermediate_values["K_grid_size"],
            "best_cost_bits": res.intermediate_values["best_cost_bits"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_4_block_fp_block_size() -> dict[str, Any]:
    """Replace undeclared block_size default in balle_hyperprior_codec."""
    start = time.perf_counter()
    res = solve_optimal_block_fp_block_size(
        BlockFPSolverInput(
            num_elements_N=10_000, header_bytes_per_block=4, quant_loss_constant=1.0,
        ),
    )
    return _build_advisory_row(
        value_id="block_fp_block_size_undeclared_default",
        solved_value=res.solved_value,
        intermediate_summary={
            "raw_continuous_optimum": res.intermediate_values["raw_continuous_optimum"],
            "num_blocks": res.intermediate_values["num_blocks"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_5_mst_frame_ordering() -> dict[str, Any]:
    """MST ordering for 8 representative frames."""
    start = time.perf_counter()
    # Synthetic dissimilarity matrix: chain pattern with some noise
    dissim = [
        [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
        [0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
        [0.3, 0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.8],
        [0.5, 0.3, 0.1, 0.0, 0.1, 0.3, 0.5, 0.7],
        [0.7, 0.5, 0.3, 0.1, 0.0, 0.1, 0.3, 0.5],
        [0.8, 0.7, 0.5, 0.3, 0.1, 0.0, 0.1, 0.3],
        [0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0, 0.1],
        [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0],
    ]
    res = solve_min_spanning_tree_frame_ordering(
        FrameOrderingInput(num_frames=8, pairwise_dissimilarity=dissim, anchor_frame_id=0),
    )
    return _build_advisory_row(
        value_id="inflate_per_frame_decode_priority_implicit",
        solved_value=res.solved_value,
        intermediate_summary={
            "mst_total_weight": res.intermediate_values["mst_total_weight"],
            "consecutive_pair_sum": res.intermediate_values["consecutive_pair_sum"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_6_roc_optimal_high_pair_invariant_threshold() -> dict[str, Any]:
    """ROC-optimal threshold on synthetic labeled corpus."""
    start = time.perf_counter()
    # Synthetic separable corpus: 6 positives in [0.7, 1.0]; 6 negatives in [0.0, 0.4]
    examples = [
        (0.05, False), (0.10, False), (0.15, False),
        (0.25, False), (0.35, False), (0.40, False),
        (0.70, True), (0.75, True), (0.80, True),
        (0.85, True), (0.95, True), (1.00, True),
    ]
    res = solve_roc_optimal_high_pair_invariant_threshold(
        ROCThresholdInput(labeled_examples=examples),
    )
    return _build_advisory_row(
        value_id="HIGH_PAIR_INVARIANT_threshold_Catalog_319",
        solved_value=res.solved_value,
        intermediate_summary={
            "best_tpr": res.intermediate_values["best_tpr"],
            "best_fpr": res.intermediate_values["best_fpr"],
            "youdens_j": res.intermediate_values["youdens_j"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_7_coupling_threshold_statistical() -> dict[str, Any]:
    """Coupling threshold on 4-pair gradient inner-product matrix."""
    start = time.perf_counter()
    matrix = [
        [1.0, 0.10, 0.20, 0.45],
        [0.10, 1.0, 0.15, 0.30],
        [0.20, 0.15, 1.0, 0.55],
        [0.45, 0.30, 0.55, 1.0],
    ]
    res = solve_coupling_threshold_statistical(
        CouplingThresholdInput(pairwise_inner_products=matrix),
    )
    return _build_advisory_row(
        value_id="coupling_threshold_0.5_master_gradient_consumers",
        solved_value=res.solved_value,
        intermediate_summary={
            "mean": res.intermediate_values["mean"],
            "std": res.intermediate_values["std"],
            "tail_fraction_at_threshold": res.intermediate_values["tail_fraction_at_threshold"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_8_sgld_t_final_welling_teh() -> dict[str, Any]:
    """SGLD t_final at canonical stack_of_stacks operating point."""
    start = time.perf_counter()
    res = solve_sgld_t_final_welling_teh(
        SGLDTFinalInput(
            variance_posterior_target=0.001,
            step_size_eta=0.01,
        ),
    )
    return _build_advisory_row(
        value_id="stack_of_stacks_langevin_t_final_1e-4",
        solved_value=res.solved_value,
        intermediate_summary={
            "raw_t_final_unclamped": res.intermediate_values["raw_t_final_unclamped"],
            "ratio_to_naive_1e_minus_4": res.coupled_adjustments["ratio_to_naive_1e_minus_4"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_9_rashomon_K() -> dict[str, Any]:
    """Rashomon K at canonical d=2, delta=0.05, epsilon=0.1."""
    start = time.perf_counter()
    res = solve_bootstrap_ci_rashomon_K(
        RashomonKInput(effective_dimensionality_d=2.0),
    )
    return _build_advisory_row(
        value_id="rashomon_ensemble_K_8_members_arbitrary",
        solved_value=res.solved_value,
        intermediate_summary={
            "raw_K_unclamped": res.intermediate_values["raw_K_unclamped"],
            "vs_naive_K_8_ratio": res.coupled_adjustments["vs_naive_K_8_ratio"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


def validate_row_10_greedy_tsp_per_pair_ordering() -> dict[str, Any]:
    """Greedy NN-TSP per-pair ordering on 8 representative pairs."""
    start = time.perf_counter()
    dissim = [
        [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
        [0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
        [0.3, 0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.8],
        [0.5, 0.3, 0.1, 0.0, 0.1, 0.3, 0.5, 0.7],
        [0.7, 0.5, 0.3, 0.1, 0.0, 0.1, 0.3, 0.5],
        [0.8, 0.7, 0.5, 0.3, 0.1, 0.0, 0.1, 0.3],
        [0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0, 0.1],
        [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0],
    ]
    res = solve_greedy_tsp_per_pair_ordering(
        PairOrderingInput(num_pairs=8, pairwise_dissimilarity=dissim, start_pair_id=0),
    )
    return _build_advisory_row(
        value_id="per_pair_file_list_ordering_sequential",
        solved_value=res.solved_value,
        intermediate_summary={
            "total_path_dissimilarity": res.intermediate_values["total_path_dissimilarity"],
            "sequential_baseline_dissimilarity": res.intermediate_values["sequential_baseline_dissimilarity"],
            "relative_improvement_fraction": res.intermediate_values["relative_improvement_fraction"],
        },
        helper_invocation=res.canonical_helper_invocation,
        literature_citation=res.literature_citation,
        wall_clock_seconds=time.perf_counter() - start,
    )


VALIDATORS = [
    validate_row_1_vram_aware_batch_size,
    validate_row_2_neural_weight_codec_K_64,
    validate_row_3_vqvae_mask_K_256,
    validate_row_4_block_fp_block_size,
    validate_row_5_mst_frame_ordering,
    validate_row_6_roc_optimal_high_pair_invariant_threshold,
    validate_row_7_coupling_threshold_statistical,
    validate_row_8_sgld_t_final_welling_teh,
    validate_row_9_rashomon_K,
    validate_row_10_greedy_tsp_per_pair_ordering,
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Truncate manifest fresh per session to avoid duplicate row accretion
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    print(f"[wave-2a-validate] writing manifest to {MANIFEST_PATH}")
    rows: list[dict[str, Any]] = []
    for i, validator in enumerate(VALIDATORS, start=1):
        try:
            row = validator()
            append_manifest_row_to_jsonl(row, output_path=MANIFEST_PATH)
            rows.append(row)
            print(
                f"[wave-2a-validate] row {i:2d}/10  "
                f"value_id={row['value_id']!s:60s}  "
                f"solved={str(row['solved_value'])[:30]:30s}  "
                f"wall={row['wall_clock_seconds']*1000:.2f}ms"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[wave-2a-validate] row {i} FAILED: {exc!r}", file=sys.stderr)
            return 1

    summary = {
        "subagent_lane": "lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
        "captured_at_utc": _utc_now(),
        "num_rows_validated": len(rows),
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "value_ids": [row["value_id"] for row in rows],
        "total_wall_clock_seconds": sum(r["wall_clock_seconds"] for r in rows),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[wave-2a-validate] summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
