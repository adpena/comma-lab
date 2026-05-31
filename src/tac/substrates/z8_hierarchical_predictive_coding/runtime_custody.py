# SPDX-License-Identifier: MIT
"""Runtime-custody contract for Z8 predictive-stack archive exports."""

from __future__ import annotations

from typing import Literal

Z8_HPC_ARCHIVE_PARSER_PIXEL_CONSUMED_SECTIONS: tuple[str, ...] = (
    "decoder_blob",
    "indices_blob",
    "wavelet_blob",
    "wyner_ziv_blob",
    "dreamer_state_blob",
)
Z8_HPC_ARCHIVE_CANDIDATE_PIXEL_CONSUMED_SECTIONS: tuple[str, ...] = (
    "decoder_blob",
    "indices_blob",
    "wavelet_coeffs_blob",
    "wyner_ziv_blob",
    "dreamer_state_blob",
)
Z8_HPC_STACK_SEMANTIC_PIXEL_CONSUMED_SECTIONS: tuple[str, ...] = (
    "decoder_blob",
    "indices_blob",
    "dreamer_state_blob",
)
Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS: tuple[str, ...] = ()
Z8_HPC_RUNTIME_CUSTODY_CONTRACT_SCHEMA = "z8_hpc1_runtime_custody_contract.v1"
Z8_HPC_TRAINED_MLX_EXPORT_BLOCKER = (
    "trained_mlx_renderer_state_not_serialized_into_z8hpc1_runtime_archive"
)
Z8_HPC_TRAINED_STACK_BLOCKERS: tuple[str, ...] = (
    "decoder_indices_dreamer_sections_pixel_consumed_by_deterministic_stack_context_projection_not_trained_export",
)


def build_z8_runtime_custody_contract(
    *,
    source: str,
    section_name_style: Literal["archive_parser", "candidate_manifest"],
    archive_bound_candidate_package_emitted: bool,
    trained_mlx_renderer_archive_export_ready: bool,
) -> dict[str, object]:
    """Return the fail-closed Z8 runtime-custody contract for emitted artifacts."""

    if section_name_style == "archive_parser":
        pixel_consumed_sections = Z8_HPC_ARCHIVE_PARSER_PIXEL_CONSUMED_SECTIONS
    else:
        pixel_consumed_sections = Z8_HPC_ARCHIVE_CANDIDATE_PIXEL_CONSUMED_SECTIONS

    blockers = list(Z8_HPC_TRAINED_STACK_BLOCKERS)
    if not trained_mlx_renderer_archive_export_ready:
        blockers.append(Z8_HPC_TRAINED_MLX_EXPORT_BLOCKER)
    if not archive_bound_candidate_package_emitted:
        blockers.append("archive_bound_candidate_package_not_emitted")

    return {
        "schema": Z8_HPC_RUNTIME_CUSTODY_CONTRACT_SCHEMA,
        "source": source,
        "section_name_style": section_name_style,
        "pixel_consumed_archive_sections": list(pixel_consumed_sections),
        "stack_semantic_pixel_consumed_sections": list(
            Z8_HPC_STACK_SEMANTIC_PIXEL_CONSUMED_SECTIONS
        ),
        "stack_custody_not_yet_pixel_consumed_sections": list(
            Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS
        ),
        "full_stack_pixel_consumption_claim": False,
        "archive_bound_candidate_package_emitted": bool(
            archive_bound_candidate_package_emitted
        ),
        "trained_mlx_renderer_archive_export_ready": bool(
            trained_mlx_renderer_archive_export_ready
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        "next_required_tasks": [
            "serialize_trained_mlx_renderer_state_into_z8hpc1_archive",
            "prove_decoder_indices_dreamer_runtime_pixel_consumption",
            "replay_archive_bound_candidate_on_contest_cpu_cuda_exact_axis",
        ],
    }


__all__ = [
    "Z8_HPC_ARCHIVE_CANDIDATE_PIXEL_CONSUMED_SECTIONS",
    "Z8_HPC_ARCHIVE_PARSER_PIXEL_CONSUMED_SECTIONS",
    "Z8_HPC_RUNTIME_CUSTODY_CONTRACT_SCHEMA",
    "Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS",
    "Z8_HPC_STACK_SEMANTIC_PIXEL_CONSUMED_SECTIONS",
    "Z8_HPC_TRAINED_MLX_EXPORT_BLOCKER",
    "Z8_HPC_TRAINED_STACK_BLOCKERS",
    "build_z8_runtime_custody_contract",
]
