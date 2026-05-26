# SPDX-License-Identifier: MIT
"""Repair-dynamics palette probe matrices for grouped component correction.

This module converts a repair-dynamics palette prior into bounded, executable
local probe plans. It does not run scorers or mutate archives. The output is a
false-authority planning artifact consumed by experiment_queue steps.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    ordered_unique,
    require_no_truthy_authority_fields,
)

REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA = (
    "frontier_rate_attack_repair_dynamics_palette_probe_matrix.v1"
)

FALSE_AUTHORITY: dict[str, bool] = {
    **PROXY_FALSE_AUTHORITY_FIELDS,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
}


class RepairDynamicsPaletteProbeError(ValueError):
    """Raised when a repair-dynamics palette cannot become a probe matrix."""


def _frame_prefix(frame_index: int) -> str | None:
    if frame_index == 0:
        return "even"
    if frame_index == 1:
        return "odd"
    return None


def _signed_token_value(token: str) -> int:
    text = str(token).strip()
    if not text:
        raise RepairDynamicsPaletteProbeError("empty signed token")
    if text[0] == "m":
        return -int(text[1:])
    if text[0] == "p":
        return int(text[1:])
    return int(text)


def _format_number(value: float | int) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def repair_mode_id_to_postfilter_mode(mode_id: str) -> dict[str, Any]:
    """Map a canonical repair mode id to a postfilter sweep mode."""

    text = str(mode_id).strip()
    if text == "none":
        return {
            "mode_id": text,
            "postfilter_mode": "none",
            "operator_family": "identity",
            "frame_scope": "all",
            "supported": True,
            "blockers": [],
            **FALSE_AUTHORITY,
        }

    match = re.fullmatch(r"frame(\d+)_(.+)", text)
    if match is None:
        return {
            "mode_id": text,
            "postfilter_mode": None,
            "operator_family": "unknown",
            "frame_scope": "unknown",
            "supported": False,
            "blockers": ["unknown_repair_mode_id_grammar"],
            **FALSE_AUTHORITY,
        }
    frame_index = int(match.group(1))
    frame_prefix = _frame_prefix(frame_index)
    if frame_prefix is None:
        return {
            "mode_id": text,
            "postfilter_mode": None,
            "operator_family": "unsupported_frame_scope",
            "frame_scope": f"frame{frame_index}",
            "supported": False,
            "blockers": ["postfilter_adapter_supports_only_frame0_frame1"],
            **FALSE_AUTHORITY,
        }

    body = match.group(2)
    postfilter_mode: str | None = None
    operator_family = "unknown"
    if match := re.fullmatch(r"luma_bias_([+-]\d+)", body):
        operator_family = "luma_bias"
        postfilter_mode = f"{frame_prefix}_bias:{int(match.group(1))}"
    elif match := re.fullmatch(r"rgb_bias_([^_]+)_([^_]+)_([^_]+)", body):
        operator_family = "rgb_bias"
        values = [_signed_token_value(part) for part in match.groups()]
        postfilter_mode = (
            f"{frame_prefix}_rgb_bias:"
            f"{_format_number(values[0])},{_format_number(values[1])},"
            f"{_format_number(values[2])}"
        )
    elif match := re.fullmatch(r"blue_chroma_amp_([+-]?\d+(?:\.\d+)?)", body):
        operator_family = "blue_chroma"
        postfilter_mode = f"{frame_prefix}_grain_chroma:{_format_number(float(match.group(1)))}"
    elif match := re.fullmatch(r"roll_dx([+-]\d+)_dy([+-]\d+)", body):
        operator_family = "geometry_roll"
        dx = int(match.group(1))
        dy = int(match.group(2))
        postfilter_mode = f"{frame_prefix}_translate:{dy},{dx}"

    return {
        "mode_id": text,
        "postfilter_mode": postfilter_mode,
        "operator_family": operator_family,
        "frame_scope": f"frame{frame_index}",
        "supported": postfilter_mode is not None,
        "blockers": [] if postfilter_mode is not None else ["unsupported_repair_mode_id"],
        **FALSE_AUTHORITY,
    }


def _counterfactual_frame1_modes(modes: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    for mode in modes:
        if mode.startswith("even_"):
            out.append("odd_" + mode[len("even_") :])
        if len(out) >= limit:
            break
    return ordered_unique(out)


def _family_expansion_modes(family_counts: Mapping[str, Any]) -> list[str]:
    modes: list[str] = []
    if int(family_counts.get("luma_bias", 0) or 0):
        modes.extend(f"even_bias:{value}" for value in (-8, -4, -2, -1, 1, 2, 4, 8))
    if int(family_counts.get("rgb_bias", 0) or 0):
        modes.extend(
            [
                "even_rgb_bias:1,-0.5,-0.5",
                "even_rgb_bias:-1,0.5,0.5",
                "even_rgb_bias:2,-1,-1",
                "even_rgb_bias:-2,1,1",
                "even_rgb_bias:4,-2,-2",
                "even_rgb_bias:-4,2,2",
            ]
        )
    if int(family_counts.get("blue_chroma", 0) or 0):
        modes.extend(
            [
                "even_grain_chroma:0.5",
                "even_grain_chroma:1",
                "even_grain_chroma:2",
                "even_grain_chroma:3",
            ]
        )
    if int(family_counts.get("geometry_roll", 0) or 0):
        modes.extend(
            [
                "even_translate:0,1",
                "even_translate:0,-1",
                "even_translate:1,0",
                "even_translate:-1,0",
            ]
        )
    return ordered_unique(modes)


def _interaction_modes(modes: list[str], *, limit: int) -> list[str]:
    base = [mode for mode in modes if mode != "none" and "+" not in mode][:10]
    out: list[str] = []
    for left, right in itertools.combinations(base, 2):
        if left == right:
            continue
        out.append(f"{left}+{right}")
        if len(out) >= limit:
            break
    return ordered_unique(out)


def _mode_args(modes: list[str]) -> list[str]:
    args: list[str] = []
    for mode in modes:
        args.extend(["--mode", mode])
    return args


def build_repair_dynamics_palette_probe_matrix(
    *,
    work_order: Mapping[str, Any],
    work_order_path: str | Path | None = None,
    probe_output_dir: str | Path | None = None,
    device: str = "mlx",
    n_pairs: int = 48,
    max_modes: int = 96,
    max_interaction_modes: int = 16,
    max_counterfactual_modes: int = 16,
) -> dict[str, Any]:
    """Build a bounded local probe matrix from a repair-dynamics work order."""

    if device not in {"cpu", "cuda", "mps", "mlx", "gpu"}:
        raise RepairDynamicsPaletteProbeError("device must be cpu, cuda, mps, mlx, or gpu")
    if n_pairs < 1:
        raise RepairDynamicsPaletteProbeError("n_pairs must be >= 1")
    if max_modes < 1:
        raise RepairDynamicsPaletteProbeError("max_modes must be >= 1")
    require_no_truthy_authority_fields(
        work_order,
        context="repair_dynamics_palette_probe_work_order",
    )
    prior = work_order.get("repair_dynamics_palette_prior")
    if not isinstance(prior, Mapping) or not prior:
        raise RepairDynamicsPaletteProbeError(
            "work order is missing repair_dynamics_palette_prior"
        )
    require_no_truthy_authority_fields(
        prior,
        context="repair_dynamics_palette_probe_prior",
    )
    palette_modes = [str(mode) for mode in prior.get("palette_modes") or [] if str(mode)]
    if not palette_modes:
        raise RepairDynamicsPaletteProbeError("repair palette prior has no modes")

    mapping_rows = [repair_mode_id_to_postfilter_mode(mode) for mode in palette_modes]
    supported_modes = ordered_unique(
        row["postfilter_mode"]
        for row in mapping_rows
        if row.get("supported") is True and isinstance(row.get("postfilter_mode"), str)
    )
    family_counts = (
        prior.get("mode_family_counts")
        if isinstance(prior.get("mode_family_counts"), Mapping)
        else {}
    )
    groups: list[dict[str, Any]] = [
        {
            "group_id": "canonical_palette_modes",
            "purpose": "measure_prior_modes_as_grouped_component_response_atoms",
            "frame_scope": "prior_declared",
            "modes": supported_modes,
            **FALSE_AUTHORITY,
        }
    ]
    expansion = _family_expansion_modes(family_counts)
    if expansion:
        groups.append(
            {
                "group_id": "family_local_expansion_modes",
                "purpose": "probe_neighboring_color_geometry_basis_before_pixel_leaf_search",
                "frame_scope": "frame0",
                "modes": expansion,
                **FALSE_AUTHORITY,
            }
        )
    if prior.get("zero_frame1_modes") is True:
        counterfactual = _counterfactual_frame1_modes(
            supported_modes,
            limit=max_counterfactual_modes,
        )
        if counterfactual:
            groups.append(
                {
                    "group_id": "frame1_counterfactual_null_probe_modes",
                    "purpose": "classify_frame1_null_space_vs_missing_search_gap",
                    "frame_scope": "frame1",
                    "modes": counterfactual,
                    **FALSE_AUTHORITY,
                }
            )
    interactions = _interaction_modes(
        ordered_unique([*supported_modes, *expansion]),
        limit=max_interaction_modes,
    )
    if interactions:
        groups.append(
            {
                "group_id": "pairwise_palette_interaction_modes",
                "purpose": "measure_synergy_antagonism_between_palette_operators",
                "frame_scope": "frame0",
                "modes": interactions,
                **FALSE_AUTHORITY,
            }
        )

    postfilter_modes = ordered_unique(
        ["none", *(mode for group in groups for mode in group.get("modes", []))]
    )
    truncated = len(postfilter_modes) > max_modes
    postfilter_modes = postfilter_modes[:max_modes]
    output_dir = Path(probe_output_dir or "<component_correction_dir>/repair_dynamics")
    cpu_sweep_path = output_dir / "repair_dynamics_palette_cpu_sweep.json"
    mlx_response_path = output_dir / "repair_dynamics_mlx_response.json"
    mlx_components_dir = output_dir / "repair_dynamics_mlx_components"
    archive_path = str(work_order.get("archive_path") or "<receiver_closed_archive.zip>")
    unsupported_rows = [row for row in mapping_rows if row.get("supported") is not True]
    if device in {"mlx", "gpu"}:
        probe_command = {
            "action_id": "run_mlx_palette_component_response",
            "command": [
                ".venv/bin/python",
                "tools/run_mlx_scorer_response_from_local_advisory.py",
                "--local-cpu-advisory",
                "<component_correction_dir>/local_cpu_advisory.json",
                "--reference-cache-dir",
                "<reference_mlx_scorer_input_cache_dir>",
                "--candidate-cache-dir",
                "<component_correction_dir>/mlx_scorer_input_cache",
                "--output",
                mlx_response_path.as_posix(),
                "--repo-root",
                ".",
                "--batch-pairs",
                "1",
                "--device",
                "gpu",
                "--allow-gpu-research-signal",
                "--allow-local-cpu-advisory-cache-identity",
                "--components-dir",
                mlx_components_dir.as_posix(),
                "--response-family",
                "repair_dynamics_palette_probe",
            ],
            "resources": {"kind": "local_mlx"},
            "blocked_until": [
                "local_cpu_advisory_cache_identity",
                "mlx_scorer_input_cache",
                "repair_dynamics_palette_probe_matrix",
            ],
            "output_schema": "mlx_scorer_response_payload.v1",
            **FALSE_AUTHORITY,
        }
    else:
        probe_command = {
            "action_id": "run_cpu_postfilter_palette_probe",
            "command": [
                ".venv/bin/python",
                "tools/screen_hdm8_postfilter_sweep.py",
                "--archive",
                archive_path,
                "--output-json",
                cpu_sweep_path.as_posix(),
                "--device",
                device,
                "--n-pairs",
                str(n_pairs),
                "--include-per-pair",
                *_mode_args(postfilter_modes),
            ],
            "resources": {"kind": "local_cpu" if device == "cpu" else "local_gpu_proxy"},
            "blocked_until": ["receiver_closed_archive", "source_runtime_available"],
            "output_schema": "hdm8_postfilter_sweep_v1",
            **FALSE_AUTHORITY,
        }
    commands = [
        probe_command,
        {
            "action_id": "harvest_palette_probe_into_component_response",
            "command": [
                ".venv/bin/python",
                "tools/harvest_frontier_targeted_component_correction_response.py",
                "--work-order",
                str(work_order_path or "<repair_dynamics_work_order.json>"),
                "--local-cpu-advisory",
                "<component_correction_dir>/local_cpu_advisory.json",
                "--output",
                "<component_correction_dir>/repair_dynamics_response_harvest.json",
                "--repo-root",
                ".",
            ],
            "blocked_until": [
                "same_axis_candidate_reference_component_response",
                "local_cpu_advisory_or_exact_axis_component_response",
            ],
            **FALSE_AUTHORITY,
        },
    ]
    return {
        "schema": REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA,
        "work_order_schema": work_order.get("schema"),
        "work_order_path": str(work_order_path) if work_order_path is not None else None,
        "acquisition_id": work_order.get("acquisition_id"),
        "candidate_id": work_order.get("candidate_id"),
        "correction_family": work_order.get("correction_family"),
        "repair_dynamics_palette_prior": dict(prior),
        "palette_mode_count": len(palette_modes),
        "supported_mode_count": len(supported_modes),
        "unsupported_mode_count": len(unsupported_rows),
        "unsupported_mapping_rows": unsupported_rows,
        "probe_group_count": len(groups),
        "probe_groups": groups,
        "postfilter_mode_count": len(postfilter_modes),
        "postfilter_modes": postfilter_modes,
        "postfilter_modes_truncated": truncated,
        "device": device,
        "n_pairs": n_pairs,
        "max_modes": max_modes,
        "commands": commands,
        "ready_for_budget_spend": False,
        "budget_spend_allowed": False,
        "allowed_use": "local_grouped_repair_dynamics_probe_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


__all__ = [
    "REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA",
    "RepairDynamicsPaletteProbeError",
    "build_repair_dynamics_palette_probe_matrix",
    "repair_mode_id_to_postfilter_mode",
]
