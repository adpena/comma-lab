# SPDX-License-Identifier: MIT
"""Campaign queues for grouped P19/P18/P11/P15 scorer-region cascades."""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA, normalize_queue_definition
from comma_lab.scheduler.scorer_region_selector_chain_queue import (
    DEFAULT_MLX_REFERENCE_CACHE_DIR,
    build_scorer_region_selector_chain_queue,
)
from tac.optimization.contest_space_action import (
    CONTEST_RATE_DENOM_BYTES,
    RATE_SCORE_PER_BYTE,
    build_contest_space_action_functional,
    build_hydration_contract,
    build_rate_distortion_action_row,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.scorer_region_operator_contract import (
    build_scorer_region_operator_contract,
)
from tac.repo_io import sha256_file
from tac.substrates.uniward_per_pixel_distortion.weight_map import (
    compute_per_pixel_uniward_weight_map_numpy,
)

SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA = (
    "scorer_region_selector_cascade_campaign_queue_metadata.v1"
)
SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA = (
    "scorer_region_selector_cascade_campaign_report.v1"
)
SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA = (
    "scorer_region_selector_cascade_acquisition_policy.v1"
)
SCORER_REGION_SELECTOR_CASCADE_SELECTION_MANIFEST_SCHEMA = (
    "scorer_region_selector_cascade_selection_manifest.v1"
)

SUPPORTED_REPACK_ORDERS = frozenset({"p11_then_p15_then_receiver_patch"})
DEFAULT_MASTER_GRADIENT_TENSOR_PATH = Path(
    ".omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy"
)
DEFAULT_PIXEL_GRADIENT_CACHE_PATH = Path(
    ".omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/"
    "real_scorer_gradients_cache.npz"
)
FULL600_MLX_OPERATING_POINT = {
    "d_seg": 0.0012223561610638473,
    "d_pose": 0.0017157510650319333,
    "rate": 0.004754685709380427,
    "score": 0.37208944003527994,
}


class ScorerRegionSelectorCascadeCampaignQueueError(ValueError):
    """Raised when a grouped cascade campaign cannot be built."""


@dataclass(frozen=True)
class CascadeVariant:
    """One grouped operator-set variant inside the campaign queue."""

    variant_id: str
    null_fraction: float
    top_regions_per_pair: int
    receiver_patch_max_pairs: int
    receiver_patch_regions_per_pair: int
    receiver_patch_rgb_delta: tuple[int, int, int]
    receiver_patch_delta_space: str
    selector_codec_families: tuple[str, ...]
    scales: tuple[int, ...]
    alphas: tuple[int, ...]
    repack_order: str

    def to_metadata(self, *, output_root: str) -> dict[str, Any]:
        payload = {
            "schema": "scorer_region_selector_cascade_variant.v1",
            "variant_id": self.variant_id,
            "output_root": output_root,
            "operator_set": {
                "p19_null_fraction": self.null_fraction,
                "p18_top_regions_per_pair": self.top_regions_per_pair,
                "receiver_patch_max_pairs": self.receiver_patch_max_pairs,
                "receiver_patch_regions_per_pair": self.receiver_patch_regions_per_pair,
                "receiver_patch_rgb_delta": list(self.receiver_patch_rgb_delta),
                "receiver_patch_delta_space": self.receiver_patch_delta_space,
                "selector_codec_families": list(self.selector_codec_families),
                "selector_scales": list(self.scales),
                "selector_alphas": list(self.alphas),
                "repack_order": self.repack_order,
            },
            "chain_position_order": ["P19", "P18", "P11", "P15"],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            payload,
            context=f"scorer_region_cascade_variant:{self.variant_id}",
        )
        return payload


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _safe_id(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return text or "variant"


def _archive_record(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise ScorerRegionSelectorCascadeCampaignQueueError(f"archive missing: {path}")
    return {
        "path": _repo_rel(resolved, repo_root),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _submission_archive(source_submission_dir: str | Path, *, repo_root: str | Path) -> Path:
    source = _resolve(source_submission_dir, repo_root)
    archive = source / "archive.zip"
    if not archive.is_file():
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            f"source submission archive missing: {archive}"
        )
    return archive


def _non_empty_codec_families(groups: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    for group in groups:
        normalized = tuple(str(item).strip() for item in group if str(item).strip())
        if normalized:
            out.append(normalized)
    if not out:
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            "at least one selector codec family group is required"
        )
    return tuple(out)


def _rgb_delta_records(
    rgb_deltas: Sequence[tuple[int, int, int]],
    yuv_deltas: Sequence[tuple[int, int, int]],
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    records: list[tuple[str, tuple[int, int, int]]] = []
    for delta in rgb_deltas:
        if len(delta) != 3:
            raise ScorerRegionSelectorCascadeCampaignQueueError(
                "receiver patch RGB deltas must contain exactly three values"
            )
        records.append(("rgb", tuple(int(v) for v in delta)))
    for y, u, v in yuv_deltas:
        # The current receiver patch edits RGB tensors. Keep YUV as an honest
        # acquisition family by applying the BT.601-equivalent RGB integer delta.
        rgb = (
            round(float(y) + 1.402 * float(v)),
            round(float(y) - 0.344136 * float(u) - 0.714136 * float(v)),
            round(float(y) + 1.772 * float(u)),
        )
        records.append(("yuv601_proxy_as_rgb", tuple(int(item) for item in rgb)))
    if not records:
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            "at least one RGB or YUV receiver patch delta is required"
        )
    return tuple(dict.fromkeys(records))


def enumerate_cascade_variants(
    *,
    null_fractions: Sequence[float],
    top_regions_per_pair_values: Sequence[int],
    receiver_patch_max_pair_values: Sequence[int],
    receiver_patch_regions_per_pair_values: Sequence[int],
    receiver_patch_rgb_deltas: Sequence[tuple[int, int, int]],
    receiver_patch_yuv_deltas: Sequence[tuple[int, int, int]] = (),
    selector_codec_family_groups: Sequence[Sequence[str]],
    scales: Sequence[int],
    alphas: Sequence[int],
    repack_orders: Sequence[str] = ("p11_then_p15_then_receiver_patch",),
    max_variants: int | None = 32,
) -> tuple[CascadeVariant, ...]:
    """Enumerate grouped P19/P18/P11/P15 operator-set variants deterministically."""

    if max_variants is not None and max_variants <= 0:
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            "max_variants must be positive or None"
        )
    codec_groups = _non_empty_codec_families(selector_codec_family_groups)
    delta_records = _rgb_delta_records(receiver_patch_rgb_deltas, receiver_patch_yuv_deltas)
    scale_values = tuple(int(item) for item in scales)
    alpha_values = tuple(int(item) for item in alphas)
    if not scale_values or not alpha_values:
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            "selector scales and alphas must be non-empty"
        )
    normalized_repack_orders = tuple(str(item).strip() for item in repack_orders if str(item).strip())
    unsupported = sorted(set(normalized_repack_orders) - SUPPORTED_REPACK_ORDERS)
    if unsupported:
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            "unsupported repack order(s): " + ", ".join(unsupported)
        )

    variants: list[CascadeVariant] = []
    seen_ids: set[str] = set()
    grid = itertools.product(
        [float(item) for item in null_fractions],
        [int(item) for item in top_regions_per_pair_values],
        [int(item) for item in receiver_patch_max_pair_values],
        [int(item) for item in receiver_patch_regions_per_pair_values],
        delta_records,
        codec_groups,
        normalized_repack_orders,
    )
    for (
        null_fraction,
        top_regions,
        max_pairs,
        regions_per_pair,
        (delta_space, rgb_delta),
        codec_group,
        repack_order,
    ) in grid:
        if null_fraction <= 0.0 or null_fraction > 1.0:
            raise ScorerRegionSelectorCascadeCampaignQueueError(
                f"null fraction outside (0,1]: {null_fraction}"
            )
        label = (
            f"nf{null_fraction:g}_r{top_regions}_p{max_pairs}_"
            f"rp{regions_per_pair}_{delta_space}_{rgb_delta[0]}_{rgb_delta[1]}_{rgb_delta[2]}_"
            f"cf{'_'.join(codec_group)}_{repack_order}"
        )
        variant_id = _safe_id(label)
        if variant_id in seen_ids:
            continue
        seen_ids.add(variant_id)
        variants.append(
            CascadeVariant(
                variant_id=variant_id,
                null_fraction=float(null_fraction),
                top_regions_per_pair=int(top_regions),
                receiver_patch_max_pairs=int(max_pairs),
                receiver_patch_regions_per_pair=int(regions_per_pair),
                receiver_patch_rgb_delta=rgb_delta,
                receiver_patch_delta_space=delta_space,
                selector_codec_families=tuple(codec_group),
                scales=scale_values,
                alphas=alpha_values,
                repack_order=repack_order,
            )
        )
        if max_variants is not None and len(variants) >= max_variants:
            break
    if not variants:
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            "campaign variant grid produced no variants"
        )
    return tuple(variants)


def _terminal_step_id(
    *,
    include_local_component_retention_plan: bool,
    include_scorer_response_dataset: bool,
    include_mlx_component_response: bool,
    include_local_component_loop: bool,
    prove_receiver_patch_output_change: bool,
) -> str:
    if include_local_component_retention_plan:
        return "plan_local_component_artifact_retention"
    if include_scorer_response_dataset:
        return "build_scorer_response_dataset"
    if include_mlx_component_response:
        return "local_mlx_component_response"
    if include_local_component_loop:
        return "local_cpu_contest_drift_eureka"
    if prove_receiver_patch_output_change:
        return "prove_receiver_patch_full_frame_output_change"
    return "emit_scorer_region_exact_ready_bridge_inputs"


def _campaign_report_command(
    *,
    repo_root: str | Path,
    output_root: Path,
    variants: Sequence[CascadeVariant],
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/build_scorer_region_selector_cascade_campaign_report.py",
        "--output",
        _repo_rel(output_root / "campaign_report.json", repo_root),
        "--overwrite",
    ]
    for variant in variants:
        command.extend(
            [
                "--variant-root",
                f"{variant.variant_id}={_repo_rel(output_root / variant.variant_id, repo_root)}",
            ]
        )
    return command


def _campaign_acquisition_policy_command(
    *,
    repo_root: str | Path,
    output_root: Path,
    master_gradient_tensor_path: str | Path | None,
    pixel_gradient_cache_path: str | Path | None,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/build_scorer_region_selector_cascade_acquisition_policy.py",
        "--campaign-report",
        _repo_rel(output_root / "campaign_report.json", repo_root),
        "--output",
        _repo_rel(output_root / "acquisition_policy.json", repo_root),
        "--overwrite",
    ]
    if master_gradient_tensor_path is not None:
        command.extend(["--master-gradient-tensor", _repo_rel(_resolve(master_gradient_tensor_path, repo_root), repo_root)])
    if pixel_gradient_cache_path is not None:
        command.extend(["--pixel-gradient-cache", _repo_rel(_resolve(pixel_gradient_cache_path, repo_root), repo_root)])
    return command


def discover_scorer_region_selector_cascade_variant_roots(
    *,
    repo_root: str | Path,
    variant_root_dir: str | Path,
) -> dict[str, str]:
    """Return immediate child directories as campaign variant roots.

    This keeps report harvest queue-owned and directory-owned instead of
    requiring an operator or agent to manually enumerate variant ids after a
    fanout run.
    """

    root = _resolve(variant_root_dir, repo_root)
    if not root.is_dir():
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            f"variant root directory missing: {root}"
        )
    roots: dict[str, str] = {}
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        roots[child.name] = _repo_rel(child, repo_root)
    if not roots:
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            f"variant root directory contains no child variant directories: {root}"
        )
    return roots


def build_scorer_region_selector_cascade_campaign_queue(
    *,
    repo_root: str | Path,
    queue_id: str,
    source_submission_dir: str | Path,
    output_root: str | Path,
    source_waterfill_work_order: str | Path | None = None,
    full_frame_inflate_parity_proof: str | Path | None = None,
    pose_null_modes_artifact: str | Path,
    segnet_softmax_16: str | Path,
    segnet_softmax_256: str | Path,
    null_fractions: Sequence[float] = (0.05, 0.10, 0.20),
    top_regions_per_pair_values: Sequence[int] = (2, 4),
    receiver_patch_max_pair_values: Sequence[int] = (12, 24, 48),
    receiver_patch_regions_per_pair_values: Sequence[int] = (1, 2),
    receiver_patch_rgb_deltas: Sequence[tuple[int, int, int]] = ((-1, -1, -1), (1, 1, 1)),
    receiver_patch_yuv_deltas: Sequence[tuple[int, int, int]] = (),
    selector_codec_family_groups: Sequence[Sequence[str]] = (
        ("fec10_adaptive_blend",),
        ("fec8_markov_static_order1",),
        ("fec8_markov_adaptive_order1",),
        (
            "fec10_adaptive_blend",
            "fec8_markov_static_order1",
            "fec8_markov_adaptive_order1",
            "fec8_markov_static_order2",
        ),
    ),
    scales: Sequence[int] = (32, 64, 128, 256),
    alphas: Sequence[int] = (1, 2, 4),
    repack_orders: Sequence[str] = ("p11_then_p15_then_receiver_patch",),
    max_variants: int | None = 32,
    prove_receiver_patch_output_change: bool = False,
    receiver_patch_output_change_file_list_entries: Sequence[str] = ("0.raw",),
    receiver_patch_output_change_expected_file_list_sha256: str | None = None,
    receiver_patch_output_change_expected_entry_count: int | None = None,
    receiver_patch_output_change_file_list_source: str | None = None,
    receiver_patch_output_change_parity_scope_kind: str = "contest_full_sample",
    receiver_patch_output_change_contest_full_sample_claim: bool = False,
    include_local_component_loop: bool = False,
    local_component_upstream_dir: str | Path = "upstream",
    local_component_video_names_file: str | Path = "upstream/public_test_video_names.txt",
    local_component_inflate_timeout_seconds: int = 1800,
    local_component_evaluate_timeout_seconds: int = 1800,
    include_mlx_component_response: bool = False,
    mlx_reference_cache_dir: str | Path = DEFAULT_MLX_REFERENCE_CACHE_DIR,
    mlx_device: str = "gpu",
    mlx_cache_batch_pairs: int = 1,
    mlx_batch_pairs: int = 1,
    mlx_max_pairs: int | None = 12,
    include_scorer_response_dataset: bool = False,
    scorer_response_baseline_score: float | None = None,
    scorer_response_baseline_archive_bytes: int | None = None,
    include_local_component_retention_plan: bool = False,
    execute_local_component_retention: bool = False,
    local_component_retention_action: str = "move",
    local_component_retention_min_bytes: str = "1",
    local_component_retention_cold_store_roots: Sequence[str | Path] = (),
    local_component_retention_cold_store_reserve_gb: float = 40.0,
    max_concurrency_local_cpu: int = 2,
    max_concurrency_local_mlx: int = 1,
    max_concurrency_local_io_heavy: int = 1,
    append_campaign_harvest: bool = True,
    append_campaign_acquisition_policy: bool = True,
    master_gradient_tensor_path: str | Path | None = DEFAULT_MASTER_GRADIENT_TENSOR_PATH,
    pixel_gradient_cache_path: str | Path | None = DEFAULT_PIXEL_GRADIENT_CACHE_PATH,
) -> dict[str, Any]:
    """Return a queue-owned grouped cascade search over scorer-null budget spends."""

    root = _resolve(output_root, repo_root)
    source_archive = _submission_archive(source_submission_dir, repo_root=repo_root)
    variants = enumerate_cascade_variants(
        null_fractions=null_fractions,
        top_regions_per_pair_values=top_regions_per_pair_values,
        receiver_patch_max_pair_values=receiver_patch_max_pair_values,
        receiver_patch_regions_per_pair_values=receiver_patch_regions_per_pair_values,
        receiver_patch_rgb_deltas=receiver_patch_rgb_deltas,
        receiver_patch_yuv_deltas=receiver_patch_yuv_deltas,
        selector_codec_family_groups=selector_codec_family_groups,
        scales=scales,
        alphas=alphas,
        repack_orders=repack_orders,
        max_variants=max_variants,
    )

    experiments: list[dict[str, Any]] = []
    variant_metadata: list[dict[str, Any]] = []
    terminal_step = _terminal_step_id(
        include_local_component_retention_plan=include_local_component_retention_plan,
        include_scorer_response_dataset=include_scorer_response_dataset,
        include_mlx_component_response=include_mlx_component_response,
        include_local_component_loop=include_local_component_loop,
        prove_receiver_patch_output_change=prove_receiver_patch_output_change,
    )
    for index, variant in enumerate(variants):
        variant_root = root / variant.variant_id
        chain_label = f"cascade_c_grouped_{variant.variant_id}"
        child = build_scorer_region_selector_chain_queue(
            repo_root=repo_root,
            queue_id=f"{queue_id}_{variant.variant_id}",
            source_submission_dir=source_submission_dir,
            output_root=variant_root,
            source_waterfill_work_order=source_waterfill_work_order,
            full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
            pose_null_modes_artifact=pose_null_modes_artifact,
            segnet_softmax_16=segnet_softmax_16,
            segnet_softmax_256=segnet_softmax_256,
            materialize_upstream_artifacts=True,
            materialize_receiver_patch=True,
            null_fraction=variant.null_fraction,
            top_regions_per_pair=variant.top_regions_per_pair,
            receiver_patch_max_pairs=variant.receiver_patch_max_pairs,
            receiver_patch_regions_per_pair=variant.receiver_patch_regions_per_pair,
            receiver_patch_rgb_delta=variant.receiver_patch_rgb_delta,
            prove_receiver_patch_output_change=prove_receiver_patch_output_change,
            receiver_patch_output_change_file_list_entries=(
                receiver_patch_output_change_file_list_entries
            ),
            receiver_patch_output_change_expected_file_list_sha256=(
                receiver_patch_output_change_expected_file_list_sha256
            ),
            receiver_patch_output_change_expected_entry_count=(
                receiver_patch_output_change_expected_entry_count
            ),
            receiver_patch_output_change_file_list_source=(
                receiver_patch_output_change_file_list_source
            ),
            receiver_patch_output_change_parity_scope_kind=(
                receiver_patch_output_change_parity_scope_kind
            ),
            receiver_patch_output_change_contest_full_sample_claim=(
                receiver_patch_output_change_contest_full_sample_claim
            ),
            include_local_component_loop=include_local_component_loop,
            local_component_upstream_dir=local_component_upstream_dir,
            local_component_video_names_file=local_component_video_names_file,
            local_component_inflate_timeout_seconds=(
                local_component_inflate_timeout_seconds
            ),
            local_component_evaluate_timeout_seconds=(
                local_component_evaluate_timeout_seconds
            ),
            include_mlx_component_response=include_mlx_component_response,
            mlx_reference_cache_dir=mlx_reference_cache_dir,
            mlx_device=mlx_device,
            mlx_cache_batch_pairs=mlx_cache_batch_pairs,
            mlx_batch_pairs=mlx_batch_pairs,
            mlx_max_pairs=mlx_max_pairs,
            include_scorer_response_dataset=include_scorer_response_dataset,
            scorer_response_baseline_score=scorer_response_baseline_score,
            scorer_response_baseline_archive_bytes=(
                scorer_response_baseline_archive_bytes
            ),
            include_local_component_retention_plan=include_local_component_retention_plan,
            execute_local_component_retention=execute_local_component_retention,
            local_component_retention_action=local_component_retention_action,
            local_component_retention_min_bytes=local_component_retention_min_bytes,
            local_component_retention_cold_store_roots=(
                local_component_retention_cold_store_roots
            ),
            local_component_retention_cold_store_reserve_gb=(
                local_component_retention_cold_store_reserve_gb
            ),
            chain_label=chain_label,
            codec_families=variant.selector_codec_families,
            scales=variant.scales,
            alphas=variant.alphas,
            max_concurrency_local_cpu=1,
        )
        child_experiment = dict(child["experiments"][0])
        child_experiment["id"] = variant.variant_id
        child_experiment["priority"] = index + 1
        child_experiment["tags"] = ordered_unique(
            [
                *child_experiment.get("tags", []),
                "grouped-cascade-campaign",
                f"delta-space:{variant.receiver_patch_delta_space}",
                f"repack-order:{variant.repack_order}",
            ]
        )
        child_experiment["metadata"] = {
            **dict(child_experiment.get("metadata") or {}),
            **variant.to_metadata(output_root=_repo_rel(variant_root, repo_root)),
        }
        experiments.append(child_experiment)
        variant_metadata.append(variant.to_metadata(output_root=_repo_rel(variant_root, repo_root)))

    harvest_path = root / "campaign_report.json"
    acquisition_policy_path = root / "acquisition_policy.json"
    if append_campaign_harvest:
        experiments.append(
            {
                "id": "campaign_harvest",
                "priority": len(experiments) + 1,
                "status": "queued",
                "tags": [
                    "frontier-rate-attack",
                    "cascade-c",
                    "grouped-campaign-harvest",
                    "no-score-authority",
                ],
                "metadata": {
                    "schema": "scorer_region_selector_cascade_campaign_harvest_metadata.v1",
                    "variant_count": len(variants),
                    "campaign_report_path": _repo_rel(harvest_path, repo_root),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "steps": [
                    {
                        "id": "harvest_campaign_learning_surface",
                        "kind": "command",
                        "requires": [
                            f"{variant.variant_id}.{terminal_step}" for variant in variants
                        ],
                        "command": _campaign_report_command(
                            repo_root=repo_root,
                            output_root=root,
                            variants=variants,
                        ),
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 240,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(harvest_path, repo_root),
                                "key": "schema",
                                "equals": SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(harvest_path, repo_root),
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [_repo_rel(harvest_path, repo_root)],
                            "input_artifact_paths": [
                                _repo_rel(root / variant.variant_id, repo_root)
                                for variant in variants
                            ],
                            "include_postcondition_paths": True,
                        },
                    }
                ],
            }
        )
    if append_campaign_harvest and append_campaign_acquisition_policy:
        experiments.append(
            {
                "id": "campaign_acquisition_policy",
                "priority": len(experiments) + 1,
                "status": "queued",
                "tags": [
                    "frontier-rate-attack",
                    "cascade-c",
                    "grouped-campaign-acquisition-policy",
                    "master-gradient-prior",
                    "uniward-pixel-prior",
                    "no-score-authority",
                ],
                "metadata": {
                    "schema": "scorer_region_selector_cascade_acquisition_policy_step_metadata.v1",
                    "campaign_report_path": _repo_rel(harvest_path, repo_root),
                    "acquisition_policy_path": _repo_rel(acquisition_policy_path, repo_root),
                    "master_gradient_tensor_path": (
                        _repo_rel(_resolve(master_gradient_tensor_path, repo_root), repo_root)
                        if master_gradient_tensor_path is not None
                        else None
                    ),
                    "pixel_gradient_cache_path": (
                        _repo_rel(_resolve(pixel_gradient_cache_path, repo_root), repo_root)
                        if pixel_gradient_cache_path is not None
                        else None
                    ),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "steps": [
                    {
                        "id": "build_campaign_acquisition_policy",
                        "kind": "command",
                        "requires": ["campaign_harvest.harvest_campaign_learning_surface"],
                        "command": _campaign_acquisition_policy_command(
                            repo_root=repo_root,
                            output_root=root,
                            master_gradient_tensor_path=master_gradient_tensor_path,
                            pixel_gradient_cache_path=pixel_gradient_cache_path,
                        ),
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 900,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(acquisition_policy_path, repo_root),
                                "key": "schema",
                                "equals": SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(acquisition_policy_path, repo_root),
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [_repo_rel(acquisition_policy_path, repo_root)],
                            "input_artifact_paths": [_repo_rel(harvest_path, repo_root)],
                            "include_postcondition_paths": True,
                        },
                    }
                ],
            }
        )

    controls = {
        "mode": "running",
        "local_first": True,
        "max_concurrency": {
            "local_cpu": int(max_concurrency_local_cpu),
            "local_io_heavy": int(max_concurrency_local_io_heavy),
        },
    }
    if include_mlx_component_response and str(mlx_device) == "gpu":
        controls["max_concurrency"]["local_mlx"] = int(max_concurrency_local_mlx)

    queue = {
        "schema": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "controls": controls,
        "metadata": {
            "schema": SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA,
            "queue_id": queue_id,
            "source_submission_dir": _repo_rel(
                _resolve(source_submission_dir, repo_root),
                repo_root,
            ),
            "source_archive": _archive_record(source_archive, repo_root=repo_root),
            "output_root": _repo_rel(root, repo_root),
            "campaign_report_path": _repo_rel(harvest_path, repo_root),
            "acquisition_policy_path": _repo_rel(acquisition_policy_path, repo_root),
            "variant_count": len(variants),
            "variant_grid_truncated": max_variants is not None,
            "variant_grid_max_variants": max_variants,
            "variants": variant_metadata,
            "operator_contract": build_scorer_region_operator_contract(
                chain_label="cascade_c_grouped_campaign",
                receiver_patch_enabled=True,
            ),
            "execution_policy": (
                "run grouped P19/P18/P11/P15 receiver-closed variants under "
                "experiment_queue.v1; use MLX/local CPU as acquisition only; "
                "exact CPU/CUDA auth eval remains gated by bridge rows"
            ),
            "exact_auth_policy": {
                "cpu_before_cuda": True,
                "requires_local_cpu_before_exact_auth": bool(include_local_component_loop),
                "mlx_is_acquisition_signal_only": bool(include_mlx_component_response),
                "dispatch_source": "per_variant_scorer_region_exact_ready_bridge",
            },
            "acquisition_policy": {
                "enabled": bool(append_campaign_harvest and append_campaign_acquisition_policy),
                "schema": SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA,
                "mathematical_action": (
                    "S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/"
                    f"{CONTEST_RATE_DENOM_BYTES}"
                ),
                "rate_score_per_byte": RATE_SCORE_PER_BYTE,
                "master_gradient_tensor_path": (
                    _repo_rel(_resolve(master_gradient_tensor_path, repo_root), repo_root)
                    if master_gradient_tensor_path is not None
                    else None
                ),
                "pixel_gradient_cache_path": (
                    _repo_rel(_resolve(pixel_gradient_cache_path, repo_root), repo_root)
                    if pixel_gradient_cache_path is not None
                    else None
                ),
            },
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "allowed_use": "queue_owned_grouped_p18_p19_p11_p15_local_campaign",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        },
        "experiments": experiments,
    }
    return normalize_queue_definition(queue)


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            f"expected JSON object: {path}"
        )
    return payload


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _best_dataset_delta(dataset: Mapping[str, Any]) -> float | None:
    rows = dataset.get("rows")
    if not isinstance(rows, list):
        return None
    values: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = row.get("delta_vs_baseline_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return min(values) if values else None


def _safe_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _saved_bytes(row: Mapping[str, Any]) -> int:
    for key in (
        "cumulative_rate_saved_bytes_vs_source",
        "selector_saved_bytes",
        "repack_saved_bytes_after_selector",
    ):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _posterior_acquisition_update(row: Mapping[str, Any]) -> dict[str, Any]:
    dataset_delta = _safe_number(row.get("best_dataset_delta_vs_baseline_score"))
    local_delta = _safe_number(row.get("local_cpu_delta_vs_auth_frontier"))
    mlx_positive = dataset_delta is not None and dataset_delta < 0.0
    cpu_present = row.get("local_cpu_present") is True
    cpu_gate_passed = row.get("candidate_passed_local_cpu_gate") is True
    cpu_negative = cpu_present and not cpu_gate_passed
    split = mlx_positive and cpu_negative
    output_change = row.get("raw_shape_preserving_output_change_observed") is True
    bridge_present = row.get("exact_ready_bridge_present") is True
    saved_bytes = _saved_bytes(row)
    stack_penalty = 0.0
    if split:
        stack_penalty += 0.22
    if not output_change:
        stack_penalty += 0.06
    if not bridge_present:
        stack_penalty += 0.04
    if saved_bytes <= 0:
        stack_penalty += 0.05
    if cpu_gate_passed:
        decision = "promote_to_exact_ready_bridge_preclaim"
    elif split:
        decision = "demote_grouped_stack_and_remeasure_cpu_before_budget"
    elif mlx_positive and not cpu_present:
        decision = "queue_cpu_pre_gate_before_exact_or_materializer_budget"
    elif cpu_negative:
        decision = "demote_or_rebudget_after_negative_cpu_pre_gate"
    else:
        decision = "hold_until_mlx_or_cpu_evidence"
    return {
        "schema": "scorer_region_cascade_posterior_acquisition_update.v1",
        "variant_id": row.get("variant_id"),
        "operator_position_group": ["P19", "P18", "P11", "P15"],
        "mlx_acquisition_positive": mlx_positive,
        "full_cpu_negative": cpu_negative,
        "mlx_positive_full_cpu_negative_split": split,
        "best_dataset_delta_vs_baseline_score": dataset_delta,
        "local_cpu_delta_vs_auth_frontier": local_delta,
        "cpu_pre_gate_status": (
            "passed" if cpu_gate_passed else "failed" if cpu_present else "missing"
        ),
        "byte_pressure": {
            "saved_bytes": saved_bytes,
            "credit_state": "available" if saved_bytes > 0 else "exhausted_or_absent",
            "byte_pressure_penalty": 0.0 if saved_bytes > 0 else 0.05,
        },
        "stack_penalty": round(stack_penalty, 6),
        "posterior_budget_routing_decision": decision,
        "negative_evidence_demotes_family_stage_scope": split or cpu_negative,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def build_scorer_region_selector_cascade_campaign_report(
    *,
    repo_root: str | Path,
    variant_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Harvest grouped cascade outputs into one false-authority learning surface."""

    rows: list[dict[str, Any]] = []
    for variant_id, root_value in sorted(variant_roots.items()):
        root = _resolve(root_value, repo_root)
        chain_report = _read_json_if_present(root / "scorer_region_selector_chain_report.json")
        output_change = _read_json_if_present(
            root
            / "frame1_region_waterfill_runtime_patch"
            / "full_frame_output_change_proof"
            / "shell_inflate_output_change.json"
        )
        local_cpu = _read_json_if_present(
            root
            / "frame1_region_waterfill_runtime_patch"
            / "local_component_spot_check"
            / "local_cpu_advisory.json"
        )
        local_cpu_eureka = _read_json_if_present(
            root
            / "frame1_region_waterfill_runtime_patch"
            / "local_component_spot_check"
            / "local_cpu_contest_drift_eureka.json"
        )
        mlx_response = _read_json_if_present(
            root
            / "frame1_region_waterfill_runtime_patch"
            / "local_component_spot_check"
            / "mlx_scorer_response.json"
        )
        scorer_dataset = _read_json_if_present(
            root
            / "frame1_region_waterfill_runtime_patch"
            / "local_component_spot_check"
            / "scorer_response_dataset.json"
        )
        bridge = _read_json_if_present(root / "scorer_region_exact_ready_bridge_report.json")
        for payload, label in (
            (chain_report, "chain_report"),
            (output_change, "output_change"),
            (local_cpu, "local_cpu"),
            (local_cpu_eureka, "local_cpu_eureka"),
            (mlx_response, "mlx_response"),
            (scorer_dataset, "scorer_dataset"),
            (bridge, "exact_ready_bridge"),
        ):
            if payload is not None:
                require_no_truthy_authority_fields(
                    payload,
                    context=f"cascade_campaign_report:{variant_id}:{label}",
                )
        local_cpu_score = local_cpu.get("canonical_score") if local_cpu else None
        if not isinstance(local_cpu_score, (int, float)) or isinstance(local_cpu_score, bool):
            local_cpu_score = None
        auth_frontier_score = (
            local_cpu_eureka.get("auth_frontier_score") if local_cpu_eureka else None
        )
        if not isinstance(auth_frontier_score, (int, float)) or isinstance(
            auth_frontier_score, bool
        ):
            auth_frontier_score = None
        local_cpu_delta_vs_auth_frontier = (
            float(local_cpu_score) - float(auth_frontier_score)
            if local_cpu_score is not None and auth_frontier_score is not None
            else None
        )
        local_cpu_eureka_trigger = (
            local_cpu_eureka.get("eureka_trigger") if local_cpu_eureka else None
        )
        bridge_blockers = bridge.get("blockers") if bridge else None
        local_cpu_gate_blockers = []
        if isinstance(bridge, Mapping):
            bridge_rows = bridge.get("rows")
            if isinstance(bridge_rows, Sequence) and bridge_rows:
                first_bridge_row = bridge_rows[0]
                if isinstance(first_bridge_row, Mapping):
                    gate = first_bridge_row.get("local_cpu_gate")
                    if isinstance(gate, Mapping):
                        local_cpu_gate_blockers = _string_list(gate.get("blockers"))
        candidate_passed_local_cpu_gate = (
            local_cpu_score is not None
            and auth_frontier_score is not None
            and local_cpu_delta_vs_auth_frontier is not None
            and local_cpu_delta_vs_auth_frontier < 0.0
            and local_cpu_eureka_trigger is True
        )
        row = {
            "schema": "scorer_region_selector_cascade_campaign_row.v1",
            "variant_id": variant_id,
            "variant_root": _repo_rel(root, repo_root),
            "chain_report_present": chain_report is not None,
            "output_change_present": output_change is not None,
            "local_cpu_present": local_cpu is not None,
            "local_cpu_eureka_present": local_cpu_eureka is not None,
            "mlx_response_present": mlx_response is not None,
            "scorer_response_dataset_present": scorer_dataset is not None,
            "exact_ready_bridge_present": bridge is not None,
            "selector_saved_bytes": (
                chain_report.get("selector_saved_bytes") if chain_report else None
            ),
            "repack_saved_bytes_after_selector": (
                chain_report.get("repack_saved_bytes_after_selector")
                if chain_report
                else None
            ),
            "cumulative_rate_saved_bytes_vs_source": (
                chain_report.get("cumulative_rate_saved_bytes_vs_source")
                if chain_report
                else None
            ),
            "selected_local_survivor_stage": (
                chain_report.get("selected_local_survivor_stage")
                if chain_report
                else None
            ),
            "output_change_observed": (
                output_change.get("output_change_observed")
                if output_change
                else None
            ),
            "raw_shape_preserving_output_change_observed": (
                output_change.get("raw_shape_preserving_output_change_observed")
                if output_change
                else None
            ),
            "differing_byte_count": (
                output_change.get("differing_byte_count") if output_change else None
            ),
            "local_cpu_canonical_score": local_cpu_score,
            "local_cpu_auth_frontier_score": auth_frontier_score,
            "local_cpu_delta_vs_auth_frontier": local_cpu_delta_vs_auth_frontier,
            "local_cpu_eureka_trigger": local_cpu_eureka_trigger,
            "local_cpu_recommended_action": (
                local_cpu_eureka.get("recommended_action") if local_cpu_eureka else None
            ),
            "local_cpu_gate_blockers": local_cpu_gate_blockers,
            "candidate_passed_local_cpu_gate": candidate_passed_local_cpu_gate,
            "local_cpu_avg_posenet_dist": (
                local_cpu.get("avg_posenet_dist") if local_cpu else None
            ),
            "local_cpu_avg_segnet_dist": (
                local_cpu.get("avg_segnet_dist") if local_cpu else None
            ),
            "mlx_canonical_score": (
                mlx_response.get("canonical_score") if mlx_response else None
            ),
            "mlx_n_samples": mlx_response.get("n_samples") if mlx_response else None,
            "best_dataset_delta_vs_baseline_score": (
                _best_dataset_delta(scorer_dataset) if scorer_dataset else None
            ),
            "bridge_dispatch_ready_count": (
                bridge.get("dispatch_ready_count") if bridge else None
            ),
            "bridge_blockers": bridge_blockers,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        rows.append(row)
    def ranking_key(row: Mapping[str, Any]) -> tuple[int, float, str]:
        local_delta = row.get("local_cpu_delta_vs_auth_frontier")
        if row.get("candidate_passed_local_cpu_gate") is True and isinstance(
            local_delta, (int, float)
        ):
            return (0, float(local_delta), str(row["variant_id"]))
        dataset_delta = row.get("best_dataset_delta_vs_baseline_score")
        if row.get("local_cpu_present") is not True and isinstance(
            dataset_delta, (int, float)
        ):
            return (1, float(dataset_delta), str(row["variant_id"]))
        if isinstance(local_delta, (int, float)):
            return (2, float(local_delta), str(row["variant_id"]))
        return (3, float("inf"), str(row["variant_id"]))

    rows.sort(key=ranking_key)
    posterior_acquisition_updates = [
        _posterior_acquisition_update(row) for row in rows
    ]
    local_cpu_rows = [row for row in rows if row.get("local_cpu_present") is True]
    local_cpu_deltas = [
        float(row["local_cpu_delta_vs_auth_frontier"])
        for row in local_cpu_rows
        if isinstance(row.get("local_cpu_delta_vs_auth_frontier"), (int, float))
        and not isinstance(row.get("local_cpu_delta_vs_auth_frontier"), bool)
    ]
    local_cpu_passed_count = sum(
        1 for row in local_cpu_rows if row.get("candidate_passed_local_cpu_gate") is True
    )
    mlx_positive_full_cpu_negative_split_count = sum(
        1
        for update in posterior_acquisition_updates
        if update["mlx_positive_full_cpu_negative_split"] is True
    )
    output_change_without_cpu_win_count = sum(
        1
        for row in rows
        if row.get("output_change_observed") is True
        and row.get("candidate_passed_local_cpu_gate") is not True
    )
    aggregate_learning = {
        "schema": "scorer_region_selector_cascade_campaign_aggregate_learning.v1",
        "local_cpu_observed_count": len(local_cpu_rows),
        "local_cpu_passed_gate_count": local_cpu_passed_count,
        "local_cpu_all_observed_failed_gate": bool(local_cpu_rows)
        and local_cpu_passed_count == 0,
        "best_local_cpu_delta_vs_auth_frontier": (
            min(local_cpu_deltas) if local_cpu_deltas else None
        ),
        "worst_local_cpu_delta_vs_auth_frontier": (
            max(local_cpu_deltas) if local_cpu_deltas else None
        ),
        "mlx_positive_full_cpu_negative_split_count": (
            mlx_positive_full_cpu_negative_split_count
        ),
        "output_change_without_cpu_win_count": output_change_without_cpu_win_count,
        "recommended_next_queue_policy": (
            "acquisition_first_or_cpu_gate_only_no_post_cpu_mlx"
            if local_cpu_rows
            and local_cpu_passed_count == 0
            and mlx_positive_full_cpu_negative_split_count > 0
            else "continue_exact_ready_eureka_gate"
        ),
        "posterior_routing_decision": (
            "demote_post_cpu_mlx_for_current_operator_family_until_acquisition_model_changes"
            if local_cpu_rows
            and local_cpu_passed_count == 0
            and mlx_positive_full_cpu_negative_split_count > 0
            else "keep_current_queue_policy"
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    best_selection_basis = None
    if rows:
        best_key = ranking_key(rows[0])
        best_selection_basis = (
            "local_cpu_gate_passed"
            if best_key[0] == 0
            else "mlx_acquisition_without_local_cpu"
            if best_key[0] == 1
            else "local_cpu_gate_failed"
            if best_key[0] == 2
            else "incomplete"
        )
    blockers = ordered_unique(
        [
            *(
                []
                if any(row["scorer_response_dataset_present"] for row in rows)
                else ["campaign_has_no_scorer_response_dataset_rows_yet"]
            ),
            *[
                blocker
                for row in rows
                for blocker in _string_list(row.get("local_cpu_gate_blockers"))
            ],
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ]
    )
    payload = {
        "schema": SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA,
        "variant_count": len(rows),
        "completed_learning_variant_count": sum(
            1 for row in rows if row["scorer_response_dataset_present"]
        ),
        "local_cpu_variant_count": sum(1 for row in rows if row["local_cpu_present"]),
        "mlx_variant_count": sum(1 for row in rows if row["mlx_response_present"]),
        "rows": rows,
        "posterior_acquisition_updates": posterior_acquisition_updates,
        "aggregate_learning": aggregate_learning,
        "mlx_positive_full_cpu_negative_split_count": (
            mlx_positive_full_cpu_negative_split_count
        ),
        "best_variant_id": rows[0]["variant_id"] if rows else None,
        "best_variant_selection_basis": best_selection_basis,
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "grouped_cascade_campaign_learning_surface",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="scorer_region_selector_cascade_campaign_report",
    )
    return payload


def _local_delta(row: Mapping[str, Any]) -> float | None:
    return _safe_number(row.get("local_cpu_delta_vs_auth_frontier"))


def _parse_variant_dimensions(variant_id: object) -> dict[str, Any]:
    text = str(variant_id or "")
    match = re.search(r"^nf(?P<nf>[0-9_]+)_r(?P<regions>\d+)_p(?P<pairs>\d+)_rp(?P<rp>\d+)_", text)
    delta_space = None
    if "_yuv601_proxy_as_rgb_" in text:
        delta_space = "yuv601_proxy_as_rgb"
    elif "_rgb_" in text:
        delta_space = "rgb"
    codec = None
    codec_match = re.search(r"_cf(?P<codec>.+?)_p11_then_p15_then_receiver_patch$", text)
    if codec_match:
        codec = codec_match.group("codec")
    return {
        "variant_id": text,
        "null_fraction": (
            float("0." + match.group("nf").split("_", 1)[1])
            if match and "_" in match.group("nf")
            else None
        ),
        "top_regions_per_pair": int(match.group("regions")) if match else None,
        "receiver_patch_max_pairs": int(match.group("pairs")) if match else None,
        "receiver_patch_regions_per_pair": int(match.group("rp")) if match else None,
        "receiver_patch_delta_space": delta_space,
        "selector_codec_family": codec,
    }


def _dimension_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dimensions = {
        "receiver_patch_delta_space": {},
        "top_regions_per_pair": {},
        "receiver_patch_max_pairs": {},
        "receiver_patch_regions_per_pair": {},
        "selector_codec_family": {},
    }
    for row in rows:
        delta = _local_delta(row)
        if delta is None:
            continue
        parsed = _parse_variant_dimensions(row.get("variant_id"))
        for key, buckets in dimensions.items():
            value = parsed.get(key)
            if value is None:
                continue
            bucket = buckets.setdefault(str(value), [])
            bucket.append(delta)
    result: dict[str, list[dict[str, Any]]] = {}
    for key, buckets in dimensions.items():
        rows_out = []
        for value, values in sorted(buckets.items(), key=lambda item: item[0]):
            rows_out.append(
                {
                    "value": value,
                    "count": len(values),
                    "best_local_cpu_delta_vs_auth_frontier": min(values),
                    "worst_local_cpu_delta_vs_auth_frontier": max(values),
                    "mean_local_cpu_delta_vs_auth_frontier": sum(values) / len(values),
                }
            )
        result[key] = rows_out
    return result


def _score_pose_marginal(d_pose: float) -> float:
    if d_pose <= 0.0 or not math.isfinite(d_pose):
        raise ScorerRegionSelectorCascadeCampaignQueueError(
            f"d_pose must be positive finite for score marginal, got {d_pose!r}"
        )
    return 5.0 / math.sqrt(10.0 * d_pose)


def _summarize_master_gradient_tensor(
    *,
    repo_root: str | Path,
    tensor_path: str | Path | None,
    max_chunk_byte_rows: int = 4096,
) -> dict[str, Any]:
    if tensor_path is None:
        return {
            "schema": "scorer_region_master_gradient_prior_summary.v1",
            "available": False,
            "blockers": ["master_gradient_tensor_not_configured"],
            **FALSE_AUTHORITY,
        }
    resolved = _resolve(tensor_path, repo_root)
    if not resolved.is_file():
        return {
            "schema": "scorer_region_master_gradient_prior_summary.v1",
            "available": False,
            "path": _repo_rel(resolved, repo_root),
            "blockers": ["master_gradient_tensor_missing"],
            **FALSE_AUTHORITY,
        }
    try:
        import numpy as np
    except ImportError:
        return {
            "schema": "scorer_region_master_gradient_prior_summary.v1",
            "available": False,
            "path": _repo_rel(resolved, repo_root),
            "blockers": ["numpy_unavailable_for_master_gradient_prior"],
            **FALSE_AUTHORITY,
        }
    arr = np.load(resolved, mmap_mode="r")
    if arr.ndim != 3 or int(arr.shape[2]) != 3:
        return {
            "schema": "scorer_region_master_gradient_prior_summary.v1",
            "available": False,
            "path": _repo_rel(resolved, repo_root),
            "shape": [int(dim) for dim in arr.shape],
            "blockers": ["master_gradient_tensor_not_per_pair_per_byte_axes3"],
            **FALSE_AUTHORITY,
        }
    n_bytes, n_pairs, _ = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
    pair_axis_l1 = np.zeros((n_pairs, 3), dtype=np.float64)
    nonzero_byte_rows = 0
    chunk_rows = max(1, int(max_chunk_byte_rows))
    for start in range(0, n_bytes, chunk_rows):
        chunk = np.asarray(arr[start : start + chunk_rows], dtype=np.float64)
        abs_chunk = np.abs(chunk)
        pair_axis_l1 += abs_chunk.sum(axis=0)
        nonzero_byte_rows += int(np.any(abs_chunk > 0.0, axis=(1, 2)).sum())
    pose_l1 = pair_axis_l1[:, 1]
    op = FULL600_MLX_OPERATING_POINT
    coeffs = np.asarray(
        [100.0, _score_pose_marginal(float(op["d_pose"])), RATE_SCORE_PER_BYTE],
        dtype=np.float64,
    )
    pair_score_l1 = pair_axis_l1 @ coeffs
    bottom_k = max(1, min(n_pairs, n_pairs // 10))
    preview_k = min(32, bottom_k)
    pose_null_order = np.argsort(pose_l1, kind="stable")
    score_null_order = np.argsort(pair_score_l1, kind="stable")
    pose_vulnerable_order = np.argsort(-pose_l1, kind="stable")
    return {
        "schema": "scorer_region_master_gradient_prior_summary.v1",
        "available": True,
        "path": _repo_rel(resolved, repo_root),
        "bytes": resolved.stat().st_size,
        "sha256": None,
        "sha256_status": "skipped_large_tensor_speed_guard",
        "shape": [n_bytes, n_pairs, 3],
        "dtype": str(arr.dtype),
        "source_axis": "[macOS-MLX research-signal]",
        "authority_boundary": (
            "reference prior only; no archive-specific score or dispatch authority"
        ),
        "operating_point": dict(op),
        "score_marginal_coefficients": {
            "seg": float(coeffs[0]),
            "pose": float(coeffs[1]),
            "rate_per_byte": float(coeffs[2]),
        },
        "nonzero_byte_rows": nonzero_byte_rows,
        "pose_l1_quantiles": {
            str(q): float(np.quantile(pose_l1, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "pair_score_l1_quantiles": {
            str(q): float(np.quantile(pair_score_l1, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "pose_null_bottom_decile_pair_indices_preview": [
            int(idx) for idx in pose_null_order[:preview_k]
        ],
        "score_null_bottom_decile_pair_indices_preview": [
            int(idx) for idx in score_null_order[:preview_k]
        ],
        "pose_vulnerable_top_decile_pair_indices_preview": [
            int(idx) for idx in pose_vulnerable_order[:preview_k]
        ],
        "blockers": ["archive_specific_master_gradient_anchor_missing_for_current_campaign"],
        **FALSE_AUTHORITY,
    }


def _summarize_pixel_gradient_cache(
    *,
    repo_root: str | Path,
    cache_path: str | Path | None,
) -> dict[str, Any]:
    if cache_path is None:
        return {
            "schema": "scorer_region_pixel_gradient_prior_summary.v1",
            "available": False,
            "blockers": ["pixel_gradient_cache_not_configured"],
            **FALSE_AUTHORITY,
        }
    resolved = _resolve(cache_path, repo_root)
    if not resolved.is_file():
        return {
            "schema": "scorer_region_pixel_gradient_prior_summary.v1",
            "available": False,
            "path": _repo_rel(resolved, repo_root),
            "blockers": ["pixel_gradient_cache_missing"],
            **FALSE_AUTHORITY,
        }
    try:
        import numpy as np
    except ImportError:
        return {
            "schema": "scorer_region_pixel_gradient_prior_summary.v1",
            "available": False,
            "path": _repo_rel(resolved, repo_root),
            "blockers": ["numpy_unavailable_for_pixel_gradient_prior"],
            **FALSE_AUTHORITY,
        }
    cache = np.load(resolved)
    if "seg_grads" not in cache or "pose_grads" not in cache:
        return {
            "schema": "scorer_region_pixel_gradient_prior_summary.v1",
            "available": False,
            "path": _repo_rel(resolved, repo_root),
            "blockers": ["pixel_gradient_cache_missing_seg_or_pose_grads"],
            **FALSE_AUTHORITY,
        }
    seg = np.asarray(cache["seg_grads"], dtype=np.float32)
    pose = np.asarray(cache["pose_grads"], dtype=np.float32)
    if seg.shape != pose.shape:
        return {
            "schema": "scorer_region_pixel_gradient_prior_summary.v1",
            "available": False,
            "path": _repo_rel(resolved, repo_root),
            "seg_shape": [int(dim) for dim in seg.shape],
            "pose_shape": [int(dim) for dim in pose.shape],
            "blockers": ["pixel_gradient_cache_shape_mismatch"],
            **FALSE_AUTHORITY,
        }
    weight = compute_per_pixel_uniward_weight_map_numpy(seg, pose)
    flat_weight = weight.reshape(-1)
    if weight.ndim >= 3:
        frame_weight = weight.reshape(weight.shape[0], -1).mean(axis=1)
        safe_frame_order = np.argsort(-frame_weight, kind="stable")[: min(16, weight.shape[0])]
    else:
        frame_weight = np.asarray([float(weight.mean())], dtype=np.float32)
        safe_frame_order = np.asarray([0], dtype=np.int64)
    return {
        "schema": "scorer_region_pixel_gradient_prior_summary.v1",
        "available": True,
        "path": _repo_rel(resolved, repo_root),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "shape": [int(dim) for dim in weight.shape],
        "source_axis": "[macOS-MLX research-signal]",
        "authority_boundary": (
            "per-pixel acquisition prior only; candidate still needs receiver proof, "
            "local CPU gate, and exact auth"
        ),
        "uniward_weight_quantiles": {
            str(q): float(np.quantile(flat_weight, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "safest_frame_indices_preview": [int(idx) for idx in safe_frame_order],
        "safest_frame_weight_mean_preview": [
            float(frame_weight[int(idx)]) for idx in safe_frame_order
        ],
        "blockers": ["pixel_gradient_cache_is_partial_sample_not_full_contest_video"],
        **FALSE_AUTHORITY,
    }


def _prior_blockers(prior: Mapping[str, Any], *, source_name: str) -> list[str]:
    blockers = []
    if prior.get("available") is not True:
        blockers.append(f"{source_name}_prior_unavailable")
    prior_blockers = prior.get("blockers")
    if isinstance(prior_blockers, list):
        blockers.extend(str(item) for item in prior_blockers if str(item))
    if prior.get("available") is True and not prior.get("sha256"):
        blockers.append(f"{source_name}_sha256_missing_or_skipped")
    return ordered_unique(blockers)


def _build_selection_manifest(
    *,
    master_gradient: Mapping[str, Any],
    pixel_gradient: Mapping[str, Any],
) -> dict[str, Any]:
    master_blockers = _prior_blockers(master_gradient, source_name="master_gradient")
    pixel_blockers = _prior_blockers(pixel_gradient, source_name="pixel_gradient")
    pose_pair_source_ready = not master_blockers
    region_source_ready = not pixel_blockers
    blockers = ordered_unique([*master_blockers, *pixel_blockers])
    selection_ready = pose_pair_source_ready and region_source_ready
    manifest = {
        "schema": SCORER_REGION_SELECTOR_CASCADE_SELECTION_MANIFEST_SCHEMA,
        "selection_ready": selection_ready,
        "selection_ready_blockers": blockers,
        "pose_pair_source": (
            "master_gradient_pose_null_bottom_decile"
            if pose_pair_source_ready
            else "blocked_until_archive_specific_master_gradient_manifest"
        ),
        "pose_pair_source_ready": pose_pair_source_ready,
        "pose_pair_source_blockers": master_blockers,
        "region_source": (
            "uniward_pixel_gradient_safe_regions"
            if region_source_ready
            else "blocked_until_full_video_pixel_gradient_manifest"
        ),
        "region_source_ready": region_source_ready,
        "region_source_blockers": pixel_blockers,
        "master_gradient_prior_path": master_gradient.get("path"),
        "master_gradient_prior_sha256": master_gradient.get("sha256"),
        "master_gradient_prior_shape": master_gradient.get("shape"),
        "pixel_gradient_prior_path": pixel_gradient.get("path"),
        "pixel_gradient_prior_sha256": pixel_gradient.get("sha256"),
        "pixel_gradient_prior_shape": pixel_gradient.get("shape"),
        "mathematical_basis": (
            "select pair and region atoms only from replayable, archive-bound gradient "
            "manifests before optimizing expected Delta S across P18/P19/P11 stacks"
        ),
        "contract_requirements": {
            "archive_bound_candidate_contract": True,
            "receiver_runtime_proof": True,
            "local_cpu_gate": True,
            "exact_auth_before_score_or_promotion": True,
        },
        "allowed_use": "queue_compilation_policy_or_migration_blocker",
        "forbidden_use": "score_claim_rank_kill_budget_spend_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        manifest,
        context="scorer_region_selector_cascade_selection_manifest",
    )
    return manifest


def build_scorer_region_selector_cascade_acquisition_policy(
    *,
    repo_root: str | Path,
    campaign_report: Mapping[str, Any],
    master_gradient_tensor_path: str | Path | None = DEFAULT_MASTER_GRADIENT_TENSOR_PATH,
    pixel_gradient_cache_path: str | Path | None = DEFAULT_PIXEL_GRADIENT_CACHE_PATH,
) -> dict[str, Any]:
    """Compile campaign learning into the next scorer-region acquisition policy."""

    require_no_truthy_authority_fields(
        campaign_report,
        context="scorer_region_selector_cascade_acquisition_policy:campaign_report",
    )
    rows_raw = campaign_report.get("rows")
    rows = [row for row in rows_raw if isinstance(row, Mapping)] if isinstance(rows_raw, list) else []
    aggregate = (
        campaign_report.get("aggregate_learning")
        if isinstance(campaign_report.get("aggregate_learning"), Mapping)
        else {}
    )
    local_cpu_rows = [row for row in rows if row.get("local_cpu_present") is True]
    passed_rows = [row for row in rows if row.get("candidate_passed_local_cpu_gate") is True]
    deltas = [delta for row in local_cpu_rows if (delta := _local_delta(row)) is not None]
    split_count = int(aggregate.get("mlx_positive_full_cpu_negative_split_count") or 0)
    observed_count = len(local_cpu_rows)
    split_rate = split_count / observed_count if observed_count else 0.0
    best_delta = min(deltas) if deltas else None
    hydration = build_hydration_contract(
        video_scope="receiver_closed_campaign_local_component_scope",
        scorer_axis="[macOS-CPU advisory]+[macOS-MLX research-signal]",
        archive_axis="current_cpu_frontier_family_candidate_archive",
        runtime_contract="inflate.sh_receiver_patch_output_change_plus_local_component_spot_checks",
        sample_count=observed_count,
    )
    action_rows = []
    for row in local_cpu_rows:
        saved = _saved_bytes(row)
        delta = _local_delta(row)
        if delta is None:
            continue
        action_rows.append(
            build_rate_distortion_action_row(
                candidate_id=str(row.get("variant_id") or ""),
                observed_net_delta_score_units=delta,
                saved_bytes=saved,
                local_cpu_score=_safe_number(row.get("local_cpu_canonical_score")),
                local_cpu_avg_segnet_dist=_safe_number(row.get("local_cpu_avg_segnet_dist")),
                local_cpu_avg_posenet_dist=_safe_number(row.get("local_cpu_avg_posenet_dist")),
                hydration=hydration,
            )
        )
    contest_action_functional = build_contest_space_action_functional(
        rows=action_rows,
        hydration=hydration,
    )
    master_gradient = _summarize_master_gradient_tensor(
        repo_root=repo_root,
        tensor_path=master_gradient_tensor_path,
    )
    pixel_gradient = _summarize_pixel_gradient_cache(
        repo_root=repo_root,
        cache_path=pixel_gradient_cache_path,
    )
    selection_manifest = _build_selection_manifest(
        master_gradient=master_gradient,
        pixel_gradient=pixel_gradient,
    )
    all_cpu_observed_failed = bool(local_cpu_rows) and not passed_rows
    if passed_rows:
        next_mode = "promote_gate_passed_rows_to_exact_readiness_bridge"
        family_status = "has_local_cpu_gate_survivor"
    elif all_cpu_observed_failed and split_count:
        next_mode = "vectorized_mlx_acquisition_then_cpu_gate_only"
        family_status = "current_sample_negative_with_mlx_cpu_split"
    elif all_cpu_observed_failed:
        next_mode = "new_basis_required_before_more_budget_spend"
        family_status = "current_sample_negative"
    else:
        next_mode = "complete_missing_local_cpu_gates"
        family_status = "incomplete_evidence"
    blockers = ordered_unique(
        [
            "exact_auth_eval_required_before_score_or_promotion_claim",
            *(
                ["current_operator_family_all_observed_local_cpu_rows_failed"]
                if all_cpu_observed_failed
                else []
            ),
            *(
                ["mlx_positive_full_cpu_negative_split_requires_calibration_or_demotion"]
                if split_count
                else []
            ),
            *(
                master_gradient.get("blockers", [])
                if isinstance(master_gradient.get("blockers"), list)
                else []
            ),
            *(
                pixel_gradient.get("blockers", [])
                if isinstance(pixel_gradient.get("blockers"), list)
                else []
            ),
        ]
    )
    payload = {
        "schema": SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA,
        "mathematical_action": {
            "formula": (
                "S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/"
                f"{CONTEST_RATE_DENOM_BYTES}"
            ),
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "empirical_budget_accounting": "observed_net_delta + saved_bytes*rate_score_per_byte",
            "optimization_direction": "minimize_expected_delta_s_under_receiver_proof_constraints",
            "contest_space_action_functional_schema": contest_action_functional["schema"],
        },
        "campaign_summary": {
            "variant_count": campaign_report.get("variant_count"),
            "completed_learning_variant_count": campaign_report.get(
                "completed_learning_variant_count"
            ),
            "local_cpu_observed_count": observed_count,
            "local_cpu_passed_gate_count": len(passed_rows),
            "mlx_positive_full_cpu_negative_split_count": split_count,
            "mlx_positive_full_cpu_negative_split_rate": split_rate,
            "best_local_cpu_delta_vs_auth_frontier": best_delta,
            "best_variant_id": campaign_report.get("best_variant_id"),
            "best_variant_selection_basis": campaign_report.get(
                "best_variant_selection_basis"
            ),
        },
        "empirical_dimension_effects": _dimension_effects(rows),
        "rate_credit_rows": action_rows,
        "contest_space_action_functional": contest_action_functional,
        "master_gradient_prior": master_gradient,
        "pixel_gradient_prior": pixel_gradient,
        "selection_manifest": selection_manifest,
        "next_queue_policy": {
            "mode": next_mode,
            "operator_family_status": family_status,
            "selection_manifest_ready": selection_manifest["selection_ready"],
            "selection_manifest_blockers": selection_manifest["selection_ready_blockers"],
            "local_cpu_gate_required": True,
            "post_cpu_mlx_authority_weight": 0.0 if split_count else None,
            "mlx_role": "broad_vectorized_acquisition_only",
            "queue_compilation": (
                "compile grouped PoseNet-null, SegNet-region, selector-codec, "
                "RGB/YUV-delta, and repack-order chains; do not rank isolated leaves"
            ),
            "preferred_next_grid": {
                "pose_pair_source": selection_manifest["pose_pair_source"],
                "region_source": selection_manifest["region_source"],
                "receiver_patch_delta_space": ["rgb"],
                "receiver_patch_max_pairs": [4, 8, 12],
                "receiver_patch_regions_per_pair": [1],
                "selector_codec_family_groups": [
                    ["fec10_adaptive_blend"],
                    ["fec8_markov_static_order1"],
                    ["fec10_adaptive_blend", "fec8_markov_static_order1"],
                ],
                "repack_order": ["p11_then_p15_then_receiver_patch"],
            },
            "parallel_execution": {
                "mlx": "single Metal device, vectorized candidate batches before CPU gate",
                "local_cpu": "full-sample spot checks only after acquisition filter",
                "exact_auth": "CPU only after local_cpu_delta<0 and eureka trigger; CUDA only after CPU clears",
            },
        },
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "next_campaign_acquisition_and_queue_compilation_policy",
        "forbidden_use": "score_claim_rank_kill_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="scorer_region_selector_cascade_acquisition_policy",
    )
    return payload


__all__ = [
    "SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA",
    "SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA",
    "SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA",
    "SCORER_REGION_SELECTOR_CASCADE_SELECTION_MANIFEST_SCHEMA",
    "ScorerRegionSelectorCascadeCampaignQueueError",
    "build_scorer_region_selector_cascade_acquisition_policy",
    "build_scorer_region_selector_cascade_campaign_queue",
    "build_scorer_region_selector_cascade_campaign_report",
    "discover_scorer_region_selector_cascade_variant_roots",
    "enumerate_cascade_variants",
]
