# SPDX-License-Identifier: MIT
"""Campaign queues for grouped P19/P18/P11/P15 scorer-region cascades."""

from __future__ import annotations

import itertools
import json
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
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.scorer_region_operator_contract import (
    build_scorer_region_operator_contract,
)
from tac.repo_io import sha256_file

SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA = (
    "scorer_region_selector_cascade_campaign_queue_metadata.v1"
)
SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA = (
    "scorer_region_selector_cascade_campaign_report.v1"
)

SUPPORTED_REPACK_ORDERS = frozenset({"p11_then_p15_then_receiver_patch"})


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


__all__ = [
    "SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA",
    "SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA",
    "ScorerRegionSelectorCascadeCampaignQueueError",
    "build_scorer_region_selector_cascade_campaign_queue",
    "build_scorer_region_selector_cascade_campaign_report",
    "enumerate_cascade_variants",
]
