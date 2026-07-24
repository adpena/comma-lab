# SPDX-License-Identifier: MIT
"""Direct scorer-intrinsic completion of the DDM MS4 metric bundle.

This is a measurement compiler, not an actuator search.  It measures the
frozen-SegNet categorical Fisher quadratic on every exact occupied PF2 support
and emits authenticated exact-zero rows only where both the PF2 atlas and its
SHA-bound event index prove absence.  The 25 RG3 terminal residual blocks are
typed ``UNREACHABLE_BY_COUNTED_COORDINATES`` and retain their complete signed
probe custody.
"""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.ddm_lambda_continuation_frontier import publish_immutable_json
from tac.optimization.ddm_metric_custody_bundle import (
    BUNDLE_SCHEMA,
    COMPONENT_SCHEMA,
    COMPOSITE_R_DIRECT_DATA_SCHEMA,
    DIRECT_BLOCK_KEY_SHA256,
    DIRECT_EVENT_INDEX_RECEIPT_SHA256,
    DIRECT_EVENT_INDEX_SHA256,
    DIRECT_METRIC_MODE,
    DIRECT_RG3_SUMMARY_SHA256,
    DIRECT_SECANT_STATUS,
    DUAL_DIRECT_DATA_SCHEMA,
    EVIDENCE_AXIS,
    G3_REGISTRY_SCHEMA,
    HARD_PAIR_ORDER,
    PAIR_COUNT,
    PF2_ATLAS_SCHEMA,
    POINTER,
    POSE_DATA_SCHEMA,
    SCORER_BATCH_SIZE,
    SEG_DIRECT_DATA_SCHEMA,
    ComponentId,
    artifact_custody,
    load_metric_custody_bundle,
)
from tac.optimization.ddm_metric_producers import (
    direct_scorer_intrinsic_bucket_statistics,
    direct_scorer_intrinsic_pair_block,
    metric_tag,
)
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.solve_diff_operator_mining import (
    SolveDiffMiningConfigV1,
    _load_production_inputs,
    _open_production_inputs,
    realize_solve_camera,
)
from tac.repo_io import sha256_file
from tac.scorer import load_default_segnet

SCHEMA: Final = "ddm_ms4d_direct_completion.v1"
CONFIG_SCHEMA: Final = "ddm_ms4d_direct_completion_config.v1"
CHECKPOINT_SCHEMA: Final = "ddm_ms4d_direct_measurement_checkpoint.v1"
CHECKPOINT_MANIFEST_SCHEMA: Final = "ddm_ms4d_direct_measurement_manifest.v1"
CHUNK_SIZE: Final = 12
PIXELS_PER_PAIR: Final = 384 * 512
MIN_FREE_BYTES: Final = 2 * 1024**3


class MS4DCompletionError(ValueError):
    """Direct metric custody, geometry, or measurement differs."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _checked_json(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if sha256_file(resolved) != expected_sha256:
        raise MS4DCompletionError(f"SHA-256 differs: {resolved}")
    value = json.loads(resolved.read_bytes())
    if not isinstance(value, Mapping):
        raise MS4DCompletionError(f"JSON source must be an object: {resolved}")
    return value


def _raw_custody(
    path: str | Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    display = str(resolved)
    if repository_root is not None:
        try:
            display = str(resolved.relative_to(repository_root.resolve(strict=True)))
        except ValueError:
            pass
    return {
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _storage_preflight(path: Path) -> dict[str, Any]:
    parent = path.expanduser().resolve().parent
    if not str(parent).startswith(("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")):
        raise MS4DCompletionError("direct measurement checkpoints must use the governed SSD waterfall")
    free = shutil.disk_usage(parent).free
    if free < MIN_FREE_BYTES:
        raise MS4DCompletionError(f"storage preflight refused: free={free}, required={MIN_FREE_BYTES}")
    return {
        "status": "PASS",
        "tier": str(parent),
        "observed_free_bytes": free,
        "required_free_bytes": MIN_FREE_BYTES,
        "auto_cleanup": "NO_EPHEMERAL_BULK_CREATED; immutable JSON checkpoints preserved",
    }


def _false_authority(config: Mapping[str, Any]) -> None:
    required = {
        "schema": CONFIG_SCHEMA,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": EVIDENCE_AXIS,
        "main_landing_review_required": True,
        "torch_threads": 4,
        "pair_count": PAIR_COUNT,
        "chunk_size": CHUNK_SIZE,
        "metric_mode": DIRECT_METRIC_MODE,
    }
    drift = {key: (config.get(key), expected) for key, expected in required.items() if config.get(key) != expected}
    if drift:
        raise MS4DCompletionError(f"direct config false-authority/fixed contract differs: {drift}")


def _head_chart(weights_path: Path) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment gate
        raise MS4DCompletionError("safetensors is required for frozen-head custody") from exc
    with safe_open(str(weights_path), framework="numpy") as handle:
        weight = np.asarray(handle.get_tensor("segmentation_head.0.weight"), dtype=np.float32)
    if weight.shape != (5, 16, 3, 3):
        raise MS4DCompletionError(f"frozen SegNet head geometry differs: {weight.shape}")
    flattened = weight.reshape(5, -1).astype(np.float64)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-6))
    if rank != 4:
        raise MS4DCompletionError(f"frozen centered SegNet head rank differs: {rank}")
    basis = np.ascontiguousarray(vh[:4].T, dtype="<f8")
    reconstructed = centered @ basis @ basis.T
    normals = {
        (left, right): np.ascontiguousarray((flattened[left] - flattened[right]) @ basis, dtype=np.float64)
        for left in range(5)
        for right in range(left + 1, 5)
    }
    return normals, {
        "weights": _raw_custody(weights_path),
        "weight_shape": list(weight.shape),
        "centered_rank": rank,
        "singular_values": singular.tolist(),
        "quotient_basis_sha256_float64le": sha256(basis.tobytes()).hexdigest(),
        "rank4_reconstruction_max_abs_float64": float(np.max(np.abs(centered - reconstructed))),
        "canonical_equation": "segnet_head_rank4_linear_flipdist_v1",
        "coordinate_domain": "POST_R_PENULTIMATE_HEAD_QUOTIENT",
    }


def _support_slice(events: np.ndarray, pair_id: int) -> np.ndarray:
    lower = pair_id * PIXELS_PER_PAIR
    upper = (pair_id + 1) * PIXELS_PER_PAIR
    begin = int(np.searchsorted(events, lower, side="left"))
    end = int(np.searchsorted(events, upper, side="left"))
    return np.asarray(events[begin:end] - lower, dtype=np.uint32)


def _pair_class_normal(
    class_ids: Sequence[int],
    normals: Mapping[tuple[int, int], np.ndarray],
) -> np.ndarray:
    if len(class_ids) != 2:
        raise MS4DCompletionError("PF2 class-pair geometry differs")
    left, right = (int(class_ids[0]), int(class_ids[1]))
    key = (min(left, right), max(left, right))
    if key not in normals:
        raise MS4DCompletionError("PF2 class pair is outside the frozen five-class head")
    return normals[key] if left < right else -normals[key]


def _segnet_chunk(
    *,
    segnet: Any,
    solved_planes: np.ndarray,
    kernel: FullResizeKernel,
) -> tuple[np.ndarray, list[float]]:
    import torch

    logits_parts: list[np.ndarray] = []
    roundtrip: list[float] = []
    microbatch_size = 3
    for start in range(0, len(solved_planes), microbatch_size):
        stop = min(len(solved_planes), start + microbatch_size)
        cameras = np.stack(
            [
                np.stack(
                    [realize_solve_camera(pair[frame], kernel) for frame in range(2)],
                    axis=0,
                )
                for pair in solved_planes[start:stop]
            ],
            axis=0,
        )
        tensor = torch.from_numpy(cameras).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            realized = segnet.preprocess_input(tensor)
            logits = segnet(realized)
        observed = realized.permute(0, 2, 3, 1).cpu().numpy()
        for local_index in range(stop - start):
            rounded = np.clip(np.rint(observed[local_index]), 0, 255).astype(np.uint8)
            if not np.array_equal(rounded, solved_planes[start + local_index, 1]):
                raise MS4DCompletionError(
                    f"pair chunk index {start + local_index} failed exact #580 R roundtrip"
                )
            roundtrip.append(
                float(
                    np.max(
                        np.abs(
                            observed[local_index].astype(np.float64)
                            - solved_planes[start + local_index, 1]
                        )
                    )
                )
            )
        logits_parts.append(logits.cpu().numpy().astype(np.float32))
        del cameras, tensor, realized, logits, observed
    result = np.concatenate(logits_parts, axis=0)
    if result.shape != (len(solved_planes), 5, 384, 512) or not np.isfinite(result).all():
        raise MS4DCompletionError(f"frozen SegNet output geometry differs: {result.shape}")
    return np.ascontiguousarray(result), roundtrip


def _checkpoint_identity(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    try:
        config_display = str(config_path.resolve(strict=True).relative_to(repository_root))
    except ValueError:
        config_display = str(config_path.resolve(strict=True))
    return {
        "run_id": config["run_id"],
        "config_path": config_display,
        "config_sha256": sha256_file(config_path),
        "metric_mode": DIRECT_METRIC_MODE,
        "rg3_summary_sha256": DIRECT_RG3_SUMMARY_SHA256,
        "pf2_event_index_sha256": DIRECT_EVENT_INDEX_SHA256,
        "segnet_weights_sha256": config["segnet_weights_sha256"],
        "source_config_sha256": config["source_config_sha256"],
    }


def _checkpoint_identity_matches(
    saved: object,
    current: Mapping[str, Any],
) -> bool:
    """Accept an exact identity or the path-bound v1 form of the same config.

    The first local measurement recorded an absolute isolated-worktree config
    path.  Its config SHA and every scientific source hash are valid, but that
    transient prefix cannot survive MAIN landing.  Normalization is limited to
    an absolute path ending in the exact current repository-relative path; no
    hash or other identity field is relaxed.
    """

    if saved == current:
        return True
    if not isinstance(saved, Mapping):
        return False
    normalized = dict(saved)
    saved_path = normalized.get("config_path")
    current_path = current.get("config_path")
    if (
        not isinstance(saved_path, str)
        or not Path(saved_path).is_absolute()
        or not isinstance(current_path, str)
        or Path(current_path).is_absolute()
        or not saved_path.endswith(f"/{current_path}")
    ):
        return False
    normalized["config_path"] = current_path
    return normalized == dict(current)


def _measure_checkpoints(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    bulk_output: Path,
    atlas_rows: Sequence[Mapping[str, Any]],
    residual_rows: Sequence[Mapping[str, Any]],
    index_receipt: Mapping[str, Any],
    normals: Mapping[tuple[int, int], np.ndarray],
    head_chart: Mapping[str, Any],
    repository_root: Path,
) -> Path:
    source_config = SolveDiffMiningConfigV1.model_validate_json(
        Path(config["source_config_path"]).read_bytes()
    )
    kernel = FullResizeKernel.build()
    context = _open_production_inputs(source_config)
    import torch

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    segnet = load_default_segnet(config["upstream_dir"], device="cpu")
    bucket_arrays = index_receipt["bucket_arrays"]
    atlas_by_id = {str(row["bucket_id"]): row for row in atlas_rows}
    occupied_rows = {bucket_id: atlas_by_id[bucket_id] for bucket_id in bucket_arrays}
    residual_by_key = {
        (int(row["pair_id"]), str(row["bucket_id"])): row for row in residual_rows
    }
    checkpoints_root = bulk_output / "checkpoints"
    identity = _checkpoint_identity(
        config,
        config_path,
        repository_root=repository_root,
    )
    checkpoint_paths: list[Path] = []
    with np.load(config["pf2_event_index_path"], allow_pickle=False) as event_index:
        events_by_bucket = {
            str(bucket_id): np.asarray(event_index[array_key], dtype=np.uint32)
            for bucket_id, array_key in bucket_arrays.items()
        }
        for chunk_id, start in enumerate(range(0, PAIR_COUNT, CHUNK_SIZE)):
            pair_ids = list(range(start, min(PAIR_COUNT, start + CHUNK_SIZE)))
            checkpoint = checkpoints_root / f"chunk_{chunk_id:03d}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_bytes())
                if (
                    saved.get("schema") != CHECKPOINT_SCHEMA
                    or not _checkpoint_identity_matches(saved.get("identity"), identity)
                    or saved.get("chunk_id") != chunk_id
                    or saved.get("pair_ids") != pair_ids
                    or saved.get("status") != "MEASURED"
                ):
                    raise MS4DCompletionError(f"resume checkpoint identity differs: {checkpoint}")
                checkpoint_paths.append(checkpoint)
                continue
            chunk = _load_production_inputs(context, source_config, pair_ids, kernel)
            logits, roundtrip = _segnet_chunk(
                segnet=segnet,
                solved_planes=chunk.solved_planes,
                kernel=kernel,
            )
            rows: list[dict[str, Any]] = []
            direct_blocks: list[dict[str, Any]] = []
            for bucket_id, atlas_row in occupied_rows.items():
                left, right = (int(value) for value in atlas_row["class_ids"])
                normal = _pair_class_normal(atlas_row["class_ids"], normals)
                events = events_by_bucket[bucket_id]
                pair_statistics: list[dict[str, Any]] = []
                for local_index, pair_id in enumerate(pair_ids):
                    support = _support_slice(events, pair_id)
                    if support.size:
                        flat = logits[local_index].reshape(5, -1)
                        margins = (
                            flat[left, support.astype(np.intp)]
                            - flat[right, support.astype(np.intp)]
                        ).astype(np.float64)
                        absolute = np.abs(margins)
                        exponential = np.exp(-absolute)
                        fisher = 2.0 * exponential / np.square(1.0 + exponential)
                        fisher_mass = float(fisher.sum(dtype=np.float64))
                        fisher_margin_sum = float(np.dot(fisher, margins))
                        margin_sum = float(margins.sum(dtype=np.float64))
                    else:
                        margins = np.empty(0, dtype=np.float64)
                        fisher_mass = fisher_margin_sum = margin_sum = 0.0
                    pair_statistics.append(
                        {
                            "pair_id": pair_id,
                            "support_count": int(support.size),
                            "support_sha256_uint32le": sha256(
                                np.asarray(support, dtype="<u4").tobytes()
                            ).hexdigest(),
                            "margin_sha256_float32le": sha256(
                                np.asarray(margins, dtype="<f4").tobytes()
                            ).hexdigest(),
                            "fisher_mass": fisher_mass,
                            "fisher_margin_sum": fisher_margin_sum,
                            "margin_sum": margin_sum,
                        }
                    )
                    residual = residual_by_key.get((pair_id, bucket_id))
                    if residual is not None:
                        direct_blocks.append(
                            direct_scorer_intrinsic_pair_block(
                                pair_id=pair_id,
                                bucket_id=bucket_id,
                                head_pair_normal=normal,
                                margins=margins,
                                probe_custody=residual["rg3_probe_blocker"],
                            )
                        )
                rows.append({"bucket_id": bucket_id, "pair_statistics": pair_statistics})
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "identity": identity,
                "chunk_id": chunk_id,
                "pair_ids": pair_ids,
                "status": "MEASURED",
                "metric_mode": DIRECT_METRIC_MODE,
                "torch_threads": 4,
                "seed": int(config["seed"]),
                "deterministic_algorithms": True,
                "source_chunk_hashes": dict(chunk.source_hashes),
                "head_chart": dict(head_chart),
                "r_roundtrip_exact_uint8": True,
                "r_roundtrip_pre_round_max_abs": roundtrip,
                "rows": rows,
                "direct_blocks": direct_blocks,
                "evidence_axis": EVIDENCE_AXIS,
                "score_claim": False,
                "research_only": True,
                "main_landing_review_required": True,
            }
            publish_immutable_json(checkpoint, payload)
            checkpoint_paths.append(checkpoint)
            del chunk, logits, rows, direct_blocks, payload
            gc.collect()
    manifest = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "run_id": config["run_id"],
        "status": "COMPLETE",
        "metric_mode": DIRECT_METRIC_MODE,
        "pair_ids": list(range(PAIR_COUNT)),
        "pair_count": PAIR_COUNT,
        "chunk_size": CHUNK_SIZE,
        "torch_threads": 4,
        "seed": int(config["seed"]),
        "deterministic_algorithms": True,
        "per_stage_checkpointed_and_preserved": True,
        "resumable_from_disk": True,
        "storage_preflight": _storage_preflight(bulk_output),
        "checkpoints": [_raw_custody(path) for path in checkpoint_paths],
        "head_chart": dict(head_chart),
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    manifest_path = bulk_output / "STAGE-COMPLETE.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_bytes())
        stable_fields = (
            "schema",
            "run_id",
            "status",
            "metric_mode",
            "pair_ids",
            "pair_count",
            "chunk_size",
            "torch_threads",
            "seed",
            "deterministic_algorithms",
            "per_stage_checkpointed_and_preserved",
            "resumable_from_disk",
            "checkpoints",
            "head_chart",
            "evidence_axis",
            "score_claim",
            "research_only",
            "main_landing_review_required",
        )
        if any(existing.get(field) != manifest.get(field) for field in stable_fields):
            raise MS4DCompletionError("existing direct stage manifest differs on stable custody")
    else:
        publish_immutable_json(manifest_path, manifest)
    return manifest_path


def _merge_checkpoints(
    *,
    manifest_path: Path,
    atlas_rows: Sequence[Mapping[str, Any]],
    residual_rows: Sequence[Mapping[str, Any]],
    normals: Mapping[tuple[int, int], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_bytes())
    stats: dict[str, dict[str, list[Any]]] = {}
    direct_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for checkpoint_ref in manifest["checkpoints"]:
        checkpoint = json.loads(Path(checkpoint_ref["path"]).read_bytes())
        for row in checkpoint["rows"]:
            bucket_id = str(row["bucket_id"])
            target = stats.setdefault(
                bucket_id,
                {
                    "counts": [0] * PAIR_COUNT,
                    "fisher_mass": [0.0] * PAIR_COUNT,
                    "fisher_margin": [0.0] * PAIR_COUNT,
                    "margin": [0.0] * PAIR_COUNT,
                },
            )
            for pair in row["pair_statistics"]:
                pair_id = int(pair["pair_id"])
                target["counts"][pair_id] = int(pair["support_count"])
                target["fisher_mass"][pair_id] = float(pair["fisher_mass"])
                target["fisher_margin"][pair_id] = float(pair["fisher_margin_sum"])
                target["margin"][pair_id] = float(pair["margin_sum"])
        for block in checkpoint["direct_blocks"]:
            key = (int(block["pair_id"]), str(block["bucket_id"]))
            if key in direct_by_key:
                raise MS4DCompletionError(f"duplicate direct residual block {key}")
            normalized = dict(block)
            fisher = np.asarray(normalized["fisher_vector"], dtype=np.float64)
            euclidean = np.asarray(normalized["euclidean_control_vector"], dtype=np.float64)
            fisher_norm = float(np.linalg.norm(fisher))
            euclidean_norm = float(np.linalg.norm(euclidean))
            if fisher_norm > 0.0 and euclidean_norm > 0.0:
                normalized["fisher_euclidean_cosine"] = float(
                    np.clip(
                        np.dot(fisher, euclidean) / (fisher_norm * euclidean_norm),
                        -1.0,
                        1.0,
                    )
                )
                normalized["fisher_to_euclidean_rel_norm"] = fisher_norm / euclidean_norm
                normalized["diagnostic_status"] = "MEASURED_NONDEGENERATE"
            direct_by_key[key] = normalized
    expected_keys = [(int(row["pair_id"]), str(row["bucket_id"])) for row in residual_rows]
    if set(direct_by_key) != set(expected_keys):
        raise MS4DCompletionError("merged direct residual set differs from exact RG3 25")
    direct_blocks = [direct_by_key[key] for key in expected_keys]
    seg_rows: list[dict[str, Any]] = []
    composite_rows: list[dict[str, Any]] = []
    dual_rows: list[dict[str, Any]] = []
    zeros_i = [0] * PAIR_COUNT
    zeros_f = [0.0] * PAIR_COUNT
    for atlas_row in atlas_rows:
        bucket_id = str(atlas_row["bucket_id"])
        source = stats.get(bucket_id)
        seg, composite, dual = direct_scorer_intrinsic_bucket_statistics(
            atlas_row,
            head_pair_normal=_pair_class_normal(atlas_row["class_ids"], normals),
            pair_support_counts=source["counts"] if source else zeros_i,
            fisher_mass_by_pair=source["fisher_mass"] if source else zeros_f,
            fisher_margin_sum_by_pair=source["fisher_margin"] if source else zeros_f,
            margin_sum_by_pair=source["margin"] if source else zeros_f,
        )
        seg_rows.append(seg)
        composite_rows.append(composite)
        dual_rows.append(dual)
    return seg_rows, composite_rows, dual_rows, direct_blocks


def _component_receipt(
    *,
    component_id: ComponentId,
    data: Mapping[str, Any],
    data_path: Path,
    lineage: Sequence[Mapping[str, Any]],
    repository_root: Path,
) -> dict[str, Any]:
    homes = {
        ComponentId.SEG_METRIC: (StreamType.SKELETON, LayerHome.L4_SCORER_FEATURE),
        ComponentId.POSE_METRIC: (StreamType.FIBER, LayerHome.L5_VERDICT),
        ComponentId.COMPOSITE_R_SECOND_ORDER: (StreamType.CONNECTION, LayerHome.L4_SCORER_FEATURE),
        ComponentId.DUAL_METRIC_DIAGNOSTICS: (StreamType.RESIDUAL, LayerHome.L5_VERDICT),
    }
    stream, layer = homes[component_id]
    return {
        "schema": COMPONENT_SCHEMA,
        "component_id": component_id.value,
        "status": "COMPLETE",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "sample_count": PAIR_COUNT,
        "scorer_batch_size": SCORER_BATCH_SIZE,
        "input_lineage": list(lineage),
        "data_artifact": artifact_custody(
            data_path,
            repository_root=repository_root,
            role=f"{component_id.value.lower()}_data",
            content_schema=str(data["schema"]),
        ).to_dict(),
        "blockers": [],
        "next_measurement": "MAIN must independently review this exact complete measurement before consumer use.",
        "typed_stream_tags": [metric_tag(stream, layer)],
        "main_landing_review_required": True,
    }


def materialize(
    config_path: str | Path,
    *,
    bulk_output: str | Path,
    receipt_output: str | Path,
    repository_root: str | Path,
) -> Mapping[str, Any]:
    """Run/resume n600 direct measurement and materialize a strict COMPLETE bundle."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    config_file = Path(config_path).expanduser().resolve(strict=True)
    config = json.loads(config_file.read_bytes())
    if not isinstance(config, Mapping):
        raise MS4DCompletionError("direct completion config must be an object")
    _false_authority(config)
    checked_paths = (
        ("pf2_path", "pf2_sha256"),
        ("g3_registry_path", "g3_registry_sha256"),
        ("rg3_summary_path", "rg3_summary_sha256"),
        ("pf2_event_index_receipt_path", "pf2_event_index_receipt_sha256"),
        ("pf2_event_index_path", "pf2_event_index_sha256"),
        ("source_config_path", "source_config_sha256"),
        ("segnet_weights_path", "segnet_weights_sha256"),
        ("upstream_modules_path", "upstream_modules_sha256"),
        ("pose_data_path", "pose_data_sha256"),
    )
    config = dict(config)
    for path_key, sha_key in checked_paths:
        candidate = Path(config[path_key]).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        config[path_key] = str(candidate.resolve(strict=True))
        if sha256_file(config[path_key]) != config[sha_key]:
            raise MS4DCompletionError(f"{path_key} SHA-256 differs")
    upstream_dir = Path(config["upstream_dir"]).expanduser()
    if not upstream_dir.is_absolute():
        upstream_dir = root / upstream_dir
    config["upstream_dir"] = str(upstream_dir.resolve(strict=True))
    if (
        config["pf2_sha256"] != "85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73"
        or config["g3_registry_sha256"]
        != "0c9ce6d0ce2b2c0830400f096438355242527d40f682fc1b201f67d8d951a4e4"
        or config["rg3_summary_sha256"] != DIRECT_RG3_SUMMARY_SHA256
        or config["pf2_event_index_receipt_sha256"] != DIRECT_EVENT_INDEX_RECEIPT_SHA256
        or config["pf2_event_index_sha256"] != DIRECT_EVENT_INDEX_SHA256
    ):
        raise MS4DCompletionError("direct completion sealed source hashes differ")
    bulk = Path(bulk_output).expanduser()
    receipts = Path(receipt_output).expanduser()
    _storage_preflight(bulk)
    bulk.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    pf2 = _checked_json(config["pf2_path"], config["pf2_sha256"])
    atlas_rows = pf2.get("typed_split_atlas", {}).get("rows")
    if not isinstance(atlas_rows, list) or len(atlas_rows) != 1200:
        raise MS4DCompletionError("PF2 atlas does not contain exactly 1,200 rows")
    rg3 = _checked_json(config["rg3_summary_path"], config["rg3_summary_sha256"])
    residual_rows = rg3.get("receiver_coordinate_derivation", {}).get("residual")
    missing = rg3.get("g3_top24_coverage", {}).get("missing_blocks")
    if (
        not isinstance(residual_rows, list)
        or len(residual_rows) != 25
        or [
            {"pair_id": row.get("pair_id"), "bucket_id": row.get("bucket_id")}
            for row in residual_rows
        ]
        != missing
        or sha256(
            _canonical_bytes(
                [
                    {"pair_id": row["pair_id"], "bucket_id": row["bucket_id"]}
                    for row in residual_rows
                ]
            )
        ).hexdigest()
        != DIRECT_BLOCK_KEY_SHA256
    ):
        raise MS4DCompletionError("RG3 exact 25-row residual contract differs")
    index_receipt = _checked_json(
        config["pf2_event_index_receipt_path"],
        config["pf2_event_index_receipt_sha256"],
    )
    if (
        index_receipt.get("index_sha256") != config["pf2_event_index_sha256"]
        or index_receipt.get("pf2_receipt_sha256") != config["pf2_sha256"]
        or index_receipt.get("occupied_bucket_count") != 37
    ):
        raise MS4DCompletionError("PF2 event-index receipt binding differs")
    normals, head_chart = _head_chart(Path(config["segnet_weights_path"]))
    manifest_path = _measure_checkpoints(
        config=config,
        config_path=config_file,
        bulk_output=bulk,
        atlas_rows=atlas_rows,
        residual_rows=residual_rows,
        index_receipt=index_receipt,
        normals=normals,
        head_chart=head_chart,
        repository_root=root,
    )
    seg_rows, composite_rows, dual_rows, direct_blocks = _merge_checkpoints(
        manifest_path=manifest_path,
        atlas_rows=atlas_rows,
        residual_rows=residual_rows,
        normals=normals,
    )
    common_sources = {
        "rg3_residual_summary": _raw_custody(
            config["rg3_summary_path"],
            repository_root=root,
        ),
        "pf2_event_index_receipt": _raw_custody(config["pf2_event_index_receipt_path"]),
        "pf2_event_index": _raw_custody(config["pf2_event_index_path"]),
        "measurement_checkpoint_manifest": _raw_custody(manifest_path),
    }
    common = {
        "pf2_atlas_sha256": config["pf2_sha256"],
        "g3_hard_pair_registry_sha256": config["g3_registry_sha256"],
        "measurement_schedule": list(HARD_PAIR_ORDER),
        "pair_count": PAIR_COUNT,
        "scorer_batch_size": SCORER_BATCH_SIZE,
        "metric_mode": DIRECT_METRIC_MODE,
        "direct_block_count": 25,
        "direct_block_key_sha256": DIRECT_BLOCK_KEY_SHA256,
        "direct_blocks": direct_blocks,
        "direct_source_custody": common_sources,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    seg_data = {
        "schema": SEG_DIRECT_DATA_SCHEMA,
        **common,
        "head_rank": 4,
        "metric_id": "MARGIN_FISHER_RANK4_DIRECT_SCORER_INTRINSIC",
        "head_chart": head_chart,
        "rows": seg_rows,
    }
    seg_path = receipts / "seg_metric_direct_n600.json"
    publish_immutable_json(seg_path, seg_data)
    composite_sources = {
        **common_sources,
        "seg_metric_data": _raw_custody(seg_path, repository_root=root),
    }
    composite_data = {
        "schema": COMPOSITE_R_DIRECT_DATA_SCHEMA,
        **common,
        "direct_source_custody": composite_sources,
        "kernel_binding": "separable_resize_full_kernel_direct_sum_v1",
        "paired_secant_pattern": DIRECT_SECANT_STATUS,
        "coordinate_domain": "POST_R_PENULTIMATE_HEAD_QUOTIENT",
        "rows": composite_rows,
    }
    composite_path = receipts / "composite_r_direct_n600.json"
    publish_immutable_json(composite_path, composite_data)
    dual_data = {
        "schema": DUAL_DIRECT_DATA_SCHEMA,
        **common,
        "primary_metric": "MARGIN_FISHER",
        "control_metric": "EUCLIDEAN_CONTROL_ONLY",
        "rows": dual_rows,
    }
    dual_path = receipts / "dual_metric_direct_n600.json"
    publish_immutable_json(dual_path, dual_data)
    pose_path = Path(config["pose_data_path"]).resolve(strict=True)
    pose_data = json.loads(pose_path.read_bytes())
    if pose_data.get("schema") != POSE_DATA_SCHEMA:
        raise MS4DCompletionError("reused Pose component schema differs")

    atlas_ref = artifact_custody(
        Path(config["pf2_path"]),
        repository_root=root,
        role="pf2_typed_atlas",
        content_schema=PF2_ATLAS_SCHEMA,
    ).to_dict()
    g3_ref = artifact_custody(
        Path(config["g3_registry_path"]),
        repository_root=root,
        role="g3_hard_pair_registry",
        content_schema=G3_REGISTRY_SCHEMA,
    ).to_dict()
    rg3_ref = artifact_custody(
        Path(config["rg3_summary_path"]),
        repository_root=root,
        role="rg3_terminal_residual_summary",
        content_schema=str(rg3["schema"]),
    ).to_dict()
    index_receipt_ref = artifact_custody(
        Path(config["pf2_event_index_receipt_path"]),
        repository_root=root,
        role="pf2_event_index_receipt",
        content_schema=str(index_receipt["schema"]),
    ).to_dict()
    manifest_ref = artifact_custody(
        manifest_path,
        repository_root=root,
        role="direct_measurement_checkpoint_manifest",
        content_schema=CHECKPOINT_MANIFEST_SCHEMA,
    ).to_dict()
    lineage = [atlas_ref, g3_ref, rg3_ref, index_receipt_ref, manifest_ref]
    component_data = {
        ComponentId.SEG_METRIC: (seg_data, seg_path),
        ComponentId.POSE_METRIC: (pose_data, pose_path),
        ComponentId.COMPOSITE_R_SECOND_ORDER: (composite_data, composite_path),
        ComponentId.DUAL_METRIC_DIAGNOSTICS: (dual_data, dual_path),
    }
    component_refs: dict[str, Any] = {}
    for component_id, (data, data_path) in component_data.items():
        component = _component_receipt(
            component_id=component_id,
            data=data,
            data_path=data_path,
            lineage=lineage,
            repository_root=root,
        )
        component_path = receipts / f"{component_id.value.lower()}_receipt.json"
        publish_immutable_json(component_path, component)
        component_refs[component_id.value] = artifact_custody(
            component_path,
            repository_root=root,
            role=f"{component_id.value.lower()}_component_receipt",
            content_schema=COMPONENT_SCHEMA,
        ).to_dict()
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": config["run_id"],
        "status": "COMPLETE",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "pointer": POINTER,
        "pointer_moved": False,
        "pf2_atlas": atlas_ref,
        "g3_hard_pair_registry": g3_ref,
        "component_receipts": component_refs,
        "hard_pair_order": list(HARD_PAIR_ORDER),
        "consumers": [
            "ms2_typed_quotient_solve",
            "pf2r_metric_active_three_formulation",
            "rd1_dimension_duals",
        ],
        "blockers": [],
        "headline_admissibility": {
            "bundle_complete": True,
            "scorer_metric_active": True,
            "pose_tube_active": True,
            "score_claim": False,
        },
        "main_landing_review_required": True,
    }
    bundle_path = receipts / "BUNDLE-COMPLETE.json"
    publish_immutable_json(bundle_path, manifest)
    loaded = load_metric_custody_bundle(
        bundle_path,
        repository_root=root,
        require_complete=True,
    )
    if not loaded.complete:
        raise MS4DCompletionError("strict loader did not admit the complete direct bundle")
    receipt = {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "bundle": _raw_custody(bundle_path, repository_root=root),
        "strict_loader_require_complete": True,
        "strict_loader_admitted": True,
        "direct_block_count": len(direct_blocks),
        "unreachable_by_counted_coordinates_count": sum(
            block["actuation_status"] == "UNREACHABLE_BY_COUNTED_COORDINATES"
            for block in direct_blocks
        ),
        "occupied_bucket_count": sum(row["event_count"] > 0 for row in seg_rows),
        "exact_empty_bucket_count": sum(row["event_count"] == 0 for row in seg_rows),
        "direct_event_count": sum(row["event_count"] for row in seg_rows),
        "pointer": POINTER,
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    receipt_path = receipts / "receipt.json"
    publish_immutable_json(receipt_path, receipt)
    return receipt


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "SCHEMA",
    "MS4DCompletionError",
    "materialize",
]
