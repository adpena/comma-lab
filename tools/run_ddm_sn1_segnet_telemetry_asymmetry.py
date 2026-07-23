#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run score-neutral SegNet telemetry and sided-boundary analysis on real n600.

The tool consumes the SHA-pinned ZIP_STORED cache of official-video frames so
it never duplicates the 5 GB source cache.  Batch margin arrays and sparse
inverse deltas live on the approved SSD tier; small canonical receipts live in
``.omx/research``.  Existing valid batch checkpoints are consumed by SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.analysis.segnet_internal_telemetry import (  # noqa: E402
    CLASS_NAMES,
    SegNetInternalTelemetry,
    SegNetTelemetryPolicy,
    assert_telemetry_argmax_identity,
    extract_ordered_pair_boundary_samples,
    measure_erf_response,
)
from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
    stored_npy_memmap,
)
from tac.optimization.ddm_dv2_sdwl1 import canonical_json_bytes  # noqa: E402
from tac.optimization.ddm_sn1_sided_tolerance import (  # noqa: E402
    TEMPORAL_STRATA,
    SidedToleranceRow,
    build_header,
    build_sided_rows,
    export_e1_bounds,
    export_jsonl,
    orientation,
    parse_jsonl,
)

CONFIG_SCHEMA: Final = "DDMSN1SegNetTelemetryAsymmetryConfigV1"
RUN_SCHEMA: Final = "ddm_sn1_segnet_telemetry_asymmetry_receipt.v1"
BATCH_SCHEMA: Final = "ddm_sn1_segnet_telemetry_asymmetry_batch.v1"
PROGRESS_SCHEMA: Final = "ddm_sn1_segnet_telemetry_asymmetry_progress.v1"
TELEMETRY_HEADER_SCHEMA: Final = "ddm_sn1_segnet_telemetry.header.v1"
TELEMETRY_FRAME_SCHEMA: Final = "ddm_sn1_segnet_telemetry.frame.v1"
INVERSE_SCHEMA: Final = "ddm_sn1_segnet_sided_inverse_demo.v1"
AXIS: Final = "[macOS-CPU frozen-SegNet advisory]"
VIDEO_BYTES: Final = 37_545_489
N600: Final = 600
SEG_HW: Final = (384, 512)
CAMERA_HW: Final = (874, 1164)
EXPECTED_CACHE_SHAPES: Final = {
    "gt_f1.npy": ((600, 874, 1164, 3), np.dtype(np.uint8)),
    "lstars.npy": ((600, 384, 512), np.dtype(np.int64)),
}
APPROVED_SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
IMPLEMENTATION_FILES: Final = {
    "runner": REPO / "tools" / "run_ddm_sn1_segnet_telemetry_asymmetry.py",
    "telemetry": REPO / "src" / "tac" / "analysis" / "segnet_internal_telemetry.py",
    "sided_tolerance": (
        REPO / "src" / "tac" / "optimization" / "ddm_sn1_sided_tolerance.py"
    ),
}
_CONFIG_KEYS: Final = {
    "schema",
    "run_id",
    "video_path",
    "video_sha256",
    "gt_cache_path",
    "gt_cache_sha256",
    "upstream_root",
    "upstream_modules_sha256",
    "segnet_weights_sha256",
    "output_directory",
    "scratch_directory",
    "pair_count",
    "batch_pairs",
    "torch_threads",
    "seed",
    "erf_pair_ids",
    "inverse_segment_count",
    "inverse_max_steps",
    "inverse_max_linf",
    "inverse_camera_radius",
    "checkpoint_policy",
    "research_only",
    "score_claim",
    "promotion_eligible",
}


class DDMSN1Error(RuntimeError):
    """Raised when source custody, checkpoint, or measured invariants fail."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    schema: str
    run_id: str
    video_path: Path
    video_sha256: str
    gt_cache_path: Path
    gt_cache_sha256: str
    upstream_root: Path
    upstream_modules_sha256: str
    segnet_weights_sha256: str
    output_directory: Path
    scratch_directory: Path
    pair_count: int
    batch_pairs: int
    torch_threads: int
    seed: int
    erf_pair_ids: tuple[int, ...]
    inverse_segment_count: int
    inverse_max_steps: int
    inverse_max_linf: int
    inverse_camera_radius: int
    checkpoint_policy: str
    research_only: bool
    score_claim: bool
    promotion_eligible: bool


def _lower_sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise DDMSN1Error(f"{name} must be a lowercase SHA-256")
    return value


def _strict_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DDMSN1Error(f"{name} must be a positive integer")
    return value


def load_config(path: Path) -> tuple[RunConfig, str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DDMSN1Error("config is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != _CONFIG_KEYS:
        raise DDMSN1Error("typed config keys are malformed")
    if value["schema"] != CONFIG_SCHEMA:
        raise DDMSN1Error("typed config schema is unknown")
    if value["research_only"] is not True or value["score_claim"] is not False:
        raise DDMSN1Error("SN1 is research-only and cannot carry a score claim")
    if value["promotion_eligible"] is not False:
        raise DDMSN1Error("SN1 local telemetry cannot be promotion eligible")
    pair_count = _strict_positive_int(value["pair_count"], name="pair_count")
    if pair_count != N600:
        raise DDMSN1Error(f"official-video authority requires n600, got {pair_count}")
    batch_pairs = _strict_positive_int(value["batch_pairs"], name="batch_pairs")
    if batch_pairs > 16:
        raise DDMSN1Error("canonical frozen-scorer batches must be <=16")
    erf_pair_ids = value["erf_pair_ids"]
    if (
        not isinstance(erf_pair_ids, list)
        or not erf_pair_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < N600
            for item in erf_pair_ids
        )
        or len(set(erf_pair_ids)) != len(erf_pair_ids)
    ):
        raise DDMSN1Error("erf_pair_ids must be unique valid n600 pair ids")
    config = RunConfig(
        schema=CONFIG_SCHEMA,
        run_id=str(value["run_id"]),
        video_path=Path(value["video_path"]).expanduser().resolve(),
        video_sha256=_lower_sha(value["video_sha256"], name="video_sha256"),
        gt_cache_path=Path(value["gt_cache_path"]).expanduser().resolve(),
        gt_cache_sha256=_lower_sha(value["gt_cache_sha256"], name="gt_cache_sha256"),
        upstream_root=Path(value["upstream_root"]).expanduser().resolve(),
        upstream_modules_sha256=_lower_sha(
            value["upstream_modules_sha256"],
            name="upstream_modules_sha256",
        ),
        segnet_weights_sha256=_lower_sha(
            value["segnet_weights_sha256"],
            name="segnet_weights_sha256",
        ),
        output_directory=Path(value["output_directory"]).expanduser().resolve(),
        scratch_directory=Path(value["scratch_directory"]).expanduser().resolve(),
        pair_count=pair_count,
        batch_pairs=batch_pairs,
        torch_threads=_strict_positive_int(value["torch_threads"], name="torch_threads"),
        seed=int(value["seed"]),
        erf_pair_ids=tuple(erf_pair_ids),
        inverse_segment_count=_strict_positive_int(
            value["inverse_segment_count"],
            name="inverse_segment_count",
        ),
        inverse_max_steps=_strict_positive_int(
            value["inverse_max_steps"],
            name="inverse_max_steps",
        ),
        inverse_max_linf=_strict_positive_int(
            value["inverse_max_linf"],
            name="inverse_max_linf",
        ),
        inverse_camera_radius=_strict_positive_int(
            value["inverse_camera_radius"],
            name="inverse_camera_radius",
        ),
        checkpoint_policy=str(value["checkpoint_policy"]),
        research_only=True,
        score_claim=False,
        promotion_eligible=False,
    )
    if not config.run_id.strip():
        raise DDMSN1Error("run_id cannot be empty")
    if config.checkpoint_policy != "atomic_preserve_each_batch_resume_by_sha":
        raise DDMSN1Error("checkpoint policy must preserve each batch and resume by SHA")
    if config.output_directory == Path("/tmp") or Path("/tmp") in config.output_directory.parents:
        raise DDMSN1Error("operator-facing evidence cannot live under /tmp")
    if config.scratch_directory == Path("/tmp") or Path("/tmp") in config.scratch_directory.parents:
        raise DDMSN1Error("resumable margin/delta scratch cannot live under /tmp")
    config_sha = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return config, config_sha


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _path_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def implementation_identity() -> dict[str, Any]:
    """Bind resumable checkpoints to the exact analysis implementation."""

    files = {name: _path_identity(path) for name, path in IMPLEMENTATION_FILES.items()}
    return {
        "schema": "ddm_sn1_implementation_identity.v1",
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def validate_source_custody(config: RunConfig) -> dict[str, Any]:
    paths = {
        "video": config.video_path,
        "gt_cache": config.gt_cache_path,
        "upstream_modules": config.upstream_root / "modules.py",
        "segnet_weights": config.upstream_root / "models" / "segnet.safetensors",
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise DDMSN1Error(f"source custody files are absent: {missing}")
    expected = {
        "video": config.video_sha256,
        "gt_cache": config.gt_cache_sha256,
        "upstream_modules": config.upstream_modules_sha256,
        "segnet_weights": config.segnet_weights_sha256,
    }
    identities = {name: _path_identity(path) for name, path in paths.items()}
    for name, identity in identities.items():
        if identity["sha256"] != expected[name]:
            raise DDMSN1Error(f"{name} SHA drift: {identity['sha256']} != {expected[name]}")
    if identities["video"]["bytes"] != VIDEO_BYTES:
        raise DDMSN1Error("official video byte count drifted")
    return identities


def storage_preflight(config: RunConfig) -> dict[str, Any]:
    config.scratch_directory.mkdir(parents=True, exist_ok=True)
    resolved = config.scratch_directory.resolve()
    matching = [
        root
        for root in APPROVED_SSD_ROOTS
        if root == resolved or root in resolved.parents
    ]
    if not matching:
        raise DDMSN1Error("scratch directory is not on the approved SSD waterfall")
    first_existing = next((root for root in APPROVED_SSD_ROOTS if root.exists()), None)
    if first_existing is None or (
        first_existing not in resolved.parents and resolved != first_existing
    ):
        raise DDMSN1Error("scratch directory does not use the first available SSD tier")
    usage = shutil.disk_usage(resolved)
    required = 512 * 1024 * 1024
    if usage.free < required:
        raise DDMSN1Error("SSD storage preflight failed")
    return {
        "status": "PASS",
        "selected_root": str(first_existing),
        "scratch_directory": str(resolved),
        "required_free_bytes": required,
        "observed_free_bytes": usage.free,
        "waterfall_order": [str(root) for root in APPROVED_SSD_ROOTS],
        "large_artifact_policy": "CERTIFY_OR_BLOCK",
        "automatic_cleanup": (
            "batch margin NPZ and sparse inverse deltas are preserved resumable evidence; "
            "no deletion is authorized by this analysis"
        ),
    }


def open_target_cache(path: Path) -> dict[str, np.memmap]:
    arrays = {
        name: stored_npy_memmap(path, name)
        for name in EXPECTED_CACHE_SHAPES
    }
    for name, value in arrays.items():
        shape, dtype = EXPECTED_CACHE_SHAPES[name]
        if value.shape != shape or value.dtype != dtype:
            raise DDMSN1Error(
                f"{name} custody mismatch: shape={value.shape}, dtype={value.dtype}"
            )
    return arrays


def load_frozen_segnet(config: RunConfig) -> Any:
    import torch
    from safetensors.torch import load_file

    if str(config.upstream_root) not in sys.path:
        sys.path.insert(0, str(config.upstream_root))
    from modules import SegNet

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)
    model = SegNet().eval().cpu()
    weights = config.upstream_root / "models" / "segnet.safetensors"
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _model_input(model: Any, camera_frame: np.ndarray) -> Any:
    import torch

    writable = np.array(camera_frame, dtype=np.uint8, order="C", copy=True)
    frame = torch.from_numpy(writable).permute(2, 0, 1)
    pair = frame[None, None].expand(1, 2, -1, -1, -1).to(dtype=torch.float32)
    return model.preprocess_input(pair)


def exact_head_norms(model: Any) -> dict[str, float]:
    weight = model.segmentation_head[0].weight.detach().cpu().numpy()
    flattened = weight.reshape(len(CLASS_NAMES), -1).astype(np.float64)
    norms: dict[str, float] = {}
    for winner in range(len(CLASS_NAMES)):
        for rival in range(len(CLASS_NAMES)):
            if winner != rival:
                norms[orientation(winner, rival)] = float(
                    np.linalg.norm(flattened[winner] - flattened[rival])
                )
    return norms


def _empty_accumulator() -> dict[str, Any]:
    return {
        "frame_count": 0,
        "per_class_logit_energy_sum": dict.fromkeys(CLASS_NAMES, 0.0),
        "layers": {},
    }


def _accumulate_summary(accumulator: dict[str, Any], summary: Mapping[str, Any]) -> None:
    accumulator["frame_count"] += 1
    for name, value in summary["per_class_logit_energy"].items():
        accumulator["per_class_logit_energy_sum"][name] += float(value)
    for name, row in summary["layer_boundary_energy"].items():
        if not row["present"]:
            accumulator["layers"].setdefault(
                name,
                {"present": False, "reason": row["reason"]},
            )
            continue
        target = accumulator["layers"].setdefault(
            name,
            {
                "present": True,
                "shape": row["shape"],
                "boundary_sample_count": 0,
                "interior_sample_count": 0,
                "boundary_energy_sum": 0.0,
                "interior_energy_sum": 0.0,
            },
        )
        if target["shape"] != row["shape"]:
            raise DDMSN1Error(f"telemetry shape drift at {name}")
        boundary_count = int(row["boundary_sample_count"])
        interior_count = int(row["interior_sample_count"])
        target["boundary_sample_count"] += boundary_count
        target["interior_sample_count"] += interior_count
        if row["boundary_mean_square"] is not None:
            target["boundary_energy_sum"] += (
                float(row["boundary_mean_square"]) * boundary_count
            )
        if row["interior_mean_square"] is not None:
            target["interior_energy_sum"] += (
                float(row["interior_mean_square"]) * interior_count
            )


def _merge_accumulator(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    target["frame_count"] += int(source["frame_count"])
    for name in CLASS_NAMES:
        target["per_class_logit_energy_sum"][name] += float(
            source["per_class_logit_energy_sum"][name]
        )
    for name, source_row in source["layers"].items():
        if not source_row["present"]:
            target["layers"].setdefault(name, dict(source_row))
            continue
        target_row = target["layers"].setdefault(
            name,
            {
                "present": True,
                "shape": source_row["shape"],
                "boundary_sample_count": 0,
                "interior_sample_count": 0,
                "boundary_energy_sum": 0.0,
                "interior_energy_sum": 0.0,
            },
        )
        if target_row["shape"] != source_row["shape"]:
            raise DDMSN1Error(f"checkpoint telemetry shape drift at {name}")
        for key in (
            "boundary_sample_count",
            "interior_sample_count",
            "boundary_energy_sum",
            "interior_energy_sum",
        ):
            target_row[key] += source_row[key]


def _finalize_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    frame_count = int(accumulator["frame_count"])
    if frame_count != N600:
        raise DDMSN1Error(f"final telemetry must cover n600, got {frame_count}")
    layers: dict[str, Any] = {}
    for name, row in accumulator["layers"].items():
        if not row["present"]:
            layers[name] = dict(row)
            continue
        boundary_count = int(row["boundary_sample_count"])
        interior_count = int(row["interior_sample_count"])
        boundary_mean = (
            float(row["boundary_energy_sum"] / boundary_count)
            if boundary_count
            else None
        )
        interior_mean = (
            float(row["interior_energy_sum"] / interior_count)
            if interior_count
            else None
        )
        layers[name] = {
            "present": True,
            "shape": row["shape"],
            "boundary_sample_count": boundary_count,
            "interior_sample_count": interior_count,
            "boundary_mean_square": boundary_mean,
            "interior_mean_square": interior_mean,
            "boundary_to_interior_energy_ratio": (
                boundary_mean / interior_mean
                if boundary_mean is not None
                and interior_mean is not None
                and interior_mean > 0.0
                else None
            ),
        }
    return {
        "schema": "ddm_sn1_segnet_telemetry.aggregate.v1",
        "frame_count": frame_count,
        "per_class_logit_energy": {
            name: float(accumulator["per_class_logit_energy_sum"][name] / frame_count)
            for name in CLASS_NAMES
        },
        "layer_boundary_energy": layers,
    }


def _component_subset(
    component: np.ndarray,
    margins: np.ndarray,
    *,
    limit: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a connected BFS prefix rooted at the component's lowest margin."""

    coordinate_to_margin = {
        (int(y), int(x)): float(margin)
        for (y, x), margin in zip(component, margins, strict=True)
    }
    seed = min(coordinate_to_margin, key=coordinate_to_margin.get)
    chosen: list[tuple[int, int]] = []
    queue = [seed]
    seen = {seed}
    while queue and len(chosen) < limit:
        current = queue.pop(0)
        chosen.append(current)
        y, x = current
        neighbors = ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
        valid = [
            item
            for item in neighbors
            if item in coordinate_to_margin and item not in seen
        ]
        valid.sort(key=coordinate_to_margin.get)
        queue.extend(valid)
        seen.update(valid)
    coords = np.asarray(chosen, dtype=np.int64)
    values = np.asarray([coordinate_to_margin[tuple(item)] for item in coords], dtype=np.float64)
    return coords, values


def select_segment_candidates(
    samples: Mapping[str, Mapping[str, Any]],
    *,
    pair_id: int,
    height: int,
    width: int,
    per_orientation: int = 1,
) -> list[dict[str, Any]]:
    """Select connected, low-margin boundary segments from one real frame."""

    candidates: list[dict[str, Any]] = []
    structure = ndimage.generate_binary_structure(2, 1)
    for key, sample in samples.items():
        coords_nyx = sample["coordinates_nyx"].detach().cpu().numpy()
        margins = sample["margins"].detach().cpu().numpy().astype(np.float64)
        if len(coords_nyx) < 3:
            continue
        coords = coords_nyx[:, 1:]
        mask = np.zeros((height, width), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        labels, count = ndimage.label(mask, structure=structure)
        rows: list[dict[str, Any]] = []
        for component_id in range(1, count + 1):
            component = np.argwhere(labels == component_id)
            if len(component) < 3:
                continue
            lookup = {
                (int(y), int(x)): float(value)
                for (y, x), value in zip(coords, margins, strict=True)
            }
            component_margins = np.asarray(
                [lookup[(int(y), int(x))] for y, x in component],
                dtype=np.float64,
            )
            subset, subset_margins = _component_subset(component, component_margins)
            if len(subset) < 3:
                continue
            winner_name, rival_name = key.split("->", 1)
            rows.append(
                {
                    "pair_id": pair_id,
                    "orientation": key,
                    "winner_id": CLASS_NAMES.index(winner_name),
                    "rival_id": CLASS_NAMES.index(rival_name),
                    "coordinates_yx": subset.tolist(),
                    "segment_pixel_count": len(subset),
                    "margin_min": float(np.min(subset_margins)),
                    "margin_median": float(np.median(subset_margins)),
                    "margin_max": float(np.max(subset_margins)),
                }
            )
        rows.sort(key=lambda row: (row["margin_median"], row["margin_min"]))
        candidates.extend(rows[:per_orientation])
    return candidates


def _frame_row(pair_id: int, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": TELEMETRY_FRAME_SCHEMA,
        "pair_id": pair_id,
        "boundary_pixel_count": int(summary["boundary_pixel_count"]),
        "per_class_logit_energy": summary["per_class_logit_energy"],
        "ordered_pair_margins": summary["ordered_pair_margins"],
        "record_constancy": {
            "claim": False,
            "reason": (
                "one frame row records a state; repeated pixel coordinates across frames "
                "do not establish record-level constancy"
            ),
        },
    }


def process_batch(
    *,
    config: RunConfig,
    config_sha256: str,
    implementation: Mapping[str, Any],
    model: Any,
    arrays: Mapping[str, np.memmap],
    start: int,
    stop: int,
    scratch_batch_directory: Path,
) -> tuple[Path, Path]:
    import torch

    accumulator = _empty_accumulator()
    frame_rows: list[dict[str, Any]] = []
    margin_arrays: dict[str, np.ndarray] = {}
    segment_candidates: list[dict[str, Any]] = []
    erf_probes: list[dict[str, Any]] = []
    started = time.perf_counter()
    with SegNetInternalTelemetry(
        model,
        policy=SegNetTelemetryPolicy.analysis_default(),
    ) as telemetry:
        for pair_id in range(start, stop):
            camera = np.asarray(arrays["gt_f1.npy"][pair_id])
            model_input = _model_input(model, camera)
            with torch.inference_mode():
                logits, summary = telemetry.run(model_input)
            assert summary is not None
            predicted = logits.argmax(dim=1)[0].cpu().numpy()
            expected = np.asarray(arrays["lstars.npy"][pair_id])
            mismatch_count = int(np.count_nonzero(predicted != expected))
            if mismatch_count:
                raise DDMSN1Error(
                    f"frozen SegNet/cache self-check failed at pair {pair_id}: "
                    f"{mismatch_count} argmax mismatches"
                )
            _accumulate_summary(accumulator, summary)
            frame_rows.append(_frame_row(pair_id, summary))
            samples = extract_ordered_pair_boundary_samples(logits)
            for winner in range(len(CLASS_NAMES)):
                for rival in range(len(CLASS_NAMES)):
                    if winner == rival:
                        continue
                    key = orientation(winner, rival)
                    margin_arrays[f"p{pair_id:04d}_w{winner}_r{rival}"] = (
                        samples[key]["margins"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
            segment_candidates.extend(
                select_segment_candidates(
                    samples,
                    pair_id=pair_id,
                    height=logits.shape[-2],
                    width=logits.shape[-1],
                )
            )
            if pair_id in config.erf_pair_ids:
                available = [
                    (
                        float(sample["margins"].min()),
                        key,
                        sample,
                    )
                    for key, sample in samples.items()
                    if sample["margins"].numel()
                ]
                if not available:
                    raise DDMSN1Error(f"ERF pair {pair_id} has no boundary sample")
                _margin, key, chosen = min(available, key=lambda item: item[0])
                minimum = int(torch.argmin(chosen["margins"]))
                _n, y, x = [
                    int(value)
                    for value in chosen["coordinates_nyx"][minimum].tolist()
                ]
                winner_name, rival_name = key.split("->", 1)
                probe = measure_erf_response(
                    model,
                    model_input,
                    y=y,
                    x=x,
                    winner=CLASS_NAMES.index(winner_name),
                    rival=CLASS_NAMES.index(rival_name),
                )
                probe["pair_id"] = pair_id
                probe["orientation"] = key
                erf_probes.append(probe)
    scratch_batch_directory.mkdir(parents=True, exist_ok=True)
    margins_path = scratch_batch_directory / f"batch_{start:04d}_{stop:04d}_margins.npz"
    _atomic_npz(margins_path, margin_arrays)
    margins_sha = sha256_file(margins_path)
    candidates = sorted(
        segment_candidates,
        key=lambda row: (row["margin_median"], row["margin_min"], row["pair_id"]),
    )[:64]
    batch = {
        "schema": BATCH_SCHEMA,
        "run_id": config.run_id,
        "config_sha256": config_sha256,
        "implementation_identity": implementation,
        "pair_window": [start, stop],
        "pair_count": stop - start,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "argmax_cache_mismatch_count": 0,
        "telemetry_accumulator": accumulator,
        "frame_rows": frame_rows,
        "segment_candidates": candidates,
        "erf_probes": erf_probes,
        "margin_arrays": {
            "path": str(margins_path),
            "bytes": margins_path.stat().st_size,
            "sha256": margins_sha,
            "array_count": len(margin_arrays),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "verdict_scope": (
            "official-video source cache, frozen CPU-Torch SegNet, declared pair window; "
            "telemetry only, not score authority"
        ),
    }
    batch_path = config.output_directory / "stage_checkpoints" / f"batch_{start:04d}_{stop:04d}.json"
    atomic_json(batch_path, batch)
    return batch_path, margins_path


def _valid_batch(
    path: Path,
    *,
    config: RunConfig,
    config_sha256: str,
    implementation: Mapping[str, Any],
    start: int,
    stop: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        margin = row["margin_arrays"]
        margin_path = Path(margin["path"])
        return (
            row["schema"] == BATCH_SCHEMA
            and row["run_id"] == config.run_id
            and row["config_sha256"] == config_sha256
            and row["implementation_identity"] == implementation
            and row["pair_window"] == [start, stop]
            and row["pair_count"] == stop - start
            and row["argmax_cache_mismatch_count"] == 0
            and margin_path.is_file()
            and margin_path.stat().st_size == margin["bytes"]
            and sha256_file(margin_path) == margin["sha256"]
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _progress(
    *,
    config: RunConfig,
    config_sha256: str,
    implementation: Mapping[str, Any],
    completed: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    covered = sum(int(row["pair_window"][1]) - int(row["pair_window"][0]) for row in completed)
    return {
        "schema": PROGRESS_SCHEMA,
        "run_id": config.run_id,
        "config_sha256": config_sha256,
        "implementation_identity": implementation,
        "completed_batches": list(completed),
        "completed_pair_count": covered,
        "next_pair": covered,
        "complete": covered == N600,
        "checkpoint_policy": config.checkpoint_policy,
        "all_preserved": True,
    }


def _load_batches(
    config: RunConfig,
    *,
    config_sha256: str,
    implementation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    paths = sorted((config.output_directory / "stage_checkpoints").glob("batch_*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = list(range(0, N600, config.batch_pairs))
    if len(rows) != len(expected):
        raise DDMSN1Error("finalization requires every n600 batch checkpoint")
    for row, start in zip(rows, expected, strict=True):
        stop = min(N600, start + config.batch_pairs)
        if row["pair_window"] != [start, stop]:
            raise DDMSN1Error("batch checkpoint windows are noncanonical")
        if row["config_sha256"] != config_sha256:
            raise DDMSN1Error("batch checkpoint config SHA drift during finalization")
        if row["implementation_identity"] != implementation:
            raise DDMSN1Error("batch implementation drift during finalization")
        margin_path = Path(row["margin_arrays"]["path"])
        if sha256_file(margin_path) != row["margin_arrays"]["sha256"]:
            raise DDMSN1Error("batch margin SHA drift during finalization")
    return rows


def _telemetry_jsonl(
    *,
    config: RunConfig,
    config_sha256: str,
    batches: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> bytes:
    header = {
        "schema": TELEMETRY_HEADER_SCHEMA,
        "run_id": config.run_id,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "config_sha256": config_sha256,
        "frame_row_count": N600,
        "telemetry_default_for_analysis": True,
        "telemetry_default_for_training": False,
        "training_cadence_reason": SegNetTelemetryPolicy.training_default().reason,
        "argmax_identity": {
            "argmax_identical": identity["argmax_identical"],
            "argmax_mismatch_count": identity["argmax_mismatch_count"],
            "logits_bitwise_identical": identity["logits_bitwise_identical"],
        },
        "aggregate": aggregate,
    }
    rows = [canonical_json_bytes(header).rstrip(b"\n")]
    for batch in batches:
        rows.extend(
            canonical_json_bytes(frame).rstrip(b"\n")
            for frame in batch["frame_rows"]
        )
    return b"\n".join(rows) + b"\n"


def _margins_for_stratum(
    batches: Sequence[Mapping[str, Any]],
    *,
    temporal_stratum: str,
) -> dict[str, np.ndarray]:
    if temporal_stratum == "n600_full":
        include = range(0, N600)
    elif temporal_stratum == "n600_first64_tail":
        include = range(0, 64)
    elif temporal_stratum == "n600_last64_tail":
        include = range(N600 - 64, N600)
    else:
        raise DDMSN1Error(f"unknown temporal stratum: {temporal_stratum}")
    included = set(include)
    values: dict[str, list[np.ndarray]] = {
        orientation(winner, rival): []
        for winner in range(len(CLASS_NAMES))
        for rival in range(len(CLASS_NAMES))
        if winner != rival
    }
    for batch in batches:
        path = Path(batch["margin_arrays"]["path"])
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                pair_id = int(key[1:5])
                if pair_id not in included:
                    continue
                winner = int(key[7])
                rival = int(key[10])
                values[orientation(winner, rival)].append(
                    np.asarray(archive[key], dtype=np.float64)
                )
    return {
        key: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        for key, chunks in values.items()
    }


def _sparse_delta_artifact(
    *,
    path: Path,
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, Any]:
    changed = np.flatnonzero(after.reshape(-1) != before.reshape(-1)).astype(np.int64)
    values = after.reshape(-1)[changed].astype(np.uint8)
    _atomic_npz(path, {"flat_indices": changed, "replacement_uint8": values})
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "changed_camera_bytes": len(changed),
        "reconstruction": (
            "copy gt_f1[pair_id], flatten C-order HWC, assign replacement_uint8 "
            "at flat_indices, reshape"
        ),
    }


def inverse_solve_segment(
    *,
    model: Any,
    camera_frame: np.ndarray,
    pair_id: int,
    candidate: Mapping[str, Any],
    max_steps: int,
    max_linf: int,
    camera_radius: int,
    artifact_path: Path,
) -> dict[str, Any]:
    """Projected camera-uint8 descent through exact R.down and frozen SegNet."""

    import torch

    base_u8 = np.ascontiguousarray(camera_frame, dtype=np.uint8)
    base = torch.from_numpy(base_u8).permute(2, 0, 1)[None].to(dtype=torch.float32)
    coords = np.asarray(candidate["coordinates_yx"], dtype=np.int64)
    winner = int(candidate["winner_id"])
    rival = int(candidate["rival_id"])
    ys = torch.from_numpy(coords[:, 0]).to(dtype=torch.long)
    xs = torch.from_numpy(coords[:, 1]).to(dtype=torch.long)
    with torch.inference_mode():
        base_logits = model(_model_input(model, base_u8))
        base_labels = base_logits.argmax(dim=1)[0]
        before_margin = float(
            (base_logits[0, winner, ys, xs] - base_logits[0, rival, ys, xs]).mean()
        )
    mask = torch.zeros_like(base, dtype=torch.bool)
    camera_y = np.floor((coords[:, 0] + 0.5) * CAMERA_HW[0] / SEG_HW[0]).astype(int)
    camera_x = np.floor((coords[:, 1] + 0.5) * CAMERA_HW[1] / SEG_HW[1]).astype(int)
    y0 = max(0, int(camera_y.min()) - camera_radius)
    y1 = min(CAMERA_HW[0], int(camera_y.max()) + camera_radius + 1)
    x0 = max(0, int(camera_x.min()) - camera_radius)
    x1 = min(CAMERA_HW[1], int(camera_x.max()) + camera_radius + 1)
    mask[:, :, y0:y1, x0:x1] = True
    current = base.clone()
    realized = False
    realized_target_count = 0
    after_margin = before_margin
    final_labels = base_labels
    steps_run = 0
    for step in range(1, max_steps + 1):
        variable = current.detach().clone().requires_grad_(True)
        pair = variable[:, None].expand(1, 2, -1, -1, -1)
        logits = model.preprocess_input(pair)
        logits = model(logits)
        margin = (logits[0, winner, ys, xs] - logits[0, rival, ys, xs]).mean()
        gradient = torch.autograd.grad(margin, variable, retain_graph=False)[0]
        proposal = current - torch.sign(gradient) * mask
        proposal = torch.maximum(torch.minimum(proposal, base + max_linf), base - max_linf)
        current = torch.round(proposal).clamp(0.0, 255.0).detach()
        with torch.inference_mode():
            realized_logits = model(
                model.preprocess_input(current[:, None].expand(1, 2, -1, -1, -1))
            )
            final_labels = realized_logits.argmax(dim=1)[0]
            realized_target_count = int(torch.count_nonzero(final_labels[ys, xs] == rival))
            after_margin = float(
                (
                    realized_logits[0, winner, ys, xs]
                    - realized_logits[0, rival, ys, xs]
                ).mean()
            )
        steps_run = step
        if realized_target_count >= math.ceil(len(coords) / 2):
            realized = True
            break
    after_u8 = current[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    delta_artifact = _sparse_delta_artifact(
        path=artifact_path,
        before=base_u8,
        after=after_u8,
    )
    changed_labels = final_labels != base_labels
    segment_mask = torch.zeros_like(changed_labels, dtype=torch.bool)
    segment_mask[ys, xs] = True
    collateral = int(torch.count_nonzero(changed_labels & ~segment_mask))
    return {
        "schema": INVERSE_SCHEMA,
        "pair_id": pair_id,
        "orientation": candidate["orientation"],
        "winner_id": winner,
        "rival_id": rival,
        "segment_pixel_count": len(coords),
        "coordinates_yx": coords.tolist(),
        "camera_bbox_half_open_yx": [y0, y1, x0, x1],
        "max_linf_uint8": max_linf,
        "steps_run": steps_run,
        "before_mean_margin": before_margin,
        "after_mean_margin": after_margin,
        "desired_rival_realized_count": realized_target_count,
        "desired_rival_realized_fraction": realized_target_count / len(coords),
        "majority_segment_transition_realized": realized,
        "all_argmax_changes": int(torch.count_nonzero(changed_labels)),
        "off_segment_argmax_changes": collateral,
        "sparse_delta_artifact": delta_artifact,
        "through_r_contract": (
            "camera uint8 -> upstream SegNet.preprocess_input bilinear 384x512 "
            "-> frozen CPU-Torch SegNet -> argmax"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "verdict_scope": (
            "bounded projected sign descent for this real boundary segment only; "
            "failure does not close the ordered-pair or inverse-receiver family"
        ),
    }


def _choose_inverse_candidates(
    batches: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    all_candidates = [
        candidate
        for batch in batches
        for candidate in batch["segment_candidates"]
    ]
    all_candidates.sort(
        key=lambda row: (
            row["margin_median"],
            row["margin_min"],
            row["pair_id"],
            row["orientation"],
        )
    )
    selected: list[dict[str, Any]] = []
    used_orientations: set[str] = set()
    for row in all_candidates:
        if row["orientation"] in used_orientations:
            continue
        selected.append(row)
        used_orientations.add(row["orientation"])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise DDMSN1Error(f"only {len(selected)} distinct inverse boundary segments available")
    return selected


def finalize(
    *,
    config: RunConfig,
    config_path: Path,
    config_sha256: str,
    implementation: Mapping[str, Any],
    source_custody: Mapping[str, Any],
    storage: Mapping[str, Any],
    model: Any,
    arrays: Mapping[str, np.memmap],
    identity: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    batches = _load_batches(
        config,
        config_sha256=config_sha256,
        implementation=implementation,
    )
    accumulator = _empty_accumulator()
    for batch in batches:
        _merge_accumulator(accumulator, batch["telemetry_accumulator"])
    aggregate = _finalize_accumulator(accumulator)
    telemetry_payload = _telemetry_jsonl(
        config=config,
        config_sha256=config_sha256,
        batches=batches,
        identity=identity,
        aggregate=aggregate,
    )
    telemetry_path = config.output_directory / "segnet_internal_telemetry_n600.jsonl"
    _atomic_bytes(telemetry_path, telemetry_payload)
    telemetry_sha = hashlib.sha256(telemetry_payload).hexdigest()
    pair_norms = exact_head_norms(model)
    sided_rows: list[SidedToleranceRow] = []
    for stratum in TEMPORAL_STRATA:
        sided_rows.extend(
            build_sided_rows(
                temporal_stratum=stratum,
                margins_by_orientation=_margins_for_stratum(
                    batches,
                    temporal_stratum=stratum,
                ),
                pair_norms_by_orientation=pair_norms,
            )
        )
    sided_header = build_header(
        source_video_sha256=config.video_sha256,
        segnet_weights_sha256=config.segnet_weights_sha256,
        upstream_modules_sha256=config.upstream_modules_sha256,
        telemetry_sha256=telemetry_sha,
    )
    sided_payload = export_jsonl(sided_header, sided_rows)
    parse_jsonl(sided_payload)
    sided_path = config.output_directory / "sdwl1_sided_tolerance_n600.jsonl"
    _atomic_bytes(sided_path, sided_payload)
    inverse_candidates = _choose_inverse_candidates(
        batches,
        count=config.inverse_segment_count,
    )
    inverse_rows = []
    for index, candidate in enumerate(inverse_candidates):
        pair_id = int(candidate["pair_id"])
        inverse_rows.append(
            inverse_solve_segment(
                model=model,
                camera_frame=np.asarray(arrays["gt_f1.npy"][pair_id]),
                pair_id=pair_id,
                candidate=candidate,
                max_steps=config.inverse_max_steps,
                max_linf=config.inverse_max_linf,
                camera_radius=config.inverse_camera_radius,
                artifact_path=(
                    config.scratch_directory
                    / "inverse_deltas"
                    / f"segment_{index:02d}_pair_{pair_id:04d}.npz"
                ),
            )
        )
    inverse_receipt = {
        "schema": "ddm_sn1_segnet_sided_inverse_demo_receipt.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "segment_count": len(inverse_rows),
        "majority_transition_realized_count": sum(
            int(row["majority_segment_transition_realized"])
            for row in inverse_rows
        ),
        "rows": inverse_rows,
    }
    inverse_path = config.output_directory / "inverse_solve_three_segments_receipt.json"
    atomic_json(inverse_path, inverse_receipt)
    erf_probes = [
        probe
        for batch in batches
        for probe in batch["erf_probes"]
    ]
    if sorted(probe["pair_id"] for probe in erf_probes) != sorted(config.erf_pair_ids):
        raise DDMSN1Error("ERF probes do not cover the typed-config pair ids")
    full_rows = {
        row.orientation: row
        for row in sided_rows
        if row.temporal_stratum == "n600_full"
    }
    e1_examples = [
        export_e1_bounds(full_rows["Road->Lane"]),
        export_e1_bounds(full_rows["Lane->Road"]),
    ]
    outputs = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (telemetry_path, sided_path, inverse_path)
    ]
    receipt = {
        "schema": RUN_SCHEMA,
        "run_id": config.run_id,
        "lane_id": "ddm_sn1_segnet_telemetry_asymmetry",
        "axis": AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "main_landing_review_required": True,
        "config": {
            "path": str(config_path),
            "sha256": config_sha256,
            "typed": {
                **asdict(config),
                "video_path": str(config.video_path),
                "gt_cache_path": str(config.gt_cache_path),
                "upstream_root": str(config.upstream_root),
                "output_directory": str(config.output_directory),
                "scratch_directory": str(config.scratch_directory),
                "erf_pair_ids": list(config.erf_pair_ids),
            },
        },
        "semantic_argv": sys.argv,
        "source_custody": source_custody,
        "implementation_identity": implementation,
        "storage_preflight": storage,
        "runtime": {
            "git_head": _git_head(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch_threads": config.torch_threads,
            "seed": config.seed,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "telemetry_identity": {
            "argmax_identical": identity["argmax_identical"],
            "argmax_mismatch_count": identity["argmax_mismatch_count"],
            "logits_bitwise_identical": identity["logits_bitwise_identical"],
            "coverage": identity["telemetry_summary"]["coverage"],
        },
        "measurement": {
            "pair_count": N600,
            "cache_argmax_mismatch_count": 0,
            "ordered_pair_count": len(full_rows),
            "temporal_strata": list(TEMPORAL_STRATA),
            "head_norms_exact": pair_norms,
            "erf_probes": erf_probes,
            "aggregate": aggregate,
            "inverse_demo": {
                "segment_count": len(inverse_rows),
                "majority_transition_realized_count": inverse_receipt[
                    "majority_transition_realized_count"
                ],
            },
            "e1_signed_bounds_examples": e1_examples,
            "record_constancy_correction": (
                "pixel recurrence is not record-level constancy; this analysis emits "
                "600 distinct frame records and makes no static-record inference"
            ),
        },
        "outputs": outputs,
        "scratch_artifacts": [
            batch["margin_arrays"] for batch in batches
        ]
        + [row["sparse_delta_artifact"] for row in inverse_rows],
        "verdict": "MEASURED_SIDED_ASYMMETRY_ADVISORY_POINTER_UNMOVED",
        "verdict_scope": (
            "real official-video n600 cached frames, exact frozen CPU-Torch SegNet, "
            "head-space D2 and bounded camera-uint8 inverse segments; no Pose, byte-closed "
            "archive, contest-CPU, or contest-CUDA authority"
        ),
        "first_rung": (
            "feed the sided bounds into one e1 receiver candidate and price the realized "
            "inner/outer errors with SHA-current e2 costates before spending any bytes"
        ),
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            ".omx/research/ddm_dv2_sdwl1_n600_20260723/receipt.json",
            ".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_receipt.json",
            ".omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z/ddm_g4_spatial_stationarity_receipt.json",
            ".omx/research/codex_findings_ddm_dr2_scc_outside_view_20260723_codex.md",
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/canonical_task_status.jsonl",
        ],
    }
    receipt_path = config.output_directory / "ddm_sn1_segnet_telemetry_asymmetry_receipt.json"
    atomic_json(receipt_path, receipt)
    return receipt


def run(config_path: Path) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    config, config_sha256 = load_config(config_path)
    implementation = implementation_identity()
    source_custody = validate_source_custody(config)
    storage = storage_preflight(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    arrays = open_target_cache(config.gt_cache_path)
    model = load_frozen_segnet(config)
    identity_input = _model_input(model, np.asarray(arrays["gt_f1.npy"][0]))
    identity = assert_telemetry_argmax_identity(model, identity_input)
    with torch.inference_mode():
        identity_labels = model(identity_input).argmax(dim=1)[0].cpu().numpy()
    if np.count_nonzero(identity_labels != np.asarray(arrays["lstars.npy"][0])):
        raise DDMSN1Error("identity frame does not match the SHA-pinned n600 target cache")
    identity_path = config.output_directory / "stage_checkpoints" / "00_telemetry_identity.json"
    atomic_json(
        identity_path,
        {
            **identity,
            "config_sha256": config_sha256,
            "implementation_identity": implementation,
        },
    )
    completed: list[dict[str, Any]] = []
    for start in range(0, N600, config.batch_pairs):
        stop = min(N600, start + config.batch_pairs)
        batch_path = (
            config.output_directory
            / "stage_checkpoints"
            / f"batch_{start:04d}_{stop:04d}.json"
        )
        if not _valid_batch(
            batch_path,
            config=config,
            config_sha256=config_sha256,
            implementation=implementation,
            start=start,
            stop=stop,
        ):
            process_batch(
                config=config,
                config_sha256=config_sha256,
                implementation=implementation,
                model=model,
                arrays=arrays,
                start=start,
                stop=stop,
                scratch_batch_directory=config.scratch_directory / "batch_margins",
            )
        completed.append(
            {
                "pair_window": [start, stop],
                "path": str(batch_path),
                "sha256": sha256_file(batch_path),
            }
        )
        atomic_json(
            config.output_directory / "stage_checkpoints" / "progress.json",
            _progress(
                config=config,
                config_sha256=config_sha256,
                implementation=implementation,
                completed=completed,
            ),
        )
        print(
            json.dumps(
                {
                    "event": "ddm_sn1_batch_complete",
                    "pair_window": [start, stop],
                    "completed_pairs": stop,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return finalize(
        config=config,
        config_path=config_path.resolve(),
        config_sha256=config_sha256,
        implementation=implementation,
        source_custody=source_custody,
        storage=storage,
        model=model,
        arrays=arrays,
        identity=identity,
        started=started,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run(args.config)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
