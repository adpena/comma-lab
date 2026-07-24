#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile and measure the DDM MENU1 realized-flip menu on exact V19C n600.

This is a local-only advisory runner.  It compiles the complete SN1 x fix
inventory, fits one counted per-class x row-band statistics payload, then
measures a dependency-ordered exact receiver chain with SegNet and PoseNet.
Every batch checkpoint is immutable and lives on the SSD tier.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.analysis.ddm_sn1_error_source_tensor import boundary_distance_bands  # noqa: E402
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_continuous_paint_ceiling import (  # noqa: E402
    render_analytic_coverage_blend,
    render_hard_camera_placement,
    resample_fields_at_pixel_centres,
    signed_distance_fields,
)
from tac.optimization.ddm_realized_flip_menu import (  # noqa: E402
    CAMERA_HW,
    EVIDENCE_AXIS,
    MENU_SCHEMA,
    RATE_DUAL,
    SEG_HW,
    RealizedFlipMenuError,
    advisory_objective,
    apply_local_statistics,
    apply_scalar_affine,
    apply_temporal_affine,
    compile_menu_rows,
    decode_target_masks,
    encode_local_statistics,
    encode_scalar_affine,
    encode_target_masks,
    encode_temporal_affine,
    greedy_telescoping_curve,
    local_statistics_sufficient_statistics,
    sha256_bytes,
    solve_local_statistics,
    transition_counts,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CLASS_ORDER,
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
)
from tac.optimization.direct_description_preuint8_channel import (  # noqa: E402
    receive_preuint8_q8_archive,
)

SCHEMA = "ddm_menu1_realized_flip_menu_measurement.v1"
CONFIG_SCHEMA = "DDMMenu1RealizedFlipMenuConfigV1"
LANE_ID = "lane_ddm_menu1_realized_flip_menu_20260723"
DELEGATION_KEY = "codex_delegate:ddm_menu1_realized_flip_menu_compiler:20260723T214943Z"
CLASS_NAMES = dict(enumerate(CLASS_ORDER))


class Menu1Config(BaseModel):
    """Strict local measurement contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMMenu1RealizedFlipMenuConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[210] = 210
    pair_count: Literal[600] = 600
    scorer_batch_size: Literal[16] = 16
    scorer_threads: Literal[4] = 4
    row_bands: Literal[16] = 16
    temporal_knots: Literal[16] = 16
    sn1_receipt_path: str
    sn1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    solve_menu_path: str
    solve_menu_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pt1_receipt_path: str
    pt1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    e2_receipt_path: str
    e2_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dr2b_receipt_path: str
    dr2b_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    c1_ledger_path: str
    c1_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19c_receipt_path: str
    v19c_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19c_archive_path: str
    v19c_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19c_archive_bytes: Literal[137827] = 137827
    target_cache_path: str
    target_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_cache_bytes: Literal[5078017610] = 5_078_017_610
    upstream_root: str
    modules_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segnet_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    posenet_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    at1x_manifest_path: str
    at1x_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pa1_receipt_path: str
    pa1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_root: str
    byte_budget: Literal[200000] = 200_000
    target_errors: Literal[136839] = 136_839
    target_d_seg: Literal[0.00116] = 0.00116
    target_d_pose: Literal[0.00161] = 0.00161
    v19c_residual_errors: Literal[2265811] = 2_265_811
    v19c_total_errors: Literal[2923991] = 2_923_991
    execution_allowed: Literal[True] = True
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _validate_paths(self) -> Menu1Config:
        for value in (
            self.target_cache_path,
            self.upstream_root,
            self.at1x_manifest_path,
            self.pa1_receipt_path,
            self.checkpoint_root,
        ):
            if not Path(value).is_absolute():
                raise ValueError(f"absolute path required: {value}")
        if not self.checkpoint_root.startswith("/Volumes/VertigoDataTier/pact/"):
            raise ValueError("checkpoint_root must use the primary SSD tier")
        return self

    def stable_hash(self) -> str:
        return sha256_bytes(
            json.dumps(
                self.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude={"pa1_receipt_path", "pa1_receipt_sha256"},
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )

    def full_hash(self) -> str:
        return sha256_bytes(
            json.dumps(
                self.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )


def _read(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise RealizedFlipMenuError(f"regular non-symlink file required: {path}")
    return path.read_bytes()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _bound(path: str, expected: str, label: str) -> bytes:
    value = _resolve(path)
    observed = _sha256_file(value)
    if observed != expected:
        raise RealizedFlipMenuError(
            f"{label} SHA-256 differs: {observed} != {expected}"
        )
    return _read(value)


def _bound_json(path: str, expected: str, label: str) -> dict[str, Any]:
    payload = _bound(path, expected, label)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RealizedFlipMenuError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise RealizedFlipMenuError(f"{label} must be one JSON object")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) != payload:
            raise RealizedFlipMenuError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _publish_json(path: Path, value: Any) -> None:
    _publish(path, _canonical(value) + b"\n")


def _publish_npz(path: Path, *, cells: np.ndarray, pose6: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["cells"], cells) or not np.array_equal(
                stored["pose6"], pose6
            ):
                raise RealizedFlipMenuError(f"immutable NPZ differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, cells=cells, pose6=pose6)
    os.replace(temporary, path)


def _publish_gzip_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    raw = b"".join(_canonical(dict(row)) + b"\n" for row in rows)
    payload = gzip.compress(raw, compresslevel=9, mtime=0)
    _publish(path, payload)


def _load_models(config: Menu1Config) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from safetensors.torch import load_file

    upstream = Path(config.upstream_root).resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    modules_path = upstream / "modules.py"
    seg_path = upstream / "models" / "segnet.safetensors"
    pose_path = upstream / "models" / "posenet.safetensors"
    for path, expected, label in (
        (modules_path, config.modules_sha256, "modules.py"),
        (seg_path, config.segnet_weights_sha256, "SegNet weights"),
        (pose_path, config.posenet_weights_sha256, "PoseNet weights"),
    ):
        if _sha256_file(path) != expected:
            raise RealizedFlipMenuError(f"{label} SHA-256 differs")
    spec = importlib.util.spec_from_file_location("ddm_menu1_upstream_modules", modules_path)
    if spec is None or spec.loader is None:
        raise RealizedFlipMenuError("cannot import frozen scorer modules")
    modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modules)
    torch.set_num_threads(config.scorer_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    segnet = modules.SegNet().eval().cpu()
    posenet = modules.PoseNet().eval().cpu()
    segnet.load_state_dict(load_file(str(seg_path), device="cpu"), strict=True)
    posenet.load_state_dict(load_file(str(pose_path), device="cpu"), strict=True)
    for model in (segnet, posenet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return segnet, posenet, {
        "modules_path": str(modules_path),
        "modules_sha256": config.modules_sha256,
        "segnet_weights_path": str(seg_path),
        "segnet_weights_sha256": config.segnet_weights_sha256,
        "posenet_weights_path": str(pose_path),
        "posenet_weights_sha256": config.posenet_weights_sha256,
        "device": "cpu",
        "threads": config.scorer_threads,
        "batch_size": config.scorer_batch_size,
        "deterministic_algorithms": True,
        "evidence_axis": EVIDENCE_AXIS,
    }


def _forward(segnet: Any, posenet: Any, camera_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    camera = np.asarray(camera_pairs)
    if (
        camera.dtype != np.uint8
        or camera.ndim != 5
        or camera.shape[1:] != (2, *CAMERA_HW, 3)
    ):
        raise RealizedFlipMenuError("scorer camera geometry differs")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(camera))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
    )
    with torch.inference_mode():
        cells = (
            segnet(segnet.preprocess_input(tensor))
            .argmax(dim=1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        pose_output = posenet(posenet.preprocess_input(tensor))
        pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose[:, :6].cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(cells), np.ascontiguousarray(pose6)


def _palette(receiver: Any) -> np.ndarray:
    carrier = receiver.base.base
    if carrier.realization_profile is None:
        raise RealizedFlipMenuError("V19C receiver has no realization profile")
    role_for_class = {
        "Road": "Road",
        "Lane": "Lane",
        "Undrivable": "UndrivableBoundary",
        "Movable": "Movable",
        "MyCar": "MyCar",
    }
    return np.stack(
        [
            carrier.realization_profile.colour_for(role_for_class[name])
            for name in CLASS_ORDER
        ],
        axis=0,
    ).astype(np.uint8)


def _semantic_cells(
    receiver: Any,
    local_ids: Sequence[int],
    camera: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    carrier = receiver.base.base
    layer_by_role = {row.role: row for row in carrier.layers}
    semantic = np.full((len(local_ids), *SEG_HW), -1, dtype=np.int16)
    owned = np.zeros_like(semantic, dtype=bool)
    for role in REALIZATION_PAINT_ORDER:
        layer = layer_by_role[role]
        for index, pair_id in enumerate(local_ids):
            mask = carrier._mask_for_layer(
                layer,
                int(pair_id),
                replace_g1_movable=True,
            )
            semantic[index, mask] = ROLE_CLASS_IDS[role]
            owned[index, mask] = True
    row_centres = (
        (np.arange(SEG_HW[0], dtype=np.int64) * CAMERA_HW[0] + CAMERA_HW[0] // 2)
        // SEG_HW[0]
    ).clip(0, CAMERA_HW[0] - 1)
    col_centres = (
        (np.arange(SEG_HW[1], dtype=np.int64) * CAMERA_HW[1] + CAMERA_HW[1] // 2)
        // SEG_HW[1]
    ).clip(0, CAMERA_HW[1] - 1)
    sampled = camera[:, 1][:, row_centres[:, None], col_centres[None, :]]
    distance = np.square(
        sampled[..., None, :].astype(np.int16)
        - palette[None, None, None].astype(np.int16)
    ).sum(axis=-1)
    fallback = np.argmin(distance, axis=-1).astype(np.int16)
    semantic[~owned] = fallback[~owned]
    if semantic.min() < 0 or semantic.max() >= len(CLASS_ORDER):
        raise RealizedFlipMenuError("decoder semantic class map escaped [0,5)")
    return semantic.astype(np.uint8), owned


def _geometry_statistics_camera(
    *,
    base_camera: np.ndarray,
    semantic: np.ndarray,
    owned: np.ndarray,
    palette: np.ndarray,
    statistics_payload: bytes,
) -> np.ndarray:
    fields = signed_distance_fields(semantic)
    camera_fields = resample_fields_at_pixel_centres(fields)
    hard = render_hard_camera_placement(camera_fields, palette)
    analytic = render_analytic_coverage_blend(
        camera_fields,
        palette,
        softness=1.0,
    )
    ordered = np.partition(camera_fields, -2, axis=-1)
    interior = (ordered[..., -1] - ordered[..., -2]) >= 1.0
    geometry = analytic.copy()
    geometry[interior] = hard[interior]
    ys = (np.arange(CAMERA_HW[0]) * SEG_HW[0] // CAMERA_HW[0]).clip(
        0, SEG_HW[0] - 1
    )
    xs = (np.arange(CAMERA_HW[1]) * SEG_HW[1] // CAMERA_HW[1]).clip(
        0, SEG_HW[1] - 1
    )
    owner_camera = owned[:, ys[:, None], xs[None, :]]
    result = base_camera.copy()
    result[:, 1][owner_camera] = geometry[owner_camera]
    return apply_local_statistics(result, semantic, statistics_payload)


def _targeted_camera(
    *,
    current: np.ndarray,
    masks: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    prototype = np.broadcast_to(palette[0], (*CAMERA_HW, 3))
    ys = (np.arange(CAMERA_HW[0]) * SEG_HW[0] // CAMERA_HW[0]).clip(
        0, SEG_HW[0] - 1
    )
    xs = (np.arange(CAMERA_HW[1]) * SEG_HW[1] // CAMERA_HW[1]).clip(
        0, SEG_HW[1] - 1
    )
    camera_mask = masks[:, ys[:, None], xs[None, :]]
    result = current.copy()
    result[:, 1][camera_mask] = np.broadcast_to(
        prototype,
        result[:, 1].shape,
    )[camera_mask]
    return result


def _config_and_inputs(config_path: Path) -> tuple[Menu1Config, dict[str, Any]]:
    config = Menu1Config.model_validate_json(_read(config_path))
    receipts = {
        "sn1": _bound_json(
            config.sn1_receipt_path,
            config.sn1_receipt_sha256,
            "SN1 receipt",
        ),
        "pt1": _bound_json(
            config.pt1_receipt_path,
            config.pt1_receipt_sha256,
            "PT1 receipt",
        ),
        "e2": _bound_json(
            config.e2_receipt_path,
            config.e2_receipt_sha256,
            "E2 receipt",
        ),
        "dr2b": _bound_json(
            config.dr2b_receipt_path,
            config.dr2b_receipt_sha256,
            "DR2B receipt",
        ),
        "c1": _bound_json(
            config.c1_ledger_path,
            config.c1_ledger_sha256,
            "C1 ledger",
        ),
        "v19c": _bound_json(
            config.v19c_receipt_path,
            config.v19c_receipt_sha256,
            "V19C receipt",
        ),
    }
    archive = _bound(
        config.v19c_archive_path,
        config.v19c_archive_sha256,
        "V19C archive",
    )
    if len(archive) != config.v19c_archive_bytes:
        raise RealizedFlipMenuError("V19C archive byte count differs")
    endpoint = receipts["v19c"]["curve"]["n600_endpoint"]
    if (
        endpoint["archive_sha256"] != config.v19c_archive_sha256
        or int(endpoint["archive_bytes"]) != len(archive)
        or round(float(endpoint["d_seg"]) * 117_964_800) != config.v19c_total_errors
    ):
        raise RealizedFlipMenuError("V19C endpoint receipt does not bind the base")
    if (
        receipts["c1"]["box"]["archive_bytes_max"] != config.byte_budget
        or float(receipts["c1"]["box"]["d_seg_max"]) != config.target_d_seg
        or receipts["c1"]["box"]["pose_stream_required"] is not True
    ):
        raise RealizedFlipMenuError("C1 box differs")
    bridge = receipts["dr2b"]["u1_lossy_tolerance_ladder"]
    if bridge["first_fit_rung"] is not None:
        raise RealizedFlipMenuError("DR2B bridge blocker unexpectedly closed")
    at1x = _bound_json(
        config.at1x_manifest_path,
        config.at1x_manifest_sha256,
        "AT1X atlas manifest",
    )
    amplitude = at1x["amplitude_factors"]
    if (
        int(amplitude["count"]) != 0
        or amplitude["through_r_uint8_survival_required_before_nonzero"] is not True
    ):
        raise RealizedFlipMenuError("AT1X FREE-affine bridge custody differs")
    pa1 = _bound_json(
        config.pa1_receipt_path,
        config.pa1_receipt_sha256,
        "PA1 pose-amplitude receipt",
    )
    pa1_rows = pa1["menu1_rows"]
    if (
        len(pa1_rows) != 3
        or {row["rung"] for row in pa1_rows}
        != {"frame0_gt", "frame0_scorer", "joint_scorer"}
        or any(row["pool_id"] != "pose_amplitude" for row in pa1_rows)
        or pa1["score_claim"] is not False
    ):
        raise RealizedFlipMenuError("PA1 pose-amplitude menu custody differs")
    pose_pool_rows = []
    for source_row in pa1_rows:
        free_candidate = source_row["rate_partition"] == "FREE"
        counted = int(source_row["delta"]["delta_bytes"])
        pose_pool_rows.append(
            {
                "row_id": f"pa1:{source_row['rung']}",
                "composition_pool_id": "pose_amplitude",
                "parent_control": "PA1 baseline; V19C composition unmeasured",
                "active_base_price_imported": False,
                "delta_errors_realized": None,
                "delta_counted_bytes": counted,
                "byte_partition": {
                    "COUNTED": counted,
                    "FREE": 0,
                    "NULL": 0,
                    "FREE_candidate": 0 if free_candidate else None,
                    "status": (
                        "PENDING_RECEIVER_SURVIVAL"
                        if free_candidate
                        else "COUNTED_CROSS_CONTROL"
                    ),
                    "law": "FREE_UNION_NULL_UNION_COUNTED",
                },
                "pa1_measurement": source_row["measurement"],
                "pa1_delta": source_row["delta"],
                "measurement_status": (
                    "MEASURED_PA1_CROSS_CONTROL_COMPOSITION_OWED"
                ),
                "waterfill_eligible": False,
                "next_measurement": source_row["next_measurement"],
                "evidence_axis": pa1["evidence_axis"],
                "research_only": True,
                "score_claim": False,
            }
        )
    target_path = Path(config.target_cache_path)
    if (
        target_path.stat().st_size != config.target_cache_bytes
        or _sha256_file(target_path) != config.target_cache_sha256
    ):
        raise RealizedFlipMenuError("target cache custody differs")
    return config, {
        "receipts": receipts,
        "archive": archive,
        "bridge": {
            "row_id": "bridge:sdwl1_to_e2_coordinate_crosswalk",
            "measurement_status": "BLOCKED_FORMULATION_BRIDGE",
            "delta_errors_realized": None,
            "delta_counted_bytes": None,
            "byte_partition": {
                "COUNTED": None,
                "FREE": 0,
                "NULL": 0,
                "status": "COUNTED_PRICE_BLOCKED_BY_COORDINATE_BRIDGE",
                "law": "FREE_UNION_NULL_UNION_COUNTED",
            },
            "composition_pool_id": "bridge:sdwl1_e2",
            "verdict_scope": bridge["verdict_scope"],
            "waterfill_eligible": False,
        },
        "bn_free_bridge": {
            "row_id": "bridge:at1x_bn_expected_stats_to_camera_affine",
            "measurement_status": "BLOCKED_NO_UINT8_SURVIVING_PROJECTION",
            "delta_errors_realized": None,
            "delta_counted_bytes": 0,
            "byte_partition": {
                "COUNTED": 0,
                "FREE": 0,
                "NULL": 0,
                "status": "FREE_CANDIDATE_BLOCKED_BEFORE_CAMERA_PROJECTION",
                "law": "FREE_UNION_NULL_UNION_COUNTED",
            },
            "composition_pool_id": "paint_amplitude:GLOBAL",
            "source_manifest_path": config.at1x_manifest_path,
            "source_manifest_sha256": config.at1x_manifest_sha256,
            "blocker": amplitude["why"],
            "verdict_scope": (
                "FORMULATION: AT1X BN closed forms lack a through-R uint8 "
                "camera projection; no FREE RGB affine may be invented"
            ),
            "waterfill_eligible": False,
        },
        "pose_amplitude_pool": {
            "source_receipt_path": config.pa1_receipt_path,
            "source_receipt_sha256": config.pa1_receipt_sha256,
            "pool_id": "pose_amplitude",
            "distinct_from": "paint_amplitude:GLOBAL",
            "composition_rule": (
                "never sum cross-control deltas; exact V19C joint composition "
                "required before active waterfill"
            ),
            "rows": pose_pool_rows,
        },
    }


def _solve_rows(config: Menu1Config) -> list[dict[str, Any]]:
    payload = _bound(
        config.solve_menu_path,
        config.solve_menu_sha256,
        "SN1 solve menu",
    )
    rows = [json.loads(line) for line in payload.splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise RealizedFlipMenuError("SN1 solve menu contains a non-object row")
    return rows


def _storage_preflight(config: Menu1Config) -> dict[str, Any]:
    root = Path(config.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    required = 512 << 20
    if free < required:
        raise RealizedFlipMenuError("SSD storage preflight failed")
    return {
        "tier": "/Volumes/VertigoDataTier/pact",
        "checkpoint_root": str(root),
        "required_free_bytes": required,
        "observed_free_bytes_at_least": required,
        "observation_policy": (
            "checked afresh before every execution; exact live free-space bytes "
            "are excluded from the immutable receipt because they are volatile"
        ),
        "status": "PASS",
        "cleanup_policy": (
            "preserve immutable measurement checkpoints; no destructive cleanup "
            "because they are the reproducibility surface"
        ),
    }


def _semantic_stage(
    *,
    config: Menu1Config,
    receiver: Any,
    target_camera: np.ndarray,
    palette: np.ndarray,
    checkpoint_root: Path,
) -> tuple[bytes, dict[str, Any]]:
    receipt_path = checkpoint_root / "01_local_statistics_receipt.json"
    payload_path = checkpoint_root / "01_local_statistics_payload.bin"
    if receipt_path.exists() and payload_path.exists():
        receipt = json.loads(_read(receipt_path))
        payload = _read(payload_path)
        if (
            receipt["typed_config_sha256"] != config.stable_hash()
            or receipt["payload_sha256"] != sha256_bytes(payload)
        ):
            raise RealizedFlipMenuError("local-statistics resume custody differs")
        return payload, receipt
    aggregate: dict[str, np.ndarray] | None = None
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        print(
            f"[MENU1] local-statistics fit {start:04d}:{stop:04d}",
            flush=True,
        )
        ids = tuple(range(start, stop))
        camera = receiver.render_camera_pairs(ids)
        semantic, _owned = _semantic_cells(receiver, ids, camera, palette)
        row = local_statistics_sufficient_statistics(
            source_rgb_u8=camera[:, 1],
            target_rgb_u8=np.asarray(target_camera[start:stop], dtype=np.uint8),
            semantic_cells=semantic,
            row_bands=config.row_bands,
        )
        if aggregate is None:
            aggregate = {key: value.copy() for key, value in row.items()}
        else:
            for key, value in row.items():
                aggregate[key] += value
    if aggregate is None:
        raise RealizedFlipMenuError("local-statistics fit produced no rows")
    scale, offset, counts = solve_local_statistics(aggregate)
    payload = encode_local_statistics(scale, offset)
    receipt = {
        "schema": "ddm_menu1_local_statistics_receipt.v1",
        "typed_config_sha256": config.stable_hash(),
        "row_bands": config.row_bands,
        "classes": list(CLASS_ORDER),
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "payload_path": str(payload_path),
        "parameter_dtype": "float16",
        "parameter_count": int(scale.size + offset.size),
        "support_count_min": int(counts.min()),
        "support_count_max": int(counts.max()),
        "support_count_total": int(counts.sum()),
        "source": "V19C endpoint camera pairs",
        "target": "SHA-bound gt_n600.gt_f1 camera bytes",
        "fit_only": True,
        "measurement_status": "COUNTED_PAYLOAD_FIT_COMPLETE",
        "research_only": True,
        "score_claim": False,
    }
    _publish(payload_path, payload)
    _publish_json(receipt_path, receipt)
    return payload, receipt


def _global_ladder_stage(
    *,
    config: Menu1Config,
    receiver: Any,
    target_camera: np.ndarray,
    checkpoint_root: Path,
) -> tuple[bytes, bytes, dict[str, Any]]:
    receipt_path = checkpoint_root / "01b_global_temporal_ladder_receipt.json"
    scalar_path = checkpoint_root / "01b_scalar_gain_bias_payload.bin"
    temporal_path = checkpoint_root / "01c_temporal_affine_payload.bin"
    if receipt_path.exists() and scalar_path.exists() and temporal_path.exists():
        receipt = json.loads(_read(receipt_path))
        scalar_payload = _read(scalar_path)
        temporal_payload = _read(temporal_path)
        if (
            receipt["typed_config_sha256"] != config.stable_hash()
            or receipt["scalar"]["payload_sha256"]
            != sha256_bytes(scalar_payload)
            or receipt["temporal"]["payload_sha256"]
            != sha256_bytes(temporal_payload)
        ):
            raise RealizedFlipMenuError("global/temporal ladder resume custody differs")
        return scalar_payload, temporal_payload, receipt
    source_sum = 0.0
    target_sum = 0.0
    source_sumsq = 0.0
    target_sumsq = 0.0
    scalar_count = 0
    temporal_count = np.zeros(config.temporal_knots, dtype=np.int64)
    temporal_source_sum = np.zeros((config.temporal_knots, 3), dtype=np.float64)
    temporal_target_sum = np.zeros_like(temporal_source_sum)
    temporal_source_sumsq = np.zeros_like(temporal_source_sum)
    temporal_target_sumsq = np.zeros_like(temporal_source_sum)
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        print(f"[MENU1] global/temporal fit {start:04d}:{stop:04d}", flush=True)
        ids = tuple(range(start, stop))
        source = receiver.render_camera_pairs(ids)[:, 1].astype(np.float64)
        target = np.asarray(target_camera[start:stop], dtype=np.float64)
        source_sum += float(source.sum(dtype=np.float64))
        target_sum += float(target.sum(dtype=np.float64))
        source_sumsq += float(np.square(source).sum(dtype=np.float64))
        target_sumsq += float(np.square(target).sum(dtype=np.float64))
        scalar_count += int(source.size)
        for local_index, pair_id in enumerate(ids):
            knot = min(
                pair_id * config.temporal_knots // config.pair_count,
                config.temporal_knots - 1,
            )
            source_row = source[local_index]
            target_row = target[local_index]
            count = int(np.prod(source_row.shape[:2]))
            temporal_count[knot] += count
            temporal_source_sum[knot] += source_row.sum(axis=(0, 1))
            temporal_target_sum[knot] += target_row.sum(axis=(0, 1))
            temporal_source_sumsq[knot] += np.square(source_row).sum(axis=(0, 1))
            temporal_target_sumsq[knot] += np.square(target_row).sum(axis=(0, 1))
    source_mean = source_sum / scalar_count
    target_mean = target_sum / scalar_count
    source_std = np.sqrt(max(source_sumsq / scalar_count - source_mean**2, 0.0))
    target_std = np.sqrt(max(target_sumsq / scalar_count - target_mean**2, 0.0))
    if source_std < 1.0e-6 or np.any(temporal_count == 0):
        raise RealizedFlipMenuError("global/temporal ladder has empty support")
    scalar_scale = target_std / source_std
    scalar_offset = target_mean - scalar_scale * source_mean
    scalar_payload = encode_scalar_affine(scalar_scale, scalar_offset)
    count_column = temporal_count[:, None]
    temporal_source_mean = temporal_source_sum / count_column
    temporal_target_mean = temporal_target_sum / count_column
    temporal_source_var = np.maximum(
        temporal_source_sumsq / count_column - np.square(temporal_source_mean),
        0.0,
    )
    temporal_target_var = np.maximum(
        temporal_target_sumsq / count_column - np.square(temporal_target_mean),
        0.0,
    )
    temporal_source_std = np.sqrt(temporal_source_var)
    temporal_target_std = np.sqrt(temporal_target_var)
    if np.any(temporal_source_std < 1.0e-6):
        raise RealizedFlipMenuError("temporal affine source variance is zero")
    temporal_scale = temporal_target_std / temporal_source_std
    temporal_offset = (
        temporal_target_mean - temporal_scale * temporal_source_mean
    )
    temporal_payload = encode_temporal_affine(temporal_scale, temporal_offset)
    receipt = {
        "schema": "ddm_menu1_global_temporal_ladder_receipt.v1",
        "typed_config_sha256": config.stable_hash(),
        "fit_source": "V19C frame1 camera bytes",
        "fit_target": "SHA-bound gt_n600.gt_f1 camera bytes",
        "frame_application": "frame1_only",
        "scalar": {
            "payload_path": str(scalar_path),
            "payload_bytes": len(scalar_payload),
            "payload_sha256": sha256_bytes(scalar_payload),
            "parameter_count": 2,
            "byte_partition": {
                "COUNTED": len(scalar_payload),
                "FREE": 0,
                "NULL": 0,
                "law": "FREE_UNION_NULL_UNION_COUNTED",
            },
        },
        "temporal": {
            "payload_path": str(temporal_path),
            "payload_bytes": len(temporal_payload),
            "payload_sha256": sha256_bytes(temporal_payload),
            "knots": config.temporal_knots,
            "parameter_count": config.temporal_knots * 6,
            "piecewise_policy": "equal_pair_count_segments",
            "support_count_min": int(temporal_count.min()),
            "support_count_max": int(temporal_count.max()),
            "byte_partition": {
                "COUNTED": len(temporal_payload),
                "FREE": 0,
                "NULL": 0,
                "law": "FREE_UNION_NULL_UNION_COUNTED",
            },
        },
        "research_only": True,
        "score_claim": False,
    }
    _publish(scalar_path, scalar_payload)
    _publish(temporal_path, temporal_payload)
    _publish_json(receipt_path, receipt)
    return scalar_payload, temporal_payload, receipt


def _checkpoint_paths(root: Path, candidate_id: str, start: int, stop: int) -> tuple[Path, Path]:
    stage = root / "02_measurements" / candidate_id
    return (
        stage / f"batch_{start:04d}_{stop:04d}.json",
        stage / f"batch_{start:04d}_{stop:04d}.npz",
    )


def _measure_candidate(
    *,
    config: Menu1Config,
    candidate_id: str,
    parent_candidate_id: str | None,
    delta_payload_bytes: int,
    transform: str,
    receiver: Any,
    palette: np.ndarray,
    statistics_payload: bytes,
    scalar_payload: bytes,
    temporal_payload: bytes,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    checkpoint_root: Path,
    target_masks: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    print(f"[MENU1] measure {candidate_id}", flush=True)
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        row_path, array_path = _checkpoint_paths(
            checkpoint_root, candidate_id, start, stop
        )
        if row_path.exists() and array_path.exists():
            row = json.loads(_read(row_path))
            if (
                row["typed_config_sha256"] != config.stable_hash()
                or row["candidate_id"] != candidate_id
            ):
                raise RealizedFlipMenuError("measurement resume identity differs")
            rows.append(row)
            continue
        print(
            f"[MENU1] {candidate_id} batch {start:04d}:{stop:04d}",
            flush=True,
        )
        ids = tuple(range(start, stop))
        base_camera = receiver.render_camera_pairs(ids)
        semantic, owned = _semantic_cells(receiver, ids, base_camera, palette)
        if transform == "base":
            camera = base_camera
        elif transform == "scalar_affine":
            camera = apply_scalar_affine(base_camera, scalar_payload)
        elif transform == "temporal_affine":
            camera = apply_temporal_affine(
                base_camera,
                pair_ids=ids,
                pair_count=config.pair_count,
                payload=temporal_payload,
            )
        elif transform == "statistics":
            camera = apply_local_statistics(
                base_camera,
                semantic,
                statistics_payload,
            )
        elif transform == "geometry_statistics":
            camera = _geometry_statistics_camera(
                base_camera=base_camera,
                semantic=semantic,
                owned=owned,
                palette=palette,
                statistics_payload=statistics_payload,
            )
        elif transform.startswith("targeted:"):
            parent_transform = transform.split(":", 1)[1]
            if parent_transform == "base":
                current = base_camera
            elif parent_transform == "scalar_affine":
                current = apply_scalar_affine(base_camera, scalar_payload)
            elif parent_transform == "temporal_affine":
                current = apply_temporal_affine(
                    base_camera,
                    pair_ids=ids,
                    pair_count=config.pair_count,
                    payload=temporal_payload,
                )
            elif parent_transform == "statistics":
                current = apply_local_statistics(
                    base_camera,
                    semantic,
                    statistics_payload,
                )
            elif parent_transform == "geometry_statistics":
                current = _geometry_statistics_camera(
                    base_camera=base_camera,
                    semantic=semantic,
                    owned=owned,
                    palette=palette,
                    statistics_payload=statistics_payload,
                )
            else:
                raise RealizedFlipMenuError("targeted parent transform differs")
            if target_masks is None or start not in target_masks:
                raise RealizedFlipMenuError("targeted mask checkpoint is absent")
            camera = _targeted_camera(
                current=current,
                masks=target_masks[start],
                palette=palette,
            )
        else:
            raise RealizedFlipMenuError(f"unknown candidate transform: {transform}")
        cells, pose6 = _forward(segnet, posenet, camera)
        if start == 0:
            replay_cells, replay_pose6 = _forward(segnet, posenet, camera)
            if not np.array_equal(cells, replay_cells) or not np.array_equal(
                pose6, replay_pose6
            ):
                raise RealizedFlipMenuError("first-batch deterministic replay differs")
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        if parent_candidate_id is None:
            before = target
        else:
            _parent_row, parent_arrays = _checkpoint_paths(
                checkpoint_root,
                parent_candidate_id,
                start,
                stop,
            )
            with np.load(parent_arrays, allow_pickle=False) as stored:
                before = np.asarray(stored["cells"], dtype=np.uint8)
        transition = transition_counts(before=before, after=cells, target=target)
        class_rows = {}
        for class_id, class_name in CLASS_NAMES.items():
            mask = target == class_id
            class_rows[class_name] = {
                "errors": int(np.count_nonzero((cells != target) & mask)),
                "sites": int(np.count_nonzero(mask)),
            }
        row = {
            "schema": "ddm_menu1_measurement_batch.v1",
            "typed_config_sha256": config.stable_hash(),
            "candidate_id": candidate_id,
            "parent_candidate_id": parent_candidate_id,
            "pair_range": [start, stop],
            "errors": int(np.count_nonzero(cells != target)),
            "sites": int(cells.size),
            "pose_squared_error_sum": float(
                np.square(pose6 - target_pose).sum(dtype=np.float64)
            ),
            "pose_coordinates": int(pose6.size),
            "transition_from_parent": transition,
            "per_class": class_rows,
            "camera_sha256": sha256_bytes(camera.tobytes()),
            "cells_sha256": sha256_bytes(cells.tobytes()),
            "pose6_sha256": sha256_bytes(pose6.tobytes()),
            "delta_payload_bytes": delta_payload_bytes,
            "camera_batch_released_after_forward": True,
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "score_claim": False,
        }
        _publish_npz(array_path, cells=cells, pose6=pose6)
        _publish_json(row_path, row)
        rows.append(row)
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    archive_bytes = config.v19c_archive_bytes + delta_payload_bytes
    classes = {
        class_name: {
            "errors": sum(
                int(row["per_class"][class_name]["errors"]) for row in rows
            ),
            "sites": sum(
                int(row["per_class"][class_name]["sites"]) for row in rows
            ),
        }
        for class_name in CLASS_NAMES.values()
    }
    for value in classes.values():
        value["d_seg"] = value["errors"] / value["sites"]
    transition = {
        key: sum(int(row["transition_from_parent"][key]) for row in rows)
        for key in (
            "errors_before",
            "errors_after",
            "errors_corrected",
            "errors_introduced",
            "errors_persisting",
            "delta_errors_realized",
        )
    }
    d_pose = pose_sse / pose_coordinates
    return {
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_candidate_id,
        "transform": transform,
        "archive_bytes": archive_bytes,
        "delta_counted_bytes_vs_v19c": delta_payload_bytes,
        "errors": errors,
        "sites": sites,
        "d_seg": errors / sites,
        "d_pose": d_pose,
        "advisory_objective": advisory_objective(
            errors=errors,
            sites=sites,
            d_pose=d_pose,
            bytes_=archive_bytes,
        ),
        "transition_from_parent": transition,
        "per_class": classes,
        "batch_count": len(rows),
        "batch_size": config.scorer_batch_size,
        "all_batches_checkpointed_and_preserved": True,
        "batch_digest_chain_sha256": sha256_bytes(
            "".join(
                row["cells_sha256"] + row["pose6_sha256"] for row in rows
            ).encode()
        ),
        "pose_stream_present": True,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }


def _target_masks(
    *,
    config: Menu1Config,
    top_row: Mapping[str, Any],
    parent_candidate_id: str,
    labels: np.ndarray,
    checkpoint_root: Path,
) -> tuple[bytes, dict[int, np.ndarray], dict[str, Any]]:
    path_parent = parent_candidate_id.replace("/", "_").replace(":", "_")
    payload_path = checkpoint_root / f"03_targeted_cluster_mask_{path_parent}.bin"
    receipt_path = (
        checkpoint_root / f"03_targeted_cluster_mask_{path_parent}_receipt.json"
    )
    if payload_path.exists() and receipt_path.exists():
        payload = _read(payload_path)
        receipt = json.loads(_read(receipt_path))
        if (
            receipt["typed_config_sha256"] != config.stable_hash()
            or receipt["payload_sha256"] != sha256_bytes(payload)
            or receipt["parent_candidate_id"] != parent_candidate_id
        ):
            raise RealizedFlipMenuError("target-mask resume custody differs")
    else:
        pair_ids = {int(value) for value in top_row["pair_ids"]}
        predicted_name, target_name = str(top_row["ordered_pair"]).split("->")
        predicted_id = CLASS_ORDER.index(predicted_name)
        target_id = CLASS_ORDER.index(target_name)
        band_names = (
            "BOUNDARY_BAND_LE1",
            "ANNULUS_2_TO_5",
            "COSTED_6_TO_8",
            "INTERIOR_GT8",
        )
        band_id = band_names.index(str(top_row["boundary_distance_band"]))
        rows: list[tuple[int, np.ndarray]] = []
        selected = 0
        for start in range(0, config.pair_count, config.scorer_batch_size):
            stop = min(start + config.scorer_batch_size, config.pair_count)
            _row_path, arrays_path = _checkpoint_paths(
                checkpoint_root,
                parent_candidate_id,
                start,
                stop,
            )
            with np.load(arrays_path, allow_pickle=False) as stored:
                predicted = np.asarray(stored["cells"], dtype=np.uint8)
            target = np.asarray(labels[start:stop], dtype=np.uint8)
            bands = np.stack(
                [boundary_distance_bands(value)[1] for value in target],
                axis=0,
            )
            pair_gate = np.asarray(
                [pair_id in pair_ids for pair_id in range(start, stop)],
                dtype=bool,
            )
            mask = (
                pair_gate[:, None, None]
                & (predicted == predicted_id)
                & (target == target_id)
                & (bands == band_id)
            )
            selected += int(np.count_nonzero(mask))
            rows.append((start, np.ascontiguousarray(mask)))
        payload = encode_target_masks(rows)
        receipt = {
            "schema": "ddm_menu1_targeted_cluster_mask_receipt.v1",
            "typed_config_sha256": config.stable_hash(),
            "parent_candidate_id": parent_candidate_id,
            "payload_path": str(payload_path),
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "selected_seg_sites": selected,
            "cluster_rank": int(top_row["menu_rank"]),
            "ordered_pair": top_row["ordered_pair"],
            "target_class": target_name,
            "predicted_class": predicted_name,
            "boundary_distance_band": top_row["boundary_distance_band"],
            "pair_gate_count": len(pair_ids),
            "exact_cluster_dimensions_not_encoded": [
                "d2_band",
                "historical_g3_tail_bucket",
                "error_source_semantic_history",
            ],
            "scope": (
                "FORMULATION: current-parent ordered-pair x boundary-band sites "
                "within the top SN1 cluster's pair support; not an exact replay of "
                "all historical SN1 cluster axes"
            ),
            "research_only": True,
            "score_claim": False,
        }
        _publish(payload_path, payload)
        _publish_json(receipt_path, receipt)
    decoded = decode_target_masks(payload)
    return payload, dict(decoded), receipt


def run(config_path: Path, output_directory: Path) -> Path:
    config, inputs = _config_and_inputs(config_path)
    storage = _storage_preflight(config)
    checkpoint_root = Path(config.checkpoint_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    solve_rows = _solve_rows(config)
    receipt_hashes = {
        "sn1": config.sn1_receipt_sha256,
        "pt1": config.pt1_receipt_sha256,
        "e2": config.e2_receipt_sha256,
        "dr2b": config.dr2b_receipt_sha256,
        "c1": config.c1_ledger_sha256,
        "v19c": config.v19c_receipt_sha256,
        "pa1": config.pa1_receipt_sha256,
    }
    menu = compile_menu_rows(
        solve_rows,
        v19c_residual_errors=config.v19c_residual_errors,
        v19c_total_errors=config.v19c_total_errors,
        receipt_sha256=receipt_hashes,
    )
    menu_path = checkpoint_root / "compiled_realized_flip_menu.jsonl.gz"
    _publish_gzip_jsonl(menu_path, menu)
    print(
        f"[MENU1] compiled {len(menu)} rows from {len(solve_rows)} clusters",
        flush=True,
    )
    menu_identity = {
        "schema": MENU_SCHEMA,
        "row_count": len(menu),
        "cluster_count": len(solve_rows),
        "fix_count": 6,
        "path": str(menu_path),
        "bytes": menu_path.stat().st_size,
        "sha256": _sha256_file(menu_path),
        "active_base": "v19c_endpoint_506fb1df",
        "active_base_residual_errors": config.v19c_residual_errors,
        "active_base_total_errors": config.v19c_total_errors,
        "base_count_distinction": (
            "2265811 is the SN1 residual Road/Undrivable/MyCar bucket; "
            "2923991 is the total frozen-scorer error count used by d_seg"
        ),
        "unpriced_count": sum(
            row["measurement_status"].startswith("UNPRICED") for row in menu
        ),
        "partially_priced_count": sum(
            row["measurement_status"].startswith("PARTIALLY") for row in menu
        ),
        "non_additive_pool_law": (
            "same composition_pool_id rows compete; only telescoping exact "
            "joint-remeasure deltas are summed"
        ),
    }
    receiver = receive_preuint8_q8_archive(inputs["archive"])
    labels = open_stored_npy_memmap(Path(config.target_cache_path), "lstars")
    poses = open_stored_npy_memmap(Path(config.target_cache_path), "gt_poses")
    target_camera = open_stored_npy_memmap(Path(config.target_cache_path), "gt_f1")
    if (
        labels.shape != (config.pair_count, *SEG_HW)
        or poses.shape != (config.pair_count, 6)
        or target_camera.shape != (config.pair_count, *CAMERA_HW, 3)
    ):
        raise RealizedFlipMenuError("target cache n600 geometry differs")
    palette = _palette(receiver)
    statistics_payload, statistics_receipt = _semantic_stage(
        config=config,
        receiver=receiver,
        target_camera=target_camera,
        palette=palette,
        checkpoint_root=checkpoint_root,
    )
    scalar_payload, temporal_payload, global_ladder_receipt = _global_ladder_stage(
        config=config,
        receiver=receiver,
        target_camera=target_camera,
        checkpoint_root=checkpoint_root,
    )
    segnet, posenet, scorer_custody = _load_models(config)
    base = _measure_candidate(
        config=config,
        candidate_id="v19c_base",
        parent_candidate_id=None,
        delta_payload_bytes=0,
        transform="base",
        receiver=receiver,
        palette=palette,
        statistics_payload=statistics_payload,
        scalar_payload=scalar_payload,
        temporal_payload=temporal_payload,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        checkpoint_root=checkpoint_root,
    )
    if (
        base["errors"] != config.v19c_total_errors
        or abs(
            base["d_pose"]
            - float(
                inputs["receipts"]["v19c"]["curve"]["n600_endpoint"]["d_pose"]
            )
        )
        > 5.0e-10
    ):
        raise RealizedFlipMenuError("fresh V19C base replay differs from receipt")
    base["candidate_id"] = "v19c_base"
    scalar = _measure_candidate(
        config=config,
        candidate_id="scalar_gain_bias_12b_frame1",
        parent_candidate_id=base["candidate_id"],
        delta_payload_bytes=len(scalar_payload),
        transform="scalar_affine",
        receiver=receiver,
        palette=palette,
        statistics_payload=statistics_payload,
        scalar_payload=scalar_payload,
        temporal_payload=temporal_payload,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        checkpoint_root=checkpoint_root,
    )
    temporal = _measure_candidate(
        config=config,
        candidate_id="temporal_affine_16knot_frame1",
        parent_candidate_id=base["candidate_id"],
        delta_payload_bytes=len(temporal_payload),
        transform="temporal_affine",
        receiver=receiver,
        palette=palette,
        statistics_payload=statistics_payload,
        scalar_payload=scalar_payload,
        temporal_payload=temporal_payload,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        checkpoint_root=checkpoint_root,
    )
    statistics = _measure_candidate(
        config=config,
        candidate_id="local_statistics_16band_frame1",
        parent_candidate_id=base["candidate_id"],
        delta_payload_bytes=len(statistics_payload),
        transform="statistics",
        receiver=receiver,
        palette=palette,
        statistics_payload=statistics_payload,
        scalar_payload=scalar_payload,
        temporal_payload=temporal_payload,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        checkpoint_root=checkpoint_root,
    )
    geometry = _measure_candidate(
        config=config,
        candidate_id="statistics_hard_analytic_composed_frame1",
        parent_candidate_id=base["candidate_id"],
        delta_payload_bytes=len(statistics_payload),
        transform="geometry_statistics",
        receiver=receiver,
        palette=palette,
        statistics_payload=statistics_payload,
        scalar_payload=scalar_payload,
        temporal_payload=temporal_payload,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        checkpoint_root=checkpoint_root,
    )
    pool_candidates = [scalar, temporal, statistics, geometry]
    improving = [
        row
        for row in pool_candidates
        if row["archive_bytes"] <= config.byte_budget
        and row["advisory_objective"] < base["advisory_objective"]
    ]
    current = (
        min(improving, key=lambda row: row["advisory_objective"])
        if improving
        else base
    )
    transform_by_candidate = {
        "v19c_base": "base",
        "scalar_gain_bias_12b_frame1": "scalar_affine",
        "temporal_affine_16knot_frame1": "temporal_affine",
        "local_statistics_16band_frame1": "statistics",
        "statistics_hard_analytic_composed_frame1": "geometry_statistics",
    }
    current_transform = transform_by_candidate[current["candidate_id"]]
    proposals = []
    for row in pool_candidates:
        selected = row["candidate_id"] == current["candidate_id"]
        proposals.append(
            {
                **row,
                "admitted": selected,
                "admission_gates": {
                    "within_byte_budget": row["archive_bytes"]
                    <= config.byte_budget,
                    "strict_joint_improvement_vs_base": row[
                        "advisory_objective"
                    ]
                    < base["advisory_objective"],
                    "same_pool_winner": selected,
                },
                "admission_reason": (
                    "STRICT_JOINT_POOL_WINNER"
                    if selected
                    else (
                        "NO_JOINT_GAIN"
                        if row["advisory_objective"]
                        >= base["advisory_objective"]
                        else "SAME_POOL_DOMINATED"
                    )
                ),
            }
        )
    target_payload, target_masks, target_receipt = _target_masks(
        config=config,
        top_row=solve_rows[0],
        parent_candidate_id=current["candidate_id"],
        labels=labels,
        checkpoint_root=checkpoint_root,
    )
    targeted = _measure_candidate(
        config=config,
        candidate_id="top_sn1_cluster_targeted_prototype_frame1",
        parent_candidate_id=current["candidate_id"],
        delta_payload_bytes=(
            current["delta_counted_bytes_vs_v19c"] + len(target_payload)
        ),
        transform=f"targeted:{current_transform}",
        receiver=receiver,
        palette=palette,
        statistics_payload=statistics_payload,
        scalar_payload=scalar_payload,
        temporal_payload=temporal_payload,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        checkpoint_root=checkpoint_root,
        target_masks=target_masks,
    )
    targeted_curve = greedy_telescoping_curve(
        base=current,
        proposals=[targeted],
        byte_budget=config.byte_budget,
    )[-1]
    proposals.append(targeted_curve)
    if targeted_curve["admitted"]:
        current = targeted
    curve = [{**base, "admitted": True, "admission_reason": "BASE"}]
    selected_pool_row = next(
        (row for row in proposals[:-1] if row["admitted"]),
        None,
    )
    if selected_pool_row is not None:
        curve.append(selected_pool_row)
    curve.append(targeted_curve)
    in_seg_box = (
        current["archive_bytes"] <= config.byte_budget
        and current["d_seg"] <= config.target_d_seg
    )
    in_joint_box = in_seg_box and current["d_pose"] <= config.target_d_pose
    residual_by_class = {
        name: int(row["errors"]) for name, row in current["per_class"].items()
    }
    binding = max(residual_by_class, key=residual_by_class.get)
    measured_rows = [
        {
            "row_id": row["candidate_id"],
            "cluster_id": (
                "GLOBAL"
                if row["candidate_id"]
                != "top_sn1_cluster_targeted_prototype_frame1"
                else menu[0]["cluster_id"]
            ),
            "mechanism_bucket": {
                "scalar_gain_bias_12b_frame1": "BN_SE_AMPLITUDE_STATISTICS",
                "temporal_affine_16knot_frame1": (
                    "BN_SE_TEMPORAL_AMPLITUDE_STATISTICS"
                ),
                "local_statistics_16band_frame1": "BN_SE_AMPLITUDE_STATISTICS",
                "statistics_hard_analytic_composed_frame1": (
                    "COMPOSED_PAINT_GEOMETRY"
                ),
                "top_sn1_cluster_targeted_prototype_frame1": (
                    "COARSE_DESCRIPTION_TARGETED"
                ),
            }[row["candidate_id"]],
            "composition_pool_id": {
                "scalar_gain_bias_12b_frame1": "paint_amplitude:GLOBAL",
                "temporal_affine_16knot_frame1": "paint_amplitude:GLOBAL",
                "local_statistics_16band_frame1": "paint_amplitude:GLOBAL",
                "statistics_hard_analytic_composed_frame1": (
                    "paint_pipeline:GLOBAL"
                ),
                "top_sn1_cluster_targeted_prototype_frame1": (
                    f"semantic_target:{menu[0]['cluster_id']}"
                ),
            }[row["candidate_id"]],
            "parent_candidate_id": row["parent_candidate_id"],
            "delta_errors_realized": row["transition_from_parent"][
                "delta_errors_realized"
            ],
            "delta_counted_bytes": (
                row["archive_bytes"]
                - next(
                    value["archive_bytes"]
                    for value in curve
                    if value["candidate_id"] == row["parent_candidate_id"]
                )
            ),
            "byte_partition": {
                "COUNTED": (
                    row["archive_bytes"]
                    - next(
                        value["archive_bytes"]
                        for value in curve
                        if value["candidate_id"] == row["parent_candidate_id"]
                    )
                ),
                "FREE": 0,
                "NULL": 0,
                "law": "FREE_UNION_NULL_UNION_COUNTED",
            },
            "collateral_retained": {
                "errors_corrected": row["transition_from_parent"][
                    "errors_corrected"
                ],
                "errors_introduced": row["transition_from_parent"][
                    "errors_introduced"
                ],
                "errors_persisting": row["transition_from_parent"][
                    "errors_persisting"
                ],
            },
            "d_seg": row["d_seg"],
            "d_pose": row["d_pose"],
            "archive_bytes": row["archive_bytes"],
            "admitted": row["admitted"],
            "admission_gates": row["admission_gates"],
            "admission_reason": row["admission_reason"],
            "measurement_status": "MEASURED_EXACT_V19C_BASE_CHAIN",
            "waterfill_eligible": True,
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "score_claim": False,
        }
        for row in proposals
    ]
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "delegation_checkpoint_key": DELEGATION_KEY,
        "typed_config_sha256": config.stable_hash(),
        "full_evidence_config_sha256": config.full_hash(),
        "input_receipt_sha256": receipt_hashes,
        "menu": menu_identity,
        "bridge_blocker_row": inputs["bridge"],
        "bn_free_affine_bridge_row": inputs["bn_free_bridge"],
        "pose_amplitude_pool": inputs["pose_amplitude_pool"],
        "global_temporal_ladder_payloads": global_ladder_receipt,
        "local_statistics_payload": statistics_receipt,
        "targeted_cluster_payload": target_receipt,
        "measured_menu_rows": measured_rows,
        "dominated_pool_rows": [
            {
                "row_id": "pt1_stratum_spectrum_cross_control",
                "composition_pool_id": "paint_amplitude:GLOBAL",
                "measurement_status": "DOMINATED_CROSS_CONTROL_NOT_ADMITTED",
                "spectrum_bytes": 186,
                "spectrum_errors": 1_034_847,
                "dominator_row_id": "pt1_global_amplitude_statistics_cross_control",
                "dominator_bytes": 30,
                "dominator_errors": 1_016_725,
                "active_base_price_imported": False,
                "reason": "more bytes and more errors under the same PT1 control",
                "byte_partition": {
                    "COUNTED": 186,
                    "FREE": 0,
                    "NULL": 0,
                    "law": "FREE_UNION_NULL_UNION_COUNTED",
                },
            }
        ],
        "curve": curve,
        "box": {
            "archive_bytes_max": config.byte_budget,
            "target_errors": config.target_errors,
            "d_seg_max": config.target_d_seg,
            "d_pose_max": config.target_d_pose,
            "pose_stream_required": True,
            "final_archive_bytes": current["archive_bytes"],
            "final_errors": current["errors"],
            "final_d_seg": current["d_seg"],
            "final_d_pose": current["d_pose"],
            "gap_errors": current["errors"] - config.target_errors,
            "gap_d_seg": current["d_seg"] - config.target_d_seg,
            "seg_box_entered": in_seg_box,
            "joint_box_entered": in_joint_box,
        },
        "residual": {
            "per_class_errors": residual_by_class,
            "binding_bucket": binding,
            "routing": (
                "R6_CANDIDATE_MAIN_REVIEW_REQUIRED"
                if in_joint_box
                else f"ROUTE_RS1_366_{binding.upper()}_RESIDUAL_AND_POSE_FINISH"
            ),
        },
        "waterfill": {
            "method": (
                "same-pool exact n600 alternative competition followed by "
                "dependency-ordered targeted joint remeasure; independent "
                "same-pool deltas never summed"
            ),
            "score_byte_dual": RATE_DUAL,
            "terminal_candidate_id": current["candidate_id"],
            "terminal_advisory_objective": current["advisory_objective"],
        },
        "scorer_custody": scorer_custody,
        "storage_preflight": storage,
        "artifact_hygiene": {
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_tree": (
                "immutable JSON+NPZ batch rows plus counted payload bytes; "
                "preserved on SSD"
            ),
            "target_cache_read_only": True,
            "frontier_archive_mutated": False,
            "paid_or_remote_dispatch": False,
            "training_launch": False,
        },
        "first_round_adversarial_review": {
            "v19c_count_premise": (
                "FIXED: delegated 2265811 count is the residual "
                "Road/Undrivable/MyCar bucket, not total d_seg errors; typed "
                "separately from the receipt-derived 2923991 total"
            ),
            "frame_application": (
                "FIXED: gt_f1-fitted paint acts on frame1 only; frame0 is "
                "preserved for PoseNet instead of receiving an ungrounded "
                "copy of frame1 semantics"
            ),
            "target_cluster_scope": (
                "OPEN_FORMULATION_LIMIT: sidecar does not encode d2, historical "
                "g3-tail, or semantic-history axes"
            ),
        },
        "triality": {
            "dsl_data": str(
                output_directory / "ddm_menu1_realized_flip_menu_receipt.json"
            ),
            "dag_feed": ".omx/research/ddm_menu1_realized_flip_menu_DAG_FEED_20260723.md",
            "canonical_equations": (
                ".omx/research/"
                "ddm_menu1_realized_flip_menu_canonical_equations_20260723.md"
            ),
        },
        "directives_consumed": [
            {
                "source": "delegated authority file",
                "effect": (
                    "compile all 2649x6 rows; price three active-base arms; "
                    "exact greedy joint remeasure; no exact contest eval"
                ),
            },
            {
                "source": "operator fleet broadcast 2026-07-19T19:42:07Z",
                "effect": (
                    "admit only measured exact joint objective gain per byte "
                    "above the 25/37545489 rate dual; never blanket-sum rows"
                ),
            },
            {
                "source": "operator fleet broadcast 2026-07-19T19:48:01Z",
                "effect": (
                    "retain SN1 margin/Fisher-ranked cluster order, measure the "
                    "realized inner chain, and introduce no Fourier residual"
                ),
            },
            {
                "source": "MAIN inbox 2026-07-23T22:18:26Z",
                "effect": (
                    "add 12-byte scalar and 16-knot temporal affine rungs; "
                    "retain stats-over-spectrum dominance and Pose collateral"
                ),
            },
            {
                "source": "MAIN inbox 2026-07-23T22:32:18Z",
                "effect": (
                    "partition every byte price as COUNTED/FREE/NULL; consume "
                    "AT1X FREE-affine source only through a real uint8 projection"
                ),
            },
            {
                "source": "MAIN inbox 2026-07-23T23:24:38Z",
                "effect": (
                    "import PA1 pose-amplitude rows as a distinct cross-control "
                    "pool; mark 0-byte rows FREE_candidate pending survival"
                ),
            },
        ],
        "verdict": (
            "R6_CANDIDATE_MAIN_REVIEW_REQUIRED"
            if in_joint_box
            else "MENU1_MEASURED_BOX_NOT_REACHED"
        ),
        "verdict_scope": (
            "FORMULATION: V19C endpoint x scalar/temporal/local statistics pool "
            "x fixed decoder-derived hard/analytic composition x counted "
            "top-cluster ordered-pair/boundary mask; families remain open"
        ),
        "pointer": {
            "score": "0.1910828242",
            "axis": "[contest-CPU]",
            "moved": False,
        },
        "main_landing_review_required": True,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "execution_allowed": True,
    }
    receipt_path = output_directory / "ddm_menu1_realized_flip_menu_receipt.json"
    _publish_json(receipt_path, receipt)
    return receipt_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute:
        raise SystemExit("execution refused: pass --execute for authorized local n600")
    path = run(_resolve(args.config), _resolve(args.output_directory))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
