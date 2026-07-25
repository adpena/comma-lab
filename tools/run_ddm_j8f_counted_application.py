#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the bounded, resumable DDM j8f DM4-to-J5 application smoke."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_dm4_j5_counted_application import (  # noqa: E402
    EVIDENCE_AXIS,
    HORIZON_GAP,
    POINTER,
    RANGE_GAUGE_POLICY,
    VALIDITY_GAP,
    DDMCountedApplicationConfigV1,
    DDMCountedApplicationError,
    SparseJ5CoordinateEffectV1,
    apply_coordinate_choices,
    descriptor_camera_delta,
    enumerate_j5_coordinate_effects,
    exact_joint_delta,
    load_ms4d_pair_metric,
    select_counted_application,
)
from tac.optimization.ddm_dm4_targeted_realization_cures import (  # noqa: E402
    _bound_inputs as bind_dm4_inputs,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    AdamStateV1,
    DirectDescriptionError,
    DirectDescriptionJointDescentTypedConfigV1,
    compile_parameterized_archive,
    lift_v15_archive,
    load_stage_checkpoint,
    realize_parameter_theta,
)
from tac.optimization.resize_full_kernel import FullResizeKernel  # noqa: E402
from tac.optimization.solve_diff_operator_mining import (  # noqa: E402
    _load_production_inputs,
    _open_production_inputs,
)
from tools.launch_ddm_joint_descent import (  # noqa: E402
    _chunked_n600_verdict,
    _load_cpu_frozen_scorers,
)

RUN_RECEIPT_SCHEMA = "ddm_j8f_counted_application_smoke.v1"
CHECKPOINT_SCHEMA = "ddm_j8f_counted_application_checkpoint.v1"
PREFLIGHT_SCHEMA = "ddm_j8f_counted_application_preflight.v1"
READY_TICKET_SCHEMA = "ddm_j8f_ready_to_fire_ticket.v1"
BLOCKER_SCHEMA = "ddm_j8f_counted_application_blocker.v1"
INVENTORY_CHECKPOINT_SCHEMA = "ddm_j8f_pair_inventory_checkpoint.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DDMCountedApplicationError(f"refusing to overwrite unequal artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_bytes(payload) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise DDMCountedApplicationError(f"JSON artifact is not an object: {path}")
    return value


def _preflight_path(config: DDMCountedApplicationConfigV1) -> Path:
    return (
        Path(config.output_root)
        / "preflight"
        / f"fresh_memory_storage_preflight.{config.typed_hash()}.json"
    )


def _inventory_identity(
    *,
    config: DDMCountedApplicationConfigV1,
    pair_id: int,
    base_archive: bytes,
    theta: np.ndarray,
    parameter_names: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    realized = np.ascontiguousarray(theta, dtype="<f4")
    payload = {
        "schema": "ddm_j8f_pair_inventory_identity.v1",
        "pair_id": int(pair_id),
        "include_lane_programs": False,
        "base_archive_bytes": len(base_archive),
        "base_archive_sha256": _sha256(base_archive),
        "theta_sha256_fp32le": _sha256(realized.tobytes()),
        "parameter_names_sha256": _sha256(_canonical_bytes(list(parameter_names))),
        "operator_source_sha256": config.source_bindings["operator_source"].sha256,
        "step4_checkpoint_sha256": config.source_bindings["step4_checkpoint"].sha256,
    }
    return _sha256(_canonical_bytes(payload)), payload


def _inventory_checkpoint_paths(
    *,
    config: DDMCountedApplicationConfigV1,
    pair_id: int,
    identity_sha256: str,
) -> tuple[Path, Path]:
    stem = f"pair_{int(pair_id):03d}.{identity_sha256}"
    root = Path(config.output_root) / "pair_inventories"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _encode_inventory_effects(
    effects: Sequence[SparseJ5CoordinateEffectV1],
) -> tuple[bytes, list[dict[str, Any]]]:
    if not effects:
        raise DDMCountedApplicationError("cannot checkpoint an empty J5 inventory")
    offsets = [0]
    rows: list[dict[str, Any]] = []
    index_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for effect in effects:
        start = offsets[-1]
        stop = start + int(effect.flat_indices.size)
        offsets.append(stop)
        index_parts.append(np.asarray(effect.flat_indices, dtype="<i8"))
        value_parts.append(np.asarray(effect.values, dtype="<i2"))
        rows.append(
            {
                "effect": effect.to_payload(),
                "camera_shape": list(effect.camera_shape),
                "start": start,
                "stop": stop,
            }
        )
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        offsets=np.asarray(offsets, dtype="<i8"),
        flat_indices=np.concatenate(index_parts),
        values=np.concatenate(value_parts),
    )
    return stream.getvalue(), rows


def _write_inventory_checkpoint(
    *,
    config: DDMCountedApplicationConfigV1,
    pair_id: int,
    base_archive: bytes,
    theta: np.ndarray,
    parameter_names: Sequence[str],
    effects: Sequence[SparseJ5CoordinateEffectV1],
    inventory_receipt: Mapping[str, Any],
) -> tuple[tuple[SparseJ5CoordinateEffectV1, ...], dict[str, Any], dict[str, Any]]:
    identity_sha256, identity = _inventory_identity(
        config=config,
        pair_id=pair_id,
        base_archive=base_archive,
        theta=theta,
        parameter_names=parameter_names,
    )
    data_path, manifest_path = _inventory_checkpoint_paths(
        config=config,
        pair_id=pair_id,
        identity_sha256=identity_sha256,
    )
    data, rows = _encode_inventory_effects(effects)
    _atomic_write(data_path, data)
    manifest = {
        "schema": INVENTORY_CHECKPOINT_SCHEMA,
        "status": "PRESERVED",
        "identity_sha256": identity_sha256,
        "identity": identity,
        "data": {
            "path": str(data_path),
            "bytes": len(data),
            "sha256": _sha256(data),
            "format": "numpy_npz_allow_pickle_false",
        },
        "effect_count": len(rows),
        "effects": rows,
        "inventory_receipt": dict(inventory_receipt),
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    _write_json(manifest_path, manifest)
    binding = {
        "path": str(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path.read_bytes()),
        "data": manifest["data"],
        "identity_sha256": identity_sha256,
    }
    return tuple(effects), dict(inventory_receipt), binding


def _load_inventory_checkpoint(
    *,
    config: DDMCountedApplicationConfigV1,
    pair_id: int,
    base_archive: bytes,
    theta: np.ndarray,
    parameter_names: Sequence[str],
) -> (
    tuple[
        tuple[SparseJ5CoordinateEffectV1, ...],
        dict[str, Any],
        dict[str, Any],
    ]
    | None
):
    identity_sha256, identity = _inventory_identity(
        config=config,
        pair_id=pair_id,
        base_archive=base_archive,
        theta=theta,
        parameter_names=parameter_names,
    )
    data_path, manifest_path = _inventory_checkpoint_paths(
        config=config,
        pair_id=pair_id,
        identity_sha256=identity_sha256,
    )
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    data_binding = manifest.get("data")
    rows = manifest.get("effects")
    if (
        manifest.get("schema") != INVENTORY_CHECKPOINT_SCHEMA
        or manifest.get("status") != "PRESERVED"
        or manifest.get("identity_sha256") != identity_sha256
        or manifest.get("identity") != identity
        or not isinstance(data_binding, Mapping)
        or data_binding.get("path") != str(data_path)
        or not isinstance(rows, list)
        or manifest.get("effect_count") != len(rows)
        or not isinstance(manifest.get("inventory_receipt"), Mapping)
    ):
        raise DDMCountedApplicationError(
            f"J5 pair inventory manifest identity differs: {manifest_path}"
        )
    data = data_path.read_bytes()
    if (
        len(data) != data_binding.get("bytes")
        or _sha256(data) != data_binding.get("sha256")
    ):
        raise DDMCountedApplicationError(
            f"J5 pair inventory data binding differs: {data_path}"
        )
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        if set(arrays.files) != {"offsets", "flat_indices", "values"}:
            raise DDMCountedApplicationError("J5 pair inventory array set differs")
        offsets = np.asarray(arrays["offsets"], dtype=np.int64)
        indices = np.asarray(arrays["flat_indices"], dtype=np.int64)
        values = np.asarray(arrays["values"], dtype=np.int16)
    if (
        offsets.shape != (len(rows) + 1,)
        or offsets[0] != 0
        or offsets[-1] != indices.size
        or indices.shape != values.shape
        or np.any(np.diff(offsets) <= 0)
    ):
        raise DDMCountedApplicationError("J5 pair inventory offsets differ")
    effects: list[SparseJ5CoordinateEffectV1] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("effect"), Mapping):
            raise DDMCountedApplicationError("J5 pair inventory effect row differs")
        effect_payload = row["effect"]
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        if row.get("start") != start or row.get("stop") != stop:
            raise DDMCountedApplicationError("J5 pair inventory row offsets differ")
        effect = SparseJ5CoordinateEffectV1(
            coordinate_index=int(effect_payload["coordinate_index"]),
            coordinate_name=str(effect_payload["coordinate_name"]),
            direction=int(effect_payload["direction"]),
            pair_id=int(effect_payload["pair_id"]),
            flat_indices=indices[start:stop],
            values=values[start:stop],
            camera_shape=tuple(int(value) for value in row["camera_shape"]),
            archive_bytes=int(effect_payload["archive_bytes"]),
            archive_sha256=str(effect_payload["archive_sha256"]),
            archive_byte_delta=int(effect_payload["archive_byte_delta"]),
            changed_channel_values=int(effect_payload["changed_channel_values"]),
        )
        if effect.to_payload() != effect_payload:
            raise DDMCountedApplicationError(
                "J5 pair inventory reconstructed payload differs"
            )
        effects.append(effect)
    binding = {
        "path": str(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path.read_bytes()),
        "data": dict(data_binding),
        "identity_sha256": identity_sha256,
    }
    return tuple(effects), dict(manifest["inventory_receipt"]), binding


def _load_or_build_inventory(
    *,
    config: DDMCountedApplicationConfigV1,
    lift: Any,
    theta: np.ndarray,
    pair_id: int,
    base_archive: bytes,
) -> tuple[
    tuple[SparseJ5CoordinateEffectV1, ...],
    dict[str, Any],
    dict[str, Any],
]:
    resumed = _load_inventory_checkpoint(
        config=config,
        pair_id=pair_id,
        base_archive=base_archive,
        theta=theta,
        parameter_names=lift.parameter_names,
    )
    if resumed is not None:
        return resumed
    _archive, _camera, effects, inventory = enumerate_j5_coordinate_effects(
        lift=lift,
        theta=theta,
        pair_id=pair_id,
        include_lane_programs=False,
    )
    if _archive != base_archive:
        raise DDMCountedApplicationError("J5 inventory base archive differs")
    del _camera
    return _write_inventory_checkpoint(
        config=config,
        pair_id=pair_id,
        base_archive=base_archive,
        theta=theta,
        parameter_names=lift.parameter_names,
        effects=effects,
        inventory_receipt=inventory,
    )


def _binding_path(
    config: DDMCountedApplicationConfigV1, name: str
) -> Path:
    return config.source_bindings[name].resolve(config.repo_root)


def _validate_output_root(path: Path) -> str:
    resolved = path.resolve()
    tiers = (
        Path("/Volumes/VertigoDataTier/pact").resolve(),
        Path("/Volumes/APDataStore/pact").resolve(),
    )
    for tier in tiers:
        if resolved == tier or resolved.is_relative_to(tier):
            return str(tier)
    raise DDMCountedApplicationError(
        "J8f bulk output must use the SSD waterfall; local bulk is refused"
    )


def _load_step4(
    config: DDMCountedApplicationConfigV1,
) -> tuple[
    DirectDescriptionJointDescentTypedConfigV1,
    AdamStateV1,
    dict[str, Any],
    Any,
    bytes,
]:
    ticket = _binding_path(config, "step4_ticket")
    typed = DirectDescriptionJointDescentTypedConfigV1.from_ticket(ticket)
    state, metadata = load_stage_checkpoint(
        _binding_path(config, "step4_checkpoint"),
        config=typed,
    )
    source = Path(typed.source_archive_path).read_bytes()
    lift = lift_v15_archive(source)
    archive, realized = compile_parameterized_archive(
        lift,
        state.theta,
        include_lane_programs=False,
    )
    expected = metadata.get("realized_archive", {})
    if (
        state.step != 4
        or realized.size != 368
        or len(archive) != int(expected.get("bytes", -1))
        or _sha256(archive) != expected.get("sha256")
        or expected.get("lane_programs_materialized") is not False
    ):
        raise DDMCountedApplicationError("Step-4 reconstructed archive custody differs")
    return typed, state, metadata, lift, archive


def _load_proposals(
    config: DDMCountedApplicationConfigV1,
    *,
    step4_archive: bytes,
) -> tuple[
    DirectDescriptionJointDescentTypedConfigV1,
    tuple[Any, ...],
    dict[str, Any],
]:
    j8e = DirectDescriptionJointDescentTypedConfigV1.from_ticket(
        _binding_path(config, "j8e_ticket")
    )
    unchanged, proposals, adapter_receipt = j8e.dm4_j5_proposal_source(
        base_archive=step4_archive,
        enabled=True,
    )
    if unchanged != step4_archive or len(proposals) != 6:
        raise DDMCountedApplicationError("J8e DM4 proposal source differs")
    return j8e, proposals, adapter_receipt


def _dm4_plane_source(
    config: DDMCountedApplicationConfigV1,
) -> tuple[dict[str, Any], Any, Any]:
    dm4_config = _read_json(_binding_path(config, "dm4_config"))
    _dm2_config, _dm2, _dm1, _index_receipt, source_config = bind_dm4_inputs(
        dm4_config
    )
    context = _open_production_inputs(source_config)
    return dm4_config, source_config, context


def _load_pair_planes(
    *,
    pair_id: int,
    source_config: Any,
    context: Any,
    kernel: FullResizeKernel,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    chunk = _load_production_inputs(
        context,
        source_config,
        [int(pair_id)],
        kernel,
    )
    return (
        np.ascontiguousarray(chunk.predictor_planes[0]),
        np.ascontiguousarray(chunk.solved_planes[0]),
        dict(chunk.source_hashes),
    )


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return 0


def _available_memory() -> tuple[int, int]:
    try:
        import psutil
    except ImportError as exc:
        raise DDMCountedApplicationError(
            "fresh memory preflight requires psutil availability"
        ) from exc
    value = psutil.virtual_memory()
    return int(value.total), int(value.available)


def _configure_torch_threads(expected: int = 4) -> int:
    if expected != 4:
        raise DDMCountedApplicationError("J8f Torch thread contract differs")
    import torch

    torch.set_num_threads(expected)
    observed = int(torch.get_num_threads())
    if observed != expected:
        raise DDMCountedApplicationError(
            f"REFUSE_J8F_TORCH_THREADS: observed {observed}, expected {expected}"
        )
    return observed


def _j8e_memory_receipt(
    config: DDMCountedApplicationConfigV1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    j8e = _read_json(_binding_path(config, "j8e_ticket"))
    binding = j8e.get("execution_custody", {}).get("worst_geometry_memory_receipt")
    if not isinstance(binding, Mapping):
        raise DDMCountedApplicationError("J8e ticket lacks measured memory custody")
    path = Path(str(binding["path"]))
    raw = path.read_bytes()
    if _sha256(raw) != binding.get("sha256"):
        raise DDMCountedApplicationError("J8e memory receipt SHA differs")
    receipt = json.loads(raw)
    if (
        receipt.get("schema") != "ddm_joint_descent_memory_preflight.v1"
        or receipt.get("admission") is not True
        or receipt.get("consumer_window")
        != (
            "sealed stage3 max window: all receiver-effective groups, 52 realized "
            "secants, paint/uint8-STE/R/frozen MLX scorer forward-backward"
        )
    ):
        raise DDMCountedApplicationError("J8e memory receipt authority differs")
    return dict(binding), receipt


def _fresh_resume_admission(
    *,
    config: DDMCountedApplicationConfigV1,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    observed_torch_threads = _configure_torch_threads(config.torch_threads)
    total_memory, available_memory = _available_memory()
    memory_binding, memory_receipt = _j8e_memory_receipt(config)
    projected_gib = float(memory_receipt["projected_peak_gib"])
    ceiling_gib = float(memory_receipt["operator_ceiling_gib"])
    system_used_gib = (total_memory - available_memory) / 2**30
    memory_admission = system_used_gib + projected_gib <= ceiling_gib
    storage = preflight.get("storage")
    if not isinstance(storage, Mapping):
        raise DDMCountedApplicationError("J8f preflight storage custody differs")
    output_root = Path(config.output_root)
    tier = _validate_output_root(output_root)
    disk = shutil.disk_usage(output_root)
    required_bytes = int(storage.get("derived_required_bytes", -1))
    storage_admission = required_bytes > 0 and disk.free >= required_bytes
    if not memory_admission:
        raise DDMCountedApplicationError(
            "REFUSE_J8F_FRESH_MEMORY_RESUME: current system use exceeds measured ceiling"
        )
    if not storage_admission:
        raise DDMCountedApplicationError(
            "REFUSE_J8F_FRESH_STORAGE_RESUME: current SSD free bytes are insufficient"
        )
    return {
        "schema": "ddm_j8f_fresh_resume_admission.v1",
        "checked_at_unix_ns": time.time_ns(),
        "memory": {
            "fresh_system_total_gib": total_memory / 2**30,
            "fresh_system_available_gib": available_memory / 2**30,
            "fresh_system_used_gib": system_used_gib,
            "bound_j8e_receipt": memory_binding,
            "bound_j8e_projected_peak_gib": projected_gib,
            "operator_ceiling_gib": ceiling_gib,
            "admission": True,
        },
        "storage": {
            "tier": tier,
            "output_root": str(output_root),
            "fresh_free_bytes": disk.free,
            "derived_required_bytes": required_bytes,
            "admission": True,
        },
        "observed_torch_threads": observed_torch_threads,
        "admission": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def _preflight(
    config: DDMCountedApplicationConfigV1,
) -> dict[str, Any]:
    if _preflight_path(config).exists():
        return _load_preflight(config)
    started = time.monotonic()
    bindings = config.validate_all_bindings()
    observed_torch_threads = _configure_torch_threads(config.torch_threads)
    output_root = Path(config.output_root)
    tier = _validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total_memory, available_memory = _available_memory()
    memory_binding, memory_receipt = _j8e_memory_receipt(config)
    projected_gib = float(memory_receipt["projected_peak_gib"])
    ceiling_gib = float(memory_receipt["operator_ceiling_gib"])
    system_used_gib = (total_memory - available_memory) / 2**30
    admission = system_used_gib + projected_gib <= ceiling_gib
    if not admission:
        raise DDMCountedApplicationError(
            "REFUSE_J8F_FRESH_MEMORY_PREFLIGHT: projected system use exceeds measured ceiling"
        )

    typed, state, metadata, lift, step4_archive = _load_step4(config)
    _j8e, proposals, adapter = _load_proposals(
        config, step4_archive=step4_archive
    )
    pair_ids = tuple(sorted({int(row.aimed_cell["pair_id"]) for row in proposals}))
    inventories: dict[str, Any] = {}
    for pair_id in pair_ids:
        effects, inventory, checkpoint = _load_or_build_inventory(
            config=config,
            lift=lift,
            theta=state.theta,
            pair_id=pair_id,
            base_archive=step4_archive,
        )
        inventories[str(pair_id)] = {
            **inventory,
            "sparse_effect_payload_bytes": int(
                sum(
                    effect.flat_indices.nbytes + effect.values.nbytes
                    for effect in effects
                )
            ),
            "preserved_checkpoint": checkpoint,
        }
        print(
            json.dumps(
                {
                    "stage": "pair_inventory_preflight",
                    "pair_id": pair_id,
                    "effect_count": len(effects),
                    "checkpoint_sha256": checkpoint["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del effects
        gc.collect()

    disk = shutil.disk_usage(output_root)
    derived_required = (
        (config.smoke_horizon + 4)
        * config.source_bindings["step4_checkpoint"].bytes
        + 4 * len(step4_archive)
    )
    if disk.free < derived_required:
        raise DDMCountedApplicationError(
            "REFUSE_J8F_STORAGE_PREFLIGHT: insufficient SSD bytes for preserved checkpoints"
        )
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": config.run_id,
        "lane_id": config.lane_id,
        "typed_config_hash": config.typed_hash(),
        "validated_bindings": bindings,
        "step4": {
            "step": state.step,
            "checkpoint_metadata_sha256": _sha256(_canonical_bytes(metadata)),
            "archive_bytes": len(step4_archive),
            "archive_sha256": _sha256(step4_archive),
            "parameter_count": int(state.theta.size),
            "dsl_compile_hash": typed.dsl_compile_hash,
        },
        "proposal_adapter": adapter,
        "proposal_count": len(proposals),
        "pair_effect_inventories": inventories,
        "memory": {
            "fresh_system_total_gib": total_memory / 2**30,
            "fresh_system_available_gib": available_memory / 2**30,
            "fresh_system_used_gib": system_used_gib,
            "fresh_post_inventory_process_rss_gib": _rss_bytes() / 2**30,
            "bound_j8e_receipt": memory_binding,
            "bound_j8e_measured_peak_rss_gib": float(
                memory_receipt["maximum_measured_peak_rss_gib"]
            ),
            "bound_j8e_projected_peak_gib": projected_gib,
            "operator_ceiling_gib": ceiling_gib,
            "admission_formula": (
                "fresh_system_used_gib + bound_real_config_projected_peak_gib "
                "<= measured_operator_ceiling_gib"
            ),
            "admission": admission,
        },
        "storage": {
            "tier": tier,
            "output_root": str(output_root),
            "free_bytes": disk.free,
            "derived_required_bytes": derived_required,
            "derivation": (
                "(horizon+4)*step4_checkpoint_bytes + 4*step4_archive_bytes"
            ),
            "cleanup": (
                "dense coordinate cameras retained only in-process; SHA-bound sparse "
                "pair inventories, stage checkpoints, and final archives preserved"
            ),
        },
        "torch_threads": observed_torch_threads,
        "smoke_horizon": config.smoke_horizon,
        "horizon_derivation": HORIZON_GAP,
        "validity_policy": VALIDITY_GAP,
        "range_gauge_policy": RANGE_GAUGE_POLICY,
        "elapsed_seconds": time.monotonic() - started,
        "admission": True,
        "execution_allowed_by_this_receipt": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
    }
    _write_json(_preflight_path(config), receipt)
    return receipt


def _load_preflight(config: DDMCountedApplicationConfigV1) -> dict[str, Any]:
    config.validate_all_bindings()
    path = _preflight_path(config)
    receipt = _read_json(path)
    if (
        receipt.get("schema") != PREFLIGHT_SCHEMA
        or receipt.get("typed_config_hash") != config.typed_hash()
        or receipt.get("admission") is not True
        or receipt.get("proposal_count") != 6
    ):
        raise DDMCountedApplicationError("J8f preflight is absent or stale")
    return {
        **receipt,
        "fresh_resume_admission": _fresh_resume_admission(
            config=config,
            preflight=receipt,
        ),
    }


def _checkpoint_path(output_root: Path, step_index: int) -> Path:
    return output_root / "checkpoints" / f"application_step_{step_index:02d}.json"


def _resume_receipts(
    *,
    output_root: Path,
    config: DDMCountedApplicationConfigV1,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    gap_seen = False
    for step_index in range(config.smoke_horizon):
        path = _checkpoint_path(output_root, step_index)
        if not path.exists():
            gap_seen = True
            continue
        if gap_seen:
            raise DDMCountedApplicationError("application checkpoints contain a gap")
        checkpoint = _read_json(path)
        if (
            checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("typed_config_hash") != config.typed_hash()
            or checkpoint.get("step_index") != step_index
            or checkpoint.get("status") != "PRESERVED"
            or not isinstance(checkpoint.get("application_receipts"), list)
            or len(checkpoint["application_receipts"]) != step_index + 1
            or checkpoint["application_receipts"][:-1] != receipts
        ):
            raise DDMCountedApplicationError(
                f"application checkpoint identity differs: {path}"
            )
        receipts = checkpoint["application_receipts"]
    return receipts


def _compiled_state_binding(
    *,
    theta: np.ndarray,
    realized: np.ndarray,
    archive: bytes,
) -> dict[str, Any]:
    if not np.array_equal(realized, theta):
        raise DDMCountedApplicationError(
            "one-quantum application clipped or changed during J5 realization"
        )
    if lift_v15_archive(archive).exact_reemit() != archive:
        raise DDMCountedApplicationError(
            "application stage archive fails exact receiver parse-back"
        )
    return {
        "theta_sha256_float32le": _sha256(
            np.asarray(theta, dtype="<f4").tobytes()
        ),
        "realized_theta_sha256_float32le": _sha256(
            np.asarray(realized, dtype="<f4").tobytes()
        ),
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "parseback_exact": True,
    }


def _checkpoint_application(
    *,
    output_root: Path,
    config: DDMCountedApplicationConfigV1,
    step_index: int,
    receipts: Sequence[Mapping[str, Any]],
    raw_theta: np.ndarray,
    projected_theta: np.ndarray,
    lift: Any,
) -> None:
    raw_archive, raw_realized = compile_parameterized_archive(
        lift, raw_theta, include_lane_programs=False
    )
    projected_archive, projected_realized = compile_parameterized_archive(
        lift, projected_theta, include_lane_programs=False
    )
    raw_state = _compiled_state_binding(
        theta=raw_theta,
        realized=raw_realized,
        archive=raw_archive,
    )
    projected_state = _compiled_state_binding(
        theta=projected_theta,
        realized=projected_realized,
        archive=projected_archive,
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "typed_config_hash": config.typed_hash(),
        "step_index": step_index,
        "status": "PRESERVED",
        "application_receipts": list(receipts),
        "raw_state": raw_state,
        "projected_state": projected_state,
        "resume": {
            "next_step_index": step_index + 1,
            "maximum_work_loss": "current descriptor application only",
            "all_prior_stage_checkpoints_preserved": True,
        },
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
    }
    _write_json(_checkpoint_path(output_root, step_index), checkpoint)


def _validate_resumed_application(
    *,
    output_root: Path,
    config: DDMCountedApplicationConfigV1,
    receipts: Sequence[Mapping[str, Any]],
    raw_theta: np.ndarray,
    projected_theta: np.ndarray,
    lift: Any,
) -> None:
    if not receipts:
        return
    step_index = len(receipts) - 1
    checkpoint = _read_json(_checkpoint_path(output_root, step_index))
    raw_archive, raw_realized = compile_parameterized_archive(
        lift, raw_theta, include_lane_programs=False
    )
    projected_archive, projected_realized = compile_parameterized_archive(
        lift, projected_theta, include_lane_programs=False
    )
    raw_state = _compiled_state_binding(
        theta=raw_theta,
        realized=raw_realized,
        archive=raw_archive,
    )
    projected_state = _compiled_state_binding(
        theta=projected_theta,
        realized=projected_realized,
        archive=projected_archive,
    )
    if (
        checkpoint.get("application_receipts") != list(receipts)
        or checkpoint.get("raw_state") != raw_state
        or checkpoint.get("projected_state") != projected_state
    ):
        raise DDMCountedApplicationError(
            "resumed application checkpoint does not re-derive byte-close"
        )


def _verdict(
    *,
    path: Path,
    archive: bytes,
    labels: np.ndarray,
    poses: np.ndarray,
    scorers: tuple[Any, Any],
    batch_size: int,
) -> dict[str, Any]:
    if path.exists():
        value = _read_json(path)
        if (
            value.get("archive_sha256") != _sha256(archive)
            or value.get("archive_bytes") != len(archive)
            or value.get("num_pairs") != 600
        ):
            raise DDMCountedApplicationError(f"resumed exact verdict differs: {path}")
        return value
    value = _chunked_n600_verdict(
        archive=archive,
        labels=labels,
        poses=poses,
        segnet=scorers[0],
        posenet=scorers[1],
        batch_size=batch_size,
    )
    _write_json(path, value)
    return value


def _load_completed_receipt(
    *,
    config: DDMCountedApplicationConfigV1,
    preflight: Mapping[str, Any],
) -> dict[str, Any] | None:
    output_root = Path(config.output_root)
    final_path = output_root / "ddm_j8f_counted_application_receipt.json"
    if not final_path.exists():
        return None
    receipt = _read_json(final_path)
    if (
        receipt.get("schema") != RUN_RECEIPT_SCHEMA
        or receipt.get("run_id") != config.run_id
        or receipt.get("lane_id") != config.lane_id
        or receipt.get("typed_config_hash") != config.typed_hash()
        or receipt.get("execution_allowed") is not False
        or receipt.get("main_landing_review_required") is not True
        or receipt.get("pointer") != POINTER
        or receipt.get("pointer_moved") is not False
        or receipt.get("score_claim") is not False
    ):
        raise DDMCountedApplicationError("completed J8f receipt identity differs")
    preflight_binding = receipt.get("preflight")
    preflight_path = _preflight_path(config)
    if (
        not isinstance(preflight_binding, Mapping)
        or preflight_binding.get("path") != str(preflight_path)
        or preflight_binding.get("sha256") != _sha256(preflight_path.read_bytes())
        or preflight_binding.get("admission") is not True
        or preflight.get("admission") is not True
    ):
        raise DDMCountedApplicationError(
            "completed J8f receipt preflight binding differs"
        )
    step4 = receipt.get("step4")
    if not isinstance(step4, Mapping) or not isinstance(
        step4.get("reference"), Mapping
    ):
        raise DDMCountedApplicationError(
            "completed J8f receipt Step-4 reference differs"
        )
    arms: dict[str, Mapping[str, Any]] = {}
    for arm_name in ("raw_arm", "range_gauge_projected_arm"):
        arm = receipt.get(arm_name)
        if not isinstance(arm, Mapping):
            raise DDMCountedApplicationError(
                f"completed J8f receipt lacks {arm_name}"
            )
        archive = arm.get("archive")
        verdict = arm.get("verdict")
        if not isinstance(archive, Mapping) or not isinstance(verdict, Mapping):
            raise DDMCountedApplicationError(
                f"completed J8f {arm_name} custody differs"
            )
        archive_path = Path(str(archive.get("path", "")))
        if (
            not archive_path.is_file()
            or archive_path.is_symlink()
            or archive.get("parseback_exact") is not True
        ):
            raise DDMCountedApplicationError(
                f"completed J8f {arm_name} archive is unavailable"
            )
        archive_bytes = archive_path.read_bytes()
        if (
            len(archive_bytes) != archive.get("bytes")
            or _sha256(archive_bytes) != archive.get("sha256")
            or verdict.get("archive_bytes") != len(archive_bytes)
            or verdict.get("archive_sha256") != _sha256(archive_bytes)
            or verdict.get("num_pairs") != 600
        ):
            raise DDMCountedApplicationError(
                f"completed J8f {arm_name} archive/verdict binding differs"
            )
        derived_delta = exact_joint_delta(
            reference=step4["reference"],
            candidate=verdict,
        )
        if arm.get("delta_vs_step4") != derived_delta:
            raise DDMCountedApplicationError(
                f"completed J8f {arm_name} joint delta differs"
            )
        arms[arm_name] = arm
    raw_delta = float(arms["raw_arm"]["delta_vs_step4"]["joint_delta"])
    projected_delta = float(
        arms["range_gauge_projected_arm"]["delta_vs_step4"]["joint_delta"]
    )
    projected_unchanged_or_better = projected_delta <= raw_delta
    ready = projected_delta < 0.0 and projected_unchanged_or_better
    if ready:
        expected_verdict = "READY_TO_FIRE_DDM_EVENT_CONTINUATION"
        control_path = output_root / "READY_TO_FIRE.ticket.json"
    elif projected_delta >= 0.0:
        expected_verdict = (
            "BLOCKED_DM4_J5_COUNTED_APPLICATION_REALIZED_JOINT_DELTA_NONNEGATIVE"
        )
        control_path = output_root / "BLOCKER.json"
    else:
        expected_verdict = (
            "BLOCKED_DM4_J5_RANGE_GAUGE_PROJECTION_REALIZED_JOINT_DELTA_WORSE"
        )
        control_path = output_root / "BLOCKER.json"
    if (
        receipt.get("verdict") != expected_verdict
        or arms["range_gauge_projected_arm"].get(
            "realized_joint_delta_unchanged_or_better_than_raw"
        )
        is not projected_unchanged_or_better
    ):
        raise DDMCountedApplicationError(
            "completed J8f READY/blocker decision differs"
        )
    control = _read_json(control_path)
    final_binding = {
        "path": str(final_path),
        "bytes": final_path.stat().st_size,
        "sha256": _sha256(final_path.read_bytes()),
    }
    if (
        control.get("status") != expected_verdict
        or control.get("receipt") != final_binding
        or control.get("execution_allowed") is not False
        or control.get("main_landing_review_required") is not True
        or control.get("pointer_moved") is not False
    ):
        raise DDMCountedApplicationError(
            "completed J8f READY/blocker control artifact differs"
        )
    return receipt


def _finalize(
    config: DDMCountedApplicationConfigV1,
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if not config.execution_allowed:
        raise DDMCountedApplicationError(
            "REFUSE_J8F_EXECUTION_DISABLED: reseal config only after three clean review passes"
        )
    observed_torch_threads = _configure_torch_threads(config.torch_threads)
    output_root = Path(config.output_root)
    completed = _load_completed_receipt(
        config=config,
        preflight=preflight,
    )
    if completed is not None:
        return completed
    started = time.monotonic()
    step4_typed, state, metadata, lift, step4_archive = _load_step4(config)
    j8e, proposals, adapter_receipt = _load_proposals(
        config, step4_archive=step4_archive
    )
    dm4_config, source_config, production_context = _dm4_plane_source(config)
    kernel = FullResizeKernel.build()
    ordered_horizon = tuple(proposals) * (config.smoke_horizon // len(proposals))
    if len(ordered_horizon) != config.smoke_horizon:
        raise DDMCountedApplicationError("smoke horizon is not an exact proposal cycle")
    receipts = _resume_receipts(output_root=output_root, config=config)
    used_raw = {
        int(row["raw_application"]["coordinate_index"]) for row in receipts
    }
    used_projected = {
        int(row["projected_application"]["coordinate_index"]) for row in receipts
    }
    base_theta = realize_parameter_theta(lift, state.theta)
    raw_theta = apply_coordinate_choices(
        base_theta, receipts, arm="raw_application"
    )
    projected_theta = apply_coordinate_choices(
        base_theta, receipts, arm="projected_application"
    )
    _validate_resumed_application(
        output_root=output_root,
        config=config,
        receipts=receipts,
        raw_theta=raw_theta,
        projected_theta=projected_theta,
        lift=lift,
    )
    inventory_cache: dict[
        int,
        tuple[
            tuple[SparseJ5CoordinateEffectV1, ...],
            dict[str, Any],
            dict[str, Any],
        ],
    ] = {}
    plane_cache: dict[int, tuple[np.ndarray, np.ndarray, dict[str, str]]] = {}
    metric_cache: dict[tuple[int, str], Any] = {}
    inventory_receipts: dict[str, Any] = {}
    descriptor_receipts: list[dict[str, Any]] = []
    for step_index in range(len(receipts), config.smoke_horizon):
        proposal = ordered_horizon[step_index]
        pair_id = int(proposal.aimed_cell["pair_id"])
        bucket_id = str(proposal.aimed_cell["bucket_id"])
        if pair_id not in inventory_cache:
            inventory_cache[pair_id] = _load_or_build_inventory(
                config=config,
                lift=lift,
                theta=base_theta,
                pair_id=pair_id,
                base_archive=step4_archive,
            )
            inventory_receipts[str(pair_id)] = {
                **inventory_cache[pair_id][1],
                "preserved_checkpoint": inventory_cache[pair_id][2],
            }
        if pair_id not in plane_cache:
            plane_cache[pair_id] = _load_pair_planes(
                pair_id=pair_id,
                source_config=source_config,
                context=production_context,
                kernel=kernel,
            )
        predictor, target, source_hashes = plane_cache[pair_id]
        descriptor_delta, descriptor_receipt = descriptor_camera_delta(
            proposal=proposal,
            predictor_planes=predictor,
            target_planes=target,
            kernel=kernel,
        )
        metric_key = (pair_id, bucket_id)
        if metric_key not in metric_cache:
            metric_cache[metric_key] = load_ms4d_pair_metric(
                path=_binding_path(config, "ms4d_direct_metric"),
                sha256=config.source_bindings["ms4d_direct_metric"].sha256,
                pair_id=pair_id,
                bucket_id=bucket_id,
            )
        application = select_counted_application(
            proposal=proposal,
            descriptor_delta=descriptor_delta,
            effects=inventory_cache[pair_id][0],
            metric=metric_cache[metric_key],
            used_raw_coordinates=used_raw,
            used_projected_coordinates=used_projected,
        )
        application["step_index"] = step_index
        application["descriptor_realization"] = descriptor_receipt
        application["dm4_source_chunk_hashes"] = source_hashes
        application["smoke_horizon"] = {
            "length": config.smoke_horizon,
            "derivation": HORIZON_GAP,
            "proposal_cycle_index": step_index // len(proposals),
        }
        receipts.append(application)
        descriptor_receipts.append(descriptor_receipt)
        used_raw.add(int(application["raw_application"]["coordinate_index"]))
        used_projected.add(
            int(application["projected_application"]["coordinate_index"])
        )
        raw_theta = apply_coordinate_choices(
            base_theta, receipts, arm="raw_application"
        )
        projected_theta = apply_coordinate_choices(
            base_theta, receipts, arm="projected_application"
        )
        _checkpoint_application(
            output_root=output_root,
            config=config,
            step_index=step_index,
            receipts=receipts,
            raw_theta=raw_theta,
            projected_theta=projected_theta,
            lift=lift,
        )
        print(
            json.dumps(
                {
                    "stage": "counted_application",
                    "step": step_index + 1,
                    "horizon": config.smoke_horizon,
                    "proposal": proposal.proposal_id,
                    "raw_coordinate": application["raw_application"][
                        "coordinate_name"
                    ],
                    "projected_coordinate": application[
                        "projected_application"
                    ]["coordinate_name"],
                    "rejected_null_gauge_energy_fraction": application[
                        "range_gauge_projection"
                    ]["rejected_null_gauge_energy_fraction"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del descriptor_delta
        gc.collect()

    descriptor_receipts = [
        dict(row["descriptor_realization"]) for row in receipts
    ]
    for proposal in proposals:
        pair_id = int(proposal.aimed_cell["pair_id"])
        bucket_id = str(proposal.aimed_cell["bucket_id"])
        metric_key = (pair_id, bucket_id)
        if metric_key not in metric_cache:
            metric_cache[metric_key] = load_ms4d_pair_metric(
                path=_binding_path(config, "ms4d_direct_metric"),
                sha256=config.source_bindings["ms4d_direct_metric"].sha256,
                pair_id=pair_id,
                bucket_id=bucket_id,
            )
        if str(pair_id) not in inventory_receipts:
            effects, inventory, checkpoint = _load_or_build_inventory(
                config=config,
                lift=lift,
                theta=base_theta,
                pair_id=pair_id,
                base_archive=step4_archive,
            )
            inventory_cache[pair_id] = (effects, inventory, checkpoint)
            inventory_receipts[str(pair_id)] = {
                **inventory,
                "preserved_checkpoint": checkpoint,
            }

    raw_archive, raw_realized = compile_parameterized_archive(
        lift, raw_theta, include_lane_programs=False
    )
    projected_archive, projected_realized = compile_parameterized_archive(
        lift, projected_theta, include_lane_programs=False
    )
    if (
        not np.array_equal(raw_realized, raw_theta)
        or not np.array_equal(projected_realized, projected_theta)
        or lift_v15_archive(raw_archive).exact_reemit() != raw_archive
        or lift_v15_archive(projected_archive).exact_reemit()
        != projected_archive
    ):
        raise DDMCountedApplicationError("final application archives fail exact parse-back")
    archives_dir = output_root / "archives"
    raw_path = archives_dir / "step4_plus_dm4_raw.zip.receipt-bytes"
    projected_path = (
        archives_dir / "step4_plus_dm4_range_gauge_projected.zip.receipt-bytes"
    )
    _atomic_write(raw_path, raw_archive)
    _atomic_write(projected_path, projected_archive)

    cache_path = Path(step4_typed.target_cache_path)
    labels = open_stored_npy_memmap(cache_path, "lstars")
    poses = open_stored_npy_memmap(cache_path, "gt_poses")
    scorers = _load_cpu_frozen_scorers(step4_typed.upstream_root)
    raw_verdict = _verdict(
        path=output_root / "verdicts" / "raw_n600.json",
        archive=raw_archive,
        labels=labels,
        poses=poses,
        scorers=scorers,
        batch_size=step4_typed.verdict_batch,
    )
    print(
        json.dumps(
            {
                "stage": "exact_n600",
                "arm": "raw",
                "d_seg": raw_verdict["d_seg"],
                "d_pose": raw_verdict["d_pose"],
                "archive_bytes": raw_verdict["archive_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    projected_verdict = _verdict(
        path=output_root / "verdicts" / "range_gauge_projected_n600.json",
        archive=projected_archive,
        labels=labels,
        poses=poses,
        scorers=scorers,
        batch_size=step4_typed.verdict_batch,
    )
    print(
        json.dumps(
            {
                "stage": "exact_n600",
                "arm": "range_gauge_projected",
                "d_seg": projected_verdict["d_seg"],
                "d_pose": projected_verdict["d_pose"],
                "archive_bytes": projected_verdict["archive_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    reference = _read_json(_binding_path(config, "step4_verdict"))
    raw_delta = exact_joint_delta(reference=reference, candidate=raw_verdict)
    projected_delta = exact_joint_delta(
        reference=reference, candidate=projected_verdict
    )
    projection_unchanged_or_better = (
        float(projected_delta["joint_delta"])
        <= float(raw_delta["joint_delta"])
    )
    ready = (
        float(projected_delta["joint_delta"]) < 0.0
        and projection_unchanged_or_better
    )
    if ready:
        verdict = "READY_TO_FIRE_DDM_EVENT_CONTINUATION"
        blocker = None
    elif float(projected_delta["joint_delta"]) >= 0.0:
        verdict = (
            "BLOCKED_DM4_J5_COUNTED_APPLICATION_REALIZED_JOINT_DELTA_NONNEGATIVE"
        )
        blocker = {
            "scope": (
                "INSTANCE x six DM4 descriptors x Step-4 J5 12-step one-quantum "
                "Newton/range-gauge application"
            ),
            "classification": verdict,
        }
    else:
        verdict = (
            "BLOCKED_DM4_J5_RANGE_GAUGE_PROJECTION_REALIZED_JOINT_DELTA_WORSE"
        )
        blocker = {
            "scope": (
                "INSTANCE x #580 nearest-integer J5 reprojection at this Step-4 "
                "application; not a #580 or J5 family negative"
            ),
            "classification": verdict,
        }
    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "lane_id": config.lane_id,
        "typed_config_hash": config.typed_hash(),
        "authority": config.source_bindings["authority"].to_payload(),
        "preflight": {
            "path": str(_preflight_path(config)),
            "sha256": _sha256(_preflight_path(config).read_bytes()),
            "admission": preflight["admission"],
            "fresh_resume_admission": preflight.get(
                "fresh_resume_admission"
            ),
        },
        "step4": {
            "checkpoint": config.source_bindings["step4_checkpoint"].to_payload(),
            "checkpoint_step": state.step,
            "checkpoint_metadata_sha256": _sha256(_canonical_bytes(metadata)),
            "reference_verdict": config.source_bindings[
                "step4_verdict"
            ].to_payload(),
            "reference": reference,
            "archive_bytes": len(step4_archive),
            "archive_sha256": _sha256(step4_archive),
        },
        "j8e": {
            "ticket": config.source_bindings["j8e_ticket"].to_payload(),
            "dsl_compile_hash": j8e.dsl_compile_hash,
            "proposal_adapter": adapter_receipt,
            "proposal_count": len(proposals),
        },
        "dm4": {
            "config": config.source_bindings["dm4_config"].to_payload(),
            "receipt": config.source_bindings["dm4_receipt"].to_payload(),
            "descriptor_receipts": descriptor_receipts,
            "source_config_schema": dm4_config["schema"],
        },
        "ms4d": {
            "direct_metric": config.source_bindings[
                "ms4d_direct_metric"
            ].to_payload(),
            "pair_bucket_metrics": [
                metric_cache[key].to_payload() for key in sorted(metric_cache)
            ],
        },
        "application": {
            "schema": "ddm_dm4_j5_counted_application_horizon.v1",
            "horizon": config.smoke_horizon,
            "horizon_derivation": HORIZON_GAP,
            "validity_policy": VALIDITY_GAP,
            "range_gauge_policy": RANGE_GAUGE_POLICY,
            "coordinate_reuse_allowed": False,
            "raw_unique_coordinate_count": len(used_raw),
            "projected_unique_coordinate_count": len(used_projected),
            "pair_inventories": inventory_receipts,
            "stage_receipts": receipts,
            "all_stage_checkpoints_preserved": True,
        },
        "raw_arm": {
            "archive": {
                "path": str(raw_path),
                "bytes": len(raw_archive),
                "sha256": _sha256(raw_archive),
                "parseback_exact": True,
            },
            "verdict": raw_verdict,
            "delta_vs_step4": raw_delta,
        },
        "range_gauge_projected_arm": {
            "archive": {
                "path": str(projected_path),
                "bytes": len(projected_archive),
                "sha256": _sha256(projected_archive),
                "parseback_exact": True,
            },
            "verdict": projected_verdict,
            "delta_vs_step4": projected_delta,
            "realized_joint_delta_unchanged_or_better_than_raw": (
                projection_unchanged_or_better
            ),
            "delta_vs_raw_joint": (
                float(projected_delta["joint_delta"])
                - float(raw_delta["joint_delta"])
            ),
        },
        "verdict": verdict,
        "blocker": blocker,
        "execution_allowed": False,
        "fire_authority": "MAIN_ONLY_AFTER_REVIEW",
        "main_landing_review_required": True,
        "torch_threads": observed_torch_threads,
        "deterministic_algorithms": True,
        "elapsed_seconds": time.monotonic() - started,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
    }
    final_path = output_root / "ddm_j8f_counted_application_receipt.json"
    _write_json(final_path, receipt)
    final_binding = {
        "path": str(final_path),
        "bytes": final_path.stat().st_size,
        "sha256": _sha256(final_path.read_bytes()),
    }
    if ready:
        ticket = {
            "schema": READY_TICKET_SCHEMA,
            "status": verdict,
            "receipt": final_binding,
            "candidate_archive": receipt["range_gauge_projected_arm"][
                "archive"
            ],
            "exact_delta_vs_step4": projected_delta,
            "execution_allowed": False,
            "fire_authority": "MAIN_ONLY_AFTER_REVIEW",
            "main_landing_review_required": True,
            "pointer": POINTER,
            "pointer_moved": False,
            "score_claim": False,
        }
        _write_json(output_root / "READY_TO_FIRE.ticket.json", ticket)
    else:
        blocker_receipt = {
            "schema": BLOCKER_SCHEMA,
            "status": verdict,
            "receipt": final_binding,
            "sha_bound_decomposition": {
                "step4": receipt["step4"],
                "raw_arm": receipt["raw_arm"],
                "range_gauge_projected_arm": receipt[
                    "range_gauge_projected_arm"
                ],
            },
            "verdict_scope": blocker["scope"] if blocker else None,
            "execution_allowed": False,
            "main_landing_review_required": True,
            "pointer": POINTER,
            "pointer_moved": False,
            "score_claim": False,
        }
        _write_json(output_root / "BLOCKER.json", blocker_receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=("validate", "preflight", "run"),
        required=True,
    )
    args = parser.parse_args(argv)
    config = DDMCountedApplicationConfigV1.from_path(args.config)
    if args.mode == "validate":
        print(
            json.dumps(
                {
                    "typed_config_hash": config.typed_hash(),
                    "bindings": config.validate_all_bindings(),
                    "execution_allowed": config.execution_allowed,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "preflight":
        receipt = _preflight(config)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    preflight = _load_preflight(config)
    receipt = _finalize(config, preflight=preflight)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "raw_joint_delta": receipt["raw_arm"]["delta_vs_step4"][
                    "joint_delta"
                ],
                "projected_joint_delta": receipt[
                    "range_gauge_projected_arm"
                ]["delta_vs_step4"]["joint_delta"],
                "main_landing_review_required": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DDMCountedApplicationError, DirectDescriptionError) as exc:
        raise SystemExit(str(exc)) from exc
