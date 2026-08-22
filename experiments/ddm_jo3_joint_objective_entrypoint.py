#!/usr/bin/env python3
"""Local, resumable JO3 trainer and real-configuration memory preflight.

This is the executable bridge between the sealed JO1 objective and the JO2
receiver-close implementation.  Training differentiates a float16-exact
``HybridOutputResidual`` through the public camera round-trip, frozen SegNet,
and frozen PoseNet.  Every stage boundary materializes all 600 camera pairs,
freshly re-solves the frame-0 carrier, races the real coders, executes the
shipped receiver, retains the complete scorer surfaces, and applies the exact
B/H/Pose/rate admission test.  It never launches itself and never claims a
contest score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.func import functional_call
from torch.nn import functional

from experiments import ddm_jo1_joint_objective_design as design
from experiments import ddm_jo1_joint_objective_worker as worker
from experiments import ddm_jo2_receiver_close as receiver_close
from experiments import ddm_jo2_residual_runtime as residual_runtime

REPO: Final = Path(__file__).resolve().parents[1]
UPSTREAM: Final = REPO / "upstream"
AXIS: Final = "[macOS-CPU advisory; frozen CPU scorers; never a contest score]"
PREFLIGHT_AXIS: Final = "[macOS-CPU real-config preflight; no score authority]"
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
TRAIN_BATCH_PAIRS: Final = 1
FIELD_BATCH_PAIRS: Final = 16
CHUNK_PAIR_LIMIT: Final = 120
EMA_DECAY: Final = 0.995
RAW_BYTES: Final = design.N_PAIRS * 2 * CAMERA_H * CAMERA_W * 3


class JO3EntrypointError(RuntimeError):
    """A governed launch, source, resume, retention, or mechanism check failed."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise JO3EntrypointError(f"retained file is absent: {resolved}")
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}


def atomic_bytes(path: Path, payload: bytes, *, replace_metadata: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() == payload:
            return file_record(path)
        if not replace_metadata:
            raise JO3EntrypointError(f"refusing to overwrite retained payload: {path.resolve()}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_record(path)


def atomic_json(path: Path, value: Any, *, replace_metadata: bool = False) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload, replace_metadata=replace_metadata)


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return atomic_bytes(path, buffer.getvalue())


def atomic_compact_json(path: Path, value: Any) -> dict[str, Any]:
    """Atomically retain compact machine-readable rows without scalar-only loss."""
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload)


def raw_array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return {
        "bytes": int(array.nbytes),
        "sha256": digest,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "order": "C",
    }


class CertifiedCandidateRetention:
    """Retain winner bytes and exact rebuild certificates for explored non-winners.

    The receiver still materializes every real camera candidate. Before those
    non-winner buffers are released, this layer atomically records their exact
    raw-byte hashes plus the complete deterministic regeneration tuple. The
    selected pair winner is regenerated once, checked against its exploration
    certificate, and retained in full.
    """

    def __init__(
        self,
        *,
        solve_root: Path,
        stage_id: str,
        workload_config_sha256: str,
        base_archive_sha256: str,
    ) -> None:
        self.solve_root = solve_root.resolve()
        self.stage_id = stage_id
        self.workload_config_sha256 = workload_config_sha256
        self.base_archive_sha256 = base_archive_sha256
        self.entrypoint_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for name, value in (
            ("entrypoint", self.entrypoint_sha256),
            ("workload", workload_config_sha256),
            ("base archive", base_archive_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise JO3EntrypointError(f"certified retention {name} SHA-256 differs")
        if not stage_id:
            raise JO3EntrypointError("certified retention stage id is absent")
        self._certificates: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}

    def _context(self, root: Path) -> dict[str, Any]:
        resolved = root.resolve()
        if self.solve_root != resolved and self.solve_root not in resolved.parents:
            raise JO3EntrypointError("certified retention root escaped the fresh solve")
        return {
            "entrypoint_sha256": self.entrypoint_sha256,
            "workload_identity_sha256": self.workload_config_sha256,
            "base_archive_sha256": self.base_archive_sha256,
            "stage_id": self.stage_id,
            "solve_phase": str(resolved.relative_to(self.solve_root)),
        }

    @staticmethod
    def _key(pair: int, coordinate_delta: Sequence[int]) -> tuple[int, tuple[int, ...]]:
        return int(pair), tuple(int(value) for value in coordinate_delta)

    def _remember(self, row: Mapping[str, Any]) -> None:
        key = self._key(int(row["pair_id"]), row["candidate_coordinate_delta"])
        payloads = {
            "slave_camera_payload": dict(row["slave_camera_payload"]),
            "pose_input_camera_payload": dict(row["pose_input_camera_payload"]),
        }
        prior = self._certificates.get(key)
        if prior is not None and prior != payloads:
            raise JO3EntrypointError("deterministic rebuild certificate drifted for one coordinate")
        self._certificates[key] = payloads

    def retain_explored(
        self,
        *,
        root: Path,
        pair: int,
        base_codes: np.ndarray,
        codes: np.ndarray,
        slave_camera: np.ndarray,
        pose_input: np.ndarray,
        pose_vectors: np.ndarray,
    ) -> dict[str, Any]:
        count = len(codes)
        if not (
            np.asarray(codes).shape == (count, receiver_close.D)
            and np.asarray(base_codes).shape == (receiver_close.D,)
            and len(slave_camera) == count
            and len(pose_input) == count
            and np.asarray(pose_vectors).shape == (count, receiver_close.POSE_DIMS)
        ):
            raise JO3EntrypointError("certified retention candidate geometry differs")
        rows = []
        for index in range(count):
            delta = np.asarray(codes[index], dtype=np.int32) - np.asarray(base_codes, dtype=np.int32)
            row = {
                "pair_id": int(pair),
                "candidate_index_in_batch": int(index),
                "candidate_coordinate_delta": delta.tolist(),
                "slave_camera_payload": raw_array_record(slave_camera[index]),
                "pose_input_camera_payload": raw_array_record(pose_input[index]),
            }
            self._remember(row)
            rows.append(row)
        manifest = {
            "schema": "ddm_jo4_certified_rebuild_batch.v1",
            "regeneration_context": self._context(root),
            "tuple_join": (
                "regeneration_context + rows[pair_id,candidate_coordinate_delta]"
            ),
            "candidate_denominator": count,
            "rows": rows,
            "all_camera_hashes_computed_while_materialized": True,
            "nonwinner_buffer_release_allowed_only_after_this_manifest": True,
        }
        try:
            certificate = atomic_compact_json(root / "CERTIFIED_REBUILD.json", manifest)
            codes_record = atomic_npy(root / "codes.int32.npy", np.asarray(codes, dtype=np.int32))
            vectors_record = atomic_npy(
                root / "pose_vectors.float32.npy", np.asarray(pose_vectors, dtype=np.float32)
            )
        except OSError as error:
            raise JO3EntrypointError(
                f"refusing candidate because its rebuild certificate could not be retained: {error}"
            ) from error
        return {
            "retention_mode": "CERTIFIED_REBUILDABLE_NONWINNER",
            "certified_rebuild_manifest": certificate,
            "codes": codes_record,
            "pose_vectors": vectors_record,
        }

    def verify_explored_result(self, result: Mapping[str, Any]) -> None:
        if result.get("retention_mode") != "CERTIFIED_REBUILDABLE_NONWINNER":
            raise JO3EntrypointError("resumed explored candidate lacks certified retention")
        for name in ("certified_rebuild_manifest", "codes", "pose_vectors"):
            receiver_close.verify_record(result[name])
        manifest_path = receiver_close.verify_record(result["certified_rebuild_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "ddm_jo4_certified_rebuild_batch.v1"
            or manifest.get("candidate_denominator") != len(manifest.get("rows", []))
        ):
            raise JO3EntrypointError("certified rebuild manifest census differs")
        context = manifest.get("regeneration_context", {})
        expected = self._context(manifest_path.parent)
        if context != expected:
            raise JO3EntrypointError("certified rebuild regeneration tuple drifted")
        for row in manifest["rows"]:
            self._remember(row)

    def retain_winner(
        self,
        *,
        root: Path,
        pair: int,
        base_codes: np.ndarray,
        codes: np.ndarray,
        slave_camera: np.ndarray,
        pose_input: np.ndarray,
        pose_vector: np.ndarray,
    ) -> dict[str, Any]:
        delta = np.asarray(codes, dtype=np.int32) - np.asarray(base_codes, dtype=np.int32)
        key = self._key(pair, delta.tolist())
        exploration = self._certificates.get(key)
        if exploration is None:
            raise JO3EntrypointError("selected winner has no exploration rebuild certificate")
        repeated = {
            "slave_camera_payload": raw_array_record(slave_camera),
            "pose_input_camera_payload": raw_array_record(pose_input),
        }
        if repeated != exploration:
            raise JO3EntrypointError("selected winner camera repeat differs from exploration hash")
        payloads = {
            "codes": atomic_npy(root / "codes.int32.npy", np.asarray(codes, dtype=np.int32)),
            "slave_camera": atomic_npy(
                root / "slave_camera.uint8.npy", np.asarray(slave_camera, dtype=np.uint8)
            ),
            "pose_input": atomic_npy(
                root / "pose_input.uint8.npy", np.asarray(pose_input, dtype=np.uint8)
            ),
            "pose_vector": atomic_npy(
                root / "pose_vector.float32.npy", np.asarray(pose_vector, dtype=np.float32)
            ),
        }
        receipt = atomic_json(
            root / "WINNER_RETENTION.json",
            {
                "schema": "ddm_jo4_full_winner_retention.v1",
                "regeneration_context": self._context(root),
                "pair_id": int(pair),
                "candidate_coordinate_delta": delta.tolist(),
                "exploration_camera_certificates": exploration,
                "winner_repeat_camera_certificates": repeated,
                "payloads": payloads,
                "deterministic_repeat_byte_identical": True,
                "retention_mode": "FULL_BYTES_STAGE_WINNER",
            },
        )
        return {"schema": "ddm_jo4_winner_retention_pointer.v1", "receipt": receipt}

    def verify_winner(self, pointer: Any) -> None:
        if not isinstance(pointer, Mapping) or pointer.get("schema") != (
            "ddm_jo4_winner_retention_pointer.v1"
        ):
            raise JO3EntrypointError("resumed pair has no full winner retention pointer")
        path = receiver_close.verify_record(pointer["receipt"])
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "ddm_jo4_full_winner_retention.v1"
            or receipt.get("deterministic_repeat_byte_identical") is not True
            or receipt.get("retention_mode") != "FULL_BYTES_STAGE_WINNER"
        ):
            raise JO3EntrypointError("resumed winner retention receipt differs")
        if receipt.get("regeneration_context") != self._context(path.parent):
            raise JO3EntrypointError("resumed winner regeneration tuple drifted")
        for record in receipt.get("payloads", {}).values():
            receiver_close.verify_record(record)

    def finalize(self) -> dict[str, Any]:
        certificates = sorted(self.solve_root.rglob("CERTIFIED_REBUILD.json"))
        winners = sorted(self.solve_root.rglob("WINNER_RETENTION.json"))
        if len(winners) != design.N_PAIRS or not certificates:
            raise JO3EntrypointError(
                "fresh solve retention inventory is incomplete: "
                f"winners={len(winners)},certificates={len(certificates)}"
            )
        inventory = atomic_json(
            self.solve_root / "RETENTION_INVENTORY.json",
            {
                "schema": "ddm_jo4_two_tier_retention_inventory.v1",
                "stage_id": self.stage_id,
                "entrypoint_sha256": self.entrypoint_sha256,
                "workload_identity_sha256": self.workload_config_sha256,
                "base_archive_sha256": self.base_archive_sha256,
                "winner_pair_denominator": len(winners),
                "certified_rebuild_manifest_denominator": len(certificates),
                "winner_receipts": [file_record(path) for path in winners],
                "certified_rebuild_manifests": [file_record(path) for path in certificates],
                "full_bytes_for_every_stage_pair_winner": True,
                "nonwinners_have_fail_closed_rebuild_certificates": True,
            },
        )
        return {"schema": "ddm_jo4_retention_inventory_pointer.v1", "receipt": inventory}

    def verify_inventory(self, pointer: Any) -> None:
        if not isinstance(pointer, Mapping) or pointer.get("schema") != (
            "ddm_jo4_retention_inventory_pointer.v1"
        ):
            raise JO3EntrypointError("fresh solve has no two-tier retention inventory")
        path = receiver_close.verify_record(pointer["receipt"])
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema") != "ddm_jo4_two_tier_retention_inventory.v1"
            or value.get("winner_pair_denominator") != design.N_PAIRS
            or value.get("full_bytes_for_every_stage_pair_winner") is not True
            or value.get("nonwinners_have_fail_closed_rebuild_certificates") is not True
            or value.get("stage_id") != self.stage_id
            or value.get("entrypoint_sha256") != self.entrypoint_sha256
            or value.get("workload_identity_sha256") != self.workload_config_sha256
            or value.get("base_archive_sha256") != self.base_archive_sha256
        ):
            raise JO3EntrypointError("two-tier retention inventory census differs")
        for name in ("winner_receipts", "certified_rebuild_manifests"):
            for record in value.get(name, []):
                receiver_close.verify_record(record)


def object_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def require_governed_admission(main_authorized: bool) -> None:
    if not main_authorized or os.environ.get("TAC_GOVERNED_ADMISSION") != "1":
        raise JO3EntrypointError("MAIN authorization and TAC_GOVERNED_ADMISSION=1 are both required")


def verify_storage(config: design.CompiledConfig) -> dict[str, Any]:
    root = Path(config.output_root).resolve()
    probe = root if root.exists() else root.parent
    usage = shutil.disk_usage(probe)
    required = config.memory_preflight.minimum_ap_free_bytes
    if usage.free < required:
        raise JO3EntrypointError(f"storage preflight failed: free={usage.free}, required={required}")
    return {"path": str(probe), "free_bytes": usage.free, "required_free_bytes": required}


def write_storage_policy(run_root: Path, config: design.CompiledConfig) -> dict[str, Any]:
    """Install the automatic certify-or-block policy before large materialization."""
    memory = design.verify_memory_receipt(config)
    projected_bytes = int(
        memory["receiver_scale_preflight"]["all_stage_plus_reserve_projected_bytes"]
    )
    probe = run_root if run_root.exists() else run_root.parent
    free_bytes = shutil.disk_usage(probe).free
    if free_bytes < projected_bytes:
        raise JO3EntrypointError(
            "storage changed after scale preflight: "
            f"free={free_bytes},projected_minimum={projected_bytes}"
        )
    return atomic_json(
        run_root / "STORAGE_POLICY.json",
        {
            "schema": "ddm_jo3_storage_policy.v1",
            "run_root": str(run_root.resolve()),
            "workload_config_sha256": config.workload_config_sha256,
            "projected_bytes": projected_bytes,
            "observed_free_bytes": free_bytes,
            "minimum_free_bytes": projected_bytes,
            "legacy_minimum_ap_free_bytes": config.memory_preflight.minimum_ap_free_bytes,
            "bulk_tier_exception": (
                "charter explicitly re-rooted this solve to local APFS after measuring 603 GB free"
            ),
            "automatic_cleanup_action": (
                "FULL_BYTES_FOR_STAGE_WINNERS_CERTIFIED_REBUILDABLE_FOR_EXPLORED_NONWINNERS"
            ),
            "cleanup_blocker": (
                "winner, training, scorer, coder, and checkpoint bytes remain full evidence; "
                "nonwinner camera buffers may be released only after their atomic rebuild certificate"
            ),
            "payload_receipt_rule": (
                "every winner payload carries path, bytes, and sha256; every explored nonwinner "
                "carries camera bytes, sha256, and the exact regeneration tuple"
            ),
        },
    )


def load_config(path: Path, expected_sha256: str) -> design.CompiledConfig:
    config = design.load_compiled_config(path.resolve(), expected_sha256)
    for record in (
        config.inputs.rc2_decoded_semantic_tokens,
        config.inputs.gt_argmax_field,
        config.inputs.rc2_base_argmax_field,
        config.inputs.fx5_base_pose6,
        config.inputs.source_pose6_targets,
        config.inputs.receiver_close_source,
        config.inputs.residual_runtime_source,
    ):
        if record is None:
            raise JO3EntrypointError("a required JO3 input binding is absent")
        design.verify_artifact(record)
    for record in (
        config.inputs.rc2_archive,
        config.inputs.rc2_runtime,
        config.inputs.source_object,
        config.inputs.segnet_weights,
        config.inputs.posenet_weights,
        config.inputs.compiler_source,
        config.inputs.worker_source,
        config.inputs.dispatcher_source,
    ):
        design.verify_artifact(record)
    if (
        config.dispatch.platform != "local"
        or config.dispatch.gpu != "CPU"
        or Path(config.inputs.dispatcher_source.path).resolve() != Path(__file__).resolve()
    ):
        raise JO3EntrypointError("compiled dispatch is not bound to this local CPU entrypoint")
    verify_storage(config)
    return config


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def load_scorers(config: design.CompiledConfig) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from tac.differentiable_eval_roundtrip import (
        assert_yuv6_forward_equivalence_to_upstream,
        patch_upstream_yuv6_globally,
    )

    equivalence = assert_yuv6_forward_equivalence_to_upstream(atol=1e-6)
    patch = patch_upstream_yuv6_globally()
    modules = importlib.import_module("modules")
    segnet = modules.SegNet().eval().cpu()
    posenet = modules.PoseNet().eval().cpu()
    segnet.load_state_dict(load_file(config.inputs.segnet_weights.path, device="cpu"))
    posenet.load_state_dict(load_file(config.inputs.posenet_weights.path, device="cpu"))
    for network in (segnet, posenet):
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    return (
        segnet,
        posenet,
        {
            "forward_equivalence": equivalence,
            "frame_utils_was_patched": patch.frame_utils_was_patched,
            "modules_was_patched": patch.modules_was_patched,
            "gradient_route": "tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6",
        },
    )


def load_semantic(config: design.CompiledConfig) -> tuple[nn.Module, Any, Any]:
    surface, modules = receiver_close.load_surface(
        Path(config.inputs.rc2_archive.path), Path(config.inputs.rc2_runtime.path)
    )
    semantic = modules.renderer.SemanticTokenRenderer(96)
    state = modules.renderer.unpack_variant_semantic_or_none(surface.parts.semantic_blob, semantic.state_dict())
    if state is None:
        raise JO3EntrypointError("fx5 semantic payload is not the exact tagged variant")
    semantic.load_state_dict(state, strict=True)
    semantic.eval().cpu()
    for parameter in semantic.parameters():
        parameter.requires_grad_(False)
    return semantic, surface, modules


def open_inputs(config: design.CompiledConfig) -> dict[str, np.ndarray]:
    token_ref = config.inputs.rc2_decoded_semantic_tokens
    assert token_ref is not None
    tokens = np.memmap(
        token_ref.path,
        mode="r",
        dtype=np.uint8,
        shape=(design.N_PAIRS, design.SEG_H, design.SEG_W),
    )
    result: dict[str, np.ndarray] = {"tokens": tokens}
    for name, record in (
        ("target", config.inputs.gt_argmax_field),
        ("base_argmax", config.inputs.rc2_base_argmax_field),
        ("base_pose6", config.inputs.fx5_base_pose6),
        ("pose_target", config.inputs.source_pose6_targets),
    ):
        assert record is not None
        result[name] = np.load(record.path, mmap_mode="r", allow_pickle=False)
    return result


def quantized_state(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Float16 receiver values in the forward pass, identity STE in backward."""
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state.items():
        quantized = value.to(torch.float16).to(torch.float32)
        result[name] = value + (quantized - value).detach()
    return result


def quantized_residual(model: worker.HybridOutputResidual, tokens: torch.Tensor) -> torch.Tensor:
    return functional_call(model, quantized_state(OrderedDict(model.named_parameters())), (tokens,))


def ste_round(value: torch.Tensor) -> torch.Tensor:
    return value + (value.round() - value).detach()


def render_training_pair(
    *,
    semantic: nn.Module,
    model: worker.HybridOutputResidual,
    tokens: torch.Tensor,
    pair: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.tensor([pair], dtype=torch.long)
    base = semantic(tokens, index)
    residual = quantized_residual(model, tokens)
    pre_r = worker.apply_output_residual(base, residual)
    camera = functional.interpolate(pre_r, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False).clamp(
        0.0, 255.0
    )
    camera = ste_round(camera)
    scorer = functional.interpolate(camera, size=(design.SEG_H, design.SEG_W), mode="bilinear", align_corners=False)
    return pre_r, camera, scorer


def rate_proxy(model: nn.Module) -> torch.Tensor:
    values = [
        torch.log1p(value.abs()).mean() for value in quantized_state(OrderedDict(model.named_parameters())).values()
    ]
    return torch.stack(values).mean()


def retain_training_step(
    root: Path,
    *,
    pair: int,
    tokens: torch.Tensor,
    pre_r: torch.Tensor,
    camera: torch.Tensor,
    seg_input: torch.Tensor,
    seg_logits: torch.Tensor,
    pose_input: torch.Tensor,
    pose6: torch.Tensor,
    target: torch.Tensor,
    base_argmax: torch.Tensor,
    metrics: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "tokens": atomic_npy(root / "tokens.uint8.npy", tokens.detach().cpu().numpy().astype(np.uint8)),
        "pre_r": atomic_npy(root / "pre_r.float32.npy", pre_r.detach().cpu().numpy()),
        "camera": atomic_npy(root / "camera.float32.npy", camera.detach().cpu().numpy()),
        "seg_input": atomic_npy(root / "seg_input.float32.npy", seg_input.detach().cpu().numpy()),
        "seg_logits": atomic_npy(root / "seg_logits.float32.npy", seg_logits.detach().cpu().numpy()),
        "pose_input": atomic_npy(root / "pose_input.float32.npy", pose_input.detach().cpu().numpy()),
        "pose6": atomic_npy(root / "pose6.float32.npy", pose6.detach().cpu().numpy()),
        "target": atomic_npy(root / "target.uint8.npy", target.detach().cpu().numpy().astype(np.uint8)),
        "base_argmax": atomic_npy(root / "base_argmax.uint8.npy", base_argmax.detach().cpu().numpy().astype(np.uint8)),
    }
    result = {
        "schema": "ddm_jo3_retained_training_step.v1",
        "pair": pair,
        "payloads": payloads,
        "metrics": {name: float(value.cpu()) for name, value in metrics.items()},
        "all_materialized_payloads_retained": True,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(root / "RESULT.json", result)
    return result


def one_training_step(
    *,
    config: design.CompiledConfig,
    stage: design.StageConfig,
    pair: int,
    semantic: nn.Module,
    model: worker.HybridOutputResidual,
    segnet: nn.Module,
    posenet: nn.Module,
    arrays: Mapping[str, np.ndarray],
    frame0: np.ndarray,
    duals: worker.DualState,
    retain_root: Path,
    optimizer: torch.optim.Optimizer | None,
    require_nonzero_gradient: bool = False,
) -> dict[str, Any]:
    token = torch.from_numpy(np.asarray(arrays["tokens"][pair]).copy())[None].long()
    target = torch.from_numpy(np.asarray(arrays["target"][pair]).copy())[None].long()
    base_argmax = torch.from_numpy(np.asarray(arrays["base_argmax"][pair]).copy())[None].long()
    pose_target = torch.from_numpy(np.asarray(arrays["pose_target"][pair]).copy())[None].float()
    first = torch.from_numpy(np.asarray(frame0[pair]).copy()).permute(2, 0, 1)[None].float()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    pre_r, camera, seg_input = render_training_pair(semantic=semantic, model=model, tokens=token, pair=pair)
    seg_logits = segnet(seg_input)
    pair_camera = torch.stack((first[0], camera[0]), dim=0)[None]
    pose_input = posenet.preprocess_input(pair_camera)
    pose6 = posenet(pose_input)["pose"][..., :6]
    loss, metrics = worker.joint_inner_objective(
        seg_logits=seg_logits,
        target=target,
        retained_base_argmax=base_argmax,
        pose6_candidate=pose6,
        pose6_target=pose_target,
        rate_proxy=rate_proxy(model),
        duals=duals,
        stage=stage,
    )
    loss.backward()
    gradient_norm = float(
        torch.sqrt(sum(value.grad.detach().square().sum() for value in model.parameters() if value.grad is not None))
    )
    if not np.isfinite(gradient_norm):
        raise JO3EntrypointError("real scorer objective produced a non-finite residual gradient")
    if require_nonzero_gradient and gradient_norm <= 0.0:
        raise JO3EntrypointError("real-config preflight produced no residual gradient")
    if optimizer is not None:
        optimizer.step()
    metrics = dict(metrics)
    metrics["gradient_norm"] = torch.tensor(gradient_norm)
    return retain_training_step(
        retain_root,
        pair=pair,
        tokens=token,
        pre_r=pre_r,
        camera=camera,
        seg_input=seg_input,
        seg_logits=seg_logits,
        pose_input=pose_input,
        pose6=pose6,
        target=target,
        base_argmax=base_argmax,
        metrics=metrics,
    )


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def receiver_scale_preflight(
    *, surface: Any, free_bytes: int, stage_denominator: int
) -> dict[str, Any]:
    """Price endpoint-safe probes and the governed two-tier retention projection."""
    codes = np.asarray(surface.codes, dtype=np.int32)
    endpoint = np.argwhere((codes == -2048) | (codes == 2047))
    endpoint_pairs = sorted({int(row) for row in endpoint[:, 0]}) if endpoint.size else []
    derivative_modes = {
        "central_second_order": 0,
        "forward_one_sided_first_order": 0,
        "backward_one_sided_first_order": 0,
    }
    endpoint_probes = []
    blocked_coordinates = []
    for pair in range(codes.shape[0]):
        for dimension in range(codes.shape[1]):
            coordinate = int(codes[pair, dimension])
            try:
                offsets, denominator, mode = receiver_close.jacobian_probe_offsets(coordinate)
                probes = [coordinate + int(offset) for offset in offsets]
                if not all(-2048 <= value <= 2047 for value in probes):
                    raise JO3EntrypointError("preflight derivative probe left int12 domain")
            except (receiver_close.JO2ReceiverCloseError, JO3EntrypointError) as error:
                blocked_coordinates.append(
                    {"pair_id": pair, "dimension": dimension, "coordinate": coordinate, "error": str(error)}
                )
                continue
            derivative_modes[mode] += 1
            if mode != "central_second_order":
                endpoint_probes.append(
                    {
                        "pair_id": pair,
                        "dimension": dimension,
                        "coordinate": coordinate,
                        "probe_coordinates": probes,
                        "denominator": denominator,
                        "mode": mode,
                        "truncation_order": "O(h) first-order one-sided",
                    }
                )
    margin = np.minimum(codes + 2048, 2047 - codes)
    full_neighbourhood_rows = int(np.count_nonzero(np.all(margin >= 35, axis=1)))
    other_rows = design.N_PAIRS - full_neighbourhood_rows
    # For rows at least 35 codes from int12 endpoints, MAX_CODE_STEP=32 plus
    # radius=2 guarantees the full 5^3 cube.  One mandatory coordinate pass
    # then has all 1+2D candidates.  Other rows use only the universal lower
    # bounds: event=1, endpoint-safe Jacobian=1+2D (the base may be one of
    # the two one-sided probes), cube>=1, descent>=1+D.
    full_row_candidates = 1 + (1 + 2 * receiver_close.D) + 5**receiver_close.NEIGHBOUR_DIMS + (
        1 + 2 * receiver_close.D
    )
    other_row_candidates = 1 + (1 + 2 * receiver_close.D) + 1 + (1 + receiver_close.D)
    minimum_candidate_denominator = (
        full_neighbourhood_rows * full_row_candidates + other_rows * other_row_candidates
    )
    camera_bytes = CAMERA_H * CAMERA_W * 3
    winner_bytes_per_pair = 3 * camera_bytes + 1024
    winner_bytes_per_stage = design.N_PAIRS * winner_bytes_per_pair
    row_model = {
        "pair_id": design.N_PAIRS - 1,
        "candidate_index_in_batch": receiver_close.POSE_BATCH - 1,
        "candidate_coordinate_delta": [-4095] * receiver_close.D,
        "slave_camera_payload": {
            "bytes": camera_bytes,
            "sha256": "f" * 64,
            "dtype": "uint8",
            "shape": [CAMERA_H, CAMERA_W, 3],
            "order": "C",
        },
        "pose_input_camera_payload": {
            "bytes": 2 * camera_bytes,
            "sha256": "f" * 64,
            "dtype": "uint8",
            "shape": [2, CAMERA_H, CAMERA_W, 3],
            "order": "C",
        },
    }
    certified_row_bytes = len(
        json.dumps(row_model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ) + 1
    context_model = {
        "schema": "ddm_jo4_certified_rebuild_batch.v1",
        "regeneration_context": {
            "entrypoint_sha256": "f" * 64,
            "workload_identity_sha256": "f" * 64,
            "base_archive_sha256": "f" * 64,
            "stage_id": "collateral_finish",
            "solve_phase": "x" * 96,
        },
        "tuple_join": "regeneration_context + rows[pair_id,candidate_coordinate_delta]",
        "candidate_denominator": receiver_close.POSE_BATCH,
        "rows": [],
        "all_camera_hashes_computed_while_materialized": True,
        "nonwinner_buffer_release_allowed_only_after_this_manifest": True,
    }
    manifest_context_bytes = len(
        json.dumps(context_model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ) + 1
    minimum_manifest_denominator = full_neighbourhood_rows * 7 + other_rows * 4
    certified_rebuild_bytes_per_stage = (
        minimum_candidate_denominator * certified_row_bytes
        + minimum_manifest_denominator * manifest_context_bytes
    )
    # Codes and Pose6 are retained as small full-byte NPYs in addition to the
    # camera rebuild certificate. This bound includes both batch and aggregate
    # copies plus headers without charging camera bytes twice.
    explored_small_state_bytes_per_candidate = 192
    explored_small_state_bytes_per_stage = (
        minimum_candidate_denominator * explored_small_state_bytes_per_candidate
    )
    one_stage_projected = (
        winner_bytes_per_stage
        + certified_rebuild_bytes_per_stage
        + explored_small_state_bytes_per_stage
    )
    all_stage_projected = one_stage_projected * stage_denominator
    blockers = []
    if blocked_coordinates:
        blockers.append(
            "FRESH_SCHUR_ENDPOINT_DERIVATIVE_BLOCKED:"
            f"blocked_coordinates={len(blocked_coordinates)}"
        )
    non_solver_reserve = 48 * 1024**3
    total_projected = all_stage_projected + non_solver_reserve
    if total_projected > free_bytes:
        blockers.append(
            "RETAINED_FRESH_SCHUR_STORAGE_BLOCKED:"
            f"all_stage_plus_reserve_projected_bytes={total_projected},free_bytes={free_bytes}"
        )
    return {
        "schema": "ddm_jo3_receiver_scale_preflight.v1",
        "passed": not blockers,
        "blockers": blockers,
        "endpoint_coordinates": endpoint.tolist(),
        "endpoint_pair_denominator": len(endpoint_pairs),
        "endpoint_one_sided_coordinate_denominator": len(endpoint_probes),
        "endpoint_blocked_coordinate_denominator": len(blocked_coordinates),
        "endpoint_probe_census": endpoint_probes,
        "blocked_coordinate_census": blocked_coordinates,
        "derivative_mode_denominators": derivative_modes,
        "full_neighbourhood_row_denominator": full_neighbourhood_rows,
        "other_row_denominator": other_rows,
        "minimum_candidate_denominator_per_stage": minimum_candidate_denominator,
        "full_winner_camera_bytes_per_pair": winner_bytes_per_pair,
        "full_winner_bytes_per_stage": winner_bytes_per_stage,
        "certified_rebuild_row_bytes_bound": certified_row_bytes,
        "certified_rebuild_manifest_context_bytes_bound": manifest_context_bytes,
        "certified_rebuild_manifest_denominator_per_stage": minimum_manifest_denominator,
        "certified_rebuild_bytes_per_stage": certified_rebuild_bytes_per_stage,
        "explored_small_state_bytes_per_candidate": explored_small_state_bytes_per_candidate,
        "explored_small_state_bytes_per_stage": explored_small_state_bytes_per_stage,
        "one_stage_projected_retained_bytes": one_stage_projected,
        "all_stage_projected_retained_bytes": all_stage_projected,
        "non_solver_and_extra_pass_reserve_bytes": non_solver_reserve,
        "all_stage_plus_reserve_projected_bytes": total_projected,
        "available_free_bytes": free_bytes,
        "coordinate_descent_extra_passes_included": 0,
        "bound_scope": (
            "two-tier baseline projection: full stage winners plus compact nonwinner camera "
            "certificates and full small state; 48 GiB separately reserves extra descent, "
            "fields, checkpoints, scorer surfaces, and coder artifacts; every write fails closed"
        ),
    }


def memory_preflight(
    *,
    config: design.CompiledConfig,
    output_receipt: Path,
    producer_command: Sequence[str],
) -> dict[str, Any]:
    configure_determinism(config.seed)
    semantic, surface, modules = load_semantic(config)
    storage = verify_storage(config)
    scale = receiver_scale_preflight(
        surface=surface,
        free_bytes=int(storage["free_bytes"]),
        stage_denominator=len(config.stages),
    )
    arrays = open_inputs(config)
    segnet, posenet, patch_receipt = load_scorers(config)
    model = worker.HybridOutputResidual(config.actuation.hidden_channels, float(config.actuation.max_rgb_delta.value))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.stages[0].learning_rate.value)
    frame0 = receiver_close.render_frame0(surface, modules, surface.codes[:1], 0)
    retained = output_receipt.parent / "retained"
    started = time.monotonic()
    step_result = one_training_step(
        config=config,
        stage=config.stages[0],
        pair=0,
        semantic=semantic,
        model=model,
        segnet=segnet,
        posenet=posenet,
        arrays=arrays,
        frame0=frame0,
        duals=worker.DualState(),
        retain_root=retained / "real_pair_0000",
        optimizer=optimizer,
        require_nonzero_gradient=True,
    )
    elapsed = time.monotonic() - started
    scorer_pair_bytes = sum(record["bytes"] for record in step_result["payloads"].values())
    measured = peak_rss_bytes()
    allocator_and_receiver_reserve = 2 * 1024**3
    projected = measured + (FIELD_BATCH_PAIRS - 1) * scorer_pair_bytes + allocator_and_receiver_reserve
    requested = config.memory_preflight.requested_memory_bytes
    headroom = requested - projected
    memory_passed = projected <= requested and headroom >= config.memory_preflight.minimum_headroom_bytes
    passed = memory_passed and bool(scale["passed"])
    patch_record = atomic_json(retained / "YUV6_PATCH_RECEIPT.json", patch_receipt)
    training_seconds = elapsed * sum(stage.fail_safe_steps for stage in config.stages)
    field_scoring_seconds = elapsed * design.N_PAIRS * len(config.stages)
    carrier_low_seconds = 3 * 600 * 39.0
    carrier_high_seconds = 3 * 600 * 64.0
    receiver_and_coder_upper_seconds = 3 * (1800.0 + 600.0)
    wall_low = training_seconds + field_scoring_seconds + carrier_low_seconds
    wall_high = (
        training_seconds
        + field_scoring_seconds
        + carrier_high_seconds
        + receiver_and_coder_upper_seconds
    )
    result = {
        "schema": design.MEMORY_RECEIPT_SCHEMA,
        "passed": passed,
        "device": "local CPU",
        "training_batch_pairs": TRAIN_BATCH_PAIRS,
        "field_batch_pairs": FIELD_BATCH_PAIRS,
        "geometry": [design.N_PAIRS, design.SEG_H, design.SEG_W],
        "measured_peak_rss_bytes": measured,
        "projected_n600_peak_rss_bytes": projected,
        "projection_method": (
            "measured one-pair full semantic-residual-R-SegNet-PoseNet forward/backward RSS "
            "+ 15 retained scorer-pair surfaces + 2 GiB allocator/receiver reserve; n600 is "
            "streamed in chunks and is never resident"
        ),
        "chunk_pair_limit": CHUNK_PAIR_LIMIT,
        "chunked_verdict": "PASS" if memory_passed else "FAIL",
        "requested_memory_bytes": requested,
        "headroom_bytes": headroom,
        "workload_config_sha256": config.workload_config_sha256,
        "producer_command": list(producer_command),
        "wall_clock_projection": {
            "one_real_training_step_seconds": elapsed,
            "lower_seconds": wall_low,
            "upper_seconds": wall_high,
            "components": {
                "training_steps": training_seconds,
                "full_field_scoring_conservative": field_scoring_seconds,
                "carrier_resolve_lower": carrier_low_seconds,
                "carrier_resolve_upper": carrier_high_seconds,
                "receiver_and_real_coder_upper": receiver_and_coder_upper_seconds,
            },
            "carrier_resolve_anchor": ("JG1 measured 39-64 seconds per pair; three fresh n600 stage solves"),
        },
        "receiver_scale_preflight": scale,
        "retained_payloads": {
            "real_training_step": step_result,
            "yuv6_patch_receipt": patch_record,
        },
        "created_at_utc": utc_now(),
    }
    atomic_json(output_receipt, result)
    if not passed:
        raise JO3EntrypointError(
            "real-scale preflight failed: "
            + ";".join(scale["blockers"] or [f"projected_rss={projected},requested={requested}"])
        )
    return result


def update_ema(ema: OrderedDict[str, torch.Tensor], model: nn.Module) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            ema[name].mul_(EMA_DECAY).add_(value.detach().cpu(), alpha=1.0 - EMA_DECAY)


def save_resume_pointer(
    checkpoint_root: Path,
    checkpoint: Path,
    manifest: Mapping[str, Any],
) -> None:
    atomic_json(
        checkpoint_root / "RESUME_LATEST.json",
        {
            "schema": "ddm_jo3_resume_pointer.v1",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_manifest_sha256": file_record(checkpoint / "CHECKPOINT.json")["sha256"],
            "stage_id": manifest["stage_id"],
            "step": manifest["step"],
            "updated_at_utc": utc_now(),
        },
        replace_metadata=True,
    )


def save_checkpoint(
    *,
    checkpoint_root: Path,
    stage_id: str,
    step: int,
    field_cursor: int,
    package_cursor: int,
    model: nn.Module,
    ema: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    duals: worker.DualState,
    config_sha256: str,
) -> dict[str, Any]:
    checkpoint = checkpoint_root / stage_id / f"step_{step:06d}_field_{field_cursor:04d}_package_{package_cursor}"
    if (checkpoint / "CHECKPOINT.json").is_file():
        manifest = worker.validate_checkpoint_bundle(checkpoint, config_sha256)
        retained_live = torch.load(checkpoint / "live.pt", map_location="cpu", weights_only=True)
        retained_ema = torch.load(checkpoint / "ema.pt", map_location="cpu", weights_only=True)
        live_differs = any(
            not torch.equal(model.state_dict()[name].detach().cpu(), value)
            for name, value in retained_live.items()
        )
        ema_differs = any(
            not torch.equal(ema[name].detach().cpu(), value)
            for name, value in retained_ema.items()
        )
        if live_differs or ema_differs:
            raise JO3EntrypointError("existing checkpoint differs from the deterministic stage state")
        save_resume_pointer(checkpoint_root, checkpoint, manifest)
        return manifest
    manifest = worker.save_checkpoint_bundle(
        checkpoint,
        model=model,
        ema_state=ema,
        optimizer=optimizer,
        duals=duals,
        cursor=worker.ResumeCursor(stage_id, step, field_cursor, package_cursor),
        config_sha256=config_sha256,
    )
    worker.validate_checkpoint_bundle(checkpoint, config_sha256)
    save_resume_pointer(checkpoint_root, checkpoint, manifest)
    return manifest


def restore_checkpoint(
    checkpoint_root: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config_sha256: str,
) -> tuple[OrderedDict[str, torch.Tensor], worker.DualState, worker.ResumeCursor] | None:
    pointer_path = checkpoint_root / "RESUME_LATEST.json"
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    checkpoint = Path(pointer["checkpoint"]).resolve()
    allowed = checkpoint_root.resolve()
    if allowed != checkpoint and allowed not in checkpoint.parents:
        raise JO3EntrypointError("resume pointer escaped its checkpoint root")
    manifest = worker.validate_checkpoint_bundle(checkpoint, expected_config_sha256)
    if file_record(checkpoint / "CHECKPOINT.json")["sha256"] != pointer["checkpoint_manifest_sha256"]:
        raise JO3EntrypointError("resume pointer manifest binding differs")
    model.load_state_dict(torch.load(checkpoint / "live.pt", map_location="cpu", weights_only=True))
    ema_raw = torch.load(checkpoint / "ema.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
    # This payload is produced locally by save_checkpoint_bundle and is
    # content-bound by CHECKPOINT.json before deserialization.
    rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch_cpu"])
    duals = worker.DualState(**json.loads((checkpoint / "duals.json").read_text()))
    cursor = worker.ResumeCursor(**json.loads((checkpoint / "resume_cursor.json").read_text()))
    if (cursor.stage_id, cursor.step) != (manifest["stage_id"], manifest["step"]):
        raise JO3EntrypointError("resume cursor differs from checkpoint manifest")
    return OrderedDict((name, value.detach().cpu()) for name, value in ema_raw.items()), duals, cursor


def materialize_candidate_master(
    *,
    stage_root: Path,
    semantic: nn.Module,
    model: worker.HybridOutputResidual,
    ema: Mapping[str, torch.Tensor],
    tokens: np.ndarray,
    workload_config_sha256: str,
    stage_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_model = worker.HybridOutputResidual(
        model.hidden_channels, model.max_rgb_delta
    ).eval()
    candidate_model.load_state_dict(ema, strict=True)
    payload = residual_runtime.encode_residual_state(
        candidate_model.state_dict(),
        hidden_channels=candidate_model.hidden_channels,
        max_rgb_delta=candidate_model.max_rgb_delta,
    )
    residual_path = stage_root / "retained/residual.j2s1"
    residual_sha256 = hashlib.sha256(payload).hexdigest()
    fingerprint = object_fingerprint(
        {
            "workload_config_sha256": workload_config_sha256,
            "stage_id": stage_id,
            "residual_payload_sha256": residual_sha256,
        }
    )
    progress_path = stage_root / "retained/MATERIALIZE_CURSOR.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("candidate_object_fingerprint_sha256") != fingerprint:
            raise JO3EntrypointError("retained candidate materialization belongs to another object")
        pre_path = Path(progress["pre_r_path"]).resolve()
        master_path = Path(progress["candidate_master_path"]).resolve()
        allowed = (stage_root / "retained").resolve()
        if (
            allowed not in pre_path.parents
            or allowed not in master_path.parents
            or not pre_path.is_file()
            or not master_path.is_file()
        ):
            raise JO3EntrypointError("bound candidate materialization payload is absent or escaped")
        first = int(progress["next_pair"])
        if not 0 <= first <= design.N_PAIRS:
            raise JO3EntrypointError("candidate materialization cursor is outside n600")
        allocation_attempt_index = int(progress["allocation_attempt_index"])
        pre = np.lib.format.open_memmap(pre_path, mode="r+", dtype=np.float32)
        master = np.lib.format.open_memmap(master_path, mode="r+", dtype=np.uint8)
    else:
        attempt_index = 0
        while (stage_root / f"retained/master_attempt_{attempt_index:04d}").exists():
            attempt_index += 1
        attempt = stage_root / f"retained/master_attempt_{attempt_index:04d}"
        attempt.mkdir(parents=True)
        pre_path = attempt / "candidate_pre_r.float32.npy"
        master_path = attempt / "candidate_master.uint8.npy"
        first = 0
        allocation_attempt_index = attempt_index
        pre = np.lib.format.open_memmap(
            pre_path,
            mode="w+",
            dtype=np.float32,
            shape=(design.N_PAIRS, 3, design.SEG_H, design.SEG_W),
        )
        master = np.lib.format.open_memmap(
            master_path,
            mode="w+",
            dtype=np.uint8,
            shape=(design.N_PAIRS, CAMERA_H, CAMERA_W, 3),
        )
        atomic_json(
            progress_path,
            {
                "schema": "ddm_jo3_master_materialization_cursor.v1",
                "next_pair": 0,
                "pair_denominator": design.N_PAIRS,
                "candidate_object_fingerprint_sha256": fingerprint,
                "pre_r_path": str(pre_path.resolve()),
                "candidate_master_path": str(master_path.resolve()),
                "allocation_attempt_index": attempt_index,
                "all_prior_attempts_retained": True,
                "all_payloads_retained": True,
            },
        )
    residual_record = atomic_bytes(residual_path, payload)
    exact_model = residual_runtime.residual_from_payload(payload).eval()
    with torch.inference_mode():
        for pair in range(first, design.N_PAIRS):
            token = torch.from_numpy(np.asarray(tokens[pair]).copy())[None].long()
            index = torch.tensor([pair], dtype=torch.long)
            value = (semantic(token, index) + exact_model(token)).clamp(0.0, 255.0)
            camera = (
                functional.interpolate(value, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
                .clamp(0.0, 255.0)
                .round()
            )
            pre[pair] = value[0].cpu().numpy()
            master[pair] = camera[0].to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            pre.flush()
            master.flush()
            atomic_json(
                progress_path,
                {
                    "schema": "ddm_jo3_master_materialization_cursor.v1",
                    "next_pair": pair + 1,
                    "pair_denominator": design.N_PAIRS,
                    "candidate_object_fingerprint_sha256": fingerprint,
                    "pre_r_path": str(pre_path.resolve()),
                    "candidate_master_path": str(master_path.resolve()),
                    "allocation_attempt_index": allocation_attempt_index,
                    "all_prior_attempts_retained": True,
                    "all_payloads_retained": True,
                },
                replace_metadata=True,
            )
    return file_record(master_path), residual_record


def materialize_base_frame0(
    *,
    run_root: Path,
    surface: Any,
    modules: Any,
    source_object_sha256: str,
) -> np.ndarray:
    """Retain the exact pair-specific fx5 frame-0 field, resumably."""
    cursor_path = run_root / "retained/FX5_FRAME0_CURSOR.json"
    if cursor_path.is_file():
        cursor = json.loads(cursor_path.read_text())
        if cursor.get("source_object_sha256") != source_object_sha256:
            raise JO3EntrypointError("retained fx5 frame-0 field belongs to another object")
        path = Path(cursor["retained_field"]).resolve()
        if (run_root / "retained").resolve() not in path.parents or not path.is_file():
            raise JO3EntrypointError("bound fx5 frame-0 field is absent or escaped")
        first = int(cursor["next_pair"])
        if not 0 <= first <= design.N_PAIRS:
            raise JO3EntrypointError("fx5 frame-0 cursor is outside n600")
        value = np.lib.format.open_memmap(path, mode="r+", dtype=np.uint8)
    else:
        attempt_index = 0
        while (run_root / f"retained/fx5_frame0_attempt_{attempt_index:04d}").exists():
            attempt_index += 1
        attempt = run_root / f"retained/fx5_frame0_attempt_{attempt_index:04d}"
        attempt.mkdir(parents=True)
        path = attempt / "fx5_base_frame0.uint8.npy"
        first = 0
        value = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.uint8,
            shape=(design.N_PAIRS, CAMERA_H, CAMERA_W, 3),
        )
        atomic_json(
            cursor_path,
            {
                "schema": "ddm_jo3_fx5_frame0_cursor.v1",
                "next_pair": 0,
                "pair_denominator": design.N_PAIRS,
                "source_object_sha256": source_object_sha256,
                "retained_field": str(path.resolve()),
                "allocation_attempt_index": attempt_index,
                "all_prior_attempts_retained": True,
            },
        )
    for pair in range(first, design.N_PAIRS):
        rendered = receiver_close.render_frame0(surface, modules, surface.codes[pair : pair + 1], pair)
        value[pair] = rendered[0]
        value.flush()
        atomic_json(
            cursor_path,
            {
                "schema": "ddm_jo3_fx5_frame0_cursor.v1",
                "next_pair": pair + 1,
                "pair_denominator": design.N_PAIRS,
                "source_object_sha256": source_object_sha256,
                "retained_field": str(path.resolve()),
            },
            replace_metadata=True,
        )
    return np.load(path, mmap_mode="r", allow_pickle=False)


def solve_fresh_compensation_resumable(
    *,
    stage_root: Path,
    candidate_master: Mapping[str, Any],
    base_pose6: Mapping[str, Any],
    semantic_object_sha256: str,
    posenet: nn.Module,
    archive: Path,
    runtime_root: Path,
    workload_config_sha256: str,
) -> dict[str, Any]:
    """Resume JO2 pair receipts and preserve/retry its narrow allocation crash window."""
    pointer_path = stage_root / "FRESH_SCHUR_POINTER.json"
    fingerprint = object_fingerprint(
        {
            "workload_config_sha256": workload_config_sha256,
            "candidate_master": dict(candidate_master),
            "base_pose6": dict(base_pose6),
            "semantic_object_sha256": semantic_object_sha256,
        }
    )
    pointer: dict[str, Any] | None = None
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        if pointer.get("solve_object_fingerprint_sha256") != fingerprint:
            raise JO3EntrypointError("retained fresh-Schur pointer belongs to another object")
        if pointer.get("status") == "COMPLETE":
            result_path = receiver_close.verify_record(pointer["result"])
            result = json.loads(result_path.read_text())
            if result.get("status") != "COMPLETE":
                raise JO3EntrypointError("retained fresh-Schur result is incomplete")
            receiver_close.verify_record(result["candidate_codes"])
            receiver_close.verify_record(result["candidate_frame0"])
            output = Path(pointer["output"]).resolve()
            allowed = stage_root.resolve()
            if allowed != output and allowed not in output.parents:
                raise JO3EntrypointError("retained fresh-Schur output escaped the stage root")
            CertifiedCandidateRetention(
                solve_root=output,
                stage_id=stage_root.name,
                workload_config_sha256=workload_config_sha256,
                base_archive_sha256=file_record(archive)["sha256"],
            ).verify_inventory(result.get("retention_inventory"))
            return result
    attempt_index = int(pointer["attempt_index"]) if pointer is not None else 0
    output = stage_root / f"fresh_schur_attempt_{attempt_index:04d}"
    if pointer is None:
        while output.exists():
            attempt_index += 1
            output = stage_root / f"fresh_schur_attempt_{attempt_index:04d}"
        atomic_json(
            pointer_path,
            {
                "schema": "ddm_jo3_fresh_schur_pointer.v1",
                "solve_object_fingerprint_sha256": fingerprint,
                "attempt_index": attempt_index,
                "status": "ACTIVE",
                "output": str(output.resolve()),
                "all_prior_attempts_retained": True,
            },
        )
    try:
        retention = CertifiedCandidateRetention(
            solve_root=output,
            stage_id=stage_root.name,
            workload_config_sha256=workload_config_sha256,
            base_archive_sha256=file_record(archive)["sha256"],
        )
        result = receiver_close.solve_fresh_compensation(
            candidate_master=candidate_master,
            base_pose6=base_pose6,
            semantic_object_sha256=semantic_object_sha256,
            output=output,
            posenet=posenet,
            archive=archive,
            runtime_root=runtime_root,
            retention=retention,
        )
    except receiver_close.JO2ReceiverCloseError as error:
        if "frame-0 partial exists without resume cursor" not in str(error):
            raise
        allocation_retries = int(pointer.get("allocation_retries", 0)) if pointer is not None else 0
        if allocation_retries >= 1:
            raise JO3EntrypointError("fresh-Schur allocation retry failed twice; retained attempts need audit") from error
        retry_index = attempt_index + 1
        retry = stage_root / f"fresh_schur_attempt_{retry_index:04d}"
        while retry.exists():
            retry_index += 1
            retry = stage_root / f"fresh_schur_attempt_{retry_index:04d}"
        atomic_json(
            pointer_path,
            {
                "schema": "ddm_jo3_fresh_schur_pointer.v1",
                "solve_object_fingerprint_sha256": fingerprint,
                "attempt_index": retry_index,
                "status": "ACTIVE",
                "output": str(retry.resolve()),
                "all_prior_attempts_retained": True,
                "allocation_retries": allocation_retries + 1,
                "retry_reason": str(error),
            },
            replace_metadata=True,
        )
        return solve_fresh_compensation_resumable(
            stage_root=stage_root,
            candidate_master=candidate_master,
            base_pose6=base_pose6,
            semantic_object_sha256=semantic_object_sha256,
            posenet=posenet,
            archive=archive,
            runtime_root=runtime_root,
            workload_config_sha256=workload_config_sha256,
        )
    result_record = file_record(output / "FRESH_SCHUR_RESULT.json")
    atomic_json(
        pointer_path,
        {
            "schema": "ddm_jo3_fresh_schur_pointer.v1",
            "solve_object_fingerprint_sha256": fingerprint,
            "attempt_index": attempt_index,
            "status": "COMPLETE",
            "output": str(output.resolve()),
            "result": result_record,
            "all_prior_attempts_retained": True,
        },
        replace_metadata=True,
    )
    return result


def compile_receiver_closed_resumable(
    *,
    stage_root: Path,
    residual: Mapping[str, Any],
    solve: Mapping[str, Any],
    candidate_master: Mapping[str, Any],
    archive: Path,
    runtime_root: Path,
    workload_config_sha256: str,
) -> dict[str, Any]:
    """Retry a crashed compile in a new retained attempt; never erase the old one."""
    pointer_path = stage_root / "RECEIVER_CLOSE_POINTER.json"
    fingerprint = object_fingerprint(
        {
            "workload_config_sha256": workload_config_sha256,
            "residual": dict(residual),
            "candidate_master": dict(candidate_master),
            "semantic_object_sha256": solve.get("semantic_object_sha256"),
            "candidate_codes": solve.get("candidate_codes"),
        }
    )
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        if pointer.get("receiver_object_fingerprint_sha256") != fingerprint:
            raise JO3EntrypointError("retained receiver-close pointer belongs to another object")
        result_path = receiver_close.verify_record(pointer["result"])
        result = json.loads(result_path.read_text())
        if result.get("status") != "COMPLETE":
            raise JO3EntrypointError("retained receiver-close result is incomplete")
        for name in ("archive", "archive_repeat", "receiver_parseback"):
            receiver_close.verify_record(result[name])
        return result
    attempt_index = 0
    while (stage_root / f"receiver_close_attempt_{attempt_index:04d}").exists():
        attempt_index += 1
    output = stage_root / f"receiver_close_attempt_{attempt_index:04d}"
    result = receiver_close.compile_receiver_closed_stage(
        residual_payload=residual,
        solve_result=solve,
        output=output,
        candidate_master=candidate_master,
        archive=archive,
        runtime_root=runtime_root,
    )
    result_record = file_record(output / "RECEIVER_CLOSE_RESULT.json")
    atomic_json(
        pointer_path,
        {
            "schema": "ddm_jo3_receiver_close_pointer.v1",
            "receiver_object_fingerprint_sha256": fingerprint,
            "attempt_index": attempt_index,
            "result": result_record,
            "all_prior_attempts_retained": True,
        },
    )
    return result


def run_receiver(*, stage_root: Path, receiver_result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer_path = stage_root / "RECEIVER_EXECUTION_POINTER.json"
    fingerprint = object_fingerprint(dict(receiver_result["archive"]))
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        if pointer.get("archive_object_fingerprint_sha256") != fingerprint:
            raise JO3EntrypointError("retained receiver execution belongs to another archive")
        receipt_path = receiver_close.verify_record(pointer["receipt"])
        receipt = json.loads(receipt_path.read_text())
        raw = receiver_close.verify_record(receipt["raw"])
        if receipt.get("archive") != dict(receiver_result["archive"]):
            raise JO3EntrypointError("retained receiver execution archive binding differs")
        return file_record(raw), receipt
    attempt_index = 0
    while (stage_root / f"receiver_execution_attempt_{attempt_index:04d}").exists():
        attempt_index += 1
    root = stage_root / f"receiver_execution_attempt_{attempt_index:04d}"
    archive_dir = root / "archive"
    output = root / "output"
    names = root / "video_names.txt"
    raw_path = output / "0.raw"
    receipt_path = root / "RECEIVER_EXECUTION.json"
    archive_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(receiver_close.verify_record(receiver_result["archive"])) as archive:
        if archive.namelist() != ["p"]:
            raise JO3EntrypointError("receiver archive member census differs")
        archive.extractall(archive_dir)
    atomic_bytes(names, b"0.mp4\n")
    receiver_root = Path(str(receiver_result["archive"]["path"])).resolve().parent
    inflate_sh = receiver_root / "submission/inflate.sh"
    if not inflate_sh.is_file() or not os.access(inflate_sh, os.X_OK):
        raise JO3EntrypointError("staged shipped inflate.sh is absent or not executable")
    command = [
        str(inflate_sh.resolve()),
        str(archive_dir),
        str(output),
        str(names),
    ]
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=inflate_sh.parent,
        env={**os.environ, "PR130_INFLATE_DEVICE": "cpu", "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log = atomic_bytes(
        root / "inflate.log",
        (process.stdout + "\n--- STDERR ---\n" + process.stderr).encode(),
    )
    if process.returncode != 0 or not raw_path.is_file() or raw_path.stat().st_size != RAW_BYTES:
        raise JO3EntrypointError(f"shipped receiver failed rc={process.returncode}; log={log['path']}")
    raw = file_record(raw_path)
    receipt = {
        "schema": "ddm_jo3_receiver_execution.v1",
        "command": command,
        "returncode": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "archive": dict(receiver_result["archive"]),
        "raw": raw,
        "log": log,
        "axis": "[macOS-CPU real shipped receiver; no score authority]",
        "score_claim": False,
    }
    receipt_record = atomic_json(receipt_path, receipt)
    atomic_json(
        pointer_path,
        {
            "schema": "ddm_jo3_receiver_execution_pointer.v1",
            "archive_object_fingerprint_sha256": fingerprint,
            "attempt_index": attempt_index,
            "receipt": receipt_record,
            "all_prior_attempts_retained": True,
        },
    )
    return raw, receipt


def score_decoded_n600(
    *,
    stage_root: Path,
    raw_record: Mapping[str, Any],
    segnet: nn.Module,
    posenet: nn.Module,
    workload_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    raw_path = receiver_close.verify_record(raw_record)
    raw = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(design.N_PAIRS * 2, CAMERA_H, CAMERA_W, 3))
    root = stage_root / "retained/scorer_n600"
    root.mkdir(parents=True, exist_ok=True)
    cursor_path = root / "SCORER_CURSOR.json"
    fingerprint = object_fingerprint(
        {
            "workload_config_sha256": workload_config_sha256,
            "decoded_raw": dict(raw_record),
        }
    )
    if cursor_path.is_file():
        cursor = json.loads(cursor_path.read_text())
        if cursor.get("scorer_object_fingerprint_sha256") != fingerprint:
            raise JO3EntrypointError("retained scorer cursor belongs to another decoded object")
        paths = {name: Path(value).resolve() for name, value in cursor["retained_paths"].items()}
        if any(root.resolve() not in path.parents or not path.is_file() for path in paths.values()):
            raise JO3EntrypointError("bound scorer surface is absent or escaped")
        argmax_path = paths["candidate_argmax"]
        pose6_path = paths["pose6"]
        logits_path = paths["seg_logits"]
        seg_input_path = paths["seg_input"]
        pose_input_path = paths["pose_input"]
        first = int(cursor["next_pair"])
        if not 0 <= first <= design.N_PAIRS:
            raise JO3EntrypointError("scorer cursor is outside n600")
        mode = "r+"
    else:
        attempt_index = 0
        while (root / f"attempt_{attempt_index:04d}").exists():
            attempt_index += 1
        attempt = root / f"attempt_{attempt_index:04d}"
        attempt.mkdir(parents=True)
        argmax_path = attempt / "candidate_argmax.uint8.npy"
        pose6_path = attempt / "pose6.float32.npy"
        logits_path = attempt / "seg_logits.float32.npy"
        seg_input_path = attempt / "seg_input.float32.npy"
        pose_input_path = attempt / "pose_input.float32.npy"
        first = 0
        mode = "w+"
    argmax = np.lib.format.open_memmap(
        argmax_path, mode=mode, dtype=np.uint8, shape=(design.N_PAIRS, design.SEG_H, design.SEG_W)
    )
    pose6 = np.lib.format.open_memmap(pose6_path, mode=mode, dtype=np.float32, shape=(design.N_PAIRS, 6))
    logits = np.lib.format.open_memmap(
        logits_path, mode=mode, dtype=np.float32, shape=(design.N_PAIRS, 5, design.SEG_H, design.SEG_W)
    )
    seg_inputs = np.lib.format.open_memmap(
        seg_input_path, mode=mode, dtype=np.float32, shape=(design.N_PAIRS, 3, design.SEG_H, design.SEG_W)
    )
    pose_inputs = np.lib.format.open_memmap(
        pose_input_path, mode=mode, dtype=np.float32, shape=(design.N_PAIRS, 12, 192, 256)
    )
    retained_paths = {
        "candidate_argmax": str(argmax_path.resolve()),
        "pose6": str(pose6_path.resolve()),
        "seg_logits": str(logits_path.resolve()),
        "seg_input": str(seg_input_path.resolve()),
        "pose_input": str(pose_input_path.resolve()),
    }
    if not cursor_path.is_file():
        atomic_json(
            cursor_path,
            {
                "schema": "ddm_jo3_chunked_scorer_cursor.v1",
                "next_pair": 0,
                "pair_denominator": design.N_PAIRS,
                "batch_pairs": FIELD_BATCH_PAIRS,
                "chunk_pair_limit": CHUNK_PAIR_LIMIT,
                "scorer_object_fingerprint_sha256": fingerprint,
                "retained_paths": retained_paths,
                "all_prior_attempts_retained": True,
                "all_materialized_payloads_retained": True,
            },
        )
    with torch.inference_mode():
        for begin in range(first, design.N_PAIRS, FIELD_BATCH_PAIRS):
            end = min(begin + FIELD_BATCH_PAIRS, design.N_PAIRS)
            frames = torch.from_numpy(np.asarray(raw[2 * begin : 2 * end]).copy()).permute(0, 3, 1, 2).float()
            pairs = frames.reshape(end - begin, 2, 3, CAMERA_H, CAMERA_W)
            seg_input = segnet.preprocess_input(pairs)
            seg_output = segnet(seg_input)
            pose_input = posenet.preprocess_input(pairs)
            pose_output = posenet(pose_input)["pose"][..., :6]
            argmax[begin:end] = seg_output.argmax(dim=1).to(torch.uint8).cpu().numpy()
            logits[begin:end] = seg_output.cpu().numpy()
            seg_inputs[begin:end] = seg_input.cpu().numpy()
            pose_inputs[begin:end] = pose_input.cpu().numpy()
            pose6[begin:end] = pose_output.cpu().numpy()
            for value in (argmax, logits, seg_inputs, pose_inputs, pose6):
                value.flush()
            atomic_json(
                cursor_path,
                {
                    "schema": "ddm_jo3_chunked_scorer_cursor.v1",
                    "next_pair": end,
                    "pair_denominator": design.N_PAIRS,
                    "batch_pairs": FIELD_BATCH_PAIRS,
                    "chunk_pair_limit": CHUNK_PAIR_LIMIT,
                    "scorer_object_fingerprint_sha256": fingerprint,
                    "retained_paths": retained_paths,
                    "all_prior_attempts_retained": True,
                    "all_materialized_payloads_retained": True,
                },
                replace_metadata=True,
            )
    return {
        "candidate_argmax_field": file_record(argmax_path),
        "pose6_outputs": file_record(pose6_path),
        "seg_logits": file_record(logits_path),
        "seg_input": file_record(seg_input_path),
        "pose_input": file_record(pose_input_path),
    }


def complete_stage(
    *,
    config: design.CompiledConfig,
    stage: design.StageConfig,
    stage_root: Path,
    semantic: nn.Module,
    model: worker.HybridOutputResidual,
    ema: Mapping[str, torch.Tensor],
    arrays: Mapping[str, np.ndarray],
    segnet: nn.Module,
    posenet: nn.Module,
    duals: worker.DualState,
    expected_config_sha256: str,
) -> tuple[dict[str, Any], worker.DualState]:
    result_path = stage_root / "STAGE_RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("status") != "ADMITTED":
            raise JO3EntrypointError(f"resumed stage is not admitted: {stage.stage_id}")
        if result.get("config_sha256") != expected_config_sha256:
            raise JO3EntrypointError("retained stage result belongs to another compiled config")
        return result, worker.DualState(**result["next_duals"])
    master, residual = materialize_candidate_master(
        stage_root=stage_root,
        semantic=semantic,
        model=model,
        ema=ema,
        tokens=arrays["tokens"],
        workload_config_sha256=str(config.workload_config_sha256),
        stage_id=stage.stage_id,
    )
    surface, _ = receiver_close.load_surface(Path(config.inputs.rc2_archive.path), Path(config.inputs.rc2_runtime.path))
    semantic_object = residual_runtime.pack_semantic_blob(
        surface.parts.semantic_blob, Path(residual["path"]).read_bytes()
    )
    semantic_sha = hashlib.sha256(semantic_object).hexdigest()
    base_pose = file_record(Path(config.inputs.fx5_base_pose6.path))  # type: ignore[union-attr]
    solve = solve_fresh_compensation_resumable(
        stage_root=stage_root,
        candidate_master=master,
        base_pose6=base_pose,
        semantic_object_sha256=semantic_sha,
        posenet=posenet,
        archive=Path(config.inputs.rc2_archive.path),
        runtime_root=Path(config.inputs.rc2_runtime.path),
        workload_config_sha256=str(config.workload_config_sha256),
    )
    closed = compile_receiver_closed_resumable(
        stage_root=stage_root,
        residual=residual,
        solve=solve,
        candidate_master=master,
        archive=Path(config.inputs.rc2_archive.path),
        runtime_root=Path(config.inputs.rc2_runtime.path),
        workload_config_sha256=str(config.workload_config_sha256),
    )
    receiver_root = Path(str(closed["archive"]["path"])).resolve().parent
    raw, receiver_execution = run_receiver(stage_root=stage_root, receiver_result=closed)
    identity = receiver_close.verify_decoded_render_identity(
        receiver_close=closed,
        candidate_frame0=solve["candidate_frame0"],
        candidate_master=master,
        decoded_raw=raw,
        output=receiver_root,
    )
    scored = score_decoded_n600(
        stage_root=stage_root,
        raw_record=raw,
        segnet=segnet,
        posenet=posenet,
        workload_config_sha256=str(config.workload_config_sha256),
    )
    candidate = np.load(scored["candidate_argmax_field"]["path"], mmap_mode="r", allow_pickle=False)
    field = worker.exact_field_decomposition(arrays["base_argmax"], candidate, arrays["target"])
    field_record = atomic_json(
        stage_root / "retained/BHW_DECOMPOSITION.json",
        {
            "schema": "ddm_jo3_bhw_decomposition.v1",
            "B_fixed": field.fixed,
            "H_introduced": field.introduced,
            "wrong_to_wrong": field.wrong_to_wrong,
            "candidate_flips": field.candidate_flips,
            "denominator": design.SEG_DENOMINATOR,
        },
    )
    pose6 = np.load(scored["pose6_outputs"]["path"], mmap_mode="r", allow_pickle=False)
    d_pose = float(np.mean(np.square(pose6.astype(np.float64) - arrays["pose_target"].astype(np.float64))))
    admission = design.stage_admission(
        fixed=field.fixed,
        introduced=field.introduced,
        wrong_to_wrong=field.wrong_to_wrong,
        d_pose_candidate=d_pose,
        candidate_archive_bytes=int(closed["archive"]["bytes"]),
        single_p=bool(closed["single_p"]),
        package_parseback_identity=bool(closed["receiver_parseback_identity"] and identity["identity"]),
    )
    metrics = {
        "schema": "ddm_jo3_stage_metrics.v1",
        "stage_id": stage.stage_id,
        "pair_denominator": design.N_PAIRS,
        "field_cell_denominator": design.SEG_DENOMINATOR,
        "d_pose_denominator": design.N_PAIRS * 6,
        "admission": admission,
        "receiver_execution": receiver_execution,
        "extra_retained_scorer_surfaces": {
            name: record for name, record in scored.items() if name not in {"candidate_argmax_field", "pose6_outputs"}
        },
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
    }
    metrics_record = atomic_json(stage_root / "retained/METRICS.json", metrics)
    identity_record = file_record(receiver_root / "DECODED_RENDER_IDENTITY.json")
    retained = {
        "candidate_argmax_field": scored["candidate_argmax_field"],
        "bhw_decomposition": field_record,
        "pose6_outputs": scored["pose6_outputs"],
        "exact_package": dict(closed["archive"]),
        "decoded_render_identity": identity_record,
        "metrics_json": metrics_record,
    }
    validated = worker.validate_stage_package(
        archive=Path(closed["archive"]["path"]),
        repeat_archive=Path(closed["archive_repeat"]["path"]),
        retained_payloads=retained,
        receiver_parseback_identity=bool(identity["identity"]),
        compensation_object_sha256=str(solve["semantic_object_sha256"]),
        expected_object_sha256=str(closed["semantic_object_sha256"]),
    )
    next_duals = worker.update_duals_at_stage_boundary(duals, field=field, d_pose_candidate=d_pose)
    result = {
        "schema": "ddm_jo3_completed_stage.v1",
        "status": "ADMITTED" if admission["admissible"] else "REJECTED",
        "stage_id": stage.stage_id,
        "config_sha256": expected_config_sha256,
        "admission": admission,
        "validated_package": validated,
        "candidate_frame0": solve["candidate_frame0"],
        "next_duals": vars(next_duals),
        "all_materialized_payloads_retained": True,
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
    }
    atomic_json(result_path, result)
    if not admission["admissible"]:
        raise JO3EntrypointError(f"stage {stage.stage_id} failed exact admission: {admission['blockers']}")
    return result, next_duals


def train(
    *,
    config: design.CompiledConfig,
    expected_config_sha256: str,
    resume_from: Path,
) -> dict[str, Any]:
    ready = design.readiness(config)
    if ready["status"] != "READY_TO_FIRE_UNDER_STANDING_GO" or ready["blockers"]:
        raise JO3EntrypointError(f"training readiness is blocked: {ready['blockers']}")
    run_root = Path(config.output_root).resolve() / config.run_id
    expected_resume = run_root / "checkpoints"
    if resume_from.resolve() != expected_resume.resolve():
        raise JO3EntrypointError("resume root differs from the sealed FIRE_ORDER")
    configure_determinism(config.seed)
    storage_policy = write_storage_policy(run_root, config)
    semantic, surface, modules = load_semantic(config)
    arrays = open_inputs(config)
    segnet, posenet, patch_receipt = load_scorers(config)
    atomic_json(run_root / "retained/YUV6_PATCH_RECEIPT.json", patch_receipt)
    model = worker.HybridOutputResidual(config.actuation.hidden_channels, float(config.actuation.max_rgb_delta.value))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.stages[0].learning_rate.value)
    ema = OrderedDict((name, value.detach().cpu().clone()) for name, value in model.state_dict().items())
    duals = worker.DualState()
    cursor: worker.ResumeCursor | None = None
    restored = restore_checkpoint(
        resume_from,
        model=model,
        optimizer=optimizer,
        expected_config_sha256=expected_config_sha256,
    )
    if restored is not None:
        ema, duals, cursor = restored
    base_frame0 = materialize_base_frame0(
        run_root=run_root,
        surface=surface,
        modules=modules,
        source_object_sha256=config.inputs.rc2_archive.sha256,
    )
    if base_frame0.shape != (design.N_PAIRS, CAMERA_H, CAMERA_W, 3):
        raise JO3EntrypointError("fx5 frame-0 base materialization differs")
    current_frame0: np.ndarray = base_frame0
    completed: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(config.stages):
        stage_root = run_root / "stages" / f"{stage_index + 1:02d}_{stage.stage_id}"
        if (stage_root / "STAGE_RESULT.json").is_file():
            prior = json.loads((stage_root / "STAGE_RESULT.json").read_text())
            if prior.get("status") != "ADMITTED":
                raise JO3EntrypointError(f"retained stage is not admitted: {stage.stage_id}")
            if prior.get("config_sha256") != expected_config_sha256:
                raise JO3EntrypointError("retained stage result belongs to another compiled config")
            if cursor is None or design.REQUIRED_STAGE_IDS.index(cursor.stage_id) < stage_index:
                raise JO3EntrypointError("admitted stage has no covering retained checkpoint")
            if cursor.stage_id == stage.stage_id and cursor.step < stage.fail_safe_steps:
                raise JO3EntrypointError("admitted stage checkpoint predates its training boundary")
            frame0_path = receiver_close.verify_record(prior["candidate_frame0"])
            current_frame0 = np.load(frame0_path, mmap_mode="r", allow_pickle=False)
            duals = worker.DualState(**prior["next_duals"])
            completed.append(prior)
            continue
        if cursor is not None and design.REQUIRED_STAGE_IDS.index(cursor.stage_id) > stage_index:
            raise JO3EntrypointError("resume cursor skipped an incomplete stage")
        start = cursor.step if cursor is not None and cursor.stage_id == stage.stage_id else 0
        for group in optimizer.param_groups:
            group["lr"] = stage.learning_rate.value
        for step in range(start, stage.fail_safe_steps):
            pair = (stage_index * 199 + step * 137) % design.N_PAIRS
            one_training_step(
                config=config,
                stage=stage,
                pair=pair,
                semantic=semantic,
                model=model,
                segnet=segnet,
                posenet=posenet,
                arrays=arrays,
                frame0=current_frame0,
                duals=duals,
                retain_root=stage_root / "retained/training_steps" / f"step_{step + 1:06d}_pair_{pair:04d}",
                optimizer=optimizer,
            )
            update_ema(ema, model)
            if (step + 1) % stage.checkpoint_every_steps == 0:
                save_checkpoint(
                    checkpoint_root=resume_from,
                    stage_id=stage.stage_id,
                    step=step + 1,
                    field_cursor=0,
                    package_cursor=0,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    duals=duals,
                    config_sha256=expected_config_sha256,
                )
        save_checkpoint(
            checkpoint_root=resume_from,
            stage_id=stage.stage_id,
            step=stage.fail_safe_steps,
            field_cursor=0,
            package_cursor=0,
            model=model,
            ema=ema,
            optimizer=optimizer,
            duals=duals,
            config_sha256=expected_config_sha256,
        )
        result, duals = complete_stage(
            config=config,
            stage=stage,
            stage_root=stage_root,
            semantic=semantic,
            model=model,
            ema=ema,
            arrays=arrays,
            segnet=segnet,
            posenet=posenet,
            duals=duals,
            expected_config_sha256=expected_config_sha256,
        )
        frame0_path = receiver_close.verify_record(result["candidate_frame0"])
        current_frame0 = np.load(frame0_path, mmap_mode="r", allow_pickle=False)
        completed.append(result)
        save_checkpoint(
            checkpoint_root=resume_from,
            stage_id=stage.stage_id,
            step=stage.fail_safe_steps,
            field_cursor=design.N_PAIRS,
            package_cursor=1,
            model=model,
            ema=ema,
            optimizer=optimizer,
            duals=duals,
            config_sha256=expected_config_sha256,
        )
        cursor = None
    result = {
        "schema": "ddm_jo3_training_result.v1",
        "status": "COMPLETE",
        "stage_denominator": len(config.stages),
        "stages": completed,
        "storage_policy": storage_policy,
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
        "next_consumer": "MAIN exact upstream/evaluate.py CPU/CUDA replay only after archive selection",
    }
    atomic_json(run_root / "TRAINING_RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("memory-preflight", "train"):
        command = sub.add_parser(name)
        command.add_argument("--compiled-config", required=True, type=Path)
        command.add_argument("--expected-config-sha256", required=True)
        command.add_argument("--main-owned-dispatch-authorization", action="store_true")
        if name == "memory-preflight":
            command.add_argument("--output-receipt", required=True, type=Path)
        else:
            command.add_argument("--resume-from", required=True, type=Path)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        require_governed_admission(args.main_owned_dispatch_authorization)
        config = load_config(args.compiled_config, args.expected_config_sha256)
        if args.command == "memory-preflight":
            result = memory_preflight(
                config=config,
                output_receipt=args.output_receipt.resolve(),
                producer_command=sys.argv,
            )
        else:
            result = train(
                config=config,
                expected_config_sha256=args.expected_config_sha256,
                resume_from=args.resume_from,
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        design.JO1Error,
        worker.JO1WorkerError,
        receiver_close.JO2ReceiverCloseError,
    ) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
