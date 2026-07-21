# SPDX-License-Identifier: MIT
"""Streamed C2 quotient-residual training inside source-centred RGB bands.

This is the dedicated trainer integration for :mod:`integer_plane_emitter`.
It never imports or invokes the level-set witness trainer.  The logical state
always contains all 600 pair codes, while dense scorer planes are fetched in
bounded pair batches from immutable ``.npy``/memmap stores.

The module owns build and local-verification authority only.  Its CLI validates
the typed policy and input custody, performs storage preflight, and writes
distinct atomic checkpoints.  It does not launch itself, score, promote, or
mutate a pointer.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from tac.boundary_math.c2_r1b4_curvelet_binding import (
    BINDING_BASIS_ID,
    C2R1B4CurveletBinding,
    C2R1B4CurveletBindingError,
)
from tac.boundary_math.integer_plane_emitter import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PLANE_COUNT,
    RGB_CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    CapacitySignature,
    StructuredEmitterState,
    deterministic_coordinate_basis,
    factor2_operator,
    torch_uint8,
)
from tac.witness_dsl.integer_plane_emitter_policy import (
    BasisMode,
    IntegerPlaneEmitterPolicy,
    IntegerPlaneEmitterStageCheckpoint,
    PolicyMode,
    STEMode,
)

LOGICAL_PAIR_COUNT: Final = 600
PLANE_SHAPE: Final = (PLANE_COUNT, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS)
BAND_SCHEMA: Final = "c2_anisotropic_band_artifact.v1"
TRAINER_CONFIG_SCHEMA: Final = "c2_integer_plane_banded_trainer_config.v1"
TRAINING_RECEIPT_SCHEMA: Final = "c2_integer_plane_banded_training_receipt.v1"
BASE_MATERIALIZATION_SCHEMA: Final = "c2_base_plane_materialization.v1"
BASE_SCORER_CACHE_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/cache/base_scorer_planes")
SCORER_PROJECTION_BATCH_PLANES: Final = 4
ALLOWED_STAGES: Final = ("warmup", "band_fit", "rate_polish")
RATE_SCORE_PER_BYTE: Final = 25.0 / 37_545_489.0
C2_MEASURED_FLIP_CANDIDATES: Final = 38_077
INACTIVE_BAND_RADIUS: Final = 255.0


class C2BandedTrainerError(ValueError):
    """Fail-closed trainer, custody, or resume contract violation."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise C2BandedTrainerError("value is not canonical JSON") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise C2BandedTrainerError(f"{field} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise C2BandedTrainerError(f"{field} must be a lowercase SHA-256") from exc
    if value.lower() != value or value == "0" * 64:
        raise C2BandedTrainerError(f"{field} must be a non-placeholder lowercase SHA-256")
    return value


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _npy_identity(path: Path, *, expected_sha256: str) -> tuple[np.memmap, str]:
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256_file(resolved)
    if actual != _require_sha256(expected_sha256, f"{resolved.name}.sha256"):
        raise C2BandedTrainerError(f"npy custody mismatch: {resolved}")
    array = np.load(resolved, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise C2BandedTrainerError(f"npy store did not open as a read-only memmap: {resolved}")
    return array, actual


@dataclass(frozen=True, slots=True)
class BandArtifact:
    """Validated positive-anisotropic or explicitly labelled zero control."""

    manifest_path: Path
    manifest_sha256: str
    mode: str
    pair_count: int
    source_planes: np.memmap
    radii: np.memmap | None
    source_sha256: str
    custody: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> BandArtifact:
        manifest_path = Path(path).expanduser().resolve(strict=True)
        raw = manifest_path.read_bytes()
        if canonical_json(json.loads(raw.decode("ascii"))) != raw:
            raise C2BandedTrainerError("band manifest must be canonical JSON")
        doc = json.loads(raw)
        required = {
            "schema",
            "mode",
            "pair_count",
            "geometry",
            "source_planes",
            "radii",
            "custody",
        }
        if not isinstance(doc, dict) or set(doc) != required:
            raise C2BandedTrainerError("band manifest fields mismatch")
        if doc["schema"] != BAND_SCHEMA:
            raise C2BandedTrainerError("band manifest schema mismatch")
        if doc["mode"] not in {"positive_anisotropic", "zero_radius_control"}:
            raise C2BandedTrainerError("band mode is unknown")
        if doc["pair_count"] != LOGICAL_PAIR_COUNT:
            raise C2BandedTrainerError("band artifact must cover exactly 600 logical pairs")
        if doc["geometry"] != [*PLANE_SHAPE]:
            raise C2BandedTrainerError("band artifact geometry mismatch")

        def array_record(name: str, *, required_array: bool) -> tuple[np.memmap | None, str | None]:
            record = doc[name]
            if record is None and not required_array:
                return None, None
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise C2BandedTrainerError(f"{name} record fields mismatch")
            array_path = (manifest_path.parent / record["path"]).resolve(strict=True)
            try:
                array_path.relative_to(manifest_path.parent)
            except ValueError as exc:
                raise C2BandedTrainerError(f"{name} must remain beneath the manifest directory") from exc
            return _npy_identity(array_path, expected_sha256=record["sha256"])

        source, source_sha = array_record("source_planes", required_array=True)
        assert source is not None and source_sha is not None
        if source.dtype != np.uint8 or tuple(source.shape) != (LOGICAL_PAIR_COUNT, *PLANE_SHAPE):
            raise C2BandedTrainerError("source planes must be uint8 [600,2,384,512,3]")
        if source.flags.writeable:
            raise C2BandedTrainerError("source-plane store must be read-only")

        positive = doc["mode"] == "positive_anisotropic"
        radii, _ = array_record("radii", required_array=positive)
        custody = doc["custody"]
        if not isinstance(custody, dict):
            raise C2BandedTrainerError("band custody must be a mapping")
        if positive:
            required_custody = {
                "derivation",
                "margins_sha256",
                "winner_sha256",
                "rival_sha256",
                "unit_head_normal_pullback_rgb_sha256",
                "pair_norms_sha256",
                "config",
                "ev_selection",
            }
            if set(custody) != required_custody:
                raise C2BandedTrainerError("positive anisotropic band custody fields mismatch")
            if custody["derivation"] != "derive_hyperplane_channel_band":
                raise C2BandedTrainerError("positive bands must be produced by derive_hyperplane_channel_band")
            for field in required_custody - {"derivation", "config", "ev_selection"}:
                _require_sha256(custody[field], f"band.custody.{field}")
            config = custody["config"]
            if not isinstance(config, dict) or set(config) != {
                "scale",
                "local_lipschitz",
                "max_rgb_radius",
            }:
                raise C2BandedTrainerError("positive band config fields mismatch")
            if (
                not math.isfinite(float(config["scale"]))
                or float(config["scale"]) <= 0.0
                or not math.isfinite(float(config["local_lipschitz"]))
                or float(config["local_lipschitz"]) <= 0.0
                or not math.isfinite(float(config["max_rgb_radius"]))
                or not 0.0 < float(config["max_rgb_radius"]) < INACTIVE_BAND_RADIUS
            ):
                raise C2BandedTrainerError("positive band custody requires scale > 0")
            ev = custody["ev_selection"]
            ev_fields = {
                "policy",
                "candidate_flip_count",
                "selected_pixel_count",
                "inactive_radius",
                "rate_break_even_score_per_byte",
                "stopped_below_break_even",
                "blanket_fix",
                "artifact_records",
                "law_refs",
                "metric",
                "carrier_basis",
                "realization_predictor",
                "pose_factorization",
                "gauge_status",
            }
            if not isinstance(ev, dict) or set(ev) != ev_fields:
                raise C2BandedTrainerError("positive band EV-selection custody fields mismatch")
            expected_candidates = (
                C2_MEASURED_FLIP_CANDIDATES if LOGICAL_PAIR_COUNT == 600 else int(ev["candidate_flip_count"])
            )
            if (
                ev["policy"] != "measured_reverse_waterfill_highest_ev_first"
                or type(ev["candidate_flip_count"]) is not int
                or ev["candidate_flip_count"] != expected_candidates
                or type(ev["selected_pixel_count"]) is not int
                or not 0 < ev["selected_pixel_count"] <= ev["candidate_flip_count"]
                or float(ev["inactive_radius"]) != INACTIVE_BAND_RADIUS
                or not math.isclose(
                    float(ev["rate_break_even_score_per_byte"]),
                    RATE_SCORE_PER_BYTE,
                    rel_tol=0.0,
                    abs_tol=1e-18,
                )
                or ev["stopped_below_break_even"] is not True
                or ev["blanket_fix"] is not False
                or ev["metric"] != "fisher_top1_top2_margin"
                or ev["carrier_basis"] != "cgauge_curvelet_parabolic_bank_v1"
                or ev["realization_predictor"] != "first_order_plus_secant_plus_qp_inner_jacobian"
                or ev["pose_factorization"] != "single_se3_xi_twist"
                or ev["gauge_status"] != "GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED"
            ):
                raise C2BandedTrainerError("positive band violates the measured EV stop policy")
            artifact_records = ev["artifact_records"]
            expected_artifacts = {
                "ranked_ev_field",
                "necessity",
                "resize",
                "channel_sensitivity",
                "kkt",
                "inner_jacobian_secant_qp",
                "curvelet_carrier",
                "xi_factorization",
                "gauge_binding",
            }
            if not isinstance(artifact_records, dict) or set(artifact_records) != expected_artifacts:
                raise C2BandedTrainerError("positive band EV artifact records mismatch")
            for name, record in artifact_records.items():
                if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                    raise C2BandedTrainerError(f"positive band EV artifact record fields mismatch: {name}")
                artifact_path = (manifest_path.parent / record["path"]).resolve(strict=True)
                try:
                    artifact_path.relative_to(manifest_path.parent)
                except ValueError as exc:
                    raise C2BandedTrainerError(f"EV artifact must remain beneath manifest: {name}") from exc
                expected_sha = _require_sha256(record["sha256"], f"band.custody.ev_selection.{name}.sha256")
                if sha256_file(artifact_path) != expected_sha:
                    raise C2BandedTrainerError(f"positive band EV artifact custody mismatch: {name}")
            expected_laws = [
                "frozen_scorer_fisher_curvature_margin_colocation_v1",
                "fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
                "realization_necessity_preimage_per_stratum_v1",
                "resize_exploit_flip_fix_frontier_v1",
                "segnet_head_rank4_linear_flipdist_v1",
                "posenet_luma_chroma_sensitivity_asymmetry_v1",
                "flip_margin_step_law_v1",
                "instant_projected_input_adjoint_v1",
                "shearlet_nterm_upper_bounds_task_rate_v1",
                "curvelet_directional_basis_dseg_reduction_v1",
                "cgauge_curvelet_parabolic_bank_v1",
                "scorer_obligation_matrix_factorization_v1",
                "lane_band_ego_factorization_source_reparam_v1",
                "witness_measured_reverse_waterfill_v1",
                "meta_lagrangian_dual_solver_per_axis_kkt_residual_v1",
                "cgauge_master_action_v1",
            ]
            if ev["law_refs"] != expected_laws:
                raise C2BandedTrainerError("positive band EV law references drifted")
            if radii is None or radii.dtype not in (np.float32, np.float64):
                raise C2BandedTrainerError("positive radii must be a float32/float64 npy store")
            if tuple(radii.shape) != (LOGICAL_PAIR_COUNT, *PLANE_SHAPE):
                raise C2BandedTrainerError("positive radii geometry mismatch")
            # A positive isotropic artifact is structurally forbidden.  This is
            # intentionally stricter than merely requiring a 3-channel shape.
            any_anisotropic = False
            selected_pixels = 0
            for start in range(0, LOGICAL_PAIR_COUNT, 4):
                sample = np.asarray(radii[start : start + 4])
                if not np.isfinite(sample).all() or np.any(sample < 0):
                    raise C2BandedTrainerError("positive radii must be finite and nonnegative")
                any_anisotropic = any_anisotropic or bool(
                    np.any(sample[..., 0] != sample[..., 1]) or np.any(sample[..., 1] != sample[..., 2])
                )
                selected = np.any(sample < np.float32(INACTIVE_BAND_RADIUS), axis=-1)
                if np.any(sample[~selected] != np.float32(INACTIVE_BAND_RADIUS)):
                    raise C2BandedTrainerError("unselected EV pixels must use the sealed inactive radius")
                max_radius = np.float32(config["max_rgb_radius"])
                if np.any((sample[selected] > max_radius) & (sample[selected] != INACTIVE_BAND_RADIUS)):
                    raise C2BandedTrainerError("selected EV radii exceed the custodied RGB-radius cap")
                selected_pixels += int(np.count_nonzero(selected))
            if not any_anisotropic:
                raise C2BandedTrainerError("positive isotropic bands are forbidden")
            if selected_pixels != ev["selected_pixel_count"]:
                raise C2BandedTrainerError("EV selected-pixel count disagrees with streamed radii")
        else:
            if radii is not None:
                raise C2BandedTrainerError("zero-radius control must not carry a radii array")
            if custody != {
                "derivation": "derive_margin_rgb_band",
                "scale": 0.0,
                "label": "zero_radius_control",
            }:
                raise C2BandedTrainerError("zero-radius control custody mismatch")
        return cls(
            manifest_path=manifest_path,
            manifest_sha256=sha256_bytes(raw),
            mode=doc["mode"],
            pair_count=doc["pair_count"],
            source_planes=source,
            radii=radii,
            source_sha256=source_sha,
            custody=json.loads(canonical_json(custody)),
        )

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source = np.asarray(self.source_planes[indices], dtype=np.float32)
        if self.radii is None:
            radii = np.zeros(source.shape, dtype=np.float32)
        else:
            radii = np.asarray(self.radii[indices], dtype=np.float32)
            if not np.isfinite(radii).all() or np.any(radii < 0):
                raise C2BandedTrainerError("band radii became nonfinite or negative while streaming")
        return source, radii


class PlaneBatchSource(Protocol):
    """Bounded streaming interface used by the trainer and smoke harness."""

    pair_count: int
    base_sha256: str
    source_sha256: str
    band_sha256: str
    band_mode: str

    def fetch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return base/source/radii float32 batches at exact scorer geometry."""


@dataclass(slots=True)
class MemmapPlaneBatchSource:
    base_planes: np.memmap
    band: BandArtifact
    base_planes_sha256: str
    base_sha256: str
    pair_count: int = LOGICAL_PAIR_COUNT

    def __post_init__(self) -> None:
        if self.base_planes.dtype != np.uint8 or tuple(self.base_planes.shape) != (
            LOGICAL_PAIR_COUNT,
            *PLANE_SHAPE,
        ):
            raise C2BandedTrainerError("base planes must be uint8 [600,2,384,512,3]")
        if self.base_planes.flags.writeable:
            raise C2BandedTrainerError("base plane store must be read-only")
        _require_sha256(self.base_planes_sha256, "base_planes_sha256")
        _require_sha256(self.base_sha256, "base_archive_sha256")

    @property
    def source_sha256(self) -> str:
        return self.band.source_sha256

    @property
    def band_sha256(self) -> str:
        return self.band.manifest_sha256

    @property
    def band_mode(self) -> str:
        return self.band.mode

    def fetch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.pair_count):
            raise C2BandedTrainerError("streamed pair indices are invalid")
        base = np.asarray(self.base_planes[indices], dtype=np.float32)
        source, radii = self.band.batch(indices)
        return base, source, radii


@dataclass(frozen=True, slots=True)
class StagePlan:
    name: str
    epochs: int
    learning_rate: float
    rate_weight: float

    def __post_init__(self) -> None:
        if self.name not in ALLOWED_STAGES:
            raise C2BandedTrainerError(f"unknown stage: {self.name}")
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs < 1:
            raise C2BandedTrainerError("stage epochs must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise C2BandedTrainerError("stage learning_rate must be positive and finite")
        if not math.isfinite(self.rate_weight) or self.rate_weight < 0:
            raise C2BandedTrainerError("stage rate_weight must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "rate_weight": self.rate_weight,
        }


DEFAULT_STAGE_PLAN: Final = (
    StagePlan("warmup", 1, 2e-3, 1e-6),
    StagePlan("band_fit", 1, 1e-3, 1e-5),
    StagePlan("rate_polish", 1, 2e-4, 1e-3),
)


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    policy: IntegerPlaneEmitterPolicy
    base_archive_sha256: str
    base_decoder_sha256: str
    source_sha256: str
    band_sha256: str
    band_mode: str
    output_dir: Path
    run_id: str
    carrier_binding: C2R1B4CurveletBinding | None = None
    seed: int = 20260719
    pair_batch_size: int = 2
    smoke_pair_cap: int = 0
    checkpoint_every_steps: int = 50
    ema_decay: float = 0.997
    stages: tuple[StagePlan, ...] = DEFAULT_STAGE_PLAN
    pair_count: int = LOGICAL_PAIR_COUNT

    def __post_init__(self) -> None:
        if self.policy.mode is not PolicyMode.BANDED_TRAINING:
            raise C2BandedTrainerError("trainer requires an active BANDED_TRAINING policy")
        for field in (
            "base_archive_sha256",
            "base_decoder_sha256",
            "source_sha256",
            "band_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.band_mode not in {"positive_anisotropic", "zero_radius_control"}:
            raise C2BandedTrainerError("trainer band_mode is unknown")
        curvelet_active = self.policy.basis is BasisMode.R1B4_WINDOWED_CURVELET
        if curvelet_active != (self.carrier_binding is not None):
            raise C2BandedTrainerError(
                "r1b4_windowed_curvelet policy and carrier binding must be present together"
            )
        if self.carrier_binding is not None and self.carrier_binding.band_manifest_sha256 != self.band_sha256:
            raise C2BandedTrainerError("carrier binding and trainer band manifest differ")
        if self.pair_count != LOGICAL_PAIR_COUNT:
            raise C2BandedTrainerError("trainer logical state is sealed to 600 pairs")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**64:
            raise C2BandedTrainerError("seed must be an integer in [0,2**64)")
        if not 1 <= self.pair_batch_size <= 32:
            raise C2BandedTrainerError("pair_batch_size must be in [1,32]")
        if self.smoke_pair_cap != 0 and not 2 <= self.smoke_pair_cap <= 24:
            raise C2BandedTrainerError("smoke_pair_cap must be 0 or in [2,24]")
        if self.smoke_pair_cap and self.pair_batch_size > self.smoke_pair_cap:
            raise C2BandedTrainerError("pair_batch_size exceeds smoke_pair_cap")
        if self.checkpoint_every_steps < 1:
            raise C2BandedTrainerError("checkpoint_every_steps must be positive")
        if not 0.0 <= self.ema_decay < 1.0 or not math.isfinite(self.ema_decay):
            raise C2BandedTrainerError("ema_decay must be finite in [0,1)")
        if tuple(stage.name for stage in self.stages) != ALLOWED_STAGES:
            raise C2BandedTrainerError("stage plan must be warmup, band_fit, rate_polish in order")
        if not self.run_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in self.run_id
        ):
            raise C2BandedTrainerError("run_id is not filename-safe")

    def identity(self) -> dict[str, Any]:
        contract = self.policy.compile_contract()
        return {
            "schema": TRAINER_CONFIG_SCHEMA,
            "policy_sha256": contract["policy_sha256"],
            "capacity_signature": contract["capacity_signature"],
            "basis": self.policy.basis.value,
            "ste": self.policy.ste.value,
            "base_archive_sha256": self.base_archive_sha256,
            "base_decoder_sha256": self.base_decoder_sha256,
            "source_sha256": self.source_sha256,
            "band_sha256": self.band_sha256,
            "band_mode": self.band_mode,
            "carrier_binding": (
                {
                    "schema": "c2_r1b4_curvelet_carrier_binding.v1",
                    "path": str(self.carrier_binding.manifest_path),
                    "sha256": self.carrier_binding.manifest_sha256,
                    "topology_sha256": self.carrier_binding.topology_sha256,
                    "receiver_schema": "r1b4_section_receiver.v1",
                    "archive_section": "boundary_coordinate.bgj",
                }
                if self.carrier_binding is not None
                else None
            ),
            "seed": self.seed,
            "pair_count": self.pair_count,
            "pair_batch_size": self.pair_batch_size,
            "smoke_pair_cap": self.smoke_pair_cap,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "ema_decay": self.ema_decay,
            "stages": [stage.to_dict() for stage in self.stages],
            "trainable_arrays": ["pair_plane_codes", "shared_rgb_head"],
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json(self.identity()))


def _tensor_payload(array: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(value).all():
        raise C2BandedTrainerError("checkpoint tensor contains nonfinite values")
    return {"dtype": "float32", "shape": list(value.shape), "data": value.reshape(-1).tolist()}


def _tensor_from_payload(payload: Mapping[str, Any]) -> np.ndarray:
    if set(payload) != {"dtype", "shape", "data"} or payload["dtype"] != "float32":
        raise C2BandedTrainerError("checkpoint tensor payload is invalid")
    out = np.asarray(payload["data"], dtype=np.float32).reshape(tuple(payload["shape"]))
    if not np.isfinite(out).all():
        raise C2BandedTrainerError("checkpoint tensor payload is nonfinite")
    return out


def _residual_payload(codes: np.ndarray, head: np.ndarray) -> dict[str, Any]:
    return {"pair_plane_codes": _tensor_payload(codes), "shared_rgb_head": _tensor_payload(head)}


@dataclass(slots=True)
class _AdamState:
    step: int
    code_m: np.ndarray
    code_v: np.ndarray
    head_m: np.ndarray
    head_v: np.ndarray

    @classmethod
    def fresh(cls, codes: np.ndarray, head: np.ndarray) -> _AdamState:
        return cls(0, np.zeros_like(codes), np.zeros_like(codes), np.zeros_like(head), np.zeros_like(head))

    def payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "algorithm": "deterministic_adam_fp32_v1",
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "code_m": _tensor_payload(self.code_m),
            "code_v": _tensor_payload(self.code_v),
            "head_m": _tensor_payload(self.head_m),
            "head_v": _tensor_payload(self.head_v),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> _AdamState:
        if payload.get("algorithm") != "deterministic_adam_fp32_v1":
            raise C2BandedTrainerError("checkpoint optimizer algorithm mismatch")
        if payload.get("beta1") != 0.9 or payload.get("beta2") != 0.999 or payload.get("epsilon") != 1e-8:
            raise C2BandedTrainerError("checkpoint optimizer constants mismatch")
        return cls(
            int(payload["step"]),
            _tensor_from_payload(payload["code_m"]),
            _tensor_from_payload(payload["code_v"]),
            _tensor_from_payload(payload["head_m"]),
            _tensor_from_payload(payload["head_v"]),
        )

    def update(
        self,
        codes: np.ndarray,
        head: np.ndarray,
        code_grad: np.ndarray,
        head_grad: np.ndarray,
        *,
        learning_rate: float,
    ) -> None:
        self.step += 1
        b1, b2 = np.float32(0.9), np.float32(0.999)
        one = np.float32(1.0)
        self.code_m[:] = b1 * self.code_m + (one - b1) * code_grad
        self.code_v[:] = b2 * self.code_v + (one - b2) * np.square(code_grad, dtype=np.float32)
        self.head_m[:] = b1 * self.head_m + (one - b1) * head_grad
        self.head_v[:] = b2 * self.head_v + (one - b2) * np.square(head_grad, dtype=np.float32)
        b1_correction = np.float32(1.0 - 0.9**self.step)
        b2_correction = np.float32(1.0 - 0.999**self.step)
        code_hat = self.code_m / b1_correction
        code_v_hat = self.code_v / b2_correction
        head_hat = self.head_m / b1_correction
        head_v_hat = self.head_v / b2_correction
        codes[:] = codes - np.float32(learning_rate) * code_hat / (
            np.sqrt(code_v_hat, dtype=np.float32) + np.float32(1e-8)
        )
        head[:] = head - np.float32(learning_rate) * head_hat / (
            np.sqrt(head_v_hat, dtype=np.float32) + np.float32(1e-8)
        )


@dataclass(slots=True)
class TrainingState:
    codes: np.ndarray
    head: np.ndarray
    ema_codes: np.ndarray
    ema_head: np.ndarray
    optimizer: _AdamState
    stage_index: int = 0
    stage_epoch: int = 0
    next_pair: int = 0
    global_step: int = 0


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _pair_order(seed: int, stage_index: int, stage_epoch: int, pair_count: int) -> np.ndarray:
    mixed = (seed ^ ((stage_index + 1) * 0x9E3779B97F4A7C15) ^ ((stage_epoch + 1) * 0xD1B54A32D192ED03)) & (2**64 - 1)
    return np.random.Generator(np.random.PCG64(mixed)).permutation(pair_count).astype(np.int64)


def _config_pair_order(config: TrainerConfig, stage_index: int, stage_epoch: int) -> np.ndarray:
    count = config.smoke_pair_cap or config.pair_count
    return _pair_order(config.seed, stage_index, stage_epoch, count)


def _coordinate_basis(config: TrainerConfig) -> np.ndarray:
    if config.carrier_binding is not None:
        return config.carrier_binding.coordinate_basis()
    return deterministic_coordinate_basis(config.policy.residual_width)


def _structured_state(config: TrainerConfig, base: np.ndarray) -> StructuredEmitterState:
    if config.carrier_binding is not None:
        return config.carrier_binding.structured_state(base)
    return StructuredEmitterState.from_base(base, residual_width=config.policy.residual_width)


def _capacity_sha(config: TrainerConfig, codes: np.ndarray, head: np.ndarray) -> str:
    basis = _coordinate_basis(config)
    code_parameters = int(codes.size)
    head_parameters = int(head.size)
    return CapacitySignature(
        pair_count=int(codes.shape[0]),
        plane_count=PLANE_COUNT,
        residual_width=int(codes.shape[-1]),
        code_parameters=code_parameters,
        head_parameters=head_parameters,
        total_parameters=code_parameters + head_parameters,
        topology_sha256=_array_sha256(basis),
    ).sha256


def _state_from_fresh(config: TrainerConfig) -> TrainingState:
    # Fresh initialization does not allocate a full dense base: only the 600x2x4
    # codes and shared 4x3 head exist here.
    rng = np.random.Generator(np.random.PCG64(config.seed))
    codes = (
        rng.standard_normal((config.pair_count, PLANE_COUNT, config.policy.residual_width), dtype=np.float32)
        * np.float32(0.01)
    ).astype(np.float32)
    head = (
        rng.standard_normal((config.policy.residual_width, RGB_CHANNELS), dtype=np.float32) * np.float32(0.01)
    ).astype(np.float32)
    if config.carrier_binding is not None:
        codes[:, 0, :] = np.float32(0.0)
    return TrainingState(codes, head, codes.copy(), head.copy(), _AdamState.fresh(codes, head))


def _run_custody(config: TrainerConfig, state: TrainingState) -> dict[str, Any]:
    contract = config.policy.compile_contract()
    return {
        "trainer_config_sha256": config.sha256,
        "policy_sha256": contract["policy_sha256"],
        "base_archive_sha256": config.base_archive_sha256,
        "base_decoder_sha256": config.base_decoder_sha256,
        "source_sha256": config.source_sha256,
        "band_sha256": config.band_sha256,
        "band_mode": config.band_mode,
        "emitter_capacity_sha256": _capacity_sha(config, state.codes, state.head),
    }


def _checkpoint(
    config: TrainerConfig,
    state: TrainingState,
    *,
    stage: StagePlan,
    stage_complete: bool,
) -> IntegerPlaneEmitterStageCheckpoint:
    contract = config.policy.compile_contract()
    order = _config_pair_order(config, state.stage_index, state.stage_epoch)
    return IntegerPlaneEmitterStageCheckpoint(
        policy_contract=contract,
        config_sha256=config.sha256,
        stage_name=stage.name,
        stage_index=state.stage_index,
        epoch=state.stage_epoch,
        global_step=state.global_step,
        next_pair=state.next_pair,
        basis_id=config.policy.basis.value,
        ste_id=STEMode.SATURATION_AWARE_UINT8.value,
        fixed_capacity_signature=contract["capacity_signature"],
        live_residual_parameters=_residual_payload(state.codes, state.head),
        ema_shadow=_residual_payload(state.ema_codes, state.ema_head),
        optimizer_state=state.optimizer.payload(),
        rng_state={
            "scheme": "seed_stage_epoch_pcg64_permutation_v1",
            "seed": config.seed,
            "order_sha256": _array_sha256(order),
            "stage_epoch": state.stage_epoch,
            "stage_complete": stage_complete,
            "run_custody": _run_custody(config, state),
        },
        topology_state_sha256=sha256_bytes(_coordinate_basis(config).tobytes()),
        discrete_state_sha256=sha256_bytes(
            canonical_json(
                {
                    "pair_count": config.pair_count,
                    "batch": config.pair_batch_size,
                    "smoke_pair_cap": config.smoke_pair_cap,
                }
            )
        ),
        event_state_sha256=sha256_bytes(canonical_json([item.to_dict() for item in config.stages])),
        dual_state_sha256=sha256_bytes(canonical_json({"stage": stage.to_dict(), "band_mode": config.band_mode})),
    )


def load_training_state(path: str | Path, config: TrainerConfig) -> TrainingState:
    checkpoint = IntegerPlaneEmitterStageCheckpoint.from_bytes(Path(path).read_bytes())
    contract = config.policy.compile_contract()
    if checkpoint.policy_contract != contract:
        raise C2BandedTrainerError("resume policy drift")
    if checkpoint.config_sha256 != config.sha256:
        raise C2BandedTrainerError("resume trainer config drift")
    expected_custody = {
        "trainer_config_sha256": config.sha256,
        "policy_sha256": contract["policy_sha256"],
        "base_archive_sha256": config.base_archive_sha256,
        "base_decoder_sha256": config.base_decoder_sha256,
        "source_sha256": config.source_sha256,
        "band_sha256": config.band_sha256,
        "band_mode": config.band_mode,
    }
    for field, value in expected_custody.items():
        custody = checkpoint.rng_state.get("run_custody")
        if not isinstance(custody, Mapping) or custody.get(field) != value:
            raise C2BandedTrainerError(f"resume {field} drift")
    live = checkpoint.live_residual_parameters
    ema = checkpoint.ema_shadow
    state = TrainingState(
        _tensor_from_payload(live["pair_plane_codes"]),
        _tensor_from_payload(live["shared_rgb_head"]),
        _tensor_from_payload(ema["pair_plane_codes"]),
        _tensor_from_payload(ema["shared_rgb_head"]),
        _AdamState.from_payload(checkpoint.optimizer_state),
        stage_index=checkpoint.stage_index,
        stage_epoch=int(checkpoint.rng_state["stage_epoch"]),
        next_pair=checkpoint.next_pair,
        global_step=checkpoint.global_step,
    )
    custody = checkpoint.rng_state["run_custody"]
    if custody["emitter_capacity_sha256"] != _capacity_sha(config, state.codes, state.head):
        raise C2BandedTrainerError("resume emitter capacity drift")
    if config.carrier_binding is not None and np.count_nonzero(state.codes[:, 0]) != 0:
        raise C2BandedTrainerError("resume contains unconsumed frame-0 curvelet codes")
    if bool(checkpoint.rng_state["stage_complete"]):
        state.stage_index += 1
        state.stage_epoch = 0
        state.next_pair = 0
    elif checkpoint.rng_state["order_sha256"] != _array_sha256(
        _config_pair_order(config, state.stage_index, state.stage_epoch)
    ):
        raise C2BandedTrainerError("resume deterministic pair order drift")
    return state


def train_streamed(
    config: TrainerConfig,
    source: PlaneBatchSource,
    *,
    resume_from: str | Path | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Train all 600 codes while materializing only bounded dense pair batches."""

    import torch

    if source.pair_count != config.pair_count:
        raise C2BandedTrainerError("stream source pair count differs from config")
    for field in ("base_sha256", "source_sha256", "band_sha256", "band_mode"):
        expected = getattr(config, "base_archive_sha256" if field == "base_sha256" else field)
        if getattr(source, field) != expected:
            raise C2BandedTrainerError(f"stream source {field} custody drift")
    if config.band_mode == "positive_anisotropic" and source.band_mode != "positive_anisotropic":
        raise C2BandedTrainerError("positive training requires a custodied anisotropic artifact")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    state = load_training_state(resume_from, config) if resume_from else _state_from_fresh(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[str] = []
    telemetry: list[dict[str, Any]] = []
    initial_rss = _peak_rss_bytes()
    starting_step = state.global_step
    if resume_from is None:
        initial = _checkpoint(
            config,
            state,
            stage=config.stages[0],
            stage_complete=False,
        )
        checkpoint_paths.append(str(initial.write_new(config.output_dir, config.run_id)))

    while state.stage_index < len(config.stages):
        stage = config.stages[state.stage_index]
        while state.stage_epoch < stage.epochs:
            order = _config_pair_order(config, state.stage_index, state.stage_epoch)
            pairs_this_epoch = len(order)
            while state.next_pair < pairs_this_epoch:
                end = min(pairs_this_epoch, state.next_pair + config.pair_batch_size)
                indices = order[state.next_pair : end]
                base, source_planes, radii = source.fetch(indices)
                expected_shape = (len(indices), *PLANE_SHAPE)
                if (
                    base.shape != expected_shape
                    or source_planes.shape != expected_shape
                    or radii.shape != expected_shape
                ):
                    raise C2BandedTrainerError("stream source returned wrong real-geometry batch")
                if any(array.dtype != np.float32 for array in (base, source_planes, radii)):
                    raise C2BandedTrainerError("stream source batches must be float32")
                structured = _structured_state(config, base)
                codes_t = torch.tensor(state.codes, dtype=torch.float32, requires_grad=True)
                head_t = torch.tensor(state.head, dtype=torch.float32, requires_grad=True)
                index_t = torch.as_tensor(indices, dtype=torch.long)
                emitted = torch_uint8(structured, codes_t.index_select(0, index_t), head_t)
                source_t = torch.as_tensor(source_planes, dtype=torch.float32)
                radii_t = torch.as_tensor(radii, dtype=torch.float32)
                violation = torch.relu(torch.abs(emitted - source_t) - radii_t)
                band_loss = torch.mean(violation * violation)
                rate_loss = torch.mean(torch.abs(codes_t.index_select(0, index_t))) + torch.mean(torch.abs(head_t))
                loss = band_loss + float(stage.rate_weight) * rate_loss
                loss.backward()
                if codes_t.grad is None or head_t.grad is None:
                    raise C2BandedTrainerError("trainable residual arrays did not receive gradients")
                code_grad = codes_t.grad.detach().cpu().numpy().astype(np.float32, copy=False)
                head_grad = head_t.grad.detach().cpu().numpy().astype(np.float32, copy=False)
                if config.carrier_binding is not None:
                    code_grad[:, 0, :] = np.float32(0.0)
                state.optimizer.update(
                    state.codes,
                    state.head,
                    code_grad,
                    head_grad,
                    learning_rate=stage.learning_rate,
                )
                if config.carrier_binding is not None:
                    state.codes[:, 0, :] = np.float32(0.0)
                    state.optimizer.code_m[:, 0, :] = np.float32(0.0)
                    state.optimizer.code_v[:, 0, :] = np.float32(0.0)
                decay = np.float32(config.ema_decay)
                state.ema_codes[:] = decay * state.ema_codes + (np.float32(1.0) - decay) * state.codes
                state.ema_head[:] = decay * state.ema_head + (np.float32(1.0) - decay) * state.head
                state.global_step += 1
                state.next_pair = end
                telemetry.append(
                    {
                        "stage": stage.name,
                        "stage_epoch": state.stage_epoch,
                        "global_step": state.global_step,
                        "pair_start": int(end - len(indices)),
                        "pair_count": len(indices),
                        "band_loss": float(band_loss.detach().cpu().item()),
                        "rate_pressure": float(rate_loss.detach().cpu().item()),
                    }
                )
                if state.global_step % config.checkpoint_every_steps == 0:
                    ckpt = _checkpoint(config, state, stage=stage, stage_complete=False)
                    checkpoint_paths.append(str(ckpt.write_new(config.output_dir, config.run_id)))
                if stop_after_steps is not None and state.global_step - starting_step >= stop_after_steps:
                    if not checkpoint_paths or not checkpoint_paths[-1].endswith(
                        _checkpoint(config, state, stage=stage, stage_complete=False).filename(config.run_id)
                    ):
                        ckpt = _checkpoint(config, state, stage=stage, stage_complete=False)
                        checkpoint_paths.append(str(ckpt.write_new(config.output_dir, config.run_id)))
                    return _training_receipt(
                        config, state, telemetry, checkpoint_paths, initial_rss, stopped_early=True
                    )
            state.stage_epoch += 1
            state.next_pair = 0
        ckpt = _checkpoint(config, state, stage=stage, stage_complete=True)
        checkpoint_paths.append(str(ckpt.write_new(config.output_dir, config.run_id)))
        state.stage_index += 1
        state.stage_epoch = 0
        state.next_pair = 0
    return _training_receipt(config, state, telemetry, checkpoint_paths, initial_rss, stopped_early=False)


def _training_receipt(
    config: TrainerConfig,
    state: TrainingState,
    telemetry: Sequence[Mapping[str, Any]],
    checkpoint_paths: Sequence[str],
    initial_rss: int,
    *,
    stopped_early: bool,
) -> dict[str, Any]:
    carrier_packet: dict[str, Any] | None = None
    if config.carrier_binding is not None and not stopped_early:
        packet_path = config.output_dir / f"{config.run_id}.ema.{BINDING_BASIS_ID}.bgj"
        try:
            carrier_packet = config.carrier_binding.write_packet_new(
                packet_path,
                state.ema_codes,
                state.ema_head,
            )
        except C2R1B4CurveletBindingError as exc:
            raise C2BandedTrainerError("final EMA carrier packet export failed") from exc
    return {
        "schema": TRAINING_RECEIPT_SCHEMA,
        "authority": "bounded local training evidence; non-score; non-promotion",
        "logical_geometry": [config.pair_count, *PLANE_SHAPE],
        "execution_scope": (
            f"capped_prefix_n{config.smoke_pair_cap}_non_n600_non_score"
            if config.smoke_pair_cap
            else "full_n600_training"
        ),
        "pair_batch_size": config.pair_batch_size,
        "trainable_arrays": ["pair_plane_codes", "shared_rgb_head"],
        "config": config.identity(),
        "config_sha256": config.sha256,
        "global_step": state.global_step,
        "stopped_early": stopped_early,
        "checkpoint_paths": list(checkpoint_paths),
        "last_telemetry": list(telemetry[-12:]),
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_rss_delta_bytes": max(0, _peak_rss_bytes() - initial_rss),
        "ema_authority_default": True,
        "carrier_packet": carrier_packet,
        "launch": False,
        "score_claim": False,
        "pointer_mutation": False,
    }


def storage_preflight(path: str | Path, *, required_free_bytes: int) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if required_free_bytes <= 0:
        raise C2BandedTrainerError("required_free_bytes must be positive")
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    return {
        "path": str(root),
        "required_free_bytes": int(required_free_bytes),
        "free_bytes": int(usage.free),
        "ok": bool(usage.free >= required_free_bytes),
        "refusal_rc": 0 if usage.free >= required_free_bytes else 4,
    }


def _zip_eof_offset(path: Path) -> int:
    """Return the exact end offset declared by the final ZIP EOCD record."""

    raw_size = path.stat().st_size
    tail_size = min(raw_size, 65_557)
    with path.open("rb") as handle:
        handle.seek(raw_size - tail_size)
        tail = handle.read()
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0 or marker + 22 > len(tail):
        raise C2BandedTrainerError("base archive has no valid terminal ZIP EOCD")
    comment_len = int.from_bytes(tail[marker + 20 : marker + 22], "little")
    return raw_size - tail_size + marker + 22 + comment_len


def inflate_worker_count() -> int:
    """Resolve the decoder worker request, preserving an explicit env override."""

    configured = os.environ.get("INFLATE_WORKERS")
    if configured is None:
        return max(1, (os.cpu_count() or 1) - 2)
    try:
        workers = int(configured)
    except ValueError as exc:
        raise C2BandedTrainerError("INFLATE_WORKERS must be a positive integer") from exc
    if workers < 1:
        raise C2BandedTrainerError("INFLATE_WORKERS must be a positive integer")
    return workers


def _project_camera_planes_batched(camera: np.ndarray, out: np.ndarray) -> None:
    """Apply the exact factor-2 scorer projection in bounded NumPy batches."""

    if camera.ndim != 5 or camera.shape[1:] != (
        PLANE_COUNT,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        RGB_CHANNELS,
    ):
        raise C2BandedTrainerError("camera projection input geometry mismatch")
    if out.shape != (camera.shape[0], *PLANE_SHAPE) or out.dtype != np.uint8:
        raise C2BandedTrainerError("scorer projection output geometry mismatch")
    operator = factor2_operator()
    row_indices = np.asarray([support.indices for support in operator.row_supports], dtype=np.intp)
    col_indices = np.asarray([support.indices for support in operator.col_supports], dtype=np.intp)
    row_numerators = np.asarray([support.numerators for support in operator.row_supports], dtype=np.int64)
    col_numerators = np.asarray([support.numerators for support in operator.col_supports], dtype=np.int64)
    denominators = {
        int(row.denominator) * int(col.denominator) for row in operator.row_supports for col in operator.col_supports
    }
    if len(denominators) != 1:
        raise C2BandedTrainerError("factor-2 projection lacks one exact denominator")
    denominator = denominators.pop()
    coefficients = (row_numerators[:, None, :, None] * col_numerators[None, :, None, :])[None, ..., None]
    camera_planes = camera.reshape((-1, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS))
    scorer_planes = out.reshape((-1, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS))
    for start in range(0, len(camera_planes), SCORER_PROJECTION_BATCH_PLANES):
        stop = min(start + SCORER_PROJECTION_BATCH_PLANES, len(camera_planes))
        source = camera_planes[start:stop].astype(np.int64, copy=False)
        blocks = source[:, row_indices[:, None, :, None], col_indices[None, :, None, :], :]
        numerators = np.sum(blocks * coefficients, axis=(3, 4), dtype=np.int64)
        scorer_planes[start:stop] = np.clip(np.rint(numerators.astype(np.float64) / denominator), 0.0, 255.0).astype(
            np.uint8
        )


def _load_shared_base_cache(
    *,
    scorer: Path,
    receipt_path: Path,
    archive_hash: str,
    decoder_hash: str,
    cache_key: str,
) -> tuple[np.memmap, str, dict[str, Any]] | None:
    """Validate every byte of a shared cache hit, or fail closed on debris."""

    if not scorer.exists() and not receipt_path.exists():
        return None
    if not scorer.is_file() or not receipt_path.is_file():
        raise C2BandedTrainerError("shared base materialization cache is incomplete")
    receipt_raw = receipt_path.read_bytes()
    try:
        receipt = json.loads(receipt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2BandedTrainerError("shared base materialization receipt is invalid JSON") from exc
    if canonical_json(receipt) != receipt_raw:
        raise C2BandedTrainerError("shared base materialization receipt is noncanonical")
    expected = {
        "schema": BASE_MATERIALIZATION_SCHEMA,
        "base_archive_sha256": archive_hash,
        "base_decoder_sha256": decoder_hash,
        "scorer_planes": str(scorer),
        "logical_geometry": [LOGICAL_PAIR_COUNT, *PLANE_SHAPE],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise C2BandedTrainerError("shared base materialization custody is stale")
    cache = receipt.get("shared_cache")
    if not isinstance(cache, dict) or cache.get("key") != cache_key or cache.get("certified_rebuildable") is not True:
        raise C2BandedTrainerError("shared base materialization cache custody is stale")
    if receipt.get("scorer_planes_bytes") != scorer.stat().st_size:
        raise C2BandedTrainerError("shared base scorer planes byte count mismatch")
    scorer_hash = sha256_file(scorer)
    if receipt.get("scorer_planes_sha256") != scorer_hash:
        raise C2BandedTrainerError("shared base scorer planes SHA mismatch")
    array = np.load(scorer, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(array, np.memmap)
        or array.dtype != np.uint8
        or array.shape
        != (
            LOGICAL_PAIR_COUNT,
            *PLANE_SHAPE,
        )
    ):
        raise C2BandedTrainerError("shared base scorer planes array geometry mismatch")
    return array, scorer_hash, receipt


def materialize_base_scorer_planes(
    *,
    base_archive: str | Path,
    base_archive_sha256: str,
    base_decoder: str | Path,
    base_decoder_sha256: str,
    scratch_root: str | Path,
    python: str = sys.executable,
    shared_cache_root: str | Path = BASE_SCORER_CACHE_ROOT,
) -> tuple[np.memmap, str, dict[str, Any]]:
    """Decode a counted base packet once, then stream-project it to scorer planes.

    Camera raw and extracted packet bytes are success-only scratch.  They are
    removed only after a canonical reproducibility/certification receipt binds
    their paths, sizes, hashes, exact command, input hashes, and durable scorer
    plane output.
    """

    archive = Path(base_archive).expanduser().resolve(strict=True)
    decoder = Path(base_decoder).expanduser().resolve(strict=True)
    archive_hash = sha256_file(archive)
    decoder_hash = sha256_file(decoder)
    if archive_hash != _require_sha256(base_archive_sha256, "base_archive_sha256"):
        raise C2BandedTrainerError("base archive SHA custody mismatch")
    if decoder_hash != _require_sha256(base_decoder_sha256, "base_decoder_sha256"):
        raise C2BandedTrainerError("base decoder SHA custody mismatch")
    if _zip_eof_offset(archive) != archive.stat().st_size:
        raise C2BandedTrainerError("base archive carries trailing bytes")
    root = Path(scratch_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    packet = root / "base_packet_0.bin"
    raw = root / "base_camera_frames.raw"
    local_scorer_partial = root / "base_scorer_planes.npy.partial"
    cache_root = Path(shared_cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = f"{archive_hash[:16]}_{decoder_hash[:16]}"
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    scorer = cache_dir / "base_scorer_planes.npy"
    scorer_partial = cache_dir / "base_scorer_planes.npy.partial"
    receipt_path = cache_dir / "base_scorer_planes.materialization.json"
    receipt_partial = cache_dir / "base_scorer_planes.materialization.json.partial"
    lock_path = cache_dir / "materialization.lock"

    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        cached = _load_shared_base_cache(
            scorer=scorer,
            receipt_path=receipt_path,
            archive_hash=archive_hash,
            decoder_hash=decoder_hash,
            cache_key=cache_key,
        )
        if cached is not None:
            return cached

        for target in (packet, raw, local_scorer_partial):
            if target.exists():
                raise C2BandedTrainerError(f"uncertified stale base scratch requires review: {target}")
        for target in (scorer_partial, receipt_partial):
            if target.exists():
                raise C2BandedTrainerError(f"uncertified stale shared cache requires review: {target}")
        with zipfile.ZipFile(archive, "r") as zf:
            infos = zf.infolist()
            if len(infos) != 1 or infos[0].filename != "0.bin" or infos[0].is_dir():
                raise C2BandedTrainerError("base archive must contain exactly the counted 0.bin section")
            if infos[0].flag_bits & 0x1:
                raise C2BandedTrainerError("encrypted base packets are forbidden")
            with zf.open(infos[0], "r") as source_handle, packet.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=8 << 20)
                target_handle.flush()
                os.fsync(target_handle.fileno())
        workers = inflate_worker_count()
        command = [str(python), str(decoder), str(packet), str(raw)]
        env = os.environ.copy()
        env.update(
            {
                "INFLATE_MAX_PAIRS": str(LOGICAL_PAIR_COUNT),
                "INFLATE_WORKERS": str(workers),
            }
        )
        proc = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise C2BandedTrainerError(f"base decoder failed rc={proc.returncode}: {proc.stderr[-1000:]}")
        expected_raw_bytes = LOGICAL_PAIR_COUNT * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS
        if not raw.exists() or raw.stat().st_size != expected_raw_bytes:
            raise C2BandedTrainerError("base decoder raw output size mismatch")
        camera = np.memmap(
            raw,
            dtype=np.uint8,
            mode="r",
            shape=(LOGICAL_PAIR_COUNT, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS),
        )
        out = np.lib.format.open_memmap(
            scorer_partial,
            mode="w+",
            dtype=np.uint8,
            shape=(LOGICAL_PAIR_COUNT, *PLANE_SHAPE),
        )
        _project_camera_planes_batched(camera, out)
        out.flush()
        del out, camera
        with scorer_partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(scorer_partial, scorer)
        scorer_hash = sha256_file(scorer)
        packet_hash = sha256_file(packet)
        raw_hash = sha256_file(raw)
        receipt = {
            "schema": BASE_MATERIALIZATION_SCHEMA,
            "base_archive": str(archive),
            "base_archive_bytes": archive.stat().st_size,
            "base_archive_sha256": archive_hash,
            "base_decoder": str(decoder),
            "base_decoder_sha256": decoder_hash,
            "command": command,
            "environment": {
                "INFLATE_MAX_PAIRS": str(LOGICAL_PAIR_COUNT),
                "INFLATE_WORKERS": str(workers),
            },
            "packet_scratch": {"path": str(packet), "bytes": packet.stat().st_size, "sha256": packet_hash},
            "camera_raw_scratch": {"path": str(raw), "bytes": raw.stat().st_size, "sha256": raw_hash},
            "scorer_planes": str(scorer),
            "scorer_planes_bytes": scorer.stat().st_size,
            "scorer_planes_sha256": scorer_hash,
            "logical_geometry": [LOGICAL_PAIR_COUNT, *PLANE_SHAPE],
            "projection": {
                "method": "exact_numpy_batched_factor2_numerators",
                "batch_planes": SCORER_PROJECTION_BATCH_PLANES,
            },
            "shared_cache": {
                "root": str(cache_root),
                "key": cache_key,
                "certified_rebuildable": True,
                "atomic_population": True,
                "source_identity": "archive_sha256_plus_decoder_sha256",
            },
            "cleanup": {
                "success_only": True,
                "certified_rebuildable": [str(packet), str(raw)],
                "reason": "deterministically regenerated from hash-bound counted base archive and decoder",
            },
        }
        with receipt_partial.open("xb") as handle:
            handle.write(canonical_json(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(receipt_partial, receipt_path)
        packet.unlink()
        raw.unlink()
        cached = _load_shared_base_cache(
            scorer=scorer,
            receipt_path=receipt_path,
            archive_hash=archive_hash,
            decoder_hash=decoder_hash,
            cache_key=cache_key,
        )
        if cached is None:  # pragma: no cover - guarded by the atomic population above
            raise C2BandedTrainerError("shared base materialization vanished after population")
        return cached


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-plane-emitter-mode", required=True, choices=[PolicyMode.BANDED_TRAINING.value])
    parser.add_argument("--integer-plane-emitter-basis", required=True, choices=[mode.value for mode in BasisMode])
    parser.add_argument("--integer-plane-emitter-policy-sha256", required=True)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--base-decoder", type=Path, required=True)
    parser.add_argument("--base-archive-sha256", required=True)
    parser.add_argument("--base-decoder-sha256", required=True)
    parser.add_argument("--band-manifest", type=Path, required=True)
    parser.add_argument("--r1b4-carrier-binding", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--required-free-bytes", type=int, default=4_000_000_000)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--pair-batch-size", type=int, default=2)
    parser.add_argument("--smoke-pair-cap", type=int, default=0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--stage-plan-json", required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def policy_from_args(args: argparse.Namespace) -> IntegerPlaneEmitterPolicy:
    policy = IntegerPlaneEmitterPolicy(
        basis=BasisMode(args.integer_plane_emitter_basis),
        mode=PolicyMode(args.integer_plane_emitter_mode),
    )
    if policy.compile_contract()["policy_sha256"] != args.integer_plane_emitter_policy_sha256:
        raise C2BandedTrainerError("typed policy hash mismatch")
    return policy


def parse_stage_plan(raw: str) -> tuple[StagePlan, ...]:
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise C2BandedTrainerError("stage plan is not JSON") from exc
    if not isinstance(rows, list):
        raise C2BandedTrainerError("stage plan must be a JSON list")
    try:
        return tuple(StagePlan(**row) for row in rows)
    except (TypeError, KeyError) as exc:
        raise C2BandedTrainerError("stage plan row fields mismatch") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preflight = storage_preflight(args.scratch_root, required_free_bytes=args.required_free_bytes)
    if not preflight["ok"]:
        print(canonical_json({"storage_preflight": preflight}).decode("ascii"), file=sys.stderr)
        return 4
    try:
        policy = policy_from_args(args)
        band = BandArtifact.load(args.band_manifest)
        carrier_binding = (
            C2R1B4CurveletBinding.load(args.r1b4_carrier_binding)
            if args.r1b4_carrier_binding is not None
            else None
        )
        base, base_planes_sha, materialization = materialize_base_scorer_planes(
            base_archive=args.base_archive,
            base_archive_sha256=args.base_archive_sha256,
            base_decoder=args.base_decoder,
            base_decoder_sha256=args.base_decoder_sha256,
            scratch_root=args.scratch_root,
        )
        source = MemmapPlaneBatchSource(
            base,
            band,
            base_planes_sha,
            args.base_archive_sha256,
        )
        config = TrainerConfig(
            policy=policy,
            base_archive_sha256=args.base_archive_sha256,
            base_decoder_sha256=args.base_decoder_sha256,
            source_sha256=band.source_sha256,
            band_sha256=band.manifest_sha256,
            band_mode=band.mode,
            output_dir=args.output_dir,
            run_id=args.run_id,
            carrier_binding=carrier_binding,
            seed=args.seed,
            pair_batch_size=args.pair_batch_size,
            smoke_pair_cap=args.smoke_pair_cap,
            checkpoint_every_steps=args.checkpoint_every_steps,
            ema_decay=args.ema_decay,
            stages=parse_stage_plan(args.stage_plan_json),
        )
        receipt = train_streamed(
            config,
            source,
            resume_from=args.resume_from,
            stop_after_steps=args.stop_after_steps,
        )
        receipt["storage_preflight"] = preflight
        receipt["base_materialization"] = materialization
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(canonical_json(receipt))
    except (C2BandedTrainerError, OSError, ValueError) as exc:
        print(f"C2 trainer custody/config refusal: {exc}", file=sys.stderr)
        return 6
    return 0


__all__ = [
    "ALLOWED_STAGES",
    "BAND_SCHEMA",
    "DEFAULT_STAGE_PLAN",
    "LOGICAL_PAIR_COUNT",
    "PLANE_SHAPE",
    "BandArtifact",
    "C2BandedTrainerError",
    "MemmapPlaneBatchSource",
    "PlaneBatchSource",
    "StagePlan",
    "TrainerConfig",
    "build_parser",
    "canonical_json",
    "inflate_worker_count",
    "load_training_state",
    "main",
    "materialize_base_scorer_planes",
    "parse_stage_plan",
    "policy_from_args",
    "sha256_file",
    "storage_preflight",
    "train_streamed",
]
