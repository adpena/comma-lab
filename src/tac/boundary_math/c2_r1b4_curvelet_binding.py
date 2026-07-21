# SPDX-License-Identifier: MIT
"""C2 trainer binding to the counted R1b4 windowed-curvelet receiver.

The binding is deliberately narrow.  It regenerates the exact generic frame
used by ``boundary_coordinate.bgj`` at train time, then exports the EMA
frame-1 residual as that counted packet.  Frame 0 remains the separately
factorized xi actuator and is therefore required to have exactly-zero C2
curvelet codes; no unconsumed latent bytes are permitted.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.boundary_math.integer_plane_emitter import (
    PLANE_COUNT,
    RGB_CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    StructuredEmitterState,
)
from tac.optimization.boundary_coordinate_joint_solve import (
    BoundaryCoordinatePacket,
    FrameFamily,
    decode_boundary_packet,
    encode_boundary_packet,
    selected_frame_features,
)

BINDING_SCHEMA: Final = "c2_r1b4_curvelet_carrier_binding.v1"
BINDING_BASIS_ID: Final = "r1b4_windowed_curvelet"
R1B4_RECEIVER_SCHEMA: Final = "r1b4_section_receiver.v1"
R1B4_PACKET_SCHEMA: Final = "boundary_coordinate_packet.v1"
R1B4_ARCHIVE_SECTION: Final = "boundary_coordinate.bgj"
LOGICAL_PAIR_COUNT: Final = 600
RESIDUAL_WIDTH: Final = 4


class C2R1B4CurveletBindingError(ValueError):
    """Malformed binding, topology drift, or unconsumed carrier state."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise C2R1B4CurveletBindingError("binding is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise C2R1B4CurveletBindingError(f"{field} must be a non-placeholder lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class C2R1B4CurveletBinding:
    """Hash-bound topology shared by the trainer and real R1b4 receiver."""

    manifest_path: Path
    manifest_sha256: str
    family: FrameFamily
    frame_config: Mapping[str, Any]
    atom_indices: np.ndarray
    band_manifest_path: Path
    band_manifest_sha256: str
    predecessor_carrier_record_sha256: str
    selected_pixel_count: int
    dead_pixel_count: int

    @classmethod
    def load(cls, path: str | Path) -> C2R1B4CurveletBinding:
        manifest_path = Path(path).expanduser().resolve(strict=True)
        raw = manifest_path.read_bytes()
        encoded = raw[:-1] if raw.endswith(b"\n") else raw
        try:
            doc = json.loads(encoded.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise C2R1B4CurveletBindingError("carrier binding must be ASCII JSON") from exc
        if not isinstance(doc, dict) or _canonical_json(doc) != encoded:
            raise C2R1B4CurveletBindingError("carrier binding must be canonical JSON")
        required = {
            "schema",
            "basis_id",
            "family",
            "frame_config",
            "atom_indices",
            "logical_pair_count",
            "scorer_geometry",
            "receiver",
            "band_manifest",
            "quantization_strata",
            "authority",
            "verdict_scope",
        }
        if set(doc) != required:
            raise C2R1B4CurveletBindingError("carrier binding fields mismatch")
        receiver = doc["receiver"]
        if not isinstance(receiver, dict) or receiver != {
            "archive_section": R1B4_ARCHIVE_SECTION,
            "byte_accounting": "counted_zip_member_actual_bytes",
            "factor2_consumption": "exact_uint8_scorer_target_to_camera_preimage",
            "packet_schema": R1B4_PACKET_SCHEMA,
            "receiver_schema": R1B4_RECEIVER_SCHEMA,
            "semantic_frame": 1,
            "unknown_or_trailing_sections": "refuse",
        }:
            raise C2R1B4CurveletBindingError("carrier receiver contract mismatch")
        if (
            doc["schema"] != BINDING_SCHEMA
            or doc["basis_id"] != BINDING_BASIS_ID
            or doc["family"] != FrameFamily.WINDOWED_CURVELET.value
            or doc["logical_pair_count"] != LOGICAL_PAIR_COUNT
            or doc["scorer_geometry"] != [SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS]
            or doc["authority"] != "build_and_local_verify_only_no_launch_no_score"
        ):
            raise C2R1B4CurveletBindingError("carrier binding sealed values mismatch")
        indices = np.asarray(doc["atom_indices"])
        if (
            indices.dtype.kind not in ("i", "u")
            or indices.shape != (RESIDUAL_WIDTH,)
            or len({int(value) for value in indices}) != RESIDUAL_WIDTH
            or np.any(indices < 0)
        ):
            raise C2R1B4CurveletBindingError("carrier atom indices must be four unique nonnegative integers")
        band_record = doc["band_manifest"]
        if not isinstance(band_record, dict) or set(band_record) != {
            "path",
            "sha256",
            "predecessor_carrier_record_sha256",
        }:
            raise C2R1B4CurveletBindingError("band-manifest binding fields mismatch")
        band_path = Path(band_record["path"]).expanduser().resolve(strict=True)
        band_sha = _require_sha256(band_record["sha256"], "band_manifest.sha256")
        if _sha256_file(band_path) != band_sha:
            raise C2R1B4CurveletBindingError("band-manifest SHA-256 drift")
        try:
            band_doc = json.loads(band_path.read_bytes().decode("ascii"))
            predecessor_record = band_doc["custody"]["ev_selection"]["artifact_records"]["curvelet_carrier"]
            predecessor_path = (band_path.parent / predecessor_record["path"]).resolve(strict=True)
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
            raise C2R1B4CurveletBindingError("band manifest lacks predecessor carrier custody") from exc
        predecessor_sha = _require_sha256(
            band_record["predecessor_carrier_record_sha256"],
            "band_manifest.predecessor_carrier_record_sha256",
        )
        if predecessor_record.get("sha256") != predecessor_sha or _sha256_file(predecessor_path) != predecessor_sha:
            raise C2R1B4CurveletBindingError("predecessor carrier record SHA-256 drift")
        try:
            predecessor_doc = json.loads(predecessor_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise C2R1B4CurveletBindingError("predecessor carrier record is invalid") from exc
        if predecessor_doc.get("c2_banded_trainer_binding") != "ABSENT":
            raise C2R1B4CurveletBindingError("binding must supersede the exact recorded ABSENT state")
        strata = doc["quantization_strata"]
        if not isinstance(strata, dict) or set(strata) != {
            "candidate_pixels",
            "realizable_pixels",
            "substep_dead_pixels",
            "carrier_spend_policy",
        }:
            raise C2R1B4CurveletBindingError("quantization-strata fields mismatch")
        candidate = strata["candidate_pixels"]
        selected = strata["realizable_pixels"]
        dead = strata["substep_dead_pixels"]
        if (
            candidate != 38_077
            or selected != 11_453
            or dead != 26_624
            or selected + dead != candidate
            or strata["carrier_spend_policy"]
            != "optimization_realizable_only_dead_zero_weight_shared_bytes_unattributed"
        ):
            raise C2R1B4CurveletBindingError("quantization-strata custody mismatch")
        # Constructing a packet is the canonical validation path for frame config
        # and atom bounds.  It also prevents a manifest-only basis label.
        packet = BoundaryCoordinatePacket(
            family=FrameFamily.WINDOWED_CURVELET,
            frame_config=doc["frame_config"],
            scorer_height=SCORER_HEIGHT,
            scorer_width=SCORER_WIDTH,
            atom_indices=indices.astype("<u4"),
            coefficients=np.zeros((LOGICAL_PAIR_COUNT, RESIDUAL_WIDTH, RGB_CHANNELS), dtype=np.int8),
            scales=np.ones((LOGICAL_PAIR_COUNT,), dtype="<f2"),
        )
        decoded = decode_boundary_packet(encode_boundary_packet(packet))
        if not np.array_equal(decoded.atom_indices, packet.atom_indices):
            raise C2R1B4CurveletBindingError("carrier packet parse-back changed atom indices")
        frozen_indices = np.asarray(packet.atom_indices, dtype="<u4").copy()
        frozen_indices.setflags(write=False)
        return cls(
            manifest_path=manifest_path,
            manifest_sha256=_sha256(raw),
            family=packet.family,
            frame_config=dict(packet.frame_config),
            atom_indices=frozen_indices,
            band_manifest_path=band_path,
            band_manifest_sha256=band_sha,
            predecessor_carrier_record_sha256=predecessor_sha,
            selected_pixel_count=int(selected),
            dead_pixel_count=int(dead),
        )

    def _packet(self, coefficients: np.ndarray, scales: np.ndarray) -> BoundaryCoordinatePacket:
        return BoundaryCoordinatePacket(
            family=self.family,
            frame_config=self.frame_config,
            scorer_height=SCORER_HEIGHT,
            scorer_width=SCORER_WIDTH,
            atom_indices=self.atom_indices,
            coefficients=coefficients,
            scales=scales,
        )

    def coordinate_basis(self) -> np.ndarray:
        """Regenerate the same selected frame columns used by R1b4."""

        packet = self._packet(
            np.zeros((LOGICAL_PAIR_COUNT, RESIDUAL_WIDTH, RGB_CHANNELS), dtype=np.int8),
            np.ones((LOGICAL_PAIR_COUNT,), dtype="<f2"),
        )
        flat = selected_frame_features(packet)
        basis = np.asarray(flat.reshape(SCORER_HEIGHT, SCORER_WIDTH, RESIDUAL_WIDTH), dtype=np.float32)
        if not np.isfinite(basis).all():
            raise C2R1B4CurveletBindingError("receiver curvelet basis is nonfinite")
        basis.setflags(write=False)
        return basis

    @property
    def topology_sha256(self) -> str:
        return _array_sha256(self.coordinate_basis())

    def structured_state(self, base: np.ndarray) -> StructuredEmitterState:
        return StructuredEmitterState(
            base=base,
            coordinate_basis=self.coordinate_basis(),
            topology_id=BINDING_BASIS_ID,
        )

    def export_packet(self, pair_plane_codes: np.ndarray, shared_rgb_head: np.ndarray) -> tuple[bytes, dict[str, Any]]:
        """Quantize EMA state into the exact counted R1b4 boundary packet."""

        codes = np.asarray(pair_plane_codes)
        head = np.asarray(shared_rgb_head)
        if codes.dtype != np.float32 or codes.shape != (LOGICAL_PAIR_COUNT, PLANE_COUNT, RESIDUAL_WIDTH):
            raise C2R1B4CurveletBindingError("pair_plane_codes must be float32 [600,2,4]")
        if head.dtype != np.float32 or head.shape != (RESIDUAL_WIDTH, RGB_CHANNELS):
            raise C2R1B4CurveletBindingError("shared_rgb_head must be float32 [4,3]")
        if not np.isfinite(codes).all() or not np.isfinite(head).all():
            raise C2R1B4CurveletBindingError("carrier parameters must be finite")
        if np.count_nonzero(codes[:, 0]) != 0:
            raise C2R1B4CurveletBindingError("frame-0 curvelet codes are unconsumed by R1b4 and must be exactly zero")
        values = np.multiply(codes[:, 1, :, None], head[None, :, :], dtype=np.float32)
        max_abs = np.max(np.abs(values), axis=(1, 2))
        scale32 = np.where(max_abs > 0.0, max_abs / np.float32(127.0), np.float32(1.0)).astype(np.float32)
        scales = scale32.astype("<f2")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise C2R1B4CurveletBindingError("carrier float16 scales are invalid")
        coefficients = np.clip(
            np.rint(values / scales.astype(np.float32)[:, None, None]),
            -127.0,
            127.0,
        ).astype(np.int8)
        packet = self._packet(coefficients, scales)
        payload = encode_boundary_packet(packet)
        parsed = decode_boundary_packet(payload)
        if encode_boundary_packet(parsed) != payload:
            raise C2R1B4CurveletBindingError("carrier packet parse/re-encode is not byte-identical")
        reconstructed = coefficients.astype(np.float32) * scales.astype(np.float32)[:, None, None]
        error = np.abs(reconstructed - values)
        receipt = {
            "schema": "c2_r1b4_curvelet_packet_export.v1",
            "basis_id": BINDING_BASIS_ID,
            "binding_sha256": self.manifest_sha256,
            "packet_schema": R1B4_PACKET_SCHEMA,
            "receiver_schema": R1B4_RECEIVER_SCHEMA,
            "archive_section": R1B4_ARCHIVE_SECTION,
            "packet_bytes": len(payload),
            "packet_sha256": _sha256(payload),
            "pair_count": LOGICAL_PAIR_COUNT,
            "selected_feature_count": RESIDUAL_WIDTH,
            "nonzero_coefficients": int(np.count_nonzero(coefficients)),
            "max_abs_coefficient_quantization_error": float(np.max(error, initial=0.0)),
            "frame0_codes_nonzero": 0,
            "source_parameter_sha256": {
                "pair_plane_codes": _array_sha256(codes),
                "shared_rgb_head": _array_sha256(head),
            },
            "quantization_strata": {
                "realizable_pixels": self.selected_pixel_count,
                "substep_dead_pixels": self.dead_pixel_count,
                "optimization_weight_on_dead_stratum": 0,
                "shared_carrier_packet_bytes": len(payload),
                "shared_bytes_pixel_attribution": "not_decomposable_before_receiver_effect_measurement",
                "dead_stratum_spatial_effect": "not_measured_no_launch",
            },
            "byte_accounting": "packet_bytes_are_counted_as_boundary_coordinate.bgj_zip_member",
            "receiver_consumed": True,
            "score_claim": False,
        }
        return payload, receipt

    def write_packet_new(
        self,
        path: str | Path,
        pair_plane_codes: np.ndarray,
        shared_rgb_head: np.ndarray,
    ) -> dict[str, Any]:
        """Atomically publish, or byte-identically resume, a counted packet."""

        target = Path(path).expanduser().resolve()
        payload, receipt = self.export_packet(pair_plane_codes, shared_rgb_head)
        if target.exists():
            if target.read_bytes() != payload:
                raise C2R1B4CurveletBindingError(
                    f"existing carrier packet differs from deterministic resume bytes: {target}"
                )
            return {
                **receipt,
                "path": str(target),
                "write_disposition": "reused_byte_identical_existing",
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        if partial.exists():
            raise C2R1B4CurveletBindingError(f"stale carrier packet temporary requires review: {partial}")
        try:
            with partial.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if partial.exists():
                partial.unlink()
        if target.read_bytes() != payload:
            raise C2R1B4CurveletBindingError("published carrier packet bytes drifted")
        return {**receipt, "path": str(target), "write_disposition": "created_new"}


__all__ = [
    "BINDING_BASIS_ID",
    "BINDING_SCHEMA",
    "R1B4_ARCHIVE_SECTION",
    "C2R1B4CurveletBinding",
    "C2R1B4CurveletBindingError",
]
