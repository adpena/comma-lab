# SPDX-License-Identifier: MIT
"""Strict helpers for RG3 obstruction closure and candidate-local PC1 tubes.

The RG3 side harvests already-settled receiver measurements.  It does not turn
an exhaustive no-event result into positive causal coverage.  The PC1 side
defines a source-preserving local receiver:

    C(q; W) = clip_u8(W + P(q; W) - P(0; W))

where ``P`` is the admitted PC1 receiver and ``W`` is the exact candidate
camera pair.  Consequently ``C(0; W) == W`` byte-for-byte while nonzero counted
PC1 coordinates retain their actual receiver and composite-R path.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tac.optimization.ddm_pc1_pose_stream import (
    PACKET_MEMBER,
    PARENT_MEMBER,
    PC1PosePacketV1,
    PC1PoseStreamError,
    parse_pc1_packet,
    receive_pc1_camera_pairs,
    serialize_pc1_packet,
)

SOURCE_LOCAL_MANIFEST_MEMBER = "manifest/pc1_source_local.json"
SOURCE_LOCAL_SCHEMA = "ddm_rg4_pc1_source_local_composition.v1"
EXPECTED_RG3_BLOCKER = "NO_TARGET_BUCKET_EVENT_CHANGED_BY_ANY_COUNTED_RG3_MAGNITUDE_OR_SIGN"
EXPECTED_MISSING_BLOCKS = 25
POSE_DIMS = 6


class DDMRG4Error(ValueError):
    """Raised when source custody or strict obstruction semantics differ."""


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic newline-terminated JSON bytes."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def apply_source_preserving_delta(
    *,
    parent_camera: np.ndarray,
    absolute_candidate: np.ndarray,
    absolute_zero_home: np.ndarray,
) -> np.ndarray:
    """Apply the realized PC1 finite difference to exact parent uint8 bytes."""

    parent = np.asarray(parent_camera)
    candidate = np.asarray(absolute_candidate)
    zero = np.asarray(absolute_zero_home)
    if (
        parent.dtype != np.uint8
        or candidate.dtype != np.uint8
        or zero.dtype != np.uint8
        or parent.shape != candidate.shape
        or parent.shape != zero.shape
        or parent.ndim != 5
        or parent.shape[-1] != 3
    ):
        raise DDMRG4Error("source-local PC1 cameras must share uint8 pair geometry")
    delta = candidate.astype(np.int16) - zero.astype(np.int16)
    result = np.clip(parent.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    if np.array_equal(candidate, zero) and not np.array_equal(result, parent):
        raise DDMRG4Error("zero receiver delta failed exact parent identity")
    return np.ascontiguousarray(result)


def receive_source_local_pc1_camera_pairs(
    *,
    parent_camera: np.ndarray,
    packet: PC1PosePacketV1,
    pair_ids: Sequence[int],
    movable_masks: np.ndarray | None = None,
    absolute_zero_home: np.ndarray | None = None,
    torch_module: Any | None = None,
) -> np.ndarray:
    """Render a PC1 packet as a finite difference around the exact candidate."""

    zero_packet = PC1PosePacketV1(
        active=True,
        pair_count=packet.pair_count,
        xi_scales=packet.xi_scales,
        residual_scale=packet.residual_scale,
        q_xi=np.zeros_like(packet.q_xi),
        q_luma_phase=np.zeros_like(packet.q_luma_phase),
    )
    if absolute_zero_home is None:
        absolute_zero_home = receive_pc1_camera_pairs(
            parent_camera=parent_camera,
            packet=zero_packet,
            pair_ids=pair_ids,
            movable_masks=movable_masks,
            torch_module=torch_module,
        )
    absolute_candidate = receive_pc1_camera_pairs(
        parent_camera=parent_camera,
        packet=packet,
        pair_ids=pair_ids,
        movable_masks=movable_masks,
        torch_module=torch_module,
    )
    return apply_source_preserving_delta(
        parent_camera=parent_camera,
        absolute_candidate=absolute_candidate,
        absolute_zero_home=absolute_zero_home,
    )


def build_source_local_composition_archive(
    *,
    parent_archive: bytes,
    parent_sha256: str,
    packet: PC1PosePacketV1,
) -> bytes:
    """Build the deterministic counted archive for the source-local adapter."""

    if sha256_bytes(parent_archive) != parent_sha256:
        raise DDMRG4Error("source-local parent archive SHA-256 differs")
    packet_bytes = serialize_pc1_packet(packet)
    manifest = {
        "active": packet.active,
        "adapter_equation": "C(q;W)=clip_u8(W+P(q;W)-P(0;W))",
        "free_receiver_code": True,
        "owner": "ddm.rg4.pc1_source_local_delta",
        "packet_member": PACKET_MEMBER,
        "packet_sha256": sha256_bytes(packet_bytes),
        "parent_member": PARENT_MEMBER,
        "parent_sha256": parent_sha256,
        "schema": SOURCE_LOCAL_SCHEMA,
        "source_local_zero_identity": True,
    }
    members = (
        (
            SOURCE_LOCAL_MANIFEST_MEMBER,
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ),
        (PACKET_MEMBER, packet_bytes),
        (PARENT_MEMBER, parent_archive),
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as archive:
        for member_name, member_bytes in members:
            info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, member_bytes)
    return buffer.getvalue()


def parse_source_local_composition_archive(
    payload: bytes,
) -> tuple[bytes, PC1PosePacketV1, dict[str, Any]]:
    """Parse, validate, and byte-reemit a source-local counted composition."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            if archive.namelist() != [
                SOURCE_LOCAL_MANIFEST_MEMBER,
                PACKET_MEMBER,
                PARENT_MEMBER,
            ]:
                raise DDMRG4Error("source-local composition member order differs")
            manifest = json.loads(archive.read(SOURCE_LOCAL_MANIFEST_MEMBER))
            packet_bytes = archive.read(PACKET_MEMBER)
            parent_bytes = archive.read(PARENT_MEMBER)
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise DDMRG4Error("source-local composition archive is invalid") from exc
    if (
        manifest.get("schema") != SOURCE_LOCAL_SCHEMA
        or manifest.get("owner") != "ddm.rg4.pc1_source_local_delta"
        or manifest.get("adapter_equation") != "C(q;W)=clip_u8(W+P(q;W)-P(0;W))"
        or manifest.get("source_local_zero_identity") is not True
        or manifest.get("packet_member") != PACKET_MEMBER
        or manifest.get("parent_member") != PARENT_MEMBER
    ):
        raise DDMRG4Error("source-local composition manifest differs")
    if sha256_bytes(packet_bytes) != manifest.get("packet_sha256"):
        raise DDMRG4Error("source-local packet hash differs")
    if sha256_bytes(parent_bytes) != manifest.get("parent_sha256"):
        raise DDMRG4Error("source-local parent hash differs")
    try:
        packet = parse_pc1_packet(packet_bytes)
    except PC1PoseStreamError as exc:
        raise DDMRG4Error("source-local PC1 packet is invalid") from exc
    rebuilt = build_source_local_composition_archive(
        parent_archive=parent_bytes,
        parent_sha256=manifest["parent_sha256"],
        packet=packet,
    )
    if rebuilt != payload:
        raise DDMRG4Error("source-local composition is not canonical on re-emission")
    return parent_bytes, packet, manifest


def active_tube_report(
    *,
    pose6: np.ndarray,
    centers: np.ndarray,
    low_rank_factors: np.ndarray,
    tube_radius: float,
) -> dict[str, Any]:
    """Decompose the exact Pose quadratic and report bounded dimension activity.

    The landed metric is one six-dimensional quadratic per pair, not six
    independent constraints.  The per-dimension activity rule therefore uses
    an explicitly labeled equal-rank-share diagnostic.  Overall membership is
    determined only by the full per-pair quadratic.
    """

    candidate = np.asarray(pose6, dtype=np.float64)
    center = np.asarray(centers, dtype=np.float64)
    factors = np.asarray(low_rank_factors, dtype=np.float64)
    if (
        candidate.shape != center.shape
        or candidate.ndim != 2
        or len(candidate) == 0
        or candidate.shape[1] != POSE_DIMS
        or factors.shape != (len(candidate), POSE_DIMS, POSE_DIMS)
    ):
        raise DDMRG4Error("active-tube inputs must have n-by-6 metric geometry")
    if (
        not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(factors))
        or not math.isfinite(float(tube_radius))
        or tube_radius <= 0.0
    ):
        raise DDMRG4Error("active-tube inputs must be finite and radius positive")
    projected = np.einsum("nij,nj->ni", factors, candidate - center)
    contributions = np.square(projected)
    pair_quadratic = contributions.sum(axis=1)
    radius_squared = float(tube_radius) ** 2
    equal_rank_share = radius_squared / POSE_DIMS
    mean_by_dimension = contributions.mean(axis=0)
    dimension_rows = []
    for dimension in range(POSE_DIMS):
        contribution = float(mean_by_dimension[dimension])
        slack = equal_rank_share - contribution
        dimension_rows.append(
            {
                "pose_output_dimension": dimension,
                "mean_projected_squared_contribution": contribution,
                "equal_rank_share_budget": equal_rank_share,
                "equal_rank_share_slack": slack,
                "active_under_equal_rank_share_diagnostic": slack <= 0.0,
            }
        )
    inside = pair_quadratic <= radius_squared
    return {
        "metric": "sum_i (L_i @ (pose6_i-center_i))^2",
        "pair_count": len(candidate),
        "tube_radius": float(tube_radius),
        "tube_radius_squared": radius_squared,
        "overall_membership_rule": "ALL_PAIR_QUADRATICS_LE_RADIUS_SQUARED",
        "all_pairs_inside": bool(np.all(inside)),
        "inside_pair_count": int(np.count_nonzero(inside)),
        "outside_pair_count": int(np.count_nonzero(~inside)),
        "inside_pair_fraction": float(np.mean(inside)),
        "mean_pair_quadratic": float(np.mean(pair_quadratic)),
        "max_pair_quadratic": float(np.max(pair_quadratic)),
        "minimum_full_quadratic_slack": float(np.min(radius_squared - pair_quadratic)),
        "dimension_activity_rule": (
            "DIAGNOSTIC_ONLY: equal share of radius^2 across six factor-output "
            "dimensions; does not replace the full quadratic membership rule"
        ),
        "active_dimension_count": sum(bool(row["active_under_equal_rank_share_diagnostic"]) for row in dimension_rows),
        "dimensions": dimension_rows,
    }


def rg3_typed_exclusions(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the 25 exhaustive-measurement obstructions, or fail closed."""

    if summary.get("schema") != "ddm_ms6_receiver_support_resume_summary.v1":
        raise DDMRG4Error("RG3 summary schema differs")
    coverage = summary.get("g3_top24_coverage")
    derivation = summary.get("receiver_coordinate_derivation")
    if not isinstance(coverage, Mapping) or not isinstance(derivation, Mapping):
        raise DDMRG4Error("RG3 coverage/derivation objects are absent")
    missing = coverage.get("missing_blocks")
    residual = derivation.get("residual")
    if (
        coverage.get("coverage_proven") is not False
        or coverage.get("missing_block_count") != EXPECTED_MISSING_BLOCKS
        or not isinstance(missing, list)
        or len(missing) != EXPECTED_MISSING_BLOCKS
        or not isinstance(residual, list)
        or len(residual) != EXPECTED_MISSING_BLOCKS
    ):
        raise DDMRG4Error("RG3 terminal 25-block falsifier geometry differs")
    missing_keys: set[tuple[int, str]] = set()
    for row in missing:
        if not isinstance(row, Mapping):
            raise DDMRG4Error("RG3 missing block row must be an object")
        key = (int(row["pair_id"]), str(row["bucket_id"]))
        if key in missing_keys:
            raise DDMRG4Error("RG3 missing blocks contain duplicate keys")
        missing_keys.add(key)
    residual_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in residual:
        if not isinstance(row, Mapping):
            raise DDMRG4Error("RG3 derivation row must be an object")
        key = (int(row["pair_id"]), str(row["bucket_id"]))
        if key in residual_by_key:
            raise DDMRG4Error("RG3 derivation contains duplicate block keys")
        residual_by_key[key] = row
    if set(residual_by_key) != missing_keys:
        raise DDMRG4Error("RG3 missing-block and derivation key sets differ")
    exclusions: list[dict[str, Any]] = []
    for missing_row in missing:
        key = (int(missing_row["pair_id"]), str(missing_row["bucket_id"]))
        source = residual_by_key.get(key)
        if source is None:
            raise DDMRG4Error(f"RG3 missing block lacks derivation row: {key}")
        blocker = source.get("rg3_probe_blocker")
        if not isinstance(blocker, Mapping):
            raise DDMRG4Error(f"RG3 probe blocker is absent: {key}")
        if blocker.get("classification") != EXPECTED_RG3_BLOCKER:
            raise DDMRG4Error(f"RG3 obstruction classification differs: {key}")
        probes = blocker.get("probes")
        if not isinstance(probes, list) or not probes:
            raise DDMRG4Error(f"RG3 probe rows are absent: {key}")
        family = source.get("rg3_family")
        raw_actuator_ids = source.get("rg3_receiver_actuator_ids")
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(raw_actuator_ids, list)
            or not raw_actuator_ids
            or any(not isinstance(value, str) or not value for value in raw_actuator_ids)
        ):
            raise DDMRG4Error(f"RG3 family/actuator custody differs: {key}")
        actuator_ids = list(raw_actuator_ids)
        if len(set(actuator_ids)) != len(actuator_ids):
            raise DDMRG4Error(f"RG3 actuator ids contain duplicates: {key}")
        magnitudes = sorted(
            {int(actuator_id.rsplit(".mag", 1)[1]) for actuator_id in actuator_ids if ".mag" in actuator_id}
        )
        if len(magnitudes) != len(actuator_ids):
            raise DDMRG4Error(f"RG3 actuator magnitudes are incomplete or duplicated: {key}")
        directions = {"NEGATIVE_ONE_QUANTUM": -1, "POSITIVE_ONE_QUANTUM": 1}
        expected_probe_keys = {
            (actuator_id, direction_id) for actuator_id in actuator_ids for direction_id in directions
        }
        observed_probe_keys: set[tuple[str, str]] = set()
        checkpoint_sha256s: list[str] = []
        for row in probes:
            if not isinstance(row, Mapping):
                raise DDMRG4Error(f"RG3 probe row must be an object: {key}")
            probe_key = (
                str(row.get("receiver_actuator_id")),
                str(row.get("direction_id")),
            )
            checkpoint_sha256 = row.get("checkpoint_sha256")
            if (
                probe_key in observed_probe_keys
                or probe_key not in expected_probe_keys
                or not isinstance(checkpoint_sha256, str)
                or len(checkpoint_sha256) != 64
                or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
                or checkpoint_sha256 in checkpoint_sha256s
            ):
                raise DDMRG4Error(f"RG3 probe identity/custody differs: {key}")
            observed_probe_keys.add(probe_key)
            checkpoint_sha256s.append(checkpoint_sha256)
        signs = sorted({directions[direction_id] for _, direction_id in observed_probe_keys})
        if (
            blocker.get("probe_count") not in {None, len(probes)}
            or observed_probe_keys != expected_probe_keys
            or signs != [-1, 1]
            or any(
                row.get("target_bucket_event_count") != 0
                or row.get("target_bucket_hit") is not False
                or row.get("target_pair_joined") is not False
                for row in probes
            )
        ):
            raise DDMRG4Error(f"RG3 exhaustive probe custody differs: {key}")
        exclusions.append(
            {
                "pair_id": key[0],
                "bucket": key[1],
                "exact_pair_bucket_joined": False,
                "blocker_classification": EXPECTED_RG3_BLOCKER,
                "production_families_tested": [family],
                "signs_tested": [-1, 1],
                "magnitudes_tested": list(magnitudes),
                "probe_checkpoint_sha256s": checkpoint_sha256s,
                "derived_next_coordinate_family": blocker.get("derived_next_coordinate_family"),
                "verdict_scope": (
                    "INSTANCE: this exact pair/bucket under the production "
                    "RG1/RG2/RG3 alphabets and measured signed magnitudes"
                ),
            }
        )
    return exclusions


__all__ = [
    "EXPECTED_MISSING_BLOCKS",
    "SOURCE_LOCAL_SCHEMA",
    "DDMRG4Error",
    "active_tube_report",
    "apply_source_preserving_delta",
    "build_source_local_composition_archive",
    "canonical_bytes",
    "parse_source_local_composition_archive",
    "receive_source_local_pc1_camera_pairs",
    "rg3_typed_exclusions",
    "sha256_bytes",
]
