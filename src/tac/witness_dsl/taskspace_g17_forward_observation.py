# SPDX-License-Identifier: MIT
"""Typed, dense-retaining G17 frozen-forward observations.

The canonical receipts emitted here are dense-free encoder evidence.  Their
identities are derived from the exact arrays and custody bytes retained by the
in-memory observation; callers never provide an object identity string.

This module is research-only.  It does not call a scorer and its receipts are
forbidden from candidate packets and decoder assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields
from typing import Any, Final, Literal

import numpy as np

TARGET_RECEIPT_SCHEMA: Final = "tac.taskspace_g17_target_forward_observation.v1"
CANDIDATE_RECEIPT_SCHEMA: Final = "tac.taskspace_g17_candidate_forward_observation.v1"
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class G17ForwardObservationError(ValueError):
    """An exact forward array, custody object, or receipt failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _immutable_array(value: np.ndarray, *, name: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise G17ForwardObservationError(f"{name} must be an exact numpy.ndarray")
    if value.dtype.hasobject or value.dtype.kind not in "biuf":
        raise G17ForwardObservationError(f"{name} must have a numeric non-object dtype")
    array = np.ascontiguousarray(value).copy()
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise G17ForwardObservationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _require_bytes(value: object, *, name: str, nonempty: bool = True) -> bytes:
    if type(value) is not bytes or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise G17ForwardObservationError(f"{name} must be {qualifier}immutable bytes")
    return value


def _require_pair_ids(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or not value or any(type(item) is not int for item in value):
        raise G17ForwardObservationError("source_pair_ids must be a nonempty tuple of exact integers")
    if value != tuple(range(value[0], value[0] + len(value))) or value[0] < 0 or value[-1] >= 600:
        raise G17ForwardObservationError("source_pair_ids must be a contiguous interval inside [0,600)")
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G17ForwardObservationError("receipt must be finite canonical ASCII JSON") from exc


def _decode_canonical_json(payload: bytes, *, expected_fields: set[str], schema: str) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G17ForwardObservationError(f"receipt repeats JSON key {key!r}")
            result[key] = value
        return result

    _require_bytes(payload, name="receipt payload")
    try:
        decoded = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G17ForwardObservationError("receipt is not valid ASCII JSON") from exc
    if type(decoded) is not dict or set(decoded) != expected_fields:
        raise G17ForwardObservationError("receipt field set is not exact")
    if decoded.get("schema") != schema or _canonical_json(decoded) != payload:
        raise G17ForwardObservationError("receipt schema or canonical parse/re-emit identity changed")
    return decoded


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise G17ForwardObservationError(f"{name} must be canonical lowercase SHA-256")
    return value


def _forward_shape_checks(
    *,
    source_pair_ids: tuple[int, ...],
    camera_frames: np.ndarray,
    r_numerators: np.ndarray,
    r_denominators: np.ndarray,
    projected_rgb: np.ndarray,
    seg_labels: np.ndarray,
    pose6: np.ndarray,
) -> None:
    pair_count = len(source_pair_ids)
    if camera_frames.shape[0] != pair_count or camera_frames.dtype != np.uint8:
        raise G17ForwardObservationError("camera_frames must be uint8 with leading population-pair axis")
    if projected_rgb.shape[0] != pair_count or projected_rgb.dtype != np.uint8:
        raise G17ForwardObservationError("projected_rgb must be uint8 with leading population-pair axis")
    if seg_labels.shape[0] != pair_count or seg_labels.dtype.kind not in "iu":
        raise G17ForwardObservationError("seg_labels must be an integer array with leading pair axis")
    if pose6.shape != (pair_count, 6) or pose6.dtype.kind != "f":
        raise G17ForwardObservationError("pose6 must be a finite floating array shaped [pair,6]")
    if r_numerators.shape[0] != pair_count or r_numerators.dtype.kind != "f":
        raise G17ForwardObservationError("r_numerators must be floating with leading pair axis")
    if r_denominators.shape[0] != pair_count or r_denominators.dtype.kind != "f":
        raise G17ForwardObservationError("r_denominators must be floating with leading pair axis")
    if np.any(r_denominators <= 0.0):
        raise G17ForwardObservationError("R denominators must be strictly positive")


@dataclass(frozen=True, slots=True)
class G17TargetForwardReceiptV1:
    schema: Literal["tac.taskspace_g17_target_forward_observation.v1"]
    source_pair_ids: tuple[int, ...]
    target_artifact_sha256: str
    target_member_sha256: str
    target_camera_frames_sha256: str
    exact_r_numerators_sha256: str
    exact_r_denominators_sha256: str
    exact_r_projected_rgb_sha256: str
    target_seg_labels_sha256: str
    target_pose6_sha256: str
    frozen_scorer_sha256: str
    scorer_runtime_environment_sha256: str
    double_forward_equal: Literal[True]
    encoder_evidence_only: Literal[True]
    candidate_payload_allowed: Literal[False]

    def __post_init__(self) -> None:
        _require_pair_ids(self.source_pair_ids)
        for item in fields(self):
            if item.name.endswith("_sha256"):
                _require_sha256(getattr(self, item.name), name=item.name)
        if self.double_forward_equal is not True:
            raise G17ForwardObservationError("target double-forward equality must be true")
        if self.encoder_evidence_only is not True or self.candidate_payload_allowed is not False:
            raise G17ForwardObservationError("target receipt crossed the encoder-only payload boundary")

    def as_dict(self) -> dict[str, Any]:
        return {
            item.name: list(value) if item.name == "source_pair_ids" else value
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        }

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_g17_target_forward_receipt(payload: bytes) -> G17TargetForwardReceiptV1:
    expected = {item.name for item in fields(G17TargetForwardReceiptV1)}
    decoded = _decode_canonical_json(payload, expected_fields=expected, schema=TARGET_RECEIPT_SCHEMA)
    decoded["source_pair_ids"] = tuple(decoded["source_pair_ids"])
    try:
        receipt = G17TargetForwardReceiptV1(**decoded)
    except (TypeError, ValueError) as exc:
        raise G17ForwardObservationError("target receipt values failed exact validation") from exc
    if receipt.to_receipt_bytes() != payload:
        raise G17ForwardObservationError("target receipt changed on parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class G17TargetForwardObservationV1:
    """Exact target forward values plus their second-forward equality witness."""

    source_pair_ids: tuple[int, ...]
    target_artifact_bytes: bytes = field(repr=False)
    target_member_bytes: bytes = field(repr=False)
    camera_frames: np.ndarray = field(repr=False)
    exact_r_numerators: np.ndarray = field(repr=False)
    exact_r_denominators: np.ndarray = field(repr=False)
    exact_r_projected_rgb: np.ndarray = field(repr=False)
    seg_labels: np.ndarray = field(repr=False)
    pose6: np.ndarray = field(repr=False)
    second_exact_r_numerators: np.ndarray = field(repr=False)
    second_exact_r_denominators: np.ndarray = field(repr=False)
    second_exact_r_projected_rgb: np.ndarray = field(repr=False)
    second_seg_labels: np.ndarray = field(repr=False)
    second_pose6: np.ndarray = field(repr=False)
    frozen_scorer_bytes: bytes = field(repr=False)
    scorer_runtime_environment_bytes: bytes = field(repr=False)
    receipt: G17TargetForwardReceiptV1 = field(init=False)

    def __post_init__(self) -> None:
        pair_ids = _require_pair_ids(self.source_pair_ids)
        artifact = _require_bytes(self.target_artifact_bytes, name="target_artifact_bytes")
        member = _require_bytes(self.target_member_bytes, name="target_member_bytes")
        scorer = _require_bytes(self.frozen_scorer_bytes, name="frozen_scorer_bytes")
        runtime = _require_bytes(
            self.scorer_runtime_environment_bytes,
            name="scorer_runtime_environment_bytes",
        )
        array_names = (
            "camera_frames",
            "exact_r_numerators",
            "exact_r_denominators",
            "exact_r_projected_rgb",
            "seg_labels",
            "pose6",
            "second_exact_r_numerators",
            "second_exact_r_denominators",
            "second_exact_r_projected_rgb",
            "second_seg_labels",
            "second_pose6",
        )
        arrays = {name: _immutable_array(getattr(self, name), name=name) for name in array_names}
        _forward_shape_checks(
            source_pair_ids=pair_ids,
            camera_frames=arrays["camera_frames"],
            r_numerators=arrays["exact_r_numerators"],
            r_denominators=arrays["exact_r_denominators"],
            projected_rgb=arrays["exact_r_projected_rgb"],
            seg_labels=arrays["seg_labels"],
            pose6=arrays["pose6"],
        )
        for primary, second in (
            ("exact_r_numerators", "second_exact_r_numerators"),
            ("exact_r_denominators", "second_exact_r_denominators"),
            ("exact_r_projected_rgb", "second_exact_r_projected_rgb"),
            ("seg_labels", "second_seg_labels"),
            ("pose6", "second_pose6"),
        ):
            if arrays[primary].dtype != arrays[second].dtype or not np.array_equal(arrays[primary], arrays[second]):
                raise G17ForwardObservationError(f"target double-forward mismatch for {primary}")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        receipt = G17TargetForwardReceiptV1(
            schema=TARGET_RECEIPT_SCHEMA,
            source_pair_ids=pair_ids,
            target_artifact_sha256=_sha256(artifact),
            target_member_sha256=_sha256(member),
            target_camera_frames_sha256=_array_sha256(arrays["camera_frames"]),
            exact_r_numerators_sha256=_array_sha256(arrays["exact_r_numerators"]),
            exact_r_denominators_sha256=_array_sha256(arrays["exact_r_denominators"]),
            exact_r_projected_rgb_sha256=_array_sha256(arrays["exact_r_projected_rgb"]),
            target_seg_labels_sha256=_array_sha256(arrays["seg_labels"]),
            target_pose6_sha256=_array_sha256(arrays["pose6"]),
            frozen_scorer_sha256=_sha256(scorer),
            scorer_runtime_environment_sha256=_sha256(runtime),
            double_forward_equal=True,
            encoder_evidence_only=True,
            candidate_payload_allowed=False,
        )
        if parse_g17_target_forward_receipt(receipt.to_receipt_bytes()) != receipt:
            raise G17ForwardObservationError("target observation receipt failed strict closure")
        object.__setattr__(self, "receipt", receipt)

    @property
    def observation_sha256(self) -> str:
        return self.receipt.receipt_sha256


@dataclass(frozen=True, slots=True)
class G17CandidateForwardReceiptV1:
    schema: Literal["tac.taskspace_g17_candidate_forward_observation.v1"]
    source_pair_ids: tuple[int, ...]
    archive_sha256: str
    member_sha256: str
    receiver_receipt_sha256: str
    decoded_output_sha256: str
    camera_y1_sha256: str
    exact_r_numerators_sha256: str
    exact_r_denominators_sha256: str
    exact_r_projected_rgb_sha256: str
    realized_seg_labels_sha256: str
    realized_pose6_sha256: str
    target_forward_receipt_sha256: str
    frozen_scorer_sha256: str
    scorer_runtime_environment_sha256: str
    per_pair_d_seg: tuple[float, ...]
    per_pair_d_pose: tuple[float, ...]
    aggregate_d_seg: float
    aggregate_d_pose: float
    double_forward_equal: Literal[True]
    encoder_evidence_only: Literal[True]
    candidate_payload_allowed: Literal[False]

    def __post_init__(self) -> None:
        pair_ids = _require_pair_ids(self.source_pair_ids)
        for item in fields(self):
            if item.name.endswith("_sha256"):
                _require_sha256(getattr(self, item.name), name=item.name)
        for name in ("per_pair_d_seg", "per_pair_d_pose"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != len(pair_ids):
                raise G17ForwardObservationError(f"{name} must have one exact value per pair")
            if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in values):
                raise G17ForwardObservationError(f"{name} contains a non-finite or negative value")
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in (self.aggregate_d_seg, self.aggregate_d_pose)
        ):
            raise G17ForwardObservationError("aggregate distances must be finite nonnegative floats")
        if self.aggregate_d_seg > 1.0 or any(value > 1.0 for value in self.per_pair_d_seg):
            raise G17ForwardObservationError("segmentation distances cannot exceed one")
        if self.double_forward_equal is not True:
            raise G17ForwardObservationError("candidate double-forward equality must be true")
        if self.encoder_evidence_only is not True or self.candidate_payload_allowed is not False:
            raise G17ForwardObservationError("candidate receipt crossed the encoder-only payload boundary")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = list(value) if type(value) is tuple else value
        return result

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_g17_candidate_forward_receipt(payload: bytes) -> G17CandidateForwardReceiptV1:
    expected = {item.name for item in fields(G17CandidateForwardReceiptV1)}
    decoded = _decode_canonical_json(payload, expected_fields=expected, schema=CANDIDATE_RECEIPT_SCHEMA)
    for name in ("source_pair_ids", "per_pair_d_seg", "per_pair_d_pose"):
        decoded[name] = tuple(decoded[name])
    try:
        receipt = G17CandidateForwardReceiptV1(**decoded)
    except (TypeError, ValueError) as exc:
        raise G17ForwardObservationError("candidate receipt values failed exact validation") from exc
    if receipt.to_receipt_bytes() != payload:
        raise G17ForwardObservationError("candidate receipt changed on parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class G17CandidateForwardObservationV1:
    """Exact received-candidate forward values measured against one typed target."""

    target: G17TargetForwardObservationV1
    archive_bytes: bytes = field(repr=False)
    member_bytes: bytes = field(repr=False)
    receiver_receipt_bytes: bytes = field(repr=False)
    decoded_output_bytes: bytes = field(repr=False)
    camera_y1: np.ndarray = field(repr=False)
    exact_r_numerators: np.ndarray = field(repr=False)
    exact_r_denominators: np.ndarray = field(repr=False)
    exact_r_projected_rgb: np.ndarray = field(repr=False)
    realized_seg_labels: np.ndarray = field(repr=False)
    realized_pose6: np.ndarray = field(repr=False)
    second_exact_r_numerators: np.ndarray = field(repr=False)
    second_exact_r_denominators: np.ndarray = field(repr=False)
    second_exact_r_projected_rgb: np.ndarray = field(repr=False)
    second_realized_seg_labels: np.ndarray = field(repr=False)
    second_realized_pose6: np.ndarray = field(repr=False)
    frozen_scorer_bytes: bytes = field(repr=False)
    scorer_runtime_environment_bytes: bytes = field(repr=False)
    receipt: G17CandidateForwardReceiptV1 = field(init=False)

    def __post_init__(self) -> None:
        if type(self.target) is not G17TargetForwardObservationV1:
            raise G17ForwardObservationError("candidate must bind an exact target observation object")
        exact_bytes = {
            name: _require_bytes(getattr(self, name), name=name)
            for name in (
                "archive_bytes",
                "member_bytes",
                "receiver_receipt_bytes",
                "decoded_output_bytes",
                "frozen_scorer_bytes",
                "scorer_runtime_environment_bytes",
            )
        }
        arrays = {
            name: _immutable_array(getattr(self, name), name=name)
            for name in (
                "camera_y1",
                "exact_r_numerators",
                "exact_r_denominators",
                "exact_r_projected_rgb",
                "realized_seg_labels",
                "realized_pose6",
                "second_exact_r_numerators",
                "second_exact_r_denominators",
                "second_exact_r_projected_rgb",
                "second_realized_seg_labels",
                "second_realized_pose6",
            )
        }
        _forward_shape_checks(
            source_pair_ids=self.target.source_pair_ids,
            camera_frames=arrays["camera_y1"],
            r_numerators=arrays["exact_r_numerators"],
            r_denominators=arrays["exact_r_denominators"],
            projected_rgb=arrays["exact_r_projected_rgb"],
            seg_labels=arrays["realized_seg_labels"],
            pose6=arrays["realized_pose6"],
        )
        for primary, second in (
            ("exact_r_numerators", "second_exact_r_numerators"),
            ("exact_r_denominators", "second_exact_r_denominators"),
            ("exact_r_projected_rgb", "second_exact_r_projected_rgb"),
            ("realized_seg_labels", "second_realized_seg_labels"),
            ("realized_pose6", "second_realized_pose6"),
        ):
            if arrays[primary].dtype != arrays[second].dtype or not np.array_equal(arrays[primary], arrays[second]):
                raise G17ForwardObservationError(f"candidate double-forward mismatch for {primary}")
        if arrays["realized_seg_labels"].shape != self.target.seg_labels.shape:
            raise G17ForwardObservationError("candidate/target segmentation shapes differ")
        if arrays["realized_pose6"].shape != self.target.pose6.shape:
            raise G17ForwardObservationError("candidate/target Pose6 shapes differ")
        if _sha256(exact_bytes["frozen_scorer_bytes"]) != self.target.receipt.frozen_scorer_sha256:
            raise G17ForwardObservationError("candidate scorer bytes differ from target scorer custody")
        if (
            _sha256(exact_bytes["scorer_runtime_environment_bytes"])
            != self.target.receipt.scorer_runtime_environment_sha256
        ):
            raise G17ForwardObservationError("candidate runtime bytes differ from target runtime custody")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

        target_labels = np.asarray(self.target.seg_labels)
        candidate_labels = np.asarray(arrays["realized_seg_labels"])
        target_pose = np.asarray(self.target.pose6, dtype=np.float64)
        candidate_pose = np.asarray(arrays["realized_pose6"], dtype=np.float64)
        per_pair_seg = tuple(
            float(np.mean(candidate_labels[i] != target_labels[i], dtype=np.float64)) for i in range(len(target_pose))
        )
        per_pair_pose = tuple(
            float(np.mean(np.square(candidate_pose[i] - target_pose[i]), dtype=np.float64))
            for i in range(len(target_pose))
        )
        aggregate_seg = float(np.mean(candidate_labels != target_labels, dtype=np.float64))
        aggregate_pose = float(np.mean(np.square(candidate_pose - target_pose), dtype=np.float64))
        receipt = G17CandidateForwardReceiptV1(
            schema=CANDIDATE_RECEIPT_SCHEMA,
            source_pair_ids=self.target.source_pair_ids,
            archive_sha256=_sha256(exact_bytes["archive_bytes"]),
            member_sha256=_sha256(exact_bytes["member_bytes"]),
            receiver_receipt_sha256=_sha256(exact_bytes["receiver_receipt_bytes"]),
            decoded_output_sha256=_sha256(exact_bytes["decoded_output_bytes"]),
            camera_y1_sha256=_array_sha256(arrays["camera_y1"]),
            exact_r_numerators_sha256=_array_sha256(arrays["exact_r_numerators"]),
            exact_r_denominators_sha256=_array_sha256(arrays["exact_r_denominators"]),
            exact_r_projected_rgb_sha256=_array_sha256(arrays["exact_r_projected_rgb"]),
            realized_seg_labels_sha256=_array_sha256(arrays["realized_seg_labels"]),
            realized_pose6_sha256=_array_sha256(arrays["realized_pose6"]),
            target_forward_receipt_sha256=self.target.receipt.receipt_sha256,
            frozen_scorer_sha256=_sha256(exact_bytes["frozen_scorer_bytes"]),
            scorer_runtime_environment_sha256=_sha256(exact_bytes["scorer_runtime_environment_bytes"]),
            per_pair_d_seg=per_pair_seg,
            per_pair_d_pose=per_pair_pose,
            aggregate_d_seg=aggregate_seg,
            aggregate_d_pose=aggregate_pose,
            double_forward_equal=True,
            encoder_evidence_only=True,
            candidate_payload_allowed=False,
        )
        if parse_g17_candidate_forward_receipt(receipt.to_receipt_bytes()) != receipt:
            raise G17ForwardObservationError("candidate observation receipt failed strict closure")
        object.__setattr__(self, "receipt", receipt)

    @property
    def observation_sha256(self) -> str:
        return self.receipt.receipt_sha256


def require_candidate_observation_matches_exact_objects(
    observation: G17CandidateForwardObservationV1,
    *,
    target: G17TargetForwardObservationV1,
    archive_bytes: bytes,
    member_bytes: bytes,
    receiver_receipt_bytes: bytes,
    decoded_output_bytes: bytes,
) -> None:
    """Refuse an arbitrary or swapped receipt/object join."""

    if type(observation) is not G17CandidateForwardObservationV1 or observation.target is not target:
        raise G17ForwardObservationError("candidate observation does not retain the exact target object")
    expected = (
        (_sha256(archive_bytes), observation.receipt.archive_sha256, "archive"),
        (_sha256(member_bytes), observation.receipt.member_sha256, "member"),
        (_sha256(receiver_receipt_bytes), observation.receipt.receiver_receipt_sha256, "receiver receipt"),
        (_sha256(decoded_output_bytes), observation.receipt.decoded_output_sha256, "decoded output"),
    )
    for actual, recorded, name in expected:
        if actual != recorded:
            raise G17ForwardObservationError(f"{name} bytes differ from typed candidate observation")
    if parse_g17_target_forward_receipt(target.receipt.to_receipt_bytes()) != target.receipt:
        raise G17ForwardObservationError("target receipt did not reopen under its strict parser")
    if parse_g17_candidate_forward_receipt(observation.receipt.to_receipt_bytes()) != observation.receipt:
        raise G17ForwardObservationError("candidate receipt did not reopen under its strict parser")


__all__ = [
    "CANDIDATE_RECEIPT_SCHEMA",
    "TARGET_RECEIPT_SCHEMA",
    "G17CandidateForwardObservationV1",
    "G17CandidateForwardReceiptV1",
    "G17ForwardObservationError",
    "G17TargetForwardObservationV1",
    "G17TargetForwardReceiptV1",
    "parse_g17_candidate_forward_receipt",
    "parse_g17_target_forward_receipt",
    "require_candidate_observation_matches_exact_objects",
]
