# SPDX-License-Identifier: MIT
"""Bind a strict G109 target capsule to a resumable V9 training trajectory.

G109 proves the frozen batch-16 target inputs.  This module is the missing
consumer seam: it verifies that the trainer's source-frame cache is the same
chronological source population, projects the capsule targets into the legacy
trainer container, and emits a complete immutable checkpoint binding.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    AGGREGATE_SCHEMA,
    PRODUCTION_BATCH_PAIRS,
    V9TrainingTargetCapsuleLoaderV1,
    V9TrainingTargetsV1,
    canonical_json_bytes,
    payload_sha256,
    sha256_file,
)

BINDING_SCHEMA = "tac.taskspace_v9_training_target_binding.v1"
CONSUMER_SCHEMA = "tac.taskspace_v9_training_target_consumer.v1"
CHECKPOINT_PROJECTION_KEY = "__cfg_g109_target_projection_json"
CHECKPOINT_PROJECTION_SHA_KEY = "__cfg_g109_target_projection_sha256"


class V9TrainingTargetBindingError(RuntimeError):
    """The capsule, source cache, checkpoint, or live batch coordinate differs."""


def _require_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V9TrainingTargetBindingError(f"{label} must be a lowercase SHA-256")
    return value


def _source_batch_sha256(
    gt_f0: Sequence[np.ndarray],
    gt_f1: Sequence[np.ndarray],
    start: int,
    stop: int,
) -> str:
    """Hash the exact C-order ``uint8[B,2,H,W,3]`` bytes without a batch copy."""

    digest = hashlib.sha256()
    expected_shape: tuple[int, ...] | None = None
    for pair_index in range(start, stop):
        for frame_name, sequence in (("gt_f0", gt_f0), ("gt_f1", gt_f1)):
            frame = np.asarray(sequence[pair_index])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise V9TrainingTargetBindingError(
                    f"{frame_name}[{pair_index}] must be uint8 HxWx3"
                )
            if expected_shape is None:
                expected_shape = tuple(int(value) for value in frame.shape)
            elif tuple(frame.shape) != expected_shape:
                raise V9TrainingTargetBindingError(
                    f"{frame_name}[{pair_index}] changes source-frame geometry"
                )
            contiguous = np.ascontiguousarray(frame)
            digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _load_batch_source_sha(binding: Mapping[str, Any]) -> str:
    try:
        path = Path(str(binding["path"]))
        if path.is_symlink():
            raise V9TrainingTargetBindingError("G109 batch receipt is a symlink")
        raw = path.read_bytes()
        if (
            len(raw) != int(binding["bytes"])
            or hashlib.sha256(raw).hexdigest() != str(binding["sha256"])
        ):
            raise V9TrainingTargetBindingError(
                "G109 batch receipt changed after aggregate validation"
            )
        receipt = json.loads(raw)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V9TrainingTargetBindingError("cannot reopen a G109 batch receipt") from exc
    if not isinstance(receipt, dict):
        raise V9TrainingTargetBindingError("G109 batch receipt is not an object")
    return _require_sha(
        receipt.get("source_pair_batch_sha256"),
        "G109 batch source SHA-256",
    )


def _g46_source_pair_chain(loader: V9TrainingTargetCapsuleLoaderV1) -> str:
    binding = loader.receipt["g46_custody"]["receipt"]
    try:
        path = Path(str(binding["path"]))
        if path.is_symlink():
            raise V9TrainingTargetBindingError("G46 receipt is a symlink")
        raw = path.read_bytes()
        if (
            len(raw) != int(binding["bytes"])
            or hashlib.sha256(raw).hexdigest() != str(binding["sha256"])
        ):
            raise V9TrainingTargetBindingError(
                "G46 receipt changed after target-capsule validation"
            )
        receipt = json.loads(raw)
        rows = receipt["pair_checkpoints"]
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise V9TrainingTargetBindingError(
            "cannot reopen the G46 pair-source custody"
        ) from exc
    if not isinstance(rows, list) or len(rows) != loader.pair_count:
        raise V9TrainingTargetBindingError("G46 pair-source custody is incomplete")
    chain = hashlib.sha256()
    for pair_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("pair_index") != pair_index:
            raise V9TrainingTargetBindingError(
                "G46 pair-source custody is reordered"
            )
        chain.update(
            bytes.fromhex(
                _require_sha(
                    row.get("source_pair_rgb_sha256"),
                    f"G46 source pair {pair_index} SHA-256",
                )
            )
        )
    return chain.hexdigest()


def _projection(loader: V9TrainingTargetCapsuleLoaderV1) -> dict[str, Any]:
    receipt = loader.receipt
    receipt_path = loader.receipt_path
    return {
        "schema": BINDING_SCHEMA,
        "aggregate_schema": AGGREGATE_SCHEMA,
        "aggregate_receipt": {
            "path": str(receipt_path),
            "bytes": int(receipt_path.stat().st_size),
            "sha256": sha256_file(receipt_path),
        },
        "aggregate_receipt_sha256": receipt["aggregate_receipt_sha256"],
        "preflight_sha256": receipt["preflight_sha256"],
        "batch_digest_chain_sha256": receipt["batch_digest_chain_sha256"],
        "g46_receipt_sha256": receipt["g46_custody"]["receipt_sha256"],
        "source_pair_chain_sha256": _g46_source_pair_chain(loader),
        "source_video_sha256": receipt["source_custody"]["source_video"]["sha256"],
        "segnet_weights_sha256": receipt["scorer_custody"]["segnet_weights"]["sha256"],
        "posenet_weights_sha256": receipt["scorer_custody"]["posenet_weights"]["sha256"],
        "arrays": {
            "seg_labels_u8": dict(receipt["raw_arrays"]["labels"]),
            "seg_top1_minus_top2_margin_f32": dict(receipt["raw_arrays"]["margins"]),
            "source_pose6_f32": dict(receipt["raw_arrays"]["poses"]),
        },
        "pair_count": int(loader.pair_count),
        "scorer_pair_batch_size": int(loader.batch_pairs),
        "same_forward_seg_margin_pose": True,
        "encoder_only": True,
        "candidate_payload_allowed": False,
    }


def reopen_v9_training_target_projection(
    *,
    projection_json: str,
    expected_projection_sha256: str,
) -> dict[str, Any]:
    """Reopen the physical G109 capsule named by a checkpoint projection.

    This is the receiver-side custody twin of :func:`bind_v9_training_targets`.
    It does not need the encoder's in-memory frame cache, but it does re-run the
    strict aggregate/batch/array/source/scorer validation and requires the
    canonical projection rebuilt from those physical bytes to match exactly.
    """

    if not isinstance(projection_json, str):
        raise V9TrainingTargetBindingError("checkpoint target projection must be JSON text")
    expected_sha = _require_sha(
        expected_projection_sha256,
        "checkpoint target projection SHA-256",
    )
    try:
        parsed = json.loads(projection_json)
    except json.JSONDecodeError as exc:
        raise V9TrainingTargetBindingError(
            "checkpoint target projection is not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise V9TrainingTargetBindingError(
            "checkpoint target projection must decode to an object"
        )
    canonical = canonical_json_bytes(parsed)
    if canonical.decode("utf-8") != projection_json:
        raise V9TrainingTargetBindingError(
            "checkpoint target projection is not canonical JSON"
        )
    if hashlib.sha256(canonical).hexdigest() != expected_sha:
        raise V9TrainingTargetBindingError(
            "checkpoint target projection SHA-256 differs"
        )
    try:
        aggregate = parsed["aggregate_receipt"]
        aggregate_path = Path(str(aggregate["path"]))
        aggregate_sha = _require_sha(
            aggregate["sha256"],
            "projected G109 aggregate receipt file SHA-256",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise V9TrainingTargetBindingError(
            "checkpoint target projection lacks physical G109 custody"
        ) from exc
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        aggregate_path,
        expected_sha256=aggregate_sha,
    )
    rebuilt = _projection(loader)
    if rebuilt != parsed:
        raise V9TrainingTargetBindingError(
            "checkpoint target projection differs from the reopened physical G109 capsule"
        )
    return rebuilt


def checkpoint_target_arrays_from_projection(
    projection: Mapping[str, Any],
    *,
    active_target_authority_sha256: str,
    verdict_batch: int,
) -> dict[str, np.ndarray]:
    """Derive every checkpoint target scalar from one reopened projection."""

    active_sha = _require_sha(
        active_target_authority_sha256,
        "active target authority SHA-256",
    )
    if int(verdict_batch) != PRODUCTION_BATCH_PAIRS:
        raise V9TrainingTargetBindingError(
            "live verdict/controller batch must equal upstream batch size 16"
        )
    projection_value = dict(projection)
    projection_sha = payload_sha256(projection_value)
    consumer_binding = payload_sha256(
        {
            "schema": CONSUMER_SCHEMA,
            "target_projection_sha256": projection_sha,
            "active_target_authority_sha256": active_sha,
            "live_verdict_batch_size": int(verdict_batch),
        }
    )
    evidence_sha = payload_sha256(
        {
            "schema": "tac.taskspace_v9_training_target_evidence.v1",
            "target_projection_sha256": projection_sha,
            "batch_digest_chain_sha256": projection_value[
                "batch_digest_chain_sha256"
            ],
            "same_forward_seg_margin_pose": True,
        }
    )
    arrays = projection_value["arrays"]
    return {
        CHECKPOINT_PROJECTION_KEY: np.asarray(
            canonical_json_bytes(projection_value).decode("utf-8")
        ),
        CHECKPOINT_PROJECTION_SHA_KEY: np.asarray(projection_sha),
        "__cfg_target_authority_sha256": np.asarray(active_sha),
        "__cfg_g46_target_labels_sha256": np.asarray(
            arrays["seg_labels_u8"]["sha256"]
        ),
        "__cfg_g46_target_margins_sha256": np.asarray(
            arrays["seg_top1_minus_top2_margin_f32"]["sha256"]
        ),
        "__cfg_g46_source_pair_chain_sha256": np.asarray(
            projection_value["source_pair_chain_sha256"]
        ),
        "__cfg_g46_margin_aggregate_schema": np.asarray(AGGREGATE_SCHEMA),
        "__cfg_g46_margin_aggregate_sha256": np.asarray(
            projection_value["aggregate_receipt_sha256"]
        ),
        "__cfg_g46_target_consumer_binding_sha256": np.asarray(consumer_binding),
        "__cfg_g46_target_evidence_sha256": np.asarray(evidence_sha),
        "__cfg_g46_target_scorer_batch_size": np.asarray(
            PRODUCTION_BATCH_PAIRS
        ),
        "__cfg_g46_margin_same_forward": np.asarray(1),
        "__cfg_verdict_batch": np.asarray(int(verdict_batch)),
    }


@dataclass(frozen=True, slots=True)
class BoundV9TrainingTargetsV1:
    """Strict target arrays plus the exact checkpoint/resume custody projection."""

    targets: V9TrainingTargetsV1
    projection: dict[str, Any]
    projection_sha256: str
    target_evidence_sha256: str

    def checkpoint_arrays(
        self,
        *,
        active_target_authority_sha256: str,
        verdict_batch: int,
    ) -> dict[str, np.ndarray]:
        arrays = checkpoint_target_arrays_from_projection(
            self.projection,
            active_target_authority_sha256=active_target_authority_sha256,
            verdict_batch=verdict_batch,
        )
        if np.asarray(arrays["__cfg_g109_target_projection_sha256"]).item() != (
            self.projection_sha256
        ):
            raise AssertionError("bound target projection identity changed")
        if np.asarray(arrays["__cfg_g46_target_evidence_sha256"]).item() != (
            self.target_evidence_sha256
        ):
            raise AssertionError("bound target evidence identity changed")
        return arrays

    def validate_checkpoint_cfg(
        self,
        cfg: Mapping[str, Any],
        *,
        active_target_authority_sha256: str,
        verdict_batch: int,
    ) -> None:
        expected = self.checkpoint_arrays(
            active_target_authority_sha256=active_target_authority_sha256,
            verdict_batch=verdict_batch,
        )
        for key, value in expected.items():
            expected_value = np.asarray(value).item()
            if cfg.get(key) != expected_value:
                raise V9TrainingTargetBindingError(
                    f"resume checkpoint target binding differs at {key}"
                )


def bind_v9_training_targets(
    *,
    aggregate_receipt_path: Path,
    expected_receipt_sha256: str,
    gt_f0: Sequence[np.ndarray],
    gt_f1: Sequence[np.ndarray],
    allowed_roots: Sequence[Path] | None = None,
) -> BoundV9TrainingTargetsV1:
    """Strictly open G109 and prove the trainer cache occupies its source fiber."""

    kwargs: dict[str, Any] = {}
    if allowed_roots is not None:
        kwargs["allowed_roots"] = tuple(allowed_roots)
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        aggregate_receipt_path,
        expected_sha256=_require_sha(
            expected_receipt_sha256,
            "G109 aggregate receipt file SHA-256",
        ),
        **kwargs,
    )
    if len(gt_f0) != loader.pair_count or len(gt_f1) != loader.pair_count:
        raise V9TrainingTargetBindingError(
            "trainer source cache pair count differs from the G109 capsule"
        )
    for batch in loader.receipt["batches"]:
        start, stop = (int(value) for value in batch["pair_range"])
        observed = _source_batch_sha256(gt_f0, gt_f1, start, stop)
        expected = _load_batch_source_sha(batch)
        if observed != expected:
            raise V9TrainingTargetBindingError(
                f"trainer source cache differs from G109 at pair range [{start},{stop})"
            )
    projection = _projection(loader)
    projection_sha = payload_sha256(projection)
    evidence_sha = payload_sha256(
        {
            "schema": "tac.taskspace_v9_training_target_evidence.v1",
            "target_projection_sha256": projection_sha,
            "batch_digest_chain_sha256": projection["batch_digest_chain_sha256"],
            "same_forward_seg_margin_pose": True,
        }
    )
    return BoundV9TrainingTargetsV1(
        targets=loader.targets,
        projection=projection,
        projection_sha256=projection_sha,
        target_evidence_sha256=evidence_sha,
    )


__all__ = [
    "BINDING_SCHEMA",
    "CHECKPOINT_PROJECTION_KEY",
    "CHECKPOINT_PROJECTION_SHA_KEY",
    "CONSUMER_SCHEMA",
    "BoundV9TrainingTargetsV1",
    "V9TrainingTargetBindingError",
    "bind_v9_training_targets",
    "checkpoint_target_arrays_from_projection",
    "reopen_v9_training_target_projection",
]
