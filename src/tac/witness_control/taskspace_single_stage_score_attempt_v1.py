# SPDX-License-Identifier: MIT
"""One explicit G121 stage to a receiver-closed upstream score attempt.

This is deliberately *not* the exhaustive G119 population controller.  The
caller names one self-hashed, distortion-open G121 COMPLETED row from an exact
append-only-ledger file identity.  The row is copied into immutable custody,
its G112 partition is physically reopened, and its joint-descent pose
initializer is measured without a Gauss-Newton update.  That baseline-only
mode costs exactly one chronological n600 PoseNet pass (38 batch forwards),
preserves one checkpoint after every batch, and makes no optimizer, Pareto, or
cross-stage coverage claim.

The measured initializer is emitted through the strict post-G105 checkpoint
ABI consumed by G110, the exact 2x2x2 semantic/XIP2/outer-ZIP matrix is
arbitrated, and the public receiver files are atomically packaged.  Two clean
public decodes must produce identical raw bytes before the frozen
``upstream/evaluate.py`` is invoked once.  Every long stage has an immutable
receipt so a crash can resume from disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

import numpy as np

from tac.boundary_math.xi_pose_coder import dequantize_xi, quantize_xi
from tac.witness_control import taskspace_g121_resumable_stage_harvest_v1 as g121
from tac.witness_control import taskspace_post_g105_pose_refit_v1 as g119
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    SSD_ROOTS,
)
from tac.witness_dsl.taskspace_g110_generated_y1_pose_product_v1 import (
    compile_g110_generated_y1_pose_v1,
)
from tac.witness_dsl.taskspace_g110_release_materializer_v1 import (
    ARCHIVE_BASENAME,
    MIN_RELEASE_FREE_BYTES,
    RECEIPT_BASENAME,
    _publish_exact_directory,
    _validate_output_root,
    capture_public_runtime_v1,
    capture_release_source_closure_v1,
)

CONFIG_SCHEMA: Final = "tac.single_stage_score_attempt_config.v1"
SELECTED_ROW_SCHEMA: Final = "tac.single_stage_score_attempt_g121_row.v1"
BASELINE_STATE_SCHEMA: Final = "tac.single_stage_score_attempt_pose_baseline_state.v1"
BASELINE_RECEIPT_SCHEMA: Final = "tac.single_stage_score_attempt_pose_baseline_receipt.v1"
RELEASE_RECEIPT_SCHEMA: Final = "tac.single_stage_score_attempt_release_receipt.v1"
DECODE_RECEIPT_SCHEMA: Final = "tac.single_stage_score_attempt_decode_receipt.v1"
EVAL_RECEIPT_SCHEMA: Final = "tac.single_stage_score_attempt_upstream_eval_receipt.v1"
FINAL_RECEIPT_SCHEMA: Final = "tac.single_stage_score_attempt_final_receipt.v1"
RANK_ZERO_CONFIG_SCHEMA: Final = "tac.single_stage_rank_zero_score_attempt_config.v1"
RANK_ZERO_RELEASE_RECEIPT_SCHEMA: Final = (
    "tac.single_stage_rank_zero_release_receipt.v1"
)
RANK_ZERO_FINAL_RECEIPT_SCHEMA: Final = (
    "tac.single_stage_rank_zero_score_attempt_final_receipt.v1"
)
BLOCKER_SCHEMA: Final = "tac.single_stage_score_attempt_blocker.v1"
PAIR_COUNT: Final = 600
BATCH_PAIRS: Final = 16
FRAME_COUNT: Final = 1200
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
CHANNELS: Final = 3
EXPECTED_RAW_BYTES: Final = FRAME_COUNT * CAMERA_H * CAMERA_W * CHANNELS
ARCHIVE_DENOMINATOR_BYTES: Final = 37_545_489
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REPORT_RESULTS = re.compile(
    r"=== Evaluation results over (?P<n>[0-9]+) samples ===\n"
    r"  Average PoseNet Distortion: (?P<pose>[0-9]+\.[0-9]{8})\n"
    r"  Average SegNet Distortion: (?P<seg>[0-9]+\.[0-9]{8})\n"
    r"  Submission file size: (?P<archive>[0-9,]+) bytes\n"
    r"  Original uncompressed size: (?P<original>[0-9,]+) bytes\n"
    r"  Compression Rate: (?P<rate>[0-9]+\.[0-9]{8})\n"
    r"  Final score: .* = (?P<rounded>[0-9]+\.[0-9]{2})\n"
)


class SingleStageScoreAttemptError(RuntimeError):
    """A custody, receiver, or scorer obligation failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SingleStageScoreAttemptError(
            "value is not finite canonical ASCII JSON"
        ) from exc


def _pretty_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SingleStageScoreAttemptError(
            "value is not finite strict JSON"
        ) from exc


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SingleStageScoreAttemptError(
            f"{name} must be canonical lowercase SHA-256"
        )
    return value


def _seal(body: Mapping[str, object], *, field: str) -> dict[str, object]:
    if field in body:
        raise SingleStageScoreAttemptError(
            f"unsealed body already contains {field}"
        )
    result = dict(body)
    result[field] = _sha256(_canonical_json(body))
    return result


def _regular_file(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise SingleStageScoreAttemptError(f"{name} must not be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise SingleStageScoreAttemptError(
            f"{name} must be a regular file"
        )
    return resolved


def _binding(path: Path) -> dict[str, object]:
    resolved = _regular_file(path, name="bound artifact")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _validate_binding(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise SingleStageScoreAttemptError(f"{name} binding key set differs")
    path = _regular_file(Path(str(value["path"])), name=name)
    if (
        value["path"] != str(path)
        or type(value["bytes"]) is not int
        or value["bytes"] != path.stat().st_size
        or value["sha256"] != _sha256_file(path)
    ):
        raise SingleStageScoreAttemptError(
            f"{name} physical identity differs"
        )
    _require_sha256(value["sha256"], name=f"{name} SHA-256")
    return dict(value)


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise SingleStageScoreAttemptError(
                f"immutable output differs: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_once(path: Path, value: object) -> None:
    _write_once(path, _pretty_json(value))


def _open_sealed_json(
    path: Path,
    *,
    schema: str,
    hash_field: str,
) -> dict[str, object]:
    resolved = _regular_file(path, name=f"{schema} receipt")
    try:
        value = json.loads(resolved.read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SingleStageScoreAttemptError(
            f"{schema} receipt is not strict JSON"
        ) from exc
    if type(value) is not dict or value.get("schema") != schema:
        raise SingleStageScoreAttemptError(
            f"{schema} receipt identity differs"
        )
    sealed = dict(value)
    observed = _require_sha256(
        sealed.pop(hash_field, None),
        name=f"{schema} receipt",
    )
    if observed != _sha256(_canonical_json(sealed)):
        raise SingleStageScoreAttemptError(
            f"{schema} receipt self-hash differs"
        )
    return value


def _stable_read(path: Path, *, expected_sha256: str, name: str) -> bytes:
    resolved = _regular_file(path, name=name)
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != after.st_size
        or _sha256(payload)
        != _require_sha256(expected_sha256, name=f"expected {name}")
    ):
        raise SingleStageScoreAttemptError(
            f"{name} changed or differs from its expected identity"
        )
    return payload


@dataclass(frozen=True, slots=True)
class SingleStageScoreAttemptConfigV1:
    run_id: str
    seed: int
    resume_from: Path
    g121_stage_ledger: Path
    expected_g121_stage_ledger_sha256: str
    g121_attempt_identity_sha256: str
    target_capsule_receipt: Path
    expected_target_capsule_receipt_sha256: str
    q_levels: int
    pose_device: str
    evaluator_device: str
    torch_num_threads: int
    expected_runtime_tree_sha256: str
    video_names_file: Path
    authority_axis: str
    competitive_target: str


@dataclass(frozen=True, slots=True)
class SingleStageScoreAttemptResultV1:
    final_receipt_path: Path
    final_receipt_sha256: str
    submission_dir: Path
    archive_sha256: str
    archive_bytes: int
    report_path: Path
    d_pose: float
    d_seg: float
    recomputed_score: float
    authority_axis: str


@dataclass(frozen=True, slots=True)
class RankZeroScoreAttemptConfigV1:
    run_id: str
    resume_from: Path
    g121_stage_ledger: Path
    expected_g121_stage_ledger_sha256: str
    g121_attempt_identity_sha256: str
    expected_runtime_tree_sha256: str
    evaluator_device: str
    video_names_file: Path
    authority_axis: str
    competitive_target: str


def _validate_config(
    config: SingleStageScoreAttemptConfigV1,
) -> SingleStageScoreAttemptConfigV1:
    if (
        type(config.run_id) is not str
        or _RUN_ID.fullmatch(config.run_id) is None
        or type(config.seed) is not int
        or type(config.q_levels) is not int
        or not 1 <= config.q_levels <= 32_767
        or config.pose_device not in {"cpu", "cuda"}
        or config.evaluator_device not in {"cpu", "cuda"}
        or type(config.torch_num_threads) is not int
        or not 1 <= config.torch_num_threads <= 64
        or config.authority_axis
        not in {"macOS-CPU advisory", "contest-CPU", "contest-CUDA"}
    ):
        raise SingleStageScoreAttemptError(
            "single-stage attempt config is outside typed bounds"
        )
    if (
        config.authority_axis == "contest-CUDA"
        and config.evaluator_device != "cuda"
    ) or (
        config.authority_axis in {"macOS-CPU advisory", "contest-CPU"}
        and config.evaluator_device != "cpu"
    ):
        raise SingleStageScoreAttemptError(
            "authority axis and evaluator device differ"
        )
    if (
        config.authority_axis == "macOS-CPU advisory"
        and platform.system() != "Darwin"
    ):
        raise SingleStageScoreAttemptError(
            "macOS advisory axis requires a Darwin host"
        )
    if (
        config.authority_axis in {"contest-CPU", "contest-CUDA"}
        and platform.system() != "Linux"
    ):
        raise SingleStageScoreAttemptError(
            "contest authority axes require the Linux contest host"
        )
    try:
        target = Decimal(config.competitive_target)
    except Exception as exc:
        raise SingleStageScoreAttemptError(
            "competitive target is not a decimal"
        ) from exc
    if not target.is_finite() or target <= 0:
        raise SingleStageScoreAttemptError(
            "competitive target must be finite and positive"
        )
    _require_sha256(
        config.expected_g121_stage_ledger_sha256,
        name="expected G121 ledger",
    )
    _require_sha256(
        config.g121_attempt_identity_sha256,
        name="G121 attempt identity",
    )
    _require_sha256(
        config.expected_target_capsule_receipt_sha256,
        name="expected target capsule receipt",
    )
    _require_sha256(
        config.expected_runtime_tree_sha256,
        name="expected public runtime tree",
    )
    return config


def _validate_rank_zero_config(
    config: RankZeroScoreAttemptConfigV1,
) -> RankZeroScoreAttemptConfigV1:
    if (
        type(config.run_id) is not str
        or _RUN_ID.fullmatch(config.run_id) is None
        or config.evaluator_device not in {"cpu", "cuda"}
        or config.authority_axis
        not in {"macOS-CPU advisory", "contest-CPU", "contest-CUDA"}
    ):
        raise SingleStageScoreAttemptError(
            "rank-zero attempt config is outside typed bounds"
        )
    if (
        config.authority_axis == "contest-CUDA"
        and config.evaluator_device != "cuda"
    ) or (
        config.authority_axis in {"macOS-CPU advisory", "contest-CPU"}
        and config.evaluator_device != "cpu"
    ):
        raise SingleStageScoreAttemptError(
            "rank-zero authority axis and evaluator device differ"
        )
    if (
        config.authority_axis == "macOS-CPU advisory"
        and platform.system() != "Darwin"
    ):
        raise SingleStageScoreAttemptError(
            "macOS advisory axis requires a Darwin host"
        )
    if (
        config.authority_axis in {"contest-CPU", "contest-CUDA"}
        and platform.system() != "Linux"
    ):
        raise SingleStageScoreAttemptError(
            "contest authority axes require the Linux contest host"
        )
    try:
        target = Decimal(config.competitive_target)
    except Exception as exc:
        raise SingleStageScoreAttemptError(
            "competitive target is not a decimal"
        ) from exc
    if not target.is_finite() or target <= 0:
        raise SingleStageScoreAttemptError(
            "competitive target must be finite and positive"
        )
    for value, name in (
        (
            config.expected_g121_stage_ledger_sha256,
            "expected G121 ledger",
        ),
        (config.g121_attempt_identity_sha256, "G121 attempt identity"),
        (
            config.expected_runtime_tree_sha256,
            "expected public runtime tree",
        ),
    ):
        _require_sha256(value, name=name)
    return config


def _rank_zero_config_document(
    config: RankZeroScoreAttemptConfigV1,
    *,
    command: Sequence[str],
) -> dict[str, object]:
    tokens = [str(token) for token in command]
    if not tokens or "--resume-from" not in tokens:
        raise SingleStageScoreAttemptError(
            "durable rank-zero command must contain --resume-from"
        )
    body: dict[str, object] = {
        "schema": RANK_ZERO_CONFIG_SCHEMA,
        "run_id": config.run_id,
        "resume_from": str(config.resume_from.expanduser().resolve()),
        "g121_stage_ledger": str(
            config.g121_stage_ledger.expanduser().resolve()
        ),
        "expected_g121_stage_ledger_sha256": (
            config.expected_g121_stage_ledger_sha256
        ),
        "g121_attempt_identity_sha256": (
            config.g121_attempt_identity_sha256
        ),
        "expected_runtime_tree_sha256": (
            config.expected_runtime_tree_sha256
        ),
        "evaluator_device": config.evaluator_device,
        "video_names_file": str(
            config.video_names_file.expanduser().resolve()
        ),
        "authority_axis": config.authority_axis,
        "competitive_target": config.competitive_target,
        "command": tokens,
        "attempt_mode": "G120_SELECTED_RANK_ZERO_ARCHIVE",
        "pose_refit_run": False,
        "gauss_newton_stages": 0,
        "exhaustive_stage_coverage_claim": False,
        "pareto_claim": False,
        "cross_stage_winner_claim": False,
        "pointer_moved": False,
    }
    return _seal(body, field="config_sha256")


def _config_document(
    config: SingleStageScoreAttemptConfigV1,
    *,
    command: Sequence[str],
) -> dict[str, object]:
    tokens = [str(token) for token in command]
    if not tokens or "--resume-from" not in tokens:
        raise SingleStageScoreAttemptError(
            "durable command must contain --resume-from"
        )
    body: dict[str, object] = {
        "schema": CONFIG_SCHEMA,
        "run_id": config.run_id,
        "seed": config.seed,
        "resume_from": str(config.resume_from.expanduser().resolve()),
        "g121_stage_ledger": str(
            config.g121_stage_ledger.expanduser().resolve()
        ),
        "expected_g121_stage_ledger_sha256": (
            config.expected_g121_stage_ledger_sha256
        ),
        "g121_attempt_identity_sha256": (
            config.g121_attempt_identity_sha256
        ),
        "target_capsule_receipt": str(
            config.target_capsule_receipt.expanduser().resolve()
        ),
        "expected_target_capsule_receipt_sha256": (
            config.expected_target_capsule_receipt_sha256
        ),
        "q_levels": config.q_levels,
        "pose_device": config.pose_device,
        "evaluator_device": config.evaluator_device,
        "torch_num_threads": config.torch_num_threads,
        "expected_runtime_tree_sha256": (
            config.expected_runtime_tree_sha256
        ),
        "video_names_file": str(
            config.video_names_file.expanduser().resolve()
        ),
        "authority_axis": config.authority_axis,
        "competitive_target": config.competitive_target,
        "command": tokens,
        "explicit_single_stage_attempt": True,
        "baseline_only_pose_measurement": True,
        "gauss_newton_stages": 0,
        "exhaustive_stage_coverage_claim": False,
        "pareto_claim": False,
        "cross_stage_winner_claim": False,
        "pointer_moved": False,
    }
    return _seal(body, field="config_sha256")


def _recursively_reopen_g120_g112(
    row: Mapping[str, object],
) -> None:
    try:
        g120_module = g121._require_safe_g120_v2()
        g120_section = row["g120"]
        physical = row["physical_stage_identity"]
        if type(g120_section) is not dict or type(physical) is not dict:
            raise SingleStageScoreAttemptError(
                "selected row recursive custody shape differs"
            )
        measurement_binding = g120_section["measurement_receipt"]
        observation_binding = g120_section["observation_receipt"]
        reopened_measurement = getattr(
            g120_module,
            g121.G120_REQUIRED_MEASUREMENT_OPENER,
        )(
            Path(str(measurement_binding["path"])),
            expected_sha256=str(measurement_binding["sha256"]),
        )
        reopened_observation = getattr(
            g120_module,
            g121.G120_REQUIRED_OBSERVATION_OPENER,
        )(
            Path(str(observation_binding["path"])),
            expected_sha256=str(observation_binding["sha256"]),
        )
        if (
            reopened_measurement.measurement_identity_sha256
            != row["measurement_identity_sha256"]
            or reopened_observation.observation_identity_sha256
            != row["observation_identity_sha256"]
        ):
            raise SingleStageScoreAttemptError(
                "G120 recursive receipt identity differs"
            )
        from tac.witness_control.taskspace_g112_exact_checkpoint_partition_v1 import (
            open_g112_partition_receipt,
        )

        g112_binding = physical["g112_partition_receipt"]
        open_g112_partition_receipt(
            Path(str(g112_binding["path"])),
            expected_sha256=str(g112_binding["sha256"]),
        )
    except SingleStageScoreAttemptError:
        raise
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SingleStageScoreAttemptError(
            "selected row failed recursive G120/G112 reopen"
        ) from exc


def _open_selected_g121_row(
    *,
    config: SingleStageScoreAttemptConfigV1 | RankZeroScoreAttemptConfigV1,
    run_root: Path,
) -> dict[str, object]:
    payload = _stable_read(
        config.g121_stage_ledger,
        expected_sha256=config.expected_g121_stage_ledger_sha256,
        name="G121 stage ledger",
    )
    if payload and not payload.endswith(b"\n"):
        raise SingleStageScoreAttemptError(
            "G121 stage ledger has a torn final row"
        )
    ledger_snapshot_path = run_root / "00_g121_stage_ledger_snapshot.jsonl"
    _write_once(ledger_snapshot_path, payload)
    try:
        rows = g121._read_stage_ledger(ledger_snapshot_path)
    except (OSError, ValueError, g121.G121StageHarvestError) as exc:
        raise SingleStageScoreAttemptError(
            "G121 stage ledger failed production validation"
        ) from exc
    selected = [
        row
        for row in rows
        if row.get("attempt_identity_sha256")
        == config.g121_attempt_identity_sha256
    ]
    if len(selected) != 1:
        raise SingleStageScoreAttemptError(
            "explicit G121 attempt identity is not unique in the ledger"
        )
    try:
        row = g121._validate_completed_attempt(
            selected[0],
            require_physical_files=True,
        )
    except (OSError, ValueError, g121.G121StageHarvestError) as exc:
        raise SingleStageScoreAttemptError(
            "selected G121 row failed recursive physical validation"
        ) from exc
    obstruction = row["prepose_obstruction"]
    if (
        row["attempt_status"] != "COMPLETED"
        or obstruction["disposition"] != g121.RETAIN_POST_G105_POSE
        or obstruction["strict_distortion_open"] is not True
        or row["retained_for_post_g105_pose"] is not True
    ):
        raise SingleStageScoreAttemptError(
            "selected G121 row is not a strict distortion-open retained row"
        )
    _recursively_reopen_g120_g112(row)
    snapshot_body: dict[str, object] = {
        "schema": SELECTED_ROW_SCHEMA,
        "source_ledger": {
            "path": str(config.g121_stage_ledger.expanduser().resolve()),
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
        "immutable_ledger_snapshot": _binding(
            ledger_snapshot_path
        ),
        "attempt_identity_sha256": (
            config.g121_attempt_identity_sha256
        ),
        "row": row,
        "recursive_physical_validation": True,
        "append_only_source_frozen_by_exact_file_sha256": True,
        "exhaustive_stage_coverage_claim": False,
        "pareto_claim": False,
        "cross_stage_winner_claim": False,
    }
    snapshot = _seal(snapshot_body, field="snapshot_sha256")
    path = run_root / "01_selected_g121_row.json"
    _write_json_once(path, snapshot)
    reopened = json.loads(path.read_text("ascii"))
    if reopened != snapshot:
        raise SingleStageScoreAttemptError(
            "immutable G121 row snapshot changed on reopen"
        )
    return row


def _baseline_state_path(root: Path, next_pair: int) -> Path:
    return (
        root
        / "20_pose_baseline_states"
        / f"pairs{next_pair:04d}_baseline.npz"
    )


def _baseline_state_arrays(
    *,
    config_sha256: str,
    run_id: str,
    seed: int,
    custody: g119.PostG105RefitCustodyV1,
    q_levels: int,
    next_pair: int,
    q: np.ndarray,
    scales: np.ndarray,
    pose_outputs: np.ndarray,
    pose_losses: np.ndarray,
    oracle: g119.ExactBatch16PoseOracleV1,
) -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(BASELINE_STATE_SCHEMA),
        "config_sha256": np.asarray(config_sha256),
        "run_id": np.asarray(run_id),
        "seed": np.asarray(seed, dtype=np.int64),
        "source_g112_partition_receipt_sha256": np.asarray(
            custody.base.partition_receipt_sha256
        ),
        "target_capsule_receipt_sha256": np.asarray(
            custody.base.target_capsule_receipt_sha256
        ),
        "pose_targets_sha256": np.asarray(
            custody.base.pose_targets_sha256
        ),
        "xi_initializer_sha256": np.asarray(
            g119._xi_digest(custody.base.pose_initializer.xi_init)
        ),
        "q_levels": np.asarray(q_levels, dtype=np.int64),
        "next_pair": np.asarray(next_pair, dtype=np.int64),
        "q": np.ascontiguousarray(q, dtype=np.int16),
        "scales": np.ascontiguousarray(scales, dtype=np.float32),
        "pose_outputs": np.ascontiguousarray(
            pose_outputs,
            dtype=np.float64,
        ),
        "pose_losses": np.ascontiguousarray(
            pose_losses,
            dtype=np.float64,
        ),
        **oracle.rng_arrays(),
    }


def _load_baseline_state(
    *,
    root: Path,
    config_sha256: str,
    run_id: str,
    seed: int,
    custody: g119.PostG105RefitCustodyV1,
    q_levels: int,
    q: np.ndarray,
    scales: np.ndarray,
    oracle: g119.ExactBatch16PoseOracleV1,
) -> tuple[int, np.ndarray, np.ndarray]:
    paths = sorted(
        (root / "20_pose_baseline_states").glob(
            "pairs[0-9][0-9][0-9][0-9]_baseline.npz"
        )
    )
    if not paths:
        return (
            0,
            np.full((PAIR_COUNT, 6), np.nan, dtype=np.float64),
            np.full((PAIR_COUNT,), np.nan, dtype=np.float64),
        )
    latest = max(paths, key=lambda path: int(path.name[5:9]))
    with np.load(latest, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    expected_members = {
        "schema",
        "config_sha256",
        "run_id",
        "seed",
        "source_g112_partition_receipt_sha256",
        "target_capsule_receipt_sha256",
        "pose_targets_sha256",
        "xi_initializer_sha256",
        "q_levels",
        "next_pair",
        "q",
        "scales",
        "pose_outputs",
        "pose_losses",
        "torch_rng_state_u8",
        "torch_cuda_rng_states_u8",
    }
    if set(arrays) != expected_members:
        raise SingleStageScoreAttemptError(
            "baseline resume state member set differs"
        )
    scalar_expected = {
        "schema": BASELINE_STATE_SCHEMA,
        "config_sha256": config_sha256,
        "run_id": run_id,
        "seed": seed,
        "source_g112_partition_receipt_sha256": (
            custody.base.partition_receipt_sha256
        ),
        "target_capsule_receipt_sha256": (
            custody.base.target_capsule_receipt_sha256
        ),
        "pose_targets_sha256": custody.base.pose_targets_sha256,
        "xi_initializer_sha256": g119._xi_digest(
            custody.base.pose_initializer.xi_init
        ),
        "q_levels": q_levels,
    }
    for name, expected in scalar_expected.items():
        if np.asarray(arrays[name]).reshape(()).item() != expected:
            raise SingleStageScoreAttemptError(
                f"baseline resume custody differs: {name}"
            )
    next_pair = int(np.asarray(arrays["next_pair"]).reshape(()).item())
    outputs = arrays["pose_outputs"]
    losses = arrays["pose_losses"]
    if (
        next_pair != int(latest.name[5:9])
        or not 0 < next_pair <= PAIR_COUNT
        or (next_pair % BATCH_PAIRS != 0 and next_pair != PAIR_COUNT)
        or not np.array_equal(arrays["q"], q)
        or not np.array_equal(arrays["scales"], scales)
        or outputs.dtype != np.float64
        or outputs.shape != (PAIR_COUNT, 6)
        or losses.dtype != np.float64
        or losses.shape != (PAIR_COUNT,)
        or not np.all(np.isfinite(outputs[:next_pair]))
        or not np.all(np.isfinite(losses[:next_pair]))
        or not np.all(np.isnan(outputs[next_pair:]))
        or not np.all(np.isnan(losses[next_pair:]))
    ):
        raise SingleStageScoreAttemptError(
            "baseline resume cursor or arrays differ"
        )
    oracle.restore_rng(arrays)
    return next_pair, outputs, losses


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SingleStageScoreAttemptError(
            "cannot bind source Git HEAD"
        ) from exc
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SingleStageScoreAttemptError(
            "source Git HEAD is not canonical SHA-1"
        )
    return value


def _explicit_source_bindings(
    repo_root: Path,
    *,
    git_sha: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in (
        "src/tac/witness_control/taskspace_single_stage_score_attempt_v1.py",
        "tools/run_taskspace_single_stage_score_attempt_v1.py",
    ):
        path = _regular_file(
            repo_root / relative,
            name=f"single-stage source {relative}",
        )
        payload = path.read_bytes()
        try:
            committed = subprocess.run(
                ["git", "show", f"{git_sha}:{relative}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SingleStageScoreAttemptError(
                "single-stage release source is not committed at bound HEAD: "
                f"{relative}"
            ) from exc
        if committed != payload:
            raise SingleStageScoreAttemptError(
                "single-stage release source differs from bound Git HEAD: "
                f"{relative}"
            )
        rows.append(
            {
                "relative_path": relative,
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "matches_git_head": True,
            }
        )
    return rows


def _run_pose_baseline(
    *,
    config: SingleStageScoreAttemptConfigV1,
    config_path: Path,
    config_sha256: str,
    selected_row: Mapping[str, object],
    target_binding: Mapping[str, object],
    run_root: Path,
    command: Sequence[str],
) -> tuple[
    g119.PostG105PoseRefitResultV1,
    dict[str, object],
]:
    physical = selected_row["physical_stage_identity"]
    if type(physical) is not dict:
        raise SingleStageScoreAttemptError(
            "selected G121 physical stage identity differs"
        )
    g112_binding = _validate_binding(
        physical["g112_partition_receipt"],
        name="selected G112 partition receipt",
    )
    baseline_root = run_root / "pose_baseline"
    baseline_root.mkdir(parents=True, exist_ok=True)
    bridge = g119.PostG105PoseRefitConfigV1(
        config_path=config_path,
        config_sha256=config_sha256,
        run_id=config.run_id,
        seed=config.seed,
        output_root=baseline_root,
        g112_partition_receipt=g112_binding,
        target_capsule_receipt=dict(target_binding),
        q_levels_candidates=(config.q_levels,),
        local_gauss_newton_stages=0,
        finite_difference_q_steps=1,
        damping=1.0,
        trust_radius_q=1,
        line_search_scales=(1.0,),
        device=config.pose_device,
        torch_num_threads=config.torch_num_threads,
    )
    custody = g119.open_custody(bridge)
    oracle = g119.ExactBatch16PoseOracleV1(
        custody,
        device=config.pose_device,
        seed=config.seed,
        torch_num_threads=config.torch_num_threads,
    )
    q, scales = quantize_xi(
        custody.base.pose_initializer.xi_init,
        q_levels=config.q_levels,
    )
    xi_eff = np.ascontiguousarray(
        dequantize_xi(q, scales),
        dtype=np.float64,
    )
    q_check, scales_check = quantize_xi(
        xi_eff,
        q_levels=config.q_levels,
    )
    if (
        not np.array_equal(q, q_check)
        or not np.array_equal(scales, scales_check)
    ):
        raise SingleStageScoreAttemptError(
            "baseline initializer is not compiler-fixed under XIP2"
        )
    next_pair, outputs, losses = _load_baseline_state(
        root=baseline_root,
        config_sha256=config_sha256,
        run_id=config.run_id,
        seed=config.seed,
        custody=custody,
        q_levels=config.q_levels,
        q=q,
        scales=scales,
        oracle=oracle,
    )
    while next_pair < PAIR_COUNT:
        start = next_pair
        stop = min(start + BATCH_PAIRS, PAIR_COUNT)
        state = g119._SolverState(
            q_levels=config.q_levels,
            stage_index=0,
            next_pair=start,
            q=q,
            scales=scales,
            pose_outputs=outputs,
            pose_losses=losses,
            accepted_rows=0,
            attempted_rows=0,
        )
        g119._baseline_batch(
            state,
            start=start,
            stop=stop,
            custody=custody,
            oracle=oracle,
        )
        next_pair = stop
        _write_once(
            _baseline_state_path(baseline_root, next_pair),
            g119._deterministic_npz(
                _baseline_state_arrays(
                    config_sha256=config_sha256,
                    run_id=config.run_id,
                    seed=config.seed,
                    custody=custody,
                    q_levels=config.q_levels,
                    next_pair=next_pair,
                    q=q,
                    scales=scales,
                    pose_outputs=outputs,
                    pose_losses=losses,
                    oracle=oracle,
                )
            ),
        )
    pose_mse = oracle.population_mse(losses)
    archive_oracle = g119.ExactCompleteArchiveRateOracleV1.prepare(custody)
    (
        xip2,
        packet,
        archive,
        selected_wire,
        alternatives,
    ) = archive_oracle.materialize(q=q, scales=scales)
    checkpoint_path = (
        baseline_root
        / "90_final"
        / "post_g105_pose_refit_final.npz"
    )
    _write_once(
        checkpoint_path,
        g119._deterministic_npz(
            g119._final_checkpoint_arrays(
                config=bridge,
                custody=custody,
                q_levels=config.q_levels,
                xi_eff=xi_eff,
                selected_xip2_coder=selected_wire.xip2_coder,
            )
        ),
    )
    checkpoint_binding = _binding(checkpoint_path)
    state_bindings = [
        _binding(path)
        for path in sorted(
            (baseline_root / "20_pose_baseline_states").glob(
                "pairs*_baseline.npz"
            )
        )
    ]
    if (
        len(state_bindings) != math.ceil(PAIR_COUNT / BATCH_PAIRS)
        or state_bindings[-1]["path"]
        != str(_baseline_state_path(baseline_root, PAIR_COUNT).resolve())
    ):
        raise SingleStageScoreAttemptError(
            "baseline-only run lacks its 38 per-batch checkpoints"
        )
    repo_root = Path(__file__).resolve().parents[3]
    base = custody.base
    run_body: dict[str, object] = {
        "schema": g119.POST_G105_REFIT_RUN_SCHEMA,
        "run_id": config.run_id,
        "seed": config.seed,
        "source_git_sha": _git_head(repo_root),
        "command": [str(token) for token in command],
        "fresh_own_lineage": True,
        "source_contract": g119.SOURCE_DOMAIN,
        "render_order": g119.RENDER_ORDER,
        "y1_selected_preimage_schema": (
            g119.V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA
        ),
        "source_g112_partition_receipt_sha256": (
            base.partition_receipt_sha256
        ),
        "source_g112_semantic_child_sha256": (
            base.semantic_child_sha256
        ),
        "source_g112_pose_initializer_sha256": (
            base.pose_initializer_sha256
        ),
        "source_g111_deploy_checkpoint_sha256": (
            base.source_deploy_checkpoint_sha256
        ),
        "source_g111_resume_checkpoint_sha256": (
            base.source_resume_checkpoint_sha256
        ),
        "source_g111_lineage_receipt_sha256": (
            base.source_lineage_receipt_sha256
        ),
        "source_g111_checkpoint_id_sha256": (
            base.source_checkpoint_id_sha256
        ),
        "source_g111_root_sha256": base.source_root_sha256,
        "semantic_packet_sha256": _sha256(custody.semantic_packet),
        "final_y1_binding_sha256": custody.final_y1_binding_sha256,
        "xi_initializer_sha256": g119._xi_digest(
            base.pose_initializer.xi_init
        ),
        "target_projection_sha256": base.target_projection_sha256,
        "target_capsule_receipt_sha256": (
            base.target_capsule_receipt_sha256
        ),
        "pose_targets_sha256": base.pose_targets_sha256,
        "selected_xip2_coder": selected_wire.xip2_coder,
        "g110_selected_xip2_coder_abi_closed": True,
        "exact_public_receiver_in_loop": True,
        "resumable_from_disk": True,
        "stage_checkpoints_preserved": True,
        "stage_checkpoints": [*state_bindings, checkpoint_binding],
        "final_checkpoint": checkpoint_binding,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    run_receipt = g119._seal(run_body, field="receipt_sha256")
    run_path = baseline_root / "91_post_g105_pose_refit_run_receipt.json"
    _write_json_once(run_path, run_receipt)
    alternatives_rows = [
        {
            "y1_wire_codec": row.y1_wire_codec.name,
            "xip2_coder": row.xip2_coder,
            "outer_zip_method": row.outer_zip_method.name,
            "archive_bytes": row.archive_bytes,
            "archive_sha256": row.archive_sha256,
        }
        for row in alternatives
    ]
    receipt_body: dict[str, object] = {
        "schema": BASELINE_RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "config": _binding(config_path),
        "g121_attempt_identity_sha256": (
            selected_row["attempt_identity_sha256"]
        ),
        "g112_partition_receipt": g112_binding,
        "target_capsule_receipt": dict(target_binding),
        "mode": "MEASURE_QUANTIZED_G112_POSE_INITIALIZER_ONLY",
        "gauss_newton_stages": 0,
        "optimizer_run": False,
        "optimizer_improvement_claim": False,
        "pair_count": PAIR_COUNT,
        "batch_pairs": BATCH_PAIRS,
        "pose_batch_forwards": math.ceil(PAIR_COUNT / BATCH_PAIRS),
        "per_batch_checkpoints": state_bindings,
        "q_levels": config.q_levels,
        "pose_mse": pose_mse,
        "pose_term": math.sqrt(10.0 * pose_mse),
        "xip2_bytes": len(xip2),
        "xip2_sha256": _sha256(xip2),
        "product_packet_bytes": len(packet),
        "product_packet_sha256": _sha256(packet),
        "complete_archive_bytes": len(archive),
        "complete_archive_sha256": _sha256(archive),
        "complete_archive_wire_candidates": alternatives_rows,
        "selected_y1_wire_codec": selected_wire.y1_wire_codec.name,
        "selected_xip2_coder": selected_wire.xip2_coder,
        "selected_outer_zip_method": selected_wire.outer_zip_method.name,
        "post_g105_refit_checkpoint": checkpoint_binding,
        "post_g105_refit_run_receipt": _binding(run_path),
        "exact_public_receiver_in_loop": True,
        "exhaustive_stage_coverage_claim": False,
        "pareto_claim": False,
        "cross_stage_winner_claim": False,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    baseline_receipt = _seal(
        receipt_body,
        field="baseline_receipt_sha256",
    )
    baseline_receipt_path = (
        baseline_root / "92_pose_baseline_receipt.json"
    )
    _write_json_once(baseline_receipt_path, baseline_receipt)
    return (
        g119.PostG105PoseRefitResultV1(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=str(checkpoint_binding["sha256"]),
            run_receipt_path=run_path,
            run_receipt_sha256=_sha256_file(run_path),
            audit_receipt_path=baseline_receipt_path,
            selected_q_levels=config.q_levels,
            selected_pose_mse=pose_mse,
            selected_archive_bytes=len(archive),
            selected_archive_sha256=_sha256(archive),
            selected_xip2_coder=selected_wire.xip2_coder,
            xip2_wire_matrix_closure={
                "matrix": "2x2x2",
                "candidate_count": len(alternatives),
            },
        ),
        baseline_receipt,
    )


def _materialize_release(
    *,
    config: SingleStageScoreAttemptConfigV1,
    run_root: Path,
    selected_row: Mapping[str, object],
    target_binding: Mapping[str, object],
    refit: g119.PostG105PoseRefitResultV1,
    baseline_receipt: Mapping[str, object],
    command: Sequence[str],
) -> tuple[Path, object, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    git_sha = _git_head(repo_root)
    source = capture_release_source_closure_v1(
        repo_root=repo_root,
        expected_git_sha=git_sha,
    )
    if not source.all_files_equal_git_head:
        raise SingleStageScoreAttemptError(
            "transitive release source differs from bound Git HEAD"
        )
    explicit_source = _explicit_source_bindings(
        repo_root,
        git_sha=git_sha,
    )
    runtime = capture_public_runtime_v1(
        repo_root=repo_root,
        expected_tree_sha256=config.expected_runtime_tree_sha256,
    )
    physical = selected_row["physical_stage_identity"]
    if type(physical) is not dict:
        raise SingleStageScoreAttemptError(
            "G121 physical stage identity differs at release"
        )
    g112_binding = _validate_binding(
        physical["g112_partition_receipt"],
        name="release G112 partition receipt",
    )
    compiled = compile_g110_generated_y1_pose_v1(
        target_capsule_receipt=Path(
            str(target_binding["path"])
        ),
        expected_target_capsule_receipt_sha256=str(
            target_binding["sha256"]
        ),
        g112_partition_receipt=Path(str(g112_binding["path"])),
        expected_g112_partition_receipt_sha256=str(
            g112_binding["sha256"]
        ),
        post_g105_refit_checkpoint=refit.checkpoint_path,
        expected_post_g105_refit_checkpoint_sha256=(
            refit.checkpoint_sha256
        ),
        post_g105_refit_run_receipt=refit.run_receipt_path,
        expected_post_g105_refit_run_receipt_sha256=(
            refit.run_receipt_sha256
        ),
    )
    if (
        compiled.archive_bytes != refit.selected_archive_bytes
        or compiled.archive_sha256 != refit.selected_archive_sha256
        or compiled.selected_xip2_coder != refit.selected_xip2_coder
    ):
        raise SingleStageScoreAttemptError(
            "G110 recompile differs from the measured baseline archive"
        )
    release_root = run_root / "submission"
    receipt_body: dict[str, object] = {
        "schema": RELEASE_RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "command": [str(token) for token in command],
        "release_source_closure": {
            "git_sha": source.git_sha,
            "transitive_files": list(source.files),
            "transitive_tree_sha256": source.tree_sha256,
            "all_transitive_files_equal_git_head": True,
            "single_stage_entry_files": explicit_source,
            "single_stage_entry_tree_sha256": _sha256(
                _canonical_json(explicit_source)
            ),
        },
        "g121_attempt_identity_sha256": (
            selected_row["attempt_identity_sha256"]
        ),
        "physical_stage_identity_sha256": (
            selected_row["physical_stage_identity_sha256"]
        ),
        "target_capsule_receipt": dict(target_binding),
        "g112_partition_receipt": g112_binding,
        "baseline_receipt_sha256": (
            baseline_receipt["baseline_receipt_sha256"]
        ),
        "post_g105_refit_checkpoint": _binding(
            refit.checkpoint_path
        ),
        "post_g105_refit_run_receipt": _binding(
            refit.run_receipt_path
        ),
        "archive": {
            "relative_path": ARCHIVE_BASENAME,
            "bytes": compiled.archive_bytes,
            "sha256": compiled.archive_sha256,
            "counted_by_upstream_evaluate_py": True,
        },
        "packet_sha256": compiled.packet_sha256,
        "final_y1_binding_sha256": (
            compiled.final_y1_binding_sha256
        ),
        "selected_y1_wire_codec": (
            compiled.selected_y1_wire_codec.name
        ),
        "selected_xip2_coder": compiled.selected_xip2_coder,
        "selected_outer_zip_method": (
            compiled.selected_outer_zip_method.name
        ),
        "public_runtime": {
            "files": list(runtime.files),
            "tree_sha256": runtime.tree_sha256,
            "packaged_in_archive": False,
        },
        "receiver_files_packaged": True,
        "explicit_single_stage_attempt": True,
        "exhaustive_stage_coverage_claim": False,
        "pareto_claim": False,
        "cross_stage_winner_claim": False,
        "clean_public_entrypoint_double_decode_run": False,
        "upstream_evaluate_py_run": False,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    release_receipt = _seal(
        receipt_body,
        field="release_receipt_sha256",
    )
    files: dict[str, tuple[bytes, int]] = {
        ARCHIVE_BASENAME: (compiled.archive, 0o644),
        RECEIPT_BASENAME: (
            _canonical_json(release_receipt),
            0o644,
        ),
    }
    files.update(
        {
            relative: (payload, mode)
            for relative, payload, mode in runtime.payloads
        }
    )
    _publish_exact_directory(
        output_root=release_root,
        files=files,
    )
    source_after = capture_release_source_closure_v1(
        repo_root=repo_root,
        expected_git_sha=git_sha,
    )
    explicit_source_after = _explicit_source_bindings(
        repo_root,
        git_sha=git_sha,
    )
    if (
        source_after != source
        or explicit_source_after != explicit_source
    ):
        raise SingleStageScoreAttemptError(
            "release source closure changed during publication"
        )
    if (
        _sha256_file(release_root / ARCHIVE_BASENAME)
        != compiled.archive_sha256
    ):
        raise SingleStageScoreAttemptError(
            "published G110 archive differs"
        )
    return release_root, compiled, release_receipt


def _g121_runtime_tree_sha256(
    selected_row: Mapping[str, object],
) -> str:
    value = selected_row.get("public_runtime_tree")
    if type(value) is not dict:
        raise SingleStageScoreAttemptError(
            "selected G121 public runtime tree differs"
        )
    if set(value) == {
        "source_pre",
        "sealed_tree_sha256",
        "source_post",
        "toctou_closed_by",
    }:
        source_pre = value["source_pre"]
        if (
            type(source_pre) is not dict
            or source_pre != value["source_post"]
            or value["toctou_closed_by"]
            != "sealed_content_execution_plus_source_postverify"
            or value["sealed_tree_sha256"]
            != source_pre.get("tree_sha256")
        ):
            raise SingleStageScoreAttemptError(
                "G121 sealed runtime TOCTOU closure differs"
            )
        return _require_sha256(
            value["sealed_tree_sha256"],
            name="G121 sealed runtime tree",
        )
    candidate = value.get(
        "content_tree_sha256",
        value.get("tree_sha256"),
    )
    return _require_sha256(
        candidate,
        name="G121 public runtime tree",
    )


def _materialize_rank_zero_release(
    *,
    config: RankZeroScoreAttemptConfigV1,
    run_root: Path,
    selected_row: Mapping[str, object],
    command: Sequence[str],
) -> tuple[Path, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    run_root.mkdir(parents=True, exist_ok=True)
    runtime_tree = _g121_runtime_tree_sha256(selected_row)
    if runtime_tree != config.expected_runtime_tree_sha256:
        raise SingleStageScoreAttemptError(
            "selected G121 runtime tree differs from the explicit expected tree"
        )
    archive_binding = _validate_binding(
        selected_row.get("selected_archive"),
        name="G120 selected rank-zero archive",
    )
    archive_payload = _stable_read(
        Path(str(archive_binding["path"])),
        expected_sha256=str(archive_binding["sha256"]),
        name="G120 selected rank-zero archive",
    )
    git_sha = _git_head(repo_root)
    source = capture_release_source_closure_v1(
        repo_root=repo_root,
        expected_git_sha=git_sha,
    )
    if not source.all_files_equal_git_head:
        raise SingleStageScoreAttemptError(
            "rank-zero transitive release source differs from bound Git HEAD"
        )
    explicit_source = _explicit_source_bindings(
        repo_root,
        git_sha=git_sha,
    )
    runtime = capture_public_runtime_v1(
        repo_root=repo_root,
        expected_tree_sha256=runtime_tree,
    )
    release_root = run_root / "submission"
    receipt_body: dict[str, object] = {
        "schema": RANK_ZERO_RELEASE_RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "command": [str(token) for token in command],
        "attempt_mode": "G120_SELECTED_RANK_ZERO_ARCHIVE",
        "g121_attempt_identity_sha256": (
            selected_row["attempt_identity_sha256"]
        ),
        "physical_stage_identity_sha256": (
            selected_row["physical_stage_identity_sha256"]
        ),
        "g120_production_receipt": _validate_binding(
            selected_row["g120"]["production_receipt"],
            name="G120 production receipt",
        ),
        "g120_measurement_receipt": _validate_binding(
            selected_row["g120"]["measurement_receipt"],
            name="G120 measurement receipt",
        ),
        "g120_observation_receipt": _validate_binding(
            selected_row["g120"]["observation_receipt"],
            name="G120 observation receipt",
        ),
        "selected_archive_source": archive_binding,
        "archive": {
            "relative_path": ARCHIVE_BASENAME,
            "bytes": len(archive_payload),
            "sha256": _sha256(archive_payload),
            "counted_by_upstream_evaluate_py": True,
            "copied_without_mutation": True,
        },
        "public_runtime": {
            "g121_sealed_tree_sha256": runtime_tree,
            "packaged_tree_sha256": runtime.tree_sha256,
            "files": list(runtime.files),
            "packaged_in_archive": False,
        },
        "release_source_closure": {
            "git_sha": source.git_sha,
            "transitive_files": list(source.files),
            "transitive_tree_sha256": source.tree_sha256,
            "all_transitive_files_equal_git_head": True,
            "single_stage_entry_files": explicit_source,
            "single_stage_entry_tree_sha256": _sha256(
                _canonical_json(explicit_source)
            ),
        },
        "g120_receiver_parseback_already_recursively_validated": True,
        "receiver_files_packaged": True,
        "pose_refit_run": False,
        "exhaustive_stage_coverage_claim": False,
        "pareto_claim": False,
        "cross_stage_winner_claim": False,
        "clean_public_entrypoint_double_decode_run": False,
        "upstream_evaluate_py_run": False,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    receipt = _seal(
        receipt_body,
        field="release_receipt_sha256",
    )
    files: dict[str, tuple[bytes, int]] = {
        ARCHIVE_BASENAME: (archive_payload, 0o644),
        RECEIPT_BASENAME: (_canonical_json(receipt), 0o644),
    }
    files.update(
        {
            relative: (payload, mode)
            for relative, payload, mode in runtime.payloads
        }
    )
    _publish_exact_directory(
        output_root=release_root,
        files=files,
    )
    if _binding(release_root / ARCHIVE_BASENAME) != {
        "path": str((release_root / ARCHIVE_BASENAME).resolve()),
        "bytes": archive_binding["bytes"],
        "sha256": archive_binding["sha256"],
    }:
        raise SingleStageScoreAttemptError(
            "published rank-zero archive differs from G120 custody"
        )
    source_after = capture_release_source_closure_v1(
        repo_root=repo_root,
        expected_git_sha=git_sha,
    )
    explicit_source_after = _explicit_source_bindings(
        repo_root,
        git_sha=git_sha,
    )
    if (
        source_after != source
        or explicit_source_after != explicit_source
    ):
        raise SingleStageScoreAttemptError(
            "rank-zero source closure changed during publication"
        )
    return release_root, receipt


def _safe_extract_archive(archive: Path, output: Path) -> None:
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise SingleStageScoreAttemptError(
                "archive extraction root differs"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            infos = handle.infolist()
            if not infos:
                raise SingleStageScoreAttemptError(
                    "counted archive is empty"
                )
            for info in infos:
                relative = Path(info.filename)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or info.is_dir()
                ):
                    raise SingleStageScoreAttemptError(
                        "counted archive member is unsafe"
                    )
            handle.extractall(output)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SingleStageScoreAttemptError(
            "counted archive could not be extracted"
        ) from exc


def _raw_tree_binding(
    *,
    inflated: Path,
    video_names_file: Path,
) -> dict[str, object]:
    names = [
        line.strip()
        for line in video_names_file.read_text("utf-8").splitlines()
        if line.strip()
    ]
    if names != ["0.mkv"]:
        raise SingleStageScoreAttemptError(
            "score attempt requires the exact frozen one-video n600 list"
        )
    expected = {f"{Path(name).stem}.raw" for name in names}
    observed = {
        path.relative_to(inflated).as_posix()
        for path in inflated.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise SingleStageScoreAttemptError(
            "public receiver raw file census differs"
        )
    rows = []
    for relative in sorted(observed):
        path = _regular_file(
            inflated / relative,
            name=f"inflated {relative}",
        )
        if path.stat().st_size != EXPECTED_RAW_BYTES:
            raise SingleStageScoreAttemptError(
                "inflated raw size differs from exact n600 geometry"
            )
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "files": rows,
        "tree_sha256": _sha256(_canonical_json(rows)),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }


def _run_decode(
    *,
    submission_dir: Path,
    video_names_file: Path,
    run_root: Path,
    decode_index: int,
) -> dict[str, object]:
    receipt_path = run_root / f"decode{decode_index}_receipt.json"
    if receipt_path.exists():
        receipt = _open_sealed_json(
            receipt_path,
            schema=DECODE_RECEIPT_SCHEMA,
            hash_field="decode_receipt_sha256",
        )
        if (
            receipt.get("decode_index") != decode_index
            or receipt.get("returncode") != 0
        ):
            raise SingleStageScoreAttemptError(
                "resumed decode receipt coordinate differs"
            )
        _validate_binding(
            receipt.get("archive"),
            name=f"decode {decode_index} archive",
        )
        _validate_binding(
            receipt.get("public_inflate_sh"),
            name=f"decode {decode_index} inflate.sh",
        )
        _validate_binding(
            receipt.get("video_names_file"),
            name=f"decode {decode_index} video names",
        )
        observed_tree = _raw_tree_binding(
            inflated=submission_dir / "inflated",
            video_names_file=video_names_file,
        )
        if receipt.get("raw_tree") != observed_tree:
            raise SingleStageScoreAttemptError(
                "resumed decode raw tree differs"
            )
        return receipt
    archive_dir = submission_dir / "archive"
    inflated_dir = submission_dir / "inflated"
    _safe_extract_archive(
        submission_dir / ARCHIVE_BASENAME,
        archive_dir,
    )
    stdout_path = run_root / f"decode{decode_index}.stdout.txt"
    stderr_path = run_root / f"decode{decode_index}.stderr.txt"
    process = subprocess.run(
        [
            "bash",
            str(submission_dir / "inflate.sh"),
            str(archive_dir),
            str(inflated_dir),
            str(video_names_file),
        ],
        cwd=submission_dir,
        capture_output=True,
        check=False,
    )
    _write_once(stdout_path, process.stdout)
    _write_once(stderr_path, process.stderr)
    if process.returncode != 0:
        raise SingleStageScoreAttemptError(
            f"public inflate.sh decode {decode_index} failed"
        )
    raw_tree = _raw_tree_binding(
        inflated=inflated_dir,
        video_names_file=video_names_file,
    )
    body: dict[str, object] = {
        "schema": DECODE_RECEIPT_SCHEMA,
        "decode_index": decode_index,
        "archive": _binding(submission_dir / ARCHIVE_BASENAME),
        "public_inflate_sh": _binding(
            submission_dir / "inflate.sh"
        ),
        "video_names_file": _binding(video_names_file),
        "raw_tree": raw_tree,
        "stdout": _binding(stdout_path),
        "stderr": _binding(stderr_path),
        "returncode": process.returncode,
        "exact_n600_geometry": True,
    }
    receipt = _seal(body, field="decode_receipt_sha256")
    _write_json_once(receipt_path, receipt)
    return receipt


def _parse_report(
    payload: str,
    *,
    archive_bytes: int,
) -> dict[str, object]:
    match = _REPORT_RESULTS.search(payload)
    if match is None:
        raise SingleStageScoreAttemptError(
            "upstream report result block differs"
        )
    sample_count = int(match.group("n"))
    reported_archive = int(match.group("archive").replace(",", ""))
    original = int(match.group("original").replace(",", ""))
    pose = Decimal(match.group("pose"))
    seg = Decimal(match.group("seg"))
    reported_rate = Decimal(match.group("rate"))
    exact_rate = Decimal(archive_bytes) / Decimal(original)
    recomputed = (
        Decimal(100) * seg
        + (Decimal(10) * pose).sqrt()
        + Decimal(25) * exact_rate
    )
    if (
        sample_count != PAIR_COUNT
        or reported_archive != archive_bytes
        or original != ARCHIVE_DENOMINATOR_BYTES
        or abs(reported_rate - exact_rate) > Decimal("0.0000000051")
    ):
        raise SingleStageScoreAttemptError(
            "upstream report sample/rate identity differs"
        )
    return {
        "sample_count": sample_count,
        "d_pose_reported_8dp": str(pose),
        "d_seg_reported_8dp": str(seg),
        "archive_bytes": reported_archive,
        "original_uncompressed_bytes": original,
        "rate_reported_8dp": str(reported_rate),
        "final_score_reported_2dp": match.group("rounded"),
        "score_recomputed_from_reported_8dp_components": (
            format(recomputed, "f")
        ),
    }


def _run_upstream_eval(
    *,
    config: SingleStageScoreAttemptConfigV1 | RankZeroScoreAttemptConfigV1,
    submission_dir: Path,
    run_root: Path,
    archive_bytes: int,
) -> tuple[Path, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    upstream = repo_root / "upstream"
    report_path = run_root / "upstream_evaluate_report.txt"
    eval_receipt_path = run_root / "eval_receipt.json"
    if eval_receipt_path.exists():
        receipt = _open_sealed_json(
            eval_receipt_path,
            schema=EVAL_RECEIPT_SCHEMA,
            hash_field="eval_receipt_sha256",
        )
        report_binding = _validate_binding(
            receipt.get("report"),
            name="resumed upstream report",
        )
        report = Path(str(report_binding["path"]))
        parsed = _parse_report(
            report.read_text("utf-8"),
            archive_bytes=archive_bytes,
        )
        if (
            receipt.get("parsed") != parsed
            or receipt.get("authority_axis") != config.authority_axis
            or receipt.get("evaluator_device")
            != config.evaluator_device
            or receipt.get("returncode") != 0
        ):
            raise SingleStageScoreAttemptError(
                "resumed upstream eval receipt differs"
            )
        return report, receipt
    stdout_path = run_root / "upstream_evaluate.stdout.txt"
    stderr_path = run_root / "upstream_evaluate.stderr.txt"
    process = subprocess.run(
        [
            sys.executable,
            str(upstream / "evaluate.py"),
            "--batch-size",
            str(BATCH_PAIRS),
            "--submission-dir",
            str(submission_dir),
            "--uncompressed-dir",
            str(upstream / "videos"),
            "--report",
            str(report_path),
            "--video-names-file",
            str(config.video_names_file.expanduser().resolve()),
            "--device",
            config.evaluator_device,
            "--seed",
            "1234",
        ],
        cwd=upstream,
        capture_output=True,
        check=False,
    )
    _write_once(stdout_path, process.stdout)
    _write_once(stderr_path, process.stderr)
    if process.returncode != 0:
        raise SingleStageScoreAttemptError(
            "frozen upstream evaluate.py failed"
        )
    report = _regular_file(
        report_path,
        name="upstream evaluate report",
    )
    parsed = _parse_report(
        report.read_text("utf-8"),
        archive_bytes=archive_bytes,
    )
    body: dict[str, object] = {
        "schema": EVAL_RECEIPT_SCHEMA,
        "authority_axis": config.authority_axis,
        "evaluator_device": config.evaluator_device,
        "upstream_evaluate_py": _binding(upstream / "evaluate.py"),
        "upstream_frame_utils_py": _binding(
            upstream / "frame_utils.py"
        ),
        "upstream_modules_py": _binding(upstream / "modules.py"),
        "upstream_posenet_weights": _binding(
            upstream / "models" / "posenet.safetensors"
        ),
        "upstream_segnet_weights": _binding(
            upstream / "models" / "segnet.safetensors"
        ),
        "submission_archive": _binding(
            submission_dir / ARCHIVE_BASENAME
        ),
        "report": _binding(report),
        "stdout": _binding(stdout_path),
        "stderr": _binding(stderr_path),
        "returncode": process.returncode,
        "parsed": parsed,
        "upstream_evaluate_py_run": True,
        "authority_eligible": (
            config.authority_axis in {"contest-CPU", "contest-CUDA"}
        ),
        "pointer_moved": False,
    }
    receipt = _seal(body, field="eval_receipt_sha256")
    _write_json_once(eval_receipt_path, receipt)
    return report, receipt


def run_rank_zero_score_attempt(
    *,
    config: RankZeroScoreAttemptConfigV1,
    command: Sequence[str],
    allowed_output_roots: Sequence[Path] = SSD_ROOTS,
) -> SingleStageScoreAttemptResultV1:
    """Score the exact receiver-valid rank-zero archive already sealed by G120."""

    config = _validate_rank_zero_config(config)
    if os.environ.get(g119.GOVERNED_MARKER_ENV) != "1":
        raise SingleStageScoreAttemptError(
            "full n600 rank-zero attempt requires governed safe_run admission"
        )
    run_root = _validate_output_root(
        config.resume_from,
        allowed_output_roots=allowed_output_roots,
        minimum_free_bytes=max(
            MIN_RELEASE_FREE_BYTES,
            EXPECTED_RAW_BYTES + (8 << 30),
        ),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    config_document = _rank_zero_config_document(
        config,
        command=command,
    )
    config_path = run_root / "00_attempt_config.json"
    _write_json_once(config_path, config_document)
    selected_row = _open_selected_g121_row(
        config=config,
        run_root=run_root,
    )
    submission_dir, release_receipt = (
        _materialize_rank_zero_release(
            config=config,
            run_root=run_root,
            selected_row=selected_row,
            command=command,
        )
    )
    archive_binding = _binding(
        submission_dir / ARCHIVE_BASENAME
    )
    video_names = _regular_file(
        config.video_names_file,
        name="video names file",
    )
    decode1 = _run_decode(
        submission_dir=submission_dir,
        video_names_file=video_names,
        run_root=run_root,
        decode_index=1,
    )
    decode2 = _run_decode(
        submission_dir=submission_dir,
        video_names_file=video_names,
        run_root=run_root,
        decode_index=2,
    )
    if decode1["raw_tree"] != decode2["raw_tree"]:
        raise SingleStageScoreAttemptError(
            "rank-zero public double decode changed exact raw bytes"
        )
    report_path, eval_receipt = _run_upstream_eval(
        config=config,
        submission_dir=submission_dir,
        run_root=run_root,
        archive_bytes=int(archive_binding["bytes"]),
    )
    parsed = eval_receipt["parsed"]
    if type(parsed) is not dict:
        raise AssertionError(
            "validated rank-zero eval receipt lost parsed results"
        )
    recomputed = Decimal(
        str(parsed["score_recomputed_from_reported_8dp_components"])
    )
    final_body: dict[str, object] = {
        "schema": RANK_ZERO_FINAL_RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "config": _binding(config_path),
        "g121_selected_row": _binding(
            run_root / "01_selected_g121_row.json"
        ),
        "g121_attempt_identity_sha256": (
            selected_row["attempt_identity_sha256"]
        ),
        "attempt_mode": "G120_SELECTED_RANK_ZERO_ARCHIVE",
        "release_receipt_sha256": (
            release_receipt["release_receipt_sha256"]
        ),
        "submission_dir": str(submission_dir),
        "archive": archive_binding,
        "decode1_receipt": _binding(
            run_root / "decode1_receipt.json"
        ),
        "decode2_receipt": _binding(
            run_root / "decode2_receipt.json"
        ),
        "double_decode_bit_identical": True,
        "raw_tree": decode2["raw_tree"],
        "eval_receipt": _binding(run_root / "eval_receipt.json"),
        "upstream_report": _binding(report_path),
        "upstream_components": parsed,
        "authority_axis": config.authority_axis,
        "competitive_target": config.competitive_target,
        "recomputed_score_below_competitive_target": (
            recomputed < Decimal(config.competitive_target)
        ),
        "pose_refit_run": False,
        "gauss_newton_stages": 0,
        "exhaustive_stage_coverage_claim": False,
        "pareto_claim": False,
        "cross_stage_winner_claim": False,
        "upstream_evaluate_py_run": True,
        "receiver_closed": True,
        "authority_eligible": (
            config.authority_axis in {"contest-CPU", "contest-CUDA"}
        ),
        "candidate_claim": False,
        "score_claim": (
            config.authority_axis in {"contest-CPU", "contest-CUDA"}
        ),
        "pointer_moved": False,
    }
    final_receipt = _seal(
        final_body,
        field="final_receipt_sha256",
    )
    final_path = run_root / "99_final_receipt.json"
    _write_json_once(final_path, final_receipt)
    return SingleStageScoreAttemptResultV1(
        final_receipt_path=final_path,
        final_receipt_sha256=_sha256_file(final_path),
        submission_dir=submission_dir,
        archive_sha256=str(archive_binding["sha256"]),
        archive_bytes=int(archive_binding["bytes"]),
        report_path=report_path,
        d_pose=float(str(parsed["d_pose_reported_8dp"])),
        d_seg=float(str(parsed["d_seg_reported_8dp"])),
        recomputed_score=float(recomputed),
        authority_axis=config.authority_axis,
    )


def run_single_stage_score_attempt(
    *,
    config: SingleStageScoreAttemptConfigV1,
    command: Sequence[str],
    allowed_output_roots: Sequence[Path] = SSD_ROOTS,
) -> SingleStageScoreAttemptResultV1:
    """Run or resume the explicit baseline-only score spine."""

    config = _validate_config(config)
    if os.environ.get(g119.GOVERNED_MARKER_ENV) != "1":
        raise SingleStageScoreAttemptError(
            "full n600 attempt requires governed safe_run admission"
        )
    run_root = _validate_output_root(
        config.resume_from,
        allowed_output_roots=allowed_output_roots,
        minimum_free_bytes=max(
            MIN_RELEASE_FREE_BYTES,
            EXPECTED_RAW_BYTES + (8 << 30),
        ),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    config_document = _config_document(config, command=command)
    config_path = run_root / "00_attempt_config.json"
    _write_json_once(config_path, config_document)
    config_sha256 = str(config_document["config_sha256"])
    target_payload = _stable_read(
        config.target_capsule_receipt,
        expected_sha256=config.expected_target_capsule_receipt_sha256,
        name="G109 target capsule receipt",
    )
    target_path = config.target_capsule_receipt.expanduser().resolve()
    target_binding = {
        "path": str(target_path),
        "bytes": len(target_payload),
        "sha256": _sha256(target_payload),
    }
    selected_row = _open_selected_g121_row(
        config=config,
        run_root=run_root,
    )
    refit, baseline_receipt = _run_pose_baseline(
        config=config,
        config_path=config_path,
        config_sha256=config_sha256,
        selected_row=selected_row,
        target_binding=target_binding,
        run_root=run_root,
        command=command,
    )
    submission_dir, compiled, release_receipt = _materialize_release(
        config=config,
        run_root=run_root,
        selected_row=selected_row,
        target_binding=target_binding,
        refit=refit,
        baseline_receipt=baseline_receipt,
        command=command,
    )
    video_names = _regular_file(
        config.video_names_file,
        name="video names file",
    )
    decode1 = _run_decode(
        submission_dir=submission_dir,
        video_names_file=video_names,
        run_root=run_root,
        decode_index=1,
    )
    decode2 = _run_decode(
        submission_dir=submission_dir,
        video_names_file=video_names,
        run_root=run_root,
        decode_index=2,
    )
    if decode1["raw_tree"] != decode2["raw_tree"]:
        raise SingleStageScoreAttemptError(
            "public double decode changed exact raw bytes"
        )
    report_path, eval_receipt = _run_upstream_eval(
        config=config,
        submission_dir=submission_dir,
        run_root=run_root,
        archive_bytes=compiled.archive_bytes,
    )
    parsed = eval_receipt["parsed"]
    if type(parsed) is not dict:
        raise AssertionError("validated eval receipt lost parsed results")
    recomputed = Decimal(
        str(parsed["score_recomputed_from_reported_8dp_components"])
    )
    competitive_target = Decimal(config.competitive_target)
    final_body: dict[str, object] = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "config": _binding(config_path),
        "g121_selected_row": _binding(
            run_root / "01_selected_g121_row.json"
        ),
        "g121_attempt_identity_sha256": (
            selected_row["attempt_identity_sha256"]
        ),
        "pose_baseline_receipt": _binding(
            refit.audit_receipt_path
        ),
        "release_receipt_sha256": (
            release_receipt["release_receipt_sha256"]
        ),
        "submission_dir": str(submission_dir),
        "archive": _binding(submission_dir / ARCHIVE_BASENAME),
        "decode1_receipt": _binding(
            run_root / "decode1_receipt.json"
        ),
        "decode2_receipt": _binding(
            run_root / "decode2_receipt.json"
        ),
        "double_decode_bit_identical": True,
        "raw_tree": decode2["raw_tree"],
        "eval_receipt": _binding(run_root / "eval_receipt.json"),
        "upstream_report": _binding(report_path),
        "upstream_components": parsed,
        "authority_axis": config.authority_axis,
        "competitive_target": config.competitive_target,
        "recomputed_score_below_competitive_target": (
            recomputed < competitive_target
        ),
        "explicit_single_stage_attempt": True,
        "baseline_only_pose_measurement": True,
        "gauss_newton_stages": 0,
        "exhaustive_stage_coverage_claim": False,
        "pareto_claim": False,
        "cross_stage_winner_claim": False,
        "upstream_evaluate_py_run": True,
        "receiver_closed": True,
        "authority_eligible": (
            config.authority_axis in {"contest-CPU", "contest-CUDA"}
        ),
        "candidate_claim": False,
        "score_claim": (
            config.authority_axis in {"contest-CPU", "contest-CUDA"}
        ),
        "pointer_moved": False,
    }
    final_receipt = _seal(
        final_body,
        field="final_receipt_sha256",
    )
    final_path = run_root / "99_final_receipt.json"
    _write_json_once(final_path, final_receipt)
    return SingleStageScoreAttemptResultV1(
        final_receipt_path=final_path,
        final_receipt_sha256=_sha256_file(final_path),
        submission_dir=submission_dir,
        archive_sha256=compiled.archive_sha256,
        archive_bytes=compiled.archive_bytes,
        report_path=report_path,
        d_pose=float(str(parsed["d_pose_reported_8dp"])),
        d_seg=float(str(parsed["d_seg_reported_8dp"])),
        recomputed_score=float(recomputed),
        authority_axis=config.authority_axis,
    )


def write_blocker_receipt(
    *,
    resume_from: Path,
    command: Sequence[str],
    error: BaseException,
) -> Path | None:
    """Best-effort durable fail-closed blocker without weakening the error."""

    root = resume_from.expanduser().resolve()
    roots = tuple(path.expanduser().resolve() for path in SSD_ROOTS)
    if not any(root == item or item in root.parents for item in roots):
        return None
    try:
        root.mkdir(parents=True, exist_ok=True)
        body: dict[str, object] = {
            "schema": BLOCKER_SCHEMA,
            "resume_from": str(root),
            "command": [str(token) for token in command],
            "error_type": type(error).__name__,
            "error": str(error),
            "prerequisite_closed": False,
            "receiver_closed": False,
            "upstream_evaluate_py_run": False,
            "exhaustive_stage_coverage_claim": False,
            "pareto_claim": False,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        }
        receipt = _seal(body, field="blocker_receipt_sha256")
        path = root / "00_fail_closed_blocker.json"
        _write_json_once(path, receipt)
        return path
    except (OSError, ValueError, SingleStageScoreAttemptError):
        return None


__all__ = [
    "BASELINE_RECEIPT_SCHEMA",
    "BASELINE_STATE_SCHEMA",
    "BLOCKER_SCHEMA",
    "CONFIG_SCHEMA",
    "DECODE_RECEIPT_SCHEMA",
    "EVAL_RECEIPT_SCHEMA",
    "FINAL_RECEIPT_SCHEMA",
    "PAIR_COUNT",
    "RANK_ZERO_CONFIG_SCHEMA",
    "RANK_ZERO_FINAL_RECEIPT_SCHEMA",
    "RANK_ZERO_RELEASE_RECEIPT_SCHEMA",
    "RELEASE_RECEIPT_SCHEMA",
    "SELECTED_ROW_SCHEMA",
    "RankZeroScoreAttemptConfigV1",
    "SingleStageScoreAttemptConfigV1",
    "SingleStageScoreAttemptError",
    "SingleStageScoreAttemptResultV1",
    "run_rank_zero_score_attempt",
    "run_single_stage_score_attempt",
    "write_blocker_receipt",
]
