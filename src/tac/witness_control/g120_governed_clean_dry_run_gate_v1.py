# SPDX-License-Identifier: MIT
"""Governed no-scorer admission gate for G120-v2 and the G121 monitor."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.witness_control import (
    taskspace_g121_resumable_stage_harvest_v1 as g121,
)
from tac.witness_dsl import (
    g120_parsed_stage_production_authority_v2 as g120,
)

SCHEMA: Final = "tac.g120_governed_clean_dry_run.v1"
BINDING_SCHEMA: Final = "tac.g120_governed_clean_dry_run_binding.v1"
CHECKPOINT_SCHEMA: Final = "tac.g120_governed_clean_dry_run_checkpoint.v1"
COMPLETION_BASENAME: Final = "g120_governed_clean_dry_run_receipt.json"
BINDING_BASENAME: Final = "g120_governed_clean_dry_run_binding.json"
CHECKPOINT_BASENAME: Final = "g120_governed_clean_dry_run_checkpoint.json"
PROBE_DIRNAME: Final = "batch_resume_probe"
CAMERA_HW: Final = (874, 1164)
MINIMUM_FREE_BYTES: Final = (
    2 * g120.PIXEL_DENOMINATOR
    + 4
    * g120.PRODUCTION_BATCH_PAIRS
    * (
        g120.PRODUCTION_SEG_HW[0] * g120.PRODUCTION_SEG_HW[1] * 3
        + CAMERA_HW[0] * CAMERA_HW[1] * 3
    )
)
_SSD_ROOTS: tuple[Path, ...] = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
_LOWER_SHA256: Final = frozenset("0123456789abcdef")


class G120GovernedDryRunError(RuntimeError):
    """The no-scorer production admission gate failed closed."""


@dataclass(frozen=True, slots=True)
class G120GovernedDryRunResultV1:
    phase: str
    receipt_path: Path
    receipt_sha256: str
    clean_dry_run_complete: bool
    scorer_calls: int = 0
    score_authority: bool = False


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G120GovernedDryRunError(
            "G120 dry-run value is not canonical finite ASCII JSON"
        ) from exc


def _sha256(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise G120GovernedDryRunError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _stable_file(path: Path, *, name: str) -> tuple[bytes, dict[str, Any]]:
    try:
        return g120._stable_file(path, name=name)
    except g120.G120ProductionAuthorityV2Error as exc:
        raise G120GovernedDryRunError(str(exc)) from exc


def _immutable_write(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        return g120._immutable_write(path, payload)
    except g120.G120ProductionAuthorityV2Error as exc:
        raise G120GovernedDryRunError(str(exc)) from exc


def _durable_directory(path: Path, *, name: str) -> Path:
    try:
        return g121._durable_directory(path, name=name)
    except g121.G121StageHarvestError as exc:
        raise G120GovernedDryRunError(str(exc)) from exc


def _source_binding(path: Path, *, name: str) -> dict[str, Any]:
    _payload, binding = _stable_file(path.resolve(), name=name)
    return binding


def _source_bindings(repo_root: Path) -> dict[str, dict[str, Any]]:
    return {
        "g120_production_wrapper": _source_binding(
            Path(g120.__file__).resolve(),
            name="G120-v2 production source",
        ),
        "g120_dry_run_gate": _source_binding(
            Path(__file__).resolve(),
            name="G120 dry-run gate source",
        ),
        "g121_harvester": _source_binding(
            Path(g121.__file__).resolve(),
            name="G121 harvester source",
        ),
        "g121_live_monitor": _source_binding(
            (repo_root / "tools/run_taskspace_g121_live_stage_harvest.py").resolve(),
            name="G121 live monitor source",
        ),
    }


def _ssd_root(path: Path) -> Path:
    resolved = path.resolve()
    for root in _SSD_ROOTS:
        if (
            root.is_dir()
            and not root.is_symlink()
            and resolved != root.resolve()
            and resolved.is_relative_to(root.resolve())
        ):
            return root.resolve()
    raise G120GovernedDryRunError(
        f"{path} is outside the configured SSD pact roots"
    )


def _storage_preflight(
    paths: tuple[Path, ...],
) -> dict[str, Any]:
    roots = {_ssd_root(path) for path in paths}
    if len(roots) != 1:
        raise G120GovernedDryRunError(
            "producer and all G120/G121 evidence paths must share one SSD root"
        )
    root = next(iter(roots))
    rows: list[dict[str, Any]] = []
    for path in paths:
        usage = shutil.disk_usage(path)
        stat = path.stat(follow_symlinks=False)
        row = {
            "path": str(path.resolve()),
            "device": stat.st_dev,
            "free_bytes": usage.free,
            "required_free_bytes": MINIMUM_FREE_BYTES,
            "passed": usage.free >= MINIMUM_FREE_BYTES,
        }
        if row["passed"] is not True:
            raise G120GovernedDryRunError(
                f"storage preflight has only {usage.free} free bytes at {path}; "
                f"requires {MINIMUM_FREE_BYTES}"
            )
        rows.append(row)
    return {
        "ssd_root": str(root),
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "bound_equation": (
            "2*N*384*512+4*16*(384*512*3+874*1164*3)"
        ),
        "paths": rows,
        "passed": True,
    }


def _storage_contract(
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ssd_root": preflight["ssd_root"],
        "minimum_free_bytes": preflight[
            "minimum_free_bytes"
        ],
        "bound_equation": preflight["bound_equation"],
        "paths": [
            {
                "path": row["path"],
                "device": row["device"],
            }
            for row in preflight["paths"]
        ],
    }


def _binding_body(
    *,
    repo_root: Path,
    producer_run_dir: Path,
    expected_launch_manifest_sha256: str,
    monitor_output_dir: Path,
    monitor_progress_dir: Path,
    measurement_cache_dir: Path,
    gate_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    producer = _durable_directory(
        producer_run_dir,
        name="producer_run_dir",
    )
    output = _durable_directory(
        monitor_output_dir,
        name="monitor_output_dir",
    )
    progress = _durable_directory(
        monitor_progress_dir,
        name="monitor_progress_dir",
    )
    cache = _durable_directory(
        measurement_cache_dir,
        name="measurement_cache_dir",
    )
    gate = _durable_directory(gate_dir, name="gate_dir")
    if len({producer, output, progress, cache, gate}) != 5:
        raise G120GovernedDryRunError(
            "producer/output/progress/cache/gate directories must be distinct"
        )
    expected_launch = _require_sha256(
        expected_launch_manifest_sha256,
        name="expected launch manifest",
    )
    try:
        launch_binding, launch_compile_sha = (
            g121._open_governed_launch_manifest(
                producer,
                expected_sha256=expected_launch,
            )
        )
    except g121.G121StageHarvestError as exc:
        raise G120GovernedDryRunError(str(exc)) from exc
    storage = _storage_preflight(
        (producer, output, progress, cache, gate)
    )
    body = {
        "schema": BINDING_SCHEMA,
        "repo_root": str(repo_root.resolve()),
        "producer_run_dir": str(producer),
        "launch_manifest": launch_binding,
        "launch_dsl_compile_sha256": launch_compile_sha,
        "monitor_output_dir": str(output),
        "monitor_progress_dir": str(progress),
        "measurement_cache_dir": str(cache),
        "gate_dir": str(gate),
        "storage_contract": _storage_contract(storage),
        "source_bindings": _source_bindings(repo_root.resolve()),
        "no_scorer_execution": True,
        "semantic_measurement_emitted": False,
    }
    body["binding_identity_sha256"] = _sha256(_canonical_json(body))
    return body, {
        "producer": producer,
        "output": output,
        "progress": progress,
        "cache": cache,
        "gate": gate,
        "storage_preflight": storage,
    }


def _write_or_reopen_binding(
    *,
    body: dict[str, Any],
    gate: Path,
) -> dict[str, Any]:
    path = gate / BINDING_BASENAME
    payload = _canonical_json(body)
    binding = _immutable_write(path, payload)
    reopened, observed = _stable_file(
        path,
        name="G120 dry-run fixed binding",
    )
    if reopened != payload or observed != binding:
        raise G120GovernedDryRunError(
            "G120 dry-run fixed binding changed"
        )
    return observed


def _probe_coordinates(
    binding_identity_sha256: str,
) -> tuple[str, np.ndarray, np.ndarray, str, str]:
    execution_key = _sha256(
        _canonical_json(
            {
                "schema": g120.BATCH_SCHEMA,
                "dry_run_binding_identity_sha256": (
                    binding_identity_sha256
                ),
                "batch_index": 0,
                "pair_start": 0,
                "pair_stop": g120.PRODUCTION_BATCH_PAIRS,
                "pair_count": g120.PRODUCTION_PAIR_COUNT,
                "seg_hw": list(g120.PRODUCTION_SEG_HW),
                "no_scorer_execution": True,
            }
        )
    )
    target = np.zeros(
        (
            g120.PRODUCTION_BATCH_PAIRS,
            *g120.PRODUCTION_SEG_HW,
        ),
        dtype=np.uint8,
    )
    predicted = np.zeros_like(target)
    scorer_sha = _sha256(
        b"G120_GOVERNED_DRY_RUN_SCORER_Y1_BATCH"
    )
    camera_sha = _sha256(
        b"G120_GOVERNED_DRY_RUN_CAMERA_Y1_BATCH"
    )
    return execution_key, target, predicted, scorer_sha, camera_sha


def _checkpoint_phase(
    *,
    body: dict[str, Any],
    binding_file: dict[str, Any],
    gate: Path,
    storage_preflight: dict[str, Any],
) -> G120GovernedDryRunResultV1:
    completion = gate / COMPLETION_BASENAME
    if completion.exists() or completion.is_symlink():
        raise G120GovernedDryRunError(
            "dry-run completion already exists; use a fresh gate directory"
        )
    checkpoint_path = gate / CHECKPOINT_BASENAME
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise G120GovernedDryRunError(
            "dry-run checkpoint already exists; run the resume phase"
        )
    probe_root = _durable_directory(
        gate / PROBE_DIRNAME,
        name="G120 dry-run batch probe directory",
    )
    execution_key, target, predicted, scorer_sha, camera_sha = (
        _probe_coordinates(body["binding_identity_sha256"])
    )
    try:
        batch = g120._persist_prediction_batch(
            receipt_path=probe_root / "batch_000_000_016.receipt.json",
            prediction_path=(
                probe_root
                / "batch_000_000_016.predicted_labels.npy"
            ),
            execution_key_sha256=execution_key,
            batch_index=0,
            pair_start=0,
            pair_stop=g120.PRODUCTION_BATCH_PAIRS,
            target_batch=target,
            scorer_y1_batch_sha256=scorer_sha,
            camera_y1_batch_sha256=camera_sha,
            predicted=predicted,
        )
    except g120.G120ProductionAuthorityV2Error as exc:
        raise G120GovernedDryRunError(str(exc)) from exc
    checkpoint: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_identity_sha256": None,
        "binding_file": binding_file,
        "binding_identity_sha256": body[
            "binding_identity_sha256"
        ],
        "checkpoint_pid": os.getpid(),
        "execution_key_sha256": execution_key,
        "batch": batch,
        "storage_preflight": storage_preflight,
        "production_batch_pairs": g120.PRODUCTION_BATCH_PAIRS,
        "production_seg_hw": list(g120.PRODUCTION_SEG_HW),
        "scorer_callable_supplied": False,
        "scorer_calls": 0,
        "restart_required": True,
    }
    checkpoint["checkpoint_identity_sha256"] = _sha256(
        _canonical_json(
            {
                key: value
                for key, value in checkpoint.items()
                if key != "checkpoint_identity_sha256"
            }
        )
    )
    checkpoint_binding = _immutable_write(
        checkpoint_path,
        _canonical_json(checkpoint),
    )
    return G120GovernedDryRunResultV1(
        phase="checkpoint",
        receipt_path=checkpoint_path,
        receipt_sha256=str(checkpoint_binding["sha256"]),
        clean_dry_run_complete=False,
    )


def _open_checkpoint(
    gate: Path,
    *,
    body: dict[str, Any],
    binding_file: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = gate / CHECKPOINT_BASENAME
    raw, physical = _stable_file(
        path,
        name="G120 dry-run checkpoint",
    )
    try:
        checkpoint = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G120GovernedDryRunError(
            "G120 dry-run checkpoint is corrupt"
        ) from exc
    claimed = (
        checkpoint.get("checkpoint_identity_sha256")
        if type(checkpoint) is dict
        else None
    )
    checkpoint_body = (
        {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_identity_sha256"
        }
        if type(checkpoint) is dict
        else {}
    )
    if (
        type(checkpoint) is not dict
        or _canonical_json(checkpoint) != raw
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or claimed != _sha256(_canonical_json(checkpoint_body))
        or checkpoint.get("binding_file") != binding_file
        or checkpoint.get("binding_identity_sha256")
        != body["binding_identity_sha256"]
        or checkpoint.get("checkpoint_pid") == os.getpid()
        or checkpoint.get("production_batch_pairs")
        != g120.PRODUCTION_BATCH_PAIRS
        or checkpoint.get("production_seg_hw")
        != list(g120.PRODUCTION_SEG_HW)
        or checkpoint.get("scorer_callable_supplied") is not False
        or checkpoint.get("scorer_calls") != 0
        or checkpoint.get("restart_required") is not True
        or type(checkpoint.get("storage_preflight")) is not dict
        or checkpoint["storage_preflight"].get("passed") is not True
        or _storage_contract(checkpoint["storage_preflight"])
        != body["storage_contract"]
    ):
        raise G120GovernedDryRunError(
            "G120 dry-run checkpoint identity/process binding differs"
        )
    return checkpoint, physical


def _resume_phase(
    *,
    body: dict[str, Any],
    binding_file: dict[str, Any],
    gate: Path,
    storage_preflight: dict[str, Any],
) -> G120GovernedDryRunResultV1:
    checkpoint, checkpoint_file = _open_checkpoint(
        gate,
        body=body,
        binding_file=binding_file,
    )
    probe_root = gate / PROBE_DIRNAME
    execution_key, target, _predicted, scorer_sha, camera_sha = (
        _probe_coordinates(body["binding_identity_sha256"])
    )
    try:
        reopened = g120._reopen_completed_prediction_batch(
            receipt_path=probe_root / "batch_000_000_016.receipt.json",
            prediction_path=(
                probe_root
                / "batch_000_000_016.predicted_labels.npy"
            ),
            expected_execution_key_sha256=execution_key,
            batch_index=0,
            pair_start=0,
            pair_stop=g120.PRODUCTION_BATCH_PAIRS,
            target_batch=target,
            scorer_y1_batch_sha256=scorer_sha,
            camera_y1_batch_sha256=camera_sha,
        )
    except g120.G120ProductionAuthorityV2Error as exc:
        raise G120GovernedDryRunError(str(exc)) from exc
    if reopened is None or reopened != checkpoint["batch"]:
        raise G120GovernedDryRunError(
            "G120 dry-run resume did not reopen the exact checkpoint batch"
        )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "receipt_identity_sha256": None,
        "clean_dry_run_complete": True,
        "binding": body,
        "binding_file": binding_file,
        "checkpoint_file": checkpoint_file,
        "checkpoint_identity_sha256": checkpoint[
            "checkpoint_identity_sha256"
        ],
        "checkpoint_pid": checkpoint["checkpoint_pid"],
        "resume_pid": os.getpid(),
        "batch_resume_proof": {
            "execution_key_sha256": execution_key,
            "batch": reopened,
            "production_batch_pairs": g120.PRODUCTION_BATCH_PAIRS,
            "production_seg_hw": list(g120.PRODUCTION_SEG_HW),
            "scorer_callable_supplied": False,
            "scorer_calls": 0,
            "completed_batch_reopened": True,
        },
        "storage_preflights": {
            "checkpoint": checkpoint["storage_preflight"],
            "resume": storage_preflight,
        },
        "authority": {
            "semantic_measurement_emitted": False,
            "contest_score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "heavy_scorer_run_launched": False,
        },
    }
    receipt["receipt_identity_sha256"] = _sha256(
        _canonical_json(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_identity_sha256"
            }
        )
    )
    path = gate / COMPLETION_BASENAME
    physical = _immutable_write(path, _canonical_json(receipt))
    return G120GovernedDryRunResultV1(
        phase="resume",
        receipt_path=path,
        receipt_sha256=str(physical["sha256"]),
        clean_dry_run_complete=True,
    )


def run_g120_governed_clean_dry_run_v1(
    *,
    phase: str,
    repo_root: Path,
    producer_run_dir: Path,
    expected_launch_manifest_sha256: str,
    monitor_output_dir: Path,
    monitor_progress_dir: Path,
    measurement_cache_dir: Path,
    gate_dir: Path,
) -> G120GovernedDryRunResultV1:
    """Run one of the two required no-scorer process phases."""

    if phase not in {"checkpoint", "resume"}:
        raise G120GovernedDryRunError(
            "dry-run phase must be checkpoint or resume"
        )
    if not repo_root.is_absolute() or not repo_root.is_dir():
        raise G120GovernedDryRunError(
            "repo_root must be an absolute physical directory"
        )
    body, paths = _binding_body(
        repo_root=repo_root,
        producer_run_dir=producer_run_dir,
        expected_launch_manifest_sha256=(
            expected_launch_manifest_sha256
        ),
        monitor_output_dir=monitor_output_dir,
        monitor_progress_dir=monitor_progress_dir,
        measurement_cache_dir=measurement_cache_dir,
        gate_dir=gate_dir,
    )
    binding_file = _write_or_reopen_binding(
        body=body,
        gate=paths["gate"],
    )
    if phase == "checkpoint":
        return _checkpoint_phase(
            body=body,
            binding_file=binding_file,
            gate=paths["gate"],
            storage_preflight=paths["storage_preflight"],
        )
    return _resume_phase(
        body=body,
        binding_file=binding_file,
        gate=paths["gate"],
        storage_preflight=paths["storage_preflight"],
    )


def open_g120_governed_clean_dry_run_v1(
    path: Path,
    *,
    expected_sha256: str,
    repo_root: Path,
    producer_run_dir: Path,
    expected_launch_manifest_sha256: str,
    monitor_output_dir: Path,
    monitor_progress_dir: Path,
    measurement_cache_dir: Path,
) -> dict[str, Any]:
    """Reopen the exact current-code admission receipt for monitor launch."""

    expected = _require_sha256(
        expected_sha256,
        name="expected G120 dry-run receipt",
    )
    raw, physical = _stable_file(
        path,
        name="G120 governed clean dry-run receipt",
    )
    if physical["sha256"] != expected:
        raise G120GovernedDryRunError(
            "G120 dry-run receipt differs from its external SHA-256"
        )
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G120GovernedDryRunError(
            "G120 dry-run receipt is corrupt"
        ) from exc
    claimed = (
        receipt.get("receipt_identity_sha256")
        if type(receipt) is dict
        else None
    )
    receipt_body = (
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_identity_sha256"
        }
        if type(receipt) is dict
        else {}
    )
    expected_body, paths = _binding_body(
        repo_root=repo_root,
        producer_run_dir=producer_run_dir,
        expected_launch_manifest_sha256=(
            expected_launch_manifest_sha256
        ),
        monitor_output_dir=monitor_output_dir,
        monitor_progress_dir=monitor_progress_dir,
        measurement_cache_dir=measurement_cache_dir,
        gate_dir=path.parent.resolve(),
    )
    proof = receipt.get("batch_resume_proof") if type(receipt) is dict else None
    authority = receipt.get("authority") if type(receipt) is dict else None
    storage = receipt.get("storage_preflights") if type(receipt) is dict else None
    if (
        type(receipt) is not dict
        or _canonical_json(receipt) != raw
        or receipt.get("schema") != SCHEMA
        or claimed != _sha256(_canonical_json(receipt_body))
        or receipt.get("clean_dry_run_complete") is not True
        or receipt.get("binding") != expected_body
        or receipt.get("checkpoint_pid") == receipt.get("resume_pid")
        or type(proof) is not dict
        or proof.get("completed_batch_reopened") is not True
        or proof.get("scorer_callable_supplied") is not False
        or proof.get("scorer_calls") != 0
        or proof.get("production_batch_pairs")
        != g120.PRODUCTION_BATCH_PAIRS
        or proof.get("production_seg_hw")
        != list(g120.PRODUCTION_SEG_HW)
        or type(storage) is not dict
        or set(storage) != {"checkpoint", "resume"}
        or any(
            type(storage.get(phase)) is not dict
            or storage[phase].get("passed") is not True
            or _storage_contract(storage[phase])
            != expected_body["storage_contract"]
            for phase in ("checkpoint", "resume")
        )
        or _storage_contract(paths["storage_preflight"])
        != expected_body["storage_contract"]
        or authority
        != {
            "semantic_measurement_emitted": False,
            "contest_score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "heavy_scorer_run_launched": False,
        }
    ):
        raise G120GovernedDryRunError(
            "G120 dry-run receipt does not bind the current production launch"
        )
    checkpoint_file = receipt.get("checkpoint_file")
    binding_file = receipt.get("binding_file")
    if type(checkpoint_file) is not dict or type(binding_file) is not dict:
        raise G120GovernedDryRunError(
            "G120 dry-run receipt lacks physical phase bindings"
        )
    _checkpoint_payload, reopened_checkpoint_file = _stable_file(
        Path(str(checkpoint_file.get("path"))),
        name="G120 dry-run checkpoint",
    )
    _binding_payload, reopened_binding_file = _stable_file(
        Path(str(binding_file.get("path"))),
        name="G120 dry-run fixed binding",
    )
    if (
        reopened_checkpoint_file != checkpoint_file
        or reopened_binding_file != binding_file
    ):
        raise G120GovernedDryRunError(
            "G120 dry-run phase receipt binding changed"
        )
    execution_key, target, _predicted, scorer_sha, camera_sha = (
        _probe_coordinates(expected_body["binding_identity_sha256"])
    )
    batch = proof.get("batch")
    if type(batch) is not dict:
        raise G120GovernedDryRunError(
            "G120 dry-run receipt lacks its resumed batch"
        )
    try:
        reopened = g120._reopen_completed_prediction_batch(
            receipt_path=Path(
                str(batch["physical_receipt"]["path"])
            ),
            prediction_path=Path(
                str(batch["prediction_file"]["path"])
            ),
            expected_execution_key_sha256=execution_key,
            batch_index=0,
            pair_start=0,
            pair_stop=g120.PRODUCTION_BATCH_PAIRS,
            target_batch=target,
            scorer_y1_batch_sha256=scorer_sha,
            camera_y1_batch_sha256=camera_sha,
        )
    except (KeyError, g120.G120ProductionAuthorityV2Error) as exc:
        raise G120GovernedDryRunError(
            "G120 dry-run physical resume proof changed"
        ) from exc
    if reopened != batch or paths["gate"] != path.parent.resolve():
        raise G120GovernedDryRunError(
            "G120 dry-run physical resume proof differs"
        )
    return receipt


__all__ = [
    "CHECKPOINT_BASENAME",
    "COMPLETION_BASENAME",
    "MINIMUM_FREE_BYTES",
    "SCHEMA",
    "G120GovernedDryRunError",
    "G120GovernedDryRunResultV1",
    "open_g120_governed_clean_dry_run_v1",
    "run_g120_governed_clean_dry_run_v1",
]
