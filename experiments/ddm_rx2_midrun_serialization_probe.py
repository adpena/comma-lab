#!/usr/bin/env python3
"""Real-byte, lossless mid-run serialization probe for the live RX2 trainer.

The canonical RX2 terminal race intentionally accepts only the epoch-60 QAT
checkpoint.  This scorer-free worker leaves that gate intact and imports the
race's real IHS1 pack, probability materialization, native RC64 encode/decode,
and whole-container build mechanisms for advisory measurements on immutable
copies of earlier checkpoints.

Every materialized payload is retained beneath the probe epoch directory.  A
probe can resume at frame and native-coder checkpoints, and the cadence loop
can reconstruct its state from the append-only receipt stream.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import lzma
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

race = importlib.import_module("experiments.ddm_rx2_mc36_identity_race")
rx1 = race.rx1
cp = race.cp

AP_RUN_ROOT = Path("/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac")
CHECKPOINT_ROOT = AP_RUN_ROOT / "training/checkpoints/mc36_hpac_best_ema.checkpoints"
PROBE_ROOT = AP_RUN_ROOT / "probes"
RECEIPT_STREAM = PROBE_ROOT / "serialization_probe.jsonl"
ALERT_PATH = PROBE_ROOT / "serialization_alert.json"

BYTE_AUTHORITY = "ADVISORY_MIDRUN_PROBE_NOT_CANONICAL"
SCORE_CLAIM = False
EXPECTED_TOKEN_SHA256 = rx1.EXPECTED_SPATIAL_SHA256
MODEL_TOKEN_BAR_BYTES = 186_073
ARCHIVE_BAR_BYTES = rx1.EXPECTED_ARCHIVE_BYTES
POLL_SECONDS = 120.0
MIN_FREE_GIB = 32

RECALL_EVIDENCE = {
    "stores_consulted": [
        ".omx/state/main_hot_state.md",
        ".omx/state/codex_arm_queue.jsonl",
        ".omx/state/codex_arm_queue.next_if_resumed.jsonl",
        ".omx/research/CANONICAL_RESEARCH_INDEX_20260629.md",
        ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md",
        ".omx/research/ddm_rx2_mc36_label_hpac_20260814.md",
        ".omx/research/ddm_wc1_hpac_throughput_port_20260814.md",
        "experiments/ddm_rx2_mc36_identity_race.py",
        "experiments/ddm_rx1_rate_representation_attack.py",
    ],
    "ownership": (
        "ddm_sp2_rx2_midrun_serialization_probe is the sole live owner of the "
        "mid-run real-byte lane; RX2 terminal harvest remains separately owned"
    ),
    "settled_constraints": [
        "fixed MC36 token labels make RX2 rate-only",
        "the terminal race epoch-60/QAT gate remains unchanged",
        "native RC64 decoded spatial-token SHA-256 is the losslessness authority",
        "all materialized payloads are retained on APDataStore",
    ],
    "findings_beyond_charter": [
        "the canonical equation registry has no dedicated HPAC serializer law; EMA run geometry is already settled",
        "WC1 found no completed HPAC throughput port and makes real IHS1 endpoint bytes, not estimated model bytes, the parity column",
        "XI owns context-extension work, so this probe keeps the RX2 probability mechanism unchanged",
    ],
    "plan_change": "retain whole-container bytes and avoid folding throughput or context experiments into this lane",
}


class ProbeError(RuntimeError):
    """Fail-closed error for a probe custody, byte, or identity gate."""


class ModelNotSerializable(ProbeError):
    """The real IHS1 pack cannot represent this checkpoint phase."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _authority() -> dict[str, Any]:
    return {"byte_authority": BYTE_AUTHORITY, "score_claim": SCORE_CLAIM}


def _atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return race.atomic_json(path, value)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode()
    with path.open("ab") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_receipts() -> list[dict[str, Any]]:
    if not RECEIPT_STREAM.is_file():
        return []
    rows = []
    for line_number, line in enumerate(RECEIPT_STREAM.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"malformed probe receipt JSONL line {line_number}") from exc
        if value.get("schema") == "ddm_rx2_midrun_serialization_probe.v1":
            rows.append(value)
    return rows


def _epoch_from_name(path: Path) -> int | None:
    matches = re.findall(r"(?:epoch[_-]?)(\d{1,4})", path.name)
    return int(matches[-1]) if matches else None


def _candidate_checkpoints() -> list[tuple[int, Path]]:
    candidates: dict[int, Path] = {}
    for path in sorted(CHECKPOINT_ROOT.rglob("*.pt")):
        if path.name.startswith("._") or path.name == "latest.pt" or ".tmp" in path.name:
            continue
        epoch = _epoch_from_name(path)
        if epoch is None:
            continue
        current = candidates.get(epoch)
        if current is None or path.stat().st_mtime_ns > current.stat().st_mtime_ns:
            candidates[epoch] = path
    return sorted(candidates.items())


def _cadence_due(epoch: int, completed_epochs: set[int]) -> bool:
    if epoch in completed_epochs:
        return False
    if not completed_epochs:
        return True
    last = max(completed_epochs)
    if epoch <= last:
        return False
    if epoch == 60:
        return True
    if 31 <= epoch <= 37:
        return epoch - last >= 2
    if last < 31 <= epoch:
        return True
    return epoch - last >= (5 if last == 1 else 6)


def _newest_due_checkpoint(completed_epochs: set[int]) -> tuple[int, Path] | None:
    due = [(epoch, path) for epoch, path in _candidate_checkpoints() if _cadence_due(epoch, completed_epochs)]
    return due[-1] if due else None


def _stable_copy(source: Path, destination: Path) -> dict[str, Any]:
    """Copy a checkpoint without ever loading the live source file."""

    if not source.is_file():
        raise ProbeError(f"checkpoint source is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = destination.with_suffix(".copy.json")
    if destination.exists():
        if not receipt_path.is_file():
            raise ProbeError(f"existing checkpoint copy lacks its custody receipt: {destination}")
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("copy") != race.file_record(destination):
            raise ProbeError(f"existing checkpoint copy failed custody: {destination}")
        return existing
    source_before = source.stat()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        source_after = source.stat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(source_before, key) != getattr(source_after, key) for key in stable_fields):
            raise ProbeError(f"checkpoint changed during immutable copy: {source}")
        if temporary.stat().st_size != source_after.st_size:
            raise ProbeError("checkpoint copy byte count differs from stable source")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = {
        "schema": "ddm_rx2_midrun_checkpoint_copy.v1",
        "copied_utc": _utc_now(),
        "source": race.file_record(source),
        "copy": race.file_record(destination),
        "source_stable_during_copy": True,
        **_authority(),
    }
    _atomic_json(receipt_path, result)
    return result


def _verify_checkpoint(checkpoint_path: Path, expected_epoch: int) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    profile = checkpoint.get("run_identity", {}).get("training_config", {}).get("profile")
    if (
        checkpoint.get("schema") != "ddm_cl1_hpac_capacity_checkpoint.v2"
        or checkpoint.get("epoch") != expected_epoch
        or checkpoint.get("deployment_weights") != "ema_shadow"
        or profile != "rx2_mc36"
    ):
        raise ProbeError("checkpoint copy is not the requested RX2 EMA checkpoint")
    trainer = importlib.import_module("tools.train_ddm_cl1_hpac_capacity")
    observed_causal = trainer._causal_state_sha256(checkpoint)
    if observed_causal != checkpoint.get("causal_state_sha256"):
        raise ProbeError("checkpoint causal-state SHA-256 does not verify")
    history = checkpoint.get("history")
    if not isinstance(history, list) or not history or history[-1].get("epoch") != expected_epoch:
        raise ProbeError("checkpoint lacks epoch-aligned surrogate telemetry")
    surrogate = history[-1]
    required = ("estimated_token_bytes", "estimated_model_bytes", "estimated_joint_bytes")
    if any(not isinstance(surrogate.get(name), int) for name in required):
        raise ProbeError("checkpoint surrogate byte telemetry is incomplete")
    identity = {
        "schema": "ddm_rx2_midrun_checkpoint_identity.v1",
        "checkpoint": race.file_record(checkpoint_path),
        "epoch": expected_epoch,
        "phase": checkpoint.get("phase"),
        "deployment_weights": checkpoint.get("deployment_weights"),
        "causal_state_sha256": observed_causal,
        "profile": profile,
        "surrogate": surrogate,
        **_authority(),
    }
    return checkpoint, identity


def _storage_preflight(root: Path, required_gib: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    required = required_gib * (1 << 30)
    if free < required:
        raise ProbeError(f"serialization probe requires {required} free bytes; observed {free}")
    return {
        "tier": str(root.resolve()),
        "free_bytes": free,
        "required_free_bytes": required,
        "status": "PASS",
        "cleanup": "certify-or-block; every materialized payload retained",
    }


def _set_race_roots(root: Path) -> None:
    race.BULK_ROOT = root
    race.WORK_ROOT = root


def _prepare_real_model(checkpoint_path: Path, root: Path, brotli: str) -> dict[str, Any]:
    """Apply the terminal race's exact pack and lossless representation race."""

    _set_race_roots(root)
    model_root = root / "retained/models/midrun"
    try:
        packed = race._pack_terminal_ihs1(checkpoint_path, model_root)
    except (ValueError, race.RX2RaceError) as exc:
        raise ModelNotSerializable(f"{type(exc).__name__}: {exc}") from exc
    ihs1_path = Path(packed["raw"]["path"])
    ihs1 = ihs1_path.read_bytes()
    if not ihs1.startswith(b"IHS1"):
        raise ProbeError("mid-run packed model is not canonical IHS1")
    representations = [
        {
            "name": "xz_custodied",
            "codec": "xz",
            "codec_id": rx1.RX1_CODEC_XZ,
            "payload": packed["xz"],
            "parseback_sha256": race.sha256_bytes(
                lzma.decompress(Path(packed["xz"]["path"]).read_bytes(), format=lzma.FORMAT_XZ)
            ),
        }
    ]
    for quality in range(12):
        payload = rx1._brotli(ihs1, quality, brotli)
        path = model_root / f"hpac.ihs1.br.q{quality}"
        race.atomic_bytes(path, payload)
        restored = rx1._brotli_restore(payload, brotli)
        if restored != ihs1:
            raise ProbeError(f"Brotli q{quality} IHS1 parse-back differs")
        representations.append(
            {
                "name": f"brotli_q{quality}",
                "codec": "brotli",
                "codec_id": rx1.RX1_CODEC_BROTLI,
                "payload": race.file_record(path),
                "parseback_sha256": race.sha256_bytes(restored),
            }
        )

    runtime_module = cp.load_runtime(race.BASE_RUNTIME)
    parts = runtime_module.read_residual_archive(race.BASE_ARCHIVE)
    fields, compact_section = rx1._hp4_fields(race.BASE_ARCHIVE)
    if len(fields) != 5 or len(parts.residual_payload) != race.TABLE_FULL_BYTES:
        raise ProbeError("frozen MC36 physical field accounting changed")
    semantic_stream, carrier_stream = fields[3], fields[4]
    if parts.residual_payload != b"RCF1" + compact_section:
        raise ProbeError("frozen MC36 residual compact accounting changed")
    frozen = root / "retained/models/mc36_frozen"
    frozen_records = {
        "semantic_stream": race.atomic_bytes(frozen / "semantic.br", semantic_stream),
        "carrier_stream": race.atomic_bytes(frozen / "carrier.br", carrier_stream),
        "base_residual": race.atomic_bytes(frozen / "residual.rcf1", parts.residual_payload),
        "base_tokens": race.atomic_bytes(frozen / "tokens.rc64", parts.token_stream),
    }

    adapted = root / "adapted_runtime"
    if not adapted.exists():
        shutil.copytree(race.BASE_RUNTIME, adapted)
    receiver = adapted / "runtime/residual_archive.py"
    race.atomic_bytes(receiver, rx1._patch_runtime(receiver.read_text(encoding="utf-8")).encode())
    winner = min(representations, key=lambda row: row["payload"]["bytes"])
    smoke_model = rx1.pack_rx1_model(
        Path(winner["payload"]["path"]).read_bytes(),
        semantic_stream,
        carrier_stream,
        codec_id=int(winner["codec_id"]),
        table_mode=rx1.RX1_TABLE_ON,
    )
    smoke_member = smoke_model + parts.residual_payload[4:] + parts.token_stream
    smoke_root = root / "retained/receiver_smoke"
    smoke_archive = rx1.deterministic_zip(smoke_member)
    smoke_records = {
        "model": race.atomic_bytes(smoke_root / "models.rx1m", smoke_model),
        "member": race.atomic_bytes(smoke_root / "p", smoke_member),
        "archive": race.atomic_bytes(smoke_root / "archive.zip", smoke_archive),
    }
    parsed = rx1._receiver_parseback_subprocess(adapted, smoke_root / "archive.zip", brotli_binary=brotli)
    expected = {
        "hpac_sha256": race.sha256_bytes(ihs1),
        "semantic_sha256": race.sha256_bytes(parts.semantic_blob),
        "carrier_sha256": race.sha256_bytes(parts.carrier_blob),
        "residual_sha256": race.sha256_bytes(parts.residual_payload),
        "token_sha256": race.sha256_bytes(parts.token_stream),
    }
    if any(parsed[name] != digest for name, digest in expected.items()):
        raise ProbeError("mid-run shipped-receiver model smoke parse-back differs")
    result = {
        "schema": "ddm_rx2_midrun_prepare.v1",
        "complete": True,
        "selected_checkpoint": race.file_record(checkpoint_path),
        "pack": packed,
        "ihs1": race.file_record(ihs1_path),
        "representations": representations,
        "representation_winner": winner,
        **frozen_records,
        "adapted_runtime_receiver": race.file_record(receiver),
        "receiver_smoke": {"artifacts": smoke_records, "parsed": parsed, "expected": expected, "exact": True},
        "all_materialized_payloads_retained": True,
        **_authority(),
    }
    _atomic_json(root / "PREPARE_RESULT.json", result)
    return result


def _install_neutral_table(root: Path) -> dict[str, Any]:
    payload = race.serialize_table(np.zeros((race.TABLE_STATES, race.CLASSES), dtype=np.int8), 1.0)
    path = root / "retained/tables/neutral.rcf1"
    race.atomic_bytes(path, payload)
    row = {
        "variant": "neutral",
        "kind": "neutral_control",
        "shrink": 0.0,
        "clip_scale": 1.0,
        "table": race.file_record(path),
    }
    result = {
        "schema": "ddm_rx2_midrun_neutral_table.v1",
        "complete": True,
        "candidates": [row],
        "selected_for_full_n600_real_rc64": ["neutral"],
        "scope": "neutral table only; carries the checkpoint model probability bytes without a fitted-table confound",
        "all_table_payloads_retained": True,
        **_authority(),
    }
    _atomic_json(root / "FIT_RESULT.json", result)
    return result


def _stage_args(variant: str, brotli: str, required_free_gib: int) -> SimpleNamespace:
    return SimpleNamespace(
        variant=variant,
        start_frame=0,
        end_frame=600,
        torch_threads=2,
        required_free_gib=required_free_gib,
        brotli=brotli,
    )


def _stamp_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(_authority())
    _atomic_json(path, value)
    return value


def _run_serialized_pipeline(
    prepared: dict[str, Any],
    root: Path,
    *,
    brotli: str,
    required_free_gib: int,
) -> dict[str, Any]:
    args = _stage_args("neutral", brotli, required_free_gib)
    exported = race.export_base(args)
    exported.update(_authority())
    _atomic_json(root / "retained/probabilities/base_float/EXPORT_RESULT.json", exported)
    _install_neutral_table(root)
    materialized = race.materialize_probabilities(args)
    materialized.update(_authority())
    _atomic_json(root / "retained/probabilities/neutral/EXPORT_RESULT.json", materialized)
    encoded = race.encode_rc64(args)
    encoded.update(_authority())
    _atomic_json(root / "retained/coders/neutral/RC64_RESULT.json", encoded)
    built = race.build(args)
    built.update(_authority())
    _atomic_json(root / "BUILD_RESULT_neutral.json", built)
    for candidate in built["candidates"]:
        result_path = Path(candidate["archive"]["path"]).parent / "RESULT.json"
        if result_path.is_file():
            _stamp_result(result_path)
    return {"prepared": prepared, "encoded": encoded, "built": built}


def _run_real_token_fallback(
    checkpoint_path: Path,
    root: Path,
    *,
    brotli: str,
    required_free_gib: int,
) -> dict[str, Any]:
    """Code real tokens from the unpacked checkpoint when IHS1 is unavailable.

    This is the same teacher-forced group order, int16 logit quantization, and
    native checkpointable RC64 coder/decoder as RX2.  Only the unavailable
    serialized-model stage is omitted and explicitly typed by the caller.
    """

    _set_race_roots(root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    packer = race._import_packer()
    model = packer.model_from_args(race._pack_args(), True)
    model.load_state_dict(checkpoint["state_dict"])
    deployed_view = True
    try:
        packer.set_deployed_bit_depths(model, True)
    except (ValueError, RuntimeError):
        packer.set_deployed_bit_depths(model, False)
        deployed_view = False
    model.eval()

    runtime_module = cp.load_runtime(race.BASE_RUNTIME)
    archive_module = importlib.import_module("runtime.residual_archive")
    renderer = runtime_module._load_renderer(race.BASE_RUNTIME / "cpr1")
    masks = renderer.group_masks(torch.device("cpu"))
    sparse = archive_module._sparse_class(race.BASE_RUNTIME / "cpr1")(
        model, renderer.EVAL_H, renderer.EVAL_W
    )
    importlib.import_module("runtime.hpac_inference").optimize_sparse_evaluator(sparse)
    group_positions = [np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)) for mask in masks]
    source = race._source()
    output = root / "retained/probabilities/neutral"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with torch.inference_mode():
        for frame in range(600):
            path = output / f"codes_{frame:04d}.npy"
            frame_receipt = output / f"codes_{frame:04d}.json"
            if path.is_file() and frame_receipt.is_file():
                record = rx1._frame_record(path, frame, "neutral")
                if json.loads(frame_receipt.read_text(encoding="utf-8")) != record:
                    raise ProbeError(f"fallback probability checkpoint changed at frame {frame}")
                continue
            events = source.frame(frame)
            previous_events = (
                np.zeros(race.EVENTS_PER_FRAME, dtype=np.uint8)
                if frame == 0
                else source.frame(frame - 1)
            )
            previous_np = (
                np.zeros((384, 512), dtype=np.uint8)
                if frame == 0
                else rx1.spatial_frame(previous_events, group_positions)
            )
            previous = torch.from_numpy(previous_np.astype(np.int64, copy=False))[None]
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame]), previous)
            frame_codes = np.empty((race.EVENTS_PER_FRAME, race.CLASSES), dtype=np.int16)
            offset = 0
            for group, positions in enumerate(group_positions):
                selected = sparse.selected_logits(current, context, group).cpu().numpy().astype(np.float32)
                end = offset + len(positions)
                symbols = events[offset:end]
                frame_codes[offset:end] = np.clip(
                    np.rint(selected * race.LOGIT_PRECISION), -32768, 32767
                ).astype(np.int16)
                current.reshape(-1)[torch.from_numpy(positions)] = torch.from_numpy(
                    symbols.astype(np.int64, copy=False)
                )
                offset = end
            if offset != race.EVENTS_PER_FRAME:
                raise ProbeError("fallback probability export did not consume one token frame")
            if not np.array_equal(current[0].numpy(), rx1.spatial_frame(events, group_positions)):
                raise ProbeError("fallback teacher-forced reconstruction changed MC36 labels")
            race.atomic_npy(path, frame_codes)
            race.atomic_json(frame_receipt, rx1._frame_record(path, frame, "neutral"))
            print(json.dumps({"fallback_probability_frame": frame + 1, "elapsed_s": time.time() - started}), flush=True)
    identity = {
        "schema": "ddm_rx2_midrun_unpacked_probability_identity.v1",
        "complete_n600": True,
        "completed_frames": 600,
        "variant": "neutral",
        "checkpoint": race.file_record(checkpoint_path),
        "source_event_order_sha256": source.digest(),
        "deployed_bit_depth_view": deployed_view,
        **_authority(),
    }
    identity_path = output / "PROBABILITY_IDENTITY.json"
    _atomic_json(identity_path, identity)
    export = {
        **identity,
        "probability_identity": race.file_record(identity_path),
        "all_probability_payloads_retained": True,
    }
    _atomic_json(output / "EXPORT_RESULT.json", export)
    _install_neutral_table(root)
    args = _stage_args("neutral", brotli, required_free_gib)
    encoded = race.encode_rc64(args)
    encoded.update(_authority())
    _atomic_json(root / "retained/coders/neutral/RC64_RESULT.json", encoded)
    return {"encoded": encoded, "deployed_bit_depth_view": deployed_view}


def _real_byte_summary(pipeline: dict[str, Any]) -> dict[str, Any]:
    encoded = pipeline["encoded"]
    built = pipeline["built"]
    winner = built["winner"]
    model_bytes = int(winner["model"]["bytes"])
    token_bytes = int(encoded["token_payload"]["bytes"])
    archive_bytes = int(winner["archive"]["bytes"])
    decoded_spatial = encoded["decoded_spatial_tokens"]
    lossless = (
        encoded.get("event_order_identity") is True
        and encoded.get("spatial_token_identity") is True
        and decoded_spatial["sha256"] == EXPECTED_TOKEN_SHA256
    )
    return {
        "model_container": winner["model"],
        "token_payload": encoded["token_payload"],
        "model_plus_token_bytes": model_bytes + token_bytes,
        "archive": winner["archive"],
        "archive_repeat": winner["repeat_archive"],
        "archive_bytes": archive_bytes,
        "representation": winner["representation"],
        "decoded_spatial_tokens": decoded_spatial,
        "decoded_spatial_token_sha256_expected": EXPECTED_TOKEN_SHA256,
        "lossless_roundtrip": lossless,
        "receiver_component_identity": winner.get("receiver_component_identity") is True,
        "repeat_byte_identical": winner.get("repeat_byte_identical") is True,
        "model_token_bar_bytes": MODEL_TOKEN_BAR_BYTES,
        "archive_bar_bytes": ARCHIVE_BAR_BYTES,
        "model_token_delta_vs_bar": model_bytes + token_bytes - MODEL_TOKEN_BAR_BYTES,
        "archive_delta_vs_bar": archive_bytes - ARCHIVE_BAR_BYTES,
    }


def _is_losslessness_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "symbol mismatch",
            "decoded-token digest differs",
            "token identity",
            "token parse-back",
            "component identity",
            "receiver parse-back changed",
            "deterministic archive repeat",
            "member parse-back differs",
        )
    )


def _retention_receipt(root: Path) -> dict[str, Any]:
    inventory = rx1.retention_inventory(root)
    inventory.update(_authority())
    _atomic_json(root / "RETENTION_INVENTORY.json", inventory)
    return {
        "inventory": race.file_record(root / "RETENTION_INVENTORY.json"),
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "all_materialized_payloads_retained": True,
    }


def _calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = []
    for row in rows:
        real = row.get("real_bytes") or {}
        surrogate = row.get("checkpoint_identity", {}).get("surrogate", {})
        if row.get("complete") and isinstance(real.get("archive_bytes"), int):
            points.append(
                (
                    int(row["epoch"]),
                    float(surrogate["estimated_joint_bytes"]),
                    float(real["archive_bytes"]),
                )
            )
    if not points:
        return {"point_count": 0, "terminal_verdict": "UNDETERMINED_NO_REAL_POINTS"}
    latest = points[-1]
    result: dict[str, Any] = {
        "point_count": len(points),
        "latest_archive_to_surrogate_ratio": latest[2] / latest[1],
        "fit_form": "real_archive_bytes ~= a * surrogate_estimated_joint_bytes + b",
        "terminal_epoch": 60,
        "terminal_verdict": "UNDETERMINED_INSUFFICIENT_SLOPE_POINTS",
    }
    if len(points) < 2:
        return result
    epochs = np.asarray([row[0] for row in points], dtype=np.float64)
    surrogate = np.asarray([row[1] for row in points], dtype=np.float64)
    real = np.asarray([row[2] for row in points], dtype=np.float64)
    design = np.column_stack([surrogate, np.ones_like(surrogate)])
    a, b = np.linalg.lstsq(design, real, rcond=None)[0]
    epoch_design = np.column_stack([epochs, np.ones_like(epochs)])
    slope, intercept = np.linalg.lstsq(epoch_design, surrogate, rcond=None)[0]
    projected_surrogate = slope * 60.0 + intercept
    projected_real = a * projected_surrogate + b
    result.update(
        {
            "a": float(a),
            "b": float(b),
            "observed_surrogate_bytes_per_epoch_slope": float(slope),
            "projected_terminal_surrogate_bytes": float(projected_surrogate),
            "projected_terminal_real_archive_bytes": float(projected_real),
            "terminal_verdict": (
                "FAILURE_KNOWN_EARLY_PROJECTED_ABOVE_BAR"
                if projected_real > ARCHIVE_BAR_BYTES
                else "PROJECTED_BELOW_BAR_NOT_A_WIN_UNTIL_REAL_ARCHIVE"
            ),
        }
    )
    return result


def _existing_receipt(epoch: int, checkpoint_sha256: str) -> dict[str, Any] | None:
    for row in reversed(_read_receipts()):
        if row.get("epoch") == epoch and row.get("checkpoint_copy", {}).get("copy", {}).get("sha256") == checkpoint_sha256:
            return row
    return None


def run_probe(
    epoch: int,
    checkpoint_copy: Path,
    checkpoint_copy_receipt: dict[str, Any],
    *,
    brotli: str,
    required_free_gib: int,
) -> dict[str, Any]:
    root = PROBE_ROOT / f"ep{epoch:04d}"
    started = time.time()
    _, checkpoint_identity = _verify_checkpoint(checkpoint_copy, epoch)
    prior = _existing_receipt(epoch, checkpoint_identity["checkpoint"]["sha256"])
    if prior is not None and prior.get("model_stage"):
        return prior
    preflight = _storage_preflight(root, required_free_gib)
    model_stage: dict[str, Any]
    try:
        prepared = _prepare_real_model(checkpoint_copy, root, brotli)
    except ModelNotSerializable as exc:
        model_stage = {
            "status": f"NOT_SERIALIZABLE_AT_PHASE_{checkpoint_identity['phase']}",
            "reason": str(exc),
            "estimate_substituted": False,
        }
        try:
            fallback = _run_real_token_fallback(
                checkpoint_copy, root, brotli=brotli, required_free_gib=required_free_gib
            )
        except (ValueError, RuntimeError, race.RX2RaceError, ProbeError) as fallback_exc:
            if not _is_losslessness_failure(fallback_exc):
                raise
            real_bytes = {
                "model_container": None,
                "token_payload": None,
                "archive": None,
                "lossless_roundtrip": False,
                "reason": f"{type(fallback_exc).__name__}: {fallback_exc}",
                "all_partial_payloads_retained": True,
            }
        else:
            encoded = fallback["encoded"]
            decoded_spatial = encoded["decoded_spatial_tokens"]
            lossless = (
                encoded.get("event_order_identity") is True
                and encoded.get("spatial_token_identity") is True
                and decoded_spatial["sha256"] == EXPECTED_TOKEN_SHA256
            )
            real_bytes = {
                "model_container": None,
                "token_payload": encoded["token_payload"],
                "archive": None,
                "decoded_spatial_tokens": decoded_spatial,
                "decoded_spatial_token_sha256_expected": EXPECTED_TOKEN_SHA256,
                "lossless_roundtrip": lossless,
                "deployed_bit_depth_view": fallback["deployed_bit_depth_view"],
                "reason": "real native-RC64 token bytes measured; unavailable model bytes were not estimated",
            }
        complete = False
    else:
        try:
            pipeline = _run_serialized_pipeline(
                prepared, root, brotli=brotli, required_free_gib=required_free_gib
            )
        except (ValueError, RuntimeError, race.RX2RaceError, ProbeError) as exc:
            if not _is_losslessness_failure(exc):
                raise
            model_stage = {"status": "SERIALIZED", "pack": prepared["pack"]}
            real_bytes = {
                "model_container": None,
                "token_payload": None,
                "archive": None,
                "lossless_roundtrip": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "all_partial_payloads_retained": True,
            }
            complete = False
        else:
            model_stage = {"status": "SERIALIZED", "pack": pipeline["prepared"]["pack"]}
            real_bytes = _real_byte_summary(pipeline)
            complete = True
    retention = _retention_receipt(root)
    surrogate = checkpoint_identity["surrogate"]
    if isinstance(real_bytes.get("model_plus_token_bytes"), int):
        ratio = real_bytes["model_plus_token_bytes"] / surrogate["estimated_joint_bytes"]
    else:
        ratio = None
    result = {
        "schema": "ddm_rx2_midrun_serialization_probe.v1",
        "generated_utc": _utc_now(),
        "complete": complete,
        "epoch": epoch,
        "phase": checkpoint_identity["phase"],
        "checkpoint_copy": checkpoint_copy_receipt,
        "checkpoint_identity": checkpoint_identity,
        "storage_preflight": preflight,
        "model_stage": model_stage,
        "real_bytes": real_bytes,
        "real_model_plus_token_to_surrogate_joint_ratio": ratio,
        "retention": retention,
        "wall_seconds": time.time() - started,
        "scorer_invoked": False,
        "## RECALL EVIDENCE": RECALL_EVIDENCE,
        **_authority(),
    }
    rows = _read_receipts()
    calibration = _calibration([*rows, result])
    result["calibration"] = calibration
    _atomic_json(root / "PROBE_RESULT.json", result)
    _append_jsonl(RECEIPT_STREAM, result)
    return result


def _alert_for(result: dict[str, Any]) -> dict[str, Any] | None:
    real = result.get("real_bytes", {})
    if real.get("lossless_roundtrip") is False:
        kind = "LOSSLESSNESS_ROUNDTRIP_FAILED"
    elif isinstance(real.get("archive_bytes"), int) and real["archive_bytes"] < ARCHIVE_BAR_BYTES:
        kind = "WINNER-KNOWN-EARLY"
    elif result.get("calibration", {}).get("terminal_verdict") == "FAILURE_KNOWN_EARLY_PROJECTED_ABOVE_BAR":
        kind = "FAILURE-KNOWN-EARLY"
    else:
        return None
    return {
        "schema": "ddm_rx2_midrun_serialization_alert.v1",
        "generated_utc": _utc_now(),
        "alert": kind,
        "epoch": result.get("epoch"),
        "probe_result": race.file_record(PROBE_ROOT / f"ep{int(result['epoch']):04d}/PROBE_RESULT.json"),
        "real_bytes": real,
        "calibration": result.get("calibration"),
        "consumer": "MAIN RX2 harvester",
        "## RECALL EVIDENCE": RECALL_EVIDENCE,
        **_authority(),
    }


def _copy_and_probe(epoch: int, source: Path, args: argparse.Namespace) -> dict[str, Any]:
    root = PROBE_ROOT / f"ep{epoch:04d}"
    copy_path = root / "checkpoint.copy.pt"
    copy_receipt = _stable_copy(source, copy_path)
    return run_probe(
        epoch,
        copy_path,
        copy_receipt,
        brotli=args.brotli,
        required_free_gib=args.required_free_gib,
    )


def cadence_loop(args: argparse.Namespace) -> int:
    PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        receipts = _read_receipts()
        for prior in reversed(receipts):
            alert = _alert_for(prior)
            if alert is not None:
                _atomic_json(ALERT_PATH, alert)
                print(json.dumps(alert, sort_keys=True), flush=True)
                return 0
        completed_epochs = {int(row["epoch"]) for row in receipts if row.get("model_stage")}
        candidate = _newest_due_checkpoint(completed_epochs)
        if candidate is None:
            print(
                json.dumps(
                    {
                        "status": "POLLING_FOR_DUE_IMMUTABLE_CHECKPOINT",
                        "completed_epochs": sorted(completed_epochs),
                        "generated_utc": _utc_now(),
                        **_authority(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(args.poll_seconds)
            continue
        epoch, source = candidate
        try:
            result = _copy_and_probe(epoch, source, args)
        except PermissionError:
            raise
        except Exception as exc:  # loop persistence: non-verdict errors are retained and retried
            error = {
                "schema": "ddm_rx2_midrun_serialization_probe_error.v1",
                "generated_utc": _utc_now(),
                "epoch": epoch,
                "source": str(source),
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "retrying": True,
                "## RECALL EVIDENCE": RECALL_EVIDENCE,
                **_authority(),
            }
            _atomic_json(PROBE_ROOT / f"ep{epoch:04d}/LAST_ERROR.json", error)
            _append_jsonl(RECEIPT_STREAM, error)
            print(json.dumps(error, sort_keys=True), flush=True)
            time.sleep(args.poll_seconds)
            continue
        alert = _alert_for(result)
        if alert is not None:
            _atomic_json(ALERT_PATH, alert)
            print(json.dumps(alert, sort_keys=True), flush=True)
            return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    once = sub.add_parser("once", help="copy and probe one immutable checkpoint")
    once.add_argument("--epoch", type=int, required=True)
    once.add_argument("--checkpoint-source", type=Path, required=True)
    loop = sub.add_parser("loop", help="resume the detached cadence loop")
    for item in (once, loop):
        item.add_argument("--brotli", default=shutil.which("brotli") or "brotli")
        item.add_argument("--required-free-gib", type=int, default=MIN_FREE_GIB)
    loop.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    loop.add_argument("--resume-from", type=Path, default=PROBE_ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.required_free_gib <= 0:
        raise SystemExit("--required-free-gib must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.command == "once":
        if not 0 <= args.epoch <= 60:
            raise SystemExit("--epoch must be between 0 and 60")
        result = _copy_and_probe(args.epoch, args.checkpoint_source, args)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result.get("complete") else 1
    if args.poll_seconds < 15:
        raise SystemExit("--poll-seconds must be at least 15")
    if args.resume_from.resolve(strict=False) != PROBE_ROOT.resolve(strict=False):
        raise SystemExit(f"--resume-from must name the canonical probe store: {PROBE_ROOT}")
    return cadence_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
