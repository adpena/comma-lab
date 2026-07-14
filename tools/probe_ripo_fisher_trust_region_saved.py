#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed custody preflight for saved n600 categorical-Fisher arrays.

The complete scorer grid is ``(600, 384, 512, 5)``.  A monolithic NPZ of the
required probabilities/logits plus proposed updates is too large to
materialize in this foreground probe.  This tool therefore reads only NPY
headers from the ZIP container, verifies producer/source bindings, closes
start/end TOCTOU custody, and emits an atomic boundedness blocker.  It never
calls training, SegNet, an evaluator, a provider, or ``numpy.load``.

No algebra or full-n600 verdict is produced until a separately reviewed,
pair-sharded streaming/resume contract exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from numpy.lib import format as npy_format

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA = "ripo_fisher_trust_region_saved_probe.v2"
CUSTODY_SCHEMA = "ripo_fisher_saved_n600_custody.v2"
PRODUCER_SCHEMA = "ripo_fisher_saved_n600_producer_receipt.v1"
CHECKPOINT_SCHEMA = "ripo_fisher_trust_region_saved_checkpoint.v2"
METRIC_ID = "argmax_native_vjp_fidelity_v1"
STATE_RECEIPT_SCHEMA = "reachable_decision_geometry_fidelity.v1"
FORMULATION_ID = "categorical_fisher_output_space_trust_region_v1"
LANE_ID = "lane_ripo_fisher_isometric_trust_region_500_20260714"
EXPECTED_PAIRS = 600
EXPECTED_HEIGHT = 384
EXPECTED_WIDTH = 512
EXPECTED_CLASSES = 5
EXPECTED_FIELD_SHAPE = (
    EXPECTED_PAIRS,
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    EXPECTED_CLASSES,
)
EXPECTED_TARGET_SHAPE = (EXPECTED_PAIRS, EXPECTED_HEIGHT, EXPECTED_WIDTH)
NO_VERDICT = "NO_VERDICT_DATA_CUSTODY"
BOUNDEDNESS_BLOCKER = "NO_VERDICT_BOUNDEDNESS_MONOLITHIC_FULL_GRID_NPZ"
TOCTOU_BLOCKER = "NO_VERDICT_TOCTOU_CUSTODY_CHANGED_DURING_PROBE"
AUTHORITY = "[macOS-CPU advisory; header/custody preflight only; no score authority]"
MAX_NPY_HEADER_BYTES = 4096

REQUIRED_CUSTODY_ARTIFACTS = (
    ("source_path", "source_sha256"),
    ("checkpoint_path", "checkpoint_sha256"),
    ("segnet_path", "segnet_sha256"),
    ("r_operator_path", "r_operator_sha256"),
)
SOURCE_PATHS = (
    "src/tac/optimization/ripo_fisher_trust_region.py",
    "src/tac/optimization/ripo_fisher_trust_region_mlx.py",
    "tools/probe_ripo_fisher_trust_region_saved.py",
    ".omx/research/ripo_fisher_isometric_trust_region_build_spec_20260714_codex.md",
)


class ProbeError(RuntimeError):
    """The request violates an isolation, resume, or custody contract."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProbeError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=reject_duplicates)
    except (OSError, ValueError) as error:
        raise ProbeError(f"cannot read valid JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ProbeError(f"JSON root must be an object: {path}")
    return value


def _resolve_path(raw: Any, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ProbeError(f"{name} must be a non-empty path string")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO / path


def _file_custody(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "bytes": int(stat.st_size),
        "sha256": _sha256(path),
    }


def _snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    unique = sorted({str(path.resolve()) if path.exists() else str(path) for path in paths})
    return {value: _file_custody(Path(value)) for value in unique}


def _snapshot_changes(
    start: Mapping[str, Mapping[str, Any]],
    end: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    keys = sorted(set(start) | set(end))
    return [key for key in keys if start.get(key) != end.get(key)]


def _snapshot_record(
    snapshot: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> Mapping[str, Any]:
    key = str(path.resolve()) if path.exists() else str(path)
    return snapshot.get(key, {})


def _hash_is_valid(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _source_paths() -> list[Path]:
    return [REPO / relative for relative in SOURCE_PATHS]


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, Any]:
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(stream, max_header_size=MAX_NPY_HEADER_BYTES)
    if version in {(2, 0), (3, 0)}:
        return npy_format.read_array_header_2_0(stream, max_header_size=MAX_NPY_HEADER_BYTES)
    raise ProbeError(f"unsupported NPY header version {version}")


def _inspect_npz_headers(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Inspect NPZ metadata without reading or materializing array payloads."""

    if not path.is_file():
        return {}, [f"saved-array NPZ is missing: {path}"]
    headers: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                blockers.append("NPZ contains duplicate ZIP member names")
            for member in names:
                if not member.endswith(".npy"):
                    blockers.append(f"NPZ contains unexpected non-NPY member: {member}")
                    continue
                key = member[:-4]
                if "/" in key or "\\" in key:
                    blockers.append(f"NPZ member must be top-level: {member}")
                    continue
                try:
                    with archive.open(member, "r") as stream:
                        shape, fortran_order, dtype = _read_npy_header(stream)
                except (OSError, ValueError, EOFError) as error:
                    blockers.append(f"cannot read bounded NPY header {member}: {error}")
                    continue
                if dtype.hasobject:
                    blockers.append(f"NPZ member {key} has forbidden object dtype")
                info = archive.getinfo(member)
                headers[key] = {
                    "shape": [int(value) for value in shape],
                    "dtype": dtype.str,
                    "dtype_kind": dtype.kind,
                    "fortran_order": bool(fortran_order),
                    "compressed_bytes": int(info.compress_size),
                    "uncompressed_bytes": int(info.file_size),
                }
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        return {}, [f"saved-array NPZ is not a valid bounded-inspectable ZIP: {error}"]
    return headers, blockers


def _shape_blockers(headers: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    field_keys = [key for key in ("probabilities", "logits") if key in headers]
    if not field_keys:
        blockers.append("NPZ requires full probabilities or logits header")
    for key in field_keys:
        header = headers[key]
        if tuple(header.get("shape", ())) != EXPECTED_FIELD_SHAPE:
            blockers.append(f"{key} must have exact shape {EXPECTED_FIELD_SHAPE}")
        if header.get("dtype_kind") != "f":
            blockers.append(f"{key} must have floating dtype")
    proposed = headers.get("proposed_logit_step")
    if proposed is None:
        blockers.append("NPZ requires proposed_logit_step header")
    else:
        if tuple(proposed.get("shape", ())) != EXPECTED_FIELD_SHAPE:
            blockers.append(f"proposed_logit_step must have exact shape {EXPECTED_FIELD_SHAPE}")
        if proposed.get("dtype_kind") != "f":
            blockers.append("proposed_logit_step must have floating dtype")
    targets = headers.get("target_classes")
    if targets is None:
        blockers.append("NPZ requires target_classes header")
    else:
        if tuple(targets.get("shape", ())) != EXPECTED_TARGET_SHAPE:
            blockers.append(f"target_classes must have exact shape {EXPECTED_TARGET_SHAPE}")
        if targets.get("dtype_kind") not in {"i", "u"}:
            blockers.append("target_classes must have integer dtype")
    pair_ids = headers.get("pair_ids")
    if pair_ids is None:
        blockers.append("NPZ requires pair_ids header")
    else:
        if tuple(pair_ids.get("shape", ())) != (EXPECTED_PAIRS,):
            blockers.append(f"pair_ids must have exact shape ({EXPECTED_PAIRS},)")
        if pair_ids.get("dtype_kind") not in {"i", "u"}:
            blockers.append("pair_ids must have integer dtype")
    allowed = {"probabilities", "logits", "proposed_logit_step", "target_classes", "pair_ids"}
    unexpected = sorted(set(headers) - allowed)
    if unexpected:
        blockers.append(f"NPZ contains undeclared arrays: {unexpected}")
    return blockers


def _artifact_paths(custody: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for path_key, _ in REQUIRED_CUSTODY_ARTIFACTS:
        raw = custody.get(path_key)
        if isinstance(raw, str) and raw:
            paths.append(_resolve_path(raw, name=f"custody.{path_key}"))
    raw_producer = custody.get("producer_receipt_path")
    if isinstance(raw_producer, str) and raw_producer:
        paths.append(_resolve_path(raw_producer, name="custody.producer_receipt_path"))
    return paths


def _producer_source_hint(custody: Mapping[str, Any]) -> list[Path]:
    raw = custody.get("producer_receipt_path")
    if not isinstance(raw, str) or not raw:
        return []
    path = _resolve_path(raw, name="custody.producer_receipt_path")
    if not path.is_file():
        return []
    try:
        producer = _load_json(path)
    except ProbeError:
        return []
    source = producer.get("producer_source")
    if not isinstance(source, Mapping):
        return []
    raw_source = source.get("path")
    if not isinstance(raw_source, str) or not raw_source:
        return []
    return [_resolve_path(raw_source, name="producer.producer_source.path")]


def _custody_blockers(
    custody: Mapping[str, Any],
    *,
    input_npz: Path,
    start_custody: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if custody.get("schema") != CUSTODY_SCHEMA:
        blockers.append(f"custody.schema must equal {CUSTODY_SCHEMA}")
    if custody.get("complete") is not True:
        blockers.append("custody.complete must be true")
    if custody.get("n_pairs") != EXPECTED_PAIRS:
        blockers.append(f"custody.n_pairs must equal {EXPECTED_PAIRS}")
    if custody.get("pair_indices") != list(range(EXPECTED_PAIRS)):
        blockers.append("custody.pair_indices must be the contiguous ordered range 0..599")
    if custody.get("actual_r_applied") is not True:
        blockers.append("custody.actual_r_applied must be true")
    if custody.get("frozen_cpu_torch_segnet") is not True:
        blockers.append("custody.frozen_cpu_torch_segnet must be true")
    if custody.get("probability_source") != "frozen_cpu_torch_segnet_after_actual_r":
        blockers.append(
            "custody.probability_source must equal frozen_cpu_torch_segnet_after_actual_r"
        )
    declared_npz_hash = custody.get("saved_arrays_sha256")
    if not _hash_is_valid(declared_npz_hash):
        blockers.append("custody.saved_arrays_sha256 must be a lowercase SHA-256")
    elif _snapshot_record(start_custody, input_npz).get("sha256") != declared_npz_hash:
        blockers.append("custody.saved_arrays_sha256 does not match input NPZ bytes")
    for path_key, hash_key in REQUIRED_CUSTODY_ARTIFACTS:
        try:
            artifact = _resolve_path(custody.get(path_key), name=f"custody.{path_key}")
        except ProbeError as error:
            blockers.append(str(error))
            continue
        declared_hash = custody.get(hash_key)
        if not artifact.is_file():
            blockers.append(f"custody.{path_key} artifact is missing: {artifact}")
        elif not _hash_is_valid(declared_hash):
            blockers.append(f"custody.{hash_key} must be a lowercase SHA-256")
        elif _snapshot_record(start_custody, artifact).get("sha256") != declared_hash:
            blockers.append(f"custody.{hash_key} does not match {path_key} bytes")
    if not _hash_is_valid(custody.get("producer_receipt_sha256")):
        blockers.append("custody.producer_receipt_sha256 must be a lowercase SHA-256")
    return blockers


def _producer_blockers(
    custody: Mapping[str, Any],
    *,
    input_npz: Path,
    headers: Mapping[str, Mapping[str, Any]],
    start_custody: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], list[str]]:
    blockers: list[str] = []
    try:
        path = _resolve_path(
            custody.get("producer_receipt_path"),
            name="custody.producer_receipt_path",
        )
    except ProbeError as error:
        return {}, [str(error)]
    if not path.is_file():
        return {}, [f"producer receipt is missing: {path}"]
    declared_hash = custody.get("producer_receipt_sha256")
    if _hash_is_valid(declared_hash) and _snapshot_record(start_custody, path).get("sha256") != declared_hash:
        blockers.append("custody.producer_receipt_sha256 does not match producer receipt bytes")
    try:
        producer = _load_json(path)
    except ProbeError as error:
        return {}, [*blockers, str(error)]
    if producer.get("schema") != PRODUCER_SCHEMA:
        blockers.append(f"producer.schema must equal {PRODUCER_SCHEMA}")
    if producer.get("complete") is not True:
        blockers.append("producer.complete must be true")
    output = producer.get("output")
    if not isinstance(output, Mapping):
        blockers.append("producer.output must be an object")
    else:
        try:
            output_path = _resolve_path(output.get("path"), name="producer.output.path")
        except ProbeError as error:
            blockers.append(str(error))
        else:
            if output_path.resolve() != input_npz.resolve():
                blockers.append("producer.output.path does not bind the probed NPZ")
        if output.get("sha256") != _snapshot_record(start_custody, input_npz).get("sha256"):
            blockers.append("producer.output.sha256 does not bind the probed NPZ bytes")
        if output.get("container") != "monolithic_npz":
            blockers.append("producer.output.container must equal monolithic_npz")
        if output.get("arrays") != dict(headers):
            blockers.append("producer.output.arrays does not exactly match inspected NPZ headers")
    bindings = producer.get("source_binding")
    if not isinstance(bindings, Mapping):
        blockers.append("producer.source_binding must be an object")
    else:
        for _, hash_key in REQUIRED_CUSTODY_ARTIFACTS:
            if bindings.get(hash_key) != custody.get(hash_key):
                blockers.append(f"producer.source_binding.{hash_key} does not match custody")
        if bindings.get("actual_r_applied") is not True:
            blockers.append("producer.source_binding.actual_r_applied must be true")
        if bindings.get("frozen_cpu_torch_segnet") is not True:
            blockers.append("producer.source_binding.frozen_cpu_torch_segnet must be true")
    producer_source = producer.get("producer_source")
    if not isinstance(producer_source, Mapping):
        blockers.append("producer.producer_source must be an object")
    else:
        try:
            source_path = _resolve_path(
                producer_source.get("path"),
                name="producer.producer_source.path",
            )
        except ProbeError as error:
            blockers.append(str(error))
        else:
            if not source_path.is_file():
                blockers.append(f"producer source is missing: {source_path}")
            elif producer_source.get("sha256") != _snapshot_record(
                start_custody,
                source_path,
            ).get("sha256"):
                blockers.append("producer.producer_source.sha256 does not match source bytes")
    return producer, blockers


def _fingerprint(
    *,
    start_custody: Mapping[str, Mapping[str, Any]],
    delta: float,
    delta_convention: str,
    mode: str,
    tolerance: float,
    tier: str,
) -> str:
    payload = {
        "start_custody": start_custody,
        "delta": float(delta),
        "delta_convention": delta_convention,
        "mode": mode,
        "tolerance": float(tolerance),
        "tier": tier,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _blocker_receipt(
    *,
    fingerprint: str,
    blockers: Sequence[str],
    tier: str,
    headers: Mapping[str, Mapping[str, Any]],
    start_custody: Mapping[str, Mapping[str, Any]],
    end_custody: Mapping[str, Mapping[str, Any]],
    changed_paths: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": NO_VERDICT,
        "verdict": NO_VERDICT,
        "verdict_scope": "DATA-CUSTODY-OR-BOUNDEDNESS ONLY; no formulation or family verdict",
        "authority": AUTHORITY,
        "lane_id": LANE_ID,
        "metric_id": METRIC_ID,
        "state_receipt_schema": STATE_RECEIPT_SCHEMA,
        "formulation_id": FORMULATION_ID,
        "tier": tier,
        "blockers": sorted(set(blockers)),
        "inspected_npz_headers": dict(headers),
        "materialization_attempted": False,
        "pair_sharded_streaming_implemented": False,
        "start_custody": dict(start_custody),
        "end_custody": dict(end_custody),
        "toctou_changed_paths": list(changed_paths),
        "progress_fingerprint": fingerprint,
        "large_artifacts_written": False,
        "score_claim": False,
        "pointer_moved": False,
        "pose": "NOT_MEASURED",
        "archive_bytes": "NOT_MEASURED",
        "written_at_utc": _utc_now(),
    }


def run_probe(
    *,
    input_npz: Path,
    custody_json: Path,
    output_dir: Path,
    delta: float,
    delta_convention: str,
    mode: str,
    tolerance: float,
    tier: str = "algebra_calibration",
) -> dict[str, Any]:
    """Inspect custody and emit a bounded, resumable no-verdict receipt."""

    input_npz = Path(input_npz)
    custody_json = Path(custody_json)
    output_dir = Path(output_dir)
    base_paths = [input_npz, custody_json, *_source_paths()]
    base_start = _snapshot(base_paths)

    blockers: list[str] = []
    if custody_json.is_file():
        try:
            custody = _load_json(custody_json)
        except ProbeError as error:
            custody = {}
            blockers.append(str(error))
    else:
        custody = {}
        blockers.append(f"custody JSON is missing: {custody_json}")
    extra_paths = [*_artifact_paths(custody), *_producer_source_hint(custody)]
    tracked_paths = [*base_paths, *extra_paths]
    start_custody = {**base_start, **_snapshot(extra_paths)}

    fingerprint = _fingerprint(
        start_custody=start_custody,
        delta=delta,
        delta_convention=delta_convention,
        mode=mode,
        tolerance=tolerance,
        tier=tier,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipt.json"
    if receipt_path.exists():
        previous = _load_json(receipt_path)
        if previous.get("progress_fingerprint") != fingerprint:
            raise ProbeError("output_dir contains a receipt for a different progress fingerprint")
        return dict(previous)

    scratch = output_dir / f".scratch.{os.getpid()}.{uuid.uuid4().hex[:12]}"
    scratch.mkdir()
    try:
        blockers.extend(
            _custody_blockers(
                custody,
                input_npz=input_npz,
                start_custody=start_custody,
            )
        )
        headers, header_blockers = _inspect_npz_headers(input_npz)
        blockers.extend(header_blockers)
        blockers.extend(_shape_blockers(headers))
        _, producer_blockers = _producer_blockers(
            custody,
            input_npz=input_npz,
            headers=headers,
            start_custody=start_custody,
        )
        blockers.extend(producer_blockers)
        if tier != "algebra_calibration":
            blockers.append(
                "fixed_global_head tier requires custodied baseline reproduction, frozen features, "
                "mutable-head-only proof, and actual-R CPU-SegNet outcomes"
            )
        # This is unconditional for the only accepted container.  It is the
        # central no-fake boundary: no full-grid array is materialized here.
        blockers.append(BOUNDEDNESS_BLOCKER)

        end_custody = _snapshot(tracked_paths)
        changed_paths = _snapshot_changes(start_custody, end_custody)
        if changed_paths:
            blockers.append(TOCTOU_BLOCKER)

        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "header_and_custody_preflight",
            "complete": True,
            "status": NO_VERDICT,
            "blockers": sorted(set(blockers)),
            "progress_fingerprint": fingerprint,
            "materialization_attempted": False,
            "start_custody": start_custody,
            "end_custody": end_custody,
            "toctou_changed_paths": changed_paths,
            "large_artifacts_written": False,
            "written_at_utc": _utc_now(),
        }
        _atomic_json(output_dir / "input_validation_checkpoint.json", checkpoint)
        receipt = _blocker_receipt(
            fingerprint=fingerprint,
            blockers=blockers,
            tier=tier,
            headers=headers,
            start_custody=start_custody,
            end_custody=end_custody,
            changed_paths=changed_paths,
        )
        _atomic_json(receipt_path, receipt)
        return receipt
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--custody-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--delta-convention", choices=("delta_kl", "delta_quad"), required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "local_directional",
            "exact_kl",
            "local_euclidean_ball",
            "uniform_l2_control",
        ),
        required=True,
    )
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--tier",
        choices=("algebra_calibration", "fixed_global_head"),
        default="algebra_calibration",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run_probe(
        input_npz=args.input_npz,
        custody_json=args.custody_json,
        output_dir=args.output_dir,
        delta=args.delta,
        delta_convention=args.delta_convention,
        mode=args.mode,
        tolerance=args.tolerance,
        tier=args.tier,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
