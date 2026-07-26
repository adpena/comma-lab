#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumably score one decoded raw witness with pair/class/event debt telemetry.

This is an encode-side local CPU diagnostic.  It consumes already-decoded
camera RGB bytes for both a zero-distortion target and a candidate, never
enters an archive, and never runs inside ``inflate``.  Scorer calls preserve
the frozen evaluator's batch-16 geometry; cached labels/poses are diagnostics,
not target authority.  The final receipt is advisory even when it is byte-bound
to a historical contest-CPU aggregate; only ``upstream/evaluate.py`` on the
exact archive can make a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _path in (REPO, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tools import measure_v10_free_predictor_floor as scorer  # noqa: E402

SCHEMA = "tac.coupled_witness_raw_debt.v3"
CONFIG_SCHEMA = "tac.coupled_witness_raw_debt_config.v3"
STAGE_SCHEMA = "tac.coupled_witness_raw_debt_stage.v3"
STATE_SCHEMA = "tac.coupled_witness_raw_debt_state.v3"
LAUNCH_SCHEMA = "tac.coupled_witness_raw_debt_launch.v3"
LEGACY_FINAL_SCHEMA = "tac.coupled_witness_raw_debt.v2"
CONTEST_SCHEMA_VERSION = 1
CONTEST_MANIFEST_SCHEMA = "contest_auth_eval_inflated_output_manifest_v1"
TARGET_SCHEMA = "m2_live_target_selection_receipt.v1"
ORIGINAL_UNCOMPRESSED_SIZE_BYTES = 37_545_489
DEFAULT_PAIR_COUNT = 600
DEFAULT_STAGE_PAIRS = 16
DEFAULT_RAW = Path("/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/out_serial/0.raw")
DEFAULT_TARGET_RAW = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m2_live_target_selection_20260720T1528Z/inflated/0.raw"
)
DEFAULT_TARGET_RECEIPT = REPO / ".omx/research/m2_live_target_selection_20260720T1548Z.json"
DEFAULT_TARGET_RECEIPT_SHA256 = "513d2fe54166b8b82904e671bc8beedca11e0b98ab68f09e860507ef82b2e574"
DEFAULT_REFERENCE = Path(
    "/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/"
    "modal_contest_cpu/harvest_fc01KXXRAR/contest_auth_eval.json"
)
DEFAULT_REFERENCE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/archive.zip")
LEGACY_V2_FINAL_PATH = REPO / (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "c1_live_target_debt_n600_batch16.json"
)
LEGACY_V2_FINAL_FILE_SHA256 = "0db8e47a994cad5367e5eb3028055e667bc4caf3f174026d13171be662e7fbd3"


class RawDebtError(RuntimeError):
    """Fail-closed decoded-byte, stage, scorer, or custody error."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RawDebtError("receipt value is not canonical-JSON encodable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RawDebtError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _with_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise RawDebtError(f"hash field already present: {field}")
    result = dict(payload)
    result[field] = _sha256_bytes(_canonical(result))
    return result


def _validate_hash(payload: Mapping[str, Any], field: str) -> None:
    digest = payload.get(field)
    if not isinstance(digest, str) or len(digest) != 64:
        raise RawDebtError(f"{field} is missing or malformed")
    body = {key: value for key, value in payload.items() if key != field}
    if _sha256_bytes(_canonical(body)) != digest:
        raise RawDebtError(f"{field} differs from receipt body")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise RawDebtError(f"{label} is not a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RawDebtError(f"{label} is not a finite number") from exc
    if not math.isfinite(result) or result < minimum:
        raise RawDebtError(f"{label} is not a finite number >= {minimum}")
    return result


def _is_exact_zero(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value == 0
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RawDebtError(f"write-once stage differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if not path.is_file() or path.read_bytes() != encoded:
                raise RawDebtError(f"concurrent write-once stage differs: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawDebtError(f"cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RawDebtError(f"{label} root must be an object")
    return payload


def _contest_reference(
    path: Path,
    *,
    raw_sha256: str,
    raw_bytes: int,
    local_archive_path: Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(path, "contest reference")
    try:
        provenance = payload["provenance"]
        inflated_wrapper = provenance["inflated_output_manifest"]
        manifest = inflated_wrapper["payload"]
        files = manifest["files"]
    except (KeyError, TypeError) as exc:
        raise RawDebtError("contest reference lacks inflated-output custody") from exc
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(inflated_wrapper, Mapping)
        or not isinstance(manifest, Mapping)
    ):
        raise RawDebtError("contest reference lacks inflated-output custody")
    if (
        payload.get("schema_version") != CONTEST_SCHEMA_VERSION
        or provenance.get("schema_version") != CONTEST_SCHEMA_VERSION
        or payload.get("n_samples") != DEFAULT_PAIR_COUNT
        or payload.get("score_axis") != "contest_cpu"
        or payload.get("evidence_grade") != "contest-CPU"
        or payload.get("lane_tag") != "[contest-CPU]"
        or payload.get("score_claim") is not True
        or payload.get("score_claim_valid") is not True
        or payload.get("score_claim_eligible") is not True
        or payload.get("cpu_leaderboard_reproduction_eligible") is not True
        or payload.get("evidence_semantics") != "public_leaderboard_cpu_reproduction"
        or not isinstance(payload.get("allowed_uses"), list)
        or "cpu_axis_score_claim" not in payload["allowed_uses"]
        or provenance.get("tool") != "experiments/contest_auth_eval.py"
        or provenance.get("device") != "cpu"
        or provenance.get("platform_system") != "Linux"
        or provenance.get("cuda_available") is not False
        or provenance.get("mps_available") is not False
    ):
        raise RawDebtError("contest reference is not exact eligible n600 contest-CPU authority")
    archive_size = payload.get("archive_size_bytes")
    archive_sha = provenance.get("archive_sha256")
    if (
        isinstance(archive_size, bool)
        or not isinstance(archive_size, int)
        or archive_size <= 0
        or provenance.get("archive_size_bytes") != archive_size
        or not _is_sha256(archive_sha)
        or payload.get("original_uncompressed_size_bytes") != ORIGINAL_UNCOMPRESSED_SIZE_BYTES
    ):
        raise RawDebtError("contest reference archive custody is inconsistent")
    if (
        manifest.get("schema") != CONTEST_MANIFEST_SCHEMA
        or manifest.get("raw_file_count") != 1
        or manifest.get("total_bytes") != raw_bytes
        or not isinstance(files, list)
        or len(files) != 1
    ):
        raise RawDebtError("contest reference decoded manifest is inconsistent")
    matching = [
        row
        for row in files
        if isinstance(row, Mapping)
        and row.get("relative_path") == "0.raw"
        and row.get("exists") is True
        and row.get("sha256") == raw_sha256
        and row.get("bytes") == raw_bytes
    ]
    aggregate_body = {
        "files": [
            {"relative_path": row.get("relative_path"), "bytes": row.get("bytes"), "sha256": row.get("sha256")}
            for row in files
            if isinstance(row, Mapping)
        ]
    }
    if len(matching) != 1 or manifest.get("aggregate_sha256") != _sha256_bytes(_canonical(aggregate_body)):
        raise RawDebtError("contest reference does not bind this exact decoded raw")
    seg = _finite_number(payload.get("avg_segnet_dist"), "contest avg_segnet_dist")
    pose = _finite_number(payload.get("avg_posenet_dist"), "contest avg_posenet_dist")
    recomputed = 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_size / ORIGINAL_UNCOMPRESSED_SIZE_BYTES
    for field in ("score_recomputed_from_components", "canonical_score"):
        value = _finite_number(payload.get(field), f"contest {field}")
        if not math.isclose(value, recomputed, rel_tol=0.0, abs_tol=1e-12):
            raise RawDebtError("contest reference canonical score does not recompute exactly")
    if payload.get("canonical_score_source") != "score_recomputed_from_components":
        raise RawDebtError("contest reference canonical score source is not exact recomputation")
    archive_path_value: str | Path | None = local_archive_path
    if archive_path_value is None:
        recorded_path = provenance.get("archive_path")
        archive_path_value = recorded_path if isinstance(recorded_path, str) else None
    if archive_path_value is None:
        raise RawDebtError("contest reference lacks locally verifiable archive bytes")
    archive_path = Path(archive_path_value).expanduser()
    if not archive_path.is_absolute():
        archive_path = path.parent / archive_path
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise RawDebtError("custodied local contest archive is absent")
    if archive_path.stat().st_size != archive_size or _sha256_file(archive_path) != archive_sha:
        raise RawDebtError("contest reference local archive bytes drifted")
    local_archive_custody = {
        "path": str(archive_path),
        "bytes": archive_size,
        "sha256": archive_sha,
    }
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "score_axis": payload.get("score_axis"),
        "evidence_grade": payload.get("evidence_grade"),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "local_archive_verified": True,
        "local_archive_custody": local_archive_custody,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "canonical_score": recomputed,
        "decoded_raw": dict(matching[0]),
    }


def _target_reference(
    path: Path,
    *,
    raw_sha256: str,
    raw_bytes: int,
    pair_count: int,
    required_path: Path | None = None,
    required_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind the exact live target raw to its measured zero-distortion receipt."""

    resolved = path.resolve()
    file_sha256 = _sha256_file(resolved)
    if required_path is not None and resolved != required_path.resolve():
        raise RawDebtError("zero-distortion target reference is not the pinned receipt")
    if required_file_sha256 is not None and file_sha256 != required_file_sha256:
        raise RawDebtError("zero-distortion target reference pinned bytes drifted")
    payload = _load_json(resolved, "zero-distortion target reference")
    try:
        candidate = payload["candidate"]
        source = payload["source_custody"]
        if not isinstance(candidate, Mapping) or not isinstance(source, Mapping):
            raise TypeError
        bound = (
            payload["schema"] == TARGET_SCHEMA
            and payload["axis"] == "[macOS-CPU advisory]"
            and payload["score_claim"] is False
            and payload["promotion_eligible"] is False
            and pair_count == DEFAULT_PAIR_COUNT
            and candidate["inflated_raw_sha256"] == raw_sha256
            and candidate["inflated_raw_bytes"] == raw_bytes
            and source["pair_count"] == DEFAULT_PAIR_COUNT
            and source["frame_count"] == 2 * DEFAULT_PAIR_COUNT
            and _is_sha256(source.get("gt_cache_sha256"))
            and isinstance(source.get("gt_cache_bytes"), int)
            and not isinstance(source.get("gt_cache_bytes"), bool)
            and source["gt_cache_bytes"] > 0
            and _is_exact_zero(candidate["d_seg"])
            and _is_exact_zero(candidate["d_pose"])
            and isinstance(candidate["archive_bytes"], int)
            and not isinstance(candidate["archive_bytes"], bool)
            and candidate["archive_bytes"] > 0
            and _is_sha256(candidate["archive_sha256"])
            and _is_sha256(candidate["inflated_raw_sha256"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RawDebtError("zero-distortion target reference lacks required custody") from exc
    if not bound:
        raise RawDebtError("zero-distortion target reference does not bind this exact raw and population")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256,
        "schema": payload.get("schema"),
        "axis": payload.get("axis"),
        "score_claim": payload.get("score_claim"),
        "archive_sha256": candidate.get("archive_sha256"),
        "archive_bytes": candidate.get("archive_bytes"),
        "decoded_raw_sha256": raw_sha256,
        "decoded_raw_bytes": raw_bytes,
        "d_seg": candidate.get("d_seg"),
        "d_pose": candidate.get("d_pose"),
        "pair_count": pair_count,
        "source_gt_cache_sha256": source.get("gt_cache_sha256"),
        "source_gt_cache_bytes": source.get("gt_cache_bytes"),
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], *, pair_count: int) -> dict[str, Any]:
    if [row.get("pair_id") for row in rows] != list(range(pair_count)):
        raise RawDebtError("stage rows are not exactly the canonical contiguous pair sequence")
    try:
        mean_d_seg = math.fsum(float(row["d_seg"]) for row in rows) / pair_count
        mean_d_pose = math.fsum(float(row["d_pose"]) for row in rows) / pair_count
        per_class = scorer._aggregate_per_class_debt(rows)
        events = [[int(row["pair_id"]), *event] for row in rows for event in row["seg_events"]]
    except (KeyError, TypeError, ValueError, scorer.PredictorFloorError) as exc:
        raise RawDebtError("cannot aggregate staged scorer debt") from exc
    if len(events) != sum(int(row["seg_mismatched_pixels"]) for row in rows):
        raise RawDebtError("Seg event rows do not reconcile to mismatch counts")
    return {
        "pair_count": pair_count,
        "mean_d_seg": mean_d_seg,
        "mean_d_pose": mean_d_pose,
        "class_order": list(scorer.CLASS_ORDER),
        "per_class": per_class,
        "seg_event_count": len(events),
        "seg_event_stream_schema": "[pair,row,col,target_class,candidate_class]",
        "seg_event_stream_sha256": _sha256_bytes(_canonical(events)),
        "cache_label_mismatches": sum(int(row["cache_label_mismatches"]) for row in rows),
        "cache_pose_max_abs_difference": max(float(row["cache_pose_max_abs_difference"]) for row in rows),
        "pair_rows_sha256": _sha256_bytes(_canonical(list(rows))),
    }


def _score_batch_against_live_target(
    segnet: Any,
    posenet: Any,
    torch: Any,
    *,
    target_frame0: Any,
    target_frame1: Any,
    candidate_frame0: Any,
    candidate_frame1: Any,
    cached_labels: Any,
    cached_poses: Any,
) -> list[dict[str, Any]]:
    """Mirror one frozen evaluator batch and retain its exact error events."""

    try:
        import einops
    except ImportError as exc:
        raise RawDebtError("einops is required by the frozen scorer path") from exc
    target0 = scorer.np.asarray(target_frame0)
    target1 = scorer.np.asarray(target_frame1)
    candidate0 = scorer.np.asarray(candidate_frame0)
    candidate1 = scorer.np.asarray(candidate_frame1)
    batch = target0.shape[0]
    expected_frames = (batch, *scorer.CAMERA_HW, 3)
    expected_labels = (batch, *scorer.SCORER_HW)
    if (
        target0.shape != expected_frames
        or target1.shape != expected_frames
        or candidate0.shape != expected_frames
        or candidate1.shape != expected_frames
    ):
        raise RawDebtError("live target/candidate batch geometry differs from the frozen camera geometry")
    labels_cache = scorer.np.asarray(cached_labels)
    poses_cache = scorer.np.asarray(cached_poses, dtype=scorer.np.float64)
    if labels_cache.shape != expected_labels or poses_cache.shape != (batch, 6):
        raise RawDebtError("cache diagnostic geometry differs from the frozen scorer geometry")

    def prepare(frame0: Any, frame1: Any) -> Any:
        pair = torch.from_numpy(scorer.np.stack((frame0, frame1), axis=1)).float()
        return einops.rearrange(pair, "b t h w c -> b t c h w")

    target_input = prepare(target0, target1)
    candidate_input = prepare(candidate0, candidate1)
    with torch.inference_mode():
        target_logits = segnet(segnet.preprocess_input(target_input))
        candidate_logits = segnet(segnet.preprocess_input(candidate_input))
        target_pose_output = posenet(posenet.preprocess_input(target_input))
        candidate_pose_output = posenet(posenet.preprocess_input(candidate_input))
        target_pose = target_pose_output["pose"] if isinstance(target_pose_output, dict) else target_pose_output
        candidate_pose = (
            candidate_pose_output["pose"] if isinstance(candidate_pose_output, dict) else candidate_pose_output
        )
        target_pose6 = target_pose[:, :6].cpu().numpy().astype(scorer.np.float64)
        candidate_pose6 = candidate_pose[:, :6].cpu().numpy().astype(scorer.np.float64)
        target_argmax = target_logits.argmax(dim=1).cpu().numpy()
        candidate_argmax = candidate_logits.argmax(dim=1).cpu().numpy()

    rows: list[dict[str, Any]] = []
    for local in range(batch):
        labels = scorer.np.asarray(target_argmax[local])
        prediction = scorer.np.asarray(candidate_argmax[local])
        mismatch = prediction != labels
        coordinates = scorer.np.argwhere(mismatch)
        events = [[int(row), int(col), int(labels[row, col]), int(prediction[row, col])] for row, col in coordinates]
        rows.append(
            {
                "d_seg": float(scorer.np.mean(mismatch)),
                "seg_mismatched_pixels": len(events),
                "seg_events": events,
                "seg_events_sha256": _sha256_bytes(_canonical(events)),
                "per_class": scorer._per_class_seg_debt(mismatch=mismatch, labels=labels),
                "d_pose": float(scorer.np.mean((candidate_pose6[local] - target_pose6[local]) ** 2)),
                "pose6": candidate_pose6[local].tolist(),
                "target_pose6": target_pose6[local].tolist(),
                "cache_label_mismatches": int(scorer.np.count_nonzero(labels != labels_cache[local])),
                "cache_pose_max_abs_difference": float(
                    scorer.np.max(scorer.np.abs(target_pose6[local] - poses_cache[local]))
                ),
            }
        )
    return rows


def _stage_path(stage_dir: Path, start: int, end: int) -> Path:
    return stage_dir / f"pairs-{start:04d}-{end - 1:04d}.json"


def _stage_payload(*, config_sha256: str, start: int, end: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _with_hash(
        {
            "schema": STAGE_SCHEMA,
            "config_sha256": config_sha256,
            "pair_start": start,
            "pair_end_exclusive": end,
            "rows": list(rows),
            "stage_complete": True,
        },
        "stage_sha256",
    )


def _state_payload(*, config_sha256: str, completed_pairs: int, latest_stage_sha256: str | None) -> dict[str, Any]:
    return _with_hash(
        {
            "schema": STATE_SCHEMA,
            "config_sha256": config_sha256,
            "completed_pairs": completed_pairs,
            "latest_stage_sha256": latest_stage_sha256,
        },
        "state_sha256",
    )


def _launch_manifest_path(state_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return state_path.with_name(f"{state_path.name}.launch.json")


def _load_launch_manifest(path: Path, *, config_sha256: str) -> dict[str, Any]:
    payload = _load_json(path, "launch manifest")
    _validate_hash(payload, "launch_sha256")
    if (
        set(payload)
        != {
            "schema",
            "written_at_utc",
            "config_sha256",
            "frontier_pointer_at_launch",
            "scientific_identity_excludes_frontier_pointer",
            "launch_sha256",
        }
        or payload.get("schema") != LAUNCH_SCHEMA
        or payload.get("config_sha256") != config_sha256
        or payload.get("scientific_identity_excludes_frontier_pointer") is not True
    ):
        raise RawDebtError("launch manifest belongs to different scientific inputs")
    return payload


def _ensure_launch_manifest(path: Path, *, config_sha256: str) -> dict[str, Any]:
    """Preserve mutable frontier context without making it scientific identity."""

    if path.exists():
        return _load_launch_manifest(path, config_sha256=config_sha256)
    payload = _with_hash(
        {
            "schema": LAUNCH_SCHEMA,
            "written_at_utc": datetime.now(UTC).isoformat(),
            "config_sha256": config_sha256,
            "frontier_pointer_at_launch": scorer._effective_pointer_target(),
            "scientific_identity_excludes_frontier_pointer": True,
        },
        "launch_sha256",
    )
    _write_once(path, payload)
    return payload


def _state_boundary_sha(stages: Sequence[Mapping[str, Any]], completed_pairs: int) -> str | None:
    if completed_pairs == 0:
        return None
    matching = [stage for stage in stages if stage.get("pair_end_exclusive") == completed_pairs]
    if len(matching) != 1 or not _is_sha256(matching[0].get("stage_sha256")):
        raise RawDebtError("resume state completed_pairs is not a preserved stage boundary")
    return str(matching[0]["stage_sha256"])


def _reconcile_state(
    state_path: Path,
    *,
    config_sha256: str,
    prefix_pairs: int,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Advance missing/lagging state from the validated write-once stage prefix."""

    if state_path.exists():
        state = _load_json(state_path, "resume state")
        _validate_hash(state, "state_sha256")
        if (
            set(state)
            != {
                "schema",
                "config_sha256",
                "completed_pairs",
                "latest_stage_sha256",
                "state_sha256",
            }
            or state.get("schema") != STATE_SCHEMA
            or state.get("config_sha256") != config_sha256
        ):
            raise RawDebtError("resume state belongs to different scientific inputs")
        completed = state.get("completed_pairs")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise RawDebtError("resume state completed_pairs is malformed")
        if completed > prefix_pairs:
            raise RawDebtError("resume state is ahead of the preserved stage prefix")
        expected_latest = _state_boundary_sha(stages, completed)
        if state.get("latest_stage_sha256") != expected_latest:
            raise RawDebtError("resume state latest_stage_sha256 differs at its completed boundary")
    else:
        completed = 0
    authoritative_latest = _state_boundary_sha(stages, prefix_pairs)
    if not state_path.exists() or completed < prefix_pairs:
        state = _state_payload(
            config_sha256=config_sha256,
            completed_pairs=prefix_pairs,
            latest_stage_sha256=authoritative_latest,
        )
        _atomic_json(state_path, state)
    return state


def _load_prefix(
    *,
    stage_dir: Path,
    pair_count: int,
    stage_pairs: int,
    config_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    expected_start = 0
    for path in sorted(stage_dir.glob("pairs-*.json")) if stage_dir.exists() else ():
        payload = _load_json(path, "debt stage")
        _validate_hash(payload, "stage_sha256")
        start = payload.get("pair_start")
        end = payload.get("pair_end_exclusive")
        if (
            set(payload)
            != {
                "schema",
                "config_sha256",
                "pair_start",
                "pair_end_exclusive",
                "rows",
                "stage_complete",
                "stage_sha256",
            }
            or payload.get("schema") != STAGE_SCHEMA
            or payload.get("config_sha256") != config_sha256
        ):
            raise RawDebtError(f"stage identity differs: {path}")
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
            raise RawDebtError(f"stage boundaries are malformed: {path}")
        if start != expected_start or end != min(start + stage_pairs, pair_count):
            raise RawDebtError(f"stage sequence is not one canonical prefix: {path}")
        if path != _stage_path(stage_dir, start, end):
            raise RawDebtError(f"stage filename differs from its canonical boundary: {path}")
        if payload.get("stage_complete") is not True:
            raise RawDebtError(f"stage is not complete: {path}")
        stage_rows = payload.get("rows")
        if (
            not isinstance(stage_rows, list)
            or any(not isinstance(row, Mapping) for row in stage_rows)
            or [row.get("pair_id") for row in stage_rows] != list(range(start, end))
        ):
            raise RawDebtError(f"stage pair rows differ: {path}")
        rows.extend(stage_rows)
        stages.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "stage_sha256": payload["stage_sha256"],
                "pair_start": start,
                "pair_end_exclusive": end,
            }
        )
        expected_start = end
    return rows, stages


def _historical_v2_matches(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    raw_sha256: str,
    target_raw_sha256: str,
    cache_sha256: str,
    pair_count: int,
    stage_pairs: int,
) -> bool:
    """Admit an immutable v2 final only as the exact historical input binding."""

    config = receipt.get("config")
    return bool(
        receipt_path.resolve() == LEGACY_V2_FINAL_PATH.resolve()
        and receipt_path.is_file()
        and _sha256_file(receipt_path) == LEGACY_V2_FINAL_FILE_SHA256
        and isinstance(config, Mapping)
        and receipt.get("schema") == LEGACY_FINAL_SCHEMA
        and config.get("schema") == "tac.coupled_witness_raw_debt_state.v2"
        and isinstance(config.get("raw"), Mapping)
        and config["raw"].get("sha256") == raw_sha256
        and isinstance(config.get("target_raw"), Mapping)
        and config["target_raw"].get("sha256") == target_raw_sha256
        and isinstance(config.get("cache"), Mapping)
        and config["cache"].get("sha256") == cache_sha256
        and config.get("pair_count") == pair_count
        and config.get("stage_pairs") == stage_pairs
        and receipt.get("config_sha256") == _sha256_bytes(_canonical(config))
    )


def _validate_completed_state(
    path: Path,
    *,
    config_sha256: str,
    pair_count: int,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not path.is_file():
        raise RawDebtError("completed final receipt lost its resume state")
    state = _load_json(path, "completed resume state")
    _validate_hash(state, "state_sha256")
    if (
        set(state)
        != {
            "schema",
            "config_sha256",
            "completed_pairs",
            "latest_stage_sha256",
            "state_sha256",
        }
        or state.get("schema") != STATE_SCHEMA
        or state.get("config_sha256") != config_sha256
        or state.get("completed_pairs") != pair_count
        or state.get("latest_stage_sha256") != _state_boundary_sha(stages, pair_count)
    ):
        raise RawDebtError("completed final receipt resume state is not its exact terminal boundary")
    return state


def _validate_input_end_barrier(
    config: Mapping[str, Any],
    *,
    contest_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Rehash every long-run input immediately before final publication."""

    rows: dict[str, dict[str, Any]] = {}
    for key in ("raw", "target_raw", "cache", "tool", "scorer_adapter"):
        value = config.get(key)
        if not isinstance(value, Mapping):
            raise RawDebtError(f"end barrier lacks config input: {key}")
        path_value = value.get("path")
        if not isinstance(path_value, str):
            raise RawDebtError(f"end barrier input path is malformed: {key}")
        path = Path(path_value).resolve()
        if not path.is_file():
            raise RawDebtError(f"end barrier input disappeared: {key}")
        observed = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        if observed["bytes"] != value.get("bytes") or observed["sha256"] != value.get("sha256"):
            raise RawDebtError(f"scientific input drifted during scoring: {key}")
        rows[key] = observed

    target_reference = config.get("target_reference")
    if not isinstance(target_reference, Mapping):
        raise RawDebtError("end barrier lacks target receipt custody")
    target_receipt_path = Path(str(target_reference.get("path", ""))).resolve()
    target_receipt_row = {
        "path": str(target_receipt_path),
        "bytes": target_receipt_path.stat().st_size if target_receipt_path.is_file() else -1,
        "sha256": _sha256_file(target_receipt_path) if target_receipt_path.is_file() else "",
    }
    if (
        target_receipt_row["bytes"] != target_reference.get("bytes")
        or target_receipt_row["sha256"] != target_reference.get("sha256")
    ):
        raise RawDebtError("zero-distortion target receipt drifted during scoring")
    rows["target_receipt"] = target_receipt_row

    scorer_custody = config.get("scorer_custody")
    if not isinstance(scorer_custody, Mapping):
        raise RawDebtError("end barrier lacks scorer custody")
    scorer._validate_scorer_custody(scorer_custody)

    if contest_reference is not None:
        reference_path = Path(str(contest_reference.get("path", ""))).resolve()
        archive = contest_reference.get("local_archive_custody")
        if not isinstance(archive, Mapping):
            raise RawDebtError("contest reference lacks preserved local archive custody")
        for key, value in (
            ("contest_reference", {"path": str(reference_path), **dict(contest_reference)}),
            ("contest_archive", archive),
        ):
            path = Path(str(value.get("path", ""))).resolve()
            if not path.is_file():
                raise RawDebtError(f"end barrier input disappeared: {key}")
            observed = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            if observed["bytes"] != value.get("bytes") or observed["sha256"] != value.get("sha256"):
                raise RawDebtError(f"authority input drifted during scoring: {key}")
            rows[key] = observed
    return {
        "schema": "tac.coupled_witness_raw_debt_end_barrier.v1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "files": rows,
        "scorer_custody_revalidated": True,
    }


def _validate_recorded_input_end_barrier(
    recorded: Any,
    current: Mapping[str, Any],
) -> None:
    """Require a prior final receipt to preserve the exact revalidated files."""

    if not isinstance(recorded, Mapping):
        raise RawDebtError("existing final receipt lacks its input end barrier")
    verified_at = recorded.get("verified_at_utc")
    if (
        set(recorded) != {"schema", "verified_at_utc", "files", "scorer_custody_revalidated"}
        or recorded.get("schema") != "tac.coupled_witness_raw_debt_end_barrier.v1"
        or not isinstance(verified_at, str)
        or not verified_at
        or recorded.get("scorer_custody_revalidated") is not True
        or recorded.get("files") != current.get("files")
    ):
        raise RawDebtError("existing final receipt input end barrier drifted")


def _validate_existing_v3_final(
    receipt: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    scorer_custody: Mapping[str, Any],
    stage_dir: Path,
    state_path: Path,
    launch_manifest_path: Path,
    pair_count: int,
    stage_pairs: int,
) -> None:
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("config") != config
        or receipt.get("scorer_custody") != scorer_custody
    ):
        raise RawDebtError("existing final receipt belongs to different inputs")
    rows, stages = _load_prefix(
        stage_dir=stage_dir,
        pair_count=pair_count,
        stage_pairs=stage_pairs,
        config_sha256=config_sha256,
    )
    if len(rows) != pair_count or receipt.get("pairs") != rows or receipt.get("stages") != stages:
        raise RawDebtError("existing final receipt lost or drifted preserved stage checkpoints")
    if receipt.get("aggregate") != _aggregate_rows(rows, pair_count=pair_count):
        raise RawDebtError("existing final aggregate differs from preserved stages")
    _validate_completed_state(
        state_path,
        config_sha256=config_sha256,
        pair_count=pair_count,
        stages=stages,
    )
    if not launch_manifest_path.is_file():
        raise RawDebtError("existing final receipt lost its launch manifest")
    launch = _load_launch_manifest(launch_manifest_path, config_sha256=config_sha256)
    launch_row = receipt.get("launch_manifest")
    expected_launch_row = {
        "path": str(launch_manifest_path),
        "bytes": launch_manifest_path.stat().st_size,
        "sha256": _sha256_file(launch_manifest_path),
        "launch_sha256": launch["launch_sha256"],
    }
    if launch_row != expected_launch_row:
        raise RawDebtError("existing final receipt launch-manifest custody drifted")
    contest_reference = receipt.get("contest_cpu_reference_same_decoded_raw")
    if contest_reference is not None and not isinstance(contest_reference, Mapping):
        raise RawDebtError("existing final receipt contest reference is malformed")
    current_barrier = _validate_input_end_barrier(config, contest_reference=contest_reference)
    _validate_recorded_input_end_barrier(receipt.get("input_end_barrier"), current_barrier)


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = args.raw.expanduser().resolve()
    target_raw_path = args.target_raw.expanduser().resolve()
    output = args.output.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    stage_dir = args.stage_dir.expanduser().resolve()
    launch_manifest_path = _launch_manifest_path(state_path, getattr(args, "launch_manifest", None))
    if len({output, state_path, launch_manifest_path}) != 3:
        raise RawDebtError("output, state, and launch-manifest paths must be distinct")
    if not raw_path.is_file():
        raise RawDebtError(f"decoded raw is absent: {raw_path}")
    if not target_raw_path.is_file():
        raise RawDebtError(f"zero-distortion target raw is absent: {target_raw_path}")
    if not 1 <= args.stage_pairs <= args.pair_count:
        raise RawDebtError("stage-pairs must be within the scored pair population")
    if args.pair_count == DEFAULT_PAIR_COUNT and args.stage_pairs != DEFAULT_STAGE_PAIRS:
        raise RawDebtError("canonical n600 debt requires the frozen evaluator batch size of 16 pairs")
    frame_bytes = scorer.CAMERA_HW[0] * scorer.CAMERA_HW[1] * 3
    expected_bytes = args.pair_count * 2 * frame_bytes
    if raw_path.stat().st_size != expected_bytes:
        raise RawDebtError(f"decoded raw bytes differ: {raw_path.stat().st_size} != {expected_bytes}")
    if target_raw_path.stat().st_size != expected_bytes:
        raise RawDebtError(
            f"zero-distortion target raw bytes differ: {target_raw_path.stat().st_size} != {expected_bytes}"
        )
    raw_sha = _sha256_file(raw_path)
    target_raw_sha = _sha256_file(target_raw_path)
    target_reference = _target_reference(
        args.target_receipt.expanduser().resolve(),
        raw_sha256=target_raw_sha,
        raw_bytes=expected_bytes,
        pair_count=args.pair_count,
        required_path=DEFAULT_TARGET_RECEIPT if args.pair_count == DEFAULT_PAIR_COUNT else None,
        required_file_sha256=(
            DEFAULT_TARGET_RECEIPT_SHA256 if args.pair_count == DEFAULT_PAIR_COUNT else None
        ),
    )
    cache_path = args.cache.expanduser().resolve()
    fields, cache_sha = scorer._load_cache(
        cache_path,
        require_canonical_hash=not args.allow_noncanonical_cache,
    )
    if args.pair_count == DEFAULT_PAIR_COUNT and (
        target_reference.get("source_gt_cache_sha256") != cache_sha
        or target_reference.get("source_gt_cache_bytes") != cache_path.stat().st_size
    ):
        raise RawDebtError("zero-distortion target receipt does not bind the scorer cache in use")
    if output.exists():
        existing = _load_json(output, "final debt receipt")
        _validate_hash(existing, "receipt_sha256")
        if existing.get("schema") == LEGACY_FINAL_SCHEMA:
            if not _historical_v2_matches(
                existing,
                receipt_path=output,
                raw_sha256=raw_sha,
                target_raw_sha256=target_raw_sha,
                cache_sha256=cache_sha,
                pair_count=args.pair_count,
                stage_pairs=args.stage_pairs,
            ):
                raise RawDebtError("historical v2 final receipt belongs to different inputs")
            return existing

    upstream = args.upstream.expanduser().resolve()
    scorer_bundle = scorer._load_scorers(upstream, args.cpu_threads)
    scorer_custody = scorer._scorer_runtime_custody(upstream, scorer_bundle[2])
    scorer._validate_scorer_custody(scorer_custody)
    config = {
        "schema": CONFIG_SCHEMA,
        "raw": {"path": str(raw_path), "bytes": expected_bytes, "sha256": raw_sha},
        "target_raw": {
            "path": str(target_raw_path),
            "bytes": expected_bytes,
            "sha256": target_raw_sha,
            "role": "zero_distortion_live_target",
        },
        "target_reference": target_reference,
        "cache": {"path": str(cache_path), "bytes": cache_path.stat().st_size, "sha256": cache_sha},
        "pair_count": args.pair_count,
        "stage_pairs": args.stage_pairs,
        "scorer_batch_pairs": args.stage_pairs,
        "camera_hw": list(scorer.CAMERA_HW),
        "scorer_hw": list(scorer.SCORER_HW),
        "cpu_threads": args.cpu_threads,
        "seed": 20260719,
        "tool": {
            "path": str(Path(__file__).resolve()),
            "bytes": Path(__file__).resolve().stat().st_size,
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "scorer_adapter": {
            "path": str(Path(scorer.__file__).resolve()),
            "bytes": Path(scorer.__file__).resolve().stat().st_size,
            "sha256": _sha256_file(Path(scorer.__file__).resolve()),
        },
        "scorer_custody": scorer_custody,
    }
    config_sha = _sha256_bytes(_canonical(config))
    if output.exists():
        existing = _load_json(output, "final debt receipt")
        _validate_hash(existing, "receipt_sha256")
        _validate_existing_v3_final(
            existing,
            config=config,
            config_sha256=config_sha,
            scorer_custody=scorer_custody,
            stage_dir=stage_dir,
            state_path=state_path,
            launch_manifest_path=launch_manifest_path,
            pair_count=args.pair_count,
            stage_pairs=args.stage_pairs,
        )
        return existing
    if launch_manifest_path.exists() and not args.resume:
        raise RawDebtError("launch manifest exists; pass --resume")
    if state_path.exists() and not args.resume:
        raise RawDebtError("state exists; pass --resume")
    if stage_dir.exists() and any(stage_dir.iterdir()) and not args.resume:
        raise RawDebtError("stage directory is nonempty; pass --resume")
    launch_manifest = _ensure_launch_manifest(launch_manifest_path, config_sha256=config_sha)
    prefix_rows, stage_custody = _load_prefix(
        stage_dir=stage_dir,
        pair_count=args.pair_count,
        stage_pairs=args.stage_pairs,
        config_sha256=config_sha,
    )
    _reconcile_state(
        state_path,
        config_sha256=config_sha,
        prefix_pairs=len(prefix_rows),
        stages=stage_custody,
    )
    rows = list(prefix_rows)
    with raw_path.open("rb") as handle, target_raw_path.open("rb") as target_handle:
        start = len(rows)
        while start < args.pair_count:
            end = min(start + args.stage_pairs, args.pair_count)
            handle.seek(start * 2 * frame_bytes)
            target_handle.seek(start * 2 * frame_bytes)
            stage_bytes = (end - start) * 2 * frame_bytes
            candidate_payload = handle.read(stage_bytes)
            target_payload = target_handle.read(stage_bytes)
            if len(candidate_payload) != stage_bytes or len(target_payload) != stage_bytes:
                raise RawDebtError("decoded raw truncated during staged scoring")
            candidate = scorer.np.frombuffer(candidate_payload, dtype=scorer.np.uint8).reshape(
                end - start, 2, *scorer.CAMERA_HW, 3
            )
            target = scorer.np.frombuffer(target_payload, dtype=scorer.np.uint8).reshape(
                end - start, 2, *scorer.CAMERA_HW, 3
            )
            stage_rows = _score_batch_against_live_target(
                scorer_bundle[0],
                scorer_bundle[1],
                scorer_bundle[2],
                target_frame0=target[:, 0],
                target_frame1=target[:, 1],
                candidate_frame0=candidate[:, 0],
                candidate_frame1=candidate[:, 1],
                cached_labels=scorer.np.asarray(fields["lstars"][start:end]),
                cached_poses=scorer.np.asarray(fields["gt_poses"][start:end]),
            )
            for pair_id, row in zip(range(start, end), stage_rows, strict=True):
                row["pair_id"] = pair_id
            stage = _stage_payload(
                config_sha256=config_sha,
                start=start,
                end=end,
                rows=stage_rows,
            )
            path = _stage_path(stage_dir, start, end)
            _write_once(path, stage)
            rows.extend(stage_rows)
            stage_custody.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "stage_sha256": stage["stage_sha256"],
                    "pair_start": start,
                    "pair_end_exclusive": end,
                }
            )
            _atomic_json(
                state_path,
                _state_payload(
                    config_sha256=config_sha,
                    completed_pairs=end,
                    latest_stage_sha256=stage["stage_sha256"],
                ),
            )
            start = end

    aggregate = _aggregate_rows(rows, pair_count=args.pair_count)
    contest_reference_path = (
        args.contest_reference.expanduser().resolve() if args.contest_reference is not None else None
    )
    contest_archive_arg = getattr(args, "contest_archive", None)
    contest_archive_path = (
        contest_archive_arg.expanduser().resolve()
        if contest_archive_arg is not None
        else DEFAULT_REFERENCE_ARCHIVE.resolve()
        if contest_reference_path == DEFAULT_REFERENCE.resolve()
        else None
    )
    contest_reference = (
        _contest_reference(
            contest_reference_path,
            raw_sha256=raw_sha,
            raw_bytes=expected_bytes,
            local_archive_path=contest_archive_path,
        )
        if contest_reference_path is not None
        else None
    )
    input_end_barrier = _validate_input_end_barrier(
        config,
        contest_reference=contest_reference,
    )
    receipt = _with_hash(
        {
            "schema": SCHEMA,
            "written_at_utc": datetime.now(UTC).isoformat(),
            "config_sha256": config_sha,
            "config": config,
            "axis": f"[{platform.system()}-{platform.machine()} CPU advisory] NON-PROMOTABLE",
            "aggregate": aggregate,
            "pairs": rows,
            "stages": stage_custody,
            "scorer_custody": scorer_custody,
            "input_end_barrier": input_end_barrier,
            "launch_manifest": {
                "path": str(launch_manifest_path),
                "bytes": launch_manifest_path.stat().st_size,
                "sha256": _sha256_file(launch_manifest_path),
                "launch_sha256": launch_manifest["launch_sha256"],
            },
            "contest_cpu_reference_same_decoded_raw": contest_reference,
            "interpretation": (
                "batch-preserving per-pair/per-class/event routing telemetry for this exact target/candidate "
                "raw pair; contest reference, when present, supplies aggregate authority only"
            ),
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "pointer_moved": False,
        },
        "receipt_sha256",
    )
    _write_once(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--target-raw", type=Path, default=DEFAULT_TARGET_RAW)
    parser.add_argument("--target-receipt", type=Path, default=DEFAULT_TARGET_RECEIPT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument(
        "--launch-manifest",
        type=Path,
        help="write-once non-scientific launch metadata (default: <state>.launch.json)",
    )
    parser.add_argument("--cache", type=Path, default=scorer.DEFAULT_CACHE)
    parser.add_argument("--upstream", type=Path, default=scorer.DEFAULT_UPSTREAM)
    parser.add_argument("--contest-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--contest-archive",
        type=Path,
        help="optional local archive path whose bytes must match the contest reference",
    )
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--stage-pairs", type=int, default=DEFAULT_STAGE_PAIRS)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-noncanonical-cache", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run(args)
    except (OSError, RawDebtError, scorer.PredictorFloorError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "receipt": str(args.output.expanduser().resolve()),
                "receipt_sha256": receipt["receipt_sha256"],
                "pair_count": receipt["aggregate"]["pair_count"],
                "mean_d_seg": receipt["aggregate"]["mean_d_seg"],
                "mean_d_pose": receipt["aggregate"]["mean_d_pose"],
                "score_claim": False,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
