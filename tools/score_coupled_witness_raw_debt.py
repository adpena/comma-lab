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

SCHEMA = "tac.coupled_witness_raw_debt.v2"
STAGE_SCHEMA = "tac.coupled_witness_raw_debt_stage.v2"
STATE_SCHEMA = "tac.coupled_witness_raw_debt_state.v2"
DEFAULT_PAIR_COUNT = 600
DEFAULT_STAGE_PAIRS = 16
DEFAULT_RAW = Path("/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/out_serial/0.raw")
DEFAULT_TARGET_RAW = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m2_live_target_selection_20260720T1528Z/inflated/0.raw"
)
DEFAULT_TARGET_RECEIPT = REPO / ".omx/research/m2_live_target_selection_20260720T1548Z.json"
DEFAULT_REFERENCE = Path(
    "/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/"
    "modal_contest_cpu/harvest_fc01KXXRAR/contest_auth_eval.json"
)


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


def _contest_reference(path: Path, *, raw_sha256: str, raw_bytes: int) -> dict[str, Any]:
    payload = _load_json(path, "contest reference")
    try:
        files = payload["provenance"]["inflated_output_manifest"]["payload"]["files"]
        matching = [row for row in files if row.get("sha256") == raw_sha256 and row.get("bytes") == raw_bytes]
    except (KeyError, TypeError) as exc:
        raise RawDebtError("contest reference lacks inflated-output custody") from exc
    if len(matching) != 1:
        raise RawDebtError("contest reference does not bind this exact decoded raw")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "score_axis": payload.get("score_axis"),
        "evidence_grade": payload.get("evidence_grade"),
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "archive_sha256": payload.get("provenance", {}).get("archive_sha256"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "canonical_score": payload.get("canonical_score"),
        "decoded_raw": dict(matching[0]),
    }


def _target_reference(path: Path, *, raw_sha256: str, raw_bytes: int, pair_count: int) -> dict[str, Any]:
    """Bind the exact live target raw to its measured zero-distortion receipt."""

    payload = _load_json(path, "zero-distortion target reference")
    try:
        candidate = payload["candidate"]
        source = payload["source_custody"]
        bound = (
            candidate["inflated_raw_sha256"] == raw_sha256
            and candidate["inflated_raw_bytes"] == raw_bytes
            and source["pair_count"] == pair_count
            and float(candidate["d_seg"]) == 0.0
            and float(candidate["d_pose"]) == 0.0
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RawDebtError("zero-distortion target reference lacks required custody") from exc
    if not bound:
        raise RawDebtError("zero-distortion target reference does not bind this exact raw and population")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "schema": payload.get("schema"),
        "axis": payload.get("axis"),
        "score_claim": payload.get("score_claim"),
        "archive_sha256": candidate.get("archive_sha256"),
        "decoded_raw_sha256": raw_sha256,
        "decoded_raw_bytes": raw_bytes,
        "d_seg": candidate.get("d_seg"),
        "d_pose": candidate.get("d_pose"),
        "pair_count": pair_count,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], *, pair_count: int) -> dict[str, Any]:
    if [row.get("pair_id") for row in rows] != list(range(pair_count)):
        raise RawDebtError("stage rows are not exactly the canonical contiguous pair sequence")
    try:
        mean_d_seg = math.fsum(float(row["d_seg"]) for row in rows) / pair_count
        mean_d_pose = math.fsum(float(row["d_pose"]) for row in rows) / pair_count
        per_class = scorer._aggregate_per_class_debt(rows)
        events = [
            [int(row["pair_id"]), *event]
            for row in rows
            for event in row["seg_events"]
        ]
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
        events = [
            [int(row), int(col), int(labels[row, col]), int(prediction[row, col])]
            for row, col in coordinates
        ]
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
        if payload.get("schema") != STAGE_SCHEMA or payload.get("config_sha256") != config_sha256:
            raise RawDebtError(f"stage identity differs: {path}")
        if start != expected_start or end != min(start + stage_pairs, pair_count):
            raise RawDebtError(f"stage sequence is not one canonical prefix: {path}")
        stage_rows = payload.get("rows")
        if not isinstance(stage_rows, list) or [row.get("pair_id") for row in stage_rows] != list(range(start, end)):
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = args.raw.expanduser().resolve()
    target_raw_path = args.target_raw.expanduser().resolve()
    output = args.output.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    stage_dir = args.stage_dir.expanduser().resolve()
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
    )
    cache_path = args.cache.expanduser().resolve()
    fields, cache_sha = scorer._load_cache(
        cache_path,
        require_canonical_hash=not args.allow_noncanonical_cache,
    )
    pointer = scorer._effective_pointer_target()
    config = {
        "schema": STATE_SCHEMA,
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
        "pointer": pointer,
        "tool": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
        "scorer_adapter": {
            "path": str(Path(scorer.__file__).resolve()),
            "sha256": _sha256_file(Path(scorer.__file__).resolve()),
        },
    }
    config_sha = _sha256_bytes(_canonical(config))
    if output.exists():
        existing = _load_json(output, "final debt receipt")
        _validate_hash(existing, "receipt_sha256")
        if existing.get("config_sha256") != config_sha:
            raise RawDebtError("existing final receipt belongs to different inputs")
        return existing
    if state_path.exists() and not args.resume:
        raise RawDebtError("state exists; pass --resume")
    if stage_dir.exists() and any(stage_dir.iterdir()) and not args.resume:
        raise RawDebtError("stage directory is nonempty; pass --resume")
    prefix_rows, stage_custody = _load_prefix(
        stage_dir=stage_dir,
        pair_count=args.pair_count,
        stage_pairs=args.stage_pairs,
        config_sha256=config_sha,
    )
    if state_path.exists():
        state = _load_json(state_path, "resume state")
        if state.get("schema") != STATE_SCHEMA or state.get("config_sha256") != config_sha:
            raise RawDebtError("resume state belongs to different inputs")
        if state.get("completed_pairs") != len(prefix_rows):
            raise RawDebtError("resume state and preserved stages disagree")
    elif prefix_rows:
        raise RawDebtError("preserved stages exist without their resume state")

    scorer_bundle = scorer._load_scorers(args.upstream.expanduser().resolve(), args.cpu_threads)
    scorer_custody = scorer._scorer_runtime_custody(args.upstream.expanduser().resolve(), scorer_bundle[2])
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
                {
                    "schema": STATE_SCHEMA,
                    "config_sha256": config_sha,
                    "completed_pairs": end,
                    "latest_stage_sha256": stage["stage_sha256"],
                },
            )
            start = end

    aggregate = _aggregate_rows(rows, pair_count=args.pair_count)
    contest_reference = (
        _contest_reference(args.contest_reference.expanduser().resolve(), raw_sha256=raw_sha, raw_bytes=expected_bytes)
        if args.contest_reference is not None
        else None
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
    parser.add_argument("--cache", type=Path, default=scorer.DEFAULT_CACHE)
    parser.add_argument("--upstream", type=Path, default=scorer.DEFAULT_UPSTREAM)
    parser.add_argument("--contest-reference", type=Path, default=DEFAULT_REFERENCE)
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
