#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an inverse-action compiler hint from dynamic sparse gate scores."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.dynamic_sparse_gate_oracle import (  # noqa: E402
    DEFAULT_RATE_SCORE_PER_BYTE,
    DynamicSparseGateOracleError,
    operation_set_compiler_hint_from_channel_gate_scores,
    operation_set_compiler_hint_from_gate_scores,
    operation_set_compiler_hint_from_materializer_feedback,
    operation_set_compiler_hint_from_observation_feedback,
)
from tac.optimization.inverse_steganalysis_acquisition import (  # noqa: E402
    InverseSteganalysisAcquisitionError,
    observations_from_queue_observation,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise SystemExit(f"{label}: expected JSON object")
    return dict(payload)


def _repo_rel(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _object_payload(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"{label}: expected JSON object")
    return value


def _operation_candidates(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        raw = payload
    else:
        obj = _object_payload(payload, label="--operation-candidates")
        raw = obj.get("operation_candidates", obj.get("selected_operations"))
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise SystemExit("--operation-candidates must contain operation_candidates[]")
    return list(raw)


def _observations(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        raw = payload
    else:
        obj = _object_payload(payload, label="--observations")
        raw = obj.get("observations", obj.get("rows"))
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise SystemExit("--observations must contain observations[] or rows[]")
    return list(raw)


def _coefficients(payload: Any) -> Any:
    if isinstance(payload, Mapping) and "coefficients" in payload:
        return payload["coefficients"]
    return payload


def _ids_from_args_or_payload(
    *,
    arg_value: str | None,
    payload: Any,
    key: str,
    label: str,
) -> list[str]:
    if arg_value:
        out = [item.strip() for item in arg_value.split(",") if item.strip()]
    elif isinstance(payload, Mapping):
        raw = payload.get(key, [])
        out = [str(item) for item in raw] if isinstance(raw, list) else []
    else:
        out = []
    if not out:
        raise SystemExit(f"{label} is required")
    return out


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    try:
        write_json_artifact(path, payload, allow_overwrite=overwrite)
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc


def _channel_ids(arg_value: str | None) -> tuple[str, ...]:
    if arg_value:
        return tuple(item.strip() for item in arg_value.split(",") if item.strip())
    return ("rate_saving", "receiver_proof", "runtime_efficiency")


def _queue_feedback_observations(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    if args.queue_performance_runtime_identity is None:
        raise SystemExit(
            "--queue-performance-runtime-identity is required with --queue-observation"
        )
    if args.queue_performance_cache_identity is None:
        raise SystemExit(
            "--queue-performance-cache-identity is required with --queue-observation"
        )
    runtime_identity = _load_mapping(
        args.queue_performance_runtime_identity,
        label="queue performance runtime identity",
    )
    cache_identity = _load_mapping(
        args.queue_performance_cache_identity,
        label="queue performance cache identity",
    )
    candidate_map = (
        _load_mapping(args.queue_performance_candidate_map, label="queue performance candidate map")
        if args.queue_performance_candidate_map is not None
        else None
    )
    observations: list[Mapping[str, Any]] = []
    for path in args.queue_observation or []:
        observations.extend(
            observations_from_queue_observation(
                _load_mapping(path, label="queue observation"),
                runtime_identity=runtime_identity,
                cache_identity=cache_identity,
                axis=args.queue_performance_axis,
                source_path=_repo_rel(path, args.repo_root),
                candidate_id_by_experiment=candidate_map,
            )
        )
    return observations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--operation-candidates", type=Path)
    source.add_argument("--observations", type=Path)
    source.add_argument("--materializer-feedback", type=Path)
    source.add_argument("--queue-observation", type=Path, action="append")
    parser.add_argument("--coefficients", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--operation-set-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-ids", default=None)
    parser.add_argument("--channel-ids", default=None)
    parser.add_argument("--max-operations", type=int, default=None)
    parser.add_argument("--min-abs-gate", type=float, default=0.0)
    parser.add_argument("--lane-id", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--coefficient-mode", default="linear")
    parser.add_argument("--shared-projection-id", default=None)
    parser.add_argument("--topology-id", default=None)
    parser.add_argument("--rate-score-per-byte", type=float, default=None)
    parser.add_argument("--queue-performance-runtime-identity", type=Path, default=None)
    parser.add_argument("--queue-performance-cache-identity", type=Path, default=None)
    parser.add_argument("--queue-performance-candidate-map", type=Path, default=None)
    parser.add_argument(
        "--queue-performance-axis",
        default="[local-queue-observation advisory]",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.observations is not None or args.queue_observation:
            observations = (
                _observations(_load_json(args.observations))
                if args.observations is not None
                else _queue_feedback_observations(args)
            )
            hint = operation_set_compiler_hint_from_observation_feedback(
                observations,
                operation_set_id=args.operation_set_id,
                channel_ids=_channel_ids(args.channel_ids),
                max_operations=args.max_operations,
                min_abs_gate=args.min_abs_gate,
                lane_id=args.lane_id,
                candidate_id=args.candidate_id,
                coefficient_mode=(
                    args.coefficient_mode
                    if args.coefficient_mode != "linear"
                    else "observation_feedback"
                ),
                shared_projection_id=args.shared_projection_id,
                topology_id=args.topology_id,
                rate_score_per_byte=(
                    args.rate_score_per_byte
                    if args.rate_score_per_byte is not None
                    else DEFAULT_RATE_SCORE_PER_BYTE
                ),
            )
            _write_json(args.out, hint, overwrite=args.overwrite_output)
            print(str(args.out))
            return 0
        if args.materializer_feedback is not None:
            hint = operation_set_compiler_hint_from_materializer_feedback(
                _load_json(args.materializer_feedback),
                operation_set_id=args.operation_set_id,
                source_path=_repo_rel(args.materializer_feedback, args.repo_root),
                channel_ids=_channel_ids(args.channel_ids),
                max_operations=args.max_operations,
                min_abs_gate=args.min_abs_gate,
                lane_id=args.lane_id,
                candidate_id=args.candidate_id,
                coefficient_mode=(
                    args.coefficient_mode
                    if args.coefficient_mode != "linear"
                    else "materializer_feedback"
                ),
                shared_projection_id=args.shared_projection_id,
                topology_id=args.topology_id,
                rate_score_per_byte=(
                    args.rate_score_per_byte
                    if args.rate_score_per_byte is not None
                    else DEFAULT_RATE_SCORE_PER_BYTE
                ),
            )
            _write_json(args.out, hint, overwrite=args.overwrite_output)
            print(str(args.out))
            return 0
        if args.operation_candidates is None or args.coefficients is None:
            raise SystemExit("--coefficients is required with --operation-candidates")
        candidate_payload = _load_json(args.operation_candidates)
        coefficient_payload = _load_json(args.coefficients)
        candidates = _operation_candidates(candidate_payload)
        coefficients = _coefficients(coefficient_payload)
        channel_mode = bool(args.channel_ids) or (
            isinstance(coefficient_payload, Mapping)
            and isinstance(coefficient_payload.get("channel_ids"), list)
        )
        if channel_mode:
            source_ids = _ids_from_args_or_payload(
                arg_value=args.source_ids,
                payload=coefficient_payload,
                key="source_ids",
                label="--source-ids",
            )
            channel_ids = _ids_from_args_or_payload(
                arg_value=args.channel_ids,
                payload=coefficient_payload,
                key="channel_ids",
                label="--channel-ids",
            )
            hint = operation_set_compiler_hint_from_channel_gate_scores(
                candidates,
                coefficients,
                operation_set_id=args.operation_set_id,
                source_ids=source_ids,
                channel_ids=channel_ids,
                max_operations=args.max_operations,
                min_abs_gate=args.min_abs_gate,
                lane_id=args.lane_id,
                candidate_id=args.candidate_id,
                coefficient_mode=args.coefficient_mode,
                shared_projection_id=args.shared_projection_id,
                topology_id=args.topology_id,
            )
        else:
            source_ids = (
                [item.strip() for item in args.source_ids.split(",") if item.strip()]
                if args.source_ids
                else []
            )
            hint = operation_set_compiler_hint_from_gate_scores(
                candidates,
                coefficients,
                operation_set_id=args.operation_set_id,
                source_ids=source_ids,
                max_operations=args.max_operations,
                min_abs_gate=args.min_abs_gate,
                lane_id=args.lane_id,
                candidate_id=args.candidate_id,
                coefficient_mode=args.coefficient_mode,
            )
    except DynamicSparseGateOracleError as exc:
        raise SystemExit(str(exc)) from exc
    except InverseSteganalysisAcquisitionError as exc:
        raise SystemExit(str(exc)) from exc
    _write_json(args.out, hint, overwrite=args.overwrite_output)
    print(str(args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
