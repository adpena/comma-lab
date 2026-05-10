#!/usr/bin/env python3
"""Materialize a completed PR101 Kaggle proxy candidate for local handoff.

This tool is deliberately bounded: it reads a completed Kaggle proxy
``best_proxy_candidate.json`` and writes deterministic local JSON artifacts
that a real archive builder can consume later. It does not build an archive,
does not run inflate, does not dispatch remote jobs, and does not claim score.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_bytes, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/materialize_kaggle_pr101_proxy_candidate.py"
SCHEMA = "pr101_kaggle_proxy_candidate_materialization_v1"
HANDOFF_SCHEMA = "pr101_kaggle_proxy_candidate_archive_builder_handoff_v1"
PARAM_SCHEMA = "pr101_kaggle_proxy_candidate_params_v1"
BIAS_RUNTIME_PARAM_SCHEMA = "pr101_kaggle_proxy_bias_runtime_params_v1"
EXPECTED_EVIDENCE_SEMANTICS = "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval"
OPTIMIZER_GUIDED_EVIDENCE_SEMANTICS = "offline_optimizer_guided_proxy_queue_not_exact_auth_eval"
ALLOWED_EVIDENCE_SEMANTICS = frozenset(
    {
        EXPECTED_EVIDENCE_SEMANTICS,
        OPTIMIZER_GUIDED_EVIDENCE_SEMANTICS,
        "a1_runtime_variant_planning_only",
        "macos_cpu_calibrated_ranking_not_dispatch_evidence",
    }
)
OPTIMIZER_QUEUE_SCHEMAS = frozenset(
    {
        "optimizer_candidate_queue_v1",
        "optimizer_guided_candidate_queue_v1",
    }
)
DEFAULT_INPUT_CANDIDATE = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/best_proxy_candidate.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/local_materialization"
)
HANDOFF_NAME = "archive_builder_handoff.json"
MANIFEST_NAME = "materialization_manifest.json"
EXPECTED_OUTPUT_NAMES = frozenset({HANDOFF_NAME, MANIFEST_NAME})
PARAM_KEYS = (
    "bias_b",
    "bias_g",
    "bias_r",
    "delta_scale",
    "latent_delta_scale",
    "smooth_weight",
)
BIAS_RUNTIME_PARAM_KEYS = (
    "bias_b",
    "bias_g",
    "bias_r",
)
PARAM_SCHEMAS = {
    PARAM_SCHEMA: PARAM_KEYS,
    BIAS_RUNTIME_PARAM_SCHEMA: BIAS_RUNTIME_PARAM_KEYS,
}
FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "score_claim_valid": False,
    "ready_for_exact_eval_dispatch": False,
    "proxy_only": True,
}
QUEUE_FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "ready_for_exact_eval_dispatch": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
}
DISPATCH_BLOCKERS = [
    "proxy_substrate_not_contest_exact_eval",
    "no_archive_zip_emitted",
    "no_inflate_runtime_emitted",
    "no_contest_cuda_auth_eval",
    "real_archive_builder_handoff_only",
]


class MaterializationError(ValueError):
    """Raised when proxy-candidate handoff materialization must fail closed."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _canonical_json_sha256(payload: Any) -> str:
    return sha256_bytes(json_text(payload).encode("utf-8"))


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterializationError(f"{field} must be a JSON object")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MaterializationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MaterializationError(f"{field} must be finite")
    return result


def validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical validated candidate payload."""

    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise MaterializationError("candidate_id must be a non-empty string")

    evidence_semantics = candidate.get("evidence_semantics")
    if evidence_semantics not in ALLOWED_EVIDENCE_SEMANTICS:
        raise MaterializationError(
            "candidate evidence_semantics must be one of "
            f"{sorted(ALLOWED_EVIDENCE_SEMANTICS)!r}"
        )

    for flag, expected in FALSE_AUTHORITY_FLAGS.items():
        if candidate.get(flag) is not expected:
            raise MaterializationError(f"{flag} must be {expected!r}")

    param_schema = candidate.get("param_schema") or PARAM_SCHEMA
    if param_schema not in PARAM_SCHEMAS:
        raise MaterializationError(
            f"param_schema must be one of {sorted(PARAM_SCHEMAS)}"
        )
    param_keys = PARAM_SCHEMAS[str(param_schema)]

    params_raw = _require_mapping(candidate.get("params"), "params")
    params: dict[str, float] = {}
    for key in param_keys:
        params[key] = _require_finite_number(params_raw.get(key), f"params.{key}")
    extra_params = sorted(set(params_raw) - set(param_keys))
    if extra_params:
        raise MaterializationError(f"params has unsupported keys: {extra_params}")

    proxy_components_raw = _require_mapping(
        candidate.get("proxy_components"),
        "proxy_components",
    )
    proxy_components = {
        str(key): _require_finite_number(value, f"proxy_components.{key}")
        for key, value in sorted(proxy_components_raw.items())
    }

    return {
        "candidate_id": candidate_id,
        "evidence_semantics": evidence_semantics,
        "optimizer": candidate.get("optimizer"),
        "optimizer_status": candidate.get("optimizer_status"),
        "params": params,
        "param_schema": param_schema,
        "param_keys": list(param_keys),
        "proxy_components": proxy_components,
        "proxy_objective": _require_finite_number(
            candidate.get("proxy_objective"),
            "proxy_objective",
        ),
        "proxy_only": True,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "score_claim_valid": False,
        "trial_index": candidate.get("trial_index"),
    }


def _file_record(path: Path) -> dict[str, Any]:
    path = _repo_path(path)
    return {
        "path": _repo_rel(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_sweep_manifest(candidate_path: Path) -> Path | None:
    sibling = candidate_path.parent / "proxy_sweep_manifest.json"
    return sibling if sibling.is_file() else None


def _queue_rows(queue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if queue.get("schema") not in OPTIMIZER_QUEUE_SCHEMAS:
        raise MaterializationError(
            "candidate queue schema must be one of "
            f"{sorted(OPTIMIZER_QUEUE_SCHEMAS)!r}"
        )
    rows = queue.get("top_k_forensic")
    if not isinstance(rows, list):
        rows = queue.get("top_k")
    if not isinstance(rows, list):
        raise MaterializationError("candidate queue must contain top_k or top_k_forensic rows")
    return [row for row in rows if isinstance(row, Mapping)]


def _queue_row_params(row: Mapping[str, Any]) -> Mapping[str, Any]:
    params = row.get("candidate_params") or row.get("op_params")
    if not isinstance(params, Mapping):
        raise MaterializationError("candidate queue row must contain candidate_params or op_params")
    return params


def _queue_row_is_bias_only(row: Mapping[str, Any]) -> bool:
    try:
        params = _queue_row_params(row)
    except MaterializationError:
        return False
    return set(params) == set(BIAS_RUNTIME_PARAM_KEYS)


def candidate_from_optimizer_queue(
    queue: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    queue_index: int | None = None,
) -> dict[str, Any]:
    """Return a bias-only candidate payload from a fail-closed optimizer queue."""

    rows = _queue_rows(queue)
    if queue_index is not None and candidate_id is not None:
        raise MaterializationError("choose either candidate_id or queue_index, not both")
    if queue_index is not None:
        if queue_index < 0 or queue_index >= len(rows):
            raise MaterializationError(f"queue_index out of range: {queue_index}")
        selected = rows[queue_index]
    elif candidate_id is not None:
        matches = [row for row in rows if row.get("candidate_id") == candidate_id]
        if len(matches) != 1:
            raise MaterializationError(
                f"candidate_id must match exactly one queue row: {candidate_id!r}"
            )
        selected = matches[0]
    else:
        selected = next((row for row in rows if _queue_row_is_bias_only(row)), None)
        if selected is None:
            raise MaterializationError("candidate queue has no bias-only row to materialize")

    selected_id = str(selected.get("candidate_id") or "")
    if not selected_id:
        raise MaterializationError("candidate queue row candidate_id must be non-empty")
    for field, expected in QUEUE_FALSE_AUTHORITY_FLAGS.items():
        if selected.get(field) is not expected:
            raise MaterializationError(f"{selected_id}: {field} must be {expected!r}")

    params_raw = _queue_row_params(selected)
    if set(params_raw) != set(BIAS_RUNTIME_PARAM_KEYS):
        raise MaterializationError(
            f"{selected_id}: optimizer queue materialization only supports bias-only "
            f"params {list(BIAS_RUNTIME_PARAM_KEYS)}"
        )
    params = {
        key: _require_finite_number(params_raw.get(key), f"{selected_id}.{key}")
        for key in BIAS_RUNTIME_PARAM_KEYS
    }
    proxy_components_raw = selected.get("proxy_components")
    proxy_components = (
        dict(proxy_components_raw)
        if isinstance(proxy_components_raw, Mapping)
        else {"rank_score": selected.get("rank_score")}
    )
    return {
        "candidate_id": selected_id,
        "evidence_semantics": selected.get("evidence_semantics")
        or OPTIMIZER_GUIDED_EVIDENCE_SEMANTICS,
        "optimizer": selected.get("optimizer") or queue.get("optimizer"),
        "optimizer_status": selected.get("optimizer_status") or queue.get("optimizer_status"),
        "param_schema": BIAS_RUNTIME_PARAM_SCHEMA,
        "params": params,
        "proxy_components": proxy_components,
        "proxy_objective": selected.get("proxy_objective") or selected.get("rank_score"),
        "proxy_only": True,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "score_claim_valid": False,
        "trial_index": selected.get("trial_index"),
    }


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    if not output_dir.is_dir():
        raise MaterializationError(f"output path is not a directory: {output_dir}")

    existing = list(output_dir.iterdir())
    if not existing:
        return
    if not force:
        raise MaterializationError(
            f"output directory is not empty; pass --force to replace: {output_dir}"
        )

    unexpected = sorted(path.name for path in existing if path.name not in EXPECTED_OUTPUT_NAMES)
    if unexpected:
        raise MaterializationError(
            "refusing --force because output directory contains unexpected files: "
            f"{unexpected}"
        )
    for path in existing:
        if path.is_dir():
            raise MaterializationError(
                f"refusing --force because expected output path is a directory: {path}"
            )
        path.unlink()


def build_handoff(candidate: Mapping[str, Any], source_record: dict[str, Any]) -> dict[str, Any]:
    validated = validate_candidate(candidate)
    return {
        "schema": HANDOFF_SCHEMA,
        "candidate_id": validated["candidate_id"],
        "param_schema": validated["param_schema"],
        "params": validated["params"],
        "proxy_evidence": {
            "evidence_semantics": validated["evidence_semantics"],
            "optimizer": validated["optimizer"],
            "optimizer_status": validated["optimizer_status"],
            "proxy_components": validated["proxy_components"],
            "proxy_objective": validated["proxy_objective"],
            "trial_index": validated["trial_index"],
        },
        "source_candidate": source_record,
        "evidence_boundary": {
            "proxy_only": True,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "exact_auth_eval_performed": False,
            "contest_cuda_auth_eval": False,
            "mps_auth_eval": False,
            "archive_zip_emitted": False,
            "inflate_runtime_emitted": False,
            "dispatch_blockers": list(DISPATCH_BLOCKERS),
        },
        "archive_builder_handoff_contract": {
            "status": "pending_real_archive_builder",
            "builder_must_consume": {
                "param_schema": validated["param_schema"],
                "param_keys": list(validated["param_keys"]),
                "source_candidate_sha256": source_record["sha256"],
            },
            "builder_must_emit_before_dispatch": [
                "byte_closed_archive_zip",
                "inflate_runtime_that_consumes_the_materialized_params",
                "candidate_archive_manifest_with_archive_bytes_and_sha256",
                "runtime_tree_sha256",
                "parser_section_manifest_or_explicit_non_applicability_rationale",
                "local_inflate_or_runtime_consumption_proof",
            ],
            "builder_must_not": [
                "claim_score_from_this_proxy_artifact",
                "set_ready_for_exact_eval_dispatch_true_from_proxy_evidence",
                "dispatch_remote_eval_without_a_separate_lane_claim",
                "load_scorers_at_inflate_time",
            ],
            "next_authoritative_gate": (
                "claimed exact CUDA auth eval of a byte-closed archive/runtime packet"
            ),
        },
    }


def materialize_candidate(
    *,
    candidate_path: Path = DEFAULT_INPUT_CANDIDATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    candidate_path = _repo_path(candidate_path)
    output_dir = _repo_path(output_dir)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate JSON not found: {candidate_path}")
    _prepare_output_dir(output_dir, force=force)

    candidate_raw = read_json(candidate_path)
    candidate = _require_mapping(candidate_raw, "candidate")
    source_record = _file_record(candidate_path)
    source_record["canonical_payload_sha256"] = _canonical_json_sha256(candidate)

    handoff = build_handoff(candidate, source_record)
    handoff_path = output_dir / HANDOFF_NAME
    write_json(handoff_path, handoff)

    input_files = [source_record]
    sweep_manifest_path = _source_sweep_manifest(candidate_path)
    if sweep_manifest_path is not None:
        input_files.append(_file_record(sweep_manifest_path))

    outputs = [_file_record(handoff_path)]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL_NAME,
        "candidate_id": handoff["candidate_id"],
        "input_files": input_files,
        "output_files": outputs,
        "handoff_artifact": _repo_rel(handoff_path),
        "candidate_params": handoff["params"],
        "param_schema": handoff["param_schema"],
        "proxy_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "exact_auth_eval_performed": False,
        "contest_cuda_auth_eval": False,
        "mps_auth_eval": False,
        "archive_zip_emitted": False,
        "inflate_runtime_emitted": False,
        "evidence_semantics": EXPECTED_EVIDENCE_SEMANTICS,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "archive_builder_handoff_contract": handoff["archive_builder_handoff_contract"],
    }
    manifest["manifest_sha256_excluding_self"] = _canonical_json_sha256(manifest)
    manifest_path = output_dir / MANIFEST_NAME
    write_json(manifest_path, manifest)
    return manifest


def materialize_optimizer_queue_candidate(
    *,
    queue_path: Path,
    output_dir: Path,
    candidate_id: str | None = None,
    queue_index: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    queue_path = _repo_path(queue_path)
    output_dir = _repo_path(output_dir)
    if not queue_path.is_file():
        raise FileNotFoundError(f"candidate queue JSON not found: {queue_path}")
    _prepare_output_dir(output_dir, force=force)

    queue_raw = read_json(queue_path)
    queue = _require_mapping(queue_raw, "candidate_queue")
    candidate = candidate_from_optimizer_queue(
        queue,
        candidate_id=candidate_id,
        queue_index=queue_index,
    )
    source_record = _file_record(queue_path)
    source_record["canonical_payload_sha256"] = _canonical_json_sha256(queue)
    source_record["source_kind"] = "optimizer_candidate_queue"

    handoff = build_handoff(candidate, source_record)
    handoff_path = output_dir / HANDOFF_NAME
    write_json(handoff_path, handoff)

    outputs = [_file_record(handoff_path)]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL_NAME,
        "candidate_id": handoff["candidate_id"],
        "input_files": [source_record],
        "output_files": outputs,
        "handoff_artifact": _repo_rel(handoff_path),
        "candidate_params": handoff["params"],
        "param_schema": handoff["param_schema"],
        "proxy_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "exact_auth_eval_performed": False,
        "contest_cuda_auth_eval": False,
        "mps_auth_eval": False,
        "archive_zip_emitted": False,
        "inflate_runtime_emitted": False,
        "evidence_semantics": handoff["proxy_evidence"]["evidence_semantics"],
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "archive_builder_handoff_contract": handoff["archive_builder_handoff_contract"],
        "source_queue_selection": {
            "candidate_id": candidate_id,
            "queue_index": queue_index,
            "selected_candidate_id": handoff["candidate_id"],
        },
    }
    manifest["manifest_sha256_excluding_self"] = _canonical_json_sha256(manifest)
    manifest_path = output_dir / MANIFEST_NAME
    write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_INPUT_CANDIDATE)
    parser.add_argument(
        "--candidate-queue",
        type=Path,
        default=None,
        help=(
            "optimizer_candidate_queue_v1 or optimizer_guided_candidate_queue_v1; "
            "materializes a bias-only row into the existing archive-builder handoff"
        ),
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="candidate_id to select from --candidate-queue; default is first bias-only row",
    )
    parser.add_argument(
        "--queue-index",
        type=int,
        default=None,
        help="zero-based row index to select from --candidate-queue",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.candidate_queue is not None:
        manifest = materialize_optimizer_queue_candidate(
            queue_path=args.candidate_queue,
            candidate_id=args.candidate_id,
            queue_index=args.queue_index,
            output_dir=args.output_dir,
            force=args.force,
        )
    else:
        if args.candidate_id is not None or args.queue_index is not None:
            raise MaterializationError("--candidate-id/--queue-index require --candidate-queue")
        manifest = materialize_candidate(
            candidate_path=args.candidate,
            output_dir=args.output_dir,
            force=args.force,
        )
    print(json_text({
        "schema": "pr101_kaggle_proxy_candidate_materialization_stdout_v1",
        "manifest": _repo_rel(_repo_path(args.output_dir) / MANIFEST_NAME),
        "handoff_artifact": manifest["handoff_artifact"],
        "candidate_id": manifest["candidate_id"],
        "proxy_only": True,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
