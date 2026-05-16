#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the TT5L move-level feasibility custody artifact.

The Dykstra score-axis artifact is a planning sanity check, not proof that the
TT5L move-level constraints have a non-empty intersection. This builder binds a
separate solver/proof artifact plus the Dykstra artifact by SHA-256 before the
L5 v2 staircase can advance to side-info curves or timing smokes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.l5_staircase_v2 import (  # noqa: E402
    TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
    TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS,
    TT5L_DYKSTRA_SUBSTRATE_ID,
    TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH,
    TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
    TT5L_MOVE_LEVEL_FEASIBILITY_SCHEMA,
    TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH,
    tt5l_move_level_feasibility_status,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_or_original(path: Path, *, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(path)


def _repo_path(value: str, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _parse_command_argv(value: str) -> list[str]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("command argv must be a JSON array") from exc
    if (
        not isinstance(loaded, list)
        or not loaded
        or not all(isinstance(item, str) and item.strip() for item in loaded)
    ):
        raise argparse.ArgumentTypeError("command argv must be a non-empty string array")
    return loaded


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"{label} must be a non-empty JSON object: {path}")
    return loaded


def _float_field(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        raise ValueError(f"proof artifact {field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"proof artifact {field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"proof artifact {field} must be finite")
    return parsed


def _proof_payload_fields(proof_payload: dict[str, Any]) -> tuple[float, float, list[str]]:
    if proof_payload.get("predicate_passed") is not True:
        raise ValueError("proof artifact predicate_passed must be true")
    if proof_payload.get("move_level_constraint_proof") is not True:
        raise ValueError("proof artifact move_level_constraint_proof must be true")

    residual_max = _float_field(proof_payload, "residual_max")
    residual_tolerance = _float_field(proof_payload, "residual_tolerance")
    if residual_max < 0.0:
        raise ValueError("proof artifact residual_max must be non-negative")
    if residual_tolerance <= 0.0:
        raise ValueError("proof artifact residual_tolerance must be positive")
    if residual_max > residual_tolerance:
        raise ValueError("proof artifact residual_max exceeds residual_tolerance")

    raw_constraint_ids = proof_payload.get("constraint_set_ids")
    if not isinstance(raw_constraint_ids, list):
        raise ValueError("proof artifact constraint_set_ids must be a list")
    constraint_ids = sorted({str(item) for item in raw_constraint_ids if str(item)})
    if set(constraint_ids) != TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS:
        raise ValueError("proof artifact constraint_set_ids do not match TT5L set")
    return residual_max, residual_tolerance, constraint_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root used for path custody validation.",
    )
    parser.add_argument(
        "--proof-artifact",
        required=True,
        help="Repo-local solver/proof artifact with move-level residual evidence.",
    )
    parser.add_argument(
        "--proof-command-argv-json",
        required=True,
        type=_parse_command_argv,
        help="Exact command argv that generated the proof artifact.",
    )
    parser.add_argument(
        "--score-axis-sanity-artifact",
        default=TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
        help="Repo-local Dykstra score-axis sanity artifact to bind by SHA-256.",
    )
    parser.add_argument(
        "--output-json",
        default=TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH,
        help="Output custody artifact path.",
    )
    return parser


def _payload_from_args(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any]:
    proof_artifact = _repo_path(args.proof_artifact, repo_root=repo_root)
    if not proof_artifact.is_file():
        raise FileNotFoundError(f"proof artifact missing: {proof_artifact}")
    score_axis_artifact = _repo_path(args.score_axis_sanity_artifact, repo_root=repo_root)
    if not score_axis_artifact.is_file():
        raise FileNotFoundError(
            f"score-axis sanity artifact missing: {score_axis_artifact}"
        )

    proof_payload = _load_json_object(proof_artifact, label="proof artifact")
    residual_max, residual_tolerance, constraint_ids = _proof_payload_fields(
        proof_payload
    )

    return {
        "schema": TT5L_MOVE_LEVEL_FEASIBILITY_SCHEMA,
        "subject_id": TT5L_DYKSTRA_SUBSTRATE_ID,
        "predicate_id": TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
        "predicate_passed": True,
        "move_level_constraint_proof": True,
        "residual_max": residual_max,
        "residual_tolerance": residual_tolerance,
        "constraint_set_ids": constraint_ids,
        "constraint_set_count": len(constraint_ids),
        "proof_artifact_path": _repo_relative_or_original(
            proof_artifact,
            repo_root=repo_root,
        ),
        "proof_artifact_sha256": _sha256_file(proof_artifact),
        "proof_schema": proof_payload.get("schema"),
        "proof_predicate_id": proof_payload.get("predicate_id"),
        "score_axis_sanity_artifact_path": _repo_relative_or_original(
            score_axis_artifact,
            repo_root=repo_root,
        ),
        "score_axis_sanity_artifact_sha256": _sha256_file(score_axis_artifact),
        "generated_by_tool": TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH,
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "command_argv": args.proof_command_argv_json,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    try:
        payload = _payload_from_args(args, repo_root=repo_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"[tt5l-move-feasibility] FATAL: {exc}", file=sys.stderr)
        return 2

    status = tt5l_move_level_feasibility_status(repo_root=repo_root)
    if status["artifact_valid"] is not True:
        print(json.dumps(status, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"artifact_path": str(output_path), "status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
