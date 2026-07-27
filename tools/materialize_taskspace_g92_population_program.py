#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile the G92 partial population atlas and seal its lowering blocker."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.witness_dsl.taskspace_g92_population_global_program_induction_v1 import (
    G51_PAYLOAD_POLICY,
    G92_FAMILY_SCHEMA,
    G92_SELECTION_CONTRACT,
    SEQUENTIAL_LOWERING_BLOCKER,
    ExactFileIdentityV1,
    PopulationProgramInductionError,
    canonical_json_bytes,
    compile_population_program_plan,
    load_sealed_g90_population,
    sha256_bytes,
    sha256_file,
)

CONFIG_SCHEMA: Final = "tac.taskspace_g92_population_global_program_config.v1"
PREFLIGHT_SCHEMA: Final = "tac.taskspace_g92_population_global_program_preflight.v1"
BLOCKER_SCHEMA: Final = "tac.taskspace_g92_population_global_program_blocker.v1"


class G92MaterializerError(RuntimeError):
    """Configuration, storage, sealed input, or authority invariant failed."""


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G92MaterializerError(f"{label} cannot be read") from exc
    if type(value) is not dict:
        raise G92MaterializerError(f"{label} is not one JSON object")
    return value


def _seal(body: dict[str, Any], *, field: str) -> dict[str, Any]:
    if field in body:
        raise G92MaterializerError(f"{field} already exists")
    return {**body, field: sha256_bytes(canonical_json_bytes(body))}


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise G92MaterializerError(f"immutable checkpoint differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise G92MaterializerError(f"immutable checkpoint differs: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value))


@dataclass(frozen=True, slots=True)
class G92ConfigV1:
    path: Path
    output_root: Path
    safety_reserve_bytes: int
    semantic_archive: ExactFileIdentityV1
    current_base_archive: ExactFileIdentityV1
    g90_aggregate: ExactFileIdentityV1
    g90_aggregate_self_sha256: str
    g51_provenance_receipt: ExactFileIdentityV1


def load_config(path: Path) -> G92ConfigV1:
    resolved = Path(path).expanduser().resolve()
    value = _load_mapping(resolved, label="G92 config")
    expected = {
        "current_base_archive",
        "g51_teacher_receipt",
        "g90_aggregate",
        "g90_aggregate_self_sha256",
        "output_root",
        "safety_reserve_bytes",
        "schema",
        "semantic_archive",
    }
    if set(value) != expected or value.get("schema") != CONFIG_SCHEMA:
        raise G92MaterializerError("G92 config schema/key set differs")
    if type(value["output_root"]) is not str or not value["output_root"]:
        raise G92MaterializerError("G92 output_root is not a path string")
    output = Path(value["output_root"]).expanduser().resolve()
    if not any(
        str(output).startswith(prefix)
        for prefix in (
            "/Volumes/VertigoDataTier/pact/",
            "/Volumes/APDataStore/pact/",
        )
    ):
        raise G92MaterializerError("production G92 output must use the SSD waterfall")
    reserve = value["safety_reserve_bytes"]
    if type(reserve) is not int or reserve <= 0:
        raise G92MaterializerError("G92 safety reserve must be one positive integer")
    expected_self = value["g90_aggregate_self_sha256"]
    if (
        type(expected_self) is not str
        or len(expected_self) != 64
        or any(character not in "0123456789abcdef" for character in expected_self)
    ):
        raise G92MaterializerError("G90 aggregate self SHA is not canonical")
    return G92ConfigV1(
        path=resolved,
        output_root=output,
        safety_reserve_bytes=reserve,
        semantic_archive=ExactFileIdentityV1.from_mapping(
            value["semantic_archive"],
            label="semantic archive",
        ),
        current_base_archive=ExactFileIdentityV1.from_mapping(
            value["current_base_archive"],
            label="current base archive",
        ),
        g90_aggregate=ExactFileIdentityV1.from_mapping(
            value["g90_aggregate"],
            label="G90 aggregate",
        ),
        g90_aggregate_self_sha256=expected_self,
        # The config key is historical V1 ABI.  File identity alone confers no
        # teacher/full-residual/payload policy authority.
        g51_provenance_receipt=ExactFileIdentityV1.from_mapping(
            value["g51_teacher_receipt"],
            label="G51 opaque provenance receipt",
        ),
    )


def _compile_plan(config: G92ConfigV1) -> tuple[dict[str, Any], dict[str, Any]]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(config.output_root)
    if usage.free <= config.safety_reserve_bytes:
        raise G92MaterializerError("SSD free bytes do not clear the configured safety reserve")
    config.semantic_archive.verify(label="semantic archive")
    config.current_base_archive.verify(label="current base archive")
    config.g51_provenance_receipt.verify(label="G51 opaque provenance receipt")
    try:
        g90 = load_sealed_g90_population(
            config.g90_aggregate,
            expected_aggregate_self_sha256=config.g90_aggregate_self_sha256,
        )
        plan = compile_population_program_plan(
            g90=g90,
            g51_receipt_identity=config.g51_provenance_receipt,
        )
    except PopulationProgramInductionError as exc:
        raise G92MaterializerError(str(exc)) from exc
    if (
        plan.current_base_archive_bytes != config.current_base_archive.bytes
        or plan.current_base_archive_sha256 != config.current_base_archive.sha256
    ):
        raise G92MaterializerError("G90 score point differs from exact current-base archive custody")
    preflight_body = {
        "schema": PREFLIGHT_SCHEMA,
        "config": {
            "path": str(config.path),
            "bytes": config.path.stat().st_size,
            "sha256": sha256_file(config.path),
        },
        "output_root": str(config.output_root),
        "inputs": {
            "semantic_archive": config.semantic_archive.to_dict(),
            "current_base_archive": config.current_base_archive.to_dict(),
            "g90_aggregate": config.g90_aggregate.to_dict(),
            "g90_aggregate_self_sha256": config.g90_aggregate_self_sha256,
            "g51_opaque_provenance_receipt": config.g51_provenance_receipt.to_dict(),
        },
        "storage": {
            "tier": "SSD",
            "free_bytes_observed": usage.free,
            "safety_reserve_bytes": config.safety_reserve_bytes,
            "scratch_bytes_retained": 0,
            "cleanup_policy": "NO_BULK_OUTPUT_PROGRAM_PLAN_AND_BLOCKER_ONLY",
        },
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    plan_body = {
        "schema": G92_FAMILY_SCHEMA,
        "g90_aggregate_sha256": plan.g90_aggregate_sha256,
        "g90_aggregate_self_sha256": plan.g90_aggregate_self_sha256,
        "g51_opaque_provenance_receipt_sha256": plan.g51_receipt_sha256,
        "current_base_archive": {
            "bytes": plan.current_base_archive_bytes,
            "sha256": plan.current_base_archive_sha256,
            "incumbent_frame_selector": "BOTH",
        },
        "new_intervention_frame_selector": "Y1",
        "shared_physical_families": [
            {
                "family_id": row.family_id,
                "role": row.role,
                "direction_rank": row.direction_rank,
                "amplitude_scale": row.amplitude_scale,
                "intervention_ids": list(row.intervention_ids),
            }
            for row in plan.shared_families
        ],
        "collision_free_partial_atlas_branches": [list(branch) for branch in plan.branches],
        "partial_enumerated_branch_state_count": plan.partial_enumerated_branch_state_count,
        "branch_order_semantics": ("CANONICAL_STORAGE_ORDER_ONLY_NO_PREFIX_OPTIMALITY_OR_COMPREHENSIVENESS"),
        "screening_only_projection_ids": list(plan.screening_only_projection_ids),
        "screening_completeness_claim": False,
        "exact_replay_atlas_complete": False,
        "selection_contract": G92_SELECTION_CONTRACT,
        "g51_payload_policy": G51_PAYLOAD_POLICY,
        "lowering_blocker": plan.lowering_blocker,
        "archive_pricing_allowed": False,
        "g83_ready": False,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    return (
        _seal(preflight_body, field="preflight_receipt_sha256"),
        _seal(plan_body, field="program_plan_sha256"),
    )


def _write_blocker(config: G92ConfigV1, *, plan_path: Path | None, exc: BaseException) -> Path:
    body = {
        "schema": BLOCKER_SCHEMA,
        "config_path": str(config.path),
        "output_root": str(config.output_root),
        "program_plan": (
            None
            if plan_path is None
            else {
                "path": str(plan_path),
                "bytes": plan_path.stat().st_size,
                "sha256": sha256_file(plan_path),
            }
        ),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "required_next_implementation": SEQUENTIAL_LOWERING_BLOCKER,
        "forbidden_false_action": ("DO_NOT_TREAT_V1_PARETO_ROWS_OR_BRANCH_ORDER_AS_COMPLETE_OR_OPTIMAL"),
        "g51_payload_policy": G51_PAYLOAD_POLICY,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "archive_emitted": False,
        "archive_priced": False,
        "research_only": True,
        "encoder_only": True,
    }
    receipt = _seal(body, field="blocker_receipt_sha256")
    path = config.output_root / "blocker_receipt.json"
    payload = canonical_json_bytes(receipt)
    if path.exists() and path.read_bytes() != payload:
        path = config.output_root / f"blocker_receipt_{receipt['blocker_receipt_sha256'][:12]}.json"
    _atomic_write_json(path, receipt)
    return path


def compile_and_block(config: G92ConfigV1) -> dict[str, Any]:
    """Persist the partial atlas, then fail closed at missing same-state rows."""

    preflight, plan = _compile_plan(config)
    preflight_path = config.output_root / "00_preflight_receipt.json"
    plan_path = config.output_root / "20_population_program_plan.json"
    _atomic_write_json(preflight_path, preflight)
    _atomic_write_json(plan_path, plan)
    blocker = _write_blocker(
        config,
        plan_path=plan_path,
        exc=G92MaterializerError(SEQUENTIAL_LOWERING_BLOCKER),
    )
    return {
        "status": "blocked",
        "program_plan": str(plan_path),
        "blocker_receipt": str(blocker),
        "required_next_implementation": SEQUENTIAL_LOWERING_BLOCKER,
        "archive_emitted": False,
        "archive_priced": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--compile-plan", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    try:
        result = compile_and_block(config)
    except Exception as exc:
        blocker = _write_blocker(config, plan_path=None, exc=exc)
        result = {
            "status": "blocked",
            "blocker_receipt": str(blocker),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "archive_emitted": False,
            "archive_priced": False,
        }
    print(json.dumps(result, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
