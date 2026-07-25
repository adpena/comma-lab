#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the strict Task #701 cross-chain waterfill admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.ddm_ms2r_r3_366box_typed_fisher_g4_waterfill import (  # noqa: E402
    LANE_ID,
    REQUIRED_INPUTS,
    VERDICT,
    build_artifacts,
    canonical_bytes,
)

CONFIG_SCHEMA: Final = "DDMMS2RR3366BoxTypedFisherG4WaterfillConfigV1"
RUN_ID: Final = (
    "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_20260725T162107Z"
)
DONE_PATH: Final = (
    REPO
    / ".omx/research/"
    "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_20260725.done"
)


class RunnerError(ValueError):
    """A typed config, bound artifact, or immutable output differs."""


class ArtifactBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    bytes: int
    schema_name: str = Field(alias="schema", serialization_alias="schema")
    role: str

    @field_validator("sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        bytes.fromhex(value)
        return value

    @field_validator("bytes")
    @classmethod
    def _bytes(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("bytes must be a positive integer")
        return value


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_name: Literal[
        "DDMMS2RR3366BoxTypedFisherG4WaterfillConfigV1"
    ] = Field(alias="schema")
    run_id: str
    lane_id: str
    output_root: str
    receipt_timestamp_utc: str
    authority: ArtifactBinding
    inputs: dict[str, ArtifactBinding]
    pair_count: Literal[600]
    scored_pixels: Literal[117964800]
    allowed_errors: Literal[136839]
    seed: Literal[1234]
    research_only: Literal[True]
    local_read_only: Literal[True]
    external_execution_allowed: Literal[False]
    score_claim: Literal[False]
    promotion_eligible: Literal[False]
    pointer_moved: Literal[False]
    main_landing_review_required: Literal[True]

    @model_validator(mode="after")
    def _boundary(self) -> RunConfig:
        if self.run_id != RUN_ID or self.lane_id != LANE_ID:
            raise ValueError("run/lane identity differs")
        if set(self.inputs) != REQUIRED_INPUTS:
            raise ValueError("input artifact inventory differs")
        if not self.output_root:
            raise ValueError("output root must be nonempty")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPO / candidate


def _read_bound(binding: ArtifactBinding) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _resolve(binding.path)
    if not path.is_file():
        raise RunnerError(f"bound input is missing: {path}")
    observed_bytes = path.stat().st_size
    observed_sha = _sha256(path)
    if observed_bytes != binding.bytes or observed_sha != binding.sha256:
        raise RunnerError(
            f"bound input drifted: {binding.path}; "
            f"bytes={observed_bytes}, sha256={observed_sha}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"bound input is not canonical JSON: {binding.path}") from exc
    if not isinstance(value, Mapping) or value.get("schema") != binding.schema_name:
        raise RunnerError(f"bound input schema differs: {binding.path}")
    return dict(value), {
        "path": binding.path,
        "bytes": binding.bytes,
        "sha256": binding.sha256,
        "schema": binding.schema_name,
        "role": binding.role,
    }


def _publish_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RunnerError(f"immutable output differs on resume: {path}")
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return {
        "path": str(path.relative_to(REPO)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve(strict=True)
    payload = config_path.read_bytes()
    try:
        config = RunConfig.model_validate_json(payload)
    except ValueError as exc:
        raise RunnerError(f"typed config is invalid: {exc}") from exc
    config_sha = hashlib.sha256(payload).hexdigest()

    authority_path = _resolve(config.authority.path)
    if (
        not authority_path.is_file()
        or authority_path.stat().st_size != config.authority.bytes
        or _sha256(authority_path) != config.authority.sha256
    ):
        raise RunnerError("delegated authority drifted")

    inputs: dict[str, dict[str, Any]] = {}
    custody: dict[str, dict[str, Any]] = {}
    for name in sorted(config.inputs):
        inputs[name], custody[name] = _read_bound(config.inputs[name])

    config_custody = {
        "path": str(config_path.relative_to(REPO)),
        "bytes": len(payload),
        "sha256": config_sha,
        "schema": CONFIG_SCHEMA,
    }
    artifacts = build_artifacts(
        inputs,
        input_custody=custody,
        config_custody=config_custody,
        available_memory_bytes=int(psutil.virtual_memory().available),
    )
    output_root = _resolve(config.output_root)
    stage_root = output_root / "stage_checkpoints"
    preflight_ref = _publish_immutable(
        stage_root / "01_cross_chain_preflight.json",
        artifacts.preflight,
    )
    table_ref = _publish_immutable(
        output_root / "priced_rung_table.json",
        artifacts.priced_rung_table,
    )
    backfill_ref = _publish_immutable(
        output_root / "rd1_162_dual_backfill.json",
        artifacts.rd1_backfill,
    )
    receipt = {
        **artifacts.receipt,
        "run_id": config.run_id,
        "finished_at_utc": config.receipt_timestamp_utc,
        "delegated_authority": config.authority.model_dump(by_alias=True),
        "typed_config": config_custody,
        "resumability": {
            "mode": "atomic immutable stage outputs",
            "stage_count": 3,
            "all_stages_preserved": True,
            "resume_behavior": "rehash inputs and require byte-identical outputs",
        },
        "output_custody": {
            "cross_chain_preflight": preflight_ref,
            "priced_rung_table": table_ref,
            "rd1_162_dual_backfill": backfill_ref,
        },
    }
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    receipt_ref = _publish_immutable(output_root / "receipt.json", receipt)
    done = {
        "schema": "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_done.v1",
        "run_id": config.run_id,
        "verdict": VERDICT,
        "receipt": receipt_ref,
        "score_claim": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    _publish_immutable(DONE_PATH, done)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.config)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "measured_task_rungs": receipt["priced_rung_table"][
                    "measured_task_rung_count"
                ],
                "finite_rd1_duals": receipt["rd1_backfill"][
                    "lambda_measured_cell_count"
                ],
                "r6_candidate_ready": receipt["r6_candidate_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
