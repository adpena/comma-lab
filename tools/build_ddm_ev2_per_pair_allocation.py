#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the fail-closed EV2 C1 pair/cell allocation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for local_path in (str(SRC), str(REPO)):
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

from tac.optimization.ddm_ev2_per_pair_allocation import (  # noqa: E402
    EV2AllocationError,
    build_ev2_allocation,
    canonical_bytes,
    canonical_sha256,
)
from tac.optimization.ddm_lambda_continuation_frontier import (  # noqa: E402
    publish_immutable_json,
)

CONFIG_SCHEMA: Final = "ddm_ev2_per_pair_allocation_config.v1"
EXPECTED_SOURCES: Final = {
    "c1_archive",
    "c1_ledger",
    "lp1",
    "ev1",
    "rd1",
    "r3",
    "ms5",
    "bundle_complete",
}
RUN_ID: Final = "ddm_ev2_per_pair_allocation_20260725T041933Z"
LANE_ID: Final = "ddm_ev2_per_pair_allocation_producer"
CHECKPOINT_KEY: Final = (
    "codex_delegate:ddm_ev2_per_pair_allocation_producer:20260725T041933Z"
)
AUTHORITY_SHA256: Final = (
    "1385a105e0ec838b8fef3a118ad9fa7b73ecc9777d66cb91284dd99e42c071d6"
)


def _read_bound_artifact(
    reference: Mapping[str, Any],
    *,
    label: str,
) -> tuple[bytes, Path, dict[str, Any]]:
    relative = reference.get("path")
    expected_sha256 = reference.get("sha256")
    expected_bytes = reference.get("bytes")
    if (
        not isinstance(relative, str)
        or not isinstance(expected_sha256, str)
        or type(expected_bytes) is not int
        or expected_bytes < 0
    ):
        raise EV2AllocationError(f"{label} custody reference is incomplete")
    path = (REPO / relative).resolve(strict=True)
    if not path.is_relative_to(REPO.resolve()):
        raise EV2AllocationError(f"{label} escapes repository root")
    payload = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_bytes or observed_sha256 != expected_sha256:
        raise EV2AllocationError(f"{label} custody differs")
    return payload, path, {
        "path": path.relative_to(REPO).as_posix(),
        "bytes": len(payload),
        "sha256": observed_sha256,
        "role": reference.get("role"),
    }


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EV2AllocationError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise EV2AllocationError(f"{label} must be a JSON object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise EV2AllocationError("EV2 config schema differs")
    for field in (
        "research_only",
        "execution_allowed",
        "score_claim",
        "promotion_eligible",
        "pointer_moved",
        "main_landing_review_required",
    ):
        expected = field in {"research_only", "main_landing_review_required"}
        if config.get(field) is not expected:
            raise EV2AllocationError(f"config.{field} must be {expected}")
    if config.get("pointer") != "0.1910828242 [contest-CPU]":
        raise EV2AllocationError("EV2 config pointer differs")
    if config.get("run_id") != RUN_ID or config.get("lane_id") != LANE_ID:
        raise EV2AllocationError("EV2 config run/lane identity differs")
    delegation = config.get("delegation")
    if (
        not isinstance(delegation, Mapping)
        or delegation.get("checkpoint_key") != CHECKPOINT_KEY
        or delegation.get("authority_sha256") != AUTHORITY_SHA256
        or delegation.get("authority_bytes") != 7_503
    ):
        raise EV2AllocationError("EV2 delegation custody differs")
    sources = config.get("source_artifacts")
    if not isinstance(sources, Mapping) or set(sources) != EXPECTED_SOURCES:
        raise EV2AllocationError("EV2 config source inventory differs")


def materialize(config_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config_payload = config_path.resolve(strict=True).read_bytes()
    config = _json_object(config_payload, label="config")
    _validate_config(config)
    expected_output = config.get("output_dir")
    if (
        not isinstance(expected_output, str)
        or output_dir != (REPO / expected_output).resolve()
    ):
        raise EV2AllocationError("output directory differs from typed config")

    source_payloads: dict[str, bytes] = {}
    source_paths: dict[str, Path] = {}
    source_custody: dict[str, dict[str, Any]] = {}
    for label in sorted(EXPECTED_SOURCES):
        payload, path, custody = _read_bound_artifact(
            config["source_artifacts"][label],
            label=label,
        )
        source_payloads[label] = payload
        source_paths[label] = path
        source_custody[label] = custody

    c1_ledger = _json_object(source_payloads["c1_ledger"], label="C1 ledger")
    if (
        c1_ledger.get("schema") != "ddm_c1_composed_candidate_spec.v1"
        or c1_ledger.get("score_claim") is not False
        or c1_ledger.get("promotion_eligible") is not False
        or c1_ledger.get("pointer_moved") is not False
    ):
        raise EV2AllocationError("C1 ledger authority differs")

    result = build_ev2_allocation(
        c1_archive=source_payloads["c1_archive"],
        lp1=_json_object(source_payloads["lp1"], label="LP1"),
        ev1=_json_object(source_payloads["ev1"], label="EV1"),
        rd1=_json_object(source_payloads["rd1"], label="RD1"),
        r3=_json_object(source_payloads["r3"], label="R3"),
        ms5=_json_object(source_payloads["ms5"], label="MS5"),
        ms5_sha256=source_custody["ms5"]["sha256"],
        bundle_path=source_paths["bundle_complete"],
        repository_root=REPO,
    )
    output_values = {
        "allocation_table.json": result.allocation_table,
        "ms5_loader_table.json": result.ms5_loader_table,
        "rd1_dual_backfill.json": result.rd1_backfill,
        "headline_replay.json": result.headline_replay,
    }
    output_custody = {
        name: {
            "path": (output_dir / name).relative_to(REPO).as_posix(),
            "bytes": len(canonical_bytes(value)),
            "sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(),
        }
        for name, value in output_values.items()
    }
    receipt = {
        **result.receipt,
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "generated_at_utc": config["generated_at_utc"],
        "typed_config": {
            "path": config_path.resolve().relative_to(REPO).as_posix(),
            "bytes": len(config_payload),
            "sha256": hashlib.sha256(config_payload).hexdigest(),
            "schema": CONFIG_SCHEMA,
        },
        "delegation": dict(config["delegation"]),
        "source_custody": source_custody,
        "output_custody": output_custody,
    }
    receipt.pop("receipt_content_sha256", None)
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)

    for name, value in output_values.items():
        publish_immutable_json(output_dir / name, value)
    publish_immutable_json(output_dir / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receipt = materialize(args.config, args.output_dir)
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "verdict": receipt["verdict"],
                "mass_conservation": receipt["mass_conservation"],
                "rd1_dual_backfill": receipt["rd1_dual_backfill"],
                "remaining_blockers": receipt["headline_replay"][
                    "remaining_blockers"
                ],
                "score_claim": receipt["score_claim"],
                "pointer_moved": receipt["pointer_moved"],
                "main_landing_review_required": receipt[
                    "main_landing_review_required"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
