#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-adjudicate a sealed K=2 replay with the corrected fallback charge.

This tool is deliberately teacher-free.  It accepts only the immutable replay
receipt tree, verifies every nested byte identity, and writes a deterministic
adjacent wrapper.  It never rewrites a source row, manifest, contract, receipt,
or completion seal.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO / "experiments/results/p0_costate_reuse_k2_n600_v3_20260713"
OUTPUT_NAME = "corrected_adjudication_receipt.json"

WRAPPER_SCHEMA = "p0_costate_reuse_k2_corrected_adjudication.v1"
CORRECTED_GATE_SCHEMA = "p0_costate_reuse_k2_corrected_admission.v1"
SOURCE_SCHEMA = "p0_costate_reuse_k2_n600.v2"
PAIR_SCHEMA = "p0_costate_reuse_k2_pair.v2"
STAGE_SCHEMA = "p0_costate_reuse_k2_stage.v2"
COMPLETE_SCHEMA = "p0_costate_reuse_k2_complete.v2"
EXPECTED_STATE_COUNT = 600
EXPECTED_STAGE_COUNT = 3
EXPECTED_K_MAX = 2
EXPECTED_FORWARD_SHARE = 0.1784755863
EXPECTED_FALLBACK_SEMANTICS = "byte-exact rollback plus full exact-teacher refresh"
EXPECTED_LANE_ID = "lane_p0_backward_closer_20260713"
EXPECTED_AXIS = "[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]"
EXPECTED_RUN_CONTRACT_SHA256 = "e9c4a6629bcbc91876d2476b0bef051dfe56fe27d93076fa79f7225a5b62d56f"
EXPECTED_OBJECTIVE_SHA256 = "af5ae342f3987b82c2d3ee5bdb12dcfca1ecab07631fd545a9e723c15cb7c9e7"
EXPECTED_SCORER_SHA256 = "584f711dfb85163c38caf8976ebeda87698baefb45f9f5979539a8c176b6b73e"
# Exact post-seal byte roots reviewed after the immutable replay completed.
# Public adjudication still refuses if either root is ever cleared.
EXPECTED_SOURCE_RECEIPT_SHA256: str | None = "4c84c1f80ae7fc1b4ee76d28395405834e3eecd439155e4ebd79d4e81530506c"
EXPECTED_SOURCE_COMPLETE_SHA256: str | None = "45ccbccee780d26bf350442ddf5551d62d483957c591b706fe5eb746dfbea34c"
SUPERSEDED_LABEL = "SUPERSEDED_INVALID_FALLBACK_CHARGE"
# This label is part of the immutable v2 source receipt.  It is accepted only
# while revalidating those historical bytes; it is not current timing routing.
HISTORICAL_SOURCE_WHOLE_EPOCH_LABEL = "UNKNOWN_IN_LOOP_TIMER_OWED"
DIAGNOSTIC_ONLY_NO_TIMING_AUTHORITY = "NOT_MEASURED_DIAGNOSTIC_ONLY"
FIDELITY_BLOCKED_STATUS = "FIDELITY_BLOCKED_PENDING_NEW_FORMULATION"
FIDELITY_ADMITTED_STATUS = "FIDELITY_ADMITTED_PENDING_PROVIDER_RESUME_PARITY_NO_TIMER_GO"


class AdjudicationError(RuntimeError):
    """The sealed-source or corrected-adjudication contract failed closed."""


def corrected_timing_routing(corrected_admission_verdict: str) -> dict[str, Any]:
    """Report the corrected fidelity stage without minting timing authority.

    A separate downstream validator must bind a fresh preregistered formulation,
    its admitted sealed n600 receipt, exact provider-gradient parity, canonical
    resume registration, and uninterrupted-versus-resumed state parity before
    any operator timing request can become eligible.
    """

    if corrected_admission_verdict == "ADMIT_K2_GUARDED_REUSE":
        return {
            "status": FIDELITY_ADMITTED_STATUS,
            "operator_go_request_eligible": False,
            "operator_go_granted": False,
        }
    if corrected_admission_verdict == "NOT_ADMITTED":
        return {
            "status": FIDELITY_BLOCKED_STATUS,
            "operator_go_request_eligible": False,
            "operator_go_granted": False,
        }
    raise AdjudicationError(f"unknown corrected admission verdict: {corrected_admission_verdict}")


@dataclass(frozen=True)
class _ExpectedSourceRoots:
    """Code-reviewed trust roots; private injection exists only for unit fixtures."""

    run_contract_sha256: str
    objective_sha256: str
    scorer_sha256: str
    measurement_receipt_sha256: str
    complete_sha256: str


def canonical_sha256(value: Any) -> str:
    """Hash the repository's canonical compact JSON representation."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise AdjudicationError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_expected_roots(roots: _ExpectedSourceRoots) -> _ExpectedSourceRoots:
    for field, value in asdict(roots).items():
        _require_sha256(value, f"expected source root {field}")
    return roots


def _public_expected_roots() -> _ExpectedSourceRoots:
    if EXPECTED_SOURCE_RECEIPT_SHA256 is None or EXPECTED_SOURCE_COMPLETE_SHA256 is None:
        raise AdjudicationError(
            "public adjudication is blocked until reviewed receipt and completion roots are patched"
        )
    return _validate_expected_roots(
        _ExpectedSourceRoots(
            run_contract_sha256=EXPECTED_RUN_CONTRACT_SHA256,
            objective_sha256=EXPECTED_OBJECTIVE_SHA256,
            scorer_sha256=EXPECTED_SCORER_SHA256,
            measurement_receipt_sha256=EXPECTED_SOURCE_RECEIPT_SHA256,
            complete_sha256=EXPECTED_SOURCE_COMPLETE_SHA256,
        )
    )


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdjudicationError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AdjudicationError(f"{label} must be an array")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdjudicationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AdjudicationError(f"{label} must be >= {minimum}")
    return value


def _require_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdjudicationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AdjudicationError(f"{label} must be finite")
    return result


def _read_json(
    source_dir: Path,
    path: Path,
    label: str,
    snapshot: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AdjudicationError(f"{label} is missing or is not a regular file")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(source_dir)
    except ValueError as exc:
        raise AdjudicationError(f"{label} escapes the sealed source directory") from exc
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"{label} is not valid JSON") from exc
    payload = _require_dict(value, label)
    custody = {
        "path": relative.as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if custody["path"] in snapshot:
        raise AdjudicationError(f"duplicate sealed source path {custody['path']}")
    snapshot[custody["path"]] = custody
    return payload, custody


def _validate_run_contract(
    source_dir: Path,
    contract: dict[str, Any],
    *,
    expected_state_count: int,
    expected_stage_count: int,
    expected_roots: _ExpectedSourceRoots,
) -> dict[str, Any]:
    payload = _require_dict(contract.get("payload"), "run contract payload")
    semantic_payload = {key: value for key, value in payload.items() if key != "git_head_at_launch"}
    contract_sha256 = _require_sha256(contract.get("sha256"), "run contract semantic hash")
    if contract_sha256 != canonical_sha256(semantic_payload):
        raise AdjudicationError("run contract semantic self-hash changed")
    if contract_sha256 != expected_roots.run_contract_sha256:
        raise AdjudicationError("run contract semantic hash is not the code-reviewed v3 root")
    if _require_sha256(
        contract.get("launch_provenance_sha256"), "run contract launch provenance hash"
    ) != canonical_sha256(payload):
        raise AdjudicationError("run contract launch provenance self-hash changed")
    if payload.get("schema") != SOURCE_SCHEMA:
        raise AdjudicationError("run contract source schema changed")
    if payload.get("output_dir") != str(source_dir):
        raise AdjudicationError("run contract output directory changed")
    if payload.get("max_pairs") is not None:
        raise AdjudicationError("bounded-prefix replay cannot be corrected-adjudicated")
    for field in ("score_claim", "promotion_eligible", "pointer_moved"):
        if payload.get(field) is not False:
            raise AdjudicationError(f"run contract carries false authority at {field}")

    constants = _require_dict(payload.get("constants"), "run contract constants")
    if _require_int(constants.get("n_pairs"), "run contract n_pairs", minimum=1) != expected_state_count:
        raise AdjudicationError("run contract state count changed")
    if (
        _require_int(constants.get("checkpoint_count"), "run contract checkpoint_count", minimum=1)
        != expected_stage_count
    ):
        raise AdjudicationError("run contract checkpoint count changed")
    if _require_int(constants.get("K_max"), "run contract K_max", minimum=1) != EXPECTED_K_MAX:
        raise AdjudicationError("run contract is not the K=2 formulation")
    alpha = _require_finite_float(constants.get("diagnostic_forward_share"), "diagnostic forward share")
    if alpha != EXPECTED_FORWARD_SHARE:
        raise AdjudicationError("diagnostic forward-share provenance changed")
    holdout_period = _require_int(constants.get("holdout_period"), "holdout period", minimum=2)
    seed = _require_int(constants.get("seed"), "replay seed")

    objective_spec = _require_dict(payload.get("objective_spec"), "objective spec")
    objective_sha256 = _require_sha256(payload.get("objective_sha256"), "objective hash")
    if objective_sha256 != canonical_sha256(objective_spec):
        raise AdjudicationError("objective self-hash changed")
    if objective_sha256 != expected_roots.objective_sha256:
        raise AdjudicationError("objective hash is not the code-reviewed v3 root")
    if objective_spec.get("fallback") != EXPECTED_FALLBACK_SEMANTICS:
        raise AdjudicationError("fallback semantics do not require rollback plus full exact refresh")

    admission_spec = _require_dict(payload.get("admission_spec"), "original admission spec")
    admission_spec_sha256 = _require_sha256(payload.get("admission_spec_sha256"), "original admission-spec hash")
    if admission_spec_sha256 != canonical_sha256(admission_spec):
        raise AdjudicationError("original admission-spec self-hash changed")

    input_custody = _require_dict(payload.get("input_custody"), "input custody")
    scorer_paths = (
        "upstream/models/segnet.safetensors",
        "upstream/models/posenet.safetensors",
    )
    scorer_custody: dict[str, Any] = {}
    for scorer_path in scorer_paths:
        entry = _require_dict(input_custody.get(scorer_path), f"input custody {scorer_path}")
        if entry.get("path") != scorer_path:
            raise AdjudicationError(f"scorer custody path changed for {scorer_path}")
        _require_int(entry.get("bytes"), f"scorer custody bytes for {scorer_path}", minimum=1)
        _require_sha256(entry.get("sha256"), f"scorer custody hash for {scorer_path}")
        scorer_custody[scorer_path] = entry
    scorer_sha256 = _require_sha256(payload.get("scorer_sha256"), "scorer hash")
    if scorer_sha256 != canonical_sha256(scorer_custody):
        raise AdjudicationError("scorer custody hash changed")
    if scorer_sha256 != expected_roots.scorer_sha256:
        raise AdjudicationError("scorer hash is not the code-reviewed v3 root")

    return {
        "payload": payload,
        "sha256": contract_sha256,
        "objective_sha256": objective_sha256,
        "scorer_sha256": scorer_sha256,
        "alpha": alpha,
        "holdout_period": holdout_period,
        "seed": seed,
    }


def _guard_passes(row: dict[str, Any]) -> bool:
    guard = row.get("reuse_guard")
    if not isinstance(guard, dict) or not guard:
        return False
    required = ("ce_strict_descent", "d_seg_nonworsening", "d_pose_nonworsening")
    return all(guard.get(key) is True for key in required)


def _metric_triplet(value: Any, label: str) -> dict[str, float]:
    metrics = _require_dict(value, label)
    return {key: _require_finite_float(metrics.get(key), f"{label} {key}") for key in ("ce", "d_seg", "d_pose")}


def _validate_behavior_semantics(row: dict[str, Any], pair_index: int) -> None:
    """Re-derive every stored behavioral predicate when candidate metrics exist."""

    eligible = row["eligible_for_k2"]
    accepted_flag = row["reuse_guard_accept"]
    status = row["status"]
    has_current = "current_metrics" in row
    has_exact_candidate = "exact_second_metrics" in row
    has_stale_candidate = "stale_second_metrics" in row
    has_complete_candidate_metrics = has_current and has_exact_candidate and has_stale_candidate
    if accepted_flag and not eligible:
        raise AdjudicationError(f"pair {pair_index} accepts reuse while ineligible")
    if accepted_flag and not has_complete_candidate_metrics:
        raise AdjudicationError(f"pair {pair_index} accepted without behavioral metric custody")
    if not has_complete_candidate_metrics:
        if has_exact_candidate or has_stale_candidate:
            raise AdjudicationError(f"pair {pair_index} has partial candidate metric custody")
        if status == "TERMINAL_OR_BLOCKED_AT_ANCHOR":
            if eligible or has_current or row.get("reuse_guard") is not None:
                raise AdjudicationError(f"pair {pair_index} anchor-terminal semantics changed")
            return
        if status == "TERMINAL_OR_BLOCKED_AT_REUSE_POINT":
            if eligible or not has_current or row.get("reuse_guard") is not None:
                raise AdjudicationError(f"pair {pair_index} reuse-terminal semantics changed")
            _metric_triplet(row.get("current_metrics"), f"pair {pair_index} current metrics")
            return
        if status == "STALE_ZERO_OR_BIT_IDENTICAL":
            expected_false_guard = {
                "ce_strict_descent": False,
                "d_seg_nonworsening": False,
                "d_pose_nonworsening": False,
            }
            if not eligible or not has_current or row.get("reuse_guard") != expected_false_guard:
                raise AdjudicationError(f"pair {pair_index} stale-zero semantics changed")
            _metric_triplet(row.get("current_metrics"), f"pair {pair_index} current metrics")
            return
        if accepted_flag or status in {"REUSE_GUARD_ACCEPT", "REUSE_GUARD_FALLBACK"}:
            raise AdjudicationError(f"pair {pair_index} stored guard contradicts missing metrics")
        raise AdjudicationError(f"pair {pair_index} has an unknown non-candidate status")

    current = _metric_triplet(row.get("current_metrics"), f"pair {pair_index} current metrics")
    exact = _metric_triplet(row.get("exact_second_metrics"), f"pair {pair_index} exact second metrics")
    stale = _metric_triplet(row.get("stale_second_metrics"), f"pair {pair_index} stale second metrics")
    derived_guard = {
        "ce_strict_descent": stale["ce"] < current["ce"],
        "d_seg_nonworsening": stale["d_seg"] <= current["d_seg"],
        "d_pose_nonworsening": stale["d_pose"] <= current["d_pose"],
    }
    if row.get("reuse_guard") != derived_guard:
        raise AdjudicationError(f"pair {pair_index} guard predicates disagree with sealed metrics")
    derived_accept = all(derived_guard.values())
    if accepted_flag is not derived_accept:
        raise AdjudicationError(f"pair {pair_index} accept flag disagrees with sealed metrics")
    expected_status = "REUSE_GUARD_ACCEPT" if derived_accept else "REUSE_GUARD_FALLBACK"
    if status != expected_status:
        raise AdjudicationError(f"pair {pair_index} status disagrees with sealed metrics")
    regret = _require_dict(row.get("stale_minus_exact_regret"), f"pair {pair_index} stale-minus-exact regret")
    expected_regret = {key: float(stale[key] - exact[key]) for key in ("ce", "d_seg", "d_pose")}
    if regret != expected_regret:
        raise AdjudicationError(f"pair {pair_index} regret disagrees with sealed metrics")


def _expected_assignment(
    pair_index: int,
    *,
    checkpoint_index: int,
    checkpoint_name: str,
    holdout_period: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "pair_index": pair_index,
        "checkpoint_index": checkpoint_index,
        "checkpoint_name": checkpoint_name,
        "split": "heldout" if pair_index % holdout_period == seed % holdout_period else "train",
    }


def _validate_pair_row(
    row: dict[str, Any],
    *,
    expected_assignment: dict[str, Any],
    run_contract_sha256: str,
) -> str:
    pair_index = expected_assignment["pair_index"]
    if row.get("schema") != PAIR_SCHEMA:
        raise AdjudicationError(f"pair {pair_index} schema changed")
    if row.get("run_contract_sha256") != run_contract_sha256:
        raise AdjudicationError(f"pair {pair_index} run-contract hash changed")
    if row.get("assignment") != expected_assignment:
        raise AdjudicationError(f"pair {pair_index} deterministic assignment changed")
    record_content_sha256 = _require_sha256(row.get("record_content_sha256"), f"pair {pair_index} content hash")
    unsigned = {key: value for key, value in row.items() if key != "record_content_sha256"}
    if record_content_sha256 != canonical_sha256(unsigned):
        raise AdjudicationError(f"pair {pair_index} content self-hash changed")
    if not isinstance(row.get("eligible_for_k2"), bool):
        raise AdjudicationError(f"pair {pair_index} eligible flag is not boolean")
    if not isinstance(row.get("reuse_guard_accept"), bool):
        raise AdjudicationError(f"pair {pair_index} accept flag is not boolean")
    if not isinstance(row.get("status"), str):
        raise AdjudicationError(f"pair {pair_index} status is missing")
    _validate_behavior_semantics(row, pair_index)
    return record_content_sha256


def _validate_exact_source_inventory(
    source_dir: Path,
    receipt_stage_custody: list[Any],
    *,
    expected_state_count: int,
    expected_stage_count: int,
) -> None:
    """Reject every unbound file, directory, or symlink in the sealed tree."""

    if len(receipt_stage_custody) != expected_stage_count:
        raise AdjudicationError("receipt does not bind the required stage-manifest count")
    stage_names: list[str] = []
    for index, untyped_entry in enumerate(receipt_stage_custody):
        entry = _require_dict(untyped_entry, f"receipt stage custody {index}")
        name = entry.get("checkpoint_name")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise AdjudicationError(f"receipt stage custody {index} has an unsafe checkpoint name")
        if name in stage_names:
            raise AdjudicationError(f"duplicate checkpoint name {name}")
        stage_names.append(name)

    expected_files = {
        ".probe.lock",
        "run_contract.json",
        "measurement_receipt.json",
        "complete.json",
        *(f"stage_{name}_complete.json" for name in stage_names),
        *(f"pairs/pair_{pair_index:04d}.json" for pair_index in range(expected_state_count)),
    }
    output_path = source_dir / OUTPUT_NAME
    if output_path.exists() or output_path.is_symlink():
        expected_files.add(OUTPUT_NAME)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in source_dir.rglob("*"):
        relative = path.relative_to(source_dir).as_posix()
        if path.is_symlink():
            raise AdjudicationError(f"sealed source contains unbound symlink {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise AdjudicationError(f"sealed source contains unsupported entry {relative}")
    if actual_directories != {"pairs"}:
        extra_directories = sorted(actual_directories - {"pairs"})
        missing_directories = sorted({"pairs"} - actual_directories)
        raise AdjudicationError(
            f"sealed source directory inventory changed: extra={extra_directories}, missing={missing_directories}"
        )
    if actual_files != expected_files:
        extra_files = sorted(actual_files - expected_files)
        missing_files = sorted(expected_files - actual_files)
        raise AdjudicationError(f"sealed source file inventory changed: extra={extra_files}, missing={missing_files}")


def _validate_stage_tree(
    source_dir: Path,
    receipt_stage_custody: list[Any],
    *,
    snapshot: dict[str, dict[str, Any]],
    contract_info: dict[str, Any],
    expected_state_count: int,
    expected_stage_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(receipt_stage_custody) != expected_stage_count:
        raise AdjudicationError("receipt does not bind the required stage-manifest count")
    stage_names: list[str] = []
    for index, untyped_entry in enumerate(receipt_stage_custody):
        entry = _require_dict(untyped_entry, f"receipt stage custody {index}")
        name = entry.get("checkpoint_name")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise AdjudicationError(f"receipt stage custody {index} has an unsafe checkpoint name")
        if name in stage_names:
            raise AdjudicationError(f"duplicate checkpoint name {name}")
        stage_names.append(name)

    rows: list[dict[str, Any]] = []
    pair_custody: list[dict[str, Any]] = []
    normalized_stage_custody: list[dict[str, Any]] = []
    seen_pair_indices: set[int] = set()
    run_contract_sha256 = contract_info["sha256"]
    holdout_period = contract_info["holdout_period"]
    seed = contract_info["seed"]

    for checkpoint_index, checkpoint_name in enumerate(stage_names):
        manifest_path = source_dir / f"stage_{checkpoint_name}_complete.json"
        manifest, manifest_file = _read_json(
            source_dir,
            manifest_path,
            f"stage manifest {checkpoint_name}",
            snapshot,
        )
        receipt_entry = _require_dict(
            receipt_stage_custody[checkpoint_index], f"receipt stage custody {checkpoint_name}"
        )
        expected_receipt_entry = {
            "checkpoint_name": checkpoint_name,
            "run_contract_sha256": run_contract_sha256,
            "state_count": manifest.get("state_count"),
            "tree_sha256": manifest.get("tree_sha256"),
            "path": str(manifest_path),
            "bytes": manifest_file["bytes"],
            "sha256": manifest_file["sha256"],
        }
        if receipt_entry != expected_receipt_entry:
            raise AdjudicationError(f"receipt custody changed for stage {checkpoint_name}")
        if manifest.get("schema") != STAGE_SCHEMA:
            raise AdjudicationError(f"stage {checkpoint_name} schema changed")
        if manifest.get("run_contract_sha256") != run_contract_sha256:
            raise AdjudicationError(f"stage {checkpoint_name} run-contract hash changed")
        if manifest.get("checkpoint_name") != checkpoint_name:
            raise AdjudicationError(f"stage {checkpoint_name} name changed")
        manifest_records = _require_list(manifest.get("records"), f"stage {checkpoint_name} record custody")
        state_count = _require_int(manifest.get("state_count"), f"stage {checkpoint_name} state count", minimum=1)
        if state_count != len(manifest_records):
            raise AdjudicationError(f"stage {checkpoint_name} state count changed")
        tree_sha256 = _require_sha256(manifest.get("tree_sha256"), f"stage {checkpoint_name} tree hash")
        if tree_sha256 != canonical_sha256(manifest_records):
            raise AdjudicationError(f"stage {checkpoint_name} tree self-hash changed")

        expected_indices = list(range(checkpoint_index, expected_state_count, expected_stage_count))
        if len(expected_indices) != state_count:
            raise AdjudicationError(f"stage {checkpoint_name} deterministic cohort size changed")
        rederived_record_custody: list[dict[str, Any]] = []
        for record_index, untyped_record in zip(expected_indices, manifest_records, strict=True):
            record = _require_dict(untyped_record, f"stage {checkpoint_name} pair {record_index} custody")
            if _require_int(record.get("pair_index"), "manifest pair index", minimum=0) != record_index:
                raise AdjudicationError(f"stage {checkpoint_name} pair ordering changed")
            relative_path = f"pairs/pair_{record_index:04d}.json"
            if record.get("path") != relative_path:
                raise AdjudicationError(f"pair {record_index} path changed")
            row_path = source_dir / relative_path
            row, row_file = _read_json(source_dir, row_path, f"pair record {record_index}", snapshot)
            expected_record_custody = {
                "pair_index": record_index,
                "path": relative_path,
                "bytes": row_file["bytes"],
                "sha256": row_file["sha256"],
            }
            if record != expected_record_custody:
                raise AdjudicationError(f"pair {record_index} byte custody changed")
            assignment = _expected_assignment(
                record_index,
                checkpoint_index=checkpoint_index,
                checkpoint_name=checkpoint_name,
                holdout_period=holdout_period,
                seed=seed,
            )
            record_content_sha256 = _validate_pair_row(
                row,
                expected_assignment=assignment,
                run_contract_sha256=run_contract_sha256,
            )
            if record_index in seen_pair_indices:
                raise AdjudicationError(f"pair {record_index} appears in multiple stages")
            seen_pair_indices.add(record_index)
            rows.append(row)
            pair_custody.append(
                {
                    **expected_record_custody,
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_name": checkpoint_name,
                    "record_content_sha256": record_content_sha256,
                    "run_contract_sha256": run_contract_sha256,
                }
            )
            rederived_record_custody.append(expected_record_custody)
        if manifest_records != rederived_record_custody:
            raise AdjudicationError(f"stage {checkpoint_name} record custody changed")
        normalized_stage_custody.append(
            {
                **manifest_file,
                "checkpoint_index": checkpoint_index,
                "checkpoint_name": checkpoint_name,
                "run_contract_sha256": run_contract_sha256,
                "state_count": state_count,
                "tree_sha256": tree_sha256,
            }
        )

    expected_indices = set(range(expected_state_count))
    if seen_pair_indices != expected_indices:
        raise AdjudicationError("stage manifests do not cover exactly the expected pair indices")
    ordered_rows = sorted(rows, key=lambda row: int(row["assignment"]["pair_index"]))
    ordered_pair_custody = sorted(pair_custody, key=lambda entry: int(entry["pair_index"]))
    return ordered_rows, normalized_stage_custody, ordered_pair_custody


def _accepted_row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("eligible_for_k2") is True]
    accepted = [row for row in eligible if row.get("reuse_guard_accept") is True and _guard_passes(row)]
    inconsistent = sum(row.get("reuse_guard_accept") is True and not _guard_passes(row) for row in eligible)
    gradient_present = 0
    gradient_lt_one = 0
    d_seg_regret_present = 0
    d_seg_regret_lte_zero = 0
    costate_cosines: list[float] = []
    costate_relative_l2: list[float] = []
    renderer_cosines: list[float] = []
    renderer_relative_l2: list[float] = []
    accepted_renderer_cosines: list[float] = []
    accepted_renderer_relative_l2: list[float] = []
    accepted_regrets: dict[str, list[float]] = {key: [] for key in ("ce", "d_seg", "d_pose")}
    eligible_regrets: dict[str, list[float]] = {key: [] for key in ("ce", "d_seg", "d_pose")}
    for row in rows:
        costate = row.get("costate_fidelity")
        if costate is not None:
            costate_dict = _require_dict(costate, "costate fidelity")
            costate_cosines.append(_require_finite_float(costate_dict.get("cosine_fp32"), "costate cosine"))
            costate_relative_l2.append(
                _require_finite_float(costate_dict.get("relative_l2_error_fp32"), "costate relative L2")
            )
        renderer = row.get("renderer_gradient_fidelity")
        if renderer is not None:
            renderer_dict = _require_dict(renderer, "renderer-gradient fidelity")
            renderer_cosines.append(_require_finite_float(renderer_dict.get("cosine_fp32"), "renderer-gradient cosine"))
            renderer_relative_l2.append(
                _require_finite_float(
                    renderer_dict.get("relative_l2_error_fp32"),
                    "renderer-gradient relative L2",
                )
            )
        if row.get("eligible_for_k2") is True and isinstance(row.get("stale_minus_exact_regret"), dict):
            regret_dict = row["stale_minus_exact_regret"]
            for key in eligible_regrets:
                eligible_regrets[key].append(_require_finite_float(regret_dict.get(key), f"eligible {key} regret"))
    for row in accepted:
        gradient = row.get("renderer_gradient_fidelity")
        if isinstance(gradient, dict):
            relative_l2 = gradient.get("relative_l2_error_fp32")
            if isinstance(relative_l2, (int, float)) and not isinstance(relative_l2, bool):
                relative_l2_float = float(relative_l2)
                if math.isfinite(relative_l2_float):
                    gradient_present += 1
                    gradient_lt_one += relative_l2_float < 1.0
                    accepted_renderer_relative_l2.append(relative_l2_float)
            cosine = gradient.get("cosine_fp32")
            if isinstance(cosine, (int, float)) and not isinstance(cosine, bool):
                cosine_float = float(cosine)
                if math.isfinite(cosine_float):
                    accepted_renderer_cosines.append(cosine_float)
        regret = row.get("stale_minus_exact_regret")
        if isinstance(regret, dict):
            d_seg = regret.get("d_seg")
            if isinstance(d_seg, (int, float)) and not isinstance(d_seg, bool):
                d_seg_float = float(d_seg)
                if math.isfinite(d_seg_float):
                    d_seg_regret_present += 1
                    d_seg_regret_lte_zero += d_seg_float <= 0.0
            for key in accepted_regrets:
                value = regret.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value_float = float(value)
                    if math.isfinite(value_float):
                        accepted_regrets[key].append(value_float)
    state_count = len(rows)
    accepted_count = len(accepted)
    eligible_count = len(eligible)
    checkpoint_counts = dict(Counter(str(row["assignment"]["checkpoint_name"]) for row in rows))
    status_counts = dict(Counter(str(row["status"]) for row in rows))
    return {
        "state_count": state_count,
        "unique_pair_count": len({int(row["assignment"]["pair_index"]) for row in rows}),
        "checkpoint_counts": checkpoint_counts,
        "status_counts": status_counts,
        "eligible_state_count": eligible_count,
        "terminal_or_blocked_state_count": state_count - eligible_count,
        "behavioral_full_facet_accept_count": accepted_count,
        "reuse_guard_accept_count": accepted_count,
        "inconsistent_accept_flag_count": inconsistent,
        "reuse_guard_fallback_count": eligible_count - accepted_count,
        "reuse_guard_accept_fraction": accepted_count / eligible_count if eligible_count else None,
        "calibration_fallback_count": state_count - accepted_count,
        "calibration_accept_fraction": accepted_count / state_count if state_count else None,
        "accepted_gradient_fidelity": {
            "present_count": gradient_present,
            "relative_l2_strict_lt_one_count": gradient_lt_one,
            "threshold": 1.0,
            "comparator": "strict_lt",
        },
        "accepted_d_seg_regret": {
            "present_count": d_seg_regret_present,
            "lte_zero_count": d_seg_regret_lte_zero,
            "threshold": 0.0,
            "comparator": "lte",
        },
        "fidelity_distributions": {
            "costate_fidelity": {
                "row_count": len(costate_relative_l2),
                "cosine_fp32": _quantiles(costate_cosines),
                "relative_l2_error_fp32": _quantiles(costate_relative_l2),
            },
            "renderer_gradient_fidelity": {
                "row_count": len(renderer_relative_l2),
                "cosine_fp32": _quantiles(renderer_cosines),
                "relative_l2_error_fp32": _quantiles(renderer_relative_l2),
            },
            "accepted_renderer_gradient_fidelity": {
                "row_count": len(accepted_renderer_relative_l2),
                "cosine_fp32": _quantiles(accepted_renderer_cosines),
                "relative_l2_error_fp32": _quantiles(accepted_renderer_relative_l2),
            },
            "accepted_stale_minus_exact_regret": {
                key: {"row_count": len(values), **_quantiles(values)} for key, values in accepted_regrets.items()
            },
            "all_eligible_stale_minus_exact_regret": {
                key: {"row_count": len(values), **_quantiles(values)} for key, values in eligible_regrets.items()
            },
        },
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def derive_corrected_economics(accept_fraction: float, forward_share_alpha: float) -> dict[str, Any]:
    """Return the corrected two-step diagnostic accounting.

    A rejection pays the stale-candidate forward guard and, after rollback, a
    complete exact forward-plus-backward refresh.  The guard forward cannot be
    recycled across the rollback boundary.
    """

    p = _require_finite_float(accept_fraction, "accept fraction")
    alpha = _require_finite_float(forward_share_alpha, "forward share")
    if not 0.0 <= p <= 1.0:
        raise ValueError("accept_fraction must be in [0,1]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("forward_share_alpha must be in [0,1)")
    q = 1.0 - p
    guarded_cost = 2.0 + alpha - p
    speedup = 2.0 / guarded_cost
    ceiling = 1.0 / (1.0 - alpha)
    threshold = 3.0 * alpha
    beats_ceiling = p > threshold
    return {
        "accept_fraction_p": p,
        "fallback_fraction_q": q,
        "forward_share_alpha": alpha,
        "two_step_baseline_cost": 2.0,
        "accepted_cycle_cost": 1.0 + alpha,
        "rejected_cycle_cost": 2.0 + alpha,
        "rejected_second_step_charge": {
            "guard_forward": alpha,
            "rollback_exact_forward_plus_backward_refresh": 1.0,
            "total": 1.0 + alpha,
            "forbidden_undercharge": "guard_forward_plus_backward_without_restored_state_forward",
        },
        "guarded_expected_cost": guarded_cost,
        "guarded_expected_cost_formula": "2+alpha-p",
        "corrected_teacher_slice_speedup_x": speedup,
        "corrected_teacher_slice_speedup_formula": "2/(2+alpha-p)",
        "positive_speedup_strict": p > alpha,
        "positive_speedup_threshold_strict_gt": alpha,
        "forward_elimination_amdahl_ceiling_x": ceiling,
        "required_accept_fraction_strict_gt": threshold,
        "required_accept_fraction_formula": "3*alpha",
        "beats_forward_elimination_amdahl_ceiling_strict": beats_ceiling,
        "exact_backward_call_amortization_x": 2.0 / (2.0 - p),
        "exact_backward_call_amortization_formula": "2/(2-p)",
        "exact_backward_call_reduction_fraction": p / 2.0,
        "exact_backward_call_reduction_formula": "p/2",
        "evidence_grade": "DERIVED_DIAGNOSTIC_NOT_IN_LOOP",
        "whole_epoch_speedup": DIAGNOSTIC_ONLY_NO_TIMING_AUTHORITY,
    }


def _expected_exact_call_economics(state_count: int, accepted_count: int) -> dict[str, Any]:
    fallback_count = state_count - accepted_count
    baseline = 2 * state_count
    guarded = state_count + fallback_count
    return {
        "calibration_two_step_opportunities": state_count,
        "accepted_reuses": accepted_count,
        "fallback_refreshes": fallback_count,
        "baseline_exact_costate_calls": baseline,
        "guarded_k2_exact_costate_calls": guarded,
        "exact_costate_calls_saved": accepted_count,
        "exact_call_amortization_x": baseline / guarded,
        "backward_call_reduction_fraction": accepted_count / baseline,
    }


def _validate_original_measurement(
    measurement: dict[str, Any],
    rederived: dict[str, Any],
    *,
    alpha: float,
) -> None:
    scalar_keys = (
        "state_count",
        "unique_pair_count",
        "checkpoint_counts",
        "status_counts",
        "eligible_state_count",
        "terminal_or_blocked_state_count",
        "reuse_guard_accept_count",
        "behavioral_full_facet_accept_count",
        "inconsistent_accept_flag_count",
        "reuse_guard_fallback_count",
        "reuse_guard_accept_fraction",
        "calibration_fallback_count",
        "calibration_accept_fraction",
    )
    for key in scalar_keys:
        if measurement.get(key) != rederived[key]:
            raise AdjudicationError(f"original receipt aggregate changed at {key}")
    accepted_count = rederived["behavioral_full_facet_accept_count"]
    if measurement.get("exact_costate_call_economics") != _expected_exact_call_economics(
        rederived["state_count"], accepted_count
    ):
        raise AdjudicationError("original exact-call accounting changed")

    gradient = _require_dict(measurement.get("renderer_gradient_fidelity"), "original renderer-gradient aggregate")
    expected_gradient = rederived["accepted_gradient_fidelity"]
    gradient_fields = {
        "accepted_calibration_row_count": accepted_count,
        "accepted_calibration_fidelity_present_count": expected_gradient["present_count"],
        "accepted_calibration_relative_l2_lt_one_count": expected_gradient["relative_l2_strict_lt_one_count"],
        "accepted_calibration_relative_l2_threshold": 1.0,
        "accepted_calibration_relative_l2_comparator": "strict_lt",
    }
    for key, value in gradient_fields.items():
        if gradient.get(key) != value:
            raise AdjudicationError(f"original accepted-gradient aggregate changed at {key}")

    regret = _require_dict(measurement.get("accepted_d_seg_regret_gate"), "original accepted d_seg-regret aggregate")
    expected_regret = rederived["accepted_d_seg_regret"]
    regret_fields = {
        "accepted_calibration_row_count": accepted_count,
        "accepted_calibration_regret_present_count": expected_regret["present_count"],
        "accepted_calibration_d_seg_regret_lte_zero_count": expected_regret["lte_zero_count"],
        "threshold": 0.0,
        "comparator": "lte",
    }
    for key, value in regret_fields.items():
        if regret.get(key) != value:
            raise AdjudicationError(f"original accepted-regret aggregate changed at {key}")

    distributions = rederived["fidelity_distributions"]
    for aggregate_key in ("costate_fidelity", "renderer_gradient_fidelity"):
        original_distribution = _require_dict(measurement.get(aggregate_key), f"original {aggregate_key} distribution")
        expected_distribution = distributions[aggregate_key]
        for metric_key in ("cosine_fp32", "relative_l2_error_fp32"):
            if original_distribution.get(metric_key) != expected_distribution[metric_key]:
                raise AdjudicationError(f"original {aggregate_key} distribution changed at {metric_key}")
    for aggregate_key in (
        "accepted_stale_minus_exact_regret",
        "all_eligible_stale_minus_exact_regret",
    ):
        original_distribution = _require_dict(measurement.get(aggregate_key), f"original {aggregate_key} distribution")
        expected_distribution = distributions[aggregate_key]
        for metric_key in ("ce", "d_seg", "d_pose"):
            expected_without_count = {
                key: value for key, value in expected_distribution[metric_key].items() if key != "row_count"
            }
            if original_distribution.get(metric_key) != expected_without_count:
                raise AdjudicationError(f"original {aggregate_key} distribution changed at {metric_key}")

    old_economics = _require_dict(
        measurement.get("diagnostic_teacher_slice_economics"),
        "original diagnostic teacher-slice economics",
    )
    if old_economics.get("forward_share_alpha") != alpha:
        raise AdjudicationError("original diagnostic forward share changed")
    source_fallback_rate = rederived["calibration_fallback_count"] / rederived["state_count"]
    if old_economics.get("fallback_rate") != source_fallback_rate:
        raise AdjudicationError("original diagnostic fallback rate changed")


def _validate_quantile_replay_environment(receipt: dict[str, Any]) -> str:
    """Fail before row aggregation when NumPy quantile semantics may drift."""

    host = _require_dict(receipt.get("host"), "original receipt host environment")
    source_numpy_version = host.get("numpy")
    if not isinstance(source_numpy_version, str) or not source_numpy_version:
        raise AdjudicationError("original receipt NumPy version is missing")
    if source_numpy_version != np.__version__:
        raise AdjudicationError(
            f"NumPy version drift blocks exact quantile replay: source={source_numpy_version}, current={np.__version__}"
        )
    return source_numpy_version


def _validate_original_receipt(
    receipt: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_info: dict[str, Any],
    complete: dict[str, Any],
    receipt_file: dict[str, Any],
    receipt_stage_custody: list[Any],
    rederived: dict[str, Any],
    expected_state_count: int,
) -> tuple[dict[str, Any], str]:
    if complete.get("schema") != COMPLETE_SCHEMA:
        raise AdjudicationError("completion-seal schema changed")
    if complete.get("receipt") != "measurement_receipt.json":
        raise AdjudicationError("completion-seal receipt path changed")
    if complete.get("receipt_bytes") != receipt_file["bytes"]:
        raise AdjudicationError("completion-seal receipt byte count changed")
    if complete.get("receipt_sha256") != receipt_file["sha256"]:
        raise AdjudicationError("completion-seal receipt hash changed")
    if receipt.get("schema") != SOURCE_SCHEMA or receipt.get("status") != "completed":
        raise AdjudicationError("original measurement receipt is not completed v2 evidence")
    if receipt.get("lane_id") != EXPECTED_LANE_ID:
        raise AdjudicationError("original measurement receipt lane changed")
    if receipt.get("axis") != EXPECTED_AXIS:
        raise AdjudicationError("original measurement receipt authority axis changed")
    if receipt.get("run_contract") != contract:
        raise AdjudicationError("measurement receipt run contract changed")
    if receipt.get("n_pairs") != expected_state_count:
        raise AdjudicationError("measurement receipt state count changed")
    if receipt.get("objective_sha256") != contract_info["objective_sha256"]:
        raise AdjudicationError("measurement receipt objective hash changed")
    if receipt.get("scorer_sha256") != contract_info["scorer_sha256"]:
        raise AdjudicationError("measurement receipt scorer hash changed")
    if receipt.get("stage_manifest_custody") != receipt_stage_custody:
        raise AdjudicationError("measurement receipt stage custody changed")

    measurement = _require_dict(receipt.get("measurement"), "original measurement aggregate")
    _validate_original_measurement(measurement, rederived, alpha=contract_info["alpha"])
    admission_content = _require_dict(receipt.get("admission_content"), "original admission content")
    admission_content_sha256 = _require_sha256(
        receipt.get("admission_content_sha256"), "original admission-content hash"
    )
    if admission_content_sha256 != canonical_sha256(admission_content):
        raise AdjudicationError("original admission-content self-hash changed")
    if admission_content.get("run_contract_sha256") != contract_info["sha256"]:
        raise AdjudicationError("original admission content changed its run contract")
    if admission_content.get("objective_sha256") != contract_info["objective_sha256"]:
        raise AdjudicationError("original admission content changed its objective")
    if admission_content.get("scorer_sha256") != contract_info["scorer_sha256"]:
        raise AdjudicationError("original admission content changed its scorers")
    if admission_content.get("stage_manifest_custody") != receipt_stage_custody:
        raise AdjudicationError("original admission content changed its stage custody")
    if admission_content.get("aggregate_sha256") != canonical_sha256(measurement):
        raise AdjudicationError("original admission content changed its aggregate")

    fidelity_gate = _require_dict(receipt.get("fidelity_gate"), "original fidelity gate")
    original_gate = _require_dict(
        fidelity_gate.get("calibration_admission_gate"), "original calibration admission gate"
    )
    original_spec = _require_dict(original_gate.get("spec"), "original gate spec")
    if original_gate.get("spec_sha256") != canonical_sha256(original_spec):
        raise AdjudicationError("original calibration admission-spec self-hash changed")
    if original_spec != contract_info["payload"].get("admission_spec"):
        raise AdjudicationError("original calibration gate changed its run-contract admission spec")
    if original_gate.get("spec_sha256") != contract_info["payload"].get("admission_spec_sha256"):
        raise AdjudicationError("original calibration gate changed its admission-spec hash")
    if admission_content.get("admission_spec_sha256") != original_gate.get("spec_sha256"):
        raise AdjudicationError("original admission content changed its admission-spec hash")
    original_passed = original_gate.get("passed")
    if not isinstance(original_passed, bool):
        raise AdjudicationError("original calibration gate verdict is not boolean")
    original_verdict = receipt.get("admission_verdict")
    expected_original_verdict = "ADMIT_K2_GUARDED_REUSE" if original_passed else "NOT_ADMITTED"
    if original_verdict != expected_original_verdict:
        raise AdjudicationError("original admission verdict disagrees with its gate")
    if fidelity_gate.get("admission") != original_verdict:
        raise AdjudicationError("original fidelity gate disagrees with its verdict")
    if admission_content.get("admission_verdict") != original_verdict:
        raise AdjudicationError("original admission content disagrees with its verdict")
    if fidelity_gate.get("complete_n600") is not True:
        raise AdjudicationError("original fidelity gate is not a complete sealed replay")
    if fidelity_gate.get("live_trainer_activation") is not False:
        raise AdjudicationError("original receipt carries live-trainer authority")
    if fidelity_gate.get("runtime_exact_gradient_access") is not False:
        raise AdjudicationError("original receipt carries runtime exact-gradient authority")

    authority = _require_dict(receipt.get("authority"), "original receipt authority")
    for field in ("score_claim", "promotion_eligible", "pointer_moved"):
        if authority.get(field) is not False:
            raise AdjudicationError(f"original receipt carries false authority at {field}")
    if authority.get("whole_epoch_speedup") != HISTORICAL_SOURCE_WHOLE_EPOCH_LABEL:
        raise AdjudicationError("original receipt claims unmeasured whole-epoch speedup")
    if authority.get("contest_cpu_cuda") != "NOT_MEASURED":
        raise AdjudicationError("original receipt carries unmeasured contest-axis authority")
    return admission_content, str(original_verdict)


def _corrected_gate(
    rederived: dict[str, Any],
    economics: dict[str, Any],
    *,
    expected_state_count: int,
    expected_stage_count: int,
) -> dict[str, Any]:
    accepted_count = int(rederived["behavioral_full_facet_accept_count"])
    gradient = rederived["accepted_gradient_fidelity"]
    regret = rederived["accepted_d_seg_regret"]
    spec = {
        "schema": CORRECTED_GATE_SCHEMA,
        "complete_state_count": expected_state_count,
        "complete_stage_count": expected_stage_count,
        "guarded_expected_cost_formula": "2+alpha-p",
        "corrected_teacher_slice_speedup_formula": "2/(2+alpha-p)",
        "positive_speedup_formula": "p>alpha, strict",
        "forward_elimination_ceiling_gate_formula": "p>3*alpha, strict",
        "gradient_relative_l2_threshold": 1.0,
        "gradient_relative_l2_comparator": "strict_lt",
        "accepted_stale_minus_exact_d_seg_threshold": 0.0,
        "accepted_stale_minus_exact_d_seg_comparator": "lte",
        "whole_epoch_speedup_routing": (
            "fidelity_stage_only; downstream_provider_resume_parity_validator_required_before_timer_request"
        ),
    }
    predicates = {
        "source_complete": True,
        "exact_state_count": rederived["state_count"] == expected_state_count,
        "exact_unique_pair_count": rederived["unique_pair_count"] == expected_state_count,
        "exact_stage_count": len(rederived["checkpoint_counts"]) == expected_stage_count,
        "has_behavioral_full_facet_accepts": accepted_count > 0,
        "positive_diagnostic_speedup_strict": economics["positive_speedup_strict"],
        "accept_fraction_strict_gt_3alpha": economics["beats_forward_elimination_amdahl_ceiling_strict"],
        "diagnostic_speedup_strictly_exceeds_amdahl_ceiling": economics[
            "beats_forward_elimination_amdahl_ceiling_strict"
        ],
        "all_accepted_gradient_fidelity_present": gradient["present_count"] == accepted_count,
        "all_accepted_gradient_relative_l2_strict_lt_one": (
            gradient["relative_l2_strict_lt_one_count"] == accepted_count
        ),
        "all_accepted_d_seg_regret_present": regret["present_count"] == accepted_count,
        "all_accepted_stale_d_seg_regret_lte_exact": regret["lte_zero_count"] == accepted_count,
        "no_inconsistent_accept_flags": rederived["inconsistent_accept_flag_count"] == 0,
    }
    passed = all(predicates.values())
    corrected_verdict = "ADMIT_K2_GUARDED_REUSE" if passed else "NOT_ADMITTED"
    timing_routing = corrected_timing_routing(corrected_verdict)
    return {
        "schema": CORRECTED_GATE_SCHEMA,
        "spec": spec,
        "spec_sha256": canonical_sha256(spec),
        "predicates": predicates,
        "passed": passed,
        "behavioral_full_facet_accept_count": accepted_count,
        "observed_state_count": rederived["state_count"],
        "observed_unique_pair_count": rederived["unique_pair_count"],
        "measured_accept_fraction": rederived["calibration_accept_fraction"],
        "required_accept_fraction_strict_gt": economics["required_accept_fraction_strict_gt"],
        "corrected_teacher_slice_speedup_x": economics["corrected_teacher_slice_speedup_x"],
        "forward_elimination_amdahl_ceiling_x": economics["forward_elimination_amdahl_ceiling_x"],
        "evidence_grade": "DERIVED_DIAGNOSTIC_NOT_IN_LOOP",
        "runtime_exact_gradient_access": False,
        "whole_epoch_speedup": timing_routing["status"],
        "timing_routing": timing_routing,
    }


def _snapshot_tree(snapshot: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [snapshot[path] for path in sorted(snapshot)]


def _assert_snapshot_unchanged(source_dir: Path, snapshot: dict[str, dict[str, Any]]) -> None:
    for relative, expected in snapshot.items():
        path = source_dir / relative
        if path.is_symlink() or not path.is_file():
            raise AdjudicationError(f"sealed source changed during adjudication at {relative}")
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise AdjudicationError(f"sealed source changed during adjudication at {relative}")


def _atomic_write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise AdjudicationError(f"refusing to overwrite non-identical {path.name}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.is_file() and path.read_bytes() == payload:
                temporary.unlink()
                return
            raise AdjudicationError(f"refusing to race non-identical {path.name}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _build_wrapper(
    source_dir: Path,
    *,
    expected_state_count: int,
    expected_stage_count: int,
    expected_roots: _ExpectedSourceRoots,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    snapshot: dict[str, dict[str, Any]] = {}
    complete, complete_file = _read_json(source_dir, source_dir / "complete.json", "completion seal", snapshot)
    receipt, receipt_file = _read_json(
        source_dir,
        source_dir / "measurement_receipt.json",
        "measurement receipt",
        snapshot,
    )
    if complete_file["sha256"] != expected_roots.complete_sha256:
        raise AdjudicationError("completion seal is not the reviewed source byte root")
    if receipt_file["sha256"] != expected_roots.measurement_receipt_sha256:
        raise AdjudicationError("measurement receipt is not the reviewed source byte root")
    source_numpy_version = _validate_quantile_replay_environment(receipt)
    contract, contract_file = _read_json(source_dir, source_dir / "run_contract.json", "run contract", snapshot)
    contract_info = _validate_run_contract(
        source_dir,
        contract,
        expected_state_count=expected_state_count,
        expected_stage_count=expected_stage_count,
        expected_roots=expected_roots,
    )
    receipt_stage_custody = _require_list(receipt.get("stage_manifest_custody"), "receipt stage-manifest custody")
    _validate_exact_source_inventory(
        source_dir,
        receipt_stage_custody,
        expected_state_count=expected_state_count,
        expected_stage_count=expected_stage_count,
    )
    rows, stage_custody, pair_custody = _validate_stage_tree(
        source_dir,
        receipt_stage_custody,
        snapshot=snapshot,
        contract_info=contract_info,
        expected_state_count=expected_state_count,
        expected_stage_count=expected_stage_count,
    )
    rederived = _accepted_row_metrics(rows)
    admission_content, original_verdict = _validate_original_receipt(
        receipt,
        contract=contract,
        contract_info=contract_info,
        complete=complete,
        receipt_file=receipt_file,
        receipt_stage_custody=receipt_stage_custody,
        rederived=rederived,
        expected_state_count=expected_state_count,
    )
    accept_fraction = rederived["calibration_accept_fraction"]
    if accept_fraction is None:
        raise AdjudicationError("sealed replay has no calibration states")
    economics = derive_corrected_economics(accept_fraction, contract_info["alpha"])
    gate = _corrected_gate(
        rederived,
        economics,
        expected_state_count=expected_state_count,
        expected_stage_count=expected_stage_count,
    )
    corrected_verdict = "ADMIT_K2_GUARDED_REUSE" if gate["passed"] else "NOT_ADMITTED"
    timing_routing = corrected_timing_routing(corrected_verdict)
    if gate["timing_routing"] != timing_routing:
        raise AdjudicationError("corrected gate timing routing disagrees with its verdict")
    economics["whole_epoch_speedup"] = timing_routing["status"]
    source_tree = _snapshot_tree(snapshot)
    wrapper: dict[str, Any] = {
        "schema": WRAPPER_SCHEMA,
        "status": "completed_source_revalidated",
        "lane_id": receipt.get("lane_id"),
        "axis": receipt.get("axis"),
        "source_adjudication": {
            "original_schema": receipt.get("schema"),
            "original_admission_verdict": original_verdict,
            "original_admission_verdict_status": SUPERSEDED_LABEL,
            "original_admission_content_sha256": receipt["admission_content_sha256"],
            "original_admission_content_revalidated_sha256": canonical_sha256(admission_content),
            "objective_sha256": contract_info["objective_sha256"],
            "scorer_sha256": contract_info["scorer_sha256"],
            "run_contract_sha256": contract_info["sha256"],
            "reviewed_source_roots": asdict(expected_roots),
        },
        "source_custody": {
            "run_contract": contract_file,
            "measurement_receipt": receipt_file,
            "complete": complete_file,
            "stage_manifests": stage_custody,
            "pair_records": pair_custody,
            "source_file_count": len(source_tree),
            "source_tree": source_tree,
            "source_tree_sha256": canonical_sha256(source_tree),
        },
        "measurement_rederived_from_sealed_rows": rederived,
        "quantile_replay_environment": {
            "source_numpy_version": source_numpy_version,
            "current_numpy_version": np.__version__,
            "exact_version_match": True,
            "quantile_schema": "min/p10/median/mean/p90/max; NumPy default linear method",
        },
        "corrected_diagnostic_economics": economics,
        "corrected_admission_gate": gate,
        "corrected_admission_verdict": corrected_verdict,
        "timing_routing": timing_routing,
        "execution": {
            "teacher_calls": 0,
            "scorer_calls": 0,
            "renderer_calls": 0,
            "source_row_rewrites": 0,
            "source_manifest_rewrites": 0,
            "source_receipt_rewrites": 0,
            "method": "deterministic byte-custody revalidation and arithmetic-only re-adjudication",
        },
        "authority": {
            "research_only": True,
            "training_signal_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "live_trainer_activation": False,
            "runtime_exact_gradient_access": False,
            "whole_epoch_speedup": timing_routing["status"],
            "operator_go_request_eligible": timing_routing["operator_go_request_eligible"],
            "operator_go_granted": timing_routing["operator_go_granted"],
            "contest_cpu_cuda": "NOT_MEASURED",
        },
    }
    wrapper["adjudication_content_sha256"] = canonical_sha256(wrapper)
    return wrapper, snapshot


def _adjudicate_with_expected_roots(
    source_dir: Path,
    *,
    expected_state_count: int,
    expected_stage_count: int,
    expected_roots: _ExpectedSourceRoots,
) -> dict[str, Any]:
    """Private seam supporting explicit, non-authority synthetic fixture roots."""

    source_dir = source_dir.resolve()
    _require_int(expected_state_count, "expected_state_count", minimum=1)
    _require_int(expected_stage_count, "expected_stage_count", minimum=1)
    _validate_expected_roots(expected_roots)
    if not source_dir.is_dir():
        raise AdjudicationError("sealed source directory is missing")
    lock_path = source_dir / ".probe.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise AdjudicationError("sealed replay lock file is missing")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AdjudicationError("source replay is still running") from exc
        wrapper, snapshot = _build_wrapper(
            source_dir,
            expected_state_count=expected_state_count,
            expected_stage_count=expected_stage_count,
            expected_roots=expected_roots,
        )
        _assert_snapshot_unchanged(source_dir, snapshot)
        encoded = (json.dumps(wrapper, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        _atomic_write_once(source_dir / OUTPUT_NAME, encoded)
        return wrapper
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def adjudicate(source_dir: Path) -> dict[str, Any]:
    """Validate and adjudicate only the authoritative n600/three-stage source."""

    return _adjudicate_with_expected_roots(
        source_dir,
        expected_state_count=EXPECTED_STATE_COUNT,
        expected_stage_count=EXPECTED_STAGE_COUNT,
        expected_roots=_public_expected_roots(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="sealed n600 v3 result directory; the corrected wrapper is written adjacent",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    wrapper = adjudicate(args.source_dir)
    output_path = args.source_dir.resolve() / OUTPUT_NAME
    print(
        json.dumps(
            {
                "corrected_admission_verdict": wrapper["corrected_admission_verdict"],
                "output": str(output_path),
                "output_bytes": output_path.stat().st_size,
                "output_sha256": file_sha256(output_path),
                "schema": wrapper["schema"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
