#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay SFESS on the sealed UGC 64-state objective without scorer calls.

This is an intentionally narrow measurement harness.  It accepts only a new
output directory or an existing run directory to resume.  The objective bytes,
budget, seed, cardinality ladder, estimator sample count, proposal rule, and
comparison floor are compiled by the typed policy; there are no semantic CLI
knobs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from tac.sfess_cached_replay import (  # noqa: E402
    PINNED_UGC64_SHA256,
    CountedCachedOracle,
    QueryRecord,
    SFESSFixedKSearch,
    cached_state_sha256,
    load_cached_objective_jsonl,
)


def _load_dependency_light_policy_module() -> Any:
    """Execute the isolated DSL leaf without importing the eager DSL package."""

    module_name = "_tac_sfess_cached_replay_policy_runtime"
    source = REPO / "src/tac/witness_dsl/sfess_cached_replay_policy.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load isolated SFESS policy from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_POLICY_MODULE = _load_dependency_light_policy_module()
SFESSCacheCustody = _POLICY_MODULE.SFESSCacheCustody
SFESSCachedReplayPolicy = _POLICY_MODULE.SFESSCachedReplayPolicy
SFESSObjectiveContext = _POLICY_MODULE.SFESSObjectiveContext

AXIS = "[macOS-CPU advisory . frozen CPU-torch exact cells . NON-PROMOTABLE]"
RESULTS_ROOT = REPO / "experiments/results"
TABLE = REPO / (
    "experiments/results/ugc_terminal_polish_ab_20260712/"
    "search_exact_enumeration_accepted_proposals.jsonl"
)
SOURCE_RECEIPT = REPO / "experiments/results/ugc_terminal_polish_ab_20260712/measurement_receipt.json"
CANDIDATE_MANIFEST = REPO / "experiments/results/ugc_terminal_polish_ab_20260712/candidate_manifest.json"
SOURCE_VIDEO = REPO / "upstream/videos/0.mkv"
SOURCE_VIDEO_SHA256 = "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"
SOURCE_VIDEO_BYTES = 37_545_489
SOURCE_RECEIPT_SHA256 = "b2f7a87b43ce5face651da5caf4cd723884445d4fa04f92c007811b40d32b357"
CANDIDATE_MANIFEST_SHA256 = "fb99a24410d4b7dbb8ccf3d8ecba67c8f2033b732440ba32a623e0dec9d6fce0"
FIXTURE_ARCHIVE_SHA256 = "9c2afa96abdd6fa401bbdfa7a29a7f26ef67c70540656b6fd9ffd87d0bb91d6c"
FIXTURE_AUTHORITY_SHA256 = "0eae0bdef32431514561acea3b0354a9fd4098e8f68af9ec838c9ec6518d62c0"
GT_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
SEED = 396_400
N_BITS = 6
K_VALUES = (1, 2, 3, 4, 5)
SAMPLES_PER_GRADIENT = 5
EVAL_BUDGET = 64
NOISE_FLOOR_S = 1.0e-12
RUN_SCHEMA = "sfess_cached_replay_ugc64_run.v2"
RECEIPT_SCHEMA = "sfess_cached_replay_ugc64_receipt.v2"
OUTPUT_NAME_PATTERN = re.compile(r"^sfess_cached_replay_ugc64_\d{8}T\d{6}Z$")
FORBIDDEN_MODULE_PREFIXES = (
    "torch",
    "mlx",
    "modal",
    "tac.boundary_math",
    "tac.click_polish",
    "tac.cuda_levelset_training",
    "tac.local_acceleration",
    "tac.renderer",
    "tac.scorer",
    "tac.through_r.mc_finisher",
    "tac.training",
    "tac.witness_dsl",
    "upstream",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execution_source_custody() -> dict[str, Any]:
    """Content-address every repository Python source loaded by this process."""

    entrypoint = Path(__file__).resolve()
    modules: dict[str, dict[str, Any]] = {
        "__entrypoint__": {
            "path": entrypoint.relative_to(REPO).as_posix(),
            "sha256": _sha256(entrypoint),
            "bytes": entrypoint.stat().st_size,
        }
    }
    for module_name, module in sorted(sys.modules.items()):
        source_name = getattr(module, "__file__", None)
        if not isinstance(source_name, str):
            continue
        source = Path(source_name).resolve()
        if source.suffix != ".py" or not source.is_file():
            continue
        try:
            relative = source.relative_to(REPO).as_posix()
        except ValueError:
            continue
        modules[module_name] = {
            "path": relative,
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
        }
    required_paths = {
        "tools/probe_sfess_cached_replay.py",
        "src/tac/sfess_cached_replay.py",
        "src/tac/witness_dsl/sfess_cached_replay_policy.py",
    }
    observed_paths = {str(row["path"]) for row in modules.values()}
    missing = sorted(required_paths - observed_paths)
    if missing:
        raise RuntimeError(f"execution source custody is missing required modules: {missing}")
    encoded = json.dumps(modules, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return {
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "module_alias_count": len(modules),
        "unique_source_count": len({str(row["path"]) for row in modules.values()}),
        "modules": modules,
    }


def _execution_source_git_state(source_custody: dict[str, Any]) -> dict[str, Any]:
    """Record whether each measured source byte string existed at the base commit."""

    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--full-tree", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    head_oids: dict[str, str] = {}
    for line in listing.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        _mode, object_kind, object_id = metadata.split()
        if object_kind == "blob":
            head_oids[path] = object_id

    unique_paths = sorted({str(row["path"]) for row in source_custody["modules"].values()})
    rows: dict[str, dict[str, Any]] = {}
    for relative in unique_paths:
        worktree_oid = subprocess.run(
            ["git", "hash-object", "--", relative],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head_oid = head_oids.get(relative)
        rows[relative] = {
            "head_blob_oid": head_oid,
            "worktree_blob_oid": worktree_oid,
            "tracked_at_head": head_oid is not None,
            "matches_head": head_oid == worktree_oid,
        }
    return {
        "base_git_head": _git_head(),
        "all_execution_sources_match_head": all(row["matches_head"] for row in rows.values()),
        "files": rows,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _forbidden_imports() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )


def _require_zero_forbidden_imports(stage: str) -> None:
    loaded = _forbidden_imports()
    if loaded:
        raise RuntimeError(f"forbidden scorer/training/cloud modules loaded at {stage}: {loaded}")


def _custody(path: Path, expected_sha256: str) -> Any:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"input SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return SFESSCacheCustody(
        kind="cache",
        path=str(path.resolve()),
        sha256=actual,
        size_bytes=path.stat().st_size,
    )


def _load_and_verify_source_receipt() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    if source.get("schema") != "ugc_terminal_polish_ab_receipt.v2":
        raise RuntimeError("source measurement receipt schema drift")
    if source.get("score_claim") is not False or source.get("promotable") is not False:
        raise RuntimeError("source receipt false-authority fields drifted")
    fixture = source.get("fixture", {})
    expected_fixture = {
        "archive_sha256": FIXTURE_ARCHIVE_SHA256,
        "authority_sha256": FIXTURE_AUTHORITY_SHA256,
        "gt_cache_sha256": GT_CACHE_SHA256,
    }
    for key, expected in expected_fixture.items():
        if fixture.get(key) != expected:
            raise RuntimeError(f"source receipt fixture {key} drift")
    arms = source.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("source receipt has no arms list")
    indexed = {str(row.get("estimator")): row for row in arms if isinstance(row, dict)}
    expected_names = {"exact_enumeration", "one_plus_one_es", "ugc", "disarm", "rloo"}
    if set(indexed) != expected_names:
        raise RuntimeError(f"source baseline set drift: {sorted(indexed)}")
    for name, row in indexed.items():
        if row.get("function_eval_budget_search") != EVAL_BUDGET:
            raise RuntimeError(f"baseline {name} search budget drift")
        if row.get("function_evals_search") != EVAL_BUDGET:
            raise RuntimeError(f"baseline {name} search call count drift")
        if not math.isfinite(float(row.get("best_s"))):
            raise RuntimeError(f"baseline {name} best_s is nonfinite")
    return source, indexed


def _build_policy() -> SFESSCachedReplayPolicy:
    context = SFESSObjectiveContext(
        objective_table_sha256=PINNED_UGC64_SHA256,
        measurement_receipt_sha256=SOURCE_RECEIPT_SHA256,
        candidate_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
        fixture_archive_sha256=FIXTURE_ARCHIVE_SHA256,
        fixture_authority_sha256=FIXTURE_AUTHORITY_SHA256,
        source_video_sha256=SOURCE_VIDEO_SHA256,
        source_video_bytes=SOURCE_VIDEO_BYTES,
        n_bits=N_BITS,
        state_count=1 << N_BITS,
        mask_order="little_endian_bit_j_equals_index_shift_j",
        axis=AXIS,
    )
    return SFESSCachedReplayPolicy(
        mode="sfess_cached_k_subset",
        research_only=True,
        score_claim=False,
        promotion_eligible=False,
        produces_costate=False,
        live_gradient_fallback="full_teacher",
        cache_failure_action="refuse",
        objective_context=context,
        objective_context_fingerprint=context.fingerprint(),
        objective_table_custody=_custody(TABLE, PINNED_UGC64_SHA256),
        measurement_receipt_custody=_custody(SOURCE_RECEIPT, SOURCE_RECEIPT_SHA256),
        candidate_manifest_custody=_custody(CANDIDATE_MANIFEST, CANDIDATE_MANIFEST_SHA256),
        source_video_custody=SFESSCacheCustody(
            kind="source_video",
            path=str(SOURCE_VIDEO.resolve()),
            sha256=_sha256(SOURCE_VIDEO),
            size_bytes=SOURCE_VIDEO.stat().st_size,
        ),
        k_values=K_VALUES,
        include_degenerate_k_controls=True,
        samples_per_gradient=SAMPLES_PER_GRADIENT,
        eval_budget_per_k=EVAL_BUDGET,
        seed=SEED,
        max_evidence_age_queries=0,
        comparison_noise_floor_s=NOISE_FLOOR_S,
        initial_mask_rule="lowest_indices",
        acceptance_rule="strict_improvement_beyond_registered_floor",
        retention_rule="strict_gated_returned_state",
        k_selection_status="post_hoc_exploratory",
        control_variate_anchor="wijk_2024_five_sample_leave_one_out",
    )


def _lookup_decision(
    *, compiled_policy: Any, table: Any, mask: np.ndarray, value: float, query_index: int
) -> dict[str, Any]:
    decision = compiled_policy.authorize_lookup(
        table_source_sha256=table.source_sha256,
        table_n_bits=table.n_bits,
        mask=tuple(int(x) for x in mask),
        value=value,
        declared_state_sha256=table.state_sha256(mask),
        query_index=query_index,
        evidence_query_index=query_index,
        current_objective_context_fingerprint=compiled_policy.source.objective_context.fingerprint(),
    )
    return decision.to_dict()


def _run_sfess_k(
    *,
    k: int,
    out: Path,
    table: Any,
    compiled_policy: Any,
) -> dict[str, Any]:
    snapshot = out / f"k{k}_stage_snapshot.json"
    oracle: CountedCachedOracle
    if k not in compiled_policy.source.k_values:
        raise RuntimeError(f"k={k} is outside the registered non-degenerate ladder")

    def authorize(mask: np.ndarray) -> bool:
        value = table.value(mask)
        query_index = oracle.calls + 1
        decision = _lookup_decision(
            compiled_policy=compiled_policy,
            table=table,
            mask=mask,
            value=value,
            query_index=query_index,
        )
        return bool(decision["admitted_for_cached_lookup"])

    # Start empty even on resume.  `resume_from` replays every semantic field
    # first and restores this exact oracle only after full validation succeeds.
    oracle = CountedCachedOracle(table, EVAL_BUDGET, authorize)
    search = (
        SFESSFixedKSearch.resume_from(
            snapshot,
            oracle,
            expected_k=k,
            expected_samples_per_gradient=compiled_policy.source.samples_per_gradient,
            expected_seed=compiled_policy.source.seed,
            expected_comparison_noise_floor_s=compiled_policy.source.comparison_noise_floor_s,
        )
        if snapshot.exists()
        else SFESSFixedKSearch(
            oracle,
            N_BITS,
            k,
            SAMPLES_PER_GRADIENT,
            SEED,
            compiled_policy.source.comparison_noise_floor_s,
        )
    )
    result = search.run(EVAL_BUDGET, snapshot)
    if not result.complete or result.calls != EVAL_BUDGET:
        raise RuntimeError(f"k={k} replay did not consume the exact budget")
    if any(sum(record.mask) != k for record in result.query_records):
        raise RuntimeError(f"k={k} query trace violated fixed cardinality")
    decisions = [
        _lookup_decision(
            compiled_policy=compiled_policy,
            table=table,
            mask=np.asarray(record.mask, dtype=np.uint8),
            value=record.value,
            query_index=record.call_index,
        )
        for record in result.query_records
    ]
    if not all(decision["admitted_for_cached_lookup"] for decision in decisions):
        raise RuntimeError(f"k={k} had a refused cached lookup")
    posthoc_states = [
        tuple((index >> bit) & 1 for bit in range(N_BITS))
        for index in range(1 << N_BITS)
        if index.bit_count() == k
    ]
    fixed_k_reference_mask = min(posthoc_states, key=lambda mask: table.value(mask))
    fixed_k_reference_value = table.value(fixed_k_reference_mask)
    query_payload = [record.to_dict() for record in result.query_records]
    query_trace_sha256 = hashlib.sha256(
        json.dumps(query_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    row = {
        "arm": f"sfess_k{k}",
        "k": search.k,
        "seed": search.seed,
        "samples_per_gradient": search.samples_per_gradient,
        "function_eval_budget": EVAL_BUDGET,
        "function_evals": result.calls,
        "gradient_sample_calls": sum(r.purpose == "sfess_sample" for r in result.query_records),
        "strict_gate_calls": sum(r.purpose == "strict_exact_gate" for r in result.query_records),
        "initial_calls": sum(r.purpose == "initial" for r in result.query_records),
        "budget_padding_calls": result.padding,
        "accepted_swaps": result.accepted,
        "start_mask": [1 if index < k else 0 for index in range(N_BITS)],
        "best_mask": list(result.best_mask),
        "best_s": result.best_value,
        "delta_s_from_fixed_k_start": result.best_value - table.value(
            tuple(1 if index < k else 0 for index in range(N_BITS))
        ),
        "retention_rule": "strict_gated_returned_state",
        "sample_values_retained_as_candidates": False,
        "posthoc_exhaustive_fixed_k_reference_mask": list(fixed_k_reference_mask),
        "posthoc_exhaustive_fixed_k_reference_s": fixed_k_reference_value,
        "reached_fixed_k_reference": abs(result.best_value - fixed_k_reference_value) <= NOISE_FLOOR_S,
        "query_trace_sha256": query_trace_sha256,
        "query_records": query_payload,
        "lookup_decisions_rederived_after_completion": decisions,
        "complete": result.complete,
        "review_status": "recovery-written-UNREVIEWED",
    }
    _atomic_json(out / f"k{k}_stage_receipt.json", row)
    return row


def _run_degenerate_control(
    *, k: int, out: Path, table: Any, compiled_policy: Any
) -> dict[str, Any]:
    snapshot = out / f"k{k}_stage_snapshot.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.exists() else {}
    if snapshot.exists():
        expected_snapshot = {
            "schema": "sfess_degenerate_control_snapshot.v1",
            "source_sha256": table.source_sha256,
            "k": k,
            "eval_budget": EVAL_BUDGET,
        }
        for key, expected in expected_snapshot.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"degenerate k={k} snapshot {key} drift")
    prior_payload = payload.get("query_records", [])
    if not isinstance(prior_payload, list):
        raise RuntimeError(f"degenerate k={k} snapshot query records malformed")
    mask = np.full(N_BITS, 1 if k == N_BITS else 0, dtype=np.uint8)
    expected_mask = tuple(int(x) for x in mask)
    expected_value = table.value(mask)
    parsed_records = tuple(
        QueryRecord.from_dict(record, N_BITS)
        for record in prior_payload
        if isinstance(record, dict)
    )
    if len(parsed_records) != len(prior_payload):
        raise RuntimeError(f"degenerate k={k} snapshot query record is not an object")
    for index, record in enumerate(parsed_records, start=1):
        expected_purpose = "degenerate_initial" if index == 1 else "budget_padding"
        if record.call_index != index:
            raise RuntimeError(f"degenerate k={k} snapshot call indices are not contiguous")
        if record.purpose != expected_purpose:
            raise RuntimeError(f"degenerate k={k} snapshot purpose schedule mismatch")
        if record.mask != expected_mask or record.value != expected_value:
            raise RuntimeError(f"degenerate k={k} snapshot state/value mismatch")
        if table.state_sha256(mask) != record.state_sha256:
            raise RuntimeError(f"degenerate k={k} snapshot state fingerprint mismatch")
    if len(parsed_records) > EVAL_BUDGET:
        raise RuntimeError(f"degenerate k={k} snapshot exceeds the query budget")
    oracle: CountedCachedOracle

    def authorize(candidate: np.ndarray) -> bool:
        value = table.value(candidate)
        query_index = oracle.calls + 1
        decision = compiled_policy.authorize_lookup(
            table_source_sha256=table.source_sha256,
            table_n_bits=table.n_bits,
            mask=tuple(int(x) for x in candidate),
            value=value,
            declared_state_sha256=cached_state_sha256(candidate, value),
            query_index=query_index,
            evidence_query_index=query_index,
            current_objective_context_fingerprint=compiled_policy.source.objective_context.fingerprint(),
        )
        return decision.admitted_for_cached_lookup

    oracle = CountedCachedOracle(table, EVAL_BUDGET, authorize)
    oracle.restore_records(parsed_records)
    if any(sum(record.mask) != k for record in oracle.records):
        raise RuntimeError(f"degenerate k={k} prior trace violates fixed cardinality")
    while oracle.calls < EVAL_BUDGET:
        oracle(mask, purpose="degenerate_initial" if oracle.calls == 0 else "budget_padding")
        _atomic_json(
            snapshot,
            {
                "schema": "sfess_degenerate_control_snapshot.v1",
                "source_sha256": table.source_sha256,
                "k": k,
                "eval_budget": EVAL_BUDGET,
                "query_records": [record.to_dict() for record in oracle.records],
            },
        )
    query_payload = [record.to_dict() for record in oracle.records]
    query_trace_sha256 = hashlib.sha256(
        json.dumps(query_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    decisions = [
        _lookup_decision(
            compiled_policy=compiled_policy,
            table=table,
            mask=np.asarray(record.mask, dtype=np.uint8),
            value=record.value,
            query_index=record.call_index,
        )
        for record in oracle.records
    ]
    if not all(decision["admitted_for_cached_lookup"] for decision in decisions):
        raise RuntimeError(f"degenerate k={k} had a refused cached lookup")
    row = {
        "arm": f"structural_control_k{k}",
        "k": k,
        "function_eval_budget": EVAL_BUDGET,
        "function_evals": oracle.calls,
        "best_mask": mask.tolist(),
        "best_s": table.value(mask),
        "budget_padding_calls": EVAL_BUDGET - 1,
        "estimator_evidence": False,
        "reason": "fixed-cardinality support contains exactly one state",
        "query_trace_sha256": query_trace_sha256,
        "query_records": query_payload,
        "lookup_decisions_rederived_after_completion": decisions,
        "complete": True,
        "review_status": "recovery-written-UNREVIEWED",
    }
    _atomic_json(out / f"k{k}_stage_receipt.json", row)
    return row


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validated_output_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(RESULTS_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("SFESS receipts must stay under experiments/results, never /tmp") from error
    if not OUTPUT_NAME_PATTERN.fullmatch(resolved.name):
        raise RuntimeError(
            "SFESS result directory must be named sfess_cached_replay_ugc64_<YYYYMMDDTHHMMSSZ>"
        )
    return resolved


def _initialize_or_resume_output(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.output_dir is not None:
        out = _validated_output_path(args.output_dir)
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("--output-dir must be absent or empty; use --resume-from for an existing run")
        out.mkdir(parents=True, exist_ok=True)
        source_custody = _execution_source_custody()
        run = {
            "schema": RUN_SCHEMA,
            "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repo": str(REPO),
            "git_head_at_start": _git_head(),
            "table_sha256": PINNED_UGC64_SHA256,
            "receipt_sha256": SOURCE_RECEIPT_SHA256,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "seed": SEED,
            "k_values": list(K_VALUES),
            "samples_per_gradient": SAMPLES_PER_GRADIENT,
            "eval_budget_per_arm": EVAL_BUDGET,
            "command_argv_at_start": list(sys.argv),
            "environment_at_start": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "PYTHONHASHSEED",
                    "PYTHONPATH",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "execution_source_custody": source_custody,
            "execution_source_git_state_at_start": _execution_source_git_state(source_custody),
        }
        _atomic_json(out / "run_manifest.json", run)
        return out, run
    out = _validated_output_path(args.resume_from)
    run_path = out / "run_manifest.json"
    if not run_path.is_file():
        raise RuntimeError("--resume-from must name an existing SFESS run directory")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected = {
        "schema": RUN_SCHEMA,
        "table_sha256": PINNED_UGC64_SHA256,
        "receipt_sha256": SOURCE_RECEIPT_SHA256,
        "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
        "seed": SEED,
        "k_values": list(K_VALUES),
        "samples_per_gradient": SAMPLES_PER_GRADIENT,
        "eval_budget_per_arm": EVAL_BUDGET,
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise RuntimeError(f"resume manifest {key} drift")
    current_source_custody = _execution_source_custody()
    if run.get("execution_source_custody") != current_source_custody:
        raise RuntimeError("resume execution source custody drift")
    if not isinstance(run.get("execution_source_git_state_at_start"), dict):
        raise RuntimeError("resume manifest lacks execution source Git-state custody")
    if not isinstance(run.get("command_argv_at_start"), list):
        raise RuntimeError("resume manifest lacks original command argv")
    if not isinstance(run.get("environment_at_start"), dict):
        raise RuntimeError("resume manifest lacks original environment custody")
    return out, run


def _build_receipt(
    *,
    run: dict[str, Any],
    policy: SFESSCachedReplayPolicy,
    table: Any,
    baselines: dict[str, dict[str, Any]],
    sfess_rows: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_rows = [
        {
            "arm": name,
            "function_eval_budget": int(row["function_eval_budget_search"]),
            "function_evals": int(row["function_evals_search"]),
            "best_mask": row["final_mask"],
            "best_s": float(row["best_s"]),
            "source": "inherited measured UGC terminal-polish receipt",
            "review_status": "fresh-eyes-reviewed(3)",
        }
        for name, row in sorted(baselines.items())
    ]
    exact_enumeration_s = float(baselines["exact_enumeration"]["best_s"])
    table_minimum = min(
        table.value(tuple((index >> bit) & 1 for bit in range(N_BITS)))
        for index in range(1 << N_BITS)
    )
    if abs(exact_enumeration_s - table_minimum) > NOISE_FLOOR_S:
        raise RuntimeError("exact-enumeration receipt no longer equals the sealed table minimum")
    best_sfess = min(sfess_rows, key=lambda row: float(row["best_s"]))
    best_baseline_s = min(float(row["best_s"]) for row in baseline_rows)
    beats_all = float(best_sfess["best_s"]) < best_baseline_s - NOISE_FLOOR_S
    if beats_all:
        verdict = "GO"
        verdict_reason = f"{best_sfess['arm']} strictly beat every registered baseline"
    else:
        verdict = "NO-GO"
        verdict_reason = "exact enumeration and (1+1)-ES remain lower; k=6 is a one-state control"
    ranking_rows = baseline_rows + [
        {
            "arm": row["arm"],
            "function_eval_budget": row["function_eval_budget"],
            "function_evals": row["function_evals"],
            "best_mask": row["best_mask"],
            "best_s": row["best_s"],
            "source": "MEASURED cached SFESS replay",
            "review_status": row["review_status"],
        }
        for row in sfess_rows
    ]
    ranking = sorted(ranking_rows, key=lambda row: (float(row["best_s"]), row["arm"]))
    return {
        "schema": RECEIPT_SCHEMA,
        # Completion is deterministic for a given durable run.  Resuming a sealed
        # run must reproduce identical receipt bytes rather than minting a new time.
        "generated_at_utc": run["created_at_utc"],
        "lane_id": "lane_sfess_cached_replay_ugc64_20260712",
        "task_id": "sfess_cached_replay_ugc64_20260712",
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "verdict_scope": {
            "scope_level": "instance/formulation",
            "objective_table_sha256": table.source_sha256,
            "objective_states": table.state_count,
            "n_bits": table.n_bits,
            "seed": SEED,
            "function_eval_budget_per_arm": EVAL_BUDGET,
            "retention_rule": "strict_gated_returned_state",
            "single_seed_across_seed_variance": "UNKNOWN",
        },
        "review_status": "recovery-written-UNREVIEWED",
        "same_budget_ranking_changed": beats_all,
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "pointer_moved": False,
        "axis": AXIS,
        "paid_dispatch": False,
        "cloud_dispatch": False,
        "live_trainer_imported_or_mutated": False,
        "protected_v9_run_read_or_mutated": False,
        "scorer_calls": 0,
        "archive_repack_calls": 0,
        "source_video": {
            "path": str(SOURCE_VIDEO.relative_to(REPO)),
            "sha256": SOURCE_VIDEO_SHA256,
            "bytes": SOURCE_VIDEO_BYTES,
        },
        "input_custody": {
            "objective_table": {
                "path": str(TABLE.relative_to(REPO)),
                "sha256": table.source_sha256,
                "bytes": TABLE.stat().st_size,
                "order_sha256": table.order_sha256,
                "objective_sha256": table.objective_sha256,
            },
            "measurement_receipt": {
                "path": str(SOURCE_RECEIPT.relative_to(REPO)),
                "sha256": SOURCE_RECEIPT_SHA256,
                "bytes": SOURCE_RECEIPT.stat().st_size,
            },
            "candidate_manifest": {
                "path": str(CANDIDATE_MANIFEST.relative_to(REPO)),
                "sha256": CANDIDATE_MANIFEST_SHA256,
                "bytes": CANDIDATE_MANIFEST.stat().st_size,
            },
            "fixture_archive_sha256": FIXTURE_ARCHIVE_SHA256,
            "fixture_authority_sha256": FIXTURE_AUTHORITY_SHA256,
            "gt_cache_sha256": GT_CACHE_SHA256,
        },
        "policy": policy.model_dump(mode="json"),
        "policy_fingerprint": hashlib.sha256(
            json.dumps(
                policy.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "control_laws": {
            "cardinality_ladder": list(K_VALUES),
            "degenerate_controls": [0, N_BITS],
            "conditional_probabilities": "constant p_i=k/6",
            "samples_per_gradient": SAMPLES_PER_GRADIENT,
            "proposal": "remove selected argmax current gradient; add unselected argmin current gradient",
            "acceptance": "strict improvement larger than the registered comparison floor",
            "completion": "exactly 64 counted cached lookups per arm; residual calls are padding",
            "comparison_noise_floor_s": NOISE_FLOOR_S,
            "failure": "refuse; live gradient fallback remains full_teacher",
        },
        "comparison_contract": {
            "sample_values_retained_as_candidates": False,
            "posthoc_best_k_selection": "exploratory only; not a 64-call mixture policy",
            "baseline_budget_contract": "B includes every objective call, including references and padding",
        },
        "baseline_rows": baseline_rows,
        "sfess_rows": sfess_rows,
        "structural_controls": controls,
        "same_budget_ranking": ranking,
        "best_non_degenerate_sfess_arm": best_sfess["arm"],
        "best_non_degenerate_sfess_s": best_sfess["best_s"],
        "exact_enumeration_s": exact_enumeration_s,
        "delta_best_sfess_minus_exact_enumeration_s": float(best_sfess["best_s"])
        - exact_enumeration_s,
        "containment": {
            "fresh_process_forbidden_prefixes": list(FORBIDDEN_MODULE_PREFIXES),
            "forbidden_modules_loaded": _forbidden_imports(),
            "objective_access": "SHA-pinned JSONL lookups only",
            "source_video_access": "direct SHA-256 and byte-count custody only; no frame decode",
        },
        "runtime": {
            "git_head_at_start": run["git_head_at_start"],
            "execution_source_custody": run["execution_source_custody"],
            "execution_source_git_state_at_start": run[
                "execution_source_git_state_at_start"
            ],
            "command_argv_at_start": run["command_argv_at_start"],
            "environment_at_start": run["environment_at_start"],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pydantic": version("pydantic"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "citations": [
            {
                "authors": "Klas Wijk; Ricardo Vinuesa; Hossein Azizpour",
                "year": 2024,
                "title": "Revisiting Score Function Estimators for k-Subset Sampling",
                "id": "arXiv:2407.16058",
                "resolved_url": "https://arxiv.org/abs/2407.16058",
            },
            {
                "authors": "Manuel Fernandez; Stuart Williams",
                "year": 2010,
                "title": "Closed-Form Expression for the Poisson-Binomial Probability Density Function",
                "id": "DOI:10.1109/TAES.2010.5461658",
                "resolved_url": "https://doi.org/10.1109/TAES.2010.5461658",
            },
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output-dir", type=Path, help="new durable result directory")
    target.add_argument("--resume-from", type=Path, help="existing durable result directory")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _require_zero_forbidden_imports("startup")
    out, run = _initialize_or_resume_output(args)
    _source, baselines = _load_and_verify_source_receipt()
    policy = _build_policy()
    compiled = policy.compile()
    table = load_cached_objective_jsonl(TABLE)
    if table.source_sha256 != policy.objective_context.objective_table_sha256:
        raise RuntimeError("loaded table/policy fingerprint mismatch")

    controls = [
        _run_degenerate_control(k=k, out=out, table=table, compiled_policy=compiled)
        for k in (0, N_BITS)
    ]
    sfess_rows = [
        _run_sfess_k(k=k, out=out, table=table, compiled_policy=compiled) for k in K_VALUES
    ]
    _require_zero_forbidden_imports("after replay")
    receipt = _build_receipt(
        run=run,
        policy=policy,
        table=table,
        baselines=baselines,
        sfess_rows=sfess_rows,
        controls=controls,
    )
    if receipt["containment"]["forbidden_modules_loaded"]:
        raise RuntimeError("containment proof failed before receipt write")
    receipt_path = out / "measurement_receipt.json"
    _atomic_json(receipt_path, receipt)
    receipt_sha256 = _sha256(receipt_path)
    _atomic_json(
        out / "receipt_manifest.json",
        {
            "measurement_receipt": receipt_path.name,
            "measurement_receipt_sha256": receipt_sha256,
            "measurement_receipt_bytes": receipt_path.stat().st_size,
            "complete": True,
        },
    )
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha256,
                "verdict": receipt["verdict"],
                "best_non_degenerate_sfess_arm": receipt["best_non_degenerate_sfess_arm"],
                "best_non_degenerate_sfess_s": receipt["best_non_degenerate_sfess_s"],
                "exact_enumeration_s": receipt["exact_enumeration_s"],
                "scorer_calls": receipt["scorer_calls"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
