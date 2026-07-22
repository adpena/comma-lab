#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the DDM G3 full-n600 flip/margin score atlas from settled caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_g3_score_atlas import (  # noqa: E402
    AXIS,
    POINTER,
    PRIMARY_RANK_KEY,
    DdmG3ScoreAtlasConfigV1,
    ScoreAtlasError,
    atomic_write,
    audit_evaluator_response_cone_maps,
    build_admission_efficiency_rows,
    build_hard_pair_registry,
    build_pair_rows,
    build_summary,
    load_evaluator_response_rows,
    measure_subset_correlations,
    reconstruct_v12_state,
    select_stratified_control,
    sha256_file,
    storage_preflight,
    write_charts,
    write_json,
    write_jsonl,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _checkpoint(root: Path, number: int, stage: str, payload: dict) -> None:
    write_json(
        root / "stage_checkpoints" / f"{number:02d}_{stage}.json",
        {
            "schema": "ddm_g3_score_atlas_stage_checkpoint.v1",
            "stage": stage,
            "written_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **payload,
        },
    )


def _output_row(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _validate_resume(receipt_path: Path, typed_hash: str) -> dict | None:
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("typed_config_sha256") != typed_hash:
        raise ScoreAtlasError("completed receipt typed-config SHA differs")
    implementation = receipt.get("implementation_custody")
    if not isinstance(implementation, dict):
        return None
    for row in implementation.get("source_files", []):
        path = _resolve(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ScoreAtlasError(f"implementation source missing or size changed: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ScoreAtlasError(f"implementation source SHA changed: {path}")
    for row in [*receipt.get("outputs", []), *receipt.get("compact_outputs", [])]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ScoreAtlasError(f"completed output missing or size changed: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ScoreAtlasError(f"completed output SHA changed: {path}")
    output = _resolve(receipt["typed_config"]["output_directory"])
    expected_stages = (
        "00_inputs_validated",
        "01_v12_state_reconstructed",
        "02_typed_rows_complete",
        "03_hard_pair_registry_complete",
        "04_charts_and_cleanup_complete",
        "05_receipt_complete",
    )
    for stage_name in expected_stages:
        path = output / "stage_checkpoints" / f"{stage_name}.json"
        if not path.is_file() or json.loads(path.read_text()).get("status") != "complete":
            raise ScoreAtlasError(f"completed stage checkpoint missing or incomplete: {path}")
    return receipt


def _source_custody(config: DdmG3ScoreAtlasConfigV1, receipt: dict) -> dict:
    target = receipt["target_custody"]
    scorer = receipt["scorer_custody"]
    return {
        "evaluator_response_atlas_path": str(_resolve(config.evaluator_atlas_path)),
        "evaluator_response_atlas_sha256": config.evaluator_atlas_sha256,
        "v12_receipt_path": str(_resolve(config.v12_receipt_path)),
        "v12_receipt_sha256": config.v12_receipt_sha256,
        "measured_admission_receipts": [item.model_dump(mode="json") for item in config.admission_receipts],
        "gt_cache_path": target["cache_path"],
        "gt_cache_bytes": target["cache_bytes"],
        "gt_cache_sha256_from_bound_target_receipt": target["cache_sha256"],
        "segnet_weights_sha256": scorer["segnet"]["weights_sha256"],
        "posenet_weights_sha256": scorer["posenet"]["weights_sha256"],
        "scorer_batch_size": scorer["segnet"]["batch_size"],
        "cache_chunk_size": config.cache_chunk_size,
        "oom_law": "canonical scorer batches <=16 only; never materialize a full-n600 scorer forward tensor",
        "scorer_outputs_recomputed": False,
        "reuse_reason": "ALREADY_SETTLED v12 SHA-bound frozen-scorer canonical-batch caches",
    }


def _load_bound_json(path_value: str, expected_sha256: str, label: str) -> dict:
    path = _resolve(path_value)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ScoreAtlasError(f"{label} SHA mismatch: expected {expected_sha256}, got {actual}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ScoreAtlasError(f"{label} must be a JSON object")
    return value


def _v13_pointer_evidence(config: DdmG3ScoreAtlasConfigV1) -> dict:
    path = _resolve(config.v13_pointer_ledger_path)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != config.v13_pointer_ledger_sha256:
        raise ScoreAtlasError(
            f"v13 pointer ledger SHA mismatch: expected {config.v13_pointer_ledger_sha256}, got {actual}"
        )
    matches = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        row = json.loads(line)
        if row.get("p0_id") == config.v13_pointer_p0_id and "ddm_v13_worldsheet_event_predictor" in str(
            row.get("next_action", "")
        ):
            matches.append((line_number, line, row))
    if len(matches) != 1:
        raise ScoreAtlasError(f"expected one canonical v13 pointer row, found {len(matches)}")
    line_number, line, row = matches[0]
    return {
        "ledger_path": config.v13_pointer_ledger_path,
        "resolved_path_at_measurement": str(path),
        "ledger_sha256": actual,
        "line_number": line_number,
        "row_sha256": hashlib.sha256(line).hexdigest(),
        "p0_id": row["p0_id"],
        "status": row["status"],
        "next_action": row["next_action"],
        "last_verified_utc": row["last_verified_utc"],
    }


def _hard_subsets(rows: list[dict]) -> dict[str, list[int]]:
    ranked = sorted(rows, key=lambda row: int(row["score_rank"]))
    top24 = [int(row["pair_index"]) for row in ranked[:24]]
    top64 = [int(row["pair_index"]) for row in ranked[:64]]
    control = select_stratified_control(rows, excluded=set(top64), k=24)
    return {"top24": top24, "top64": top64, "stratified_control24": control}


def _cleanup_manifest(paths: list[Path], argv: list[str]) -> dict:
    return {
        "schema": "certified_rebuildable_artifact_manifest.v1",
        "policy": "certify-or-block",
        "artifacts": [
            {
                **_output_row(path),
                "rebuildable": True,
                "delete_authorized": False,
                "reason": "deterministically rebuilt from SHA-bound frozen-scorer and evaluator-atlas inputs",
            }
            for path in paths
        ],
        "semantic_argv": argv,
        "temporary_directory_policy": "context-managed chart stage; success moves outputs atomically",
        "false_authority": {"evidence_axis": AXIS, "score_claim": False, "promotion_eligible": False},
    }


def _implementation_custody(config: DdmG3ScoreAtlasConfigV1) -> dict:
    sources = [
        Path("src/tac/optimization/ddm_g3_score_atlas.py"),
        Path("tools/build_ddm_g3_score_atlas.py"),
        Path(".omx/research/configs/ddm_g3_score_atlas_n600_20260722.json"),
    ]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_head_at_build": git_head,
        "typed_config_sha256": config.typed_hash(),
        "source_files": [
            {"path": str(path), "sha256": sha256_file(REPO / path), "bytes": (REPO / path).stat().st_size}
            for path in sources
        ],
    }


def run(config: DdmG3ScoreAtlasConfigV1, *, resume: bool, argv: list[str]) -> Path:
    output = _resolve(config.output_directory)
    compact = _resolve(config.compact_receipt_directory)
    receipt_path = compact / "ddm_g3_score_atlas_receipt.json"
    if resume:
        completed = _validate_resume(receipt_path, config.typed_hash())
        if completed is not None:
            print(json.dumps({"resumed": True, "receipt": str(receipt_path), "verdict": completed["verdict"]}))
            return receipt_path
    preflight = storage_preflight(output)
    compact.mkdir(parents=True, exist_ok=True)

    v12_receipt = _load_bound_json(config.v12_receipt_path, config.v12_receipt_sha256, "v12 receipt")
    admission_sources = [
        (
            item,
            _load_bound_json(item.path, item.sha256, f"{item.version} measured-admission receipt"),
        )
        for item in config.admission_receipts
    ]
    v13_evidence = _v13_pointer_evidence(config)
    source_custody = _source_custody(config, v12_receipt)
    gt_cache = Path(source_custody["gt_cache_path"])
    if not gt_cache.is_file() or gt_cache.stat().st_size != int(source_custody["gt_cache_bytes"]):
        raise ScoreAtlasError("settled GT cache missing or byte count changed")
    evaluator_header, evaluator_rows = load_evaluator_response_rows(
        _resolve(config.evaluator_atlas_path), config.evaluator_atlas_sha256, n_pairs=config.n_pairs
    )
    cone_map_audit = audit_evaluator_response_cone_maps(evaluator_rows)
    source_custody["evaluator_response_cone_map_audit"] = cone_map_audit
    _checkpoint(
        output,
        0,
        "inputs_validated",
        {
            "typed_config_sha256": config.typed_hash(),
            "source_custody": source_custody,
            "storage_preflight": preflight,
            "evaluator_header": evaluator_header,
            "evaluator_response_cone_map_audit": cone_map_audit,
            "v13_pointer_evidence": v13_evidence,
            "status": "complete",
        },
    )

    state = reconstruct_v12_state(REPO, v12_receipt, n_pairs=config.n_pairs)
    _checkpoint(
        output,
        1,
        "v12_state_reconstructed",
        {
            "final_errors": state.final_errors,
            "final_pose_squared_error": state.final_pose_squared_error,
            "archive_bytes": state.archive_bytes,
            "archive_sha256": state.archive_sha256,
            "admission_rows": len(state.admission_rows),
            "status": "complete",
        },
    )

    target_cells = open_stored_npy_memmap(gt_cache, "lstars")
    target_margins = open_stored_npy_memmap(gt_cache, "margins")
    target_poses = open_stored_npy_memmap(gt_cache, "gt_poses")
    rows = build_pair_rows(
        target_cells=target_cells,
        target_margins=target_margins,
        target_poses=target_poses,
        state=state,
        evaluator_rows=evaluator_rows,
        source_custody=source_custody,
    )
    rows_path = output / "ddm_g3_score_atlas_n600.jsonl"
    write_jsonl(rows_path, rows)
    _checkpoint(
        output,
        2,
        "typed_rows_complete",
        {
            "row_count": len(rows),
            "rows": _output_row(rows_path),
            "primary_rank_key": PRIMARY_RANK_KEY,
            "energy_or_l2_rank_allowed": False,
            "status": "complete",
        },
    )

    subsets = _hard_subsets(rows)
    correlations = measure_subset_correlations(REPO, state, target_cells, target_poses, subsets)
    admissions = [
        row
        for item, admission_receipt in admission_sources
        for row in build_admission_efficiency_rows(
            admission_receipt,
            source_version=item.version,
            n_pairs=item.pair_count,
        )
    ]
    admissions_path = output / "measured_admission_efficiency.jsonl"
    write_jsonl(admissions_path, admissions)
    summary = build_summary(rows, state)
    summary["hard_subsets"] = subsets
    summary["hard_subset_correlations"] = correlations
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    hard_registry = build_hard_pair_registry(
        rows,
        correlations,
        atlas_jsonl_path=rows_path,
        atlas_jsonl_sha256=sha256_file(rows_path),
        source_custody=source_custody,
    )
    hard_path = output / "hard_pair_registry.json"
    write_json(hard_path, hard_registry)
    _checkpoint(
        output,
        3,
        "hard_pair_registry_complete",
        {
            "registry": _output_row(hard_path),
            "correlations": correlations,
            "status": "complete",
        },
    )

    charts = write_charts(output, rows, admissions, summary)
    pre_cleanup_paths = [rows_path, admissions_path, summary_path, hard_path, *charts]
    cleanup = _cleanup_manifest(pre_cleanup_paths, argv)
    cleanup_path = output / "cleanup_manifest.json"
    write_json(cleanup_path, cleanup)
    _checkpoint(
        output,
        4,
        "charts_and_cleanup_complete",
        {
            "charts": [_output_row(path) for path in charts],
            "cleanup_manifest": _output_row(cleanup_path),
            "status": "complete",
        },
    )

    compact_summary = compact / "summary.json"
    compact_hard = compact / "hard_pair_registry.json"
    atomic_write(compact_summary, summary_path.read_bytes())
    atomic_write(compact_hard, hard_path.read_bytes())
    outputs = [
        _output_row(path) for path in [rows_path, admissions_path, summary_path, hard_path, cleanup_path, *charts]
    ]
    pointer_row = {
        "schema": "ddm_g3_score_atlas_live_arm_pointer.v1",
        "source_arm": config.run_id,
        "consumer_arm": "ddm_v13_worldsheet_event_predictor",
        "consumer_status": "SPAWNED_OPERATOR_P0_LEDGER_IN_PROGRESS",
        "canonical_pointer_evidence": v13_evidence,
        "atlas_jsonl": _output_row(rows_path),
        "hard_pair_registry": _output_row(hard_path),
        "costate_consumer_contract": "read each ddm_g3_score_atlas_pair.v1 costate_signal object",
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
    }
    pointer_path = compact / "live_v13_pointer.json"
    write_json(pointer_path, pointer_row)
    receipt = {
        "schema": "ddm_g3_score_atlas_receipt.v1",
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_hash(),
        "semantic_argv": argv,
        "source_custody": source_custody,
        "implementation_custody": _implementation_custody(config),
        "storage_preflight": preflight,
        "outputs": outputs,
        "compact_outputs": [_output_row(path) for path in [compact_summary, compact_hard, pointer_path]],
        "summary": summary,
        "verdict": "MEASURED_ADVISORY_N600_SCORE_ATLAS_COMPLETE_V13_POINTER_BOUND",
        "verdict_scope": (
            "v12 frozen-scorer cached cells/poses joined to #36 response geometry; local advisory only; "
            "hard-subset correlations are historical replay over v12 measured proposals"
        ),
        "blocker_delta_vs_603": (
            "#603 per-pair score debt, Fisher/margin/topology cube, Pose sensitivity, byte amortization, "
            "scene covariates, and hard-subset replay are indexed; the canonical spawned v13 arm is "
            "SHA-bound. Live v13 consumption and contemporaneous candidate subset/full replay remain owed."
        ),
        "round1_self_review_required": True,
        "main_landing_review_required": True,
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            config.evaluator_atlas_path,
            config.v12_receipt_path,
            *[item.path for item in config.admission_receipts],
            config.v13_pointer_ledger_path,
            ".omx/research/ddm_g2_solve_diff_op_mining_n600_20260722T194000Z/aggregate_ledger.json",
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/canonical_task_status.jsonl",
        ],
        "pointer": POINTER,
        "pointer_moved": False,
        "research_only": True,
        "execution_allowed": False,
        "evidence_axis": AXIS,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
    }
    write_json(receipt_path, receipt)
    _checkpoint(output, 5, "receipt_complete", {"receipt": _output_row(receipt_path), "status": "complete"})
    print(json.dumps({"resumed": False, "receipt": str(receipt_path), "verdict": receipt["verdict"]}))
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = _resolve(args.config)
    try:
        config = DdmG3ScoreAtlasConfigV1.model_validate(json.loads(config_path.read_text()))
        semantic_argv = [
            "tools/build_ddm_g3_score_atlas.py",
            "--config",
            str(args.config),
            "--resume",
        ]
        run(config, resume=args.resume, argv=semantic_argv)
    except (OSError, ValueError, KeyError, ScoreAtlasError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
