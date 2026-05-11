#!/usr/bin/env python3
"""Build generated-schema HNeRV architecture-shrink training artifacts.

This is the concrete handoff between rate-side planning and actual training.
It turns a PR101/PR106-style checkpoint into:

- a generated HNeRV schema for a smaller width,
- a deterministic overlap-initialized state_dict that matches that schema,
- a manifest with exact bytes/SHA fields and non-promotable blockers,
- the next training/export commands for the operator.

No archive is produced, no scorer is loaded, and no score is claimed. A shrunk
checkpoint is not dispatchable until a matching runtime loader/export packet and
exact CUDA auth eval exist.
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_arch_schema import (  # noqa: E402
    HNeRVArchConfig,
    compare_schema_shapes,
    generate_hnerv_state_schema,
    initialize_state_dict_by_overlap,
    schema_fingerprint,
    schema_numel,
    schema_to_jsonable,
    select_base_channels_for_element_retention,
    state_dict_schema_rows,
)
from tac.repo_io import json_text, repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/build_hnerv_arch_shrink_driver.py"
SCHEMA_VERSION = "hnerv_arch_shrink_training_driver.v1"
DEFAULT_OUTPUT_DIR = Path("experiments/results/hnerv_arch_shrink_training_driver_20260507_codex")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_rel(path: Path) -> str:
    return repo_relative(path, REPO_ROOT)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"expected state_dict dict, got {type(state).__name__}")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"bad state_dict entry {key!r}: {type(value).__name__}")
    return state


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _select_config(args: argparse.Namespace) -> HNeRVArchConfig:
    if args.base_channels is not None:
        return HNeRVArchConfig(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            eval_size=(args.eval_height, args.eval_width),
        )
    return select_base_channels_for_element_retention(
        element_retention=args.element_retention,
        latent_dim=args.latent_dim,
        baseline_base_channels=args.baseline_base_channels,
        eval_size=(args.eval_height, args.eval_width),
        floor=not args.nearest_not_floor,
    )


def _operator_commands(
    *,
    output_dir: Path,
    state_dict_path: Path,
    targets_json: Path | None,
    run_label: str,
) -> list[str]:
    pieces = [
        ".venv/bin/python tools/run_deltaepszeta_training.py",
        f"--state-dict {_repo_rel(state_dict_path)}",
        "--n-epochs 1",
        "--steps-per-epoch 2",
        f"--log-dir {_repo_rel(output_dir / 'deltaepszeta_cpu_sanity')}",
        f"--run-label {run_label}",
    ]
    if targets_json is not None:
        pieces.insert(3, f"--targets-json {_repo_rel(targets_json)}")
    return [
        " ".join(pieces),
        ".venv/bin/python tools/claim_lane_dispatch.py claim "
        "--lane-id hnerv_arch_shrink_generated_schema_gpu --status planned "
        "--notes 'blocked until runtime export packet and local inflate parity exist'",
    ]


def build_driver_artifacts(
    *,
    source_state_dict_path: Path,
    output_dir: Path,
    scenario_name: str,
    target_config: HNeRVArchConfig,
    targets_json: Path | None = None,
    started_at_utc: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    source_state_dict_path = (
        source_state_dict_path
        if source_state_dict_path.is_absolute()
        else REPO_ROOT / source_state_dict_path
    )
    output_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"output_dir is non-empty: {output_dir}; pass --force to overwrite owned artifacts"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_state = _load_state_dict(source_state_dict_path)
    target_schema = generate_hnerv_state_schema(target_config)
    target_state = initialize_state_dict_by_overlap(
        source_state,
        target_schema=target_schema,
    )
    actual_schema = state_dict_schema_rows(target_state)
    schema_mismatches = compare_schema_shapes(target_schema, actual_schema)
    if schema_mismatches:
        raise RuntimeError(f"generated state_dict schema mismatch: {schema_mismatches}")

    target_state_path = output_dir / "initial_state_dict.pt"
    torch.save(target_state, target_state_path)

    generated_schema_path = output_dir / "generated_schema.json"
    schema_payload = {
        "schema": "hnerv_generated_state_schema.v1",
        "tool": TOOL_NAME,
        "started_at_utc": started_at_utc or _utc_iso(),
        "scenario_name": scenario_name,
        "target_config": target_config.to_jsonable(),
        "state_schema": schema_to_jsonable(target_schema),
        "n_state_elements": schema_numel(target_schema),
        "schema_fingerprint": schema_fingerprint(target_schema),
    }
    _save_json(generated_schema_path, schema_payload)

    targets_path: Path | None = None
    if targets_json is not None:
        targets_path = targets_json if targets_json.is_absolute() else REPO_ROOT / targets_json
        if not targets_path.is_file():
            raise FileNotFoundError(f"targets_json not found: {targets_path}")

    manifest_path = output_dir / "training_driver_manifest.json"
    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "started_at_utc": started_at_utc or _utc_iso(),
        "scenario_name": scenario_name,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "empirical-generated-checkpoint-no-score",
        "evidence_semantics": (
            "generated_schema_and_overlap_initialized_checkpoint_only_"
            "no_runtime_packet_no_cuda_eval"
        ),
        "inputs": {
            "source_state_dict_path": _repo_rel(source_state_dict_path),
            "source_state_dict_sha256": sha256_file(source_state_dict_path),
            "targets_json": _repo_rel(targets_path) if targets_path else None,
            "targets_json_sha256": sha256_file(targets_path) if targets_path else None,
        },
        "outputs": {
            "output_dir": _repo_rel(output_dir),
            "generated_schema_path": _repo_rel(generated_schema_path),
            "generated_schema_sha256": sha256_file(generated_schema_path),
            "initial_state_dict_path": _repo_rel(target_state_path),
            "initial_state_dict_bytes": target_state_path.stat().st_size,
            "initial_state_dict_sha256": sha256_file(target_state_path),
        },
        "target_config": target_config.to_jsonable(),
        "source_schema": {
            "n_tensors": len(source_state),
            "n_state_elements": schema_numel(state_dict_schema_rows(source_state)),
            "fingerprint": schema_fingerprint(state_dict_schema_rows(source_state)),
        },
        "overlap_initialization": {
            "policy": "copy_prefix_slices_zero_fill_missing_or_new_channels",
            "deterministic": True,
            "missing_source_tensors": [
                name for name, _shape in target_schema if name not in source_state
            ],
        },
        "operator_next_commands": _operator_commands(
            output_dir=output_dir,
            state_dict_path=target_state_path,
            targets_json=targets_path,
            run_label=scenario_name,
        ),
        "dispatch_blockers": [
            "generated_checkpoint_has_no_trained_distortion_evidence",
            "generated_schema_runtime_loader_not_integrated_into_submission_packet",
            "target_codec_export_for_generated_schema_not_implemented",
            "local_inflate_output_parity_missing",
            "strict_pre_submission_compliance_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }
    _save_json(manifest_path, manifest)
    manifest["outputs"]["training_driver_manifest_path"] = _repo_rel(manifest_path)
    manifest["outputs"]["training_driver_manifest_sha256"] = sha256_file(manifest_path)
    _save_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="build HNeRV generated-schema architecture-shrink training artifacts"
    )
    parser.add_argument("--source-state-dict", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scenario-name", default="hnerv_arch_shrink_generated_schema")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base-channels", type=int)
    group.add_argument("--element-retention", type=float)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument("--baseline-base-channels", type=int, default=36)
    parser.add_argument("--eval-height", type=int, default=384)
    parser.add_argument("--eval-width", type=int, default=512)
    parser.add_argument("--nearest-not-floor", action="store_true")
    parser.add_argument("--targets-json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    manifest = build_driver_artifacts(
        source_state_dict_path=Path(args.source_state_dict),
        output_dir=Path(args.output_dir),
        scenario_name=args.scenario_name,
        target_config=_select_config(args),
        targets_json=Path(args.targets_json) if args.targets_json else None,
        force=args.force,
    )
    sys.stdout.write(
        json_text(
            {
                "manifest": manifest["outputs"]["training_driver_manifest_path"],
                "initial_state_dict": manifest["outputs"]["initial_state_dict_path"],
                "generated_schema": manifest["outputs"]["generated_schema_path"],
                "ready_for_exact_eval_dispatch": manifest["ready_for_exact_eval_dispatch"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
