#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan strict MLX parent-contract closure after a Modal tensor export.

The output is a command plan only. It does not create score, promotion, or
dispatch authority; every generated step preserves the existing MLX false-
authority contract and routes through the strict production-contract gate.
"""

from __future__ import annotations

import argparse
import glob
import json
import shlex
from pathlib import Path
from typing import Any

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

TENSOR_VOLUME_FILES = (
    "manifest.json",
    "pair_indices.npy",
    "posenet_yuv6_pair.npy",
    "segnet_last_rgb.npy",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-eval-dir", required=True, type=Path)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help=(
            "Legacy/input dataset path. By default this planner rebuilds a fresh "
            "same-axis dataset from recovered candidate windows plus "
            "--baseline-window-response inputs and uses the rebuilt dataset for "
            "bundle/refresh gates."
        ),
    )
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--candidate-family", default="mlx_decoder_q")
    parser.add_argument("--candidate-label", default="decoderq")
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--existing-contract", action="append", default=[], type=Path)
    parser.add_argument("--existing-cache-auth-audit", action="append", default=[], type=Path)
    parser.add_argument("--baseline-calibration-label", default="fec6_auth_parent")
    parser.add_argument("--baseline-mlx-response", type=Path)
    parser.add_argument("--baseline-cpu-auth-eval", type=Path)
    parser.add_argument(
        "--baseline-window-response",
        action="append",
        default=[],
        help=(
            "Per-window baseline MLX scorer-response JSON path or glob used to "
            "rebuild the same-axis dataset. May repeat."
        ),
    )
    parser.add_argument(
        "--skip-dataset-rebuild",
        action="store_true",
        help=(
            "Use --dataset directly instead of rebuilding from recovered "
            "candidate windows. Intended only for legacy planning audits."
        ),
    )
    parser.add_argument(
        "--allow-partial-full-auth-calibration",
        action="store_true",
        help=(
            "Permit planning a score-calibration row that compares a partial "
            "MLX window to a full auth-eval payload. This is research-only and "
            "must stay off for strict production-contract closure."
        ),
    )
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plan = build_plan(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(plan), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "status": plan["status"],
                "next_blocker": plan["next_blocker"],
                "step_count": len(plan["steps"]),
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    auth_dir = args.auth_eval_dir.resolve()
    out_root = args.output_root.resolve()
    run_id = args.run_id or _run_id_from_auth_dir(auth_dir)
    local_request = _read_optional_object(auth_dir / "modal_cpu_auth_eval_local_request.json")
    spawn = _read_optional_object(auth_dir / "modal_auth_eval_spawn.json")
    recover_summary = _read_optional_object(auth_dir / "modal_auth_eval_recover_summary.json")
    auth_eval_path = auth_dir / "contest_auth_eval.json"
    tensor_manifest_path = auth_dir / "scorer_input_cache_tensor_volume_manifest.json"

    archive_size = _archive_size_bytes(local_request, auth_eval_path)
    volume_run_id = str(
        local_request.get("scorer_input_cache_tensor_volume_run_id")
        or run_id
    )
    volume_name = str(
        local_request.get("scorer_input_cache_tensor_volume_name")
        or "comma-auth-eval-cache-artifacts"
    )
    volume_cache_path = f"{volume_run_id}/scorer_input_cache_tensors"
    download_root = out_root / "modal_tensor_volume_download"
    downloaded_cache_dir = download_root / "scorer_input_cache_tensors"
    downloaded_tensor_files = {
        name: downloaded_cache_dir / name for name in TENSOR_VOLUME_FILES
    }
    cache_dir = out_root / "candidate_cache"
    cache_audit = out_root / "candidate_cache_auth_audit.json"
    window_label = _window_label(args.start_pair, args.max_pairs)
    parent_response = out_root / f"candidate_parent_{window_label}.json"
    components_dir = out_root / "candidate_parent_components"
    candidate_windows_dir = out_root / "candidate_windows"
    candidate_windows_index = out_root / "candidate_windows_index.json"
    rebuilt_dataset = out_root / "candidate_same_axis_window_response_dataset.json"
    rebuilt_dataset_md = out_root / "candidate_same_axis_window_response_dataset.md"
    dataset_for_contract = args.dataset if args.skip_dataset_rebuild else rebuilt_dataset
    baseline_window_paths = _expand_existing_paths(args.baseline_window_response)
    candidate_parity = out_root / f"candidate_torch_parity_sweep_cpu_singleton_pairs{args.start_pair}_{args.max_pairs}.json"
    reference_parity = out_root / f"reference_torch_parity_sweep_cpu_singleton_pairs{args.start_pair}_{args.max_pairs}.json"
    profile = out_root / f"candidate_profile_cpu_singleton_pairs{args.start_pair}_{args.max_pairs}_repeat2.json"
    profile_stability = out_root / f"candidate_profile_stability_cpu_singleton_pairs{args.start_pair}_{args.max_pairs}_repeat2.json"
    calibration_rows = out_root / "candidate_score_calibration_rows.json"
    calibration = out_root / "candidate_score_calibration_cpu.json"
    contract = out_root / "candidate_parent_contract_strict_v1.json"
    bundle = out_root / "mlx_parent_contract_bundle_with_candidate.json"
    refreshed_plan = out_root / "parent_production_contract_plan_after_candidate.json"
    refreshed_plan_md = out_root / "parent_production_contract_plan_after_candidate.md"

    steps: list[dict[str, Any]] = []
    steps.append(
        _step(
            "recover_modal_call",
            "Recover the detached Modal CPU tensor export.",
            [
                ".venv/bin/python",
                "tools/recover_modal_auth_eval.py",
                "--output-dir",
                str(auth_dir),
            ],
            ready=True,
        )
    )
    steps.append(
        _step(
            "prepare_tensor_download_dir",
            "Create the deterministic local directory for Modal tensor files.",
            [
                "mkdir",
                "-p",
                str(downloaded_cache_dir),
            ],
            ready=auth_eval_path.is_file() and tensor_manifest_path.is_file(),
            requires=["recover_modal_call"],
        )
    )
    download_step_ids: list[str] = []
    for name, local_path in downloaded_tensor_files.items():
        step_id = f"download_tensor_{name.replace('.', '_')}"
        download_step_ids.append(step_id)
        steps.append(
            _step(
                step_id,
                f"Download Modal tensor cache file {name}.",
                [
                    ".venv/bin/modal",
                    "volume",
                    "get",
                    "--force",
                    volume_name,
                    f"{volume_cache_path}/{name}",
                    str(local_path),
                ],
                ready=downloaded_cache_dir.is_dir(),
                requires=["prepare_tensor_download_dir"],
            )
        )
    steps.append(
        _step(
            "materialize_downloaded_cache",
            "Validate downloaded tensors against auth-eval provenance and stamp a local MLX cache.",
            [
                ".venv/bin/python",
                "tools/materialize_mlx_scorer_cache_from_auth_eval.py",
                "--auth-eval-dir",
                str(auth_dir),
                "--downloaded-tensor-cache-dir",
                str(downloaded_cache_dir),
                "--tensor-volume-manifest",
                str(tensor_manifest_path),
                "--output-cache-dir",
                str(cache_dir),
                "--audit-output",
                str(cache_audit),
                "--force",
            ],
            ready=all(path.is_file() for path in downloaded_tensor_files.values())
            and auth_eval_path.is_file()
            and tensor_manifest_path.is_file(),
            requires=download_step_ids,
        )
    )
    steps.append(
        _step(
            "run_parent_response",
            "Compute the singleton CPU MLX parent response over the parent window.",
            [
                ".venv/bin/python",
                "tools/run_mlx_scorer_response_cache.py",
                "--reference-cache-dir",
                str(args.reference_cache_dir),
                "--candidate-cache-dir",
                str(cache_dir),
                "--archive-size-bytes",
                str(archive_size),
                "--batch-pairs",
                "1",
                "--device",
                "cpu",
                "--start-pair",
                str(args.start_pair),
                "--max-pairs",
                str(args.max_pairs),
                "--components-dir",
                str(components_dir),
                "--response-family",
                args.candidate_family,
                "--output",
                str(parent_response),
                "--repo-root",
                str(args.repo_root),
            ],
            ready=cache_dir.is_dir(),
            requires=["materialize_downloaded_cache"],
        )
    )
    if not args.skip_dataset_rebuild:
        steps.append(
            _step(
                "split_parent_windows",
                "Split the recovered Modal-backed parent response into singleton window rows.",
                [
                    ".venv/bin/python",
                    "tools/split_mlx_scorer_response_windows.py",
                    "--response",
                    str(parent_response),
                    "--output-dir",
                    str(candidate_windows_dir),
                    "--index-out",
                    str(candidate_windows_index),
                    "--components-dir",
                    str(components_dir),
                    "--window-pairs",
                    "1",
                    "--stride-pairs",
                    "1",
                    "--max-windows",
                    str(args.max_pairs),
                    "--prefix",
                    "candidate_pair",
                ],
                ready=parent_response.is_file(),
                requires=["run_parent_response"],
            )
        )
        dataset_command = [
            ".venv/bin/python",
            "tools/build_mlx_window_response_dataset.py",
            "--candidate-response",
            str(candidate_windows_dir / "*.json"),
        ]
        for baseline in args.baseline_window_response:
            dataset_command.extend(["--baseline-response", baseline])
        dataset_command.extend(
            [
                "--json-out",
                str(rebuilt_dataset),
                "--md-out",
                str(rebuilt_dataset_md),
            ]
        )
        steps.append(
            _step(
                "build_rebuilt_dataset",
                (
                    "Rebuild the same-axis MLX window-response dataset from "
                    "fresh decoder-q Modal windows plus FEC6 auth parent windows."
                ),
                dataset_command,
                ready=candidate_windows_index.is_file()
                and len(baseline_window_paths) >= args.max_pairs,
                requires=["split_parent_windows"],
            )
        )
    for name, cache_path, output in (
        ("candidate_torch_parity", cache_dir, candidate_parity),
        ("reference_torch_parity", args.reference_cache_dir, reference_parity),
    ):
        steps.append(
            _step(
                name,
                f"Run singleton PyTorch-vs-MLX parity sweep for {name.split('_')[0]} cache.",
                [
                    ".venv/bin/python",
                    "tools/audit_mlx_scorer_torch_parity_sweep.py",
                    "--cache-dir",
                    str(cache_path),
                    "--output",
                    str(output),
                    "--repo-root",
                    str(args.repo_root),
                    "--device",
                    "cpu",
                    "--start-pair",
                    str(args.start_pair),
                    "--max-pairs",
                    str(args.max_pairs),
                    "--window-pairs",
                    "1",
                    "--stride-pairs",
                    "1",
                    "--max-segnet-argmax-diff-pixels",
                    "1",
                    "--run-id",
                    f"{run_id}_{name}",
                ],
                ready=cache_dir.is_dir() if cache_path == cache_dir else cache_path.is_dir(),
                requires=["materialize_downloaded_cache"],
            )
        )
    steps.append(
        _step(
            "profile_parent_response",
            "Profile singleton CPU MLX parent response twice for stability gating.",
            [
                ".venv/bin/python",
                "tools/profile_mlx_scorer_response_cache.py",
                "--reference-cache-dir",
                str(args.reference_cache_dir),
                "--candidate-cache-dir",
                str(cache_dir),
                "--archive-size-bytes",
                str(archive_size),
                "--batch-pairs",
                "1",
                "--devices",
                "cpu",
                "--start-pair",
                str(args.start_pair),
                "--max-pairs",
                str(args.max_pairs),
                "--repeat",
                "2",
                "--output",
                str(profile),
                "--repo-root",
                str(args.repo_root),
            ],
            ready=cache_dir.is_dir(),
            requires=["materialize_downloaded_cache"],
        )
    )
    steps.append(
        _step(
            "check_profile_stability",
            "Convert profile rows into the strict profile-stability gate.",
            [
                ".venv/bin/python",
                "tools/check_mlx_scorer_response_profile_stability.py",
                "--profile",
                str(profile),
                "--output",
                str(profile_stability),
                "--baseline-device",
                "cpu",
                "--baseline-batch-pairs",
                "1",
                "--run-id",
                f"{run_id}_profile_stability",
            ],
            ready=profile.is_file(),
            requires=["profile_parent_response"],
        )
    )
    strict_full_calibration = args.start_pair == 0 and args.max_pairs == 600
    calibration_input = _calibration_rows_payload(
        args=args,
        parent_response=parent_response,
        auth_eval_path=auth_eval_path,
        strict_full_calibration=strict_full_calibration,
    )
    steps.append(
        {
            "id": "write_score_calibration_rows",
            "description": "Write the two-row strict CPU calibration input for FEC6 baseline plus decoder-q candidate.",
            "ready": auth_eval_path.is_file()
            and bool(args.baseline_mlx_response and args.baseline_cpu_auth_eval)
            and (strict_full_calibration or args.allow_partial_full_auth_calibration),
            "requires": ["run_parent_response", "recover_modal_call"],
            "write_json_path": str(calibration_rows),
            "write_json_payload": calibration_input,
            **FALSE_AUTHORITY,
        }
    )
    steps.append(
        _step(
            "calibrate_score_axis",
            "Build strict MLX-vs-CPU score calibration for spend-triage use.",
            [
                ".venv/bin/python",
                "tools/calibrate_mlx_scorer_response_scores.py",
                "--input",
                str(calibration_rows),
                "--output",
                str(calibration),
                "--repo-root",
                str(args.repo_root),
                "--run-id",
                f"{run_id}_score_calibration",
            ],
            ready=calibration_rows.is_file(),
            requires=["write_score_calibration_rows"],
        )
    )
    steps.append(
        _step(
            "build_strict_contract",
            "Build the strict non-authoritative MLX production contract for the decoder-q parent response.",
            [
                ".venv/bin/python",
                "tools/check_mlx_scorer_production_contract.py",
                "--response",
                str(parent_response),
                "--cache-auth-audit",
                str(cache_audit),
                "--torch-parity",
                str(candidate_parity),
                "--reference-torch-parity",
                str(reference_parity),
                "--profile-stability",
                str(profile_stability),
                "--score-calibration",
                str(calibration),
                "--require-score-calibration",
                "--output",
                str(contract),
                "--run-id",
                f"{run_id}_strict_parent_contract",
            ],
            ready=all(
                path.is_file()
                for path in (parent_response, cache_audit, candidate_parity, reference_parity, profile_stability, calibration)
            ),
            requires=[
                "run_parent_response",
                "candidate_torch_parity",
                "reference_torch_parity",
                "check_profile_stability",
                "calibrate_score_axis",
            ],
        )
    )
    bundle_command = [
        ".venv/bin/python",
        "tools/build_mlx_production_contract_bundle.py",
    ]
    for existing in args.existing_contract:
        bundle_command.extend(["--contract", str(existing)])
    bundle_command.extend(
        [
            "--contract",
            str(contract),
            "--dataset",
            str(dataset_for_contract),
            "--output",
            str(bundle),
            "--run-id",
            f"{run_id}_bundle",
        ]
    )
    bundle_requires = ["build_strict_contract"]
    if not args.skip_dataset_rebuild:
        bundle_requires.append("build_rebuilt_dataset")
    steps.append(
        _step(
            "build_contract_bundle",
            "Bundle existing strict contracts with the new decoder-q strict contract and validate dataset coverage.",
            bundle_command,
            ready=contract.is_file()
            and dataset_for_contract.is_file()
            and all(path.is_file() for path in args.existing_contract),
            requires=bundle_requires,
        )
    )
    refresh_command = [
        ".venv/bin/python",
        "tools/plan_mlx_parent_production_contracts.py",
        "--dataset",
        str(dataset_for_contract),
        "--production-contract",
        str(bundle),
        "--json-out",
        str(refreshed_plan),
        "--md-out",
        str(refreshed_plan_md),
        "--allow-blocked-output",
    ]
    for audit in args.existing_cache_auth_audit:
        refresh_command.extend(["--cache-auth-audit", str(audit)])
    refresh_command.extend(["--cache-auth-audit", str(cache_audit)])
    steps.append(
        _step(
            "refresh_parent_plan",
            "Refresh the 600-row parent-contract plan; this should move decoder-q from missing to covered if all gates pass.",
            refresh_command,
            ready=bundle.is_file(),
            requires=["build_contract_bundle"],
        )
    )

    next_blocker = _first_blocker(
        auth_eval_path,
        tensor_manifest_path,
        downloaded_cache_dir,
        cache_dir,
        parent_response,
        contract,
        rebuild_dataset=not args.skip_dataset_rebuild,
        baseline_window_response=args.baseline_window_response,
        baseline_window_count=len(baseline_window_paths),
        expected_baseline_window_count=args.max_pairs,
        candidate_windows_index=candidate_windows_index,
        dataset_for_contract=dataset_for_contract,
        downloaded_tensor_files=list(downloaded_tensor_files.values()),
        strict_full_calibration=strict_full_calibration
        or args.allow_partial_full_auth_calibration,
    )
    return {
        "schema_version": "mlx_parent_contract_closure_plan.v1",
        "status": "blocked" if next_blocker else "ready_for_refresh",
        "next_blocker": next_blocker,
        "auth_eval_dir": str(auth_dir),
        "call_id": spawn.get("call_id") or _read_text(auth_dir / "modal_call_id.txt"),
        "recover_status": recover_summary.get("status"),
        "archive_sha256": local_request.get("archive_sha256"),
        "archive_size_bytes": archive_size,
        "pair_window": [args.start_pair, args.start_pair + args.max_pairs],
        "strict_full_auth_calibration": strict_full_calibration,
        "tensor_volume_run_id": volume_run_id,
        "tensor_volume_cache_path": volume_cache_path,
        "downloaded_cache_dir": str(downloaded_cache_dir),
        "downloaded_tensor_files": {
            name: str(path) for name, path in downloaded_tensor_files.items()
        },
        "candidate_cache_dir": str(cache_dir),
        "legacy_dataset": str(args.dataset),
        "dataset_for_contract": str(dataset_for_contract),
        "dataset_rebuild_enabled": not args.skip_dataset_rebuild,
        "candidate_windows_dir": str(candidate_windows_dir),
        "candidate_windows_index": str(candidate_windows_index),
        "baseline_window_responses": list(args.baseline_window_response),
        "baseline_window_existing_count": len(baseline_window_paths),
        "expected_baseline_window_count": args.max_pairs,
        "candidate_contract": str(contract),
        "bundle": str(bundle),
        "steps": steps,
        **FALSE_AUTHORITY,
    }


def _step(
    step_id: str,
    description: str,
    command: list[str],
    *,
    ready: bool,
    requires: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": step_id,
        "description": description,
        "ready": bool(ready),
        "requires": list(requires or []),
        "command": command,
        "shell": " ".join(shlex.quote(part) for part in command),
        **FALSE_AUTHORITY,
    }


def _window_label(start_pair: int, max_pairs: int) -> str:
    return f"{start_pair:04d}_{start_pair + max_pairs:04d}"


def _expand_existing_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        for item in matches:
            path = Path(item).resolve()
            if path in seen:
                continue
            if path.is_file():
                paths.append(path)
                seen.add(path)
    return paths


def _calibration_rows_payload(
    *,
    args: argparse.Namespace,
    parent_response: Path,
    auth_eval_path: Path,
    strict_full_calibration: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if args.baseline_mlx_response is not None and args.baseline_cpu_auth_eval is not None:
        rows.append(
            {
                "label": args.baseline_calibration_label,
                "mlx_response_path": str(args.baseline_mlx_response),
                "cpu_auth_eval_path": str(args.baseline_cpu_auth_eval),
            }
        )
    rows.append(
        {
            "label": args.candidate_label,
            "mlx_response_path": str(parent_response),
            "cpu_auth_eval_path": str(auth_eval_path),
        }
    )
    return {
        "schema_version": "mlx_score_calibration_rows.v1",
        "strict_full_calibration": strict_full_calibration,
        "mlx_pair_window": [args.start_pair, args.start_pair + args.max_pairs],
        "auth_eval_sample_count_contract": 600,
        "rows": rows,
        **FALSE_AUTHORITY,
    }


def _first_blocker(
    auth_eval_path: Path,
    tensor_manifest_path: Path,
    downloaded_cache_dir: Path,
    cache_dir: Path,
    parent_response: Path,
    contract: Path,
    *,
    rebuild_dataset: bool,
    baseline_window_response: list[str],
    baseline_window_count: int,
    expected_baseline_window_count: int,
    candidate_windows_index: Path,
    dataset_for_contract: Path,
    downloaded_tensor_files: list[Path],
    strict_full_calibration: bool,
) -> str | None:
    if not auth_eval_path.is_file():
        return "modal_auth_eval_not_recovered"
    if not tensor_manifest_path.is_file():
        return "tensor_volume_manifest_missing"
    if not downloaded_cache_dir.is_dir() or not all(
        path.is_file() for path in downloaded_tensor_files
    ):
        return "tensor_volume_not_downloaded"
    if not cache_dir.is_dir():
        return "downloaded_tensor_cache_not_materialized"
    if not parent_response.is_file():
        return "parent_response_missing"
    if rebuild_dataset:
        if not baseline_window_response:
            return "baseline_window_response_missing"
        if baseline_window_count < expected_baseline_window_count:
            return "baseline_window_response_coverage_incomplete"
        if not candidate_windows_index.is_file():
            return "parent_windows_not_split"
        if not dataset_for_contract.is_file():
            return "same_axis_dataset_not_rebuilt"
    elif not dataset_for_contract.is_file():
        return "dataset_missing"
    if not strict_full_calibration:
        return "partial_mlx_full_auth_calibration_forbidden"
    if not contract.is_file():
        return "strict_parent_contract_missing"
    return None


def _run_id_from_auth_dir(auth_dir: Path) -> str:
    local = _read_optional_object(auth_dir / "modal_cpu_auth_eval_local_request.json")
    value = local.get("scorer_input_cache_tensor_volume_run_id")
    if isinstance(value, str) and value:
        return value
    return auth_dir.name


def _archive_size_bytes(local_request: dict[str, Any], auth_eval_path: Path) -> int:
    value = local_request.get("archive_size_bytes")
    if isinstance(value, int) and value >= 0:
        return value
    if auth_eval_path.is_file():
        auth_eval = _read_optional_object(auth_eval_path)
        value = auth_eval.get("archive_size_bytes")
        if isinstance(value, int) and value >= 0:
            return value
    raise SystemExit("archive_size_bytes missing; recover Modal auth eval first")


def _read_optional_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# MLX Parent Contract Closure Plan",
        "",
        "## Authority",
        "",
    ]
    for key in FALSE_AUTHORITY:
        lines.append(f"- {key}: `{plan[key]}`")
    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- Status: `{plan['status']}`",
            f"- Next blocker: `{plan['next_blocker']}`",
            f"- Call ID: `{plan.get('call_id')}`",
            f"- Tensor volume run ID: `{plan.get('tensor_volume_run_id')}`",
            "",
            "## Steps",
            "",
        ]
    )
    for step in plan["steps"]:
        lines.append(f"### {step['id']}")
        lines.append("")
        lines.append(f"- Ready now: `{step['ready']}`")
        if step.get("requires"):
            lines.append(f"- Requires: `{step['requires']}`")
        lines.append("")
        if "shell" in step:
            lines.append("```bash")
            lines.append(step["shell"])
            lines.append("```")
        else:
            lines.append(f"- Write JSON: `{step['write_json_path']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
