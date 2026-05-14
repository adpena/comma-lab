#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed decoded-baseline build preflight for Lane 12 NeRV.

This tool proves the local inputs for a future build-only Lane 12 run without
launching remote work and without writing the L2 clearance packet. It consumes
the same decoded-baseline mask loader and Alpha-Geo primitive-contract validator
as ``experiments/train_nerv_mask.py`` so a green preflight means the trainer's
target contract is locally coherent before any GPU setup time is spent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lane12_decoded_baseline_build_preflight_v1"
REPORT_ID = "lane12_decoded_baseline_build_preflight_20260502"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane12_l2_unblock_readiness_20260502"
    / "decoded_baseline_build_preflight.json"
)
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_12_nerv_20260430_codex_jsonfix40"
    / "alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.primitive_contract.json"
)
BASELINE_FALLBACKS = (
    "experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip",
    "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip",
    "experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip",
)
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "remote_lane_nerv.sh"
DEFAULT_CLEARANCE = REPO_ROOT / ".omx" / "state" / "lane12_nerv_l2_clearance.json"
EXPECTED_SHAPE = [1200, 384, 512]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_path(path: str | Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _display_path(path: Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} JSON root must be an object")
    return payload


def _path_record(path: Path | None, repo_root: Path) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "is_file": False}
    record: dict[str, Any] = {
        "path": _display_path(path, repo_root),
        "exists": path.exists(),
        "is_file": path.is_file(),
    }
    if path.is_file():
        record["size_bytes"] = int(path.stat().st_size)
    return record


def _contract_baseline_path(contract: dict[str, Any], repo_root: Path) -> Path | None:
    source = contract.get("source")
    if not isinstance(source, dict):
        return None
    baseline = source.get("baseline")
    if not isinstance(baseline, dict):
        return None
    path = baseline.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = _resolve_path(path, repo_root)
    return candidate if candidate is not None and candidate.exists() else None


def _choose_decoded_baseline_path(
    *,
    repo_root: Path,
    contract: dict[str, Any],
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        return explicit
    contract_path = _contract_baseline_path(contract, repo_root)
    if contract_path is not None:
        return contract_path
    for rel in BASELINE_FALLBACKS:
        candidate = repo_root / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no decoded-baseline archive found; pass --decoded-baseline-path"
    )


def _launcher_guard_report(script_path: Path, repo_root: Path) -> dict[str, Any]:
    record = _path_record(script_path, repo_root)
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    if not script_path.is_file():
        return {
            "path": record,
            "passed": False,
            "checks": checks,
            "blockers": ["remote lane script is missing"],
        }
    text = script_path.read_text()
    required_substrings = {
        "defaults_decoded_baseline": 'GT_MASKS_SOURCE="${GT_MASKS_SOURCE:-decoded-baseline}"',
        "build_only_default": 'RUN_AUTH_EVAL="${RUN_AUTH_EVAL:-0}"',
        "requires_alpha_contract": "requires ALPHA_PRIMITIVE_CONTRACT",
        "l2_clearance_fail_closed": "No new NeRV retraining is allowed until this packet is valid",
        "trainer_uses_alpha_contract": "--alpha-primitive-contract",
        "forbids_retired_segnet_default": "ALLOW_RETIRED_SEGNET_TARGET",
        "run_auth_eval_requires_promotion_geometry": "Alpha-Geo threshold_preset must be promotion",
    }
    for name, needle in required_substrings.items():
        passed = needle in text
        checks[name] = {"passed": passed, "needle": needle}
        if not passed:
            blockers.append(name)
    return {
        "path": record,
        "passed": not blockers,
        "checks": checks,
        "blockers": blockers,
    }


def _runtime_closure_report(repo_root: Path) -> dict[str, Any]:
    """Static fail-closed checks for Lane 12 runtime unpack/inflate closure."""
    paths = {
        "unpack_renderer_payload": repo_root / "submissions" / "robust_current" / "unpack_renderer_payload.py",
        "inflate_renderer": repo_root / "submissions" / "robust_current" / "inflate_renderer.py",
        "inflate_sh": repo_root / "submissions" / "robust_current" / "inflate.sh",
    }
    path_records = {name: _path_record(path, repo_root) for name, path in paths.items()}
    blockers: list[str] = []
    checks: dict[str, Any] = {}

    texts: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            blockers.append(f"{name}:missing")
            texts[name] = ""
        else:
            texts[name] = path.read_text(errors="ignore")

    required_tokens = {
        "unpack_renderer_payload": {
            "nrv_magic": "NERV_MAGIC = b\"NRV1\"",
            "nrv_qzs3_qp1_parser": "_parse_public_pr67_nerv_qzs3_qp1_payload",
            "nrv_qzs3_payload_format": "public_pr67_nerv_qzs3_qp1_fixed_slices",
            "qzs3_renderer_content_validation": "renderer.startswith(b\"QZS3\")",
            "qp1_pose_content_validation": "pose_qp1.startswith(b\"QP1\")",
            "self_describing_nrv_length": "_parse_nerv_member_length",
        },
        "inflate_renderer": {
            "archive_default_mask_source": 'os.environ.get("INFLATE_MASK_SOURCE", "archive")',
            "nrv_mask_loader": "def _load_masks_from_nrv",
            "nrv_member_candidate": 'archive / "masks.nrv"',
            "segnet_fallback_explicit_only": "INFLATE_MASK_SOURCE=archive",
            "tto_default_off": 'os.environ.get("INFLATE_TTO", "0") == "1"',
            "strict_scorer_banner": "[strict-scorer-rule]",
        },
        "inflate_sh": {
            "packed_payload_unpack": "unpack_renderer_payload.py",
            "unpack_summary_json": "renderer_payload_unpack_summary.json",
        },
    }
    for file_key, token_map in required_tokens.items():
        file_text = texts[file_key]
        for check_name, token in token_map.items():
            passed = token in file_text
            checks[f"{file_key}:{check_name}"] = {"passed": passed, "token": token}
            if not passed:
                blockers.append(f"{file_key}:{check_name}")

    forbidden_default_tokens = {
        "inflate_renderer:segnet_default": 'os.environ.get("INFLATE_MASK_SOURCE", "segnet")',
        "inflate_renderer:tto_default_on": 'os.environ.get("INFLATE_TTO", "1") == "1"',
    }
    for check_name, token in forbidden_default_tokens.items():
        present = token in texts["inflate_renderer"]
        checks[check_name] = {"passed": not present, "forbidden_token": token}
        if present:
            blockers.append(check_name)

    return {
        "schema": "lane12_runtime_closure_static_preflight_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "paths": path_records,
        "passed": not blockers,
        "checks": checks,
        "blockers": sorted(set(blockers)),
    }


def preflight_lane12_decoded_baseline_build(
    *,
    repo_root: Path = REPO_ROOT,
    decoded_baseline_path: Path | None = None,
    decoded_baseline_member: str = "masks.mkv",
    alpha_primitive_contract: Path = DEFAULT_CONTRACT,
    clearance_json: Path = DEFAULT_CLEARANCE,
    remote_lane_script: Path = DEFAULT_SCRIPT,
    output_json: Path | None = DEFAULT_OUTPUT,
    force: bool = False,
    expected_shape: list[int] | None = None,
    use_default_artifact_globs: bool = True,
    command: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    expected = expected_shape or list(EXPECTED_SHAPE)
    contract_path = _resolve_path(alpha_primitive_contract, repo_root)
    if contract_path is None:
        raise ValueError("alpha_primitive_contract must be provided")
    contract = _load_json(contract_path)
    baseline_path = _choose_decoded_baseline_path(
        repo_root=repo_root,
        contract=contract,
        explicit=_resolve_path(decoded_baseline_path, repo_root),
    )
    script_path = _resolve_path(remote_lane_script, repo_root)
    clearance_path = _resolve_path(clearance_json, repo_root)

    train_mod = _load_module(
        "_lane12_preflight_train_nerv_mask", repo_root / "experiments" / "train_nerv_mask.py"
    )
    planner_mod = _load_module(
        "_lane12_preflight_plan_l2", repo_root / "experiments" / "plan_lane12_l2_unblock.py"
    )

    decoded_blockers: list[str] = []
    contract_blockers: list[str] = []
    decoded_metadata: dict[str, Any] = {}
    contract_metadata: dict[str, Any] = {}
    contract_gates: dict[str, Any] | None = None
    sampling_provenance: dict[str, Any] | None = None
    masks_shape: list[int] | None = None
    target_mask_sha256: str | None = None

    try:
        masks, decoded_metadata = train_mod._load_decoded_baseline_masks(
            baseline_path,
            archive_member=decoded_baseline_member,
            expected_frames=expected[0],
        )
        train_mod._validate_decoded_baseline_target_shape(
            masks,
            expected_frames=expected[0],
            expected_height=expected[1],
            expected_width=expected[2],
        )
        masks_shape = [int(v) for v in masks.shape]
        target_mask_sha256 = train_mod._mask_tensor_sha256(masks)
    except Exception as exc:  # pragma: no cover - exercised by functional tests
        decoded_blockers.append(f"{type(exc).__name__}: {exc}")
        masks = None

    try:
        contract_loaded, contract_metadata = train_mod._load_alpha_primitive_contract(
            contract_path
        )
        if masks is None:
            contract_blockers.append("decoded baseline did not load; contract validation skipped")
        else:
            contract_gates = train_mod._validate_alpha_primitive_contract(
                contract_loaded,
                contract_metadata=contract_metadata,
                masks_THW=masks,
                decoded_metadata=decoded_metadata,
            )
            _pool, sampling_provenance = train_mod._build_alpha_primitive_sampling_pool(
                contract_loaded,
                masks,
                seed=12,
                contract_sha256=contract_metadata["sha256"],
            )
    except Exception as exc:
        contract_blockers.append(f"{type(exc).__name__}: {exc}")

    launcher = _launcher_guard_report(script_path, repo_root) if script_path is not None else {
        "passed": False,
        "blockers": ["remote lane script path did not resolve"],
    }
    runtime_closure = _runtime_closure_report(repo_root)
    readiness = planner_mod.plan_lane12_l2_unblock(
        repo_root=repo_root,
        clearance_json=clearance_path,
        primitive_contract_jsons=[contract_path],
        use_default_artifact_globs=use_default_artifact_globs,
        output_json=None,
        expected_shape=expected,
        command=["experiments/plan_lane12_l2_unblock.py", "[called by decoded-baseline preflight]"],
    )
    summary = readiness["readiness_summary"]

    decoded_contract_passed = not decoded_blockers and not contract_blockers
    remote_training_blockers: list[str] = []
    if not decoded_contract_passed:
        remote_training_blockers.extend(decoded_blockers)
        remote_training_blockers.extend(contract_blockers)
    if not launcher.get("passed"):
        remote_training_blockers.extend(f"launcher:{item}" for item in launcher.get("blockers", []))
    if not runtime_closure.get("passed"):
        remote_training_blockers.extend(
            f"runtime_closure:{item}" for item in runtime_closure.get("blockers", [])
        )
    if not summary.get("ready_for_retraining_unblock"):
        remote_training_blockers.extend(summary.get("blockers", []))

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "report_id": REPORT_ID,
        "deterministic_report": True,
        "score_claim": False,
        "promotion_eligible": False,
        "training_performed": False,
        "remote_job_launched": False,
        "clearance_state_written": False,
        "decoded_baseline_contract_preflight_passed": decoded_contract_passed,
        "ready_for_build_only_remote_training": (
            decoded_contract_passed
            and bool(launcher.get("passed"))
            and bool(runtime_closure.get("passed"))
            and bool(summary.get("ready_for_retraining_unblock"))
        ),
        "ready_for_exact_eval_dispatch": False,
        "inputs": {
            "repo_root": str(repo_root),
            "decoded_baseline_path": _display_path(baseline_path, repo_root),
            "decoded_baseline_member": decoded_baseline_member,
            "alpha_primitive_contract": _display_path(contract_path, repo_root),
            "remote_lane_script": _display_path(script_path, repo_root)
            if script_path is not None
            else None,
            "clearance_json": _display_path(clearance_path, repo_root)
            if clearance_path is not None
            else None,
            "expected_shape": expected,
        },
        "decoded_baseline": {
            "path": _path_record(baseline_path, repo_root),
            "metadata": decoded_metadata,
            "target_mask_shape": masks_shape,
            "target_mask_sha256": target_mask_sha256,
            "blockers": decoded_blockers,
        },
        "alpha_primitive_contract": {
            "path": _path_record(contract_path, repo_root),
            "metadata": contract_metadata,
            "consumption_gates": contract_gates,
            "sampling_pool": sampling_provenance,
            "blockers": contract_blockers,
        },
        "launcher_guards": launcher,
        "runtime_closure": runtime_closure,
        "l2_unblock_readiness": {
            "ready_for_retraining_unblock": summary.get("ready_for_retraining_unblock"),
            "ready_for_exact_eval_dispatch": summary.get("ready_for_exact_eval_dispatch"),
            "blockers": summary.get("blockers", []),
            "missing_prerequisites": readiness["evidence_buckets"]["missing_prerequisites"],
        },
        "remote_training_blockers": sorted(set(remote_training_blockers)),
        "next_action_if_green": {
            "requires_dispatch_claim": True,
            "score_claim": False,
            "training_performed_by_this_tool": False,
            "command_template": [
                "WORKSPACE=/workspace/pact",
                "LOG_DIR=/workspace/pact/lane_12_nerv_decoded_baseline_results",
                "RUN_AUTH_EVAL=0",
                "GT_MASKS_SOURCE=decoded-baseline",
                f"DECODED_BASELINE_PATH=/workspace/pact/{_display_path(baseline_path, repo_root)}",
                f"DECODED_BASELINE_MEMBER={decoded_baseline_member}",
                f"ALPHA_PRIMITIVE_CONTRACT=/workspace/pact/{_display_path(contract_path, repo_root)}",
                "bash scripts/remote_lane_nerv.sh",
            ],
        },
        "provenance": {
            "tool": "experiments/preflight_lane12_decoded_baseline_build.py",
            "command": command
            or ["experiments/preflight_lane12_decoded_baseline_build.py"],
        },
    }

    if output_json is not None:
        output_path = _resolve_path(output_json, repo_root)
        if output_path is None:
            raise ValueError("output_json did not resolve")
        if output_path.exists() and not force:
            raise FileExistsError(f"{output_path} exists; use --force to overwrite")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--decoded-baseline-path", type=Path, default=None)
    parser.add_argument("--decoded-baseline-member", default="masks.mkv")
    parser.add_argument("--alpha-primitive-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--clearance-json", type=Path, default=DEFAULT_CLEARANCE)
    parser.add_argument("--remote-lane-script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_SHAPE[0])
    parser.add_argument("--expected-height", type=int, default=EXPECTED_SHAPE[1])
    parser.add_argument("--expected-width", type=int, default=EXPECTED_SHAPE[2])
    parser.add_argument(
        "--no-default-artifact-globs",
        action="store_true",
        help="Do not let the nested L2 readiness planner read repo-default artifacts.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = preflight_lane12_decoded_baseline_build(
        repo_root=args.repo_root,
        decoded_baseline_path=args.decoded_baseline_path,
        decoded_baseline_member=args.decoded_baseline_member,
        alpha_primitive_contract=args.alpha_primitive_contract,
        clearance_json=args.clearance_json,
        remote_lane_script=args.remote_lane_script,
        output_json=args.output_json,
        force=args.force,
        expected_shape=[args.expected_frames, args.expected_height, args.expected_width],
        use_default_artifact_globs=not args.no_default_artifact_globs,
        command=[
            "experiments/preflight_lane12_decoded_baseline_build.py",
            *(argv if argv is not None else sys.argv[1:]),
        ],
    )
    print(
        "[lane12-decoded-baseline-preflight] "
        f"wrote {args.output_json} "
        f"contract_preflight={report['decoded_baseline_contract_preflight_passed']} "
        f"ready_for_build_only_remote_training={report['ready_for_build_only_remote_training']} "
        "training_performed=false clearance_state_written=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
