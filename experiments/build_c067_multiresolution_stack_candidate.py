#!/usr/bin/env python3
"""Emit a deterministic C067/apogee multi-resolution stack build manifest.

This is a planning-to-build bridge, not an archive composer. It consumes the
C067 multi-resolution planner JSON, emits runnable standalone builder commands
for the first supported pass components, and fails closed for unsupported
multi-pass composition. No scorer is loaded, no archive is written, no remote
job is dispatched, and no score claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/build_c067_multiresolution_stack_candidate.py"
SCHEMA = "c067_multiresolution_stack_build_manifest_v1"
PLANNER_SCHEMA = "c067_multiresolution_stack_planner_v1"
EVIDENCE_GRADE = "planning_to_build_non_promotable"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
DEFAULT_PLAN_JSON = (
    REPO_ROOT
    / "experiments/results/c067_multiresolution_stack_planner_20260502/"
    "c067_multiresolution_stack_plan.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/c067_multiresolution_stack_builder_20260502"
)
DEFAULT_MANIFEST_NAME = "c067_multiresolution_stack_build_manifest.json"


class BridgeError(ValueError):
    """Raised when a requested stack build would exceed the bridge contract."""


@dataclass(frozen=True)
class BuilderCommand:
    component_id: str
    pass_index: int
    builder_script: str
    command: tuple[str, ...]
    reason: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BridgeError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise BridgeError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _assert_no_score_claim_true(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        if value.get("score_claim") is True:
            raise BridgeError(f"{context} contains score_claim=true")
        for key, child in value.items():
            _assert_no_score_claim_true(child, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_score_claim_true(child, context=f"{context}[{index}]")


def _repo_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _path_from_record(record: dict[str, Any], key: str) -> Path | None:
    value = record.get(key)
    if isinstance(value, dict):
        raw = value.get("path")
        if isinstance(raw, str) and raw:
            return Path(raw)
    return None


def _safe_id(raw: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:96] or "component"


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BridgeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _script_and_args(command: tuple[str, ...]) -> tuple[str, list[str]]:
    for index, token in enumerate(command):
        if token.endswith(".py"):
            return token, list(command[index + 1 :])
    raise BridgeError(f"builder command has no Python script token: {list(command)!r}")


def _validate_builder_command(command: tuple[str, ...]) -> dict[str, Any]:
    script, argv = _script_and_args(command)
    script_path = (REPO_ROOT / script).resolve()
    if not script_path.exists():
        raise BridgeError(f"builder command references missing script: {script}")
    module = _load_module(script_path, f"_c067_multires_bridge_{_safe_id(script)}")
    parse_args = getattr(module, "parse_args", None)
    if parse_args is None:
        raise BridgeError(f"{script} has no parse_args function to validate against")
    parse_args(argv)
    return {
        "argparse_validated": True,
        "validated_script": script,
        "validated_arg_count": len(argv),
        "score_claim": False,
    }


def _manifest_for_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BridgeError(f"artifact {artifact.get('artifact_id')!r} has no path")
    path = Path(raw_path)
    if not path.exists():
        raise BridgeError(f"artifact manifest does not exist: {path}")
    payload = _read_json(path)
    _assert_no_score_claim_true(payload, context=str(path))
    return payload


def _common_frontier_and_mask_args(
    manifest: dict[str, Any],
    *,
    component_id: str,
) -> tuple[Path, Path]:
    frontier = _path_from_record(manifest, "frontier_archive")
    decoded = _path_from_record(manifest, "decoded_mask_array")
    if frontier is None:
        raise BridgeError(f"{component_id} manifest has no frontier_archive.path")
    if decoded is None:
        raise BridgeError(f"{component_id} manifest has no decoded_mask_array.path")
    return frontier, decoded


def _cmg2_command(
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    *,
    step_output_dir: Path,
) -> BuilderCommand:
    component_id = str(artifact["artifact_id"])
    frontier, decoded = _common_frontier_and_mask_args(manifest, component_id=component_id)
    cmg2 = manifest.get("cmg2")
    if not isinstance(cmg2, dict):
        raise BridgeError(f"{component_id} manifest has no cmg2 record")
    scale = cmg2.get("scale")
    if not (isinstance(scale, list) and len(scale) == 2):
        raise BridgeError(f"{component_id} manifest has no cmg2.scale [y,x]")
    compressor = str(cmg2.get("compressor", "bz2"))
    command = (
        "python",
        "experiments/build_cmg2_downsample_candidate.py",
        "--frontier-archive",
        _repo_display_path(frontier),
        "--decoded-mask-array",
        _repo_display_path(decoded),
        "--output-dir",
        _repo_display_path(step_output_dir),
        "--scale-y",
        str(int(scale[0])),
        "--scale-x",
        str(int(scale[1])),
        "--compressor",
        compressor,
        "--force",
    )
    return BuilderCommand(
        component_id=component_id,
        pass_index=int(artifact.get("pass_index", 1)),
        builder_script="experiments/build_cmg2_downsample_candidate.py",
        command=command,
        reason="standalone CMG2 pass rebuild from existing manifest parameters",
    )


def _cmg3_nonzero_command(
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    *,
    step_output_dir: Path,
) -> BuilderCommand:
    component_id = str(artifact["artifact_id"])
    frontier, decoded = _common_frontier_and_mask_args(manifest, component_id=component_id)
    cmg3 = manifest.get("cmg3")
    if not isinstance(cmg3, dict):
        raise BridgeError(f"{component_id} manifest has no cmg3 record")
    command = (
        "python",
        "experiments/build_cmg3_nonzero_runs_candidate.py",
        "--frontier-archive",
        _repo_display_path(frontier),
        "--decoded-mask-array",
        _repo_display_path(decoded),
        "--output-dir",
        _repo_display_path(step_output_dir),
        "--max-runs-per-row",
        str(int(cmg3.get("max_runs_per_row", 2))),
        "--compressor",
        str(cmg3.get("compressor", "auto")),
        "--force",
    )
    return BuilderCommand(
        component_id=component_id,
        pass_index=int(artifact.get("pass_index", 1)),
        builder_script="experiments/build_cmg3_nonzero_runs_candidate.py",
        command=command,
        reason="standalone CMG3 nonzero row-run pass rebuild from existing manifest parameters",
    )


def _pmg_hotspot_command(
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    *,
    step_output_dir: Path,
) -> BuilderCommand:
    component_id = str(artifact["artifact_id"])
    frontier, decoded = _common_frontier_and_mask_args(manifest, component_id=component_id)
    plan = _path_from_record(manifest, "pmg_hotspot_plan")
    if plan is None:
        raise BridgeError(f"{component_id} manifest has no pmg_hotspot_plan.path")
    pmg = manifest.get("pmg_hotspot_cmg3")
    if not isinstance(pmg, dict):
        raise BridgeError(f"{component_id} manifest has no pmg_hotspot_cmg3 record")
    command: tuple[str, ...] = (
        "python",
        "experiments/build_pmg_hotspot_candidate.py",
        "--plan-json",
        _repo_display_path(plan),
        "--frontier-archive",
        _repo_display_path(frontier),
        "--decoded-mask-array",
        _repo_display_path(decoded),
        "--output-dir",
        _repo_display_path(step_output_dir),
        "--candidate-id",
        str(pmg.get("candidate_id")),
        "--compressor",
        str(pmg.get("compressor", "lzma_xz")),
        "--force",
    )
    return BuilderCommand(
        component_id=component_id,
        pass_index=int(artifact.get("pass_index", 1)),
        builder_script="experiments/build_pmg_hotspot_candidate.py",
        command=command,
        reason="standalone PMG-HOTSPOT pass rebuild from existing manifest parameters",
    )


def _field_policy_command(
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    *,
    step_output_dir: Path,
) -> BuilderCommand:
    component_id = str(artifact["artifact_id"])
    raw_command = manifest.get("concrete_builder_command_if_safe")
    if not isinstance(raw_command, list) or not all(isinstance(item, str) for item in raw_command):
        raise BridgeError(f"{component_id} plan has no concrete_builder_command_if_safe")
    command = list(raw_command)
    if "--output-dir" not in command:
        raise BridgeError(f"{component_id} concrete command has no --output-dir")
    command[command.index("--output-dir") + 1] = _repo_display_path(step_output_dir)
    return BuilderCommand(
        component_id=component_id,
        pass_index=int(artifact.get("pass_index", 2)),
        builder_script="experiments/build_cmg3_adaptive_runs_candidate.py",
        command=tuple(command),
        reason="standalone CMG3 adaptive field-policy pass from planner concrete command",
    )


def _standalone_command_for_component(
    artifact: dict[str, Any],
    *,
    step_output_dir: Path,
) -> BuilderCommand | None:
    if artifact.get("builder_consumable") is not True:
        return None
    schema = artifact.get("schema")
    manifest = _manifest_for_artifact(artifact)
    if schema == "cmg2_downsample_candidate_v1":
        return _cmg2_command(artifact, manifest, step_output_dir=step_output_dir)
    if schema == "cmg3_nonzero_row_runs_candidate_v1":
        return _cmg3_nonzero_command(artifact, manifest, step_output_dir=step_output_dir)
    if schema == "pmg_hotspot_cmg3_candidate_v1":
        return _pmg_hotspot_command(artifact, manifest, step_output_dir=step_output_dir)
    if schema == "c067_hotspot_mask_geometry_compiler_v1":
        return _field_policy_command(artifact, manifest, step_output_dir=step_output_dir)
    return None


def _unsupported_component_reason(artifact: dict[str, Any]) -> str | None:
    if artifact.get("pass_index") == 0:
        return "anchor is existing custody/eval input, not a new builder step"
    if artifact.get("builder_consumable") is not True:
        return "component has no byte-closed standalone builder command in the planner artifact"
    if artifact.get("schema") not in {
        "cmg2_downsample_candidate_v1",
        "cmg3_nonzero_row_runs_candidate_v1",
        "pmg_hotspot_cmg3_candidate_v1",
        "c067_hotspot_mask_geometry_compiler_v1",
    }:
        return f"unsupported builder schema for this bridge: {artifact.get('schema')!r}"
    return None


def _policy_blockers(
    policy: dict[str, Any],
    artifacts_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if policy.get("existing_builder_can_consume_full_stack") is not True:
        blockers.append(
            "policy has existing_builder_can_consume_full_stack=false; no byte-closed stack builder consumes this composition"
        )
    if policy.get("dispatchable_from_this_plan") is not True:
        blockers.append(
            "planner marks dispatchable_from_this_plan=false; remote or exact eval dispatch must not be launched from this manifest"
        )
    non_anchor_members: dict[str, list[str]] = {}
    for component_id in _coerce_str_list(policy.get("component_ids")):
        artifact = artifacts_by_id.get(component_id)
        if artifact is None:
            blockers.append(f"policy references missing component artifact: {component_id}")
            continue
        if int(artifact.get("pass_index", 0)) == 0:
            continue
        for member in _coerce_str_list(artifact.get("logical_members")):
            non_anchor_members.setdefault(member, []).append(component_id)
    overlapping = {
        member: sorted(ids)
        for member, ids in sorted(non_anchor_members.items())
        if len(set(ids)) > 1
    }
    for member, component_ids in overlapping.items():
        blockers.append(
            "unsupported score-affecting member composition on "
            f"{member}: {', '.join(component_ids)}"
        )
    for component_id in _coerce_str_list(policy.get("component_ids")):
        artifact = artifacts_by_id.get(component_id)
        if artifact is None:
            continue
        if int(artifact.get("pass_index", 0)) in {3, 4} and artifact.get("builder_consumable") is not True:
            blockers.append(
                f"pass{artifact.get('pass_index')} component {component_id} is planning/profile input only; no charged builder is available"
            )
    return sorted(set(blockers))


def _policy_record(
    policy: dict[str, Any],
    *,
    artifacts_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    policy_id = str(policy.get("policy_id", "policy"))
    policy_dir = output_dir / "standalone_steps" / _safe_id(policy_id)
    runnable_steps: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    seen_step_components: set[str] = set()

    for component_id in _coerce_str_list(policy.get("component_ids")):
        artifact = artifacts_by_id.get(component_id)
        if artifact is None:
            unsupported.append(
                {
                    "component_id": component_id,
                    "reason": "policy references missing component artifact",
                    "score_claim": False,
                }
            )
            continue
        reason = _unsupported_component_reason(artifact)
        if reason is not None and artifact.get("builder_consumable") is not True:
            unsupported.append(
                {
                    "component_id": component_id,
                    "pass_index": artifact.get("pass_index"),
                    "schema": artifact.get("schema"),
                    "reason": reason,
                    "score_claim": False,
                }
            )
            continue
        if component_id in seen_step_components:
            continue
        step_output_dir = policy_dir / _safe_id(component_id)
        try:
            command = _standalone_command_for_component(
                artifact,
                step_output_dir=step_output_dir,
            )
        except BridgeError as exc:
            unsupported.append(
                {
                    "component_id": component_id,
                    "pass_index": artifact.get("pass_index"),
                    "schema": artifact.get("schema"),
                    "reason": str(exc),
                    "score_claim": False,
                }
            )
            continue
        if command is None:
            unsupported.append(
                {
                    "component_id": component_id,
                    "pass_index": artifact.get("pass_index"),
                    "schema": artifact.get("schema"),
                    "reason": _unsupported_component_reason(artifact)
                    or "no standalone command emitted",
                    "score_claim": False,
                }
            )
            continue
        validation = _validate_builder_command(command.command)
        source_path = Path(str(artifact["path"]))
        runnable_steps.append(
            {
                "step_id": f"{_safe_id(policy_id)}__{_safe_id(command.component_id)}",
                "component_id": command.component_id,
                "pass_index": command.pass_index,
                "builder_script": command.builder_script,
                "command": list(command.command),
                "command_status": "argparse_valid",
                "validation": validation,
                "standalone_only": True,
                "part_of_byte_closed_stack": False,
                "output_dir": _repo_display_path(step_output_dir),
                "source_artifact": {
                    "path": _repo_display_path(source_path),
                    "sha256": _sha256_file(source_path),
                    "schema": artifact.get("schema"),
                    "score_claim": False,
                },
                "reason": command.reason,
                "score_claim": False,
            }
        )
        seen_step_components.add(component_id)

    blockers = _policy_blockers(policy, artifacts_by_id)
    if not runnable_steps:
        blockers.append("no runnable standalone builder step exists for this policy")
    return {
        "policy_id": policy_id,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "byte_closed_stack_archive_emitted": False,
        "archive_path": None,
        "dispatchable": False,
        "full_stack_status": "blocked",
        "blocker_reasons": sorted(set(blockers)),
        "runnable_standalone_steps": runnable_steps,
        "unsupported_components": unsupported,
        "exact_eval_branch_rule": {
            "score_claim": False,
            "required_before_any_score_claim": True,
            "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
            "dispatch_claim_required_before_remote_job": True,
            "branch": (
                "implement byte-closed stack composer -> verify archive bytes, "
                "payload closure, runtime tree hash, and no sidecars -> claim lane "
                "-> exact CUDA auth eval"
            ),
        },
    }


def _select_policies(plan: dict[str, Any], policy_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    policies = plan.get("candidate_policies")
    if not isinstance(policies, list):
        raise BridgeError("planner JSON has no candidate_policies list")
    selected: list[dict[str, Any]] = []
    requested = set(policy_ids)
    seen: set[str] = set()
    for policy in policies:
        if not isinstance(policy, dict):
            raise BridgeError("candidate_policies contains a non-object policy")
        policy_id = str(policy.get("policy_id", ""))
        if policy_ids and policy_id not in requested:
            continue
        if policy_id in seen:
            raise BridgeError(f"duplicate policy_id in planner JSON: {policy_id}")
        seen.add(policy_id)
        selected.append(policy)
    missing = sorted(requested - seen)
    if missing:
        raise BridgeError(f"requested policy_id not found: {', '.join(missing)}")
    if not selected:
        raise BridgeError("no candidate policies selected")
    return selected


def build_manifest(
    *,
    plan_json: Path = DEFAULT_PLAN_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_json: Path | None = None,
    policy_ids: tuple[str, ...] = (),
    require_byte_closed_stack: bool = False,
) -> dict[str, Any]:
    plan_json = plan_json.resolve()
    output_dir = output_dir.resolve()
    manifest_json = (
        (output_dir / DEFAULT_MANIFEST_NAME)
        if manifest_json is None
        else manifest_json.resolve()
    )
    plan = _read_json(plan_json)
    if plan.get("schema") != PLANNER_SCHEMA:
        raise BridgeError(f"expected planner schema {PLANNER_SCHEMA!r}, got {plan.get('schema')!r}")
    _assert_no_score_claim_true(plan, context=str(plan_json))

    artifacts = plan.get("loaded_artifacts")
    if not isinstance(artifacts, list):
        raise BridgeError("planner JSON has no loaded_artifacts list")
    artifacts_by_id: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise BridgeError("loaded_artifacts contains a non-object artifact")
        artifact_id = str(artifact.get("artifact_id", ""))
        if not artifact_id:
            raise BridgeError("loaded artifact has no artifact_id")
        if artifact_id in artifacts_by_id:
            raise BridgeError(f"duplicate artifact_id in planner JSON: {artifact_id}")
        artifacts_by_id[artifact_id] = artifact

    selected_policies = _select_policies(plan, policy_ids)
    policy_records = [
        _policy_record(policy, artifacts_by_id=artifacts_by_id, output_dir=output_dir)
        for policy in selected_policies
    ]
    runnable_step_count = sum(
        len(record["runnable_standalone_steps"]) for record in policy_records
    )
    all_blockers = sorted(
        {
            blocker
            for record in policy_records
            for blocker in record["blocker_reasons"]
        }
    )
    if require_byte_closed_stack:
        raise BridgeError(
            "byte-closed C067/apogee stack construction is unsupported by this bridge: "
            + "; ".join(all_blockers)
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "no_score_claim": True,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "planning_to_build_bridge": True,
        "byte_closed_stack_archive_emitted": False,
        "cuda_jobs_launched": False,
        "remote_jobs_dispatched": False,
        "output_dir": _repo_display_path(output_dir),
        "manifest_json": _repo_display_path(manifest_json),
        "plan_json": {
            "path": _repo_display_path(plan_json),
            "sha256": _sha256_file(plan_json),
            "schema": plan.get("schema"),
            "score_claim": False,
        },
        "selected_policy_ids": [str(policy["policy_id"]) for policy in selected_policies],
        "candidate_policy_count": len(selected_policies),
        "runnable_standalone_step_count": runnable_step_count,
        "build_policy_manifests": policy_records,
        "global_blocker_reasons": all_blockers,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "required_next_steps": [
            "Use emitted commands only as standalone byte-screen rebuilds; they are not a stacked archive.",
            "Add a reviewed byte-closed composer before combining score-affecting mask, pose, runtime, or packer passes.",
            "After a byte-closed stack exists, claim the lane before any remote job and run exact CUDA auth eval on the exact archive bytes.",
        ],
    }
    _assert_no_score_claim_true(payload, context="bridge_manifest")
    _write_json(manifest_json, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/c067_multiresolution_stack_build_manifest.json.",
    )
    parser.add_argument(
        "--policy-id",
        action="append",
        default=[],
        help="Restrict output to one policy id. Repeat to include multiple policies.",
    )
    parser.add_argument(
        "--require-byte-closed-stack",
        action="store_true",
        help="Fail if the selected policy cannot produce a byte-closed stacked archive.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_manifest(
            plan_json=args.plan_json,
            output_dir=args.output_dir,
            manifest_json=args.manifest_json,
            policy_ids=tuple(args.policy_id),
            require_byte_closed_stack=bool(args.require_byte_closed_stack),
        )
    except BridgeError as exc:
        print(json.dumps({"error": str(exc), "score_claim": False}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest_json": manifest["manifest_json"],
                "runnable_standalone_step_count": manifest["runnable_standalone_step_count"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
