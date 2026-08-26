#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build or explicitly execute the provider-neutral V9 CGauge CUDA plan."""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.deploy.witness_cloud_launcher import (  # noqa: E402
    DEFAULT_DSL_CONFIG,
    DEFAULT_TORCH_COMPILE_MODE,
    EXACT_H100_GPU,
    POSENET_WEIGHTS,
    RESULTS_VOLUME,
    SEGNET_WEIGHTS,
    TASK438_GT_CACHE_BYTES,
    TASK438_GT_CACHE_SHA256,
    TORCH_COMPILE_MODES,
    V9_DSL_CONFIGS,
    build_plan,
    render_command,
)

OPERATOR_TOKEN = "GO-CLOUD-381"
IMAGE_CACHE_BYPASS_ENV = ("MODAL_FORCE_BUILD", "MODAL_IGNORE_CACHE")
OUTER_V9_GUARD_NAMESPACE = "witness_cloud_outer_dispatch_guards"
MODAL_APP_DEFINITION = "experiments/modal_train_lane.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file_sha256(path: Path, *, custody_name: str) -> str:
    if not path.is_file():
        raise SystemExit(f"REFUSED: required {custody_name} custody file is missing: {path}")
    return _sha256_file(path)


def _verify_local_asset_stage_contract(
    plan, *, actual_sha256: str | None = None
) -> dict[str, object]:
    """Prove local GT custody maps to one exact Volume destination.

    This check is local-only: it never imports Modal, contacts the provider,
    or executes the advisory ``asset_stage_argv``.
    """
    if plan.gt_cache_sha256 is None:
        raise SystemExit("REFUSED: plan lacks an exact GT digest")
    local_cache = REPO / plan.local_gt_cache
    if not local_cache.is_file():
        raise SystemExit(f"REFUSED: required GT cache custody file is missing: {local_cache}")
    local_bytes = local_cache.stat().st_size
    if local_bytes != plan.gt_cache_bytes:
        raise SystemExit(
            "REFUSED: GT cache byte-size mismatch: "
            f"{local_bytes} != {plan.gt_cache_bytes}"
        )
    actual = actual_sha256 or _sha256_file(local_cache)
    if actual != plan.gt_cache_sha256:
        raise SystemExit(
            f"REFUSED: GT cache SHA-256 mismatch: {actual} != {plan.gt_cache_sha256}"
        )
    remote_relative = f"assets/v9_cgauge/gt_{plan.gt_cache_sha256}.npz"
    expected_stage_argv = (
        ".venv/bin/modal",
        "volume",
        "put",
        RESULTS_VOLUME,
        plan.local_gt_cache,
        remote_relative,
    )
    if tuple(plan.asset_stage_argv) != expected_stage_argv:
        raise SystemExit("REFUSED: asset_stage_argv drifted from exact SHA-addressed custody")
    expected_remote = f"/modal_results/{remote_relative}"
    if plan.remote_gt_cache != expected_remote:
        raise SystemExit("REFUSED: mounted GT path drifted from the staged Volume path")
    return {
        "schema": "witness_asset_stage_readiness.v1",
        "status": "passed",
        "provider_contacted": False,
        "staging_executed": False,
        "volume": RESULTS_VOLUME,
        "local_path": plan.local_gt_cache,
        "bytes": local_bytes,
        "sha256": actual,
        "volume_relative_path": remote_relative,
        "mounted_path": expected_remote,
        "asset_stage_argv": list(expected_stage_argv),
    }


def _main_git_head() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "main"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise SystemExit(f"REFUSED: could not resolve an exact main HEAD, got {head!r}")
    return head


def _require_clean_main_worktree(expected_git_head: str) -> None:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    if branch != "main":
        raise SystemExit(f"REFUSED: execution requires main, found {branch or 'detached HEAD'}")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise SystemExit("REFUSED: execution requires a clean main worktree before asset staging")
    actual_git_head = _main_git_head()
    if actual_git_head != expected_git_head:
        raise SystemExit(
            "REFUSED: main HEAD changed after plan review; rebuild and re-review the plan "
            f"({actual_git_head} != {expected_git_head})"
        )


def _require_modal_image_cache_policy() -> None:
    """Refuse explicit execution when the bounded image-cache assumption is off."""
    bypasses = {
        name: value
        for name in IMAGE_CACHE_BYPASS_ENV
        if (value := os.environ.get(name, ""))
    }
    if bypasses:
        rendered = ", ".join(f"{name}={value!r}" for name, value in sorted(bypasses.items()))
        raise SystemExit(
            "REFUSED: Modal image-cache bypass invalidates the budgeted image/staging "
            f"allowance ({rendered})"
        )


def _require_no_active_v9_claim(plan) -> None:
    """Refuse any active Task438 V9 claim, regardless of its label."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/claim_lane_dispatch.py",
                "summary",
                "--claims-path",
                ".omx/state/active_lane_dispatch_claims.md",
                "--format",
                "json",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise SystemExit(
            "REFUSED: active V9 dispatch-claim read failed before provider I/O"
        ) from exc
    if proc.returncode != 0:
        raise SystemExit(
            "REFUSED: active V9 dispatch-claim summary failed before provider I/O: "
            f"rc={proc.returncode} stderr={proc.stderr.strip()[:500]!r}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "REFUSED: active V9 dispatch-claim summary is not valid JSON"
        ) from exc
    active = payload.get("active") if isinstance(payload, dict) else None
    if not isinstance(active, list):
        raise SystemExit("REFUSED: active V9 dispatch-claim summary lacks an active list")
    conflicts = [
        row
        for row in active
        if isinstance(row, dict) and row.get("lane_id") == plan.lane_id
    ]
    if conflicts:
        labels = sorted({str(row.get("instance_job_id", "")) for row in conflicts})
        raise SystemExit(
            "REFUSED: Task438 V9 lane already has an active claim; lane-global "
            f"duplicate spending is forbidden (active labels={labels})"
        )


def _outer_v9_guard_path(plan) -> Path:
    """Return one label-independent guard path for the Task438 V9 lane."""
    digest = hashlib.sha256(str(plan.lane_id).encode()).hexdigest()
    return REPO / ".omx/state" / OUTER_V9_GUARD_NAMESPACE / f"{digest}.lock"


@contextmanager
def _outer_v9_lane_dispatch_guard(plan):
    """Serialize pre-egress checks through Modal call-ID registration.

    This namespace is deliberately distinct from modal_train_lane's inner
    ``modal_train_lane_dispatch_guards`` lock.  The Modal child therefore can
    take its own preflight-to-registration lock without self-deadlocking.
    """
    guard_path = _outer_v9_guard_path(plan)
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    guard_fh = guard_path.open("a+")
    locked = False
    try:
        fcntl.flock(guard_fh.fileno(), fcntl.LOCK_EX)
        locked = True
        yield guard_path
    finally:
        try:
            if locked:
                fcntl.flock(guard_fh.fileno(), fcntl.LOCK_UN)
        finally:
            guard_fh.close()


def _ast_dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _run_local_modal_app_definition_preflight() -> None:
    """Validate the actual Modal app source and SDK without provider I/O."""
    app_path = REPO / MODAL_APP_DEFINITION
    try:
        source = app_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(app_path))
    except (OSError, SyntaxError) as exc:
        raise SystemExit("REFUSED: local Modal app definition is unreadable") from exc

    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        if isinstance(target, ast.Name):
            assignments[target.id] = node.value
    expected_literals = {
        "RESULTS_VOL": "comma-train-lane-results",
        "MODAL_PREFLIGHT_HARD_TIMEOUT_SECONDS": 600,
        "MODAL_GPU_CPU_REQUEST_LIMIT": (4.0, 4.0),
        "MODAL_GPU_MEMORY_REQUEST_LIMIT_MB": (32768, 32768),
    }
    for name, expected in expected_literals.items():
        try:
            actual = ast.literal_eval(assignments[name])
        except (KeyError, ValueError, TypeError) as exc:
            raise SystemExit(
                f"REFUSED: local Modal app definition lacks literal {name}"
            ) from exc
        if actual != expected:
            raise SystemExit(
                f"REFUSED: local Modal app definition {name}={actual!r}, expected {expected!r}"
            )

    volume_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _ast_dotted_name(node.func) == "modal.Volume.from_name"
    ]
    if len(volume_calls) != 1:
        raise SystemExit("REFUSED: Modal app must define exactly one canonical Volume lookup")
    volume_call = volume_calls[0]
    if any(keyword.arg is None for keyword in volume_call.keywords):
        raise SystemExit("REFUSED: canonical Modal Volume lookup forbids **kwargs")
    volume_keywords = {kw.arg: kw.value for kw in volume_call.keywords if kw.arg is not None}
    if (
        len(volume_call.args) != 1
        or not isinstance(volume_call.args[0], ast.Name)
        or volume_call.args[0].id != "RESULTS_VOL"
        or set(volume_keywords) != {"create_if_missing"}
        or not isinstance(volume_keywords["create_if_missing"], ast.Constant)
        or volume_keywords["create_if_missing"].value is not False
    ):
        raise SystemExit(
            "REFUSED: canonical Modal Volume must use RESULTS_VOL with "
            "create_if_missing=False"
        )

    endpoints: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and _ast_dotted_name(decorator.func) == "app.function"
            ):
                endpoints.append((node, decorator))
    if len(endpoints) != 1 or endpoints[0][0].name != "run_lane_training_cpu":
        names = [node.name for node, _decorator in endpoints]
        raise SystemExit(
            "REFUSED: Modal app must export only run_lane_training_cpu; "
            f"found {names}"
        )
    endpoint, decorator = endpoints[0]
    if decorator.args or any(keyword.arg is None for keyword in decorator.keywords):
        raise SystemExit(
            "REFUSED: sole Modal endpoint decorator forbids positional args and **kwargs"
        )
    decorator_keywords = {
        kw.arg: kw.value for kw in decorator.keywords if kw.arg is not None
    }
    if set(decorator_keywords) != {"image", "cpu", "memory", "timeout", "volumes"}:
        raise SystemExit(
            "REFUSED: sole Modal endpoint static resource keys drifted: "
            f"{sorted(decorator_keywords)}"
        )
    expected_names = {
        "image": "training_image",
        "cpu": "MODAL_GPU_CPU_REQUEST_LIMIT",
        "memory": "MODAL_GPU_MEMORY_REQUEST_LIMIT_MB",
        "timeout": "MODAL_PREFLIGHT_HARD_TIMEOUT_SECONDS",
    }
    for key, expected_name in expected_names.items():
        value = decorator_keywords[key]
        if not isinstance(value, ast.Name) or value.id != expected_name:
            raise SystemExit(
                f"REFUSED: Modal endpoint {key} must reference {expected_name}"
            )
    volumes = decorator_keywords["volumes"]
    if (
        not isinstance(volumes, ast.Dict)
        or len(volumes.keys) != 1
        or not isinstance(volumes.keys[0], ast.Constant)
        or volumes.keys[0].value != "/modal_results"
        or not isinstance(volumes.values[0], ast.Name)
        or volumes.values[0].id != "results_vol"
    ):
        raise SystemExit(
            "REFUSED: sole Modal endpoint must mount only canonical results_vol"
        )
    endpoint_parameters = {arg.arg for arg in (*endpoint.args.posonlyargs, *endpoint.args.args)}
    required_endpoint_parameters = {
        "requested_gpu",
        "dispatch_stage",
        "dispatch_nonce",
        "cpu_preflight_receipt",
        "dispatch_receipt_label",
    }
    if not required_endpoint_parameters <= endpoint_parameters:
        raise SystemExit(
            "REFUSED: Modal endpoint lacks dispatch-stage parameters: "
            f"{sorted(required_endpoint_parameters - endpoint_parameters)}"
        )

    try:
        import modal

        with_options_parameters = set(
            inspect.signature(modal.Function.with_options).parameters
        )
        volume_from_name_parameters = set(
            inspect.signature(modal.Volume.from_name).parameters
        )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise SystemExit("REFUSED: installed Modal SDK signatures are unavailable") from exc
    required_dynamic_options = {"gpu", "timeout", "retries", "cpu", "memory"}
    if not required_dynamic_options <= with_options_parameters:
        raise SystemExit(
            "REFUSED: installed Modal Function.with_options lacks required controls: "
            f"{sorted(required_dynamic_options - with_options_parameters)}"
        )
    if "create_if_missing" not in volume_from_name_parameters:
        raise SystemExit(
            "REFUSED: installed Modal Volume.from_name lacks create_if_missing"
        )
    print(
        json.dumps(
            {
                "schema": "witness_modal_app_definition_preflight.v1",
                "status": "passed",
                "provider_contacted": False,
                "source": MODAL_APP_DEFINITION,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "exported_endpoint": endpoint.name,
                "static_gpu": False,
                "static_timeout_seconds": 600,
                "dynamic_options": sorted(required_dynamic_options),
            },
            sort_keys=True,
        )
    )


def _run_local_v9_preflight(plan, *, local_cache: Path) -> None:
    """Run the real trainer's read-only input/scorer/DSL preflight locally."""
    preflight_out = (
        REPO / ".omx/preflight/witness_cloud_launcher" / f"{plan.plan_sha256}-must-not-exist"
    )
    if preflight_out.exists():
        raise SystemExit(
            f"REFUSED: local V9 preflight output sentinel already exists: {preflight_out}"
        )
    command = [
        sys.executable,
        "experiments/train_levelset_witness_realized_through_R_torch.py",
        "--gt-cache",
        str(local_cache),
        "--num-pairs",
        str(plan.num_pairs),
        "--epochs",
        str(plan.epochs),
        "--out-dir",
        str(preflight_out),
        "--device",
        "cpu",
        "--dsl-config",
        str(plan.dsl_config),
        "--no-implicit-resume",
        "--expected-segnet-sha256",
        str(plan.segnet_sha256),
        "--expected-posenet-sha256",
        str(plan.posenet_sha256),
        "--preflight-only",
    ]
    if plan.stop_after_epochs is not None:
        command[-1:-1] = ["--stop-after-epochs", str(plan.stop_after_epochs)]
    try:
        proc = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise SystemExit("REFUSED: local V9 trainer preflight could not run") from exc
    if preflight_out.exists():
        raise SystemExit(
            "REFUSED: local V9 trainer preflight mutated its must-not-exist output path"
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-1000:]
        raise SystemExit(
            f"REFUSED: local V9 trainer preflight failed rc={proc.returncode}: {detail}"
        )
    receipt = None
    for line in reversed(proc.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("schema") == "v9_cgauge_torch_preflight.v1":
            receipt = candidate
            break
    if receipt is None:
        raise SystemExit("REFUSED: local V9 trainer preflight omitted its passed receipt")
    expected_end_epoch = (
        plan.epochs if plan.stop_after_epochs is None else plan.stop_after_epochs
    )
    expected_values = {
        "status": "passed",
        "dsl_config": plan.dsl_config,
        "typed_total_epochs": plan.epochs,
        "runtime_stop_after_epochs": plan.stop_after_epochs,
        "runtime_epoch_window": [1, expected_end_epoch],
        "resume_epoch": 0,
        "num_pairs": plan.num_pairs,
    }
    mismatches = [
        f"{name}={receipt.get(name)!r} expected {expected!r}"
        for name, expected in expected_values.items()
        if receipt.get(name) != expected
    ]
    if receipt.get("output_created") is not False:
        mismatches.append(
            f"output_created={receipt.get('output_created')!r} expected False"
        )
    if receipt.get("no_implicit_resume") is not True:
        mismatches.append(
            f"no_implicit_resume={receipt.get('no_implicit_resume')!r} expected True"
        )
    if receipt.get("resume_checkpoint") is not None:
        mismatches.append(
            f"resume_checkpoint={receipt.get('resume_checkpoint')!r} expected None"
        )
    coverage = receipt.get("cuda_v9_port_coverage")
    if not isinstance(coverage, dict):
        mismatches.append("cuda_v9_port_coverage must be an object")
    else:
        if coverage.get("status") != "COMPLETE_1_TO_1":
            mismatches.append(
                "cuda_v9_port_coverage.status must equal COMPLETE_1_TO_1"
            )
        if coverage.get("blockers") != []:
            mismatches.append("cuda_v9_port_coverage.blockers must be empty")
    scorer_load = receipt.get("scorer_constructor_load")
    if not isinstance(scorer_load, dict):
        mismatches.append("scorer_constructor_load must be an object")
    else:
        if scorer_load.get("status") != "passed" or scorer_load.get("device") != "cpu":
            mismatches.append("scorer_constructor_load must be passed on cpu")
        networks = scorer_load.get("networks")
        if not isinstance(networks, dict) or set(networks) != {"segnet", "posenet"}:
            mismatches.append(
                "scorer_constructor_load.networks must contain exactly segnet and posenet"
            )
        else:
            for name, row in networks.items():
                if (
                    not isinstance(row, dict)
                    or row.get("eval") is not True
                    or row.get("frozen") is not True
                    or not isinstance(row.get("class"), str)
                    or not row["class"]
                ):
                    mismatches.append(f"scorer_constructor_load.{name} is not eval/frozen")
    scorer_custody = receipt.get("scorer_custody")
    expected_scorers = {
        "segnet": str(plan.segnet_sha256).lower(),
        "posenet": str(plan.posenet_sha256).lower(),
    }
    if not isinstance(scorer_custody, dict) or set(scorer_custody) != set(expected_scorers):
        mismatches.append("scorer_custody must contain exactly segnet and posenet")
    else:
        for name, expected_sha256 in expected_scorers.items():
            row = scorer_custody[name]
            if not isinstance(row, dict):
                mismatches.append(f"scorer_custody.{name} must be an object")
                continue
            actual = str(row.get("sha256", "")).lower()
            expected_echo = str(row.get("expected_sha256", "")).lower()
            if actual != expected_sha256 or expected_echo != expected_sha256:
                mismatches.append(f"scorer_custody.{name} does not match plan SHA-256")
            if row.get("sha_authority") != "PLAN_EXPECTED_MATCH":
                mismatches.append(
                    f"scorer_custody.{name}.sha_authority is not PLAN_EXPECTED_MATCH"
                )
    if mismatches:
        raise SystemExit(
            "REFUSED: local V9 trainer preflight receipt mismatch: " + "; ".join(mismatches)
        )
    print(
        json.dumps(
            {
                "schema": "witness_local_v9_preflight_scope.v1",
                "status": "passed",
                "scope": "input_scorer_dsl_only",
                "remote_resume_requested": plan.resume_from is not None,
                "remote_resume_path": plan.resume_from,
                "remote_resume_content_validated": False,
                "remote_resume_authority": (
                    "bounded same-image provider CPU preflight before H100! claim"
                ),
            },
            sort_keys=True,
        )
    )


def _modal_asset_is_reusable(plan, *, local_size: int) -> bool:
    """Reject output collisions and inspect one asset without provider mutation."""
    if plan.gt_cache_sha256 is None or not plan.asset_stage_argv:
        raise SystemExit("REFUSED: executable plan is missing SHA-addressed GT asset custody")
    remote_asset = plan.asset_stage_argv[-1]
    expected_asset = f"assets/v9_cgauge/gt_{plan.gt_cache_sha256}.npz"
    if remote_asset != expected_asset:
        raise SystemExit("REFUSED: plan GT asset destination is not exact SHA-addressed custody")
    try:
        import modal

        volume = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=False)

        def _listdir_or_empty(path: str) -> list:
            # A modal NotFoundError means the path does not exist on the volume
            # yet. For a fresh (non-resume) label this is the NORMAL no-collision
            # state and for an unstaged SHA-asset it means "stage it" -- both are
            # the empty-listing semantics the callers below already handle.
            # Re-raising here mislabels a missing fresh-label dir as the
            # "refusing blind upload" money-safety refusal (2026-07-12 bug).
            # Match by class name (no modal.exception import) so this stays
            # robust across modal versions AND monkeypatched test fakes; any
            # OTHER failure (auth/offline/etc.) still re-raises to the real
            # fail-closed refusal below.
            try:
                return list(volume.listdir(path))
            except Exception as exc:
                if type(exc).__name__ == "NotFoundError":
                    return []
                raise

        output_path = f"{plan.label}/output"
        output_entries = _listdir_or_empty(plan.label)
        if any(
            str(getattr(entry, "path", "")).lstrip("/").rstrip("/")
            == plan.label
            for entry in output_entries
        ):
            raise SystemExit(
                f"REFUSED: remote label custody root is not a directory: {plan.label}"
            )
        output_exists = any(
            (path := str(getattr(entry, "path", "")).lstrip("/").rstrip("/"))
            == output_path
            or path.startswith(output_path + "/")
            for entry in output_entries
        )
        if plan.resume_from is None:
            if output_exists:
                raise SystemExit(
                    "REFUSED: non-resume label already has remote output custody: "
                    f"/modal_results/{output_path}"
                )
        else:
            remote_resume = str(plan.resume_from).removeprefix("/modal_results/")
            if output_exists and not remote_resume.startswith(output_path + "/"):
                raise SystemExit(
                    "REFUSED: populated resume destination requires checkpoint under "
                    f"the same output lineage /modal_results/{output_path}/"
                )
            resume_entries = _listdir_or_empty(remote_resume)
            exact_resume_entries = [
                entry
                for entry in resume_entries
                if str(getattr(entry, "path", "")).lstrip("/") == remote_resume
            ]
            if not exact_resume_entries:
                raise SystemExit(
                    "REFUSED: exact remote resume checkpoint is missing: "
                    f"{plan.resume_from}"
                )
            if len(exact_resume_entries) != 1:
                raise SystemExit(
                    "REFUSED: exact remote resume checkpoint lookup is ambiguous"
                )
            resume_entry = exact_resume_entries[0]
            try:
                resume_is_file = int(resume_entry.type) == 1
            except (AttributeError, TypeError, ValueError) as exc:
                raise SystemExit(
                    "REFUSED: remote resume checkpoint lookup omitted a usable file type"
                ) from exc
            if not resume_is_file:
                raise SystemExit("REFUSED: remote resume checkpoint is not a regular file")
            try:
                resume_size = int(resume_entry.size)
            except (AttributeError, TypeError, ValueError) as exc:
                raise SystemExit(
                    "REFUSED: remote resume checkpoint lookup omitted a usable byte size"
                ) from exc
            if resume_size <= 0:
                raise SystemExit("REFUSED: remote resume checkpoint must be a nonzero file")
            print(
                json.dumps(
                    {
                        "schema": "witness_remote_resume_presence.v1",
                        "path": plan.resume_from,
                        "bytes": resume_size,
                        "regular_file": True,
                        "expected_sha256": plan.resume_sha256,
                        "checkpoint_content_validated": False,
                        "checkpoint_content_authority": (
                            "bounded same-image provider CPU preflight before H100! claim"
                        ),
                    },
                    sort_keys=True,
                )
            )
        entries = _listdir_or_empty(remote_asset)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            "REFUSED: read-only Modal Volume asset lookup failed; refusing blind upload"
        ) from exc
    exact_entries = [
        entry
        for entry in entries
        if str(getattr(entry, "path", "")).lstrip("/") == remote_asset
    ]
    if not exact_entries:
        if entries:
            raise SystemExit("REFUSED: Modal asset lookup returned an unexpected path")
        return False
    if len(exact_entries) != 1:
        raise SystemExit("REFUSED: Modal asset lookup returned ambiguous exact-path rows")
    entry = exact_entries[0]
    try:
        is_regular_file = int(entry.type) == 1
    except (AttributeError, TypeError, ValueError) as exc:
        raise SystemExit("REFUSED: Modal asset lookup omitted a usable file type") from exc
    if not is_regular_file:
        raise SystemExit("REFUSED: Modal SHA-addressed asset path is not a regular file")
    try:
        remote_size = int(entry.size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SystemExit("REFUSED: Modal asset lookup omitted a usable byte size") from exc
    return remote_size == local_size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=("modal", "aws", "gcp"), default="modal")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--label", default="v9-cgauge-cuda-438")
    ap.add_argument("--gpu", default=EXACT_H100_GPU)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument(
        "--dsl-config",
        choices=V9_DSL_CONFIGS,
        default=DEFAULT_DSL_CONFIG,
        help=(
            "Named typed-DSL config the remote driver compiles "
            "(smoke_regime = #438 NON-PROMOTABLE cost-regime variant)"
        ),
    )
    ap.add_argument(
        "--stop-after-epochs",
        type=int,
        default=None,
        help=(
            "Explicit runtime stop (1..3). Default None = timeout-stop: the "
            "1500-second child timeout is the stop"
        ),
    )
    ap.add_argument(
        "--torch-compile-mode",
        choices=TORCH_COMPILE_MODES,
        default=DEFAULT_TORCH_COMPILE_MODE,
        help=(
            "Backend Inductor mode threaded to the trainer (runtime control, "
            "never typed-DSL science). MEASURED 2026-07-15 H100 r5 rc=124: "
            "max-autotune ate the whole time-boxed window; this launcher's "
            "1500s dispatches default to 'default'"
        ),
    )
    ap.add_argument("--resume-from")
    ap.add_argument("--resume-sha256")
    ap.add_argument("--gt-cache-sha256", help="Required exact SHA-256 for custody and staging")
    ap.add_argument("--expected-plan-sha256", help="Required exact hash of the reviewed plan")
    ap.add_argument("--output")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--operator-go-token")
    args = ap.parse_args(argv)

    if (
        args.gt_cache_sha256 is not None
        and args.gt_cache_sha256.lower() != TASK438_GT_CACHE_SHA256
    ):
        raise SystemExit(
            "REFUSED: --gt-cache-sha256 must equal the canonical Task438 n600 digest"
        )

    source_git_head = _main_git_head()
    segnet_sha256 = _required_file_sha256(
        REPO / SEGNET_WEIGHTS, custody_name="SegNet"
    )
    posenet_sha256 = _required_file_sha256(
        REPO / POSENET_WEIGHTS, custody_name="PoseNet"
    )
    plan = build_plan(
        provider=args.provider, source_git_head=source_git_head,
        gt_cache=args.gt_cache, label=args.label, gpu=args.gpu,
        epochs=args.epochs, num_pairs=args.num_pairs,
        dsl_config=args.dsl_config,
        torch_compile_mode=args.torch_compile_mode,
        stop_after_epochs=args.stop_after_epochs,
        resume_from=args.resume_from,
        resume_sha256=args.resume_sha256,
        gt_cache_sha256=args.gt_cache_sha256,
        gt_cache_bytes=TASK438_GT_CACHE_BYTES,
        segnet_sha256=segnet_sha256,
        posenet_sha256=posenet_sha256,
    )
    stage_readiness = None
    if not args.execute and plan.execution_allowed:
        stage_readiness = _verify_local_asset_stage_contract(plan)
    payload = plan.to_dict()
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("asset_stage:", render_command(plan.asset_stage_argv))
    print("dispatch:", render_command(plan.dispatch_argv))
    print("harvest:", render_command(plan.harvest_argv))
    if stage_readiness is not None:
        print(json.dumps(stage_readiness, sort_keys=True))
    if not args.execute:
        print("PLAN ONLY: no provider contacted and no paid resource dispatched.")
        return 0
    if args.operator_go_token != OPERATOR_TOKEN:
        raise SystemExit("REFUSED: --execute requires --operator-go-token GO-CLOUD-381")
    if args.expected_plan_sha256 != plan.plan_sha256:
        raise SystemExit("REFUSED: --execute requires the exact reviewed --expected-plan-sha256")
    if not plan.execution_allowed:
        raise SystemExit(f"REFUSED: provider {plan.provider} is {plan.status}, not executable")
    _require_modal_image_cache_policy()
    _require_clean_main_worktree(plan.source_git_head)
    local_cache = REPO / plan.local_gt_cache
    if not local_cache.is_file():
        raise SystemExit(f"REFUSED: asset missing: {local_cache}")
    actual_sha256 = _sha256_file(local_cache)
    if actual_sha256 != plan.gt_cache_sha256:
        raise SystemExit(
            f"REFUSED: GT cache SHA-256 mismatch: {actual_sha256} != {plan.gt_cache_sha256}"
        )
    print(
        json.dumps(
            _verify_local_asset_stage_contract(plan, actual_sha256=actual_sha256),
            sort_keys=True,
        )
    )
    if _required_file_sha256(REPO / SEGNET_WEIGHTS, custody_name="SegNet") != plan.segnet_sha256:
        raise SystemExit("REFUSED: SegNet SHA-256 changed after plan review")
    if _required_file_sha256(REPO / POSENET_WEIGHTS, custody_name="PoseNet") != plan.posenet_sha256:
        raise SystemExit("REFUSED: PoseNet SHA-256 changed after plan review")
    # Close source races around the potentially expensive local cache/scorer
    # reads before any provider read or mutation.
    _require_clean_main_worktree(plan.source_git_head)
    # The label-independent outer guard closes the no-claim/read/upload/spawn
    # race across two local launchers. Keep it until the Modal CLI returns,
    # after modal_train_lane has durably registered its detached call ID.
    with _outer_v9_lane_dispatch_guard(plan):
        _require_no_active_v9_claim(plan)
        _run_local_modal_app_definition_preflight()
        _run_local_v9_preflight(plan, local_cache=local_cache)
        # A competing dispatcher outside this launcher may claim the lane
        # during the local scorer/GT preflight. Refresh before provider read.
        _require_no_active_v9_claim(plan)
        # The exact filename carries the local SHA, and the read-only lookup
        # also requires equal bytes. The CPU preflight then rehashes mounted
        # bytes before it can claim an H100!, closing content-level trust.
        if not _modal_asset_is_reusable(
            plan,
            local_size=local_cache.stat().st_size,
        ):
            raise SystemExit(
                "REFUSED: exact SHA-addressed GT asset is absent from the canonical "
                "Modal volume. Automatic `modal volume put` is disabled because this "
                "plan has no staging-specific authorization, timeout, retention, or "
                "enforced cost cap. Stage the plan's asset_stage_argv explicitly under "
                "separate operator authorization, then rerun for read-only verification."
            )
        print("ASSET REUSE: exact SHA-addressed GT path and byte size already present.")
        # Local/provider preflights can outlive a concurrent serializer
        # edit/commit. Recheck literally adjacent to the command that snapshots
        # source; execution never uploads the asset implicitly.
        _require_no_active_v9_claim(plan)
        _require_clean_main_worktree(plan.source_git_head)
        subprocess.run(
            plan.dispatch_argv,
            cwd=REPO,
            check=True,
            timeout=plan.invocation_timeout_seconds,
        )
    print("DISPATCHED: use the exact call-id-scoped harvest command printed by the plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
