#!/usr/bin/env python3
# ruff: noqa: E402
# SPDX-License-Identifier: MIT
"""Resumable stage runner for the locked DDM AT1x scorer atlas.

This runner does not dispatch, train, or silently evaluate.  ``environment``
is the only stage that invokes a package manager.  ``calibration`` consumes a
completed official-run report and its explicit custody paths; the official run
itself remains a separate, operator-visible command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.scorer_analytic_atlas import SourceHashStamp
from tac.optimization.scorer_atlas_materialization import (
    AtlasMaterializationError,
    build_atlas_manifest,
    build_calibration_blocker_receipt,
    build_calibration_receipt,
    build_environment_receipt,
    certify_tree,
    derive_network_closed_forms,
    file_identity,
    payload_sha256,
    require_locked_inventory,
    require_ssd_environment,
    sha256_file,
    shared_receipt_contract,
    storage_preflight,
    validate_and_contract_sidecars,
    write_factor_shards,
    write_immutable_receipt,
)
from tac.optimization.scorer_module_inventory import (
    PACKAGE_IMPORT_NAMES,
    build_inventory,
)
from tac.optimization.scorer_module_inventory import (
    wrap_receipt as wrap_inventory_receipt,
)

DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_CAMPAIGN = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json"
)
DEFAULT_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/evidence/"
    "ddm_e2_pose_stream_and_doctrine_export_20260723/"
    "upstream_harness/submission/archive.zip"
)
DEFAULT_VJP_INVENTORY = (
    ROOT / ".omx/research/ddm_at1_scorer_analytic_atlas_20260723T194312Z/scorer_module_inventory_receipt.json"
)
DEFAULT_V19_RECEIPT = (
    ROOT / ".omx/research/ddm_v19_pure_priced_objective_20260723T041500Z/ddm_v19_pure_priced_objective_receipt.json"
)
E2_ARCHIVE_SHA256 = "8891012e4019e474d1e8ae7578104d74f27c25838c7b68a3798af35853469819"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AtlasMaterializationError(f"expected JSON object: {path}")
    return value


def _python(environment: Path) -> Path:
    candidates = (environment / "bin" / "python", environment / "Scripts" / "python.exe")
    selected = next((path for path in candidates if path.is_file()), None)
    if selected is None:
        raise AtlasMaterializationError(f"environment has no Python: {environment}")
    return selected.absolute()


def _run(argv: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )


def _locked_environment(environment: Path) -> dict[str, str]:
    values = os.environ.copy()
    values.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(environment),
            "UV_LINK_MODE": "copy",
        }
    )
    return values


def _package_versions(python: Path) -> dict[str, str]:
    program = (
        "import importlib,json;"
        f"m={json.dumps(PACKAGE_IMPORT_NAMES)};"
        "print(json.dumps({p:str(getattr(importlib.import_module(i),'__version__','UNKNOWN'))"
        " for p,i in m.items()},sort_keys=True))"
    )
    result = _run([str(python), "-c", program], cwd=ROOT)
    return json.loads(result.stdout)


def stage_environment(args: argparse.Namespace) -> None:
    environment = require_ssd_environment(args.environment)
    preflight = storage_preflight(environment, required_free_bytes=args.required_free_bytes)
    receipt_path = args.output_dir / "stage_environment.json"
    if receipt_path.exists():
        prior = _json(receipt_path)
        current_tree = certify_tree(environment)
        if current_tree["tree_sha256"] != prior["tree_certificate"]["tree_sha256"]:
            raise AtlasMaterializationError("locked environment drifted after certification; rebuild at a new path")
        return
    environment.parent.mkdir(parents=True, exist_ok=True)
    sync_argv = ["uv", "sync", "--frozen", "--group", "cpu", "--python", "3.11"]
    try:
        sync = _run(
            sync_argv,
            cwd=args.upstream_root,
            environment=_locked_environment(environment),
        )
    except subprocess.CalledProcessError as error:
        rows = []
        for stream, text in (("stdout", error.stdout), ("stderr", error.stderr)):
            rows.extend(
                {
                    "stream": stream,
                    "line_number": line_number,
                    "text": line,
                }
                for line_number, line in enumerate((text or "").splitlines(), start=1)
            )
        failure = {
            "schema": "ddm_at1x_locked_environment_failure.v1",
            **shared_receipt_contract(scoped_verdict="locked environment materialization failed"),
            "status": "BLOCKED_LOCK_MATERIALIZATION_FAILED",
            "argv": sync_argv,
            "cwd": str(args.upstream_root),
            "environment": {
                "UV_PROJECT_ENVIRONMENT": str(environment),
                "UV_LINK_MODE": "copy",
            },
            "returncode": error.returncode,
            "per_package_failure_rows": rows,
            "storage_preflight": preflight,
            "environment_deleted": False,
        }
        write_immutable_receipt(args.output_dir / "stage_environment_frozen_failure.json", failure)
        raise
    python = _python(environment)
    uv_version = _run(["uv", "--version"], cwd=args.upstream_root).stdout.strip()
    receipt = build_environment_receipt(
        environment=environment,
        upstream_root=args.upstream_root,
        python_path=python,
        uv_version=uv_version,
        package_versions=_package_versions(python),
        preflight=preflight,
        tree_certificate=certify_tree(environment),
    )
    receipt["sync"]["stdout_sha256"] = payload_sha256(sync.stdout)
    receipt["sync"]["stderr_sha256"] = payload_sha256(sync.stderr)
    rejected_locked = args.output_dir / "stage_environment_failure.json"
    receipt["locked_validation_attempt"] = (
        {
            "status": "REJECTED_STALE_LOCK_METADATA",
            "receipt": file_identity(rejected_locked),
            "fallback": ("uv --frozen consumed the supplied uv.lock verbatim without mutating upstream"),
        }
        if rejected_locked.exists()
        else {"status": "NOT_RUN"}
    )
    write_immutable_receipt(receipt_path, receipt)


def stage_inventory(args: argparse.Namespace) -> None:
    environment_receipt = _json(args.output_dir / "stage_environment.json")
    python = Path(environment_receipt["sync"]["python"]).absolute()
    if python != Path(sys.executable).resolve() and not args.internal_locked:
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "inventory",
            *args.forwarded_common,
            "--internal-locked",
        ]
        _run(command, cwd=ROOT, environment={**os.environ, "PYTHONPATH": str(SRC)})
        return
    inventory = build_inventory(
        upstream_root=args.upstream_root,
        created_at_utc=args.created_at_utc,
    )
    require_locked_inventory(inventory)
    inventory.update(
        shared_receipt_contract(scoped_verdict=("locked scorer module inventory with zero package version drift"))
    )
    write_immutable_receipt(
        args.output_dir / "stage_inventory.json",
        wrap_inventory_receipt(inventory),
    )


def _source_stamp(path: Path, source_id: str) -> SourceHashStamp:
    identity = file_identity(path)
    return SourceHashStamp(
        source_id=source_id,
        path=str(identity["path"]),
        bytes=int(identity["bytes"]),
        sha256=str(identity["sha256"]),
        validity_horizon="exact hash equality at consumption; rederive on mismatch",
    )


def stage_closed_forms(args: argparse.Namespace) -> None:
    environment_receipt = _json(args.output_dir / "stage_environment.json")
    python = Path(environment_receipt["sync"]["python"]).absolute()
    if python != Path(sys.executable).resolve() and not args.internal_locked:
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "closed-forms",
            *args.forwarded_common,
            "--internal-locked",
        ]
        _run(command, cwd=ROOT, environment={**os.environ, "PYTHONPATH": str(SRC)})
        return
    inventory = _json(args.output_dir / "stage_inventory.json")
    require_locked_inventory(inventory)
    from tac.scorer import load_default_scorers

    posenet, segnet = load_default_scorers(args.upstream_root, device="cpu")
    checkpoint_paths = {
        "posenet": args.upstream_root / "models" / "posenet.safetensors",
        "segnet": args.upstream_root / "models" / "segnet.safetensors",
    }
    factors = derive_network_closed_forms(
        networks={"posenet": posenet, "segnet": segnet},
        inventory=inventory,
        checkpoint_sha256s={name: sha256_file(path) for name, path in checkpoint_paths.items()},
        package_version_set_sha256=environment_receipt["package_version_set_sha256"],
        source_hashes={name: (_source_stamp(path, f"{name}_checkpoint"),) for name, path in checkpoint_paths.items()},
    )
    factor_root = args.output_dir / "factor_shards"
    index = write_factor_shards(
        factors=factors,
        shard_root=factor_root,
        reconstruction_command=(f"{python} {Path(__file__).resolve()} closed-forms " + " ".join(args.forwarded_common)),
    )
    write_immutable_receipt(args.output_dir / "stage_closed_forms.json", index)


def stage_gaze(args: argparse.Namespace) -> None:
    campaign = _json(args.campaign_receipt)
    vjp_inventory = _json(args.vjp_inventory_receipt)
    vjp_sources = vjp_inventory["body"]["source_strata"]["B_imported_library_sources"]
    observed_versions = {package: str(row["version"]) for package, row in sorted(vjp_sources["observed"].items())}
    observed_version_sha256 = payload_sha256(observed_versions)
    v19_receipt = _json(args.v19_receipt)
    exact_v19_pair_ids = v19_receipt["typed_config"]["pair_ids"]
    atlas = validate_and_contract_sidecars(
        campaign=campaign,
        version_stamp_id=f"at1-observed-vjp-env-{observed_version_sha256[:16]}",
        version_set_sha256=observed_version_sha256,
        exact_v19_pair_ids=exact_v19_pair_ids,
        checkpoint_dir=args.output_dir / "gaze_pair_checkpoints",
        process_pair_ids=range(args.pair_start, args.pair_stop),
        verify_archive_hashes=False,
    )
    atlas["vjp_producer_environment"] = {
        "versions": observed_versions,
        "version_set_sha256": observed_version_sha256,
        "inventory_receipt": file_identity(args.vjp_inventory_receipt),
        "version_drift_vs_lock": vjp_sources["version_drift"],
    }
    atlas["v19_join_source"] = {
        "receipt": file_identity(args.v19_receipt),
        "exact_pair_ids": list(exact_v19_pair_ids),
    }
    if not atlas["full_n600_coverage"]:
        write_immutable_receipt(
            args.output_dir / f"stage_gaze_shard_{args.pair_start:04d}_{args.pair_stop:04d}.json",
            {
                "schema": "ddm_at1x_gaze_shard_stage.v1",
                **shared_receipt_contract(scoped_verdict="pair-checkpoint shard of settled n600 gaze"),
                "pair_start": args.pair_start,
                "pair_stop": args.pair_stop,
                "pair_count": atlas["pair_count"],
                "tensor_index_count": atlas["tensor_index_count"],
                "checkpoint_dir": atlas["resume"]["checkpoint_dir"],
                "campaign": file_identity(args.campaign_receipt),
                "vjp_producer_environment": atlas["vjp_producer_environment"],
                "v19_join_source": atlas["v19_join_source"],
            },
        )
        return
    atlas_path = args.output_dir / "gaze_contraction_atlas.json"
    write_immutable_receipt(atlas_path, atlas)
    write_immutable_receipt(
        args.output_dir / "stage_gaze.json",
        {
            "schema": "ddm_at1x_gaze_stage.v1",
            **shared_receipt_contract(scoped_verdict="small index for settled n600 gaze contractions"),
            "pair_count": atlas["pair_count"],
            "tensor_index_count": atlas["tensor_index_count"],
            "atlas": file_identity(atlas_path),
            "campaign": file_identity(args.campaign_receipt),
            "recomputation": "NONE; settled VJP sidecars validated and contracted",
        },
    )


def stage_calibration(args: argparse.Namespace) -> None:
    if sha256_file(args.archive) != E2_ARCHIVE_SHA256:
        raise AtlasMaterializationError("E2 archive SHA-256 does not match the pin")
    stdout = args.calibration_stdout or args.calibration_report
    stderr = args.calibration_stderr or args.calibration_report
    if stdout is None or stderr is None:
        raise AtlasMaterializationError("calibration requires explicit stdout/stderr custody")
    runtime = args.calibration_runtime or args.upstream_root / "evaluate.sh"
    upstream_files = {
        name: file_identity(args.upstream_root / relative)
        for name, relative in {
            "evaluate_sh": "evaluate.sh",
            "evaluate_py": "evaluate.py",
            "uv_lock": "uv.lock",
            "video_names": "public_test_video_names.txt",
            "posenet_checkpoint": "models/posenet.safetensors",
            "segnet_checkpoint": "models/segnet.safetensors",
        }.items()
    }
    upstream_git_sha = _run(["git", "rev-parse", "HEAD"], cwd=args.upstream_root).stdout.strip()
    execution_environment = {
        "UV_PROJECT_ENVIRONMENT": str(args.environment),
        "UV_LINK_MODE": "copy",
        "PYTHON": str(_python(args.environment)),
        "PATH_PREFIX": str(_python(args.environment).parent),
    }
    execution_argv = json.loads(args.calibration_argv_json)
    if args.calibration_exit_code != 0:
        receipt = build_calibration_blocker_receipt(
            stderr_text=stderr.read_text(encoding="utf-8"),
            exit_code=args.calibration_exit_code,
            argv=execution_argv,
            environment=execution_environment,
            archive=file_identity(args.archive),
            runtime=file_identity(runtime),
            upstream={
                "path": str(args.upstream_root.resolve()),
                "git_sha": upstream_git_sha,
                "files": upstream_files,
                "file_set_sha256": payload_sha256(upstream_files),
            },
            stdout=file_identity(stdout),
            stderr=file_identity(stderr),
            wallclock_seconds=args.wallclock_seconds,
        )
        write_immutable_receipt(args.output_dir / "stage_calibration_blocked.json", receipt)
        return
    if args.calibration_report is None:
        raise AtlasMaterializationError("--calibration-report is required")
    parsed = args.calibration_report.read_text(encoding="utf-8")
    receipt = build_calibration_receipt(
        parsed=parsed,
        argv=execution_argv,
        environment=execution_environment,
        archive=file_identity(args.archive),
        runtime=file_identity(runtime),
        upstream={
            "path": str(args.upstream_root.resolve()),
            "git_sha": upstream_git_sha,
            "files": upstream_files,
            "file_set_sha256": payload_sha256(upstream_files),
        },
        stdout=file_identity(stdout),
        stderr=file_identity(stderr),
        report=file_identity(args.calibration_report),
        wallclock_seconds=args.wallclock_seconds,
    )
    receipt["scoped_verdict"] = (
        "locked scorer-only official evaluate.py replay over the prior "
        "SHA-certified E2 inflation; full evaluate.sh remains separately gated"
    )
    receipt["full_harness_passed"] = False
    if args.inflation_manifest is not None:
        receipt["custody"]["inflation_manifest"] = file_identity(args.inflation_manifest)
    write_immutable_receipt(args.output_dir / "stage_calibration.json", receipt)


def stage_manifest(args: argparse.Namespace) -> None:
    environment = _json(args.output_dir / "stage_environment.json")
    factors = _json(args.output_dir / "stage_closed_forms.json")
    gaze_stage = _json(args.output_dir / "stage_gaze.json")
    contraction = _json(Path(gaze_stage["atlas"]["path"]))
    calibration_path = args.output_dir / "stage_calibration.json"
    blocked_calibration_path = args.output_dir / "stage_calibration_blocked.json"
    if calibration_path.exists():
        calibration = _json(calibration_path)
    elif blocked_calibration_path.exists():
        calibration = _json(blocked_calibration_path)
    else:
        calibration = None
    manifest = build_atlas_manifest(
        environment_receipt=environment,
        factor_index=factors,
        contraction_atlas=contraction,
        calibration_receipt=calibration,
        reconstruction_commands=(
            f"uv sync --frozen --group cpu --python 3.11 # UV_PROJECT_ENVIRONMENT={args.environment}",
            f"{_python(args.environment)} {Path(__file__).resolve()} closed-forms " + " ".join(args.forwarded_common),
            f"{sys.executable} {Path(__file__).resolve()} gaze " + " ".join(args.forwarded_common),
        ),
    )
    if blocked_calibration_path.exists():
        manifest["full_locked_harness_blocker"] = file_identity(blocked_calibration_path)
    write_immutable_receipt(args.output_dir / "atlas_manifest.json", manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "stage",
        choices=("environment", "inventory", "closed-forms", "gaze", "calibration", "manifest"),
    )
    result.add_argument("--environment", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    result.add_argument("--campaign-receipt", type=Path, default=DEFAULT_CAMPAIGN)
    result.add_argument("--vjp-inventory-receipt", type=Path, default=DEFAULT_VJP_INVENTORY)
    result.add_argument("--v19-receipt", type=Path, default=DEFAULT_V19_RECEIPT)
    result.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    result.add_argument("--required-free-bytes", type=int, default=20 * 1024**3)
    result.add_argument("--pair-start", type=int, default=0)
    result.add_argument("--pair-stop", type=int, default=600)
    result.add_argument("--created-at-utc")
    result.add_argument("--calibration-report", type=Path)
    result.add_argument("--calibration-stdout", type=Path)
    result.add_argument("--calibration-stderr", type=Path)
    result.add_argument("--calibration-runtime", type=Path)
    result.add_argument(
        "--inflation-manifest",
        type=Path,
        help="SHA-certified manifest for a pre-existing scorer input inflation",
    )
    result.add_argument("--calibration-argv-json", default="[]")
    result.add_argument("--calibration-exit-code", type=int, default=0)
    result.add_argument("--wallclock-seconds", type=float, default=0.0)
    result.add_argument("--internal-locked", action="store_true", help=argparse.SUPPRESS)
    return result


def main() -> int:
    args = parser().parse_args()
    args.environment = args.environment.resolve()
    args.output_dir = args.output_dir.resolve()
    args.upstream_root = args.upstream_root.resolve()
    args.campaign_receipt = args.campaign_receipt.resolve()
    args.vjp_inventory_receipt = args.vjp_inventory_receipt.resolve()
    args.v19_receipt = args.v19_receipt.resolve()
    args.archive = args.archive.resolve()
    if args.inflation_manifest is not None:
        args.inflation_manifest = args.inflation_manifest.resolve()
    if args.stage == "inventory" and not args.created_at_utc:
        raise AtlasMaterializationError("inventory requires --created-at-utc")
    if not (0 <= args.pair_start < args.pair_stop <= 600):
        raise AtlasMaterializationError("pair shard must satisfy 0 <= start < stop <= 600")
    args.forwarded_common = [
        "--environment",
        str(args.environment),
        "--output-dir",
        str(args.output_dir),
        "--upstream-root",
        str(args.upstream_root),
        "--campaign-receipt",
        str(args.campaign_receipt),
        "--vjp-inventory-receipt",
        str(args.vjp_inventory_receipt),
        "--v19-receipt",
        str(args.v19_receipt),
        "--archive",
        str(args.archive),
        "--required-free-bytes",
        str(args.required_free_bytes),
        "--pair-start",
        str(args.pair_start),
        "--pair-stop",
        str(args.pair_stop),
    ]
    if args.created_at_utc:
        args.forwarded_common.extend(["--created-at-utc", args.created_at_utc])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "environment": stage_environment,
        "inventory": stage_inventory,
        "closed-forms": stage_closed_forms,
        "gaze": stage_gaze,
        "calibration": stage_calibration,
        "manifest": stage_manifest,
    }
    stages[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
