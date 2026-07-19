"""Fail-closed materializer for the 2026-07-19 duty-queue fire tickets.

This module deliberately creates *evidence packages*, never runnable training
commands.  It keeps the treatment declaration useful while the ep725/v9c3 and
#518 readiness joins remain absent from the current tree.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from tac.witness_dsl.curriculum_dsl import TRAINER_PATH, real_trainer_flags
from tac.witness_dsl.lever_registry import resolve_composable_lever

SCHEMA = "duty_queue_fire_ticket.v1"
DEFAULT_MAIN_REPO = Path("/Users/adpena/Projects/pact")
DEFAULT_MAIN_SOURCE_REPO = Path("/Users/adpena/Projects/pact")
DEFAULT_OUT_DIR = Path(".omx/research/duty_queue_fire_tickets_20260719")
EXPECTED_CHECKPOINT_BYTES = 460448
EXPECTED_CHECKPOINT_SHA256 = "b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef"
EXPECTED_BEST_JSON_SHA256 = "66950d15bf8575eb4bae28c39c0276b0960f8e14364ed45da24b30b960f27d9e"
EXPECTED_BEST_DSEG = 0.003457972208658854
EXPECTED_BEST_EPOCH = 725
EXPECTED_CURVELET_FIRE_SHA256 = "cdfb05f4dbd0fc825cd6729f8bd2f71a47e1780af5164702213b00e575311bac"
_TICKETS = (
    ("01_dseg_aware_taper", "DsegAwareTaper", "78.9%"),
    ("02_horizon_weighted_margin", "HorizonWeightedMargin", "47.3%"),
    ("03_step_native_activation", "StepNativeActivation", "34.2%"),
)
_MAIN_SOURCE_AUTHORITY_INPUTS = (
    ("curriculum_dsl", "src/tac/witness_dsl/curriculum_dsl.py"),
    ("spec_v9_cgauge", "src/tac/witness_dsl/spec_v9_cgauge.py"),
    ("lever_registry", "src/tac/witness_dsl/lever_registry.py"),
    ("trainer", "experiments/train_levelset_witness_realized_through_R_mlx.py"),
    ("ticket_497_fire_script", "tools/fire_curvelet_matched_bytes_ab_p0_497.py"),
    ("governed_launcher", "tools/launch_witness_run.py"),
    ("durable_daemon", "tools/spawn_durable_daemon.py"),
    ("operator_authorize", "tools/operator_authorize.py"),
    ("safe_run", "tools/safe_run.py"),
    ("readiness_gate", "tools/witness_launch_readiness_gate.py"),
    (
        "beta2_admissible_window_equation",
        "src/tac/canonical_equations/cgauge_parametrization_optima_20260711.py",
    ),
    (
        "beta2_rewarmup_length_equation",
        "src/tac/canonical_equations/adam_v_variance_warmup_20260717.py",
    ),
    ("ncde_module", "src/tac/witness_control/ncde_trajectory.py"),
    ("ticket_518_memo", ".omx/research/p0_resume_warmup_geometry_build_20260717.md"),
    ("frontier_pointer", "reports/latest.md"),
)
_ISO_CONTRASTS = {
    "DsegAwareTaper": {
        "wrapper": "compile_v9_cgauge_432_taper_off_launch_config",
        "operation": "REMOVE_COMPLETE_CONTROL_LEVER",
        "declared_lever_role": "CONTROL",
    },
    "HorizonWeightedMargin": {
        "wrapper": "compile_v9_cgauge_432_horizon_iso_launch_config",
        "operation": "ADD_TREATMENT_LEVER",
        "declared_lever_role": "TREATMENT",
    },
    "StepNativeActivation": {
        "wrapper": "compile_v9_cgauge_432_step_iso_launch_config",
        "operation": "REPLACE_TREATMENT_LEVER_ENDPOINT",
        "declared_lever_role": "TREATMENT",
    },
}


@dataclass(frozen=True)
class TicketSummary:
    ticket_id: str
    status: str
    path: str
    treatment_delta_hash: str | None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n")


def _artifact_manifest(directory: Path) -> dict[str, Any]:
    files = []
    excluded = {"artifact_manifest.json", "summary_manifest.json"}
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name not in excluded):
        files.append(
            {"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    return {"schema": "duty_ticket_artifact_manifest.v1", "files": files}


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _git_head(repo: Path) -> dict[str, Any]:
    """Resolve a checkout HEAD without mutating or refreshing repository state."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD^{commit}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {
            "resolved": False,
            "sha": None,
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    sha = result.stdout.strip() if result.returncode == 0 else None
    resolved = bool(sha and len(sha) == 40 and all(character in "0123456789abcdef" for character in sha))
    return {
        "resolved": resolved,
        "sha": sha if resolved else None,
        "returncode": result.returncode,
        "error": None if resolved else (result.stderr.strip() or result.stdout.strip() or "HEAD did not resolve"),
    }


def _git_file_at_head(repo: Path, head_sha: str | None, relative_path: str) -> dict[str, Any]:
    """Hash the exact tracked blob addressed by ``HEAD:path`` without checkout mutation."""
    if head_sha is None:
        return {
            "resolved": False,
            "sha256": None,
            "returncode": None,
            "error": "HEAD is unresolved",
        }
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{head_sha}:{relative_path}"],
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return {
            "resolved": False,
            "sha256": None,
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    resolved = result.returncode == 0
    return {
        "resolved": resolved,
        "sha256": _sha256_bytes(result.stdout) if resolved else None,
        "returncode": result.returncode,
        "error": None
        if resolved
        else result.stderr.decode("utf-8", errors="replace").strip() or "tracked file did not resolve at HEAD",
    }


def _main_source_custody(
    source_root: Path,
    main_source_repo: Path,
    created_utc: str | None,
) -> dict[str, Any]:
    """Compare every authority input against a descendant MAIN checkout."""
    source_head = _git_head(source_root)
    main_head = _git_head(main_source_repo)
    mismatches: list[dict[str, Any]] = []
    if not source_head["resolved"]:
        mismatches.append(
            {
                "code": "SOURCE_HEAD_UNRESOLVED",
                "path": str(source_root.resolve()),
                "detail": source_head["error"],
            }
        )
    if not main_head["resolved"]:
        mismatches.append(
            {
                "code": "MAIN_HEAD_UNRESOLVED",
                "path": str(main_source_repo.resolve()),
                "detail": main_head["error"],
            }
        )

    ancestry: dict[str, Any]
    if source_head["resolved"] and main_head["resolved"]:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(main_source_repo),
                    "merge-base",
                    "--is-ancestor",
                    str(source_head["sha"]),
                    str(main_head["sha"]),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                ancestry_result = "SOURCE_IS_ANCESTOR_OF_MAIN"
            elif result.returncode == 1:
                ancestry_result = "SOURCE_IS_NOT_ANCESTOR_OF_MAIN"
                mismatches.append(
                    {
                        "code": "MAIN_NOT_DESCENDANT_OF_SOURCE_HEAD",
                        "source_head": source_head["sha"],
                        "main_head": main_head["sha"],
                    }
                )
            else:
                ancestry_result = "ANCESTRY_CHECK_ERROR"
                mismatches.append(
                    {
                        "code": "ANCESTRY_CHECK_ERROR",
                        "returncode": result.returncode,
                        "detail": result.stderr.strip() or result.stdout.strip(),
                    }
                )
            ancestry = {
                "result": ancestry_result,
                "returncode": result.returncode,
                "source_head": source_head["sha"],
                "main_head": main_head["sha"],
            }
        except OSError as exc:
            ancestry = {
                "result": "ANCESTRY_CHECK_ERROR",
                "returncode": None,
                "source_head": source_head["sha"],
                "main_head": main_head["sha"],
            }
            mismatches.append(
                {
                    "code": "ANCESTRY_CHECK_ERROR",
                    "returncode": None,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    else:
        ancestry = {
            "result": "BLOCKED_UNRESOLVED_HEAD",
            "returncode": None,
            "source_head": source_head["sha"],
            "main_head": main_head["sha"],
        }

    comparisons: list[dict[str, Any]] = []
    for authority, relative_path in _MAIN_SOURCE_AUTHORITY_INPUTS:
        source_path = source_root / relative_path
        main_path = main_source_repo / relative_path
        source_sha256 = _sha256_file(source_path)
        main_sha256 = _sha256_file(main_path)
        source_head_file = _git_file_at_head(source_root, source_head["sha"], relative_path)
        main_head_file = _git_file_at_head(main_source_repo, main_head["sha"], relative_path)
        compared_hashes = (
            source_sha256,
            source_head_file["sha256"],
            main_sha256,
            main_head_file["sha256"],
        )
        match = all(value is not None for value in compared_hashes) and len(set(compared_hashes)) == 1
        comparison = {
            "authority": authority,
            "path": relative_path,
            "source_exists": source_path.is_file(),
            "main_exists": main_path.is_file(),
            "source_sha256": source_sha256,
            "source_head_blob": source_head_file,
            "main_sha256": main_sha256,
            "main_head_blob": main_head_file,
            "match": match,
        }
        comparisons.append(comparison)
        if source_sha256 is None:
            mismatches.append({"code": "SOURCE_AUTHORITY_INPUT_MISSING", "path": relative_path})
        if main_sha256 is None:
            mismatches.append({"code": "MAIN_AUTHORITY_INPUT_MISSING", "path": relative_path})
        if source_head["resolved"] and not source_head_file["resolved"]:
            mismatches.append(
                {
                    "code": "SOURCE_HEAD_AUTHORITY_INPUT_UNRESOLVED",
                    "path": relative_path,
                    "detail": source_head_file["error"],
                }
            )
        if main_head["resolved"] and not main_head_file["resolved"]:
            mismatches.append(
                {
                    "code": "MAIN_HEAD_AUTHORITY_INPUT_UNRESOLVED",
                    "path": relative_path,
                    "detail": main_head_file["error"],
                }
            )
        if all(value is not None for value in compared_hashes) and len(set(compared_hashes)) != 1:
            mismatches.append(
                {
                    "code": "AUTHORITY_INPUT_SHA256_MISMATCH",
                    "path": relative_path,
                    "source_sha256": source_sha256,
                    "source_head_blob_sha256": source_head_file["sha256"],
                    "main_sha256": main_sha256,
                    "main_head_blob_sha256": main_head_file["sha256"],
                }
            )

    verified = (
        source_head["resolved"]
        and main_head["resolved"]
        and ancestry["result"] == "SOURCE_IS_ANCESTOR_OF_MAIN"
        and all(comparison["match"] for comparison in comparisons)
        and not mismatches
    )
    return {
        "schema": "main_source_custody.v1",
        "status": "VERIFIED_SNAPSHOT" if verified else "BLOCKED",
        "created_utc_snapshot_label": created_utc,
        "source_worktree": str(source_root.resolve()),
        "main_source_repo": str(main_source_repo.resolve()),
        "source_worktree_head": source_head,
        "main_head": main_head,
        "main_descends_from_source_head": True
        if ancestry["result"] == "SOURCE_IS_ANCESTOR_OF_MAIN"
        else False
        if ancestry["result"] == "SOURCE_IS_NOT_ANCESTOR_OF_MAIN"
        else None,
        "merge_base_is_ancestor": ancestry,
        "authority_input_sha256_comparisons": comparisons,
        "mismatches": mismatches,
    }


def _source_hits(path: Path, needles: Iterable[str], repo_root: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    hits: list[dict[str, Any]] = []
    for needle in sorted(set(needles)):
        for number, line in enumerate(lines, start=1):
            if needle in line:
                hits.append({"source": _repo_relative(path, repo_root), "line": number, "needle": needle})
                break
    return hits


def _checkpoint_custody(main_repo: Path) -> dict[str, Any]:
    """Find a supplied BEST/EMA ep725 file without touching its run directory."""
    candidates = (
        main_repo / "experiments/results/banks/v9c2_defensive_bank_20260718/levelset_witness_ema_BEST.npz",
        main_repo / "experiments/results/levelset_n600_witness_20260717T113932Z/levelset_witness_ema_BEST.npz",
    )
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        return {
            "requested_identity": "v9c2 BEST EMA epoch 725",
            "status": "OPEN",
            "path": None,
            "bytes": None,
            "sha256": None,
            "blocker": "No supplied ep725 BEST/EMA checkpoint was found under main_repo; no run bytes were modified.",
        }
    digest = _sha256_file(checkpoint)
    size = checkpoint.stat().st_size
    best_json = checkpoint.parent / "levelset_best.json"
    sidecars = {name: _sha256_file(checkpoint.parent / name) for name in ("levelset_best.json", "PROVENANCE.md")}
    try:
        best_metadata = json.loads(best_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        best_metadata = None
    metadata_exact = bool(
        isinstance(best_metadata, dict)
        and int(best_metadata.get("epoch", -1)) == EXPECTED_BEST_EPOCH
        and float(best_metadata.get("d_seg", float("nan"))) == EXPECTED_BEST_DSEG
        and best_metadata.get("path") == checkpoint.name
        and sidecars["levelset_best.json"] == EXPECTED_BEST_JSON_SHA256
    )
    exact = size == EXPECTED_CHECKPOINT_BYTES and digest == EXPECTED_CHECKPOINT_SHA256 and metadata_exact
    return {
        "requested_identity": "v9c2 BEST EMA epoch 725",
        "status": "VERIFIED_EXPECTED" if exact else "BLOCKED_MISMATCH",
        "path": str(checkpoint),
        "bytes": size,
        "sha256": digest,
        "expected_epoch": EXPECTED_BEST_EPOCH,
        "expected_d_seg": EXPECTED_BEST_DSEG,
        "expected_bytes": EXPECTED_CHECKPOINT_BYTES,
        "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        "best_metadata": best_metadata,
        "best_metadata_exact": metadata_exact,
        "sidecar_hashes": sidecars,
        "blocker": "Path/hash custody does not prove the required BEST boundary, optimizer/RNG state, or paired-arm comparability."
        if exact
        else "Selected exact candidate does not match the required ep725 byte/hash custody.",
    }


def _common_readiness(repo_root: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    safe_run = repo_root / "tools" / "safe_run.py"
    requirements = {
        "dedicated_ep725_v9c3_typed_composer": "BLOCKED: no dedicated ep725 v9c3 typed composer is selected by this materializer.",
        "margin_step_cap_measured_cap": "BLOCKED: #518 MarginStepCap must be derived from measured provenance, never configured as prose or a float.",
        "best_boundary_and_rng_state": "OPEN: BEST boundary plus exact optimizer/RNG state is not established by checkpoint bytes alone.",
        "powered_slope_threshold_joins": "BLOCKED: beta2 relaxation, response time, predicted term share, and measured NCDE/costate noise are not joined.",
        "identical_checkpoint_cadence_rng_data_order": "OPEN: the paired OFF twin has not yet proven identical checkpoint/cadence/RNG/data-order custody.",
        "ep_loss_gt_zero": "OPEN: no live telemetry proves ep_loss > 0.",
        "spike_guard": "OPEN: no armed-run receipt proves the spike guard is active.",
        "lever_engage_and_term_share": "OPEN: no live lever_engage and term-share telemetry exists.",
        "live_vs_ema_verdict": "OPEN: no verdict receipt establishes live-vs-EMA source and EMA-lag handling.",
        "resume_event_reanchor": "OPEN: no resume-event reanchor receipt exists.",
        "n600_only": "OPEN: static declaration cannot prove a fresh n600 ordering/projection.",
        "powered_ncde_costate_window": "BLOCKED: powered NCDE/costate measurement window is unjoined.",
        "positive_control_sentinel": "OPEN: positive-control sentinel has not been emitted by an arm.",
        "safe_run_containment": "VERIFIED_STATIC" if safe_run.is_file() else "BLOCKED: tools/safe_run.py is absent.",
    }
    return {
        "checkpoint_custody": checkpoint,
        "requirements": requirements,
        "full_launch_readiness": "BLOCKED",
        "blockers": [value for value in requirements.values() if value.startswith(("OPEN", "BLOCKED"))],
        "geometry_518": {
            "beta2_rewarm": "DERIVED_PROVISIONAL: 27 epochs / 2025 optimizer steps; 8-vs-27 A/B owed",
            "fork_head_solve": "WIRED_DEFAULT_OFF",
            "margin_step_cap": "WIRED_DEFAULT_OFF_MEASURED_CAP_ABSENT",
            "v0_positioning": "OPEN",
            "resume_event_boundary_persistence": "PARTIAL",
            "sources": _apparatus_hashes(repo_root),
        },
        "memory": {"base_n600_gib": 71.54, "safe_ceiling_gib": 89.6, "per_arm_projection": "ABSENT_BLOCKED"},
        "config_freshness": {
            "status": "RC6_REQUIRED_BLOCKED",
            "exact_cadence_obligations": "identical eval/observer/checkpoint cadence required; no paired receipt",
        },
        "verdict_thresholds": {
            "FIRED-PAYS": "UNSEALED/BLOCKED",
            "FIRED-NEUTRAL": "UNSEALED/BLOCKED",
            "FIRED-HURTS": "UNSEALED/BLOCKED",
            "missing_join": "beta2 + lever response time + predicted share + paired NCDE/costate noise",
            "non_authorizing_context": {
                "costate_slope_s_per_ep": -0.0001280414014,
                "half_width_s_per_ep": 2.5222969e-5,
                "half_width_25ep": 0.0006305742,
                "ncde_r2": 0.06020,
            },
        },
    }


def _apparatus_hashes(repo_root: Path) -> dict[str, dict[str, str | None]]:
    files = {
        "launcher": "tools/launch_witness_run.py",
        "governor_daemon": "tools/spawn_durable_daemon.py",
        "operator_authorize": "tools/operator_authorize.py",
        "safe_run": "tools/safe_run.py",
        "readiness_gate": "tools/witness_launch_readiness_gate.py",
    }
    return {key: {"path": value, "sha256": _sha256_file(repo_root / value)} for key, value in files.items()}


def _scientific_lever_declaration(factory_name: str) -> tuple[Any, dict[str, Any]]:
    """Rebuild the lever declaration used by the canonical V9 contrast."""
    imported = importlib.import_module("tac.witness_dsl.curriculum_dsl")
    arguments: dict[str, dict[str, Any]] = {
        "DsegAwareTaper": {
            "strength": 1.0,
            "scale": 0.0,
            "floor": 0.05,
            "scientific_declaration": True,
        },
        "HorizonWeightedMargin": {
            "weight": 0.15,
            "target": 0.5,
            "margin_lo": 0.3,
            "margin_hi": 0.5,
            "row_lo": 96,
            "row_hi": 288,
            "start_epoch": 726,
            "window": 0,
            "stage_share_derived_live": True,
            "scientific_declaration": True,
        },
        "StepNativeActivation": {
            "beta_start": 1.0,
            "beta_end": 8.0,
            "anneal": "linear",
            "window": 0,
            "basis": "annealed_hosc",
            "omega": 1.0,
            "finer_bias_init": False,
            "scientific_declaration": True,
        },
    }
    selected = arguments[factory_name]
    return getattr(imported, factory_name)(**selected), selected


@lru_cache(maxsize=3)
def _canonical_iso_contract(factory_name: str) -> dict[str, Any]:
    """Re-derive the signed control-to-treatment contrast from the canonical compiler."""
    spec = importlib.import_module("tac.witness_dsl.spec_v9_cgauge")
    wrapper_name = _ISO_CONTRASTS[factory_name]["wrapper"]
    wrapped = getattr(spec, wrapper_name)(out_dir=f"__static_duty_ticket_{factory_name}")
    contract = _json_safe(wrapped.dsl_program_manifest.get("iso_contract"))
    if not isinstance(contract, dict) or not contract.get("one_lever_delta"):
        raise ValueError(f"{factory_name}: canonical ISO compiler did not emit a one-Lever contract")
    return contract


def _typed_delta(factory_name: str) -> dict[str, Any]:
    lever, factory_arguments = _scientific_lever_declaration(factory_name)
    contrast = _ISO_CONTRASTS[factory_name]
    contract = _canonical_iso_contract(factory_name)
    lever_declaration = {
        "kind": "WitnessProgram.Lever",
        "factory": factory_name,
        "factory_arguments": factory_arguments,
        "lever_name": lever.name,
        "overrides": dict(sorted(lever.overrides.items())),
        "epochs_delta": lever.epochs_delta,
        "notes": lever.notes,
        "lawrefs": _json_safe(getattr(lever, "lawrefs", {})),
        "constant_manifest": _json_safe(getattr(lever, "constant_manifest", {})),
        "runtime_receipt_schemas": _json_safe(getattr(lever, "runtime_receipt_schemas", {})),
    }
    return {
        "schema": "typed_treatment_delta.v2",
        "kind": "WitnessProgram.SignedLeverContrast",
        "factory": factory_name,
        "declaration_source": f"src/tac/witness_dsl/spec_v9_cgauge.py::{contrast['wrapper']}",
        "volatile_resolution_metadata_omitted": ["resolved_at"],
        "operation": contrast["operation"],
        "declared_lever_role": contrast["declared_lever_role"],
        "control_config": contract["control_config"],
        "treatment_config": contract["config_id"],
        "duty_to_measure_percent": contract["duty_to_measure_percent"],
        "one_lever_delta": contract["one_lever_delta"],
        "signed_argv_diff": contract["argv_diff"],
        "lever_declaration": lever_declaration,
    }


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if str(key) != "resolved_at"}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _treatment_audit(factory_name: str, repo_root: Path) -> dict[str, Any]:
    delta = _typed_delta(factory_name)
    registry_lever = resolve_composable_lever(factory_name)
    trainer_flags = real_trainer_flags(TRAINER_PATH)
    override_flags = sorted(delta["signed_argv_diff"])
    accepted = {flag: flag in trainer_flags for flag in override_flags}
    normalized = [flag[2:].replace("-", "_") for flag in override_flags]
    source_hits = _source_hits(TRAINER_PATH, normalized, repo_root)
    imported = importlib.import_module("tac.witness_dsl.curriculum_dsl")
    return {
        "treatment_delta": delta,
        "canonical_treatment_delta_hash": _sha256_bytes(_canonical_bytes(delta)),
        "factory_resolution": {
            "module": "tac.witness_dsl.curriculum_dsl",
            "imported": imported is not None,
            "resolved_through": "tac.witness_dsl.lever_registry.resolve_composable_lever",
            "bare_registry_lever_name": registry_lever.name,
            "scientific_delta_uses_explicit_existing_iso_arguments": True,
            "signed_operation": delta["operation"],
            "declared_lever_role": delta["declared_lever_role"],
        },
        "factory_source": {
            "path": "src/tac/witness_dsl/curriculum_dsl.py",
            "line": _source_hits(Path(imported.__file__ or ""), [f"def {factory_name}("], repo_root)[0]["line"],
            "signature": str(__import__("inspect").signature(getattr(imported, factory_name))),
        },
        "treatment_delta_count": 1,
        "treatment_activity": (
            "DECLARED_NONZERO_DERIVED_LIVE_REQUIRES_BOUNDARY_RECEIPT"
            if factory_name == "HorizonWeightedMargin"
            else "DECLARED_COMPLETE_CONTROL_LEVER_REMOVAL"
            if factory_name == "DsegAwareTaper"
            else "DECLARED_TYPED_ENDPOINT_REPLACEMENT"
        ),
        "trainer_surface": {
            "trainer": _repo_relative(TRAINER_PATH, repo_root),
            "trainer_sha256": _sha256_file(TRAINER_PATH),
            "override_flags_accepted": accepted,
            "consumer_source_hits": source_hits,
            "status": "VERIFIED_STATIC"
            if all(accepted.values()) and len(source_hits) == len(override_flags)
            else "BLOCKED",
        },
    }


def _launch_script(ticket_id: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' 'REFUSE: {ticket_id} is a non-authorizing duty-ticket package.'
printf '%s\\n' 'No trainer command is present; satisfy compiled-config and custody blockers first.'
exit 6
"""


def _dry_receipt(ticket_id: str, static_checks: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "duty_ticket_dry_start_receipt.v1",
        "ticket_id": ticket_id,
        "verdict": "GREEN",
        "scope": "materializer_integrity",
        "checks": static_checks,
        "launch_ready": False,
        "refusal_wrapper_executed": True,
        "trainer_or_governed_launch_executed": False,
        "expected_refusal_exit_code": 6,
    }


def _confound_audit(readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "duty_ticket_confound_self_audit.v1",
        "verdict": "BLOCKED",
        "paired_off_twin_required": True,
        "banked_control": "NONCOMPARABLE_UNLESS_BYTE_LEVEL_PROOF_OF_IDENTICAL_518_RECIPE",
        "checks": readiness["requirements"],
        "conclusion": "No full A/B launch is authorized by static package generation.",
    }


def _markdown_verdict(ticket_id: str, factory: str | None, blockers: list[str]) -> str:
    if factory == "DsegAwareTaper":
        treatment = "remove the complete `DsegAwareTaper` Lever from the ON control"
    else:
        treatment = f"`{factory}`" if factory else "existing #497 source audit"
    lines = [
        f"# {ticket_id}",
        "",
        "Status: `BLOCKED` (non-authorizing; no trainer was invoked).",
        "",
        f"Treatment: {treatment}.",
        "",
        "## Blocking prerequisites",
        "",
    ]
    lines.extend(f"- {blocker}" for blocker in blockers)
    lines.extend(
        ("", "The dry-start receipt is GREEN only for bounded static checks and does not imply launch readiness.", "")
    )
    return "\n".join(lines)


def _write_common_files(
    ticket_dir: Path,
    config: dict[str, Any],
    provenance: dict[str, Any],
    readiness: dict[str, Any],
    *,
    factory: str | None,
) -> None:
    _atomic_write(ticket_dir / "launch.sh", _launch_script(ticket_dir.name))
    os.chmod(ticket_dir / "launch.sh", 0o755)
    _write_json(ticket_dir / "compiled_config.json", config)
    _write_json(ticket_dir / "provenance.json", provenance)
    launch = ticket_dir / "launch.sh"
    launch_text = launch.read_text(encoding="utf-8")
    static_ok = "train_levelset_witness" not in launch_text and "subprocess" not in launch_text
    run = subprocess.run([str(launch)], capture_output=True, text=True, check=False)
    receipt = _dry_receipt(
        ticket_dir.name,
        {"launch_script_has_no_trainer_or_subprocess": static_ok, "compile_state": config["compile_state"]},
    )
    receipt.update(
        {
            "verdict": "GREEN_STATIC_REFUSAL" if static_ok and run.returncode == 6 else "BLOCKED",
            "returncode": run.returncode,
            "stdout": run.stdout,
            "stderr": run.stderr,
        }
    )
    _write_json(ticket_dir / "dry_start_receipt.json", receipt)
    confounds = _confound_audit(readiness)
    _write_json(ticket_dir / "confound_self_audit.json", confounds)
    _atomic_write(
        ticket_dir / "confound_self_audit.md",
        "# Confound self-audit\n\n```json\n" + json.dumps(confounds, indent=2, sort_keys=True) + "\n```\n",
    )
    _atomic_write(ticket_dir / "verdict_card.md", _markdown_verdict(ticket_dir.name, factory, readiness["blockers"]))


def _materialize_treatment(
    ticket_id: str,
    factory: str,
    share: str,
    out_dir: Path,
    repo_root: Path,
    checkpoint: dict[str, Any],
    main_source_custody: dict[str, Any],
    created_utc: str | None,
) -> TicketSummary:
    ticket_dir = out_dir / ticket_id
    audit = _treatment_audit(factory, repo_root)
    readiness = _common_readiness(repo_root, checkpoint)
    ticket_specific = {
        "DsegAwareTaper": (
            "BLOCKED: the canonical 78.9% treatment removes the complete DsegAwareTaper Lever from "
            "the ON control; this structural epoch-0 contrast cannot be created by removing the "
            "Lever only after resuming the ep725 c2 checkpoint."
        ),
        "HorizonWeightedMargin": (
            "BLOCKED: the 0.15 share is a DERIVED-LIVE request, not a realized weight; no ep725 fork "
            "boundary receipt resolves the nonzero weight or measured MarginStepCap."
        ),
        "StepNativeActivation": (
            "BLOCKED: the treatment changes the live c2 HOSC endpoint/schedule at ep725; no matching "
            "resume receipt repositions v0 and activation geometry or proves a response window."
        ),
    }[factory]
    readiness["requirements"]["ticket_specific_warm_start_compatibility"] = ticket_specific
    readiness["blockers"] = [ticket_specific, *readiness["blockers"]]
    config = {
        "schema": SCHEMA,
        "ticket_id": ticket_id,
        "created_utc": created_utc,
        "status": "BLOCKED",
        "compile_state": "BLOCKED_BEFORE_FULL_COMPILE",
        "full_dsl_compile_hash": None,
        "full_dsl_compile_hash_blocker": "No dedicated ep725 v9c3 typed composer and #518/power joins are available; argv has no authority.",
        "argv_authority": "NONE",
        "duty_share_requested": share,
        "historical_iso_status": (
            "NOT_SELECTED_FOR_LAUNCH: the existing fresh mod19/3000-epoch ISO is not an ep725 c2/#518 weights-only fork"
        ),
        "paired_off_twin": {
            "required": True,
            "selection": "FRESH_PAIRED_OFF_TWIN",
            "banked_control": "NONCOMPARABLE_WITHOUT_IDENTICAL_518_BYTE_LEVEL_PROOF",
        },
        "treatment": audit,
        "common_readiness": readiness,
    }
    provenance = {
        "schema": "duty_ticket_provenance.v1",
        "ticket_id": ticket_id,
        "created_utc": created_utc,
        "repository": {"root": str(repo_root), "trainer_sha256": _sha256_file(TRAINER_PATH)},
        "main_repo_checkpoint_custody": checkpoint,
        "main_source_custody": main_source_custody,
        "treatment_delta_hash": audit["canonical_treatment_delta_hash"],
    }
    _write_common_files(ticket_dir, config, provenance, readiness, factory=factory)
    _write_json(ticket_dir / "typed_treatment_delta.json", audit["treatment_delta"])
    _write_json(
        ticket_dir / "witness_program.json",
        {
            "status": "UNMATERIALIZED/BLOCKED",
            "full_dsl_compile_hash": None,
            "argv": None,
            "missing_joins": ["dedicated_ep725_v9c3_typed_composer", "#518 geometry", "powered slope thresholds"],
        },
    )
    return TicketSummary(ticket_id, "BLOCKED", str(ticket_dir), audit["canonical_treatment_delta_hash"])


def _curvelet_source_audit(repo_root: Path) -> dict[str, Any]:
    source = repo_root / "tools" / "fire_curvelet_matched_bytes_ab_p0_497.py"
    text = source.read_text(encoding="utf-8") if source.is_file() else ""
    imports: list[str] = []
    if text:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
    import_closure: dict[str, bool] = {}
    for module_name in sorted(set(imports)):
        try:
            importlib.import_module(module_name)
            import_closure[module_name] = True
        except Exception:  # an unavailable optional dependency is itself an audit fact
            import_closure[module_name] = False
    source_sha256 = _sha256_file(source)
    pure_compile = _curvelet_pure_compile_evidence(repo_root)
    pure_schedule_clean = bool(
        pure_compile.get("classification") == "MEASURED_AT_CURRENT_HEAD_PURE_PRODUCTION_EQUIVALENT"
        and pure_compile.get("arms")
        and all(
            row.get("schedule_rc6") == 0 and row.get("schedule_violation_count") == 0
            for row in pure_compile["arms"].values()
        )
    )
    return {
        "source": _repo_relative(source, repo_root),
        "source_sha256": source_sha256,
        "expected_source_sha256": EXPECTED_CURVELET_FIRE_SHA256,
        "source_sha256_matches_expected": source_sha256 == EXPECTED_CURVELET_FIRE_SHA256,
        "source_present": source.is_file(),
        "imports": sorted(set(imports)),
        "import_closure": import_closure,
        "arm_cli_detected": all(token in text for token in ("--arm", "control", "treatment")),
        "operator_go_gate_detected": "--operator-go" in text,
        "skip_c2_gate_present": "--skip-c2-gate" in text,
        "c2_gate_exception_scope": "ImportError_only" if "except ImportError" in text else "UNKNOWN",
        "resume_from_flag_detected": "--resume-from" in text or "--resume-model-from" in text,
        "governed_launcher_dry_run_detected": "--dry-run" in text,
        "geometry_518_binding_detected": "#518" in text,
        "epoch_semantics": "PURE_COMPILE_RECORDS_FRESH_3000_EPOCH_ARMS",
        "one_factor_status": "BLOCKED: treatment changes basis plus AA/native-render behavior",
        "sequential_equal_byte_finalizer": "DOCUMENTED_BUT_NOT_ENFORCED_BY_WRAPPER",
        "lever_telemetry_scope": "front_end_only",
        "pure_compile_evidence": pure_compile,
        "c2_run_dir_gate_declared": "C2_RUN_DIR_NAME" in text,
        "warm_start_checkpoint_input": "ABSENT",
        "config_declared": "v9_cgauge_ideal_mod32" in text,
        "pure_schedule_rc6_clean": pure_schedule_clean,
        "governed_config_freshness_receipt": "ABSENT",
        "status": "BLOCKED",
        "blockers": [
            *(
                []
                if source_sha256 == EXPECTED_CURVELET_FIRE_SHA256
                else ["The #497 source hash differs from the delegated expected source hash."]
            ),
            "Static source audit cannot prove both arm inputs/checkpoints are current and comparable.",
            "No current compiled-config freshness receipt is available to this materializer.",
            "This delegated lane may not invoke either real arm or its governed dry-run.",
        ],
    }


def _curvelet_pure_compile_evidence(repo_root: Path) -> dict[str, Any]:
    """Pure config inspection only; never imports or executes the #497 fire script."""
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(repo_root / "tools"))
        import launch_witness_run as lwr
        import schedule_provenance_gate as spg

        out_dirs = {
            "control": "/Volumes/VertigoDataTier/pact/experiments/results/curvelet_matched_bytes_ab_control_n600_seed0",
            "treatment": "/Volumes/VertigoDataTier/pact/experiments/results/curvelet_matched_bytes_ab_literal_n600_seed0",
        }
        purposes = {
            "control": "p0_497 matched-bytes A/B: equal-config legacy-Fourier control; operator-GO only",
            "treatment": "p0_497 matched-bytes A/B: literal polar-curvelet + native-orient treatment; operator-GO only",
        }
        rows: dict[str, Any] = {}
        trainer_text = lwr.TRAINER_PATH.read_text()
        for arm in ("control", "treatment"):
            cfg = lwr.derive_named_config(
                "v9_cgauge_ideal_mod32",
                "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
                num_pairs=600,
                epochs=None,
                overfit=True,
            )
            if arm == "treatment":
                cfg = cfg.with_dsl_lever_factories("LiteralPolarCurveletBasis")
            cfg = cfg.with_purpose(purposes[arm])
            document, _, _ = lwr.compile_dsl_document_for_config(
                cfg,
                Path(out_dirs[arm]),
                program_name="v9_cgauge_ideal_mod32",
            )
            pairs = list(cfg.to_trainer_flags(out_dirs[arm]))
            pair_values = dict(pairs)
            verdicts = spg.classify_launch(
                pairs,
                registry=spg.schedule_when_flags(trainer_text),
                manifest_keys=set(getattr(cfg, "constants_manifest", {}) or {}),
                governance=getattr(cfg, "schedule_governance", {}) or {},
                event_registry=spg.event_start_flags(trainer_text),
            )
            ok, violations, _ = spg.gate_report(verdicts)
            typed = cfg.typed if getattr(cfg, "typed", None) is not None else cfg
            rows[arm] = {
                "dsl_compile_hash": document.get("dsl_compile_hash"),
                "typed_config_hash": typed.typed_config_hash(),
                "resolved_argv_hash": _sha256_bytes(_canonical_bytes(document["resolved_argv"])),
                "resolved_argv_hash_semantic": "sha256(canonical compact JSON of dsl_document.resolved_argv)",
                "epochs": int(pair_values["--epochs"]),
                "num_pairs": int(pair_values["--num-pairs"]),
                "schedule_rc6": 0 if ok else 6,
                "schedule_verdict_count": len(verdicts),
                "schedule_violation_count": len(violations),
            }
        return {
            "classification": "MEASURED_AT_CURRENT_HEAD_PURE_PRODUCTION_EQUIVALENT",
            "method": "derive_named_config + with_dsl_lever_factories + with_purpose + compile_dsl_document_for_config + schedule gate",
            "arms": rows,
        }
    except Exception as exc:
        return {
            "classification": "BLOCKED_PURE_COMPILE_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "schedule_freshness": "RC6_UNPROVEN",
        }
    finally:
        sys.path[:] = original_path


def _materialize_curvelet(
    out_dir: Path,
    repo_root: Path,
    checkpoint: dict[str, Any],
    main_source_custody: dict[str, Any],
    created_utc: str | None,
) -> TicketSummary:
    ticket_id = "04_curvelet_matched_bytes_p0_497"
    ticket_dir = out_dir / ticket_id
    audit = _curvelet_source_audit(repo_root)
    readiness = _common_readiness(repo_root, checkpoint)
    readiness = dict(readiness)
    readiness["blockers"] = list(readiness["blockers"]) + audit["blockers"]
    config = {
        "schema": SCHEMA,
        "ticket_id": ticket_id,
        "created_utc": created_utc,
        "status": "BLOCKED",
        "compile_state": "BLOCKED_BEFORE_FULL_COMPILE",
        "full_dsl_compile_hash": None,
        "full_dsl_compile_hash_blocker": "Ticket 4 audits an existing fire script and does not recompose it; no argv authority is granted.",
        "argv_authority": "NONE",
        "audit": audit,
        "common_readiness": readiness,
    }
    provenance = {
        "schema": "duty_ticket_provenance.v1",
        "ticket_id": ticket_id,
        "created_utc": created_utc,
        "current_source_sha256": audit["source_sha256"],
        "main_repo_checkpoint_custody": checkpoint,
        "main_source_custody": main_source_custody,
    }
    _write_common_files(ticket_dir, config, provenance, readiness, factory=None)
    _write_json(ticket_dir / "audited_config.json", audit)
    return TicketSummary(ticket_id, "BLOCKED", str(ticket_dir), None)


def materialize_duty_queue_fire_tickets(
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    main_repo: str | Path = DEFAULT_MAIN_REPO,
    main_source_repo: str | Path = DEFAULT_MAIN_SOURCE_REPO,
    repo_root: str | Path | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Write the four ordered packages, using only static inspection and pure DSL resolution."""
    output = Path(out_dir)
    source_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    checkpoint = _checkpoint_custody(Path(main_repo))
    source_custody = _main_source_custody(source_root, Path(main_source_repo), created_utc)
    source_custody_path = output / "main_source_custody.json"
    _write_json(source_custody_path, source_custody)
    source_custody_reference = {
        "schema": source_custody["schema"],
        "status": source_custody["status"],
        "path": "../main_source_custody.json",
        "sha256": _sha256_file(source_custody_path),
        "created_utc_snapshot_label": source_custody["created_utc_snapshot_label"],
        "source_worktree_head": source_custody["source_worktree_head"]["sha"],
        "main_head": source_custody["main_head"]["sha"],
    }
    summaries = [
        _materialize_treatment(
            ticket_id,
            factory,
            share,
            output,
            source_root,
            checkpoint,
            source_custody_reference,
            created_utc,
        )
        for ticket_id, factory, share in _TICKETS
    ]
    summaries.append(_materialize_curvelet(output, source_root, checkpoint, source_custody_reference, created_utc))
    for item in summaries:
        _write_json(Path(item.path) / "artifact_manifest.json", _artifact_manifest(Path(item.path)))
    summary = {
        "schema": "duty_queue_fire_ticket_summary.v1",
        "created_utc": created_utc,
        "ticket_order": [item.ticket_id for item in summaries],
        "tickets": [
            {"ticket_id": item.ticket_id, "status": item.status, "treatment_delta_hash": item.treatment_delta_hash}
            for item in summaries
        ],
        "all_status": "BLOCKED",
        "main_source_custody": {
            "status": source_custody["status"],
            "path": "main_source_custody.json",
            "sha256": source_custody_reference["sha256"],
        },
        "wrappers_executed": 4,
        "trainer_governed_launches_executed": 0,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "summary_manifest.json", _artifact_manifest(output))
    return summary


__all__ = [
    "DEFAULT_MAIN_REPO",
    "DEFAULT_MAIN_SOURCE_REPO",
    "DEFAULT_OUT_DIR",
    "materialize_duty_queue_fire_tickets",
]
