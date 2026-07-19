#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #537 real-data crash/resume proof on the live MLX trainer.

This is an evidence probe, not a score run.  It uses a real n>=24 ground-truth
cache, runs two uninterrupted controls to measure the host replay floor, kills
an entire trainer process group after an atomic intra-stage checkpoint, resumes
from that immutable checkpoint, and compares the final state.  CPU mode requires
bit identity.  GPU mode reports the measured two-control floor and judges the
resumed trajectory against that floor instead of pretending separate Metal
processes are bit deterministic.  On success it removes rebuildable checkpoint
bulk after first writing a machine-readable receipt; on failure it preserves it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _import_root in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
DEFAULT_GT = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n24.npz")
DEFAULT_BASE = REPO / "experiments" / "results" / "resumability_537_real_n24_smoke"
DEFAULT_RECEIPT = REPO / ".omx" / "research" / "resumability_537_n24_crash_resume_receipt_20260719.json"
_MODEL_PREFIXES = ("liveP__", "emaP__")
_OPT_PREFIXES = ("optP__",)
_STRUCTURAL_PREFIXES = (
    "__rng_", "__evt_", "__mg_", "__lbg_", "__scg_", "__tsg_", "__pag_",
)
_STRUCTURAL_EXACT = {
    "__resume_epoch", "__resume_stage", "__resume_event_ledger_json", "__resume_has_opt",
}
_RECEIPT_SCHEMA = "tac.resumability_537_real_crash_resume.v2"


@dataclass(frozen=True)
class _Task537TypedLaunch:
    """Small adapter exposing a directly authored TypedWitnessConfig to launcher helpers."""

    typed: Any
    dsl_program_manifest: dict[str, Any]
    constants_manifest: dict[str, Any] = field(default_factory=dict)
    name: str = "resumability_537_real_n24"
    purpose: str = "Task 537 local n24 durability proof; no score authority"

    def to_trainer_flags(self, out_dir: str | None = None):
        return self.typed.to_trainer_flags(out_dir)

    def to_command(self, out_dir: str | None = None, *, perf_env: bool = True):
        return self.typed.to_command(out_dir, perf_env=perf_env)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _state_hashes(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as z:
        for key in sorted(z.files):
            if key.startswith(("liveP__", "emaP__")):
                arr = np.ascontiguousarray(np.asarray(z[key]))
                out[key] = hashlib.sha256(arr.tobytes()).hexdigest()
    return out


def _trajectory_state_hashes(path: Path) -> dict[str, str]:
    """Hash optimizer/RNG/event/stage state that determines post-resume trajectory."""
    prefixes = ("optP__", "__rng_", "__evt_", "__mg_", "__lbg_", "__scg_", "__tsg_", "__pag_")
    exact = {"__resume_epoch", "__resume_stage", "__resume_event_ledger_json", "__resume_has_opt"}
    out: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as z:
        for key in sorted(z.files):
            if key in exact or key.startswith(prefixes):
                arr = np.ascontiguousarray(np.asarray(z[key]))
                out[key] = hashlib.sha256(arr.tobytes()).hexdigest()
    return out


def _selected_state_arrays(
    path: Path, *, prefixes: tuple[str, ...], exact: set[str] | None = None,
) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    exact = exact or set()
    with np.load(path, allow_pickle=False) as z:
        for key in sorted(z.files):
            if key in exact or key.startswith(prefixes):
                selected[key] = np.ascontiguousarray(np.asarray(z[key]))
    return selected


def _array_delta(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return strict byte identity plus numeric deltas for one state partition."""
    left_keys = set(left)
    right_keys = set(right)
    per_key: dict[str, dict[str, Any]] = {}
    compatible = left_keys == right_keys
    for key in sorted(left_keys & right_keys):
        a = left[key]
        b = right[key]
        same_layout = a.shape == b.shape and a.dtype == b.dtype
        byte_identical = same_layout and a.tobytes() == b.tobytes()
        numeric = bool(same_layout and np.issubdtype(a.dtype, np.number))
        max_abs: float | None = None
        rms: float | None = None
        if numeric:
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            max_abs = float(np.max(delta)) if delta.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0
        per_key[key] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "same_layout": same_layout,
            "bit_identical": byte_identical,
            "numeric": numeric,
            "max_abs": max_abs,
            "rms": rms,
        }
        compatible = compatible and same_layout
    numeric_rows = [row for row in per_key.values() if row["numeric"]]
    return {
        "compatible": compatible,
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "count": len(per_key),
        "bit_identical": compatible and bool(per_key) and all(
            row["bit_identical"] for row in per_key.values()),
        "max_abs": max((float(row["max_abs"]) for row in numeric_rows), default=0.0),
        "max_rms": max((float(row["rms"]) for row in numeric_rows), default=0.0),
        "per_key": per_key,
    }


def _state_delta(
    left_path: Path, right_path: Path, *, prefixes: tuple[str, ...],
    exact: set[str] | None = None,
) -> dict[str, Any]:
    return _array_delta(
        _selected_state_arrays(left_path, prefixes=prefixes, exact=exact),
        _selected_state_arrays(right_path, prefixes=prefixes, exact=exact),
    )


def _within_measured_floor(
    control_floor: dict[str, Any], resumed_to_a: dict[str, Any], resumed_to_b: dict[str, Any],
) -> dict[str, Any]:
    """Judge each numeric key against the measured two-control envelope.

    The tolerance is not a guessed multiplier: for each key it is exactly the
    max-absolute delta measured between control A and control B.  The resumed
    result may match either independent control.  A zero measured floor therefore
    still requires exact numeric equality for that key.
    """
    keys = set(control_floor["per_key"])
    compatible = (
        control_floor["compatible"] and resumed_to_a["compatible"]
        and resumed_to_b["compatible"]
        and keys == set(resumed_to_a["per_key"]) == set(resumed_to_b["per_key"])
    )
    per_key: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        floor_row = control_floor["per_key"][key]
        a_row = resumed_to_a["per_key"][key]
        b_row = resumed_to_b["per_key"][key]
        if not floor_row["numeric"]:
            passed = bool(a_row["bit_identical"] or b_row["bit_identical"])
            measured_floor = None
            resumed_delta = None
        else:
            measured_floor = float(floor_row["max_abs"])
            resumed_delta = min(float(a_row["max_abs"]), float(b_row["max_abs"]))
            passed = resumed_delta <= measured_floor
        per_key[key] = {
            "measured_control_floor_max_abs": measured_floor,
            "resumed_nearest_control_max_abs": resumed_delta,
            "pass": passed,
        }
    return {
        "compatible": compatible,
        "pass": compatible and bool(per_key) and all(row["pass"] for row in per_key.values()),
        "per_key": per_key,
    }


def _resume_epoch(path: Path) -> int:
    with np.load(path, allow_pickle=False) as z:
        return int(np.asarray(z["__resume_epoch"]).item())


def _final_stage_pair(out_dir: Path, epoch: int) -> tuple[list[Path], list[Path]]:
    """Return only the preserved final/stage pair, never the periodic pair."""
    ep = int(epoch)
    return (
        sorted(Path(out_dir).glob(f"levelset_ckpt_*_ep{ep}.npz")),
        sorted(Path(out_dir).glob(f"levelset_resume_*_ep{ep}.npz")),
    )


def _failure_classification(exc: BaseException, base: Path) -> dict[str, Any]:
    checkpoint_files = sorted(
        str(path.relative_to(base))
        for path in base.glob("*/**/*.npz") if path.is_file()
    ) if base.exists() else []
    crash_candidates = sorted(base.glob("crash/levelset_periodic_resume_*_ep3.npz"))
    crash_epoch = None
    if len(crash_candidates) == 1:
        try:
            crash_epoch = _resume_epoch(crash_candidates[0])
        except Exception:
            pass
    final_pair_by_arm = {
        arm: len(_final_stage_pair(base / arm, 4)[0]) == 1
        and len(_final_stage_pair(base / arm, 4)[1]) == 1
        for arm in ("control_a", "control_b", "continuous", "resumed")
        if (base / arm).is_dir()
    }
    no_metal = "No Metal device available" in str(exc) and not checkpoint_files
    return {
        "checkpoint_files": checkpoint_files,
        "crash_epoch": crash_epoch,
        "final_pair_by_arm": final_pair_by_arm,
        "final_pair_preserved": any(final_pair_by_arm.values()),
        "status": (
            "BLOCKED_ENVIRONMENT_NO_METAL_DEVICE" if no_metal
            else "EXECUTED_PROOF_ERROR" if checkpoint_files
            else "PROOF_SETUP_ERROR"
        ),
        "exact_blocker": (
            "MLX initialization reported no Metal device before any checkpoint was written"
            if no_metal else f"{type(exc).__name__}: {exc}"
        ),
        "next_action": (
            "MAIN must rerun this exact probe on the Metal-capable host"
            if no_metal else
            "Review the preserved execution artifacts and repair the exact error before rerun"
        ),
    }


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "upstream"), str(REPO / "experiments"), str(REPO)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return env


@contextmanager
def _worktree_venv(python: Path):
    """Provide the DSL-emitted ``.venv/bin/python`` in isolated worktrees, then remove our link."""
    link = REPO / ".venv"
    created = False
    if not link.exists():
        supplied = python.absolute()
        venv_root = supplied.parent.parent
        if not supplied.is_file() or supplied.parent.name != "bin" or not (venv_root / "pyvenv.cfg").is_file():
            raise FileNotFoundError(f"--python must resolve to a virtualenv bin/python: {python}")
        link.symlink_to(venv_root, target_is_directory=True)
        created = True
    try:
        yield
    finally:
        if created and link.is_symlink():
            link.unlink()


def _out_arg(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def _prepare_dsl_launch(
    out_dir: Path, gt_cache: Path, *, mlx_device: str, seed: int,
    num_pairs: int, resume_from: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Compile an exact typed-DSL bounded-smoke launch and return its bash argv/custody hashes."""
    from tac.local_acceleration.scorer_throughput_gate import derive_wall_clock_budget_days
    from tac.witness_dsl.typed_config import (
        ProvenanceClass,
        Provenanced,
        TypedAnneal,
        TypedLever,
        TypedWitnessConfig,
    )
    from tools.launch_witness_run import write_dsl_bound_launch

    def identity(value: float) -> Provenanced:
        return Provenanced(
            value=value,
            provenance=ProvenanceClass.HARDCODED_WITH_WAIVER,
            unit="dimensionless",
            source="Task #537 bounded durability-proof identity schedule",
            waiver=("Proof-only identity constant; re-derive if pair count, cache geometry, or epoch "
                    "budget changes"),
        )
    overrides: dict[str, Any] = {
        "--seed": int(seed),
        "--eval-every": 4,
        "--verdict-pairs": 2,
        "--async-verdict": False,
        "--render-h": 96,
        "--render-w": 128,
        "--render-aa": "none",
        "--hidden-dim": 32,
        "--n-hidden": 2,
        "--self-orient": False,
        "--w-pose": 0.0,
        "--eikonal-weight": 0.0,
        "--length-weight": 0.0,
        "--w-seg": 100.0,
        "--score-domain-loss": True,
        "--seg-form-unify-tau": True,
        "--ckpt-every": 1,
        "--ckpt-retain-per-stage": 1,
        "--mod-dim-dynamics": False,
        "--mod-dim-ablation": False,
        "--jacobian-basin-telemetry": False,
        "--jacobian-basin-t0": False,
        "--jacobian-basin-stratify-t": False,
        "--annulus-telemetry": False,
    }
    if resume_from is not None:
        overrides["--resume-from"] = _out_arg(resume_from)
    typed = TypedWitnessConfig(
        name=("resumability_537_real_n24_resume" if resume_from is not None
              else "resumability_537_real_n24_control"),
        out_dir=str(out_dir),
        gt_cache=str(gt_cache),
        num_pairs=int(num_pairs),
        epochs=4,
        wall_clock_budget_days=Provenanced(
            value=derive_wall_clock_budget_days(4),
            provenance=ProvenanceClass.DERIVED_AT_CONFIG,
            unit="days",
            source="scorer_throughput_gate.derive_wall_clock_budget_days(epochs)",
        ),
        mlx_device=str(mlx_device),
        seed=int(seed),
        purpose="Task 537 local n24 durability proof; no score authority",
        temp=TypedAnneal(start=identity(1.0), end=identity(1.0)),
        levers=(TypedLever(
            name=("Task537RealN24Resume" if resume_from is not None
                  else "Task537RealN24Control"),
            overrides=overrides,
            notes="Bounded real-cache crash/resume custody proof; score_claim=false",
        ),),
    )
    violations = typed.validate_program()
    if violations:
        raise RuntimeError(f"Task #537 typed proof config invalid: {violations}")
    from tac.witness_dsl.typed_config import build_launch_manifest
    cfg = _Task537TypedLaunch(
        typed=typed,
        dsl_program_manifest=build_launch_manifest(
            program_name=typed.name,
            emitted_flag_names=[flag for flag, _ in typed.to_trainer_flags(str(out_dir))],
            typed_config_hash=typed.typed_config_hash(),
        ),
    )
    launch, provenance, manifest, document = write_dsl_bound_launch(
        cfg, out_dir, program_name="resumability_537_real_n24",
    )
    custody = {
        "launch_sh": _sha256(launch),
        "dsl_provenance": _sha256(provenance),
        "launch_manifest": _sha256(manifest),
        "dsl_compile_hash": str(document["dsl_compile_hash"]),
    }
    return ["bash", str(launch)], custody


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=REPO, env=_env(), text=True, capture_output=True, timeout=timeout,
    )


def _run_and_kill(argv: list[str], timeout: float) -> tuple[int, bool, list[str]]:
    proc = subprocess.Popen(
        argv, cwd=REPO, env=_env(), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=1, start_new_session=True,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    killed = False
    deadline = time.monotonic() + timeout
    try:
        for line in proc.stdout:
            lines.append(line.rstrip())
            if '"kind": "intra_stage"' in line and '"epoch": 3' in line:
                os.killpg(proc.pid, signal.SIGKILL)
                killed = True
                break
            if time.monotonic() > deadline:
                os.killpg(proc.pid, signal.SIGKILL)
                raise TimeoutError("trainer did not reach the epoch-3 intra-stage checkpoint")
    finally:
        proc.stdout.close()
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait(timeout=30)
    return rc, killed, lines


def _json_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=REPO, capture_output=True, check=True,
    ).stdout
    return {"head": head, "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(), "dirty": bool(diff)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base.resolve()
    receipt = args.receipt.resolve()
    gt_cache = args.gt_cache.resolve()
    if not gt_cache.is_file():
        raise FileNotFoundError(f"real gt cache missing: {gt_cache}")
    if base.exists():
        raise FileExistsError(f"refusing to overwrite existing proof directory: {base}")
    if int(args.num_pairs) < 24:
        raise ValueError(f"--num-pairs must be >=24 for the real proof, got {args.num_pairs}")
    control_a = base / "control_a"
    control_b = base / "control_b"
    crash = base / "crash"
    resumed = base / "resumed"
    for path in (control_a, control_b, crash, resumed):
        path.mkdir(parents=True, exist_ok=False)

    control_a_argv, control_a_custody = _prepare_dsl_launch(
        control_a, gt_cache, mlx_device=args.mlx_device, seed=args.seed,
        num_pairs=args.num_pairs,
    )
    control_b_argv, control_b_custody = _prepare_dsl_launch(
        control_b, gt_cache, mlx_device=args.mlx_device, seed=args.seed,
        num_pairs=args.num_pairs,
    )
    crash_argv, crash_custody = _prepare_dsl_launch(
        crash, gt_cache, mlx_device=args.mlx_device, seed=args.seed,
        num_pairs=args.num_pairs,
    )
    started = time.monotonic()
    control_a_run = _run(control_a_argv, args.timeout)
    if control_a_run.returncode != 0:
        raise RuntimeError(
            f"control A failed rc={control_a_run.returncode}\n"
            f"{control_a_run.stdout[-3000:]}\n{control_a_run.stderr[-3000:]}")
    control_b_run = _run(control_b_argv, args.timeout)
    if control_b_run.returncode != 0:
        raise RuntimeError(
            f"control B failed rc={control_b_run.returncode}\n"
            f"{control_b_run.stdout[-3000:]}\n{control_b_run.stderr[-3000:]}")
    crash_rc, killed, crash_lines = _run_and_kill(crash_argv, args.timeout)
    crash_resume_candidates = sorted(crash.glob("levelset_periodic_resume_*_ep3.npz"))
    crash_ema_candidates = sorted(crash.glob("levelset_periodic_ema_*_ep3.npz"))
    crash_state = crash_resume_candidates[0] if len(crash_resume_candidates) == 1 else None
    if (
        not killed or crash_state is None or len(crash_ema_candidates) != 1
        or _resume_epoch(crash_state) != 3
    ):
        raise RuntimeError(
            f"crash arm proof failed: killed={killed} rc={crash_rc} "
            f"resume_candidates={[p.name for p in crash_resume_candidates]} "
            f"ema_candidates={[p.name for p in crash_ema_candidates]} "
            f"epoch={_resume_epoch(crash_state) if crash_state is not None else None}"
        )
    resume_argv, resume_custody = _prepare_dsl_launch(
        resumed, gt_cache, mlx_device=args.mlx_device, seed=args.seed,
        num_pairs=args.num_pairs, resume_from=crash_state,
    )
    res = _run(resume_argv, args.timeout)
    if res.returncode != 0:
        raise RuntimeError(f"resume arm failed rc={res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")

    control_a_state = control_a / "levelset_resume_state.npz"
    control_b_state = control_b / "levelset_resume_state.npz"
    resumed_state = resumed / "levelset_resume_state.npz"
    control_a_hashes = _state_hashes(control_a_state)
    control_b_hashes = _state_hashes(control_b_state)
    resumed_hashes = _state_hashes(resumed_state)
    control_a_trajectory = _trajectory_state_hashes(control_a_state)
    control_b_trajectory = _trajectory_state_hashes(control_b_state)
    resumed_trajectory = _trajectory_state_hashes(resumed_state)
    resume_rows = _json_rows(res.stdout)
    resume_row = next((row for row in resume_rows if row.get("stage") == "resume"), {})
    reanchor_rows = [row for row in resume_rows if "reanchor" in str(row.get("stage", "")).lower()]

    periodic = sorted(control_a.glob("levelset_periodic_*_ep*.npz"))
    final_pairs_by_arm = {
        "control_a": _final_stage_pair(control_a, 4),
        "control_b": _final_stage_pair(control_b, 4),
        "resumed": _final_stage_pair(resumed, 4),
    }
    final_pair_preserved = all(
        len(ema_files) == 1 and len(resume_files) == 1
        for ema_files, resume_files in final_pairs_by_arm.values()
    )
    periodic_by_kind = {
        "ema": [p.name for p in periodic if "resume" not in p.name],
        "resume": [p.name for p in periodic if "resume" in p.name],
    }
    for import_path in (REPO, REPO / "src", REPO / "upstream"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    from tools.levelset_byte_close_and_eval import _load_levelset_ckpt

    byte_close_loaded: dict[str, int] = {}
    for checkpoint in periodic:
        if "resume" in checkpoint.name:
            continue
        params, _cfg = _load_levelset_ckpt(control_a, npz_name=checkpoint.name)
        byte_close_loaded[checkpoint.name] = len(params)
    all_checkpoint_hashes = {p.name: _sha256(p) for p in sorted(control_a.glob("*.npz"))}

    control_floor_model = _state_delta(
        control_a_state, control_b_state, prefixes=_MODEL_PREFIXES)
    resumed_to_a_model = _state_delta(
        control_a_state, resumed_state, prefixes=_MODEL_PREFIXES)
    resumed_to_b_model = _state_delta(
        control_b_state, resumed_state, prefixes=_MODEL_PREFIXES)
    control_floor_opt = _state_delta(
        control_a_state, control_b_state, prefixes=_OPT_PREFIXES)
    resumed_to_a_opt = _state_delta(
        control_a_state, resumed_state, prefixes=_OPT_PREFIXES)
    resumed_to_b_opt = _state_delta(
        control_b_state, resumed_state, prefixes=_OPT_PREFIXES)
    control_floor_structural = _state_delta(
        control_a_state, control_b_state,
        prefixes=_STRUCTURAL_PREFIXES, exact=_STRUCTURAL_EXACT)
    resumed_to_a_structural = _state_delta(
        control_a_state, resumed_state,
        prefixes=_STRUCTURAL_PREFIXES, exact=_STRUCTURAL_EXACT)
    model_floor_verdict = _within_measured_floor(
        control_floor_model, resumed_to_a_model, resumed_to_b_model)
    opt_floor_verdict = _within_measured_floor(
        control_floor_opt, resumed_to_a_opt, resumed_to_b_opt)

    common_assertions = {
        "real_num_pairs_gte_24": int(args.num_pairs) >= 24,
        "sigkill_observed": killed and crash_rc < 0,
        "crash_epoch_is_3": _resume_epoch(crash_state) == 3,
        "resume_final_epoch_is_4": _resume_epoch(resumed_state) == 4,
        "restored_optimizer": bool(resume_row.get("restored_opt")),
        "periodic_pair_retention_bounded_to_1": all(len(v) == 1 for v in periodic_by_kind.values()),
        "periodic_ema_byte_close_loadable": len(byte_close_loaded) == 1,
        "final_pair_preserved": final_pair_preserved,
        "exact_continuation_not_reanchored": not reanchor_rows,
    }
    if args.mlx_device == "cpu":
        mode_assertions = {
            "control_twice_live_ema_bit_identical": control_floor_model["bit_identical"],
            "control_twice_optimizer_rng_event_stage_bit_identical": (
                control_floor_opt["bit_identical"] and control_floor_structural["bit_identical"]),
            "live_and_ema_bit_identical": resumed_to_a_model["bit_identical"],
            "optimizer_rng_event_stage_bit_identical": (
                resumed_to_a_opt["bit_identical"] and resumed_to_a_structural["bit_identical"]),
        }
    else:
        mode_assertions = {
            "control_twice_host_floor_measured": (
                control_floor_model["compatible"] and control_floor_opt["compatible"]),
            "live_and_ema_within_measured_host_floor": model_floor_verdict["pass"],
            "optimizer_within_measured_host_floor": opt_floor_verdict["pass"],
            "rng_event_stage_bit_identical": resumed_to_a_structural["bit_identical"],
        }
    assertions = {**common_assertions, **mode_assertions}
    all_pass = all(assertions.values())
    report: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "authority": "LOCAL_DURABILITY_PROOF_ONLY",
        "axis": (
            "macOS-MLX CPU-locked durability proof; not contest CPU/CUDA"
            if args.mlx_device == "cpu"
            else "macOS-MLX GPU host-floor durability probe; not contest CPU/CUDA"
        ),
        "score_claim": False,
        "frontier_pointer_mutated": False,
        "git": _git_state(),
        "mode": args.mlx_device,
        "comparison_mode": (
            "cpu_bit_identity" if args.mlx_device == "cpu"
            else "gpu_measured_host_floor"
        ),
        "crash_epoch": _resume_epoch(crash_state),
        "final_pair_preserved": final_pair_preserved,
        "input": {"gt_cache": str(gt_cache), "bytes": gt_cache.stat().st_size, "sha256": _sha256(gt_cache), "num_pairs": int(args.num_pairs), "seed": int(args.seed)},
        "commands": {"control_a": control_a_argv, "control_b": control_b_argv, "crash": crash_argv, "resume": resume_argv},
        "dsl_custody": {
            "control_a": control_a_custody,
            "control_b": control_b_custody,
            "crash": crash_custody,
            "resume": resume_custody,
        },
        "crash": {"returncode": crash_rc, "last_rows": crash_lines[-12:]},
        "resume": {"returncode": res.returncode, "resume_row": resume_row, "reanchor_rows": reanchor_rows},
        "checkpoint_files": {
            "periodic": periodic_by_kind,
            "final_by_arm": {
                arm: [path.name for paths in pair for path in paths]
                for arm, pair in final_pairs_by_arm.items()
            },
            "crash_pair": [crash_ema_candidates[0].name, crash_state.name],
            "sha256": all_checkpoint_hashes,
            "byte_close_loaded_param_counts": byte_close_loaded,
        },
        "tensor_state": {
            "count": len(control_a_hashes), "control_a": control_a_hashes,
            "control_b": control_b_hashes, "resumed": resumed_hashes,
            "trajectory_count": len(control_a_trajectory),
            "control_a_trajectory": control_a_trajectory,
            "control_b_trajectory": control_b_trajectory,
            "resumed_trajectory": resumed_trajectory,
        },
        "host_nondeterminism_floor": {
            "definition": "per-key max-absolute delta between two uninterrupted controls",
            "control_a_vs_b_live_ema": control_floor_model,
            "control_a_vs_b_optimizer": control_floor_opt,
            "control_a_vs_b_rng_event_stage": control_floor_structural,
        },
        "resume_comparison": {
            "control_a_vs_resumed_live_ema": resumed_to_a_model,
            "control_b_vs_resumed_live_ema": resumed_to_b_model,
            "control_a_vs_resumed_optimizer": resumed_to_a_opt,
            "control_b_vs_resumed_optimizer": resumed_to_b_opt,
            "control_a_vs_resumed_rng_event_stage": resumed_to_a_structural,
            "live_ema_floor_verdict": model_floor_verdict,
            "optimizer_floor_verdict": opt_floor_verdict,
        },
        "assertions": assertions,
        "all_pass": all_pass,
        "wall_seconds": round(time.monotonic() - started, 3),
        "cleanup": {"policy": "success-only rebuildable scratch", "source_gt_deleted": False, "proof_dirs_deleted": False},
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if all_pass:
        shutil.rmtree(base)
        report["cleanup"]["proof_dirs_deleted"] = True
        receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--gt-cache", type=Path, default=DEFAULT_GT)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--mlx-device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=537)
    args = ap.parse_args()
    try:
        with _worktree_venv(args.python):
            report = run(args)
    except BaseException as exc:
        base = args.base.resolve()
        artifact_hashes: dict[str, str] = {}
        if base.exists():
            for path in sorted(base.glob("*/**/*")):
                if path.is_file() and path.name in {
                    "launch.sh", "dsl_provenance.json", "launch_manifest.json",
                }:
                    artifact_hashes[str(path.relative_to(base))] = _sha256(path)
        failure = _failure_classification(exc, base)
        blocker = {
            "schema": _RECEIPT_SCHEMA,
            "created_utc": datetime.now(UTC).isoformat(),
            "authority": "LOCAL_DURABILITY_PROOF_ONLY",
            "axis": "macOS-MLX local environment; not contest CPU/CUDA",
            "score_claim": False,
            "frontier_pointer_mutated": False,
            "status": failure["status"],
            "all_pass": False,
            "error": repr(exc),
            "exact_blocker": failure["exact_blocker"],
            "next_action": failure["next_action"],
            "input": {
                "gt_cache": str(args.gt_cache),
                "bytes": args.gt_cache.stat().st_size if args.gt_cache.is_file() else None,
                "num_pairs": int(args.num_pairs),
                "seed": int(args.seed),
                "mlx_device": str(args.mlx_device),
            },
            "checkpoint_files": failure["checkpoint_files"],
            "crash_epoch": failure["crash_epoch"],
            "final_pair_by_arm": failure["final_pair_by_arm"],
            "final_pair_preserved": failure["final_pair_preserved"],
            "dsl_artifact_hashes": artifact_hashes,
            "cleanup": {
                "policy": "failure-preserve for exact blocker custody",
                "base_preserved": str(base),
                "source_gt_deleted": False,
            },
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "all_pass": False, "error": repr(exc), "base_preserved": str(args.base),
            "blocker_receipt": str(args.receipt),
        }), file=sys.stderr)
        raise
    print(json.dumps({"all_pass": report["all_pass"], "receipt": str(args.receipt), "wall_seconds": report["wall_seconds"]}))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
