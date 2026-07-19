#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #537 real-data crash/resume proof on the live MLX trainer.

This is an evidence probe, not a score run.  It uses the real n>=24 ground-truth
cache, kills a trainer after an atomic intra-stage checkpoint, resumes in a new
directory, and compares the final live and EMA tensors with an uninterrupted
control.  On success it removes the rebuildable checkpoint bulk after first
writing a machine-readable receipt; on failure it leaves the bytes in place.
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


def _resume_epoch(path: Path) -> int:
    with np.load(path, allow_pickle=False) as z:
        return int(np.asarray(z["__resume_epoch"]).item())


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
    out_dir: Path, gt_cache: Path, *, resume_from: Path | None = None,
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

    identity = lambda value: Provenanced(
        value=value,
        provenance=ProvenanceClass.HARDCODED_WITH_WAIVER,
        unit="dimensionless",
        source="Task #537 bounded durability-proof identity schedule",
        waiver=("Proof-only identity constant; re-derive if pair count, cache geometry, or epoch "
                "budget changes"),
    )
    overrides: dict[str, Any] = {
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
        num_pairs=24,
        epochs=4,
        wall_clock_budget_days=Provenanced(
            value=derive_wall_clock_budget_days(4),
            provenance=ProvenanceClass.DERIVED_AT_CONFIG,
            unit="days",
            source="scorer_throughput_gate.derive_wall_clock_budget_days(epochs)",
        ),
        mlx_device="cpu",
        seed=537,
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
        stderr=subprocess.STDOUT, bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    killed = False
    deadline = time.monotonic() + timeout
    try:
        for line in proc.stdout:
            lines.append(line.rstrip())
            if '"kind": "intra_stage"' in line and '"epoch": 3' in line:
                os.kill(proc.pid, signal.SIGKILL)
                killed = True
                break
            if time.monotonic() > deadline:
                os.kill(proc.pid, signal.SIGKILL)
                raise TimeoutError("trainer did not reach the epoch-3 intra-stage checkpoint")
    finally:
        proc.stdout.close()
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.kill(proc.pid, signal.SIGKILL)
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
    continuous = base / "continuous"
    crash = base / "crash"
    resumed = base / "resumed"
    for path in (continuous, crash, resumed):
        path.mkdir(parents=True, exist_ok=False)

    continuous_argv, continuous_custody = _prepare_dsl_launch(continuous, gt_cache)
    crash_argv, crash_custody = _prepare_dsl_launch(crash, gt_cache)
    resume_argv, resume_custody = _prepare_dsl_launch(
        resumed, gt_cache, resume_from=crash,
    )
    started = time.monotonic()
    cont = _run(continuous_argv, args.timeout)
    if cont.returncode != 0:
        raise RuntimeError(f"continuous arm failed rc={cont.returncode}\n{cont.stdout[-3000:]}\n{cont.stderr[-3000:]}")
    crash_rc, killed, crash_lines = _run_and_kill(crash_argv, args.timeout)
    crash_state = crash / "levelset_resume_state.npz"
    if not killed or not crash_state.is_file() or _resume_epoch(crash_state) != 3:
        raise RuntimeError(
            f"crash arm proof failed: killed={killed} rc={crash_rc} "
            f"state={crash_state.is_file()} epoch={_resume_epoch(crash_state) if crash_state.is_file() else None}"
        )
    res = _run(resume_argv, args.timeout)
    if res.returncode != 0:
        raise RuntimeError(f"resume arm failed rc={res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")

    cont_state = continuous / "levelset_resume_state.npz"
    resumed_state = resumed / "levelset_resume_state.npz"
    cont_hashes = _state_hashes(cont_state)
    resumed_hashes = _state_hashes(resumed_state)
    cont_trajectory = _trajectory_state_hashes(cont_state)
    resumed_trajectory = _trajectory_state_hashes(resumed_state)
    resume_rows = _json_rows(res.stdout)
    resume_row = next((row for row in resume_rows if row.get("stage") == "resume"), {})
    reanchor_rows = [row for row in resume_rows if "reanchor" in str(row.get("stage", "")).lower()]

    periodic = sorted(continuous.glob("*periodic*ep*.npz"))
    final = sorted(continuous.glob("levelset_*stage*_ep4.npz"))
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
        params, _cfg = _load_levelset_ckpt(continuous, npz_name=checkpoint.name)
        byte_close_loaded[checkpoint.name] = len(params)
    all_checkpoint_hashes = {p.name: _sha256(p) for p in sorted(continuous.glob("*.npz"))}
    assertions = {
        "real_num_pairs_gte_24": True,
        "sigkill_observed": killed and crash_rc < 0,
        "crash_epoch_is_3": _resume_epoch(crash_state) == 3,
        "resume_final_epoch_is_4": _resume_epoch(resumed_state) == 4,
        "restored_optimizer": bool(resume_row.get("restored_opt")),
        "live_and_ema_bit_identical": cont_hashes == resumed_hashes and bool(cont_hashes),
        "optimizer_rng_event_stage_bit_identical": (
            cont_trajectory == resumed_trajectory and bool(cont_trajectory)
        ),
        "periodic_pair_retention_bounded_to_1": all(len(v) == 1 for v in periodic_by_kind.values()),
        "periodic_ema_byte_close_loadable": len(byte_close_loaded) == 1,
        "final_pair_preserved": len(final) == 2,
        "exact_continuation_not_reanchored": not reanchor_rows,
    }
    all_pass = all(assertions.values())
    report: dict[str, Any] = {
        "schema": "tac.resumability_537_real_crash_resume.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "authority": "LOCAL_DURABILITY_PROOF_ONLY",
        "axis": "macOS-MLX-CPU advisory; not contest CPU/CUDA",
        "score_claim": False,
        "frontier_pointer_mutated": False,
        "git": _git_state(),
        "input": {"gt_cache": str(gt_cache), "bytes": gt_cache.stat().st_size, "sha256": _sha256(gt_cache), "num_pairs": 24},
        "commands": {"continuous": continuous_argv, "crash": crash_argv, "resume": resume_argv},
        "dsl_custody": {
            "continuous": continuous_custody,
            "crash": crash_custody,
            "resume": resume_custody,
        },
        "crash": {"returncode": crash_rc, "last_rows": crash_lines[-12:]},
        "resume": {"returncode": res.returncode, "resume_row": resume_row, "reanchor_rows": reanchor_rows},
        "checkpoint_files": {
            "periodic": periodic_by_kind,
            "final": [p.name for p in final],
            "sha256": all_checkpoint_hashes,
            "byte_close_loaded_param_counts": byte_close_loaded,
        },
        "tensor_state": {
            "count": len(cont_hashes), "continuous": cont_hashes, "resumed": resumed_hashes,
            "trajectory_count": len(cont_trajectory),
            "continuous_trajectory": cont_trajectory,
            "resumed_trajectory": resumed_trajectory,
        },
        "assertions": assertions,
        "all_pass": all_pass,
        "wall_seconds": round(time.monotonic() - started, 3),
        "cleanup": {"policy": "success-only rebuildable scratch", "source_gt_deleted": False, "proof_dirs_deleted": False},
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not all_pass:
        raise AssertionError(json.dumps(assertions, indent=2))
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
    args = ap.parse_args()
    try:
        with _worktree_venv(args.python):
            report = run(args)
    except BaseException as exc:
        artifact_hashes: dict[str, str] = {}
        if args.base.exists():
            for path in sorted(args.base.glob("*/**/*")):
                if path.is_file() and path.name in {
                    "launch.sh", "dsl_provenance.json", "launch_manifest.json",
                }:
                    artifact_hashes[str(path.relative_to(args.base))] = _sha256(path)
        blocker = {
            "schema": "tac.resumability_537_real_crash_resume.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "authority": "LOCAL_DURABILITY_PROOF_ONLY",
            "axis": "macOS-MLX local environment; not contest CPU/CUDA",
            "score_claim": False,
            "frontier_pointer_mutated": False,
            "status": "BLOCKED_ENVIRONMENT_NO_METAL_DEVICE",
            "all_pass": False,
            "error": repr(exc),
            "exact_blocker": (
                "MLX imports mlx.nn through mx.compile and refuses because this execution "
                "environment exposes no Metal device; no trainer epoch or checkpoint ran"
            ),
            "next_action": (
                "MAIN must rerun this exact probe on the Metal-capable host; do not promote the "
                "real crash-resume proof until the receipt becomes all_pass=true"
            ),
            "input": {
                "gt_cache": str(args.gt_cache),
                "bytes": args.gt_cache.stat().st_size if args.gt_cache.is_file() else None,
                "num_pairs": 24,
            },
            "dsl_artifact_hashes": artifact_hashes,
            "cleanup": {
                "policy": "failure-preserve for exact blocker custody",
                "base_preserved": str(args.base),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
