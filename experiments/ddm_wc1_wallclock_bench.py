#!/usr/bin/env python3
"""WC1 one-shot wall-clock bench for the lifted PR130 semantic renderer.

Default behavior is plan-only: write the guarded ticket plus planned receipt
rows, but do not touch Metal. Pass --execute only in the MAIN-owned Metal gap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DRIVER = Path("experiments/ddm_mx1_pr130_semantic_renderer.py")
FIRE_GUARD = Path("tools/mx1_fire_guard.py")
SAFE_RUN = Path("tools/safe_run.py")
DEFAULT_SOURCE_TICKET = Path(".omx/research/ddm_mx1e_20260807/regen2/probe_result.json")
DEFAULT_SOURCE_ARGV_KEY = "argv_n32_arm_cap"
DEFAULT_OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_wc1_20260807/wallclock_bench")
DEFAULT_TICKET_OUT = Path(".omx/research/ddm_wc1_20260807/wc1_bench_ticket.json")
DEFAULT_RECEIPTS_JSONL = Path(".omx/research/ddm_wc1_20260807/wc1_bench_receipts.jsonl")
RECEIPT_SCHEMA = "ddm_wc1_wallclock_bench_receipt.v1"
TICKET_SCHEMA = "ddm_wc1_wallclock_bench_ticket.v1"
THREAD_PIN_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MLX_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Variant:
    name: str
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    concurrent_cpu_verdict: bool = False
    ane_parity: bool = False


VARIANTS = {
    "baseline": Variant("baseline"),
    "threads": Variant("threads", ("--perf-thread-pin", "one"), THREAD_PIN_ENV),
    "batched": Variant("batched", ("--microbatch-pairs", "32")),
    "compile": Variant("compile", ("--compile-train-loss",)),
    "fp16-train": Variant("fp16-train", ("--train-compute-dtype", "fp16")),
    "hygiene-step": Variant("hygiene-step", ("--microbatch-hygiene", "per-step")),
    "chunk-cache": Variant("chunk-cache", ("--microbatch-chunk-cache",)),
    "saturated": Variant(
        "saturated",
        ("--train-compute-dtype", "fp16", "--microbatch-hygiene", "per-step",
         "--microbatch-chunk-cache"),
    ),
    "ram-cache": Variant("ram-cache", ("--cache-residency", "ram-full")),
    "derived-microbatch-4": Variant("derived-microbatch-4", ("--microbatch-policy", "auto")),
    "concurrent-cpu-verdict": Variant("concurrent-cpu-verdict", concurrent_cpu_verdict=True),
    "ane-verdict": Variant("ane-verdict", ane_parity=True),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def unwrap_safe_run(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1 :] if "--" in argv else argv


def flag_map(argv: list[str]) -> dict[str, str]:
    raw = unwrap_safe_run(argv)
    out: dict[str, str] = {}
    index = 0
    while index < len(raw):
        token = raw[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if index + 1 < len(raw) and not raw[index + 1].startswith("--"):
                out[key] = raw[index + 1]
                index += 2
                continue
            out[key] = "true"
        index += 1
    return out


def load_source_flags(ticket_path: Path, argv_key: str) -> dict[str, str]:
    payload = json.loads(ticket_path.read_text(encoding="utf-8"))
    ticket = payload.get("launch_ticket", payload)
    argv = ticket.get(argv_key)
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError(f"{ticket_path} missing string-list argv key {argv_key!r}")
    flags = flag_map(argv)
    if "run_dir" not in flags:
        raise ValueError(f"{argv_key} has no --run-dir; cannot derive resume checkpoint")
    return flags


def checkpoint_step_npz(path: Path) -> int:
    with np.load(path, allow_pickle=False) as payload:
        if "meta::step" not in payload.files:
            raise ValueError(f"checkpoint {path} missing meta::step")
        return int(payload["meta::step"][0])


def safe_run_wrap(
    inner: list[str],
    *,
    label: str,
    status_receipt: Path,
    child_pidfile: Path,
    rss_mb: int,
    timeout_s: float,
    projected_gib: float,
) -> list[str]:
    return [
        ".venv/bin/python",
        str(SAFE_RUN),
        "--rss-mb",
        str(rss_mb),
        "--timeout",
        str(timeout_s),
        "--projected-gib",
        str(projected_gib),
        "--label",
        label,
        "--status-receipt",
        str(status_receipt),
        "--child-pidfile",
        str(child_pidfile),
        "--",
        *inner,
    ]


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_flags = load_source_flags(args.source_ticket, args.source_argv_key)
    resume_from = args.resume_from or Path(source_flags.get("resume_from") or Path(source_flags["run_dir"]) / "mlx.latest.npz")
    resume_step = checkpoint_step_npz(resume_from)
    step_horizon = resume_step + int(args.bench_steps)
    attempt_id = f"{utc_now().replace(':', '').replace('-', '').replace('.', '_')}_pid{os.getpid()}"
    ticket_path = args.ticket_out
    selected_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in selected_names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; choices={sorted(VARIANTS)}")

    ticket: dict[str, Any] = {
        "schema": TICKET_SCHEMA,
        "score_claim": False,
        "axis": "[macOS-MLX research-signal bench harness]",
        "created_at_utc": utc_now(),
        "source_ticket": str(args.source_ticket),
        "source_argv_key": args.source_argv_key,
        "resume_from": str(resume_from),
        "resume_step": resume_step,
        "bench_steps_after_resume": int(args.bench_steps),
        "step_horizon": step_horizon,
        "mem_probe_receipt_required": True,
        "mem_probe_receipt_paths": {},
        "fire_guard_required": True,
        "receipt_schema": RECEIPT_SCHEMA,
        "wc2_extension": {
            "schema": "ddm_wc2_bench_extension.v1",
            "score_claim": False,
            "added_variants": [
                "ram-cache",
                "derived-microbatch-4",
                "concurrent-cpu-verdict",
                "ane-verdict",
            ],
            "cpu_verdict_contract": "subprocess process-group; OMP/MKL/OpenBLAS/vecLib/NumExpr <=4",
            "ane_contract": "CoreML FP32 SegNet parity must pass on real rendered frames before ANE verdict use",
        },
    }
    rows: list[dict[str, Any]] = []

    for name in selected_names:
        variant = VARIANTS[name]
        argv_key = f"argv_wc1_{name.replace('-', '_')}"
        run_dir = args.output_root / attempt_id / name
        mem_probe_dir = run_dir / "mem_probe"
        fire_guard_path = run_dir / "fire_guard" / f"{argv_key}.json"
        result_path = run_dir / "result.json"
        mem_probe_receipt = mem_probe_dir / "mem_probe_receipt.json"
        base_train = [
            ".venv/bin/python",
            str(DRIVER),
            "--mode",
            "mlx-train",
            "--device",
            "gpu",
            "--pairs",
            str(args.pairs),
            "--steps",
            str(step_horizon),
            "--lr",
            source_flags.get("lr", "2e-07"),
            "--ce-fraction",
            source_flags.get("ce_fraction", "0.0"),
            "--softplus-fraction",
            source_flags.get("softplus_fraction", "-999.0"),
            "--bits",
            source_flags.get("bits", "4"),
            "--seed",
            source_flags.get("seed", str(args.seed)),
            "--checkpoint-every",
            str(max(1, int(args.bench_steps))),
            "--eval-every",
            "1",
            "--input-cache",
            source_flags["input_cache"],
            "--target-cache",
            source_flags["target_cache"],
            "--init",
            source_flags["init"],
            "--run-dir",
            str(run_dir),
            "--out",
            str(result_path),
            "--resume-from",
            str(resume_from),
            "--fire-guard-verdict",
            str(fire_guard_path),
            "--launch-ticket-path",
            str(ticket_path),
            "--fire-argv-key",
            argv_key,
            *variant.extra_args,
        ]
        mem_probe = [
            ".venv/bin/python",
            str(DRIVER),
            "--mode",
            "mem-probe",
            "--device",
            "gpu",
            "--pairs",
            str(args.pairs),
            "--mem-probe-steps",
            str(args.mem_probe_steps),
            "--lr",
            source_flags.get("lr", "2e-07"),
            "--ce-fraction",
            source_flags.get("ce_fraction", "0.0"),
            "--softplus-fraction",
            source_flags.get("softplus_fraction", "-999.0"),
            "--bits",
            source_flags.get("bits", "4"),
            "--seed",
            source_flags.get("seed", str(args.seed)),
            "--checkpoint-every",
            str(max(1, int(args.mem_probe_steps))),
            "--eval-every",
            "1",
            "--input-cache",
            source_flags["input_cache"],
            "--target-cache",
            source_flags["target_cache"],
            "--init",
            source_flags["init"],
            "--run-dir",
            str(mem_probe_dir),
            "--out",
            str(mem_probe_dir / "mem_probe_result.json"),
            "--resume-from",
            str(resume_from),
            *variant.extra_args,
        ]
        safe_train = safe_run_wrap(
            base_train,
            label=f"ddm_wc1_{name}",
            status_receipt=run_dir / "safe_run" / f"{argv_key}.status.json",
            child_pidfile=run_dir / "safe_run" / f"{argv_key}.child.pid",
            rss_mb=args.rss_mb,
            timeout_s=args.timeout_s,
            projected_gib=args.projected_gib,
        )
        safe_probe = safe_run_wrap(
            mem_probe,
            label=f"ddm_wc1_{name}_mem_probe",
            status_receipt=mem_probe_dir / "safe_run.status.json",
            child_pidfile=mem_probe_dir / "safe_run.child.pid",
            rss_mb=args.rss_mb,
            timeout_s=args.timeout_s,
            projected_gib=args.projected_gib,
        )
        guard = [
            ".venv/bin/python",
            str(FIRE_GUARD),
            "--ticket",
            str(ticket_path),
            "--argv-key",
            argv_key,
            "--out",
            str(fire_guard_path),
        ]
        cpu_verdict_result = run_dir / "cpu_verdict" / "result.json"
        cpu_verdict = [
            ".venv/bin/python",
            str(DRIVER),
            "--mode",
            "torch-verdict",
            "--device",
            "cpu",
            "--input-cache",
            source_flags["input_cache"],
            "--target-cache",
            source_flags["target_cache"],
            "--init",
            str(resume_from),
            "--out",
            str(cpu_verdict_result),
            "--verdict-batch-size",
            str(int(getattr(args, "concurrent_verdict_batch_size", 16))),
        ]
        ane_parity_result = run_dir / "ane_parity" / "result.json"
        ane_parity = [
            ".venv/bin/python",
            str(DRIVER),
            "--mode",
            "coreml-segnet-parity",
            "--device",
            "cpu",
            "--pairs",
            str(int(getattr(args, "ane_parity_pairs", args.pairs))),
            "--input-cache",
            source_flags["input_cache"],
            "--target-cache",
            source_flags["target_cache"],
            "--init",
            str(resume_from),
            "--out",
            str(ane_parity_result),
            "--verdict-batch-size",
            str(int(getattr(args, "concurrent_verdict_batch_size", 16))),
            "--coreml-compute-units",
            str(getattr(args, "coreml_compute_units", "CPU_AND_NE")),
        ]
        ticket[argv_key] = safe_train
        ticket["mem_probe_receipt_paths"][argv_key] = str(mem_probe_receipt)
        if variant.concurrent_cpu_verdict:
            ticket[f"{argv_key}_concurrent_cpu_verdict"] = cpu_verdict
        if variant.ane_parity:
            ticket[f"{argv_key}_ane_parity"] = ane_parity
        rows.append(
            {
                "schema": RECEIPT_SCHEMA,
                "status": "planned",
                "score_claim": False,
                "axis": "[macOS-MLX research-signal bench harness]",
                "variant": name,
                "argv_key": argv_key,
                "env_overrides": variant.env or {},
                "run_dir": str(run_dir),
                "mem_probe_receipt": str(mem_probe_receipt),
                "fire_guard_verdict": str(fire_guard_path),
                "result_path": str(result_path),
                "resume_from": str(resume_from),
                "resume_step": resume_step,
                "step_horizon": step_horizon,
                "bench_steps_after_resume": int(args.bench_steps),
                "mem_probe_command": safe_probe,
                "fire_guard_command": guard,
                "train_command": safe_train,
                "concurrent_cpu_verdict": variant.concurrent_cpu_verdict,
                "cpu_verdict_command": cpu_verdict if variant.concurrent_cpu_verdict else None,
                "cpu_verdict_result_path": str(cpu_verdict_result) if variant.concurrent_cpu_verdict else None,
                "ane_parity": variant.ane_parity,
                "ane_parity_command": ane_parity if variant.ane_parity else None,
                "ane_parity_result_path": str(ane_parity_result) if variant.ane_parity else None,
                "seconds_per_step": None,
                "d_seg_batch_sanity": None,
            }
        )
    return ticket, rows


def extract_train_metrics(result_path: Path) -> dict[str, Any]:
    if not result_path.exists():
        return {"result_exists": False}
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    train = payload.get("mlx_train") or {}
    history = train.get("history") or []
    dseg_rows = [row for row in history if isinstance(row, dict) and "d_seg_batch" in row]
    return {
        "result_exists": True,
        "result_sha256": sha256_file(result_path),
        "seconds_per_step": train.get("seconds_per_step"),
        "d_seg_batch_sanity": None if not dseg_rows else dseg_rows[-1].get("d_seg_batch"),
        "last_history_step": None if not history else history[-1].get("step"),
        "status": payload.get("status"),
    }


def extract_cpu_verdict_metrics(result_path: Path) -> dict[str, Any]:
    if not result_path.exists():
        return {"cpu_verdict_result_exists": False}
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    verdict = payload.get("torch_verdict") or {}
    return {
        "cpu_verdict_result_exists": True,
        "cpu_verdict_result_sha256": sha256_file(result_path),
        "cpu_verdict_status": verdict.get("status", payload.get("status")),
        "cpu_verdict_aggregate_d_seg": verdict.get("aggregate_d_seg"),
        "cpu_verdict_elapsed_seconds": verdict.get("elapsed_seconds"),
        "cpu_verdict_batch_size": verdict.get("segnet_batch_size"),
    }


def run_logged(command: list[str], *, env_overrides: dict[str, str], stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(env_overrides)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, check=False)
    return {
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def start_logged_async(
    command: list[str],
    *,
    env_overrides: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[subprocess.Popen[None], dict[str, Any]]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(env_overrides)
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    return proc, {
        "pid": proc.pid,
        "started_at_monotonic": time.time(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "_stdout_handle": stdout,
        "_stderr_handle": stderr,
    }


def finish_logged_async(
    proc: subprocess.Popen[None],
    state: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    started = float(state.pop("started_at_monotonic"))
    stdout = state.pop("_stdout_handle")
    stderr = state.pop("_stderr_handle")
    timed_out = False
    killed = False
    try:
        try:
            returncode = proc.wait(timeout=max(0.0, timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            killed = True
            try:
                returncode = proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait(timeout=10.0)
    finally:
        stdout.close()
        stderr.close()
    return {
        **state,
        "returncode": returncode,
        "elapsed_seconds": time.time() - started,
        "timed_out": timed_out,
        "process_group_reclaimed": killed,
    }


def extract_coreml_parity_metrics(result_path: Path) -> dict[str, Any]:
    if not result_path.exists():
        return {"ane_result_exists": False}
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    row = payload.get("coreml_segnet_parity") or {}
    return {
        "ane_result_exists": True,
        "ane_result_sha256": sha256_file(result_path),
        "ane_status": row.get("status", payload.get("status")),
        "ane_argmax_diff_pixels": row.get("argmax_diff_pixels"),
        "ane_argmax_diff_rate": row.get("argmax_diff_rate"),
        "ane_logit_abs_max_delta": row.get("logit_abs_max_delta"),
        "ane_blocker": row.get("blocker"),
    }


def execute_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    bench_started = time.time()
    for row in rows:
        variant = row["variant"]
        if time.time() - bench_started > float(args.metal_budget_s):
            row = dict(row)
            row["status"] = "skipped_metal_budget_exhausted"
            executed.append(row)
            continue
        env_overrides = dict(row.get("env_overrides") or {})
        run_dir = Path(row["run_dir"])
        row = dict(row)
        row["started_at_utc"] = utc_now()
        row["mem_probe_run"] = run_logged(
            row["mem_probe_command"],
            env_overrides=env_overrides,
            stdout_path=run_dir / "logs" / "mem_probe.stdout.log",
            stderr_path=run_dir / "logs" / "mem_probe.stderr.log",
        )
        if row["mem_probe_run"]["returncode"] != 0:
            row["status"] = "mem_probe_failed"
            row["finished_at_utc"] = utc_now()
            executed.append(row)
            continue
        row["fire_guard_run"] = run_logged(
            row["fire_guard_command"],
            env_overrides={},
            stdout_path=run_dir / "logs" / "fire_guard.stdout.log",
            stderr_path=run_dir / "logs" / "fire_guard.stderr.log",
        )
        if row["fire_guard_run"]["returncode"] != 0:
            row["status"] = "fire_guard_refused"
            row["finished_at_utc"] = utc_now()
            executed.append(row)
            continue
        if row.get("ane_parity_command"):
            row["ane_parity_run"] = run_logged(
                row["ane_parity_command"],
                env_overrides={},
                stdout_path=run_dir / "logs" / "ane_parity.stdout.log",
                stderr_path=run_dir / "logs" / "ane_parity.stderr.log",
            )
            row.update(extract_coreml_parity_metrics(Path(row["ane_parity_result_path"])))
            if row["ane_parity_run"]["returncode"] != 0 or row.get("ane_status") != "passed":
                row["status"] = "ane_parity_blocked_or_failed"
                row["finished_at_utc"] = utc_now()
                executed.append(row)
                continue
        async_cpu: tuple[subprocess.Popen[None], dict[str, Any]] | None = None
        if row.get("cpu_verdict_command"):
            cpu_env = {
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "VECLIB_MAXIMUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4",
            }
            async_cpu = start_logged_async(
                row["cpu_verdict_command"],
                env_overrides=cpu_env,
                stdout_path=run_dir / "logs" / "cpu_verdict.stdout.log",
                stderr_path=run_dir / "logs" / "cpu_verdict.stderr.log",
            )
        row["train_run"] = run_logged(
            row["train_command"],
            env_overrides=env_overrides,
            stdout_path=run_dir / "logs" / "train.stdout.log",
            stderr_path=run_dir / "logs" / "train.stderr.log",
        )
        if async_cpu is not None:
            proc, state = async_cpu
            row["cpu_verdict_run"] = finish_logged_async(
                proc,
                state,
                timeout_s=float(getattr(args, "cpu_verdict_timeout_s", 900.0)),
            )
            row["cpu_verdict_result"] = extract_cpu_verdict_metrics(Path(row["cpu_verdict_result_path"]))
        metrics = extract_train_metrics(Path(row["result_path"]))
        row.update(metrics)
        row["status"] = "passed" if row["train_run"]["returncode"] == 0 else "train_failed"
        row["finished_at_utc"] = utc_now()
        executed.append(row)
        print(json.dumps({"variant": variant, "status": row["status"], **metrics}, sort_keys=True), flush=True)
    return executed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ticket", type=Path, default=DEFAULT_SOURCE_TICKET)
    parser.add_argument("--source-argv-key", default=DEFAULT_SOURCE_ARGV_KEY)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ticket-out", type=Path, default=DEFAULT_TICKET_OUT)
    parser.add_argument("--receipts-jsonl", type=Path, default=DEFAULT_RECEIPTS_JSONL)
    parser.add_argument(
        "--variants",
        default=(
            "baseline,threads,batched,compile,fp16-train,"
            "ram-cache,derived-microbatch-4,concurrent-cpu-verdict,ane-verdict"
        ),
    )
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--bench-steps", type=int, default=5)
    parser.add_argument("--mem-probe-steps", type=int, default=2)
    parser.add_argument("--rss-mb", type=int, default=45_000)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--projected-gib", type=float, default=24.0)
    parser.add_argument("--metal-budget-s", type=float, default=600.0)
    parser.add_argument("--concurrent-verdict-batch-size", type=int, default=16)
    parser.add_argument("--cpu-verdict-timeout-s", type=float, default=900.0)
    parser.add_argument("--ane-parity-pairs", type=int, default=32)
    parser.add_argument("--coreml-compute-units", default="CPU_AND_NE")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.pairs <= 0 or args.bench_steps <= 0 or args.mem_probe_steps <= 0:
        parser.error("--pairs, --bench-steps, and --mem-probe-steps must be positive")
    if args.concurrent_verdict_batch_size <= 0 or args.ane_parity_pairs <= 0:
        parser.error("--concurrent-verdict-batch-size and --ane-parity-pairs must be positive")
    ticket, rows = build_plan(args)
    ticket["host"] = {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    write_json_atomic(args.ticket_out, ticket)
    final_rows = execute_rows(args, rows) if args.execute else rows
    write_jsonl_atomic(args.receipts_jsonl, final_rows)
    print(
        json.dumps(
            {
                "schema": "ddm_wc1_wallclock_bench_driver.v1",
                "status": "executed" if args.execute else "planned",
                "ticket": str(args.ticket_out),
                "ticket_sha256": sha256_file(args.ticket_out),
                "receipts_jsonl": str(args.receipts_jsonl),
                "receipts_jsonl_sha256": sha256_file(args.receipts_jsonl),
                "rows": len(final_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
