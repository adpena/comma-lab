#!/usr/bin/env python
"""ddm_mx1 PR130 semantic renderer lift/port driver.

This is a scorer-slot-free harness for Row-1 readiness.  It can:

* run a tiny lifted-torch CPU smoke on real label tensors;
* probe local MLX availability without hiding runtime failures;
* run the torch-vs-MLX parity gate when MLX is available;
* run a CPU-torch verdict over a saved MLX checkpoint's own pair set;
* emit a MAIN launch ticket for the n32 -> n120 stratified Metal run.

It does not run n600 scorer work and does not claim a contest score.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.admission_guard import assert_governed_admission
from tac.optimization.trajectory_stopping import (
    StaircaseStopConfig,
    TrajectoryStopConfig,
    build_cap_stop_receipt,
    evaluate_staircase_aware_stop,
)
from tac.pr130_lift import SOURCE_REPO_HEAD, SOURCE_REPO_ROOT
from tac.pr130_lift.checkpoint_schema import architecture_config_from_checkpoint
from tac.pr130_lift.mlx_semantic_renderer import (
    MlxSemanticConfig,
    MlxUnavailableError,
    curriculum_loss_mlx,
    fake_quantize_parameter_tree,
    load_stage_checkpoint_npz,
    load_torch_state_dict_into_mlx,
    make_mlx_renderer,
    mlx_device_probe,
    require_mlx,
    save_stage_checkpoint_npz,
)

REPO = Path(__file__).resolve().parents[1]
LIFTED = REPO / "src" / "tac" / "pr130_lift" / "lifted"
SSD_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_mx1_20260806")
DEFAULT_INPUT_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt")
DEFAULT_TARGET_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt")
DEFAULT_INIT = Path(
    "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/"
    "repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt"
)
CONTEST_DENOMINATOR_BYTES = 37_545_489
GIB = 1024.0**3
TRAIN_COMPUTE_DTYPES = ("fp32", "bf16", "fp16")
THREAD_PIN_MODES = ("off", "one")
CACHE_RESIDENCY_MODES = ("selected", "ram-full")
MICROBATCH_POLICIES = ("auto", "legacy-4", "full")
MICROBATCH_HYGIENE_MODES = ("per-chunk", "per-step", "off")
MEM_PROBE_RECEIPT_SCHEMA = "ddm_mx1_load_phase_peak_receipt.v1"
MX1_FIRE_GUARD_VERDICT_SCHEMA = "ddm_mx1_fire_guard_verdict.v1"
SAFE_RUN_RECEIPT_SENTINEL = "REQUIRES_FRESH_MEM_PROBE"
SAFE_RUN_RECEIPT_PROJECTION_SCHEMA = "ddm_mx1_receipt_derived_safe_run_projection.v1"
SAFE_RUN_PROJECTION_MULTIPLIER = 1.5
SAFE_RUN_PROJECTED_GIB_FLOOR = 15
SAFE_RUN_RSS_MB_FLOOR = 45_000
ROW1_SAFE_RUN_RSS_MB = 90_000
ROW1_SAFE_RUN_TIMEOUT_S = 28_800
DEFAULT_WIRED_LIMIT_FRACTION = 0.35
FP1_FLAT_PAINT_FLOOR_D_SEG = 0.008305
MX1H_STEP1500_AUTHORITY_D_SEG = 0.0010689099629720051
MX1T_DEFAULT_CHECKPOINT_DIR = Path(".omx/research/ddm_mx1e_20260807/regen2/launch_arm_cap/n32_metal")
MX1T_DEFAULT_OUT_DIR = Path(".omx/research/ddm_mx1t_20260807")
M1_EVAL_JOURNAL_SCHEMA = "ddm_m1_eval_journal.v1"
M1_STOP_DECISION_SCHEMA = "ddm_m1_stop_decision.v1"
M1_TERMINAL_RECEIPT_SCHEMA = "ddm_m1_terminal_receipt.v1"
M1_SCHEDULE_SELECTION_SCHEMA = "ddm_m1_schedule_selection.v1"
EMA_DECAY_LAW_REF = "ema_decay_run_geometry_v1"
SEG_CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
WC2_AUTO_MICROBATCH_ANCHOR = {
    "schema": "ddm_wc2_microbatch_law_anchor.v1",
    "axis": "[macOS-MLX research-signal bench harness]",
    "source_receipt": ".omx/research/ddm_wc1_20260807/wc1_bench_receipts.jsonl",
    "source_commit": "wc1 charter pin 26f6a5aa3d; measured receipts consumed 2026-08-08",
    "n32_baseline_microbatch_pairs": 4,
    "n32_baseline_seconds_per_step": 10.441456365585328,
    "n32_full_batch_microbatch_pairs": 32,
    "n32_full_batch_seconds_per_step": 11.596535015106202,
    "selected_default": 4,
    "selection_rule": (
        "use the fastest measured safe n32 footprint until a fresh same-vehicle "
        "bench proves a larger microbatch wins under the same guard"
    ),
}
MARGIN_BINS: tuple[tuple[str, float, float | None], ...] = (
    ("0-0.05", 0.0, 0.05),
    ("0.05-0.1", 0.05, 0.1),
    ("0.1-0.25", 0.1, 0.25),
    ("0.25-0.5", 0.25, 0.5),
    (">0.5", 0.5, None),
)


class MemoryLimitConfigurationError(RuntimeError):
    """Raised when GPU mode cannot install a fail-closed software budget."""


class MemoryBudgetExceeded(RuntimeError):
    """Raised when measured MLX active memory plus RSS delta exceeds budget."""

    def __init__(self, message: str, *, check: dict[str, Any]) -> None:
        super().__init__(message)
        self.check = check


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl_durable(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one complete JSON line with one O_APPEND write and an fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), sort_keys=True, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(fd, encoded)
        if written != len(encoded):
            raise OSError(f"short append to {path}: {written} != {len(encoded)}")
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_active_m1_eval_rows(path: Path) -> list[dict[str, Any]]:
    """Return the resume-active journal view without deleting abandoned tail rows."""

    active: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"journal {path}:{line_number} is not a complete JSONL row")
            row = json.loads(line)
            if row.get("schema") != M1_EVAL_JOURNAL_SCHEMA:
                raise ValueError(f"journal {path}:{line_number} has wrong schema")
            if row.get("row_kind") == "segment_start":
                resume_step = int(row["resume_step"])
                active = {step: item for step, item in active.items() if step <= resume_step}
            elif row.get("row_kind") == "eval":
                active[int(row["step"])] = row
            else:
                raise ValueError(f"journal {path}:{line_number} has unknown row_kind")
    return [active[step] for step in sorted(active)]


def _m1_cosine_lr(base_lr: float, step: int, schedule_horizon_steps: int) -> float:
    """Monotone cosine whose terminal value is held across resumed extensions."""

    if base_lr <= 0.0 or schedule_horizon_steps <= 0 or step < 0:
        raise ValueError("base_lr, schedule_horizon_steps, and step are out of domain")
    clamped_step = min(step, schedule_horizon_steps - 1)
    progress = clamped_step / max(schedule_horizon_steps - 1, 1)
    return base_lr * (0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress)))


def _load_m1_executor_policy(args: argparse.Namespace) -> dict[str, Any] | None:
    ticket_path = getattr(args, "launch_ticket_path", None)
    argv_key = str(getattr(args, "fire_argv_key", "") or "")
    if ticket_path is None or not argv_key:
        return None
    ticket = json.loads(Path(ticket_path).read_text(encoding="utf-8"))
    executor = dict((ticket.get("stop_policy") or {}).get("executor") or {})
    if argv_key not in set(executor.get("child_argv_keys") or []):
        return None
    predicate = dict((ticket.get("stop_policy") or {}).get("predicate") or {})
    n_pairs = int(predicate["N"])
    height = int(predicate["H"])
    width = int(predicate["W"])
    one_flip = 100.0 / float(n_pairs * height * width)
    if not math.isclose(one_flip, float(predicate["one_sample_flip_S"]), rel_tol=0.0, abs_tol=1e-18):
        raise ValueError("ticket one_sample_flip_S drifted from N*H*W geometry")
    eval_every = int(predicate["eval_every_steps"])
    event_free_evals = int(executor["event_free_horizon_evals"])
    trajectory_config = TrajectoryStopConfig(
        score_units_per_objective=1.0,
        marginal_score_gain_per_compute=float(predicate["marginal_bar_S_per_step"]),
        min_fit_points=int(predicate["min_eval_rows"]),
    )
    staircase_config = StaircaseStopConfig(
        min_eval_rows=int(predicate["min_eval_rows"]),
        window_rows=int(predicate["window_rows"]),
        event_free_horizon_compute=float(event_free_evals * eval_every),
        event_score_delta=one_flip,
        creep_score_per_compute=(100.0 * float(predicate["creep_eps_dseg_per_eval"]) / float(eval_every)),
        sustained_erosion_windows=int(predicate["sustained_erosion_windows"]),
    )
    return {
        "ticket": ticket,
        "ticket_path": Path(ticket_path),
        "argv_key": argv_key,
        "executor": executor,
        "predicate": predicate,
        "trajectory_config": trajectory_config,
        "staircase_config": staircase_config,
    }


def _derive_m1_ema_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    ema = dict(policy["executor"]["ema"])
    updates = int(ema["updates_per_run"])
    warmup_fraction = float(ema["warmup_fraction"])
    from tac.canonical_equations.evaluators import eval_ema_decay_run_geometry

    decay = float(
        eval_ema_decay_run_geometry(
            {
                "mode": "decay_from_warmup_fraction",
                "warmup_fraction": warmup_fraction,
                "updates_per_run": updates,
            }
        )
    )
    if not math.isclose(decay, float(ema["derived_decay"]), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("ticket EMA decay drifted from ema_decay_run_geometry_v1")
    return {**ema, "derived_decay": decay, "law_ref": EMA_DECAY_LAW_REF}


def _update_m1_ema_flat(
    ema_flat: Mapping[str, Any],
    live_flat: Mapping[str, Any],
    *,
    decay: float,
) -> dict[str, Any]:
    """Apply one real Polyak update while failing closed on tree drift."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0,1)")
    if set(ema_flat) != set(live_flat):
        raise ValueError("M1 EMA parameter set drifted during training")
    return {name: decay * ema_flat[name] + (1.0 - decay) * live_value for name, live_value in live_flat.items()}


def _write_tail_average_npz(
    member_paths: list[Path],
    out_path: Path,
    *,
    selection_extra: Mapping[str, Any],
) -> Path:
    """Materialize a loadable simple-mean checkpoint from the exact member NPZs."""

    if not member_paths:
        raise ValueError("tail average requires at least one checkpoint")
    loaded = [np.load(path, allow_pickle=False) for path in member_paths]
    try:
        keys = set(loaded[0].files)
        if any(set(item.files) != keys for item in loaded[1:]):
            raise ValueError("tail-average checkpoint key sets differ")
        payload = {key: np.asarray(loaded[-1][key]) for key in loaded[-1].files}
        for key in sorted(k for k in keys if k.startswith("param::")):
            shapes = {tuple(item[key].shape) for item in loaded}
            if len(shapes) != 1:
                raise ValueError(f"tail-average parameter shape mismatch for {key}")
            payload[key] = (
                np.stack([np.asarray(item[key], dtype=np.float64) for item in loaded], axis=0)
                .mean(axis=0)
                .astype(loaded[-1][key].dtype, copy=False)
            )
        old_extra = {}
        if "meta::extra_json" in payload:
            old_extra = json.loads(bytes(payload["meta::extra_json"]).decode("utf-8"))
        payload["meta::extra_json"] = np.frombuffer(
            json.dumps(old_extra | dict(selection_extra), sort_keys=True).encode("utf-8"),
            dtype=np.uint8,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(tmp, out_path)
        return out_path
    finally:
        for item in loaded:
            item.close()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _host_fingerprint() -> dict[str, str]:
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def _gib_or_none(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return round(float(num_bytes) / GIB, 6)


def _safe_label_token(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _apply_perf_thread_pin(mode: str, *, torch_module: Any | None = None) -> dict[str, Any]:
    mode = str(mode)
    if mode not in THREAD_PIN_MODES:
        raise ValueError(f"unknown --perf-thread-pin {mode!r}")
    if mode == "off":
        return {"mode": mode, "applied": False, "env": {}, "torch": {}}
    env_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MLX_NUM_THREADS",
    )
    before_env = {key: os.environ.get(key) for key in env_keys}
    for key in env_keys:
        os.environ[key] = "1"
    torch_obj = torch if torch_module is None else torch_module
    torch_report: dict[str, Any] = {}
    for attr, value in (("set_num_threads", 1), ("set_num_interop_threads", 1)):
        setter = getattr(torch_obj, attr, None)
        if setter is None:
            torch_report[attr] = {"status": "missing"}
            continue
        try:
            setter(value)
            torch_report[attr] = {"status": "applied", "value": value}
        except Exception as exc:
            torch_report[attr] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "mode": mode,
        "applied": True,
        "env": {key: {"before": before_env[key], "after": os.environ.get(key)} for key in env_keys},
        "torch": torch_report,
    }


def _checkpoint_step_npz(path: Path) -> int:
    with np.load(path, allow_pickle=False) as payload:
        if "meta::step" not in payload.files:
            raise ValueError(f"checkpoint {path} missing meta::step")
        return int(payload["meta::step"][0])


def _ticket_attempt_id() -> str:
    return f"{_safe_label_token(_utc_now_iso())}_pid{os.getpid()}"


def _sentinel_safe_run_projection(
    *,
    argv_key: str,
    receipt_path: Path,
    reason_code: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SAFE_RUN_RECEIPT_PROJECTION_SCHEMA,
        "axis": "[load-phase memory telemetry projection; score_claim=false]",
        "score_claim": False,
        "argv_key": argv_key,
        "status": "requires_fresh_mem_probe",
        "reason_code": reason_code,
        "detail": detail or {},
        "receipt_path": str(receipt_path),
        "receipt_sha256": None,
        "projected_gib": SAFE_RUN_RECEIPT_SENTINEL,
        "safe_run_rss_mb": SAFE_RUN_RECEIPT_SENTINEL,
        "safe_run_timeout_s": ROW1_SAFE_RUN_TIMEOUT_S,
        "margin_rule": (
            "fresh passed receipt required; when present use measured_peak=max(peak_rss_gib, "
            "peak_mlx_reported_gib, peak_mlx_active_gib+peak_mlx_cache_gib), "
            f"projected_gib=max({SAFE_RUN_PROJECTED_GIB_FLOOR}, ceil(measured_peak*"
            f"{SAFE_RUN_PROJECTION_MULTIPLIER})), rss_mb=max({SAFE_RUN_RSS_MB_FLOOR}, "
            "ceil(projected_gib*1024))"
        ),
        "fail_closed_rule": (
            f"{SAFE_RUN_RECEIPT_SENTINEL} is intentionally non-numeric; tools/safe_run.py argparse "
            "will refuse the wrapper before governor admission if this ticket is fired without a "
            "fresh passed mem-probe receipt."
        ),
    }


def _peak_candidate_gib(peak: Mapping[str, Any]) -> dict[str, float]:
    candidates: dict[str, float] = {}
    for key in ("peak_rss_gib", "peak_mlx_reported_gib", "peak_mlx_active_gib", "peak_mlx_cache_gib"):
        value = peak.get(key)
        if value is None:
            continue
        try:
            candidates[key] = float(value)
        except (TypeError, ValueError):
            continue
    if "peak_mlx_active_gib" in candidates or "peak_mlx_cache_gib" in candidates:
        candidates["peak_mlx_active_plus_cache_gib"] = candidates.get("peak_mlx_active_gib", 0.0) + candidates.get(
            "peak_mlx_cache_gib", 0.0
        )
    return candidates


def _derive_receipt_safe_run_projection(
    *,
    argv_key: str,
    raw_argv: list[str],
    receipt_path: Path,
) -> dict[str, Any]:
    try:
        repo_root = str(REPO)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from tools import mx1_fire_guard as guard
    except Exception as exc:
        return _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code="fire_guard_import_failed",
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )
    if not receipt_path.exists():
        return _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code="mem_probe_receipt_missing",
        )
    receipt_sha = _sha256_file(receipt_path)
    ok, reason, detail = guard._validate_receipt_freshness(receipt_path)
    if not ok:
        row = _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code=reason,
            detail=detail,
        )
        row["receipt_sha256"] = receipt_sha
        return row
    try:
        receipt = guard._load_json(receipt_path)
    except Exception as exc:
        row = _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code="mem_probe_receipt_parse_error",
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )
        row["receipt_sha256"] = receipt_sha
        return row
    if not isinstance(receipt, dict) or receipt.get("schema") != MEM_PROBE_RECEIPT_SCHEMA:
        return _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code="receipt_schema_mismatch",
            detail={"schema": None if not isinstance(receipt, dict) else receipt.get("schema")},
        )
    if receipt.get("status") != "passed" or receipt.get("metal_fire_clearance") is not True:
        row = _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code="receipt_status_not_clearance",
            detail={
                "status": receipt.get("status"),
                "metal_fire_clearance": receipt.get("metal_fire_clearance"),
            },
        )
        row["receipt_sha256"] = receipt_sha
        return row
    validation_checks = []
    for name, validator in (
        ("host", guard._validate_host),
        ("samples", guard._validate_samples),
        ("memory_limits", guard._validate_memory_limits),
    ):
        ok, reason, detail = validator(receipt)
        validation_checks.append(
            {
                "name": name,
                "status": "passed" if ok else "failed",
                "reason": reason,
                "detail": detail,
            }
        )
        if not ok:
            row = _sentinel_safe_run_projection(
                argv_key=argv_key,
                receipt_path=receipt_path,
                reason_code=reason,
                detail={"checks": validation_checks},
            )
            row["receipt_sha256"] = receipt_sha
            return row
    fire_config = guard._parsed_fire_config(raw_argv)
    receipt_config = guard._receipt_config(receipt)
    ok, reason, detail = guard._validate_config_match(fire_config, receipt_config)
    validation_checks.append(
        {
            "name": "config_match",
            "status": "passed" if ok else "failed",
            "reason": reason,
            "detail": detail,
        }
    )
    if not ok:
        row = _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code=reason,
            detail={"checks": validation_checks},
        )
        row["receipt_sha256"] = receipt_sha
        return row
    peak = receipt.get("peak")
    if not isinstance(peak, Mapping):
        row = _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code="receipt_peak_missing",
        )
        row["receipt_sha256"] = receipt_sha
        return row
    peak_candidates = _peak_candidate_gib(peak)
    if not peak_candidates:
        row = _sentinel_safe_run_projection(
            argv_key=argv_key,
            receipt_path=receipt_path,
            reason_code="receipt_peak_numeric_fields_missing",
            detail={"peak": peak},
        )
        row["receipt_sha256"] = receipt_sha
        return row
    measured_peak_gib = max(peak_candidates.values())
    projected_gib = max(
        SAFE_RUN_PROJECTED_GIB_FLOOR,
        math.ceil(measured_peak_gib * SAFE_RUN_PROJECTION_MULTIPLIER),
    )
    safe_run_rss_mb = max(SAFE_RUN_RSS_MB_FLOOR, math.ceil(projected_gib * 1024.0))
    return {
        "schema": SAFE_RUN_RECEIPT_PROJECTION_SCHEMA,
        "axis": "[load-phase memory telemetry projection; score_claim=false]",
        "score_claim": False,
        "argv_key": argv_key,
        "status": "passed",
        "reason_code": "receipt_projection_derived",
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "freshness_window_seconds": guard.RECEIPT_FRESHNESS_WINDOW_SECONDS,
        "measured_peak_gib": round(measured_peak_gib, 6),
        "peak_candidates_gib": {key: round(value, 6) for key, value in peak_candidates.items()},
        "projected_gib": projected_gib,
        "safe_run_rss_mb": safe_run_rss_mb,
        "safe_run_timeout_s": ROW1_SAFE_RUN_TIMEOUT_S,
        "margin_multiplier": SAFE_RUN_PROJECTION_MULTIPLIER,
        "projected_gib_floor": SAFE_RUN_PROJECTED_GIB_FLOOR,
        "rss_mb_floor": SAFE_RUN_RSS_MB_FLOOR,
        "margin_rule": (
            "measured_peak=max(peak_rss_gib, peak_mlx_reported_gib, "
            "peak_mlx_active_gib+peak_mlx_cache_gib); "
            f"projected_gib=max({SAFE_RUN_PROJECTED_GIB_FLOOR}, "
            f"ceil(measured_peak*{SAFE_RUN_PROJECTION_MULTIPLIER})); "
            f"rss_mb=max({SAFE_RUN_RSS_MB_FLOOR}, ceil(projected_gib*1024))"
        ),
        "validation_checks": validation_checks,
    }


def _safe_run_arg(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _wrap_fire_argv(
    raw_argv: list[str],
    *,
    label: str,
    projection: dict[str, Any],
    status_receipt: Path,
    child_pidfile: Path,
) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/safe_run.py",
        "--rss-mb",
        _safe_run_arg(projection["safe_run_rss_mb"]),
        "--timeout",
        str(ROW1_SAFE_RUN_TIMEOUT_S),
        "--projected-gib",
        _safe_run_arg(projection["projected_gib"]),
        "--label",
        label,
        "--status-receipt",
        str(status_receipt),
        "--child-pidfile",
        str(child_pidfile),
        "--",
        *raw_argv,
    ]


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().split()[0]) * 1024
    except Exception:
        return None
    return None


def _system_available_bytes() -> int | None:
    try:
        import psutil

        # RAW_VM_BASIS_OK: telemetry-only load-phase receipt/default limit hint,
        # not an admission guard or launch clearance.
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _system_total_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _mlx_allocator_bytes(mx: Any | None) -> dict[str, int | None]:
    if mx is None:
        return {"active": None, "cache": None, "peak": None}
    out: dict[str, int | None] = {}
    for key, names in {
        "active": ("get_active_memory", "metal.get_active_memory"),
        "cache": ("get_cache_memory", "metal.get_cache_memory"),
        "peak": ("get_peak_memory", "metal.get_peak_memory"),
    }.items():
        value: int | None = None
        for name in names:
            try:
                obj: Any = mx
                for part in name.split("."):
                    obj = getattr(obj, part)
                value = int(obj())
                break
            except Exception:
                continue
        out[key] = value
    return out


class LoadPhaseMemoryProbe:
    """Small typed recorder for load-phase RSS + MLX allocator telemetry."""

    def __init__(self, *, emit_log_lines: bool = False) -> None:
        self.samples: list[dict[str, Any]] = []
        self.emit_log_lines = emit_log_lines
        self._start_rss_bytes: int | None = None
        self._start_available_bytes: int | None = None
        self._software_budget_bytes: int | None = None
        self._software_budget_required = False
        self._budget_check_count = 0
        self._budget_peak_bytes = 0
        self._last_budget_check: dict[str, Any] | None = None

    def _emit_sample_line(self, sample: Mapping[str, Any]) -> None:
        if not self.emit_log_lines:
            return
        line = {
            "schema": "ddm_mx1_load_phase_checkpoint.v1",
            "event_index": sample.get("event_index"),
            "stage": sample.get("stage"),
            "timestamp_utc": sample.get("timestamp_utc"),
            "rss_gib": sample.get("rss_gib"),
            "rss_delta_from_start_gib": sample.get("rss_delta_from_start_gib"),
            "sys_available_gib": sample.get("sys_available_gib"),
            "sys_available_delta_from_start_gib": sample.get("sys_available_delta_from_start_gib"),
            "mlx_active_gib": sample.get("mlx_active_gib"),
            "mlx_cache_gib": sample.get("mlx_cache_gib"),
            "mlx_peak_gib": sample.get("mlx_peak_gib"),
            "note": sample.get("note"),
        }
        print(f"[mx1-load-phase] {json.dumps(line, sort_keys=True)}", file=sys.stderr, flush=True)

    def sample(self, stage: str, *, mx: Any | None = None, note: str | None = None) -> dict[str, Any]:
        rss_bytes = _process_rss_bytes()
        available_bytes = _system_available_bytes()
        if self._start_rss_bytes is None:
            self._start_rss_bytes = rss_bytes
        if self._start_available_bytes is None:
            self._start_available_bytes = available_bytes
        mlx_bytes = _mlx_allocator_bytes(mx)
        sample = {
            "event_index": len(self.samples),
            "stage": stage,
            "timestamp_utc": _utc_now_iso(),
            "rss_gib": _gib_or_none(rss_bytes),
            "rss_delta_from_start_gib": _gib_or_none(
                None if rss_bytes is None or self._start_rss_bytes is None else rss_bytes - self._start_rss_bytes
            ),
            "sys_available_gib": _gib_or_none(available_bytes),
            "sys_available_delta_from_start_gib": _gib_or_none(
                None
                if available_bytes is None or self._start_available_bytes is None
                else available_bytes - self._start_available_bytes
            ),
            "mlx_active_gib": _gib_or_none(mlx_bytes["active"]),
            "mlx_cache_gib": _gib_or_none(mlx_bytes["cache"]),
            "mlx_peak_gib": _gib_or_none(mlx_bytes["peak"]),
        }
        if note:
            sample["note"] = note
        self.samples.append(sample)
        self._emit_sample_line(sample)
        return sample

    def install_software_budget(self, memory_limits: Mapping[str, Any]) -> None:
        budget = memory_limits.get("software_budget_bytes")
        self._software_budget_bytes = None if budget is None else int(budget)
        self._software_budget_required = bool(memory_limits.get("software_cap_required"))
        if self._software_budget_required and self._software_budget_bytes is None:
            raise MemoryLimitConfigurationError("REFUSED gpu mode: software memory budget was not installed")

    def sample_and_check(
        self,
        stage: str,
        *,
        mx: Any | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        sample = self.sample(stage, mx=mx, note=note)
        self.check_budget(stage, mx=mx)
        return sample

    def check_budget(self, stage: str, *, mx: Any | None = None) -> dict[str, Any] | None:
        if self._software_budget_bytes is None:
            if self._software_budget_required:
                raise MemoryLimitConfigurationError(
                    "REFUSED gpu mode: software memory budget check requested before installation"
                )
            return None
        rss_bytes = _process_rss_bytes()
        if self._start_rss_bytes is None:
            self._start_rss_bytes = rss_bytes
        if rss_bytes is None or self._start_rss_bytes is None:
            raise MemoryLimitConfigurationError("REFUSED gpu mode: software budget cannot read process RSS")
        rss_delta = max(0, int(rss_bytes) - int(self._start_rss_bytes))
        mlx_active: int
        if mx is None:
            mlx_active = 0
        else:
            mlx_active_raw = _mlx_allocator_bytes(mx).get("active")
            if mlx_active_raw is None:
                raise MemoryLimitConfigurationError(
                    "REFUSED gpu mode: software budget cannot read mx.get_active_memory()"
                )
            mlx_active = int(mlx_active_raw)
        combined = int(mlx_active) + int(rss_delta)
        self._budget_check_count += 1
        self._budget_peak_bytes = max(self._budget_peak_bytes, combined)
        check = {
            "stage": stage,
            "timestamp_utc": _utc_now_iso(),
            "budget_bytes": self._software_budget_bytes,
            "budget_gib": _gib_or_none(self._software_budget_bytes),
            "mlx_active_gib": _gib_or_none(mlx_active),
            "rss_delta_from_start_gib": _gib_or_none(rss_delta),
            "combined_gib": _gib_or_none(combined),
            "within_budget": combined <= self._software_budget_bytes,
            "check_index": self._budget_check_count,
        }
        self._last_budget_check = check
        if combined > self._software_budget_bytes:
            self.sample(stage, mx=mx, note="software memory budget exceeded")
            raise MemoryBudgetExceeded(
                "software memory budget exceeded: "
                f"stage={stage} combined={combined} budget={self._software_budget_bytes}",
                check=check,
            )
        return check

    def budget_summary(self) -> dict[str, Any]:
        return {
            "enforcement": "software_stage_step_cap",
            "budget_bytes": self._software_budget_bytes,
            "budget_gib": _gib_or_none(self._software_budget_bytes),
            "required": self._software_budget_required,
            "check_count": self._budget_check_count,
            "peak_combined_gib": _gib_or_none(self._budget_peak_bytes),
            "last_check": self._last_budget_check,
            "rule": "mx.get_active_memory() + max(0, process_rss - start_process_rss) <= budget",
        }

    def peak(self) -> dict[str, Any]:
        def max_present(key: str) -> float | None:
            values = [row[key] for row in self.samples if row.get(key) is not None]
            return max(values) if values else None

        def min_present(key: str) -> float | None:
            values = [row[key] for row in self.samples if row.get(key) is not None]
            return min(values) if values else None

        return {
            "sample_count": len(self.samples),
            "peak_rss_gib": max_present("rss_gib"),
            "min_sys_available_gib": min_present("sys_available_gib"),
            "peak_mlx_active_gib": max_present("mlx_active_gib"),
            "peak_mlx_cache_gib": max_present("mlx_cache_gib"),
            "peak_mlx_reported_gib": max_present("mlx_peak_gib"),
        }


def _load_lifted_semantic() -> Any:
    if str(LIFTED) not in sys.path:
        sys.path.insert(0, str(LIFTED))
    spec = importlib.util.spec_from_file_location(
        "mx1_lifted_semantic_renderer_oracle",
        LIFTED / "semantic_renderer_oracle.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load lifted semantic_renderer_oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastTokenSegProxy(torch.nn.Module):
    """Fast differentiable proxy used only when upstream SegNet is not requested.

    The proxy consumes RGB and emits five logits.  It is deliberately labeled
    proxy and exists for checkpoint/resume smoke only; scorer-in-loop smoke uses
    upstream SegNet via ``--scorer upstream``.
    """

    def forward(self, rgb_nchw: torch.Tensor) -> torch.Tensor:
        x = rgb_nchw.float() / 255.0
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack(
            [
                2.0 * r - g - b,
                2.0 * g - r - b,
                2.0 * b - r - g,
                r + g - b,
                b + g - r,
            ],
            dim=1,
        )


def _load_upstream_segnet(device: torch.device) -> torch.nn.Module:
    root = REPO / "upstream"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import modules  # type: ignore[import-not-found]
    from safetensors.torch import load_file

    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)
    return segnet


def _select_stratified_indices(n: int, total: int = 600, seed: int = 20260806) -> list[int]:
    rng = np.random.default_rng(seed)
    buckets = np.array_split(np.arange(total), n)
    selected = [int(rng.choice(bucket)) for bucket in buckets if len(bucket)]
    return sorted(selected)


def _load_selected_seg_tokens(cache_path: Path, pair_ids: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load only selected cache rows into the retained tensor.

    The cache file is still a monolithic ``torch.save`` payload, so PyTorch must
    deserialize it. The fix is to index+clone immediately and drop the full
    cache before any MLX arrays or scorer weights are built.
    """

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    try:
        seg_all = payload["seg"]
        idx = torch.tensor(pair_ids, dtype=torch.long)
        selected = seg_all.index_select(0, idx).long().clone().contiguous()
        meta = {
            "cache_path": str(cache_path),
            "cache_bytes": cache_path.stat().st_size if cache_path.exists() else None,
            "full_shape_seen": list(seg_all.shape),
            "full_dtype_seen": str(seg_all.dtype),
            "selected_shape": list(selected.shape),
            "selected_dtype": str(selected.dtype),
            "selected_pair_count": len(pair_ids),
        }
    finally:
        del payload
        if "seg_all" in locals():
            del seg_all
        gc.collect()
    return selected, meta


def _load_selected_seg_tokens_with_residency(
    cache_path: Path,
    pair_ids: list[int],
    *,
    retain_full_cache: bool,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None]:
    """Load selected rows and optionally keep the full real cache resident."""

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    retained: torch.Tensor | None = None
    try:
        seg_all = payload["seg"]
        idx = torch.tensor(pair_ids, dtype=torch.long)
        selected = seg_all.index_select(0, idx).long().clone().contiguous()
        if retain_full_cache:
            retained = seg_all
        meta = {
            "cache_path": str(cache_path),
            "cache_bytes": cache_path.stat().st_size if cache_path.exists() else None,
            "full_shape_seen": list(seg_all.shape),
            "full_dtype_seen": str(seg_all.dtype),
            "selected_shape": list(selected.shape),
            "selected_dtype": str(selected.dtype),
            "selected_pair_count": len(pair_ids),
            "full_cache_resident": retain_full_cache,
            "retained_tensor_shape": list(seg_all.shape) if retain_full_cache else None,
            "retained_tensor_dtype": str(seg_all.dtype) if retain_full_cache else None,
        }
    finally:
        del payload
        if not retain_full_cache and "seg_all" in locals():
            del seg_all
        gc.collect()
    return selected, meta, retained


def _load_selected_token_arrays(
    *,
    input_cache: Path,
    target_cache: Path,
    pair_ids: list[int],
    memory_probe: LoadPhaseMemoryProbe | None,
    cache_residency: str = "selected",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[torch.Tensor]]:
    if cache_residency not in CACHE_RESIDENCY_MODES:
        raise ValueError(f"unknown cache_residency {cache_residency!r}")
    retain_full_cache = cache_residency == "ram-full"
    retained: list[torch.Tensor] = []
    if memory_probe is not None:
        memory_probe.sample_and_check("before_selected_cache_load")
    input_tokens, input_meta, input_retained = _load_selected_seg_tokens_with_residency(
        input_cache,
        pair_ids,
        retain_full_cache=retain_full_cache,
    )
    if input_retained is not None:
        retained.append(input_retained)
    if memory_probe is not None:
        memory_probe.sample_and_check("after_input_cache_selected_clone")
    if input_cache == target_cache:
        target_tokens = input_tokens
        target_meta = {**input_meta, "shared_with_input_cache": True}
    else:
        target_tokens, target_meta, target_retained = _load_selected_seg_tokens_with_residency(
            target_cache,
            pair_ids,
            retain_full_cache=retain_full_cache,
        )
        if target_retained is not None:
            retained.append(target_retained)
        if memory_probe is not None:
            memory_probe.sample_and_check("after_target_cache_selected_clone")

    conditioning_np = input_tokens.numpy().astype(np.int32, copy=True)
    target_np = target_tokens.numpy().astype(np.int32, copy=True)
    del input_tokens
    del target_tokens
    gc.collect()
    if memory_probe is not None:
        memory_probe.sample_and_check("after_selected_cache_numpy_copy_and_torch_free")
    return (
        conditioning_np,
        target_np,
        {
            "input": input_meta,
            "target": target_meta,
            "subset_before_materialize": "torch.load_index_clone_del_full_cache",
            "cache_residency": cache_residency,
            "full_cache_resident_tensor_count": len(retained),
        },
        retained,
    )


def run_torch_smoke(args: argparse.Namespace) -> dict[str, Any]:
    lifted = _load_lifted_semantic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    conditioning_np, target_np, cache_meta, _cache_handles = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=None,
    )
    idx = torch.tensor(pair_ids, dtype=torch.long, device=device)
    conditioning = torch.from_numpy(conditioning_np).long().to(device)
    target = torch.from_numpy(target_np).long().to(device)
    del conditioning_np, target_np
    gc.collect()

    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    config = architecture_config_from_checkpoint(
        checkpoint, consumer="ddm_mx1.run_torch_smoke"
    )
    model = lifted.SemanticTokenRenderer(
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        frame_dim=int(config["frame_dim"]),
        num_pairs=600,
        phase_y=int(config.get("phase_y", 1)),
        phase_x=int(config.get("phase_x", 1)),
        temporal_radius=int(config.get("temporal_radius", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    segnet: torch.nn.Module
    scorer_axis: str
    if args.scorer == "upstream":
        segnet = _load_upstream_segnet(device)
        scorer_axis = "[macOS-CPU advisory torch upstream SegNet]"
    else:
        segnet = FastTokenSegProxy().eval().to(device)
        scorer_axis = "[proxy smoke no scorer authority]"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.steps, 1), eta_min=args.lr * 0.01)
    history: list[dict[str, Any]] = []
    start_time = time.time()
    for step in range(args.steps):
        model.train()
        frame = lifted.render_for_seg(model, conditioning, idx, exact_path=args.train_exact_path)
        logits = segnet(frame)
        loss, phase = lifted.curriculum_loss(
            logits,
            target,
            step=step,
            total_steps=args.steps,
            ce_fraction=args.ce_fraction,
            softplus_fraction=args.softplus_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            dseg = float((pred != target).float().mean())
        history.append(
            {
                "step": step + 1,
                "phase": phase,
                "loss": float(loss.detach()),
                "d_seg_batch": dseg,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
    elapsed = time.time() - start_time
    stage_path = args.run_dir / f"torch_smoke_stage_steps{args.steps:04d}.pt"
    latest_path = args.run_dir / "torch_smoke.latest.pt"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "pair_ids": pair_ids,
        "seed": args.seed,
        "scorer_axis": scorer_axis,
        "score_claim": False,
    }
    torch.save(payload, stage_path)
    torch.save(payload, latest_path)
    resume = torch.load(latest_path, map_location="cpu", weights_only=False)
    resume_ok = sorted(resume.keys()) == sorted(payload.keys()) and resume["pair_ids"] == pair_ids
    return {
        "schema": "ddm_mx1_torch_smoke.v1",
        "status": "passed" if resume_ok else "blocked",
        "scorer_axis": scorer_axis,
        "score_claim": False,
        "pairs": pair_ids,
        "cache_load": cache_meta,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(args.steps, 1),
        "history": history,
        "checkpoint_stage": str(stage_path),
        "checkpoint_latest": str(latest_path),
        "checkpoint_latest_bytes": latest_path.stat().st_size,
        "checkpoint_latest_sha256": _sha256_file(latest_path),
        "resume_load_ok": resume_ok,
    }


def _new_torch_renderer_from_config(lifted: Any, config: Mapping[str, Any]) -> torch.nn.Module:
    return lifted.SemanticTokenRenderer(
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        frame_dim=int(config["frame_dim"]),
        num_pairs=int(config.get("num_pairs", 600)),
        num_tokens=int(config.get("num_tokens", 5)),
        phase_y=int(config.get("phase_y", 1)),
        phase_x=int(config.get("phase_x", 1)),
        temporal_radius=int(config.get("temporal_radius", 0)),
    ).eval()


def _build_torch_renderer(lifted: Any, checkpoint: Mapping[str, Any]) -> torch.nn.Module:
    model = _new_torch_renderer_from_config(
        lifted,
        architecture_config_from_checkpoint(
            checkpoint, consumer="ddm_mx1._build_torch_renderer"
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _json_from_uint8_array(value: np.ndarray) -> Any:
    return json.loads(bytes(value).decode("utf-8"))


def _torch_tensor_from_mlx_param(
    name: str,
    value: np.ndarray,
    *,
    expected: torch.Tensor,
) -> torch.Tensor:
    arr = np.asarray(value)
    if arr.ndim == 4 and expected.ndim == 4:
        arr = np.transpose(arr, (0, 3, 1, 2))
    if tuple(arr.shape) != tuple(expected.shape):
        raise ValueError(
            "MLX checkpoint tensor shape mismatch for "
            f"{name}: mapped {tuple(arr.shape)} != torch {tuple(expected.shape)}"
        )
    if expected.dtype.is_floating_point:
        arr = arr.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=expected.dtype)
    return tensor.clone()


def _load_mlx_npz_checkpoint_for_torch(
    path: Path,
    *,
    lifted: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        files = set(payload.files)
        required_meta = {"meta::config_json", "meta::step", "meta::history_json"}
        missing_meta = sorted(required_meta - files)
        if missing_meta:
            raise ValueError(f"MLX checkpoint missing metadata fields: {missing_meta}")
        config = _json_from_uint8_array(payload["meta::config_json"])
        history = _json_from_uint8_array(payload["meta::history_json"])
        extra = _json_from_uint8_array(payload["meta::extra_json"]) if "meta::extra_json" in files else {}
        step = int(np.asarray(payload["meta::step"]).reshape(-1)[0])
        expected_model = _new_torch_renderer_from_config(lifted, config)
        expected_state = expected_model.state_dict()
        present_names = {key.removeprefix("param::") for key in files if key.startswith("param::")}
        expected_names = set(expected_state)
        missing = sorted(expected_names - present_names)
        unexpected = sorted(present_names - expected_names)
        if missing or unexpected:
            raise ValueError(f"MLX checkpoint parameter set mismatch; missing={missing}; unexpected={unexpected}")
        state_dict = {
            name: _torch_tensor_from_mlx_param(
                name,
                payload[f"param::{name}"],
                expected=expected_state[name],
            )
            for name in sorted(expected_names)
        }
    checkpoint = {"config": config, "state_dict": state_dict}
    meta = {
        "format": "mlx_npz",
        "step": step,
        "history": history,
        "extra": extra,
        "param_count": len(state_dict),
    }
    return checkpoint, meta


def _load_torch_or_mlx_checkpoint_for_verdict(
    path: Path,
    *,
    lifted: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.suffix == ".npz":
        return _load_mlx_npz_checkpoint_for_torch(path, lifted=lifted)
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if not (magic.startswith(b"PK\x03\x04") or magic[:1] == b"\x80"):
        raise ValueError(
            f"verdict checkpoint {path} is not a PyTorch pickle/zip "
            f"(magic {magic!r}); refusing torch.load"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("torch verdict checkpoint must contain config and state_dict")
    history = list(checkpoint.get("history", []))
    step = int(checkpoint.get("step") or (history[-1]["step"] if history else 0))
    meta = {
        "format": "torch_pt",
        "step": step,
        "history": history,
        "extra": {
            "pair_ids": checkpoint.get("pair_ids"),
            "score_claim": checkpoint.get("score_claim", False),
            "axis": checkpoint.get("scorer_axis"),
        },
        "param_count": len(checkpoint["state_dict"]),
    }
    return checkpoint, meta


def _checkpoint_pair_ids(meta: Mapping[str, Any]) -> list[int]:
    extra = meta.get("extra")
    pair_ids = extra.get("pair_ids") if isinstance(extra, Mapping) else None
    if pair_ids is None:
        raise ValueError("verdict checkpoint missing pair_ids; refusing to re-derive from seed")
    out = [int(item) for item in pair_ids]
    if not out:
        raise ValueError("verdict checkpoint pair_ids is empty")
    return out


def _history_row_at_step(history: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for row in reversed(history):
        if int(row.get("step", -1)) == int(step):
            if "d_seg_batch" not in row:
                raise ValueError(f"checkpoint history row at step {step} lacks d_seg_batch")
            return dict(row)
    raise ValueError(f"checkpoint history has no row for step {step}")


def _iter_tensor_slices(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def _as_float(value: Any) -> float:
    return float(np.asarray(value).reshape(()))


def _margin_histogram_empty() -> dict[str, Any]:
    return {
        "total": 0,
        "bins": [{"range": label, "lo": lo, "hi": hi, "count": 0} for label, lo, hi in MARGIN_BINS],
    }


def _margin_histogram_update(hist: dict[str, Any], margins: torch.Tensor, mask: torch.Tensor) -> None:
    selected = margins[mask]
    hist["total"] += int(selected.numel())
    if selected.numel() == 0:
        return
    for bucket, (_label, lo, hi) in zip(hist["bins"], MARGIN_BINS, strict=True):
        if hi is None:
            count = int((selected >= lo).sum().item())
        else:
            count = int(((selected >= lo) & (selected < hi)).sum().item())
        bucket["count"] += count


def _margin_histogram_finalize(hist: dict[str, Any]) -> dict[str, Any]:
    total = int(hist["total"])
    return {
        "total": total,
        "bins": [
            {
                "range": bucket["range"],
                "lo": bucket["lo"],
                "hi": bucket["hi"],
                "count": int(bucket["count"]),
                "fraction": None if total == 0 else int(bucket["count"]) / total,
            }
            for bucket in hist["bins"]
        ],
    }


def _boundary_band_mask(labels: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :, 1:] |= labels[:, :, 1:] != labels[:, :, :-1]
    mask[:, :, :-1] |= labels[:, :, :-1] != labels[:, :, 1:]
    mask[:, 1:, :] |= labels[:, 1:, :] != labels[:, :-1, :]
    mask[:, :-1, :] |= labels[:, :-1, :] != labels[:, 1:, :]
    return mask


def _new_class_accumulators() -> dict[str, np.ndarray]:
    return {
        "gt_sites": np.zeros(len(SEG_CLASS_NAMES), dtype=np.int64),
        "gt_mispredicted": np.zeros(len(SEG_CLASS_NAMES), dtype=np.int64),
        "pred_sites": np.zeros(len(SEG_CLASS_NAMES), dtype=np.int64),
        "pred_false_positive": np.zeros(len(SEG_CLASS_NAMES), dtype=np.int64),
        "confusion": np.zeros((len(SEG_CLASS_NAMES), len(SEG_CLASS_NAMES)), dtype=np.int64),
    }


def _update_class_accumulators(
    accum: dict[str, np.ndarray],
    target: torch.Tensor,
    pred: torch.Tensor,
) -> None:
    target_cpu = target.detach().cpu().long()
    pred_cpu = pred.detach().cpu().long()
    mismatch_cpu = pred_cpu != target_cpu
    for class_id in range(len(SEG_CLASS_NAMES)):
        gt_mask = target_cpu == class_id
        pred_mask = pred_cpu == class_id
        accum["gt_sites"][class_id] += int(gt_mask.sum().item())
        accum["gt_mispredicted"][class_id] += int((gt_mask & mismatch_cpu).sum().item())
        accum["pred_sites"][class_id] += int(pred_mask.sum().item())
        accum["pred_false_positive"][class_id] += int((pred_mask & mismatch_cpu).sum().item())
    flat = (target_cpu.reshape(-1) * len(SEG_CLASS_NAMES) + pred_cpu.reshape(-1)).clamp(
        min=0,
        max=len(SEG_CLASS_NAMES) * len(SEG_CLASS_NAMES) - 1,
    )
    counts = torch.bincount(flat, minlength=len(SEG_CLASS_NAMES) * len(SEG_CLASS_NAMES))
    accum["confusion"] += counts.reshape(len(SEG_CLASS_NAMES), len(SEG_CLASS_NAMES)).numpy()


def _finalize_class_accumulators(accum: dict[str, np.ndarray]) -> dict[str, Any]:
    per_class = []
    for class_id, class_name in enumerate(SEG_CLASS_NAMES):
        gt_sites = int(accum["gt_sites"][class_id])
        gt_mispredicted = int(accum["gt_mispredicted"][class_id])
        pred_sites = int(accum["pred_sites"][class_id])
        pred_false_positive = int(accum["pred_false_positive"][class_id])
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_sites": gt_sites,
                "gt_mispredicted": gt_mispredicted,
                "gt_mispredicted_rate": None if gt_sites == 0 else gt_mispredicted / gt_sites,
                "pred_sites": pred_sites,
                "pred_false_positive": pred_false_positive,
                "pred_false_positive_rate": None if pred_sites == 0 else pred_false_positive / pred_sites,
            }
        )
    directed = []
    for gt_id, gt_name in enumerate(SEG_CLASS_NAMES):
        for pred_id, pred_name in enumerate(SEG_CLASS_NAMES):
            directed.append(
                {
                    "gt_class_id": gt_id,
                    "gt_class_name": gt_name,
                    "pred_class_id": pred_id,
                    "pred_class_name": pred_name,
                    "pixels": int(accum["confusion"][gt_id, pred_id]),
                    "is_correct": gt_id == pred_id,
                }
            )
    return {
        "class_order": [
            {"class_id": class_id, "class_name": class_name} for class_id, class_name in enumerate(SEG_CLASS_NAMES)
        ],
        "class_order_provenance": "CLAUDE.md canonical comma10k order; no luma-sort",
        "per_class_d_seg": per_class,
        "directed_confusion_counts": directed,
    }


def _stage_step_from_path(path: Path) -> int:
    stem = path.stem
    prefix = "mlx_stage_step"
    if not stem.startswith(prefix):
        raise ValueError(f"not an MX1 stage checkpoint name: {path}")
    return int(stem.removeprefix(prefix))


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _mx1t_copy_checkpoint(src: Path, copy_dir: Path) -> dict[str, Any]:
    copy_dir.mkdir(parents=True, exist_ok=True)
    src_sha = _sha256_file(src)
    dst = copy_dir / src.name
    status = "copied"
    if dst.exists() and _sha256_file(dst) == src_sha:
        status = "reused_existing_copy"
    else:
        tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    dst_sha = _sha256_file(dst)
    if dst_sha != src_sha:
        raise ValueError(f"checkpoint copy sha mismatch for {src}: {src_sha} != {dst_sha}")
    return {
        "schema": "ddm_mx1t_checkpoint_copy.v1",
        "status": status,
        "copied_at_utc": _utc_now_iso(),
        "step": _stage_step_from_path(src),
        "source_path": str(src),
        "source_bytes": src.stat().st_size,
        "source_sha256": src_sha,
        "copy_path": str(dst),
        "copy_bytes": dst.stat().st_size,
        "copy_sha256": dst_sha,
    }


def _mx1t_evaluate_checkpoint_facets(
    *,
    lifted: Any,
    checkpoint: Mapping[str, Any],
    checkpoint_meta: Mapping[str, Any],
    checkpoint_info: Mapping[str, Any],
    conditioning: torch.Tensor,
    target: torch.Tensor,
    idx: torch.Tensor,
    pair_ids: list[int],
    segnet: torch.nn.Module,
    batch_size: int,
    previous_mismatch_set: np.ndarray | None,
    row_kind: str,
    tail_average: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    model = _build_torch_renderer(lifted, checkpoint).to(torch.device("cpu"))
    per_pair: list[dict[str, Any]] = []
    total_mismatch = 0
    total_pixels = 0
    scorer_batch_shapes: list[list[int]] = []
    chunk_batch_sizes: list[int] = []
    mismatch_parts: list[np.ndarray] = []
    class_accum = _new_class_accumulators()
    mismatch_hist = _margin_histogram_empty()
    correct_boundary_hist = _margin_histogram_empty()
    start_time = time.time()
    with torch.no_grad():
        for start, stop in _iter_tensor_slices(len(pair_ids), batch_size):
            cond_chunk = conditioning[start:stop]
            target_chunk = target[start:stop]
            idx_chunk = idx[start:stop]
            frame_r = lifted.render_for_seg(
                model,
                cond_chunk,
                idx_chunk,
                exact_path=True,
            )
            logits = segnet(frame_r)
            top2 = torch.topk(logits, k=2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]
            pred = logits.argmax(dim=1)
            mismatch = pred != target_chunk
            boundary = _boundary_band_mask(target_chunk)
            _margin_histogram_update(mismatch_hist, margins, mismatch)
            _margin_histogram_update(correct_boundary_hist, margins, (~mismatch) & boundary)
            _update_class_accumulators(class_accum, target_chunk, pred)
            flat = mismatch.reshape(mismatch.shape[0], -1)
            chunk_pixels = int(flat.shape[1])
            chunk_counts = flat.sum(dim=1).cpu().numpy().astype(np.int64)
            scorer_batch_shapes.append(list(frame_r.shape))
            chunk_batch_sizes.append(int(stop - start))
            mismatch_parts.append(mismatch.detach().cpu().numpy().astype(np.bool_, copy=False).reshape(-1))
            for pair_id, count in zip(pair_ids[start:stop], chunk_counts, strict=True):
                mismatches = int(count)
                per_pair.append(
                    {
                        "pair_id": int(pair_id),
                        "mismatch_pixels": mismatches,
                        "pixels": chunk_pixels,
                        "d_seg": mismatches / max(chunk_pixels, 1),
                    }
                )
                total_mismatch += mismatches
                total_pixels += chunk_pixels
            del frame_r, logits, top2, margins, pred, mismatch, boundary, flat
    elapsed = time.time() - start_time
    current_mismatch_set = np.concatenate(mismatch_parts) if mismatch_parts else np.zeros(0, dtype=np.bool_)
    churn: dict[str, Any]
    if previous_mismatch_set is None:
        churn = {
            "status": "not_available_first_checkpoint",
            "symmetric_difference_pixels": None,
            "denominator_current_mismatch_pixels": int(current_mismatch_set.sum()),
            "ratio_vs_current_mismatch": None,
        }
    else:
        if previous_mismatch_set.shape != current_mismatch_set.shape:
            raise ValueError(
                "mismatch-set shape changed across checkpoints: "
                f"{previous_mismatch_set.shape} vs {current_mismatch_set.shape}"
            )
        sym = int(np.logical_xor(previous_mismatch_set, current_mismatch_set).sum())
        denom = int(current_mismatch_set.sum())
        churn = {
            "status": "measured",
            "symmetric_difference_pixels": sym,
            "denominator_current_mismatch_pixels": denom,
            "ratio_vs_current_mismatch": None if denom == 0 else sym / denom,
        }
    aggregate_dseg = total_mismatch / max(total_pixels, 1)
    row: dict[str, Any] = {
        "schema": "ddm_mx1t_checkpoint_facet_row.v1",
        "status": "passed",
        "row_kind": row_kind,
        "axis": "[macOS-CPU advisory torch upstream SegNet]",
        "score_claim": False,
        "verdict_scope": f"n{len(pair_ids)} arm-instrument checkpoint-series facets",
        "host": _host_fingerprint(),
        "source_repo_head": SOURCE_REPO_HEAD,
        "source_repo_root": SOURCE_REPO_ROOT,
        "checkpoint": dict(checkpoint_info),
        "checkpoint_meta": {
            "format": checkpoint_meta.get("format"),
            "step": checkpoint_meta.get("step"),
            "param_count": checkpoint_meta.get("param_count"),
            "axis": (checkpoint_meta.get("extra") or {}).get("axis")
            if isinstance(checkpoint_meta.get("extra"), Mapping)
            else None,
            "score_claim": (checkpoint_meta.get("extra") or {}).get("score_claim")
            if isinstance(checkpoint_meta.get("extra"), Mapping)
            else None,
        },
        "pair_ids": pair_ids,
        "pair_count": len(pair_ids),
        "token_batch_shape": list(conditioning.shape),
        "target_batch_shape": list(target.shape),
        "segnet_batch_size": batch_size,
        "segnet_chunk_batch_sizes": chunk_batch_sizes,
        "scorer_batch_shapes": scorer_batch_shapes,
        "contest_faithful_roundtrip": "lifted.render_for_seg(..., exact_path=True): bilinear up to 874x1164, uint8 STE, bilinear down to 384x512",
        "per_pair_d_seg": per_pair,
        "aggregate_d_seg": aggregate_dseg,
        "total_mismatch_pixels": total_mismatch,
        "total_pixels": total_pixels,
        "margin_histogram_mismatched_pixels": _margin_histogram_finalize(mismatch_hist),
        "margin_histogram_correct_boundary_band_pixels": _margin_histogram_finalize(correct_boundary_hist),
        "class_facets": _finalize_class_accumulators(class_accum),
        "flip_set_churn_vs_previous": churn,
        "elapsed_seconds": elapsed,
    }
    if tail_average is not None:
        row["tail_average"] = dict(tail_average)
    history = list(checkpoint_meta.get("history") or [])
    step = int(checkpoint_meta.get("step") or 0)
    if history:
        row["mlx_history_row"] = _history_row_at_step(history, step)
        proxy_dseg = float(row["mlx_history_row"]["d_seg_batch"])
        row["comparison_to_mlx_proxy"] = {
            "mlx_in_training_d_seg_batch": proxy_dseg,
            "authority_minus_mlx_proxy_d_seg": aggregate_dseg - proxy_dseg,
            "authority_over_mlx_proxy_d_seg": aggregate_dseg / proxy_dseg if proxy_dseg else None,
        }
    return row, current_mismatch_set


def _load_average_torch_checkpoint(
    paths: list[Path],
    *,
    lifted: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not paths:
        raise ValueError("tail-average requires at least one checkpoint")
    loaded = [_load_mlx_npz_checkpoint_for_torch(path, lifted=lifted) for path in paths]
    configs = [json.dumps(item[0]["config"], sort_keys=True) for item in loaded]
    if len(set(configs)) != 1:
        raise ValueError("tail-average checkpoint configs differ")
    pair_sets = [_checkpoint_pair_ids(item[1]) for item in loaded]
    if any(pair_ids != pair_sets[0] for pair_ids in pair_sets):
        raise ValueError("tail-average checkpoint pair_ids differ")
    names = sorted(loaded[0][0]["state_dict"])
    state_dict: dict[str, torch.Tensor] = {}
    for name in names:
        tensors = [item[0]["state_dict"][name] for item in loaded]
        shape_set = {tuple(tensor.shape) for tensor in tensors}
        dtype_set = {tensor.dtype for tensor in tensors}
        if len(shape_set) != 1 or len(dtype_set) != 1:
            raise ValueError(f"tail-average tensor mismatch for {name}")
        if tensors[0].dtype.is_floating_point:
            state_dict[name] = torch.stack(tensors, dim=0).mean(dim=0).to(dtype=tensors[0].dtype)
        else:
            state_dict[name] = tensors[-1].clone()
    steps = [int(item[1]["step"]) for item in loaded]
    return (
        {
            "config": loaded[0][0]["config"],
            "state_dict": state_dict,
        },
        {
            "format": "torch_fp32_simple_mean_from_mlx_npz",
            "step": max(steps),
            "history": [],
            "extra": {
                "pair_ids": pair_sets[0],
                "score_claim": False,
                "axis": "[macOS-CPU advisory torch upstream SegNet]",
            },
            "param_count": len(state_dict),
            "member_steps": steps,
        },
    )


def _hist_fraction(row: Mapping[str, Any], hist_key: str, ranges: set[str]) -> float | None:
    hist = row.get(hist_key)
    if not isinstance(hist, Mapping) or not hist.get("total"):
        return None
    total = int(hist["total"])
    count = sum(int(bucket.get("count", 0)) for bucket in hist.get("bins", []) if bucket.get("range") in ranges)
    return count / total


def _top_pairs(row: Mapping[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    pairs = list(row.get("per_pair_d_seg") or [])
    pairs.sort(key=lambda item: float(item.get("d_seg", 0.0)), reverse=True)
    return pairs[:limit]


def _top_class_residuals(row: Mapping[str, Any]) -> dict[str, Any]:
    facets = (row.get("class_facets") or {}).get("per_class_d_seg") or []
    gt_sorted = sorted(facets, key=lambda item: int(item.get("gt_mispredicted", 0)), reverse=True)
    fp_sorted = sorted(facets, key=lambda item: int(item.get("pred_false_positive", 0)), reverse=True)
    return {
        "gt_mispredicted_top": gt_sorted[:3],
        "pred_false_positive_top": fp_sorted[:3],
    }


def _mx1t_iteration_verdict(
    checkpoint_rows: list[dict[str, Any]],
    tail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first = checkpoint_rows[0]
    last = checkpoint_rows[-1]
    near_ranges = {"0-0.05", "0.05-0.1"}
    first_near = _hist_fraction(first, "margin_histogram_mismatched_pixels", near_ranges)
    last_near = _hist_fraction(last, "margin_histogram_mismatched_pixels", near_ranges)
    first_far = _hist_fraction(first, "margin_histogram_mismatched_pixels", {">0.5"})
    last_far = _hist_fraction(last, "margin_histogram_mismatched_pixels", {">0.5"})
    churn_values = [
        float(row["flip_set_churn_vs_previous"]["ratio_vs_current_mismatch"])
        for row in checkpoint_rows[1:]
        if row["flip_set_churn_vs_previous"].get("ratio_vs_current_mismatch") is not None
    ]
    tail_winners = [row for row in tail_rows if row.get("tail_average", {}).get("delta_vs_final_d_seg", 0.0) < 0.0]
    if first_near is not None and last_near is not None and last_near > first_near:
        margin_verdict = "near_flip_fraction_rising"
    elif first_far is not None and last_far is not None and last_far >= first_far:
        margin_verdict = "far_margin_stuck_not_improving"
    else:
        margin_verdict = "mixed_margin_trend"
    if churn_values:
        median_churn = float(np.median(np.asarray(churn_values, dtype=np.float64)))
        if median_churn >= 1.0:
            churn_verdict = "high_churn_trading_pixels"
        elif median_churn <= 0.25:
            churn_verdict = "low_churn_stable_residual"
        else:
            churn_verdict = "moderate_churn"
    else:
        median_churn = None
        churn_verdict = "unmeasured_single_row"
    if tail_winners:
        best_tail = min(tail_winners, key=lambda row: row["aggregate_d_seg"])
        tail_verdict = "tail_average_wins_here"
        tail_delta = best_tail["tail_average"]["delta_vs_final_d_seg"]
        tail_k = best_tail["tail_average"]["k"]
    else:
        best_tail = min(tail_rows, key=lambda row: row["aggregate_d_seg"]) if tail_rows else None
        tail_verdict = "tail_average_loses_or_unavailable"
        tail_delta = None if best_tail is None else best_tail["tail_average"]["delta_vs_final_d_seg"]
        tail_k = None if best_tail is None else best_tail["tail_average"]["k"]
    if (
        last["aggregate_d_seg"] < first["aggregate_d_seg"]
        and margin_verdict == "near_flip_fraction_rising"
        and churn_verdict != "high_churn_trading_pixels"
    ):
        next_delta = "extend_steps_same_lr_schedule_before_mechanism_pivot"
        next_delta_basis = "d_seg descended, mismatch margins moved toward near-flip, and churn was not high"
    elif tail_verdict == "tail_average_wins_here":
        next_delta = "apply_tail_average_selection_symmetrically_to_arm_cap_arm_veh_and_n120"
        next_delta_basis = f"avg-K={tail_k} beat final by d_seg {tail_delta}"
    else:
        next_delta = "do_not_assume_more_steps_pay_without_new_objective_or_capacity_change"
        next_delta_basis = "series did not show the clean near-flip/low-churn continuation signature"
    return {
        "margin_verdict": margin_verdict,
        "first_mismatch_near_fraction_le_0p1": first_near,
        "last_mismatch_near_fraction_le_0p1": last_near,
        "first_mismatch_far_fraction_gt_0p5": first_far,
        "last_mismatch_far_fraction_gt_0p5": last_far,
        "churn_verdict": churn_verdict,
        "median_churn_ratio_vs_current_mismatch": median_churn,
        "tail_average_verdict": tail_verdict,
        "best_tail_average_k": tail_k,
        "best_tail_average_delta_vs_final_d_seg": tail_delta,
        "residual_classes_latest": _top_class_residuals(last),
        "residual_pairs_latest_top5": _top_pairs(last, limit=5),
        "recommended_next_config_delta": next_delta,
        "recommended_next_config_delta_basis": next_delta_basis,
    }


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _mx1t_findings_markdown(result: Mapping[str, Any]) -> str:
    checkpoint_rows = list(result["checkpoint_rows"])
    tail_rows = list(result["tail_average_rows"])
    verdict = result["iteration_verdict"]
    latest = checkpoint_rows[-1]
    anchor = result["anchor_check"]
    lines = [
        "# ddm_mx1t findings",
        "",
        "## Verdict",
        "",
        "MX1T completed the ARM-CAP n32 checkpoint-series facet analyzer and tail-average A/B.",
        "",
        "| field | value |",
        "|---|---:|",
        f"| axis | {result['axis']} |",
        f"| score_claim | {str(result['score_claim']).lower()} |",
        f"| checkpoint rows | {len(checkpoint_rows)} |",
        f"| tail-average rows | {len(tail_rows)} |",
        f"| step-1500 anchor expected | {anchor['expected_d_seg']} |",
        f"| step-1500 anchor measured | {anchor['measured_d_seg']} |",
        f"| step-1500 abs diff | {anchor['abs_diff']} |",
        f"| latest step | {latest['checkpoint']['step']} |",
        f"| latest aggregate d_seg | {latest['aggregate_d_seg']} |",
        f"| latest mismatch pixels | {latest['total_mismatch_pixels']} |",
        "",
        f"Receipts JSONL: `{result['receipts_jsonl']}`",
        "",
        "## Facet Trajectory",
        "",
        "| step | aggregate d_seg | mismatch px | near-margin mismatch <=0.1 | far-margin mismatch >0.5 | churn/current |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoint_rows:
        near = _hist_fraction(row, "margin_histogram_mismatched_pixels", {"0-0.05", "0.05-0.1"})
        far = _hist_fraction(row, "margin_histogram_mismatched_pixels", {">0.5"})
        churn = row["flip_set_churn_vs_previous"].get("ratio_vs_current_mismatch")
        lines.append(
            "| {step} | {dseg:.12f} | {mismatch} | {near} | {far} | {churn} |".format(
                step=row["checkpoint"]["step"],
                dseg=float(row["aggregate_d_seg"]),
                mismatch=row["total_mismatch_pixels"],
                near="n/a" if near is None else f"{near:.6f}",
                far="n/a" if far is None else f"{far:.6f}",
                churn="n/a" if churn is None else f"{float(churn):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Tail Average A/B",
            "",
            "| row | d_seg | delta vs final | verdict |",
            "|---|---:|---:|---|",
        ]
    )
    final_dseg = float(latest["aggregate_d_seg"])
    lines.append(f"| final step {latest['checkpoint']['step']} | {final_dseg:.12f} | 0 | baseline |")
    for row in tail_rows:
        ta = row["tail_average"]
        delta = float(ta["delta_vs_final_d_seg"])
        lines.append(
            f"| avg-K={ta['k']} | {float(row['aggregate_d_seg']):.12f} | {delta:.12f} | "
            f"{'wins' if delta < 0 else 'loses'} |"
        )
    lines.extend(
        [
            "",
            "## Iteration Verdict",
            "",
            "| question | answer | measurement basis |",
            "|---|---|---|",
            f"| near-flip vs stuck | {verdict['margin_verdict']} | mismatch <=0.1 fraction {verdict['first_mismatch_near_fraction_le_0p1']} -> {verdict['last_mismatch_near_fraction_le_0p1']}; >0.5 fraction {verdict['first_mismatch_far_fraction_gt_0p5']} -> {verdict['last_mismatch_far_fraction_gt_0p5']} |",
            f"| residual owner classes | {verdict['residual_classes_latest']} | latest checkpoint per-class GT-mispredicted and predicted false-positive counts |",
            f"| residual owner pairs | {verdict['residual_pairs_latest_top5']} | latest checkpoint per-pair d_seg vector |",
            f"| churn regime | {verdict['churn_verdict']} | median churn/current {verdict['median_churn_ratio_vs_current_mismatch']} |",
            f"| tail-average verdict | {verdict['tail_average_verdict']} | best K {verdict['best_tail_average_k']}, delta {verdict['best_tail_average_delta_vs_final_d_seg']} |",
            f"| recommended next-config delta | {verdict['recommended_next_config_delta']} | {verdict['recommended_next_config_delta_basis']} |",
            "",
            "## RECALL EVIDENCE",
            "",
            "| scope | query / source | found beyond charter seeds | changed plan |",
            "|---|---|---|---|",
        ]
    )
    for row in result["recall_evidence"]:
        lines.append(f"| {row['scope']} | `{row['query']}` | {row['found']} | {row['changed_plan']} |")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Axis: [macOS-CPU advisory torch upstream SegNet].",
            "- Scope: n32 ARM-CAP checkpoint-series instrument only.",
            "- No Metal, MLX training, n600 scorer job, archive build, remote dispatch, or `upstream/evaluate.py` run.",
            "- Live run directory was copied from before reading and otherwise kept read-only.",
            "- Score claim is false; this is not a contest-CPU or contest-CUDA row.",
            "",
            "Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.",
            "",
        ]
    )
    return "\n".join(lines)


def run_torch_facets(args: argparse.Namespace) -> dict[str, Any]:
    lifted = _load_lifted_semantic()
    device = torch.device("cpu")
    batch_size = int(args.verdict_batch_size)
    checkpoint_dir = args.facet_checkpoint_dir
    out_dir = args.facet_out_dir
    copy_dir = out_dir / "checkpoint_copies"
    all_stage_paths = sorted(
        checkpoint_dir.glob("mlx_stage_step*.npz"),
        key=_stage_step_from_path,
    )
    if args.facet_steps:
        wanted = set(_parse_int_list(args.facet_steps))
        all_stage_paths = [path for path in all_stage_paths if _stage_step_from_path(path) in wanted]
    if not all_stage_paths:
        raise ValueError(f"no checkpoints found under {checkpoint_dir}")
    copy_receipts = [_mx1t_copy_checkpoint(path, copy_dir) for path in all_stage_paths]
    copied_paths = [Path(row["copy_path"]) for row in copy_receipts]
    first_checkpoint, first_meta = _load_torch_or_mlx_checkpoint_for_verdict(
        copied_paths[0],
        lifted=lifted,
    )
    pair_ids = _checkpoint_pair_ids(first_meta)
    del first_checkpoint
    conditioning_np, target_np, cache_meta, _cache_handles = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=None,
    )
    idx = torch.tensor(pair_ids, dtype=torch.long, device=device)
    conditioning = torch.from_numpy(conditioning_np).long().to(device)
    target = torch.from_numpy(target_np).long().to(device)
    del conditioning_np, target_np
    gc.collect()
    segnet = _load_upstream_segnet(device)
    # rr16-F1/rr18-F1 cache-bound provenance: every facet row must be self-contained
    # (cache identity + SHA-256 + pair IDs + replay argv), not just the aggregate block.
    input_cache_path = Path(args.input_cache)
    target_cache_path = Path(args.target_cache)
    caches_shared = input_cache_path.resolve() == target_cache_path.resolve()
    input_cache_sha = _sha256_file(input_cache_path)
    cache_provenance: dict[str, Any] = {
        "input_cache": {
            "path": str(input_cache_path),
            "bytes": input_cache_path.stat().st_size,
            "sha256": input_cache_sha,
        },
        "target_cache": (
            {"shared_with_input_cache": True, "sha256": input_cache_sha}
            if caches_shared
            else {
                "path": str(target_cache_path),
                "bytes": target_cache_path.stat().st_size,
                "sha256": _sha256_file(target_cache_path),
            }
        ),
        "pair_ids": [int(p) for p in pair_ids],
        "source_repo_head": SOURCE_REPO_HEAD,
        "replay_argv": list(sys.argv),
        "axis": "[macOS-CPU advisory torch upstream SegNet]",
        "score_claim": False,
    }
    checkpoint_rows: list[dict[str, Any]] = []
    previous_mismatch_set: np.ndarray | None = None
    anchor_row: dict[str, Any] | None = None
    for copy_receipt, copied_path in zip(copy_receipts, copied_paths, strict=True):
        checkpoint, checkpoint_meta = _load_torch_or_mlx_checkpoint_for_verdict(
            copied_path,
            lifted=lifted,
        )
        if _checkpoint_pair_ids(checkpoint_meta) != pair_ids:
            raise ValueError(f"pair_ids changed in {copied_path}")
        row, previous_mismatch_set = _mx1t_evaluate_checkpoint_facets(
            lifted=lifted,
            checkpoint=checkpoint,
            checkpoint_meta=checkpoint_meta,
            checkpoint_info=copy_receipt,
            conditioning=conditioning,
            target=target,
            idx=idx,
            pair_ids=pair_ids,
            segnet=segnet,
            batch_size=batch_size,
            previous_mismatch_set=previous_mismatch_set,
            row_kind="checkpoint",
        )
        checkpoint_rows.append(row)
        if int(copy_receipt["step"]) == int(args.facet_anchor_step):
            anchor_row = row
            diff = abs(float(row["aggregate_d_seg"]) - float(args.facet_anchor_d_seg))
            if diff > float(args.facet_anchor_tolerance):
                blocker = {
                    "schema": "ddm_mx1t_anchor_blocker.v1",
                    "status": "blocked",
                    "axis": "[macOS-CPU advisory torch upstream SegNet]",
                    "score_claim": False,
                    "expected_d_seg": float(args.facet_anchor_d_seg),
                    "measured_d_seg": float(row["aggregate_d_seg"]),
                    "abs_diff": diff,
                    "tolerance": float(args.facet_anchor_tolerance),
                    "checkpoint": row["checkpoint"],
                    "reason": "step-1500 CPU-torch anchor did not reproduce mx1h; stopped before later checkpoints/tail averages",
                }
                write_json_atomic(out_dir / "MX1T_ANCHOR_BLOCKER.json", blocker)
                raise RuntimeError(blocker["reason"])
    if anchor_row is None:
        raise ValueError(f"required anchor step {args.facet_anchor_step} was not present")
    latest_row = checkpoint_rows[-1]
    tail_average_rows: list[dict[str, Any]] = []
    tail_ks = _parse_int_list(args.facet_tail_average_ks)
    for k in tail_ks:
        if k <= 0:
            raise ValueError("tail average K values must be positive")
        if len(copied_paths) < k:
            continue
        member_paths = copied_paths[-k:]
        avg_checkpoint, avg_meta = _load_average_torch_checkpoint(member_paths, lifted=lifted)
        member_receipts = [copy_receipts[copied_paths.index(path)] for path in member_paths]
        avg_info = {
            "schema": "ddm_mx1t_tail_average_checkpoint.v1",
            "step": int(avg_meta["step"]),
            "source": "in_memory_simple_mean",
            "k": k,
            "member_steps": [int(row["step"]) for row in member_receipts],
            "member_copy_paths": [row["copy_path"] for row in member_receipts],
            "member_copy_sha256": [row["copy_sha256"] for row in member_receipts],
            "averaging_rule": "torch.stack(member_tensors).mean(dim=0) for floating tensors; non-floating metadata copied from latest member",
        }
        row, _avg_mismatch_set = _mx1t_evaluate_checkpoint_facets(
            lifted=lifted,
            checkpoint=avg_checkpoint,
            checkpoint_meta=avg_meta,
            checkpoint_info=avg_info,
            conditioning=conditioning,
            target=target,
            idx=idx,
            pair_ids=pair_ids,
            segnet=segnet,
            batch_size=batch_size,
            previous_mismatch_set=None,
            row_kind="tail_average",
            tail_average={
                "k": k,
                "member_steps": avg_info["member_steps"],
                "final_reference_step": latest_row["checkpoint"]["step"],
                "final_reference_d_seg": latest_row["aggregate_d_seg"],
                "delta_vs_final_d_seg": None,
            },
        )
        row["tail_average"]["delta_vs_final_d_seg"] = row["aggregate_d_seg"] - latest_row["aggregate_d_seg"]
        row["tail_average"]["wins_vs_final"] = row["tail_average"]["delta_vs_final_d_seg"] < 0.0
        tail_average_rows.append(row)
    iteration_verdict = _mx1t_iteration_verdict(checkpoint_rows, tail_average_rows)
    for provenance_row in (*checkpoint_rows, *tail_average_rows):
        provenance_row["cache_provenance"] = cache_provenance
    receipts_jsonl = out_dir / "mx1t_facets_receipts.jsonl"
    _write_jsonl(receipts_jsonl, [*checkpoint_rows, *tail_average_rows])
    copy_jsonl = out_dir / "mx1t_checkpoint_copy_receipts.jsonl"
    _write_jsonl(copy_jsonl, copy_receipts)
    anchor_diff = abs(float(anchor_row["aggregate_d_seg"]) - float(args.facet_anchor_d_seg))
    result: dict[str, Any] = {
        "schema": "ddm_mx1t_torch_facets_result.v1",
        "status": "passed",
        "axis": "[macOS-CPU advisory torch upstream SegNet]",
        "score_claim": False,
        "verdict_scope": f"n{len(pair_ids)} arm-instrument checkpoint-series facets",
        "host": _host_fingerprint(),
        "checkpoint_dir": str(checkpoint_dir),
        "out_dir": str(out_dir),
        "checkpoint_copy_receipts_jsonl": str(copy_jsonl),
        "receipts_jsonl": str(receipts_jsonl),
        "checkpoint_rows": checkpoint_rows,
        "tail_average_rows": tail_average_rows,
        "anchor_check": {
            "step": int(args.facet_anchor_step),
            "expected_d_seg": float(args.facet_anchor_d_seg),
            "measured_d_seg": float(anchor_row["aggregate_d_seg"]),
            "abs_diff": anchor_diff,
            "tolerance": float(args.facet_anchor_tolerance),
            "status": "passed",
        },
        "cache_load": cache_meta,
        "cache_provenance": cache_provenance,
        "batching_scheme": {
            "segnet_batch_size": batch_size,
            "token_batch_shape": list(conditioning.shape),
            "target_batch_shape": list(target.shape),
        },
        "iteration_verdict": iteration_verdict,
        "recall_evidence": [
            {
                "scope": "Governing files",
                "query": "CHARTER.md, _common_contract.md, PROGRAM.md, CLAUDE.md/AGENTS.md, docs/operating_manual_craft_handoff.md, .omx/state/main_hot_state.md, upstream/evaluate.py",
                "found": "mx1t owns only the n32 CPU-torch scorer instrument; live frontier is S=0.7534578126155775 @ 357,837 B and contest pointer is borrowed/unmoved.",
                "changed_plan": "Kept CPU-only, score_claim=false, copied checkpoints before reading, and used mx1h step-1500 as a hard anchor.",
            },
            {
                "scope": "Prior MX1 verdict",
                "query": "torch-verdict|mx1h|d7f557bb7c|0.0010689099629720051",
                "found": "MX1H already implemented strict MLX NPZ -> torch loading and CPU upstream SegNet verdict; RR14 added fail-closed NPZ/history tests.",
                "changed_plan": "Extended the existing loader/verdict path with torch-facets instead of adding a new loader.",
            },
            {
                "scope": "Tail-average precedent",
                "query": "git log --grep dy2 and .omx/research/ddm_dy2_20260805/RECEIPT.md",
                "found": "dy2 registered jd1_plateau_tail_average_ema_v1 and documented an explicit growing-horizon tail average law for JD1.",
                "changed_plan": "Used a scoped post-hoc simple parameter mean for MX1 checkpoints and labeled it as this vehicle/stage only, not a general EMA verdict.",
            },
            {
                "scope": "Canonical equations",
                "query": ".venv/bin/python tools/list_canonical_equations.py --json | rg 'jd1_plateau|ema_decay|score_marginal|SegNet'",
                "found": "Relevant entries include score_marginal_lagrange_multipliers_v1, ema_decay_substrate_stage_aware_v1, and jd1_plateau_tail_average_ema_v1.",
                "changed_plan": "No score recomputation was promoted; tail average stayed a measured A/B row under the n32 advisory axis.",
            },
            {
                "scope": "Class order",
                "query": "CLAUDE.md SegNet class table and class-order corpus search",
                "found": "Canonical comma10k order is Road/Lane/Undrivable/Movable/MyCar; luma-sort is forbidden and historically wrong.",
                "changed_plan": "Per-class facets use that fixed class order and record the provenance in every row.",
            },
        ],
    }
    findings_path = out_dir / "MX1T_FINDINGS.md"
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_findings = findings_path.with_name(f".{findings_path.name}.tmp.{os.getpid()}")
    tmp_findings.write_text(_mx1t_findings_markdown(result), encoding="utf-8")
    os.replace(tmp_findings, findings_path)
    result["findings_path"] = str(findings_path)
    result_path = out_dir / "mx1t_facets_result.json"
    write_json_atomic(result_path, result)
    result["result_path"] = str(result_path)
    return result


def run_torch_verdict(args: argparse.Namespace) -> dict[str, Any]:
    lifted = _load_lifted_semantic()
    device = torch.device("cpu")
    checkpoint, checkpoint_meta = _load_torch_or_mlx_checkpoint_for_verdict(
        args.init,
        lifted=lifted,
    )
    pair_ids = _checkpoint_pair_ids(checkpoint_meta)
    step = int(checkpoint_meta["step"])
    history = list(checkpoint_meta.get("history") or [])
    comparison_row = _history_row_at_step(history, step)
    batch_size = int(args.verdict_batch_size)
    if batch_size <= 0:
        raise ValueError("--verdict-batch-size must be positive")

    conditioning_np, target_np, cache_meta, _cache_handles = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=None,
    )
    idx = torch.tensor(pair_ids, dtype=torch.long, device=device)
    conditioning = torch.from_numpy(conditioning_np).long().to(device)
    target = torch.from_numpy(target_np).long().to(device)
    del conditioning_np, target_np
    gc.collect()

    model = _build_torch_renderer(lifted, checkpoint).to(device)
    segnet = _load_upstream_segnet(device)
    per_pair: list[dict[str, Any]] = []
    total_mismatch = 0
    total_pixels = 0
    scorer_batch_shapes: list[list[int]] = []
    chunk_batch_sizes: list[int] = []
    start_time = time.time()
    with torch.no_grad():
        for start, stop in _iter_tensor_slices(len(pair_ids), batch_size):
            cond_chunk = conditioning[start:stop]
            target_chunk = target[start:stop]
            idx_chunk = idx[start:stop]
            frame_r = lifted.render_for_seg(
                model,
                cond_chunk,
                idx_chunk,
                exact_path=True,
            )
            logits = segnet(frame_r)
            pred = logits.argmax(dim=1)
            mismatch = pred != target_chunk
            flat = mismatch.reshape(mismatch.shape[0], -1)
            chunk_pixels = int(flat.shape[1])
            chunk_counts = flat.sum(dim=1).cpu().numpy().astype(np.int64)
            scorer_batch_shapes.append(list(frame_r.shape))
            chunk_batch_sizes.append(int(stop - start))
            for pair_id, count in zip(pair_ids[start:stop], chunk_counts, strict=True):
                mismatches = int(count)
                per_pair.append(
                    {
                        "pair_id": int(pair_id),
                        "mismatch_pixels": mismatches,
                        "pixels": chunk_pixels,
                        "d_seg": mismatches / max(chunk_pixels, 1),
                    }
                )
                total_mismatch += mismatches
                total_pixels += chunk_pixels
    elapsed = time.time() - start_time
    aggregate_dseg = total_mismatch / max(total_pixels, 1)
    proxy_dseg = float(comparison_row["d_seg_batch"])
    checkpoint_score_claim = None
    extra = checkpoint_meta.get("extra")
    if isinstance(extra, Mapping):
        checkpoint_score_claim = extra.get("score_claim")
    return {
        "schema": "ddm_mx1_torch_verdict.v1",
        "status": "passed",
        "axis": "[macOS-CPU advisory torch upstream SegNet]",
        "score_claim": False,
        "verdict_scope": f"n{len(pair_ids)} arm-selection instrument",
        "host": _host_fingerprint(),
        "source_repo_head": SOURCE_REPO_HEAD,
        "source_repo_root": SOURCE_REPO_ROOT,
        "checkpoint": {
            "path": str(args.init),
            "bytes": args.init.stat().st_size,
            "sha256": _sha256_file(args.init),
            "format": checkpoint_meta["format"],
            "step": step,
            "param_count": checkpoint_meta["param_count"],
            "axis": extra.get("axis") if isinstance(extra, Mapping) else None,
            "score_claim": checkpoint_score_claim,
        },
        "input_cache": str(args.input_cache),
        "target_cache": str(args.target_cache),
        "cache_load": cache_meta,
        "pair_ids": pair_ids,
        "pair_count": len(pair_ids),
        "token_batch_shape": list(conditioning.shape),
        "target_batch_shape": list(target.shape),
        "segnet_batch_size": batch_size,
        "segnet_chunk_batch_sizes": chunk_batch_sizes,
        "scorer_batch_shapes": scorer_batch_shapes,
        "contest_faithful_roundtrip": "lifted.render_for_seg(..., exact_path=True): bilinear up to 874x1164, uint8 STE, bilinear down to 384x512",
        "per_pair_d_seg": per_pair,
        "aggregate_d_seg": aggregate_dseg,
        "total_mismatch_pixels": total_mismatch,
        "total_pixels": total_pixels,
        "comparison_row": {
            "checkpoint_step": step,
            "mlx_in_training_d_seg_batch": proxy_dseg,
            "mlx_history_row": comparison_row,
            "authority_minus_mlx_proxy_d_seg": aggregate_dseg - proxy_dseg,
            "authority_over_mlx_proxy_d_seg": aggregate_dseg / proxy_dseg if proxy_dseg else None,
            "fp1_flat_paint_floor_d_seg": FP1_FLAT_PAINT_FLOOR_D_SEG,
            "authority_minus_fp1_floor_d_seg": aggregate_dseg - FP1_FLAT_PAINT_FLOOR_D_SEG,
            "authority_over_fp1_floor_d_seg": aggregate_dseg / FP1_FLAT_PAINT_FLOOR_D_SEG,
        },
        "elapsed_seconds": elapsed,
    }


def _coreml_compute_unit(coremltools: Any, name: str) -> Any:
    compute_unit = str(name).upper()
    mapping = {
        "CPU_ONLY": "CPU_ONLY",
        "CPU_AND_GPU": "CPU_AND_GPU",
        "CPU_AND_NE": "CPU_AND_NE",
        "ALL": "ALL",
    }
    attr = mapping.get(compute_unit)
    if attr is None:
        raise ValueError(f"unknown CoreML compute unit {name!r}")
    value = getattr(coremltools.ComputeUnit, attr, None)
    if value is None:
        raise RuntimeError(f"coremltools has no ComputeUnit.{attr}")
    return value


def run_coreml_segnet_parity(args: argparse.Namespace) -> dict[str, Any]:
    """Parity-gate the frozen upstream SegNet through a real CoreML conversion."""

    axis = "[CoreML-FP32 ANE advisory parity-gate; CPU-torch authority]"
    start_time = time.time()
    try:
        import coremltools as ct  # type: ignore[import-not-found]
    except Exception as exc:
        return {
            "schema": "ddm_mx1_coreml_segnet_parity.v1",
            "status": "blocked",
            "axis": axis,
            "score_claim": False,
            "verdict_scope": "ENVIRONMENT: coremltools import",
            "blocker": {"error_type": type(exc).__name__, "error": str(exc)},
        }
    try:
        compute_units = _coreml_compute_unit(ct, str(args.coreml_compute_units))
    except Exception as exc:
        return {
            "schema": "ddm_mx1_coreml_segnet_parity.v1",
            "status": "blocked",
            "axis": axis,
            "score_claim": False,
            "verdict_scope": "ENVIRONMENT: CoreML ANE compute-unit availability",
            "blocker": {"error_type": type(exc).__name__, "error": str(exc)},
        }
    lifted = _load_lifted_semantic()
    device = torch.device("cpu")
    checkpoint, checkpoint_meta = _load_torch_or_mlx_checkpoint_for_verdict(
        args.init,
        lifted=lifted,
    )
    pair_ids = _checkpoint_pair_ids(checkpoint_meta)
    if args.pairs > 0:
        pair_ids = pair_ids[: min(int(args.pairs), len(pair_ids))]
    if not pair_ids:
        raise ValueError("CoreML parity received no checkpoint pair IDs")
    conditioning_np, target_np, cache_meta, _cache_handles = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=None,
    )
    idx = torch.tensor(pair_ids, dtype=torch.long, device=device)
    conditioning = torch.from_numpy(conditioning_np).long().to(device)
    del conditioning_np, target_np
    gc.collect()
    model = _build_torch_renderer(lifted, checkpoint).to(device)
    segnet = _load_upstream_segnet(device)
    with torch.no_grad():
        frames = lifted.render_for_seg(model, conditioning, idx, exact_path=True).contiguous()
        torch_logits = segnet(frames).detach().cpu()
    try:
        traced = torch.jit.trace(segnet, frames[:1], strict=False)
        precision = getattr(getattr(ct, "precision", object()), "FLOAT32", None)
        convert_kwargs: dict[str, Any] = {
            "convert_to": "mlprogram",
            "inputs": [ct.TensorType(name="rgb_nchw", shape=frames[:1].shape)],
            "compute_units": compute_units,
        }
        if precision is not None:
            convert_kwargs["compute_precision"] = precision
        mlmodel = ct.convert(traced, **convert_kwargs)
    except Exception as exc:
        return {
            "schema": "ddm_mx1_coreml_segnet_parity.v1",
            "status": "blocked",
            "axis": axis,
            "score_claim": False,
            "verdict_scope": "FORMULATION: CoreML conversion of upstream SegNet",
            "host": _host_fingerprint(),
            "pair_ids": pair_ids,
            "cache_load": cache_meta,
            "blocker": {"error_type": type(exc).__name__, "error": str(exc)},
        }
    coreml_logits_parts: list[np.ndarray] = []
    try:
        for start, stop in _iter_tensor_slices(len(pair_ids), 1):
            batch = frames[start:stop].detach().cpu().numpy().astype(np.float32, copy=False)
            pred = mlmodel.predict({"rgb_nchw": batch})
            if not isinstance(pred, Mapping) or not pred:
                raise RuntimeError("CoreML predict returned no output mapping")
            array = np.asarray(next(iter(pred.values())), dtype=np.float32)
            coreml_logits_parts.append(array)
    except Exception as exc:
        return {
            "schema": "ddm_mx1_coreml_segnet_parity.v1",
            "status": "blocked",
            "axis": axis,
            "score_claim": False,
            "verdict_scope": "ENVIRONMENT: CoreML predict on real rendered frames",
            "host": _host_fingerprint(),
            "pair_ids": pair_ids,
            "cache_load": cache_meta,
            "blocker": {"error_type": type(exc).__name__, "error": str(exc)},
        }
    coreml_logits = torch.from_numpy(np.concatenate(coreml_logits_parts, axis=0))
    if list(coreml_logits.shape) != list(torch_logits.shape):
        return {
            "schema": "ddm_mx1_coreml_segnet_parity.v1",
            "status": "failed",
            "axis": axis,
            "score_claim": False,
            "verdict_scope": "INSTANCE: CoreML output shape mismatch",
            "host": _host_fingerprint(),
            "pair_ids": pair_ids,
            "torch_logits_shape": list(torch_logits.shape),
            "coreml_logits_shape": list(coreml_logits.shape),
            "cache_load": cache_meta,
        }
    torch_argmax = torch_logits.argmax(dim=1)
    coreml_argmax = coreml_logits.argmax(dim=1)
    diff_pixels = int((torch_argmax != coreml_argmax).sum().item())
    total_pixels = int(torch_argmax.numel())
    logit_delta = torch.abs(torch_logits.float() - coreml_logits.float())
    max_argmax_diff = int(args.coreml_parity_max_argmax_diff)
    status = "passed" if diff_pixels <= max_argmax_diff else "failed"
    return {
        "schema": "ddm_mx1_coreml_segnet_parity.v1",
        "status": status,
        "axis": axis,
        "score_claim": False,
        "verdict_scope": "n32-or-smaller real-frame CoreML SegNet parity gate",
        "host": _host_fingerprint(),
        "source_repo_head": SOURCE_REPO_HEAD,
        "source_repo_root": SOURCE_REPO_ROOT,
        "compute_units": str(args.coreml_compute_units),
        "pair_ids": pair_ids,
        "pair_count": len(pair_ids),
        "cache_load": cache_meta,
        "checkpoint": {
            "path": str(args.init),
            "bytes": args.init.stat().st_size,
            "sha256": _sha256_file(args.init),
            "format": checkpoint_meta["format"],
            "step": checkpoint_meta["step"],
        },
        "torch_logits_shape": list(torch_logits.shape),
        "coreml_logits_shape": list(coreml_logits.shape),
        "argmax_diff_pixels": diff_pixels,
        "total_pixels": total_pixels,
        "argmax_diff_rate": diff_pixels / max(total_pixels, 1),
        "max_allowed_argmax_diff_pixels": max_argmax_diff,
        "logit_abs_max_delta": float(logit_delta.max().item()),
        "logit_abs_mean_delta": float(logit_delta.mean().item()),
        "elapsed_seconds": time.time() - start_time,
    }


def run_mlx_parity(args: argparse.Namespace) -> dict[str, Any]:
    """Compare lifted torch CPU behavior to the MLX port on the selected host."""

    mx, _nn, _optim = require_mlx(device=args.device)
    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    lifted = _load_lifted_semantic()
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    torch_model = _build_torch_renderer(lifted, checkpoint)
    config = MlxSemanticConfig.from_pr130_checkpoint(
        checkpoint, consumer="ddm_mx1.run_mlx_parity"
    )
    mlx_model = make_mlx_renderer(config, device=args.device)
    load_torch_state_dict_into_mlx(mlx_model, checkpoint["state_dict"], device=args.device)

    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    idx_torch = torch.tensor(pair_ids, dtype=torch.long)
    conditioning_np, target_np, cache_meta, _cache_handles = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=None,
    )
    conditioning_torch = torch.from_numpy(conditioning_np).long()
    target_torch = torch.from_numpy(target_np).long()
    del conditioning_np, target_np
    gc.collect()

    with torch.no_grad():
        torch_frame = torch_model(conditioning_torch, idx_torch)
        torch_frame_r = lifted.render_for_seg(torch_model, conditioning_torch, idx_torch, exact_path=True)
        segnet_torch = _load_upstream_segnet(torch.device("cpu"))
        torch_logits = segnet_torch(torch_frame_r)
        torch_loss, torch_phase = lifted.curriculum_loss(
            torch_logits,
            target_torch,
            step=0,
            total_steps=max(args.steps, 1),
            ce_fraction=args.ce_fraction,
            softplus_fraction=args.softplus_fraction,
        )

    conditioning_mlx = mx.array(conditioning_torch.numpy().astype(np.int32, copy=False))
    target_mlx = mx.array(target_torch.numpy().astype(np.int32, copy=False))
    idx_mlx = mx.array(np.asarray(pair_ids, dtype=np.int32))
    mlx_frame = mlx_model(conditioning_mlx, idx_mlx)
    mlx_frame_r = apply_contest_faithful_roundtrip_nhwc(mlx_frame, output_hw=(384, 512), ste_round=True)
    segnet_mlx = torch_segnet_to_mlx(segnet_torch)
    mlx_logits_nhwc = segnet_mlx(mlx_frame_r)
    mlx_logits_nchw = mx.transpose(mlx_logits_nhwc, (0, 3, 1, 2))
    mlx_loss, mlx_phase = curriculum_loss_mlx(
        mx,
        mlx_logits_nchw,
        target_mlx,
        step=0,
        total_steps=max(args.steps, 1),
        ce_fraction=args.ce_fraction,
        softplus_fraction=args.softplus_fraction,
    )
    mx.eval(mlx_frame, mlx_logits_nchw, mlx_loss)

    torch_frame_nhwc = torch_frame.detach().permute(0, 2, 3, 1).cpu().numpy()
    mlx_frame_np = np.asarray(mlx_frame)
    torch_pred = torch_logits.argmax(dim=1).cpu().numpy()
    mlx_pred = np.asarray(mx.argmax(mlx_logits_nchw, axis=1))
    frame_max_abs = float(np.max(np.abs(torch_frame_nhwc - mlx_frame_np)))
    argmax_diff_count = int(np.count_nonzero(torch_pred != mlx_pred))
    argmax_equal = argmax_diff_count == 0
    loss_abs = abs(float(torch_loss.detach()) - _as_float(mlx_loss))
    return {
        "schema": "ddm_mx1_mlx_parity.v1",
        "status": "passed",
        "axis": "[torch-CPU reference vs MLX host parity]",
        "score_claim": False,
        "parity_input": "real built label caches, not synthetic tensors",
        "token_batch_shape": list(conditioning_torch.shape),
        "scorer_batch_shape": list(torch_frame_r.shape),
        "scorer_adapter": "tac.local_acceleration.mlx_scorer_adapters.torch_segnet_to_mlx",
        "gradient_parity_claim": False,
        "gradient_parity_scope": "not measured by this mode; training telemetry remains research-signal unless a separate gradient-parity check is added",
        "pairs": pair_ids,
        "cache_load": cache_meta,
        "raw_frame_max_abs": frame_max_abs,
        "seg_argmax_equal": argmax_equal,
        "seg_argmax_diff_count": argmax_diff_count,
        "loss_abs_delta": loss_abs,
        "torch_phase": torch_phase,
        "mlx_phase": mlx_phase,
        "torch_loss": float(torch_loss.detach()),
        "mlx_loss": _as_float(mlx_loss),
    }


def _derive_mem_budget_gb(explicit_gb: float | None, *, mem_probe: bool = False) -> dict[str, Any]:
    if explicit_gb is not None:
        if explicit_gb <= 0:
            raise ValueError("--mem-budget-gb must be positive when provided")
        return {
            "budget_gb": float(explicit_gb),
            "source": "explicit_cli",
            "mem_probe_cap_gb": 24.0 if mem_probe else None,
            "available_gib_at_start": _gib_or_none(_system_available_bytes()),
        }
    available = _system_available_bytes()
    if available is None:
        return {
            "budget_gb": None,
            "source": "unavailable_no_limit_applied",
            "mem_probe_cap_gb": 24.0 if mem_probe else None,
            "available_gib_at_start": None,
        }
    available_gib = float(available) / GIB
    default_budget = max(1.0, available_gib * 0.35)
    if mem_probe:
        default_budget = min(24.0, default_budget)
    return {
        "budget_gb": round(default_budget, 3),
        "source": "default_35pct_of_available_memory_at_start"
        if not mem_probe
        else "mem_probe_min_24gb_default_35pct_of_available_memory_at_start",
        "mem_probe_cap_gb": 24.0 if mem_probe else None,
        "available_gib_at_start": round(available_gib, 6),
    }


def _resolve_attr(obj: Any, dotted_name: str) -> Any:
    cur = obj
    for part in dotted_name.split("."):
        cur = getattr(cur, part)
    return cur


def _call_mlx_limit_with_signature(
    mx: Any,
    dotted_name: str,
    value: int,
    *,
    require_hard: bool,
    allow_soft: bool,
) -> dict[str, Any]:
    try:
        obj = _resolve_attr(mx, dotted_name)
    except AttributeError:
        return {
            "target": dotted_name,
            "status": "unavailable",
            "value_bytes": value,
            "hard_limit": False,
            "signature_form": "missing",
        }

    relaxed_supported: bool | None
    signature_text: str | None = None
    try:
        sig = inspect.signature(obj)
        signature_text = str(sig)
        relaxed_supported = "relaxed" in sig.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
        )
    except (TypeError, ValueError):
        relaxed_supported = None

    if relaxed_supported is not False:
        try:
            obj(value, relaxed=False)
            return {
                "target": dotted_name,
                "status": "applied",
                "value_bytes": value,
                "hard_limit": True,
                "relaxed": False,
                "signature": signature_text,
                "signature_form": "value_relaxed_false"
                if relaxed_supported is True
                else "value_relaxed_false_uninspectable",
            }
        except TypeError as exc:
            if relaxed_supported is True or (require_hard and not allow_soft):
                return {
                    "target": dotted_name,
                    "status": "failed",
                    "value_bytes": value,
                    "hard_limit": False,
                    "signature": signature_text,
                    "signature_form": "value_relaxed_false",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        except Exception as exc:
            return {
                "target": dotted_name,
                "status": "failed",
                "value_bytes": value,
                "hard_limit": False,
                "signature": signature_text,
                "signature_form": "value_relaxed_false",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    if require_hard and not allow_soft:
        return {
            "target": dotted_name,
            "status": "refused_soft_only",
            "value_bytes": value,
            "hard_limit": False,
            "signature": signature_text,
            "signature_form": "value_only",
            "error": "installed MLX memory limit API has no relaxed=False hard-cap form",
        }
    try:
        obj(value)
        return {
            "target": dotted_name,
            "status": "applied_soft_allowed",
            "value_bytes": value,
            "hard_limit": False,
            "relaxed": "default",
            "signature": signature_text,
            "signature_form": "value_only",
            "soft_limit_allowed_by_cli": allow_soft,
        }
    except Exception as exc:
        return {
            "target": dotted_name,
            "status": "failed",
            "value_bytes": value,
            "hard_limit": False,
            "signature": signature_text,
            "signature_form": "value_only",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _call_mlx_limit_value_only(mx: Any, dotted_name: str, value: int) -> dict[str, Any]:
    try:
        obj = _resolve_attr(mx, dotted_name)
    except AttributeError:
        return {
            "target": dotted_name,
            "status": "unavailable",
            "value_bytes": value,
            "hard_limit": False,
            "signature_form": "missing",
        }
    signature_text: str | None = None
    try:
        signature_text = str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature_text = None
    try:
        previous = obj(value)
        return {
            "target": dotted_name,
            "status": "applied",
            "value_bytes": value,
            "previous_value_bytes": previous if isinstance(previous, int) else None,
            "hard_limit": False,
            "signature": signature_text,
            "signature_form": "value_only_soft_guideline",
        }
    except Exception as exc:
        return {
            "target": dotted_name,
            "status": "failed",
            "value_bytes": value,
            "hard_limit": False,
            "signature": signature_text,
            "signature_form": "value_only_soft_guideline",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _derive_wired_limit_bytes(memory_limit: int) -> dict[str, Any]:
    total = _system_total_bytes()
    if total is None:
        return {
            "wired_limit": memory_limit,
            "system_total_bytes": None,
            "derived_wired_fraction": DEFAULT_WIRED_LIMIT_FRACTION,
            "source": "budget_bytes_total_memory_unavailable",
        }
    derived = int(float(total) * DEFAULT_WIRED_LIMIT_FRACTION)
    return {
        "wired_limit": min(memory_limit, derived),
        "system_total_bytes": total,
        "derived_wired_fraction": DEFAULT_WIRED_LIMIT_FRACTION,
        "source": "min_budget_bytes_35pct_total_memory",
    }


def _configure_mlx_memory_limits(
    mx: Any,
    explicit_gb: float | None,
    *,
    device: str,
    allow_soft_mem_limit: bool = False,
    mem_probe: bool = False,
    derived_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    derived = dict(derived_budget or _derive_mem_budget_gb(explicit_gb, mem_probe=mem_probe))
    budget_gb = derived["budget_gb"]
    software_cap_required = str(device).lower() == "gpu"
    if budget_gb is None:
        if software_cap_required:
            raise MemoryLimitConfigurationError(
                "REFUSED gpu mode: available memory could not be read, so no software memory budget can be derived"
            )
        return {
            **derived,
            "enforcement": "software_stage_step_cap",
            "software_cap_required": software_cap_required,
            "software_cap_installed": False,
            "software_budget_bytes": None,
            "memory_limit": None,
            "cache_limit": None,
            "wired_limit": None,
            "hard_limit_required": False,
            "hard_limit_satisfied": False,
            "soft_limit_allowed_by_cli": allow_soft_mem_limit,
            "calls": [],
        }
    memory_limit = int(float(budget_gb) * GIB)
    cache_limit = int(max(256 * 1024 * 1024, memory_limit * 0.25))
    wired = _derive_wired_limit_bytes(memory_limit)
    wired_limit = int(wired["wired_limit"])
    calls = [
        _call_mlx_limit_value_only(mx, "set_memory_limit", memory_limit),
        _call_mlx_limit_value_only(mx, "set_cache_limit", cache_limit),
        _call_mlx_limit_value_only(mx, "set_wired_limit", wired_limit),
    ]
    software_cap_installed = memory_limit > 0
    if software_cap_required and not software_cap_installed:
        raise MemoryLimitConfigurationError("REFUSED gpu mode: software memory budget was not installed")
    return {
        **derived,
        "enforcement": "software_stage_step_cap",
        "software_cap_required": software_cap_required,
        "software_cap_installed": software_cap_installed,
        "software_budget_bytes": memory_limit,
        "software_budget_rule": "mx.get_active_memory() + max(0, process_rss - start_process_rss) <= budget",
        "memory_limit": memory_limit,
        "cache_limit": cache_limit,
        "wired_limit": wired_limit,
        "wired_limit_derivation": wired,
        "wired_limit_semantics": (
            "MLX 0.31.2 set_wired_limit limits memory kept resident on macOS 15+; "
            "set_memory_limit is only a graph-evaluation guideline and is not a hard allocation cap."
        ),
        "hard_limit_required": False,
        "hard_limit_satisfied": False,
        "hard_limit_deprecated_reason": "MLX 0.31.2 set_memory_limit(limit) is soft and has no relaxed=False API.",
        "soft_limit_allowed_by_cli": allow_soft_mem_limit,
        "calls": calls,
    }


def _clear_mlx_cache(mx: Any) -> None:
    for name in ("clear_cache", "metal.clear_cache"):
        try:
            obj: Any = mx
            for part in name.split("."):
                obj = getattr(obj, part)
            obj()
            return
        except Exception:
            continue


def _mx_eval_setup_barrier(
    mx: Any,
    memory_probe: LoadPhaseMemoryProbe,
    stage: str,
    *values: Any,
    note: str | None = None,
) -> None:
    before_stage = f"before_{stage.removeprefix('after_')}" if stage.startswith("after_") else f"before_{stage}"
    memory_probe.sample_and_check(before_stage, mx=mx, note="before mx.eval setup barrier")
    if values:
        mx.eval(*values)
    memory_probe.sample_and_check(stage, mx=mx, note=note or "after mx.eval setup barrier")


def _derive_train_microbatch_plan(args: argparse.Namespace, *, total_pairs: int) -> dict[str, Any]:
    explicit = int(getattr(args, "microbatch_pairs", 0) or 0)
    policy = str(getattr(args, "microbatch_policy", "auto") or "auto")
    if policy not in MICROBATCH_POLICIES:
        raise ValueError(f"unknown --microbatch-policy {policy!r}")
    if explicit > 0:
        microbatch_pairs = max(1, min(explicit, total_pairs))
        return {
            "total_pairs": total_pairs,
            "microbatch_pairs": microbatch_pairs,
            "chunk_count": len(_iter_pair_chunks(total_pairs, microbatch_pairs)),
            "mode": "full_batch" if microbatch_pairs >= total_pairs else "serial_gradient_accumulation",
            "source": "explicit_cli",
            "policy": policy,
            "derivation": {"explicit_microbatch_pairs": explicit},
        }
    if str(getattr(args, "device", "")).lower() != "gpu":
        microbatch_pairs = max(1, total_pairs)
        return {
            "total_pairs": total_pairs,
            "microbatch_pairs": microbatch_pairs,
            "chunk_count": len(_iter_pair_chunks(total_pairs, microbatch_pairs)),
            "mode": "full_batch",
            "source": "cpu_full_batch_default",
            "policy": policy,
            "derivation": {"reason": "CPU path has no Metal allocator pressure"},
        }
    if policy == "full":
        microbatch_pairs = max(1, total_pairs)
        source = "microbatch_policy_full"
        derivation: dict[str, Any] = {
            "reason": "operator-requested full-batch footprint measurement via mem-probe",
        }
    elif policy == "legacy-4":
        microbatch_pairs = max(1, min(4, total_pairs))
        source = "legacy_gpu_default_4_pairs"
        derivation = {"reason": "explicit legacy policy requested"}
    else:
        microbatch_pairs = max(1, min(int(WC2_AUTO_MICROBATCH_ANCHOR["selected_default"]), total_pairs))
        source = "wc2_auto_empirical_wallclock_anchor"
        derivation = dict(WC2_AUTO_MICROBATCH_ANCHOR)
    return {
        "total_pairs": total_pairs,
        "microbatch_pairs": microbatch_pairs,
        "chunk_count": len(_iter_pair_chunks(total_pairs, microbatch_pairs)),
        "mode": "full_batch" if microbatch_pairs >= total_pairs else "serial_gradient_accumulation",
        "source": source,
        "policy": policy,
        "derivation": derivation,
    }


def _derive_train_microbatch_pairs(args: argparse.Namespace, *, total_pairs: int) -> int:
    return int(_derive_train_microbatch_plan(args, total_pairs=total_pairs)["microbatch_pairs"])


def _iter_pair_chunks(total_pairs: int, microbatch_pairs: int) -> list[tuple[int, int]]:
    if total_pairs <= 0:
        raise ValueError("total_pairs must be positive")
    if microbatch_pairs <= 0:
        raise ValueError("microbatch_pairs must be positive")
    return [(start, min(start + microbatch_pairs, total_pairs)) for start in range(0, total_pairs, microbatch_pairs)]


def _mlx_token_chunk(
    mx: Any,
    conditioning_np: np.ndarray,
    target_np: np.ndarray,
    pair_ids: list[int],
    start: int,
    stop: int,
) -> tuple[Any, Any, Any]:
    conditioning = mx.array(np.ascontiguousarray(conditioning_np[start:stop]))
    target = mx.array(np.ascontiguousarray(target_np[start:stop]))
    pair_idx = mx.array(np.asarray(pair_ids[start:stop], dtype=np.int32))
    return conditioning, target, pair_idx


def _tree_add_scaled(
    mx: Any,
    tree_flatten: Any,
    tree_unflatten: Any,
    accum: Mapping[str, Any] | None,
    update: Mapping[str, Any],
    scale: float,
) -> Mapping[str, Any]:
    scaled = [(name, value * scale if hasattr(value, "shape") else value) for name, value in tree_flatten(update)]
    if accum is None:
        return tree_unflatten(scaled)
    accum_flat = tree_flatten(accum)
    out = []
    for (left_name, left), (right_name, right) in zip(accum_flat, scaled, strict=True):
        if left_name != right_name:
            raise ValueError(f"gradient tree mismatch: {left_name!r} != {right_name!r}")
        out.append((left_name, left + right if hasattr(left, "shape") else left))
    return tree_unflatten(out)


def _resolve_train_compute_dtype(mx: Any, mode: str) -> Any | None:
    mode = str(mode)
    if mode == "fp32":
        return None
    if mode == "fp16":
        return mx.float16
    if mode == "bf16":
        dtype = getattr(mx, "bfloat16", None)
        if dtype is None:
            raise ValueError("MLX runtime has no bfloat16 dtype for --train-compute-dtype bf16")
        return dtype
    raise ValueError(f"unknown --train-compute-dtype {mode!r}")


def _cast_mlx_parameter_tree(
    tree_flatten: Any,
    tree_unflatten: Any,
    params: Mapping[str, Any],
    dtype: Any | None,
) -> Mapping[str, Any]:
    if dtype is None:
        return params
    casted = []
    for name, value in tree_flatten(params):
        if hasattr(value, "astype") and hasattr(value, "shape"):
            casted.append((name, value.astype(dtype)))
        else:
            casted.append((name, value))
    return tree_unflatten(casted)


def _maybe_compile_loss_function(mx: Any, fn: Any, *, enabled: bool) -> Any:
    if not enabled:
        return fn
    return mx.compile(fn)


def run_mlx_train(
    args: argparse.Namespace,
    *,
    memory_probe: LoadPhaseMemoryProbe | None = None,
) -> dict[str, Any]:
    """Run the real MLX Row-1 training path when MLX is available."""

    probe = memory_probe if memory_probe is not None else LoadPhaseMemoryProbe()
    mem_probe_mode = getattr(args, "mode", "") == "mem-probe"
    thread_pin = _apply_perf_thread_pin(getattr(args, "perf_thread_pin", "off"))
    budget_plan = _derive_mem_budget_gb(args.mem_budget_gb, mem_probe=mem_probe_mode)
    probe.install_software_budget(
        {
            **budget_plan,
            "software_cap_required": str(args.device).lower() == "gpu",
            "software_budget_bytes": None
            if budget_plan["budget_gb"] is None
            else int(float(budget_plan["budget_gb"]) * GIB),
        }
    )
    probe.sample_and_check("start")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    controller_policy = _load_m1_executor_policy(args) if not mem_probe_mode else None
    schedule_horizon_steps = int(args.steps)
    ema_policy: dict[str, Any] | None = None
    if controller_policy is not None:
        executor = controller_policy["executor"]
        predicate = controller_policy["predicate"]
        if int(args.pairs) != int(predicate["N"]):
            raise ValueError("M1 controller ticket N does not match --pairs")
        if int(args.eval_every) != int(predicate["eval_every_steps"]):
            raise ValueError("M1 controller ticket eval cadence does not match --eval-every")
        if int(args.checkpoint_every) != int(predicate["checkpoint_every_steps"]):
            raise ValueError("M1 controller ticket checkpoint cadence does not match --checkpoint-every")
        schedule_horizon_steps = int(executor["schedule"]["horizon_steps"])
        ema_policy = _derive_m1_ema_policy(controller_policy)
        if args.resume_from is not None:
            selection_path = Path(executor["schedule"]["selection_receipt_path"])
            if not selection_path.exists():
                raise ValueError(
                    f"M1 resume requires the same-object CPU schedule selection receipt at {selection_path}"
                )
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if (
                selection.get("schema") != M1_SCHEDULE_SELECTION_SCHEMA
                or selection.get("status") != "passed"
                or selection.get("selected_schedule") != "monotone_clamped_cosine"
            ):
                raise ValueError("M1 schedule selection receipt did not admit resume")
    probe.sample_and_check("before_init_checkpoint_torch_load")
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    probe.sample_and_check("after_init_checkpoint_torch_load")
    config = MlxSemanticConfig.from_pr130_checkpoint(
        checkpoint, consumer="ddm_mx1.run_mlx_train"
    )
    config = MlxSemanticConfig(
        **(
            config.asdict()
            | {
                "bits": args.bits,
                "lr": args.lr,
                "steps": schedule_horizon_steps,
                "ce_fraction": args.ce_fraction,
                "softplus_fraction": args.softplus_fraction,
            }
        )
    )
    conditioning_np, target_np, cache_meta, resident_cache_handles = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=probe,
        cache_residency=str(getattr(args, "cache_residency", "selected")),
    )

    probe.sample_and_check("before_require_mlx", note=f"device={args.device}")
    mx, _nn, optim = require_mlx(device=args.device)
    train_compute_dtype = _resolve_train_compute_dtype(
        mx,
        getattr(args, "train_compute_dtype", "fp32"),
    )
    compile_train_loss = bool(getattr(args, "compile_train_loss", False))
    try:
        probe.sample_and_check("before_mlx_memory_limit_configuration", mx=mx)
        memory_limits = _configure_mlx_memory_limits(
            mx,
            args.mem_budget_gb,
            device=args.device,
            allow_soft_mem_limit=bool(getattr(args, "allow_soft_mem_limit", False)),
            mem_probe=mem_probe_mode,
            derived_budget=budget_plan,
        )
        probe.install_software_budget(memory_limits)
    except Exception as exc:
        probe.sample_and_check(
            "after_require_mlx_memory_limit_configuration_failed",
            mx=mx,
            note=f"{type(exc).__name__}: {exc}",
        )
        raise
    probe.sample_and_check("after_require_mlx_and_memory_limits", mx=mx)
    from mlx.utils import tree_flatten, tree_unflatten

    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    probe.sample_and_check("before_model_init", mx=mx)
    model = make_mlx_renderer(config, device=args.device)
    probe.sample_and_check("before_optimizer_init", mx=mx)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    probe.sample_and_check("after_optimizer_init", mx=mx)
    _mx_eval_setup_barrier(mx, probe, "after_model_init", model.parameters())
    start_step = 0
    history: list[dict[str, Any]] = []
    resume_extra: dict[str, Any] = {}
    if args.resume_from is not None:
        resume = load_stage_checkpoint_npz(args.resume_from, model=model, optimizer=optimizer, mx=mx)
        start_step = int(resume["step"])
        history = list(resume["history"])
        resume_extra = dict(resume.get("extra") or {})
        resume_pair_ids = resume_extra.get("pair_ids")
        if resume_pair_ids is not None and [int(item) for item in resume_pair_ids] != pair_ids:
            raise ValueError(
                "resume checkpoint pair_ids do not match requested --pairs/--seed; "
                "refusing to train on a mismatched sample set"
            )
        del checkpoint
        gc.collect()
        _clear_mlx_cache(mx)
        probe.sample_and_check("after_resume_load_and_init_checkpoint_free", mx=mx)
    else:
        state_dict = checkpoint["state_dict"]
        load_torch_state_dict_into_mlx(model, state_dict, device=args.device)
        _mx_eval_setup_barrier(mx, probe, "after_model_weight_mlx_conversion", model.parameters())
        del state_dict
        del checkpoint
        gc.collect()
        _clear_mlx_cache(mx)
        probe.sample_and_check("after_torch_checkpoint_free", mx=mx)

    ema_flat: dict[str, Any] = {}
    if ema_policy is not None:
        live_flat = dict(tree_flatten(model.trainable_parameters()))
        if args.resume_from is None:
            ema_flat = {name: mx.array(value) for name, value in live_flat.items()}
        else:
            ema_checkpoint = resume_extra.get("ema_checkpoint")
            if not ema_checkpoint:
                raise ValueError("M1 resume checkpoint lacks required ema_checkpoint custody")
            with np.load(Path(ema_checkpoint), allow_pickle=False) as ema_payload:
                ema_flat = {
                    key.removeprefix("param::"): mx.array(ema_payload[key])
                    for key in ema_payload.files
                    if key.startswith("param::")
                }
            if set(ema_flat) != set(live_flat):
                raise ValueError("M1 EMA checkpoint parameter set differs from live checkpoint")
            for name, live_value in live_flat.items():
                if tuple(ema_flat[name].shape) != tuple(live_value.shape):
                    raise ValueError(f"M1 EMA shape mismatch for {name}")

    total_pairs = len(pair_ids)
    chunk_plan = {
        **_derive_train_microbatch_plan(args, total_pairs=total_pairs),
        "train_compute_dtype": str(getattr(args, "train_compute_dtype", "fp32")),
        "compile_train_loss": compile_train_loss,
        "perf_thread_pin": str(getattr(args, "perf_thread_pin", "off")),
        "cache_residency": str(getattr(args, "cache_residency", "selected")),
        "resident_cache_handle_count": len(resident_cache_handles),
        "microbatch_hygiene": str(getattr(args, "microbatch_hygiene", "per-chunk")),
        "microbatch_chunk_cache": bool(getattr(args, "microbatch_chunk_cache", False)),
    }
    microbatch_pairs = int(chunk_plan["microbatch_pairs"])
    conditioning = target = pair_idx = None
    probe.sample_and_check(
        "before_selected_tokens_mlx_conversion_plan",
        mx=mx,
        note=json.dumps(chunk_plan, sort_keys=True),
    )
    if microbatch_pairs >= total_pairs:
        conditioning, target, pair_idx = _mlx_token_chunk(
            mx,
            conditioning_np,
            target_np,
            pair_ids,
            0,
            total_pairs,
        )
        _mx_eval_setup_barrier(
            mx,
            probe,
            "after_selected_tokens_mlx_conversion",
            conditioning,
            target,
            pair_idx,
            note=json.dumps(chunk_plan, sort_keys=True),
        )
        del conditioning_np, target_np
        gc.collect()
        _clear_mlx_cache(mx)
        probe.sample_and_check("after_selected_token_numpy_free", mx=mx)
    else:
        probe.sample_and_check(
            "after_selected_tokens_lazy_chunk_plan",
            mx=mx,
            note=json.dumps(chunk_plan, sort_keys=True),
        )

    probe.sample_and_check("before_upstream_segnet_torch_load", mx=mx)
    segnet_torch = _load_upstream_segnet(torch.device("cpu"))
    probe.sample_and_check("after_upstream_segnet_torch_load", mx=mx)
    probe.sample_and_check("before_segnet_mlx_conversion", mx=mx)
    segnet_mlx = torch_segnet_to_mlx(segnet_torch)
    segnet_params = segnet_mlx.parameters() if hasattr(segnet_mlx, "parameters") else []
    _mx_eval_setup_barrier(mx, probe, "after_segnet_mlx_conversion", segnet_params)
    del segnet_torch
    gc.collect()
    _clear_mlx_cache(mx)
    probe.sample_and_check("after_segnet_torch_free", mx=mx)
    start_time = time.time()
    last_stage_path: Path | None = None

    microbatch_hygiene = str(getattr(args, "microbatch_hygiene", "per-chunk"))
    chunk_cache: dict[tuple[int, int], tuple[Any, Any, Any]] | None = None
    if bool(getattr(args, "microbatch_chunk_cache", False)) and microbatch_pairs < total_pairs:
        chunk_cache = {
            (start, stop): _mlx_token_chunk(mx, conditioning_np, target_np, pair_ids, start, stop)
            for start, stop in _iter_pair_chunks(total_pairs, microbatch_pairs)
        }
        for cached_chunk in chunk_cache.values():
            mx.eval(*cached_chunk)
        probe.sample_and_check("after_microbatch_chunk_cache_materialize", mx=mx)

    journal_path: Path | None = None
    decision_path: Path | None = None
    terminal_receipt_path: Path | None = None
    if controller_policy is not None:
        executor = controller_policy["executor"]
        journal_path = Path(executor["journal_path"])
        decision_path = Path(executor["decision_path"])
        terminal_receipt_path = Path(executor["terminal_receipt_paths_by_key"][controller_policy["argv_key"]])
        if args.resume_from is None and journal_path.exists():
            raise ValueError(f"fresh M1 fire refuses an existing append-only journal: {journal_path}")
        if args.resume_from is None and any(args.run_dir.glob("mlx_stage_step??????.npz")):
            raise ValueError(f"fresh M1 fire refuses existing stage checkpoints under sacred run dir {args.run_dir}")
        if args.resume_from is not None and not journal_path.exists():
            raise ValueError("M1 resume requires the prior durable eval journal")
        _append_jsonl_durable(
            journal_path,
            {
                "schema": M1_EVAL_JOURNAL_SCHEMA,
                "row_kind": "segment_start",
                "segment_id": _ticket_attempt_id(),
                "resume_step": start_step,
                "resume_from": None if args.resume_from is None else str(args.resume_from),
                "generated_utc": _utc_now_iso(),
            },
        )

    def evaluate_current_dseg(eval_step: int) -> float:
        if microbatch_pairs >= total_pairs:
            assert conditioning is not None and target is not None and pair_idx is not None
            frame = model(conditioning, pair_idx)
            frame_r = apply_contest_faithful_roundtrip_nhwc(frame, output_hw=(384, 512), ste_round=True)
            logits = segnet_mlx(frame_r)
            pred = mx.argmax(logits, axis=-1)
            dseg_value = mx.mean((pred != target).astype(mx.float32))
            mx.eval(dseg_value)
            probe.check_budget(f"after_eval_step_{eval_step:06d}", mx=mx)
            return float(dseg_value)
        mismatch_count = 0.0
        pixel_count = 0
        for chunk_index, (start, stop) in enumerate(
            _iter_pair_chunks(total_pairs, microbatch_pairs),
            start=1,
        ):
            conditioning_chunk, target_chunk, pair_idx_chunk = _mlx_token_chunk(
                mx,
                conditioning_np,
                target_np,
                pair_ids,
                start,
                stop,
            )
            frame = model(conditioning_chunk, pair_idx_chunk)
            frame_r = apply_contest_faithful_roundtrip_nhwc(frame, output_hw=(384, 512), ste_round=True)
            logits = segnet_mlx(frame_r)
            pred = mx.argmax(logits, axis=-1)
            mismatch = mx.sum((pred != target_chunk).astype(mx.float32))
            mx.eval(mismatch)
            probe.check_budget(
                f"after_eval_step_{eval_step:06d}_chunk_{chunk_index:03d}",
                mx=mx,
            )
            mismatch_count += float(mismatch)
            pixel_count += int(stop - start) * int(target_np.shape[-2]) * int(target_np.shape[-1])
            del conditioning_chunk, target_chunk, pair_idx_chunk, frame, frame_r, logits, pred, mismatch
            gc.collect()
            _clear_mlx_cache(mx)
        return mismatch_count / max(pixel_count, 1)

    def checkpoint_extra(ema_path: Path | None, tail_path: Path | None) -> dict[str, Any]:
        return {
            "pair_ids": pair_ids,
            "score_claim": False,
            "axis": "[macOS-MLX research-signal]",
            "source_repo_head": SOURCE_REPO_HEAD,
            "throughput_flags": {
                "train_compute_dtype": str(getattr(args, "train_compute_dtype", "fp32")),
                "compile_train_loss": compile_train_loss,
                "perf_thread_pin": str(getattr(args, "perf_thread_pin", "off")),
                "microbatch_policy": str(getattr(args, "microbatch_policy", "auto")),
                "cache_residency": str(getattr(args, "cache_residency", "selected")),
                "microbatch_hygiene": str(getattr(args, "microbatch_hygiene", "per-chunk")),
                "microbatch_chunk_cache": bool(getattr(args, "microbatch_chunk_cache", False)),
            },
            "schedule": {
                "base_lr": float(args.lr),
                "base_lr_status": "BORROWED_CANDIDATE_NOT_ADOPTED",
                "horizon_steps": schedule_horizon_steps,
                "extension_rule": "hold terminal cosine value; never recompute horizon",
            },
            "ema_checkpoint": None if ema_path is None else str(ema_path),
            "tail_average_checkpoint": None if tail_path is None else str(tail_path),
            "selection_status": "QUEUED_SAME_OBJECT_CPU_FACETS",
        }

    def save_checkpoint_bundle(checkpoint_step: int) -> tuple[Path, Path | None, Path | None]:
        nonlocal last_stage_path
        live_path = args.run_dir / f"mlx_stage_step{checkpoint_step:06d}.npz"
        ema_path = args.run_dir / f"mlx_ema_step{checkpoint_step:06d}.npz" if ema_policy is not None else None
        member_paths = sorted(args.run_dir.glob("mlx_stage_step??????.npz"))
        anticipated_members = [*member_paths, live_path]
        tail_k = 0 if ema_policy is None else int(ema_policy["tail_average_k"])
        tail_path = (
            args.run_dir / f"mlx_tail_average_k{tail_k}_step{checkpoint_step:06d}.npz"
            if tail_k > 0 and len(anticipated_members) >= tail_k
            else None
        )
        extra = checkpoint_extra(ema_path, tail_path)
        save_stage_checkpoint_npz(
            live_path,
            model=model,
            config=config,
            step=checkpoint_step,
            history=history,
            optimizer_state=optimizer.state,
            extra=extra,
        )
        save_stage_checkpoint_npz(
            args.run_dir / "mlx.latest.npz",
            model=model,
            config=config,
            step=checkpoint_step,
            history=history,
            optimizer_state=optimizer.state,
            extra=extra,
        )
        if ema_path is not None:
            live_weights = list(tree_flatten(model.parameters()))
            model.load_weights(list(ema_flat.items()), strict=False)
            mx.eval(model.parameters())
            save_stage_checkpoint_npz(
                ema_path,
                model=model,
                config=config,
                step=checkpoint_step,
                history=history,
                optimizer_state=None,
                extra=extra
                | {
                    "parameter_basis": "ema_shadow",
                    "ema_law_ref": EMA_DECAY_LAW_REF,
                    "ema_decay": float(ema_policy["derived_decay"]),
                },
            )
            save_stage_checkpoint_npz(
                args.run_dir / "mlx.ema.latest.npz",
                model=model,
                config=config,
                step=checkpoint_step,
                history=history,
                optimizer_state=None,
                extra=extra
                | {
                    "parameter_basis": "ema_shadow",
                    "ema_law_ref": EMA_DECAY_LAW_REF,
                    "ema_decay": float(ema_policy["derived_decay"]),
                },
            )
            model.load_weights(live_weights, strict=False)
            mx.eval(model.parameters())
        if tail_path is not None:
            _write_tail_average_npz(
                anticipated_members[-tail_k:],
                tail_path,
                selection_extra={
                    "parameter_basis": "tail_average",
                    "tail_average_k": tail_k,
                    "tail_average_member_paths": [str(path) for path in anticipated_members[-tail_k:]],
                    "selection_status": "QUEUED_SAME_OBJECT_CPU_FACETS",
                },
            )
        last_stage_path = live_path
        return live_path, ema_path, tail_path

    if controller_policy is not None and start_step == 0:
        initial_dseg = evaluate_current_dseg(0)
        history.append(
            {
                "step": 0,
                "phase": "initial",
                "loss": None,
                "lr": float(args.lr),
                "d_seg_batch": initial_dseg,
            }
        )
        save_checkpoint_bundle(0)

    for step in range(start_step, args.steps):
        optimizer.learning_rate = _m1_cosine_lr(args.lr, step, schedule_horizon_steps)
        base_params = model.trainable_parameters()
        step_for_loss = min(step, schedule_horizon_steps - 1)

        def loss_for_batch(
            params: Mapping[str, Any],
            conditioning_batch: Any,
            target_batch: Any,
            pair_idx_batch: Any,
            *,
            step_for_loss: int = step_for_loss,
        ) -> Any:
            if step_for_loss < args.float_warmup_steps:
                active_params = params
                phase_prefix = "float_"
            else:
                active_params = fake_quantize_parameter_tree(
                    mx,
                    tree_flatten,
                    tree_unflatten,
                    params,
                    bits=args.bits,
                )
                phase_prefix = ""
            active_params = _cast_mlx_parameter_tree(
                tree_flatten,
                tree_unflatten,
                active_params,
                train_compute_dtype,
            )
            model.update(active_params)
            frame = model(conditioning_batch, pair_idx_batch)
            frame_r = apply_contest_faithful_roundtrip_nhwc(frame, output_hw=(384, 512), ste_round=True)
            logits_nhwc = segnet_mlx(frame_r)
            logits_nchw = mx.transpose(logits_nhwc, (0, 3, 1, 2))
            loss, phase = curriculum_loss_mlx(
                mx,
                logits_nchw,
                target_batch,
                step=max(0, step_for_loss - args.float_warmup_steps),
                total_steps=max(1, schedule_horizon_steps - args.float_warmup_steps),
                ce_fraction=args.ce_fraction,
                softplus_fraction=args.softplus_fraction,
            )
            loss_for_batch.phase = phase_prefix + phase  # type: ignore[attr-defined]
            return loss

        if microbatch_pairs >= total_pairs:
            assert conditioning is not None and target is not None and pair_idx is not None
            if step == start_step or args.steps <= 3:
                probe.sample_and_check(
                    f"before_train_step_{step + 1:06d}_full_batch_value_and_grad",
                    mx=mx,
                )

            def loss_for_params(params: Mapping[str, Any]) -> Any:
                return loss_for_batch(params, conditioning, target, pair_idx)

            value, grads = mx.value_and_grad(
                _maybe_compile_loss_function(mx, loss_for_params, enabled=compile_train_loss)
            )(base_params)
            phase = getattr(loss_for_batch, "phase", "unknown")
        else:
            value = None
            grads = None
            phase = "unknown"
            for chunk_index, (start, stop) in enumerate(
                _iter_pair_chunks(total_pairs, microbatch_pairs),
                start=1,
            ):
                if chunk_cache is not None:
                    conditioning_chunk, target_chunk, pair_idx_chunk = chunk_cache[(start, stop)]
                else:
                    conditioning_chunk, target_chunk, pair_idx_chunk = _mlx_token_chunk(
                        mx,
                        conditioning_np,
                        target_np,
                        pair_ids,
                        start,
                        stop,
                    )
                if step == start_step or args.steps <= 3:
                    probe.sample_and_check(
                        f"before_train_step_{step + 1:06d}_chunk_{chunk_index:03d}_value_and_grad",
                        mx=mx,
                        note=f"rows={start}:{stop}",
                    )

                def loss_for_params(
                    params: Mapping[str, Any],
                    conditioning_chunk: Any = conditioning_chunk,
                    target_chunk: Any = target_chunk,
                    pair_idx_chunk: Any = pair_idx_chunk,
                ) -> Any:
                    return loss_for_batch(params, conditioning_chunk, target_chunk, pair_idx_chunk)

                chunk_value, chunk_grads = mx.value_and_grad(
                    _maybe_compile_loss_function(mx, loss_for_params, enabled=compile_train_loss)
                )(base_params)
                mx.eval(chunk_value, chunk_grads)
                probe.check_budget(
                    f"after_train_step_{step + 1:06d}_chunk_{chunk_index:03d}_value_and_grad",
                    mx=mx,
                )
                weight = float(stop - start) / float(total_pairs)
                grads = _tree_add_scaled(
                    mx,
                    tree_flatten,
                    tree_unflatten,
                    grads,
                    chunk_grads,
                    weight,
                )
                value = chunk_value * weight if value is None else value + chunk_value * weight
                mx.eval(value, grads)
                phase = getattr(loss_for_batch, "phase", "unknown")
                del conditioning_chunk, target_chunk, pair_idx_chunk, chunk_value, chunk_grads
                if microbatch_hygiene == "per-chunk":
                    gc.collect()
                    _clear_mlx_cache(mx)
                probe.check_budget(
                    f"after_train_step_{step + 1:06d}_chunk_{chunk_index:03d}_free",
                    mx=mx,
                )
            if value is None or grads is None:
                raise RuntimeError("microbatch path produced no gradients")
        model.update(base_params)
        model.update(optimizer.apply_gradients(grads, base_params))
        mx.eval(value, model.parameters(), optimizer.state)
        if ema_policy is not None:
            decay = float(ema_policy["derived_decay"])
            live_after_step = dict(tree_flatten(model.trainable_parameters()))
            ema_flat = _update_m1_ema_flat(ema_flat, live_after_step, decay=decay)
            mx.eval(*ema_flat.values())
        if microbatch_hygiene == "per-step" and microbatch_pairs < total_pairs:
            gc.collect()
            _clear_mlx_cache(mx)
        probe.check_budget(f"after_train_step_{step + 1:06d}", mx=mx)
        record: dict[str, Any] = {
            "step": step + 1,
            "phase": phase,
            "loss": float(value),
            "lr": float(optimizer.learning_rate),
        }
        if step == start_step or step + 1 == int(args.steps) or args.steps <= 3:
            # Resume-path probes have steps = resume_base + probe_window (e.g. 5253), so the
            # old first-step-only condition never emitted the required final-stage sample
            # (after_train_step_{steps}) that metal_fire_clearance demands.
            probe.sample(f"after_train_step_{step + 1:06d}", mx=mx)
        controller_should_halt = False
        controller_decision_payload: dict[str, Any] | None = None
        if (step + 1) % max(args.eval_every, 1) == 0 or step + 1 == args.steps:
            record["d_seg_batch"] = evaluate_current_dseg(step + 1)
            if args.steps <= 3:
                probe.sample(f"after_eval_step_{step + 1:06d}", mx=mx)
            if controller_policy is not None:
                assert journal_path is not None and decision_path is not None
                active_before = _read_active_m1_eval_rows(journal_path)
                prior_best = min(
                    (float(row["d_seg_batch_mlx"]) for row in active_before),
                    default=float(record["d_seg_batch"]),
                )
                checkpoint_path = last_stage_path
                journal_row = {
                    "schema": M1_EVAL_JOURNAL_SCHEMA,
                    "row_kind": "eval",
                    "segment_id": None,
                    "step": step + 1,
                    "objective_S": 100.0 * float(record["d_seg_batch"]),
                    "d_seg_batch_mlx": float(record["d_seg_batch"]),
                    "best_d_seg_batch_mlx": min(prior_best, float(record["d_seg_batch"])),
                    "loss": float(record["loss"]),
                    "lr": float(record["lr"]),
                    "wall_seconds": time.time() - start_time,
                    "weights_stepped": step + 1,
                    "accepted_batch_fraction": 1.0,
                    "checkpoint": None
                    if checkpoint_path is None
                    else {
                        "path": str(checkpoint_path),
                        "sha256": _sha256_file(checkpoint_path),
                        "step": _checkpoint_step_npz(checkpoint_path),
                    },
                    "generated_utc": _utc_now_iso(),
                }
                _append_jsonl_durable(journal_path, journal_row)
                active_rows = _read_active_m1_eval_rows(journal_path)
                executor = controller_policy["executor"]
                safety_bound = float(executor["safety_bound_steps_by_key"][controller_policy["argv_key"]])
                at_step_boundary = step + 1 >= int(safety_bound)
                decision = evaluate_staircase_aware_stop(
                    active_rows,
                    controller_policy["trajectory_config"],
                    controller_policy["staircase_config"],
                    safety_bound_compute=safety_bound,
                    boundary_kind="steps" if at_step_boundary else None,
                )
                controller_decision_payload = decision.to_payload()
                schedule_gate: dict[str, Any] | None = None
                schedule = executor["schedule"]
                selection_path = Path(schedule["selection_receipt_path"])
                if (
                    args.resume_from is None
                    and step + 1 >= int(schedule["calibration_step"])
                    and not selection_path.exists()
                ):
                    controller_decision_payload["action"] = "QUEUE_RESUME"
                    controller_decision_payload["should_halt"] = True
                    controller_decision_payload["blockers"] = [
                        *controller_decision_payload["blockers"],
                        "same_object_cpu_schedule_calibration_owed",
                    ]
                    schedule_gate = {
                        "status": "QUEUED",
                        "calibration_step": int(schedule["calibration_step"]),
                        "selection_receipt_path": str(selection_path),
                    }
                controller_should_halt = bool(controller_decision_payload["should_halt"])
                _append_jsonl_durable(
                    decision_path,
                    {
                        "schema": M1_STOP_DECISION_SCHEMA,
                        "step": step + 1,
                        "generated_utc": _utc_now_iso(),
                        "ticket_path": str(controller_policy["ticket_path"]),
                        "ticket_sha256": _sha256_file(controller_policy["ticket_path"]),
                        "journal_path": str(journal_path),
                        "journal_sha256": _sha256_file(journal_path),
                        "decision": controller_decision_payload,
                        "schedule_gate": schedule_gate,
                    },
                )
        history.append(record)
        if (step + 1) % max(args.checkpoint_every, 1) == 0 or step + 1 == args.steps or controller_should_halt:
            live_path, ema_path, tail_path = save_checkpoint_bundle(step + 1)
            if args.steps <= 3:
                probe.sample_and_check(f"after_checkpoint_step_{step + 1:06d}", mx=mx)
            if controller_should_halt:
                assert terminal_receipt_path is not None and controller_decision_payload is not None
                action = str(controller_decision_payload["action"])
                trajectory_payload = controller_decision_payload.get("trajectory_decision") or {}
                at_step_cap = controller_decision_payload.get("boundary_kind") == "steps"
                canonical_receipt = None
                terminal_mode = "event_stop"
                if action == "STOP_CONVERGED":
                    canonical_receipt = build_cap_stop_receipt(
                        stop_reason="converged",
                        steps_run=step + 1,
                        cap=None,
                        still_descending=False,
                    ).to_payload()
                elif action == "ROLLBACK_OR_RETREAT":
                    canonical_receipt = build_cap_stop_receipt(
                        stop_reason="failed",
                        steps_run=step + 1,
                        cap=None,
                        still_descending=None,
                    ).to_payload()
                elif at_step_cap:
                    terminal_mode = "step_cap"
                    canonical_receipt = build_cap_stop_receipt(
                        stop_reason="cap_bound",
                        steps_run=step + 1,
                        cap=int(trajectory_payload["safety_bound_compute"]),
                        still_descending=action == "QUEUE_RESUME",
                    ).to_payload()
                elif "same_object_cpu_schedule_calibration_owed" in set(
                    controller_decision_payload.get("blockers") or []
                ):
                    terminal_mode = "schedule_calibration_boundary"
                write_json_atomic(
                    terminal_receipt_path,
                    {
                        "schema": M1_TERMINAL_RECEIPT_SCHEMA,
                        "status": "halted",
                        "terminal_mode": terminal_mode,
                        "action": action,
                        "step": step + 1,
                        "decision": controller_decision_payload,
                        "cap_stop_receipt": canonical_receipt,
                        "live_checkpoint": str(live_path),
                        "ema_checkpoint": None if ema_path is None else str(ema_path),
                        "tail_average_checkpoint": None if tail_path is None else str(tail_path),
                        "resume_argv_key": controller_policy["executor"]["resume_argv_key"],
                        "resume_argv": controller_policy["ticket"][controller_policy["executor"]["resume_argv_key"]],
                        "same_object_cpu_selection": controller_policy["executor"]["same_object_cpu_selection"],
                        "generated_utc": _utc_now_iso(),
                    },
                )
                break
    elapsed = time.time() - start_time
    latest_path = args.run_dir / "mlx.latest.npz"
    resume_check = load_stage_checkpoint_npz(latest_path, model=model, optimizer=optimizer, mx=mx)
    probe.sample_and_check("after_resume_check", mx=mx)
    return {
        "schema": "ddm_mx1_mlx_train.v1",
        "status": "passed",
        "axis": "[macOS-MLX research-signal]",
        "score_claim": False,
        "pairs": pair_ids,
        "cache_load": cache_meta,
        "memory_limits": memory_limits,
        "microbatch_plan": chunk_plan,
        "throughput_flags": {
            "train_compute_dtype": str(getattr(args, "train_compute_dtype", "fp32")),
            "compile_train_loss": compile_train_loss,
            "perf_thread_pin": thread_pin,
        },
        "software_budget": probe.budget_summary(),
        "load_memory_samples": probe.samples,
        "load_memory_peak": probe.peak(),
        "steps": args.steps,
        "steps_run": int(history[-1]["step"]) if history else start_step,
        "start_step": start_step,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(args.steps - start_step, 1),
        "history": history,
        "stage_checkpoint": str(last_stage_path) if last_stage_path else None,
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_bytes": latest_path.stat().st_size if latest_path.exists() else None,
        "latest_checkpoint_sha256": _sha256_file(latest_path) if latest_path.exists() else None,
        "ema_checkpoint": (
            str(args.run_dir / "mlx.ema.latest.npz") if (args.run_dir / "mlx.ema.latest.npz").exists() else None
        ),
        "ema_policy": ema_policy,
        "schedule_horizon_steps": schedule_horizon_steps,
        "eval_journal": None if journal_path is None else str(journal_path),
        "stop_decisions": None if decision_path is None else str(decision_path),
        "terminal_receipt": (None if terminal_receipt_path is None else str(terminal_receipt_path)),
        "selection_status": "QUEUED_SAME_OBJECT_CPU_FACETS" if ema_policy is not None else None,
        "resume_load": resume_check,
    }


def _mem_probe_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["mode"] = "mem-probe"
    resume_from = payload.get("resume_from")
    resume_step = _checkpoint_step_npz(Path(resume_from)) if resume_from is not None else 0
    payload["steps"] = resume_step + int(args.mem_probe_steps)
    payload["mem_probe_resume_base_step"] = resume_step
    payload["eval_every"] = 1
    payload["checkpoint_every"] = max(1, int(args.mem_probe_steps))
    return argparse.Namespace(**payload)


def _build_mem_probe_receipt(
    *,
    args: argparse.Namespace,
    probe_args: argparse.Namespace,
    probe: LoadPhaseMemoryProbe,
    status: str,
    train_result: dict[str, Any] | None,
    blocker: dict[str, Any] | None,
) -> dict[str, Any]:
    required_stage = f"after_train_step_{int(probe_args.steps):06d}"
    final_step_sample = next(
        (row for row in probe.samples if row.get("stage") == required_stage),
        None,
    )
    has_mlx_allocator_telemetry = bool(
        final_step_sample
        and any(final_step_sample.get(key) is not None for key in ("mlx_active_gib", "mlx_cache_gib", "mlx_peak_gib"))
    )
    software_budget = (
        train_result.get("software_budget")
        if train_result is not None and isinstance(train_result.get("software_budget"), dict)
        else probe.budget_summary()
    )
    software_last = software_budget.get("last_check")
    software_budget_clearance = (
        # Resume-path probes end at an absolute step index (e.g. 5253) but only
        # EXECUTE mem_probe_steps of them; the budget-check floor is the executed
        # window, never the absolute end index (which no 3-step probe can reach).
        int(software_budget.get("check_count") or 0) >= int(args.mem_probe_steps)
        and isinstance(software_last, dict)
        and software_last.get("within_budget") is True
    )
    metal_fire_clearance = (
        status == "passed"
        and final_step_sample is not None
        and has_mlx_allocator_telemetry
        and software_budget_clearance
    )
    return {
        "schema": MEM_PROBE_RECEIPT_SCHEMA,
        "status": status,
        "axis": "[load-phase memory telemetry; score_claim=false]",
        "score_claim": False,
        "source_repo_head": SOURCE_REPO_HEAD,
        "source_repo_root": SOURCE_REPO_ROOT,
        "host": _host_fingerprint(),
        "mode": "mem-probe",
        "device_request": args.device,
        "pairs": int(args.pairs),
        "pair_ids": _select_stratified_indices(args.pairs, seed=args.seed),
        "requested_training_steps": int(probe_args.steps),
        "mem_probe_steps_after_resume": int(args.mem_probe_steps),
        "resume_from": None if args.resume_from is None else str(args.resume_from),
        "resume_base_step": int(getattr(probe_args, "mem_probe_resume_base_step", 0)),
        "mem_budget_gb_arg": args.mem_budget_gb,
        "input_cache": str(args.input_cache),
        "target_cache": str(args.target_cache),
        "init_checkpoint": str(args.init),
        "argv_config": {
            "device": args.device,
            "pairs": int(args.pairs),
            "lr": float(args.lr),
            "ce_fraction": float(args.ce_fraction),
            "softplus_fraction": float(args.softplus_fraction),
            "bits": int(args.bits),
            "microbatch_pairs": int(getattr(args, "microbatch_pairs", 0) or 0),
            "microbatch_policy": str(getattr(args, "microbatch_policy", "auto")),
            "cache_residency": str(getattr(args, "cache_residency", "selected")),
            "microbatch_hygiene": str(getattr(args, "microbatch_hygiene", "per-chunk")),
            "microbatch_chunk_cache": bool(getattr(args, "microbatch_chunk_cache", False)),
            "verdict_batch_size": int(getattr(args, "verdict_batch_size", 32)),
            "float_warmup_steps": int(getattr(args, "float_warmup_steps", 0)),
            "train_compute_dtype": str(getattr(args, "train_compute_dtype", "fp32")),
            "compile_train_loss": bool(getattr(args, "compile_train_loss", False)),
            "perf_thread_pin": str(getattr(args, "perf_thread_pin", "off")),
            "mem_budget_gb": args.mem_budget_gb,
            "allow_soft_mem_limit": bool(getattr(args, "allow_soft_mem_limit", False)),
            "input_cache": str(args.input_cache),
            "target_cache": str(args.target_cache),
            "init": str(args.init),
        },
        "memory_limits": None if train_result is None else train_result.get("memory_limits"),
        "software_budget": software_budget,
        "samples": probe.samples,
        "peak": probe.peak(),
        "train_result_summary": None
        if train_result is None
        else {
            "status": train_result.get("status"),
            "steps": train_result.get("steps"),
            "seconds_per_step": train_result.get("seconds_per_step"),
            "memory_limits": train_result.get("memory_limits"),
            "microbatch_plan": train_result.get("microbatch_plan"),
            "throughput_flags": train_result.get("throughput_flags"),
            "stage_checkpoint": train_result.get("stage_checkpoint"),
            "latest_checkpoint": train_result.get("latest_checkpoint"),
            "latest_checkpoint_sha256": train_result.get("latest_checkpoint_sha256"),
            "load_memory_peak": train_result.get("load_memory_peak"),
            "software_budget": train_result.get("software_budget"),
        },
        "blocker": blocker,
        "clearance_checks": {
            "required_stage": required_stage,
            "has_required_stage_sample": final_step_sample is not None,
            "has_mlx_allocator_telemetry_at_required_stage": has_mlx_allocator_telemetry,
            "software_budget_check_count": software_budget.get("check_count"),
            "software_budget_within_limit": None
            if not isinstance(software_last, dict)
            else software_last.get("within_budget"),
        },
        "metal_fire_clearance": metal_fire_clearance,
        "clearance_rule": (
            "A Metal launch may consume this receipt only when status=passed, "
            "samples include the required final mem-probe train step with MLX allocator telemetry, and peak fits the composed "
            "one-Metal-fire-at-a-time schedule under the host ceiling."
        ),
    }


def run_mem_probe(args: argparse.Namespace) -> dict[str, Any]:
    probe = LoadPhaseMemoryProbe(emit_log_lines=True)
    probe_args = _mem_probe_args(args)
    train_result: dict[str, Any] | None = None
    blocker: dict[str, Any] | None = None
    status = "passed"
    budget_exc: MemoryBudgetExceeded | None = None
    try:
        train_result = run_mlx_train(probe_args, memory_probe=probe)
        status = str(train_result.get("status", "passed"))
    except Exception as exc:
        status = "blocked" if isinstance(exc, MlxUnavailableError) else "failed"
        if isinstance(exc, MemoryBudgetExceeded):
            budget_exc = exc
        last_sample = probe.samples[-1] if probe.samples else None
        blocker = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "last_sample_stage": None if last_sample is None else last_sample.get("stage"),
            "sample_count": len(probe.samples),
            "software_budget": probe.budget_summary(),
            "boundary": (
                "CPU load-path telemetry may be present before this blocker; "
                "full MLX allocator/three-step telemetry requires MAIN Metal."
            ),
        }
        if isinstance(exc, MlxUnavailableError):
            blocker["verdict_scope"] = "ENVIRONMENT: local sandbox MLX/Metal unavailable"
        elif isinstance(exc, MemoryLimitConfigurationError):
            blocker["verdict_scope"] = "INSTANCE: MLX software memory-budget configuration"
        elif isinstance(exc, MemoryBudgetExceeded):
            blocker["verdict_scope"] = "INSTANCE: software stage/step memory budget"
        else:
            blocker["verdict_scope"] = "INSTANCE: mem-probe execution"
    receipt = _build_mem_probe_receipt(
        args=args,
        probe_args=probe_args,
        probe=probe,
        status=status,
        train_result=train_result,
        blocker=blocker,
    )
    receipt_path = args.run_dir / "mem_probe_receipt.json"
    write_json_atomic(receipt_path, receipt)
    result = {
        "schema": "ddm_mx1_mem_probe.v1",
        "status": status,
        "score_claim": False,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "receipt": receipt,
    }
    if budget_exc is not None:
        raise budget_exc
    return result


def launch_ticket(args: argparse.Namespace, smoke: dict[str, Any] | None, mlx_probe: dict[str, Any]) -> dict[str, Any]:
    base_seconds = float(smoke["seconds_per_step"]) if smoke else None
    horizon = int(args.steps)
    if base_seconds is None:
        estimate = "blocked_local_mlx_probe_no_measured_mlx_step_time"
    else:
        estimate = {
            "local_torch_cpu_seconds_per_step": base_seconds,
            "naive_n32_seconds_at_same_backend": base_seconds * horizon * 32 / max(args.pairs, 1),
            "note": "MAIN must replace with first Metal n32 measured s/step; CPU smoke is not a Metal estimate.",
        }
    n32 = _select_stratified_indices(32, seed=args.seed)
    n120 = _select_stratified_indices(120, seed=args.seed + 1)
    cap_cache = args.target_cache  # GT labels as tokens AND targets
    veh_cache = args.input_cache  # public-wire (tq1c) labels as tokens, GT targets
    if Path(cap_cache).resolve() == Path(veh_cache).resolve():
        # A probe reauthor run with --input-cache pointed at the GT cache collapses
        # ARM-VEH's public-wire discriminator into a duplicate of ARM-CAP, silently
        # defeating the RR3-F1 two-arm requirement (the RR11-F1 incident, replayed
        # 2026-08-08 by a resume-leg reauthor that copied ARM-CAP's mem-probe flags).
        raise ValueError(
            "ticket author refuses: --input-cache (ARM-VEH tokens) == --target-cache "
            "(ARM-CAP tokens); ARM-VEH must consume the public-wire (tq1c) cache as "
            "input or the two n32 arms answer the same question"
        )
    ticket_path = _ticket_path_for_args(args)
    attempt_id = _ticket_attempt_id()

    # RR3-F1: a Row-1 verdict requires TWO arms — ARM-CAP (GT tokens -> GT targets, receiver
    # CAPACITY vs the fp1 flat-paint floor and PR130's external number) and ARM-VEH (public-wire
    # tq1c tokens -> GT targets, composed-vehicle correction reach). A single-arm ticket
    # conflates the two questions, so the bare argv_n32/argv_n120 keys no longer exist.
    def _arm_argv(
        pairs: int,
        seed: int,
        input_cache: Path,
        subdir: str,
        argv_key: str,
        *,
        resume: bool,
    ) -> list[str]:
        run_dir = args.run_dir / subdir
        fire_guard_verdict = run_dir / "fire_guard" / f"{argv_key}.{attempt_id}.json"
        argv = [
            ".venv/bin/python",
            "experiments/ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mlx-train",
            "--device",
            "gpu",
            "--pairs",
            str(pairs),
            "--steps",
            str(horizon),
            "--lr",
            str(args.lr),
            "--ce-fraction",
            str(args.ce_fraction),
            "--softplus-fraction",
            str(args.softplus_fraction),
            "--bits",
            str(args.bits),
            "--seed",
            str(seed),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--eval-every",
            str(args.eval_every),
            "--input-cache",
            str(input_cache),
            "--target-cache",
            str(args.target_cache),
            "--init",
            str(args.init),
            "--run-dir",
            str(run_dir),
            "--out",
            str(run_dir / "result.json"),
            "--fire-guard-verdict",
            str(fire_guard_verdict),
            "--launch-ticket-path",
            str(ticket_path),
            "--fire-argv-key",
            argv_key,
        ]
        if resume:
            argv.extend(["--resume-from", str(run_dir / "mlx.latest.npz")])
        if args.mem_budget_gb is not None:
            argv.extend(["--mem-budget-gb", str(args.mem_budget_gb)])
        if getattr(args, "allow_soft_mem_limit", False):
            argv.append("--allow-soft-mem-limit")
        if str(getattr(args, "train_compute_dtype", "fp32")) != "fp32":
            argv.extend(["--train-compute-dtype", str(args.train_compute_dtype)])
        if bool(getattr(args, "compile_train_loss", False)):
            argv.append("--compile-train-loss")
        if str(getattr(args, "perf_thread_pin", "off")) != "off":
            argv.extend(["--perf-thread-pin", str(args.perf_thread_pin)])
        if str(getattr(args, "microbatch_policy", "auto")) != "auto":
            argv.extend(["--microbatch-policy", str(args.microbatch_policy)])
        if str(getattr(args, "cache_residency", "selected")) != "selected":
            argv.extend(["--cache-residency", str(args.cache_residency)])
        return argv

    base_arm_specs = {
        "argv_n32_arm_cap": (32, args.seed, cap_cache, "launch_arm_cap/n32_metal"),
        "argv_n32_arm_veh": (32, args.seed, veh_cache, "launch_arm_veh/n32_metal"),
        "argv_n120_arm_cap": (120, args.seed + 1, cap_cache, "launch_arm_cap/n120_metal"),
        "argv_n120_arm_veh": (120, args.seed + 1, veh_cache, "launch_arm_veh/n120_metal"),
    }
    arm_specs: dict[str, tuple[int, int, Path, str, bool]] = {}
    for key, (pairs, seed, cache, subdir) in base_arm_specs.items():
        arm_specs[key] = (pairs, seed, cache, subdir, False)
        arm_specs[f"{key}_resume"] = (pairs, seed, cache, subdir, True)

    mem_probe_receipt_paths = {
        key: str(args.run_dir / subdir / ("mem_probe_resume" if resume else "mem_probe") / "mem_probe_receipt.json")
        for key, (_pairs, _seed, _cache, subdir, resume) in arm_specs.items()
    }

    def _mem_probe_command(
        pairs: int,
        seed: int,
        input_cache: Path,
        subdir: str,
        *,
        resume: bool,
    ) -> list[str]:
        run_dir = args.run_dir / subdir
        probe_run_dir = run_dir / ("mem_probe_resume" if resume else "mem_probe")
        command = [
            ".venv/bin/python",
            "experiments/ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mem-probe",
            "--device",
            "gpu",
            "--pairs",
            str(pairs),
            "--mem-probe-steps",
            str(args.mem_probe_steps),
            "--lr",
            str(args.lr),
            "--ce-fraction",
            str(args.ce_fraction),
            "--softplus-fraction",
            str(args.softplus_fraction),
            "--bits",
            str(args.bits),
            "--seed",
            str(seed),
            "--checkpoint-every",
            str(max(1, int(args.mem_probe_steps))),
            "--eval-every",
            "1",
            "--input-cache",
            str(input_cache),
            "--target-cache",
            str(args.target_cache),
            "--init",
            str(args.init),
            "--run-dir",
            str(probe_run_dir),
            "--out",
            str(probe_run_dir / "mem_probe_result.json"),
            "--launch-ticket-path",
            str(ticket_path),
        ]
        if resume:
            command.extend(["--resume-from", str(run_dir / "mlx.latest.npz")])
        if args.mem_budget_gb is not None:
            command.extend(["--mem-budget-gb", str(args.mem_budget_gb)])
        if getattr(args, "allow_soft_mem_limit", False):
            command.append("--allow-soft-mem-limit")
        if str(getattr(args, "train_compute_dtype", "fp32")) != "fp32":
            command.extend(["--train-compute-dtype", str(args.train_compute_dtype)])
        if bool(getattr(args, "compile_train_loss", False)):
            command.append("--compile-train-loss")
        if str(getattr(args, "perf_thread_pin", "off")) != "off":
            command.extend(["--perf-thread-pin", str(args.perf_thread_pin)])
        if str(getattr(args, "microbatch_policy", "auto")) != "auto":
            command.extend(["--microbatch-policy", str(args.microbatch_policy)])
        if str(getattr(args, "cache_residency", "selected")) != "selected":
            command.extend(["--cache-residency", str(args.cache_residency)])
        return command

    mem_probe_commands = {
        key: _mem_probe_command(pairs, seed, cache, subdir, resume=resume)
        for key, (pairs, seed, cache, subdir, resume) in arm_specs.items()
    }
    mem_probe_receipt_path = Path(mem_probe_receipt_paths["argv_n32_arm_cap"])
    mem_probe_command = mem_probe_commands["argv_n32_arm_cap"]
    fire_guard_verdict_paths = {
        key: str(args.run_dir / subdir / "fire_guard" / f"{key}.{attempt_id}.json")
        for key, (_pairs, _seed, _cache, subdir, _resume) in arm_specs.items()
    }
    fire_guard_commands = {
        key: [
            ".venv/bin/python",
            "tools/mx1_fire_guard.py",
            "--ticket",
            str(ticket_path),
            "--argv-key",
            key,
            "--out",
            fire_guard_verdict_paths[key],
        ]
        for key in arm_specs
    }
    raw_fire_argvs = {
        key: _arm_argv(pairs, seed, cache, subdir, key, resume=resume)
        for key, (pairs, seed, cache, subdir, resume) in arm_specs.items()
    }
    safe_run_projections = {
        key: _derive_receipt_safe_run_projection(
            argv_key=key,
            raw_argv=raw_fire_argvs[key],
            receipt_path=Path(mem_probe_receipt_paths[key]),
        )
        for key in arm_specs
    }
    safe_run_status_receipt_paths = {
        key: str(args.run_dir / subdir / "safe_run" / f"{key}.{attempt_id}.status.json")
        for key, (_pairs, _seed, _cache, subdir, _resume) in arm_specs.items()
    }
    safe_run_child_pidfile_paths = {key: f"{path}.child.pid" for key, path in safe_run_status_receipt_paths.items()}
    fire_argvs = {
        key: _wrap_fire_argv(
            raw_fire_argvs[key],
            label=f"ddm_mx1_row1_{_safe_label_token(subdir)}" + ("_resume" if resume else ""),
            projection=safe_run_projections[key],
            status_receipt=Path(safe_run_status_receipt_paths[key]),
            child_pidfile=Path(safe_run_child_pidfile_paths[key]),
        )
        for key, (_pairs, _seed, _cache, subdir, resume) in arm_specs.items()
    }
    detached_done_receipt_names = {key: f"mx1_{_safe_label_token(key)}_{attempt_id}" for key in arm_specs}
    detached_done_receipt_paths = {
        key: f".omx/tmp/codex_runs/{name}.done" for key, name in detached_done_receipt_names.items()
    }
    detached_fire_commands = {
        key: [
            ".venv/bin/python",
            "tools/launch_detached_process.py",
            "--output-dir",
            str(args.run_dir / subdir / "detached" / attempt_id),
            "--cwd",
            str(REPO),
            "--purpose",
            f"MX1 {key} receipt-derived fire attempt {attempt_id}",
            "--authority",
            "local detached execution; downstream artifacts decide authority",
            "--done-receipt",
            detached_done_receipt_names[key],
            "--",
            *fire_argvs[key],
        ]
        for key, (_pairs, _seed, _cache, subdir, _resume) in arm_specs.items()
    }

    return {
        "schema": "ddm_mx1_row1_launch_ticket.v4_software_cap_fire_guarded",
        "score_claim": False,
        "launch_ticket_path": str(ticket_path),
        "ticket_attempt_id": attempt_id,
        "mem_probe_receipt_required": True,
        "mem_probe_receipt_path": str(mem_probe_receipt_path),
        "mem_probe_receipt_paths": mem_probe_receipt_paths,
        "mem_probe_command": mem_probe_command,
        "mem_probe_commands": mem_probe_commands,
        "fire_guard_required": True,
        "fire_guard_tool": "tools/mx1_fire_guard.py",
        "fire_guard_verdict_schema": MX1_FIRE_GUARD_VERDICT_SCHEMA,
        "fire_guard_verdict_paths": fire_guard_verdict_paths,
        "fire_guard_commands": fire_guard_commands,
        "main_fire_sequence": [
            {
                "step": "guard_precheck",
                "command": fire_guard_commands["argv_n32_arm_cap"],
                "expected": "REFUSE until the matching mem_probe_receipt exists and passes",
            },
            {
                "step": "probe",
                "command": mem_probe_command,
                "expected": "writes mem_probe_receipt.json atomically with status=passed",
            },
            {
                "step": "gate",
                "command": fire_guard_commands["argv_n32_arm_cap"],
                "expected": "writes fire_guard_verdict.json with status=passed",
            },
            {
                "step": "fire",
                "command": detached_fire_commands["argv_n32_arm_cap"],
                "expected": (
                    "detached wrapper uses an attempt-unique done receipt, safe_run writes "
                    "an attempt-unique status receipt/child pidfile, and the entrypoint re-runs "
                    "tools.mx1_fire_guard against --launch-ticket-path/--fire-argv-key before MLX setup"
                ),
            },
        ],
        "scheduling": (
            "SEQUENTIAL one-Metal-fire-at-a-time — operator machine OOM 2026-08-06; "
            "ARM-VEH fires only after ARM-CAP completes or a composed measured-peak "
            "projection shows headroom under 116GiB"
        ),
        "fire_protocol": {
            "pre_fire_liveness_proof": (
                "A SUCCESSFUL enumerator is required before every fire. If pgrep returns rc>=2, "
                "run ps axo command; if ps also fails or is denied (rc!=0), REFUSE. Never map "
                "a denied enumerator to 0 candidates."
            ),
            "rr8_f1_refuse_condition": "pgrep rc>=2 AND ps rc!=0",
            "anti_pattern": "no `|| true` around the fallback enumerator; denied ps is not quiescence",
            "scheduling": "SEQUENTIAL one-Metal-fire-at-a-time",
        },
        "safe_run_projection": safe_run_projections["argv_n32_arm_cap"],
        "safe_run_projections": safe_run_projections,
        "safe_run_projection_policy": {
            "schema": SAFE_RUN_RECEIPT_PROJECTION_SCHEMA,
            "sentinel": SAFE_RUN_RECEIPT_SENTINEL,
            "fresh_receipt_required": True,
            "freshness_window_seconds": 6 * 60 * 60,
            "margin_rule": (
                "For each argv key, read that key's mem-probe receipt path. If it is fresh, passed, "
                "guard-validated, and config-matching, measured_peak=max(peak_rss_gib, "
                "peak_mlx_reported_gib, peak_mlx_active_gib+peak_mlx_cache_gib); "
                f"projected_gib=max({SAFE_RUN_PROJECTED_GIB_FLOOR}, "
                f"ceil(measured_peak*{SAFE_RUN_PROJECTION_MULTIPLIER})); "
                f"rss_mb=max({SAFE_RUN_RSS_MB_FLOOR}, ceil(projected_gib*1024)). "
                "Otherwise emit REQUIRES_FRESH_MEM_PROBE so safe_run refuses before launch."
            ),
        },
        "safe_run_status_receipt_paths": safe_run_status_receipt_paths,
        "safe_run_child_pidfile_paths": safe_run_child_pidfile_paths,
        "detached_done_receipt_names": detached_done_receipt_names,
        "detached_done_receipt_paths": detached_done_receipt_paths,
        "detached_fire_commands": detached_fire_commands,
        "resume_protocol": {
            "resume_keys": [key for key in arm_specs if key.endswith("_resume")],
            "resume_checkpoint_rule": "<arm run_dir>/mlx.latest.npz",
            "fresh_mem_probe_required": True,
            "freshness_window_seconds": 6 * 60 * 60,
            "same_chunked_microbatch_config_required": True,
            "guard_binding": (
                "resume argv keys use the same mx1_fire_guard binding as fresh fires; "
                "--resume-from is intentionally outside the guard config comparison set but "
                "the ticket gives resume its own key and mem_probe_resume receipt path."
            ),
        },
        "source_repo_root": SOURCE_REPO_ROOT,
        "source_repo_head": SOURCE_REPO_HEAD,
        "owned_run_root": str(args.run_dir),
        "input_cache": str(args.input_cache),
        "target_cache": str(args.target_cache),
        "init_checkpoint": str(args.init),
        "n32_stratified_indices": n32,
        "n120_stratified_indices": n120,
        "arm_selection_rule": (
            "fire BOTH n32 arms (arm_cap: GT->GT receiver capacity; arm_veh: tq1c->GT composed-"
            "vehicle reach); NO n120 dispatch until the scaled arm is explicitly selected from "
            "the two n32 CPU-torch verdicts; MLX telemetry is research-signal only"
        ),
        "argv_n32_arm_cap": fire_argvs["argv_n32_arm_cap"],
        "argv_n32_arm_veh": fire_argvs["argv_n32_arm_veh"],
        "argv_n120_arm_cap": fire_argvs["argv_n120_arm_cap"],
        "argv_n120_arm_veh": fire_argvs["argv_n120_arm_veh"],
        "argv_n32_arm_cap_resume": fire_argvs["argv_n32_arm_cap_resume"],
        "argv_n32_arm_veh_resume": fire_argvs["argv_n32_arm_veh_resume"],
        "argv_n120_arm_cap_resume": fire_argvs["argv_n120_arm_cap_resume"],
        "argv_n120_arm_veh_resume": fire_argvs["argv_n120_arm_veh_resume"],
        "verdict_protocol": {
            "axis": "[macOS-MLX research-signal] for train telemetry; frozen CPU-torch SegNet through exact R for d_seg; no contest promotion without upstream/evaluate.py on byte-closed archive",
            "compare_against": {
                "fp1_flat_paint_floor_d_seg": 0.008305,
                "pr130_external_d_seg": 0.00029660,
            },
            "selection": "stratified-random n32, then n120; never prefix; n600 only after scorer slot assignment",
        },
        "memory_projection": {
            "renderer_width": 96,
            "blocks": 4,
            "stage": "PR130 stage 08 tail from retained semantic_renderer_w96_b4_qat4_12k checkpoint",
            "configured_horizon_steps": horizon,
            "mem_budget_gb_arg": args.mem_budget_gb,
            "mem_budget_default_policy": "software cap at 35% of available memory at process start when --mem-budget-gb is omitted; mem-probe caps default at min(24GB, default)",
            "enforcement": "software_stage_step_cap",
            "software_budget_rule": "mx.get_active_memory() + max(0, process_rss - start_process_rss) <= budget",
            "wired_limit_policy": "attempt mx.set_wired_limit(min(budget, 35% of system total)) when available",
            "allow_soft_mem_limit": bool(getattr(args, "allow_soft_mem_limit", False)),
            "train_compute_dtype": str(getattr(args, "train_compute_dtype", "fp32")),
            "compile_train_loss": bool(getattr(args, "compile_train_loss", False)),
            "perf_thread_pin": str(getattr(args, "perf_thread_pin", "off")),
            "microbatch_policy": str(getattr(args, "microbatch_policy", "auto")),
            "cache_residency": str(getattr(args, "cache_residency", "selected")),
            "microbatch_hygiene": str(getattr(args, "microbatch_hygiene", "per-chunk")),
            "microbatch_chunk_cache": bool(getattr(args, "microbatch_chunk_cache", False)),
            "verdict_batch_size": int(getattr(args, "verdict_batch_size", 32)),
            "float_warmup_steps": int(getattr(args, "float_warmup_steps", 0)),
            "microbatch_auto_anchor": WC2_AUTO_MICROBATCH_ANCHOR,
            "lr": args.lr,
            "ce_fraction": args.ce_fraction,
            "softplus_fraction": args.softplus_fraction,
            "checkpoint_size_source_bytes": args.init.stat().st_size if args.init.exists() else None,
            "label_cache_inputs_bytes": args.input_cache.stat().st_size if args.input_cache.exists() else None,
            "label_cache_targets_bytes": args.target_cache.stat().st_size if args.target_cache.exists() else None,
        },
        "wall_clock_estimate": estimate,
        "local_mlx_probe": mlx_probe,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _torch_verdict_dseg(path: Path) -> tuple[float, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = payload.get("torch_verdict", payload)
    if verdict.get("schema") != "ddm_mx1_torch_verdict.v1" or verdict.get("status") != "passed":
        raise ValueError(f"CPU verdict {path} is missing a passed ddm_mx1_torch_verdict.v1")
    return float(verdict["aggregate_d_seg"]), verdict


def run_m1_schedule_selection(args: argparse.Namespace) -> dict[str, Any]:
    """Select the post-calibration schedule from two same-object CPU facets."""

    if args.launch_ticket_path is None or args.resume_from is None:
        raise ValueError("m1-schedule-select requires --launch-ticket-path and --resume-from")
    ticket = json.loads(args.launch_ticket_path.read_text(encoding="utf-8"))
    predicate = dict((ticket.get("stop_policy") or {}).get("predicate") or {})
    baseline_dseg, baseline = _torch_verdict_dseg(args.init)
    candidate_dseg, candidate = _torch_verdict_dseg(args.resume_from)
    if baseline.get("pair_ids") != candidate.get("pair_ids"):
        raise ValueError("same-object schedule facets have different pair IDs")
    required_improvement = float(predicate["one_sample_flip_S"]) / 100.0
    improvement = baseline_dseg - candidate_dseg
    passed = improvement >= required_improvement
    return {
        "schema": M1_SCHEDULE_SELECTION_SCHEMA,
        "status": "passed" if passed else "refused",
        "selected_schedule": "monotone_clamped_cosine" if passed else "rollback_no_resume",
        "axis": "[macOS-CPU advisory torch upstream SegNet]",
        "score_claim": False,
        "verdict_scope": f"n{len(candidate['pair_ids'])} same-object schedule calibration",
        "baseline_result": str(args.init),
        "candidate_result": str(args.resume_from),
        "baseline_d_seg": baseline_dseg,
        "candidate_d_seg": candidate_dseg,
        "improvement_d_seg": improvement,
        "required_improvement_d_seg": required_improvement,
        "decision_rule": (
            "admit the ticketed monotone cosine only when the step-250 same-object CPU facet "
            "improves by at least one n120 Seg lattice flip; otherwise rollback and do not resume"
        ),
        "generated_utc": _utc_now_iso(),
    }


def run_m1_controlled_train(args: argparse.Namespace) -> dict[str, Any]:
    """Run one ticketed safe_run child and survive it to receipt a wall cap."""

    if args.launch_ticket_path is None or not args.fire_argv_key:
        raise ValueError("controlled-train requires --launch-ticket-path and --fire-argv-key")
    ticket = json.loads(args.launch_ticket_path.read_text(encoding="utf-8"))
    routes = dict(((ticket.get("stop_policy") or {}).get("executor") or {}).get("controller_routes") or {})
    if args.fire_argv_key not in routes:
        raise ValueError(f"ticket has no controlled-train route for {args.fire_argv_key}")
    route = dict(routes[args.fire_argv_key])
    child_key = str(route["child_argv_key"])
    child_argv = list(ticket[child_key])
    if child_argv[:2] != [".venv/bin/python", "tools/safe_run.py"]:
        raise ValueError("controlled-train child must be the ticketed governed safe_run argv")
    completed = subprocess.run(child_argv, cwd=REPO, check=False)
    status_path = Path(route["safe_run_status_receipt_path"])
    if not status_path.exists():
        raise ValueError(f"safe_run child returned without durable status receipt: {status_path}")
    safe_status = json.loads(status_path.read_text(encoding="utf-8"))
    terminal_path = Path(route["terminal_receipt_path"])
    if completed.returncode == 124:
        child_policy = _load_m1_executor_policy(
            argparse.Namespace(
                launch_ticket_path=args.launch_ticket_path,
                fire_argv_key=child_key,
            )
        )
        if child_policy is None:
            raise ValueError("wall-cap child key is not bound to the M1 executor")
        journal_path = Path(child_policy["executor"]["journal_path"])
        rows = _read_active_m1_eval_rows(journal_path)
        last_step = int(rows[-1]["step"]) if rows else 0
        safety_bound = float(child_policy["executor"]["safety_bound_steps_by_key"][child_key])
        decision = evaluate_staircase_aware_stop(
            rows,
            child_policy["trajectory_config"],
            child_policy["staircase_config"],
            safety_bound_compute=max(safety_bound, float(last_step)),
            boundary_kind="wall_clock_seconds",
        )
        decision_path = Path(child_policy["executor"]["decision_path"])
        _append_jsonl_durable(
            decision_path,
            {
                "schema": M1_STOP_DECISION_SCHEMA,
                "step": last_step,
                "generated_utc": _utc_now_iso(),
                "ticket_path": str(args.launch_ticket_path),
                "ticket_sha256": _sha256_file(args.launch_ticket_path),
                "journal_path": str(journal_path),
                "journal_sha256": _sha256_file(journal_path) if journal_path.exists() else None,
                "decision": decision.to_payload(),
                "boundary_status_receipt": str(status_path),
            },
        )
        cap_receipt = build_cap_stop_receipt(
            stop_reason="cap_bound",
            steps_run=last_step,
            cap=None,
            still_descending=decision.action != "STOP_CONVERGED",
            bound_kind="wall_clock_seconds",
            bound_value=float(safe_status["timeout_s"]),
            observed_value=max(float(safe_status["elapsed_s"]), float(safe_status["timeout_s"])),
        ).to_payload()
        write_json_atomic(
            terminal_path,
            {
                "schema": M1_TERMINAL_RECEIPT_SCHEMA,
                "status": "halted",
                "terminal_mode": "wall_clock_cap",
                "action": "QUEUE_RESUME",
                "step": last_step,
                "decision": decision.to_payload(),
                "cap_stop_receipt": cap_receipt,
                "safe_run_status_receipt": str(status_path),
                "safe_run_status_sha256": _sha256_file(status_path),
                "resume_argv_key": child_policy["executor"]["resume_argv_key"],
                "resume_argv": ticket[child_policy["executor"]["resume_argv_key"]],
                "same_object_cpu_selection": child_policy["executor"]["same_object_cpu_selection"],
                "generated_utc": _utc_now_iso(),
            },
        )
    elif not terminal_path.exists():
        write_json_atomic(
            terminal_path,
            {
                "schema": M1_TERMINAL_RECEIPT_SCHEMA,
                "status": "failed",
                "terminal_mode": "child_exit_without_training_terminal",
                "action": "ROLLBACK_OR_RETREAT",
                "child_exit": completed.returncode,
                "safe_run_status_receipt": str(status_path),
                "safe_run_status_sha256": _sha256_file(status_path),
                "generated_utc": _utc_now_iso(),
            },
        )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    return {
        "schema": "ddm_m1_controlled_train.v1",
        "status": "passed" if terminal.get("status") == "halted" else "failed",
        "child_argv_key": child_key,
        "child_exit": completed.returncode,
        "safe_run_status_receipt": str(status_path),
        "terminal_receipt": str(terminal_path),
        "terminal": terminal,
    }


def _ticket_path_for_args(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "launch_ticket_path", None)
    if explicit is not None:
        return Path(explicit)
    return args.run_dir / "launch_ticket_v4_fire_guarded.json"


def _canonical_existing_or_repo_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    resolved = (REPO / resolved).resolve() if not resolved.is_absolute() else resolved.resolve()
    return str(resolved)


def _assert_gpu_fire_guard(args: argparse.Namespace) -> None:
    verdict_path = args.fire_guard_verdict
    ticket_path = args.launch_ticket_path
    argv_key = args.fire_argv_key
    if verdict_path is None or ticket_path is None or not argv_key:
        print(
            "[mx1-fire-guard] REFUSED: gpu mlx-train requires --fire-guard-verdict, "
            "--launch-ticket-path, and --fire-argv-key",
            file=sys.stderr,
        )
        raise SystemExit(9)
    try:
        # Detached launches put the script dir (experiments/) on sys.path, not the
        # repo root — anchor the guard import to this file's parent so the
        # in-process re-evaluation works from any cwd.
        _repo_root = str(Path(__file__).resolve().parent.parent)
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from tools.mx1_fire_guard import evaluate_guard

        evaluated = evaluate_guard(ticket_path, argv_key)
    except Exception as exc:
        print(
            f"[mx1-fire-guard] REFUSED: in-process guard evaluation failed for "
            f"{ticket_path} {argv_key}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(9) from exc
    if evaluated.get("schema") != MX1_FIRE_GUARD_VERDICT_SCHEMA or evaluated.get("status") != "passed":
        print(
            f"[mx1-fire-guard] REFUSED: in-process guard failed: "
            f"status={evaluated.get('status')!r} reason={evaluated.get('reason_code')!r}",
            file=sys.stderr,
        )
        raise SystemExit(9)
    expected_verdict_path = (evaluated.get("fire_config") or {}).get("fire_guard_verdict")
    if _canonical_existing_or_repo_path(expected_verdict_path) != _canonical_existing_or_repo_path(verdict_path):
        print(
            "[mx1-fire-guard] REFUSED: --fire-guard-verdict does not match ticket fire argv "
            f"expected={expected_verdict_path!r} actual={str(verdict_path)!r}",
            file=sys.stderr,
        )
        raise SystemExit(9)
    try:
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[mx1-fire-guard] REFUSED: could not read guard verdict at {verdict_path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(9) from exc
    if (
        verdict.get("schema") != MX1_FIRE_GUARD_VERDICT_SCHEMA
        or verdict.get("status") != "passed"
        or verdict.get("reason_code") != "fire_guard_passed"
    ):
        print(
            f"[mx1-fire-guard] REFUSED: guard verdict failed or malformed at {verdict_path}: "
            f"status={verdict.get('status')!r} reason={verdict.get('reason_code')!r}",
            file=sys.stderr,
        )
        raise SystemExit(9)
    for key in ("ticket_path", "receipt_path"):
        if _canonical_existing_or_repo_path(verdict.get(key)) != _canonical_existing_or_repo_path(evaluated.get(key)):
            print(
                f"[mx1-fire-guard] REFUSED: guard verdict {key} does not match fresh evaluation",
                file=sys.stderr,
            )
            raise SystemExit(9)
    if verdict.get("argv_key") != argv_key:
        print(
            "[mx1-fire-guard] REFUSED: guard verdict argv_key does not match current fire argv",
            file=sys.stderr,
        )
        raise SystemExit(9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "probe",
            "torch-smoke",
            "torch-verdict",
            "torch-facets",
            "coreml-segnet-parity",
            "mlx-parity",
            "mlx-train",
            "mem-probe",
            "controlled-train",
            "m1-schedule-select",
        ],
        default="probe",
    )
    parser.add_argument("--input-cache", type=Path, default=DEFAULT_INPUT_CACHE)
    parser.add_argument("--target-cache", type=Path, default=DEFAULT_TARGET_CACHE)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--run-dir", type=Path, default=SSD_ROOT)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--ce-fraction", type=float, default=0.0)
    parser.add_argument("--softplus-fraction", type=float, default=-999.0)
    parser.add_argument("--train-exact-path", action="store_true")
    parser.add_argument("--scorer", choices=["upstream", "proxy"], default="upstream")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--float-warmup-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--microbatch-pairs", type=int, default=0)
    parser.add_argument("--microbatch-policy", choices=MICROBATCH_POLICIES, default="auto")
    parser.add_argument("--cache-residency", choices=CACHE_RESIDENCY_MODES, default="selected")
    parser.add_argument("--train-compute-dtype", choices=TRAIN_COMPUTE_DTYPES, default="fp32")
    parser.add_argument("--compile-train-loss", action="store_true")
    parser.add_argument("--perf-thread-pin", choices=THREAD_PIN_MODES, default="off")
    parser.add_argument("--microbatch-hygiene", choices=MICROBATCH_HYGIENE_MODES, default="per-chunk")
    parser.add_argument("--microbatch-chunk-cache", action="store_true")
    parser.add_argument("--verdict-batch-size", type=int, default=32)
    parser.add_argument("--mem-budget-gb", type=float)
    parser.add_argument("--mem-probe-steps", type=int, default=3)
    parser.add_argument("--allow-soft-mem-limit", action="store_true")
    parser.add_argument("--fire-guard-verdict", type=Path)
    parser.add_argument("--launch-ticket-path", type=Path)
    parser.add_argument("--fire-argv-key")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--facet-checkpoint-dir", type=Path, default=MX1T_DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--facet-out-dir", type=Path, default=MX1T_DEFAULT_OUT_DIR)
    parser.add_argument("--facet-tail-average-ks", default="2,4,8")
    parser.add_argument("--facet-steps", default="")
    parser.add_argument("--facet-anchor-step", type=int, default=1500)
    parser.add_argument("--facet-anchor-d-seg", type=float, default=MX1H_STEP1500_AUTHORITY_D_SEG)
    parser.add_argument("--facet-anchor-tolerance", type=float, default=1e-12)
    parser.add_argument("--coreml-compute-units", default="CPU_AND_NE")
    parser.add_argument("--coreml-parity-max-argmax-diff", type=int, default=0)
    parser.add_argument("--out", type=Path, default=SSD_ROOT / "mx1_driver_result.json")
    args = parser.parse_args()
    if args.mem_probe_steps <= 0:
        parser.error("--mem-probe-steps must be positive")
    if args.microbatch_pairs < 0:
        parser.error("--microbatch-pairs must be non-negative")
    if args.verdict_batch_size <= 0:
        parser.error("--verdict-batch-size must be positive")
    if args.mode == "m1-schedule-select":
        selection = run_m1_schedule_selection(args)
        write_json_atomic(args.out, selection)
        print(json.dumps(selection, indent=2, sort_keys=True, default=str))
        if selection["status"] != "passed":
            raise SystemExit(3)
        return
    if args.mode == "controlled-train":
        controlled = run_m1_controlled_train(args)
        write_json_atomic(args.out, controlled)
        print(json.dumps(controlled, indent=2, sort_keys=True, default=str))
        if controlled["status"] != "passed":
            raise SystemExit(3)
        return
    if args.mode in {"mlx-train", "torch-smoke"}:
        assert_governed_admission(f"ddm_mx1_pr130_semantic_renderer:{args.mode}")
    if args.mode == "mlx-train" and str(args.device).lower() == "gpu":
        _assert_gpu_fire_guard(args)

    result: dict[str, Any] = {
        "schema": "ddm_mx1_pr130_semantic_renderer_driver.v1",
        "mode": args.mode,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "score_claim": False,
        "source_repo_head": SOURCE_REPO_HEAD,
        "source_repo_root": SOURCE_REPO_ROOT,
    }
    if args.mode in {"torch-verdict", "torch-facets", "coreml-segnet-parity"}:
        mlx_probe = {
            "status": "not_run",
            "reason": f"{args.mode} is CPU-only and does not import or probe MLX/Metal",
        }
    else:
        mlx_probe = mlx_device_probe(device=args.device)
    result["mlx_probe"] = mlx_probe
    smoke: dict[str, Any] | None = None
    if args.mode == "torch-smoke":
        smoke = run_torch_smoke(args)
        result["torch_smoke"] = smoke
    elif args.mode == "torch-verdict":
        result["torch_verdict"] = run_torch_verdict(args)
        result["status"] = result["torch_verdict"]["status"]
    elif args.mode == "torch-facets":
        result["torch_facets"] = run_torch_facets(args)
        result["status"] = result["torch_facets"]["status"]
    elif args.mode == "coreml-segnet-parity":
        result["coreml_segnet_parity"] = run_coreml_segnet_parity(args)
        result["status"] = result["coreml_segnet_parity"]["status"]
    elif args.mode == "mlx-parity":
        if mlx_probe["status"] == "blocked":
            result["status"] = "blocked"
            result["blocker"] = "local MLX runtime unavailable; run parity on MAIN Metal host"
        else:
            result["mlx_parity"] = run_mlx_parity(args)
            result["status"] = result["mlx_parity"]["status"]
    elif args.mode == "mlx-train":
        if mlx_probe["status"] == "blocked":
            result["status"] = "blocked"
            result["blocker"] = "local MLX runtime unavailable; run the launch-ticket argv on MAIN Metal host"
        else:
            result["mlx_train"] = run_mlx_train(args)
            result["status"] = result["mlx_train"]["status"]
    elif args.mode == "mem-probe":
        result["mem_probe"] = run_mem_probe(args)
        result["status"] = result["mem_probe"]["status"]
    if args.mode == "probe":
        # RR11-F1: the canonical launch ticket is an immutable fire order. ONLY the
        # ticket-authoring mode may write it; mem-probe/mlx-train/parity/smoke runs
        # must never regenerate it (a probe once collapsed ARM-VEH's tq1c cache
        # discriminator to gt by rewriting the ticket it was supporting).
        result["launch_ticket"] = launch_ticket(args, smoke, mlx_probe)
        write_json(_ticket_path_for_args(args), result["launch_ticket"])
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.mode in {"mlx-parity", "mlx-train"} and mlx_probe["status"] == "blocked":
        raise SystemExit(2)
    if args.mode == "mem-probe" and result.get("status") != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
