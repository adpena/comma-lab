#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Transfer-test exact frozen-SegNet CPU forward arms on real receiver bytes.

This local-only probe extends :mod:`probe_segnet_exact_forward`.  A real-pair
canary selects an argmax-exact strategy/thread arm, then a resumable matched
alternating-order run compares it with eager NCHW.  It never invokes the
contest evaluator, a provider, or a live trainer.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = Path(__file__).resolve()
DEPENDENCY_PATH = REPO_ROOT / "tools/probe_segnet_exact_forward.py"
for _import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from tac.witness_dsl.segnet_exact_forward_transfer_policy import (  # noqa: E402
    SegNetExactForwardTransferPolicy,
)

SCHEMA = "frozen_segnet_exact_forward_transfer_probe_v4"
CHECKPOINT_SCHEMA = "frozen_segnet_exact_forward_transfer_checkpoint_v4"
STRATEGIES = SegNetExactForwardTransferPolicy.supported_strategies()
# CONFIG-SEALED by the operator's mission: n600 is the transfer-verdict surface.
VERDICT_PAIR_CARDINALITY = 600
# ASSUMED false-positive budget for the exact one-sided matched sign test.
MATCHED_SIGN_ALPHA = 0.01
# DERIVED: two SHA-256 hex digests plus timing/JSON framing per pair.  The
# multiplier is a conservative upper envelope, not a measured artifact size.
CHECKPOINT_BYTES_PER_PAIR_UPPER_BOUND = 512
# ASSUMED operating reserve, explicit so it cannot masquerade as a measured law.
STORAGE_METADATA_RESERVE_BYTES = 8 * 1024 * 1024


def _load_dependency() -> Any:
    spec = importlib.util.spec_from_file_location("probe_segnet_exact_forward_dependency", DEPENDENCY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import dependency {DEPENDENCY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_dependency()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return base._sha256_file(path)


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_files(paths: Sequence[Path], *, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = []
    for path in sorted({item.resolve() for item in paths if item.is_file()}):
        relative = path.relative_to(root.resolve()).as_posix()
        file_sha = sha256_file(path)
        size = path.stat().st_size
        digest.update(relative.encode())
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_sha))
        rows.append({"path": relative, "bytes": size, "sha256": file_sha})
    return {"files": rows, "tree_sha256": digest.hexdigest()}


def torch_runtime_custody(torch_module: Any) -> dict[str, Any]:
    """Bind the labeled Torch build to its loaded Python and native binaries."""

    root = Path(torch_module.__file__).resolve().parent
    native_paths = list(root.rglob("*.so")) + list(root.rglob("*.dylib"))
    python_entry = Path(torch_module.__file__).resolve()
    return {
        "package_root": str(root),
        "version": str(torch_module.__version__),
        "git_version": str(getattr(getattr(torch_module, "version", None), "git_version", "unknown")),
        "python_entry_sha256": sha256_file(python_entry),
        "python_source_tree": base._python_source_tree_custody(torch_module),
        "native_binaries": _sha256_files(native_paths, root=root),
    }


def cpu_host_identity() -> dict[str, Any]:
    """Fingerprint the host/CPU that defines a local-only timing verdict."""

    cpu_model = platform.processor()
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                continue
            if value:
                cpu_model = value
                break
    return {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "byteorder": sys.byteorder,
    }


def relevant_thread_environment() -> dict[str, str | None]:
    prefixes = ("OMP", "MKL", "OPENBLAS", "VECLIB", "NUMEXPR", "TORCH", "PYTORCH", "CUDA")
    known = {
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "CUDA_VISIBLE_DEVICES",
    }
    names = known | {key for key in os.environ if key.startswith(prefixes)}
    return {key: os.environ.get(key) for key in sorted(names)}


def scorer_and_build_custody(torch_module: Any) -> dict[str, Any]:
    return {
        "scorer_runtime": base._scorer_runtime_custody(),
        "torch_runtime": torch_runtime_custody(torch_module),
        "thread_environment": relevant_thread_environment(),
        "cpu_host_identity": cpu_host_identity(),
    }


def measurement_custody(raw: Path, weights: Path, torch_module: Any) -> dict[str, Any]:
    return {
        "raw_path": str(raw),
        "raw_bytes": raw.stat().st_size,
        "raw_sha256": sha256_file(raw),
        "weights_path": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": sha256_file(weights),
        "tool_sha256": sha256_file(TOOL_PATH),
        "dependency_sha256": sha256_file(DEPENDENCY_PATH),
        "build": scorer_and_build_custody(torch_module),
    }


@contextlib.contextmanager
def exclusive_run_lock(out: Path):
    """Hold a nonblocking process lock for the entire checkpoint namespace."""

    out = base.validate_durable_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_namespace(out)
    lock_path = checkpoint_dir.parent / f".{checkpoint_dir.name}.run.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another transfer probe owns {lock_path}") from exc
        owner = {
            "pid": os.getpid(),
            "host": platform.node(),
            "started_at_utc": utc_now(),
            "lock_path": str(lock_path),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield owner
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def checkpoint_namespace(out: Path) -> Path:
    """Return the unique checkpoint namespace guarded by the run lock."""

    return out.parent / f"{out.stem}_checkpoints"


def discover_thread_topology(
    *,
    torch_default: int,
    os_logical: int | None,
    psutil_logical: int | None,
    psutil_physical: int | None,
) -> dict[str, Any]:
    """Derive a finite topology-bound tournament without choosing an optimum."""

    observed = {
        "torch_default": torch_default,
        "os_logical": os_logical,
        "psutil_logical": psutil_logical,
        "psutil_physical": psutil_physical,
    }
    positive = [int(value) for value in observed.values() if value is not None and int(value) > 0]
    if not positive:
        raise RuntimeError("no positive CPU/thread topology observation")
    ceiling = min(positive)
    # The exact integer optimum cannot be inferred from topology alone.  The
    # finite exhaustive set is therefore derived from the CPU-cardinality law
    # 1 <= k <= ceiling; the measured real-forward tournament chooses k*.
    candidates = list(range(1, ceiling + 1))
    return {
        "label": "DERIVED",
        "observed": observed,
        "effective_ceiling": ceiling,
        "candidate_threads": candidates,
        "derivation": "every integer worker cardinality in [1, min positive torch/os/psutil topology]",
    }


def runtime_thread_topology(torch_module: Any) -> dict[str, Any]:
    physical = logical = None
    try:
        import psutil

        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return discover_thread_topology(
        torch_default=int(torch_module.get_num_threads()),
        os_logical=os.cpu_count(),
        psutil_logical=logical,
        psutil_physical=physical,
    )


def derive_candidate_arms(topology: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"strategy": strategy, "threads": int(threads)}
        for strategy in STRATEGIES
        for threads in topology["candidate_threads"]
    ]


def derive_canary_count(topology: Mapping[str, Any], *, n_pairs: int) -> int:
    """Return the typed DSL's explicitly heuristic real-pair screen size."""

    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    cardinality = len(topology["candidate_threads"])
    if cardinality < 1:
        raise ValueError("candidate thread set must be non-empty")
    return min(n_pairs, max(1, math.ceil(math.log2(cardinality + 1))))


def compile_policy_contracts(topology: Mapping[str, Any], model_input: Any) -> dict[str, dict[str, object]]:
    observed = topology["observed"]
    physical = observed.get("psutil_physical") or observed.get("psutil_logical") or observed.get("os_logical")
    shape = tuple(int(value) for value in model_input.shape)
    if len(shape) != 4:
        raise RuntimeError(f"expected NCHW forward shape, got {shape}")
    contracts: dict[str, dict[str, object]] = {}
    for strategy in STRATEGIES:
        policy = SegNetExactForwardTransferPolicy(
            physical_core_count=int(physical),
            torch_default_intraop_threads=int(observed["torch_default"]),
            batch_size=shape[0],
            channels=shape[1],
            height=shape[2],
            width=shape[3],
            strategy=strategy,
            verdict_pair_cardinality=VERDICT_PAIR_CARDINALITY,
            matched_sign_alpha=MATCHED_SIGN_ALPHA,
        )
        contract = policy.compile_measurement_contract()
        if contract["candidate_threads"] != list(topology["candidate_threads"]):
            raise RuntimeError("typed DSL and runtime topology candidate sets differ")
        contracts[strategy] = contract
    return contracts


def forward_signature(
    model: Any,
    model_input: Any,
    torch_module: Any,
    weights: Path,
    *,
    topology: Mapping[str, Any],
    runtime_custody: Mapping[str, Any],
) -> dict[str, Any]:
    config = getattr(torch_module, "__config__", None)
    config_text = config.show() if config is not None and hasattr(config, "show") else "unavailable"
    return {
        "input_shape": list(model_input.shape),
        "input_dtype": str(model_input.dtype),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "model_weights_sha256": sha256_file(weights),
        "torch_version": str(torch_module.__version__),
        "torch_git_version": str(getattr(getattr(torch_module, "version", None), "git_version", "unknown")),
        "torch_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": str(Path(sys.executable).resolve()),
        "thread_topology": dict(topology),
        "runtime_custody": dict(runtime_custody),
        "dependency_sha256": sha256_file(DEPENDENCY_PATH),
    }


def selection_key(
    signature: Mapping[str, Any],
    arms: Sequence[Mapping[str, Any]],
    *,
    policy_contracts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    payload = {
        "forward_signature": dict(signature),
        "candidate_arms": [dict(arm) for arm in arms],
        "policy_contracts": {key: dict(value) for key, value in (policy_contracts or {}).items()},
    }
    return {"payload": payload, "sha256": sha256_json(payload)}


def select_canary_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if int(row["argmax_flip_count"]) == 0]
    if not eligible:
        raise RuntimeError("no zero-argmax-flip canary arm")
    return dict(
        min(
            eligible,
            key=lambda row: (
                float(row["forward_ms_median"]),
                str(row["strategy"]),
                int(row["threads"]),
            ),
        )
    )


def admit_selected_arm(
    *,
    selected: Mapping[str, Any],
    n_pairs: int,
    verdict_pair_cardinality: int,
    baseline_median_ms: float,
    selected_median_ms: float,
    flip_count: int,
    reference_sha256: str,
    candidate_sha256: str | None,
    matched_sign_pvalue: float,
    matched_sign_alpha: float,
) -> bool:
    return (
        n_pairs == verdict_pair_cardinality
        and int(flip_count) == 0
        and reference_sha256 == candidate_sha256
        and selected_median_ms < baseline_median_ms
        and matched_sign_pvalue <= matched_sign_alpha
        and not (
            selected["strategy"] == "eager_nchw_autograd"
            and int(selected["threads"]) == int(selected["baseline_threads"])
        )
    )


def storage_preflight(checkpoint_dir: Path, *, n_pairs: int, free_bytes: int | None = None) -> dict[str, Any]:
    """Fail closed on the small durable-checkpoint budget.

    Hard argmax tensors never land on disk.  Resume state stores only timings,
    per-pair digests, and flip counts; the final raw sequence digest is rebuilt
    by a deterministic reference replay.
    """

    checkpoint_dir = base.validate_durable_output(checkpoint_dir)
    required = n_pairs * CHECKPOINT_BYTES_PER_PAIR_UPPER_BOUND + STORAGE_METADATA_RESERVE_BYTES
    probe_path = checkpoint_dir
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    available = int(
        os.statvfs(probe_path).f_bavail * os.statvfs(probe_path).f_frsize if free_bytes is None else free_bytes
    )
    result = {
        "label": "DERIVED",
        "checkpoint_dir": str(checkpoint_dir),
        "argmax_bulk_bytes": 0,
        "checkpoint_bytes_per_pair_upper_bound": CHECKPOINT_BYTES_PER_PAIR_UPPER_BOUND,
        "required_free_bytes": required,
        "available_free_bytes": available,
        "passed": available >= required,
        "policy": {
            "metadata_reserve_bytes": STORAGE_METADATA_RESERVE_BYTES,
            "provenance": (
                "per-pair checkpoint envelope DERIVED from two digests plus timing/JSON fields; "
                "metadata reserve ASSUMED operating margin"
            ),
            "disk_hygiene": "no tensor/cache scratch is created; success cleanup is not applicable",
        },
    }
    if not result["passed"]:
        raise RuntimeError(f"SSD storage preflight failed: required={required} available={available}")
    return result


def build_run_fingerprint(
    *,
    raw: Path,
    weights: Path,
    n_pairs: int,
    checkpoint_interval: int,
    canary_indices: Sequence[int],
    selection: Mapping[str, Any],
    start_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    custody = dict(start_custody or {})
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "raw_path": str(raw),
        "raw_bytes": int(custody.get("raw_bytes", raw.stat().st_size)),
        "raw_sha256": str(custody.get("raw_sha256") or sha256_file(raw)),
        "weights_sha256": str(custody.get("weights_sha256") or sha256_file(weights)),
        "tool_sha256": str(custody.get("tool_sha256") or sha256_file(TOOL_PATH)),
        "dependency_sha256": str(custody.get("dependency_sha256") or sha256_file(DEPENDENCY_PATH)),
        "n_pairs": n_pairs,
        "checkpoint_interval": checkpoint_interval,
        "canary_indices": list(canary_indices),
        "selection_key_sha256": selection["sha256"],
    }
    return {"payload": payload, "sha256": sha256_json(payload)}


def validate_resume_state(state: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> int:
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("resume checkpoint schema mismatch")
    if state.get("fingerprint_sha256") != fingerprint.get("sha256"):
        raise RuntimeError("resume fingerprint mismatch; refusing mixed custody")
    completed = int(state.get("completed_pairs", -1))
    n_pairs = int(fingerprint["payload"]["n_pairs"])
    if completed < 0 or completed > n_pairs:
        raise RuntimeError("resume completed-pair count is invalid")
    if len(state.get("baseline_ms", [])) != completed or len(state.get("selected_ms", [])) != completed:
        raise RuntimeError("resume timing vector length mismatch")
    for field in ("reference_pair_sha256", "candidate_pair_sha256", "pair_flip_counts"):
        if len(state.get(field, [])) != completed:
            raise RuntimeError(f"resume {field} length mismatch")
    for field in ("reference_pair_sha256", "candidate_pair_sha256"):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in state[field]
        ):
            raise RuntimeError(f"resume {field} contains an invalid SHA-256")
    flips = state["pair_flip_counts"]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in flips):
        raise RuntimeError("resume pair_flip_counts contains an invalid count")
    for index, (left, right, flip_count) in enumerate(
        zip(state["reference_pair_sha256"], state["candidate_pair_sha256"], flips, strict=True)
    ):
        if (left == right) != (flip_count == 0):
            raise RuntimeError(f"resume pair digest/flip inconsistency at pair {index}")
    for field in ("baseline_ms", "selected_ms"):
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in state[field]):
            raise RuntimeError(f"resume {field} contains invalid timing")
    return completed


def summarize_ms(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("empty timing vector")
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples_ms": list(values),
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "p05_ms": float(np.percentile(array, 5)),
        "p95_ms": float(np.percentile(array, 95)),
        "count": len(values),
    }


def matched_sign_test(baseline_ms: Sequence[float], selected_ms: Sequence[float]) -> dict[str, Any]:
    if len(baseline_ms) != len(selected_ms) or not baseline_ms:
        raise ValueError("matched timing vectors must have equal positive length")
    gaps = [float(control) - float(candidate) for control, candidate in zip(baseline_ms, selected_ms, strict=True)]
    wins = sum(gap > 0.0 for gap in gaps)
    losses = sum(gap < 0.0 for gap in gaps)
    non_ties = wins + losses
    pvalue = 1.0
    if non_ties:
        pvalue = sum(math.comb(non_ties, k) for k in range(wins, non_ties + 1)) / (2**non_ties)
    gap_array = np.asarray(gaps, dtype=np.float64)
    gap_median = float(np.median(gap_array))
    mad = float(np.median(np.abs(gap_array - gap_median)))
    return {
        "label": "DERIVED_FROM_MATCHED_MEASUREMENTS",
        "wins": wins,
        "losses": losses,
        "ties": len(gaps) - non_ties,
        "one_sided_exact_binomial_pvalue": pvalue,
        "median_gap_ms": gap_median,
        "median_absolute_deviation_ms": mad,
        "alpha": MATCHED_SIGN_ALPHA,
    }


def compare_argmax_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.ascontiguousarray(reference.astype(np.uint8, copy=False))
    right = np.ascontiguousarray(candidate.astype(np.uint8, copy=False))
    if left.shape != right.shape:
        raise RuntimeError("argmax shape mismatch")
    return {
        "reference_pair_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "candidate_pair_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
        "argmax_flip_count": int(np.count_nonzero(left != right)),
    }


def _strategy_model(base_model: Any, strategy: str) -> Any:
    import copy

    import torch

    model = copy.deepcopy(base_model)
    if strategy == "eager_channels_last_autograd":
        model = model.to(memory_format=torch.channels_last)
    elif strategy != "eager_nchw_autograd":
        raise ValueError(f"unknown strategy: {strategy}")
    return model


def _forward(model: Any, model_input: Any, *, strategy: str, threads: int) -> tuple[Any, float]:
    import torch

    torch.set_num_threads(threads)
    sample = model_input.detach().clone()
    sample.requires_grad_(True)
    start = time.perf_counter_ns()
    if strategy == "eager_channels_last_autograd":
        sample = sample.to(memory_format=torch.channels_last)
    logits = model(sample)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    if not logits.requires_grad or logits.grad_fn is None:
        raise RuntimeError("autograd strategy failed to preserve the input-gradient graph")
    return logits.detach(), elapsed_ms


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    base.atomic_write_json(path, dict(payload))


def _checkpoint_payload(
    *,
    fingerprint: Mapping[str, Any],
    completed_pairs: int,
    baseline_ms: Sequence[float],
    selected_ms: Sequence[float],
    reference_pair_sha256: Sequence[str],
    candidate_pair_sha256: Sequence[str],
    pair_flip_counts: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "written_at_utc": utc_now(),
        "fingerprint_sha256": fingerprint["sha256"],
        "completed_pairs": completed_pairs,
        "baseline_ms": list(baseline_ms),
        "selected_ms": list(selected_ms),
        "reference_pair_sha256": list(reference_pair_sha256),
        "candidate_pair_sha256": list(candidate_pair_sha256),
        "pair_flip_counts": list(pair_flip_counts),
    }


def canonical_matched_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project stage-02 fields whose digest is bound into terminal replay."""

    return {
        "schema": state["schema"],
        "written_at_utc": state["written_at_utc"],
        "fingerprint_sha256": state["fingerprint_sha256"],
        "completed_pairs": state["completed_pairs"],
        "baseline_ms": list(state["baseline_ms"]),
        "selected_ms": list(state["selected_ms"]),
        "reference_pair_sha256": list(state["reference_pair_sha256"]),
        "candidate_pair_sha256": list(state["candidate_pair_sha256"]),
        "pair_flip_counts": list(state["pair_flip_counts"]),
    }


def validate_canary_state(
    canary: Mapping[str, Any],
    *,
    fingerprint: Mapping[str, Any],
    expected_arms: Sequence[Mapping[str, Any]],
    baseline_threads: int,
) -> dict[str, Any]:
    """Validate stage-01 independently of receipt-owned canary fields."""

    if canary.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("canary checkpoint schema mismatch")
    if canary.get("fingerprint_sha256") != fingerprint.get("sha256"):
        raise RuntimeError("canary fingerprint mismatch")
    rows = canary.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("canary rows must be a list")
    expected_keys = {(str(arm["strategy"]), int(arm["threads"])) for arm in expected_arms}
    expected_sample_count = len(fingerprint["payload"].get("canary_indices", []))
    observed_keys = [(str(row.get("strategy")), int(row.get("threads", -1))) for row in rows]
    if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys:
        raise RuntimeError("canary arm set is incomplete, duplicated, or unexpected")
    for row in rows:
        samples = row.get("forward_ms_samples")
        if not isinstance(samples, list) or not samples:
            raise RuntimeError("canary timing samples are missing")
        if expected_sample_count and len(samples) != expected_sample_count:
            raise RuntimeError("canary timing sample count differs from fingerprinted indices")
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in samples):
            raise RuntimeError("canary timing sample is invalid")
        if not math.isclose(
            float(row.get("forward_ms_median", math.nan)),
            float(statistics.median(float(value) for value in samples)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("canary median is not derived from its samples")
        flip_count = row.get("argmax_flip_count")
        if not isinstance(flip_count, int) or isinstance(flip_count, bool) or flip_count < 0:
            raise RuntimeError("canary flip count is invalid")
    expected_selected = select_canary_arm(rows)
    expected_selected["baseline_threads"] = baseline_threads
    selected = canary.get("selected")
    if selected != expected_selected:
        raise RuntimeError("canary selected arm is not the exact zero-flip argmin")
    expected_sha = sha256_json(
        {
            "schema": canary["schema"],
            "fingerprint_sha256": canary["fingerprint_sha256"],
            "rows": rows,
            "selected": selected,
        }
    )
    if canary.get("sha256") != expected_sha:
        raise RuntimeError("canary checkpoint digest mismatch")
    return dict(selected)


def _canary_tournament(
    *, model: Any, raw: Path, indices: Sequence[int], arms: Sequence[Mapping[str, Any]], baseline_threads: int
) -> list[dict[str, Any]]:
    models = {strategy: _strategy_model(model, strategy) for strategy in STRATEGIES}
    references: dict[int, Any] = {}
    for index in indices:
        inp = base._model_input(model, base._read_frame1(raw, index))
        references[index], _ = _forward(
            models["eager_nchw_autograd"], inp, strategy="eager_nchw_autograd", threads=baseline_threads
        )
    rows = []
    for arm in arms:
        timings: list[float] = []
        flips = 0
        warm_input = base._model_input(model, base._read_frame1(raw, indices[0]))
        _forward(
            models[str(arm["strategy"])],
            warm_input,
            strategy=str(arm["strategy"]),
            threads=int(arm["threads"]),
        )
        for index in indices:
            inp = base._model_input(model, base._read_frame1(raw, index))
            logits, elapsed = _forward(
                models[str(arm["strategy"])], inp, strategy=str(arm["strategy"]), threads=int(arm["threads"])
            )
            timings.append(elapsed)
            flips += base._compare_logits(references[index], logits)["argmax_flip_count"]
        rows.append(
            {
                **arm,
                "forward_ms_samples": timings,
                "forward_ms_median": statistics.median(timings),
                "argmax_flip_count": flips,
            }
        )
    return rows


def _git_custody() -> dict[str, Any]:
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
    }


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    current_custody: Mapping[str, Any],
    latest_state: Mapping[str, Any],
    canary_state: Mapping[str, Any],
    terminal_stage: Mapping[str, Any],
) -> None:
    """Re-derive every load-bearing verdict field before trusting bytes."""

    measurement = receipt["measurement"]
    n_pairs = int(measurement["n_real_pairs"])
    fingerprint = receipt["resume"]["fingerprint"]
    fingerprint_payload = fingerprint["payload"]
    completed = validate_resume_state(latest_state, fingerprint)
    baseline = summarize_ms(latest_state["baseline_ms"])
    selected_summary = summarize_ms(latest_state["selected_ms"])
    expected_sign = matched_sign_test(latest_state["baseline_ms"], latest_state["selected_ms"])
    expected_speedup = float(baseline["median_ms"]) / float(selected_summary["median_ms"])
    selection_key = receipt["selection_key"]
    expected_arms = selection_key["payload"]["candidate_arms"]
    baseline_threads = int(receipt["topology"]["observed"]["torch_default"])
    expected_selected = validate_canary_state(
        canary_state,
        fingerprint=fingerprint,
        expected_arms=expected_arms,
        baseline_threads=baseline_threads,
    )
    selected = receipt["selected_arm"]
    terminal_replay = latest_state.get("terminal_replay", {})
    terminal_payload = {key: value for key, value in terminal_replay.items() if key != "sha256"}
    checkpoint_sha256 = sha256_json(canonical_matched_state(latest_state))
    checkpoint_flip_count = sum(int(value) for value in latest_state["pair_flip_counts"])
    expected_admission = admit_selected_arm(
        selected=expected_selected,
        n_pairs=n_pairs,
        verdict_pair_cardinality=VERDICT_PAIR_CARDINALITY,
        baseline_median_ms=float(baseline["median_ms"]),
        selected_median_ms=float(selected_summary["median_ms"]),
        flip_count=checkpoint_flip_count,
        reference_sha256=str(terminal_replay.get("reference_argmax_sha256", "")),
        candidate_sha256=terminal_replay.get("candidate_argmax_sha256"),
        matched_sign_pvalue=float(expected_sign["one_sided_exact_binomial_pvalue"]),
        matched_sign_alpha=MATCHED_SIGN_ALPHA,
    )
    expected_verdict = (
        "GO" if expected_admission else "DIAGNOSTIC_ONLY" if n_pairs != VERDICT_PAIR_CARDINALITY else "NO-GO"
    )
    checks = {
        "schema": receipt.get("schema") == SCHEMA,
        "selection_key_digest": selection_key.get("sha256") == sha256_json(selection_key["payload"]),
        "selection_key_fingerprint": fingerprint_payload.get("selection_key_sha256") == selection_key.get("sha256"),
        "topology_copy": receipt["topology"] == selection_key["payload"]["forward_signature"]["thread_topology"],
        "policy_contract_copy": receipt["policy_contracts"] == selection_key["payload"]["policy_contracts"],
        "verdict": receipt.get("verdict") == expected_verdict,
        "n_pairs": n_pairs == int(fingerprint_payload["n_pairs"]),
        "full_latest_checkpoint": int(latest_state.get("completed_pairs", -1)) == n_pairs,
        "latest_checkpoint_valid": completed == n_pairs,
        "baseline_summary_from_checkpoint": measurement["baseline"] == baseline,
        "selected_summary_from_checkpoint": measurement["selected"] == selected_summary,
        "speedup": math.isclose(float(measurement["matched_speedup_x"]), expected_speedup, rel_tol=0.0, abs_tol=1e-12),
        "sign_test": measurement["matched_sign_test"] == expected_sign,
        "flip_count": int(measurement["argmax_flip_count"]) == checkpoint_flip_count,
        "flip_rate": math.isclose(
            float(measurement["argmax_flip_rate"]),
            int(measurement["argmax_flip_count"]) / (n_pairs * base.SEG_H * base.SEG_W),
            rel_tol=0.0,
            abs_tol=1e-18,
        ),
        "sha_equal": bool(measurement["argmax_sha256_equal"])
        == (
            int(measurement["argmax_flip_count"]) == 0
            and measurement["reference_argmax_sha256"] == measurement["candidate_argmax_sha256"]
        ),
        "receipt_canary_rows": receipt["canary_tournament"] == canary_state["rows"],
        "selected_canary": selected == expected_selected,
        "canary_sha_receipt": receipt["resume"]["canary_checkpoint_sha256"] == canary_state["sha256"],
        "canary_sha_terminal": terminal_replay.get("canary_checkpoint_sha256") == canary_state["sha256"],
        "matched_checkpoint_sha_receipt": receipt["resume"]["matched_checkpoint_sha256"] == checkpoint_sha256,
        "matched_checkpoint_sha_terminal": terminal_replay.get("matched_checkpoint_sha256") == checkpoint_sha256,
        "terminal_replay_fingerprint": terminal_replay.get("fingerprint_sha256") == fingerprint.get("sha256"),
        "terminal_replay_digest": terminal_replay.get("sha256") == sha256_json(terminal_payload),
        "standalone_terminal_stage": terminal_stage == terminal_replay,
        "terminal_replay_reference_sha": terminal_replay.get("reference_argmax_sha256")
        == measurement["reference_argmax_sha256"],
        "terminal_replay_candidate_sha": terminal_replay.get("candidate_argmax_sha256")
        == measurement["candidate_argmax_sha256"],
        "terminal_replay_flips": int(terminal_replay.get("argmax_flip_count", -1))
        == int(measurement["argmax_flip_count"]),
        "fingerprint": fingerprint.get("sha256") == sha256_json(fingerprint_payload),
        "start_end_custody": receipt["custody"]["start"] == receipt["custody"]["end"],
        "current_custody": receipt["custody"]["end"] == current_custody,
        "authority": receipt["authority"]
        == {"score_claim": False, "pointer_moved": False, "promotion_eligible": False},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"transfer receipt validation failed: {failed}")


def run_probe(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    *,
    lock_owner: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    raw = args.raw.expanduser().resolve()
    if not raw.is_file():
        raise FileNotFoundError(raw)
    available_pairs = base._raw_pair_count(raw)
    if args.n_pairs > available_pairs:
        raise ValueError(f"requested {args.n_pairs} pairs but raw has {available_pairs}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_interop_threads(args.interop_threads)
    baseline_threads = int(torch.get_num_threads())
    topology = runtime_thread_topology(torch)
    arms = derive_candidate_arms(topology)
    model, weights = base._load_model()
    first_input = base._model_input(model, base._read_frame1(raw, 0))
    start_custody = measurement_custody(raw, weights, torch)
    policy_contracts = compile_policy_contracts(topology, first_input)
    signature = forward_signature(
        model,
        first_input,
        torch,
        weights,
        topology=topology,
        runtime_custody=start_custody["build"],
    )
    key = selection_key(signature, arms, policy_contracts=policy_contracts)
    policy_canary_count = int(policy_contracts["eager_nchw_autograd"]["canary_count"])
    canary_count = min(args.n_pairs, policy_canary_count)
    if canary_count != derive_canary_count(topology, n_pairs=args.n_pairs):
        raise RuntimeError("typed DSL and runtime heuristic canary counts differ")
    canary_count_authority = "ASSUMED_HEURISTIC_SCREEN_ONLY"
    canary_indices = base.evenly_spaced_indices(args.n_pairs, canary_count)

    out = base.validate_durable_output(args.out)
    checkpoints = checkpoint_namespace(out)
    checkpoints.mkdir(parents=True, exist_ok=True)
    preflight = storage_preflight(checkpoints, n_pairs=args.n_pairs)

    fingerprint = build_run_fingerprint(
        raw=raw,
        weights=weights,
        n_pairs=args.n_pairs,
        checkpoint_interval=args.checkpoint_interval,
        canary_indices=canary_indices,
        selection=key,
        start_custody=start_custody,
    )
    fingerprint_path = checkpoints / "stage_00_fingerprint.json"
    if fingerprint_path.exists():
        prior = json.loads(fingerprint_path.read_text())
        if prior.get("sha256") != fingerprint["sha256"]:
            raise RuntimeError("existing run fingerprint mismatch")
    else:
        _atomic_checkpoint(fingerprint_path, fingerprint)

    canary_path = checkpoints / "stage_01_canary.json"
    if canary_path.exists():
        canary = json.loads(canary_path.read_text())
        canary_rows = canary["rows"]
        selected = validate_canary_state(
            canary,
            fingerprint=fingerprint,
            expected_arms=arms,
            baseline_threads=baseline_threads,
        )
    else:
        canary_rows = _canary_tournament(
            model=model, raw=raw, indices=canary_indices, arms=arms, baseline_threads=baseline_threads
        )
        selected = select_canary_arm(canary_rows)
        selected["baseline_threads"] = baseline_threads
        canary = {
            "schema": CHECKPOINT_SCHEMA,
            "fingerprint_sha256": fingerprint["sha256"],
            "rows": canary_rows,
            "selected": selected,
        }
        canary["sha256"] = sha256_json(canary)
        _atomic_checkpoint(canary_path, canary)
        selected = validate_canary_state(
            canary,
            fingerprint=fingerprint,
            expected_arms=arms,
            baseline_threads=baseline_threads,
        )

    state_path = checkpoints / "latest.json"
    if out.exists():
        existing = json.loads(out.read_text())
        prior_fingerprint = existing.get("resume", {}).get("fingerprint", {}).get("sha256")
        if prior_fingerprint != fingerprint["sha256"]:
            raise RuntimeError("completed receipt fingerprint mismatch")
        if existing.get("completed_at_utc"):
            if not state_path.exists():
                raise RuntimeError("completed receipt is missing its terminal checkpoint")
            terminal_path = checkpoints / "stage_03_dual_replay.json"
            if not terminal_path.exists():
                raise RuntimeError("completed receipt is missing standalone terminal replay")
            latest_state = json.loads(state_path.read_text())
            terminal_stage = json.loads(terminal_path.read_text())
            current_custody = measurement_custody(raw, weights, torch)
            validate_receipt(
                existing,
                current_custody=current_custody,
                latest_state=latest_state,
                canary_state=canary,
                terminal_stage=terminal_stage,
            )
            return existing
    baseline_ms: list[float]
    selected_ms: list[float]
    reference_pair_sha256: list[str]
    candidate_pair_sha256: list[str]
    pair_flip_counts: list[int]
    if state_path.exists():
        state = json.loads(state_path.read_text())
        completed = validate_resume_state(state, fingerprint)
        baseline_ms = [float(value) for value in state["baseline_ms"]]
        selected_ms = [float(value) for value in state["selected_ms"]]
        reference_pair_sha256 = [str(value) for value in state["reference_pair_sha256"]]
        candidate_pair_sha256 = [str(value) for value in state["candidate_pair_sha256"]]
        pair_flip_counts = [int(value) for value in state["pair_flip_counts"]]
    else:
        completed = 0
        baseline_ms, selected_ms = [], []
        reference_pair_sha256, candidate_pair_sha256, pair_flip_counts = [], [], []

    baseline_model = _strategy_model(model, "eager_nchw_autograd")
    selected_model = _strategy_model(model, str(selected["strategy"]))
    for pair_index in range(completed, args.n_pairs):
        inp = base._model_input(model, base._read_frame1(raw, pair_index))
        if pair_index % 2 == 0:
            reference, ref_ms = _forward(
                baseline_model,
                inp,
                strategy="eager_nchw_autograd",
                threads=baseline_threads,
            )
            candidate, cand_ms = _forward(
                selected_model,
                inp,
                strategy=str(selected["strategy"]),
                threads=int(selected["threads"]),
            )
        else:
            candidate, cand_ms = _forward(
                selected_model,
                inp,
                strategy=str(selected["strategy"]),
                threads=int(selected["threads"]),
            )
            reference, ref_ms = _forward(
                baseline_model,
                inp,
                strategy="eager_nchw_autograd",
                threads=baseline_threads,
            )
        reference_argmax = reference.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False)
        candidate_argmax = candidate.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False)
        comparison = compare_argmax_arrays(reference_argmax, candidate_argmax)
        baseline_ms.append(ref_ms)
        selected_ms.append(cand_ms)
        reference_pair_sha256.append(comparison["reference_pair_sha256"])
        candidate_pair_sha256.append(comparison["candidate_pair_sha256"])
        pair_flip_counts.append(comparison["argmax_flip_count"])
        done = pair_index + 1
        if done % args.checkpoint_interval == 0 or done == args.n_pairs:
            checkpoint = _checkpoint_payload(
                fingerprint=fingerprint,
                completed_pairs=done,
                baseline_ms=baseline_ms,
                selected_ms=selected_ms,
                reference_pair_sha256=reference_pair_sha256,
                candidate_pair_sha256=candidate_pair_sha256,
                pair_flip_counts=pair_flip_counts,
            )
            _atomic_checkpoint(checkpoints / f"stage_02_matched_{done:06d}.json", checkpoint)
            _atomic_checkpoint(state_path, checkpoint)

    # Directly replay BOTH arms for every pair.  This authenticates a resumed
    # prefix and constructs each sequence SHA from its own raw argmax bytes.
    reference_sequence_digest = hashlib.sha256()
    candidate_sequence_digest = hashlib.sha256()
    replay_flip_count = 0
    for pair_index in range(args.n_pairs):
        inp = base._model_input(model, base._read_frame1(raw, pair_index))
        reference_replay, _ = _forward(
            baseline_model,
            inp,
            strategy="eager_nchw_autograd",
            threads=baseline_threads,
        )
        candidate_replay, _ = _forward(
            selected_model,
            inp,
            strategy=str(selected["strategy"]),
            threads=int(selected["threads"]),
        )
        reference_array = np.ascontiguousarray(
            reference_replay.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False)
        )
        candidate_array = np.ascontiguousarray(
            candidate_replay.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False)
        )
        comparison = compare_argmax_arrays(reference_array, candidate_array)
        expected = (
            reference_pair_sha256[pair_index],
            candidate_pair_sha256[pair_index],
            pair_flip_counts[pair_index],
        )
        observed = (
            comparison["reference_pair_sha256"],
            comparison["candidate_pair_sha256"],
            comparison["argmax_flip_count"],
        )
        if observed != expected:
            raise RuntimeError(f"dual-arm argmax replay drift at pair {pair_index}")
        reference_sequence_digest.update(reference_array.tobytes())
        candidate_sequence_digest.update(candidate_array.tobytes())
        replay_flip_count += int(comparison["argmax_flip_count"])
    reference_sequence_sha256 = reference_sequence_digest.hexdigest()
    candidate_sequence_sha256 = candidate_sequence_digest.hexdigest()
    flip_count = sum(pair_flip_counts)
    if replay_flip_count != flip_count:
        raise RuntimeError("dual-arm replay flip total differs from checkpointed measurement")
    equality = {
        "reference_argmax_sha256": reference_sequence_sha256,
        "candidate_argmax_sha256": candidate_sequence_sha256,
        "argmax_sha256_equal": reference_sequence_sha256 == candidate_sequence_sha256,
        "argmax_flip_count": flip_count,
        "sequence_sha256_method": "direct raw argmax bytes from independent full replays of both arms",
    }
    terminal_replay = {
        "written_at_utc": utc_now(),
        "fingerprint_sha256": fingerprint["sha256"],
        "canary_checkpoint_sha256": canary["sha256"],
        "matched_checkpoint_sha256": sha256_json(canonical_matched_state(json.loads(state_path.read_text()))),
        **equality,
    }
    terminal_replay["sha256"] = sha256_json(terminal_replay)
    latest_state = json.loads(state_path.read_text())
    latest_state["terminal_replay"] = terminal_replay
    _atomic_checkpoint(checkpoints / "stage_03_dual_replay.json", terminal_replay)
    _atomic_checkpoint(state_path, latest_state)
    terminal_stage = json.loads((checkpoints / "stage_03_dual_replay.json").read_text())
    baseline_summary = summarize_ms(baseline_ms)
    selected_summary = summarize_ms(selected_ms)
    sign_test = matched_sign_test(baseline_ms, selected_ms)
    selected["baseline_threads"] = baseline_threads
    admitted = admit_selected_arm(
        selected=selected,
        n_pairs=args.n_pairs,
        verdict_pair_cardinality=VERDICT_PAIR_CARDINALITY,
        baseline_median_ms=baseline_summary["median_ms"],
        selected_median_ms=selected_summary["median_ms"],
        flip_count=equality["argmax_flip_count"],
        reference_sha256=equality["reference_argmax_sha256"],
        candidate_sha256=equality["candidate_argmax_sha256"],
        matched_sign_pvalue=float(sign_test["one_sided_exact_binomial_pvalue"]),
        matched_sign_alpha=MATCHED_SIGN_ALPHA,
    )
    verdict = "GO" if admitted else "DIAGNOSTIC_ONLY" if args.n_pairs != VERDICT_PAIR_CARDINALITY else "NO-GO"
    end_custody = measurement_custody(raw, weights, torch)
    if end_custody != start_custody:
        raise RuntimeError("measurement custody changed between start and terminal replay")
    receipt = {
        "schema": SCHEMA,
        "completed_at_utc": utc_now(),
        "verdict": verdict,
        "verdict_scope": (
            f"formulation and observed host/build only: frozen torch fp32 SegNet over first {args.n_pairs} "
            "receiver-realized pairs; n<600 is diagnostic-only; no transfer to another model, build, "
            "host, input set, backward, trainer, or contest CPU/CUDA axis"
        ),
        "labels": {
            "topology_and_storage": "DERIVED",
            "canary_count": canary_count_authority,
            "canary_and_matched_timing": "MEASURED",
            "matched_sign_test": "DERIVED_FROM_MATCHED_MEASUREMENTS",
        },
        "authority": {"score_claim": False, "pointer_moved": False, "promotion_eligible": False},
        "topology": topology,
        "policy_contracts": policy_contracts,
        "selection_key": key,
        "canary_tournament": canary_rows,
        "selected_arm": selected,
        "measurement": {
            "label": "MEASURED",
            "n_real_pairs": args.n_pairs,
            "alternating_order": "baseline first on even pair indices; selected first on odd pair indices",
            "baseline": baseline_summary,
            "selected": selected_summary,
            "matched_speedup_x": baseline_summary["median_ms"] / selected_summary["median_ms"],
            "matched_sign_test": sign_test,
            "argmax_flip_rate": equality["argmax_flip_count"] / (args.n_pairs * base.SEG_H * base.SEG_W),
            **equality,
            "input_gradient_graph_preserved": True,
        },
        "storage": {
            "preflight": preflight,
            "large_artifacts_created": False,
            "argmax_tensor_bytes_persisted": 0,
            "cleanup_policy": "not applicable; only small atomic JSON checkpoints are persisted",
        },
        "resume": {
            "fingerprint": fingerprint,
            "canary_checkpoint_sha256": canary["sha256"],
            "matched_checkpoint_sha256": terminal_replay["matched_checkpoint_sha256"],
            "checkpoint_dir": str(checkpoints),
            "interval_pairs": args.checkpoint_interval,
            "recovered_from_completed_pairs": completed,
        },
        "custody": {
            "argv": [sys.executable, str(TOOL_PATH), *raw_argv],
            "git": _git_custody(),
            "lock_owner": dict(lock_owner),
            "start": start_custody,
            "end": end_custody,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "torch_config_sha256": signature["torch_config_sha256"],
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "validation": {
            "status": "self-validated-before-and-after-atomic-write",
            "validator": "validate_receipt",
        },
    }
    latest_state = json.loads(state_path.read_text())
    validate_receipt(
        receipt,
        current_custody=end_custody,
        latest_state=latest_state,
        canary_state=canary,
        terminal_stage=terminal_stage,
    )
    base.atomic_write_json(out, receipt)
    persisted = json.loads(out.read_text())
    validate_receipt(
        persisted,
        current_custody=end_custody,
        latest_state=latest_state,
        canary_state=canary,
        terminal_stage=terminal_stage,
    )
    return persisted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n_pairs < 1:
        parser.error("--n-pairs must be positive")
    if args.checkpoint_interval < 1 or args.interop_threads < 1 or args.seed < 0:
        parser.error("checkpoint/thread counts must be positive and seed non-negative")
    args.out = base.validate_durable_output(args.out)
    if args.out.suffix != ".json":
        parser.error("--out must end in .json so its checkpoint namespace is unique")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    with exclusive_run_lock(args.out) as lock_owner:
        receipt = run_probe(args, raw_argv, lock_owner=lock_owner)
    print(json.dumps({"verdict": receipt["verdict"], "measurement": receipt["measurement"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
