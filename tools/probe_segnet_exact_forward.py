#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure an exact frozen-SegNet CPU forward control law on real witness bytes.

The probe streams frame 1 from receiver-realized ``.raw`` pairs, applies the
canonical SegNet preprocessing, and compares the existing six-thread fp32
forward with a finite, canary-selected fp32 thread count.  Forward timing keeps
autograd enabled and the input requires gradients, matching the training
forward's graph-building obligation; backward is deliberately outside scope.

This is local macOS-CPU advisory evidence only.  It does not edit a trainer,
launch a run, call the evaluator, or make a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = Path(__file__).resolve()
CAMERA_H = 874
CAMERA_W = 1164
CHANNELS = 3
SEG_H = 384
SEG_W = 512
SCHEMA = "frozen_segnet_exact_forward_probe_v1"
AXIS = "[macOS-CPU advisory; torch-fp32; autograd-enabled forward; no MPS/CUDA]"
_TRANSIENT_PREFIXES = (
    Path("/tmp"),
    Path("/private/tmp"),
    Path("/var/tmp"),
    Path("/private/var/folders"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_source_tree_custody(package: Any) -> dict[str, Any]:
    """Hash the installed Python source tree, not only its version label."""

    init_path = Path(package.__file__).resolve()
    root = init_path.parent
    digest = hashlib.sha256()
    source_bytes = 0
    source_files = 0
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        source_bytes += len(payload)
        source_files += 1
    return {
        "package_root": str(root),
        "version": importlib.metadata.version(package.__name__),
        "python_source_files": source_files,
        "python_source_bytes": source_bytes,
        "python_source_tree_sha256": digest.hexdigest(),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_durable_output(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if any(_is_relative_to(resolved, prefix) for prefix in _TRANSIENT_PREFIXES):
        raise ValueError(f"refusing transient evidence path: {resolved}")
    approved = (REPO_ROOT, Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact"))
    if not any(_is_relative_to(resolved, root.resolve()) for root in approved):
        raise ValueError(f"output must be under the repo or approved Pact SSD: {resolved}")
    return resolved


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def parse_thread_candidates(value: str, baseline_threads: int) -> tuple[int, ...]:
    if value.strip().lower() == "auto":
        return tuple(range(1, baseline_threads + 1))
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("thread candidates must be comma-separated positive integers") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError("thread candidates must be comma-separated positive integers")
    return tuple(sorted({*parsed, baseline_threads}))


def evenly_spaced_indices(n_pairs: int, count: int) -> tuple[int, ...]:
    if n_pairs < 1 or count < 1 or count > n_pairs:
        raise ValueError("require 1 <= canary_count <= n_pairs")
    if count == 1:
        return (0,)
    return tuple(round(i * (n_pairs - 1) / (count - 1)) for i in range(count))


def select_thread_arm(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["argmax_flip_count"] == 0]
    if not eligible:
        raise RuntimeError("no zero-flip thread candidate survived the canary")
    return min(eligible, key=lambda row: (row["forward_ms_median"], row["threads"]))


def summarize_ms(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty timing vector")
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples_ms": [float(value) for value in values],
        "count": len(values),
        "median_ms": float(np.median(array)),
        "mean_ms": float(np.mean(array)),
        "p05_ms": float(np.percentile(array, 5)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def derive_economics(
    *,
    cheap_ms: float,
    matched_control_ms: float,
    anchor_forward_ms: float,
    anchor_backward_ms: float,
) -> dict[str, float]:
    if min(cheap_ms, matched_control_ms, anchor_forward_ms) <= 0 or anchor_backward_ms < 0:
        raise ValueError("timings must be positive and backward must be non-negative")
    anchor_total = anchor_forward_ms + anchor_backward_ms
    return {
        "matched_forward_speedup_x": matched_control_ms / cheap_ms,
        "matched_forward_time_reduction_fraction": 1.0 - cheap_ms / matched_control_ms,
        "profile_anchor_forward_speedup_x_unmatched_instrumentation": anchor_forward_ms / cheap_ms,
        "profile_anchor_forward_time_reduction_fraction_unmatched_instrumentation": 1.0 - cheap_ms / anchor_forward_ms,
        "profile_anchor_scorer_slice_speedup_if_backward_unchanged_x": anchor_total / (cheap_ms + anchor_backward_ms),
        "profile_anchor_yopo_ideal_speedup_if_backward_removed_x": anchor_total / cheap_ms,
        "profile_anchor_yopo_multiplier_uplift_vs_old_forward_x": anchor_forward_ms / cheap_ms,
        "profile_anchor_surrogate_validation_exact_call_uplift_x": anchor_forward_ms / cheap_ms,
    }


def admit_thread_arm(
    *,
    selected_threads: int,
    baseline_threads: int,
    argmax_flip_count: int,
    cheap_median_ms: float,
    baseline_median_ms: float,
    composed_timing_noise_floor_ms: float,
    controls_passed: bool,
) -> bool:
    """Fail closed unless the arm is distinct, exact, material, and canary-valid."""

    speed_gap_ms = baseline_median_ms - cheap_median_ms
    return (
        selected_threads != baseline_threads
        and argmax_flip_count == 0
        and cheap_median_ms < baseline_median_ms
        and speed_gap_ms > composed_timing_noise_floor_ms
        and controls_passed
    )


def validate_report(report: dict[str, Any]) -> None:
    """Recompute load-bearing receipt fields before atomic persistence."""

    measurement = report["measurement"]
    control = report["control_law"]
    controls_passed = report["controls"]["passed"]
    baseline = measurement["baseline_forward"]
    cheap = measurement["cheap_forward"]
    n_pairs = measurement["n_real_pairs"]
    flip_count = measurement["argmax_flip_count"]
    if n_pairs < 64:
        raise RuntimeError("receipt admission requires at least 64 real pairs")
    expected_pixels = n_pairs * SEG_H * SEG_W
    if measurement["total_argmax_pixels"] != expected_pixels:
        raise RuntimeError("receipt total_argmax_pixels is inconsistent")
    expected_speed_gap = baseline["median_ms"] - cheap["median_ms"]
    expected_floor = (
        baseline["p95_ms"] - baseline["p05_ms"] + cheap["p95_ms"] - cheap["p05_ms"]
    )
    expected_admission = admit_thread_arm(
        selected_threads=control["selected_threads"],
        baseline_threads=control["baseline_threads"],
        argmax_flip_count=flip_count,
        cheap_median_ms=cheap["median_ms"],
        baseline_median_ms=baseline["median_ms"],
        composed_timing_noise_floor_ms=expected_floor,
        controls_passed=controls_passed,
    )
    expected_verdict = "GO" if expected_admission else "NO-GO"
    checks = {
        "verdict": report["verdict"] == expected_verdict,
        "argmax_bit_identical": measurement["argmax_bit_identical"] == (flip_count == 0),
        "argmax_flip_rate": math.isclose(
            measurement["argmax_flip_rate"], flip_count / expected_pixels, abs_tol=1e-15
        ),
        "argmax_digest": flip_count != 0
        or measurement["reference_argmax_sha256"] == measurement["candidate_argmax_sha256"],
        "matched_speed_gap_ms": math.isclose(
            measurement["matched_speed_gap_ms"], expected_speed_gap, abs_tol=1e-9
        ),
        "composed_timing_noise_floor_ms": math.isclose(
            measurement["composed_timing_noise_floor_ms"], expected_floor, abs_tol=1e-9
        ),
        "speed_gap_exceeds_composed_floor": measurement[
            "speed_gap_exceeds_composed_floor"
        ]
        == (expected_speed_gap > expected_floor),
        "controls": bool(controls_passed),
        "score_claim": report["authority"]["score_claim"] is False,
        "pointer_moved": report["authority"]["pointer_moved"] is False,
        "tool_sha256": report["custody"]["tool_sha256"] == _sha256_file(TOOL_PATH),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"receipt validation failed: {failed}")


def _ensure_import_paths() -> None:
    for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "upstream"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _load_model() -> tuple[Any, Path]:
    _ensure_import_paths()
    import torch
    from modules import SegNet
    from safetensors.torch import load_file

    weights = REPO_ROOT / "upstream/models/segnet.safetensors"
    model = SegNet().eval().to("cpu")
    model.load_state_dict(load_file(weights, device="cpu"), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.use_deterministic_algorithms(True)
    return model, weights


def _scorer_runtime_custody() -> dict[str, Any]:
    _ensure_import_paths()
    import modules
    import segmentation_models_pytorch as smp
    import timm

    modules_path = Path(modules.__file__).resolve()
    return {
        "upstream_modules_path": str(modules_path.relative_to(REPO_ROOT)),
        "upstream_modules_bytes": modules_path.stat().st_size,
        "upstream_modules_sha256": _sha256_file(modules_path),
        "segmentation_models_pytorch": _python_source_tree_custody(smp),
        "timm": _python_source_tree_custody(timm),
    }


def _canonical_pointer_state() -> dict[str, Any]:
    from tac.frontier_scan import build_frontier_scan_payload

    payload = build_frontier_scan_payload(REPO_ROOT)
    cpu_rows = payload["top_5_per_axis"]["contest_cpu"]
    defensive = cpu_rows[0]
    submittable = next(
        row
        for row in cpu_rows
        if "nonsubmission" not in str(row.get("extra", {}).get("architecture_class", "")).lower()
    )
    return {
        "nonsubmission_defensive_bank": float(defensive["score"]),
        "nonsubmission_defensive_bank_sha256": defensive["archive_sha256"],
        "submittable_contest_cpu_pointer": float(submittable["score"]),
        "submittable_contest_cpu_sha256": submittable["archive_sha256"],
        "source": "tac.frontier_scan.build_frontier_scan_payload",
    }


def _raw_pair_count(raw_path: Path) -> int:
    frame_bytes = CAMERA_H * CAMERA_W * CHANNELS
    pair_bytes = 2 * frame_bytes
    size = raw_path.stat().st_size
    if size % pair_bytes:
        raise ValueError(f"raw byte count {size} is not an integral pair count")
    return size // pair_bytes


def _read_frame1(raw_path: Path, pair_index: int) -> np.ndarray:
    frame_bytes = CAMERA_H * CAMERA_W * CHANNELS
    with raw_path.open("rb") as handle:
        handle.seek((2 * pair_index + 1) * frame_bytes)
        payload = handle.read(frame_bytes)
    if len(payload) != frame_bytes:
        raise ValueError(f"raw file ended before pair {pair_index} frame 1")
    return np.frombuffer(payload, dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, CHANNELS).copy()


def _source_archive_custody(raw_path: Path) -> dict[str, Any]:
    packet_dir = raw_path.parent.parent if raw_path.parent.name == "inflated" else None
    if packet_dir is None:
        return {"status": "unknown", "reason": "raw path is not under an inflated packet directory"}
    rows: dict[str, Any] = {"status": "located", "packet_dir": str(packet_dir.relative_to(REPO_ROOT))}
    for name in ("archive.zip", "inflate.py", "inflate.sh"):
        path = packet_dir / name
        if path.is_file():
            rows[name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        else:
            rows[name] = {"missing": True}
            rows["status"] = "incomplete"
    return rows


def _model_input(model: Any, frame_hwc_u8: np.ndarray) -> Any:
    import torch

    frame = torch.from_numpy(frame_hwc_u8).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
    result = model.preprocess_input(frame).detach().contiguous()
    if tuple(result.shape) != (1, 3, SEG_H, SEG_W) or result.dtype != torch.float32:
        raise RuntimeError(f"canonical SegNet input drift: {tuple(result.shape)}/{result.dtype}")
    return result


def _forward(model: Any, model_input: Any, threads: int) -> tuple[Any, float]:
    import torch

    torch.set_num_threads(threads)
    sample = model_input.detach().clone().requires_grad_(True)
    started = time.perf_counter_ns()
    logits = model(sample)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if not logits.requires_grad or logits.grad_fn is None:
        raise RuntimeError("forward did not build the required input-gradient graph")
    return logits.detach(), elapsed_ms


def _warm_thread_pool(model: Any, model_input: Any, threads: int) -> None:
    _forward(model, model_input, threads)


def _quantization_canary(model_input: Any, reference: Any, threads: int) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    for label, dtype in (("fp16", torch.float16), ("bfloat16", torch.bfloat16)):
        try:
            candidate_model, _ = _load_model()
            candidate_model.to(dtype=dtype)
            candidate_input = model_input.to(dtype=dtype)
            _warm_thread_pool(candidate_model, candidate_input, threads)
            logits, elapsed_ms = _forward(candidate_model, candidate_input, threads)
            comparison = _compare_logits(reference.float(), logits.float())
            rows.append(
                {
                    "dtype": label,
                    "status": "measured",
                    "forward_ms": elapsed_ms,
                    **comparison,
                }
            )
        except Exception as exc:  # fail-closed evidence row, never admission
            rows.append(
                {
                    "dtype": label,
                    "status": "unsupported",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return rows


def _compare_logits(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    reference_argmax = reference.argmax(dim=1)
    candidate_argmax = candidate.argmax(dim=1)
    flips = int(torch.count_nonzero(reference_argmax != candidate_argmax))
    return {
        "argmax_flip_count": flips,
        "argmax_flip_rate": flips / reference_argmax.numel(),
        "logit_max_abs_delta": float((reference - candidate).abs().max()),
        "logit_mean_abs_delta": float((reference - candidate).abs().mean()),
    }


def _git_custody() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=REPO_ROOT,
        text=True,
    )
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def run_probe(args: argparse.Namespace, raw_argv: Sequence[str]) -> dict[str, Any]:
    import torch

    started_at = _utc_now()
    raw_path = args.raw.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    if not raw_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("raw witness and profile anchor must both exist")
    available_pairs = _raw_pair_count(raw_path)
    if args.n_pairs > available_pairs:
        raise ValueError(f"requested {args.n_pairs} pairs but raw has {available_pairs}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_interop_threads(args.interop_threads)
    model, weights = _load_model()
    pointer_state = _canonical_pointer_state()
    candidates = parse_thread_candidates(args.thread_candidates, args.baseline_threads)
    canary_indices = evenly_spaced_indices(args.n_pairs, args.canary_count)
    input0 = _model_input(model, _read_frame1(raw_path, canary_indices[0]))

    canary_references: dict[int, Any] = {}
    for pair_index in canary_indices:
        model_input = _model_input(model, _read_frame1(raw_path, pair_index))
        _warm_thread_pool(model, model_input, args.baseline_threads)
        canary_references[pair_index], _ = _forward(
            model, model_input, args.baseline_threads
        )

    canary_rows: list[dict[str, Any]] = []
    for threads in candidates:
        _warm_thread_pool(model, input0, threads)
        timings: list[float] = []
        flips = 0
        max_abs = 0.0
        for pair_index in canary_indices:
            model_input = _model_input(model, _read_frame1(raw_path, pair_index))
            logits, elapsed_ms = _forward(model, model_input, threads)
            comparison = _compare_logits(canary_references[pair_index], logits)
            timings.append(elapsed_ms)
            flips += comparison["argmax_flip_count"]
            max_abs = max(max_abs, comparison["logit_max_abs_delta"])
        canary_rows.append(
            {
                "threads": threads,
                "forward_ms_samples": timings,
                "forward_ms_median": statistics.median(timings),
                "argmax_flip_count": flips,
                "argmax_flip_rate": flips / (len(canary_indices) * SEG_H * SEG_W),
                "logit_max_abs_delta": max_abs,
            }
        )
    selected = select_thread_arm(canary_rows)
    selected_threads = int(selected["threads"])

    # P4 controls: exact same-arm rerun must be identical; a class-axis rotation
    # must be detected as non-identical by the same flip meter.
    _warm_thread_pool(model, input0, args.baseline_threads)
    positive_a, _ = _forward(model, input0, args.baseline_threads)
    positive_b, _ = _forward(model, input0, args.baseline_threads)
    positive = _compare_logits(positive_a, positive_b)
    negative = _compare_logits(positive_a, torch.roll(positive_a, shifts=1, dims=1))
    controls_passed = (
        positive["argmax_flip_count"] == 0
        and positive["logit_max_abs_delta"] == 0.0
        and negative["argmax_flip_count"] == SEG_H * SEG_W
    )
    if not controls_passed:
        raise RuntimeError(f"argmax meter controls failed: positive={positive}, negative={negative}")

    _warm_thread_pool(model, input0, selected_threads)
    fp32_quant_control, fp32_quant_control_ms = _forward(model, input0, selected_threads)
    quantization_canary = _quantization_canary(
        input0, fp32_quant_control, selected_threads
    )

    for threads in (args.baseline_threads, selected_threads):
        _warm_thread_pool(model, input0, threads)

    baseline_ms: list[float] = []
    cheap_ms: list[float] = []
    flip_count = 0
    max_abs_delta = 0.0
    mean_abs_weighted = 0.0
    min_reference_margin = math.inf
    reference_argmax_digest = hashlib.sha256()
    candidate_argmax_digest = hashlib.sha256()

    for pair_index in range(args.n_pairs):
        model_input = _model_input(model, _read_frame1(raw_path, pair_index))
        if pair_index % 2 == 0:
            reference, reference_ms = _forward(model, model_input, args.baseline_threads)
            candidate, candidate_ms = _forward(model, model_input, selected_threads)
        else:
            candidate, candidate_ms = _forward(model, model_input, selected_threads)
            reference, reference_ms = _forward(model, model_input, args.baseline_threads)
        comparison = _compare_logits(reference, candidate)
        baseline_ms.append(reference_ms)
        cheap_ms.append(candidate_ms)
        flip_count += comparison["argmax_flip_count"]
        max_abs_delta = max(max_abs_delta, comparison["logit_max_abs_delta"])
        mean_abs_weighted += comparison["logit_mean_abs_delta"]
        values = torch.topk(reference, k=2, dim=1).values
        min_reference_margin = min(
            min_reference_margin, float((values[:, 0] - values[:, 1]).min())
        )
        ref_argmax = reference.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False)
        cand_argmax = candidate.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False)
        reference_argmax_digest.update(np.ascontiguousarray(ref_argmax).tobytes())
        candidate_argmax_digest.update(np.ascontiguousarray(cand_argmax).tobytes())

    baseline_summary = summarize_ms(baseline_ms)
    cheap_summary = summarize_ms(cheap_ms)
    total_pixels = args.n_pairs * SEG_H * SEG_W
    profile = json.loads(profile_path.read_text())
    anchor_forward_ms = float(profile["measured"]["forward_ms"])
    anchor_backward_ms = float(profile["measured"]["input_gradient_backward_ms"])
    economics = derive_economics(
        cheap_ms=cheap_summary["median_ms"],
        matched_control_ms=baseline_summary["median_ms"],
        anchor_forward_ms=anchor_forward_ms,
        anchor_backward_ms=anchor_backward_ms,
    )
    speed_gap_ms = baseline_summary["median_ms"] - cheap_summary["median_ms"]
    composed_timing_noise_floor_ms = (
        baseline_summary["p95_ms"]
        - baseline_summary["p05_ms"]
        + cheap_summary["p95_ms"]
        - cheap_summary["p05_ms"]
    )
    admitted = admit_thread_arm(
        selected_threads=selected_threads,
        baseline_threads=args.baseline_threads,
        argmax_flip_count=flip_count,
        cheap_median_ms=cheap_summary["median_ms"],
        baseline_median_ms=baseline_summary["median_ms"],
        composed_timing_noise_floor_ms=composed_timing_noise_floor_ms,
        controls_passed=controls_passed,
    )
    verdict = "GO" if admitted else "NO-GO"
    verdict_scope = (
        "registered formulation/substrate regime only: exact torch-fp32 frozen SegNet forward, "
        f"macOS arm64 CPU, receiver-realized first {args.n_pairs} witness pairs, "
        f"{args.baseline_threads} threads versus canary-selected {selected_threads}; "
        "no transfer to MLX/Metal, MPS, CUDA, another host, another Torch build, or unseen pairs"
    )

    command = [sys.executable, str(TOOL_PATH), *raw_argv]
    return {
        "schema": SCHEMA,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "verdict": verdict,
        "verdict_scope": verdict_scope,
        "review_status": "recovery-written-UNREVIEWED",
        "axis": AXIS,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "pointer_moved": False,
            **pointer_state,
        },
        "control_law": {
            "kind": "event-conditioned tested predicate with finite completion guarantee",
            "candidate_threads": list(candidates),
            "canary_pair_indices": list(canary_indices),
            "selection": "minimum canary median forward_ms among arms with exactly zero argmax flips versus baseline",
            "completion_guarantee": "finite candidate list times finite canary list; always includes baseline",
            "candidate_derivation": "auto enumerates every positive integer thread count from 1 through the profile-anchored baseline",
            "canary_recess_measurement": "task456_even_quartile_canary_recess; full n64 paired gate is the admission authority",
            "runtime_gate": "activate selected thread count only while exact paired argmax flip_count remains zero; otherwise use baseline",
            "selected_threads": selected_threads,
            "baseline_threads": args.baseline_threads,
        },
        "controls": {
            "positive_same_arm_rerun": positive,
            "negative_class_axis_rotation": negative,
            "passed": controls_passed,
        },
        "quantization_canary": {
            "scope": "pair 0 only; diagnostic screen, never n64 admission evidence",
            "fp32_selected_thread_control_ms": fp32_quant_control_ms,
            "arms": quantization_canary,
        },
        "canary_tournament": canary_rows,
        "measurement": {
            "n_real_pairs": args.n_pairs,
            "total_argmax_pixels": total_pixels,
            "argmax_flip_count": flip_count,
            "argmax_flip_rate": flip_count / total_pixels,
            "argmax_bit_identical": flip_count == 0,
            "reference_argmax_sha256": reference_argmax_digest.hexdigest(),
            "candidate_argmax_sha256": candidate_argmax_digest.hexdigest(),
            "logit_max_abs_delta": max_abs_delta,
            "logit_mean_abs_delta_mean_over_pairs": mean_abs_weighted / args.n_pairs,
            "minimum_reference_top1_top2_margin": min_reference_margin,
            "baseline_forward": baseline_summary,
            "cheap_forward": cheap_summary,
            "matched_speed_gap_ms": speed_gap_ms,
            "composed_timing_noise_floor_ms": composed_timing_noise_floor_ms,
            "speed_gap_exceeds_composed_floor": speed_gap_ms > composed_timing_noise_floor_ms,
            "alternating_order": "baseline-first on even pairs; candidate-first on odd pairs",
            "timed_region": "model(model_input) with autograd enabled and input requires_grad=True; canonical preprocessing excluded",
        },
        "profile_anchor": {
            "path": str(profile_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(profile_path),
            "forward_ms": anchor_forward_ms,
            "input_gradient_backward_ms": anchor_backward_ms,
            "warning": "single-sample hook-instrumented profile; ratios to the uninstrumented n64 probe are explicitly unmatched",
        },
        "economics": economics,
        "attack_surface_disposition": {
            "argmax_preserving_quantization": (
                "see quantization_canary: pair-0 CPU diagnostic only; any negative is formulation/"
                "substrate scoped, not a family verdict and not counted as n64 evidence"
            ),
            "fused_r_plus_stem": (
                "blocked on this session because MLX reports no Metal device; existing #212 surface "
                "contains fused-R but no source-level fused-R-plus-SegNet-stem primitive was located"
            ),
            "forward_activation_banking": (
                "not exact across changed witness inputs; requires task #454 trust-region certificate, "
                "so no bit-identical claim is made here"
            ),
        },
        "custody": {
            "command": shlex.join(command),
            "argv": command,
            "git": _git_custody(),
            "raw_path": str(raw_path.relative_to(REPO_ROOT)),
            "raw_bytes": raw_path.stat().st_size,
            "raw_sha256": _sha256_file(raw_path),
            "raw_available_pairs": available_pairs,
            "raw_contract": "uint8 receiver-realized camera frames, 2x874x1164x3 bytes per pair; frame 1 scored",
            "model_weights_path": str(weights.relative_to(REPO_ROOT)),
            "model_weights_bytes": weights.stat().st_size,
            "model_weights_sha256": _sha256_file(weights),
            "tool_path": str(TOOL_PATH.relative_to(REPO_ROOT)),
            "tool_sha256": _sha256_file(TOOL_PATH),
            "profile_anchor_sha256": _sha256_file(profile_path),
            "source_archive": _source_archive_custody(raw_path),
            "scorer_runtime": _scorer_runtime_custody(),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "interop_threads": torch.get_num_interop_threads(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "limitations": [
            "Single deterministic n64 spine; across-seed and unseen-pair argmax safety are UNKNOWN.",
            "Local arm64 CPU only; no Metal/MPS/CUDA or contest-hardware transfer authority.",
            "The raw witness is receiver-realized, so R and parse-back have already occurred; this probe times SegNet forward only.",
            "No backward timing, training integration, full-loop speed, d_seg, d_pose, archive, or score measurement.",
            "The 1,656 ms profile anchor is instrumented and single-sample; only the alternating n64 control is a matched speed denominator.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=64)
    parser.add_argument("--canary-count", type=int, default=4)
    parser.add_argument("--thread-candidates", default="auto")
    parser.add_argument("--baseline-threads", type=int, default=6)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n_pairs < 64:
        parser.error("--n-pairs must be >=64 for task #456")
    if not 1 <= args.canary_count <= args.n_pairs:
        parser.error("require 1 <= --canary-count <= --n-pairs")
    if min(args.baseline_threads, args.interop_threads) < 1 or args.seed < 0:
        parser.error("thread counts must be positive and seed must be non-negative")
    try:
        parse_thread_candidates(args.thread_candidates, args.baseline_threads)
    except ValueError as exc:
        parser.error(str(exc))
    args.out = validate_durable_output(args.out)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    report = run_probe(args, raw_argv)
    validate_report(report)
    atomic_write_json(args.out, report)
    print(json.dumps({
        "receipt": str(args.out),
        "verdict": report["verdict"],
        "selected_threads": report["control_law"]["selected_threads"],
        "n_real_pairs": report["measurement"]["n_real_pairs"],
        "argmax_flip_count": report["measurement"]["argmax_flip_count"],
        "matched_forward_speedup_x": report["economics"]["matched_forward_speedup_x"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
