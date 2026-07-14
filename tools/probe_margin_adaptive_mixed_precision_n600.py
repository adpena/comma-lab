#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable real-n600 margin-adaptive mixed-precision SegNet Metal probe.

The native candidate is one frame/label-independent per-layer precision map.
The per-pixel margin waterfill is reported separately as a finite-ladder lower
bound and never promoted to a spatial-kernel speed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "upstream", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bench_fixedpoint_authority_kernels import (  # noqa: E402
    _hash_array,
    _load_calibration,
    _load_segnet,
    _seg_input,
    _selected_tie_snap_rule,
)
from probe_exact_int64_scorer_forward_n600 import (  # noqa: E402
    _fingerprint,
    _storage_preflight,
)
from probe_fixedpoint_scorer_forward_n600 import (  # noqa: E402
    N600,
    _git_head,
)

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
    stored_npy_memmap,
)
from tac.local_acceleration.argmax_tie_snap import (  # noqa: E402
    class_pair_tie_snap_argmax_mlx,
)
from tac.local_acceleration.margin_adaptive_mixed_precision import (  # noqa: E402
    DEFAULT_PROFILE_CAPS,
    ProfileCertificate,
    build_metal_margin_adaptive_int64_segnet_adapter,
    derive_capped_precision_map,
    interval_argmax_certificate_mask,
    solve_finite_profile_waterfill,
    weighted_average_bits,
)

SCHEMA = "margin_adaptive_mixed_precision_n600.v1"
DESIGN_STOP = 264
BOUND_KIND = "CORPUS_OBSERVED_PER_PIXEL_ABS_FP32_VS_FIXEDPOINT_LOGIT_ERROR"
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_CALIBRATION = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json"
)
DEFAULT_INTEGER_PRECURSOR = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "weight_l1_class_pair_tie_snap_scorer_forward_n600.json"
)
DEFAULT_UNIFORM_NOGO = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "fixedpoint_scorer_forward_n600_fresh_89b970ff60.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/results/margin_adaptive_mixed_precision_20260714/"
    "margin_adaptive_mixed_precision_n600.json"
)
MARGIN_BANDS: tuple[float, ...] = (
    0.0,
    float(2.0**-19),
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    float("inf"),
)


def _profile_name(cap: int) -> str:
    return f"cap{int(cap)}"


def _parse_caps(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(piece) for piece in raw.split(",") if piece.strip()}))
    if not values or values[0] < 8 or values[-1] > 31:
        raise ValueError("profile caps must be a non-empty comma list within 8..31")
    return values


def _load_uniform_nogo(path: Path, *, gt_cache: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    contract = payload.get("contract", {})
    if (
        payload.get("schema") != "fixedpoint_scorer_forward_n600.v2"
        or summary.get("status") != "MEASURED"
        or summary.get("full_real_n600") is not True
        or summary.get("minimum_argmax_exact_arm") is not None
        or summary.get("rung2_verdict")
        != "NO_ADMITTED_PRECISION_IN_LADDER"
        or contract.get("activation_scale_mode") != "fixed_calibration"
        or contract.get("native_integer_speed_claim") is not False
    ):
        raise ValueError("uniform fixed-scale predecessor is not the full-n600 NO-GO")
    custody = payload.get("custody", {})
    cache_hash = custody.get("gt_cache_sha256") or payload.get("cache_custody", {}).get(
        "gt_cache_sha256"
    )
    if cache_hash is not None and cache_hash != sha256_file(gt_cache):
        raise ValueError("uniform fixed-scale predecessor names a different GT cache")
    return payload


def _stage_path(output: Path, stage: str) -> Path:
    return output.with_name(f"{output.stem}.{stage}{output.suffix}")


def _argmax_digest(rows: list[dict[str, Any]], *, profile: str, field: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: int(item["pair_index"])):
        digest.update(
            f"{row['pair_index']}:{row['profiles'][profile][field]}\n".encode("ascii")
        )
    return digest.hexdigest()


def _reference_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: int(item["pair_index"])):
        digest.update(
            f"{row['pair_index']}:{row['reference_argmax_sha256']}\n".encode("ascii")
        )
    return digest.hexdigest()


def _conv_macs_and_logits(model: Any, value: Any) -> tuple[dict[str, int], Any]:
    import torch

    paths = {id(module): name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)}
    macs: dict[str, int] = {}
    handles = []

    def hook(module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        name = paths[id(module)]
        height, width = map(int, output.shape[-2:])
        fan_in = (
            int(module.in_channels // module.groups)
            * int(module.kernel_size[0])
            * int(module.kernel_size[1])
        )
        macs[name] = int(height * width * int(module.out_channels) * fan_in)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    try:
        logits = model(value)
    finally:
        for handle in handles:
            handle.remove()
    if set(macs) != set(paths.values()):
        raise RuntimeError("Conv2d MAC census did not cover the frozen SegNet")
    return macs, logits


def _margin_band_rows(
    margin: np.ndarray,
    selected_index: np.ndarray,
    average_bits: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lower = -float("inf")
    for upper in MARGIN_BANDS:
        mask = (margin > lower) & (margin <= upper) & (selected_index >= 0)
        count = int(np.count_nonzero(mask))
        selected = selected_index[mask]
        bits_sum = float(sum(average_bits[int(index)] for index in selected))
        rows.append(
            {
                "margin_gt": None if not np.isfinite(lower) else float(lower),
                "margin_le": None if not np.isfinite(upper) else float(upper),
                "certified_pixels": count,
                "selected_bits_sum": bits_sum,
            }
        )
        lower = upper
    return rows


def _profile_aggregate(
    rows: list[dict[str, Any]],
    *,
    profile: str,
    split: str,
) -> dict[str, Any]:
    selected = rows if split == "full" else [row for row in rows if row["split"] == split]
    if not selected:
        return {"status": "INCOMPLETE", "pairs": 0}
    profile_rows = [row["profiles"][profile] for row in selected]
    flips = sum(int(row["flips"]) for row in profile_rows)
    pixels = sum(int(row["pixels"]) for row in profile_rows)
    strict = sum(int(row["strict_interval_certified_pixels"]) for row in profile_rows)
    hybrid = sum(
        int(row["strict_interval_or_frozen_tie_rule_certified_pixels"])
        for row in profile_rows
    )
    worst = max(
        zip(selected, profile_rows, strict=True),
        key=lambda pair: (float(pair[1]["flip_fraction"]), -int(pair[0]["pair_index"])),
    )
    return {
        "status": "MEASURED",
        "pairs": len(selected),
        "pixels": pixels,
        "flips": flips,
        "aggregate_flip_fraction": float(flips / pixels),
        "worst_pair_index": int(worst[0]["pair_index"]),
        "worst_pair_flip_fraction": float(worst[1]["flip_fraction"]),
        "argmax_exact_gate": flips == 0,
        "strict_interval_certified_pixels": strict,
        "strict_interval_certified_fraction": float(strict / pixels),
        "strict_interval_or_frozen_tie_rule_certified_pixels": hybrid,
        "strict_interval_or_frozen_tie_rule_certified_fraction": float(hybrid / pixels),
        "source_corpus_zero_flip_certificate_gate": hybrid == pixels,
        "maximum_abs_logit_error": max(float(row["max_abs_logit_error"]) for row in profile_rows),
        "metal_median_seconds_per_pair": statistics.median(
            float(row["metal_seconds"]) for row in profile_rows
        ),
        "metal_total_seconds": float(sum(float(row["metal_seconds"]) for row in profile_rows)),
        "argmax_corpus_sha256": _argmax_digest(selected, profile=profile, field="candidate_argmax_sha256"),
    }


def _summarize(receipt: dict[str, Any]) -> dict[str, Any]:
    rows = list(receipt.get("rows", []))
    expected = set(range(int(receipt["contract"]["pair_start"]), int(receipt["contract"]["pair_stop"])))
    indices = [int(row["pair_index"]) for row in rows]
    complete = len(indices) == len(expected) and set(indices) == expected
    if not complete:
        return {
            "status": "INCOMPLETE",
            "pairs": len(indices),
            "unique_pair_indices": len(set(indices)),
            "full_real_n600": False,
            "verdict": "INCOMPLETE",
        }
    profiles = receipt["profiles"]
    aggregates: dict[str, Any] = {}
    for name in profiles:
        aggregates[name] = {
            split: _profile_aggregate(rows, profile=name, split=split)
            for split in ("design", "second_validation", "full")
        }
        aggregates[name]["mac_weighted_average_bits"] = float(
            profiles[name]["mac_weighted_average_bits"]
        )
    order = sorted(
        profiles,
        key=lambda name: (float(profiles[name]["mac_weighted_average_bits"]), name),
    )
    design_certified = [
        name
        for name in order
        if aggregates[name]["design"]["argmax_exact_gate"]
        and aggregates[name]["design"]["source_corpus_zero_flip_certificate_gate"]
    ]
    minimum_bits_design = design_certified[0] if design_certified else None
    timing_ranked_design = sorted(
        design_certified,
        key=lambda name: (
            float(aggregates[name]["design"]["metal_median_seconds_per_pair"]),
            float(profiles[name]["mac_weighted_average_bits"]),
            name,
        ),
    )
    frozen = timing_ranked_design[0] if timing_ranked_design else None
    full_certified = [
        name
        for name in order
        if aggregates[name]["full"]["argmax_exact_gate"]
        and aggregates[name]["full"]["source_corpus_zero_flip_certificate_gate"]
    ]
    diagnostic_full_minimum = full_certified[0] if full_certified else None
    selected = aggregates.get(frozen, {}) if frozen is not None else {}
    selected_full = selected.get("full", {})
    selected_validation = selected.get("second_validation", {})
    trials = [row for row in receipt.get("trials", []) if row.get("status") == "MEASURED"]
    trial_hashes = [str(row["argmax_corpus_sha256"]) for row in trials]
    expected_hash = selected_full.get("argmax_corpus_sha256")
    deterministic = bool(
        frozen is not None
        and len(trials) == int(receipt["contract"]["n_processes"])
        and len(set(trial_hashes)) == 1
        and trial_hashes[0] == expected_hash
    )
    cpu_seconds = [float(row["reference_seconds"]) for row in rows]
    selected_metal_median = selected_full.get("metal_median_seconds_per_pair")
    speedup = (
        statistics.median(cpu_seconds) / float(selected_metal_median)
        if selected_metal_median not in (None, 0.0)
        else None
    )
    strict_pixels = sum(int(row["strict_waterfill"]["certified_pixels"]) for row in rows)
    total_pixels = sum(int(row["pixels"]) for row in rows)
    strict_bits_sum = sum(
        float(row["strict_waterfill"]["selected_bits_sum"]) for row in rows
    )
    exact_pixels = sum(int(row["exact_observed_waterfill"]["certified_pixels"]) for row in rows)
    exact_bits_sum = sum(
        float(row["exact_observed_waterfill"]["selected_bits_sum"]) for row in rows
    )
    margin_bands: list[dict[str, Any]] = []
    for index in range(len(MARGIN_BANDS)):
        band_rows = [row["strict_waterfill"]["margin_bands"][index] for row in rows]
        count = sum(int(row["certified_pixels"]) for row in band_rows)
        bits_sum = sum(float(row["selected_bits_sum"]) for row in band_rows)
        margin_bands.append(
            {
                "margin_gt": band_rows[0]["margin_gt"],
                "margin_le": band_rows[0]["margin_le"],
                "certified_pixels": count,
                "average_selected_bits": float(bits_sum / count) if count else None,
            }
        )
    exact = bool(
        frozen is not None
        and selected.get("design", {}).get("argmax_exact_gate") is True
        and selected_validation.get("argmax_exact_gate") is True
        and selected_full.get("argmax_exact_gate") is True
    )
    certificate = bool(
        frozen is not None
        and selected.get("design", {}).get("source_corpus_zero_flip_certificate_gate")
        is True
        and selected_validation.get("source_corpus_zero_flip_certificate_gate") is True
        and selected_full.get("source_corpus_zero_flip_certificate_gate") is True
    )
    positive_speed = speedup is not None and speedup > 1.0
    selected_storage_bits = (
        float(profiles[frozen]["mac_weighted_average_storage_bits"])
        if frozen is not None
        else None
    )
    physical_width_reduction = bool(
        selected_storage_bits is not None and selected_storage_bits < 32.0
    )
    metal_exact_candidate = bool(exact and certificate and deterministic and positive_speed)
    admitted = bool(metal_exact_candidate and physical_width_reduction)
    measured_segnet_cpu = float(sum(cpu_seconds))
    measured_segnet_metal = (
        float(selected_full["metal_total_seconds"]) if frozen is not None else None
    )
    derived_pose_n600 = float(372.6 * 0.226)
    projected_combined = (
        float(measured_segnet_metal + derived_pose_n600)
        if measured_segnet_metal is not None
        else None
    )
    return {
        "status": "MEASURED",
        "pairs": len(rows),
        "unique_pair_indices": len(set(indices)),
        "full_real_n600": expected == set(range(N600)),
        "profiles": aggregates,
        "design_minimum_average_bits_profile": minimum_bits_design,
        "design_selected_profile": frozen,
        "design_selection_rule": "among design-exact and design-certified profiles, minimum measured Metal median seconds/pair then minimum MAC-weighted logical bits; no second-validation reselection",
        "design_certified_timing_ranking": timing_ranked_design,
        "diagnostic_full_corpus_minimum_profile": diagnostic_full_minimum,
        "selected_second_validation_exact": bool(
            selected_validation.get("argmax_exact_gate") is True
        ),
        "selected_full_exact": bool(selected_full.get("argmax_exact_gate") is True),
        "selected_source_corpus_zero_flip_certificate": certificate,
        "strict_interval_certificate": {
            "bound_kind": BOUND_KIND,
            "certificate": "L_reference_top1 > max(U_reference_rival)",
            "certified_pixels": strict_pixels,
            "pixels": total_pixels,
            "certified_fraction": float(strict_pixels / total_pixels),
            "minimum_average_selected_bits_over_certified_pixels": (
                float(strict_bits_sum / strict_pixels) if strict_pixels else None
            ),
            "zero_margin_ties_can_remain_uncertified": True,
            "unseen_input_ibp_claim": False,
        },
        "per_region_margin_waterfill_lower_bound": {
            "exact_pointwise_minimum_over_finite_profile_ladder": True,
            "native_region_execution_claim": False,
            "global_dependency_blocker": "23 squeeze-excite reductions plus measured skip-inclusive halo685 imply full-frame exact closure",
            "margin_bands": margin_bands,
        },
        "exact_observed_profile_lower_bound": {
            "label_dependent_diagnostic_only": True,
            "pixels": exact_pixels,
            "average_selected_bits": float(exact_bits_sum / exact_pixels) if exact_pixels else None,
        },
        "cross_process": {
            "required_processes": int(receipt["contract"]["n_processes"]),
            "measured_processes": len(trials),
            "unique_argmax_corpus_hashes": len(set(trial_hashes)),
            "argmax_identical": deterministic,
        },
        "selected_precision": {
            "logical_mac_weighted_average_bits": (
                float(profiles[frozen]["mac_weighted_average_bits"])
                if frozen is not None
                else None
            ),
            "physical_mac_weighted_average_storage_bits": selected_storage_bits,
            "physical_width_reduction_vs_int32_gate": physical_width_reduction,
            "accumulator": "exact_signed_int64",
        },
        "timing": {
            "axis": "[macOS M5-Max custom Metal local candidate; CPU Torch one-thread control]",
            "cpu_reference_median_seconds_per_pair": statistics.median(cpu_seconds),
            "selected_metal_median_seconds_per_pair": selected_metal_median,
            "cpu_to_selected_metal_speedup_x": speedup,
            "measured_n600_segnet_cpu_seconds": measured_segnet_cpu,
            "measured_n600_selected_metal_seconds": measured_segnet_metal,
            "measured_n600_segnet_seconds_saved": (
                measured_segnet_cpu - measured_segnet_metal
                if measured_segnet_metal is not None
                else None
            ),
            "derived_pose_n600_seconds_from_n96_share": derived_pose_n600,
            "projected_combined_n600_seconds_with_pose_unchanged": projected_combined,
            "projected_combined_time_delta_vs_372_6s": (
                projected_combined - 372.6 if projected_combined is not None else None
            ),
            "projection_note": "Pose and 372.6s combined baselines are DERIVED from measured n96; SegNet rows above are measured by this n600 probe",
        },
        "native_margin_adaptive_candidate_admitted": admitted,
        "custom_metal_exact_integer_candidate_gate": metal_exact_candidate,
        "verdict": (
            "MARGIN_ADAPTIVE_MIXED_PRECISION_NATIVE_CANDIDATE_ADMITTED"
            if admitted
            else (
                "EXACT_INT_METAL_CANDIDATE_ONLY_NO_MARGIN_ADAPTIVE_PHYSICAL_WIDTH_REDUCTION"
                if metal_exact_candidate
                else "NO_ADMITTED_MARGIN_ADAPTIVE_NATIVE_PROFILE_IN_LADDER"
            )
        ),
        "verdict_scope": (
            "n600 SOURCE-CORPUS INSTANCE: frozen SegNet, capped per-layer frozen-weight-L1 exact-int64 profiles, "
            "per-output-channel weight scales, dynamic per-layer activation scale, frozen ordered (4,0)->0 tie rule; "
            "custom Metal host and supplied finite profile ladder"
        ),
        "pointer_moved": False,
        "score_claim": False,
    }


def _child_profile(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        canary = mx.sum(mx.arange(8, dtype=mx.float32))
        mx.eval(canary)
        if float(canary.item()) != 28.0:
            raise RuntimeError("evaluated Metal canary returned wrong value")
    except Exception as exc:
        return {
            "status": "BLOCKED_NOT_MEASURED",
            "blocker": f"evaluated Metal unavailable: {type(exc).__name__}: {exc}",
            "verdict_scope": "ENVIRONMENT: no evaluated MLX Metal device",
        }
    calibration, _, _, _, precursor = _load_calibration(
        args.calibration_receipt,
        bits=26,
        integer_precursor_path=args.integer_precursor_receipt,
    )
    rule = _selected_tie_snap_rule(precursor)
    if not rule or rule.get("kind") != "ordered_class_pair":
        raise ValueError("margin-adaptive probe requires the frozen ordered class-pair precursor")
    model, _ = _load_segnet()
    cap = int(args.child_profile.removeprefix("cap"))
    precision = derive_capped_precision_map(model, cap_bits=cap)
    adapter, manifest = build_metal_margin_adaptive_int64_segnet_adapter(
        model,
        precision_by_path=precision,
        operator_absmax=calibration,
        require_opt_in=False,
    )
    frame1 = stored_npy_memmap(args.gt_cache, "gt_f1.npy")
    digest = hashlib.sha256()
    seconds: list[float] = []
    for pair_index in range(args.pair_start, args.pair_stop):
        _, nhwc = _seg_input(np.asarray(frame1[pair_index]))
        value = mx.array(nhwc, dtype=mx.float32)
        tick = time.perf_counter()
        candidate = adapter(value)
        decision = class_pair_tie_snap_argmax_mlx(
            candidate,
            epsilon=float(rule["epsilon"]),
            winner_class=int(rule["winner_class"]),
            runner_class=int(rule["runner_class"]),
        )
        mx.eval(candidate, decision)
        seconds.append(time.perf_counter() - tick)
        argmax = np.asarray(decision)[0]
        digest.update(f"{pair_index}:{_hash_array(argmax)}\n".encode("ascii"))
    return {
        "status": "MEASURED",
        "trial_index": int(args.trial_index),
        "profile": args.child_profile,
        "precision_map_sha256": manifest["precision_map_sha256"],
        "argmax_corpus_sha256": digest.hexdigest(),
        "median_seconds_per_pair": statistics.median(seconds),
        "total_seconds": float(sum(seconds)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        canary = mx.sum(mx.arange(8, dtype=mx.float32))
        mx.eval(canary)
        if float(canary.item()) != 28.0:
            raise RuntimeError("evaluated Metal canary returned wrong value")
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "summary": {
                "status": "BLOCKED_NOT_MEASURED",
                "blocker": f"evaluated Metal unavailable: {type(exc).__name__}: {exc}",
                "verdict_scope": "ENVIRONMENT: no evaluated MLX Metal device",
                "pointer_moved": False,
                "score_claim": False,
            },
        }
    import torch

    args.output = args.output.resolve()
    args.gt_cache = args.gt_cache.resolve()
    args.calibration_receipt = args.calibration_receipt.resolve()
    args.integer_precursor_receipt = args.integer_precursor_receipt.resolve()
    args.uniform_nogo_receipt = args.uniform_nogo_receipt.resolve()
    if args.pair_start != 0 or args.pair_stop != N600:
        raise ValueError("decisive margin-adaptive probe requires exact pairs 0..599")
    if args.n_processes < 1:
        raise ValueError("n_processes must be positive")
    caps = _parse_caps(args.profile_caps)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    preflight = _storage_preflight(args.output)
    uniform_nogo = _load_uniform_nogo(args.uniform_nogo_receipt, gt_cache=args.gt_cache)
    calibration, calibration_receipt, _, _, precursor = _load_calibration(
        args.calibration_receipt,
        bits=26,
        integer_precursor_path=args.integer_precursor_receipt,
    )
    rule = _selected_tie_snap_rule(precursor)
    if not rule or rule.get("kind") != "ordered_class_pair":
        raise ValueError("margin-adaptive probe requires the admitted frozen class-pair predecessor")
    model, weights = _load_segnet()
    maps = {cap: derive_capped_precision_map(model, cap_bits=cap) for cap in caps}
    adapters: dict[str, Any] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for cap in caps:
        name = _profile_name(cap)
        adapters[name], manifests[name] = build_metal_margin_adaptive_int64_segnet_adapter(
            model,
            precision_by_path=maps[cap],
            operator_absmax=calibration,
            require_opt_in=False,
        )
    custody = {
        "probe_sha256": sha256_file(Path(__file__)),
        "allocator_module_sha256": sha256_file(
            REPO / "src/tac/local_acceleration/margin_adaptive_mixed_precision.py"
        ),
        "gt_cache_sha256": sha256_file(args.gt_cache),
        "segnet_weights_sha256": sha256_file(weights),
        "uniform_nogo_receipt_sha256": sha256_file(args.uniform_nogo_receipt),
        "uniform_nogo_fingerprint": uniform_nogo.get("fingerprint"),
        "calibration_receipt_sha256": sha256_file(args.calibration_receipt),
        "calibration_receipt_fingerprint": calibration_receipt.get("fingerprint"),
        "integer_precursor_receipt_sha256": sha256_file(args.integer_precursor_receipt),
        "integer_precursor_fingerprint": precursor.get("fingerprint"),
    }
    contract = {
        "pair_start": int(args.pair_start),
        "pair_stop": int(args.pair_stop),
        "design_split": [0, DESIGN_STOP],
        "second_validation_split": [DESIGN_STOP, N600],
        "profile_caps": list(caps),
        "profile_assignment": "bits_l=min(profile_cap, frozen_weight_l1_signed_int64_safe_ceiling_l)",
        "profile_selection": "minimum design measured Metal median seconds/pair among design-exact and design-certified profiles; MAC-weighted logical bits tie-break; no second-validation reselection",
        "per_channel_weight_scales": True,
        "per_layer_dynamic_activation_scales": True,
        "integer_operand_storage_buckets": [8, 16, 32],
        "accumulation": "exact_signed_int64",
        "finalization": "one fp32 dequantization and bias per Conv2d output",
        "decision_rule": precursor["contract"]["decision_rule"],
        "bound_kind": BOUND_KIND,
        "unseen_input_ibp_claim": False,
        "spatial_waterfill_native_execution_claim": False,
        "native_integer_speed_claim": True,
        "n_processes": int(args.n_processes),
        "checkpoint_every_pairs": 1,
        "resumable_from_disk": True,
        "stage_checkpoints_preserved": True,
        "score_claim": False,
    }
    fingerprint = _fingerprint(
        {
            "schema": SCHEMA,
            "contract": contract,
            "custody": custody,
            "precision_maps": {name: manifest["precision_map_sha256"] for name, manifest in manifests.items()},
        }
    )
    if args.resume and args.output.is_file():
        receipt = json.loads(args.output.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA or receipt.get("fingerprint") != fingerprint:
            raise ValueError("resume receipt fingerprint differs")
    else:
        receipt = {
            "schema": SCHEMA,
            "lane_id": "margin_adaptive_mixed_precision",
            "task_id": 494,
            "axis": "[macOS M5-Max custom Metal local candidate; one-thread CPU-Torch control]",
            "score_claim": False,
            "pointer_moved": False,
            "promotion_eligible": False,
            "git_head": _git_head(),
            "host": platform.node(),
            "argv": list(sys.argv),
            "storage_preflight": preflight,
            "contract": contract,
            "custody": custody,
            "fingerprint": fingerprint,
            "profiles": {},
            "rows": [],
            "trials": [],
        }
        receipt["summary"] = _summarize(receipt)
        atomic_json(args.output, receipt)
    frame1 = stored_npy_memmap(args.gt_cache, "gt_f1.npy")
    cached_labels = stored_npy_memmap(args.gt_cache, "lstars.npy")
    cached_margins = stored_npy_memmap(args.gt_cache, "margins.npy")
    completed = {int(row["pair_index"]) for row in receipt["rows"]}
    if completed and completed != set(range(args.pair_start, args.pair_start + len(completed))):
        raise ValueError("resume rows are not a contiguous prefix")
    macs: dict[str, int] | None = None
    if receipt.get("macs_by_path"):
        macs = {str(path): int(value) for path, value in receipt["macs_by_path"].items()}
    started = time.perf_counter()
    with torch.inference_mode():
        for pair_index in range(args.pair_start, args.pair_stop):
            if pair_index in completed:
                continue
            cpu_input, nhwc = _seg_input(np.asarray(frame1[pair_index]))
            tick = time.perf_counter()
            if macs is None:
                macs, reference_logits = _conv_macs_and_logits(model, cpu_input)
                receipt["macs_by_path"] = dict(sorted(macs.items()))
                for name, manifest in manifests.items():
                    precision = manifest["precision_by_path"]
                    receipt["profiles"][name] = {
                        "cap_bits": int(name.removeprefix("cap")),
                        "mac_weighted_average_bits": weighted_average_bits(precision, macs),
                        "mac_weighted_average_storage_bits": weighted_average_bits(
                            manifest["integer_storage_bits_by_path"],
                            macs,
                        ),
                        "manifest": manifest,
                    }
            else:
                reference_logits = model(cpu_input)
            reference_seconds = time.perf_counter() - tick
            reference_np = reference_logits.detach().cpu().numpy().astype(np.float32, copy=False)
            reference_argmax = np.argmax(reference_np, axis=1)[0]
            top2 = np.partition(reference_np, kth=-2, axis=1)[:, -2:, :, :]
            margin = np.diff(np.sort(top2, axis=1), axis=1)[0, 0]
            value = mx.array(nhwc, dtype=mx.float32)
            profile_rows: dict[str, dict[str, Any]] = {}
            strict_profiles: list[ProfileCertificate] = []
            exact_profiles: list[ProfileCertificate] = []
            ordered_names = sorted(
                receipt["profiles"],
                key=lambda name: (
                    float(receipt["profiles"][name]["mac_weighted_average_bits"]),
                    name,
                ),
            )
            rotation = pair_index % len(ordered_names)
            execution_names = ordered_names[rotation:] + ordered_names[:rotation]
            for name in execution_names:
                tick = time.perf_counter()
                candidate = adapters[name](value)
                decision = class_pair_tie_snap_argmax_mlx(
                    candidate,
                    epsilon=float(rule["epsilon"]),
                    winner_class=int(rule["winner_class"]),
                    runner_class=int(rule["runner_class"]),
                )
                mx.eval(candidate, decision)
                metal_seconds = time.perf_counter() - tick
                candidate_nhwc = np.asarray(candidate).astype(np.float32, copy=False)
                candidate_np = np.ascontiguousarray(candidate_nhwc.transpose(0, 3, 1, 2))
                decided = np.asarray(decision)[0]
                error = np.abs(candidate_np - reference_np)
                strict_mask = interval_argmax_certificate_mask(
                    reference_np,
                    error,
                    expected_argmax=reference_argmax[np.newaxis, ...],
                    class_axis=1,
                )[0]
                plain = np.argmax(candidate_np, axis=1)[0]
                exact_mask = decided == reference_argmax
                frozen_tie_rule_mask = (decided != plain) & exact_mask
                hybrid_certificate_mask = strict_mask | frozen_tie_rule_mask
                flips = ~exact_mask
                average_bits = float(receipt["profiles"][name]["mac_weighted_average_bits"])
                strict_profiles.append(ProfileCertificate(name, average_bits, strict_mask))
                exact_profiles.append(ProfileCertificate(name, average_bits, exact_mask))
                profile_rows[name] = {
                    "flips": int(np.count_nonzero(flips)),
                    "pixels": int(flips.size),
                    "flip_fraction": float(np.mean(flips)),
                    "strict_interval_certified_pixels": int(np.count_nonzero(strict_mask)),
                    "strict_interval_certified_fraction": float(np.mean(strict_mask)),
                    "frozen_tie_rule_certified_pixels": int(
                        np.count_nonzero(frozen_tie_rule_mask)
                    ),
                    "strict_interval_or_frozen_tie_rule_certified_pixels": int(
                        np.count_nonzero(hybrid_certificate_mask)
                    ),
                    "strict_interval_or_frozen_tie_rule_certified_fraction": float(
                        np.mean(hybrid_certificate_mask)
                    ),
                    "max_abs_logit_error": float(np.max(error)),
                    "candidate_argmax_sha256": _hash_array(decided),
                    "candidate_logits_sha256": _hash_array(candidate_nhwc),
                    "metal_seconds": float(metal_seconds),
                }
            strict_waterfill = solve_finite_profile_waterfill(strict_profiles)
            exact_waterfill = solve_finite_profile_waterfill(exact_profiles)
            average_bits = [
                float(receipt["profiles"][name]["mac_weighted_average_bits"])
                for name in strict_waterfill.profile_order
            ]
            row = {
                "pair_index": pair_index,
                "split": "design" if pair_index < DESIGN_STOP else "second_validation",
                "pixels": int(reference_argmax.size),
                "reference_seconds": float(reference_seconds),
                "reference_argmax_sha256": _hash_array(reference_argmax),
                "reference_logits_sha256": _hash_array(reference_np),
                "reference_margin_min": float(np.min(margin)),
                "cache_argmax_mismatch_pixels": int(
                    np.count_nonzero(reference_argmax != np.asarray(cached_labels[pair_index]))
                ),
                "cache_margin_max_abs_delta": float(
                    np.max(np.abs(margin - np.asarray(cached_margins[pair_index])))
                ),
                "profiles": profile_rows,
                "profile_execution_order": execution_names,
                "strict_waterfill": {
                    **strict_waterfill.to_summary(),
                    "selected_bits_sum": float(
                        (strict_waterfill.average_selected_bits or 0.0)
                        * np.count_nonzero(strict_waterfill.certified_mask)
                    ),
                    "margin_bands": _margin_band_rows(
                        margin,
                        strict_waterfill.selected_profile_index,
                        average_bits,
                    ),
                },
                "exact_observed_waterfill": {
                    **exact_waterfill.to_summary(),
                    "selected_bits_sum": float(
                        (exact_waterfill.average_selected_bits or 0.0)
                        * np.count_nonzero(exact_waterfill.certified_mask)
                    ),
                    "label_dependent_diagnostic_only": True,
                },
            }
            receipt["rows"].append(row)
            receipt["summary"] = _summarize(receipt)
            receipt["last_completed_pair"] = pair_index
            receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
            atomic_json(args.output, receipt)
    receipt["summary"] = _summarize(receipt)
    atomic_json(args.output, receipt)
    atomic_json(_stage_path(args.output, "search_complete"), receipt)
    selected = receipt["summary"].get("design_selected_profile")
    if selected is not None:
        while len(receipt["trials"]) < args.n_processes:
            trial_index = len(receipt["trials"])
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-profile",
                str(selected),
                "--trial-index",
                str(trial_index),
                "--pair-start",
                str(args.pair_start),
                "--pair-stop",
                str(args.pair_stop),
                "--gt-cache",
                str(args.gt_cache),
                "--calibration-receipt",
                str(args.calibration_receipt),
                "--integer-precursor-receipt",
                str(args.integer_precursor_receipt),
                "--uniform-nogo-receipt",
                str(args.uniform_nogo_receipt),
            ]
            process = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
            if process.returncode not in (0, 3):
                raise RuntimeError(f"selected-profile child failed: {process.stderr[-1600:]}")
            trial = json.loads(process.stdout.strip().splitlines()[-1])
            receipt["trials"].append(trial)
            receipt["summary"] = _summarize(receipt)
            atomic_json(args.output, receipt)
            atomic_json(_stage_path(args.output, f"trial_{trial_index:02d}"), trial)
    receipt["summary"] = _summarize(receipt)
    receipt["completed"] = receipt["summary"].get("status") == "MEASURED"
    receipt["frontier_math_precision_manifest"] = {
        "schema": "frontier_math_precision_manifest.v1",
        "bound_kind": BOUND_KIND,
        "error_budget": "strict top1/rival output interval; corpus-observed per-pixel radii",
        "profiles": {
            name: {
                "precision_map_sha256": row["manifest"]["precision_map_sha256"],
                "precision_by_path": row["manifest"]["precision_by_path"],
                "mac_weighted_average_bits": row["mac_weighted_average_bits"],
                "mac_weighted_average_storage_bits": row[
                    "mac_weighted_average_storage_bits"
                ],
                "measured_cost_seconds_per_pair": receipt["summary"]["profiles"][name]["full"][
                    "metal_median_seconds_per_pair"
                ],
                "maximum_corpus_observed_logit_error": receipt["summary"]["profiles"][name][
                    "full"
                ]["maximum_abs_logit_error"],
            }
            for name, row in receipt["profiles"].items()
        },
    }
    atomic_json(args.output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-stop", type=int, default=N600)
    parser.add_argument(
        "--profile-caps",
        default=",".join(map(str, DEFAULT_PROFILE_CAPS)),
    )
    parser.add_argument("--n-processes", type=int, default=10)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--calibration-receipt", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--integer-precursor-receipt",
        type=Path,
        default=DEFAULT_INTEGER_PRECURSOR,
    )
    parser.add_argument("--uniform-nogo-receipt", type=Path, default=DEFAULT_UNIFORM_NOGO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child-profile", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--trial-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_profile:
        payload = _child_profile(args)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload.get("status") == "MEASURED" else 3
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    status = payload["summary"].get("status")
    return 0 if status == "MEASURED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
