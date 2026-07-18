#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Governed read-only reproduction of the v10 frame-195 numerical mismatch."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as torch_functional

try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.boundary_math.power_diagram_witness import (  # noqa: E402
    affine_head_to_power_diagram,
    decode_pdw1,
    encode_pdw1,
    measure_f32_target_parity,
    open_stored_npy_memmap,
    power_assign,
    power_scores,
    sha256_file,
)
from tac.witness_control.factorized_features import load_frozen_segnet_cpu  # noqa: E402
from tools.harvest_v10_power_diagram_blocked_prefix import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_FEATURE_CACHE_SHA256,
    EXPECTED_PREFIX_SAMPLES,
    VerifiedFile,
    _absolute,
    _assert_unchanged,
    _is_relative_to,
    _verify_file,
    atomic_write_json_no_overwrite,
    open_and_validate_feature_cache,
    validate_blocked_checkpoint,
    validate_durable_output,
    validate_historical_lineage,
)
from tools.harvest_v10_power_diagram_blocked_prefix import (  # noqa: E402
    REPO_ROOT as HARVEST_REPO_ROOT,
)
from tools.v10_power_diagram_blocked_evidence import (  # noqa: E402
    DEFAULT_TORCH_INTEROP_THREADS,
    DEFAULT_TORCH_THREADS,
    EXPECTED_CAMERA_HWC,
    EXPECTED_CLASSES,
    EXPECTED_HEAD_RANK,
    EXPECTED_PAIRS,
    EXPECTED_SEG_HW,
    FEATURE_CACHE_NAME,
    PROGRESS_CHECKPOINT_NAME,
    quotient_convolution,
)

SCHEMA: Final = "v10_power_diagram_frame195_diagnostic.v1"
STATUS: Final = "MEASURED_REPRODUCTION_FRAME195_DIAGNOSTIC"
MEASURED_REPRODUCTION: Final = "MEASURED_REPRODUCTION"
MEASURED_PRESERVED_STATE: Final = "MEASURED_PRESERVED_STATE"
FRAME_INDEX: Final = 195
PIXEL_Y: Final = 214
PIXEL_X: Final = 112


def native_f32_power_scores(point: np.ndarray, sites: np.ndarray, weights: np.ndarray) -> np.ndarray:
    z = np.asarray(point, dtype=np.float32)
    s = np.asarray(sites, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if z.shape != (s.shape[1],) or w.shape != (s.shape[0],):
        raise ValueError("native-f32 power geometry mismatch")
    dot = np.sum(s * z[None, :], axis=1, dtype=np.float32)
    norm = np.sum(s * s, axis=1, dtype=np.float32)
    return np.asarray(np.float32(2.0) * dot + w - norm, dtype=np.float32)


def _margin(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("margin requires a finite score vector")
    top = np.partition(values, -2)[-2:]
    return float(top[1] - top[0])


def analyze_pixel(
    *,
    logits: np.ndarray,
    quotient_point: np.ndarray,
    target: Any,
    cached_lstar: int,
    parity_receipt: Any,
) -> dict[str, Any]:
    logits64 = np.asarray(logits, dtype=np.float64)
    point64 = np.asarray(quotient_point, dtype=np.float64)
    if logits64.shape != (EXPECTED_CLASSES,) or point64.shape != (EXPECTED_HEAD_RANK,):
        raise ValueError("pixel diagnostic geometry drift")
    if not np.isfinite(logits64).all() or not np.isfinite(point64).all():
        raise ValueError("pixel diagnostic values must be finite")
    generic_scores = np.asarray(power_scores(point64[None, :], target)[0], dtype=np.float64)
    native_scores = native_f32_power_scores(point64, target.sites, target.weights)
    generic_class = int(power_assign(point64[None, :], target)[0])
    native_class = int(target.class_ids[int(np.argmax(native_scores))])
    torch_class = int(np.argmax(logits64))
    return {
        "label": MEASURED_REPRODUCTION,
        "frame": FRAME_INDEX,
        "pixel_y": PIXEL_Y,
        "pixel_x": PIXEL_X,
        "cached_lstar": int(cached_lstar),
        "cpu_torch": {
            "logits": logits64.tolist(),
            "argmax": torch_class,
            "winner_margin": _margin(logits64),
        },
        "rank4_quotient": point64.tolist(),
        "generic_f64_power": {
            "scores": generic_scores.tolist(),
            "argmax": generic_class,
            "winner_margin": _margin(generic_scores),
        },
        "native_f32_power": {
            "scores": native_scores.astype(np.float64).tolist(),
            "argmax": native_class,
            "winner_margin": _margin(native_scores),
        },
        "ties_and_disagreement": {
            "generic_exact_top_tie": _margin(generic_scores) == 0.0,
            "native_exact_top_tie": _margin(native_scores) == 0.0,
            "generic_vs_native_disagree": generic_class != native_class,
            "generic_vs_torch_disagree": generic_class != torch_class,
            "native_vs_torch_disagree": native_class != torch_class,
        },
        "serialized_target_parity_uncertainty": {
            "label": parity_receipt.authority_label,
            "sample_count": parity_receipt.sample_count,
            "mismatch_count": parity_receipt.mismatch_count,
            "sample_agreement": parity_receipt.sample_agreement,
            "max_pair_score_error": parity_receipt.max_pair_score_error,
            "minimum_affine_winner_margin": parity_receipt.minimum_affine_winner_margin,
            "f32_tie_uncertain_count": parity_receipt.f32_tie_uncertain_count,
            "exact_on_samples": parity_receipt.exact_on_samples,
            "boundary_exactness": parity_receipt.boundary_exactness,
        },
    }


def validate_receipt_authority(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != SCHEMA or receipt.get("status") != STATUS:
        raise ValueError("diagnostic schema/status drift")
    authority = receipt.get("authority", {})
    expected = {
        "new_run_label": MEASURED_REPRODUCTION,
        "checkpoint_aggregate_label": MEASURED_PRESERVED_STATE,
        "one_frame_one_pixel_only": True,
        "n600_rerun": False,
        "through_r_authority": False,
        "receiver_authority": False,
        "score_authority": False,
        "promotion_eligible": False,
        "cleanup_performed": False,
        "ssd_write_performed": False,
    }
    for key, value in expected.items():
        actual = authority.get(key)
        if (isinstance(value, bool) and actual is not value) or (not isinstance(value, bool) and actual != value):
            raise ValueError(f"diagnostic authority field {key!r} drift")
    preserved = receipt.get("preserved_checkpoint_state", {})
    if (
        preserved.get("label") != MEASURED_PRESERVED_STATE
        or preserved.get("next_canonical_frame") != 195
        or preserved.get("sample_count") != EXPECTED_PREFIX_SAMPLES
        or preserved.get("power_target_mismatch_count") != 1
        or preserved.get("cpu_torch_forward_mismatch_count") != 0
    ):
        raise ValueError("preserved checkpoint authority drift")
    reproduction = receipt.get("reproduction", {})
    if (
        reproduction.get("label") != MEASURED_REPRODUCTION
        or reproduction.get("frame") != FRAME_INDEX
        or reproduction.get("pixel_y") != PIXEL_Y
        or reproduction.get("pixel_x") != PIXEL_X
    ):
        raise ValueError("new reproduction scope drift")
    wrapper = receipt.get("outer_wrapper_custody", {})
    if wrapper.get("label") != "OPERATOR_SUPPLIED_NOT_INNER_RUNTIME_MEASUREMENT":
        raise ValueError("outer wrapper custody label drift")
    if wrapper.get("original_governed_run_limits_claimed") is not False:
        raise ValueError("original safe_run cap/timeout must not be claimed")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--gt-cache", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--historical-container", required=True, type=Path)
    parser.add_argument("--historical-manifest", required=True, type=Path)
    parser.add_argument("--current-tombstone", required=True, type=Path)
    parser.add_argument("--wrapper-launcher", required=True, type=Path)
    parser.add_argument("--wrapper-timeout-seconds", required=True, type=int)
    parser.add_argument("--wrapper-memory-limit-mb", required=True, type=int)
    parser.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--torch-interop-threads", type=int, default=DEFAULT_TORCH_INTEROP_THREADS)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _source_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _rehash_verified_inputs(verified_inputs: dict[str, VerifiedFile]) -> None:
    for role, verified in verified_inputs.items():
        _assert_unchanged(verified, role=role, verify_hash=True)


def run_diagnostic(args: argparse.Namespace, *, exact_inner_argv: list[str]) -> dict[str, Any]:
    output = validate_durable_output(args.output)
    if HARVEST_REPO_ROOT != REPO_ROOT:
        raise RuntimeError("diagnostic/harvester repository-root drift")
    checkpoint = _absolute(args.checkpoint, option="checkpoint", must_exist=True)
    feature_cache_path = _absolute(args.feature_cache, option="feature-cache", must_exist=True)
    gt_cache = _absolute(args.gt_cache, option="gt-cache", must_exist=True)
    upstream_root = _absolute(args.upstream_root, option="upstream-root", must_exist=True)
    historical_container = _absolute(args.historical_container, option="historical-container", must_exist=True)
    historical_manifest_path = _absolute(args.historical_manifest, option="historical-manifest", must_exist=True)
    current_tombstone = _absolute(args.current_tombstone, option="current-tombstone", must_exist=True)
    wrapper_launcher = _absolute(args.wrapper_launcher, option="wrapper-launcher", must_exist=True)
    if checkpoint.name != PROGRESS_CHECKPOINT_NAME or feature_cache_path.name != FEATURE_CACHE_NAME:
        raise ValueError("checkpoint/cache names are noncanonical")
    if checkpoint.parent != feature_cache_path.parent:
        raise ValueError("checkpoint/cache must share the preserved scratch directory")
    if args.wrapper_timeout_seconds <= 0 or args.wrapper_memory_limit_mb <= 0:
        raise ValueError("explicit wrapper timeout/memory values must be positive")
    if wrapper_launcher != (REPO_ROOT / "tools/safe_run.py").resolve(strict=True):
        raise ValueError("wrapper launcher must be the canonical tools/safe_run.py")

    wrapper_verified = _verify_file(
        wrapper_launcher,
        expected_sha256=sha256_file(wrapper_launcher),
        role="wrapper_launcher",
    )
    verified_checkpoint = _verify_file(checkpoint, expected_sha256=EXPECTED_CHECKPOINT_SHA256, role="checkpoint")
    verified_cache = _verify_file(
        feature_cache_path,
        expected_sha256=EXPECTED_FEATURE_CACHE_SHA256,
        role="feature_cache",
    )
    manifest, lineage = validate_historical_lineage(
        historical_container=historical_container,
        historical_manifest=historical_manifest_path,
        current_tombstone=current_tombstone,
    )
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    state, identity_files = validate_blocked_checkpoint(
        checkpoint_payload,
        upstream_root=upstream_root,
        gt_cache=gt_cache,
        historical_manifest=manifest,
        lineage_files=lineage,
    )
    cache = open_and_validate_feature_cache(feature_cache_path)
    del cache
    lstars = open_stored_npy_memmap(gt_cache, "lstars")
    gt_f1 = open_stored_npy_memmap(gt_cache, "gt_f1")
    if tuple(lstars.shape) != (EXPECTED_PAIRS, *EXPECTED_SEG_HW):
        raise ValueError("lstars geometry drift")
    if tuple(gt_f1.shape) != (EXPECTED_PAIRS, *EXPECTED_CAMERA_HWC):
        raise ValueError("gt_f1 geometry drift")

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != args.torch_threads or torch.get_num_interop_threads() != args.torch_interop_threads:
        raise RuntimeError("effective CPU thread geometry drift")
    prepend_paths(upstream_root)
    for module_name in ("frame_utils", "modules"):
        existing = sys.modules.get(module_name)
        if existing is not None and not _is_relative_to(Path(existing.__file__).resolve(), upstream_root):
            raise RuntimeError(f"refusing pre-imported foreign upstream module {module_name!r}")
    model = load_frozen_segnet_cpu(upstream_root)
    frame_utils = importlib.import_module("frame_utils")
    modules = importlib.import_module("modules")
    for imported in (frame_utils, modules):
        if not _is_relative_to(Path(imported.__file__).resolve(), upstream_root):
            raise RuntimeError(f"upstream import escaped explicit root: {imported.__file__}")
    if tuple(frame_utils.camera_size) != (EXPECTED_CAMERA_HWC[1], EXPECTED_CAMERA_HWC[0]):
        raise RuntimeError("upstream camera_size contract drift")
    if tuple(frame_utils.segnet_model_input_size) != (EXPECTED_SEG_HW[1], EXPECTED_SEG_HW[0]):
        raise RuntimeError("upstream SegNet input geometry drift")
    final_head = model.segmentation_head[0]
    if not isinstance(final_head, torch.nn.Conv2d):
        raise RuntimeError("frozen final head is not Conv2d")
    frozen_head = affine_head_to_power_diagram(
        final_head.weight.detach().cpu().numpy(),
        final_head.bias.detach().cpu().numpy(),
    )
    pdw1 = encode_pdw1(frozen_head.target)
    if encode_pdw1(decode_pdw1(pdw1)) != pdw1:
        raise RuntimeError("frozen PDW1 parse-back drift")

    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 1:
            raise RuntimeError("final-head hook geometry drift")
        captured.append(inputs[0].detach())

    hook = final_head.register_forward_pre_hook(capture)
    try:
        frame = np.array(gt_f1[FRAME_INDEX], dtype=np.uint8, copy=True, order="C")
        batch = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
        if tuple(batch.shape) != (1, 1, 3, EXPECTED_CAMERA_HWC[0], EXPECTED_CAMERA_HWC[1]):
            raise RuntimeError("frame-195 batch geometry drift")
        with torch.inference_mode():
            logits = model(model.preprocess_input(batch))
            if len(captured) != 1:
                raise RuntimeError("final-head input capture count drift")
            quotient = quotient_convolution(
                captured[0],
                frozen_head.quotient_basis,
                head_weight_shape=tuple(int(value) for value in final_head.weight.shape),
                stride=tuple(int(value) for value in final_head.stride),
                padding=tuple(int(value) for value in final_head.padding),
                dilation=tuple(int(value) for value in final_head.dilation),
                groups=int(final_head.groups),
            )
        if tuple(logits.shape) != (1, EXPECTED_CLASSES, *EXPECTED_SEG_HW):
            raise RuntimeError("frame-195 logits geometry drift")
        if tuple(quotient.shape) != (1, EXPECTED_HEAD_RANK, *EXPECTED_SEG_HW):
            raise RuntimeError("frame-195 quotient geometry drift")
        unfolded = torch_functional.unfold(
            captured[0],
            kernel_size=tuple(int(value) for value in final_head.kernel_size),
            dilation=tuple(int(value) for value in final_head.dilation),
            padding=tuple(int(value) for value in final_head.padding),
            stride=tuple(int(value) for value in final_head.stride),
        )
        flat_index = PIXEL_Y * EXPECTED_SEG_HW[1] + PIXEL_X
        patch = unfolded[0, :, flat_index].cpu().numpy()[None, :]
        parity = measure_f32_target_parity(patch, frozen_head)
        reproduction = analyze_pixel(
            logits=logits[0, :, PIXEL_Y, PIXEL_X].cpu().numpy(),
            quotient_point=quotient[0, :, PIXEL_Y, PIXEL_X].cpu().numpy(),
            target=frozen_head.target,
            cached_lstar=int(lstars[FRAME_INDEX, PIXEL_Y, PIXEL_X]),
            parity_receipt=parity,
        )
    finally:
        hook.remove()
        del lstars
        del gt_f1

    all_verified: dict[str, VerifiedFile] = {
        "checkpoint": verified_checkpoint,
        "feature_cache": verified_cache,
        **identity_files,
    }
    _rehash_verified_inputs({**all_verified, "wrapper_launcher": wrapper_verified})
    receipt = {
        "schema": SCHEMA,
        "status": STATUS,
        "exact_inner_argv": exact_inner_argv,
        "authority": {
            "new_run_label": MEASURED_REPRODUCTION,
            "checkpoint_aggregate_label": MEASURED_PRESERVED_STATE,
            "one_frame_one_pixel_only": True,
            "n600_rerun": False,
            "through_r_authority": False,
            "receiver_authority": False,
            "score_authority": False,
            "promotion_eligible": False,
            "cleanup_performed": False,
            "ssd_write_performed": False,
        },
        "outer_wrapper_custody": {
            "label": "OPERATOR_SUPPLIED_NOT_INNER_RUNTIME_MEASUREMENT",
            "launcher": wrapper_verified.receipt_row(),
            "timeout_seconds": args.wrapper_timeout_seconds,
            "memory_limit_mb": args.wrapper_memory_limit_mb,
            "original_governed_run_limits_claimed": False,
        },
        "custody": {role: verified.receipt_row() for role, verified in sorted(all_verified.items())},
        "current_sources": {
            "diagnostic": _source_receipt(Path(__file__).resolve()),
            "read_only_evidence": _source_receipt(
                (REPO_ROOT / "tools/v10_power_diagram_blocked_evidence.py").resolve()
            ),
            "harvester": _source_receipt((REPO_ROOT / "tools/harvest_v10_power_diagram_blocked_prefix.py").resolve()),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            "device": "cpu",
            "dtype": "torch.float32",
            "batch_size": 1,
            "torch_threads_requested": args.torch_threads,
            "torch_threads_effective": torch.get_num_threads(),
            "torch_interop_threads_requested": args.torch_interop_threads,
            "torch_interop_threads_effective": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "camera_size": list(frame_utils.camera_size),
            "segnet_model_input_size": list(frame_utils.segnet_model_input_size),
        },
        "frozen_target": {
            "label": MEASURED_REPRODUCTION,
            "pdw1_hex": pdw1.hex(),
            "pdw1_bytes": len(pdw1),
            "pdw1_sha256": hashlib.sha256(pdw1).hexdigest(),
            "strict_parseback_byte_identical": True,
            "singular_values": np.asarray(frozen_head.singular_values, dtype=np.float64).tolist(),
            "quotient_rank": frozen_head.target.rank,
        },
        "preserved_checkpoint_state": {
            "label": MEASURED_PRESERVED_STATE,
            "next_canonical_frame": state.next_frame,
            "sample_count": state.statistics.sample_count,
            "power_target_mismatch_count": state.positive_power_mismatches,
            "cpu_torch_forward_mismatch_count": state.positive_forward_mismatches,
            "blocked_reason": state.blocked_reason,
        },
        "reproduction": reproduction,
    }
    validate_receipt_authority(receipt)
    atomic_write_json_no_overwrite(output, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    assert_governed_admission("diagnose_v10_power_diagram_frame195")
    exact_inner_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *(sys.argv[1:] if argv is None else argv),
    ]
    run_diagnostic(args, exact_inner_argv=exact_inner_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
