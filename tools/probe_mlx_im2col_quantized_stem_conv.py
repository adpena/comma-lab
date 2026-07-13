#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MLX im2col + quantized_matmul proof for the frozen SegNet stem.

This is a local training-backend probe, not a native quantized-convolution
claim.  It lowers only the EfficientNet-B2 stem convolution to nine strided
NHWC views, concatenates them into an im2col matrix, and calls MLX's native
weight-quantized matmul.  The remaining SegNet stays fp32.  W8A8 additionally
applies dynamic symmetric activation QDQ with an identity STE; MLX still
receives a floating activation matrix, so that row is explicitly not a native
integer-activation kernel.

The probe reads the preserved v7.5.2 EMA and real n600 cache, performs no
optimizer step, writes an atomic row per quality pair, and resumes only rows
whose run fingerprint matches.  All results are macOS-MLX research signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_mlx_real_n600_precision as P  # noqa: E402

EXPECTED_CHECKPOINT_SHA256 = "ef2c097f98f74dbd16e77c6f7b60f05e0a630b6bd65ee55bf334336c4549c965"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"expected scalar or pair, got {value!r}")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class Candidate:
    name: str
    bits: int
    activation_qdq: bool


CANDIDATES = (
    Candidate("w8_only_affine_g32", 8, False),
    Candidate("w8a8_qdq_affine_g32", 8, True),
    Candidate("w6_only_affine_g32", 6, False),
    Candidate("w4_only_affine_g32", 4, False),
)


def _dynamic_symmetric_int8_ste(value: Any) -> Any:
    import mlx.core as mx

    x = value.astype(mx.float32)
    absmax = mx.max(mx.abs(x))
    scale = mx.maximum(absmax / 127.0, mx.array(1.0e-12, dtype=mx.float32))
    qdq = mx.clip(mx.round(x / scale), -127, 127) * scale
    return x + mx.stop_gradient(qdq - x)


class Im2ColQuantizedStemConv:
    """Dense Conv2d lowered to MLX affine group-quantized matmul."""

    def __init__(self, original: Any, candidate: Candidate, *, group_size: int = 32):
        import mlx.core as mx

        weight = original["weight"]
        if len(weight.shape) != 4:
            raise ValueError(f"expected OHWI weight, got {weight.shape}")
        out_channels, kernel_h, kernel_w, in_channels = map(int, weight.shape)
        if (kernel_h, kernel_w, in_channels) != (3, 3, 3):
            raise ValueError(
                "proof is intentionally bound to the 3x3x3 EfficientNet stem, "
                f"got {(kernel_h, kernel_w, in_channels)}"
            )
        self.candidate = candidate
        self.stride = _pair(original.stride)
        self.padding = _pair(original.padding)
        self.dilation = _pair(original.dilation)
        self.groups = int(original.groups)
        if self.dilation != (1, 1) or self.groups != 1:
            raise ValueError("proof supports dense dilation-1 stem only")
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.logical_k = kernel_h * kernel_w * in_channels
        self.group_size = int(group_size)
        self.padded_k = int(math.ceil(self.logical_k / self.group_size) * self.group_size)
        matrix = mx.reshape(weight, (out_channels, self.logical_k)).astype(mx.float32)
        if self.padded_k != self.logical_k:
            matrix = mx.pad(matrix, ((0, 0), (0, self.padded_k - self.logical_k)))
        self.qweight, self.scales, self.qbiases = mx.quantize(
            matrix,
            group_size=self.group_size,
            bits=candidate.bits,
            mode="affine",
        )
        self.bias = original.get("bias")

    def __call__(self, value: Any) -> Any:
        import mlx.core as mx

        x = value.astype(mx.float32)
        if self.candidate.activation_qdq:
            x = _dynamic_symmetric_int8_ste(x)
        pad_h, pad_w = self.padding
        if pad_h or pad_w:
            x = mx.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        batch, height, width, channels = map(int, x.shape)
        if channels != self.in_channels:
            raise ValueError(f"input channels {channels} != stem channels {self.in_channels}")
        stride_h, stride_w = self.stride
        out_h = (height - self.kernel_h) // stride_h + 1
        out_w = (width - self.kernel_w) // stride_w + 1
        patches = [
            x[
                :,
                kh : kh + out_h * stride_h : stride_h,
                kw : kw + out_w * stride_w : stride_w,
                :,
            ]
            for kh in range(self.kernel_h)
            for kw in range(self.kernel_w)
        ]
        matrix = mx.reshape(mx.concatenate(patches, axis=-1), (-1, self.logical_k))
        if self.padded_k != self.logical_k:
            matrix = mx.pad(matrix, ((0, 0), (0, self.padded_k - self.logical_k)))
        output = mx.quantized_matmul(
            matrix,
            self.qweight,
            self.scales,
            self.qbiases,
            transpose=True,
            group_size=self.group_size,
            bits=self.candidate.bits,
            mode="affine",
        )
        output = mx.reshape(output, (batch, out_h, out_w, self.out_channels))
        if self.bias is not None:
            output = output + self.bias
        return output

    def receipt(self) -> dict[str, Any]:
        return {
            "candidate": asdict(self.candidate),
            "kernel": [self.kernel_h, self.kernel_w],
            "channels": [self.in_channels, self.out_channels],
            "stride": list(self.stride),
            "padding": list(self.padding),
            "logical_inner_dimension": self.logical_k,
            "padded_inner_dimension": self.padded_k,
            "group_size": self.group_size,
            "quantized_weight_dtype": str(self.qweight.dtype),
            "scales_dtype": str(self.scales.dtype),
            "biases_dtype": str(self.qbiases.dtype),
            "native_quantized_convolution_claim": False,
            "native_integer_activation_claim": False,
            "lowering": "9 NHWC strided views -> concatenate -> affine quantized_matmul",
        }


def _cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64, copy=False).reshape(-1)
    cand = candidate.astype(np.float64, copy=False).reshape(-1)
    denominator = math.sqrt(float(np.dot(ref, ref)) * float(np.dot(cand, cand)))
    return float(np.dot(ref, cand) / denominator) if denominator > 0.0 else 0.0


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64, copy=False)
    cand = candidate.astype(np.float64, copy=False)
    denominator = float(np.linalg.norm(ref.reshape(-1)))
    return float(np.linalg.norm((cand - ref).reshape(-1)) / denominator) if denominator else 0.0


def _time_operator(operator: Any, value: Any, *, warmup: int, repeats: int) -> dict[str, Any]:
    import mlx.core as mx

    for _ in range(warmup):
        output = operator(value)
        mx.eval(output)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = operator(value)
        mx.eval(output)
        samples.append(time.perf_counter() - start)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "samples_s": samples,
    }


def _fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = (args.run_dir / args.checkpoint).resolve()
    return {
        "schema": "mlx_im2col_quantized_stem_conv.v1",
        "git_sha": _git_sha(),
        "probe_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "gt_cache_sha256": _sha256(args.gt_cache.resolve()),
        "quality_pairs": args.quality_pairs,
        "batch_sizes": args.batch_sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = (args.run_dir / args.checkpoint).resolve()
    gt_cache = args.gt_cache.resolve()
    if _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA does not match the preregistered v7.5.2 EMA")
    fingerprint = _fingerprint(args)
    receipt: dict[str, Any] = {
        "schema": "mlx_im2col_quantized_stem_conv.v1",
        "written_at_utc": _utc(),
        "lane_id": "lane_precision_backend_matrix_20260713",
        "axis": "[macOS-MLX research-signal; NON-PROMOTABLE]",
        "research_only": True,
        "training_launched": False,
        "authority": {
            "score_claim": False,
            "pointer_moved": False,
            "native_quantized_convolution_claim": False,
        },
        "fingerprint": fingerprint,
        "provenance": {
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "gt_cache": str(gt_cache.relative_to(REPO)),
            "probe_source": str(Path(__file__).resolve().relative_to(REPO)),
        },
        "verdict_scope": (
            "im2col plus MLX affine quantized_matmul for only the frozen SegNet EfficientNet-B2 "
            "3x3 stride-2 stem on the preregistered real receiver states and this M5-class host; "
            "not native quantized conv, not a full-network precision-family verdict"
        ),
    }
    preflight = P._metal_preflight()
    receipt["metal_preflight"] = preflight
    if not preflight["available"]:
        receipt.update(
            {
                "status": "BLOCKED-NOT-MEASURED",
                "operator_receipts": None,
                "timing": None,
                "quality": None,
                "next_command": (
                    "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python "
                    "tools/probe_mlx_im2col_quantized_stem_conv.py "
                    f"--quality-pairs {args.quality_pairs} --batch-sizes "
                    + " ".join(str(value) for value in args.batch_sizes)
                    + f" --warmup {args.warmup} --repeats {args.repeats} --out "
                    + str(args.out.relative_to(REPO))
                ),
            }
        )
        return receipt

    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    partial = args.out.with_name(args.out.stem + ".quality_stage.json")
    with temporary_mlx_device("gpu"):
        state = P._load_real_state_context(checkpoint, gt_cache)
        fp32 = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        fp_stem = fp32.segnet.encoder.model.stem.conv_stem
        candidates: dict[str, tuple[Any, Im2ColQuantizedStemConv]] = {}
        for spec in CANDIDATES:
            adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
            original = adapter.segnet.encoder.model.stem.conv_stem
            lowered = Im2ColQuantizedStemConv(original, spec)
            adapter.segnet.encoder.model.stem.conv_stem = lowered
            candidates[spec.name] = (adapter, lowered)
        arrays = P._iter_mlx_arrays(fp32)
        for adapter, _lowered in candidates.values():
            arrays.extend(P._iter_mlx_arrays(adapter))
        mx.eval(*arrays)

        timing: dict[str, Any] = {}
        for batch_size in args.batch_sizes:
            frames = np.stack(
                [state["post_r_pair"](index)[1] for index in range(batch_size)], axis=0
            ).astype(np.float32)
            value = mx.array(frames, dtype=mx.float32)
            baseline = _time_operator(fp_stem, value, warmup=args.warmup, repeats=args.repeats)
            row: dict[str, Any] = {"fp32_native_conv2d": baseline}
            for name, (_adapter, lowered) in candidates.items():
                measured = _time_operator(lowered, value, warmup=args.warmup, repeats=args.repeats)
                measured["speedup_vs_fp32"] = baseline["median_s"] / measured["median_s"]
                row[name] = measured
            timing[f"batch_{batch_size}"] = row

        prior_rows: list[dict[str, Any]] = []
        if partial.is_file():
            prior = json.loads(partial.read_text())
            if prior.get("fingerprint") != fingerprint:
                raise ValueError("quality-stage fingerprint mismatch; refusing mixed-source resume")
            prior_rows = list(prior.get("rows", []))
        complete = {int(row["pair_index"]) for row in prior_rows}
        rows = prior_rows
        for pair_index in range(args.quality_pairs):
            if pair_index in complete:
                continue
            frame = state["post_r_pair"](pair_index)[1:2]
            value = mx.array(frame, dtype=mx.float32)
            fp_stem_output = fp_stem(value)
            fp_logits = fp32.segnet(value).astype(mx.float32)
            candidate_outputs: dict[str, tuple[Any, Any]] = {}
            to_eval = [fp_stem_output, fp_logits]
            for name, (adapter, lowered) in candidates.items():
                stem_output = lowered(value)
                logits = adapter.segnet(value).astype(mx.float32)
                candidate_outputs[name] = (stem_output, logits)
                to_eval.extend((stem_output, logits))
            mx.eval(*to_eval)
            ref_stem = np.asarray(fp_stem_output, dtype=np.float32)
            ref_logits = np.asarray(fp_logits, dtype=np.float32)
            ref_stem_argmax = ref_stem.argmax(axis=-1)
            ref_class_argmax = ref_logits.argmax(axis=-1)
            metrics: dict[str, Any] = {}
            for name, (stem_output, logits) in candidate_outputs.items():
                stem = np.asarray(stem_output, dtype=np.float32)
                final = np.asarray(logits, dtype=np.float32)
                metrics[name] = {
                    "stem_cosine": _cosine(ref_stem, stem),
                    "stem_relative_l2": _relative_l2(ref_stem, stem),
                    "stem_channel_argmax_flips": int(
                        np.count_nonzero(ref_stem_argmax != stem.argmax(axis=-1))
                    ),
                    "stem_channel_argmax_elements": int(ref_stem_argmax.size),
                    "final_logit_cosine": _cosine(ref_logits, final),
                    "final_logit_relative_l2": _relative_l2(ref_logits, final),
                    "segnet_class_argmax_flips": int(
                        np.count_nonzero(ref_class_argmax != final.argmax(axis=-1))
                    ),
                    "segnet_class_argmax_pixels": int(ref_class_argmax.size),
                }
            rows.append({"pair_index": pair_index, "metrics": metrics})
            rows.sort(key=lambda row: int(row["pair_index"]))
            _atomic_json(
                partial,
                {
                    "schema": "mlx_im2col_quantized_stem_quality_stage.v1",
                    "fingerprint": fingerprint,
                    "rows": rows,
                    "last_completed_at_utc": _utc(),
                },
            )

        aggregates: dict[str, Any] = {}
        for spec in CANDIDATES:
            selected = [row["metrics"][spec.name] for row in rows[: args.quality_pairs]]
            aggregates[spec.name] = {
                "quality_pairs": args.quality_pairs,
                "stem_cosine_min": min(row["stem_cosine"] for row in selected),
                "stem_cosine_median": statistics.median(row["stem_cosine"] for row in selected),
                "stem_channel_argmax_flips": sum(
                    row["stem_channel_argmax_flips"] for row in selected
                ),
                "stem_channel_argmax_elements": sum(
                    row["stem_channel_argmax_elements"] for row in selected
                ),
                "final_logit_cosine_min": min(row["final_logit_cosine"] for row in selected),
                "final_logit_cosine_median": statistics.median(
                    row["final_logit_cosine"] for row in selected
                ),
                "segnet_class_argmax_flips": sum(
                    row["segnet_class_argmax_flips"] for row in selected
                ),
                "segnet_class_argmax_pixels": sum(
                    row["segnet_class_argmax_pixels"] for row in selected
                ),
            }

    receipt.update(
        {
            "status": "MEASURED",
            "operator_receipts": {
                name: lowered.receipt() for name, (_adapter, lowered) in candidates.items()
            },
            "timing": timing,
            "quality": {"aggregate": aggregates, "stage_receipt": str(partial.relative_to(REPO))},
            "interpretation": (
                "A speedup establishes only the selected stem lowering. Full-network and gradient-bearing "
                "speed require separate measurement; W8A8 is activation QDQ plus a W8 matmul kernel."
            ),
        }
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=P.DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default=P.DEFAULT_CHECKPOINT)
    parser.add_argument("--gt-cache", type=Path, default=P.DEFAULT_GT_CACHE)
    parser.add_argument("--quality-pairs", type=int, default=16)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.gt_cache = args.gt_cache.resolve()
    args.out = args.out.resolve()
    if not (1 <= args.quality_pairs <= 600):
        parser.error("--quality-pairs must be in 1..600")
    if not args.batch_sizes or any(value < 1 or value > 64 for value in args.batch_sizes):
        parser.error("--batch-sizes must contain values in 1..64")
    if args.warmup < 1 or args.repeats < 3:
        parser.error("--warmup must be >=1 and --repeats must be >=3")
    if str(args.out).startswith(("/tmp/", "/private/tmp/")):
        parser.error("refusing a temporary durable evidence path")
    payload = run(args)
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
