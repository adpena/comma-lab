#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable n600 W8A8-QDQ frozen-scorer quality and gradient probe.

The candidate uses int8 fake quantization for weights and activations, float32
accumulation, and an identity STE.  It measures emulation cost only; it cannot
claim native int8 convolution speed.  The probe is read-only on the v7.5.2 run
and performs no optimizer update or training launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_mlx_real_n600_precision as P  # noqa: E402

from tac.local_acceleration.mlx_int8_teacher_fakequant import (  # noqa: E402
    instrument_frozen_scorer_w8a8_fakequant,
)
from tac.local_acceleration.mlx_training_precision_probe import (  # noqa: E402
    PrecisionGoBars,
    evaluate_precision_gate,
)

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


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = (args.run_dir / args.checkpoint).resolve()
    gt_cache = args.gt_cache.resolve()
    if _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA does not match the preregistered v7.5.2 EMA")
    receipt: dict[str, Any] = {
        "schema": "int8_teacher_w8a8_qdq_n600.v1",
        "written_at_utc": _utc(),
        "lane_id": "lane_int8_training_rungs_20260713",
        "axis": "[macOS-MLX research-signal; NON-PROMOTABLE]",
        "research_only": True,
        "training_launched": False,
        "authority": {
            "score_claim": False,
            "pointer_moved": False,
            "native_int8_speed_claim": False,
        },
        "provenance": {
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "gt_cache": str(gt_cache.relative_to(REPO)),
            "probe_source": str(Path(__file__).resolve().relative_to(REPO)),
            "probe_source_sha256": _sha256(Path(__file__).resolve()),
            "instrumentation_source": "src/tac/local_acceleration/mlx_int8_teacher_fakequant.py",
            "instrumentation_source_sha256": _sha256(REPO / "src/tac/local_acceleration/mlx_int8_teacher_fakequant.py"),
            "reused_precision_probe_source": "tools/probe_mlx_real_n600_precision.py",
            "reused_precision_probe_source_sha256": _sha256(REPO / "tools/probe_mlx_real_n600_precision.py"),
        },
        "go_bars": PrecisionGoBars().to_dict(),
        "verdict_scope": (
            "W8A8 QDQ with float32 accumulation and identity STE on the exact v7.5.2 EMA, "
            "real n600 states, and this M5-class MLX host; no native int8-kernel, ANE, score, "
            "contest-CPU/CUDA, or promotion transfer"
        ),
    }
    preflight = P._metal_preflight()
    receipt["metal_preflight"] = preflight
    if not preflight["available"]:
        receipt.update(
            {
                "status": "BLOCKED_NOT_MEASURED",
                "quality": None,
                "timing": None,
                "gate": None,
                "reformulation_queue": [
                    "run this exact resumable command in a Metal-entitled host process",
                    "retain float32 accumulation and the explicit W8A8 group receipt",
                    "if cosine fails, waterfill fp32 over the highest-damage scorer operators",
                ],
            }
        )
        return receipt

    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    with temporary_mlx_device("gpu"):
        fp32 = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        int8 = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        int8, instrumentation = instrument_frozen_scorer_w8a8_fakequant(int8)
        arrays = P._iter_mlx_arrays(fp32) + P._iter_mlx_arrays(int8)
        mx.eval(*arrays)
        state = P._load_real_state_context(checkpoint, gt_cache)
        fp_ops = P._make_scorer_ops(fp32, "float32", state)
        int8_ops = P._make_scorer_ops(int8, "float32", state)
        timing_indices = (
            __import__("numpy").linspace(0, 599, args.timing_pairs, dtype=__import__("numpy").int64).tolist()
        )
        fp_timing = P._time_mode(
            pair_indices=timing_indices,
            state=state,
            ops=fp_ops,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        int8_timing = P._time_mode(
            pair_indices=timing_indices,
            state=state,
            ops=int8_ops,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        partial = args.out.with_name(args.out.stem + ".quality_stage.json")
        quality = P._quality_mode(
            candidate_name="int8_w8a8_qdq_fp32_accum_ste",
            state=state,
            fp32_ops=fp_ops,
            candidate_ops=int8_ops,
            quality_pairs=args.quality_pairs,
            partial_path=partial,
        )
        gate = evaluate_precision_gate(
            fp32_seconds=fp_timing["median_forward_backward_s"],
            candidate_seconds=int8_timing["median_forward_backward_s"],
            global_cosine=quality["global_gradient_cosine"],
            pair_cosine_min=quality["pair_gradient_metrics"]["cosine_min"],
            quality_pairs=quality["n_pairs"],
        )
    receipt.update(
        {
            "status": "MEASURED",
            "instrumentation": instrumentation,
            "quality": quality,
            "timing": {"fp32": fp_timing, "int8_qdq_emulation": int8_timing},
            "gate": gate,
            "speed_interpretation": (
                "QDQ emulation overhead only; native int8 convolution/ANE speed remains unmeasured"
            ),
            "reformulation_queue": (
                []
                if gate["verdict"] == "GO"
                else [
                    "operator-wise precision waterfill from per-pair gradient damage",
                    "W8A16 activation relaxation while preserving int8 weights",
                    "ANE forward plus MLX QDQ custom-VJP only after the same n600 cosine gate",
                ]
            ),
        }
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=P.DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default=P.DEFAULT_CHECKPOINT)
    parser.add_argument("--gt-cache", type=Path, default=P.DEFAULT_GT_CACHE)
    parser.add_argument("--timing-pairs", type=int, default=8)
    parser.add_argument("--quality-pairs", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not (1 <= args.timing_pairs <= 600 and 1 <= args.quality_pairs <= 600):
        raise SystemExit("timing/quality pairs must be in 1..600")
    if str(args.out).startswith(("/tmp/", "/private/tmp/")):
        raise SystemExit("refusing a temporary durable evidence path")
    payload = run(args)
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
