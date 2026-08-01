#!/usr/bin/env python
"""Real-frame SegNet-argmax parity gate for the two grouped+strided conv routings.

THE QUESTION. ``torch_conv2d_to_mlx`` routes a grouped+strided Conv2d to one of two adapters:

  * ``MLXCustomKernelStridedGroupedConvAdapter`` -- the DEFAULT since 2026-06-27
    (``fcbb6112ee``), selected when ``TAC_MLX_CUSTOM_GROUPED_BACKWARD`` is unset or ``1``.
    Its backward is the ~17x custom Metal grouped VJP; its FORWARD is native ``mx.conv2d``.
  * ``MLXReferenceConv2dAdapter(accumulation_mode="fixed_fp32")`` -- the fixed-order loop,
    selected with ``TAC_MLX_CUSTOM_GROUPED_BACKWARD=0``.

The 2026-06-27 default flip was justified on the BACKWARD (grad cosine ~1.0, ~17x faster). It
silently swapped the FORWARD too -- and a 2026-06-11 real-frame probe had REJECTED exactly that
forward swap for flipping a SegNet argmax pixel on ``0.mkv``. The gate re-running that check was
recorded as OWED and had not been run against the live default. This tool is that gate.

WHAT IT IS NOT. MLX is NEVER a score authority (CLAUDE.md "MPS auth eval is NOISE" +
"MLX portable-local-substrate authority"); the frozen CPU-torch scorer is. A flip here does not
corrupt any reported d_seg. It perturbs the TRAINING-TIME seg signal, which is a real but
different claim, and this tool reports it as such.

METHOD. Both adapters are built from the SAME frozen upstream ``DistortionNet.segnet``. The
routing is frozen at ADAPTER CONSTRUCTION (every ``torch_conv2d_to_mlx`` call site is inside an
``__init__``), so the env var must be set BEFORE each build and the adapter rebuilt -- flipping
it afterwards has no effect. Inputs are real ``0.mkv`` frames from the canonical scorer-input
cache, already preprocessed exactly as upstream does it (``.float()`` to [0,255], no /255, no
normalization, bilinear resize to 384x512, last frame of each pair) -- see
``tac.local_acceleration.mlx_preprocess`` and ``upstream/modules.py:107``.

Usage:
    .venv/bin/python tools/probe_real_frame_dseg_grouped_conv_routing.py --pairs 96 \
        --out reports/raw/real_frame_dseg_grouped_conv_routing.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = (
    REPO_ROOT
    / "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
)
ROUTING_ENV = "TAC_MLX_CUSTOM_GROUPED_BACKWARD"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="scorer-input cache dir (must hold real 0.mkv frames)")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--pairs", type=int, default=96, help="number of cache pairs to score")
    ap.add_argument("--chunk", type=int, default=4, help="frames per forward (memory bound)")
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--out", default="", help="write the full per-frame JSON here")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    from tac.local_acceleration.mlx_scorer_adapters import (
        run_mlx_segnet_nchw,
        temporary_mlx_device,
        torch_segnet_to_mlx,
    )
    from tac.local_acceleration.mlx_scorer_response import (
        _load_upstream_distortion_net,
        _resolve_upstream_dir,
        load_scorer_input_cache,
    )

    cache_dir = Path(args.cache_dir)
    # ``integrity_mode="manifest"`` reuses the manifest hashes instead of re-hashing 2.7 GB of
    # arrays; the identity fields we report below still come from the manifest.
    cache = load_scorer_input_cache(cache_dir, integrity_mode="manifest")
    source = str(cache.manifest.get("source", ""))
    if not source.endswith(".mkv"):
        raise SystemExit(
            f"REFUSE: cache source {source!r} is not a video; this gate is meaningless on "
            "synthetic or reconstructed frames (see CLAUDE.md NO-FAKE class 3)."
        )
    dist = _load_upstream_distortion_net(_resolve_upstream_dir(repo_root))

    n_pairs = min(int(args.pairs), int(cache.segnet_last_rgb.shape[0]))
    if n_pairs < 1:
        raise SystemExit("REFUSE: --pairs must select at least one frame")

    prior_env = os.environ.get(ROUTING_ENV)
    total_pixels = 0
    total_flips = 0
    per_frame: list[dict] = []
    worst: dict | None = None
    try:
        for lo in range(0, n_pairs, max(1, int(args.chunk))):
            hi = min(n_pairs, lo + max(1, int(args.chunk)))
            x = np.asarray(cache.segnet_last_rgb[lo:hi], dtype=np.float32)
            logits: dict[str, np.ndarray] = {}
            for flag in ("0", "1"):
                os.environ[ROUTING_ENV] = flag
                with temporary_mlx_device(args.device):
                    # Rebuild: the routing decision is frozen at construction time.
                    seg = torch_segnet_to_mlx(dist.segnet)
                    logits[flag] = np.asarray(run_mlx_segnet_nchw(seg, x), dtype=np.float32)
            ref, custom = logits["0"], logits["1"]
            arg_ref, arg_custom = ref.argmax(axis=1), custom.argmax(axis=1)
            mismatch = arg_ref != arg_custom
            total_pixels += int(arg_ref.size)
            total_flips += int(mismatch.sum())
            ordered = np.sort(ref, axis=1)
            margin = ordered[:, -1, ...] - ordered[:, -2, ...]
            for k in range(hi - lo):
                flips = int(mismatch[k].sum())
                per_frame.append({
                    "pair": lo + k,
                    "argmax_flip_pixels": flips,
                    "logit_abs_delta_max": float(np.abs(custom[k] - ref[k]).max()),
                    "reference_top2_margin_min": float(margin[k].min()),
                })
                if flips and (worst is None or flips > worst["argmax_flip_pixels"]):
                    ys, xs = np.nonzero(mismatch[k])
                    worst = {
                        "pair": lo + k,
                        "argmax_flip_pixels": flips,
                        "mismatch_min_top2_margin": float(margin[k][mismatch[k]].min()),
                        "mismatch_logit_abs_delta_max": float(
                            np.abs(custom[k] - ref[k]).max(axis=0)[mismatch[k]].max()),
                        "example_pixels_yx": [[int(y), int(x)] for y, x in
                                              zip(ys[:8], xs[:8], strict=False)],
                    }
    finally:
        if prior_env is None:
            os.environ.pop(ROUTING_ENV, None)
        else:
            os.environ[ROUTING_ENV] = prior_env

    fraction = total_flips / total_pixels if total_pixels else 0.0
    result = {
        "verdict": "ARGMAX_FLIPS_ON_REAL_FRAMES" if total_flips else "NO_ARGMAX_FLIP",
        "pairs": n_pairs,
        "total_pixels": total_pixels,
        "argmax_flip_pixels": total_flips,
        "argmax_flip_fraction": fraction,
        "frames_with_any_flip": sum(1 for r in per_frame if r["argmax_flip_pixels"]),
        "max_logit_abs_delta": max((r["logit_abs_delta_max"] for r in per_frame), default=0.0),
        "worst_frame": worst,
        "default_routing": f"{ROUTING_ENV} unset -> MLXCustomKernelStridedGroupedConvAdapter",
        "compared_against": "MLXReferenceConv2dAdapter(fixed_fp32)",
        "authority": (
            "MLX is NEVER a score authority; this measures TRAINING-SIGNAL shape only. "
            "score_claim=false, promotion_eligible=false."
        ),
        "cache_dir": str(cache_dir),
        "source": source,
        "source_video_sha256": cache.manifest.get("source_video_sha256"),
        "device": args.device,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({**result, "per_frame": per_frame}, indent=2))
        print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
