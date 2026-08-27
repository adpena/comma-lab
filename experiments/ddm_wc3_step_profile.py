"""ddm_wc3 — FULL forward+backward component profile of ONE mx1 mlx-train step (instrument == vehicle).

Operator 2026-08-08: "full profiling of the forward and backward in all aspects of everything
for profiling and identification of hotspots and opportunities for optimization."

Times the exact `run_mlx_train` loss path by cumulative-graph differencing (MLX-lazy-safe:
each stage-depth is a REAL evaluated graph; marginal cost = difference of adjacent cumulative
timings), in BOTH directions:

  FORWARD  F1..F5   cumulative mx.eval of the stage-k output
  BACKWARD V1..V5   cumulative mx.value_and_grad of a scalar reduction of the stage-k output
                    (V5 uses the REAL curriculum loss, already scalar)
  bwd marginal of stage k = (V_k - V_{k-1}) - (F_k - F_{k-1})
  plus OPT          optimizer.update(model, grads) + state eval (once per step, not per chunk)

Stages (the exact trainer closure order, trainer lines ~2935-2977):
  S1 quantize+cast+update  fake_quantize_parameter_tree -> _cast_mlx_parameter_tree -> model.update
  S2 renderer fwd          model(conditioning, pair_idx)
  S3 R roundtrip           apply_contest_faithful_roundtrip_nhwc (874x1164 up, uint8 STE, down)
  S4 segnet fwd            segnet_mlx(frame_r) + transpose
  S5 loss                  curriculum_loss_mlx

Dual-dtype: --dtypes fp32,fp16 profiles the baseline anatomy AND the bench-winning fp16-train
anatomy in one run. All functions IMPORTED from the trainer module + its routed tac modules —
no re-implementation. One microbatch chunk (4 pairs, the live GPU default) is profiled;
per-step projection = chunks * V5 + OPT. Axis: [macOS-Metal wall-clock instrument];
score_claim=false; NO scorer authority implied.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "src"), str(REPO / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4, help="microbatch pairs per chunk (live GPU default 4)")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtypes", type=str, default="fp32,fp16",
                    help="comma list of --train-compute-dtype modes to profile")
    ap.add_argument("--project-pairs", type=str, default="32,120",
                    help="comma list of n values for per-step projections")
    ap.add_argument("--input-cache", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt"))
    ap.add_argument("--target-cache", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt"))
    ap.add_argument("--init", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/"
                                 "artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt"))
    ap.add_argument("--out", type=Path,
                    default=Path(".omx/research/ddm_wc3_20260808/step_profile_result.json"))
    args = ap.parse_args()

    tr = importlib.import_module("ddm_mx1_pr130_semantic_renderer")
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    import torch
    from mlx.utils import tree_flatten, tree_unflatten

    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )
    from tac.pr130_lift.mlx_semantic_renderer import (
        MlxSemanticConfig,
        curriculum_loss_mlx,
        fake_quantize_parameter_tree,
        load_torch_state_dict_into_mlx,
        make_mlx_renderer,
    )

    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MlxSemanticConfig.from_pr130_checkpoint(
        checkpoint, consumer="ddm_wc3_step_profile"
    )
    config = MlxSemanticConfig(**(config.asdict() | {"bits": args.bits}))

    pair_ids = tr._select_stratified_indices(args.pairs, seed=args.seed)
    conditioning_np, target_np, cache_meta, _handles = tr._load_selected_token_arrays(
        input_cache=args.input_cache, target_cache=args.target_cache, pair_ids=pair_ids,
        memory_probe=None,
    )

    model = make_mlx_renderer(config, device="gpu")
    load_torch_state_dict_into_mlx(model, checkpoint["state_dict"], device="gpu")
    mx.eval(model.parameters())

    segnet_torch = tr._load_upstream_segnet(torch.device("cpu"))
    segnet_mlx = torch_segnet_to_mlx(segnet_torch)
    if hasattr(segnet_mlx, "parameters"):
        mx.eval(segnet_mlx.parameters())
    del segnet_torch

    n = args.chunk
    conditioning = mx.array(np.ascontiguousarray(conditioning_np[:n]))
    target = mx.array(np.ascontiguousarray(target_np[:n]))
    pair_idx = mx.array(np.asarray(pair_ids[:n], dtype=np.int32))
    mx.eval(conditioning, target, pair_idx)

    labels = ["S1_quantize_cast_update", "S2_renderer_fwd", "S3_R_roundtrip",
              "S4_segnet_fwd", "S5_loss"]

    def tree_scalar(t):
        leaves = [v for _, v in tree_flatten(t)]
        return sum((leaf.astype(mx.float32).sum() for leaf in leaves), mx.array(0.0))

    def profile_dtype(dtype_mode: str) -> dict:
        train_compute_dtype = tr._resolve_train_compute_dtype(mx, dtype_mode)

        def stage(depth: int, p):
            active = fake_quantize_parameter_tree(mx, tree_flatten, tree_unflatten, p, bits=args.bits)
            active = tr._cast_mlx_parameter_tree(tree_flatten, tree_unflatten, active, train_compute_dtype)
            model.update(active)
            if depth == 1:
                return active
            frame = model(conditioning, pair_idx)
            if depth == 2:
                return frame
            frame_r = apply_contest_faithful_roundtrip_nhwc(frame, output_hw=(384, 512), ste_round=True)
            if depth == 3:
                return frame_r
            logits_nhwc = segnet_mlx(frame_r)
            logits_nchw = mx.transpose(logits_nhwc, (0, 3, 1, 2))
            if depth == 4:
                return logits_nchw
            loss, _phase = curriculum_loss_mlx(
                mx, logits_nchw, target, step=0, total_steps=1000,
                ce_fraction=0.0, softplus_fraction=-999.0,
            )
            return loss

        def stage_scalar(depth: int, p):
            out = stage(depth, p)
            if depth == 5:
                return out
            if isinstance(out, dict):
                return tree_scalar(out)
            return out.astype(mx.float32).sum()

        def timed(fn) -> float:
            for _ in range(args.warmup):
                mx.eval(fn())
            t0 = time.perf_counter()
            for _ in range(args.reps):
                mx.eval(fn())
            return (time.perf_counter() - t0) / args.reps

        base_params = model.trainable_parameters()
        fwd_cum: dict[str, float] = {}
        vag_cum: dict[str, float] = {}
        grads_holder: dict[str, object] = {}
        for depth, label in enumerate(labels, start=1):
            fwd_cum[label] = timed(lambda d=depth: stage(d, base_params))

            def vag_once(d=depth):
                value, grads = mx.value_and_grad(lambda p: stage_scalar(d, p))(base_params)
                grads_holder["g"] = grads
                return value, grads

            for _ in range(args.warmup):
                v, g = vag_once()
                mx.eval(v, g)
            t0 = time.perf_counter()
            for _ in range(args.reps):
                v, g = vag_once()
                mx.eval(v, g)
            vag_cum[label] = (time.perf_counter() - t0) / args.reps

        optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)
        grads = grads_holder["g"]

        def opt_once():
            optimizer.update(model, grads)
            return model.parameters(), optimizer.state

        for _ in range(args.warmup):
            pr, st = opt_once()
            mx.eval(pr, st)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            pr, st = opt_once()
            mx.eval(pr, st)
        opt_s = (time.perf_counter() - t0) / args.reps

        marg = {}
        prev_f = prev_v = 0.0
        for label in labels:
            f_m = fwd_cum[label] - prev_f
            v_m = (vag_cum[label] - prev_v) - f_m
            marg[label] = {"fwd_s": f_m, "bwd_s": v_m, "fwd_plus_bwd_s": f_m + v_m}
            prev_f, prev_v = fwd_cum[label], vag_cum[label]

        full_chunk = vag_cum["S5_loss"]
        projections = {}
        for np_pairs in (int(x) for x in args.project_pairs.split(",")):
            chunks = max(1, np_pairs // args.chunk)
            projections[f"n{np_pairs}"] = {
                "chunks": chunks,
                "projected_step_s": full_chunk * chunks + opt_s,
            }
        return {
            "train_compute_dtype": dtype_mode,
            "cumulative_fwd_s_per_chunk": fwd_cum,
            "cumulative_vag_s_per_chunk": vag_cum,
            "marginal_s_per_chunk": marg,
            "optimizer_update_s_per_step": opt_s,
            "full_chunk_vag_s": full_chunk,
            "shares_of_full_chunk": {
                k: {"fwd": v["fwd_s"] / full_chunk, "bwd": v["bwd_s"] / full_chunk}
                for k, v in marg.items()
            },
            "projections": projections,
        }

    git_head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO,  # subprocess-no-check-OK: git-head provenance capture; empty-on-failure is visible in the receipt
                              capture_output=True, text=True).stdout.strip()
    result = {
        "schema": "ddm_wc3_step_profile.v2",
        "axis": "[macOS-Metal wall-clock instrument]",
        "score_claim": False,
        "repo_head": git_head,
        "chunk_pairs": args.chunk,
        "pairs_loaded": args.pairs,
        "reps": args.reps,
        "warmup": args.warmup,
        "init": str(args.init),
        "cache_meta": {k: str(v) for k, v in (cache_meta or {}).items()} if isinstance(cache_meta, dict) else str(cache_meta),
        "profiles": {},
    }
    for dtype_mode in args.dtypes.split(","):
        dtype_mode = dtype_mode.strip()
        result["profiles"][dtype_mode] = profile_dtype(dtype_mode)
        print(f"=== {dtype_mode} ===")
        print(json.dumps(result["profiles"][dtype_mode]["marginal_s_per_chunk"], indent=1))
        print(json.dumps({"opt_s": result["profiles"][dtype_mode]["optimizer_update_s_per_step"],
                          "projections": result["profiles"][dtype_mode]["projections"]}, indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
