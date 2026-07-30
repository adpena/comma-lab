# SPDX-License-Identifier: MIT
"""ddm_pj1 — PROJECTION PROBE: renderer-class capacity-floor fit (scorer-free).

Freezes the QA24 endpoint renderer (EMA-shadow deploy weights) and fits ONLY the
token field (``tokens_base`` + per-pair ``tokens_delta``) to reproduce the C1
EXACT-solve frames (the frames that realize d_seg ~1.52e-4 through the real
path, materialized scorer-free by ddm_b2p) THROUGH the deploy path
(cell-mask -> STE-round 16-level quant -> renderer -> contest-faithful R). No
SegNet/PoseNet is run in the fit loop (scorer-FREE); the realized d_seg of the
fitted state is the deliberately-separate ENDPOINT GATE (compile ->
``tools/pb1_receiver_realized_verdict.py``, slot-disciplined).

Output ``f`` = the renderer-class conditional capacity FLOOR at the QA24
granularity/rate. Because the fit minimises a photometric objective (not d_seg
directly), the realised d_seg it reaches is an ACHIEVABLE point, hence an UPPER
BOUND on the true capacity floor (the renderer could only do better with a
d_seg-aware objective). ``--loss margin`` tightens that bound by concentrating
capacity on the boundary annulus (Contrarian rider: fit in a margin-aware metric
OR report the L2<->margin gap).

Fork discriminator (gc9 §2):
  f << 0.00528  -> vehicle CAN express Gate-B fidelity; the 25.58x gap is
                  TARGET-infeasibility -> QA75 distill burn-3 is well-founded.
  f ~= 0.00528  -> renderer-class CAPACITY wall at INSTANCE(QA24 geometry) ->
                  capacity fork (renderer-class change / granularity re-race).

Pointer honesty: ``0.1910828242 [contest-CPU]`` UNMOVED. Every number this
script produces is ``[macOS-CPU advisory]``, ``score_claim=false``,
``research_only``. The fit is deterministic (seeded) + resumable-from-disk
(atomic per-stage checkpoints) per the P0 launch non-negotiable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Canonical custody defaults (all recalled + SHA-verified in the ddm_pj1 memo).
# --------------------------------------------------------------------------
DEFAULT_CKPT = (
    "/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out/checkpoints/"
    "stage_seg_trunk_tau_final.npz"
)
DEFAULT_SOLVE = "/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa75_solve_frames"
DEFAULT_GT_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUT = "/Volumes/VertigoDataTier/pact/ddm_pj1_20260730"
SCORER_HW = (384, 512)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CKPT))
    ap.add_argument("--solve-dir", type=Path, default=Path(DEFAULT_SOLVE))
    ap.add_argument("--gt-cache", type=Path, default=Path(DEFAULT_GT_CACHE),
                    help="only read when --loss margin (margins field)")
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--tag", type=str, default="warm_l2",
                    help="run tag => subdir of --out-dir")
    ap.add_argument("--init", choices=["warm", "cold"], default="warm",
                    help="warm = EMA-shadow tokens (deploy state); cold = zeros")
    ap.add_argument("--loss", choices=["l2", "margin"], default="l2")
    ap.add_argument("--margin-temp", type=float, default=0.5,
                    help="margin-loss weight = 1 + gain*exp(-margin/temp)")
    ap.add_argument("--margin-gain", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lr-final", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--max-wall-seconds", type=float, default=9000.0,
                    help="hard wall-clock cap (~2.5h) — fail-open: stop + write")
    ap.add_argument("--early-stop-rel", type=float, default=2e-4,
                    help="stop if rel loss drop over the last 30-epoch window < this")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def load_cfg_and_module(checkpoint: Path):
    """Build the TR1 module, load the FROZEN EMA-shadow renderer, return (cfg, model,
    ema_dict, endpoint_arrays, meta_json_bytes)."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from experiments.train_tr1_partition_renderer_mlx import TR1Config, build_module

    stored = np.load(checkpoint, allow_pickle=True)
    meta_json_bytes = bytes(stored["meta::json"].tolist())
    meta = json.loads(meta_json_bytes.decode("utf-8"))
    cfg_d = meta["cfg"]
    fields = {f.name for f in dataclasses.fields(TR1Config)}
    unknown = set(cfg_d) - fields
    # fail-closed on drift: config keys the dataclass does not know about would be
    # silently dropped (never-invent-geometry). All bc1 keys must map.
    if unknown:
        raise ValueError(f"checkpoint cfg has keys unknown to TR1Config: {sorted(unknown)}")
    cfg = TR1Config(**{k: v for k, v in cfg_d.items() if k in fields})
    model = build_module(cfg)
    ema = {k[5:]: mx.array(stored[k]) for k in stored.files if k.startswith("ema::")}
    # Load EMA shadow into ALL params (renderer + tokens); renderer is then frozen.
    model.update(tree_unflatten([(k, ema[k]) for k in ema]))
    model._quant_engaged = True  # deploy quant on (post-knee endpoint)
    endpoint_arrays = {k: np.asarray(stored[k]) for k in stored.files}
    return cfg, model, ema, endpoint_arrays, meta_json_bytes


def build_targets(solve_dir: Path, out_dir: Path, num_pairs: int):
    """Precompute the scorer-input targets (bilinear-down of solve frame1 to 384x512)
    ONCE to SSD as fp16 memmap; return (targets_memmap, cache_path, cache_sha)."""
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        resize_nhwc_align_corners_false,
    )
    from tac.witness_dsl.qa75_solve_frame_targets import SolveFrameTargets

    cache_path = out_dir / f"targets_scorer_input_f16_n{num_pairs}.npy"
    if cache_path.is_file():
        tm = np.load(cache_path, mmap_mode="r")
        if tm.shape == (num_pairs, SCORER_HW[0], SCORER_HW[1], 3):
            return tm, cache_path, _sha256_file(cache_path)
    tgt = SolveFrameTargets.load(solve_dir)
    if tgt.pair_count < num_pairs:
        raise ValueError(f"solve targets only {tgt.pair_count} < {num_pairs}")
    out = np.empty((num_pairs, SCORER_HW[0], SCORER_HW[1], 3), dtype=np.float16)
    B = 20
    for s in range(0, num_pairs, B):
        e = min(s + B, num_pairs)
        fr = np.stack([np.asarray(tgt.frame1(i), dtype=np.float32) for i in range(s, e)], 0)
        with mx.stream(mx.cpu):
            d = resize_nhwc_align_corners_false(mx.array(fr), size=SCORER_HW, mode="bilinear")
            mx.eval(d)
        out[s:e] = np.asarray(d, dtype=np.float32).astype(np.float16)
    tmp = cache_path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, out)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cache_path)
    tm = np.load(cache_path, mmap_mode="r")
    return tm, cache_path, _sha256_file(cache_path)


def build_margin_weights(gt_cache: Path, num_pairs: int, temp: float, gain: float):
    """Per-pair (384,512) inverse-margin weight = 1 + gain*exp(-margin/temp).
    Small GT margin (boundary annulus) => high weight. Read margins member only."""
    import zipfile

    z = zipfile.ZipFile(gt_cache)
    with z.open("margins.npy") as f:
        margins = np.lib.format.read_array(f)  # (600,384,512) fp32
    m = margins[:num_pairs].astype(np.float32)
    w = 1.0 + gain * np.exp(-np.maximum(m, 0.0) / max(temp, 1e-6))
    return w.astype(np.float32)


def main() -> int:
    args = parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from experiments.train_witness_realized_through_R_mlx import _apply_R

    run_dir = args.out_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry = run_dir / "fit_telemetry.jsonl"
    ckpt_state = run_dir / "fit_state.npz"          # tokens + opt + epoch (resume)
    fitted_out = run_dir / "fitted_checkpoint.npz"  # compiler-ready

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    cfg, model, ema, endpoint_arrays, meta_json_bytes = load_cfg_and_module(args.checkpoint)
    ckpt_sha = _sha256_file(args.checkpoint)
    num_pairs = cfg.num_pairs

    # --- init tokens ---
    if args.init == "cold":
        model.tokens_base = mx.zeros_like(model.tokens_base)
        model.tokens_delta = mx.zeros_like(model.tokens_delta)
    # (warm = already loaded EMA-shadow tokens)

    # --- freeze renderer, keep only tokens trainable ---
    model.freeze()
    model.unfreeze(keys="tokens_base")
    model.unfreeze(keys="tokens_delta")
    trainable = sorted(dict(tree_flatten(model.trainable_parameters())).keys())
    if trainable != ["tokens_base", "tokens_delta"]:
        raise RuntimeError(f"freeze failed: trainable = {trainable}")

    # --- targets (scorer-input) ---
    targets, tgt_cache_path, tgt_cache_sha = build_targets(args.solve_dir, args.out_dir, num_pairs)
    margin_w = None
    if args.loss == "margin":
        margin_w = build_margin_weights(args.gt_cache, num_pairs, args.margin_temp, args.margin_gain)

    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(model, ids, T, W):
        outs = [_apply_R(model.render_frame(int(i))) for i in ids]
        R = mx.concatenate(outs, axis=0)  # (B,384,512,3) scorer-input float
        d2 = (R - T) ** 2
        if W is not None:
            return mx.sum(d2 * W) / mx.sum(W)
        return mx.mean(d2)

    lag = nn.value_and_grad(model, loss_fn)

    # --- resume ---
    start_epoch = 0
    loss_hist: list[float] = []
    if args.resume and ckpt_state.is_file():
        st = np.load(ckpt_state, allow_pickle=True)
        model.tokens_base = mx.array(st["tokens_base"])
        model.tokens_delta = mx.array(st["tokens_delta"])
        start_epoch = int(st["epoch"]) + 1
        loss_hist = list(np.asarray(st["loss_hist"], dtype=np.float64))
        # rng continuity
        mx.random.seed(args.seed + start_epoch)
        np.random.seed(args.seed + start_epoch)
        print(f"[resume] from epoch {start_epoch}, last loss {loss_hist[-1] if loss_hist else float('nan'):.4f}")

    steps_per_epoch = (num_pairs + args.batch - 1) // args.batch
    total_steps = max(1, args.epochs * steps_per_epoch)
    t_start = time.monotonic()

    def cosine_lr(global_step: int) -> float:
        p = min(1.0, global_step / total_steps)
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1.0 + np.cos(np.pi * p))

    def save_state(epoch: int) -> None:
        _atomic_save_npz(ckpt_state, {
            "tokens_base": np.asarray(model.tokens_base, dtype=np.float32),
            "tokens_delta": np.asarray(model.tokens_delta, dtype=np.float32),
            "epoch": np.int64(epoch),
            "loss_hist": np.asarray(loss_hist, dtype=np.float64),
        })

    stopped_reason = "epochs_complete"
    for epoch in range(start_epoch, args.epochs):
        rng = np.random.default_rng(args.seed * 100003 + epoch)
        order = rng.permutation(num_pairs)
        ep_losses = []
        for s in range(0, num_pairs, args.batch):
            ids = [int(x) for x in order[s:s + args.batch]]
            T = mx.array(np.asarray(targets[ids], dtype=np.float32))
            W = None
            if margin_w is not None:
                W = mx.array(margin_w[ids][..., None])  # (B,384,512,1)
            gstep = epoch * steps_per_epoch + (s // args.batch)
            opt.learning_rate = float(cosine_lr(gstep))
            loss, grads = lag(model, ids, T, W)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            ep_losses.append(float(loss))
        ep_loss = float(np.mean(ep_losses))
        loss_hist.append(ep_loss)
        rms = float(np.sqrt(ep_loss))
        elapsed = time.monotonic() - t_start
        row = {
            "event": "epoch", "epoch": epoch, "ep_loss_mse": ep_loss,
            "ep_rms_px": rms, "lr": float(opt.learning_rate),
            "elapsed_s": round(elapsed, 1), "init": args.init, "loss": args.loss,
        }
        _jsonl(telemetry, row)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"ep {epoch:4d}  mse {ep_loss:10.3f}  rms {rms:7.3f}px  lr {opt.learning_rate:.2e}  {elapsed:6.0f}s")
        if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_state(epoch)
        # early-stop: rel drop over last 30-epoch window
        if len(loss_hist) >= 60:
            past = loss_hist[-31]
            now = loss_hist[-1]
            if past > 0 and (past - now) / past < args.early_stop_rel:
                stopped_reason = "plateau"
                save_state(epoch)
                print(f"[early-stop] plateau at epoch {epoch}: rel drop < {args.early_stop_rel}")
                break
        if elapsed > args.max_wall_seconds:
            stopped_reason = "wall_cap"
            save_state(epoch)
            print(f"[wall-cap] stop at epoch {epoch}: {elapsed:.0f}s")
            break

    # --- write compiler-ready fitted checkpoint (copy endpoint, replace ema tokens) ---
    fitted = dict(endpoint_arrays)
    fitted["ema::tokens_base"] = np.asarray(model.tokens_base, dtype=np.float32)
    fitted["ema::tokens_delta"] = np.asarray(model.tokens_delta, dtype=np.float32)
    # keep param:: keys as-is (unused by compiler; renderer ema unchanged; meta::json byte-identical)
    _atomic_save_npz(fitted_out, fitted)

    final_rms = float(np.sqrt(loss_hist[-1])) if loss_hist else float("nan")
    summary = {
        "event": "summary",
        "tag": args.tag, "init": args.init, "loss": args.loss,
        "epochs_run": len(loss_hist), "stopped_reason": stopped_reason,
        "final_ep_mse": loss_hist[-1] if loss_hist else None,
        "final_rms_px": final_rms,
        "first_ep_mse": loss_hist[0] if loss_hist else None,
        "checkpoint_src": str(args.checkpoint), "checkpoint_sha256": ckpt_sha,
        "targets_cache": str(tgt_cache_path), "targets_sha256": tgt_cache_sha,
        "fitted_checkpoint": str(fitted_out),
        "fitted_checkpoint_sha256": _sha256_file(fitted_out),
        "num_pairs": num_pairs, "batch": args.batch, "seed": args.seed,
        "lr": args.lr, "lr_final": args.lr_final,
        "axis": "[macOS-CPU advisory]", "score_claim": False, "research_only": True,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
    }
    _jsonl(telemetry, summary)
    _atomic_write_bytes(run_dir / "fit_summary.json",
                        json.dumps(summary, indent=1, sort_keys=True).encode())
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
