#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LEVER-B generator trainer WITH checkpoint save — the missing campaign artifact.

The lever-B descent smoke (`tools/lever_b_score_native_argmax_smoke.py`) proved the seg term
is generatable but discarded the trained weights. The campaign (run the closed-spec boundary
solver on the lever_b base + build the legal-frame bridge) needs the trained generator's argmax
residual. This tool trains the SAME MLX generator and SAVES it to a portable .npz (via
`tac.boundary_math.lever_b_generator.save_generator_npz`) so the solver + bridge can reconstruct
the generator argmax WITHOUT retraining.

It reuses the smoke's `ScoreNativeSegGenerator` (MLX) for training and verifies MLX->numpy
argmax-parity on the saved checkpoint (the portability contract). NO MPS. Targets on the SSD tier.

Evidence ``[macOS-MLX research-signal]`` (generator forward); ``promotion_eligible=false``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from tac.boundary_math.lever_b_generator import (  # noqa: E402
    GeneratorConfig,
    build_coords,
    generator_argmax,
    save_generator_npz,
)

# Reuse the smoke's MLX generator definition (single source of truth for the arch).
from tools.lever_b_score_native_argmax_smoke import ScoreNativeSegGenerator  # noqa: E402

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


def _mlx_params_to_numpy(model: ScoreNativeSegGenerator) -> dict[str, np.ndarray]:
    flat = dict(tree_flatten(model.parameters()))
    return {k: np.array(v).astype(np.float32) for k, v in flat.items()}


def train_and_checkpoint(
    targets_dir: Path,
    ckpt_path: Path,
    num_pairs: int,
    epochs: int,
    eval_every: int,
    hinge_weight: float,
    seed: int,
    lr: float,
    px_per_step: int,
) -> dict[str, Any]:
    _refuse_tmp(ckpt_path, "ckpt_path")
    mx.random.seed(seed)
    np.random.seed(seed)

    meta = json.loads((targets_dir / "targets_meta.json").read_text())
    n_pairs_built = int(meta["num_pairs_built"])
    H, W = meta["seg_input_hw"]
    n_pairs = min(num_pairs, n_pairs_built)

    argmax = np.memmap(targets_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="r",
                       shape=(n_pairs_built, H, W))
    margin = np.memmap(targets_dir / "gt_segnet_margin.f16", dtype=np.float16, mode="r",
                       shape=(n_pairs_built, H, W))
    tgt_labels = np.asarray(argmax[:n_pairs]).astype(np.int32)
    tgt_margin = np.asarray(margin[:n_pairs]).astype(np.float32)

    coords_full_np = build_coords(H, W)
    coords_full = mx.array(coords_full_np)
    n_px = H * W

    model = ScoreNativeSegGenerator(num_pairs=n_pairs)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    tau = 1.0

    def loss_fn(model, pair_idx, coord_batch, labels, w_px):
        logits = model(coord_batch, pair_idx)
        ce = nn.losses.cross_entropy(logits, labels, reduction="none")
        return (ce * w_px).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def eval_d_seg() -> float:
        disagree = total = 0
        for pi in range(n_pairs):
            logits = model(coords_full, pi)
            pred = np.array(mx.argmax(logits, axis=-1)).astype(np.int32)
            gt = tgt_labels[pi].ravel()
            disagree += int((pred != gt).sum())
            total += pred.size
        return disagree / total

    cfg = GeneratorConfig(num_pairs=n_pairs, n_fourier=model.n_fourier,
                          hidden_dim=model.hidden_dim, n_hidden=model.n_hidden,
                          mod_dim=model.mod_dim, n_classes=model.n_classes)

    history: list[dict[str, Any]] = []
    t0 = time.time()
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        order = rng.permutation(n_pairs)
        for pi in order:
            idx = rng.integers(0, n_px, size=px_per_step)
            cb = coords_full[mx.array(idx)]
            lab = mx.array(tgt_labels[pi].ravel()[idx])
            mg = tgt_margin[pi].ravel()[idx]
            wpx = mx.array(1.0 + hinge_weight * np.exp(-np.maximum(mg, 0.0) / tau))
            loss, grads = loss_and_grad(model, int(pi), cb, lab, wpx)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            ep_loss += float(loss)
        if ep % eval_every == 0 or ep == 1 or ep == epochs:
            d_seg = eval_d_seg()
            row = {"epoch": ep, "train_loss": round(ep_loss / n_pairs, 5),
                   "d_seg": round(d_seg, 6), "wall_s": round(time.time() - t0, 1)}
            history.append(row)
            print(json.dumps(row), flush=True)
            # INCREMENTAL checkpoint save (crash-resilient: the bash harness sends
            # SIGURG to long bg tasks at ~3-5min; saving every eval window means a
            # killed run still leaves the latest converged artifact on disk).
            save_generator_npz(ckpt_path, _mlx_params_to_numpy(model), cfg)

    # --- final checkpoint (the missing artifact) ---
    params = _mlx_params_to_numpy(model)
    save_generator_npz(ckpt_path, params, cfg)

    # --- portability parity: MLX argmax vs numpy argmax on pair 0 (a slice) ---
    parity = _verify_parity(model, params, cfg, coords_full, coords_full_np, H, W, n_check=4)

    result = {
        "subagent": "score_native_first_candidate_55_56",
        "utc": _utc(),
        "evidence_grade": "[macOS-MLX research-signal]",
        "promotion_eligible": False,
        "score_claim": False,
        "checkpoint": str(ckpt_path),
        "config": cfg.to_dict(),
        "history": history,
        "final_d_seg": history[-1]["d_seg"],
        "portability_parity": parity,
    }
    return result


def _verify_parity(model, params, cfg, coords_full_mx, coords_full_np, H, W, n_check) -> dict[str, Any]:
    """MLX argmax vs numpy-portable argmax on the saved checkpoint (parity gate)."""
    rng = np.random.default_rng(0)
    agree = []
    for pi in rng.integers(0, cfg.num_pairs, size=min(n_check, cfg.num_pairs)):
        mlx_logits = np.array(model(coords_full_mx, int(pi)))
        mlx_argmax = mlx_logits.argmax(-1)
        # argmax parity is the portability contract (logit drift is expected + irrelevant to d_seg)
        np_argmax = generator_argmax(params, cfg, coords_full_np, int(pi), H, W).ravel()
        agree.append(float((mlx_argmax == np_argmax).mean()))
    return {
        "argmax_agreement_min": float(min(agree)),
        "argmax_agreement_mean": float(np.mean(agree)),
        "pairs_checked": len(agree),
        "parity_pass": bool(min(agree) >= 0.999),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LEVER-B generator trainer + checkpoint")
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--ckpt", type=Path, default=Path(base) / "generator_ckpt" / "generator_n600.npz")
    ap.add_argument("--out-json", type=Path, default=Path(base) / "generator_ckpt" / "train_result.json")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--px-per-step", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    result = train_and_checkpoint(
        targets_dir=args.targets_dir, ckpt_path=args.ckpt, num_pairs=args.num_pairs,
        epochs=args.epochs, eval_every=args.eval_every, hinge_weight=args.hinge_weight,
        seed=args.seed, lr=args.lr, px_per_step=args.px_per_step,
    )
    _refuse_tmp(args.out_json, "out_json")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print("\n=== TRAIN RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
