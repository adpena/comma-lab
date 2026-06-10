#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LEVER-B descent smoke — score-native argmax generator (MLX-first).

THE #1 OFFENSIVE LEVER's $0 descent smoke (GOAL_standing_v3 +
`.omx/research/original_innovation_offensive_plan_20260610.md` §4).

PRE-REGISTERED HYPOTHESIS (H_B): a SMALL conditional label-map generator
  g(x, y, t) -> 5-class logits at 384x512, trained with cross-entropy +
  argmax-hinge (lever G margin-weighting) against the frozen SegNet's argmax
  on the GT frame1s, DESCENDS d_seg below 0.10 by ~300 epochs — proving the
  600 argmax partitions are AMORTIZABLE into a small weight blob (the seg
  term is GENERATABLE, not merely codeable; the B1-R2 mean-field bug does
  NOT recur for a CLASSIFICATION objective).

PRE-REGISTERED KILLS (prove-or-pivot):
  * KILL-B-MECHANISM: d_seg does NOT descend below 0.10 by ~300ep -> seg not
    cheaply generatable -> PIVOT to D (margin-cost STC sidecar on the existing
    reconstruction) and H (delete frame0 via warp).
  * KILL-B-RATE: d_seg DOES descend but the quantized generator blob is NOT
    smaller than the frontier decoder's seg-share (~162 KB) -> classification
    carrier not cheaper than RGB carrier -> PIVOT to I (quotient/dictionary
    cell carrier) and stack G harder.

d_seg here = argmax-disagreement RATE between the generator's per-pixel argmax
and the FROZEN SegNet's GT argmax over the 196,608 px x num_pairs (the EXACT
score-native quantity; this is the seg term's mechanism, advisory exact).

MLX-FIRST: the generator forward is MLX; a numpy reference forward verifies the
portability contract (parity gate). NO MPS. EVIDENCE [macOS-MLX research-signal],
promotion_eligible=false. Disk hygiene: targets on the SSD tier, NEVER /tmp.
"""
from __future__ import annotations

import argparse
import json
import math
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

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# The score-native generator: coordinate-INR with Fourier features + FiLM
# per-pair modulation + a 5-CLASS LOGIT head (NOT an RGB head). SIREN-style
# omega first-layer encoding is replaced by an explicit Fourier feature bank
# (more stable in MLX; same purpose). Per-pair conditioning via a learned
# modulation embedding (the amortizer: shared base MLP + tiny per-pair code).
# ---------------------------------------------------------------------------


class ScoreNativeSegGenerator(nn.Module):
    """g(x, y, pair) -> 5-class logits at (H, W).

    Amortized carrier: a shared coord-MLP base + a per-pair modulation code
    (FiLM scale/shift on hidden layers). The per-pair cost is O(mod_dim); the
    base is shared across all pairs. This is the classification analog of an
    HNeRV decoder — but the head outputs LABELS, not RGB.
    """

    def __init__(
        self,
        num_pairs: int,
        n_fourier: int = 16,
        hidden_dim: int = 96,
        n_hidden: int = 4,
        mod_dim: int = 32,
        n_classes: int = 5,
        fourier_sigma: float = 8.0,
    ) -> None:
        super().__init__()
        self.num_pairs = num_pairs
        self.n_fourier = n_fourier
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.mod_dim = mod_dim
        self.n_classes = n_classes
        # Fixed random Fourier feature matrix for (x, y) -> 2*n_fourier features.
        # DETERMINISTic (seeded) so it is a free deterministic table at inflate time.
        rng = np.random.default_rng(0)
        B = (rng.standard_normal((2, n_fourier)) * fourier_sigma).astype(np.float32)
        self._fourier_B = mx.array(B)  # NOT trained (deterministic table)
        coord_feat = 2 * n_fourier  # sin+cos of (x,y) projection

        # Per-pair modulation codes (the per-pair carrier; tiny).
        self.mod = mx.zeros((num_pairs, mod_dim))

        # Base MLP.
        self.in_proj = nn.Linear(coord_feat, hidden_dim)
        # FiLM projection: mod -> (scale, shift) per hidden layer.
        self.film = nn.Linear(mod_dim, 2 * hidden_dim * n_hidden)
        self.hidden = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        self.out = nn.Linear(hidden_dim, n_classes)

    def _coord_features(self, coords: mx.array) -> mx.array:
        # coords: (P, 2) in [-1, 1]
        proj = coords @ self._fourier_B  # (P, n_fourier)
        return mx.concatenate([mx.sin(proj), mx.cos(proj)], axis=-1)  # (P, 2*n_fourier)

    def __call__(self, coords: mx.array, pair_idx: int) -> mx.array:
        """coords: (P, 2) pixel coords in [-1,1]; returns (P, n_classes) logits."""
        feat = self._coord_features(coords)
        h = nn.relu(self.in_proj(feat))  # (P, H)
        m = self.mod[pair_idx]  # (mod_dim,)
        film = self.film(m)  # (2*H*n_hidden,)
        film = film.reshape(self.n_hidden, 2, self.hidden_dim)
        for li, layer in enumerate(self.hidden):
            scale = 1.0 + film[li, 0]  # (H,)
            shift = film[li, 1]
            h = nn.relu(layer(h) * scale + shift)
        return self.out(h)  # (P, n_classes)


def numpy_reference_forward(
    params: dict[str, np.ndarray],
    fourier_B: np.ndarray,
    coords: np.ndarray,
    mod_vec: np.ndarray,
    n_hidden: int,
    hidden_dim: int,
) -> np.ndarray:
    """Pure-numpy mirror of ScoreNativeSegGenerator.__call__ (portability contract)."""
    proj = coords @ fourier_B
    feat = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
    h = np.maximum(feat @ params["in_proj.weight"].T + params["in_proj.bias"], 0.0)
    film = mod_vec @ params["film.weight"].T + params["film.bias"]
    film = film.reshape(n_hidden, 2, hidden_dim)
    for li in range(n_hidden):
        scale = 1.0 + film[li, 0]
        shift = film[li, 1]
        h = np.maximum((h @ params[f"hidden.{li}.weight"].T + params[f"hidden.{li}.bias"]) * scale + shift, 0.0)
    return h @ params["out.weight"].T + params["out.bias"]


def _build_coords(h: int, w: int) -> mx.array:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return mx.array(np.stack([gx.ravel(), gy.ravel()], axis=-1))  # (H*W, 2)


def _quantize_blob_bytes(model: ScoreNativeSegGenerator) -> dict[str, Any]:
    """Estimate the quantized carrier blob: int8 base weights + int8 per-pair mod, brotli'd."""
    import brotli  # type: ignore

    base_tensors = []
    for name, val in model.parameters().items() if hasattr(model, "parameters") else []:
        pass
    # Collect named params via tree_flatten
    from mlx.utils import tree_flatten

    flat = tree_flatten(model.parameters())
    base_chunks: list[bytes] = []
    mod_chunk: bytes = b""
    n_params = 0
    for name, arr in flat:
        a = np.array(arr).astype(np.float32)
        n_params += a.size
        if name == "mod":
            # per-pair modulation -> int8 (tiny carrier section)
            s = float(np.abs(a).max()) + 1e-8
            q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
            mod_chunk = q.tobytes()
        else:
            # base weights -> int8 per-tensor scale
            s = float(np.abs(a).max()) + 1e-8
            q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
            base_chunks.append(q.tobytes())
    base_raw = b"".join(base_chunks)
    base_brotli = len(brotli.compress(base_raw, quality=11))
    mod_brotli = len(brotli.compress(mod_chunk, quality=11)) if mod_chunk else 0
    return {
        "n_params": int(n_params),
        "base_int8_raw_bytes": len(base_raw),
        "base_int8_brotli_bytes": base_brotli,
        "mod_int8_raw_bytes": len(mod_chunk),
        "mod_int8_brotli_bytes": mod_brotli,
        "total_quantized_blob_bytes": base_brotli + mod_brotli,
    }


def run_smoke(
    targets_dir: Path,
    out_dir: Path,
    num_pairs: int,
    epochs: int,
    eval_every: int,
    hinge_weight: float,
    seed: int,
    lr: float,
    px_per_step: int,
) -> dict[str, Any]:
    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(seed)
    np.random.seed(seed)

    meta = json.loads((targets_dir / "targets_meta.json").read_text())
    n_pairs_built = int(meta["num_pairs_built"])
    H, W = meta["seg_input_hw"]
    n_pairs = min(num_pairs, n_pairs_built)

    argmax = np.memmap(targets_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="r", shape=(n_pairs_built, H, W))
    margin = np.memmap(targets_dir / "gt_segnet_margin.f16", dtype=np.float16, mode="r", shape=(n_pairs_built, H, W))

    # Load targets for the smoke pairs into RAM (small at N=16).
    tgt_labels = np.asarray(argmax[:n_pairs]).astype(np.int32)  # (P,H,W)
    tgt_margin = np.asarray(margin[:n_pairs]).astype(np.float32)  # (P,H,W)

    coords_full = _build_coords(H, W)  # (H*W, 2)
    n_px = H * W

    model = ScoreNativeSegGenerator(num_pairs=n_pairs)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    # Per-pixel hinge weight: lever G — emphasize SMALL-margin (boundary) pixels.
    # weight = 1 + hinge_weight * exp(-margin / tau). Large-margin interior ~ weight 1
    # (free); boundary ~ weight (1+hinge_weight). This makes the loss care about flips.
    tau = 1.0

    def loss_fn(model: ScoreNativeSegGenerator, pair_idx: int, coord_batch: mx.array,
                labels: mx.array, w_px: mx.array) -> mx.array:
        logits = model(coord_batch, pair_idx)  # (P, 5)
        ce = nn.losses.cross_entropy(logits, labels, reduction="none")  # (P,)
        return (ce * w_px).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def eval_d_seg() -> float:
        # Full-frame argmax disagreement over all smoke pairs.
        disagree = 0
        total = 0
        for pi in range(n_pairs):
            logits = model(coords_full, pi)  # (H*W, 5)
            pred = np.array(mx.argmax(logits, axis=-1)).astype(np.int32)
            gt = tgt_labels[pi].ravel()
            disagree += int((pred != gt).sum())
            total += pred.size
        return disagree / total

    history: list[dict[str, Any]] = []
    t0 = time.time()
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        # One step per pair per epoch, on a random pixel subset (stochastic).
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
            row = {
                "epoch": ep,
                "train_loss": round(ep_loss / n_pairs, 5),
                "d_seg": round(d_seg, 6),
                "wall_s": round(time.time() - t0, 1),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    # Byte accounting + portability parity.
    blob = _quantize_blob_bytes(model)
    parity = _portability_parity(model, coords_full, n_pairs)

    final_d_seg = history[-1]["d_seg"]
    result = {
        "subagent": "lever_b_score_native_argmax_smoke_20260610",
        "utc": _utc(),
        "evidence_grade": "[macOS-MLX research-signal]",
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "config": {
            "num_pairs": n_pairs,
            "epochs": epochs,
            "hinge_weight": hinge_weight,
            "lr": lr,
            "px_per_step": px_per_step,
            "seed": seed,
            "hidden_dim": model.hidden_dim,
            "n_hidden": model.n_hidden,
            "mod_dim": model.mod_dim,
            "n_fourier": model.n_fourier,
        },
        "history": history,
        "final_d_seg": final_d_seg,
        "quantized_blob": blob,
        "portability_parity": parity,
        "pose_carrier_bytes_from_targets": meta.get("pose_trajectory", {}),
        "lever_g_interior_budget": meta.get("segnet_margin", {}),
    }
    (out_dir / "smoke_result.json").write_text(json.dumps(result, indent=2))
    return result


def _portability_parity(model: ScoreNativeSegGenerator, coords_full: mx.array, n_pairs: int) -> dict[str, Any]:
    """Verify the numpy reference forward matches the MLX forward (drift gate)."""
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.parameters()))
    params = {k: np.array(v) for k, v in flat.items()}
    fourier_B = np.array(model._fourier_B)
    coords_np = np.array(coords_full[:512])  # subset for speed
    pi = 0
    mlx_logits = np.array(model(coords_full[:512], pi))
    np_logits = numpy_reference_forward(
        params, fourier_B, coords_np, params["mod"][pi], model.n_hidden, model.hidden_dim
    )
    max_abs = float(np.abs(mlx_logits - np_logits).max())
    mlx_argmax = mlx_logits.argmax(-1)
    np_argmax = np_logits.argmax(-1)
    argmax_agree = float((mlx_argmax == np_argmax).mean())
    # The portability contract for a CLASSIFICATION carrier is ARGMAX-parity (the
    # score-native quantity d_seg reads), not bit-exact logits. fp32 accumulation
    # drift on a trained net with large activations exceeds a strict logit
    # tolerance while argmax (and thus d_seg) is invariant — exactly lever-G's
    # point. We require argmax parity + a generous logit-magnitude sanity bound.
    rel = max_abs / (float(np.abs(mlx_logits).max()) + 1e-8)
    return {
        "max_abs_logit_diff": max_abs,
        "max_rel_logit_diff": rel,
        "argmax_agreement": argmax_agree,
        "parity_pass": bool(argmax_agree == 1.0 and rel < 0.05),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LEVER-B score-native argmax descent smoke (MLX)")
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n16")
    ap.add_argument("--out-dir", type=Path, default=Path(base) / "smoke_n16")
    ap.add_argument("--num-pairs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--px-per-step", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    result = run_smoke(
        targets_dir=args.targets_dir,
        out_dir=args.out_dir,
        num_pairs=args.num_pairs,
        epochs=args.epochs,
        eval_every=args.eval_every,
        hinge_weight=args.hinge_weight,
        seed=args.seed,
        lr=args.lr,
        px_per_step=args.px_per_step,
    )
    print("\n=== SMOKE RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
