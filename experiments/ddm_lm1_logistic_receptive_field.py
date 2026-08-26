#!/usr/bin/env python3
"""ddm_lm1 stage 2 -- a LEARNED PARAMETRIC model with a large causal receptive
field, fitted on a temporally-stratified train split and scored on held-out
frames, with its parameters serialised and coded so W is a REAL byte count.

WHY THIS EXISTS SEPARATELY FROM STAGE 1
---------------------------------------
Stage 1 (`ddm_lm1_learned_model_falsifier.py`) scores DISCRETE context models:
count tables over k causal neighbours.  Its hindsight-ideal is a rigorous lower
bound for ANY model whose input is those k neighbours -- but only for those k.
A count table cannot represent a large receptive field: the context count is
6**k, so k = 278 is not merely expensive, it is impossible.

The shipped HPAC network's power comes from exactly the thing a table lacks:
CONTINUOUS PARAMETER SHARING over a large receptive field.  Read at source,
`cpr1/hpac_integer.py` stacks a masked 7x7 (`conv_a`), a dilated-2 5x5 and a
dilated-4 3x3 depthwise pair (`conv_b1`, `conv_b2`) plus a 3x3 convolution over
the whole previous frame (`conv_past`), 64 channels wide -- an effective causal
receptive field on the order of 25x25, at ~9k int8 parameters.

So stage 1 cannot close the family on its own, and saying otherwise would be a
negative-existence overclaim.  This module supplies the missing half: the
smallest honest model class that DOES share parameters across a large receptive
field -- a multinomial logistic model over one-hot causal neighbours, which is
exactly a linear layer over an EmbeddingBag and is trainable on CPU at $0.

    logit[k] = bias[k] + Sum_j  Wt[slot_j, value_j, k]

Every slot is a causal offset; every value is one of 5 classes or UNK.  The
parameter count is n_slots * 6 * 5, which is linear in receptive-field area
rather than exponential, so R can grow until W is spent.

SPLIT
-----
Temporally stratified by CONTIGUOUS BLOCK, never a prefix: `ddm_bp2` measured
that a prefix of this population is a different population (pose prefixes read
2.54-4.21x harder).  600 frames are cut into 20 blocks of 30; test takes every
4th block.  Blocks keep adjacent-frame correlation inside one side of the split
instead of leaking it across, which an interleaved split would not do.

W IS MEASURED, NOT COUNTED IN PARAMETERS
----------------------------------------
The fitted weights are symmetrically quantised to int8 per output class, packed,
and compressed with the real section coders (brotli / lzma, whichever is
smaller -- the same choice the archive makes).  W is the length of those bytes.
Held-out code length is then recomputed in float64 from the DEQUANTISED weights,
so the reported stream bytes are the bytes a decoder holding exactly those W
bytes would actually emit.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    import brotli  # type: ignore
except Exception:  # pragma: no cover - optional; lzma is always present
    brotli = None

from ddm_lm1_learned_model_falsifier import (
    DEMAND_BYTES,
    FIELD_DIR,
    HPAC_DELTA,
    N_CLASS,
    N_FRAMES,
    SHIPPED_TOTAL_BYTES,
    UNK,
    W_FRAME,
    H,
    _shift,
    bars_for,
    group_index,
    load_field,
)

ALPHABET = 6
TOTAL_POSITIONS = N_FRAMES * H * W_FRAME

BLOCK_LEN = 30
N_BLOCKS = N_FRAMES // BLOCK_LEN  # 20
TEST_BLOCK_STRIDE = 4  # test = blocks 1, 5, 9, 13, 17 -> 150 frames = 25%
TEST_BLOCK_PHASE = 1


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def split_frames() -> tuple[np.ndarray, np.ndarray]:
    """Contiguous-block temporally stratified split. Deterministic, no RNG."""
    blocks = np.arange(N_BLOCKS)
    test_blocks = blocks[blocks % TEST_BLOCK_STRIDE == TEST_BLOCK_PHASE]
    is_test = np.zeros(N_FRAMES, dtype=bool)
    for b in test_blocks:
        is_test[b * BLOCK_LEN : (b + 1) * BLOCK_LEN] = True
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def build_slots(r_cur: int, r_prv: int, r_prv2: int) -> list[tuple[str, int, int]]:
    """Causal slot set. Current frame keeps only mask-A offsets (dx + 2dy < 0)."""
    slots: list[tuple[str, int, int]] = []
    for dy in range(-r_cur, r_cur + 1):
        for dx in range(-r_cur, r_cur + 1):
            if dx + HPAC_DELTA * dy < 0:
                slots.append(("cur", dy, dx))
    for dy in range(-r_prv, r_prv + 1):
        for dx in range(-r_prv, r_prv + 1):
            slots.append(("prv", dy, dx))
    for dy in range(-r_prv2, r_prv2 + 1):
        for dx in range(-r_prv2, r_prv2 + 1):
            slots.append(("prv2", dy, dx))
    return slots


def causal_mask_for(groups: np.ndarray, dy: int, dx: int) -> np.ndarray:
    mask = np.zeros((H, W_FRAME), dtype=bool)
    ys0, ys1 = max(0, -dy), min(H, H - dy)
    xs0, xs1 = max(0, -dx), min(W_FRAME, W_FRAME - dx)
    if ys0 < ys1 and xs0 < xs1:
        mask[ys0:ys1, xs0:xs1] = (
            groups[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx] < groups[ys0:ys1, xs0:xs1]
        )
    return mask


class SlotFeatures:
    """Lazily produces the (n_slots,) causal value vector for chosen positions."""

    def __init__(self, tokens: np.ndarray, slots: list[tuple[str, int, int]], groups: np.ndarray):
        self.tokens = tokens
        self.slots = slots
        self.masks = {
            (dy, dx): causal_mask_for(groups, dy, dx)
            for kind, dy, dx in slots
            if kind == "cur"
        }
        self.blank = np.full((H, W_FRAME), UNK, dtype=np.uint8)

    def frame_planes(self, f: int) -> np.ndarray:
        """(n_slots, H*W) uint8 causal values for frame f."""
        cur = self.tokens[f]
        prv = self.tokens[f - 1] if f >= 1 else self.blank
        prv2 = self.tokens[f - 2] if f >= 2 else self.blank
        out = np.empty((len(self.slots), H * W_FRAME), dtype=np.uint8)
        for j, (kind, dy, dx) in enumerate(self.slots):
            if kind == "cur":
                vals = np.where(self.masks[(dy, dx)], _shift(cur, dy, dx), np.uint8(UNK))
            elif kind == "prv":
                vals = _shift(prv, dy, dx)
            else:
                vals = _shift(prv2, dy, dx)
            out[j] = vals.ravel()
        return out


def slot_offsets(n_slots: int) -> np.ndarray:
    return (np.arange(n_slots, dtype=np.int64) * ALPHABET)[:, None]


def forward_logits(weights: np.ndarray, bias: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """idx: (n_slots, n_pos) flat weight rows. weights: (n_slots*6, 5)."""
    return weights[idx].sum(axis=0) + bias


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def quantise_int8(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-class symmetric int8 quantisation."""
    scale = np.abs(weights).max(axis=0) / 127.0
    scale = np.where(scale <= 0, 1.0, scale)
    q = np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)
    return q, scale


def serialise_model(q: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> tuple[bytes, str]:
    """Real bytes: packed int8 weights + fp16 scales + fp16 bias, best real coder."""
    raw = q.tobytes() + scale.astype(np.float16).tobytes() + bias.astype(np.float16).tobytes()
    cands: dict[str, bytes] = {
        "lzma": lzma.compress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 16, "lc": 1, "lp": 0, "pb": 0}],
        ),
        "bz2": bz2.compress(raw, 9),
    }
    if brotli is not None:
        cands["brotli"] = brotli.compress(raw, quality=11)
    best = min(cands.items(), key=lambda kv: len(kv[1]))
    return best[1], best[0]


def train(
    feats: SlotFeatures,
    tokens: np.ndarray,
    train_frames: np.ndarray,
    n_slots: int,
    epochs: int,
    samples_per_epoch: int,
    lr: float,
    weight_decay: float,
    seed: int,
    chunk: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    # Scatter working set is chunk * n_slots; hold it near 2M entries so wide
    # receptive fields do not blow the gradient buffer.
    chunk = chunk or max(4_096, 2_000_000 // max(1, n_slots))
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_slots * ALPHABET, N_CLASS), dtype=np.float64)
    bias = np.zeros(N_CLASS, dtype=np.float64)
    # ADAGRAD, not Adam.  The features are one-hot, so a weight row receives a
    # gradient only on the steps where its (slot, value) actually occurs.  Adam's
    # moment buffers decay on EVERY step regardless, so a rare row accumulates a
    # near-zero v and then takes an enormous lr/sqrt(v) step -- measured here as
    # outright divergence (train loss 0.0326 -> 0.0487 -> 0.0984 bits/sym).
    # Adagrad's accumulator only grows where the gradient is nonzero, so the
    # dense update is exactly equivalent to the sparse one and rare rows stay put.
    g_w = np.zeros_like(weights)
    g_b = np.zeros_like(bias)
    off = slot_offsets(n_slots)
    eps = 1e-10

    trajectory: list[float] = []
    per_frame_samples = max(1, samples_per_epoch // len(train_frames))
    for ep in range(epochs):
        order = rng.permutation(train_frames)
        tot_bits, tot_n = 0.0, 0
        for f in order:
            planes = feats.frame_planes(int(f))
            pick = rng.integers(0, H * W_FRAME, size=per_frame_samples)
            y = tokens[f].reshape(-1)[pick].astype(np.int64)
            idx = planes[:, pick].astype(np.int64) + off
            for lo in range(0, y.size, chunk):
                sl = slice(lo, min(lo + chunk, y.size))
                sub, ysub = idx[:, sl], y[sl]
                p = softmax_rows(forward_logits(weights, bias, sub))
                n = ysub.size
                tot_bits += -np.log2(np.clip(p[np.arange(n), ysub], 1e-15, 1.0)).sum()
                tot_n += n
                diff = p
                diff[np.arange(n), ysub] -= 1.0
                diff /= n
                gw = np.zeros_like(weights)
                flat = sub.ravel()
                # sub is (n_slots, n) so flat[j*n + i] = idx[j, i]; the matching
                # weight is diff[i, k], which np.tile lays out in exactly that
                # order at a fraction of the cost of repeat+reshape+transpose.
                for k in range(N_CLASS):
                    gw[:, k] = np.bincount(
                        flat,
                        weights=np.tile(diff[:, k], n_slots),
                        minlength=weights.shape[0],
                    )
                if weight_decay:
                    gw += weight_decay * weights
                gb = diff.sum(axis=0)
                g_w += gw * gw
                g_b += gb * gb
                weights -= lr * gw / (np.sqrt(g_w) + eps)
                bias -= lr * gb / (np.sqrt(g_b) + eps)
        epoch_bits = tot_bits / max(tot_n, 1)
        trajectory.append(epoch_bits)
        print(f"    epoch {ep+1:3d}/{epochs}  train {epoch_bits:.6f} bits/sym", flush=True)
    return weights, bias, trajectory


def evaluate(
    feats: SlotFeatures,
    tokens: np.ndarray,
    frames: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    n_slots: int,
    chunk: int = 0,
) -> tuple[float, int]:
    chunk = chunk or max(4_096, 4_000_000 // max(1, n_slots))
    off = slot_offsets(n_slots)
    total_bits, total_n = 0.0, 0
    for f in frames:
        planes = feats.frame_planes(int(f))
        y = tokens[f].reshape(-1).astype(np.int64)
        for lo in range(0, y.size, chunk):
            sl = slice(lo, min(lo + chunk, y.size))
            idx = planes[:, sl].astype(np.int64) + off
            p = softmax_rows(forward_logits(weights, bias, idx))
            ysub = y[sl]
            total_bits += -np.log2(
                np.clip(p[np.arange(ysub.size), ysub], 1e-15, 1.0)
            ).sum()
            total_n += ysub.size
    return total_bits, total_n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--field-dir", type=Path, default=FIELD_DIR)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--r-cur", type=int, default=4)
    ap.add_argument("--r-prv", type=int, default=4)
    ap.add_argument("--r-prv2", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--samples-per-epoch", type=int, default=2_000_000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--eval-frame-stride", type=int, default=1)
    ap.add_argument("--tag", type=str, default="rf")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    retained = args.out_dir / "retained"
    retained.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    field = load_field(args.field_dir, verify_sha=False)
    groups = group_index()
    slots = build_slots(args.r_cur, args.r_prv, args.r_prv2)
    n_slots = len(slots)
    n_params = n_slots * ALPHABET * N_CLASS
    print(
        f"[lm1-rf] slots={n_slots} (cur r={args.r_cur}, prv r={args.r_prv}, "
        f"prv2 r={args.r_prv2})  params={n_params:,}",
        flush=True,
    )

    train_frames, test_frames = split_frames()
    print(
        f"[lm1-rf] split: train {train_frames.size} frames, test {test_frames.size} frames "
        f"(contiguous blocks of {BLOCK_LEN}, test phase {TEST_BLOCK_PHASE} stride {TEST_BLOCK_STRIDE})",
        flush=True,
    )

    feats = SlotFeatures(field.tokens, slots, groups)
    weights, bias, trajectory = train(
        feats,
        field.tokens,
        train_frames,
        n_slots,
        args.epochs,
        args.samples_per_epoch,
        args.lr,
        args.weight_decay,
        args.seed,
    )

    q, scale = quantise_int8(weights)
    payload, coder = serialise_model(q, scale, bias)
    model_bytes = len(payload)
    deq = q.astype(np.float64) * scale
    (retained / f"model_{args.tag}.bin").write_bytes(payload)
    np.savez_compressed(
        retained / f"weights_{args.tag}.npz", q=q, scale=scale, bias=bias
    )
    print(
        f"[lm1-rf] model serialised: {model_bytes:,} B via {coder} "
        f"(raw {q.nbytes + scale.nbytes + bias.nbytes:,} B)",
        flush=True,
    )

    eval_frames = test_frames[:: args.eval_frame_stride]
    print(f"[lm1-rf] evaluating held-out on {eval_frames.size} frames ...", flush=True)
    ho_bits, ho_n = evaluate(
        feats, field.tokens, eval_frames, deq, bias, n_slots
    )
    tr_eval = train_frames[:: max(1, args.eval_frame_stride * 4)]
    tr_bits, tr_n = evaluate(feats, field.tokens, tr_eval, deq, bias, n_slots)

    bits_per_sym = ho_bits / ho_n
    full_field_bytes = bits_per_sym * TOTAL_POSITIONS / 8.0
    shipped_bits_per_sym = field.shipped_bits_total / TOTAL_POSITIONS

    bars = bars_for(full_field_bytes, model_bytes)
    result = {
        "arm": "ddm_lm1",
        "stage": "2_logistic_receptive_field",
        "axis": "[macOS-CPU advisory / scorer-free lossless coding measurement]",
        "score_claim": False,
        "promotable": False,
        "git_sha": _git_sha(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "config": {
            "r_cur": args.r_cur,
            "r_prv": args.r_prv,
            "r_prv2": args.r_prv2,
            "n_slots": n_slots,
            "n_params": n_params,
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "split": {
            "block_len": BLOCK_LEN,
            "n_blocks": N_BLOCKS,
            "test_block_stride": TEST_BLOCK_STRIDE,
            "test_block_phase": TEST_BLOCK_PHASE,
            "train_frames": int(train_frames.size),
            "test_frames": int(test_frames.size),
            "evaluated_test_frames": int(eval_frames.size),
            "stratification": "contiguous blocks, NOT a prefix (ddm_bp2 prefix-bias law)",
        },
        "optimizer": "adagrad (sparse-correct for one-hot features; dense Adam DIVERGES here)",
        "train_loss_trajectory_bits_per_symbol": trajectory,
        "train_loss_last_epoch_relative_drop": (
            (trajectory[-2] - trajectory[-1]) / trajectory[-2] if len(trajectory) >= 2 else None
        ),
        "convergence_note": (
            "The objective is CONVEX (multinomial logistic over one-hot features), so an "
            "undertrained fit yields an UPPER bound on the achievable code length and any "
            "negative drawn from it is correspondingly weak. The trajectory is recorded so the "
            "plateau is auditable rather than asserted."
        ),
        "model_bytes_W": model_bytes,
        "model_coder": coder,
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "heldout_bits_per_symbol": bits_per_sym,
        "heldout_positions": int(ho_n),
        "train_bits_per_symbol": tr_bits / tr_n,
        "generalisation_gap_bits_per_symbol": bits_per_sym - tr_bits / tr_n,
        "shipped_bits_per_symbol": shipped_bits_per_sym,
        "ratio_heldout_to_shipped": bits_per_sym / shipped_bits_per_sym,
        "extrapolated_full_field_stream_bytes": full_field_bytes,
        "extrapolation_method": (
            "held-out bits/symbol * 117,964,800 positions. Assumes the test blocks are "
            "representative of the full field on the RATE axis; the train/test bits-per-symbol "
            "gap is reported so the reader can bound the optimism."
        ),
        "bars": bars,
        "elapsed_seconds": time.time() - t0,
    }
    out = args.out_dir / f"RESULT_stage2_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"\n=== stage 2 ({args.tag}) ===")
    print(f"  W (real coded model bytes)     {model_bytes:>12,} B")
    print(f"  held-out bits/symbol           {bits_per_sym:.8f}")
    print(f"  shipped bits/symbol            {shipped_bits_per_sym:.8f}")
    print(f"  ratio to shipped               {bits_per_sym/shipped_bits_per_sym:.4f}x")
    print(f"  extrapolated stream            {full_field_bytes:>12,.1f} B")
    print(f"  total (stream + W)             {bars['total_bytes']:>12,.1f} B")
    print(f"  shipped total                  {SHIPPED_TOTAL_BYTES:>12,} B")
    print(f"  net vs shipped                 {bars['net_vs_shipped_bytes']:>12,.1f} B")
    print(f"  clears break-even              {bars['clears_break_even']}")
    print(f"  clears demand ({DEMAND_BYTES:,} B)      {bars['clears_demand']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
