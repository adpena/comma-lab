# SPDX-License-Identifier: MIT
"""ddm_fp1 — f' PROBE: class-field-projection capacity of the frozen token bottleneck.

The question (gc11 §4 / gc10 sketch-2 / BR-B): how much CLASS-SEMANTIC capacity does
the FROZEN TR1 token bottleneck carry when the OUTPUT SPACE is scorer-native (a K=5
class-field / margin-channel head + deterministic prototype-paint receiver) instead of
the incumbent RGB head?  This CURES the pj1 confound (class-field -> class-field on the
SAME manifold; the QA75 solve FIELD is the teacher, not cross-vehicle solve PIXELS).

f' = realized d_seg (n600, through the REAL path: paint -> bicubic-up-to-camera -> uint8
-> frozen CPU-torch SegNet -> argmax vs GT).  Falsifiers (gc11 §4):
  f' >= 0.0051  -> graft INSTANCE-dead (this head form x this parent); reformulation queue.
  f' <= 5e-4    -> BR-B fires.
  5e-4 < f' <= 2e-3 -> MIXED (the rg3 split says which limit dominates).

DECISIVE-FIRST ordering (MVP): the FLAT-PROTOTYPE-PAINT RECEIVER FLOOR (paint the GT/
teacher argmax -> R -> SegNet) is a LOWER BOUND on f' that is INDEPENDENT of the head.
If the receiver floor already >= 0.0051, the receiver itself is the wall (SPEC_v8 seam
risk; O4) -> INSTANCE-dead decisively, no head training needed.  Else train the head.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED.  Every number here is
[macOS-CPU advisory], score_claim=false, research_only.  Seeded + resumable + atomic.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/Users/adpena/Projects/pact")
DEFAULT_CKPT = (
    "/Volumes/VertigoDataTier/pact/ddm_pa1r_20260730/control_tail/"
    "checkpoints/stage_seg_trunk_tau_final.npz"
)
DEFAULT_TEACHER = "/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730"
DEFAULT_GT_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUT = "/Volumes/VertigoDataTier/pact/ddm_fp1_20260731"
SCORER_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_CLASSES = 5
GRID_H, GRID_W = 24, 32
# QA83 comma10k class luma anchors [Road, Lane, Undrivable, Movable, MyCar] (canonical order).
COMMA10K_ANCHORS = (41.0, 76.0, 90.0, 124.0, 161.0)


# --------------------------------------------------------------------------- io
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


# ------------------------------------------------------------------ teacher/gt
def load_teacher_argmax(teacher_dir: Path, num_pairs: int) -> np.ndarray:
    """(num_pairs, 384, 512) uint8 argmax of the QA75 solve FIELD (the teacher)."""
    out = np.empty((num_pairs, *SCORER_HW), dtype=np.uint8)
    for i in range(num_pairs):
        z = np.load(teacher_dir / f"pair-{i:06d}.npz")
        out[i] = np.asarray(z["argmax"], dtype=np.uint8)
    return out


def load_gt_lstars(gt_cache: Path, num_pairs: int) -> np.ndarray:
    import zipfile

    z = zipfile.ZipFile(gt_cache)
    with z.open("lstars.npy") as f:
        lst = np.lib.format.read_array(f)  # (600,384,512) int64
    return np.ascontiguousarray(lst[:num_pairs].astype(np.int64))


# ------------------------------------------------------------- frozen TR1 model
def load_frozen_module(checkpoint: Path):
    """Build the TR1 module, load the FROZEN EMA-shadow (deploy) weights.  Returns
    (cfg, model).  The renderer + tokens are ALL frozen (only a NEW head trains later)."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    sys.path.insert(0, str(REPO / "experiments"))
    from train_tr1_partition_renderer_mlx import TR1Config, build_module

    stored = np.load(checkpoint, allow_pickle=True)
    meta = json.loads(bytes(stored["meta::json"].tolist()).decode("utf-8"))
    cfg_d = meta["cfg"]
    fields = {f.name for f in dataclasses.fields(TR1Config)}
    unknown = set(cfg_d) - fields
    if unknown:
        raise ValueError(f"checkpoint cfg has keys unknown to TR1Config: {sorted(unknown)}")
    cfg = TR1Config(**{k: v for k, v in cfg_d.items() if k in fields})
    model = build_module(cfg)
    ema = {k[5:]: mx.array(stored[k]) for k in stored.files if k.startswith("ema::")}
    model.update(tree_unflatten([(k, ema[k]) for k in ema]))
    model._quant_engaged = True  # deploy quant on (post-knee endpoint)
    return cfg, model


def trunk_features(model, cfg, idx: int):
    """Replicate render_frame's forward UP TO (not including) the head conv:
    (1, SEG_H, SEG_W, renderer_width) frozen trunk features.  stop_gradient (frozen)."""
    import mlx.core as mx
    import mlx.nn as nn

    x = model.quantized_tokens(idx)[None]  # (1, gh, gw, c)
    x = mx.conv2d(x, model._weight("conv0"), padding=1) + model.b_conv0
    x = nn.gelu(x)
    for k in range(cfg.n_upsample):
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        x = mx.conv2d(x, model._weight(f"up{k}"), padding=1) + getattr(model, f"b_up{k}")
        x = nn.gelu(x)
    return mx.stop_gradient(x)  # (1, SEG_H, SEG_W, W)


# ------------------------------------------------------------- realized R path
def paint_argmax_to_camera_uint8(argmax_hw: np.ndarray, proto: np.ndarray) -> np.ndarray:
    """Deterministic flat-prototype paint receiver: argmax(H,W) -> RGB(H,W,3 float)
    via proto[c] -> bicubic-up-to-camera -> uint8.  Returns (874,1164,3) uint8.
    ``proto`` is (5,3) float in [0,255]."""
    from tac.optimization.ddm_tr1_runtime import bicubic_up_to_camera_float

    painted = proto[argmax_hw.astype(np.int64)]  # (384,512,3) float
    up = bicubic_up_to_camera_float(np.ascontiguousarray(painted, dtype=np.float32))
    return np.ascontiguousarray(np.clip(np.rint(up), 0, 255).astype(np.uint8))


# --------------------------------------------------------- prototype-color solve
def solve_prototypes(args) -> int:
    """Solve the (5,3) margin-optimal prototype colours against the frozen CPU-torch
    SegNet: minimise CE(SegNet(R(paint(GT; proto))), GT) over a sample of pairs.
    Differentiable in proto (paint is linear in proto given fixed GT one-hot)."""
    import torch

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    from tac.boundary_math.seg_core import load_real_segnet

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tel = out_dir / "solve_proto_telemetry.jsonl"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seg = load_real_segnet("cpu")
    for p in seg.parameters():
        p.requires_grad = False

    gt = load_gt_lstars(Path(DEFAULT_GT_CACHE), args.num_pairs)
    rng = np.random.default_rng(args.seed)
    sample_ids = sorted(rng.choice(args.num_pairs, size=min(args.proto_sample, args.num_pairs),
                                   replace=False).tolist())
    # one-hot GT for the sample (float) at 384x512
    gt_oh = np.zeros((len(sample_ids), *SCORER_HW, N_CLASSES), dtype=np.float32)
    for j, i in enumerate(sample_ids):
        for c in range(N_CLASSES):
            gt_oh[j, ..., c] = (gt[i] == c)
    gt_lbl = torch.from_numpy(np.stack([gt[i] for i in sample_ids], 0)).long()  # (S,384,512)
    oh = torch.from_numpy(gt_oh)  # (S,384,512,5)

    # proto = sigmoid(theta)*255, init from comma10k gray anchors.
    a = np.asarray(COMMA10K_ANCHORS, dtype=np.float32) / 255.0
    a = np.clip(a, 1e-3, 1 - 1e-3)
    theta0 = np.log(a / (1 - a))[:, None].repeat(3, axis=1)  # (5,3) gray init
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.proto_lr)

    def forward_ce(batch_ids):
        proto = torch.sigmoid(theta) * 255.0  # (5,3)
        losses = []
        for b in batch_ids:
            painted = torch.einsum("hwc,ck->hwk", oh[b], proto)  # (384,512,3) float
            cam = torch.nn.functional.interpolate(
                painted.permute(2, 0, 1)[None], size=CAMERA_HW,
                mode="bicubic", align_corners=False)  # (1,3,874,1164)
            pair = cam[None]  # (1,1,3,H,W) -> preprocess wants (B,T,C,H,W); T last frame
            pair = torch.cat([pair, pair], dim=1)  # (1,2,3,H,W)
            seg_in = seg.preprocess_input(pair)  # (1,3,384,512)
            logits = seg(seg_in)  # (1,5,384,512)
            losses.append(torch.nn.functional.cross_entropy(logits, gt_lbl[b][None]))
        return torch.stack(losses).mean()

    t0 = time.monotonic()
    order = list(range(len(sample_ids)))
    for step in range(args.proto_steps):
        np.random.default_rng(args.seed + step).shuffle(order)
        bids = order[: args.proto_batch]
        opt.zero_grad()
        loss = forward_ce(bids)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == args.proto_steps - 1:
            proto_now = (torch.sigmoid(theta) * 255.0).detach().numpy()
            row = {"event": "step", "step": step, "ce": float(loss),
                   "elapsed_s": round(time.monotonic() - t0, 1),
                   "proto": proto_now.round(1).tolist()}
            _jsonl(tel, row)
            print(f"[proto] step {step:3d}  ce {float(loss):.4f}  {time.monotonic()-t0:5.0f}s")

    proto_final = (torch.sigmoid(theta) * 255.0).detach().numpy().astype(np.float32)
    gray_proto = np.stack([np.asarray(COMMA10K_ANCHORS)] * 3, axis=1).astype(np.float32)  # (5,3) gray
    _atomic_save_npz(out_dir / "prototypes.npz", {
        "proto_solved": proto_final, "proto_gray": gray_proto,
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
    })
    summary = {"event": "summary", "proto_solved": proto_final.round(2).tolist(),
               "proto_gray_anchor": gray_proto.round(1).tolist(),
               "proto_sample": len(sample_ids), "steps": args.proto_steps,
               "final_ce": float(loss), "axis": "[macOS-CPU advisory]",
               "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED"}
    _jsonl(tel, summary)
    _atomic_write_bytes(out_dir / "prototypes_summary.json",
                        json.dumps(summary, indent=1).encode())
    print(json.dumps(summary, indent=1))
    return 0


# ---------------------------------------------- realized n600 gate (chunked)
def realized_gate(args, argmax_source: np.ndarray, proto: np.ndarray, tag: str) -> dict:
    """Realized d_seg (n600, chunked <=120): paint(argmax_source; proto) -> R -> SegNet
    -> argmax vs GT.  argmax_source (num_pairs,384,512) uint8/int.  Returns receipt dict."""
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    from train_witness_realized_through_R_mlx import cpu_verdict_d_seg_argmax_batch

    from tac.boundary_math.seg_core import load_real_segnet

    gt = load_gt_lstars(Path(DEFAULT_GT_CACHE), args.num_pairs)
    seg = load_real_segnet("cpu")
    out_dir = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    dsegs = np.zeros(args.num_pairs, dtype=np.float64)
    class_flips = np.zeros((args.num_pairs, N_CLASSES), dtype=np.int64)
    class_gt = np.zeros((args.num_pairs, N_CLASSES), dtype=np.int64)
    t0 = time.time()
    for c0 in range(0, args.num_pairs, args.chunk):
        c1 = min(c0 + args.chunk, args.num_pairs)
        cpath = out_dir / f"chunk_{c0:04d}_{c1:04d}.npz"
        if cpath.exists():
            z = np.load(cpath)
            dsegs[c0:c1] = z["dsegs"]
            class_flips[c0:c1] = z["class_flips"]
            class_gt[c0:c1] = z["class_gt"]
            print(f"[skip] {cpath.name}", flush=True)
            continue
        idxs = list(range(c0, c1))
        frames = [paint_argmax_to_camera_uint8(argmax_source[i], proto) for i in idxs]
        cds = np.zeros(len(idxs)); ccf = np.zeros((len(idxs), N_CLASSES), np.int64)
        ccg = np.zeros((len(idxs), N_CLASSES), np.int64)
        for b0 in range(0, len(idxs), args.seg_batch):
            b1 = min(b0 + args.seg_batch, len(idxs))
            gts = [gt[i] for i in idxs[b0:b1]]
            ds, realized = cpu_verdict_d_seg_argmax_batch(seg, frames[b0:b1], gts)
            for j, (d, g) in enumerate(zip(ds, gts, strict=True)):
                k = b0 + j
                cds[k] = d
                flip = realized[j] != g
                for c in range(N_CLASSES):
                    gm = g == c
                    ccg[k, c] = int(gm.sum())
                    ccf[k, c] = int((gm & flip).sum())
        _atomic_save_npz(cpath, {"idxs": np.asarray(idxs), "dsegs": cds,
                                 "class_flips": ccf, "class_gt": ccg})
        dsegs[c0:c1] = cds; class_flips[c0:c1] = ccf; class_gt[c0:c1] = ccg
        print(f"[{tag} {c0}:{c1}] dseg_mean {cds.mean():.7f}  {time.time()-t0:.0f}s", flush=True)

    total_px = float(class_gt.sum())
    per_class = [float(class_flips[:, c].sum()) / total_px for c in range(N_CLASSES)]
    receipt = {
        "schema": "ddm_fp1_realized_gate.v1", "tag": tag,
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False,
        "n_pairs": int(args.num_pairs), "d_seg_mean": float(dsegs.mean()),
        "d_seg_max": float(dsegs.max()), "d_seg_max_pair": int(dsegs.argmax()),
        "per_class_d_seg": per_class,
        "per_class_order": "[Road, Lane, Undrivable, Movable, MyCar]",
        "proto": np.asarray(proto).round(2).tolist(),
        "wall_seconds": round(time.time() - t0, 1),
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
    }
    _atomic_write_bytes(out_dir / "realized_verdict.json",
                        (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode())
    return receipt


def floor_gate(args) -> int:
    """Receiver-floor gate: paint(GT lstars) and paint(teacher argmax) -> R -> SegNet.
    A LOWER BOUND on f' (best-case head).  Runs the solved proto AND the gray-anchor
    proto (P-C flat-paint floor decomposition)."""
    protos = np.load(args.out_dir / "prototypes.npz")
    gt = load_gt_lstars(Path(DEFAULT_GT_CACHE), args.num_pairs)
    which = args.proto_which
    proto = protos["proto_solved"] if which == "solved" else protos["proto_gray"]
    print(f"[floor-gate] proto={which}\n{proto.round(1)}")
    rec_gt = realized_gate(args, gt, proto, tag=f"floor_gt_{which}")
    print("=== RECEIVER FLOOR (paint GT argmax) ===")
    print(json.dumps({k: rec_gt[k] for k in
                      ("d_seg_mean", "d_seg_max", "per_class_d_seg", "wall_seconds")}, indent=1))
    if args.also_teacher:
        teach = load_teacher_argmax(Path(DEFAULT_TEACHER), args.num_pairs)
        agree = float(np.mean([np.mean(teach[i] == gt[i]) for i in range(args.num_pairs)]))
        rec_t = realized_gate(args, teach, proto, tag=f"floor_teacher_{which}")
        print(f"=== teacher-argmax floor (teacher~GT agree {agree:.6f}) ===")
        print(json.dumps({k: rec_t[k] for k in ("d_seg_mean", "d_seg_max")}, indent=1))
    return 0


# --------------------------------------------------------------- head training
def build_head_init(cfg, model, teacher_dir: Path, sample_ids, seed: int):
    """Identity-preserving ridge init of the K=5 head (#208/#532): least-squares fit
    trunk_feats -> teacher logits over a sample so the head STARTS near the teacher field
    (no dead rare-class channel).  Returns (Wk5 (5,3,3,W), bk5 (5,))."""
    import mlx.core as mx

    W = cfg.renderer_width
    # gather (pixel, feat) with 3x3 neighbourhood folded via a 1x1 approx: use the
    # centre-pixel feature -> teacher logit ridge (a 1x1 identity init; the 3x3 conv is
    # then trained).  This gives an identity-preserving, rare-class-protected start.
    feats = []
    tgts = []
    tlbl = []
    subpix = 4000  # per pair, random pixels
    rng = np.random.default_rng(seed)
    for i in sample_ids:
        tf = np.asarray(trunk_features(model, cfg, int(i))[0])  # (384,512,W)
        z = np.load(teacher_dir / f"pair-{i:06d}.npz")
        logit = np.asarray(z["distill_logits"], dtype=np.float32).transpose(1, 2, 0)  # (384,512,5)
        targ = np.asarray(z["argmax"], dtype=np.int64)  # (384,512)
        H, Wd = tf.shape[:2]
        ys = rng.integers(0, H, subpix); xs = rng.integers(0, Wd, subpix)
        feats.append(tf[ys, xs]); tgts.append(logit[ys, xs]); tlbl.append(targ[ys, xs])
    X = np.concatenate(feats, 0).astype(np.float64)  # (N, W)
    Y = np.concatenate(tgts, 0).astype(np.float64)   # (N, 5)
    lbl = np.concatenate(tlbl, 0)                     # (N,) teacher argmax
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], 1)  # bias (float64)
    lam = 1.0
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    B = Xb.T @ Y
    coef = np.linalg.lstsq(A, B, rcond=None)[0]  # (W+1, 5) ridge normal-eqn, robust solve
    Wlin = coef[:-1].T.astype(np.float64)  # (5, W)
    blin = coef[-1].astype(np.float64)     # (5,)
    # ---- #208 RARE-CLASS PROTECTION: per-class bias frequency-calibration so no class
    # (esp. Lane(1)/Movable(3)) is dead at init.  Match head argmax freq to teacher freq
    # (the training target); Platt-style multiplicative-in-logit-space iteration on the sample.
    tgt_freq = np.array([(lbl == c).mean() for c in range(N_CLASSES)]) + 1e-6
    tgt_freq /= tgt_freq.sum()
    base_logit = X @ Wlin.T  # (N,5) fixed linear part
    for _ in range(200):
        am = (base_logit + blin[None]).argmax(1)
        freq = np.array([(am == c).mean() for c in range(N_CLASSES)]) + 1e-6
        blin += 0.5 * (np.log(tgt_freq) - np.log(freq))  # rare classes get a positive boost
    Wlin = Wlin.astype(np.float32); blin = blin.astype(np.float32)
    # place the 1x1 linear map at the CENTRE of a 3x3 conv (5,3,3,W); zeros elsewhere.
    Wk5 = np.zeros((N_CLASSES, 3, 3, W), dtype=np.float32)
    Wk5[:, 1, 1, :] = Wlin
    return mx.array(Wk5), mx.array(blin.astype(np.float32))


def head_forward(model, cfg, idx: int, Wk5, bk5):
    import mlx.core as mx

    tf = trunk_features(model, cfg, idx)  # (1,384,512,W) stop_grad
    return mx.conv2d(tf, Wk5, padding=1) + bk5  # (1,384,512,5) logits


def train_head(args) -> int:
    import mlx.core as mx
    import mlx.optimizers as optim

    cfg, model = load_frozen_module(args.checkpoint)
    out_dir = args.out_dir / "head"
    out_dir.mkdir(parents=True, exist_ok=True)
    tel = out_dir / "train_telemetry.jsonl"
    teacher_dir = Path(DEFAULT_TEACHER)
    gt = load_gt_lstars(Path(DEFAULT_GT_CACHE), args.num_pairs)

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    init_ids = sorted(rng.choice(args.num_pairs, size=min(8, args.num_pairs), replace=False).tolist())
    Wk5, bk5 = build_head_init(cfg, model, teacher_dir, init_ids, args.seed)

    # ---- HEAD-INIT GATE: rendered-init verification (#208/#532) ----
    init_receipt = _verify_head_init(model, cfg, Wk5, bk5, gt, args, teacher_dir)
    _atomic_write_bytes(out_dir / "head_init_receipt.json",
                        json.dumps(init_receipt, indent=1).encode())
    print("=== HEAD-INIT GATE ==="); print(json.dumps(init_receipt, indent=1))
    if init_receipt["dead_class_channels"]:
        print("[FAIL] init has dead rare-class channels; abort per #208")
        return 3

    # preload teacher argmax + margin for training
    teach_arg = load_teacher_argmax(teacher_dir, args.num_pairs)
    margins = np.empty((args.num_pairs, *SCORER_HW), dtype=np.float32)
    for i in range(args.num_pairs):
        margins[i] = np.asarray(np.load(teacher_dir / f"pair-{i:06d}.npz")["distill_margin"],
                                dtype=np.float32)
    # inverse-margin boundary weight (dw1 attack-set law; NOT uniform)
    mw = 1.0 + args.margin_gain * np.exp(-np.maximum(margins, 0.0) / max(args.margin_temp, 1e-6))

    params = {"Wk5": Wk5, "bk5": bk5}

    def loss_fn(params, ids):
        tot = mx.array(0.0)
        for i in ids:
            logit = head_forward(model, cfg, int(i), params["Wk5"], params["bk5"])[0]  # (H,W,5)
            tgt = mx.array(teach_arg[i].astype(np.int32))  # (H,W)
            w = mx.array(mw[i])  # (H,W)
            lse = mx.logsumexp(logit, axis=-1)  # (H,W)
            tgt_logit = mx.take_along_axis(logit, tgt[..., None].astype(mx.int32), axis=-1)[..., 0]
            ce = (lse - tgt_logit) * w  # (H,W)
            tot = tot + mx.sum(ce) / mx.sum(w)
        return tot / len(ids)

    lag = mx.value_and_grad(loss_fn)
    opt = optim.Adam(learning_rate=args.head_lr)

    start_ep = 0; loss_hist: list[float] = []
    state_p = out_dir / "head_state.npz"
    if args.resume and state_p.is_file():
        st = np.load(state_p)
        params["Wk5"] = mx.array(st["Wk5"]); params["bk5"] = mx.array(st["bk5"])
        start_ep = int(st["epoch"]) + 1
        loss_hist = list(np.asarray(st["loss_hist"]))
        print(f"[resume] head from ep {start_ep}")

    t0 = time.monotonic()
    stopped = "epochs_complete"
    for ep in range(start_ep, args.head_epochs):
        order = np.random.default_rng(args.seed * 7 + ep).permutation(args.num_pairs)
        eloss = []
        for s in range(0, args.num_pairs, args.head_batch):
            ids = [int(x) for x in order[s:s + args.head_batch]]
            loss, grads = lag(params, ids)
            opt.update(params, grads)
            mx.eval(params, opt.state)
            eloss.append(float(loss))
        el = float(np.mean(eloss)); loss_hist.append(el)
        _jsonl(tel, {"event": "epoch", "epoch": ep, "ce": el,
                     "elapsed_s": round(time.monotonic() - t0, 1)})
        if ep % 5 == 0 or ep == args.head_epochs - 1:
            print(f"[head] ep {ep:3d}  ce {el:.5f}  {time.monotonic()-t0:5.0f}s")
            _atomic_save_npz(state_p, {"Wk5": np.asarray(params["Wk5"]),
                                       "bk5": np.asarray(params["bk5"]),
                                       "epoch": np.int64(ep),
                                       "loss_hist": np.asarray(loss_hist)})
        if len(loss_hist) >= 20:
            past, now = loss_hist[-11], loss_hist[-1]
            if past > 0 and (past - now) / past < args.head_early_stop:
                stopped = "plateau"; break
        if time.monotonic() - t0 > args.max_wall_seconds:
            stopped = "wall_cap"; break
    _atomic_save_npz(state_p, {"Wk5": np.asarray(params["Wk5"]),
                               "bk5": np.asarray(params["bk5"]),
                               "epoch": np.int64(ep), "loss_hist": np.asarray(loss_hist)})
    summ = {"event": "summary", "stopped": stopped, "epochs_run": len(loss_hist),
            "final_ce": loss_hist[-1] if loss_hist else None,
            "first_ce": loss_hist[0] if loss_hist else None,
            "axis": "[macOS-CPU advisory]", "score_claim": False}
    _jsonl(tel, summ); print(json.dumps(summ, indent=1))
    return 0


def _verify_head_init(model, cfg, Wk5, bk5, gt, args, teacher_dir) -> dict:
    """Render the initialized head argmax on a sample; check all 5 classes present +
    per-class mass vs GT priors (rare-class Lane/Movable not dead)."""
    rng = np.random.default_rng(args.seed + 99)
    ids = sorted(rng.choice(args.num_pairs, size=min(8, args.num_pairs), replace=False).tolist())
    head_mass = np.zeros(N_CLASSES)
    gt_mass = np.zeros(N_CLASSES)
    tot = 0
    match = []
    for i in ids:
        logit = np.asarray(head_forward(model, cfg, int(i), Wk5, bk5)[0])  # (H,W,5)
        am = logit.argmax(-1)
        for c in range(N_CLASSES):
            head_mass[c] += int((am == c).sum())
            gt_mass[c] += int((gt[i] == c).sum())
        tot += am.size
        match.append(float(np.mean(am == gt[i])))
    dead = [c for c in range(N_CLASSES) if head_mass[c] == 0]
    return {"schema": "ddm_fp1_head_init.v1", "sample_ids": ids,
            "head_class_frac": (head_mass / tot).round(5).tolist(),
            "gt_class_frac": (gt_mass / tot).round(5).tolist(),
            "init_argmax_match_gt_mean": round(float(np.mean(match)), 5),
            "dead_class_channels": dead,
            "init_mode": "identity_preserving_ridge_1x1_center (#208/#532)",
            "note": "rare-class protected iff Lane(1)/Movable(3) frac > 0"}


def fprime_gate(args) -> int:
    """Realized f' gate: paint(argmax(trained head)) -> R -> SegNet -> d_seg + per-class."""
    cfg, model = load_frozen_module(args.checkpoint)
    st = np.load(args.out_dir / "head" / "head_state.npz")
    import mlx.core as mx
    Wk5 = mx.array(st["Wk5"]); bk5 = mx.array(st["bk5"])
    # compute head argmax for all pairs
    head_arg = np.empty((args.num_pairs, *SCORER_HW), dtype=np.uint8)
    for i in range(args.num_pairs):
        logit = np.asarray(head_forward(model, cfg, i, Wk5, bk5)[0])
        head_arg[i] = logit.argmax(-1).astype(np.uint8)
    protos = np.load(args.out_dir / "prototypes.npz")
    proto = protos["proto_solved"] if args.proto_which == "solved" else protos["proto_gray"]
    rec = realized_gate(args, head_arg, proto, tag=f"fprime_{args.proto_which}")
    print("=== f' (trained head realized) ===")
    print(json.dumps({k: rec[k] for k in
                      ("d_seg_mean", "d_seg_max", "per_class_d_seg", "wall_seconds")}, indent=1))
    return 0


# --------------------------------------------------------------------- argparse
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["solve-proto", "floor-gate", "train-head", "fprime-gate"])
    ap.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CKPT))
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--seg-batch", type=int, default=12)
    # proto solve
    ap.add_argument("--proto-sample", type=int, default=32)
    ap.add_argument("--proto-steps", type=int, default=150)
    ap.add_argument("--proto-batch", type=int, default=6)
    ap.add_argument("--proto-lr", type=float, default=0.05)
    # gate
    ap.add_argument("--proto-which", choices=["solved", "gray"], default="solved")
    ap.add_argument("--also-teacher", action="store_true")
    # head
    ap.add_argument("--head-epochs", type=int, default=60)
    ap.add_argument("--head-batch", type=int, default=8)
    ap.add_argument("--head-lr", type=float, default=1e-2)
    ap.add_argument("--head-early-stop", type=float, default=1e-3)
    ap.add_argument("--margin-gain", type=float, default=8.0)
    ap.add_argument("--margin-temp", type=float, default=0.5)
    ap.add_argument("--max-wall-seconds", type=float, default=6000.0)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "solve-proto":
        return solve_prototypes(args)
    if args.cmd == "floor-gate":
        return floor_gate(args)
    if args.cmd == "train-head":
        return train_head(args)
    if args.cmd == "fprime-gate":
        return fprime_gate(args)
    return 1


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "upstream"))
    raise SystemExit(main())
