# SPDX-License-Identifier: MIT
"""LEVEL-SET WITNESS through R (MLX) — softmax-of-SDF + curvelet, REALIZED d_seg, pose-legal RGB.

Composes the GO'd level-set representation into the existing realized-through-R witness vehicle
WITHOUT duplicating it: this trainer IMPORTS the RGB-render / R / frozen-MLX-scorer / frozen-
CPU-torch-verdict / EMA / curriculum-loss / byte-close primitives from
``experiments/train_witness_realized_through_R_mlx.py`` (owned by a7660df3 — NOT edited here)
and ``tools/witness_byte_close_and_eval.py`` (byte-close parity), and the SDF/curvelet head +
reg + quantize from ``tac.boundary_math.lever_b_levelset_generator`` (this campaign's module)
and the byte-closeable self-orientation directional basis from ``tac.boundary_math.lever_b_generator``.

THE COMPOSITION (the decisive sub-0.15 vehicle):
  curvelet/shearlet (or self-orientation) front-end  →  FiLM-modulated WIRE/HOSC trunk  →
    (a) K SDF fields phi  (1-Lipschitz level-set partition; argmax_k phi_k = the seg structure)
    (b) per-(pair,frame) RGB texture  (pose-carrying luma+chroma detail)
  RGB = sigmoid( softmax(phi/T) @ palette  +  texture ) * 255      (POSE-LEGAL, not flat palette)
  RGB --R--> frozen SegNet argmax  ==>  REALIZED d_seg  (the SDF makes the COLOR boundary track
  the 1-Lipschitz level set -> the SegNet argmax boundary R-survives, the GO'd -587x lever)
  RGB --R--> frozen PoseNet YUV6   ==>  REALIZED d_pose  (the texture carries pose; the
  stored-pose target is the Quantizr sidecar's GT — pose is solved, witness's job is d_seg)

WHY pose-legal (the coordinator's make-or-break): a flat ``softmax(phi/T)@palette`` frame is
POSE-BLIND (measured S=11.65). The additive per-(pair,frame) ``texture`` head restores the
luma/chroma detail PoseNet's YUV6 needs while the palette term keeps the SegNet argmax pinned to
the SDF partition. d_seg is REALIZED (render -> _torch_R_to_camera_uint8 -> frozen CPU-torch
SegNet argmax), NEVER a field-level proxy.

COMPUTE-SUBSTRATE LAW / NO-FAKE / authority: identical to the imported trainer — MLX (cpu/gpu)
is the fp32 TRAINING-GRADIENT device; the d_seg/d_pose VERDICT is the FROZEN CPU-torch SegNet
argmax + PoseNet MSE (NEVER MLX, NEVER MPS). Evidence ``[macOS-MLX training-gradient]`` /
verdict ``[macOS-CPU advisory]``; promotion_eligible=False; pointer UNMOVED until a byte-closed
exact-eval row (tools/witness_byte_close_and_eval.py) lands sub-0.19110.

BORROWED-SUBSTRATE (NO-FAKE #7): BORROWED = the entire realized-through-R RGB-witness pipeline
(a7660df3), curvelets/shearlets, WIRE/HOSC, FiLM, Eikonal/Chan-Vese, the frozen scorers + CPU
authority. OURS-ORIGINAL = composing the SegNet argmax as a softmax-of-SDF level set whose
1-Lipschitz boundary R-survives, rendered as POSE-LEGAL palette+texture RGB, driven by a generic
(byte-closeable, GT-free) curvelet front-end — the joint R-aliasing + directional-byte-close fix.
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

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── imports from the RGB-witness trainer (a7660df3 — reuse, do NOT edit) ──
from train_witness_realized_through_R_mlx import (  # noqa: E402
    MlxEMA,
    SEG_H,
    SEG_W,
    _build_render_coords,
    _render_rgb_render_res,
    _torch_R_to_camera_uint8,
    cpu_verdict_d_pose_batch,
    cpu_verdict_d_seg_batch,
    implied_score_from_verdict,
    load_gt_from_cache,
    make_loss_fn,
    precompute_gt,
    quantize_witness_blob,
)

# ── imports from this campaign's level-set module + the byte-closeable directional basis ──
from tac.boundary_math.lever_b_generator import self_orientation_directional_feats  # noqa: E402
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    LevelSetConfig,
    curvelet_directional_B,
    curvelet_feats,
    int8_dequant_params,
    levelset_rgb_forward_numpy,
    quantize_levelset_blob,
    save_levelset_npz,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# The LEVEL-SET RGB witness (MLX). Interface-compatible with the imported render/loss path:
#   __call__(coord_feats, code_idx) -> (P, 3) RGB    (used by make_loss_fn / _render_rgb_render_res)
#   call_batch(coord_feats, code_indices) -> (K, P, 3)
#   sdf(coord_feats, code_idx) -> (P, K)             (used by the Eikonal/length reg)
# ---------------------------------------------------------------------------
def build_levelset_rgb_witness(
    num_pairs: int,
    in_feat: int,
    hidden_dim: int,
    n_hidden: int,
    mod_dim: int,
    n_classes: int,
    activation: str,
    softmax_temp: float,
    wire_w0: float,
    wire_s0: float,
    hosc_beta: float,
    hosc_omega: float,
    chroma: bool,
):
    import mlx.core as mx
    import mlx.nn as nn

    class LevelSetRGBWitness(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_pairs = num_pairs
            self.n_hidden = n_hidden
            self.hidden_dim = hidden_dim
            self.n_classes = n_classes
            self.activation = str(activation)
            self.softmax_temp = float(softmax_temp)
            self.wire_w0 = float(wire_w0)
            self.wire_s0 = float(wire_s0)
            self.hosc_beta = float(hosc_beta)
            self.hosc_omega = float(hosc_omega)
            # periodic_omega exposed for parity with the RGB witness verdict-forward convention.
            self.periodic_omega = float(hosc_omega)
            self.wire_scale = float(wire_s0)
            self.chroma = bool(chroma)
            self.code = mx.zeros((num_pairs * 2, mod_dim))
            self.in_proj = nn.Linear(in_feat, hidden_dim)
            self.film = nn.Linear(mod_dim, 2 * hidden_dim * n_hidden)
            self.hidden = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
            self.out_sdf = nn.Linear(hidden_dim, n_classes)     # K SDF fields (LINEAR)
            self.out_tex = nn.Linear(hidden_dim, 3)             # pose-carrying RGB texture
            # learned per-class palette (K,3); init to a distinct luma/chroma spread so classes
            # start separable (sigmoid(palette) spans the color cube).
            pal = np.zeros((n_classes, 3), np.float32)
            for k in range(n_classes):
                t = (k / max(n_classes - 1, 1)) * 2.0 - 1.0
                pal[k] = np.array([t, -t, 0.5 * t], np.float32) * 2.0
            self.palette = mx.array(pal)

        def _act(self, u):
            if self.activation == "wire":
                return mx.cos(self.wire_w0 * u) * mx.exp(-((self.wire_s0 * u) ** 2))
            if self.activation == "hosc":
                return mx.tanh(self.hosc_beta * mx.sin(self.hosc_omega * u))
            return nn.relu(u)

        def _trunk(self, coord_feats, code_idx):
            h = self._act(self.in_proj(coord_feats))
            film = mx.reshape(self.film(self.code[code_idx]), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                h = self._act(layer(h) * (1.0 + film[li, 0]) + film[li, 1])
            return h  # (P, hidden)

        def sdf(self, coord_feats, code_idx):
            return self.out_sdf(self._trunk(coord_feats, code_idx))  # (P, K)

        def _compose_rgb(self, h):
            phi = self.out_sdf(h)                                   # (..., K)
            tex = self.out_tex(h)                                   # (..., 3)
            soft = mx.softmax(phi / self.softmax_temp, axis=-1)     # (..., K)
            base = soft @ self.palette                             # (..., 3) class color (SDF-pinned)
            rgb = mx.sigmoid(base + tex) * 255.0                   # POSE-LEGAL (texture carries pose)
            if not self.chroma:
                luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
                rgb = mx.concatenate([luma, luma, luma], axis=-1)
            return rgb

        def __call__(self, coord_feats, code_idx):
            return self._compose_rgb(self._trunk(coord_feats, code_idx))  # (P, 3)

        def call_batch(self, coord_feats, code_indices):
            h0 = self._act(self.in_proj(coord_feats))               # (P, hidden) shared
            codes = self.code[code_indices]                        # (K, mod)
            film = mx.reshape(self.film(codes), (-1, self.n_hidden, 2, self.hidden_dim))
            h = mx.broadcast_to(h0[None], (film.shape[0], h0.shape[0], h0.shape[1]))
            for li, layer in enumerate(self.hidden):
                h = self._act(layer(h) * (1.0 + film[:, li, 0][:, None, :]) + film[:, li, 1][:, None, :])
            return self._compose_rgb(h)                            # (K, P, 3)

    return LevelSetRGBWitness()


# ---------------------------------------------------------------------------
# MLX level-set regularizers (differentiable twins of the numpy reference). On phi (P,K)
# reshaped to (H,W,K): Eikonal drives |grad phi|->1 (true SDF); length is the Chan-Vese
# boundary-perimeter prior (short, smooth class boundaries). Kept SMALL (topology bias, not
# the driver — the realized seg loss drives d_seg).
# ---------------------------------------------------------------------------
def _eikonal_length_mlx(phi_pk, render_h: int, render_w: int, len_eps: float = 1.0):
    """(fix h) Eikonal + Chan-Vese length on the DECISION MARGIN m = phi_top1 - phi_top2 (the
    quantity the argmax boundary lives on), NOT each field's own zero-set. Eikonal drives
    |grad m|->1 (the 1-Lipschitz margin = the R-survival quantity); the length term
    delta_eps(m)*|grad m| penalizes the perimeter of the ACTUAL inter-class boundary {m=0}."""
    import mlx.core as mx

    phi = mx.reshape(phi_pk, (render_h, render_w, -1))
    srt = mx.sort(phi, axis=-1)
    m = srt[..., -1] - srt[..., -2]  # (H,W) >=0 decision margin (top1-top2)
    gy = m[1:, :] - m[:-1, :]
    gx = m[:, 1:] - m[:, :-1]
    gmag = mx.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2 + 1e-8)  # (H-1,W-1)
    eik = mx.mean((gmag - 1.0) ** 2)
    mc = m[:-1, :-1]
    delta = (len_eps / np.pi) / (len_eps * len_eps + mc * mc)  # delta_eps at the {m=0} boundary
    length = mx.mean(delta * gmag)
    return eik, length, mx.mean(gx * gx) + mx.mean(gy * gy)


# ---------------------------------------------------------------------------
# Curriculum seg_form by epoch (PR95 d_seg sequence): ce -> tau_softplus -> l7_softplus.
# (fix i) Muon is NOT yet wired here — the optimizer is hardcoded AdamW (mlx.optimizers.AdamW
# below). The l7 stage runs under AdamW; the PR95 Muon finisher is a follow-up (no false claim).
# ---------------------------------------------------------------------------
def _seg_form_for_epoch(ep: int, args) -> str:
    if not args.curriculum:
        return args.seg_loss
    if ep < args.tau_softplus_start_epoch:
        return "ce"
    if ep < args.l7_start_epoch:
        return "tau_softplus"
    return "l7_softplus"


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # --- GT (frozen CPU authority) ---
    t0 = time.time()
    if args.gt_cache:
        gt, seg_cpu, posenet_cpu = load_gt_from_cache(Path(args.gt_cache), args.num_pairs)
    else:
        gt, seg_cpu, posenet_cpu = precompute_gt(args.num_pairs)
    P = gt.n_pairs
    print(json.dumps({"stage": "gt", "n_pairs": P, "secs": round(time.time() - t0, 1)}), flush=True)

    render_h, render_w = args.render_h, args.render_w
    coords_np = _build_render_coords(render_h, render_w)

    # --- FRONT-END: generic curvelet/shearlet bank (byte-closeable, GT-free) ---
    bank = CurveletBankConfig(
        n_scales=args.bank_n_scales, n_orient0=args.bank_n_orient0,
        f0=args.bank_f0, base=args.bank_base, n_iso=args.bank_n_iso,
    )
    B = curvelet_directional_B(bank)
    curv_feats_np = curvelet_feats(coords_np, B).astype(np.float32)  # (P, 2*cols)
    in_feat = curv_feats_np.shape[1]
    # SELF-ORIENTATION directional augmentation (byte-closeable; tangent from the witness's OWN
    # argmax, cos 0.89-0.91 vs GT). Recomputed every --reorient-every epochs from the live SDF
    # argmax; concatenated to the curvelet feats. OFF by default (the from-scratch smoke uses
    # curvelet only — self-orientation is a finetune lever needing a roughly-learned partition).
    use_self_orient = bool(args.self_orient)
    n_dir_freqs = int(args.n_dir_freqs)
    if use_self_orient:
        # (fix e) FAIL-CLOSED, NOT a silent crash-stub. Wiring self_orientation_directional_feats
        # is PER-PAIR (tangent from each pair's live SDF argmax, recomputed every --reorient-every)
        # -> per-pair coord_feats, which the imported shared-coord render/loss path does not thread.
        # It is the byte-closeable -48% lever (cos 0.89-0.91 vs GT) and the #1 follow-up, but a
        # half-wired version would diverge train/verdict/byte-close (NO-FAKE). The transfer-probe
        # GATE passed with curvelet-only (generic, byte-closeable), so this is not soundness-blocking.
        raise NotImplementedError(
            "--self-orient is the #1 follow-up (per-pair tangent from the live SDF argmax must "
            "thread per-pair coord_feats through train+verdict+byte-close as ONE codepath). Run "
            "curvelet-only (generic, byte-closeable) until it is wired end-to-end; "
            "self_orientation_directional_feats is imported and ready."
        )
    print(json.dumps({"stage": "front_end", "curvelet_cols": int(B.shape[1]),
                      "in_feat": int(in_feat), "self_orient": use_self_orient,
                      "front_end": "generic-curvelet only (self-orient = fail-closed follow-up)"}), flush=True)

    adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device=args.mlx_device)
    coord_feats_mx = mx.array(curv_feats_np)

    model = build_levelset_rgb_witness(
        num_pairs=P, in_feat=in_feat, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
        mod_dim=args.mod_dim, n_classes=5, activation=args.activation, softmax_temp=args.softmax_temp_start,
        wire_w0=args.wire_w0, wire_s0=args.wire_s0, hosc_beta=args.hosc_beta, hosc_omega=args.hosc_omega,
        chroma=args.chroma,
    )
    mx.eval(model.parameters())
    ema = MlxEMA(model, decay=args.ema_decay)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    base_loss = make_loss_fn(
        adapter, render_h, render_w, score_domain=args.score_domain_loss, pose_eps=args.pose_eps,
        seg_loss=args.seg_loss, tau_softplus_tau=args.tau_softplus_tau, l7_mult=args.l7_mult,
        l7_threshold=args.l7_threshold,
    )

    def total_loss_fn(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w):
        L = base_loss(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form=seg_form)
        phi0 = model.sdf(cf, c0)
        eik, length, _ = _eikonal_length_mlx(phi0, render_h, render_w)
        return L + eik_w * eik + len_w * length

    value_and_grad = nn.value_and_grad(model, total_loss_fn)

    # one-hot L* + margin per pair at the SegNet OUTPUT res (gt.lstars/gt.margins are 384x512,
    # matching the realized seg_logits = adapter.segnet(R(rgb))). NOT render res.
    def _lstar_oh(pi: int):
        lr = np.asarray(gt.lstars[pi])  # (384,512)
        oh = np.eye(5, dtype=np.float32)[lr.ravel()].reshape(lr.shape[0], lr.shape[1], 5)
        mg = np.asarray(gt.margins[pi], np.float32)  # (384,512)
        return mx.array(oh[None]), mx.array(mg[None])

    pose_tgts = [mx.array(np.asarray(gt.gt_poses[pi], np.float32)) for pi in range(P)]
    lstar_cache = [_lstar_oh(pi) for pi in range(P)]

    # ---- realized CPU-torch verdict over a subset (the AUTHORITY trajectory) ----
    vpairs = list(range(0, P, max(1, P // max(args.verdict_pairs, 1)))) if args.verdict_pairs < P else list(range(P))
    vpairs = vpairs[: args.verdict_pairs] if args.verdict_pairs else list(range(P))

    def _render_numpy_deploy(deploy: dict[str, np.ndarray], code_row: np.ndarray) -> np.ndarray:
        """THE ONE CODEPATH (fp32 numpy, deploy-faithful) — same forward the byte-close/inflate use.
        ``deploy`` = int8-dequantized EMA-shadow params; ``curv_feats_np`` = the free curvelet bank."""
        rgb, _phi = levelset_rgb_forward_numpy(
            deploy, curv_feats_np, code_row, n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
            n_classes=5, activation=args.activation, softmax_temp=float(model.softmax_temp),
            wire_w0=args.wire_w0, wire_s0=args.wire_s0, hosc_beta=args.hosc_beta, hosc_omega=args.hosc_omega,
            chroma=args.chroma,
        )
        return _torch_R_to_camera_uint8(rgb.reshape(render_h, render_w, 3))

    def realized_verdict() -> dict[str, float]:
        # (fix a+b+c) verdict the EMA SHADOW, int8-DEQUANTIZED, via the fp32 numpy ONE CODEPATH
        # (NOT the MLX-GPU reduced-precision forward — the 4th artifact). This IS the deploy render.
        ema_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
        deploy = int8_dequant_params(ema_np)
        f0s, f1s = [], []
        for pi in vpairs:
            f0s.append(_render_numpy_deploy(deploy, deploy["code"][2 * pi + 0]))
            f1s.append(_render_numpy_deploy(deploy, deploy["code"][2 * pi + 1]))
        ds = cpu_verdict_d_seg_batch(seg_cpu, f1s, [gt.lstars[pi] for pi in vpairs])
        # pose VERDICT still measured (monitoring) but pose is NOT the witness's job (w_pose=0
        # default; deploy pose rides the SOLVED Quantizr stored-pose sidecar, d_pose 3.4e-5).
        dp = cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, [gt.gt_poses[pi] for pi in vpairs])
        return {"d_seg": float(np.mean(ds)), "d_pose": float(np.mean(dp))}

    history: list[dict[str, Any]] = []
    v0 = realized_verdict()
    blob = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()})
    s0 = implied_score_from_verdict(v0["d_seg"], v0["d_pose"], blob["total_quantized_blob_bytes"])
    print(json.dumps({"stage": "verdict", "epoch": 0, **{k: round(v, 6) for k, v in v0.items()},
                      "blob_bytes": blob["total_quantized_blob_bytes"], "implied_S": round(s0, 4),
                      "axis": "[macOS-CPU advisory] NON-PROMOTABLE"}), flush=True)
    history.append({"epoch": 0, **v0, "implied_S": s0})

    recent_losses: list[float] = []
    with temporary_mlx_device(args.mlx_device):
        for ep in range(1, args.epochs + 1):
            seg_form = _seg_form_for_epoch(ep, args)
            # (config-review #4) ANNEAL softmax-temp hi->lo (cosine): start soft (gradients flow,
            # no RGB-level Gibbs) -> end sharp (the SDF partition pinned). Fixing T=0.1 reintroduces
            # Gibbs at the RGB level per deep-math; anneal like the hosc_beta schedule.
            prog_t = (ep - 1) / max(args.epochs - 1, 1)
            model.softmax_temp = float(args.softmax_temp_end + 0.5 * (args.softmax_temp_start - args.softmax_temp_end) * (1 + np.cos(np.pi * prog_t)))
            # LR warmup->cosine
            if args.lr_schedule:
                if ep <= args.warmup_epochs:
                    lr = args.lr * ep / max(args.warmup_epochs, 1)
                else:
                    prog = (ep - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
                    lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * prog))
                opt.learning_rate = float(lr)
            order = np.random.permutation(P)
            ep_loss = 0.0
            for s in range(0, P, args.accum_pairs):
                chunk = order[s:s + args.accum_pairs]
                accum = None
                lsum = 0.0
                for pi_np in chunk:
                    pi = int(pi_np)
                    oh, mg = lstar_cache[pi]
                    loss, grads = value_and_grad(
                        model, coord_feats_mx, 2 * pi + 0, 2 * pi + 1, oh, mg, pose_tgts[pi],
                        args.w_seg, args.w_pose, args.hinge_weight, args.margin_target_end, seg_form,
                        args.eikonal_weight, args.length_weight,
                    )
                    mx.eval(loss, grads)  # materialize per pair (bound the lazy fwd+bwd graph)
                    lsum += float(loss)
                    accum = grads if accum is None else tree_map(lambda a, b: a + b, accum, grads)
                    mx.eval(accum)
                nb = max(len(chunk), 1)
                batch_loss = lsum / nb
                mean_grads = tree_map(lambda g, c=float(nb): g / c, accum)
                clipped, total = optim.clip_grad_norm(mean_grads, args.grad_clip if args.grad_clip > 0 else 1e30)
                mx.eval(total)
                gnorm = float(total)
                # spike-guard: skip non-finite / >spike_factor x running median.
                med = float(np.median(recent_losses)) if recent_losses else None
                skip = (not np.isfinite(batch_loss)) or (not np.isfinite(gnorm)) or (
                    med is not None and batch_loss > args.spike_factor * med
                )
                if skip:
                    print(json.dumps({"stage": "spike_skip", "ep": ep,
                                      "batch_loss": (round(batch_loss, 4) if np.isfinite(batch_loss) else "nonfinite"),
                                      "gnorm": (round(gnorm, 4) if np.isfinite(gnorm) else "nonfinite")}), flush=True)
                    continue
                opt.update(model, clipped)
                mx.eval(model.parameters(), opt.state)
                ema.update(model)
                mx.eval(list(ema.shadow.values()))
                recent_losses.append(batch_loss)
                if len(recent_losses) > 50:
                    recent_losses.pop(0)
                ep_loss += batch_loss
            if args.mlx_device == "gpu":
                mx.clear_cache()
            if ep % args.eval_every == 0 or ep == args.epochs:
                v = realized_verdict()
                blob = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()})
                s = implied_score_from_verdict(v["d_seg"], v["d_pose"], blob["total_quantized_blob_bytes"])
                print(json.dumps({"stage": "verdict", "epoch": ep, "seg_form": seg_form,
                                  **{k: round(vv, 6) for k, vv in v.items()},
                                  "blob_bytes": blob["total_quantized_blob_bytes"], "implied_S": round(s, 4),
                                  "ep_loss": round(ep_loss, 3)}), flush=True)
                history.append({"epoch": ep, **v, "implied_S": s})

    # (fix c) save the EMA SHADOW (the deploy weights), NOT live. Persist the bank cfg + render
    # cfg so the ONE-CODEPATH byte-close/inflate reconstructs the exact deploy render.
    ck = out_dir / "levelset_witness_ema_mlx.npz"
    flat = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
    flat["__cfg_n_hidden"] = np.asarray(args.n_hidden)
    flat["__cfg_hidden_dim"] = np.asarray(args.hidden_dim)
    flat["__cfg_softmax_temp"] = np.asarray(float(model.softmax_temp))
    flat["__cfg_activation"] = np.asarray(args.activation)
    flat["__cfg_chroma"] = np.asarray(int(bool(args.chroma)))
    flat["__cfg_wire_w0"] = np.asarray(args.wire_w0); flat["__cfg_wire_s0"] = np.asarray(args.wire_s0)
    flat["__cfg_hosc_beta"] = np.asarray(args.hosc_beta); flat["__cfg_hosc_omega"] = np.asarray(args.hosc_omega)
    flat["__bank_n_scales"] = np.asarray(args.bank_n_scales); flat["__bank_n_orient0"] = np.asarray(args.bank_n_orient0)
    flat["__bank_f0"] = np.asarray(args.bank_f0); flat["__bank_base"] = np.asarray(args.bank_base)
    flat["__bank_n_iso"] = np.asarray(args.bank_n_iso); flat["__render_hw"] = np.asarray([render_h, render_w])
    np.savez(ck, **flat)
    result = {
        "utc": _utc(), "n_pairs": P, "epochs": args.epochs, "render_hw": [render_h, render_w],
        "front_end": "curvelet" + ("+self_orient" if use_self_orient else ""),
        "activation": args.activation, "in_feat": int(in_feat),
        "history": history, "checkpoint": str(ck),
        "axis": "[macOS-MLX training-gradient]/[macOS-CPU advisory] verdict; promotion_eligible=false; pointer UNMOVED",
    }
    (out_dir / "levelset_train_result.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LEVEL-SET witness through R (MLX): softmax-of-SDF + curvelet, realized d_seg")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1500,
                    help="(fix d) >=1500 for the PR95 d_seg curriculum (ce->tau->l7). Fail-closed asserted vs curriculum boundaries.")
    ap.add_argument("--eval-every", type=int, default=25)
    # (config-review #1) render-384 is the MEASURED R-survival floor (render-192 pre-caps at
    # 0.00085 d_seg = +0.085 S, mathematically blocking sub-0.15). camera-R + SegNet dominate
    # wall-clock, so 384 is ~free vs 192. The "SDF smooth -> low-res ok" assumption is FALSIFIED.
    ap.add_argument("--render-h", type=int, default=384)
    ap.add_argument("--render-w", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--n-hidden", type=int, default=4)
    # (config-review #2) mod-32 (with hidden-96) -> ~122-130KB at n600 = the RD-optimum B*~122KB
    # (rate 0.081); mod-48/hidden-128 -> 161KB (0.107) overshoots by +0.026 S. n96 = capacity sweep.
    ap.add_argument("--mod-dim", type=int, default=32)
    # (config-review #4) softmax-temp ANNEAL hi->lo (not fixed 0.1, which reintroduces RGB Gibbs).
    ap.add_argument("--softmax-temp-start", type=float, default=1.0, help="anneal START (soft; gradients flow).")
    ap.add_argument("--softmax-temp-end", type=float, default=0.05, help="anneal END (sharp; SDF partition pinned).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema-decay", type=float, default=0.997)
    ap.add_argument("--lr-schedule", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--w-seg", type=float, default=100.0)
    # (fix g) DROP pose-from-texture (the COLLAPSED amortized carrier, d_pose 2.67-12.66). Pose is
    # SOLVED by the Quantizr stored-pose sidecar (3.4e-5); the witness's ONLY binding job is d_seg.
    # w_pose=0 by default -> the texture head serves SegNet realism (seg), not pose reconstruction.
    ap.add_argument("--w-pose", type=float, default=0.0)
    ap.add_argument("--score-domain-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pose-eps", type=float, default=1e-2)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    ap.add_argument("--accum-pairs", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--spike-factor", type=float, default=5.0)
    ap.add_argument("--verdict-pairs", type=int, default=24,
                    help="realized fp32-numpy EMA-shadow verdict subset (0=all); ALWAYS fp32 one-codepath, never mlx-gpu.")
    ap.add_argument("--mlx-device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-cache", type=str, default=None)
    ap.add_argument("--chroma", action=argparse.BooleanOptionalAction, default=True)
    # FRONT-END
    ap.add_argument("--bank-n-scales", type=int, default=4)
    ap.add_argument("--bank-n-orient0", type=int, default=6)
    ap.add_argument("--bank-f0", type=float, default=2.0)
    ap.add_argument("--bank-base", type=float, default=2.0)
    ap.add_argument("--bank-n-iso", type=int, default=4)
    ap.add_argument("--self-orient", action=argparse.BooleanOptionalAction, default=False,
                    help="add byte-closeable self-orientation directional feats (finetune lever; needs a roughly-learned base).")
    ap.add_argument("--n-dir-freqs", type=int, default=6)
    ap.add_argument("--reorient-every", type=int, default=50)
    # ACTIVATION
    # (config-review #3) HOSC is the ONLY descent evidence (probe 0.0066; A/B 0.221 hosc vs 0.265
    # wire). WIRE was a paper-default guess; default HOSC, run wire as a sweep arm.
    ap.add_argument("--activation", choices=["wire", "hosc", "relu"], default="hosc")
    ap.add_argument("--wire-w0", type=float, default=20.0)
    ap.add_argument("--wire-s0", type=float, default=10.0)
    ap.add_argument("--hosc-beta", type=float, default=4.0)
    ap.add_argument("--hosc-omega", type=float, default=1.0)
    # SEG LOSS / CURRICULUM
    ap.add_argument("--seg-loss", choices=["ce", "tau_softplus", "l7_softplus", "margin_hinge"], default="ce")
    ap.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tau-softplus-start-epoch", type=int, default=300)
    ap.add_argument("--l7-start-epoch", type=int, default=800)
    ap.add_argument("--tau-softplus-tau", type=float, default=0.3)
    ap.add_argument("--l7-mult", type=float, default=4.0)
    ap.add_argument("--l7-threshold", type=float, default=1.0)
    ap.add_argument("--margin-target-end", type=float, default=0.5)
    # LEVEL-SET REG
    ap.add_argument("--eikonal-weight", type=float, default=0.01, help="Eikonal |grad phi|->1 (topology bias, small).")
    ap.add_argument("--length-weight", type=float, default=0.001, help="Chan-Vese boundary-length (short smooth boundaries).")
    args = ap.parse_args(argv)

    # (fix d) curriculum boundaries must be strictly ordered and fit inside the budget, else the
    # tau_softplus / l7 stages silently never run (or run for ~0 epochs) -> untrustworthy d_seg.
    if args.curriculum:
        if not (0 < args.tau_softplus_start_epoch < args.l7_start_epoch <= args.epochs):
            raise ValueError(
                f"--curriculum requires 0 < tau_softplus_start_epoch ({args.tau_softplus_start_epoch}) "
                f"< l7_start_epoch ({args.l7_start_epoch}) <= epochs ({args.epochs}). The PR95 d_seg "
                "sequence (ce->tau_softplus->l7) needs each stage to actually run; tau_softplus is "
                "THE primary d_seg drop and must not be skipped."
            )

    result = run_train(args)
    print("\n=== LEVEL-SET WITNESS RESULT (realized through R) ===")
    print(json.dumps({"front_end": result["front_end"], "history": result["history"],
                      "axis": result["axis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
