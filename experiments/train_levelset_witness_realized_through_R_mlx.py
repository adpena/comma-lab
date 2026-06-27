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
import os
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
    render_through_R_mlx,
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
    rebuild_per_pair_feats_in_place,
    save_levelset_npz,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# INTERMEDIATE CHECKPOINT + RESUME (FEED-dz, additive, default-off). The trainer historically saved
# the EMA-shadow npz ONLY at loop-end -> a multi-day n600 run is non-resumable (crash = total loss)
# + no early byte-close. These pure-numpy (MLX-free, unit-testable) helpers let the run loop write
# a deploy EMA checkpoint (the byte-close ONE-CODEPATH consumes it) AND a separate resume-state
# sidecar (live weights + EMA shadow + optimizer + epoch) every --ckpt-every epochs, atomically.
#
# DESIGN (NO-FAKE / EMA non-negotiable / byte-close clean):
#   * ``levelset_witness_ema_mlx.npz`` = the EMA SHADOW (deploy weights, NOT live) + ``__cfg_*`` /
#     ``__bank_*`` / ``__render_hw`` scalars. EXACTLY what tools/levelset_byte_close_and_eval.py
#     reads (params = unprefixed keys; cfg = ``__``-prefixed, read selectively). Adding new ``__cfg_*``
#     provenance keys is harmless (byte-close ``.get(...)``s the ones it knows + ignores the rest).
#   * ``levelset_resume_state.npz`` = SEPARATE sidecar (so the EMA npz stays byte-close-clean). Live
#     model params (``liveP__*``), EMA shadow (``emaP__*``), optimizer state (``optP__*``, best-effort),
#     + ``__resume_epoch``. Self-orient dir-feats are NOT stored (they are O(GBs) at n600 and are
#     deterministically regenerable from the EMA argmax fixed-point at resume -> recompute, no bloat).
#   * Atomic write: tmp + os.replace (no partial/corrupt npz if the process dies mid-write).
# ---------------------------------------------------------------------------
_RESUME_LIVE_PREFIX = "liveP__"
_RESUME_EMA_PREFIX = "emaP__"
_RESUME_OPT_PREFIX = "optP__"


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Atomic ``np.savez`` (tmp + os.replace) per the durable-state discipline. Refuses /tmp.

    np.savez given a *file object* writes the zip directly (no implicit ``.npz`` suffix append), so
    the temp path is replaced onto the final path atomically on the same filesystem.
    """
    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def _build_ema_checkpoint_arrays(
    shadow_np: dict[str, np.ndarray], *, args: Any, softmax_temp: float,
    render_h: int, render_w: int, epoch: int, in_feat: int,
) -> dict[str, np.ndarray]:
    """The deploy (byte-close) npz contents: EMA SHADOW params + cfg scalars. MLX-free.

    Reproduces EVERY key the loop-end save historically wrote (so the byte-close path is unchanged)
    and ADDS provenance keys (``__epoch`` + the self-orient/curriculum/w_pose scalars the trainer
    previously did NOT persist -- the gap flagged in tools/levelset_byte_close_and_eval.py)."""
    flat: dict[str, np.ndarray] = {k: np.asarray(v, np.float32) for k, v in shadow_np.items()}
    # ---- EXISTING keys (loop-end save parity; do NOT change names/encodings) ----
    flat["__cfg_n_hidden"] = np.asarray(args.n_hidden)
    flat["__cfg_hidden_dim"] = np.asarray(args.hidden_dim)
    flat["__cfg_softmax_temp"] = np.asarray(float(softmax_temp))
    flat["__cfg_activation"] = np.asarray(args.activation)
    flat["__cfg_chroma"] = np.asarray(int(bool(args.chroma)))
    flat["__cfg_wire_w0"] = np.asarray(args.wire_w0)
    flat["__cfg_wire_s0"] = np.asarray(args.wire_s0)
    flat["__cfg_hosc_beta"] = np.asarray(args.hosc_beta)
    flat["__cfg_hosc_omega"] = np.asarray(args.hosc_omega)
    flat["__bank_n_scales"] = np.asarray(args.bank_n_scales)
    flat["__bank_n_orient0"] = np.asarray(args.bank_n_orient0)
    flat["__bank_f0"] = np.asarray(args.bank_f0)
    flat["__bank_base"] = np.asarray(args.bank_base)
    flat["__bank_n_iso"] = np.asarray(args.bank_n_iso)
    flat["__render_hw"] = np.asarray([render_h, render_w])
    flat["__cfg_max_bank_freq"] = np.asarray(-1.0 if args.max_bank_freq is None else float(args.max_bank_freq))
    flat["__cfg_lane_edge_weight"] = np.asarray(float(args.lane_edge_weight))
    flat["__cfg_lane_edge_class"] = np.asarray(int(args.lane_edge_class))
    # ---- NEW provenance (additive; closes the self-orient/curriculum trainer-persist gap) ----
    flat["__epoch"] = np.asarray(int(epoch))
    flat["__cfg_in_feat"] = np.asarray(int(in_feat))
    flat["__cfg_self_orient"] = np.asarray(int(bool(args.self_orient)))
    flat["__cfg_n_dir_freqs"] = np.asarray(int(args.n_dir_freqs))
    flat["__cfg_freq_across"] = np.asarray(float(args.freq_across))
    flat["__cfg_freq_along"] = np.asarray(float(args.freq_along))
    flat["__cfg_reorient_every"] = np.asarray(int(args.reorient_every))
    flat["__cfg_w_pose"] = np.asarray(float(args.w_pose))
    flat["__cfg_curriculum"] = np.asarray(int(bool(args.curriculum)))
    flat["__cfg_tau_softplus_start_epoch"] = np.asarray(int(args.tau_softplus_start_epoch))
    flat["__cfg_l7_start_epoch"] = np.asarray(int(args.l7_start_epoch))
    return flat


def _build_resume_state_arrays(
    live_np: dict[str, np.ndarray], ema_np: dict[str, np.ndarray],
    opt_np: dict[str, np.ndarray] | None, *, args: Any, epoch: int, in_feat: int,
) -> dict[str, np.ndarray]:
    """The resume-state sidecar contents (NOT byte-close-read): prefixed live / EMA / optimizer
    tensors + epoch + light cfg provenance. MLX-free (caller converts mx->np)."""
    out: dict[str, np.ndarray] = {}
    for k, v in live_np.items():
        out[_RESUME_LIVE_PREFIX + k] = np.asarray(v, np.float32)
    for k, v in ema_np.items():
        out[_RESUME_EMA_PREFIX + k] = np.asarray(v, np.float32)
    has_opt = bool(opt_np)
    if has_opt:
        for k, v in opt_np.items():
            out[_RESUME_OPT_PREFIX + k] = np.asarray(v)
    out["__resume_epoch"] = np.asarray(int(epoch))
    out["__resume_has_opt"] = np.asarray(int(has_opt))
    out["__cfg_n_hidden"] = np.asarray(args.n_hidden)
    out["__cfg_hidden_dim"] = np.asarray(args.hidden_dim)
    out["__cfg_mod_dim"] = np.asarray(args.mod_dim)
    out["__cfg_self_orient"] = np.asarray(int(bool(args.self_orient)))
    out["__cfg_in_feat"] = np.asarray(int(in_feat))
    out["__cfg_w_pose"] = np.asarray(float(args.w_pose))
    return out


def _load_resume_state(npz_path: Path) -> dict[str, Any]:
    """Parse a resume sidecar OR (fallback) a plain EMA deploy npz. Returns live/ema/opt dicts +
    epoch + has_opt + cfg. NO-FAKE: a missing/garbage file raises. MLX-free."""
    z = np.load(Path(npz_path), allow_pickle=False)
    live: dict[str, np.ndarray] = {}
    ema: dict[str, np.ndarray] = {}
    opt: dict[str, np.ndarray] = {}
    cfg: dict[str, Any] = {}
    for k in z.files:
        if k.startswith(_RESUME_LIVE_PREFIX):
            live[k[len(_RESUME_LIVE_PREFIX):]] = np.asarray(z[k], np.float32)
        elif k.startswith(_RESUME_EMA_PREFIX):
            ema[k[len(_RESUME_EMA_PREFIX):]] = np.asarray(z[k], np.float32)
        elif k.startswith(_RESUME_OPT_PREFIX):
            opt[k[len(_RESUME_OPT_PREFIX):]] = np.asarray(z[k])
        elif k.startswith("__"):
            a = z[k]
            cfg[k] = a.item() if a.size == 1 else a.tolist()
        else:
            # plain EMA deploy npz: unprefixed keys are the EMA-shadow params. Use them as the
            # live-weight fallback (resume from the deploy checkpoint when no sidecar exists).
            live.setdefault(k, np.asarray(z[k], np.float32))
    epoch = int(cfg.get("__resume_epoch", cfg.get("__epoch", 0)))
    return {
        "live": live, "ema": ema, "opt": opt,
        "epoch": epoch, "has_opt": bool(int(cfg.get("__resume_has_opt", 0))), "cfg": cfg,
    }


def _resolve_resume_path(p: Path) -> Path:
    """Accept a run dir (prefer the resume sidecar, fall back to the EMA deploy npz) OR an explicit
    npz file. NO-FAKE: nonexistent -> FileNotFoundError (never fabricate a resume)."""
    p = Path(p)
    if p.is_dir():
        for name in ("levelset_resume_state.npz", "levelset_witness_ema_mlx.npz"):
            cand = p / name
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"--resume-from dir {p} has neither levelset_resume_state.npz nor "
            "levelset_witness_ema_mlx.npz (nothing to resume from).")
    if p.exists():
        return p
    raise FileNotFoundError(f"--resume-from path {p} does not exist (NO-FAKE: refusing to fabricate).")


_STAGE_TAGS = {"ce": "stageCE", "tau_softplus": "stageTau", "l7_softplus": "stageL7", "margin_hinge": "stageHinge"}


def _stage_tag(seg_form: str) -> str:
    """Filename-safe stage tag for the PRESERVED per-stage checkpoint (PR95 curriculum stages)."""
    return _STAGE_TAGS.get(str(seg_form), f"stage_{seg_form}")


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
    palette_init_logit: np.ndarray | None = None,
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
            # (DIAGNOSED FIX) learned per-class palette (K,3), in LOGIT space (sigmoid(palette)*255
            # = the class color). DEFAULT: anchor to the NATURAL per-class mean GT RGB (logit) —
            # the transfer probe hit realized d_seg 0.0049 with this palette; a generic luma-ramp
            # init left SegNet unable to separate classes (witness plateaued ~0.51). The palette
            # stays LEARNABLE (it can move off the anchor) but STARTS in SegNet's distribution.
            if palette_init_logit is not None:
                pal = np.asarray(palette_init_logit, np.float32).reshape(n_classes, 3)
            else:
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
def validate_lane_edge_config(
    *, lane_edge_weight: float, lane_edge_start_epoch: int, epochs: int,
    lane_edge_class: int, n_classes: int = 5,
) -> None:
    """(FEED-df R2) LEVER-3 fail-closed config guard (pure; testable; fail LOUD not silent).

    A lane lever that never engages (start_epoch > epochs) is a silent no-op = a FALSE 'lane-edge
    does not help' verdict; an out-of-range class index would IndexError mid-training (after GPU
    spend). When the lever is OFF (weight<=0) the guard is a NO-OP so the additive default path is
    never gated by a lever that is not in use.
    """
    if lane_edge_weight <= 0.0:
        return
    if lane_edge_start_epoch > epochs:
        raise ValueError(
            f"--lane-edge-weight {lane_edge_weight} > 0 but --lane-edge-start-epoch "
            f"({lane_edge_start_epoch}) > --epochs ({epochs}): the lane hinge would NEVER engage "
            "-> a silent no-op = a FALSE 'lane-edge does not help' verdict. Set "
            "--lane-edge-start-epoch <= --epochs (0 = engage from ep1)."
        )
    if not (0 <= lane_edge_class <= n_classes - 1):
        raise ValueError(
            f"--lane-edge-class ({lane_edge_class}) out of range [0,{n_classes - 1}] for the "
            f"{n_classes}-class comma10k partition [Road0,Lane1,MyCar2,Undrivable3,Movable4]; would "
            "IndexError mid-training. Use 1 for the lane orbit (the d_seg gate)."
        )


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
    # LEVER-2 (stem-Nyquist) cap (default None = no cap = current behavior). Drops curvelet atoms
    # above the SegNet-stem Nyquist (free byte/alias budget; see stem_nyquist_max_freq_*).
    B = curvelet_directional_B(bank, max_freq=args.max_bank_freq)
    curv_feats_np = curvelet_feats(coords_np, B).astype(np.float32)  # (P, 2*cols)
    in_feat = curv_feats_np.shape[1]
    # SELF-ORIENTATION directional augmentation (byte-closeable; tangent from the witness's OWN
    # argmax, cos 0.89-0.91 vs GT). Recomputed every --reorient-every epochs from the live SDF
    # argmax; concatenated to the curvelet feats. OFF by default (the from-scratch smoke uses
    # curvelet only — self-orientation is a finetune lever needing a roughly-learned partition).
    # SELF-ORIENT (#1 follow-up, WIRED): the byte-closeable -48% directional lever. The tangent is
    # computed from the decoder's OWN cheap-forward argmax (self-orientation FIXED POINT: start with
    # zero-directional = curvelet-only iso pass -> argmax -> tangent -> directional feats -> converge),
    # so it is reconstructible at decode with NO GT leak (cos 0.89-0.91 vs GT). PER-PAIR feats are
    # concatenated to the shared curvelet feats and threaded through train+verdict (ONE codepath).
    use_self_orient = bool(args.self_orient)
    n_dir_freqs = int(args.n_dir_freqs)
    dir_w = 4 * n_dir_freqs
    if use_self_orient:
        in_feat += dir_w
    # per-pair directional feats (zeros until the first reorient -> ep<reorient = pure curvelet).
    dir_feats_per_pair = [np.zeros((curv_feats_np.shape[0], dir_w), np.float32) for _ in range(P)] if use_self_orient else None

    def _feats_np_for_pair(pi: int) -> np.ndarray:
        if not use_self_orient:
            return curv_feats_np
        return np.concatenate([curv_feats_np, dir_feats_per_pair[pi]], axis=-1).astype(np.float32)

    print(json.dumps({"stage": "front_end", "curvelet_cols": int(B.shape[1]), "dir_w": int(dir_w),
                      "in_feat": int(in_feat), "self_orient": use_self_orient,
                      "front_end": ("curvelet+self_orient" if use_self_orient else "generic-curvelet only")}), flush=True)

    # (DEVICE BUG FIX) the adapter LOADS the upstream torch scorers then converts to MLX — the
    # torch .device() must be "cpu" (torch has no "gpu"; args.mlx_device="gpu" crashed here in 3.4s).
    # The MLX render runs on mx.gpu via temporary_mlx_device(args.mlx_device) below; the torch
    # scorer/R/verdict are CPU authority. The device SPLIT: MLX "gpu" -> render; torch -> "cpu".
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
    coord_feats_mx = mx.array(curv_feats_np)

    # (DIAGNOSED FIX) natural per-class palette = mean GT RGB per L* class (the transfer-probe's
    # winning ingredient; logit space). Anchors the learned palette inside SegNet's distribution so
    # the partition is READABLE from epoch 0 (a luma-ramp init plateaued ~0.51). NO GT leak at
    # deploy: the palette is a LEARNED weight counted in the archive (it is a fixed (5,3) tensor,
    # not the per-frame GT). --no-palette-anchor restores the generic ramp (ablation).
    palette_init = None
    if args.palette_anchor:
        import torch
        import torch.nn.functional as F
        sums = np.zeros((5, 3), np.float64); cnts = np.zeros(5, np.float64)
        for pi in range(min(P, 64)):
            f1 = torch.from_numpy(np.asarray(gt.gt_f1[pi], np.float32)).permute(2, 0, 1)[None]
            lr = np.asarray(gt.lstars[pi]); hh, ww = lr.shape
            small = F.interpolate(f1, size=(hh, ww), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
            for k in range(5):
                msk = lr == k
                if msk.any():
                    sums[k] += small[msk].sum(0); cnts[k] += int(msk.sum())
        mean = np.where(cnts[:, None] > 0, sums / np.maximum(cnts[:, None], 1), 127.0)
        palette_init = np.log(np.clip(mean / 255.0, 1e-3, 1 - 1e-3) / (1 - np.clip(mean / 255.0, 1e-3, 1 - 1e-3))).astype(np.float32)
        print(json.dumps({"stage": "palette_anchor", "mean_rgb": mean.round(1).tolist()}), flush=True)

    model = build_levelset_rgb_witness(
        num_pairs=P, in_feat=in_feat, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
        mod_dim=args.mod_dim, n_classes=5, activation=args.activation, softmax_temp=args.softmax_temp_start,
        wire_w0=args.wire_w0, wire_s0=args.wire_s0, hosc_beta=args.hosc_beta, hosc_omega=args.hosc_omega,
        chroma=args.chroma, palette_init_logit=palette_init,
    )
    mx.eval(model.parameters())
    # SIREN init (Sitzmann 2020) for the periodic family (hosc/wire) — the canonical from-scratch
    # trainability fix (parent: hosc-without-SIREN-init was d_seg 0.689). Reuses the parent's
    # apply_siren_init on in_proj (first) + hidden (subsequent); out_sdf/out_tex/palette/film keep
    # default init (FiLM must stay nonzero or the code-gradient dies).
    if args.activation in {"hosc", "wire"} and args.siren_init:
        from train_witness_realized_through_R_mlx import apply_siren_init
        omega_init = args.hosc_omega if args.activation == "hosc" else args.wire_w0
        apply_siren_init(model, omega=omega_init)
        mx.eval(model.parameters())
    # STRUCTURED-PRIOR phi INIT (FEED-ef, ADDITIVE, default-off). PRETRAIN phi so argmax(phi) ~= the
    # validated self-detected static-core partition (hood+sky+road[+lane] deep SDFs; FEED-dm/du/dw/dx).
    # The one-shot linear-readout init is broken (the random INR trunk's linear span ~= majority class,
    # disagree ~0.51 across hosc/relu/wire); the trunk must be ADAPTED, so this is a short subsampled
    # Adam pretrain of model.sdf -> the clipped structured SDF target (the network has the capacity:
    # trained mod-32 reaches d_seg 0.00124; pretrain reaches direct disagree ~0.025 in ~600 steps).
    # The static-core is generic same-rig camera geometry (rule-118 FREE; train-time init ships 0 bytes
    # -- the archive ships the TRAINED weights). Built on the cached L* (frozen CPU-torch argmax). EMA
    # is created AFTER so the shadow starts at the structured init. Default OFF => skipped => byte-identical.
    # MEASURED CAVEAT (n24 realized-through-R): NO epoch-0 realized win (the render is texture-dominated
    # at init -> SegNet reads random out_tex, not the partition; structured realized 0.586 ~ random 0.506).
    # Value is a training-trajectory A/B only (UNPROVEN). hosc/SIREN-init-fragile -> loud WARN if it stalls.
    if args.structured_init:
        from tac.boundary_math.lever_b_levelset_generator import build_static_core_phi_target
        lstar_shape = tuple(np.asarray(gt.lstars[0]).shape)
        if (render_h, render_w) != lstar_shape:
            raise ValueError(
                f"--structured-init requires --render-h/--render-w == the L* res {lstar_shape} "
                f"(got {(render_h, render_w)}); the static-core masks are built on the cached L*."
            )
        lst_stack_si = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)], axis=0)
        phi_tgt_hwk, sc_roles, sc_meta = build_static_core_phi_target(
            lst_stack_si, n_classes=5, include_lane=args.structured_init_include_lane,
            static_thresh=args.structured_init_thresh,
        )
        sc_part = phi_tgt_hwk.argmax(-1).reshape(-1)
        sc_feats_np = _feats_np_for_pair(0)  # pair-0 feats (curvelet[+zeros]); all codes 0 at init -> SHARED
        sc_clip = float(args.structured_init_sdf_clip)
        sc_tgt_np = np.clip(phi_tgt_hwk.reshape(render_h * render_w, 5), -sc_clip, sc_clip).astype(np.float32)
        sc_ns = min(int(args.structured_init_subsample), sc_feats_np.shape[0])
        sc_rng = np.random.default_rng(args.seed)

        def _structured_init_loss(m, fb, tb):
            return mx.mean((m.sdf(fb, 0) - tb) ** 2)

        sc_vg = nn.value_and_grad(model, _structured_init_loss)
        sc_opt = optim.AdamW(learning_rate=float(args.structured_init_lr))
        for _s in range(int(args.structured_init_steps)):
            sc_idx = sc_rng.integers(0, sc_feats_np.shape[0], sc_ns)
            _sL, _sg = sc_vg(model, mx.array(sc_feats_np[sc_idx]), mx.array(sc_tgt_np[sc_idx]))
            # FREEZE the per-frame code embedding: pretrain the SHARED trunk (code=0) so EVERY frame
            # (all codes 0 at init) starts at the structured partition, not just frame 0. Without this
            # the loss on sdf(.,0) also adapts code[0] -> only frame 0 is structured (MEASURED: a
            # code=0 frame disagrees 0.67 vs 0.011 frozen). Keeps the init a true SHARED prior.
            if "code" in _sg:
                _sg["code"] = mx.zeros_like(_sg["code"])
            sc_opt.update(model, _sg)
            mx.eval(model.parameters())
        sc_phi = np.asarray(model.sdf(mx.array(sc_feats_np), 0))
        sc_disagree = float(np.count_nonzero(sc_phi.argmax(-1) != sc_part)) / sc_part.size
        mx.eval(model.parameters())
        print(json.dumps({"stage": "structured_init", "roles": sc_roles.as_dict(),
                          "pretrain_direct_argmax_disagree_vs_part": round(sc_disagree, 5),
                          "steps": int(args.structured_init_steps), "lr": float(args.structured_init_lr),
                          **{k: v for k, v in sc_meta.items() if k != "roles"}}), flush=True)
        if sc_disagree > 0.30:
            print(json.dumps({"stage": "structured_init_WARN",
                              "msg": "pretrain did NOT structure the partition (disagree>0.30); init ~ random "
                              "(hosc/SIREN trainability fragility). Try --structured-init-lr/-steps or another --seed.",
                              "disagree": round(sc_disagree, 5)}), flush=True)
    ema = MlxEMA(model, decay=args.ema_decay)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    base_loss = make_loss_fn(
        adapter, render_h, render_w, score_domain=args.score_domain_loss, pose_eps=args.pose_eps,
        seg_loss=args.seg_loss, tau_softplus_tau=args.tau_softplus_tau, l7_mult=args.l7_mult,
        l7_threshold=args.l7_threshold,
    )

    # LEVER-3 (lane-edge fragility weighting) hyperparameters captured from args (static; closure
    # constants, NOT value_and_grad args -> ZERO change to the call site). lane_edge_weight=0.0
    # (default) => the branch below is skipped => behavior IDENTICAL to before (fully additive).
    lane_w = float(args.lane_edge_weight)
    lane_cls = int(args.lane_edge_class)
    lane_tgt = float(args.lane_margin_target)
    lane_start = int(args.lane_edge_start_epoch)
    # OPTIMAL-FORM (recursive review, FEED-df): the lane margin hinge is a margin-SHARPENING loss;
    # running it from ep0 during the COARSE ce stage risks the known margin-from-scratch-starves-
    # interior failure (the partition isn't formed yet). ``lane_gate`` is a python bool RE-READ
    # inside total_loss_fn each value_and_grad call (so the lane branch is included/excluded per
    # epoch); the epoch loop sets it = (ep >= lane_start). Default lane_start=0 => engaged from ep1
    # = IDENTICAL to before (fully additive). When lane_start>1 the engagement epoch RE-TREATS the
    # spike-guard (clears recent_losses) so the loss jump from the added term is NOT silently
    # spike-skipped (operator 2026-06-26 "different stages need different treatment ... transitions
    # must re-treat"; margin-engage spike-skip is the named failure this prevents).
    lane_gate = {"on": lane_start <= 1}

    def total_loss_fn(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w):
        L = base_loss(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form=seg_form)
        phi0 = model.sdf(cf, c0)
        eik, length, _ = _eikonal_length_mlx(phi0, render_h, render_w)
        L = L + eik_w * eik + len_w * length
        # LEVER-3 (lane-edge fragility weighting, operator 2026-06-27 Yousfi-grounding): contest
        # SegNet argmax order (PIL-luma sort of class_values [42,76,90,124,161]; FEED-da):
        # [Road0, Lane1, MyCar2, Undrivable3, Movable4]. Class0=Road & Class1=Lane are CONFIRMED
        # (all memos agree; this lever uses ONLY class 1; 2/3/4 labels disputed vs frozen_source but
        # not load-bearing here). Lane (class 1) is thin
        # all-boundary double-edges (19% of d_seg flips) and UNDER-FIT because the CE baseline has NO
        # class weighting. This ADDITIVE term up-weights the REALIZED (through-R SegNet) margin hinge
        # at GT-lane pixels: it renders f1 -> R -> frozen SegNet logits, takes the live decision
        # margin (gt_logit - top_competitor) ONLY where GT==lane, and penalizes relu(target-margin)
        # there. The hinge fires exactly on SMALL-MARGIN (fragile = boundary) lane pixels, so it
        # adds gradient pressure to widen the lane margin at the lane double-edges. Default-off
        # (lane_w=0). When ON it costs a SECOND realized seg forward (acceptable per operator
        # "score > training time"; the optimal-form fusion into the base seg loss needs a parent
        # edit, out of scope for this additive prep).
        if lane_w > 0.0 and lane_gate["on"]:
            f1 = render_through_R_mlx(model, cf, c1, render_h, render_w)  # (1, SEG_H, SEG_W, 3)
            seg_logits = adapter.segnet(f1)                              # (1, H, W, 5)
            gt_logit = mx.sum(seg_logits * lstar_oh, axis=-1)           # (1, H, W)
            runner_up = mx.max(seg_logits + lstar_oh * (-1e9), axis=-1)  # (1, H, W) max competitor
            signed = gt_logit - runner_up                              # (1, H, W) decision margin
            lane_mask = lstar_oh[..., lane_cls]                         # (1, H, W) 1.0 where GT==lane
            hinge_map = mx.maximum(lane_tgt - signed, 0.0) * lane_mask  # fragile lane pixels only
            lane_term = mx.sum(hinge_map) / (mx.sum(lane_mask) + 1e-6)  # mean hinge over lane px
            L = L + lane_w * lane_term
        return L

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

    def _fwd_numpy(deploy: dict[str, np.ndarray], feats_np: np.ndarray, code_row: np.ndarray):
        return levelset_rgb_forward_numpy(
            deploy, feats_np, code_row, n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
            n_classes=5, activation=args.activation, softmax_temp=float(model.softmax_temp),
            wire_w0=args.wire_w0, wire_s0=args.wire_s0, hosc_beta=args.hosc_beta, hosc_omega=args.hosc_omega,
            chroma=args.chroma,
        )

    def _render_numpy_deploy(deploy: dict[str, np.ndarray], pi: int, fk: int) -> np.ndarray:
        """THE ONE CODEPATH (fp32 numpy, deploy-faithful) — same forward the byte-close/inflate use.
        Uses the PER-PAIR feats (curvelet [+ self-orient]) so the verdict == the deploy render."""
        rgb, _phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + fk])
        return _torch_R_to_camera_uint8(rgb.reshape(render_h, render_w, 3))

    def recompute_self_orient(deploy: dict[str, np.ndarray]) -> float:
        """Self-orientation FIXED-POINT step: from the EMA deploy frame1 argmax (current feats),
        recompute each pair's directional feats. Returns the mean |dir feat| (non-triviality check)."""
        if not use_self_orient:
            return 0.0
        mag = 0.0
        for pi in range(P):
            _rgb, phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + 1])
            argmax = phi.argmax(-1).reshape(render_h, render_w).astype(np.int64)
            df = self_orientation_directional_feats(
                coords_np, argmax, n_freqs=n_dir_freqs, freq_across=args.freq_across, freq_along=args.freq_along)
            dir_feats_per_pair[pi] = df.astype(np.float32)
            mag += float(np.abs(df).mean())
        return mag / max(P, 1)

    def realized_verdict() -> dict[str, float]:
        # (fix a+b+c) verdict the EMA SHADOW, int8-DEQUANTIZED, via the fp32 numpy ONE CODEPATH
        # (NOT the MLX-GPU reduced-precision forward — the 4th artifact). This IS the deploy render.
        ema_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
        deploy = int8_dequant_params(ema_np)
        f0s, f1s = [], []
        for pi in vpairs:
            f0s.append(_render_numpy_deploy(deploy, pi, 0))
            f1s.append(_render_numpy_deploy(deploy, pi, 1))
        ds = cpu_verdict_d_seg_batch(seg_cpu, f1s, [gt.lstars[pi] for pi in vpairs])
        # pose VERDICT still measured (monitoring) but pose is NOT the witness's job (w_pose=0
        # default; deploy pose rides the SOLVED Quantizr stored-pose sidecar, d_pose 3.4e-5).
        dp = cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, [gt.gt_poses[pi] for pi in vpairs])
        return {"d_seg": float(np.mean(ds)), "d_pose": float(np.mean(dp))}

    history: list[dict[str, Any]] = []

    # per-pair MLX coord-feats cache: shared curvelet tensor when no self-orient; rebuilt on each
    # reorient when self-orient is on (so the train forward uses the SAME per-pair feats the
    # numpy verdict/deploy uses -> ONE codepath).
    cf_mx_cache: list[Any] | None = None

    def _rebuild_cf_mx_cache() -> None:
        # MEMORY-BOUNDED in-place rebuild (FEED-eh): free each OLD per-pair MLX feats entry BEFORE
        # allocating the new one (the naive list-comprehension held old+new => 2x ~41GB at n600 =>
        # OOM at the ep50 reorient). Peak now ~= ONE cache; BIT-IDENTICAL values.
        nonlocal cf_mx_cache
        cf_mx_cache = rebuild_per_pair_feats_in_place(
            cf_mx_cache, P, _feats_np_for_pair, mx_array=mx.array, mx_eval=mx.eval)

    def _cf_mx(pi: int):
        return coord_feats_mx if not use_self_orient else cf_mx_cache[pi]

    if use_self_orient:
        _rebuild_cf_mx_cache()  # ep<reorient: dir feats are zeros -> pure curvelet iso pass

    # ---- CHECKPOINT closures (FEED-dz; mx->np snapshot + atomic save of the deploy EMA npz + the
    # resume sidecar). The deploy npz keeps the canonical name so the byte-close tool consumes it
    # as-is; the resume sidecar is separate so the deploy npz stays byte-close-clean. ----
    def _snapshot_numpy_state() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        shadow_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
        live_np = {k: np.asarray(v, np.float32) for k, v in tree_flatten(model.parameters())}
        opt_np: dict[str, np.ndarray] = {}
        try:  # best-effort: optimizer moments accelerate resume but a fresh AdamW re-warms in steps.
            for k, v in tree_flatten(opt.state):
                arr = np.asarray(v)
                if arr.dtype.kind in "fiub":
                    opt_np[k] = arr
        except Exception:
            opt_np = {}
        return shadow_np, live_np, opt_np

    def _do_checkpoint(epoch: int, *, stage_tag: str | None = None) -> dict[str, Any]:
        shadow_np, live_np, opt_np = _snapshot_numpy_state()
        ema_arrays = _build_ema_checkpoint_arrays(
            shadow_np, args=args, softmax_temp=float(model.softmax_temp),
            render_h=render_h, render_w=render_w, epoch=epoch, in_feat=in_feat)
        resume_arrays = _build_resume_state_arrays(
            live_np, shadow_np, opt_np, args=args, epoch=epoch, in_feat=in_feat)
        # rolling latest: the byte-close default name + the quick resume target (overwritten atomically).
        _atomic_savez(out_dir / "levelset_witness_ema_mlx.npz", ema_arrays)
        _atomic_savez(out_dir / "levelset_resume_state.npz", resume_arrays)
        written: dict[str, Any] = {
            "epoch": epoch, "ema_latest": "levelset_witness_ema_mlx.npz",
            "resume_latest": "levelset_resume_state.npz", "has_opt": bool(opt_np)}
        if stage_tag is not None:  # PRESERVED stage-encoded ckpt (NOT overwritten -> per-stage A/B).
            ema_pres = f"levelset_ckpt_{stage_tag}_ep{epoch}.npz"
            res_pres = f"levelset_resume_{stage_tag}_ep{epoch}.npz"
            _atomic_savez(out_dir / ema_pres, ema_arrays)
            _atomic_savez(out_dir / res_pres, resume_arrays)
            written["ema_preserved"] = ema_pres
            written["resume_preserved"] = res_pres
        return written

    # ---- RESUME restore (FEED-dz; --resume-from None => fresh start => behavior UNCHANGED). Loads
    # decoder + per-pair codes (live) + EMA shadow + optimizer (best-effort) + the epoch position;
    # self-orient dir feats are regenerated from the restored EMA argmax (not stored -> no GB bloat).
    start_epoch = 1
    if args.resume_from:
        from mlx.utils import tree_unflatten
        rp = _resolve_resume_path(Path(args.resume_from))
        rs = _load_resume_state(rp)
        if not rs["live"]:
            raise ValueError(f"--resume-from {rp} has no live/param tensors (NO-FAKE: cannot resume).")
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in rs["live"].items()]))
        mx.eval(model.parameters())
        ema_src = rs["ema"] if rs["ema"] else rs["live"]
        for k in list(ema.shadow.keys()):
            if k in ema_src:
                ema.shadow[k] = mx.array(ema_src[k])
        mx.eval(list(ema.shadow.values()))
        start_epoch = int(rs["epoch"]) + 1
        restored_opt = False
        if rs["has_opt"] and rs["opt"]:
            try:
                opt.init(model.trainable_parameters())
                flat_state = dict(tree_flatten(opt.state))
                for k in list(flat_state.keys()):
                    if k in rs["opt"]:
                        flat_state[k] = mx.array(rs["opt"][k])
                opt.state = tree_unflatten(list(flat_state.items()))
                mx.eval(opt.state)
                restored_opt = True
            except Exception as e:  # best-effort: a fresh AdamW re-warms its moments in a few steps.
                print(json.dumps({"stage": "resume_opt_warn",
                                  "note": f"optimizer-state restore failed ({type(e).__name__}: {e}); "
                                  "continuing with fresh AdamW moments (best-effort)"}), flush=True)
        if use_self_orient:
            ema_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
            mag = recompute_self_orient(int8_dequant_params(ema_np))
            _rebuild_cf_mx_cache()
            print(json.dumps({"stage": "resume_reorient", "mean_abs_dir_feat": round(mag, 5)}), flush=True)
        print(json.dumps({"stage": "resume", "from": str(rp), "resumed_epoch": int(rs["epoch"]),
                          "start_epoch": start_epoch, "restored_opt": restored_opt}), flush=True)

    # baseline verdict (epoch 0, or the resumed epoch) -- reflects any restored weights.
    v0 = realized_verdict()
    blob = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()})
    s0 = implied_score_from_verdict(v0["d_seg"], v0["d_pose"], blob["total_quantized_blob_bytes"])
    print(json.dumps({"stage": "verdict", "epoch": start_epoch - 1, **{k: round(v, 6) for k, v in v0.items()},
                      "blob_bytes": blob["total_quantized_blob_bytes"], "implied_S": round(s0, 4),
                      "axis": "[macOS-CPU advisory] NON-PROMOTABLE"}), flush=True)
    history.append({"epoch": start_epoch - 1, **v0, "implied_S": s0})

    if lane_w > 0.0:
        print(json.dumps({"stage": "lane_edge", "active": True, "weight": lane_w, "lane_class": lane_cls,
                          "margin_target": lane_tgt, "start_epoch": lane_start,
                          "note": "additive realized lane-class margin hinge (2nd seg forward when "
                          "active; default-off; engages at ep>=start_epoch with spike-guard re-treat)"}), flush=True)
    if args.max_bank_freq is not None:
        from tac.boundary_math.lever_b_levelset_generator import stem_nyquist_max_freq_cycles_per_unit
        nyq = stem_nyquist_max_freq_cycles_per_unit(scorer_w=SEG_W)
        print(json.dumps({"stage": "stem_nyquist", "max_bank_freq": float(args.max_bank_freq),
                          "stem_nyquist_cycles_per_unit": nyq, "curvelet_cols_after_cap": int(B.shape[1])}), flush=True)

    recent_losses: list[float] = []
    last_ep = start_epoch - 1
    stage_ckpts: list[dict[str, Any]] = []
    with temporary_mlx_device(args.mlx_device):
        for ep in range(start_epoch, args.epochs + 1):
            seg_form = _seg_form_for_epoch(ep, args)
            # lane-edge engagement gate + transition RE-TREAT (spike-guard reset at the engage epoch
            # so the added margin-hinge term's loss jump is not silently spike-skipped; no-op when
            # lane_start<=1 i.e. the default always-on-from-ep1 path -> zero behavior change).
            if lane_w > 0.0:
                _was_on = lane_gate["on"]
                lane_gate["on"] = ep >= lane_start
                if lane_gate["on"] and not _was_on:
                    recent_losses.clear()
                    print(json.dumps({"stage": "lane_edge_engage", "epoch": ep, "lane_start": lane_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # SELF-ORIENT reorient cadence (fixed-point): recompute per-pair directional feats from
            # the EMA deploy argmax every --reorient-every epochs (skip ep1: argmax is random).
            if use_self_orient and ep > 1 and (ep - 1) % max(args.reorient_every, 1) == 0:
                ema_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
                mag = recompute_self_orient(int8_dequant_params(ema_np))
                _rebuild_cf_mx_cache()
                print(json.dumps({"stage": "reorient", "epoch": ep, "mean_abs_dir_feat": round(mag, 5)}), flush=True)
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
                        model, _cf_mx(pi), 2 * pi + 0, 2 * pi + 1, oh, mg, pose_tgts[pi],
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
            # ---- CHECKPOINTING (FEED-dz; mandatory per operator "never launch non-resumable / save
            # per-stage" rule). PER-STAGE: at every curriculum-stage TRANSITION save a PRESERVED,
            # stage-encoded, byte-close-loadable ckpt (per-stage A/B of which stage moves d_seg).
            # INTRA-STAGE: every --ckpt-every epochs save the rolling latest (crash-resume window).
            is_transition = (
                args.stage_checkpoints and ep < args.epochs
                and _seg_form_for_epoch(ep + 1, args) != seg_form)
            do_periodic = args.ckpt_every > 0 and ep % args.ckpt_every == 0
            if is_transition:
                w = _do_checkpoint(ep, stage_tag=_stage_tag(seg_form))
                stage_ckpts.append(w)
                print(json.dumps({"stage": "checkpoint", "kind": "stage_transition", **w}), flush=True)
            elif do_periodic:
                w = _do_checkpoint(ep)
                print(json.dumps({"stage": "checkpoint", "kind": "intra_stage", **w}), flush=True)
            last_ep = ep

    # FINAL checkpoint (replaces the historical loop-end-only save, which is now FORBIDDEN). Always
    # writes the rolling latest + a PRESERVED final stage-encoded ckpt -> the run is byte-closeable
    # and resumable from disk at completion. Saves the EMA SHADOW (deploy), NOT live (EMA rule).
    final_form = _seg_form_for_epoch(last_ep, args) if last_ep >= 1 else args.seg_loss
    final = _do_checkpoint(last_ep, stage_tag=(_stage_tag(final_form) if args.stage_checkpoints else None))
    stage_ckpts.append({**final, "kind": "final"})
    ck = out_dir / "levelset_witness_ema_mlx.npz"
    print(json.dumps({"stage": "checkpoint", "kind": "final", **final}), flush=True)
    result = {
        "utc": _utc(), "n_pairs": P, "epochs": args.epochs, "final_epoch": last_ep,
        "render_hw": [render_h, render_w],
        "front_end": "curvelet" + ("+self_orient" if use_self_orient else ""),
        "activation": args.activation, "in_feat": int(in_feat),
        "history": history, "checkpoint": str(ck), "stage_checkpoints": stage_ckpts,
        "resumable": True, "ckpt_every": int(args.ckpt_every),
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
    # RESUMABILITY + CHECKPOINTING (FEED-dz; additive). Per operator "never launch non-resumable /
    # save+preserve a checkpoint at the end of each stage": per-stage PRESERVED ckpts default ON;
    # --ckpt-every adds intra-stage rolling saves (crash window). --resume-from continues a run.
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save the rolling EMA+resume checkpoint every N epochs (0=off; per-stage + final "
                    "saves always happen). Set e.g. 100 to bound a crash/OOM to <=N epochs of loss "
                    "and enable early byte-close during a multi-day run.")
    ap.add_argument("--stage-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                    help="save a PRESERVED, stage-encoded, byte-close-loadable ckpt at every curriculum "
                    "stage transition + at the final epoch (default ON; --no-stage-checkpoints only for "
                    "throwaway smokes -- loop-end-only is forbidden for real rows).")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="resume a run from a checkpoint: a run DIR (prefers levelset_resume_state.npz, "
                    "falls back to levelset_witness_ema_mlx.npz) OR an explicit npz. Restores decoder + "
                    "per-pair codes + EMA shadow + optimizer (best-effort) + the epoch position.")
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
    ap.add_argument("--palette-anchor", action=argparse.BooleanOptionalAction, default=True,
                    help="(DIAGNOSED FIX) init learnable palette to natural per-class mean GT RGB (transfer-probe ingredient; "
                    "breaks the ~0.51 luma-ramp plateau). --no-palette-anchor = generic ramp ablation.")
    # FRONT-END
    ap.add_argument("--bank-n-scales", type=int, default=4)
    ap.add_argument("--bank-n-orient0", type=int, default=6)
    ap.add_argument("--bank-f0", type=float, default=2.0)
    ap.add_argument("--bank-base", type=float, default=2.0)
    ap.add_argument("--bank-n-iso", type=int, default=4)
    # LEVER-2 (stem-Nyquist rate/anti-alias): cap curvelet-bank freqs (cycles/unit) at the SegNet
    # stem Nyquist (default 64 for SEG_W=512, stem-stride-2). None (default) = no cap = current
    # behavior. The DEFAULT curvelet bank (max 16 cyc/unit) is already sub-Nyquist so this is a
    # no-op there; the over-Nyquist waste is in --n-dir-freqs/--freq-across (see the memo). Additive.
    ap.add_argument("--max-bank-freq", type=float, default=None,
                    help="LEVER-2: drop curvelet atoms above this freq (cycles/unit); None=no cap. "
                    "Stem Nyquist = SEG_W/(4*stem_stride) = 64 for the default 512/stride-2.")
    ap.add_argument("--self-orient", action=argparse.BooleanOptionalAction, default=False,
                    help="add byte-closeable self-orientation directional feats (finetune lever; needs a roughly-learned base).")
    ap.add_argument("--n-dir-freqs", type=int, default=6)
    ap.add_argument("--reorient-every", type=int, default=50)
    ap.add_argument("--freq-across", type=float, default=32.0, help="self-orient: HIGH freq across the edge (normal).")
    ap.add_argument("--freq-along", type=float, default=4.0, help="self-orient: LOW freq along the edge (tangent).")
    # ACTIVATION
    # (config-review #3) HOSC is the ONLY descent evidence (probe 0.0066; A/B 0.221 hosc vs 0.265
    # wire). WIRE was a paper-default guess; default HOSC, run wire as a sweep arm.
    ap.add_argument("--activation", choices=["wire", "hosc", "relu"], default="hosc")
    ap.add_argument("--wire-w0", type=float, default=20.0)
    ap.add_argument("--wire-s0", type=float, default=10.0)
    ap.add_argument("--hosc-beta", type=float, default=4.0)
    ap.add_argument("--hosc-omega", type=float, default=1.0)
    ap.add_argument("--siren-init", action=argparse.BooleanOptionalAction, default=True,
                    help="SIREN init (Sitzmann 2020) for hosc/wire periodic layers (from-scratch trainability fix).")
    # SEG LOSS / CURRICULUM
    ap.add_argument("--seg-loss", choices=["ce", "tau_softplus", "l7_softplus", "margin_hinge"], default="ce")
    ap.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tau-softplus-start-epoch", type=int, default=300)
    ap.add_argument("--l7-start-epoch", type=int, default=800)
    ap.add_argument("--tau-softplus-tau", type=float, default=0.3)
    ap.add_argument("--l7-mult", type=float, default=4.0)
    ap.add_argument("--l7-threshold", type=float, default=1.0)
    ap.add_argument("--margin-target-end", type=float, default=0.5)
    # LEVER-3 (lane-edge fragility weighting): up-weight class-1 (Lane) flips in the REALIZED margin
    # hinge. Lane is thin all-boundary double-edges (19% of d_seg flips) under-fit by the unweighted
    # CE baseline. Default 0.0 = OFF = current behavior (fully additive). When >0, costs a 2nd
    # realized seg forward (acceptable per operator "score > training time"). SegNet class order
    # (PIL-luma sort, class_values [42,76,90,124,161]): [Road0, Lane1, MyCar2, Undrivable3, Movable4];
    # class 0=Road & 1=Lane CONFIRMED (the lever uses only class 1; 2/3/4 disputed vs frozen_source).
    ap.add_argument("--lane-edge-weight", type=float, default=0.0,
                    help="LEVER-3: weight on the additive realized lane-class margin hinge (0=off).")
    ap.add_argument("--lane-edge-class", type=int, default=1,
                    help="LEVER-3: GT class index to up-weight (1=Lane, CONFIRMED; luma-sort order "
                    "[Road0,Lane1,MyCar2,Undrivable3,Movable4] for 2/3/4).")
    ap.add_argument("--lane-margin-target", type=float, default=0.5,
                    help="LEVER-3: target decision margin for the lane hinge relu(target - margin).")
    ap.add_argument("--lane-edge-start-epoch", type=int, default=0,
                    help="LEVER-3 OPTIMAL-FORM: engage the lane hinge only at ep>=this (0=from ep1=current "
                    "behavior). Gate to the tau_softplus/l7 margin stage (e.g. 300) to avoid the "
                    "margin-from-scratch-starves-interior failure; the engage epoch re-treats the spike-guard.")
    # LEVEL-SET REG
    ap.add_argument("--eikonal-weight", type=float, default=0.01, help="Eikonal |grad phi|->1 (topology bias, small).")
    ap.add_argument("--length-weight", type=float, default=0.001, help="Chan-Vese boundary-length (short smooth boundaries).")
    # STRUCTURED-PRIOR phi INIT (FEED-ef, ADDITIVE, DEFAULT-OFF). When ON, initialize out_sdf so
    # argmax(phi) ~= the VALIDATED self-detected static-core partition (hood+sky+road[+lane] deep SDFs;
    # FEED-dm/du/dw/dx) instead of random/SIREN -> the row STARTS at the ~0.006 structured floor and
    # LEARNS only the residual (lane wall + Movable). DEFAULT OFF = random/SIREN init = byte-identical
    # to the current row. The static-core is GENERIC same-rig camera geometry (rule-118 FREE); as a
    # TRAIN-TIME init it ships 0 bytes (the archive ships TRAINED weights). Requires render res == the
    # L* res (the static masks are built on the cached frozen CPU-torch L*).
    # MEASURED CAVEAT (FEED-ef, n24 realized-through-R): structuring phi gives NO epoch-0 realized
    # d_seg win — the render is texture-dominated at init (random out_tex), so SegNet reads texture
    # NOT the partition (structured-init realized 0.586 ~ random-init 0.506; even IDEAL flat-palette
    # is 0.125, never the 0.006 DIRECT/field-level floor). The structured prior is field-level only;
    # this flag's sole value is a TRAINING-TRAJECTORY A/B (does a correct partition init converge
    # faster?), UNPROVEN. The one-shot linear-readout init is broken (random trunk can't span the
    # partition, disagree ~0.51); this flag uses a short pretrain (adapts the trunk -> direct
    # disagree ~0.025) which is hosc/SIREN-init-FRAGILE (loud WARN if it stalls). Default OFF.
    ap.add_argument("--structured-init", action=argparse.BooleanOptionalAction, default=False,
                    help="FEED-ef: pretrain phi to the structured static-core partition (DEFAULT OFF=random/SIREN, byte-identical). "
                    "MEASURED: no epoch-0 realized win (texture-gated) -> trajectory A/B only.")
    ap.add_argument("--structured-init-include-lane", action=argparse.BooleanOptionalAction, default=True,
                    help="FEED-ef: include a SHARED static lane band in the structured init (lane is also learned per-frame).")
    ap.add_argument("--structured-init-thresh", type=float, default=0.5,
                    help="FEED-ef: majority-vote threshold for the static-core region masks.")
    ap.add_argument("--structured-init-steps", type=int, default=600,
                    help="FEED-ef: subsampled Adam steps to pretrain phi -> structured target.")
    ap.add_argument("--structured-init-lr", type=float, default=5e-3,
                    help="FEED-ef: LR for the structured-init pretrain (5e-3 converges; 8e-3 stalls).")
    ap.add_argument("--structured-init-subsample", type=int, default=8192,
                    help="FEED-ef: pixels/step for the structured-init pretrain (full-grid is CPU-slow).")
    ap.add_argument("--structured-init-sdf-clip", type=float, default=20.0,
                    help="FEED-ef: clip the SDF target to +/-this (argmax-preserving, well-conditioned).")
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

    # (FEED-df R2) LEVER-3 fail-closed config guard (pure helper; fails LOUD before any GPU spend).
    validate_lane_edge_config(
        lane_edge_weight=args.lane_edge_weight, lane_edge_start_epoch=args.lane_edge_start_epoch,
        epochs=args.epochs, lane_edge_class=args.lane_edge_class, n_classes=5,
    )

    result = run_train(args)
    print("\n=== LEVEL-SET WITNESS RESULT (realized through R) ===")
    print(json.dumps({"front_end": result["front_end"], "history": result["history"],
                      "axis": result["axis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
