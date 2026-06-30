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
import threading
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
    film_modulation_participation_ratio,
    film_rank_floor_penalty,
    int8_dequant_params,
    lane_thin_weight_map,
    levelset_rgb_forward_numpy,
    quantize_levelset_blob,
    rebuild_per_pair_feats_in_place,
    save_levelset_npz,
)
from tac.optimization.muon_finisher_mlx import (  # noqa: E402
    build_muon_finisher_optimizer,
    count_muon_adamw_split,
)
from tac.optimization.md_decoupling import (  # noqa: E402
    stiefel_project_columns,
    stiefel_residual,
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


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> Path:
    """Atomic JSON write (tmp + os.replace) per the durable-state discipline. Refuses /tmp.

    Used for the tiny best-checkpoint POINTER (``levelset_best.json``) so a harvester / early-stop
    reads the run's best realized-d_seg artifact WITHOUT re-deriving it from the log."""
    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def _is_new_best(d_seg: float, prev_best: float) -> bool:
    """NEW-best promotion rule (NO-FAKE): a FINITE, STRICTLY-better realized d_seg only. NaN/inf
    never win; a tie keeps the EARLIER best (reproducible). The 1e-12 guard avoids float-noise
    churn rewriting the best ckpt for sub-ULP "improvements". Module-level + pure -> unit-tested."""
    return bool(np.isfinite(d_seg)) and (float(d_seg) < float(prev_best) - 1e-12)


def _build_ema_checkpoint_arrays(
    shadow_np: dict[str, np.ndarray], *, args: Any, softmax_temp: float,
    render_h: int, render_w: int, epoch: int, in_feat: int,
    hosc_beta: float | None = None,
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
    # (FEED-fb) persist the CURRENT (possibly annealed) beta so the byte-close/inflate deploy forward
    # uses the SAME activation sharpness the EMA shadow was trained at (NO-FAKE). When the caller does
    # not thread it (hosc_beta is None) OR anneal is off, this == args.hosc_beta => byte-identical cfg.
    flat["__cfg_hosc_beta"] = np.asarray(args.hosc_beta if hosc_beta is None else float(hosc_beta))
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
    # (review R2a-MED-1) ARCH flags that change the param KEYS / training geometry: persist them in the
    # resume sidecar so a crash-resume from the ckpt dir ALONE can fail-closed if the resume command
    # omits the flag the run was trained with (the silent-param-drop risk -- MLX model.update only
    # touches EXISTING params, so a model rebuilt without film_pl/concat_pl would silently DROP the
    # trained per-layer FiLM params). film_per_layer/film_concat_code add params (film_pl./concat_pl.);
    # film_stiefel constrains the existing film.weight (training-dynamics, no new keys). The resume
    # sidecar is NOT byte-closed -> these provenance scalars cost ZERO archive bytes. Per the
    # resumability + deterministic-reproducibility non-negotiables.
    out["__cfg_film_per_layer"] = np.asarray(int(bool(getattr(args, "film_per_layer", False))))
    out["__cfg_film_concat_code"] = np.asarray(int(bool(getattr(args, "film_concat_code", False))))
    out["__cfg_film_stiefel"] = np.asarray(int(bool(getattr(args, "film_stiefel", False))))
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


def _load_decoder_params(npz_path: Path) -> dict[str, np.ndarray]:
    """Load ONLY the SHARED-DECODER params from a level-set EMA/deploy npz (FEED-eo amortization).

    Returns the decoder tensors (in_proj/film/hidden.*/out_sdf/out_tex {weight,bias} + palette) but
    EXCLUDES ``code`` (the per-(pair,frame) latents, which the freeze-decoder-fit-codes mode RE-FITS
    for a different pair count) and the free deterministic bank ``B``/``*_B`` (rule 118) and the
    ``__``-prefixed cfg scalars. NO-FAKE: a missing/garbage file raises. MLX-free."""
    z = np.load(Path(npz_path), allow_pickle=False)
    dec: dict[str, np.ndarray] = {}
    for k in z.files:
        if k.startswith("__"):
            continue
        if k == "code" or k.endswith("code"):
            continue
        if k == "B" or k.endswith("_B"):
            continue
        dec[k] = np.asarray(z[k], np.float32)
    if "in_proj.weight" not in dec:
        raise ValueError(
            f"--freeze-decoder-fit-codes {npz_path} has no 'in_proj.weight' (not a level-set witness "
            "decoder npz?); NO-FAKE: refusing to fit codes against a non-decoder file.")
    return dec


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
    film_per_layer: bool = False,
    film_concat_code: bool = False,
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
            # LEVER-A (FiLM-rank-fix) toggles (default OFF => the extra submodules are NOT created =>
            # model.parameters() / EMA / checkpoints / byte-close are BYTE-IDENTICAL to the pre-LEVER-A
            # witness, and the forward branches below are skipped).
            self.film_per_layer = bool(film_per_layer)
            self.film_concat_code = bool(film_concat_code)
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
            # LEVER-A1 (--film-per-layer): SEPARATE per-layer RESIDUAL FiLM projections, IDENTITY at
            # init (zero weight+bias => the residual scale (+0) and shift (+0) are 0 => the modulation
            # at init == the shared-FiLM-only forward; with the flag ON the per-layer route then learns
            # INDEPENDENT per-pair (scale,shift) modulation, raising the per-pair modulation rank to
            # attack the MEASURED participation-ratio collapse 3.34@CE -> 1.19@l7). siren_init touches
            # ONLY in_proj+hidden, so these stay zero at init.
            if self.film_per_layer:
                self.film_pl = [nn.Linear(mod_dim, 2 * hidden_dim) for _ in range(n_hidden)]
                for _lin in self.film_pl:
                    _lin.weight = mx.zeros_like(_lin.weight)
                    _lin.bias = mx.zeros_like(_lin.bias)
            # LEVER-A2 (--film-concat-code): an ADDITIVE per-pair code-injection route added to each
            # hidden pre-activation. This is the algebraically-FOLDED concat: concat([h, code]) @ W
            # == h @ W_h + code @ W_c, folded into ONE zero-init projection mod_dim->hidden_dim
            # (concat_pl[li]) -- a NON-collapsing per-pair TRANSLATION route alongside the
            # multiplicative FiLM (what a moving lane needs). Zero init => no-op at init
            # (identity-residual); shape-safe (no existing layer dims change).
            if self.film_concat_code:
                self.concat_pl = [nn.Linear(mod_dim, hidden_dim) for _ in range(n_hidden)]
                for _lin in self.concat_pl:
                    _lin.weight = mx.zeros_like(_lin.weight)
                    _lin.bias = mx.zeros_like(_lin.bias)
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
            code = self.code[code_idx]
            film = mx.reshape(self.film(code), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                # DEFAULT-OFF => scale==(1.0+film[li,0]), shift==film[li,1], no concat =>
                # pre == layer(h)*(1.0+film[li,0])+film[li,1] => BYTE-IDENTICAL to pre-LEVER-A.
                scale = 1.0 + film[li, 0]
                shift = film[li, 1]
                if self.film_per_layer:
                    pl = mx.reshape(self.film_pl[li](code), (2, self.hidden_dim))
                    scale = scale + pl[0]
                    shift = shift + pl[1]
                pre = layer(h) * scale + shift
                if self.film_concat_code:
                    pre = pre + self.concat_pl[li](code)
                h = self._act(pre)
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
                # DEFAULT-OFF => BYTE-IDENTICAL to the pre-LEVER-A batched forward (same expression).
                scale = 1.0 + film[:, li, 0][:, None, :]
                shift = film[:, li, 1][:, None, :]
                if self.film_per_layer:
                    pl = mx.reshape(self.film_pl[li](codes), (-1, 2, self.hidden_dim))
                    scale = scale + pl[:, 0][:, None, :]
                    shift = shift + pl[:, 1][:, None, :]
                pre = layer(h) * scale + shift
                if self.film_concat_code:
                    pre = pre + self.concat_pl[li](codes)[:, None, :]
                h = self._act(pre)
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
# MLX-GPU SDF->argmax forward (FEED-eo, the --gpu-reorient core, additive). This is the MLX-GPU
# TWIN of the phi path in ``levelset_rgb_forward_numpy`` (the numpy ONE CODEPATH). It runs the same
# in_proj -> FiLM -> hidden -> out_sdf forward on the dequantized deploy weights, but in fp32 ON THE
# GPU (vs the numpy fp64 accumulation), so it is NOT bit-identical (the GPU vs numpy reduction order
# differs) -> the per-pair argmax it returns is PARITY-GATED, never an authority. Its sole consumer
# is the self-orientation reorient (recompute per-pair directional feats from the EMA argmax), which
# is itself a byte-closeable train-time PRIOR (cos 0.89-0.91 vs GT; the dir feats are a deterministic
# function of the witness's own argmax). Eliminating the 600 GPU-idle numpy CPU forwards (~499s every
# --reorient-every epochs at n600) is the ~6.2% wall-clock lever. NO mx ops touch ema.shadow/model.
# ---------------------------------------------------------------------------
def levelset_sdf_argmax_mlx(
    deploy_mx: dict,
    feats_mx,
    code_row_mx,
    *,
    n_hidden: int,
    hidden_dim: int,
    activation: str,
    wire_w0: float,
    wire_s0: float,
    hosc_beta: float,
    hosc_omega: float,
):
    """Return ``argmax_k phi_k`` (P,) int via the MLX-GPU twin of the numpy deploy forward.

    ``deploy_mx`` are the DEQUANTIZED deploy weights already as ``mx.array`` (in_proj/film/hidden.*/
    out_sdf {weight,bias}); ``feats_mx`` is the (P, in_feat) per-pair coord feature grid (curvelet
    [+ self-orient dir]); ``code_row_mx`` is the (mod_dim,) per-(pair,frame) FiLM code. Mirrors
    ``mlx.nn.Linear`` (``x @ W.T + b``) + ``LevelSetRGBWitness._act`` EXACTLY (only the device +
    fp32-vs-fp64 accumulation differ -> parity-gated, NOT the verdict authority). out_tex/palette/
    softmax are NOT computed (argmax of phi is the only quantity the reorient needs)."""
    import mlx.core as mx

    def _act(u):
        if activation == "wire":
            return mx.cos(wire_w0 * u) * mx.exp(-((wire_s0 * u) ** 2))
        if activation == "hosc":
            return mx.tanh(hosc_beta * mx.sin(hosc_omega * u))
        return mx.maximum(u, 0.0)

    h = _act(feats_mx @ deploy_mx["in_proj.weight"].T + deploy_mx["in_proj.bias"])
    film = (code_row_mx @ deploy_mx["film.weight"].T + deploy_mx["film.bias"]).reshape(n_hidden, 2, hidden_dim)
    # LEVER-A AUTO-DETECT (parity-gated reorient): apply the OPTIONAL per-layer FiLM / code-concat
    # routes when their keys are present so the self-orient reorient argmax reflects the trained
    # witness. ABSENT keys (default-off) => BYTE-IDENTICAL to the pre-LEVER-A twin.
    _has_film_pl = any(str(k).startswith("film_pl.") for k in deploy_mx)
    _has_concat = any(str(k).startswith("concat_pl.") for k in deploy_mx)
    for li in range(n_hidden):
        scale = 1.0 + film[li, 0]
        shift = film[li, 1]
        if _has_film_pl:
            pl = (code_row_mx @ deploy_mx[f"film_pl.{li}.weight"].T + deploy_mx[f"film_pl.{li}.bias"]).reshape(2, hidden_dim)
            scale = scale + pl[0]
            shift = shift + pl[1]
        pre = (h @ deploy_mx[f"hidden.{li}.weight"].T + deploy_mx[f"hidden.{li}.bias"]) * scale + shift
        if _has_concat:
            pre = pre + (code_row_mx @ deploy_mx[f"concat_pl.{li}.weight"].T + deploy_mx[f"concat_pl.{li}.bias"])
        h = _act(pre)
    phi = h @ deploy_mx["out_sdf.weight"].T + deploy_mx["out_sdf.bias"]  # (P, K)
    return mx.argmax(phi, axis=-1)


# ---------------------------------------------------------------------------
# Curriculum seg_form by epoch (PR95 d_seg sequence): ce -> tau_softplus -> l7_softplus.
# OPTIMIZER curriculum (DAG FEED-fi): AdamW for the CE/tau/l7 stages, then an OPTIONAL PR95
# stage-8 MUON FINISHER (--muon-start-epoch, default None=AdamW-throughout=BIT-IDENTICAL). At
# the switch epoch the optimizer becomes mlx.optimizers.MultiOptimizer([Muon(2D hidden weights),
# AdamW(biases/code/out_sdf/out_tex)]) via tac.optimization.muon_finisher_mlx (Newton-Schulz
# orthogonalized momentum = THE measured d_seg drop, CLAUDE.md frontier "Muon is THE drop"). The
# switch is a per-stage TREATMENT boundary (re-treat: spike-guard cleared) and saves a PRESERVED
# stage-encoded ckpt (independently byte-closeable + resumable). NO false claim: this is a build;
# the d_seg verdict is the realized-through-R eval, the score is upstream/evaluate.py only.
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
            f"{n_classes}-class comma10k CANONICAL partition [Road0,Lane1,Undrivable2,Movable3,MyCar4]; would "
            "IndexError mid-training. Use 1 for the lane orbit (the d_seg gate)."
        )


def validate_lane_thin_config(
    *, lane_thin_weight: float, lane_thin_start_epoch: int, epochs: int,
    lane_thin_class: int, lane_thin_radius: int, n_classes: int = 5,
) -> None:
    """(LEVER-B) thin-lane dropped-dash prior fail-closed config guard (pure; testable; fail LOUD).

    Mirrors ``validate_lane_edge_config``: a thin-lane lever that never engages (start > epochs) is a
    silent no-op = a FALSE 'thin-lane prior does not help' verdict; an out-of-range class would
    IndexError mid-training; a negative radius is malformed. When OFF (weight<=0) the guard is a
    NO-OP so the additive default path is never gated by a lever that is not in use."""
    if lane_thin_weight <= 0.0:
        return
    if lane_thin_start_epoch > epochs:
        raise ValueError(
            f"--lane-thin-weight {lane_thin_weight} > 0 but --lane-thin-start-epoch "
            f"({lane_thin_start_epoch}) > --epochs ({epochs}): the thin-lane hinge would NEVER engage "
            "-> a silent no-op = a FALSE 'thin-lane prior does not help' verdict. Set "
            "--lane-thin-start-epoch <= --epochs (0 = engage from ep1)."
        )
    if not (0 <= lane_thin_class <= n_classes - 1):
        raise ValueError(
            f"--lane-thin-class ({lane_thin_class}) out of range [0,{n_classes - 1}] for the "
            f"{n_classes}-class comma10k CANONICAL partition [Road0,Lane1,Undrivable2,Movable3,MyCar4]; "
            "would IndexError mid-training. Use 1 for the lane orbit (the d_seg gate)."
        )
    if lane_thin_radius < 0:
        raise ValueError(f"--lane-thin-radius ({lane_thin_radius}) must be >= 0 (window half-width).")


def lever_gate_on_at_epoch(weight: float, start_epoch: int, ep: int) -> bool:
    """Engagement predicate for the additive margin levers (lane-edge / margin-saliency / thin-lane).

    A lever is ENGAGED at training epoch ``ep`` iff its weight is > 0 AND the epoch has reached its
    ``start_epoch``. This is the SINGLE source of truth the epoch loop uses to (re-)flip every
    per-lever engagement gate every epoch. Extracting + unit-testing it is the SELF-PROTECT against
    the C1 silent-no-op class (review FEED-hp/hr): a gate initialized OFF for ``start_epoch>1`` that is
    NEVER re-flipped in the loop -> ``--<lever>-start-epoch>1`` (the help-RECOMMENDED 300) silently
    never engages -> a FALSE '<lever> does nothing' verdict from dead code. The C1 regression is
    EXACTLY ``lever_gate_on_at_epoch(w>0, start>1, ep=start)`` returning False; this predicate returns
    True, and the loop assigns its result, so the bug cannot silently re-emerge while this helper is
    the live decision. Pure + total => unit-testable at $0 (the realized-through-R loop needs MLX + the
    frozen scorer + the GT cache; this predicate does not). Per CLAUDE.md "Bugs must be permanently
    fixed AND self-protected against"."""
    return float(weight) > 0.0 and int(ep) >= int(start_epoch)


def _seg_form_for_epoch(ep: int, args) -> str:
    if not args.curriculum:
        return args.seg_loss
    if ep < args.tau_softplus_start_epoch:
        return "ce"
    if ep < args.l7_start_epoch:
        return "tau_softplus"
    return "l7_softplus"


def _hosc_beta_for_epoch(ep: int, args) -> float | None:
    """(FEED-fb) Annealed hosc ``beta`` at 1-based epoch ``ep``, or ``None`` when NO anneal applies.

    Returns ``None`` (caller leaves ``model.hosc_beta`` UNTOUCHED => BIT-IDENTICAL constant-beta path)
    when: activation != ``hosc``, OR ``--hosc-beta-end`` is unset, OR end == start. Otherwise anneals
    ``beta`` from ``--hosc-beta`` (at ep==1) to ``--hosc-beta-end`` (at ep==args.epochs) on a linear
    (default) or cosine schedule. The step-native L-infinity-optimal lever: ``beta -> inf`` makes
    ``tanh(beta*sin)`` approach a step (the topology-matched chart for the piecewise-constant argmax,
    no Gibbs). Pure (no model/MLX); unit-tested. Mirrors ``_seg_form_for_epoch``.
    """
    if (getattr(args, "activation", None) != "hosc"
            or getattr(args, "hosc_beta_end", None) is None
            or args.hosc_beta_end == args.hosc_beta):
        return None
    # (review C2) same anneal denominator as _softmax_temp_for_epoch: --anneal-epochs (schedule
    # length) NOT --epochs (run length). Default None => args.epochs => BIT-IDENTICAL.
    _ae = getattr(args, "anneal_epochs", None) or args.epochs
    prog = (ep - 1) / max(_ae - 1, 1)
    if getattr(args, "hosc_beta_anneal", "linear") == "cosine":
        return float(args.hosc_beta_end + 0.5 * (args.hosc_beta - args.hosc_beta_end) * (1 + np.cos(np.pi * prog)))
    return float(args.hosc_beta + (args.hosc_beta_end - args.hosc_beta) * prog)


def _softmax_temp_for_epoch(ep: int, args) -> float:
    """(config-review #4) Cosine-annealed softmax temperature at 1-based epoch ``ep`` (hi->lo: soft
    start so gradients flow with no RGB-level Gibbs -> sharp end with the SDF partition pinned). Pure
    (no model/MLX); unit-tested. Mirrors ``_seg_form_for_epoch`` / ``_hosc_beta_for_epoch``. Extracted
    from the inline loop anneal so the MUON FINISHER can FREEZE it at the muon-start value (FEED-fm).
    Returns the EXACT value the pre-extraction inline formula produced (BIT-IDENTICAL) when
    --anneal-epochs is unset.

    (review C2) ANNEAL DENOMINATOR: the cosine progress uses ``--anneal-epochs`` (the SCHEDULE length)
    NOT ``--epochs`` (the run length). Default None => falls back to ``args.epochs`` => BIT-IDENTICAL.
    A WARM-START arm (resume the CE ckpt @ ep299, run 100 epochs => --epochs 399) must set
    --anneal-epochs to the ORIGINAL schedule length (1500) so ep300->400 reproduces the DISEASE
    regime temp (~0.91->0.84), not the schedule tail (~0.19->0.05). ``None or x == x`` and 0 is
    treated as unset, so the default path is the pre-C2 formula bit-for-bit."""
    _ae = getattr(args, "anneal_epochs", None) or args.epochs
    prog_t = (ep - 1) / max(_ae - 1, 1)
    return float(args.softmax_temp_end + 0.5 * (args.softmax_temp_start - args.softmax_temp_end) * (1 + np.cos(np.pi * prog_t)))


def _stage_rewarmup_factor(
    ep: int, last_boundary_epoch: "int | None", rewarmup_epochs: int, floor: float, shape: str,
) -> float:
    """(BUILD 1 / FEED-fw) LR re-warmup multiplier in (0, 1] at 1-based epoch ``ep`` after an
    AdamW->AdamW stage boundary. DEFAULT-OFF: ``rewarmup_epochs <= 0`` (or no boundary yet) =>
    returns EXACTLY 1.0 => the LR schedule is BIT-IDENTICAL to the pre-FEED-fw path (x*1.0 == x for
    finite IEEE floats). After a registered stage TRANSITION at ``last_boundary_epoch``, ramp the
    multiplier from ``floor`` (at the boundary epoch, offset 0) back to 1.0 over ``rewarmup_epochs``
    epochs -- linear (default) or cosine.

    Rationale (operator 2026-06-26 "different stages need different treatment ... transitions must
    re-treat"; FEED-ft#3 tau-jump root cause): a loss-landscape change at a boundary, hit with FULL
    LR + stale AdamW momentum, is the instability. Ramping the LR back up gives the (optionally
    reset) optimizer state time to re-warm against the NEW stage's landscape, making the transition
    stable by construction. Pure (no model/MLX); unit-tested. Mirrors the per-epoch schedule helpers
    above."""
    if rewarmup_epochs <= 0 or last_boundary_epoch is None:
        return 1.0
    d = ep - last_boundary_epoch
    if d < 0 or d >= rewarmup_epochs:
        return 1.0
    floor = float(min(max(floor, 0.0), 1.0))
    prog = d / float(rewarmup_epochs)  # 0 at the boundary epoch -> ->1 across the window
    if shape == "cosine":
        return float(floor + (1.0 - floor) * 0.5 * (1.0 - np.cos(np.pi * prog)))
    return float(floor + (1.0 - floor) * prog)


def _rng_state_arrays(hardness_rng: "np.random.Generator | None") -> dict[str, np.ndarray]:
    """(FEED-fm FIX-1) Snapshot EVERY RNG the TRAINING LOOP advances, so a ``--resume-from`` run
    reproduces the CONTINUOUS draw sequence bit-for-bit (the deterministic-reproducibility
    non-negotiable: resume == continuous). The loop advances exactly TWO streams:

      * the GLOBAL ``np.random`` MT19937 -- the per-epoch ``np.random.permutation(P)`` pair order
        (and the ``permutation(concat)`` when hardness-oversample extras are appended); and
      * the LEVER-5 ``hardness_rng`` PCG64 ``Generator`` -- the ``hardness_rng.choice`` oversample.

    NO OTHER ``np.random.*`` call exists in the loop (verified: verdict/quantize/reorient/hardness-
    precompute touch neither global state), so snapshotting at checkpoint time + restoring at resume
    is exact. Keys are ``__``-prefixed so ``_load_resume_state`` routes them to ``cfg`` (the 624-key
    MT19937 array becomes a list there; the PCG64 dict is JSON-stringified). MLX-free; allow_pickle
    is NOT required to reload (plain arrays + unicode str)."""
    out: dict[str, np.ndarray] = {}
    algo, keys, pos, has_gauss, cached_gauss = np.random.get_state(legacy=True)
    out["__rng_np_algo"] = np.asarray(str(algo))
    out["__rng_np_keys"] = np.asarray(keys, np.uint32)
    out["__rng_np_pos"] = np.asarray(int(pos))
    out["__rng_np_has_gauss"] = np.asarray(int(has_gauss))
    out["__rng_np_cached_gauss"] = np.asarray(float(cached_gauss))
    if hardness_rng is not None:
        out["__rng_hardness_json"] = np.asarray(json.dumps(hardness_rng.bit_generator.state))
    return out


def _restore_rng_state(cfg: dict[str, Any], hardness_rng: "np.random.Generator | None") -> dict[str, bool]:
    """(FEED-fm FIX-1) Restore the RNG snapshot from a resume sidecar's ``cfg`` (the dict
    ``_load_resume_state`` returns). DEFAULT-SAFE / back-compat: a pre-FEED-fm checkpoint lacking the
    ``__rng_*`` keys leaves the freshly-seeded RNGs UNTOUCHED (exactly the pre-fix behavior; no crash)
    -- guarded by presence checks. Returns which streams were restored (observability). NO-FAKE: this
    really sets the global MT19937 + the PCG64 generator state so the next draw matches a continuous
    run; it is not a marker."""
    restored = {"np_global": False, "hardness": False}
    if "__rng_np_keys" in cfg and "__rng_np_pos" in cfg:
        keys = np.asarray(cfg["__rng_np_keys"], dtype=np.uint32)
        np.random.set_state((
            str(cfg.get("__rng_np_algo", "MT19937")), keys, int(cfg["__rng_np_pos"]),
            int(cfg.get("__rng_np_has_gauss", 0)), float(cfg.get("__rng_np_cached_gauss", 0.0)),
        ))
        restored["np_global"] = True
    if hardness_rng is not None and "__rng_hardness_json" in cfg:
        try:
            hardness_rng.bit_generator.state = json.loads(str(cfg["__rng_hardness_json"]))
            restored["hardness"] = True
        except Exception:  # malformed/foreign state: keep the fresh PCG64 (best-effort, no crash).
            pass
    return restored


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
        film_per_layer=bool(getattr(args, "film_per_layer", False)),
        film_concat_code=bool(getattr(args, "film_concat_code", False)),
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
        # BUILD 2 (FEED-fw): inject the openpilot deg-3 centerline lane SDF into the phi1 channel of
        # the structured target BEFORE the joint pretrain absorbs it. DEFAULT-OFF (--lane-prior-phi1
        # off) => phi_tgt_hwk is UNTOUCHED => the structured-init pretrain is BIT-IDENTICAL. Reuses
        # the standalone-geometry helpers (numpy/scipy, $0 CPU): build_structured_lane_sdf is the
        # ground-plane homography (K @ scorer-res {fx=910*512/1164=400.3,...}) -> deg-3 lane curve ->
        # per-pixel signed distance (FEED-fs separatrix, residual 1.9e-5); inject_lane_sdf writes it
        # into the K-field stack. The fit is from the cached L* (frozen CPU-torch argmax) of the
        # chosen pair. rule-118 FREE generic structure: train-time init only, ships 0 archive bytes.
        if getattr(args, "lane_prior_phi1", False):
            from tac.boundary_math.lane_sdf_component import (
                build_structured_lane_sdf,
                inject_lane_sdf,
            )
            _lp_pair = int(args.lane_prior_phi1_source_pair)
            if not (0 <= _lp_pair < P):
                raise ValueError(
                    f"--lane-prior-phi1-source-pair ({_lp_pair}) out of range [0,{P - 1}].")
            _lp_lstar = np.asarray(gt.lstars[_lp_pair], np.int64)
            phi1_lane, lp_meta = build_structured_lane_sdf(
                _lp_lstar, lane_cls=1, dash_gate=bool(args.lane_prior_phi1_dash_gate),
                centerline_deg=3)
            phi_tgt_hwk = inject_lane_sdf(
                phi_tgt_hwk, phi1_lane, lane_cls=1, mode=args.lane_prior_phi1_mode,
                bias_scale=float(args.lane_prior_phi1_bias_scale))
            print(json.dumps({"stage": "lane_prior_phi1", "active": True, "source_pair": _lp_pair,
                              "mode": args.lane_prior_phi1_mode,
                              "dash_gate": bool(args.lane_prior_phi1_dash_gate),
                              **{f"lane_{k}": v for k, v in lp_meta.items()},
                              "note": "openpilot deg-3 centerline SDF injected into structured-init "
                              "phi1 target (FEED-fs Road<->Lane separatrix; train-time init, 0 "
                              "archive bytes)"}), flush=True)
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
    # AMORTIZATION (FEED-eo, --freeze-decoder-fit-codes, ADDITIVE, default-off). The witness factors
    # into a SHARED decoder (in_proj/film/hidden/out_sdf/out_tex/palette) + per-(pair,frame) latent
    # codes (1200 x mod_dim). A full from-scratch n600 row co-fits BOTH (days). This mode LOADS a
    # decoder trained on a SUBSET (n96/n192), FREEZES it, and fits ONLY the ~num_pairs*2*mod_dim
    # codes for all pairs (a small per-pair optimization through the frozen render+R+scorer ->
    # embarrassingly parallel per pair; hours not days) -> the future-row fast path IF the frozen
    # shared decoder generalizes (the small-n estimate measures this). Loaded BEFORE EMA so the EMA
    # shadow (the deploy weights) starts at the frozen decoder; freeze BEFORE value_and_grad so the
    # grad/optimizer/weight-decay only ever touch ``code`` (the decoder cannot drift). Default
    # None => skipped => byte-identical to a normal joint run.
    freeze_decoder = bool(getattr(args, "freeze_decoder_fit_codes", None))
    if freeze_decoder:
        if args.resume_from:
            raise ValueError("--freeze-decoder-fit-codes is incompatible with --resume-from (one "
                             "loads a frozen decoder + FRESH codes; the other restores a full state).")
        if args.structured_init:
            raise ValueError("--freeze-decoder-fit-codes is incompatible with --structured-init "
                             "(the decoder is frozen-from-file, not pretrained).")
        if args.film_stiefel:
            # (review Med2) the freeze invariant is "only `code` trains"; --film-stiefel projects
            # model.film.weight (a FROZEN decoder param) every step, mutating a frozen weight OUTSIDE
            # the optimizer/freeze mechanism = a freeze-invariant violation AND a silent no-op for the
            # cure (the decoder is fixed, so there is nothing to orthonormalize the trajectory of).
            raise ValueError("--film-stiefel is incompatible with --freeze-decoder-fit-codes: the "
                             "Stiefel projection mutates the FROZEN decoder's film.weight every step "
                             "(violates the 'only code trains' freeze invariant). Run the Stiefel cure "
                             "on a joint (unfrozen) run.")
        from mlx.utils import tree_unflatten
        dec = _load_decoder_params(Path(args.freeze_decoder_fit_codes))
        got_in = int(dec["in_proj.weight"].shape[1])
        if got_in != in_feat:
            raise ValueError(
                f"--freeze-decoder-fit-codes in_feat MISMATCH: the decoder's in_proj expects {got_in} "
                f"but the current front-end config yields in_feat={in_feat}. Match the decoder's "
                "training config (--bank-*/--max-bank-freq/--self-orient/--n-dir-freqs) so the curvelet"
                "[+dir] feature width agrees; NO-FAKE: refusing to fit codes against a width-mismatched "
                "decoder.")
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in dec.items()]))
        mx.eval(model.parameters())
        model.freeze(recurse=True)
        model.unfreeze(keys=["code"])
        tnames = sorted(k for k, _ in tree_flatten(model.trainable_parameters()))
        if tnames != ["code"]:
            raise RuntimeError(
                f"--freeze-decoder-fit-codes: expected ONLY 'code' trainable after freeze, got {tnames} "
                "(MLX freeze/unfreeze contract changed); fail-closed so the decoder cannot silently train.")
        print(json.dumps({"stage": "freeze_decoder_fit_codes", "decoder_from": str(args.freeze_decoder_fit_codes),
                          "in_feat": int(in_feat), "trainable": tnames, "n_code_params": int(model.code.size),
                          "note": "shared decoder FROZEN (no weight-decay drift); fitting per-pair codes only "
                          "(amortization fast path -- viability per the small-n generalization estimate)"}), flush=True)
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

    # LEVER-4 (margin-saliency) closure constants (static; ZERO change to the value_and_grad call
    # site). msal_w=0.0 (default) => the branch is skipped => behavior IDENTICAL (fully additive).
    msal_w = float(args.margin_saliency_weight)
    msal_tau = float(args.margin_saliency_tau)
    msal_tgt = float(args.margin_saliency_target)
    msal_start = int(args.margin_saliency_start_epoch)
    msal_uni = bool(args.margin_saliency_uniward)
    msal_uni_beta = float(args.margin_saliency_uniward_beta)
    msal_gate = {"on": msal_start <= 1}

    # LEVER-A (FiLM-rank-fix) loss term closure constants. A SOFT participation-ratio FLOOR on the
    # realized per-pair FiLM modulation M = film(code) so the curriculum cannot funnel it to rank-1
    # (MEASURED collapse PR 3.34@CE -> 1.19@l7). rankfloor_w=0.0 (default) => the branch is skipped =>
    # behavior IDENTICAL (fully additive). Computed over a FIXED deterministic subsample of the
    # per-(pair,frame) codes (<= cap, strided) so the S x S Gram is cheap; the penalty is
    # pair-INDEPENDENT, so accumulating it per-pair then averaging counts it ONCE (correct magnitude;
    # redundant compute bounded by the cap). It penalizes the SHARED film route (the measured-collapse
    # determinant); film_pl residual routes are not directly penalized but the shared route dominates
    # the per-pair modulation rank.
    rankfloor_w = float(getattr(args, "film_rank_floor_weight", 0.0))
    rankfloor_tgt = float(getattr(args, "film_rank_floor_target", 4.0))
    rankfloor_idx = None
    if rankfloor_w > 0.0:
        _ncodes = 2 * P
        _cap = 256
        _stride = max(1, _ncodes // _cap)
        rankfloor_idx = mx.array(np.arange(0, _ncodes, _stride)[:_cap].astype(np.int32))

    # DM1b (code spectral-entropy) loss-term closure. A CAPACITY log-barrier -beta*log(PR(cov(code)))
    # on the per-pair code covariance (keeps all ~mod_dim code directions live). Pair-INDEPENDENT (a
    # function of the whole code matrix), so -- exactly like the rank-floor -- accumulating it per-pair
    # then averaging counts it ONCE. PR is computed via the (D,D) covariance Gram (cheap, no eigh),
    # the EXACT MLX twin of tac...code_spectral_entropy_penalty. code_spec_w=0.0 (default) => the
    # branch is skipped => behavior IDENTICAL (fully additive). Same fixed deterministic subsample as
    # the rank-floor so the Gram is bounded.
    code_spec_w = float(getattr(args, "code_spectral_entropy_weight", 0.0))
    code_spec_idx = None
    if code_spec_w > 0.0:
        _ncodes2 = 2 * P
        _cap2 = 256
        _stride2 = max(1, _ncodes2 // _cap2)
        code_spec_idx = mx.array(np.arange(0, _ncodes2, _stride2)[:_cap2].astype(np.int32))

    # LEVER-B (thin-lane dropped-dash prior) closure constants. Up-weight the realized through-R seg
    # margin hinge on THIN GT-lane structures the unweighted mean loss drops (MEASURED: 52.7% of
    # GT-lane connected components wholesale-missed, miss-fraction monotone in dash size). lane_thin_w
    # =0.0 (default) => the branch is skipped => behavior IDENTICAL (fully additive). The per-pair
    # thin-lane weight map (local lane density in a (2r+1)^2 window) is PRECOMPUTED ONCE from the
    # cached L* (deterministic; NOT recomputed per step) and looked up by pair index inside the loss.
    # When lane_thin_start>1 the engagement epoch RE-TREATS the spike-guard (same as LEVER-3/4).
    lane_thin_w = float(getattr(args, "lane_thin_weight", 0.0))
    lane_thin_tgt = float(getattr(args, "lane_thin_target", 0.5))
    lane_thin_cls = int(getattr(args, "lane_thin_class", 1))
    lane_thin_rad = int(getattr(args, "lane_thin_radius", 4))
    lane_thin_start = int(getattr(args, "lane_thin_start_epoch", 0))
    lane_thin_gate = {"on": lane_thin_start <= 1}
    thin_maps_mx = None
    if lane_thin_w > 0.0:
        thin_maps_mx = {
            pi: mx.array(lane_thin_weight_map(
                np.asarray(gt.lstars[pi]), lane_class=lane_thin_cls, radius=lane_thin_rad)[None])
            for pi in range(P)
        }

    def total_loss_fn(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w):
        L = base_loss(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form=seg_form)
        phi0 = model.sdf(cf, c0)
        eik, length, _ = _eikonal_length_mlx(phi0, render_h, render_w)
        L = L + eik_w * eik + len_w * length
        # (review R2b-M3) SHARED realized through-R seg forward. LEVER-3 (lane-edge), LEVER-4
        # (margin-saliency) and LEVER-B (thin-lane) all need the SAME realized decision margin
        # ``signed = gt_logit - top_competitor`` from the SAME render(cf,c1)->R->frozen SegNet. The
        # render is deterministic (uint8-STE round; no training noise), so computing it ONCE and
        # reusing it across the stacked levers is BIT-IDENTICAL to the prior 3-separate-forwards code
        # while doing 1 (not up to 3) of the expensive forward. Computed ONLY when >=1 seg-margin lever
        # is engaged; default-off (all weights 0) => _seg_levers_on False => block skipped =>
        # byte-identical to the additive default path. ``_f1`` is also reused for LEVER-4's UNIWARD
        # texture map (same rendered frame).
        _seg_levers_on = ((lane_w > 0.0 and lane_gate["on"]) or
                          (msal_w > 0.0 and msal_gate["on"]) or
                          (lane_thin_w > 0.0 and lane_thin_gate["on"]))
        if _seg_levers_on:
            _f1 = render_through_R_mlx(model, cf, c1, render_h, render_w)  # (1, SEG_H, SEG_W, 3)
            _slog = adapter.segnet(_f1)                                    # (1, H, W, 5)
            _sig_gt = mx.sum(_slog * lstar_oh, axis=-1)                    # (1, H, W) gt-class logit
            _sig_run = mx.max(_slog + lstar_oh * (-1e9), axis=-1)          # (1, H, W) top competitor
            _signed = _sig_gt - _sig_run                                   # (1, H, W) realized margin
        # LEVER-3 (lane-edge fragility weighting, operator 2026-06-27 Yousfi-grounding): contest
        # SegNet argmax order is the comma10k CANONICAL order (MEASURED 2026-06-27 from the cached
        # argmax; CLAUDE.md NON-NEGOTIABLE): [Road0, Lane1, Undrivable2, Movable3, MyCar4]. The
        # FORBIDDEN luma-sort of class_values [41,76,90,124,161] -> [Road0,Lane1,MyCar2,Undriv3,Movable4]
        # is WRONG for 2/3/4 (bit us 3x); do NOT use it. Class0=Road & Class1=Lane are CONFIRMED in
        # BOTH orders (so this lever, which uses ONLY class 1, is correct regardless). Lane (class 1) is thin
        # all-boundary double-edges (19% of d_seg flips) and UNDER-FIT because the CE baseline has NO
        # class weighting. This ADDITIVE term up-weights the REALIZED (through-R SegNet) margin hinge
        # at GT-lane pixels: it renders f1 -> R -> frozen SegNet logits, takes the live decision
        # margin (gt_logit - top_competitor) ONLY where GT==lane, and penalizes relu(target-margin)
        # there. The hinge fires exactly on SMALL-MARGIN (fragile = boundary) lane pixels, so it
        # adds gradient pressure to widen the lane margin at the lane double-edges. Default-off
        # (lane_w=0). When ON it reuses the SHARED realized seg forward above (review R2b-M3: no
        # longer a separate render -- bit-identical, 1 forward shared across the stacked levers).
        if lane_w > 0.0 and lane_gate["on"]:
            lane_mask = lstar_oh[..., lane_cls]                         # (1, H, W) 1.0 where GT==lane
            hinge_map = mx.maximum(lane_tgt - _signed, 0.0) * lane_mask  # fragile lane pixels only
            lane_term = mx.sum(hinge_map) / (mx.sum(lane_mask) + 1e-6)  # mean hinge over lane px
            L = L + lane_w * lane_term
        # LEVER-4 (margin-saliency, all-class generalization of LEVER-3). Same realized through-R
        # decision margin, but the hinge is weighted PER-PIXEL by the GT-margin fragility saliency
        # sal=exp(-gt_margin/tau) over EVERY GT pixel (not a single class mask). The flip-prone band
        # (small GT margin) lives across all classes (Road 47% / Lane 19% / Undriv 14% / ...), so this
        # adds widen-the-margin pressure exactly where d_seg lives. CLASS-AGNOSTIC. Default-off.
        if msal_w > 0.0 and msal_gate["on"]:
            sgn = _signed                                              # (1, H, W) SHARED realized margin (R2b-M3)
            sal = mx.exp(-margin / msal_tau)                            # (1, H, W) fragility weight
            if msal_uni:
                # UNIWARD: down-weight textured regions (SegNet-undetectable) -> concentrate on the
                # SMOOTH boundary. Texture energy from the realized frame's spatial gradients, used as
                # a STOP-GRAD weight (a cost map, not a loss path). Reuses the SHARED rendered frame _f1.
                lum = mx.mean(mx.stop_gradient(_f1), axis=-1)            # (1, H, W)
                dy = mx.pad(mx.abs(lum[:, 1:, :] - lum[:, :-1, :]), [(0, 0), (0, 1), (0, 0)])
                dx = mx.pad(mx.abs(lum[:, :, 1:] - lum[:, :, :-1]), [(0, 0), (0, 0), (0, 1)])
                tex = dy + dx
                tex = tex / (mx.max(tex) + 1e-6)                         # [0,1]
                sal = sal / (1.0 + msal_uni_beta * tex)
            hmap = mx.maximum(msal_tgt - sgn, 0.0) * sal                 # fragile pixels weighted
            msal_term = mx.sum(hmap) / (mx.sum(sal) + 1e-6)             # saliency-weighted mean hinge
            L = L + msal_w * msal_term
        # LEVER-A (FiLM-rank-fix) soft participation-ratio FLOOR. Pushes the per-pair modulation PR up
        # toward rankfloor_tgt (opposing the measured rank-1 collapse). PR computed Gram-wise (NO
        # eigendecomposition): trace(C)=||Mc||_F^2 (== mx.sum(Mc*Mc)), ||C||_F^2=||Mc Mc^T||_F^2. The
        # numpy reference is tac...film_modulation_participation_ratio / film_rank_floor_penalty.
        # Default-off (rankfloor_w=0). Mirrors the numpy reference EXACTLY (one math, two backends).
        if rankfloor_w > 0.0 and rankfloor_idx is not None:
            M = model.film(model.code[rankfloor_idx])                   # (S, D) modulation
            Mc = M - mx.mean(M, axis=0, keepdims=True)
            tr = mx.sum(Mc * Mc)                                        # trace(Gram) = sum eigenvalues
            G = Mc @ Mc.T                                               # (S, S) Gram
            fro2 = mx.sum(G * G)                                        # sum eigenvalues^2
            pr = (tr * tr) / (fro2 + 1e-12)                            # participation ratio in [1, S]
            L = L + rankfloor_w * mx.maximum(rankfloor_tgt - pr, 0.0)
        # DM1b (code spectral-entropy CAPACITY penalty): -beta*log(PR(cov(code))) on the per-pair code
        # covariance C = cov(code). Maximizes PR(cov(code)) => keeps all ~mod_dim code directions live;
        # via the Stiefel identity (--film-stiefel) WᵀW=I => PR(M)=PR(cov(code)) this is the other half
        # of the byte-free DM1 cure. PR via the (D,D) covariance Gram (no eigendecomposition): C=Cc^T Cc
        # (the 1/(S-1) cancels in the ratio). Default-off (code_spec_w=0). EXACT MLX twin of the numpy
        # tac...code_spectral_entropy_penalty (one math, two backends). The gradient flows to the
        # `code` latent (spreading its spectrum); film.weight is handled by the Stiefel projection, so
        # the two halves target DIFFERENT params (no double-count, design memo §3 routing).
        if code_spec_w > 0.0 and code_spec_idx is not None:
            Cm = model.code[code_spec_idx]                              # (S, D) per-pair codes
            Cc = Cm - mx.mean(Cm, axis=0, keepdims=True)
            Cov = Cc.T @ Cc                                            # (D, D) ~ cov(code)
            ctr = mx.sum(Cc * Cc)                                      # trace(Cov) = sum eigenvalues
            cfro2 = mx.sum(Cov * Cov)                                  # sum eigenvalues^2
            cpr = (ctr * ctr) / (cfro2 + 1e-12)                        # PR(cov(code)) in [1, D]
            L = L - code_spec_w * mx.log(cpr + 1e-12)                  # -beta*log(PR) => raises PR
        # LEVER-B (thin-lane dropped-dash prior): realized through-R margin hinge weighted by the
        # PRECOMPUTED thin-lane map (nonzero ONLY on thin GT-lane pixels). Same realized decision
        # margin as LEVER-3 but concentrated on the DROPPED thin dashes (the PC0 residual). c0=2*pi
        # so c0//2 == pi keys the per-pair thin map to THIS pair's lstar_oh. Default-off (lane_thin_w
        # =0). Reuses the SHARED realized seg forward above (review R2b-M3: no separate render --
        # bit-identical, 1 forward shared across the stacked levers).
        if lane_thin_w > 0.0 and lane_thin_gate["on"] and thin_maps_mx is not None:
            tw = thin_maps_mx[int(c0) // 2]                            # (1, H, W) thin-lane weight (>=0)
            hmap_t = mx.maximum(lane_thin_tgt - _signed, 0.0) * tw     # fragile thin-lane pixels only
            L = L + lane_thin_w * (mx.sum(hmap_t) / (mx.sum(tw) + 1e-6))
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
            # (FEED-fb) CURRENT (possibly annealed) beta -> the verdict/deploy render uses the SAME
            # beta the model is at now (NO-FAKE). Bit-identical when anneal off: model.hosc_beta == args.hosc_beta.
            wire_w0=args.wire_w0, wire_s0=args.wire_s0, hosc_beta=float(model.hosc_beta), hosc_omega=args.hosc_omega,
            chroma=args.chroma,
        )

    def _render_numpy_deploy(deploy: dict[str, np.ndarray], pi: int, fk: int) -> np.ndarray:
        """THE ONE CODEPATH (fp32 numpy, deploy-faithful) — same forward the byte-close/inflate use.
        Uses the PER-PAIR feats (curvelet [+ self-orient]) so the verdict == the deploy render."""
        rgb, _phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + fk])
        return _torch_R_to_camera_uint8(rgb.reshape(render_h, render_w, 3))

    def _dir_feats_from_argmax(argmax: np.ndarray) -> np.ndarray:
        """argmax (H,W) int -> self-orientation directional feats (P, dir_w). SAME numpy/scipy
        tangent->fourier path for BOTH the numpy and GPU reorient (only the argmax SOURCE differs)."""
        return self_orientation_directional_feats(
            coords_np, argmax, n_freqs=n_dir_freqs,
            freq_across=args.freq_across, freq_along=args.freq_along).astype(np.float32)

    def _recompute_self_orient_gpu(deploy: dict[str, np.ndarray]) -> float:
        """FEED-eo --gpu-reorient: the per-pair argmax (the GPU-idle 600-numpy-forward bottleneck,
        ~499s every reorient at n600) is computed on MLX-GPU via the fp32 twin forward instead. The
        downstream tangent->directional-fourier feats stay the SAME numpy/scipy path. PARITY-GATED
        (fp32-GPU vs fp64-numpy argmax differs at boundary px) -> default-off; adopt only after the
        probe shows cos>0.999 + negligible d_seg A/B. The deploy weights are dequantized ONCE to mx;
        per-pair feats are built+freed one-at-a-time (memory-bounded, like the numpy path)."""
        deploy_mx = {k: mx.array(np.asarray(v, np.float32)) for k, v in deploy.items()
                     if k not in ("code",) and not (k == "B" or k.endswith("_B"))}
        codes_np = np.asarray(deploy["code"], np.float32)
        mag = 0.0
        with temporary_mlx_device(args.mlx_device):
            for pi in range(P):
                feats_mx = mx.array(_feats_np_for_pair(pi))
                code_row = mx.array(codes_np[2 * pi + 1])
                amx = levelset_sdf_argmax_mlx(
                    deploy_mx, feats_mx, code_row, n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
                    activation=args.activation, wire_w0=args.wire_w0, wire_s0=args.wire_s0,
                    hosc_beta=float(model.hosc_beta), hosc_omega=args.hosc_omega)  # FEED-fb current beta
                mx.eval(amx)
                argmax = np.asarray(amx).reshape(render_h, render_w).astype(np.int64)
                df = _dir_feats_from_argmax(argmax)
                dir_feats_per_pair[pi] = df
                mag += float(np.abs(df).mean())
                del feats_mx, amx, code_row
            mx.clear_cache()
        return mag / max(P, 1)

    def recompute_self_orient(deploy: dict[str, np.ndarray]) -> float:
        """Self-orientation FIXED-POINT step: from the EMA deploy frame1 argmax (current feats),
        recompute each pair's directional feats. Returns the mean |dir feat| (non-triviality check)."""
        if not use_self_orient:
            return 0.0
        if getattr(args, "gpu_reorient", False):
            return _recompute_self_orient_gpu(deploy)
        mag = 0.0
        for pi in range(P):
            _rgb, phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + 1])
            argmax = phi.argmax(-1).reshape(render_h, render_w).astype(np.int64)
            df = self_orientation_directional_feats(
                coords_np, argmax, n_freqs=n_dir_freqs, freq_across=args.freq_across, freq_along=args.freq_along)
            dir_feats_per_pair[pi] = df.astype(np.float32)
            mag += float(np.abs(df).mean())
        return mag / max(P, 1)

    def _project_shadow_film_np(params_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """(review Med1) Re-orthonormalize the EMA SHADOW's film.weight for the DEPLOYED artifact.

        The EMA shadow is an arithmetic average of (per-step on-manifold) film.weight matrices, which
        is itself NOT orthonormal -> the shipped/verdicted weight drifts OFF-Stiefel and PR(M)=PR(cov
        code) no longer holds for what actually ships. Re-project film.weight onto orthonormal columns
        so the DEPLOYED (verdict + byte-close) weight is on-manifold. Returns a SHALLOW copy with
        film.weight replaced; the live ``ema.shadow`` is UNTOUCHED so --resume-from stays bit-faithful
        to a continuous run (the resume sidecar keeps the un-projected shadow). No-op unless
        --film-stiefel (default OFF => byte-identical)."""
        if not args.film_stiefel or "film.weight" not in params_np:
            return params_np
        out = dict(params_np)
        out["film.weight"] = np.asarray(
            stiefel_project_columns(mx.array(params_np["film.weight"])), np.float32)
        return out

    def realized_verdict() -> dict[str, float]:
        # (fix a+b+c) verdict the EMA SHADOW, int8-DEQUANTIZED, via the fp32 numpy ONE CODEPATH
        # (NOT the MLX-GPU reduced-precision forward — the 4th artifact). This IS the deploy render.
        # (review Med1) project the shadow film.weight back onto Stiefel so the advisory d_seg reflects
        # the ON-MANIFOLD deployed weight (no-op unless --film-stiefel => bit-identical).
        ema_np = _project_shadow_film_np({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()})
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

    # ---- ASYNC verdict (FEED-em; ADDITIVE, DEFAULT-OFF via --async-verdict). The realized
    # CPU-torch verdict (render fp32 numpy + SegNet/PoseNet) is PURELY OBSERVATIONAL — the
    # training loop NEVER reads its result — so running it in a BACKGROUND THREAD off a
    # POINT-IN-TIME snapshot does NOT change the training trajectory at all (BIT-IDENTICAL
    # weights/checkpoints; only the verdict CADENCE may self-throttle under load). Mirrors the
    # base_ch20 async-CPU-authority pattern in src/tac/torch_vehicle/driver.py. The snapshot is
    # captured on the MAIN thread (cheap) so the worker reads ONLY its own copies + constants
    # (curv_feats_np, gt, frozen scorers) -> RACE-FREE (it never touches ema.shadow / model /
    # dir_feats_per_pair / cf_mx_cache, all of which the main loop keeps mutating). The worker
    # uses NO MLX op (pure numpy+torch) so it cannot race the GPU stream.
    def _capture_verdict_snapshot() -> dict[str, Any]:
        return {
            # (review Med1) project the shadow film.weight on-manifold so the ASYNC verdict matches the
            # deployed (byte-closed) artifact (no-op unless --film-stiefel => bit-identical snapshot).
            "ema_np": _project_shadow_film_np({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}),
            "softmax_temp": float(model.softmax_temp),
            "hosc_beta": float(model.hosc_beta),  # FEED-fb: snapshot the live (possibly annealed) beta
            "dir": ({pi: dir_feats_per_pair[pi].copy() for pi in vpairs} if use_self_orient else None),
        }

    def _feats_for_snapshot(pi: int, dir_snap) -> np.ndarray:
        if not use_self_orient:
            return curv_feats_np
        return np.concatenate([curv_feats_np, dir_snap[pi]], axis=-1).astype(np.float32)

    def _verdict_from_snapshot(snap: dict[str, Any]) -> dict[str, float]:
        # BIT-IDENTICAL to realized_verdict() on the captured state: same int8 dequant, same
        # fp32 ONE-CODEPATH forward, same softmax_temp, same per-pair feats, same CPU scorers.
        deploy = int8_dequant_params(snap["ema_np"])
        st = snap["softmax_temp"]
        sb = snap["hosc_beta"]  # FEED-fb: the live beta captured at schedule time (anneal-correct, NO-FAKE)
        f0s, f1s = [], []
        for pi in vpairs:
            fnp = _feats_for_snapshot(pi, snap["dir"])
            rgb0, _ = levelset_rgb_forward_numpy(
                deploy, fnp, deploy["code"][2 * pi + 0], n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
                n_classes=5, activation=args.activation, softmax_temp=st, wire_w0=args.wire_w0,
                wire_s0=args.wire_s0, hosc_beta=sb, hosc_omega=args.hosc_omega, chroma=args.chroma)
            rgb1, _ = levelset_rgb_forward_numpy(
                deploy, fnp, deploy["code"][2 * pi + 1], n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
                n_classes=5, activation=args.activation, softmax_temp=st, wire_w0=args.wire_w0,
                wire_s0=args.wire_s0, hosc_beta=sb, hosc_omega=args.hosc_omega, chroma=args.chroma)
            f0s.append(_torch_R_to_camera_uint8(rgb0.reshape(render_h, render_w, 3)))
            f1s.append(_torch_R_to_camera_uint8(rgb1.reshape(render_h, render_w, 3)))
        ds = cpu_verdict_d_seg_batch(seg_cpu, f1s, [gt.lstars[pi] for pi in vpairs])
        dp = cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, [gt.gt_poses[pi] for pi in vpairs])
        return {"d_seg": float(np.mean(ds)), "d_pose": float(np.mean(dp))}

    history: list[dict[str, Any]] = []
    _verdict_lock = threading.Lock()
    _verdict_thread: dict[str, Any] = {"t": None, "ep": None}
    _verdict_skipped = [0]
    # ---- BEST-d_seg checkpoint tracker (EMA non-negotiable + per-stage discipline). The rolling
    # "latest" + per-stage ckpts in _do_checkpoint can DRIFT PAST the best realized d_seg (tau
    # over-trains past its knee; l7/Muon oscillate on the plateau) -> the best EMA shadow would be
    # LOST (the gap that forced a manual ep725 snapshot worse than the ep700 best). Per-ARM scope
    # (each out_dir tracks its own best); the campaign compares arm-bests across arms.
    _best: dict[str, Any] = {"d_seg": float("inf"), "ep": None, "path": None}

    def _verdict_inflight() -> bool:
        t = _verdict_thread["t"]
        return t is not None and t.is_alive()

    def _emit_verdict_row(v: dict[str, float], ema_np: dict[str, np.ndarray], ep: int,
                          seg_form: str, ep_loss: float, *, async_tag: bool) -> None:
        blob = quantize_levelset_blob(ema_np)
        s = implied_score_from_verdict(v["d_seg"], v["d_pose"], blob["total_quantized_blob_bytes"])
        with _verdict_lock:
            row = {"stage": "verdict", "epoch": ep, "seg_form": seg_form,
                   **{k: round(vv, 6) for k, vv in v.items()},
                   "blob_bytes": blob["total_quantized_blob_bytes"], "implied_S": round(s, 4),
                   "ep_loss": round(ep_loss, 3),
                   # ADDITIVE telemetry: UTC emit wall-time so dashboards read verdict
                   # arrival times DIRECTLY (the no-timestamp root cause the self-calibrating
                   # dashboard otherwise self-observes). Purely observational; never read back
                   # into training/resume/parity, not appended to history/result.json.
                   "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
            if async_tag:
                row["async"] = True
            print(json.dumps(row), flush=True)
            history.append({"epoch": ep, **v, "implied_S": s})

    def _maybe_preserve_best(d_seg: float, ep: int, shadow_np_proj: dict[str, np.ndarray],
                             softmax_temp: float, hosc_beta: float) -> None:
        """Preserve the EMA SHADOW that achieved a NEW best realized-through-R d_seg, as a DEPLOY
        npz (shadow + cfg) -> byte-close-ready AND warm-startable (resume seeds live<-shadow).

        NO-FAKE: only a FINITE, strictly-better d_seg promotes the best (NaN/inf never wins). The
        ``shadow_np_proj`` is the SAME Stiefel-projected shadow the verdict measured (async: the
        point-in-time snapshot; sync: the current shadow) -> the preserved artifact is EXACTLY what
        produced the score (no drift). Atomic (tmp+os.replace). Thread-safe: holds _verdict_lock,
        and only one async verdict is in flight at a time, so best writes never race."""
        with _verdict_lock:
            if not _is_new_best(d_seg, _best["d_seg"]):  # finite + strictly-better only
                return
            prev = _best["d_seg"]
            ema_arrays = _build_ema_checkpoint_arrays(
                shadow_np_proj, args=args, softmax_temp=float(softmax_temp),
                render_h=render_h, render_w=render_w, epoch=int(ep), in_feat=in_feat,
                hosc_beta=float(hosc_beta))
            _atomic_savez(out_dir / "levelset_witness_ema_BEST.npz", ema_arrays)
            _atomic_write_json(out_dir / "levelset_best.json", {
                "d_seg": float(d_seg), "epoch": int(ep),
                "path": "levelset_witness_ema_BEST.npz",
                "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")})
            _best.update(d_seg=float(d_seg), ep=int(ep), path="levelset_witness_ema_BEST.npz")
            print(json.dumps({"stage": "checkpoint", "kind": "best", "epoch": int(ep),
                              "d_seg": round(float(d_seg), 6),
                              "prev_best": (round(prev, 6) if np.isfinite(prev) else None),
                              "path": "levelset_witness_ema_BEST.npz"}), flush=True)

    def _schedule_async_verdict(ep: int, seg_form: str, ep_loss: float) -> bool:
        if _verdict_inflight():
            _verdict_skipped[0] += 1
            with _verdict_lock:
                print(json.dumps({"stage": "verdict_skip", "epoch": ep,
                                  "inflight_epoch": _verdict_thread["ep"],
                                  "total_skipped": _verdict_skipped[0],
                                  "note": "prior async verdict still running; cadence self-throttles "
                                  "(GPU never blocks)"}), flush=True)
            return False
        snap = _capture_verdict_snapshot()  # MAIN thread, cheap, point-in-time
        _verdict_thread["ep"] = ep

        def _worker() -> None:
            t0 = time.time()
            try:
                v = _verdict_from_snapshot(snap)
                _emit_verdict_row(v, snap["ema_np"], ep, seg_form, ep_loss, async_tag=True)
                # HARDENING: preserve the best EMA shadow from the SAME snapshot the verdict scored
                # (snap["ema_np"] is the point-in-time Stiefel-projected shadow; cfg from the snap).
                _maybe_preserve_best(v["d_seg"], ep, snap["ema_np"],
                                     snap["softmax_temp"], snap["hosc_beta"])
                with _verdict_lock:
                    print(json.dumps({"stage": "verdict_async_done", "epoch": ep,
                                      "secs": round(time.time() - t0, 1)}), flush=True)
            except Exception as exc:  # an eval failure must NOT kill training (daemon thread).
                with _verdict_lock:
                    print(json.dumps({"stage": "verdict_async_failed", "epoch": ep,
                                      "err": f"{type(exc).__name__}: {exc}"}), flush=True)

        t = threading.Thread(target=_worker, name=f"async-verdict-ep{ep}", daemon=True)
        _verdict_thread["t"] = t
        t.start()
        return True

    def _join_async_verdict() -> None:
        t = _verdict_thread["t"]
        if t is not None and t.is_alive():
            print(json.dumps({"stage": "verdict_async_join",
                              "note": "waiting for in-flight async verdict before continuing"}), flush=True)
            t.join()
        _verdict_thread["t"] = None

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
        # (review Med1) the BYTE-CLOSE deploy npz ships the EMA shadow; re-project its film.weight onto
        # Stiefel so the shipped artifact is ON-MANIFOLD (PR(M)=PR(cov code) holds for what ships). The
        # RESUME sidecar keeps the UN-projected shadow (bit-faithful continuous resume). No-op unless
        # --film-stiefel (default OFF => byte-identical deploy + resume npz).
        deploy_shadow_np = _project_shadow_film_np(shadow_np)
        ema_arrays = _build_ema_checkpoint_arrays(
            deploy_shadow_np, args=args, softmax_temp=float(model.softmax_temp),
            render_h=render_h, render_w=render_w, epoch=epoch, in_feat=in_feat,
            hosc_beta=float(model.hosc_beta))  # FEED-fb: persist CURRENT annealed beta in deploy cfg
        resume_arrays = _build_resume_state_arrays(
            live_np, shadow_np, opt_np, args=args, epoch=epoch, in_feat=in_feat)
        # FEED-fm FIX-1: snapshot the loop's RNG streams (global MT19937 + LEVER-5 hardness PCG64)
        # INTO the resume sidecar so --resume-from is bit-faithful to a continuous run. hardness_rng
        # is a run_train local assigned before any _do_checkpoint call (closure ref; safe).
        resume_arrays.update(_rng_state_arrays(hardness_rng))
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
    resume_cfg: dict[str, Any] | None = None  # FEED-fm FIX-1: holds the sidecar cfg for the RNG
    # restore that must run AFTER hardness_rng is constructed (below); None => fresh start.
    if args.resume_from:
        from mlx.utils import tree_unflatten
        rp = _resolve_resume_path(Path(args.resume_from))
        rs = _load_resume_state(rp)
        resume_cfg = rs["cfg"]
        if not rs["live"]:
            raise ValueError(f"--resume-from {rp} has no live/param tensors (NO-FAKE: cannot resume).")
        # (review R2a-MED-1) FAIL-CLOSED arch-drift guard BEFORE model.update. MLX model.update only
        # writes params the model ALREADY has, so a resume whose ckpt carries trained params the
        # freshly-built model lacks (e.g. the run trained with --film-per-layer / --film-concat-code but
        # the resume command omitted it) would SILENTLY DROP those trained tensors -> a corrupted,
        # non-reproducible resume discovered only at exact-eval. Refuse loudly instead. The check is
        # arch-general (any missing key), not film-specific; the persisted __cfg_film_* flags name the
        # likely cause + fix. Per CLAUDE.md resumability + deterministic-reproducibility + NO-FAKE.
        _model_param_keys = {k for k, _ in tree_flatten(model.parameters())}
        _missing_in_model = sorted(set(rs["live"]) - _model_param_keys)
        if _missing_in_model:
            _ckpt_pl = bool(int(resume_cfg.get("__cfg_film_per_layer", 0) or 0))
            _ckpt_concat = bool(int(resume_cfg.get("__cfg_film_concat_code", 0) or 0))
            _hint = []
            if _ckpt_pl and not bool(getattr(args, "film_per_layer", False)):
                _hint.append("add --film-per-layer")
            if _ckpt_concat and not bool(getattr(args, "film_concat_code", False)):
                _hint.append("add --film-concat-code")
            raise ValueError(
                f"--resume-from {rp}: the checkpoint carries {len(_missing_in_model)} trained param(s) the "
                f"rebuilt model has NO slot for (first few: {_missing_in_model[:6]}) -> model.update would "
                "SILENTLY DROP them = a corrupted, non-reproducible resume. The resume command's ARCH flags "
                f"must MATCH the trained run. Ckpt arch flags: film_per_layer={_ckpt_pl}, "
                f"film_concat_code={_ckpt_concat}, film_stiefel="
                f"{bool(int(resume_cfg.get('__cfg_film_stiefel', 0) or 0))}. "
                + (f"Fix: {', '.join(_hint)}." if _hint else
                   "Rebuild the model with the SAME architecture the checkpoint was trained with."))
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
                      "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "axis": "[macOS-CPU advisory] NON-PROMOTABLE"}), flush=True)
    history.append({"epoch": start_epoch - 1, **v0, "implied_S": s0})

    if lane_w > 0.0:
        print(json.dumps({"stage": "lane_edge", "active": True, "weight": lane_w, "lane_class": lane_cls,
                          "margin_target": lane_tgt, "start_epoch": lane_start,
                          "note": "additive realized lane-class margin hinge (2nd seg forward when "
                          "active; default-off; engages at ep>=start_epoch with spike-guard re-treat)"}), flush=True)
    if msal_w > 0.0:
        print(json.dumps({"stage": "margin_saliency", "active": True, "weight": msal_w, "tau": msal_tau,
                          "target": msal_tgt, "start_epoch": msal_start, "uniward": msal_uni,
                          "uniward_beta": (msal_uni_beta if msal_uni else None),
                          "note": "LEVER-4 ALL-CLASS GT-margin-saliency-weighted realized margin hinge "
                          "(generalizes class-1 lane-edge to every inter-class edge; class-agnostic)"}), flush=True)

    # LEVER-5 (per-pair hardness) precompute: per-pair sampling probability for the oversampled extras.
    # Default --hardness-oversample 0.0 => n_extra 0 => order == permutation(P) => byte-identical.
    n_extra = int(round(P * max(args.hardness_oversample, 0.0)))
    hardness_prob = None
    hardness_rng = np.random.default_rng(int(args.seed) + 777)
    if n_extra > 0:
        if args.hardness_weighted and args.hardness_source == "realized":
            # one-time per-pair BASELINE realized d_seg over ALL pairs (frozen-decoder reconstruction
            # quality with init codes). CPU-torch authority path (no GPU contention with the daemon).
            ema_np0 = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
            deploy0 = int8_dequant_params(ema_np0)
            f1_all = [_render_numpy_deploy(deploy0, pi, 1) for pi in range(P)]
            ds_pp = np.asarray(cpu_verdict_d_seg_batch(seg_cpu, f1_all, [gt.lstars[pi] for pi in range(P)]),
                               dtype=np.float64).reshape(-1)
            h = ds_pp
            hsrc = "realized_per_pair_dseg"
        else:
            # $0 cached-GT hardness: per-pair fraction of flip-prone (small-GT-margin) pixels.
            band = float(args.hardness_band)
            h = np.asarray([float(np.mean(np.asarray(gt.margins[pi], np.float32) < band)) for pi in range(P)],
                           dtype=np.float64)
            hsrc = "margin_small_frac"
        h = np.clip(h, 1e-12, None) ** float(args.hardness_power)
        if not args.hardness_weighted:
            h = np.ones_like(h)  # uniform extras (the FAIR same-total-steps A/B baseline)
            hsrc = "uniform_oversample"
        hardness_prob = h / h.sum()
        print(json.dumps({"stage": "hardness", "oversample": float(args.hardness_oversample),
                          "n_extra_per_epoch": n_extra, "weighted": bool(args.hardness_weighted),
                          "source": hsrc, "power": float(args.hardness_power),
                          "hard_easy_spread": round(float(hardness_prob.max() / max(hardness_prob.min(), 1e-12)), 3),
                          "top_pairs": [int(i) for i in np.argsort(-hardness_prob)[:6]]}), flush=True)
    if args.max_bank_freq is not None:
        from tac.boundary_math.lever_b_levelset_generator import stem_nyquist_max_freq_cycles_per_unit
        nyq = stem_nyquist_max_freq_cycles_per_unit(scorer_w=SEG_W)
        print(json.dumps({"stage": "stem_nyquist", "max_bank_freq": float(args.max_bank_freq),
                          "stem_nyquist_cycles_per_unit": nyq, "curvelet_cols_after_cap": int(B.shape[1])}), flush=True)

    # FEED-fm FIX-1: RESTORE the RNG streams NOW -- after hardness_rng is built and the (RNG-free)
    # hardness precompute, before the FIRST epoch's permutation draw. Nothing between the resume
    # load and here advances the global MT19937 or hardness_rng (verdict/precompute are RNG-free), so
    # the next permutation/choice continues the CONTINUOUS stream bit-for-bit. DEFAULT-SAFE: no
    # resume, or a pre-FEED-fm sidecar without __rng_* keys => fresh-seeded RNGs untouched.
    if resume_cfg is not None:
        _rng_restored = _restore_rng_state(resume_cfg, hardness_rng)
        print(json.dumps({"stage": "resume_rng", "np_global_restored": _rng_restored["np_global"],
                          "hardness_restored": _rng_restored["hardness"],
                          "note": ("bit-faithful RNG resume" if _rng_restored["np_global"] else
                                   "pre-FEED-fm sidecar (no RNG state); fresh-seeded RNGs (back-compat)")}),
              flush=True)

    recent_losses: list[float] = []
    last_ep = start_epoch - 1
    stage_ckpts: list[dict[str, Any]] = []
    # CURRICULUM stage-transition spike-guard re-treat tracker (operator 2026-06-26 "different
    # stages need different treatment ... transitions must re-treat"). Init to the START epoch's
    # seg_form so a fresh-start / resume does NOT spuriously re-treat (prev == current at ep0).
    prev_seg_form = _seg_form_for_epoch(start_epoch, args)
    # MUON FINISHER (FEED-fi) per-stage optimizer switch state. muon_start_epoch None (default) =>
    # muon_switched stays False forever => the switch block + tag suffix never fire => BIT-IDENTICAL
    # to the pre-FEED-fi AdamW-throughout path. Effective LRs default to 0.1*lr (PR95 ~0.1x finetune).
    muon_switched = False
    # BUILD 1 (FEED-fw): stage-transition treatment tracker. None until a registered AdamW->AdamW
    # boundary fires (curriculum seg-form change / lane-edge engage / margin-saliency engage); the LR
    # re-warmup + (optional) AdamW moment reset key off it. DEFAULT-OFF flags
    # (--stage-transition-rewarmup-epochs 0 + no --stage-transition-reset-moments) => this is set but
    # never consumed => BIT-IDENTICAL. NOT persisted across resume (re-derived; None at resume start
    # => no spurious re-warmup until a real boundary).
    last_boundary_epoch: "int | None" = None
    # (review C2) anneal SCHEDULE length: --anneal-epochs decouples the cosine denominator (the
    # schedule the temp/LR were designed against) from --epochs (this run's length). Default None =>
    # args.epochs => the LR cosine below is BIT-IDENTICAL. A warm-start arm sets it to the ORIGINAL
    # schedule (e.g. 1500) so resuming the CE ckpt @ ep299 reproduces the DISEASE regime, not the tail.
    anneal_epochs = int(args.anneal_epochs) if getattr(args, "anneal_epochs", None) else int(args.epochs)
    muon_lr_eff = float(args.muon_lr) if args.muon_lr is not None else 0.1 * float(args.lr)
    muon_adamw_lr_eff = float(args.muon_adamw_lr) if args.muon_adamw_lr is not None else 0.1 * float(args.lr)
    muon_wd_eff = float(args.muon_weight_decay) if args.muon_weight_decay is not None else float(args.weight_decay)
    with temporary_mlx_device(args.mlx_device):
        for ep in range(start_epoch, args.epochs + 1):
            seg_form = _seg_form_for_epoch(ep, args)
            # BUILD 1 (FEED-fw): detect an AdamW->AdamW stage boundary at THIS epoch BEFORE the
            # existing transition blocks mutate prev_seg_form / lane_gate / msal_gate. Consumed below
            # (after the Muon block, so muon_switched is current) to register the LR re-warmup anchor
            # + optionally reset the AdamW moments. The Muon switch is intentionally EXCLUDED (it
            # already re-treats with a fresh optimizer per FEED-fi, and the base LR schedule is frozen
            # during the finisher). DEFAULT-OFF flags => these booleans are computed but never
            # consumed => BIT-IDENTICAL (pure-python reads, no MLX/model touch).
            _bnd_curriculum = (seg_form != prev_seg_form)
            _bnd_lane = (lane_w > 0.0 and (ep >= lane_start) and not lane_gate["on"])
            _bnd_msal = (msal_w > 0.0 and (ep >= msal_start) and not msal_gate["on"])
            # (review R3-M1) LEVER-B thin-lane engagement is ALSO an AdamW->AdamW treatment boundary
            # (mirrors _bnd_lane/_bnd_msal). Default lane_thin_w=0.0 => never fires => bit-identical.
            _bnd_lane_thin = (lane_thin_w > 0.0 and (ep >= lane_thin_start) and not lane_thin_gate["on"])
            _stage_boundary_now = _bnd_curriculum or _bnd_lane or _bnd_msal or _bnd_lane_thin
            # CURRICULUM stage-transition RE-TREAT (operator 2026-06-26 "transitions must re-treat";
            # PR95-8-stage generalized). The seg LOSS FORM change (ce -> tau_softplus -> l7_softplus)
            # is a per-stage treatment boundary; clear the spike-guard running median so the new
            # stage's loss scale is NOT judged against the prior stage's median (the named "stage
            # inheriting base-stage treatment" failure). The l7 weight is mean-1-renormalized so the
            # scale jump is small in THIS loss design, but the discipline is binding regardless of
            # carrier. Additive: non-curriculum runs have a constant seg_form => prev == current =>
            # NEVER clears => byte-identical. Non-finite guards are unaffected (still always armed).
            if seg_form != prev_seg_form:
                recent_losses.clear()
                print(json.dumps({"stage": "curriculum_transition", "epoch": ep,
                                  "from_seg_form": prev_seg_form, "to_seg_form": seg_form,
                                  "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
                prev_seg_form = seg_form
            # MUON FINISHER switch (FEED-fi; PR95 stage-8). Fires once at the first epoch >= the
            # start (the >= handles RESUME into the finisher too). DEFAULT-OFF (start is None) =>
            # never fires => byte-identical. The switch is a per-stage TREATMENT boundary (operator
            # 2026-06-26 "transitions must re-treat"): rebuild opt AdamW->MultiOptimizer(Muon 2D
            # weights + AdamW rest), re-init optimizer state, CLEAR the spike-guard (the orthogonalized
            # lower-lr step has a different loss scale; do NOT judge it against the prior AdamW stage's
            # median), and SAVE a PRESERVED stage-encoded ckpt so the Muon-finished decoder is
            # independently byte-closeable + resumable. The Muon momentum re-warms from scratch here
            # (best-effort, like the resume path); the DECODER weights are unchanged at the switch.
            if (args.muon_start_epoch is not None) and (not muon_switched) and (ep >= args.muon_start_epoch):
                n_muon, n_adamw = count_muon_adamw_split(model.trainable_parameters())
                opt = build_muon_finisher_optimizer(
                    muon_lr=muon_lr_eff, muon_adamw_lr=muon_adamw_lr_eff,
                    muon_momentum=float(args.muon_momentum), muon_weight_decay=muon_wd_eff,
                    muon_ns_steps=int(args.muon_ns_steps), adamw_weight_decay=float(args.weight_decay),
                )
                opt.init(model.trainable_parameters())
                mx.eval(opt.state)
                muon_switched = True
                recent_losses.clear()
                print(json.dumps({"stage": "muon_finisher_switch", "epoch": ep,
                                  "muon_start_epoch": int(args.muon_start_epoch), "muon_lr": muon_lr_eff,
                                  "muon_adamw_lr": muon_adamw_lr_eff, "muon_momentum": float(args.muon_momentum),
                                  "muon_ns_steps": int(args.muon_ns_steps), "muon_weight_decay": muon_wd_eff,
                                  "n_muon_params": n_muon, "n_adamw_params": n_adamw,
                                  "note": "AdamW->Muon (2D hidden weights; biases/code/heads stay AdamW); "
                                  "spike-guard re-treated; LR schedule frozen for the finisher"}), flush=True)
                if args.stage_checkpoints:
                    _wm = _do_checkpoint(ep, stage_tag="stageMuonStart")
                    stage_ckpts.append(_wm)
                    print(json.dumps({"stage": "checkpoint", "kind": "muon_finisher_start", **_wm}), flush=True)
            # lane-edge engagement gate + transition RE-TREAT (spike-guard reset at the engage epoch
            # so the added margin-hinge term's loss jump is not silently spike-skipped; no-op when
            # lane_start<=1 i.e. the default always-on-from-ep1 path -> zero behavior change).
            if lane_w > 0.0:
                _was_on = lane_gate["on"]
                lane_gate["on"] = lever_gate_on_at_epoch(lane_w, lane_start, ep)
                if lane_gate["on"] and not _was_on:
                    recent_losses.clear()
                    print(json.dumps({"stage": "lane_edge_engage", "epoch": ep, "lane_start": lane_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # LEVER-4 margin-saliency engagement gate + transition RE-TREAT (same discipline as lane).
            if msal_w > 0.0:
                _msal_was = msal_gate["on"]
                msal_gate["on"] = lever_gate_on_at_epoch(msal_w, msal_start, ep)
                if msal_gate["on"] and not _msal_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "margin_saliency_engage", "epoch": ep, "start": msal_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # LEVER-B thin-lane engagement gate + transition RE-TREAT (review R3-M1: the gate was
            # initialized at :lane_thin_gate but NEVER flipped, so --lane-thin-start-epoch > 1 left the
            # gate stuck OFF => the loss branch at `lane_thin_gate["on"]` never fired => a SILENT NO-OP
            # = a FALSE 'thin-lane prior does nothing' verdict). Mirrors the lane/margin-saliency gates.
            # No-op when lane_thin_start<=1 (default-on-from-ep1) => zero behavior change.
            if lane_thin_w > 0.0:
                _lt_was = lane_thin_gate["on"]
                lane_thin_gate["on"] = lever_gate_on_at_epoch(lane_thin_w, lane_thin_start, ep)
                if lane_thin_gate["on"] and not _lt_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "lane_thin_engage", "epoch": ep, "start": lane_thin_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # BUILD 1 (FEED-fw): apply stage-transition TREATMENT for an AdamW->AdamW boundary
            # detected above. Skipped during the Muon finisher (muon_switched True; it re-treats
            # itself + freezes the base LR schedule). The spike-guard re-treat already happened in the
            # blocks above (recent_losses cleared); this adds (1) the LR re-warmup anchor and (2) an
            # OPTIONAL fresh-AdamW moment reset. DEFAULT-OFF: --stage-transition-reset-moments False
            # AND --stage-transition-rewarmup-epochs 0 => only sets last_boundary_epoch (then unused
            # by the gated factor) => BIT-IDENTICAL. The fresh AdamW preserves the current
            # learning_rate; the LR-schedule block below resets it for the epoch anyway. (MLX
            # Optimizer.init only fills MISSING state, so a TRUE moment reset requires a fresh
            # optimizer object -- exactly how the Muon switch resets, FEED-fi.)
            if _stage_boundary_now and not muon_switched:
                last_boundary_epoch = ep
                if args.stage_transition_reset_moments:
                    opt = optim.AdamW(learning_rate=float(opt.learning_rate),
                                      weight_decay=args.weight_decay)
                    opt.init(model.trainable_parameters())
                    mx.eval(opt.state)
                    print(json.dumps({"stage": "stage_transition_reset_moments", "epoch": ep,
                                      "from_curriculum": bool(_bnd_curriculum),
                                      "from_lane_engage": bool(_bnd_lane),
                                      "from_margin_saliency_engage": bool(_bnd_msal),
                                      "from_lane_thin_engage": bool(_bnd_lane_thin),
                                      "note": "AdamW m/v zeroed (fresh optimizer); spike-guard already "
                                      "re-treated; stale-momentum-through-landscape-change avoided"}),
                          flush=True)
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
            # FEED-fm FIX-2: FREEZE softmax_temp AND hosc_beta DURING THE MUON FINISHER. At/after the
            # switch (muon_switched True) hold BOTH at their muon-START value -- i.e. the value at
            # epoch == muon_start_epoch (deterministic in muon_start_epoch, NOT the process-local fire
            # epoch, so RESUME-into-finisher reproduces the same frozen target). This mirrors the LR
            # freeze already gated on `not muon_switched` below: the orthogonalized finisher conditions
            # boundary PLACEMENT against a STATIONARY target (clean Eikonal=slope / Muon=placement
            # attribution per FEED-fk). DEFAULT-SAFE: --muon-start-epoch None => muon_switched is
            # always False => _anneal_ep == ep => the _softmax_temp_for_epoch / _hosc_beta_for_epoch
            # calls reproduce the pre-FEED-fm inline formulas exactly => BIT-IDENTICAL.
            _anneal_ep = int(args.muon_start_epoch) if muon_switched else ep
            model.softmax_temp = _softmax_temp_for_epoch(_anneal_ep, args)
            # (FEED-fb) ANNEAL hosc_beta start->end (the step-native L-infinity-optimal lever;
            # beta->inf = step-native tanh(beta*sin)). The model's _act reads self.hosc_beta FRESH
            # each forward, so mutating model.hosc_beta per epoch retunes the activation (exactly how
            # softmax_temp is annealed above). DEFAULT-SAFE: _hosc_beta_for_epoch returns None when
            # --hosc-beta-end is unset (or == --hosc-beta, or activation != hosc) -> model.hosc_beta
            # is NEVER touched => stays at its construction value (== args.hosc_beta) every epoch =>
            # BIT-IDENTICAL to the pre-FEED-fb path (and the finisher freeze is then a no-op too). The
            # verdict/checkpoint/byte-close forwards read float(model.hosc_beta) so realized d_seg is
            # measured (and deploy cfg saved) at the CURRENT beta (NO-FAKE).
            _beta = _hosc_beta_for_epoch(_anneal_ep, args)
            if _beta is not None:
                model.hosc_beta = _beta
            # LR warmup->cosine. Gated OFF once the Muon finisher is active (operator 2026-06-26
            # "different stages need different treatment"): the finisher is a PR95 flat low-LR
            # polish at its own muon_lr/muon_adamw_lr, NOT the base cosine, and the MultiOptimizer's
            # children own their own LRs (setting opt.learning_rate would not reach them). Default
            # (no --muon-start-epoch) => muon_switched False => identical to before (BIT-IDENTICAL).
            if args.lr_schedule and not muon_switched:
                if ep <= args.warmup_epochs:
                    lr = args.lr * ep / max(args.warmup_epochs, 1)
                else:
                    # (review C2) cosine denominator = anneal_epochs (schedule length), NOT args.epochs
                    # (run length). anneal_epochs defaults to args.epochs => BIT-IDENTICAL; a warm-start
                    # arm sets --anneal-epochs to the ORIGINAL schedule so the post-resume LR matches the
                    # disease regime (~0.9*peak at ep300/1500) instead of the run-length tail.
                    prog = (ep - args.warmup_epochs) / max(anneal_epochs - args.warmup_epochs, 1)
                    lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * prog))
                # BUILD 1 (FEED-fw): stage-transition LR re-warmup. DEFAULT-OFF
                # (--stage-transition-rewarmup-epochs 0) => _rw is EXACTLY 1.0 => lr*1.0 == lr =>
                # BIT-IDENTICAL. After a registered AdamW->AdamW boundary, ramp the scheduled LR up
                # from the floor over N epochs so the post-boundary landscape change is not hit at
                # full LR with (possibly reset) momentum (the FEED-ft#3 tau-jump root cause).
                _rw = _stage_rewarmup_factor(
                    ep, last_boundary_epoch, args.stage_transition_rewarmup_epochs,
                    args.stage_transition_rewarmup_floor, args.stage_transition_rewarmup_shape)
                lr = lr * _rw
                opt.learning_rate = float(lr)
            # LEVER-5: base permutation (every pair >=1 step, never starved) + hardness-allocated
            # extras. n_extra=0 (default) => order == permutation(P) => byte-identical to before.
            order = np.random.permutation(P)
            if n_extra > 0 and hardness_prob is not None:
                extra = hardness_rng.choice(P, size=n_extra, replace=True, p=hardness_prob)
                order = np.random.permutation(np.concatenate([order, extra]))
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
                # DM1a (Stiefel-W): project the LIVE film.weight onto orthonormal columns AFTER the
                # optimizer step, so PR(M)=PR(cov(code)) holds (to the projection's ~1e-2 residual) for
                # the LIVE weight.
                # Default-off (--film-stiefel) => skipped => byte-identical. The cubic Newton-Schulz
                # polar re-normalizes columns, which also neutralizes the global-magnitude component of
                # AdamW weight-decay on W (the design's WD=0-on-W intent). NOTE: composes with the Muon
                # finisher (the projection runs whichever optimizer produced the step).
                #   (review Med1) The EMA update below averages the (per-step on-manifold) LIVE weight
                #   into the shadow; an arithmetic EMA of orthonormal matrices is NOT itself orthonormal,
                #   so the DEPLOYED shadow drifts OFF-Stiefel. The shipped artifact is re-projected at
                #   verdict + byte-close via _project_shadow_film_np (NOT here -- mutating the shadow
                #   in place would break resume bit-faithfulness). This comment formerly claimed "the
                #   deploy shadow tracks the on-manifold weight" -- FALSE; corrected.
                if args.film_stiefel:
                    model.film.weight = stiefel_project_columns(model.film.weight)
                    mx.eval(model.film.weight)
                ema.update(model)
                mx.eval(list(ema.shadow.values()))
                recent_losses.append(batch_loss)
                if len(recent_losses) > 50:
                    recent_losses.pop(0)
                ep_loss += batch_loss
            if args.mlx_device == "gpu":
                mx.clear_cache()
            if ep % args.eval_every == 0 or ep == args.epochs:
                if args.async_verdict:
                    # FEED-em: offload the observational verdict to a background thread so the
                    # GPU loop never idles. BIT-IDENTICAL training (verdict is never read back).
                    # At the FINAL epoch, JOIN first so the last verdict row is not skip-throttled.
                    if ep == args.epochs:
                        _join_async_verdict()
                    _schedule_async_verdict(ep, seg_form, ep_loss)
                else:
                    v = realized_verdict()
                    blob = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()})
                    s = implied_score_from_verdict(v["d_seg"], v["d_pose"], blob["total_quantized_blob_bytes"])
                    print(json.dumps({"stage": "verdict", "epoch": ep, "seg_form": seg_form,
                                      **{k: round(vv, 6) for k, vv in v.items()},
                                      "blob_bytes": blob["total_quantized_blob_bytes"], "implied_S": round(s, 4),
                                      "ep_loss": round(ep_loss, 3),
                                      "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}), flush=True)
                    history.append({"epoch": ep, **v, "implied_S": s})
                    # HARDENING: preserve the best EMA shadow (sync path = current shadow IS what
                    # realized_verdict just scored; project film.weight on-manifold like the verdict).
                    _maybe_preserve_best(
                        v["d_seg"], ep,
                        _project_shadow_film_np({k: np.asarray(vv, np.float32)
                                                 for k, vv in ema.shadow.items()}),
                        float(model.softmax_temp), float(model.hosc_beta))
            # DM1 telemetry (decisive-smoke signals; design memo §6 firewall). At eval cadence, log
            # PR(M) (per-pair FiLM modulation participation ratio), PR(cov(code)) and the Stiefel
            # residual ‖WᵀW−I‖_F so the A/B can SEPARATE "means fixed" (PR held >~3.0) from "end moved"
            # (advisory d_seg, in the verdict row above).
            #   (review C1) GATE WIDENED to include --dm1-telemetry so the A0 BASELINE (no DM1 lever)
            #     also logs the row -- otherwise the "baseline collapses" half of the firewall is
            #     UNMEASURABLE. Default-off (all three off) => never fires => bit-identical observability.
            #   (review Med1) The DEPLOYED weight is the EMA SHADOW, not live. An arithmetic EMA of
            #     orthonormal matrices is NOT orthonormal => the shadow drifts off-Stiefel. The firewall
            #     must read what SHIPS, so report BOTH the LIVE and the SHADOW PR(M)+residual (shadow
            #     modulation M_shadow = code @ W_shadowᵀ + b_shadow, ISOLATING the W drift on the same
            #     codes). Pure read (no model/grad touch).
            if (args.film_stiefel or code_spec_w > 0.0 or args.dm1_telemetry) and (ep % args.eval_every == 0 or ep == args.epochs):
                _S = min(2 * P, 256)
                _ssub = np.arange(0, 2 * P, max(1, (2 * P) // _S))[:_S].astype(np.int32)
                _codes = model.code[mx.array(_ssub)]
                _M = model.film(_codes)                                # (S, 2*H*L) LIVE modulation
                _pr_m = float(film_modulation_participation_ratio(np.asarray(_M, np.float32)))
                _pr_c = float(film_modulation_participation_ratio(np.asarray(_codes, np.float32)))
                _sres = stiefel_residual(model.film.weight) if args.film_stiefel else None
                # Med1: the SHADOW (deployed) film.weight modulation + its Stiefel residual.
                _Ws = ema.shadow.get("film.weight")
                _bs = ema.shadow.get("film.bias")
                _pr_m_shadow = None
                _sres_shadow = None
                if _Ws is not None:
                    _M_shadow = _codes @ _Ws.T
                    if _bs is not None:
                        _M_shadow = _M_shadow + _bs
                    _pr_m_shadow = float(film_modulation_participation_ratio(np.asarray(_M_shadow, np.float32)))
                    _sres_shadow = stiefel_residual(_Ws) if args.film_stiefel else None
                print(json.dumps({"stage": "dm1_telemetry", "epoch": ep, "seg_form": seg_form,
                                  "pr_film_M": round(_pr_m, 4), "pr_cov_code": round(_pr_c, 4),
                                  "stiefel_residual": (round(_sres, 5) if _sres is not None else None),
                                  "pr_film_M_shadow": (round(_pr_m_shadow, 4) if _pr_m_shadow is not None else None),
                                  "stiefel_residual_shadow": (round(_sres_shadow, 5) if _sres_shadow is not None else None),
                                  "film_stiefel": bool(args.film_stiefel),
                                  "code_spec_w": code_spec_w}), flush=True)
            # ---- CHECKPOINTING (FEED-dz; mandatory per operator "never launch non-resumable / save
            # per-stage" rule). PER-STAGE: at every curriculum-stage TRANSITION save a PRESERVED,
            # stage-encoded, byte-close-loadable ckpt (per-stage A/B of which stage moves d_seg).
            # INTRA-STAGE: every --ckpt-every epochs save the rolling latest (crash-resume window).
            is_transition = (
                args.stage_checkpoints and ep < args.epochs
                and _seg_form_for_epoch(ep + 1, args) != seg_form)
            do_periodic = args.ckpt_every > 0 and ep % args.ckpt_every == 0
            if is_transition:
                # FEED-fi: tag the preserved ckpt with the optimizer phase too, so a curriculum
                # transition DURING the Muon finisher is distinctly byte-closeable (suffix "" when
                # the finisher is off => identical filename to the pre-FEED-fi path).
                w = _do_checkpoint(ep, stage_tag=_stage_tag(seg_form) + ("_muon" if muon_switched else ""))
                stage_ckpts.append(w)
                print(json.dumps({"stage": "checkpoint", "kind": "stage_transition", **w}), flush=True)
            elif do_periodic:
                w = _do_checkpoint(ep)
                print(json.dumps({"stage": "checkpoint", "kind": "intra_stage", **w}), flush=True)
            last_ep = ep

    # FEED-em: JOIN any in-flight async verdict so the final verdict row + history land BEFORE
    # result.json is written (the DONE-marker contract). No-op when --async-verdict is off.
    if args.async_verdict:
        _join_async_verdict()

    # FINAL checkpoint (replaces the historical loop-end-only save, which is now FORBIDDEN). Always
    # writes the rolling latest + a PRESERVED final stage-encoded ckpt -> the run is byte-closeable
    # and resumable from disk at completion. Saves the EMA SHADOW (deploy), NOT live (EMA rule).
    final_form = _seg_form_for_epoch(last_ep, args) if last_ep >= 1 else args.seg_loss
    # FEED-fi: the FINAL ckpt is the Muon-finished decoder when the finisher ran -> tag it "_muon"
    # so it is distinctly byte-closeable (suffix "" when off => identical to the pre-FEED-fi path).
    _final_tag = (_stage_tag(final_form) + ("_muon" if muon_switched else "")) if args.stage_checkpoints else None
    final = _do_checkpoint(last_ep, stage_tag=_final_tag)
    stage_ckpts.append({**final, "kind": "final"})
    ck = out_dir / "levelset_witness_ema_mlx.npz"
    print(json.dumps({"stage": "checkpoint", "kind": "final", **final}), flush=True)
    result = {
        "utc": _utc(), "n_pairs": P, "epochs": args.epochs, "final_epoch": last_ep,
        "render_hw": [render_h, render_w],
        "front_end": "curvelet" + ("+self_orient" if use_self_orient else ""),
        "activation": args.activation, "in_feat": int(in_feat),
        "history": history, "checkpoint": str(ck), "stage_checkpoints": stage_ckpts,
        # HARDENING: the BEST realized-d_seg EMA-shadow ckpt (None if no finite verdict landed).
        # The harvester / next-arm warm-start reads this (or levelset_best.json) instead of the
        # rolling "latest", which can have drifted past the best.
        "best": (dict(_best) if _best["ep"] is not None else None),
        "resumable": True, "ckpt_every": int(args.ckpt_every),
        # (review C2) anneal schedule length (deterministic-reproducibility provenance). None default =>
        # records the resolved value (== epochs) so a reader knows the exact cosine denominator used.
        "anneal_epochs": int(anneal_epochs),
        # (review C1/Med1) DM1 telemetry + shadow-projection provenance (all default-OFF paths recorded).
        "dm1_telemetry": bool(getattr(args, "dm1_telemetry", False)),
        "film_stiefel": bool(getattr(args, "film_stiefel", False)),
        "code_spectral_entropy_weight": float(getattr(args, "code_spectral_entropy_weight", 0.0)),
        # BUILD 1/2 (FEED-fw) provenance (deterministic-reproducibility: record config with the
        # result). All default-OFF => these reflect the bit-identical path.
        "stage_transition_rewarmup_epochs": int(getattr(args, "stage_transition_rewarmup_epochs", 0)),
        "stage_transition_rewarmup_floor": float(getattr(args, "stage_transition_rewarmup_floor", 0.1)),
        "stage_transition_rewarmup_shape": str(getattr(args, "stage_transition_rewarmup_shape", "linear")),
        "stage_transition_reset_moments": bool(getattr(args, "stage_transition_reset_moments", False)),
        "lane_prior_phi1": bool(getattr(args, "lane_prior_phi1", False)),
        "lane_prior_phi1_mode": str(getattr(args, "lane_prior_phi1_mode", "replace")),
        # LEVER-A / LEVER-B provenance (deterministic-reproducibility; all default-OFF => the
        # bit-identical path is recorded as off).
        "film_per_layer": bool(getattr(args, "film_per_layer", False)),
        "film_concat_code": bool(getattr(args, "film_concat_code", False)),
        "film_rank_floor_weight": float(getattr(args, "film_rank_floor_weight", 0.0)),
        "film_rank_floor_target": float(getattr(args, "film_rank_floor_target", 4.0)),
        "lane_thin_weight": float(getattr(args, "lane_thin_weight", 0.0)),
        "lane_thin_class": int(getattr(args, "lane_thin_class", 1)),
        "lane_thin_radius": int(getattr(args, "lane_thin_radius", 4)),
        "lane_thin_target": float(getattr(args, "lane_thin_target", 0.5)),
        "lane_thin_start_epoch": int(getattr(args, "lane_thin_start_epoch", 0)),
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
    ap.add_argument("--anneal-epochs", type=int, default=None,
                    help="(review C2) SCHEDULE length for the softmax-temp + hosc-beta + LR cosine anneals "
                    "(the cosine DENOMINATOR), decoupled from --epochs (the RUN length). None (default) "
                    "=> use --epochs => BIT-IDENTICAL. A WARM-START arm (e.g. --resume-from a CE ckpt @ "
                    "ep299, --epochs 399) MUST set this to the ORIGINAL schedule length (e.g. 1500) so "
                    "ep300->400 reproduces the DISEASE regime (temp ~0.91->0.84, LR ~0.9*peak) the lever "
                    "must be tested in -- NOT the schedule tail (temp ~0.19->0.05, LR ~0.15*peak).")
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
    ap.add_argument("--freeze-decoder-fit-codes", type=str, default=None,
                    help="FEED-eo AMORTIZATION (days->hours): load the SHARED decoder from this level-set "
                    "EMA/deploy npz (trained on a SUBSET, e.g. n96/n192), FREEZE it, and fit ONLY the "
                    "per-pair codes for all --num-pairs pairs (embarrassingly-parallel per-pair latent "
                    "fit through the frozen render+R+scorer). The front-end config (--bank-*/--max-bank-"
                    "freq/--self-orient/--n-dir-freqs) MUST match the decoder's in_feat. Incompatible "
                    "with --resume-from/--structured-init. DEFAULT None = normal joint train.")
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
    ap.add_argument("--async-verdict", action=argparse.BooleanOptionalAction, default=False,
                    help="FEED-em: run the OBSERVATIONAL CPU-torch verdict in a BACKGROUND THREAD off a "
                    "point-in-time snapshot so the MLX-GPU loop never idles (~4.7%% wall-clock reclaim @ "
                    "n600). BIT-IDENTICAL training (the verdict is never read back); only the verdict "
                    "CADENCE may self-throttle under load (at-most-one in-flight). DEFAULT OFF = the "
                    "current synchronous bit-identical behavior.")
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
    ap.add_argument("--gpu-reorient", action=argparse.BooleanOptionalAction, default=False,
                    help="FEED-eo: compute the per-pair reorient argmax on MLX-GPU (fp32 twin forward) "
                    "instead of the 600 GPU-idle numpy CPU forwards (~6.2%% wall-clock reclaim @ n600). "
                    "PARITY-GATED (fp32-GPU vs fp64-numpy argmax differs at boundary px): adopt only "
                    "after experiments/probe_levelset_gpu_reorient_parity.py shows cos>0.999 + negligible "
                    "d_seg A/B. DEFAULT OFF = the bit-faithful numpy reorient (current behavior).")
    ap.add_argument("--freq-across", type=float, default=32.0, help="self-orient: HIGH freq across the edge (normal).")
    ap.add_argument("--freq-along", type=float, default=4.0, help="self-orient: LOW freq along the edge (tangent).")
    # ACTIVATION
    # (config-review #3) HOSC is the ONLY descent evidence (probe 0.0066; A/B 0.221 hosc vs 0.265
    # wire). WIRE was a paper-default guess; default HOSC, run wire as a sweep arm.
    ap.add_argument("--activation", choices=["wire", "hosc", "relu"], default="hosc")
    ap.add_argument("--wire-w0", type=float, default=20.0)
    ap.add_argument("--wire-s0", type=float, default=10.0)
    ap.add_argument("--hosc-beta", type=float, default=4.0)
    # (FEED-fb) BETA-ANNEAL: the named UNSWEPT step-native L-infinity-optimal lever. hosc is
    # tanh(beta*sin(omega*u)); beta->inf => STEP-native (the topology-matched chart for the
    # piecewise-constant argmax target, no Gibbs). --hosc-beta-end is the anneal TARGET; when it is
    # None (default) OR == --hosc-beta, NO anneal occurs and beta stays CONSTANT every epoch =>
    # BIT-IDENTICAL to the pre-FEED-fb path. The optimal-form decoder build sharpens beta start->end
    # (e.g. --hosc-beta 4 --hosc-beta-end 8) so the activation step-sharpens as the SDF partition
    # pins (sister of the softmax-temp anneal at the top of the epoch loop).
    ap.add_argument("--hosc-beta-end", type=float, default=None,
                    help="hosc beta anneal TARGET (None => no anneal, beta constant at --hosc-beta => bit-identical).")
    ap.add_argument("--hosc-beta-anneal", choices=["linear", "cosine"], default="linear",
                    help="hosc beta anneal schedule start->end (only used when --hosc-beta-end is set).")
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
    # realized seg forward (acceptable per operator "score > training time"). SegNet class order is the
    # comma10k CANONICAL order (MEASURED 2026-06-27; CLAUDE.md NON-NEGOTIABLE): [Road0, Lane1,
    # Undrivable2, Movable3, MyCar4]. The luma-sort [Road0,Lane1,MyCar2,Undriv3,Movable4] is FORBIDDEN/
    # WRONG for 2/3/4. Class 0=Road & 1=Lane CONFIRMED in both (the lever uses only class 1). LEVER-4
    # (class-agnostic margin-saliency) is PREFERRED as it sidesteps the class index entirely.
    ap.add_argument("--lane-edge-weight", type=float, default=0.0,
                    help="LEVER-3: weight on the additive realized lane-class margin hinge (0=off).")
    ap.add_argument("--lane-edge-class", type=int, default=1,
                    help="LEVER-3: GT class index to up-weight (1=Lane, CONFIRMED; comma10k CANONICAL "
                    "order [Road0,Lane1,Undrivable2,Movable3,MyCar4] for 2/3/4 -- NOT the forbidden luma-sort).")
    ap.add_argument("--lane-margin-target", type=float, default=0.5,
                    help="LEVER-3: target decision margin for the lane hinge relu(target - margin).")
    ap.add_argument("--lane-edge-start-epoch", type=int, default=0,
                    help="LEVER-3 OPTIMAL-FORM: engage the lane hinge only at ep>=this (0=from ep1=current "
                    "behavior). Gate to the tau_softplus/l7 margin stage (e.g. 300) to avoid the "
                    "margin-from-scratch-starves-interior failure; the engage epoch re-treats the spike-guard.")
    # LEVER-A (FiLM-RANK-FIX, ADDITIVE, ALL DEFAULT-OFF). Attacks the MEASURED per-pair FiLM modulation
    # participation-ratio collapse (3.34@CE -> 1.27@tau -> 1.19@l7: 91.8% of per-pair variation in ONE
    # axis -> the decoder receives ~1 effective per-pair direction -> caps d_seg AND held-out
    # amortization). All-off => byte-identical to the pre-LEVER-A witness (the extra submodules / loss
    # term are not created). See build_levelset_rgb_witness + the rank-floor branch in total_loss_fn.
    ap.add_argument("--film-per-layer", action="store_true",
                    help="LEVER-A1 [CAPACITY, NOT rank -- review M2/FEED-ht]: add SEPARATE per-layer "
                    "RESIDUAL FiLM projections (identity at init). +~25k params (~+0.01 rate). MEASURED "
                    "(M2): does NOT raise modulation rank -- A1/A2/shared-FiLM are all functions of the "
                    "SAME mod_dim code, so PR(M) <= rank(codes) <= mod_dim regardless of capacity. The "
                    "byte-FREE rank lever is --film-stiefel (+ --code-spectral-entropy-weight): PR(M) "
                    "1.19->4.57 at 0 added bytes. Prefer those. Default OFF = shared-FiLM-only.")
    ap.add_argument("--film-concat-code", action="store_true",
                    help="LEVER-A2 [CAPACITY, NOT rank -- review M2/FEED-ht]: add an ADDITIVE per-pair "
                    "code-injection route (folded concat; identity at init). +~12k params. Same mod_dim "
                    "rank ceiling as A1 (cannot raise PR(M) above rank(codes)); use --film-stiefel for "
                    "the byte-free rank fix. Default OFF.")
    ap.add_argument("--film-rank-floor-weight", type=float, default=0.0,
                    help="LEVER-A3 [DOMINATED by --film-stiefel; NOT recommended -- review FEED-ht/M1]: "
                    "weight of a SOFT participation-ratio FLOOR penalty relu(target-PR) on M=film(code). "
                    "0.0 (default) = OFF. CAVEAT (review M1): the PR measure is 0-homogeneous so its grad "
                    "~1/||M|| can blow up at small codes (no warm-in/start-gate here) and proxy-games "
                    "low-gain directions. Prefer the byte-free --film-stiefel (+ --code-spectral-entropy-"
                    "weight), which makes PR(M)=PR(cov(code)) hold by construction. Kept for ablation only.")
    ap.add_argument("--film-rank-floor-target", type=float, default=4.0,
                    help="LEVER-A3: the participation-ratio FLOOR (effective-dim target) the penalty pushes "
                    "M toward (must be > 1 when --film-rank-floor-weight > 0; PR >= 1 always). Default 4.0.")
    # DM1 minimal cure (design memo per_stage_fractal_optimizer_priming_reheat_anneal_20260629 §0/§4).
    # Two byte-free structural moves that make PR(M)=PR(cov(code)) hold to the projection's ~1e-2
    # residual (Stiefel isometry) + keep the code spectrum spread. Both DEFAULT-OFF => no new params,
    # the train step + loss branches are skipped => byte-identical to the pre-DM1 path.
    ap.add_argument("--film-stiefel", action="store_true",
                    help="DM1a: each optimizer step, project film.weight (W) onto the Stiefel manifold of "
                    "ORTHONORMAL COLUMNS (WᵀW=I) via the cubic Newton-Schulz polar W(WᵀW)^-1/2. Then W is "
                    "an isometry => PR(M)=PR(cov(code)) to the projection's ~1e-2 residual (the resonance "
                    "cannot concentrate through W). Re-normalizing columns each step also neutralizes the "
                    "global-magnitude component of AdamW weight-decay on W (the design's 'WD=0 on W' "
                    "intent) WITHOUT touching the optimizer. Default OFF = byte-identical.")
    ap.add_argument("--code-spectral-entropy-weight", type=float, default=0.0,
                    help="DM1b: weight beta of a CAPACITY spectral-entropy penalty -beta*log(PR(cov(code))) "
                    "on the per-pair code covariance, keeping all ~mod_dim code directions live (the other "
                    "half of the byte-free FiLM rank-collapse cure; via WᵀW=I this raises PR(M)). PR is "
                    "(D,D)-Gram-computed (no eigendecomposition). 0.0 (default) = OFF = byte-identical.")
    ap.add_argument("--dm1-telemetry", action="store_true",
                    help="(review C1) FORCE the dm1_telemetry row (PR(M) live+shadow, PR(cov code), "
                    "Stiefel residual) at eval cadence EVEN when no DM1 lever is active -- so the A0 "
                    "BASELINE logs the PR-collapse half of the firewall verdict (else 'baseline "
                    "collapses' is unmeasurable). Pure READ (no model/grad touch); default OFF => "
                    "the row only fires when --film-stiefel/--code-spectral-entropy-weight is on => "
                    "BIT-IDENTICAL observability to the pre-C1 path.")
    # LEVER-B (THIN-LANE DROPPED-DASH PRIOR, ADDITIVE, DEFAULT-OFF). Attacks the MEASURED dominant
    # residual: 57% Road<->Lane confusion, PC0 (34.5% of residual variance) = Lane->Road DROP, 52.7% of
    # GT-lane connected components WHOLESALE-MISSED, miss-fraction monotone in dash size (<5px 93%
    # missed). The unweighted mean seg loss UNDER-fits thin 3px dashes. This up-weights the realized
    # through-R margin hinge on THIN GT-lane pixels (a precomputed local-lane-density weight map). NOTE:
    # distinct from --lane-prior-phi1 (the structured-init lane SDF prior); this is the --lane-thin-*
    # realized-margin prior. Default lane_thin_weight=0.0 = OFF = byte-identical.
    ap.add_argument("--lane-thin-weight", type=float, default=0.0,
                    help="LEVER-B: weight of the realized through-R thin-lane margin hinge (up-weights "
                    "thin/dropped GT-lane dashes). 0.0 (default) = OFF.")
    ap.add_argument("--lane-thin-class", type=int, default=1,
                    help="LEVER-B: the lane class index in the comma10k CANONICAL order "
                    "[Road0,Lane1,Undrivable2,Movable3,MyCar4]. Default 1 (Lane).")
    ap.add_argument("--lane-thin-radius", type=int, default=4,
                    help="LEVER-B: half-width of the (2r+1)^2 window for the local-lane-density thinness "
                    "measure (thin dashes => low local density => high weight). Default 4.")
    ap.add_argument("--lane-thin-target", type=float, default=0.5,
                    help="LEVER-B: the decision-margin target for the thin-lane hinge relu(target-margin). "
                    "Default 0.5 (matching --lane-margin-target).")
    ap.add_argument("--lane-thin-start-epoch", type=int, default=0,
                    help="LEVER-B: engage the thin-lane hinge only at ep>=this (0=from ep1). Gate to the "
                    "tau/l7 margin stage (e.g. 300) to avoid margin-from-scratch starvation; the engage "
                    "epoch re-treats the spike-guard.")
    # LEVER-4 (MARGIN-SALIENCY weighting, DAG FEED-eq, ADDITIVE, DEFAULT-OFF). GENERALIZES LEVER-3
    # from the class-1-only mask to the ALL-CLASS flip-prone band: the realized through-R decision
    # margin hinge is weighted PER-PIXEL by the GT-margin fragility saliency sal=exp(-gt_margin/tau)
    # (small GT margin = near a decision boundary = flip-prone; ~1 at the boundary annulus, ->0 in the
    # confident interior). MEASURED (FEED-eq, gt_n96, band 0.5): the flip-prone band is Road 47% / Lane
    # 19% / Undrivable 14% / Movable 9% / MyCar 11% -> LEVER-3 (class 1) defends only 19% of it; this
    # all-class saliency defends 100%. CLASS-AGNOSTIC (weights by fragility, not class index) so it
    # sidesteps the class-order dispute entirely. Default 0.0=OFF=byte-identical. When >0, costs ONE
    # realized seg forward (a 2nd if LEVER-3 is also on; nobody runs both). Fridrich square-root-law:
    # spread small corrections across the boundary, do not concentrate. NO scorer weights ship (the
    # saliency is computed from the PROVIDED frozen scorer at train time; rule-118 FREE).
    ap.add_argument("--margin-saliency-weight", type=float, default=0.0,
                    help="LEVER-4: weight on the additive ALL-CLASS GT-margin-saliency-weighted realized "
                    "margin hinge (0=off; generalizes --lane-edge-weight to every inter-class edge).")
    ap.add_argument("--margin-saliency-tau", type=float, default=0.5,
                    help="LEVER-4: GT-margin saliency softness sal=exp(-gt_margin/tau); smaller tau = "
                    "tighter focus on the most fragile (smallest-margin) boundary pixels. ~p1 of the "
                    "GT-margin dist (gt_n96 p1~0.38, p5~2.16) keeps the weight on the flip-prone band.")
    ap.add_argument("--margin-saliency-target", type=float, default=0.5,
                    help="LEVER-4: target decision margin for the saliency hinge relu(target - margin).")
    ap.add_argument("--margin-saliency-start-epoch", type=int, default=0,
                    help="LEVER-4 OPTIMAL-FORM: engage only at ep>=this (0=from ep1). Gate to the "
                    "tau_softplus/l7 margin stage to avoid margin-from-scratch-starves-interior; the "
                    "engage epoch re-treats the spike-guard (same discipline as --lane-edge-start-epoch).")
    ap.add_argument("--margin-saliency-uniward", action="store_true",
                    help="LEVER-4 UNIWARD (Fridrich inverse-steganalysis): additionally DOWN-weight the "
                    "saliency in TEXTURED regions (SegNet-undetectable) so capacity concentrates on the "
                    "SMOOTH flip-prone boundary. Texture energy from the realized frame's spatial "
                    "gradients (stop-grad WEIGHT). Default off.")
    ap.add_argument("--margin-saliency-uniward-beta", type=float, default=4.0,
                    help="LEVER-4 UNIWARD: texture down-weight strength sal /= (1 + beta*tex_norm).")
    # LEVER-5 (per-pair HARDNESS-weighted code-fit / training, DAG FEED-eq, ADDITIVE, DEFAULT-OFF).
    # WATERFILL the per-epoch pair-iteration budget toward HARD pairs (high d_seg debt). The frozen-
    # decoder code-fit fits independent per-pair codes, so giving a hard pair MORE update STEPS (not a
    # bigger loss scale -- Adam normalizes per-pair loss-scale to ~no-op) converges its codes further.
    # Mechanism: each epoch keeps the full permutation(P) (every pair >=1 step, never starved) PLUS
    # round(P*oversample) EXTRA steps drawn ~ hardness^power. The FAIR A/B at fixed --hardness-oversample
    # is --hardness-weighted on (extras ~ hardness) vs off (extras uniform): SAME total steps, different
    # allocation. Default --hardness-oversample 0.0 => no extras => byte-identical. MEASURED CAVEAT
    # (FEED-eq): per-pair GT-margin hardness spread on gt_n96 is only 1.31x (the fragile band is ~1.3%
    # of pixels per pair, nearly constant) -> margin-source reallocation is modest; --hardness-source
    # realized (per-pair baseline realized d_seg, which varies with the frozen decoder's per-pair
    # reconstruction quality) is the SHARPER signal for the code-fit and is the recommended source.
    ap.add_argument("--hardness-oversample", type=float, default=0.0,
                    help="LEVER-5: extra per-epoch pair-iteration steps as a fraction of P (0=off="
                    "byte-identical; e.g. 0.5 = +50%% steps, allocated by --hardness-weighted).")
    ap.add_argument("--hardness-weighted", action="store_true",
                    help="LEVER-5: draw the --hardness-oversample extra steps ~ per-pair hardness^power "
                    "(on) vs uniformly (off). On = waterfill hard pairs more code-fit budget.")
    ap.add_argument("--hardness-source", choices=["margin", "realized"], default="margin",
                    help="LEVER-5 hardness signal: 'margin' = $0 cached GT small-margin pixel fraction "
                    "(weak 1.31x spread); 'realized' = one-time per-pair baseline realized d_seg over ALL "
                    "pairs (CPU, no GPU contention; sharper; the recommended code-fit source).")
    ap.add_argument("--hardness-power", type=float, default=1.0,
                    help="LEVER-5: sharpness exponent on the per-pair hardness sampling probability.")
    ap.add_argument("--hardness-band", type=float, default=0.5,
                    help="LEVER-5 (margin source): GT-margin threshold defining a flip-prone pixel for "
                    "the per-pair hardness = mean(gt_margin < band).")
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
    # MUON FINISHER (DAG FEED-fi, PR95 stage-8, ADDITIVE, DEFAULT-OFF). The most-potent measured
    # d_seg stage (CLAUDE.md frontier "Muon is THE drop"); the prior 'Muon NOT yet wired' gap.
    # --muon-start-epoch None (default) => AdamW throughout => BIT-IDENTICAL to the pre-FEED-fi path.
    # When set, at that epoch the 2-D hidden weight matrices (in_proj/film/hidden.*) switch to
    # mlx.optimizers.Muon (Newton-Schulz orthogonalized momentum); biases/1-D + the per-pair code
    # latent + the out_sdf/out_tex final heads stay AdamW (MLX Muon docstring: final FC + embeddings
    # are Muon-suboptimal). Routed via MultiOptimizer in tac.optimization.muon_finisher_mlx.
    ap.add_argument("--muon-start-epoch", type=int, default=None,
                    help="MUON FINISHER (PR95 stage-8): epoch to switch 2-D hidden weights AdamW->Muon "
                    "(default None = AdamW throughout = bit-identical). Set AFTER the l7 stage "
                    "(>= --l7-start-epoch) so the orthogonalized finisher polishes a formed partition.")
    ap.add_argument("--muon-lr", type=float, default=None,
                    help="MUON FINISHER: Muon-group LR (default None => 0.1*--lr, the PR95 ~0.1x-base "
                    "finetune relationship). Muon normalizes its update to ~unit spectral norm, so this "
                    "is a spectral-norm step size; TUNE to the lever's own optimum (OPTIMAL-FORM): a "
                    "typical Muon finisher lr is ~1e-3 to 5e-3.")
    ap.add_argument("--muon-adamw-lr", type=float, default=None,
                    help="MUON FINISHER: AdamW-fallback-group LR for biases/code/heads during the "
                    "finisher (default None => 0.1*--lr).")
    ap.add_argument("--muon-momentum", type=float, default=0.95, help="MUON FINISHER: Muon momentum.")
    ap.add_argument("--muon-weight-decay", type=float, default=None,
                    help="MUON FINISHER: Muon-group decoupled weight decay (default None => --weight-decay).")
    ap.add_argument("--muon-ns-steps", type=int, default=5,
                    help="MUON FINISHER: Newton-Schulz iteration count (Keller Jordan default 5).")
    # ---- BUILD 1 (FEED-fw): STAGE-TRANSITION TREATMENT (ADDITIVE, all default-OFF => BIT-IDENTICAL).
    # "different stages need different treatment" applied to the TRANSITIONS so the AdamW->AdamW stage
    # boundaries (ce->tau, tau->l7) + the lane-edge / margin-saliency re-engage epochs are stable by
    # construction (the l7->Muon switch already re-treats via a fresh optimizer, FEED-fi). The
    # spike-guard re-treat already exists at every boundary; these add (1) LR re-warmup + (2) optional
    # AdamW moment reset. theta*-prereq; NOT a score row.
    ap.add_argument("--stage-transition-rewarmup-epochs", type=int, default=0,
                    help="BUILD 1: N>0 ramps LR from --stage-transition-rewarmup-floor back to the "
                    "scheduled LR over N epochs after each AdamW->AdamW stage boundary (default 0=OFF "
                    "=> bit-identical). Requires --lr-schedule; no effect during the Muon finisher.")
    ap.add_argument("--stage-transition-rewarmup-floor", type=float, default=0.1,
                    help="BUILD 1: LR fraction at the boundary epoch for re-warmup (used only when "
                    "--stage-transition-rewarmup-epochs > 0; must be in [0,1]).")
    ap.add_argument("--stage-transition-rewarmup-shape", choices=["linear", "cosine"], default="linear",
                    help="BUILD 1: re-warmup ramp shape (used only when rewarmup-epochs > 0).")
    ap.add_argument("--stage-transition-reset-moments", action="store_true",
                    help="BUILD 1: at each AdamW->AdamW stage boundary, rebuild the AdamW optimizer so "
                    "the m/v moments are zeroed (stale momentum through a loss-landscape change is the "
                    "FEED-ft#3 tau-jump root cause). Default OFF => bit-identical. No-op during the "
                    "Muon finisher (it already re-inits a fresh optimizer).")
    # ---- BUILD 2 (FEED-fw): LANE-PRIOR phi1 (ADDITIVE, default-OFF => structured-init BIT-IDENTICAL).
    # Initialize the structured-init target's phi1 (lane-class SDF) channel to the signed distance of
    # the openpilot deg-3 centerline curve (FEED-fs: that centerline IS the Road<->Lane separatrix,
    # residual 1.9e-5). REUSES tac.boundary_math.lane_sdf_component (build_structured_lane_sdf: the
    # ground-plane homography K @ scorer-res {fx=910*512/1164=400.3, ...} -> image-space deg-3 lane
    # curve -> per-pixel signed distance; + inject_lane_sdf). rule-118 FREE generic structure: a
    # better TRAINING-TIME starting point that ships 0 archive bytes (only if the centerline coords
    # were SHIPPED would they be COUNTED, ~8 floats/frame -- a SEPARATE archive-side option, NOT this
    # build). Requires --structured-init (the pretrain mechanism that absorbs the target).
    ap.add_argument("--lane-prior-phi1", action=argparse.BooleanOptionalAction, default=False,
                    help="BUILD 2: init the structured-init target's lane (phi1) channel to the "
                    "openpilot deg-3 centerline signed distance (default OFF => bit-identical). "
                    "Requires --structured-init.")
    ap.add_argument("--lane-prior-phi1-mode", choices=["replace", "bias"], default="replace",
                    help="BUILD 2: inject the centerline SDF by REPLACE (lane channel becomes the "
                    "openpilot fit) or BIAS (add to the static-core lane channel). Default replace.")
    ap.add_argument("--lane-prior-phi1-bias-scale", type=float, default=1.0,
                    help="BUILD 2: scale for --lane-prior-phi1-mode bias (unused for replace).")
    ap.add_argument("--lane-prior-phi1-source-pair", type=int, default=0,
                    help="BUILD 2: which cached pair's L* argmax the centerline is fit from (default "
                    "0, matching the structured-init pretrain's pair-0 feats convention).")
    ap.add_argument("--lane-prior-phi1-dash-gate", action=argparse.BooleanOptionalAction, default=True,
                    help="BUILD 2: model the lane dash period (deg-3 centerline + dash). Default on.")
    args = ap.parse_args(argv)

    # (review C2) --anneal-epochs guard: must be >= 1 when set (it is a cosine DENOMINATOR). A value
    # < --epochs means the anneal COMPLETES before the run ends (temp/LR clamp past their end values
    # for the tail) -- legal for a warm-start window but usually a mistake otherwise, so WARN (do not
    # fail). None (default) => no guard fires => bit-identical.
    if getattr(args, "anneal_epochs", None) is not None:
        if args.anneal_epochs < 1:
            raise ValueError(f"--anneal-epochs ({args.anneal_epochs}) must be >= 1 (cosine denominator).")
        if args.anneal_epochs < args.epochs:
            print(json.dumps({"stage": "anneal_epochs_WARN", "anneal_epochs": int(args.anneal_epochs),
                              "epochs": int(args.epochs),
                              "msg": "--anneal-epochs < --epochs: the temp/LR anneal completes BEFORE the "
                              "run ends; the tail epochs run at the clamped end values. Intended for a "
                              "WARM-START window (resume mid-schedule); verify this is what you want."}),
                  flush=True)

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

    # (LEVER-B) thin-lane dropped-dash prior fail-closed config guard (same NO-FAKE silent-no-op class).
    validate_lane_thin_config(
        lane_thin_weight=args.lane_thin_weight, lane_thin_start_epoch=args.lane_thin_start_epoch,
        epochs=args.epochs, lane_thin_class=args.lane_thin_class, lane_thin_radius=args.lane_thin_radius,
        n_classes=5,
    )

    # (LEVER-A) FiLM-rank-fix fail-closed config guards (same NO-FAKE silent-no-op class).
    # A rank-floor with target <= 1 can NEVER penalize (PR >= 1 always) = a silent no-op = a FALSE
    # 'rank-floor does nothing' verdict. The film-per-layer / film-concat-code architecture routes are
    # loaded from a frozen-decoder npz that does NOT contain them, so --freeze-decoder-fit-codes would
    # leave them zero-init AND frozen = never trained = a silent no-op = a FALSE 'film-fix does nothing'.
    if args.film_rank_floor_weight > 0.0 and args.film_rank_floor_target <= 1.0:
        raise ValueError(
            f"--film-rank-floor-weight {args.film_rank_floor_weight} > 0 but "
            f"--film-rank-floor-target ({args.film_rank_floor_target}) <= 1: the participation ratio is "
            ">= 1 by construction, so relu(target - PR) would be 0 always -> a silent no-op = a FALSE "
            "'rank-floor does nothing' verdict. Set --film-rank-floor-target > 1 (e.g. 4).")
    if (args.film_per_layer or args.film_concat_code) and args.freeze_decoder_fit_codes:
        raise ValueError(
            "--film-per-layer / --film-concat-code are incompatible with --freeze-decoder-fit-codes: "
            "the frozen decoder npz has no film_pl/concat_pl keys, so those routes would stay zero-init "
            "AND frozen = never trained = a silent no-op = a FALSE 'film-fix does nothing' verdict. Run "
            "the FiLM-rank-fix on a joint (unfrozen) run.")

    # (FEED-eq) LEVER-4 fail-closed config guard: a saliency lever that never engages (start > epochs)
    # is a silent no-op = a FALSE 'margin-saliency does not help' verdict (same NO-FAKE class the lane
    # validator extincts). Also guard tau>0 so exp(-margin/tau) is well-defined.
    if args.margin_saliency_weight > 0.0:
        if args.margin_saliency_start_epoch > args.epochs:
            raise ValueError(
                f"--margin-saliency-weight {args.margin_saliency_weight} > 0 but "
                f"--margin-saliency-start-epoch ({args.margin_saliency_start_epoch}) > --epochs "
                f"({args.epochs}): the saliency hinge would NEVER engage -> a silent no-op = a FALSE "
                "'margin-saliency does not help' verdict. Set --margin-saliency-start-epoch <= --epochs."
            )
        if args.margin_saliency_tau <= 0.0:
            raise ValueError(f"--margin-saliency-tau ({args.margin_saliency_tau}) must be > 0 "
                             "(sal=exp(-gt_margin/tau)).")

    # (FEED-fi) MUON FINISHER fail-closed config guard (same NO-FAKE class as the lane/saliency
    # validators): a finisher that never engages (start > epochs) is a silent no-op = a FALSE
    # 'Muon does not help d_seg' verdict; a finisher with NO trainable 2-D weights (frozen decoder)
    # routes everything to AdamW = the Muon group is empty = the same false verdict. Fail LOUD.
    if args.muon_start_epoch is not None:
        if not (1 <= args.muon_start_epoch <= args.epochs):
            raise ValueError(
                f"--muon-start-epoch ({args.muon_start_epoch}) must be in [1, --epochs ({args.epochs})]: "
                "outside the budget the Muon finisher would NEVER engage -> a silent no-op = a FALSE "
                "'Muon does not help' verdict. PR95 places it as the FINAL stage (set it >= "
                f"--l7-start-epoch {args.l7_start_epoch} when --curriculum is on)."
            )
        if args.freeze_decoder_fit_codes:
            raise ValueError(
                "--muon-start-epoch is incompatible with --freeze-decoder-fit-codes: the only trainable "
                "param then is the per-pair `code` latent, which is AdamW-routed (Muon-suboptimal for "
                "embeddings) -> the Muon group would be EMPTY = a silent no-op = a FALSE 'Muon does not "
                "help' verdict. Muon finishes the DECODER weight matrices; run it on a joint (unfrozen) run."
            )
        # FEED-fm FIX-3 (RULE-6 freedom): placing the finisher BEFORE the l7_softplus stage (under
        # curriculum) is the PR95-suboptimal placement (Muon polishes a not-yet-formed partition), but
        # it is the operator's CHOICE to make -> WARN loudly, do NOT fail closed. The range [1,epochs]
        # + freeze-decoder guards above STAY hard raises (those are silent-no-op / empty-Muon-group
        # NO-FAKE traps, not placement preferences). Gated on --curriculum: l7_start_epoch only governs
        # a stage that exists under curriculum, so the warning is meaningful only there.
        if args.curriculum and args.muon_start_epoch < args.l7_start_epoch:
            print(json.dumps({"stage": "muon_finisher_WARN",
                              "muon_start_epoch": int(args.muon_start_epoch),
                              "l7_start_epoch": int(args.l7_start_epoch),
                              "msg": "--muon-start-epoch < --l7-start-epoch: the Muon finisher engages "
                              "BEFORE the l7_softplus stage forms the partition. PR95 places Muon as the "
                              "FINAL stage; an orthogonalized finisher on a not-yet-formed partition is "
                              "likely weaker d_seg. ALLOWED (operator freedom); set >= --l7-start-epoch "
                              "for the PR95 placement."}), flush=True)

    # BUILD 1 (FEED-fw) fail-closed config guards (same NO-FAKE silent-no-op class as the lane/muon
    # validators). DEFAULT-OFF (rewarmup-epochs 0, lane-prior off) => none of these fire => unchanged.
    if args.stage_transition_rewarmup_epochs < 0:
        raise ValueError(
            f"--stage-transition-rewarmup-epochs ({args.stage_transition_rewarmup_epochs}) must be "
            ">= 0 (0 = OFF).")
    if args.stage_transition_rewarmup_epochs > 0:
        if not args.lr_schedule:
            raise ValueError(
                "--stage-transition-rewarmup-epochs > 0 requires --lr-schedule: the re-warmup "
                "multiplies the SCHEDULED LR, so with --no-lr-schedule it would be a silent no-op = "
                "a FALSE 're-warmup does nothing' verdict.")
        if not (0.0 <= args.stage_transition_rewarmup_floor <= 1.0):
            raise ValueError(
                f"--stage-transition-rewarmup-floor ({args.stage_transition_rewarmup_floor}) must be "
                "in [0, 1] (the LR fraction at the boundary epoch).")
    # BUILD 2 (FEED-fw) fail-closed guard: the lane prior is injected into the structured-init
    # pretrain target, so without --structured-init it would NEVER be applied = a silent no-op = a
    # FALSE 'lane prior does nothing' verdict.
    if getattr(args, "lane_prior_phi1", False) and not args.structured_init:
        raise ValueError(
            "--lane-prior-phi1 requires --structured-init: the openpilot centerline SDF is injected "
            "into the structured-init pretrain target; without --structured-init the prior would "
            "never be applied = a silent no-op = a FALSE 'lane prior does nothing' verdict.")

    result = run_train(args)
    print("\n=== LEVEL-SET WITNESS RESULT (realized through R) ===")
    print(json.dumps({"front_end": result["front_end"], "history": result["history"],
                      "axis": result["axis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
