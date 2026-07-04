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
    _seed_muon_momentum_from_adam,
    _torch_R_to_camera_uint8,
    cpu_verdict_d_pose_batch,
    cpu_verdict_d_seg_batch,
    implied_score_from_verdict,
    load_gt_from_cache,
    make_loss_fn,
    maybe_enable_mx_compile_r,
    precompute_gt,
    quantize_witness_blob,
    r_isolated_microbench,
    render_through_R_mlx,
    set_fused_r_kernel,
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


def _rss_gib() -> float:
    """Best-effort resident-set-size of THIS process in GiB (psutil, then resource fallback).

    Used only for observability (the #205 OOM instrumentation) -- NEVER read back into training
    (BIT-IDENTICAL). Returns -1.0 when unavailable (NO-FAKE: never a fabricated number)."""
    try:
        import psutil  # noqa: PLC0415

        return float(psutil.Process().memory_info().rss) / (1024.0 ** 3)
    except Exception:
        try:
            import resource  # noqa: PLC0415

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS ru_maxrss is BYTES; Linux is KiB. Heuristic: >1e9 => already bytes.
            return float(ru) / (1024.0 ** 3) if ru > 1e9 else float(ru) / (1024.0 ** 2)
        except Exception:
            return -1.0


def _verdict_dseg_dpose_chunked(
    seg_cpu: Any, posenet_cpu: Any,
    f0s: list, f1s: list, lstars: list, poses: list, *, vbatch: int,
) -> tuple[float, float]:
    """(#205 REAL OOM FIX) mean d_seg / d_pose over N pairs, running SegNet/PoseNet in CHUNKS of
    ``vbatch`` instead of one N-wide torch batch.

    The batched verdict (``cpu_verdict_d_seg_batch`` / ``cpu_verdict_d_pose_batch``) casts a
    ``(N, 2, 3, 874, 1164)`` uint8 stack to **fp32** (= ~14.6 GiB at N=600) and forwards it through
    EfficientNet-B2 / FastViT-T12 in ONE batch -> tens of GiB of activations. That transient spike,
    on top of the resident ~41 GiB self-orient cf_mx_cache, is what tripped the 90 GB safe-run guard
    and killed the n600 launch before its first checkpoint. Chunking bounds the transient to
    ``vbatch`` pairs.

    BIT-IDENTICAL: the scorers run under ``torch.inference_mode()`` in EVAL mode -> BatchNorm uses
    RUNNING stats (batch-size-independent), argmax is per-pixel, MSE is per-pair -> the per-chunk
    concatenation equals the single N-wide batch to the last bit. ``vbatch<=0`` restores the
    single-batch (pre-fix) path for the A/B parity check."""
    n = len(f1s)
    if vbatch is None or vbatch <= 0 or vbatch >= n:
        ds = cpu_verdict_d_seg_batch(seg_cpu, f1s, lstars)
        dp = cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, poses)
        return float(np.mean(ds)), float(np.mean(dp))
    ds_all: list[float] = []
    dp_all: list[float] = []
    for s in range(0, n, vbatch):
        e = min(s + vbatch, n)
        ds_all.extend(cpu_verdict_d_seg_batch(seg_cpu, f1s[s:e], lstars[s:e]))
        dp_all.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[s:e], f1s[s:e], poses[s:e]))
    return float(np.mean(ds_all)), float(np.mean(dp_all))


def _mlx_mem_gib(mx: Any) -> dict[str, float]:
    """MLX Metal allocator stats in GiB: active (LIVE arrays), cache (freed-but-pooled buffers),
    peak (high-water since last reset). The active/cache split is the #205 OOM diagnosis instrument:
    a small active + huge cache => the buffer POOL is the leak (fixed by ``mx.clear_cache()`` inside
    the accum loop), NOT the live working set. Pure read; NEVER read back into training."""
    out: dict[str, float] = {}
    for key, fn in (("active", "get_active_memory"), ("cache", "get_cache_memory"),
                    ("peak", "get_peak_memory")):
        try:
            out[key] = float(getattr(mx, fn)()) / (1024.0 ** 3)
        except Exception:
            out[key] = -1.0
    return out


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _git_provenance() -> dict[str, Any]:
    """Best-effort git provenance captured ONCE at launch (deterministic-reproducibility
    non-negotiable: provenance with every result = git hash + seed + config + upstream snapshot sha).

    NO-FAKE: when git is unavailable / not a repo, every field is ``"unknown"`` / ``False`` -- NEVER
    a fabricated sha. ``git_sha`` (repo HEAD) pins the trainer code AND the committed pinned
    ``upstream/`` snapshot (both live in the same tree); ``git_dirty`` flags an uncommitted working
    tree (a run from a dirty tree is NOT reproducible from the sha alone); ``upstream_tree_sha`` is the
    ``upstream/`` subtree object id (the frozen-scorer snapshot the verdict authority runs)."""
    import subprocess

    def _g(*a: str) -> str:
        try:
            r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=8)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    sha = _g("rev-parse", "HEAD") or "unknown"
    dirty = bool(_g("status", "--porcelain"))
    # upstream/ is an UNTRACKED pinned snapshot (not in git HEAD), so the frozen-scorer authority is
    # pinned by the CANONICAL content hash (tac.contest_compliance) -- the same upstream_snapshot_sha256
    # every ledger/anchor carries -- NOT a git tree sha. Best-effort; "unknown" when absent (NO-FAKE).
    try:
        from tac.contest_compliance import compute_upstream_snapshot_sha256
        upstream_sha = compute_upstream_snapshot_sha256(REPO) or "unknown"
    except Exception:
        upstream_sha = "unknown"
    return {"git_sha": sha, "git_dirty": dirty, "upstream_snapshot_sha256": upstream_sha}


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
    hosc_beta: float | None = None, provenance: dict[str, Any] | None = None,
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
    # ---- PROVENANCE (deterministic-reproducibility: git sha + upstream snapshot sha in EVERY
    # per-stage byte-close artifact so a shipped checkpoint traces to the exact code + frozen scorer.
    # Additive + byte-close-ignored (.get()s only keys it knows); default "unknown" = NO-FAKE, never
    # a fabricated sha). ----
    _prov = provenance or {}
    flat["__cfg_git_sha"] = np.asarray(str(_prov.get("git_sha", "unknown")))
    flat["__cfg_git_dirty"] = np.asarray(int(bool(_prov.get("git_dirty", False))))
    flat["__cfg_upstream_snapshot_sha256"] = np.asarray(str(_prov.get("upstream_snapshot_sha256", "unknown")))
    # ---- #224 AA-SDF observation-map render cfg (additive provenance; the exact-eval decode
    # (#202) reconstructs the SAME AA mode deterministically -- NO extra archive bytes: the IPE
    # attenuation is a function of (B, render_hw, footprint) all already in the ckpt, and the
    # supersample grid is deterministic at decode). DEFAULT none/1/1.0 => byte-identical cfg. ----
    flat["__cfg_render_aa"] = np.asarray(str(getattr(args, "render_aa", "none")))
    flat["__cfg_aa_supersample"] = np.asarray(int(getattr(args, "aa_supersample", 1)))
    flat["__cfg_aa_ipe_footprint"] = np.asarray(float(getattr(args, "aa_ipe_footprint", 1.0)))
    return flat


def _build_resume_state_arrays(
    live_np: dict[str, np.ndarray], ema_np: dict[str, np.ndarray],
    opt_np: dict[str, np.ndarray] | None, *, args: Any, epoch: int, in_feat: int,
    recent_losses: "list[float] | None" = None, provenance: dict[str, Any] | None = None,
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
    # (F2 fix) #224 render-side LEVER cfg: persist the levers whose engagement CHANGES the loss /
    # render target mid-run so a --resume-from can FAIL-CLOSED (via _resume_lever_divergences) when the
    # resume command silently drops or diverges a lever the run was trained with (a deterministic-repro
    # violation the film-arch guard does NOT cover -- these are loss/render-only, they add no param
    # KEYS so the missing-param guard cannot see them). ZERO archive bytes (resume sidecar is not
    # byte-closed). hosc_beta_end None -> -1.0 sentinel (matches the current-arg encoding in the guard).
    out["__cfg_lane_render_band"] = np.asarray(int(bool(getattr(args, "lane_render_band", False))))
    out["__cfg_lane_band_start_epoch"] = np.asarray(int(getattr(args, "lane_band_start_epoch", 300)))
    out["__cfg_persistence_loss_weight"] = np.asarray(float(getattr(args, "persistence_loss_weight", 0.0)))
    out["__cfg_amplify_weight"] = np.asarray(float(getattr(args, "amplify_weight", 0.0)))
    out["__cfg_render_aa"] = np.asarray(str(getattr(args, "render_aa", "none")))
    _hbe = getattr(args, "hosc_beta_end", None)
    out["__cfg_hosc_beta_end"] = np.asarray(-1.0 if _hbe is None else float(_hbe))
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
    # SPIKE-GUARD running-median window (the last <=50 batch losses). It GATES step-skipping
    # (loss > spike_factor * median => the optimizer.update is skipped), so it is part of the
    # weight trajectory: a resume with an EMPTY window (median None => never skips) would diverge
    # from a continuous run that WOULD have skipped. Persist it so --resume-from is bit-faithful even
    # across a spike. Empty list => a 0-length array (default-safe; a pre-fix ckpt lacks the key =>
    # the loop's fresh [] is used, i.e. the prior behavior). Per the deterministic-repro non-negotiable.
    out["__recent_losses"] = np.asarray(list(recent_losses or []), np.float64)
    # ---- PROVENANCE (git sha + upstream snapshot sha; cost ZERO archive bytes -- the resume sidecar
    # is not byte-closed; makes a --resume-from traceable to the exact code + frozen scorer). ----
    _prov = provenance or {}
    out["__cfg_git_sha"] = np.asarray(str(_prov.get("git_sha", "unknown")))
    out["__cfg_git_dirty"] = np.asarray(int(bool(_prov.get("git_dirty", False))))
    out["__cfg_upstream_snapshot_sha256"] = np.asarray(str(_prov.get("upstream_snapshot_sha256", "unknown")))
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


def _resume_lever_divergences(resume_cfg: dict[str, Any], args: Any) -> list[str]:
    """(F2) List render-side LEVER cfg keys that DIVERGE between the resume sidecar (what the run was
    trained with) and the current argv (what this resume would run). A non-empty list means a
    ``--resume-from`` would SILENTLY change / drop a lever = a deterministic-reproducibility violation
    the film-arch guard cannot see (these loss/render-only levers add NO param keys). Only keys PRESENT
    in the sidecar are checked, so a pre-F2 sidecar (which lacks them) yields NO spurious divergence.
    Pure / MLX-free -> unit-tested. ``lane_band_start_epoch`` is only flagged when the band is engaged
    in EITHER config (a start-epoch change is inert while the band is OFF in both)."""
    div: list[str] = []
    _hbe = getattr(args, "hosc_beta_end", None)
    cur_hbe = -1.0 if _hbe is None else float(_hbe)
    cur_band = int(bool(getattr(args, "lane_render_band", False)))
    # (key, current value, is_float) — non-float compared as string (int/bool/str all normalize).
    checks: list[tuple[str, object, bool]] = [
        ("__cfg_mod_dim", int(getattr(args, "mod_dim", 0)), False),
        ("__cfg_lane_render_band", cur_band, False),
        ("__cfg_persistence_loss_weight", float(getattr(args, "persistence_loss_weight", 0.0)), True),
        ("__cfg_amplify_weight", float(getattr(args, "amplify_weight", 0.0)), True),
        ("__cfg_render_aa", str(getattr(args, "render_aa", "none")), False),
        ("__cfg_hosc_beta_end", cur_hbe, True),
    ]
    for key, cur, is_float in checks:
        if key not in resume_cfg:
            continue
        ckpt = resume_cfg[key]
        if is_float:
            try:
                diverged = abs(float(ckpt) - float(cur)) > 1e-6
            except (TypeError, ValueError):
                diverged = str(ckpt) != str(cur)
        else:
            diverged = str(ckpt) != str(cur)
        if diverged:
            div.append(f"{key[len('__cfg_'):]}: ckpt={ckpt!r} != resume-argv={cur!r}")
    # lane_band_start_epoch: inert while the band is OFF in BOTH -> only flag when engaged in either.
    if "__cfg_lane_band_start_epoch" in resume_cfg:
        ckpt_band = int(resume_cfg.get("__cfg_lane_render_band", cur_band) or 0)
        if (ckpt_band or cur_band):
            ckpt_se = int(resume_cfg["__cfg_lane_band_start_epoch"])
            cur_se = int(getattr(args, "lane_band_start_epoch", 300))
            if ckpt_se != cur_se:
                div.append(f"lane_band_start_epoch: ckpt={ckpt_se} != resume-argv={cur_se}")
    return div


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

        # ---- #224 accessors for the analytic-lane render-band (ADDITIVE; only called when
        # --lane-render-band is ON => the default render is byte-identical). ----
        def call_margin(self, coord_feats, code_idx):
            """top1-top2 softmax decision margin (PROB scale) of the witness partition — the
            #141 quantity the analytic-lane uncertainty gate rides. Returns (P,); reshape to
            (H,W) at the call site."""
            soft = mx.softmax(self.out_sdf(self._trunk(coord_feats, code_idx)) / self.softmax_temp, axis=-1)
            s = mx.sort(soft, axis=-1)                              # ascending
            return s[..., -1] - s[..., -2]                          # (P,) top1 - top2

        def render_lane_appearance(self, coord_feats, code_idx, lane_cls: int = 1):
            """The witness's OWN per-pixel lane color = sigmoid(palette[lane_cls] + tex)*255
            (self-consistent, byte-free; gradient flows through tex/palette per the band spec).
            luma-collapsed when not chroma (matches _compose_rgb)."""
            tex = self.out_tex(self._trunk(coord_feats, code_idx))  # (P,3)
            rgb = mx.sigmoid(self.palette[lane_cls] + tex) * 255.0  # (P,3)
            if not self.chroma:
                luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
                rgb = mx.concatenate([luma, luma, luma], axis=-1)
            return rgb

    return LevelSetRGBWitness()


# ---------------------------------------------------------------------------
# MLX level-set regularizers (differentiable twins of the numpy reference). On phi (P,K)
# reshaped to (H,W,K): Eikonal drives |grad phi|->1 (true SDF); length is the Chan-Vese
# boundary-perimeter prior (short, smooth class boundaries). Kept SMALL (topology bias, not
# the driver — the realized seg loss drives d_seg).
# ---------------------------------------------------------------------------
def _eikonal_length_mlx(phi_pk, render_h: int, render_w: int, len_eps: float = 1.0,
                        junction_relax: float = 0.0, junction_tau: float = 0.5):
    """(fix h) Eikonal + Chan-Vese length on the DECISION MARGIN m = phi_top1 - phi_top2 (the
    quantity the argmax boundary lives on), NOT each field's own zero-set. Eikonal drives
    |grad m|->1 (the 1-Lipschitz margin = the R-survival quantity); the length term
    delta_eps(m)*|grad m| penalizes the perimeter of the ACTUAL inter-class boundary {m=0}.

    (THETA* TIER-2 STRETCH-1) ``junction_relax`` (default 0.0 = OFF = BIT-IDENTICAL) down-weights the
    Eikonal |grad m|->1 residual near TRIPLE JUNCTIONS, where 3+ classes meet and the top1-top2 margin
    surface m is genuinely non-smooth (a crease/kink), so forcing |grad m|=1 there fights the geometry
    and injects boundary noise. Triple-junction proximity is the top2-top3 SDF gap g3 =
    sort(phi)[-2]-sort(phi)[-3] (small => near a 3-way meet; needs >=3 classes). The per-pixel weight
    w = 1 - junction_relax*exp(-g3/junction_tau) in [1-relax, 1] multiplies the SQUARED Eikonal residual
    BEFORE the mean. junction_relax=0 => w==1.0 exactly => mean is BIT-IDENTICAL (x*1.0==x for finite
    IEEE floats). The LENGTH term is unchanged (delta_eps already localizes it to the {m=0} boundary)."""
    import mlx.core as mx

    phi = mx.reshape(phi_pk, (render_h, render_w, -1))
    srt = mx.sort(phi, axis=-1)
    m = srt[..., -1] - srt[..., -2]  # (H,W) >=0 decision margin (top1-top2)
    gy = m[1:, :] - m[:-1, :]
    gx = m[:, 1:] - m[:, :-1]
    gmag = mx.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2 + 1e-8)  # (H-1,W-1)
    eik_resid = (gmag - 1.0) ** 2
    if junction_relax > 0.0 and phi.shape[-1] >= 3:
        # (STRETCH-1) triple-junction proximity weight: down-weight the Eikonal where 3 classes nearly
        # meet (small top2-top3 gap). Aligned to the (H-1,W-1) gmag grid by the matching [:-1,:-1] slice.
        g3 = srt[..., -2] - srt[..., -3]                                  # (H,W) top2-top3 gap (>=0)
        w = 1.0 - float(junction_relax) * mx.exp(-g3[:-1, :-1] / float(junction_tau))  # (H-1,W-1)
        eik = mx.mean(w * eik_resid)
    else:
        eik = mx.mean(eik_resid)  # DEFAULT: BIT-IDENTICAL to the pre-theta* `mx.mean((gmag-1.0)**2)`.
    mc = m[:-1, :-1]
    delta = (len_eps / np.pi) / (len_eps * len_eps + mc * mc)  # delta_eps at the {m=0} boundary
    length = mx.mean(delta * gmag)
    return eik, length, mx.mean(gx * gx) + mx.mean(gy * gy)


def _nuclear_norm_smooth_mlx(code, *, rel_eps: float = 1e-3, ns_iters: int = 25):
    """(THETA* TIER-2 MUST-2) DIFFERENTIABLE smoothed nuclear norm of the per-(pair,frame) FiLM code
    matrix ``code`` (shape (num_pairs*2, mod_dim)) -- a convex low-rank relaxation that drives the
    learned per-pair codes toward a low-rank subspace (-> fewer effective DOF -> lower entropy / rate
    at byte-close). DEFAULT-OFF at the call site (--code-nuclear-weight 0.0 => never invoked => the
    loss is byte-identical).

    WHY smoothed + Newton-Schulz (the differentiable-path choice, documented per NO-FAKE): MLX 0.31
    has NO vjp for ``mx.linalg.svd`` NOR ``mx.linalg.eigvalsh`` ([Primitive::vjp] Not implemented),
    so NEITHER can be a LOSS term (verified on CPU). The nuclear norm = sum of singular values =
    trace(sqrt(C^T C)). The matrix square root is computed by the coupled Newton-Schulz iteration
    (matmuls ONLY -> fully autodiff-able in MLX). Plain NS DIVERGES (->NaN) on exact-zero singular
    values -- exactly the rank-deficient codes the penalty itself produces -- so we compute the
    SMOOTHED nuclear norm ``trace(sqrt(C^T C + eps*||C^T C||_F * I)) = sum_i sqrt(sigma_i^2 +
    eps*||G||_F)`` with a small RELATIVE floor ``eps`` (default 1e-3). This is a standard smoothed
    nuclear-norm surrogate: -> the exact nuclear norm as eps->0; matches it to ~0.3% on well-conditioned
    full-rank inputs (verified, gradient cosine 1.0000 vs the exact U V^T); stays FINITE +
    monotone-in-the-singular-values (still drives low-rank) on rank-deficient inputs; and ->0 as the
    codes ->0. It is NOT the exact nuclear norm (the smoothing floor over-counts near-zero singular
    directions) -- labelled SMOOTHED, not exact, per NO-FAKE. MLX matmuls only (no model/scorer; runs
    + autodiffs on CPU). Empirical anchor: experiments/tests/test_levelset_theta_star_tier2_levers.py."""
    import mlx.core as mx

    G = code.T @ code                              # (mod_dim, mod_dim) Gram, PSD
    n = G.shape[0]
    eye = mx.eye(n)
    normG = mx.sqrt(mx.sum(G * G)) + 1e-20         # ||G||_F (scalar)
    Y0 = G / normG + float(rel_eps) * eye          # eigvals in [rel_eps, 1+rel_eps] -> NS-stable
    s = mx.sqrt(mx.sum(Y0 * Y0)) + 1e-20           # spectral renormalization (NS safety margin)
    Y = Y0 / s
    Z = eye
    for _ in range(int(ns_iters)):
        Tm = 0.5 * (3.0 * eye - Z @ Y)             # coupled Newton-Schulz for the matrix sqrt
        Y = Y @ Tm
        Z = Tm @ Z
    # trace(sqrt(G + eps*||G||_F I)) = sqrt(normG)*sqrt(s)*trace(sqrt(Y))  [sqrt homogeneous deg-1/2]
    return mx.trace(Y) * mx.sqrt(s) * mx.sqrt(normG)


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


def _adam_bias_correction_for(adam_beta2: float) -> bool:
    """#224 Wave C FIX-1 (LAUNCH-BLOCKER): MLX ``optim.AdamW`` ``bias_correction`` DEFAULTS FALSE.

    Without bias correction, at step ``t`` the second-moment ``v`` is ``(1-beta2) * mean(g^2)`` and is
    NOT divided by ``(1-beta2^t)``, so ``sqrt(v)`` is ``~sqrt(1-beta2)`` too small early. With the
    all-levers small-n beta2 (0.9999999, 1-beta2=1e-7) that is ``sqrt(1e-7)/sqrt(1e-3) ~ 316/31.6 ~
    10`` smaller than the 0.999 default => the step-1 effective LR blows up ~100x (measured ratio 99.99x)
    => AdamW random-walk / divergence. The arXiv small-n derivation (1-beta2 <~ (1-beta1^5)/n^3.5) is
    faithful ONLY with bias correction (which makes vhat = v/(1-beta2^t) => step-1 update ~ lr*sign(g)
    independent of beta2). So bias correction is REQUIRED on the high-beta2 path.

    Gate ON only OFF THE DEFAULT (adam_beta2 != 0.999). At 0.999 (== the MLX/proven_base default) we
    keep ``bias_correction`` at the MLX default (False) so the DEFAULT AdamW construction is BYTE-
    IDENTICAL to the pre-FIX-1 path (the --adam-beta2 default stays 0.999 => byte-identical-off 7/7).
    Pure + total => $0 unit-testable. Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
    against"."""
    return abs(float(adam_beta2) - 0.999) > 1e-9


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
    treated as unset, so the default path is the pre-C2 formula bit-for-bit.

    (THETA* TIER-2 MUST-1) ``--tau-anneal-shape`` selects the homotopy/continuation curve tau(ep) walks
    from ``softmax_temp_start`` -> ``softmax_temp_end`` (the anneal denominator stays --anneal-epochs):
      * ``cosine``      (DEFAULT) the pre-theta* cosine. BIT-IDENTICAL to the inline formula.
      * ``geometric``   log-spaced (exponential) decay tau = start*(end/start)**prog == start**(1-prog)
                        * end**prog -> spends MORE epochs at small tau (slows the near-tau->0
                        continuation step that drives the measured late-tau d_seg volatility). Requires
                        start>0, end>0 (guarded in main()).
      * ``cosine_hold`` cosine that reaches the floor at ``--tau-hold-frac`` of the window, then HOLDS
                        at ``softmax_temp_end``. ``--tau-hold-frac 1.0`` (DEFAULT) == NO hold == the
                        cosine branch (BIT-IDENTICAL: prog/1.0==prog exactly for finite IEEE floats and
                        hold_frac>=1.0 routes through the SAME final cosine line below).
    Returns the EXACT value the pre-theta* inline cosine produced when shape=='cosine' (or
    'cosine_hold' with hold_frac>=1.0) -- the #1 bit-identical-when-off gate. ``float(args.x) == args.x``
    for the argparse floats, so the named locals do not perturb the arithmetic."""
    _ae = getattr(args, "anneal_epochs", None) or args.epochs
    prog_t = (ep - 1) / max(_ae - 1, 1)
    shape = str(getattr(args, "tau_anneal_shape", "cosine"))
    start = float(args.softmax_temp_start)
    end = float(args.softmax_temp_end)
    if shape == "geometric":
        # log-spaced (exponential) decay; endpoints are exact at prog 0/1. main() guards start>0,end>0.
        return float(start * (end / start) ** prog_t)
    if shape == "cosine_hold":
        hold_frac = float(getattr(args, "tau_hold_frac", 1.0))
        if hold_frac < 1.0:
            if prog_t >= hold_frac:
                return end                       # held at the floor for the tail of the window
            prog_t = prog_t / hold_frac          # rescale [0,hold_frac)->[0,1); falls through to cosine
        # hold_frac>=1.0: NO hold -> fall through with the ORIGINAL prog_t -> BIT-IDENTICAL cosine.
    # DEFAULT cosine (and cosine_hold w/ hold_frac>=1.0): the pre-theta* inline formula, unchanged.
    return float(end + 0.5 * (start - end) * (1 + np.cos(np.pi * prog_t)))


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
    # #205 PROVENANCE: capture git sha + upstream snapshot sha ONCE at launch (threaded into result.json
    # AND every per-stage checkpoint cfg so the #205 run + each byte-close artifact is reproducible from
    # provenance). NO-FAKE: "unknown" when git is unavailable, never fabricated.
    _run_provenance = _git_provenance()
    print(json.dumps({"stage": "provenance", **_run_provenance,
                      "seed": int(args.seed)}), flush=True)
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
    # #224 (Wave B) AA-supersample + self-orient FINE dir-feat state (declared here so the render/
    # reorient closures below see run-scope defaults even when AA/self-orient is OFF). Populated only
    # when --render-aa supersample + --self-orient + --aa-self-orient-fine-mode {batch,full}. The base
    # argmax per pair (H,W int8, ~118MB @ n600 — cheap) is snapshotted at each reorient so the fine
    # dir-feats can be recomputed (NN-upsample argmax -> ss*grid -> fine EDT-tangent -> directional
    # Fourier) without re-running the witness argmax.
    _aa_so_fine = False
    _aa_fine_mode = "refuse"
    _aa_coords_fine = None
    base_argmax_per_pair: list = [None] * P
    _aa_fine_dir_full: list = [None] * P      # full mode: per-pair fine dir-feats (mx), rebuilt @ reorient
    _aa_fine_lru: dict = {}                    # batch mode: bounded FIFO cache of per-pair fine dir-feats

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
    # ---- #224 AA-SDF observation-map render (aa_sdf_observation_render; MEASURED #1 rep lever,
    # DAG FEED-ly/-ma). DEFAULT --render-aa none => this block is a NO-OP (curv_feats_np unchanged +
    # coord_feats_fine_mx None) => BYTE-IDENTICAL. Two AA modes: (ipe) attenuate the curvelet basis
    # columns by the mip-NeRF cone footprint (analytical, base grid; touches ONLY the curvelet feats,
    # NOT the self-orient dir feats); (supersample) build a SEPARATE fine-grid feats for the render
    # path only -- the BASE-grid coord_feats/_feats_np_for_pair stay base-grid so the eikonal/sdf(cf)
    # reshape to (render_h, render_w) is unaffected; _render_R dispatches to render_aa_through_R_mlx. ----
    render_aa = str(getattr(args, "render_aa", "none"))
    aa_ss = int(getattr(args, "aa_supersample", 1))
    coord_feats_fine_mx = None  # (supersample only) the fine-grid render feats; None => point-sample
    if render_aa == "ipe":
        from tac.boundary_math.aa_sdf_observation_render import (
            apply_ipe_attenuation,
            ipe_curvelet_attenuation,
            ipe_footprint_sigma,
        )
        _aa_sx, _aa_sy = ipe_footprint_sigma(render_h, render_w, float(args.aa_ipe_footprint))
        _aa_att = ipe_curvelet_attenuation(B, _aa_sx, _aa_sy)
        curv_feats_np = apply_ipe_attenuation(curv_feats_np, _aa_att).astype(np.float32)  # (P, 2*cols)
        print(json.dumps({"stage": "render_aa_ipe", "footprint": float(args.aa_ipe_footprint),
                          "sigma_x": round(float(_aa_sx), 4), "sigma_y": round(float(_aa_sy), 4),
                          "note": "curvelet basis attenuated (mip-NeRF cone); base grid; ~0 compute"}), flush=True)
    elif render_aa == "supersample" and aa_ss > 1:
        # Fail-closed on the un-wired combinations (NO-FAKE: no silent wrong result). The fine-grid
        # self-orient per-pair dir-feat recompute + the structured-init render-res==L*-res invariant
        # are not yet wired; refuse rather than render on mismatched feats.
        #
        # #224 Wave D AA CORRECTION (aa_feasibility_reconciliation_20260702.md): supersample is NOT
        # the launch AA — it is train-only (neither shipped inflate applies ss → train/decode
        # observation MISMATCH), its fp64 decode is 41min > the 30min budget, AND it HURTS the witness
        # −49% (the 0.00086 floor is a REAL-FRAME ceiling, not witness-realized). The launch config
        # ships --render-aa none + the analytic coverage-integrated --lane-render-band. This whole
        # supersample+self-orient fail-closed path therefore never fires from the all-levers launch;
        # it stays BUILT + fail-closeable for research only. Memory arithmetic below is RESOLVED
        # (reconciliation Q3) but MOOT for the launch given the decode + witness-harm disqualification.
        #
        # #224 Option-B DECISION (FAIL-CLOSED, precise n600 blocker — NOT a shape/impl gap):
        # AA-supersample + --self-orient needs PER-PAIR fine-grid feats = curvelet(coords_fine) ⊕
        # dir_feats_fine(pair), where dir_feats_fine is the spec's argmax-NN-upsample→fine-EDT→
        # directional-Fourier (docs/aa_sdf_observation_render_wire_in_spec.md). Reconciled n600 memory
        # (Q3), ss=2, n_dir_freqs=2 (the shipped config), 384×512:
        #   (a) The fine CURVELET feats are pair-INDEPENDENT → ONE SHARED tensor (~0.23GB), NOT
        #       per-pair. ONLY the fine DIR-feats are per-pair: 25.2 MB/pair @ ndf2 × 600 = ~14GB
        #       (full mode) — NOT the ~164GB the pre-reconciliation comment feared (that was the NAIVE
        #       full-fine-feats-per-pair @ ndf6 over-estimate). Peak ≈ 63GB (fine 14 + base cf_mx_cache
        #       ~41 [held STEADY via the in-place rebuild, L~2411, not 2×] + fwd ~8) → memory-SAFE on
        #       the 128GB M5 Max, but this is a SCALED EXTRAPOLATION (24MB/pair measured), not a real
        #       n600 allocation.
        #   (b) ON-DEMAND fine feats (no fine cache, memory-safe): recompute the fine EDT per render
        #       call. The base path amortizes P EDTs across --reorient-every (~50) epochs via the
        #       cache; on-demand does P fine (ss^2×) EDTs EVERY epoch (~50× more, 4× larger) =>
        #       minutes/epoch of scipy EDT over thousands of epochs => non-n600-viable wall-clock.
        # Neither the cache-memory budget nor the on-demand wall-clock can be measured under
        # CONTAINMENT (no GPU). Per CLAUDE.md OPERATOR PRIORITY (fail-closed when "can't be verified
        # correct without a GPU run") this lever stays fail-closed with THIS precise blocker rather
        # than shipping an unverified / non-n600-viable path. WIRED self-orient-compatible AA/lane
        # alternatives (use these for the from-scratch launch): --render-aa ipe (basis-level cone AA,
        # touches ONLY the shared curvelet columns, self-orient-compatible, ~0 compute) AND/OR
        # --lane-render-band (class-1 render authority, NOW self-orient-composable per the Option-B
        # lane-band wire-in below). AA-supersample WITHOUT --self-orient also still works.
        _aa_fine_mode = str(getattr(args, "aa_self_orient_fine_mode", "refuse"))
        if use_self_orient and _aa_fine_mode == "refuse":
            # FAIL-CLOSED default (Wave B SHARPENED, MEASURED blocker). The per-pair fine-grid dir-feats
            # (argmax→ss*grid→fine-EDT→directional-Fourier, docs/aa_sdf_observation_render_wire_in_spec.md)
            # face a MEASURED memory↔wall-clock tradeoff that cannot be BOTH-satisfied AND n600-validated
            # under the no-launch CONTAINMENT (measured local-MLX, ss=2, 384x512):
            #   * fine-EDT recompute = ~49 ms/pair; per-pair fine dir-feat = 25.2 MB @ n_dir_freqs=2
            #     (the shipped config) — the older 75.5 MB was the ndf6 figure (reconciliation Q3).
            #   * MEMORY-SAFE (--aa-self-orient-fine-mode batch): a batch-bounded on-demand cache is
            #     ~cap*25MB (0.2 GB @ cap=8 vs ~14 GB all-600 @ ndf2) => memory SOLVED. BUT every pair
            #     renders every epoch, so a batch-bounded cache THRASHES => P fine-EDTs/epoch ~29 s/epoch
            #     @ n600 (50x the base --reorient-every amortization) => wall-clock NON-viable for the
            #     multi-thousand-epoch CE→tau→l7→Muon curriculum.
            #   * WALL-CLOCK-viable (--aa-self-orient-fine-mode full): compute the fine dir-feats ONCE
            #     per --reorient-every (amortized ~0.6 s/epoch) BUT store all P => ~14 GB @ ss=2, ndf2
            #     (the fine curvelet feats are pair-independent → ONE shared ~0.23GB tensor, NOT per-pair);
            #     peak ≈ 63 GB (fine 14 + base cf_mx_cache ~41 held STEADY via the in-place rebuild + fwd
            #     ~8). This is a SCALED EXTRAPOLATION (24MB/pair measured); MOOT for the launch (supersample
            #     is disqualified by the decode-budget + −49% witness-harm per the Wave D header above).
            # Both opt-in modes ARE now BUILT + small-MLX-verified (render finite+shape; memory scales
            # ~batch); the DEFAULT stays fail-closed so no unverified OOM / 50x-slow n600 run fires by
            # accident. This is THE operator's-call item: pick `full` after an n600 memory-fit check, or
            # `batch` if the extra CPU-EDT wall-clock is acceptable. Self-orient-compatible alternatives
            # that ARE fully wired: --render-aa ipe (basis-level cone AA, ~0 compute) and/or
            # --lane-render-band; AA-supersample WITHOUT --self-orient also works.
            raise ValueError(
                "--render-aa supersample + --self-orient is fail-closed by DEFAULT (Wave B). The fine "
                "dir-feat path is BUILT + verified; enable it explicitly with "
                "--aa-self-orient-fine-mode full (wall-clock-viable, ~14GB@ss2n600 @ndf2, peak ~63GB — "
                "validate the n600 memory fit first) OR --aa-self-orient-fine-mode batch (memory-safe ~cap*25MB, but "
                "~P fine-EDTs/epoch ~29s@n600). Or use --render-aa ipe / --lane-render-band (both "
                "self-orient-compatible + fully wired), or AA-supersample WITHOUT --self-orient.")
        # #224 Wave C FIX-2: supersample + --structured-init is NOW WIRED (was fail-closed as
        # "not-yet-wired", NOT proven-incompatible). The two operate on DIFFERENT grids and compose:
        # structured-init pretrains the coord-INR witness weights on the BASE grid against the cached L*
        # (its invariant `(render_h,render_w) == lstar_shape` is checked at the structured-init block
        # below and is UNCHANGED by supersample — aa_ss multiplies only the internal fine render grid,
        # NOT render_h/render_w). The fine render then evaluates the SAME shared weights at fine coords
        # (a coord-INR generalizes across coordinate resolution by construction). Verified at small-MLX
        # n4 (ss=2 + self-orient full + structured-init + lane-prior-phi1: finite render + descent). Per
        # the Wave B precedent (LEVER 3 relaxed the self-orient guard behind an opt-in after BUILD +
        # small-MLX verify). The REAL render==L* invariant stays enforced at the structured-init block.
        print(json.dumps({"stage": "render_aa_supersample_structured_init",
                          "structured_init": bool(args.structured_init),
                          "note": "supersample composes with structured-init: base-grid pretrain + "
                          "shared coord-INR weights evaluated at fine coords (render_h/w == L* unchanged)"}),
              flush=True)
        from tac.boundary_math.aa_sdf_observation_render import build_supersampled_coords
        _coords_fine = build_supersampled_coords(render_h, render_w, aa_ss)          # (ss^2*P, 2)
        coord_feats_fine_mx = mx.array(curvelet_feats(_coords_fine, B).astype(np.float32))
        if use_self_orient:
            # opt-in fine self-orient (batch|full). _cf_fine_mx (below) sources per-pair fine dir-feats;
            # rebuilt/invalidated at each reorient. Pre-first-reorient -> zeros -> pure-curvelet fine.
            _aa_so_fine = True
            _aa_coords_fine = _coords_fine
        print(json.dumps({"stage": "render_aa_supersample", "ss": aa_ss,
                          "fine_grid": [render_h * aa_ss, render_w * aa_ss],
                          "self_orient_fine_mode": (_aa_fine_mode if use_self_orient else "n/a"),
                          "note": "separate fine-grid render feats; base-grid eikonal/sdf unaffected"}), flush=True)
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
    # #218 facet-1a — fixed simplex-ETF head (Yang et al. 2022, neural-collapse optimal). Replaces the
    # LEARNED out_sdf weight with a deterministic simplex ETF (equal-norm, max-equiangular K prototypes)
    # and FREEZES it: removes the minority-class NORM COLLAPSE that erases Lane/Movable, AND is
    # regenerable from a fixed seed at inflate => the K x d head weight is FREE (rate win). out_sdf.bias
    # stays trainable. args.head != "etf" (default) => untouched => byte-identical.
    if str(getattr(args, "head", "softmax")) == "etf":
        from tac.boundary_math.laguerre_logit_offset import etf_gram_offdiag, simplex_etf
        _etf_w = simplex_etf(5, args.hidden_dim).astype(np.float32)
        model.out_sdf.weight = mx.array(_etf_w)
        model.out_sdf.freeze(keys=["weight"])
        mx.eval(model.parameters())
        print(json.dumps({"stage": "head_etf", "offdiag_cos": round(float(etf_gram_offdiag(_etf_w)), 4),
                          "target_cos": round(-1.0 / 4.0, 4), "frozen_weight": True}), flush=True)
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
    # ---- #224 (3) warp-real-luma frame0 POSE CARRIER build + CHILD-ATTACH (BEFORE EMA/opt so the
    # EMA shadow + AdamW/Muon state + checkpoints all track the carrier residual through the SAME
    # machinery). DEFAULT OFF (--pose-carrier) => no attach => model.trainable_parameters() unchanged
    # => value_and_grad/opt/ema BYTE-IDENTICAL. The RENDER dispatch (even code=f0->carrier warp,
    # odd=f1->witness) is wired below at the render-fn assembly (replacing the old fail-closed guard).
    # The residual co-grad rides the ONE nn.value_and_grad(model, ...) (probe-verified: child dxi gets
    # a finite grad; the carrier's self.freeze(["xi_stored"]) keeps the stored twist out of the
    # trainable tree under parent recursion, so the optimizer never corrupts it).
    pose_carrier = None
    pose_carrier_geom = None
    pose_carrier_xi_stored = None
    if bool(getattr(args, "pose_carrier", False)):
        if bool(getattr(args, "freeze_decoder_fit_codes", False)):
            raise ValueError(
                "--pose-carrier is incompatible with --freeze-decoder-fit-codes: the decoder freeze "
                "runs BEFORE the carrier attach and its trainable-set assertion (only 'code') would "
                "either fail or freeze the carrier residual. Run them separately.")
        if float(args.w_pose) <= 0.0:
            raise ValueError(
                "--pose-carrier requires --w-pose > 0: the residual dxi trains ONLY on the realized "
                "d_pose term; with w_pose=0 the carrier stays at the stored-twist init (no co-grad).")
        from tac.boundary_math.warp_real_luma_frame0 import (
            GroundHomographyGeom as _PCGeom,
            WarpRealLumaFrame0Carrier as _PCCarrier,
            warp_frame0_uint8_numpy as _pc_warp_uint8_np,
            xi_from_pose_calibration as _pc_xi_from_calib,
        )
        _pc_nat_h, _pc_nat_w = int(np.asarray(gt.gt_f0[0]).shape[0]), int(np.asarray(gt.gt_f0[0]).shape[1])
        pose_carrier_geom = _PCGeom.eon(native_hw=(_pc_nat_h, _pc_nat_w), pitch=float(args.pose_carrier_pitch))
        _pc_sr = float(args.pose_carrier_s_r)
        _pc_pitch = float(args.pose_carrier_pitch)
        if args.pose_carrier_s_t is not None:
            _pc_st = float(args.pose_carrier_s_t)
            _pc_fit = None
        else:
            # self-calibrating s_t fit on the frozen CPU-torch PoseNet d_pose grid (mirrors
            # tools/measure_warp_real_luma_frame0_dpose): deterministic, GT-derived, NEVER MPS.
            _pc_nf = max(1, min(int(args.pose_carrier_fit_pairs), P))
            _pc_grid = [0.0, 0.02, 0.044, 0.08, 0.12, 0.16, 0.22, 0.30]

            def _pc_mean_dpose(_st: float) -> float:
                f0s = [np.asarray(gt.gt_f0[p]) for p in range(_pc_nf)]
                preds = [_pc_warp_uint8_np(
                    f0s[p], _pc_xi_from_calib(np.asarray(gt.gt_poses[p]), _st, _pc_sr, _pc_pitch),
                    pose_carrier_geom) for p in range(_pc_nf)]
                dps = cpu_verdict_d_pose_batch(
                    posenet_cpu, f0s, preds, [np.asarray(gt.gt_poses[p]) for p in range(_pc_nf)])
                return float(np.mean(dps))

            _pc_fit = {s: _pc_mean_dpose(s) for s in _pc_grid}
            _pc_st = float(min(_pc_fit, key=_pc_fit.get))
        pose_carrier_xi_stored = np.stack([
            _pc_xi_from_calib(np.asarray(gt.gt_poses[p]), _pc_st, _pc_sr, _pc_pitch)
            for p in range(P)]).astype(np.float32)
        _pc_code_dim = int(args.mod_dim) if str(args.pose_carrier_residual_mode) == "film" else None
        pose_carrier = _PCCarrier.build(
            pose_carrier_xi_stored, pose_carrier_geom,
            residual_mode=str(args.pose_carrier_residual_mode),
            residual_scale=float(args.pose_carrier_residual_scale),
            code_dim=_pc_code_dim, film_hidden=32)
        mx.eval(pose_carrier.parameters())
        model.pose_carrier = pose_carrier.impl   # child-attach: dxi joins model.trainable_parameters()
        mx.eval(model.parameters())
        print(json.dumps({"stage": "pose_carrier", "residual_mode": str(args.pose_carrier_residual_mode),
                          "source": str(getattr(args, "pose_carrier_source", "real_keyframe")),
                          "s_t": round(_pc_st, 5), "s_r": _pc_sr, "pitch": _pc_pitch,
                          "s_t_fit": ({str(k): round(v, 3) for k, v in _pc_fit.items()} if _pc_fit else None),
                          "native_hw": [_pc_nat_h, _pc_nat_w], "n_pairs": P,
                          "note": (("STORE-NOTHING: frame0 = warp(witness's OWN render, xi); stores ONLY "
                                    "xi/H (~0 marginal bytes)" if str(getattr(args, "pose_carrier_source",
                                    "real_keyframe")) == "generated" else
                                    "frame0 real-luma SE(3)-warp pose carrier (stored keyframe)")
                                   + "; residual co-grad via child-attach (ONE value_and_grad + opt + EMA); "
                                     "advisory; pointer 0.19110 UNMOVED")}),
              flush=True)

    ema = MlxEMA(model, decay=args.ema_decay)
    # (THETA* TIER-2 MUST-3) SWA / wider-finisher EMA. DEFAULT-OFF: --ema-decay-finisher None =>
    # ema_finisher_decay None => the loop NEVER mutates ema.decay => the EMA trajectory is
    # BIT-IDENTICAL to the --ema-decay path. When set, from the resolved finisher-start epoch onward
    # the EMA update uses the WIDER decay (averages over the late oscillation -> a flat-basin center,
    # SWA-style). Start resolves to --ema-decay-finisher-start-epoch, else --muon-start-epoch (the
    # natural finisher boundary). main() guards range + start-resolvability when the decay is set.
    ema_finisher_decay = (float(args.ema_decay_finisher)
                          if getattr(args, "ema_decay_finisher", None) is not None else None)
    ema_finisher_start = (int(args.ema_decay_finisher_start_epoch)
                          if getattr(args, "ema_decay_finisher_start_epoch", None) is not None
                          else (int(args.muon_start_epoch) if args.muon_start_epoch is not None else None))
    # #224 Wave C FIX-1: bias_correction ON only on the high-beta2 all-levers path (0.9999999); at the
    # 0.999 default it stays MLX-default False => BYTE-IDENTICAL. Without it high beta2 => ~100x step-1
    # LR blowup => divergence (see _adam_bias_correction_for).
    _adam_bc = _adam_bias_correction_for(getattr(args, "adam_beta2", 0.999))
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay,
                      betas=[0.9, float(getattr(args, "adam_beta2", 0.999))],
                      bias_correction=_adam_bc)

    # ---- RESIDUAL-ONLY MODE (v2 hybrid; gap #1). Load the FIXED deterministic bulk + the
    # bulk-derived composition mask, and build the compose hooks. The bulk arrays live in CLOSURE
    # SCOPE (NOT model attributes) -> they are NEVER in model.parameters() => the EMA / optimizer /
    # quantized blob / checkpoints see ONLY the INR (the bulk does NOT ship; THAT is the rate win).
    # Every realized render (loss + levers + verdict) routes through ``_render_R`` / ``_compose_np``
    # so the d_seg loss is on the COMPOSED witness (bulk (+) INR). Default OFF => _render_R is the
    # bare render + _compose_np is None => byte-identical to the full-partition witness.
    residual_mode = bool(getattr(args, "residual_mode", False))
    _compose_np = None
    _render_R = render_through_R_mlx
    if residual_mode:
        from tac.v2_compose.residual_compose import load_residual_training_bundle

        _rb = load_residual_training_bundle(Path(args.residual_target_npz))
        if (_rb.render_h, _rb.render_w) != (render_h, render_w):
            raise ValueError(
                f"--residual-target-npz render res {(_rb.render_h, _rb.render_w)} != "
                f"--render-h/--render-w {(render_h, render_w)}: the composition is elementwise at "
                "render res, so they MUST match.")
        if _rb.n_pairs < P:
            raise ValueError(
                f"--residual-target-npz has {_rb.n_pairs} pairs < --num-pairs {P}: the bundle must "
                "cover every trained pair (a larger bundle is fine -- the first P are used).")
        _bulk_rgb_np = np.asarray(_rb.bulk_rgb_render_res[:P], np.float32)   # (P,H,W,3) pre-R RGB
        _resid_mask_np = np.asarray(_rb.composition_mask[:P], bool)          # (P,H,W) override region
        _bulk_rgb_mx = mx.array(_bulk_rgb_np)                                # (P,H,W,3)
        _resid_mask_mx = mx.array(_resid_mask_np.astype(np.float32))[..., None]  # (P,H,W,1)

        def _compose_mx(rgb_nhwc, code_idx):
            # composed = where(mask, INR, bulk) = bulk*(1-m) + INR*m. ``code_idx`` is the per-frame
            # index; the bulk frame is shared across f0/f1 of a pair => pair = code_idx // 2. The
            # bulk is a CONSTANT (no grad) => gradients flow ONLY through the masked residual region.
            pair = int(code_idx) // 2
            m = _resid_mask_mx[pair][None]                # (1,H,W,1)
            return _bulk_rgb_mx[pair][None] * (1.0 - m) + rgb_nhwc * m

        # (#224) the per-frame _render_R that chains _compose_mx is built in the UNIFIED RENDER
        # PATH block below (so residual bulk composes with the analytic-lane band + AA supersample).
        def _compose_np(rgb_hw3, pi):  # noqa: F811 (residual override)
            m = _resid_mask_np[pi][..., None]             # (H,W,1)
            return np.where(m, np.asarray(rgb_hw3, np.float32), _bulk_rgb_np[pi])

        print(json.dumps({"stage": "residual_mode", "npz": str(args.residual_target_npz),
                          "n_pairs": int(_rb.n_pairs), "learn_classes": list(_rb.learn_classes),
                          "dilate": int(_rb.dilate),
                          "composition_override_frac": float(_resid_mask_np.mean()),
                          "note": "INR trains on the COMPOSED-render d_seg (bulk (+) INR); the bulk "
                          "is OUTSIDE the counted weights (rate win). advisory; pointer UNMOVED 0.19110"}),
              flush=True)

    # =====================================================================================
    # #224 UNIFIED RENDER PATH — compose the render-side levers onto _render_R (the per-frame
    # realized render used by make_loss_fn(render_fn=...) AND the shared seg-lever forward). DEFAULT
    # (no residual / no AA / no lane-band / no pose-carrier) => _render_R stays render_through_R_mlx +
    # render_fn=None => BYTE-IDENTICAL. Each lever is opt-in. Docs: analytic_lane_render_band /
    # aa_sdf_observation_render wire-in specs.
    # -------------------------------------------------------------------------------------
    _aa_on = (render_aa == "supersample" and aa_ss > 1)
    _band_active = bool(getattr(args, "lane_render_band", False))
    if _band_active and _aa_on:
        raise ValueError(
            "--lane-render-band + --render-aa supersample are not wired together: the band coverage is "
            "base-grid (H,W) but the AA compose happens at the ss*grid. Use --render-aa ipe (base grid) "
            "with the band, or run them separately.")
    # (2) analytic-lane render-band compose_fn (FEED-dv #203/#213/#215). Precompute the per-code
    # LaneBandPrior ONCE from the frozen GT class-1 mask; ride the witness margin (#141) as the
    # FP-killer uncertainty gate. compose_fn coverage/u_mask are stop-grad constants; the gradient
    # flows through the witness rgb + the witness-derived lane appearance.
    band_compose_fn = None
    band_gate = {"on": False}
    _band_start = int(getattr(args, "lane_band_start_epoch", 300))
    if _band_active:
        # #224 Option-B WIRE-IN (self-orient composable): BOTH the witness margin provider
        # (call_margin) AND the lane RGB provider (render_lane_appearance) feed the model in_proj,
        # which expects base+dir_w feats when --self-orient is on. The pre-Option-B code hardcoded
        # the shared no-self-orient coord_feats_mx -> MLX matmul shape crash under --self-orient at
        # --lane-band-start-epoch. FIX: feed the PER-PAIR self-orient feats (base curvelet + this
        # pair's live dir feats) via _band_feats(code_idx) below (mirrors _cf_mx). NO-FAKE: when
        # --self-orient is OFF this returns the SAME shared coord_feats_mx object (numerically
        # byte-identical to the pre-Option-B measured no-self-orient band path); when ON it returns
        # mx.array(_feats_np_for_pair(pair)) = base curvelet ⊕ this pair's dir feats (zeros pre-first-
        # reorient -> pure-curvelet width base+dir_w -> correct in_proj shape from epoch 0). Sister
        # of the --render-aa supersample + --self-orient wire-in below.
        def _band_feats(code_idx):
            # base-grid per-pair coord feats for the band providers, via the canonical _cf_mx accessor
            # (late-bound; _cf_mx + cf_mx_cache are defined below at main scope and exist by the time
            # band_compose_fn calls this during training). _cf_mx returns the shared coord_feats_mx
            # when no self-orient (exact-object-identical to the measured path) and the already-synced
            # per-pair cf_mx_cache[pi] when self-orient -- BIT-IDENTICAL to mx.array(_feats_np_for_pair
            # (pi)) (rebuild_per_pair_feats_in_place guarantees it) but REUSES the cache rebuilt after
            # every reorient instead of a fresh full-res np.concatenate + mx.array per call (senior-
            # review efficiency fix: kills ~2400 redundant full-res rebuilds/epoch once the band gate
            # opens at --lane-band-start-epoch; serves the shortest-train / MLX-first discipline).
            return _cf_mx(int(code_idx) // 2)
        from tac.boundary_math.analytic_lane_render_band import (
            build_analytic_lane_band_prior,
            make_lane_band_compose_fn,
        )
        _lane_priors: dict[int, Any] = {}
        for _pi in range(P):
            _prior = build_analytic_lane_band_prior(
                np.asarray(gt.lstars[_pi]), lane_cls=1, softness=float(args.lane_band_softness),
                dash_gate=True, dash_forward_max_m=float(args.lane_band_dash_forward_max_m))
            _lane_priors[2 * _pi + 1] = _prior   # frame1 (the SegNet-scored frame)
            _lane_priors[2 * _pi] = _prior       # frame0 seg-free; keep symmetric
        _u_src = str(args.lane_band_uncertainty_source)
        if _u_src == "witness":
            def _band_margin_provider(code_idx):
                # per-pair self-orient feats (base curvelet ⊕ this pair's dir feats) via _band_feats;
                # == shared coord_feats_mx when --self-orient is OFF (measured no-self-orient config).
                return mx.stop_gradient(
                    model.call_margin(_band_feats(code_idx), int(code_idx))).reshape(render_h, render_w)
            _margin_provider: Any = _band_margin_provider
        elif _u_src == "gt":
            _margin_provider = {c: mx.array(np.asarray(gt.margins[c // 2], np.float32)) for c in _lane_priors}
        else:
            _margin_provider = None

        def _band_lane_rgb(code_idx):
            return model.render_lane_appearance(_band_feats(code_idx), int(code_idx), lane_cls=1).reshape(
                render_h, render_w, 3)

        band_compose_fn = make_lane_band_compose_fn(
            _lane_priors, lane_rgb_provider=_band_lane_rgb, margin_provider=_margin_provider,
            tau=float(args.lane_band_tau), eps=float(args.lane_band_eps),
            weight=float(args.lane_band_weight), use_mlx=True)
        band_gate["on"] = _band_start <= 1
        _band_recalls = [float(_lane_priors[2 * pi + 1].band_recall) for pi in range(P)
                         if np.isfinite(_lane_priors[2 * pi + 1].band_recall)]
        print(json.dumps({"stage": "lane_render_band", "n_pairs": P, "uncertainty_source": _u_src,
                          "start_epoch": _band_start,
                          "band_vs_gt_lane_recall_mean": (round(float(np.mean(_band_recalls)), 4)
                                                          if _band_recalls else None),
                          "note": "class-1 render-time authority composited PRE-R; gated at start_epoch "
                          "(spike-guard re-treat); advisory; pointer 0.19110 UNMOVED"}), flush=True)
    # #224 (5) island SEED compose state (LATE-BOUND; populated at the seed build below, which runs
    # AFTER this chain is defined but BEFORE value_and_grad + the training loop, so _compose_chain
    # reads it at CALL time). The seed is a SEPARATE module (own optimizer group) -> NOT in
    # model.parameters()/EMA/blob/deploy, so the verdict (witness-alone) == the 0-byte-accelerant
    # deploy (NO-FAKE, honestly measured). Default OFF (--seed-islands) => seed_state stays empty =>
    # the seed branch never fires => _compose_chain BYTE-IDENTICAL.
    seed_on = bool(getattr(args, "seed_islands", False))
    seed_state: dict[str, Any] = {"mod": None, "masks": None}
    # assemble the compose chain (residual bulk FIRST, then lane band, then island seed). None => bare.
    _use_chain = residual_mode or _band_active or seed_on

    def _compose_chain(rgb_nhwc, code_idx):
        if residual_mode:
            rgb_nhwc = _compose_mx(rgb_nhwc, code_idx)
        if _band_active and band_gate["on"]:
            rgb_nhwc = band_compose_fn(rgb_nhwc, code_idx)
        if seed_state["mod"] is not None and (int(code_idx) % 2 == 1):
            # frame1 (SegNet-scored) ONLY: add the protected per-pair island seed residual (masked to
            # the self-detected island support). Reads the LIVE seed_mod.residual -> the dual
            # value_and_grad co-differentiates it; the compose flows through the SHARED _f1 -> _slog ->
            # _signed (no 2nd SegNet). frame0 (even) is seg-free -> unseeded.
            _pi = int(code_idx) // 2
            rgb_nhwc = rgb_nhwc + seed_state["mod"].residual[_pi] * seed_state["masks"][_pi]
        return rgb_nhwc

    # (1) AA supersample render dispatch: IGNORE the passed base-grid coord_feats and use the
    # fine-grid feats (the base-grid eikonal/sdf(cf) in total_loss_fn is unaffected -> still base grid).
    if _aa_on or _use_chain:
        from tac.boundary_math.aa_sdf_observation_render import (  # noqa: E402
            render_aa_through_R_mlx as _render_aa_R,
        )

        def _render_R(witness, coord_feats, code_idx, rh, rw):  # noqa: F811 (unified #224 override)
            _cf = _compose_chain if _use_chain else None
            if _aa_on:
                # per-pair FINE feats when --self-orient (shared curvelet-fine [+ fine dir-feats]);
                # else the shared curvelet-fine tensor. _cf_fine_mx is late-bound (defined below).
                _feats_fine = _cf_fine_mx(int(code_idx) // 2) if _aa_so_fine else coord_feats_fine_mx
                return _render_aa_R(witness, _feats_fine, code_idx, rh, rw, aa_ss, compose_fn=_cf)
            return render_through_R_mlx(witness, coord_feats, code_idx, rh, rw, compose_fn=_cf)

    render_fn = _render_R if (_aa_on or _use_chain) else None

    # (3) warp-real-luma frame0 pose carrier — the parity-dispatch render_fn (even code=f0 -> the
    # SE(3) ground-homography warp of the REAL keyframe luma; odd=f1 -> witness). The carrier BUILD +
    # child-ATTACH (residual co-grad through the ONE value_and_grad + AdamW/Muon + EMA) happened
    # ABOVE, pre-EMA; here we only WRAP render_fn with the parity dispatch. The measured s_t/s_r/pitch
    # calibration is self-fit at build (or --pose-carrier-s-t); the residual dxi co-grad rides the
    # child-attach. Default OFF (pose_carrier is None) => render_fn unchanged => byte-identical.
    if pose_carrier is not None:
        _pc_witness_render = _render_R if (_aa_on or _use_chain) else render_through_R_mlx
        _pc_source_generated = str(getattr(args, "pose_carrier_source", "real_keyframe")) == "generated"

        _pc_code_provider = None
        if str(args.pose_carrier_residual_mode) == "film":
            def _pc_code_provider(pi: int):
                return model.code[2 * pi + 0]   # frame0 per-pair code for the FiLM residual MLP

        if _pc_source_generated:
            # #205 STORE-NOTHING-but-xi: frame0 = warp(the witness's OWN plain frame0 render, xi_eff).
            # NO stored keyframe -> stores ONLY xi/H (~0 marginal bytes; the render is FREE, rule-118).
            # The plain (no-compose) witness f0 render is up-sampled to camera-native (the R "up" step,
            # == the byte-close store_nothing warp source _R), then the carrier warps it + R-downs to
            # SEG. The dxi residual co-grads THROUGH the witness f0 render (the co-adaptation).
            from tac.local_acceleration.pr95_hnerv_mlx_training import (
                CAMERA_HW as _PC_CAMERA_HW,
                apply_contest_faithful_roundtrip_nhwc as _pc_up_to_camera,
            )
            _pc_impl = pose_carrier.impl

            def render_fn(model, coord_feats, code_idx, rh, rw):
                if int(code_idx) % 2 == 1:  # f1 -> witness render (drives d_seg)
                    return _pc_witness_render(model, coord_feats, code_idx, rh, rw)
                # f0 -> STORE-NOTHING: the witness's OWN plain frame0 render, up to camera-native, warped.
                pair_idx = int(code_idx) // 2
                rgb = mx.reshape(model(coord_feats, code_idx), (1, rh, rw, 3))
                src_native = _pc_up_to_camera(rgb, output_hw=_PC_CAMERA_HW, ste_round=True)[0]
                code_vec = model.code[2 * pair_idx] if (_pc_code_provider is not None) else None
                return _pc_impl.render_f0(src_native, pair_idx, code_vec, ste_round=True)
        else:
            def _pc_gt_f0_provider(pi: int):
                # native-res (H,W,3) REAL keyframe luma as mx float32; per-call (transient, no P-length
                # fp32 cache -> n600-memory-safe; the uint8 GT already resides in gt.gt_f0).
                return mx.array(np.asarray(gt.gt_f0[pi], np.float32))

            render_fn = pose_carrier.make_pair_render_dispatch(
                _pc_witness_render, _pc_gt_f0_provider, code_provider=_pc_code_provider)
        print(json.dumps({"stage": "pose_carrier_render_dispatch", "residual_mode": str(args.pose_carrier_residual_mode),
                          "source": ("generated" if _pc_source_generated else "real_keyframe"),
                          "witness_render": ("aa/chain" if (_aa_on or _use_chain) else "bare"),
                          "note": ("STORE-NOTHING: frame0 = warp(witness's OWN render, xi); stores ONLY xi/H"
                                   if _pc_source_generated else
                                   "parity dispatch (even code=f0->carrier warp of the stored real keyframe, "
                                   "odd=f1->witness)") + "; advisory; pointer 0.19110 UNMOVED"}), flush=True)

    base_loss = make_loss_fn(
        adapter, render_h, render_w, score_domain=args.score_domain_loss, pose_eps=args.pose_eps,
        seg_loss=args.seg_loss, tau_softplus_tau=args.tau_softplus_tau, l7_mult=args.l7_mult,
        l7_threshold=args.l7_threshold,
        render_fn=render_fn,
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
    # LEVER-4 REACHABILITY (default-off; REPLACES the texture path when on): per-pair through-R
    # margin-Jacobian S_R weight. _sR_provider is a list[mx.array (1,H,W)] indexed by pi==int(c1)//2
    # (same key as island_weight_mx); it is POPULATED after lstar_cache is built (see the build site)
    # ONLY when msal_reach AND msal_w>0. Declared None here so the closure name always exists and the
    # OFF path (msal_reach False) NEVER references it => byte-identical to the pre-reachability code.
    msal_reach = bool(getattr(args, "margin_saliency_reachability", False))
    _sR_provider: Any = None

    # LEVER-4b (SUB-PIXEL BOUNDARY-PLACEMENT `t`, DIRECTIONAL upgrade of the scalar margin-saliency
    # #141; asymmetry probe a8afad40 GREEN 2026-07-03). The cross-boundary GT margin RATIO
    # t = M_GT[p] / (M_GT[p] + M_GT[q]) (p,q = the two straddle pixels across an inter-class edge) is a
    # FREE sub-pixel boundary-POSITION localizer LATENT in the already-computed GT margin field (no
    # SegNet forward; pure numpy from gt.margins/gt.lstars). It upgrades LEVER-4's DIRECTIONLESS
    # per-pixel weight -> a SIGNED sub-pixel placement TARGET: where the GT margin V is genuine, supervise
    # the witness's OWN realized margin ratio t_wit = Mw[p]/(Mw[p]+Mw[q]) toward the GT t (a DENSER,
    # sub-pixel, differentiable signal than the argmax weight). Reuses the SHARED realized through-R
    # margin ``_signed`` (Mw = relu(_signed) = witness GT-class margin, the honest mirror of the GT
    # top1-top2 the target is built from) -- NO 2nd SegNet forward (bit-identical to LEVER-4's forward,
    # ``_seg_levers_on`` gated). subpix_w=0.0 (DEFAULT) => the branch is skipped => byte-identical (fully
    # additive). Providers declared None here (closure binds the cells) so the OFF path never references
    # them; POPULATED after lstar_cache is built (spike-map style, inline -- theta-independent + cheap).
    subpix_w = float(getattr(args, "seg_subpix_boundary_weight", 0.0))
    subpix_start = int(getattr(args, "seg_subpix_boundary_start_epoch", 0))
    subpix_band = float(getattr(args, "seg_subpix_boundary_v_band", 1.0))
    subpix_eps = 1e-6
    subpix_gate = {"on": subpix_start <= 1}
    _subpix_t_prov: Any = None     # list[mx.array (1,H,W)] f32, GT t in [0,1] where active, -1.0 sentinel
    _subpix_dir_prov: Any = None   # list[mx.array (1,H,W)] f32 in {0,1}, dominant-straddle dir (0=right,1=down)

    # (THETA* TIER-2 MUST-2) nuclear-norm low-rank code penalty closure constants. code_nuc_w=0.0
    # (DEFAULT) => the branch in total_loss_fn is skipped => L is byte-identical (fully additive).
    code_nuc_w = float(getattr(args, "code_nuclear_weight", 0.0))
    code_nuc_eps = float(getattr(args, "code_nuclear_eps", 1e-3))
    code_nuc_iters = int(getattr(args, "code_nuclear_ns_iters", 25))
    # (THETA* TIER-2 STRETCH-1) junction-aware Eikonal relax closure constants. eik_jrelax=0.0
    # (DEFAULT) => _eikonal_length_mlx takes its BIT-IDENTICAL branch (w==1.0) => unchanged.
    eik_jrelax = float(getattr(args, "eikonal_junction_relax", 0.0))
    eik_jtau = float(getattr(args, "eikonal_junction_tau", 0.5))

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

    # #218 MARGIN-FIELD HEAD levers (facets 1b + 3, BYTE-FREE). A REALIZED through-R per-class margin
    # hinge composing with LEVER-3/4/B on the SHARED _signed. mfh_w=0.0 (default) => branch skipped =>
    # L is byte-identical. Per-pixel margin TARGET b_c = additive-margin (facet-1b, when head==
    # additive-margin) + facet-3 Menon boost on RARE classes: tau*relu(-log pi_c) mean-centered so ONLY
    # rare classes (Lane/Movable) RAISE their target (common classes stay at base). Priors from cached
    # GT L* (deterministic; this is a TRAIN-TIME loss shape => 0 archive bytes). facet-1a (ETF head) is
    # applied at model build above and is orthogonal to this lever.
    mfh_w = float(getattr(args, "margin_field_head_weight", 0.0))
    mfh_target_mx = None
    if mfh_w > 0.0:
        from tac.boundary_math.laguerre_logit_offset import menon_logit_adjustment_offsets
        _mfh_counts = np.bincount(
            np.concatenate([np.asarray(gt.lstars[pi]).reshape(-1) for pi in range(P)]),
            minlength=5).astype(np.float64)
        _mfh_base = float(getattr(args, "additive_margin", 0.0)) if str(getattr(args, "head", "softmax")) == "additive-margin" else 0.0
        _mfh_tgt = np.full(5, _mfh_base, np.float64)
        if bool(getattr(args, "logit_adjust_per_class", False)):
            _mfh_tgt = _mfh_tgt + float(getattr(args, "logit_adjust_tau", 1.0)) * np.maximum(
                menon_logit_adjustment_offsets(_mfh_counts, tau=1.0), 0.0)
        mfh_target_mx = mx.array(_mfh_tgt.reshape(1, 1, 1, 5).astype(np.float32))
        print(json.dumps({"stage": "margin_field_head", "weight": mfh_w,
                          "per_class_margin_target": [round(float(v), 4) for v in _mfh_tgt]}), flush=True)

    # #224 (4) PERSISTENCE/TOPOLOGY loss setup (persistence_topology_loss; #218/TopologyLossGauge).
    # persist_w=0 (default) => persist_classes=() + persist_gate["w"]=0 => branch inert => byte-identical.
    persist_w = float(getattr(args, "persistence_loss_weight", 0.0))
    persist_recall_w = float(getattr(args, "persistence_recall_weight", 1.0))
    persist_cldice_iters = int(getattr(args, "cldice_iters", 5))
    persist_warmup = int(getattr(args, "persistence_warmup_epochs", 0))
    persist_classes: tuple[int, ...] = ()
    persist_gate = {"w": 0.0}   # epoch-annealed weight (set in the loop); 0 => branch inert
    if persist_w > 0.0:
        from tac.boundary_math.persistence_topology_loss import (
            detect_persistence_tail_classes,
            persistence_anneal_weight,
            persistence_topology_loss_mlx,
        )
        _pc = str(getattr(args, "persistence_classes", "auto")).strip()
        if _pc.lower() == "auto":
            _lst_stack_p = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)], axis=0)
            persist_classes, _pev = detect_persistence_tail_classes(_lst_stack_p, n_classes=5)
        else:
            persist_classes = tuple(int(x) for x in _pc.split(",") if x.strip() != "")
        print(json.dumps({"stage": "persistence_loss", "target_classes": list(persist_classes),
                          "weight": persist_w, "recall_weight": persist_recall_w,
                          "cldice_iters": persist_cldice_iters, "warmup_epochs": persist_warmup,
                          "note": "soft-clDice + persistence-weighted island recall on the SHARED "
                          "realized seg forward; annealed; advisory; pointer 0.19110 UNMOVED"}), flush=True)

    # (--cache-gt-skeleton, #260) declared here (in the enclosing scope, BEFORE total_loss_fn) so the
    # closure binds the cell; POPULATED after lstar_cache is built (see the build block below). None
    # (default OFF) => total_loss_fn passes sg_precomputed=None => byte-identical to the pre-flag path.
    cache_gt_skeleton = bool(getattr(args, "cache_gt_skeleton", False))
    _sg_cache: dict[int, Any] | None = None

    # #224 (5) ISLAND AMPLIFICATION setup (island_protection; #208/IslandProtectionGauge.AMPLIFY_ONLY).
    # Rides the SHARED LEVER-4 realized margin _signed (#141) -- NO 2nd saliency / SegNet forward.
    # amplify_w=0 (default) => island_weight_mx None => branch skipped => byte-identical.
    amplify_w = float(getattr(args, "amplify_weight", 0.0))
    amplify_form = str(getattr(args, "amplify_form", "hinge"))
    amplify_mtgt = float(getattr(args, "amplify_margin_target", 1.0))
    island_weight_mx: dict[int, Any] | None = None
    if amplify_w > 0.0:
        from tac.boundary_math.island_protection import (
            build_island_masks,
            identify_island_classes,
            island_birth_from_signed_mx,
            island_persistence_weight,
        )
        _lst_stack_i = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)], axis=0)
        _idet = identify_island_classes(_lst_stack_i, n_classes=5)
        island_weight_mx = {}
        for pi in range(P):
            _im = build_island_masks(np.asarray(gt.lstars[pi], np.int64), _idet.lane_cls,
                                     _idet.movable_cls, dilate_px=int(args.island_dilate_px))
            _iw = island_persistence_weight(_im.any_mask, kind=str(args.amplify_persist))
            island_weight_mx[pi] = mx.array(np.asarray(_iw, np.float32)[None])   # (1,H,W)
        print(json.dumps({"stage": "island_amplify", "island_classes": list(_idet.island_classes),
                          "lane_cls": _idet.lane_cls, "movable_cls": _idet.movable_cls,
                          "weight": amplify_w, "form": amplify_form, "margin_target": amplify_mtgt,
                          "persist": str(args.amplify_persist),
                          "note": "island-birth rides the SHARED realized _signed margin (#141); "
                          "advisory; pointer 0.19110 UNMOVED"}), flush=True)
    # #224 (5) island SEED + CONTAINMENT build (SEPARATE protected-seed module + its OWN AdamW group;
    # grad-shield applied to the seed leaf BETWEEN the dual value_and_grad and seed_opt.update — NEVER
    # touching the witness grouped-backward / MD-decoupling grads). The seed is a per-pair RGB residual
    # seeded at ep0 from the GT island appearance (build_island_seed), masked to the self-detected
    # lane+movable island band; composited into the SEGNET-scored frame1 BEFORE R (via _compose_chain
    # above) so it rides the SHARED realized _f1/_signed (no 2nd SegNet). Because it is a SEPARATE
    # module (NOT model.parameters()), it is absent from EMA/blob/deploy => the verdict is witness-alone
    # == the deploy render == the 0-byte training-time ACCELERANT semantics, HONESTLY measured (the
    # verdict d_seg IS the deploy-absorption readout; the containment keeps the seed alive during
    # training so the witness has a formed island to absorb). Default OFF => byte-identical.
    seed_mod = None
    seed_opt = None
    seed_spec = None
    _seed_shield = None
    if seed_on:
        if float(args.w_seg) <= 0.0:
            raise ValueError("--seed-islands requires --w-seg > 0: the seed helps ONLY through the "
                             "realized seg loss on the composed frame1; with w_seg=0 it is inert.")
        import mlx.nn as _seed_nn
        from tac.boundary_math.island_protection import (
            ContainmentSpec as _SeedSpec,
            build_island_masks as _build_isl_masks,
            build_island_seed as _build_isl_seed,
            contain_protected_grad_mx as _contain_grad_mx,
            identify_island_classes as _ident_isl,
        )
        _seed_shield = _contain_grad_mx
        _lst_stack_s = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)], axis=0)
        _sdet = _ident_isl(_lst_stack_s, n_classes=5)
        _seed_res_np = np.zeros((P, render_h, render_w, 3), np.float32)
        _seed_msk_np = np.zeros((P, render_h, render_w, 1), np.float32)
        _s_supp = []
        for pi in range(P):
            _im = _build_isl_masks(np.asarray(gt.lstars[pi], np.int64), _sdet.lane_cls,
                                   _sdet.movable_cls, dilate_px=int(args.island_dilate_px))
            _gt1 = np.asarray(gt.gt_f1[pi], np.float32)
            if _gt1.shape[:2] != (render_h, render_w):
                import torch  # noqa: PLC0415
                import torch.nn.functional as _tF  # noqa: PLC0415
                _gt1 = _tF.interpolate(torch.from_numpy(_gt1).permute(2, 0, 1)[None],
                                       size=(render_h, render_w), mode="bilinear", align_corners=False
                                       )[0].permute(1, 2, 0).numpy()
            _seed = _build_isl_seed(_gt1, _im, base_render_segres=None, blend=float(args.seed_blend))
            _seed_res_np[pi] = _seed.residual
            _seed_msk_np[pi, ..., 0] = np.asarray(_im.any_mask, np.float32)
            _s_supp.append(float(_seed.support_frac))

        class _SeedMod(_seed_nn.Module):
            def __init__(self, res):
                super().__init__()
                self.residual = mx.array(res)

        seed_mod = _SeedMod(_seed_res_np)
        mx.eval(seed_mod.parameters())
        _seed_masks_mx = mx.array(_seed_msk_np)
        seed_state["mod"] = seed_mod
        seed_state["masks"] = [_seed_masks_mx[pi] for pi in range(P)]
        seed_spec = _SeedSpec(mode=str(args.containment_mode), damp=float(args.containment_damp),
                              protected_mask=None)
        seed_opt = optim.AdamW(learning_rate=float(args.seed_lr), weight_decay=0.0)
        print(json.dumps({"stage": "island_seed", "lane_cls": _sdet.lane_cls, "movable_cls": _sdet.movable_cls,
                          "island_classes": list(_sdet.island_classes),
                          "mean_support_frac": round(float(np.mean(_s_supp)), 5),
                          "containment_mode": str(args.containment_mode), "seed_lr": float(args.seed_lr),
                          "n_pairs": P,
                          "note": "SEPARATE protected seed module (own AdamW; NOT in EMA/blob/deploy = "
                          "0-byte accelerant; verdict=witness-alone=deploy=absorption readout); "
                          "shield-grad defends it; advisory; pointer 0.19110 UNMOVED"}), flush=True)

    def total_loss_fn(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w):
        # (--seg-spike-reweight) per-pixel spike/coherent map for THIS pair (pi==int(c1)//2); None when
        # the lever is off => base_loss byte-identical. A stop-grad theta-independent constant multiplier.
        _seg_px_w = _spike_w_mx[int(c1) // 2] if _spike_reweight_on else None
        L = base_loss(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form=seg_form, seg_pixel_w=_seg_px_w)
        phi0 = model.sdf(cf, c0)
        # (THETA* TIER-2 STRETCH-1) junction relax threaded; eik_jrelax=0.0 (default) => BIT-IDENTICAL.
        eik, length, _ = _eikonal_length_mlx(phi0, render_h, render_w,
                                             junction_relax=eik_jrelax, junction_tau=eik_jtau)
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
                          (lane_thin_w > 0.0 and lane_thin_gate["on"]) or
                          (mfh_w > 0.0 and mfh_target_mx is not None) or          # #218 facets 1b/3
                          (amplify_w > 0.0 and island_weight_mx is not None) or   # #224 island amplify
                          (subpix_w > 0.0 and subpix_gate["on"] and               # LEVER-4b sub-pixel t
                           _subpix_t_prov is not None) or
                          (persist_gate["w"] > 0.0 and bool(persist_classes)))    # #224 persistence loss
        if _seg_levers_on:
            # _render_R composes the FIXED bulk before R in residual mode (else == bare render) so
            # the surgical levers (lane-thin/margin-saliency/lane-edge) weight the COMPOSED-render
            # d_seg -- the residual IS the Lane+Movable annulus, so they are maximally relevant.
            _f1 = _render_R(model, cf, c1, render_h, render_w)  # (1, SEG_H, SEG_W, 3)
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
            if msal_reach and _sR_provider is not None:
                # REACHABILITY (REPLACES the inert texture proxy; MEASURED Pearson -0.033 for 1/(1+b*tex)):
                # multiply the fragility saliency by the cached THROUGH-R margin-Jacobian S_R for THIS pair
                # (stop-grad [0,1] weight; theta-independent reachability of the CORRECT answer). Product
                # concentrates capacity where the pixel is BOTH fragile (small GT margin) AND reachable
                # (high S_R) = the actionable margin-boundary band. pi==int(c1)//2 (same key as
                # island_weight_mx). The cache is precomputed => this is CHEAPER than the per-step tex recompute.
                sal = sal * _sR_provider[int(c1) // 2]                   # (1, H, W) reachability-weighted
            elif msal_uni:
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
        # LEVER-4b (SUB-PIXEL BOUNDARY-PLACEMENT `t`; DIRECTIONAL upgrade of LEVER-4, GREEN 2026-07-03).
        # Rides the SAME SHARED realized through-R margin `_signed` (Mw = relu(_signed) = the witness
        # GT-class margin at every pixel, the honest mirror of the GT top1-top2 the target `t` is built
        # from) -- NO 2nd SegNet forward. At each pre-selected genuine-V straddle pixel p (dominant
        # inter-class edge; both GT margins in the flip band) the GT boundary sits at fractional position
        # t = M_GT[p]/(M_GT[p]+M_GT[q]) between p and its cross-edge partner q. We supervise the witness's
        # OWN realized margin ratio t_wit = Mw[p]/(Mw[p]+Mw[q]) toward that GT t: a DENSER, sub-pixel,
        # differentiable placement signal that pulls both Mw[p] and Mw[q] to seat the witness boundary at
        # the correct sub-pixel spot. q is the precomputed dominant direction (0=right (i,j+1), 1=down
        # (i+1,j)); Mw[q] is a pure shift of the SHARED Mw (fully vectorized, both differentiable). Masked
        # to the precomputed active straddle set (sentinel t<0 => weight 0). Default subpix_w=0 => skipped
        # => byte-identical. MODEST 2nd-order refinement (weakest on thin lanes; effect in the 1-2px flip
        # band, #149) -> an A/B arm, NOT a claim. pointer 0.19110 UNMOVED.
        if subpix_w > 0.0 and subpix_gate["on"] and _subpix_t_prov is not None:
            _pi_sp = int(c1) // 2
            _t_tgt = _subpix_t_prov[_pi_sp]                              # (1,H,W) f32, -1 sentinel
            _dir_m = _subpix_dir_prov[_pi_sp]                            # (1,H,W) f32 in {0,1}
            _active = (_t_tgt >= 0.0).astype(_signed.dtype)             # (1,H,W) genuine-V straddle mask
            _Mw = mx.maximum(_signed, 0.0)                              # (1,H,W) witness GT-class margin
            # partner margin via a pure shift of the SHARED Mw (edge-col/row pad is inert: those pixels
            # can never be active in the corresponding direction, so the pad value is masked out).
            _M_right = mx.pad(_Mw[:, :, 1:], [(0, 0), (0, 0), (0, 1)])   # _M_right[i,j] = Mw[i,j+1]
            _M_down = mx.pad(_Mw[:, 1:, :], [(0, 0), (0, 1), (0, 0)])    # _M_down[i,j]  = Mw[i+1,j]
            _Mq = mx.where(_dir_m < 0.5, _M_right, _M_down)             # dominant cross-edge partner
            _t_wit = _Mw / (_Mw + _Mq + subpix_eps)                     # witness sub-pixel boundary ratio
            _t_ref = mx.maximum(_t_tgt, 0.0)                            # sentinel -1 -> 0 (masked anyway)
            _sq = mx.square(_t_wit - _t_ref) * _active                  # placement error on genuine-V px
            subpix_term = mx.sum(_sq) / (mx.sum(_active) + 1e-6)        # mean over active straddles
            L = L + subpix_w * subpix_term
        # #224 (5) ISLAND AMPLIFICATION — the island-birth term on the SHARED realized _signed margin
        # (island x persistence weight; orthogonal to LEVER-4's fragility x all-class weight). Default
        # amplify_w=0 => skipped => byte-identical. c1 = 2*pi+1 (the SegNet-scored frame) => pi=c1//2.
        if amplify_w > 0.0 and island_weight_mx is not None:
            L = L + amplify_w * island_birth_from_signed_mx(
                _signed, island_weight_mx[int(c1) // 2], amplify_mtgt, form=amplify_form)
        # #224 (4) PERSISTENCE/TOPOLOGY loss — soft-clDice + persistence-weighted island recall on the
        # SHARED realized seg logits (_slog). GT-presence-gated inside the module (never hallucinate).
        # Annealed weight persist_gate["w"] (set per-epoch, coarse->fine); 0 => branch inert.
        if persist_gate["w"] > 0.0 and persist_classes:
            # (--cache-gt-skeleton #260) reuse the precomputed CONSTANT GT skeleton for THIS pair
            # (pi == c0//2, the SAME key thin_maps_mx/island_weight_mx use). None => inline recompute
            # (byte-identical default); a cache MISS also falls back to None (still bit-identical).
            _sg_pre = _sg_cache.get(int(c0) // 2) if _sg_cache is not None else None
            L = L + persist_gate["w"] * persistence_topology_loss_mlx(
                _slog, lstar_oh, persist_classes, cldice_iters=persist_cldice_iters,
                w_cldice=1.0, w_recall=persist_recall_w, sg_precomputed=_sg_pre)
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
        # #218 facets 1b/3 (MARGIN-FIELD HEAD, byte-free): realized through-R PER-CLASS margin hinge.
        # per-pixel target = additive-margin (facet-1b) + per-class Menon boost on rare classes
        # (facet-3), broadcast to each pixel by its GT class via lstar_oh. Reuses the SHARED _signed
        # (R2b-M3). Default-off (mfh_w=0) => byte-identical. This widens the realized SegNet decision
        # margin MORE for the erasure-prone rare classes (Lane<->Road 57% tail, #209).
        if mfh_w > 0.0 and mfh_target_mx is not None:
            per_pix_tgt = mx.sum(lstar_oh * mfh_target_mx, axis=-1)     # (1,H,W) per-class margin target
            hmap_m = mx.maximum(per_pix_tgt - _signed, 0.0)            # fragile pixels below their target
            L = L + mfh_w * mx.mean(hmap_m)
        # (THETA* TIER-2 MUST-2) nuclear-norm low-rank code penalty. DEFAULT-OFF: code_nuc_w=0.0 =>
        # this branch NEVER runs => L is byte-identical (fully additive). When >0 it adds
        # weight * smoothed_nuclear_norm(model.code) -> drives the per-pair FiLM codes
        # (num_pairs*2 x mod_dim) toward a low-rank subspace (rate). The code matrix is identical for
        # every pair, so the per-pair value_and_grad sees the same term and the mean-over-chunk grad
        # applies it ONCE per opt step (NOT P-scaled). Recomputed per value_and_grad call (a parent
        # fusion could hoist it once-per-step; out of scope for this additive prep).
        if code_nuc_w > 0.0:
            L = L + code_nuc_w * _nuclear_norm_smooth_mlx(
                model.code, rel_eps=code_nuc_eps, ns_iters=code_nuc_iters)
        return L

    value_and_grad = nn.value_and_grad(model, total_loss_fn)

    # #224 (5) DUAL value_and_grad for the island SEED (its OWN param tree + optimizer). Co-differentiate
    # the witness (model) AND the seed (seed_mod) w.r.t. the SAME loss (the seed enters via _compose_chain
    # -> _f1 -> seg_l). Default OFF (seed_mod None) => _dual_vg None => the loop takes the single
    # value_and_grad path (BYTE-IDENTICAL). The witness grad tree (grads[0]) is IDENTICAL to the single
    # path (same loss, same model params); only the extra grads[1] (seed) is new -> the shield acts on
    # grads[1] ONLY, then seed_opt (a DISTINCT AdamW) applies it -> the witness opt.update + MD-decoupling
    # + grouped-backward path is UNTOUCHED.
    _dual_vg = None
    if seed_mod is not None:
        def _combined_seed_loss(model_p, seed_p, cf, c0, c1, oh, mg, ptg, ws, wp, hg, mt, sf, ew, lw):
            model.update(model_p)
            seed_mod.update(seed_p)
            return total_loss_fn(model, cf, c0, c1, oh, mg, ptg, ws, wp, hg, mt, sf, ew, lw)

        _dual_vg = mx.value_and_grad(_combined_seed_loss, argnums=(0, 1))

    # ===================================================================================
    # (--micro-batch-pairs, DAG FEED 2026-07-03c) BATCHED twin of ``total_loss_fn``. OPT-IN
    # (--micro-batch-pairs > 1); the DEFAULT B=1 path NEVER calls this (the accum loop keeps its
    # UNCHANGED serial per-pair value_and_grad). The ONLY batched operations are the EXPENSIVE realized
    # render + FROZEN-SCORER forwards (one segnet over the B f1 frames, one posenet over the B pairs) —
    # the measured bottleneck (single-pair EfficientNet-B2 under-utilizes the GPU). EVERY per-pair loss
    # reduction (base seg-form; the score-domain pose ``sqrt(10*d_pose)`` which is NONLINEAR so
    # sqrt(mean)!=mean(sqrt); and every weighted-mean lever ``sum(x*w)/sum(w)``) is computed PER PAIR on
    # the batched scorer outputs and MEAN-ed over B, so
    #     total_loss_fn_batch(B pairs) == mean_b total_loss_fn(pair_b)
    # WITHIN fp tolerance (batched conv/BN is batch-independent in SegNet/PoseNet eval mode -> per-frame
    # logits are unchanged by batching; the mean-over-B is the only reduction re-order). That EXACT
    # per-pair-mean identity makes the accum-loop grad match the serial mean-over-chunk EXACTLY (the
    # accum loop weights each group's mean-grad by its pair count) and lets the numerical-equivalence
    # test pin batched-grad == mean-of-per-pair-grad. The realized segnet(f1) forward is computed ONCE
    # and SHARED by the base seg-form AND the lever ``_signed`` — bit-identical to total_loss_fn's two
    # deterministic-render forwards ((f'+g')·dS == f'·dS + g'·dS). The once-per-step per-MODEL code
    # penalties (rankfloor / code-spec / code-nuc) are added ONCE (matching the serial mean-over-chunk
    # of an identical-per-pair term). NOT bit-identical to the serial path (batched fp reduction order):
    # a trajectory-affecting opt-in validated by a short A/B. Mirrors total_loss_fn op-for-op.
    # ===================================================================================
    # (--micro-batch-pairs) BATCHED twin of total_loss_fn -> delegates to the importable + unit-tested
    # tac.boundary_math.levelset_micro_batch_loss (the nested closure cannot be reached from a test).
    # The LeverConfig SNAPSHOTS the ~30 lever closures; the gate dicts (lane_gate / msal_gate /
    # lane_thin_gate / persist_gate) + the lever tensor dicts (island_weight_mx / thin_maps_mx) are
    # passed BY REFERENCE and are MUTATED-IN-PLACE by the epoch loop, so the per-epoch gate/anneal
    # changes are seen live -- exactly like total_loss_fn re-reads them each value_and_grad call.
    from tac.boundary_math.levelset_micro_batch_loss import (
        LeverConfig as _MicroBatchLeverConfig,
        batched_realized_loss as _micro_batched_realized_loss,
    )

    _micro_batch_lc = _MicroBatchLeverConfig(
        seg_loss_default=args.seg_loss, tau_use=float(args.tau_softplus_tau),
        l7_thr_use=float(args.l7_threshold), l7_mult=float(args.l7_mult),
        score_domain=bool(args.score_domain_loss), pose_eps=float(args.pose_eps),
        eik_jrelax=eik_jrelax, eik_jtau=eik_jtau,
        eikonal_length=_eikonal_length_mlx, nuclear_norm_smooth=_nuclear_norm_smooth_mlx,
        lane_w=lane_w, lane_gate=lane_gate, lane_cls=lane_cls, lane_tgt=lane_tgt,
        msal_w=msal_w, msal_gate=msal_gate, msal_tau=msal_tau, msal_tgt=msal_tgt,
        msal_uni=msal_uni, msal_uni_beta=msal_uni_beta,
        amplify_w=amplify_w, island_weight_mx=island_weight_mx, amplify_mtgt=amplify_mtgt,
        amplify_form=amplify_form,
        persist_gate=persist_gate, persist_classes=persist_classes,
        persist_cldice_iters=persist_cldice_iters, persist_recall_w=persist_recall_w,
        lane_thin_w=lane_thin_w, lane_thin_gate=lane_thin_gate, thin_maps_mx=thin_maps_mx,
        lane_thin_tgt=lane_thin_tgt,
        mfh_w=mfh_w, mfh_target_mx=mfh_target_mx,
        rankfloor_w=rankfloor_w, rankfloor_idx=rankfloor_idx, rankfloor_tgt=rankfloor_tgt,
        code_spec_w=code_spec_w, code_spec_idx=code_spec_idx,
        code_nuc_w=code_nuc_w, code_nuc_eps=code_nuc_eps, code_nuc_iters=code_nuc_iters,
    )

    def total_loss_fn_batch(model, cf_list, c0_list, c1_list, oh_list, mg_list, pose_tgt_list,
                            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w):
        # THIN wrapper: the batched loss body + equivalence contract live in the importable module.
        # render_fn (witness / residual-compose / AA / pose-carrier) or the bare R render, exactly as
        # base_loss picks it in total_loss_fn.
        _render = render_fn if render_fn is not None else render_through_R_mlx
        return _micro_batched_realized_loss(
            model, adapter, _render, render_h, render_w,
            cf_list, c0_list, c1_list, oh_list, mg_list, pose_tgt_list,
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, _micro_batch_lc)

    # OPT-IN batched accum path. Build the batched value_and_grad ONLY when engaged (>1). Fail CLOSED
    # with --seed-islands: the seed co-grad dual value_and_grad path is per-pair only (batched + seed
    # co-grad is unsupported -> raise rather than silently diverge). DEFAULT (1) => _use_micro_batch
    # False => value_and_grad_batch None => the accum loop takes the UNCHANGED serial path.
    _micro_batch_pairs = int(getattr(args, "micro_batch_pairs", 1))
    if _micro_batch_pairs > 1 and seed_mod is not None:
        raise ValueError(
            "--micro-batch-pairs > 1 is not supported together with --seed-islands (the seed co-grad "
            "dual value_and_grad path is per-pair only). Run seed-islands at --micro-batch-pairs 1.")
    _use_micro_batch = _micro_batch_pairs > 1
    value_and_grad_batch = nn.value_and_grad(model, total_loss_fn_batch) if _use_micro_batch else None
    if _use_micro_batch:
        print(json.dumps({"stage": "micro_batch_pairs", "B": int(_micro_batch_pairs),
                          "accum_pairs": int(args.accum_pairs),
                          "note": "OPT-IN batched scorer forward (B pairs/forward); trajectory-affecting "
                          "(batched fp reduction) but grad == serial mean-over-chunk within fp tol; "
                          "advisory; pointer 0.19110 UNMOVED"}), flush=True)

    # one-hot L* + margin per pair at the SegNet OUTPUT res (gt.lstars/gt.margins are 384x512,
    # matching the realized seg_logits = adapter.segnet(R(rgb))). NOT render res.
    def _lstar_oh(pi: int):
        lr = np.asarray(gt.lstars[pi])  # (384,512)
        oh = np.eye(5, dtype=np.float32)[lr.ravel()].reshape(lr.shape[0], lr.shape[1], 5)
        mg = np.asarray(gt.margins[pi], np.float32)  # (384,512)
        return mx.array(oh[None]), mx.array(mg[None])

    pose_tgts = [mx.array(np.asarray(gt.gt_poses[pi], np.float32)) for pi in range(P)]
    lstar_cache = [_lstar_oh(pi) for pi in range(P)]

    # LEVER-4 REACHABILITY (default-off): POPULATE the per-pair through-R S_R provider ONLY when
    # --margin-saliency-reachability AND msal_w>0. Loads the precomputed 'sR' (P,H,W) [0,1] weight from
    # --gt-cache (lazy npz: inflates ONLY the sR member) -> a list[mx.array (1,H,W)] indexed by pi (==
    # int(c1)//2, the SAME key island_weight_mx / _sR_provider[...] use). Fails CLOSED (never silently)
    # if the flag is set but the cache lacks 'sR' (run tools/precompute_sR_reachability.py) or if
    # micro-batch is on (the batched twin's LEVER-4 does not yet consume S_R). When OFF, _sR_provider
    # stays None (declared above) and the LEVER-4 branch never references it => byte-identical resume.
    if msal_reach and msal_w > 0.0:
        if _use_micro_batch:
            raise ValueError(
                "--margin-saliency-reachability is not supported with --micro-batch-pairs>1 (the batched "
                "LEVER-4 twin does not consume S_R yet); run the reachability arm at --micro-batch-pairs 1.")
        if not args.gt_cache:
            raise ValueError(
                "--margin-saliency-reachability requires --gt-cache (the 'sR' reachability map is cached "
                "there); build it with tools/precompute_sR_reachability.py --gt-cache <path>.")
        _zc = np.load(Path(args.gt_cache), allow_pickle=False)
        if "sR" not in _zc.files:
            raise ValueError(
                f"--gt-cache {args.gt_cache} has no 'sR' key; build it first: "
                f"tools/precompute_sR_reachability.py --gt-cache {args.gt_cache} --num-pairs {P}.")
        _sR_all = _zc["sR"]  # (cached_P, H, W) float32 in [0,1]; inflate the sR member ONCE
        if int(_sR_all.shape[0]) < P:
            raise ValueError(
                f"--gt-cache {args.gt_cache} 'sR' has {int(_sR_all.shape[0])} pairs < --num-pairs {P}; "
                "re-run tools/precompute_sR_reachability.py at >= the requested size.")
        _sR_provider = [mx.array(np.asarray(_sR_all[pi], np.float32)[None]) for pi in range(P)]
        print(json.dumps({"stage": "margin_saliency_reachability", "active": True, "n_pairs": int(P),
                          "gt_cache": str(args.gt_cache),
                          "sR_norm_mean": round(float(np.asarray(_sR_all[:P]).mean()), 5),
                          "note": "LEVER-4 saliency weighted by cached through-R margin-Jacobian S_R "
                          "(REPLACES the measured-inert 1/(1+beta*tex) texture path); advisory build, "
                          "A/B owed (needs GO); pointer 0.19110 UNMOVED"}), flush=True)

    # (--seg-spike-reweight, source-split MEASURED 2026-07-03) precompute the theta-INDEPENDENT per-pair
    # spike/coherent weight map from the GT argmax TEMPORAL neighbors (list[mx.array (1,H,W)] indexed by
    # pi == int(c1)//2, the SAME key _sR_provider/island_weight_mx use). A pixel is a SPIKE at pair pi if
    # lstar[pi] differs from BOTH neighbors lstar[pi-1] & lstar[pi+1] (single-frame argmax FLICKER a
    # per-frame witness structurally cannot fit -- MEASURED n600 ~88.6% IRREDUCIBLE appearance change, so
    # smooth-is-optimal there); COHERENT = temporally-UNSTABLE but matches >=1 neighbor (the winnable
    # boundary residual). Map = downweight@spike, upweight@coherent, 1.0 else. Endpoints (pi in {0,P-1},
    # only one neighbor) => all-1.0. Default OFF (_spike_w_mx None) OR both scalars==1.0 (map==1.0) =>
    # base_loss gets seg_pixel_w=None/ones => BYTE-IDENTICAL. Fails CLOSED with micro-batch (serial path
    # only; the batched twin does not consume seg_pixel_w yet). A/B owed (needs GO); pointer 0.19110 UNMOVED.
    _spike_reweight_on = bool(getattr(args, "seg_spike_reweight", False))
    _spike_w_mx = None
    if _spike_reweight_on:
        if _use_micro_batch:
            raise ValueError(
                "--seg-spike-reweight is not supported with --micro-batch-pairs>1 (the batched twin does "
                "not consume the per-pixel seg reweight yet); run this arm at --micro-batch-pairs 1.")
        _sp_dn = float(getattr(args, "seg_spike_downweight", 1.0))
        _sp_up = float(getattr(args, "seg_coherent_upweight", 1.0))
        _sp_H, _sp_W = np.asarray(gt.lstars[0]).shape
        _sp_stack = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)])  # (P,H,W)
        _spike_w_mx = []
        _sp_n_spike = 0
        _sp_n_coh = 0
        for pi in range(P):
            wmap = np.ones((_sp_H, _sp_W), np.float32)
            if 0 < pi < P - 1:
                c_, p_, n_ = _sp_stack[pi], _sp_stack[pi - 1], _sp_stack[pi + 1]
                dp, dn = (c_ != p_), (c_ != n_)
                sp = dp & dn                 # differs from BOTH neighbors = unfittable flicker
                coh = (dp | dn) & (~sp)       # unstable but matches >=1 neighbor = winnable boundary
                wmap[coh] = _sp_up
                wmap[sp] = _sp_dn
                _sp_n_spike += int(sp.sum())
                _sp_n_coh += int(coh.sum())
            _spike_w_mx.append(mx.array(wmap[None]))  # (1,H,W)
        _sp_byte_identical = (_sp_dn == 1.0 and _sp_up == 1.0)
        print(json.dumps({"stage": "seg_spike_reweight", "active": True, "n_pairs": int(P),
                          "downweight": _sp_dn, "upweight": _sp_up,
                          "spike_px_total": _sp_n_spike, "coherent_px_total": _sp_n_coh,
                          "byte_identical_scalars": _sp_byte_identical,
                          "note": "per-pixel seg-CE reweight: down-weight single-frame flicker "
                          "(~88.6%% irreducible, smooth-is-optimal), up-weight coherent boundary; "
                          "A/B owed (needs GO); pointer 0.19110 UNMOVED"}), flush=True)

    # LEVER-4b (SUB-PIXEL BOUNDARY `t`) PRECOMPUTE. theta-INDEPENDENT + cheap (pure numpy from the cached
    # gt.margins/gt.lstars -- NO SegNet forward, NO torch autograd), so it is built INLINE here (spike-map
    # style), not by a separate tool like the through-R S_R. For each pair pi, per pixel p=(i,j) we form
    # the two axis-aligned inter-class straddles -- RIGHT (p,(i,j+1)) and DOWN (p,(i+1,j)) -- keep only
    # GENUINE-V straddles (lstar differs AND both GT margins < the flip-band `subpix_band`; MEASURED n96:
    # band 1.0 -> ~2196 active px/frame = 1.12%% of pixels, t mean 0.527 std 0.263 ~ informative Uniform),
    # and assign p its DOMINANT straddle = the one with the SHALLOWER partner margin (== the smaller-sum V,
    # since p's own margin is shared; the sharpest / most-defined boundary). Stored per pair: t_map (1,H,W)
    # f32 = the GT ratio M_GT[p]/(M_GT[p]+M_GT[q]) in [0,1] where active, -1.0 sentinel elsewhere (encodes
    # the active mask); dir_map (1,H,W) f32 in {0,1} = the dominant direction (0=right,1=down) the loss
    # shifts Mw by to gather Mw[q]. Providers stay None unless subpix_w>0 => the OFF path is byte-identical.
    # Fails CLOSED with micro-batch (the batched twin's LeverConfig does not carry this lever yet). Memory
    # ~ 2x the down-weight map (t + dir float maps): P*H*W*4*2 ~= 940 MB at n600 (trivial vs RAM; noted for
    # the launcher preflight). A/B owed (needs GO); pointer 0.19110 UNMOVED.
    if subpix_w > 0.0:
        if _use_micro_batch:
            raise ValueError(
                "--seg-subpix-boundary-weight>0 is not supported with --micro-batch-pairs>1 (the batched "
                "twin does not consume the sub-pixel boundary lever yet); run this arm at "
                "--micro-batch-pairs 1.")
        _sx_H, _sx_W = np.asarray(gt.lstars[0]).shape
        _subpix_t_prov = []
        _subpix_dir_prov = []
        _sx_n_active = 0
        _sx_t_sum = 0.0
        for pi in range(P):
            _lst = np.asarray(gt.lstars[pi], np.int64)
            _mg = np.asarray(gt.margins[pi], np.float32)
            # RIGHT straddles (p,(i,j+1)) live in cols [:, :W-1]; DOWN straddles (p,(i+1,j)) in rows [:H-1, :].
            _dh = _lst[:, :-1] != _lst[:, 1:]
            _mph = _mg[:, :-1]; _mqh = _mg[:, 1:]
            _th = _mph / (_mph + _mqh + subpix_eps)
            _vh = _dh & (_mph < subpix_band) & (_mqh < subpix_band)     # genuine-V RIGHT straddles
            _dv = _lst[:-1, :] != _lst[1:, :]
            _mpv = _mg[:-1, :]; _mqv = _mg[1:, :]
            _tv = _mpv / (_mpv + _mqv + subpix_eps)
            _vv = _dv & (_mpv < subpix_band) & (_mqv < subpix_band)     # genuine-V DOWN straddles
            # per-pixel candidate fields (inf partner margin where no candidate -> loses the min).
            _has_r = np.zeros((_sx_H, _sx_W), bool); _has_r[:, :_sx_W - 1] = _vh
            _qr = np.full((_sx_H, _sx_W), np.inf, np.float32); _qr[:, :_sx_W - 1] = _mqh
            _tr = np.zeros((_sx_H, _sx_W), np.float32); _tr[:, :_sx_W - 1] = _th
            _has_d = np.zeros((_sx_H, _sx_W), bool); _has_d[:_sx_H - 1, :] = _vv
            _qd = np.full((_sx_H, _sx_W), np.inf, np.float32); _qd[:_sx_H - 1, :] = _mqv
            _td = np.zeros((_sx_H, _sx_W), np.float32); _td[:_sx_H - 1, :] = _tv
            # dominant = shallower partner margin (ties -> right). p's own margin is shared, so this is the
            # smaller-sum (sharpest) V.
            _pick_r = _has_r & (~_has_d | (_qr <= _qd))
            _pick_d = _has_d & (~_has_r | (_qd < _qr))
            _t_full = np.full((_sx_H, _sx_W), -1.0, np.float32)
            _dir_full = np.zeros((_sx_H, _sx_W), np.float32)
            _t_full[_pick_r] = _tr[_pick_r]; _dir_full[_pick_r] = 0.0
            _t_full[_pick_d] = _td[_pick_d]; _dir_full[_pick_d] = 1.0
            _act = _pick_r | _pick_d
            _sx_n_active += int(_act.sum())
            _sx_t_sum += float(_t_full[_act].sum()) if _act.any() else 0.0
            _subpix_t_prov.append(mx.array(_t_full[None]))              # (1,H,W)
            _subpix_dir_prov.append(mx.array(_dir_full[None]))          # (1,H,W)
        _sx_t_mean = round(_sx_t_sum / _sx_n_active, 4) if _sx_n_active else 0.0
        print(json.dumps({"stage": "seg_subpix_boundary", "active": True, "n_pairs": int(P),
                          "weight": subpix_w, "v_band": subpix_band, "start_epoch": int(subpix_start),
                          "active_px_total": int(_sx_n_active),
                          "active_px_per_frame": round(_sx_n_active / max(P, 1), 1),
                          "t_target_mean": _sx_t_mean,
                          "note": "sub-pixel boundary-placement target t=M_GT[p]/(M_GT[p]+M_GT[q]) on "
                          "genuine-V straddles; supervises the witness realized margin ratio (DIRECTIONAL "
                          "upgrade of LEVER-4 #141); A/B owed (needs GO); pointer 0.19110 UNMOVED"}), flush=True)

    # (--cache-gt-skeleton, #260 SPEED) BUILD the per-pair GT soft-skeleton cache ONCE (each sg is
    # mx.eval'd to a concrete constant, detached from any lazy graph, OUTSIDE any value_and_grad
    # transform -> safe + bit-identical to the inline recompute). Keyed by pair index pi (== c0//2 ==
    # c1//2, the SAME key thin_maps_mx / island_weight_mx use). Gated on persist-on AND not
    # micro-batch (the serial total_loss_fn is the only consumer; the batched twin recomputes). The
    # per-pair sg built here matches persistence_topology_loss_mlx's inline `g` construction op-for-op
    # (precompute_sg_mlx uses the identical stack->reshape->soft_skeleton), so sg_precomputed== inline.
    if cache_gt_skeleton and persist_w > 0.0 and persist_classes and not _use_micro_batch:
        from tac.boundary_math.persistence_topology_loss import precompute_sg_mlx as _precompute_sg_mlx
        _sg_cache = {}
        for _pi in range(P):
            _sg = _precompute_sg_mlx(lstar_cache[_pi][0], persist_classes, persist_cldice_iters)
            mx.eval(_sg)  # materialize as a concrete constant (bit-identical to the inline recompute)
            _sg_cache[_pi] = _sg
        print(json.dumps({"stage": "cache_gt_skeleton", "n_pairs": int(P),
                          "target_classes": list(persist_classes), "cldice_iters": persist_cldice_iters,
                          "note": "precomputed CONSTANT GT soft-skeleton per pair (bit-identical "
                          "speed-only; skips ~half the clDice recompute); pointer 0.19110 UNMOVED"}),
              flush=True)

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

    def _pc_verdict_f0_uint8(pi: int, deploy: dict[str, np.ndarray]) -> np.ndarray:
        """#224/#205 pose-carrier NO-FAKE verdict frame0: warp the carrier SOURCE by the DEPLOYED
        (EMA-shadow, int8-dequant) carrier twist xi_eff, so the advisory d_pose measures what the
        carrier actually produces. xi_stored uses the original fp32 table (the stored twist ships fp16,
        not int8; the trained residual dxi rides the deploy dict). Native-res uint8 (874x1164x3)
        matching the witness f1 verdict contract + cpu_verdict_d_pose.

        SOURCE (per --pose-carrier-source): real_keyframe (default) warps the STORED gt_f0;
        generated (STORE-NOTHING) warps the witness's OWN plain frame0 render up-to-camera (== the
        byte-close store_nothing warp source: _fwd_numpy(f0 code) -> _torch_R_to_camera_uint8), so the
        advisory d_pose reflects the store-nothing decode (NOT the real keyframe)."""
        from tac.boundary_math.warp_real_luma_frame0 import warp_frame0_uint8_numpy as _pc_warp_u8
        xi = np.asarray(pose_carrier_xi_stored[pi], np.float64)
        scale = float(args.pose_carrier_residual_scale)
        if str(args.pose_carrier_residual_mode) == "table":
            dxi = np.asarray(deploy.get("pose_carrier.dxi"), np.float64)[pi]
        else:  # film: numpy twin of gelu(film_in(code)) -> film_out (advisory reconstruction)
            from scipy.special import erf
            code = np.asarray(deploy["code"][2 * pi + 0], np.float64)
            w_in = np.asarray(deploy["pose_carrier.film_in.weight"], np.float64)
            b_in = np.asarray(deploy["pose_carrier.film_in.bias"], np.float64)
            w_out = np.asarray(deploy["pose_carrier.film_out.weight"], np.float64)
            b_out = np.asarray(deploy["pose_carrier.film_out.bias"], np.float64)
            h = code @ w_in.T + b_in
            h = 0.5 * h * (1.0 + erf(h / np.sqrt(2.0)))    # exact gelu (matches mlx.nn.gelu)
            dxi = (h @ w_out.T + b_out).reshape(-1)
        xi_eff = xi + scale * dxi
        if str(getattr(args, "pose_carrier_source", "real_keyframe")) == "generated":
            # STORE-NOTHING: warp the witness's OWN plain frame0 render (up to camera-native uint8),
            # not the stored real keyframe -> the same source the store_nothing byte-close decodes.
            rgb, _phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + 0])
            src_native = _torch_R_to_camera_uint8(rgb.reshape(render_h, render_w, 3))
        else:
            src_native = np.asarray(gt.gt_f0[pi])
        return _pc_warp_u8(src_native, xi_eff, pose_carrier_geom)

    def _render_numpy_deploy(deploy: dict[str, np.ndarray], pi: int, fk: int) -> np.ndarray:
        """THE ONE CODEPATH (fp32 numpy, deploy-faithful) — same forward the byte-close/inflate use.
        Uses the PER-PAIR feats (curvelet [+ self-orient]) so the verdict == the deploy render. In
        residual mode the INR RGB is COMPOSED with the FIXED bulk (where(mask, INR, bulk)) BEFORE R,
        so the advisory d_seg reflects the COMPOSED witness that ships (NO-FAKE). #224 pose-carrier:
        frame0 (fk==0) routes through the carrier warp so the d_pose verdict measures the carrier."""
        if pose_carrier is not None and fk == 0:
            return _pc_verdict_f0_uint8(pi, deploy)
        rgb, _phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + fk])
        rgb_hw3 = rgb.reshape(render_h, render_w, 3)
        if _compose_np is not None:
            rgb_hw3 = _compose_np(rgb_hw3, pi)
        return _torch_R_to_camera_uint8(rgb_hw3)

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
                if _aa_so_fine:  # snapshot base argmax (int8) for the fine dir-feat recompute
                    base_argmax_per_pair[pi] = argmax.astype(np.int8)
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
            if _aa_so_fine:  # snapshot base argmax (int8) for the fine dir-feat recompute
                base_argmax_per_pair[pi] = argmax.astype(np.int8)
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
        # (#205 OOM fix) chunk the CPU-scorer inference (bit-identical; eval-mode BN running stats).
        # pose VERDICT still measured (monitoring) but pose is NOT the witness's job at w_pose=0
        # (default). The deploy d_pose is OPEN on the witness — measured through the byte-closed
        # store_nothing/table carrier (#205 R1), NO ancestor number (the 3.4e-5 was ANCESTOR-RGB,
        # never validated on this vehicle; see CLAUDE.md "Pose is SOLVED" caveat + axis-9).
        d_seg, d_pose = _verdict_dseg_dpose_chunked(
            seg_cpu, posenet_cpu, f0s, f1s,
            [gt.lstars[pi] for pi in vpairs], [gt.gt_poses[pi] for pi in vpairs],
            vbatch=int(args.verdict_batch))
        return {"d_seg": d_seg, "d_pose": d_pose}

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
            _r0 = rgb0.reshape(render_h, render_w, 3)
            _r1 = rgb1.reshape(render_h, render_w, 3)
            if _compose_np is not None:  # residual mode: compose the FIXED bulk before R (NO-FAKE)
                _r0 = _compose_np(_r0, pi)
                _r1 = _compose_np(_r1, pi)
            # #224 pose-carrier: frame0 through the carrier warp (deploy int8-dequant xi_eff) so the
            # ASYNC d_pose verdict measures the carrier too (matches the sync _render_numpy_deploy path).
            if pose_carrier is not None:
                f0s.append(_pc_verdict_f0_uint8(pi, deploy))
            else:
                f0s.append(_torch_R_to_camera_uint8(_r0))
            f1s.append(_torch_R_to_camera_uint8(_r1))
        # (#205 OOM fix) chunk the CPU-scorer inference (bit-identical; eval-mode BN running stats).
        d_seg, d_pose = _verdict_dseg_dpose_chunked(
            seg_cpu, posenet_cpu, f0s, f1s,
            [gt.lstars[pi] for pi in vpairs], [gt.gt_poses[pi] for pi in vpairs],
            vbatch=int(args.verdict_batch))
        return {"d_seg": d_seg, "d_pose": d_pose}

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

    # #224 (Wave B) FINE self-orient dir-feats (AA-supersample + --self-orient). Recompute from the
    # snapshotted base argmax: NN-upsample to the ss*grid -> fine EDT-tangent -> directional Fourier
    # (the SAME self_orientation_directional_feats path as the base, at fine coords). Pre-first-reorient
    # (argmax None) -> zeros -> pure-curvelet fine (matches the base zeros-until-reorient contract).
    def _fine_dir_feats_np(pi: int) -> np.ndarray:
        cols = _aa_coords_fine.shape[0]
        ba = base_argmax_per_pair[pi]
        if ba is None:
            return np.zeros((cols, dir_w), np.float32)
        arg_fine = np.kron(np.asarray(ba, np.int64), np.ones((aa_ss, aa_ss), np.int64))
        return self_orientation_directional_feats(
            _aa_coords_fine, arg_fine, n_freqs=n_dir_freqs,
            freq_across=args.freq_across, freq_along=args.freq_along).astype(np.float32)

    def _rebuild_fine_dir_cache() -> None:
        # full mode: recompute ALL P fine dir-feats ONCE (amortized across the reorient window).
        # batch mode: just INVALIDATE the bounded LRU (recomputed lazily on next use).
        if not _aa_so_fine:
            return
        _aa_fine_lru.clear()
        if _aa_fine_mode == "full":
            for pi in range(P):
                _aa_fine_dir_full[pi] = mx.array(_fine_dir_feats_np(pi))
            mx.eval([x for x in _aa_fine_dir_full if x is not None])

    def _cf_fine_mx(pi: int):
        # per-pair FINE render feats = shared curvelet-fine (coord_feats_fine_mx) [+ fine dir-feats].
        if not _aa_so_fine:
            return coord_feats_fine_mx
        if _aa_fine_mode == "full":
            df = _aa_fine_dir_full[pi]
            if df is None:                       # pre-first-reorient safety
                df = mx.array(_fine_dir_feats_np(pi))
        else:  # batch: bounded FIFO on-demand cache (memory ~ cap*per-pair)
            df = _aa_fine_lru.get(pi)
            if df is None:
                df = mx.array(_fine_dir_feats_np(pi))
                _aa_fine_lru[pi] = df
                _cap = max(1, int(getattr(args, "aa_self_orient_fine_cache_cap", 16)))
                while len(_aa_fine_lru) > _cap:
                    _aa_fine_lru.pop(next(iter(_aa_fine_lru)))
        return mx.concatenate([coord_feats_fine_mx, df], axis=-1)

    if use_self_orient:
        _rebuild_cf_mx_cache()  # ep<reorient: dir feats are zeros -> pure curvelet iso pass
        _rebuild_fine_dir_cache()  # AA fine self-orient (no-op unless --aa-self-orient-fine-mode)
        if os.environ.get("TAC_MEM_PROBE", "0") not in ("", "0", "false", "False"):
            _mm = _mlx_mem_gib(mx)
            print(json.dumps({"stage": "mem_probe", "phase": "after_cf_mx_cache_build",
                              "n_pairs": P, "rss_gib": round(_rss_gib(), 2),
                              "mlx_active_gib": round(_mm["active"], 2),
                              "mlx_cache_gib": round(_mm["cache"], 2)}), flush=True)

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
            hosc_beta=float(model.hosc_beta),  # FEED-fb: persist CURRENT annealed beta in deploy cfg
            provenance=_run_provenance)        # #205: git sha + upstream snapshot sha in EVERY deploy ckpt
        resume_arrays = _build_resume_state_arrays(
            live_np, shadow_np, opt_np, args=args, epoch=epoch, in_feat=in_feat,
            # #205: persist the spike-guard window (bit-faithful step-skip on resume) + git provenance.
            recent_losses=recent_losses, provenance=_run_provenance)
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
    # #205: True when --resume-from lands INSIDE the Muon finisher window (start_epoch > muon_start)
    # -> the resume block rebuilds the Muon MultiOptimizer BEFORE restoring its state, and the loop's
    # muon_switched initializes True so the in-loop switch does NOT re-init a fresh (momentum-lost)
    # optimizer. Default False (fresh start / pre-finisher resume) => BIT-IDENTICAL to the prior path.
    _resume_into_finisher = False
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
        # (F2) FAIL-CLOSED render-side LEVER-drift guard (BEFORE model.update, like the film guard).
        # The loss/render-only levers add no param KEYS, so the missing-param guard above cannot see
        # them; a resume that silently drops/changes a lever the run was trained with is a
        # deterministic-repro violation. Escape: --resume-allow-lever-drift (explicit warm-start).
        if not bool(getattr(args, "resume_allow_lever_drift", False)):
            _lever_div = _resume_lever_divergences(resume_cfg, args)
            if _lever_div:
                raise ValueError(
                    f"--resume-from {rp}: {len(_lever_div)} render-side LEVER(s) DIVERGE between the "
                    "checkpoint's training config and this resume command -> a silent lever drop/change "
                    "= a deterministic-reproducibility violation (these loss/render-only levers add no "
                    "param keys, so the arch guard above cannot catch them). Diverged: "
                    + "; ".join(_lever_div)
                    + ". Fix: MATCH the trained run's lever flags, OR pass --resume-allow-lever-drift "
                    "if this is an INTENTIONAL warm-start re-treatment.")
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in rs["live"].items()]))
        mx.eval(model.parameters())
        ema_src = rs["ema"] if rs["ema"] else rs["live"]
        for k in list(ema.shadow.keys()):
            if k in ema_src:
                ema.shadow[k] = mx.array(ema_src[k])
        mx.eval(list(ema.shadow.values()))
        start_epoch = int(rs["epoch"]) + 1
        # #205 MUON-FINISHER RESUME: if the resumed epoch is INSIDE the finisher window, the saved
        # optimizer state is the Muon MultiOptimizer's -> rebuild it HERE (before the restore below)
        # and mark _resume_into_finisher so (a) the state restore keys match and (b) the loop's in-line
        # switch is skipped (muon_switched initializes True). Otherwise a resume-into-finisher would
        # re-init a FRESH optimizer at start_epoch, LOSING the Muon+AdamW momentum accumulated since
        # muon_start_epoch = a NON-bit-identical continuation (the deterministic-repro non-negotiable).
        _resume_into_finisher = (args.muon_start_epoch is not None
                                 and start_epoch > int(args.muon_start_epoch))
        if _resume_into_finisher:
            _mlr = float(args.muon_lr) if args.muon_lr is not None else 0.1 * float(args.lr)
            _malr = float(args.muon_adamw_lr) if args.muon_adamw_lr is not None else 0.1 * float(args.lr)
            _mwd = float(args.muon_weight_decay) if args.muon_weight_decay is not None else float(args.weight_decay)
            # GAP 1 (default-off): rebuild with the SAME cosine schedule the switch block built (anchored
            # on muon_start_epoch -> epochs), so the RESTORED opt.step reproduces the bit-faithful finisher
            # LR. WARM-START (GAP 2) is N/A here: the Muon momentum ('v') is restored from the checkpoint
            # below (there is no live outgoing AdamW). final_frac >= 1.0 (default) => scalar LR => the
            # rebuild is byte-identical to the pre-GAP-1 resume construction.
            _r_final_frac = float(getattr(args, "muon_lr_final_frac", 1.0))
            _r_anneal_steps = 0
            if _r_final_frac < 1.0:
                _r_steps_per_ep = max(1, (P + args.accum_pairs - 1) // args.accum_pairs)
                _r_anneal_steps = max(
                    1, (int(args.epochs) - int(args.muon_start_epoch) + 1) * _r_steps_per_ep
                )
            opt = build_muon_finisher_optimizer(
                muon_lr=_mlr, muon_adamw_lr=_malr, muon_momentum=float(args.muon_momentum),
                muon_weight_decay=_mwd, muon_ns_steps=int(args.muon_ns_steps),
                adamw_weight_decay=float(args.weight_decay),
                muon_lr_final_frac=_r_final_frac, muon_anneal_steps=_r_anneal_steps,
            )
            print(json.dumps({"stage": "resume_muon_rebuild", "start_epoch": start_epoch,
                              "muon_start_epoch": int(args.muon_start_epoch),
                              "muon_lr_final_frac": _r_final_frac,
                              "muon_lr_decay_active": bool(_r_final_frac < 1.0),
                              "muon_anneal_steps": _r_anneal_steps,
                              "note": "resuming INSIDE the Muon finisher; rebuilt MultiOptimizer before "
                              "state restore (bit-faithful finisher continuation)"}), flush=True)
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
            _rebuild_fine_dir_cache()  # AA fine self-orient (no-op unless --aa-self-orient-fine-mode)
            print(json.dumps({"stage": "resume_reorient", "mean_abs_dir_feat": round(mag, 5)}), flush=True)
        print(json.dumps({"stage": "resume", "from": str(rp), "resumed_epoch": int(rs["epoch"]),
                          "start_epoch": start_epoch, "restored_opt": restored_opt,
                          "resumed_into_finisher": bool(_resume_into_finisher)}), flush=True)

    # baseline verdict (epoch 0, or the resumed epoch) -- reflects any restored weights.
    if os.environ.get("TAC_MEM_PROBE", "0") not in ("", "0", "false", "False"):
        _mm = _mlx_mem_gib(mx)
        print(json.dumps({"stage": "mem_probe", "phase": "before_v0_verdict", "n_pairs": P,
                          "verdict_batch": int(args.verdict_batch), "rss_gib": round(_rss_gib(), 2),
                          "mlx_active_gib": round(_mm["active"], 2)}), flush=True)
    v0 = realized_verdict()
    if os.environ.get("TAC_MEM_PROBE", "0") not in ("", "0", "false", "False"):
        _mm = _mlx_mem_gib(mx)
        print(json.dumps({"stage": "mem_probe", "phase": "after_v0_verdict", "n_pairs": P,
                          "verdict_batch": int(args.verdict_batch), "rss_gib": round(_rss_gib(), 2),
                          "mlx_active_gib": round(_mm["active"], 2), "d_seg": round(v0["d_seg"], 6),
                          "d_pose": round(v0["d_pose"], 6)}), flush=True)
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
                          # reachability field ADDED to the JSON ONLY when it is on -> the OFF-path print
                          # (incl. any live --resume) is byte-identical to the pre-reachability telemetry.
                          **({"reachability": True} if (msal_reach and _sR_provider is not None) else {}),
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
    # #205: restore the spike-guard window so --resume-from is bit-faithful across a spike-skip
    # (the median gates step-skipping = part of the trajectory). DEFAULT-SAFE: no resume, or a pre-#205
    # sidecar without __recent_losses => the fresh [] is used (prior behavior). MLX-free.
    if resume_cfg is not None and "__recent_losses" in resume_cfg:
        _rl = resume_cfg["__recent_losses"]
        recent_losses = [float(x) for x in (_rl if isinstance(_rl, list) else [_rl])]
        print(json.dumps({"stage": "resume_spike_guard", "restored_recent_losses": len(recent_losses)}),
              flush=True)
    last_ep = start_epoch - 1
    stage_ckpts: list[dict[str, Any]] = []
    # CURRICULUM stage-transition spike-guard re-treat tracker (operator 2026-06-26 "different
    # stages need different treatment ... transitions must re-treat"). Init to the START epoch's
    # seg_form so a fresh-start / resume does NOT spuriously re-treat (prev == current at ep0).
    prev_seg_form = _seg_form_for_epoch(start_epoch, args)
    # MUON FINISHER (FEED-fi) per-stage optimizer switch state. muon_start_epoch None (default) =>
    # muon_switched stays False forever => the switch block + tag suffix never fire => BIT-IDENTICAL
    # to the pre-FEED-fi AdamW-throughout path. Effective LRs default to 0.1*lr (PR95 ~0.1x finetune).
    # #205: initialize True when resuming INSIDE the finisher (the opt was rebuilt as the Muon
    # MultiOptimizer in the resume block above) so the in-loop switch does NOT re-init a fresh
    # optimizer. Default False (fresh start / pre-finisher resume) => the in-loop switch fires normally.
    muon_switched = bool(_resume_into_finisher)
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
    # (#205 OOM instrumentation) env-gated per-accum-batch memory telemetry. Default OFF -> no
    # per-batch prints in production; set TAC_MEM_PROBE=1 to trace active/cache/peak/RSS for the
    # first TAC_MEM_PROBE_EPOCHS epochs (the OOM-diagnosis + fix-verification A/B). Pure observability
    # -> BIT-IDENTICAL training whether on or off.
    _mem_probe_on = os.environ.get("TAC_MEM_PROBE", "0") not in ("", "0", "false", "False")
    _mem_probe_epochs = int(os.environ.get("TAC_MEM_PROBE_EPOCHS", "3"))
    # ── compute-facet #252 activation (DEFAULT-OFF => byte-identical to the pre-#252 path) ──
    _profile_timing = bool(getattr(args, "profile_timing", False))
    set_fused_r_kernel(bool(getattr(args, "fused_r_kernel", False)))
    with temporary_mlx_device(args.mlx_device):
        if getattr(args, "fused_r_kernel", False):
            if args.mlx_device != "gpu":
                raise ValueError("--fused-r-kernel requires --mlx-device gpu (the fused R is a Metal kernel).")
            from tac.local_acceleration.metal_fused_r_operator import assert_metal_matches_cpu_oracle
            _fr_gate = assert_metal_matches_cpu_oracle()  # per-chip parity: FAILS CLOSED if not bit-identical
            print(json.dumps({"stage": "fused_r_kernel", "active": True,
                              "forward_bit_identical": bool(_fr_gate["forward_bit_identical"]),
                              "grad_bit_identical": bool(_fr_gate["grad_bit_identical"]),
                              "note": "fused Metal R roundtrip active; per-chip parity gate PASSED; buys "
                              "SPEED not score (verdict stays numpy/torch-CPU authority); pointer 0.19110 UNMOVED"}),
                  flush=True)
        _mxc = maybe_enable_mx_compile_r(bool(getattr(args, "mx_compile", False)), render_hw=(render_h, render_w))
        if _mxc:
            print(json.dumps({"stage": "mx_compile_r", "active": True, **_mxc,
                              "note": "mx.compile'd R installed (startup bit-identity gate PASSED)"}), flush=True)
        for ep in range(start_epoch, args.epochs + 1):
            _prof = {"ep_start": time.perf_counter(), "step_s": 0.0, "verdict_s": 0.0} if _profile_timing else None
            if _mem_probe_on and args.mlx_device == "gpu":
                try:
                    mx.reset_peak_memory()  # per-epoch high-water so mem_probe peak is this-epoch scoped
                except Exception:
                    pass
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
            # LEVER-4b sub-pixel boundary engagement is ALSO an AdamW->AdamW treatment boundary (mirrors
            # _bnd_lane/_bnd_msal). Default subpix_w=0.0 => never fires => bit-identical.
            _bnd_subpix = (subpix_w > 0.0 and (ep >= subpix_start) and not subpix_gate["on"])
            # (F3 fix) #224 analytic-lane render-band engagement is ALSO an AdamW->AdamW treatment
            # boundary (the band's render-target CHANGES at --lane-band-start-epoch): its sibling levers
            # (lane/margin/thin) already OR into _stage_boundary_now, but the band did NOT, so the
            # LR re-warmup + optional moment-reset never fired on band engagement -> stale AdamW momentum
            # pushed through the render-target change. Mirrors _bnd_lane exactly (computed BEFORE the band
            # gate flips at the engage block below). Default --lane-render-band OFF => _band_active False
            # => never fires => bit-identical.
            _bnd_band = (_band_active and (ep >= _band_start) and not band_gate["on"])
            _stage_boundary_now = (_bnd_curriculum or _bnd_lane or _bnd_msal or _bnd_lane_thin
                                   or _bnd_band or _bnd_subpix)
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
                # GAP 2 (default-off): capture the OUTGOING AdamW first-moment (state 'm') BEFORE `opt`
                # is rebound, to warm-start the fresh Muon momentum (state 'v'). Only a plain Adam/AdamW
                # base is transferable; anything else (or the flag off) leaves the Muon at cold zeros.
                _warm_start = bool(getattr(args, "muon_warm_start_momentum", False))
                _old_adam_state = (
                    opt.state if (_warm_start and isinstance(opt, (optim.Adam, optim.AdamW))) else None
                )
                # GAP 1 (default-off): cosine-DECAY the Muon-group LR from muon_lr -> muon_lr*final_frac
                # across the finisher span (muon_start_epoch -> epochs). Anchored on muon_start_epoch (NOT
                # `ep`) so the schedule is deterministic in the config -> a resume rebuilds the SAME
                # schedule. opt_updates_per_epoch == ceil(P / accum_pairs) (one opt.update per accum chunk;
                # spike-skips only shorten it, matching the base trainer's step-count semantics).
                # final_frac >= 1.0 (default) => muon_anneal_steps stays 0 => scalar LR => byte-identical.
                _muon_final_frac = float(getattr(args, "muon_lr_final_frac", 1.0))
                _muon_anneal_steps = 0
                if _muon_final_frac < 1.0:
                    _steps_per_ep = max(1, (P + args.accum_pairs - 1) // args.accum_pairs)
                    _muon_anneal_steps = max(
                        1, (int(args.epochs) - int(args.muon_start_epoch) + 1) * _steps_per_ep
                    )
                opt = build_muon_finisher_optimizer(
                    muon_lr=muon_lr_eff, muon_adamw_lr=muon_adamw_lr_eff,
                    muon_momentum=float(args.muon_momentum), muon_weight_decay=muon_wd_eff,
                    muon_ns_steps=int(args.muon_ns_steps), adamw_weight_decay=float(args.weight_decay),
                    # #224 Wave D (R4 #2): thread the same beta2 as the main AdamW so the finisher
                    # rest-group is consistent (default 0.999 => byte-identical).
                    adamw_beta2=float(getattr(args, "adam_beta2", 0.999)),
                    # GAP 1: default (1.0 / 0) => scalar Muon LR => byte-identical.
                    muon_lr_final_frac=_muon_final_frac, muon_anneal_steps=_muon_anneal_steps,
                )
                opt.init(model.trainable_parameters())
                mx.eval(opt.state)
                # GAP 2 (default-off): seed the fresh Muon child's momentum (v) from the captured AdamW m.
                # The Muon child is opt.optimizers[0] (MultiOptimizer([Muon, AdamW], [filter])); its state
                # flattens to '<path>.v' matching the outgoing AdamW's '<path>.m'. try/except cold-fallback
                # so a mismatch never crashes the run (deterministic-repro: cold zeros is the safe default).
                _warm_seeded = 0
                if _old_adam_state is not None:
                    try:
                        _warm_seeded = _seed_muon_momentum_from_adam(opt.optimizers[0], _old_adam_state)
                    except Exception as _warm_err:  # fall back to cold start; never crash the run
                        _warm_seeded = -1
                        print(json.dumps({
                            "stage": "muon_warm_start_FAILED_cold_fallback", "epoch": ep,
                            "err": str(_warm_err),
                        }), flush=True)
                    mx.eval(opt.state)
                muon_switched = True
                recent_losses.clear()
                print(json.dumps({"stage": "muon_finisher_switch", "epoch": ep,
                                  "muon_start_epoch": int(args.muon_start_epoch), "muon_lr": muon_lr_eff,
                                  "muon_adamw_lr": muon_adamw_lr_eff, "muon_momentum": float(args.muon_momentum),
                                  "muon_ns_steps": int(args.muon_ns_steps), "muon_weight_decay": muon_wd_eff,
                                  "n_muon_params": n_muon, "n_adamw_params": n_adamw,
                                  "muon_lr_final_frac": _muon_final_frac,
                                  "muon_lr_decay_active": bool(_muon_final_frac < 1.0),
                                  "muon_anneal_steps": _muon_anneal_steps,
                                  "muon_warm_start_momentum": _warm_start,
                                  "muon_warm_seeded_leaves": _warm_seeded,
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
            # LEVER-4b sub-pixel boundary engagement gate + transition RE-TREAT (same discipline as LEVER-4).
            if subpix_w > 0.0:
                _subpix_was = subpix_gate["on"]
                subpix_gate["on"] = lever_gate_on_at_epoch(subpix_w, subpix_start, ep)
                if subpix_gate["on"] and not _subpix_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "seg_subpix_boundary_engage", "epoch": ep, "start": subpix_start,
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
            # #224 (2) analytic-lane render-band engagement gate + transition RE-TREAT (mirrors the
            # lane/margin/thin gates). No-op when --lane-render-band off (band never applies).
            if _band_active:
                _band_was = band_gate["on"]
                band_gate["on"] = _band_start <= ep
                if band_gate["on"] and not _band_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "lane_render_band_engage", "epoch": ep, "start": _band_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # #224 (4) persistence loss anneal (linear warm-up; coarse->fine). No-op when persist_w=0.
            if persist_w > 0.0 and persist_classes:
                persist_gate["w"] = persistence_anneal_weight(ep, persist_w, persist_warmup)
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
                    # #224 Wave C FIX-1: the fresh moment-reset optimizer inherits the SAME bias_correction
                    # gate as the main construction (ON only on the high-beta2 all-levers path). A fresh
                    # AdamW resets step->0, so bias_correction correctly re-warms the reset moments.
                    opt = optim.AdamW(learning_rate=float(opt.learning_rate),
                                      weight_decay=args.weight_decay,
                                      betas=[0.9, float(getattr(args, "adam_beta2", 0.999))],
                                      bias_correction=_adam_bias_correction_for(
                                          getattr(args, "adam_beta2", 0.999)))
                    opt.init(model.trainable_parameters())
                    mx.eval(opt.state)
                    print(json.dumps({"stage": "stage_transition_reset_moments", "epoch": ep,
                                      "from_curriculum": bool(_bnd_curriculum),
                                      "from_lane_engage": bool(_bnd_lane),
                                      "from_margin_saliency_engage": bool(_bnd_msal),
                                      "from_lane_thin_engage": bool(_bnd_lane_thin),
                                      "from_lane_render_band_engage": bool(_bnd_band),
                                      "note": "AdamW m/v zeroed (fresh optimizer); spike-guard already "
                                      "re-treated; stale-momentum-through-landscape-change avoided"}),
                          flush=True)
            # SELF-ORIENT reorient cadence (fixed-point): recompute per-pair directional feats from
            # the EMA deploy argmax every --reorient-every epochs (skip ep1: argmax is random).
            if use_self_orient and ep > 1 and (ep - 1) % max(args.reorient_every, 1) == 0:
                ema_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
                mag = recompute_self_orient(int8_dequant_params(ema_np))
                _rebuild_cf_mx_cache()
                _rebuild_fine_dir_cache()  # AA fine self-orient (no-op unless --aa-self-orient-fine-mode)
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
            # (THETA* TIER-2 MUST-3) SWA / wider-finisher EMA: from the finisher-start epoch onward,
            # widen the EMA decay so the EMA shadow averages over the late oscillation (a flat-basin
            # center). Idempotent per-epoch set (keys off `ep`, not state) => RESUME-safe (ema.decay
            # is not persisted; re-applied on resume into the finisher window). DEFAULT-OFF:
            # ema_finisher_decay None => ema.decay is NEVER touched => the EMA trajectory is
            # BIT-IDENTICAL to the --ema-decay path.
            if (ema_finisher_decay is not None and ema_finisher_start is not None
                    and ep >= ema_finisher_start and ema.decay != ema_finisher_decay):
                _prev_decay = ema.decay
                ema.decay = ema_finisher_decay
                print(json.dumps({"stage": "ema_finisher_widen", "epoch": ep,
                                  "from_decay": float(_prev_decay), "to_decay": float(ema_finisher_decay),
                                  "start_epoch": int(ema_finisher_start),
                                  "note": "SWA-style wider EMA averaging for the finisher (flat-basin "
                                  "center over the late oscillation)"}), flush=True)
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
            if _prof is not None:
                _prof["_step0"] = time.perf_counter()  # #252 profile: fwd+bwd+opt+ema step start
            for s in range(0, P, args.accum_pairs):
                chunk = order[s:s + args.accum_pairs]
                accum = None
                accum_seed = None   # #224 (5): seed grad accumulator (None unless --seed-islands)
                lsum = 0.0
                if _use_micro_batch:
                    # (--micro-batch-pairs B) sub-batch each accum chunk into B-pair groups; ONE batched
                    # value_and_grad per group. Weight each group's MEAN grad/loss by its pair count so
                    # sum-over-groups / nb == the serial per-pair mean-over-chunk (mean_grads + batch_loss
                    # below are UNCHANGED). _use_micro_batch guarantees seed_mod is None (fail-closed
                    # at build) so there is NO seed co-grad leg here.
                    _B = _micro_batch_pairs
                    for _ss in range(0, len(chunk), _B):
                        _sub = [int(p) for p in chunk[_ss:_ss + _B]]
                        _bn = len(_sub)
                        loss_b, grads_b = value_and_grad_batch(
                            model,
                            [_cf_mx(p) for p in _sub],
                            [2 * p + 0 for p in _sub], [2 * p + 1 for p in _sub],
                            [lstar_cache[p][0] for p in _sub], [lstar_cache[p][1] for p in _sub],
                            [pose_tgts[p] for p in _sub],
                            args.w_seg, args.w_pose, args.hinge_weight, args.margin_target_end, seg_form,
                            args.eikonal_weight, args.length_weight,
                        )
                        mx.eval(loss_b, grads_b)  # materialize per group (bound the lazy fwd+bwd graph)
                        lsum += float(loss_b) * _bn          # mean-over-group * count = group sum
                        _wg = tree_map(lambda g, c=float(_bn): g * c, grads_b)  # mean-grad * count = group-sum grad
                        accum = _wg if accum is None else tree_map(lambda a, b: a + b, accum, _wg)
                        mx.eval(accum)
                else:
                    for pi_np in chunk:
                        pi = int(pi_np)
                        oh, mg = lstar_cache[pi]
                        if _dual_vg is None:
                            loss, grads = value_and_grad(
                                model, _cf_mx(pi), 2 * pi + 0, 2 * pi + 1, oh, mg, pose_tgts[pi],
                                args.w_seg, args.w_pose, args.hinge_weight, args.margin_target_end, seg_form,
                                args.eikonal_weight, args.length_weight,
                            )
                        else:
                            # #224 (5) dual co-grad: witness grads[0] (== the single-path grads, same loss/
                            # params) + seed grads[1]. The seed leg is accumulated + shielded separately below.
                            loss, (grads, sgrads) = _dual_vg(
                                model.trainable_parameters(), seed_mod.trainable_parameters(),
                                _cf_mx(pi), 2 * pi + 0, 2 * pi + 1, oh, mg, pose_tgts[pi],
                                args.w_seg, args.w_pose, args.hinge_weight, args.margin_target_end, seg_form,
                                args.eikonal_weight, args.length_weight,
                            )
                            accum_seed = sgrads if accum_seed is None else tree_map(lambda a, b: a + b, accum_seed, sgrads)
                            mx.eval(accum_seed)
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
                    # (#205 OOM fix) still return the render/backward buffer POOL to the OS even on a
                    # spike-skipped batch, so a RUN of consecutive skips cannot balloon the Metal cache.
                    if (args.mlx_device == "gpu" and args.mlx_cache_clear_accum > 0
                            and ((s // args.accum_pairs) % args.mlx_cache_clear_accum == 0)):
                        mx.clear_cache()
                    continue
                opt.update(model, clipped)
                mx.eval(model.parameters(), opt.state)
                # #224 (5) SEED CONTAINMENT step: shield the seed grad (defend the seeded islands from
                # the bulk-CE wash) then apply the SEPARATE seed AdamW. The shield touches ONLY the seed
                # 'residual' leaf; the witness opt.update above + MD-decoupling below + grouped-backward
                # are all UNTOUCHED (distinct optimizer + distinct param tree). Gated by the same spike
                # skip as the witness step (only steps when the batch was not spike-skipped).
                if seed_mod is not None and accum_seed is not None:
                    _mean_sg = tree_map(lambda g, c=float(nb): g / c, accum_seed)
                    _sg = _seed_shield(_mean_sg["residual"], seed_mod.residual, seed_spec)  # leaf-only shield
                    seed_opt.update(seed_mod, {"residual": _sg})
                    mx.eval(seed_mod.parameters(), seed_opt.state)
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
                # (#205 OOM fix) return the Metal buffer POOL to the OS every N accum-batches. The
                # lazy graph is already materialized per-pair; this frees the CACHED (already-freed)
                # render+backward buffers so peak RSS ~= active working set + one batch (NOT a whole
                # epoch's freed-buffer pool). clear_cache never touches LIVE arrays => BIT-IDENTICAL.
                _bidx = s // args.accum_pairs
                if (args.mlx_device == "gpu" and args.mlx_cache_clear_accum > 0
                        and (_bidx % args.mlx_cache_clear_accum == 0)):
                    mx.clear_cache()
                if _mem_probe_on and ep <= _mem_probe_epochs:
                    _mm = _mlx_mem_gib(mx)
                    print(json.dumps({"stage": "mem_probe", "ep": ep, "accum_batch": _bidx,
                                      "rss_gib": round(_rss_gib(), 2),
                                      "mlx_active_gib": round(_mm["active"], 2),
                                      "mlx_cache_gib": round(_mm["cache"], 2),
                                      "mlx_peak_gib": round(_mm["peak"], 2),
                                      "clear_accum": int(args.mlx_cache_clear_accum)}), flush=True)
            if _prof is not None:
                _prof["step_s"] = time.perf_counter() - _prof["_step0"]  # #252 profile: step (fwd+bwd+opt+ema)
            if args.mlx_device == "gpu":
                mx.clear_cache()
            # #224 (5) SEED SURVIVAL telemetry: mean |seed residual| ON the island support. The
            # containment shield should keep this ABOVE ~0 (the seeded islands survive the bulk-CE
            # wash); WITHOUT the shield the bulk wash drives it toward 0 (the failure this defends).
            # Purely observational (never read back). Default OFF (seed_mod None) => never fires.
            if seed_mod is not None and (ep % args.eval_every == 0 or ep == args.epochs):
                _sr = np.asarray(seed_mod.residual)                       # (P,H,W,3)
                _sm = np.stack([np.asarray(m) for m in seed_state["masks"]], axis=0)  # (P,H,W,1)
                _mon = float(np.sum(np.abs(_sr) * _sm) / (np.sum(_sm) * 3.0 + 1e-9))
                print(json.dumps({"stage": "seed_survival", "epoch": ep,
                                  "mean_abs_seed_on_island": round(_mon, 5),
                                  "containment_mode": str(args.containment_mode),
                                  "note": "shield keeps seeded islands alive vs bulk-CE wash (advisory)"}),
                      flush=True)
            if _prof is not None:
                _prof["_v0"] = time.perf_counter()  # #252 profile: verdict start
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
            if _prof is not None:
                _prof["verdict_s"] += time.perf_counter() - _prof["_v0"]  # #252 profile: verdict wall-clock
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
            # ── #252 per-epoch timing emit (advisory; at eval cadence so no per-epoch spam). The
            # split is fwd+bwd-step (INR+R+scorer+loss+backward+opt+ema, fused inside value_and_grad)
            # vs verdict vs overhead (gates/reorient/permutation/LR). R is NOT separable inside the
            # fused graph, so its share is measured DIRECTLY by an isolated in-situ R micro-bench at the
            # real render resolution (reference vs fused, fwd + fwd+bwd) -> R_fraction = R_fwdbwd *
            # frames/epoch / step_s, and the realized whole-run speedup follows by Amdahl. Emitted only
            # when --profile-timing (default OFF => this whole block is skipped => byte-identical).
            if _prof is not None and (ep % args.eval_every == 0 or ep == args.epochs):
                _ep_s = time.perf_counter() - _prof["ep_start"]
                _frames = int(2 * len(order))  # 2 frames (f0,f1) per pair-visit this epoch
                _rmb = r_isolated_microbench(render_h=render_h, render_w=render_w, n_frames=2, reps=15)
                _step_s = float(_prof["step_s"])
                _rfwdbwd_ms = _rmb.get("ref_fwdbwd_ms_per_frame")
                _r_share = (
                    (_rfwdbwd_ms / 1e3 * _frames / _step_s) if (_rfwdbwd_ms and _step_s > 0) else None)
                print(json.dumps({
                    "stage": "profile_timing", "epoch": ep,
                    "t_epoch_s": round(_ep_s, 4),
                    "t_step_fwd_bwd_opt_ema_s": round(_step_s, 4),
                    "t_verdict_s": round(float(_prof["verdict_s"]), 4),
                    "t_overhead_s": round(max(_ep_s - _step_s - float(_prof["verdict_s"]), 0.0), 4),
                    "frames_per_epoch": _frames,
                    "R_isolated": _rmb,
                    "R_fraction_of_step_est": (round(_r_share, 4) if _r_share is not None else None),
                    "fused_r_active": bool(getattr(args, "fused_r_kernel", False)),
                    "note": "R fraction from isolated in-situ R fwd+bwd; whole-run speedup by Amdahl "
                    "1/((1-f)+f/su_R); advisory, buys SPEED not score; pointer 0.19110 UNMOVED"}),
                    flush=True)
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
        # #205 PROVENANCE (deterministic-reproducibility: git sha + upstream snapshot sha + seed).
        "provenance": {**_run_provenance, "seed": int(args.seed)},
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
        # THETA* TIER-2 levers (deterministic-reproducibility: record config with the result). All
        # default-OFF => these values reflect the bit-identical path.
        "tau_anneal_shape": str(getattr(args, "tau_anneal_shape", "cosine")),
        "tau_hold_frac": float(getattr(args, "tau_hold_frac", 1.0)),
        "code_nuclear_weight": float(getattr(args, "code_nuclear_weight", 0.0)),
        "code_nuclear_eps": float(getattr(args, "code_nuclear_eps", 1e-3)),
        "code_nuclear_ns_iters": int(getattr(args, "code_nuclear_ns_iters", 25)),
        "ema_decay_finisher": (float(args.ema_decay_finisher)
                               if getattr(args, "ema_decay_finisher", None) is not None else None),
        "ema_decay_finisher_start_epoch": (int(args.ema_decay_finisher_start_epoch)
                                           if getattr(args, "ema_decay_finisher_start_epoch", None) is not None else None),
        "eikonal_junction_relax": float(getattr(args, "eikonal_junction_relax", 0.0)),
        "eikonal_junction_tau": float(getattr(args, "eikonal_junction_tau", 0.5)),
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
    ap.add_argument("--resume-allow-lever-drift", action=argparse.BooleanOptionalAction, default=False,
                    help="(F2) allow a --resume-from whose render-side LEVERS (lane_render_band / "
                    "persistence_loss_weight / amplify_weight / lane_band_start_epoch / render_aa / "
                    "hosc_beta_end / mod_dim) DIFFER from the checkpoint's training config. DEFAULT OFF "
                    "= FAIL-CLOSED (a silent lever drop is a deterministic-repro violation). Set ON only "
                    "for an INTENTIONAL warm-start re-treatment (loss/render-only levers add no params).")
    ap.add_argument("--freeze-decoder-fit-codes", type=str, default=None,
                    help="FEED-eo AMORTIZATION (days->hours): load the SHARED decoder from this level-set "
                    "EMA/deploy npz (trained on a SUBSET, e.g. n96/n192), FREEZE it, and fit ONLY the "
                    "per-pair codes for all --num-pairs pairs (embarrassingly-parallel per-pair latent "
                    "fit through the frozen render+R+scorer). The front-end config (--bank-*/--max-bank-"
                    "freq/--self-orient/--n-dir-freqs) MUST match the decoder's in_feat. Incompatible "
                    "with --resume-from/--structured-init. DEFAULT None = normal joint train.")
    # ---- RESIDUAL-ONLY MODE (v2 hybrid; gap #1; ADDITIVE, default-OFF => BIT-IDENTICAL). The
    # rate-bearing fix: train the small INR on the RESIDUAL the FIXED deterministic bulk leaves,
    # with the bulk GENERATED at decode (OUTSIDE the counted weights) and COMPOSED before R --
    # NOT baked into the weights via --structured-init (which does NOT shrink the rate). Every
    # realized render becomes ``composed = where(bulk_label_mask, INR, bulk)`` (the mask is
    # bulk-LABEL-derived => regenerated FREE at inflate, 0 counted bytes). The d_seg loss + ALL
    # surgical levers (lane-thin/margin-saliency/hardness) then weight the COMPOSED-render d_seg,
    # so the INR only has to flip the Lane+Movable residual annulus -> it can be SMALL (the rate
    # win). --residual-mode OFF (default) => NONE of this fires => byte-identical to the
    # full-partition witness. See tac.v2_compose.residual_compose + the landing memo.
    ap.add_argument("--residual-mode", action=argparse.BooleanOptionalAction, default=False,
                    help="RESIDUAL-ONLY MODE (v2 hybrid): compose the FIXED deterministic bulk (+) "
                    "the small INR residual before R; train the INR on the COMPOSED-render d_seg. "
                    "Requires --residual-target-npz. DEFAULT OFF => byte-identical full-partition "
                    "witness. The rate win: the bulk is OUTSIDE the counted weights (NOT "
                    "--structured-init, which bakes it IN).")
    ap.add_argument("--residual-target-npz", type=str, default=None,
                    help="RESIDUAL-ONLY MODE input: the residual training bundle "
                    "(tac.v2_compose.residual_compose.save_residual_training_bundle) carrying the "
                    "deterministic bulk RGB (render res, pre-R) + the bulk-derived composition mask "
                    "per pair. Required when --residual-mode. The COUNTED bytes are the INR weights "
                    "this run produces -- NEVER this bundle.")
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
    # (THETA* TIER-2 MUST-1) softmax-temp anneal SHAPE (additive; default 'cosine' == bit-identical to
    # the pre-theta* cosine). 'geometric' = log-spaced decay (more epochs at small tau; damps late-tau
    # d_seg volatility). 'cosine_hold' = cosine to the floor at --tau-hold-frac, then HOLD at the end.
    ap.add_argument("--tau-anneal-shape", choices=["cosine", "geometric", "cosine_hold"], default="cosine",
                    help="THETA* MUST-1: softmax-temp anneal curve. cosine (default, bit-identical) | "
                    "geometric (log-spaced, more epochs at small tau) | cosine_hold (reach floor at "
                    "--tau-hold-frac then hold). geometric requires --softmax-temp-start/-end > 0.")
    ap.add_argument("--tau-hold-frac", type=float, default=1.0,
                    help="THETA* MUST-1: for --tau-anneal-shape cosine_hold, the fraction (0,1] of the "
                    "anneal window at which tau reaches --softmax-temp-end and HOLDS. 1.0 (default) = "
                    "no hold = BIT-IDENTICAL to cosine.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # (#222 deep-math gap-1) Adam second-moment decay beta2. Default 0.999 == the MLX AdamW default
    # betas=[0.9, 0.999] => BIT-IDENTICAL to the pre-flag path. The n600 accumulated-microbatch regime
    # has only n = P/accum_pairs ~ 75 optimizer steps per epoch's worth of distinct gradient statistics;
    # the arXiv 2603.02092 small-n rule 1-beta2 <~ (1-beta1^5)/n^3.5 => for beta1=0.9, n=75:
    # (1-0.59049)/75^3.5 ~ 1.12e-7 => beta2* ~ 0.99999988. The default 0.999 (1-beta2=1e-3) is ~4 orders
    # ABOVE that floor = under-smoothed for n~75. The launch config (witness_autoconfig all_levers)
    # sets 0.9999999 (1-beta2=1e-7 < 1.12e-7 => clears the threshold). beta1 stays 0.9 (MLX default).
    ap.add_argument("--adam-beta2", type=float, default=0.999,
                    help="#222 AdamW second-moment decay beta2 (beta1 fixed 0.9). Default 0.999 = MLX "
                    "default => bit-identical. Small-n (n~75 accum steps) optimum ~0.9999999 per "
                    "arXiv 2603.02092 (1-beta2 <~ (1-beta1^5)/n^3.5).")
    ap.add_argument("--ema-decay", type=float, default=0.997)
    # (THETA* TIER-2 MUST-3) SWA / wider-finisher EMA (additive; default None == bit-identical to the
    # --ema-decay path). When set, from the resolved finisher-start epoch onward the EMA uses this
    # WIDER decay (averages over the late oscillation -> flat-basin center, SWA-style).
    ap.add_argument("--ema-decay-finisher", type=float, default=None,
                    help="THETA* MUST-3: wider EMA decay applied from the finisher-start epoch onward "
                    "(SWA-style late-oscillation averaging). None (default) = use --ema-decay everywhere "
                    "= BIT-IDENTICAL. Typically > --ema-decay (e.g. 0.999/0.9995). Must be in (0,1).")
    ap.add_argument("--ema-decay-finisher-start-epoch", type=int, default=None,
                    help="THETA* MUST-3: 1-based epoch at which the wider --ema-decay-finisher engages. "
                    "None (default) = fall back to --muon-start-epoch. Required (here or via "
                    "--muon-start-epoch) when --ema-decay-finisher is set.")
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
    # (SPEED LEVER, DAG FEED 2026-07-03c) micro-batch B pairs per forward. DEFAULT 1 => the accum loop
    # takes the UNCHANGED serial per-pair value_and_grad path (BYTE-IDENTICAL). B>1 renders + scores B
    # pairs in ONE batched frozen-scorer forward (EfficientNet-B2 SegNet / FastViT PoseNet saturate the
    # GPU far better than single-pair batches) -> the measured ~2-4x speed lever. NOT bit-identical
    # (batched fp reduction order) => a trajectory-affecting opt-in, validated by a short trajectory A/B.
    # The per-pair loss reductions (base seg-form, score-domain pose sqrt, weighted-mean levers) are
    # computed PER PAIR on the batched scorer outputs and MEAN-ed over B, so total_loss_fn_batch(B) ==
    # mean_b total_loss_fn(pair_b) within fp tolerance -> the accum-loop grad matches the serial
    # mean-over-chunk EXACTLY (see the accum loop's per-group `* _bn` weighting). Incompatible with
    # --seed-islands (fail-closed at build). Score-neutral verdict authority is unaffected.
    ap.add_argument("--micro-batch-pairs", type=int, default=1,
                    help="(speed lever) pairs per batched value_and_grad forward (1 = serial "
                    "byte-identical per-pair path; >1 = opt-in batched scorer forward, trajectory-"
                    "affecting, ~2-4x). Sub-batches each --accum-pairs chunk; grads weighted by pair "
                    "count so the accum-step grad == the serial mean-over-chunk. NOT with --seed-islands.")
    # (--cache-gt-skeleton, #260 SPEED, BIT-IDENTICAL) opt-in per-pair cache of the CONSTANT GT
    # soft-skeleton the persistence loss recomputes every step. sg=soft_skeleton(gt) is a function of
    # the FROZEN GT argmax one-hot ONLY (constant across epochs) + carries NO gradient (it multiplies
    # pred in tsens), so precomputing it once per pair + reusing via sg_precomputed= is BIT-IDENTICAL
    # (a materialized concrete constant == the inline recompute) while skipping ~half the clDice cost.
    # Default OFF => total_loss_fn passes sg_precomputed=None => byte-identical to the pre-flag path.
    # No-op unless --persistence-loss-weight>0 (the only consumer); skipped under --micro-batch-pairs>1.
    ap.add_argument("--cache-gt-skeleton", action="store_true",
                    help="(speed, bit-identical) cache the CONSTANT per-pair GT soft-skeleton for the "
                    "persistence loss (sg=soft_skeleton(gt) is epoch-invariant + gradient-free). "
                    "Default OFF = byte-identical. No-op unless --persistence-loss-weight>0; "
                    "skipped under --micro-batch-pairs>1 (serial total_loss_fn is the only consumer).")
    # (#205 OOM FIX) MLX Metal caching-allocator hygiene. The lazy graph is already materialized
    # per-pair (mx.eval(loss, grads) + mx.eval(accum)); the leak is the Metal buffer POOL (freed
    # render/backward buffers stay CACHED, not returned to the OS) growing across an epoch's ~P/8
    # accum-batches -> a ~15 GiB active working set peaked at 90 GiB and tripped the 90 GB safe-run
    # guard (killed the run before the first checkpoint). Calling mx.clear_cache() every N accum
    # batches returns the pool to the OS -> peak RSS ~= active + one batch. clear_cache frees ONLY
    # pooled (already-freed) buffers, NEVER live arrays, and MLX is lazy-but-deterministic -> WHEN we
    # clear the pool cannot change WHAT is computed => BIT-IDENTICAL loss/d_seg (verified n64 A/B).
    # 1 = clear every accum-batch (safest peak); 0 = never inside the loop (the pre-fix behaviour,
    # for the A/B). GPU-only (no-op on cpu). The existing per-epoch clear at loop-end is preserved.
    ap.add_argument("--mlx-cache-clear-accum", type=int, default=1,
                    help="(#205 OOM fix) mx.clear_cache() every N accum-batches inside the epoch loop "
                    "(GPU only; score-neutral). 0 disables the in-loop clear (pre-fix behaviour).")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--spike-factor", type=float, default=5.0)
    # (#205 REAL OOM fix) chunk the CPU-scorer verdict inference into vbatch-pair torch batches so
    # the fp32 (N,2,3,874,1164) cast + EfficientNet/FastViT activations do NOT spike ~30-50 GiB at
    # N=600 on top of the resident ~41 GiB self-orient cf_mx_cache (the 90 GB OOM). BIT-IDENTICAL
    # (eval-mode BN running stats). 0 = single N-wide batch (pre-fix, for the A/B parity check).
    ap.add_argument("--verdict-batch", type=int, default=32,
                    help="(#205 OOM fix) CPU-scorer verdict inference chunk size (pairs per torch "
                    "batch); 0 = single N-wide batch (pre-fix). Score-neutral (eval-mode BN).")
    ap.add_argument("--verdict-pairs", type=int, default=24,
                    help="realized fp32-numpy EMA-shadow verdict subset (0=all); ALWAYS fp32 one-codepath, never mlx-gpu.")
    ap.add_argument("--async-verdict", action=argparse.BooleanOptionalAction, default=False,
                    help="FEED-em: run the OBSERVATIONAL CPU-torch verdict in a BACKGROUND THREAD off a "
                    "point-in-time snapshot so the MLX-GPU loop never idles (~4.7%% wall-clock reclaim @ "
                    "n600). BIT-IDENTICAL training (the verdict is never read back); only the verdict "
                    "CADENCE may self-throttle under load (at-most-one in-flight). DEFAULT OFF = the "
                    "current synchronous bit-identical behavior.")
    ap.add_argument("--mlx-device", choices=["gpu", "cpu"], default="gpu")
    # ── compute-facet #252 (MLX + custom Metal). All DEFAULT-OFF + bit-identical when off. ──
    ap.add_argument("--fused-r-kernel", action=argparse.BooleanOptionalAction, default=False,
                    help="(#252) swap the pure-MLX R roundtrip for the fused Metal kernel "
                    "(metal_fused_r_operator; bit-identical fwd to the numpy-fp32 authority, ~1 ULP VJP). "
                    "A startup per-chip parity gate (assert_metal_matches_cpu_oracle) fails CLOSED if the "
                    "kernel is not bit-identical on this GPU. NO-FAKE: buys SPEED, never a score. Default OFF.")
    ap.add_argument("--mx-compile", action=argparse.BooleanOptionalAction, default=False,
                    help="(#252) install an mx.compile'd reference R, GATED by a startup bit-identity check. "
                    "MEASURED 2026-07-03: mx.compile reintroduces fp-contraction that flips the uint8-STE "
                    "d_seg argmax (fwd Δ~4.8e-3, ~1.11x) so this FAILS CLOSED on non-bit-identical hosts. "
                    "Prefer --fused-r-kernel. Default OFF.")
    ap.add_argument("--profile-timing", action=argparse.BooleanOptionalAction, default=False,
                    help="(#252) emit per-epoch wall-clock phase split (fwd+bwd step / opt+ema / verdict / "
                    "overhead) + an isolated R micro-bench (fwd and fwd+bwd, reference vs fused) so the R "
                    "fraction -> realized whole-run speedup is MEASURED, not estimated. Advisory; DEFAULT OFF "
                    "=> zero added work, byte-identical.")
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
    # #218 MARGIN-FIELD HEAD levers (facets 1 & 3, BYTE-FREE; see src/tac/boundary_math/laguerre_logit_offset.py
    # + experiments/probe_laguerre_logit_offset_sweep.py). Default (head=softmax, weight 0) => byte-identical.
    ap.add_argument("--head", choices=["softmax", "etf", "additive-margin"], default="softmax",
                    help="#218 facet-1: out_sdf head geometry. 'etf'=fixed simplex-ETF weight (frozen, "
                    "byte-free + rate-win, neural-collapse minority-norm fix). 'additive-margin'=use the AM "
                    "realized-margin hinge target from --additive-margin. 'softmax'=default (byte-identical).")
    ap.add_argument("--additive-margin", type=float, default=0.0,
                    help="#218 facet-1b: AM-softmax margin (target realized SegNet decision margin) fed to the "
                    "margin-field hinge when --head additive-margin.")
    ap.add_argument("--logit-adjust-per-class", action="store_true",
                    help="#218 facet-3 (Menon 2007.07314): raise the realized-margin target for RARE classes "
                    "(Lane/Movable) by tau*relu(-log pi_c). Byte-free. Needs --margin-field-head-weight>0.")
    ap.add_argument("--logit-adjust-tau", type=float, default=1.0, help="#218 facet-3 tau scale.")
    ap.add_argument("--margin-field-head-weight", type=float, default=0.0,
                    help="#218 facets 1b/3 loss weight for the realized through-R per-class margin hinge "
                    "(0.0=off=byte-identical). Composes with LEVER-3/4/B on the shared _signed.")
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
    ap.add_argument("--margin-saliency-reachability", action="store_true",
                    help="LEVER-4 REACHABILITY (REPLACES the UNIWARD texture path when set): multiply the "
                    "fragility saliency by the cached THROUGH-R fragility-weighted margin-Jacobian S_R "
                    "(reachability of the CORRECT answer at the GT target frame) instead of 1/(1+beta*tex). "
                    "The texture proxy was MEASURED inert (Pearson -0.033 vs S_R, top-5%% Jaccard 0.024 = "
                    "statistical chance, mildly misdirects); S_R lives on the fragile margin band where the "
                    "d_seg debt is. Requires an 'sR' key in --gt-cache (build via "
                    "tools/precompute_sR_reachability.py). Default OFF => byte-identical (texture path "
                    "unchanged). NOT supported with --micro-batch-pairs>1 (serial path only; fails closed).")
    # LEVER-4b SUB-PIXEL BOUNDARY-PLACEMENT `t` (asymmetry probe a8afad40 GREEN 2026-07-03; DIRECTIONAL
    # upgrade of LEVER-4 #141; ADDITIVE, DEFAULT-OFF). Supervises the witness's realized margin ratio
    # t_wit = Mw[p]/(Mw[p]+Mw[q]) toward the FREE GT cross-boundary margin ratio t = M_GT[p]/(M_GT[p]+
    # M_GT[q]) on genuine-V straddles (a denser sub-pixel placement signal than the argmax weight). Reuses
    # the SHARED realized through-R margin (no 2nd SegNet forward). subpix_w=0.0 (DEFAULT) => byte-identical.
    ap.add_argument("--seg-subpix-boundary-weight", type=float, default=0.0,
                    help="LEVER-4b: weight on the additive sub-pixel boundary-placement loss "
                    "(t_wit - t_GT)^2 over genuine-V inter-class straddles (0=off). The GT target "
                    "t=M_GT[p]/(M_GT[p]+M_GT[q]) is a FREE sub-pixel localizer latent in the GT margin "
                    "field; supervises the witness's OWN realized margin ratio. Reuses the SHARED "
                    "LEVER-4 through-R margin forward. NOT supported with --micro-batch-pairs>1 (fails closed).")
    ap.add_argument("--seg-subpix-boundary-v-band", type=float, default=1.0,
                    help="LEVER-4b: genuine-V flip-band. A straddle qualifies only when BOTH GT margins "
                    "are < this (t is meaningful only where the margin V is clean). MEASURED gt_n96: "
                    "band 1.0 -> ~2196 active px/frame (1.12%% of px), t mean 0.527 std 0.263 "
                    "(informative ~Uniform); the straddle set saturates by ~2.0 (boundary pixels are "
                    "already low-margin on both sides).")
    ap.add_argument("--seg-subpix-boundary-start-epoch", type=int, default=0,
                    help="LEVER-4b OPTIMAL-FORM: engage only at ep>=this (0=from ep1). Gate to the "
                    "tau_softplus/l7 margin stage (placement is meaningful once the argmax is roughly "
                    "correct); the engage epoch re-treats the spike-guard (same discipline as LEVER-4).")
    # SPIKE-AWARE seg REWEIGHT (source-split MEASURED n600 2026-07-03; ADDITIVE, DEFAULT-OFF). Reweight
    # the per-pixel base seg CE by a theta-INDEPENDENT map from the GT argmax TEMPORAL neighbors: a SPIKE
    # pixel (lstar[t] != lstar[t-1] AND != lstar[t+1]) is single-frame argmax FLICKER a per-frame witness
    # structurally CANNOT fit. MEASURED: the flicker is ~88.6%% IRREDUCIBLE appearance-change (spike luma
    # temporal-delta 34 vs 4 stable = 8.4x) -> a SMOOTH witness is PROVABLY optimal there (d_seg=q(1-r)+
    # r(1-q), min r=0 for q<0.5). DOWN-weight the unfittable flicker gradient; UP-weight the COHERENT
    # temporally-consistent boundary (the winnable residual). Default scalars 1.0/1.0 => map==1.0 =>
    # BYTE-IDENTICAL even with --seg-spike-reweight set. MODEST headroom (live residual d_seg ~ the
    # popout floor; benefit is 2nd-order reallocation) -> an A/B arm, NOT a claim. Store-the-flicker is
    # net-NEGATIVE (rate +0.56 > d_seg 0.52) and the REPLICATE alternative is not warranted (predictable
    # fraction ~11.4%%, weak ego-coupling r=0.16). NOT supported with --micro-batch-pairs>1 (fails closed).
    ap.add_argument("--seg-spike-reweight", action="store_true",
                    help="Enable the spike-aware per-pixel seg-CE reweight (DEFAULT OFF => byte-identical). "
                    "Down-weights unfittable single-frame argmax flicker, up-weights the coherent boundary.")
    ap.add_argument("--seg-spike-downweight", type=float, default=1.0,
                    help="Per-pixel seg-loss weight at SPIKE (single-frame flicker) pixels. <1.0 down-weights "
                    "the unfittable flicker gradient. 1.0 (DEFAULT) => no change (byte-identical).")
    ap.add_argument("--seg-coherent-upweight", type=float, default=1.0,
                    help="Per-pixel seg-loss weight at COHERENT (temporally-consistent, unstable) boundary "
                    "pixels. >1.0 concentrates capacity on the winnable residual. 1.0 (DEFAULT) => no change.")
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
    # (THETA* TIER-2 MUST-2) nuclear-norm low-rank code penalty (additive; default 0.0 == OFF ==
    # bit-identical loss). Drives the per-pair FiLM codes toward a low-rank subspace (rate). Computed
    # as a DIFFERENTIABLE smoothed nuclear norm via Newton-Schulz matrix-sqrt trace (MLX has no svd/
    # eigvalsh vjp); see _nuclear_norm_smooth_mlx.
    ap.add_argument("--code-nuclear-weight", type=float, default=0.0,
                    help="THETA* MUST-2: weight on the smoothed nuclear norm of the per-pair code "
                    "matrix (low-rank -> rate). 0.0 (default) = OFF = bit-identical loss.")
    ap.add_argument("--code-nuclear-eps", type=float, default=1e-3,
                    help="THETA* MUST-2: relative smoothing floor for the nuclear norm (keeps "
                    "Newton-Schulz stable on rank-deficient codes). ~0.3%% bias on well-conditioned "
                    "inputs at 1e-3. Must be > 0.")
    ap.add_argument("--code-nuclear-ns-iters", type=int, default=25,
                    help="THETA* MUST-2: Newton-Schulz iterations for the matrix sqrt (converged by "
                    "~25 for mod_dim<=48). Must be >= 1.")
    # (THETA* TIER-2 STRETCH-1) junction-aware Eikonal relax (additive; default 0.0 == OFF ==
    # bit-identical). Down-weights the Eikonal residual near triple junctions (the margin crease).
    ap.add_argument("--eikonal-junction-relax", type=float, default=0.0,
                    help="THETA* STRETCH-1: down-weight the Eikonal |grad m|->1 residual near triple "
                    "junctions by factor (1 - relax*exp(-g3/tau)). 0.0 (default) = OFF = bit-identical. "
                    "Must be in [0, 1).")
    ap.add_argument("--eikonal-junction-tau", type=float, default=0.5,
                    help="THETA* STRETCH-1: top2-top3 SDF-gap scale for the junction relax weight. "
                    "Must be > 0.")
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
    ap.add_argument(
        "--muon-lr-final-frac", type=float, default=1.0,
        help="(GAP 1, default-off) COSINE-DECAY the Muon-group LR from --muon-lr down to "
        "--muon-lr * this fraction across the Muon-stage span (muon_start_epoch -> --epochs). Muon's "
        "Newton-Schulz fixes update MAGNITUDE so a flat LR cannot self-reduce the step near the minimum "
        "(river-valley Muon 2606.21514); the decay lets the finisher settle. 1.0 (default) = flat/"
        "unchanged = byte-identical; e.g. 0.1 decays to 10%% of --muon-lr by stage end. Only the Muon "
        "group decays (the AdamW fallback self-adapts via its second moment). The schedule is anchored on "
        "--muon-start-epoch so a RESUME into the finisher rebuilds the SAME schedule (bit-faithful). "
        "A/B-ready; no effect until the Muon stage. Must be in (0, 1].",
    )
    ap.add_argument(
        "--muon-warm-start-momentum", action=argparse.BooleanOptionalAction, default=False,
        help="(GAP 2, default-off) WARM-START the fresh Muon momentum buffer (state 'v') from the "
        "OUTGOING AdamW first-moment (state 'm') for the shared param paths at the switch, instead of "
        "cold zeros. Both are gradient EMAs and Newton-Schulz re-normalizes the update, so the "
        "transferred DIRECTION removes the cold-start 'wild unit-norm direction from one noisy gradient' "
        "boundary thrash / d_seg spike. Default OFF = cold zero start = byte-identical. Only a plain "
        "Adam/AdamW base is transferable; non-Adam bases fall back to cold. On a RESUME INTO the "
        "finisher this is N/A (the Muon momentum is restored from the checkpoint). A/B-ready.",
    )
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
    # =====================================================================================
    # #224 CONSOLIDATED WIRE-IN — the 6 LANDED components. ALL flags DEFAULT-OFF => the
    # default render+loss+init path is BYTE-IDENTICAL to the pre-#224 baseline (the
    # non-negotiable acceptance bar; proven by tools/wire_in_224_byte_identical_smoke.py).
    # Each flag routes to the REAL (tested) module function when ON (NO-FAKE). Nothing here
    # fires unless explicitly enabled. Docs: docs/aa_sdf_observation_render_wire_in_spec.md +
    # docs/analytic_lane_render_band_wire_in_spec.md + the in-module WIRE-IN SPECs.
    # -------- (1) AA-SDF observation-map render (aa_sdf_observation_render; MEASURED #1 rep lever) --
    ap.add_argument("--render-aa", choices=["none", "supersample", "ipe"], default="none",
                    help="#224/#220 AA observation-map render mode (default none = byte-identical "
                    "point-sample). supersample=render at ss*grid+box-down; ipe=mip-NeRF cone "
                    "attenuation of the curvelet basis (analytical, ~0-compute).")
    ap.add_argument("--aa-supersample", type=int, default=1,
                    help="#224 supersample factor ss for --render-aa supersample (ss=1 byte-identical).")
    ap.add_argument("--aa-ipe-footprint", type=float, default=1.0,
                    help="#224 footprint std scale for --render-aa ipe (1.0 = one-pixel box).")
    ap.add_argument("--aa-self-orient-fine-mode", type=str, default="refuse",
                    choices=["refuse", "batch", "full"],
                    help="#224 (Wave B) how --render-aa supersample + --self-orient sources the per-pair "
                    "fine-grid dir-feats (measured: fine-EDT ~49ms/pair @ ss=2; per-pair fine dir-feat "
                    "75.5MB). refuse (default) = fail-closed (memory/wall-clock tradeoff is the operator's "
                    "call, see the guard). batch = MEMORY-SAFE bounded on-demand cache (~batch*75MB, e.g. "
                    "0.6GB @ batch=8 vs 45GB all-600) but wall-clock-heavy (P fine-EDTs/epoch ~29s @ "
                    "n600, since every pair renders every epoch => a batch-bounded cache thrashes). full "
                    "= WALL-CLOCK-viable (fine dir-feats computed ONCE per --reorient-every, amortized) "
                    "but ~45GB @ ss=2 n600 (on top of the ~41GB base cache => ~86GB; needs an n600 "
                    "memory-fit validation the no-launch CONTAINMENT forbids this wave).")
    ap.add_argument("--aa-self-orient-fine-cache-cap", type=int, default=16,
                    help="#224 (Wave B) bounded per-pair fine dir-feat cache size for "
                    "--aa-self-orient-fine-mode batch (memory ~ cap*75MB @ ss=2).")
    # -------- (2) analytic-lane render-band (analytic_lane_render_band; FEED-dv #203/#213/#215) ------
    ap.add_argument("--lane-render-band", action=argparse.BooleanOptionalAction, default=False,
                    help="#224/FEED-dv: composite the analytic-lane render-band via compose_fn "
                    "(class-1 render-time authority). DEFAULT OFF => byte-identical.")
    ap.add_argument("--lane-band-softness", type=float, default=1.0,
                    help="#224 AA-SDF coverage ramp width (px) on the band lateral edge.")
    ap.add_argument("--lane-band-dash-forward-max-m", type=float, default=55.0,
                    help="#224/#215 SegNet-Nyquist: dash-gate ONLY where forward < this (m); continuous beyond.")
    ap.add_argument("--lane-band-uncertainty-source", type=str, default="witness",
                    choices=["witness", "gt", "none"],
                    help="#224 uncertainty margin source for the FP-killer gate (witness margin PROB; "
                    "gt margin LOGIT; none disables the gate).")
    ap.add_argument("--lane-band-tau", type=float, default=0.85,
                    help="#224 uncertainty threshold (witness margin PROB [0,1]; gt margin LOGIT ~[0,13]).")
    ap.add_argument("--lane-band-eps", type=float, default=0.35, help="#224 uncertainty ramp width.")
    ap.add_argument("--lane-band-weight", type=float, default=1.0, help="#224 band strength (curriculum ramp).")
    ap.add_argument("--lane-band-start-epoch", type=int, default=300, help="#224 engage the band at this epoch.")
    # -------- (3) warp-real-luma frame0 pose carrier (warp_real_luma_frame0; PoseGauge.WARP_REAL_LUMA) --
    ap.add_argument("--pose-carrier", action=argparse.BooleanOptionalAction, default=False,
                    help="#224: render frame0 THROUGH the SE(3) ground-homography warp of the REAL "
                    "keyframe luma (seg-free f0 -> real-luma pose carrier). Parity-dispatch render_fn "
                    "(even code=f0->carrier, odd=f1->witness). Requires --w-pose>0. DEFAULT OFF => "
                    "byte-identical (the witness's own f0 render).")
    ap.add_argument("--pose-carrier-source", type=str, default="real_keyframe",
                    choices=["real_keyframe", "generated"],
                    help="#205 pose-carrier frame0 SOURCE (Track B store-nothing-but-xi, 18927a1ae). "
                    "real_keyframe (default) = warp the STORED real keyframe luma (gt_f0; COUNTS the "
                    "keyframe in archive.zip). generated = STORE-NOTHING: warp the witness's OWN plain "
                    "frame0 INR render (up to camera-native) by the twist -> stores ONLY xi/H (~0 "
                    "marginal bytes; the render is FREE, rule-118). The dxi residual co-adapts to the "
                    "witness-render warp. Default real_keyframe => byte-identical (unchanged wiring).")
    ap.add_argument("--pose-carrier-residual-mode", type=str, default="table", choices=["table", "film"],
                    help="#224 pose-carrier residual parametrization: table (per-pair (P,6), byte-minimal) "
                    "or film (code-conditioned MLP). Default table.")
    ap.add_argument("--pose-carrier-residual-scale", type=float, default=1.0,
                    help="#224 pose-carrier learnable-residual scale (dxi = scale * residual).")
    ap.add_argument("--pose-carrier-s-t", type=float, default=None,
                    help="#224 pose-carrier ground-homography translation scale s_t for the stored twist "
                    "xi = xi_from_pose_calibration(gt_pose, s_t, s_r, pitch). None (default) => FIT s_t at "
                    "startup on --pose-carrier-fit-pairs via the frozen CPU-torch PoseNet d_pose grid "
                    "(self-calibrating, deterministic; mirrors tools/measure_warp_real_luma_frame0_dpose).")
    ap.add_argument("--pose-carrier-s-r", type=float, default=0.0,
                    help="#224 pose-carrier rotation scale s_r for the stored twist (default 0.0 = the "
                    "measured d_pose-optimal whole-ground calibration).")
    ap.add_argument("--pose-carrier-pitch", type=float, default=0.0,
                    help="#224 pose-carrier ground-plane pitch (rad) for the homography geom (default 0.0).")
    ap.add_argument("--pose-carrier-fit-pairs", type=int, default=24,
                    help="#224 # pairs for the startup s_t fit grid (only when --pose-carrier-s-t is None).")
    # -------- (4) persistence/topology loss (persistence_topology_loss; TopologyLossGauge) -----------
    ap.add_argument("--persistence-loss-weight", type=float, default=0.0,
                    help="#224/#218: weight of the soft-clDice + persistence-weighted island-recall "
                    "term on the SHARED realized-through-R seg forward (births the finest-scale "
                    "erasure-tail the CE drops). 0 (default) => branch skipped => byte-identical.")
    ap.add_argument("--persistence-recall-weight", type=float, default=1.0,
                    help="#224 w_recall inside the persistence class loss (clDice weight fixed 1.0).")
    ap.add_argument("--cldice-iters", type=int, default=5,
                    help="#224 soft-skeleton peeling iterations for the clDice connectivity term.")
    ap.add_argument("--persistence-warmup-epochs", type=int, default=0,
                    help="#224 linear warm-up (epochs) for the persistence weight (coarse->fine; "
                    "0=full weight immediately).")
    ap.add_argument("--persistence-classes", type=str, default="auto",
                    help="#224 target classes: 'auto' self-detects the thin/small erasure-tail classes "
                    "from the cached GT argmax (detect_persistence_tail_classes), or a comma list e.g. '1,3'.")
    # -------- (5) island seed/containment/amplification (island_protection; IslandProtectionGauge) ---
    ap.add_argument("--seed-islands", action=argparse.BooleanOptionalAction, default=False,
                    help="#224/#208: EARLY-SEED the finest-scale islands (self-detected lane+movable) "
                    "into the structured-init phi target (accelerant; ships 0 archive bytes). Requires "
                    "--structured-init. DEFAULT OFF => byte-identical.")
    ap.add_argument("--island-dilate-px", type=int, default=1, help="#224 annulus dilation of the island masks.")
    ap.add_argument("--seed-blend", type=float, default=1.0,
                    help="#224 island-seed blend (residual = blend*(gt_island_rgb - base) on the island).")
    ap.add_argument("--seed-lr", type=float, default=0.02,
                    help="#224 learning rate for the SEPARATE island-seed AdamW group (its own optimizer; "
                    "the seed is NOT in the witness EMA/blob/deploy).")
    ap.add_argument("--containment-mode", type=str, default="shield", choices=["freeze", "damp", "shield"],
                    help="#224 how the seeded island grad is protected from the bulk-CE wash "
                    "(shield=zero only the destructive same-sign component).")
    ap.add_argument("--containment-damp", type=float, default=0.1, help="#224 damp factor for --containment-mode damp.")
    ap.add_argument("--amplify-weight", type=float, default=0.0,
                    help="#224/#208: weight of the island-birth term (rides the SHARED LEVER-4 _signed "
                    "margin; NO 2nd saliency/SegNet forward). 0 (default) => skipped => byte-identical.")
    ap.add_argument("--amplify-form", type=str, default="hinge", choices=["hinge", "softplus"],
                    help="#224 island-birth penalty form.")
    ap.add_argument("--amplify-margin-target", type=float, default=1.0,
                    help="#224 the margin the island must WIN its pixels by.")
    ap.add_argument("--amplify-persist", type=str, default="inverse_thickness",
                    choices=["uniform", "inverse_thickness"],
                    help="#224 island birth-weight kind (inverse_thickness up-weights the thinnest tail).")
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

    # RESIDUAL-ONLY MODE fail-closed config guards (same NO-FAKE silent-no-op class). --residual-mode
    # without the bundle would be a silent no-op (no composition) = a FALSE 'residual mode does
    # nothing'. --structured-init / --lane-prior-phi1 / --freeze-decoder-fit-codes are the CONTRADICTORY
    # mechanism (they bake the bulk INTO the weights = the opposite of residual mode, which keeps the
    # bulk OUTSIDE the counted weights and composes it deterministically) -> fail LOUD rather than
    # silently ship a non-shrinking INR. The loss-weighting surgical levers (--lane-thin-* /
    # --margin-saliency-* / --hardness-*) ARE compatible (they weight the COMPOSED-render d_seg) and
    # are intentionally NOT forbidden.
    if getattr(args, "residual_mode", False):
        if not args.residual_target_npz:
            raise ValueError(
                "--residual-mode requires --residual-target-npz (the residual training bundle): "
                "without it the composition has no bulk to compose = a silent no-op = a FALSE "
                "'residual mode does nothing' verdict.")
        if args.structured_init or getattr(args, "lane_prior_phi1", False):
            raise ValueError(
                "--residual-mode is incompatible with --structured-init / --lane-prior-phi1: those "
                "BAKE the bulk/static-core INTO the INR weights (a train-time init that ships the "
                "bulk inside the counted weights = NO rate shrink), which is the EXACT mechanism "
                "residual mode replaces (the bulk is GENERATED deterministically OUTSIDE the weights "
                "and COMPOSED before R). Run residual mode WITHOUT --structured-init.")
        if args.freeze_decoder_fit_codes:
            raise ValueError(
                "--residual-mode is incompatible with --freeze-decoder-fit-codes: residual mode "
                "trains the INR's decoder to flip the residual annulus; a frozen decoder cannot "
                "(only the per-pair code would move) = a silent no-op = a FALSE 'residual mode does "
                "nothing' verdict.")
    elif args.residual_target_npz:
        raise ValueError(
            "--residual-target-npz was given but --residual-mode is OFF: the bundle would be "
            "loaded-and-ignored = a silent no-op = a FALSE 'residual bundle does nothing'. Pass "
            "--residual-mode to engage the composition, or drop --residual-target-npz.")

    # (THETA* TIER-2 MUST-1) tau-anneal-shape fail-closed guards (pure; fail LOUD before any GPU spend).
    if args.tau_anneal_shape == "geometric" and not (args.softmax_temp_start > 0.0 and args.softmax_temp_end > 0.0):
        raise ValueError(
            f"--tau-anneal-shape geometric requires --softmax-temp-start ({args.softmax_temp_start}) > 0 "
            f"AND --softmax-temp-end ({args.softmax_temp_end}) > 0: the log-spaced curve "
            "tau=start*(end/start)**prog is undefined / non-positive otherwise.")
    if not (0.0 < args.tau_hold_frac <= 1.0):
        raise ValueError(
            f"--tau-hold-frac ({args.tau_hold_frac}) must be in (0, 1] (the fraction of the anneal "
            "window at which cosine_hold reaches the floor; 1.0 = no hold = bit-identical cosine).")

    # (THETA* TIER-2 MUST-2) nuclear-norm penalty fail-closed guards.
    if args.code_nuclear_weight < 0.0:
        raise ValueError(f"--code-nuclear-weight ({args.code_nuclear_weight}) must be >= 0 (0 = OFF).")
    if args.code_nuclear_weight > 0.0:
        if args.code_nuclear_eps <= 0.0:
            raise ValueError(
                f"--code-nuclear-eps ({args.code_nuclear_eps}) must be > 0 (relative smoothing floor "
                "that keeps Newton-Schulz stable on rank-deficient codes).")
        if args.code_nuclear_ns_iters < 1:
            raise ValueError(
                f"--code-nuclear-ns-iters ({args.code_nuclear_ns_iters}) must be >= 1.")

    # (THETA* TIER-2 MUST-3) SWA / wider-finisher EMA fail-closed guards (same NO-FAKE silent-no-op
    # class as the lane/muon validators): a finisher decay set with no resolvable start would NEVER
    # engage = a FALSE 'wider EMA does nothing' verdict.
    if args.ema_decay_finisher is not None:
        if not (0.0 < args.ema_decay_finisher < 1.0):
            raise ValueError(
                f"--ema-decay-finisher ({args.ema_decay_finisher}) must be in (0, 1).")
        _ema_fin_start = (args.ema_decay_finisher_start_epoch
                          if args.ema_decay_finisher_start_epoch is not None else args.muon_start_epoch)
        if _ema_fin_start is None:
            raise ValueError(
                "--ema-decay-finisher requires a start epoch: set --ema-decay-finisher-start-epoch "
                "(or --muon-start-epoch, which it falls back to). Without one the wider EMA would "
                "NEVER engage = a silent no-op = a FALSE 'wider EMA does nothing' verdict.")
        if not (1 <= _ema_fin_start <= args.epochs):
            raise ValueError(
                f"--ema-decay-finisher start epoch ({_ema_fin_start}) must be in [1, --epochs "
                f"({args.epochs})]: outside the budget the wider EMA would never engage = a silent "
                "no-op.")
    elif args.ema_decay_finisher_start_epoch is not None:
        raise ValueError(
            "--ema-decay-finisher-start-epoch set without --ema-decay-finisher: the start epoch has "
            "no effect = a silent no-op. Set --ema-decay-finisher too, or drop the start flag.")

    # (THETA* TIER-2 STRETCH-1) junction-aware Eikonal relax fail-closed guards.
    if not (0.0 <= args.eikonal_junction_relax < 1.0):
        raise ValueError(
            f"--eikonal-junction-relax ({args.eikonal_junction_relax}) must be in [0, 1) (0 = OFF; "
            "the weight 1-relax*exp(-g3/tau) must stay positive).")
    if args.eikonal_junction_relax > 0.0 and args.eikonal_junction_tau <= 0.0:
        raise ValueError(
            f"--eikonal-junction-tau ({args.eikonal_junction_tau}) must be > 0 (the top2-top3 SDF-gap "
            "scale in exp(-g3/tau)).")

    result = run_train(args)
    print("\n=== LEVEL-SET WITNESS RESULT (realized through R) ===")
    print(json.dumps({"front_end": result["front_end"], "history": result["history"],
                      "axis": result["axis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
