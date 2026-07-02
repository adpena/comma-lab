#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LEVEL-SET WITNESS byte-close -> full-output inflate -> realized d_seg/d_pose parity -> staged exact eval.

Sister of ``tools/witness_byte_close_and_eval.py`` (the RGB witness path), keyed on the
LEVEL-SET checkpoint (``levelset_witness_ema_mlx.npz`` produced by
``experiments/train_levelset_witness_realized_through_R_mlx.py``). The RGB tool is HARD-KEYED to
``params["out.weight"]`` (a single RGB head); the level-set witness has NO ``out.weight`` -- it
has ``out_sdf`` (K SDF fields) + ``out_tex`` (pose-carrying RGB texture) + ``palette`` (K,3) +
``code`` (per-(pair,frame) FiLM). So the RGB tool KeyErrors on a level-set npz; THIS tool is the
missing level-set exact-eval path -- THE row-enabler (per the FEED-df recursive review memo
``.omx/research/yousfi_levers_optimal_form_review_20260627T063335Z.md``: until it exists, ALL the
level-set d_seg descent is advisory-only and the pointer CANNOT move).

ONE command turns a trained level-set checkpoint into a self-contained contest packet and proves
it end-to-end ON CPU at $0:

  (a) archive.zip       -- int8+brotli LEARNED-payload blob (the MEASURED rate term st_size).
                           Matches ``lever_b_levelset_generator.quantize_levelset_blob`` accounting
                           (one brotli stream for the base weights + one for ``code``).
  (b) inflate.py        -- numpy-only (numpy + brotli + torch-for-R [+ scipy only when the
                           checkpoint used self-orient]) FULL-OUTPUT decoder: emits the complete
                           (2*n_pairs, 874, 1164, 3) uint8 ``.raw`` (frame0 AND frame1 per pair,
                           camera-res, no header) -- the evaluator's expected layout. Runs the ONE
                           CODEPATH (``levelset_rgb_forward_numpy``, inlined) + the torch R
                           (bicubic up to camera -> round -> uint8), byte-identical to the trainer
                           verdict (``_torch_R_to_camera_uint8``).
  (c) realized d_seg/d_pose on the INFLATED FRAMES (read back from the .raw) via the FROZEN
      CPU-torch SegNet/PoseNet vs GT -- NOT a field-level proxy, NOT a generator byte-repro.
  (d) the staged contest-CPU exact-eval command (NOT run here).

RULE 118 (FREE vs COUNTED -- the rate game):
  * FREE (regenerated in inflate.py from cfg, NOT in archive.zip): the curvelet/shearlet
    directional bank (a deterministic parametric polar frequency grid from the 5 scalars
    n_scales/n_orient0/f0/base/n_iso + max_freq) AND, when the checkpoint used self-orientation,
    the directional feats (a fixed-point of the DECODER'S OWN argmax -- GT-free, reconstructible).
    These are generic ALGORITHM, not video-derived -> 0 bytes (CLAUDE.md "inflate.py is a FREE
    interpreter").
  * COUNTED (int8+brotli in archive.zip): the LEARNED video-derived payload -- in_proj / film /
    hidden.* / out_sdf / out_tex / palette weights + the per-frame ``code`` table.

SELF-ORIENT (the mod-32 target uses it): when ``in_proj.weight`` has MORE input columns than the
curvelet bank produces, the checkpoint was trained with ``--self-orient`` (directional feats
concatenated to the curvelet feats). The trainer reconstructs those feats at deploy via a
FIXED-POINT on the decoder's own argmax (cos 0.89-0.91 vs GT, the byte-closeable -48% directional
lever). inflate.py reproduces the same fixed point -> GT-free, 0 extra bytes. PARITY-GAP CAVEAT
(loud, NO-FAKE): the deploy fixed point converges on the FINAL weights, whereas the trainer's
reported implied_S used the dir feats accumulated along the training TRAJECTORY (reorient every N
epochs). They are close (the fixed point is the design) but NOT bit-identical -> the realized
d_seg on the inflated frames is the TRUTH; any gap vs the trainer's number is itself a finding.
ALSO: the trainer does NOT persist the self-orient params (freq_across/freq_along/tau/iters) into
the npz (a trainer gap, flagged) -> they default to the trainer defaults here and are overridable
via ``--so-*`` (generic hyperparameters, rule-118 clean, a handful of manifest bytes).

POSE (the verdict this tool RESOLVES): the scorer computes
``d_pose = MSE(PoseNet(generated_pair)[:6], PoseNet(original_pair)[:6])`` ON THE FRAMES. A stored
6-scalar pose sidecar is just BYTES the scorer never reads -- it does NOT change the rendered
frames, so it does NOT lower realized d_pose. A POSE-BLIND render (the w_pose=0 mod-32 config)
stays d_pose ~O(100) regardless of any sidecar. Therefore the level-set ROW REQUIRES a
POSE-TRAINED render (``--w-pose > 0`` supervising the texture/code to hit the 6 PoseNet targets in
the SegNet-null space) -- exactly like the RGB witness carries pose in its per-(pair,frame) codes.
``--fold-pose-sidecar`` is provided for parity with the RGB tool but is OFF by default and, when
on, LOUDLY records that it adds COUNTED bytes WITHOUT lowering realized d_pose (the scorer reads
frames, not the sidecar).

AUTHORITY: ``[macOS-CPU advisory] NON-PROMOTABLE``. CPU only (no MPS, no CUDA, no paid eval). The
realized d_seg/d_pose here is the frozen CPU-torch mirror of evaluate.py over the measured pair
subset. The reported S is advisory until the SAME packet runs through ``upstream/evaluate.py`` on
contest-compliant Linux x86_64 CPU (the staged command). NO score/frontier/promotion claim;
pointer UNMOVED unless a real byte-closed sub-frontier exact-eval row lands.

Usage:
    .venv/bin/python tools/levelset_byte_close_and_eval.py \\
        --ckpt-dir experiments/results/<run> \\
        [--max-pairs 8] [--gt-cache ...gt_n6.npz] [--keep-packet] [--out reports/...json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_witness_realized_through_R_mlx as twr  # noqa: E402  (frozen CPU-torch verdict + GT + R)

from tac.decode_memory_tier import (  # noqa: E402  (#224 Wave D decode memory-tier surface)
    DECODE_MEMORY_TIERS,
    DEFAULT_TIER_NAME,
    require_contest_tier,
    resolve_eval_device,
    resolve_tier,
    tier_inflate_env,
)

from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    _int8_symmetric,
    curvelet_directional_B,
    levelset_band_forward_numpy,
    levelset_rgb_forward_numpy,
    quantize_levelset_blob,
)
from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402  (#224 Wave E canonical band)
    DEFAULT_DASH_FORWARD_MAX_M,
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    composite_band_on_render,
    deserialize_lane_band,
    deserialize_lane_band_any,  # Wave-F: LBND1/LBND2 magic dispatch
    lane_band_rd_rate_report,   # Wave-F: measured per-lever byte accounting
    rasterize_lane_coverage_range_dependent,
    render_config_from_header,
    serialize_lane_band,
    serialize_lane_band_any,    # Wave-F: format-preserving re-serialize (capped inflate)
    serialize_lane_band_rd,     # Wave-F: LBND2 optimal RD serializer
    witness_uncertainty_mask,
)
from tac.local_acceleration import torch_levelset_inflate as _tli  # noqa: E402  (canonical FREE-table regen)

# canonical FREE-table regen fns (rule-118 curvelet bank + self-orient dir feats) — the bit-exact
# oracle reference reuses these so the gate compares against the SAME free tables the inflate uses.
_canon_coords_grid = _tli.coords_grid
_canon_curvelet_B = _tli.curvelet_B
_canon_curvelet_feats = _tli.curvelet_feats
_canon_dir_feats = _tli.dir_feats

CAMERA_H, CAMERA_W = 874, 1164
RATE_DENOM = 37_545_489.0
_MAGIC = b"LVLS1\x00"  # level-set softmax-of-SDF carrier v1
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _advisory_axis_label() -> str:
    """Device-truthful advisory authority. macOS CPU-torch is NOT 1:1 with the contest Linux
    x86_64 CPU runner -> [macOS-CPU advisory]; only a real Linux x86_64 host earns [contest-CPU
    advisory]. Always NON-PROMOTABLE here (no MPS/CUDA/paid axis)."""
    import platform

    if platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"):
        return "[contest-CPU advisory] NON-PROMOTABLE"
    return "[macOS-CPU advisory] NON-PROMOTABLE"


_AUTHORITY = _advisory_axis_label()


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# checkpoint loading -- separate LEARNED params from the __cfg/__bank/__render scalars.
# ---------------------------------------------------------------------------
def _load_levelset_ckpt(
    ckpt_dir: Path, npz_name: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load (params, cfg) from a level-set run dir. NO-FAKE: missing files raise.

    Params = every npz key NOT prefixed ``__`` (the learned weights + ``code``). cfg = the
    ``__cfg_*`` / ``__bank_*`` / ``__render_hw`` scalars (parsed; defaults + tensor-shape inference
    fill any that an older save block omitted, with a loud warning)."""
    candidates = [npz_name] if npz_name else ["levelset_witness_ema_mlx.npz", "levelset_witness_live_mlx.npz"]
    npz = None
    for c in candidates:
        p = ckpt_dir / c
        if p.exists():
            npz = p
            break
    if npz is None:
        raise FileNotFoundError(
            f"level-set checkpoint missing in {ckpt_dir} (looked for {candidates}); "
            "refusing to fabricate (NO-FAKE). If the run is still training, its npz is saved only "
            "at the END of the loop -- wait for completion or point at a finished run dir."
        )
    z = np.load(npz, allow_pickle=False)
    params: dict[str, np.ndarray] = {}
    raw_cfg: dict[str, Any] = {}
    for k in z.files:
        if k.startswith("__"):
            a = z[k]
            raw_cfg[k] = a.item() if a.size == 1 else a.tolist()
        else:
            params[k] = np.asarray(z[k], dtype=np.float32)

    for req in ("code", "in_proj.weight", "out_sdf.weight", "out_tex.weight", "palette"):
        if req not in params:
            raise ValueError(f"level-set npz {npz} lacks required param {req!r} (NO-FAKE).")

    cfg: dict[str, Any] = {"npz_name": npz.name}
    cfg["n_classes"] = int(params["out_sdf.weight"].shape[0])
    cfg["hidden_dim"] = int(raw_cfg.get("__cfg_hidden_dim", params["in_proj.weight"].shape[0]))
    cfg["n_hidden"] = int(raw_cfg.get(
        "__cfg_n_hidden", sum(1 for k in params if k.startswith("hidden.") and k.endswith(".weight"))))
    cfg["mod_dim"] = int(params["code"].shape[1])
    cfg["n_pairs"] = int(params["code"].shape[0] // 2)
    cfg["in_feat"] = int(params["in_proj.weight"].shape[1])
    cfg["activation"] = str(raw_cfg.get("__cfg_activation", "hosc"))
    cfg["softmax_temp"] = float(raw_cfg.get("__cfg_softmax_temp", 0.05))
    cfg["chroma"] = bool(int(raw_cfg.get("__cfg_chroma", 1)))
    cfg["wire_w0"] = float(raw_cfg.get("__cfg_wire_w0", 20.0))
    cfg["wire_s0"] = float(raw_cfg.get("__cfg_wire_s0", 10.0))
    cfg["hosc_beta"] = float(raw_cfg.get("__cfg_hosc_beta", 4.0))
    cfg["hosc_omega"] = float(raw_cfg.get("__cfg_hosc_omega", 1.0))
    cfg["bank_n_scales"] = int(raw_cfg.get("__bank_n_scales", 4))
    cfg["bank_n_orient0"] = int(raw_cfg.get("__bank_n_orient0", 6))
    cfg["bank_f0"] = float(raw_cfg.get("__bank_f0", 2.0))
    cfg["bank_base"] = float(raw_cfg.get("__bank_base", 2.0))
    cfg["bank_n_iso"] = int(raw_cfg.get("__bank_n_iso", 4))
    mbf = raw_cfg.get("__cfg_max_bank_freq")
    cfg["max_bank_freq"] = None if (mbf is None or float(mbf) < 0) else float(mbf)
    rh = raw_cfg.get("__render_hw")
    if rh is None:
        cfg["render_h"], cfg["render_w"] = 384, 512
        print("[WARN] npz lacks __render_hw -> defaulting render 384x512 (may NOT match the "
              "trained render res; pass --render-h/--render-w if known).", flush=True)
    else:
        cfg["render_h"], cfg["render_w"] = int(rh[0]), int(rh[1])
    cfg["lane_edge_weight"] = float(raw_cfg.get("__cfg_lane_edge_weight", 0.0))  # provenance only
    missing = [k for k in ("__cfg_activation", "__bank_n_scales", "__render_hw") if k not in raw_cfg]
    if missing:
        print(f"[WARN] npz omitted cfg keys {missing} (older save block) -> using inferred/defaults; "
              "provenance-incomplete (the deploy render may diverge if a default is wrong).", flush=True)
    return params, cfg


def _bank_cfg(cfg: dict[str, Any]) -> CurveletBankConfig:
    return CurveletBankConfig(
        n_scales=cfg["bank_n_scales"], n_orient0=cfg["bank_n_orient0"],
        f0=cfg["bank_f0"], base=cfg["bank_base"], n_iso=cfg["bank_n_iso"],
    )


def _curvelet_feat_width(cfg: dict[str, Any]) -> int:
    """2 * (number of curvelet bank columns after the optional max_freq cap)."""
    B = curvelet_directional_B(_bank_cfg(cfg), max_freq=cfg["max_bank_freq"])
    return 2 * int(B.shape[1])


def detect_self_orient(cfg: dict[str, Any], so_overrides: dict[str, Any]) -> dict[str, Any]:
    """dir_w = in_feat - curvelet_feat_width. >0 => self-orient was used; n_dir_freqs = dir_w//4.
    freq_across/freq_along/tau/iters are NOT persisted by the trainer -> sourced from CLI overrides
    (defaults = trainer defaults). Returns a dict the manifest + inflate consume."""
    curv_w = _curvelet_feat_width(cfg)
    dir_w = int(cfg["in_feat"]) - curv_w
    if dir_w < 0:
        raise ValueError(
            f"in_feat ({cfg['in_feat']}) < curvelet_feat_width ({curv_w}): the bank cfg in the npz "
            "does NOT reproduce the trained in_proj input width. The __bank_* scalars are wrong/"
            "missing (NO-FAKE: cannot regenerate the free bank that matches the weights). Pass the "
            "correct --bank-* overrides or use a checkpoint with full __bank_* provenance.")
    if dir_w == 0:
        return {"self_orient": False, "curvelet_feat_width": curv_w, "dir_w": 0}
    if dir_w % 4 != 0:
        raise ValueError(
            f"self-orient dir_w ({dir_w}) not divisible by 4 (= 4*n_dir_freqs). The bank/in_feat "
            "mismatch means the free bank does not reproduce the weights -- refuse (NO-FAKE).")
    return {
        "self_orient": True,
        "curvelet_feat_width": curv_w,
        "dir_w": dir_w,
        "n_dir_freqs": dir_w // 4,
        "freq_across": float(so_overrides["freq_across"]),
        "freq_along": float(so_overrides["freq_along"]),
        "tau": float(so_overrides["tau"]),
        "iters": int(so_overrides["iters"]),
    }


# ---------------------------------------------------------------------------
# byte-close: int8 + brotli (matches quantize_levelset_blob accounting; bank is FREE).
# ---------------------------------------------------------------------------
def build_levelset_blob(
    params: dict[str, np.ndarray], cfg: dict[str, Any], so: dict[str, Any], pose_sidecar: bytes | None,
    lane_band_bytes: bytes | None = None, lane_manifest: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Layout: magic | u32 manifest_len | manifest_json | u32 base_brotli_len | base_brotli |
            u32 code_brotli_len | code_brotli | u32 pose_len | pose_sidecar
            [| u32 lane_len | lane_band(optional, #224 Wave E COUNTED lane manifold coords)].
    base = int8(all params except code) concat -> ONE brotli stream (== quantize_levelset_blob);
    code = int8(code) -> a SECOND brotli stream. The curvelet bank + the lane COVERAGE raster are
    NOT stored (free, rule 118); only the lane MANIFOLD COORDS (``lane_band_bytes``) are counted.
    Absent ``lane_band_bytes`` the manifest + blob are BYTE-IDENTICAL to the pre-Wave-E grammar.
    """
    import brotli

    base_order = [k for k in params if k != "code"]
    base_chunks: list[bytes] = []
    shapes: dict[str, list[int]] = {}
    scales: dict[str, float] = {}
    for name in base_order:
        a = np.asarray(params[name], dtype=np.float32)
        q, scale = _int8_symmetric(a)
        base_chunks.append(q.astype(np.int8).tobytes())
        shapes[name] = list(a.shape)
        scales[name] = float(scale)
    base_brotli = brotli.compress(b"".join(base_chunks), quality=11)

    code = np.asarray(params["code"], dtype=np.float32)
    qc, code_scale = _int8_symmetric(code)
    code_brotli = brotli.compress(qc.astype(np.int8).tobytes(), quality=11)

    manifest = {
        "format_version": 1,
        "n_pairs": int(cfg["n_pairs"]),
        "n_classes": int(cfg["n_classes"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "n_hidden": int(cfg["n_hidden"]),
        "mod_dim": int(cfg["mod_dim"]),
        "activation": str(cfg["activation"]),
        "softmax_temp": float(cfg["softmax_temp"]),
        "chroma": bool(cfg["chroma"]),
        "wire_w0": float(cfg["wire_w0"]), "wire_s0": float(cfg["wire_s0"]),
        "hosc_beta": float(cfg["hosc_beta"]), "hosc_omega": float(cfg["hosc_omega"]),
        "bank_n_scales": int(cfg["bank_n_scales"]), "bank_n_orient0": int(cfg["bank_n_orient0"]),
        "bank_f0": float(cfg["bank_f0"]), "bank_base": float(cfg["bank_base"]),
        "bank_n_iso": int(cfg["bank_n_iso"]),
        "max_bank_freq": (None if cfg["max_bank_freq"] is None else float(cfg["max_bank_freq"])),
        "render_h": int(cfg["render_h"]), "render_w": int(cfg["render_w"]),
        "camera_h": CAMERA_H, "camera_w": CAMERA_W,
        "base_param_order": base_order,
        "base_shapes": shapes,
        "base_scales": scales,
        "code_shape": list(code.shape),
        "code_scale": float(code_scale),
        "self_orient": bool(so["self_orient"]),
        "n_dir_freqs": int(so.get("n_dir_freqs", 0)),
        "so_freq_across": float(so.get("freq_across", 0.0)),
        "so_freq_along": float(so.get("freq_along", 0.0)),
        "so_tau": float(so.get("tau", 4.0)),
        "so_iters": int(so.get("iters", 0)),
        "has_pose_sidecar": bool(pose_sidecar is not None),
    }
    # #224 Wave E: the analytic-lane RENDER-BAND cfg (ONLY when active -> default-off byte-identical).
    # The lane MANIFOLD COORDS ride the 5th block (counted); this cfg (scalars) rides the manifest so
    # inflate reproduces the coverage raster + composite decode-consistently (rule 118 FREE rasterizer).
    if lane_band_bytes is not None and lane_manifest is not None:
        manifest["lane_render_band"] = lane_manifest
    mj = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    out = _io_pack(mj, base_brotli, code_brotli, pose_sidecar, lane_band_bytes)
    # cross-check our accounting against the canonical quantize_levelset_blob (same int8 grammar).
    canon = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in params.items()})
    breakdown = {
        "n_params": int(sum(int(np.prod(s)) for s in shapes.values())) + int(np.prod(code.shape)),
        "manifest_bytes": len(mj),
        "base_int8_brotli_bytes": len(base_brotli),
        "code_int8_brotli_bytes": len(code_brotli),
        "pose_sidecar_bytes": (len(pose_sidecar) if pose_sidecar else 0),
        "lane_band_counted_bytes": (len(lane_band_bytes) if lane_band_bytes else 0),
        "magic_and_prefixes_bytes": len(_MAGIC) + 16 + (4 if lane_band_bytes is not None else 0),
        "total_0bin_bytes": len(out),
        "canonical_quantize_blob_bytes": int(canon["total_quantized_blob_bytes"]),
        "accounting_matches_canonical": bool(
            len(base_brotli) == canon["base_int8_brotli_bytes"]
            and len(code_brotli) == canon["code_int8_brotli_bytes"]),
    }
    return out, breakdown


def _io_pack(
    manifest: bytes, base: bytes, code: bytes, pose: bytes | None,
    lane_band: bytes | None = None,
) -> bytes:
    """Pack the LVLS1 blob. The 5th ``lane_band`` block (#224 Wave E) is OPTIONAL and only
    appended when non-None -> absent it, the output is BYTE-IDENTICAL to the pre-Wave-E 4-block
    grammar (the default-off guarantee). ``lane_band`` = brotli(serialize_lane_band(...)), the
    COUNTED video-derived lane manifold coords (rule 118)."""

    buf = bytearray()
    buf += _MAGIC
    for chunk in (manifest, base, code, (pose or b"")):
        buf += struct.pack("<I", len(chunk))
        buf += chunk
    if lane_band is not None:
        buf += struct.pack("<I", len(lane_band))
        buf += lane_band
    return bytes(buf)


# ---------------------------------------------------------------------------
# the self-contained inflate.py (numpy fwd + torch R [+ scipy iff self-orient]).
# ---------------------------------------------------------------------------
# REVIEW-GATE: inflate.py exceeds the 100-LOC default budget (~250 LOC) under an explicit <=260-LOC
# waiver: it inlines TWO rule-118 FREE levers the archive-counted statistics require -- (1) the
# SELF-ORIENT fixed-point (curvelet bank regen + decoder-own-argmax tangent + directional feats), the
# byte-closeable directional lever the mod-32 target uses; (2) the #224 Wave E analytic-lane
# RENDER-BAND decode reproduction (_lane_parse + _lane_coverage AA-SDF rasterizer + _lane_composite),
# which EXPANDS the counted per-pair lane manifold coords into the (H,W) coverage + composites the
# band over the render FREE (0 archive bytes). Both are op-for-op mirrors of the canonical numpy-fp32
# authority (levelset_rgb_forward_numpy / levelset_band_forward_numpy / rasterize_lane_coverage_range_
# dependent / composite_band_on_render) and are BIT-EXACT-gate proven vs it.
# FEED-eg (2026-06-27): n600 inflate is float64-activation-bound (~50-60 min on a 4-core CPU for
# 600 pairs x 6 forwards x 5 tanh(sin) layers @ 384x512 -- OVER the 30-min budget). The forward is
# split into _in_proj_h0 (feats-only) + _outputs_from_h0 (code-dependent, want_rgb flag) to enable
# THREE BIT-IDENTICAL (sha256-proven) decode speedups, archive bytes UNCHANGED (inflate code = FREE,
# rule 118): (1) self-orient EARLY-STOP at the argmax fixed point (the big lever on a CONVERGED
# decoder -- up to ~2x); (2) self-orient forwards skip the rgb head (argmax(phi) needs no rgb);
# (3) the in_proj h0 is shared by a pair's 2 final frames. Guaranteed ~3% (skip-rgb+share-h0);
# early-stop is data-dependent (measure on the first real ckpt). Even so, contest-LEGALITY needs a
# trainer-side config cut (so_iters/render-res/n_hidden), which CHANGES the witness -> re-measure.
_INFLATE_PY = r'''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# LEVEL-SET witness inflate -- numpy fwd + torch R, self-contained. Emits the FULL
# (2*n_pairs, camera_h, camera_w, 3) uint8 .raw (frame0+frame1 per pair). The curvelet bank +
# (when used) the self-orient directional feats are REGENERATED for FREE (rule 118); only the
# learned int8+brotli payload comes from the archive.
# FEED-eg bit-identical decode speedups (inflate.py code only -> FREE, archive bytes UNCHANGED):
#   (1) self-orient EARLY-STOP at the argmax fixed point (consecutive-equal argmax => dirf frozen
#       => remaining iters are no-ops; bit-identical, the big lever on a converged decoder);
#   (2) self-orient forwards compute ONLY phi (argmax needs no rgb head -- out_tex/palette/softmax/
#       sigmoid skipped, they do not feed phi);
#   (3) the in_proj activation h0 is computed ONCE per pair and shared by the pair's 2 final frames
#       (identical feats => identical h0). All three preserve the EXACT float64/float32 op order of
#       levelset_rgb_forward_numpy -> bit-identical .raw (proven by sha256 parity).
# FEED-eh PARALLEL decode speedup (inflate.py code only -> FREE, archive bytes UNCHANGED, BIT-IDENTICAL):
#   the n_pairs are INDEPENDENT -> render them across a process Pool (1-thread BLAS per worker so N
#   workers scale cleanly instead of oversubscribing the already-threaded GEMM), each worker writing its
#   2 frames to DISJOINT offsets of the preallocated .raw. Proven 10.8x on M5 Max (15w),
#   max_abs_uint8_diff=0 vs the serial output. Contest: Linux fork (workers inherit setup); keeps the
#   inflate.py cost well inside the 30-min upstream/evaluate.py budget (inflate + scoring).
import os
for _tv in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")  # 1-thread BLAS/worker (set BEFORE numpy import); user-overridable
import sys, json, struct, multiprocessing as mp
import numpy as np
import brotli
import torch

MAGIC = b"LVLS1\x00"


def _read_blob(path):
    raw = open(path, "rb").read()
    assert raw[:len(MAGIC)] == MAGIC, "bad level-set magic"
    off = len(MAGIC); out = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        out.append(raw[off:off + n]); off += n
    lane_b = None  # #224 Wave E: optional 5th COUNTED lane-band block (absent -> pre-Wave-E grammar)
    if off < len(raw):
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        lane_b = raw[off:off + n]; off += n
    return json.loads(out[0].decode("utf-8")), out[1], out[2], out[3], lane_b


def _dequant(blob, order, shapes, scales):
    out, off = {}, 0
    flat = np.frombuffer(blob, dtype=np.int8)
    for name in order:
        shp = tuple(shapes[name]); n = int(np.prod(shp))
        out[name] = (flat[off:off + n].astype(np.float32) * float(scales[name])).reshape(shp); off += n
    return out


def _coords(h, w):
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32); xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def _curvelet_B(ns, no0, f0, base, niso, max_freq):
    cols = []
    for j in range(int(ns)):
        f_j = float(f0) * (float(base) ** j); l_j = int(no0) * (2 ** (j // 2))
        for l in range(l_j):
            th = np.pi * l / l_j; cols.append([f_j * np.cos(th), f_j * np.sin(th)])
    for i in range(int(niso)):
        th = np.pi * i / max(int(niso), 1); fl = float(f0) * 0.5; cols.append([fl * np.cos(th), fl * np.sin(th)])
    B = np.asarray(cols, np.float32).T  # (2, n)
    if max_freq is not None:
        nrm = np.sqrt((B.astype(np.float64) ** 2).sum(0)); keep = nrm <= float(max_freq) + 1e-6
        if not keep.any(): keep = nrm <= float(nrm.min()) + 1e-6
        B = B[:, keep]
    return B


def _curvelet_feats(coords, B):
    proj = (2.0 * np.pi) * (np.asarray(coords, np.float64) @ np.asarray(B, np.float64))
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


def _act(u, kind, w0, s0, beta, omega):
    u = np.asarray(u, np.float64)
    if kind == "wire": return (np.cos(w0 * u) * np.exp(-((s0 * u) ** 2))).astype(np.float32)
    if kind == "hosc": return np.tanh(beta * np.sin(omega * u)).astype(np.float32)
    return np.maximum(u, 0.0).astype(np.float32)


def _in_proj_h0(P, feats, m):
    # h0 = act(feats @ in_proj.weight.T + b) -- depends ONLY on feats (NOT code) -> shared by a
    # pair's two final frames (identical feats). Returns float32 (== _act), exactly as the monolithic
    # forward's first layer. P is the float64 param dict (converted once in main).
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _act(np.asarray(feats, np.float64) @ P["in_proj.weight"].T + P["in_proj.bias"], *kw)


def _outputs_from_h0(P, h0, code_row, m, want_rgb, want_lane=False):
    # op-for-op mirror of levelset_rgb_forward_numpy AFTER in_proj. h0 = the float32 in_proj act.
    # want_rgb=False -> return (phi, None) skipping the rgb head (out_tex/palette/softmax/sigmoid do
    # NOT feed phi, so argmax(phi) is identical) -- used by the self-orient fixed point.
    # want_lane=True (#224 Wave E) -> ALSO return the witness's OWN lane color sigmoid(palette[lane]+
    # tex)*255 + the softmax decision margin (top1-top2) for the decode-consistent render-band; mirror
    # of tac.boundary_math.lever_b_levelset_generator.levelset_band_forward_numpy. want_lane=False (all
    # non-band callers) -> BYTE-IDENTICAL 2-tuple return as before (default-off guarantee).
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    cr = np.asarray(code_row, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        film = (cr @ P["film.weight"].T + P["film.bias"]).reshape(m["n_hidden"], 2, m["hidden_dim"])
        # LEVER-A (FiLM-rank-fix) byte-close forward: AUTO-DETECT the OPTIONAL per-layer residual FiLM
        # (film_pl.*) + additive code-concat (concat_pl.*) routes from the loaded param keys so the
        # byte-close render reflects the ACTUAL trained witness (NO-FAKE). ABSENT keys (default-off
        # witness) => both branches skipped => BYTE-IDENTICAL to the pre-LEVER-A byte-close forward.
        _has_film_pl = any(str(k).startswith("film_pl.") for k in P)
        _has_concat = any(str(k).startswith("concat_pl.") for k in P)
        h = h0
        for li in range(m["n_hidden"]):
            scale = 1.0 + film[li, 0]
            shift = film[li, 1]
            if _has_film_pl:
                pl = (cr @ P["film_pl.%d.weight" % li].T + P["film_pl.%d.bias" % li]).reshape(2, m["hidden_dim"])
                scale = scale + pl[0]
                shift = shift + pl[1]
            pre = (h @ P["hidden.%d.weight" % li].T + P["hidden.%d.bias" % li]) * scale + shift
            if _has_concat:
                pre = pre + (cr @ P["concat_pl.%d.weight" % li].T + P["concat_pl.%d.bias" % li])
            h = _act(pre, *kw)
        phi = h @ P["out_sdf.weight"].T + P["out_sdf.bias"]
        if not want_rgb:
            return phi.astype(np.float32), None
        tex = h @ P["out_tex.weight"].T + P["out_tex.bias"]
        z = phi / float(m["softmax_temp"]); z = z - z.max(-1, keepdims=True)
        soft = np.exp(z); soft = soft / soft.sum(-1, keepdims=True)
        rgb = (1.0 / (1.0 + np.exp(-(soft @ P["palette"] + tex)))) * 255.0
        lane_rgb = margin = None
        if want_lane:
            lc = int(m.get("lane_render_band", {}).get("lane_cls", 1))
            lane_rgb = (1.0 / (1.0 + np.exp(-(P["palette"][lc][None, :] + tex)))) * 255.0
            ss = np.sort(soft, axis=-1); margin = (ss[:, -1] - ss[:, -2])
        if not m["chroma"]:
            luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb = np.concatenate([luma, luma, luma], axis=-1)
            if want_lane:
                ll = 0.299 * lane_rgb[:, 0:1] + 0.587 * lane_rgb[:, 1:2] + 0.114 * lane_rgb[:, 2:3]
                lane_rgb = np.concatenate([ll, ll, ll], axis=-1)
    if want_lane:
        return (phi.astype(np.float32), rgb.astype(np.float32),
                lane_rgb.astype(np.float32), margin.astype(np.float32))
    return phi.astype(np.float32), rgb.astype(np.float32)


def _dir_feats(coords, argmax_hw, n_freqs, fa, fc, tau):
    # self-orient: tangent of the decoder's OWN all-class argmax boundary -> oriented Fourier feats.
    from scipy import ndimage
    a = np.asarray(argmax_hw); h, w = a.shape
    b = np.zeros(a.shape, bool)
    b[:-1, :] |= a[:-1, :] != a[1:, :]; b[1:, :] |= a[:-1, :] != a[1:, :]
    b[:, :-1] |= a[:, :-1] != a[:, 1:]; b[:, 1:] |= a[:, :-1] != a[:, 1:]
    if not b.any():
        tx = np.ones(h * w, np.float32); ty = np.zeros(h * w, np.float32)
    else:
        dist = ndimage.distance_transform_edt(~b).astype(np.float32)
        gy = np.zeros_like(dist); gx = np.zeros_like(dist)
        gy[1:-1, :] = (dist[2:, :] - dist[:-2, :]) * 0.5; gx[:, 1:-1] = (dist[:, 2:] - dist[:, :-2]) * 0.5
        tx = -gy.ravel(); ty = gx.ravel(); nrm = np.sqrt(tx * tx + ty * ty)
        flat = nrm < 1e-6; nrm = np.maximum(nrm, 1e-8); tx = tx / nrm; ty = ty / nrm
        tx[flat] = 1.0; ty[flat] = 0.0
    cx = coords[:, 0]; cy = coords[:, 1]; nx = ty; ny = -tx
    u_t = cx * tx + cy * ty; u_n = cx * nx + cy * ny; tp = 2.0 * np.pi
    feats = []
    for k in range(int(n_freqs)):
        a_f = float(fa) * (2.0 ** k); c_f = float(fc) * (2.0 ** k)
        feats += [np.sin(tp * a_f * u_t), np.cos(tp * a_f * u_t), np.sin(tp * c_f * u_n), np.cos(tp * c_f * u_n)]
    return np.stack(feats, axis=-1).astype(np.float32)


# --- #224 Wave E: decode-consistent analytic-lane RENDER-BAND (FREE rasterizer, rule 118) ------------
# The lane MANIFOLD COORDS (per-pair LaneLine coeffs) are the COUNTED 5th block; these fns REGENERATE
# the (H,W) coverage + composite the band over the witness render for FREE (0 archive bytes) -- op-for-
# op mirrors of tac.boundary_math.{lane_sdf_component,analytic_lane_render_band} + composite_band_on_render.
LANE_MAGIC = b"LBND1\x00"


def _lane_parse(blob):
    # bit-exact inverse of analytic_lane_render_band.serialize_lane_band. Returns (pairs_lines, header);
    # each line = (centerline f64[], halfwidth f64[], dash_period, dash_phase, dash_duty, fwd0, fwd1).
    assert blob[:len(LANE_MAGIC)] == LANE_MAGIC, "bad lane-band magic"
    off = len(LANE_MAGIC); (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    hdr = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    vals = np.frombuffer(blob[off:], dtype=np.float64)
    pairs = []; vi = 0
    for line_metas in hdr["pairs"]:
        lines = []
        for meta in line_metas:
            nc = int(meta["nc"]); nh = int(meta["nh"]); hd = bool(meta["has_dash"])
            cc = np.asarray(vals[vi:vi + nc], np.float64); vi += nc
            hc = np.asarray(vals[vi:vi + nh], np.float64); vi += nh
            dp = dph = 0.0; dd = 0.5
            if hd:
                dp = float(vals[vi]); dph = float(vals[vi + 1]); dd = float(vals[vi + 2]); vi += 3
            fr0 = float(vals[vi]); fr1 = float(vals[vi + 1]); vi += 2
            lines.append((cc, hc, dp, dph, dd, fr0, fr1))
        pairs.append(lines)
    return pairs, hdr


# Wave-F: LBND2 optimal RD parse -- bit-exact inverse of analytic_lane_render_band.
# serialize_lane_band_rd. PURE numpy/struct/json (NO constriction/mlx dep): quantize ->
# temporal-delta -> zigzag-uint32 stream + brotli (outer) is the entropy backend. Returns
# the SAME (cc, hc, dp, dph, dd, fr0, fr1) tuple format _lane_coverage consumes.
LANE_MAGIC_RD = b"LBND2\x00"
_D_SLOT_RD = 11


def _lane_parse_rd(blob):
    assert blob[:len(LANE_MAGIC_RD)] == LANE_MAGIC_RD, "bad LBND2 magic"
    off = len(LANE_MAGIC_RD); (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    hdr = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    (plen,) = struct.unpack_from("<I", blob, off); off += 4
    pbytes = blob[off:off + plen]; off += plen
    rd = hdr["rd"]; K = int(rd["K"]); P = int(rd["n_pairs"]); ds = int(rd["d_slot"])
    steps = np.asarray(rd["base_steps"], np.float64); D = K * ds
    pairs = []
    if K:
        presence = np.unpackbits(np.frombuffer(pbytes, dtype=np.uint8))[:P * K].reshape(P, K).astype(bool)
        zz = np.frombuffer(blob[off:], dtype=np.uint32).reshape(P, D).astype(np.int64)
        dq = (zz >> 1) ^ -(zz & 1)                     # zigzag -> signed int64
        Q = np.cumsum(dq, axis=0)                      # undo temporal delta (row0=seed)
        M = Q.astype(np.float64) * np.tile(steps, K)   # dequant (float64, bit-exact vs module)
        for p in range(P):
            lines = []
            for slot in range(K):
                if presence[p, slot]:
                    v = M[p, slot * ds:(slot + 1) * ds]
                    lines.append((np.asarray(v[0:4], np.float64), np.asarray(v[4:6], np.float64),
                                  float(v[6]), float(v[7]), float(v[8]), float(v[9]), float(v[10])))
            pairs.append(lines)
    else:
        pairs = [[] for _ in range(P)]
    return pairs, hdr


def _lane_parse_any(blob):
    # magic dispatch: LBND2 (Wave-F RD) else LBND1 (Wave-E naive).
    if blob[:len(LANE_MAGIC_RD)] == LANE_MAGIC_RD:
        return _lane_parse_rd(blob)
    return _lane_parse(blob)


def _lane_coverage(lines, rh, rw, hdr):
    # AA-SDF range-dependent coverage raster (float64 -> float32), mirror of
    # rasterize_lane_coverage_range_dependent + _line_row_params + _forward_of_rows. Returns (rh,rw).
    g = hdr["geom"]; cam_h = float(g["cam_h"]); fx = float(g["fx"]); fy = float(g["fy"])
    v_h = float(hdr["v_h"]); soft = max(float(hdr["softness"]), 1e-6)
    dash_gate = bool(hdr["dash_gate"]); dfm = float(hdr["dash_forward_max_m"])
    cxx = float(rw / 2.0) if hdr.get("cx") is None else float(hdr["cx"])
    cov = np.zeros((rh, rw), np.float32)
    if not lines:
        return cov
    rows = np.arange(rh, dtype=np.float64); below = rows > (v_h + 1.0)
    if not below.any():
        return cov
    vr = rows[below]; col = np.arange(rw, dtype=np.float64)[None, :]
    forward = cam_h * fy / np.maximum(vr - v_h, 1e-3)
    acc = np.zeros((int(below.sum()), rw), np.float64)
    for (cc, hc, dp, dph, dd, fr0, fr1) in lines:
        lateral = np.polyval(cc, forward)
        u_c = cxx - lateral * fx / forward
        hw = np.maximum(np.polyval(hc, vr), 0.5)
        in_range = (forward >= fr0 - 1.0) & (forward <= fr1 + 5.0)
        on = np.ones_like(vr, bool)
        if dash_gate and dp > 0.0:
            near = forward < dfm
            phase = np.mod(forward - dph, dp) / dp
            on = np.where(near, phase < dd, True)
        gate = (on & in_range).astype(np.float64)
        s = hw[:, None] - np.abs(col - u_c[:, None])
        acc = np.maximum(acc, np.clip(s / soft + 0.5, 0.0, 1.0) * gate[:, None])
    cov[below] = acc.astype(np.float32)
    return cov


def _lane_uncert(margin, tau, eps):
    # mirror of analytic_lane_render_band.witness_uncertainty_mask.
    return np.clip((float(tau) - np.asarray(margin, np.float32)) / max(float(eps), 1e-6) + 0.5, 0.0, 1.0).astype(np.float32)


def _lane_composite(rgb, lane_rgb, cov_flat, margin, hdr):
    # comp = rgb*(1-a) + lane_rgb*a ; a = (cov*weight)*u_mask  (mirror composite_band_on_render/band_alpha).
    a = np.asarray(cov_flat, np.float32) * np.float32(hdr["weight"])
    um = hdr.get("u_mask")
    if um is not None and margin is not None:
        a = a * _lane_uncert(margin, um["tau"], um["eps"])
    a = a[:, None]
    return (np.asarray(rgb, np.float32) * (1.0 - a) + np.asarray(lane_rgb, np.float32) * a).astype(np.float32)


def _R(rgb, rh, rw, ch, cw):
    x = torch.from_numpy(np.ascontiguousarray(rgb.reshape(rh, rw, 3))).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


_G = {}


def _setup(src):
    # per-worker (spawn) / inherited-then-reset (fork) setup: dequant params + regen the FREE curvelet
    # bank + coords. Deterministic + identical across workers -> bit-exact. ~150ms, amortized over the
    # worker's pairs. Same op order as the serial main -> identical output.
    m, base_b, code_b, _pose, lane_b = _read_blob(src)
    params = _dequant(brotli.decompress(base_b), m["base_param_order"], m["base_shapes"], m["base_scales"])
    code = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32) * float(m["code_scale"])).reshape(m["code_shape"])
    rh, rw, ch, cw = int(m["render_h"]), int(m["render_w"]), int(m["camera_h"]), int(m["camera_w"])
    coords = _coords(rh, rw)
    B = _curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"], m["bank_n_iso"], m["max_bank_freq"])
    curv = _curvelet_feats(coords, B)
    P = {k: np.asarray(v, np.float64) for k, v in params.items()}  # convert once (bit-identical)
    # #224 Wave E: parse the OPTIONAL lane render-band (per-pair coords + hdr) -> per-pair coverage
    # rasters (FREE regen; computed ONCE per pair, cached). Absent -> lane_pairs=None -> band skipped.
    lane_pairs = lane_hdr = None
    if m.get("lane_render_band") is not None and lane_b is not None:
        lane_pairs, lane_hdr = _lane_parse_any(brotli.decompress(lane_b))  # Wave-F: LBND1/LBND2 dispatch
    _G.update(m=m, code=code, coords=coords, curv=curv, P=P, rh=rh, rw=rw, ch=ch, cw=cw,
              framebytes=ch * cw * 3, dst=None, lane_pairs=lane_pairs, lane_hdr=lane_hdr)


def _render_pair(pi):
    # op-for-op the serial per-pair body; each pair is INDEPENDENT so parallel == serial (bit-identical).
    # Writes the pair's 2 frames to disjoint offsets of the preallocated .raw (POSIX-safe concurrent write).
    m, code, coords, curv, P = _G["m"], _G["code"], _G["coords"], _G["curv"], _G["P"]
    rh, rw, ch, cw = _G["rh"], _G["rw"], _G["ch"], _G["cw"]
    if m["self_orient"]:
        # fixed-point: dir feats from the decoder's OWN frame1 argmax (GT-free, 0 bytes).
        dirf = np.zeros((curv.shape[0], 4 * int(m["n_dir_freqs"])), np.float32)
        prev_am = None
        for _ in range(int(m["so_iters"])):
            feats = np.concatenate([curv, dirf], axis=-1)
            phi, _ = _outputs_from_h0(P, _in_proj_h0(P, feats, m), code[2 * pi + 1], m, False)
            am = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
            if prev_am is not None and np.array_equal(am, prev_am):
                break  # argmax fixed point: dirf would not change -> remaining iters are no-ops
            dirf = _dir_feats(coords, am, m["n_dir_freqs"], m["so_freq_along"], m["so_freq_across"], m["so_tau"])
            prev_am = am
        feats = np.concatenate([curv, dirf], axis=-1)
    else:
        feats = curv
    h0 = _in_proj_h0(P, feats, m)  # shared across the pair's 2 frames (identical feats)
    lane_pairs, lane_hdr = _G.get("lane_pairs"), _G.get("lane_hdr")
    band = lane_pairs is not None and pi < len(lane_pairs)
    cov_flat = None
    if band:
        # coverage depends ONLY on the pair's lines (per-pair) -> rasterize ONCE, share across f0/f1.
        cov_flat = _lane_coverage(lane_pairs[pi], rh, rw, lane_hdr).reshape(-1)
    frames = []
    for fk in range(2):
        if band:
            _phi, rgb, lane_rgb, margin = _outputs_from_h0(P, h0, code[2 * pi + fk], m, True, True)
            rgb = _lane_composite(rgb, lane_rgb, cov_flat, margin, lane_hdr)
        else:
            _phi, rgb = _outputs_from_h0(P, h0, code[2 * pi + fk], m, True)
        frames.append(_R(rgb, rh, rw, ch, cw).tobytes())
    fb = _G["framebytes"]
    with open(_G["dst"], "r+b") as f:
        f.seek(pi * 2 * fb); f.write(b"".join(frames))
    return pi


def _init_worker(src, dst):
    _setup(src); _G["dst"] = dst


def main():
    src, dst = sys.argv[1], sys.argv[2]
    _setup(src)  # main process: get n_pairs + dims (workers re-setup via _init_worker)
    m = _G["m"]; n_pairs = int(m["n_pairs"]); ch, cw = _G["ch"], _G["cw"]
    _cap = int(os.environ.get("INFLATE_MAX_PAIRS", "0"))  # 0 => all pairs (contest default); >0 = debug/CI bounded inflate
    if _cap > 0:
        n_pairs = min(n_pairs, _cap)
    with open(dst, "wb") as f:  # preallocate the full .raw so workers write disjoint offsets
        f.truncate(2 * n_pairs * _G["framebytes"])
    nworkers = max(1, min(n_pairs, int(os.environ.get("INFLATE_WORKERS", "0")) or (os.cpu_count() or 1)))
    if nworkers == 1:  # serial fallback (bit-identical) -- e.g. INFLATE_WORKERS=1 for debugging
        _G["dst"] = dst
        for pi in range(n_pairs):
            _render_pair(pi)
    else:
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork" if "fork" in methods else "spawn")  # Linux fork / macOS spawn
        with ctx.Pool(nworkers, initializer=_init_worker, initargs=(src, dst)) as pool:
            for _ in pool.imap_unordered(_render_pair, range(n_pairs), chunksize=1):
                pass
    print("inflated %d frames (%d pairs) -> %s [%dx%dx%dx3 uint8] (%d workers)" % (2 * n_pairs, n_pairs, dst, 2 * n_pairs, ch, cw, nworkers), flush=True)


if __name__ == "__main__":
    main()
'''

_INFLATE_SH = """#!/usr/bin/env bash
# Level-set inflate launcher. Produces <OUTPUT_DIR>/<base>.raw = flat uint8 (N,874,1164,3).
# Interpreter resolution: honor ${PYTHON} (canonical contest_auth_eval._run_inflate sets it to the
# deps-complete eval interpreter), else fall back to python3 (python3 is present on macOS + most
# Linux x86_64 CPU hosts where bare `python` is NOT -- the runtime-closure bug class that left
# inflate.sh dead with `python: command not found`). Generic launcher code, rule-118 clean.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${PYTHON:-python3}"
DATA_DIR="$1"; OUTPUT_DIR="$2"; FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  printf "Inflating %s ... " "$line"
  "$PYBIN" "${HERE}/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""


# ---------------------------------------------------------------------------
# packet assembly: archive.zip (0.bin) + inflate.py + inflate.sh
# ---------------------------------------------------------------------------
def assemble_packet(blob: bytes, packet_dir: Path) -> tuple[Path, int]:
    packet_dir.mkdir(parents=True, exist_ok=True)
    zip_path = packet_dir / "archive.zip"
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, blob)
    (packet_dir / "inflate.py").write_text(_INFLATE_PY)
    sh = packet_dir / "inflate.sh"
    sh.write_text(_INFLATE_SH)
    sh.chmod(0o755)
    return zip_path, int(zip_path.stat().st_size)


# ---------------------------------------------------------------------------
# run inflate (subprocess, exactly as the contest evaluate.sh does)
# ---------------------------------------------------------------------------
def run_inflate(packet_dir: Path, n_pairs_total: int, max_pairs: int | None) -> dict[str, Any]:
    """Unzip archive.zip -> 0.bin, run inflate.py -> 0.raw, validate the FULL output shape.

    ``max_pairs`` (test/speed): when set < n_pairs, a CAPPED 0.bin is written so inflate emits
    only the first ``max_pairs`` pairs (still frame0+frame1 each). The full archive.zip always
    encodes ALL codes (the rate term is the full blob)."""
    archive_dir = packet_dir / "archive"
    inflated_dir = packet_dir / "inflated"
    archive_dir.mkdir(exist_ok=True)
    inflated_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(packet_dir / "archive.zip") as zf:
        zf.extractall(archive_dir)
    src_bin = archive_dir / "0.bin"

    eval_pairs = n_pairs_total if max_pairs is None else min(int(max_pairs), n_pairs_total)
    if eval_pairs < n_pairs_total:
        import brotli

        man, base_b, code_b, pose_b, lane_b = _read_blob_bytes(src_bin.read_bytes())
        code = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32)
                * man["code_scale"]).reshape(man["code_shape"])
        code_cap = code[: 2 * eval_pairs]
        qc, sc = _int8_symmetric(code_cap)
        man["n_pairs"] = eval_pairs
        man["code_shape"] = list(code_cap.shape)
        man["code_scale"] = float(sc)
        # #224 Wave E: cap the lane render-band to eval_pairs too (slice + re-serialize).
        lane_cap = None
        if lane_b is not None and man.get("lane_render_band") is not None:
            all_pairs, lane_hdr = deserialize_lane_band_any(brotli.decompress(lane_b))  # Wave-F dispatch
            lane_cfg_cap = render_config_from_header({**man["lane_render_band"], "pairs": []})
            lane_cap = brotli.compress(
                serialize_lane_band_any(all_pairs[:eval_pairs], lane_cfg_cap, lane_hdr), quality=11)
        mj = json.dumps(man, separators=(",", ":")).encode()
        capped = _io_pack(mj, base_b, brotli.compress(qc.astype(np.int8).tobytes(), quality=11), pose_b or None, lane_cap)
        src_bin.write_bytes(capped)

    dst_raw = inflated_dir / "0.raw"
    cmd = [sys.executable, str(packet_dir / "inflate.py"), str(src_bin), str(dst_raw)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(packet_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"inflate.py FAILED rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
    n_frames_expected = 2 * eval_pairs
    expected_bytes = n_frames_expected * CAMERA_H * CAMERA_W * 3
    actual_bytes = dst_raw.stat().st_size
    return {
        "inflate_stdout": proc.stdout.strip(),
        "eval_pairs": eval_pairs,
        "n_frames_emitted": n_frames_expected,
        "raw_path": str(dst_raw),
        "raw_bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "full_output_shape_ok": bool(actual_bytes == expected_bytes),
        "frame_layout": f"({n_frames_expected}, {CAMERA_H}, {CAMERA_W}, 3) uint8 [f0,f1 per pair]",
    }


# ---------------------------------------------------------------------------
# d_seg / d_pose parity ON THE INFLATED FRAMES (frozen CPU-torch, vs GT)
# ---------------------------------------------------------------------------
def parity_on_inflated(raw_path: Path, eval_pairs: int, gt_cache: str | None, num_pairs: int) -> dict[str, Any]:
    """Read the .raw back, run the FROZEN CPU-torch SegNet/PoseNet on the INFLATED frames over all
    eval pairs, return realized d_seg/d_pose vs GT (the contest-faithful realized numbers)."""
    if gt_cache:
        gt, seg_cpu, posenet_cpu = twr.load_gt_from_cache(Path(gt_cache), num_pairs)
    else:
        gt, seg_cpu, posenet_cpu = twr.precompute_gt(num_pairs)
    P = min(eval_pairs, gt.n_pairs)
    frame_bytes = CAMERA_H * CAMERA_W * 3
    f0s, f1s = [], []
    with open(raw_path, "rb") as f:
        for _pi in range(P):
            f0s.append(np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
            f1s.append(np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
    d_segs = twr.cpu_verdict_d_seg_batch(seg_cpu, f1s, [gt.lstars[pi] for pi in range(P)])
    d_poses = twr.cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, [gt.gt_poses[pi] for pi in range(P)])
    return {
        "pairs_scored": P,
        "d_seg_realized_on_inflated": float(np.mean(d_segs)),
        "d_pose_realized_on_inflated": float(np.mean(d_poses)),
    }


# ---------------------------------------------------------------------------
# BIT-EXACT ROUND-TRIP GATE (the correctness proof):
#   shipped inflate.py(archive) output  ==  the canonical numpy-fp32 ORACLE forward
#   (tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy) of the SAME
#   int8-DEQUANTIZED checkpoint, over the SAME regenerated feats (curvelet bank + self-orient
#   fixed point), bit-for-bit on the uint8 .raw frames (np.array_equal), which subsumes the SegNet
#   argmax the score reads. This proves the shipped decoder faithfully implements the generator
#   (it does NOT measure quant error -- both sides use the SAME dequantized weights; quant error is
#   the separate realized-vs-fp32-EMA quantity the trainer verdict tracks).
# ---------------------------------------------------------------------------
def _torch_R_reference(rgb: np.ndarray, rh: int, rw: int, ch: int, cw: int) -> np.ndarray:
    """Contest R, byte-identical to the shipped inflate.py ``_R``: bicubic up render->camera, round,
    clamp -> uint8. ALWAYS fp32 (matches the shipped ``.float()`` cast)."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(rgb.reshape(rh, rw, 3))).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def numpy_oracle_reference_frames(
    params: dict[str, np.ndarray], code: np.ndarray, manifest: dict[str, Any], n_pairs: int,
    lane_pairs: list[list[Any]] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Regenerate the FULL uint8 camera frames for the first ``n_pairs`` pairs via the CANONICAL
    numpy-fp32 oracle (``levelset_rgb_forward_numpy`` / ``levelset_band_forward_numpy``) + the
    canonical FREE-table regen (``torch_levelset_inflate`` numpy helpers) + the reference R. This is
    the independent authority the shipped inflate.py must match bit-for-bit. Returns (frames
    [2*n_pairs uint8 arrays, f0,f1 per pair], final_frame_argmax [render-res int argmax per pair]).

    The self-orient fixed point mirrors the shipped inflate EXACTLY. When ``manifest`` carries the
    ``lane_render_band`` cfg AND ``lane_pairs`` (the deserialized per-pair ``LaneLine`` lists), the
    CANONICAL render-band (``rasterize_lane_coverage_range_dependent`` + ``composite_band_on_render``
    + ``witness_uncertainty_mask``) is composited over each frame BEFORE R -- so the bit-exact gate
    proves the shipped inflate's inline band == this canonical band. ``params``/``code`` are the
    int8-DEQUANTIZED values read back from the byte-closed blob (both sides render the SAME weights)."""
    rh, rw = int(manifest["render_h"]), int(manifest["render_w"])
    ch, cw = int(manifest["camera_h"]), int(manifest["camera_w"])
    coords = _canon_coords_grid(rh, rw)
    B = _canon_curvelet_B(
        manifest["bank_n_scales"], manifest["bank_n_orient0"], manifest["bank_f0"],
        manifest["bank_base"], manifest["bank_n_iso"], manifest["max_bank_freq"],
    )
    curv = _canon_curvelet_feats(coords, B)
    lr = manifest.get("lane_render_band")
    band_on = bool(lr is not None and lane_pairs is not None)
    lane_cfg = render_config_from_header({**lr, "pairs": []}) if band_on else None
    fwd_kw = {
        "n_hidden": int(manifest["n_hidden"]), "hidden_dim": int(manifest["hidden_dim"]),
        "n_classes": int(manifest["n_classes"]), "activation": str(manifest["activation"]),
        "softmax_temp": float(manifest["softmax_temp"]), "wire_w0": float(manifest["wire_w0"]),
        "wire_s0": float(manifest["wire_s0"]), "hosc_beta": float(manifest["hosc_beta"]),
        "hosc_omega": float(manifest["hosc_omega"]), "chroma": bool(manifest["chroma"]),
    }
    frames: list[np.ndarray] = []
    argmaxes: list[np.ndarray] = []
    for pi in range(n_pairs):
        if bool(manifest["self_orient"]):
            ndf = int(manifest["n_dir_freqs"])
            dirf = np.zeros((curv.shape[0], 4 * ndf), np.float32)
            prev_am = None
            for _ in range(int(manifest["so_iters"])):
                feats = np.concatenate([curv, dirf], axis=-1)
                _rgb, phi = levelset_rgb_forward_numpy(params, feats, code[2 * pi + 1], **fwd_kw)
                am = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
                if prev_am is not None and np.array_equal(am, prev_am):
                    break  # argmax fixed point -> dirf frozen -> remaining iters no-ops (== shipped)
                dirf = _canon_dir_feats(
                    coords, am, ndf, float(manifest["so_freq_along"]),
                    float(manifest["so_freq_across"]), float(manifest["so_tau"]),
                )
                prev_am = am
            feats = np.concatenate([curv, dirf], axis=-1)
        else:
            feats = curv
        cov = None
        if band_on and pi < len(lane_pairs):
            cov = rasterize_lane_coverage_range_dependent(
                lane_pairs[pi], h=rh, w=rw, softness=lane_cfg.softness,
                dash_gate=lane_cfg.dash_gate, dash_forward_max_m=lane_cfg.dash_forward_max_m,
                v_h=lane_cfg.v_h, cx=lane_cfg.cx,
            ).reshape(-1)
        for fk in range(2):
            if cov is not None:
                rgb, phi, lane_rgb, margin = levelset_band_forward_numpy(
                    params, feats, code[2 * pi + fk], lane_cls=lane_cfg.lane_cls, **fwd_kw)
                um = (witness_uncertainty_mask(margin, tau=lane_cfg.u_mask_tau, eps=lane_cfg.u_mask_eps)
                      if lane_cfg.u_mask_enabled else None)
                rgb = composite_band_on_render(rgb, lane_rgb, cov, um, lane_cfg.weight)
            else:
                rgb, phi = levelset_rgb_forward_numpy(params, feats, code[2 * pi + fk], **fwd_kw)
            frames.append(_torch_R_reference(rgb, rh, rw, ch, cw))
            if fk == 1:
                argmaxes.append(phi.argmax(-1).reshape(rh, rw).astype(np.int64))
    return frames, argmaxes


def bit_exact_roundtrip_gate(
    packet_dir: Path, blob: bytes, gate_pairs: int, strict: bool
) -> dict[str, Any]:
    """THE correctness proof. Inflate a ``gate_pairs``-capped 0.bin with the SHIPPED inflate.py,
    read back the uint8 .raw, and assert it is BIT-IDENTICAL (np.array_equal) to the canonical
    numpy-fp32 oracle forward of the SAME dequantized checkpoint. Returns a report dict; raises in
    ``strict`` mode on any mismatch."""
    import brotli

    # dequant the blob EXACTLY as the shipped inflate does (int8 -> fp32 * scale).
    m, base_b, code_b, _pose, lane_b = _read_blob_bytes(blob)
    order = m["base_param_order"]
    flat = np.frombuffer(brotli.decompress(base_b), dtype=np.int8)
    params: dict[str, np.ndarray] = {}
    off = 0
    for name in order:
        shp = tuple(m["base_shapes"][name])
        n = int(np.prod(shp))
        params[name] = (flat[off:off + n].astype(np.float32) * float(m["base_scales"][name])).reshape(shp)
        off += n
    code = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32)
            * float(m["code_scale"])).reshape(m["code_shape"])
    n_pairs_total = int(m["n_pairs"])
    gp = max(1, min(int(gate_pairs), n_pairs_total))

    # SHIPPED inflate.py on a gp-capped 0.bin -> .raw frames.
    gate_root = packet_dir / "_bitexact_gate"
    gate_root.mkdir(parents=True, exist_ok=True)
    code_cap = code[: 2 * gp]
    qc, sc = _int8_symmetric(code_cap)
    man = dict(m)
    man["n_pairs"] = gp
    man["code_shape"] = list(code_cap.shape)
    man["code_scale"] = float(sc)
    # #224 Wave E: carry the lane render-band through the gp-capped repack (slice pairs to gp) so the
    # shipped inflate composites the SAME band the oracle does over the SAME gp pairs.
    lane_pairs_cap = None
    lane_b_cap = None
    if lane_b is not None and m.get("lane_render_band") is not None:
        all_pairs, lane_hdr = deserialize_lane_band_any(brotli.decompress(lane_b))  # Wave-F dispatch
        lane_pairs_cap = all_pairs[:gp]
        lane_cfg_cap = render_config_from_header({**m["lane_render_band"], "pairs": []})
        lane_b_cap = brotli.compress(
            serialize_lane_band_any(lane_pairs_cap, lane_cfg_cap, lane_hdr), quality=11)
    mj = json.dumps(man, separators=(",", ":")).encode()
    capped_bin = gate_root / "gate.bin"
    capped_bin.write_bytes(_io_pack(
        mj, base_b, brotli.compress(qc.astype(np.int8).tobytes(), quality=11), None, lane_b_cap))
    gate_raw = gate_root / "gate.raw"
    proc = subprocess.run(
        [sys.executable, str(packet_dir / "inflate.py"), str(capped_bin), str(gate_raw)],
        capture_output=True, text=True, cwd=str(packet_dir),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bit-exact gate: shipped inflate.py FAILED rc={proc.returncode}\n{proc.stderr}")
    ch, cw = int(m["camera_h"]), int(m["camera_w"])
    fb = ch * cw * 3
    shipped: list[np.ndarray] = []
    with open(gate_raw, "rb") as f:
        for _ in range(2 * gp):
            shipped.append(np.frombuffer(f.read(fb), dtype=np.uint8).reshape(ch, cw, 3))

    # canonical numpy-fp32 ORACLE reference (dequant capped code so both sides see identical values).
    ref_params: dict[str, np.ndarray] = {}
    for name in order:  # re-dequant from the SAME capped blob for byte-identical inputs
        ref_params[name] = params[name]
    ref_code = (qc.astype(np.float32) * sc).reshape(code_cap.shape)
    ref_frames, ref_argmax = numpy_oracle_reference_frames(ref_params, ref_code, man, gp, lane_pairs_cap)

    max_abs = 0
    all_equal = True
    n_diff_frames = 0
    for a, b in zip(shipped, ref_frames, strict=True):
        if not np.array_equal(a, b):
            all_equal = False
            n_diff_frames += 1
            max_abs = max(max_abs, int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()))
    import shutil
    shutil.rmtree(gate_root, ignore_errors=True)  # disk-hygiene (rebuildable from archive.zip)

    result = {
        "bit_exact": bool(all_equal),
        "gate_pairs": gp,
        "frames_compared": 2 * gp,
        "n_frames_differing": n_diff_frames,
        "max_abs_uint8_diff": int(max_abs),
        "reference": "tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy "
                     "(numpy-fp32 oracle) + canonical FREE-table regen + reference R",
        "proves": "shipped inflate.py bit-identically implements the canonical generator on the "
                  "int8-dequantized checkpoint (argmax the score reads is byte-determined)",
    }
    verdict = "BIT-EXACT" if all_equal else f"MISMATCH ({n_diff_frames} frames, max_abs={max_abs})"
    print(f"[bit-exact gate] {verdict} over {2 * gp} frames vs numpy-fp32 oracle  {_AUTHORITY}", flush=True)
    if not all_equal and strict:
        raise RuntimeError(
            f"BIT-EXACT GATE FAILED: shipped inflate.py != numpy-fp32 oracle on {n_diff_frames} of "
            f"{2 * gp} frames (max_abs={max_abs}). The shipped decoder diverges from the canonical "
            "generator -> the byte-close is NOT a faithful witness (NO-FAKE).")
    return result


def _read_blob_bytes(blob: bytes) -> tuple[dict[str, Any], bytes, bytes, bytes, bytes | None]:
    """Parse an in-memory LVLS1 blob (same grammar as the shipped inflate._read_blob). Returns
    (manifest, base, code, pose, lane_band|None); ``lane_band`` is None when the 4-block (pre-Wave-E)
    grammar is present (default-off)."""
    assert blob[: len(_MAGIC)] == _MAGIC, "bad level-set magic"
    off = len(_MAGIC)
    out: list[bytes] = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", blob, off)
        off += 4
        out.append(blob[off:off + n])
        off += n
    lane_band: bytes | None = None
    if off < len(blob):
        (n,) = struct.unpack_from("<I", blob, off)
        off += 4
        lane_band = blob[off:off + n]
        off += n
    return json.loads(out[0].decode("utf-8")), out[1], out[2], out[3], lane_band


# ---------------------------------------------------------------------------
# REAL upstream/evaluate.py wrapper (CPU only; NEVER MPS) on the EXACT archive bytes.
#   archive.zip (0.bin) + inflated/0.raw  ->  upstream/evaluate.py --device cpu  ->  real
#   d_seg + d_pose + rate + S, cross-checked against tac.contest_score.compute_contest_score.
# This is the exact-eval path that turns the advisory realized-parity into a REAL evaluate.py row.
# ---------------------------------------------------------------------------
_EVAL_REPORT_PATTERNS = {
    "d_pose": r"Average PoseNet Distortion:\s*([0-9.eE+\-]+)",
    "d_seg": r"Average SegNet Distortion:\s*([0-9.eE+\-]+)",
    "rate": r"Compression Rate:\s*([0-9.eE+\-]+)",
    "final_score": r"Final score[^=]*=\s*([0-9.eE+\-]+)",
    "n_samples": r"Evaluation results over\s*([0-9]+)\s*samples",
}


def _parse_evaluate_report(text: str) -> dict[str, Any]:
    """Parse upstream/evaluate.py's report block (printed lines 93-101) into a structured dict.
    NO-FAKE: a missing required field raises (never fabricate a score)."""
    out: dict[str, Any] = {}
    for key, pat in _EVAL_REPORT_PATTERNS.items():
        mobj = re.search(pat, text)
        if mobj is None:
            if key == "n_samples":
                out[key] = None
                continue
            raise ValueError(f"evaluate.py report missing {key!r} (pattern {pat!r}); refusing to "
                             "fabricate a score (NO-FAKE). Report text:\n" + text[:2000])
        out[key] = int(mobj.group(1)) if key == "n_samples" else float(mobj.group(1))
    return out


def run_upstream_evaluate(
    packet_dir: Path, *, device: str, uncompressed_dir: Path, video_names_file: Path,
    archive_bytes: int, timeout: int,
) -> dict[str, Any]:
    """Run the REAL contest scorer (``upstream/evaluate.py --device <cpu|cuda>``) on the exact
    packet bytes (``packet_dir/archive.zip`` + ``packet_dir/inflated/0.raw``, both already produced
    by run_inflate) and return the real d_seg/d_pose/rate + evaluate.py's own Final score, plus our
    recomputed S via ``compute_contest_score`` (cross-check). CPU by default; MPS is REFUSED."""
    if device == "mps":
        raise ValueError("MPS is NEVER a score authority (CLAUDE.md). Use --eval-device cpu (or cuda).")
    from tac.contest_score import compute_contest_score

    submission_dir = packet_dir  # has archive.zip AND inflated/0.raw (== upstream/evaluate.py layout)
    inflated = submission_dir / "inflated"
    if not (submission_dir / "archive.zip").exists():
        raise FileNotFoundError(f"run_upstream_evaluate: missing {submission_dir/'archive.zip'} (NO-FAKE).")
    raws = list(inflated.glob("*.raw"))
    if not raws:
        raise FileNotFoundError(
            f"run_upstream_evaluate: no inflated .raw in {inflated} -- the full inflate must run "
            "FIRST (do NOT cap --max-pairs for the exact row; evaluate.py needs all 600 pairs).")
    uncompressed_dir = Path(uncompressed_dir).resolve()
    video_names_file = Path(video_names_file).resolve()
    if not uncompressed_dir.exists():
        raise FileNotFoundError(f"--uncompressed-dir missing: {uncompressed_dir} (needs the GT 0.mkv).")
    if not video_names_file.exists():
        raise FileNotFoundError(f"--video-names-file missing: {video_names_file}.")

    report_path = submission_dir / "report.txt"
    evaluate_py = _REPO / "upstream" / "evaluate.py"
    cmd = [
        sys.executable, str(evaluate_py),
        "--submission-dir", str(submission_dir.resolve()),
        "--uncompressed-dir", str(uncompressed_dir),
        "--video-names-file", str(video_names_file),
        "--device", device,
        "--report", str(report_path.resolve()),
        "--batch-size", "8",
    ]
    env = dict(os.environ)
    up = str((_REPO / "upstream").resolve())
    env["PYTHONPATH"] = up + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU (defensive: never let it pick an MPS/CUDA path)
    print(f"[exact-eval] running upstream/evaluate.py --device {device} on {submission_dir.name} "
          f"(this is the REAL contest scorer; CPU 600-pair ~1-2h)  {_AUTHORITY}", flush=True)
    proc = subprocess.run(cmd, cwd=up, env=env, capture_output=True, text=True, timeout=timeout)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"upstream/evaluate.py FAILED rc={proc.returncode}\n{combined[-4000:]}")

    src_text = report_path.read_text() if report_path.exists() else combined
    parsed = _parse_evaluate_report(src_text)
    d_seg = float(parsed["d_seg"])
    d_pose = float(parsed["d_pose"])
    recomputed_S = compute_contest_score(d_seg, d_pose, archive_bytes)
    axis = "[contest-CPU]" if (_AUTHORITY.startswith("[contest-CPU")) else "[macOS-CPU advisory]"
    return {
        "ran": True,
        "device": device,
        "evaluate_py_final_score": float(parsed["final_score"]),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "rate_from_evaluate": float(parsed["rate"]),
        "n_samples": parsed.get("n_samples"),
        "recomputed_S_compute_contest_score": recomputed_S,
        "recomputed_vs_evaluate_delta": abs(recomputed_S - float(parsed["final_score"])),
        "archive_bytes_scored": int(archive_bytes),
        "report_path": str(report_path),
        "score_axis": axis,
        "authority": _AUTHORITY,
        "promotion_claim": False,
    }


# ---------------------------------------------------------------------------
# #224 Wave E: build the decode-consistent analytic-lane render-band section.
# ---------------------------------------------------------------------------
def _lane_manifest_from_cfg(cfg: LaneBandRenderConfig, fit_stats: dict[str, Any]) -> dict[str, Any]:
    """The manifest ``lane_render_band`` cfg (scalars ``render_config_from_header`` reads + geometry +
    fit observability). The lane MANIFOLD COORDS ride the counted 5th block, NOT this."""
    from tac.boundary_math.lane_sdf_component import _CAM_H, _FX, _FY, _SEG_H, _SEG_W

    return {
        "softness": float(cfg.softness), "dash_gate": bool(cfg.dash_gate),
        "dash_forward_max_m": float(cfg.dash_forward_max_m), "v_h": float(cfg.v_h),
        "cx": (None if cfg.cx is None else float(cfg.cx)), "weight": float(cfg.weight),
        "lane_cls": int(cfg.lane_cls), "lane_rgb_mode": str(cfg.lane_rgb_mode),
        "u_mask": ({"source": "witness_margin", "tau": float(cfg.u_mask_tau), "eps": float(cfg.u_mask_eps)}
                   if cfg.u_mask_enabled else None),
        "geom": {"cam_h": _CAM_H, "fx": _FX, "fy": _FY, "seg_h": _SEG_H, "seg_w": _SEG_W},
        "fit_stats": fit_stats,
    }


def build_lane_band_section(
    gt_cache: str | None, n_pairs: int, cfg: LaneBandRenderConfig, *, rd: bool = True,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Fit the per-pair lane manifold coords from the GT SegNet argmax cache (compress-time; the
    source is fully available), serialize + brotli them (COUNTED), and build the manifest cfg.
    Returns (brotli_lane_bytes, lane_manifest, report). NO-FAKE: missing GT cache raises.

    ``rd=True`` (Wave-F default) uses the OPTIMAL LBND2 rate-distortion codec
    (quantize->temporal-delta->L4-slots->zigzag, brotli entropy backend). ``rd=False`` uses
    the naive LBND1 float64 serializer (kept for the default-off byte-identical gate + the
    naive-vs-RD rate comparison). The report carries the MEASURED per-lever byte accounting."""
    import brotli

    if not gt_cache:
        raise ValueError("--lane-render-band requires --gt-cache (the frozen SegNet argmax lstars) to "
                         "fit the lane lines at compress time. NO-FAKE: refusing to fabricate.")
    p = Path(gt_cache)
    if not p.exists():
        raise FileNotFoundError(f"--gt-cache {p} not found (needed to fit the lane band lines).")
    z = np.load(p, allow_pickle=False)
    if "lstars" not in z.files:
        raise ValueError(f"gt cache {p} lacks 'lstars' (the frozen SegNet argmax) -- cannot fit lines.")
    lstars = z["lstars"]
    ncap = min(int(n_pairs), int(len(lstars)))
    lst_list = [np.asarray(lstars[i], np.int64) for i in range(ncap)]
    pairs_lines, fit_stats = build_lane_band_pairs_from_lstars(lst_list, cfg)
    # Wave-F: measured per-lever rate accounting (naive LBND1 vs optimal LBND2 RD).
    rate_report = lane_band_rd_rate_report(pairs_lines, cfg)
    raw = serialize_lane_band_rd(pairs_lines, cfg) if rd else serialize_lane_band(pairs_lines, cfg)
    lane_bytes = brotli.compress(raw, quality=11)
    lane_manifest = _lane_manifest_from_cfg(cfg, fit_stats)
    report = {
        "active": True,
        "codec": ("LBND2_rd" if rd else "LBND1_naive"),
        "source_gt_cache": str(p),
        "n_pairs_fit": ncap,
        "serialized_raw_bytes": len(raw),
        "counted_brotli_bytes": len(lane_bytes),
        "counted_rate_term_contribution": 25.0 * len(lane_bytes) / RATE_DENOM,
        "fit_stats": fit_stats,
        "rate_report": rate_report,  # Wave-F: naive-vs-RD + Shannon floor + PTC1 + induced lat RMS
        "u_mask_enabled": bool(cfg.u_mask_enabled),
        "rule_118": {
            "COUNTED (archive.zip)": ("per-pair LaneLine manifold coords, QUANTIZED (geometric-tolerance "
                                      "steps) + temporal-delta + zigzag -- the video-derived statistic"
                                      if rd else
                                      "per-pair LaneLine manifold coords (float64) -- the video-derived statistic"),
            "FREE (inflate.py)": "quantize/dequantize + rasterize_lane_coverage_range_dependent (the AA-SDF "
                                 "coverage raster) + the composite -- generic deterministic algorithm, 0 archive bytes",
            "no_gt_no_scorer": "no GT mask, no SegNet/PoseNet weights, no per-pixel table ship",
        },
    }
    return lane_bytes, lane_manifest, report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(
    ckpt_dir: Path,
    *,
    npz_name: str | None,
    max_pairs: int | None,
    fold_pose_sidecar: bool,
    pose_sidecar_path: Path | None,
    gt_cache: str | None,
    keep_packet: bool,
    packet_dir: Path | None,
    skip_parity: bool,
    so_overrides: dict[str, Any],
    lane_render_band: bool = False,
    lane_band_cfg: LaneBandRenderConfig | None = None,
    lane_rd: bool = True,  # Wave-F: LBND2 optimal RD codec (default); False -> naive LBND1
    verify_bit_exact: bool = False,
    bit_exact_pairs: int = 2,
    bit_exact_strict: bool = True,
    run_exact_eval: bool = False,
    eval_device: str = "cpu",
    uncompressed_dir: Path | None = None,
    video_names_file: Path | None = None,
    eval_timeout: int = 18000,
) -> dict[str, Any]:
    params, cfg = _load_levelset_ckpt(ckpt_dir, npz_name)
    n_pairs = int(cfg["n_pairs"])
    so = detect_self_orient(cfg, so_overrides)
    # The exact-eval row needs the FULL packet + ALL 600 pairs inflated on disk.
    if run_exact_eval:
        keep_packet = True
        if max_pairs is not None:
            print(f"[exact-eval] ignoring --max-pairs={max_pairs}: the real evaluate.py row needs ALL "
                  f"{n_pairs} pairs (evaluate.py zips full GT with the inflated frames).", flush=True)
            max_pairs = None

    pose_bytes: bytes | None = None
    pose_note = ("off (level-set witness carries pose in per-(pair,frame) codes/texture; train "
                 "--w-pose>0. A stored sidecar adds COUNTED bytes the scorer never reads -> does "
                 "NOT lower realized d_pose).")
    if fold_pose_sidecar:
        if pose_sidecar_path and Path(pose_sidecar_path).exists():
            pose_bytes = Path(pose_sidecar_path).read_bytes()
            pose_note = (f"folded {len(pose_bytes)} COUNTED B from {pose_sidecar_path}; WARNING: the "
                         "inflate render does NOT read it -> realized d_pose is UNCHANGED by the "
                         "sidecar (dead bytes on a code/texture-pose witness; NO-FAKE honesty).")
        else:
            raise FileNotFoundError(
                "--fold-pose-sidecar requires --pose-sidecar-path <posenet_targets.bin>. NO-FAKE: "
                "refusing to fabricate.")

    lane_band_bytes: bytes | None = None
    lane_manifest: dict[str, Any] | None = None
    lane_report: dict[str, Any] = {"active": False}
    if lane_render_band:
        lane_cfg = lane_band_cfg or LaneBandRenderConfig()
        lane_band_bytes, lane_manifest, lane_report = build_lane_band_section(
            gt_cache, n_pairs, lane_cfg, rd=lane_rd)
        _rr = lane_report.get("rate_report", {})
        print(f"[lane-band] active ({lane_report['codec']}): {lane_report['n_pairs_fit']} pairs fit, "
              f"COUNTED brotli={lane_report['counted_brotli_bytes']} B "
              f"(rate_term += {lane_report['counted_rate_term_contribution']:.5f}); "
              f"naive-LBND1 brotli={_rr.get('naive_lbnd1_brotli_bytes','?')} B "
              f"(rd/naive={_rr.get('rd_vs_naive_ratio', float('nan')):.3f}); "
              f"u_mask={lane_report['u_mask_enabled']}; quantize+raster = FREE (rule 118)  {_AUTHORITY}",
              flush=True)

    blob, breakdown = build_levelset_blob(params, cfg, so, pose_bytes, lane_band_bytes, lane_manifest)
    if not breakdown["accounting_matches_canonical"]:
        print(f"[WARN] byte-close accounting (base={breakdown['base_int8_brotli_bytes']}, "
              f"code={breakdown['code_int8_brotli_bytes']}) != canonical quantize_levelset_blob "
              f"({breakdown['canonical_quantize_blob_bytes']}) -- investigate (should match exactly).", flush=True)

    packet_dir = packet_dir or (
        _REPO / "experiments" / "results"
        / f"levelset_packet_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    _refuse_tmp(packet_dir, "packet_dir")
    zip_path, zip_bytes = assemble_packet(blob, packet_dir)
    rate = zip_bytes / RATE_DENOM
    rate_term = 25.0 * rate

    print(f"[ckpt] {ckpt_dir}/{cfg['npz_name']}  n_pairs={n_pairs}  params={breakdown['n_params']}  "
          f"self_orient={so['self_orient']}  {_AUTHORITY}", flush=True)
    print(f"[byte-close] 0.bin={breakdown['total_0bin_bytes']} B  archive.zip={zip_bytes} B  "
          f"rate={rate:.6f} rate_term={rate_term:.4f}  bank=FREE(rule118)  pose={pose_note}", flush=True)

    bit_exact: dict[str, Any] = {"checked": False}
    if verify_bit_exact:
        bit_exact = bit_exact_roundtrip_gate(packet_dir, blob, bit_exact_pairs, bit_exact_strict)
        bit_exact["checked"] = True

    inflate_info = run_inflate(packet_dir, n_pairs, max_pairs)
    print(f"[inflate] {inflate_info['frame_layout']}  full_output_ok={inflate_info['full_output_shape_ok']}  "
          f"raw_bytes={inflate_info['raw_bytes']}", flush=True)

    parity: dict[str, Any] = {"skipped": True}
    if not skip_parity:
        parity = parity_on_inflated(Path(inflate_info["raw_path"]), inflate_info["eval_pairs"], gt_cache, n_pairs)
        d_seg = parity["d_seg_realized_on_inflated"]
        d_pose = parity["d_pose_realized_on_inflated"]
        seg_term = 100.0 * d_seg
        pose_term = (10.0 * d_pose + 1e-12) ** 0.5
        score = seg_term + pose_term + rate_term
        parity.update({"seg_term": seg_term, "pose_term": pose_term, "rate_term": rate_term,
                       "implied_S_advisory": score})
        pose_blind = d_pose > 1.0
        print(f"[parity] d_seg={d_seg:.6f} d_pose={d_pose:.6f} (realized on INFLATED frames, "
              f"{parity['pairs_scored']} pairs) | S_advisory={score:.4f}  {_AUTHORITY}", flush=True)
        if pose_blind:
            print(f"[POSE-BLIND] realized d_pose={d_pose:.3f} >> 0 -> the render is POSE-BLIND "
                  "(w_pose=0). The S pose term dominates and is GARBAGE. A pose-TRAINED render "
                  "(--w-pose>0) is REQUIRED for the level-set ROW; a stored sidecar does NOT fix "
                  "it (the scorer runs PoseNet on the FRAMES).", flush=True)
        parity["pose_blind"] = bool(pose_blind)

    exact_eval: dict[str, Any] = {"ran": False}
    if run_exact_eval:
        exact_eval = run_upstream_evaluate(
            packet_dir,
            device=eval_device,
            uncompressed_dir=(uncompressed_dir or (_REPO / "upstream" / "videos")),
            video_names_file=(video_names_file or (_REPO / "upstream" / "public_test_video_names.txt")),
            archive_bytes=zip_bytes,
            timeout=eval_timeout,
        )
        print(f"[exact-eval] REAL evaluate.py --device {eval_device}: d_seg={exact_eval['d_seg']:.8f} "
              f"d_pose={exact_eval['d_pose']:.8f} -> S={exact_eval['evaluate_py_final_score']:.5f} "
              f"(recomputed {exact_eval['recomputed_S_compute_contest_score']:.5f}, "
              f"delta {exact_eval['recomputed_vs_evaluate_delta']:.2e}) {exact_eval['score_axis']}", flush=True)

    contest_cmd = (
        f".venv/bin/python experiments/contest_auth_eval.py "
        f"--archive {zip_path} --inflate-sh {packet_dir / 'inflate.sh'} "
        f"--device cpu  # [contest-CPU] authoritative ONLY on Linux x86_64 (Modal CPU); macOS-local = advisory"
    )

    report: dict[str, Any] = {
        "tool": "levelset_byte_close_and_eval",
        "authority": _AUTHORITY,
        "promotion_claim": False,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ckpt_dir": str(ckpt_dir),
        "npz_name": cfg["npz_name"],
        "n_pairs_total": n_pairs,
        "self_orient": so,
        "config": {k: cfg.get(k) for k in (
            "n_classes", "hidden_dim", "n_hidden", "mod_dim", "activation", "softmax_temp",
            "chroma", "render_h", "render_w", "in_feat", "max_bank_freq", "lane_edge_weight")},
        "byte_close": {
            **breakdown,
            "archive_zip_bytes": zip_bytes,
            "zip_container_overhead_bytes": zip_bytes - breakdown["total_0bin_bytes"],
            "rate": rate, "rate_term": rate_term, "rate_denom_bytes": int(RATE_DENOM),
            "archive_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
            "free_vs_counted": {
                "FREE_rule118": "curvelet bank (5 scalars) + self-orient directional feats "
                                "(decoder-own-argmax fixed point) -- generic algorithm, 0 bytes",
                "COUNTED": "in_proj/film/hidden.*/out_sdf/out_tex/palette weights + per-frame code "
                           "(int8+brotli, the learned video-derived payload)",
            },
            "pose_sidecar": pose_note,
            "lane_render_band": lane_report,
        },
        "lane_render_band": lane_report,
        "inflate": inflate_info,
        "bit_exact_roundtrip_gate": bit_exact,
        "parity_on_inflated_frames": parity,
        "exact_eval_upstream_evaluate": exact_eval,
        "contest_cpu_eval_cmd": contest_cmd,
        "packet_dir": str(packet_dir),
        "self_orient_parity_caveat": (
            "self-orient deploy dir feats are a FIXED-POINT on the FINAL weights; the trainer's "
            "implied_S used dir feats accumulated along the training trajectory -> close but not "
            "bit-identical; realized d_seg on the inflated frames is the TRUTH."
            if so["self_orient"] else "n/a (no self-orient)"),
        "contest_ready_full_600": bool(n_pairs == 600),
        "contest_ready_note": (
            "n_pairs==600 -> full 1200-frame .raw, contest-ready"
            if n_pairs == 600 else
            f"n_pairs={n_pairs} != 600 -> inflate emits {2 * n_pairs} frames; a 600-pair witness is "
            "required for the 1200-frame contest .raw (this checkpoint is test-only)"),
    }
    if not keep_packet:
        import shutil
        shutil.rmtree(packet_dir, ignore_errors=True)  # disk-hygiene: the .raw is GBs (certify: rebuildable from archive.zip + inflate.py)
        report["packet_dir"] = "(deleted; pass --keep-packet to retain archive+inflate for the exact-eval row)"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="level-set run dir with levelset_witness_{ema,live}_mlx.npz")
    ap.add_argument("--npz-name", type=str, default=None,
                    help="explicit npz filename (default: prefer *_ema_mlx.npz then *_live_mlx.npz)")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap inflate+parity pairs for SPEED (default: all). Archive always encodes all codes.")
    ap.add_argument("--gt-cache", type=str, default=None, help="shared GT npz for parity (e.g. gt_n6.npz)")
    ap.add_argument("--skip-parity", action="store_true", help="byte-close + inflate only (no GT decode)")
    ap.add_argument("--keep-packet", action="store_true", help="retain the packet dir for the exact-eval row")
    ap.add_argument("--fold-pose-sidecar", action="store_true",
                    help="append a COUNTED stored-pose section; requires --pose-sidecar-path. OFF by "
                         "default. The inflate render does NOT read it -> does NOT lower realized d_pose.")
    ap.add_argument("--pose-sidecar-path", type=Path, default=None,
                    help="prebuilt posenet_targets.bin (tac.scorer_targets.extract_and_save)")
    # #224 Wave E: decode-consistent analytic-lane RENDER-BAND (fork B). OFF by default -> byte-identical.
    # Fits per-pair lane manifold coords from --gt-cache (COUNTED), reproduces the coverage+composite
    # in inflate (FREE, rule 118) so the band is a REAL (non-phantom) score. Closes R5_BLOCK.
    ap.add_argument("--lane-render-band", action="store_true",
                    help="composite the decode-consistent analytic-lane render-band (needs --gt-cache). "
                         "COUNTS the per-pair lane coords in archive.zip; reproduces the coverage raster "
                         "FREE in inflate.py. OFF by default (archive byte-identical when off).")
    ap.add_argument("--lane-band-softness", type=float, default=1.0, help="AA-SDF coverage edge softness (px).")
    ap.add_argument("--lane-band-dash-forward-max", type=float, default=DEFAULT_DASH_FORWARD_MAX_M,
                    help="range-dependent dash-gate horizon (m); beyond it dashes read continuous (#215).")
    ap.add_argument("--lane-band-weight", type=float, default=1.0, help="global band strength in [0,1].")
    ap.add_argument("--lane-band-lane-cls", type=int, default=1, help="lane class index (comma10k canonical=1).")
    ap.add_argument("--lane-band-umask", action="store_true",
                    help="gate the band by the WITNESS's own softmax margin (top1-top2) -- the "
                         "decode-consistent FP killer (c_full_wit). OFF -> coverage-only band (c_range).")
    ap.add_argument("--lane-band-tau", type=float, default=0.85, help="witness-margin uncertainty threshold (prob).")
    ap.add_argument("--lane-band-eps", type=float, default=0.35, help="witness-margin uncertainty ramp.")
    ap.add_argument("--lane-band-naive", action="store_true",
                    help="Wave-F: use the NAIVE LBND1 float64 serializer instead of the OPTIMAL LBND2 RD "
                         "codec (quantize+temporal-delta+zigzag). Default = LBND2 RD (rate-viable). The "
                         "naive path is kept for the naive-vs-RD rate comparison + the default-off gate.")
    # self-orient params the trainer does NOT persist (a trainer gap, flagged) -> trainer defaults.
    ap.add_argument("--so-freq-across", type=float, default=32.0, help="self-orient HIGH freq across the edge (trainer default 32).")
    ap.add_argument("--so-freq-along", type=float, default=4.0, help="self-orient LOW freq along the edge (trainer default 4).")
    ap.add_argument("--so-tau", type=float, default=4.0, help="self-orient boundary-proximity tau (trainer default 4).")
    ap.add_argument("--so-iters", type=int, default=4, help="self-orient fixed-point iterations at decode (convergence).")
    # BIT-EXACT round-trip gate (the correctness proof: shipped inflate == numpy-fp32 oracle).
    ap.add_argument("--verify-bit-exact", action="store_true",
                    help="prove inflate(archive) == numpy-fp32 oracle forward, bit-for-bit on the "
                         ".raw frames (the correctness gate). Runs BEFORE the (long) full inflate.")
    ap.add_argument("--bit-exact-pairs", type=int, default=2,
                    help="pairs to compare in the bit-exact gate (per-pixel; 2 proves the forward).")
    ap.add_argument("--no-bit-exact-strict", action="store_true",
                    help="warn instead of raise on a bit-exact mismatch (default: strict/raise).")
    # REAL upstream/evaluate.py exact-eval row (CPU; NEVER MPS). Forces --keep-packet + all pairs.
    ap.add_argument("--run-exact-eval", action="store_true",
                    help="after inflating ALL pairs, run upstream/evaluate.py on the exact archive "
                         "bytes -> real d_seg/d_pose/rate/S. CPU 600-pair ~1-2h. Forces --keep-packet.")
    ap.add_argument("--eval-device", type=str, default=None, choices=["cpu", "cuda"],
                    help="evaluate.py device for --run-exact-eval (default: driven by --memory-tier — "
                         "cpu for decode_cpu_16gb, cuda for decode_t4_16gb; MPS is REFUSED).")
    ap.add_argument("--memory-tier", type=str, default=DEFAULT_TIER_NAME,
                    choices=sorted(DECODE_MEMORY_TIERS),
                    help="#224 Wave D decode (inflate) memory-tier: decode_cpu_16gb (DEFAULT, the "
                         "proven contest CPU path) / decode_t4_16gb (contest T4 CUDA host; SAME "
                         "bit-exact inflate) / production_edge (edge runtime, REFUSED here). Contest "
                         "tiers are bit-exact (fp64/CPU-torch-R/1-thread BLAS); a fp32/CUDA/multithread "
                         "tier is FORBIDDEN on contest tiers.")
    ap.add_argument("--uncompressed-dir", type=Path, default=None,
                    help="GT videos dir for evaluate.py (default upstream/videos; the rate denominator).")
    ap.add_argument("--video-names-file", type=Path, default=None,
                    help="video-names file for evaluate.py (default upstream/public_test_video_names.txt).")
    ap.add_argument("--eval-timeout", type=int, default=18000,
                    help="upstream/evaluate.py timeout seconds (default 5h for a CPU 600-pair run).")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = ap.parse_args(argv)

    # #224 Wave D: resolve + validate the decode memory-tier BEFORE anything. Fail-closed: the
    # contest byte-close/eval path REFUSES production_edge (its relaxed fp32/CUDA/multithread numeric
    # path can flip uint8 boundaries and is DEFERRED #228). Apply the tier's inflate env (1-thread
    # BLAS contract + optional worker cap) so the inflate subprocesses inherit it; drive the downstream
    # evaluate.py device from the tier (explicit --eval-device wins).
    tier = require_contest_tier(resolve_tier(args.memory_tier))  # raises DecodeTierError if refused
    _tier_env = tier_inflate_env(tier)
    for _k, _v in _tier_env.items():
        os.environ[_k] = _v
    eval_device = resolve_eval_device(tier, args.eval_device)
    print(f"# decode memory-tier: {tier.name} (contest={tier.contest}, bit_exact={tier.bit_exact_contract}, "
          f"eval_device={eval_device}); inflate env={_tier_env or '{} (inflate defaults)'}", flush=True)

    report = run(
        args.ckpt_dir,
        npz_name=args.npz_name,
        max_pairs=args.max_pairs,
        fold_pose_sidecar=args.fold_pose_sidecar,
        pose_sidecar_path=args.pose_sidecar_path,
        gt_cache=args.gt_cache,
        keep_packet=args.keep_packet,
        packet_dir=None,
        skip_parity=args.skip_parity,
        so_overrides={"freq_across": args.so_freq_across, "freq_along": args.so_freq_along,
                      "tau": args.so_tau, "iters": args.so_iters},
        lane_render_band=args.lane_render_band,
        lane_band_cfg=LaneBandRenderConfig(
            softness=args.lane_band_softness, dash_gate=True,
            dash_forward_max_m=args.lane_band_dash_forward_max, weight=args.lane_band_weight,
            lane_cls=args.lane_band_lane_cls, u_mask_enabled=args.lane_band_umask,
            u_mask_tau=args.lane_band_tau, u_mask_eps=args.lane_band_eps,
        ),
        lane_rd=not args.lane_band_naive,
        verify_bit_exact=args.verify_bit_exact,
        bit_exact_pairs=args.bit_exact_pairs,
        bit_exact_strict=not args.no_bit_exact_strict,
        run_exact_eval=args.run_exact_eval,
        eval_device=eval_device,
        uncompressed_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        eval_timeout=args.eval_timeout,
    )
    # record the decode-tier contract in the report (observability + provenance).
    report["decode_memory_tier"] = {
        "name": tier.name, "contest": tier.contest, "bit_exact_contract": tier.bit_exact_contract,
        "eval_device": eval_device, "inflate_env": _tier_env, "note": tier.note,
    }
    out = args.out or (_REPO / "reports" / f"levelset_byte_close_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {out}  {_AUTHORITY}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
