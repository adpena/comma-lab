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
from tac.boundary_math import warp_real_luma_frame0 as _wrl  # noqa: E402  (#205 pose carrier: warp-real-luma frame0)
from tac.boundary_math import xi_pose_coder as _xip  # noqa: E402  (#257 store-nothing derive-H + ξ entropy coder)

# canonical FREE-table regen fns (rule-118 curvelet bank + self-orient dir feats) — the bit-exact
# oracle reference reuses these so the gate compares against the SAME free tables the inflate uses.
_canon_coords_grid = _tli.coords_grid
_canon_curvelet_B = _tli.curvelet_B
_canon_curvelet_feats = _tli.curvelet_feats
_canon_dir_feats = _tli.dir_feats

CAMERA_H, CAMERA_W = 874, 1164
RATE_DENOM = 37_545_489.0
_MAGIC = b"LVLS1\x00"  # level-set softmax-of-SDF carrier v1
_PCAR_MAGIC = b"PCAR1\x00"  # #205 pose carrier: warp-real-luma frame0 (stored keyframe luma + per-pair homography)
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
    pose_carrier_bytes: bytes | None = None, pose_carrier_manifest: dict[str, Any] | None = None,
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
    # #205 warp-real-luma pose carrier (6th block). Manifest flag gates the READ (so the reader
    # knows to expect the trailing block); default-off -> byte-identical to the pre-#205 grammar.
    if pose_carrier_bytes is not None and pose_carrier_manifest is not None:
        manifest["pose_carrier"] = pose_carrier_manifest
    mj = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    out = _io_pack(mj, base_brotli, code_brotli, pose_sidecar, lane_band_bytes, pose_carrier_bytes)
    # cross-check our accounting against the canonical quantize_levelset_blob (same int8 grammar).
    canon = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in params.items()})
    breakdown = {
        "n_params": int(sum(int(np.prod(s)) for s in shapes.values())) + int(np.prod(code.shape)),
        "manifest_bytes": len(mj),
        "base_int8_brotli_bytes": len(base_brotli),
        "code_int8_brotli_bytes": len(code_brotli),
        "pose_sidecar_bytes": (len(pose_sidecar) if pose_sidecar else 0),
        "lane_band_counted_bytes": (len(lane_band_bytes) if lane_band_bytes else 0),
        "pose_carrier_counted_bytes": (len(pose_carrier_bytes) if pose_carrier_bytes else 0),
        "magic_and_prefixes_bytes": (len(_MAGIC) + 16 + (4 if lane_band_bytes is not None else 0)
                                     + (4 if pose_carrier_bytes is not None else 0)),
        "total_0bin_bytes": len(out),
        "canonical_quantize_blob_bytes": int(canon["total_quantized_blob_bytes"]),
        "accounting_matches_canonical": bool(
            len(base_brotli) == canon["base_int8_brotli_bytes"]
            and len(code_brotli) == canon["code_int8_brotli_bytes"]),
    }
    return out, breakdown


def _io_pack(
    manifest: bytes, base: bytes, code: bytes, pose: bytes | None,
    lane_band: bytes | None = None, pose_carrier: bytes | None = None,
) -> bytes:
    """Pack the LVLS1 blob. The 5th ``lane_band`` block (#224 Wave E) and the 6th
    ``pose_carrier`` block (#205 warp-real-luma frame0) are OPTIONAL and appended, in that order,
    ONLY when non-None -> absent both, the output is BYTE-IDENTICAL to the pre-Wave-E 4-block
    grammar (the default-off guarantee). Trailing blocks are gated at READ time by the manifest
    flags (``lane_render_band`` / ``pose_carrier``), so the reader knows how many trailing blocks
    to expect -- NEVER by a bare ``off < len(raw)`` (which would misread a lone pose_carrier block
    as lane). ``lane_band`` = the COUNTED lane manifold coords; ``pose_carrier`` = the COUNTED
    real-luma keyframe payload + per-pair homography (rule 118: keyframe COUNTED, warp decoder FREE)."""

    buf = bytearray()
    buf += _MAGIC
    for chunk in (manifest, base, code, (pose or b"")):
        buf += struct.pack("<I", len(chunk))
        buf += chunk
    for opt in (lane_band, pose_carrier):
        if opt is not None:
            buf += struct.pack("<I", len(opt))
            buf += opt
    return bytes(buf)


# ---------------------------------------------------------------------------
# #205 POSE CARRIER (warp-real-luma frame0): serialize / warp / parse.
#   The scorer reads d_pose = MSE(PoseNet(gen_pair)[:6], PoseNet(gt_pair)[:6]) ON THE FRAMES.
#   A dead pose SIDECAR (block 3) is bytes the render never reads -> does NOT lower realized d_pose.
#   The POSE CARRIER instead REPLACES the render's FRAME0 (SEG-free: SegNet reads only frame1,
#   upstream/modules.py:108) with a STORED REAL keyframe SE(3)-warped by the stored per-pair ego
#   twist -> PoseNet reads real warped motion and the pose becomes measurable through the FRAMES
#   (the ``tac.boundary_math.warp_real_luma_frame0`` semantics). Rule-118 boundary:
#     * COUNTED (archive.zip): the stored real keyframe luma (video-derived) + per-pair homography.
#     * FREE (inflate.py): the inverse-warp bilinear + R (generic algorithm). The dual-use twist xi
#       is stored (fp16) for provenance; the per-pair H (fp64) is what the decode uses (bit-exact,
#       no exp_se3 needed in the decode path -> the shipped inflate == numpy authority by
#       construction). H is a deterministic function of xi (byte-optimal design stores ONLY xi and
#       derives H FREE; H stored here is the decode-simplicity choice, ~72 B/pair, negligible vs
#       the keyframe payload).
# ---------------------------------------------------------------------------
def _pcar_warp_frame0_from_H(src_native: np.ndarray, H: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    """Inverse-warp a native ``(H,W,3)`` real-luma frame by the fp64 homography ``H`` -> uint8.

    OP-FOR-OP identical to ``tac.boundary_math.warp_real_luma_frame0.warp_frame0_native_numpy`` FROM
    ``Hinv = inv(H)`` ONWARD (bilinear inverse-sample + persist fallback + round/clip), so that with
    ``H = homography_from_xi_numpy(xi, geom)`` (fp64) this returns the module authority
    ``warp_frame0_uint8_numpy(src, xi, geom)`` BIT-FOR-BIT (proven 0-diff, tests). The shipped
    inflate inlines a VERBATIM copy of this body -> shipped == this oracle by construction."""
    src = np.asarray(src_native, dtype=np.float64)
    Hh, Ww = int(native_hw[0]), int(native_hw[1])
    if src.shape[:2] != (Hh, Ww):
        raise ValueError(f"pose-carrier keyframe native {src.shape[:2]} != {(Hh, Ww)} (NO-FAKE)")
    C = src.shape[2]
    flat = src.reshape(-1, C)
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    grid = np.stack([us.ravel(), vs.ravel(), np.ones(Hh * Ww)], 0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(np.asarray(H, np.float64))
        src_h = Hinv @ grid
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0)
             & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1))
    su_c = np.clip(su, 0.0, Ww - 1)
    sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64)
    y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1)
    y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]
    wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]
    Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]
    Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat).reshape(Hh, Ww, C)
    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)


def serialize_pose_carrier(
    H_stack: np.ndarray, xi_stack: np.ndarray, keyframes: list[np.ndarray],
    kf_of_pair: list[int], hdr_extra: dict[str, Any],
) -> bytes:
    """Serialize the pose-carrier section (NOT re-brotli'd; keyframes are brotli'd individually).

    Layout: PCAR1 | u32 hdr_len | hdr_json | H(P*9 fp64) | xi(P*6 fp16) | u32 n_kf |
            [u32 blen | brotli(uint8 (kf_h,kf_w,3))]*n_kf.
    ``kf_of_pair[p]`` indexes into ``keyframes``; H/xi are per-pair (P rows)."""
    import brotli

    P = int(H_stack.shape[0])
    hdr = {
        "n_pairs": P,
        "native_h": CAMERA_H, "native_w": CAMERA_W,
        "kf_of_pair": [int(k) for k in kf_of_pair],
        "n_keyframes": len(keyframes),
        **hdr_extra,
    }
    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    buf = bytearray()
    buf += _PCAR_MAGIC
    buf += struct.pack("<I", len(hj)); buf += hj
    buf += np.asarray(H_stack, dtype=np.float64).tobytes()
    buf += np.asarray(xi_stack, dtype=np.float16).tobytes()
    buf += struct.pack("<I", len(keyframes))
    for kf in keyframes:
        kb = brotli.compress(np.ascontiguousarray(kf, dtype=np.uint8).tobytes(), quality=11)
        buf += struct.pack("<I", len(kb)); buf += kb
    return bytes(buf)


def serialize_pose_carrier_store_nothing(
    xi_stack: np.ndarray, hdr_extra: dict[str, Any], *, coder: str = "delta_ar", q_levels: int = 4096,
) -> tuple[bytes, dict[str, Any]]:
    """#257 STORE-NOTHING v2 serializer -- stores ONLY the (quantized, coded) per-pair ego twist ξ.

    The redundant fp64 H (43,200 B/600 pairs = 83% of the old section, FINDING-1) is DROPPED and
    DERIVED FREE at decode (``_xip.homographies_from_xi``, rule-118). The useless
    ``kf_of_pair=[0]*P`` header list is also DROPPED (unused when n_keyframes=0). Layout:
        PCAR1\\x00 | u32 hdr_len | hdr_json | u32 xi_payload_len | xi_payload | u32 n_kf(=0)
    ``xi_payload`` (``tac.boundary_math.xi_pose_coder``) carries the per-channel scales + q
    (``coder='none'`` raw int16 ~0.005 rate, or ``coder='delta_ar'`` temporal-Δ + arithmetic ~0.002).
    Returns (blob, quant_report). The shipped inflate inlines the inverse (ξ decode + derive-H)."""
    q, scales = _xip.quantize_xi(np.asarray(xi_stack, dtype=np.float64), q_levels=int(q_levels))
    xi_payload = _xip.serialize_xi_payload(q, scales, coder=coder)
    P = int(q.shape[0])
    hdr = {
        "n_pairs": P,
        "native_h": CAMERA_H, "native_w": CAMERA_W,
        "n_keyframes": 0,
        "pcar_store_nothing_v": 2,          # discriminator: v2 = derive-H, NO stored H block
        "xi_coder": str(coder),
        "xi_q_levels": int(q_levels),
        **hdr_extra,
    }
    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    buf = bytearray()
    buf += _PCAR_MAGIC
    buf += struct.pack("<I", len(hj)); buf += hj
    buf += struct.pack("<I", len(xi_payload)); buf += xi_payload
    buf += struct.pack("<I", 0)              # n_kf = 0 (store-nothing: NO keyframe luma)
    quant_report = {
        "xi_coder": str(coder), "xi_q_levels": int(q_levels),
        "xi_payload_bytes": len(xi_payload),
        "xi_raw_bytes_ref": len(_xip.serialize_xi_payload(q, scales, coder="none")),
    }
    return bytes(buf), quant_report


def parse_pose_carrier(blob: bytes) -> dict[str, Any]:
    """Inverse of ``serialize_pose_carrier`` / ``serialize_pose_carrier_store_nothing`` (tool-side;
    the shipped inflate inlines the same). Discriminates on the header ``pcar_store_nothing_v``:
    v2 store-nothing DERIVES H from the coded ξ (no stored H block); everything else reads the
    legacy fp64-H block (warp_real_luma stays byte-identical)."""
    import brotli

    assert blob[: len(_PCAR_MAGIC)] == _PCAR_MAGIC, "bad pose-carrier magic"
    off = len(_PCAR_MAGIC)
    (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    hdr = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    P = int(hdr["n_pairs"])
    if int(hdr.get("pcar_store_nothing_v", 0)) == 2:
        # #257 v2: ξ payload -> dequant ξ -> DERIVE H (FREE, rule-118). No stored H, no kf_of_pair.
        (xlen,) = struct.unpack_from("<I", blob, off); off += 4
        xi_dq = _xip.decode_xi_payload(blob[off:off + xlen]); off += xlen
        (n_kf,) = struct.unpack_from("<I", blob, off); off += 4
        if n_kf != 0:
            raise ValueError(f"store-nothing v2 must have n_kf=0; got {n_kf} (NO-FAKE)")
        H = _xip.homographies_from_xi(xi_dq, float(hdr["pitch"]))
        return {"hdr": hdr, "H": H, "xi": xi_dq, "keyframes": [], "kf_of_pair": [0] * P}
    # legacy: stored fp64 H block (warp_real_luma; and any pre-#257 store-nothing archive).
    H = np.frombuffer(blob[off:off + P * 9 * 8], dtype=np.float64).reshape(P, 3, 3).copy(); off += P * 9 * 8
    xi = np.frombuffer(blob[off:off + P * 6 * 2], dtype=np.float16).reshape(P, 6).astype(np.float64); off += P * 6 * 2
    (n_kf,) = struct.unpack_from("<I", blob, off); off += 4
    kh, kw = int(hdr["kf_store_h"]), int(hdr["kf_store_w"])
    keyframes = []
    for _ in range(n_kf):
        (blen,) = struct.unpack_from("<I", blob, off); off += 4
        raw = brotli.decompress(blob[off:off + blen]); off += blen
        keyframes.append(np.frombuffer(raw, dtype=np.uint8).reshape(kh, kw, 3).copy())
    return {"hdr": hdr, "H": H, "xi": xi, "keyframes": keyframes, "kf_of_pair": hdr["kf_of_pair"]}


def _pcar_upsample_to_native(kf: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear-upsample a stored keyframe (kh,kw,3) uint8 -> native fp64. Deterministic torch CPU
    (bit-identical run-to-run + between inflate/oracle). Identity when already native."""
    import torch

    Hh, Ww = int(native_hw[0]), int(native_hw[1])
    if kf.shape[:2] == (Hh, Ww):
        return np.asarray(kf, dtype=np.float64)
    x = torch.from_numpy(np.ascontiguousarray(kf)).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(Hh, Ww), mode="bilinear", align_corners=False)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.float64)


def pose_carrier_frame0(pc: dict[str, Any], pi: int) -> np.ndarray:
    """Decode frame0 for pair ``pi`` = warp(stored keyframe, stored H[pi]) at native res -> uint8.
    The tool-side authority (warp_real_luma mode); the shipped inflate inlines a verbatim copy."""
    native_hw = (int(pc["hdr"]["native_h"]), int(pc["hdr"]["native_w"]))
    kf = pc["keyframes"][int(pc["kf_of_pair"][pi])]
    src = _pcar_upsample_to_native(kf, native_hw)
    return _pcar_warp_frame0_from_H(src, pc["H"][pi], native_hw)


def pose_carrier_mode(pc: dict[str, Any]) -> str:
    """The pose-carrier decode mode from the PCAR hdr. ``warp_real_luma`` (default; stored real
    keyframe) or ``store_nothing`` (store ONLY xi/H; frame0 = warp(GENERATED witness render, H))."""
    return str(pc["hdr"].get("pose_carrier_mode", "warp_real_luma"))


def pose_carrier_frame0_from_source(pc: dict[str, Any], pi: int, src_native: np.ndarray) -> np.ndarray:
    """STORE-NOTHING frame0 = warp(a GENERATED camera-native ``src_native``, stored H[pi]) -> uint8.

    ``src_native`` is the witness's OWN frame0 RGB render at camera resolution (NOT a stored real
    keyframe -- store_nothing stores ONLY xi/H, ~0 marginal bytes). Op-for-op the SAME warp as
    ``_pcar_warp_frame0_from_H`` so the shipped inflate's inlined store-nothing warp == this oracle
    bit-for-bit (the general bit-exact gate proves it over frame0 too)."""
    native_hw = (int(pc["hdr"]["native_h"]), int(pc["hdr"]["native_w"]))
    return _pcar_warp_frame0_from_H(np.asarray(src_native, dtype=np.float64), pc["H"][pi], native_hw)


def _downscale_keyframe(gt_f0_native: np.ndarray, store_hw: tuple[int, int]) -> np.ndarray:
    """Store-res keyframe: bilinear-downscale native gt_f0 -> (store_h,store_w,3) uint8 (torch CPU,
    deterministic). Identity when store_hw == native. Round-trips through the decode-time upsample."""
    import torch

    sh, sw = int(store_hw[0]), int(store_hw[1])
    if (sh, sw) == (gt_f0_native.shape[0], gt_f0_native.shape[1]):
        return np.ascontiguousarray(gt_f0_native, dtype=np.uint8)
    x = torch.from_numpy(np.ascontiguousarray(gt_f0_native)).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        dn = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
        dn = torch.clamp(torch.round(dn), 0.0, 255.0)
    return dn[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def build_pose_carrier_section(
    gt_cache: str | None, n_pairs: int, pc: dict[str, Any],
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Build the #205 pose-carrier section from the GT cache (compress-time: the ego pose
    ``gt_poses`` [+ real keyframe luma ``gt_f0`` for warp_real_luma] are fully available). Returns
    (pose_carrier_bytes, manifest_cfg, report). NO-FAKE: missing cache raises.

    Two selectable modes (``pc["mode"]``, A/B-able against each other at #205; NEITHER removed):

    * ``warp_real_luma`` (default) -- store the real keyframe luma; frame0 = warp(stored keyframe,
      per-pair H). Keyframes stored at ``store_hw`` (native=lossless; downscaled=cheap-rate lever).
      Rate = keyframe brotli + H(72 B/pair) + xi(12 B/pair), COUNTED in archive.zip.
    * ``store_nothing`` (Track B ``18927a1ae`` / ``keyframe_rate_minimization_builds``) -- store ONLY
      xi (+ H for decode simplicity), NO real keyframe. frame0 = warp(the witness's OWN frame0 RGB
      render [generated FREE by the INR from the counted weights+code -- rule-118], per-pair H). The
      keyframe payload collapses to ~0 marginal bytes (rate ~= xi/H only). MEASURED n600 (Track B,
      classmean proxy): d_pose 4.97 pre-residual (witness render is richer -> <= 4.97); the trained
      rank-6 dxi residual (#205, w_pose>0) closes the fixed offset toward ~3.4e-5 (UNMEASURED here).

    For each pair p: the per-pair ego twist ``xi[p] = xi_from_pose_calibration(gt_poses[p], s_t,
    s_r, pitch)``; ``H[p] = homography_from_xi_numpy(xi[p], geom)`` (fp64, the exact warp). ``stride``
    keyframes (warp_real_luma only): pair p uses keyframe ``(p//stride)*stride`` (stride=1 -> per-pair
    exact upper bound; stride>1 -> shared, geometrically-approximate, LOUDLY flagged)."""
    mode = str(pc.get("mode", "warp_real_luma"))
    if mode not in ("warp_real_luma", "store_nothing"):
        raise ValueError(f"pose-carrier mode {mode!r} unknown (warp_real_luma|store_nothing).")
    store_nothing = mode == "store_nothing"
    if not gt_cache:
        raise ValueError("--pose-carrier requires --gt-cache (gt_poses [+ gt_f0 for warp_real_luma]). "
                         "NO-FAKE: refusing to fabricate the pose payload.")
    cp = Path(gt_cache)
    if not cp.exists():
        raise FileNotFoundError(f"--gt-cache {cp} not found (pose-carrier needs gt_poses [+ gt_f0]).")
    z = np.load(cp, allow_pickle=False)
    # store_nothing needs ONLY the ego poses (no real keyframe luma -> the pure "store nothing but xi").
    required = ("gt_poses",) if store_nothing else ("gt_f0", "gt_poses")
    for req in required:
        if req not in z.files:
            raise ValueError(f"gt cache {cp} lacks {req!r} (pose-carrier mode={mode}).")
    gt_poses = np.asarray(z["gt_poses"], dtype=np.float64)
    P = int(min(int(n_pairs), int(gt_poses.shape[0])))
    if not store_nothing:
        gt_f0 = z["gt_f0"]
        P = min(P, int(gt_f0.shape[0]))
    s_t, s_r, pitch = float(pc["s_t"]), float(pc["s_r"]), float(pc["pitch"])
    stride = max(1, int(pc["stride"]))
    ds = max(1, int(pc.get("downscale", 1)))
    xi_coder = str(pc.get("xi_coder", "delta_ar"))     # #257: 'delta_ar' (default) | 'none' (raw fallback)
    xi_q_levels = int(pc.get("xi_q_levels", 4096))     # #257: ξ quantization precision (store_nothing only)
    store_hw = (CAMERA_H, CAMERA_W) if (store_nothing or ds == 1) else (CAMERA_H // ds, CAMERA_W // ds)
    geom = _wrl.GroundHomographyGeom.eon(pitch=pitch)

    # per-pair ego twist ξ (fp64 authority) -- both modes. H is DERIVED FREE at decode for
    # store_nothing (#257), STORED only for warp_real_luma (the legacy stored-keyframe path).
    xi_stack = np.zeros((P, 6), dtype=np.float64)
    for p in range(P):
        xi_stack[p] = _wrl.xi_from_pose_calibration(gt_poses[p], s_t, s_r, pitch)

    quant_report: dict[str, Any] = {}
    if store_nothing:
        # #257 STORE-NOTHING v2: store ONLY the (quantized+coded) ξ. NO stored H (43,200 B redundancy
        # DROPPED -> derived FREE at decode), NO kf_of_pair junk list. frame0 = warp(witness's OWN
        # frame0 render, DERIVED H). The whole keyframe payload is GONE (rule-118: render is FREE).
        keyframes: list[np.ndarray] = []
        hdr_extra = {
            "pose_carrier_mode": mode,
            "s_t": s_t, "s_r": s_r, "pitch": pitch, "stride": stride,
            "kf_store_h": int(store_hw[0]), "kf_store_w": int(store_hw[1]),
            "keyframe_lossless_native": True,
            "generator": "witness_render_frame0",
            "calibration_note": ("STORE-NOTHING v2 (#257): store ONLY the per-pair ego twist ξ "
                                 "(quantized+coded); H = homographies_from_xi(ξ, pitch) DERIVED FREE at "
                                 "decode (rule-118). frame0 = warp(witness's OWN frame0 INR render, H)."),
        }
        blob, quant_report = serialize_pose_carrier_store_nothing(
            xi_stack, hdr_extra, coder=xi_coder, q_levels=xi_q_levels)
        kf_of_pair: list[int] = [0] * P
    else:
        # warp_real_luma (legacy): store fp64 H + keyframe luma (BYTE-IDENTICAL to pre-#257).
        H_stack = np.zeros((P, 3, 3), dtype=np.float64)
        for p in range(P):
            H_stack[p] = _wrl.homography_from_xi_numpy(xi_stack[p], geom)
        kf_indices = sorted({min(P - 1, (p // stride) * stride) for p in range(P)})
        kf_pos = {k: i for i, k in enumerate(kf_indices)}
        kf_of_pair = [kf_pos[min(P - 1, (p // stride) * stride)] for p in range(P)]
        keyframes = [_downscale_keyframe(np.asarray(gt_f0[k]), store_hw) for k in kf_indices]
        hdr_extra = {
            "pose_carrier_mode": mode,
            "s_t": s_t, "s_r": s_r, "pitch": pitch, "stride": stride,
            "kf_store_h": int(store_hw[0]), "kf_store_w": int(store_hw[1]),
            "keyframe_lossless_native": bool(ds == 1),
            "generator": "stored_real_keyframe",
            "calibration_note": ("per-pair xi = xi_from_pose_calibration(gt_poses, s_t, s_r, pitch); "
                                 "H = homography_from_xi_numpy(xi). warp-real-luma frame0 semantics "
                                 "(tac.boundary_math.warp_real_luma_frame0)."),
        }
        blob = serialize_pose_carrier(H_stack, xi_stack, keyframes, kf_of_pair, hdr_extra)

    kf_bytes_total = sum(int(len(s)) for s in _pcar_keyframe_blob_sizes(keyframes))  # 0 for store_nothing
    manifest = {
        "mode": mode,
        "s_t": s_t, "s_r": s_r, "pitch": pitch, "stride": stride,
        "kf_store_h": int(store_hw[0]), "kf_store_w": int(store_hw[1]),
        "n_keyframes": len(keyframes), "n_pairs": P,
        "keyframe_lossless_native": bool(store_nothing or ds == 1),
    }
    report = {
        "active": True,
        "mode": mode,
        "generator": ("witness_render_frame0" if store_nothing else "stored_real_keyframe"),
        "source_gt_cache": str(cp),
        "n_pairs": P,
        "n_keyframes": len(keyframes),
        "stride": stride,
        "keyframe_store_hw": [int(store_hw[0]), int(store_hw[1])],
        "keyframe_lossless_native": bool(store_nothing or ds == 1),
        "pose_carrier_section_bytes": len(blob),
        "keyframe_blob_bytes_total": kf_bytes_total,
        # #257: store_nothing v2 stores NO H (derived FREE) -> H_bytes=0; xi_bytes = the coded/raw
        # ξ payload actually stored. warp_real_luma keeps the legacy stored fp64-H (72 B/pair).
        "H_bytes": 0 if store_nothing else int(P * 9 * 8),
        "xi_bytes": int(quant_report.get("xi_payload_bytes", P * 6 * 2)) if store_nothing else int(P * 6 * 2),
        "xi_coder": quant_report.get("xi_coder") if store_nothing else None,
        "xi_q_levels": quant_report.get("xi_q_levels") if store_nothing else None,
        "xi_raw_bytes_ref": quant_report.get("xi_raw_bytes_ref") if store_nothing else None,
        "calibration": {"s_t": s_t, "s_r": s_r, "pitch": pitch},
        "stride_note": (
            "store_nothing v2 (#257): no stored keyframe AND no stored H -> frame0 = warp(witness's OWN "
            "frame0 render, DERIVED per-pair H); stride unused (only the coded ξ is stored)."
            if store_nothing else
            ("stride>1 shares one stored keyframe across a window but warps each pair by ITS OWN intra-pair "
             "xi (no kf->pair displacement) -> geometrically approximate for shared keyframes; stride=1 "
             "(per-pair) is the EXACT warp-real-luma upper bound." if stride > 1 else
             "stride=1: per-pair keyframe, exact warp-real-luma.")),
        "rule_118": {
            "COUNTED (archive.zip)": (
                "ONLY the per-pair ego twist ξ (quantized + entropy-coded) + 6 per-channel scales + the "
                "scalar pitch; NO stored H (43,200 B redundancy DROPPED -> derived FREE), NO keyframe luma"
                if store_nothing else
                "stored real keyframe luma (video-derived) + per-pair homography H; warped at decode by "
                "the per-pair ego twist to synthesize frame0"),
            "FREE (inflate.py)": (
                "the witness INR frame0 render + exp_se3 + H=K(R-t nT/d)K^-1 (DERIVE-H) + the ξ arithmetic "
                "decoder + inverse-warp bilinear + R -- all generic algorithm, 0 archive bytes (rule-118)"
                if store_nothing else
                "inverse-warp bilinear + R (generic algorithm); the dual-use twist xi is the pose sidecar "
                "(~0 marginal bytes in the byte-optimal design)"),
            "no_gt_no_scorer": "no GT mask, no SegNet/PoseNet weights, no per-pixel table ship",
        },
        "byte_optimal_note": (
            "STORE-NOTHING v2 (#257): stores ONLY the coded ξ ({} B, coder={}, q_levels={}; raw-ξ ref {} B). "
            "The redundant per-pair fp64 H (43,200 B = 83% of the pre-#257 section, FINDING-1) is DROPPED "
            "and DERIVED FREE at decode (exp_se3 + plane homography, rule-118); the kf_of_pair junk list "
            "is DROPPED. ZERO keyframe bytes. d_pose is measured through the REAL byte-closed decode (the "
            "derived H reproduces the fp64-H warp; the trained rank-6 dξ residual (#205 w_pose>0) closes "
            "d_pose -- UNMEASURED here; NO borrowed number).".format(
                quant_report.get("xi_payload_bytes", "?"), quant_report.get("xi_coder", "?"),
                quant_report.get("xi_q_levels", "?"), quant_report.get("xi_raw_bytes_ref", "?"))
            if store_nothing else
            "H (fp64, 72 B/pair) is stored for BIT-EXACT decode simplicity (no exp_se3 in the decode path); "
            "the byte-optimal design stores ONLY xi (fp16, 12 B/pair, dual-use with the pose sidecar) and "
            "derives H FREE via exp_se3. Keyframe SPARSITY (stride>1 with kf->pair relative warp) + lossy "
            "keyframe codec are the OPEN rate levers -- NOT applied/borrowed here (measured rate is of what "
            "is ACTUALLY stored)."),
    }
    return blob, manifest, report


def _pcar_keyframe_blob_sizes(keyframes: list[np.ndarray]) -> list[bytes]:
    """Recompute the per-keyframe brotli blobs (for the byte report). Deterministic == the serializer."""
    import brotli

    return [brotli.compress(np.ascontiguousarray(kf, dtype=np.uint8).tobytes(), quality=11) for kf in keyframes]


def _cap_pose_carrier(pcar_bytes: bytes, eval_pairs: int) -> bytes:
    """Slice a pose-carrier section to the first ``eval_pairs`` pairs (for the capped/gate inflate):
    keep H/xi[:eval_pairs], prune to the referenced keyframes, remap kf_of_pair. Re-serialized ==
    the full serializer, so the shipped inflate + oracle stay bit-exact on the capped set."""
    pc = parse_pose_carrier(pcar_bytes)
    P = min(int(eval_pairs), int(pc["hdr"]["n_pairs"]))
    hdr = pc["hdr"]
    hdr_extra = {k: hdr[k] for k in ("pose_carrier_mode", "generator", "s_t", "s_r", "pitch", "stride",
                                     "kf_store_h", "kf_store_w", "keyframe_lossless_native",
                                     "calibration_note") if k in hdr}
    if int(hdr.get("pcar_store_nothing_v", 0)) == 2:  # #257 v2: re-serialize the sliced coded ξ (derive-H)
        capped, _qr = serialize_pose_carrier_store_nothing(
            np.asarray(pc["xi"][:P], dtype=np.float64), hdr_extra,
            coder=str(hdr.get("xi_coder", "delta_ar")), q_levels=int(hdr.get("xi_q_levels", 4096)))
        return capped
    if len(pc["keyframes"]) == 0:  # legacy store_nothing: no keyframes; kf_of_pair unused placeholder
        return serialize_pose_carrier(pc["H"][:P], pc["xi"][:P], [], [0] * P, hdr_extra)
    kf_of_pair = [int(k) for k in pc["kf_of_pair"][:P]]
    used = sorted(set(kf_of_pair))
    remap = {old: i for i, old in enumerate(used)}
    keyframes = [pc["keyframes"][old] for old in used]
    kf_of_pair_new = [remap[k] for k in kf_of_pair]
    return serialize_pose_carrier(pc["H"][:P], pc["xi"][:P], keyframes, kf_of_pair_new, hdr_extra)


# ---------------------------------------------------------------------------
# the self-contained inflate.py (numpy fwd + torch R [+ scipy iff self-orient]).
# ---------------------------------------------------------------------------
# REVIEW-GATE: inflate.py exceeds the 100-LOC default budget (~320 LOC) under an explicit <=340-LOC
# waiver: it inlines THREE rule-118 FREE levers the archive-counted statistics require -- (1) the
# SELF-ORIENT fixed-point (curvelet bank regen + decoder-own-argmax tangent + directional feats), the
# byte-closeable directional lever the mod-32 target uses; (2) the #224 Wave E analytic-lane
# RENDER-BAND decode reproduction (_lane_parse + _lane_coverage AA-SDF rasterizer + _lane_composite),
# which EXPANDS the counted per-pair lane manifold coords into the (H,W) coverage + composites the
# band over the render FREE (0 archive bytes); (3) the #257 STORE-NOTHING DERIVE-H + ξ ARITHMETIC
# DECODER (_ar_decode + _xip_parse + _xip_H_from_xi), which EXPANDS the counted per-pair ego twist ξ
# into the per-pair plane-induced homography H (exp_se3 + K(R-t nT/d)K^-1) FREE at decode -- dropping
# the 43,200 B redundant stored fp64 H (FINDING-1). All THREE are op-for-op mirrors of the canonical
# numpy authority (levelset_rgb_forward_numpy / levelset_band_forward_numpy / rasterize_lane_coverage_
# range_dependent / composite_band_on_render / tac.boundary_math.xi_pose_coder / tac.lossless.range_
# coder.decode_static_symbols) and are BIT-EXACT-gate proven vs it.
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
# FEED-ei bit-identical _act fusion (inflate.py code only -> FREE, archive bytes UNCHANGED): the hosc
#   activation tanh(beta*sin(omega*u)) is the #1 hot path (~66% of decode; ~29 acts/pair on the (H*W,
#   hidden) float64 grid = ~151MB/array). (1) SKIP the identity full-array multiplies when omega==1 /
#   beta==1 (1.0*x === x in IEEE754, but numpy still executes the whole ~151MB DRAM round-trip); (2)
#   compute sin -> (beta) -> tanh IN-PLACE in one buffer (the 'pre' input is dead after _act) -> fewer
#   151MB allocs. In-place ufunc == out-of-place ufunc elementwise, so the exact float64 op values are
#   preserved -> BIT-IDENTICAL .raw (proven: cmp -s == identical vs the pre-fusion forward, n40/4w). The
#   DRAM-traffic cut helps MOST under the parallel Pool (workers share memory bandwidth): measured 1.24x
#   4-worker wall-clock (contest-4-core full-600 inflate ~20 min -> ~16 min).
# FEED-ej bit-identical self-orient CONVERGED-h0 reuse (inflate.py code only -> FREE, archive UNCHANGED):
#   the self-orient fixed point (task #281) EARLY-STOPS when the decoder's argmax stops changing (FEED-eg);
#   on the converging iteration the loop's `feats` are already the FINAL feats, so its in_proj activation
#   h0 IS the post-loop h0 -> cache + reuse it instead of recomputing _in_proj_h0 once more. BIT-EXACT: the
#   reuse only fires on the early-break (feats frozen); when the loop runs to completion (no convergence)
#   h0 is recomputed exactly as before. Proven byte-identical vs baseline on the #205 config (n24) AND by a
#   determinism micro-proof on the real weights (dir_feats/in_proj_h0 are pure functions of their inputs).
#   Like the early-stop it is a NO-OP on a mid-training archive (the argmax does not converge even at
#   so_iters=12) and pays only on a well-trained archive whose argmax stabilizes within so_iters.
# FEED-ek OPT-IN fp32 forward (task #281; INFLATE_FP32=1, DEFAULT OFF -> the shipped path stays fp64, the
#   bit-exact cross-host authority). fp32 runs the INR forward in float32: ~1.7-1.9x faster (sin/tanh 3.7x,
#   GEMM 4.3x). It is LOCAL FAST-DECODE ONLY and MUST NOT ship in a contest archive: fp32 last-bit rounding
#   can move an argmax-boundary pixel by up to ~22 uint8 levels, and a different BLAS/SIMD/FMA host produces
#   DIFFERENT boundary flips -> it BREAKS cross-host bit-identity (the deterministic-reproducibility hard
#   limit: same archive.zip -> bit-identical inflate output on EVERY host). Measured score-preservation is
#   only WITHIN ~1e-4 on OUR host (#205 mid-train n24: 100*Δd_seg=-8.5e-5, Δsqrt(10*d_pose)=+1.6e-7); that
#   is same-host-deterministic (run1==run2 byte-identical) but NOT contest-portable. fp64 already meets the
#   30-min budget via the parallel Pool, so the default never needs fp32; the flag exists for dev-loop speed.
import os
for _tv in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")  # 1-thread BLAS/worker (set BEFORE numpy import); user-overridable
import sys, json, struct, multiprocessing as mp
from bisect import bisect_right as _bisect_right  # #257 ξ arithmetic decoder (pure stdlib)
import numpy as np
import brotli
import torch

# FEED-ek OPT-IN fp32 forward flag (default OFF). Read ONCE at process start (deterministic within a run).
# _FDT is the forward's working dtype: float64 (default, bit-exact authority) or float32 (local fast mode).
_FP32 = os.environ.get("INFLATE_FP32", "0") == "1"
_FDT = np.float32 if _FP32 else np.float64

MAGIC = b"LVLS1\x00"
PCAR_MAGIC = b"PCAR1\x00"  # #205 warp-real-luma pose carrier (stored keyframe luma + per-pair homography)
XIP_MAGIC = b"XIP2"        # #257 store-nothing ξ payload (quantized + coded ego twist)


def _read_blob(path):
    raw = open(path, "rb").read()
    assert raw[:len(MAGIC)] == MAGIC, "bad level-set magic"
    off = len(MAGIC); out = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        out.append(raw[off:off + n]); off += n
    m = json.loads(out[0].decode("utf-8"))
    # Optional trailing blocks are gated by the MANIFEST flags (NOT bare off<len): 5th = lane band
    # (#224 Wave E), 6th = pose carrier (#205). A lone pose-carrier block must NOT be misread as lane.
    lane_b = pcar_b = None
    if m.get("lane_render_band") is not None:
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        lane_b = raw[off:off + n]; off += n
    if m.get("pose_carrier") is not None:
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        pcar_b = raw[off:off + n]; off += n
    return m, out[1], out[2], out[3], lane_b, pcar_b


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
    # _FDT = float64 (default, bit-exact) or float32 (FEED-ek opt-in fast mode). Default path unchanged.
    proj = (2.0 * np.pi) * (np.asarray(coords, _FDT) @ np.asarray(B, _FDT))
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


def _act(u, kind, w0, s0, beta, omega):
    u = np.asarray(u, _FDT)  # _FDT = float64 (default, bit-exact) or float32 (FEED-ek opt-in fast mode)
    if kind == "hosc":
        # tanh(beta*sin(omega*u)). BIT-IDENTICAL fused form: (1) omega==1 / beta==1 skip the identity
        # full-array float64 multiply (1.0*x === x in IEEE754 -- these are ~151MB DRAM round-trips per
        # act, ~29x/pair); (2) in-place np.multiply/np.tanh reuse the sin buffer (u='pre' is not reused
        # after _act) -> fewer 151MB allocs. In-place ufunc == out-of-place ufunc elementwise -> the
        # exact float64 op values are preserved (proven by sha256 .raw parity vs the pre-fusion forward).
        t = np.sin(u) if omega == 1.0 else np.sin(omega * u)
        if beta != 1.0: np.multiply(t, beta, out=t)
        np.tanh(t, out=t)
        return t.astype(np.float32)
    if kind == "wire": return (np.cos(w0 * u) * np.exp(-((s0 * u) ** 2))).astype(np.float32)
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


# --- #205 warp-real-luma pose carrier (FREE inverse-warp; the keyframe luma + H are COUNTED) --------
# frame0 = warp(stored keyframe, stored per-pair homography H) at native res (seg-free: SegNet reads
# only frame1). VERBATIM op-for-op mirror of the tool-side _pcar_warp_frame0_from_H / pose_carrier_frame0
# (== tac.boundary_math.warp_real_luma_frame0.warp_frame0_uint8_numpy when H=homography_from_xi_numpy(xi)),
# so the shipped inflate == the numpy oracle bit-for-bit (proven by the bit-exact gate on frame0 too).
# --- #257 store-nothing ξ arithmetic decoder + DERIVE-H (rule-118 FREE: 0 archive bytes). VERBATIM
# op-for-op copies of tac.lossless.range_coder.decode_static_symbols + tac.boundary_math.xi_pose_coder
# (parse_xi_payload + homographies_from_xi) -> the shipped inflate derives the SAME H the tool oracle
# derives, bit-for-bit (bit-exact-gate proven). The 43,200 B redundant stored fp64 H is GONE.
_ST_BITS = 32; _FULL = 1 << _ST_BITS; _HALF = _FULL >> 1; _QTR = _HALF >> 1; _TQTR = _QTR * 3
_XI_FX = 910.0; _XI_CX = 582.0; _XI_CY = 437.0; _XI_D = 1.22; _XI_EPS = 1e-6


def _ar_decode(encoded, count, freqs):
    # op-for-op range_coder.decode_static_symbols (static-frequency arithmetic decode, pure stdlib).
    if count <= 0:
        return []
    cum = [0]
    for f in freqs:
        cum.append(cum[-1] + int(f))
    total = cum[-1]
    nby = len(encoded); pos = [0, 0]  # [byte_index, bit_index]
    def _bit():
        if pos[0] >= nby:
            return 0
        b = (encoded[pos[0]] >> (7 - pos[1])) & 1
        pos[1] += 1
        if pos[1] == 8:
            pos[1] = 0; pos[0] += 1
        return b
    low = 0; high = _FULL - 1; code = 0
    for _ in range(_ST_BITS):
        code = (code << 1) | _bit()
    out = []
    for _ in range(count):
        rng = high - low + 1
        scaled = ((code - low + 1) * total - 1) // rng
        sym = _bisect_right(cum, scaled) - 1
        out.append(sym)
        high = low + (rng * cum[sym + 1] // total) - 1
        low = low + (rng * cum[sym] // total)
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF; high -= _HALF; code -= _HALF
            elif low >= _QTR and high < _TQTR:
                low -= _QTR; high -= _QTR; code -= _QTR
            else:
                break
            low <<= 1; high = (high << 1) | 1; code = (code << 1) | _bit()
    return out


def _xip_parse(blob):
    # op-for-op xi_pose_coder.parse_xi_payload -> (q int16 (P,D), scales fp32 (D,)).
    assert blob[:4] == XIP_MAGIC, "bad xi-payload magic"
    off = 4
    cid, P, D = struct.unpack_from("<BHB", blob, off); off += 4
    scales = np.frombuffer(blob[off:off + D * 4], dtype=np.float32).copy(); off += D * 4
    if cid == 0:
        q = np.frombuffer(blob[off:off + P * D * 2], dtype=np.int16).reshape(P, D).copy()
        return q, scales
    cols = []
    for _k in range(D):
        seed, lo, hi = struct.unpack_from("<iii", blob, off); off += 12
        (mlen,) = struct.unpack_from("<I", blob, off); off += 4
        counts = np.frombuffer(brotli.decompress(blob[off:off + mlen]), dtype=np.uint32) if mlen else None
        off += mlen
        (slen,) = struct.unpack_from("<I", blob, off); off += 4
        stream = blob[off:off + slen]; off += slen
        col = np.empty(P, dtype=np.int64)
        if P:
            col[0] = seed
        if P > 1:
            if hi > lo and counts is not None:
                d = np.asarray(_ar_decode(stream, P - 1, counts.tolist()), dtype=np.int64) + lo
            else:
                d = np.full(P - 1, lo, dtype=np.int64)
            col[1:] = seed + np.cumsum(d)
        cols.append(col)
    return np.stack(cols, axis=1).astype(np.int16), scales


def _xip_H_from_xi(xi, pitch):
    # op-for-op xi_pose_coder.homographies_from_xi (exp_se3 -> plane-induced homography, batched).
    xi = np.asarray(xi, np.float64); rho = xi[:, :3]; om = xi[:, 3:]
    a, b, c = om[:, 0], om[:, 1], om[:, 2]; z = np.zeros_like(a)
    Ks = np.stack([np.stack([z, -c, b], -1), np.stack([c, z, -a], -1), np.stack([-b, a, z], -1)], -2)
    Ks2 = Ks @ Ks
    th2 = (om * om).sum(-1); th = np.sqrt(np.maximum(th2, 0.0))
    sm = th < _XI_EPS
    ths = np.maximum(th, _XI_EPS); th2s = np.maximum(th2, _XI_EPS * _XI_EPS); th3s = np.maximum(th ** 3, _XI_EPS ** 3)
    A = np.where(sm, 1.0 - th2 / 6.0 + th2 * th2 / 120.0, np.sin(th) / ths)
    B = np.where(sm, 0.5 - th2 / 24.0 + th2 * th2 / 720.0, (1.0 - np.cos(th)) / th2s)
    C = np.where(sm, 1.0 / 6.0 - th2 / 120.0 + th2 * th2 / 5040.0, (th - np.sin(th)) / th3s)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), Ks.shape).copy()
    R = eye + A[..., None, None] * Ks + B[..., None, None] * Ks2
    V = eye + B[..., None, None] * Ks + C[..., None, None] * Ks2
    t = (V @ rho[..., None])[..., 0]
    K = np.array([[_XI_FX, 0.0, _XI_CX], [0.0, _XI_FX, _XI_CY], [0.0, 0.0, 1.0]], np.float64)
    Kinv = np.linalg.inv(K)
    n = np.array([0.0, -np.cos(float(pitch)), -np.sin(float(pitch))], np.float64)
    M = R - (t[..., :, None] * n[None, None, :]) / _XI_D
    return K[None] @ M @ Kinv[None]


def _pcar_parse(blob):
    assert blob[:len(PCAR_MAGIC)] == PCAR_MAGIC, "bad pose-carrier magic"
    off = len(PCAR_MAGIC)
    (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    hdr = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    P = int(hdr["n_pairs"])
    if int(hdr.get("pcar_store_nothing_v", 0)) == 2:
        # #257 store-nothing v2: decode ξ -> DERIVE H FREE (rule-118). No stored H, no kf_of_pair.
        (xlen,) = struct.unpack_from("<I", blob, off); off += 4
        q, scales = _xip_parse(blob[off:off + xlen]); off += xlen
        xi = q.astype(np.float64) * scales.astype(np.float64)
        H = _xip_H_from_xi(xi, float(hdr["pitch"]))
        return {"hdr": hdr, "H": H, "keyframes": [], "kf_of_pair": [0] * P}
    H = np.frombuffer(blob[off:off + P * 9 * 8], dtype=np.float64).reshape(P, 3, 3).copy(); off += P * 9 * 8
    off += P * 6 * 2  # xi (fp16, provenance-only; the legacy decode uses the stored H)
    (n_kf,) = struct.unpack_from("<I", blob, off); off += 4
    kh, kw = int(hdr["kf_store_h"]), int(hdr["kf_store_w"])
    kfs = []
    for _ in range(n_kf):
        (blen,) = struct.unpack_from("<I", blob, off); off += 4
        raw = brotli.decompress(blob[off:off + blen]); off += blen
        kfs.append(np.frombuffer(raw, dtype=np.uint8).reshape(kh, kw, 3).copy())
    return {"hdr": hdr, "H": H, "keyframes": kfs, "kf_of_pair": hdr["kf_of_pair"]}


def _pcar_upsample(kf, ch, cw):
    if kf.shape[0] == ch and kf.shape[1] == cw:
        return np.asarray(kf, dtype=np.float64)
    x = torch.from_numpy(np.ascontiguousarray(kf)).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bilinear", align_corners=False)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.float64)


def _pcar_warp_f0(src, H, ch, cw):
    src = np.asarray(src, dtype=np.float64)
    Hh, Ww, C = int(ch), int(cw), src.shape[2]
    flat = src.reshape(-1, C)
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    grid = np.stack([us.ravel(), vs.ravel(), np.ones(Hh * Ww)], 0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(np.asarray(H, np.float64))
        src_h = Hinv @ grid
        z = src_h[2]; su = src_h[0] / z; sv = src_h[1] / z
    valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0)
             & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1))
    su_c = np.clip(su, 0.0, Ww - 1); sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64); y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1); y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]; wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]; Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]; Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx; bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat).reshape(Hh, Ww, C)
    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)


def _pcar_frame0(pc, pi, ch, cw):
    kf = pc["keyframes"][int(pc["kf_of_pair"][pi])]
    src = _pcar_upsample(kf, ch, cw)
    return _pcar_warp_f0(src, pc["H"][pi], ch, cw)


_G = {}


def _setup(src):
    # per-worker (spawn) / inherited-then-reset (fork) setup: dequant params + regen the FREE curvelet
    # bank + coords. Deterministic + identical across workers -> bit-exact. ~150ms, amortized over the
    # worker's pairs. Same op order as the serial main -> identical output.
    m, base_b, code_b, _pose, lane_b, pcar_b = _read_blob(src)
    params = _dequant(brotli.decompress(base_b), m["base_param_order"], m["base_shapes"], m["base_scales"])
    code = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32) * float(m["code_scale"])).reshape(m["code_shape"])
    rh, rw, ch, cw = int(m["render_h"]), int(m["render_w"]), int(m["camera_h"]), int(m["camera_w"])
    coords = _coords(rh, rw)
    B = _curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"], m["bank_n_iso"], m["max_bank_freq"])
    curv = _curvelet_feats(coords, B)
    P = {k: np.asarray(v, _FDT) for k, v in params.items()}  # convert once; _FDT=float64 default (bit-identical), float32 = FEED-ek opt-in fast mode
    # #224 Wave E: parse the OPTIONAL lane render-band (per-pair coords + hdr) -> per-pair coverage
    # rasters (FREE regen; computed ONCE per pair, cached). Absent -> lane_pairs=None -> band skipped.
    lane_pairs = lane_hdr = None
    if m.get("lane_render_band") is not None and lane_b is not None:
        lane_pairs, lane_hdr = _lane_parse_any(brotli.decompress(lane_b))  # Wave-F: LBND1/LBND2 dispatch
    # #205 warp-real-luma pose carrier: parse the OPTIONAL keyframe luma + per-pair H (frame0 warp).
    pcar = None
    if m.get("pose_carrier") is not None and pcar_b is not None:
        pcar = _pcar_parse(pcar_b)
    _G.update(m=m, code=code, coords=coords, curv=curv, P=P, rh=rh, rw=rw, ch=ch, cw=cw,
              framebytes=ch * cw * 3, dst=None, lane_pairs=lane_pairs, lane_hdr=lane_hdr, pcar=pcar)


def _render_pair(pi):
    # op-for-op the serial per-pair body; each pair is INDEPENDENT so parallel == serial (bit-identical).
    # Writes the pair's 2 frames to disjoint offsets of the preallocated .raw (POSIX-safe concurrent write).
    m, code, coords, curv, P = _G["m"], _G["code"], _G["coords"], _G["curv"], _G["P"]
    rh, rw, ch, cw = _G["rh"], _G["rw"], _G["ch"], _G["cw"]
    if m["self_orient"]:
        # fixed-point: dir feats from the decoder's OWN frame1 argmax (GT-free, 0 bytes).
        dirf = np.zeros((curv.shape[0], 4 * int(m["n_dir_freqs"])), np.float32)
        prev_am = None; _h0_conv = None  # FEED-ej: cache the converged iter's h0 (== the post-loop h0)
        for _ in range(int(m["so_iters"])):
            feats = np.concatenate([curv, dirf], axis=-1)
            _h0_it = _in_proj_h0(P, feats, m)
            phi, _ = _outputs_from_h0(P, _h0_it, code[2 * pi + 1], m, False)
            am = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
            if prev_am is not None and np.array_equal(am, prev_am):
                _h0_conv = _h0_it  # converged: feats frozen from here -> this h0 IS the final h0 (bit-exact reuse)
                break  # argmax fixed point: dirf would not change -> remaining iters are no-ops
            dirf = _dir_feats(coords, am, m["n_dir_freqs"], m["so_freq_along"], m["so_freq_across"], m["so_tau"])
            prev_am = am
        feats = np.concatenate([curv, dirf], axis=-1)
        # FEED-ej: on the early-break `feats` is unchanged from the converged iter -> reuse its h0
        # (bit-exact: identical feats -> identical _in_proj_h0). No convergence -> recompute (as before).
        h0 = _h0_conv if _h0_conv is not None else _in_proj_h0(P, feats, m)
    else:
        feats = curv
        h0 = _in_proj_h0(P, feats, m)  # shared across the pair's 2 frames (identical feats)
    lane_pairs, lane_hdr = _G.get("lane_pairs"), _G.get("lane_hdr")
    band = lane_pairs is not None and pi < len(lane_pairs)
    pcar = _G.get("pcar")
    cov_flat = None
    if band:
        # coverage depends ONLY on the pair's lines (per-pair) -> rasterize ONCE, share across f0/f1.
        cov_flat = _lane_coverage(lane_pairs[pi], rh, rw, lane_hdr).reshape(-1)
    frames = []
    for fk in range(2):
        # #205: frame0 (fk==0) is the pose carrier (seg-free: SegNet reads only frame1). frame1
        # (fk==1) stays the witness render (d_seg). Absent pose carrier -> frame0 is the INR render.
        #  * warp_real_luma: frame0 = warp(stored keyframe, stored per-pair H).
        #  * store_nothing:  frame0 = warp(the witness's OWN frame0 INR render, stored per-pair H)
        #                    -- NO stored keyframe (rule-118: the render is FREE, only xi/H COUNTED).
        if fk == 0 and pcar is not None and pi < int(pcar["hdr"]["n_pairs"]):
            if pcar["hdr"].get("pose_carrier_mode") == "store_nothing":
                _phi0, rgb0 = _outputs_from_h0(P, h0, code[2 * pi + 0], m, True)
                src0 = _R(rgb0, rh, rw, ch, cw).astype(np.float64)  # witness frame0 render, camera-native
                frames.append(_pcar_warp_f0(src0, pcar["H"][pi], ch, cw).tobytes())
            else:
                frames.append(_pcar_frame0(pcar, pi, ch, cw).tobytes())
            continue
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

        man, base_b, code_b, pose_b, lane_b, pcar_b = _read_blob_bytes(src_bin.read_bytes())
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
        # #205: cap the pose carrier to eval_pairs (slice H/xi + prune to referenced keyframes).
        pcar_cap = None
        if pcar_b is not None and man.get("pose_carrier") is not None:
            pcar_cap = _cap_pose_carrier(pcar_b, eval_pairs)
            man["pose_carrier"] = {**man["pose_carrier"], "n_pairs": eval_pairs}
        mj = json.dumps(man, separators=(",", ":")).encode()
        capped = _io_pack(mj, base_b, brotli.compress(qc.astype(np.int8).tobytes(), quality=11),
                          pose_b or None, lane_cap, pcar_cap)
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


def _dequant_blob(blob: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray,
                                        list[list[Any]] | None, dict[str, Any] | None]:
    """Dequant (int8 -> fp32*scale) the base params + code from a LVLS1 blob, EXACTLY as the shipped
    inflate does (the numpy-fp32 oracle authority uses the SAME dequantized weights). Returns
    (manifest, params, code, lane_pairs|None, pose_carrier_parsed|None). Used by store_nothing's
    ``pose_carrier_confirm`` to regenerate the witness-render frame0 authority (no stored keyframe)."""
    import brotli

    m, base_b, code_b, _pose, lane_b, pcar_b = _read_blob_bytes(blob)
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
    lane_pairs: list[list[Any]] | None = None
    if lane_b is not None and m.get("lane_render_band") is not None:
        lane_pairs, _hdr = deserialize_lane_band_any(brotli.decompress(lane_b))
    pcar_parsed = parse_pose_carrier(pcar_b) if (pcar_b is not None and m.get("pose_carrier") is not None) else None
    return m, params, code, lane_pairs, pcar_parsed


def pose_carrier_confirm(
    raw_path: Path, eval_pairs: int, gt_cache: str | None, num_pairs: int, pose_carrier_bytes: bytes,
    blob: bytes | None = None,
) -> dict[str, Any]:
    """CONFIRM the pose-carrier decode through the REAL byte-closed inflate (the #205 launch gate).

    (1) frame0 DECODE-REPRODUCTION: the inflated .raw frame0 == the numpy AUTHORITY frame0 BIT-FOR-BIT
        (proves the shipped inflate warps by the stored H exactly -> the training-side warp d_pose IS
        the real-decode d_pose, no gap by construction). The authority frame0 is mode-dependent:
          * warp_real_luma: ``pose_carrier_frame0(pc, pi)`` = warp(stored keyframe, H).
          * store_nothing:  the numpy-fp32 ORACLE frame0 = warp(the witness's OWN f0 render, H) --
            regenerated from ``blob`` via ``numpy_oracle_reference_frames`` (needs ``blob``; NO-FAKE).
    (2) CONTEXT d_pose triad (frozen CPU-torch PoseNet, the authority; NEVER MPS):
        * carrier (raw f0 = warp, raw f1 = witness) -- the REAL row d_pose (what evaluate.py scores);
        * unwarped (gt_f0 real, raw f1 = witness) -- isolates the warp's marginal effect;
        * CEILING (gt_f0 real, warp(gt_f0, H)) -- the (real, warped-real) pair the reference tool
          ``measure_warp_dpose_through_R`` measures. The carrier-vs-ceiling GAP is the OPEN
          witness-f1 / trained-residual work.
    NON-PROMOTABLE (macOS-CPU advisory)."""
    pc = parse_pose_carrier(pose_carrier_bytes)
    mode = pose_carrier_mode(pc)
    if gt_cache:
        gt, _seg, posenet_cpu = twr.load_gt_from_cache(Path(gt_cache), num_pairs)
    else:
        gt, _seg, posenet_cpu = twr.precompute_gt(num_pairs)
    P = min(int(eval_pairs), gt.n_pairs, int(pc["hdr"]["n_pairs"]))
    fb = CAMERA_H * CAMERA_W * 3
    raw_f0, raw_f1 = [], []
    with open(raw_path, "rb") as f:
        for _pi in range(P):
            raw_f0.append(np.frombuffer(f.read(fb), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
            raw_f1.append(np.frombuffer(f.read(fb), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
    gt_f0 = [np.asarray(gt.gt_f0[pi]) for pi in range(P)]
    poses = [gt.gt_poses[pi] for pi in range(P)]
    # (1) frame0 decode-reproduction (numpy authority == the shipped inflate .raw).
    if mode == "store_nothing":
        if blob is None:
            raise ValueError(
                "store_nothing pose_carrier_confirm needs the LVLS1 blob to regenerate the "
                "witness-render frame0 authority (the frame0 source is the FREE INR render, not a "
                "stored keyframe). NO-FAKE: refusing to fabricate the authority.")
        m, params, code, lane_pairs, pc_oracle = _dequant_blob(blob)
        oracle_frames, _am = numpy_oracle_reference_frames(params, code, m, P, lane_pairs, pose_carrier=pc_oracle)
        auth_f0 = [np.asarray(oracle_frames[2 * pi]) for pi in range(P)]
        # store_nothing ceiling = (real f0, warp(real f0, H)) -- the keyframe-INDEPENDENT reference
        # (matches measure_warp_dpose_through_R), NOT warp(witness render) which is non-photoreal.
        ceiling_f1 = [pose_carrier_frame0_from_source(pc, pi, gt_f0[pi]) for pi in range(P)]
    else:
        auth_f0 = [pose_carrier_frame0(pc, pi) for pi in range(P)]
        ceiling_f1 = auth_f0  # warp_real_luma: stored keyframe ~= gt_f0, warp(keyframe, H) is the ceiling
    max_abs = 0
    n_diff = 0
    for pi in range(P):
        d = int(np.abs(raw_f0[pi].astype(np.int32) - auth_f0[pi].astype(np.int32)).max())
        if d != 0:
            n_diff += 1
        max_abs = max(max_abs, d)
    # (2) d_pose triad (authority PoseNet).
    dp_carrier = twr.cpu_verdict_d_pose_batch(posenet_cpu, raw_f0, raw_f1, poses)
    dp_unwarped = twr.cpu_verdict_d_pose_batch(posenet_cpu, gt_f0, raw_f1, poses)
    dp_ceiling = twr.cpu_verdict_d_pose_batch(posenet_cpu, gt_f0, ceiling_f1, poses)
    return {
        "pairs": P,
        "pose_carrier_mode": mode,
        "frame0_decode_bit_exact": bool(max_abs == 0),
        "frame0_max_abs_uint8_diff": int(max_abs),
        "frame0_n_frames_differing": int(n_diff),
        "d_pose_carrier_warp_f0_witness_f1": float(np.mean(dp_carrier)),
        "d_pose_unwarped_gtf0_witness_f1": float(np.mean(dp_unwarped)),
        "d_pose_ceiling_gtf0_warpf0": float(np.mean(dp_ceiling)),
        "d_pose_null_pose_blind_ref": 189.62,
        "training_side_vs_real_decode_parity": (
            ("IDENTICAL by construction: raw frame0 == numpy-authority "
             + ("store-nothing witness-render warp" if mode == "store_nothing" else "stored-keyframe warp")
             + " frame0 bit-for-bit (above) AND raw frame1 == witness render (bit-exact gate) -> the "
               "training-side warp d_pose EQUALS the real-decode d_pose (no surrogate gap).")
            if max_abs == 0 else
            f"MISMATCH: raw frame0 != authority (max_abs={max_abs}) -> the byte-close does NOT "
            "faithfully reproduce the pose decode (NO-FAKE: investigate)."),
        "gap_note": (
            "carrier (warp-f0 + WITNESS-f1) vs ceiling (real-f0 + warped-real-f1): the gap is the OPEN "
            "witness-composition work -- the witness f1 is non-photoreal + the trained dxi residual "
            "(w_pose>0) is UNMEASURED. The ceiling is the reference (real,warped-real) pair, NOT the "
            "contest-legal witness pair. NO borrowed 3.4e-5 (ancestor-RGB, never validated on the witness)."
            + (" store_nothing stores ONLY xi/H (~0 marginal bytes) -- the frame0 source is the FREE "
               "witness render, so the WHOLE keyframe payload (the warp_real_luma rate) is GONE." if mode == "store_nothing" else "")),
        "authority": _AUTHORITY,
        "promotion_claim": False,
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
    lane_pairs: list[list[Any]] | None = None, pose_carrier: dict[str, Any] | None = None,
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
            # #205 pose carrier: frame0 (fk==0) is the carrier (seg-free), NOT the plain INR frame0.
            # frame1 (fk==1) stays the witness render (it drives d_seg + the argmax). argmax is taken
            # at fk==1 -> the pose carrier is argmax-free. Mode-aware (bit-mirrors the shipped inflate):
            #  * warp_real_luma: warp(stored keyframe, H); store_nothing: warp(witness's OWN f0 render, H).
            if fk == 0 and pose_carrier is not None:
                if pose_carrier_mode(pose_carrier) == "store_nothing":
                    rgb0, _phi0 = levelset_rgb_forward_numpy(params, feats, code[2 * pi + 0], **fwd_kw)
                    src0 = _torch_R_reference(rgb0, rh, rw, ch, cw).astype(np.float64)
                    frames.append(pose_carrier_frame0_from_source(pose_carrier, pi, src0))
                else:
                    frames.append(pose_carrier_frame0(pose_carrier, pi))
                continue
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
    m, base_b, code_b, _pose, lane_b, pcar_b = _read_blob_bytes(blob)
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
    # #205: carry the pose carrier through the gp-capped repack (slice to gp) so the shipped inflate
    # warps the SAME frame0's the oracle does over the SAME gp pairs (bit-exact frame0 too).
    pcar_cap = None
    pose_carrier_oracle = None
    if pcar_b is not None and m.get("pose_carrier") is not None:
        pcar_cap = _cap_pose_carrier(pcar_b, gp)
        pose_carrier_oracle = parse_pose_carrier(pcar_cap)
        man["pose_carrier"] = {**m["pose_carrier"], "n_pairs": gp}
    mj = json.dumps(man, separators=(",", ":")).encode()
    capped_bin = gate_root / "gate.bin"
    capped_bin.write_bytes(_io_pack(
        mj, base_b, brotli.compress(qc.astype(np.int8).tobytes(), quality=11), None, lane_b_cap, pcar_cap))
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
    ref_frames, ref_argmax = numpy_oracle_reference_frames(
        ref_params, ref_code, man, gp, lane_pairs_cap, pose_carrier=pose_carrier_oracle)

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


def _read_blob_bytes(
    blob: bytes,
) -> tuple[dict[str, Any], bytes, bytes, bytes, bytes | None, bytes | None]:
    """Parse an in-memory LVLS1 blob (same grammar as the shipped inflate._read_blob). Returns
    (manifest, base, code, pose, lane_band|None, pose_carrier|None). Trailing optional blocks are
    gated by the MANIFEST flags (lane_render_band / pose_carrier), NOT a bare off<len -- so a lone
    pose_carrier block is not misread as lane. Default-off -> both trailing are None."""
    assert blob[: len(_MAGIC)] == _MAGIC, "bad level-set magic"
    off = len(_MAGIC)
    out: list[bytes] = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", blob, off)
        off += 4
        out.append(blob[off:off + n])
        off += n
    manifest = json.loads(out[0].decode("utf-8"))
    lane_band: bytes | None = None
    pose_carrier: bytes | None = None
    if manifest.get("lane_render_band") is not None:
        (n,) = struct.unpack_from("<I", blob, off); off += 4
        lane_band = blob[off:off + n]; off += n
    if manifest.get("pose_carrier") is not None:
        (n,) = struct.unpack_from("<I", blob, off); off += 4
        pose_carrier = blob[off:off + n]; off += n
    return manifest, out[1], out[2], out[3], lane_band, pose_carrier


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
    pose_carrier: bool = False,  # #205 warp-real-luma frame0 pose carrier
    pose_carrier_cfg: dict[str, Any] | None = None,
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

    # #205 warp-real-luma pose carrier: build the COUNTED keyframe luma + per-pair homography section.
    pose_carrier_bytes: bytes | None = None
    pose_carrier_manifest: dict[str, Any] | None = None
    pose_carrier_report: dict[str, Any] = {"active": False}
    keyframe_accounting: dict[str, Any] | None = None
    if pose_carrier:
        pc_cfg = pose_carrier_cfg or {"s_t": 0.16, "s_r": 1.0, "pitch": 0.02, "stride": 1,
                                      "downscale": 1, "mode": "warp_real_luma"}
        pc_mode = str(pc_cfg.get("mode", "warp_real_luma"))
        pose_carrier_bytes, pose_carrier_manifest, pose_carrier_report = build_pose_carrier_section(
            gt_cache, n_pairs, pc_cfg)
        # REUSE the canonical keyframe_payload_accounting (tools.compose_witness_archive, f7c6abdea) for
        # the rule-118 line item -- fed the MEASURED keyframe blob bytes actually stored (0 for store_nothing).
        from tools.compose_witness_archive import keyframe_payload_accounting  # noqa: PLC0415
        kf_measured = int(pose_carrier_report["keyframe_blob_bytes_total"])
        keyframe_accounting = keyframe_payload_accounting(
            argparse.Namespace(keyframe_payload_path=None, keyframe_payload_bytes=kf_measured,
                               keyframe_count=int(pose_carrier_report["n_keyframes"])))
        _pc_section = int(pose_carrier_report["pose_carrier_section_bytes"])
        if pc_mode == "store_nothing":
            pose_note = (f"STORE-NOTHING (xi-only) carrier ACTIVE: NO stored keyframe (0 B) -- frame0 = "
                         f"warp(the witness's OWN frame0 render, per-pair H). Section={_pc_section} B "
                         f"(xi/H only; ~0 marginal keyframe rate). Track B store-nothing-but-xi (18927a1ae).")
            print(f"[pose-carrier] STORE-NOTHING (xi-only) ACTIVE: keyframe_bytes=0 "
                  f"section={_pc_section} B (H+xi only; witness render warped by xi = FREE frame0)  "
                  f"{_AUTHORITY}", flush=True)
        else:
            pose_note = (f"WARP-REAL-LUMA frame0 carrier ACTIVE: {pose_carrier_report['n_keyframes']} stored "
                         f"keyframe(s) (store_hw={pose_carrier_report['keyframe_store_hw']}, "
                         f"lossless_native={pose_carrier_report['keyframe_lossless_native']}) COUNTED "
                         f"{kf_measured} B; frame0 = warp(keyframe, per-pair H) -> PoseNet reads real warped "
                         f"motion. Section={_pc_section} B.")
            print(f"[pose-carrier] warp-real-luma frame0 ACTIVE: n_kf={pose_carrier_report['n_keyframes']} "
                  f"stride={pose_carrier_report['stride']} store_hw={pose_carrier_report['keyframe_store_hw']} "
                  f"keyframe_bytes={kf_measured} section={_pc_section} B "
                  f"(rate += {keyframe_accounting['keyframe_blob_rate']:.5f})  {_AUTHORITY}", flush=True)

    blob, breakdown = build_levelset_blob(params, cfg, so, pose_bytes, lane_band_bytes, lane_manifest,
                                          pose_carrier_bytes, pose_carrier_manifest)
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

    # #205: CONFIRM the pose-carrier decode through the REAL byte-closed inflate (frame0 bit-exact
    # reproduction + the d_pose triad). Only when the carrier is active AND we have GT to score.
    pose_carrier_confirmation: dict[str, Any] = {"checked": False}
    if pose_carrier and pose_carrier_bytes is not None and not skip_parity:
        pose_carrier_confirmation = pose_carrier_confirm(
            Path(inflate_info["raw_path"]), inflate_info["eval_pairs"], gt_cache, n_pairs,
            pose_carrier_bytes, blob=blob)
        pose_carrier_confirmation["checked"] = True
        pc_c = pose_carrier_confirmation
        print(f"[pose-carrier CONFIRM] frame0 decode bit-exact={pc_c['frame0_decode_bit_exact']} "
              f"(max_abs={pc_c['frame0_max_abs_uint8_diff']}) | d_pose: carrier(warp-f0+witness-f1)="
              f"{pc_c['d_pose_carrier_warp_f0_witness_f1']:.4f}  unwarped(gtf0+witness-f1)="
              f"{pc_c['d_pose_unwarped_gtf0_witness_f1']:.4f}  ceiling(gtf0+warp-gtf0)="
              f"{pc_c['d_pose_ceiling_gtf0_warpf0']:.4f}  (null~189.62)  {_AUTHORITY}", flush=True)
        if not pc_c["frame0_decode_bit_exact"]:
            print("[pose-carrier CONFIRM] WARNING: frame0 decode NOT bit-exact vs the numpy authority "
                  "-> the byte-close does NOT faithfully reproduce the pose decode (NO-FAKE).", flush=True)

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
            "pose_carrier": pose_carrier_report,
        },
        "lane_render_band": lane_report,
        "pose_carrier": pose_carrier_report,
        "pose_carrier_keyframe_accounting": (
            None if keyframe_accounting is None else {
                **keyframe_accounting,
                # VERIFY the counted keyframe bytes against the ACTUAL bytes in the composed archive.
                "verified_against_archive": {
                    "pose_carrier_section_bytes_in_0bin": int(breakdown.get("pose_carrier_counted_bytes", 0)),
                    "section_bytes_match_report": bool(
                        int(breakdown.get("pose_carrier_counted_bytes", 0))
                        == int(pose_carrier_report.get("pose_carrier_section_bytes", -1))),
                    "keyframe_bytes_le_section": bool(
                        int(pose_carrier_report.get("keyframe_blob_bytes_total", 0))
                        <= int(breakdown.get("pose_carrier_counted_bytes", 0))),
                    "note": "keyframe payload is COUNTED inside archive.zip (0.bin 6th block -> the "
                            "measured rate term already includes it); the accounting is cross-checked "
                            "against the real section st_size (not an estimate).",
                },
            }),
        "pose_carrier_confirmation": pose_carrier_confirmation,
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
    # #205 WARP-REAL-LUMA FRAME0 POSE CARRIER. OFF by default -> byte-identical. Needs --gt-cache
    # (real gt_f0 keyframes + gt_poses). Replaces the render's SEG-free frame0 with a stored real
    # keyframe warped by the per-pair ego homography -> PoseNet reads real warped motion; the keyframe
    # payload is COUNTED in archive.zip (rule 118: keyframe COUNTED, warp decoder FREE).
    ap.add_argument("--pose-carrier", action="store_true",
                    help="#205 frame0 pose carrier (needs --gt-cache). frame0 = warp(source, per-pair "
                         "ego H); source per --pose-carrier-mode. OFF by default (archive byte-identical "
                         "when off).")
    ap.add_argument("--pose-carrier-mode", type=str, default="warp_real_luma",
                    choices=["warp_real_luma", "store_nothing"],
                    help="pose-carrier frame0 source. warp_real_luma (default) = warp a STORED real "
                         "keyframe (COUNTS keyframe luma+H). store_nothing = store ONLY xi/H (~0 marginal "
                         "bytes); frame0 = warp(the witness's OWN frame0 INR render, H) -- Track B "
                         "store-nothing-but-xi (18927a1ae). A/B-able against warp_real_luma at #205.")
    ap.add_argument("--pc-s-t", type=float, default=0.16, help="pose-carrier translation calibration scale (s_t).")
    ap.add_argument("--pc-s-r", type=float, default=1.0, help="pose-carrier rotation calibration scale (s_r).")
    ap.add_argument("--pc-pitch", type=float, default=0.02, help="pose-carrier road-plane pitch (rad).")
    ap.add_argument("--pc-keyframe-stride", type=int, default=1,
                    help="pose-carrier keyframe stride (1 = per-pair, the exact warp-real-luma upper "
                         "bound; >1 shares one stored keyframe across a window -> cheaper rate, "
                         "geometrically approximate for shared keyframes -- LOUDLY flagged).")
    ap.add_argument("--pc-keyframe-downscale", type=int, default=1,
                    help="pose-carrier keyframe store downscale factor (1 = native lossless; 2 = "
                         "native/2, upsampled before warp -- the cheap-rate lever, ~matched to PoseNet's "
                         "512x384 working res).")
    # #257 store-nothing ξ coder (store_nothing mode only; H is DERIVED FREE at decode, kf_of_pair DROPPED).
    ap.add_argument("--pc-xi-coder", type=str, default="delta_ar", choices=["delta_ar", "none"],
                    help="store-nothing ξ payload coder: delta_ar (DEFAULT) = per-channel temporal-delta + "
                         "arithmetic coder (~0.002 n600 rate); none = raw int16 ξ (~0.005, the "
                         "GUARANTEED-today fallback + strict-parity reference). Both decode to the IDENTICAL "
                         "quantized ξ. Ignored for warp_real_luma.")
    ap.add_argument("--no-xi-coder", action="store_true",
                    help="store-nothing: store raw int16 ξ (== --pc-xi-coder none). The guaranteed fallback.")
    ap.add_argument("--pc-xi-qlevels", type=int, default=4096,
                    help="store-nothing ξ quantization levels per channel (default 4096 ~ fp16+; finer -> "
                         "more coded bytes, coarser -> looser d_pose. Both coders share it -> strict parity).")
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
        pose_carrier=args.pose_carrier,
        pose_carrier_cfg={
            "s_t": args.pc_s_t, "s_r": args.pc_s_r, "pitch": args.pc_pitch,
            "stride": args.pc_keyframe_stride, "downscale": args.pc_keyframe_downscale,
            "mode": args.pose_carrier_mode,
            "xi_coder": ("none" if args.no_xi_coder else args.pc_xi_coder),  # #257
            "xi_q_levels": args.pc_xi_qlevels,
        },
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

    # #247 CLOSE: a byte-closed realized verdict landed -> drain the DSL levers this run FIRED from the
    # activation ledger's duty_to_measure (fired -> measured). Fail-safe: the ledger must NEVER break the
    # byte-close. Only when parity actually ran (a real realized d_seg/d_pose verdict, not --skip-parity).
    if not report.get("parity", {}).get("skipped", False):
        try:
            from tac.witness_dsl.activation_ledger import record_measured_for_run
            rows = record_measured_for_run(
                str(args.ckpt_dir), verdict_ref=str(out),
                reason="byte-closed realized-parity verdict landed", agent="levelset_byte_close",
            )
            if rows:
                print(f"[activation-ledger] recorded measured for {len(rows)} fired lever(s): "
                      f"{[r['lever'] for r in rows]}", flush=True)
        except Exception as _e:  # noqa: BLE001 — advisory ledger, never blocks the verdict
            print(f"[activation-ledger] measured-record skipped (non-fatal): {_e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
