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
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    _int8_symmetric,
    curvelet_directional_B,
    quantize_levelset_blob,
)

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
    mbf = raw_cfg.get("__cfg_max_bank_freq", None)
    cfg["max_bank_freq"] = None if (mbf is None or float(mbf) < 0) else float(mbf)
    rh = raw_cfg.get("__render_hw", None)
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
    params: dict[str, np.ndarray], cfg: dict[str, Any], so: dict[str, Any], pose_sidecar: bytes | None
) -> tuple[bytes, dict[str, Any]]:
    """Layout: magic | u32 manifest_len | manifest_json | u32 base_brotli_len | base_brotli |
            u32 code_brotli_len | code_brotli | u32 pose_len | pose_sidecar(optional).
    base = int8(all params except code) concat -> ONE brotli stream (== quantize_levelset_blob);
    code = int8(code) -> a SECOND brotli stream. The curvelet bank is NOT stored (free, rule 118).
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
    mj = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    out = _io_pack(mj, base_brotli, code_brotli, pose_sidecar)
    # cross-check our accounting against the canonical quantize_levelset_blob (same int8 grammar).
    canon = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in params.items()})
    breakdown = {
        "n_params": int(sum(int(np.prod(s)) for s in shapes.values())) + int(np.prod(code.shape)),
        "manifest_bytes": len(mj),
        "base_int8_brotli_bytes": len(base_brotli),
        "code_int8_brotli_bytes": len(code_brotli),
        "pose_sidecar_bytes": (len(pose_sidecar) if pose_sidecar else 0),
        "magic_and_prefixes_bytes": len(_MAGIC) + 16,
        "total_0bin_bytes": len(out),
        "canonical_quantize_blob_bytes": int(canon["total_quantized_blob_bytes"]),
        "accounting_matches_canonical": bool(
            len(base_brotli) == canon["base_int8_brotli_bytes"]
            and len(code_brotli) == canon["code_int8_brotli_bytes"]),
    }
    return out, breakdown


def _io_pack(manifest: bytes, base: bytes, code: bytes, pose: bytes | None) -> bytes:
    buf = bytearray()
    buf += _MAGIC
    for chunk in (manifest, base, code, (pose or b"")):
        buf += struct.pack("<I", len(chunk))
        buf += chunk
    return bytes(buf)


# ---------------------------------------------------------------------------
# the self-contained inflate.py (numpy fwd + torch R [+ scipy iff self-orient]).
# ---------------------------------------------------------------------------
# REVIEW-GATE: inflate.py exceeds the 100-LOC default budget (~178 LOC) under the explicit
# <=200-LOC waiver: it inlines the byte-closeable SELF-ORIENT fixed-point (curvelet bank regen +
# decoder-own-argmax tangent + directional feats) which is the rule-118 FREE directional lever the
# mod-32 target requires. The forward is an op-for-op mirror of levelset_rgb_forward_numpy.
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
import sys, json, struct
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
    return json.loads(out[0].decode("utf-8")), out[1], out[2], out[3]


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


def _outputs_from_h0(P, h0, code_row, m, want_rgb):
    # op-for-op mirror of levelset_rgb_forward_numpy AFTER in_proj. h0 = the float32 in_proj act.
    # want_rgb=False -> return (phi, None) skipping the rgb head (out_tex/palette/softmax/sigmoid do
    # NOT feed phi, so argmax(phi) is identical) -- used by the self-orient fixed point.
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    cr = np.asarray(code_row, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        film = (cr @ P["film.weight"].T + P["film.bias"]).reshape(m["n_hidden"], 2, m["hidden_dim"])
        h = h0
        for li in range(m["n_hidden"]):
            h = _act((h @ P["hidden.%d.weight" % li].T + P["hidden.%d.bias" % li]) * (1.0 + film[li, 0]) + film[li, 1], *kw)
        phi = h @ P["out_sdf.weight"].T + P["out_sdf.bias"]
        if not want_rgb:
            return phi.astype(np.float32), None
        tex = h @ P["out_tex.weight"].T + P["out_tex.bias"]
        z = phi / float(m["softmax_temp"]); z = z - z.max(-1, keepdims=True)
        soft = np.exp(z); soft = soft / soft.sum(-1, keepdims=True)
        rgb = (1.0 / (1.0 + np.exp(-(soft @ P["palette"] + tex)))) * 255.0
        if not m["chroma"]:
            luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb = np.concatenate([luma, luma, luma], axis=-1)
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


def _R(rgb, rh, rw, ch, cw):
    x = torch.from_numpy(np.ascontiguousarray(rgb.reshape(rh, rw, 3))).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m, base_b, code_b, _pose = _read_blob(src)
    params = _dequant(brotli.decompress(base_b), m["base_param_order"], m["base_shapes"], m["base_scales"])
    code = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32) * float(m["code_scale"])).reshape(m["code_shape"])
    rh, rw, ch, cw = int(m["render_h"]), int(m["render_w"]), int(m["camera_h"]), int(m["camera_w"])
    coords = _coords(rh, rw)
    B = _curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"], m["bank_n_iso"], m["max_bank_freq"])
    curv = _curvelet_feats(coords, B)
    n_pairs = int(m["n_pairs"])
    P = {k: np.asarray(v, np.float64) for k, v in params.items()}  # convert once (bit-identical)
    with open(dst, "wb") as f:
        for pi in range(n_pairs):
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
            for fk in range(2):
                _phi, rgb = _outputs_from_h0(P, h0, code[2 * pi + fk], m, True)
                f.write(_R(rgb, rh, rw, ch, cw).tobytes())
    print("inflated %d frames (%d pairs) -> %s [%dx%dx%dx3 uint8]" % (2 * n_pairs, n_pairs, dst, 2 * n_pairs, ch, cw), flush=True)


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

        raw = src_bin.read_bytes()
        assert raw[: len(_MAGIC)] == _MAGIC
        off = len(_MAGIC)
        parts = []
        for _ in range(4):
            (n,) = struct.unpack_from("<I", raw, off)
            off += 4
            parts.append(raw[off : off + n])
            off += n
        man = json.loads(parts[0].decode())
        code = (np.frombuffer(brotli.decompress(parts[2]), dtype=np.int8).astype(np.float32)
                * man["code_scale"]).reshape(man["code_shape"])
        code_cap = code[: 2 * eval_pairs]
        qc, sc = _int8_symmetric(code_cap)
        man["n_pairs"] = eval_pairs
        man["code_shape"] = list(code_cap.shape)
        man["code_scale"] = float(sc)
        mj = json.dumps(man, separators=(",", ":")).encode()
        capped = _io_pack(mj, parts[1], brotli.compress(qc.astype(np.int8).tobytes(), quality=11), parts[3] or None)
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
) -> dict[str, Any]:
    params, cfg = _load_levelset_ckpt(ckpt_dir, npz_name)
    n_pairs = int(cfg["n_pairs"])
    so = detect_self_orient(cfg, so_overrides)

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

    blob, breakdown = build_levelset_blob(params, cfg, so, pose_bytes)
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
        },
        "inflate": inflate_info,
        "parity_on_inflated_frames": parity,
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
    # self-orient params the trainer does NOT persist (a trainer gap, flagged) -> trainer defaults.
    ap.add_argument("--so-freq-across", type=float, default=32.0, help="self-orient HIGH freq across the edge (trainer default 32).")
    ap.add_argument("--so-freq-along", type=float, default=4.0, help="self-orient LOW freq along the edge (trainer default 4).")
    ap.add_argument("--so-tau", type=float, default=4.0, help="self-orient boundary-proximity tau (trainer default 4).")
    ap.add_argument("--so-iters", type=int, default=4, help="self-orient fixed-point iterations at decode (convergence).")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = ap.parse_args(argv)

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
    )
    out = args.out or (_REPO / "reports" / f"levelset_byte_close_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {out}  {_AUTHORITY}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
