# SPDX-License-Identifier: MIT
"""TORCH level-set inflate decode — fast, device-flexible (CPU/CUDA) parallel of the
numpy-fp64 reference (``tools/levelset_byte_close_and_eval._INFLATE_PY`` / the canonical
``tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy``).

WHY (FEED-ei/FEED-ej, operator 2026-06-27): the shipped numpy-fp64 level-set inflate measured
53-61 min for n600 on a 4-core CPU (FEED-eg) — OVER the contest 30-min budget — and is dominated
~81% by per-element ``tanh(4*sin(u))`` transcendentals in the self-orient fixed-point forwards.
inflate.py can run on a T4 (the contest CUDA axis) as well as CPU; the transcendental-heavy
forward is embarrassingly parallel. This module re-implements the DECODE forward in TORCH, fp32,
device-flexible, mirroring the numpy reference op-for-op — a COMPLEMENTARY path to the already
numpy-CPU-optimized a7613e19 template (self-orient early-stop + share-h0 + skip-rgb-head), NOT a
re-do. It is the FAST SHIPPABLE decode (torch-fp32, CPU or T4); the numpy-fp64 template remains
the bit-identical reference/authority. Parity-gated: the torch decode is admissible only when its
SegNet argmax map matches the numpy reference (argmax-exact, tiny exact-tie flips OK).

RULE 118 / COMPLIANCE (unchanged): a torch ``inflate.py`` is GENERIC decode code -> FREE (zero
archive bytes; it reads the SAME ``LVLS1`` int8+brotli blob the numpy inflate reads). NO weights /
SegNet / PoseNet ship in the archive; the witness frames are byte-determined by the archive +
generic code. So the rate term and the witness are IDENTICAL; only the decode arithmetic changes,
which is why the torch path must pass the argmax parity gate before it may be the shipped decode.

AUTHORITY: numpy-fp64 reference = the bit-identical verdict authority (CLAUDE.md
deterministic-reproducibility). torch (any device) is the fast decode, NEVER an authority.
MPS is NEVER used here (CPU or CUDA only). The realized d_seg/d_pose verdict is still the FROZEN
CPU-torch SegNet/PoseNet on the inflated frames; this module only changes HOW the frames are
generated, parity-gated to the numpy reference.

Op-for-op mirror of the reference forward (all linear layers are mlx.nn.Linear convention
``y = x @ W.T + b``):

    h0   = act(feats @ in_proj.weight.T + in_proj.bias)          # shared by a pair's 2 frames
    film = (code_row @ film.weight.T + film.bias).reshape(nH,2,H) # FiLM per (pair,frame)
    h    = h0
    for li in range(nH):
        h = act((h @ hidden.li.weight.T + hidden.li.bias) * (1+film[li,0]) + film[li,1])
    phi  = h @ out_sdf.weight.T + out_sdf.bias                    # K SDF fields (argmax => seg)
    tex  = h @ out_tex.weight.T + out_tex.bias                    # pose-carrying RGB texture
    soft = softmax(phi / softmax_temp)
    rgb  = sigmoid(soft @ palette + tex) * 255  [+ luma collapse iff not chroma]
    frame= R(rgb)  # bicubic up render->camera, round, clamp, uint8  (ALWAYS fp32, like the ref)

The curvelet bank + (when used) the self-orient directional feats are regenerated for FREE
(rule 118), identically to the numpy reference. The self-orient fixed point keeps the SAME
EARLY-STOP + share-h0 + skip-rgb-head structure as the optimized numpy template (bit-identical
op order at fp64; argmax-faithful at fp32). The directional feat construction (scipy EDT on the
decoder's OWN argmax) stays on CPU/numpy — it is cheap and not the transcendental bottleneck; only
the per-pixel matmul+activation forward (the 81% cost) is moved to torch/device.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"LVLS1\x00"
CAMERA_H, CAMERA_W = 874, 1164


# ---------------------------------------------------------------------------
# blob read (identical grammar to the numpy inflate; the archive is unchanged)
# ---------------------------------------------------------------------------
def read_levelset_blob(path: str | Path) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    raw = Path(path).read_bytes()
    assert raw[: len(MAGIC)] == MAGIC, "bad level-set magic"
    off = len(MAGIC)
    out: list[bytes] = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", raw, off)
        off += 4
        out.append(raw[off : off + n])
        off += n
    return json.loads(out[0].decode("utf-8")), out[1], out[2], out[3]


def dequant_params(blob: bytes, order: list[str], shapes: dict[str, Any], scales: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    off = 0
    flat = np.frombuffer(blob, dtype=np.int8)
    for name in order:
        shp = tuple(shapes[name])
        n = int(np.prod(shp))
        out[name] = (flat[off : off + n].astype(np.float32) * float(scales[name])).reshape(shp)
        off += n
    return out


# ---------------------------------------------------------------------------
# FREE generic tables (rule 118) — regenerated identically to the numpy reference.
# ---------------------------------------------------------------------------
def coords_grid(h: int, w: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def curvelet_B(ns: int, no0: int, f0: float, base: float, niso: int, max_freq: float | None) -> np.ndarray:
    cols: list[list[float]] = []
    for j in range(int(ns)):
        f_j = float(f0) * (float(base) ** j)
        l_j = int(no0) * (2 ** (j // 2))
        for l in range(l_j):
            th = np.pi * l / l_j
            cols.append([f_j * np.cos(th), f_j * np.sin(th)])
    for i in range(int(niso)):
        th = np.pi * i / max(int(niso), 1)
        fl = float(f0) * 0.5
        cols.append([fl * np.cos(th), fl * np.sin(th)])
    B = np.asarray(cols, np.float32).T  # (2, n)
    if max_freq is not None:
        nrm = np.sqrt((B.astype(np.float64) ** 2).sum(0))
        keep = nrm <= float(max_freq) + 1e-6
        if not keep.any():
            keep = nrm <= float(nrm.min()) + 1e-6
        B = B[:, keep]
    return B


def curvelet_feats(coords: np.ndarray, B: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # benign macOS Accelerate matmul warns
        proj = (2.0 * np.pi) * (np.asarray(coords, np.float64) @ np.asarray(B, np.float64))
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


def dir_feats(coords: np.ndarray, argmax_hw: np.ndarray, n_freqs: int, fa: float, fc: float, tau: float) -> np.ndarray:
    """Self-orient directional feats: tangent of the decoder's OWN all-class argmax boundary ->
    oriented Fourier feats. Bit-mirror of the numpy inflate ``_dir_feats`` (scipy EDT, CPU).
    Cheap relative to the per-pixel forward; kept on CPU so the device path stays simple."""
    from scipy import ndimage

    a = np.asarray(argmax_hw)
    h, w = a.shape
    b = np.zeros(a.shape, bool)
    b[:-1, :] |= a[:-1, :] != a[1:, :]
    b[1:, :] |= a[:-1, :] != a[1:, :]
    b[:, :-1] |= a[:, :-1] != a[:, 1:]
    b[:, 1:] |= a[:, :-1] != a[:, 1:]
    if not b.any():
        tx = np.ones(h * w, np.float32)
        ty = np.zeros(h * w, np.float32)
    else:
        dist = ndimage.distance_transform_edt(~b).astype(np.float32)
        gy = np.zeros_like(dist)
        gx = np.zeros_like(dist)
        gy[1:-1, :] = (dist[2:, :] - dist[:-2, :]) * 0.5
        gx[:, 1:-1] = (dist[:, 2:] - dist[:, :-2]) * 0.5
        tx = -gy.ravel()
        ty = gx.ravel()
        nrm = np.sqrt(tx * tx + ty * ty)
        flat = nrm < 1e-6
        nrm = np.maximum(nrm, 1e-8)
        tx = tx / nrm
        ty = ty / nrm
        tx[flat] = 1.0
        ty[flat] = 0.0
    cx = coords[:, 0]
    cy = coords[:, 1]
    nx = ty
    ny = -tx
    u_t = cx * tx + cy * ty
    u_n = cx * nx + cy * ny
    tp = 2.0 * np.pi
    feats = []
    for k in range(int(n_freqs)):
        a_f = float(fa) * (2.0 ** k)
        c_f = float(fc) * (2.0 ** k)
        feats += [np.sin(tp * a_f * u_t), np.cos(tp * a_f * u_t), np.sin(tp * c_f * u_n), np.cos(tp * c_f * u_n)]
    return np.stack(feats, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# TORCH forward (device-flexible). The 81%-cost per-pixel matmul+activation path.
# ---------------------------------------------------------------------------
def _torch_act(u, kind: str, w0: float, s0: float, beta: float, omega: float):
    import torch

    if kind == "wire":
        return torch.cos(w0 * u) * torch.exp(-((s0 * u) ** 2))
    if kind == "hosc":
        return torch.tanh(beta * torch.sin(omega * u))
    return torch.clamp(u, min=0.0)


def torch_in_proj_h0(P: dict, feats_t, m: dict):
    """h0 = act(feats @ in_proj.weight.T + b). Depends ONLY on feats (NOT code) -> shared by a
    pair's two final frames. ``P`` is a dict of torch tensors (on device, of the working dtype)."""
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    return _torch_act(feats_t @ P["in_proj.weight"].T + P["in_proj.bias"], *kw)


def torch_outputs_from_h0(P: dict, h0, code_row_t, m: dict, want_rgb: bool):
    """Op-for-op mirror of ``_outputs_from_h0`` (numpy) AFTER in_proj, in torch.
    want_rgb=False -> return (phi, None), skipping the rgb head (argmax(phi) is identical) —
    used by the self-orient fixed point."""
    import torch

    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    nH = int(m["n_hidden"])
    hd = int(m["hidden_dim"])
    film = (code_row_t @ P["film.weight"].T + P["film.bias"]).reshape(nH, 2, hd)
    h = h0
    for li in range(nH):
        h = _torch_act(
            (h @ P["hidden.%d.weight" % li].T + P["hidden.%d.bias" % li]) * (1.0 + film[li, 0]) + film[li, 1], *kw
        )
    phi = h @ P["out_sdf.weight"].T + P["out_sdf.bias"]
    if not want_rgb:
        return phi, None
    tex = h @ P["out_tex.weight"].T + P["out_tex.bias"]
    z = phi / float(m["softmax_temp"])
    z = z - z.max(-1, keepdim=True).values
    soft = torch.exp(z)
    soft = soft / soft.sum(-1, keepdim=True)
    rgb = torch.sigmoid(soft @ P["palette"] + tex) * 255.0
    if not m["chroma"]:
        luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        rgb = torch.cat([luma, luma, luma], dim=-1)
    return phi, rgb


def torch_R(rgb_t, rh: int, rw: int, ch: int, cw: int) -> np.ndarray:
    """Contest R: bicubic up render->camera, round, clamp -> uint8 numpy. ALWAYS fp32 (matches the
    numpy reference ``_R`` which casts ``.float()`` before interpolate), so R is identical across
    the numpy-fp64 and torch-fp32 paths; only the pre-R rgb arithmetic differs."""
    import torch

    x = rgb_t.reshape(rh, rw, 3).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Full decode (mirrors the numpy inflate loop). Writes the .raw and/or returns frames + argmax.
# ---------------------------------------------------------------------------
def decode_levelset_torch(
    manifest: dict[str, Any],
    params_np: dict[str, np.ndarray],
    code_np: np.ndarray,
    *,
    device: str = "cpu",
    dtype: str = "fp32",
    max_pairs: int | None = None,
    dst_raw: str | Path | None = None,
    return_frames: bool = False,
    collect_phi_argmax: bool = False,
):
    """Torch level-set decode. Returns a dict with ``n_frames``, optional ``frames`` (list of
    (f0,f1) uint8 camera arrays per pair) and optional ``phi_argmax`` (final-frame witness argmax,
    render-res, per pair) for parity. ``device`` in {cpu,cuda}; ``dtype`` in {fp32,fp64}.
    """
    import torch

    td = torch.float64 if dtype == "fp64" else torch.float32
    dev = torch.device(device)
    m = dict(manifest)
    rh, rw = int(m["render_h"]), int(m["render_w"])
    ch, cw = int(m["camera_h"]), int(m["camera_w"])
    n_pairs = int(m["n_pairs"]) if max_pairs is None else min(int(max_pairs), int(m["n_pairs"]))

    coords = coords_grid(rh, rw)
    B = curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"], m["bank_n_iso"], m["max_bank_freq"])
    curv = curvelet_feats(coords, B)  # (HW, curv_w) fp32, numpy (identical to ref)

    P = {k: torch.as_tensor(np.asarray(v), dtype=td, device=dev) for k, v in params_np.items()}
    code = torch.as_tensor(np.asarray(code_np), dtype=td, device=dev)

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    phi_argmax: list[np.ndarray] = []
    f_out = open(dst_raw, "wb") if dst_raw is not None else None
    try:
        for pi in range(n_pairs):
            if m["self_orient"]:
                dirf = np.zeros((curv.shape[0], 4 * int(m["n_dir_freqs"])), np.float32)
                prev_am = None
                for _ in range(int(m["so_iters"])):
                    feats = np.concatenate([curv, dirf], axis=-1)
                    feats_t = torch.as_tensor(feats, dtype=td, device=dev)
                    h0 = torch_in_proj_h0(P, feats_t, m)
                    phi, _ = torch_outputs_from_h0(P, h0, code[2 * pi + 1], m, False)
                    am = phi.argmax(-1).reshape(rh, rw).cpu().numpy().astype(np.int64)
                    if prev_am is not None and np.array_equal(am, prev_am):
                        break  # argmax fixed point -> remaining iters no-ops (matches numpy early-stop)
                    dirf = dir_feats(coords, am, m["n_dir_freqs"], m["so_freq_along"], m["so_freq_across"], m["so_tau"])
                    prev_am = am
                feats = np.concatenate([curv, dirf], axis=-1)
            else:
                feats = curv
            feats_t = torch.as_tensor(feats, dtype=td, device=dev)
            h0 = torch_in_proj_h0(P, feats_t, m)  # shared across the pair's 2 frames
            pair_frames = []
            for fk in range(2):
                phi, rgb = torch_outputs_from_h0(P, h0, code[2 * pi + fk], m, True)
                frame = torch_R(rgb, rh, rw, ch, cw)
                if fk == 1 and collect_phi_argmax:
                    phi_argmax.append(phi.argmax(-1).reshape(rh, rw).cpu().numpy().astype(np.int64))
                if f_out is not None:
                    f_out.write(frame.tobytes())
                if return_frames:
                    pair_frames.append(frame)
            if return_frames:
                frames.append((pair_frames[0], pair_frames[1]))
    finally:
        if f_out is not None:
            f_out.close()

    res: dict[str, Any] = {"n_frames": 2 * n_pairs, "n_pairs": n_pairs, "device": device, "dtype": dtype}
    if return_frames:
        res["frames"] = frames
    if collect_phi_argmax:
        res["phi_argmax"] = phi_argmax
    return res


def inflate(src_bin: str | Path, dst_raw: str | Path, *, device: str = "cpu", dtype: str = "fp32") -> dict[str, Any]:
    """Drop-in torch inflate: read the LVLS1 blob -> write the full (2*n_pairs, camera) uint8 .raw.
    Identical archive bytes + witness layout as the numpy inflate; only the decode arithmetic
    differs (parity-gated). Generic FREE code (rule 118)."""
    import brotli

    m, base_b, code_b, _pose = read_levelset_blob(src_bin)
    params = dequant_params(brotli.decompress(base_b), m["base_param_order"], m["base_shapes"], m["base_scales"])
    code = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32) * float(m["code_scale"])).reshape(m["code_shape"])
    return decode_levelset_torch(m, params, code, device=device, dtype=dtype, dst_raw=dst_raw)


# ---------------------------------------------------------------------------
# Emitter: the self-contained torch inflate.py the byte-close tool can ship (deliverable 5).
# Kept as a string here so it does NOT collide with the a7613e19 numpy template; the byte-close
# tool would add a ``--torch-inflate`` flag that writes THIS instead of (or alongside) the numpy
# one. GENERIC decode code -> FREE (rule 118); archive bytes UNCHANGED; parity-gated to numpy.
# ---------------------------------------------------------------------------
TORCH_INFLATE_PY = r'''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# LEVEL-SET witness inflate -- TORCH fp32, device-flexible (CPU or CUDA/T4), self-contained.
# Emits the FULL (2*n_pairs, camera_h, camera_w, 3) uint8 .raw. Reads the SAME LVLS1 int8+brotli
# blob the numpy inflate reads (archive UNCHANGED). The curvelet bank + (when used) self-orient
# directional feats are regenerated for FREE (rule 118). torch decode is the FAST shippable path;
# the numpy-fp64 inflate is the bit-identical reference/authority (parity-gated, argmax-exact).
# Device: honors ${INFLATE_DEVICE} (default: cuda if available else cpu). NO MPS.
import os, sys, json, struct
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
    B = np.asarray(cols, np.float32).T
    if max_freq is not None:
        nrm = np.sqrt((B.astype(np.float64) ** 2).sum(0)); keep = nrm <= float(max_freq) + 1e-6
        if not keep.any(): keep = nrm <= float(nrm.min()) + 1e-6
        B = B[:, keep]
    return B


def _curvelet_feats(coords, B):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # benign macOS Accelerate matmul warns
        proj = (2.0 * np.pi) * (np.asarray(coords, np.float64) @ np.asarray(B, np.float64))
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


def _act(u, kind, w0, s0, beta, omega):
    if kind == "wire": return torch.cos(w0 * u) * torch.exp(-((s0 * u) ** 2))
    if kind == "hosc": return torch.tanh(beta * torch.sin(omega * u))
    return torch.clamp(u, min=0.0)


def _in_proj_h0(P, feats_t, m):
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    return _act(feats_t @ P["in_proj.weight"].T + P["in_proj.bias"], *kw)


def _outputs_from_h0(P, h0, code_row, m, want_rgb):
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    nH, hd = int(m["n_hidden"]), int(m["hidden_dim"])
    film = (code_row @ P["film.weight"].T + P["film.bias"]).reshape(nH, 2, hd)
    h = h0
    for li in range(nH):
        h = _act((h @ P["hidden.%d.weight" % li].T + P["hidden.%d.bias" % li]) * (1.0 + film[li, 0]) + film[li, 1], *kw)
    phi = h @ P["out_sdf.weight"].T + P["out_sdf.bias"]
    if not want_rgb:
        return phi, None
    tex = h @ P["out_tex.weight"].T + P["out_tex.bias"]
    z = phi / float(m["softmax_temp"]); z = z - z.max(-1, keepdim=True).values
    soft = torch.exp(z); soft = soft / soft.sum(-1, keepdim=True)
    rgb = torch.sigmoid(soft @ P["palette"] + tex) * 255.0
    if not m["chroma"]:
        luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        rgb = torch.cat([luma, luma, luma], dim=-1)
    return phi, rgb


def _dir_feats(coords, argmax_hw, n_freqs, fa, fc, tau):
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


def _R(rgb_t, rh, rw, ch, cw):
    x = rgb_t.reshape(rh, rw, 3).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    dev = os.environ.get("INFLATE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    assert dev != "mps", "MPS is NEVER an inflate device (CLAUDE.md)"
    device = torch.device(dev)
    m, base_b, code_b, _pose = _read_blob(src)
    params = _dequant(brotli.decompress(base_b), m["base_param_order"], m["base_shapes"], m["base_scales"])
    code_np = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32) * float(m["code_scale"])).reshape(m["code_shape"])
    rh, rw, ch, cw = int(m["render_h"]), int(m["render_w"]), int(m["camera_h"]), int(m["camera_w"])
    coords = _coords(rh, rw)
    B = _curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"], m["bank_n_iso"], m["max_bank_freq"])
    curv = _curvelet_feats(coords, B)
    P = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in params.items()}
    code = torch.as_tensor(code_np, dtype=torch.float32, device=device)
    n_pairs = int(m["n_pairs"])
    with open(dst, "wb") as f:
        for pi in range(n_pairs):
            if m["self_orient"]:
                dirf = np.zeros((curv.shape[0], 4 * int(m["n_dir_freqs"])), np.float32)
                prev_am = None
                for _ in range(int(m["so_iters"])):
                    feats_t = torch.as_tensor(np.concatenate([curv, dirf], axis=-1), dtype=torch.float32, device=device)
                    phi, _ = _outputs_from_h0(P, _in_proj_h0(P, feats_t, m), code[2 * pi + 1], m, False)
                    am = phi.argmax(-1).reshape(rh, rw).cpu().numpy().astype(np.int64)
                    if prev_am is not None and np.array_equal(am, prev_am):
                        break
                    dirf = _dir_feats(coords, am, m["n_dir_freqs"], m["so_freq_along"], m["so_freq_across"], m["so_tau"])
                    prev_am = am
                feats = np.concatenate([curv, dirf], axis=-1)
            else:
                feats = curv
            h0 = _in_proj_h0(P, torch.as_tensor(feats, dtype=torch.float32, device=device), m)
            for fk in range(2):
                _phi, rgb = _outputs_from_h0(P, h0, code[2 * pi + fk], m, True)
                f.write(_R(rgb, rh, rw, ch, cw).tobytes())
    print("inflated %d frames (%d pairs) on %s -> %s" % (2 * n_pairs, n_pairs, dev, dst), flush=True)


if __name__ == "__main__":
    main()
'''


def render_torch_inflate_py() -> str:
    """Return the self-contained torch inflate.py source (deliverable 5 emitter)."""
    return TORCH_INFLATE_PY


__all__ = [
    "MAGIC",
    "CAMERA_H",
    "CAMERA_W",
    "read_levelset_blob",
    "dequant_params",
    "coords_grid",
    "curvelet_B",
    "curvelet_feats",
    "dir_feats",
    "torch_in_proj_h0",
    "torch_outputs_from_h0",
    "torch_R",
    "decode_levelset_torch",
    "inflate",
    "TORCH_INFLATE_PY",
    "render_torch_inflate_py",
]
