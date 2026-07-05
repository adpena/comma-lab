#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""FEED-ej harness: torch level-set inflate PARITY (vs numpy-fp64 authority) + CPU TIMING + T4 projection.

$0 CPU. NO MPS, NO paid eval. Builds a SYNTHETIC level-set checkpoint at the OOM'd row dims
(n_hidden=4 hidden=96 mod=32 in_feat=88 -> curvelet 80 + self-orient 8 [n_dir_freqs=2], 5-class
hosc, chroma, render 384x512), byte-closes it via the canonical ``tools/levelset_byte_close_and_eval``
(real int8+brotli archive.zip + the SHIPPED numpy inflate.py), then:

  1. PARITY: torch-fp32 decode vs the SHIPPED numpy-fp64 inflate .raw (the bit-identical authority):
       * uint8 frame parity (% pixels identical, max abs diff);
       * witness phi.argmax parity (the decode's own class partition);
       * SegNet argmax agreement on the rendered frames (THE d_seg-relevant metric) -- frozen
         CPU-torch SegNet on the numpy frames vs the torch frames;
       * d_pose MSE delta -- frozen CPU-torch PoseNet pose vector on the numpy pair vs the torch pair.
     Also runs torch-fp64 as a port-correctness gate (should be ~exact vs numpy-fp64).
  2. TIMING: in-process numpy-fp64 decode (CERTIFIED byte-identical to the shipped inflate) vs
     torch-fp32 CPU decode, per-pair, with OMP/torch threads = 4 (contest 4-core CPU) -> the CPU
     speedup + the n600 extrapolation vs the FEED-eg 53-61 min baseline.
  3. T4 PROJECTION: analytic projection of the torch-fp32 forward on a T4 (the transcendental-
     dominated per-pixel forward is embarrassingly parallel) + the exact Modal T4 smoke command.

SYNTHETIC-FIXTURE HONESTY (NO-FAKE class #3): the parity numbers here are a DECODE-LEVEL numerical
parity (torch-fp32 vs numpy-fp64 of the SAME weights) -- a property of the ARITHMETIC, not of any
trained witness. Random hosc weights give a busy, small-margin-heavy argmax -> a CONSERVATIVE
(upper-bound) flip stress test. The realized d_seg/d_pose of the actual witness is re-measured on
the FIRST real ckpt. This harness proves the torch decode is parity-faithful + fast, NOT a score.

AUTHORITY: numpy-fp64 = bit-identical reference. torch = fast decode (parity-gated), NEVER an
authority. ``[macOS-CPU advisory] NON-PROMOTABLE``. pointer UNMOVED 0.19110.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "tools", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import levelset_byte_close_and_eval as lbc  # noqa: E402  (canonical byte-close + numpy inflate authority)
from tac.local_acceleration import torch_levelset_inflate as tli  # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164
_FRAME_BYTES = CAMERA_H * CAMERA_W * 3


def _pick_npz(ckpt_dir: Path) -> Path:
    """Prefer *_ema_mlx.npz then *_live_mlx.npz (same precedence as levelset_byte_close_and_eval)."""
    ckpt_dir = Path(ckpt_dir)
    for name in ("levelset_witness_ema_mlx.npz", "levelset_witness_live_mlx.npz"):
        p = ckpt_dir / name
        if p.exists():
            return p
    cands = sorted(ckpt_dir.glob("*_mlx.npz"))
    if not cands:
        raise FileNotFoundError(f"no level-set *_mlx.npz in {ckpt_dir}")
    return cands[0]


# ---------------------------------------------------------------------------
# synthetic checkpoint at the row dims (non-degenerate argmax for a real parity stress)
# ---------------------------------------------------------------------------
def build_synthetic_ckpt(ckpt_dir: Path, *, n_pairs: int, seed: int = 0) -> Path:
    """Write a synthetic ``levelset_witness_ema_mlx.npz`` (params + __cfg/__bank/__render scalars)
    at the OOM'd row config. Weights scaled so hosc(tanh(4*sin(u))) produces a varied multi-class
    argmax (a genuine boundary-bearing partition, not a constant field)."""
    rng = np.random.default_rng(seed)
    hidden, n_hidden, mod, n_classes = 96, 4, 32, 5
    # curvelet default bank (n_scales4 n_orient0=6 f0=2 base=2 n_iso4) -> 40 cols -> 80 curv feats;
    # self-orient n_dir_freqs=2 -> +8 -> in_feat=88 (the row).
    in_feat = 88

    def g(shape, s):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    params = {
        "in_proj.weight": g((hidden, in_feat), 1.0),   # large -> sin(u) fully varies across pixels
        "in_proj.bias": g((hidden,), 0.3),
        "film.weight": g((n_hidden * 2 * hidden, mod), 0.1),
        "film.bias": g((n_hidden * 2 * hidden,), 0.05),
        "out_sdf.weight": g((n_classes, hidden), 1.0 / np.sqrt(hidden)),
        "out_sdf.bias": g((n_classes,), 0.2),
        "out_tex.weight": g((3, hidden), 1.0 / np.sqrt(hidden)),
        "out_tex.bias": g((3,), 0.2),
        "palette": g((n_classes, 3), 1.0),
        "code": g((2 * n_pairs, mod), 0.6),
    }
    for li in range(n_hidden):
        params["hidden.%d.weight" % li] = g((hidden, hidden), 1.0 / np.sqrt(hidden))
        params["hidden.%d.bias" % li] = g((hidden,), 0.1)

    flat: dict[str, Any] = {k: np.asarray(v, np.float32) for k, v in params.items()}
    flat["__cfg_hidden_dim"] = np.asarray(hidden)
    flat["__cfg_n_hidden"] = np.asarray(n_hidden)
    flat["__cfg_activation"] = np.asarray("hosc")
    flat["__cfg_softmax_temp"] = np.asarray(0.05)
    flat["__cfg_chroma"] = np.asarray(1)
    flat["__cfg_wire_w0"] = np.asarray(20.0)
    flat["__cfg_wire_s0"] = np.asarray(10.0)
    flat["__cfg_hosc_beta"] = np.asarray(4.0)
    flat["__cfg_hosc_omega"] = np.asarray(1.0)
    flat["__bank_n_scales"] = np.asarray(4)
    flat["__bank_n_orient0"] = np.asarray(6)
    flat["__bank_f0"] = np.asarray(2.0)
    flat["__bank_base"] = np.asarray(2.0)
    flat["__bank_n_iso"] = np.asarray(4)
    flat["__cfg_max_bank_freq"] = np.asarray(-1.0)  # None
    flat["__render_hw"] = np.asarray([384, 512])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "levelset_witness_ema_mlx.npz"
    np.savez(out, **flat)
    return out


# ---------------------------------------------------------------------------
# in-process numpy-fp64 decode (mirror of the SHIPPED template) -- for CERTIFIED timing
# ---------------------------------------------------------------------------
def _np_act(u, kind, w0, s0, beta, omega):
    u = np.asarray(u, np.float64)
    if kind == "wire":
        return (np.cos(w0 * u) * np.exp(-((s0 * u) ** 2))).astype(np.float32)
    if kind == "hosc":
        return np.tanh(beta * np.sin(omega * u)).astype(np.float32)
    return np.maximum(u, 0.0).astype(np.float32)


def _np_in_proj_h0(P, feats, m):
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _np_act(np.asarray(feats, np.float64) @ P["in_proj.weight"].T + P["in_proj.bias"], *kw)


def _np_outputs_from_h0(P, h0, code_row, m, want_rgb):
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    cr = np.asarray(code_row, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        film = (cr @ P["film.weight"].T + P["film.bias"]).reshape(m["n_hidden"], 2, m["hidden_dim"])
        h = h0
        for li in range(m["n_hidden"]):
            h = _np_act((h @ P["hidden.%d.weight" % li].T + P["hidden.%d.bias" % li]) * (1.0 + film[li, 0]) + film[li, 1], *kw)
        phi = h @ P["out_sdf.weight"].T + P["out_sdf.bias"]
        if not want_rgb:
            return phi.astype(np.float32), None
        tex = h @ P["out_tex.weight"].T + P["out_tex.bias"]
        z = phi / float(m["softmax_temp"])
        z = z - z.max(-1, keepdims=True)
        soft = np.exp(z)
        soft = soft / soft.sum(-1, keepdims=True)
        rgb = (1.0 / (1.0 + np.exp(-(soft @ P["palette"] + tex)))) * 255.0
        if not m["chroma"]:
            luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb = np.concatenate([luma, luma, luma], axis=-1)
    return phi.astype(np.float32), rgb.astype(np.float32)


def numpy_decode_inproc(m, params_np, code_np, *, max_pairs, dst_raw=None):
    """numpy-fp64 in-process decode mirroring the shipped inflate template (incl. self-orient
    early-stop + share-h0 + skip-rgb-head). Certified byte-identical to the subprocess inflate."""
    rh, rw, ch, cw = int(m["render_h"]), int(m["render_w"]), int(m["camera_h"]), int(m["camera_w"])
    n_pairs = min(int(max_pairs), int(m["n_pairs"]))
    coords = tli.coords_grid(rh, rw)
    B = tli.curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"], m["bank_n_iso"], m["max_bank_freq"])
    curv = tli.curvelet_feats(coords, B)
    P = {k: np.asarray(v, np.float64) for k, v in params_np.items()}
    out = bytearray() if dst_raw is None else None
    f = open(dst_raw, "wb") if dst_raw is not None else None
    try:
        for pi in range(n_pairs):
            if m["self_orient"]:
                dirf = np.zeros((curv.shape[0], 4 * int(m["n_dir_freqs"])), np.float32)
                prev_am = None
                for _ in range(int(m["so_iters"])):
                    feats = np.concatenate([curv, dirf], axis=-1)
                    phi, _ = _np_outputs_from_h0(P, _np_in_proj_h0(P, feats, m), code_np[2 * pi + 1], m, False)
                    am = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
                    if prev_am is not None and np.array_equal(am, prev_am):
                        break
                    dirf = tli.dir_feats(coords, am, m["n_dir_freqs"], m["so_freq_along"], m["so_freq_across"], m["so_tau"])
                    prev_am = am
                feats = np.concatenate([curv, dirf], axis=-1)
            else:
                feats = curv
            h0 = _np_in_proj_h0(P, feats, m)
            for fk in range(2):
                _phi, rgb = _np_outputs_from_h0(P, h0, code_np[2 * pi + fk], m, True)
                frame = tli.torch_R(_to_torch(rgb), rh, rw, ch, cw)
                if f is not None:
                    f.write(frame.tobytes())
                else:
                    out += frame.tobytes()
    finally:
        if f is not None:
            f.close()
    return bytes(out) if out is not None else None


def _to_torch(a):
    import torch

    return torch.from_numpy(np.ascontiguousarray(a)).float()


# ---------------------------------------------------------------------------
# parity helpers
# ---------------------------------------------------------------------------
def _read_raw_frames(path: Path, n_pairs: int):
    f0s, f1s = [], []
    with open(path, "rb") as fh:
        for _ in range(n_pairs):
            f0s.append(np.frombuffer(fh.read(_FRAME_BYTES), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
            f1s.append(np.frombuffer(fh.read(_FRAME_BYTES), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
    return f0s, f1s


def _segnet_argmax_maps(seg_cpu, frames1):
    import torch

    arr = np.stack([np.asarray(f)[None] for f in frames1], axis=0)  # (N,1,H,W,3)
    xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        return seg_cpu(seg_cpu.preprocess_input(xp)).argmax(dim=1).cpu().numpy().astype(np.int64)  # (N,h,w)


def _posenet_vecs(posenet_cpu, f0s, f1s):
    import einops
    import torch

    arr = np.stack([np.stack([np.asarray(a), np.asarray(b)], axis=0) for a, b in zip(f0s, f1s)], axis=0)
    x = einops.rearrange(torch.from_numpy(arr).float(), "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        out = posenet_cpu(posenet_cpu.preprocess_input(x))
        pose = out["pose"] if isinstance(out, dict) else out
        half = pose.shape[-1] // 2
        for hh in posenet_cpu.hydra.heads:
            if hh.name == "pose":
                half = hh.out // 2
                break
        return pose[:, :half].cpu().numpy().astype(np.float64)  # (N, half)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity-pairs", type=int, default=3, help="pairs for the full parity decode (default 3)")
    ap.add_argument("--time-pairs", type=int, default=2, help="pairs for the per-pair timing (default 2)")
    ap.add_argument("--n600", type=int, default=600, help="extrapolation target pair count (default 600)")
    ap.add_argument("--gt-cache", type=str, default=str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"))
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (contest 4-core); 0=leave default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=str, default=None,
                    help="REAL trained level-set run dir (has levelset_witness_ema_mlx.npz). When set, "
                         "byte-close the ACTUAL weights instead of a synthetic fixture (NO-FAKE #3 "
                         "real-weight parity gate). Default None = synthetic fixture (arithmetic parity).")
    ap.add_argument("--so-tau", type=float, default=4.0, help="self-orient tau (persisted-gap default 4).")
    ap.add_argument("--so-iters", type=int, default=4, help="self-orient fixed-point iters (default 4).")
    ap.add_argument("--keep-packet", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.threads > 0:
        import torch

        torch.set_num_threads(int(args.threads))

    work = _REPO / "experiments" / "results" / f"levelset_torch_parity_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    # REAL-WEIGHT path (--ckpt-dir): byte-close an actual trained level-set checkpoint (NOT a
    # synthetic fixture) so the parity+timing reflect the WITNESS score path, per NO-FAKE #3
    # ("re-measure on the FIRST real ckpt"). so_overrides freq_across/freq_along are read from the
    # npz __cfg (persisted); tau/iters default to the trainer/byte-close CLI defaults (4/4, the
    # trainer-gap-not-persisted scalars). Synthetic path unchanged (default).
    synthetic = args.ckpt_dir is None
    if synthetic:
        ckpt_dir = work / "ckpt"
        n_total = max(args.parity_pairs, args.time_pairs)
        build_synthetic_ckpt(ckpt_dir, n_pairs=n_total, seed=args.seed)
        so_overrides = {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4}
    else:
        ckpt_dir = Path(args.ckpt_dir)
        _npz = np.load(_pick_npz(ckpt_dir))
        so_overrides = {
            "freq_across": float(_npz["__cfg_freq_across"]) if "__cfg_freq_across" in _npz else 32.0,
            "freq_along": float(_npz["__cfg_freq_along"]) if "__cfg_freq_along" in _npz else 4.0,
            "tau": float(args.so_tau), "iters": int(args.so_iters),
        }

    # --- byte-close + SHIPPED numpy inflate (authority .raw) at parity_pairs ---
    packet_dir = work / "packet"
    rep = lbc.run(
        ckpt_dir, npz_name=None, max_pairs=args.parity_pairs, fold_pose_sidecar=False,
        pose_sidecar_path=None, gt_cache=None, keep_packet=True, packet_dir=packet_dir,
        skip_parity=True,
        so_overrides=so_overrides,
    )
    numpy_raw = Path(rep["inflate"]["raw_path"])  # shipped numpy-fp64 .raw (authority)
    capped_0bin = packet_dir / "archive" / "0.bin"  # capped to parity_pairs by run_inflate
    m, base_b, code_b, _pose = tli.read_levelset_blob(capped_0bin)
    import brotli

    params_np = tli.dequant_params(brotli.decompress(base_b), m["base_param_order"], m["base_shapes"], m["base_scales"])
    code_np = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32) * float(m["code_scale"])).reshape(m["code_shape"])

    # --- torch decode (fp32 + fp64) on the SAME capped 0.bin ---
    torch_fp32_raw = packet_dir / "torch_fp32.raw"
    torch_fp64_raw = packet_dir / "torch_fp64.raw"
    r32 = tli.decode_levelset_torch(m, params_np, code_np, device="cpu", dtype="fp32", max_pairs=args.parity_pairs, dst_raw=torch_fp32_raw, collect_phi_argmax=True)
    _r64 = tli.decode_levelset_torch(m, params_np, code_np, device="cpu", dtype="fp64", max_pairs=args.parity_pairs, dst_raw=torch_fp64_raw)

    # --- numpy in-process decode CERTIFY vs shipped (same math) ---
    np_inproc_raw = packet_dir / "numpy_inproc.raw"
    numpy_decode_inproc(m, params_np, code_np, max_pairs=args.parity_pairs, dst_raw=np_inproc_raw)
    certify_np = (np_inproc_raw.read_bytes() == numpy_raw.read_bytes())

    # --- parity: uint8 frames ---
    a_np = np.frombuffer(numpy_raw.read_bytes(), dtype=np.uint8)
    a32 = np.frombuffer(torch_fp32_raw.read_bytes(), dtype=np.uint8)
    a64 = np.frombuffer(torch_fp64_raw.read_bytes(), dtype=np.uint8)
    n = min(a_np.size, a32.size, a64.size)
    a_np, a32, a64 = a_np[:n].astype(np.int16), a32[:n].astype(np.int16), a64[:n].astype(np.int16)
    px_match_32 = float(np.mean(a_np == a32))
    px_match_64 = float(np.mean(a_np == a64))
    max_abs_32 = int(np.max(np.abs(a_np - a32)))
    max_abs_64 = int(np.max(np.abs(a_np - a64)))

    # --- argmax class distribution sanity (witness phi argmax) ---
    am32 = r32.get("phi_argmax", [])
    class_hist = {}
    if am32:
        u, c = np.unique(np.concatenate([a.ravel() for a in am32]), return_counts=True)
        class_hist = {int(k): float(v) / float(sum(c)) for k, v in zip(u, c)}

    # --- SegNet argmax agreement + PoseNet pose MSE delta (frozen CPU-torch) ---
    import train_witness_realized_through_R_mlx as twr  # noqa: E402

    gt, seg_cpu, posenet_cpu = twr.load_gt_from_cache(Path(args.gt_cache), args.parity_pairs)
    np_f0, np_f1 = _read_raw_frames(numpy_raw, args.parity_pairs)
    t_f0, t_f1 = _read_raw_frames(torch_fp32_raw, args.parity_pairs)
    seg_np = _segnet_argmax_maps(seg_cpu, np_f1)
    seg_t = _segnet_argmax_maps(seg_cpu, t_f1)
    seg_agree = float(np.mean(seg_np == seg_t))
    seg_flip_px = int(np.count_nonzero(seg_np != seg_t))
    seg_total_px = int(seg_np.size)
    pose_np = _posenet_vecs(posenet_cpu, np_f0, np_f1)
    pose_t = _posenet_vecs(posenet_cpu, t_f0, t_f1)
    pose_mse_delta = float(np.mean((pose_np - pose_t) ** 2))
    pose_max_abs = float(np.max(np.abs(pose_np - pose_t)))
    # context: typical pose magnitude (so the delta is interpretable vs the d_pose scale)
    pose_ref_rms = float(np.sqrt(np.mean(pose_np ** 2)))

    # --- TIMING: numpy-fp64 vs torch-fp32, per-pair (warm) ---
    tp = args.time_pairs
    _ = numpy_decode_inproc(m, params_np, code_np, max_pairs=1, dst_raw=None)  # warm
    t0 = time.perf_counter()
    numpy_decode_inproc(m, params_np, code_np, max_pairs=tp, dst_raw=None)
    np_secs = time.perf_counter() - t0
    _ = tli.decode_levelset_torch(m, params_np, code_np, device="cpu", dtype="fp32", max_pairs=1)  # warm
    t0 = time.perf_counter()
    tli.decode_levelset_torch(m, params_np, code_np, device="cpu", dtype="fp32", max_pairs=tp)
    torch_secs = time.perf_counter() - t0
    np_per_pair = np_secs / tp
    torch_per_pair = torch_secs / tp
    speedup = np_per_pair / max(torch_per_pair, 1e-9)
    np_n600_min = np_per_pair * args.n600 / 60.0
    torch_n600_min = torch_per_pair * args.n600 / 60.0

    # --- T4 projection (analytic) ---
    # The per-pair forward is ~(so_iters_eff + 2) forwards of [in_proj + 4 hidden + heads] over
    # HW=render_h*render_w pixels x hidden=96, dominated by tanh(sin(.)) transcendentals. On a T4
    # (~8 TFLOP fp32, thousands of CUDA cores) the (HW x 96) elementwise transcendentals + small
    # matmuls run in ~ms/forward vs ~100s of ms/forward on a 4-core CPU. Conservative GPU speedup
    # for this memory-light, transcendental-bound, perfectly-parallel profile: 15x-50x over
    # torch-CPU-4thread. Projection band below uses [15x, 50x].
    t4_lo_min = torch_n600_min / 50.0
    t4_hi_min = torch_n600_min / 15.0

    legal_cpu_torch = bool(torch_n600_min < 30.0)
    legal_cpu_numpy = bool(np_n600_min < 30.0)
    legal_t4 = bool(t4_hi_min < 30.0)  # legal even at the conservative (15x) end

    modal_t4_cmd = (
        "modal run experiments/modal_levelset_torch_t4_smoke.py "  # (to-build; mirror modal_cpu eval infra w/ gpu='T4')
        f"--archive {packet_dir/'archive.zip'} --inflate-py <torch_inflate.py> --n-pairs {args.parity_pairs} "
        "# INFLATE_DEVICE=cuda; measures per-pair T4 time + confirms CUDA decode runs; <$0.20"
    )

    report: dict[str, Any] = {
        "tool": "levelset_torch_inflate_parity",
        "feed": "FEED-ej",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE (numpy-fp64=bit-identical reference; torch=fast decode, parity-gated)",
        "promotion_claim": False,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "synthetic_fixture": synthetic,
        "ckpt_dir": (None if synthetic else str(ckpt_dir)),
        "row_config": {"n_hidden": 4, "hidden": 96, "mod": 32, "in_feat": int(m["in_feat"]) if "in_feat" in m else 88,
                       "n_classes": 5, "activation": m["activation"], "self_orient": bool(m["self_orient"]),
                       "n_dir_freqs": int(m["n_dir_freqs"]), "render": [int(m["render_h"]), int(m["render_w"])],
                       "chroma": bool(m["chroma"])},
        "parity_pairs": args.parity_pairs,
        "certify_numpy_inproc_eq_shipped": bool(certify_np),
        "witness_argmax_class_frac": class_hist,
        "parity": {
            "uint8_frame_pixel_match_torch_fp32_vs_numpy_fp64": px_match_32,
            "uint8_frame_max_abs_diff_torch_fp32": max_abs_32,
            "uint8_frame_pixel_match_torch_fp64_vs_numpy_fp64": px_match_64,
            "uint8_frame_max_abs_diff_torch_fp64": max_abs_64,
            "segnet_argmax_agreement_torch_fp32_vs_numpy_fp64": seg_agree,
            "segnet_argmax_flip_px": seg_flip_px,
            "segnet_argmax_total_px": seg_total_px,
            "d_pose_mse_delta_torch_fp32_vs_numpy_fp64": pose_mse_delta,
            "d_pose_max_abs_component_delta": pose_max_abs,
            "pose_ref_rms_magnitude": pose_ref_rms,
        },
        "timing": {
            "threads": args.threads,
            "time_pairs": tp,
            "numpy_fp64_sec_per_pair": np_per_pair,
            "torch_fp32_cpu_sec_per_pair": torch_per_pair,
            "cpu_speedup_torch_vs_numpy": speedup,
            "numpy_fp64_n600_minutes": np_n600_min,
            "torch_fp32_cpu_n600_minutes": torch_n600_min,
            "feed_eg_numpy_baseline_minutes": "53-61 (measured FEED-eg, n600 4-core CPU)",
        },
        "t4_projection": {
            "assumed_gpu_speedup_band_vs_cpu4thread": [15, 50],
            "t4_n600_minutes_band": [t4_lo_min, t4_hi_min],
            "modal_t4_smoke_cmd": modal_t4_cmd,
        },
        "verdict": {
            "legal_under_30min_cpu_numpy_fp64": legal_cpu_numpy,
            "legal_under_30min_cpu_torch_fp32": legal_cpu_torch,
            "legal_under_30min_t4_torch_fp32": legal_t4,
        },
        "packet_dir": str(packet_dir),
    }

    if not args.keep_packet:
        shutil.rmtree(work, ignore_errors=True)
        report["packet_dir"] = "(deleted; pass --keep-packet to retain)"

    out = args.out or (_REPO / "reports" / f"levelset_torch_inflate_parity_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[report] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
