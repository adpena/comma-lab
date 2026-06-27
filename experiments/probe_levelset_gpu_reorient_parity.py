# SPDX-License-Identifier: MIT
"""FEED-eo PARITY PROBE — MLX-GPU reorient vs numpy reorient (the --gpu-reorient 6.2% lever gate).

$0/CHEAP, NO training: loads a REAL trained level-set witness checkpoint (the n600 epoch-25 ckpt the
live row resumed from), then for a SMALL-n pair sample computes the self-orientation reorient TWO ways
off the SAME dequantized deploy weights + SAME (curvelet + zeros-dir) input feats:
  (A) numpy fp64 ONE-CODEPATH forward -> argmax -> tangent -> directional fourier feats  (the AUTHORITY)
  (B) MLX-GPU fp32 twin forward       -> argmax -> tangent -> directional fourier feats  (--gpu-reorient)
and reports the PARITY GATE: cos(dir_feats_A, dir_feats_B) and the realized d_seg A/B (render f1 with
each path's dir feats -> R -> frozen CPU-torch SegNet argmax). ADOPT iff cos>0.999 AND |d_seg A/B|
small (~<1e-4); else KEEP numpy (the 6.2% is not safely reclaimable). The dir feats are a deterministic
function of the witness's OWN argmax, so the GPU vs numpy gap is purely the fp32-GPU-vs-fp64-numpy
argmax disagreement at boundary pixels. NON-PROMOTABLE advisory; pointer UNMOVED 0.19110.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_witness_realized_through_R_mlx import (  # noqa: E402
    _build_render_coords,
    _torch_R_to_camera_uint8,
    cpu_verdict_d_seg_batch,
)
from train_levelset_witness_realized_through_R_mlx import levelset_sdf_argmax_mlx  # noqa: E402
from tac.boundary_math.lever_b_generator import self_orientation_directional_feats  # noqa: E402
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    curvelet_directional_B,
    curvelet_feats,
    int8_dequant_params,
    levelset_rgb_forward_numpy,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0  # both all-zero (degenerate argmax) -> identical
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FEED-eo MLX-GPU reorient parity probe")
    ap.add_argument("--ckpt", type=str,
                    default="experiments/results/levelset_n600_wpose1_20260627T123627Z/levelset_witness_ema_mlx.npz")
    ap.add_argument("--gt-cache", type=str, default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--out-dir", type=str,
                    default="experiments/results/levelset_gpu_reorient_parity_probe")
    args = ap.parse_args(argv)

    t0 = time.time()
    z = np.load(Path(args.ckpt), allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    cfg = {k: (z[k].item() if z[k].size == 1 else z[k].tolist()) for k in z.files if k.startswith("__")}
    n_hidden = int(cfg["__cfg_n_hidden"]); hidden_dim = int(cfg["__cfg_hidden_dim"])
    activation = str(cfg["__cfg_activation"]); softmax_temp = float(cfg["__cfg_softmax_temp"])
    wire_w0 = float(cfg["__cfg_wire_w0"]); wire_s0 = float(cfg["__cfg_wire_s0"])
    hosc_beta = float(cfg["__cfg_hosc_beta"]); hosc_omega = float(cfg["__cfg_hosc_omega"])
    chroma = bool(int(cfg["__cfg_chroma"]))
    n_dir_freqs = int(cfg["__cfg_n_dir_freqs"]); freq_across = float(cfg["__cfg_freq_across"])
    freq_along = float(cfg["__cfg_freq_along"])
    max_bank_freq = float(cfg["__cfg_max_bank_freq"]); max_bank_freq = None if max_bank_freq < 0 else max_bank_freq
    render_h, render_w = int(cfg["__render_hw"][0]), int(cfg["__render_hw"][1])
    bank = CurveletBankConfig(n_scales=int(cfg["__bank_n_scales"]), n_orient0=int(cfg["__bank_n_orient0"]),
                              f0=float(cfg["__bank_f0"]), base=float(cfg["__bank_base"]), n_iso=int(cfg["__bank_n_iso"]))
    dir_w = 4 * n_dir_freqs

    coords = _build_render_coords(render_h, render_w)
    B = curvelet_directional_B(bank, max_freq=max_bank_freq)
    curv = curvelet_feats(coords, B).astype(np.float32)  # (P, 2*cols)
    in_feat_curv = curv.shape[1]
    in_feat = in_feat_curv + dir_w
    deploy = int8_dequant_params(params)
    ckpt_in = int(deploy["in_proj.weight"].shape[1])
    print(json.dumps({"stage": "loaded", "ckpt_epoch": int(cfg.get("__epoch", -1)),
                      "render_hw": [render_h, render_w], "in_feat_built": in_feat,
                      "in_feat_ckpt": ckpt_in, "dir_w": dir_w, "secs": round(time.time() - t0, 1)}), flush=True)
    if ckpt_in != in_feat:
        raise SystemExit(f"in_feat mismatch built={in_feat} vs ckpt={ckpt_in}")

    import mlx.core as mx
    deploy_mx = {k: mx.array(np.asarray(v, np.float32)) for k, v in deploy.items()
                 if k not in ("code",) and not (k == "B" or k.endswith("_B"))}
    codes = np.asarray(deploy["code"], np.float32)

    N = int(args.n_pairs)
    zeros_dir = np.zeros((curv.shape[0], dir_w), np.float32)
    feats_zeros = np.concatenate([curv, zeros_dir], axis=-1).astype(np.float32)  # the iso-pass input (88)
    feats_zeros_mx = mx.array(feats_zeros)

    cos_list: list[float] = []
    argmax_agree: list[float] = []
    f1_numpy: list[np.ndarray] = []
    f1_gpu: list[np.ndarray] = []

    def _render_f1(dir_feats: np.ndarray, pi: int) -> np.ndarray:
        feats = np.concatenate([curv, dir_feats], axis=-1).astype(np.float32)
        rgb, _phi = levelset_rgb_forward_numpy(
            deploy, feats, codes[2 * pi + 1], n_hidden=n_hidden, hidden_dim=hidden_dim, n_classes=5,
            activation=activation, softmax_temp=softmax_temp, wire_w0=wire_w0, wire_s0=wire_s0,
            hosc_beta=hosc_beta, hosc_omega=hosc_omega, chroma=chroma)
        return _torch_R_to_camera_uint8(rgb.reshape(render_h, render_w, 3))

    t1 = time.time()
    for pi in range(N):
        code_row = codes[2 * pi + 1]
        # (A) numpy fp64 ONE-CODEPATH argmax (frame1) -- the authority the reorient uses today.
        _rgb_np, phi_np = levelset_rgb_forward_numpy(
            deploy, feats_zeros, code_row, n_hidden=n_hidden, hidden_dim=hidden_dim, n_classes=5,
            activation=activation, softmax_temp=softmax_temp, wire_w0=wire_w0, wire_s0=wire_s0,
            hosc_beta=hosc_beta, hosc_omega=hosc_omega, chroma=chroma)
        argmax_np = phi_np.argmax(-1).reshape(render_h, render_w).astype(np.int64)
        # (B) MLX-GPU fp32 twin argmax (the --gpu-reorient path).
        amx = levelset_sdf_argmax_mlx(
            deploy_mx, feats_zeros_mx, mx.array(code_row), n_hidden=n_hidden, hidden_dim=hidden_dim,
            activation=activation, wire_w0=wire_w0, wire_s0=wire_s0, hosc_beta=hosc_beta, hosc_omega=hosc_omega)
        mx.eval(amx)
        argmax_gpu = np.asarray(amx).reshape(render_h, render_w).astype(np.int64)
        agree = float(np.count_nonzero(argmax_np == argmax_gpu)) / argmax_np.size
        argmax_agree.append(agree)
        dir_np = self_orientation_directional_feats(coords, argmax_np, n_freqs=n_dir_freqs,
                                                    freq_across=freq_across, freq_along=freq_along).astype(np.float32)
        dir_gpu = self_orientation_directional_feats(coords, argmax_gpu, n_freqs=n_dir_freqs,
                                                     freq_across=freq_across, freq_along=freq_along).astype(np.float32)
        cos_list.append(_cos(dir_np, dir_gpu))
        # d_seg A/B: render f1 with each path's dir feats (the realized effect of the dir-feat gap).
        f1_numpy.append(_render_f1(dir_np, pi))
        f1_gpu.append(_render_f1(dir_gpu, pi))
        del amx
    mx.clear_cache()
    print(json.dumps({"stage": "reorient_both", "n_pairs": N, "secs": round(time.time() - t1, 1)}), flush=True)

    # ---- d_seg A/B via the frozen CPU-torch SegNet authority (lstars mmap'd -> no 3.6GB frame load) ----
    from tac.boundary_math.seg_core import load_real_segnet
    seg_cpu = load_real_segnet("cpu")
    gz = np.load(Path(args.gt_cache), allow_pickle=False, mmap_mode="r")
    lstars = [np.asarray(gz["lstars"][pi]) for pi in range(N)]
    ds_numpy = cpu_verdict_d_seg_batch(seg_cpu, f1_numpy, lstars)
    ds_gpu = cpu_verdict_d_seg_batch(seg_cpu, f1_gpu, lstars)
    d_seg_numpy = float(np.mean(ds_numpy)); d_seg_gpu = float(np.mean(ds_gpu))
    d_seg_ab = abs(d_seg_gpu - d_seg_numpy)

    cos_min = float(np.min(cos_list)); cos_mean = float(np.mean(cos_list))
    agree_min = float(np.min(argmax_agree)); agree_mean = float(np.mean(argmax_agree))
    adopt = (cos_min > 0.999) and (d_seg_ab < 1e-4)
    verdict = {
        "stage": "PARITY_VERDICT", "n_pairs": N,
        "cos_dir_feats_min": round(cos_min, 6), "cos_dir_feats_mean": round(cos_mean, 6),
        "argmax_agree_min": round(agree_min, 6), "argmax_agree_mean": round(agree_mean, 6),
        "d_seg_numpy": round(d_seg_numpy, 6), "d_seg_gpu": round(d_seg_gpu, 6),
        "d_seg_AB_abs": round(d_seg_ab, 7),
        "gate_cos_gt_0p999": bool(cos_min > 0.999), "gate_dseg_AB_lt_1e-4": bool(d_seg_ab < 1e-4),
        "VERDICT": ("ADOPT --gpu-reorient (parity holds)" if adopt
                    else "KEEP numpy reorient (parity FAILS; 6.2% not safely reclaimable)"),
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; pointer UNMOVED 0.19110",
        "total_secs": round(time.time() - t0, 1),
    }
    print(json.dumps(verdict, indent=2), flush=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity_result.json").write_text(json.dumps(
        {**verdict, "cos_per_pair": [round(c, 6) for c in cos_list],
         "argmax_agree_per_pair": [round(a, 6) for a in argmax_agree],
         "d_seg_numpy_per_pair": [round(x, 6) for x in ds_numpy],
         "d_seg_gpu_per_pair": [round(x, 6) for x in ds_gpu], "ckpt": str(args.ckpt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
