#!/usr/bin/env python3
"""Witness CODE-RATE lever (Wave-3): PCA-compress the per-pair FiLM codes, measure the
rate-vs-realized-d_seg Pareto, and byte-close the best K.

WHY (measured): the level-set witness's per-pair ``code`` (400x32) is the COUNTED video-derived
payload (rule 118 -- the shared decoder is amortized; the per-pair codes are counted). The
dimensionality lens measured the codes' effective rank ~11-16 of 32, so storing 32 dims wastes
bytes. PCA 32->K (basis 32xK shared + coords 400xK counted) should cut the counted payload at
~0 d_seg cost. This is a BANKED rate lever (the witness is currently S~0.46, NOT a pointer-mover);
the win composes when the witness becomes competitive.

NO-FAKE: the K=32-no-PCA GATE reproduces the trainer's realized d_seg (~0.0032 calibration ref)
via the EXACT deploy-faithful ONE CODEPATH (``levelset_rgb_forward_numpy`` + contest-exact R +
frozen CPU-torch SegNet argmax) used by the trainer's ``realized_verdict``. Each PCA-K point
reconstructs codes FROM the DEQUANTIZED PCA representation (deploy-faithful) before the realized
verdict -- the byte-close measures the bytes of the same representation it decodes. CPU/numpy only;
NO GPU, NO MLX render, NO training. Render at 384, SegNet on CPU (the authority). The self-orient
directional feats are recomputed from the witness's OWN frame1 argmax (the deploy fixed point; no
GT leak), per-vpair (the verdict only renders vpairs).

axis: [macOS-CPU advisory] -- NON-PROMOTABLE. score_claim=false. The only score is upstream/evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import brotli  # noqa: E402

from tac.boundary_math.lever_b_generator import (  # noqa: E402
    self_orientation_directional_feats,
)
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    _int8_symmetric,
    curvelet_directional_B,
    curvelet_feats,
    int8_dequant_params,
    levelset_rgb_forward_numpy,
    quantize_levelset_blob,
)
from train_witness_realized_through_R_mlx import (  # noqa: E402
    _build_render_coords,
    _torch_R_to_camera_uint8,
    cpu_verdict_d_pose_batch,
    cpu_verdict_d_seg_batch,
    load_gt_from_cache,
)

CONTEST_N = 37_545_489  # upstream/evaluate.py rate denominator


def _refuse_tmp(p: Path, field: str) -> None:
    if str(p).startswith("/tmp/") or str(p).startswith("/private/tmp/"):
        raise ValueError(f"{field}={p} is a /tmp path; durable evidence must not live in /tmp.")


# ---------------------------------------------------------------------------
# byte-close helpers for the PCA code representation. brotli q11 (matches the
# existing witness byte-close). Each returns (n_bytes, dequantized_array).
# ---------------------------------------------------------------------------
def _bc_fp16(a: np.ndarray, coder: str = "brotli") -> tuple[int, np.ndarray]:
    """fp16 round-trip + entropy-code the raw fp16 bytes."""
    h = np.asarray(a, np.float32).astype(np.float16)
    raw = h.tobytes()
    n = _entropy_len(raw, coder)
    return n, h.astype(np.float32)


def _bc_int8_pertensor(a: np.ndarray, coder: str = "brotli") -> tuple[int, np.ndarray]:
    """single (per-tensor) symmetric int8 scale + entropy-code. +4 bytes for the fp32 scale."""
    q, scale = _int8_symmetric(np.asarray(a, np.float32))
    n = _entropy_len(q.tobytes(), coder) + 4  # one fp32 scale
    return n, (q.astype(np.float32) * scale).astype(np.float32)


def _bc_int8_percol(a: np.ndarray, coder: str = "brotli") -> tuple[int, np.ndarray]:
    """per-COLUMN symmetric int8 (each PC coord-dim gets its own scale -> exploits the fast
    PC-variance decay) + entropy-code. + 2*K bytes for the K fp16 per-col scales."""
    a = np.asarray(a, np.float32)
    cols = a.shape[1]
    q = np.empty(a.shape, np.int8)
    scales = np.empty(cols, np.float32)
    for j in range(cols):
        col = a[:, j]
        s = float(np.abs(col).max()) + 1e-8
        q[:, j] = np.clip(np.round(col / s * 127.0), -127, 127).astype(np.int8)
        scales[j] = s / 127.0
    n = _entropy_len(q.tobytes(), coder) + 2 * cols  # K fp16 scales
    deq = (q.astype(np.float32) * scales[None, :]).astype(np.float32)
    return n, deq


def _entropy_len(raw: bytes, coder: str) -> int:
    if coder == "brotli":
        return len(brotli.compress(raw, quality=11))
    if coder == "zlib":
        return len(zlib.compress(raw, 9))
    raise ValueError(f"unknown coder {coder}")


_COORD_METHODS = {
    "int8_pertensor": _bc_int8_pertensor,
    "int8_percol": _bc_int8_percol,
    "fp16": _bc_fp16,
}


# ---------------------------------------------------------------------------
# The deploy-faithful realized verdict over vpairs (numpy ONE CODEPATH == the trainer's
# realized_verdict). Self-orient dir feats recomputed from the witness's own frame1 argmax.
# ---------------------------------------------------------------------------
def realized_verdict_for_codes(
    *,
    deploy_base: dict[str, np.ndarray],
    codes_deploy: np.ndarray,
    curv_feats: np.ndarray,
    coords_np: np.ndarray,
    gt: Any,
    seg_cpu: Any,
    posenet_cpu: Any,
    vpairs: list[int],
    cfg: dict[str, Any],
    reorient_iters: int,
    seg_chunk: int,
) -> dict[str, float]:
    render_h, render_w = cfg["render_h"], cfg["render_w"]
    use_self_orient = bool(cfg["self_orient"])
    n_dir_freqs = int(cfg["n_dir_freqs"])
    dir_w = 4 * n_dir_freqs
    fwd_kw = dict(
        n_hidden=cfg["n_hidden"], hidden_dim=cfg["hidden_dim"], n_classes=5,
        activation=cfg["activation"], softmax_temp=cfg["softmax_temp"],
        wire_w0=cfg["wire_w0"], wire_s0=cfg["wire_s0"],
        hosc_beta=cfg["hosc_beta"], hosc_omega=cfg["hosc_omega"], chroma=bool(cfg["chroma"]),
    )
    deploy = {**deploy_base, "code": np.asarray(codes_deploy, np.float32)}

    # per-vpair self-orient directional feats (start at zeros -> curvelet-only iso pass).
    dir_feats = {pi: np.zeros((curv_feats.shape[0], dir_w), np.float32) for pi in vpairs}

    def _feats(pi: int) -> np.ndarray:
        if not use_self_orient:
            return curv_feats
        return np.concatenate([curv_feats, dir_feats[pi]], axis=-1).astype(np.float32)

    # self-orientation FIXED POINT: from each pair's frame1 argmax recompute its dir feats.
    if use_self_orient:
        for _ in range(max(reorient_iters, 1)):
            for pi in vpairs:
                _rgb, phi = levelset_rgb_forward_numpy(deploy, _feats(pi), deploy["code"][2 * pi + 1], **fwd_kw)
                argmax = phi.argmax(-1).reshape(render_h, render_w).astype(np.int64)
                dir_feats[pi] = self_orientation_directional_feats(
                    coords_np, argmax, n_freqs=n_dir_freqs,
                    freq_across=cfg["freq_across"], freq_along=cfg["freq_along"]).astype(np.float32)

    # render f0 + f1 through the contest-exact R -> camera uint8.
    f0s, f1s, lstars, gtposes = [], [], [], []
    for pi in vpairs:
        feats_pi = _feats(pi)
        rgb0, _ = levelset_rgb_forward_numpy(deploy, feats_pi, deploy["code"][2 * pi + 0], **fwd_kw)
        rgb1, _ = levelset_rgb_forward_numpy(deploy, feats_pi, deploy["code"][2 * pi + 1], **fwd_kw)
        f0s.append(_torch_R_to_camera_uint8(rgb0.reshape(render_h, render_w, 3)))
        f1s.append(_torch_R_to_camera_uint8(rgb1.reshape(render_h, render_w, 3)))
        lstars.append(gt.lstars[pi])
        gtposes.append(gt.gt_poses[pi])

    # SegNet/PoseNet CPU verdict, chunked to bound memory (the frozen .eval() net -> batch-exact).
    ds_all, dp_all = [], []
    for c0 in range(0, len(vpairs), seg_chunk):
        sl = slice(c0, c0 + seg_chunk)
        ds_all.extend(cpu_verdict_d_seg_batch(seg_cpu, f1s[sl], lstars[sl]))
        dp_all.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[sl], f1s[sl], gtposes[sl]))
    return {"d_seg": float(np.mean(ds_all)), "d_pose": float(np.mean(dp_all))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=str(
        REPO / "experiments/results/levelset_l7_preserved_snapshots/l7_ema_snapshot_20260628T003440Z.npz"))
    ap.add_argument("--gt-cache", default=str(REPO / "experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz"))
    ap.add_argument("--num-pairs", type=int, default=200)
    ap.add_argument("--verdict-pairs", type=int, default=24,
                    help="number of strided verdict pairs (matches the trainer default); 0 = all pairs.")
    ap.add_argument("--ks", default="24,20,16,11", help="comma list of PCA K to sweep.")
    ap.add_argument("--reorient-iters", type=int, default=1, help="self-orient fixed-point passes.")
    ap.add_argument("--seg-chunk", type=int, default=24, help="SegNet/PoseNet batch chunk (memory bound).")
    ap.add_argument("--gate-lo", type=float, default=0.0028)
    ap.add_argument("--gate-hi", type=float, default=0.0038)
    ap.add_argument("--out", required=True, help="output JSON FILE path.")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    _refuse_tmp(out_path, "--out")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- load snapshot params + cfg ---
    z = np.load(Path(args.snapshot), allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    if "code" not in params:
        raise ValueError(f"{args.snapshot} has no 'code' -- not a level-set witness snapshot.")
    code_fp32 = params["code"].astype(np.float32)  # (2P, mod)
    cfg = {
        "render_h": int(z["__render_hw"][0]), "render_w": int(z["__render_hw"][1]),
        "n_hidden": int(z["__cfg_n_hidden"]), "hidden_dim": int(z["__cfg_hidden_dim"]),
        "activation": str(z["__cfg_activation"]), "softmax_temp": float(z["__cfg_softmax_temp"]),
        "wire_w0": float(z["__cfg_wire_w0"]), "wire_s0": float(z["__cfg_wire_s0"]),
        "hosc_beta": float(z["__cfg_hosc_beta"]), "hosc_omega": float(z["__cfg_hosc_omega"]),
        "chroma": int(z["__cfg_chroma"]), "self_orient": int(z["__cfg_self_orient"]),
        "n_dir_freqs": int(z["__cfg_n_dir_freqs"]), "freq_across": float(z["__cfg_freq_across"]),
        "freq_along": float(z["__cfg_freq_along"]),
        "bank_n_scales": int(z["__bank_n_scales"]), "bank_n_orient0": int(z["__bank_n_orient0"]),
        "bank_f0": float(z["__bank_f0"]), "bank_base": float(z["__bank_base"]),
        "bank_n_iso": int(z["__bank_n_iso"]), "max_bank_freq": float(z["__cfg_max_bank_freq"]),
        "epoch": int(z["__epoch"]),
    }
    mod_dim = code_fp32.shape[1]
    print(json.dumps({"stage": "snapshot", "epoch": cfg["epoch"], "code_shape": list(code_fp32.shape),
                      "mod_dim": mod_dim, **{k: cfg[k] for k in ("render_h", "render_w", "softmax_temp",
                      "hosc_beta", "self_orient", "n_dir_freqs", "chroma")}}), flush=True)

    # --- curvelet front-end (free deterministic bank, rule 118) ---
    bank = CurveletBankConfig(n_scales=cfg["bank_n_scales"], n_orient0=cfg["bank_n_orient0"],
                              f0=cfg["bank_f0"], base=cfg["bank_base"], n_iso=cfg["bank_n_iso"])
    B = curvelet_directional_B(bank, max_freq=cfg["max_bank_freq"])
    coords_np = _build_render_coords(cfg["render_h"], cfg["render_w"])
    curv_feats = curvelet_feats(coords_np, B).astype(np.float32)
    in_feat = curv_feats.shape[1] + (4 * cfg["n_dir_freqs"] if cfg["self_orient"] else 0)
    print(json.dumps({"stage": "front_end", "curvelet_cols": int(B.shape[1]), "in_feat": int(in_feat)}), flush=True)

    # --- GT (frozen CPU-torch authority) + scorers ---
    t0 = time.time()
    gt, seg_cpu, posenet_cpu = load_gt_from_cache(Path(args.gt_cache), args.num_pairs)
    P = gt.n_pairs
    print(json.dumps({"stage": "gt", "n_pairs": P, "secs": round(time.time() - t0, 1)}), flush=True)

    if 2 * P != code_fp32.shape[0]:
        raise ValueError(f"code has {code_fp32.shape[0]} rows but GT has {P} pairs (expect {2*P}).")

    # verdict pairs (strided, matching the trainer's vpairs construction).
    if args.verdict_pairs and args.verdict_pairs < P:
        vpairs = list(range(0, P, max(1, P // max(args.verdict_pairs, 1))))[: args.verdict_pairs]
    else:
        vpairs = list(range(P))
    print(json.dumps({"stage": "vpairs", "n_vpairs": len(vpairs), "first8": vpairs[:8]}), flush=True)

    # --- decoder base bytes (UNCHANGED across K) + raw-code bytes (full-32 baseline) ---
    blob = quantize_levelset_blob(params)
    base_bytes = int(blob["base_int8_brotli_bytes"])
    code_bytes_raw = int(blob["code_int8_brotli_bytes"])  # existing int8-per-tensor+brotli on raw codes
    full_total = base_bytes + code_bytes_raw
    print(json.dumps({"stage": "byteclose_baseline", "base_bytes": base_bytes,
                      "code_bytes_raw_int8": code_bytes_raw, "full_total_bytes": full_total,
                      "full_rate_term": round(25 * full_total / CONTEST_N, 6)}), flush=True)

    # deploy decoder weights (int8-dequant; the deploy-faithful base, code handled per-arm).
    deploy_base = int8_dequant_params({k: v for k, v in params.items() if k != "code"})

    results: dict[str, Any] = {
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; score_claim=false; pointer UNMOVED",
        "snapshot": str(Path(args.snapshot)), "epoch": cfg["epoch"], "mod_dim": mod_dim,
        "n_pairs": P, "n_vpairs": len(vpairs), "reorient_iters": args.reorient_iters,
        "base_bytes": base_bytes, "code_bytes_raw_int8_brotli": code_bytes_raw,
        "full32_total_bytes": full_total, "full32_rate_term": 25 * full_total / CONTEST_N,
        "contest_N": CONTEST_N, "rows": [],
    }

    # ===================== NO-FAKE GATE: K=32 no-PCA (existing deploy path) =====================
    t0 = time.time()
    code_gate = int8_dequant_params({"code": code_fp32})["code"]  # int8 round-trip == existing deploy
    v_gate = realized_verdict_for_codes(
        deploy_base=deploy_base, codes_deploy=code_gate, curv_feats=curv_feats, coords_np=coords_np,
        gt=gt, seg_cpu=seg_cpu, posenet_cpu=posenet_cpu, vpairs=vpairs, cfg=cfg,
        reorient_iters=args.reorient_iters, seg_chunk=args.seg_chunk)
    gate_pass = bool(args.gate_lo <= v_gate["d_seg"] <= args.gate_hi)
    results["gate"] = {
        "kind": "K32_noPCA_existing_deploy_path", "d_seg": v_gate["d_seg"], "d_pose": v_gate["d_pose"],
        "calibration_ref_d_seg": 0.003254, "band": [args.gate_lo, args.gate_hi], "pass": gate_pass,
        "total_bytes": full_total, "rate_term": 25 * full_total / CONTEST_N, "secs": round(time.time() - t0, 1),
    }
    print(json.dumps({"stage": "GATE", **results["gate"]}), flush=True)
    if not gate_pass:
        results["gate"]["note"] = ("GATE FAILED: realized d_seg outside calibration band -> the witness "
                                   "render reproduction is wrong; PCA Pareto NOT trustworthy. STOP.")
        out_path.write_text(json.dumps(results, indent=2))
        print(json.dumps({"stage": "ABORT", "reason": "gate_failed", "out": str(out_path)}), flush=True)
        return 2

    # ===================== PCA of the codes =====================
    mean = code_fp32.mean(axis=0).astype(np.float32)  # (mod,)
    centered = code_fp32 - mean[None, :]
    # SVD of centered codes: centered = U S Vt ; PCs are rows of Vt (=cols of V). coords = centered @ V.
    with np.errstate(all="ignore"):  # silence spurious macOS-Accelerate FP-flag warnings (values exact)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)  # Vt (mod, mod)
    V = Vt.T  # (mod, mod), columns are principal axes
    sv = S.astype(np.float64)
    energy = (sv ** 2)
    cum = np.cumsum(energy) / energy.sum()
    results["pca"] = {"singular_values": sv.round(5).tolist(),
                      "cum_energy": cum.round(5).tolist()}
    print(json.dumps({"stage": "pca", "top_sv": sv[:8].round(4).tolist(),
                      "cum_energy_at_11_16_20_24": [round(float(cum[min(k-1, mod_dim-1)]), 5)
                                                    for k in (11, 16, 20, 24)]}), flush=True)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    # add K=32 (full-rank PCA) as a sanity point if not present and mod_dim==32.
    if mod_dim not in ks:
        ks = ks + [mod_dim]
    ks = sorted(set(k for k in ks if 1 <= k <= mod_dim), reverse=True)

    for K in ks:
        t0 = time.time()
        V_K = V[:, :K].astype(np.float32)               # (mod, K) basis (shared, counted)
        with np.errstate(all="ignore"):  # spurious macOS-Accelerate matmul FP-flag (values exact)
            coords_K = (np.asarray(centered, np.float64) @ V_K).astype(np.float32)  # (2P, K) coords

        # byte-close the basis + mean ONCE (fp16+brotli; tiny, orthonormal -> fp16 plenty).
        basis_bytes, V_K_dq = _bc_fp16(V_K, "brotli")
        mean_bytes, mean_dq = _bc_fp16(mean, "brotli")

        # try each coord QUANT method; render ONCE per method (the coder choice -- brotli/zlib --
        # changes ONLY the stored bytes, NOT the dequantized coords -> NOT d_seg, so we render once
        # and price both coders off the SAME decoded coords). Pick the smallest faithful arm.
        method_rows = []
        best = None
        for mname, mfn in _COORD_METHODS.items():
            # decode the coords ONCE (coder-independent dequant; brotli used to obtain coords_dq).
            _cbytes_brotli, coords_dq = mfn(coords_K, "brotli")
            with np.errstate(all="ignore"):  # spurious macOS-Accelerate matmul FP-flag (values exact)
                codes_K_deploy = (mean_dq[None, :] + np.asarray(coords_dq, np.float64)
                                  @ np.asarray(V_K_dq, np.float64).T).astype(np.float32)  # (2P, mod)
            v = realized_verdict_for_codes(
                deploy_base=deploy_base, codes_deploy=codes_K_deploy, curv_feats=curv_feats,
                coords_np=coords_np, gt=gt, seg_cpu=seg_cpu, posenet_cpu=posenet_cpu, vpairs=vpairs,
                cfg=cfg, reorient_iters=args.reorient_iters, seg_chunk=args.seg_chunk)
            for coder in ("brotli", "zlib"):
                cbytes = _cbytes_brotli if coder == "brotli" else mfn(coords_K, "zlib")[0]
                code_total = int(cbytes + basis_bytes + mean_bytes)
                total = base_bytes + code_total
                row = {
                    "K": K, "coord_method": mname, "coder": coder,
                    "coords_bytes": int(cbytes), "basis_bytes": int(basis_bytes),
                    "mean_bytes": int(mean_bytes), "code_total_bytes": code_total,
                    "total_bytes": total, "rate_term": 25 * total / CONTEST_N,
                    "d_seg": v["d_seg"], "d_pose": v["d_pose"],
                    "d_seg_vs_full_pct": 100.0 * (v["d_seg"] - v_gate["d_seg"]) / v_gate["d_seg"],
                }
                method_rows.append(row)
                # "best" = smallest total_bytes among arms whose d_seg is within +2% of the full-32 gate.
                within = row["d_seg"] <= v_gate["d_seg"] * 1.02
                if within and (best is None or total < best["total_bytes"]):
                    best = row
        # if NO arm is within +2%, fall back to the smallest-d_seg arm (report honestly).
        if best is None:
            best = min(method_rows, key=lambda r: r["d_seg"])
            best = {**best, "note": "no arm within +2% d_seg of full-32; reporting min-d_seg arm"}
        best_row = {"K": K, "best": best, "all_methods": method_rows, "secs": round(time.time() - t0, 1)}
        results["rows"].append(best_row)
        print(json.dumps({"stage": "K_done", "K": K, "best_method": best["coord_method"],
                          "best_coder": best["coder"], "total_bytes": best["total_bytes"],
                          "rate_term": round(best["rate_term"], 6), "d_seg": round(best["d_seg"], 6),
                          "d_seg_vs_full_pct": round(best["d_seg_vs_full_pct"], 2),
                          "secs": best_row["secs"]}), flush=True)

    # ===================== pick the knee K* =====================
    # K* = the K with the LOWEST total_bytes whose best-arm d_seg is within +2% of the full-32 gate.
    candidates = [r["best"] for r in results["rows"]
                  if r["best"]["d_seg"] <= v_gate["d_seg"] * 1.02 and r["best"]["K"] < mod_dim]
    if candidates:
        kstar = min(candidates, key=lambda r: r["total_bytes"])
        savings = full_total - kstar["total_bytes"]
        results["verdict"] = {
            "kstar": kstar["K"], "method": kstar["coord_method"], "coder": kstar["coder"],
            "total_bytes": kstar["total_bytes"], "rate_term": kstar["rate_term"],
            "d_seg": kstar["d_seg"], "d_seg_vs_full_pct": kstar["d_seg_vs_full_pct"],
            "byte_savings_vs_full32": int(savings),
            "byte_savings_pct": 100.0 * savings / full_total,
            "rate_term_delta": kstar["rate_term"] - 25 * full_total / CONTEST_N,
            "framing": ("BANKED rate lever (PROVEN/measured, [macOS-CPU advisory]); the witness is "
                        "S~0.46 (NOT a pointer-mover) -- this composes when the witness is competitive."),
        }
    else:
        results["verdict"] = {"kstar": None,
                              "note": "no K<32 stayed within +2% d_seg of full-32; PCA-compression not free here."}
    print(json.dumps({"stage": "VERDICT", **results["verdict"]}), flush=True)

    results["total_secs"] = round(time.time() - t_start, 1)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps({"stage": "DONE", "out": str(out_path), "total_secs": results["total_secs"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
