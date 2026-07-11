#!/usr/bin/env python3
"""Export a REAL level-set witness (Kernel B) fixture for the WebGPU/WebNN demo.

NO-FAKE contract
----------------
This loads an ACTUAL trained witness checkpoint (the EMA-BEST npz from a live n600
run), reconstructs the byte-closeable curvelet + self-orient directional front-end
with the repo's OWN authoritative functions, and computes the numpy-fp32 reference
partition (``argmax_k phi_k``) via the EXACT forward the trainer uses
(``levelset_sdf_argmax_mlx``'s numpy twin: in_proj -> FiLM hidden x N -> out_sdf).

It writes, for a handful of representative frames:
  * ``fixture.json``  -- MLP weights (in_proj/film/hidden.*/out_sdf), palette, cfg
                         scalars, per-frame FiLM codes, and the shared+per-frame
                         coord feature grid (P x in_feat) at a demo-sized grid.
  * ``reference.bin`` -- the numpy-fp32 reference partition (uint8, P per frame),
                         the parity target for the WebGPU/WebNN port.

The parity contract for the demo is: WGSL/WebNN forward vs THIS numpy-fp32 forward
on the IDENTICAL shipped (feats, weights). That makes the port fidelity claim exact
and verifiable. Everything the demo shows is a real witness forward pass; it is
labelled ``[WebGPU/WebNN demo -- NON-AUTHORITY]``. No contest score is produced here
(only ``upstream/evaluate.py`` on byte-closed archive bytes is a score).

Usage:
    .venv/bin/python demo/witness_webgpu/export_fixture.py \
        --ckpt experiments/results/levelset_n600_witness_20260705T015247Z/levelset_witness_ema_BEST.npz \
        --grid-h 96 --grid-w 128 --frames 0 199 399 599 799 999
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tac.boundary_math.lever_b_generator import self_orientation_directional_feats  # noqa: E402
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    curvelet_directional_B,
    curvelet_feats,
)


def _act(u: np.ndarray, name: str, w0: float, s0: float, beta: float, omega: float) -> np.ndarray:
    u = np.asarray(u, np.float64)
    if name == "wire":
        return (np.cos(w0 * u) * np.exp(-((s0 * u) ** 2))).astype(np.float64)
    if name == "hosc":
        return np.tanh(beta * np.sin(omega * u)).astype(np.float64)
    return np.maximum(u, 0.0)


def numpy_forward(p, feats, code, n_hidden, hidden_dim, act_name, akw):
    """Bit-for-bit twin of ``levelset_sdf_argmax_mlx`` (numpy fp64). Returns (P, K) phi."""
    h = _act(feats @ p["in_proj.weight"].T + p["in_proj.bias"], act_name, **akw)
    film = (code @ p["film.weight"].T + p["film.bias"]).reshape(n_hidden, 2, hidden_dim)
    for li in range(n_hidden):
        scale = 1.0 + film[li, 0]
        shift = film[li, 1]
        pre = (h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * scale + shift
        h = _act(pre, act_name, **akw)
    return h @ p["out_sdf.weight"].T + p["out_sdf.bias"]


def build_coords(gh: int, gw: int) -> np.ndarray:
    """(P,2) coord grid on [-1,1]^2, row-major (y outer, x inner) -- the render convention."""
    ys = np.linspace(-1.0, 1.0, gh, dtype=np.float64)
    xs = np.linspace(-1.0, 1.0, gw, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (P,2) [x,y]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/results/levelset_n600_witness_20260705T015247Z/levelset_witness_ema_BEST.npz")
    ap.add_argument("--grid-h", type=int, default=96)
    ap.add_argument("--grid-w", type=int, default=128)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 199, 399, 599, 799, 999])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    ck = Path(args.ckpt)
    if not ck.is_absolute():
        ck = _REPO / ck
    z = np.load(ck, allow_pickle=True)

    n_hidden = int(z["__cfg_n_hidden"])
    hidden_dim = int(z["__cfg_hidden_dim"])
    in_feat = int(z["__cfg_in_feat"])
    act_name = str(z["__cfg_activation"])
    akw = {
        "w0": float(z["__cfg_wire_w0"]), "s0": float(z["__cfg_wire_s0"]),
        "beta": float(z["__cfg_hosc_beta"]), "omega": float(z["__cfg_hosc_omega"]),
    }
    mf = float(z["__cfg_max_bank_freq"])
    max_freq = None if mf <= 0 else mf
    bank = CurveletBankConfig(
        int(z["__bank_n_scales"]), int(z["__bank_n_orient0"]),
        float(z["__bank_f0"]), float(z["__bank_base"]), int(z["__bank_n_iso"]),
    )
    B = curvelet_directional_B(bank, max_freq=max_freq)  # (2, ncols)
    n_curv = 2 * B.shape[1]

    p = {k: np.asarray(z[k], np.float64) for k in z.files if not k.startswith("__")}
    code_all = np.asarray(z["code"], np.float64)  # (nframes, mod_dim)
    palette = np.asarray(z["palette"], np.float32) if "palette" in z.files else None

    gh, gw = int(args.grid_h), int(args.grid_w)
    coords = build_coords(gh, gw)
    P = coords.shape[0]

    cf = curvelet_feats(coords, B).astype(np.float64)  # (P, n_curv)
    self_orient_gap = in_feat - n_curv
    if self_orient_gap < 0 or self_orient_gap % 4 != 0:
        raise SystemExit(f"in_feat={in_feat} incompatible with curvelet={n_curv}")
    so_nfreqs = self_orient_gap // 4  # 0 if none

    frames = [f for f in args.frames if 0 <= f < code_all.shape[0]]
    per_frame_feats = []
    ref_parts = []
    for f in frames:
        code = code_all[f]
        if self_orient_gap > 0:
            # cheap partition = witness's own curvelet-only-ish first-pass argmax on the
            # zero-directional feats (byte-closeable self-orientation fixed point, documented).
            feats0 = np.concatenate([cf, np.zeros((P, self_orient_gap), np.float64)], axis=1)
            phi0 = numpy_forward(p, feats0, code, n_hidden, hidden_dim, act_name, akw)
            cheap = phi0.argmax(axis=-1).reshape(gh, gw).astype(np.int32)
            so = self_orientation_directional_feats(coords, cheap, n_freqs=so_nfreqs).astype(np.float64)
            feats = np.concatenate([cf, so], axis=1)
        else:
            feats = cf
        if feats.shape[1] != in_feat:
            raise SystemExit(f"feats dim {feats.shape[1]} != in_feat {in_feat} (frame {f})")
        phi = numpy_forward(p, feats, code, n_hidden, hidden_dim, act_name, akw)
        part = phi.argmax(axis=-1).astype(np.uint8)  # (P,)
        per_frame_feats.append(feats.astype(np.float32))
        ref_parts.append(part)

    # ---- write reference.bin (uint8 partitions, frame-major) ----
    out = Path(args.out)
    ref = np.concatenate(ref_parts, axis=0).astype(np.uint8)
    (out / "reference.bin").write_bytes(ref.tobytes())

    # ---- write feats.bin (fp32, frame-major P*in_feat) ----
    feats_flat = np.concatenate([ff.ravel() for ff in per_frame_feats]).astype(np.float32)
    (out / "feats.bin").write_bytes(feats_flat.tobytes())

    def flat(name):
        return np.asarray(p[name], np.float32).ravel().tolist()

    fixture = {
        "meta": {
            "authority": "[WebGPU/WebNN demo -- NON-AUTHORITY]",
            "ckpt": str(ck.relative_to(_REPO)),
            "epoch": int(z["__epoch"]) if "__epoch" in z.files else -1,
            "activation": act_name, "akw": akw,
            "n_hidden": n_hidden, "hidden_dim": hidden_dim, "in_feat": in_feat,
            "n_classes": int(p["out_sdf.weight"].shape[0]),
            "mod_dim": int(code_all.shape[1]),
            "grid_h": gh, "grid_w": gw, "P": P,
            "frames": frames, "n_frames": len(frames),
            "curvelet_cols": int(B.shape[1]), "self_orient_nfreqs": int(so_nfreqs),
            "note": "feats.bin: fp32 frame-major (P*in_feat) computed by numpy-fp32 authority. "
                    "reference.bin: uint8 frame-major (P) argmax partition. Parity target.",
        },
        # MLP weights (row-major as stored; shader multiplies x @ W.T + b).
        "weights": {
            "in_proj.weight": flat("in_proj.weight"), "in_proj.bias": flat("in_proj.bias"),
            "film.weight": flat("film.weight"), "film.bias": flat("film.bias"),
            "out_sdf.weight": flat("out_sdf.weight"), "out_sdf.bias": flat("out_sdf.bias"),
            **{f"hidden.{li}.weight": flat(f"hidden.{li}.weight") for li in range(n_hidden)},
            **{f"hidden.{li}.bias": flat(f"hidden.{li}.bias") for li in range(n_hidden)},
        },
        "codes": [code_all[f].astype(np.float32).tolist() for f in frames],
        "palette": (palette.tolist() if palette is not None else None),
    }
    (out / "fixture.json").write_text(json.dumps(fixture))

    # ---- sanity + provenance line ----
    print(json.dumps({
        "wrote": ["fixture.json", "feats.bin", "reference.bin"],
        "frames": frames, "P": P, "in_feat": in_feat, "n_hidden": n_hidden,
        "activation": act_name, "self_orient_nfreqs": int(so_nfreqs),
        "reference_bytes": int(ref.nbytes), "feats_bytes": int(feats_flat.nbytes),
        "class_hist_frame0": np.bincount(ref_parts[0], minlength=5).tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
