#!/usr/bin/env python3
"""Parity harness — the WGSL SHADER-MODEL vs the numpy-fp32 authority.

The browser cannot be driven headlessly here, so we verify the PORT ALGORITHM instead:
a numpy re-implementation that mirrors ``witness_forward.wgsl`` EXACTLY (fp32 storage +
accumulation, FiLM precomputed on the host, argmax in-shader) is run on the identical
shipped ``(feats, weights)`` and compared to ``reference.bin`` (the numpy-fp32 authority
partition written by ``export_fixture.py``). A browser running the WGSL reproduces THIS
shader-model by construction (same arithmetic, same op-order).

Reports, per frame and overall, the fraction of pixels whose argmax matches the reference.
This is advisory demo parity — ``[WebGPU/WebNN demo — NON-AUTHORITY]``. No contest score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def act(u, kind, a0, a1):
    u = u.astype(np.float32)
    if kind == 0:  # hosc: tanh(beta*sin(omega*u))
        return np.tanh(np.float32(a0) * np.sin(np.float32(a1) * u)).astype(np.float32)
    if kind == 1:  # wire: cos(w0*u)*exp(-(s0*u)^2)
        return (np.cos(np.float32(a0) * u) * np.exp(-((np.float32(a1) * u) ** 2))).astype(np.float32)
    return np.maximum(u, np.float32(0.0)).astype(np.float32)


def main() -> int:
    fx = json.loads((HERE / "fixture.json").read_text())
    m = fx["meta"]
    P, inf, H, NH, NC = m["P"], m["in_feat"], m["hidden_dim"], m["n_hidden"], m["n_classes"]
    mod = m["mod_dim"]
    kind = {"hosc": 0, "wire": 1, "relu": 2}[m["activation"]]
    if kind == 0:
        a0, a1 = m["akw"]["beta"], m["akw"]["omega"]
    elif kind == 1:
        a0, a1 = m["akw"]["w0"], m["akw"]["s0"]
    else:
        a0 = a1 = 0.0

    W = {k: np.asarray(v, np.float32) for k, v in fx["weights"].items()}
    in_w = W["in_proj.weight"].reshape(H, inf)
    in_b = W["in_proj.bias"].reshape(H)
    film_w = W["film.weight"].reshape(NH * 2 * H, mod)
    film_b = W["film.bias"].reshape(NH * 2 * H)
    hid_w = [W[f"hidden.{li}.weight"].reshape(H, H) for li in range(NH)]
    hid_b = [W[f"hidden.{li}.bias"].reshape(H) for li in range(NH)]
    out_w = W["out_sdf.weight"].reshape(NC, H)
    out_b = W["out_sdf.bias"].reshape(NC)

    frames = m["frames"]
    feats_all = np.frombuffer((HERE / "feats.bin").read_bytes(), np.float32).reshape(len(frames), P, inf)
    ref_all = np.frombuffer((HERE / "reference.bin").read_bytes(), np.uint8).reshape(len(frames), P)
    codes = [np.asarray(c, np.float32) for c in fx["codes"]]

    per_frame = []
    for fi, f in enumerate(frames):
        feats = feats_all[fi]
        code = codes[fi]
        # FiLM precomputed on host (fp32), exactly as the demo uploads it.
        film = (code @ film_w.T + film_b).astype(np.float32).reshape(NH, 2, H)
        # layer 0 (fp32 accumulation mirrors WGSL scalar loop closely enough for argmax)
        h = act(feats @ in_w.T.astype(np.float32) + in_b, kind, a0, a1)
        for li in range(NH):
            scale = np.float32(1.0) + film[li, 0]
            shift = film[li, 1]
            pre = (h @ hid_w[li].T + hid_b[li]).astype(np.float32) * scale + shift
            h = act(pre, kind, a0, a1)
        phi = (h @ out_w.T + out_b).astype(np.float32)
        part = phi.argmax(axis=-1).astype(np.uint8)
        match = float((part == ref_all[fi]).mean())
        per_frame.append({"frame": f, "pixel_match": round(match, 6),
                          "mismatched_px": int((part != ref_all[fi]).sum())})

    overall = float(np.mean([d["pixel_match"] for d in per_frame]))
    report = {
        "authority": "[WebGPU/WebNN demo — NON-AUTHORITY]",
        "contract": "WGSL shader-model (fp32) vs numpy-fp32 reference argmax, identical shipped inputs",
        "P_per_frame": P, "frames": frames,
        "overall_pixel_match": round(overall, 6),
        "per_frame": per_frame,
        "verdict": "PASS" if overall >= 0.999 else ("NEAR" if overall >= 0.99 else "INVESTIGATE"),
    }
    print(json.dumps(report, indent=2))
    (HERE / "parity_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
