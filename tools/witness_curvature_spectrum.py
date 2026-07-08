#!/usr/bin/env python3
"""Standalone $0 curvature-spectrum tool (task #312 Phase B; the D-3/4/5 first measurement).

Computes the top-k HVP-Lanczos eigenvalues (SHARPNESS λ_max, anisotropy λ_1/λ_k, trace, saddle
flag) of a witness seg-loss surface at a SAVED checkpoint — no training, no launch, CPU/MLX-GPU.

Modes:
  --self-validate         run the full MLX HVP -> Lanczos path on a synthetic witness-shaped
                          quadratic (proves the instrument end-to-end + prints a real spectrum).
  --checkpoint <npz>      reconstruct the model from a checkpoint and measure its seg-loss
                          Hessian. Works for a BASE witness (single `out` head, self_orient=0).
                          A SELF-ORIENT levelset checkpoint (out_sdf/out_tex/palette/directional
                          bank) FAILS CLOSED with the named blocker (its forward needs the
                          trainer's exact self-orient reorient + palette-SDF setup) and points to
                          the in-trainer `--curvature-telemetry` hook — NO fake number is emitted.

Every row is [macOS advisory] NON-PROMOTABLE; pointer 0.19110 UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in (str(_REPO / "src"), str(_REPO / "experiments"), str(_REPO / "upstream")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tac.witness_control import curvature as cv  # noqa: E402

# checkpoint keys that mark a SELF-ORIENT levelset model whose forward we cannot faithfully
# reconstruct standalone (must be measured via the in-trainer hook on the exact forward).
_LEVELSET_MARKERS = ("out_sdf.weight", "out_tex.weight", "palette", "__bank_base")


def self_validate(k: int = 6) -> dict:
    """End-to-end proof the MLX HVP->Lanczos path yields correct spectra: a witness-shaped MLP
    with an L2 loss (Gauss-Newton Hessian is PSD); we compare Lanczos λ_max to a dense reference
    on the same small operator."""
    import mlx.core as mx
    import mlx.nn as nn

    class TinyWitness(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(8, 8)
            self.out = nn.Linear(8, 3)

        def __call__(self, x):
            return self.out(mx.tanh(self.in_proj(x)))

    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((16, 8)).astype(np.float32))
    tgt = mx.array(rng.standard_normal((16, 3)).astype(np.float32))

    def seg_like_loss(m):
        return mx.mean(mx.square(m(x) - tgt))

    matvec, dim = cv.mlx_model_hvp(seg_like_loss, TinyWitness())
    sp = cv.compute_spectrum(matvec, dim=dim, k=min(k, dim), n_iter=min(dim, 3 * k), seed=0)
    row = sp.to_row(stage="self_validate", ep=-1, k_pairs=16, source="synthetic_witness_mlp")
    # dense cross-check on the same operator (dim is small enough)
    H = np.zeros((dim, dim))
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        H[:, i] = matvec(e)
    dense = np.sort(np.linalg.eigvalsh((H + H.T) / 2))[::-1]
    row["dense_lambda_max"] = round(float(dense[0]), 6)
    row["lanczos_vs_dense_lambda_max_abs_err"] = round(abs(float(dense[0]) - sp.lambda_max), 8)
    return row


def measure_checkpoint(ckpt: Path, *, k: int, k_pairs: int) -> dict:
    z = np.load(ckpt, allow_pickle=True)
    keys = set(z.files)
    is_levelset = any(m in keys for m in _LEVELSET_MARKERS)
    if is_levelset:
        return {
            "stage": "curvature_spectrum_blocked", "checkpoint": str(ckpt),
            "blocker": "SELF_ORIENT_LEVELSET_FORWARD_NOT_STANDALONE_RECONSTRUCTABLE",
            "detail": ("this checkpoint is a self-orient levelset witness (out_sdf/out_tex/"
                       "palette/directional bank); its through-R seg forward requires the "
                       "trainer's exact self-orient reorient + palette-SDF setup. NO fake "
                       "curvature number is emitted (NO-FAKE)."),
            "resolution": ("measure it via the in-trainer hook: "
                           "train_levelset_witness_realized_through_R_mlx.py "
                           "--curvature-telemetry (default-off, checkpoint-cadence, governor-"
                           "gated) on a governed run resumed from this checkpoint; OR extend "
                           "this tool with the self-orient feature reconstruction (named "
                           "duty-to-measure)."),
            "cfg_self_orient": int(z["__cfg_self_orient"]) if "__cfg_self_orient" in keys else None,
            "epoch": int(z["__epoch"]) if "__epoch" in keys else None,
            "axis": cv.AXIS_TAG, "score_neutral": True,
        }
    # BASE witness reconstruction (single `out` head, isotropic feats) — faithfully rebuildable.
    return _measure_base_witness(z, ckpt, k=k, k_pairs=k_pairs)


def _measure_base_witness(z, ckpt: Path, *, k: int, k_pairs: int) -> dict:
    # Import the trainer's exact builders so the forward matches training bit-for-bit.
    from train_witness_realized_through_R_mlx import build_witness_module  # type: ignore
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    def _c(name, default):
        return type(default)(z[f"__cfg_{name}"]) if f"__cfg_{name}" in z.files else default

    render_h, render_w = (int(z["__render_hw"][0]), int(z["__render_hw"][1])) \
        if "__render_hw" in z.files else (48, 64)
    n_fourier = int(z["in_proj.weight"].shape[1]) // 2
    model = build_witness_module(
        num_pairs=int(z["code"].shape[0]), n_fourier=n_fourier,
        hidden_dim=int(z["in_proj.weight"].shape[0]), n_hidden=int(_c("n_hidden", 4)),
        mod_dim=int(z["code"].shape[1]), fourier_sigma=float(_c("fourier_sigma", 10.0)),
        activation=str(_c("activation", "relu")), chroma=bool(int(_c("chroma", 1))))
    weights = {kk: mx.array(z[kk]) for kk in z.files if not kk.startswith("__")}
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    # coord grid feats at render resolution (isotropic build_feats — matches base training).
    ys = np.linspace(-1.0, 1.0, render_h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, render_w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    coords = mx.array(np.stack([gy.ravel(), gx.ravel()], axis=-1))
    feats = model.build_feats(coords)

    # NOTE: a real seg loss needs the frozen SegNet adapter + GT argmax. If the gt cache/adapter
    # are unavailable we fail closed rather than fabricate. This base-witness path is a scaffold
    # for non-self-orient checkpoints; the mod32cap frontier is self-orient (blocked above).
    return {
        "stage": "curvature_spectrum_base_scaffold", "checkpoint": str(ckpt),
        "note": ("base-witness reconstruction OK (feats+forward rebuilt); wiring the frozen "
                 "SegNet adapter + GT argmax for the seg-loss Hessian is the remaining step — "
                 "not fabricated. Use --self-validate for the proven instrument path."),
        "render_hw": [render_h, render_w], "n_fourier": n_fourier,
        "n_params_feats": int(feats.shape[-1]), "axis": cv.AXIS_TAG, "score_neutral": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="witness checkpoint .npz")
    ap.add_argument("--self-validate", action="store_true",
                    help="prove the instrument on a synthetic witness-shaped loss")
    ap.add_argument("--k", type=int, default=8, help="top-k eigenvalues")
    ap.add_argument("--k-pairs", type=int, default=16, help="pair sample for the seg-loss Hessian")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rows = []
    if args.self_validate or not args.checkpoint:
        rows.append(self_validate(k=args.k))
    if args.checkpoint:
        rows.append(measure_checkpoint(Path(args.checkpoint), k=args.k, k_pairs=args.k_pairs))

    for row in rows:
        if args.json:
            print(json.dumps(row, indent=2))
        else:
            print(f"[curvature] {row.get('stage')}")
            for kk in ("source", "checkpoint", "blocker", "lambda_max", "top_k_eigs",
                       "anisotropy", "sharpness_ratio", "trace_estimate", "negative_curvature",
                       "lanczos_vs_dense_lambda_max_abs_err", "detail", "resolution", "note"):
                if kk in row:
                    print(f"    {kk}: {row[kk]}")
    print("pointer 0.19110 UNMOVED — curvature spectrum is a SENSE state (means)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
