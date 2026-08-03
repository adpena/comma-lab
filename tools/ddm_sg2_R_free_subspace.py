# SPDX-License-Identifier: MIT
"""ddm_sg2 -- the EXACT singular spectrum of the contest R operator, and the size of the
byte-neutral free subspace it hands us.

$0. NO SegNet/PoseNet forward or backward. No evaluator slot. `score_claim=false`.

WHY THIS EXISTS
---------------
`ddm_sg2`'s n600 reduction measured the through-R input-sensitivity map `sR` and found it nearly
FLAT: the fragile set (margin < t* = 0.153) is 0.431% of pixels but carries 1.17% of the leverage,
a tilt of only **2.71x** -- not the 238x that the output-side "0.42% scored-active" statistic
suggests.  So spatial (pixel-space) reweighting of the render objective has a small ceiling.

But a pixel-space map is blind to a FREQUENCY-space null space by construction.  The contest R
operator is

    R = D . U ,   U = bicubic upsample 384x512 -> 874x1164 (align_corners=False)
                  D = bilinear downsample 874x1164 -> 384x512 (align_corners=False)

i.e. an up-then-down resample: a LOW-PASS.  Whatever R annihilates, SegNet cannot see, because
SegNet only ever reads `R(x)`.  Render error placed in R's near-null modes is therefore FREE --
free in bytes, free in d_seg -- and it would be invisible to `sR` because it is spread over the
whole frame.  This measures exactly how big that subspace is.

METHOD (exact, deterministic)
-----------------------------
Both interpolate modes are separable, so R = R_y (x) R_x with
    R_x : 512 -> 1164 -> 512     R_y : 384 -> 874 -> 384
Each 1-D matrix is built EXACTLY by pushing unit impulses through the very same torch calls the
production `_R_torch` uses (`tools/precompute_sR_reachability.py::_R_torch`), so this is the real
operator, not a model of it.  Separability is then VERIFIED against the full 2-D operator on random
inputs (a probe that can return the negative) before any spectrum is reported.

The 2-D singular values are the outer product {sigma_i^y * sigma_j^x}, 384*512 = 196,608 of them,
enumerated exactly.

WHAT IS AND IS NOT CLAIMED
--------------------------
The production R contains `clamp(0,255)` and `round()` at camera resolution.  Under the uint8-STE
used everywhere in training, `round` has identity gradient, so the LINEARIZED operator is exactly
D.U -- that is the object whose spectrum is reported, and it is the correct object for a
first-order / gradient-reach argument.  It is NOT a claim that a null-space perturbation is
bit-exactly free through the true rounded pipeline: rounding is a genuine nonlinearity, and a
perturbation that R_lin annihilates can still cross a rounding boundary at camera resolution and
survive.  Therefore every "free" number here is a FIRST-ORDER CEILING on the byte-neutral
subspace, and its realization through the true uint8 path is a separate, owed measurement
(that is precisely the object `ddm_ll1`'s window solve manipulates).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RENDER_HW = (384, 512)
CAMERA_HW = (874, 1164)


def _R_2d(x_bchw):
    """The production R, verbatim in structure from precompute_sR_reachability._R_torch, minus the
    clamp/round (which is identity under the STE linearization and would break linearity here)."""
    import torch.nn.functional as F

    up = F.interpolate(x_bchw, size=CAMERA_HW, mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=RENDER_HW, mode="bilinear", align_corners=False)
    return down


def _build_1d(n_in: int, n_mid: int, axis: str):
    """Exact 1-D resample matrix (n_in -> n_mid -> n_in) built by unit impulses through torch."""
    import torch
    import torch.nn.functional as F

    eye = torch.eye(n_in, dtype=torch.float64)
    if axis == "x":
        t = eye.reshape(n_in, 1, 1, n_in)                      # (N,1,1,W)
        up = F.interpolate(t, size=(1, n_mid), mode="bicubic", align_corners=False)
        dn = F.interpolate(up, size=(1, n_in), mode="bilinear", align_corners=False)
        M = dn.reshape(n_in, n_in)
    else:
        t = eye.reshape(n_in, 1, n_in, 1)                      # (N,1,H,1)
        up = F.interpolate(t, size=(n_mid, 1), mode="bicubic", align_corners=False)
        dn = F.interpolate(up, size=(n_in, 1), mode="bilinear", align_corners=False)
        M = dn.reshape(n_in, n_in)
    # rows of M = image of each basis vector => operator matrix is M^T applied to a column vector
    return M.T.numpy().astype(np.float64)


def verify_separability(Rx: np.ndarray, Ry: np.ndarray, *, trials: int = 3, seed: int = 0) -> dict:
    """Probe that CAN return the negative: compare the true 2-D R against Ry @ X @ Rx^T."""
    import torch

    rng = np.random.default_rng(seed)
    rels, scale_checks = [], []
    for _ in range(trials):
        X = rng.standard_normal(RENDER_HW)
        t = torch.from_numpy(X)[None, None].double()
        true = _R_2d(t)[0, 0].numpy()
        sep = Ry @ X @ Rx.T
        num = float(np.abs(true - sep).max())
        den = float(np.abs(true).max())
        rels.append(num / den)
        scale_checks.append(den)
    # negative control: a deliberately WRONG factorization must fail loudly
    X = rng.standard_normal(RENDER_HW)
    t = torch.from_numpy(X)[None, None].double()
    true = _R_2d(t)[0, 0].numpy()
    wrong = float(np.abs(true - (Ry @ X)).max()) / float(np.abs(true).max())
    return {"max_rel_err": max(rels), "per_trial_rel_err": rels,
            "negative_control_wrong_factorization_rel_err": wrong,
            "probe_can_return_negative": wrong > 1e-3}


def spectrum(Rx: np.ndarray, Ry: np.ndarray) -> dict:
    sx = np.linalg.svd(Rx, compute_uv=False)
    sy = np.linalg.svd(Ry, compute_uv=False)
    s2d = np.sort((sy[:, None] * sx[None, :]).ravel())[::-1]
    n = s2d.size
    e = s2d ** 2
    tot_e = float(e.sum())
    out = {
        "n_modes": int(n),
        "sigma_x": {"max": float(sx.max()), "min": float(sx.min()), "median": float(np.median(sx))},
        "sigma_y": {"max": float(sy.max()), "min": float(sy.min()), "median": float(np.median(sy))},
        "sigma_2d": {"max": float(s2d.max()), "min": float(s2d.min()), "median": float(np.median(s2d))},
        # A white/isotropic render error keeps this fraction of its ENERGY through R:
        "white_error_energy_survival": tot_e / n,
        "white_error_amplitude_survival": float(np.sqrt(tot_e / n)),
    }
    frac_modes = {}
    for thr in (0.5, 0.25, 0.1, 0.03, 0.01, 0.003, 0.001):
        k = int(np.count_nonzero(s2d < thr))
        frac_modes[f"sigma_lt_{thr:g}"] = {
            "n_modes": k, "mode_frac": k / n,
            "energy_frac_in_those_modes": float(e[s2d < thr].sum() / tot_e),
        }
    out["attenuated_modes"] = frac_modes
    # cumulative energy: how few modes carry most of what survives
    ce = np.cumsum(e) / tot_e
    out["modes_for_energy"] = {f"{q:g}": int(np.searchsorted(ce, q) + 1) for q in (0.5, 0.9, 0.99, 0.999)}
    return out


def _build_down_1d(n_in: int, n_out: int, axis: str) -> np.ndarray:
    """Exact 1-D bilinear DOWNsample matrix (n_in -> n_out), align_corners=False, antialias=False:
    the scorer's own `preprocess_input` resize (upstream/modules.py:73,109). Shape (n_out, n_in)."""
    import torch
    import torch.nn.functional as F

    eye = torch.eye(n_in, dtype=torch.float64)
    if axis == "x":
        t = eye.reshape(n_in, 1, 1, n_in)
        d = F.interpolate(t, size=(1, n_out), mode="bilinear", align_corners=False)
        M = d.reshape(n_in, n_out)
    else:
        t = eye.reshape(n_in, 1, n_in, 1)
        d = F.interpolate(t, size=(n_out, 1), mode="bilinear", align_corners=False)
        M = d.reshape(n_in, n_out)
    return M.T.numpy().astype(np.float64)  # (n_out, n_in)


def camera_space_free_subspace() -> dict:
    """The operator our SHIPPED error actually passes through.

    inflate.py writes CAMERA-resolution frames (874x1164); the scorer's own preprocess_input then
    resizes to 384x512 with `mode='bilinear'` (align_corners=False, antialias=False).  So the map
    from our shipped per-pixel error to anything SegNet can see is D = D_y (x) D_x, a genuine
    DIMENSION REDUCTION 1,017,336 -> 196,608.  Unlike R (a resample round-trip), D has a huge exact
    null space -- and that null space is EXACTLY the seg-free direction set.
    """
    Dx = _build_down_1d(CAMERA_HW[1], RENDER_HW[1], "x")   # (512, 1164)
    Dy = _build_down_1d(CAMERA_HW[0], RENDER_HW[0], "y")   # (384, 874)
    sx = np.linalg.svd(Dx, compute_uv=False)
    sy = np.linalg.svd(Dy, compute_uv=False)
    s2d = np.sort((sy[:, None] * sx[None, :]).ravel())[::-1]
    n_cam = CAMERA_HW[0] * CAMERA_HW[1]
    n_seg = RENDER_HW[0] * RENDER_HW[1]
    # exactly-unsampled camera pixels: a column of D that is identically zero
    zero_cols_x = int(np.count_nonzero(np.abs(Dx).sum(axis=0) == 0.0))
    zero_cols_y = int(np.count_nonzero(np.abs(Dy).sum(axis=0) == 0.0))
    touched_x = CAMERA_HW[1] - zero_cols_x
    touched_y = CAMERA_HW[0] - zero_cols_y
    untouched_px = n_cam - touched_x * touched_y
    e = s2d ** 2
    return {
        "operator": "D = scorer preprocess_input bilinear resize 874x1164 -> 384x512 "
                    "(align_corners=False, antialias=False), upstream/modules.py:73,109",
        "dim_camera": n_cam, "dim_scorer": n_seg,
        "rank": int(s2d.size),
        "null_space_dim": n_cam - int(s2d.size),
        "null_space_dim_frac": (n_cam - int(s2d.size)) / n_cam,
        "exactly_unsampled_camera_px": untouched_px,
        "exactly_unsampled_frac": untouched_px / n_cam,
        "sigma_2d": {"max": float(s2d.max()), "min": float(s2d.min()), "median": float(np.median(s2d))},
        "condition_on_row_space": float(s2d.max() / s2d.min()),
        # An isotropic (white) camera-space error keeps this fraction of its energy through D:
        "white_error_energy_survival": float(e.sum() / n_cam),
        "white_error_amplitude_survival": float(np.sqrt(e.sum() / n_cam)),
        "note": "null_space_dim_frac is the fraction of shipped-error DIRECTIONS that are exactly "
                "invisible to SegNet. It is seg-FREE capacity, NOT a d_seg lever: moving error into "
                "it cannot lower d_seg, it can only make room. The d_seg-relevant question is the "
                "reach spread WITHIN the row space, given by condition_on_row_space.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Exact singular spectrum of the contest R operator.")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    Rx = _build_1d(RENDER_HW[1], CAMERA_HW[1], "x")
    Ry = _build_1d(RENDER_HW[0], CAMERA_HW[0], "y")
    ver = verify_separability(Rx, Ry)
    out = {
        "tool": "ddm_sg2_R_free_subspace", "scorer_fired": False, "score_claim": False,
        "axis": "[macOS-CPU advisory] $0 exact linear-operator spectrum",
        "operator": "R_lin = bilinear_down(874x1164 -> 384x512) . bicubic_up(384x512 -> 874x1164), align_corners=False",
        "separability_check": ver,
    }
    if ver["max_rel_err"] > 1e-9:
        out["SEPARABILITY_REFUTED"] = True
        print(json.dumps(out, indent=2))
        return 2
    out["spectrum"] = spectrum(Rx, Ry)
    out["camera_space"] = camera_space_free_subspace()
    txt = json.dumps(out, indent=2)
    print(txt)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
