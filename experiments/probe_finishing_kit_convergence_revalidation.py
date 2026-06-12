# SPDX-License-Identifier: MIT
"""Finishing-kit CONVERGENCE re-validation — the decisive under-power test.

THE QUESTION (operator's "training time / config for true signal" lens):
the distortion finishing-kit measured PR98+T10 = **-0.058 distortion-score** at
n=24 on the MID-BASIN ep340 fork-point (POSE-axis: d_pose 0.001478 -> 0.000401).
This is the SAME operating-point risk that flipped LeverD ("GO" mid-basin -> NO-GO
at convergence). The mid-basin decoder is UNDER-TRAINED; a converged decoder
(trained against the real scorer) may have ALREADY learned the per-frame color
balance the PR98/T10 bias supplies -> the win SHRINKS toward 0 (an under-training
artifact, do NOT bank) OR the win PERSISTS (a real uint8-round-trip artifact, like
PR101's shipped PR98 on a converged decoder -> bank as a convergence-robust win).

RESOLUTION: re-fit the PR98/T10 constants ON the MORE-CONVERGED basin **ep2120
best** decoder (d_pose 0.00034, ~2.4x lower than the fork-point's 0.000831 over
600 pairs / ~3.7x lower than the n=24 mid-basin slice's 0.001478) + re-measure the
gain. Two slices for robustness against small-slice ``sqrt(10*d_pose)`` noise.

This RE-USES the probe's fitting functions (``refit_pr98_bias`` / ``fit_t10_affine``
/ ``_render_camera_pairs`` / ``_decode_gt_pairs`` / ``_measure_bias_affine``) — it
is NOT a reimplementation; only the decoder LOADER (converged archive vs the
fork-point) + the multi-slice harness are new. The constants are re-fit ON the
converged decoder (C's finding: PR101's canonical constants do NOT transfer; the
ep340 constants may not transfer to ep2120 either).

Authority: ``[contest-CPU advisory] NON-PROMOTABLE`` — every number is a frozen-CPU
advisory measurement; no byte-closed ``upstream/evaluate.py`` row; frontier UNMOVED.
This is a MEANS (a convergence re-validation) toward the END (a lower exact score);
it moves no row. REAL frozen scorer + REAL ``frame_utils.yuv420_to_rgb`` GT
(NO MPS, NO PyAV-rgb24, NO synthetic fixture — per CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure the repo root is importable (so ``experiments`` is a package) regardless of
# the invocation cwd — the original probe only imports ``tac.*`` (already on path via
# ``src/``); this probe RE-USES the sibling probe module, so it needs the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# RE-USE the probe's fitting + rendering + scorer plumbing (no reimplementation).
from experiments.probe_track_a_distortion_finishing_kit import (  # noqa: E402
    _decode_gt_pairs,
    _measure_bias_affine,
    _render_camera_pairs,
    _seg_score,
    fit_t10_affine,
    refit_pr98_bias,
)

_ADVISORY = "[contest-CPU advisory] NON-PROMOTABLE"

# The MORE-CONVERGED basin best (ep2120: d_seg 0.0026, d_pose 0.00034, S 0.378).
_CONVERGED_ARCHIVE = Path(
    "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best/best_archive.bin"
)
# The MID-BASIN fork-point the original n=24 probe used (ep340: d_pose 0.000831).
_FORK_ARCHIVE = Path(
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z/best/best_archive.bin"
)


def _load_decoder(archive_path: Path) -> tuple[torch.nn.Module, torch.Tensor, dict[str, Any]]:
    """Load a base_ch=20 decoder + latents from a byte-closed archive (the SAME
    vendored ``parse_archive`` + ``HNeRVDecoder`` the production driver uses — no
    fork)."""
    from tac.torch_vehicle.driver import import_vendored_bundle

    v = import_vendored_bundle()
    arch = archive_path.read_bytes()
    dec_sd, latents, meta = v.parse_archive(arch)
    dec = v.HNeRVDecoder(
        latent_dim=meta.get("latent_dim", 28),
        base_channels=meta.get("base_channels", 20),
        eval_size=tuple(meta.get("eval_size", (384, 512))),
    )
    dec.load_state_dict({k: vv for k, vv in dec_sd.items()})
    dec.eval()
    return dec, latents, meta


def _measure_kit_on_slice(
    net: Any,
    dec: torch.nn.Module,
    latents: torch.Tensor,
    video_path: str,
    n: int,
    *,
    pr98_only_bias: np.ndarray | None = None,
    affine_scale: np.ndarray | None = None,
    affine_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    """Render+decode a slice, fit (or apply supplied) PR98+T10, return measured deltas.

    If ``pr98_only_bias``/``affine_*`` are None: re-fit on THIS slice (the
    convergence re-fit). If supplied: APPLY the supplied constants (cross-slice
    transfer test — fit on slice-1, measure on slice-2)."""
    cam_float = _render_camera_pairs(dec, latents[:n], n)
    gt_u8 = _decode_gt_pairs(video_path, n)
    base = _measure_bias_affine(net, gt_u8, cam_float, None, None)
    base_score = _seg_score(base["d_seg"], base["d_pose"])

    out: dict[str, Any] = {
        "n_pairs": n,
        "baseline_d_seg": base["d_seg"],
        "baseline_d_pose": base["d_pose"],
        "baseline_distortion_score": base_score,
    }

    if pr98_only_bias is None:
        # RE-FIT PR98 + T10 on this (converged) slice.
        a = refit_pr98_bias(net, gt_u8, cam_float, base_distortion=base)
        pr98_bias = np.asarray(a["best_bias_frame_channel"], dtype=np.float64)
        b = fit_t10_affine(net, gt_u8, cam_float, pr98_bias)
        out["pr98_refit"] = {
            "best_bias_frame_channel": a["best_bias_frame_channel"],
            "fit_d_seg": a["fit_d_seg"],
            "fit_d_pose": a["fit_d_pose"],
            "fit_distortion_score": a["fit_distortion_score"],
            "pr98_delta_vs_base": a["joint_delta_vs_base"],
            "canonical_pr98_measured": a["candidates_measured"].get("canonical_pr98"),
        }
        out["t10_affine"] = {
            "best_scale_frame_channel": b["best_scale_frame_channel"],
            "best_bias_frame_channel": b["best_bias_frame_channel"],
            "affine_d_seg": b["affine_d_seg"],
            "affine_d_pose": b["affine_d_pose"],
            "affine_distortion_score": b["affine_distortion_score"],
            "affine_delta_vs_pr98": b["affine_delta_vs_pr98"],
        }
        out["full_kit_delta_vs_base"] = b["affine_distortion_score"] - base_score
        out["full_kit_distortion_score"] = b["affine_distortion_score"]
    else:
        # APPLY supplied constants (transfer test).
        ps = pr98_only_bias
        pr98_m = _measure_bias_affine(net, gt_u8, cam_float, None, ps)
        pr98_sc = _seg_score(pr98_m["d_seg"], pr98_m["d_pose"])
        full_m = _measure_bias_affine(net, gt_u8, cam_float, affine_scale, affine_bias)
        full_sc = _seg_score(full_m["d_seg"], full_m["d_pose"])
        out["applied_pr98"] = {
            "d_seg": pr98_m["d_seg"], "d_pose": pr98_m["d_pose"],
            "distortion_score": pr98_sc, "pr98_delta_vs_base": pr98_sc - base_score,
        }
        out["applied_full_kit"] = {
            "d_seg": full_m["d_seg"], "d_pose": full_m["d_pose"],
            "distortion_score": full_sc, "full_kit_delta_vs_base": full_sc - base_score,
        }
        out["full_kit_delta_vs_base"] = full_sc - base_score
        out["full_kit_distortion_score"] = full_sc
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-primary", type=int, default=24, help="primary re-fit slice (matches the n=24 mid-basin probe)")
    p.add_argument("--n-secondary", type=int, default=48, help="second slice for robustness (small-slice d_pose noise)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    t0 = time.time()
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.vendored_imports import import_vendored

    data = import_vendored("data")
    video_path = data.get_default_video_path()
    net = load_frozen_distortion_net(device="cpu")

    dec_c, lat_c, meta_c = _load_decoder(_CONVERGED_ARCHIVE)
    print(f"[reval] loaded scorer + CONVERGED decoder (ep2120) in {time.time()-t0:.1f}s meta={meta_c}", flush=True)

    result: dict[str, Any] = {
        "authority": _ADVISORY,
        "converged_archive": str(_CONVERGED_ARCHIVE),
        "converged_meta": meta_c,
        "mid_basin_n24_reference": {
            "note": "from track_a_distortion_finishing_kit_20260612T220727Z.md (ep340 fork-point)",
            "baseline_d_seg": 0.003532, "baseline_d_pose": 0.001478,
            "baseline_distortion_score": 0.47479,
            "pr98_distortion_score": 0.42681, "pr98_delta": -0.047981,
            "full_kit_distortion_score": 0.41677, "full_kit_delta": -0.058,
        },
        "n_primary": args.n_primary,
        "n_secondary": args.n_secondary,
    }

    out_path = args.out
    if out_path is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = Path(f".omx/research/finishing_kit_convergence_revalidation_{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush() -> None:
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    _flush()

    # SLICE 1 (primary; n matches the mid-basin probe so the comparison is apples-to-apples).
    print(f"[reval] SLICE-1 re-fit on CONVERGED decoder, n={args.n_primary} ...", flush=True)
    s1 = _measure_kit_on_slice(net, dec_c, lat_c, video_path, args.n_primary)
    result["converged_slice1_refit"] = s1
    print(f"[reval] SLICE-1: base_dscore={s1['baseline_distortion_score']:.6f} "
          f"full_kit_delta={s1['full_kit_delta_vs_base']:.6f}", flush=True)
    _flush()

    # SLICE 2 (robustness; different n -> different pairs, exposes small-slice d_pose noise).
    print(f"[reval] SLICE-2 re-fit on CONVERGED decoder, n={args.n_secondary} ...", flush=True)
    s2 = _measure_kit_on_slice(net, dec_c, lat_c, video_path, args.n_secondary)
    result["converged_slice2_refit"] = s2
    print(f"[reval] SLICE-2: base_dscore={s2['baseline_distortion_score']:.6f} "
          f"full_kit_delta={s2['full_kit_delta_vs_base']:.6f}", flush=True)
    _flush()

    # CROSS-SLICE TRANSFER: apply slice-1's fitted constants to slice-2 (does the
    # converged fit generalize, or is each slice's gain a same-slice overfit?).
    s1_pr98 = np.asarray(s1["pr98_refit"]["best_bias_frame_channel"], dtype=np.float64)
    s1_scale = np.asarray(s1["t10_affine"]["best_scale_frame_channel"], dtype=np.float64)
    s1_bias = np.asarray(s1["t10_affine"]["best_bias_frame_channel"], dtype=np.float64)
    print("[reval] CROSS-SLICE transfer: slice-1 constants -> slice-2 ...", flush=True)
    transfer = _measure_kit_on_slice(
        net, dec_c, lat_c, video_path, args.n_secondary,
        pr98_only_bias=s1_pr98, affine_scale=s1_scale, affine_bias=s1_bias,
    )
    result["cross_slice_transfer_s1_to_s2"] = transfer
    print(f"[reval] TRANSFER: full_kit_delta={transfer['full_kit_delta_vs_base']:.6f}", flush=True)
    _flush()

    # VERDICT: SHRINKS toward 0 (under-training artifact) vs PERSISTS (round-trip artifact).
    mid = result["mid_basin_n24_reference"]["full_kit_delta"]
    c1 = s1["full_kit_delta_vs_base"]
    c2 = s2["full_kit_delta_vs_base"]
    ct = transfer["full_kit_delta_vs_base"]
    # "PERSISTS" = the converged gain retains >= 25% of the mid-basin magnitude AND
    # is consistent in sign across both slices + the transfer. "SHRINKS" otherwise.
    converged_mean = float(np.mean([c1, c2]))
    retained_frac = converged_mean / mid if mid != 0 else 0.0
    sign_consistent = (c1 < 0) and (c2 < 0) and (ct < 0)
    persists = bool(sign_consistent and retained_frac >= 0.25)
    result["verdict"] = {
        "mid_basin_n24_full_kit_delta": mid,
        "converged_slice1_full_kit_delta": c1,
        "converged_slice2_full_kit_delta": c2,
        "converged_transfer_full_kit_delta": ct,
        "converged_mean_full_kit_delta": converged_mean,
        "retained_fraction_vs_mid_basin": retained_frac,
        "sign_consistent_across_slices_and_transfer": sign_consistent,
        "verdict": "PERSISTS" if persists else "SHRINKS",
        "interpretation": (
            "PERSISTS -> real uint8-round-trip / color-balance artifact the converged "
            "decoder did NOT learn away (like PR101's shipped PR98); BANK as a "
            "convergence-robust candidate. SHRINKS -> the gain was an under-training "
            "artifact (the converged decoder already learned the color balance); do "
            "NOT bank — it is a mid-basin mirage."
        ),
    }
    result["elapsed_seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[reval] VERDICT={result['verdict']['verdict']} "
          f"mid={mid:.4f} conv_mean={converged_mean:.6f} retained={retained_frac:.2f}", flush=True)
    print(f"[reval] wrote {out_path} ({result['elapsed_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
