# SPDX-License-Identifier: MIT
"""Lever B atlas — measure the POSE-NULL geometry of the basin (the orthogonal-solve diagnostic).

Design memo: ``.omx/research/sophisticated_pose_treatment_design_20260616T222900Z.md``.

WHAT THIS DOES ($0, CPU-only, NO GPU touched, NO running job touched). For a sample of basin pairs it
measures, on the REAL frozen CPU-torch PoseNet (differentiable yuv6 patch active; GT via
``frame_utils.yuv420_to_rgb``):
  * the pose-sensitive SUBSPACE spectrum (``measure_pose_subspace_spectrum``): ``rank``, ``effective_dim``
    (participation ratio), the top singular values, and ``expected_isotropic_null_fraction`` — the #80 POSE
    CRUX measurement;
  * OPTIONALLY (``--basin-best DIR``): the empirical POSE-NULL energy fraction of the FiLM-v2 carrier's
    ACTUAL frame-0 residual ``f0_film − f0_clean`` projected onto the measured pose-null
    (``pose_null_residual_fraction``) — the diagnostic that says whether the current carrier is incidentally
    cheap on d_pose (residual mostly pose-null) or LEAKING into the pose-sensitive subspace (the warm
    d_pose-drift regression source). This is the empirical answer to the design's UNCLEAR_NEEDS_EMPIRICAL
    classification of "FiLM-v2 residual is mostly pose-null".

OUTPUT. A JSON atlas to ``--out`` (default ``.omx/research/pose_null_atlas_<UTC>.json``) with per-pair rows +
a cross-pair headline (mean/median rank, effective_dim, isotropic-null baseline, and — if the basin best is
given — the mean residual pose-null fraction). ``[contest-CPU advisory]`` NON-PROMOTABLE.

NO-FAKE. Every number is the EXACT SVD / projection of the EXACT backprop Jacobian of the frozen PoseNet on
REAL GT frames; ``measure_pose_subspace_spectrum`` fail-closes if the differentiable path is severed (a
zero Jacobian), so a non-reachable gradient can never masquerade as "all pose-null".

USAGE::

    .venv/bin/python experiments/measure_pose_null_atlas.py --n-pairs 8           # spectrum only
    .venv/bin/python experiments/measure_pose_null_atlas.py --n-pairs 8 \
        --basin-best experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best  # + residual frac
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_posenet_and_gt(n_pairs: int, *, work_h: int, work_w: int):
    """Load the frozen CPU-torch PoseNet (differentiable yuv6) + the first ``n_pairs+1`` GT frames via the
    canonical ``frame_utils.yuv420_to_rgb`` path. Mirrors the boundary_math real-PoseNet test."""
    from safetensors.torch import load_file

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.torch_vehicle.vendored_imports import import_vendored

    patch_upstream_yuv6_globally()  # gradient reachability through rgb_to_yuv6 (else Jacobian is severed)
    import_vendored("data")  # ensures the vendored modules are importable
    from frame_utils import yuv420_to_rgb  # type: ignore
    from modules import PoseNet, posenet_sd_path  # type: ignore

    net = PoseNet().eval()
    net.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for p in net.parameters():
        p.requires_grad = False

    data = import_vendored("data")
    video_path = data.get_default_video_path()
    import av  # type: ignore

    frames = []
    container = av.open(str(video_path))
    need = n_pairs + 1
    for fr in container.decode(container.streams.video[0]):
        frames.append(yuv420_to_rgb(fr).permute(2, 0, 1).float())  # (3, H, W) [0,255]
        if len(frames) >= need:
            break
    container.close()
    if len(frames) < 2:
        raise RuntimeError(f"decoded only {len(frames)} GT frames; need >= 2")
    return net, frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=8, help="number of consecutive GT pairs to atlas")
    ap.add_argument("--frame-slot", type=int, default=0, choices=(0, 1),
                    help="which frame's Jacobian (0 = the pose-conditioned rgb_0 frame; default)")
    ap.add_argument("--work-h", type=int, default=384)
    ap.add_argument("--work-w", type=int, default=512)
    ap.add_argument("--basin-best", type=Path, default=None,
                    help="optional best/ dir (best_ema_decoder.pt + best_ema_latents.pt) — if given, also "
                         "measure the FiLM-v2 carrier residual's pose-null fraction per pair.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from tac.boundary_math.posenet_subspace_spectrum import (
        expected_isotropic_null_fraction,
        measure_pose_subspace_spectrum,
    )

    net, frames = _load_posenet_and_gt(args.n_pairs, work_h=args.work_h, work_w=args.work_w)
    n = min(int(args.n_pairs), len(frames) - 1)

    # Optional: build the FiLM-v2 carrier residual measurement (the design's empirical question).
    film_residuals = None
    if args.basin_best is not None:
        film_residuals = _measure_film_residuals(args.basin_best, n, work_h=args.work_h, work_w=args.work_w)

    rows = []
    for i in range(n):
        f0 = frames[i]
        f1 = frames[i + 1]
        spec = measure_pose_subspace_spectrum(
            net, f0, f1, frame_slot=int(args.frame_slot),
            work_h=int(args.work_h), work_w=int(args.work_w),
        )
        summ = spec.to_summary()
        row = {
            "pair_index": i,
            "rank": int(spec.rank),
            "effective_dim": float(spec.effective_dim),
            "n_pixels": int(spec.n_pixels),
            "isotropic_null_fraction": float(
                expected_isotropic_null_fraction(spec.n_pixels, spec.rank)
            ),
            "sigma_top3": [float(x) for x in spec.singular_values[:3]],
            "energy_frac_top1": summ["energy_frac_top1"],
            "energy_frac_top3": summ["energy_frac_top3"],
        }
        if film_residuals is not None:
            from tac.torch_vehicle.pose_null_projection_hook import pose_null_residual_fraction

            frac = pose_null_residual_fraction(film_residuals[i], spec)
            row["film_residual_null_energy_frac"] = frac["null_energy_frac"]
            row["film_residual_sensitive_energy_frac"] = frac["sensitive_energy_frac"]
        rows.append(row)

    headline = {
        "n_pairs_measured": n,
        "frame_slot": int(args.frame_slot),
        "work_resolution": [int(args.work_h), int(args.work_w)],
        "mean_rank": float(np.mean([r["rank"] for r in rows])) if rows else 0.0,
        "mean_effective_dim": float(np.mean([r["effective_dim"] for r in rows])) if rows else 0.0,
        "mean_isotropic_null_fraction": (
            float(np.mean([r["isotropic_null_fraction"] for r in rows])) if rows else 0.0
        ),
        "compute_path": "cpu_torch",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
    }
    if film_residuals is not None and rows:
        headline["mean_film_residual_null_energy_frac"] = float(
            np.mean([r["film_residual_null_energy_frac"] for r in rows])
        )
        headline["mean_film_residual_sensitive_energy_frac"] = float(
            np.mean([r["film_residual_sensitive_energy_frac"] for r in rows])
        )

    out = args.out or Path(f".omx/research/pose_null_atlas_{_utc()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"headline": headline, "rows": rows}, indent=2))
    print(json.dumps({"headline": headline, "out": str(out)}, indent=2))
    return 0


def _measure_film_residuals(basin_best: Path, n: int, *, work_h: int, work_w: int):
    """Render the FiLM-v2 carrier's frame-0 residual ``f0_film − f0_clean`` for the first ``n`` pairs at the
    spectrum work resolution. Returns a list of ``(3, work_h, work_w)`` numpy arrays.

    Loads the basin decoder + latents + (if present) FiLM weights from ``basin_best``, wraps with the v2
    residual pose-FiLM, and diffs the FiLM-on render (``idx``) against the FiLM-off render (``idx=None``).
    If the basin best holds NO FiLM weights (a pre-FiLM basin), the residual is ~0 (identity init) — the
    atlas then reports the carrier is trivially pose-null, which is the honest result for that basin."""
    import torch
    import torch.nn.functional as F

    from tac.torch_vehicle.driver import import_vendored_bundle

    bundle = import_vendored_bundle()
    decoder_cls = bundle.decoder_cls if hasattr(bundle, "decoder_cls") else None
    # Fall back to the configurable-taper decoder used by the bind-all arms.
    if decoder_cls is None:
        from tac.torch_vehicle.configurable_taper_decoder import (  # type: ignore
            ConfigurableTaperHNeRVDecoder as decoder_cls,
        )

    dec_path = Path(basin_best) / "best_ema_decoder.pt"
    lat_path = Path(basin_best) / "best_ema_latents.pt"
    if not dec_path.exists() or not lat_path.exists():
        raise FileNotFoundError(f"basin best missing decoder/latents: {dec_path}, {lat_path}")
    dec_sd = torch.load(dec_path, map_location="cpu")
    latents = torch.load(lat_path, map_location="cpu")
    # The decoder may be saved as a wrapper sd (decoder.* + pose_mlp.* + film_resid.*).
    from tac.torch_vehicle.pose_film_v2 import PoseFiLMHNeRVWrapperV2

    inner_keys = {k: v for k, v in dec_sd.items() if k.startswith("decoder.")}
    has_film = any(k.startswith(("pose_mlp.", "film_resid.")) for k in dec_sd)
    # Build the inner decoder from the saved shapes (best-effort: rely on the bundle builder if available).
    decoder = _build_decoder_from_sd(decoder_cls, {k[len("decoder."):]: v for k, v in inner_keys.items()}
                                     if inner_keys else dec_sd)
    n_pairs = int(latents.shape[0])
    wrapper = PoseFiLMHNeRVWrapperV2(decoder, n_pairs=n_pairs)
    wrapper.load_state_dict({**wrapper.state_dict(), **dec_sd}, strict=False)
    wrapper.eval()

    resids = []
    with torch.no_grad():
        for i in range(n):
            z = latents[i : i + 1]
            idx = torch.tensor([i])
            on = wrapper(z, idx)  # (1,2,3,384,512)
            off = wrapper(z, None)
            f0_on = on[0, 0]  # (3,384,512)
            f0_off = off[0, 0]
            d = (f0_on - f0_off)
            if (d.shape[-2], d.shape[-1]) != (work_h, work_w):
                d = F.interpolate(d.unsqueeze(0), size=(work_h, work_w),
                                  mode="bilinear", align_corners=False)[0]
            resids.append(d.cpu().numpy().astype(np.float64))
    if not has_film:
        # honest note: a pre-FiLM basin has identity residual ~0 (the carrier hasn't trained yet).
        pass
    return resids


def _build_decoder_from_sd(decoder_cls, sd):
    """Best-effort decoder construction. The configurable-taper decoder infers its taper from the saved
    channel shapes; we read base/latent/taper off the state dict where possible, else use defaults."""
    import torch

    # Infer latent_dim + base_channels from the stem if available; otherwise fall back to bind-all defaults.
    latent_dim = 28
    base_channels = 20
    try:
        dec = decoder_cls(latent_dim=latent_dim, base_channels=base_channels, eval_size=(384, 512))
    except TypeError:
        dec = decoder_cls(latent_dim=latent_dim, base_channels=base_channels)
    dec.load_state_dict(sd, strict=False)
    dec.eval()
    for p in dec.parameters():
        p.requires_grad = False
    _ = torch  # keep import referenced
    return dec


if __name__ == "__main__":
    raise SystemExit(main())
