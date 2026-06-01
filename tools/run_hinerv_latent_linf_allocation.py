#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 HiNeRV inverse-steganalysis carrier runner: L-inf latent allocation.

The full $0 macOS-CPU stack on the HiNeRV carrier (design memo Phase-2
candidate; cheapest-measured-RD NeRV-family carrier, arXiv 2306.09818):

  1. TRAIN a light HiNeRV carrier on REAL ``upstream/videos/0.mkv`` frames at an
     AGGRESSIVELY-SMALL byte budget by construction (tiny latent dims + a small
     decoder; the NeRV family is parameterized by a target byte budget — the
     carrier is cheap BY CONSTRUCTION, the property HPRC/wavelet lack). NeRV-style
     per-pixel MSE fit (no scorer load in the training inner loop; the oracle is
     applied POST-training in the allocation step — the receiver never loads the
     scorer).
  2. PROVE the dense decoder-VJP adjoint EXACT (G3): adjoint dot-product test
     ``<J x, y> == <x, J^T y>`` (~1e-6) + central finite-difference JVP corroboration.
  3. Push the score-exact oracle ``rho_i`` (s_seg P18 + s_pose P19, combined at the
     contest score-derivative weights) into the HiNeRV LATENT domain via the
     Fisher-pullback ``s_latent_k = sum_i (dframe_i/dz_k)^2 s_pixel_i``.
  4. ALLOCATE the carrier's latent bit budget by L-inf margin-budget (cost=rho_i)
     vs the L2 uniform baseline at EQUAL latent rate (the §7-proven objective,
     now in the carrier's coefficient domain).
  5. QUANTIZE latents at the allocated steps, BYTE-CLOSE an HIV1 archive, and
     ADVISORY re-measure ``d_seg``/``d_pose``/rate on the verified bit-exact CPU
     mirror. Compare L-inf vs L2 vs the un-quantized carrier.

CONTEST COMPLIANCE / authority
------------------------------
$0 macOS-CPU ONLY. The scorer is loaded for OFFLINE allocation analysis +
advisory re-measurement; it NEVER enters the inflate runtime. ALL numerics are
``[macOS-CPU advisory]`` — NON-PROMOTABLE (``score_claim=false``,
``promotable=false``) per Catalog #341/#192/#127/#323. No score claim; paired
CPU+CUDA (Catalog #246) reserved for operator authorization. The training device
may be MPS/CPU (research-signal); ALL d_seg/d_pose MEASUREMENT runs on the
bit-exact CPU mirror.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.hinerv_latent_linf_allocation import (  # noqa: E402
    NON_PROMOTABLE_MARKERS,
    adjoint_dotproduct_residual,
    allocate_l2_uniform_latent_steps,
    allocate_linf_latent_steps,
    finite_difference_vjp_residual,
    push_pixel_saliency_to_latent,
    quantize_latents_with_steps,
    scale_jvp_norm,
)
from tac.analysis.inverse_steganalysis_linf_vs_l2_gate import (  # noqa: E402
    measure_pair_d_seg_d_pose,
)
from tac.analysis.score_exact_saliency import (  # noqa: E402
    compute_s_pose_fisher,
    compute_s_seg_flip_risk,
    decode_real_pairs,
    load_score_exact_scorers,
)
from tac.substrates.hi_nerv.architecture import (  # noqa: E402
    HinervConfig,
    HinervSubstrate,
)
from tac.substrates.hi_nerv.archive import pack_archive  # noqa: E402

# Contest constants (verified bit-exact mirror).
CONTEST_N = 37_545_489.0
SEG_W = 100.0
POSE_SQRT_W = float(np.sqrt(10.0))


def _contest_distortion(d_seg: float, d_pose: float) -> float:
    """``100·d_seg + sqrt(10·d_pose)`` — the contest distortion (rate excluded)."""
    return SEG_W * d_seg + POSE_SQRT_W * float(np.sqrt(max(d_pose, 0.0)))


@dataclass(frozen=True)
class CarrierConfig:
    """An aggressively-small HiNeRV carrier (super-small-rate-by-design)."""

    latent_dim_coarse: int
    latent_dim_mid: int
    latent_dim_fine: int
    embed_dim: int
    decoder_channels: tuple[int, ...]
    sin_frequency: float
    num_pairs: int
    output_height: int
    output_width: int

    def to_hinerv(self) -> HinervConfig:
        n_blocks = len(self.decoder_channels) - 1
        return HinervConfig(
            latent_dim_coarse=self.latent_dim_coarse,
            latent_dim_mid=self.latent_dim_mid,
            latent_dim_fine=self.latent_dim_fine,
            embed_dim=self.embed_dim,
            initial_grid_h=3,
            initial_grid_w=4,
            decoder_channels=self.decoder_channels,
            sin_frequency=self.sin_frequency,
            num_upsample_blocks=n_blocks,
            mid_injection_block_index=max(0, n_blocks // 3),
            fine_injection_block_index=max(1, (2 * n_blocks) // 3),
            num_pairs=self.num_pairs,
            output_height=self.output_height,
            output_width=self.output_width,
        )


def _decoder_state_dict(model: HinervSubstrate) -> dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {
        k: v
        for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }


def _carrier_archive_bytes(
    model: HinervSubstrate,
    cfg: HinervConfig,
    *,
    lc: torch.Tensor,
    lm: torch.Tensor,
    lf: torch.Tensor,
) -> bytes:
    """Byte-close the HIV1 archive for the given (possibly quantized) latents."""
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "mid_injection_block_index": cfg.mid_injection_block_index,
        "fine_injection_block_index": cfg.fine_injection_block_index,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }
    return pack_archive(_decoder_state_dict(model), lc, lm, lf, meta)


def train_light_carrier(
    pairs_native: torch.Tensor,
    cfg: HinervConfig,
    *,
    epochs: int,
    lr: float,
    device: str,
    seed: int,
) -> HinervSubstrate:
    """NeRV-style per-pixel-MSE fit of a tiny HiNeRV carrier on REAL frames.

    The carrier is trained to memorize the (downscaled-to-carrier-resolution)
    real pairs. No scorer load in the inner loop (the receiver never loads the
    scorer; the oracle is applied POST-training). The carrier resolution is the
    config output_h/w; the real native frames are bilinearly resized to it for
    the fit target (the carrier is deliberately small/cheap).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    model = HinervSubstrate(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Build per-pair fit targets at carrier resolution.
    n = pairs_native.shape[0]
    tgt = torch.nn.functional.interpolate(
        pairs_native.reshape(n * 2, 3, *pairs_native.shape[-2:]).to(dev) / 255.0,
        size=(cfg.output_height, cfg.output_width),
        mode="bilinear",
        align_corners=False,
    ).reshape(n, 2, 3, cfg.output_height, cfg.output_width)

    idx = torch.arange(n, device=dev, dtype=torch.long)
    for _ in range(int(epochs)):
        rgb_0, rgb_1 = model(idx)
        loss = (rgb_0 - tgt[:, 0]).pow(2).mean() + (rgb_1 - tgt[:, 1]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def combined_pixel_rho(
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    gt_pair_native: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """Combine s_seg (P18) + s_pose (P19) at score-derivative weights -> (H, W).

    Both oracle surfaces are produced at their scorer-input resolution; we resize
    them to the CARRIER resolution (out_h, out_w) so the Fisher-pullback can push
    them through the carrier decoder's latent Jacobian. The score-derivative
    weights are the contest marginals: d_seg enters S linearly at weight SEG_W,
    d_pose enters via sqrt at POSE_SQRT_W — at the frontier operating point pose
    is the dominant marginal (per the §7 GREEN result + CLAUDE.md operating-point
    analysis), so we weight s_pose by POSE_SQRT_W and s_seg by SEG_W in the
    combined saliency the allocator aims by.
    """
    seg = compute_s_seg_flip_risk(segnet, gt_pair_native)
    pose = compute_s_pose_fisher(posenet, gt_pair_native, method="batched_vjp")
    s_seg = seg.flip_risk  # (H_seg, W_seg) at scorer-input resolution
    s_pose = pose.s_pose  # (H_pose, W_pose)

    def _to_carrier(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x.reshape(1, 1, *x.shape), size=(out_h, out_w), mode="bilinear",
            align_corners=False,
        ).reshape(out_h, out_w)

    s_seg_c = _to_carrier(s_seg)
    s_pose_c = _to_carrier(s_pose)
    # Normalize each surface to unit mean so the two weights are comparable, then
    # combine at the contest score-derivative weights.
    s_seg_c = s_seg_c / (s_seg_c.mean() + 1e-12)
    s_pose_c = s_pose_c / (s_pose_c.mean() + 1e-12)
    return (SEG_W * s_seg_c + POSE_SQRT_W * s_pose_c).clamp(min=0.0)


def _render_carrier_pair_native(
    model: HinervSubstrate,
    pair_index: int,
    *,
    native_h: int,
    native_w: int,
    device: str,
) -> torch.Tensor:
    """Render the carrier pair and upscale to native (1, 2, 3, H, W) in [0,255]."""
    dev = torch.device(device)
    idx = torch.tensor([pair_index], dtype=torch.long, device=dev)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    pair = torch.stack([rgb_0[0], rgb_1[0]])  # (2, 3, h, w) in [0,1]
    pair = torch.nn.functional.interpolate(
        pair, size=(native_h, native_w), mode="bilinear", align_corners=False
    )
    return (pair.clamp(0, 1) * 255.0).unsqueeze(0)  # (1, 2, 3, H, W)


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "experiments" / "results" / "hinerv_invsteg_carrier_advisory"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Decode real pairs (native) for the fit + oracle + measurement.
    pairs_native = decode_real_pairs(
        REPO_ROOT / "upstream" / "videos" / "0.mkv",
        args.num_pairs,
        pair_stride=args.pair_stride,
        device="cpu",
    )  # (N, 2, 3, Hn, Wn)
    native_h, native_w = int(pairs_native.shape[-2]), int(pairs_native.shape[-1])

    # 2. Aggressively-small carrier (super-small-rate-by-design).
    carrier = CarrierConfig(
        latent_dim_coarse=args.latent_dim_coarse,
        latent_dim_mid=args.latent_dim_mid,
        latent_dim_fine=args.latent_dim_fine,
        embed_dim=args.embed_dim,
        decoder_channels=tuple(int(c) for c in args.decoder_channels.split(",")),
        sin_frequency=args.sin_frequency,
        num_pairs=args.num_pairs,
        output_height=args.carrier_h,
        output_width=args.carrier_w,
    )
    cfg = carrier.to_hinerv()
    model = train_light_carrier(
        pairs_native, cfg,
        epochs=args.epochs, lr=args.lr, device=args.train_device, seed=args.seed,
    )
    model = model.to("cpu").eval()

    # 3. PROVE the decoder-VJP adjoint exact (G3) on the trained carrier.
    # The adjoint dot-product test is the PRIMARY machine-exact proof (robust to
    # near-null Jacobian columns). The fd-sweep is a corroborating numerical-
    # Jacobian convergence check; it is only MEANINGFUL on scales whose Jacobian
    # column is non-degenerate (a trained carrier can drive a whole latent scale's
    # Jacobian to ~0, where the fd ratio is noise/noise = ~1.0 even though the
    # analytic and numerical JVPs both vanish and DO agree). We therefore probe the
    # per-scale JVP norm and only require fd-convergence on non-degenerate scales.
    adjoint_residuals = {
        scale: adjoint_dotproduct_residual(model, 0, scale=scale, seed=3)
        for scale in ("coarse", "mid", "fine")
    }
    fd_residuals = {
        scale: finite_difference_vjp_residual(model, 0, scale=scale, seed=5)
        for scale in ("coarse", "mid", "fine")
    }
    jvp_norms = {
        scale: scale_jvp_norm(model, 0, scale=scale, seed=5)
        for scale in ("coarse", "mid", "fine")
    }
    # Non-degenerate = the scale's Jacobian has magnitude above the fp32 noise
    # floor (the carrier actually uses that latent scale at the operating point;
    # below ~1e-6 RMS the column is numerically null and the fd ratio is
    # noise/noise). The PRIMARY proof is always the adjoint dot-product test
    # (machine-exact regardless of column magnitude).
    nondegenerate = {s for s, n in jvp_norms.items() if n > 1e-6}
    adjoint_exact = all(r < 1e-4 for r in adjoint_residuals.values())
    fd_exact_nondegenerate = all(
        fd_residuals[s] < 1e-2 for s in nondegenerate
    ) if nondegenerate else False

    # 4. Load the bit-exact CPU mirror for the oracle + advisory measurement.
    posenet, segnet = load_score_exact_scorers(REPO_ROOT / "upstream", device="cpu")

    # 5. The baseline (un-quantized carrier) reference latents.
    lc0 = model.latents_coarse.detach().cpu()
    lm0 = model.latents_mid.detach().cpu()
    lf0 = model.latents_fine.detach().cpu()
    baseline_bytes = _carrier_archive_bytes(model, cfg, lc=lc0, lm=lm0, lf=lf0)

    # 6. Per-pair: push oracle -> latent, allocate L-inf vs L2 at equal rate,
    #    quantize, then re-measure on the mirror. We allocate PER-PAIR latents.
    per_pair: list[dict[str, object]] = []
    lc_linf = lc0.clone()
    lm_linf = lm0.clone()
    lf_linf = lf0.clone()
    lc_l2 = lc0.clone()
    lm_l2 = lm0.clone()
    lf_l2 = lf0.clone()

    target_bits = float(args.latent_target_bits)
    for p in range(args.num_pairs):
        gt_native = pairs_native[p : p + 1]  # (1, 2, 3, Hn, Wn)
        rho_pixel = combined_pixel_rho(
            posenet, segnet, gt_native, out_h=cfg.output_height, out_w=cfg.output_width
        )
        ls = push_pixel_saliency_to_latent(model, p, rho_pixel, frame_slot=1)
        lat_vals = {
            "coarse": model.latents_coarse[p].detach().cpu().numpy(),
            "mid": model.latents_mid[p].detach().cpu().numpy(),
            "fine": model.latents_fine[p].detach().cpu().numpy(),
        }
        l2 = allocate_l2_uniform_latent_steps(lat_vals, target_bits=target_bits)
        linf = allocate_linf_latent_steps(ls.s_latent, lat_vals, target_bits=target_bits)
        q_linf = quantize_latents_with_steps(lat_vals, linf.steps)
        q_l2 = quantize_latents_with_steps(lat_vals, l2.steps)

        for bank, scale, src in (
            (lc_linf, "coarse", q_linf), (lm_linf, "mid", q_linf), (lf_linf, "fine", q_linf),
        ):
            bank[p] = torch.from_numpy(src[scale]).to(bank.dtype)
        for bank, scale, src in (
            (lc_l2, "coarse", q_l2), (lm_l2, "mid", q_l2), (lf_l2, "fine", q_l2),
        ):
            bank[p] = torch.from_numpy(src[scale]).to(bank.dtype)

        # Re-measure d_seg/d_pose on the mirror for each allocation vs the carrier
        # baseline (un-quantized). All at NATIVE resolution (the scorer's input).
        # Save the untouched baseline latents to restore between measurements.
        base_lc, base_lm, base_lf = (
            model.latents_coarse[p].clone(),
            model.latents_mid[p].clone(),
            model.latents_fine[p].clone(),
        )
        # baseline (un-quantized carrier)
        with torch.no_grad():
            model.latents_coarse[p] = base_lc
            model.latents_mid[p] = base_lm
            model.latents_fine[p] = base_lf
        cand_b = _render_carrier_pair_native(
            model, p, native_h=native_h, native_w=native_w, device="cpu"
        )
        d_seg_b, d_pose_b = measure_pair_d_seg_d_pose(posenet, segnet, gt_native, cand_b)

        # L-inf
        with torch.no_grad():
            model.latents_coarse[p] = lc_linf[p]
            model.latents_mid[p] = lm_linf[p]
            model.latents_fine[p] = lf_linf[p]
        cand_linf = _render_carrier_pair_native(
            model, p, native_h=native_h, native_w=native_w, device="cpu"
        )
        d_seg_linf, d_pose_linf = measure_pair_d_seg_d_pose(
            posenet, segnet, gt_native, cand_linf
        )

        # L2
        with torch.no_grad():
            model.latents_coarse[p] = lc_l2[p]
            model.latents_mid[p] = lm_l2[p]
            model.latents_fine[p] = lf_l2[p]
        cand_l2 = _render_carrier_pair_native(
            model, p, native_h=native_h, native_w=native_w, device="cpu"
        )
        d_seg_l2, d_pose_l2 = measure_pair_d_seg_d_pose(
            posenet, segnet, gt_native, cand_l2
        )

        # Restore baseline latents in the model for the next pair's oracle pass.
        with torch.no_grad():
            model.latents_coarse[p] = base_lc
            model.latents_mid[p] = base_lm
            model.latents_fine[p] = base_lf

        per_pair.append({
            "pair_index": int(args.pair_stride * p),
            "latent_rate_l2_bits": float(l2.total_bits),
            "latent_rate_linf_bits": float(linf.total_bits),
            "baseline": {"d_seg": d_seg_b, "d_pose": d_pose_b,
                         "contest_distortion": _contest_distortion(d_seg_b, d_pose_b)},
            "l2_uniform": {"d_seg": d_seg_l2, "d_pose": d_pose_l2,
                           "contest_distortion": _contest_distortion(d_seg_l2, d_pose_l2)},
            "linf_margin_budget": {"d_seg": d_seg_linf, "d_pose": d_pose_linf,
                                   "contest_distortion": _contest_distortion(d_seg_linf, d_pose_linf)},
        })

    # 7. Byte-close the two allocation archives (the carrier IS cheap-by-design).
    linf_bytes = _carrier_archive_bytes(model, cfg, lc=lc_linf, lm=lm_linf, lf=lf_linf)
    l2_bytes = _carrier_archive_bytes(model, cfg, lc=lc_l2, lm=lm_l2, lf=lf_l2)

    # Aggregate advisory contest distortion (mean over pairs).
    def _mean(key: str, sub: str) -> float:
        return float(np.mean([row[sub][key] for row in per_pair]))  # type: ignore[index]

    agg = {
        "baseline": {
            "d_seg": _mean("d_seg", "baseline"),
            "d_pose": _mean("d_pose", "baseline"),
            "contest_distortion": _mean("contest_distortion", "baseline"),
            "archive_bytes": len(baseline_bytes),
            "rate_term": 25.0 * len(baseline_bytes) / CONTEST_N,
        },
        "l2_uniform": {
            "d_seg": _mean("d_seg", "l2_uniform"),
            "d_pose": _mean("d_pose", "l2_uniform"),
            "contest_distortion": _mean("contest_distortion", "l2_uniform"),
            "archive_bytes": len(l2_bytes),
            "rate_term": 25.0 * len(l2_bytes) / CONTEST_N,
        },
        "linf_margin_budget": {
            "d_seg": _mean("d_seg", "linf_margin_budget"),
            "d_pose": _mean("d_pose", "linf_margin_budget"),
            "contest_distortion": _mean("contest_distortion", "linf_margin_budget"),
            "archive_bytes": len(linf_bytes),
            "rate_term": 25.0 * len(linf_bytes) / CONTEST_N,
        },
    }
    for k in ("baseline", "l2_uniform", "linf_margin_budget"):
        agg[k]["advisory_total_S"] = (
            agg[k]["contest_distortion"] + agg[k]["rate_term"]
        )

    linf_vs_l2_distortion_delta = (
        agg["linf_margin_budget"]["contest_distortion"]
        - agg["l2_uniform"]["contest_distortion"]
    )
    linf_beats_l2 = linf_vs_l2_distortion_delta < 0.0

    result = {
        "schema": "hinerv_latent_linf_allocation_runner.v1",
        **NON_PROMOTABLE_MARKERS,
        "carrier": asdict(carrier),
        "carrier_params": int(model.num_parameters()),
        "carrier_decoder_params": int(
            sum(p.numel() for n, p in model.named_parameters()
                if n not in ("latents_coarse", "latents_mid", "latents_fine"))
        ),
        "num_pairs": int(args.num_pairs),
        "pair_stride": int(args.pair_stride),
        "native_hw": [native_h, native_w],
        "latent_target_bits_per_pair": float(target_bits),
        "g3_adjoint_dotproduct_residual": adjoint_residuals,
        "g3_adjoint_exact": bool(adjoint_exact),
        "g3_finite_difference_jvp_residual": fd_residuals,
        "g3_finite_difference_eps_sweep": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
        "g3_per_scale_jvp_norm": {k: float(v) for k, v in jvp_norms.items()},
        "g3_nondegenerate_scales": sorted(nondegenerate),
        "g3_finite_difference_exact_nondegenerate": bool(fd_exact_nondegenerate),
        "per_pair": per_pair,
        "aggregate": agg,
        "linf_vs_l2_contest_distortion_delta": linf_vs_l2_distortion_delta,
        "linf_beats_l2": bool(linf_beats_l2),
        "frontier_reference": {
            "note": "pointer-only per Catalog #343",
            "pointer": ".omx/state/canonical_frontier_pointer.json",
        },
        "wall_clock_s": round(time.time() - t0, 2),
    }

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    anchor = out_dir / f"hinerv_latent_linf_allocation_{stamp}.json"
    anchor.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "anchor": str(anchor),
        "carrier_archive_bytes_baseline": len(baseline_bytes),
        "carrier_archive_bytes_linf": len(linf_bytes),
        "carrier_archive_bytes_l2": len(l2_bytes),
        "g3_adjoint_exact": bool(adjoint_exact),
        "g3_fd_exact_nondegenerate": bool(fd_exact_nondegenerate),
        "g3_nondegenerate_scales": sorted(nondegenerate),
        "linf_beats_l2": bool(linf_beats_l2),
        "linf_vs_l2_distortion_delta": round(linf_vs_l2_distortion_delta, 6),
        "advisory_total_S": {k: round(agg[k]["advisory_total_S"], 6)
                             for k in ("baseline", "l2_uniform", "linf_margin_budget")},
    }, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-pairs", type=int, default=6)
    p.add_argument("--pair-stride", type=int, default=100)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-device", type=str, default="cpu",
                   help="cpu or mps (research-signal); measurement always CPU mirror")
    p.add_argument("--latent-dim-coarse", type=int, default=8)
    p.add_argument("--latent-dim-mid", type=int, default=10)
    p.add_argument("--latent-dim-fine", type=int, default=12)
    p.add_argument("--embed-dim", type=int, default=48)
    p.add_argument("--decoder-channels", type=str, default="40,32,24,16,12")
    p.add_argument("--sin-frequency", type=float, default=30.0)
    p.add_argument("--carrier-h", type=int, default=96)
    p.add_argument("--carrier-w", type=int, default=128)
    p.add_argument("--latent-target-bits", type=float, default=120.0,
                   help="per-pair latent bit budget (super-small-rate-by-design)")
    p.add_argument("--output-dir", type=str, default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
