# SPDX-License-Identifier: MIT
"""MED-2 probe — validate Lever-4's INDIRECT byte-win hypothesis against REAL bytes.

The R1 audit (`layer2_levers_independent_audit_20260612T151829Z.md` MEDIUM-2) flagged
that score-aware QAT (``src/tac/torch_vehicle/score_aware_qat.py``) does NOT change the
archive grammar — the vendored codec ALWAYS quantizes every tensor at ``N_QUANT=127``
levels. So Lever-4's byte-win claim is an UNPROVEN second-order ("indirect") effect:
training a low-sensitivity tensor robust at a COARSER grid is hypothesized to make its
127-level codec encoding collapse to more repeated symbols that brotli loves.

This probe measures whether applying the score-aware (per-tensor sensitivity-driven)
grid to the basin weights actually produces a SMALLER brotli decoder-blob than the
uniform-127 grid, through the REAL vendored codec
(``codec.encode_decoder(quantize_state_dict(sd))``). It is the empirical bit-spend
proof per CLAUDE.md Catalog #304.

DESIGN (honest):
  * sensitivity = REAL ``s_t = ||∂S/∂w_t||`` from ONE real score-domain forward+backward
    through the FROZEN contest SegNet/PoseNet on a small set of 0.mkv GT pairs
    (``RealScorerContext`` capped + ``accumulate_tensor_sensitivity``) — NOT a synthetic
    or fabricated sensitivity (Slot EEE forbidden-class-3). The advisory d_seg/d_pose
    are measured on the SAME pair set (labeled advisory; it is NOT a 600-pair contest
    eval).
  * uniform arm  = ``apply_score_aware_qat(decoder, sensitivity=None)`` → bit-identical
    to the vendored uniform-127 QAT (the default-preserving guard).
  * score-aware arm = ``apply_score_aware_qat(decoder, real_sensitivity)`` → per-tensor
    grid: low-sensitivity tensors snapped to a coarser (down to 64-level) grid, high
    ones kept at 127.
  * BOTH arms are then byte-closed through the SAME real codec; we compare the brotli
    decoder-blob bytes AND the full-archive bytes AND the advisory d_seg/d_pose.

VERDICT:
  * SUPPORTED      — score-aware blob is SMALLER than uniform at non-worse advisory
    distortion (the coarse-grid-robust → brotli-friendly hypothesis holds; quantify it).
  * UNSUPPORTED    — score-aware blob is NOT smaller (or distortion is worse): the
    indirect-win hypothesis is not borne out by JUST applying the grid; Lever-4 must be
    HONESTLY-MARKED ("byte-win unvalidated; needs the full A/B / a variable-level codec
    path") so it is not shipped as a confirmed rate lever.

NOTE on what this probe CAN and CANNOT prove: it applies the score-aware grid to the
ALREADY-TRAINED basin weights (a one-shot snap), which is the codec-byte effect of the
grid in isolation. The full Lever-4 mechanism additionally TRAINS the decoder to be
robust at the coarse grid (eval_roundtrip sister discipline) — that training effect can
only be measured by a paired training A/B (uniform-QAT-trained vs score-aware-QAT-trained
arm). This probe is the NECESSARY (codec-side) half: if the grid snap does not shrink the
blob even on a tensor TUNED toward it, the indirect-win hypothesis is on weak ground.

Authority: $0 / local / torch-CPU (TRUSTED). Every number is ``[macOS-CPU advisory]``
NON-PROMOTABLE.

Usage::

    .venv/bin/python experiments/probe_lever4_qat_brotli_blob_delta.py --max-pairs 8
    .venv/bin/python experiments/probe_lever4_qat_brotli_blob_delta.py --max-pairs 8 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.torch_vehicle.score_aware_qat import (
    ScoreAwareQATConfig,
    accumulate_tensor_sensitivity,
    apply_score_aware_qat,
    per_tensor_levels_from_sensitivity,
    restore_score_aware_qat,
)
from tac.torch_vehicle.vendored_imports import import_vendored

_BASIN_CKPT = (
    _REPO_ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
    / "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = _REPO_ROOT / "upstream/videos/0.mkv"
_EVAL_H, _EVAL_W = 384, 512


def _real_sensitivity_and_advisory_distortion(
    decoder: nn.Module, latents: torch.Tensor, max_pairs: int
) -> tuple[dict[str, float], float, float]:
    """ONE real score-domain forward+backward through the frozen contest scorer on
    ``max_pairs`` 0.mkv GT pairs → returns (sensitivity_ema, advisory_d_seg,
    advisory_d_pose). Replicates the vendored ``common.py`` roundtrip exactly."""
    from tac.torch_vehicle.scorer_context import RealScorerContext

    ctx = RealScorerContext(
        str(_VIDEO),
        device="cpu",
        max_pairs=max_pairs,
        targets_cache=str(_REPO_ROOT / ".omx/tmp/lever4_probe_targets"),
    )
    n = ctx.n_pairs
    idx = torch.arange(n)
    for p in decoder.parameters():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad = None

    # Vendored common.py roundtrip: decode -> bicubic up -> bilinear down -> STE round.
    decoded_pair = decoder(latents[idx])  # (n, 2, 3, 384, 512)
    flat = decoded_pair.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
    dc = decoded_bhwc.clamp(0, 255)
    dr = dc.round()
    decoded_bhwc = dc + (dr - dc).detach()

    net = ctx.distortion_net
    posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
    seg_out = net.segnet(segnet_in)
    pose_out = net.posenet(posenet_in)
    pose6 = pose_out["pose"][:, :6]

    # Score-domain loss (the real S terms): seg = soft surrogate of argmax-flip via CE
    # to the hard GT (the gradient that protects the argmax boundary), pose = the
    # contest sqrt(10*MSE). We use the vendored seg loss fn through the ctx targets.
    seg_l = F.cross_entropy(seg_out, ctx.seg_targets_hard[idx])
    pose_mse = F.mse_loss(pose6, ctx.pose_targets[idx])
    pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
    loss = 100.0 * seg_l + pose_l
    loss.backward()

    sens: dict[str, float] = {}
    accumulate_tensor_sensitivity(decoder, sens, decay=0.0)  # decay=0 → exact ||grad||

    # Advisory distortion on this pair set (NOT a contest eval): the contest d_seg is
    # the argmax-flip RATE vs GT; d_pose is MSE on the first 6 pose dims.
    with torch.no_grad():
        d_seg = float(
            (seg_out.argmax(dim=1) != ctx.seg_targets_hard[idx]).float().mean().item()
        )
        d_pose = float(F.mse_loss(pose6, ctx.pose_targets[idx]).item())
    for p in decoder.parameters():
        if p.grad is not None:
            p.grad = None
    return sens, d_seg, d_pose


def _advisory_distortion_for_weights(
    decoder: nn.Module, latents: torch.Tensor, ctx, n: int
) -> tuple[float, float]:
    """Compute advisory d_seg/d_pose for the CURRENT decoder weights on ``n`` pairs
    (no grad)."""
    idx = torch.arange(n)
    with torch.no_grad():
        decoded_pair = decoder(latents[idx])
        flat = decoded_pair.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
        decoded_bhwc = down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
        decoded_bhwc = decoded_bhwc.clamp(0, 255).round()
        net = ctx.distortion_net
        posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
        seg_out = net.segnet(segnet_in)
        pose6 = net.posenet(posenet_in)["pose"][:, :6]
        d_seg = float(
            (seg_out.argmax(dim=1) != ctx.seg_targets_hard[idx]).float().mean().item()
        )
        d_pose = float(F.mse_loss(pose6, ctx.pose_targets[idx]).item())
    return d_seg, d_pose


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=8, help="GT pairs for the real sensitivity backward")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not _BASIN_CKPT.exists():
        raise FileNotFoundError(f"basin checkpoint not found: {_BASIN_CKPT}")
    if not _VIDEO.exists():
        raise FileNotFoundError(f"0.mkv not found: {_VIDEO}")

    codec = import_vendored("codec")
    model = import_vendored("model")

    ck = torch.load(_BASIN_CKPT, map_location="cpu", weights_only=False)
    basin_sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}
    basin_latents = ck["ema_latents"].detach().float()

    # --- REAL sensitivity from one frozen-scorer backward ---------------------
    dec = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec.load_state_dict(basin_sd)
    sens, base_d_seg, base_d_pose = _real_sensitivity_and_advisory_distortion(
        dec, basin_latents, args.max_pairs
    )
    # The codec names tensors by state_dict key ("blocks.0.weight"); the sensitivity
    # EMA keys are MODULE names ("blocks.0"). Map module-name sensitivity -> the codec's
    # weight-tensor name so per_tensor_levels can be inspected.
    names_modules = [
        name
        for name, mod in dec.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and getattr(mod, "weight", None) is not None
    ]
    levels = per_tensor_levels_from_sensitivity(sens, names_modules, ScoreAwareQATConfig())
    n_coarsened = sum(1 for v in levels.values() if v < 127)

    # --- UNIFORM-127 arm (default-preserving) ---------------------------------
    dec_u = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec_u.load_state_dict(basin_sd)
    orig_u = apply_score_aware_qat(dec_u, sensitivity=None)  # == vendored uniform-127
    uniform_sd = {k: v.detach().clone() for k, v in dec_u.state_dict().items()}
    restore_score_aware_qat(dec_u, orig_u)

    # --- SCORE-AWARE arm ------------------------------------------------------
    dec_s = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec_s.load_state_dict(basin_sd)
    # Apply the score-aware grid IN PLACE (no restore — dec_s now holds the quantized
    # weights, which is exactly what we byte-close + advisory-eval below).
    apply_score_aware_qat(dec_s, sensitivity=sens)
    score_aware_sd = {k: v.detach().clone() for k, v in dec_s.state_dict().items()}

    # --- REAL codec byte-close BOTH arms --------------------------------------
    def _dec_blob(sd):
        return len(codec.encode_decoder(codec.quantize_state_dict(sd)))

    def _archive(sd):
        meta = {"base_channels": 20, "latent_dim": 28}
        return len(codec.build_archive(sd, basin_latents, meta))

    uniform_blob = _dec_blob(uniform_sd)
    score_blob = _dec_blob(score_aware_sd)
    uniform_archive = _archive(uniform_sd)
    score_archive = _archive(score_aware_sd)

    # --- advisory distortion for each arm (same pair set) ---------------------
    from tac.torch_vehicle.scorer_context import RealScorerContext

    ctx = RealScorerContext(
        str(_VIDEO),
        device="cpu",
        max_pairs=args.max_pairs,
        targets_cache=str(_REPO_ROOT / ".omx/tmp/lever4_probe_targets"),
    )
    n = ctx.n_pairs
    dec_u2 = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec_u2.load_state_dict(uniform_sd)
    u_d_seg, u_d_pose = _advisory_distortion_for_weights(dec_u2, basin_latents, ctx, n)
    dec_s2 = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec_s2.load_state_dict(score_aware_sd)
    s_d_seg, s_d_pose = _advisory_distortion_for_weights(dec_s2, basin_latents, ctx, n)

    blob_delta = score_blob - uniform_blob  # negative = score-aware is SMALLER (win)
    archive_delta = score_archive - uniform_archive

    # Verdict: SUPPORTED iff score-aware blob is strictly smaller AND distortion is not
    # materially worse (advisory d_seg within a small tolerance — this is a one-shot snap
    # on already-trained weights, so a tiny d_seg uptick is expected; the real fix is
    # training-time robustness, measured by a separate training A/B).
    SEG_TOL = 0.02  # advisory argmax-flip-rate tolerance on the small pair set
    seg_ok = (s_d_seg - u_d_seg) <= SEG_TOL
    if blob_delta < 0 and seg_ok:
        verdict = "SUPPORTED"
        verdict_note = (
            f"score-aware decoder blob is {-blob_delta} B SMALLER than uniform-127 "
            f"({score_blob} vs {uniform_blob}) at advisory d_seg within {SEG_TOL} "
            f"(uniform {u_d_seg:.4f} → score-aware {s_d_seg:.4f}). The coarse-grid-robust "
            "→ brotli-friendly hypothesis is SUPPORTED on the basin weights; quantify the "
            "win in the A/B. Catalog #304 empirical bit-spend proof: POSITIVE."
        )
    else:
        reason = []
        if blob_delta >= 0:
            reason.append(
                f"score-aware blob is NOT smaller ({score_blob} vs uniform {uniform_blob}; "
                f"delta {blob_delta:+d} B)"
            )
        if not seg_ok:
            reason.append(
                f"advisory d_seg WORSE by {s_d_seg - u_d_seg:+.4f} (> {SEG_TOL})"
            )
        verdict = "UNSUPPORTED"
        verdict_note = (
            "the indirect-win hypothesis is NOT borne out by applying the grid to the "
            f"basin weights: {'; '.join(reason)}. HONESTLY-MARK Lever-4: byte-win "
            "unvalidated by the grid snap; it needs the full training A/B (uniform-QAT-"
            "trained vs score-aware-QAT-trained) OR a variable-level codec path that lets "
            "the per-tensor grid actually change archive bytes. Do NOT ship Lever-4 as a "
            "confirmed rate lever on this basis."
        )

    result = {
        "probe": "lever4_qat_brotli_blob_delta",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE (Catalog #304 bit-spend proof)",
        "basin_checkpoint": str(_BASIN_CKPT.relative_to(_REPO_ROOT)),
        "max_pairs": n,
        "n_tensors": len(levels),
        "n_coarsened_tensors": n_coarsened,
        "per_tensor_levels": {k: levels[k] for k in sorted(levels)},
        "uniform_decoder_blob_bytes": uniform_blob,
        "score_aware_decoder_blob_bytes": score_blob,
        "decoder_blob_delta_bytes": blob_delta,
        "uniform_full_archive_bytes": uniform_archive,
        "score_aware_full_archive_bytes": score_archive,
        "full_archive_delta_bytes": archive_delta,
        "advisory_d_seg_uniform": round(u_d_seg, 6),
        "advisory_d_seg_score_aware": round(s_d_seg, 6),
        "advisory_d_pose_uniform": round(u_d_pose, 8),
        "advisory_d_pose_score_aware": round(s_d_pose, 8),
        "base_d_seg_full_precision": round(base_d_seg, 6),
        "base_d_pose_full_precision": round(base_d_pose, 8),
        "verdict": verdict,
        "verdict_note": verdict_note,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 78)
        print("MED-2 PROBE — Lever-4 score-aware QAT vs uniform-127, REAL brotli blob delta")
        print("=" * 78)
        print(f"basin: {result['basin_checkpoint']}")
        print(f"authority: {result['authority']}")
        print(f"real sensitivity from {n} frozen-scorer GT pairs; "
              f"{n_coarsened}/{len(levels)} tensors coarsened below 127 levels")
        print()
        print(f"  uniform-127 decoder blob : {uniform_blob:>8d} B   full archive {uniform_archive:>8d} B")
        print(f"  score-aware decoder blob : {score_blob:>8d} B   full archive {score_archive:>8d} B")
        print(f"  decoder-blob delta       : {blob_delta:>+8d} B   (negative = score-aware SMALLER)")
        print(f"  full-archive delta       : {archive_delta:>+8d} B")
        print()
        print(f"  advisory d_seg  uniform {u_d_seg:.4f} -> score-aware {s_d_seg:.4f}  (Δ {s_d_seg-u_d_seg:+.4f})")
        print(f"  advisory d_pose uniform {u_d_pose:.6f} -> score-aware {s_d_pose:.6f}")
        print(f"  (full-precision basin advisory: d_seg {base_d_seg:.4f}, d_pose {base_d_pose:.6f})")
        print()
        print(f"VERDICT: {verdict}")
        print(f"  {verdict_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
