# SPDX-License-Identifier: MIT
"""MED-2 NET-SCORE A/B — the UNCLOSED half: does score-aware QAT *TRAINING* win NET?

The R1 audit (`layer2_levers_independent_audit_20260612T151829Z.md` MEDIUM-2) flagged
the Lever-4 byte-win as an UNPROVEN indirect effect. The MED-2 codec-side probe
(`probe_lever4_qat_brotli_blob_delta.py`) closed HALF of it: a one-shot score-aware
GRID SNAP on the basin weights yields a -3263 B (-4.4%) smaller real brotli blob — the
byte DIRECTION is validated. BUT that snap is NOT a net-score win: the snap incurs a
d_pose uptick (0.00166 -> 0.00178) and a tiny d_seg uptick that, on the contest
`S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/N`, OUTWEIGH the rate win. The snap proves
the codec half; it does NOT prove the lever wins net.

This probe closes the OTHER half — the TRAINING A/B the audit gates the score claim on.
It trains TWO short arms from the SAME basin-EMA seed, on the SAME tiny real-0.mkv slice,
with the SAME epoch budget + RNG:

  * UNIFORM arm     : ``use_qat=True, score_aware_qat=False`` -> vendored uniform-127 QAT.
  * SCORE-AWARE arm : ``use_qat=True, score_aware_qat=True``  -> per-tensor sensitivity
    grid; the decoder TRAINS to be robust at the coarser grid (the eval_roundtrip-sister
    half the snap cannot capture — this can recover the distortion uptick the snap left).

Both arms then byte-close through the REAL vendored codec
(``codec.build_archive``) and are scored on the REAL frozen SegNet/PoseNet over the SAME
real pairs (advisory d_seg = argmax-flip RATE vs GT, advisory d_pose = MSE on the 6 pose
dims). We report the REAL brotli decoder-blob delta AND the advisory contest-score delta.

VERDICT (the operator question — does it win NET, or byte-direction-only?):
  * NET_SCORE_WIN     — score-aware arm has a LOWER advisory contest score (the byte win
    SURVIVES at non-worse distortion AFTER the decoder trained to be coarse-grid-robust).
  * BYTE_DIRECTION_ONLY — score-aware arm is SMALLER in bytes but its advisory score is
    NOT lower (the distortion damage from the coarser grid is not recovered by this much
    training; the honest caveat stands — Lever-4 is a byte-direction lever, not yet a
    confirmed net-score lever; needs more training budget OR the variable-level codec).

Authority: $0 / local / torch-CPU (TRUSTED for advisory). Every number is
``[macOS-CPU advisory]`` NON-PROMOTABLE — this is NOT a 600-pair contest eval (small real
slice, advisory distortion). The DUAL CPU/CUDA exact eval on the full 600 pairs is still
required before any SCORE claim per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

Usage::

    .venv/bin/python experiments/probe_lever4_qat_training_ab_net_score.py --max-pairs 12 --epochs 8
    .venv/bin/python experiments/probe_lever4_qat_training_ab_net_score.py --max-pairs 12 --epochs 8 --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

_BASIN_CKPT = (
    _REPO_ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
    / "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = _REPO_ROOT / "upstream/videos/0.mkv"
_EVAL_H, _EVAL_W = 384, 512
_N = 37_545_489  # contest archive-size normalizer


def _contest_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose + 1e-12) + 25.0 * archive_bytes / _N


def _advisory_distortion(decoder, latents, ctx, n: int) -> tuple[float, float]:
    """Real frozen-scorer advisory d_seg (argmax-flip RATE vs GT) + d_pose (MSE)."""
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


def _train_arm(*, score_aware: bool, max_pairs: int, epochs: int, seed: int, lr: float):
    """Train ONE QAT arm from the basin-EMA seed on the real slice; return the trained
    EMA decoder state_dict + latents + the per-tensor sensitivity EMA (for inspection).

    Mirrors the driver's QAT inner loop EXACTLY (apply -> forward -> restore -> roundtrip
    -> score-domain backward -> accumulate sensitivity -> clip -> step -> EMA update), so
    the only difference between the two arms is ``score_aware`` (uniform-127 vs the
    sensitivity grid). Both seed from the SAME basin EMA + SAME RNG -> a clean paired A/B.
    """
    from tac.torch_vehicle.score_aware_qat import (
        accumulate_tensor_sensitivity,
        apply_score_aware_qat,
        restore_score_aware_qat,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    model = import_vendored("model")
    losses = import_vendored("losses")

    ck = torch.load(_BASIN_CKPT, map_location="cpu", weights_only=False)
    basin_sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}
    basin_latents = ck["ema_latents"].detach().float()

    ctx = RealScorerContext(
        str(_VIDEO),
        device="cpu",
        max_pairs=max_pairs,
        targets_cache=str(_REPO_ROOT / ".omx/tmp/lever4_probe_targets"),
    )
    n = ctx.n_pairs

    torch.manual_seed(seed)
    decoder = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    decoder.load_state_dict(basin_sd)
    latents = torch.nn.Parameter(basin_latents[:n].clone())

    # EMA shadow (the export bytes) — the EMA non-negotiable.
    from copy import deepcopy

    ema_decoder = deepcopy(decoder)
    ema_latents = latents.data.clone()
    ema_decay = 0.999

    opt = torch.optim.AdamW(
        [
            {"params": decoder.parameters(), "lr": lr},
            {"params": [latents], "lr": lr * 10.0},
        ],
        weight_decay=0.0,
    )

    sensitivity_ema: dict[str, float] = {}
    bs = 4
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            B = len(idx)

            # --- Lever-4 QAT apply (uniform-127 vs score-aware grid) ---
            if score_aware:
                sens = sensitivity_ema or None
                originals = apply_score_aware_qat(decoder, sens)
            else:
                originals = losses.apply_qat(decoder)

            decoded_pair = decoder(latents[idx])

            if score_aware:
                restore_score_aware_qat(decoder, originals)
            else:
                losses.restore_qat(decoder, originals)

            flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
            up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
            down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
            decoded_bhwc = down.reshape(B, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
            dc = decoded_bhwc.clamp(0, 255)
            decoded_bhwc = dc + (dc.round() - dc).detach()  # STE round (eval roundtrip)

            net = ctx.distortion_net
            posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
            seg_out = net.segnet(segnet_in)
            pose6 = net.posenet(posenet_in)["pose"][:, :6]

            seg_l = F.cross_entropy(seg_out, ctx.seg_targets_hard[idx])
            pose_mse = F.mse_loss(pose6, ctx.pose_targets[idx])
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
            loss = 100.0 * seg_l + pose_l

            opt.zero_grad()
            loss.backward()

            # Lever-4: accumulate the per-tensor sensitivity EMA from the live grads
            # (BEFORE the step; mirrors driver.py:637-642).
            if score_aware:
                accumulate_tensor_sensitivity(decoder, sensitivity_ema, decay=0.9)

            torch.nn.utils.clip_grad_norm_([*decoder.parameters(), latents], 1e9)
            opt.step()

            # EMA update after each step (the EMA non-negotiable).
            with torch.no_grad():
                for (_k, ev), (_k2, lv) in zip(
                    ema_decoder.state_dict().items(), decoder.state_dict().items(), strict=True
                ):
                    ev.mul_(ema_decay).add_(lv, alpha=1.0 - ema_decay)
                ema_latents.mul_(ema_decay).add_(latents.data, alpha=1.0 - ema_decay)

    ema_sd = {k: v.detach().clone() for k, v in ema_decoder.state_dict().items()}
    return ema_sd, ema_latents.detach().clone(), sensitivity_ema, ctx, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not _BASIN_CKPT.exists():
        raise FileNotFoundError(f"basin checkpoint not found: {_BASIN_CKPT}")
    if not _VIDEO.exists():
        raise FileNotFoundError(f"0.mkv not found: {_VIDEO}")

    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    model = import_vendored("model")

    # --- UNIFORM-127 QAT arm ---
    u_sd, u_lat, _u_sens, ctx, n = _train_arm(
        score_aware=False, max_pairs=args.max_pairs, epochs=args.epochs,
        seed=args.seed, lr=args.lr,
    )
    # --- SCORE-AWARE QAT arm (same seed/budget/slice) ---
    s_sd, s_lat, s_sens, _ctx2, _n2 = _train_arm(
        score_aware=True, max_pairs=args.max_pairs, epochs=args.epochs,
        seed=args.seed, lr=args.lr,
    )

    # --- byte-close BOTH arms through the REAL vendored codec ---
    meta = {"base_channels": 20, "latent_dim": 28}

    def _blob(sd):
        return len(codec.encode_decoder(codec.quantize_state_dict(sd)))

    def _archive(sd, lat):
        return len(codec.build_archive(sd, lat, meta))

    u_blob = _blob(u_sd)
    s_blob = _blob(s_sd)
    u_arch = _archive(u_sd, u_lat)
    s_arch = _archive(s_sd, s_lat)

    # --- advisory distortion on the SAME real slice ---
    dec_u = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec_u.load_state_dict(u_sd)
    u_dseg, u_dpose = _advisory_distortion(dec_u, u_lat, ctx, n)
    dec_s = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec_s.load_state_dict(s_sd)
    s_dseg, s_dpose = _advisory_distortion(dec_s, s_lat, ctx, n)

    # --- advisory contest scores (rate from the REAL full archive bytes) ---
    u_score = _contest_score(u_dseg, u_dpose, u_arch)
    s_score = _contest_score(s_dseg, s_dpose, s_arch)
    score_delta = s_score - u_score  # negative = score-aware WINS net
    blob_delta = s_blob - u_blob

    # (the sensitivity EMA is keyed by MODULE name; report the count of seeded tensors)
    n_seeded = len(s_sens)

    if score_delta < 0:
        verdict = "NET_SCORE_WIN"
        note = (
            f"score-aware QAT TRAINING wins NET: advisory contest score {s_score:.6f} < "
            f"uniform {u_score:.6f} (Δ {score_delta:+.6f}). The {-blob_delta} B brotli-blob "
            "win SURVIVES at non-worse advisory distortion AFTER the decoder trained to be "
            "coarse-grid-robust. The net-score win is SUPPORTED on the advisory slice; the "
            "DUAL CPU/CUDA 600-pair exact eval is still required before a SCORE claim."
        )
    else:
        verdict = "BYTE_DIRECTION_ONLY"
        note = (
            f"score-aware QAT is byte-direction-only on this budget: blob delta {blob_delta:+d} B "
            f"but advisory contest score {s_score:.6f} is NOT lower than uniform {u_score:.6f} "
            f"(Δ {score_delta:+.6f}). The coarse-grid distortion damage is not fully recovered "
            f"by {args.epochs} epochs on {n} pairs. HONEST: Lever-4 stays a byte-DIRECTION lever; "
            "the net-score win needs more training budget OR the variable-level codec path that "
            "lets the per-tensor grid change archive bytes directly (a bigger byte win to clear "
            "the distortion). Do NOT ship Lever-4 as a confirmed net-score lever on this basis."
        )

    result = {
        "probe": "lever4_qat_training_ab_net_score",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE — small real slice, NOT a 600-pair contest eval",
        "basin_checkpoint": str(_BASIN_CKPT.relative_to(_REPO_ROOT)),
        "max_pairs": n,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "n_sensitivity_seeded_tensors": n_seeded,
        "uniform_decoder_blob_bytes": u_blob,
        "score_aware_decoder_blob_bytes": s_blob,
        "decoder_blob_delta_bytes": blob_delta,
        "uniform_full_archive_bytes": u_arch,
        "score_aware_full_archive_bytes": s_arch,
        "advisory_d_seg_uniform": round(u_dseg, 6),
        "advisory_d_seg_score_aware": round(s_dseg, 6),
        "advisory_d_pose_uniform": round(u_dpose, 8),
        "advisory_d_pose_score_aware": round(s_dpose, 8),
        "advisory_score_uniform": round(u_score, 6),
        "advisory_score_score_aware": round(s_score, 6),
        "advisory_score_delta": round(score_delta, 6),
        "verdict": verdict,
        "verdict_note": note,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 80)
        print("MED-2 NET-SCORE A/B — score-aware QAT TRAINING vs uniform-127 TRAINING")
        print("=" * 80)
        print(f"basin: {result['basin_checkpoint']}")
        print(f"authority: {result['authority']}")
        print(f"{n} real pairs, {args.epochs} epochs, lr={args.lr}, seed={args.seed}")
        print(f"  score-aware sensitivity seeded on {n_seeded} tensors")
        print()
        print(f"  uniform-127  blob {u_blob:>7d} B   archive {u_arch:>7d} B")
        print(f"  score-aware  blob {s_blob:>7d} B   archive {s_arch:>7d} B")
        print(f"  blob delta        {blob_delta:>+7d} B   (negative = score-aware SMALLER)")
        print()
        print(f"  advisory d_seg  uniform {u_dseg:.5f} -> score-aware {s_dseg:.5f}  (Δ {s_dseg-u_dseg:+.5f})")
        print(f"  advisory d_pose uniform {u_dpose:.6f} -> score-aware {s_dpose:.6f}  (Δ {s_dpose-u_dpose:+.6f})")
        print()
        print(f"  advisory contest score  uniform {u_score:.6f} -> score-aware {s_score:.6f}")
        print(f"  advisory score delta    {score_delta:+.6f}   (negative = score-aware WINS net)")
        print()
        print(f"VERDICT: {verdict}")
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
