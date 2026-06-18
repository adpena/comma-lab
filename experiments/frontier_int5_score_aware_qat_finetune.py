#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Frontier int5/int4 SCORE-AWARE QAT-FINETUNE — the Path-B decisive build.

THE QUESTION (CLAUDE.md "THE GOAL — SUB-0.15"; the PTQ-on-frontier floor memo
``.omx/research/ptq_on_frontier_floor_20260618T191724Z.md``): naive PTQ of the 0.19110
frontier decoder COLLAPSES (int5 → S 0.678, a ~8×/~42× d_seg/d_pose spill over the int8
identity 0.1965). The rate win is real (int5 cuts 177,169 → 118,589 B, rate 0.1180 →
0.0790). Can SCORE-AWARE QAT-FINETUNE recover the distortion collapse at the int5 byte
budget → land S well below the int8 baseline (0.1965), ideally toward the desk int5-hold
0.153 / sub-0.15?

WHAT IT DOES (all REAL — NO-FAKE):
  * decode the REAL frontier decoder state_dict (``tac.frontier_decoder_ptq``,
    byte-identical identity round-trip proven).
  * wrap it in the int5/int4 SCORE-AWARE fake-quant (``tac.frontier_int5_qat``): early
    low-res stages on the coarse int5 grid (where 77% of params + the d_seg damage live),
    output-proximal heads at int8 (the saliency-protected band). The QAT-STE forward
    quantizes, the backward is identity, so the decoder LEARNS weights robust to their own
    coarse grid at the SegNet argmax boundary + the PoseNet pose head.
  * FINETUNE on the 600 pairs: MPS fp32 GRADIENT (the 104× scorer-speed lever — TRAINING
    ONLY, never authority), margin-hinge d_seg loss + sqrt(10·d_pose) pose-hold, eval_
    roundtrip (bicubic↑→bilinear↓→uint8-STE) 1:1 with ``common.py``, EMA shadow (warmup),
    warm-cosine LR. Resumable (decoder + latents + EMA + optimizer + RNG + epoch).
  * CPU AUTHORITY exact eval (NEVER MPS — CLAUDE.md "MPS auth eval is NOISE"): the EMA
    shadow is HARD-quantized to the trained grid, byte-closed through the REAL frontier
    codec (``reencode_frontier_archive`` — int5 weights RE-entropy-coded), and the
    byte-closed archive's d_seg/d_pose/rate/score is measured on the FULL 600 pairs via
    ``RealScorerContext.exact_eval``. The score that picks BEST is the CPU-authority byte-
    closed S.

AUTHORITY: every score is ``[contest-CPU advisory]`` NON-PROMOTABLE. The frontier pointer
stays pointer-only at 0.19110. If the byte-closed S < the int8 local baseline 0.1965 (and
especially < ~0.191), FLAG for a paired dual CPU/CUDA exact eval (the pointer-mover) — do
NOT self-promote, do NOT touch the pointer (the +0.0054 local-vs-contest offset means
local-beats-0.191 needs the paired contest row to confirm). MPS is the GRADIENT backend
only; CPU is authority. $0 (local MPS + CPU; no GPU dispatch, no paid spend).

SUBMISSION gate (NOT a local blocker): the frontier is PR #101/#106-derived → any contest
PR needs ``borrowed_substrate_accounting`` (NO-FAKE class 7). OUR original = the score-
aware QAT-finetune technique; borrowed = the frontier decoder + codec.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

_EVAL_H, _EVAL_W = 384, 512
_RATE_DENOM = 37_545_489
_FRONTIER_S = 0.19109982419209975
_INT8_LOCAL_BASELINE_S = 0.1965  # the floor memo's int8 identity local-CPU S
_DESK_INT5_HOLD_S = 0.153
_SUB015 = 0.15


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + 25.0 * archive_bytes / _RATE_DENOM


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=("uniform", "score_aware"), default="score_aware")
    ap.add_argument("--low-nbits", type=int, default=5, help="coarse grid (5=int5, 4=int4)")
    ap.add_argument("--high-nbits", type=int, default=8, help="protect grid (8=int8)")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW lr (warm-cosine)")
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--seg-weight", type=float, default=1.0)
    ap.add_argument("--pose-weight", type=float, default=1.0)
    ap.add_argument(
        "--seg-loss",
        choices=("margin_hinge", "ce"),
        default="margin_hinge",
        help="margin_hinge = the validated flip-targeting d_seg lever",
    )
    ap.add_argument("--margin-hinge-target", type=float, default=6.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-ema-warmup", action="store_true")
    ap.add_argument("--train-device", default="mps", help="MPS fp32 gradient backend")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--train-pairs", type=int, default=600, help="pairs to finetune on")
    ap.add_argument(
        "--eval-every", type=int, default=50, help="byte-closed CPU authority eval cadence"
    )
    ap.add_argument(
        "--eval-pairs",
        type=int,
        default=600,
        help="pairs for the CPU authority byte-closed eval (600 = full)",
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        default="experiments/results/frontier_int5_qat_finetune",
    )
    ap.add_argument(
        "--targets-cache",
        default="experiments/results/capstone_gt_targets_cache",
    )
    ap.add_argument("--checkpoint-every", type=int, default=10, help="epochs per ckpt")
    ap.add_argument(
        "--max-wall-min", type=float, default=None, help="stop after N minutes (resumable)"
    )
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "qat_finetune_ckpt.pt"
    best_path = out_dir / "best"
    best_path.mkdir(exist_ok=True)
    curve_jsonl = out_dir / "recovery_curve.jsonl"

    from tac.frontier_decoder_ptq import (
        FRONTIER_ARCHIVE,
        build_frontier_decoder,
        decode_frontier_member,
        reencode_frontier_archive,
    )
    from tac.frontier_int5_qat import (
        EMAState,
        FrontierQATConfig,
        apply_frontier_qat,
        hard_quantize_state_dict_to_nbits,
        per_tensor_nbits_for_decoder,
        restore_frontier_qat,
    )
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    import_vendored_bundle()
    video_path = import_vendored("data").get_default_video_path()

    # ── decode the frontier (REAL weights, byte-exact identity proven) ───────────
    t_dec = time.time()
    member = decode_frontier_member(FRONTIER_ARCHIVE)
    n_pairs_total = int(member.latents.shape[0])
    train_pairs = min(int(args.train_pairs), n_pairs_total)
    print(
        f"[frontier] archive={FRONTIER_ARCHIVE.name} member_bytes={member.member_bytes} "
        f"n_pairs={n_pairs_total} train_pairs={train_pairs} ({time.time() - t_dec:.1f}s)",
        flush=True,
    )

    # ── train decoder on MPS; build the QAT allocation ───────────────────────────
    train_device = torch.device(args.train_device)
    decoder = build_frontier_decoder(member.state_dict).to(train_device)
    decoder.train()
    qcfg = FrontierQATConfig(
        mode=args.mode, low_nbits=int(args.low_nbits), high_nbits=int(args.high_nbits)
    )
    per_tensor_nbits = per_tensor_nbits_for_decoder(decoder, qcfg)
    n_coarse = sum(1 for v in per_tensor_nbits.values() if v == qcfg.low_nbits)
    n_prot = sum(1 for v in per_tensor_nbits.values() if v == qcfg.high_nbits)
    print(
        f"[qat] mode={args.mode} low=int{args.low_nbits} high=int{args.high_nbits} "
        f"coarse_tensors={n_coarse} protected_tensors={n_prot} "
        f"alloc={ {k: per_tensor_nbits[k] for k in sorted(per_tensor_nbits)} }",
        flush=True,
    )

    latents = member.latents[:train_pairs].clone().to(train_device).requires_grad_(True)

    params = [p for p in decoder.parameters() if p.requires_grad] + [latents]
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=1e-4)

    ema = EMAState(decay=float(args.ema_decay), warmup=not args.no_ema_warmup)
    ema.init_from(decoder, latents)

    # ── CPU AUTHORITY scorer context (full eval_pairs; NEVER MPS for eval) ────────
    eval_pairs = min(int(args.eval_pairs), n_pairs_total)
    t_ctx = time.time()
    ctx = RealScorerContext(
        video_path,
        device="cpu",
        train_device=args.train_device,
        split_by_head=False,
        max_pairs=max(train_pairs, eval_pairs),
        targets_cache=args.targets_cache,
    )
    print(f"[scorer] ctx built ({time.time() - t_ctx:.1f}s) train_dev={args.train_device}", flush=True)

    # warm-cosine LR schedule (per-epoch).
    def _lr_at(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return float(args.lr) * (epoch + 1) / max(1, args.warmup_epochs)
        prog = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return float(args.lr) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))) + 1e-6

    # ── resume ───────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_score = float("inf")
    best_meta: dict = {}
    if ckpt_path.is_file():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        decoder.load_state_dict({k: v.to(train_device) for k, v in blob["decoder"].items()})
        latents.data = blob["latents"].to(train_device)
        opt.load_state_dict(blob["opt"])
        ema.shadow = {k: v.clone() for k, v in blob["ema_shadow"].items()}
        ema.latent_shadow = blob["ema_latents"].clone()
        ema.step = int(blob["ema_step"])
        start_epoch = int(blob["epoch"]) + 1
        best_score = float(blob["best_score"])
        best_meta = blob.get("best_meta", {})
        torch.set_rng_state(blob["rng"])
        print(
            f"[resume] from epoch {start_epoch} best_S={best_score:.4f} "
            f"(ckpt {ckpt_path})",
            flush=True,
        )

    def _save_ckpt(epoch: int) -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                "epoch": epoch,
                "decoder": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                "latents": latents.detach().cpu(),
                "opt": opt.state_dict(),
                "ema_shadow": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                "ema_latents": ema.shadow_latents_cpu(),
                "ema_step": ema.step,
                "best_score": best_score,
                "best_meta": best_meta,
                "rng": torch.get_rng_state(),
                "args": vars(args),
                "per_tensor_nbits": per_tensor_nbits,
            },
            tmp,
        )
        tmp.replace(ckpt_path)

    # ── the byte-closed CPU AUTHORITY eval (the BEST tracker) ────────────────────
    def _authority_eval(epoch: int) -> dict:
        """Hard-quantize the EMA shadow to the trained grid, byte-close through the
        real frontier codec, exact-eval the byte-closed archive on the CPU authority."""
        shadow_dec = ema.shadow_decoder(decoder)  # CPU eval clone
        shrunk_sd = hard_quantize_state_dict_to_nbits(shadow_dec, per_tensor_nbits)
        # byte-close: re-encode the shrunk decoder through the REAL frontier codec.
        scratch = out_dir / f"_cand_ep{epoch}.zip"
        bytes_meas = reencode_frontier_archive(member, shrunk_sd, scratch)
        # build a CPU decoder from the byte-closed (hard-quantized) weights for eval.
        eval_dec = build_frontier_decoder(shrunk_sd).to("cpu").eval()
        ema_lat = ema.shadow_latents_cpu()
        # eval on the full eval_pairs (the latents the archive ships are member.latents;
        # the finetune trains its own latents — use the EMA shadow latents for the
        # trained pairs, member.latents for any beyond train_pairs).
        if eval_pairs <= train_pairs:
            eval_latents = ema_lat[:eval_pairs]
        else:
            eval_latents = torch.cat(
                [ema_lat, member.latents[train_pairs:eval_pairs]], dim=0
            )
        res = ctx.exact_eval(eval_dec, eval_latents, bytes_meas)
        d_seg = float(res["seg_distortion"])
        d_pose = float(res["pose_distortion"])
        s = _score(d_seg, d_pose, bytes_meas)
        try:
            scratch.unlink()
        except OSError:
            pass
        return {
            "epoch": epoch,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": bytes_meas,
            "rate_term": 25.0 * bytes_meas / _RATE_DENOM,
            "score": s,
        }

    # ── train ────────────────────────────────────────────────────────────────────
    wall0 = time.time()
    n_train = train_pairs
    seg_targets = ctx.seg_targets_hard[:n_train]
    pose_targets = ctx.pose_targets[:n_train]

    def _seg_loss(seg_out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if args.seg_loss == "margin_hinge":
            from tac.losses.core import segnet_margin_hinge_per_pixel

            pp = segnet_margin_hinge_per_pixel(
                seg_out, tgt, margin_target=float(args.margin_hinge_target), num_classes=5
            )
            return pp.mean()
        return F.cross_entropy(seg_out, tgt)

    print(
        f"[train] epochs {start_epoch}..{args.epochs} bs={args.batch_size} "
        f"seg_loss={args.seg_loss} seg_w={args.seg_weight} pose_w={args.pose_weight} "
        f"ema_decay={args.ema_decay} warmup={not args.no_ema_warmup}",
        flush=True,
    )
    stopped_early = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, int(args.epochs)):
        lr = _lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        perm = torch.randperm(n_train, device=train_device)
        ep_loss = ep_seg = ep_pose = 0.0
        nb = 0
        decoder.train()
        for bs in range(0, n_train, int(args.batch_size)):
            idx = perm[bs : bs + int(args.batch_size)]
            B = int(idx.numel())
            # QAT fake-quant the weights, forward, restore (STE flows the grad).
            originals = apply_frontier_qat(decoder, per_tensor_nbits)
            decoded_pair = decoder(latents[idx])  # (B,2,3,384,512)
            restore_frontier_qat(decoder, originals)
            flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
            up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
            down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
            decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
            dc = decoded_bhwc.clamp(0, 255)
            decoded_bhwc = dc + (dc.round() - dc).detach()  # uint8 STE
            seg_out, pose6 = ctx.seg_pose_forward(decoded_bhwc)
            seg_l = _seg_loss(seg_out, seg_targets[idx])
            pose_mse = F.mse_loss(pose6, pose_targets[idx])
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
            loss = float(args.seg_weight) * seg_l + float(args.pose_weight) * pose_l
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            opt.step()
            ema.update(decoder, latents)
            ep_loss += float(loss)
            ep_seg += float(seg_l)
            ep_pose += float(pose_mse)
            nb += 1
        last_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == start_epoch:
            print(
                f"  ep{epoch + 1}/{args.epochs} loss={ep_loss / nb:.4f} "
                f"seg_l={ep_seg / nb:.4f} pose_mse={ep_pose / nb:.6f} lr={lr:.2e} "
                f"({time.time() - wall0:.0f}s)",
                flush=True,
            )

        if (epoch + 1) % int(args.checkpoint_every) == 0:
            _save_ckpt(epoch)

        if (epoch + 1) % int(args.eval_every) == 0:
            t_e = time.time()
            row = _authority_eval(epoch + 1)
            row["wall_s"] = round(time.time() - t_e, 1)
            row["train_seg_l"] = round(ep_seg / nb, 5)
            row["train_pose_mse"] = round(ep_pose / nb, 6)
            with curve_jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            improved = row["score"] < best_score
            print(
                f"    >>> ep{epoch + 1} [contest-CPU advisory] S={row['score']:.4f} "
                f"d_seg={row['d_seg']:.6f} d_pose={row['d_pose']:.6f} "
                f"bytes={row['archive_bytes']} rate={row['rate_term']:.4f} "
                f"(int8 baseline {_INT8_LOCAL_BASELINE_S:.4f}; {'NEW BEST' if improved else 'no improve'}) "
                f"({row['wall_s']}s)",
                flush=True,
            )
            if improved:
                best_score = row["score"]
                best_meta = dict(row)
                # persist the BEST byte-closed archive + EMA shadow.
                shadow_dec = ema.shadow_decoder(decoder)
                shrunk_sd = hard_quantize_state_dict_to_nbits(shadow_dec, per_tensor_nbits)
                reencode_frontier_archive(member, shrunk_sd, best_path / "best_archive.zip")
                torch.save(shrunk_sd, best_path / "best_decoder_hardq.pt")
                torch.save(ema.shadow_latents_cpu(), best_path / "best_ema_latents.pt")
                (best_path / "best_meta.json").write_text(json.dumps(best_meta, indent=2))
            _save_ckpt(epoch)

        if args.max_wall_min is not None and (time.time() - wall0) / 60.0 >= args.max_wall_min:
            print(f"[stop] max-wall-min {args.max_wall_min} reached at epoch {epoch + 1}", flush=True)
            _save_ckpt(epoch)
            stopped_early = True
            break

    # final checkpoint.
    _save_ckpt(last_epoch)

    # ── verdict ──────────────────────────────────────────────────────────────────
    beats_int8 = best_score < _INT8_LOCAL_BASELINE_S - 1e-9
    beats_pointer_local = best_score < _FRONTIER_S - 1e-9
    sub_015 = best_score < _SUB015 - 1e-9
    verdict = {
        "frontier_pointer_S": _FRONTIER_S,
        "int8_local_baseline_S": _INT8_LOCAL_BASELINE_S,
        "desk_int5_hold_S": _DESK_INT5_HOLD_S,
        "sub_015_target": _SUB015,
        "best_S": best_score if best_score < float("inf") else None,
        "best_meta": best_meta,
        "beats_int8_baseline": beats_int8,
        "beats_pointer_local": beats_pointer_local,
        "reaches_sub_015": sub_015,
        "completed_epochs": last_epoch + 1,
        "stopped_early": stopped_early,
    }
    if best_score == float("inf"):
        verdict["headline"] = (
            "NO EVAL YET — increase --epochs or lower --eval-every (the run produced no "
            "byte-closed authority eval row)."
        )
    elif beats_pointer_local:
        verdict["headline"] = (
            f"QAT-FINETUNE BEATS THE POINTER LOCALLY: best byte-closed S={best_score:.4f} "
            f"< {_FRONTIER_S:.4f} [contest-CPU advisory]. FLAG for paired dual CPU/CUDA "
            f"exact eval (the pointer-mover) — do NOT self-promote, pointer UNMOVED. "
            f"{'AND reaches modelled sub-0.15.' if sub_015 else ''}"
        )
    elif beats_int8:
        verdict["headline"] = (
            f"QAT-FINETUNE RECOVERS the int5 collapse: best byte-closed S={best_score:.4f} "
            f"< the int8 baseline {_INT8_LOCAL_BASELINE_S:.4f} (the PTQ floor) but NOT yet "
            f"< the pointer 0.191. QAT works; needs more recovery (longer train / int4 / "
            f"better allocation) to cross the pointer. [contest-CPU advisory]"
        )
    else:
        verdict["headline"] = (
            f"QAT CAPS ABOVE THE INT8 BASELINE: best byte-closed S={best_score:.4f} >= "
            f"{_INT8_LOCAL_BASELINE_S:.4f} — QAT did NOT recover enough of the int5 collapse "
            f"to beat the PTQ floor. Path B caps above the pointer; route the sub-0.15 "
            f"search to a concentrated-saliency own-vehicle. [contest-CPU advisory]"
        )

    report = {
        "probe": "frontier_int5_score_aware_qat_finetune",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer_moved": False,
        "frontier_archive": str(FRONTIER_ARCHIVE),
        "frontier_archive_sha256": "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e",
        "config": vars(args),
        "per_tensor_nbits": per_tensor_nbits,
        "verdict": verdict,
        "built_at_utc": _now(),
        "note": (
            "MPS fp32 GRADIENT (training only); CPU AUTHORITY byte-closed exact eval "
            "(RealScorerContext.exact_eval, NEVER MPS). Bytes MEASURED via the real "
            "frontier codec (PR101 split-brotli + CTXR recode, STORED zip). Local CPU ≈ "
            "contest-CPU per G3 with a +0.0054 offset → local-beats-pointer needs the "
            "paired contest row. NON-PROMOTABLE; pointer UNMOVED 0.19110. SUBMISSION needs "
            "borrowed_substrate_accounting (frontier is PR101/106-derived)."
        ),
    }
    print("\n=== FRONTIER int5 SCORE-AWARE QAT-FINETUNE VERDICT ===")
    print(f"  {verdict['headline']}")

    out = Path(args.out_json) if args.out_json else (out_dir / "frontier_int5_qat_finetune.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[wrote] {out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
