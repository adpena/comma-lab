# SPDX-License-Identifier: MIT
"""Capstone campaign runner (Task #78/#65) — the missing local actuator.

Trains the ORIGINAL VQ-NeRV + FiLM-pose capstone basis against the LIVE frozen
contest scorer (the #82 1:1-MLX-port bridge), at a chosen byte budget
(``base_channels`` + ``decoder_dtype``), byte-closes the int8 archive, and
recomputes the EXACT advisory score from the live-render d_seg/d_pose + the
real archive.zip size (``evaluate.py`` law). $0, local, MLX-renderer +
torch-CPU-scorer (TRUSTED per CLAUDE.md "local CPU + MLX GPU good"; MPS NEVER).

This is the thin CLI per CLAUDE.md "thin CLIs delegate to tac modules": every
real piece lives in ``tac`` — ``score_aware_loop.targets`` (frozen DistortionNet
+ GT targets), ``mlx_pr95_port.score_bridge`` (the torch<->mlx vjp bridge),
``capstone_vq_nerv`` (bundle + trainer + int8 export).

Authority: the score this prints is ``[macOS-CPU advisory]`` (the torch scorer
on local CPU is trusted but it is NOT a contest-axis row). It RANKS + gates; it
does NOT move the canonical frontier pointer. A sub-0.15 advisory here is the
GATE to a paired contest-CPU+CUDA exact eval (the only pointer-moving step).

Usage (smoke):
    .venv/bin/python experiments/run_capstone_campaign.py \
        --max-pairs 2 --base-channels 16 --epochs 3 --decoder-dtype int8 \
        --out-dir experiments/results/capstone_smoke

Usage (decisive budget run, local detached daemon):
    nohup .venv/bin/python experiments/run_capstone_campaign.py \
        --max-pairs 600 --base-channels 16 --epochs 300 --decoder-dtype int8 \
        --muon-lr 3e-2 --grad-clip 50 --grad-clip-muon 50 \
        --out-dir experiments/results/capstone_full_b16_int8 \
        > .omx/tmp/capstone_full_b16_int8.log 2>&1 &
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

RATE_DENOM = 37_545_489  # evaluate.py:64


class _StreamingTelemetry(list):
    """Telemetry sink that STREAMS each trajectory row to a JSONL + stdout as the
    trainer appends it (``CapstoneTrainer.train`` calls ``cfg.telemetry.append``
    per eval_every). Without this the trajectory is only returned at the END of
    train() — a long run shows NO mid-run signal (the "Max observability"
    non-negotiable). Tail the JSONL for the live RD curve."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._fh = open(path, "w")  # noqa: SIM115 (lifetime = the run)

    def append(self, row: dict) -> None:  # type: ignore[override]
        super().append(row)
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        print(f"  [epoch {row.get('epoch')}] exact_d_seg={row.get('exact_d_seg'):.5f} "
              f"mean_d_pose={row.get('mean_d_pose'):.5f}", flush=True)


def _archive_with_config(payload: bytes, config: dict) -> bytes:
    """Wrap ``payload`` (member ``x``) + the ``capstone_config_v1`` JSON sidecar
    in a STORED ZIP. The sidecar carries the render basis config the contest
    inflate needs (base_channels / pose_mean,std / film_enabled / num_pairs /
    decoder_dtype) — a payload-only archive is NOT inflatable for a FiLM bundle
    (the subagent NO-FAKE find). Both STORED (the payload is already compressed);
    the sidecar adds ~one ZIP member of overhead (~250 B), the price of a
    contest-VALID archive vs the #79 minimal-but-uninflatable container."""
    cfg_bytes = json.dumps(config, separators=(",", ":"), sort_keys=True).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", payload)
        zf.writestr("capstone_config_v1", cfg_bytes)
    return buf.getvalue()


def _load_or_build_targets(max_pairs: int, cache_dir: Path, device: str):
    """Cache (seg_targets_hard, pose_targets) — the slow GT precompute. Reusable
    across base_channels / epochs sweeps."""
    from tac.score_aware_loop.targets import build_gt_targets, load_frozen_distortion_net

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"gt_targets_n{max_pairs}.pt"
    net = load_frozen_distortion_net(device=device)
    if cache.exists():
        blob = torch.load(cache, map_location=device, weights_only=False)
        seg_t, pose_t, n = blob["seg"], blob["pose"], int(blob["n"])
        print(f"[targets] loaded cache {cache} n={n}", flush=True)
        return net, seg_t, pose_t, n
    t0 = time.time()
    seg_t, pose_t, n = build_gt_targets(net, max_pairs=max_pairs, device=device)
    torch.save({"seg": seg_t, "pose": pose_t, "n": n}, cache)
    print(f"[targets] built+cached n={n} in {time.time()-t0:.1f}s -> {cache}", flush=True)
    return net, seg_t, pose_t, n


def _export_int8_archive(trainer, pose_store: np.ndarray, decoder_dtype: str):
    """Byte-close the CONTEST-INFLATABLE archive (carrier-aware) + config sidecar.

    Branches on ``trainer.bundle.carrier``:

    * ``vq_index``: the FULL render basis (decoder + per-frame FiLM, contest-keyed)
      + codebook + REAL trained bit-packed VQ indices + stored pose.
    * ``stored_latent``: the FULL render basis + REAL trained per-pair 28-d latent
      (temporal-delta + LZMA, NO codebook/index) + stored pose. The rich-carrier
      VQ-index-impoverishment fix (28 floats/pair >> 8 bits the index gives).

    Verified score-parity with the numpy inflate (d_seg EXACT). Returns
    (archive_zip_bytes, account, payload_bytes, config).

    [A1] The render-basis weights AND the per-pair carrier (vq codebook or stored
    latents) are the EMA SHADOW (via ``trainer.export_render_weights()`` /
    ``trainer.export_stored_latents()``) — NOT the live final-step weights. This is
    the EMA non-negotiable: the archive bytes equal the same averaged shadow the
    advisory d_seg/d_pose are measured on (eval+export the shadow)."""
    import dataclasses

    from tac.capstone_vq_nerv.export import (
        build_capstone_archive_bytes,
        build_capstone_stored_latent_archive_bytes,
    )
    from tac.capstone_vq_nerv.numpy_reference import decode_config_from_bundle

    bundle = trainer.bundle
    decoder_weights = trainer.export_render_weights()  # [A1] EMA shadow, contest naming
    config = dataclasses.asdict(decode_config_from_bundle(bundle))
    config["num_pairs"] = int(pose_store.shape[0])
    config["decoder_dtype"] = decoder_dtype
    config["carrier"] = bundle.carrier

    if bundle.carrier == "stored_latent":
        latents = trainer.export_stored_latents()  # [A1] EMA shadow, REAL trained
        payload, account = build_capstone_stored_latent_archive_bytes(
            decoder_weights=decoder_weights,
            latents=latents,
            pose_scalars=np.asarray(pose_store, dtype=np.float32),
            decoder_dtype=decoder_dtype,
        )
    else:
        codebook = np.asarray(bundle.quantizer._codebook, dtype=np.float32)
        vq_indices = np.asarray(bundle.all_vq_indices(), dtype=np.int32)  # REAL trained
        payload, account = build_capstone_archive_bytes(
            decoder_weights=decoder_weights,
            codebook=codebook,
            vq_indices=vq_indices,
            pose_scalars=np.asarray(pose_store, dtype=np.float32),
            codebook_size=int(codebook.shape[0]),
            decoder_dtype=decoder_dtype,
        )
    archive_zip = _archive_with_config(payload, config)
    return archive_zip, account, payload, config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=2)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--decoder-dtype", choices=("fp16", "int8"), default="int8")
    ap.add_argument(
        "--carrier", choices=("vq_index", "stored_latent"), default="vq_index",
        help="Per-pair carrier geometry. vq_index = the legacy 8-bit VQ codebook "
             "index (pose-impoverished). stored_latent = store the rich 28-d latent "
             "directly (temporal-delta + LZMA, the frontier's pose-capable carrier).",
    )
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument(
        "--hinerv-grid-pe", action="store_true",
        help="Enable the HiNeRV grid positional-encoding (deterministic coord grid "
             "+ tiny learned projection added to the stem feature; ~0 stored bytes "
             "beyond the projection). Default off = byte-identical to the HNeRV stem.",
    )
    ap.add_argument(
        "--grid-pe-num-freqs", type=int, default=4,
        help="Grid-PE encoding bandwidth (pe_dim = 4*num_freqs). Default 4 (the "
             "spectral-atlas-principled low-freq budget: scorer energy is LOW-freq).",
    )
    ap.add_argument(
        "--tie-depth", type=int, default=0,
        help="L1 weight-tie: share the first N leading base_ch->base_ch upsample "
             "blocks' conv (the LARGEST tensors) with a per-stage FiLM symmetry-"
             "breaker — the inflate-compute rate lever that removes (N-1) conv "
             "tensors from the int8 decoder blob. 0/1 = no tie (byte-identical). The "
             "canonical taper allows max 2 (blocks 0,1 are base_ch->base_ch).",
    )
    ap.add_argument(
        "--margin-hinge-weight", type=float, default=0.0,
        help="[L7] Cross-hardware-robust margin hinge weight. >0 ADDS "
             "weight*mean(relu(margin_floor - margin)) to the stage seg-loss so the "
             "boundary argmax survives macOS->numpy->Linux/CUDA logit drift "
             "(numpy-portability guard). 0 = off. NOT supported with --scorer-backend "
             "mlx_gpu (fails closed; the shared MLX-GPU bridge has no wrappable hook).",
    )
    ap.add_argument(
        "--margin-hinge-floor", type=float, default=0.1,
        help="[L7] The required margin floor (anchor ~0.1 > the measured ~0.096 "
             "cross-hardware logit drift). Used only when --margin-hinge-weight>0.",
    )
    ap.add_argument("--seg-weight", type=float, default=100.0)
    ap.add_argument("--pose-weight", type=float, default=1.0)
    ap.add_argument("--muon-lr", type=float, default=3e-2)
    ap.add_argument("--grad-clip", type=float, default=50.0)
    ap.add_argument("--grad-clip-muon", type=float, default=50.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--curriculum", choices=("none", "pr95_8stage"), default="none",
        help="PR95 L14 8-stage curriculum (the d_seg-floor-breaking schedule). "
             "'none' = the legacy single fixed stage (ce_seg_loss + fixed Muon LR).",
    )
    ap.add_argument(
        "--optimizer-schedule",
        choices=("pr95_adamw_then_muon", "muon_throughout"),
        default="muon_throughout",
        help="pr95_adamw_then_muon = FAITHFUL to PR95 (AdamW stages 1-7, Muon stage 8 "
             "only). muon_throughout = #77 deviation (Muon from stage 1).",
    )
    ap.add_argument(
        "--curriculum-total-epochs", type=int, default=None,
        help="Spread this many epochs across the 8 stages proportionally to the "
             "canonical counts. Default: use --epochs as the total.",
    )
    ap.add_argument("--device", default="cpu")  # torch scorer device; NEVER mps
    ap.add_argument(
        "--scorer-backend",
        choices=("torch_cpu_bridge", "mlx_gpu"),
        default="torch_cpu_bridge",
        help="Per-step score-aware gradient backend. torch_cpu_bridge (DEFAULT) = "
             "the frozen torch-CPU scorer (the AUTHORITY; ~18min/epoch bottleneck). "
             "mlx_gpu = the MLX-GPU end-to-end scorer-loss path (fast training "
             "SIGNAL; torch-CPU re-scores every --authority-recheck-every steps + "
             "every eval for the reported d_seg/d_pose). See "
             ".omx/research/mlx_gpu_scorer_training_wirein_20260611.md.",
    )
    ap.add_argument(
        "--authority-recheck-every", type=int, default=0,
        help="torch-CPU authority d_seg re-score cadence (steps) when "
             "--scorer-backend=mlx_gpu. 0 disables per-step re-scoring (eval still "
             "uses torch-CPU). Telemetry only; does NOT change the gradient.",
    )
    ap.add_argument("--out-dir", default="experiments/results/capstone_smoke")
    ap.add_argument("--targets-cache", default="experiments/results/capstone_gt_targets_cache")
    args = ap.parse_args()

    if args.device == "mps":
        raise SystemExit("MPS is NEVER an authority (CLAUDE.md). Use --device cpu.")

    # [FP32-EXACT GPU SCORER] When the MLX-GPU training scorer is selected, force the
    # non-NAX Metal GEMM kernel via the arch override so the GPU SegNet/PoseNet are
    # FP32-EXACT vs torch-CPU (243->0 d_seg flips, pose 2.76e-4->8.7e-11), at zero
    # throughput cost, decoder unaffected (whole-process override is safe). MUST be
    # set BEFORE the first ``import mlx`` (MLX reads the env at runtime init); the
    # campaign imports MLX lazily inside the trainer/bundle, so setting it here at the
    # top of main() (before those imports) is in time. Per
    # ``.omx/research/arch_override_fp32_exact_gpu_training_scorer_20260611.md``.
    if args.scorer_backend == "mlx_gpu":
        import os

        os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g15")
        print(
            f"[fp32-exact] MLX_METAL_GPU_ARCH={os.environ['MLX_METAL_GPU_ARCH']} "
            "(non-NAX -> GPU scorer FP32-exact vs torch-CPU)", flush=True
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    net, seg_t, pose_t, n = _load_or_build_targets(
        args.max_pairs, Path(args.targets_cache), args.device
    )

    from tac.capstone_vq_nerv.capstone_trainer import CapstoneTrainConfig, CapstoneTrainer
    from tac.capstone_vq_nerv.vq_nerv_bundle import CapstoneVqNervBundle, CapstoneVqNervConfig
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=n, base_channels=args.base_channels,
            codebook_size=args.codebook_size, carrier=args.carrier, seed=args.seed,
            hinerv_grid_pe=args.hinerv_grid_pe,
            grid_pe_num_freqs=args.grid_pe_num_freqs,
            tie_depth=args.tie_depth,
        )
    )
    bridge = TorchScorerBridge(
        net, seg_t, pose_t,
        seg_loss_form="ce_seg_loss",
        seg_weight=args.seg_weight, pose_weight=args.pose_weight,
        eval_roundtrip=True,
    )
    pose_store = pose_t.float().cpu().numpy()
    cfg = CapstoneTrainConfig(
        epochs=args.epochs, seg_weight=args.seg_weight, pose_weight=args.pose_weight,
        muon_lr=args.muon_lr, grad_clip=args.grad_clip, grad_clip_muon=args.grad_clip_muon,
        eval_every=args.eval_every, seed=args.seed,
        scorer_backend=args.scorer_backend,
        authority_recheck_every=args.authority_recheck_every,
        margin_hinge_weight=args.margin_hinge_weight,
        margin_hinge_floor=args.margin_hinge_floor,
        telemetry=_StreamingTelemetry(out / "trajectory.jsonl"),  # live mid-run RD curve
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_store, cfg)

    d_seg_init = trainer.exact_d_seg()
    d_pose_init = trainer.mean_d_pose()
    print(f"[init] n={n} base_ch={args.base_channels} d_seg={d_seg_init:.5f} "
          f"d_pose={d_pose_init:.5f} curriculum={args.curriculum} "
          f"opt_schedule={args.optimizer_schedule}", flush=True)

    if args.curriculum == "none":
        train_out = trainer.train()
    else:
        from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum

        total = args.curriculum_total_epochs or args.epochs
        stages = build_pr95_8stage_curriculum(total_epochs=total)
        print(f"[curriculum] {args.curriculum}: 8 stages, "
              f"{[s.epochs for s in stages]} epochs (total {sum(s.epochs for s in stages)}), "
              f"opt_schedule={args.optimizer_schedule}", flush=True)

        def _on_stage_done(i, spec, summary):
            print(f"[curriculum] stage {i+1}/8 {spec.name} DONE: "
                  f"d_seg {summary['d_seg_initial']:.5f}->{summary['d_seg_final']:.5f} "
                  f"(best {summary['d_seg_best']:.5f}) muon={summary['use_muon']} "
                  f"qat={spec.use_qat} c1a_lambda={spec.cat_lambda} "
                  f"sigma_noise={spec.sigma_weight_noise}", flush=True)

        cur_result = trainer.run_curriculum(
            stages, optimizer_schedule=args.optimizer_schedule,
            on_stage_done=_on_stage_done,
        )
        train_out = {
            "curriculum": args.curriculum,
            "optimizer_schedule": cur_result.optimizer_schedule,
            "d_seg_initial": cur_result.d_seg_initial,
            "d_seg_final": cur_result.d_seg_final,
            "d_seg_best": cur_result.d_seg_best,
            "stages": cur_result.stages,
        }

    # [A1] live (EMA-shadow) advisory: the shadow d_seg/d_pose, the SAME point the
    # archive bytes (the EMA shadow is exported below).
    d_seg_live = trainer.exact_d_seg()
    d_pose_live = trainer.mean_d_pose()

    archive_zip, account, payload, config = _export_int8_archive(
        trainer, pose_store, args.decoder_dtype
    )
    archive_bytes = len(archive_zip)
    (out / "archive.zip").write_bytes(archive_zip)

    # [A2] RELOAD the int8 archive and re-score the reloaded int8 frames through the
    # SAME bridge. THIS (the int8-quantized archive's d_seg/d_pose) is the honest
    # inflate.sh->evaluate.py predictor; the live number is reported alongside for
    # the quant gap. The advisory score is computed on the RELOADED int8 terms.
    from tac.capstone_vq_nerv.advisory import score_reloaded_int8_archive

    reloaded = score_reloaded_int8_archive(payload, config, bridge)
    d_seg = reloaded.d_seg
    d_pose = reloaded.d_pose

    rate_term = 25.0 * archive_bytes / RATE_DENOM
    seg_term = 100.0 * d_seg
    pose_term = float(np.sqrt(10.0 * d_pose))
    score = seg_term + pose_term + rate_term

    # The live (pre-quant) advisory score, for the gap.
    seg_term_live = 100.0 * d_seg_live
    pose_term_live = float(np.sqrt(10.0 * d_pose_live))
    score_live = seg_term_live + pose_term_live + rate_term

    summary = {
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "n_pairs": n,
        "base_channels": args.base_channels,
        "carrier": args.carrier,
        "decoder_dtype": args.decoder_dtype,
        "epochs": args.epochs,
        "curriculum": args.curriculum,
        "optimizer_schedule": args.optimizer_schedule,
        "cosine_lr_schedule": cfg.cosine_lr_schedule,
        "ema_decay": cfg.ema_decay,
        "use_ema_for_eval": cfg.use_ema_for_eval,
        "muon_lr": args.muon_lr,
        "grad_clip": args.grad_clip,
        "d_seg_init": d_seg_init, "d_pose_init": d_pose_init,
        # [A2] the advisory d_seg/d_pose are the RELOADED int8 terms (the honest
        # contest predictor); the live (pre-quant, EMA-shadow fp32) terms alongside.
        "d_seg_final": d_seg, "d_pose_final": d_pose,
        "d_seg_final_live": d_seg_live, "d_pose_final_live": d_pose_live,
        "reloaded_int8_advisory": reloaded.as_dict(),
        "advisory_quant_gap_d_seg": d_seg - d_seg_live,
        "advisory_quant_gap_score": score - score_live,
        "archive_bytes": archive_bytes,
        "payload_bytes": len(payload),
        "account": account.as_dict(),
        "decoder_bytes": account.decoder_bytes,
        "codebook_bytes": account.codebook_bytes,
        "index_bytes": account.index_bytes,
        "latent_bytes": account.latent_bytes,
        "score_seg_contribution": seg_term,
        "score_pose_contribution": pose_term,
        "score_rate_contribution": rate_term,
        "advisory_score": score,
        "advisory_score_live_prequant": score_live,
        "sub_0_15": score < 0.15,
        "sub_0_19": score < 0.19,
        "wall_s": time.time() - t_start,
        "traj": train_out.get("traj") if isinstance(train_out, dict) else None,
        "curriculum_stages": (
            train_out.get("stages") if isinstance(train_out, dict) else None
        ),
    }
    (out / "capstone_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nADVISORY S = {score:.5f}  (reloaded int8: seg {seg_term:.4f} + "
          f"pose {pose_term:.4f} + rate {rate_term:.4f})  "
          f"[macOS-CPU advisory, NOT a pointer move]", flush=True)
    print(f"  live (pre-quant EMA-shadow) S = {score_live:.5f}  "
          f"quant gap dS = {score - score_live:+.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
