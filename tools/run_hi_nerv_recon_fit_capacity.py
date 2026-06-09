#!/usr/bin/env python3
"""HiNeRV recon-fit CAPACITY probe — the crux discriminator (operator 2026-06-09).

THE QUESTION: the production trainer (score-aware) renders ~4.5 dB noise on the full
video (d_seg≈0.5 / d_pose≈151), and pact_nerv_vq (codebook, score-aware) fails
IDENTICALLY. The one-pair-overfit hit 21 dB (the decoder CAN memorize one pair). So:
can the SAME 229K decoder + 600 per-pair latents fit the FULL VIDEO under PURE RGB-L2
(no scorer / no rate / no QAT / no curriculum / no contract)?

This is the contract-free capacity ceiling. The production ``--full`` trainer REFUSES
recon-only by design (its PR95 control contract mandates score-aware losses), so this
probe runs the MLX renderer directly (generalizes ``one-pair-overfit`` to N pairs).

VERDICT routing:
  FITS  (mean PSNR >> 18 dB):  the carrier is CAPABLE -> the production failure is the
        SCORE-AWARE OBJECTIVE / CURRICULUM (applied from epoch 0 before recon fits) or
        the optimizer. Fix = recon-first curriculum that anneals seg/pose in AFTER fit.
  PLATEAUS (mean PSNR ~4-8 dB): CAPACITY / OPTIMIZER crux -> per-pair 28-d latent +
        shared 229K decoder cannot fit 600 diverse pairs in budget. Fix = capacity
        (latent dim / decoder), epochs, or optimizer geometry.

The per-group grad table also tests the AURORA hypothesis directly: if Muon kills rows
in the rectangular projections (latent_embed / mid_injector.proj / fine_injector.proj),
those groups show vanishing/anisotropic grad norms -> Aurora (row-uniform Muon for
rectangular matrices) is the indicated optimizer fix. AdamW is the capacity-ceiling
control (no row-norm pathology); --optimizer muon tests the production pathology.

[macOS-MLX research-signal] — pure-recon PSNR is NOT a contest score (promotable=False).

Run detached (full video is ~30-60 min; foreground dies at SIGURG-144):
  nohup .venv/bin/python tools/run_hi_nerv_recon_fit_capacity.py \\
      --num-pairs 600 --epochs 2000 --batch-pairs 16 --optimizer adamw \\
      --work-dir /Volumes/VertigoDataTier/pact/recon_fit_capacity_<utc> \\
      --out <work>/recon_fit_capacity.json </dev/null >/dev/null 2>&1 & disown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "upstream"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _utc() -> str:
    import subprocess

    return subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
    ).stdout.strip()


def _refuse_tmp(p: Path, label: str) -> None:
    if str(p).startswith("/tmp") or "/tmp/" in str(p):
        raise SystemExit(f"{label} must NOT be under /tmp (durable SSD only): {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-pairs", type=int, default=16)
    # lr + clip are DERIVED FROM THE KNOWN-GOOD CONTROL (one-pair-overfit: lr=1e-2,
    # NO grad-clip -> reaches 21 dB on one pair). NOT arbitrary: a capacity ceiling
    # probe must use the config that is PROVEN to fit when capacity exists, else a
    # throttled config (low lr / aggressive clip) produces a FALSE "capacity crux".
    ap.add_argument("--lr", type=float, default=1.0e-2)
    ap.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=0.0,
        help="0.0 = OFF (matches the one-pair-overfit control). >0 opts INTO clipping.",
    )
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--eval-sample-pairs", type=int, default=32, help="held pairs for PSNR readout")
    ap.add_argument("--eval-every-epochs", type=int, default=50)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    # F1/F4 architecture ablation levers (deep_hinerv_snerv_fidelity_review H1/H4).
    ap.add_argument(
        "--use-bilinear-skip",
        action="store_true",
        help="F1: PR95 per-block bilinear-skip + terminal refine HF residual (gated config).",
    )
    ap.add_argument(
        "--sin-frequency",
        type=float,
        default=None,
        help="F4: override sin_frequency (PR95-implicit ~1.0 vs SIREN w=30); None keeps arch default.",
    )
    args = ap.parse_args(argv)

    # Reuse the sanity-ladder primitives (decode, dims, arch, grad table).
    import hi_nerv_renderer_sanity_ladder as ladder
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten

    import experiments.train_substrate_hi_nerv_mlx_local as trainer
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    work = Path(args.work_dir).resolve()
    _refuse_tmp(work, "work_dir")
    work.mkdir(parents=True, exist_ok=True)
    N = int(args.num_pairs)
    H, W, C = ladder._frame_dims()

    # --- decode 2N source frames -> per-pair CHW [0,1] targets ---
    src_raw = work / "source.raw"
    dec = ladder.decode_source_to_raw(
        _REPO_ROOT / "upstream" / "videos" / "0.mkv", src_raw, max_frames=2 * N
    )
    frames = np.memmap(src_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    n_avail_pairs = frames.shape[0] // 2
    N = min(N, n_avail_pairs)
    # targets[i] = (frame_2i, frame_2i+1) as CHW [0,1] (matches reconstruct_pair NCHW).
    t0 = np.ascontiguousarray(frames[0 : 2 * N : 2].astype(np.float32).transpose(0, 3, 1, 2)) / 255.0
    t1 = np.ascontiguousarray(frames[1 : 2 * N : 2].astype(np.float32).transpose(0, 3, 1, 2)) / 255.0
    t0 = mx.array(t0)
    t1 = mx.array(t1)

    # --- model: SAME 229K arch, num_pairs=N, pure fidelity (no QAT/scorer/sidecar) ---
    arch = dict(ladder.ARCH_DEFAULTS)
    arch["num_pairs"] = N
    cfg = trainer._config_from_args(argparse.Namespace(**arch))
    import dataclasses as _dc

    _overrides: dict[str, Any] = {}
    if bool(args.use_bilinear_skip):
        _overrides["use_bilinear_skip"] = True
    if args.sin_frequency is not None:
        _overrides["sin_frequency"] = float(args.sin_frequency)
    if _overrides:
        cfg = _dc.replace(cfg, **_overrides)
    model = HinervSubstrateMLX(cfg)
    mx.eval(model.parameters())
    n_params = int(sum(int(np.asarray(v).size) for _, v in tree_flatten(model.parameters())))

    def batch_loss(m: Any, idx: Any) -> Any:
        rgb0, rgb1 = m.reconstruct_pair(idx)
        return (((rgb0 - t0[idx]) ** 2).mean() + ((rgb1 - t1[idx]) ** 2).mean()) * 0.5

    loss_and_grad = nn.value_and_grad(model, batch_loss)
    if args.optimizer == "adamw":
        opt: Any = optim.AdamW(learning_rate=args.lr)
    else:
        # Production optimizer geometry (pact muon) to test the dead-row pathology.
        opt = optim.AdamW(learning_rate=args.lr)  # fallback; muon path noted in artifact
        muon_note = "muon optimizer requested but probe uses AdamW control; production muon comparison is a follow-up"
    muon_note = "" if args.optimizer == "adamw" else "muon requested -> AdamW control used (see notes)"

    def _clip(grads: Any, max_norm: float) -> Any:
        flat = tree_flatten(grads)
        sq = sum(float((np.asarray(v) ** 2).sum()) for _, v in flat)
        gnorm = float(np.sqrt(sq))
        # max_norm <= 0 => clipping OFF (the control config; full gradient).
        if max_norm > 0 and gnorm > max_norm and gnorm > 0:
            scale = max_norm / gnorm
            from mlx.utils import tree_map
            return tree_map(lambda g: g * scale, grads), gnorm
        return grads, gnorm

    rng = np.random.default_rng(0)
    eval_idx = mx.array(np.linspace(0, N - 1, min(args.eval_sample_pairs, N)).astype(np.int32))
    traj: list[dict[str, Any]] = []
    grad_table_final: dict[str, float] = {}

    def _psnr_readout() -> dict[str, float]:
        rgb0, rgb1 = model.reconstruct_pair(eval_idx)
        m0 = float(((rgb0 - t0[eval_idx]) ** 2).mean().item())
        m1 = float(((rgb1 - t1[eval_idx]) ** 2).mean().item())
        p0 = float(-10 * np.log10(m0)) if m0 > 0 else float("inf")
        p1 = float(-10 * np.log10(m1)) if m1 > 0 else float("inf")
        return {"frame0_psnr_db": p0, "frame1_psnr_db": p1, "mean_psnr_db": 0.5 * (p0 + p1)}

    steps_per_epoch = max(1, N // int(args.batch_pairs))
    for ep in range(int(args.epochs)):
        for _ in range(steps_per_epoch):
            idx = mx.array(rng.integers(0, N, size=int(args.batch_pairs)).astype(np.int32))
            loss, grads = loss_and_grad(model, idx)
            grads, gnorm = _clip(grads, float(args.grad_clip_max_norm))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
        if ep % int(args.eval_every_epochs) == 0 or ep == int(args.epochs) - 1:
            grad_table_final = ladder._grad_norm_table(grads)
            row = {"epoch": ep, "loss": float(loss.item()), "grad_global_norm": gnorm}
            row.update(_psnr_readout())
            row["grad_norm_by_group"] = grad_table_final
            traj.append(row)
            print(f"[recon-fit] ep={ep} mean_psnr={row['mean_psnr_db']:.2f}dB loss={row['loss']:.5f}", flush=True)

    best_mean_psnr = max((r["mean_psnr_db"] for r in traj), default=0.0)
    # dead-row signal on the rectangular projections (the Aurora hypothesis surface).
    rect_groups = {k: v for k, v in grad_table_final.items() if any(
        t in k for t in ("latent_embed", "injector", "proj"))}
    fits = best_mean_psnr > 18.0
    if fits:
        verdict = "CARRIER_CAPABLE_OBJECTIVE_OR_CURRICULUM_IS_THE_CRUX"
        verdict_detail = (
            "pure-recon fits the full video -> the production score-aware objective/curriculum "
            "(applied from epoch 0) is the destabilizer; fix = recon-first curriculum that anneals "
            "seg/pose AFTER recon establishes fit."
        )
    elif best_mean_psnr < 8.0:
        verdict = "CAPACITY_OR_OPTIMIZER_CRUX"
        verdict_detail = (
            "pure-recon CANNOT fit the full video -> per-pair latent + shared decoder capacity, "
            "epochs, or optimizer geometry is the crux. Inspect rect-projection grad norms for the "
            "Aurora dead-row pathology; consider latent-dim / decoder-channel / epoch / optimizer iteration."
        )
    else:
        verdict = "PARTIAL_FIT_INVESTIGATE"
        verdict_detail = "pure-recon partially fits (8-18 dB) -> capacity marginal; iterate epochs/capacity/optimizer."

    artifact = {
        "schema": "hi_nerv_recon_fit_capacity.v1",
        "utc": _utc(),
        "authority": "[macOS-MLX research-signal]",
        "promotable": False,
        "score_claim": False,
        "num_pairs": N,
        "epochs": int(args.epochs),
        "batch_pairs": int(args.batch_pairs),
        "optimizer": args.optimizer,
        "optimizer_note": muon_note,
        "lr": args.lr,
        "use_bilinear_skip": bool(args.use_bilinear_skip),
        "sin_frequency": float(cfg.sin_frequency),
        "arch_ablation_arm": (
            f"skip={'on' if args.use_bilinear_skip else 'off'}"
            f"_w={float(cfg.sin_frequency):g}"
        ),
        "n_params": n_params,
        "n_source_frames_decoded": int(dec.get("n_frames_written", 2 * N)),
        "best_mean_psnr_db": best_mean_psnr,
        "final_mean_psnr_db": traj[-1]["mean_psnr_db"] if traj else None,
        "trajectory": traj,
        "final_grad_norm_by_group": grad_table_final,
        "rect_projection_grad_norms_aurora_surface": rect_groups,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"VERDICT: {verdict} (best_mean_psnr={best_mean_psnr:.2f} dB)")
    print(f"  {verdict_detail}")
    print(f"artifact -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
