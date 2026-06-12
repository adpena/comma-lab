# SPDX-License-Identifier: MIT
"""TRACK B step-3 — the REAL Cool-Chic carrier-thesis test: does the JOINT
latent+synthesis+AR-prior loss WITH THE RATE LEVER ACTIVE make the latents
compressible enough to put Cool-Chic's rate floor below HNeRV's ~161 KB?

Why this probe exists (the open question step-2 left):
    Step-1 built a REAL lossless AR-Gaussian entropy coder (``encode_latent_chain``)
    and proved Cool-Chic is a real rate carrier (57.5% vs raw int16) but sits ~44x
    above HNeRV's decoder-weight floor on the 6-epoch smoke. Step-2 (NO-GO,
    ``aaace2a46``) proved training the AR prior on FIXED latents saturates at ~6
    epochs (sigma/std ~= 0.75, flat) -> the prior-training lever is exhausted, and
    explicitly named the UNTESTED high-value lever:

        "Re-optimize the LATENTS for compressibility (the full-campaign lever this
         smoke deliberately excluded). The joint loss's AR rate term pushes the
         latents themselves to be more predictable (lower conditional entropy),
         which is a fundamentally different and potentially stronger lever than
         fitting a prior to fixed latents. This is the single highest-value next
         probe — but it is a FULL-model training question, and it must be MEASURED,
         not assumed."

    This probe measures EXACTLY that.

What it does (REAL, no fake):
    1. Warm-starts from the step-1 trained checkpoint (appearance already learned,
       latents std ~0.07). A warm start means a BOUNDED run can actually move the
       rate-distortion point (a cold init at 6 epochs would conflate "didn't learn
       appearance" with "couldn't compress").
    2. Builds a JOINT score-aware loss with the RATE LEVER ACTIVE:
         L = beta*d_seg + gamma*sqrt(d_pose) + lambda_rate * [25 * ar_bits/8 / N]
       where ``ar_bits`` is ``ar_gaussian_predicted_bits`` (the step-1 Layer-1<->2
       unifier — the EXACT differentiable quantity the real coder emits, confirmed
       in step-2 to track real coded bytes within ~0.09 bit/elem). The
       ``25*B/8/N`` factor makes lambda_rate=1.0 the CONTEST-FAITHFUL rate weight;
       higher lambda pushes compressibility harder to trace the RD curve.
    3. JOINTLY updates latents + synthesis + AR prior (NOT prior-only — that was
       step-2). This is the lever step-2 excluded.
    4. Sweeps a few lambda_rate values (the rate-distortion tradeoff).
    5. Traces, across training + at the end: the REAL coded latent bytes via
       ``encode_latent_chain`` (lossless), the advisory d_seg/d_pose, and the
       differentiable ar_bits/elem — so the RD point's trajectory is MEASURED.

Device discipline (CLAUDE.md "MPS is a VALID TRAINING-GRADIENT device — NEVER a
score authority" + "MPS auth eval is NOISE"):
    * TRAIN/GRADIENT device = ``mps`` (the Apple GPU; ~0.35 s/step measured under
      basin contention -> ~26 s/epoch viable). ``patch_scorer_for_mps()`` applied.
    * AUTHORITY device = ``cpu`` for the d_seg/d_pose that are reported. EVERY score
      reported is ``[contest-CPU advisory] NON-PROMOTABLE`` (no contest eval, no PR).
    * The basin daemon (pid 33911) owns the MPS GPU; this probe interleaves (MPS GPU
      kernels time-share); keep n_pairs/epochs bounded + note the contention.

Reuse targets (SEARCH-FIRST):
    * ``ar_gaussian_predicted_bits`` / ``encode_latent_chain`` / ``LatentGrid`` /
      ``choose_grid_step`` — the step-1 entropy coder (the rate measurement surface).
    * ``score_pair_components_dispatch`` — the canonical real-scorer d_seg/d_pose.
    * ``decode_real_pairs`` — the canonical real-0.mkv GT loader.
    * ``apply_eval_roundtrip_during_training`` + ``patch_upstream_yuv6_globally`` —
      the differentiable eval-roundtrip + PoseNet-grad-reachable yuv6 (NON-NEGOTIABLE).
    * ``EMA`` — the canonical weight EMA (NON-NEGOTIABLE).
    NOT reimplemented: no new coder, no new scorer, no new GT loader.

Honest framing: this is a $0 local MEANS-feasibility probe. The frontier
(``.omx/state/canonical_frontier_pointer.json``) is UNMOVED. The END is a lower
exact score; this probe resolves whether the Cool-Chic continuous-latent carrier
CAN reach a competitive floor (GO / NEEDS-DESIGN / NO-GO).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_WARMSTART = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "cool_chic_mps_smoke_20260612T121501Z"
    / "best.pt"
)
CONTEST_NORMALIZER = 37_545_489.0
HNERV_DECODER_FLOOR_BYTES = 161_000  # ~91% of HNeRV's ~177 KB frontier archive

# Contest score weights (CLAUDE.md). gamma = sqrt(10).
BETA_SEG = 100.0
GAMMA_POSE = math.sqrt(10.0)
ALPHA_RATE = 25.0


# --------------------------------------------------------------------------- #
# Rate lever (the active, contest-faithful, differentiable rate term)         #
# --------------------------------------------------------------------------- #


def _ar_predicted_bits_total(model) -> torch.Tensor:
    """Total differentiable AR-coded bits across BOTH latent axes at a grid that
    tracks the latent std (so the bits == what ``encode_latent_chain`` emits).

    Uses the step-1 ``ar_gaussian_predicted_bits`` (the Layer-1<->2 unifier). The
    grid step is recomputed from the CURRENT latents each call (the latents move,
    so their std moves; ``bits_per_std=5`` is the same 32-levels/std the real coder
    measurement used). Differentiable in (latents, mean, log_scale).
    """
    from tac.substrates.cool_chic.architecture import _ARPriorNet  # noqa: F401
    from tac.substrates.cool_chic.entropy_coder import (
        LatentGrid,
        ar_gaussian_predicted_bits,
        choose_grid_step,
    )

    total = torch.zeros((), device=model.latents_coarse.device)
    for latents, prior_net in (
        (model.latents_coarse, model.ar_prior_coarse),
        (model.latents_fine, model.ar_prior_fine),
    ):
        # Replay the AR causal chain (mirror of architecture._ar_log_prob_chain):
        # prev = [zero, latents[0..N-2]].
        num_pairs = latents.shape[0]
        prev = torch.zeros_like(latents[0:1])
        prev_stack = torch.cat([prev, latents[: num_pairs - 1]], dim=0)
        mean, log_scale = prior_net(prev_stack)
        step = choose_grid_step(latents, bits_per_std=5.0)
        grid = LatentGrid(step=step)
        total = total + ar_gaussian_predicted_bits(latents, mean, log_scale, grid)
    return total


def _rate_score_term(ar_bits_total: torch.Tensor) -> torch.Tensor:
    """Contest-faithful rate term in SCORE units: 25 * (bits/8) / N.

    With lambda_rate=1.0 this is EXACTLY the contest archive-rate weight applied to
    the AR-coded latent bytes (the dominant Cool-Chic byte sink). Differentiable.
    """
    bytes_total = ar_bits_total / 8.0
    return ALPHA_RATE * bytes_total / CONTEST_NORMALIZER


# --------------------------------------------------------------------------- #
# Real coded-byte measurement (the lossless ground truth, periodic)           #
# --------------------------------------------------------------------------- #


def _real_coded_latent_bytes(model, *, window_pairs: int) -> dict:
    """Measure REAL coded latent bytes via ``encode_latent_chain`` (lossless) on a
    representative leading window of pairs. The pure-Python coder is slow, so we
    code a window; step-2 confirmed the per-element rate is window-stationary so the
    window faithfully tracks the full-archive trend. Returns coarse+fine bytes +
    the full-600-pair extrapolation + bits/elem.
    """
    from tac.substrates.cool_chic.entropy_coder import (
        LatentGrid,
        choose_grid_step,
        encode_latent_chain,
    )

    out = {}
    total_full_bytes = 0
    for name, latents, prior_net in (
        ("coarse", model.latents_coarse, model.ar_prior_coarse),
        ("fine", model.latents_fine, model.ar_prior_fine),
    ):
        full_pairs = latents.shape[0]
        w = min(window_pairs, full_pairs)
        z_win = latents[:w].detach().to("cpu", torch.float32)
        step = choose_grid_step(latents, bits_per_std=5.0)
        grid = LatentGrid(step=step)
        prior_cpu = prior_net.to("cpu")
        blob = encode_latent_chain(z_win, prior_cpu, grid)
        prior_net.to(latents.device)  # restore
        elems_win = int(z_win.numel())
        bits_per_elem = (len(blob) * 8.0) / max(1, elems_win)
        # extrapolate to full 600 pairs (window-stationary per-elem rate)
        elems_full = int(latents.numel())
        full_bytes = bits_per_elem * elems_full / 8.0
        total_full_bytes += full_bytes
        out[name] = {
            "window_pairs": w,
            "window_bytes": len(blob),
            "bits_per_elem": round(bits_per_elem, 5),
            "grid_step": step,
            "full_bytes_extrapolated": int(round(full_bytes)),
        }
    out["total_latent_bytes_full"] = int(round(total_full_bytes))
    out["rate_score_full"] = round(ALPHA_RATE * total_full_bytes / CONTEST_NORMALIZER, 5)
    out["vs_hnerv_floor_x"] = round(total_full_bytes / HNERV_DECODER_FLOOR_BYTES, 2)
    return out


# --------------------------------------------------------------------------- #
# d_seg / d_pose advisory measurement on the CPU authority (NON-PROMOTABLE)    #
# --------------------------------------------------------------------------- #


def _advisory_distortion(
    model, pair_tensor, *, segnet_auth, posenet_auth, authority_device, val_idx, batch
) -> dict:
    """Advisory d_seg / d_pose on the CPU authority over a validation slice."""
    from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training
    from tac.substrates.score_aware_common import score_pair_components_dispatch

    model_auth = model  # caller passes a CPU-authority EMA copy
    seg_sum = pose_sum = 0.0
    n = 0
    with torch.no_grad():
        for start in range(0, val_idx.shape[0], batch):
            idx = val_idx[start : start + batch]
            if idx.numel() == 0:
                continue
            rgb_0, rgb_1 = model_auth(idx)
            r0 = apply_eval_roundtrip_during_training(rgb_0 * 255.0)
            r1 = apply_eval_roundtrip_during_training(rgb_1 * 255.0)
            gt = pair_tensor[idx]
            seg_term, pose_term = score_pair_components_dispatch(
                seg_scorer=segnet_auth,
                pose_scorer=posenet_auth,
                rgb_0_rt=r0,
                rgb_1_rt=r1,
                gt_rgb_0=gt[:, 0],
                gt_rgb_1=gt[:, 1],
            )
            bs = int(idx.numel())
            seg_sum += float(seg_term) * bs
            pose_sum += float(pose_term) * bs
            n += bs
    d_seg = seg_sum / max(1, n)
    d_pose = pose_sum / max(1, n)
    return {
        "d_seg": round(d_seg, 6),
        "d_pose": round(d_pose, 6),
        "score_seg_contribution": round(BETA_SEG * d_seg, 5),
        "score_pose_contribution": round(GAMMA_POSE * math.sqrt(max(d_pose, 0.0)), 5),
    }


# --------------------------------------------------------------------------- #
# One training arm (one lambda_rate value)                                    #
# --------------------------------------------------------------------------- #


@dataclass
class ArmResult:
    lambda_rate: float
    trajectory: list
    final_real_bytes: dict
    final_distortion: dict
    sec_per_epoch: float


def _run_arm(
    *,
    lambda_rate: float,
    warmstart_ck,
    pair_tensor_train,
    pair_tensor_auth,
    segnet,
    posenet,
    segnet_auth,
    posenet_auth,
    train_device,
    authority_device,
    epochs: int,
    batch: int,
    lr: float,
    grad_clip: float,
    ema_decay: float,
    val_idx_auth,
    coded_window_pairs: int,
    measure_every: int,
    seed: int,
    progress_jsonl: Path | None = None,
) -> ArmResult:
    from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training
    from tac.substrates.cool_chic.architecture import (
        CoolChicConfig,
        CoolChicSubstrate,
    )
    from tac.substrates.score_aware_common import score_pair_components_dispatch
    from tac.training import EMA

    torch.manual_seed(seed)

    cfg = CoolChicConfig(**warmstart_ck["config"])
    model = CoolChicSubstrate(cfg)
    model.load_state_dict(warmstart_ck["state_dict"])
    model = model.to(train_device)

    ema = EMA(model, decay=ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    n_pairs = pair_tensor_train.shape[0]
    train_idx = torch.arange(n_pairs, device=train_device, dtype=torch.long)

    trajectory = []

    def _append_progress(row: dict) -> None:
        """Durable, orphan-safe per-measure checkpoint (JSONL append).

        Per CLAUDE.md "LONG RESUMABLE SATURATION SWEEPS": each measured row is
        flushed to disk the moment it is produced so an orphaned/killed run still
        leaves its signal. The killed prior step-3 run had NO such checkpoint and
        lost 5h of (contended) compute — this closes that.
        """
        if progress_jsonl is None:
            return
        with progress_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"lambda_rate": lambda_rate, **row}) + "\n")
            fh.flush()

    def _measure(epoch_label) -> dict:
        # Snapshot EMA shadow onto a CPU-authority copy for the advisory numbers
        # (NEVER MPS as authority).
        ema_cpu = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
        auth_model = CoolChicSubstrate(cfg).to(authority_device)
        auth_model.load_state_dict({k: v.to(authority_device) for k, v in ema_cpu.items()})
        auth_model.eval()
        dist = _advisory_distortion(
            auth_model,
            pair_tensor_auth,
            segnet_auth=segnet_auth,
            posenet_auth=posenet_auth,
            authority_device=authority_device,
            val_idx=val_idx_auth,
            batch=batch,
        )
        coded = _real_coded_latent_bytes(auth_model, window_pairs=coded_window_pairs)
        # differentiable ar bits on the EMA shadow (full ensemble), for cross-check
        with torch.no_grad():
            ar_bits = _ar_predicted_bits_total(auth_model)
            ar_bits_per_elem = float(ar_bits) / max(1, int(auth_model.latents_coarse.numel() + auth_model.latents_fine.numel()))
        advisory_S = (
            dist["score_seg_contribution"]
            + dist["score_pose_contribution"]
            + coded["rate_score_full"]
        )
        row = {
            "epoch": epoch_label,
            "advisory_S": round(advisory_S, 5),
            "ar_bits_per_elem_predicted": round(ar_bits_per_elem, 5),
            **dist,
            "coded": coded,
        }
        return row

    # baseline (warm-start, before any rate-lever steps)
    _base_row = {"phase": "warmstart_baseline", **_measure(0)}
    trajectory.append(_base_row)
    _append_progress(_base_row)
    _b = _base_row
    print(
        f"  [lambda={lambda_rate:g} BASELINE] S={_b['advisory_S']:.4f} "
        f"d_seg={_b['d_seg']:.4f} d_pose={_b['d_pose']:.4f} "
        f"latent_bytes={_b['coded']['total_latent_bytes_full']:,} "
        f"({_b['coded']['vs_hnerv_floor_x']:.1f}x HNeRV)",
        flush=True,
    )

    t_arm = time.time()
    epochs_done = 0
    for epoch in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(n_pairs, device=train_device)]
        for start in range(0, n_pairs, batch):
            idx = perm[start : start + batch]
            if idx.numel() == 0:
                continue
            rgb_0, rgb_1 = model(idx)
            r0 = apply_eval_roundtrip_during_training(rgb_0 * 255.0)
            r1 = apply_eval_roundtrip_during_training(rgb_1 * 255.0)
            gt = pair_tensor_train[idx]
            seg_term, pose_term = score_pair_components_dispatch(
                seg_scorer=segnet,
                pose_scorer=posenet,
                rgb_0_rt=r0,
                rgb_1_rt=r1,
                gt_rgb_0=gt[:, 0],
                gt_rgb_1=gt[:, 1],
            )
            ar_bits = _ar_predicted_bits_total(model)
            rate_term = _rate_score_term(ar_bits)
            loss = (
                BETA_SEG * seg_term
                + GAMMA_POSE * torch.sqrt(pose_term.clamp(min=1e-12))
                + lambda_rate * rate_term
            )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            ema.update(model)
        scheduler.step()
        epochs_done += 1
        if (epoch + 1) % measure_every == 0 or epoch == epochs - 1:
            _row = {"phase": "train", **_measure(epoch + 1)}
            trajectory.append(_row)
            _append_progress(_row)
            last = trajectory[-1]
            print(
                f"  [lambda={lambda_rate:g} ep {epoch + 1}/{epochs}] "
                f"S={last['advisory_S']:.4f} d_seg={last['d_seg']:.4f} "
                f"d_pose={last['d_pose']:.4f} "
                f"latent_bytes={last['coded']['total_latent_bytes_full']:,} "
                f"({last['coded']['vs_hnerv_floor_x']:.1f}x HNeRV)",
                flush=True,
            )
    sec_per_epoch = (time.time() - t_arm) / max(1, epochs_done)

    final = trajectory[-1]
    return ArmResult(
        lambda_rate=lambda_rate,
        trajectory=trajectory,
        final_real_bytes=final["coded"],
        final_distortion={k: final[k] for k in ("d_seg", "d_pose", "advisory_S")},
        sec_per_epoch=sec_per_epoch,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--warmstart", type=Path, default=DEFAULT_WARMSTART)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument(
        "--lambda-rate",
        type=float,
        nargs="+",
        default=[1.0, 8.0],
        help="rate-lever weights to sweep (1.0 = contest-faithful; higher pushes "
        "compressibility harder to trace the RD curve).",
    )
    p.add_argument("--val-pairs", type=int, default=48,
                   help="advisory d_seg/d_pose validation slice size (CPU authority).")
    p.add_argument("--coded-window-pairs", type=int, default=24,
                   help="leading-window pairs for the REAL encode_latent_chain "
                        "measurement (pure-Python coder is slow; window-stationary).")
    p.add_argument("--measure-every", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-device", choices=["mps", "cpu"], default="mps")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Patch the differentiable yuv6 globally BEFORE scorers load (PoseNet grad reach).
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    patch_upstream_yuv6_globally()

    train_device = torch.device(args.train_device)
    if args.train_device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available")
        from tac.torch_mps_compat import patch_scorer_for_mps
        mps_report = patch_scorer_for_mps()
        print(f"[mps] patch_scorer_for_mps report: {mps_report}")
    authority_device = torch.device("cpu")

    from tac.scorer import load_differentiable_scorers
    from tac.substrates._shared.trainer_skeleton import decode_real_pairs

    # GT pairs: real 0.mkv frames (NO synthetic data).
    print(f"[gt] decoding {args.n_pairs} real pairs from {args.video} ...", flush=True)
    t0 = time.time()
    pair_tensor = decode_real_pairs(
        args.video,
        n_pairs=args.n_pairs,
        substrate_tag="cool_chic_joint_latent_compress",
        repo_root=REPO_ROOT,
    )
    print(f"[gt] decoded {pair_tensor.shape} in {time.time() - t0:.1f}s", flush=True)
    pair_tensor_train = pair_tensor.to(train_device)
    pair_tensor_auth = pair_tensor.to(authority_device)

    # Scorers: train-device (gradient) + authority-device (advisory d_seg/d_pose).
    print("[scorers] loading train-device + authority-device scorers ...", flush=True)
    posenet, segnet = load_differentiable_scorers(str(args.upstream), device=train_device)
    if train_device == authority_device:
        posenet_auth, segnet_auth = posenet, segnet
    else:
        posenet_auth, segnet_auth = load_differentiable_scorers(
            str(args.upstream), device=authority_device
        )

    warmstart_ck = torch.load(args.warmstart, map_location="cpu", weights_only=False)
    print(f"[warmstart] {args.warmstart} config={warmstart_ck['config']}", flush=True)

    # Validation slice for advisory distortion (fixed across arms, CPU authority).
    val_idx_auth = torch.arange(
        min(args.val_pairs, args.n_pairs), device=authority_device, dtype=torch.long
    )

    arms = []
    for lam in args.lambda_rate:
        print(f"\n=== ARM lambda_rate={lam:g} ===", flush=True)
        arm = _run_arm(
            lambda_rate=lam,
            warmstart_ck=warmstart_ck,
            pair_tensor_train=pair_tensor_train,
            pair_tensor_auth=pair_tensor_auth,
            segnet=segnet,
            posenet=posenet,
            segnet_auth=segnet_auth,
            posenet_auth=posenet_auth,
            train_device=train_device,
            authority_device=authority_device,
            epochs=args.epochs,
            batch=args.batch_size,
            lr=args.lr,
            grad_clip=args.grad_clip,
            ema_decay=args.ema_decay,
            val_idx_auth=val_idx_auth,
            coded_window_pairs=args.coded_window_pairs,
            measure_every=args.measure_every,
            seed=args.seed,
            progress_jsonl=args.out_dir / "progress.jsonl",
        )
        arms.append(arm)
        # Durable partial report after EACH arm (orphan-safe).
        (args.out_dir / "partial_report.json").write_text(
            json.dumps(
                {
                    "probe": "cool_chic_joint_latent_compressibility",
                    "axis": "[contest-CPU advisory] NON-PROMOTABLE",
                    "arms_done": len(arms),
                    "last_arm": {
                        "lambda_rate": arm.lambda_rate,
                        "sec_per_epoch": round(arm.sec_per_epoch, 2),
                        "baseline": arm.trajectory[0],
                        "final_distortion": arm.final_distortion,
                        "final_real_bytes": arm.final_real_bytes,
                    },
                },
                indent=2,
            )
        )

    # Provenance + report
    import subprocess

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        git_head = "unknown"

    report = {
        "probe": "cool_chic_joint_latent_compressibility",
        "axis": "[contest-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "git_head": git_head,
        "warmstart": str(args.warmstart),
        "warmstart_config": warmstart_ck["config"],
        "n_pairs": args.n_pairs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_device": str(train_device),
        "authority_device": str(authority_device),
        "hnerv_decoder_floor_bytes": HNERV_DECODER_FLOOR_BYTES,
        "contest_normalizer": CONTEST_NORMALIZER,
        "val_pairs": int(val_idx_auth.numel()),
        "coded_window_pairs": args.coded_window_pairs,
        "arms": [
            {
                "lambda_rate": a.lambda_rate,
                "sec_per_epoch": round(a.sec_per_epoch, 2),
                "baseline": a.trajectory[0],
                "final_distortion": a.final_distortion,
                "final_real_bytes": a.final_real_bytes,
                "trajectory": a.trajectory,
            }
            for a in arms
        ],
    }
    out_json = args.out_dir / "joint_latent_compressibility.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\n[report] wrote {out_json}", flush=True)

    # Headline summary
    print("\n=== HEADLINE (RD point per arm; vs HNeRV ~161 KB floor) ===")
    base = arms[0].trajectory[0]
    print(
        f"  warmstart baseline: d_seg={base['d_seg']:.4f} d_pose={base['d_pose']:.4f} "
        f"latent_bytes={base['coded']['total_latent_bytes_full']:,} "
        f"({base['coded']['vs_hnerv_floor_x']:.1f}x HNeRV) S={base['advisory_S']:.4f}"
    )
    for a in arms:
        f = a.trajectory[-1]
        print(
            f"  lambda={a.lambda_rate:g}: d_seg={f['d_seg']:.4f} d_pose={f['d_pose']:.4f} "
            f"latent_bytes={f['coded']['total_latent_bytes_full']:,} "
            f"({f['coded']['vs_hnerv_floor_x']:.1f}x HNeRV) S={f['advisory_S']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
