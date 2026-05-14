# SPDX-License-Identifier: MIT
"""Probe: world-model recurrent vs independent-frame decoders for C1.

Per Catalog #125 hook #6 + the design-tension memo
``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md``:
when a design choice has 2+ defensible interpretations, ship BOTH modes via
callable interface + build a probe that returns the regime-conditional
verdict. The probe IS the arbitration; the trainer/codec/solver consumes
the verdict.

For C1 the world-model design tension is:

1. **World-model recurrent (GRU/LSTM/Transformer)**: z_t = WM(z_{t-1}).
   Exploits inter-frame redundancy; predicted asymptote <100 B/frame of
   residual surprise for stationary-ergodic driving.
2. **Independent per-frame decoder** (HNeRV-class baseline): z_t = idx_to_z(t).
   Each frame decoded from a position-only embedding; no temporal coupling.
   The Z1 MDL ablation showed this class saturates at density 97-99% on
   A1/PR101 archives.

The probe fits BOTH decoder types on a tiny synthetic sequence (smoke; no
contest video required), measures the per-frame residual entropy after
the decoder, and emits a typed verdict the C1 trainer consumes.

Verdict schema (JSON output)::

    {
      "world_model_gru": {
        "residual_l2": float,
        "params_bytes_est_fp4": int,
        "fit_wallclock_sec": float
      },
      "world_model_lstm": {
        "residual_l2": float,
        "params_bytes_est_fp4": int,
        "fit_wallclock_sec": float
      },
      "independent_frame_baseline": {
        "residual_l2": float,
        "params_bytes_est_fp4": int,
        "fit_wallclock_sec": float
      },
      "verdict": "world_model_gru" | "world_model_lstm" | "independent_frame_baseline" | "tie",
      "verdict_rationale": str,
      "evidence_grade": "proxy",
      "score_claim_valid": false,
      "score_axis": "proxy_synthetic",
      "ready_for_exact_eval_dispatch": false,
      "promotion_eligible": false,
      "rank_or_kill_eligible": false,
      "result_review_blockers": [
        "smoke_proxy_synthetic_not_contest_video",
        "no_scorer_load",
        "non_promotable_evidence_grade"
      ]
    }

Usage::

    .venv/bin/python tools/probe_c1_world_model_vs_independent_frames_disambiguator.py \\
        --n-frames 64 --latent-dim 16 --epochs 200 \\
        --output reports/raw/c1_wm_probe_<utc>.json

The probe runs in a few seconds on CPU and is suitable for free macOS
smoke. The verdict is `[proxy]`-tagged per CLAUDE.md axis-discipline; a
production trainer dispatching the chosen decoder type still requires
the full contest-CUDA auth eval per Catalog #221.

Cross-ref:
  .omx/research/campaign_c1_world_model_foveation_20260514.md (campaign ledger)
  feedback_zen_floor_field_medal_grade_council_landed_20260514.md (across-class)
  src/tac/substrates/c1_world_model_foveation/ (substrate package)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tac.substrates.c1_world_model_foveation.architecture import (  # noqa: E402
    WorldModelConfig,
    WorldModelModule,
    WorldModelRecurrenceMode,
)


def _real_video_feature_target(
    video_path: Path, n_frames: int, feature_dim: int
) -> torch.Tensor:
    """Extract a per-frame feature target from the real contest video.

    The probe's synthetic target is a slowly-varying random walk in
    ``feature_dim`` dimensions; the real-video analog is a low-dimensional
    spatial summary of each decoded frame. We sample ``n_frames`` evenly
    spaced frames from ``upstream/videos/0.mkv``, downsample each frame to
    a coarse ``(feature_dim // 3) ~ ceil`` luma/chroma summary, and stack
    into ``(n_frames, feature_dim)``.

    Concretely: each frame is RGB ``(874, 1164, 3)``. We:
      1. Mean over the 3 RGB channels (luma proxy).
      2. ``F.adaptive_avg_pool2d`` to ``(grid_h, grid_w)`` where
         ``grid_h * grid_w == feature_dim``.
      3. Flatten to a ``(feature_dim,)`` row.

    The result is a real-video stand-in for the synthetic stationary-ergodic
    random walk: it has the same shape but reflects ACTUAL inter-frame
    redundancy (slowly-varying driving content) rather than synthetic noise.

    Returns
    -------
    target : torch.Tensor
        Shape ``(n_frames, feature_dim)``, dtype float32. Centered to zero
        mean per-column so the world-model + per-frame baseline both fit
        from a comparable starting point (matches synthetic target which
        is a zero-centered cumsum).
    """
    import av  # local import keeps probe lightweight when synthetic-only

    # Pick a grid that factorizes feature_dim cleanly.
    grid_h = max(1, int(round(feature_dim ** 0.5)))
    while feature_dim % grid_h != 0:
        grid_h -= 1
        if grid_h < 1:
            grid_h = 1
            break
    grid_w = feature_dim // grid_h

    # Decode the full stream; we don't know total frames upfront (codec
    # reports 0 for HEVC stream count in some packs). Buffer everything
    # and uniformly subsample.
    container = av.open(str(video_path))
    luma_frames: list[torch.Tensor] = []
    for frame in container.decode(video=0):
        rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        luma = torch.from_numpy(rgb).float().mean(dim=2)  # (H, W)
        luma_frames.append(luma)
    container.close()

    total = len(luma_frames)
    if total == 0:
        raise RuntimeError(f"video at {video_path} produced 0 decoded frames")

    # Uniformly sample n_frames indices across the full stream.
    if total >= n_frames:
        step = total / n_frames
        idxs = [min(total - 1, int(i * step)) for i in range(n_frames)]
    else:
        # Pad by replicating last frame (degenerate corner case for short clips).
        idxs = list(range(total)) + [total - 1] * (n_frames - total)

    rows: list[torch.Tensor] = []
    for idx in idxs:
        luma = luma_frames[idx].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        pooled = F.adaptive_avg_pool2d(luma, (grid_h, grid_w))  # (1, 1, gh, gw)
        rows.append(pooled.reshape(-1))  # (feature_dim,)

    target = torch.stack(rows, dim=0)  # (n_frames, feature_dim)
    # Center to zero-mean per-column so absolute scale doesn't dominate.
    target = target - target.mean(dim=0, keepdim=True)
    # Rescale so std matches the synthetic target's ~0.5 scale; this keeps
    # the world-model + baseline AdamW hyperparams in the same regime.
    std = target.std()
    if std > 1e-6:
        target = target * (0.5 / std)
    return target


def _independent_frame_baseline(
    n_frames: int, latent_dim: int, epochs: int, target: torch.Tensor
) -> tuple[float, int, float]:
    """Fit an HNeRV-class baseline: per-frame index -> latent.

    Returns (residual_l2, params_bytes_est_fp4, fit_wallclock_sec).
    """
    # Per-frame independent latent embedding (no temporal coupling).
    embed = torch.nn.Embedding(n_frames, latent_dim)
    head = torch.nn.Linear(latent_dim, target.shape[-1])
    params = list(embed.parameters()) + list(head.parameters())
    n_params = sum(p.numel() for p in params)
    # FP4 = 0.5 bytes per param; cap at >= 4 bytes.
    params_bytes = max(4, n_params // 2)

    indices = torch.arange(n_frames)
    opt = torch.optim.AdamW(params, lr=1e-2)

    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        z = embed(indices)  # (n_frames, latent_dim)
        pred = head(z)  # (n_frames, feature_dim)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        opt.step()
    fit_wall = time.perf_counter() - t0

    with torch.no_grad():
        z = embed(indices)
        pred = head(z)
        residual = (pred - target).pow(2).mean().item()
    return residual, params_bytes, fit_wall


def _world_model_fit(
    n_frames: int,
    latent_dim: int,
    epochs: int,
    target: torch.Tensor,
    recurrence: WorldModelRecurrenceMode,
) -> tuple[float, int, float]:
    """Fit a world-model: z_t = WM(z_{t-1}) + linear head.

    Returns (residual_l2, params_bytes_est_fp4, fit_wallclock_sec).
    """
    wm_cfg = WorldModelConfig(
        recurrence_mode=recurrence,
        latent_dim=latent_dim,
        hidden_dim=latent_dim,
    )
    wm = WorldModelModule(wm_cfg)
    head = torch.nn.Linear(latent_dim, target.shape[-1])
    z_init = torch.nn.Parameter(torch.zeros(latent_dim))
    params = list(wm.parameters()) + list(head.parameters()) + [z_init]
    n_params = sum(p.numel() for p in params)
    params_bytes = max(4, n_params // 2)

    opt = torch.optim.AdamW(params, lr=1e-2)

    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        latents = wm.unroll(z_init, n_frames)  # (n_frames, latent_dim)
        pred = head(latents)  # (n_frames, feature_dim)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        opt.step()
    fit_wall = time.perf_counter() - t0

    with torch.no_grad():
        latents = wm.unroll(z_init, n_frames)
        pred = head(latents)
        residual = (pred - target).pow(2).mean().item()
    return residual, params_bytes, fit_wall


def run_probe(
    n_frames: int = 64,
    latent_dim: int = 16,
    feature_dim: int = 32,
    epochs: int = 200,
    seed: int = 0,
    target_video: Path | None = None,
) -> dict:
    """Run the probe and emit the verdict dict.

    When ``target_video`` is ``None`` (default), the target sequence is a
    synthetic stationary-ergodic signal: each frame's feature vector is a
    slowly-varying random walk. This is the regime where world-models
    should clearly outperform per-frame independent decoders.

    When ``target_video`` is provided (e.g. ``upstream/videos/0.mkv``), the
    target is extracted from REAL contest video via
    :func:`_real_video_feature_target` -- pyav-decoded luma frames pooled
    to ``(feature_dim,)`` per frame, centered + rescaled to match the
    synthetic target's ~0.5 std. The verdict's ``score_axis`` becomes
    ``proxy_real_video`` (still ``evidence_grade=proxy`` per Catalog #127
    -- this is a feature-space proxy, NOT a scorer call).
    """
    torch.manual_seed(seed)
    if target_video is not None:
        # Real-video feature target: pyav-decoded frames -> coarse luma grid.
        target = _real_video_feature_target(target_video, n_frames, feature_dim)
        target_source = "real_video"
        target_video_path = str(target_video)
    else:
        # Synthetic target: slowly-varying random walk (mimics stationary-ergodic
        # driving). Each frame is the previous + small noise.
        steps = torch.randn(n_frames, feature_dim) * 0.05
        target = steps.cumsum(dim=0)
        target_source = "synthetic_random_walk"
        target_video_path = None

    wm_gru = _world_model_fit(
        n_frames, latent_dim, epochs, target,
        WorldModelRecurrenceMode.GRU,
    )
    wm_lstm = _world_model_fit(
        n_frames, latent_dim, epochs, target,
        WorldModelRecurrenceMode.LSTM,
    )
    indep = _independent_frame_baseline(n_frames, latent_dim, epochs, target)

    # Verdict: lowest residual_l2 wins (the probe measures fit quality on
    # the slowly-varying target; in production the trainer also weights
    # params_bytes via the rate-term Lagrangian, but the probe is a
    # capability probe -- which family CAN fit the signal -- not a
    # rate-distortion arbitration).
    by_residual = sorted(
        [
            ("world_model_gru", wm_gru[0]),
            ("world_model_lstm", wm_lstm[0]),
            ("independent_frame_baseline", indep[0]),
        ],
        key=lambda x: x[1],
    )
    best_name, best_residual = by_residual[0]
    second_residual = by_residual[1][1]
    margin = (second_residual - best_residual) / max(second_residual, 1e-6)

    if margin < 0.05:
        verdict = "tie"
        rationale = (
            f"Top two candidates within 5%: best={best_name} "
            f"residual={best_residual:.6f} vs runner-up "
            f"{by_residual[1][0]} residual={second_residual:.6f}; margin {margin:.2%}."
        )
    else:
        verdict = best_name
        rationale = (
            f"Best fit: {best_name} residual={best_residual:.6f} vs "
            f"runner-up {by_residual[1][0]} residual={second_residual:.6f}; "
            f"margin {margin:.2%}."
        )

    return {
        "world_model_gru": {
            "residual_l2": wm_gru[0],
            "params_bytes_est_fp4": wm_gru[1],
            "fit_wallclock_sec": wm_gru[2],
        },
        "world_model_lstm": {
            "residual_l2": wm_lstm[0],
            "params_bytes_est_fp4": wm_lstm[1],
            "fit_wallclock_sec": wm_lstm[2],
        },
        "independent_frame_baseline": {
            "residual_l2": indep[0],
            "params_bytes_est_fp4": indep[1],
            "fit_wallclock_sec": indep[2],
        },
        "verdict": verdict,
        "verdict_rationale": rationale,
        "evidence_grade": "proxy",
        "score_claim_valid": False,
        "score_axis": (
            "proxy_real_video" if target_source == "real_video" else "proxy_synthetic"
        ),
        "target_source": target_source,
        "target_video_path": target_video_path,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": (
            [
                "proxy_real_video_feature_space_not_scorer_output",
                "no_scorer_load",
                "non_promotable_evidence_grade",
            ]
            if target_source == "real_video"
            else [
                "smoke_proxy_synthetic_not_contest_video",
                "no_scorer_load",
                "non_promotable_evidence_grade",
            ]
        ),
        "config": {
            "n_frames": n_frames,
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "epochs": epochs,
            "seed": seed,
            "target_video": target_video_path,
        },
        "lane_id": "lane_c1_world_model_foveation_campaign_l1_scaffold_20260514",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Probe-disambiguator for C1: world-model recurrent vs "
            "independent-frame decoder. Catalog #125 hook #6."
        )
    )
    p.add_argument("--n-frames", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target-video", type=Path, default=None,
        help=(
            "Optional path to a real video (e.g. upstream/videos/0.mkv). "
            "When set, the probe extracts a per-frame luma-pool feature "
            "target via pyav instead of generating a synthetic random walk. "
            "Verdict's score_axis becomes proxy_real_video; still "
            "evidence_grade=proxy (no scorer call). Default: None (synthetic)."
        ),
    )
    p.add_argument(
        "--output", "--output-json", dest="output", type=Path, default=None,
        help="Optional output path for verdict JSON (sorted-keys, indented).",
    )
    args = p.parse_args(argv)

    verdict = run_probe(
        n_frames=args.n_frames,
        latent_dim=args.latent_dim,
        feature_dim=args.feature_dim,
        epochs=args.epochs,
        seed=args.seed,
        target_video=args.target_video,
    )

    out_json = json.dumps(verdict, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_json, encoding="utf-8")
        print(f"[c1-probe-wm] wrote {args.output}")
    print(out_json)
    return 0


if __name__ == "__main__":  # pragma: no cover -- CLI entry
    sys.exit(main())
