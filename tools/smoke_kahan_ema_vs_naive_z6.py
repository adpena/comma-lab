# SPDX-License-Identifier: MIT
"""Carmack 30-min smoke: Kahan-EMA vs naive Polyak head-to-head on Z6 L2.

Per T3 grand council OP #3 (commit ``7d04474cb``) Carmack MVP-first smoke
verification: implement the smallest possible head-to-head empirical
comparison between the canonical naive ``PolyakEMAShadow`` and the
canonical Kahan-compensated ``KahanCompensatedPolyakEMAShadow`` on the
Z6 L2 substrate's real training trajectory.

The methodology is the simplest faithful comparison: run the Z6 L2
trainer ONCE, snapshot the initial + per-step live-weight sequence in
memory, then replay that sequence through BOTH a naive and a Kahan EMA
shadow plus an fp64 NumPy reference. The reference-relative errors, not
live-vs-shadow lag, determine whether Kahan materially reduces numerical
EMA accumulation error on the real Z6 trajectory. Because both shadows
see the exact same live-weight sequence, the comparison isolates EMA
accumulation arithmetic without needing two full retraining runs.

Cost: $0 (all MLX-local per CLAUDE.md "MLX portable-local-substrate
authority"); wall-clock ~5 minutes total at 300/500/1000 epoch grid.

Per CLAUDE.md "MLX portable-local-substrate authority" every output is
tagged ``[macOS-MLX research-signal]`` non-promotable per Catalog
#127/#192/#317/#341.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": every
drift number in the output carries the originating ``[empirical:<path>]``
tag via the canonical Provenance per Catalog #323.

Usage::

    .venv/bin/python tools/smoke_kahan_ema_vs_naive_z6.py \\
        --epochs-grid 300 500 1000 \\
        --output-dir experiments/results/kahan_ema_smoke_<utc>/

Output: prints comparison table to stdout + writes canonical JSON to
``<output-dir>/kahan_vs_naive_drift_comparison.json`` for downstream
consumers (cathedral autopilot, canonical equation registry per
Catalog #344).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _capture_live_weight_sequence(
    *,
    num_pairs: int,
    epochs: int,
    output_h: int,
    output_w: int,
    seed: int,
    video_path: Path,
    lambda_residual: float,
    learning_rate: float = 1e-3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run canonical Z6 L2 adapter's train_step per epoch + snapshot live state.

    Uses the CANONICAL ``Z6LongTrainingAdapter.train_step`` invocation so
    the captured trajectory is byte-stable identical to what the canonical
    L2 trainer would produce; the smoke is a real Z6 L2 training run with
    in-memory per-step state snapshots.

    Returns (live_state_sequence, telemetry) where live_state_sequence
    is a list of cloned per-step state_dicts and telemetry carries
    wall-clock + loss trajectory for the canonical artifact.
    """
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    from tac.data import decode_video
    from tac.substrates.time_traveler_l5_z6.architecture import Z6PredictiveCodingConfig
    from tac.substrates.time_traveler_l5_z6.long_training_adapter import Z6LongTrainingAdapter

    np.random.seed(seed)
    mx.random.seed(seed)

    cfg = Z6PredictiveCodingConfig(
        latent_dim=24,
        num_pairs=num_pairs,
        output_height=output_h,
        output_width=output_w,
        decoder_num_upsample_blocks=1,  # 48x64 path
        decoder_channels=(6,),
        decoder_embed_dim=16,
        predictor_depth=1,
        lambda_residual_entropy=lambda_residual,
    )

    print(f"[smoke] decoding {2 * num_pairs} frames at {output_h}x{output_w}...")
    t_decode = time.time()
    gt_frames = decode_video(
        video_path,
        target_h=output_h,
        target_w=output_w,
        max_frames=2 * num_pairs,
    )
    print(f"[smoke] decoded {len(gt_frames)} frames in {time.time() - t_decode:.1f}s")
    gt_arr = np.stack([f.numpy() for f in gt_frames], axis=0)
    gt_pairs = gt_arr.reshape(num_pairs, 2, output_h, output_w, 3)
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))

    adapter = Z6LongTrainingAdapter(
        config=cfg,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        lambda_residual=lambda_residual,
    )
    # Seed ego-motion buffer per Catalog #311 ego-motion-conditioning + L2 trainer pattern.
    ego = (
        np.random.RandomState(seed + 100).randn(num_pairs, cfg.predictor_ego_motion_dim).astype(np.float32)
        * 0.1
    )
    adapter.model.ego_motion_buffer = mx.array(ego)

    flat_initial = dict(tree_flatten(adapter.model.parameters()))
    live_state_sequence: list[dict[str, Any]] = [
        {k: np.array(v, copy=True) for k, v in flat_initial.items()}
    ]
    loss_traj: list[float] = []
    batch_size = min(num_pairs, 8)
    loss_weights = {"recon": 1.0, "residual": lambda_residual}

    t_train = time.time()
    for epoch in range(epochs):
        batch = adapter.sample_batch(batch_size, seed=seed * 100000 + epoch)
        loss_result = adapter.train_step(batch, learning_rate, loss_weights)
        loss_traj.append(float(loss_result["total"]))
        # Snapshot live state per step (flatten + clone via numpy conversion
        # so the snapshot is independent of subsequent MLX in-place updates).
        flat = dict(tree_flatten(adapter.model.parameters()))
        live_state_sequence.append(
            {k: np.array(v, copy=True) for k, v in flat.items()}
        )
    wall = time.time() - t_train
    print(
        f"[smoke] epochs={epochs} wall={wall:.1f}s "
        f"loss_init={loss_traj[0]:.4f} loss_final={loss_traj[-1]:.4f}"
    )
    telemetry: dict[str, Any] = {
        "epochs": epochs,
        "wall_seconds": float(wall),
        "loss_initial": float(loss_traj[0]),
        "loss_final": float(loss_traj[-1]),
        "loss_reduction_pct": float(100.0 * (1.0 - loss_traj[-1] / loss_traj[0])),
        "num_pairs": num_pairs,
        "output_h": output_h,
        "output_w": output_w,
        "seed": seed,
        "captured_initial_state": True,
        "n_param_keys": len(live_state_sequence[0]),
    }
    return live_state_sequence, telemetry


def _run_ema_shadow_on_sequence(
    live_state_sequence: list[dict[str, Any]],
    *,
    decay: float,
    enable_kahan: bool,
) -> tuple[dict[str, Any], float]:
    """Replay a captured live-state sequence through a fresh EMA shadow.

    Returns (final_shadow_state, drift_l2_vs_final_live) where drift_l2
    is the canonical PolyakEMAShadow.drift_l2 metric between the final
    EMA shadow and the final live state. Both modes see EXACTLY the same
    sequence so any divergence is M2 (EMA accumulation) attributable.
    """
    from tac.training.long_training_canonical import PolyakEMAShadow

    class _FrozenModel:
        """Duck-typed model exposing state_dict for PolyakEMAShadow."""

        def __init__(self, state: dict[str, Any]):
            self._state = state

        def state_dict(self) -> dict[str, Any]:
            return self._state

    # Initialize shadow from FIRST state in the sequence (canonical seed).
    initial = _FrozenModel(live_state_sequence[0])
    shadow = PolyakEMAShadow(initial, decay=decay, enable_kahan=enable_kahan)
    # Replay remaining states (positions 1..N-1):
    for state in live_state_sequence[1:]:
        shadow.update(_FrozenModel(state))
    # Drift between final shadow and final live state:
    final_live = _FrozenModel(live_state_sequence[-1])
    drift_l2 = shadow.drift_l2(final_live)
    return shadow.state_dict(), drift_l2


def _run_fp64_reference_ema_on_sequence(
    live_state_sequence: list[dict[str, Any]],
    *,
    decay: float,
) -> dict[str, Any]:
    """Replay the EMA recurrence in NumPy fp64 as the numerical reference."""
    import numpy as np

    reference: dict[str, Any] = {
        k: np.asarray(v, dtype=np.float64).copy()
        for k, v in live_state_sequence[0].items()
    }
    for state in live_state_sequence[1:]:
        for key, live_value in state.items():
            live64 = np.asarray(live_value, dtype=np.float64)
            if key not in reference:
                reference[key] = live64.copy()
                continue
            reference[key] = decay * reference[key] + (1.0 - decay) * live64
    return reference


def _measure_shadow_error_vs_reference(
    shadow: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float]:
    """Measure L2/max_abs shadow error against an fp64 reference EMA."""
    import numpy as np

    total_l2_sq = 0.0
    max_abs = 0.0
    n_elements = 0
    for key, ref_value in reference.items():
        if key not in shadow:
            continue
        shadow_value = np.asarray(shadow[key], dtype=np.float64)
        ref_arr = np.asarray(ref_value, dtype=np.float64)
        if shadow_value.shape != ref_arr.shape:
            continue
        diff = shadow_value - ref_arr
        total_l2_sq += float(np.sum(diff * diff))
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        n_elements += int(diff.size)
    return {
        "error_vs_fp64_l2": float(total_l2_sq ** 0.5),
        "error_vs_fp64_max_abs": max_abs,
        "n_reference_elements": float(n_elements),
    }


def _measure_pairwise_shadow_divergence(
    naive_shadow: dict[str, Any],
    kahan_shadow: dict[str, Any],
) -> dict[str, float]:
    """Measure the canonical L2 / max-abs divergence between the two shadows."""
    import numpy as np

    total_l2_sq = 0.0
    max_abs = 0.0
    n_elements = 0
    for k in naive_shadow:
        if k not in kahan_shadow:
            continue
        nv = np.asarray(naive_shadow[k])
        kv = np.asarray(kahan_shadow[k])
        if nv.shape != kv.shape:
            continue
        diff = nv.astype(np.float64) - kv.astype(np.float64)
        total_l2_sq += float(np.sum(diff * diff))
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        n_elements += int(diff.size)
    return {
        "kahan_vs_naive_shadow_l2": float(total_l2_sq ** 0.5),
        "kahan_vs_naive_shadow_max_abs": max_abs,
        "n_shadow_elements": float(n_elements),
    }


def _build_canonical_provenance(
    output_dir: Path,
) -> dict[str, Any]:
    """Per Catalog #323 canonical Provenance with non-promotable markers."""
    return {
        "axis_tag": "[macOS-MLX research-signal]",
        "evidence_grade": "macOS-MLX-research-signal",
        "hardware_substrate": "darwin_arm64_m5_max_macos_mlx",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "captured_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "captured_artifact_path": str(output_dir),
        "canonical_helper": (
            "tools.smoke_kahan_ema_vs_naive_z6:_run_ema_shadow_on_sequence"
        ),
        "t3_council_anchor": "commit_7d04474cb",
        "canonical_kahan_reference": "Kahan_1965_Pracniques_CACM_8_1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs-grid",
        type=int,
        nargs="+",
        default=[300, 500, 1000],
        help="Training depths to sweep (default per T3 OP #3 30-min smoke).",
    )
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--output-h", type=int, default=48)
    parser.add_argument("--output-w", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-residual", type=float, default=1.0)
    parser.add_argument(
        "--video-path", type=Path, default=Path("upstream/videos/0.mkv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            f"experiments/results/kahan_ema_smoke_"
            f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        ),
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.997,
        help="Canonical Polyak decay per Catalog #2.",
    )
    args = parser.parse_args(argv)

    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print(
            "[smoke] FATAL: MLX required (Apple Silicon); see CLAUDE.md "
            "'MLX portable-local-substrate authority'",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[smoke] Carmack 30-min smoke: Kahan-EMA vs naive Polyak head-to-head")
    print(f"[smoke] epochs_grid={args.epochs_grid}; decay={args.decay}")
    print(f"[smoke] output_dir={args.output_dir}")
    print()

    results: list[dict[str, Any]] = []
    t_total_start = time.time()
    for epochs in args.epochs_grid:
        print(f"[smoke] === EPOCHS={epochs} ===")
        # 1) Capture live weight sequence by running Z6 L2 training.
        live_seq, telemetry = _capture_live_weight_sequence(
            num_pairs=args.num_pairs,
            epochs=epochs,
            output_h=args.output_h,
            output_w=args.output_w,
            seed=args.seed,
            video_path=args.video_path,
            lambda_residual=args.lambda_residual,
        )
        # 2) Replay through NAIVE Polyak EMA shadow.
        t_naive = time.time()
        naive_shadow, naive_drift = _run_ema_shadow_on_sequence(
            live_seq, decay=args.decay, enable_kahan=False
        )
        wall_naive = time.time() - t_naive
        # 3) Replay through KAHAN-compensated EMA shadow.
        t_kahan = time.time()
        kahan_shadow, kahan_drift = _run_ema_shadow_on_sequence(
            live_seq, decay=args.decay, enable_kahan=True
        )
        wall_kahan = time.time() - t_kahan
        # 4) Compute fp64-reference numerical error and pairwise divergence.
        fp64_reference = _run_fp64_reference_ema_on_sequence(
            live_seq,
            decay=args.decay,
        )
        naive_error = _measure_shadow_error_vs_reference(naive_shadow, fp64_reference)
        kahan_error = _measure_shadow_error_vs_reference(kahan_shadow, fp64_reference)
        divergence = _measure_pairwise_shadow_divergence(naive_shadow, kahan_shadow)
        # 5) Canonical numerical-error reduction ratio. live-vs-shadow drift is
        # lag telemetry, not proof that one accumulator is more correct.
        drift_reduction_ratio = (
            naive_drift / kahan_drift if kahan_drift > 0.0 else float("inf")
        )
        naive_error_l2 = naive_error["error_vs_fp64_l2"]
        kahan_error_l2 = kahan_error["error_vs_fp64_l2"]
        if kahan_error_l2 > 0.0:
            error_reduction_ratio = naive_error_l2 / kahan_error_l2
        elif naive_error_l2 > 0.0:
            error_reduction_ratio = float("inf")
        else:
            error_reduction_ratio = 1.0
        # 6) Sister #1265 verdict bookkeeping.
        # NB: Sister #1265 gate's canonical metric is max_abs(mlx_decoder_output -
        # pytorch_decoder_output) on a reconstruction probe — i.e. CROSS-RUNTIME
        # decoder-output parity — NOT live-vs-shadow drift_l2. This smoke
        # MEASURES the M2 mitigation magnitude via kahan_vs_naive shadow
        # divergence (the canonical column ``kahan_vs_naive_shadow_max_abs``);
        # the columns below tag drift_l2 (live-vs-shadow) for completeness so
        # readers can correlate with the canonical L2 trainer's
        # ``final_ema_drift_L2`` telemetry signal (per the DRIFT-VS-DEPTH-CHAR
        # landing memo's anchor table). The 0.001 threshold below is per
        # Sister #1265 CANONICAL anchor (max_abs space; included here only
        # so downstream analysts can sanity-check the order-of-magnitude;
        # a true Sister #1265 verdict requires running ``tools/gate_mlx_
        # candidate_contest_equivalence_z6.py`` on a real archive).
        sister_1265_proxy_naive = "PASS" if naive_drift < 0.001 else "FAIL"
        sister_1265_proxy_kahan = "PASS" if kahan_drift < 0.001 else "FAIL"
        row: dict[str, Any] = {
            **telemetry,
            "decay": args.decay,
            "naive_drift_l2": float(naive_drift),
            "kahan_drift_l2": float(kahan_drift),
            "drift_reduction_ratio_naive_over_kahan": float(drift_reduction_ratio),
            "naive_error_vs_fp64_l2": naive_error_l2,
            "kahan_error_vs_fp64_l2": kahan_error_l2,
            "naive_error_vs_fp64_max_abs": naive_error["error_vs_fp64_max_abs"],
            "kahan_error_vs_fp64_max_abs": kahan_error["error_vs_fp64_max_abs"],
            "error_reduction_ratio_naive_over_kahan": float(error_reduction_ratio),
            "kahan_vs_naive_shadow_divergence_l2": divergence["kahan_vs_naive_shadow_l2"],
            "kahan_vs_naive_shadow_divergence_max_abs": divergence["kahan_vs_naive_shadow_max_abs"],
            "n_shadow_elements": divergence["n_shadow_elements"],
            "wall_seconds_naive_shadow": float(wall_naive),
            "wall_seconds_kahan_shadow": float(wall_kahan),
            "sister_1265_proxy_verdict_naive": sister_1265_proxy_naive,
            "sister_1265_proxy_verdict_kahan": sister_1265_proxy_kahan,
        }
        results.append(row)
        print(
            f"[smoke] epochs={epochs:>4d} | "
            f"naive_drift={naive_drift:.4e} ({sister_1265_proxy_naive}) | "
            f"kahan_drift={kahan_drift:.4e} ({sister_1265_proxy_kahan}) | "
            f"error_reduction={error_reduction_ratio:.3f}x | "
            f"kahan_vs_naive_max_abs={divergence['kahan_vs_naive_shadow_max_abs']:.4e}"
        )

    total_wall = time.time() - t_total_start
    print()
    print(f"[smoke] TOTAL wall_seconds={total_wall:.1f}")

    # 7) Print canonical comparison table.
    print()
    print("=" * 100)
    print("KAHAN-EMA vs NAIVE POLYAK CANONICAL COMPARISON TABLE")
    print("=" * 100)
    print(
        f"{'epochs':>6} | {'wall_s':>6} | {'loss_init':>9} | {'loss_final':>10} | "
        f"{'naive_err64':>13} | {'kahan_err64':>13} | {'err_red_x':>11} | "
        f"{'naive_1265':>10} | {'kahan_1265':>10}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['epochs']:>6d} | {r['wall_seconds']:>6.1f} | "
            f"{r['loss_initial']:>9.4f} | {r['loss_final']:>10.4f} | "
            f"{r['naive_error_vs_fp64_l2']:>13.4e} | "
            f"{r['kahan_error_vs_fp64_l2']:>13.4e} | "
            f"{r['error_reduction_ratio_naive_over_kahan']:>11.3f} | "
            f"{r['sister_1265_proxy_verdict_naive']:>10s} | "
            f"{r['sister_1265_proxy_verdict_kahan']:>10s}"
        )
    print("=" * 100)

    # 8) Write canonical JSON output for downstream consumers.
    provenance = _build_canonical_provenance(args.output_dir)
    output: dict[str, Any] = {
        "schema_version": "kahan_ema_vs_naive_smoke_v1_20260526",
        "carmack_30min_smoke": True,
        "t3_council_op": "OP_3_carmack_30_min_smoke_verification",
        "t3_council_anchor": "commit_7d04474cb",
        "canonical_provenance": provenance,
        "config": {
            "epochs_grid": args.epochs_grid,
            "num_pairs": args.num_pairs,
            "output_h": args.output_h,
            "output_w": args.output_w,
            "seed": args.seed,
            "decay": args.decay,
            "lambda_residual": args.lambda_residual,
            "video_path": str(args.video_path),
        },
        "results": results,
        "total_wall_seconds": float(total_wall),
        "verdict": _classify_verdict(results),
    }
    out_json = args.output_dir / "kahan_vs_naive_drift_comparison.json"
    out_json.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"[smoke] canonical artifact: {out_json}")
    print(f"[smoke] [empirical:{out_json}] -- Catalog #287 evidence-tag discipline")

    return 0


def _classify_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify fp64-reference error reduction per T3 council 2x threshold."""
    if not results:
        return {"summary": "NO_RESULTS", "max_reduction_ratio": 0.0}
    ratios = [r["error_reduction_ratio_naive_over_kahan"] for r in results]
    finite_ratios = [r for r in ratios if r != float("inf")]
    max_ratio = max(finite_ratios) if finite_ratios else 0.0
    if max_ratio >= 2.0:
        summary = (
            f"KAHAN_EMA_MATERIALLY_REDUCES_DRIFT_{max_ratio:.2f}x; "
            f"recommend canonical equation registration per Catalog #344 "
            f"(kahan_ema_drift_mitigation_v1)"
        )
    elif max_ratio >= 1.1:
        summary = (
            f"KAHAN_EMA_MARGINAL_REDUCTION_{max_ratio:.2f}x; "
            f"keep opt-in per Catalog #265 narrow public API"
        )
    else:
        summary = (
            f"KAHAN_EMA_FP_NOISE_ONLY_{max_ratio:.2f}x; M2 sub-dominant; "
            f"keep opt-in (still principled hardening per T3 verdict)"
        )
    return {
        "summary": summary,
        "max_reduction_ratio_observed": float(max_ratio),
        "all_reduction_ratios": [float(x) for x in ratios],
        "ratio_metric": "fp64_reference_error_l2_naive_over_kahan",
        "threshold_canonical_equation": 2.0,
        "registration_recommended": bool(max_ratio >= 2.0),
    }


if __name__ == "__main__":
    sys.exit(main())
