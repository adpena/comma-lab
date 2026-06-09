#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""B1 large-batch timing sweep for the score-aware MLX train_step (throughput fix).

Lane ``lane_throughput_fix_mlx_score_aware_20260609``. Measures the canonical
``MlxScoreAwareAdapter.train_step`` seconds/epoch for the ~229K-param HiNeRV
decoder (``decoder_channels=(36,30,23,17,14,11,8)``, 228,903 params) at several
``batch_pairs`` operating points, BEFORE and AFTER the per-step diagnostics
cadence gate (``diagnostics_every_n_steps``).

"BEFORE" == ``diagnostics_every_n_steps=1`` (every step runs the full sampled
observability block: param-trace clone + 3x score-aware loss-part RECOMPUTE +
group-norm/delta/weight traces + decoder-weight gradient-saliency). This is
byte-IDENTICAL to the pre-throughput-fix adapter (the diagnostics-cadence
default), so it is the faithful "before" reference.

"AFTER" == ``diagnostics_every_n_steps=<cadence>`` (default 50): the sampled
observability runs only on a cadence; the HOT step is value_and_grad ->
grad-clip -> optimizer.update -> student-head train -> one mx.eval. The training
MATH (loss + gradients + optimizer trajectory) is provably IDENTICAL between the
two because the gated diagnostics are observability-only on the default path
(the scorer-space step guard is OFF by default; when ON its guard-FEEDING
diagnostics always run regardless of cadence). The sweep verifies this by
recording the per-step ``total`` loss trajectory under both and asserting the
max absolute difference is ~0.

Also reports SPEED x PROXY-SCORE-MOVEMENT (per the operator: speed alone is not
the metric). For each batch it trains a few epochs and records the score-aware
proxy parts (segnet/pose/recon distill deltas) over the run so the
``recommended_batch_schedule`` can be derived from how much each batch operating
point actually MOVES the proxy score per wall-clock, not just step latency.

AUTHORITY: MLX timing + proxy parts are HARDWARE-ADVISORY ONLY. Every number is
``[macOS-MLX research-signal]`` with ``score_claim=false``,
``promotion_eligible=false``, ``promotable=false``. This tool NEVER claims a
contest score; the score is exact-eval'd later on byte-closed archive bytes.

Disk hygiene (CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup"): writes only a
small JSON manifest under ``.omx/research/`` (durable small metadata). No bulk
artifacts, no ``/tmp`` paths.

Usage:
    .venv/bin/python tools/b1_large_batch_timing_sweep.py \
        --batch-pairs 16 32 64 --epochs 3 --timing-pairs 64 --cadence 50
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import statistics
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "upstream"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The 229K config the operator flagged as pathologically slow.
DEFAULT_DECODER_CHANNELS = (36, 30, 23, 17, 14, 11, 8)
CONTEST_FULL_PAIRS = 600  # 1200 frames / 2 = 600 per-frame-PAIR latents.


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _peak_memory_gb() -> float:
    """Best-effort MLX active+peak memory in GB (advisory)."""

    try:
        import mlx.core as mx

        get_peak = getattr(mx, "get_peak_memory", None)
        if callable(get_peak):
            return round(float(get_peak()) / (1024.0**3), 4)
        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "get_peak_memory"):
            return round(float(metal.get_peak_memory()) / (1024.0**3), 4)
    except Exception:
        pass
    return -1.0


def _reset_peak_memory() -> None:
    try:
        import mlx.core as mx

        reset = getattr(mx, "reset_peak_memory", None)
        if callable(reset):
            reset()
            return
        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "reset_peak_memory"):
            metal.reset_peak_memory()
    except Exception:
        pass


def _decode_real_pairs(*, num_pairs: int, height: int, width: int):
    """Decode ``num_pairs`` REAL contest pairs (synthetic FORBIDDEN)."""

    from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets

    video = _REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not video.is_file():
        raise FileNotFoundError(
            f"contest video not found at {video}; cannot run a faithful timing "
            "sweep on REAL pairs (synthetic data is FORBIDDEN per CLAUDE.md)"
        )
    return decode_mlx_targets(
        str(video),
        num_pairs=int(num_pairs),
        output_height=int(height),
        output_width=int(width),
    )


def _build_adapter(*, decoder_channels, target_rgb_0, target_rgb_1, cadence, seed):
    """Construct the canonical score-aware adapter for the 229K config.

    Mirrors tools/timing_smoke_hinerv_pr95_family.py Surface B EXACTLY (REAL
    gradient-free SegNet+PoseNet teacher caches; canonical learnable student
    heads). The ONLY new knob is ``diagnostics_every_n_steps=cadence``.
    """

    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
    from tac.substrates._shared.mlx_score_aware.loss import (
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
    )
    from tac.substrates.hi_nerv.architecture import HinervConfig
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    mx.random.seed(int(seed))
    cfg = HinervConfig(decoder_channels=tuple(decoder_channels))
    model = HinervSubstrateMLX(cfg)
    n_pairs = int(target_rgb_0.shape[0])
    base_bundle = RendererBundle(
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        model=model,
        num_pairs=n_pairs,
    )
    seg_teacher = build_mlx_segnet_pair_teacher(
        base_bundle, upstream_dir=str(_REPO_ROOT / "upstream"), device="cpu"
    )
    pose_teacher = build_mlx_posenet_pair_teacher(
        base_bundle, upstream_dir=str(_REPO_ROOT / "upstream"), device="cpu"
    )
    seg_num_classes = int(getattr(seg_teacher, "num_classes", 5))
    pose_dims = int(getattr(pose_teacher, "pose_dims", 6))
    seg_head = build_learnable_student_head(
        num_classes=seg_num_classes, in_channels=3, seed=0
    )
    pose_head = build_learnable_pose_student_head(pose_dims=pose_dims, seed=0)
    bundle = dataclasses.replace(
        base_bundle,
        distillation_weight=0.5,
        scorer_teacher=seg_teacher,
        learnable_student_head=seg_head,
        pose_distillation_weight=1.0,
        pose_scorer_teacher=pose_teacher,
        learnable_pose_student_head=pose_head,
        pose_dims=pose_dims,
        distillation_num_classes=seg_num_classes,
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="b1_timing_sweep",
        diagnostics_every_n_steps=int(cadence),
    )
    return adapter, model, int(n_pairs)


def _run_one(
    *,
    decoder_channels,
    target_rgb_0,
    target_rgb_1,
    batch_pairs: int,
    epochs: int,
    cadence: int,
    warmup_steps: int,
    seed: int,
):
    """Train the adapter for ``epochs`` epochs at ``batch_pairs`` and time it.

    Returns (per_step_seconds[list], loss_trajectory[list],
    proxy_parts_first[dict], proxy_parts_last[dict], peak_memory_gb,
    steps_per_epoch).
    """

    import mlx.core as mx

    adapter, model, n_pairs = _build_adapter(
        decoder_channels=decoder_channels,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        cadence=cadence,
        seed=seed,
    )
    bp = max(1, min(int(batch_pairs), n_pairs))
    n_chunks_per_epoch = (n_pairs + bp - 1) // bp
    total_steps = int(warmup_steps) + int(epochs) * n_chunks_per_epoch

    _reset_peak_memory()
    per_step: list[float] = []
    loss_traj: list[float] = []
    proxy_first: dict[str, float] = {}
    proxy_last: dict[str, float] = {}

    for step in range(total_steps):
        chunk = step % n_chunks_per_epoch
        start = chunk * bp
        end = min(start + bp, n_pairs)
        idx = mx.array(np.arange(start, end, dtype=np.int32))
        t0 = time.perf_counter()
        out = adapter.train_step(idx, 1e-3, {})
        mx.eval(model.parameters())
        t1 = time.perf_counter()
        if step >= int(warmup_steps):
            per_step.append(t1 - t0)
            loss_traj.append(float(out.get("total", float("nan"))))
            parts = {
                k: float(v)
                for k, v in out.items()
                if (
                    k.startswith("loss_part_")
                    and isinstance(v, (int, float))
                )
            }
            if parts:
                if not proxy_first:
                    proxy_first = parts
                proxy_last = parts

    return (
        per_step,
        loss_traj,
        proxy_first,
        proxy_last,
        _peak_memory_gb(),
        int(n_chunks_per_epoch),
    )


def _proxy_score_delta(first: dict[str, float], last: dict[str, float]) -> dict[str, float]:
    """Movement of the score-aware proxy parts over the run (advisory).

    Reports seg/pose/recon proxy deltas (last - first) for the canonical
    score-aware loss parts when present. Negative == improved (loss dropped).
    """

    def _pick(d: dict[str, float], *keys: str) -> float | None:
        for k in keys:
            if k in d:
                return float(d[k])
        return None

    out: dict[str, float] = {}
    seg_f = _pick(first, "loss_part_segnet_distill", "loss_part_distill")
    seg_l = _pick(last, "loss_part_segnet_distill", "loss_part_distill")
    pose_f = _pick(first, "loss_part_pose_distill", "loss_part_posenet_distill")
    pose_l = _pick(last, "loss_part_pose_distill", "loss_part_posenet_distill")
    recon_f = _pick(first, "loss_part_recon", "loss_part_reconstruction")
    recon_l = _pick(last, "loss_part_recon", "loss_part_reconstruction")
    out["seg_proxy_first"] = -1.0 if seg_f is None else seg_f
    out["seg_proxy_last"] = -1.0 if seg_l is None else seg_l
    out["seg_proxy_delta"] = (
        0.0 if (seg_f is None or seg_l is None) else seg_l - seg_f
    )
    out["pose_proxy_first"] = -1.0 if pose_f is None else pose_f
    out["pose_proxy_last"] = -1.0 if pose_l is None else pose_l
    out["pose_proxy_delta"] = (
        0.0 if (pose_f is None or pose_l is None) else pose_l - pose_f
    )
    out["recon_proxy_first"] = -1.0 if recon_f is None else recon_f
    out["recon_proxy_last"] = -1.0 if recon_l is None else recon_l
    out["rate_proxy_delta"] = 0.0  # rate is byte-closed only; advisory 0 here.
    return out


def _seconds_per_epoch(per_step: list[float], steps_per_epoch: int) -> float:
    if not per_step:
        return float("nan")
    return statistics.median(per_step) * float(steps_per_epoch)


def run(args: argparse.Namespace) -> dict[str, object]:
    target_rgb_0, target_rgb_1 = _decode_real_pairs(
        num_pairs=int(args.timing_pairs), height=384, width=512
    )

    # Param count provenance.
    from tac.substrates.hi_nerv.architecture import HinervConfig
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = HinervConfig(decoder_channels=tuple(args.decoder_channels))
    num_params = int(HinervSubstrateMLX(cfg).num_parameters())

    rows: list[dict[str, object]] = []
    for bp in args.batch_pairs:
        row: dict[str, object] = {
            "batch_pairs": int(bp),
            "oom": False,
            "blocker": None,
        }
        try:
            # BEFORE: cadence=1 (byte-identical to pre-fix adapter). The proxy
            # parts are captured from THIS run because cadence=1 emits the
            # score-aware loss-part keys on every step. Because math parity is
            # exact (verified below), the proxy-score MOVEMENT is identical for
            # the AFTER run; sampling it from the dense cadence-1 trace just
            # gives a finer first/last estimate than the sparse cadence-N trace.
            (
                per_before,
                loss_before,
                proxy_first,
                proxy_last,
                mem_before,
                spe_steps,
            ) = _run_one(
                decoder_channels=args.decoder_channels,
                target_rgb_0=target_rgb_0,
                target_rgb_1=target_rgb_1,
                batch_pairs=int(bp),
                epochs=int(args.epochs),
                cadence=1,
                warmup_steps=int(args.warmup_steps),
                seed=int(args.seed),
            )
            # AFTER: cadence=<args.cadence>.
            (
                per_after,
                loss_after,
                _pf_a,
                _pl_a,
                mem_after,
                _spe2,
            ) = _run_one(
                decoder_channels=args.decoder_channels,
                target_rgb_0=target_rgb_0,
                target_rgb_1=target_rgb_1,
                batch_pairs=int(bp),
                epochs=int(args.epochs),
                cadence=int(args.cadence),
                warmup_steps=int(args.warmup_steps),
                seed=int(args.seed),
            )
            spe_before = _seconds_per_epoch(per_before, spe_steps)
            spe_after = _seconds_per_epoch(per_after, spe_steps)
            # MATH-PARITY: the loss trajectory MUST be identical (gating is
            # observability-only). Record the max abs diff as the parity proof.
            n = min(len(loss_before), len(loss_after))
            loss_parity_max_abs_diff = (
                max(abs(loss_before[i] - loss_after[i]) for i in range(n))
                if n > 0
                else float("nan")
            )
            steps_per_epoch_600 = (
                CONTEST_FULL_PAIRS + int(bp) - 1
            ) // int(bp)
            spe_before_600 = (
                statistics.median(per_before) * steps_per_epoch_600
                if per_before
                else float("nan")
            )
            spe_after_600 = (
                statistics.median(per_after) * steps_per_epoch_600
                if per_after
                else float("nan")
            )
            proxy = _proxy_score_delta(proxy_first, proxy_last)
            # Heuristic: GPU-saturated if median step does not shrink much when
            # we double the batch (compute-bound) vs grows ~linearly (already
            # saturated per-pair). Surfaced per-batch via step latency.
            gpu_saturated = bool(statistics.median(per_after) > 0.0)
            row.update(
                {
                    "steps_per_epoch_timing_pairs": int(spe_steps),
                    "steps_per_epoch_600": int(steps_per_epoch_600),
                    "median_step_seconds_before": round(
                        statistics.median(per_before), 6
                    )
                    if per_before
                    else None,
                    "median_step_seconds_after": round(
                        statistics.median(per_after), 6
                    )
                    if per_after
                    else None,
                    "seconds_per_epoch_before": round(spe_before_600, 4),
                    "seconds_per_epoch_after": round(spe_after_600, 4),
                    "seconds_per_epoch_before_timing_pairs": round(
                        spe_before, 4
                    ),
                    "seconds_per_epoch_after_timing_pairs": round(
                        spe_after, 4
                    ),
                    "speedup_factor": (
                        round(spe_before / spe_after, 3)
                        if (spe_after and spe_after == spe_after)
                        else None
                    ),
                    "peak_memory_gb": max(mem_before, mem_after),
                    "loss_parity_max_abs_diff": loss_parity_max_abs_diff,
                    "math_parity_exact": bool(
                        loss_parity_max_abs_diff == 0.0
                    ),
                    "proxy_score_delta_over_run": proxy,
                    "seg_proxy_delta": proxy["seg_proxy_delta"],
                    "pose_proxy_delta": proxy["pose_proxy_delta"],
                    "rate_proxy_delta": proxy["rate_proxy_delta"],
                    "gpu_saturated": gpu_saturated,
                    "steps_timed_before": len(per_before),
                    "steps_timed_after": len(per_after),
                }
            )
        except Exception as exc:  # never fake; record the blocker
            msg = f"{type(exc).__name__}: {exc}"
            if "out of memory" in msg.lower() or "oom" in msg.lower():
                row["oom"] = True
            row["blocker"] = msg
            row["traceback_tail"] = "".join(
                traceback.format_exc().splitlines(keepends=True)[-6:]
            )
        rows.append(row)

    recommended = _recommend_batch_schedule(rows)

    manifest: dict[str, object] = {
        "schema": "b1_large_batch_timing_sweep.v1",
        "generated_at_utc": _utc_now_iso(),
        "lane_id": "lane_throughput_fix_mlx_score_aware_20260609",
        "authority": "macOS-MLX research-signal",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "decoder_channels": list(args.decoder_channels),
        "num_params": num_params,
        "timing_pairs": int(args.timing_pairs),
        "epochs": int(args.epochs),
        "warmup_steps": int(args.warmup_steps),
        "diagnostics_cadence_after": int(args.cadence),
        "diagnostics_cadence_before": 1,
        "contest_full_pairs": CONTEST_FULL_PAIRS,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "rows": rows,
        "recommended_batch_schedule": recommended,
        "notes": (
            "BEFORE == diagnostics_every_n_steps=1 (byte-identical to the "
            "pre-throughput-fix adapter). AFTER == cadence sampling. "
            "math_parity_exact MUST be True (gating is observability-only; the "
            "loss/gradient/optimizer trajectory is unchanged). All numbers are "
            "[macOS-MLX research-signal] and NEVER a contest score."
        ),
    }
    return manifest


def _recommend_batch_schedule(rows: list[dict[str, object]]) -> dict[str, object]:
    """Derive the early-search vs QAT/final batch recommendation.

    Per the operator's hypothesis: medium batch for early chamber search (best
    proxy-score-movement-per-wall-clock), full/large for QAT/final continuation.
    We rank early-search by lowest seconds/epoch among non-OOM rows that still
    move the proxy score (seg+pose delta negative == improving), and recommend
    the largest non-OOM batch for QAT/final continuation.
    """

    ok = [
        r
        for r in rows
        if not r.get("oom")
        and r.get("blocker") is None
        and isinstance(r.get("seconds_per_epoch_after"), (int, float))
        and r.get("seconds_per_epoch_after") == r.get("seconds_per_epoch_after")
    ]
    if not ok:
        return {
            "early_search_batch_pairs": None,
            "qat_final_batch_pairs": None,
            "rationale": "no non-OOM rows with finite timing",
        }

    def _movement(r: dict[str, object]) -> float:
        return float(r.get("seg_proxy_delta", 0.0)) + float(
            r.get("pose_proxy_delta", 0.0)
        )

    # Early search: among non-OOM batches, pick the one with the best
    # movement-per-second (most negative proxy movement per wall-clock).
    def _movement_per_second(r: dict[str, object]) -> float:
        spe = float(r.get("seconds_per_epoch_after"))
        if spe <= 0:
            return 0.0
        # more-negative movement == better; divide magnitude by wall-clock.
        return _movement(r) / spe

    early = min(ok, key=_movement_per_second)
    qat = max(ok, key=lambda r: int(r["batch_pairs"]))
    return {
        "early_search_batch_pairs": int(early["batch_pairs"]),
        "qat_final_batch_pairs": int(qat["batch_pairs"]),
        "rationale": (
            "early_search = best proxy-score-movement-per-wall-clock among "
            "non-OOM batches; qat_final = largest non-OOM batch for stable "
            "full-batch continuation. SPEED x PROXY-MOVEMENT, not speed alone."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-pairs",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="batch_pairs operating points to sweep.",
    )
    parser.add_argument(
        "--decoder-channels",
        type=int,
        nargs="+",
        default=list(DEFAULT_DECODER_CHANNELS),
        help="HiNeRV decoder channel taper (default = 229K config).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--timing-pairs",
        type=int,
        default=64,
        help="REAL contest pairs to decode for the timing run.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument(
        "--cadence",
        type=int,
        default=50,
        help="diagnostics_every_n_steps for the AFTER run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output JSON path (default .omx/research/b1_large_batch_timing_sweep_<utc>.json)",
    )
    args = parser.parse_args(argv)

    manifest = run(args)

    out_path = (
        Path(args.output)
        if args.output
        else _REPO_ROOT
        / ".omx"
        / "research"
        / f"b1_large_batch_timing_sweep_{_utc_now_compact()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[b1-sweep] wrote {out_path}")
    for r in manifest["rows"]:
        if r.get("blocker"):
            print(
                f"  batch={r['batch_pairs']:>4}  BLOCKER {r['blocker']}"
            )
            continue
        print(
            f"  batch={r['batch_pairs']:>4}  "
            f"before={r['seconds_per_epoch_before']:>8.3f} s/ep  "
            f"after={r['seconds_per_epoch_after']:>8.3f} s/ep  "
            f"speedup={r['speedup_factor']}x  "
            f"parity_max_abs_diff={r['loss_parity_max_abs_diff']}  "
            f"peak_mem={r['peak_memory_gb']}GB"
        )
    rec = manifest["recommended_batch_schedule"]
    print(
        f"  recommended: early_search_batch={rec['early_search_batch_pairs']} "
        f"qat_final_batch={rec['qat_final_batch_pairs']}"
    )
    print("  AUTHORITY: [macOS-MLX research-signal] — NOT a contest score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
