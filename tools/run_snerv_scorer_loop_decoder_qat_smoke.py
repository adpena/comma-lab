#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Run a bounded local SNeRV scorer-loop decoder/QAT smoke."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (  # noqa: E402
    BYTE_GROWTH_ADMISSION_MODES,
    COMPONENT_GUARD_MODES,
    DEFAULT_DYNAMIC_RANGE_REPAIR_GAINS,
    run_snerv_scorer_loop_decoder_qat_smoke,
)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_scorer_loop_decoder_qat_smoke_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--progress-jsonl",
        default=None,
        help=(
            "Append one false-authority JSON row after every receiver-replayed "
            "scorer-loop evaluation. This is progress custody only; it is not "
            "promotion authority."
        ),
    )
    parser.add_argument("--n-pairs", type=int, default=1)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--wavelet", default="db2")
    parser.add_argument("--target-bits-per-coeff", type=float, default=5.0)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--upstream-dir", default="upstream")
    parser.add_argument("--video-path", default="upstream/videos/0.mkv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--step-map-bins", type=int, default=16)
    parser.add_argument("--snerv-spectra-preserving-adapter", action="store_true")
    parser.add_argument("--snerv-fc-dim", type=int, default=9)
    parser.add_argument("--snerv-emb-size", type=int, default=0)
    parser.add_argument("--snerv-patch-radius", type=int, default=1)
    parser.add_argument("--snerv-mfu-scales", default="1,2,4")
    parser.add_argument("--snerv-hfr-gain", type=float, default=0.0)
    parser.add_argument("--snerv-temporal-context", type=int, default=0)
    parser.add_argument(
        "--snerv-temporal-mode",
        default="delta",
        choices=("delta", "official_haar_dwt1d_lowpass"),
    )
    parser.add_argument(
        "--snerv-scorer-loop-lf-payload-codec",
        default="portfolio_auto",
    )
    parser.add_argument("--qat-bits", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=2)
    parser.add_argument(
        "--search-mode",
        choices=(
            "random_signed",
            "top_weight_coordinate",
            "learned_random_subspace",
            "nes_pair_robust",
        ),
        default="random_signed",
    )
    parser.add_argument("--perturb-scale", type=float, default=0.02)
    parser.add_argument("--byte-pressure-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--section-value-pressure-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier for train-time SNAR1 optional-section neutralization "
            "pressure. Zero disables this binding and leaves the run advisory."
        ),
    )
    parser.add_argument("--max-archive-byte-growth", type=int, default=None)
    parser.add_argument(
        "--byte-growth-admission-mode",
        choices=BYTE_GROWTH_ADMISSION_MODES,
        default="hard_cap",
        help=(
            "hard_cap rejects archive growth above --max-archive-byte-growth; "
            "rate_paid admits extra bytes only when the byte-pressured local "
            "objective still improves. False-authority training guard only."
        ),
    )
    parser.add_argument("--pose-slack", type=float, default=0.0)
    parser.add_argument("--seg-slack", type=float, default=0.0)
    parser.add_argument(
        "--component-guard-mode",
        choices=COMPONENT_GUARD_MODES,
        default="score_primary",
    )
    parser.add_argument("--pair-guard-min-score-improved-fraction", type=float, default=0.0)
    parser.add_argument("--pair-guard-max-pose-worsened-fraction", type=float, default=1.0)
    parser.add_argument(
        "--dynamic-range-repair-gains",
        default="",
        help=(
            "Comma-separated HF-decoder gain candidates to receiver-replay before "
            "the perturbation search, or 'auto' for the bounded default set."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    progress_callback = _build_progress_callback(args.progress_jsonl)

    result = run_snerv_scorer_loop_decoder_qat_smoke(
        n_pairs=args.n_pairs,
        levels=args.levels,
        wavelet=args.wavelet,
        target_bits_per_coeff=args.target_bits_per_coeff,
        pair_stride=args.pair_stride,
        start_pair=args.start_pair,
        upstream_dir=args.upstream_dir,
        video_path=args.video_path,
        device=args.device,
        step_map_bins=args.step_map_bins,
        snerv_spectra_preserving_adapter=args.snerv_spectra_preserving_adapter,
        snerv_fc_dim=args.snerv_fc_dim,
        snerv_emb_size=args.snerv_emb_size,
        snerv_patch_radius=args.snerv_patch_radius,
        snerv_mfu_scales=_parse_positive_int_csv(args.snerv_mfu_scales),
        snerv_hfr_gain=args.snerv_hfr_gain,
        snerv_temporal_context=args.snerv_temporal_context,
        snerv_temporal_mode=args.snerv_temporal_mode,
        lf_payload_codec=args.snerv_scorer_loop_lf_payload_codec,
        qat_bits=args.qat_bits,
        max_trials=args.max_trials,
        search_mode=args.search_mode,
        perturb_scale=args.perturb_scale,
        byte_pressure_multiplier=args.byte_pressure_multiplier,
        section_value_pressure_multiplier=args.section_value_pressure_multiplier,
        max_archive_byte_growth=args.max_archive_byte_growth,
        byte_growth_admission_mode=args.byte_growth_admission_mode,
        pose_slack=args.pose_slack,
        seg_slack=args.seg_slack,
        component_guard_mode=args.component_guard_mode,
        pair_guard_min_score_improved_fraction=(
            args.pair_guard_min_score_improved_fraction
        ),
        pair_guard_max_pose_worsened_fraction=(
            args.pair_guard_max_pose_worsened_fraction
        ),
        dynamic_range_repair_gains=_parse_dynamic_range_repair_gains(
            args.dynamic_range_repair_gains
        ),
        seed=args.seed,
        progress_callback=progress_callback,
    )

    out_path = Path(args.out) if args.out else _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result.as_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[SNeRV scorer-loop decoder/QAT smoke] false-authority")
    print(f"  n_pairs: {result.n_pairs}")
    print(f"  evaluations: {result.scorer_loop_evaluations}")
    print(f"  adapter: {result.snerv_model_size_adapter}")
    print(f"  mfu_scales: {list(result.snerv_mfu_scales)}")
    print(f"  hfr_gain: {result.snerv_hfr_gain:g}")
    print(f"  temporal_mode: {result.snerv_temporal_mode}")
    print(f"  decoder_feature_count: {result.decoder_feature_count}")
    print(f"  lf_payload_codec: {result.lf_payload_codec}")
    print(f"  baseline_score_linf: {result.baseline.score_linf}")
    print(f"  best_score_linf: {result.best.score_linf}")
    print(f"  component_guard_mode: {result.component_guard_mode}")
    print(f"  dynamic_range_repair_gains: {list(result.dynamic_range_repair_gains)}")
    print(f"  byte_pressure_multiplier: {result.byte_pressure_multiplier}")
    print(
        "  section_value_pressure_multiplier: "
        f"{result.section_value_pressure_multiplier}"
    )
    print(f"  max_archive_byte_growth: {result.max_archive_byte_growth}")
    print(f"  byte_growth_admission_mode: {result.byte_growth_admission_mode}")
    print(
        "  best_rate_aware_objective_linf: "
        f"{result.best.rate_aware_objective_linf}"
    )
    print(f"  accepted_improvement: {result.accepted_improvement}")
    print(f"  ready_for_pose_guard_gate: {result.ready_for_pose_guard_gate}")
    print(f"  ready_for_exact_eval_dispatch: {result.ready_for_exact_eval_dispatch}")
    if result.blockers:
        print(f"  blockers: {list(result.blockers)}")
    print(f"  wrote {out_path}")
    return 0


def _build_progress_callback(raw_path: str | None):
    if raw_path is None:
        return None
    progress_path = Path(raw_path)
    if not progress_path.is_absolute():
        progress_path = REPO_ROOT / progress_path
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def callback(row) -> None:
        payload = {
            "schema": "snerv_scorer_loop_decoder_qat_progress.v1",
            "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "axis_tag": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "row": row.as_jsonable(),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()

    return callback


def _parse_positive_int_csv(raw: str) -> tuple[int, ...]:
    values = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        value = int(text)
        if value < 1:
            raise ValueError("positive integer list values must be >= 1")
        values.append(value)
    if not values:
        raise ValueError("at least one positive integer is required")
    return tuple(values)


def _parse_dynamic_range_repair_gains(raw: str) -> tuple[float, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    if text.lower() == "auto":
        return DEFAULT_DYNAMIC_RANGE_REPAIR_GAINS
    values = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())
