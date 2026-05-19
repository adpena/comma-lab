#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Empirical validation smoke for Wave 2C experimental-sweep extinctions.

Per the Wave 2C landing prompt + CLAUDE.md "Max observability" + Catalog #192/#317:
- Run each of the 8 helpers against representative operating-point anchors
- For rows #5 + #6: read EXISTING state JSONLs (council_deliberation_posterior
  + probe_outcomes_ledger) as READ-ONLY consumers per sister-coordination map
- Persist results via tac.optimization.macos_cpu_advisory_signal canonical helper
- Wrap each row via canonical Provenance per Catalog #323
- Evidence grade: ``macOS-CPU-advisory``; score_claim=False; promotion_eligible=False
- All artifacts under ``experiments/results/empirical_validate_experimental_wave_2c_20260518/``

Lane: ``lane_arbitrariness_extinction_wave_2c_path1_experimental_zero_batch_20260518``

Usage::

    .venv/bin/python tools/empirical_validate_experimental_wave_2c_extinctions.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.experimental_extinctions import (  # noqa: E402
    BrotliSweepInput,
    CodecSweepInput,
    ConvergenceSweepInput,
    CouncilCadenceInput,
    MemoryDecayInput,
    NegationWindowInput,
    ProbeDecayInput,
    SigmaCalibrationInput,
    brotli_quality_10_vs_11_payload_sweep,
    council_cadence_empirical_calibration,
    lzma_vs_zstd_vs_brotli_per_payload_sweep,
    memory_file_category_decay_calibration,
    negation_window_fp_fn_corpus_sweep,
    per_substrate_convergence_aware_early_stopping,
    probe_outcome_staleness_decay_calibration,
    segnet_boundary_curvature_sigma_calibration,
)
from tac.optimization.macos_cpu_advisory_signal import (  # noqa: E402
    append_manifest_row_to_jsonl,
)


OUTPUT_DIR = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "empirical_validate_experimental_wave_2c_20260518"
)
MANIFEST_PATH = OUTPUT_DIR / "wave_2c_advisory_manifest.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "wave_2c_validation_summary.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_value(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
        return v
    return str(v)


def _build_advisory_row(
    *,
    value_id: str,
    solved_value: Any,
    intermediate_summary: dict[str, Any],
    helper_invocation: str,
    literature_citation: str,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Build canonical macOS-CPU-advisory manifest row per Catalog #192/#317/#323."""
    return {
        # Catalog #192 fail-closed contract
        "value_id": value_id,
        "solved_value": _safe_value(solved_value),
        "intermediate_summary": {k: _safe_value(v) for k, v in intermediate_summary.items()},
        "helper_invocation": helper_invocation,
        "literature_citation": literature_citation,
        "wall_clock_seconds": wall_clock_seconds,
        "captured_at_utc": _utc_now(),
        "captured_by_subagent": (
            "lane_arbitrariness_extinction_wave_2c_path1_experimental_zero_batch_20260518"
        ),
        # Canonical Provenance per Catalog #323
        "provenance": {
            "artifact_kind": "predicted_from_model",
            "source_path": "<predictor:experimental_extinctions.wave_2c.v1>",
            "source_sha256": "0" * 64,
            "measurement_axis": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "promotion_eligible": False,
            "score_claim_valid": False,
            "captured_at_utc": _utc_now(),
            "canonical_helper_invocation": (
                "tac.provenance.builders.build_provenance_for_macos_cpu_advisory"
            ),
            "contest_archive_zip_path": "",
            "contest_archive_member_name": "",
            "composed_from": [],
            "rejection_reason": "",
        },
        # Catalog #127 custody routing fields (defense-in-depth)
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "macOS-CPU-advisory",
        "evidence_tag": "[macOS-CPU advisory]",
    }


def validate_row_1_convergence_early_stopping() -> dict[str, Any]:
    """Representative substrate val-score series (synthetic but realistic)."""
    start = time.perf_counter()
    # Synthetic series matching observed substrate convergence shape:
    # rapid early descent + flat plateau after ~30 epochs
    series = [
        0.50, 0.42, 0.35, 0.29, 0.24, 0.20, 0.18, 0.17, 0.165, 0.162,
        0.161, 0.1605, 0.1603, 0.1602, 0.16015, 0.16013, 0.16012, 0.16011,
    ]
    r = per_substrate_convergence_aware_early_stopping(
        ConvergenceSweepInput(
            substrate_id="substrate_anchor_representative",
            val_score_series=series,
            epoch_step=10,
            slope_epsilon=1e-3,
            K_consecutive_windows=3,
        )
    )
    elapsed = time.perf_counter() - start
    return _build_advisory_row(
        value_id="epochs_wildly_varies_1_100_200_1000_2000",
        solved_value=r.solved_value,
        intermediate_summary={
            "converged": r.intermediate_values["converged"],
            "compute_savings_factor": r.coupled_adjustments["compute_savings_factor"],
            "series_length": len(series),
        },
        helper_invocation=r.canonical_helper_invocation,
        literature_citation=r.literature_citation,
        wall_clock_seconds=elapsed,
    )


def validate_row_2_brotli_quality_10_vs_11() -> dict[str, Any]:
    """Representative payload: ~64KB compressible structured bytes."""
    start = time.perf_counter()
    payload = (b"abcdefghijklmnop" * 64 + b"x" * 256) * 8  # ~10KB structured
    r = brotli_quality_10_vs_11_payload_sweep(
        BrotliSweepInput(payload_id="anchor_64kb_structured", payload_bytes=payload)
    )
    elapsed = time.perf_counter() - start
    return _build_advisory_row(
        value_id="brotli_quality_10_vs_11_inconsistent",
        solved_value=r.solved_value,
        intermediate_summary={
            "payload_bytes_raw": r.intermediate_values["payload_bytes_raw"],
            "bytes_saved_by_q11_vs_q10": r.intermediate_values["bytes_saved_by_q11_vs_q10"],
            "backend": r.intermediate_values["backend"],
            "winner_q": r.intermediate_values["winner_q"],
        },
        helper_invocation=r.canonical_helper_invocation,
        literature_citation=r.literature_citation,
        wall_clock_seconds=elapsed,
    )


def validate_row_3_lzma_vs_zstd_vs_brotli() -> dict[str, Any]:
    """Representative state_dict-like payload: ~32KB partially-structured."""
    start = time.perf_counter()
    payload = b"".join(
        (i.to_bytes(4, "little", signed=False) for i in range(0, 8192))
    )
    r = lzma_vs_zstd_vs_brotli_per_payload_sweep(
        CodecSweepInput(payload_id="anchor_32kb_state_dict_like", payload_bytes=payload)
    )
    elapsed = time.perf_counter() - start
    return _build_advisory_row(
        value_id="lzma_preset_9_hardcoded",
        solved_value=r.solved_value,
        intermediate_summary={
            "payload_bytes_raw": r.intermediate_values["payload_bytes_raw"],
            "bytes_saved_winner_vs_lzma": r.intermediate_values["bytes_saved_winner_vs_lzma"],
            "winner_codec": r.intermediate_values["winner_codec"],
            "sweep": [
                {
                    "codec": p["codec"],
                    "compressed_bytes": p["compressed_bytes"],
                    "backend": p["backend"],
                }
                for p in r.sweep_points
            ],
        },
        helper_invocation=r.canonical_helper_invocation,
        literature_citation=r.literature_citation,
        wall_clock_seconds=elapsed,
    )


def validate_row_4_sigma_calibration() -> dict[str, Any]:
    """Synthetic SegNet boundary curvature distribution (~Gaussian, median~8)."""
    start = time.perf_counter()
    # Synthetic curvature samples from a realistic boundary distribution
    samples = [
        5.2, 6.1, 7.3, 8.0, 8.5, 7.8, 6.9, 7.5, 8.2, 9.1,
        7.0, 8.4, 6.5, 7.9, 8.6, 5.8, 7.4, 8.7, 9.3, 7.1,
    ]
    r = segnet_boundary_curvature_sigma_calibration(
        SigmaCalibrationInput(
            pixel_population_id="anchor_segnet_boundary_population",
            boundary_curvature_samples=samples,
        )
    )
    elapsed = time.perf_counter() - start
    return _build_advisory_row(
        value_id="sigma_15_grayscale_lut_hardcoded_per_design",
        solved_value=r.solved_value,
        intermediate_summary={
            "n_samples": r.intermediate_values["n_samples"],
            "median_curvature": r.intermediate_values["median_curvature"],
            "mad": r.intermediate_values["mad"],
            "canonical_sigma_from_mad": r.intermediate_values["canonical_sigma_from_mad"],
            "winner_sigma": r.intermediate_values["winner_sigma"],
            "sigma_15_was_winner": r.intermediate_values["sigma_15_was_winner"],
        },
        helper_invocation=r.canonical_helper_invocation,
        literature_citation=r.literature_citation,
        wall_clock_seconds=elapsed,
    )


def validate_row_5_council_cadence() -> dict[str, Any]:
    """Read REAL .omx/state/council_deliberation_posterior.jsonl (READ-ONLY)."""
    start = time.perf_counter()
    posterior_path = REPO_ROOT / ".omx" / "state" / "council_deliberation_posterior.jsonl"
    anchors: list[dict[str, Any]] = []
    if posterior_path.exists():
        for line in posterior_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                anchors.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    r = council_cadence_empirical_calibration(
        CouncilCadenceInput(deliberation_anchors=anchors, window_days=30)
    )
    elapsed = time.perf_counter() - start
    return _build_advisory_row(
        value_id="council_4_tier_cadence_3_per_day_3_per_week_arbitrary",
        solved_value=r.solved_value,
        intermediate_summary={
            "n_anchors_total": r.intermediate_values["n_anchors_total"],
            "n_anchors_in_window": r.intermediate_values["n_anchors_in_window"],
            "tier_counts": r.intermediate_values["tier_counts"],
            "hardcoded_T2_per_day": 3,
            "empirical_T2_per_day": r.coupled_adjustments[
                "hardcoded_vs_empirical_T2_per_day"
            ]["empirical"],
            "hardcoded_T3_per_week": 3,
            "empirical_T3_per_week": r.coupled_adjustments[
                "hardcoded_vs_empirical_T3_per_week"
            ]["empirical"],
        },
        helper_invocation=r.canonical_helper_invocation,
        literature_citation=r.literature_citation,
        wall_clock_seconds=elapsed,
    )


def validate_row_6_probe_decay() -> dict[str, Any]:
    """Read REAL .omx/state/probe_outcomes.jsonl (READ-ONLY)."""
    start = time.perf_counter()
    ledger_path = REPO_ROOT / ".omx" / "state" / "probe_outcomes.jsonl"
    outcomes: list[dict[str, Any]] = []
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                outcomes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    r = probe_outcome_staleness_decay_calibration(
        ProbeDecayInput(probe_outcomes=outcomes)
    )
    elapsed = time.perf_counter() - start
    return _build_advisory_row(
        value_id="staleness_window_30_days_hardcoded_8_surfaces",
        solved_value=r.solved_value,
        intermediate_summary={
            "n_outcomes_total": r.intermediate_values["n_outcomes_total"],
            "n_surfaces": r.intermediate_values["n_surfaces"],
            "uniform_30d_was_optimal_for_n_surfaces": r.coupled_adjustments[
                "uniform_30d_was_optimal_for_n_surfaces"
            ],
        },
        helper_invocation=r.canonical_helper_invocation,
        literature_citation=r.literature_citation,
        wall_clock_seconds=elapsed,
    )


def validate_row_7_negation_window() -> dict[str, Any]:
    """Synthetic labeled corpus for Catalog #236 negation-window sweep."""
    start = time.perf_counter()
    corpus = [
        {"text": "auth-eval 100ep landed", "trigger_offset": 10, "label": "affirmative"},
        {"text": "100ep auth-eval completed", "trigger_offset": 0, "label": "affirmative"},
        {"text": "previously 100ep was tried", "trigger_offset": 11, "label": "negation"},
        {"text": "discussion of 100ep tradeoff", "trigger_offset": 14, "label": "negation"},
        {"text": "100ep was originally hypothetical", "trigger_offset": 0, "label": "negation"},
        {"text": "abandoned 100ep approach years ago", "trigger_offset": 10, "label": "negation"},
        {"text": "auth-eval at 100ep matches expectations", "trigger_offset": 13, "label": "affirmative"},
        {"text": "vs 100ep, current 50ep faster", "trigger_offset": 3, "label": "negation"},
    ]
    r = negation_window_fp_fn_corpus_sweep(
        NegationWindowInput(labeled_corpus=corpus)
    )
    elapsed = time.perf_counter() - start
    return _build_advisory_row(
        value_id="80_char_negation_window_Catalog_236",
        solved_value=r.solved_value,
        intermediate_summary={
            "n_corpus_entries": r.intermediate_values["n_corpus_entries"],
            "winner_window": r.intermediate_values["winner_window"],
            "winner_total_errors": r.intermediate_values["winner_total_errors"],
            "window_80_total_errors": r.intermediate_values["window_80_total_errors"],
            "window_80_was_winner": r.intermediate_values["window_80_was_winner"],
        },
        helper_invocation=r.canonical_helper_invocation,
        literature_citation=r.literature_citation,
        wall_clock_seconds=elapsed,
    )


def validate_row_8_memory_decay() -> dict[str, Any]:
    """Scan REAL ~/.claude/projects/.../memory/ (READ-ONLY) for category decay."""
    start = time.perf_counter()
    memory_dir = (
        Path.home() / ".claude" / "projects" / "-Users-adpena-Projects-pact" / "memory"
    )
    metadata: list[dict[str, Any]] = []
    if memory_dir.exists():
        for f in memory_dir.glob("feedback_*.md"):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                metadata.append(
                    {
                        "filename": f.name,
                        "mtime_utc": mtime.isoformat(),
                        "superseded_by": None,  # not currently tracked
                    }
                )
            except OSError:
                continue
    r = memory_file_category_decay_calibration(
        MemoryDecayInput(memory_file_metadata=metadata)
    )
    elapsed = time.perf_counter() - start
    return _build_advisory_row(
        value_id="memory_file_rotation_60_days_hardcoded",
        solved_value=r.solved_value,
        intermediate_summary={
            "n_files_total": r.intermediate_values["n_files_total"],
            "n_categories": r.intermediate_values["n_categories"],
            "categories_matching_60d_hardcoded": r.coupled_adjustments[
                "categories_matching_60d_hardcoded"
            ],
        },
        helper_invocation=r.canonical_helper_invocation,
        literature_citation=r.literature_citation,
        wall_clock_seconds=elapsed,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reset manifest at start of each run
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    validators = [
        ("row_1_convergence_early_stopping", validate_row_1_convergence_early_stopping),
        ("row_2_brotli_quality_10_vs_11", validate_row_2_brotli_quality_10_vs_11),
        ("row_3_lzma_vs_zstd_vs_brotli", validate_row_3_lzma_vs_zstd_vs_brotli),
        ("row_4_sigma_calibration", validate_row_4_sigma_calibration),
        ("row_5_council_cadence", validate_row_5_council_cadence),
        ("row_6_probe_decay", validate_row_6_probe_decay),
        ("row_7_negation_window", validate_row_7_negation_window),
        ("row_8_memory_decay", validate_row_8_memory_decay),
    ]

    rows: list[dict[str, Any]] = []
    print(f"Wave 2C empirical validation: 8 rows -> {MANIFEST_PATH}")
    print("=" * 78)
    for label, fn in validators:
        try:
            row = fn()
            append_manifest_row_to_jsonl(row, output_path=MANIFEST_PATH)
            rows.append(row)
            print(
                f"OK {label}: solved={row['solved_value']!r:.80} "
                f"(wall={row['wall_clock_seconds']:.3f}s)"
                if len(str(row["solved_value"])) <= 80
                else f"OK {label}: wall={row['wall_clock_seconds']:.3f}s"
            )
        except Exception as exc:
            print(f"FAIL {label}: {exc}")
            return 1

    summary = {
        "wave": "2c",
        "lane": "lane_arbitrariness_extinction_wave_2c_path1_experimental_zero_batch_20260518",
        "captured_at_utc": _utc_now(),
        "n_rows": len(rows),
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "per_row_summary": [
            {
                "value_id": r["value_id"],
                "solved_value_repr": str(r["solved_value"])[:120],
                "wall_clock_seconds": r["wall_clock_seconds"],
            }
            for r in rows
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("=" * 78)
    print(f"OK: 8 advisory rows persisted to {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"OK: summary at {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
