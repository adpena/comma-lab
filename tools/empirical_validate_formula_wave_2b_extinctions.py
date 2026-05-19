#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Empirical validation smoke for Wave 2B formula extinctions.

Runs each of the 11 ``tac.formula_extinctions`` helpers against published
canonical anchors and verifies the helpers REPRODUCE the canonical values
(Goyal 5% / He 10% / Pascanu 1.0/5.0 / Smith 0.1 / Cover-Thomas qint
{1,2,3,4,5}-bit grid / Catalog #299 quota 400 / etc.).

Persists per-row results via
``tac.optimization.macos_cpu_advisory_signal.append_manifest_row_to_jsonl``
per Catalog #192 / #317.

Per CLAUDE.md "macOS auth eval is NOISE" + Catalog #192 / #317 non-negotiables
all artifacts are tagged ``[macOS-CPU advisory]`` with ``score_claim=False``
and ``promotion_eligible=False``.

Usage
-----
    .venv/bin/python tools/empirical_validate_formula_wave_2b_extinctions.py
    .venv/bin/python tools/empirical_validate_formula_wave_2b_extinctions.py --output-dir <dir>
    .venv/bin/python tools/empirical_validate_formula_wave_2b_extinctions.py --strict

Lane: ``lane_arbitrariness_extinction_wave_2b_path3_formula_batch_20260518``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.formula_extinctions import (  # noqa: E402
    canonical_warmup_steps,
    WarmupScheduleInput,
    canonical_validation_split,
    ValidationSplitInput,
    canonical_qint_max_grid_rd_proof,
    QintMaxGridInput,
    canonical_inflate_device_pin_metadata,
    InflateDevicePinInput,
    canonical_bayesian_aggregation_quorum,
    QuorumInput,
    canonical_early_stopping_patience,
    EarlyStoppingInput,
    canonical_inflate_py_loc_budget,
    LOCBudgetInput,
    canonical_gradient_clipping_norm,
    GradientClipInput,
    canonical_catalog_quota_from_preflight_budget,
    CatalogQuotaInput,
    canonical_lr_warmup_init_lr_factor,
    WarmupInitLRInput,
)
from tac.optimization.macos_cpu_advisory_signal import (  # noqa: E402
    append_manifest_row_to_jsonl,
    build_macos_cpu_advisory_signal_manifest,
    detect_macos_cpu_hardware_substrate,
)


@dataclass(frozen=True)
class RowVerdict:
    row_name: str
    canonical_value_expected: object
    canonical_value_recovered: object
    matches: bool
    citation: str
    notes: str = ""


def _check(name: str, expected: object, recovered: object, citation: str, *, tol: float = 0.0) -> RowVerdict:
    if isinstance(expected, float) and isinstance(recovered, float):
        matches = math.isclose(expected, recovered, abs_tol=tol or 1e-9, rel_tol=tol or 1e-9)
    else:
        matches = expected == recovered
    return RowVerdict(
        row_name=name,
        canonical_value_expected=expected,
        canonical_value_recovered=recovered,
        matches=matches,
        citation=citation,
    )


def _validate_row_1_goyal_warmup() -> RowVerdict:
    r = canonical_warmup_steps(WarmupScheduleInput(total_steps=1_200_000))
    return _check(
        "row1_goyal_5pct_warmup_1.2M_steps",
        60_000,
        r.solved_value,
        "Goyal et al 2017 arxiv:1706.02677 §2.2",
    )


def _validate_row_1_he_warmup() -> RowVerdict:
    r = canonical_warmup_steps(
        WarmupScheduleInput(total_steps=10_000, fraction_of_total=0.10)
    )
    return _check(
        "row1_he_10pct_warmup_10K_steps",
        1000,
        r.solved_value,
        "He et al 2016 arxiv:1512.03385 ResNet canonical",
    )


def _validate_row_2_stratified_kfold() -> RowVerdict:
    r = canonical_validation_split(ValidationSplitInput(total_chunks=10))
    return _check(
        "row2_bengio_stratified_kfold_10_chunks",
        (0, 5),
        r.solved_value,
        "Bengio 2012 arxiv:1206.5533 §2.2",
    )


def _validate_row_3_cover_thomas_grid() -> RowVerdict:
    r = canonical_qint_max_grid_rd_proof()
    return _check(
        "row3_cover_thomas_qint_grid_rd_optimal",
        True,
        r.intermediate_values["all_canonical"],
        "Cover-Thomas 1991 Ch.13",
    )


def _validate_row_4_inflate_device_pin() -> RowVerdict:
    r = canonical_inflate_device_pin_metadata(InflateDevicePinInput(
        device="cpu", score_axis="contest_cpu", linux_x86_64_compliant=True,
    ))
    return _check(
        "row4_inflate_device_pin_linux_cpu_to_contest_cpu",
        "[contest-CPU]",
        r.solved_value["score_axis_canonical_tag"],
        "Catalog #205 + A1 PR Council F1/F11 anchor",
    )


def _validate_row_5_bayesian_quorum() -> RowVerdict:
    r = canonical_bayesian_aggregation_quorum(QuorumInput(
        member_count=6, per_member_calibration=0.55, tier="T1",
    ))
    return _check(
        "row5_bayesian_quorum_T1_low_calibration_simple_majority",
        4,
        r.solved_value,
        "Surowiecki 2004 + Kemeny-Snell 1962",
    )


def _validate_row_7_prechelt_early_stopping() -> RowVerdict:
    r = canonical_early_stopping_patience(EarlyStoppingInput(
        val_loss_history=[0.1] * 12, window_size=10, patience_count=3, patience_counter=2,
    ))
    stop_now, counter = r.solved_value
    return _check(
        "row7_prechelt_early_stopping_K_consecutive_flat",
        (True, 3),
        (stop_now, counter),
        "Prechelt 1998 Tricks of the Trade Ch.II.5",
    )


def _validate_row_8_loc_budget() -> RowVerdict:
    r = canonical_inflate_py_loc_budget(LOCBudgetInput(
        loc=100, cyclomatic_complexity=5, external_dependencies=1,
    ))
    return _check(
        "row8_hnerv_l4_loc_budget_small_inflate_passes",
        True,
        r.intermediate_values["passes_30_sec_criterion"],
        "HNeRV parity L4 + Catalog #328 + McCabe 1976",
    )


def _validate_row_9_pascanu_rnn() -> RowVerdict:
    r = canonical_gradient_clipping_norm(GradientClipInput(architecture_class="rnn"))
    return _check(
        "row9_pascanu_2013_rnn_canonical_1.0",
        1.0,
        r.solved_value,
        "Pascanu+Mikolov+Bengio 2013 arxiv:1211.5063 §4.2",
    )


def _validate_row_9_pascanu_cnn() -> RowVerdict:
    r = canonical_gradient_clipping_norm(GradientClipInput(architecture_class="cnn"))
    return _check(
        "row9_pascanu_2013_cnn_canonical_5.0",
        5.0,
        r.solved_value,
        "Pascanu canonical CNN feed-forward stable default",
    )


def _validate_row_10_catalog_quota() -> RowVerdict:
    r = canonical_catalog_quota_from_preflight_budget()
    return _check(
        "row10_catalog_299_quota_from_30s_75ms_baseline",
        400,
        r.solved_value,
        "Catalog #299 + #184 (30s preflight budget)",
    )


def _validate_row_11_smith_init_lr() -> RowVerdict:
    r = canonical_lr_warmup_init_lr_factor(WarmupInitLRInput(base_lr=5e-4))
    return _check(
        "row11_smith_2017_warmup_init_lr_5e-4_to_5e-5",
        5e-5,
        r.solved_value,
        "Smith 2017 arxiv:1506.01186 §3.2",
        tol=1e-12,
    )


_ALL_VALIDATORS = (
    _validate_row_1_goyal_warmup,
    _validate_row_1_he_warmup,
    _validate_row_2_stratified_kfold,
    _validate_row_3_cover_thomas_grid,
    _validate_row_4_inflate_device_pin,
    _validate_row_5_bayesian_quorum,
    _validate_row_7_prechelt_early_stopping,
    _validate_row_8_loc_budget,
    _validate_row_9_pascanu_rnn,
    _validate_row_9_pascanu_cnn,
    _validate_row_10_catalog_quota,
    _validate_row_11_smith_init_lr,
)


def _emit_per_row_artifact(
    verdict: RowVerdict,
    *,
    run_id: str,
    output_dir: Path,
) -> dict[str, object]:
    """Persist a per-row JSON artifact + return a roll-up manifest row.

    The macOS-CPU advisory signal manifest helper is specialized for full
    score observations (d_seg / d_pose / archive_bytes). Our validation
    rows are pure canonical-recovery proofs (no score), so we write
    per-row JSON to ``output_dir`` and return the row for a roll-up
    manifest. Tagged ``[macOS-CPU advisory]`` per Catalog #192/#317 with
    ``score_claim=False`` / ``promotion_eligible=False``.
    """
    row: dict[str, object] = {
        "schema_version": "empirical_validate_formula_wave_2b_extinctions_v1",
        "row_name": verdict.row_name,
        "canonical_value_expected": repr(verdict.canonical_value_expected),
        "canonical_value_recovered": repr(verdict.canonical_value_recovered),
        "matches": verdict.matches,
        "citation": verdict.citation,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "evidence_grade": "macOS-CPU-advisory",
        "evidence_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "hardware_substrate": detect_macos_cpu_hardware_substrate(),
        "provenance": {
            "artifact_kind": "predicted_from_model",
            "source_path": "<canonical_recovery:tac.formula_extinctions>",
            "measurement_axis": "[macOS-CPU advisory]",
            "evidence_grade": "macOS-CPU-advisory",
            "promotion_eligible": False,
            "score_claim_valid": False,
            "canonical_helper_invocation": (
                "tools.empirical_validate_formula_wave_2b_extinctions._emit_per_row_artifact"
            ),
        },
    }
    per_row_path = output_dir / f"{verdict.row_name}.json"
    per_row_path.write_text(json.dumps(row, indent=2, sort_keys=True))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "results"
        / "empirical_validate_formula_wave_2b_20260518",
        help="Output directory for per-row JSON artifacts",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit rc=1 if any canonical recovery fails",
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "results"
        / "empirical_validate_formula_wave_2b_20260518"
        / "manifest.jsonl",
        help="Output JSONL path for macOS-CPU advisory manifest rows",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"wave_2b_formula_{int(time.time())}"
    print(f"# Wave 2B formula extinctions empirical validation (run_id={run_id})")
    print(f"# Output dir: {args.output_dir}")
    print(f"# Manifest JSONL: {args.manifest_jsonl}")
    print()

    verdicts: list[RowVerdict] = []
    manifest_rows: list[dict[str, object]] = []
    for v in _ALL_VALIDATORS:
        verdict = v()
        verdicts.append(verdict)
        status = "OK " if verdict.matches else "FAIL"
        print(
            f"  {status}  {verdict.row_name}: "
            f"expected={verdict.canonical_value_expected!r} "
            f"recovered={verdict.canonical_value_recovered!r}"
        )
        row = _emit_per_row_artifact(
            verdict,
            run_id=run_id,
            output_dir=args.output_dir,
        )
        manifest_rows.append(row)

    # Append every row to the canonical JSONL (one per call -> atomic appends).
    args.manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_jsonl.open("a") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    n = len(verdicts)
    matched = sum(1 for v in verdicts if v.matches)
    print()
    print(f"# Verdict: {matched}/{n} canonical recoveries verified")
    print(
        f"# All rows tagged [macOS-CPU advisory] per Catalog #192/#317; "
        f"score_claim=False; promotion_eligible=False"
    )

    if args.strict and matched != n:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
