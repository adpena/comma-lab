#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Canonical PASS/FAIL gate for MLX-trained Path 3 candidate dispatch eligibility.

Wraps tools/measure_pr95_mlx_pytorch_actual_contest_score_difference.py
to lock the corrected #1258 methodology fix (2026-05-26 empirical anchor:
|S_MLX − S_PyTorch| = 0.000011, 72× smaller than PR110 frontier delta) into
permanent infrastructure.

Operational contract: any Path 3 candidate (DreamerV3 RSSM / Z7-Mamba-2 /
NSCS06 v8 chroma_lut / etc.) that's been MLX-trained and exported to a
contest archive via tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py
(#1257) MUST pass this gate before any paid CUDA dispatch is authorized.

Gate semantics:
  PASS: |S_MLX − S_PyTorch| < --gate-threshold-contest-units  (default 0.001)
  FAIL: otherwise

The default 0.001 threshold is 90× margin over the empirical anchor (0.000011)
and 1.3× the PR110 vs PR101 frontier delta (0.000789) — i.e., a candidate that
fails this gate would be UNABLE to faithfully measure even a frontier-tightening
score difference.

Exit codes:
  0 = PASS (candidate's MLX scorer faithfully predicts contest score)
  1 = FAIL (candidate's MLX scorer drift exceeds gate; do NOT dispatch)
  2 = CLI / measurement error

Per CLAUDE.md "MLX portable-local-substrate authority": output carries
axis_tag="[macOS-MLX research-signal]", score_claim=False, promotable=False,
ready_for_exact_eval_dispatch=False per Catalog #127/#192/#317/#341 +
canonical Provenance per Catalog #323.

Integration pattern for operator-authorize wrappers (Path 3 dispatch):

    # In scripts/operator_authorize_substrate_dreamer_v3_modal_t4_dispatch.sh:
    .venv/bin/python tools/gate_mlx_candidate_contest_equivalence.py \\
        --archive-zip "$CANDIDATE_ARCHIVE" \\
        --gate-threshold-contest-units 0.001 \\
        --output-json "$REPORT_DIR/equivalence_gate.json" \\
        || { echo "Gate FAIL — refusing paid CUDA dispatch"; exit 1; }
    # if gate PASSES, proceed to canonical operator_authorize.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SISTER_TOOL = REPO_ROOT / "tools" / "measure_pr95_mlx_pytorch_actual_contest_score_difference.py"
SCHEMA_VERSION = "mlx_candidate_contest_equivalence_gate_v1"

# Canonical empirical anchor per #1258 corrected closure footer (2026-05-26)
EMPIRICAL_ANCHOR_DRIFT = 0.000011  # |S_MLX − S_PyTorch| on PR95 hnerv_muon archive
PR110_FRONTIER_DELTA = 0.000789  # PR110 (0.192051) − PR101 (0.192840)
DEFAULT_GATE_THRESHOLD = 0.001  # 90× margin over empirical anchor; 1.3× frontier delta


def _hash_inputs(*paths: Path) -> str:
    """Hash candidate inputs for canonical Provenance inputs_sha256."""
    import hashlib
    h = hashlib.sha256()
    for p in paths:
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-zip",
        type=Path,
        required=True,
        help="Path to candidate's contest archive zip (must contain MLX-decodable state_dict + latents + meta).",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Path to ground-truth contest video (default = upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        help="Number of frame pairs to measure (default=100; matches empirical anchor; 600 = full contest window).",
    )
    parser.add_argument(
        "--gate-threshold-contest-units",
        type=float,
        default=DEFAULT_GATE_THRESHOLD,
        help=f"Gate threshold for |S_MLX − S_PyTorch|. Default {DEFAULT_GATE_THRESHOLD} = 90× margin over empirical anchor 0.000011.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON path for gate verdict.",
    )
    parser.add_argument(
        "--candidate-label",
        type=str,
        default="anonymous_mlx_candidate",
        help="Label for canonical Provenance + verdict row.",
    )
    args = parser.parse_args()

    if not args.archive_zip.is_file():
        print(f"[gate] ERROR: --archive-zip does not exist: {args.archive_zip}", file=sys.stderr)
        return 2
    if not args.video_path.is_file():
        print(f"[gate] ERROR: --video-path does not exist: {args.video_path}", file=sys.stderr)
        return 2
    if args.gate_threshold_contest_units <= 0:
        print(f"[gate] ERROR: --gate-threshold-contest-units must be > 0", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    # Run the canonical corrected measurement (sister tool)
    measurement_json = args.output_json.with_name(args.output_json.stem + "_measurement.json")
    print(f"[gate] Invoking canonical measurement: {SISTER_TOOL.name}")
    print(f"[gate]   archive: {args.archive_zip}")
    print(f"[gate]   threshold: |S_MLX − S_PyTorch| < {args.gate_threshold_contest_units}")
    cmd = [
        sys.executable,
        str(SISTER_TOOL),
        "--archive-zip", str(args.archive_zip),
        "--video-path", str(args.video_path),
        "--n-pairs", str(args.n_pairs),
        "--output-json", str(measurement_json),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"[gate] ERROR: measurement returned rc={result.returncode}", file=sys.stderr)
        return 2

    if not measurement_json.is_file():
        print(f"[gate] ERROR: measurement did not produce {measurement_json}", file=sys.stderr)
        return 2

    measurement = json.loads(measurement_json.read_text())
    actual_drift = float(measurement.get("actual_contest_score_difference", float("inf")))

    # Gate decision
    if actual_drift < args.gate_threshold_contest_units:
        verdict = "PASS"
        exit_code = 0
    else:
        verdict = "FAIL"
        exit_code = 1

    # Build canonical Provenance per Catalog #323
    from tac.provenance.builders import build_provenance_for_predicted
    from tac.provenance.validator import provenance_to_dict
    inputs_sha = _hash_inputs(args.archive_zip, args.video_path)
    prov = build_provenance_for_predicted(
        model_id=f"mlx_candidate_contest_equivalence_gate:{args.candidate_label}",
        inputs_sha256=inputs_sha,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="darwin_arm64_apple_silicon",
    )

    gate_verdict = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "actual_contest_score_difference": actual_drift,
        "gate_threshold_contest_units": args.gate_threshold_contest_units,
        "margin_below_threshold": args.gate_threshold_contest_units - actual_drift,
        "ratio_actual_vs_empirical_anchor": actual_drift / EMPIRICAL_ANCHOR_DRIFT if EMPIRICAL_ANCHOR_DRIFT > 0 else None,
        "ratio_actual_vs_pr110_frontier_delta": actual_drift / PR110_FRONTIER_DELTA,
        "candidate_label": args.candidate_label,
        "archive_zip": str(args.archive_zip),
        "video_path": str(args.video_path),
        "n_pairs": args.n_pairs,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "axis_tag": "[macOS-MLX research-signal]",
        "evidence_grade": "macOS-MLX-research-signal",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "blockers": [
            "macos_mlx_research_signal_not_contest_authority",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
            f"gate_threshold_is_drift_bound_not_contest_score_claim_threshold_{args.gate_threshold_contest_units}",
        ],
        "provenance": provenance_to_dict(prov),
        "canonical_anchor": {
            "empirical_anchor_drift": EMPIRICAL_ANCHOR_DRIFT,
            "pr110_vs_pr101_frontier_delta": PR110_FRONTIER_DELTA,
            "empirical_anchor_source_landing_memo": ".omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md",
        },
        "operator_routable_per_verdict": (
            "PROCEED: candidate's MLX scorer faithfully predicts contest score within gate threshold; "
            "Path 3 dispatch eligible for paid CUDA proof"
            if verdict == "PASS"
            else "REFUSE: candidate's MLX scorer drift exceeds gate threshold; do NOT dispatch to paid CUDA; "
            "audit candidate's MLX↔PyTorch decoder parity per #1251 + #1257 + #1258 corrected methodology"
        ),
        "measurement_artifact": str(measurement_json),
    }
    args.output_json.write_text(json.dumps(gate_verdict, indent=2))

    print()
    print("=== MLX CANDIDATE CONTEST-EQUIVALENCE GATE ===")
    print(f"  candidate: {args.candidate_label}")
    print(f"  actual drift: {actual_drift:.6f} contest-units")
    print(f"  threshold:    {args.gate_threshold_contest_units:.6f} contest-units")
    print(f"  margin:       {args.gate_threshold_contest_units - actual_drift:.6f}")
    print(f"  ratio vs empirical anchor (0.000011): {actual_drift / EMPIRICAL_ANCHOR_DRIFT:.2f}×")
    print(f"  ratio vs PR110 frontier delta (0.000789): {actual_drift / PR110_FRONTIER_DELTA:.4f}×")
    print(f"  VERDICT: {verdict}")
    print(f"  exit code: {exit_code}")
    print(f"  output: {args.output_json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
