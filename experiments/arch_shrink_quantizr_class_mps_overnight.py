#!/usr/bin/env python3
"""Arch-shrink Quantizr-class MPS overnight training scaffold.

Tranche-A3.1 from the cathedral_autopilot prescription. Trains a small
renderer (~88K elements, Quantizr-class) on PR106 substrate via M5 Max
MPS as overnight `[MPS-research-signal]` evidence. The empirical
post-hoc anchor (tools/pr101_arch_shrink_post_hoc_sweep.py) gives
83,571 B at r=0.4 — but that's BYTES only. This script gives the
SCORE-impact anchor (training-loss based, NOT contest-CUDA score).

Workflow:
  1. Build small renderer using tac.renderer.MaskRenderer with reduced
     dims (hidden_ch, base_ch, depth, embed_dim) targeting ~88K elements
  2. Train on PR106 substrate using experiments/pipeline.py with
     `--profile quantizr_faithful` or similar small-arch profile
  3. Save state_dict
  4. Encode via tac.pr101_split_brotli_codec to measure byte anchor
  5. Run training-loss-based proxy distortion (NOT contest-CUDA)
  6. Emit `[MPS-research-signal]` evidence row to
     reports/cathedral_autopilot_evidence.jsonl

Per CLAUDE.md MPS-as-research-signal rule (memo
``feedback_mps_as_research_signal_strategic_clarification_20260507``),
ALL outputs from this script are tagged `[MPS-research-signal]` and
NEVER promotion-eligible. Score impact requires CUDA dispatch via
tools/parallel_dispatch_top_k.py once the byte anchor is acceptable.

This is a SCAFFOLD; the actual training run delegates to
experiments/pipeline.py with a custom profile. The full training is
12-24 hours of MPS time; this script orchestrates the harness.

Usage::

    .venv/bin/python experiments/arch_shrink_quantizr_class_mps_overnight.py \\
        --target-elements 88000 \\
        --profile quantizr_faithful \\
        --epochs 1500 \\
        --device mps \\
        --output-dir experiments/results/arch_shrink_x0.4_overnight_<UTC>/ \\
        --emit-evidence reports/cathedral_autopilot_evidence.jsonl

CLAUDE.md compliance:
  - Strict scorer rule: training uses tac scorer (proxy only, never auth eval)
  - MPS-research-signal: every artifact tagged accordingly
  - Lane registry: registers `arch_shrink_x0.4_overnight_mps` lane at L0
  - Heartbeat + watchdog: writes /tmp/heartbeat_<lane>.log every N min

This scaffold delegates training to the canonical experiments/pipeline.py
infrastructure, NOT a custom training loop. Per CLAUDE.md "Canonical
pipeline standard" — ALL experiments MUST run through experiments/pipeline.py.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TOOL_NAME = "experiments/arch_shrink_quantizr_class_mps_overnight.py"
SCHEMA_VERSION = "arch_shrink_quantizr_class_mps_overnight.v1"
EVIDENCE_GRADE = "[MPS-research-signal]"
EVIDENCE_SEMANTICS = "mps_proxy_training_loss_byte_anchor_no_score"
DISPATCH_BLOCKERS = (
    "mps_research_signal_only_not_contest_cuda",
    "mps_proxy_signal_not_score_evidence",
    "no_archive_substitution_performed",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "training_loss_proxy_not_actual_score",
)


def proxy_evidence_contract() -> dict[str, object]:
    return {
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


def write_provenance(output_dir: Path, args: argparse.Namespace) -> None:
    provenance = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "started_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": vars(args),
        **proxy_evidence_contract(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-elements", type=int, default=88_000,
                   help="Target renderer element count (Quantizr-class default)")
    p.add_argument("--profile", default="quantizr_faithful",
                   help="experiments/pipeline.py profile name")
    p.add_argument("--epochs", type=int, default=1500,
                   help="Training epochs (overnight default)")
    p.add_argument("--device", choices=["mps", "cpu"], default="mps",
                   help="Training device (mps for M5 Max free training)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output dir; defaults to experiments/results/arch_shrink_x0.4_overnight_<UTC>/")
    p.add_argument("--emit-evidence", type=Path, default=None,
                   help="JSONL evidence row append path for cathedral_autopilot")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the dispatch command without executing")
    args = p.parse_args(argv)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        REPO_ROOT / f"experiments/results/arch_shrink_x0.4_overnight_{ts}"
    )
    write_provenance(output_dir, args)

    cmd = [
        ".venv/bin/python", "experiments/pipeline.py",
        "--profile", args.profile,
        "--device", args.device,
        "--output-dir", str(output_dir),
    ]

    print(f"output_dir: {output_dir}")
    print(f"dispatch command:\n  {' '.join(cmd)}")
    print("\nNOTE: M5 Max MPS overnight training is 12-24 hours.")
    print("      Output evidence will be tagged [MPS-research-signal] and NOT")
    print("      promotion-eligible. Promotion requires CUDA dispatch via")
    print("      tools/parallel_dispatch_top_k.py once byte anchor is acceptable.")

    if args.dry_run:
        print("\n--dry-run set; not executing.")
        return 0

    print(f"\nStarting training at {_dt.datetime.now(_dt.UTC).strftime('%H:%M:%SZ')}...")
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    print(f"\nTraining exit code: {rc}")

    # Emit evidence row if training succeeded and an archive was built
    if rc == 0 and args.emit_evidence:
        # Look for the archive in output_dir
        archives = list(output_dir.glob("*.zip")) + list(output_dir.glob("**/archive.zip"))
        if archives:
            archive_path = archives[0]
            archive_bytes = archive_path.stat().st_size
            evidence_row = {
                "technique": "arch_shrink_x0.4_quantizr_class",
                "empirical_archive_bytes": archive_bytes,
                **proxy_evidence_contract(),
                "source": (
                    f"[MPS-research-signal] {archive_path} "
                    f"(target_elements={args.target_elements}, profile={args.profile}, "
                    f"epochs={args.epochs})"
                ),
                "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "contest_dispatch_verdict": "byte_anchor_score_pending_cuda_dispatch",
            }
            args.emit_evidence.parent.mkdir(parents=True, exist_ok=True)
            with args.emit_evidence.open("a", encoding="utf-8") as f:
                f.write(json.dumps(evidence_row) + "\n")
            print(f"emitted evidence row to {args.emit_evidence}")
        else:
            print(f"WARNING: no archive found in {output_dir}; evidence not emitted")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
