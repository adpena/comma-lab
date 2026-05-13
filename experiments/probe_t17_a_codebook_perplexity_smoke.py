#!/usr/bin/env python3
"""T17-A probe scaffold — VQ codebook perplexity smoke (Phase 2 pre-design).

This is a SCAFFOLD ONLY. It does not dispatch any GPU job, does not load
scorer networks, does not build a contest archive, and does not produce any
score claim. It emits a typed-atom row that the cathedral autopilot consumes
to surface the probe to the operator for explicit dispatch authorization.

Per the Phase 2 pre-design memo (2026-05-09; Yousfi/Selfcomp/MacKay/Hotz/
van den Oord council), T17 (shared VQ codebook on per-frame latents) carries
codebook-collapse risk: when the codebook is too large or learning rate too
high, training collapses to a small subset of codes (perplexity << K) and
the per-frame entropy savings vanish. The T17-A probe is the sentinel that
fires BEFORE the main T17 dispatch.

Probe specification
-------------------
- **smoke mode only** (default 200 iterations / 1 epoch / 64-frame minibatch)
- monitors codebook perplexity = exp(H(p_code)) every 10 iterations
- HARD GATE: refuses to recommend the main T17 dispatch if smoke perplexity
  drops below 0.25 * K within the first 100 iterations (codebook collapse)
- estimated cost on Modal T4: ~$2.00 (15-20 minute smoke; the operator must
  explicitly authorize the dispatch via the cathedral autopilot)

CLAUDE.md compliance tags
-------------------------
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``no_score_claim_only_predicted_band``
- ``no_kill_verdict``
- ``no_tmp_paths``
- ``forbidden_premature_kill_without_research_exhaustion``
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROBE_SCHEMA = "tac_probe_t17_a_codebook_perplexity_smoke_v1"
PROBE_NAME = "t17_a_codebook_perplexity_smoke"
PROBE_LANE_ID = "lane_phase2_probes_t17ab_t18ab"
DEFAULT_SMOKE_ITERS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_CODEBOOK_SIZE = 512
DEFAULT_PERPLEXITY_GATE_RATIO = 0.25
DEFAULT_ESTIMATED_COST_USD = 2.00


@dataclass
class T17ASmokeConfig:
    """Smoke-mode configuration for the T17-A probe."""

    smoke_iters: int = DEFAULT_SMOKE_ITERS
    batch_size: int = DEFAULT_BATCH_SIZE
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    perplexity_gate_ratio: float = DEFAULT_PERPLEXITY_GATE_RATIO
    estimated_cost_usd: float = DEFAULT_ESTIMATED_COST_USD
    operator_authorized: bool = False  # explicit operator opt-in required


@dataclass
class T17ATypedAtomRow:
    """One typed-atom row that the cathedral autopilot consumes."""

    candidate_id: str
    family: str = "shared_vq_codebook_t17"
    probe_name: str = PROBE_NAME
    lane_id: str = PROBE_LANE_ID
    smoke_only: bool = True
    estimated_dispatch_cost_usd: float = DEFAULT_ESTIMATED_COST_USD
    predicted_score_delta: float = 0.0  # smoke probe; no predicted score claim
    expected_information_gain: float = 0.30  # high EIG for codebook-collapse risk
    blockers: list[str] = field(default_factory=list)
    notes: str = ""
    schema: str = PROBE_SCHEMA
    evidence_grade: str = "[predicted; T17-A pre-design probe]"
    claude_md_compliance_tags: list[str] = field(default_factory=lambda: [
        "operator_gate_non_negotiable_at_every_dispatch",
        "no_score_claim_only_predicted_band",
        "no_kill_verdict",
        "smoke_only_estimated_le_2_dollar",
    ])


def perplexity_gate_threshold(cfg: T17ASmokeConfig) -> float:
    """Return the codebook-collapse perplexity threshold."""
    return cfg.perplexity_gate_ratio * cfg.codebook_size


def assess_perplexity_trajectory(
    perplexities: list[float],
    cfg: T17ASmokeConfig,
    *,
    early_window_iters: int = 100,
) -> tuple[bool, str]:
    """Return ``(passes_gate, reason)`` based on a measured perplexity trace.

    The smoke gate fires if the perplexity drops below
    ``perplexity_gate_ratio * codebook_size`` at any point within the
    first ``early_window_iters`` iterations of the smoke run.
    """
    if not perplexities:
        return False, "no perplexity samples were recorded"
    threshold = perplexity_gate_threshold(cfg)
    early_window = perplexities[: early_window_iters // 10 + 1]
    min_early = min(early_window) if early_window else float("inf")
    if min_early < threshold:
        return False, (
            f"codebook collapse detected: min perplexity in early window "
            f"{min_early:.2f} < threshold {threshold:.2f} "
            f"({cfg.perplexity_gate_ratio:.0%} * K={cfg.codebook_size})"
        )
    return True, (
        f"perplexity stayed above threshold {threshold:.2f} in early window "
        f"(min observed {min_early:.2f})"
    )


def codebook_perplexity_from_counts(counts: list[int]) -> float:
    """Pure helper — compute perplexity = exp(H(p)) from codebook usage counts.

    No GPU, no torch — pure stdlib so unit tests cover the math.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log(p)
    return math.exp(entropy)


def emit_typed_atom_row(
    cfg: T17ASmokeConfig,
    *,
    candidate_id: str | None = None,
    extra_notes: str = "",
    extra_blockers: list[str] | None = None,
) -> T17ATypedAtomRow:
    """Build the typed-atom row consumed by the cathedral autopilot."""
    blockers: list[str] = list(extra_blockers or [])
    if not cfg.operator_authorized:
        blockers.append("operator_authorization_required_for_dispatch")
    notes_parts = [
        f"smoke_iters={cfg.smoke_iters}",
        f"batch_size={cfg.batch_size}",
        f"codebook_size={cfg.codebook_size}",
        (
            f"perplexity_collapse_threshold="
            f"{perplexity_gate_threshold(cfg):.2f}"
        ),
        "NN-2 sentinel (codebook collapse) per Phase 2 pre-design",
    ]
    if extra_notes:
        notes_parts.append(extra_notes)
    return T17ATypedAtomRow(
        candidate_id=candidate_id or f"{PROBE_NAME}_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}",
        estimated_dispatch_cost_usd=cfg.estimated_cost_usd,
        blockers=blockers,
        notes=" | ".join(notes_parts),
    )


def serialize(row: T17ATypedAtomRow) -> dict[str, Any]:
    return dataclasses.asdict(row)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--smoke-iters", type=int, default=DEFAULT_SMOKE_ITERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--codebook-size", type=int, default=DEFAULT_CODEBOOK_SIZE)
    parser.add_argument(
        "--perplexity-gate-ratio", type=float,
        default=DEFAULT_PERPLEXITY_GATE_RATIO,
        help="Perplexity collapse threshold = ratio * K",
    )
    parser.add_argument(
        "--estimated-cost-usd", type=float,
        default=DEFAULT_ESTIMATED_COST_USD,
    )
    parser.add_argument(
        "--operator-authorized", action="store_true",
        help="OPT-IN: operator has explicitly authorized this probe dispatch",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the typed-atom row JSON")
    args = parser.parse_args(argv)

    if args.smoke_iters <= 0:
        print("probe_t17_a: --smoke-iters must be > 0", file=sys.stderr)
        return 2
    if not (0.0 < args.perplexity_gate_ratio < 1.0):
        print("probe_t17_a: --perplexity-gate-ratio must be in (0,1)", file=sys.stderr)
        return 2

    cfg = T17ASmokeConfig(
        smoke_iters=args.smoke_iters,
        batch_size=args.batch_size,
        codebook_size=args.codebook_size,
        perplexity_gate_ratio=args.perplexity_gate_ratio,
        estimated_cost_usd=args.estimated_cost_usd,
        operator_authorized=args.operator_authorized,
    )
    row = emit_typed_atom_row(cfg)
    payload = serialize(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
