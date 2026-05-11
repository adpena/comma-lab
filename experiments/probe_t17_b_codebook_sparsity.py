#!/usr/bin/env python3
"""T17-B probe scaffold — VQ codebook sparsity gate (Phase 2 pre-design).

This is a SCAFFOLD ONLY. It does not dispatch any GPU job, does not load
scorer networks, does not build a contest archive, and does not produce any
score claim. It emits a typed-atom row that the cathedral autopilot consumes.

Sister of :mod:`probe_t17_a_codebook_perplexity_smoke` — where T17-A guards
against codebook collapse (too few codes used), T17-B guards against the
DIAMETRIC failure: codebook *spread* without sparsity, where every code is
used roughly uniformly and per-frame entropy savings vanish under the prior.

Probe specification
-------------------
- **smoke mode only** (default 200 iterations, sparsity penalty enabled)
- monitors codebook usage entropy AND L1-style usage sparsity every 10 iters
- HARD GATE: refuses to recommend the main T17 dispatch if usage entropy
  saturates above ``log(K) - sparsity_floor`` (no sparsity emerged)
- estimated cost on Modal T4: ~$2.00 (15-20 minute smoke; operator-gated)

CLAUDE.md compliance tags
-------------------------
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``no_score_claim_only_predicted_band``
- ``no_kill_verdict``
- ``no_tmp_paths``
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

PROBE_SCHEMA = "tac_probe_t17_b_codebook_sparsity_v1"
PROBE_NAME = "t17_b_codebook_sparsity"
PROBE_LANE_ID = "lane_phase2_probes_t17ab_t18ab"
DEFAULT_SMOKE_ITERS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_CODEBOOK_SIZE = 512
DEFAULT_SPARSITY_FLOOR_BITS = 0.50  # entropy must be at most log2(K) - 0.5
DEFAULT_ESTIMATED_COST_USD = 2.00


@dataclass
class T17BSmokeConfig:
    smoke_iters: int = DEFAULT_SMOKE_ITERS
    batch_size: int = DEFAULT_BATCH_SIZE
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    sparsity_floor_bits: float = DEFAULT_SPARSITY_FLOOR_BITS
    estimated_cost_usd: float = DEFAULT_ESTIMATED_COST_USD
    operator_authorized: bool = False


@dataclass
class T17BTypedAtomRow:
    candidate_id: str
    family: str = "shared_vq_codebook_t17"
    probe_name: str = PROBE_NAME
    lane_id: str = PROBE_LANE_ID
    smoke_only: bool = True
    estimated_dispatch_cost_usd: float = DEFAULT_ESTIMATED_COST_USD
    predicted_score_delta: float = 0.0
    expected_information_gain: float = 0.25
    blockers: list[str] = field(default_factory=list)
    notes: str = ""
    schema: str = PROBE_SCHEMA
    evidence_grade: str = "[predicted; T17-B pre-design probe]"
    claude_md_compliance_tags: list[str] = field(default_factory=lambda: [
        "operator_gate_non_negotiable_at_every_dispatch",
        "no_score_claim_only_predicted_band",
        "no_kill_verdict",
        "smoke_only_estimated_le_2_dollar",
    ])


def usage_entropy_bits(counts: list[int]) -> float:
    """Compute Shannon entropy in bits from codebook usage counts (pure stdlib)."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def usage_entropy_ceiling_bits(codebook_size: int) -> float:
    """Maximum entropy for a codebook of size K is log2(K) bits."""
    if codebook_size <= 1:
        return 0.0
    return math.log2(codebook_size)


def assess_sparsity(
    counts: list[int],
    cfg: T17BSmokeConfig,
) -> tuple[bool, str]:
    """Return ``(passes_gate, reason)`` from final codebook usage counts."""
    if not counts:
        return False, "no codebook-usage counts recorded"
    h = usage_entropy_bits(counts)
    ceiling = usage_entropy_ceiling_bits(cfg.codebook_size)
    sparsity_target = ceiling - cfg.sparsity_floor_bits
    if h > sparsity_target:
        return False, (
            f"no sparsity emerged: usage entropy {h:.2f} bits > target "
            f"{sparsity_target:.2f} bits (ceiling {ceiling:.2f} - floor "
            f"{cfg.sparsity_floor_bits:.2f})"
        )
    return True, (
        f"sparsity gate passed: usage entropy {h:.2f} bits <= target "
        f"{sparsity_target:.2f} bits"
    )


def emit_typed_atom_row(
    cfg: T17BSmokeConfig,
    *,
    candidate_id: str | None = None,
    extra_notes: str = "",
    extra_blockers: list[str] | None = None,
) -> T17BTypedAtomRow:
    blockers = list(extra_blockers or [])
    if not cfg.operator_authorized:
        blockers.append("operator_authorization_required_for_dispatch")
    notes_parts = [
        f"smoke_iters={cfg.smoke_iters}",
        f"batch_size={cfg.batch_size}",
        f"codebook_size={cfg.codebook_size}",
        (
            f"sparsity_target_bits="
            f"{usage_entropy_ceiling_bits(cfg.codebook_size) - cfg.sparsity_floor_bits:.2f}"
        ),
        "NN-2 sister of T17-A (codebook sparsity emergence)",
    ]
    if extra_notes:
        notes_parts.append(extra_notes)
    return T17BTypedAtomRow(
        candidate_id=candidate_id or f"{PROBE_NAME}_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}",
        estimated_dispatch_cost_usd=cfg.estimated_cost_usd,
        blockers=blockers,
        notes=" | ".join(notes_parts),
    )


def serialize(row: T17BTypedAtomRow) -> dict[str, Any]:
    return dataclasses.asdict(row)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--smoke-iters", type=int, default=DEFAULT_SMOKE_ITERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--codebook-size", type=int, default=DEFAULT_CODEBOOK_SIZE)
    parser.add_argument("--sparsity-floor-bits", type=float,
                        default=DEFAULT_SPARSITY_FLOOR_BITS)
    parser.add_argument("--estimated-cost-usd", type=float,
                        default=DEFAULT_ESTIMATED_COST_USD)
    parser.add_argument("--operator-authorized", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.smoke_iters <= 0:
        print("probe_t17_b: --smoke-iters must be > 0", file=sys.stderr)
        return 2
    if args.sparsity_floor_bits <= 0:
        print("probe_t17_b: --sparsity-floor-bits must be > 0", file=sys.stderr)
        return 2

    cfg = T17BSmokeConfig(
        smoke_iters=args.smoke_iters,
        batch_size=args.batch_size,
        codebook_size=args.codebook_size,
        sparsity_floor_bits=args.sparsity_floor_bits,
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
