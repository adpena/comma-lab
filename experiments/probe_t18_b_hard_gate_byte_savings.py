#!/usr/bin/env python3
"""T18-B probe scaffold — Balle nonlinear-transform HARD GATE byte savings.

This is a SCAFFOLD ONLY. No GPU dispatch, no scorer load, no archive build,
no score claim. Emits a typed-atom row for the cathedral autopilot.

Per Phase 2 pre-design memo (2026-05-09; UNANIMOUS council vote on the HARD
GATE), T18 Balle 2018-style nonlinear transform on PR106 r2 latents only
earns dispatch authorization if it actually reduces archive bytes. T18-B
runs a smoke build that codes a small held-out latent slice with the trained
nonlinear transform + entropy bottleneck and compares the resulting byte
count against the PR106 r2 baseline coder. If net savings <= 0, the gate
REFUSES to recommend the main T18 dispatch — there's no point spending
~$43 on a full dispatch that the smoke says won't reduce bytes.

Probe specification
-------------------
- **smoke mode only** (codes a 64-frame slice; ~5 minute wall-clock)
- compares: ``baseline_bytes`` (PR106 r2 coder on slice) vs
  ``t18_bytes`` (trained Balle transform + entropy bottleneck on slice)
- HARD GATE: refuses dispatch if ``t18_bytes >= baseline_bytes`` (no savings)
  OR if savings ratio < ``minimum_savings_ratio`` (default 1%, council-set)
- estimated cost on Modal T4: ~$2.00 (full T18 dispatch is ~$43; this gate
  saves $41 on average across negative outcomes)

CLAUDE.md compliance tags
-------------------------
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``no_score_claim_only_predicted_band``
- ``no_kill_verdict``
- ``no_tmp_paths``
- ``hard_gate_byte_savings_unanimous_council``
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROBE_SCHEMA = "tac_probe_t18_b_hard_gate_byte_savings_v1"
PROBE_NAME = "t18_b_hard_gate_byte_savings"
PROBE_LANE_ID = "lane_phase2_probes_t17ab_t18ab"
DEFAULT_SLICE_FRAMES = 64
DEFAULT_MINIMUM_SAVINGS_RATIO = 0.01  # net savings >= 1% of baseline
DEFAULT_ESTIMATED_COST_USD = 2.00
DEFAULT_SAVED_BY_GATE_USD = 41.00  # avg saved on negative-outcome dispatch


@dataclass
class T18BHardGateConfig:
    slice_frames: int = DEFAULT_SLICE_FRAMES
    minimum_savings_ratio: float = DEFAULT_MINIMUM_SAVINGS_RATIO
    estimated_cost_usd: float = DEFAULT_ESTIMATED_COST_USD
    operator_authorized: bool = False


@dataclass
class ByteSavingsMeasurement:
    """Result of one smoke comparison: baseline vs T18-coded bytes on a slice."""

    baseline_bytes: int
    t18_bytes: int
    slice_frames: int

    @property
    def absolute_savings_bytes(self) -> int:
        return self.baseline_bytes - self.t18_bytes

    @property
    def savings_ratio(self) -> float:
        if self.baseline_bytes <= 0:
            return 0.0
        return self.absolute_savings_bytes / self.baseline_bytes


@dataclass
class T18BTypedAtomRow:
    candidate_id: str
    family: str = "balle_nonlinear_transform_t18"
    probe_name: str = PROBE_NAME
    lane_id: str = PROBE_LANE_ID
    smoke_only: bool = True
    estimated_dispatch_cost_usd: float = DEFAULT_ESTIMATED_COST_USD
    predicted_score_delta: float = 0.0
    expected_information_gain: float = 0.40
    blockers: list[str] = field(default_factory=list)
    notes: str = ""
    schema: str = PROBE_SCHEMA
    evidence_grade: str = "[predicted; T18-B HARD GATE pre-design probe]"
    claude_md_compliance_tags: list[str] = field(default_factory=lambda: [
        "operator_gate_non_negotiable_at_every_dispatch",
        "no_score_claim_only_predicted_band",
        "no_kill_verdict",
        "smoke_only_estimated_le_2_dollar",
        "hard_gate_byte_savings_unanimous_council",
    ])


def assess_hard_gate(
    measurement: ByteSavingsMeasurement,
    cfg: T18BHardGateConfig,
) -> tuple[bool, str]:
    """Return ``(passes_hard_gate, reason)``.

    Refuses ``main_t18_dispatch`` recommendation if the smoke slice does not
    show net byte savings at or above the council-set minimum ratio.
    """
    if measurement.baseline_bytes <= 0:
        return False, "baseline_bytes <= 0; cannot compute savings ratio"
    if measurement.t18_bytes >= measurement.baseline_bytes:
        return False, (
            f"HARD GATE refused: t18_bytes={measurement.t18_bytes} "
            f">= baseline_bytes={measurement.baseline_bytes} (no net savings)"
        )
    if measurement.savings_ratio < cfg.minimum_savings_ratio:
        return False, (
            f"HARD GATE refused: savings ratio {measurement.savings_ratio:.4f} "
            f"< minimum {cfg.minimum_savings_ratio:.4f} (council-set floor)"
        )
    return True, (
        f"HARD GATE passed: savings ratio {measurement.savings_ratio:.4f} "
        f"({measurement.absolute_savings_bytes} bytes saved on "
        f"{measurement.slice_frames}-frame slice)"
    )


def emit_typed_atom_row(
    cfg: T18BHardGateConfig,
    *,
    candidate_id: str | None = None,
    extra_notes: str = "",
    extra_blockers: list[str] | None = None,
) -> T18BTypedAtomRow:
    blockers = list(extra_blockers or [])
    if not cfg.operator_authorized:
        blockers.append("operator_authorization_required_for_dispatch")
    notes_parts = [
        f"slice_frames={cfg.slice_frames}",
        f"minimum_savings_ratio={cfg.minimum_savings_ratio:.4f}",
        f"hard_gate_avg_dispatch_savings_usd={DEFAULT_SAVED_BY_GATE_USD:.2f}",
        "HARD GATE per UNANIMOUS Phase 2 pre-design council vote",
    ]
    if extra_notes:
        notes_parts.append(extra_notes)
    return T18BTypedAtomRow(
        candidate_id=candidate_id or f"{PROBE_NAME}_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}",
        estimated_dispatch_cost_usd=cfg.estimated_cost_usd,
        blockers=blockers,
        notes=" | ".join(notes_parts),
    )


def serialize(row: T18BTypedAtomRow) -> dict[str, Any]:
    return dataclasses.asdict(row)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slice-frames", type=int, default=DEFAULT_SLICE_FRAMES)
    parser.add_argument("--minimum-savings-ratio", type=float,
                        default=DEFAULT_MINIMUM_SAVINGS_RATIO)
    parser.add_argument("--estimated-cost-usd", type=float,
                        default=DEFAULT_ESTIMATED_COST_USD)
    parser.add_argument("--operator-authorized", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.slice_frames <= 0:
        print("probe_t18_b: --slice-frames must be > 0", file=sys.stderr)
        return 2
    if not (0.0 < args.minimum_savings_ratio < 1.0):
        print("probe_t18_b: --minimum-savings-ratio must be in (0,1)", file=sys.stderr)
        return 2

    cfg = T18BHardGateConfig(
        slice_frames=args.slice_frames,
        minimum_savings_ratio=args.minimum_savings_ratio,
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
