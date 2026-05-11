#!/usr/bin/env python3
"""T18-A probe scaffold — Balle nonlinear-transform invertibility smoke.

This is a SCAFFOLD ONLY. No GPU dispatch, no scorer load, no archive build,
no score claim. Emits a typed-atom row for the cathedral autopilot.

Per Phase 2 pre-design memo (2026-05-09; Balle/Selfcomp/MacKay/Boyd/Hotz),
T18 (Balle 2018-style nonlinear-transform code on PR106 r2 latents) requires
that ``invert(forward(z_e)) approx z_e`` to within numerical noise. If the
GDN/IGDN nonlinearities saturate or the entropy bottleneck loses too many
bits, the inverse drifts and downstream pose/seg distortion explodes.

Probe specification
-------------------
- **smoke mode only** (default 200 iters; small batch; CPU or T4)
- monitors residual ``||z_e - invert(forward(z_e))||^2`` every 10 iters
- HARD GATE: refuses to recommend the main T18 dispatch if residual L2
  exceeds ``invertibility_floor`` (default 0.5) sustained over 50+ iters
- estimated cost on Modal T4: ~$2.00 (15-20 min smoke; operator-gated)

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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROBE_SCHEMA = "tac_probe_t18_a_invertibility_smoke_v1"
PROBE_NAME = "t18_a_invertibility_smoke"
PROBE_LANE_ID = "lane_phase2_probes_t17ab_t18ab"
DEFAULT_SMOKE_ITERS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_INVERTIBILITY_FLOOR = 0.50  # ||z_e - invert(forward(z_e))||^2 floor
DEFAULT_SUSTAIN_WINDOW = 50  # iters of sustained breach to fail the gate
DEFAULT_ESTIMATED_COST_USD = 2.00


@dataclass
class T18ASmokeConfig:
    smoke_iters: int = DEFAULT_SMOKE_ITERS
    batch_size: int = DEFAULT_BATCH_SIZE
    invertibility_floor: float = DEFAULT_INVERTIBILITY_FLOOR
    sustain_window: int = DEFAULT_SUSTAIN_WINDOW
    estimated_cost_usd: float = DEFAULT_ESTIMATED_COST_USD
    operator_authorized: bool = False


@dataclass
class T18ATypedAtomRow:
    candidate_id: str
    family: str = "balle_nonlinear_transform_t18"
    probe_name: str = PROBE_NAME
    lane_id: str = PROBE_LANE_ID
    smoke_only: bool = True
    estimated_dispatch_cost_usd: float = DEFAULT_ESTIMATED_COST_USD
    predicted_score_delta: float = 0.0
    expected_information_gain: float = 0.30
    blockers: list[str] = field(default_factory=list)
    notes: str = ""
    schema: str = PROBE_SCHEMA
    evidence_grade: str = "[predicted; T18-A pre-design probe]"
    claude_md_compliance_tags: list[str] = field(default_factory=lambda: [
        "operator_gate_non_negotiable_at_every_dispatch",
        "no_score_claim_only_predicted_band",
        "no_kill_verdict",
        "smoke_only_estimated_le_2_dollar",
    ])


def assess_invertibility(
    residual_trace: list[float],
    cfg: T18ASmokeConfig,
) -> tuple[bool, str]:
    """Return ``(passes_gate, reason)`` for an invertibility residual trace.

    The gate fails if the residual L2 exceeds ``invertibility_floor`` for
    every step in any sliding window of ``sustain_window`` iters. A single
    spike does not fail the gate (it could be a finite-precision step).
    """
    if not residual_trace:
        return False, "no invertibility-residual samples were recorded"
    over_floor = [r > cfg.invertibility_floor for r in residual_trace]
    sustained = False
    sustained_at = -1
    for i in range(len(over_floor) - cfg.sustain_window + 1):
        window = over_floor[i: i + cfg.sustain_window]
        if all(window):
            sustained = True
            sustained_at = i
            break
    if sustained:
        return False, (
            f"NN-3 invertibility breach: residual L2 stayed above floor "
            f"{cfg.invertibility_floor:.3f} for {cfg.sustain_window} sustained "
            f"iters starting at step {sustained_at}"
        )
    max_obs = max(residual_trace)
    return True, (
        f"invertibility gate passed: max residual {max_obs:.3f} "
        f"<= floor {cfg.invertibility_floor:.3f} OR breach not sustained "
        f"for {cfg.sustain_window} iters"
    )


def emit_typed_atom_row(
    cfg: T18ASmokeConfig,
    *,
    candidate_id: str | None = None,
    extra_notes: str = "",
    extra_blockers: list[str] | None = None,
) -> T18ATypedAtomRow:
    blockers = list(extra_blockers or [])
    if not cfg.operator_authorized:
        blockers.append("operator_authorization_required_for_dispatch")
    notes_parts = [
        f"smoke_iters={cfg.smoke_iters}",
        f"batch_size={cfg.batch_size}",
        f"invertibility_floor={cfg.invertibility_floor:.3f}",
        f"sustain_window={cfg.sustain_window}",
        "NN-3 invertibility sentinel per Phase 2 pre-design",
    ]
    if extra_notes:
        notes_parts.append(extra_notes)
    return T18ATypedAtomRow(
        candidate_id=candidate_id or f"{PROBE_NAME}_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}",
        estimated_dispatch_cost_usd=cfg.estimated_cost_usd,
        blockers=blockers,
        notes=" | ".join(notes_parts),
    )


def serialize(row: T18ATypedAtomRow) -> dict[str, Any]:
    return dataclasses.asdict(row)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--smoke-iters", type=int, default=DEFAULT_SMOKE_ITERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--invertibility-floor", type=float,
                        default=DEFAULT_INVERTIBILITY_FLOOR)
    parser.add_argument("--sustain-window", type=int,
                        default=DEFAULT_SUSTAIN_WINDOW)
    parser.add_argument("--estimated-cost-usd", type=float,
                        default=DEFAULT_ESTIMATED_COST_USD)
    parser.add_argument("--operator-authorized", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.smoke_iters <= 0:
        print("probe_t18_a: --smoke-iters must be > 0", file=sys.stderr)
        return 2
    if args.invertibility_floor <= 0:
        print("probe_t18_a: --invertibility-floor must be > 0", file=sys.stderr)
        return 2

    cfg = T18ASmokeConfig(
        smoke_iters=args.smoke_iters,
        batch_size=args.batch_size,
        invertibility_floor=args.invertibility_floor,
        sustain_window=args.sustain_window,
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
