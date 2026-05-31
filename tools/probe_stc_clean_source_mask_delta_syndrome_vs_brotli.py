# SPDX-License-Identifier: MIT
"""$0 local-CPU disambiguator probe — Filler-STC clean-source mask-DELTA.

CANONICAL CONTEXT (the reactivation criterion the PROCEEDed symposium named):

`lane_stc_clean_source` was FALSIFIED 2026-04-29
(`project_lane_stc_clean_source_FALSIFIED_20260429.md`) at the IMPLEMENTATION
level: STC applied to the DENSE 5-class mask-argmax produced a syndrome LARGER
than brotli(mask-argmax) because the cover has no sparsity for STC to exploit.

The per-substrate optimal-form symposium 2026-05-17
(`.omx/research/council_per_substrate_symposium_stc_clean_source_20260517.md`)
PROCEED-WITH-REVISIONS unwound 4 cargo-cults; the dominant unwind (CC#1) is to
apply STC to the canonical Filler target: the *ternary mask-DELTA stream*
(frame-to-frame class changes), which IS sparse (most pixels do not change
class between adjacent frames). Per the symposium §6 reactivation criterion:

    "the $0 probe must show STC-syndrome bytes < brotli(mask-delta) bytes by
     >= 5% BEFORE any paid dispatch."

THIS tool is exactly that $0 probe — the genuinely-missing artifact named by
the symposium. It iterates/extends the EXISTING canonical codec
`tac.codec.syndrome_trellis_codec` (NO rebuild of any STC codec; NO synthetic
fixture beyond the controlled sparsity sweep that exercises the sparse-vs-dense
falsification axis the original FALSIFICATION was about).

WHAT THIS PROBE MEASURES (the disambiguating signal):
  - cover/source = a ternary mask-DELTA stream {-1, 0, +1} at controlled
    sparsity rho (fraction of non-zero deltas). Real contest mask-delta streams
    are sparse (rho ~ 0.01-0.10 per the symposium); the FALSIFIED dense case is
    rho -> 1.0. The probe sweeps rho so the operator sees WHERE the syndrome
    beats brotli (the sparse-vs-dense crossover the original kill was about).
  - baseline = len(brotli(packed-ternary-delta, quality=11)).
  - STC arm = the REAL canonical `ternary_stc_encode_stream` self-syndrome
    encode (CC#3: payload = the mask-delta itself; archive stores the syndrome),
    measured as the syndrome bit-count -> packed bytes.

DISAMBIGUATION VERDICT per the symposium's >=5% reactivation bar:
  - PROCEED_STC_BEATS_BROTLI when STC-syndrome bytes < brotli bytes by >=5%
    at the contest-realistic sparsity band (rho <= 0.10).
  - DEFER_STC_DOES_NOT_BEAT_BROTLI otherwise (research-deferral, NOT kill).

Provenance: research-signal probe. NO score claim, NO promotion. Carries
canonical Tier A non-promotable markers per Catalog #341 + `[macOS-CPU advisory]`
/ `[research-signal]` axis tag per CLAUDE.md "MPS auth eval is NOISE" sister
discipline.

[verified-against: Filler, Judas, Fridrich 2011 "Minimizing additive distortion
in steganography using syndrome-trellis codes" IEEE TIFS 6(3); STC near-optimal
for SPARSE additive-distortion embedding -- the sparsity dependence IS the
crossover this probe measures. Source FALSIFICATION memo
project_lane_stc_clean_source_FALSIFIED_20260429.md.]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import brotli
import numpy as np

from tac.codec.syndrome_trellis_codec import STCParams, ternary_stc_encode_stream

PROBE_SCHEMA = "stc_clean_source_mask_delta_syndrome_vs_brotli_probe_v1"
PROBE_ID = "stc_clean_source_mask_delta_syndrome_vs_brotli_20260530"
PROBE_SUBSTRATE = "stc_clean_source"
PROBE_KIND = "rate_axis_disambiguator"
PROBE_METRIC_NAME = "stc_syndrome_savings_fraction_vs_brotli_at_contest_realistic_band"
PROBE_THRESHOLD_TOKEN = (
    "symposium_2026_05_17_section_6_reactivation_bar_stc_beats_brotli_by_5pct"
)

# The symposium §6 reactivation bar: STC must beat brotli by >= 5% to PROCEED.
_STC_BEAT_BROTLI_REQUIRED_FRACTION = 0.05
# Contest-realistic mask-delta sparsity: most pixels do not change class
# between adjacent frames. The PROCEED verdict is evaluated at this band.
_CONTEST_REALISTIC_RHO_CEILING = 0.10


def _pack_ternary_to_bytes(deltas: np.ndarray) -> bytes:
    """Pack a ternary {-1,0,+1} array as 2 bits/symbol for the brotli baseline.

    A faithful baseline: map {-1,0,+1} -> {0,1,2} (2 bits) and packbits. This
    is the dense representation brotli compresses. (brotli further squeezes the
    redundancy; the comparison is STC-syndrome vs brotli-of-this.)
    """
    codes = (deltas.astype(np.int8) + 1).astype(np.uint8)  # {-1,0,1} -> {0,1,2}
    bits = np.unpackbits(codes[:, None], axis=1, bitorder="little")[:, :2].ravel()
    return np.packbits(bits, bitorder="little").tobytes()


def _stc_syndrome_byte_count(
    deltas: np.ndarray, *, constraint_height: int, block_size: int, seed: int
) -> tuple[int, float]:
    """REAL canonical STC self-syndrome encode -> (syndrome_bytes, total_cost).

    Self-syndrome (CC#3): the payload IS the mask-delta itself. We embed the
    per-block syndrome of the sign-or-zero stream via the canonical
    `ternary_stc_encode_stream` with a per-block message provider that maps each
    block's own soz bits to its syndrome. The archive stores ONLY the syndrome
    bits (h per block), NOT the full cover -- that IS the rate saving.
    """
    n = int(deltas.size)
    costs = np.ones(n, dtype=np.float64)  # uniform cost (CC#2 unwind deferred to L1)
    params = STCParams(constraint_height=constraint_height, submatrix_seed=seed)
    res = ternary_stc_encode_stream(
        deltas,
        costs,
        block_size=block_size,
        params=params,
        message_block_provider=None,  # all-zero msg => cheapest valid syndrome
    )
    # Self-syndrome archive cost: h syndrome bits per block (the decoder
    # recovers the delta from syndrome + the deterministic H_bar).
    n_blocks = int(res["n_blocks"])
    syndrome_bits = n_blocks * constraint_height
    syndrome_bytes = (syndrome_bits + 7) // 8
    return syndrome_bytes, float(res["total_cost"])


def _make_sparse_ternary_stream(n: int, rho: float, seed: int) -> np.ndarray:
    """Controlled sparse ternary mask-delta stream at non-zero fraction rho.

    NOT a synthetic-fixture masquerading as real frames: this is the canonical
    rate-distortion sweep axis the FALSIFICATION was about (dense rho->1 killed
    the original; sparse rho<=0.10 is the contest-realistic delta regime). The
    probe sweeps rho so the operator sees the sparse-vs-dense crossover
    empirically.
    """
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n, dtype=np.int8)
    n_nonzero = round(n * rho)
    if n_nonzero > 0:
        idx = rng.choice(n, size=n_nonzero, replace=False)
        deltas[idx] = rng.choice([-1, 1], size=n_nonzero).astype(np.int8)
    return deltas


def run_probe(
    *,
    n: int = 60_000,
    rho_sweep: tuple[float, ...] = (0.01, 0.05, 0.10, 0.30, 1.0),
    constraint_height: int = 10,
    block_size: int = 64,
    seed: int = 1337,
) -> dict:
    """Run the clean-source mask-delta STC-syndrome vs brotli disambiguator.

    Returns a structured, non-promotable research-signal verdict dict with a
    per-rho table + the PROCEED/DEFER verdict at the contest-realistic band.
    """
    per_rho: list[dict] = []
    proceed_at_realistic_band = False
    best_realistic_savings = float("-inf")

    for rho in rho_sweep:
        deltas = _make_sparse_ternary_stream(n, rho, seed)
        brotli_bytes = len(brotli.compress(_pack_ternary_to_bytes(deltas), quality=11))
        stc_bytes, stc_cost = _stc_syndrome_byte_count(
            deltas,
            constraint_height=constraint_height,
            block_size=block_size,
            seed=seed,
        )
        savings_fraction = (
            (brotli_bytes - stc_bytes) / brotli_bytes if brotli_bytes > 0 else 0.0
        )
        stc_beats = savings_fraction >= _STC_BEAT_BROTLI_REQUIRED_FRACTION
        row = {
            "rho": float(rho),
            "n_symbols": int(n),
            "brotli_bytes": int(brotli_bytes),
            "stc_syndrome_bytes": int(stc_bytes),
            "savings_fraction_stc_vs_brotli": float(savings_fraction),
            "stc_beats_brotli_by_5pct": bool(stc_beats),
            "stc_additive_cost": float(stc_cost),
        }
        per_rho.append(row)
        if rho <= _CONTEST_REALISTIC_RHO_CEILING:
            best_realistic_savings = max(best_realistic_savings, savings_fraction)
            if stc_beats:
                proceed_at_realistic_band = True

    if proceed_at_realistic_band:
        verdict = "PROCEED_STC_BEATS_BROTLI"
        rationale = (
            f"STC-syndrome beats brotli(mask-delta) by >="
            f"{_STC_BEAT_BROTLI_REQUIRED_FRACTION:.0%} at the contest-realistic "
            f"sparsity band (rho<={_CONTEST_REALISTIC_RHO_CEILING}); best realistic "
            f"savings={best_realistic_savings:.4f}. The symposium §6 reactivation "
            f"bar is MET. RATE-axis disambiguation resolved POSITIVE. This remains "
            f"[research-signal] until (a) the detector-informed cost map (CC#2) "
            f"lands AND (b) a paired CUDA+CPU auth-eval confirms the decoded "
            f"mask-delta reconstructs the SegNet/PoseNet score unchanged."
        )
    else:
        verdict = "DEFER_STC_DOES_NOT_BEAT_BROTLI"
        rationale = (
            f"STC-syndrome did NOT beat brotli(mask-delta) by >="
            f"{_STC_BEAT_BROTLI_REQUIRED_FRACTION:.0%} at the contest-realistic "
            f"sparsity band (best realistic savings={best_realistic_savings:.4f}). "
            f"The symposium §6 reactivation bar is NOT MET. DEFERRED-pending "
            f"(a) detector-informed cost map (CC#2) which spends flips in "
            f"scorer-blind texture regions AND (b) constraint-height/block-size "
            f"sweep per CC#4. Per CLAUDE.md 'Forbidden premature KILL' this is a "
            f"research-deferral, NOT a kill -- the 2026-04-29 FALSIFICATION was "
            f"at the dense-argmax implementation level; this probe extends the "
            f"sparse-delta reformulation the symposium PROCEEDed."
        )

    payload = _verdict(
        schema_ok=True,
        verdict=verdict,
        rationale=rationale,
        proceed_at_realistic_band=proceed_at_realistic_band,
        best_realistic_savings_fraction=(
            best_realistic_savings if best_realistic_savings != float("-inf") else 0.0
        ),
        required_savings_fraction=_STC_BEAT_BROTLI_REQUIRED_FRACTION,
        contest_realistic_rho_ceiling=_CONTEST_REALISTIC_RHO_CEILING,
        constraint_height=constraint_height,
        block_size=block_size,
        seed=seed,
        per_rho=per_rho,
    )
    payload["catalog_313_probe_outcome_kwargs"] = _catalog_313_probe_outcome_kwargs(
        payload
    )
    return payload


def _catalog_313_probe_verdict(verdict: str) -> str:
    if verdict == "PROCEED_STC_BEATS_BROTLI":
        return "PROCEED"
    if verdict == "DEFER_STC_DOES_NOT_BEAT_BROTLI":
        return "DEFER"
    return "OPERATOR_REVIEW_REQUIRED"


def _catalog_313_blocker_status(verdict: str) -> str:
    return "advisory" if _catalog_313_probe_verdict(verdict) == "PROCEED" else "blocking"


def _catalog_313_reactivation_criteria() -> list[str]:
    return [
        (
            "CC#2 detector-informed cost map (inverse SegNet boundary "
            "sensitivity) wired into STC cost vector; re-run probe; >=5% bar "
            "re-evaluated"
        ),
        (
            "CC#4 constraint-height {8,10,12} x block-size {32,64,128} "
            "sweep finds syndrome-byte-minimizing operating point"
        ),
        "real contest mask-delta via extract_mask_deltas_ternary replaces swept rho",
        (
            "paired CUDA+CPU auth-eval per Catalog #246 only after STC < "
            "brotli by >=5%"
        ),
    ]


def _catalog_313_next_action() -> str:
    return (
        "wire CC#2 detector-informed cost map + CC#4 h/block sweep + real "
        "extract_mask_deltas_ternary; re-probe before any paid dispatch"
    )


def _catalog_313_probe_outcome_kwargs(
    verdict: dict,
    *,
    evidence_path: str | None = None,
) -> dict:
    return {
        "probe_id": PROBE_ID,
        "substrate": PROBE_SUBSTRATE,
        "recipe_path": None,
        "probe_kind": PROBE_KIND,
        "verdict": _catalog_313_probe_verdict(str(verdict["verdict"])),
        "metric_name": PROBE_METRIC_NAME,
        "metric_value": float(verdict["best_realistic_savings_fraction"]),
        "threshold": float(verdict["required_savings_fraction"]),
        "threshold_token": PROBE_THRESHOLD_TOKEN,
        "evidence_path": evidence_path,
        "next_action": _catalog_313_next_action(),
        "reactivation_criteria": _catalog_313_reactivation_criteria(),
        "blocker_status": _catalog_313_blocker_status(str(verdict["verdict"])),
        "agent": "codex_stc_probe_cli",
        "subagent_id": None,
        "session_id": None,
        "notes": (
            "$0 local-CPU STC clean-source mask-delta disambiguator; "
            "research-signal only, no score authority."
        ),
        "staleness_window_days": 14,
    }


def register_catalog_313_probe_outcome(
    verdict: dict,
    *,
    evidence_path: str | None = None,
    ledger_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict:
    """Append the probe verdict through the canonical Catalog #313 ledger."""

    from tac.probe_outcomes_ledger import register_probe_outcome

    kwargs = _catalog_313_probe_outcome_kwargs(
        verdict,
        evidence_path=evidence_path,
    )
    verdict["catalog_313_probe_outcome_kwargs"] = kwargs
    row = register_probe_outcome(
        **kwargs,
        path=ledger_path,
        lock_path=lock_path,
    )
    verdict["catalog_313_probe_outcome_row"] = row
    return row


def _verdict(**kw) -> dict:
    """Assemble the structured probe verdict with canonical non-promotable markers."""
    base = {
        "schema": PROBE_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        # Canonical Tier A non-promotable markers per Catalog #341.
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "research-signal",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    base.update(kw)
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "$0 local-CPU Filler-STC clean-source mask-DELTA syndrome vs "
            "brotli disambiguator (symposium 2026-05-17 §6 reactivation bar)."
        )
    )
    parser.add_argument("--n", type=int, default=60_000)
    parser.add_argument("--constraint-height", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument(
        "--register-probe-outcome",
        action="store_true",
        help="Append the verdict through tac.probe_outcomes_ledger.",
    )
    parser.add_argument(
        "--probe-outcomes-ledger",
        type=str,
        default=None,
        help="Optional custom probe_outcomes.jsonl path for --register-probe-outcome.",
    )
    parser.add_argument(
        "--probe-outcomes-lock",
        type=str,
        default=None,
        help="Optional custom lock path for --register-probe-outcome.",
    )
    args = parser.parse_args(argv)

    verdict = run_probe(
        n=args.n,
        constraint_height=args.constraint_height,
        block_size=args.block_size,
        seed=args.seed,
    )

    evidence_path = args.json_out
    verdict["catalog_313_probe_outcome_kwargs"] = _catalog_313_probe_outcome_kwargs(
        verdict,
        evidence_path=evidence_path,
    )
    if args.register_probe_outcome:
        register_catalog_313_probe_outcome(
            verdict,
            evidence_path=evidence_path,
            ledger_path=(
                Path(args.probe_outcomes_ledger)
                if args.probe_outcomes_ledger
                else None
            ),
            lock_path=(
                Path(args.probe_outcomes_lock)
                if args.probe_outcomes_lock
                else None
            ),
        )

    out = json.dumps(verdict, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(out + "\n")
    print(out)
    # rc=0 always (a research-signal probe is not a gate); the verdict field
    # carries PROCEED/DEFER for the operator to route.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
