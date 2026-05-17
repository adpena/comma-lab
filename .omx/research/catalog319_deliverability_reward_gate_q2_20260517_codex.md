# Catalog #319 Q2 Deliverability-Gated Venn Reward

Date: 2026-05-17
Owner: codex
Lane: lane_q2_q3_batched_catalog_319_gate_plus_autopilot_reweight_v2_20260517
research_only: false
score_claim: false
promotion_eligible: false

## Finding

The Q1 Wyner-Ziv deliverability package landed with a reserved preflight stub
while the cathedral autopilot already had a positive HIGH_PAIR_INVARIANT Venn
reward path. That created a false-authority gap: Venn classification could say
bytes are pair-shared, but the ranker could still reward them before any
per-archive proof showed those bytes are contest-deliverable without scorer
access, network fetches, runtime-budget violations, or an unapproved
inflate.py waiver.

## Fix

The positive Venn reward now fails closed:

1. `adjust_predicted_delta_for_venn_classification` still reads the Venn
   sidecar and still applies the conservative HIGH_PAIR_SPECIFIC demotion.
2. The HIGH_PAIR_INVARIANT reward calls
   `_venn_deliverability_reward_factor_for_archive`.
3. That helper loads the newest `DeliverabilityProof` for the archive and runs
   `verify_deliverability_proof_contest_compliance`.
4. Missing proof, invalid proof, import failure, or non-compliant proof returns
   factor `1.0` (no positive reward).
5. Compliant proofs apply a byte-weighted reward factor:
   Tier 1 = `1.20`, Tier 2 = `1.10`, approved Tier 3 = `1.05`, neutral bytes
   = `1.0`.

Catalog #319 now has a concrete scanner at
`src/tac/preflight.py::check_substrate_wyner_ziv_reweight_has_deliverability_proof`
and a strict `preflight_all(strict=True)` call. Live count is zero at flip.
It refuses a blanket
HIGH_PAIR_INVARIANT factor or any high-invariant reward branch missing the
proof loader, verifier, and factor helper. Same-line waiver:
`VENN_REWEIGHT_DELIVERABILITY_OK:<rationale>`; placeholder rationale is
rejected.

## Evidence

Focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_check_319_wyner_ziv_deliverability_gate.py \
  src/tac/tests/test_cathedral_autopilot_venn_risk_composition.py \
  src/tac/tests/test_wyner_ziv_deliverability_proof_builder.py \
  src/tac/tests/test_wyner_ziv_deliverability_prober.py \
  src/tac/tests/test_session_20260517_cli_flag_additions.py -q
```

Expected result: all tests pass.

This is a ranker-custody and false-authority fix only. It makes no score claim,
dispatch claim, archive promotion, or leaderboard-axis claim.
