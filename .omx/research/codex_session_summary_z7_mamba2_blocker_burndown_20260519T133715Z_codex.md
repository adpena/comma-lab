# Codex Session Summary - Z7-Mamba-2 Blocker Burndown - 2026-05-19

session_actor: codex
date_utc: 2026-05-19T13:37:15Z
score_claim: false
promotion_eligible: false
dispatch_fired: false

## Changed

- Hardened `tools/verify_z7_exact_eval_handoff.py` against exact-eval readiness
  false authority and hard-coded LSTM identity leakage.
- Hardened
  `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` with
  substrate-specific probe contracts and axis-label normalization.
- Hardened
  `experiments/train_substrate_time_traveler_l5_z7_mamba2.py` with portable
  generated inflate runtime, active lane smoke identity, LR warmup, gradient
  clipping, recorded stability telemetry, and no-score authority fields.
- Hardened `src/tac/optimization/mamba2_predictor.py` so stateful `mamba_ssm`
  cannot produce misleading evidence before real state replay exists.
- Hardened `src/tac/substrates/time_traveler_l5_z7_mamba2/archive.py` so parsed
  configs preserve decoder geometry.
- Hardened `scripts/remote_lane_substrate_time_traveler_l5_z7_mamba_2.sh` with
  recipe-level full-mode refusal and `ready_for_paid_dispatch` completion
  checks.
- Updated the operator-authorize recipe to reference the Mamba-specific
  disambiguator artifact.
- Added focused tests for each bug class.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools .venv/bin/python -m pytest -q ...`
  on the focused Z7-Mamba bundle: `83 passed, 12 warnings`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools .venv/bin/python -m ruff check ...`:
  `All checks passed`.
- `bash -n scripts/remote_lane_substrate_time_traveler_l5_z7_mamba_2.sh`.
- `git diff --check`.
- `tools/operator_authorize.py --recipe substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch --dry-run`:
  refused as expected.
- Tiny proxy packet with real video input:
  `experiments/results/z7_mamba2_codex_tiny_e2e_hardened_20260519T133511Z`.
- Readiness assessment remains `DEFER` with evidence-only blockers and no local
  implementation blockers surfaced by this pass.

## Authority State

The Z7-Mamba-2 path is still `research_only=true` and `dispatch_enabled=false`.
This landing burns down local implementation blockers but does not make a score
claim, promotion claim, or paid-dispatch authorization.
