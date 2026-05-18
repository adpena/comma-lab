---
review_kind: recipe_reconciliation
review_id: c6_beta_ib_sweep_recipe_reconciliation_20260518_codex
review_date: "2026-05-18"
lane_id: lane_c6_ibps_beta_ib_sweep_build_per_symposium_parallel_path_a_20260518
substrate: c6_e4_mdl_ibps_beta_ib_sweep
evidence_axis: pre_dispatch_contract_guard
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
---

# C6 Beta-IB Sweep Recipe Reconciliation

## Summary

The C6 optimal-form symposium requires path (a), a fixed-width beta_ib sweep
over `[0.0001, 0.001, 0.01, 0.1, 1.0]`, to run in parallel with the latent_dim
sweep under the $5 envelope. While this landing was in progress, overlapping
partner WIP created four `beta_ib_*` recipes. I stopped editing during the
realtime churn window, then reconciled to the partner naming and lane surface
after the tree was quiet.

The authoritative beta_ib recipe set is now:

- `substrate_c6_e4_mdl_ibps_beta_ib_0p0001_modal_t4_smoke_dispatch.yaml`
- `substrate_c6_e4_mdl_ibps_beta_ib_0p001_modal_t4_smoke_dispatch.yaml`
- `substrate_c6_e4_mdl_ibps_beta_ib_0p01_modal_t4_smoke_dispatch.yaml`
- `substrate_c6_e4_mdl_ibps_beta_ib_0p1_modal_t4_smoke_dispatch.yaml`
- `substrate_c6_e4_mdl_ibps_beta_ib_1p0_modal_t4_smoke_dispatch.yaml`

No provider job was launched. No [contest-CUDA] or [contest-CPU] score claim is
made here.

## Repair

- Added the missing `beta_ib_0p01` control recipe so the symposium-specified
  five-point sweep is complete.
- Removed the duplicate `beta0p*` recipe filenames from my own WIP and kept
  the partner `beta_ib_*` naming/umbrella lane as the single operator-facing
  surface.
- Extended the C6 remote driver terminal dispatch-claim notes to include
  `beta_ib`, matching the already-present provenance JSON field and trainer
  `--beta-ib` argv.
- Updated `src/tac/tests/test_c6_ibps_beta_ib_sweep_build.py` to cover all
  five beta_ib recipes, including the 0.01 fixed-width control and the
  terminal-claim `beta_ib` note.

## Verification

- `bash -n scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh`
- `.venv/bin/python -m pytest src/tac/tests/test_c6_ibps_beta_ib_sweep_build.py -q`
  - `84 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_c6_ibps_latent_dim_sweep_build.py -q`
  - `42 passed`
- `.venv/bin/python tools/audit_predicted_band_provenance.py --strict`
  - `Recipes scanned: 80`
  - `PASS: 19`
  - `FAIL: 0`
- `.venv/bin/python tools/claim_lane_dispatch.py summary --live-only --format json`
  - active lane remains Z6 Wave 2 only; no active C6 dispatch claim.

## Six-Hook Wire-In

1. Sensitivity-map contribution: N/A. This is a pre-dispatch recipe/driver
   contract repair, not an empirical score anchor.
2. Pareto constraint: active. The five-point sweep isolates beta_ib at fixed
   latent_dim=24 so rate/distortion movement can be compared against the
   latent_dim path without conflating width.
3. Bit-allocator hook: N/A. No archive bytes changed in this landing.
4. Cathedral autopilot dispatch hook: active guardrail. The recipe surface is
   non-dispatchable (`research_only: true`, `dispatch_enabled: false`) until
   operator sign-off, and the driver now preserves beta_ib in terminal claim
   notes for future harvest/audit.
5. Continual-learning posterior update: this ledger records the overlap
   reconciliation and the false-authority hazard that beta_ib results must
   carry their sweep value in dispatch-level custody, not only in local JSON.
6. Probe-disambiguator: active. The beta_ib sweep is the path (a)
   Lagrangian-multiplier disambiguator from the C6 symposium; the latent_dim
   sweep remains the path (b) capacity disambiguator.

## Reactivation Criteria

The beta_ib sweep remains non-promotional until operator sign-off and a claimed
dispatch produces variant-specific archive/runtime custody with beta_ib in the
recipe, terminal claim notes, provenance JSON, trainer argv, paired CPU/CUDA
axis labels, and post-training Catalog #324 Tier-C validation.
