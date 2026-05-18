---
review_kind: dispatch_contract_repair
review_id: c6_latent_sweep_remote_driver_contract_repair_20260518_codex
review_date: "2026-05-18"
lane_id: lane_c6_ibps_reactivation_path_b_latent_dim_sweep_build_20260518
substrate: c6_e4_mdl_ibps_latent_dim_sweep
evidence_axis: pre_dispatch_contract_guard
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
---

# C6 Latent Sweep Remote Driver Contract Repair

## Summary

The C6 latent-dim sweep recipes for latent_dim 48, 96, and 192 reused the
baseline C6 remote driver, but the driver still hard-coded the baseline lane
id and baseline recipe path. The driver also required a dispatch instance id
but did not verify that an active lane-claim row existed before bootstrapping
the remote job.

This was a pre-dispatch false-authority bug class: a latent_dim sweep smoke
could fail claim verification against the wrong lane, write baseline custody
metadata for a sweep artifact, or proceed without proving the active claim
contract that AGENTS.md requires before paid training.

No provider job was launched by this repair. No [contest-CUDA] or
[contest-CPU] score claim is made here.

## Repair

- `scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh` now accepts
  `C6_E4_MDL_IBPS_LANE_ID` / `PACT_DISPATCH_LANE_ID` instead of hard-coding
  the baseline C6 lane id.
- The driver now accepts `C6_E4_MDL_IBPS_RECIPE_PATH` and writes the actual
  recipe path into `provenance.json`.
- The driver now verifies the active dispatch claim row by lane id and
  instance/job id before remote bootstrap.
- Terminal claim notes now include latent_dim and recipe path.
- The heartbeat cleanup now kills and waits on the heartbeat child so scripted
  callers do not hang on inherited pipes.
- The latent48, latent96, and latent192 recipes now thread
  `C6_E4_MDL_IBPS_LANE_ID`, `C6_E4_MDL_IBPS_RECIPE_PATH`, and a variant
  `TAG` through `env_overrides`.

## Verification

- `bash -n scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh`
- `.venv/bin/python -m pytest src/tac/tests/test_c6_ibps_latent_dim_sweep_build.py -q`
  - `42 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_c6_ibps_latent_dim_sweep_build.py src/tac/tests/test_time_traveler_l5_z6_remote_driver.py -q`
  - `44 passed`
- `.venv/bin/python tools/audit_predicted_band_provenance.py --strict`
  - `Recipes scanned: 75`
  - `PASS: 19`
  - `FAIL: 0`
- `.venv/bin/python tools/claim_lane_dispatch.py summary`
  - `active=1`
  - active lane is Z6 Wave 2, not C6

## Six-Hook Wire-In

1. Sensitivity-map contribution: N/A. This is a pre-dispatch contract repair,
   not a new empirical anchor.
2. Pareto constraint: protects the latent-width Pareto test by ensuring the
   launched variant's lane/provenance identify the actual width.
3. Bit-allocator hook: N/A. No archive bytes changed.
4. Cathedral autopilot dispatch hook: active guardrail. The C6 driver now
   fails closed unless the active dispatch claim matches the recipe lane and
   instance/job id.
5. Continual-learning posterior update: this ledger records the false-authority
   bug class before any latent sweep spend.
6. Probe-disambiguator: active. The latent_dim sweep is the C6 SegNet-collapse
   disambiguator for the 24-dim bottleneck hypothesis; the driver now preserves
   the lane identity needed to compare latent48, latent96, and latent192
   evidence without baseline-lane contamination.

## Reactivation Criteria

The C6 latent sweep remains non-promotional until a claimed dispatch produces
variant-specific custody with the sweep lane id, actual recipe path, latent_dim,
archive/runtime evidence, paired CPU/CUDA axis labels, and post-training
Catalog #324 Tier-C validation. Any future latent sweep output with the
baseline C6 lane id or baseline recipe path must be classified as
`indeterminate_harness_or_runtime_mismatch`, not as a width-sweep method result.
