# L5 v2 TT5L Adversarial Fix Wave

Date: 2026-05-16
Author: codex
Status: landed
Scope: L5 v2 / TT5L staircase hardening

## Findings Addressed

1. Side-info consumption proof was not causal enough. The proof now records
   per-section baseline/mutated hashes and requires
   `non_target_sections_identical=true` for `world_model_blob`, `ac_state_blob`,
   and `meta_blob`. The gate validator rejects missing or false non-target
   section identity.
2. Committed side-info proof custody was mutable. The committed full-frame
   proof is now pinned by SHA-256 in the L5 v2 registry:
   `7efdac0ed0ac026d7adb7eb706b1a8f844ec6e15422b7defbb34114db5ef775a`.
3. TT5L Dykstra `INDETERMINATE` and `INFEASIBLE` verdicts could structurally
   validate. TT5L readiness now unlocks only when the verdict is `FEASIBLE`.
   The CLI exits nonzero on `INDETERMINATE` unless an explicit allowance flag
   is passed.
4. The contest full-frame proof CLI exited 0 on failed predicates. It now
   returns 1 when `predicate_passed=false`.
5. The TT5L Modal recipe still contained active-prediction wording for the
   retired additive band. The wording now labels that band retired and
   non-authoritative.
6. The paired-axis next action lacked terminal claim custody. The readiness
   payload now includes terminal-claim requirements, pair group id, per-axis
   job-id fields, and success/failure closeout templates.

## Artifacts

- `.omx/research/tt5l_contest_sideinfo_consumption_proof_20260516_codex.json`
- `.omx/research/tt5l_contest_sideinfo_outputs_manifest_20260516_codex.json`
- `.omx/research/l5_v2_probe_template_20260516_codex.json`
- `.omx/operator_authorize_recipes/substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml`

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/substrates/time_traveler_l5_autonomy/tests/test_consumption_proof.py \
  src/tac/tests/test_check_substrate_dykstra_feasibility.py \
  src/tac/tests/test_l5_v2_probe_disambiguator.py -q

.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/substrates/time_traveler_l5_autonomy/consumption_proof.py \
  src/tac/substrates/time_traveler_l5_autonomy/tests/test_consumption_proof.py \
  tools/build_tt5l_contest_sideinfo_consumption_proof.py \
  tools/check_substrate_dykstra_feasibility.py \
  src/tac/tests/test_check_substrate_dykstra_feasibility.py \
  src/tac/tests/test_l5_v2_probe_disambiguator.py
```

Results: 115 passed; ruff clean.

## Remaining Work

Populate the emitted L5 v2 probe template with measured paired CPU/CUDA
observations for C1, Z5, and TT5L. This remains non-promotional until the
probe disambiguator validates all candidates and paired axis custody exists.
