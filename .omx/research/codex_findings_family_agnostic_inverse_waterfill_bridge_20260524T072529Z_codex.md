# Codex Findings: Family-Agnostic Inverse Waterfill Bridge

UTC: 2026-05-24T07:25:29Z

## Scope

The original objective is not pair-drop search. It is to learn the scorer/receiver
decision surface around the raw contest video and candidate reconstructions, then
encode only what SegNet/PoseNet/rate cannot infer for free. The practical control
surface should be a discrete action functional over bytes, tensors, packet
members, archive sections, frames, pairs, regions, scorer components, calibration
residuals, and interaction terms.

## Landing

Codex wired byte-shaving campaign surfaces and campaign plans into the
inverse-steganalysis action functional:

- `tac.optimization.inverse_steganalysis_acquisition.action_atoms_from_byte_shaving_signal_surface`
- `tac.optimization.inverse_steganalysis_acquisition.action_atoms_from_byte_shaving_campaign_plan`
- CLI flags on `tools/build_inverse_steganalysis_action_functional.py`:
  `--byte-shaving-signal-surface` and `--byte-shaving-campaign-plan`
- `comma_lab.scheduler.byte_shaving_campaign_queue` materializer work rows can
  now compile inverse action functionals from scorer-response rows, inverse
  scorer surfaces, byte-shaving signal surfaces, or byte-shaving campaign plans.

This keeps HNeRV payload-section work, BoostNeRV/NeRV tensor overlays, and
non-NeRV packet/member/byte-range candidates in one water-fill model instead of
forcing them through DQS1 pair-drop logic. Coupled operation sets preserve
selected operations, chosen order, active interactions, byte savings, expected
score gain, second-order synergy/antagonism, and source refs as false-authority
provenance.

## Safeguards

The bridge recursively rejects truthy authority fields before consuming
byte-shaving signal surfaces or campaign plans. Emitted action atoms and nested
operation-set provenance remain:

- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`

Routine proof requirements such as inflate parity or exact auth before promotion
are not treated as scorer fragility. Real fragility/discontinuity/parity-failed
blockers still penalize the water-fill surface.

## Verification

- `src/tac/tests/test_inverse_steganalysis_acquisition.py`
- `src/tac/tests/test_inverse_steganalysis_action_functional_cli.py`
- `src/tac/tests/test_byte_shaving_campaign_queue.py`
- `src/tac/tests/test_byte_shaving_campaign.py`
- `src/tac/tests/test_optimizer_candidate_queue.py`

Focused runs passed:

- `30 passed` for inverse acquisition + action-functional CLI
- `71 passed` for byte-shaving queue + inverse acquisition + action-functional CLI
- `77 passed` for byte-shaving campaign + optimizer queue + inverse acquisition + action-functional CLI

Ruff passed on touched bridge, CLI, queue, registry, and test files.

## Remaining Gaps

1. Materializer coverage is still narrower than the planner. HNeRV section
   recodes, BoostNeRV tensor overlays, broader NeRV variants, and non-NeRV
   packet/member candidates can now be ranked in the inverse water-fill surface,
   but most still backlog as missing materializer adapters.
2. Local MLX/NumPy training backends are not yet a generic queue-owned runner for
   representation-training probe manifests. Reuse
   `local_training_runtime_profile.py` and
   `representation_training_probe_integration.py`; map `local_numpy` to
   scheduler resource `local_cpu`, and keep `local_mlx` under strict
   research-signal authority.
3. The MLX scorer-response bridge is fast enough to be the broad acquisition map,
   but MLX is still not score authority. Same-candidate calibration and strict
   auth-axis gates remain required before exact-eval spend triage or promotion.
4. The next production tranche should turn the new action-functional output into
   queue-owned local backend jobs for HNeRV/BoostNeRV/NeRV/non-NeRV families,
   then harvest observations back into the action surface.
