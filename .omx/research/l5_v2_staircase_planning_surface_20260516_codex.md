# L5 v2 Staircase Planning Surface

Date: 2026-05-16
Owner: Codex
Scope: Time-Traveler L5 v2 staircase priority path
Evidence grade: planning-only, no score claim

## Summary

L5 v2 now has a typed planning surface in
`src/tac/optimization/l5_staircase_v2.py`. The surface records the ordered
staircase steps, non-negotiable gates, canonical research basis ids, and a
source-backed prediction-band payload. The band is deliberately rank-blocked
until an axis-matched baseline, byte-closed temporal side-info proof, C1/Z5/TT5L
probe disambiguator, and paired CPU/CUDA empirical anchor exist.

The L5 composition row now carries this structured prediction-band payload
instead of an uncustodied naked numeric band. Cathedral/autopilot consumers can
see the L5 v2 path, but the validator still refuses rank reward or promotion.

Follow-up hardening in the same pass fixed two L5 v2 false-mechanism risks:

- `TimeTravelerSubstrate.render_pair()` now consumes both per-pair `pose_codes`
  and `EgoMotionDynamicsPrior` through a bounded ego-motion coordinate warp.
  Tests prove same-pair output changes when pose or dynamics changes, and
  gradients reach renderer, pose codes, and dynamics parameters.
- The TT5L inflate contract now matches actual runtime imports: torch + brotli,
  no `av`, and a realistic 350-LOC budget.
- The trainer records byte-proxy calibration (`proxy_bytes`, packed `0.bin`
  bytes, zipped archive bytes, deltas, and ratios) and shape-readiness metadata
  so smoke/partial/full-CPU artifacts cannot masquerade as promotable contest
  packets.

## 9-dimension success checklist evidence

1. Uniqueness: L5 v2 is tracked as a predictive-receiver staircase, not a
   generic HNeRV/NeRV refinement.
2. Beauty and elegance: the surface is a small typed module with tests and no
   provider side effects.
3. Distinctness: the required gates separate temporal side-info consumption,
   C1/Z5/TT5L disambiguation, paired axis evidence, and stack-of-stacks entry.
4. Rigor: prediction bands require source ids, local ledgers, uncertainty,
   supersession, and empirical-anchor state.
5. Optimization per technique: canonical source aliases are allowed, but L5 v2
   has its own source stack and gates.
6. Stack-of-stacks composability: the final step is explicitly blocked until
   component anchors exist.
7. Deterministic reproducibility: no dispatch authority is granted without
   archive SHA, runtime tree SHA, and paired CPU/CUDA axis custody.
8. Extreme optimization and performance: decode/runtime timing remains an
   explicit blocker instead of an assumption.
9. Minimal contest score: the predicted delta remains planning-only until exact
   evidence proves real score movement.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_research_basis.py src/tac/tests/test_prediction_band.py src/tac/tests/test_substrate_composition_matrix.py -q`
- `.venv/bin/python -m pytest src/tac/substrates/time_traveler_l5_autonomy/tests/test_time_traveler_architecture.py src/tac/substrates/time_traveler_l5_autonomy/tests/test_registered_substrate.py src/tac/tests/test_train_time_traveler_full_cpu_mode.py -q`
- `.venv/bin/python -m py_compile src/tac/optimization/l5_staircase_v2.py src/tac/optimization/substrate_composition_matrix.py src/tac/optimization/research_basis.py src/tac/optimization/prediction_band.py src/tac/tests/test_l5_staircase_v2.py`
- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/optimization/substrate_composition_matrix.py src/tac/optimization/research_basis.py src/tac/optimization/prediction_band.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_research_basis.py src/tac/tests/test_prediction_band.py`
- `git diff --check -- src/tac/optimization/l5_staircase_v2.py src/tac/optimization/substrate_composition_matrix.py src/tac/optimization/research_basis.py src/tac/optimization/prediction_band.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_research_basis.py src/tac/tests/test_prediction_band.py`
