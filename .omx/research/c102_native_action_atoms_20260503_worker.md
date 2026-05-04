# C102 Native Action Atom Planner - 2026-05-03 Worker

Scope: local-only C102/top192 native tile-action atom planning. No remote GPU
job was dispatched, no lane claim was opened, and no score claim is made.

## Anchor

- Current A++ frontier: `C102/top192`
- Archive:
  `experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip`
- Component trace:
  `experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/component_trace.json`
- Score: `0.31514430182167497`
- Bytes: `276485`
- SHA-256:
  `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8`
- Strict `<=0.31` component improvement needed at anchor bytes:
  `0.005144301821674968`

## Tool And Outputs

- Tool:
  `experiments/plan_c102_native_action_atoms.py`
- Focused tests:
  `src/tac/tests/test_plan_c102_native_action_atoms.py`
- JSON:
  `experiments/results/c102_native_action_atoms_20260503_worker/ranked_atom_policy.json`
- Atom CSV:
  `experiments/results/c102_native_action_atoms_20260503_worker/ranked_atoms.csv`
- Policy CSV:
  `experiments/results/c102_native_action_atoms_20260503_worker/ranked_policies.csv`

The planner mines PR75/PR77 action records and exact component traces, computes
C102-minus-candidate component deltas by pair, shares multi-record pair deltas
across selected action records, and marks records already present in the C102
action stream as `exact_anchor_duplicate_noop`.

## Evidence Used

- PR75 top25/top40 P3 exact traces.
- PR75 top25 ampminus1 P3 exact trace.
- PR75 lag-eval pose4 top67 P6 exact trace.
- PR77 public replay exact trace.
- PR77 action/C089 pose fixed-slice exact negative trace.
- Existing C101/top192 native-action builder manifests only for byte context.
- C101 renderer/top192 exact negative:
  `experiments/results/lightning_batch/exact_eval_c101_renderer_x_top192_stack_t4_20260503T1540Z/contest_auth_eval.adjudicated.json`

## Results

- Ranked atoms: `220`
- Classification counts:
  - `component_positive_pose_safe`: `22`
  - `component_positive_pose_risky`: `51`
  - `exact_anchor_duplicate_noop`: `91`
  - `pose_toxic`: `42`
  - `pose_toxic_pair`: `5`
  - `neutral_or_negative`: `9`
- Exact eval justified: `false`
- Dispatchable policies: `0`

Top policy proxies:

| Policy | Records | Expected component benefit proxy | Archive delta vs C102 | Required component improvement to `<=0.31` | Dispatchable |
| --- | ---: | ---: | ---: | ---: | --- |
| `c102_consensus_positive_top32_proxy` | 22 | `0.00007186622247503413` | `-162` | `0.0050364326712691865` | `false` |
| `c102_pose_safe_positive_top48_proxy` | 22 | `0.00007186622247503413` | `-162` | `0.0050364326712691865` | `false` |
| `c102_consensus_positive_top64_proxy` | 22 | `0.00007186622247503413` | `-153` | `0.005042425401847295` | `false` |

Only 22 atoms survived the non-noop, component-positive, pose-safe filter, so
the top32/top48/top64 planning rows collapse to the same record set except
where an existing builder manifest supplies actual byte context. The best
modeled benefit is about `70x` smaller than the strict `<=0.31` component
break-even after rate.

## Dispatch Boundary

Do not dispatch from this planning artifact. A future exact eval would require
all of:

1. A fresh C102-native archive with parser-safe payload closure and no duplicate
   or hidden members.
2. Manifest proof that the action stream is non-noop relative to the exact
   C102 anchor SHA above.
3. Either actual archive bytes low enough to close the byte-only gap
   (`<=268758` bytes for strict `<0.31` with unchanged components, or
   `<=268759` for `<=0.31` up to floating-point recomputation) or a component
   benefit model that clears the policy-specific break-even by a large margin.
4. No membership in the known exact-negative SHA set:
   `d79d1556b55ba7e36c5aaf91d5b04320587975f1303698d8f1089bd5f399d0f3` or
   `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8`.
5. Lane claim via `tools/claim_lane_dispatch.py claim ...` before any
   Lightning/Vast/Modal exact CUDA auth eval submission.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile experiments/plan_c102_native_action_atoms.py src/tac/tests/test_plan_c102_native_action_atoms.py
.venv/bin/python -m pytest src/tac/tests/test_plan_c102_native_action_atoms.py -q
.venv/bin/python experiments/plan_c102_native_action_atoms.py --output-dir experiments/results/c102_native_action_atoms_20260503_worker
```

Results:

- focused tests: `3 passed`
- atom policy: `220` atoms
- exact eval justified: `false`
- remote dispatch performed: `false`
