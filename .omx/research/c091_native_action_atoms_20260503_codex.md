# C091 Native Action Atom Planner - 2026-05-03 Codex

Scope: local-only C091-native tile-action atom planning. No GPU job was
dispatched. `.omx/state` was not read or modified.

## Inputs

- C091 PR75 public replay anchor:
  - score: `0.31516575028285976`
  - bytes: `276481`
  - SHA-256: `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
  - component trace:
    `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/component_trace.json`
- PR77 public replay exact T4:
  - score: `0.31537874750377204`
  - bytes: `276551`
  - SHA-256: `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`
  - component trace:
    `experiments/results/lightning_batch/exact_eval_pr77_tile_delta_public_replay_t4_20260503T1116Z/component_trace.json`
- PR77 action / C089 pose / C091 mask-renderer fixed-slice exact T4:
  - score: `0.318426107391119`
  - bytes: `276329`
  - SHA-256: `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8`
  - PoseNet avg: `0.0005419`
  - SegNet avg: `0.00060816`
  - score delta vs C091: `+0.0032603571082592264`
  - component trace:
    `experiments/results/lightning_batch/exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z/component_trace.json`
- PR75 action-subset exact traces used as non-C091-context atom feedback:
  `top25_p3`, `top40_p3`, `top25_ampminus1_p3`, `top40_p6`,
  `pose_safe_positive_ampminus1_p6`, `lag_eval_top67_p6`,
  `lag_eval_pose2_top67_p6`, and `lag_eval_pose4_top67_p6`.

## Outputs

- Tool:
  `experiments/plan_c091_native_action_atoms.py`
- Focused tests:
  `src/tac/tests/test_plan_c091_native_action_atoms.py`
- Policy artifacts:
  - `experiments/results/c091_native_action_atoms_20260503_codex/ranked_atom_policy.json`
  - `experiments/results/c091_native_action_atoms_20260503_codex/ranked_atom_policy.csv`

## Method

The planner uses C091 pair-level component trace samples as the anchor. For
each exact trace, positive deltas mean lower component contribution than C091.
Manifest-backed PR75 subset records are attributed by equal pair-share because
their exact traces contain multi-record action policies, not isolated causal
single-atom evals. PR77 public replay is treated as direct C091-native action
feedback because its decoded mask, renderer, and pose streams match the PR75
minp replay context. The fixed-slice PR77/C089-pose run is treated only as
pose-toxicity feedback, not as a replay candidate.

## Findings

- Ranked atoms emitted: `257`.
- Classification counts:
  - `component_positive_pose_safe`: `63`
  - `component_positive_pose_risky`: `78`
  - `pose_toxic`: `79`
  - `pose_toxic_pair`: `13`
  - `neutral_or_negative`: `24`
- Conservative positive eligible upper bound from the top 40 atoms:
  `0.00012712011530774194` component-score improvement.
- Best byte-screen break-even in the observed set is the `276317` byte PR75
  pose-safe P6 row:
  - delta bytes vs C091: `-164`
  - unchanged-component score: `0.31505654941454775`
  - sub-0.314 component improvement needed:
    `0.0010565494145477472`
  - byte-equivalent remaining gap: `1587`
- The upper bound is roughly `8.3x` smaller than the best observed
  sub-0.314 break-even. No C091-native action atom candidate is exact-eval
  justified from this evidence.

Top fixed-slice pose-heavy pairs from the harvested trace:

`105, 164, 69, 60, 64, 67, 106, 128, 197, 130, 136, 125, 108, 420, 423, 59, 97, 153, 99, 70`

Top fixed-slice SegNet-heavy pairs from the harvested trace:

`522, 517, 518, 519, 592, 488, 510, 514, 506, 542, 104, 502`

## Dispatch Decision

`exact_eval_justified=false`.

No candidate archive was emitted. The direct PR77 action/C089 pose fixed-slice
row is a scoped negative and should remain feedback only. Any future exact eval
needs a fresh non-noop C091-native archive whose manifest shows parser-safe
payload closure and at least `0.00105655` modeled component improvement after
rate accounting, followed by the lane-claim protocol before dispatch.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile experiments/plan_c091_native_action_atoms.py src/tac/tests/test_plan_c091_native_action_atoms.py
.venv/bin/python -m pytest src/tac/tests/test_plan_c091_native_action_atoms.py -q
.venv/bin/python experiments/plan_c091_native_action_atoms.py --output-dir experiments/results/c091_native_action_atoms_20260503_codex
```

Results:

- focused tests: `4 passed`
- atom policy: `257` atoms
- exact eval justified: `false`
- candidate archives emitted: `0`
