# Codex Findings: Master-Gradient Per-Frame Decomposition

Timestamp: 2026-05-20T21:39:45Z
Lane: `lane_master_gradient_frame_decomposition_20260520`

## Verdict

Landed an additive per-frame decomposition surface for existing per-pair
master-gradient tensors. This is a queryable structural projection, not a new
score authority and not a byte-mutation proof.

## Important correction

The operator insight that SegNet and PoseNet have asymmetric input topology is
correct:

- `upstream/modules.py` SegNet uses the last frame only via `x[:, -1, ...]`.
- `upstream/modules.py` PoseNet consumes both frames in the seq_len=2 sample.

The initial sliding-pair phrasing needed correction. `upstream/frame_utils.py`
uses `seq_len = 2` and emits non-overlapping samples, so canonical evaluator
pairs are `(0,1), (2,3), ...`, not sliding `(N-1,N)` and `(N,N+1)`. The new
helper therefore defaults to `topology="non_overlapping"` and marks
`topology="sliding"` as exploratory only.

## Landed implementation

- `src/tac/master_gradient_frame_decomposition.py`
  - Projects `(N_bytes, N_pairs, 3)` per-pair tensors into `(N_frames, 3)`.
  - Applies contest score marginal coefficients before L1 aggregation.
  - Routes SegNet sensitivity to each pair's last frame.
  - Splits PoseNet sensitivity across both pair frames.
  - Preserves rate sensitivity only as allocated bookkeeping, not scorer input
    topology.
  - Emits `score_claim=false`, `promotion_eligible=false`, and
    `ready_for_exact_eval_dispatch=false`.
- `tools/build_master_gradient_frame_decomposition.py`
  - Reproducible CLI from `.npy` per-pair tensors to JSON/Markdown/optional
    dense frame-axis `.npy`.
  - Uses either ledger anchor coefficients, explicit operating point, or
    explicit coefficients.
- `src/tac/cathedral_consumers/per_frame_sensitivity_consumer/__init__.py`
  - Auto-discovered cathedral consumer.
  - Exposes top-frame ordering to bit allocators/curricula/autopilot routing.
  - Remains observability-only with `predicted_delta_adjustment=0.0`.
- `src/tac/tests/test_master_gradient_frame_decomposition.py`
  - Covers non-overlapping topology, exploratory sliding topology, shape/config
    rejection, JSON/Markdown safety, and cathedral consumer routing.

## Materialized artifact

Built from:

`.omx/state/master_gradient_pr101_lc_v2_diagnostic_8pair_per_pair_20260518.npy`

with explicit PR101 LC-v2 operating point:

- `d_seg=0.000881195068359375`
- `d_pose=0.0017539353144626132`
- `rate=0.004747787410626081`
- `score=0.3392504150340868`

Outputs:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_decomposition_20260520_codex.json`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_decomposition_20260520_codex.md`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_axis_l1_20260520_codex.npy`

Key output:

- Tensor domain: `178158` bytes x `8` pairs x `3` axes.
- Projection: `16` canonical non-overlapping frames.
- Conservation: true.
- Pair-axis L1 sum equals frame-axis L1 sum within fp64 tolerance:
  - Seg: `223.13600781627656`
  - Pose: `57.65829099052398`
  - Rate: `0.0`
- Top frames: `15`, `13`, `5`, `9`, `11`, `7`, `1`, `3`.

Interpretation: odd frames dominate because they carry both SegNet
last-frame sensitivity and half of PoseNet sensitivity. Even frames are
PoseNet-only under canonical upstream pairing. This makes the SegNet/PoseNet
asymmetry queryable for frame-level budgets and LL/Hinton surrogate sampling.

## LL scorer-response dataset side effect

Also landed the non-promotional LL scorer-response dataset and next-probe
planner:

- `src/tac/optimization/scorer_response_dataset.py`
- `tools/build_scorer_response_dataset.py`
- `tools/plan_ll_scorer_response_next.py`
- `src/tac/tests/test_scorer_response_dataset.py`

Artifacts:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_response_dataset_20260520_codex.json`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_response_dataset_20260520_codex.md`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_20260520_codex.json`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_20260520_codex.md`

Dataset summary:

- Rows: `26`
- Total-score improvements: `0`
- Scorer-term improvements: `2`, both microscopic and over-budget.
- Current prohibition: do not widen coordinate sparse residual sidecar.
- Next probes: byte-neutral decoder-q response model, amortized residual
  grammar gate, and response dataset expansion.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_master_gradient_frame_decomposition.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_scorer_gradient_sparse_residual.py src/tac/tests/test_sparse_residual_oracle.py`
  - `21 passed`
- `.venv/bin/python -m py_compile src/tac/master_gradient_frame_decomposition.py src/tac/cathedral_consumers/per_frame_sensitivity_consumer/__init__.py tools/build_master_gradient_frame_decomposition.py tools/build_scorer_response_dataset.py tools/plan_ll_scorer_response_next.py`
  - passed
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1080 lane(s) validated cleanly`
- `.venv/bin/python - <<'PY' ... validate_consumer_module(per_frame_sensitivity_consumer)`
  - returned `contract_compliant=True`
- `git diff --check -- <touched files>`
  - passed

## Next action

Use the per-frame artifact as the natural training/sampling index for the LL
scorer surrogate:

1. Train/evaluate SegNet surrogate per frame, with higher sampling probability
   on high `seg_l1` odd frames.
2. Train/evaluate PoseNet surrogate per canonical pair, with pair weights
   inherited from the sum of both frame `pose_l1` entries.
3. Keep sparse residual widening blocked until an amortized or byte-neutral
   grammar can satisfy the observed break-even byte budget.
