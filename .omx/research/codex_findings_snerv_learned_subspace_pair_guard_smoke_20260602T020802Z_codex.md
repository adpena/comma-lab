# Codex Findings: SNeRV Learned-Subspace Pair-Guard Smoke

UTC: 2026-06-02T02:08:02Z
Agent: codex
Axis: [macOS-CPU advisory]
Authority: false-authority local continuation only

## Verdict

The next SNeRV QAT slice landed a non-coordinate
`learned_random_subspace` scorer-loop mode plus pair-robust acceptance guards.
The local 2-pair strided smoke found aggregate score-lowering directions, but
the strict pair pose guard rejected every candidate. This is a useful negative:
the contraction was not just a top-weight-coordinate artifact; aggregate
decoder-weight moves still produce pair-local pose damage.

Promotion/exact verdict remains `NO_GO_FOR_PROMOTION_OR_EXACT_EVAL`.

## Implementation

Changed surfaces:

- `src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py`
  - added `learned_random_subspace` search mode
  - added `pair_guard_min_score_improved_fraction`
  - added `pair_guard_max_pose_worsened_fraction`
  - exported row-level `accepted` and `blockers` into gate rows
- `tools/run_snerv_scorer_loop_decoder_qat_smoke.py`
  - exposed the learned-subspace mode and pair-guard knobs
- `src/tac/analysis/snerv_pose_guarded_decoder_gate.py`
  - now honors explicit scorer-loop row acceptance/blockers
  - prevents aggregate-good but source-rejected rows from entering
    `accepted_rows`
- focused tests cover pair-local score cancellation, pair-local pose worsening,
  learned-subspace labels, and the gate source-contract bug.

## Local Smoke

Command:

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/run_snerv_scorer_loop_decoder_qat_smoke.py \
  --n-pairs 2 \
  --start-pair 16 \
  --pair-stride 8 \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --search-mode learned_random_subspace \
  --max-trials 4 \
  --perturb-scale 0.02 \
  --pair-guard-min-score-improved-fraction 0.5 \
  --pair-guard-max-pose-worsened-fraction 0.0 \
  --out .omx/research/snerv_scorer_loop_decoder_qat_learned_subspace_2pair_strided_smoke_20260602T0200Z.json
```

Artifact:

- `.omx/research/snerv_scorer_loop_decoder_qat_learned_subspace_2pair_strided_smoke_20260602T0200Z.json`
- sha256: `20414f80e1bc0fa295fa3885cc78ada679b0fc25cbd32086f7bf3c4d34811d4d`

Result:

- baseline score: `1.0234573726510832`
- best accepted label: `least_squares_qat_baseline`
- accepted improvement: `false`
- ready for exact eval dispatch: `false`

Lowest-score rejected rows:

- `learned_subspace_004_minus`
  - score: `1.0019297393887077`
  - blocker: `pair_pose_worsening_fraction_guard_failed`
- `learned_subspace_001_plus`
  - score: `1.0023714919022129`
  - blocker: `pair_pose_worsening_fraction_guard_failed`
- `learned_subspace_002_minus`
  - score: `1.0028047609172444`
  - blockers: `seg_gate_failed`,
    `pair_pose_worsening_fraction_guard_failed`
- `learned_subspace_003_plus`
  - score: `1.0123153111050525`
  - blocker: `pair_pose_worsening_fraction_guard_failed`

## Gate

Command:

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/build_snerv_pose_guarded_decoder_gate.py \
  .omx/research/snerv_scorer_loop_decoder_qat_learned_subspace_2pair_strided_smoke_20260602T0200Z.json \
  --out .omx/research/snerv_scorer_loop_decoder_qat_learned_subspace_2pair_strided_pose_gate_20260602T0202Z.json
```

Artifact:

- `.omx/research/snerv_scorer_loop_decoder_qat_learned_subspace_2pair_strided_pose_gate_20260602T0202Z.json`
- sha256: `c347ad5c9207a89333c6d97e5c98d5f8d403f3aaadd22dce20742e4d971c3729`

Gate result:

- accepted rows: `0`
- verdict: `NO_GO_FOR_PROMOTION_OR_EXACT_EVAL`
- blocker: `no_candidate_passes_pose_guarded_local_continuation_gate`

The gate now records the source-contract blockers, for example:

- `source_scorer_loop_rejected`
- `source:pair_pose_worsening_fraction_guard_failed`

## Verification

Passed:

```bash
/Users/adpena/Projects/pact/.venv/bin/ruff check \
  src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py \
  src/tac/analysis/snerv_pose_guarded_decoder_gate.py \
  src/tac/tests/test_snerv_pose_guarded_decoder_gate.py \
  tools/run_snerv_scorer_loop_decoder_qat_smoke.py \
  tools/build_snerv_pose_guarded_decoder_gate.py

/Users/adpena/Projects/pact/.venv/bin/python -m pytest \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py \
  src/tac/tests/test_snerv_pose_guarded_decoder_gate.py
```

Pytest result: `17 passed`.

## Dispatch State

PR101 CPU recovery poll remained pending at `2026-06-02T02:06:00Z`:

- call id: `fc-01KT2BZT54G6CXPMD94SY43MMH`
- output dir:
  `/Users/adpena/Projects/pact/experiments/results/modal_auth_eval_cpu/pr101_storage_order_len24_cpu_20260601T1955Z`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

No full-video, exact, CUDA, or new dispatch work was launched.

## Next Actions

1. Stop coordinate and simple random-subspace decoder sweeps unless they add a
   materially different pair-robust optimizer.
2. Move to a genuinely trained scorer-loop decoder update, e.g. finite
   difference/NES with pair-robust objective, or a nonlinear HF decoder whose
   receiver grammar is explicitly byte-accounted.
3. Keep the pair-guard contract in the pose gate; do not accept aggregate-only
   rows when source smoke rejected them.
4. Start rate work in parallel: mixed-precision decoder grammar or
   decoder-delta packing, because the current fake-quant path still emits fp32
   receiver payload.
5. Keep PR101 CPU as the hard dispatch guard until the canonical recovery tool
   yields a terminal artifact.
