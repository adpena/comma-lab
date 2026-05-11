# Score-lowering next tranche custody (2026-05-11)

## Scope

This ledger records local, non-dispatch score-lowering work done while the
active T1 Ballé Modal dispatch is still pending. No GPU job was launched.

Active dispatch gate:

- lane: `t1_balle_128k_endtoend`
- job: `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- Modal call id: `fc-01KR955JSYQAVTTYZA48VAV7WJ`
- current status at this tranche: `pending`

## PR103-on-PR106 raw-output custody plan

The current exact paired PR103-on-PR106 artifacts are score-paired but not
mechanism-complete:

- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- archive bytes: `185578`
- runtime content tree SHA-256:
  `f2ebe56a408a55b39070f9f86ba77fb11a9b43d83c0e02692f0acc0bf1ff28bb`
- CPU score artifact:
  `experiments/results/dual_device_auth_eval/pr103_pr106_dual_runtime_cpu_v2_20260511T022553Z/contest_auth_eval.adjudicated.json`
- CUDA score artifact:
  `experiments/results/modal_auth_eval/pr103_pr106_dual_runtime_cuda_v2_20260511T022553Z/contest_auth_eval.json`

Current exact-pair analysis:

- valid_for_pair_score_analysis: `true`
- valid_for_mechanism_analysis: `false`
- raw_output_pairing_status: `raw_output_manifest_missing`
- CUDA minus CPU score gap: `-0.02067461068280979`
- CUDA pose term advantage: `-0.022166610682809812`
- CUDA seg term disadvantage: `+0.0014919999999999933`

Prepared next plan artifact (ignored raw artifact; summary tracked here):

- `.omx/research/artifacts/pr103_pr106_raw_pair_after_t1_20260511/plan_existing_pair.json`
- `.omx/research/artifacts/pr103_pr106_raw_pair_after_t1_20260511/drift_existing_pair.json`
- `.omx/research/artifacts/pr103_pr106_raw_pair_after_t1_20260511/drift_existing_pair.md`

Next execution after T1 clears: rerun the same PR103-on-PR106 archive/runtime
with retained raw outputs and explicit per-axis JSON:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip \
  --inflate-sh submissions/pr103_pr106_final_runtime/inflate.sh \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --device cuda \
  --work-dir experiments/results/dual_device_auth_eval/pr103_pr106_raw_pair_after_t1_20260511/cuda \
  --json-out experiments/results/dual_device_auth_eval/pr103_pr106_raw_pair_after_t1_20260511/cuda/contest_auth_eval.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 1800 \
  --keep-work-dir
```

CPU rerun must be Linux x86_64 if it will be labeled `[contest-CPU]`; local
macOS CPU remains advisory only.

## Sub-0.17 HNeRV low-rank plan correction

The sub-0.17 planner still gives useful byte-budget math, but its previous
posthoc factorization path is now explicitly blocked by the relerr schedule
probe:

- updated tool: `tools/plan_sub017_cpu_frontier.py`
- updated test: `src/tac/tests/test_plan_sub017_cpu_frontier.py`
- current plan artifact:
  `.omx/research/artifacts/current_sub017_frontier_plan_20260511/plan.json`
- recommended projection id: `svd_stem_blocks012_balanced`
- projected score if components hold: `0.159553699633`
- posthoc relerr probe status: `falsified_for_posthoc_factorization`
- best isolated brotli savings in the measured posthoc probe: `0`

Interpretation: do not spend GPU or dispatch time on a posthoc factorized HNeRV
archive rewrite. The reactivation path is a trained low-rank/factorized HNeRV
substrate with eval-roundtrip-aware loss, score-domain or QAT pressure on the
factor streams, and a byte-closed runtime consuming factor/residual sections.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_plan_sub017_cpu_frontier.py \
  src/tac/tests/test_plan_factorized_hnerv_relerr_schedule.py
```

Result: `8 passed in 0.96s`.

## Dispatch classification

- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- dispatch_attempted: `false`
