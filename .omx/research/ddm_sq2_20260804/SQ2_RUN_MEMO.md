# sq2 #935 convergence-stop rerun

## Scope

Axis: `[macOS-CPU frozen-scorer advisory]` for the n32 stage receipt. No score claim and no promotion from this memo alone.

Charter: `.omx/tmp/codex_runs/sq2_prompt.md` plus `.omx/tmp/codex_runs/_common_contract.md`.

Live scorer-slot boundary, observed from `.omx/state/main_hot_state.md` and `.omx/research/scorer_batch_20260804.md`: `sb1` owns the fleet-wide full-n600 scorer slot in the current batch state. This arm can run the sq1 n32 convergence receipt and aggregate it. Any full-n600 spend remains gated by the sq2 R8 arithmetic and by the live single-slot boundary.

## Inputs

- selected pairs: `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_selected_pairs.npy`
  - bytes: 384
  - sha256: `f85c034bbf16303246cbdf8ab506103d3a7e4c811e560bb8c3567e830fac2d4c`
- v0 aggregate input: `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_eta_seg_n32.json`
  - bytes: 106057
  - sha256: `17da6ccc90ec12be87acbfc8bff5a776ea9ea84d6a282cc836c49a71cdf3e97f`
- source submission archive: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2/archive.zip`
  - bytes: 353805
  - sha256: `c72ef357416b66e716b2863c4c49360306b80cc0fafd094e02394c8a4dd37209`
- GT video: `upstream/videos/0.mkv`
  - bytes: 37545489
  - sha256: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

SSD preflight: `/Volumes/VertigoDataTier` had 179 GiB available before launch.

## Stage Command

```bash
.venv/bin/python experiments/ddm_sq1_stage_decomposition_and_solved_paint.py \
  --sub-dir /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2 \
  --gt-mkv upstream/videos/0.mkv \
  --pairs-npy /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_selected_pairs.npy \
  --out /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap100_sq2.json \
  --threads 6 --steps 100 --eval-every 5 --convergence-patience-evals 3 --resume
```

Argparse verified before launch: every flag above exists on `experiments/ddm_sq1_stage_decomposition_and_solved_paint.py`.

## Aggregate Command

```bash
.venv/bin/python experiments/ddm_sq1_aggregate.py \
  /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_eta_seg_n32.json \
  /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap100_sq2.json \
  /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32_uncap100_sq2.json
```

## Prior Receipts Consumed

- 25-step aggregate: `eta_net_pooled=0.7895095948827292`; no explicit convergence stop reason.
- 50-step aggregate: `eta_net_pooled=0.8620042643923241`; cap census 31/32 `iteration_cap_best_at_cap`, 1/32 `iteration_cap_before_plateau`; pose mean after `0.04925062965230609` on the selected subset.

## RECALL EVIDENCE

Searches run for this sq2 verification pass:

- `rg -n "sq1|sq2|ddm_sq1|solved-paint|solved paint|stage_decomposition|CONVERGED-STOP|qo1|m46|jd5|R8 guard|pose bank" /Users/adpena/.codex/memories/MEMORY.md`
- `rg -n "sq1|sq2|#935|solved-paint|solved paint|stage_n32_uncap|uncap100|eta|iteration_cap|converged|sq1_aggregate|ddm_sq2" .omx/research experiments reports docs .ralph`
- `rg -n "sq1|sq2|#935|uncap100|solved-paint|solved paint|eta_net_pooled|R8|pose-bank" .omx/research/ddm_sq2_20260804 .omx/research/scorer_batch_20260804.md .omx/state/main_hot_state.md .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_*`
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered to `trajectory_derived_stopping_law_v1`
- `rg -n "#935|sq1|sq2|solved-paint|solved paint" .omx/state .omx/research/harness_tasklist_bridge_20260803.jsonl .omx/state/canonical_task_status.jsonl`
- `sed -n '1,14p' .omx/state/active_lane_dispatch_claims.md`

Findings beyond the charter seeds:

- `.omx/state/canonical_task_status.jsonl` records the source-verified sm1 correction: the original 25-step sq1 receipt had `{'dec@25': 31, 'dec@20': 1}` and `truth` start won 0/32, so the multi-start rider was unmeasured for the sq1 headline; it also established under-convergence as a live owed item.
- `.omx/state/canonical_equations_registry.jsonl` now includes `trajectory_derived_stopping_law_v1`, whose scope explicitly excludes using an iteration cap as a convergence certificate. That confirms the sq2 uncap100 result is a higher floor, not a converged stop.
- `.omx/state/active_lane_dispatch_claims.md` records sl2 as completed bounded n32 SQ2 persisted-frame + terminal-pose work only, with no full-n600/evaluate.py/archive claim. It does not satisfy this charter's n600 promotion gate and does not conflict with this receipt.
- `.omx/state/main_hot_state.md` still carries sq1/sq2 as a scorer-gated route and separately records the later sl2 n32 result as non-promotable bounded work. No direct sq2-specific stronger completion was found in the searched canonical index/DAG scope beyond this memo, the gate JSON, scorer batch append, and the sl2 bounded n32 receipt.

Plan change from recall: do not treat `iteration_cap_*` rows as convergence; do not fire n600 while the R8 pose guard fails; keep the only follow-on as a queued 200-step floor rung with an explicit aggregate-and-R8 gate before any receiver-closed n600 spend.

## 2026-08-05 Resume Verification

- MEASURED: the charter stage command was re-executed with `--resume`; it reported `32 rows on disk, 0 remaining` and left the stage receipt hash unchanged at `dc7ecfe5c1578cc6a7f2668c070f04251b7e570a3e288d2789364d4e8ecead0b`.
- MEASURED: the aggregate command was re-executed and reproduced the existing aggregate hash `f6d5ef091fd574d34fbc06cf4230c13a4b1654db94600b6fcf822d221f1c113a`.
- MEASURED aggregate values after rerun: eta `0.9112579957356077`; stop census 21/32 `iteration_cap_best_at_cap`, 11/32 `iteration_cap_before_plateau`, 0/32 converged; solved-paint subset mean d_pose `0.07768548923741037`.
- DERIVED: the gate disposition remains unchanged: no receiver-closed n600 build and no full-n600 scorer row because the receipt is still cap-class and the R8 pose-bank erosion remains `+0.7968937215422536` vs the `+0.005` allowance.

## Next If Resumed

The uncap100 run completed and aggregated. Result: higher floor, not convergence.

- MEASURED eta: `0.9112579957356077`
- MEASURED stop census, n=32: 21/32 `iteration_cap_best_at_cap`, 11/32 `iteration_cap_before_plateau`, 0/32 convergence stops.
- MEASURED subset pose mean after solved paint: `0.07768548923741037`, pose term `0.8813937215422536`.
- DERIVED pre-pose delta S from the aggregate: `-0.13744489822327935`.
- DERIVED R8 pose-bank erosion vs `0.0845`: `+0.7968937215422536`, exceeding the `+0.005` allowance.
- DISPOSITION: no receiver-closed n600 candidate build and no full n600 scorer run.

Receipt hashes:

- `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap100_sq2.json`
  - bytes: 188487
  - sha256: `dc7ecfe5c1578cc6a7f2668c070f04251b7e570a3e288d2789364d4e8ecead0b`
- `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32_uncap100_sq2.json`
  - bytes: 6285
  - sha256: `f6d5ef091fd574d34fbc06cf4230c13a4b1654db94600b6fcf822d221f1c113a`
- `.omx/research/ddm_sq2_20260804/sq2_gate_verdict.json`

NEXT-IF-RESUMED: the only direct continuation is an optional 200-step rung, because uncap100 still hit iteration-cap class on every row. If fired, use the same selected n32 pairs, write a new `sq1_stage_n32_uncap200_*` receipt, aggregate before any n600 spend, and refuse promotion unless both non-cap convergence and R8 pose-bank accounting pass. Given the uncap100 wall time (`15006.8s`) and the pose-bank failure, this follow-on is QUEUED-WITH-FIRE-ORDER, not fired in this run.
