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
