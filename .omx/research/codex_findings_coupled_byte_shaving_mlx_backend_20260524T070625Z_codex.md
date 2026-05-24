# Codex Findings: Coupled Byte-Shaving And Local MLX Backend Gap

created_at_utc: 2026-05-24T07:06:25Z
lane_id: lane_codex_mlx_coupled_byte_shaving_20260524
research_only: false
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Verdict

The inverse-steganalysis/water-fill idea is partially implemented, but it was
not fully wired into automated final byte operations before this patch.

Implemented before this pass:

- `inverse_steganalysis_discrete_action_functional.v1` can represent a
  Riemann-sum-style scorer action functional with water-bucket selected cells.
- `byte_shaving_campaign_plan.v1` can rank units, combinations, interactions,
  and bounded operation permutations.
- DQS1 materializer queue compilation can turn supported pair-drop operations
  into local-first queue actions.

Missing before this pass:

- no durable coupled-operation-set artifact between combo/permutation planning
  and execution;
- queue compilation preferred raw combo/prefix rows and could lose set identity,
  chosen operation order, and interaction context;
- the generic optimizer candidate queue did not prefer operation-set rows, so
  HNeRV/BoostNeRV/NeRV/non-NeRV grouped edits could degrade back into ordinary
  planning candidates;
- expensive L5 v2 / substrate candidates are still mostly provider/exact-eval
  plans, not local MLX/NumPy trainable backend work units.

## Landed In This Pass

- Added `byte_shaving_coupled_operation_set.v1` in `tac.optimization.byte_shaving_campaign`.
  It preserves selected operations, chosen order, active interactions, modeled
  byte/score deltas, confidence, and false-authority blockers.
- Extended `comma_lab.scheduler.byte_shaving_campaign_queue` to prefer
  `operation_set_ladder` over raw `combination_ladder` when present, and to
  preserve `operation_set_id`, `chosen_operation_sequence`, and
  `active_interactions` in materialization rows and portfolio source metadata.
- Extended `tac.optimizer.candidate_queue` so generic candidate queues prefer
  operation-set rows from byte-shaving plans. This is the path that keeps the
  abstraction useful for HNeRV variants, BoostNeRV bolt-ons, NeRV-family
  variants, tensor/codebook edits, byte-range entropy coders, and non-NeRV
  archive grammars.
- Patched `tools/run_dqs1_local_first_tranche.py` so a strict portfolio rebuild
  that fails only because the pairset observation model is inactive retries in
  exploratory mode instead of wasting a long local run at the final planning
  gate. The fallback is recorded in the tranche round payload and all other
  failures remain strict.

## Current Architecture

Reusable math and candidate semantics live in `tac`:

- scorer inverse/action functional;
- byte-shaving unit and operation-set schemas;
- operation ranking, interactions, conflicts, and permutation priors;
- candidate queue adapters for cross-family planners.

Lab orchestration lives in `comma_lab`:

- experiment queues;
- staircase DAGs;
- materializer resolution;
- local resource concurrency;
- portfolio and source metadata custody.

This is the right split. Do not move queue custody into `tac`, and do not bury
reusable MLX/NumPy backend math inside provider dispatch planners.

## Outstanding Work

Priority 1: local backend substrate bridge.

- Add a reusable local backend contract for `local_numpy` and `local_mlx` work
  units with false-authority fields, resource mapping, export manifest fields,
  and postconditions.
- Add TT5L/L5 v2 local smoke through that contract first, because those
  candidates were about to cost provider money for training/optimization.
- Keep the contract generic enough for BoostNeRV, HNeRV/PR101 variants,
  FF-NeRV/DS-NeRV/HiNeRV, VQ-VAE, Cool-Chic, Balle, byte-range entropy
  coders, and tensor/codebook bolt-ons.

Priority 2: MLX acquisition batches.

- Add `mlx_acquisition_batch.v1` under `tac.local_acceleration`.
- Ingest MLX scorer response, quality/speed delta, calibration, training/export
  manifests, and window selections.
- Emit operation sets, not single rows.
- Feed those sets into `inverse_steganalysis_acquisition` and
  `byte_shaving_campaign`.

Priority 3: materializers beyond DQS1.

- Register materializer adapters for HNeRV section transforms, HNeRV/BoostNeRV
  archive exports, tensor/codebook edits, byte-range entropy recodes, and
  inverse-scorer cell candidates.
- Every materializer must prove runtime consumption and byte-closed archive
  custody before exact-eval dispatch.

Concrete adapter source map:

- HNeRV: entropy packets, low-level repack, decoder recode, generated schema,
  packet-section transforms, wavelet transforms, factorized archive, and HDM3
  archive builders should map to `archive_section`, `byte_range`, `tensor`,
  and `packet_member` units. Target operations include `section_entropy_recode`,
  `section_proceduralize`, `quantize_tensor`, `factorize_tensor`,
  `shared_codebook_tensor`, `literal_elide`, and `member_recompress`.
- BoostNeRV and broader NeRV-family: trainers should emit
  `representation_training_probe_manifest_v1` via the shared trainer skeleton,
  then a section/tensor adapter should read parser-section manifests and archive
  custody to emit `tensor` and `archive_section` units.
- PR75/PR79/PR85/QMA9/C067 and similar candidate matrices: add an explicit
  candidate-matrix adapter. Negative archive deltas become
  `candidate_saved_bytes`; trace/component deltas become quality-cost terms;
  selected atoms/records/pairs become `operation_params`.
- Cross-family portfolios: convert family/source/exact-custody rows into
  `archive_section`, `tensor`, or `correction_target` units, and preserve family
  couplings as signal-surface `interactions`.

Priority 4: final autopilot loop.

- Planner: inverse action surface -> operation sets.
- Local executor: local NumPy/MLX smoke and export.
- Materializer: byte-closed archive/runtime proof.
- Exact readiness: CPU/CUDA axis payload blockers must pass.
- Dispatcher: claimed exact eval only after readiness.
- Learner: harvest results back into scorer-response/action-surface/posterior
  ledgers.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_dqs1_local_first_tranche.py \
  src/tac/tests/test_byte_shaving_campaign.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_optimizer_candidate_queue.py -q
```

Result: `96 passed in 1.03s`.

```bash
.venv/bin/ruff check \
  tools/run_dqs1_local_first_tranche.py \
  src/tac/tests/test_dqs1_local_first_tranche.py \
  src/tac/optimization/byte_shaving_campaign.py \
  src/comma_lab/scheduler/byte_shaving_campaign_queue.py \
  src/tac/tests/test_byte_shaving_campaign.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/optimizer/candidate_queue.py \
  src/tac/tests/test_optimizer_candidate_queue.py
```

Result: `All checks passed!`.
