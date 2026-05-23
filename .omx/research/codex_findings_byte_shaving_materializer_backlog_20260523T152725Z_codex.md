# Codex Findings - Byte-Shaving Materializer Backlog

Timestamp: 2026-05-23T15:27:25Z
Agent: Codex
Scope: byte-shaving plan -> materializer registry/backlog -> DQS1 local-first queue bridge

## What Landed

The byte-shaving compiler now emits a canonical `byte_shaving_materializer_backlog.v1`
surface. Blocked candidate rows are no longer just side text in `blocked_rows`;
they are aggregated into ranked adapter/receiver work orders with:

- `gap_class`
- `receiver_contract_status`
- materializer/target/unit/operation identifiers
- affected units and source selections
- blocker counts
- summed expected score gain and candidate saved bytes
- false-authority gates preserved

The DQS1 drop-pair adapter now carries an explicit receiver contract:

- `receiver_contract_id=dqs1_pairset_decoderq_receiver.v1`
- `receiver_contract_kind=archive_charged_pairset_runtime_selector`
- `cooperative_receiver_required=true`

The materializer registry also exposes the cooperative-receiver packet grammar
hook registry so byte-shaving materializers and packet/compiler work can be
planned against the same receiver/grammar boundary instead of drifting apart.

The CLI `tools/build_byte_shaving_campaign_queue.py` can now write a standalone
materializer backlog artifact with `--materializer-backlog-out`, guarded by the
same no-overwrite-by-default artifact writer and optional expected SHA.

## Smoke Artifact

Command:

```bash
.venv/bin/python tools/build_byte_shaving_campaign_queue.py \
  --plan .omx/research/byte_shaving_campaign_master_gradient_plan_20260523T144718Z.json \
  --materialization-out .omx/research/byte_shaving_campaign_master_gradient_materializer_backlog_smoke_20260523T152725Z.json \
  --portfolio-out .omx/research/byte_shaving_campaign_master_gradient_materializer_backlog_portfolio_20260523T152725Z.json \
  --action-summary-out .omx/research/byte_shaving_campaign_master_gradient_materializer_backlog_action_summary_20260523T152725Z.json \
  --materializer-backlog-out .omx/research/byte_shaving_campaign_master_gradient_materializer_backlog_20260523T152725Z.json \
  --repo-root . \
  --candidate-limit 8
```

Result:

- executable rows: 0
- blocked rows: 36
- backlog rows: 3
- score authority: false
- queue: null

Top backlog rows:

1. `byte_range/entropy_recode`: 36 blocked selections, 4 affected byte-span
   units, 268542 candidate saved bytes summed, expected score gain sum
   `0.17058307590560334`, receiver status
   `receiver_target_contract_required`.
2. `byte_range/null_remove_or_seed`: 31 blocked selections, 4 affected
   byte-span units, 227720 candidate saved bytes summed, expected score gain
   sum `0.14340138172125005`, receiver status
   `receiver_target_contract_required`.
3. `byte_range/delta_encode`: 16 blocked selections, 2 affected byte-span
   units, 64432 candidate saved bytes summed, expected score gain sum
   `0.03790735020124522`, receiver status
   `receiver_target_contract_required`.

Interpretation: for the current master-gradient byte-range plan, the throughput
bottleneck is not DQS1 pairset scheduling. It is missing archive/byte-range
materializer and receiver contracts, with entropy recoding first in the work
queue.

## Optimizer Relevance

Muon, AdamW, NAMO, Muown, and related optimizer work are relevant upstream and
as acquisition features, but not as direct post-training byte-shaving authority.
They should feed:

- trained candidate provenance;
- optimizer family and schedule metadata;
- convergence/cost priors;
- architecture/weight-spectrum/normalization signals;
- uncertainty estimates used by the learned acquisition function.

They should not bypass materializer/runtime custody or exact-eval authority.
Optimizer-improved HNeRV/NeRV/non-NeRV substrates still have to emit byte-closed
archives, receiver contracts, locality proofs, local advisory rows, and exact
auth-eval readiness artifacts before promotion.

Repo wiring to use:

- `src/tac/optimization/muon.py` preserves no-score-authority semantics and
  exposes Muon parameter partition telemetry.
- `src/tac/optimization/optimizer_scheduler_registry.py` is the recipe catalog
  for AdamW, Muon+AdamW, scheduler, and newer optimizer probes.
- `src/tac/optimization/optimizer_guided_candidate_generation.py` emits
  deterministic proxy candidates with solver-stack wire-in.
- `src/tac/optimization/pr95_muon_local_training_integration.py` adapts PR95
  Muon local training manifests into candidate shape.
- `src/tac/optimization/mlx_dynamic_learned_sweep.py` is the same-candidate
  optimizer/scheduler/MLX ablation surface.
- `tools/build_optimizer_candidate_queue.py` is the canonical adapter from raw
  optimizer-guided queues to `optimizer_candidate_queue_v1`; do not feed raw
  `optimizer_guided_candidate_queue_v1` directly into the byte-shaving surface.
- `src/tac/optimization/optimizer_signal_atoms.py` should be used when optimizer
  rows are signal/provenance only and not byte-shaving units.

Recent optimizer literature reinforces this split:

- BOCA treats cheap approximations as first-class fidelities for expensive
  black-box optimization: https://proceedings.mlr.press/v70/kandasamy17a.html
- FABOLAS models both loss and training time to trade information gain against
  cost: https://arxiv.org/abs/1605.07079
- Hyperband and BOHB motivate bracketed, early-stopped, high-throughput
  resource allocation: https://www.jmlr.org/beta/papers/v18/16-558.html and
  https://proceedings.mlr.press/v80/falkner18a.html
- TuRBO's local trust-region BO is a good fit for DQS1/pairset neighborhoods,
  byte-span neighborhoods, and substrate-family local basins:
  https://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization
- NAMO/NAMO-D and Muown show that Muon-family optimizer geometry is still moving
  quickly; these should become candidate-generation/training priors, not score
  authority: https://arxiv.org/abs/2602.17080 and
  https://arxiv.org/abs/2605.10797

## Next Engineering Hooks

1. Add the first non-DQS1 executable materializer for `byte_range/entropy_recode`.
   It must produce a byte-closed archive/runtime packet and a cooperative
   receiver contract, not just a rewritten byte span.
2. Generalize DQS1 harvest observations into a byte-shaving observation ledger
   consumed by the next signal-surface build.
3. Add learned acquisition fields to units: posterior mean/variance, axis
   calibration, estimated wall-clock, storage bytes, materializer readiness,
   optimizer provenance, and receiver-contract status.
4. Feed the backlog and learned acquisition scores into `staircase_dag` so
   local CPU/MLX work saturates available resources while preserving storage
   tier and false-authority guards.
5. Treat cooperative receiver work as the executable boundary: every optimizer,
   packet, byte, frame, pair, pixel, tensor, and archive-section operation
   needs a materializer plus deterministic receiver/runtime consumption proof.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign_queue.py`
  -> 11 passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign.py`
  -> 8 passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_autopilot.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_cooperative_receiver_packet_grammars.py`
  -> 114 passed
- `.venv/bin/ruff check ...`
  -> passed
- `git diff --check`
  -> passed
