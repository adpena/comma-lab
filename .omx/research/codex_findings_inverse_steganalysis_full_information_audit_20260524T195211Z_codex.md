# Codex Findings - Inverse Steganalysis Full-Information Audit

## Verdict

The repo has real inverse-steganalysis primitives, but not yet a closed-loop
automated final-rate attack. Current surfaces can build inverse scorer/action
functionals, lower a small operation set into PacketIR/materializers, queue
conservative local work, and keep exact-eval authority fail-closed. The missing
breakthrough is not another singleton byte/pixel leaf; it is broader compiler
coverage plus harvest-driven replanning.

## Implemented Today

- `tac.optimization.inverse_steganalysis_acquisition` builds multiscale action
  surfaces over byte, pixel, frame, pair, region, batch, component, and
  full-video coordinates.
- `tac.optimization.scorer_inverse_decision_surface` converts scorer-response
  datasets into planning surfaces.
- `tac.optimization.byte_shaving_campaign` converts signal surfaces into
  combo/permutation/operation-set/PacketIR campaign plans and classifies
  inverse-action water-bucket cells as queue-consumable, compiler-required, or
  leaf-probe-only.
- `tac.optimization.family_agnostic_materializers` has conservative executable
  materializers for packet recompress, archive-section entropy recode, and
  tensor factorize.
- `comma_lab.scheduler.byte_shaving_campaign_queue` builds fail-closed local
  materializer work queues.
- `tac.optimization.inverse_scorer_exact_eval_queue` bridges toward exact eval
  only after runtime proof and full-frame parity gates.

## Missing For The Intended System

- A recurring hydrator that pulls auth/runtime/scorer/archive/queue/MLX/CPU/CUDA
  calibration state into one acquisition surface every cycle.
- Broader `operation_set_compiler` coverage. The default compiled target set is
  still too small for HNeRV/PR95/NeRV-family, decoder-q, and archive-grammar
  transforms.
- Family-specific materializers for HNeRV/PR95/NeRV/decoder-q/archive-layout
  transforms.
- Batch-aware water fill with interactions, uncertainty, CPU/CUDA divergence,
  and queue throughput costs. Current water fill is mostly greedy planning.
- Runtime-consumption proofs for offset/layout-changing transforms.
- Automatic harvest -> replan -> rematerialize loop as a default actuator.

## Layering

- `tac`: inverse scorer/action math, acquisition, calibration, proxy authority
  contract, PacketIR lowering, materializers, archive/runtime proof, reusable
  codec/scorer/planner primitives.
- `comma_lab`: queue/DAG execution, experiment custody, harvest, exact-readiness
  bridge, provider dispatch plans, lane claims, operator surfaces.
- `reverse_engineering`: curated public PR/archive anatomy, adapters, and byte
  forensics.
- `.omx/research`: append-only findings and blockers only; not execution truth.

## Highest-EV Engineering Patches

1. Expand `operation_set_compiler` and materializer registry coverage for
   HNeRV/PR95 section transforms, decoder-q selective mutations, archive layout
   transforms, and scorer-response candidate materialization.
2. Make strict MLX/scorer-response rows emit compiler hints in
   `tac.local_acceleration.mlx_acquisition_batch` and
   `tac.optimization.inverse_steganalysis_acquisition`, then test that
   queue-consumable materializer counts increase.
3. Replace greedy water fill with a beam/knapsack batch optimizer with marginal
   recomputation, interaction penalties, calibration residuals, CPU/CUDA split
   handling, and queue cost.
4. Add harvest-driven replanning in `comma_lab.scheduler.materializer_chain_harvest`
   and `comma_lab.scheduler.byte_shaving_campaign_queue` so materializer
   outcomes become observations, duplicates are suppressed, and the next batch is
   enqueued.
5. Harden runtime-consumption proofs in
   `tac.optimization.family_agnostic_materializers` for layout-changing
   transforms.

## Safeguards

Keep MLX, scorer-response, PR95 preprocess, decoder-q advisory, inverse-action,
and water-bucket rows fail-closed:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Do not relax `proxy_candidate_contract`, scorer-response strict gates, or
full-frame parity/exact-auth gates. Better local planning signal must increase
candidate quality and queue throughput, not become score or dispatch authority.
