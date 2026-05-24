# Codex Findings: Inverse-Steganalysis Authority Audit

generated_at_utc: 2026-05-24T18:31:32Z
agent: codex
scope: read-mostly adversarial audit of inverse-steganalysis/action-surface implementation, materialization wiring, queue/DAG authority, and exact/proxy separation
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Current Verdict

The repository has real inverse-steganalysis apparatus, not just prose. The
current implementation includes MLX/scorer response ingestion, inverse-scorer
decision surfaces, a discrete action functional with water-bucket selection,
byte-shaving campaign composition, PacketIR lowering, family-agnostic
materializers, materializer work queues, exact-readiness handoff, experiment
queue control, staircase DAG planning, and strict false-authority helpers.

The apparatus is not yet a full-authority frontier engine. It is partially
end-to-end for DQS1/pairset-style byte operations and for local
family-agnostic candidate emission, but HNeRV/NeRV/bolton/non-NeRV paths are
mostly advisory or proof-chain-local until concrete materializer contexts,
runtime-consumption proof, same-runtime parity, exact-readiness audit, lane
claim, and exact auth eval close.

## Evidence Map

- `src/tac/optimization/inverse_steganalysis_acquisition.py:47` defines the
  inverse-steganalysis schemas; lines 82-120 name byte/pixel/region/boundary/
  frame/pair/batch/full-video axes; lines 761-900 build the discrete
  Riemann-sum action functional with first/second-order terms, water-bucket
  planning, and explicit false authority.
- `src/tac/optimization/scorer_inverse_decision_surface.py:1` is planning-only
  inverse scorer modeling; lines 312-434 aggregate response rows into cells and
  emit both probe and materialize operations while keeping exact auth blockers.
- `tools/build_inverse_steganalysis_action_functional.py:124` accepts scorer
  response, inverse-scorer surface, MLX acquisition, byte-shaving signal,
  campaign, exact-auth calibration, and queue-performance inputs; lines 278-314
  require runtime/cache identity for queue-performance feedback.
- `src/tac/optimization/byte_shaving_campaign.py:1390` ranks units and builds
  prefix/combo/permutation/operation-set ladders; lines 2254-2400 bridge
  selected water buckets into byte-shaving units or explicit compiler gaps;
  lines 2773-2902 lower supported compiler hints into operation-set provenance.
- `src/comma_lab/scheduler/byte_shaving_materializer_registry.py:150` registers
  executable DQS1, inverse-scorer, archive-section, packet-member, and tensor
  adapters plus non-executable contracts for the rest; lines 557-660 fail
  closed when an operation lacks an executable candidate-emitting materializer.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py:543` lowers PacketIR
  to materializer backlog rows; lines 593-681 merge PacketIR rows into the
  authoritative backlog; lines 1768-2040 compile backlog rows into local proof
  chain work; lines 2125-2286 add harvest/readiness/dispatch-plan follow-ups.
- `src/tac/optimization/family_agnostic_materializers.py:1` can emit
  byte-closed archive-section, packet-member, and tensor candidates, but lines
  429-487 make receiver/runtime proof mandatory before readiness.
- `src/tac/optimization/proxy_candidate_contract.py:15` and
  `src/tac/local_acceleration/mlx_score_calibration.py:54` enforce false
  authority for proxy/MLX signals; `src/tac/auth_eval_schema.py:415` requires
  strict contest CPU/CUDA payloads for auth-axis calibration targets.
- `src/tac/optimizer/exact_readiness.py:1368` blocks score/proxy rows and
  requires archive/runtime/proof/claim checks; lines 1664-1781 produce
  exact-eval-ready/no-score rows that still require a dispatch claim.
- `src/comma_lab/scheduler/experiment_queue.py:16` is the queue authority
  surface; lines 980-1035 gate execution by mode, dependencies, local/cloud
  allowlists, and resource concurrency. `src/comma_lab/scheduler/staircase_dag.py:1`
  is explicitly planning-only and keeps score semantics in queue/auth surfaces.

## Missing End-to-End Links

- Concrete high-level inverse cells often fall to
  `high_level_operation_compiler_required` unless source provenance or compiler
  hints exist; placeholders from MLX scorer-response are deliberately filtered
  as non-materializable.
- PacketIR rows are merged into backlog, but non-DQS1 targets still need
  materializer work-queue contexts and runtime proofs before they become
  exact-ready candidates.
- Family-agnostic materializers emit candidate archives, but most receivers are
  not yet closed with runtime-consumption proof and same-runtime parity, so the
  candidates stop at local proof-chain/advisory status.
- The action functional names the expanded coordinate system, but available
  empirical rows are still mostly first-order/local: rate, SegNet/PoseNet
  aggregate terms, local MLX windows, DQS1 pair/window data, and archive-level
  byte operations. Missing dense dimensions include SegNet class/boundary/
  region maps, PoseNet joint/body/time geometry, batch/runtime/scorer-device
  coupling, archive header/member/section interactions in the scorer-response
  model, operation-order synergy, and paired CPU/CUDA exact calibration across
  most candidate families.
- Runner feedback writes queue-performance and replan artifacts, but the
  inspected path still leaves the next action-functional replan as an artifact
  rather than a queue-owned follow-up row.

## Highest-EV Patch Queue

1. `tac.optimization` + `comma_lab.scheduler`: close family-agnostic
   materializer context/proof bridges for `archive_section_entropy_recode_v1`,
   `packet_member_recompress_v1`, and `tensor_factorize_v1`. Artifact:
   accepted materializer chain manifests with runtime-consumption proof and
   same-runtime parity. Verify with materializer, queue, and exact-readiness
   tests.
2. `tac.optimization.inverse_steganalysis_acquisition`: make MLX/acquisition
   rows emit concrete `operation_set_compiler` hints only when target kind,
   archive/member/tensor coordinates, and source evidence are present. Artifact:
   compiler-ready action cells instead of compiler-gap units. Verify with
   inverse-steganalysis acquisition and byte-shaving campaign tests.
3. `comma_lab.scheduler` / `tools`: enqueue queue-performance feedback as a
   paused/follow-up `experiment_queue.v1` row instead of only writing
   `queue_feedback_replan_request.json`. Artifact: queue-owned replan command
   with runtime/cache identity dependencies. Verify with runner and queue tests.
4. `tac.optimization.scorer_response_dataset`: extend response rows beyond
   aggregate rate/pose/seg into component-local maps, pair/frame/batch/runtime
   axes, and archive-member/section coordinates. Artifact: hydrated
   action-functional atoms with paired exact-auth calibration slots. Verify
   with dataset validation and action-functional tests.
5. `tac.packet_compiler`: finish PacketIR exact closure for supported operation
   sequences, including golden vectors, runtime proof vocabulary, and full-frame
   parity/rate-only control. Artifact: PacketIR chain accepted by exact
   readiness. Verify with deterministic compiler and queue lowering tests.
6. `reverse_engineering` + `tac`: make HNeRV/NeRV/bolton intake source-faithful
   before treating those surfaces as materializer inputs. Artifact: byte
   anatomy, runtime loader, export parity, and exact-smoke blockers or pass
   records. Verify with source archive replay and exact/auth smoke.
7. `tools/operator_briefing.py` / `comma_lab`: expose no-signal-loss counters
   for compiler-gap rows, PacketIR blockers, executable conversion rate,
   exact-readiness handoffs, and queue-feedback readiness. Artifact: normal
   operator flow shows what signal is blocked where. Verify with briefing
   snapshot tests.

## Red Flags

- `ready_for_exact_eval_dispatch=true` appears in exact-readiness promoted rows,
  but the durable rule is that this flag is not sufficient authority; dispatch
  still needs exact-dispatch audit and a real lane claim.
- Planning-only inverse-scorer/action-functional rows are intentionally
  executable as local artifacts in some paths, but they do not emit candidate
  archives. Consumers must keep honoring `emits_candidate_archive=false` and
  planning-only blockers.
- The action surface can express second-order/synergy terms, but the current
  empirical population is sparse relative to the dimensionality claimed.
- HNeRV/NeRV/bolton/non-NeRV are present in adapter descriptions and
  materializer contracts, but not yet closed as source-faithful, exact-ready
  frontier-moving loops.
- Queue feedback is becoming system intelligence, but the last inspected step
  is still artifact-first. A queued replan edge is the next no-signal-loss
  boundary.
