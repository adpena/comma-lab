# Codex Findings: Byte-Shaving Signal Surface Builder

- utc: `2026-05-23T04:05:25Z`
- lane_id: `codex_byte_shaving_signal_surface_builder_20260523`
- evidence_axis: `[planning-only]`, `[macOS-CPU advisory]`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Landed Integration

This pass converted the byte-shaving planner's missing upstream ingestion layer
into code. `tac.optimization.byte_shaving_signal_surface_builder` now builds one
validated `byte_shaving_signal_surface.v1` from optimizer candidate queues,
master-gradient anchors, sanitized auth-eval refs, MLX calibration refs,
scorer-response datasets, X-ray hook inventory, canonical equations, and atom
ledger rows.

The new surface is intentionally planning-only. Auth-eval refs may record that
the source artifact carried a valid score claim, but raw score-authority fields
are not embedded in the surface itself. Proxy/local/MLX/scorer-response inputs
are rejected if they contain truthy score, promotion, rank/kill, or dispatch
authority.

## Bug-Class Fixes

Banach found that `archive_candidate_verified` was set after archive path and
size checks but before hashing the archive file. The candidate queue now streams
SHA-256 over the actual archive before setting verification and adds
`candidate_archive_sha256_mismatch` on disagreement.

Banach also found that queue observation was not read-only because observe
called state initialization. `observe_experiment_queue` now opens SQLite in
read-only mode and reports `definition_drift` instead of inserting/requeueing
rows. Worker execution remains the explicit state-sync path.

## DQS1 Local Result

Candidate `pairset_drop_two_r029_021_p0259_0371` completed local-first gates:

- `plan_packet`: succeeded
- `materialize`: succeeded
- `locality_controls`: succeeded
- `local_cpu_advisory`: `0.19203861709818362` `[macOS-CPU advisory]`
- drift-projected conservative contest score: `0.19203111709818363`
- current CPU auth frontier pointer: `0.19202828295713675`
- eureka margin: `-0.0000028341410468757378`
- verdict: `observe_only`, no exact-auth request

The queue was rerouted to `pairset_drop_two_r029_013_p0259_0327`; that next
candidate has completed plan and materialization and is currently at local
control/advisory gates depending on the latest worker state.

## Subagent Findings Captured

Mencius identified the next missing layer: a byte-shaving materialization bridge
from `byte_shaving_campaign_plan.v1` candidate rows into byte-closed
materialization manifests, then candidate-queue ingestion, exact-readiness
validation, and exact auth eval promotion. This is the next code tranche, not a
planner promotion path.

Franklin independently recommended the same architectural shape: keep
`byte_shaving_campaign.py` as planner/validator and add a reusable source
ingestion layer. This pass implemented that layer under a separate module and
thin CLI.

## Next Engineering Hooks

1. Add `byte_shaving_materialization.v1` and
   `tools/materialize_byte_shaving_candidate.py`.
2. Ingest materialized byte-shaving rows into `optimizer_candidate_queue_v1`
   while preserving `score_affecting_payload_changed=true` and all score
   authority false.
3. Extend exact-readiness to validate the materialization manifest, archive
   SHA/bytes, runtime-consumption proof, locality controls, and source-row
   digest before promotion to exact-ready.
4. Add autonomous DQS1 worker harvest/reroute mode so a queue can run through
   candidate boundaries without turn-by-turn manual invocations.
5. Upgrade master-gradient byte-span surfaces from theoretical saved bytes to
   grammar/materializer-proven charged-byte estimates before they influence
   queue ranking.
