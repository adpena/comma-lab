# Codex Session Summary

Timestamp UTC: 2026-05-24T15:35:02Z

## Landed Work

- Preserved the PR95/HNeRV MLX lane as non-authoritative by labeling the current
  executable path synthetic-timing-only and carrying source-faithfulness blockers
  through PR95 MLX manifests, optimizer descriptors, queue tooling, and tests.
- Generalized materializer harvest from chain manifests to harvestable
  materializer manifests so family-agnostic candidate outputs can enter
  `optimizer_candidate_queue_v1`.
- Wired scheduler follow-up discovery through family candidate
  `json_completion_contract` postconditions so archive-section, packet-member,
  and tensor-family candidates flow to harvest/exact-readiness bridge instead
  of being skipped as permanently unwired.
- Hardened family receiver proof handling so exact readiness can accept
  `family_agnostic_runtime_consumption_proof_v1` only when it is present,
  archive-bound, false-authority clean, and passed.
- Fixed inverse-steganalysis acquisition economics to use the canonical contest
  rate denominator (`25 / 37_545_489`) and added a boundary regression for the
  stale-denominator bug class.
- Promoted water-bucket output into an explicit materialization portfolio that
  survives direct conversion, merged signal-surface construction, and campaign
  planning. Bare cells now default to a fail-closed operation-set compiler gap;
  IAS1 coordinate-cell probing is an explicit diagnostic opt-in.
- Hardened IAS1 exact-mode queue execution so candidate chains require inflate
  parity context and add `--fail-if-inflate-parity-blocked` by default.
- Preserved packet-member recompress probe artifacts as negative rate signal:
  the tested deflate-9 path increased PR95/100/101/103/105/107 archives by
  about 55 bytes and still lacks runtime-consumption proof.
- Hardened the final-byte materializer context compiler so unsupported backlog
  rows emit blocked context rows instead of disappearing as unsupported counts.
- Added a fail-closed PacketIR/compiler bridge hint for unsupported final-byte
  rows, pointing future inverse-steg portfolio lowering at deterministic
  PacketIR operation sets and required archive/runtime proofs. The operation-set
  schema/order/proof vocabulary is centralized in the deterministic compiler so
  scheduler wrappers cannot drift into duplicate mini-contracts.
- Hardened the parent/child handoff: operation-set scheduler rows now require a
  matching PacketIR operation set before execution, supported final-byte
  contexts carry the same PacketIR bridge contract, and family-agnostic receiver
  proofs reject canonical false-authority leakage.

## Verification

- Focused bridge tests: 3 passed.
- Materializer, optimizer queue, exact-readiness suite slice: 113 passed.
- Byte-shaving campaign queue: 47 passed.
- PR95 MLX truth-labeling slice: 3 passed.
- Broad inverse/materializer/exact-readiness slice: 233 passed, 1 duplicate-ZIP
  warning.
- PR95 MLX/optimizer registry slice: 32 passed.
- Byte-shaving campaign plus signal-surface builder: 27 passed.
- Final-byte operation contexts: 7 passed.
- Targeted Ruff check: passed.

## Remaining Work

- PR95 MLX is still not full source-faithful replication. Missing pieces are
  real contest/source-video loader, source-matched PR95 loss and preprocessing,
  stage schedules/hparams, QAT/resume semantics, export parity, runtime
  consumption proof, and exact CPU/CUDA auth eval.
- Family-agnostic materializers now reach source queues and exact-readiness
  blockers, but cooperative receivers and full-frame inflate parity still need
  concrete per-family implementations before promotion.
- The inverse-steganalysis/action-surface planner now carries materialization
  portfolios, but still needs the high-level operation-set compiler that maps
  bare cells into concrete family materializers instead of only reporting the
  compiler gap.
- Exact auth dispatch is still a future gate for this slice: no new lane-claim
  plus exact-eval actuator ran, and no score/frontier movement is claimed from
  these local planning, materializer, or MLX authority-boundary changes.
