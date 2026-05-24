# Codex Findings: Operator Storage Waterfall

UTC: 2026-05-24T02:26:48Z

## Scope

This pass canonicalized the storage policy needed for queue/DAG materializer
work under disk pressure. The durable policy is: use external work tiers first,
derive cold-store roots from the same order, and keep local disk disabled unless
an operator explicitly opts in.

## Findings

1. Storage preflight callers were carrying storage roots and cold-store roots as
   repeated local configuration, which makes drift likely and makes proactive
   cleanup behavior hard to audit.
2. Move-mode cleanup previously required explicit cold-store roots at each
   caller. That was safe but too manual for autonomous queue execution.
3. Storage and cleanup plan artifacts needed embedded catalog metadata so
   generated plan, cleanup, and journal files carry their own custody boundary.

## Fixes Landed

- Added `comma_lab.operator_storage_waterfall` with policy id
  `operator_storage_waterfall.v1`.
- Default work tier order is:
  1. `/Volumes/VertigoDataTier/pact`
  2. `/Volumes/APDataStore/pact`
- Cold-store roots derive from the same order as `<tier>/cold_store`.
- `comma_lab.storage_tiers.DEFAULT_TIERS` now references the operator policy.
- Scheduler storage preflight now emits the policy payload, artifact catalog
  metadata, storage plan path, cleanup plan path, journal path, and lifecycle
  kind into experiment metadata and telemetry.
- `tools/plan_experiment_storage.py` and
  `tools/compact_experiment_artifacts.py` now include policy/catalog metadata
  in their JSON outputs.
- Materializer and DQS1 queue tests were updated so move-mode cleanup can use
  policy-derived cold-store defaults instead of per-caller repeated roots.
- `.gitignore` now has generic queue-owned storage/preflight JSON patterns so
  future storage policy consumers do not accidentally track per-run plan,
  cleanup, or journal files.

## Self-Protection

New and updated tests assert:

- policy tier order is VertigoDataTier then APDataStore;
- local disk is disabled by default and explicitly marked opt-in only;
- cold-store roots follow tier order;
- absolute or parent-directory cold-store subdirs are rejected;
- scheduler preflight uses default policy roots;
- storage and cleanup tools emit artifact catalog metadata;
- DQS1 and materializer queue builders inherit policy cold-store defaults.

## Authority Boundary

Storage policy artifacts are operational custody and cleanup planning artifacts.
They are false-authority by construction and cannot claim scores, promote
candidates, rank/kill candidates, or mark anything ready for exact eval.

## Remaining Wiring

The runner path can still delegate cold-store defaults to the downstream queue
builder rather than printing default cold roots in every wrapper command. That
is acceptable for this landing because the canonical policy is now the consumer
source of truth, but a future operator UX pass can make those inherited defaults
more visible in dry-run output.
