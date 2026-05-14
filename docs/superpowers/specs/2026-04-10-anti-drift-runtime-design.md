# Anti-Drift Runtime Hardening Design

## Goal

Eliminate the two recurring split-brain failures in this repo:

1. promoted score and report surfaces drifting out of sync
2. training and cloud execution logic drifting away from the hardened `tac` path

The design makes one machine-readable promoted-result record the root truth, regenerates every mirrored surface from that record, and moves future cloud launchers toward thin wrappers over `src/tac`.

## Context

The immediate regression is concrete:

- [reports/results.jsonl](../../../reports/results.jsonl) already contains the authoritative `1.33` promoted result
- [reports/timeline.jsonl](../../../reports/timeline.jsonl) already contains the `1.33` promotion event
- [reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt](../../../reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt) is the authoritative evidence
- but [reports/raw/robust_current-current_workflow-cpu-summary.json](../../../reports/raw/robust_current-current_workflow-cpu-summary.json) and [reports/latest.md](../../../reports/latest.md) still claim `1.51`

That kind of drift is unacceptable at promotion and publication boundaries.

## Design Principles

### Single canonical truth

Promoted score state must have one root record. Everything else is derived.

### Hybrid enforcement

- normal operator flows may auto-repair stale mirrors
- promotion and publish boundaries must fail hard if consistency cannot be proven

### `tac` owns semantics

Anything that changes training behavior, checkpoint semantics, metadata format, or evaluation semantics belongs in `src/tac`.

### Wrappers own platform mechanics only

Kaggle, Modal, Colab, GCP, and local launchers may choose paths, env vars, and remote execution settings, but must not fork model or checkpoint semantics.

## Canonical Data Model

Add one canonical promoted-result record for Track B under repo-controlled state. The record must contain:

- schema version
- run id
- track
- named accounting views, for example:
  - `current_workflow`
  - `rule_faithful` when available
- for each view:
  - score
  - pose distortion
  - seg distortion
  - rate
  - archive bytes
- authoritative raw report path
- canonical copied report path
- promoted artifact path
- digests for:
  - promoted artifact
  - authoritative raw report
  - canonical copied report
- provenance:
  - platform
  - variant
  - epoch
  - eval mode
  - timestamp
  - upstream commit
  - sample count

There are two distinct durability classes:

- **current promoted pointer**
  - one mutable record
  - source of truth for mirrored “latest” surfaces
- **append-only history**
  - `reports/results.jsonl`
  - `reports/timeline.jsonl`
  - updated by deterministic merge rules keyed by run/event identity
  - never regenerated wholesale

This record becomes the source for:

- [reports/raw/robust_current-current_workflow-cpu-summary.json](../../../reports/raw/robust_current-current_workflow-cpu-summary.json)
- [reports/raw/robust_current-current_workflow-cpu-report.txt](../../../reports/raw/robust_current-current_workflow-cpu-report.txt)
- [reports/results.jsonl](../../../reports/results.jsonl)
- [reports/timeline.jsonl](../../../reports/timeline.jsonl)
- [reports/latest.md](../../../reports/latest.md)
- [.omx/state/current_focus.md](../../../.omx/state/current_focus.md)
- [.omx/state/next_experiments.md](../../../.omx/state/next_experiments.md)
- [.omx/research/findings.md](../../../.omx/research/findings.md)
- [.ralph/run_log.md](../../../.ralph/run_log.md)

## Projection Pipeline

Add a projection/sync layer under `src/comma_lab` that:

1. acquires a repo-local promotion/state lock
2. loads the canonical promoted-result record
3. validates the authoritative report exists, parses cleanly, and matches the stored digests
4. rewrites every mirrored surface deterministically
5. merges append-only ledgers by stable identity instead of regenerating history
6. writes atomically
7. emits a drift report when an existing mirror disagrees before rewrite

Operator rule:

- evidence files may be produced by experiments
- promoted mirrors may only be rewritten by the sync layer

## Boundary Gates

### Promotion gate

Promotion must fail unless:

- authoritative report exists
- canonical promoted record validates
- promoted artifact digest matches the record
- authoritative report digest matches the record
- promoted artifact exists
- regenerated summary matches the canonical record
- results ledger and timeline event either already match or can be updated deterministically

### Publish/report gate

Static report generation must fail if:

- canonical promoted result exists
- mirrored canonical score/report surfaces are stale

### State doctor

Add a read-only drift audit command that reports mismatches such as:

- promoted record says `1.33`, summary says `1.51`
- manifest says `running_managed_session`, but process is gone
- status file says Kaggle run is live, but Kaggle reports `NOT_PUSHED` or timeout

## `tac` Consolidation

Move toward this boundary:

### `src/tac` owns

- training loops
- checkpoint selection
- resume-state save/load
- int8 export/load
- metadata emission
- evaluation/proxy parsing helpers
- launchable config schemas for common postfilter families

### platform wrappers own

- environment bootstrap
- remote dataset mounting
- CLI argument bridging
- job metadata manifests
- status polling and output ingestion

The rule is simple:

- if a change alters model behavior or artifact semantics, it lands in `src/tac`
- if a change alters remote execution or I/O plumbing, it may live in wrappers

`tac` also needs an explicit compatibility contract:

- versioned checkpoint metadata schema
- versioned resume-state schema
- adapter policy for reading older artifacts
- cloud wrappers may emit only supported schema versions

## UX / DX Surfaces

Target operator commands:

- `comma-lab state doctor`
- `comma-lab state sync`
- `comma-lab promote <result-record>`
- `comma-lab cloud launch <platform> <experiment>`
- `comma-lab cloud ingest <platform> <run>`

Desired behavior:

- one obvious path for promotion
- one obvious path for drift repair
- one obvious path for cloud launch
- no need to hand-edit mirrored markdown/json files

## Error Handling

- all generated files written via atomic temp-and-rename
- sync, promote, and ingest operations serialized with a lock so concurrent writers cannot clobber canonical state
- generated surfaces should be deterministic and idempotent
- stale manifests must be downgraded from “running” if their backing process or platform state is gone
- sync should refuse heuristic promotion; if evidence is ambiguous, fail hard

## Testing Strategy

### Score/report drift

- regression test for the exact `1.51` summary plus `1.33` ledger split-brain bug
- golden tests for generated markdown/json projections
- integration test for promote + sync on a fake authoritative report
- doctor test for stale summary/report/ledger combinations

### Runtime-path drift

- tests proving cloud launchers call into `tac` config/entrypoint helpers rather than embedding divergent training semantics
- metadata compatibility tests across local and cloud outputs
- stale-process status downgrade tests

## Delivery Order

1. add canonical promoted-result state model and sync pipeline
2. repair the current `1.33` split-brain surfaces with that pipeline
3. add doctor and publish/promotion gates
4. move cloud launch surfaces toward `tac` config/entrypoint ownership
5. add regression tests so the current failure cannot recur

## Success Criteria

- one command can deterministically regenerate all promoted mirrors from canonical truth
- the repo cannot ship a stale promoted summary while ledgers say something else
- platform status files do not keep claiming “running” after the process or platform says otherwise
- future Kaggle/Modal/Colab/GCP trainers share `tac` semantics instead of drifting
