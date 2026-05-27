# Codex Findings - Final Rate Feedback Learning Guard - 2026-05-27T16:08Z

## Scope

Queue-owned final-rate feedback refresh for the current `[contest-CPU]`
frontier archive after the end-to-end materializer sweep:

- archive SHA-256:
  `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`
- archive bytes: `178546`
- CPU frontier score: `0.19202062679074616`
- refresh artifact:
  `.omx/research/frontier_rate_attack_feedback_cycle_final_rate_target_profile_20260527T1705Z/`
- external execution/result root:
  `/Volumes/VertigoDataTier/experiments/results/frontier_rate_attack_feedback/final_rate_target_profile_20260527T1705Z/`

## Findings

The queue-owned final-rate sweep produced two materializer observations and both
were receiver-satisfied but rate-flat:

- `packet_member_recompress_v1`: `saved_bytes=0`
- `packet_member_zip_header_elide_v1`: `saved_bytes=0`

`materializer_feedback_bridge.json` now records both as demoted for this matching
archive class with
`receiver_contract_satisfied_but_no_archive_delta`. The rate-budget preservation
plan is intentionally inactive with `no_rate_positive_rows_to_preserve`, so
distortion-budget spending is refused instead of inventing a repair budget from
zero freed bytes.

## Bug Class Closed

The upstream chain compiler queue previously allowed an unbound multisurface
chain to silently inherit PR103 byte-range defaults. That was a false-context
risk: the generated queue could run PR103 schema/runtime/archive inputs while the
operator intended current-frontier PR110/FEC10 optimization.

`build_frontier_byte_range_stage_inputs` now disables PR103 defaults whenever an
unbound chain is missing `payload_grammar_schema_manifest`. The emitted
`byte_range_stage_inputs.json` proves the current state:

- `local_chain_queueable=false`
- `local_chain_command=[]`
- `default_pr103_context_disabled=true`
- `default_pr103_context_disable_reason=unbound_chain_missing_payload_grammar_contract`
- blockers include missing schema manifest, beam probes, runtime dir, source
  archive, and member name

This preserves the PR103 control-arm path when explicitly bound, but prevents it
from being used as an implicit stand-in for the current frontier.

## Target Profile Canonicalization

The feedback refresh now carries a typed target optimization profile instead of
letting tools implicitly assume the contest video. The regenerated artifact has:

- schema: `frontier_rate_attack_target_optimization_profile.v1`
- target profile id: `contest_video_0`
- target mode: `contest_video_overfit`
- declared target video: `upstream/videos/0.mkv`
- target video SHA-256:
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- `profile_ready=true`

This intentionally permits contest-video overfitting for this contest, while
making the target an input contract. Corpus or hybrid runs must bind a corpus
manifest before execution, so the same materializer/search/receiver machinery
can be reused for comma10k19-style corpora instead of forking ad hoc scripts.

## Automation Guard

Auxiliary queue execution in `tools/run_frontier_rate_attack_feedback_cycle.py`
now uses a queue-local SQLite state file with an explicit noncanonical-state
rationale. This prevents regenerated child queues with the same queue id from
colliding with stale global `.omx/state/experiment_queue_*.sqlite` rows while
still preserving the experiment-queue orphan/definition-drift guards.

The bounded 17:05Z refresh also rewrote two embedded `--state` references inside
the operation-materializer auxiliary queue so downstream harvest steps read the
same artifact-local SQLite state that the worker wrote. That closes the
follow-up failure where a queue step succeeded locally but the embedded harvest
looked at an empty canonical state.

## Validation

- `ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py tools/run_frontier_rate_attack_feedback_cycle.py`
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` (`54 passed`)
- `pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_campaign_queue.py -q` (`130 passed`)
- `py_compile` for `tools/run_frontier_rate_attack_feedback_cycle.py` and
  `tools/build_frontier_byte_range_stage_inputs.py`
- `tools/experiment_queue.py validate` on the regenerated operation-chain queue
- bounded local execution of the chain compiler through `emit_byte_range_stage_inputs`
- bounded auxiliary execution: three queues executed, no failures; chain
  compiler `success_count=2`, operation materializer `success_count=2`,
  receiver repair `success_count=1`

## Next Required Binding

The next real rate-attack step is not another outer-ZIP pass. It is to build a
current-frontier payload grammar/schema manifest and runtime/archive binding for
the FEC10/DQS1 inner selector stream, then rerun the same generic byte-range and
chain compiler surfaces against that declared context.
