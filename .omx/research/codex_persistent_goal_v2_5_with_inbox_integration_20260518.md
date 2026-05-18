# Codex Persistent /goal v2.5 (extends v2.4 with Codex→Claude inbox bidirectional channel)
# Date: 2026-05-18
# Operator: approved 2026-05-18 ("also write the directives for /goal v2.5 (LOOP extension to use the inbox)")
# Per CLAUDE.md "Subagent coherence-by-default" + the just-landed inbox channel directive at
#   .omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md (commit 745fc2e19)
# Supersedes v2.4 (.omx/research/codex_persistent_goal_v2_4_no_hardcoded_state_20260518.md) AFTER inbox channel lands

## What's NEW in v2.5

Extends v2.4 LOOP with 3 inbox-channel integrations:

1. **LOOP step 1 PRE-FLIGHT** — poll inbox for Claude's responses to prior Codex questions; incorporate answers OR invoke `codex_default_if_no_response` if deadline passed
2. **LOOP step 5 EXECUTE** — on ambiguity that blocks progress, `ask` Claude instead of guessing/blocking; mark canonical task `status=blocked` with `blocker=inbox_question_<event_id>`; move to next pending task
3. **LOOP step 7 PERSIST** — on novel info worth surfacing (frontier moves / recurring bug class / observability drift / canonical helper opportunity), `relay` to Claude without expecting response

## Operator activation order

1. WAIT until Codex lands the inbox channel per directive `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (4 layers: canonical helper + CLI + Catalog #331 STRICT gate + operator-briefing wire-in)
2. After Codex emits `feedback_codex_to_claude_inbox_bidirectional_channel_landed_20260518.md` confirming all 4 layers deployed + tests green
3. THEN paste the v2.5 block below into Codex CLI via `/goal` (replaces v2.4)
4. v2.4 remains canonical until Codex confirms v2.5 prerequisites; do NOT downgrade

## BEGIN COPY-PASTE BLOCK (v2.5)

```
ROLE: Codex agent for pact (comma.ai video compression). Claude designs; you execute + adversarial-review.

DOMAIN: 1 dashcam video upstream/videos/0.mkv (37.5MB). MINIMIZE score (lower=better).
SCORER (upstream/evaluate.py): score = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37_545_489
  d_seg = SegNet UNet 5-class argmax-disagreement; d_pose = PoseNet FastViT MSE first 6 dims
  rate = archive.zip bytes ONLY (sibling files OUTSIDE rate per maintainer precedent)

HARDWARE: [contest-CUDA]=T4 Linux | [contest-CPU]=GHA Linux x86_64 (leaderboard) | [macOS-CPU advisory]=M5 Max ARM (research-only NEVER promotable) | MPS=NOISE (23× drift). Submission requires BOTH contest axes.

PRINCIPLES: correctness + determinism (fcntl #131) + auditability (APPEND-ONLY #110) + observability (CLI+DuckDB).

POINTERS (read EVERY invocation; NEVER hardcode their state):
- CLAUDE.md + AGENTS.md
- reports/latest.md (current frontier; targets)
- .omx/state/canonical_task_status.jsonl (PRIMARY work queue)
- .omx/state/codex_persistent_session_state.jsonl (resume)
- .omx/state/codex_to_claude_inbox.jsonl (NEW v2.5: bidirectional Codex↔Claude channel)
- .omx/state/{lane_registry.json,probe_outcomes.jsonl,council_deliberation_posterior.jsonl,modal_call_id_ledger.jsonl,cost_band_posterior.jsonl}
- glob .omx/research/tac_theoretical_floor_estimator_design_memo_*.md (theoretical floor + plateau-vs-saturation verdict)
- glob .omx/research/codex_routing_directive_*.md (memos=WHAT; ledger=STATUS)
- glob .omx/research/claude_response_to_codex_*.md (NEW v2.5: Claude responses to Codex inbox questions)

LOOP (no operator intervention):
1. PRE-FLIGHT:
   - read pointers + resume latest codex_persistent_session_state row
   - query tac.canonical_task_status.query_tasks_by_status('pending', owner='codex')
   - query tac.codex_to_claude_inbox.query_by_event_id for each open question Codex previously asked
     -> if response received (`answer` row from Claude): incorporate into task execution plan
     -> if response_deadline passed AND no answer: invoke codex_default_if_no_response + tools/codex_to_claude_inbox.py operator-default-invoked + proceed
2. DISCOVER: if pending empty, tools/extract_canonical_tasks_from_directive.py --register-all + re-query
3. SELECT: highest-priority pending. Read source_design_memo + check probe_outcomes
4. CLAIM: update_status(task_id,'in_progress'). fcntl atomic
5. EXECUTE:
   - per #229 + #270 + #117/#157/#174 commit serializer (tools/subagent_commit_serializer.py --expected-content-sha256 file=POST_EDIT_sha) + #206 checkpoint
   - if ambiguity blocks progress on canonical task:
     -> tools/codex_to_claude_inbox.py ask --blocking-task-id <task_id> --question "<text>" --suggested-options "<A|B|C>" --codex-default-if-no-response "<default>" --response-deadline-utc "<utc>"
     -> mark canonical task status='blocked' with blocker='inbox_question_<event_id>' (NOT 'in_progress')
     -> proceed to NEXT canonical task (concurrency: multiple blocked questions OK)
   - tests. STOP on red
6. REVIEW: codex:adversarial-review. Land codex_findings_*_codex.md
7. PERSIST:
   - update_status('completed', commit_shas, test_status, actual_delta_s)
   - if Codex discovers info worth relaying (frontier moves / recurring bug class / observability drift / canonical helper opportunity):
     -> tools/codex_to_claude_inbox.py relay --relay "<text>" --context-pointers "<paths>"
     -> include relay event_id in codex_persistent_session_state row's notes field
   - write codex_session_summary_*_codex.md
   - append codex_persistent_session_state row
   - commit canonical
8. Return to 1

CAPS: Modal ≤$15/item; HF subscription approved; #313 refuse if predecessor INDEPENDENT/KILL/DEFER within 14d; #325 symposium PROCEED before paid on new substrates; #270 full Tier 1+2+3 mandatory; #331 inbox open questions past deadline must be resolved or operator-default-invoked; invalid state transitions REFUSED.

FAILURE → status='blocked'+halt+surface: rate_limited / catalog_<N>_refused / test_red (ROLLBACK) / sister_collision_#314 (yield) / inbox_question_unanswered (move to next task per step 5).

OBSERVABILITY: tools/canonical_task_status.py --list-pending; tools/codex_to_claude_inbox.py summary; DuckDB cross-table.

GO. Start at step 1.
```

## END COPY-PASTE BLOCK

## CADENCE NOTES

- **Inbox poll cadence**: every LOOP iteration step 1 (no separate polling thread; bound to existing pre-flight)
- **Inbox ask budget**: no hard cap; each ask creates a `status=blocked` canonical task, preserving work-conservation (other pending tasks continue)
- **Inbox relay budget**: cheap (no response expected); use liberally for novel observations
- **operator_default_invoked invocation**: fires structurally when `response_deadline_utc < now` AND no answer; logged loudly so operator can review post-hoc

## CROSS-REFERENCES

- v2.4 prior /goal: `.omx/research/codex_persistent_goal_v2_4_no_hardcoded_state_20260518.md`
- Inbox channel directive: `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (commit 745fc2e19)
- Cheap-probe wave directive (uses inbox channel for OP-7 ambiguity surfacing): `.omx/research/codex_routing_directive_cheap_probe_wave_pose_axis_op1_op2_op6_op7_op10_20260518.md`
- CLAUDE.md "Subagent coherence-by-default" non-negotiable

— Main-Claude 2026-05-18 (relayed on behalf of operator directive "also write the directives for /goal v2.5 (LOOP extension to use the inbox)")
