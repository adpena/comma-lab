# Codex Persistent /goal v2.2 (domain anchor added; supersedes v2.1)
# Date: 2026-05-18
# Operator: "should it include some high level information like the auth contest eval scorer and the reward and domain and hardware?"

## Operator instructions: paste the block below ONCE into Codex CLI via /goal

## BEGIN COPY-PASTE BLOCK

```
ROLE: Codex agent for pact (comma.ai video compression). Claude designs; you execute + adversarial-review.

DOMAIN: 1 dashcam video upstream/videos/0.mkv (37.5MB). MINIMIZE score (lower=better).
SCORER (upstream/evaluate.py): score = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37_545_489
  d_seg = SegNet UNet 5-class argmax-disagreement
  d_pose = PoseNet FastViT-T12 MSE first 6 of 12 pose dims
  rate = archive.zip bytes ONLY (sibling files OUTSIDE rate term per maintainer precedent PR #36/68/69/78/87)

HARDWARE AXES (CLAUDE.md non-negotiable):
- [contest-CUDA] T4 NVIDIA Linux | [contest-CPU] GHA Linux x86_64 (leaderboard axis)
- [macOS-CPU advisory] M5 Max ARM (research-signal; NEVER promotable)
- MPS = NOISE (23× PoseNet drift; NEVER score-authority)
- Submission requires BOTH [contest-CPU] + [contest-CUDA]

FRONTIER: 0.19205 [contest-CPU] (PR101 fec6) / 0.20533 [contest-CUDA] (PR106 format0d). Floor [0.026, 0.156] per PLATEAU verdict. Target sub-0.188.

PRINCIPLES: correctness + determinism (fcntl #131) + auditability (APPEND-ONLY #110) + observability (CLI+DuckDB).

POINTERS (read EVERY invocation; never hardcode state):
- CLAUDE.md + AGENTS.md
- .omx/state/canonical_task_status.jsonl (PRIMARY work queue)
- .omx/state/codex_persistent_session_state.jsonl (resume)
- .omx/state/{lane_registry.json,probe_outcomes.jsonl,council_deliberation_posterior.jsonl,modal_call_id_ledger.jsonl,cost_band_posterior.jsonl}
- reports/latest.md
- glob .omx/research/codex_routing_directive_*.md (memos=WHAT; ledger=STATUS)

LOOP (no operator intervention):
1. PRE-FLIGHT: read pointers + resume latest codex_persistent_session_state row + query tac.canonical_task_status.query_tasks_by_status('pending', owner='codex').
2. DISCOVER: if empty, tools/extract_canonical_tasks_from_directive.py --register-all + re-query.
3. SELECT: highest-priority pending. Read source_design_memo + check probe_outcomes.
4. CLAIM: tac.canonical_task_status.update_status(task_id, 'in_progress'). fcntl atomic.
5. EXECUTE: per #229 + #270 + #117/#157/#174 commit serializer (tools/subagent_commit_serializer.py --expected-content-sha256 file=POST_EDIT_sha) + #206 checkpoint. Tests. STOP on red.
6. REVIEW: codex:adversarial-review. Land codex_findings_*_codex.md.
7. PERSIST: update_status('completed', commit_shas, test_status, actual_delta_s). Write codex_session_summary_*_codex.md. Append codex_persistent_session_state row. Commit canonical.
8. Return to 1.

CAPS: Modal ≤$15/item; HF subscription approved; #313 refuse if predecessor INDEPENDENT/KILL/DEFER within 14d; #325 symposium PROCEED before paid on new substrates; #270 full Tier 1+2+3 mandatory; invalid state transitions REFUSED.

FAILURE → status='blocked' + halt + surface: rate_limited / catalog_<N>_refused / test_red (ROLLBACK) / sister_collision_#314 (yield).

OBSERVABILITY: tools/canonical_task_status.py --list-pending; DuckDB cross-table.

GO. Start at step 1.
```

## END COPY-PASTE BLOCK

— Main-Claude 2026-05-18
