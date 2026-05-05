---
name: PATTERNS THAT WORK — codex CLI long-running, 3-session subagent, recursive review per landing
description: 2026-04-29 PM consolidated pattern knowledge. Detach pattern (Pattern A) survives bash-144. 3-session subagent pattern (design → implement → verify) ships 982 LOC + 16/16 tests. Recursive adversarial review per landing is mandatory. Capture forward velocity.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Pattern A — Detached BG bash for codex CLI (PROVEN 2026-04-29)

```bash
mkdir -p /tmp/codex_runs
nohup bash -c '
  codex exec --skip-git-repo-check --sandbox <read-only|workspace-write> \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    -o /tmp/codex_runs/<label>.last.txt \
    "<prompt>" \
    2>&1 | tee /tmp/codex_runs/<label>.log > /dev/null
' < /dev/null > /tmp/codex_runs/<label>.outer.log 2>&1 &
disown
```

**Verified**: 13K-line log produced over 3+ minutes with no bash-144 kill. Codex MCP-plugin (rmcp) auth may fail separately — core codex API still works.

## Pattern B — Agent tool wrapper for multi-stage codex orchestration

When the codex session needs to be orchestrated through multiple stages (read context → reason → write code → verify), use the `Agent` tool with `subagent_type: general-purpose` + `run_in_background: true`. The Agent has its own bash environment AND poll-and-wait logic.

## 3-session subagent pattern (PROVEN — produced Lane Ω-W: 982 LOC + 16/16 tests)

The subagent prompt is structured as 3 codex sessions:
- **SESSION 1 (DESIGN, codex --sandbox read-only, ~3-5min)**: read context files + produce design doc with math, function signatures, integration points, test cases. Commit design doc.
- **SESSION 2 (IMPLEMENT, codex --sandbox workspace-write, ~5-10min)**: implement modules + tests per design doc. Commit code.
- **SESSION 3 (VERIFY, codex --sandbox read-only, ~3-5min)**: run tests, run preflight, mark .py files reviewed via tools/review_tracker.py mark-file --reviewer council and --reviewer codex. Commit verification artifacts.

**Why each session is short enough to complete**: smaller scoped prompts → codex doesn't need to load everything at once → less risk of hang.

**Contingency**: if codex hangs, the subagent should TAKE DIRECT OWNERSHIP per the user mandate: "If codex fails, take direct ownership and produce the analysis yourself" (verified working with the senior engineer review subagent that produced 5 CRITICAL findings without codex).

## Recursive adversarial review per landing — MANDATORY (CLAUDE.md non-negotiable)

**Every landing triggers a recursive adversarial review with rotating perspectives**:
- Subagent commit (writes new code) → spawn codex/Agent review
- Modal lane completes (rc=0 or rc≠0) → spawn codex/Agent review of result + code path
- New module/lane script lands → spawn codex/Agent review of correctness + math + engineering rigor
- Council session output → spawn senior engineer review (catches optimism inflation)

**Review rounds rotate perspectives** (different bug classes each round):
- Round 1 (Selfcomp port): clamp range, non-existent kwarg, dead noise_std (3 CRITICAL)
- Round 2 (Selfcomp polish): bilinear-vs-bicubic, yuv420p, NaN/Inf, LUT divergence, format-compat (1 CRITICAL + 4 Medium)
- Round 3 (Subagent H hardening + Lane MM v2 falsification)
- Round 4 (segmap modules + Subagent H — codex hung, need re-run)
- Round 5+ (per-landing): water-filling allocator (Lane Ω-W just shipped), Lane STC (when shipped), SC++ v4 result (when landed), q_faithful_v3 result (when landed)
- Senior engineer review (independent of round counter): catches CRITICAL flaws in council deliberations themselves (0.24→0.245 + 30-40% prob)

**Promotion gate**: 3 consecutive clean rounds before code is cleared for deployment to submission.

## Subagent shipping pattern observation

When a 3-session subagent finishes:
- It commits the SHARED MODULE files (e.g. src/tac/water_filling_codec.py + tests)
- But may LEAVE UNCOMMITTED the driver + lane script (e.g. experiments/lane_omega_w*.py + scripts/remote_lane_*.sh)
- Parent must check for untracked files and commit them.

## Cross-refs

- CLAUDE.md "Codex CLI invocation" section (Pattern A + Pattern B)
- CLAUDE.md "Recursive adversarial review protocol" section
- feedback_codex_detach_pattern_works_20260429.md (detach pattern proof)
- feedback_persistent_codex_review_protocol_20260429.md (per-landing review)
- project_senior_engineer_review_floor_revised_245_20260429.md (senior eng catches optimism inflation)
