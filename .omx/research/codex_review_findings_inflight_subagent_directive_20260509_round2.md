# Codex adversarial review (round 2) — findings + in-flight subagent directive (2026-05-09)

<!-- generated_at: 2026-05-09T08:00:00Z, from_state_hash: codex_review_b1iwamg1q -->

## Codex thread `b1iwamg1q`, verdict: needs-attention

3 findings against the working tree post-unified-solver-integration landing. ALL THREE are custody/correctness failures that must be fixed before the affected components are used in concurrent dispatch.

### HIGH 1 — GHA dispatcher submission-name matching is prefix-unsafe

**File**: `tools/dispatch_cpu_eval_via_github_actions.py:329-418`

Bug: `run_log_mentions_submission` and `download_artifact` use unbounded substring match on `submission_name`. With existing names like `apogee` and `apogee_stack_b100`, dispatching `apogee` can attach a `[contest-CPU]` score from a concurrent `apogee_stack...` run. **Custody failure** — wrong score gets adjudicated as the wrong archive's authoritative result.

Fix: Match `--submission-dir` / `submission_dir:` with regex boundary OR `Path(value).name == submission_name`. Regression tests for `apogee` vs `apogee_stack_b100`. Fallback artifact selection MUST fail closed on ambiguity.

**Owners**: fix subagent (spawning now). Affects ALL future GHA-CPU dispatches including a3c89347's A1 bias correction sweep variants (12 variants × `apogee_stack...`-style names).

### HIGH 2 — continual_learning.py posterior promotion trusts tags without validating axis/hardware custody

**File**: `src/tac/continual_learning.py:115-116`

Bug: `is_authoritative()` checks only that evidence tag ∈ `AUTHORITATIVE_TAGS`. `posterior_update` only rejects macOS afterward. So `[contest-CPU]` on a non-GHA Linux host, CPU tag with `axis='cuda'`, or CUDA tag on wrong substrate all get accepted. Comment at line 58 says short CPU tag valid only with explicit Linux x86_64 metadata, but that's not enforced.

Fix: Replace tag-only predicate with custody validator that checks tag + axis + hardware_substrate + required metadata together. Require exact GHA Linux x86_64 for `[contest-CPU]`, known CUDA substrates for `[contest-CUDA]`. Reject axis/tag mismatches. Tests for each refused class.

**Owners**: fix subagent (spawning now). Affects the unified-solver continual-learning posterior — every empirical anchor that hits the posterior right now is potentially miscategorized.

### MEDIUM — Posterior writes unsafe under parallel harvesters

**File**: `src/tac/continual_learning.py:239-259`

Bug: `save_posterior` uses single fixed `.tmp` path + `os.replace` without lock. Two parallel harvesters: both load same old posterior → both update different anchors → last replace silently drops the other update. Shared `.tmp` path can also clobber. Given this repo explicitly relies on parallel dispatch and reseeding (cathedral_autopilot, parallel_dispatch_top_k), this WILL hide empirical anchors and corrupt accepted/refused counts.

Fix: Locked transactional update API. `fcntl.flock(LOCK_EX)` on `.omx/state/.continual_learning.lock`. Inside lock: reload state, re-run duplicate checks, apply updates, write to UNIQUE temp file, fsync, replace. Multiprocessing test for concurrent updates of distinct anchors.

**Owners**: fix subagent (spawning now). Affects every continual-learning posterior write.

## Convergence with substrate-vs-codec meta-pattern

All 3 findings are the same META class as the prior codex review (round 1, thread `b6uice9t6`): **"math/wiring that looks correct but a calibration-anchor / custody / concurrency mismatch produces silent failure."**

This is the third independent codex review surfacing this meta-class within 2 days. The pattern is: every new code surface ships with a "looks correct" implementation that fails in production-scale concurrency or custody scenarios.

## Action items

1. **Fix subagent (spawning)**: address all 3 findings with custody validators + fcntl locks + regex-boundary submission matching. 3-clean-pass review. STRICT preflight check planned: `check_authoritative_tag_requires_custody_metadata` + `check_continual_learning_writes_use_lock`.

2. **a3c89347 (A1 bias correction sweep, in flight)**: must NOT use the GHA dispatcher for the 12 variants until HIGH 1 is fixed. If a3c89347 has already started the dispatch, mark all results as `[contest-CPU candidate; pending custody verification]` and re-validate post-fix.

3. **a5551a (unified solver integration, in flight)**: continual_learning posterior-update wire-in must be GATED on HIGH 2 + MEDIUM fix landing.

4. **a473a40 (profiling/xray, in flight)**: their xray tools may consume continual_learning posterior; verify post-fix.

5. **ab30efd (domain catalog, in flight)**: their typed atoms may emit `[contest-CPU]` claims; verify custody validator catches uncalibrated emissions.

## References

- Codex thread ID: 019e0c52-d04f-7fa1-ac08-477703771657
- Background task ID: b1iwamg1q
- Prior round-1 codex review: `.omx/research/codex_adversarial_review_findings_for_inflight_subagents_20260509.md` (3 HIGH all fixed; same meta-class)
- Substrate-vs-codec meta: `~/.claude/projects/.../feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`
