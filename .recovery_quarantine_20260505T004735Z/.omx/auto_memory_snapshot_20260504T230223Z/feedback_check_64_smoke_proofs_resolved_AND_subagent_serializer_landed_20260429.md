---
name: Check 64 smoke proofs landed for NWC + PSD-standard + subagent commit serializer landed
description: 2026-04-29 PM. Two bug classes hardened. (A) Pre-existing Check 64 violations on scripts/remote_lane_nwc.sh + scripts/remote_lane_psd_standard.sh (which had forced 3 separate commits today to use PREFLIGHT_HOOK_ENABLED=0) resolved via Option A1 — ran experiments/canonical_local_auth_eval_smoke.py for both lanes, both passed all 10 stages in <100ms each. (B) Subagent-commit-message-swap bug class hardened structurally via tools/subagent_commit_serializer.py — fcntl.flock(LOCK_EX) on .omx/state/.commit-lock + JSONL log at .omx/state/commit-serializer.log + CLAUDE.md non-negotiable section.
type: feedback
originSessionId: bug-classes-hardening-262
---

## What was fixed

### Bug Class 1 — Check 64 smoke proofs (Option A1 path)

Two lane scripts had been violating Check 64 for days, forcing every commit that touched them to bypass the preflight hook with `PREFLIGHT_HOOK_ENABLED=0`:

- `scripts/remote_lane_nwc.sh`
- `scripts/remote_lane_psd_standard.sh`

Today's commits that bypassed: `16ae6405` (Council C), `d8a1abe7` (Council D), `1bd8882b` (parent). The CLAUDE.md "Review gate — non-negotiable" section explicitly forbids `REVIEW_GATE_OVERRIDE` on .py files, and the same anti-pattern via `PREFLIGHT_HOOK_ENABLED=0` was happening daily.

**Option A1 fix** (preferred — actual proofs, not exempt waivers):

```bash
.venv/bin/python experiments/canonical_local_auth_eval_smoke.py --lane remote_lane_nwc
.venv/bin/python experiments/canonical_local_auth_eval_smoke.py --lane remote_lane_psd_standard
```

Both lanes passed ALL 10 stages (extract / whitelist / renderer_magic / masks_present / config_env / inflate_dispatch_path / inflate_renderer_imports / upstream_evaluate_arity / gt_video_present / launcher_includes_env) in 0.02s each against the canonical Lane G v3 fixture archive (`experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`, sha256 `9b20bdfca246…`).

The smoke proofs landed in `.omx/state/lane_e2e_smoke_proofs.json` via commit `a226f227`. Check 64 now reports `74 proven, 16 waived, 0 violations` across 90 remote_lane_*.sh scripts.

**The exempt mechanism (Option A2 fallback) was NOT needed** — the smoke script is a static-analysis tool that runs in <100ms with no GPU; both lanes had no actual blocker beyond never running the smoke. Future operators should always try A1 first.

### Bug Class 2 — concurrent subagent commit-message swap

Memory `feedback_concurrent_subagent_commit_message_swap_20260429.md` documents the canonical fix: serialize concurrent commits via a file lock. Implemented as `tools/subagent_commit_serializer.py`:

- Acquires `fcntl.flock(LOCK_EX)` on `.omx/state/.commit-lock` (blocking with 120s default timeout, configurable via `--timeout-seconds`).
- Inside the lock: `git add -- <files>` (NEVER `-A` / `.` per CLAUDE.md), then `git commit -m <msg>`.
- Pre-commit hook (preflight + ruff F821 + review gate) runs INSIDE the lock — failures release the lock and exit non-zero so the next waiter can proceed.
- Every attempt is appended JSONL to `.omx/state/commit-serializer.log` for forensics (label, pid, host, files, message head, wait_seconds, commit_seconds, head_after, stderr_tail).
- `.gitignore` updated to exclude both `.commit-lock` and `commit-serializer.log` (local forensics only).

Usage from a subagent:

```bash
SUBAGENT_LABEL="my-lane" python tools/subagent_commit_serializer.py \
    --message "$(cat <<'EOF'
<commit message>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" \
    --files src/tac/foo.py src/tac/tests/test_foo.py
```

CLAUDE.md gained a new non-negotiable section "Subagent commits MUST use serializer — NON-NEGOTIABLE" between "Git discipline" and "Review gate", documenting the rule + the exact wrapper invocation pattern + cross-ref to this memory file.

### The bug class still bit ME during this very landing (forensic note)

While creating commit `b7ee5656` for the serializer + CLAUDE.md, parallel subagent #260 (Lane 19, commit `142b5777`) committed in the gap between my preceding smoke-proofs commit and my retry of the failed serializer commit. Their commit STAGED my serializer (305 LOC) and CLAUDE.md (22 lines) addition under THEIR "Lane 19 (SegNet logit-margin)" message. My follow-up commit `b7ee5656` only carries the 1-line `io.IOBase` fix (which I had to apply because ruff F821 blocked the first attempt).

This is exactly the bug pattern: code lands intact, attribution is shuffled. The good news: HEAD is now correct (the wrapper IS in `tools/subagent_commit_serializer.py`, the rule IS in `CLAUDE.md`); operators reading `git log --stat` will need to know that commit `142b5777` actually contains both Lane 19 + serializer + CLAUDE.md. This memory file is the durable forensic record.

**This is the LAST time the body-shuffle should ever happen** — from this commit forward, the serializer is the only sanctioned commit path for subagents.

## What workaround pattern is now extinct

`PREFLIGHT_HOOK_ENABLED=0 git commit ...` should not appear in any future commit log entry. If preflight blocks, the right answer is to fix the violation (e.g., run `experiments/canonical_local_auth_eval_smoke.py --lane <name>` for Check 64), not bypass.

`REVIEW_GATE_OVERRIDE=1` on .py files was already forbidden per CLAUDE.md.

## Cross-refs

- Wrapper: `tools/subagent_commit_serializer.py`
- Lock file: `.omx/state/.commit-lock` (fcntl-advisory; gitignored)
- Forensics log: `.omx/state/commit-serializer.log` (JSONL append; gitignored)
- CLAUDE.md "Subagent commits MUST use serializer — NON-NEGOTIABLE"
- Memory: `feedback_concurrent_subagent_commit_message_swap_20260429.md` (the design doc)
- Smoke tool: `experiments/canonical_local_auth_eval_smoke.py`
- Smoke proof JSON: `.omx/state/lane_e2e_smoke_proofs.json`
- Commits: `a226f227` (smoke proofs), `142b5777` (serializer + CLAUDE.md, attributed to Lane 19 due to the very bug being fixed), `b7ee5656` (1-line io.IOBase fix to serializer)

## When to use this knowledge

- Any operator about to use `PREFLIGHT_HOOK_ENABLED=0`: instead, run `experiments/canonical_local_auth_eval_smoke.py --lane <name>` for the violating lane(s).
- Any subagent about to call `git commit` directly: use `tools/subagent_commit_serializer.py` instead.
- Any parent agent dispatching ≥2 subagents that will land code: include the serializer invocation in the subagent prompt template.
- For forensics on commit-body-vs-content mismatch: `git show <sha> --stat` + `tail -20 .omx/state/commit-serializer.log` (the log shows what each subagent CLAIMED to commit vs what landed).
