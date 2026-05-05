---
name: Grand Council adversarial review — IMP permanent-fix design + KILL process protocol + 4 STRICT preflight checks
description: 2026-04-30 ~23:05 UTC. Inner council 10-member deliberation on (DD1) how to wire train_distill into IMP, (DD2) 4 new STRICT preflight checks for the bug classes uncovered, (DD3) KILL memory file process protocol requiring council review, (DD4) re-launch strategy. All design decisions made under "extreme paranoia" mandate from user 2026-04-30 ~22:55 UTC.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## User mandate (2026-04-30 ~22:55 UTC)

> "yes but permanently fix all bugs and bugs classes and metabugs and everything and have all design decisions and ultimate experiment subject to extreme paranoia and adversarial grand council reviews"

This memory captures the inner-council deliberation on the 4 design decisions arising from the IMP cycle 0 measurement-bug incident. The user's challenge correctly caught a premature KILL verdict; the council's job is now to ensure (a) the fix is correct, (b) the underlying bug classes are structurally extinct, (c) the process protocol prevents re-occurrence.

## DD1: HOW to wire train_distill swap into IMP cycle

### Options considered

**Option A** — `train_imp_cycle.py` calls train_distill internally as a function
- Pros: single-script execution, simple
- Cons: refactor train_distill to be importable, changes the lightweight-test contract

**Option B** — Add a Stage 1.X to the dispatch script that runs train_distill on each cycle's renderer.pt AFTER train_imp_cycle.py's prune+rewind+stub finishes
- Pros: clean stage separation, dispatch is the orchestrator (matches CLAUDE.md "canonical pipeline standard")
- Cons: 2 invocations per cycle, more dispatch script complexity

**Option C** — Add `--use-train-distill` flag that delegates via subprocess
- Pros: backward compatible
- Cons: subprocess plumbing complexity

**Option D** — Delete the stub loop entirely; smoke tests use a separate file
- Pros: removes the "stub pretending to be real" footgun permanently
- Cons: refactor work, more files

### Council vote (10 inner members)

| Member | Vote | Rationale |
|---|---|---|
| Shannon (LEAD) | B | Cleanest R(D) stage separation; deterministic per-stage boundaries |
| Dykstra (CO-LEAD) | B | Convex feasibility is per-stage; B preserves that |
| Yousfi | B | Audit trail benefits from explicit stages |
| Fridrich | D | Most paranoid — eliminates the bug class entirely |
| Contrarian | D | Stub WILL be re-introduced as footgun if we leave it |
| Quantizr | B | Operationally simpler |
| Hotz | B + assertion | B with STRICT preflight that train_distill IS called per cycle |
| Selfcomp | B + assertion | Same as Hotz |
| MacKay | D | MDL-cleanest; no contradictory comments |
| Ballé | B + assertion | Same as Hotz |

**VERDICT: 6 for B+assertion / 3 for D / 1 for B.** Winner: **Option B + STRICT preflight assertion (PCC1) that the train_distill stage exists in the dispatch script**.

Implementation: dispatch script gets a Stage 1b after each cycle's train_imp_cycle invocation that runs `train_distill.py --resume <cycle_dir>/renderer.pt --epochs $REAL_EPOCHS_PER_CYCLE --auth-eval-on-best`.

## DD2: 4 NEW STRICT preflight checks

### PCC1: check_imp_dispatch_calls_train_distill (STRICT)
**Bug class**: stub-pretending-to-be-real (the cycle 0 = 1.98 metabug)
**Detection**: scan scripts/remote_lane_j_imp_*.sh — if `train_imp_cycle.py` is invoked, `train_distill.py` MUST also be invoked subsequently in the same script.
**Reason**: train_imp_cycle.py contains a documented stub loop (`_finetune` at line 378+) that the comment promises gets "swapped for train_distill in the deploy script". The promise was a comment, not a contract. STRICT enforcement makes it a contract.
**Failure mode caught**: cycle 0 = 1.98 [contest-CUDA] regression that was actually a 3.5-second stub loop, not real fine-tune.

### PCC2: check_no_comment_only_contracts (STRICT)
**Bug class**: comment-only contracts (the meta-bug)
**Detection**: AST/regex scan all .py files for comment patterns matching:
- `"deploy script swaps in <X>"`
- `"wrapper handles <Y>"`
- `"caller is responsible for <Z>"`
- `"the wrapper script does <W>"`
For each match, require an inline assertion or a sibling preflight check that verifies the wrapper actually does what the comment promises. Without backing assertion, FAIL.
**Reason**: comments rot; assertions don't. The IMP stub had a clear "deploy script swaps in train_distill" comment that was load-bearing for correctness but enforced by nothing.
**Failure mode caught**: this bug class generally — any time a placeholder relies on caller responsibility without an assertion.

### PCC3: check_stats_json_internal_consistency (STRICT)
**Bug class**: internal-inconsistency in reporting
**Detection**: scan scripts that write stats.json-style files for internal-consistency assertions. Specifically, if the stats includes both `epochs` and `elapsed_sec` (or `steps` and `wall_time`), the producer code MUST contain `assert elapsed_sec >= MIN_SECONDS_PER_EPOCH * epochs` (or similar) before writing the JSON.
**Reason**: stats.json said `epochs: 200, elapsed_sec: 3.47` — internally inconsistent (200 epochs in 3.5s impossible). A 1-line assertion would have caught this.
**Failure mode caught**: any stub-loop or measurement-bug that produces a stats file claiming work that didn't happen.

### PCC4: check_kill_memory_files_have_council_review (STRICT)
**Bug class**: premature KILL verdict process bug
**Detection**: scan `~/.claude/projects/<repo>/memory/project_lane_*_killed_*.md` and any memory file containing `"VERDICT: KILL"` or `"FALSIFIED"` for the literal string `"Grand Council"` or `"Council vote:"` followed by at least 5 named council member positions. Without it, FAIL.
**Reason**: the user's challenge caught my premature KILL verdict on Lane 17. Without the user explicitly demanding adversarial review, I would have buried Lane 17 in the registry as KILLED based on a measurement bug. Memory files with KILL/FALSIFIED claims must have evidence of council scrutiny.
**Failure mode caught**: the metabug of premature kills polluting the project's strategic record.

## DD3: KILL memory file process protocol

### Add to CLAUDE.md as non-negotiable

```markdown
## KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS

Any memory file claiming a lane is KILLED, FALSIFIED, DEAD, or RETIRED MUST contain:
1. A "Grand Council adversarial review" section with at least 5 named council member
   positions (from the inner 10: Shannon/Dykstra/Yousfi/Fridrich/Contrarian/Quantizr/
   Hotz/Selfcomp/MacKay/Ballé).
2. An explicit "internal-consistency check" subsection listing what the verifier
   checked (e.g., "elapsed_sec >= epochs * MIN_SEC", "EMA shadow used at eval",
   "auth-eval archive matches submission archive bytes").
3. A "what would change my mind" subsection listing the conditions under which
   the KILL would be reactivated. (e.g., "if cycle 0 with proper train_distill
   fine-tune scores < 1.10, KILL retracted").

Preflight check PCC4 enforces this STRICT.

Memory linter rejects any project_lane_*_killed_*.md OR any file containing
"VERDICT: KILL" / "FALSIFIED" without all three sections. Override requires
user explicit "council review skipped" annotation in the file body.
```

### Add to CLAUDE.md "Council conduct"

```markdown
## Adversarial council review of design decisions — NON-NEGOTIABLE

Per user mandate 2026-04-30 ~22:55 UTC: "all design decisions and ultimate
experiment subject to extreme paranoia and adversarial grand council reviews".

A DESIGN DECISION is any choice between options where the wrong choice
costs > $1 of GPU time, > 1 hour of wall clock, OR has 2+ alternatives that
the council members have non-trivial preferences over.

For every design decision:
1. Enumerate the options with pros/cons
2. Get explicit positions from at least 5 of the 10 inner council members
3. Tally the vote with a clear verdict
4. Capture the deliberation in a memory file

This memory file is the canonical example.

The council's job is NOT to reach consensus — it's to surface disagreement.
A unanimous vote on a non-trivial decision is a signal that the council
isn't thinking adversarially enough.
```

## DD4: Re-launch strategy for Lane 17 (post-fix)

### Prerequisites

1. Council review (this file) committed to memory ✓ (this turn)
2. PCC1+PCC2+PCC3+PCC4 STRICT preflight checks landed (this turn)
3. Dispatch script gets the train_distill swap (this turn)
4. CLAUDE.md updated with KILL-process + council-review-of-design non-negotiables (this turn)
5. **User action**: switch Lightning Studio to L40S/H100 GPU mode in the UI (Lightning currently has no GPU attached; CPU-only)

### Re-launch parameters (council-vetted)

- **Cycle 0 fine-tune**: real train_distill, NOT the stub. ~10-30 min on L40S, ~$0.10-0.30
- **Verdict gate**: cycle 0 score must be ≤ 1.10 × anchor (1.155 absolute) to continue
- **If gate fails**: KILL with proper evidence (council review, internal-consistency check, what-would-change-my-mind subsection)
- **If gate passes**: continue cycles 1-9 with proper train_distill per cycle (~10h total, ~$6 on L40S)
- **Per-cycle revert-on-regression**: still active per Council Q4 9/10
- **Total Lane 17 budget cap**: $25 (unchanged from 2026-04-30 user approval)

### Cost projection

- Cycle 0 only (verdict gate): $0.30 (well within budget)
- Full 10-cycle if gate passes: $6 (well within budget)
- Total worst case: $6.30 (still 25% of $25 cap)

## What "permanent fix" means

The 3 IMP-specific bugs from `feedback_imp_dispatch_shape_mismatch_fix_20260430.md` are already fixed in commit 9fdabc9e:
1. ✅ Shape mismatch (build_renderer → load_asymmetric_checkpoint)
2. ✅ Export API drift (f.write(export_*) → export_*(model, path))
3. ✅ Score parser NaN (regex → brace-balance + correct field name)

The 4 NEW bug classes uncovered THIS turn require structural fixes:
4. ⏳ Stub-loop pretending to be real training (PCC1 + dispatch script train_distill swap)
5. ⏳ Comment-only contracts (PCC2 — generic across all .py files)
6. ⏳ Internal-inconsistency reporting (PCC3 — assertion enforcement)
7. ⏳ Premature KILL verdict process bug (PCC4 + CLAUDE.md non-negotiable)

## Cross-refs

- project_lane_17_imp_killed_cycle_0_198_regression_20260430.md (the KILL retraction)
- feedback_imp_dispatch_shape_mismatch_fix_20260430.md (3 fixed bugs from previous incident)
- feedback_imp_local_backport_landed_20260430.md (commit 9fdabc9e — local backport)
- CLAUDE.md "Council conduct — non-negotiable" (existing council protocol; this extends it)
- CLAUDE.md "Recursive adversarial review protocol — non-negotiable" (existing 3-clean-pass gate; this adds the per-design-decision council vote)
