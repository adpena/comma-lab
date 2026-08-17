# Recursive adversarial review — three fresh-eyes charters (PERSISTED, not lost)

**Status:** charters WRITTEN + SPAWNED 2026-08-17; all three arms TERMINATED by an Anthropic
weekly-limit API error (resets Aug 18 12:00 America/Chicago) before producing findings.
Persisted here per the operator's no-signal-loss directive so the work is respawnable verbatim.

## Why fresh eyes (operator, binding)
MAIN reviewing MAIN's own code is the failure mode this extincts: I authored the tool, so my
round-1 self-review found only the defects I was already primed to look for. Two REAL bugs did
come out of that self-pass (below), which is evidence the surface is defect-bearing, not evidence
that self-review suffices.

## Round-1 self-review findings (MEASURED, already fixed — do NOT re-report)
| # | defect | evidence | genus |
|---|---|---|---|
| 1 | reported denominator was `top_k+1` (`checked += 1` ran before the `> top_k` break) | `--top-k 5` printed "top 6"; `--top-k 4` printed "top 5" | the DENOMINATOR genus (#1084) — in the very tool built to fix a denominator problem |
| 2 | `_MEMO_CODE` read `deferral` out of `ddm_deferral_queue_ledger_*`, so a memo merely SAYING "deferral" counted as CITING it | `cites("we discussed the deferral queue", ledger) -> True` | FALSE NEGATIVE — silently suppresses advisories |

Fixes: bound-before-count (`if checked >= top_k: break` precedes the increment); arm-code must
contain a digit. Both verified against the exact failing inputs + both controls re-run.

## ARM 1 — tools/recall_neighborhood_check.py code review
Attack surfaces named: `title_terms` on real memos (no H1 / YAML frontmatter / `#` inside a code
fence, run on 20+ real files) · further `cites()` false pos/neg (substring collisions, case,
`.md` stripping, unicode) · self-exclusion by BASENAME only (duplicate basenames across dirs;
non-path refs; invalid `stores`) · `--terms` CLI (`args.terms.split()`, quoting, empty string,
terms that tokenize away inside `run_query`) · **`run_query` can return `{"truncated": True}` and
the tool IGNORES it — a truncated query silently changing the verdict is SCOPE CENSORING**, a
known repo bug class · VERIFY >=2 of the docstring's MEASURED numbers by executing (the three-row
refuted-ranking table, the char counts, "sf1 at rank 3") · is the tool's own positive control real
or vacuous by construction (live repo lesson: #1086).

## ARM 2 — memo claim audit (ddm_frd077_lever_verdict_and_zero_row_nan_20260817.md)
Four independent verdicts owed (CONFIRMED / REFUTED / UNVERIFIABLE), each with file:line or
command + observed output:
1. **NaN mechanism** — `train_semantic_quantized.py:50-54`; verify 1e-8/7 = 1.428571e-09 < fp16
   subnormal 5.960464e-08 NUMERICALLY; verify `limit` really is 7 at bits=4 (find where it is
   computed); does the claim hold at bits=3 / bits=8?
2. **Receiver distinction** — `ddm_mp2_semantic_receiver.py::_decode_row_prune` quantizes only the
   KEPT rows into `(keep, cols)` and scatters into `torch.zeros`, so a dropped row never reaches a
   quantiser. Confirm or refute the prune-then-quantize vs zero-then-quantize distinction.
3. **Scope** — is INSTANCE right, or did the author OVER-narrow? Does sf1 actually say what the
   memo says (+0.062227 S; SD1M/SM3R undecodable sum; #1058 closed the family at FAMILY scope
   2026-08-16)? Quote sf1's lines.
4. **Residual over-claim sweep** — the A/B numbers (−2,286 vs −2,128; +119 mean; 855 noise band;
   40,252 packed bytes both arms), the "50.48% = predict Undrivable everywhere" claim, and the
   inverted-dropout reading of `_row_dropout` (find it, read it).
Explicitly instructed to hunt BOTH directions: author wrong, AND author needlessly self-critical
(over-correction destroys real signal).

## ARM 3 — assumption challenge + error sweep
**Assumption axis (mandatory per repo protocol):** is a per-memo CITATION check even the right cure
for a recall failure, or symptom-treatment? Is "did the author cite it" the right observable versus
making the corpus ANSWER questions instead of making authors search it? Is an ADVISORY worth
anything given this repo's measured adoption-decay/write-only-API history (#936)? Is the real
defect upstream — memos as the unit of knowledge at all?
**Error sweep:** focused tests covering today's touched code (grep tools/tests + src/tac/tests for
recall_neighborhood / corpus_query / canonical_doc_registry / triality_drift_detector) · import
checks on `git diff --name-only HEAD~3..HEAD` · VERIFY at source that
`triality_drift_detector.is_ledger_or_dag_append` EXCLUDES `.omx/research/*.md` verdict memos from
the RECALL-DEPTH leg (deliberate scoping w/ stated reason, or unnoticed gap?) · any other error.

## Standing rules given to all three
`.venv/bin/python` (bare `python` not on PATH) · READ-ONLY, no edit/commit/git-add, MAIN adjudicates
· every finding = file:line + exact repro command + OBSERVED output · label MEASURED/DERIVED/
INFERRED/ASSUMED · a CLEAN surface is a real result, do not manufacture findings · known-RED
MLX-gated tests are named/xfailed, distinguish from new breakage · cite
`docs/operating_manual_craft_handoff.md`: re-derive from primary artifacts, attack your own
conclusion before reporting.
