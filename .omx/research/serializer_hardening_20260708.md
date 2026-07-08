# Serializer hardening — post-commit verification (rc=7) + patch-file mode (2026-07-08)

Task #354 (SERIALIZER-HARDENING, READY∧high-EV, fired per anti-deferral after 2
same-day serializer incidents). `[no-triality]` — this is apparatus (the commit
tool every agent uses), not a witness lever/finding, so no DSL/equations leg.

## STORES CONSULTED
- `tools/subagent_commit_serializer.py` (full read — the canonical contract extended).
- CLAUDE.md §"Subagent commits MUST use serializer" + §"Canonical sha discipline"
  (rc=4/5/6 semantics: post-edit `--expected-content-sha256` (#157/#216) guards the
  edit→lock window; `--base-content-sha256` (#6, FIX-ABSORPTION 2026-07-07) guards the
  edit-START surface vs HEAD).
- `.omx/state/commit-serializer.log` (JSONL format — every field the log rows carry).
- `src/tac/tests/test_subagent_commit_serializer_base_sha.py` (throwaway-repo harness
  reused: `_make_throwaway_repo` / `_Patched` / `_run_main` / `_log_outcomes`).
- `src/tac/subagent_contract.py` + `check_subagent_contract_module_integrity`
  (preflight.py) — the contract module + its anti-rot gate (required-constant list
  hardcoded gate-side per anti-self-waive design).

## The two incident anatomies (both 2026-07-08)

### Incident 1 — pre-snapshot clobber (→ POST-COMMIT verification, rc=7)
A sibling's file REVERT landed in the working tree BEFORE a builder computed its
`--expected-content-sha256` snapshot. Every PRE-commit guard reads the WORKING
TREE / INDEX, never HEAD after the ref moves: rc=4 (pre-lock working-tree vs
declared) and rc=5 (staged-blob vs declared) both compared against the
already-clobbered content and PASSED by construction; rc=0 committed the
sibling's copy under the builder's body. Caught only by a human post-commit
`git show`. STRUCTURAL gap: no check reads HEAD after commit.

FIX: `_post_commit_content_check` — after the commit lands (HEAD moved), re-read
each declared file AT HEAD (`git cat-file blob HEAD:<file>`, reused
`_hash_head_blob_files`) and compare to the declared sha. Runs automatically when
`--expected-content-sha256` is passed (opt-in ⇒ backward-compatible). Mismatch →
**rc=7** + LOUD message naming file / declared sha / committed sha / likely cause
(pre-snapshot clobber) + reconcile guidance (`git show HEAD:<file>`, re-apply via
`--patch-file`, or `git revert --no-commit <sha>`). The commit is KEPT, NOT
auto-reverted — the committed content may be the sibling's newer legitimate
landing; surfacing beats destroying. JSONL outcome
`post_commit_content_sha_mismatch` with declared+committed shas.

HONEST LIMIT: rc=7 catches divergence between the caller's DECLARED intent and
what landed at HEAD. If a clobber precedes the snapshot AND the caller re-derives
its expected sha from the clobbered tree (declared == clobbered), no sha check can
catch it — the caller has no record of its intended content. rc=7 is defense-in-
depth for the case where declared == intended; `--patch-file` is the hard fix.

### Incident 2 — whole-file `git add` swept a sibling's hunks (→ `--patch-file`)
A whole-file `git add` staged a DIFFERENT sibling's uncommitted hunks into the
wrong commit body (mis-attribution). Full hunk-attribution is intractable cheaply.
Two mitigations:
- **`--patch-file` intent-manifest mode (the real fix):** caller supplies a patch
  of EXACTLY its hunks; serializer applies it with `git apply --cached` to a temp
  index seeded from HEAD, IGNORING the working tree entirely — no sibling hunk can
  leak in. A patch not cleanly based on HEAD fails LOUDLY at apply time (feature).
  Working-tree sha checks (rc=4/5/6) are skipped as inapplicable in patch mode;
  post-commit rc=7 still runs if `--expected-content-sha256` is passed. `--files`
  is derived from the patch headers when omitted.
- **`--expected-diff-lines <file>=<N>` (warn-only heuristic):** hint your own edit
  line count; a staged diff grossly larger (>2x) than the hint WARNS + logs
  (`hunk_attribution_overshoot_warned`) but NEVER refuses — a whole-file `git add`
  that swept sibling hunks shows up as a gross overshoot.

## Return-code map (unchanged except new rc=7)
0 ok · 2 fatal/malformed/timeout · 3 concurrent-edit (lock-wait) · 4 pre-lock
expected-sha mismatch · 5 staged-sha mismatch / high-risk-missing-sha · 6 base-sha
mismatch (absorption) · **7 POST-COMMIT HEAD mismatch (clobber) — NEW** · 8/9
sister-checkpoint ABORT/WAIT · 10 bare-override · 11 corrupt-checkpoint.
rc=5 was already taken (staged/high-risk); rc=7 is the next unused code.

## Backward compatibility (SACRED — the tool every agent commits through)
- Post-commit check is opt-in (only when `--expected-content-sha256` is declared);
  on the happy path committed==declared ⇒ rc=0, byte-identical outcome to before.
- Verified this repo's pre-commit hook (`tools/preflight_hook.py`) does NOT mutate
  or re-stage staged content → rc=7 cannot false-fire for existing callers.
- New flags (`--patch-file`, `--expected-diff-lines`) default None ⇒ no effect when
  absent. All working-tree checks and skip-conditions gated behind `not patch_mode`.
- Regression proof: 69 existing serializer tests pass unchanged (base-sha,
  concurrent-edit, #157, #216, #174, #117, 4-proc stress) + 18 contract/dispatch
  tests + the strict `check_subagent_contract_module_integrity` gate.

## Contract addendum (item 3)
`tac.subagent_contract.COMMIT_DISCIPLINE` (#405) — composed into
`standard_contract()` so every future build dispatch inherits "post-commit verify
is automatic (rc=7); shared-file edits use --patch-file". Registered in
`CONTRACT_CONSTANT_NAMES` + `KEY_PHRASES` + `__all__` and in preflight's
hardcoded `_SUBAGENT_CONTRACT_REQUIRED_CONSTANTS` (anti-self-waive). Serializer
module docstring gained a FIX-CLOBBER section + the return-code map.

## Tests
`src/tac/tests/test_subagent_commit_serializer_postcommit_clobber.py` — 17 tests:
post-commit rc=7 via a mutating pre-commit hook (faithful clobber: declared==
intended, HEAD lands sibling) incl. commit-kept + log-row + message contents ·
unit `_post_commit_content_check` · rc=7 distinct from rc=4 · patch-file commits
only the patch ignoring a clobbered working tree · derives files from patch ·
patch-not-based-on-HEAD fails loudly · empty patch errors · patch+wrong-declared
→ rc=7 · diff-lines overshoot WARNS-not-refuses · within-bounds no-warn ·
malformed diff-lines → rc=2 · parser units · backward-compat happy path unchanged.
