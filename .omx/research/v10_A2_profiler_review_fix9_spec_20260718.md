# V10 A2 profiler review FIX9 specification — 2026-07-18

`research_only=true`  
`authority_axis=[macOS-CPU advisory]`  
`score_claim=false`  
`promotion_eligible=false`  
`pointer_delta=0`

## Trigger

Exact bundle `c90481d02a27ce571db3b31b6cd702780849bf79691224f342198864cbd5ac0f`
failed operational review with two P1 findings:

1. The profiler resolved the supplied GT-cache and feature-cache paths before
   no-follow validation. A symlink alias therefore became the target path and
   escaped the cache validator's root `lstat` refusal.
2. The profiler admitted output under any existing approved SSD tier. It did
   not require the first existing tier or persist enough ordered-root custody
   to revalidate that decision.

Round 6 is `NOT_CLEAN`; prior clean reviews of the superseded bytes do not
authorize a seal.

## Required implementation

- Resolve profiler input and output paths component-by-component without
  following symlinks. Reject a symlink at the final component or anywhere in
  the supplied path before hashing, opening, or cache validation.
- Require the GT cache to be one local regular file with link count one.
- Require feature-cache and scorer-upstream roots to be local non-symlink
  directories before their existing content/source validation.
- Enumerate approved SSD roots in canonical order using `lstat`; reject an
  approved root that is a symlink or non-directory.
- Outside tiny pytest scope, require output below the first existing approved
  SSD root. A lower tier is a blocker while a higher tier exists.
- Persist `existing_approved_roots` in the storage preflight receipt and
  validate that it is an ordered subset of the canonical waterfall and that
  production output lies below its first entry.

## Required regressions

- A lower-tier profiler output is refused when both approved tiers exist; the
  first tier passes and its ordered-root custody is recorded.
- Reordered or lower-tier persisted preflight custody is refused.
- Symlinked GT file, feature-cache root, scorer-upstream root, and output path
  components are refused before their targets can be substituted.
- A hard-linked GT cache is refused.

## Stop rule

Do not launch, freeze source, or count a clean round until all focused tests,
format/lint/compile checks, and a new three-round exact-byte seal are clean.
No result from this lane is a contest score or promotion authority.
