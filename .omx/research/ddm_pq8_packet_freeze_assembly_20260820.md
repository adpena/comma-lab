# ddm_pq8 — packet freeze-assembly: every accumulated edit applied, short of publish

`date_utc: 2026-08-20` · `owner: ddm_pq8` · `score_claim: false` · `frontier_moved: false`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`,
archive `f3bce5d2…`.** Unmoved by this arm, and it could not have been: this arm wrote no
runtime bytes and fired no evaluation.

**No submission action was taken.** No push to the contest repo, no hosting, no PR, no
publication. Those remain the operator's one-line confirm.

---

## 0. The proof that matters first

`runtime_tree_sha256` = **`2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`,
UNCHANGED**, re-derived independently from the 33 enumerated manifest rows after the
re-stage — not read back from the receipt that produced it. All 33 rows verify
byte-identical on disk, and the archive is still `f3bce5d2…` at 180,625 B. The
`0.14839100138338618` row therefore still applies to the staged tree.

The shipped `MANIFEST.sha256` **verifies 33/33** under `shasum -a 256 -c`, so the identity
claim is executable by a reviewer rather than merely asserted.

## 1. What was applied

| Step | Landed | Receipt |
|---|---|---|
| Re-stage pq4/pq5's three edited docs | yes, through the canonical stager | `STAGING_RECEIPT.json` |
| pq6 §A–§F merged into the README | yes | commit `a36bb35712` |
| pq7 B2 — PR #138 lineage sentence | yes, verbatim from the patch proposal | commit `a36bb35712` |
| pq7 B3 — LICENSE + THIRD_PARTY_NOTICES | yes, staged | staging receipt rows |
| pq7 S3 — `MANIFEST.sha256` | yes, and it verifies 33/33 | commit `76c378526e` |
| `compress.py` + `COMPRESS.md` | yes, staged; subset rule recorded FIRST | `runtime_tree_sha256_scope` |
| rr7 band honesty — all three frames | yes, with the derivation | report + PR body |
| rr7 native-decode disclosure | MEASURED CLOSED, published | report + README |
| rr7 corrector repricing | written to stay true either way | report + PR body |
| nv1 — 4 internal wrong-referent lines | yes, **plus a fifth** | commit `48e95d1140` |
| nv1 — two caveats | yes | report + PR body |
| Version-control disclosure | rewritten: **34/34, MEASURED** | commit `a36bb35712` |
| pq4 §4.1 band question | **WITHDRAWN** | commit `48e95d1140` |
| #135 template shape + LLM-policy notice | yes, as a DRAFT notice | commit `a36bb35712` |

## 2. Three things I measured that changed a published claim

**(a) The version-control gap is CLOSED at 34/34, not "3 files of residue".** The charter
said the oc2 consolidation left a jg5 residue of 3 files. I walked the evaluated candidate
tree and hashed both sides instead of inheriting that: **34 of 34 files are git-tracked,
clean in `git status`, and byte-identical by SHA-256** to the tree the score was measured
on. The README's "24 of 34 have no source in version control" was a real limit when pq4
wrote it and is now false; it is replaced by a positive statement of where the source is,
with the correction disclosed rather than silently swapped.

**(b) A fifth wrong-referent line the charter did not name.**
`ddm_wc2c_native_split_identity_and_speedup_20260820.md:8` also read "1,341.540 s = 95.72%
of it" against inflate elapsed. Same defect class as the four named, fixed the same way,
with both denominators now stated. `ddm_wc2_wall_clock_pass_20260820.md:606` was checked
and is **correct** — its table header names the 1,401.58 s denominator explicitly.

**(c) pq7's `compress.py` byte count is wrong; its identity is right.** The plan records
30,742 B. The file is **40,498 B / 855 lines**. The sha256
`47d0e23dc3b17862c543f5af3823cab71a4859113e60a91123bee742480e28b9` matches pq7's recorded
sha exactly and the line count matches, so this is a transcription error in the plan, not a
changed file. Staged from the committed path as pq7 directed.

## 3. Where I did not follow the plan, and why

**The pq6 MERGE_PLAN's README length budget is arithmetically unreachable against its own
slot table, and I honoured the slot table.**

Measured: `SECTIONS.md` §A–§F is **6,323 words**. The slot table directs §D to be published
"in full" (1,719 words), §B as "items 1–9 **plus** the closed-directions block" (1,542) and
§C with its full derivation list (1,160). Those three alone are **4,421 words** — already
38% above the plan's own 3,200-word ceiling, before §A, §E and §F are added.

I cut whole claims rather than trimming qualifiers, per the plan's own rule: the upstream
scorer facts a maintainer already knows better than we do, the predict-then-diff process
preamble, the internal training guards for training this candidate does not do, the
learned-proposal-ordering and lineage-assertion roadmap items, two granular mixer negatives,
and several statements duplicated across sections. That removed ~640 words and left the
README at **+5,450**. Reaching 3,200 would have required gutting §C's derivation or §D,
which the same plan calls the highest-value section to publish in full.

**This is a deviation, recorded as one.** The constraint that actually binds — the PR body
staying template-terse, which is what the maintainer asked for — was met: §A–§G contribute
**202 words** against a 200-word budget.

## 4. Apparatus landed

The stager could not write packet documents at all; the three public docs had been placed by
hand in earlier generations. That is the "packet assembled by three mechanisms" problem, so
document staging now goes through the canonical tool:

- `--doc SRC=DESTREL`, repeatable, re-hashed after copy and recorded in the receipt.
- A destination not in `DECLARED_NON_RUNTIME` **refuses**, which is what keeps the stager and
  `packet_census_guard.py` from drifting about what "declared" means. A destination colliding
  with a runtime manifest row also refuses.
- Documents are copied **after** the tree identity is proved, so a document can never be
  implicated in a failed identity proof.
- Every doc spec is validated **before** the output directory is created, so a bad spec leaves
  nothing behind.

**`runtime_tree_sha256_scope`** is now written into every staging receipt. It states that the
pin is over the enumerated rows and that re-validation must hash those rows rather than
re-walk the directory — pq7's condition (A), recorded in the receipt instead of in one arm's
memo. The rule was already operative and undocumented before this arm: `report.txt` and
`archive_manifest.json` are manifest-eligible by suffix yet absent from the 33 rows, so a
staged packet **already** differed from a fresh walk. `compress.py` widens that by one file
and changes no score, because `evaluate.py` sizes `archive.zip` only.

**The stager now has tests** — the freeze checklist's owed item (f)(1). 21 new tests cover the
identity proof, every fail-closed path including the leaves-no-directory invariant, the
document surface, spec parsing, and a guard asserting the two tools' declared sets are
identical. One pre-existing census test hardcoded `7 non-runtime`; it now derives the count
from the constant, because a literal there is the same drift class the test exists to catch.
47 tests pass across both suites.

## 5. Owed, and to whom

**Operator only:**

1. **The one-line confirm.** Hosting and PR opening.
2. **The coding-agents/LLM policy.** The PR description must be operator-written; the LLM
   setup should be disclosed; the "most of the code" test needs an honest answer. No arm can
   clear this, and publishing without resolving it risks a ban rather than a rejection. A
   notice at the top of `PR_BODY_DRAFT.md` now says so in the file itself.
3. **The GPU-routing variant.** Variant (b) is staged and carries the valid authority row.
4. **The hosted archive URL** — the PR body still has no URL and must not get a placeholder.
5. **The corpus-hygiene decision** oc2 surfaced: 365 files carrying `/Volumes/` and 399
   carrying `/Users/adpena` are public. The three packet documents are clean — I scanned
   them — but the surrounding research corpus is not.

**Still owed to a future arm:**

- The five-consecutive-clean-pass review counter is **0 of 5**. This arm assembled the
  packet and therefore cannot run it.
- The shipping-axis decomposition that would price a corrector port: the report emits no
  native-versus-Python sub-stage split, so the rr7 row cannot be decomposed. One T4 row with
  that breakdown decides whether ~2,100 lines of corrector C can ever beat 1,341.5 s.
- `GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json` remain declared-but-absent. Census
  reports them as missing and stays CLEAN; neither is load-bearing for the identity proof.

## 6. Custody

Packet: `generations/gen5_jg5_waterfill` — **43 files**, `CENSUS_CLEAN`. Receipts:
`generations/gen5_receipts` — `STAGING_RECEIPT.json`, `CENSUS_gen5.json`, and a new
`PRE_PURGE_CENSUS_gen5.json` recording all 38 pre-purge files with their digests, so the
purge was certified before it happened rather than justified after. 47 AppleDouble sidecars
written by macOS onto the ExFAT volume were purged; the census then ran on the final state
and the trailing sidecar from the receipt write itself was removed afterwards. Zero remain.

**Pointer: UNMOVED**, by construction. `0.14839100138338618` `[contest-CUDA T4, n600]`,
archive `f3bce5d2…`, 180,625 B.
