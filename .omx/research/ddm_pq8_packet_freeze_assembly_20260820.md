# ddm_pq8 — packet freeze-assembly: every accumulated edit applied, short of publish

`date_utc: 2026-08-20` · `owner: ddm_pq8` · `score_claim: false` · `frontier_moved: false`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`,
archive `f3bce5d2…`.** Unmoved by this arm, and it could not have been: this arm wrote no
runtime bytes and fired no evaluation.

**No submission action was taken.** No push to the contest repo, no hosting, no PR, no
publication. Those remain the operator's one-line confirm.

---

## 0. The proof that matters first

> ⚠ **CORRECTED after review round 1 (rv15 F2).** The claim below originally leaned on the
> staging tool's own re-derivation. **That re-derivation was a tautology:**
> `rederive_tree_sha256` consumed the manifest's OWN rows, so it re-hashed its own input and
> could not fail on content drift. The real check at the time was the per-file re-hash beside
> it. **Fixed in the mechanism, not the wording** (`d678b60c24`): the derivation now consumes
> the FRESHLY MEASURED sha/bytes of every staged copy, the per-file diffs are demoted to the
> diagnostic that explains a failure, and three tests pin it — including one that mutates staged
> content and asserts the tree check itself catches it, and one that asserts substituting a
> measured digest moves the derived hash. My own verification below was always honest, because
> it hashed the bytes on disk rather than trusting the tool; that is why its value stands.

`runtime_tree_sha256` = **`2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`,
UNCHANGED**, re-derived from the MEASURED bytes of the 33 enumerated rows after the
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

---

# ROUND-1 REVIEW (rv15) FIX BATCH — 2026-08-20

## F1 (HIGH) — "15.3% slower" carried the wrong denominator on all three surfaces

**Confirmed and fixed.** 15.3% is the **token-stage** ratio, `1546.617/1341.540 = 1.15287`. It
was paired with the **inflate** pair `1612.579/1419.904`, which is **13.57%**. Two different
denominators, one number.

Fixed by stating **both** ratios with their denominators inline on every surface, rather than
picking one — the port is slower on both, and the smaller inflate figure has a mechanism worth
saying out loud: the rest of inflate is unchanged and dilutes the stage ratio.

| Surface | Before | After |
|---|---|---|
| `report.txt` | "1612.6 s against 1419.9 s … 15.3% SLOWER" | both pairs, both ratios, with the dilution reason |
| `README.md:36` | "15.3% slower (1612.6 s against 1419.9 s)" | "15.3% slower on the token stage it replaces (1546.6/1341.5) and 13.6% slower on whole inflate (1612.6/1419.9)" |
| `PR_BODY:245` | "15.3% SLOWER: 1,612.6 s against 1,419.9 s" | both, with denominators |
| `PR_BODY:377` | "15.3% slower on a contest T4" (no numbers) | "…on the token stage it replaces (1,546.6 s against 1,341.5 s)" |
| `PR_BODY:163-164` | inside the mirrored block | fixed by regenerating the mirror from `report.txt` |

Residual check: every remaining `15.3%` occurrence was scanned; **0** pair it with the inflate
times alone.

## Denominator sweep — round-2 demand #4

Two wrong-referent defects in one wave, so this is a live class, not two incidents. Swept
**78 distinct percentage/ratio/× values** across the three public surfaces. Every value whose
numerator and denominator are both quoted was recomputed:

| Value | Denominator | Reproduces? |
|---|---|---|
| 94.5% | 1341.5 / 1419.9 inflate | 94.478 ✓ |
| 95.72% | 1341.540 / 1401.58 stage sum | 95.716 ✓ |
| **15.3%** | **1546.617 / 1341.540 token stage** | 15.287 ✓ (was mis-paired) |
| **13.6%** | **1612.579 / 1419.904 inflate** | 13.570 ✓ (new, named) |
| 81.05% | rate 0.12027077 / S 0.14839100 | 81.050 ✓ |
| 37.7% / 62.2% | 66,528 and 109,696 / 176,420 B | 37.710 / 62.179 ✓ |
| 13.4× | 0.172 pose / 0.012847 seg | 13.388 ✓ |
| 443× | (0.15 − S) / 3.633e-06 | 442.884 ✓ |
| 1249.86× | 8.7110e-03 / 6.969573e-06 | 1249.861 ✓ |
| 8.0× | 3.268e-3 / 4.089e-4 | 7.992 ✓ |
| 467× / 195× | 2.5345e-03/5.4316e-06 · 2.1402e-03/1.0997e-05 | 466.6 / 194.6 ✓ |
| 12.8× | (0.0054967−0.0051147) / 2.99e-5 | 12.776 ✓ |
| 266× | 50.48% / 0.190% | 265.684 ✓ |
| 455× | 5.127114e-5 / 1.126177e-7 | 455.267 ✓ |
| 4.1% / 95.9% | complementary | ✓ |
| 4× / 2× | 1/0.23 and 1.87/1.0 | 4.348 / 1.870 ✓ |

**Four failed the sweep and were fixed:**

1. **"about 2.5× under its own estimate"** — the band is 120–180 s against 51.4 s measured, i.e.
   **2.3× to 3.5×**. The single figure sat inside the band but named no denominator. Now states
   "51.4 s measured against a 120–180 s allowance, i.e. 2.3x to 3.5x under."
2. **"roughly 1000× to matter"** — the reciprocal of the quoted 0.06% headroom is ~1,667×, so
   the figure did not reproduce from its own sentence. Restated as "clear that 0.06% headroom by
   roughly three orders of magnitude", which does.
3. **"2.8× the gap"** — denominator unnamed. Now "2.8× the gap it was measured against".
4. **"58× the incumbent"** — denominator unnamed. Now "58× the incumbent carrier's `d_pose`".

Also tightened: **"1.8× faster locally"** → the measured range **1.77–1.83×**, so the figure
carries its own uncertainty rather than a round number.

Values whose denominator is named in words and not derivable from quoted numbers (measured
ratios, acceptable): 2.12× the debt paid off · 203,000× the interior flip rate · 1.662×
attenuation · 38,700×/2,518× weight-to-render amplification · 90× underwater.

## F2 (MED) — the named identity proof was a tautology

**Confirmed, and fixed in the mechanism.** `rederive_tree_sha256` consumed the manifest's own
rows (`staged_rows.append(row)`), so it re-hashed the tool's input: it could not fail on content
drift, while being the thing the receipt and my ledger both called the proof.

Fix (`d678b60c24`): each staged copy is measured after the copy and a **measured row** carrying
those digests is what the derivation consumes. The tree comparison is now genuinely load-bearing
and the per-file diffs are the **diagnostic** attached to its failure message. Receipt field
renamed `runtime_tree_sha256_rederived_from_measured_staged_bytes`, with a sibling
`runtime_tree_sha256_rederivation_input` naming the input and the tautology it avoids. Module
docstring point 3 rewritten. Three tests pin it: content drift caught *by the tree check* with
the file named, digest/byte substitution moving the derived hash, and the receipt naming its
input. **49 tests pass**, ruff clean.

## F3 (LOW-MED) — the 202-word claim was self-graded

The template-shaped body has no §A–§G markers, so the figure was not independently checkable.
**Attribution published** — the claim covers exactly these three inserted passages:

| pq6 section | Lands in | Opens with | Words |
|---|---|---|---|
| §C + §E + §F | `# changes from upstream` | "What changed is on our side. Every mechanism that moved this score…" | 82 |
| §A + §D | `# competitive or innovative?` | "Where the headroom is, measured: rate is **81.05%** of this score…" | 76 |
| §B | `# additional comments` | "**Priced and unbuilt.** A third admission branch that drops the token outright…" | 44 |
| | | **total** | **202** |

Anyone can recompute it from those three anchors. It remains **2 words over** the plan's 200,
which I am reporting rather than shaving a qualifier to hide.

## F4 (LOW) — rr6 commit headline overstates its diff

**Confirmed.** `a886ddb340` says "default it on"; the shipped tree defaults to `python`
(`inflate.sh:36`) and hard-refuses `native-hpac` (`f26_inflate.py:435-441`) — which is correct,
and which rr7 later vindicated by measuring the port slower on the contest axis. The pushed
message cannot be amended, so an **append-only headline correction** was added to
`ddm_rr6_native_token_decode_ship_20260820.md`; nothing in that memo's body is retracted.

## Re-stage receipt

Purge → stage → sidecar purge → census → receipt, in that order, no writes between census and
receipt.

| | before | after |
|---|---|---|
| packet `README.md` | `1c0ed5b46c5bb7a7…` | `d12ca41cd17b422b…` |
| packet `report.txt` | `99eadeb36e1af8b8…` | `cbc73b68c6cca447…` |
| packet `archive.zip` | `f3bce5d259a08183…` | `f3bce5d259a08183…` **unchanged** |
| `runtime_tree_sha256` | `2103073d739fc3f2…` | `2103073d739fc3f2…` **unchanged** |

Verified independently of the tool, by hashing the bytes on disk and re-deriving from the
measured digests: **33/33 rows byte-identical**, tree sha `2103073d…` reproduced, archive
`f3bce5d2…` at 180,625 B. Census `CENSUS_CLEAN` / `PREP_CLEAN` / `RECEIPTS_CLEAN`, rc=0, 43
files, 0 sidecars. Mirror re-verified by construction: PR block == prep report == packet
`report.txt`. **No runtime row and no archive byte was touched.**

## Two things found while fixing, worth their own rows

1. **A SIGKILL leaves a half-staged packet.** The tool's fail-closed cleanup runs on
   `StagingError`, not on signals; a foreground timeout killed a stage mid-copy and left 30 of 43
   files on disk. The census would have caught it, but the tool's own guarantee has a hole. Not
   fixed here — it needs a signal handler or a stage-to-temp-then-rename, which is a design
   change past this fix batch's scope. **Recorded as owed.**
2. **The SSD tier is at 96% capacity** (77 GiB free), and staging 43 small files took ~9 minutes
   against seconds earlier in the same session, with `dasd` pinned at 95%. The certify-and-move
   reclaim is already owed elsewhere; this is a second, independent reason it is now urgent.
