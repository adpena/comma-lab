# ddm_fm3 — fmtools as the advisory second lane: two lanes that pay, one that does not

Tokens: `[no-triality] [p0-ledger-ok]` · Arm: ddm_fm3 · 2026-09-04 · Cost: **$0** (on-device, no network)
Charter: `.omx/research/charters/ddm_fm3_fmtools_advisory_lanes_and_polish_20260904.md` (commit `2e6f2e7ec`)

**Exact pointer: UNMOVED.** `afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]`. This arm is
apparatus; it cannot move the score and did not try to. Read every number below as instrument quality,
not as progress toward sub-0.12.

---

## The headline

Three regex censuses got an advisory on-device second lane. **Two of the three pay; the third is a
measured negative and is reported as one.**

| lane | what it does | verdict |
|---|---|---|
| **A1** Catalog #344 memo classification | second opinion on "does this memo state a measured finding?" | **PAYS.** F1 **0.769** vs the shipped gate's **0.303**. Recovers 11 of the 15 true positives the gate now misses. **Landed** as an advisory column in the commit hook. |
| **A2** GT `.npz` lineage census | classify 318 newly-lit consumers so the widening can be scoped | **PAYS.** Decided 312 of 318 classes (the deterministic force-list reached only 6). **Widening landed REPORT-ONLY.** |
| **A3** constant provenance | label a bare float's provenance from its source window | **NEGATIVE.** 9 of 25 retired-literal sites labelled correctly (36%); the `unknown` class — the one that mattered — was never emitted once. Scoped by a binary control, not killed. |

And the tool the three lanes share went from a hand-rolled subprocess incantation to a supported CLI:
**fmtools 0.0.218 → 0.0.219**.

---

## A1 — Catalog #344: MEASURED agreement on 29 memos

**Corpus.** The 29 memos Catalog #344 reported live at `d3212bed1`, read **at that commit** — before
ddm_eq1 appended its equations-leg addenda. Reading them as they stand today would leak the answer key
into both lanes: the addendum literally prints `MEASURED` or the waiver.

**Ground truth.** ddm_eq1's own adjudication (`ddm_eq1_equations_leg_backfill_20260904.md` §2):
**20 MEASURED** (states a measured empirical finding about the object of study) / **9 WAIVER**
(review, verdict-on-process, hygiene, consumption sweep, apparatus debt).

**Independent reproduction of eq1's counts** (this arm, same commit, same predicates): plain-substring
tokens flag **29/29**; the shipped `(?<!st)ratified` form flags **13/29**. eq1's 29→13 reproduces exactly.

### The table

| lane | precision | recall | F1 | accuracy | tp/fp/fn/tn |
|---|---:|---:|---:|---:|---|
| regex before the fix — **FULL text** | 0.690 | 1.000\* | 0.816 | 0.690 | 20/9/0/0 |
| **regex as it ships today — FULL text** | **0.385** | **0.250** | **0.303** | 0.207 | 5/8/15/1 |
| **fmtools majority-of-3** | **0.789** | **0.750** | **0.769** | 0.690 | 15/4/5/5 |
| union (shipped regex OR fm) | 0.640 | 0.800 | 0.711 | 0.552 | 16/9/4/0 |
| intersect (pre-fix regex AND fm) | 0.789 | 0.750 | 0.769 | 0.690 | 15/4/5/5 |
| regex before the fix — excerpt control | 0.562 | 0.450 | 0.500 | 0.379 | 9/7/11/2 |
| regex as it ships — excerpt control | 0.125 | 0.050 | 0.071 | 0.103 | 1/7/19/2 |
| fmtools run 1 / 2 / 3 | 0.750 / 0.867 / 0.842 | 0.750 / 0.650 / 0.800 | 0.750 / 0.743 / 0.821 | — | — |

Two framings, both reported on purpose. **FULL** is production-faithful — the gate reads the whole memo,
so those rows describe the live gate. **excerpt** is the matched-input control: the same first 4,000
characters the advisory lane gets, so neither lane is credited or penalised for how much text it saw.

### What the numbers say

**The fix that stopped the false positives also removed almost every true positive.** eq1's
`(?<!st)ratified` cure was correct for the problem it was aimed at — a 55.2% false-positive rate was
refusing honest commits. But measured against "does this memo state a measured finding", it took the
flagged set from 29 to 13, and **15 of the 16 memos it removed were true positives.** The shipped gate's
recall on this corpus is **0.250**: of the 20 memos that genuinely owe an equations leg, the token set
now reaches 5. That is a finding about gate efficacy, not a criticism of eq1 — the cure fixed the
measured disease and exposed a second one underneath it.

**The lanes are complementary, not competing.** Of the 15 true positives the shipped regex now misses,
the FM majority **recovers 11**. Of the 8 false positives it still carries, the FM majority **correctly
rejects 5**. That is why what landed is a *column*, not a replacement.

### Caveats that travel with these numbers

1. **Selection bias — the recall figures are relative, not absolute.** The corpus IS what the pre-fix
   regex flagged. So `regex_before_fix.recall = 1.000` is *tautological*, and every other recall here is
   measured *within the pre-fix flagged set*. Absolute recall over all `.omx/research` memos is
   **unmeasured**. Same genus as [[m88]] (a prefix of a skewed population is a different population).
2. **The advisory lane is non-deterministic.** 7 of 29 memos (24.1%) flipped label across 3 repeats.
   Majority-of-3 is the mitigation, and the per-run rows are published so the spread is visible.
3. **The model refused one memo outright.** `ddm_nx1_next_object_route_20260831` returned
   `Unsupported language or locale` on all 3 runs. The fail-open contract turned that into *no advice*
   rather than a fabricated label — which is the contract working, and a 3.4% refusal rate to budget for.
4. **n = 29.** One corpus, one campaign, one instruction. This is an instrument calibration, not a law.

### What landed

`tools/preflight_hook.py::_canonical_equation_fm_advisory` — an advisory column on the #344 hook step.
It runs **only** on staged in-scope memos the deterministic lane did *not* flag (the silent-drift set),
caps at 6 memos, times out at 30 s, fails open on every error, **never blocks**, and writes each
disagreement to `.omx/state/fmtools_advisory_disagreements.jsonl`. Opt out with
`CANONICAL_EQUATION_FM_ADVISORY=0`.

**MEASURED latency, since this sits on a commit-time path: 2.48 s for one staged memo; 0.0000 s when
opted out.** That is a real cost on every memo-staging commit and it is stated rather than hidden.

---

## A2 — the GT `.npz` lineage-consumer census and the widening

**The blind spot.** `_GT_LINEAGE_ARTIFACT_PATTERNS` matched only `gt_first6*.npy` and `gt_cache_*.pt`.
`gt_n600.npz` — the table the born trainer PINS as authority, and the **PyAV** decode lineage (20,671
argmax sites off DALI; same-object pose ceiling 1.69e-5) — is `.npz`, and was therefore invisible to the
one gate built to catch undeclared solve objectives (ddm_bh1 finding 2).

**MEASURED live counts** (same scanner, same exclusions, only the vocabulary widened):

| vocabulary | findings | files |
|---|---:|---:|
| primary (`.npy` + `.pt`) | **2** | 2 |
| + `.npz` widening | **378** | 318 |
| the widening's own contribution (primary subtracted) | **376** | 317 |

### The class-aware widening plan

**318 newly-lit files classified.** The deterministic force-list (path names a trainer / eval /
byte-close) reached only **6**; the advisory lane decided the other **312**, so the lane is load-bearing
here rather than decorative. It spared **30 files / 32 findings** from the refusing class.

| class | meaning | files | findings | acceptance the table justifies |
|---|---|---:|---:|---|
| `authority_consumer` | fits / trains / scores / byte-closes against the table — the table IS its solve objective | 269 | **326** | REFUSE |
| `continuity_frame` | reads it to stay comparable with an earlier measurement | 15 | 17 | lineage LABEL |
| `advisory_instrument` | probe, census, plot, diagnostic; output is never a score | 4 | 4 | lineage LABEL |
| `test_fixture` | tests and synthetic fixtures | 3 | 3 | out of scope |
| `historical_memo` | archived one-shots, doc helpers | 8 | 8 | out of scope |
| `unclassified` | model returned no label | 19 | 19 | owed |

**The charter's fire rule decides the landing: the refusing class's live count is 326, not 0, so the
widening lands REPORT-ONLY.** It is counted and summarised on the gate's verbose line, kept out of the
return value, and never raised — so the standalone strict surface's 2-finding contract is unchanged.
A 362-row warn-only flood in the primary list would be "a gauge nobody reads", which is this scanner's
own recorded lesson from the sp2 `gt_argmax` exclusion.

**The class split is NOT shipped as a gate rule, deliberately.** 303 of 309 classes came from the
advisory lane, and an advisory label may never gate anything. Converting the split into refusal needs a
*deterministic* authority-consumer predicate. That is named in NEXT_IF_RESUMED, not smuggled in here.

### The two instruments still differ by two files, and the difference is located

The census walks **320** files; the shipped gate's widening scope is **318**. The difference:

* **census-only (2 files)** — `experiments/.scratch/ane_venv_20260713/*`. The gate's ripgrep prefilter
  honours `.gitignore` and skips a scratch venv; the census's pure-Python walker did not. **The gate is
  right**; the census over-counted.
* **granularity** — the census excluded at FILE level, the gate at FINDING level, so a file carrying
  both a primary and a widened finding is treated differently. The gate's finding-level exclusion is the
  more precise one.
An earlier draft of this census carried its OWN copy of the `.npz` pattern; when the shipped pattern was
re-widened to reach `gt_strided_n200` and `gt_heldout_n400`, the copy silently kept the old form and the
census measured a scope the gate no longer had. It now **imports** the shipped tuple, and the two agree
to within the `.scratch` exclusion above. Two statements of one law drift — so there is one.

The gate's **376 additional findings / 317 files** is the authoritative number for the shipped code.
Sister of [[m123]] — two validators disagreeing is a finding, not a rounding.

**The class labels are themselves non-deterministic.** Re-running the census under the widened pattern
moved `continuity_frame` 10 → 15 and `advisory_instrument` 9 → 4 on a largely overlapping file set. The
class *counts* are therefore soft; the load-bearing number — 326 findings in the refusing class, far
from 0 — is robust to that jitter, which is the only thing the landing decision turns on.

### A regression I introduced, caught in my own second pass

The first version of the prefilter fix used ONE shared stem list for both scopes. **MEASURED:** that
made the PRIMARY scan pay the widening's prefilter cost — candidate files went **160 → 676** and the
verbose gate went **0.47 s → 3.83 s**, undoing exactly the ripgrep optimisation the sp2 landing had put
there. Each scope now carries its own stems (`_GT_LINEAGE_PRIMARY_PREFILTER_STEMS` /
`_GT_LINEAGE_NPZ_PREFILTER_STEMS`, with the union derived), and a test pins that the primary stems do
not match a `.npz` name. Primary is back to **160 candidates / 0.47 s**.

The widening's own scan still costs **~3.2 s**. It is paid ONLY on the `verbose` path — the human-facing
report, where the count is the point — so the quiet aggregate call pays zero. That cost is recorded in
the function's docstring rather than left for someone to rediscover.

### Also fixed: a latent VACUITY==PASS hazard

The ripgrep prefilter was a hardcoded `"gt_first6|gt_cache_"` literal at the call site, independent of
the pattern tuple. A pattern added to the tuple but not to that literal would have matched nothing and
the gate would have reported a clean 0 while real findings stood — the exact shape that already bit this
scanner once (a relative root made 139 candidates skip and 11 real findings read as 0). The prefilter is
now **derived** from a stem tuple, and a test asserts every shipped pattern is reachable through it.

### Scope of the vocabulary, with its residual named

The discriminator is that **a lineage-sensitive cache NAMES ITS SAMPLE COUNT**. Measured against every
`gt_*.npz` literal in the tree, `gt_[\w.\-]*n\d[\w.\-]*\.npz` splits them cleanly:

* **IN** — `gt_n600` `gt_n96` `gt_n24` `gt_n6` `gt_n2` `gt_n8` `gt_strided_n200` `gt_heldout_n400`
  `gt_n600_lstars_slim` `gt_n600_sR`
* **OUT** — `gt_tiny` `gt_synth` `gt_bad` `gt_bad_geometry` `gt_nokey` `gt_nomargin` (fixtures),
  `gt_nN` (a template placeholder), `gt_exact`

**Known residual, named not hidden:** `gt_pose_raw.npz` (6 sites) and `gt_cache.npz` (1 site) carry no
count and stay out of scope. They belong to the next widening.

---

## A3 — constant provenance: a MEASURED NEGATIVE

**The task.** ql3's value fingerprint FINDS an unlabelled constant but cannot say what it IS. The lane
was asked to read a ±6-line source window and label the constant's provenance as `measured_n600` /
`measured_prefix` / `derived` / `waived` / `unknown`.

**The result, cross-tabulated against the deterministic fingerprint** (which is exact and always wins):

| fingerprint verdict | advisory `measured_n600` | `measured_prefix` | `derived` |
|---|---:|---:|---:|
| **retired** (the literal IS a retired prefix value) | **13 — wrong** | 9 — right | 3 |
| n600_replacement | 10 | 7 | 6 |

(Cross-tab from the first pass, before the self-hit exclusion; the re-run after excluding the scanner's
own tuple gives 19 / 21 / 7 across all 40 fingerprint hits and does not change the verdict.)

**On the one class where truth is known, the lane scored 9 of 25 (36%).** Worse: across all 56 rows it
**never emitted `unknown` or `waived` a single time** — and `unknown` was the highest-value output, the
class that operationalises ql3's own sentence, *"an unlabelled constant is not a low-provenance
constant; it is an unfindable one."* A guided decode over a closed label set will always pick
*something*; asked to name a provenance that is not stated, it invented one from the variable name —
exactly what the instruction forbade in its last line.

**verdict_scope: ABSENCE-DETECTION IN A ±6-LINE SOURCE WINDOW, two formulations.** Stronger than a
single failed instruction, weaker than a family kill — window size and a two-stage retrieval design are
both untested. The lane is not killed:

* a **binary control** was run alongside — the strictly easier single-bit question ("do these lines state
  a provenance at all?"), which is the bit ql3's finding actually turns on. **It fails the same way:
  46 of 48 constants came back `provenance_is_stated`, 1 `no_provenance_stated`, 1 no advice.** ql3's
  whole finding is that many of these sites carry *no* provenance comment, so the control's answer is
  near-uniformly wrong in the same direction as the five-way task: **the lane will not report absence.**
  Two independent formulations failing identically moves this from "my instruction was bad" to a
  property of the task as posed.
* a **self-hit was fixed mid-flight**: the scanner's own `RETIRED_LITERALS` tuple was being counted as
  live constant sites (48 → 40 fingerprint hits once excluded). The ruler is not a consumer.

**Nothing from A3 was wired into a gate.** A lane that is wrong 64% of the time on the one class with
ground truth has no business beside a verdict, and shipping it because it was in the charter would be
the means-as-ends fake.

---

## Part B — fmtools 0.0.218 → 0.0.219

Repo: `~/Projects/fmtools` (ours), commits `3d005ff` (feature) + `a44dc0e` (format chore) on `75fa5d9`.

**Why anything changed at all.** Every pact caller reaching the on-device model was re-implementing the
same subprocess dance by hand: build a `@generable` schema, wrap it in `local_extract`, drive an asyncio
loop, swallow every exception, hope the JSON parsed. Three lanes in one arm would have been the fourth,
fifth and sixth copy.

**What landed:**

* `fmtools.classify` — `classify_batch` (async) / `classify_batch_sync` / `build_label_schema` plus
  `ClassifyRequest` / `ClassifyResult`. Exactly one result per input, in input order, never fewer.
  The label set is closed and pushed into the *generation guide*, so an `ok=True` label is always in-set
  and an out-of-set answer is reported as `label_out_of_set` rather than passed through.
* `fmtools classify` CLI — JSON Lines in, JSON Lines out, documented exit codes (`0` ok / `2` setup error
  / `3` strict failure / `4` usage error), with `--label`, `--instruction`/`--instruction-file`,
  `--timeout`, `--retries`, `--max-concurrency`, `--max-chars`, `--fail-open/--no-fail-open`.
* **Fail-open made explicit and tested.** Unavailable model, per-item timeout, or a guardrail refusal
  yields an `ok:false` row with an `error` string instead of raising. Caller-side misconfiguration
  (empty or duplicate labels, non-positive knobs) still raises `ValueError` — those are bugs, not
  conditions.
* **Tests: 39 new, offline by default.** A recorded backend covers ordering, per-item isolation, timeout,
  truncation, fail-open, every exit code, and every JSONL parse error — no Apple Silicon needed. Two
  `live_fm`-marked tests exercise the real Neural Engine and assert latency only as a **ceiling**.
* **The live tests are genuinely live.** `conftest` mocks `apple_fm_sdk` process-wide, so a naive live
  test would be a mock asserting against itself. The `live_fm` fixture swaps in the real SDK **and
  clears the `lru_cache` in `backends.apple_sdk._import_apple_fm_sdk` on both entry and exit** — that
  cache memoises whichever module is resolved first, and without clearing it the live tests failed in
  full-suite order while passing alone. They skip, never fail, when the SDK or model is unavailable.
* README section, CHANGELOG entry, `live_fm` marker registered, version bumped in both root and chat
  project (the repo's `check_version_policy.py` enforces parity — my first bump failed it).

**Gates: 691 passed, 12 skipped; `ruff check` clean; `ty` clean on the new modules.** CI already existed
and is thorough, so none was added.

**Found, not fixed, and reported:** `ruff format --check .` was **already red on main before this
branch** (5 files), so the CI Lint & Format job was failing for reasons unrelated to any feature work.
Landed as a **separate** formatter-only commit (`a44dc0e`) so the feature diff stays reviewable. `ty`
reports **7 pre-existing diagnostics** in `debugging.py` / `cache.py` / `async_generators.py` /
`decorators.py`; the baseline was measured by stashing, is unchanged, and none are in the new modules.

---

## What I did NOT do, and why

* **I did not wire the advisory lane into `src/tac/preflight.py`**, though the charter names
  `check_empirical_finding_memo_references_canonical_equation(verbose=True)`. CLAUDE.md's `tac`
  cleanliness rule is explicit: `tac` holds reusable codec / contest-runtime / contest-preflight logic,
  not Claude-workflow apparatus. An on-device advisory classifier is workflow apparatus. It lives in
  `tools/fmtools_advisory.py` and is consumed by `tools/preflight_hook.py`, which is where the charter's
  own "advisory column in the hook's output" belongs. A binding repo rule outranks a charter convenience.
* **I did not flip the `.npz` widening to strict.** 318 findings in the refusing class; the charter's own
  rule says warn-only unless that count is 0.
* **I did not ship the A2 class split as a gate rule.** Advisory labels may not gate.
* **I did not fix the 5 pre-existing `ruff format` files inside the feature commit**, nor the 7
  pre-existing `ty` diagnostics, nor `src/tac/canonical_equations/tests/` (still red on main, ddm_eq1 §3,
  owned elsewhere). Absorbing a sister's work is the collision the serializer discipline exists to
  prevent.
* **I did not touch** `ng2`/`ng3` (Metal) or `fs1`/`gv1`/`ng4` (CPU) custody, `upstream/`, or
  `submissions/semantic_joint_ctxmix/`.

---

## MEASURED / DERIVED

* **MEASURED:** every rate in the A1 table (n=29, ground truth ddm_eq1 §2, corpus at `d3212bed1`); the
  29/29 and 13/29 reproduction of eq1's counts; 11-of-15 recovery and 5-of-8 rejection; 7/29 label
  instability across repeats; the 1 locale-guardrail refusal; the A2 counts 2 → 364/310 and 362/308; the
  309-file class table and the 6-vs-303 force-list split; the A3 cross-tab (9/25 correct on retired
  literals, 0 `unknown` emitted); the 2.48 s hook latency; fmtools 691 passed / 12 skipped; the 5-file
  format and 7-diagnostic `ty` baselines measured by stashing.
* **DERIVED:** the union and intersection lanes, computed from the retained per-memo rows rather than
  re-run — the model is non-deterministic, so re-running would have produced a different companion to
  the same regex numbers.
* **NOT MEASURED:** absolute recall of any lane over the whole `.omx/research` corpus (the A1 corpus is
  selection-biased by construction); whether the A2 class labels are correct (303 of 309 are advisory,
  unverified against human adjudication, and shown to jitter between runs); whether the A3 negative survives a different window size or a
  two-stage formulation beyond the single binary control run here.

## STORES CONSULTED

`CLAUDE.md` (`tac` cleanliness · never-authority firewall · off-is-a-tracked-queue · payload retention) ·
`docs/operating_manual_craft_handoff.md` §§4–6 · memory
`reference_apple_ondevice_fm_fmtools_classifier_capability_20260703` · `[[m88]]` `[[m123]]` `[[m50]]` ·
`.omx/research/ddm_eq1_equations_leg_backfill_20260904.md` · `.omx/research/ddm_ql3_apparatus_debt_20260904.md` ·
`src/tac/preflight.py` (#344 tokens, GT-lineage scanner + the sp2 scope note) ·
`tools/magnitude_dismissal_detector.py` (the subprocess-under-fmtools-venv precedent) ·
fmtools `CLAUDE.md` / `AGENTS.md`.

## NEXT_IF_RESUMED

1. **A deterministic authority-consumer predicate for the GT-lineage widening.** The class table exists
   and 318 findings sit in the refusing class; what is missing is a rule that does not rest on an
   advisory label. Until then the widening stays report-only. *Owner: unassigned.*
2. **Measure absolute recall of the #344 token set** over all `.omx/research` memos, not just the
   selection-biased 29. That is the number that says whether the gate's 0.250 is the whole story.
3. **A3 second formulation** if the binary control shows the single bit is readable: a two-stage
   "is provenance stated? → if yes, which?" instead of one five-way call. If the control also fails, the
   family is closed for a ±6-line window and the next lever is window size, not wording.
4. **`gt_pose_raw.npz` / `gt_cache.npz`** — 7 sites outside the count-bearing vocabulary.
5. **`src/tac/canonical_equations/tests/` is red on main** (6 tests, pre-existing, ddm_eq1 §3). Still
   unowned.

<!-- # FORMALIZATION_PENDING: This memo measures INSTRUMENT quality -- the precision and recall of two
detector lanes against a human adjudication -- not a property of the codec, the score, or the archive.
The law it would need is a detector-efficacy law relating a gate's token-set narrowing to its recall
loss (the shape measured here: removing a 55.2% false-positive source cost 15 of 20 true positives), and
no such law is registered in tac.canonical_equations. Registering one on a single selection-biased
corpus of n=29 would be exactly the cross-regime constant transfer this campaign names as poison, so the
law is OWED, not invented. Reactivation: register it once a second, non-selection-biased detector corpus
exists -- see NEXT_IF_RESUMED item 2. -->
