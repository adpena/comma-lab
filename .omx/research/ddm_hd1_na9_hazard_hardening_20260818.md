---
arm: ddm_hd1
title: "Fix + self-protect + automate the four na9 hazards. H1's latched receiver pin is replaced by a consistency invariant DERIVED at consumption, verified green on the real fx1 candidate the old literal refused by construction, and consumed by the one Modal fire path so the refusal now lands before the meter starts. H2's root cause is measured: the corrections indexer has NO TRIGGER -- built once by hand on 08-05, never re-run; rebuilt (horizon 0805->0818, blind 13 days -> 0) and fitted with a freshness banner derived from the index's own rows. H3 narrowed the coder-axis closure at source in three places and landed a scope-word detector. H4 carries a premise CORRECTION: the 600x6 pose array is NOT free to retain."
utc: 2026-08-18
axis: "[$0 local apparatus hardening] -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "rr4 S 0.15853325034789678 @ 181,161 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "stated inline per fix"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_hd1 — the four na9 hazards, fixed and self-protected

**Operator binding 2026-08-18: _"Must fix and self protect against and make automated and dynamic
and recursively hardened and polished."_**

Every fix is two landings — the cure, and the gate that refuses re-introduction — and every gate
was run in **both directions**: red on the disease, green on the cure. No Modal, no scorer run, no
Metal, $0. **Pointer UNMOVED.**

| fix | cure | gate | controls | status |
|---|---|---|---|---|
| **H1** latched receiver pin | `ab271e4021` | 18 tests + 2 re-introduction gates | 4 executed on the **real fx1 tree** | **LANDED** |
| **H2** index date horizon | `34c54f3b88` | 8 tests, both directions | live store, before **and** after rebuild | **LANDED** |
| **H3** closure names wrong object | `f22fd3e7d5` | 14 tests + corpus run | pos/neg controls inside the detector | **LANDED, precision partial** |
| **H4** retained signal | `5f4692c0d2` | 8 + 6 tests | red/green on the reduction check | **LANDED w/ premise correction + named remainder** |

---

## H1 — the latched receiver pin (the LIVE hazard)

### What was wrong

`experiments/ddm_pq2_compress_e2e.py` carried `EXPECTED_ARCHIVE_SHA256` / `EXPECTED_ARCHIVE_BYTES`
as **module-level literals with no CLI override**. The fail-closed check was right; the mechanism
was a latched value, so the one entry point built to byte-close candidates **refused fx1's
180,601 B row and sa1's 18 candidates by construction**.

A second, deeper surface had no local check at all: rr4-lineage receivers pin the archive they
decode, but nothing verified that the staged `inflate.py` named the staged `archive.zip`. That
mismatch was only catchable **remotely, at decode time, after the meter started**.

### The cure — a consistency constraint, not a stored value

`src/tac/candidate_seal.py` (new; brick 1 of the #1115 seal contract, built so #1115 consumes it
rather than growing a twin). It never latches what the pin *should* say. It measures the staged
archive at consumption and asks one question: **does the receiver beside it name these exact
bytes?** That question answers for rr4, fx1, sa1 and every successor without editing a literal.

* `read_receiver_pin` — AST parse, never an import (a shipped `inflate.py` imports torch at module
  scope; importing it would be a side effect and would fail off its own tree).
* `check_pin_consistency` → `CONSISTENT | MISMATCH | PIN_ABSENT | RECEIVER_MISSING | ARCHIVE_MISSING`.
  **`PIN_ABSENT` is never `CONSISTENT`** — a vacuous check is reported, not passed.
* `repin_receiver` — the compose-time staging step MAIN previously did by hand. It **re-reads and
  re-checks its own output** and restores the original bytes if the rewrite did not produce a
  consistent tree.
* `read_frontier_archive_identity` — the dynamic default, straight from
  `canonical_frontier_pointer.json` at call time, which is the hot state's own BINDING CURE.

`experiments/ddm_pq2_compress_e2e.py` now resolves the expected candidate at run time (CLI →
`--candidate-runtime` → pointer) and separates the **rebuild recipe** (corrector, token stream,
base archive) from the **candidate identity**. Asking for candidate X under candidate Y's recipe is
refused **up front, by name**, instead of failing deep inside the rebuild and blaming the algorithm.

`tools/fire_modal_auth_eval.py` — the ONE Modal fire path — consumes the check at a new **stage 3b
SEAL**, refusing with `rc=6` before dispatch, with `--repin-receiver` as the automated staging fix.

### Controls — EXECUTED, both directions, on the real retained fixture

```
=== CONTROL 1 (GREEN, REAL FIXTURE): retained fx1 candidate_runtime, NON-rr4 ===
SEAL PIN CONSISTENT: receiver pins the staged archive (180,601 B sha 65c75d7f097df930…)
  measured_sha256 65c75d7f…  measured_bytes 180601   pinned 65c75d7f… / 180601

=== CONTROL 2 (RED, THE DISEASE): same fx1 archive under an rr4-pinned receiver ===
SEAL PIN MISMATCH: ARCHIVE_SHA256 pins 35ac2b9beb7e6fa8… but the staged archive is
65c75d7f097df930…; ARCHIVE_BYTES pins 181,161 B but the staged archive is 180,601 B;
archive and runtime are ONE sealed object: re-pin the receiver from the staged archive …

=== CONTROL 3 (THE CURE): re-pin that broken tree at compose time ===
changed=true  verdict_before=MISMATCH  verdict_after=CONSISTENT
old 35ac2b9b…/181,161  ->  new 65c75d7f…/180,601

=== CONTROL 4 (GAUGE STAYS LIVE): stage different bytes after the cure ===
SEAL PIN MISMATCH: … pins 65c75d7f… but the staged archive is 872165167eb23c05…
```

Control 4 is the detector-does-not-zero-on-its-own-cure leg: after a successful re-pin the gauge
still fires on the next mismatch.

pq2, executed:

* **RED** — fx1's tree under rr4's recipe: `REFUSING a recipe/candidate mismatch: the expected
  archive is 65c75d7f… (180,601 B, from candidate_runtime:…) but the loaded rebuild recipe is
  'rr4', which produces 35ac2b9b… (181,161 B).`
* **GREEN** — fx1's tree **with** an fx1 recipe: passes the identity gate and reaches input
  resolution. **This is the decisive control: the candidate that was refused by construction is now
  accepted.**
* **GREEN** — no flags at all: identity derived from the live pointer, matches, proceeds.
* **RED** — half an identity (bytes without sha): refused.

### Re-introduction gate

`test_pq2_does_not_latch_an_expected_archive_identity_again` fails if
`EXPECTED_ARCHIVE_SHA256`/`EXPECTED_ARCHIVE_BYTES` reappear as module constants, or if the four
caller-facing flags disappear. `test_the_fire_path_consumes_the_seal_check` fails if the fire path
stops importing the brick or stops refusing. **18/18 pass, no skips** — the real-fixture leg ran.

**Scope: INSTANCE ×2** (the pq2 literal; the missing local seal check). The latched-literal genus
itself is unchanged — this closes two of its measured sites, not the class.

---

## H2 — the corrections-index date horizon

### Root cause — MEASURED, and it is not what my charter predicted

The charter predicted *"an ingestion path/glob that a directory rename orphaned around 08-05."*
**That is wrong.** The measured cause is simpler and worse:

> **The tool has no trigger. It was run exactly once, by hand, on 2026-08-05, and nothing ever
> calls it again.**

Evidence, with the alternatives ruled out rather than assumed:

| hypothesis | verdict | evidence |
|---|---|---|
| never re-run | **CAUSE** | `git log` on both the tool and its output dir returns **exactly one commit** (`7b72d4edaf`, 08-05); all six outputs share a 27-second mtime window; no hook, preflight gate, cron or scheduler references it |
| glob orphaned by a rename | ruled out | the glob was **executed** against the live tree: 8,830 memos reached, max date 20260818 |
| hardcoded date/limit | ruled out | the literal `20260805` appears once, as an **output folder name**, never an input filter |
| crash/skip on newer files | ruled out | `read_text(errors="replace")`; the rebuild processed all 8,830 files with zero exceptions |
| date parser fails on new names | ruled out **by absence** | there is no date parser — rows carry no date field; the horizon is a token in the source path |

Also corrected: the index lives at `.omx/research/ddm_au1_20260805/au1_corrections_index.jsonl`,
**not** `.omx/state/` (that path is gitignored, deliberately).

### The cure — a DERIVED banner, plus the rebuild

Rebuilt: **4.2 s, $0, pure local CPU.**

| | before | after |
|---|---:|---:|
| horizon | 20260805 | **20260818** |
| rows | 11,840 | **14,076** |
| sources | 2,733 | **3,310** |
| corpus files past horizon | **1,029 of 8,830** | **0** |
| blind days | **13** | **0** |

`src/tac/corrections_index_freshness.py` derives the horizon **from the index's own rows at read
time** and compares it against the corpus that exists at read time. Nothing trusts a recorded build
date: a stamp is one more constant that can rot, and the rot would again be invisible. Wired into
`tools/codex_arm_queue.py` as an **unconditional banner** on every `lint` (a healthy index announces
itself too) and as a sixth advisory leg at spawn time.

### Controls — EXECUTED, both directions, on the live store

```
BEFORE (red):
[corrections-index] horizon 20260805 · 11,840 rows over 2,733 sources (21 undated) ·
corpus reaches 20260818 across 8,830 files · BLIND 13 day(s) · STALE
  WARN corrections index horizon is 20260805 (13.0 days old, bar 2.0 days) — REBUILD …
  WARN 1029 corpus file(s) of 8830 are dated past the index horizon 20260805 …

AFTER (green):
[corrections-index] horizon 20260818 · 14,076 rows over 3,310 sources (41 undated) ·
corpus reaches 20260818 across 8,830 files · FRESH
  WARN the stale-number lint leg is FAIL-CLOSED: index rows carry no 'quantity' field …
```

The recursive leg (the freshness check gets its own control) is in the 8 unit tests: a stale
fixture warns, a **fresh fixture does not** — without the green leg an always-warning check would
look like it works. One test proves the horizon is derived from rows and **not** from mtime: it
`touch`es the file and the verdict stays STALE.

### The finding I did not go looking for

**A rebuild alone would NOT have revived the consumer.** `_lint_stale_numbers` has been fail-closed
since 2026-08-17 (`03b02ddc99`): it returns `[]` unless index rows carry a `quantity` field, because
the schema pairs adjacent numbers in a window and cannot say *which quantity* a number is. So a
freshly-rebuilt index would have read as healthy while emitting nothing — the vacuity-passes disease
one layer up. **The banner therefore prints the consumer's own fail-closed state beside the
horizon**, and it does so right now, on the live rebuilt store (see the AFTER block above).

**Named remainder:** adding `quantity` to the writer's schema is the precision repair, adjudicated
separately on 08-17. It is **not** done here and the banner says so out loud.

**Scope: INSTANCE** (this index). The no-trigger genus is not closed — nothing yet guarantees a
rebuild; the banner guarantees you cannot *fail to notice* that one is owed.

---

## H3 — a closure is only as good as the object it names

### Narrowed at source — headline AND body, append-only

Per `corrections land in bodies, headlines keep the stale number`, each edit lands in both places;
per Catalog #110/#113 the original text is **preserved**, never rewritten, and each superseded line
carries its own inline marker so a mid-document reader cannot miss it.

1. **`ddm_dc1_…_20260816.md`** — H1 retitled to *"the coder-SWAP ceiling is CLOSED on hv1 at fixed
   probabilities"*; a banner naming its four superseded lines individually; inline `⚠ NARROWED`
   markers at the §Verdict-conditional sentence, the §Consequence "coding half" sentence, and
   owed-row 4 (*"Retire the coder family"* → **coder-SWAP family only**).
   The memo's own §"What I did NOT measure" item 3 already said *"A logistic-mixing CM coder was not
   built … I am relying on the bound"* — that is the honest sentence the headline should have
   inherited, and the banner says so.
2. **`ddm_nx1_…_20260816.md`** — row 1 and *Genuinely closed* #1 both narrowed to coder-SWAP, with
   the measured refutation (`fx1`, −560.07 B, 180,601 B, sha `65c75d7f…`, ΔS −3.72881e-4, **72×**
   the ≤7.8 B ceiling) named in place.
3. **`.omx/state/main_hot_state.md`** NEXT_BOUNDARIES — *"POST-HOC BYTE SURGERY … IS EXHAUSTED"* →
   **"POST-HOC LOSSLESS BYTE SURGERY …"**, with both post-hoc rows measured *after* it (fx1 −560 B
   pure-rate; sa1 −2,889 B lossy, distortion unmeasured) named beneath it. Edited through
   `tools/main_hot_state.py --set-section`, so the other six sections are untouched.

**Both measurements stand.** dc1's ≤7.8 B is unrefuted; only the word was one level too wide.

### The detector — and an honest miss

`tools/au1_measurement_integrity_audit.py` gains a **6th pass**, `scope_vs_object_detector`, wired
into `run()` in the four places the monolith requires. It asks na9's one-line question of every
closure sentence and is **warn-only**.

**My charter predicted 10–40 flagged sentences. My first pass emitted 3,062.** That is a ~100× miss
and the first pass was a toy: 382 flags were our own phrase **"byte-closed"**, 257 were
**"closed-form"** (a kind of maths), and 1,588 came from treating "all/any/every/whole" as
wide-object words. I rebuilt it at optimal form — whole-word matching, a measured false-friend list
of our house dialect, a 60-char proximity requirement, and a narrowed object vocabulary:

| | first pass | tightened |
|---|---:|---:|
| rows | 3,062 | **609** |
| "byte-closed" false positives | 382 | **4** |
| "closed-form" false positives | 257 | **1** |
| already-scoped (the passing denominator) | 1,548 | **381** |

Live-corpus run: **8,833 memos scanned · 609 rows over 431 sources · 381 closure sentences already
carried a scope word and passed.** Restricted to the live window (dated ≥ 20260801): **182 rows over
135 sources** — the actionable set.

**Honest limit:** 609 is still ~15× my prediction, and flagged:scoped at 609:381 means the gauge
fires on 62% of closure sentences it examines. It is readable and warn-only; it is **not**
gate-grade precision and must not be promoted to blocking on this evidence. The denominators are
reported so a reader can tell "clean" from "did not look".

The detector carries its own positive control (na9's instance-4 sentence must be caught) and
negative control (the same claim, correctly scoped, must **not** be caught but must be **counted as
scoped**). Both are asserted in the emitted summary and in tests. **14/14 pass.**

**Scope: INSTANCE ×3** for the narrowings; the detector is **FORMULATION** — one warn-only lexical
formulation of the scope question, not the class.

---

## H4 — retained signal

### (a) The per-pair pose array — with a PREMISE CORRECTION I must carry

The charter described this as *"~14 KB, $0 — zero extra compute."* **Measured: that is false.**

`upstream/evaluate.py:81` is the discard site — `posenet_dists += posenet_dist.sum()` consumes the
`(B,)` per-pair vector. That vector is created and destroyed **inside the upstream process**; it
never enters our address space, our harness only ever parses scalars out of `report.txt`, and
`upstream/` is read-only. Verified absent from all **535** `contest_auth_eval.json` files under
`experiments/results/` — no per-pair key exists anywhere today.

So the only legal retention is a **second pass we own**: measured **~40 s contest-CUDA T4** and
**~214 s contest-CPU** from harvested `evaluate_elapsed_seconds`. The payload really is ~14 KB; the
compute is not free. (A `PYTHONPATH`-shim tee would be zero-cost but changes what the scorer process
imports — a score-identity risk under NO-FAKE #8. **Rejected, and recorded as rejected.**)

Landed `src/tac/pose_per_pair_retention.py`, **opt-in and default OFF**, running strictly *after*
the scored result exists so it cannot perturb a number. It imports upstream's own dataset and
`DistortionNet` as libraries rather than reimplementing the loop — a reimplementation would be a
different computation wearing the same name. Threaded into `experiments/contest_auth_eval.py` as
`--retain-per-pair-distortion-dir` (`default=None`, so blast radius when unset is one false `if`),
inside a `try/except` that can never fail an eval.

**The load-bearing property is the self-check, not the file.** The retained vector is reduced
exactly as upstream reduces it and compared to the scalar upstream reported; disagreement beyond
1e-6 relative marks it **UNVERIFIED** in its own manifest, because a per-pair map that does not add
up to the scored number is worse than no map. Controls: green (faithful vector verifies), red (a
vector that misses is flagged, **and its bytes are still kept** as evidence of the disagreement),
plus unanchored/empty cases. **8/8 pass.**

**Named remainder — deliberately NOT half-done.** `experiments/modal_auth_eval.py` threads its
optional flags through **six wrapper layers** (`:468, :976, :1044, :1098, :1152, :1231`) plus the
argv builder (`~:752`) and `_collect_artifacts` (`:432-445`). Threading part of that chain would
land an **inert flag** — precisely the orphan class this arm exists to extinct. The capability is
complete and usable at the `contest_auth_eval.py` layer; the Modal pass-through is owed, with its
call sites named above. **Owner: MAIN or the eval-harness maintainer.**

### (b) The ensemble-calibrated-falsifier law — registered

Confirmed **absent** first (0 hits for `ensemble`/`falsifier`/`stochastic`/`single_seed` across all
7 builder modules, all 114 ledger rows and all 82 live ids). Landed
`single_seed_falsifier_on_stochastic_endpoint_v1` (`PARADIGM_RIGOR_LOSS`, `SEVERITY_HIGH`) with the
LR-ladder anchor: single-control **REFUTED** → 2-seed ensemble **WEAKENED-DIRECTIONAL** (lr-up) and
**NULL** (lr-down), with **zero new treatment runs**. The verdict moved because the instrument
moved. Its unwind path also covers the unaffordable case: where an ensemble cannot be run, the
verdict is INSTANCE and the row must say the band is uncalibrated rather than report REFUTED.

Registry **82 → 83 live rows**. It becomes live to Catalog #373's gate immediately — that gate is
generic and needs no per-id entry. **6/6 tests pass; 202 registry tests still pass.**

---

## PRIOR-LAW PREDICTION — verdict

| prediction | verdict |
|---|---|
| H1 ≈ 50-line staging change + 2 controls, fx1 fixture makes it real | **HIT** — the fixture was decisive; 4 controls executed on it |
| H2 root cause = a glob orphaned by a directory rename | **MISS** — it is *no trigger at all*; the glob is healthy, proven by executing it |
| H3 scope pass flags 10–40, most honest | **MISS, ~15×** — 609 after tightening (3,062 before). The instrument reports its denominator rather than assuming disease, which is the part that held |
| H4a ≤10-line wire | **MISS on the premise** — not $0, not zero-compute; it is a second scorer pass |
| escape clause: if H2 cannot be re-run cheaply, land the banner alone | **did not fire** — the rebuild is 4.2 s / $0, so both the banner *and* the repair landed |

Three of five missed. Two of the misses were my charter believing a cost that had never been
measured — the same disease as the hazards this unit was sent to fix, one level up.

---

## STORES CONSULTED

* **Governing:** `CLAUDE.md` (NO-FAKE; THE GOAL; ALWAYS KEEP THE PAYLOAD; two-landing law;
  Catalog #110/#113 append-only; verdict-scope ladder) · `.omx/state/main_hot_state.md`
  (POINTER_LINE, NEXT_BOUNDARIES, the stale-constant genus + its BINDING CURE) ·
  `.omx/research/charters/ddm_hd1_na9_hazard_hardening_20260818.md`.
* **H1:** `.omx/research/ddm_na9_…_20260818.md` N3 (the cure-defect finding) ·
  `experiments/ddm_pq2_compress_e2e.py` read in full incl. the argparse block ·
  `tools/fire_modal_auth_eval.py` read in full · the retained
  `/Volumes/APDataStore/pact/ddm_fx1/candidate_runtime/` (archive + its own re-pinned `inflate.py`) ·
  `.omx/state/canonical_frontier_pointer.json` · memory
  `freeze_and_constrain_through_engineering_20260818` (#1115 named consumer) · dy1 scope-law
  (`.omx/research/ddm_dy1r_20260805/RECEIPT.md`).
* **H2:** `tools/au1_measurement_integrity_audit.py` read in full · `tools/codex_arm_queue.py`
  `:1322`, `:1491-1608`, `:1750-1780`, `cmd_lint` · `git log` on the tool and its output dir ·
  the index parsed directly (11,840 rows / 2,733 sources / date histogram) · commit `03b02ddc99`
  (the fail-closed retirement).
* **H3:** `ddm_dc1_…_20260816.md` (`:1`, `:190`, `:208`, `:227`, and `:203` — the honest sentence) ·
  `ddm_nx1_…_20260816.md` (`:124`, `:218`, `:242-248`, `:342-343`) · `ddm_fx1_…_20260817.md`
  (`:1-27`, §9.1 gate table) · `tools/main_hot_state.py` (`SECTIONS`, `--set-section`) · memories
  `corrections_land_in_bodies_headlines_keep_the_stale_number_20260805`,
  `measured_object_vs_named_object_20260816`.
* **H4:** `upstream/evaluate.py:79-104` + `upstream/modules.py:82-84,154-158` (**read only**) ·
  `experiments/contest_auth_eval.py` (`_run_upstream_evaluate`, the scorer-input-cache precedent at
  `:1431`/`:2971`) · `experiments/modal_auth_eval.py` (`:432-445`, `:704-765`, the six wrapper
  layers) · `src/tac/sidechannel_score_table.py:169-176` (`load_distortion_net`, the
  import-upstream-as-a-library precedent) · `src/tac/canonical_anti_patterns/` (contract, registry,
  `na3_subset_bias_builders.py` as the house template) · 535 harvested `contest_auth_eval.json`.

---

## Honest limits of this unit

1. **I ran no scorer, so `compute_per_pair_distortion` is untested against a real eval.** Its pure
   core (retention, verification, manifest) is fully tested; the second-pass loop mirrors upstream's
   by importing upstream's own objects, and its correctness is asserted at runtime by the reduction
   check rather than by a test I could execute here. That is the honest state.
2. **The H3 detector is warn-only and its precision is not gate-grade** (609 rows, 62% fire rate on
   examined closure sentences). Do not promote it to blocking on this evidence.
3. **The Modal thread for H4a is owed, not done** — named above with its exact call sites. I chose a
   working single layer over an inert flag.
4. **The corrections index's `quantity` precision repair is owed** and is not mine; the banner
   reports the consumer's fail-closed state so it cannot be mistaken for healthy.
5. **The rebuilt index still writes into a folder literally named `ddm_au1_20260805`.** The reader
   hardcodes that path, so redirecting `--outdir` would orphan it. The stale *name* over fresh
   *content* is now harmless because the horizon is derived from row contents, never the folder
   name — but it reads badly and a rename needs the reader moved in the same landing.
6. **Two pre-existing lint findings in `tools/codex_arm_queue.py` and three in
   `experiments/contest_auth_eval.py`** are untouched; I verified counts are identical before and
   after my edits so none is mine.
7. **No payload was materialized** by this unit beyond test fixtures in `tmp_path`, so there is
   nothing to retain under `/Volumes/APDataStore/pact/ddm_hd1/`. The ALWAYS-KEEP-THE-PAYLOAD law is
   satisfied vacuously, and I say so rather than leaving it unstated.

## NEXT_IF_RESUMED

1. **QUEUED — owner MAIN.** Thread `--retain-per-pair-distortion-dir` through
   `experiments/modal_auth_eval.py`'s six wrapper layers + argv builder + `_collect_artifacts`.
   **Fire trigger: before the next carrier re-solve** — `t1h` §11.4 item 1 blocks its own item 4
   behind this array.
2. **QUEUED — owner MAIN.** Add `quantity` to the corrections-index writer schema. The
   `_lint_stale_numbers` leg self-reactivates on a rebuild that carries the field; no second code
   change is needed. **Fire trigger: next time a charter lint is trusted to catch a stale number.**
3. **QUEUED — owner MAIN.** Give the indexer a trigger (preflight, SessionStart hook, or a
   rebuild-if-stale call inside the banner). It is 4.2 s and $0; the banner currently makes the debt
   visible but does not pay it.
4. **QUEUED — owner MAIN.** Rename `ddm_au1_20260805/` to an undated home **and move the reader's
   hardcoded path in the same landing** (`codex_arm_queue.py:1322`).
5. **FOLDED.** H1, H3's three narrowings, and H4b are complete; no successor owed.

**Own-vehicle frontier: `S 0.15853325034789678 @ 181,161 B [contest-CUDA T4, n600]`, archive sha
`35ac2b9b…` — UNMOVED by this unit.** This arm ran no scorer, fired nothing, spent $0, and did not
lower the score. It replaced one latched literal with a measured invariant, restored a 13-day-blind
instrument and made its horizon impossible to miss, narrowed a FAMILY closure that a byte-closed row
refutes at the name level, and registered one law the corpus was missing.
