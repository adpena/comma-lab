---
schema: ddm_iv1_inventory_drain.v1
date_utc: 2026-08-03
arm: ddm_iv1 (hv2-D7 — the $0 inventory-drain arm)
lane_id: "lane_ddm_iv1_inventory_drain_20260803"
charter_row: "hv2-D7 (.omx/research/ddm_hv2_harvest_queue_20260803.jsonl:39)"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory; $0 read-only over cached artifacts + source; ZERO scorer forwards, ZERO launches, ZERO dispatch]"
verdict_scope: "INSTANCE (these artifacts, this window)"
empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_iv1 — inventory drain: two headline claims REFUTED, one gating measurement re-typed

## §0 Answer first

**Nothing here moved any score.** Pointer `0.1910828242 [contest-CPU]` UNMOVED; live own-vehicle
best `S 0.7910689 [macOS-CPU advisory]`. This arm produced **no candidate, no archive, no run**.

Three results are load-bearing, and two of them delete work rather than add it:

1. **HEAD 1 — φ is not readable from the ms4d bundle at any cost.** gc15's "`$0` but needs the
   composite-R adjoint from the ms4d bundle — **a read, not a run**" is **FALSIFIED at the artifact
   level**. The artifact is complete and honest; it is the **wrong operator**. φ is re-typed from
   *read* to *run*, and it **inherits a named blocker recorded 11 days before gc15 asked for it**.
2. **HEAD 2 — the kl1 −880 B codec is NOT "built-unwired".** It is wired end-to-end — encoder,
   live call sites, and receiver — and has been **on the shipping path since v4c**. p1a's P0
   grade-5 orphan claim is **REFUTED by mechanism-join**. QA55's container half is **already
   banked by the real builder**.
3. **HEAD 3 — 9 of 16 T1 rows (56%) needed only a join, not work.** The never-named instrument
   measures commit messages; it cannot see the parking store.

---

## §1 HEAD 1 — the φ preflight: FALSIFIED as a read, RE-TYPED as a run

### What gc15 asked for

`ddm_gc15_fresh_vs_warm_20260731.md:259` defines it exactly:

> **the visible fraction φ** … *the fraction of a uniform **parameter-space** kick whose image lies
> in `range(A)`*

with a pre-registration at the same line: **"if the preflight returns φ > 0.8, I withdraw arms D±."**
`gc15:390` calls 1/φ *"the binding unknown"*; `gc15:466` states the preflight *"is `$0` but needs the
composite-R adjoint from the ms4d bundle — a read, not a run."* `gc15:445` makes it the **gate on
whether arms D± run at all**.

### What the artifact actually contains — MEASURED, re-derived by me

Primary artifact: `.omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/composite_r_direct_n600.json`
(1,132,352 B; sha256 `231d74bf7b0a690a9fadcb855e64429e8ce34e28ad4c8683766cf6ebe65a7807` per its own
component receipt).

| property | MEASURED value | denominator |
|---|---|---|
| `coordinate_domain` | `POST_R_PENULTIMATE_HEAD_QUOTIENT` | — |
| `metric_mode` | `DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT` | — |
| blocks present | **25** (over **15 distinct pairs**, 8 buckets) | `pair_count` field says 600; `measurement_schedule` lists `full_n600` — **the artifact carries 25 blocks, not 600** |
| `actuation_status` | `UNREACHABLE_BY_COUNTED_COORDINATES` | **25 / 25** |
| `composite_r_adjoint_readback` vs `fisher_vector` | **bitwise identical** | 25 / 25 |
| `composite_r_model_hessian` vs `margin_fisher_gram` | **bitwise identical** | 25 / 25 |
| rank of the second-order form | **exactly 1** — `G = c·f fᵀ`, max rel. residual **3.26e-16**; top-eigenvalue share of trace = **1.000000000000000** | 25 / 25 |
| `fisher_euclidean_cosine` | 16×`1.0`, 6×`0.9999999999999999`, 2×`0.9999999999999998`, **1×`−1.0`** | 25 / 25 |
| the one sign flip | pair **42**, bucket `lane_movable__boundary__static_in_image`, `support_count` **2** | — |
| stamped vs recomputed (cosine, and `‖f‖/‖e‖`) | **agree in 25/25** — the stamped fields are **honest** | 25 / 25 |
| `fisher_to_euclidean_rel_norm` spread | 8.31e-08 … 4.97e-01, median 5.46e-03 — **5.98e6× range** | 25 / 25 |

Two consequences worth carrying independently of φ:

- **Four field names, two numbers.** Per block, `composite_r_adjoint_readback`, `fisher_vector`,
  `composite_r_model_hessian` and `margin_fisher_gram` reduce to a 4-vector `f` and a scalar `c`.
  Any consumer treating them as four independent measurements is **double-counting**.
- **The dual metric here is angle-free.** Fisher and Euclidean are parallel or antiparallel in
  25/25. The entire dual-metric content is (a) a per-block **scale spanning 6 orders of magnitude**
  and (b) **exactly one sign flip**, at a bucket whose support is **2 sites** — the thinnest
  possible evidence. MEMORY's "never one alone" is vindicated on the **scale** axis: a 5.98e6×
  range would badly misrank anything ordered by the wrong one.

### The verdict

φ is defined over a map whose **domain is θ**. This artifact has **no θ dimension anywhere**: its
coordinates are the post-R penultimate head quotient, its metric mode is explicitly
`NO_ACTUATOR_INPUT`, and every block is stamped `UNREACHABLE_BY_COUNTED_COORDINATES`.

The bundle's own `BUNDLE-COMPLETE.json` declares
`consumers = ["ms2_typed_quotient_solve", "pf2r_metric_active_three_formulation", "rd1_dimension_duals"]`
— **φ / arms D± is not among them**, and `blockers = []`. The bundle is **COMPLETE for what it
declares** and is simply not the operator gc15 needs.

**⇒ The `$0` read cannot return φ. Not "is hard to"; cannot.**

### The convergence that settles it — and that nobody joined

`.omx/research/codex_premise_falsification_r1b3_receiver_pullback_20260720T185300Z_codex.md`
(**2026-07-20, eleven days before gc15**) already measured this exact gap, verbatim:

> The frozen centered SegNet head is exactly rank **4** in its **144**-dimensional penultimate
> feature-patch chart. Pulling that chart back to decoded RGB still traverses the nonlinear feature
> extractor, resize, and uint8 realization path… **No present artifact supplies that pullback** or
> endpoint perturbation custody.

Named blocker: **`R1B3_P1_RECEIVER_COORDINATE_JACOBIAN_AND_REALIZED_SECANT_ABSENT`**; reactivation
criteria on file in `r1b3_producers_20260720T185300Z.md`.

That falsification **explains the rank-4 shape I measured independently** (4-vectors, 4×4 grams,
`rank4_pair_normal`). It is the same missing pullback. **φ's preflight and p1a item 25 (`r1b3
E_n600`) are the SAME owed measurement** — discharging one discharges both. That fold is the
consolidation this head produces.

### What I explicitly REFUSE to report

I decline to publish a φ number. The two available proxies — rank-1-within-the-4-dim-block, and
4-of-144 head rank in the chart — are both in **scorer-feature space, not parameter space**, and
rest on **15 pairs**. Converting the first to "1/φ ≈ 4" would land squarely inside gc15's own
"strongly indicated, 2–5×" band, which is the **most dangerous kind of wrong**: a wrong-space
number that confirms the hypothesis. **gc15's pre-registration is discharged in NEITHER direction.
Arms D± remain GATED.**

### The re-type (this is the deliverable)

| | before (gc15) | after (MEASURED here) |
|---|---|---|
| cost class | `$0` read | **RUN** — parameter-space Jacobian/JVP through the frozen scorer **plus** the R-pullback to decoded RGB |
| blocker | none named | **inherits `R1B3_P1_RECEIVER_COORDINATE_JACOBIAN_AND_REALIZED_SECANT_ABSENT`** (dated 07-20) |
| scheduling | "rides along regardless of which axis owns the slot" | **cannot ride along**; needs the r1b3 producer first |

### My own defect, recorded

My first pass left a `break` in the loop and I reported `relnorm` "constant at 5.617e-04" and
"stamped cosine all exactly 1.0" — both from a **1-element sample**. I then built a hypothesis on
it ("the stamped field hides a sign flip"). Re-deriving **refuted my own hypothesis**: stamped ==
recomputed in 25/25; the fields are honest. This is the vacuity genus (`m50`) biting **inside an
arm whose subject is instrument honesty**, one hour after writing about it. Recorded, not buried.

---

## §2 HEAD 2 — the kl1 −880 B codec is WIRED. p1a's P0 orphan claim is REFUTED.

### The claim under test

p1a §4: *"the verified −880 B codec is **built and unwired**. Repo-wide grep (positive + negative
control, both passed): `ddm_kl1_pose_field_receiver` appears in exactly 4 files… **No exporter,
builder, or packet path imports it.** Per `built_elsewhere_unwired_is_p0_20260801` that is the P0
grade-5 class."*

### MEASURED — the codec is wired on both sides, on the shipping path

| leg | file:line | evidence |
|---|---|---|
| magic | `experiments/ddm_v4c_build_composed_archive.py:39`, `ddm_v4d_build_composed_archive.py:42` | `KL1_MAGIC = b"KL1PWF01"` |
| encoder | `ddm_v4c…:51-59`, `ddm_v4d…:54-62` | `encode_kl1_field()` — hi/lo byte planes, column-major, `brotli.compress(…, quality=11)`. **This is the kl1 byte-plane codec.** |
| **live call sites** | `ddm_v4d…:257`, `:258` | `tp_member = encode_kl1_field(pose_store…)`, `ab_member = encode_kl1_field(ab…)` |
| shipping member | `ddm_v4d…:300`, member list `:46` | both feed `state/pose_warp.stp`, which is in `MEMBER_ORDER` |
| **receiver** | `experiments/inflate_runner_v4c.py:54,70`; `inflate_runner_v4d.py:91,106` | both parse `if member[:8] != KL1_MAGIC` |

The `−880 B` has therefore been **banked in the shipping path since v4c**, through v4d
(0.9639878) and every own-vehicle row after it.

### The defect class — and why the controls could not save it

p1a grepped the **filename** `ddm_kl1_pose_field_receiver`, found 4 files (its own source + 3
memos), and inferred "no wiring". The codec's **mechanism** was re-implemented **inline** under
`KL1_MAGIC` in the v4c/v4d builders and their receivers.

**A filename-join answers "is this FILE imported?", not "is this CODEC used?"** p1a reported
"positive + negative control, both passed" — and they did pass, correctly, on the filename. **A
passing control cannot rescue a mis-specified join key.** That is the generalization worth keeping,
and it is the exact inverse of p1a's own excellent QA52 finding (a runner token that was really the
arm's *product*). Both are name-joins standing in for mechanism-joins.

### QA55's recipient question — resolved, and mostly already banked

`DEFLATE_MEMBERS = {"manifest.json", "state/selector.sec"}` at **`ddm_v4c…:44` and `ddm_v4d…:47`**
(identical). QA55's own fire-condition — *"IF the real builder already deflates members the
container slack shrinks"* — has **FIRED IN THE NEGATIVE**:

| QA55 claimed win | status against the REAL builder |
|---|---|
| `manifest.json` 1,266→609 (−657 B) | **already DEFLATED** — banked, not open |
| `state/selector.sec` 535→277 (−258 B) | **already DEFLATED** — banked, not open |
| `state/pose_warp.stp` 6,844→5,964 (**−880 B**) | STORED, but the win is the byte-plane **reformat** (intrinsic, as the ledger says) — **already wired** via `encode_kl1_field` |
| `pose_stub` 83→76 (−7 B) | `pose_stub` is INERT on TR1 (`m08`); negligible |

QA55's ~1.8 KB was measured against the **ck1 STANDIN** builder (`ddm_ck1_build_composed_archive.py:143-152`,
**ZIP_STORED for everything**), which is **not the shipping builder**. ~915 B of container slack is
already banked by v4c/v4d; the −880 B is banked by the inline codec.

**Anchored magnitude** (baseline stated, per `m66`/qd1 law): −880 B = **25·880/37,545,489 = 5.860e-4 S**.
Against the live gap **0.6189279** (live best `S 0.7910689` → PR130 floor `0.172141`), that is
**0.0947 % of the gap** — consistent with the charter's 9,295 B-per-1 % exchange rate.

**⇒ QA55 should be marked "container half already banked by the real builder; pose-member win
already wired", NOT carried as ~1.8 KB open. No wiring work is owed. Head 2 requires no code.**

---

## §3 HEAD 3 — p2a T1 tier: 16 rows, all owned

Count: the memo's **16** is right (§4's never-named table holds **18**; §5's T0 tier claims 2 →
16). The charter's "15" is off by one; the only arithmetic reproducing 15 subtracts a *different*
3-row subset. List was **derived** from §4's table minus §5's T0 pair — p2a never enumerates T1
explicitly and no p2a-produced enumeration file exists in the searched scope.

**All 16 T1 ids are ABSENT from `.omx/state/canonical_task_status.jsonl` (423 rows / 149 distinct
ids).** Per `m89` that is a **missing JOIN, not evidence of absence** — every row was located by
CONTENT.

| verdict | rows |
|---|---|
| **ALREADY-CLOSED, drainable now (5)** | `#858` (receiver strict key-set + pinning test), `#450` (lens_engine, 5 modules landed), `#236` (dashboard + named tunnel all exist), `#860` (diagnosis half — gc16 states "6 of 25 gates RED, ~316 violations" verbatim), `#834` (reclaim↔spend contest settled: *"An exchange requires r < 0. The measured r is +0.212"*) |
| **REAL-OPEN, typed blocker + owner (6)** | `#877` (REPORT 2-dec censoring cannot resolve 0.0044), `#859` (SMEVR −2,781 B, blocked on a format change — re-confirmed today), `#198` (`fleet.local.toml` exists; **0 of 13,888 `.py` load it** — the `m56` unwired-but-built genus), `#844` (triality drift detector is **path-prefix** based; the syntactic predicate does not exist), `#840` (cf1's unswept 91% = **1,260** `codex_findings_*.md`), `#833` (weakest — see caveat) |
| **REAL-OPEN, parked with fire-conditions (4)** | `#716` (QD06), `#670` (QD07 — receives #860's 316 violations), `#706` (QE02/QE13, pre-arc), `#775` (behind sibling arm sb1) |
| **SUPERSEDED (1)** | `#556` (QF02 cluster, V9→TR1 vehicle pivot) |
| **UNDECIDABLE** | **0** |

**Structural finding, and the reason this tier drains cheaply: 9 of 16 (56 %) required no new work
at all — only a join.** `ddm_deferral_queue_ledger_20260729.md` **already holds** #716/#670/#556/#706
under QD06/QD07/QF02/QE02. p2a's never-named instrument could not see them because the ledger cites
rows **by content in a table cell**, not by `#N` in a commit message. **"Never-named" measures
commit messages; it does not measure the parking store.** That is a second instrument, owed.

Honest limits carried forward: `#833` is a **scoped negative** (did not find a capacity/floor-probe
degenerate-baseline control in 7,019 `.md` + 13,888 `.py`) — a different search key could flip it.
`#450`'s memo says "increment 1"; later increments may be owed. `#860` is **two things in one row**
(closed measurement + open remediation) — split rather than forced. Row subjects come from p2a's
transcription, so any scope lost there is inherited.

Instrument receipts: positive control exit 0 / negative control exit 1 before any adjudication;
**p2a §7 defect 3 reproduced live** (unquoted `--include=*.md` → `(eval):7: no matches found`, empty
output that reads as "no matches"); `git rev-list --count HEAD` = **13,934** (never bare `git log`,
which the `rtk` hook silently caps at 50).

---

## §4 HEAD 8 — VEHICLE-SCOPE-OWED: the first question answered for p1a items 2/23/24/25/29

The live composed vehicle's members (`ddm_v4d_build_composed_archive.py:45-46`): `manifest.json`,
`state/tokens.dr7t`, `state/renderer.sec`, `state/selector.sec`, `state/pose_stub.sec`,
`state/pose_warp.stp`. Live grammars: `PFS1WPD1` + `KL1PWF01`.

| item | first question: does this still apply? | MEASURED basis | disposition |
|---|---|---|---|
| **2** — LP1 G4 same-object context price for **the v15 stream** | **NO — off-path.** `v15` has **0 hits** in `ddm_v4d_build_composed_archive.py` and **0** in `inflate_runner_v4d.py`. The 89,161 B context-vs-explicit existence proof was measured on a stream the live vehicle does not carry. | grep, both live files | **RE-AIM** to `state/tokens.dr7t` (99.0 % of rate, `m08`) or retire. Not runnable as written. |
| **23** — rg3 `score_units_per_byte_status` | **GATE IS NOW OPEN, but the owed quantity is finer than assumed.** Its gate was "deferred until a byte-closed score measurement exists"; byte-closed rows now exist (dc1_fold 0.8983775 @ 360,309 B, 08-02). **However** the hb1 policy note requires *"measured coder bytes **per action**, never parameter counts"* — a **per-action** denominator, which a total archive size does not supply. | **37 rows**, all `OWED_NOT_ADMITTED`, in `ddm_hb1_hope_bn_capacity_20260727T0001Z/hope_per_stratum_capacity_table.json` | **RE-AIM**: gate open; owed = per-action rate, not total. 37-row debt sized. |
| **24** — c2 integer-plane-emitter rate/receiver custody | **Code EXISTS** (`train_c2_integer_plane_emitter_banded.py`, `integer_plane_banded_trainer.py`, `integer_plane_emitter_byte_close.py`, `c2_r1b4_curvelet_binding.py`) but the **family verdict is measured**: `SPEC_v10_integer_plane_vehicle_20260719.md:735` — *"RATE-DEAD at FAMILY scope: 5 codec families within 1.9× of the ~334 KB/pair floor (#541 n48)"*. | primary spec, family-scoped, n48 | **RETIRE-UNLESS-REOPENED.** The custody question is moot while the family verdict stands. Note `SPEC_v10:43` scopes it `verdict_scope: formulation` — so re-opening is a *formulation* question, not a new custody run. |
| **25** — r1b3 `E_n600` producer | **Premise FALSIFIED 07-20** — but explicitly *"an implementation-custody gap only, **not** a rank4/head/boundary-carrier family negative."* | `codex_premise_falsification_r1b3_receiver_pullback_20260720T185300Z_codex.md` | **FOLD INTO HEAD 1.** Same blocker `R1B3_P1_RECEIVER_COORDINATE_JACOBIAN_AND_REALIZED_SECANT_ABSENT`, same missing pullback as gc15's φ. **One measurement discharges both.** |
| **29** — blindspot op-routable-4 `$0` batch (5 sub-items) | **Citation unresolved.** p1a's line keys (`blindspot:38`, `:130/161/173`) do **not** resolve against the content at those lines in `feedback_pantheon_comprehensive_blindspot_pan1_20260725.md` (searched scope: that file + 9 other `blindspot`-named artifacts in `.omx/research/`). One sub-item (LP1 G4 price) **duplicates item 2**, re-aimed above. | grep, named scope stated | **DEFERRED-with-fire-condition**: resolve the citation key first; ~1/5 already subsumed by item 2. |

Also carried from the same blindspot register, still true today and directly relevant to every
advisory row this arm read: *"Advisory rows can steer a long campaign without a fresh same-byte
contest-axis calibration"* — `classification: UNCLEAR`, `ASSUMED_AWAITING_VERIFICATION`, rationale
*"DDM has no exact contest-CPU row; the latest bank is borrowed 2026-07-12 evidence."* The pointer
is still `0.1910828242` and still borrowed.

---

## §4.5 The one paid row (p1a #28, Modal T4 smoke): staging **NOT VERIFIED** — do not treat as ready

The charter required verifying staging and reporting readiness **without dispatching**. I did not
dispatch. **Readiness is NOT established, and the row should not be scheduled as a ready $0.20 fire.**

**The claim is a 2-hop citation with no terminal artifact.** `p1a:181` cites `ua2:419`; `ua2:419`
asserts the smoke is *"already **staged** by #214 deliverable (c)"* — and names **no path, no file,
no runner**. `"deliverable (c)"` appears in exactly **2 files** in `.omx/research/*.md` (p1a and
ua2), and **both are citing, neither defines**. `#214` is a bare task id, which per `m89` the repo
store largely does not hold.

**Searched for the artifact by content, denominator stated:** **10,235** tracked `.py` files under
`tools/`, `experiments/`, `src/`. Found **12** Modal runners with `gpu="T4"` — all of them
old-lineage harnesses (`modal_phase_a1_score_gradient_pr101`, `modal_t1_balle_endtoend`,
`modal_yousfi_r3_pr95_resume`, `modal_alpha_geo0_pose_regen`, `modal_hdm8_postfilter_sweep`, …).
**None is identifiable as the CUDA-ladder smoke, and none is aimed at the live own-vehicle archive.**
Per the no-old-lineage ban those are lessons-only harnesses, not a staged own-vehicle smoke.

**⇒ `did not find, in the named scope above,` a staged #214 T4 smoke.** That is a scoped negative,
not "it does not exist" — but the burden now sits with whoever asserts staging to name the file.
**Firing $0.20 against an unverified runner is how you buy a measurement of the wrong thing.**

**Better-verified sibling, and I checked it myself:** ua2's **M2** — *"`workflow_dispatch` `eval.yml`
on our own fork with `submission_name: baseline`, explicitly exempted from the uniqueness check at
`eval.yml:45`"*, which ua2 calls **"Highest value/cost ratio in this memo," $0, one dispatch**. I
**VERIFIED the exemption in the primary artifact**: `upstream/.github/workflows/eval.yml` exists and
its `check_name` step reads
`if [ "${{ inputs.submission_name }}" != "baseline" ] && git ls-tree … | grep -q .` — i.e. `baseline`
**does** bypass the uniqueness gate. M2 is real and ready.

**But M2 is out of scope for THIS arm and needs operator-GO**: dispatching `eval.yml` runs
`upstream/evaluate.py` over 600 samples — that **is** a scorer forward, which this charter forbids
at $0. Reported as READY, deliberately **not fired**.

---

## §5 Cross-findings

**→ whoever schedules gc15's arms D±.** Do not schedule them as `$0`-gated. The gate is a **run**
with an 11-day-old named blocker. The r1b3 producer is the prerequisite for **both** the φ preflight
and p1a item 25 — build it once.

**→ the follow-on / orphan detector owners (`fo1`, `p1a`, `p2a`).** Two distinct join defects, both
measured here, both with passing controls:
- **filename-join ≠ mechanism-join** (head 2: a codec re-implemented inline under a magic constant
  reads as "unwired" to a filename grep);
- **commit-message-join ≠ parking-store-join** (head 3: 4 of 16 rows were already parked with owners
  in a ledger that cites them by content in a table cell).
A passing positive+negative control validates the *instrument*, never the *join key*.

**→ the QA55 / v4c-grammar owner.** Mark QA55: container half already banked by the real builder
(`DEFLATE_MEMBERS`), pose-member win already wired (`encode_kl1_field`). Carrying it as ~1.8 KB open
overstates available rate by roughly the amount it claims.

**→ anyone quoting the ms4d bundle.** Its four per-block metric fields reduce to two numbers; and
its `pair_count: 600` / `full_n600` schedule entry do **not** mean 600 blocks are present — there
are **25, over 15 pairs**. State that denominator when citing it.

---

## §6 What I could NOT do / OWED

- **No φ value**, by design and stated above — the conversion available is in the wrong space and
  I refuse to publish it as φ. gc15's pre-registration remains undischarged in both directions.
- **p1a item 29's citation keys do not resolve**; I did not locate the 5-sub-item batch in the
  named scope and did not guess at a substitute.
- **The r1b3 reactivation criteria** in `r1b3_producers_20260720T185300Z.md` were not read in full;
  I confirmed the blocker id and the "custody gap, not family negative" scoping only.
- **`#833`'s negative is scoped**, not exhaustive (see §3).
- *(Was owed; **DRAINED in-arm** — see §2.1 below. Left here only to record that it was raised and
  then closed rather than deferred.)*

### §2.1 Two-implementation hazard: CHECKED and CLEARED (MEASURED, `$0`)

I flagged, then discharged, the obvious hazard behind head 2: if the inline `encode_kl1_field`
diverged from the standalone `experiments/ddm_kl1_pose_field_receiver.py`, "wired" would be true and
still dangerous — two encoders for one shipping member.

Extracted both function bodies by AST and ran them on identical `float16` input:

| shape | standalone `encode_pose_field` | v4d `encode_kl1_field` | v4c `encode_kl1_field` | bit-exact |
|---|---:|---:|---:|---|
| **(600, 6)** — the production pose-field shape | 6,183 B | 6,183 B | 6,183 B | **yes** |
| (600, 2) | 2,103 B | 2,103 B | 2,103 B | **yes** |
| (128, 6) | 1,366 B | 1,366 B | 1,366 B | **yes** |
| (7, 3) | 62 B | 62 B | 62 B | **yes** |

**VERDICT: byte-identical output. No divergence hazard.** The standalone file is **functionally
redundant** with the wired inline encoder.

*Scope, stated honestly:* this is 4 shapes on seeded random `float16` (`default_rng(1234)`), not a
proof over all inputs. It is strong because it is corroborated by reading both sources: identical
algorithm (column-major `.T` → hi/lo `uint8` byte planes → `brotli quality=11`), identical header
(`MAGIC + struct.pack("<HHI", n, d, len(payload))`). The *source text* differs only in function name,
magic variable name and comments — the emitted bytes do not.

---

## §7 NEXT-IF-RESUMED

1. **Build the r1b3 receiver-coordinate Jacobian + realized-secant producer.** It is the single
   prerequisite that unblocks *both* gc15's φ preflight (arms D± gate) and p1a item 25. Read
   `r1b3_producers_20260720T185300Z.md` for the pre-registered reactivation criteria first. This is
   a **run**, not a read — budget it as such.
2. **DONE in-arm (§2.1) — byte-identical, no hazard.** Residual: *decide* whether to delete
   `experiments/ddm_kl1_pose_field_receiver.py`. It is redundant as an encoder, but it also carries
   the `decode_pose_field` + `_verify` bit-exactness harness that the inline builders do **not**
   have. **Recommendation: keep the file, retitle it as the codec's reference/verification harness,
   and add a one-line pointer in it to the two live call sites** — that kills the "unwired orphan"
   ghost permanently without discarding the only round-trip verifier. Deleting it would remove
   verification, not duplication.
3. **Mark QA55** in `ddm_deferral_queue_ledger_20260729.md` per §2 (container half banked,
   pose-member wired). Do not carry ~1.8 KB as open rate.
4. **Drain head 3's 5 already-closed rows** (`#858 #450 #236 #860`-diagnosis `#834`) and route
   `#860`'s ~316-violation remediation to `#670`/QD07 where it already has a home.
5. **Build the parking-store join.** The never-named instrument reads commit messages; the
   canonical deferral queue ledger cites rows by content in table cells. 4 of 16 rows this window
   were falsely "never-named" for that reason alone.
6. **Re-aim item 2 to `state/tokens.dr7t`** (99.0 % of rate) or retire it — the v15 stream it was
   written for is not on the live vehicle.
7. **Item 23**: the gate is open but wants **per-action** coder bytes; 37 `OWED_NOT_ADMITTED` rows
   are sized and waiting in the hb1 capacity table.

**Pointer honesty:** `0.1910828242 [contest-CPU]` **UNMOVED**. Zero scorer forwards, zero training
launches, zero dispatch, zero paid spend. `score_claim=false`, `promotion_eligible=false`.
