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

**The one result that could move a score, stated with its blockers first:** `#826` **inverts back to a
win at $0** — gr1's token codes are **bit-identical** to the live best's (**0 of 1,843,200** differ),
so its +5,183 B is **coder generation** (`DR7T`→`IX2TOK01`), **not content**. Repacked, it is
**352,048 B = 1,969 B under** the break-even target and **1,757 B smaller than the live best itself**
(ΔS(seg+rate) **−0.0013110**; **−0.0011699 on the rate leg alone**, which does not depend on the seg
question). **Two typed blockers stand between this and any score claim: receiver acceptance is
UNPROVEN (no inflate run), and the seg leg is CROSS-INSTRUMENT (1.4 ppm, needs one exact eval).**
`score_claim=false`. §4.8.

Four further results are load-bearing, and three of them **delete** work rather than add it:

1. **HEAD 1 — φ is not readable from the ms4d bundle at any cost.** gc15's "`$0` but needs the
   composite-R adjoint from the ms4d bundle — **a read, not a run**" is **FALSIFIED at the artifact
   level**. The artifact is complete and honest; it is the **wrong operator**. φ is re-typed from
   *read* to *run*, and it **inherits a named blocker recorded 11 days before gc15 asked for it**.
2. **HEAD 2 — the kl1 −880 B codec is NOT "built-unwired".** It is wired end-to-end — encoder,
   live call sites, and receiver — and has been **on the shipping path since v4c**. p1a's P0
   grade-5 orphan claim is **REFUTED by mechanism-join**. QA55's container half is **already
   banked by the real builder**.
3. **HEAD 3 — 9 of 16 T1 rows (56 %) needed only a join, not work.** The never-named instrument
   measures commit messages; it cannot see the parking store.
4. **HEAD 4 — two of oh1's "ORPHANED" verdicts are FALSIFIED,** because `ddm_oh1` **never consulted
   `.omx/state/deferral_ledger.md`** (0 hits in its own memo) — the one store whose header says *"no
   deferral exists outside this file."* An orphan-detector made a negative-existence claim without
   scanning the parking store. `deferral_ledger::D1` was `ARMED` against a **trigger the TR1 pivot
   superseded** — an armed row that can never fire, which reads as parked and is strictly worse than
   an open one. **Landed + resolved** (`819e7661f7`).

**The through-line across all eight heads: every headline defect this arm found is one join key
away.** filename-vs-mechanism (head 2), commit-message-vs-parking-store (heads 3, 4),
line-scope-vs-intra-document-supersession (head 4), scorer-feature-space-vs-parameter-space (head 1),
coder-generation-vs-content (head 6), position-vs-content (head 7). **Not one is a hard problem; every
one produced a confidently wrong published claim.**

---

## §0.5 PER-HEAD DISPOSITION — every head exits OWNED

| # | head | fired? | disposition | owner / fire-condition |
|---|---|---|---|---|
| **1** | p1a #1 φ from the composite-R adjoint | **FIRED** | **DONE-WITH-RECEIPT — premise FALSIFIED.** Not readable at any cost; **re-typed read → run**; inherits `R1B3_…_ABSENT`. No φ published (refused: wrong space). gc15 pre-registration undischarged **both** ways; arms D± **GATED**. | Whoever builds the r1b3 producer. Fire: that producer lands (also discharges p1a #25). |
| **2** | p1a kl1 −880 B codec → QA55 | **FIRED** | **DONE-WITH-RECEIPT — claim REFUTED.** Codec is **wired** (encoder v4d:257-258 → `pose_warp.stp`; receiver parses `KL1_MAGIC`), shipping since v4c; byte-identical to the standalone on 4 shapes. **No wiring owed.** QA55 container half **already banked**. | QA55 owner: re-mark the ledger row. Nothing to build. |
| **3** | p2a T1 grep tier | **FIRED** | **DONE — 16/16 owned**, 0 UNDECIDABLE. 5 closed · 6 typed-blocker · 4 parked · 1 superseded. **1 of my republished claims corrected (§3.1).** | Per-row owners in §3; `#860`'s remediation → `#670`/QD07. |
| **4** | oh1 rows 3-6 re-aims | **FIRED** | **DONE — 2 of 4 ORPHANED verdicts FALSIFIED.** Row 4 landed **and** resolved (`819e7661f7`); row 5 CLOSED-MEASURED; row 3 re-aimed to a join; row 6 genuinely open, form re-aimed. | Row 3 → **codex/root**; row 6 → **MAIN**. |
| **5** | qd1 hygiene marks | **FIRED (partial)** | **D52a LANDED** (`819e7661f7`). **`#874→#885` fold SPECIFIED** (append to existing law, target outside repo frontier). **`#904` SPECIFIED but BLOCKED.** qd1 §C correction specified. | `#904`: **parent**, fire = `ddm_op3` commits its 6 pending rows. |
| **6** | op3 `#826` re-encode within 212 B | **FIRED** | **DONE-WITH-RECEIPT — spec SATISFIED at $0 with 1,969 B margin; row INVERTS to a win.** Budget **exactly invariant** under the baseline move. **2 typed blockers** before any score claim. | 6.4 receiver acceptance → **MAIN/v4d owner**; 6.5 exact seg eval → **scorer-slot holder**. |
| **7** | mt1 ab_trace 5-LOC + `--mode photo` | **PARTIAL** | **Diff WRITTEN + anchor-verified (not applied — needs 2 review passes).** Re-run **BLOCKED ×2**: PoseNet forwards (1,507 s MEASURED) **and** it is a **silent no-op** as prescribed (dead `--resume` flag, `args.resume` referenced 0×). %-of-gap re-anchored. | Diff → **parent** (review passes); re-run → **scorer-slot holder**, after the 1-LOC dead-flag fix. |
| **8** | VEHICLE-SCOPE-OWED (p1a 2/23/24/25/29) | **FIRED** | **DONE — all 5 answered.** #2 off-path (re-aim to `tokens.dr7t`) · #23 gate open, owed quantity re-typed to per-action · #24 retire-unless-reopened · #25 **folds into head 1** · #29 citation unresolved. | §4 table. |
| **+** | p1a #28 Modal T4 (the one paid row) | **NOT FIRED — correctly** | **STAGING NOT VERIFIED.** 2-hop citation, no terminal artifact in 10,235 `.py`. **Not ready; not dispatched.** Sibling ua2-M2 verified ready but is a scorer forward ⇒ operator-GO. | Whoever asserts staging must name the file. |

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

### §1.1 Downstream consequence — a live consumer of these fields, MEASURED

The rank-1 finding is not academic. `src/tac/ddm_lambda_ranker.py` consumes this exact artifact:

**(a) `adjoint_norm` is computed and never read — dead compute.** At `:593` it is initialised and at
`:601` accumulated (`grouped[pair_id]["adjoint_norm"] += norm(adjoint)`). **Those are the only 2
occurrences of the key in the file** (full-file count, not truncated). The 3 hits elsewhere in the
tracked tree are a **different identifier** (`anchor_adjoint_norm_upper` in
`src/tac/scorer_surrogate/costate_trust_region.py`) — a substring false-positive, not a consumer.
Per `m68` (#417 unconsumed = INERT) this is a small but genuine orphan.

Note it could never have been an independent feature anyway: per block `adjoint ≡ f` bitwise and
`trace(gram) = c·‖f‖²`, so `fisher_trace` and `adjoint_norm` are **the same vector measured twice**.

**(b) A 95 % confidence interval whose width is scale-convention-dependent by ~2–3 orders of
magnitude, stamped as measured.** At `:1454-1462`:

```python
fisher_trace = row["direct_fisher_trace"]
if fisher_trace is not None and float(fisher_trace) > 0.0:
    nominal_standard_error = residual_sigma / math.sqrt(float(fisher_trace))
    interval = [prediction - 1.96*standard_error, prediction + 1.96*standard_error]
    precision_status = "DERIVED_FROM_MEASURED_DIRECT_MS4D_FISHER"
```

`SE = σ/√I` is the ordinary Cramér–Rao form and is **not wrong for a scalar target** — I am *not*
claiming a bug. What I measured is that the inputs make the interval far more convention-dependent
than the stamp suggests:

| MEASURED | value | consequence for the CI |
|---|---|---|
| `gram` rank | **exactly 1** (25/25) | `trace(gram)` is its single nonzero eigenvalue — one-dimensional information, not 4-D |
| scale factor `c` in `gram = c·f fᵀ` | **0.1002 … 20,836.85** = **2.08e5× spread** | SE ∝ 1/√trace ⇒ **~456× spread in interval width** from the scale convention alone |
| Fisher-vs-Euclidean relation | cosine **±1** in 25/25; `‖f‖/‖e‖` spans **5.98e6×** | the two metrics differ by a *pure rescale* — feeding the other one moves the CI by orders of magnitude while changing no direction |
| coverage | **15 of 600 pairs** have a block at all | `direct_fisher_trace is None` for the rest; the branch silently falls through |

⇒ **The label `DERIVED_FROM_MEASURED_DIRECT_MS4D_FISHER` reads as convention-free and measured; the
width is neither.** Recommendation: record which metric convention produced `trace`, and report the
15/600 coverage alongside any interval. *Caveat: I verified the code path, not that this ranker is
currently invoked by a live consumer — I did not trace its callers.*

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
| **REAL-OPEN, typed blocker + owner (6)** | `#877` (REPORT 2-dec censoring cannot resolve 0.0044), `#859` (SMEVR −2,781 B, blocked on a format change — re-confirmed today), `#198` (**see correction §3.1 — narrower than first reported**), `#844` (triality drift detector is **path-prefix** based; the syntactic predicate does not exist), `#840` (cf1's unswept 91% = **1,260** `codex_findings_*.md`), `#833` (weakest — see caveat) |
| **REAL-OPEN, parked with fire-conditions (4)** | `#716` (QD06), `#670` (QD07 — receives #860's 316 violations), `#706` (QE02/QE13, pre-arc), `#775` (behind sibling arm sb1) |
| **SUPERSEDED (1)** | `#556` (QF02 cluster, V9→TR1 vehicle pivot) |
| **UNDECIDABLE** | **0** |

**Structural finding, and the reason this tier drains cheaply: 9 of 16 (56 %) required no new work
at all — only a join.** `ddm_deferral_queue_ledger_20260729.md` **already holds** #716/#670/#556/#706
under QD06/QD07/QF02/QE02. p2a's never-named instrument could not see them because the ledger cites
rows **by content in a table cell**, not by `#N` in a commit message. **"Never-named" measures
commit messages; it does not measure the parking store.** That is a second instrument, owed.

### §3.2 Spot-verification of the other two negative-existence claims — both HOLD

Having caught one false negative (§3.1), I re-derived the other two from primary artifacts rather
than trusting the pattern:

- **`#844` — HOLDS, and it is a *well-formed* scoped negative.** `tools/triality_drift_detector.py`
  is **1,385 lines**; `syntactic` = **0** hits, `predicate` = **0** hits; detection is demonstrably
  path-prefix based — `:176` `any((f or "").startswith(TRIALITY_PREFIXES) …)` and `:179-181`
  `touched_dsl()` → `startswith("src/tac/witness_dsl/")`. The scope here is **one named file**, so
  the negative is bounded by construction — which is exactly why it survived and `#198` did not.
- **`#840` — HOLDS, with a denominator refinement.** `.omx/research/codex_findings_*.md` = **1,260**
  (exactly as reported). But **repo-wide tracked = 1,313**. If cf1's sweep is scoped to
  `.omx/research/` the figure is right; if it is meant repo-wide the unswept set is **1,313**.
  Stated so the successor picks the scope deliberately rather than inheriting it.

Honest limits carried forward: `#833` is a **scoped negative** (did not find a capacity/floor-probe
degenerate-baseline control in 7,019 `.md` + 13,888 `.py`) — a different search key could flip it.
`#450`'s memo says "increment 1"; later increments may be owed. `#860` is **two things in one row**
(closed measurement + open remediation) — split rather than forced. Row subjects come from p2a's
transcription, so any scope lost there is inherited.

### §3.1 CORRECTION — `#198`: I nearly published a false negative-existence claim, and caught it

**As first reported (by the head-3 sub-arm, and republished by me):** *"`fleet.local.toml` exists but
**0 of 13,888 `.py`** load it — the `m56` unwired-but-built genus."*

**MEASURED, on spot-verification: that is WRONG. A loader exists.**
`scripts/lane_watchdog.py:107` — `def load_fleet(path: Path = FLEET_TOML) -> dict[str, dict]:`,
with `FLEET_TOML = REPO / "fleet.local.toml"` at `:57`, documented graceful degradation when the
file is absent (`:54-56`), and a `--list-fleet` CLI flag at `:293`. Both `fleet.local.toml` (1,378 B)
and `fleet.example.toml` (1,361 B) exist.

**Root cause — the vacuity genus (`m50`), third instance in this arm.** The sub-arm's stated
denominator was *"`src/` + `tools/` + `experiments/` (excl. `results/`) `*.py` = 13,888"*. **`scripts/`
was never in that scope**, and the loader lives in `scripts/`. The sub-arm **declared its denominator
honestly** — and then phrased the result as *"0 of 13,888 `.py` load it"*, which **reads as a global
negative**. I republished it in that form. Per `m53` (negative-existence claims are the #1
false-claim class), the honest phrasing is *"did not find in `src/`+`tools/`+`experiments/`"* — and
that phrasing would have made the missing `scripts/` obvious immediately.

**The row's surviving, narrower debt.** `#198`'s subject is a **canonical** fleet-config loader **plus
preflight self-protect**. What exists is **one script's private loader**, not a canonical shared
surface, and no preflight gate. Separately, `joint_descent_p0_launch_prep_20260708.md:106` records
`scripts/bat00.py` still taking `BAT00_IP`/`BAT00_USER` from operator env rather than the fleet file
— so a second consumer does bypass it. **REAL-OPEN stands; "no consumer exists" does not.**

**Carried as a cross-finding:** a denominator stated in a methods line does not travel with the
finding when the finding is quoted. **Put the scope inside the claim sentence, not beside it.**

---

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

## §4.6 HEAD 4 — oh1 rows 3-6: two "ORPHANED" verdicts FALSIFIED

**The single biggest finding of this head: `ddm_oh1` never consulted `.omx/state/deferral_ledger.md`.**
MEASURED: `grep -c "deferral_ledger"` on oh1's memo = **0**, and its STORES CONSULTED block does not
list it. That file's own header states the governing rule — *"no deferral exists outside this file +
a task with a named trigger."* **2 of oh1's 4 "real owed item" rows are already correctly parked or
closed there.** oh1's *"6/9 point at a real owed item (67%)"* is an over-count: a negative-existence
claim made without scanning the one store designed to hold the join — `m53`'s #1 false-claim class,
committed by an instrument built to catch orphans.

**A fourth false-ORPHANED class, beyond oh1's own three: intra-document supersession.** Row 5's cited
line sits **37 lines above** `owed16_bounded_ab_and_drystart_20260710.md:178` — verbatim
`## MEASURED VERDICT (2026-07-10, appended — supersedes OWED-BLOCKED above)`. A line-scoped extractor
cannot see that a block it quotes has been superseded *in the same file*.

| oh1 row | verdict | disposition / owner |
|---|---|---|
| **3** — g111 macro-release → `v6` | **Not orphan, not ADVANCED — BLOCKED.** `pact-g111-first-real-n600-capstone-run` = `blocked`, `launch_authorized:false`, **5 named blockers**. oh1 inferred "likely ADVANCED" from commit `ead282f6f0` — verified: that commit moved **G121 harvest machinery only** (4 files); the gating run `v6` never fired. `v6` is a **triple slug collision**. | **RE-AIM to a JOIN**, no new measurement. Owner **codex/root**. Fire: V9/taskspace resumes live AND the 5 blockers close. |
| **4** — gpu-verdict Measurement 2 | **ORPHANED FALSIFIED** — it is `deferral_ledger::D1`, **ARMED**. Real defect = **STALE TRIGGER**. **LANDED + then RESOLVED — see below.** | **DONE-WITH-RECEIPT** (commit `819e7661f7`) |
| **5** — owed-16 A/B | **ORPHANED FALSIFIED — CLOSED-MEASURED.** Both arms ran; realized directional contribution ≈ **ZERO** (\|Δ\|≤1.4 % at every matched cell); residual cell also measured (`+3.2e-05`, worse). Artifacts on disk. The −48 % is a **PROXY**, non-transferable — CLAUDE.md already forbids routing on it. | Retire. Residual (*from-scratch* A/B) is aimed at a DOF the shipped TR1 vehicle appears not to have. |
| **6** — ORDER 2c council adjudication | **GENUINELY OPEN** (9 d). `ORDER 2c` in **2 of 7,019** memos; **0 hits** in the deferral ledger. But the *form* is the `m39` anti-pattern (13 convocations → 0 pointer). Consumer join MEASURED: **2 of 8 lineages live** (C1 via waterfill, C8 via pc2). | **RE-AIM the form**: 8 pre-registered falsifiers priced against 1 % of gap = **9,295 B / 7,301 flips**, not a 19th convocation. Owner **MAIN** (`m45`). |

### D1 — flagged, then RESOLVED in-arm by the $0 read it named (commit `819e7661f7`)

D1 was `ARMED (re-pointed)` against trigger *"#385 chosen-chain (v7.5.2\|v8) PRE-LAUNCH"* — **a chain
the TR1 pivot superseded**. An armed row whose trigger can never fire is a **silent orphan** (`m37`),
strictly worse than an open one because it reads as parked.

I then ran the $0 code read it needed. **MEASURED: `verdict_device` / `--verdict-device` occurs in
exactly 10 of 10,683 tracked `.py`, ALL in the levelset/witness family** (the two trainers,
`witness_autoconfig.py`, `witness_control/gpu_verdict.py`, `witness_dsl/{curriculum_dsl,typed_config}.py`,
+4 tests). **All 12 TR1-line builders/receivers/probes** (`ddm_v4d_*`, `inflate_runner_v4{b,c,d}*`,
`ddm_pu2_*`, `ddm_dc1_*`, `ddm_gr1_*`) score **0 hits**. ⇒ **the lever is stranded on the ancestor.**
**DISPOSITION: RETIRE-WITH-REACTIVATION** (fire: a levelset-family trainer re-enters the live line).
The separate *throughput-on-the-contended-evaluator-slot* need is real but needs a **fresh TR1-native
row** — it is not this one.

---

## §4.7 HEAD 5 — qd1 hygiene: one landed, two specified, one blocked

**(i) `D52a` — LANDED (commit `819e7661f7`).** qd1 **completed it itself** (`canonical_task_status.jsonl`
at HEAD: `completed`, `2026-08-03T11:14:39Z`, `commit_shas:["0bfeb8733b"]`, `test_status: green`) and
then listed it in its own NEXT-IF-RESUMED as still-owed — **self-stale**. But the file the row names
as canonical, `.omx/state/deferral_ledger.md`, still said OPEN at **two** sites. Both now marked
COMPLETED. Safety verified before mutating: the file is **not** registered in
`artifact_kind_registry.yaml`, and in-cell status mutation is **its own convention** (D3/D4/D20 carry
`**CLOSED-CONFIRMED 2026-07-09**` in-cell).

**(ii) `#874 → #885` fold — SPECIFIED, unblocked, not landed here.** The fold **target already
exists** (`vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801.md:174-200`), so this
is a *local append to an existing law*, not new machinery — exactly what `m54` demands. The
generalization: **a CAP and a TRUNCATION are ONE class — the instrument cannot emit its own limit.**
A silent cap (`rtk`'s `-n 50`; a subset default; a timeout; a `head`; a governor ceiling) and a scan
reporting completeness it never achieved are the same defect in two costumes. The obligation sits on
the **producer**: emit `(n_returned, limit_applied, was_truncated)` or emit nothing. **Do not build a
second cap detector.** *(Target lives outside this repo's mutation frontier — left for the parent
with exact insertion text in the sub-arm receipt.)*

**(iii) `#904` re-scope — SPECIFIED, BLOCKED on a live sibling.** Both halves of qd1's "diagnosis half
wrong" VERIFIED at source: **Class A** — gd5's F1 was *already repo-wide* (**1,229 of 3,251** modules,
38 %, REFUTED as too loud to be a queue), so **scope was not the cause**; **Class B** —
`src/tac/v9_provenance_gates.py:922` `_trainer_consumers` ast-parses **exactly one file** (single
caller `:1051`) and counts **any** `args.<dest>` Load *or* `getattr(args,"<dest>",…)` as proof, so
`_x = float(getattr(args,"x",2.0))` reads as consumed — **scope AND predicate are both wrong**.
**TYPED BLOCKER:** `.omx/state/canonical_task_status.jsonl` carries **6 uncommitted rows, all
`event_actor: ddm_op3`** — a live sibling holds that file. **I did not touch it.** Owner: parent;
fire-condition: `ddm_op3` commits.

**(iv) Corrections owed to qd1 §C.** Its claim *"the entire `#874–#911` band is ABSENT from the repo
store"* is **over-scoped**: `#882` and `#909` are both present and `completed`, both written
`2026-08-03T04:37:56Z` by `ddm_pj2` — i.e. **before qd1's own run**. Max numeric id is **909**, not
871. **The directional finding survives intact** (`#874`/`#885`/`#904` *are* genuinely absent; arms
see only the repo store; cite CONTENT not ids) — only the universal quantifier fails.

---

## §4.8 HEAD 6 — op3's `#826`: the 212 B spec is SATISFIED at $0, and the row INVERTS back to a win

**⚠ NOT A SCORE CLAIM. `[macOS-CPU advisory]`, `score_claim=false`, pointer UNMOVED.** This is a
static container/coder byte measurement with **two typed blockers** standing between it and any
score. Stated up front because the number is attractive and that is exactly when NO-FAKE binds.

**Control first.** Re-running op3's own registered equation reproduced it exactly:
`W = 1.2731082153320312` B/flip (`EXACT_INVARIANT`), ΔS `+0.0034632`, F 166.5, 32.52 B/flip, **25.5×W**,
budget **211.9 B ≈ op3's 212**.

**Re-anchored to the live best — the budget is EXACTLY INVARIANT.**

| | at cx1 (op3's basis) | at pu2 (LIVE best) | moved |
|---|---:|---:|---:|
| break-even budget | 211.9 B | **211.9 B** | **±0.000 B** |
| target archive size | 354,020 B | 354,017 B | −3 B |
| ΔS(seg+rate) | +0.0034632 | +0.0034652 | +2.0e-06 |

DERIVED: the budget is `F × W`; `W` is exactly invariant and `F` is unchanged because the base's
d_seg is **bit-identical** across the whole chain (the entire cx1→pu2 move was pose + 3 bytes).

**Why the spec is satisfiable: the excess is FORMAT, not CONTENT.** MEASURED —
**gr1's token codes are BIT-IDENTICAL to the live best's: 0 of 1,843,200 entries differ** (both
`(600,24,32,4)` uint8). Re-encoding gr1's decoded codes with `ix2.encode_token_frame` yields
**exactly 341,295 B = pu2's bulk**, lossless round-trip verified. **The +5,183 B is coder generation
(`DR7T` → `IX2TOK01`), not information.** gr1 also ships **6 ZIP members** (686 B overhead + 1,234 B
JSON manifest) vs the live best's **1 member** `0.bin` (108 B overhead).

Repack ladder, every rung verified section-bit-identical by `parse_payload` byte-compare:

| rung | archive | vs 354,017 B target |
|---|---:|---:|
| A shipped-equivalent (DR7T, no config) | 357,161 | +3,144 |
| B + ix2 token recode | 351,978 | **−2,039** |
| **C + pu2-shaped config section** | **352,048** | **−1,969** |
| D + pose sections merged as pu2 does | 352,021 | −1,996 |

At rung C the repack is **1,757 B smaller than the live best itself** ⇒ ΔS(seg+rate) vs pu2 =
**−0.0013110**; and **worst case, assuming gr1's entire seg advantage is instrument artifact,
−0.0011699 on the rate leg alone**. **The rate leg does not depend on the seg question.**

**BLOCKER 6.4 — receiver acceptance UNPROVEN.** `parse_payload` round-trip proves the *container* is
faithful; it does **not** prove the modern `inflate_runner.py` consumes gr1-generation sections
(gr1's `renderer.sec` 3,341 B vs pu2's 3,266 B; pose split `pose_stub`83+`pose_warp`6,864 vs a single
8,751 B section). *Owner: MAIN / v4d build owner. Fire: before any gate or eval on a repacked gr1.*

**BLOCKER 6.5 — the seg leg is CROSS-INSTRUMENT.** gr1's `d_seg 0.004310379` came from gr1's own
realized harness, **not** `upstream/evaluate.py`; pu2's `0.00431179` did. The whole advantage is
**166 flips in 117,964,800 cells = 1.4 ppm** — well inside cross-instrument drift. One exact eval
settles it. *Owner: n600 scorer-slot holder.*

**Spillover (DERIVED, one decode from MEASURED).** `gd3_CONTROL_identity_rebuild.zip` and
`v4d_composed_pw1_archive.zip` also ship `state/tokens.dr7t` at **exactly 346,478 B**. If their codes
are likewise identical, **every banked DR7T-generation archive** (dc1_fold 360,309 · ms8 360,374 ·
pj2 360,406 · pb2 360,339 · mq1 360,702) is carrying **~5,183 B of dead format**. Proven for gr1 only.

---

## §4.9 HEAD 7 — mt1's `ab_trace` + `--mode photo`

**mt1's line numbers have drifted — the exact failure mt1 itself warned about** (*"a triage keyed to
a position in a live tree mis-attributes without saying so"*): `ab_trace` is at
`experiments/ddm_v4d_resolve.py:385` (mt1 said :372); the v4c mirror at `ddm_v4c_resolve.py:926-930`
(mt1 said :818-822); `--mode photo` is at `ddm_v4c_resolve.py:1036`. **The two halves of the ask live
in different files** — `photo` is a v4c mode; v4d's modes are `qa66/refine/resummarize/qa72a`.

**mt1's claim REPRODUCED:** `ab_trace` occurs **exactly once** in the 44,016 B v4d file — bound at
`:385`, never read. And `{'ABSENT': 600}` reproduced exactly: 600 rows / 600 distinct pairs, with
`ab_stop`/`ab_start`/`ab_starts_tried`/`ab_relins`/`ab_damp_used`/`obj_traj` absent on **600 of 600**.

**The 5-LOC diff is written and diff-ready (NOT applied — `.py` edits need 2 recorded review passes;
handed to the parent).** It inserts `ab_stop`/`ab_relins`/`ab_relins_bound`/`ab_damp_used`/
`ab_obj_traj` into the emitted row at `ddm_v4d_resolve.py:402-403` (anchor verified unique). All five
read state **already computed** — score-neutral, zero extra scorer evaluations, byte-identical shipped
output. *Do not copy v4c's five verbatim: `start`/`starts_tried` are added by v4c at `:900` and are
not in the returned trace.*

**The `--mode photo` re-run is BLOCKED for TWO independent reasons — and the second one mt1 missed:**
1. **Not $0.** `run_photo`'s `pose6()` is a live **PoseNet forward** (`ddm_v4c_resolve.py:876-878`)
   inside `ab_damped_gn` (4 relins × 4 damp levels) + 2 rung-A evals per pair × 600. **MEASURED cost
   of the original run: 1,507 s wall at n600.** Categorically forbidden by this charter.
2. **As prescribed it is a SILENT NO-OP.** `run_photo` loads the existing 600-row cache
   **unconditionally** (`:856 cache = _load_jl(jl)`) and `continue`s on `if pidx in cache`.
   `--resume/--no-resume` is declared at `:1056` but **`args.resume` is referenced 0 times** in either
   file (denominator: the full 52,264 B v4c + 44,016 B v4d). Run as written it skips all 600 pairs,
   writes zero rows, **exits 0**, and leaves the census `{'ABSENT': 600}` — a **dead flag**
   (CLAUDE.md "Forbidden CLI flag inventions") wearing the vacuity genus (**exit-0 == "done" ==
   "nothing happened"**). Cure specified, 1 LOC: `cache = _load_jl(jl) if args.resume else {}`.

**Re-anchoring mt1's %-of-gap (its token baseline is stale by 5,183 B — the same DR7T generation as
head 6):** net saving 258,635 → **253,452 B**; ΔS_rate −0.172214 → **−0.168763**; gap denominator
0.7262356 → **0.6189276**; **% of gap 23.71 % → 27.27 %**. **The percentage RISES while the absolute
shrinks** — a second worked instance of op3's law that a %-of-gap claim is *under*-stated by its own
drift. Honest bound: the ds=32 arms also ship DR7T, so if they gain proportionally the correction
shrinks — **correction ∈ [0, +0.003451]**. mt1's §5 #1 remains the largest measured seg+rate row
either way.

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

**→ everyone writing scoped negatives (this is the arm's most transferable finding).** A denominator
stated in a *methods* line **does not travel with the claim when the claim is quoted**. `#198` was
reported as *"0 of 13,888 `.py` load it"* by a sub-arm that had honestly declared its scope excluded
`scripts/` — where the loader actually lives (§3.1). The honest form puts the scope **inside the
claim sentence**: *"did not find in `src/`+`tools/`+`experiments/`"*. Three instrument failures in
this arm (my `break`-truncated sample, my alternation-grep false lead, this) are **all the same
genus**: a result that is true of the instrument's actual scope, read as true of the world.

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

0. **HIGHEST VALUE — close `#826`'s two blockers (§4.8).** The byte spec is already SATISFIED with
   **1,969 B of margin**, at $0, and the rate leg (**−0.0011699**) does **not** depend on the seg
   question. Two steps, in order: **(a)** run `inflate_runner` on the rung-C repack to prove receiver
   acceptance (an inflate run, **not** a scorer forward — cheap, and it is the gating unknown);
   **(b)** one exact `upstream/evaluate.py` n600 row to settle the 1.4 ppm cross-instrument seg leg.
   Until (a) passes, this is a container measurement and **nothing more** — do not quote it as a score.
0b. **Then test the spillover, one decode (~2 min, $0).** `gd3_CONTROL_identity_rebuild.zip` and
   `v4d_composed_pw1_archive.zip` ship `state/tokens.dr7t` at **exactly 346,478 B**. If their codes
   are identical too, **every banked DR7T archive** (dc1_fold · ms8 · pj2 · pb2 · mq1) carries
   ~5,183 B of dead format — a fleet-wide rate win from a format swap, with zero distortion risk.
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
8. **Land the mt1 5-LOC `ab_trace` diff** (§4.9) — written, anchor verified unique, score-neutral,
   byte-identical output. Needs 2 recorded review passes (`tools/review_tracker.py`). **And fix the
   1-LOC dead flag first** (`cache = _load_jl(jl) if args.resume else {}`), or the `--mode photo`
   re-run silently skips all 600 pairs and exits 0.
9. **Land the `#874 → #885` fold** as an append to the existing vacuity law (target and exact text in
   §4.7) — **a cap and a truncation are one class**. Do **not** build a second cap detector.
10. **`#904` re-scope** into its two measured classes — **blocked until `ddm_op3` commits** its 6
   pending rows in `canonical_task_status.jsonl`. Do not race it.
11. **Correct qd1 §C's band claim** (`#882`/`#909` are present, max id 909 not 871); its directional
   finding survives and should be preserved.

**Pointer honesty:** `0.1910828242 [contest-CPU]` **UNMOVED**. Zero scorer forwards, zero training
launches, zero dispatch, zero paid spend. `score_claim=false`, `promotion_eligible=false`.

---

# §8 APPEND (2026-08-03, MAIN-authorized follow-up) — custody persisted, and **`#826` is REFUTED**

*Supersedes nothing. §4.8's byte arithmetic is REPRODUCED EXACTLY; its **conclusion** is overturned by
the two checks §4.8 never made — receiver acceptance and pose-content normalisation.*

## §8.0 Answer first

**The coordinator's catch was right and it is the reason the error was caught.** My five commits
persisted **prose only** — the rung archives existed nowhere on disk. Building them for real, and
then running the receiver against them, **refuted the row.**

**`#826` does not invert. It is REFUTED on two independent grounds, either sufficient alone:**

1. **Every rung is REJECTED by the live-best receiver** (positive control ACCEPTED, so the harness is
   trusted). gr1's **original** archive is rejected too — on *both* receiver paths.
2. **Pose-normalised, the "win" evaporates.** gr1 carries **1,804 B less pose content** than the live
   best. That exceeds the entire apparent −1,757 B, leaving rung C at **+47 B on the wrong side**.

**Routing consequence — act on this:** the `#826` inversion should **NOT** enter gc16's P2 rate lane.
It is not a rate win awaiting an eval; it is an apples-to-apples artefact. **Cost to have learned this:
$0.** Cost of *not* having built the bytes: a wrong row would have consumed the scorer slot the moment
`ph5o` released it.

## §8.1 PERSIST — custody landed (the gap that was left)

Builder: **`experiments/ddm_iv1_repack_rungs.py`** (ruff-clean; 2 recorded review passes; 9/9 entities).
Cold store: **`/Volumes/VertigoDataTier/pact/ddm_iv1_20260803/`** — 4 rung archives + a manifest
carrying bytes, sha256, full source shas, and the exact rebuild command (certify-or-block satisfied).

**Byte-verification against §4.8: all four rungs reproduce the memo EXACTLY.**

| rung | rebuilt | memo §4.8 | delta | vs live best | receiver |
|---|---:|---:|---:|---:|---|
| A shipped-equivalent (DR7T) | 357,161 | 357,161 | **+0** | +3,356 | REJECTED |
| B + ix2 token recode | 351,978 | 351,978 | **+0** | −1,827 | REJECTED |
| C + pu2-shaped config | 352,048 | 352,048 | **+0** | −1,757 | REJECTED |
| D + pose merged | 352,021 | 352,021 | **+0** | −1,784 | REJECTED |

The premise re-derived independently and is **stronger than §4.8 stated**: the ix2 re-encode of gr1's
decoded codes is not merely the same *size* as the live best's bulk — it is **byte-identical** to it.
Codes differ in **0 of 1,843,200** entries; round-trip lossless; format excess exactly **5,183 B**.

## §8.2 RECEIVER ACCEPTANCE (blocker 6.4) — typed verdicts: **ALL REJECTED**

Harness: the live-best submission's **own** `inflate_runner.py`. **Positive control ACCEPTED
(`n_pairs=600`)** — the test can return both answers.

| case | verdict | exact error |
|---|---|---|
| **ctl — pu2 live best** | **ACCEPTED_CONSTRUCT** | `n_pairs=600` |
| rung A | REJECTED | `IX2ContainerError: unknown table format 78` |
| rung B | REJECTED | `IX2ContainerError: unknown table format 78` |
| rung C | REJECTED | `SystemExit: ix2 container holds 5 sections, expected 4` |
| rung D | REJECTED | `IX2ContainerError: renderer frame magic differs` |
| **gr1 ORIGINAL 6-member** | REJECTED | `SystemExit: v4d receiver requires frame0_policy=warp_two_plane_static_photo_beta_v4d` |

**The coordinator's hypothesis is explicitly REFUTED.** The guess was that C's split pose sections are
the blocker and D (pu2-merged grammar) would pass. **D does clear the section-count and config checks —
and then fails on the RENDERER.** The blocker is **renderer generation, not pose split.**

**Root cause, measured: gr1 is a superseded vehicle generation on THREE independent axes.**

| axis | gr1 | live best |
|---|---|---|
| renderer magic | `TR1REN1!` (3,341 B) | `IX2REN01` (3,266 B) |
| pose grammar | `PFS1WPB1` (6,947 B incl. inert stub) | `PFS1WPD1` (8,751 B) |
| `frame0_policy` | not `warp_two_plane_static_photo_beta_v4d` | — |

**No rung is a candidate. There is no `−1,996` number to carry.**

## §8.3 The apples-to-apples check §4.8 never made — and it alone kills the row

CLAUDE.md's discipline is explicit: never compare archives that do not hold the same thing. **gr1
predates the pose work the live best carries.**

| rung | raw vs live best | **pose-normalised** | still a win? |
|---|---:|---:|---|
| A | +3,356 | +5,160 | no |
| B | −1,827 | −23 | *marginal* — but B has **no config section at all** and is REJECTED |
| **C** | **−1,757** | **+47** | **NO** |
| D | −1,784 | +20 | **NO** |

**The entire apparent −1,757 B is gr1 not having 1,804 B of pose.** The live best's extra pose bytes
are what *bought* the d_pose descent that produced most of its advantage. Comparing on total bytes
while ignoring that is the classic apples-to-apples failure — and it is, once more, **a join key never
checked** ("same pose content?").

## §8.4 SPILLOVER (§7 item 0b) — real, and **already banked**

| archive | total | `tokens.dr7t` | dead format | codes ≡ live best | renderer |
|---|---:|---:|---:|---|---|
| `gd3_CONTROL_identity_rebuild.zip` | 360,309 | 346,478 | **5,183 B** | **yes** | `TR1REN1!` |
| `v4d_composed_pw1_archive.zip` | 360,323 | 346,478 | **5,183 B** | **yes** | `TR1REN1!` |

The 5,183 B is confirmed on current-generation archives with bit-identical codes. **But it is NOT
harvestable rate on the live line: the live best ALREADY ships ix2 tokens at 341,295 B — it banked
this saving.** Both archives are ~6.5 KB *larger* than the live best and carry the legacy renderer.
**Correct reading: the 5,183 B measures how STALE the gen-1 archives are, not headroom on the frontier.**
§7 item 0b's "fleet-wide rate win" framing is **withdrawn**.

## §8.5 Self-protect finding from my own round-2 review

The submission **vendors its own `ddm_r7_token_coder.py` and it is NOT byte-identical to the repo copy**
(39,404 B vs 61,695 B, different sha256), while `inflate_runner.py:50` imports it **by bare name**. My
script imports the repo copy at module load, so it **shadows** the vendored one via `sys.modules`.

**Verdicts here are unaffected** — every rung verdict is produced inside the ix2 path (which never calls
r7; and `ddm_ix2_archive_container.py` **is** byte-identical between repo and submission), and the one
legacy case is refused at the `frame0_policy` check before any token decode. **But the harness must not
be able to mislead silently**, so it now emits a `_module_binding` receipt recording which copy actually
bound, whether it was the vendored one, and why that does or does not matter. *A future caller reaching
the legacy DR7T path would otherwise exercise the wrong coder with no warning.*

## §8.6 NEXT-IF-RESUMED (supersedes §7 items 0 and 0b)

1. ~~Close `#826`'s blockers~~ — **DONE: the row is REFUTED.** Remove it from the P2 rate lane. Do
   **not** spend the scorer slot on it. Blocker 6.5 (the 1.4 ppm seg leg) is **moot**: there is no
   receiver-valid candidate for it to adjudicate.
2. **The genuinely open question this leaves** — the live best's renderer is `IX2REN01` at 3,266 B
   while every gen-1 archive carries `TR1REN1!` at 3,341 B. **Is any *current* work still being built
   on the gen-1 grammar?** If so it is 6.5 KB behind before it starts, and that is a real, cheap,
   fleet-wide check (`$0`, container-level).
3. **Transferable law for the successor:** *an archive-size comparison across vehicles is void until
   the content is normalised.* This arm produced two instances in one day — mt1's stale token baseline
   (§4.9) and this one. Both were "smaller archive ⇒ better" read off vehicles holding different things.
4. Items 1–11 of §7 stand unchanged.

**Pointer honesty (§8):** `0.1910828242 [contest-CPU]` **UNMOVED**. This follow-up ran **zero scorer
forwards**, zero training launches, zero dispatch, $0. Archive decodes and container arithmetic only.
`score_claim=false`. **A smaller archive was never a score — and this one was not even smaller.**
