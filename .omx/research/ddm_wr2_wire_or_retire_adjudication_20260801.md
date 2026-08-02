# ddm_wr2 (#864) — WIRE-or-RETIRE adjudication of the unowned built-elsewhere-unwired rows

**Date:** 2026-08-01 · **Arm:** ddm_wr2 · **P0:** `p0_864_built_elsewhere_unwired_is_p0_20260801`
(operator verbatim *"All built-elsewhere-unwired is p0"*) · **Axis:** `[macOS-CPU advisory]` ·
`score_claim=false`, `promotable=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false`. **0 scorer forwards. $0.** `upstream/` untouched.

**POINTER HONESTY, FIRST.** Nothing here lowered any score. `effective_frontier` **0.172**
(official leaderboard) UNMOVED; our own-vehicle line (v4d **0.9639878**) UNMOVED; the borrowed
`0.1880443979880752` [contest-CPU] row is harvest-only and NON-SUBMITTABLE. This is **APPARATUS** —
MEANS, not END. Its only defensible justification is the one the p0 itself states: a mis-graded
debt inventory **produces wrong decisions**, and this one did.

**Review status:** pre-registered-only + own-round-1-reviewed (one real defect found in my own work
and fixed — §6) + fresh-context verifier dispatched against the spec (§7).

---

## §0 HEADLINE (answer first)

**Of the 7 unowned rows I own, some satisfy the p0's literal predicate — but ZERO satisfy its HARM
clause, which is the clause that makes the class a p0. The inventory's grade is wrong, and its own
source ledger already said so.**

The p0 states a predicate and, separately, the harm that justifies acting on it:

> *"a built-elsewhere-unwired component is one that EXISTS, is TESTED, and has FIRED on another
> vehicle, but has zero consumers on the live one… When the unwired component is a **measured-BETTER
> successor**, the live path is running the measured-WORSE thing, so the class costs score
> continuously and silently."*

Those are two different tests, and the distinction is the whole finding:

| test | met by |
|---|---:|
| **literal predicate** — donor exists + tested + fired elsewhere, no tr1 consumer | **4 of 7** (rows 1, 3, 4, 6). Row 2 fails it (the mechanism is built *nowhere*); rows 5 and 7 fail it (they *have* live consumers, just not trainer-side) |
| **HARM clause** — a measured-better successor to a live tr1 mechanism | **0 of 7** |

The harm requires a **recipient**: a live tr1 mechanism doing the same job, worse. No row has one.
Applying that test:

| what I found | rows |
|---|---:|
| **architecturally absent on tr1** — no recipient mechanism can exist, so "wire it" is the wrong verdict, not a deferred one | **3** |
| **mis-scoped, not unwired** — the component HAS live consumers, just not in the trainer | **2** |
| **purpose already delivered** — it was an AGREEMENT instrument and it agreed 17/17 | **1** |
| **mis-attributed donor** — the cited artifact is not the named mechanism; the mechanism is unbuilt everywhere | **1** |

**Three concrete errors, each independently checkable:**

1. **sb2's two surfaces contradict each other, in the same commit (`c44c7565af`).** The memo's §2b
   grades 8 components **BUILT-ELSEWHERE-UNWIRED-HERE** (grade 3). The machine-readable
   `.omx/state/required_component_ledger.jsonl` landed by the *same commit* grades **5 of those same
   components `not-even-designed`** (grade 5) — `TR1KDWarmStart`, `TR1GradientSurgery`,
   `TR1SigmaCCPrime`, `TR1BirthSeeding`, `TR1Hb1CapacityConsumer`. The remaining 2 (ms4d, #425) have
   **no ledger row at all**. **The ledger is right and the prose is wrong** — and the p0 quoted the
   prose (*"the 8 from ddm_sb2… sec 2b"*), so it inherited the wrong grade. Grade 3 vs grade 5 here
   is exactly a **donor-vs-recipient confusion**: §2b names the DONOR (real, built, fired elsewhere)
   while the debt it implies is a RECIPIENT mechanism the ledger correctly calls never-designed.

2. **sb2's KD correction is right but incomplete.** sb2 correctly found that tr1's `--distill-*` is a
   precomputed teacher-logit cache, not a warm start. But tr1 **does** have a teacher-derived student
   initialization: `--token-init-mode solve_project` (`train_tr1_partition_renderer_mlx.py:1396`,
   implemented `:1654-1709`), whose own help text calls it the **"eu1 teacher-as-init-oracle
   mechanism"** and which writes `model.tokens_base`/`tokens_delta` (or `model.tokens`) from a
   projected solution-set member. Its v1 and v2 rival formulations are recorded **MEASURED
   inadmissible** in-code; v3 was adopted.
   **Stated precisely, because the loose version would overstate it** (fresh-eyes correction): that
   "teacher" is **GT frame_1**, not a converged network. So what is live on tr1 is *teacher-derived
   initialization*; what is **not** live is the specific capability the HNeRV KD provided — carrying
   a **converged basin across an architecture/taper change**. The retirement does not rest on those
   being the same thing. It rests on (a) the donor being banned-lineage and architecture-typed, so it
   cannot be ported, and (b) tr1 having a measured, own-vehicle init path today. Row 1's reactivation
   trigger names the missing capability exactly.

3. **The "#425 phase carrier" row points at the wrong module.** sb2 cites
   `boundary_math/dash_phase_carrier.py`. The #425 **measured byte-close row** is
   `boundary_math/phase_residual_carrier.py` (#359 raster tie-field residual) — a different carrier.
   And `dash_phase_carrier.py` is not unwired: it has **5 non-test importers** —
   `witness_dsl/taskspace_r10_n600_maximum_inverse_fitter.py:26`,
   `witness_dsl/taskspace_r10_feature_texture_relay.py:32`,
   `optimization/predictor_r3_causal.py:31`, `tools/measure_dash_phase_carrier_n600.py:42`,
   `tools/run_taskspace_r10_feature_texture_relay.py:31` (+ `tools/levelset_byte_close_and_eval.py:3844`).
   Both modules additionally carry a `#425` tag in some artifact, so "#425 phase carrier" alone is
   genuinely ambiguous in-tree — anyone reasoning off the label conflates two distinct carriers.

**Why this matters beyond bookkeeping — the direct input to the gd5 detector (p0 NEXT_ACTION 2).**
The proposed predicate is *"src/tac modules imported by ≥1 tool/experiment but unreachable from the
live vehicle."* **That predicate reproduces this exact false-positive population** — every one of my
7 rows would fire under it, and none of them is the harmful class. The harmful instance (warp-pose6,
39× better / 38× cheaper, RACED) is harmful because a **live mechanism is running a measured-worse
rival**. The detector must be keyed on the **recipient** (a live mechanism with a measured-better
unwired successor), not on donor reachability. Reachability finds architecture mismatches; the
successor predicate finds score being lost.

---

## §1 PROVENANCE

**STORES CONSULTED:** `tools/corpus_query.py` ×2 (research 7355 · equations 864 · memory 2043 ·
dag 908 · council 292 · tasks 396 · docs 96) → loaded `ddm_sb2_complete_the_stubs_20260731`,
`ddm_wi1_wrong_instrument_sweep_20260731`, `ddm_ba31_negative_surfaces_20260731`,
`ddm_lg1_lane_guard_20260731`, `ddm_of1_offset_field_and_flicker_coherence_20260729`,
`p0_425_phase_carrier_byte_close_row_20260716`, `dmtz_taskaware_rate_lever_design_20260709`;
`.omx/state/required_component_ledger.jsonl`; `.omx/state/operator_p0_ledger.jsonl` (tail);
`.omx/state/canonical_frontier_pointer.json`. **Deliberately NOT loaded:** the pose-pair evidence
(`terminal_pose_gn.py`, `pb1_terminal_pose_gn_600.py`) — owned by `ddm_pw1`; the rank-4 /
`lane_guard.py:64-65` literals — owned by `ddm_hl1`; the grade-5 detector build — owned by `ddm_gd5`.

| item | value |
|---|---|
| venv | `/Users/adpena/Projects/pact/src/tac/__init__.py` — hijack check **CLEAN** |
| scorer forwards | **0** |
| tests | **31 pass** in `src/tac/tests/test_build_completeness_grades.py` (6 new + 1 regression); `src/tac/witness_dsl/tests/ -k "ledger or registry or completeness"` **11 pass**. The FULL `witness_dsl` sweep was SIGURG-killed by the harness (exit 144) and is **NOT** claimed clean — blast radius is one file (below), so this is bounded, not covered. ruff `--select F` clean |
| blast radius | **MEASURED:** the only consumer of `VALID_BUILD_GRADES` / `BUILD_NOT_DESIGNED` / `not_even_designed` / `build_completeness_report` / `record_required_component` outside the module is `src/tac/tests/test_build_completeness_grades.py`. `tools/costate_digest.py` runs clean. |
| `check_no_stub_lever_factories` live count | **10, unchanged** (no gate regression) |

---

## §2 THE ADJUDICATION — one row, one verdict

Ranked by the p0's stated rule: **measured delta where a receipt exists, cost-to-falsify where none
does.** No row is ranked by predicted ΔS (per `ddm_gc17`: all six #1-ranked levers of the 86-row
backcast were refuted by the measurements they ordered, bookings ~100× optimistic).

| # | row | re-derived? | TRUE live status | verdict | reason | reactivation trigger | owner |
|---|---|---|---|---|---|---|---|
| 1 | **KD warm start** (#74/#129) | **YES** — full read of `torch_vehicle/kd_warm_start.py` + `driver.py:1680-1710` + tr1 argparse/impl | Donor is **HNeRV-typed by construction**: `driver.py:1705` passes `vendored_decoder_cls=self.v.HNeRVDecoder`; transfer carries `(n_pairs,28)` latents + `best_ema_decoder.pt`. tr1 has neither, and HNeRV/PR95 lineage is **BANNED** as vehicle/carrier. **The function is ALREADY LIVE** on tr1 as `--token-init-mode solve_project`. | **RETIRE** | Not a measured-better successor: the live path already runs an own-vehicle teacher-init whose rivals were measured and rejected. Porting the HNeRV one is a banned-lineage import, not a repair. | A tr1 run needs init from a teacher **CHECKPOINT** (a converged tr1 predecessor at a *different* token geometry) that `solve_project` cannot express. Then build a **tr1-native token-space** KD — never port the HNeRV one. | ddm_wr2 → closed |
| 2 | **gradient surgery** (lg1 leg a) | **YES** — read `island_protection.py:594` + levelset `:8426` + `ddm_lg1:114` | **MIS-ATTRIBUTED.** `contain_protected_grad_mx` is a **mask-based containment** (freeze/damp/shield) used by the levelset `--seed-islands` seed module — **not** a 2-backward Fisher projection. `ddm_lg1:114` lists the real mechanism as **DEFERRED**, naming `contain_protected_grad_mx` only as *"the projection primitive — do NOT fork"*. **The mechanism is built NOWHERE.** | **NOT THIS CLASS** — stays open under its correct grade | It is grade 5, not grade 3: nothing was built elsewhere. Cost-to-falsify is **high** (~120 LOC + measured **~1.8× step wall-clock** on a live multi-day burn), and the live path already runs lg1 legs 1–3 (the λ_Lane primal-dual guard, FIRING in burn-4). Retiring it would falsely close legitimate lg1 debt. | Unchanged from lg1's own deferred table. Ledger row `TR1GradientSurgery` left **open**. | **ddm_lg1 successor** (unchanged) |
| 3 | **σ_cc′ length/tension** | **YES** — read `boundary_math/length_sigma.py` + tr1 `DUTY_TO_MEASURE:116` (entry `:135-140`) | **ALREADY TRACKED, absent by construction.** tr1's own `DUTY_TO_MEASURE` (`:116`, entry `:135-140`) carries `perclass_pair_surface_tension_sigma_ccprime`, `state="never-fired"`, receipt #382, and the exact precondition. **MEASURED:** the only `length/curvature/MCF/eikonal` string in the entire 2,543-line trainer **is that note itself**. `length_sigma` weights a Chan-Vese length integrand on the margin `m = φ_top1 − φ_top2`; tr1 has **zero** `phi_top`/`sdf`. | **RETIRE** (duplicate of a live tracked queue entry) | Not orphaned signal — it is *already* a properly-registered never-fired queue item with a receipt and a named trigger, i.e. exactly what "'off' is a tracked queue" requires. Counting it as orphan debt double-counts a working surface. | Verbatim the trigger tr1 already states: **a curvature/length/MCF regularizer is added to the tr1 loss.** | ddm_wr2 → closed |
| 4 | **birth-seeding Lever** | **YES** — read `curriculum_dsl.py:4251` (`SeedIslandBirth`) + tr1 grep | **MEASURED:** the string `island` appears **0 times** (case-insensitive) in the tr1 trainer; so do `signed_dist`/`sdf`/`phi_top`. The Lever (`:4251`, overrides `:4264-4265`) emits `--seed-islands` + `--witness-alone-island-loss`, whose only argparse definitions in the repo are the levelset trainer `:20021` / `:20122`, and its semantics presuppose an island/level-set objective. | **RETIRE** | The recipient cannot exist: there is no connected-component/island objective on tr1 to seed. This is architecture mismatch, not missing wiring. | tr1 grows a per-class connected-component (island birth/persistence) objective. | ddm_wr2 → closed |
| 5 | **ms4d metric bundle** | **YES**, and **fresh-eyes CORRECTED my first reading** | **Its PRODUCT is consumed live; its MODULE is not imported anywhere but its own runner.** Precisely: `ddm_campaign_costate.py:157-162` declares `CampaignSource("ms4d_metric_bundle", ".../BUNDLE-COMPLETE.json", "ddm_metric_custody_bundle.v1", <staleness>)` and that artifact **exists (2.7 KB)** — but a `CampaignSource` is a **path + schema string, not a module import**. The only non-test importer of `ddm_ms4d_direct_completion` is its own CLI `tools/complete_ddm_ms4d_direct_metric.py:11`. Its docstring: *"a measurement compiler, not an actuator search."* | **RETIRE** (mis-scoped) | A measurement compiler's **wiring is its artifact being consumed**, not its module being imported — and the artifact IS consumed, by the costate organ (a CLAUDE.md core sense organ), under a declared staleness horizon. *"No tr1 consumer"* is literally true and is the wrong question. **I do not claim module-level consumers** — my first draft did, and the fresh-context verifier refuted it. | Verbatim the staleness condition the costate source already declares: **the scorer, R operator, target cache, or 25-bucket membership changes.** | ddm_wr2 → closed |
| 6 | **#725 BN capacity** | **YES** — read `hope_bn_capacity.py` header + `hope_rg3_agreement_receipt.json` | **PURPOSE DELIVERED.** `validation_gate.required` = *"reproduce-or-refine the 17 hand-derived RG3 Fisher-margin codebook rows"*; verdict **`REPRODUCED_17_OF_17`**, `capacity_refined_equal_to_recorded=17`. Its own receipt declares `rate_denominator_policy = "no rate columns; score_units_per_byte_status=OWED_NOT_ADMITTED"`. | **RETIRE** | An instrument that **agrees 17/17** with the incumbent is by construction **not** a measured-better successor — the p0's harm model cannot apply. It self-declares not-admitted to the score path. Its value was the agreement, and the agreement was delivered. | A **MEASURED disagreement** between HOPE per-channel capacity and the RG3/ms4d Fisher-margin codebook, **OR** a real measured-coder-byte denominator lifting `score_units_per_byte` out of `OWED_NOT_ADMITTED`. | ddm_wr2 → closed |
| 7 | **#425 phase carrier** | **YES** — resolved the module confusion; read the byte-close row + `ddm_tr1_runtime.SECTION_CONTRACT`; **derived the falsifier** | **Wrong module cited** (see §0.3). Three *structural* blockers, none a wiring job: **(a)** the measured row is on the **levelset v9_cgauge** vehicle and its `recovered_d_seg` is **EXPLICITLY OWED** — the benefit is unmeasured, so it cannot be ranked at all; **(b)** the carrier's INPUT is a margin/tie field (`compute_tie_field_from_margins`) tr1 does not compute; **(c)** the live packet grammar is **CLOSED** — `SECTION_CONTRACT` (`ddm_tr1_runtime.py:90-95`) = `(tokens, lotto_renderer, selector, pose_stub)`, each bound to a named receiver consumer, and the parser **REFUSES** at `:363-366` if `section_order`/`section_consumers` differ (metadata key set is closed too, `:360`). A 5th section is a **receiver-schema change**. | **RETIRE** with a pre-registered falsifier | The only row of the 7 with a genuinely open door, but it is a *design* decision with an **unmeasured benefit**, not an unwired module. Per P7 (falsifier-before-build) it gets a kill criterion now, not a build. | **DERIVED, pre-registered (§4):** at the measured 10,682 B section cost the carrier is net-negative **iff it fixes > 0.6346 flips per stored residual** (8,390.5 flips; d_seg reach 7.113e-05). Re-open when a tr1-side margin field exists and that through-R n600 A/B can run. | ddm_wr2 → closed, trigger armed |

**Verdict scope on every negative above: INSTANCE.** Each retirement is scoped to *this component
against the tr1 vehicle as it stands*, with a named structural precondition. None is a family or
paradigm claim: level-set island seeding, Chan-Vese σ_cc′, KD warm starts and phase carriers are all
live, sound families **on the vehicles that have their preconditions**.

---

## §3 THE COUNT — the prompt's "six" is off by one; the real miscount is deeper

**Surface arithmetic.** The p0 evidence names the inventory explicitly: **10 = sb2 §2b's 8 + the 2
found 2026-08-01** (warp-pose6 basis, tt1 analytic Jacobian). Owned elsewhere: the pose pair (2,
`ddm_pw1`) + rank-4 (1, `ddm_hl1`) = **3**. Unowned = **7**, which is exactly the table I was handed.
My dispatch said "six"; the table said seven. **The table is right.** ("The detector, `ddm_gd5`" is
p0 NEXT_ACTION item (2) — apparatus to be built, never one of the ten.)

**The substantive miscount.** `8` is not the count of grade-3 components. On my slice of 7:

| grade | as claimed in sb2 §2b | as re-derived |
|---|---:|---:|
| BUILT-ELSEWHERE-UNWIRED-HERE (grade 3) | 7 | **0** |
| NOT-EVEN-DESIGNED (grade 5) — recipient never designed | 0 | **4** (rows 1, 3, 4, 6) |
| NOT-EVEN-DESIGNED — mechanism unbuilt *anywhere* | 0 | **1** (row 2) |
| BUILT-AND-FIRED, non-trainer consumers | 0 | **2** (rows 5, 7) |

sb2's own ledger already carried 5 of these as grade 5. The prose table is the surface that got
escalated.

---

## §4 THE ONE DERIVED NUMBER, and its independent check

The #425 falsifier is derived from the **registered** water law, not borrowed. Reproduced
independently here from first principles (`N_total = 384·512·600 = 117,964,800`):

```
S per flip = 100 / 117,964,800            = 8.477105e-07
S per byte = 25  / 37,545,489             = 6.658590e-07
water      = 8.477105e-07 / 6.658590e-07  = 1.2731082153  B/flip
```

That reproduces the registered `WATER_B_PER_FLIP` **1.2731082153320312** to 10 significant figures —
an independent check that my arithmetic is on the canonical law, not a remembered constant. Applied
to the measured #425 section (10,682 B / 13,222 residuals, `p0_425_phase_carrier_byte_close_row_20260716`):

```
rate cost      = 10,682 · 6.658590e-07 = 0.007113 S   (matches the memo's 0.007113)
break-even     = 0.007113 / 8.477105e-07 = 8,390.5 flips   ⇒ d_seg reach 7.113e-05
per residual   = 8,390.5 / 13,222        = 0.6346 flips
```

**Direction correction, stated because it is load-bearing.** `ddm_ba31` §B.5 tabulates the *W1-COH*
phase carrier at `0.075–0.141 B/flip`, "9.0×–17.0× BELOW" the water, and its prose groups it with
carrier families "**dominated** by an order of magnitude." The arithmetic says the opposite —
`B/flip < 1.2731` is precisely the admission condition — and the primary source agrees in as many
words: `ddm_of1` records *"9–17× under water"*, **"a PRICED, OPEN door"**, *"enters the waterfill as
a real action."* I did **not** transfer that number to #425: **W1-COH (per-region flicker phase) and
#425 (raster tie-field residual) are different carriers**, and borrowing the price across them would
be the exact error this memo is auditing. Flagged for ba31's owner as a possible sign inversion —
the same class ba31 itself found in the wr1 ceiling column.

---

## §5 THE APPARATUS REPAIR (paid on the existing surface, not a new one)

**The problem, found while trying to record the verdicts:** the required-component ledger **could not
express a RETIRE.** `VALID_BUILD_GRADES` had four values and `not_even_designed()`'s docstring is
explicit — *"this list can only be drained by BUILDING, never by editing a memo."* That is the right
instinct against memo-editing, but it means a component whose **recipient cannot exist** nags
forever, which trains readers to ignore the queue. Meanwhile the **ACTIVATION** axis of the very same
module has had `STATE_RETIRED` (*dormant-with-reactivation*) since it landed, and CLAUDE.md's "'Off'
is a tracked queue" names `retired-with-reason` as a **mandatory** state. sb2 added the BUILD axis
without the one state its sibling already had.

**The repair (≈15 lines, `src/tac/witness_dsl/activation_ledger.py`), deliberately minimal:**

- `BUILD_RETIRED = "retired-with-reason"` added to `VALID_BUILD_GRADES`, mirroring `STATE_RETIRED`.
- `record_required_component` **REFUSES** `grade=retired-with-reason` without substantive `notes` —
  the reactivation trigger is mandatory-by-refusal, matching the existing charter-field style. *A
  retirement without a trigger is a KILL, which CLAUDE.md forbids as a resting state.*
- `not_even_designed()` excludes retired rows; `read_required_components()` still returns them, so a
  retirement is **dormant-and-auditable, never deleted**.
- `build_completeness_report()` keeps a retired row's own grade instead of re-asserting
  `not-even-designed` — otherwise a retirement silently becomes a build order again.
- `BUILD_GRADE_ORDER` **exported** so no consumer hand-copies it (see §6).

**Effect, measured:** live grade-5 debt **13 → 9**; 6 rows retired-with-reason and still auditable;
`check_no_stub_lever_factories` live count **10, unchanged**. `TR1GradientSurgery` correctly stays
**open** — the repair does not launder legitimate debt.

**One durability observation, found by the serializer refusing my commit (rc=13).**
`.omx/state/required_component_ledger.jsonl` is **gitignored** — correctly, per CLAUDE.md's
live-state rule. But that means the charter queue the p0 is built on, and these verdicts, exist only
on this machine and **do not survive a fresh checkout**. The tracked durable record is this memo. Not
a defect of the repair (it predates it, and applies equally to sb2's original 12 charters), but the
apparatus's own "the registry drains this queue, not a human editing a memo" claim is weaker than it
reads: on a fresh clone the queue is empty and the memo is all there is. **Flagged, unowned.**

---

## §6 MY OWN ROUND-1 REVIEW — one real defect in my own work

Per the #337 contract I reviewed my own output before handing it over, and it found a defect that my
new tests did **not** catch:

**The first pass of retirements did not drain anything.** `read_required_components` is
latest-row-wins on the composite key **`(component, needed_by)`**. I recorded my retirements under a
new `needed_by` (`"p0_864_…"`) while sb2's charters used `"path_a_from_start_optimal"` /
`"path_b_reset_arm_dpm"`. Different key ⇒ **no supersession**: both rows survived, the original
grade-5 rows stayed live, and the queue *looked* adjudicated while being unchanged (13 rows, not 9).
My tmp_path tests passed throughout because each fixture had exactly one row per component — the
classic "would these tests still pass if the code were broken?" failure.

**The fix is to use the key correctly, not to weaken it.** The `(component, needed_by)` key is
deliberate: one component can be required by several configs, and retiring it for one must not retire
it for the others. I re-recorded each retirement onto its **original** charter key, and added
`test_retirement_is_per_charter_key_not_per_component`, which asserts both directions (a third,
unrelated key drains nothing; the charter's own key drains exactly that charter). The mis-keyed rows
remain in the append-only ledger — history is not rewritten — and are harmless, since a retired row
creates no debt.

**A second, smaller class fix.** Adding a 5th grade broke a pre-existing test that had **hand-copied**
the 4-grade order map. Rather than patch the copy, I exported `BUILD_GRADE_ORDER` from the module and
made the test consume it, plus `assert set(BUILD_GRADE_ORDER) == set(VALID_BUILD_GRADES)`. The
duplicated constant *was* the drift generator; patching the instance would have left the class alive.

**Both fixes are unreviewed new code and reset the clean-pass counter to 0.**

---

## §7 WHAT I RE-DERIVED vs COULD NOT

**RE-DERIVED at source (file:line read, not relayed) — every row in §2.** Specifically:
`kd_warm_start.py` in full · `driver.py:1680-1710` · tr1 `--token-init-mode` argparse + its
`:1654-1709` implementation · `island_protection.py:594` + the levelset `:8426` call site ·
`ddm_lg1:21,114` (the DEFERRED row) · `length_sigma.py` header + preset derivations · tr1
`DUTY_TO_MEASURE:116-140` · `curriculum_dsl.py:4251` · `ddm_ms4d_direct_completion.py` header +
consumer enumeration + the on-disk bundle · `hope_rg3_agreement_receipt.json` · the #425 byte-close
row · `ddm_tr1_runtime.SECTION_CONTRACT` · `.omx/state/required_component_ledger.jsonl` (all 19 rows)
· the water-level arithmetic from first principles.

**RE-DERIVED and found WRONG in the seed:** the grade-3 classification (§0.1), the KD-path claim
"no path exists" (§0.2), the #425 module attribution (§0.3), and the dispatch's count of six (§3).

**FRESH-CONTEXT VERIFIER (dispatched against the spec, not against my recollection).** 6 of 7 claims
VERIFIED at file:line; **1 PARTIAL, and it corrected me**: I had called `ddm_campaign_costate`'s
`CampaignSource("ms4d_metric_bundle")` a *module consumer*; it is a **path + schema string**, and the
only non-test importer of `ddm_ms4d_direct_completion` is its own CLI. I re-checked that myself and
the verifier is right — row 5 is restated accordingly. It also supplied four line-number corrections
(all applied).

**The verifier itself reproduced the day's dominant error class, which is worth recording.** It
reported `dash_phase_carrier.py` as having *"exactly one non-test importer"*. Direct grep returns
**five** (`taskspace_r10_n600_maximum_inverse_fitter:26`, `taskspace_r10_feature_texture_relay:32`,
`predictor_r3_causal:31`, `measure_dash_phase_carrier_n600:42`,
`run_taskspace_r10_feature_texture_relay:31`). That is a **negative-existence claim made without
exhaustive search** — the class already recorded as today's most frequent false claim. I did not take
either of its verdicts on faith: I re-derived both at source, which is why the ms4d correction was
accepted and the dash correction was rejected. **A verifier's negative is evidence, not a finding.**

**COULD NOT check — stated, not assumed:**

- **Whether retiring `TR1KDWarmStart` blocks Path A.** The sb2 charter says it *"gates Path A and
  both deferred fresh cells."* I verified the mechanism claim, **not** the gating claim — I did not
  read the Path-A config to confirm `solve_project` satisfies whatever Path A actually needs. If
  Path A's owner disagrees, the ledger row is one append away from re-opening. **Flagged to burn-4's
  owner.** *(`ddm_ba31` independently records "from-birth-KD `DEFERRED`; `TR1KDWarmStart` was never
  buildable on tr1", which supports the retirement — but that is corroboration, not my check.)*
- **The ba31 sign question (§4).** I established that of1's primary source says "open door" and that
  the arithmetic supports it. I did **not** re-read ba31's full §B.5 derivation to determine whether
  its prose is a genuine inversion or a compressed restatement of a prior verdict it is auditing.
  **Flagged, not concluded.**
- **Any score effect.** Zero scorer forwards; no measurement was run or claimed.
- **Whether the retirements are correct on any vehicle other than tr1.** Scope is INSTANCE/tr1 only.
- **Consumers living outside `src/`, `tools/`, `experiments/`.** Repo-wide greps time out because
  `.claude/worktrees/` holds many full tree copies, so every consumer count here is scoped to those
  three roots (the verifier hit the identical wall independently). A consumer elsewhere would have
  been missed — so read every "N importers" above as *"N in the scanned roots"*, never as a bare
  negative.
- **Transitive reachability.** tr1's non-consumption was established from its 11 `tac.*` import
  sites; I did not fully expand all 11 transitively. The direct-import negatives are solid; a deep
  transitive edge is possible.

---

## §8 CROSS-FINDINGS

**→ `ddm_gd5` (the grade-5 auto-detector, p0 NEXT_ACTION 2) — the highest-value output of this arm.**
Do **not** key the detector on import-reachability from the live entry point. That predicate fires on
all 7 rows here, 0 of which are the harmful class; it would ship a detector whose dominant output is
architecture mismatch. Key it on the **recipient**: *a live mechanism that has an unwired successor
with a measured-better number on a comparable surface.* That predicate isolates warp-pose6 (39×) and
excludes everything in §2 — which is the discrimination CLAUDE.md P4 demands (a new meter needs a
positive **and** a negative control before its readings count). §2 is a ready-made negative-control
set of 7; the pose pair is the positive control.

**→ the `p0_864` owner.** The inventory of 10 should be restated as **2 confirmed + 1 (`ddm_hl1`) +
7 adjudicated-not-this-class**. The class is real and the operator found its real instance by hand;
what is not real is the population of 8 the memo prose attached to it.

**→ `ddm_ba31`'s owner.** Possible sign inversion at §B.5 (§4 above) — same class as the wr1 ceiling
column ba31 itself surfaced.

**→ `ddm_lg1` successor.** `TR1GradientSurgery` stays open and is **yours**; it is grade 5, not
grade 3, and `island_protection.py:594` is a *primitive to reuse*, not a prior build.

---

## §9 WHAT THIS DID NOT DO

It did not lower any score, and it did not wire anything into the live vehicle. The honest summary is
that the highest-value action for the live line is **not** in this inventory: it is the pose pair
`ddm_pw1` owns (pose is **0.292941** of v4d's **0.9639878**, with a measured 39× successor sitting
unwired). Everything here is apparatus that stops seven rows from consuming attention that belongs
there.
