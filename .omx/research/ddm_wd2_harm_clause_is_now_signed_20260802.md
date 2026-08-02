# ddm_wd2 (#864/#861/#868) — the P0 grade's harm clause checked PRESENCE, never SIGN

**Date:** 2026-08-02 · **Arm:** ddm_wd2 · **P0:** `p0_864_built_elsewhere_unwired_is_p0_20260801`
(operator verbatim *"All built-elsewhere-unwired is p0"*) · **Axis:** `[macOS-CPU advisory]` ·
`score_claim=false`, `promotable=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false`. **0 scorer forwards. $0. 0 bytes shipped.** `upstream/`
untouched. No live run touched.

**POINTER HONESTY, FIRST.** Nothing here lowered any score. `effective_frontier` **0.172**
(upstream official leaderboard) UNMOVED; our own-vehicle line UNMOVED. This is **APPARATUS —
MEANS, not END.** Its justification is narrow and specific: the P0 grade that the operator
designated P0 could be declared for a component measured **~3,400× WORSE** than the live path, and
that row would have sorted **above every live debt row**. I proved that on the pre-fix module and
closed it.

---

## §0 HEADLINE (answer first)

**All three handed rows were already closed by sister arms before I started. The one thing left in
my lane was the defect `ddm_wd1` named and left unowned: `record_required_component` enforced that
a harm comparison was PRESENT, never that it pointed the right way.**

| handed row | state on arrival | who | I re-derived it? |
|---|---|---|---|
| **ROW C** #868 — grade absent from `VALID_BUILD_GRADES` | **CLOSED** | `ddm_wt1` `758858ccb3` | yes — read the live vocabulary |
| **ROW A** #820/#864 — the 8 built-elsewhere-unwired | **CLOSED**, 0 of 10 confirmed | `ddm_wr2` (7) + `ddm_wd1` (2) + `ddm_hl1` (1) | queue state measured |
| **ROW B** #861 — live pose solver on the superseded basis | **FALSIFIED, and it INVERTS** | `ddm_wd1` | **yes, independently — §2** |

**The dispatch premise for ROW B is wrong, and I reproduced the falsification myself rather than
relaying it.** My prompt called ROW B "the one row with a named recipient, so it is the priority of
the three." There is no such recipient: `warp-pose6` **is already live**, and the cosine basis the
P0 names as live has **no live consumer at all**.

**What I did, on the existing surface:** made the harm clause **SIGNED**. The declaration now
requires the two already-measured scalars plus the metric's direction, and is REFUSED unless the
candidate **strictly beats** the live recipient. Live grade-5 count **0 → 0** (no breakage), and
the pre-fix hole is proven with a two-sided control.

---

## §1 PROVENANCE

**STORES CONSULTED:** `.omx/research/ddm_wr2_wire_or_retire_adjudication_20260801.md`,
`ddm_hl1_disguised_builtelsewhere_copied_constant_tables_20260801.md`,
`ddm_sb2_complete_the_stubs_20260731.md`, `ddm_wd1_pose_wiring_falsified_and_correction_minimum_scale_20260802.md`,
`ddm_os1_optimization_sweep_termination_census_20260802.md`; `.omx/state/operator_p0_ledger.jsonl`
(the `p0_864` row, read verbatim); `.omx/state/required_component_ledger.jsonl`;
`.omx/state/canonical_frontier_pointer.json`; memories `built_elsewhere_unwired_is_p0_20260801`,
`grade5_not_derivable_20260801`. **Deliberately NOT loaded:** the burn/live-run state (this arm
touches no run); the seg token path.

| item | value |
|---|---|
| venv | `/Users/adpena/Projects/pact/src/tac/__init__.py` — hijack check **CLEAN** |
| scorer forwards | **0** |
| tests | **48 pass** `src/tac/tests/test_build_completeness_grades.py` (9 new); **12 pass** `src/tac/witness_dsl/tests/ -k "ledger or registry or completeness or activation"`; ruff `--select F` clean |
| blast radius | **MEASURED, 2 files.** Only consumers of `record_required_component` / `built_elsewhere_unwired` / `harm_advantage` in `src`+`tools`+`experiments` (worktrees excluded) are `activation_ledger.py` (16) and its test (46). `followon_ledger.py`'s single hit is a **docstring reference**, not a call. |
| live consumers | `tools/costate_digest.py` **rc=0**; `check_no_stub_lever_factories` live count **10, unchanged**; `built_elsewhere_unwired()` **0**, `not_even_designed()` **10**, report **202 rows** — all unchanged |
| review gate | two clean passes, no `REVIEW_GATE_OVERRIDE` |

---

## §2 ROW B RE-DERIVED — the premise inverts `verdict_scope: INSTANCE`

`ddm_wd1` falsified this yesterday. Per the never-relay rule I re-enumerated independently, in a
**named scope** (`src` + `tools` + `experiments`, `.claude/worktrees` excluded) with the
denominator and a positive control declared, because a bare zero is exactly the
negative-existence claim that is the day's dominant error class:

```
denominator                      62,923 .py files
positive control ('def ')        53,396 files  -> the zeros below are REAL, not empty-scope vacuity

eg1_generic_low_frequency_six    4 hits: tools/pb1_p5_byte_close_and_eval.py,
                                         tools/pb1_terminal_pose_gn_600.py (x2),
                                         tools/rehearse_terminal_pose_gn.py
                                 ZERO in experiments/   <- where the live vehicle lives
solve_terminal_pose_gn           4 files: own module, own tests, pb1_*, rehearse_*
                                 ZERO live callers
pfs1_warp_receiver (warp-pose6)  experiments/inflate_runner_v4d.py:57   <- LIVE
```

**So the P0's own MEASURED INSTANCE is inverted.** `p0_864` states *"terminal_pose_gn is wired to
ONE basis `eg1_generic_low_frequency_six_v1` … while warp-pose6 measures … ~39x better"*. In fact
warp-pose6 **is** the live basis; the cosine basis is confined to rehearsal tooling; and
`terminal_pose_gn` itself is the thing with no live caller. `ddm_wd1` adds the number that settles
it: that family plateaus at d_pose **~29–30** against a live realized **0.00858133** — an unwired
far-**worse predecessor**, so wiring it would be a **REGRESSION**.

I did **not** re-derive the ~29–30 plateau or the 0.00858133 myself; those are `ddm_wd1`'s and
`bc1`'s measurements, cited, not re-measured. What I re-derived is the **wiring topology**, which
is what the ROW B verdict turns on.

---

## §3 THE DEFECT, AND ITS TWO-SIDED CONTROL

`record_required_component` refused `grade=built-elsewhere-unwired` unless `live_recipient` and
`measured_comparison` were present — validated **only** by `len(val.strip()) >= 3` — while the
refusal message promised *"a MEASURED COMPARISON showing the component **beats** it"* and
`BUILD_GRADE_ORDER` placed the grade at **rank 0** on the stated grounds that it is *"the only
grade whose declaration is refused without a MEASURED comparison proving the live path is currently
worse off."*

That is a **comment-only contract**, which CLAUDE.md forbids, sitting on a P0 grade.

**Exhaustively verified before acting** (named scope as above): `measured_comparison` appears at 5
sites in `activation_ledger.py` (parameter, non-emptiness check, row write, report passthrough,
comment) and 4 in the test. **Zero sign checks anywhere.**

**Two-sided control — the hole was real, not theoretical.** I loaded `HEAD`'s pre-fix module
standalone (working tree untouched) and recorded `ddm_wd1`'s measured pose pair through it:

```
PRE-FIX  ACCEPTED a ~3,400x REGRESSION -> grade: built-elsewhere-unwired
PRE-FIX  P0 wiring queue length: 1
PRE-FIX  build_completeness_report()[0] == PoseBasisSwap    <- ABOVE every live debt row
POST-FIX REFUSED
```

**This is the mechanism that admitted the P0's own headline instance.** A row asserting "~39×
better" for a pair measured ~3,400× worse was recordable verbatim, and would have out-ranked all
real debt. The apparatus could not have caught the error the P0 itself made.

---

## §4 WHAT WAS BUILT (≈93 lines, one existing function, no new module)

`grade=built-elsewhere-unwired` now additionally requires `live_measured`, `candidate_measured`,
and `metric_direction ∈ {lower-is-better, higher-is-better}`, and REFUSES unless the candidate
**strictly** beats the recipient in that direction.

* **Strict inequality** — a tie is an unwired *equal*, which carries no present loss, so rank 0
  would be wrong for it.
* **Refuses non-finite** (NaN/±inf) — a failed measurement is not a comparison — and refuses
  `bool` explicitly, since `True` is an `int` in Python and would coerce silently.
* **Direction is checked, not inferred.** Without knowing which way the metric runs, "beats" is
  undecidable; there is no default.
* **`harm_advantage`** (dimensionless ratio) is DERIVED and recorded as evidence, and is `None`
  when a ratio is meaningless (zero or signed magnitudes) rather than fabricated. The strict
  inequality, not the ratio, is the gate.
* **NO-FAKE boundary:** the caller supplies two already-measured scalars; only their *comparison*
  is derived. This surface cannot manufacture a win — but see §6 on what it also cannot verify.

**The rank-0 comment needed no weakening.** It asserted something the code did not do; the fix
makes the existing claim true. That is the direction a repair should run.

**One docstring claim I did have to correct.** `built_elsewhere_unwired()` said the queue was
*"ranked by quantified harm rather than by guess."* It never was — the order is
`(fire_order, component)`. I did **not** make it rank by `harm_advantage`, because that would be a
guess wearing a number: the ratio is dimensionless and not comparable across axes (a 2× win on an
axis worth 0.29 S beats a 100× win on an axis worth 1e-9 S). Honest ranking needs each advantage
converted to S units, which no caller supplies today. So the docstring now states the real order,
the inputs are recorded for a future ranker, and `test_queue_order_is_fire_order_not_harm_as_the_docstring_now_states`
pins claim and behaviour together.

**9 tests, all asserting behaviour on real inputs.** The founding case (`ddm_wd1`'s real measured
numbers, REFUSED) with its negative control (same shape, direction reversed, ACCEPTED — without
which the refusal could be unconditional and every test would still pass); both metric directions;
tie; missing / non-finite / bool measurements; the undefined-ratio case; the queue-order claim; the
report passthrough; and a verbatim regression guard reproducing the exact pre-fix call shape.

---

## §5 MY OWN ROUND-1 REVIEW — one real defect in my own work

**I recorded a measurement the operator-facing report then dropped.** `build_completeness_report`
passed through `live_recipient` and `measured_comparison` but not my four new fields — so the
signed evidence existed in the store and was invisible on the surface an operator actually reads.
That is the **built-but-unsurfaced** class this grade exists to name, reproduced inside the module
that names it, by me, in the commit that closes its sibling. Fixed, with
`test_report_carries_the_signed_evidence_not_just_the_prose`.

**A second, smaller one:** my first test pass covered `metric_direction=""` only via a fixture
override, not via the literal historical call shape. Added
`test_the_exact_pre_fix_call_shape_is_now_refused`, which reproduces the pre-fix call verbatim —
the shape I had already proven was accepted at `HEAD`.

Both fixes are unreviewed new code and reset the clean-pass counter; both then passed two clean
review-tracker passes.

---

## §6 WHAT I COULD NOT CHECK — stated, not assumed

* **The read path bypasses the write gate. MEASURED, not reasoned.** A hand-appended JSONL row with
  `grade=built-elsewhere-unwired` and no evidence fields at all is admitted by
  `read_required_components`, enters `built_elsewhere_unwired()`, and sorts to report position 0. I
  reproduced this. **I did not fix it**, deliberately: the two available repairs are *drop the row*
  (signal loss — the opposite failure) or *surface it loudly* (a different mechanism, on a read
  surface I do not own). It is **pre-existing** and applies equally to `ddm_wt1`'s
  `live_recipient`/`measured_comparison` fields. **Flagged, unowned.** This matters more than it
  looks: `ddm_gd5` measured its auto-detector against real controls and **deleted** it, so the
  declaration path is the *only* route into this grade — and its store is hand-editable.
* **Whether the supplied numbers are true.** The gate enforces internal consistency (direction,
  finiteness, strictness), never that a scalar corresponds to a real measurement. `measured_comparison`
  remains the human-readable citation. A caller who fabricates both numbers defeats this, and that
  is a NO-FAKE violation at the caller, not something this surface can detect.
* **numpy scalars.** `np.float64` subclasses `float` and passes; `np.float32` does **not** and will
  be refused with the "requires a numeric" message. Fail-closed with a clear message, and I chose
  not to couple this module to numpy (it does not import it today) — but callers converting from
  arrays should pass `float(...)`.
* **#850 (the truncated GN solve).** NOT closed by me and NOT mine to close. `ddm_os1` measured the
  live census — `experiments/ddm_pfs1_ep_warp_pose_solve.py:183 solve_pair_gn`, converged **0/600**,
  stopped-on-a-bound **600/600** — and note `985f9aaf6b` is an os1 **CORRECTION** ("I read the
  solver at the wrong REVISION"), so anyone picking this up must re-derive at the current revision
  before quoting the census. It is a *bound/criterion* defect, categorically distinct from the
  (falsified) basis-swap row; I did not conflate them and did not touch it.
* **The full `witness_dsl` sweep.** I ran the targeted selector (12 pass), not the 1,181-test sweep.
  Blast radius is measured at 2 files, so this is bounded — but it is not a claim of full coverage.
* **Any score effect.** Zero scorer forwards. None run, none claimed.

---

## §7 CROSS-FINDINGS

* **→ the `p0_864` owner.** The inventory of 10 now restates as **0 confirmed + 10
  adjudicated-not-this-class** (wr2's 7 + wd1's 2 + hl1's 1). The **class is real** — the operator
  named it correctly — but the population attached to it was not, and the P0's own headline
  instance is inverted (§2). Recommend closing on population-empty with the refusal clause armed;
  the gate now makes a future false entry structurally impossible via the canonical path.
* **→ whoever owns the read surface.** §6's first bullet: the JSONL read path admits
  evidence-less grade-5 rows. Loud-surface, not drop.
* **→ `ddm_wd1`'s cross-finding is now DISCHARGED.** It wrote *"the predicate must be SIGNED, and
  nothing currently checks the sign of wt1's `measured_comparison` field."* That is what this
  landing closes, at the declaration path — the only path left after `ddm_gd5` deleted the detector.
* **→ any successor tempted to wire the pose basis.** Do not. §2: it is live already, and the
  unwired rival is a far-worse predecessor. The largest own-vehicle axis is still pose, but the
  lever is not this one.

## §8 WHAT THIS DID NOT DO

It did not lower any score, did not wire anything into the live vehicle, and did not drain a queue
that was already empty. It removed one way for the apparatus to be confidently wrong about a P0 —
which is worth exactly what a gate is worth, and no more.
