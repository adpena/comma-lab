---
schema: ddm_wd1_pose_wiring_and_correction_scale.v1
date_utc: 2026-08-02
arm: ddm_wd1
tasks: ["832", "864", "820", "861", "868"]
axis: "[macOS-CPU advisory] NON-PROMOTABLE"
score_claim: false
promotable: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
verdict_scope: INSTANCE
council_predicted_mission_contribution: frontier_protecting
consumes: [ddm_pw1_pose_menu_saturation_20260801, ddm_wr2_wire_or_retire_adjudication_20260801,
  ddm_dc1_correction_label_cost_and_qa03_censoring_20260801, ddm_ba31_negative_surfaces_20260731,
  ddm_bc1_qa24_compose_and_fire_20260731, ddm_gd1_generic_default_census_20260731]
consumers: ["#864", "#832", ddm_gd5_grade5_detector, ddm_ba31_owner, ddm_dc1_owner]
---

# ddm_wd1 — the pose wiring debt does not exist, and the correction stream has a MINIMUM SCALE

**POINTER HONESTY, FIRST.** Nothing here lowered any score. `effective_frontier` **0.172**
(official) UNMOVED. Our own-vehicle line is **v4d→pw1 0.9476091** `[macOS-CPU advisory]`,
gap-to-bar **0.7754681**. This arm fired **0 scorer forwards**, shipped **0 bytes**, and did not
touch `experiments/ddm_v4c_resolve.py` (live run, pid 18732). It is APPARATUS and one derived
number — MEANS, not END.

---

## §0 HEADLINE (answer first)

**Both surfaces I was handed had already been landed by sister arms before I started, and the
P0 half's central premise is FALSIFIED at source. The one thing neither arm did, I did: the
correction stream has a minimum viable scale, and the experiment that condemned it ran 78× below
that scale.**

| handed surface | state when I started | my contribution |
|---|---|---|
| **(A) #832** — "the 0.264 S sign swing sits on a scorer-free test nobody has run" | **ddm_dc1 ran it** 2026-08-01 (receipt generated `01:07:19Z`). The swing was adjudicated: arithmetic verified, but it compares **two prices of different objects** | independent re-derivation (✔ reproduces ba31's row exactly), a **conservatism discrepancy** in dc1's own headline, re-anchoring to the moved gap, **and the minimum-scale finding** |
| **(B) #864/#820/#861** — "wire the pose pair; warp-pose6 is RACED-BETTER but unwired" | **warp-pose6 has been LIVE since 07-30** (ddm_pw1); ddm_wt1 already accepted the scope correction and did not re-litigate it | independent source verification with denominators, the **regression finding** (the unwired candidate is ~3,400× *worse*), and the ledger rows so the inventory stops carrying the false entry |
| **(C) #868** — "the grade CANNOT BE RECORDED; fix that first" | **ddm_wt1 landed it** (commit `758858ccb3`): `built-elsewhere-unwired` is `VALID_BUILD_GRADES[5]` | verified present; used it — and it correctly **refused** both pose rows |

**The three headline results:**

1. **The pose wiring debt has no recipient, and wiring it would be a REGRESSION.** Not merely
   "already live" — the unwired candidate family plateaus at **d_pose ~29–30** while the live
   chain realizes **0.00858133** (pw1 AB **0.00764506**). That is **~3,400× worse**. The P0's harm
   clause reads "the live path is running the measured-WORSE thing"; here the live path is running
   the measured-**better** thing by three orders of magnitude.

2. **The P0's own machine-readable queue is EMPTY, and that is the apparatus working.**
   `built_elsewhere_unwired()` returns **0 of 20** ledger rows. wt1 made the harm clause
   mandatory-by-refusal; when I attempted to record the pose pair under the P0's designated grade,
   **the refusal is what a correct instrument does** — neither row can name a live recipient. Both
   are now recorded `retired-with-reason` with reactivation triggers.

3. **NEW — the correction stream has a minimum viable scale of ~29% of the residual, and QA03 ran
   at 0.37%.** Measured B/flip falls monotonically with support density, so a stream correcting a
   *fraction* of the residual codes a *sparser*, **dearer** support. Break-even is at
   **ρ_c = 1.2334e-3** ⇒ **f\* = 0.2860** (live seg base). QA03 addressed **0.0128×** that
   fraction. **QA03's negative verdict was taken 78× below the scale at which any correction stream
   can pay** — independently of the cap-censoring dc1 separately found. Two distinct censorings,
   and they compound.

---

## §1 PROVENANCE

**STORES CONSULTED:** `tools/corpus_query.py` (research 7373 · equations 869 · memory 2044 ·
dag 913 · council 292 · tasks 398 · docs 96) → loaded `ddm_wr2_wire_or_retire_adjudication_20260801`,
`ddm_pw1_pose_menu_saturation_20260801`, `ddm_dc1_correction_label_cost_and_qa03_censoring_20260801`,
`ddm_ba31_negative_surfaces_20260731` §B.2–B.3, `ddm_bc1_qa24_compose_and_fire_20260731` §3.5;
`.omx/state/operator_p0_ledger.jsonl`, `.omx/state/canonical_task_status.jsonl`,
`.omx/state/canonical_frontier_pointer.json`, `.omx/state/main_hot_state.md`,
`.omx/state/required_component_ledger.jsonl` (all 20 rows); `.omx/research/ddm_dc1_label_price_n600_20260801.json`;
`git log`/`git show 758858ccb3`. Primary code read at source: `src/tac/witness_dsl/activation_ledger.py`,
`experiments/inflate_runner_v4d.py`, `experiments/ddm_v4d_build_composed_archive.py`.
**Deliberately NOT loaded:** the burn/b4s seg line; the r7/SMEVR coder arm.

| item | value |
|---|---|
| venv hijack check | `tac.__file__` = `/Users/adpena/Projects/pact/src/tac/__init__.py` — **CLEAN** |
| scorer forwards | **0** |
| live run touched | **none** (`ddm_v4c_resolve.py` pid 18732 untouched) |
| ruff `--select F` | clean on the one new file |

---

## §2 (B) THE POSE WIRING — FALSIFIED AT SOURCE, with denominators

I did **not** take pw1's correction on faith; a relayed negative is not a finding. I re-ran the
enumeration myself over a deliberately **wider** scope than pw1 used.

**Scan denominator: 62,913 `.py` files** under `src/`, `tools/`, `experiments/`
(`.claude/worktrees/` excluded). **Positive control: 53,386 / 62,913 files contain `def `** — the
scan reaches files, so a zero is a real zero within this scope and not an empty-scope vacuity.

| symbol | hits | where |
|---|---:|---|
| `eg1_generic_low_frequency_six` (the "superseded control basis") | **4** | `tools/pb1_p5_byte_close_and_eval.py:264`, `tools/pb1_terminal_pose_gn_600.py:15,47`, `tools/rehearse_terminal_pose_gn.py:50` — **ZERO in `experiments/`** |
| `solve_terminal_pose_gn` | **18** | 2 self (`terminal_pose_gn.py:898,1295`) + **11 its own tests** + 5 in `tools/pb1_*`/`rehearse_*` — **ZERO live callers** |
| `pfs1_warp_receiver` (warp-pose6) | **24** | includes **`experiments/inflate_runner_v4d.py:57`** — the LIVE receiver |
| `warp_two_plane_static_photo_beta_v4d` | **2** | `experiments/ddm_v4d_build_composed_archive.py:43`, `experiments/inflate_runner_v4d.py:69` |

**Leg 1 — the basis. VERDICT: no recipient exists.** The live chain
(`ddm_v4c_resolve.py` → `ddm_v4d_resolve.py` → `ddm_v4d_build_composed_archive.py` →
`inflate_runner_v4d.py`) imports `pfs1_warp_receiver` and sets
`FRAME0_POLICY="warp_two_plane_static_photo_beta_v4d"`. The cosine basis appears **only** in the
retired `pb1`/`rehearse` instruments. The 39×/38× comparison in the P0 evidence is real, but it
compares **the live basis against a retired one** — i.e. it is the receipt showing the live path
already picked the winner, read as though it showed the opposite.

**Leg 2 — the analytic Jacobian. VERDICT: PARITY, not supersession, and the race is confounded.**
Re-derived at `ddm_bc1_qa24_compose_and_fire_20260731.md:66-73`:

| solver | parameterization | plateau d_pose |
|---|---|---:|
| FD-LM-GN | warp-pose6, FD Jacobian | **~30** |
| analytic-LM-GN | warp-pose6, STE autograd Jacobian | **~29** (200→48 in one relinearization, then STALLS) |

bc1's **own** verdict at `:73`: *"ALL FOUR plateau at d_pose ~10-38 … This is FUNDAMENTAL, not
[solver quality]."* And `:81` records the arms ran with `s_t=1.0` while `ST_GRID` is 0.005–0.24 —
so the comparison is **confounded** as well as at parity. `gd1` P6's "RACED-SUPERSEDED, adoption
OWED" promoted bc1's *motivating hypothesis* and dropped bc1's *table*. That is the
escalation-by-retelling class, third instance today.

**The finding neither the P0 nor pw1 stated in one line.** Both legs sit on a family whose
plateau is **d_pose ~29–30**, against a live realized **0.00858133** (pw1 AB **0.00764506**).

```
29 / 0.00858133  =  3,380x WORSE than the live path
```

So the row is not "an unwired better successor." It is an unwired **far-worse predecessor**.
Wiring it is a regression, and the P0's harm model — which prices the loss from running the worse
thing — inverts: the loss would be *created* by the wiring, not removed by it.

**Verdict scope: INSTANCE.** Scoped to these two components against the live v4d chain as it
stands. Neither analytic Jacobians nor alternative pose bases are dead families — warp-pose6 *is*
one, and it won.

### §2.1 What I recorded, and the refusal that is the point

`built_elsewhere_unwired()` = **0 of 20** rows before and after my writes. wt1's harm clause
(`live_recipient` + `measured_comparison`, mandatory-by-refusal) **refuses** both pose rows,
correctly. I recorded them at the honest grade instead:

| component | grade | reactivation trigger (mandatory, recorded) |
|---|---|---|
| `TR1WarpPose6BasisAdoption` | `retired-with-reason` | the live receiver chain imports a pose basis other than `pfs1_warp_receiver`, **OR** a successor measures d_pose below the live realized 0.00764506 through the real receiver at n600 |
| `TR1AnalyticJacobianAdoption` | `retired-with-reason` | an analytic-Jacobian solve measures **below** the live realized d_pose at n600 through the real receiver, on a run where `s_t` is drawn from `ST_GRID` rather than pinned to 1.0 |

**Durability caveat, inherited and restated (wr2 §5):** `required_component_ledger.jsonl` is
gitignored, so these rows do not survive a fresh checkout. **This memo is the durable record.**

**⇒ The P0's inventory of 10 should now be restated as: 0 confirmed + 1 (`ddm_hl1`, unread by me)
+ 9 adjudicated-not-this-class** (wr2's 7 + my 2). The class is real; its population is empty.

---

## §3 (A) #832 — dc1 landed it; here is what re-derivation adds

`ddm_dc1` ran ba31's named cheap test on 2026-08-01 with both controls passing. I re-derived its
arithmetic from its own receipt rather than relaying it.

**REPRODUCED exactly.** Water from first principles:
`(100/117,964,800) / (25/37,545,489) = 1.2731082153` — matches registered
`1.2731082153320312`. And ba31's label-free row reproduces to 6 dp:

| price basis | B/flip | bytes | rate +S | seg −S | **NET S** | % of gap |
|---|---:|---:|---:|---:|---:|---:|
| ba31 uniform position, **label unpaid** | 0.9822 | 499,579 | +0.332649 | −0.431179 | **−0.098530** | **12.44%** (old gap) |

**That −0.098530 / 0.7918468 = 12.44% is exactly the number my dispatch called "the
interpolation-free bound."** Confirmed — and **superseded**: it prices only half the cost, and dc1
measured the other half. With label paid the win is **larger**, not smaller.

**Two corrections to the numbers as they now stand:**

**(i) The gap moved after dc1 wrote.** `ddm_pw1` moved the own-vehicle line
**0.9639878 → 0.9476091**, so gap-to-bar went **0.7918468 → 0.7754681**. Every percentage in
ba31/dc1 is anchored to the old gap. Re-anchored (my re-derivation, log-linear in density, the
method ba31 used and dc1 verified to 4 dp on the position component):

| base | B/flip (pos+label) | × water | NET S | % of gap **0.7754681** |
|---|---:|---:|---:|---:|
| ja1/v4c = **the LIVE seg term** | 0.8507 | 0.668 | **−0.143049** | **18.45%** |
| burn ep854 | 0.8837 | 0.694 | **−0.120605** | **15.55%** |

This composes onto the live pointer: pw1 moved **pose**; seg is `0.4311790` **UNCHANGED from v4d**
(same tokens), so the ja1 base *is* the live seg term.

**(ii) A conservatism discrepancy in dc1's headline, flagged not concluded.** dc1 states
**0.883–0.915 B/flip** and nets **−0.132 / −0.111**. I cannot reproduce those endpoints from
dc1's own receipt under any standard interpolation:

| method | ja1 | burn |
|---|---:|---:|
| log-linear (ba31's, dc1-verified) | 0.8507 | 0.8837 |
| linear in density | 0.8718 | 0.9053 |
| log-log | 0.8421 | 0.8745 |
| **dc1 stated** | **0.883** | **0.915** |

All three are **cheaper** than dc1's stated range, i.e. **dc1 under-claimed its own win by
~0.03 B/flip ≈ 0.011 S**. The direction is unaffected and the error is in the safe direction;
I could not determine dc1's exact method from the artifacts. **Flagged to dc1's owner, not
concluded** — this is a discrepancy, not a defect.

---

## §4 THE NEW FINDING — the correction stream has a MINIMUM SCALE

**The question neither arm asked.** ba31 and dc1 both priced the correction at **one** density —
the full residual, `f = 1.0`. ba31 flagged the gap in its own caveat (a): *"this prices the slope,
not a realized move."* But the caveat was left as a caveat, and the decision-relevant consequence
was never derived.

**The mechanism.** Measured B/flip falls **monotonically** with support density (1.7237 at
ρ=2.2e-4 → 0.3300 at ρ=2.2e-2 — a 5.2× span across dc1's grid). A stream correcting a fraction
`f` of the residual codes a support at density `f·ρ₀` — **sparser, therefore dearer per flip**.
So `f = 1.0` is the **cheapest** point on the curve, and every partial correction is worse.

**MEASURED** (`experiments/ddm_wd1_correction_scale_threshold.py`, receipt
`.omx/research/ddm_wd1_correction_scale_threshold_20260802.json`; log-linear, **never extrapolated
off-grid**), at the live seg base ρ₀ = 4.31179e-3:

| fraction corrected | flips | B/flip | × water | NET S | % of gap |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 50,864 | 1.5602 | 1.225 | **+0.009722** | *lose* |
| 0.20 | 101,728 | 1.3735 | 1.079 | **+0.006802** | *lose* |
| **0.286 = f\*** | 145,471 | 1.2731 | **1.000** | **0.000000** | break-even |
| 0.40 | 203,456 | 1.1693 | 0.918 | −0.014068 | 1.81% |
| 0.60 | 305,184 | 1.0357 | 0.813 | −0.048252 | 6.22% |
| 0.80 | 406,912 | 0.9330 | 0.733 | −0.092147 | 11.88% |
| **1.00** | 508,639 | 0.8507 | 0.668 | **−0.143049** | **18.45%** |

**Three consequences, each decision-relevant:**

1. **`f* = 0.2860` (ρ_c = 1.2334e-3) is a hard floor.** Below ~29% of the residual, **no**
   correction stream pays, at any solver quality. (Cross-check: the break-even *density* is
   identical for both bases — 1.2334e-3 — as it must be, since the price depends on density alone;
   `f*` differs only because the bases do. Independent agreement with dc1's registered band edge
   **1.285e-3**, 4.0% apart, from a different construction.)

2. **There is NO interior optimum.** Net S is monotone-improving in `f` all the way to 1.0. The
   correction strategy is **all-or-nothing** — every flip is worth exactly the same S, and the
   per-flip price only improves as you take more. "Correct the most valuable flips first" is not
   available as a strategy on this axis.

3. **QA03's verdict is SCALE-CENSORED, and this is a second, independent censoring.** QA03
   addressed 1,866 / 508,639 flips = **0.3669%** of the residual = **0.0128× f\***, i.e. **78×
   below the minimum viable scale**. Its density (1.58e-5) is **14× below dc1's measured grid** —
   I refuse to extrapolate a price there, but the direction is not in doubt: it is far into the
   region where B/flip exceeds water.

   **This is distinct from — and compounds with — dc1's finding.** dc1 measured that QA03's
   *solver* was censored (the `--max-quanta 4` cap fired on 42.5% of instances, producing 64.7% of
   realized flips). I measure that QA03's *experiment scale* was below the break-even threshold by
   78×. **Even a perfectly converged QA03 would still have lost**, because at 0.37% of the residual
   the arithmetic cannot come out positive. The uncapped re-measure dc1 queued
   (§4.3, ~30–60 min on one scorer slot) will therefore report **more flips and still a loss** —
   and that outcome must not be read as a third confirmation that corrections are priced out.

**⇒ The `do_not_spend` mark on the corrections pool rests on four "confirmations" (QA03, QA04, and
two prior white-jitter rows) all taken at small scale. Small-scale correction probes are correctly
priced out. That says nothing about the full-residual stream, which is the only regime where the
strategy can pay — and where it measures −0.143 S = 18.45% of the gap.**

**Caveat, stated because it bounds the claim.** The fraction curve assumes a sub-support at
density `f·ρ₀` has the coherence of dc1's margin-threshold-selected support at that density — the
same family the curve was measured on. A **deliberately spatially-clustered** selection would code
cheaper, which would **lower** `f*`. So `f* = 0.2860` is an **upper bound** on the break-even
fraction under margin-threshold selection, not a universal floor. That is a further open door, not
a weakening.

---

## §5 A DENOMINATOR ERROR IN MY OWN DISPATCH

My charter states *"the gap to the bar is 0.7754681 against the live own-vehicle frontier
0.9352823."* Those two numbers are **inconsistent**:

```
0.9476091 - 0.172141 = 0.7754681   <- the quoted gap belongs to THIS frontier (pw1)
0.9352823 - 0.172141 = 0.7631413   <- what the quoted frontier would imply
```

The **gap is right**; the **frontier figure is not**. `0.9352823` appears in no artifact I read
(`main_hot_state.md` POINTER_LINE records the line as
`20.27 → 1.5343 → 0.992972 → 0.9639878 → 0.9476091`). I used **0.9476091 / gap 0.7754681**
throughout and flag the discrepancy rather than silently picking one — a mis-stated denominator is
exactly the class this campaign has been fighting, and it reached me inside the instruction to
report denominators.

---

## §6 WHAT I DID NOT DO, AND COULD NOT CHECK

- **I ran no scorer job.** The n600 slot is occupied (pid 18732). The one measurement that would
  close #832 — dc1's uncapped QA03 re-solve (§4.3, resumable from `qa03_instances.jsonl`) —
  remains unrun, and **§4 now predicts its outcome in advance**: more flips, still a loss, for a
  reason that is not the solver's. Pre-registering that prediction is the point; if it converges to
  a *win* at 0.37% of the residual, my §4 is wrong.
- **I did not re-run dc1's coder.** My §3/§4 numbers are arithmetic over dc1's receipt. If that
  receipt is wrong, so am I. Its controls (positive/negative/round-trip) all pass and it reproduces
  pp1's registered `b_per_err_best` to 4 dp, which is why I consumed it.
- **I did not read `ddm_hl1`** (the rank-4 / `lane_guard.py:64-65` literals) — owned elsewhere; it
  is the 1 row of the inventory of 10 I make no claim about.
- **I did not resolve dc1's interpolation method** (§3 ii). Flagged, not concluded.
- **Consumers outside `src/`, `tools/`, `experiments/`** are outside my scan. Read every "N hits"
  in §2 as *"N in the scanned roots"*, never as a bare negative.
- **Transitive reachability.** §2's negatives are direct-import negatives. A deep transitive edge
  into the cosine basis is not excluded, though the live chain sets `FRAME0_POLICY` explicitly,
  which makes one unlikely.
- **I did not verify pw1's ΔS.** I consumed `0.9476091` as the live pointer from `main_hot_state`
  and the pw1 memo; I did not re-run its A/B.

---

## §7 CROSS-FINDINGS

**→ the `p0_864` owner.** The inventory of 10 is now **0 confirmed + 1 unread (`ddm_hl1`) + 9
adjudicated-not-this-class**. `built_elsewhere_unwired()` = 0. The operator's diagnosis of the
*class* was right; the *population* attached to it was assembled from prose that three separate
arms have now each found wrong at source. **Recommend the P0 be closed on population-empty, with
wt1's refusal clause left armed as the standing detector.**

**→ `ddm_gd5`.** wr2 supplied 7 negative controls; **I supply 2 more, and a sharper predicate.**
Reachability is not the discriminator — *direction* is. Both pose rows are import-unreachable from
the live entry point AND measured 3,400× worse. A detector keyed on reachability calls that a
wiring backlog. The predicate must be **signed**: a live mechanism with an unwired successor whose
measured number is **better on a comparable surface**. wt1's `measured_comparison` field is the
right place; nothing currently checks its **sign**.

**→ `ddm_dc1`'s owner.** (a) The 0.883–0.915 B/flip range is ~0.03 dearer than any interpolation
of your own receipt (§3 ii) — your win is understated. (b) Your §4.3 re-measure now has a
pre-registered prediction it must beat (§4.3 above).

**→ `ddm_ba31`'s owner.** Your §B.3 caveat (a) — *"this prices the slope, not a realized move"* —
is now DERIVED into a number: `f* = 0.2860`. It is the most decision-relevant line in §B.3 and it
was the one left unquantified.

**→ whoever routes the corrections pool.** The `do_not_spend` mark should be re-scoped from
*"corrections"* to *"correction probes below ~29% of the residual."* Those two are not the same
verdict, and the four confirmations on record support only the second.
