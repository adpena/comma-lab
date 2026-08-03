---
schema: ddm_qd2_rebaseline.v1
date_utc: 2026-08-03
arm: ddm_qd2 (re-price every banked ΔS against the live best; make staleness detectable)
lane_id: "lane_ddm_qd2_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false   # exact contest pointer 0.19108 UNMOVED. No new eval was run by this arm.
verdict_scope: CLASS   # the defect is in how claims are CARRIED, not in any one claim
axis: "[macOS-CPU advisory - real evaluator receipts, recomputed from components] NON-PROMOTABLE.
  Every number below is recomputed from an existing report.txt or from the executable gap
  equation. NO new scorer run, NO training, NO paid dispatch, NO pointer mutation."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/*/report.txt  (14 real evaluator rows)
  - tac.canonical_equations.gap_decomposition_against_floor_20260802
  - .omx/research/ddm_qd1_backlog_drain_20260803.md
  - .omx/research/ddm_cr1_composition_row_827_20260801.md
  - .omx/research/ddm_cr2r_ep854_pose_resolve_refuted_matched_control_20260802.md
  - .omx/research/ddm_rh1_rate_harvest_20260801.md
  - .omx/research/ddm_cx1_pj2_container_compose_20260802.md
  - .omx/state/canonical_task_status.jsonl
consumers: [MAIN]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_qd2 — every banked ΔS, re-priced against `cx1`

## §0 ANSWER FIRST

**#826 is not an incident. It is one of SIX distinct staleness mechanisms, and the charter that
sent me to sweep for them was itself carrying one.**

The single sharpest result: **the "1% of gap = 10,907 B" constant in my own charter is the
`dc1_fold`-era gap. Against `cx1` the gap is 0.6543559 and 1% is 9,827.2 B — the handed constant is
11.0% too large.** Recomputed with the executable equation, not re-typed.

**The §1 prediction was HALF RIGHT, and wrong in the more interesting half.** #826, #827, #881
and #853 do all sit on the `cell_drop50` axis — but **with opposite polarity**. #826 has
`cell_drop50` as its **SUBJECT** (so it *inherits* the +5,413 B penalty → INVERTS). #827/#881/#853
have `cell_drop50` as their **REFERENCE** (so they *shed* it → they SHRINK by exactly the same
0.0036043, and stay large). They do not inherit the penalty; they were being flattered by it.

**But that re-pricing is moot, because the row underneath them is already measured at full S and it
is a catastrophe.** `v4d_cr2_ep854/report.txt` — the #827 composition, 285,529 B — recomputes to
**S = 20.0465770**, i.e. **+19.2200798 against `cx1`**. The advertised "−0.0866 S UNLOCK" is a
**seg+rate-only partial composite**; its missing pose term is measured at **+19.302316**, which is
**234.7× the seg+rate prize it was advertising** (0.0822362, measured leg).

**And the ledger's only live nonzero own-vehicle ΔS is stamped on three rows and is already spent:**
`−0.0675451` appears under **#850, #873 and #882** — three *distinct* task scopes (GN termination /
menu-as-RD-codebook / start-is-the-lever) all closed by the **one** `pj2` run, each row stamped with
that run's **total** delta. It is `pj2`'s pose win, which is already inside `cx1`. Summing the ledger
banks `−0.2026353` for a delta whose true re-priced value is **0.0000000**.

**Live count of the structural defect: 7 of the 8 ledger rows carrying `actual_delta_s` do not name
the baseline they were measured against.** The contract at `contract.py:160` already refuses an
`actual_delta_s` without an `[empirical:<path>]` source — it is **one clause short** of also
requiring the baseline. §4.

### The ranked table

`cx1` = **353,808 B**, seg 0.4311790 · pose 0.1597320 · rate 0.2355862 · **S 0.8264972** —
verified in §1 from its own `report.txt`, not from a memo.

| # / row | advertised ΔS | baseline it was measured against | re-priced vs `cx1` | verdict | why |
|---|---:|---|---:|---|---|
| **#826** `gr1_cell_drop50` | −0.0983195 | v4d-era ref `0.7685479` (seg+rate) | **+0.0034632** | **INVERTED** | subject inherits its own +5,413 B; buys 166.4 flips at 32.52 B/flip = **25.54× `W`** |
| **#827 / #881** ep854 × cell_drop50 | −0.0867981 (cr1) / −0.0865743 (uv1) / −0.035996 (registered) | `gr1_cell_drop50` seg+rate 0.6703695 | **−0.0822362** on seg+rate | **SHRANK, then REFUTED** | shrinks by 0.0036043 (reference's own deficit); then the measured full-S row is **+19.2200798** |
| **#853** "−0.0499 S pose-independent rate half" | −0.0499214 | component of #827's delta | **NOT-COMPARABLE** | **REFUTED (rh1 §4) + REPRESENTATION CHANGED** | 99.0% carried by the token *event* field = the renderer input, so not pose-independent; the genuine lever is −0.0018518 (27× smaller) and was measured on v4d's **SMEVR** field, while `cx1` ships **IX2TOK01** |
| **#850 / #873 / #882** pj2 pose | −0.0675451 (one run's total, stamped on 3 distinct scopes) | `ms8` 0.8984335 | **0.0000000** | **ABSORBED, triple-stamped** | `cx1` = `pj2 × ix2`; `d_pose` identical to 8 digits. Three scopes closed by one run each carry the run total ⇒ summing triple-counts |
| `ms8` −0.0491770 | −0.0491770 | `pw1` | **0.0000000** | **SUPERSEDED-NOT-ADDITIVE** | pj2 measured it a *reparameterization* of the same coordinate — the two are alternatives, not addends |
| `pw1` / `mq1` / `dc1_fold` chain | −0.0163786 / −0.0123270 / −0.0000560 | each other | **0.0000000** | **ABSORBED** | the whole v4d→cx1 chain sums to **−0.1374906**, which *is* `cx1 − v4d`. Not re-earnable |
| burn-lineage `S_add` rows (rung-1 **−0.0651**, burn-4 w02 **−0.018303**, ep854 endpoint **−0.013558**) | as listed | within-burn checkpoints | **NOT-COMPARABLE** | **PARTIAL COMPOSITE** | all are seg+rate `S_add` on the TR1/ep854 base whose one measured full-S row is 20.0466. Real as *seg training progress*; not S deltas against the live vehicle |
| every "% of gap" figure in the corpus | — | six different frozen gaps | **GREW as a fraction** | **DENOMINATOR DRIFT** | §3a |

**Nothing in this table is a new win. The honest total of re-priceable banked ΔS against `cx1` is
0.0000000.** Everything real is already inside `cx1`; everything outstanding is refuted, stranded,
or partial.

---

## §1 THE DENOMINATOR, VERIFIED FIRST

The charter warned that if my `cx1` numbers are wrong the whole table is wrong, and that MAIN's own
error today was a wrong denominator twice. So `cx1` is re-derived from its own receipt before
anything divides by it.

`/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_cx1_pj2ix2/report.txt`:

```
  Average PoseNet Distortion: 0.00255143
  Average SegNet Distortion:  0.00431179
  Submission file size: 353,808 bytes
  Final score: ... = 0.83          <- rounded; cannot distinguish cx1 from pj2 at all
```

Recomputed from components: `100·0.00431179 + sqrt(10·0.00255143) + 25·353808/37545489`
= `0.4311790 + 0.1597320 + 0.2355862` = **0.8264971874742499**. Matches the `cx1` memo to 1e-7.
**The denominator is sound.**

The full own-vehicle lineage, every row recomputed from its own `report.txt` (14 evaluator reports
parsed; 14 matched; 0 unparsed):

| submission | bytes | d_seg | d_pose | segC | poseC | rateC | **S (recomputed)** | printed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v4c_static_photo_celldrop50` | 359,750 | 0.00431179 | 0.01038450 | 0.431179 | 0.322250 | 0.239543 | 0.9929717 | 0.99 |
| `v4d_refine_celldrop50` | 360,238 | 0.00431179 | 0.00858145 | 0.431179 | 0.292941 | 0.239868 | 0.9639878 | 0.96 |
| `v4d_pw1` | 360,323 | 0.00431179 | 0.00764555 | 0.431179 | 0.276506 | 0.239924 | 0.9476092 | 0.95 |
| `v4d_mq1_partial` | 360,702 | 0.00431179 | 0.00696572 | 0.431179 | 0.263927 | 0.240177 | 0.9352822 | 0.94 |
| `v4d_ms8` | 360,374 | 0.00431179 | 0.00516636 | 0.431179 | 0.227296 | 0.239958 | 0.8984335 | 0.90 |
| `v4d_dc1_fold` | 360,309 | 0.00431179 | 0.00516578 | 0.431179 | 0.227284 | 0.239915 | 0.8983775 | 0.90 |
| `v4d_pj2` | 360,406 | 0.00431179 | 0.00255143 | 0.431179 | 0.159732 | 0.239980 | 0.8308905 | 0.83 |
| **`v4d_cx1_pj2ix2`** | **353,808** | 0.00431179 | 0.00255143 | 0.431179 | 0.159732 | 0.235586 | **0.8264972** | 0.83 |
| `v4d_cr2_ep854` (#827) | 285,529 | 0.00394407 | **37.87713242** | 0.394407 | **19.462048** | 0.190122 | **20.0465770** | 20.05 |

**A structural fact that falls out and that no per-row memo states:** `d_seg = 0.00431179` is
**bit-identical across all eight** rows from `v4c_static_photo_celldrop50` to `cx1`. Every own-vehicle
move since 2026-07-30 has been **pose or rate**. The seg axis has not moved by one ULP — which is why
its share of the gap has *risen* even as S fell (§3a).

---

## §2 THE §1 PREDICTION — HALF RIGHT, AND THE WRONG HALF IS THE INFORMATIVE ONE

Parentage established from the artifacts (byte counts and token-field sizes), not from the row text,
because row text is exactly what was stale in #826.

`.omx/research/ddm_cr1_composition_row_827_20260801.md:80` is the load-bearing arithmetic:

```
gr1 cell_drop50   tokens 346,478 B (SMEVR)   composed 359,221 B   rate 0.2391905
ep854 burn        tokens 271,505 B (SMEVR)   composed 284,248 B   rate 0.1892691   (DERIVED)

gr1    seg 0.4311790 (exact) + rate 0.2391905 = 0.6703695
ep854  seg 0.3943024 (advisory) + rate 0.1892691 = 0.5835714
DELTA  -0.0867981  =  seg -0.0368766  +  rate -0.0499214
```

⇒ **`cell_drop50` is the REFERENCE of #827/#881/#853, and the SUBJECT of #826.** Opposite polarity.
The charter's phrase "share the `cell_drop50` rate parent" is true as a dependency but inverts the
consequence: a subject *inherits* its reference's deficit, a candidate *sheds* it.

- **#826 (subject):** `0.6702284 - 0.6667652 = +0.0034632`. Reproduces `qd1` exactly. **INVERTED.**
- **#827/#881 (candidate):** the delta shifts by exactly the reference's rate deficit,
  `0.2391905 - 0.2355862 = +0.0036043`, so `-0.0867981 -> -0.0831938`. **SHRINKS, does not invert.**

Better still, that leg no longer needs `cr1`'s derived number, because the composition was
subsequently **built and evaluated**: using the exact evaluator row (`285,529 B`, seg `0.394407`),
seg+rate = `0.584529`, so vs `cx1` seg+rate → **−0.0822362 MEASURED**. `cr1`'s derived
`−0.0831938` agrees to 9.6e-4 — a clean cross-check of a DERIVED figure by a later MEASURED one.
*(Incidental calibration: `cr1`'s advisory seg 0.3943024 vs the exact 0.394407 = 1.05e-4 optimism,
consistent with the standing ~1.41e-4 advisory-seg calibration.)*

**And then the missing term.** Both #827 and #881 advertise a **seg+rate-only** number. The pose term
was measured on 2026-08-02:

| | seg | pose | rate | **S** |
|---|---:|---:|---:|---:|
| `cx1` | 0.431179 | 0.159732 | 0.235586 | **0.8264972** |
| `v4d_cr2_ep854` (#827) | 0.394407 | **19.462048** | 0.190122 | **20.0465770** |

**Re-priced full-S: +19.2200798.** The pose penalty (**+19.302316**) is **234.7× the seg+rate prize
(0.0822362)** the row was advertising — both legs from the same measured row. *(On `cr1`'s derived
leg 0.0831938 the ratio is 232.0×; quoting 232.0 against the measured 0.0822362, as this memo's first
draft did, is exactly the mixed-convention slip §5(c) warns about, caught in round-1 review.)*
`ddm_uv1` rejected it by pre-registered arithmetic and
`ddm_cr2r` proved with a **matched control** on 74 shared pairs that the defect is the **base**, not
the solver (celldrop50 base mean `d_best_static` 0.0778 vs ep854 base 11.5904; ep854 better on
1/74). #881's premise — *"pose must be re-solved against ep854"* — **was executed and refuted**;
that is a stronger disposition than staleness.

**#853** is the rate component of that same delta. `ddm_rh1` §4 already adjudicated it **NO**: 99.0%
of the −0.0499214 is carried by the token **event** field, which *is* the renderer input, so
harvesting it means shipping ep854's renders — the same object measured at 6.36× `d_pose`. The
genuinely pose-independent lever `rh1` found is **−0.0018518 (27× smaller)**, and it was measured on
v4d's **SMEVR** token field. **`cx1` ships a different token representation** (`IX2TOK01`, cell-major
nibble, 341,295 B vs v4d's 346,478 B SMEVR), so even the surviving −0.0018518 is **NOT-COMPARABLE**
until re-measured on `cx1`'s field. That is a re-measurement, not a claim.

---

## §3 W IS INVARIANT — THE CHARTER'S PREMISE #2 IS FALSE, AND THE TRUE OPERATING-POINT TERM POINTS THE OTHER WAY

The charter warned that `W = 1.273108215332031` B/flip "may have moved… as the archive shrinks
360,406 → 353,808 B, bytes get *dearer* relative to flips." **Derived, that is false.**

```
neutrality:  25·dB/DEN  =  100·d(d_seg)          flips = d(d_seg)·PX,  PX = 600·512·384
             dB = d(d_seg)·100·DEN/25
      W  =  dB/flips  =  (100 · 37,545,489 / 25) / 117,964,800  =  1.2731082153320312
```

Reproduces the banked constant to 1e-12. **There is no archive-size term.** The rate leg is linear in
bytes and the seg leg is linear in `d_seg`; a ratio of two linear terms has no operating point.
**`W` is exactly invariant and every B/flip comparison in the table stands unchanged.**

The cross-check the charter asked for **agrees**: #826's realized rate is
`5,413 B / 166.4 flips = 32.52 B/flip = 25.54 × W`, and the required byte cut is the same 25.54×
(5,413 → **≤ 211.9 B**). MAIN's 25.55×, `qd1`'s 25.5×, and this derivation are one number.

**The operating-point sensitivity is real, but it lives in the POSE term, and it points the opposite
way.** `poseC = sqrt(10·d_pose)` is concave, so `dS/d(d_pose) = 5/sqrt(10·d_pose)` **rises** as pose
improves:

| row | `d_pose` | `dS/d(d_pose)` |
|---|---:|---:|
| `pw1` | 0.00764555 | 18.0828 |
| `ms8` | 0.00516636 | 21.9977 |
| **`cx1`** | **0.00255143** | **31.3024** |

**A pose distortion reduction banked at the `pw1` operating point is worth 1.73× more S today
(1.42× from `ms8`).** So the correct general rule is the opposite of a single caveat:

> **Banked BYTE deltas are exactly invariant. Banked SEG deltas are exactly invariant. Banked POSE
> deltas GROW as pose improves — a pose ΔD is worth more now than when it was measured, and any
> pose lever parked at an older operating point is UNDER-priced, not over-priced.**

That is a re-pricing that finds value rather than removing it, and it is the one direction nobody
has been checking.

### §3a Denominator drift — the mechanism that caught my own charter

`% of gap` is not a property of a claim; it is a property of a claim **and the instant it was
written**. Six different gap denominators are frozen across the corpus:

| era | gap to bar | 1% of gap | a claim of "10.0% of gap" written then = |
|---|---:|---:|---:|
| v4d | 0.7918465 | 11,892.1 B | 12.10% now |
| pw1 | 0.7754679 | 11,646.1 B | 11.85% now |
| (DAG row) | 0.7631413 | 11,461.0 B | 11.66% now |
| `ms8`/`dc1_fold` | 0.7262362 | 10,906.8 B | 11.10% now |
| `pj2` | 0.6587492 | 9,893.2 B | 10.07% now |
| **`cx1` LIVE** | **0.6543559** | **9,827.2 B** | 10.00% |

**The charter handed me `total gap 0.7262358, 1% = 10,907 B`. That is the `dc1_fold` row
(0.8983775 − 0.1721413 = 0.7262362). It is two frontier moves old and 11.0% too large.** Recomputed
via `tac.canonical_equations.gap_decomposition_against_floor_20260802` with `cx1` as `ours` and the
PR130 triple (`d_seg 2.966e-4`, `d_pose 2.331e-5`, `191,052 B`) as `floor`.

**Live gap decomposition against the bar — this supersedes the `dc1_fold`-era figures in `cv1`/`gd4`:**

| axis | gap | share |
|---|---:|---:|
| **seg** | **0.4015190** | **61.4%** |
| pose | 0.1444644 | 22.1% |
| rate | 0.1083725 | 16.6% |

`rank_by_gap() = ('seg', 'pose', 'rate')`. **Seg's share has risen 55.3% → 61.4%** — not because seg
got worse (it is bit-identical) but because the other two axes were the only ones anyone moved.
Note the direction: **fractions GROW as the gap shrinks**, so a "% of gap" claim is *under*-stated by
its own drift, while an *absolute* ΔS is invariant. That is why the absolute belongs in the ledger and
the percentage belongs to the moment.

---

## §4 THE STRUCTURAL HALF — one clause, not a registry, and not a gate

The defect is in how claims are **carried**. #826's caveat existed in `current_focus.md` and was
stripped in transit to the ledger row. So: make staleness **detectable rather than discoverable**.

**The existing surface already has the right shape and is exactly one clause short.**
`src/tac/canonical_task_status/contract.py:160`:

```python
if self.actual_delta_s is not None and "[empirical:" not in self.event_notes:
    raise ValueError("actual_delta_s rows must include an [empirical:<path>] event note")
```

That enforces the **source** of a delta. It does not enforce the **baseline** — and a delta without
its baseline is precisely the unanchored number this arm was sent to sweep. The sister equation
already encodes the correct discipline in its own docstring: *"an unsourced triple cannot anchor a
gap, and a gap without a source is exactly the unanchored-ΔS failure this equation exists to stop."*
**The equation demands custody; the ledger does not. Aligning them is the whole fix.**

**MEASURED live count — 417 ledger rows examined; 8 carry `actual_delta_s`; 7 do not name a
baseline (87.5%):**

| task | ΔS | `[empirical:]` | names baseline |
|---|---:|:--:|:--:|
| `z8_canonical_l28…` | −0.3800000 | yes | **yes** |
| `closed_spec_boundary_solver_v1` | 0.0000000 | yes | no |
| `cross_pair_waterfilled_corrector` | 0.0000000 | yes | no |
| `taskspace_codec::g55_g57…` | 0.0000000 | yes | no |
| `g51_full_n600_quotient_harvest` | 0.0000000 | yes | no |
| **#850** | **−0.0675451** | yes | no |
| **#873** | **−0.0675451** | yes | no |
| **#882** | **−0.0675451** | yes | no |

**The three nonzero own-vehicle rows are three DISTINCT scopes each stamped with the SAME one-run
total, and that total is already inside `cx1`.** A reader summing this ledger banks `−0.2026353` for
a delta worth `0.0000000`. That is the class, live, in eight rows — and note that the per-row
`[empirical:]` path the contract *does* enforce is identical on all three, so the existing invariant
sees nothing wrong.

### The proposal — a convention plus (optionally) one clause

**Convention (zero code, effective immediately, greppable):** an `actual_delta_s` row carries, in
`event_notes` alongside `[empirical:<path>]`, a baseline token:

```
[baseline:<artifact-locator>=<S recomputed from components>]
e.g.  [baseline:eval_root/submissions/v4d_ms8/report.txt=0.8984335]
```

Two properties make it worth anything: the locator is **re-derivable** (a reader recomputes the
baseline's S from its own receipt), and it makes the row **self-describing** — a reader or the
costate SENSE layer can see at a glance that `v4d_ms8` has been superseded twice and that the delta
therefore needs re-pricing, without reading any memo.

**If — and only if — the operator wants it enforced,** the one-clause diff is the same shape as the
line above it, and its **live count is 7**, so it must land WARN-ONLY per the strict-flip atomicity
rule and be flipped after backfill. **I did not land it**, for the reason `qd1` gave when it declined
to build a gate for the `rtk -n 50` class: *the class lives in agent behaviour, so a preflight gate
would have been a point-fix on the wrong surface.* Stripping a caveat in transit is a **writing**
behaviour; the field makes the omission **visible**, which is the honest ceiling of a schema fix.
Building the gate before the convention has ever been used once would be
**built-instead-of-paid** on a surface with no users.

**Explicitly NOT proposed:** a parallel registry, a new ledger, a new state file, or a
recompute-the-world tool. The ledger, the equation and the costate digest already exist; this is one
token inside a field one of them already validates.

---

## §5 ROUND-1 ADVERSARIAL SELF-REVIEW

**(a) The trap named in my charter — my `cx1` numbers.** Addressed first, in §1: `cx1` is recomputed
from its own `report.txt`, not from the memo, and matches to 1e-7. Every lineage row in §1 is parsed
from its own receipt. **Caught by doing it: my first parse of the 14 reports silently truncated
`353,808` to `353` on the comma and read `100` out of `100*segnet_dist` as the final score** — it
produced a plausible, entirely wrong table. Only cross-checking one row against the `cx1` memo
exposed it. A regex that half-matches is the vacuity class wearing a number.

**(b) I trusted a grep that could not return the negative.** `grep -rn "0.7685479"` over
`.omx/state/current_focus.md` returned **nothing** for a string I had read in that file minutes
earlier. I did not conclude "absent"; I re-ran the search in Python and found **20 hits across 61,297
files**, including the two in `current_focus.md`. Under the standing rule that a probe which cannot
return the negative is not a probe, **every "not found" in this memo is a Python `os.walk` result with
a printed denominator, never a grep.** I still cannot say *why* the grep missed; I can only say I
stopped believing it.

**(c) My re-priced #827 leg mixes measurement conventions.** `cr1`'s `0.5835714` uses an **advisory**
seg while `cx1` uses the **exact** evaluator. I therefore quote the **measured** exact-evaluator leg
(−0.0822362) as primary and `cr1`'s derived (−0.0831938) as the cross-check. Both are seg+rate-only,
and neither is the answer — §2's full-S row is.

**(d) The claim I am least sure of.** The burn-lineage `S_add` rows (−0.0651, −0.018303, −0.013558) I
class as PARTIAL COMPOSITE on the strength of ep854's measured `d_pose = 37.877`. Those rows are
*within-burn* checkpoint comparisons, so they are legitimately real as **seg training progress**; my
claim is only that they are **not S deltas against the live vehicle**, because the base they sit on
has no measured pose better than 37.877. If a future joint-descent burn gives that base a legible
pose, they re-enter. `cr2r` explicitly preserves that door
(*"NOT refuted (FAMILY): pose on ep854 as such"*), and I have not closed it.

**(e) Scope honesty.** The corpus sweep examined 7,015 files by grep-pattern and read ~30 in full; it
did **not** read all 465 `ddm_*.md` line-by-line, and `sub015_DAG_*.md` was sampled, not transcribed.
The 14 evaluator receipts are a **complete** enumeration of
`ddm_pfs1_20260729/d1/eval_root/submissions/*/report.txt`. **For anything outside those two scopes the
correct statement is "did not find in the scope I searched," not "there are no others."**

**(f) The task-id join is broken and I worked around it, not through it.** **9 of the 14 seed ids in
my charter are ABSENT from the repo ledger** — #832 #853 #859 #866 #869 #881 #891 #897 #900. Only
#826 #827 #873 #882 #909 are present. The absent ones live in the harness TaskList, which arms cannot
see. I therefore built the population from **artifacts and memo content**, never from bare ids
(#850 — which carries one of the three stamped deltas — is not in the charter's seed list at all and
was found only by scanning the ledger for `actual_delta_s`). Anyone re-running this by id will find
nothing and conclude wrongly; that is a missing JOIN, not an absent row.

**(g) What would change my mind.** If `gr1_cell_drop50` can be re-encoded to within **212 B** of
`cx1` (a 25.54× byte cut) then #826 is net-positive again and the INVERTED verdict flips. That number
is a **specification**, not a kill — it is the one line in this memo that converts a dominated row
into a target.

---

## §6 NEXT-IF-RESUMED

1. **Re-price the pose backlog UPWARD (§3).** The only item here that finds value. Every pose lever
   parked at a `pw1`- or `ms8`-era operating point is under-priced by **1.73×** / **1.42×**. Nobody
   has swept in that direction because everyone assumed staleness only destroys.
2. **Re-measure `rh1`'s −0.0018518 base-rule lever on `cx1`'s `IX2TOK01` field.** The only survivor
   of the #853 family and genuinely pose-independent; NOT-COMPARABLE today only because the token
   representation changed underneath it. Cheap, $0, no scorer.
3. **Collapse #850/#873/#882 to one row** marked ABSORBED-INTO-`cx1`, so no future reader sums it.
   This is the live triple-count.
4. **Correct the `dc1_fold`-era gap constants** in `cv1`'s gap decomposition and the `gd3`/`gd4`
   headers to the `cx1` figures (§3a), *append-only* — noting that seg's share is now **61.4%**,
   which strengthens rather than weakens the standing "seg is the majority of what is left" reading.
5. **Use the convention once before considering the gate** (§4). One hand-written row is the evidence
   that decides whether the clause is worth landing.

**Pointer delta: NONE. The exact contest pointer 0.19108 is UNMOVED, our own-vehicle best is
`cx1` 0.8264972, and this arm produced no new score — it removed a phantom `−0.2026` from the
ledger's apparent bank and found one direction (pose) where banked value is under-counted.**
