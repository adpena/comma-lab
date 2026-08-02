# ddm_uv1 (#881 × #882) — the composition row REJECTS, and the blocker was never the solver. ep854's palette is POSE-ILLEGIBLE: it buys d_seg with the very photometric structure PoseNet reads.

**Arm:** ddm_uv1 · 2026-08-02 · axis `[macOS-CPU frozen-PoseNet advisory]` · `score_claim=false`
`promotable=false` · exact contest pointer **UNMOVED** · **no n600 scorer job fired** (MAIN owns the slot).

**STORES CONSULTED:** `.omx/research/ddm_cr2_composition_row_ep854_base_20260801.md` (§6 pre-registered
falsifier) · `.omx/research/ddm_sv1_solver_termination_sweep_20260801.md` (§2b restart control) ·
`.omx/research/ddm_pw1_pose_menu_saturation_20260801.md` · `.omx/research/ddm_mq1_pose_menu_rd_audit_20260801.md` ·
the live chain `ddm_v4c_resolve.py` / `ddm_v4d_resolve.py` / `ddm_v4d_build_composed_archive.py` /
`inflate_runner_v4d.py` / `stage_v4d_realized_gate.sh` · the four evaluated `report.txt` under
`ddm_pfs1_20260729/d1/eval_root/submissions/` · commit `d7d11ef96f` (sv1's shared solver).
**Deliberately not loaded:** the gc14/gc16/su2 memo bodies, burn telemetry.

---

## 1. Verdict against the pre-registered criteria — stated plainly

cr2 pre-registered, before its gate ran: break-even **d_pose ≤ 0.0131903**; ≤ 0.0076455 realizes the
full **S 0.8610340**; *"above break-even ⇒ the composition stays REJECT."*

**The composition REJECTS.** Not because the pose solve was under-powered — because ep854 cannot carry a
pose payload at all. The seg+rate win is real and stays stranded.

| | live `v4d_pw1` (MEASURED n600) | composed `cr2_ep854` (MEASURED n600) |
|---|---:|---:|
| d_seg | 0.00431179 → 0.4311790 S | 0.00394407 → **0.3944070 S** |
| rate | 360,323 B → 0.2399243 S | 285,529 B → **0.1901220 S** |
| pose | 0.00764555 → 0.2765059 S | 37.87713242 → **19.4621 S** |
| **S** | **0.9476092** | **20.0466** |

seg+rate **−0.0865743 S = 11.16 % of the 0.7754681 gap**, EXACT vs cr2's −0.0866789 prediction. Pose
destroys it 225-fold.

### REJECT is PROVEN by arithmetic, not estimated by extrapolation

Full θ re-solve (single-plane GN + two-plane GN from 3 starts + (a,b) re-fit from 2 starts) on **4 pairs**:

```
break-even total n600 budget      7.914180   (0.0131903 x 600)
ep854, 4 pairs, AFTER re-solve    8.555757   [0.53951, 7.01989, 0.99609, 0.00027]
                                  --------
                                  1.0811x    exceeds the WHOLE budget by 8.1%
```

d_pose ≥ 0, so the other 596 pairs cannot subtract. **n600 mean ≥ 0.0142596 > 0.0131903.** Pair 1 alone
(7.01989) is **88.7 % of the entire n600 break-even budget**. No extrapolation is used.

*Honesty limit:* the live chain adds a dim0-offset refine and beta select on top of what I ran; that
composite stage historically moved this chain 0.010384 → 0.007646 (**−27 %**). Rescuing this row would
require it to deliver **−99.4 %** on these 4 pairs *and* the other 596 to contribute ≈ 0.
`verdict_scope: INSTANCE` — this kills the ep854-base *composition*, not the base-swap family.

## 2. The positive control is what makes the verdict load-bearing

Same solver, same code path, same pairs, same starts — **only the base differs**:

| base | mean d_pose after full re-solve | |
|---|---:|---|
| gr1 cell_drop50 (the base the pose WAS solved against) | **0.000709** | PASS — reproduces in-lineage quality |
| ep854 | **2.138939** | **REJECT — 162× over break-even** |

**3,019× separation on identical machinery, with the control passing.** This is a property of the base,
not a bound, not a start, not a menu. Had the control failed I would have had a broken instrument and no
verdict — and it did fail once (§5).

## 3. The MECHANISM (this is the finding, and it generalizes)

The rendered `frame_1` of the two bases are **not the same image**:

| | gr1 cell_drop50 | ep854 |
|---|---:|---:|
| per-channel mean | [66.9, 36.5, 45.9] | [78.1, 112.1, 105.7] |
| whole-frame mean / sd | 49.8 / 40.6 | 98.6 / 38.4 |
| **Pearson corr(gr1, ep854)** | — | **+0.119** |
| best global affine, residual | — | g=+0.113, c=+93.0, resid mean&#124;·&#124; 28.2 |
| mean&#124;ΔF1&#124; · frac pixels changed | — | **62.7 / 255 · 99.7 %** |

A correlation of **+0.119** is structural noise. Yet ep854's d_seg is *better* (0.00394 vs 0.00431). Both
facts are true because **the LOTTO renderer paints a task-space palette, not a photometric
reconstruction**, and the two readouts are not the same functional:

- **SegNet** consumes only the **argmax** — a palette may move arbitrarily far while the argmax partition
  is preserved or improved. d_seg is invariant to exactly the deformation that occurred.
- **PoseNet** consumes **dense photometric correspondence** between `f0 = warp(f1)` and `f1`. That
  correspondence is destroyed by the same deformation.

ep854's later training therefore descended a d_seg gradient **straight through the pose-legible
manifold**. The warp geometry still transfers — identity `f0:=f1` scores **173.43 on both bases**
(identical to 5 decimals: 173.429257 vs 173.430419 — the ego-motion is a property of the scene, not the
palette), and the transplanted warp still recovers 173.4 → 28.0, i.e. **84 % of the geometric work**.
What does not transfer is PoseNet's *readout* of that warp.

**This is the photometric wall CLAUDE.md §"Pose is SOLVED" already names** — *"frames trained on seg
alone do not carry pose-legible photometric signal"* — measured for the first time at the **base-swap**
surface, and sharpened: it is not merely that seg-only training *fails to add* pose legibility, it is
that seg-only training **actively spends** it. Pose legibility is a consumable the seg objective is free
to burn.

**Consequence for the vehicle line (the transferable part):** any future base promoted on seg+rate alone
inherits this risk. The cheap guard is now known and costs no scorer time — **corr(f1_new, f1_incumbent)
and the per-channel means are computable from two renders in seconds**, and corr ≈ 0.12 predicted a
2,871× pose miss here. A base-swap candidate should carry that number *before* a pose campaign is priced.

## 4. What was built (debt paid on the EXISTING live surface)

**`experiments/ddm_v4c_resolve.py`** — three changes, no new surface:

1. **`resolve_base()` + `--base-archive`** — `BASES` was a hardcoded 2-entry dict, so the solver could
   only ever solve against bases that existed when it was written, while the **builder had already been
   parametrized by cr2**. That one-sided bridge could produce a transplant but never the re-solve the
   transplant needs. Adding a third literal was the alternative and was **rejected — it reproduces the
   same wall one row later**. Both refusal paths (unknown label / missing archive) are tested.
2. **`derive_ab_starts()` + `ab_multistart_gn()`** — the DERIVED restart policy sv1 asked for, replacing
   its explicitly-generic 5-point displacement control. Starts cost **zero scorer evaluations**:
   `neutral` (shipped), `moment_match` (the `d_ctrl`-implied exposure — the affine carrying f0's first
   two moments onto f1's), `neighbour`, `sel_median`. **Opt-in** via `--ab-starts`; `neutral` is the
   default and is byte-identical, guarded by `test_neutral_only_is_byte_identical`. The receipt carries
   `rungB_ab_start_census` so "held but never fired" is visible — default-off is a *tracked* state.
3. **Deleted the dead `RS_BETAS`** — an unconsumed copy of `ddm_qa44_photometric_rungs_probe.py:78`
   (live there at :189), verified by exhaustive `git grep`. Zero consumers ⇒ zero selections ⇒ the
   registered discriminator returns `UNDETERMINED_EMPTY` (*"VACUOUS scope, never a CLOSED verdict"*). It
   could never have been audited — only **mistaken for an audited menu**. This is the vacuity-reads-as-
   pass genus at the menu surface.

**`src/tac/tests/test_ddm_uv1_base_resolution_and_ab_restarts.py`** — 19 tests (44 with sv1's suite).

**MEASURED restart effect** (8 probe pairs, ep854): best-of-derived beat neutral-only on **5/8** pairs,
7.833291 → 6.911169 (**−11.8 %**), winners `moment_match` 2 / `sel_median` 1 / `neighbour` 2 / neutral 3.
**sv1's start-bias finding reproduces on a third independent surface**, and every start still terminated
on a bound — so it is not a bound effect. **NOT yet measured on the live gr1 chain**; that is a scorer
campaign and is staged, not self-fired.

## 5. Menu audit via the registered discriminator (`ddm_pw1_menu_saturation_discriminator_v1`)

**`RS_GLOBAL_G`** (v4c rung-A, n=600): occupancy {0.0: 415, 0.5: 84, **1.0: 101**}, mass {3.613, 0.503,
**1.604**} → **SATURATED** both ways. Terminal mass fraction 28.0 %, **mass ratio 3.19 vs count ratio
1.20 — a 2.66× disagreement** that independently reproduces pw1's measured "count is the wrong weight"
false negative (they saw ~2×). Both readings honestly self-report `sufficient_for_verdict: false`
(3 entries, no interior trend).

**The LIVE shipped menu** (13-entry signed table, post-pw1, `sufficient_for_verdict: TRUE`) — a table
bounded at both ends, so called twice:

| bound | verdict | mass ratio | count ratio | terminal mass |
|---|---|---:|---:|---:|
| top (+4.5) | **SATURATED** | 16.52 | 4.00 | 0.22 % |
| bottom (−7.5) | **CLOSED_INTERIOR_OPTIMUM** | 0.906 | 1.00 | 1.05 % |

**Asymmetric: pw1's Swann bracket closed the negative side; the positive side still binds.** But the
terminal bin holds only **0.22 % of pose mass**, which prices the entire remaining top-bound gain at
**ΔS ≈ −0.0003 (0.04 % of gap)** — i.e. **SATURATED, and NOT worth a campaign**. Recorded because the
law's own contract says a SATURATED verdict *"is a request for ONE measurement, not a claim that freeing
it will pay"*; here the measurement is the mass fraction, and it says no.

## 6. My own round-1 adversarial review — defects in my own work

- **My first probe's positive control FAILED and I stopped rather than patched.** Reproducing the
  transplant from `final_pw1.jsonl`, the gr1 control gave 1.939 where the gate measured 0.00765 — a
  2,460× miss on the *control*. The bug was mine: `beta_idx` in that JSONL is the **arm's** indexing, and
  the builder remaps it; indexing the 13-entry manifest table with it resolved `beta_idx=0` to **−7.5**
  instead of 0.0. Had I trusted the first run I would have reported a fabricated mechanism. Fix: drive
  the **actual receivers on the actual evaluated archives** — zero transcription — after which the
  control reproduced the shipped per-pair `d_final` to 6 decimals.
- **I nearly made a false negative-existence claim.** I observed `tr1_metadata` "missing" from ep854's
  manifest and began building a story on it; it was present — my `sorted(keys)[:14]` slice truncated it
  alphabetically. Caught by printing the membership test instead of the slice. The `RS_BETAS` claim was
  therefore made only after an **exhaustive `git grep`**, not a single directory scan.
- **My first restart test was vacuous.** It used a separable quadratic — which Gauss-Newton solves
  exactly from *any* start, so every start ties and the test would have passed against a wrapper that
  ignored all starts. Replaced with a bimodal fixture (the regime sv1 actually measured: *"the per-start
  spread is large and multi-modal"*) plus an explicit non-vacuity assertion.
- **A stale pointer in my own error message.** The refusal text referenced `--base-label`, which exists in
  the *builder* but not in this resolver. Corrected.
- **Would my tests pass if the code were broken?** `test_neutral_only_is_byte_identical` is the
  load-bearing one — without it "opt-in" is a claim, not a property. `test_start_census_reads_untraced_
  rows_as_neutral_not_as_a_win` exists because the opposite default would silently inflate the win rate
  on every pre-policy cache.
- **Scope I did not close:** the derived restart policy is measured on ep854 (the rejected base) only.
  Its value on the live gr1 chain is **unmeasured**, and I say so rather than inheriting sv1's number.

## 7. Nothing staged — and why that is the right outcome

`stage_v4d_realized_gate.sh:3` forbids self-firing, and I did not fire. **I also have no candidate worth
MAIN's slot**: the REJECT was reached at ~4 minutes of local scorer time with a passing control, and
firing an n600 gate on a row already proven above break-even would spend the scarcest resource on a known
answer. The slot stays free for a candidate that can move the pointer.

**The stranded prize is unchanged and named:** −0.0865743 S of *measured* seg+rate (11.16 % of gap) is
available from ep854's token field to any carrier that does not require pose-legible photometry — or to
a base trained with a pose term in the loop. That is the next question, and it is a *training* question,
not a solver one.

## 8. Receipts

- `/private/tmp/.../scratchpad/uv1_theta.json` — L3 re-solve, both bases (the REJECT arithmetic)
- `/private/tmp/.../scratchpad/uv1_dof_price.json` — DOF pricing + restart winners
- probes: `uv1_localize.py` (failed control, kept), `uv1_localize2.py` (receiver-driven, control passes),
  `uv1_localize3.py` (mechanism), `uv1_dof_price.py`, `uv1_theta.py`
- evaluated gates re-read at source: `submissions/{v4d_pw1,v4d_cr2_ep854,v4d_refine_celldrop50,v4c_static_photo_celldrop50}/report.txt`
