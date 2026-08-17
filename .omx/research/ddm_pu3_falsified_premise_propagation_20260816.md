# ddm_pu3 — the uncapped pose solve was already built, already priced, already REFUSED. The live defect is that its dead premise is still being written into charters.

- arm: `ddm_pu3_uncapped_pose_solve` (pose axis). Charter target: "locate the cap, measure what
  uncapping buys, build the uncapped solve, price it."
- date: 2026-08-16 · cost **$0** · no dispatch, no governed launch, no n600 scorer job.
- axis: `[contest-CUDA T4, n600]` for every frontier component; arithmetic is exact float.
  `score_claim=false`, `promotable=false`. **Pointer UNMOVED.**

## ANSWER FIRST

1. **Every one of my four charter premises was dead before I was spawned.** The cap was removed
   2026-08-01 (`f13ffdf4b3`); the question was closed four times on three vehicles; the uncapped
   solve exists, converged, was dispatched, and was **REFUSED at +1.686e-02 S** on T4 ~18 h before
   my spawn. I did not rebuild it.
2. **The headline defect is not the solve — it is the premise.** "Still descending 13–23% per
   iteration" comes from **one n=1 STALE_REHEARSAL receipt** whose own `authority_blocker` field
   reads *"Pinned frozen PoseNet was evaluated on exactly one stale composed pair."* It carries
   `score_claim: false`, `production_accepted: false`. It has been **falsified four times** at
   larger n — **1.2%, 1.2%, 0.1549%, 0.07%** — and has still propagated into **seven charters over
   three weeks, two of them written *after* the third falsification**, including mine.
3. **Why it keeps recurring is structural and I measured it.** The existing spawn-site leg
   `_lint_stale_numbers` extracts only literals of **≥4 digits** (`\b\d[\d,]{3,}`) — a necessary
   precision guard, because the index it reads is auto-scraped with 11,840 rows whose
   `refuted_value` entries go down to `"0"` and `"6"`. **A falsified percentage is structurally
   invisible to it.** Loosening that regex is the wrong cure: it would fire on any charter
   containing "2".
4. **CURE BUILT AND WIRED** — a small curated registry plus a sixth advisory leg at the spawn site.
   11 tests. It fires on my own charter's text.
5. **The pose axis is converging to a negative across six *distinct* mechanisms** (not one defect
   repeated). Ceiling **−0.008294577 S = 86.43% of the gap**; no mechanism has reached any of it,
   and the one that shipped made pose **8.93× worse**.

**STORES CONSULTED:** `ddm_pv1_pose_floor_and_admission_bar_20260816.md` ·
`ddm_ps1u_uncapped_pose_solve_20260816.md` · `ddm_ps1u_r2_dual_axis_pose_verdict_20260816.md` ·
`ddm_pg1_pose_gn_convergence_20260802.md` · `ddm_ss1_selection_vs_search_20260803.md` ·
`ddm_sv1_solver_termination_sweep_20260801.md` · `ddm_pw1_pose_menu_saturation_20260801.md` ·
`ddm_gc17_from_here_gradient_not_coordinates_20260801.md` · `ddm_p3v2_optimal_form_pose_resolve_20260729.md` ·
`ddm_pk3/pk4` · `ddm_qs5` · `ddm_js1/js8` · `ddm_ra3` · `ddm_fb1` · `ddm_eg1_pose_gn_rehearsal_20260728.json` ·
`.omx/state/canonical_task_status.jsonl` (#850) · the hv1 r2 T4 receipt.

---

## 1. The four charter premises, re-derived at source

| charter claim | status | evidence |
|---|---|---|
| "hard-capped at 2–3 relinearizations" | **FALSE — cap removed 2026-08-01** | `src/tac/optimization/terminal_pose_gn.py:535` is `_integer(..., minimum=2)` with **no maximum**; commit `f13ffdf4b3` |
| "NO convergence test" | **FALSE** | same commit: stop-on-rejection *proof* (`:517-534`) + `marginal_value_floor` soft stop (`:498-511`) |
| "still descending 13–23%/iter" | **FALSE — n=1 stale rehearsal, falsified 4×** | §2 |
| "nobody has run it past its cap" | **FALSE — run, dispatched, refused** | `ps1u` r2, T4, `+1.686e-02 S` |

Closed **four** times, independently: `ddm_sv1` (08-01, "STALE — already cured, and off the live
chain"), commit `f13ffdf4b3` (08-01), `ddm_pj2` task #850 completion (08-03), `ddm_pv1` (08-16,
n=50: **0.1549%** forfeited, **0/50** pairs stop on any cap).

**One thing I checked that looked like a live finding and is not.** `tools/pb1_terminal_pose_gn_600.py:111`
still defaults `--relinearizations 2` — the floor — after the library ceiling was removed. That is
*not* an unwired-successor defect: `ddm_pw1` re-derived at source that **the tool is not live**
("`pb1` is unreferenced by any v4d script"), and `ddm_sv1` confirms the whole `terminal_pose_gn`
chain is off the live path. Dead code. Reporting it as a finding would have been a fake.

---

## 2. The propagation defect — a non-authoritative n=1 number, laundered

**Origin.** `.omx/research/ddm_eg1_pose_gn_rehearsal_20260728.json`. Two solves, `max_pairs: 1`,
`--relinearizations 2`. The 13–23% figures are the *final-step* `pose_mse` drops:

| solve | iter | pose_mse before → after | Δ% |
|---|---|---|---:|
| `frozen_posenet_rehearsal` | 1 | 1.55612364 → 1.35045947 | **−13.2164%** |
| `mechanism_canary` | 1 | 0.35899379 → 0.27556262 | **−23.2403%** |

Its own fields: `authority_mode: "STALE_REHEARSAL"` · `score_claim: false` ·
`production_accepted: false` · `governed_handoff_eligible: false` ·
`evidence_axis: "[macOS-CPU frozen-PoseNet one-pair advisory]"`.

**Falsified four times, every time at larger n:**

| receipt | measured | scale |
|---|---|---|
| `ddm_pg1_pose_gn_convergence_20260802.md:28` | **1.2%**/relin at the shipped bound (ep854) | n>1 |
| `ddm_ss1_selection_vs_search_20260803.md:112` | **1.2%**/relin (confirms pg1) | n>1 |
| `ddm_pv1_..._20260816.md` §1 | **0.1549%** mass-weighted forfeited; **0/50** stop on a cap | n=50 seeded-random |
| `ddm_ps1u_..._20260816.md:33` | **0.07%** | vehicle |

**Every measurement at n≥6 lands at 1.2%, 0.155%, or 0.07%. The 13–23% band exists at exactly one
place in the corpus: two n=1 final steps in a stale rehearsal.**

**Propagated into** `ddm_cv1` (08-02) · `gc16` (08-04) · `ddm_seg_bank_routing` (08-05) ·
`ddm_op1r/CHARTER` (08-09) · `ddm_b2e_..._charter_20260816` · `ddm_ns1_..._20260816` · **this
arm's charter**. The last three were written *after* pg1 and ss1 had already falsified it.

It also entered **code** as motivation: `terminal_pose_gn.py:520` and
`tests/test_terminal_pose_gn.py:451` both cite it. That is how an advisory n=1 number acquires the
look of a derived constant. This is the **cross-regime constant transfer** genus
(`[[cross-regime-constant-transfer-genus-finishing-stage]]`) crossed with the **n=1/prefix**
population law (m88/m96) — and note the direction: a 1-pair sample overstated the rate by **~85×
to ~150×** against the n=50 figure.

---

## 3. The cure, built and wired — `_lint_falsified_premises`

**Why a new store rather than the existing one.** `_lint_stale_numbers` already asks the right
question but cannot see this class, for a reason that is itself sound: it consumes au1's
**auto-regenerated** index (`tools/au1_measurement_integrity_audit.py`, 11,840 rows) whose
`refuted_value` entries include `"0"`, `"6"`, `"12"`. Its `≥4-digit` filter is the only thing
keeping it from firing constantly. A hand-written row in that file would also be **overwritten on
the next sweep**. So: a separate, small, curated, claim-level store.

- **`.omx/state/falsified_premise_registry.jsonl`** — one row per dead premise: `claim_patterns`,
  `origin` (path, n, authority_mode, score_claim, the verbatim `authority_blocker`),
  `falsifications[]` (path, measured value, scale), `verdict_scope`, `propagated_into[]`, and
  `why_it_recurs`.
- **`tools/codex_arm_queue.py::_lint_falsified_premises`** — sixth leg of
  `lint_charter_recall_advisories`, so it runs on every `codex_arm_queue.py add`. Normalises
  en/em dashes. Advisory-only; a missing, empty, or malformed store yields **silence**, never a
  block. Capped at 5 warnings so one charter cannot flood the spawn site.

Fired against my own charter text it emits the origin's `n=1`, its `STALE_REHEARSAL` authority,
and all four later measurements. **The detector zeroes on the cure:** with the row present, a
charter restating the premise is flagged at birth; with it absent, nothing changes for anyone else.

**11 tests** (`tools/tests/test_codex_arm_queue_falsified_premises.py`), behaviour not constants,
including that the leg is actually *called* (the orphan class this repo rates P0) and that a broken
store cannot block a spawn. `ruff --select F` clean; 13 pre-existing `codex_arm_queue` tests green.

**Round-1 self-review caught two real defects in my own code**, both fixed before commit: a bare
`13-23` pattern that would have fired on "pairs 13-23" (warnings are blocking here, so noise is a
cost), and an unguarded `origin` dereference that would have raised on a malformed row. Both now
have tests.

---

## 4. The pose axis, priced

Payload: `/Volumes/APDataStore/pact/ddm_pu3_20260816/retained/pose_axis_arithmetic.json`,
**2,423 B**, sha256 `50f4142fbe7bf414825d8ca37b55c2f6d251009c000995cb759a3262e4e95a49`.

Base **hv1 ep0634**, sha `80d9c8c6…`, 182,759 B, `d_seg 0.00029611`, `d_pose 6.88e-06`.
S rebuilds to **exact float equality** — independently reproduced, confirming pv1 §3:

| term | S | share |
|---|---:|---:|
| rate | 0.121691716 | **76.25%** |
| seg | 0.029611000 | 18.55% |
| **pose** | **0.008294577** | **5.20%** |
| gap to 0.15 | −0.009597293 | |

- **Pose ceiling: −0.008294577 = 86.43% of the gap.** Zeroing pose entirely still leaves
  **0.001302716** on the table. Pose alone cannot close it.
- **Exchange rate: a 1% relative `d_pose` cut = 4.157709e-05 S = 62.4 counted bytes.**
- **Break-even required cut for a candidate costing ΔB bytes**
  (`f = 1 − (1 − ΔB·6.658589531e-07 / 0.008294576541)²`):
  26 B → 0.417% · 31 B → 0.497% · 100 B → 1.599% · 588 B → **9.2177%** · 997 B → 15.367% ·
  1,749 B → 26.109%. The 588 B row **reproduces pv1's 9.2177%** and the 31 B row **reproduces
  pk3's independently-derived 0.497%** — two cross-checks from different arms.
- **Instrument resolution.** The receipt reports `avg_posenet_dist` to **3 significant figures**,
  and canonical S is recomputed *from* that rounded value. One reported bin spans **6.028e-06 S**;
  the minimum relative cut that changes the reported bin is **0.0727%**. Not binding for any
  gap-sized win, but it means sub-0.07% pose claims are invisible on the shipping receipt.

### 4b. Correcting my own charter's d_pose

The charter's `d_pose = 6.8799e-6` is a round-trip artifact of back-deriving from a truncated pose
term. The receipt value is **`6.88e-06`**; `0.0082945765² / 10 = 6.879999931e-06`. Use `6.88e-06`.
Separately, the 16-digit `6.885642960696714e-06` pinned in 13+ modules belongs to **CP135 @
186,252 B**, not hv1 — worth **+3.4009e-06 S** of error when carried onto hv1 (pv1 §3, independently
re-derived here: it lies *outside* the rounding interval of 6.88e-06, so it cannot be the unrounded
source).

---

## 5. Where hv1's pose actually comes from — and why that closes the charter's framing

**There is no live pose solve on the hv1 lineage.** hv1's pose is a **frozen inheritance**. The
causal object is the **22,161 B CPR1 carrier stream** that renders **frame_0 only**
(`inflate.py:660,669`). SegNet reads `x[:, -1]` = frame_1, so the carrier is **seg-invisible by
construction and is a pure pose actuator**. Its basis and coefficients were fit by joint gradient
descent in **PR130's pose leg** and have been **byte-identical through cp135 → mc36 → e480b → hv1**;
`ddm_hv1_harvest_compose.py` proves the freeze by full raw-output byte identity and moves **only
the rate term**. `ddm_fb1:58`: *"`seg + pose` is decode-identical across the entire lineage."*

Consequence worth stating plainly: **the whole recent frontier has moved on rate alone.** d_seg and
d_pose have not changed at all from MC36 (186,269 B) through e480b (183,502 B) to hv1 (182,759 B).

The carrier's own price: **22,161 B = 0.0147561 S of rate = 1.54× the entire gap**, to deliver a
pose term of 0.0082946. Its rate cost *exceeds* its pose benefit by 0.0064615 S. That makes the
deletion direction arithmetically interesting — deleting it is net-favourable iff carrier-free
`d_pose` stays below **5.313e-05**, i.e. iff removal degrades pose by less than **7.72×**.
**I am not proposing that as an open lever**: `ddm_ra3` (2026-08-16, exact n600) measured the
incumbent carrier projection at `ΔS_pose = +0.136386` — ~14× the gap — and closed the
trust-regioned per-pair re-fit family at **35.5×**. Carrier perturbation is measured to be
catastrophic for pose, which predicts deletion blows well past 7.72×. Recorded as arithmetic, not
as a candidate.

---

## 6. Six *distinct* mechanisms, all negative

The law `[[same_defect_negatives_masquerade_as_family_convergence_20260805]]` says N negatives
sharing one defect are one instance. I checked: these do **not** share a defect — refit, overlay,
carrier swap, GN solve, basis, and iteration budget are different mechanisms. This is genuine
family convergence.

| mechanism | result | scope | axis |
|---|---|---|---|
| relinearization budget | 0.1549% of achievable | formulation | advisory (n=50) |
| rank-6 cosine basis run to convergence (p3v2) | plateaus 38.06 → ~15.29, RANK_DEFICIENT | formulation | advisory |
| linear frame-0 overlays (pk4) | all 3 rungs heldout-**negative** (−1.0e-5, 0.0, −1.1e-5) | formulation | torch-CPU advisory |
| ps135b pose carrier | d_pose 1.4674e-4 = **21.3×** worse | instance | **contest-CUDA** |
| ps1u uncapped solve, shipped | d_pose 6.146e-5 = **8.93×** worse, +1.686e-02 S | instance | **contest-CUDA** |
| trust-regioned per-pair re-fit (ra3) | works, still **refuses 35.5×**; family closes | family | advisory n600 |

Plus the un-drawn inference pv1 named: the uncapped solve's **converged floor** on its own object
is `1.285917e-05`, **1.87× worse** than what the CUDA decode delivers for free.

> **VERDICT (mine, bounded): no measured carrier-addressable pose headroom exists on the shipping
> object.** `verdict_scope: **formulation**` — frame-0 carrier-side pose actuation on the
> cp135→hv1 vehicle. **NOT a paradigm verdict:** the CUDA-side floor is genuinely unmeasured
> (§7), and joint-descent-through-the-shipping-receiver — the mechanism that *built* the incumbent
> carrier in PR130 — has never been re-run on this vehicle.

---

## 7. Two corrections to sibling memos (both live, both narrow)

**(a) `ddm_ps1u_r2_dual_axis_pose_verdict` mis-attributes the mechanism.** It says *"The build
never measured pose. It carried 0.0 as a placeholder."* At source that is wrong. `ddm_ps1u_pose_seal.py`
measured pose on the CPU-decode object (41.8% cut, 60/600 pairs) and shipped an explicit
`DEVICE_DEPENDENT_DECODE_WARNING` stating *"the local advisory pose reduction was solved against
the CPU-decode object and its CUDA-axis value is UNMEASURED. **This row measures the transfer.**
The advisory row cannot admit."* The row did what it said. `local_pose_delta: 0.0 /
pose_unmeasured: true` is the **worker transport law**, not evidence of an unmeasured build.

**(b) That memo's "owed next" fire-order is unexecutable as written.** It requires that *"a pose
candidate may not dispatch with `pose_unmeasured: true`."* Three workers **hard-refuse the
opposite**: `ddm_re1t_modal_t4_sign_gate.py:564-565`, `ddm_qs2_compensation_rate_rung.py:811-816`,
`ddm_js1c_cuda_custody_stage0.py:372-373` each raise unless `local_pose_delta == 0.0` **and**
`pose_unmeasured is True`. Implementing it literally breaks the transport contract. The correct
cure is not "measure pose locally" (ps1u did) but **transfer validity**: when the screen object and
the adjudication object differ, the seal must declare itself a **transfer probe with a bar it can
clear**, not an admission row set at the frontier S — which pre-registers it to fail.

---

## 8. Sealed fire-orders for MAIN

1. **(free, do first) Re-lint the two live charters that still carry the dead premise** —
   `ddm_b2e_train_for_editability_burn2_charter_20260816.md` and
   `ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md`:
   `.venv/bin/python tools/codex_arm_queue.py lint --name <arm> --prompt <charter path>`
   Expect `charter-lint WARN … FALSIFIED premise`. Any arm routed on 13–23% should be re-aimed.
2. **(free) Scope-fix the pinned constant.** `6.885642960696714e-06` is CP135 @ 186,252 B; every
   hv1-based bar needs `6.88e-06`. 13+ modules. pv1 raised this; it is still open. Two-landing:
   fix + a gate refusing an archive-unqualified `BASE_D_POSE` pin.
3. **(free) Adopt the break-even closed form** in §4 as the pose screen for any future candidate:
   `f = 1 − (1 − ΔB·6.658589531e-07 / 0.008294576541)²`, adjudicated on
   `score_recomputed_from_components` vs `0.15959729295498598` — never on a d_pose restatement
   (pv1 §3b false-admit window).
4. **The only measurement that reopens the pose axis:** a **CUDA-side floor** — solve the carrier
   against CUDA-decoded frames. Blocked on the device-dependent-decode cure (ps1u §5b). Needs CUDA
   compute; I hold neither budget nor slot. **Do not fund it before item 2**, or it will be
   adjudicated against the wrong base.
5. **DO NOT re-open** the relinearization cap (closed 4×), the linear frame-0 overlay family
   (pk4, formulation), or the trust-regioned carrier re-fit (ra3, family).

---

## 9. What I did NOT do

- **No re-run of the solve, no rebuild.** It existed, uncapped and converged, before I was spawned.
  My charter's task 3 was already complete; executing it would have been the duplication the
  charter warned about, pointed the wrong way.
- **No dispatch, no governed/Metal launch, no n600 scorer job.** $0. Scorer slot untouched
  (live training pid 4832).
- **No CUDA-side measurement.** The honest gap in §6, and the reason §6's verdict is scoped
  `formulation` and not `paradigm`.
- **No claim that the pose axis is closed.** It is not. Six mechanisms are negative on one vehicle;
  the shipping-axis floor is unmeasured.

## 10. My own round-1 adversarial review

1. **Am I just re-reporting pv1?** Partly, and I say so rather than dressing it up: §1 and the
   0.1549% are pv1's. My independent contribution is the arithmetic re-derivation (which
   *confirmed* pv1 exactly, and reproduced pk3's 0.497% as a second cross-check), the propagation
   census, the structural reason for recurrence, the built cure, and §7's two corrections.
2. **Is the propagation count solid?** The seven sites are grepped and listed. I did *not* verify
   that each one *routed a decision* on the premise — only that it restates it. Stated as
   propagation, not as damage.
3. **Is my cure orphan-risk?** It runs inside an existing spawn-site chain that already fires on
   every `add`, and a test asserts it is actually called. But it is **one row**: its value depends
   on future arms adding rows. That is a real limitation, not a solved problem.
4. **Did I overstate the instrument-resolution finding?** I nearly reported "three archives report
   identical d_seg *and* d_pose" as an invariance finding. Checked first: `ddm_fb1:58` records the
   lineage as decode-identical, so it is a **tautology**. Killed it before writing.
5. **Is §5's carrier arithmetic a candidate?** No, and I labelled it so. ra3's +0.136386 predicts
   deletion fails the 7.72× tolerance badly. Presenting it as an open lever would have been the
   vacuous-denominator trap ra3 itself named.
6. **Unverified:** whether the two 08-16 charters were actually spawned through `codex_arm_queue`
   (if hand-spawned, the lint never ran and cannot have caught them); and whether
   `terminal_pose_gn.py:520`'s citation should be annotated — I left that dead module untouched
   rather than churn off-chain code.

## NEXT_IF_RESUMED

Fire-order 1, then 2. The pose axis needs no more local analysis: it needs either the CUDA-side
floor (funded, after the constant fix) or reassignment. **On the arithmetic, seg (0.029611, 3.09×
the gap) and rate (0.121692) are where the gap is closeable; pose caps out at 86.43% of it and has
six negatives against it.**
