# ddm_cr1 (#827) — THE COMPOSITION ROW: the seg+rate prize is 2.4× larger than recorded, and it is gated entirely by a MEASURED pose wall

**Date:** 2026-08-01 · **Arm:** ddm_cr1 · **Axis:** `[macOS-CPU advisory]` ·
`score_claim=false` · `pointer_moved=false` · **Pointer UNMOVED.**
**Review status:** pre-registered-falsifier + own-round-1 reviewed (1 pass); NOT fresh-eyes reviewed.

**STORES CONSULTED:** `tools/corpus_query.py "cell_drop50 composition row 827 burn seg rate lever
byte-closed"` (research 7347 / equations 864 / memory 2039 / dag 908 / council 292 / tasks 396 /
docs 96); `.omx/state/main_hot_state.md`; `.omx/state/current_focus.md`;
`.omx/state/canonical_task_status.jsonl` (row 391, task 827); `docs/operating_manual_craft_handoff.md`;
`tac.subagent_contract.standard_contract()`. **Deliberately NOT loaded:** the burn-4 charter's
window-by-window telemetry, the gc-series convocation memos beyond their corpus excerpts, and the
QA24/QA84 grammar-race memos — none bear on the two arithmetic reconciliations this arm owed.

---

## 1. THE POINTER DID NOT MOVE

No exact-eval row was produced by this arm. The full-n600 scorer slot was **deliberately not spent** —
see §6. Own-vehicle frontier remains **v4d = 0.9639878** `[macOS-CPU advisory]`
(seg 0.431179 + pose 0.292941 + rate 0.2398677 @ 360,238 B). Bar 0.172141. Gap 0.7918468.

## 2. THE TWO RECORDED ΔS FIGURES — RECONCILED (both correct, different objects)

| figure | object | baseline | scope |
|---|---|---|---|
| **−0.042259 S** (`main_hot_state`) | burn rung-1 endpoint **d_seg 0.0038892 → seg 0.388920** | **v4d seg 0.431179** | **seg term ONLY**; no rate, no pose, not byte-closed as a composed archive |
| **−0.035996 S** (`#827` title) | **ep854** byte-closed **S 0.634232** @ 360,331 B | **gr1_cell_drop50 seg+rate 0.6702284** @ 359,221 B | **seg+rate**, pose-blind on both sides |

Both arithmetics reproduce exactly (MEASURED):
`0.388920 − 0.431179 = −0.042259`; `0.6702284190 − 0.6342319906 = −0.0359964`.
They are **different epochs of different burn windows** — rung-1 is window_01 ep399, `#827` is
window_03 **ep854** — and different term sets. Neither is wrong; they are not comparable to each other.

**Custody (MEASURED, SHA'd this session):**

| artifact | bytes | sha256 |
|---|---|---|
| `ddm_gr1_20260730/gr1_cell_drop50_archive.zip` | 359,221 | `a6398e441f4bc818adde88f16afc2d031fb51a449371a6cf00ee72548f4ca310` |
| `ddm_ep2_20260731/archives/w03_ep854_representative/archive.zip` | 360,331 | `37ba7a96abb47f7a9e872935581eac311187445de27f33d66185454405506069` |
| `ddm_b4s_20260731/window_03/checkpoints/intra_seg_trunk_tau_ep00854.npz` | 14.3 MB | `d99c5cb2fdd751b2bde674e086ad3e93e65f9b6f00f573203e4220a5082c4e25` |
| `ddm_v4d_20260731/v4d_composed_refine_celldrop50_archive.zip` | 360,238 | `f1f3288062468e97c090ffe88ac81a6d6f76925743bd83aecb15307c0314a220` |

## 3. THE −0.035996 HEADLINE IS THREE-WAY CONFOUNDED (MEASURED)

The two archives differ in **three** variables at once, not one:

1. **seg base** (the intended variable);
2. **token coder** — `gr1` ships `state/tokens.dr7t` magic `DR7T` = **r7 SMEVR-coded**;
   `ep854` ships `state/tr1.ddt1` whose tokens section is magic `TR1TOK1!` = a **Brotli frame**;
3. **pose presence** — `gr1` carries `state/pose_warp.stp` **6,864 B**; `ep854`'s `pose_stub` is
   **INERT (83 B)** and it carries **no pose payload at all**.

(2) and (3) push in **opposite** directions, so the headline is not a bound in either direction.

**Archive grammars are also different** (this is why a member swap is not a valid composition):

```
gr1 / v4c / v4d   6 ZIP members, schema ddm_pfs1_composed_archive.v3_warp
                  manifest.json · tokens.dr7t · renderer.sec · selector.sec · pose_stub.sec · pose_warp.stp
ep854             2 ZIP members, schema ddm_tr1_runtime_archive.v1
                  manifest.json · state/tr1.ddt1  (4 sections inside a packet directory)
```
`renderer.sec` differs between them (the burn trained the lotto gains/biases, as expected);
`selector.sec` and `pose_stub.sec` are **byte-identical**.

### 3a. Corrected apples-to-apples seg+rate

Re-coded ep854's token codes through the **same** committed coder the v3_warp builder uses
(`encode_token_codes(codes, levels=16, codec="smevr")`, roundtrip verified EXACT against the codes):

| | tokens | composed archive | rate term |
|---|---:|---:|---:|
| gr1 cell_drop50 | 346,478 B (already SMEVR) | 359,221 B | 0.2391905 |
| **ep854 burn** | **271,505 B** (SMEVR; 355,182 B as shipped Brotli) | **284,248 B** (DERIVED) | **0.1892691** |

```
gr1    seg 0.4311790 (EXACT evaluator) + rate 0.2391905 = 0.6703695
ep854  seg 0.3943024 (advisory n600)   + rate 0.1892691 = 0.5835714
DELTA  −0.0867981 S   =  seg −0.0368766  +  rate −0.0499214
```

**The recorded −0.035996 understates the true seg+rate prize by 2.41×.** More than half of the
corrected delta (**57%**) is **rate**, not seg — because the burn trained against
`byte_ledger_coder: "smevr"` and its token field codes 21.6% smaller than the cell_drop50 field.

*Independent corroboration:* `ddm_ep2_receipt.json` records `counted_ledger_bytes 275,005` for ep854;
my reconstruction (271,505 + 3,341 + 535) = 275,381 — agreement to 376 B, from a separate instrument.

**Relative significance of −0.0867981:** **10.96%** of the 0.7918468 gap to the bar;
**89.1%** of the 0.097465 S total known inventory. This is the largest single item in the inventory.

*Caveat (labelled):* the 284,248 B composed size is **DERIVED** — it assumes ep854's re-solved
`pose_warp.stp` is the same 6,864 B as gr1's. Its `s_t` SMEVR part will differ slightly; expect
±O(100 B) = ±O(7e-5 S). The token, renderer, selector, pose_stub and ZIP-overhead terms are MEASURED.

## 4. THE NAMED BLOCKER IS NOT A TRANSFER COST — IT IS A PHOTOMETRIC WALL (MEASURED)

`#827`'s named blocker was "v4d's pose was solved against gr1's RENDERS, so a base swap ships
corrections fitted to different pixels." I parameterized `tools/pfs1_recompose_warp_base_and_eval.py`
with `--seg-archive` (§5) and **re-solved the warp pose on ep854's OWN shipped frame_1**, which is
exactly what that tool exists to do, and measured it against a matched control.

**MATCHED-PAIR MEASUREMENT** — same tool, same frozen PoseNet, same shipped f16 targets, same
protocol, pairs 0–60, only the seg base differs:

| base | mean d_pose | median | max | contribution √(10·mean) |
|---|---:|---:|---:|---:|
| control `p2c_aimed` (pfs1 D1 solve) | **0.489332** | 0.339111 | 2.1190 | 2.2121 |
| **ep854 burn base** | **3.112073** | 2.833984 | 6.3640 | **5.5786** |

**6.36× worse, on 61 of 61 pairs — unanimous.** Not an outlier effect.

**The transfer hypothesis is REFUTED by its own control.** gr1's shipped `pose_warp.stp` is the
p2c_aimed solve carried over **verbatim** to a different base, and it cost only
`0.234817 − 0.221547 = +0.013270` d_pose (**+6.0%**) — measured by the exact evaluator. So base swaps
per se do **not** break this carrier. **The burn base specifically does.**

**Mechanism (DERIVED, and predicted by our own standing law).** `window_03/tr1_config.json` carries
`w_seg 100.0`, `w_rate 0.05`, `rate_model "entropy"` — and **no pose term of any kind**. The burn drove
the renderer purely toward SegNet-argmax fidelity, a task-lossy objective that is free to destroy the
photometric texture PoseNet reads. That is verbatim the CLAUDE.md pose clarification: *"frames trained
on seg alone do not carry pose-legible photometric signal, and no post-hoc storage fixes frames never
shaped for it."* The solver's `s_t` histogram corroborates: control concentrates at index 7 (s_t=0.08),
ep854 shifts **down** to index 6 (s_t=0.06) — both **interior** to the 11-point grid, so this is
**not** grid saturation, it is the solve choosing a smaller ground-plane scale because the render
supports less warp. That closes the cheapest "it's just the grid" reformulation.

**Net at the composed level.** Pose cost ≈ **+3.367 S** (matched-pair, MEASURED) against a corrected
seg+rate gain of **−0.0868 S** — the pose cost is **≈39×** the prize. Extrapolating the control's
own difficulty profile (pairs 0–60 are 2.21× the run mean) puts ep854's full-n600 mean near
**~1.41** → contribution ~3.75 → **+2.27 S** cost, still **≈26×** the prize. That extrapolation is
**INFERRED**, not measured; the 61-pair matched ratio is MEASURED.

## 5. RECEIVER ADJUDICATION (#417-class) — the blocker as stated does not exist; the tool expectation was mis-routed

MAIN's flag: *"the dr7t archive grammar's readers live in experiments/ + tools/, and no receiver was
found under src/tac/ where r6cal_byteclose_and_eval expects one."* MEASURED adjudication:

1. **A src/tac receiver EXISTS and WORKS on the live archive.**
   `src/tac/optimization/ddm_tr1_runtime.py` (54.9 KB) parsed the ep854 archive cleanly this session
   via `parse_archive` → selector, 600×24×32×4 uint8 token codes, all 4 sections, masks/gains/biases.
2. **`tools/r6cal_byteclose_and_eval.py` is bound to a different vehicle.** Its line 101 imports
   `tac.witness_dsl.v10_production_receiver` — the **V10 describe/realize** receiver
   (`src/tac/witness_dsl/v10_production_receiver.py`, 55.4 KB, present). It was never the TR1/dr7t
   reader. Expecting the TR1 receiver there is a **mis-routing, not a missing receiver.**
3. **The dr7t 6-member composed grammar is genuinely NOT in `src/tac`.** Its reader is the vendored
   `pfs1_warp_receiver.py`, which exists **only** as generated bytes inside built submission trees
   (no repo source file); the decode entry point is `experiments/inflate_runner_v4d.py`. Under the
   operator's 2026-08-01 ruling that `src/tac` is the source of truth, **that is a real and open
   #417-class debt** — but it is a debt on the *composed v3_warp/v4d* grammar, not on TR1, and it did
   **not** block this arm: `tools/pfs1_recompose_warp_base_and_eval.py` and
   `tools/pb1_p5_byte_close_and_eval.py` both build and evaluate through it today, and the gr1 archive
   already rode a real `upstream/evaluate.py` n600 to rc=0 through it (§7).
4. **Correction to a standing record.** `main_hot_state` FRESHEST_RECEIPTS states *"The 6-member
   'dr7t' grammar in my working memory was WRONG"* and that the archive is 2 members. That is true of
   **raw TR1** and false of the **live v4c/v4d/gr1 lineage**, which is 6-member dr7t on disk. Both
   grammars are real and in use; the correction over-generalized from a TR1 rehearsal receipt.

**Landed:** `tools/pfs1_recompose_warp_base_and_eval.py` gains `--seg-archive` (default = the previous
hardcoded `p2c_aimed` constant → **no behavior change for existing callers**), matching the flag
`tools/pb1_p5_byte_close_and_eval.py:206` already exposes. Second, an anti-fake fix in the same file:
the build receipt hardcoded `"seg_from_pb1": 0.38901` and folded it into `S_pred`; on a non-default
base that constant is silently wrong, so it is now **withheld (`None`)** rather than carried over.

## 6. WHY THE n600 EXACT-EVAL SLOT WAS NOT SPENT

Pre-registered falsifier (P7): *a composed candidate earns the slot only if it can plausibly beat
v4d 0.9639878.* Composed prediction for ep854 + warp-base pose:

```
seg 0.394302 + pose 3.75 (INFERRED) .. 5.579 (MEASURED @ n61) + rate 0.189269
      =>  S ≈ 4.33 .. 6.16     vs v4d 0.9639878
```

That is 4.5–6.4× worse than the live frontier. Spending ~20 min of scorer solve plus ~17 min of
evaluator to produce a row that cannot move the pointer, and cannot calibrate anything new (gr1
already calibrated this exact grammar against the real evaluator, §7), would be means-as-ends. **The
slot is left free.** The partial solve is on disk and resumable:

```
.venv/bin/python tools/pfs1_recompose_warp_base_and_eval.py --mode solve \
  --work-dir /Volumes/VertigoDataTier/pact/ddm_cr1_20260801/ep854_compose \
  --seg-archive /Volumes/VertigoDataTier/pact/ddm_ep2_20260731/archives/w03_ep854_representative/archive.zip \
  --n-pairs 600 --resume            # 61/600 done, per-pair fsync'd JSONL
```

## 7. #826 IS ALREADY DONE — the first exact `upstream/evaluate.py` n600 row on the gr1 base exists

`main_hot_state` lists `#826` as an unspent fidelity-law check. It **completed** and the receipt is on
disk at `/Volumes/VertigoDataTier/pact/ddm_ep2_20260731/gr1_eval/d1_eval_receipt.json`
(rc=0, 600 samples, wall 1002.17 s, `--device cpu`, archive sha `a6398e44…` — verified this session).

**S recomputed from components** (the printed `Final score: 2.20` is rounded and must never be quoted):

```
seg  100 × 0.00431179   = 0.4311790
pose √(10 × 0.23481703) = 1.5323741
rate 25 × 359,221/37,545,489 = 0.2391905
S = 2.2027435921291647          [macOS-CPU advisory — real evaluator, real bytes]
```

**Advisory-vs-exact calibration on seg (NEW):** advisory n600 d_seg 0.004310379 vs exact 0.00431179 →
**Δ = 1.411e-6 d_seg = 1.411e-4 S**. Our advisory seg protocol runs ~1.4e-4 S optimistic on this
vehicle. (Do not carry over the "d_seg agrees to 1e-8" line from the pfs1 D1 receipt — that was a
different archive.)

## 8. VERDICT SCOPE

**FORMULATION-level negative.** Scope: *"compose the ep854 burn seg base with the **warp-base** pose
carrier, pose re-solved post-hoc on the new renders."* One formulation, optimal-form for that carrier
(bespoke re-solve on the base's own renders — the strongest form the carrier admits).

**Explicitly NOT killed, and the open reformulations:**

1. **The v4c/v4d pose rungs on the burn base** — v4c/v4d add per-pair exposure `(a,b)` + a static
   two-plane photo term, i.e. machinery whose *purpose* is absorbing photometric drift. Untested here.
   Requires parameterizing `ddm_v4d_build_composed_archive.py` (its `BASE` is a module constant,
   there is **no** `--seg-archive`, contra the `current_focus` note — that flag is on `pb1_p5`).
2. **Joint pose-in-the-loop on the burn base** — CLAUDE.md's own clarification says only joint descent
   crosses the photometric wall; every burn window to date ran with **zero** pose term. A burn window
   with a nonzero pose weight is the direct test, and it is the only reformulation the measured
   mechanism actually endorses.
3. **The rate half is independent of pose.** −0.0499214 S from SMEVR-coding the burn token field is a
   pure coder fact that does not depend on any pose carrier. If the burn base is ultimately rejected,
   the question "why does the burn field code 21.6% smaller" should still be harvested.
4. **A pose carrier that does not read the renders** — untested class.

**NOT tested, do not infer:** window_01/ep399, ep809, ep934, or the window finals; across-seed
variance (single seed throughout, per design-philosophy P2 our across-seed variance is UNKNOWN).

## 9. MAIN's truncated-GN finding (#850) — scope check

MAIN's measured finding that `solve_terminal_pose_gn` is capped at `relinearizations ≤ 3` (validator
ceiling), has no convergence test, and stopped both rehearsal solves at 2 iterations while still
descending 13.2%/23.2% — **does not bear on the numbers above**, because this arm's carrier is the
**warp-base** carrier (grid search over `s_t` + stored f16 targets), not the terminal GN packet. Those
are different sections: `state/pose_warp.stp` (`PFS1WPB1`) vs `state/pose.tpgn` (`TerminalPosePacketV1`).

It **does** bear on the `pb1_p5` route, and there is a measured reason not to take that route anyway:
`ddm_pb1_20260729/p5/p5_eval_receipt.json` records the `pose.tpgn` carrier at
**d_pose 38.06224823 → S 20.27** (rc=0, n600, real evaluator). That is the "own-vehicle line 20.27"
in memory. Any future attempt to fix ep854's pose through the terminal-GN route would carry MAIN's
2-relinearization truncation **on top of** the photometric wall measured here — two confounds, not one.

---

## Observability surface

**Inspectable per layer:** per-pair solve rows (`pair`, `s_t_idx`, `d_pose`) fsync'd to
`d1_warp_solve.partial.jsonl`; per-section byte ledger from `parse_archive`; per-member ZIP ledger.
**Decomposable per signal:** S split seg/pose/rate at every claim; the corrected delta split
seg −0.0368766 / rate −0.0499214. **Diff-able across runs:** matched-pair join against the pfs1 D1
solve JSONL on identical pair indices. **Queryable post-hoc:** all inputs SHA'd in §2; receipts on the
SSD. **Cite-able:** every number in this memo carries its producing artifact path. **Counterfactual-able:**
`--seg-archive` makes the base a free variable, so any TR1 endpoint can now be swapped and re-measured
under an otherwise identical pipeline.

## Wire-in (6 hooks)

1. **sensitivity-map** — ACTIVE: token-coder choice is a first-class rate lever worth −0.0499 S on this
   base; pose-carrier↔render-photometry is a measured coupling with a 6.36× coefficient.
2. **Pareto constraint** — ACTIVE: the burn seg base is admissible only jointly with a pose carrier that
   survives seg-only-trained renders; seg+rate and pose are **not** separable on this base.
3. **bit-allocator** — ACTIVE: SMEVR on the burn token field (271,505 B vs 346,478 B).
4. **cathedral autopilot** — N/A (no archive promoted; nothing dispatchable).
5. **continual-learning posterior** — ACTIVE via this memo + the canonical-equation anchor.
6. **probe-disambiguator** — N/A: the matched-pair control **is** the disambiguator and it returned
   unanimous (61/61); no second interpretation survives it.
