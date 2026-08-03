# ddm_df1 (#908, classification half) — which campaign verdicts sit inside the #903 floor

**Date:** 2026-08-03 · **Arm:** `ddm_df1` · **Axis:** `[macOS-CPU advisory]` / apparatus ·
`score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`, `ready_for_exact_eval_dispatch=false`.
**ZERO scorer forwards. ZERO training runs. No launch, no dispatch, no paid anything.** This memo
re-reads artifacts already on disk; it moves no score and claims none.

**STORES CONSULTED:** `.omx/research/ddm_dt1_determinism_floor_20260803.md` (#903, the source) ·
`ddm_fp1_class_field_projection_20260731.md` (#799) · `ddm_pa1r_pool_a_race_20260730.md` (#793) ·
`ddm_dw1_qa75_distill_window_20260730.md` (#790) · `ddm_pj1_projection_probe_20260730.md` (#788) ·
`ddm_gc11_burn3_gate_20260730.md` (the branch function) · `ddm_gc12_wall_branch_convocation_20260731.md`
(#800) · `ddm_lg1_lane_guard_20260731.md` (#808) · `ddm_bs1_margin_density_preflight_20260801.md` (#815) ·
`src/tac/witness_dsl/ax1_pool_a_race_20260730.py` · `experiments/ddm_fp1_class_field_projection.py` ·
`experiments/ddm_pj1_token_projection_fit.py` · `experiments/train_tr1_partition_renderer_mlx.py` ·
`.omx/state/main_hot_state.md` · task rows #903/#908/#863/#824/#815 (reconstructed from the session
transcript — the harness ledger still has no on-disk mirror, per `ddm_p2a` §1).

---

## §0 ANSWER FIRST

**The pre-registered falsifier is NOT met, on BOTH of its clauses, at BOTH ends of the bracket.
#908 is CONSEQUENTIAL, not hygiene.**

- Clause A ("< 3 retrain-based verdicts with |effect|/floor < 3 at the 40% end"): **8 of 8** ranked
  retrain-based rows are below 3 at the 40% end; **7 of 8** are below 3 even at the *most favourable*
  8.2% end, and **6 of 8** are below **1** there.
- Clause B ("none of them is branch-selecting"): **one is.** The `pa1r` leg of the #800 BR-D
  adjudication ranks 1st, 2nd and 4th-most-exposed in the whole table.

**And I found the root defect, which is sharper than "some verdicts are noisy."** The campaign's
decision threshold is a named constant in shipped code —
`ax1_pool_a_race_20260730.py:62  DEFAULT_DSEG_NOISE_FLOOR = 2.99e-5` → band `100·that = 0.00299 S`.
Its provenance (`dw1` §5, verbatim) is **"B's gate residual std about its own trend"** — a
**WITHIN-TRAJECTORY** dispersion. It was then used to compare **SEPARATELY TRAINED ARMS**. A
within-run residual std is structurally incapable of seeing run-to-run divergence. Against dt1's
measured bracket at the campaign's own operating point (d_seg ≈ 0.0042–0.0056), the band in use
**understates the between-run floor by 11.5× (8.2% end) to 67.9× (39.7% end).**

That is the same genus dt1 named for itself — *the number we were checking was the one that could
not see the problem* — one level up: not a hidden scalar, a **hidden estimator**.

**Third finding, and it is a signal-loss one: #903 was already MEASURED and WRITTEN DOWN on
2026-07-31, three days before dt1, by `ddm_lg1` (task #808)** — machine-readable, in
`/Volumes/VertigoDataTier/pact/ddm_lg1_20260731/lg1_custody_manifest.json`:
`"vehicle_rerun_nondeterministic": true`. It sat in a section titled *Honest negatives / limitations*.
The campaign kept using `2.99e-5` anyway. I re-derived lg1's numbers independently (§5) and they
**replicate dt1's bracket from a different arm's banked artifacts at $0** — so #903 is not an artifact
of dt1's harness. lg1 also makes a claim that **CONFLICTS** with dt1's (§5.2); I report it unresolved
rather than picking the convenient side.

**What still stands, stated positively (weight positives over negatives):** the entire
same-checkpoint population — **101 of 113** enumerated run directories wrote no trained artifact at
all — plus `fp1`'s load-bearing receiver floor and `pj1`'s verdict, both of which survive a
deliberately hostile transfer of the floor. The pointer line is untouched, as MAIN already
cross-checked.

**Order note, as MAIN asked me to record:** #908 as written puts the operating-point floor at STEP 1
and the classification at STEP 2. MAIN inverted it because the floor needs a scorer slot that
`ddm_bp2` may hold and the classification needs none. **The inversion was correct and it also
produced something the stated order could not**: the classification found a banked, already-paid
determinism replication (§5) and the estimator defect (§4), neither of which required the floor.

---

## §1 METHOD — and what would have made this instrument lie

Classification is derived from the **CODE PATH**, per charter — the question is always *does this
measurement run the R upsample VJP inside an iterative optimizer?* — never from the memo's prose.

**Two independent censuses, each with its denominator, so neither can be vacuously "clean":**

**Census A — the code paths (what CAN contaminate).** All **10,591** tracked `.py` files scanned for
`{_apply_R, set_fused_r_kernel, render_through_R, render_batch_through_R, bicubic_up_to_camera}` ×
`{mx.value_and_grad, nn.value_and_grad, mx.grad(, mx.vjp}` × `{optim.Adam/AdamW/SGD, Muon}`.
**156** files touch an R token or an MLX gradient. Of those, exactly **6** pair an R token with an MLX
optimizer loop — the only shape that *could* route a loss through the scatter. Reading each: **4 are
campaign-live AND contaminating**, 1 is campaign-live but reads clean, 1 is not campaign-live:

| file | verdict | how established |
|---|---|---|
| `experiments/train_tr1_partition_renderer_mlx.py` | **CONTAMINATING** | `:661` imports `_apply_R`; `:2012` `optim.Adam`; the live TR1 trainer, dt1's own subject |
| `experiments/train_witness_realized_through_R_mlx.py` | **CONTAMINATING** | defines `_apply_R`; own optimizer loop (dt1 §8.4, source-verified, floors UNMEASURED) |
| `experiments/train_levelset_witness_realized_through_R_mlx.py` | **CONTAMINATING** | calls `_base_da._apply_R`; own optimizer loop (same status) |
| `experiments/ddm_pj1_token_projection_fit.py` | **CONTAMINATING** | `:205` `from …train_witness… import _apply_R`; `:240` `optim.Adam`; `:243` `loss_fn` = `_apply_R(model.render_frame(i))` — the loss is *literally* R∘render |
| `experiments/ddm_fp1_class_field_projection.py` | **NOT contaminating** (read, not assumed) | its two gradient paths are *disjoint from the MLX R*: the prototype solve is **CPU-torch** (`:162 import torch`, `:195 torch.optim.Adam`, `:202 torch.nn.functional.interpolate`), and the MLX head loss (`:432-442`) is CE at 384×512 on a `mx.stop_gradient` trunk with **no R in the graph** |
| `experiments/sg_drf_single_frame_feasibility_probe.py` | contaminating-capable, **not campaign-live** | no DDM run dir |

**Census B — the run directories (what DID contaminate).** **113** `ddm_*` run dirs
(111 on `/Volumes/VertigoDataTier/pact` + 2 in `experiments/results`) scanned for a **locally
written** trained/fit artifact: `stage_*.npz` · trainer `telemetry.jsonl` · `fit_state.npz` ·
`head_state.npz` · `tr1_window_receipt.json` · `race_receipts.json`. **12 of 113** wrote one.

**Instrument controls I ran on my own scan, because a census is exactly where vacuity hides:**
- **First version was WRONG and I caught it.** Signature `{checkpoints/ dir, telemetry.jsonl}` returned
  13 dirs and **MISSED `ddm_pj1`** — whose fit state is `warm_l2/fit_state.npz`, a name the signature
  did not know. A detector that misses a *known* contaminated run is a failed positive control, so the
  signature was widened and re-run. **`ddm_pj1` is the positive control and it now fires.**
- A broader substring pass returned 33/111; hand-inspection showed 21 of those matched only files
  *named* like `checkpoints_*` inside pure analysis dirs (`ms2r`, `rg1-3`, `pf2/pf3`, `sn1`, `g3/g4`).
  Those are **false positives of the loose pattern**, and the tight signature above excludes them.
- **Stated boundary (this is a scope, not a clean bill):** `/Volumes/APDataStore/pact` is **NOT
  MOUNTED** and was not scanned. Non-`ddm_*`-prefixed dirs (the pre-DDM campaign) are out of scope.
  So "12 of 113" is *"12 found in the scanned scope"*, **never** "12 exist".

---

## §2 THE CLASSIFICATION — the 8 named candidates (denominator 8/8 classified, 0 UNDETERMINED)

| # | row | class | code-path evidence |
|---|---|---|---|
| #863 | bp1 bias-correction A/B | **RETRAIN-BASED** | 2 × 40-epoch TR1 windows, `ddm_bp1_20260731/{arm_A,arm_Bprime}`, both `stop_reason=epochs_complete` |
| #824 | η(t) transient surface | **RETRAIN-BASED** — *same two runs as #863* | the "surface" is the harvested bp1 diagonal; no additional run exists |
| #815 | bs1 excursion A/B | **NOT FIRED** (no verdict to grade) | no `ddm_bs1_*` run dir exists. What DID fire is the margin-density preflight — a read of a banked QA80 atlas, **scorer-free, no checkpoint, no gradient** ⇒ SAME-CHECKPOINT-class |
| #790 | dw1 distill-window | **RETRAIN-BASED** | 3 matched 40-ep TR1 windows, `ddm_dw1_20260730` (stage ckpts + telemetry + window receipts) |
| #799 | fp1 f′ — **F1 receiver floor 0.008305 (the load-bearing number)** | **SAME-CHECKPOINT / forward-only** | `paint_argmax_to_camera_uint8` = numpy bicubic (`:150-154`); `realized_gate` = `load_real_segnet("cpu")` + `cpu_verdict_d_seg_argmax_batch` (`:248-283`) — a CPU-torch forward. **Checkpoint-independent**: it paints *GT* argmax |
| #799 | fp1 f′ — trained-head 0.499366 (the secondary wall) | **TRAINS, but NOT through R** | MLX Adam on a 5-ch conv head, loss has no R (§1). Not contaminated by the measured mechanism — INFERRED from dt1 rows 5–7 (generic MLX ops + SegNet fwd/bwd clean), not measured on fp1's own config |
| #788 | pj1 capacity floor f | **RETRAIN-BASED** | `ddm_pj1_20260730/warm_l2/fit_state.npz`; the loss *is* `_apply_R(render)` under `optim.Adam` |
| — | pa1r rowband / margin-coupled-quant / delta-shrinkage | **RETRAIN-BASED** (2 of 4 arms); **rowband + joint NEVER RAN** | `ddm_pa1r_20260730` stage ckpts + `race_receipts.json`; 59-epoch D16 warm tails. rowband blocked on grid (no D8 parent exists), joint deferred |
| #800 | gc12 BR-D adjudication | **MIXED — one leg each way.** §6 | — |

**Corpus-wide (denominator 113 run dirs): 12 retrain-capable, 101 SAME-CHECKPOINT by construction,
0 UNDETERMINED within the scanned scope.** The 12: `tb1`, `lv1`, `dw1`, `pa1r`, `bc1`, `b4s`, `bp1`,
`lg1`, `r1c`, `pj1`, `fp1`(head only), `dt1`. Everything else — the byte-close/eval/solve/atlas/
coder/analysis population, **101 dirs** — wrote no trained artifact and re-runs no VJP.

**A distinction I nearly let slide, so I am stating it explicitly: contamination lands on
DELTAS ATTRIBUTED TO LEVERS, not on absolute measured S.** An exact `evaluate.py` row on a fixed
archive is a fact about *those bytes* and stays a fact — nothing about #903 touches it. But the
*checkpoint* those bytes came from was produced by a training run in the **12**. So:
"archive X scores S" is **sound**; "this endpoint is the best reachable / the continuation
dividend is −0.011 S / lever L is worse" are **single draws** from a cloud whose width was never
measured. The own-vehicle line's *absolute* values are not impugned; the *attributions* along it
are what §3 ranks. Row 5 of the ranking is exactly this case and is listed for that reason.

MAIN's pre-cleared rows are consistent with the mechanical census and I did not re-litigate them:
`ddm_v4d_20260731`, `ddm_ms8_20260802`, `ddm_cr1/cr2`, `ddm_ll1`, `ddm_gr1` all appear in the
**101**, i.e. they contain no locally-written trained artifact at all.

---

## §3 THE RANKING — |effect| / floor, at both bracket ends

Floor transferred as dt1's **relative** statistic (the only transferable one dt1 licenses):
`floor_S(row) = rel × 100 × d_seg_at_that_row`, `rel ∈ {0.082, 0.397}`. Sorted by exposure.

| rank | row | \|ΔS\| | d_seg op. | floor@8.2% | floor@39.7% | **r@8.2%** | **r@39.7%** | re-grade |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | pa1r delta_sparsity vs control @ep464 (matched budget) | 0.00312 | 0.005271 | 0.0432 | 0.2093 | **0.07** | **0.01** | UNGROUNDED |
| 2 | pa1r control_tail vs B — the "NEW BEST" −0.011 | 0.01100 | 0.004941 | 0.0405 | 0.1962 | **0.27** | **0.06** | UNGROUNDED |
| 3 | #863/#824 bp1 arm_B′ vs arm_A | 0.01196 | 0.004200 | 0.0344 | 0.1667 | **0.35** | **0.07** | UNGROUNDED |
| 4 | pa1r delta_sparsity final vs control final | 0.02350 | 0.005214 | 0.0428 | 0.2070 | **0.55** | **0.11** | UNGROUNDED |
| 5 | #790 dw1 arm C (chart-relax) vs B | 0.03390 | 0.005115 | 0.0419 | 0.2031 | **0.81** | **0.17** | UNGROUNDED |
| 6 | #790 dw1 B−A — **the fork discriminator** | 0.03820 | 0.005115 | 0.0419 | 0.2031 | **0.91** | **0.19** | UNGROUNDED |
| 7 | pa1r margin_quant vs control | 0.06440 | 0.005587 | 0.0458 | 0.2218 | **1.41** | **0.29** | UNGROUNDED |
| 8 | #788 pj1 f_photometric vs the gc9 2.6e-3 threshold | 50.22 | 0.504824 | 4.1396 | 20.0415 | **12.13** | **2.51** | **STANDS** |

**Threshold rows, tested the hostile way** — not `(X−T)/floor`, but *"does the claim survive shrinking
the measurement by the full relative bracket?"* `X·(1−rel)` vs `T`:

| claim | @8.2% | @39.7% | verdict |
|---|---:|---:|---|
| fp1 f′ ≥ 0.008305 vs **BR-D's 2e-3** | 3.81× | **2.50×** | survives (and is forward-only, so the test is gratuitous) |
| fp1 f′ ≥ 0.008305 vs **BR-B's 0.0051** | 1.49× | **0.98×** | would be *marginal* under hostile transfer — see §6 |
| pj1 0.504824 vs gc9's 2.6e-3 | 178× | **117×** | survives overwhelmingly |

**Two readings that matter more than the ordering:**

1. **Seven of the eight ungrounded rows are NEGATIVES** — levers judged worse, families closed, a
   burn-3 opening declared NO-GO. Per the weight-positives discipline, an ungrounded negative is the
   costlier kind: it *removed* options. Rank 2 is the exception and cuts the other way — the
   `control_tail` "NEW BEST realized point" is a **positive** claim also inside the floor, so the
   exposure is genuinely bidirectional and I am not sign-flipping anything.
2. **dw1's own under-drive guard is inside the floor too.** dw1 guard-4 passed on
   *"B total window descent 2.45e-4 = 8.2× noise ⇒ NOT under-driven."* 2.45e-4 d_seg = 0.0245 S =
   **0.58×** the 8.2%-end floor. The guard that certified the experiment had enough signal was
   calibrated against the same blind estimator.

---

## §4 THE ROOT DEFECT — a within-run statistic used as a between-run threshold

This is a code-level finding, not a reading of prose.

```
src/tac/witness_dsl/ax1_pool_a_race_20260730.py:59-62
  #: d_seg measurement noise floor (dw1 precedent 2.99e-5; re-derive per-race from a control
  #: repeat if available).  The additive-S noise band = 100·(this) …
  DEFAULT_DSEG_NOISE_FLOOR: float = 2.99e-5

:438   s_band = 100.0 * float(dseg_noise_floor)          # 0.00299 S
:464   if ds < -s_band: better.append(r.lever)           # ← the BR-A predicate, verbatim
```

`dw1` §5 defines the constant: **"noise floor = 2.99e-5 (B's gate residual std about its own trend)."**
`pa1r` §7 records that it did **not** re-derive it: *"the dw1 precedent, not re-measured here
(control_tail was a drift arm, not a repeat)."* The code comment even names the missing input —
*"re-derive per-race from a control repeat if available"* — and no race ever had one.

**Understatement of the operative band vs the dt1 bracket, at each row's own operating point:**

| lineage | d_seg | band used | floor @8.2% | floor @39.7% | understated by |
|---|---:|---:|---:|---:|---:|
| pa1r control_tail | 0.0049411 | 0.00299 S | 0.0405 S | 0.1962 S | **13.6× – 65.6×** |
| bp1 arms (#863) | ~0.0042 | 0.00299 S | 0.0344 S | 0.1667 S | **11.5× – 55.8×** |
| dw1 B (#790) | 0.0051147 | 0.00299 S | 0.0419 S | 0.2031 S | **14.0× – 67.9×** |

Why the estimator is *structurally* blind, not merely small: gate-to-gate residual about a trend
measures the **jitter of one trajectory**. Run-to-run divergence is a **displacement between
trajectories** that a single trajectory contains no sample of. This is the "governance knobs are
secretly optimizers" / unladdered-control-provenance class: a decision threshold whose provenance
rung was never checked, inherited by default into every race.

**Consequence for the campaign's habit, stated plainly:** *n=1 per arm* was the norm — `#863` says
so itself (*"n=1. Both arms seed=0, single run, no replicate ⇒ NO MEASURED NOISE FLOOR"*), and
`pa1r` §7 flags its own matched-epoch call as *"only ~1.04× the noise band."* With n=1 and a blind
band, "beyond noise" was never actually tested by any of these rows.

---

## §5 A BANKED, ALREADY-PAID REPLICATION — and one conflict I am not going to smooth over

### 5.1 The replication ($0, from artifacts, no run fired)

51 `tr1_window_receipt.json` files carry a `config_hash`. **Three hashes appear more than once** —
i.e. the campaign already paid for same-config repeats and never read them as such. All in
`ddm_lg1_20260731` (n=4 pairs, 2 epochs, seed 0, lr 2e-3, `variant=plain`):

| config_hash | runs | realized gate d_seg | range (S) | rel. range |
|---|---|---|---:|---:|
| `19e94034943a` | `cpu_post_off`, `cpu_post_off2`, `smoke_post_off`, `smoke_post_off2` | 0.454381 / 0.481303 / 0.447647 / 0.441348 | **3.9955** | **8.76%** |
| `dcad32a8f17e` | `cpu_pre_off`, `cpu_pre_off2`, `smoke_pre_off` | 0.506104 / 0.521726 / 0.445731 | **7.5995** | **15.47%** |
| `167e92182b7f` | `smoke_post_on`, `smoke_post_on2` | 0.489015 / 0.524578 | **3.5563** | **7.02%** |

**Relative spreads 7.0–15.5%, from a different arm, a different config, a different week —
independently consistent with dt1's 8.2–39.7% bracket at its low end.** #903 is replicated and is
not a property of dt1's harness. (Same caveat as dt1's: n≤4 ⇒ each range is a LOWER bound, and these
are n4/2ep, *even further* from convergence than dt1's n6 — so this replicates the premise, **not**
the operating-point floor.)

The lg1 memo and its custody manifest state the conclusion outright, on 2026-07-31:
`"vehicle_rerun_nondeterministic": true`, *"the tr1 vehicle is rerun-NONDETERMINISTIC with identical
code+argv on BOTH devices … counted bytes vary ±2 even at ep0."* It was filed under *Honest
negatives / limitations* of a lane-guard byte-identity receipt. **Nothing consumed it.**

### 5.2 The conflict — stated, not resolved

**lg1 claims forced-CPU is ALSO nondeterministic** (`cpu_post_off` 0.4544 vs `cpu_post_off2` 0.4813;
`cpu_pre_off` 0.5061 vs `cpu_pre_off2` 0.5217). **dt1 measured MLX-CPU CLEAN** — 41/41 arrays and
134/134 telemetry fields identical over 3 repeats. Both are MEASURED claims and they disagree.

**Why I cannot settle it from the artifacts:** `--mlx-device` exists on the trainer (`:1724`,
introduced 2026-07-28 by `ddm_tb1`, so lg1 *could* have used it) — but **`mlx_device` is absent from
the persisted `cfg`, from `tr1_config.json`, and therefore from `config_hash`.** Two runs with an
identical config hash may have executed on different devices, and nothing on disk records which. The
`cpu_*` prefix is a directory name, not a receipt field.

That is itself a defect worth naming: **the device is not in the config hash**, so the campaign's
own "identical config" identity relation does not include the thing dt1 proved is decisive.

**The resolving measurement (cheap, named, NOT fired here):** 2 × `--mlx-device cpu` windows at the
lg1 config (n4, 2 ep, ~15 s each) through dt1's landed `tools/ddm_dt1_compare_run_determinism.py`
with its mandatory `--self-check`. If they are bit-identical, lg1's `cpu_*` runs were mis-labelled
and dt1's cure is complete; if they differ, there is a **second** nondeterminism source and
`--deterministic-r` is a partial cure. I did not fire it: it is a training run, and my charter is the
classification. **Until it is fired, "R was the only source" remains scoped to dt1's control lever
set, exactly as dt1 itself declared.**

---

## §6 ELEVATED — task #800 (gc12), the branch selection

**The predicate, verbatim from gc11 §2:**

| branch | condition | action |
|---|---|---|
| BR-A | *any pa1r arm beats control in additive S at matched bytes (beyond its noise floor)* | in-loop rate burn on B |
| BR-B | `f′ ≤ 5e-4` realized n600 | class-field graft vehicle |
| BR-C | BR-A ∧ BR-B | sequence both |
| **BR-D** | **pa1r all-on-contour ∧ f′ > 2e-3** | **THE WALL BRANCH; convene the 14th convocation ONLY here** |
| mid-band | `5e-4 < f′ ≤ 2e-3` | treat as BR-B with re-derived budget |

### 6.1 Leg 1 — `f′ > 2e-3`: **GROUNDED. It stands, and it is not touched by #903 at all.**

The measured `f′ ≥ 0.008305` is fp1's **F1 receiver floor**, and its code path contains **no MLX
gradient, no R adjoint, and no trained TR1 checkpoint**:

- the field painted is **GT argmax** — so no checkpoint enters;
- the paint→camera step is `bicubic_up_to_camera_float`, deterministic numpy (`:150-154`);
- the scorer is `load_real_segnet("cpu")` + `cpu_verdict_d_seg_argmax_batch`, a **CPU-torch forward**
  (`:248-283`);
- the only fitted object is the 15-number prototype table, solved in **CPU torch**
  (`:195 torch.optim.Adam`, `:202 torch.nn.functional.interpolate`) — not MLX, not Metal, not the
  scatter dt1 isolated;
- and the branch-relevant statement is the **≥ argument**, which is combinatorial, not empirical.

Margin: **4.15×** over the 2e-3 threshold. Under a *deliberately hostile* transfer of the floor
(shrink the measurement by the full 39.7%) it is still **2.50×**. **Leg 1 needs no re-measurement.**

*Two honest riders, both OUTSIDE #903's scope so I am not laundering them into this verdict:*
(a) the same hostile transfer puts fp1 vs the **BR-B** threshold 0.0051 at **0.98×** — so if anyone
ever wants BR-B's death re-stated with a floor attached, that one is marginal where BR-D's is not;
(b) "any head ≤ GT can only ADD flips" is asserted, not proven — a head that pre-compensated for
R+SegNet distortion could in principle beat the GT-painted field through the receiver. That is a
**separate, non-determinism exposure**; I flag it and leave it.

### 6.2 Leg 2 — `pa1r all-on-contour`: **UNGROUNDED.**

The predicate is literally `ds < -s_band` with `s_band = 0.00299 S` (§4). The arms:

| arm | ΔS vs control | r@8.2% | r@39.7% |
|---|---:|---:|---:|
| delta_sparsity (matched budget) | +0.00312 | 0.07 | 0.01 |
| delta_sparsity (full budget) | +0.02350 | 0.55 | 0.11 |
| margin_quant | +0.06440 | 1.41 | 0.29 |
| rowband | **NEVER RAN** (no D8 parent exists) | — | — |
| joint | **NEVER RAN** (deferred, sealed) | — | — |

**The asymmetry that makes this branch-selecting rather than merely noisy:** BR-A is a
***"any arm BEATS control"*** predicate. Noise able to author a spurious *worse* is equally able to
author a spurious *better*. So a floor that swamps these margins does not just weaken the "arms are
worse" reading — **it leaves BR-A's truth value undetermined in both directions.**

**And BR-A is the only live alternative.** BR-B is excluded by leg 1 (uncontaminated); mid-band is
excluded by leg 1 (uncontaminated); BR-C requires BR-B, so it is excluded too. Therefore **the entire
branch decision reduces to BR-A vs BR-D, and that reduction is decided solely by the leg that sits
inside the floor.**

### 6.3 Verdict on #800, and the honest bound on what it cost

**BR-D is HALF-GROUNDED: its f′ leg is solid; its pa1r leg is UNGROUNDED. The convocation that
selected the campaign's roadmap turned on a predicate whose measured margins are 0.07×–1.41× the
floor. That is a branch selection inside the noise, exactly as #908 feared.**

Three things I will not overstate:

1. **A structural leg of BR-A's falsity does not depend on effect size at all**: 2 of 4 pa1r arms
   **never ran**, so "any arm" was evaluated on half the arm set regardless of noise. That weakness
   predates #903 and is not cured by re-measuring.
2. **pa1r's three consistent reads are not three samples.** The memo rests the direction jointly on
   the matched-epoch gap, the endpoint gap, and the A1 realized-flatness refuse. All three are
   computed from **the same single pair of trajectories**. Consistency *across metrics on one
   trajectory pair* is not replication *across runs* — it is precisely what a within-run estimator
   would also show.
3. **The realized damage looks bounded, and saying so is part of being honest.** BR-D's own terminal
   verdict came back negative on its own merits (`qa92` rung-0: `P·O = 0.017 < 0.05` ⇒ the Contrarian
   bound fired ⇒ rung 2 SKIPPED), and every pointer move since — `v4d → pw1 → ms8` — came from
   same-checkpoint rate work in the **101**. So the recommended action is **not** to re-litigate the
   convocation. It is to re-open the two *levers* the ungrounded leg closed (§7).

---

## §7 RE-GRADES — bidirectional, each with its resolving measurement

**UNGROUNDED ≠ refuted. No sign is flipped. Each row keeps its recorded direction as a hypothesis and
loses its status as evidence.** Ordered by re-measurement priority = exposure × what the verdict closed.

| # | row | was | now | resolving measurement |
|---|---|---|---|---|
| 1 | pa1r `delta_sparsity` (in-loop group-L2) — `worse_s`, exchange 0.14–0.16, **exited the burn stack** | measured negative, INSTANCE scope | **UNGROUNDED** (0.07× / 0.01×) | N≥3 same-seed repeats of control_tail **and** the delta arm, **both under `--deterministic-r`**, at the pa1r config; ΔS re-read against the measured between-run range |
| 2 | pa1r `margin_quant` — `worse_s` ΔS +0.0644, **exited the burn stack** | measured negative | **UNGROUNDED** (1.41× / 0.29×) | same protocol, same parent |
| 3 | #790 dw1 B−A — **the fork discriminator**; authored "burn-3 distill-opening NO-GO" and "optimization/capacity leads" | measured negative, 12.8× "noise" | **UNGROUNDED** (0.91× / 0.19×) — the 12.8× is 12.8× a *within-run* std | re-run the 3 matched 40-ep windows under `--deterministic-r`, or N≥3 repeats of B alone to get the true between-run band at d_seg≈0.0051 |
| 4 | #863/#824 bp1 — "bias_correction OFF wins by ~0.012 S" | n=1 diagonal, already self-caveated | **UNGROUNDED** (0.35× / 0.07×) | the longer window #824 already specifies (≥3–4 β₂ time constants) **plus** ≥2 seeds, under `--deterministic-r`. The 2×2 off-diagonal cells stay owed independently |
| 5 | pa1r `control_tail` vs B — the **"NEW BEST realized point"** −0.011 S | positive, headline | **UNGROUNDED** (0.27× / 0.06×) — *listed explicitly so the re-grade is not one-directional* | N≥3 repeats of the 58-epoch continuation under `--deterministic-r`; is the continuation dividend reproducible or a draw? |
| 6 | #790 dw1 arm C (chart-relax) vs B | measured negative | **UNGROUNDED** (0.81× / 0.17×) | rides #3 |
| 7 | dw1 guard-4 "B is NOT under-driven (8.2× noise)" | passed guard | **UNGROUNDED** (0.58× at the 8.2% end) | rides #3; the guard must be re-expressed against a between-run band |
| 8 | #788 pj1 f_photometric = 0.5048 ⇒ probe NOT-ADJUDICABLE | measured, FORMULATION scope | **STANDS** (12.13× / 2.51×; hostile-shrink 117×) | none. Its mechanism (R1–R5: the frozen renderer floors at mean 67.95 and cannot reach mean-20 targets) is a set of **forward-only** pixel statistics anyway |
| 9 | #799 fp1 F1 receiver floor ⇒ BR-B dead, `f′ > 2e-3` | measured lower bound | **STANDS — not retrain-based** | none for #903 purposes (see §6.1 riders for the two non-#903 exposures) |
| 10 | #799 fp1 trained-head f′ = 0.4994 | measured, INSTANCE scope | **STANDS** — trains, but not through R; and 150× over the BR-D threshold under hostile shrink | none |
| 11 | #815 bs1 A/B | **never fired** | no verdict exists to grade | when it fires it must run under `--deterministic-r` with N≥2 seeds, and per its own #815 note must vary **excursion magnitude**, not only cadence |
| 12 | #815 bs1 ρ₀ preflight (flip-budget density 0.11091) | measured, scorer-free | **STANDS** | none — static atlas read, no checkpoint, no gradient |
| 13 | #800 BR-D | AUTO-fired branch | **HALF-GROUNDED** (§6) | rows 1–2 are the resolving measurements. Do **not** re-run the convocation |

**A blocker on those re-measurements, found while writing them down and FIXED in this unit
(commit `06fa0ad37d`).** Every row above says *"re-run under `--deterministic-r`"* — and
`experiments/train_tr1_partition_renderer_mlx.py --help` **CRASHED**
(`TypeError: %o format: an integer is required, not dict`), so that flag was **invisible** on the
trainer that needs it. Same crash on
`experiments/train_levelset_witness_realized_through_R_mlx.py` (the canonical capstone θ* trainer
CLAUDE.md names). Cause: argparse renders help as `text % params` with `params` a **dict**, so one
unescaped `%` raises; nothing in CI ever calls `--help`. An AST sweep of all 10,591 tracked `.py`
found **7 more** across 6 further files — the predicted 6–7× spread. All 10 escaped, repo-wide live
count now **0** (2,718 files declaring `add_argument`, 10,428 literal help strings), guarded by
`src/tac/tests/test_cli_help_strings_render.py` with a positive control, a negative control, and a
denominator assertion so a collapsed scope cannot pass silently.

**The one apparatus change that would retire this whole class** (naming it, not building it — that is
a separate landing and I am not paying a new-machinery debt inside an audit): `DEFAULT_DSEG_NOISE_FLOOR`
must be **refused** when a race has no same-config repeat, instead of silently defaulting to a
borrowed within-run number. The code comment already asks for it (*"re-derive per-race from a control
repeat if available"*); making the *absence* fail-closed rather than default-quiet is the two-landing
sister of dt1's cure. Sister defect from §5.2: **put `mlx_device` inside `config_hash`.**

---

## §8 WHAT I DID NOT DO, AND WHY

- **STEP 1, the operating-point floor: NOT MEASURED.** Per MAIN's inversion. It needs N≥4 training
  windows from a live-vehicle checkpoint (d_seg ≈ 0.0039) × 2 arms — hours of trainer time and a
  scorer slot `ddm_bp2` may hold. Everything in §3 is therefore an **extrapolation of dt1's relative
  statistic**, labelled as such in every table header, and every ratio is reported at **both** ends
  because the direction of the change near convergence is genuinely unknown (dt1 and #908 both say
  it could go either way; §5.1's replication is also far-from-converged and does not settle it).
- **I re-ran no A/B**, per charter. The ranking exists so re-measurement can be targeted.
- **I fired no training run** — including the one in §5.2 I would most like to have.
- **Triality:** `[no-triality]`, per this arm's dispatch. The `flip_budget_density_at_zero_v1`
  equations-leg debt from #815 is still owed and is not mine.

## §9 verdict_scope ledger

- Falsifier NOT met on both clauses at both bracket ends: **MEASURED** (arithmetic over dt1's
  published bracket and the rows' own published receipts; reproducible from the tables).
- Code-path classification of the 8 named candidates: **MEASURED by source inspection**, file+line
  cited for each; `ddm_fp1` and `ddm_pj1` were read, not inferred from their names.
- "Exactly 4 campaign-live contaminating code paths": **negative-existence claim, scoped** to 10,591
  tracked `.py` files under the token set in §1 Census A. Untracked files, notebooks, and shell-only
  paths are outside it.
- "101 of 113 run dirs are same-checkpoint": **scoped** to `ddm_*`-prefixed dirs on the mounted
  VertigoDataTier + `experiments/results`. **APDataStore was not mounted.**
- lg1 replication (7.0–15.5% relative over 3 duplicated `config_hash` groups): **MEASURED** from the
  banked receipts; n≤4 ⇒ LOWER bounds; n4/2ep ⇒ far-from-converged, **not** an operating-point floor.
- lg1-vs-dt1 CPU conflict: **UNRESOLVED**, and unresolvable from persisted artifacts because
  `mlx_device` is absent from the receipt cfg. Resolving measurement named in §5.2.
- Every re-grade in §7 is **UNGROUNDED = no evidence either way**, never "refuted", and never
  sign-flipped.

**Pointer: own-vehicle `ms8 0.8984335` [macOS-CPU advisory] UNMOVED by this unit; effective frontier
0.172141 UNMOVED. This memo is MEANS. It lowered no score and it says so.**

`[no-triality]` `[p0-ledger-ok]`
