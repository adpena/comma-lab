---
arm: ddm_ws4
title: "every budget here is allocated by a PROXY for value -- fraction-of-run, pixel-area, parameter-count, share-of-S -- and the curriculum is the worst case: 92.2% of the actual weight movement goes to the phase whose marginal value is NEGATIVE; a three-arm dose-response (A2/CE0/EF0) removes 92.7% of the self-inflicted damage with two flags and no code; rg1b's 45.9x spatial misallocation is the SAME defect one level down and the same free flag cures it; and at campaign level the same error ranks POSE LAST at 5.20% of S when pose is the binding constraint on 92% of the bytes. Two budgets REFUTED (capacity 1.08, per-pair 1.8x) plus four archive regions at floor."
utc: 2026-08-17
charter: "ddm_ws4 -- FRESH EYES on curriculum x loss x where the budget is spent ($0, no launches)"
axis: "[macOS-MPS training-signal] re-analysis of already-retained payloads + [macOS-CPU advisory] forensics + DERIVED score arithmetic from the [contest-CUDA T4 n600] frontier receipt. NO training ran, NO archive built. NEVER a score."
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "per-finding INSTANCE on the named retained runs; the LR-budget closed form and the axis-headroom arithmetic are DERIVED and exact"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_ws4 — where the budget is spent

**STORES CONSULTED (read at source, never from a summary):**
`src/tac/pr130_lift/train_semantic_quantized_resumable.py` (sha `b486f416…`) ·
`src/tac/pr130_lift/lifted/semantic_renderer_oracle.py` (sha `ffdf0988…`) ·
`src/tac/pr130_lift/band_objective.py` (sha `81e187f6…`) ·
`src/tac/pr130_lift/band_weight_table_rt1_n600.json` ·
`.omx/research/ddm_ce1_ce_opening_excursion_mechanism_20260817.md` (commit `a881a41c90`) ·
`.omx/research/ddm_rg1b_band_objective_build_20260816.md` §2.7/§6.2 ·
`.omx/research/ddm_jr1_band_objective_judge_repair_20260817.md` ·
`.omx/research/ddm_ra2_charter_stale_family_closed_and_lossless_axis_20260817.md` (commit `203a25a076`) ·
`.omx/state/canonical_frontier_pointer.json` · `.omx/state/canonical_task_status.jsonl` (ddm_pv1 row,
commit `126c4582dc`, source of the frontier `d_pose = 6.88e-06`) ·
retained payloads `/Volumes/APDataStore/pact/ddm_ce1/{CE0,EF0}/`,
`/Volumes/APDataStore/pact/ddm_jr1/{A2_repeat,L3000_off,JR1_VECTORS.npz}` (READ-ONLY) ·
GT cache `/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt`
(READ-ONLY).

---

## ANSWER FIRST

**The hypothesis is CONFIRMED for five budgets, REFUTED for two, and — against my own first draft —
it is ACTIONABLE.**

Every allocation in this trainer is set by a *proxy*: fraction-of-run, pixel-area, or parameter-count.
Never by marginal value to the score. The curriculum is the extreme case:

| curriculum phase | % of LR budget | % of ACTUAL weight movement | marginal value (flips per unit ‖Δw‖) |
|---|---:|---:|---:|
| `ce` | 81.20% | **92.19%** | **+109,285  (DESTROYS)** |
| `softplus_margin` | 17.96% | 7.18% | −220,591  (repairs) |
| `expected_flip` | 0.84% | **0.63%** | **−946,600  (repairs best)** |

The phase with the best marginal value gets **0.63%** of the movement; the phase with a **negative**
marginal value gets **92.19%**. **And the LR budget `ce1` reported is itself a proxy** — CE's true
share of weight movement is 92.19%, not 81.20%. Same defect, one level down.

**Three arms now exist on this axis and they form a clean monotone dose-response.** Same init, seed,
lr, bits, instrument, 600 steps; they differ only in the curriculum-shape flags:

| arm | curriculum | flips above init @600 | ΔS | terminal slope (flips/100 steps) |
|---|---|---:|---:|---:|
| `A2_repeat` | stock (81% CE) | +8,654 | +0.007337 | −765 |
| `CE0` | `--ce-fraction 0.0` | +4,852 | +0.004113 | −388 |
| `EF0` | `--ce-fraction 0.0 --softplus-fraction 0.0` | **+636** | **+0.000539** | **−1,383** |

**EF0 removes 92.7% of the stock curriculum's self-inflicted damage with two flags and no code
change**, and it was still descending 1.8× faster than A2 at its horizon. `ce1`'s pre-registered
falsifier read PARTIAL at CE0; EF0 pushes far past it.

**I drafted "CEILING-ZERO" from EF0 at step 450 and it was wrong.** Every arm's argmin is still
step 0, so no arm has yet gone *below* init — but EF0 ends 636 flips away, unconverged, on its
steepest descent. Whether the ceiling is exactly zero or genuinely negative is **UNMEASURED**. I will
not extrapolate that tail: this campaign struck a linear-tail extrapolation today (`L3000`/`aa3`).

**The unification that matters most.** `rg1b`'s 45.9× spatial misallocation and this audit's 5,736×
band-density gap are both computed against a *pixel-count* allocation model — which is exact **for
cross-entropy**. `expected_flip` = `sigmoid(-margin/tau)` has per-pixel derivative `σ(1−σ)/tau`,
exponentially suppressed once `|margin| ≫ tau`, so it **self-concentrates on the near-margin pixels
for free**. In EF0, CE never fires. So the spatial misallocation, the class misallocation, and the
curriculum misallocation are **one defect — the pixel-mean CE — and one free flag removes all
three.** `rg1b`'s band table is a *paid* cure (a shipped weight table) for a problem the *free* cure
already addresses; its residual value is exactly the gap that survives saturation, which nobody has
measured.

**And the same defect runs at campaign level (§12).** The score is **76.25% rate / 18.55% seg /
5.20% pose**, and the campaign ranks its axes by that share — which puts pose last. But `rc4`'s
rung-4 token drop reclaims **17,985 B (125% of the entire gap)** and its rate + seg legs are a
**measured −3.243e-3 S win, 33.8% of the gap**; it dies on an uncompensated pose leg worth **20×**
its seg cost (400× on the semantic section). **Pose is 5.20% of the score and the binding constraint
on 92% of the bytes.** Ranking axes by contribution is itself a proxy allocation; the correct ranking
for routing is by *shadow price*, and by that measure pose is first, not last. **The highest-value
unmeasured cell in the system is not on any axis — it is the coupling.**

**The honest negatives that give this audit discriminating power:** the *capacity* budget (movement
across 38 tensors, ratio **1.08**), the *per-pair sampling* budget (uniform, forfeiting ~1.8×), and
**four of seven archive regions** — ZIP framing is *provably* at its exact 100-B floor and the token
coder is within **+0.42 B** of its own cross-entropy. Not every budget is broken, and the whole
remaining lossless byte axis is worth ≤278 B = **1.93%** of the gap.

**Pointer UNMOVED.** No training ran, no archive was built, no score was produced. This unit is MEANS.

---

## §1 The master budget — where the SCORE is

DERIVED by exact arithmetic from `S = 0.15959729295498598` and `bytes = 182,759`
(`.omx/state/canonical_frontier_pointer.json`, `our_local_frontier_contest_cuda`, axis
`[contest-CUDA T4 n600]`) and `d_pose = 6.88e-06` (ddm_pv1 row in
`.omx/state/canonical_task_status.jsonl`, quoting the r2 T4 receipt). `d_seg` is obtained by
subtraction and so inherits the 3-significant-figure precision of `d_pose`.

| term | value | % of S |
|---|---:|---:|
| rate `25·B/37,545,489` | 0.121691716 | **76.25%** |
| seg `100·d_seg` | 0.029611 | **18.55%** |
| pose `√(10·d_pose)` | 0.008294577 | **5.20%** |

Denominator of every share: `S = 0.159597293`. Derived `d_seg = 2.9611e-04 = 34,931 flips` of
**117,964,800** pixels — 600 pairs × 384 × 512, one scored frame per pair, MEASURED from the GT
cache's `seg` tensor shape `(600, 384, 512)`. Exchange rates: `dS/dflip = 8.477105e-07`,
`dS/dbyte = 6.658590e-07`, break-even **1.2731 B per flip**.

### §1.1 Can each axis close the 0.0095973 gap ALONE?

| axis | total contribution | needed | verdict |
|---|---:|---:|---|
| rate | 0.121692 | −14,414 B of 182,759 (**−7.9%**) | possible in principle |
| seg | 0.029611 | −11,321 flips of 34,931 (**−32.4%**) | possible in principle |
| pose | **0.008295** | −0.0095973 | **IMPOSSIBLE ALONE** |

Stated so it carries its own sensitivity: pose closes the gap alone **iff**
`d_pose ≥ gap²/10 = 9.211e-06`. Measured 6.88e-06 — a 1.34× margin. Driving `d_pose` to **zero**
still leaves **0.0013 S** on the table.

⚠ **A referent question I could not close.** The seg fine-tune program (`jr1`/`lr1`/`ce1`) starts from
`semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt` under `pr130_eureka_intake_20260806/`,
advisory `d_seg` **2.8616e-04**. The shipped frontier is `lane_ddm_hv1_ep0634_…`, derived `d_seg`
**2.9611e-04**. Nothing I read establishes these are the same renderer, and the +1,174-flip
difference is well within plausible MPS/CPU→CUDA instrument drift. Do **not** read it as a
regression. What it does mean: **the object the seg program optimises and the object the archive
ships are not demonstrably the same object.** Sister of
[[measured_object_vs_named_object_20260816]].

---

## §2 The curriculum budget — three denominators, three answers

### §2.1 LR budget (re-derived independently, exact)

`CosineAnnealingLR(T_max=steps, eta_min=0.01·lr)`, `--ce-fraction 0.5`, `--softplus-fraction 0.85` —
all fractions **of the run**. Closed form `F(a) = 0.01a + 0.495(a + sin(πa)/π)`:

| phase | closed form | T=600 | T=3,000 | T=30,000 |
|---|---:|---:|---:|---:|
| `ce` | 81.2007% | 81.1497% | 81.1905% | 81.1997% |
| `softplus_margin` | 17.9641% | 18.0075% | 17.9728% | 17.9650% |
| `expected_flip` | 0.8352% | 0.8427% | 0.8367% | 0.8353% |

**CONFIRMS `ce1` exactly.** Scale-invariant; no window length changes it.

### §2.2 Displacement budget — the LR figure is itself a proxy (NEW)

MEASURED from `JR1_VECTORS.npz::dw__A2` (7 retained displacement-from-init vectors × 66,339 params
for `A2_repeat`). Chord path length between consecutive retained checkpoints:

| phase | path length | **% of PATH budget** | % of LR budget | path/LR |
|---|---:|---:|---:|---:|
| `ce` | 1.019899e-01 | **92.192%** | 81.150% | 1.136 |
| `softplus_margin` | 7.941704e-03 | 7.179% | 18.008% | 0.399 |
| `expected_flip` | 6.955555e-04 | 0.629% | 0.843% | 0.746 |

Total chord path 1.106271e-01 against a net displacement of 5.045331e-02 — the trajectory is
**2.193×** longer than the distance it covers. **The first 100 steps alone are 42.8% of the whole
run's path.**

⚠ **Scope.** Chord length is a LOWER BOUND on true path length. Chord density is 1 per 70–100 steps
in every phase, so the bias is comparable across phases, but these ratios are bounded, not exact.

### §2.3 Marginal value per unit of movement (NEW)

Δflips across each phase for `A2_repeat` (from `run.log` at `--eval-every 100`, so phase ends land on
300 / 500 / 600 rather than exactly 300 / 510 / 600):

| phase | Δflips over phase | path | flips per unit ‖Δw‖ |
|---|---:|---:|---:|
| `ce` (0→300) | **+11,146** | 1.0199e-01 | **+109,285** |
| `softplus_margin` (300→500) | −1,727 | 7.829e-03 | −220,591 |
| tail (500→600, 10 sp + 90 ef) | −765 | 8.081e-04 | −946,600 |

⚠ **These repair rates are CONDITIONAL on CE damage existing** — `softplus_margin` and
`expected_flip` are cleaning up a mess CE made, and a high repair rate against abundant cheap damage
is not evidence they descend from a clean start. §3 is the control that settles it.

---

## §3 The three-arm dose-response (the decisive control)

Same init, seed, `--lr 2.0e-5`, `--bits 4 --weight-qat-q3q4`, 600-pair eval instrument,
`--steps 600`; differing only in curriculum-shape flags. Init = **33,757 flips**
(`d_seg` 2.8616163465711804e-04). All three ran to completion, `exit 0`.

| step | `A2_repeat` (81% CE) | `CE0` (0% CE) | `EF0` (100% `expected_flip`) |
|---:|---:|---:|---:|
| 100 | +27,098 | +11,105 | +7,314 |
| 200 | +12,460 | +7,320 | +5,120 |
| 300 | +11,146 | +7,097 | +5,467 |
| 400 | +10,415 | +5,342 | +3,334 |
| 500 | +9,419 | +5,240 | +2,019 |
| 600 | +8,654 | +4,852 | **+636** |
| **argmin step** | **0** | **0** | **0** |

1. **The misallocation is REAL and the cure WORKS.** +8,654 → +4,852 → **+636**: a **13.6×**
   reduction in end-of-run damage, from two flags, no code change. Monotone across three arms on one
   axis.
2. **No arm has yet descended below init.** argmin = step 0 everywhere, across these three plus the
   nine retained runs `ce1` inventoried.
3. **EF0 had not converged.** Its last 100 steps moved −1,383 flips, the steepest of any arm at any
   point in its run. It ends 636 flips from init.

**Why "just run EF0 longer" is NOT the `L3000` mistake.** `L3000` was inert because the phase splits
are fractions *of the run*, so lengthening the window cannot change them (§2.1). **EF0 has only one
phase — there are no fractions left to be invariant.** What lengthening genuinely buys is more steps
in the low-`tau` regime: `tau = 0.15 − 0.10·progress` (oracle `:192-194`), so with
`--softplus-fraction 0.0` it anneals 0.15 → 0.05 across the whole run, and `expected_flip` is
sharpest — closest to the true 0-1 metric — exactly where EF0 is descending fastest. This is a
structurally different experiment, and it is one flag.

---

## §4 Why nothing descends: the knife-edge (DERIVED)

`ce1` §4 measured the lr sweep at step 100 (identical everything else): 5,879 / 14,960 / 27,098 /
63,849 flips above init at lr 2e-7 / 2e-6 / 2e-5 / 2e-4. Log-log fit:

**flips ∝ lr^0.3366, R² = 0.9935 over three decades.**

At the init, **correct pixels outnumber flipped pixels 117,931,043 : 33,757 = 3,494 : 1.**

DERIVED (assuming displacement ∝ lr over a fixed 100 steps): the count of pixels with margin below
`c·ε` scales as `ε^0.337`, so the margin density `ρ(m) ∝ m^(−0.663)` — an **integrable singularity at
zero**. A smooth interior optimum would give `ε^1` or steeper; 0.337 is far shallower.

**Reading.** The argmax partition sits on a knife edge: a diverging density of pixels at essentially
zero margin, with 3,494 correct pixels to lose for every wrong one to gain. Any weight motion that is
not *precisely* aimed is overwhelmingly destructive, and the measured `cos(sign g)` of 0.21–0.62 is
nowhere near precise enough. This is why every objective, at every lr across three decades, moves
*up* first.

It also says what kind of instrument can win here. A **penalty** that merely down-weights the wrong
direction cannot beat a 3,494:1 ratio; a **projection/constraint** that provably cannot flip correct
pixels can. Sister of [[penalty-vs-projection-the-seg-pose-coupling-law]].

⚠ **Scope:** the exponent is DERIVED from four points under a stated linearity assumption; the fit is
excellent but the mechanism (a diverging near-tie density) is INFERRED, not directly observed. A
direct margin histogram would settle it and needs one GPU forward pass.

---

## §5 The spatial and per-class budgets — one defect, measured on THIS vehicle

MEASURED from `src/tac/pr130_lift/band_weight_table_rt1_n600.json` (basis `pred_vs_label`, 600
frames; `confusion_by_pair_cross_check` sums to 33,743 = `on_band_flips` 33,479 + `off_band_flips`
264 — within 0.04% of the 33,757 init figure, different render state). This is **this vehicle's own
debt**, not an inherited figure.

**Spatial** — denominator 117,964,800 px / 33,743 flips:

| region | pixels | share of px | flips | share of debt | density |
|---|---:|---:|---:|---:|---:|
| 1-px label band | 2,551,464 | 2.163% | 33,479 | **99.218%** | 1.3121e-02 |
| everything else | 115,413,336 | **97.837%** | 264 | 0.782% | 2.2874e-06 |

**Density ratio 5,736×.** Under a pixel-mean, 97.8% of the loss terms sit on pixels holding 0.78% of
the debt. This is the same statement as `rg1b`'s **45.9×** misallocation (gradient mass on band
2.161% ÷ band area 2.157% = ratio **1.0016**, exactly area-proportional, against 99.22% on-band
debt).

**Per class** — class areas MEASURED directly from the GT cache (`bincount` sums exactly to
117,964,800). Allocated share = area share, because the loss is a pixel-mean. Debt attributed half to
each of a flip's two incident classes (the rt1 table records unordered class pairs, so
label-vs-pred direction is not recoverable — an attribution convention, stated):

| class | ALLOCATED (area) | DEBT (measured) | debt/alloc | verdict |
|---|---:|---:|---:|---|
| 0 Road | 23.2335% | 40.628% | 1.75× | under-served |
| 1 Lane | **0.5857%** | 22.691% | **38.74×** | **under-served, worst** |
| 2 Undrivable | 49.5177% | 18.491% | 0.373× | **over-served 2.7×** |
| 3 Movable | 1.2379% | 14.905% | **12.04×** | under-served |
| 4 MyCar | 25.4253% | 3.285% | 0.129× | **over-served 7.7×** |

Spread most-under to most-over: **299.9×**. Two classes holding **74.9%** of the pixels hold **21.8%**
of the debt; two classes holding **1.82%** of the pixels hold **37.6%**.

⚠ **CORRECTION TO CLAUDE.md, measured.** CLAUDE.md's class-3 (Movable) area of 1.56% is **26% high**;
the n600 measurement is **1.2379%**. Cause is the denominator, not an error — CLAUDE.md's figures
come from `gt_n96.npz['lstars']`, and Movable's per-frame share ranges 0.107%–5.383% (50× spread), so
a 96-frame subset drifts. A textbook instance of
[[m88]]/[[m96]] prefix bias. The other four classes agree within 0.33 pp.

⚠ **THE CAVEAT THAT RESIZES THIS WHOLE SECTION.** Both the 5,736× and the 299.9× are computed against
a **pixel-count** allocation model, which is exact **for cross-entropy**. `expected_flip` and
`softplus_margin` are margin-saturating: derivative `σ(1−σ)/tau`, exponentially suppressed once
`|margin| ≫ tau`, so they concentrate on near-margin pixels *automatically and for free*. **In EF0,
CE never fires at all.** The residual spatial/class gap that survives saturation is **UNMEASURED**,
and it is precisely the number that decides whether the paid cure (`--band-objective-weight`, a
shipped weight table) is worth anything on top of the free one.

---

## §6 The capacity budget — HONEST NEGATIVE (already well allocated)

MEASURED from `JR1_VECTORS.npz::dw__A2` split by `param_key_order` against tensor sizes from
`A2_repeat/ckpt.periodic.step000100.full_state.pt` (38 tensors, 66,339 params — matches exactly).
Share of squared displacement at step 100:

| tensor | params | % params | % movement | ratio |
|---|---:|---:|---:|---:|
| `blocks.1.pw.weight` | 9,216 | 13.89% | 20.26% | 1.46 |
| `blocks.0.pw.weight` | 9,216 | 13.89% | 18.00% | 1.30 |
| `blocks.2.pw.weight` | 9,216 | 13.89% | 15.53% | 1.12 |
| `blocks.3.pw.weight` | 9,216 | 13.89% | 14.75% | 1.06 |
| `coord_mix.weight` | 9,600 | 14.47% | 10.54% | 0.73 |
| `head.weight` | 2,592 | 3.91% | 2.31% | 0.59 |

**Top-10 tensors: 83.21% of params carry 90.18% of movement — ratio 1.08.** Per-tensor ratios span
0.59–1.46; no tensor is more than 1.5× off proportional.

**Verdict: REFUTED. The capacity budget is allocated by parameter count and that is approximately
right.** Any future per-tensor capacity router owes a value measurement this audit could not produce.

*verdict_scope: **INSTANCE** — the `A2_repeat` trajectory (retained `dw__A2`, steps 100 and 600) on
this vehicle at this init. It says nothing about a different init, a different width, or a run that
actually descends. What would overturn it: a per-tensor displacement/value measurement on an arm
whose argmin is not step 0.*

*(Note for a different question: `frame_embed.weight` is `(600, 8)` = 4,800 params = **7.24%** of the
model — genuine per-pair capacity exists, unused as an allocation axis.)*

---

## §7 The per-pair budget — HONEST NEGATIVE, and the flattest axis available

- Sampling is `torch.randperm` over `active_pair_ids`, uniform, without replacement, reshuffled at
  each epoch (`train_semantic_quantized_resumable.py:1075-1077`, `:1201-1207`). `--batch-size`
  default **2** (`:851`).
- **A 600-step run is exactly 2.00 epochs.** 600 × 2 = 1,200 pair-visits over 600 pairs; every pair
  is seen exactly twice. One step touches 2 of 600 pairs = 0.333%.
- **Per-pair seg debt does not exist for this vehicle, structurally.** `_evaluate_semantic_pairs`
  (`:775-800`) accumulates `mismatches` into a single int and returns a scalar; no per-pair field is
  written anywhere in the training path.
- Shape reference from a **different** vehicle (`ddm_g3_score_atlas_n600`, d_seg 0.034 — 119× larger,
  so shape only): top-decile pair concentration **1.80×**, max/min 6.93×, **no pair with zero debt**.

**Verdict: REFUTED as a priority.** Uniform sampling forfeits at most ~1.8× against perfect
debt-proportional pair selection. Against 300× (class) and 5,736× (band), **the pair axis is the one
already close to right — anyone reaching for hard-pair mining here is optimizing the flattest axis.**

*verdict_scope: **INSTANCE**, and weaker than the others — the 1.80× concentration is measured on the
`ddm_g3_score_atlas_n600` vehicle, whose d_seg is **119× larger** than this one's. It is a SHAPE
proxy, not this vehicle's number, because per-pair debt does not exist here (the eval sums to a
scalar). A concentrated per-pair debt on THIS vehicle would overturn it; producing that number needs
one GPU pass and is the cheap way to close this row properly.*

Secondary measured defect, worth one line: at `--batch-size 2` the per-step debt CV is 0.236 and
p95/p05 = 2.08; at bs=16 the CV falls to 0.082. AdamW's second-moment normalization then strips much
of the remaining magnitude signal, so a high-debt step does not produce a proportionally larger
update.

---

## §8 The wall-clock budget — the instrument costs more than the optimisation (NEW)

Three completed runs, all `--steps 600 --lr 2.0e-5`, from `resource_safe_run_status.json::elapsed_s`.
`--eval-every` and `--checkpoint-every` move together in all three.

| arm | cadence | events | elapsed |
|---|---:|---:|---:|
| `A2_repeat` | 100 | 6 | 408.3 s |
| `CE0` | 25 | 24 | 865.5 s |
| `EF0` | 25 | 24 | 842.0 s |

Least squares: **elapsed = 259.8 s + 24.75 s × n_events**.

| arm | eval+ckpt cost | share of wall-clock |
|---|---:|---:|
| `A2_repeat` | 148.5 s | **36.4%** |
| `CE0` | 593.9 s | **68.6%** |
| `EF0` | 593.9 s | **70.5%** |

⚠ **Confound, priced not asserted.** Eval and checkpoint cadence are tied, so 24.75 s is their JOINT
cost. The checkpoint is **1,702,875 B**; at a pessimistic 50 MB/s that is 0.034 s — **0.14%** of the
joint cost. DERIVED: the 24.75 s is essentially all *evaluation* (a 600-pair SegNet forward through
the full exact path). The split is bounded, not measured.

**This corrects a constant registered today.** The wall-clock law `total = F + r·n` with
`F = 144.3 s, r = 0.4395 s/step` (`wallclock_fixed_cost_prefix_bias_v1`, `fba57694f5`) predicts
`144.3 + 0.4395·600 = 408.0 s` — which equals `A2_repeat`'s **entire** elapsed time. The law was fit
on `eval_every=100` runs and **silently absorbed their 6 evals into F and r**; it carries no
eval-count term. Applied at `eval_every=25` it underprices by **865.5 / 408.0 = 2.12×**. A fresh
instance of [[cross-regime-constant-transfer-genus-finishing-stage]]: the denominator (eval cadence)
moved and the constant did not. `ce1` §7's "≈12 min per arm" rests on this law plus a `~14.8 s/save`
term that `ddm_aa3` (`28c766be16`) already struck as back-solved-then-reused; actual `CE0` elapsed
was **14.4 min** — close by luck, wrong by mechanism. **The cost is the EVAL, not the SAVE.**

**Corrected form for planning:** `total ≈ (F + r·n) + 24.75·⌈n / eval_every⌉`, with `F + 600r`
measured at **259.8 s**. F and r are not separately identifiable from these points; say so rather
than re-latching them.

---

## §9 The DSL budget — the live vehicle's curriculum levers are held by nothing (NEW)

The triality law says the DSL HOLDS every designed lever. Measured against the live trainer:

- `train_semantic_quantized_resumable.py` exposes **38** `add_argument` flags.
- **8 of 38 (21.1%)** appear anywhere under `src/tac/witness_dsl/`: `--cache --device
  --distill-weight --eval-every --lr --out --resume-from --seed`. Every one is a **name collision**
  with the identically-spelled flag on the levelset witness trainer.
- **No `witness_dsl` module declares `TRAINER_RELPATH` binding to this trainer.** The only module
  mentioning `pr130_lift` at all (`hr1_prestage.py`) references the *lifted oracle* and
  `train_semantic_full.py` as source paths, not this trainer. So `lever_registry.completeness()`
  reconciles against the levelset entry point and is **structurally blind to the live vehicle**.
- **Zero** curriculum/loss-shape levers are held: `--ce-fraction`, `--softplus-fraction`,
  `--band-objective-weight`, `--float-warmup-steps`, `--distill-max-seg`,
  `--ema-target-seed-fraction`, `--weight-qat-q3q4`, `--bits`, `--steps`.

**This is the mechanism behind `ce1`'s "nobody has ever varied it."** Three decades of learning rate
were swept because `--lr` is an obvious scalar knob; the curriculum shape was swept **zero** times
across nine runs because nothing in the apparatus surfaced it. `--ce-fraction` had no validator, no
registry row, no duty-to-measure — and it turned out to be worth 13.6× on the seg axis (§3). This is
[[default_off_is_orphaned_signal_activation_ledger_reconciliation_20260706]] at whole-vehicle scale:
**the DSL holds the levers of a vehicle that is not the one shipping.**

Re-weighting axes that exist vs fire:

| axis | status |
|---|---|
| spatial band × class-pair edge (`--band-objective-weight`) | EXISTS, **default 0.0** (`:930`); fired once as `band_a1`, verdict `verdict_scope: none` — **unjudged** |
| per-class loss weight (`weight=` to CE) | **ABSENT** — no `weight=` kwarg at either CE callsite |
| per-frame-pair loss weight / sampling probability | **ABSENT** — uniform |
| hard-example mining / OHEM / focal / importance sampling | **ABSENT** |
| margin saturation (implicit per-pixel reweight) | **USED — and in EF0 it is the only active term** |

---

## §10 THE GAP TABLE

| # | budget | allocated by | allocation share | value share | misallocation | verdict |
|---|---|---|---|---|---|---|
| 1 | curriculum phase (movement) | fraction-of-run | ce **92.19%** | ce marginal value **negative** | sign-inverted | **CONFIRMED — 13.6× recovered (§3)** |
| 2 | curriculum phase (LR) | fraction-of-run | ce 81.20% | — | proxy of a proxy | CONFIRMED |
| 3 | gradient mass, spatial | pixel `.mean()` → area | band 2.163% | band debt **99.22%** | **5,736× density / 45.9× mass** | CONFIRMED **for CE only** (§5 caveat) |
| 4 | gradient mass, per class | pixel `.mean()` → area | Lane 0.586% | Lane debt 22.69% | **38.74×** (spread 299.9×) | CONFIRMED **for CE only** (§5 caveat) |
| 5 | wall-clock | eval cadence flag | eval **70.5%** @ every-25 | eval buys nothing while argmin=0 | large | **CONFIRMED — free to fix** |
| 6 | model capacity | parameter count | top-10 83.21% | top-10 90.18% movement | **1.08** | **REFUTED — well allocated** |
| 7 | per-pair sampling | uniform | 1/600 each | top-decile 1.80× (proxy vehicle) | **~1.8×** | **REFUTED — flattest axis** |
| 8 | archive bytes (lossless coding) | entropy floor | tokens 61.3% | coder within **+0.42 B** of cross-entropy | ~0 | **REFUTED — at floor (§11.4)** |
| 9 | archive bytes (content) | pose constraint | tokens 61.3% | rc4 drop pays on rate+seg, blocked by pose | **20–400× (pose:seg)** | **CONFIRMED (§12)** |
| 10 | campaign attention across axes | share of S | pose ranked last (5.20%) | pose gates 92% of bytes | inverted | **CONFIRMED (§12)** |
| 11 | DSL lever coverage | vehicle lineage | 0 curriculum levers held | — | structural | **CONFIRMED (§9)** |

---

## §11 The byte budget — four of seven regions are AT FLOOR, and pose prices all of them

The `ra2` census was **re-derived from the archive bytes and is exactly correct** — all seven numbers,
sum closing to zero unaccounted bytes. Archive located at
`/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/cpu_decode/best_rx2/lifted_submission_cpu/archive.zip`,
sha256 `80d9c8c6…` ✓, 182,759 B ✓, one `ZIP_STORED` member `p`, parsed with the shipped receiver's own
`RX1_MODEL_HEADER` constants (`runtime/residual_archive.py`).

| section | bytes | % of archive | S-cost `25b/N` | % of S |
|---|---:|---:|---:|---:|
| RC64 token stream | 112,110 | **61.343%** | 0.0746494 | 46.77% |
| WANS1 semantic (renderer weights) | 34,763 | 19.021% | 0.0231473 | 14.50% |
| CAP1 carrier (+selector +compensation) | 22,161 | 12.126% | 0.0147561 | 9.25% |
| IHS1 HPAC probability model | 13,515 | 7.395% | 0.0089991 | 5.64% |
| ZIP framing | 100 | 0.055% | 0.0000666 | 0.04% |
| RCF1 residual table | 96 | 0.053% | 0.0000639 | 0.04% |
| RX1M header | 14 | 0.008% | 0.0000093 | 0.01% |

Denominators: 182,759 B and `S = 0.159597293`. Sub-census not in `ra2`: semantic = 66,339 params /
38 tensors = **4.192 bits/param**; tokens = 117,964,800 symbols = **0.00760 bits/symbol**.

### §11.1 ⚠ MY CHARTER'S PREMISE WAS WRONG — the source wins

I chartered the byte leg to look for the *absence* of a token rate-distortion curve. **A measured one
exists**: `.omx/research/ddm_rc4_rung4_token_drop_verdict_20260816.md`, dated one day *before* `ra2`.
`ddm_mp2`, `ddm_wd2`, `ddm_ra2c` carry the same for the semantic and carrier sections. I recorded a
negative-existence expectation without exhausting the search — the exact class named in
[[m53]]. Corrected here rather than quietly dropped.

### §11.2 The exchange-rate table (break-even = 1.00 byte-equivalents)

DERIVED from the MEASURED endpoints of each rung: `ΔS_distortion / (25/N) / bytes_reclaimed`.
Averages over the rung, not marginals.

| lever (section) | B reclaimed | ΔS distortion | byte-equiv | source |
|---|---:|---:|---:|---|
| **tokens — rc4 drop, SEG LEG ONLY** | 17,985 | +0.008733 | **0.73 ✅ PAYS** | `ddm_rc4…:232-236` (adv n=120) |
| RCF1 table — remove | 96 | rate only | 3.50 ❌ | EXACT, `FINAL_RESULT.json` |
| HPAC — remove the prior | 13,515 | rate only | 3.81 ❌ | EXACT, `ddm_dc1…:213` |
| semantic — wd2 ep60 width student (seg leg) | 16,000 | +0.088100 | 8.27 ❌ | adv n600 |
| **tokens — rc4 drop, seg + pose as measured** | 17,985 | +0.183052 | **15.29 ❌** | `ddm_rc4…:336-342` |
| semantic — mp2 mixed q3/q4 | 823 | +0.047224 | 86.2 ❌ | adv n600 |
| carrier — ra2c rank-4 | 14,662 | +1.843151 | 188.8 ❌ | adv n600 |
| carrier — ra2c α=0 (delete all) | 22,161 | +22.694322 | 1,538 ❌ | adv n600 |

⚠ **Instrument caveat, load-bearing.** Every advisory row reads the same 182,759-B archive as
`d_seg 0.00042714 / d_pose 0.00014747`. Against the contest-CUDA authority
(`0.00029611 / 6.880e-06`) the advisory instrument reads seg **1.44×** high and pose **21.4×** high.
Ratios plausibly transfer; **absolutes do not.** Only the `FINAL_RESULT.json` and `dc1`/`hm1` rate
rows are EXACT.

### §11.3 The structural split, measured

The decoder splits by frame parity (`cpr1/inflate.py:289-346`): master frames (frame_1) come from
`semantic(tokens)`; slave frames (frame_0) come from the carrier basis. SegNet reads only the last
frame, so **d_seg is exclusively a function of {tokens, semantic, hpac, table}**. PoseNet reads both.
Confirmed empirically, not just structurally: deleting the entire carrier (α=0) left
`d_seg 0.00042714 → 0.00042714 exactly unchanged` while `d_pose` went **350,428×**
(`ddm_ra2c_alpha0_verdict…:15-16`).

### §11.4 Sections already AT or NEAR floor — the byte-side honest negatives

1. **ZIP framing, 100 B — provably at the structural floor.** A minimal one-member `ZIP_STORED`
   archive named `p` with an empty payload is exactly 100 B (30 LFH + 1 name + 46 CDH + 1 name +
   22 EOCD), constructed and measured. Zero recoverable.
2. **RCF1 residual table, 96 B — net-negative cost.** It *returns* 336 token bytes (112,446 →
   112,110), a 3.50× payback, selected from 25 fitted candidates. Nine alternative contexts swept;
   only the shipped rung has slope < −1.
3. **IHS1 HPAC, 13,515 B — bracketed at its local optimum.** Returns 3.81 B per counted byte. The
   local derivative brackets the shipped point: −420 model B → +484 token B (slope −1.15, still
   pays); +145 B → −2.2 token B (slope −0.016, stops paying).
4. **The RC64 token *coder* — at its information floor.** 112,110 B shipped vs 112,109.578 B model
   cross-entropy = **+0.42 B**. The best static-context ORACLE (8-spatial + 9-prev, 65,536-cell table
   assumed free) bottoms at 144,167 B — **+32,057 B worse**. The coder is closed; only the *content*
   is open.
5. **CAP1 Rice coder — measured better than the alternative.** Adaptive arithmetic loses by 415 B.

**The whole remaining lossless axis is ≤ ~278 B = 1.93% of the 14,413 B gap** — `ra2`'s claim,
CONFIRMED.

---

## §12 THE SYNTHESIS — the campaign's own attention is allocated by contribution, not shadow price

Put §1's master table next to §11.2 and one thing jumps out.

| axis | share of S | shadow price on the largest available byte move |
|---|---:|---|
| rate | 76.25% | — (it *is* the move) |
| seg | 18.55% | +0.008733 S on the rc4 token drop → **0.73 byte-equiv, PAYS** |
| pose | **5.20%** | **+0.174319 S on the same move → 15.29 byte-equiv, BLOCKS** |

**The rc4 token drop reclaims 17,985 B — 125% of the entire 14,413 B gap — and its rate + seg legs
are a measured −3.243e-3 S win, 33.8% of the gap.** It dies on an *uncompensated pose leg* worth 20×
its seg cost. On the semantic section the ratio is ~400× (`mp2` keep75: seg +0.000107, pose
+0.04157).

So: **pose is 5.20% of the score and the binding constraint on 92% of the bytes.** It owns no d_seg
at all (§11.3) and cannot close the gap alone (§1.1) — and it *gates the single largest measured move
in the archive*.

This is the audit's own thesis turned on the campaign. Every budget inside the trainer is allocated
by a proxy — fraction-of-run, pixel-area, parameter-count. The campaign's *attention* budget is
allocated by the same kind of proxy: **share of S**. Ranking axes by contribution puts pose last. The
correct ranking for routing is by **shadow price**, and by that measure pose is first. My own §1.1
arithmetic ("pose cannot close the gap alone") is correct and would license exactly the wrong
inference if read as "therefore deprioritise pose."

**The highest-value unmeasured cell in the whole system is not on any axis. It is the coupling:
whether a carrier/compensation edit can absorb the +0.1743 S pose leg of a byte move that is already
a measured win on rate + seg.**

⚠ Scope: the pose leg is advisory (n=48 for the rc4 pose point) on an instrument reading pose 21.4×
high. The *ratio* is what carries; the absolute must be re-measured on the CUDA authority before any
promotion claim.

---

## §13 RANKED CURES

Ranked by |ΔS| reachable per unit cost, cheapest-credible-first per the MVP-first law. Every row
names its falsifier.

| # | cure | cost | ΔS reachable | falsifier |
|---|---|---|---|---|
| 1 | **`--steps 1200` on the EF0 config** — one flag; the ONLY arm still descending (−1,383 flips per 100 steps at its horizon) | ~28 min local, $0 | UNKNOWN, bounded below by 0.000539; first arm with a real chance of `argmin ≠ 0` | argmin stays at step 0 → the ceiling is init and the seg-descent family closes at this formulation |
| 2 | **The pose-compensated token drop (§12)** — can a carrier/compensation edit absorb the +0.1743 S pose leg of `rc4`'s rung-4 drop, whose rate + seg legs are already a measured **−3.243e-3 S win = 33.8% of the gap**? | build + GPU | **the largest measured prize in the system**: 17,985 B = 125% of the gap | the pose leg is irreducible under any compensation → the token content axis closes and the archive is rate-bound |
| 3 | **`--eval-every 100`** on any arm not doing best-state selection | 0 (flag) | ΔS 0; **2.1× throughput** = 2.1× arms per hour (§8) | a run where finer `best_state` granularity actually selects a non-zero step |
| 4 | **Adopt EF0's curriculum as the default** for any future seg fine-tune | 0 (flag) | +0.006798 vs stock A2 — but this is **repair of self-inflicted damage, not gain over init** | a from-scratch (not fine-tune) run where CE genuinely establishes structure |
| 5 | Measure the **residual spatial/class gap after margin saturation** (§5 caveat) | 1 GPU forward | decides a built, unjudged, default-off lever (`--band-objective-weight`) | residual ≈ 0 → the band table is dead weight, superseded by the free flag |
| 6 | Direct **margin histogram at init** to confirm or kill §4's derived `ρ(m) ∝ m^(−0.663)` | 1 GPU forward | routes the whole seg axis: penalty vs projection vs search | density bounded at 0 → the knife-edge reading is wrong and ordinary descent should work |
| 7 | Bind the live vehicle into the DSL / lever registry (§9) | small build | ΔS 0 directly; stops the next 13.6× lever hiding for nine runs | — |

**Explicitly NOT recommended:** hard-pair mining (§7 — flattest axis, ~1.8×); a per-tensor capacity
router (§6 — ratio 1.08); further *lossless* byte work (§11.4 — four regions at floor, ≤278 B left =
1.93% of the gap); and **deprioritising pose because it is 5.20% of S** (§12 — it is the gate).

---

## §14 HONEST NEGATIVES

1. **The capacity budget is fine (§6).** Ratio 1.08. Do not build a per-tensor router on this
   evidence.
2. **The per-pair sampling budget is fine (§7).** ~1.8× against 300× and 5,736× elsewhere.
3. **The LR budget is not wrong — it is just not the right denominator (§2.2).** `ce1`'s 81.20% is
   exactly correct as an LR share; it simply is not the quantity that moves weights.
4. **Pose cannot close the gap alone (§1.1)** — not because pose work is bad, but because the whole
   term is 0.008295 against a 0.0095973 gap.
5. **The spatial/class misallocations may already be mostly cured for free (§5 caveat).** They are
   exact for CE and unmeasured under margin saturation. Quoting 45.9× or 5,736× as a live defect of
   the *EF0* configuration would be wrong.
6. **My own draft said "CEILING-ZERO" and was wrong.** I wrote it from EF0 at step 450; EF0 finished
   at +636 with its steepest descent. Recorded here rather than silently edited —
   [[corrections_land_in_bodies_headlines_keep_the_stale_number_20260805]].
7. **The trainer has no train/eval render mismatch.** Training and eval both use
   `render_quantized(..., exact_path=True)` (`:1219-1221`, `:797`) — the full 384→874→uint8→384
   camera round-trip is in the gradient. I checked expecting a defect and did not find one. (The
   *lifted oracle* trains at `exact_path=False`; the live trainer does not inherit that.)
8. **Four of seven archive regions are at or near floor (§11.4)**, one of them (ZIP framing, 100 B)
   provably exactly at it, and the token *coder* is within **+0.42 B** of its own cross-entropy. The
   byte budget is largely not the problem; the token *content* is.
9. **I asserted a negative-existence claim without exhausting the search (§11.1)** — I chartered a
   leg to confirm that no token RD curve existed, and one had existed since the day before. Logged as
   my own defect, not the leg's.

---

## §15 Provenance — the payloads this unit READ (nothing was written to them)

This unit materialized **no payload**: it is pure re-analysis of already-retained bytes plus DERIVED
arithmetic. `CE0` and `EF0` are MAIN's fires, retained by MAIN's launcher; I read them read-only
after they completed (`exit 0`). Hashes pinned so the next consumer can prove byte-identity:

| path | bytes | sha256 (first 16) |
|---|---:|---|
| `/Volumes/APDataStore/pact/ddm_ce1/CE0/run.log` | 46,580 | `2bf2b61861a9ac6e…` |
| `/Volumes/APDataStore/pact/ddm_ce1/EF0/run.log` | 46,058 | `b1287a2652a26c4f…` |
| `/Volumes/APDataStore/pact/ddm_jr1/A2_repeat/run.log` | 26,040 | `8b52c13bd58d31f6…` |
| `/Volumes/APDataStore/pact/ddm_jr1/JR1_VECTORS.npz` | 12,424,461 | `c701f5290d53757b…` |
| `src/tac/pr130_lift/band_weight_table_rt1_n600.json` | 2,989 | `72658f02012c640c…` |
| frontier `archive.zip` (`hv1 ep0634`) | 182,759 | `80d9c8c6fdc72caa…` |

Source pins re-hashed this unit: trainer `b486f416…` · lifted oracle `ffdf0988…` · band objective
`81e187f6…`.

---

## NEXT_IF_RESUMED

1. **Fire cure #1** — `EF0` config at `--steps 1200 --eval-every 100` (cure #3 folded in: saves ~9
   min and costs nothing while argmin = 0). Pre-registered falsifier: **argmin ≠ step 0** would be
   the first descent this vehicle has ever produced; argmin = 0 closes the seg-descent family at this
   formulation. $0, local Metal, ~28 min. MAIN owns the fire.
2. **Charter cure #2 — the pose-compensated token drop.** This is the largest measured prize in the
   system (§12) and the one cell nobody has entered. Start from
   `.omx/research/ddm_rc4_rung4_token_drop_verdict_20260816.md` §"rate+seg" and
   `ddm_ra2c`'s carrier ladder. Re-measure the pose leg on the CUDA authority before any promotion
   claim — the advisory instrument reads pose **21.4×** high.
3. **Do not extrapolate EF0's tail** (`L3000`/`aa3` lesson). The −1,383/100-step slope is a reason to
   run the arm, never a substitute for it.
4. **Do not re-open `--band-objective-weight` before cure #5.** §5 says its 45.9× is a CE-specific
   defect and CE no longer fires in the best arm; the paid cure may be wholly superseded by the free
   flag.
5. **Carry the corrected wall-clock form** `(F + r·n) + 24.75·⌈n/eval_every⌉` into every future
   budget line; do not re-latch F and r from two points.
6. **Fix CLAUDE.md's class-3 area** (1.56% → **1.2379%**, n600-measured) — a live n96 prefix-bias
   constant sitting in the always-loaded instructions.
7. **Open question, unresolved:** whether the seg program's init and the shipped `hv1 ep0634` renderer
   are the same object (§1.1). Cheap to settle; nothing downstream should assume it.
