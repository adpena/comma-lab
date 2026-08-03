# ddm_rt2 — "the realized `d_seg` is computed and discarded" is FALSE; the surviving defect is a different one, and it has a vehicle number

**Arm:** `ddm_rt2`. **Date:** 2026-08-03. **Adjudicates:** `ddm_rz1` (`5c52c6b58e`) §3.1 "A1. Stop
discarding the realized margin" (its rank-1 item), which relays `ddm_er1`
(`.omx/research/ddm_er1_realized_trip_in_the_describe_objective_20260802.md`) §0, and the standing
charge in task **#888**.

All measurements below are **n600** and **scorer-free** (cached `lstars` / `margins`, plus a pure-MLX
autodiff probe). **No scorer slot was taken** — `ddm_pu2` holds it.

---

## 0. HEADLINE

| claim (as relayed) | verdict | evidence |
|---|---|---|
| The describe path realizes the full trip (uint8 STE → R → real SegNet/PoseNet) | **TRUE** | `:2347`, `:2400–2405`, `:2407`, `:2415–2417` |
| A realized `d_seg = mean(argmax != targets)` is computed at `:2419` | **TRUE, exact line** | `:2419` verbatim |
| `:2359` trains CE on the label | **FALSE — misattributed line.** `:2359` is `) -> Any:`, the close of the `_loss` signature. CE is at **`:2410`** | `sed -n 2359p` |
| "four lines later" | **FALSE** — 2359→2419 is **60** lines | arithmetic |
| The realized `d_seg` is **discarded** | **FALSE at every layer except the gradient.** It is the step-acceptance criterion (`:528`) and, in the full run, the promotion gate via an **n600 CPU-authority realized verdict on the compiled archive** | `:2489`, `:2501`, `:528`, `:1877–1894` |
| Cure = "stop discarding it", one line, rank 1 | **REFUTED BY EXECUTION** — the `:2419` expression has an **identically zero gradient** | probe: 0/640 nonzero vs control 640/640 |
| #888: "no `eval_roundtrip` equivalent" | **REFUTED** (er1/rz1 are right here) | as row 1 |
| #888: "optimizes LABEL not MARGIN" | **UPHELD IN EFFECT, now with a vehicle number** — a margin term exists and is default-ON, but contributes **7.06e-06** | §4, n600 |

**One sentence.** `rz1`/`er1` correctly found that the describe path realizes the trip — which does
refute #888's first clause — but "computed and **discarded**" is wrong (the realized quantity is the
*gate*; the surrogate only *proposes*), the cited line `:2359` is misattributed, and the proposed
one-line cure is impossible because the quantity is non-differentiable. **The real defect is
adjacent and cheaper to name:** the seg gradient is effectively label-CE because the margin term that
*is* already wired is numerically inert at its shipped floor.

---

## 1. LOCATION — the file `rz1` did not name

`rz1` and `er1` both cite bare line numbers. The module is
**`src/tac/optimization/direct_description_joint_descent.py`** (2,870 lines), driven by
`tools/launch_ddm_joint_descent.py`.

`git log` shows the module is **unchanged since `1b8ea1493c` (2026-07-25)**, so `er1` (08-02),
`rz1` (08-03) and I read **the same bytes**. The line errors below are misattribution, not staleness.

### 1.1 `er1`'s line map is 5-of-7 wrong (denominator = 7 cited lines)

| `er1` cites | for | actually at 2359 etc. | correct line |
|---|---|---|---|
| `:2347` | uint8 STE round | `return clipped + mx.stop_gradient(mx.round(clipped) - clipped)` | **✓ exact** |
| `:2345–2350` | `fused_r_roundtrip` | `:2345` = `mx.tensordot`; `:2350` = `self,` | ✗ → **`:2400–2405`** |
| `:2352` | SegNet | `*,` | ✗ → **`:2407`** |
| `:2360–2363` | PoseNet | `mx = self.mx` … `pair_ids=pair_ids,` | ✗ → **`:2415–2417`** |
| `:2359` | `seg = ce_seg_loss_mlx(...)` | `) -> Any:` | ✗ → **`:2410`** |
| `:2349` | `_loss` unpacks `seg, pose_mse, _` | `def _loss(` — unpack is at `:2361` | ~ (function start, substantively right) |
| `:2419` | realized `d_seg` | `d_seg = mx.mean(mx.not_equal(mx.argmax(seg_logits, axis=-1), targets).astype(mx.float32))` | **✓ exact** |

Every wrong citation lands inside the `_loss` signature block (`:2349–2363`) — a consistent ≈−50-line
offset applied to the `_components` body. **The misattribution does not change the mechanism**: every
described operation genuinely exists in `_components`. I record it because a bare line number is how
a claim gets relayed without being re-opened, and this is the second such case today.

---

## 2. IS IT REAL? — the trip is realized, and the targets are the true oracle

`_components` (`:2373–2420`) does, in order: paint → `clip` → **uint8 STE round** (`:2347`) → **`fused_r_roundtrip(camera_hw=(874,1164), output_hw=(384,512), ste_round=True)`** (`:2400`) → real MLX
SegNet on the round-tripped frame_1 (`:2407`) → real MLX PoseNet on YUV6 from *the same* round-tripped
pair (`:2415–2417`).

**The targets are the canonical oracle**, not a proxy: the launcher loads
`labels = open_stored_npy_memmap(cache_path, "lstars")` at all four sites (`:475`, `:1198`, `:1540`,
`:2598`), and `:2409` indexes it. So `:2419` is the **true contest quantity** — mean argmax
disagreement against the GT SegNet argmax, on the round-tripped scorer plane.

**`er1`'s first clause is correct and #888's "no `eval_roundtrip` equivalent" is refuted** for this
path. That part of `rz1` survives intact and is the memo's real contribution.

---

## 3. WHAT IS IT DISCARDED *IN FAVOUR OF*? — surrogate-propose, realized-dispose

This is the question that decides the finding, and the answer refutes the word "discarded".

`_components` returns `(seg, pose_mse, d_seg)`. There are **exactly two** callers:

- **`_loss` (`:2361`)** — `seg, pose_mse, _ = self._components(...)`. This is the **gradient** path. It
  drops `d_seg`. **This is the only place the value is dropped.**
- **`measure_components` (`:2489`)** — `seg, pose, d_seg = self._components(...)`, surfaced at `:2501`
  as `{"seg_ce_margin", "d_seg", "d_pose", "joint_objective_no_rate"}`.

`measure_components` has **8 callsites** (7 in `launch_ddm_joint_descent.py` at `:497 :522 :1697 :1799
:2728`, 1 in `tools/smoke_ddm_fd1_gn_engine.py:111`), and all three launcher paths are dispatched
(`:673`, `:2884`, `:2891` — **not dead code**). What they do with it:

1. **Bounded-smoke step acceptance (`:528`)** — the realized `d_seg` *is* the acceptance test and
   selects the learning rate:
   ```python
   if metrics["d_seg"] < initial_metrics["d_seg"] or metrics["d_pose"] < initial_metrics["d_pose"]:
       accepted = candidate, metrics, learning_rate
   ```
   with `INSTANCE_BLOCKER_BOUNDED_PAIR_NO_DSEG_OR_DPOSE_DESCENT_FOR_PREREGISTERED_STEP_GRID` raised if
   nothing improves it, and `d_seg_decreased` written to the receipt (`:546`).
2. **Full-run promotion gate (`:1877–1894`)** — when the *realized* (quantized) description changes,
   the launcher compiles the candidate archive and runs **`_chunked_n600_verdict`** on **frozen CPU
   scorers**, then `classify_realized_stage_verdict(reference_d_seg, candidate_d_seg, target_d_seg,
   …)` plus `pure_priced_realized_delta`. That is a *stronger* realized authority than the MLX
   `:2419` value.
3. The CE proxy ranks candidates **only within a rung** — and says so itself. Docstring of
   `_seg_lexicographic_attempt_key` (`:930`): *"Rank within each rung; **exact n600 receiver replay
   remains the gate**."*

**So the architecture is surrogate-propose / realized-dispose:** CE+hinge supplies the *direction*,
the realized `d_seg` supplies the *accept/reject* and the *promotion gate*. That is a standard and
legitimate pattern (surrogate descent under a true-objective line search), not a lost signal. The
correct sentence is **"the realized `d_seg` is excluded from the gradient"** — not "discarded".

### 3.1 Why the one-line cure is impossible — EXECUTED

The relayed cure is "stop discarding it": change `_` to `d_seg` and put it in the objective. Probe
(pure MLX, verbatim `:2419` expression, synthetic logits, positive control on the *same* tensor):

```
d_seg (:2419 verbatim)      value=0.820312  grad_absmax=0.000e+00  grad_nonzero=0/640
POSITIVE CONTROL mean(x^2)  value=1.000914  grad_absmax=1.219e-02  grad_nonzero=640/640
```

**The gradient is identically zero** (`argmax` → int → `not_equal` → bool → `astype` breaks the chain);
the control proves autodiff was live. Adding `100·d_seg` to `_loss` would produce a **byte-identical
descent** plus wasted compute. The one-line framing is refuted.

`er1` itself does not actually propose that: its §0 sentence conflates *"stop discarding the realized
quantity"* with *"replace the CE surrogate with the exact power-diagram margin"*. The first is
impossible; the second is a real but **larger** build (`realized_margin_and_gradient` in
`power_diagram_witness.py`). `rz1` relayed the first half as the headline and ranked it #1 at "zero
counted bytes"; the charter escalated it to "a named one-line cure". **Each hop dropped a qualifier.**

---

## 4. THE SURVIVING DEFECT — measured, n600, and it undercuts my own first draft

My round-1 self-review target was my own claim that *"#888's 'LABEL not MARGIN' is imprecise because a
margin term already exists."* That claim is **technically true and practically misleading**, and the
measurement says so.

The seg leg is **not** pure CE (`:2410–2414`):
```python
seg = ce_seg_loss_mlx(seg_logits_nchw, targets)
if self.margin_hinge_weight > 0.0:
    seg = seg + self.margin_hinge_weight * margin_floor_hinge_mlx(seg_logits_nchw, targets,
                                                                  margin_floor=self.margin_floor)
```
`margin_floor_hinge_mlx` is a genuine margin objective — `mean(relu(margin_floor − margin))`,
`margin = target_logit − max_competing_logit` — and it is **ON by default**
(`margin_hinge_weight=0.05`, `margin_floor=0.1`, `:2280–2281`).

**But how much of the objective can it possibly touch?** By its own contract a site with
`margin ≥ floor` contributes exactly 0 value and 0 gradient. Measured on the cached n600 GT margins:

```
denominator: n_pairs=600  sites=117,964,800  (600, 384, 512) float32
sites with margin < 0.1 (the live floor): 333,078  = 0.2824% of sites
sites with margin < 0.0                 : 0        = 0.0000%      [control: confirms GT-reference margins]
hinge value mean(relu(0.1 - margin))    = 1.412186e-04
  -> contribution to seg leg at weight 0.05 = 7.060931e-06
margin quantiles p0.01% / p0.1% / p1% / p5% / median
                = 0.0036 / 0.0353 / 0.3552 / 2.0582 / 5.8934
```

**The median margin is 5.8934 — 59× the shipped floor. p5% is 2.06, still 20× the floor.** The margin
term contributes **7.06e-06** to the seg leg. It is not a dead branch (0.2824% of sites do activate
it) but it is **numerically inert as an objective**.

So **#888's second clause is vindicated in effect, with a vehicle number**: the seg gradient *is*
label-CE for all practical purposes, not because no margin term was built, but because the one that
was built is floored 59× below the median margin.

**Why that floor:** `margin_floor=0.1` was chosen as the **L7 cross-hardware numpy-portability guard**
(docstring: "anchor ~0.1 > the measured ~0.096 cross-hardware logit drift") — i.e. it exists to make
the local argmax survive macOS→Linux fp32 drift, **not** to aim the descent at the separatrix. It is
the right floor for its own job and the wrong floor for #888's.

**And it is unreachable from config.** All three constructor sites (`:481`, `:1543`, `:2601`) pass
exactly four kwargs (`lift`, `scorer_adapter`, `seg_targets`, `pose_targets`). `margin_hinge_weight`
and `margin_floor` appear **nowhere** in `DirectDescriptionJointDescentTypedConfigV1`. They are
hardcoded defaults with no DSL owner — the *config-orphan / unladdered-governance-knob* class named in
CLAUDE.md, and an already-built-but-unfired lever under the "off is a tracked queue" discipline.

### 4.1 Scope caveat on §4 (stated against my own result)

The cached `margins` are the **GT-reference** margins (which is why `margin < 0` is exactly 0.0000% —
the control confirms the definition). Under descent a candidate's margins move, and the hinge
activates precisely where they are pushed down. **So 0.2824% / 7.06e-06 bounds hinge activity at the
descent's starting point, not throughout.** The mid-descent value is unmeasured and needs the scorer.

### 4.2 What `er1`'s supporting alignment evidence actually says

`rz1` ranks A1 #1 partly on `er1` §2.3. Read at strength, `er1` **argues against its own thesis** and
should be quoted that way: Spearman(per-site CE, realized margin) = **−0.9452**, and *"the honest claim
is **'CE is a good ranker but a poor allocator, and is scale-sensitive where the score is not'** — not
'CE is misaligned.'"* Its allocation number (bottom-5%-margin sites carry 10.91% of CE mass, 2.2×
enrichment) is on **isotropic-Gaussian synthetic features**, and `er1` caps it explicitly: *"chart
geometry, not vehicle numbers… The real alignment must be measured on real `z` and is staged."*

**So no vehicle-level alignment measurement supports A1's rank-1 placement.** My §4 is, as far as I
found, the first vehicle number on this question — and it supports the *direction* of A1 (the seg
gradient is effectively label-CE) while refuting its *mechanism* (nothing is discarded) and its
*cost* (not one line).

---

## 5. THE OWED MEASUREMENT — and its pre-registered falsifier

The real A/B is **which quantity supplies the gradient**, and it needs a scorer slot (`pu2` holds it).
Two candidate cures, cheapest first:

- **C1 (cheap, already built):** raise `margin_floor` toward the measured p5% (**2.0582**) and/or lift
  `margin_hinge_weight` above 0.05, exposing both through the typed config. No new surface — this pays
  the debt on the existing lever, per `built_new_machinery_instead_of_paying_identified_debt`.
- **C2 (larger):** swap the CE leg for `realized_margin_and_gradient` (er1/#539), honoring its derived
  consumer constraint that any floor sit well above ~1e-7 quotient units.

**Pre-registered falsifier (either cure).** Matched pair at identical description budget and seed,
n600 through the real byte-close, reporting **d_seg *and* d_pose *and* bytes** against the live
baseline — **`cx1`: S = 0.8264972, 353,808 B, seg leg 0.4311790** (gap to the PR130 bar 0.172141 =
**0.6543562**; `W = 1.273108215332031` B/flip; `cx1` total flips 508,639):
- **KILL** if Δd_seg ≤ the single-seed noise floor;
- **KILL** if proxy margin improves but realized d_seg does not;
- **KILL C1 specifically** if raising the floor degrades the L7 cross-hardware portability guard it was
  sized for (`~0.096` drift) — that guard is the floor's original job and must not be traded away
  silently.

A seg-only A/B is forbidden here: `uv1` measured a 3,019× d_pose separation between two bases under an
otherwise identical solver.

---

## 6. VERDICT SCOPES

- `verdict_scope: INSTANCE — src/tac/optimization/direct_description_joint_descent.py + tools/launch_ddm_joint_descent.py`:
  "the realized `d_seg` is computed and discarded" is **REFUTED**. It is excluded from the gradient
  only; it is the step-acceptance criterion and the n600 CPU-authority promotion gate.
- `verdict_scope: INSTANCE — the :2419 expression`: "the cure is one line" is **REFUTED BY EXECUTION**
  (gradient identically zero, 0/640, against a live 640/640 control).
- `verdict_scope: FORMULATION — the joint-descent describe path`: #888's *"no `eval_roundtrip`
  equivalent"* clause is **REFUTED**; the trip is realized in-loop. I did **not** examine the other
  describe surfaces (`direct_description_minimizer.py`, `direct_description_coupled_margin.py`), so I
  make no claim about them.
- `verdict_scope: INSTANCE — the GT-reference operating point, n600`: #888's *"optimizes LABEL not
  MARGIN"* clause is **UPHELD IN EFFECT** — margin-term contribution **7.06e-06**, only **0.2824%** of
  117,964,800 sites able to activate it. Mid-descent behaviour is **unmeasured**.

**Net for #888:** do not close it, and do not re-word it as `rz1` proposes. Its first clause is wrong;
its second clause is right for a reason nobody had written down — an already-built margin lever
floored 59× too high and orphaned from the config.

---

## 7. WHAT I DID NOT DO

- No scorer slot taken; no n600 scorer pass; no training run. The gradient A/B in §5 is **owed**.
- I did not measure the CE term's magnitude in a live run (I searched `.omx/research/` and the `ddm`
  run dirs for a `seg_ce_margin` receipt and **did not find one in that scope** — a negative-existence
  statement scoped to where I looked, not a claim that none exists). So §4 reports the hinge's
  **absolute** contribution and the **activatable-site fraction**, which are baseline-free, rather
  than a CE ratio I would have had to assume.
- I did not re-derive the real-vehicle error concentration; `pt1` (4.4% of pixels carrying 52% of
  errors, 24.09×) and `ddm_pc2`'s flip-graph decomposition already own that question.

## 8. PROBES (reproducible, scorer-free)

Both probes are single-file and rerunnable; each carries a control that would have exposed a dead
instrument:
1. **Gradient probe** — `mx.value_and_grad` on the verbatim `:2419` expression vs `mean(x²)` on the
   same tensor. Result §3.1.
2. **Hinge-activity probe** — cached `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` `margins`,
   chunked 50 pairs at a time, reporting the full denominator (117,964,800 sites). Result §4. Control:
   `margin < 0` returns exactly 0, confirming the array is the GT-reference (target = `lstars`) margin.
