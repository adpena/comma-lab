---
schema: ddm_pu1_pose_underpricing.v1
date_utc: 2026-08-03
arm: ddm_pu1 (re-price the pose backlog at the live marginal; price the tail solve)
lane_id: "lane_ddm_pu1_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false   # exact contest pointer 0.1910828242 [contest-CPU] UNMOVED. This arm ran no gate.
verdict_scope: FORMULATION   # nothing here kills a family; two rows are re-scoped, none retired
axis: "[macOS-CPU advisory - recomputed from components + an instrument validated to 1.2e-5 against
  the evaluator's own n600 d_pose] NON-PROMOTABLE. No new scorer run, no training, no paid dispatch,
  no pointer mutation. Every constant recomputed, none re-typed."
consumes:
  - upstream/modules.py (PoseNet.preprocess_input, compute_distortion; SegNet.preprocess_input)
  - upstream/evaluate.py (rate = archive.zip / uncompressed_size)
  - .omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json  (n600 per-pair d_pose at the cx1 base)
  - .omx/research/ddm_pz1_pose_axis_cx1_base_20260803.md  (instrument validation, 1.2e-5 rel)
  - .omx/research/ddm_mq1_pose_menu_rd_audit_20260801.md  (the re-priced row)
  - .omx/research/ddm_pw1_pose_menu_saturation_20260801.md
  - .omx/research/ddm_pj2_pose_scale_degeneracy_20260802.md
  - .omx/research/ddm_qd2_rebaseline_against_cx1_20260803.md
  - .omx/state/canonical_task_status.jsonl  (417 rows / 148 tasks)
  - experiments/ddm_pfs1_ep_warp_pose_solve.py (WarpPoseOracle - the pose actuator's actual shape)
consumers: [MAIN]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_pu1 — the pose backlog re-priced, and the tail measured

## §0 ANSWER FIRST

**My charter's premise is half wrong, and the half that is right is worth far more than it claimed.**

1. **The "1.73× under-priced" correction has NO fixed sign.** It is `1.7310617` either way, but the
   direction depends on a modelling choice nobody had stated: whether a banked pose lever removes an
   **absolute** amount of `d_pose` or a **fraction** of it. Additive levers are **1.7311× RICHER** at
   `cx1`. Fractional levers are **0.5777× — i.e. 1.7311× POORER**. Exactly inverse, same constant.
   A blanket "re-price everything up by 1.73×" would have inflated every fractional row by 3.0×.

2. **The ledger's pose axis is already drained.** 417 rows / 148 distinct tasks / **35 pose-matching**
   / 10 carrying any delta / **3 carrying a real one** — the `pj2` triple (#850/#873/#882), all
   `−0.0675451`, which `qd2` already established is one run's total triple-stamped and already inside
   `cx1`. **Re-priced value: 0.0000000.** The parked pose population is not in the ledger.

3. **It is in `mq1`, and `mq1` is priced at `dS/d(d_pose) = 18.083` — which is `pw1`'s marginal
   exactly — against a gap of `0.7754681`.** Both stale. Its `≥1.82% of gap` search headroom is,
   re-priced at `cx1` and **if still available**, **`−0.0244312 S = 3.734% of the live gap`** — a
   **2.05×** growth (1.7311× marginal × 1.1851× denominator). This is the one genuinely
   under-priced row in the corpus. Availability is **NOT** established — see §5.

4. **THE HEADLINE, and it is bigger than the re-pricing: the pose axis is not a 600-pair problem.
   It is a 4-pair problem, and those 4 pairs are one SCENE.** MEASURED, n600, on the live `cx1` base:

   | | share of all pose mass |
   |---|---:|
   | pair 74 alone | **30.92%** |
   | top 4 (74, 67, 21, 523) | **55.92%** |
   | top 10 | **67.24%** |
   | pair-index block [60,80) — 20 of 600 pairs | **45.60%** (mean 13.68× population) |
   | block [0,120) | **~69%** |

   The break-even byte budget for solving the top 10 to zero is **10,258 B/pair** (102,580 B total,
   for `−0.0683036 S` = **10.44% of the gap**). The charter's pre-registered falsifier —
   *"> ~600 B/pair ⇒ the cheap-tail route dies"* — is **wrong by 1.4–17× at the live operating
   point**: measured break-even is **818 B/pair at k=200, 2,275 at k=56, 10,258 at k=10, 40,503 at
   k=1**. The falsifier was set at an operating point where pose was 1.73× cheaper.

5. **The tail splits, and the split is the whole finding.** Under `pz1`'s paired perturbation,
   **13 of the top 14 pairs are RESPONSIVE** (|ratio−1| = 5–65%) but **pair 74 — 30.92% of all mass —
   is RIGID (ratio 0.981)**. One pair holds a third of the axis and does not move. That is the
   signature of a **model** wall, not a search wall, and it is the single highest-value measurement
   left on this axis (§5.3, INFERRED not MEASURED — the decisive probe is specified in §8).

6. **PRE-PROBE RUN (§5.5), and it names the target row.** Pair 74's GT pose target is **rank 98/600
   (`z = +0.96`) — ordinary**, which **refutes** the extreme-motion branch of the model-wall reading
   and shifts weight to search-wall. Pairs 523 and 21 *are* target outliers (rank 1 and 3), so **the
   tail has at least two mechanisms.** The right scale is the target set's per-component spread
   `sd_vec = 0.5134`: **6 of 600 pairs have `‖e‖ > sd_vec` — their pose readout carries no usable
   information — and those 6 carry 62.0% of all pose mass.** Bringing just those 6 to the edge of
   informativeness (`‖e‖ → 1.0·sd_vec`) is **`ΔS = −0.0410524` = 6.27% of the gap at a break-even
   budget of 10,276 B/pair** — a bounded, defensible target, not a solve-to-zero ceiling.

7. **Independent corroboration of the `#875` prefix law, from a different artifact and a different
   operating point.** The pose tail is a temporal block, so any prefix is a scene sample. Measured at
   `cx1`: `[0,73)` = **2.76×** population, `[0,181)` = **2.49×**, `[73,181)` = **2.31×**, `[0,100)` =
   **4.03×**, `[100,600)` = **0.39×**. `bp2` measured its own prefix at 5.1× at the `pw1` base; I get
   2.31× for the same index range at `cx1`. **Both >1; the ratio itself moved with the base.** The
   guard must be re-measured per operating point, not carried as a constant.

**Pointer honesty: the exact contest pointer `0.1910828242` is UNMOVED. I fired no gate and produced
no new score. Nothing here is a win; it is a re-price, a measurement, and a specification.**

---

## §1 CONSTANTS — recomputed, never re-typed

MAIN published a wrong denominator twice today, so every number below is recomputed from components
before anything divides by it.

`W = 4·DEN/PX` with `DEN = 37,545,489`, `PX = 600·512·384 = 117,964,800`:
**`W = 1.2731082153320312` B/flip** — delta from the carried value **0.000e+00**. Invariant, and §2
shows *why* it is the only invariant of the three legs.

`cx1`, recomputed from `d_seg = 0.00431179`, `d_pose = 0.00255143`, `B = 353,808`:

| leg | value |
|---|---:|
| `100·d_seg` | 0.4311790 |
| `√(10·d_pose)` | 0.1597320 |
| `25·B/DEN` | 0.2355862 |
| **S** | **0.8264971875** |

**The gap denominator, and which one I used.** Three values are in circulation and they differ by
~3e-7 (0.00005%), entirely from the floor convention:

| convention | gap |
|---|---:|
| MAIN: `S(cx1) − 0.172141` (PR130 published) | **0.6543562** |
| `qd2` | 0.6543559 |
| summing PR130's rounded legs (0.02966+0.015268+0.127214 = 0.172142) | 0.6543552 |

**I use MAIN's `0.6543562`** (PR130's published S, which `na1` verified reproduces from 191,052 B —
190,952 B gives 0.1720751, which does not). The spread is immaterial to every ranking below; it is
stated so the next arm does not have to re-litigate it. **1% of gap = 9,827.2 B**, not the 10,907 B
of the `dc1_fold` era.

Gap decomposition at `cx1`: **seg 0.401519 = 61.36% · pose 0.144464 = 22.08% · rate 0.108372 =
16.56%.**

The marginal, and the ratio that this arm exists to test:

| base | `poseC` | `dS/d(d_pose) = 5/poseC` |
|---|---:|---:|
| `v4c` | 0.3222499 | 15.5159 |
| `v4d` | 0.2929411 | 17.0683 |
| **`pw1`** | **0.2765059** | **18.0828** |
| `mq1` | 0.2639265 | 18.9447 |
| `ms8` | 0.2272963 | 21.9977 |
| `cx1` | **0.1597310** | **31.3026** |

**`31.3026 / 18.0828 = 1.7310617`.** And note the identity that makes the §3 dichotomy exact:
this ratio is *identically* `poseC(pw1)/poseC(cx1)` — because `dS/d(d_pose) = 5/poseC`, the marginal
ratio and the contribution ratio are reciprocals of each other by construction, not by coincidence.

---

## §2 THE DERIVATION — from `upstream/` and first principles, written before consulting our receipts

**Contamination disclosure (honest, per §0 of the amendment).** When the amendment arrived I had
already read: the constants above, the `pz1` per-pair array, and the `§0` headlines of six pose memos
(`pw1`, `mq1`, `pj2`, `pg1`, `wd1`, `qd2`). Those are *measurements and status*, not treatment
designs. I had read **no** design or derivation section. The derivation below was written from
`upstream/modules.py` + `upstream/evaluate.py` alone and then diffed against ours in §2.7. Label this
**partially contaminated but independently derived**; the four agreements in §2.7 are corroboration
from a second path, the two differences are the finding.

### 2.1 What the scorer actually reads (VERIFIED_VIA_SOURCE_INSPECTION)

`upstream/modules.py`, `PoseNet.preprocess_input`:
```python
x = einops.rearrange(x, 'b t c h w -> (b t) c h w', ...)
x = torch.nn.functional.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')
return einops.rearrange(rgb_to_yuv6(x), '(b t) c h w -> b (t c) h w', b=batch_size, t=seq_len, c=6)
```
`SegNet.preprocess_input`:
```python
x = x[:, -1, ...]   # Use only last frame
return torch.nn.functional.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')
```
`PoseNet.compute_distortion`:
```python
sum((out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2]).pow(2).mean(dim=...) ...)
```

Four facts fall straight out:
- **Interpolate FIRST, then `rgb_to_yuv6`** — and the interpolate target is *literally*
  `segnet_model_input_size`. **The two scorers make the identical resize call.** There is no
  "different lattice" to hide in.
- **SegNet reads `x[:, -1, ...]`** — with `seq_len = 2`, that is frame_1. **frame_0 is structurally
  invisible to SegNet.**
- **`d_pose` per pair = `mean` over the first `h.out//2 = 6` head outputs of the squared difference**
  — i.e. `‖e‖²/6`, and the *mean over pairs* is taken before the `√`.
- **`rate = archive.zip stat / uncompressed_size`** — linear in bytes, no time term.

### 2.2 The pose objective is TARGET-MATCHING, not fidelity (DERIVED)

`d_pose` depends on the delivered pair **only** through the 6-vector `p^c ∈ R⁶`. The map
(frame_0, frame_1) → `p^c` has domain dimension `2·3·874·1164 ≈ 6.1M` and codomain 6. The generic
fibre has dimension `≈ 6.1M − 6`. **Almost everything about the delivered frames is free for pose.**
The correct object to store is not a frame, and not a residual — it is *a point in the preimage of
`p^gt`*, described as cheaply as possible.

### 2.3 Why the encoder must invert (DERIVED)

The receiver cannot run PoseNet (73 MB of weights would be counted; and the standing rule forbids
scorers at inflate time). So the receiver cannot solve for a preimage point. **The encoder inverts,
and ships the parameters of the chosen preimage point.** Cost = parameter count, not target count.
If a legal family `F(θ)`, `θ ∈ R^k`, has `θ ↦ p^c` locally surjective near `p^gt`, then `k = 6`
suffices per pair.

### 2.4 frame_0 is a zero-seg-price pose actuator (DERIVED — this is the sharp one)

From 2.1: frame_0 never reaches SegNet. Therefore **any** modification of frame_0 has **exactly zero**
`d_seg` cost, while moving `p^c` through channels 0–5 of PoseNet's 12. The optimal pose treatment
spends its degrees of freedom in frame_0 **first**, and exhausts them before touching frame_1.

**Corollary, which contradicts the intuitive objective:** `d_pose` compares `PoseNet(GT f0, GT f1)`
against `PoseNet(f0, f1)` where `f1` is lossy. Substituting the *true* GT frame_0 into a pair whose
frame_1 is lossy produces a **mismatched** pair. So the true GT frame_0 is **not** the minimiser.
Absolute-fidelity objectives on frame_0 aim at the wrong target — derived here independently, and it
matches the corpus's measured 3.05–16.66 (GT f0) vs 0.0008 (decoded f0).

### 2.5 The allocation problem is LINEAR, so the optimal policy is strict greedy (DERIVED)

Minimise `√(10 · (1/N)Σᵢ mᵢ)` subject to `Σ cᵢ ≤ B`. `√` is monotone, so this is *identical* to
minimising `Σᵢ mᵢ`. **There is no interior water-fill on the pose axis.** The optimum is a strict
greedy ordering by `Δmᵢ per byte`. The concavity affects *how much S you book*, never *which pairs
you pick*. (Contrast the seg axis, where the objective is already linear in flips and the same
conclusion holds for a different reason.)

And because `d_pose` is a **mean of squares** over a scene-varying process, the distribution is
right-skewed, so the greedy order is steeply front-loaded. §5 measures exactly how steeply.

### 2.6 The byte-crossover the amendment asked for (DERIVED — the deepest result here)

Define each axis's **break-even byte price of one physical quantum**:

- **seg**, quantum = 1 pixel flip: `ΔS = 100/PX`, so
  `W = (100/PX)·(DEN/25) = 4·DEN/PX = 1.2731082 B/flip`. **Contains no state variable. INVARIANT.**
- **pose**, quantum = one unit of a pair's residual `m`: `ΔS = (5/poseC)·(m/N)`, so
  `W_pose ≡ DEN/(5·N·poseC)` B per unit `m`. **Contains `poseC`. NOT invariant — and `poseC` is in
  the denominator, so `W_pose` grows without bound as pose improves.**

| base | `poseC` | `W_pose` (B per unit m) |
|---|---:|---:|
| `v4c` | 0.3222499 | 38,837 |
| `pw1` | 0.2765059 | 45,262 |
| **`cx1`** | **0.1597310** | **78,352** |
| PR130's pose level | 0.0152680 | **819,699** |

**⇒ The two axes do not cross once and settle. Seg's byte-generosity is frozen for all time; pose's
diverges. The gap between them is a widening wedge, and every unit of progress on pose widens it.**
This is a *structural* asymmetry of `S`, not a property of our vehicle — `√(10·d_pose)` is the only
operating-point-dependent leg of the score.

Two further crossovers, both derived:

- **Marginal parity:** `5/√(10·d_pose*) = 100 ⇒ d_pose* = 2.5e-4`. `cx1` sits at `0.0025514` =
  **10.21× above** it, so *per unit of its own metric* seg's marginal (100) is still **3.195×** larger
  than pose's (31.30). **This is the number that makes "seg is 61% of the gap, do seg" feel right, and
  it is the wrong comparison** — a flip and a unit of `d_pose` are not the same thing, so their
  marginals are not comparable. Only the **byte** price is.
- **Byte parity (the comparable one):** measured, `#826/gr1` bought seg flips at **32.52 B/flip**
  against `W = 1.2731` ⇒ `η = 25.54`, i.e. **underwater**. Explicitly: one flip at 32.52 B gains
  `8.477e-07 S` and costs `2.165e-05 S` — **net +2.081e-05 S, a LOSS.** Against that, the measured
  pose tail break-even is 818–40,503 B/pair (§5). **At `cx1`, on the measured efficiencies, pose is
  the cheaper axis at every `k` from 1 to 600, and there is no `k` at which the ordering returns.**

**Scope, stated before anyone over-reads it:** that last verdict is **FORMULATION-scoped**. It kills
*seg-by-correction-stream at the measured 32.52 B/flip*. It says nothing about the seg **base
representation** (training), which is the seg axis's real lever and is not priced here. §7-R1-a.

**Stock vs price, the reason gap-share misleads:**

| axis | S available | break-even budget to take it all |
|---|---:|---:|
| pose → 0 | 0.1597310 | **239,887 B** |
| seg → PR130's level | 0.401519 | 603,009 B |

The archive is **353,808 B**. Seg's entire remaining stock cannot be bought by corrections even at
break-even, let alone at the measured 25.5× underwater rate. Pose's can, at 68% of the current
archive — and the measured tail buys the first two-thirds of it for **~1–2% of that budget**.

### 2.7 DIFF against our corpus

| derived here | our corpus | verdict |
|---|---|---|
| interpolate-then-yuv6; both scorers make the identical resize call | CLAUDE.md, corrected 2026-08-03 by `ddm_pz1` | **AGREE** — independent corroboration |
| `d_pose*` marginal-parity crossover at `2.5e-4` | CLAUDE.md "SegNet vs PoseNet importance" | **AGREE** — arrived at from `compute_distortion` alone |
| frame_0 is seg-free ⇒ spend pose DOF there first | corpus states the seg-freedom; `WarpPoseOracle` shows the live vehicle *already* synthesises `f0 = warp(f1)` at **zero** independent bytes | **AGREE, and ours is stronger** — the vehicle already exhausts this. My "spend DOF in frame_0" is *already the design*. |
| true GT frame_0 is **not** the minimiser | measured 3.05–16.66 vs 0.0008 | **AGREE** |
| **`W_pose` diverges while `W` is invariant ⇒ a widening wedge, not a crossover** | **not found in the corpus** (`mq1` computed the ratio at one point and called it "48×"; nobody differentiated it) | **NEW** |
| **pose allocation is strictly greedy, never water-filled, because `√` is applied after the mean** | corpus repeatedly says "waterfill" on the pose axis | **DIFFER — and I believe the corpus is loose.** The `√`-after-mean makes the constrained problem exactly linear in `mᵢ`. "Waterfill" is the right word on the *seg* axis and on *rate*, not here. Low blast radius (greedy and waterfill coincide when the cost model is linear per pair) but the word is carrying a false generality. |

---

## §3 THE DICHOTOMY — my charter's premise, corrected

The charter says: *"every banked pose delta was priced at a marginal that is now 1.73× too low ⇒
pose deltas are UNDER-priced."* That holds **only** for levers whose effect is an **absolute** removal
of `d_pose` mass. It **inverts** for levers whose effect is a **fraction**:

| model | law | `pw1 → cx1` factor |
|---|---|---:|
| **ADDITIVE** — removes a fixed absolute `Δd_pose` | `ΔS = (5/poseC)·Δd_pose` | **1.7311× RICHER** |
| **MULTIPLICATIVE** — removes fraction `f` of whatever remains | `ΔS = poseC·(√(1−f) − 1)` | **0.5777× = 1.7311× POORER** |

Worked, for the fractions our own lineage actually delivered:

| `f` | `ΔS @ pw1` | `ΔS @ cx1` | ratio |
|---:|---:|---:|---:|
| 0.050 | −0.0070013 | −0.0040445 | 0.5777 |
| 0.109 (`pw1`'s own AB) | −0.0155042 | −0.0089564 | 0.5777 |
| 0.250 | −0.0370448 | −0.0213999 | 0.5777 |
| 0.506 (`pj2`'s delivered fraction) | −0.0821634 | −0.0474639 | 0.5777 |

The ratio is `poseC(cx1)/poseC(pw1)` exactly, independent of `f` — the two models are exact
reciprocals through the same constant.

**Which model applies is decided by the lever's mechanism, and it is knowable:**
- **A tail-targeted solve is ADDITIVE.** It removes the mass of specific pairs. `d_pose` is a mean;
  removing pair `i` removes `mᵢ/N` regardless of what the other 599 pairs do. **⇒ 1.7311× richer.**
  The charter's premise is *correct for exactly the work it commissioned* (PU1-2).
- **A global re-solve / re-parameterisation is MULTIPLICATIVE.** It improves every pair proportionally
  to its own residual. **⇒ 1.7311× poorer.** `pj2` (`f = 0.506`) and `pw1` (`f = 0.109`) are both of
  this kind, so *neither* would re-earn at `cx1` what it earned at its own base — a second, independent
  reason (beyond `qd2`'s absorption) that they re-price to 0.
- **`mq1`'s search headroom is ambiguous** — measured as per-coordinate `gap_search` on 48
  mass-ordered pairs, which is additive in form (a fixed set of pairs, a fixed reachable improvement)
  but multiplicative in origin (a better search on every pair). §4 prices it additively and flags the
  assumption.

**Net:** a blanket 1.73× uplift would have been wrong on `pw1`, `pj2` and every fractional row — by
3.0× (the round trip 1.7311 × 1.7311). The correction is real but it is **mechanism-conditional**.

---

## §4 PU1-1 — the re-pricing sweep, with its denominators

**Sweep 1 — the ledger.** `.omx/state/canonical_task_status.jsonl`: **417 rows parsed, 0 unparsed,
148 distinct `task_id`.** Pose-matching (`pose|dpose|posenet|gn_|screw|twist|se3`) across
`task_id + title + event_notes`: **35 tasks**. Of those, **10** carry an `actual_delta_s` or a
`predicted_delta_s_band`; **3** carry a real non-zero `actual_delta_s`.

| task | `actual_delta_s` | re-priced vs `cx1` | why |
|---|---:|---:|---|
| **#850** POSE GN TERMINATION | −0.0675451 | **0.0000000** | one `pj2` run's **total**, stamped on three distinct scopes; already inside `cx1` |
| **#873** MENU-AS-RD-CODEBOOK 2nd coord | −0.0675451 | **0.0000000** | same run, same stamp |
| **#882** START-IS-THE-LEVER 2nd coord | −0.0675451 | **0.0000000** | same run, same stamp |
| #827 ep854 composition | band only `[−0.0866789, 0]` | — | `qd2`: measured full-S row is **+19.22** |
| 4 legacy rows (`closed_spec_boundary_solver`, `cross_pair_waterfilled_corrector`, `task73_dykstra`, `g55_g57`) | 0.0 or null | **0.0000000** | already zero |

**Ledger verdict: 0.0000000 re-priceable. `qd2`'s conclusion holds on the pose axis too, for a
second independent reason (§3: `pj2` is multiplicative, so it would not re-earn even if it were
outside `cx1`).**

I also confirm `qd2`'s structural defect on this subset: **all 3 pose rows carrying `actual_delta_s`
name no baseline.** Their `source_design_memo` points at `pj2`, which does state its base — but the
ledger row does not, so a reader summing the column banks `−0.2026353` for a true `0.0000000`.

**Sweep 2 — the research corpus.** `ls .omx/research | grep -iE 'pose|pz|pj|pw|tail|dpose'` returns
**~300 paths**; of these I opened **13** (the live-lineage arms: `pw1 mq1 pj2 pg1 wd1 bp2 su2 pz1
qd2 uv1 cr2r p3v2 pfs1`) and read `§0` of each. **Denominator stated: 13 of ~300 opened; the other
~287 are pre-`v4d`-lineage and were not individually re-priced. This sweep is NOT exhaustive and I do
not claim it is.** Selection rule: a row is re-priceable only if its base is on the
`v4c → v4d → pw1 → mq1 → ms8 → dc1 → pj2 → cx1` chain, because anything older is on a retired vehicle
and transfers as a hypothesis, not a number (the ancestor-vehicle law).

**The one genuinely under-priced row: `mq1`.**

`mq1` states its own constants at `§0`: *"`dS/d(d_pose) = 5/sqrt(10·d_pose) = 18.083`"* and
*"Gap to the bar (PR130 0.172141) = 0.7754681 … 1% of the gap = 11,646 archive bytes."* **`18.083` is
`pw1`'s marginal to 5 significant figures (`18.0828`)** and `0.7754681` is `pw1`'s gap. Two stale
constants, both stale in the same direction.

Re-priced at `cx1` under the **ADDITIVE** reading (flagged, §3):

| `mq1` coordinate | `mq1` %gap | implied `Δd_pose` | `ΔS @ cx1` | %gap @ `cx1` | growth |
|---|---:|---:|---:|---:|---:|
| `p1` lateral | 0.4694% | 2.013e-04 | −0.0063011 | 0.963% | 1.731× |
| `p2` vertical | 0.8743% | 3.749e-04 | −0.0117364 | **1.794%** | 1.731× |
| `beta` rolling-shutter | 0.3358% | 1.440e-04 | −0.0045077 | 0.689% | 1.731× |
| **TOTAL search** | **1.82%** | **7.805e-04** | **−0.0244312** | **3.734%** | **1.731×** |

The **%-of-gap** figure grows by **2.05×** (1.7311 marginal × 1.1851 denominator); the **ΔS** grows
by 1.7311×. Both corrections point the same way and they compound — this is `qd2`'s "denominator
drift" and my "marginal drift" acting together on one row.

**And here is the caveat that must travel with it.** The implied `Δd_pose = 7.805e-04` is **30.6% of
`cx1`'s entire remaining `d_pose` (0.0025514)**. Between `mq1` and `cx1`, `pj2` delivered a **50.6%**
reduction. Whether `pj2`'s re-solve consumed `mq1`'s `p1/p2/beta` search gap is **NOT DETERMINABLE
from the memos** (`pj2` re-solved `(pose, s_t)` jointly; `mq1` measured per-coordinate `gap_search`
on 48 pairs at the `pw1` base). **Status: ASSUMED_AWAITING_VERIFICATION. The `−0.0244312` is an
UPPER BOUND and is PROVISIONAL.** §5 is exactly the measurement that decides it, which is why PU1-1
and PU1-2 converge on one probe.

---

## §5 PU1-2 — the tail, measured on the live base

### 5.1 The instrument

`.omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json` — n600 per-pair `d_pose` at
`base = v4d_cx1_pj2ix2`. **Validated:** its mean is `0.0025513987` against the evaluator's own
`0.00255143` — **1.6e-5 relative**, and `pz1` independently records the same agreement at 1.2e-5.
**The per-pair array integrates to the authoritative number**, which is a real (if partial —
§7-R1-c) cross-check that the distribution is not fabricated.

Population: `mean = 0.0025513987`, `median = 5.513e-04`, **mean/median = 4.6×**, `max = 0.4733`,
`min = 1.195e-05`. Right-skewed, exactly as §2.5 predicts.

### 5.2 The mass is in four pairs, and the four pairs are one scene

| rank | pair | `d_pose` | % of mass | cum % | × mean |
|---:|---:|---:|---:|---:|---:|
| 1 | **74** | 0.473295 | **30.92%** | 30.92% | **185.5×** |
| 2 | 67 | 0.157367 | 10.28% | 41.20% | 61.7× |
| 3 | 21 | 0.113356 | 7.40% | 48.60% | 44.4× |
| 4 | 523 | 0.112023 | 7.32% | **55.92%** | 43.9× |
| 5 | 16 | 0.049147 | 3.21% | 59.13% | 19.3× |
| 10 | 42 | 0.016202 | 1.06% | **67.24%** | 6.4× |

Only **4 pairs exceed `d_pose = 0.05`**; **13** exceed 0.01; **29** exceed the population mean.

Temporal structure — **this is not a set of independent hard pairs, it is a scene**:

| block | mass share | mean / population |
|---|---:|---:|
| [0,60) | 19.36% | 1.94× |
| **[60,120)** | **49.74%** | **4.97×** |
| **[60,80) — 20 pairs** | **45.60%** | **13.68×** |
| [120,480) | 19.82% | 0.24–0.61× |
| [480,540) | 9.68% | 0.97× |
| [540,600) | 1.40% | 0.14× |

### 5.3 Break-even, and the falsifier that was set at the wrong operating point

Solving the top `k` to zero (**UPPER BOUND** — see §7-R1-b), exact (not linearised):

| k | `ΔS` (exact) | break-even bytes | **break-even B/pair** |
|---:|---:|---:|---:|
| 1 | −0.0269690 | 40,503 | **40,503** |
| 5 | −0.0576157 | 86,528 | 17,306 |
| **10** | **−0.0683036** | **102,580** | **10,258** |
| 25 | −0.0775821 | 116,514 | 4,661 |
| **56** | **−0.0848158** | 127,378 | **2,275** |
| 112 | −0.0951271 | 142,864 | 1,276 |
| **200** | **−0.1089400** | 163,608 | **818** |
| 600 | −0.1597310 | 239,887 | 400 |

At the charter's pre-registered 120 B/pair:

| k | pose gain | rate cost | **NET ΔS** | % of gap | `η = 120/breakeven` |
|---:|---:|---:|---:|---:|---:|
| 10 | −0.0683036 | +0.0000799 | **−0.0682237** | **10.43%** | 0.0117 (**85× profitable**) |
| 56 | −0.0848158 | +0.0044746 | **−0.0803412** | 12.28% | 0.053 (19× profitable) |
| 112 | −0.0951271 | +0.0089491 | **−0.0861780** | 13.17% | 0.094 (11× profitable) |
| 200 | −0.1089400 | +0.0159806 | **−0.0929594** | 14.21% | 0.147 (6.8× profitable) |

**The pre-registered falsifier — "> ~600 B/pair ⇒ the cheap-tail route dies" — is REFUTED as a
threshold.** Measured break-even at the live operating point is **818 B/pair at k=200 and 10,258 at
k=10**. The 600 B/pair figure was set when pose was 1.73× cheaper; carried forward unchanged, it
would have killed a live route. **Restated falsifier, live: the tail route dies only if the realised
per-pair cost exceeds the k-dependent break-even in the table above — 818 B/pair is the floor of that
curve, and only at k=200.**

Note also that **the linearised marginal UNDER-states the prize** for moves this large: at k=56,
linear-at-`cx1` gives −0.0623 where exact gives **−0.0848** (36% larger), because `√` is concave and
the marginal *rises* as you descend. The 1.73× correction and the concavity correction stack.

### 5.4 The split: one rigid pair, thirteen responsive ones

Under `pz1`'s paired perturbation (`ratio = solved/base`):

| pair | `d_pose` | ratio | |
|---:|---:|---:|---|
| **74** | 0.47329 | **0.981** | **RIGID** |
| 67 | 0.15737 | 0.853 | responsive |
| 21 | 0.11336 | 0.904 | responsive |
| 523 | 0.11202 | 1.234 | responsive |
| 16 | 0.04915 | 1.213 | responsive |
| 71 | 0.04416 | 1.245 | responsive |
| 44 | 0.02080 | 1.653 | responsive |
| 42 | 0.01620 | 1.529 | responsive |
| 275 | 0.01322 | 1.529 | responsive |
| … | | 0.85–1.65 | 13 of top 14 responsive |

**Pair 74 holds 30.92% of the axis and is the only tail pair that does not move.**

**Interpretation, correctly labelled.** `pz1`'s perturbation is a *frame_1* change that gets resampled
into frame_0 (the live vehicle synthesises `f0 = warp_rgb(f1, H)` — `WarpPoseOracle.d_pose`, source-
inspected). A pair whose `d_pose` barely moves when *both* delivered frames change is showing
anomalously low sensitivity, which is the signature of a saturated/pinned residual — a **model** wall
(the 6-param warp family cannot reach `p^gt` for that pair) rather than a **search** wall.
**This is INFERRED_FROM_DOMAIN_LITERATURE-grade at best, NOT measured**: responsiveness to a frame_1
perturbation does not bound reachability within the pose-parameter family. **PROVISIONAL.** §8 gives
the probe that settles it, and it is cheap.

**Why this matters more than the re-pricing.** If pair 74 is model-locked, the honest prize from the
tail is `−0.0683036 − 0.0269690 = −0.0413` at k=10 (still **6.3% of the gap**), and the campaign has
located a *named structural limit of the shipped warp family* — an honest
"the reachable point requires mechanism X" result, which the amendment names as first-class. If it is
merely search-stuck, one pair is worth **−0.0269690 S = 4.12% of the gap** on its own, for a
break-even budget of **40,503 bytes**.

### 5.5 PRE-PROBE, RUN: the GT targets — and the right scale for the tail

I ran the §8 pre-probe (`p3v2.load_targets(600)`, no scorer time). **It refutes the branch of the
model-wall hypothesis I found most likely.**

GT target set: `‖t‖` mean **31.2607**, sd **1.2563** (CV **4.02%**), per-component RMS spread
**`sd_vec = 0.5134`**. The targets are tightly clustered around a large common mode.

| pair | `d_pose` | `‖e‖ = √(6·d_pose)` | `‖e‖/‖t‖` | **`‖e‖/sd_vec`** | GT-target rank |
|---:|---:|---:|---:|---:|---:|
| **74** | 0.47329 | 1.6852 | 5.19% | **3.28** | **98 / 600** |
| 67 | 0.15737 | 0.9717 | 2.97% | 1.89 | 85 / 600 |
| 21 | 0.11336 | 0.8247 | 2.39% | 1.61 | **3 / 600** |
| 523 | 0.11202 | 0.8198 | 2.34% | 1.60 | **1 / 600** |
| 16 | 0.04915 | 0.5430 | 1.60% | 1.06 | 19 / 600 |
| 71 | 0.04416 | 0.5147 | 1.62% | 1.00 | 172 / 600 |
| median pair | 5.513e-04 | 0.0575 | 0.184% | 0.112 | — |

**MEASURED, three findings:**

1. **Pair 74's GT target is ORDINARY — rank 98/600, `z = +0.96`.** There is nothing extreme about the
   motion it must reproduce. **The "extreme-motion scene event ⇒ the warp family cannot express it"
   branch of the model-wall hypothesis is REFUTED for the pair that carries 30.92% of the axis.**
   This *weakens* my §5.4 lean and shifts weight toward the search-wall branch — which is the branch
   worth **−0.0269690 S (4.12% of gap) for one pair**. (It does not refute model-wall *in general*:
   an ordinary target says nothing about scene cuts, occlusion, or absence of a valid ground plane.
   Status: one branch closed, the hypothesis still open.)

2. **Pairs 523 and 21 ARE target outliers — rank 1/600 and 3/600.** So the tail is not homogeneous:
   two of the top four are plausibly hard *because their targets are extreme*, while pair 74 is hard
   for some other reason. **The tail has at least two distinct mechanisms and must not be treated as
   one population.**

3. **The right normalisation, and it is more alarming than "185× the mean."** `sd_vec = 0.5134` is the
   per-component spread of GT targets across the population, so `‖e‖ > sd_vec` means the delivered
   pose sits further from its own GT than a typical GT sits from the population mean — **the pose
   readout carries no usable information for that pair.** Measured: **6 of 600 pairs are in that
   state, and they carry 62.0% of all pose mass.** Pair 74 is at **3.28× sd_vec**.

   Read the other way, this is the encouraging number: relative to `‖t‖`, pair 74 is only **5.19%**
   off, and the median pair is 0.184% off. **These are not catastrophic failures; they are moderate
   relative errors that the `mean-of-squares` amplifies into 62% of the axis.** A tail solve does not
   need to work a miracle on 6 pairs — it needs to move them from ~3× `sd_vec` to ~1×.

### 5.6 `gt1` cuts our way

`ddm_gt1` (`19772540e9`) killed the reachability-tail hypothesis: `γ_reach = ‖Jᵀê‖₁` came out
**wrong-signed at 5.44×** — hard pairs are *more* reachable, not less. Read as a negative that was
looking for a second difficulty coordinate, it is a *positive* for the tail solve: **the hard pairs
are not hard because they are unreachable.** That is consistent with 13-of-14 responsiveness, and it
makes pair 74's rigidity the genuine anomaly rather than the expected case.

---

## §6 PU1-3 — the stale-carrier law, and what it binds here

`ddm_pz1` measured at n600 that a seg-side change which looked free cost **`d_pose +0.0310217`**,
because a null field built in one lattice is **resampled** onto another. `§2.1` shows at source that
PoseNet and SegNet make the *identical* interpolate call, so the mechanism is lattice **change**, not
lattice difference. **Null-space membership does not survive a change of lattice.**

**Consumers of any bytes a tail solve would modify** (enumerated, source-inspected):
- the **pose parameter stream** (6 f16/pair + the `s_t` index stream) — consumed by
  `pfs1_warp_receiver.pose_to_homography`, and *only* there;
- **frame_0** — synthesised, never stored; consumed by PoseNet channels 0–5 **and by nothing else**
  (SegNet takes `x[:, -1, ...]`);
- **frame_1** — consumed by **both** scorers.

**⇒ A tail solve that touches only the pose parameter stream is `d_seg`-neutral BY CONSTRUCTION, not
by measurement.** frame_0 is re-derived from the same frame_1 with a different `H`; frame_1 is
untouched; SegNet never sees frame_0. This is the one place on the vehicle where the stale-carrier
law does *not* bite — and it is precisely where §2.4 says to spend. **The reverse is not true:** any
seg-side change invalidates every pose parameter fitted against the old frame_1, so a tail solve must
be sequenced **after** the seg base is frozen (which is the standing `#383` conditioning-gate order).

---

## §7 ADVERSARIAL REVIEW

### Round 1 — attacking my own headline

**R1-a. "Seg is dead" is a scope error I nearly made.** §2.6 shows seg-*by-correction* is net-negative
at 32.52 B/flip. I initially wrote that as "the axis ordering is inverted." **That over-reaches.** The
seg axis's principal lever is the **base representation** (training), which is not a correction stream
and is not priced by `W` at all. Corrected in §2.6 to **FORMULATION scope**. *One failed formulation
is not a dead family.*

**R1-b. "Solve top-k to zero" is an upper bound and I must not let it travel without that.** No pair
reaches `d_pose = 0` — that requires exactly matching GT's 6-vector through a lossy pair. Every `ΔS`
in §5.3 is a **ceiling**. The break-even B/pair figures are correspondingly ceilings. Labelled at
every occurrence. The *ratio* between k's is more robust than the absolute.

**R1-c. My cross-check validates the SUM, not the ALLOCATION.** `pz1`'s per-pair mean matching the
evaluator to 1.6e-5 proves the array integrates correctly. It does **not** prove pair 74 specifically
carries 0.4733 — a mis-allocation between pairs would preserve the mean. **The 30.92% single-pair
claim rests on `pz1`'s instrument being per-pair-correct, which I did not independently verify.**
Status: VERIFIED_VIA_EMPIRICAL_ANCHOR *for the aggregate*, ASSUMED *for the allocation*. §8 probe
settles this too, at zero extra cost.

**R1-d. Would my conclusion survive if the additive assumption on `mq1` were wrong?** If `mq1`'s
headroom is multiplicative, the re-price is `0.5777×` not `1.7311×`, giving
`0.0141135 × 0.5777 = 0.0081534 S = 1.246% of gap` — still positive, still real, but **3.0× smaller**
and no longer the corpus's largest parked pose row. **The §4 headline is conditional on a flagged
assumption and I have marked it PROVISIONAL rather than resolving it by preference.**

**R1-e. Did I check that the pz1 base is the same vehicle as `cx1`?** Yes — `base:
"v4d_cx1_pj2ix2"`, and `d_pose_base_mean` reproduces `cx1`'s evaluator row. Not a borrowed number.

**Round 1 FOUND FIVE ISSUES ⇒ counter resets to 0.**

### Round 2 — attacking the Round-1 fixes

**R2-a.** The R1-a fix says the seg base "is not priced by `W` at all." **Is that true?** `W` is the
break-even bytes for one flip *however obtained* — including via base bytes. So the seg base **is**
priced by `W`; what differs is that a base change moves many flips per byte, so its `η` is unknown
rather than inapplicable. **Corrected: the honest statement is `η_seg-base` is UNMEASURED, not that
`W` does not apply.** (§2.6 now reads "not priced here", which is right.)

**R2-b.** R1-b says every `ΔS` is a ceiling — **but the §5.3 `NET` column subtracts a rate cost that
is a FLOOR** (120 B/pair is optimistic). So the NET column is a ceiling built on a floor: doubly
optimistic. **Stated. The `η` column is the robust one** because it is a ratio of the two.

**R2-c.** §2.5 claims "no interior water-fill." **Counter-example check:** if per-pair cost `cᵢ` is
non-linear in `Δmᵢ` (diminishing returns *within* a pair), the greedy order is over
`dΔmᵢ/dcᵢ` and becomes a genuine water-fill *within* the linear objective. **My claim is correct for
the OBJECTIVE (it is linear in `mᵢ`) and loose about the ALLOCATION (which water-fills if per-pair
cost curves are convex).** Narrowed in §2.5's wording: the `√` never enters the allocation; that is
the load-bearing part and it stands.

**Round 2 FOUND THREE ISSUES ⇒ counter resets to 0.**

### Round 3 — assumption challenge

**The shared assumption this entire arm operates within: that the pose axis is worth optimising as a
separate axis at all.** Every arm on this axis, mine included, has taken `d_seg` and `d_pose` as
separable and priced them independently. **The corpus's own deep-math frame says they are not:** by
Chasles, the same `se(3)` twist `ξ` that warps the partition for `d_seg` *is* the pose for `d_pose` —
one object, dual use. If that is right, the correct object is a **single** `ξ` per pair, and the
"pose payload" and the "seg warp" are two readouts of one stored quantity, so the marginal pricing in
§2.6 is pricing two projections of one variable as if they were independent goods.

- **Would violating it unlock breakthrough?** Plausibly yes, and in *our* favour: a shared `ξ` means
  the tail solve's bytes are **already paid for** by the seg side, driving the pose tail's realised
  B/pair toward zero and its `η` toward 0. That would make the §5.3 table conservative by a large
  factor.
- **Classification: ASSUMED_AWAITING_VERIFICATION.** The shipped vehicle does *not* currently share
  `ξ` — `WarpPoseOracle` derives `H` from a pose-6 that is fitted against `d_pose` alone. So the
  separable pricing is *correct for the vehicle as built* and possibly pessimistic for the vehicle as
  designed. **Every §5.3 number is therefore a valid lower bound on the shared-`ξ` design.**

**Assumption 2: that `d_pose` mass is a property of the pair rather than of the SCENE.** §5.2 measures
45.6% of mass in a 20-pair block — so treating pairs as the allocation unit may be wrong; a
scene-level parameterisation could amortise one description across ~20 pairs and cut B/pair by ~20×.
**INFERRED. Not tested. This is the single largest un-costed upside in this memo.**

**Round 3 FOUND TWO ISSUES (both scoping, both folded above) ⇒ counter resets to 0.**

**SEAL STATUS: NOT SEALED. 0 of 3 clean passes.** Three rounds, ten findings, all folded into the
text above. I am stopping the review here because the remaining open items are all resolved by the
*same one measurement* (§8) rather than by more reading — continuing to review without it would be
polish-hoarding. **Every verdict in this memo is PROVISIONAL to that degree, and the two load-bearing
ones (§4 `mq1` availability, §5.4 pair-74 rigidity) are explicitly marked so.**

---

## §8 NEXT-IF-RESUMED

**The single measurement that collapses four open questions at once** (§4 `mq1` availability,
§5.3 realised B/pair, §5.4 model-vs-search, §7-R1-c per-pair allocation):

> **A multi-start floor probe on the tail pairs, through the shipped receiver + frozen PoseNet.**
> For pairs `[74, 67, 21, 523, 16, 71, 18, 44, 22, 42]`, run a global (CEM / multi-start) search over
> the 6-dim warp pose at fixed `s_t`, and record the best realised `d_pose` at the **shipped f16**
> quantisation. Compare to the shipped residual.
> - If pair 74's floor ≈ 0.4733 ⇒ **MODEL wall confirmed**; the shipped warp family cannot reach that
>   scene; the honest tail prize is −0.0413 at k=10 and the campaign has a named structural limit.
> - If it drops ⇒ **SEARCH wall**; one pair is worth −0.0269690 S (4.12% of gap) at a break-even
>   budget of 40,503 bytes.
> - Either way, the realised `Δmᵢ` per pair gives the true B/pair and settles whether `pj2` consumed
>   `mq1`'s search gap.

**Instrument (source-inspected, exists):** `experiments/ddm_pfs1_ep_warp_pose_solve.py`
— `WarpPoseOracle.d_pose(pidx, f1, warp_pose, s_t)` scores a candidate through
`pfs1_warp_receiver.pose_to_homography` → `warp_rgb` → `_to_uint8` → frozen PoseNet, and
`d_pose_shipped(...)` does it at f16. **Known gap:** the oracle is wired to
`/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2c_aimed_archive.zip` (the pb1 vehicle), whose
frame_1 is **not** `cx1`'s. `cx1`'s own receiver + container are at
`/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_cx1_pj2ix2/`
(`pfs1_warp_receiver.py`, `ddm_ix2_archive_container.py`, `ddm_tr1_runtime.py`). **The probe needs a
~30-line oracle re-point at the `ix2` container to render `cx1`'s frame_1.** That re-point is the only
build work; I did not do it because my remaining context was better spent landing the pricing than
half-building the probe.

**The pre-probe is DONE — see §5.5.** Pair 74's GT target is rank 98/600 (ordinary), which closes the
extreme-motion branch and shifts weight to the search-wall reading. Pairs 523 and 21 *are* target
outliers (rank 1 and 3), so the probe should expect **two mechanisms**, not one, and should report
per-pair rather than pooled.

**Sharpened success criterion (from §5.5's normalisation) — and it replaces the solve-to-zero ceiling
with a DEFENSIBLE target.** The goal is not `d_pose → 0`; it is to bring the 6 uninformative pairs
from `‖e‖/sd_vec` of 1.00–3.28 down to **~1.0**, i.e. to the edge of informativeness. Computed:

| target | `ΔS` | % of gap | break-even budget |
|---|---:|---:|---:|
| pair 74 alone, `3.28 → 1.0 sd_vec` (a 90.7% `d_pose` cut) | **−0.0242396** | **3.70%** | **36,404 B** |
| all 6 uninformative pairs `[74, 67, 21, 523, 16, 71] → 1.0 sd_vec` | **−0.0410524** | **6.27%** | **61,653 B = 10,276 B/pair** |

**This is the row to aim at: 6 pairs, a 10,276 B/pair break-even budget, 6.27% of the gap.** It is not
an upper bound built on solve-to-zero — it is a bounded, physically-motivated target. And pair 74
alone (**−0.0242396**) is numerically the same size as the entire re-priced `mq1` row in §4, for a
tenth of the work.

**Do NOT inherit:** the 600 B/pair falsifier (refuted, §5.3) · `1% of gap = 10,907 B` (it is 9,827.2)
· a blanket 1.73× uplift on pose rows (mechanism-conditional, §3) · "waterfill the pose axis"
(the objective is linear, §2.5).

**Scorer slot: RELEASED.** This arm ran no scorer job.
