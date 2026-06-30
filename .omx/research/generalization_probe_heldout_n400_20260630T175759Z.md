# GENERALIZATION PROBE — do our value-GENERATORS produce STABLE outputs on held-out data, or did we overfit one clip?

`[macOS-numpy advisory · NON-PROMOTABLE]` — $0 CPU-only, NO GPU, READ-ONLY on GT caches.
Pointer **0.19110 UNMOVED**. No score claim. Not committed (left for review).

- generated: 2026-06-30T17:57:59Z
- method: re-ran the **same two generators** (no math changed) on held-out GT and compared to the anchor.
- threshold: **HOLDS = held-out within ~±15% of anchor** (domain+frozen-scorer-fundamental → generator portable, value = generator's per-instance output). **SHIFTS = instance/window-conditioned.**

## Data provenance (verified disjoint)

`gt_n600.npz` = full clip (600 pairs). The subsets map exactly onto n600 frame indices (pose-row match, max dist 0.0):

| cache | frames | n600 indices | role |
|---|---:|---|---|
| `gt_n96.npz` | 96 | **0..95 contiguous** | tool-1 anchor (opening segment) |
| `gt_strided_n200.npz` | 200 | stride-3 {0,3,6,…,597} | tool-2 anchor |
| `gt_heldout_n400.npz` | 400 | **complement of n200** (idx%3≠0) | held-out — **0 overlap with n200** ✔ |

Held-out n400 is the **exact complement** of the n200 anchor (0 shared frames) → clean for **tool 2**.
It overlaps the n96 anchor by 64 frames, so for **tool 1** I built a strictly-disjoint held-out set:
- `gt_heldout_vs_n96_slim.npz` = 336 held-out frames, **0 overlap with n96** (dropped the 64).
- `gt_contig96_300_395_slim.npz` = a **contiguous-96 control** (frames 300..395, disjoint from n96) to separate *window-length* from *which-frames* from *sample-count*.
- (slim caches are rebuildable scratch; lstars/margins/gt_poses only — see "Rebuild" footer.)

Same-code anchor reproduction: running tool 1 on `gt_n96.npz` with the current code reproduces the landed anchors **exactly** (pose_eff_dim 4.08, manifold_eff_dim 5.964, pose-R² 0.514/0.700, residual_eff_dim 21.1, per-class Fisher mass Road 0.470/Undriv 0.252/MyCar 0.133/Lane 0.084/Movable 0.060). So the comparison below is apples-to-apples on one code version.

---

## HEADLINE VERDICT

**MOSTLY YES — the value-generators are PORTABLE and produce STABLE outputs on held-out data, with ONE clearly-isolated exception.**

- The **scorer-geometry family** — per-class Fisher-mass distribution, codim-1 thinness (Gini/mass-concentration), nullspace fraction, per-class **topological rate floor**, **PH⁰ persistence-dimension**, and **R-survival %** — **HOLDS on data 100% disjoint from the anchor**, most within ±5%, several bit-for-bit. These are **domain + frozen-scorer fundamental**. The contest-derived values are the generator's per-instance output, NOT magic constants. **The framework generalizes.**
- The **ONE genuinely instance-conditioned family** is the **ego-pose ↔ Fisher-manifold-motion regression** (manifold eff-dim, pose-R² linear/quadratic, per-class pose-R², off-pose residual fraction). It scales strongly with **trajectory span/diversity**, NOT with the frozen scorer. The n96 anchor's rosy values (pose explains **51%/70%** of the manifold; eff-dim 5.96) came from an **unusually smooth 96-frame opening segment** — a *different* contiguous-96 window of the **same clip** already gives eff-dim 14.6 / pose-R² 0.28, and the full clip gives eff-dim 26 / pose-R² 0.17. The **generator is still portable** (it runs and produces consistently-**ordered** output — Road > Lane pose-explanation in **every** run), but its scalar **R² value is window-conditioned and must not be cited as a domain constant.**

Conservative-direction consequence (load-bearing for v2): over held-out/full-clip data the **off-pose residual is ~83% of the Fisher-manifold variance** (anchor said 49%). The witness's irreducible **learned** job (lane-survival + movables) is **bigger** than the n96 number implied; "pose explains the manifold / pose is a near-free dual-use d_seg lever" is **overstated** by the opening-segment anchor.

---

## TOOL 1 — `north_star_fisher_manifold_dim.py` (Fisher geometry + pose generative factors)

anchor = n96 (frames 0..95) · held-out = heldout336 (disjoint) · controls = contig96@300 (disjoint, same length) + n600 (full clip).

| quantity | n96 ANCHOR | contig96@300 | **heldout336** | n600 full | held-out vs anchor | verdict |
|---|---:|---:|---:|---:|---:|---|
| **A** Gini of Fisher mass | 0.798 | 0.798 | **0.806** | 0.805 | +1.0% | **HOLDS** |
| **A** frac px holding 90% mass | 0.325 | 0.324 | **0.293** | 0.298 | −9.8% | **HOLDS** |
| **A** frac px holding 99% mass | 0.824 | 0.832 | **0.819** | 0.820 | −0.5% | **HOLDS** |
| **B** manifold area frac (¬nullspace) | 0.139 | 0.137 | **0.148** | 0.146 | +6.5% | **HOLDS** |
| **B** manifold mass frac | 0.824 | 0.818 | **0.837** | 0.835 | +1.6% | **HOLDS** |
| **C** Fisher mass — Road | 0.470 | 0.483 | **0.476** | 0.475 | +1.3% | **HOLDS** |
| **C** Fisher mass — Lane | 0.0843 | 0.0921 | **0.0817** | 0.0821 | −3.1% | **HOLDS** |
| **C** Fisher mass — Undrivable | 0.252 | 0.232 | **0.239** | 0.241 | −5.2% | **HOLDS** |
| **C** Fisher mass — Movable | 0.0602 | 0.0473 | **0.0567** | 0.0572 | −5.8% | **HOLDS** |
| **C** Fisher mass — MyCar | 0.133 | 0.145 | **0.147** | 0.145 | +10.4% | **HOLDS** |
| **C** mass-per-area — Lane (highest) | 14.32 | 14.97 | **13.94** | 14.02 | −2.6% | **HOLDS** |
| **C** mass-per-area — Road | 2.05 | 2.13 | **2.04** | 2.04 | −0.2% | **HOLDS** |
| **D** pose eff-dim (of 6; ceiling 4) | 4.08 | 3.66 | **4.52** | 4.69 | +10.8% | **HOLDS** |
| **D** Road pose-R² > Lane pose-R² (ordering) | ✔ | ✔ | **✔** | ✔ | — | **HOLDS (qual.)** |
| **D** manifold eff-dim | 5.96 | 14.65 | **25.26** | 26.33 | **+324%** | **SHIFTS** |
| **D** manifold dim @90% var | 28 | 32 | **75** | 90 | +168% | **SHIFTS** |
| **D** pose **linear** R² on manifold | 0.514 | 0.277 | **0.156** | 0.167 | **−70%** | **SHIFTS** |
| **D** pose **quadratic** R² on manifold | 0.700 | 0.488 | **0.288** | 0.301 | **−59%** | **SHIFTS** |
| **D** off-pose residual var frac | 0.486 | 0.723 | **0.844** | 0.833 | +74% | **SHIFTS** |
| **D** off-pose residual eff-dim | 21.1 | 19.4 | **31.8** | 33.8 | +51% | **SHIFTS** |
| **D** Road kappa pose-R² | 0.527 | 0.273 | **0.187** | 0.195 | −65% | **SHIFTS** |
| **D** Lane kappa pose-R² | 0.363 | 0.196 | **0.086** | 0.099 | −76% | **SHIFTS** |

**Read:** A/B/C all HOLD (scorer geometry is domain-fundamental). The whole **D pose-regression block SHIFTS**, and the contig96@300 control proves it is a **window/diversity** effect, not a frozen-scorer property — two equal-length contiguous windows of the *same clip* disagree by 2.5× on eff-dim. The shift is **monotone in span** (n96 < contig96 < heldout336 ≲ n600), confirming "pose-R² is a trajectory-window statistic." pose eff-dim (≈ rank-(K−1)=4 ceiling) and the Road≫Lane pose ordering survive.

---

## TOOL 2 — `birth_death_persistence_dseg.py` (persistent-homology rate floor / R-survival)

anchor = n200 (n_gt=100) · held-out = n400 (n_gt=200), **0 overlap**. Witness stage `l7` (fixed in both).

### GT-only quantities (the actual held-out test) — items 1/3/4/5

| quantity (per class) | anchor n200 | **held-out n400** | verdict |
|---|---:|---:|---|
| **rate floor** #feat>1.0 — Lane | 6 | **6** | **HOLDS (exact)** |
| **rate floor** #feat>0.5 — Lane | 23 | **23** | **HOLDS (exact)** |
| #feat>1.0 — Road / Undriv / MyCar | 32 / 12 / 10 | **33 / 12 / 10** | **HOLDS** |
| #feat(all)/frame — Lane / Road | 166 / 787 | **165 / 784** | **HOLDS (≤1%)** |
| persistence **Zipf** exp (all classes) | −1.54…−2.43 | **identical to 2 dp** | **HOLDS (exact)** |
| **PH⁰ dim — Lane (highest)** | 0.83 | **0.84** | **HOLDS** |
| PH⁰ dim — Road / Undriv / Movable / MyCar | 0.29 / 0.26 / 0.36 / 0.27 | **0.29 / 0.26 / 0.36 / 0.27** | **HOLDS (exact)** |
| **R-survival** feat — Lane | 0.851 | **0.853** | **HOLDS (exact)** |
| **R-survival** pers — Lane | 0.900 | **0.901** | **HOLDS (exact)** |
| **R-survival** dash — ALL classes | 1.000 | **1.000** | **HOLDS (exact)** |
| R-survival feat — Road | 0.846 | **0.811** | **HOLDS** (−4%) |
| vineyard mean-life — Lane | 1.57 | **1.78** | HOLDS* |
| vineyard mean-life — Movable (highest) | 3.60 | **4.60** | SHIFTS* (window) |
| vineyard frac-coherent(≥3) — Lane | 0.124 | **0.137** | HOLDS* |

\* vineyard is the only tool-2 quantity with measurable drift, and it is a **measurement-window artifact**: held-out ran n_gt=200 vs anchor n_gt=100, and longer observation windows let tracks live longer (drift is uniformly in the "more coherent" direction). The **ordering is preserved** (Movable most coherent, Undrivable/MyCar most ephemeral, Lane mostly ephemeral). Not a domain shift.

### Item 2 (error∝1/persistence, annulus flip-rate) — **WITNESS-BOUND, NOT a held-out test**

Item 2 reads the fixed trained witness (`maps_l7.npz` + `_gt_argmax_subset.npy` + witness `gt_margin`), **not** `--gt-cache`, so it is **byte-identical** across the two runs by construction:
- Lane by-size flip ratio **4.97×** (≈ the "~5-6×" anchor); Road 64.25×; Movable 36.07×.
- annulus: lowest-GT-margin-bin flip-rate **0.764**, ~0.000 every higher bin (ratio = ∞).

These quantify the **one trained witness** (overfit to the contest clip by construction); there is **no held-out witness**, so generalization of these is **untestable here** — they are the artifact we are trying to generalize *to*, not a domain+scorer generator. The **structural shape** they assert (flips concentrate on small/shallow features in the sub-0.10-margin annulus) is *consistent* with the GT-only R-survival + PH⁰ results that DO hold on held-out.

---

## What this means for the witness design (advisory)

**Design-relevant generators all generalize:**
1. **Lane is the binding residual** — highest PH⁰ dim (0.84), multi-scale, ~6 persistent + long tail dashes/frame. **HOLDS on held-out.**
2. **The d_seg residual is R-recoverable, not R-destroyed** — Lane feat-survival 85%, **dash-survival 100%** across all classes. **HOLDS on held-out.**
3. **Per-class Fisher-mass priority** Road ≫ Undriv ≫ MyCar ≫ Lane ≫ Movable, with Lane highest mass-per-area. **HOLDS on held-out.**

**One optimism to temper (v2-relevant):** the pose↔manifold regression that fed "ego-pose is a near-free dual-use d_seg lever" is **window-conditioned and overstated by n96.** Over held-out/full-clip data ego-pose linearly explains only **~16%** (quadratic ~30%) of the Fisher-manifold motion, and the **off-pose residual is ~83%** of it — so the per-class warp lever is real (Road>Lane ordering holds) but the **achievable warp fraction is smaller** and the **irreducible learned residual (lane-survival + movables) is larger** than the opening-segment anchor implied. The v2 framing "trained INR shrinks to ONLY the lane-survival residual + small movables" is directionally right; held-out says that residual is the **dominant** share of the manifold, so do not under-budget it.

---

## Artifacts (uncommitted; rebuildable)

- tool-1 JSONs: `experiments/results/genprobe_20260630T175321Z/fisher_{n96,contig96,heldout336,n600}/results.json`
- tool-2 reports+JSON: `experiments/results/genprobe_20260630T175321Z/bd_{anchor_n200,heldout_n400}.md` (+ `.<stem>_ckpt/results.json`)
- Rebuild slim held-out caches (lstars/margins/gt_poses only, no RGB):
  - `gt_heldout_vs_n96_slim.npz` = `gt_heldout_n400.npz` frames whose n600 index ∉ {0..95} (336 frames).
  - `gt_contig96_300_395_slim.npz` = `gt_n600.npz` frames [300:396].
  - (deleted after this run to free ~1 GB; deterministic to regenerate from the GT caches.)

`[macOS-numpy advisory · NON-PROMOTABLE]` · pointer 0.19110 UNMOVED.
