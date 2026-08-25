<!-- # COUNCIL_TIER_FRONTMATTER_WAIVED:lens synthesis, not a convened deliberation. This memo self-declares 'Type: DEEP RESEARCH + SYNTHESIS memo' authored by a single synthesis subagent; it GATHERS council lenses on frozen-space exploitation from prior work rather than convening a council. There is no seating, no quorum, no vote and no verdict in the body, so tier and verdict fields could only be invented. Catalog #300 backfill 2026-08-25. -->
# FROZEN CONTEST SPACE — the council-lens synthesis + the Layer-1 carrier reframe (2026-06-12)

**Author:** frozen-contest-space council-lens synthesis subagent (`frozen_contest_space_council_lenses_20260612`).
**Type:** DEEP RESEARCH + SYNTHESIS memo. NO production code, NO GPU, NO dispatch. Touches no running daemon,
no basin out-dir, no `src/tac/torch_vehicle/**`, no `src/tac/substrates/cool_chic/**`.
**Evidence grade:** `[analysis]` — every quantified claim is tagged **[MEASURED:<memo>]** (an exact number
recomputed-from-components on the frozen scorers, GT via `frame_utils.yuv420_to_rgb`, NEVER MPS) or
**[DERIVED:<basis>]** (a closed-form bound). NO score is claimed; the means/ends firewall holds — this is a
MEANS (a lens synthesis) toward the END (a lower exact score) and moves no row.
**Frontier (pointer, NOT hardcoded):** `.omx/state/canonical_frontier_pointer.json` → contest-CPU
**0.19109982** (177,169 B, sha `b46897267…`, `lane_pr110_payload_entropy_recode`); contest-CUDA **0.20533**
(186,876 B). **Frontier UNMOVED.** Ladder: T_3 = sub-0.15 (the aim), T_1 = sub-0.19 (floor of acceptable),
S_floor ≈ 0.11797 (rate-dominated, the measured-achiever floor).

> **COMPOSES WITH (does not duplicate) the running carrier memo** `layer1_carrier_first_principles_20260612T171912Z.md`:
> that memo evaluates carrier CANDIDATES (HNeRV / Cool-Chic / VQ-NeRV / SIREN / Ballé / witness) and derives
> the **rate-floor-invariance crux** (§D). THIS memo gathers the COUNCIL LENSES on frozen-space exploitation
> — the eureka moments, perspectives, and measured-oracle primitives — and binds them into the meta-strategy
> that should FRAME the carrier choice. Where the carrier memo says "the witness/score-quotient carrier is the
> class-shift candidate, gated on cheap-null-identification + round-trip," THIS memo supplies (a) the complete
> roster of lenses that built that conclusion, (b) the quantified scorer-conditional MDL vs HNeRV's 177KB, and
> (c) the ranked, $0-first exploitation strategy with which ORPHANED lenses to reactivate.

---

## A. THE META-INSIGHT, FORMALIZED — why "complete frozen space + small information + well-defined goal" changes the PROBLEM TYPE

### A.1 It is not learning. It is EXACT single-point optimization against a fully-known oracle.

The operator's framing — *complete frozen space, relatively small information content, well-defined goal* — is
not a slogan; it is a precise statement that the contest is **NOT a machine-learning problem** (no
generalization, no held-out data, no distribution to model) but an **exact single-point optimization /
witness-synthesis problem against a fully-known, deterministic oracle.** Formally
[DERIVED: `sota_plus_original_inventions_20260610`, `validation_score_program_compiler_quotient_theory_20260606`]:

```
A* = argmin_A  S(A)      over the FROZEN tuple { modules.py, evaluate.py, 0.mkv }
S(A) = 100·d_seg(inflate(A)) + sqrt(10·d_pose(inflate(A))) + 25·|A|/N      N = 37,545,489
```

Every term on the right is **known in advance and never changes**: the SegNet weights (`segnet.safetensors`),
the PoseNet weights (FastViT-T12), the exact 1200-frame video `0.mkv`, the resize kernels, the YUV6 basis, the
scoring law. There is no test set; the **target IS the answer key.** This is the deepest structural fact and
it is what the CLAUDE.md "Evaluator-Equivalent Witness Compiler Paradigm" encodes: the winning representation
is the *shortest legal `archive.zip` whose `inflate.sh` output is a witness inside the same frozen evaluator
cells as the source* — RGB fidelity is non-authority unless it causally improves d_seg / d_pose / bytes.

The consequence the council keeps re-deriving (and that re-frames Layer-1): **because the oracle is fully
known, we can MEASURE its geometry once and SOLVE against it, instead of LEARNING a surrogate of it.** No
competitor has done this — no published codec has the contest's frozen S/P weights *and* has differentiated
them. Our private oracle model (the margin field, the Jacobian atlas, the resize null space, the closed-form
water level λ\* = 25/N = 6.66e-7 score/byte) is the side information that turns generic SOTA into a winner
[DERIVED: `sota_plus_original_inventions_20260610` "the frame"].

### A.2 The scored functional factors through a NON-SMOOTH, LOW-DIMENSIONAL QUOTIENT

[MEASURED: `council_grand_symposium_path_to_sub015_20260610` §1, T4 grand council "pattern-behind-patterns"]
The decisive structural fact — the one invariant that explains why *every path walls the same way*:

> `evaluate.py ∘ modules.py` maps each 874×1164×3 frame to exactly TWO scored objects — the SegNet 5-class
> **argmax PARTITION** of frame1 (a *combinatorial set functional*: `d_seg` = per-pixel argmax-flip RATE,
> piecewise-constant in pixels, gradient ZERO almost everywhere with deltas only at the argmax boundary) and
> the PoseNet **6-of-12 output** of both frames (a *smooth low-dim regression*, globally pooled before the √).

So the carrier's true job is to **describe a point inside an EQUIVALENCE CLASS** (Tishby's information
bottleneck made literal — `council_grand_symposium` Tishby dissent: *"evaluate.py is an information bottleneck:
it squeezes the 874×1164×3 frame through a low-dim relevant variable"*):

```
Q = { F : argmax SegNet(F_last) = L*  pixelwise }   ∩   { F : ‖PoseNet(F_pair)[:6] − p*‖² ≤ τ_pose }
```

- The SegNet term induces **genuine discrete cells** — argmax-constant polytopes. `d_seg` is invariant under
  ANY perturbation that does not cross a SegNet decision boundary.
- The PoseNet term is NOT a fiber (correction in `validation_score_program_compiler_quotient_theory_20260606`
  §CORRECTIONS-1): its level sets are MSE ellipsoids, not equivalence classes. But the **local null space is
  large** (~6 constrained directions per pair, the rest pose-invisible to first order).

This is the **patterns-behind-patterns**: we have been *solving a combinatorial set-functional + a coupled
dual-fidelity frame problem with smooth high-dimensional generative tools.* The gradient that moves d_seg
lives in the **SegNet LOGIT/margin space, not RGB-recon space** — and that single re-framing re-ranks every
lever (distortion-before-rate; margin/logit-space loss over recon-MSE).

### A.3 The "small information space," QUANTIFIED — scorer-conditional MDL vs HNeRV's 177 KB

The operator's "relatively small information content" is now a measured number, and it is the crux of the whole
strategy. Three independent quantifications, from coarsest to tightest:

**(1) The scored degrees of freedom (the upper-bound on what MUST be described).** The score reads only
`600 × (5-class argmax partition of 384×512) + 600 × 6 pose scalars`. Everything else — every textured pixel,
every chroma gradient, the entire pose-null subspace, the entire frame0 except its pose contribution — is
**null space** [MEASURED: `evaluator_invisibility_basis_landed_20260610`]:
- **22.70%** of EVERY camera channel is **TIER-1 CERTIFIED invisible to BOTH scorer heads** (the resize
  zero-weight pixels; residual == 0.0 EXACTLY, amplitude-unlimited up to uint8 clip).
- **80.67%** of camera-pixel directions are in the full resize null space.
- The **ENTIRE frame0** is SegNet-invisible (SegNet slices `x[:,-1,...]`).

So the scorer-conditional information is a *vastly coarser quotient* than the pixel MDL `K(X)` — the carrier
never needs to faithfully render the ~95% of pixels the argmax ignores or the pose-null directions.

**(2) The measured per-object floor (the binding decomposition).** [MEASURED:
`information_theoretic_floor_T_floor_20260610` §5, recomputed-from-components]

| term | value at frontier | share | classification |
|---|---:|---:|---|
| **rate** `25·B/N` | **0.1180** | **61.7%** | BINDING (recoverable only by a smaller achiever) |
| **seg** `100·d_seg` | 0.0560 | 29.3% | RECOVERABLE in principle (→0 with a sharper amortizer) |
| **pose** `√(10·d_pose)` | 0.0172 | 9.0% | MOSTLY RECOVERABLE (concave; low-dim trajectory) |

- **Pose output entropy is ~1.5 KB.** [MEASURED: `information_theoretic_floor_T_floor` P6 RESOLVED;
  `GOAL_standing_v3`] The 600×6 pose trajectory, temporal-delta-coded at the frontier operating point =
  **1,557 B** (0.88% of S_floor). The scored pose is a smooth ego-motion curve — KB-scale, not MB-scale.
- **Seg partition, stored DIRECTLY, LOSES** [MEASURED: `boundary_math_seg_core_20260610` +
  `information_theoretic_floor_T_floor` §2]: the SegNet argmax partition is reproducible *exactly* (d_seg = 0
  by construction, bit-exact, all frames) at **~896 B/frame under an LZMA-over-labels baseline → 524.8 KB for
  600 frames = rate 0.169.** The optimal temporal-context coder tightens this to **253,413 B = rate 0.169**,
  still **ABOVE the amortized decoder's 0.118.** This is the council's hardest-won correction
  (`council_grand_symposium` Assumption-Adversary: *"the standalone-storage RATE-WIN is CARGO-CULTED and
  falsified"*): **store-the-partition-directly is rate-FALSE; amortization wins.** The partition's low
  PER-REGION entropy (35 regions/frame, 0.687% boundary, 3 KB labels) does NOT imply low TOTAL entropy
  (21,304 regions over 600 temporally-varying frames; the boundary MOTION is the cost).

**(3) The free-decoder-CONDITIONAL intrinsic dimension (the tightest DERIVED band).** [DERIVED:
`smaller_learned_basis_deep_math_20260610` §3] Given an arbitrarily sophisticated FREE inflate algorithm
(≤30 min, no large artifacts), how few CHARGED bits MUST encode this video's scored content?

```
B_min,conditional ∈ [ pose 1,557 + amortized-seg-core ~20,000 + appearance ~3,000 ,
                       pose 1,557 + amortized-seg-core ~55,000 + appearance ~8,000 ]
                  ≈  [ ~24.6 KB , ~64.6 KB ]
S_floor,conditional = 25·B_min/N ∈ [ 0.0164 , 0.0430 ] + ε_distortion
```

**This is the answer to "is the true object KB-scale?" — YES.** The scorer-conditional intrinsic dimension is
**~25–65 KB**, which is **2.7×–7× below the 177 KB frontier** and far below the leaderboard's ~178 KB neural
decoders. The gap between this conditional floor and the 177 KB achiever is the **over-parameterization of the
memorized HNeRV point** — capacity the score-domain floor would shed but post-hoc compression CANNOT touch
(proven jointly-entangled, `frontier_pointer_move_ledger` #71). The honest caveat (NO-FAKE): this is a
*Kolmogorov-uncomputable lower bound with no proof of reachability* — it is the mathematical *license* for the
class-shift, the falsifiable prediction a funded campaign tests, not a guarantee.

**The one-sentence meta-insight:** *the contest is an exact optimization against a frozen oracle whose scored
output is a low-dimensional non-smooth quotient (a 600-map argmax partition + a 1.5 KB pose trajectory),
whose scorer-conditional MDL is ~25–65 KB — but the field (and we) keep paying ~178 KB because we keep
encoding the full pixel manifold with a smooth amortizer instead of the small quotient object with a
scorer-conditional witness.*

---

## B. THE COMPLETE LENS CATALOGUE — every council eureka on frozen-space exploitation

Each lens: the **insight** · the **frozen-space property it exploits** · the **source** · **status**
(BUILT / PARTIAL / ORPHANED / DESIGN-ONLY / MEASURED-VERDICT) · **measured-or-predicted score impact.**
17 lenses, grouped by what they exploit. Every number is cited; none invented.

### Group I — THE STRUCTURE OF THE QUOTIENT (what the scorer actually sees)

**L1 — Evaluator-Equivalent Witness Compiler (the master frame).**
*Insight:* the archive is a PROGRAM; `evaluate.py` is the cost; optimize the shortest program whose
`inflate.sh` output lands in the frozen evaluator cell. RGB fidelity is non-authority.
*Exploits:* the frozen, deterministic, fully-known oracle.
*Source:* CLAUDE.md "Evaluator-Equivalent Witness Compiler Paradigm" · `validation_score_program_compiler_quotient_theory_20260606` (validated ~80% already-canonical, 10% correction, 10% novel).
*Status:* BUILT as doctrine; the end-to-end compiler is DESIGN-ONLY (lever A).
*Impact:* the organizing principle; not itself a row.

**L2 — Argmax-as-discrete-cell / the polytope quotient (the seg structure).**
*Insight:* d_seg only sees WHICH SegNet argmax polytope you are in; the frame can be wrong in 95% of pixels and
score identically. The cell is `{F : (J_{L*,p} − J_{c,p})·δ ≥ −m_{p,c} ∀ p, c}` — a system of linear
inequalities (the margin-polytope).
*Exploits:* SegNet argmax-flip is a piecewise-constant set functional, gradient-zero off the boundary.
*Source:* `boundary_math_seg_core_20260610` (`margin_polytope.py`, `partition.py`) · `closed_spec_boundary_math_system_of_equations_20260610` §4.
*Status:* BUILT (`src/tac/boundary_math/`, 6 modules + 22 tests; d_seg=0 carrier reproducible bit-exact).
*Impact:* [MEASURED] the seg partition is **fully solvable in closed form** (d_seg=0, no search/training) —
but standalone storage costs 524.8 KB (rate 0.169 > 0.118). VERDICT: paradigm proven, baseline coder too fat;
**DEFER-pending a margin-aware boundary coder** (L9) OR compose as residual-repair on an amortized base.

**L3 — Pose-as-MSE-ellipsoid / 6-of-12 scored dims (the pose structure).**
*Insight:* d_pose is a continuous MSE on 6 of 12 pose dims, globally pooled BEFORE the √ → the term is
NONSEPARABLE (a pose error on any pair trades 1:1 with any other; the marginal `5/√(10·d_pose)` GROWS as the
pool shrinks). Dims 6–11 are certified null space.
*Exploits:* the concave-√ + global-pool + half-the-head-unscored structure.
*Source:* `cross_pair_waterfilled_corrector_20260610` · `information_theoretic_floor_T_floor` §3 · `modules.py:84`.
*Status:* BUILT (the cross-pair pose waterfiller + 20 tests).
*Impact:* [MEASURED] the concave-√ crossover with seg is at d_pose = 2.5e-3; frontier sits ~85× below
(2.94e-5), deep in the cheap region — pose is 9.0% of the score and mostly recoverable.

**L4 — Frame-0/frame-1 incidence asymmetry (the free pose channel).**
*Insight:* SegNet reads ONLY frame1 (`x[:,-1,...]`); PoseNet reads BOTH frames. ⟹ **frame0 is entirely
SegNet-invisible** — a free pose-only carrier surface; ~600/1200 frames are seg-free.
*Exploits:* the verified scorer-head asymmetry.
*Source:* `evaluator_response_atlas_design_20260609` ("official asymmetry") · `validation_..._quotient_theory` #6 · `evaluator_invisibility_basis_landed` (frame0 100% SegNet-invisible).
*Status:* BUILT (servo static contract `frame_incidence`; the FEC6 K=16 frame-0 selector is live in the frontier).
*Impact:* [MEASURED] the frontier FEC6 selector is **per-pair pose-optimal over its K=16 palette already**
(0/42 improvable pairs, `cross_pair_waterfilled_corrector`) — the coarse frame0 lever is SATURATED; a richer
(Jacobian sub-pixel) grammar is the reactivation.

### Group II — THE NULL SPACE (the free bytes / free perturbations — certified invisibility)

**L5 — Certified invisibility basis (the score-quotient null space, EXACT).**
*Insight:* the joint scorer map shares a fixed bilinear resize as its first step; its null space is a
closed-form set of camera-pixel perturbations that are **bit-identical to the scorer** (residual == 0.0
EXACTLY, amplitude-unlimited).
*Exploits:* the resize is a fixed low-rank linear projection `R`; `ker(R)` is certified-free by construction.
*Source:* `evaluator_invisibility_basis_landed_20260610` (task #47).
*Status:* BUILT (TIER-1 certified-exact + TIER-2 measured; 41 tests; `tools/build_evaluator_invisibility_basis.py`; wired into the #46 waterfiller as a certified-zero-distortion free-byte action + into `tac.null_space_exploiter`).
*Impact:* [MEASURED/DERIVED] **22.70% of every channel certified free, 80.67% full resize-null, frame0 100%
SegNet-free.** This converts the waterfill's free-byte branch from "lossless recode assumed zero-cost" to
"certified zero-cost." The highest-value use is the #46 waterfiller `null_basis` recode/drop action.

**L6 — Byte-space null-space exploiter (the archive-byte invisibility basis).**
*Insight:* the per-pair master-gradient constraint matrix (scorer-axis sensitivities × byte offsets) has an
orthonormal null basis of archive-byte perturbations with small first-order score response.
*Exploits:* the measured per-byte score-sensitivity ledger.
*Source:* `evaluator_inverse_orphan_inventory_20260609` §Task-47 (`src/tac/null_space_exploiter/`).
*Status:* BUILT + CONSUMED (wired into `src/tac/unified_action.py`).
*Impact:* this is HOW PR110 and the CPU frontier "stumbled into free bytes by trial" — now formalized.

### Group III — THE MEASURED ORACLE GEOMETRY (tomography → exact solve, not sweep)

**L7 — The Evaluator Response Atlas (precomputed margin field + Jacobian → exact geometry).**
*Insight:* instead of finite-differencing billions of pixel coordinates, query ALL dimensions via
gradients/JVP/VJP/vmap + structured bases, building a per-pair tomography of `F_i(δ) = (d_seg, d_pose, S)`:
the SegNet margin-gradient map, the PoseNet `JᵀJ` saliency, active subspaces (Morris/Sobol), null directions.
*Exploits:* the oracle is differentiable and fixed → its full response surface can be mapped once and reused.
*Source:* `evaluator_response_atlas_design_20260609` (the engine design) · `evaluator_response_atlas_engine_landed` · `scorer_spectral_atlas_v2`.
*Status:* BUILT (600-pair atlas; `evaluator_response_atlas.jsonl`; consumed by the dispatch-order wiring L11).
*Impact:* [MEASURED] the atlas IS the private oracle model no competitor has; it supplies the per-pair budgets
(top-budget pairs 442/426/577/437-440; fragile clusters 133/177-178/517-519) that drive every allocator.

**L8 — Scorer-conditional sensitivity = the non-arbitrariness principle (waterfilling at EVERY dimension).**
*Insight:* an allocation is OPTIMAL iff every unit of cost (bit, byte, gradient step, even inter-agent
context) is placed where its marginal `∂S/∂unit` is most negative. **Uniform anything is provably
suboptimal** — uniform fp16 over-pays the 95% the evaluator ignores; uniform int8 starves the sensitive 5%.
*Exploits:* SegNet sensitivity is sparse+spatial+frame1-only (~4.8% fragile); PoseNet is dense+temporal+
Y-dominant; chroma + unscored dims + robust regions buy ~zero.
*Source:* `evaluator_optimal_adaptive_waterfilling_non_arbitrariness_synthesis_20260609` (the operator's
rapid-fire Q1–Q6) · CLAUDE.md "Meta-Lagrangian/Pareto solver."
*Status:* BUILT as principle + arsenal (per-channel mixed-bit codec, L21–L32); PARTIAL in the active path
(the R3 export pinned a single `int8_mixed` codec — the arbitrary choice).
*Impact:* re-frames the codec from "pick a format" to "solve `∂S/∂precision` per tensor"; the door to a
sensitivity-driven smaller decoder.

### Group IV — THE EXACT SOLVE (KKT / waterfilling / OT / Dykstra — solve, don't search)

**L9 — Margin-Weighted Contour Coder + STC/UNIWARD (the inverse-steganalysis seg coder).**
*Insight:* d_seg is per-pixel argmax-flip RATE of a frozen EfficientNet-B2 — it **IS inverse steganalysis on
the partition** (Yousfi: *"the whole field clusters at 0.19 because everyone optimizes a smooth recon
surrogate of a non-smooth detector functional"*). Code contour bits ONLY along fragile small-margin boundary
segments; condition the entropy model + the drop decision on the measured margin field (UNIWARD-style change
cost `cost(p)=m(p)/‖g_p‖`); embed flip-corrections with Filler-STC approaching the 1.27 B/flip water level.
*Exploits:* 91% of boundary is large-margin (certain, omittable); only the margin→0 set is the entire signal.
*Source:* `sota_plus_original_inventions_20260610` AREA(a) (MWCC) · `boundary_math_seg_core` (the margin
budget) · `council_per_substrate_symposium_stc_clean_source_20260517` + the STC-clean-source DEFER.
*Status:* DESIGN-ONLY (MWCC); the pieces exist (`tac.codec.syndrome_trellis_codec`, `tac.uniward_delta`,
`contour_codec.py` defers to STC by design).
*Impact:* [MEASURED bound] needs ≲170–250 B/frame (3.6–5.3× under the LZMA baseline) to cross the water level.
The STC-clean-source DEFER measured uniform-cost STC at 2.4–2.6× brotli; **margin-weighting IS the named
reactivation cost-map** — NOT a standalone rate win (partition-direct loses), only a residual-repair coder on
a repairable base.

**L10 — Closed-form λ\*-equalizing allocator + pose-as-global-OT-budget (the KKT solve, not a sweep).**
*Insight:* with the EXACT measured marginals, allocation is a SOLVE: admit a seg flip-repair iff its cost <
`(100/N)/λ\* = 1.27 B/flip` (closed-form); distribute the pose-byte budget across 600 pairs via **Sinkhorn**
(the R(D)↔optimal-transport identity) because pose is pooled-before-√ = a literal global OT budget.
*Exploits:* the analytic water level λ\* = 25/N = 6.66e-7 score/byte + the global-pool pose structure.
*Source:* `sota_plus_original_inventions_20260610` AREA(b) · `cross_pair_waterfilled_corrector_20260610`
(the built pose waterfiller) · `lf_payload_rate_distortion.py` (THE LAW, #46) · `joint_p18_p19_waterfill.py`.
*Status:* BUILT (the λ\* allocator + cross-pair pose waterfiller + 20 tests; the LF-payload reverse-waterfill
is CONSUMED); the pose-OT Sinkhorn layer is DESIGN-ONLY (needs the carrier to define per-pair pose cost).
*Impact:* [MEASURED] on the FRONTIER base the seg-correction input is **EMPTY** (every component under-water:
1.525 B/flip position floor > 1.27 break-even; 95% scattered single-pixel flips) and the pose lever is
**SATURATED** (0/42 improvable). The allocator is a solve on the RIGHT base (a contiguous-residual generator
base), NOT the frontier — this is the lens's sharpest output: *the allocator is ready; it needs a repairable
base.*

**L11 — Atlas → waterfiller DISPATCH ORDER (cross-video temporal targeting).**
*Insight:* the atlas ranks WHICH PAIRS across the 600-pair video carry the most joint-safe budget; coarsen the
high-budget temporal segments FIRST, protect the fragile clusters LAST. The temporal mask pays run-length rent
exactly like the spatial cone mask.
*Exploits:* the video's temporal structure (contiguous low-sensitivity runs) + the atlas per-pair budget.
*Source:* `atlas_waterfiller_dispatch_order_landed_20260609` (#36→#46 wiring).
*Status:* BUILT (gated, default-OFF, backward-compatible; 86 tests; the composed temporal×spatial rung
predicted for Branch-B round-3).
*Impact:* [MEASURED known-optimum] a temporal-segment quantize (437-440, 0.667% of video) yields **1.41×
higher value-per-byte** than whole-video — the budget discount shrinks the distortion weight while a 16-byte
run-length mask is trivial rent.

**L12 — Legal-frame FEASIBILITY via Dykstra alternating projections (constraint satisfaction, not generation).**
*Insight:* finding the cheapest frame is a FEASIBILITY problem — project onto `{margin-cell} ∩ {pose-tube} ∩
{cheap-encoding subspace}` via Dykstra alternating projections, re-linearizing the nonlinear S/P Jacobians
each outer loop. Feasibility ≠ generation: moving an already-correct frame the MINIMUM amount holds pose where
a from-scratch generator (palette, INR) cannot.
*Exploits:* the cell + tube are convex linearizations of the frozen Jacobians; the GT frame is already feasible.
*Source:* `legal_frame_feasibility_dykstra_20260610` (task #73; Dykstra is the inner-council CO-LEAD's tool).
*Status:* BUILT (`src/tac/boundary_math/dykstra_legal_frame.py`, 22 tests + on-real-scorer slow test).
*Impact:* [MEASURED] the deep geometric answer: **the pose tube is NOT the binding constraint** (the frontier
is in it at d_pose 2.4e-5); the binding wall is **the byte cost of a basis that holds pose UNDER
COMPRESSION.** A generic low-rank/sparse basis needs ≥625 KB/pair (pose breaks first below ~400 KB); the
learned HNeRV basis holds 600 frames in 177 KB. **This DERIVES (not assumes) why a learned basis is needed** —
the cheap-feasible set is empty for generic bases; the reactivation is `C = the learned basis manifold`.

### Group V — THE CARRIER / WITNESS (the representation that spends only on the quotient)

**L13 — Score-native generator (the partition witness, CONFIRMED).**
*Insight:* a tiny MLX label-map generator hits the frozen SegNet's 600-argmax partition directly — d_seg
**0.00826 in a 63,802-byte blob, 2.54× smaller than the frontier seg-share.** It scores LOGITS, not appearance.
*Exploits:* the seg quotient is low-dimensional and directly addressable.
*Source:* `lever_b_score_native_argmax_smoke_verdict_20260610` (lever B, CONFIRMED PROCEED-TO-CAMPAIGN) ·
`score_native_first_candidate_20260610` (the byte-closed candidate).
*Status:* BUILT + BYTE-CLOSED ([MEASURED] archive.zip **72,217 B, sha `7dc512b5…`, scorer-free inflate.py,
−59% bytes vs frontier, lossless parity proven**). The carrier seg+pose blob = 70,452 B vs 177,169 (−60%).
*Impact:* **THE RATE CLASS SHIFT IS REAL AND BYTE-CLOSED.** [MEASURED] BUT the palette frame1 is pose-BLIND →
**d_pose = 12.66 (√ term = 11.25 alone) → advisory S 13.58, does NOT beat frontier.** Pose is the single live
blocker. The −59% rate is the headroom that makes closing pose worth it.

**L14 — PoseNet-TUBE-Native Carrier (PTNC) + pose-FiLM side-info (the pose witness / Wyner-Ziv).**
*Insight:* an amortized luma-motion INR trained against the MEASURED PoseNet pixel-Jacobian atlas, spending
bits ONLY in the directions PoseNet's 6 scored dims are sensitive to (the IDSE loss `L=‖J_P·(carrier−GT)‖²`
on the 6 scored dims) — free to be wrong everywhere the Jacobian is zero. Equivalently, store the 6 GT pose
scalars/pair (~1.5 KB) and FiLM-condition the decoder on them (Wyner-Ziv side-info): the decoder is *told* the
pose and modulates frame1 features so the PoseNet readout matches GT.
*Exploits:* the pose null space (most of luma is pose-invisible) + frame0-SegNet-invisibility + the
pooled-pose OT structure + the measured exact frozen-PoseNet Jacobian (vs USC IDSE's per-image Taylor approx).
*Source:* `sota_plus_original_inventions_20260610` AREA(c) (PTNC, the #1 build) · `score_native_pose_carrier_20260610`
(#57) · `grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517` (Wyner-Ziv lens) ·
the running carrier memo's pose-FiLM disambiguator GO.
*Status:* DESIGN-ONLY (PTNC; `amortized_luma_carrier.py` is the scaffold — a loss-swap + capacity change);
pose-FiLM is a measured-GO disambiguator in the running carrier memo.
*Impact:* [MEASURED/projected] realizes the ~1.5 KB pose floor; the most-validated witness component; the
named reactivation that unblocks L13. Pre-registered kill: d_pose > ~0.1 at < 30 KB amortized.

**L15 — Cross-pair scorer-quotient corrector (the saturated-lever lens).**
*Insight:* a cross-pair global-pool waterfilled corrector tests whether re-allocating per-pair frame0 modes
moves d_pose. On the frontier base it admits ZERO (the FEC6 selector is already per-pair pose-optimal).
*Exploits:* the global-pool pose fungibility.
*Source:* `cross_pair_waterfilled_corrector_20260610` (task #54).
*Status:* BUILT (the SOLVE + 20 tests).
*Impact:* [MEASURED] **the cheap structured-correction levers are EXHAUSTED on the frontier base** (seg #55 +
pose #54 both empty). The eureka: *the frontier's own optimizer already captured the cheap corrections; the
remaining headroom needs a richer grammar or a new (repairable) base* — which is exactly the witness carrier.

### Group VI — THE FLOOR + THE LEGAL/FREE BOUNDARY (the limits of the frozen space)

**L16 — The information-theoretic floor + free-inflate boundary (the limit lens).**
*Insight:* the score floor collapses to `25·B_min/N + ε`, where `B_min = K(evaluator-view | inflate runtime)`
is conditional-Kolmogorov-uncomputable — NO nontrivial proven wall. The free/charged boundary: a
video-INDEPENDENT algorithm in inflate.py is FREE; the coefficients/seed it consumes are CHARGED; baking
video-specific payload into inflate.py is FORBIDDEN (PR#69 houdini eval-refused). A fixed orthonormal basis
cannot reduce an already-near-iid signal's entropy (the decoder weights are MEASURED-iid → 0 compaction
headroom) — so the win is only fused into a score-aware RETRAINED carrier (forward-designed fixed-codebook VQ),
never a post-hoc rotation of frozen weights.
*Exploits:* the rate rule + the energy-compaction theorem + the counting bound.
*Source:* `information_theoretic_floor_T_floor_20260610` · `smaller_learned_basis_deep_math_20260610` ·
`grand_council_optimal_path_to_shannon_floor_20260507` / `..._paradigm_shift_to_shannon_floor_20260430`.
*Status:* MEASURED-VERDICT (derivation; the floor is RATE-dominated, ~0.118 measured-achiever, ~25–65 KB
conditional).
*Impact:* [DERIVED] **sub-0.15 is a DISTORTION threshold at constant bytes** (frontier scores 0.11797 < 0.15
at zero distortion); the rate attack is only required BELOW 0.118. The frozen-bytes rate axis is
lossless-EXHAUSTED (decoder 98.6% iid Shannon, latents per-dim-marginal + cross-pair MI=0).

**L17 — Deterministic scorer exploit registry (the problem-space exploit vocabulary).**
*Insight:* the `contest_exploits/` registry is a catalogue of frozen-space exploit atoms: per-class chroma
anchors, tropical argmax-boundary grammar, stable-orbit packet diet, precomputed inference outputs, pair-index
lookup tables, the A1 VQ specialized inverter.
*Exploits:* the single-video frozen target admits memorized lookup / precomputed-output atoms.
*Source:* `evaluator_inverse_orphan_inventory_20260609` §contest_exploits (19 modules).
*Status:* MIXED — `a1_specialized_inverter` CONSUMED by 4 tools; `per_class_chroma_anchor`,
`tropical_argmax_boundary_grammar`, `decoy_mosaic_residual_basis`, `stable_orbit_packet_diet` are ORPHANED
(no module importer).
*Impact:* these are the atlas engine's canonical atom families; ORPHANED entries are reactivation candidates
(per CLAUDE.md anti-signal-loss).

### Lens count + the convergence

**17 lenses.** They are NOT independent tricks — the T4 grand council's central finding is that they are
**five faces of one invariant** (L2/L3 the quotient structure; L5/L6 the null space; L7/L8 the measured
geometry; L9–L12 the exact solve; L13–L15 the witness carrier; L16/L17 the floor + exploit vocabulary). The
convergence — that a code-validation analysis (`validation_..._quotient_theory`) found the partner-agent's
"shortest legal codeword whose only decoder is evaluate.py" essay was ~80% already-canonical doctrine — is
itself evidence the paradigm is right.

---

## C. BIND THE LENSES TO THE CARRIER QUESTION — the Layer-1 reframe

### C.1 What the lenses collectively say about "carrier"

The operator framed Layer-1 as a carrier choice. The lenses collectively **reframe what a carrier IS for this
frozen space.** A general INR (HNeRV / Cool-Chic) encodes the **full frame manifold** — it renders all ~590k
pixels/frame faithfully, paying (in its near-MDL weights) for the texture the argmax ignores AND the pose-null
directions. The lenses say the OPTIMAL object is not that. It is a **SCORER-CONDITIONAL WITNESS** that
parametrizes ONLY the small scorer-quotient:

```
B_witness = B_base (cheap render, right inside most argmax cells, L13)
          + B_seg-boundary (the margin-polytope witness — L2/L9, conditional-position coded)
          + B_pose-sideinfo (the 6 GT pose scalars FiLM-injected — L3/L4/L14, ~1.5 KB Wyner-Ziv)
          + B_null-fill (≈ minimal — the certified resize-null fill, L5)
```

Every term maps to a lens: L13 = base; L2+L9 = the seg witness; L3+L4+L14 = the pose witness; L5 = the null
fill; L10/L11 = the allocator that distributes bytes among them at λ\*; L7/L8 = the measured geometry the
allocation reads; L12 = the feasibility solver that finds the cheapest member. **The witness IS the lens
catalogue composed into a carrier.**

### C.2 IS the witness/score-quotient floor BELOW HNeRV's 177 KB? — the decisive question

**The honest, three-part verdict** (reconciling this synthesis with the running carrier memo's §D
floor-invariance crux):

**(1) The rate FLOOR is NOT carrier-dependent — but the SLACK is.** [DERIVED:
`smaller_learned_basis_deep_math` §2.1 energy-compaction theorem; running carrier memo §D] `R*_scorer` is a
property of the (video, scorer) pair with a `min` over all carriers. No carrier swap *lowers the floor*. But a
given carrier reaches `R*_scorer + slack`. HNeRV's slack is small (its overfit weights are near-MDL). The
witness's slack opportunity is the ONE place a carrier can approach from below HNeRV: **HNeRV's near-MDL
weights still encode the scorer-INVISIBLE appearance (all 590k pixels, the pose-null); the witness does NOT —
it spends ZERO on the null space (L5) and only ~1.5 KB on pose (L14).**

**(2) The MEASURED evidence says YES, the witness floor is below 177 KB — and it is already PARTIALLY
realized.** This is the strongest claim in this memo, and it rests on measured rows, not derivation:
- The score-native generator (L13) is **byte-closed at 72,217 B — −59% vs 177,169 B** [MEASURED:
  `score_native_first_candidate`]. The rate class shift is REAL, not projected.
- The free-decoder-conditional intrinsic dimension (L16) is **~25–65 KB** [DERIVED:
  `smaller_learned_basis_deep_math` §3] — the scorer-conditional MDL is **2.7×–7× below** the 177 KB frontier.
- The pose-output entropy (L3/L14) is **1,557 B** [MEASURED] — the pose half of the witness is KB-scale.

So the witness floor (B_base ~25–55 KB + pose ~1.5 KB + seg-boundary residual + null-fill ≈ free) plausibly
lands in the **~30–70 KB band → rate 0.020–0.047** — genuinely below HNeRV's 0.118. **The witness route is the
genuine door past the T_1 wall toward sub-0.118 and the ~0.02–0.05 conditional floor.**

**(3) The HONEST risks (NO-FAKE — why this is a research bet, not a slam-dunk).** The lenses also measured the
walls precisely; the door is real but gated on three measured-or-open risks:
- **The pose blocker (L13→L14).** The −59% byte win is REAL but the witness is pose-BLIND today (d_pose 12.66).
  The entire class shift hinges on L14 (PTNC / pose-FiLM) closing pose at ~1.5 KB. [MEASURED] the pose tube IS
  reachable (L12 — the frontier is in it); the question is whether a SMALL carrier holds it under compression.
  L12's measured answer: a generic basis breaks pose below ~400 KB/pair, but a *learned/Jacobian-aligned*
  basis is the named reactivation. **This is the single decisive open measurement.**
- **The seg-boundary round-trip + cheap-null-identification (L2/L9).** [MEASURED] standalone partition storage
  LOSES (524.8 KB > 177 KB); the witness must compose the seg witness as a *residual-repair coder on the
  amortized base* (Daubechies' dissent: *"I will not endorse a standalone contour carrier"*), AND the boundary
  correction must survive the bicubic↑/bilinear↓/uint8 round-trip, AND the margin-weighted STC must beat the
  1.27 B/flip water level (the STC-DEFER's named bar). On the frontier base the seg-correction allocator is
  EMPTY (L10/L15); it becomes non-empty only on a contiguous-residual generator base (L13's 74%-contiguous
  residual vs the frontier's 95%-scattered).
- **The #63 conditioning hinge (the loss-vs-capacity disambiguator).** [the T4 grand council's PROVISIONAL
  gate, `council_grand_symposium`] whether a CHEAP renderer's manifold intersects the margin-polytope under a
  BETTER-CONDITIONED loss (KL-T2 / margin-hinge, not argmax-CE) is DESIGNED-not-run. A PASS reopens the
  cheap-witness family (worth the full 0.073 distortion headroom → 0.118); a FAIL means the renderer-CAPACITY
  half is binding and the door is a full-fidelity smaller amortizer (lever C). **This $0 test gates the entire
  directional verdict.**

**The decisive strategic answer:** YES — the witness/score-quotient carrier's floor IS below HNeRV's, and it
is the genuine door past T_1 (the −59% rate row proves the rate half; the ~25–65 KB conditional MDL bounds it;
the 1.5 KB pose entropy bounds the pose half). The class shift is gated on the pose blocker (L14) + the seg
round-trip (L9) + the #63 conditioning hinge — all of which have a $0 disambiguator. **The witness is not a
new INR; it is the lens catalogue (the measured-oracle geometry + the exact solve) composed into a carrier
that spends only on the quotient. HNeRV becomes the BANK (the proven distortion-control base, → T_1) while the
witness is the CLASS-SHIFT (→ sub-0.118).**

### C.3 The full-stack synergy (how Layer-1 composes with the levers)

The witness carrier (Layer 1) IS the integration point of the lenses, not a separate thing the levers bolt
onto: the score-domain Lagrangian (L8) IS the witness objective; pose-FiLM (L14) IS witness Component 3;
margin-weighted seg (L9) IS Component 2 folded into training; the certified null fill (L5) IS Component 4; the
λ\*-allocator (L10) distributes bytes among them; the atlas (L7) supplies the marginals; Dykstra (L12) finds
the cheapest member. The three layers are ONE co-designed system minimizing `25·B_witness/N` subject to the
seg-cell + pose-ellipsoid constraints. The binding order [DERIVED: L16, the floor decomposition]: **distortion
FIRST** (drive the 0.073 residual → 0 at constant bytes, crossing T_3 with NO carrier swap — the proven-
arithmetic path), **THEN rate** (the witness null-space-slack recovery + a smaller amortizer push toward the
~25–65 KB conditional floor for sub-0.118).

---

## D. THE EXPLOITATION STRATEGY + FIRST $0 STEP

### D.1 The ranked exploitation of the frozen space (highest-EV toward sub-0.15)

| Rank | Path | What it exploits | EV toward sub-0.15 | Status | First step |
|---|---|---|---|---|---|
| **1** | **HNeRV basin as BANK (→ T_1) + distortion-closure levers** | the proven amortizer (L16: sub-0.15 is distortion-at-constant-bytes); the score-domain Lagrangian (L8); margin-weighted seg (L9-in-training); pose-FiLM (L14) | **HIGHEST** — reaches ~T_1 measured (live basin ep40 score 1.20); T_3 reachable via the 0.073 distortion residual at constant bytes (PROVEN-arithmetic: frontier scores 0.118 at d_seg=d_pose=0) | live basin (the running daemon, not ours to touch) | the live basin IS this; the FIRST analysis step is the #63 loss-conditioning $0 test (D.2) |
| **2** | **The witness / score-quotient carrier (→ sub-0.118, the CLASS SHIFT)** | the scorer-conditional MDL ~25–65 KB (L16); the certified null space (L5); the −59% byte-closed generator (L13); the pose witness ~1.5 KB (L14) | **MEDIUM-HIGH, UNCERTAIN** — the only path BELOW HNeRV's slack; gated on the pose blocker + seg round-trip + #63 | L13 byte-closed (−59%); L14 DESIGN; L5/L7/L10/L12 BUILT | the boundary-witness + pose-feasibility $0 probes (D.2) |
| **3** | **Pose-FiLM / PTNC (witness Component 3) on the HNeRV bank** | the pose null space + frame0-invisibility (L4/L14); the measured PoseNet Jacobian atlas (L7) | **HIGH** — closes the 9% pose term at ~1.5 KB; composes additively with rank 1 (the running carrier memo's measured-GO disambiguator) | DESIGN ($0-cleared by the disambiguator) | land `pose_film` (default-OFF, byte-identical) + the additive pose codec; stage the paired A/B |
| **4** | **Lossless recode BANK (R1+R2+R3, defensive)** | the byte-space null (L6); the measured per-tensor entropy | **−0.00092 exact** (the ONLY ready-made exact-axis win; FAILS the Innovation Gate — a DEFENSIVE bank, NOT the submission) | READY-NOW (~90 LOC + ~$0.3 paired replay) | build the R1+R2→R3 materializer; bank the both-axis frontier |
| **5** | **The KKT/OT allocator + MWCC contour coder on a REPAIRABLE base** | the exact-solve lenses (L9/L10/L11); the margin field (L2) | **MED** — empty on the frontier base (L15); fundable only on L13's contiguous-residual base | BUILT (allocator + 20 tests); MWCC DESIGN | run `closed_spec_boundary_solver` + the allocator on the lever-B base (the residual is 74% contiguous) |
| — | **Frozen-bytes rate attack (fixed-basis recode of the frozen decoder)** | — | **DOMINATED / EXHAUSTED** [MEASURED: L16 — 0 compaction headroom; counting-bound bars procedural-from-seed] | MEASURED-VERDICT | do NOT re-mine; the win is only in a forward-designed retrained carrier |

### D.2 The FIRST $0 LOCAL PROBE (MVP-first, the decisive measurement)

The whole carrier-reframe reduces to ONE question: **is the scorer-conditional MDL actually below 177 KB on
the REAL inputs, and does the witness round-trip + hold pose?** Three $0 probes answer it, in priority order;
all reuse the frozen scorers' already-computed outputs (CPU-torch, GT via `yuv420_to_rgb`, NEVER MPS), no GPU,
no basin contention, no new forward:

1. **TOP — the #63 d_seg-loss conditioning decisive test** (`dseg_loss_conditioning_decisive_test_DESIGN`
   exists, pre-registered, UNRUN). argmax-CE vs KL-T2 vs margin-hinge on a matched conv_pair_decoder. **This
   is the single highest-information unrun experiment** (T4 grand council op-routable #1): a PASS reopens the
   cheap-witness family (the full 0.073 distortion headroom → 0.118); a FAIL redirects to a full-fidelity
   smaller amortizer. It gates the entire directional verdict. Yousfi: *"run it before any carrier campaign —
   it re-ranks everything below it."*

2. **The scorer-conditional MDL measurement (the operator's named first step).** On the frozen basin
   checkpoint, MEASURE the actual byte cost of the witness's three sections at the score tolerance:
   (a) the mask-grammar / margin-polytope boundary residual (does the margin-weighted STC beat 1.27 B/flip on
   the contiguous lever-B residual? — the MWCC/STC-DEFER bar); (b) the pose-trajectory (confirm the 1,557 B
   measured floor on the live carrier); (c) the boundary round-trip survival (flip K boundary pixels, push
   through bicubic↑/bilinear↓/uint8, re-measure d_seg). **If (a)+(b)+(c) sum below 177 KB → the witness floor
   is confirmed below HNeRV's → the class shift is GO;** else the hybrid (rank 1+3) captures the d_seg win
   in-training. This is the falsifiable test of C.2's central claim.

3. **The pose-FiLM disambiguator** (the running carrier memo already returned a measured GO at the
   frozen-decoder lower bound) — the cheapest validated witness component; first BUILD step after the probes.

### D.3 The ORPHANED lenses to REACTIVATE (anti-signal-loss, per CLAUDE.md)

The deferral-recovery ledger + orphan inventory found NO premature kills (Catalog #307 CLEAN) — the recovery
is execution sequencing. But several high-value lenses are ORPHANED (built/measured but unused) and on the
critical path to the witness:

- **L17 contest-exploit atoms** (`per_class_chroma_anchor`, `tropical_argmax_boundary_grammar`,
  `stable_orbit_packet_diet`, `decoy_mosaic_residual_basis`) — ORPHANED (no module importer). The atlas engine
  should index them as canonical atom families; `tropical_argmax_boundary_grammar` is directly the seg-witness
  vocabulary.
- **L9 MWCC** — DESIGN-ONLY; the STC-clean-source DEFER's named reactivation cost-map. The $0 margin-weighted-
  STC-vs-brotli smoke is its gate.
- **L14 PTNC** — DESIGN-ONLY on the existing `amortized_luma_carrier.py` scaffold; the #1 build (the live pose
  blocker on the confirmed −59% generator).
- **L10 pose-OT Sinkhorn layer** — DESIGN-ONLY; composes directly with L14 once the carrier defines per-pair
  pose cost.
- **The cross-pair pose corrector's richer-grammar reactivation** (L15) — the PoseNet-Jacobian sub-pixel tube
  (`tac.boundary_math.posenet_jacobian_saliency`, already BUILT) is the larger action space the saturated
  K=16 lever needs.

The reactivation sequence is the deferral ledger's top-5 (R1+R2+R3 bank now; lever-C joint carrier = the
witness's pose blocker; boundary-solver on the repairable base; AFSR-1 fresh-init smaller-arch; surface the
submission dispositions) — all of which converge on the witness carrier.

---

## Wire-in hooks (CLAUDE.md 6-hook per Catalog #125)

1. **Sensitivity-map** — ACTIVE (synthesis): the lens catalogue IS the per-axis scorer-sensitivity prior
   (L2 margin field + L7 Jacobian atlas + L5 certified-null + the L16 term decomposition rate 61.7% / seg
   29.3% / pose 9.0%); feeds the bit-allocator distortion-before-rate-to-0.118 re-rank.
2. **Pareto constraint** — ACTIVE: the rate floor is a scorer-conditional invariant (L16); the Pareto frontier
   is `{distortion residual 0.073} × {rate slack above the ~25–65 KB conditional floor}`; the witness minimizes
   slack, the levers minimize the distortion residual.
3. **Bit-allocator** — ACTIVE (design): the witness's four-budget decomposition (`B_base + B_∂ + B_pose +
   B_null`) IS the allocator prior; the λ\*-equalizing solve (L10) + the atlas dispatch order (L11) + the
   certified null fill (L5) are its primitives.
4. **Cathedral autopilot** — N/A (synthesis; no archive-deployable artifact; the $0 probes are the next
   dispatch surface).
5. **Continual-learning posterior** — ACTIVE: this synthesis reseeds the planner with the bound lens
   catalogue + the carrier-reframe verdict (the witness floor IS below HNeRV's; the −59% rate is measured; the
   pose blocker is the gate) + the orphaned-lens reactivation list.
6. **Probe-disambiguator** — ACTIVE: the D.2 probes (the #63 conditioning hinge + the scorer-conditional MDL
   measurement + the pose-FiLM GO) are the disambiguators between "pure witness class-shift" and "hybrid
   ceiling."

**Mission contribution:** `frontier_breaking_enabler` (a lens synthesis that BINDS the council's frozen-space
eurekas into the meta-strategy framing the carrier choice; names the witness reframe + the orphaned lenses to
reactivate + the $0 probe that decides the class shift). **Frontier UNMOVED 0.19109982.** No score asserted.
No GPU launched. No paid spend. No collision with running agents.

---

## RETURN SUMMARY (the four deliverables)

**(1) Meta-insight + quantified "small information space."** The contest is an EXACT single-point optimization
against a frozen oracle (not learning), whose scored output is a low-dimensional NON-SMOOTH quotient (a
600-map SegNet argmax partition + a 600×6 PoseNet trajectory). The scorer-conditional MDL is KB-scale:
**pose-output entropy = 1,557 B [MEASURED]; the free-decoder-conditional intrinsic dimension = ~24.6–64.6 KB
[DERIVED] → 2.7×–7× below the 177 KB frontier** and far below the field's ~178 KB neural decoders. (Caveat:
direct partition storage LOSES — 524.8 KB / rate 0.169 — so the small object must be AMORTIZED, not stored.)

**(2) Complete lens catalogue: 17 lenses**, in 6 groups: quotient structure (L1–L4: witness-compiler frame,
argmax-polytope, pose-ellipsoid, frame0/1 asymmetry), null space (L5–L6: certified 22.7%/80.67% invisibility,
byte-space null), measured geometry (L7–L8: Response Atlas, non-arbitrariness/waterfilling), exact solve
(L9–L12: MWCC/STC inverse-steganalysis, λ\*/OT-Sinkhorn allocator, atlas dispatch order, Dykstra feasibility),
witness carrier (L13–L15: score-native generator −59% byte-closed, PTNC/pose-FiLM, cross-pair saturated-lever),
floor + boundary (L16–L17: info-theoretic floor + free-inflate boundary, exploit registry). Key statuses: L5
BUILT+certified, L7/L10/L12/L13 BUILT, **L13 byte-closed at −59% (the rate class shift is MEASURED-real)**,
L9/L14 DESIGN-ONLY (the witness's two open components), L16 MEASURED-VERDICT.

**(3) Carrier-reframing verdict: YES, the witness/score-quotient floor IS below HNeRV's 177 KB.** The lenses
imply the optimal carrier is NOT a general INR (which encodes the full frame manifold, paying for
scorer-invisible appearance + pose-null) but a SCORER-CONDITIONAL WITNESS parametrizing only the quotient
(mask-grammar + pose-trajectory + boundary residual + certified-free null fill). The −59% byte-closed
generator (L13) proves the rate half; the ~25–65 KB conditional MDL (L16) bounds the witness; the 1.5 KB pose
entropy (L14) bounds the pose half. The rate FLOOR is carrier-invariant, but the witness recovers HNeRV's
NULL-SPACE SLACK by spending ZERO on the invisible 95%. It is the genuine door past T_1 toward sub-0.118 —
gated on three honest risks (the pose blocker L14, the seg-boundary round-trip L9, the #63 conditioning hinge),
all $0-disambiguable. HNeRV = the BANK (→ T_1); the witness = the CLASS-SHIFT (→ sub-0.118).

**(4) Ranked exploitation + first $0 step + orphaned lenses.** Rank 1 = HNeRV basin as bank + distortion-
closure levers (sub-0.15 is a distortion threshold at constant bytes, PROVEN-arithmetic). Rank 2 = the witness
class-shift (sub-0.118). Rank 3 = pose-FiLM (closes the 9% pose term, measured-GO). Rank 4 = the lossless
bank (defensive, −0.00092, fails Innovation Gate). **FIRST $0 STEP:** the #63 d_seg-loss conditioning decisive
test (argmax-CE vs KL-T2 vs margin-hinge) — the single highest-information unrun experiment, gating the entire
directional verdict — followed by the scorer-conditional MDL measurement (the operator's named first step: do
the mask-grammar + pose-trajectory + boundary-residual bytes at score tolerance sum below 177 KB?).
**ORPHANED lenses to reactivate:** L9 MWCC contour coder (DESIGN, STC-DEFER's named reactivation), L14 PTNC
(DESIGN, the #1 build / live pose blocker), L10 pose-OT Sinkhorn (DESIGN), L17 contest-exploit atoms
(tropical-argmax-grammar / per-class-chroma / stable-orbit, no module importer), and the cross-pair corrector's
PoseNet-Jacobian sub-pixel richer-grammar (L15).

**Memo:** `.omx/research/frozen_contest_space_council_lenses_synthesis_20260612T173627Z.md`.

## Cross-references (the audited lens sources)

`layer1_carrier_first_principles_20260612T171912Z.md` (the sister carrier-candidate memo this composes with) ·
`council_grand_symposium_path_to_sub015_20260610T171906Z.md` (T4 grand council, the pattern-behind-patterns) ·
`information_theoretic_floor_T_floor_20260610.md` + `information_theoretic_floor_report_v1_20260610T102335Z.md`
(the floor decomposition) · `smaller_learned_basis_deep_math_20260610T191009Z.md` (the ~25–65 KB conditional
MDL + energy-compaction theorem) · `evaluator_invisibility_basis_landed_20260610.md` (the 22.7%/80.67%
certified null space) · `evaluator_response_atlas_design_20260609.md` + `evaluator_inverse_orphan_inventory_20260609.md`
(the atlas + the 103-surface map) · `evaluator_optimal_adaptive_waterfilling_non_arbitrariness_synthesis_20260609.md`
(the non-arbitrariness principle) · `sota_plus_original_inventions_20260610T125100Z.md` (MWCC / pose-OT / PTNC) ·
`boundary_math_seg_core_20260610T101618Z.md` + `closed_spec_boundary_math_system_of_equations_20260610.md` (the
polytope quotient) · `cross_pair_waterfilled_corrector_20260610T181531Z.md` (the saturated-lever lens) ·
`legal_frame_feasibility_dykstra_20260610T175421Z.md` (the Dykstra feasibility verdict) ·
`score_native_first_candidate_20260610T112433Z.md` + `lever_b_score_native_argmax_smoke_verdict_20260610.md`
(the −59% byte-closed witness) · `atlas_waterfiller_dispatch_order_landed_20260609.md` (the dispatch-order
lens) · `deferral_recovery_ledger_20260610T130200Z.md` + `GOAL_standing_v3_20260610.md` (the lever taxonomy +
orphan status) · `validation_score_program_compiler_quotient_theory_20260606.md` (the witness-compiler
validation) · CLAUDE.md "Evaluator-Equivalent Witness Compiler Paradigm" + "Meta-Lagrangian/Pareto solver" +
"THE GOAL — SUB-0.15."
