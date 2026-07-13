# MuonH + Manifold Muon deep-math dig — 2026-07-13

**Lane:** `lane_muonh_manifold_muon_dig_20260713`  
**Mode:** DESIGN / SOURCE AUDIT / SMALL LOCAL STATIC PROBE ONLY  
**Execution:** no training, no paid dispatch, no scorer launch, no archive mutation  
**Authority:** research-only; all optimizer outcomes below are `UNMEASURED` unless an older named anchor is explicitly marked `MEASURED`  
**Pointer delta:** NONE. No contest score or frontier movement is claimed.

## Verdict capsule

**Manifold-Muon geometry match: `MATCH-AT-NORM/MANIFOLD-FAMILY; GO-BUILD-POLAR-CHART; NOT FIREABLE`.** The current witness wants ordinary spectral Muon on the coordinate trunk matrices, exact Stiefel-tangent spectral descent on the **orthonormal polar factor** of `film.weight`, and evaluator-semantic/product norms—not spectral Muon—on codes, pose coordinates, heads, biases, and palette. The unit-Stiefel replacement `W -> polar(W)` is **not** a valid finishing retrofit: the live V9 checkpoint has `film.weight` singular values `[7.2831, 10.0768]`, and direct unit projection changes the matrix by `0.8823 ||W||_F` (`MEASURED`, static fp64 probe). The bounded candidate is the function-preserving polar chart

\[
W_0=Q_0H_0,\qquad Q_0\in\mathrm{St}(768,19),\quad
H_0=(W_0^\top W_0)^{1/2},
\]

then freeze `H0`, optimize only `Q` by exact tangent-spectral Manifold Muon, and deploy the folded matrix `QH0`.

**One-line module assignment:** `in_proj + hidden[0:4] -> unconstrained RMS→RMS spectral/Muon; film -> W=QH0, Q∈St(768,19), tangent spectral/Manifold-Muon; code -> (R^19)^1200 task-pullback/AdamW; pose dxi -> se(3)^600 task-pullback/AdamW; out_sdf -> RMS→linf margin norm/head solve; out_tex -> R/PoseNet pullback; biases -> native output-vector norms; palette -> l1→linf/max-entry pullback.`

**MuonH identified-as:** `Muon Hyperball`, Wen–Dang–Lyu–Ma–Liang, arXiv:2606.16899 (2026), **not Hessian Muon**. It fixes each selected matrix's Frobenius radius and normalizes the base-optimizer update before radial reprojection. **Current-witness verdict: `NO-GO / FORMULATION`** for generic full-matrix MuonH on the present unnormalized HOSC/FiLM witness because radial scale is functional and no trainable normalization gain removes it. This does not kill gain-decoupled hypersphere charts, the existing magnitude-direction decoupling path, or a future block that first proves scale invariance.

**A/B ticket:** `film_polar_chart_exact_manifold_muon_finisher`, status `WIRING_NEEDED`, cold-origin n600 common-boundary fork, target `d_seg <= 0.98*d_seg_start`, 250-finisher-epoch right-censor, treatment versus the **tuned** warm-momentum / LR-annealed Muon incumbent. No MuonH A/B is admitted now; its reactivation probe is radial-gradient plus gain-gauge custody, not a training launch.

---

## 1. Sources identified first

### 1.1 Canonical Bernstein / Modula line

| Source | What it establishes | Status used here |
|---|---|---|
| Jeremy Bernstein, [Modular Manifolds](https://thinkingmachines.ai/blog/modular-manifolds/), 2025-09-26 | A module is `(forward map, weight manifold, weight norm)`; Euclidean matrix + spectral norm gives Muon; Stiefel matrix + spectral tangent norm gives Manifold Muon; composed modules use a sensitivity-weighted max norm. | **PRIMARY / canonical Manifold-Muon writeup** |
| Modula docs, [Stiefel manifold algorithm](https://docs.modula.systems/algorithms/manifold/stiefel/) | Exact tangent-constrained spectral linear minimization oracle and dual-ascent construction for rectangular Stiefel matrices. | **PRIMARY algorithm derivation** |
| Large et al., [Scalable Optimization in the Modular Norm](https://arxiv.org/abs/2405.14813), NeurIPS 2024 | Modular max norm, feature-space scaling, and layer sensitivity budgeting. | **PRIMARY theory** |
| Bernstein & Newhouse, [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265), 2024 | Dualization of architecture-specific norms into optimizer directions. | **PRIMARY theory** |
| Bernstein & Newhouse, [Old Optimizer, New Norm](https://arxiv.org/abs/2409.20325), 2024 | The norm must follow a tensor's functional role; identical matrix shapes can require different norms. | **PRIMARY theory / assignment rule** |
| [modula-systems/modula](https://github.com/modula-systems/modula) | JAX OSS implementing modular construction, dualization, and projection. | **PRIMARY OSS** |
| [thinking-machines-lab/manifolds](https://github.com/thinking-machines-lab/manifolds) | Supporting reference implementation for the 2025 Manifold-Muon writeup (`manifold_muon.py`, matrix sign, hyperspherical descent). | **PRIMARY OSS / educational implementation** |

The canonical Stiefel-spectral direction at `Q` is

\[
A^*=\arg\min_A\ \langle G_Q,A\rangle
\quad\text{s.t.}\quad
A^\top Q+Q^\top A=0,\quad \|A\|_2\le 1,
\]

followed by a retraction. Ordinary Muon instead solves only the spectral-ball part in ambient matrix space. Therefore `ambient Muon step -> polar projection` is not definitionally Manifold Muon: the ambient direction can have a large normal component, so the learning rate is not the actual tangent step length.

### 1.2 Direct 2026 successors / alternatives

| Source | Relation to this lane | Disposition |
|---|---|---|
| Yang & Lai, [Manifold Constrained Steepest Descent](https://arxiv.org/abs/2601.21487), 2026 | MCSD; its Stiefel spectral specialization SPEL applies an LMO to the Riemannian gradient then projects in a single loop, avoiding Bernstein's nested tangent dual solve. | **Plausible implementation alternative, not algebraically identical to the exact tangent LMO.** Keep as the first reformulation if exact dual ascent is too expensive. |
| Xie et al., [Controlled LLM Training on Spectral Sphere](https://arxiv.org/abs/2601.08393), 2026; [OSS](https://github.com/Unakar/Spectral-Sphere-Optimizer) | Constrains weights and updates on a spectral sphere. | **Related weight-manifold family; not the derived FiLM Stiefel polar chart.** |
| Ren et al., [HyperP](https://arxiv.org/abs/2603.28743), 2026; [ArchScale OSS](https://github.com/microsoft/ArchScale) | Frobenius hypersphere parameterization with Muon; shows Depth-μP remains necessary rather than sphere constraints automatically solving depth transfer. | **Adversarial successor to broad MuonH transfer claims; useful only after scale invariance is established.** |

### 1.3 “MuonH” identity and ambiguity scan

The precise 2026 method named **MuonH** is Wen et al., [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899). In that paper:

- `H` means **Hyperball**, not Hessian.
- A base update `U_t` (Muon for MuonH) is Frobenius-normalized.
- The trial matrix is stepped by a prescribed angular/radius-scaled learning rate and renormalized to the fixed initialization radius `R=||W_0||_F`.
- Attention/MLP matrices receive the constraint; embeddings, norm gains, and other norm-semantic parameters remain on Adam.
- The paper's expressivity argument assumes a normalized architecture in which a trainable normalization gain can absorb matrix scale.
- Its theory deliberately uses idealized infinite-history and iid isotropic stationary-gradient assumptions; the paper itself labels the isotropy model idealized.
- The reported 20–30% token-equivalent speedup is on Qwen3-style language-model training and is **not evidence for this witness**.

No author-linked official MuonH repository was located in the paper or the source search as of 2026-07-13. That is an OSS-custody gap, not evidence that the method is nonexistent.

The plausible Hessian-flavored name collision is Du & Su, [The Newton-Muon Optimizer](https://arxiv.org/abs/2604.01472), with [official OSS](https://github.com/zhehangdu/Newton-Muon). Newton-Muon uses

\[
\operatorname{msign}\!\left(G(ZZ^\top)^{-1}\right),
\]

so it adds **right preconditioning by the layer-input second moment**, not the Hyperball radial constraint. Its reported 6% step and about 4% wall reductions belong to a NanoGPT reproduction. It is not called MuonH in the source. This memo covers it only to close the operator's “Hessian-informed Muon?” ambiguity honestly.

---

## 2. Repository truth: what the current Muon stage actually does

### 2.1 Source-inspected V9 architecture

**Custody:**

- checkpoint: `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_ckpt_stageOctave1_ep251.npz`
- checkpoint SHA-256: `c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758`
- launch SHA-256: `bd760505c445d51dc51d0b31eadd5a4d2628261220ffa46e2474ca83f358c601`
- Muon router SHA-256 before this landing: `f406c4f3b9ae7943a536ee69fdc22992d98bac58dee5fac81540703a99152569`
- Stiefel/MD helper SHA-256 before this landing: `fa90eaddce24eafd5f51a1b32d005f1e76c2ee69b80b2678742f68ab6f3ec27b`

The checkpoint contains **87,575 trainable parameters** (`DERIVED` by exact shape sum). Current `muon_finisher_param_filter` sends exactly six matrices—`in_proj.weight`, `film.weight`, and four `hidden.*.weight` matrices—totalling **59,136 parameters** to Muon (`DERIVED`, 67.53%). Everything else remains AdamW. The prompt's “flat spectral everywhere” is therefore an inherited shorthand, not the literal current implementation. It is “flat spectral on the six selected internal matrices.”

The PR95 figure `177K / 229K` remains valid historical heritage for that older architecture. It must not be silently substituted for current V9's `59,136 / 87,575` partition.

The tuned V9 incumbent is source-confirmed as:

- `--muon-lr 0.002`
- `--muon-momentum 0.95`
- `--muon-ns-steps 5`
- `--muon-warm-start-momentum`
- `--muon-lr-final-frac 0.1`
- `--muon-start-event powerlaw_meat`
- final heads, codes, pose carrier, biases, palette remain in the AdamW fallback.

`--film-stiefel` is **OFF** in that launch. If enabled, the existing code takes whichever ambient optimizer step was produced and then directly projects `film.weight` to orthonormal columns. That implementation is valuable as a separate DM1 structural arm, but it is not the exact tangent LMO above.

### 2.2 Small local static geometry probe

Machine-readable receipt: `.omx/research/muonh_manifold_muon_static_probe_20260713.json`.

| Quantity | Value | Label / scope |
|---|---:|---|
| `film.weight` shape | `768 x 19` | `MEASURED` checkpoint metadata |
| singular min / max | `7.283143 / 10.076844` | `MEASURED`, NumPy-fp64 SVD |
| stable rank / ratio to max rank | `13.3925 / 0.70487` | `DERIVED` from measured singular values |
| `||W^T W-I||_F` | `312.70284` | `MEASURED` static geometry |
| direct unit-polar `||Q-W||_F / ||W||_F` | `0.882329` | `MEASURED`; no score meaning |
| best scalar-Stiefel projection scale | `8.426739` | `DERIVED` as mean singular value |
| best scalar-Stiefel relative delta | `0.088815` | `MEASURED/DERIVED`; still not function-preserving |
| exact polar reconstruction `QH-W` | algebraically zero; fp32 parity unmeasured | `DERIVED`; build must measure actual byte-level parity |

**Consequence:** direct unit projection at the finishing boundary is rejected before an A/B. The exact polar chart preserves the incumbent function in real arithmetic and isolates the optimizer geometry. It also exposes why “Stiefel conditioning axis” should refer to `Q`, not blindly to the raw, unit-scaled `W` in an unnormalized network.

---

## 3. Canonical module-to-norm derivation

### 3.1 Product law

For modules `q` with native manifold `M_q`, perturbation norm `N_q`, and a positive common-budget multiplier `s_q`, the witness product geometry is

\[
\boxed{
\mathcal M=\prod_q\mathcal M_q,
\qquad
\|\Delta\theta\|_{\mathrm{mod}}
=\max_q s_qN_q(\Delta\theta_q)
}
\tag{MM-WITNESS-1}
\]

This is canonicalized in `src/tac/canonical_equations/witness_modular_norm_assignment_20260713.py` as `witness_modular_norm_assignment_v1`.

**Honesty boundary:** the **norm families and manifolds** below are `DERIVED` from the actual forward roles plus the primary modular-norm law. The inter-module `s_q` values are **UNRESOLVED / UNMEASURED** because the current trainer has no complete Modula sensitivity/mass annotation. Equation MM-WITNESS-1 may not be used to set learning rates until every `s_q` has custody.

Useful exact induced-norm identities:

\[
\|W\|_{\mathrm{RMS}(m)\to\mathrm{RMS}(n)}
=\sqrt{\frac{m}{n}}\,\|W\|_2,
\qquad
\|W\|_{\mathrm{RMS}(m)\to\ell_\infty}
=\sqrt m\max_i\|W_{i,:}\|_2.
\]

### 3.2 Actual V9 assignment

| Module / shape | Forward role | Derived manifold + norm | Current route | Candidate delta |
|---|---|---|---|---|
| `in_proj.weight` `96x80` | coordinate-feature RMS → hidden RMS | unconstrained matrix; `sqrt(80/96)*spectral` | Muon | **none** |
| `hidden.0..3.weight` `4x(96x96)` | hidden RMS → hidden RMS | four unconstrained matrices; spectral | Muon | **none** |
| `film.weight` `768x19` | conditioning coordinate → all FiLM scales/shifts | polar chart `W=QH0`, `Q∈St(768,19)`; spectral norm on tangent `A`, with `H0` defining the input-coordinate metric | ambient Muon; optional project-after-step arm OFF | **exact Manifold Muon on `Q`; freeze `H0` first formulation** |
| `code` `1200x19` | state-semantic FiLM coordinates | `(R^19)^1200`, local evaluator-pullback/product Euclidean; radius is functional | AdamW | **none; no sphere** |
| `pose_carrier.dxi` `600x6` | per-pair `se(3)` residual | `se(3)^600`, local R/PoseNet pullback | AdamW | **none** |
| `in_proj.bias` + four hidden biases `5x96` | hidden translations | product vector RMS norms | AdamW | none in this ticket |
| `film.bias` `768` | FiLM translation | output RMS vector norm | AdamW | none |
| `out_sdf.weight` `5x96` | hidden → class/SDF logits | RMS→`linf`: `sqrt(96)*max_row_l2`, refined locally by SegNet margin pullback | AdamW | none; TerminalSolve may compose separately |
| `out_sdf.bias` `5` | class-logit translation | `linf` / margin norm | AdamW | none |
| `out_tex.weight` `3x96` | hidden → pose-carrying RGB texture logits | local sigmoid/R/PoseNet pullback; RGB-RMS spectral is only a proxy | AdamW | none |
| `out_tex.bias` `3` | RGB texture translation | local sigmoid/R/PoseNet pullback | AdamW | none |
| `palette` `5x3` | class-probability `l1` → RGB-logit `linf` | `l1→linf = max_abs_entry`, refined by scorer pullback | AdamW | none |

### 3.3 Answer to the operator's proposed geometry

1. **Spectral trunk matrices: YES.** `in_proj` and the four hidden matrices are genuine vector multipliers between RMS feature spaces, so ordinary Muon is the correct modular dualizer family. Current routing already matches.
2. **Stiefel conditioning axis: YES, with a chart correction.** The Stiefel variable is the polar factor `Q` of the 768x19 FiLM operator. On the current nonnormalized witness, replacing raw `W` by a unit `Q` is a large function-changing scale intervention. `W=QH0` preserves the boundary function and makes `Q` the derived conditioning-axis isometry.
3. **Sphere “where #217 said”: NOT CURRENTLY DERIVED FOR AN ACTUAL PARAMETER BLOCK.** The local #217 artifact is a design-only post-Muon leap-residual reheat: margin reweighting of low-persistence pixels, a Muon/Stiefel/sphere curvature reading, optional log-decay SGLD, and a #216 saddle-signature gate. It is not a landed per-module sphere assignment. Codes and raw HOSC/FiLM trunk weights have meaningful radius. A sphere becomes admissible only after a gain or exact gauge absorbs that radius.
4. **Heads and palette: current exclusion is directionally correct.** Their authority norms are class-margin and R/PoseNet pullbacks, not generic spectral geometry.

### 3.4 What “geometry match” does and does not mean

`MATCH` means the primary theory selects the same **family** already derived by Pact: spectral trunk, Stiefel conditioning axis, semantic product geometry elsewhere. It does **not** establish:

- calibrated per-module learning-rate masses `s_q`;
- that exact dual ascent beats SPEL/MCSD in wall time;
- that the unit-Stiefel DM1 arm improves d_seg;
- that any sphere constraint is function-preserving on current raw weights;
- that the candidate beats tuned Muon;
- a score or pointer change.

---

## 4. MuonH adversarial evaluation

### 4.1 Delta over Muon

Muon normalizes the **update in spectral geometry** but lets the weight radius evolve under the optimizer and weight decay. MuonH adds:

1. Frobenius normalization of the Muon base update;
2. a fixed Frobenius weight radius set from initialization;
3. radial reprojection after every step;
4. an angular-learning-rate interpretation.

It adds **no Hessian**, no input-covariance inverse, no scorer curvature, and no new momentum structure beyond the base optimizer's state. Its novelty is explicit radial control.

### 4.2 Why generic MuonH is a NO-GO here

The paper's safe expressivity argument needs a scale-invariant block or a trainable gain that absorbs scale. The present witness does not have that property:

- scaling `in_proj` or a hidden matrix changes HOSC preactivation and therefore the represented function;
- scaling `film.weight` changes both the additive shift and the multiplicative term `1+scale`;
- code radius directly controls conditioning amplitude;
- no layer-normalization gain sits after these blocks to restore the same function.

Therefore a hard fixed Frobenius radius changes the function class/trajectory rather than merely choosing a gauge. The Hyperball paper itself excludes norm-semantic tensors; by the same rule, the witness's raw matrices require proof before constraint.

The checkpoint's stable-rank ratios make a second adversarial point. The four hidden matrices have ratios only `0.057–0.097`, so Frobenius control is not close to spectral control up to a near-constant full-rank factor. Hyperball's own stable-rank diagnostic therefore warns against treating Frobenius and spectral geometry as interchangeable on this instance.

**MuonH verdict:** `NO-GO / FORMULATION x CURRENT-ARCHITECTURE`. Generic Frobenius Hyperball on all current Muon matrices is not admitted against the tuned incumbent.

**Verdict scope:** this is not a family kill. It does not close:

- a future normalized/gain-decoupled witness block;
- `MDDecoupledOptimizer(base="muon")`, which already separates a spherical direction from trainable row/column gains and is structurally safer;
- the exact polar `Q,H0` chart above;
- a spectral-sphere rather than Frobenius-sphere formulation;
- Hyperball on another architecture with proven scale invariance.

**Reformation/reactivation queue:** at a common boundary, record for each proposed block

\[
\rho_{\rm radial}
=\frac{|\langle G,W\rangle|}{\|G\|_F\|W\|_F},
\qquad
r_{\rm stable}=\frac{\|W\|_F^2}{\min(m,n)\|W\|_2^2}.
\]

Only reactivate if the radial derivative is at its measured numerical/noise floor **and** an explicit gain/gauge proves function preservation. No guessed tolerance is registered here.

### 4.3 Relation to #423 and #217

| Existing lever | Actual scope | Composition / supersession verdict |
|---|---|---|
| `#423` Hessian-preconditioned OT/head-offset | Built preconditioned 5x5 class-offset Newton solve. `MEASURED` n600 averaged Hessian condition `7.66`; preconditioned and legacy paths reach identical `b*` and exact through-R d_seg `0.0048921034`. Negative scope is the dense averaged formulation; iterative #341/#396 remains open. | **MuonH neither composes via curvature nor supersedes it**, because MuonH has no Hessian. Exact Manifold Muon is a training geometry and can precede an accepted terminal head solve. If a future TerminalSolve mutates `film`, it must solve in the tangent chart or accept/rollback after refolding; a post-hoc projection may undo it. |
| `#217` leap-residual | Design-only post-Muon margin/persistence reweight + curvature/SGLD stage, gated by #216. No trainer flag/build. | Manifold Muon can supply the geometric optimizer inside #217; it does not replace the margin reweighting, saddle-signature gate, or optional stochastic escape. Never apply two independent Stiefel projections. |
| Newton-Muon ambiguity | Matrix-sign of gradient right-preconditioned by input covariance. | Potentially composable as a **training** input-conditioning preconditioner before a terminal scorer-Hessian solve, but it is not full evaluator curvature and does not supersede #423/#341. No ticket here: the operator asked MuonH after identification, and the exact source identifies MuonH as Hyperball. |

Recommended sequence if all separately survive: tuned/polar-chart Manifold Muon finish → #216 signature gate → #217 reheat if armed → accepted terminal head/full-P solve. Each stage preserves a distinct checkpoint and reports its own d_seg contribution.

---

## 5. Steps-dimension A/B ticket

The machine-readable ticket is `MANIFOLD_MUON_AB_TICKET` in `witness_modular_norm_assignment_20260713.py`.

### Ticket `film_polar_chart_exact_manifold_muon_finisher`

**Status:** `WIRING_NEEDED / UNMEASURED / NON-PROMOTABLE`.

**Cold-origin custody:** both arms start from the same seed-0 cold n600 vehicle and share an exact, complete pre-Muon stage checkpoint. The fork is allowed only because the treatment's polar factorization satisfies `W0=Q0H0`; before any update it must reconstruct the exact fp32 `film.weight`, emit its hash, and pass the common n600 through-R verdict within the measured deterministic parity floor. If it does not, block—the treatment is not the registered formulation.

**Control:** the compiler-emitted **tuned** incumbent, not vanilla Muon:

- ordinary Muon on `in_proj.weight`, `film.weight`, `hidden.*.weight`;
- LR `0.002`, momentum `0.95`, NS steps `5`;
- warm-started momentum;
- Muon LR final fraction `0.1`;
- event-defined `powerlaw_meat` start;
- identical AdamW fallback, losses, data order, speed settings, and frozen evaluator cells.

**Treatment:** identical except:

- factor common-boundary `film.weight W0 = Q0H0` deterministically;
- freeze `H0` for this first formulation;
- warm/transport the FiLM momentum into `T_Q St` with complete state custody;
- use the exact tangent-spectral LMO on `Q` and a deterministic retraction;
- fold `QH0` for EMA verdict, checkpoint export, and byte-close deployment;
- keep ordinary Muon on `in_proj` and `hidden.*`; keep all fallback tensors identical.

**Milestone:** measure common-boundary exact n600 `d_seg_start`; set

\[
d_{\rm target}=0.98\,d_{\rm seg,start}.
\]

`0.98` is an **ASSUMED preregistration policy** inherited from the steps ticket. The numeric target is **DERIVED** only after the common boundary replay.

**Crossing law:** first emitted exact n600 d_seg at or below target; no interpolation. Evaluate at the predeclared cadence. Right-censor after 250 nominal finisher epochs; a missing crossing remains `None`, never zero. Record accepted optimizer updates separately from tangent-dual iterations.

**Required receipt fields:**

- cold seed/config/source/checkpoint hashes;
- `Q0`, `H0`, reconstruction hashes and parity residual;
- all optimizer, momentum, dual, schedule, RNG, EMA, stage/epoch, and resume state;
- exact optimizer updates, skipped updates, dual iterations, retractions;
- direct elapsed to crossing, recurring verdict time, one-time polar cost, terminal critical path;
- tangent residual `||A^TQ+Q^TA||_F` and Stiefel residual `||Q^TQ-I||_F`;
- `||A||_2`, ambient-Muon/tangent direction alignment, and actual retracted geodesic/chord step;
- holistic facets: per-class d_seg, anchors/island birth, d_pose, archive bytes;
- deterministic NumPy-fp32 through-R + frozen CPU-torch n600 authority; MLX is advisory.

**Admission:** treatment must have strictly fewer accepted optimizer updates **and** lower direct elapsed to the same milestone, with no worse exact total score at its crossing and no topology/pose/rate guard regression. The incumbent is not displaced by a proxy-loss win.

### Why no optimizer/DSL implementation landed

The algebra is small; the **honest apparatus landing is not MLX-trivial**. A compliant implementation must close together:

1. deterministic rectangular polar factorization and fp32 fold-back parity;
2. exact or explicitly named approximate tangent LMO;
3. tangent momentum transport/warm start;
4. dual state and iteration-budget checkpointing;
5. atomic stage/intra-stage resume with `Q`, frozen `H0`, optimizer/EMA state;
6. EMA/deploy refolding and byte-close reference parity;
7. typed scheduled/default-OFF DSL compilation and never-invent-flags checks.

Landing a flag that merely composes current Muon with `--film-stiefel` would falsely relabel the existing approximation. Therefore **no DSL lever stub and no optimizer variant were built**. The DAG feed names the exact build surface and refuses launch until it exists.

### MuonH ticket disposition

No A/B ticket is admitted for generic MuonH on this architecture. Its bounded next action is the read-only radial/gain diagnostic above. If that probe establishes a scale-invariant block with a compensating gain, a new ticket must compare the best tuned Hyperball radius/angular schedule to the same tuned Muon incumbent; vanilla defaults are inadmissible.

---

## 6. Triality and system wire-in

### Equations leg

- `src/tac/canonical_equations/witness_modular_norm_assignment_20260713.py`
- canonical ID: `witness_modular_norm_assignment_v1`
- pure helpers for weighted modular max norm and exact RMS-induced scaling;
- complete 12-row current V9 parameter inventory;
- unmeasured build-gated n600 ticket;
- no registry append and no empirical score anchor.

### DAG leg

- `.omx/research/muonh_manifold_muon_DAG_FEED_20260713.md`
- contains source/geometry gates, reference/parity gates, DSL/resume gates, and the final governed A/B edge.

### DSL leg

- Existing `--film-stiefel`: **separate approximation/structural arm**, not renamed.
- Proposed exact lever: **not built**; typed object/flags remain a DAG build obligation.
- Never-invent-flags respected.

### Six-hook coherence declaration

1. **Sensitivity map:** equation exposes per-module norm families; scalar `s_q` calibration is explicitly missing and build-gated.
2. **Pareto constraint:** admission includes d_seg milestone, no-worse total score, d_pose, topology, rate, and direct wall.
3. **Bit allocator:** no bit-allocation mutation. Folded `QH0` retains the existing deployed tensor shape; archive bytes must nevertheless be remeasured because training can change entropy.
4. **Cathedral/autopilot:** DAG feed supplies `WIRING_NEEDED` and refuses dispatch until reference/resume/DSL gates close.
5. **Continual learning:** no posterior/probe-outcome row is appended because no empirical optimizer result exists. The static geometry receipt is durable source signal only.
6. **Probe disambiguator:** exact Manifold-Muon vs current project-after-step approximation is resolved by the registered matched n600 ticket after build; generic MuonH must first pass the radial/gain probe.

---

## 7. Verdict scopes and reformulation queue

| Candidate | Verdict | Exact scope | Reactivation |
|---|---|---|---|
| exact polar-chart Manifold Muon | `GO-BUILD / NOT FIREABLE` | film-only `Q∈St(768,19)`, frozen common-boundary `H0`, tuned incumbent control | close NumPy reference, MLX parity, momentum/dual resume, EMA fold, typed DSL; then run ticket |
| direct unit-Stiefel finishing projection | `NO-GO / INSTANCE x INITIALIZATION FORMULATION` | named ep251 V9 checkpoint; 88.23% relative matrix displacement | cold co-designed unit-Stiefel vehicle or exact gauge/gain reparameterization; do not retrofit silently |
| current ambient Muon + polar projection | `PLAUSIBLE APPROXIMATION / NOT MANIFOLD-MUON IDENTITY` | existing `--film-stiefel` semantics | may be a third diagnostic arm only if separately named and wall-charged |
| generic MuonH on current selected matrices | `NO-GO / FORMULATION x CURRENT ARCHITECTURE` | raw unnormalized HOSC/FiLM V9 blocks | prove radial derivative null + explicit compensating gain/gauge; retune radius/angular LR |
| Newton-Muon | `IDENTIFIED ALTERNATE / NO TICKET` | input-covariance right preconditioning, not MuonH | separate operator request after source-specific architecture/Hessian audit |
| #217 sphere wording | `INSUFFICIENT FOR MODULE ASSIGNMENT` | design memo's leap-residual curvature reading | build #216 signature and identify an actual gain-decoupled sphere block first |

---

## 8. Stores consulted

- `CLAUDE.md` and supplied `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/steps_dimension_95kill_20260713_SPEC.md`
- `.omx/research/sub015_DAG_steps_dimension_95kill_20260713.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- latest local Codex findings/session summary, latest T3/design memos, fleet full-research directive
- `experiments/train_levelset_witness_realized_through_R_mlx.py`
- `src/tac/optimization/muon_finisher_mlx.py`
- `src/tac/optimization/md_decoupling.py`
- `src/tac/boundary_math/laguerre_logit_offset.py`
- `src/tac/witness_dsl/curriculum_dsl.py`
- named V9 launch/checkpoint and local research ledgers for #217/#423/#269/#270
- the primary external sources enumerated in §1

## 9. Verification receipt

- local static probe only; no training launched;
- canonical module inventory reconciles to 87,575 trainable / 59,136 current-Muon parameters;
- focused tests: `4 passed` for `tests/test_witness_modular_norm_assignment_20260713.py`;
- static probe JSON parses cleanly;
- score claim: `false`;
- promotion eligible: `false`;
- pointer moved: `false`.

