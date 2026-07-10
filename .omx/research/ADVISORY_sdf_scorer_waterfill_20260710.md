# ADVISORY — exact scorer geometry and interaction-aware SDF bit water-filling — 2026-07-10

`research_only=true`

Authority: mathematical allocation advisory. This memo proposes no score, changes no canonical
equation/DSL/DAG, launches no experiment, and does not authorize a pointer move.

## Answer first

The frontier cannot be optimized by independently lowering proxy Seg loss, Pose loss, and payload
size. The exact score is nonlinear in pose, discrete in Seg pixels, discontinuous through uint8 and
argmax, and nonadditive across interacting carrier mutations. The optimal controller is therefore a
hierarchical, interaction-aware water-filler:

1. generate legal atomic proposals in the SDF geometry (`G`), exact-D texture (`T`), pose carrier
   (`P`), and compiler (`C`) homes;
2. compile each proposal through the real receiver and final ZIP;
3. measure exact component and byte deltas;
4. estimate material pair/group interactions;
5. select a budgeted compatible set;
6. compile and score the joint archive again;
7. update the global waterline and repeat until no exact negative-`Delta S` proposal remains.

Continuous KKT/gradient calculations are useful proposal heuristics only. Fresh decoded archive score
is the admission authority.

## 1. Exact objective and atomic mutations

Let `q` be a fully quantized, receiver-parseable candidate description and `A(q)` its final archive.

\[
S(q)=100d_s(q)+\sqrt{10d_p(q)}+\lambda_B B(q),
\qquad
\lambda_B=\frac{25}{37{,}545{,}489}
            \approx 6.6585895\times10^{-7}\ \text{score/byte}.
\]

For an atomic mutation `m_j`, the exact delta is

\[
\Delta S_j=
100\,[d_s(q\oplus m_j)-d_s(q)]
+\sqrt{10d_p(q\oplus m_j)}-\sqrt{10d_p(q)}
+\lambda_B[B(q\oplus m_j)-B(q)].
\]

Accepting `m_j` is favorable only if `Delta S_j<0` after final ZIP and parse-back. Report:

- Seg delta and affected class/cells;
- Pose delta and the nonlinear square-root score delta;
- exact archive-byte delta, including headers/contexts/alignment/code;
- authority axis and exact candidate/archive hashes;
- receiver survival and raw mutation hash;
- interaction scope and verdict scope.

## 2. Local exchange rates

### 2.1 One Seg cell versus bytes

At n600, one hard Seg cell is worth

\[
\Delta S_{1cell}=8.4771050347\times10^{-7}.
\]

At the exact byte price, this equals

\[
\frac{8.4771050347\times10^{-7}}
     {6.6585895312\times10^{-7}}
=1.273108\ \text{bytes}.
\]

This surprising ratio is a decision aid, not permission to assume independent pixels. SegNet is
global and hard argmax changes can interact across cells.

### 2.2 Pose marginal versus bytes

For infinitesimal pose change at debt `d_p>0`,

\[
\frac{\partial S}{\partial d_p}=\frac{\sqrt{10}}{2\sqrt{d_p}}.
\]

The pose-debt reduction that pays for one byte locally is

\[
\epsilon_{1B}(d_p)
=\frac{2\lambda_B\sqrt{d_p}}{\sqrt{10}}.
\]

The derivative diverges as `d_p` approaches zero, so a small absolute pose improvement can remain
valuable near the frontier. The exact finite difference is always safer:

\[
G_{pose}(d,\epsilon)=\sqrt{10d}-\sqrt{10(d-\epsilon)}.
\]

### 2.3 Score-unit value per byte

For a proposal that costs `Delta B>0`, define

\[
v_j=-\frac{100\Delta d_{s,j}
+\sqrt{10(d_p+\Delta d_{p,j})}-\sqrt{10d_p}}
{\Delta B_j}.
\]

The proposal beats the rate term when `v_j>lambda_B`. For byte-saving proposals with
`Delta B<0`, use exact `Delta S` directly; ratios become misleading near zero denominators.

## 3. Target waterline algebra

For target score `S*`, bytes `B`, and Seg debt `d_s`, the maximum admissible pose debt is

\[
d_p^{max}(S^*,B,d_s)=
\frac{\max\{S^*-100d_s-\lambda_BB,0\}^2}{10}.
\]

Using `S*=0.19109982419209975` only as a current local CPU design waterline:

| final archive bytes | d_seg | max d_pose |
|---:|---:|---:|
| 75,000 | 0.0012 | 4.4776e-5 |
| 75,000 | 0.0010 | 1.69418e-4 |
| 75,000 | 0.0007 | 5.06380e-4 |
| 90,000 | 0.0012 | 1.24825e-5 |
| 90,000 | 0.0010 | 9.71726e-5 |
| 90,000 | 0.0007 | 3.74208e-4 |
| 90,000 | 0.0005 | 6.58898e-4 |
| 100,000 | 0.0010 | 6.00933e-5 |
| 100,000 | 0.0007 | 2.97177e-4 |
| 120,000 | 0.0010 | 1.25367e-5 |

The feasibility frontier is strongly curved. Adding bytes without lowering Seg forces a much smaller
pose debt; lowering Seg by `3e-4` can relax pose requirements by several fold. Vehicle design must
therefore optimize the triplet jointly.

For the longer-term `S*<0.15` objective at 90 kB:

| final archive bytes | d_seg | max d_pose for S=0.15 |
|---:|---:|---:|
| 90,000 | 0.0010 | infeasible |
| 90,000 | 0.0007 | 4.029e-5 |
| 90,000 | 0.0005 | 1.606e-4 |

This joint waterline prevents a false roadmap in which pose alone is polished while the geometry
partition remains too expensive.

## 4. Why independent greedy allocation fails

For mutations `i,j`, define the exact interaction

\[
H_{ij}=S(q\oplus i\oplus j)-S(q\oplus i)-S(q\oplus j)+S(q).
\]

- `H_ij>0`: antagonism; joint benefit is worse than the sum;
- `H_ij<0`: synergy;
- `H_ij=0`: score-additive at the tested operating point.

Likely interaction sources include:

- shared network weights and latents;
- class-potential argmax competition;
- one boundary update moving multiple Seg cells;
- frame1 texture moving both Seg and Pose;
- entropy contexts changing the byte cost of other symbols;
- ZIP compression and section headers;
- quantization thresholds and tie flicker;
- topology changes creating/removing whole islands;
- scorer receptive fields coupling remote pixels;
- optimizer/curriculum state when proposals are trained rather than compiled.

The evaluator advisory established a precise local Pose footprint but not SegNet locality. Until a
same/adjacent/remote interaction matrix exists, “pair-local,” “cell-local,” and “diagonal batching”
are hypotheses, not authorities.

## 5. Interaction-aware selection

### 5.1 Proposal passport

Each proposal `m_j` should carry:

```text
proposal_id
base_archive_sha256
receiver_manifest_sha256
home = G | T | P | C
frame/pair/class/edge/footprint scope
quantized payload mutation
final archive byte delta
raw output mutation hash
d_seg delta and per-class/cell support
d_pose delta and output support
exact Delta S
runtime/RSS delta
interactions measured
verdict_scope
```

### 5.2 Conflict graph

Build a graph with proposals as vertices and edges when any of the following holds:

- they mutate the same quantized symbol/section;
- their measured `|H_ij|` exceeds a registered materiality threshold;
- one invalidates the other's base archive;
- they share an entropy context whose joint cost is unmeasured;
- they change the same topology event or class gauge;
- their receiver/runtime resources conflict.

Independent sets are candidates for batched evaluation. An absent edge means measured compatibility,
not merely disjoint filenames.

### 5.3 Budgeted joint selection

For binary decisions `x_j`, a quadratic proposal model is

\[
\min_{x\in\{0,1\}^n}
\sum_j x_j\Delta S_j+
\sum_{i<j}x_ix_jH_{ij}
\]

subject to legality, runtime, class/topology, and optional byte-budget constraints. For large `n`, use
screening and block-coordinate selection:

1. reject illegal/receiver-blind proposals;
2. reject individually unfavorable proposals unless a pre-registered synergy group exists;
3. cluster by interaction support;
4. solve each small cluster exactly or by branch-and-bound;
5. merge cluster winners;
6. compile and score the full joint archive;
7. attribute the residual model error and update interactions.

The quadratic model proposes; the joint fresh archive decides.

## 6. Hierarchical water-filling over the SDF stack

### Level 0 — architecture budget

Compare complete legal grammar families:

- v7.5.3 single trunk with separate G/T/P homes;
- v8 class-owned K potentials;
- optional integrable E-edge reparameterization;
- terminal HNeRV/other INR controls.

Do not mix a candidate residual with a control decoder and call the delta a full codec score.

### Level 1 — home budget (`G/T/P/C`)

Allocate until the best exact proposal in each home reaches a common global marginal. `C` proposals
may have negative bytes with zero distortion and should normally be admitted first, subject to
runtime/determinism.

### Level 2 — frame budget

- frame0: pose-only generator;
- frame1: geometry, exact-D texture, and limited pose finishing;
- joint frame changes require interaction measurement.

### Level 3 — class/edge budget

Rank class and edge annuli by:

- hard Seg cell debt;
- margin to the competing potential;
- island birth/death consequence;
- temporal persistence;
- entropy/description cost;
- measured off-region effects.

### Level 4 — footprint/atom budget

Within a 2x2 Pose footprint, allocate among the six output-effective/null atoms according to the home:

- frame1 texture uses the exact null subspace when it can move Seg;
- frame0 pose uses the output-effective complement;
- lift to camera pixels only after solving the exact resize preimage.

### Level 5 — pair/code budget

Use a small common pose-code dimension, then sparse hard-pair corrections. Avoid a uniform high K
when the residual value is heavy-tailed.

### Level 6 — symbol/entropy budget

Choose quantizer steps and context splits from final ZIP deltas. A “better entropy model” that adds
more table/code/header bytes than it saves is rejected.

## 7. Continuous KKT view, with its boundary

For differentiable rate models `R_k(theta_k)` and surrogate debts, a relaxed problem gives

\[
\nabla_{\theta_k}
\left(100\widetilde d_s+\sqrt{10\widetilde d_p}+\lambda_B\sum_kR_k\right)=0.
\]

Equivalently, active components have equal marginal score benefit per modeled bit. This is valuable
for proposing allocation schedules. It cannot certify:

- hard Seg argmax flips;
- uint8 rounding;
- actual entropy/ZIP bytes;
- receiver consumption;
- topology discontinuities;
- exact CPU/CUDA drift.

Therefore use KKT at stage boundaries to initialize or reallocate component budgets. Never adjust
loss weights per step based on noisy proxy gradients; that destroys the causal treatment and can
violate the stage-boundary contract.

## 8. SDF-specific proposal families

### Geometry `G`

- class-area multiplier updates;
- curvature/bandwidth changes by edge scale;
- island-birth corrections;
- junction-local potential atoms;
- temporal screw/advection residuals;
- class-owned latent/basis increments;
- gauge-preserving potential re-centering;
- quantizer refinement only near active margins.

### Texture `T`

- exact-D null atoms on frame1;
- annulus-limited Seg corrections;
- palette gauge removal;
- deterministic-bank regeneration;
- class/edge-conditioned coefficient sparsification;
- ChromaRung add-back after receiver closure.

### Pose `P`

- `xi` refinement;
- analytic basis coefficient;
- learned global basis column;
- SDF-conditioned generator width/frequency block;
- hard-pair coefficient sidecar;
- joint frame1 pose finishing only with Seg interaction.

### Compiler `C`

- section removal;
- quantizer/packing change;
- entropy context merge/split;
- header/version simplification;
- deterministic bank/code generation;
- dead-tensor elimination;
- receiver implementation shrink subject to runtime and portability.

## 9. Necessary empirical maps

### 9.1 Seg interaction stencil

For a controlled frame1 perturbation at cell `u`, measure effects at:

- same output cell;
- four/eight adjacent cells;
- same object boundary;
- same class remote region;
- different class remote region;
- different pair/frame.

Repeat over interior, simple edge, thin lane, island, and triple/higher junction strata. Record hard
argmax flips and continuous logit changes. This map determines whether diagonal batching is legal.

### 9.2 Pose atom map

For each exact 2x2 atom and frame:

- raw YUV6 delta;
- six Pose-output delta;
- quantization survival interval;
- interaction with neighboring footprints;
- value per encoded coefficient.

### 9.3 Entropy cross-effects

Measure final archive size for:

- each section alone;
- pairs of sections;
- changed code ordering;
- context resets versus shared state;
- sparse indices and headers;
- generated versus serialized fixed banks.

### 9.4 Runtime cross-effects

Measure decode seconds and RSS for combined features. Free bytes do not imply free wall time.

## 10. Candidate search protocol

1. Freeze a receiver-closed base archive.
2. Enumerate proposals from typed components only; no invented command-line flags.
3. Quantize and serialize each proposal.
4. Fresh inflate, cardinality check, raw hash, exact component score.
5. Compute exact `Delta S` and value per byte.
6. Probe interactions for overlapping support and a random remote control.
7. Build the conflict graph and select a joint batch.
8. Fresh compile/score the joint archive.
9. Compare predicted and realized joint deltas; update the interaction model.
10. Preserve accepted/rejected ledger and resume state.
11. Stop when no legal measured proposal crosses the waterline.
12. Select one exact archive; evaluate contest-CPU and contest-CUDA separately.

This loop is naturally resumable: every proposal and joint batch is content-addressed and complete.

## 11. Smallest convincing proof matrix

| Claim | Minimum proof |
|---|---|
| one byte has a stated price | final ZIP byte delta |
| one Seg cell was repaired | fresh hard evaluator output and cell coordinate |
| pose improved | fresh exact d_pose and nonlinear score delta |
| proposal is receiver-real | parse-back changes expected raw bytes |
| proposals are independent | same/adjacent/remote interaction tests below threshold |
| batch is favorable | full joint archive exact `Delta S<0` |
| allocation converged | no remaining measured legal proposal above waterline |
| result is reproducible | candidate passport and deterministic reinflate |

## 12. Apparatus defects this controller must refuse

- a scorer silently accepting a short output prefix;
- payload bytes reported without ZIP/header effects;
- a fixed bank counted despite being advertised as regenerated/free;
- a trainable tensor serialized but ignored by inflate;
- an in-memory counter disagreeing with the canonical archive builder;
- a shared-latent class update labeled independent;
- a diagonal batch asserted from pair labels rather than measured support;
- a proxy-loss improvement promoted without exact component deltas;
- a macOS advisory result presented as contest-CPU/CUDA;
- a public author claim presented as official evaluator authority.

## 13. Literal dispositions

- Waterline calculator/spec: `GO — ADVISORY/BUILD ONLY`.
- Proposal-passport and interaction-map build: `GO — BUILD ONLY`, under new claimed lanes.
- Diagonal click/atomic batching: `HOLD` until independence is measured.
- Continuous per-step adaptive loss weighting: `REFUSED`.
- Stage-boundary component reallocation: `GO` after typed config and resume persistence.
- Exact joint candidate selection: `HOLD` until receiver identity/cardinality close.
- Pointer movement: `HOLD` until separate exact axes and canonical promotion gates.

## 14. Exact blockers

1. SegNet same/adjacent/remote dependencies are not measured sufficiently.
2. The raw Pose Jacobian/atom receipt is incomplete.
3. Current v7.5.3/v8 optional tensors are receiver-blind.
4. Final-ZIP cross-effects for proposed G/T/P sections are unmeasured.
5. The exact pose-carrier proposal set does not yet exist.
6. v8 class updates are coupled by shared trainable code.
7. No content-addressed accepted/rejected proposal ledger is wired to the allocator.
8. No receiver-closed candidate exists for joint selection or exact-axis evaluation.

## 15. Triality and future wire-in

This memo changes no shared triality surface. A future implementation unit should land:

- DSL: typed proposal families, stage-boundary budgets, material interaction threshold, and
  mutually exclusive carrier arms;
- DAG: enumerate -> compile -> parse-back -> score -> interact -> select -> joint-rescore -> stop;
- equations: exact mutation delta, interaction `H_ij`, conflict-constrained selection, and target
  pose waterline.

Every measured proposal should update the sensitivity map, Pareto constraints, bit allocator,
cathedral consumer, continual-learning posterior, and probe disambiguator. A Markdown-only allocator
is not the final apparatus; this document deliberately stops at advisory authority.

## 16. Stores and primary research consulted

Local stores:

- `ADVISORY_evaluator_video_geometry_20260710.md`;
- `r1_dxi_shippability_byteclose_20260708.md`;
- `fullstack_fractal_optimal_synthesis_20260710.md`;
- v7.5.2/v7.5.3/v8 specs and fresh-eyes advisories;
- canonical frontier pointer, evaluator, byte-close builder, and current receiver sources.

Primary research context:

- task-aware coding: <https://arxiv.org/abs/2108.09993>
- task-oriented lossy compression: <https://arxiv.org/abs/2405.04144>
- classical rate-distortion background is used only as mathematics; all task-specific exchange rates
  above are derived from the frozen contest score and local evaluator geometry.
