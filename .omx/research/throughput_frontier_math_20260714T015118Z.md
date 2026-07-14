# Task #494 frontier mathematics for bit-identical heterogeneous throughput

**UTC:** 2026-07-14T01:51:18Z  
**Amended through:** 2026-07-14T03:01:20Z (ordinal / Minkowski-sigma / plasticity directives)  
**Lane:** `lane_throughput_frontier_math_20260713` · L0 · `research_only=true`  
**Pointer status:** **UNCHANGED.** The submittable pointer remains `0.19108282419209976`
`[contest-CPU Linux x86_64]`; the separate PR128-derived defensive bank remains
`0.1880443979880752` `[contest-CPU]` and non-submission. This lane built MEANS only: no archive,
evaluator, training, paid dispatch, live-run mutation, or score claim.

## Executive result

The deepest useful result is also the simplest: **do not quantize the entire world merely to cure
one reordered reduction. Put exactness on the nondeterministic reduction cutset.** If every other
node of a byte-level graph is deterministic and each reorderable reduction is replaced by an exact
commutative accumulator whose full reachable range is representable, followed by one deterministic
finalization, then the complete graph is bit-identical by induction over its DAG.

For the already-measured real render-R Q15 transpose instance, the maximum absolute accumulator is
`B = 11,159,918`. The literally minimal two's-complement width representing every value in
`[-B,B]` is

```text
w_min = 1 + bit_length(B) = 25 signed bits.
```

Therefore int32 is **MEASURED-scope sufficient** with `192.428263989×` positive-limit headroom, but
not information-minimal. This is the narrow algebraic reason the float cell produced 10 distinct
hashes in 10 Metal processes while Q15/int32 produced one. The statement is about the declared
quantized reduction. Equivalence to NumPy-fp32 logits and preservation of SegNet argmax are separate
error/certificate obligations.

The ranked conclusions, corrected by main's later n96 authority-verdict timer and the operator's
score-relevant arithmetic-convergence amendment, are:

1. **Matched real-n600 CE versus zero-margin winner/rival hinge — MEASURE FIRST OVERALL.** This is
   the cheapest already-typed direct test of whether arithmetic loss is `DOMINANT`, `CONTRIBUTORY`,
   `INERT`, or a `TRADEOFF` for Lane/Movable convergence. It is score-relevant, not a throughput
   claim, and geometric hard/easy strata must remain separate.
2. **Metric-admissible class-pair perimeter versus all-ones — BUILD/MEASURE SECOND.** Existing scalar
   `sigma_cc'` is pair-weighted but spatially isotropic. The fitted matrix violates a triangle
   inequality and therefore relaxes by wetting; it is not presently a proved Gamma-limit energy.
3. **Certified fixed-point SegNet forward — BUILD/MEASURE FIRST WITHIN THROUGHPUT.** The authority verdict is
   forward-only. **MEASURED n96:** `59.615 s` combined, `0.621 s/pair`, SegNet share `0.774`, PoseNet
   share `0.226`. **DERIVED linear n600 projection:** `372.6 s = 6.21 min`. The highest-EV end-state
   is a bit-identical GPU/ANE fixed-point SegNet whose n600 argmax is certified and whose actual
   integer residency/latency is measured.
4. **Certificate-gated precision cascade — BUILD with the fixed-point ladder.** Compute cheaply,
   accept a pixel only when its interval winner separates, and refine the remaining uncertainty set.
   This is the valid tropical/margin synthesis; replacing an ordinary CNN by max-plus is not.
5. **Exact reduction cutset — APPLY inside the fixed-point scorer and training R-adjoint.** It is the
   minimal structural cure for reordered reductions. For throughput, its first target is the SegNet
   integer-MAC accumulator path; integer R-adjoint is valuable training reproducibility, not the
   measured authority-verdict bottleneck.
6. **Exact-forward/distilled-forward alternative — BUILD/MEASURE against the measured per-pair and
   derived 6.21-minute n600 baseline.**
   A student must preserve the centered decision quotient and worst-pair argmax, not merely mean logits;
   full input-VJP fidelity remains an additional training gate rather than the primary verdict gate.
7. **Weight-decay plasticity A/B — BUILD ONLY AFTER TYPED PREREGISTRATION.** The LM paper motivates
   the measurement but does not transfer authority to a single-video memorization INR. Measure
   per-class convergence, one stable effective-rank definition, and exact archive-byte custody.
8. **Per-layer discrete margin waterfill — MEASURE after real error/cost points exist.** The KKT law
   is a lower bound/initializer; the executable verdict is the exact discrete Pareto allocation.
9. **PoseNet needs a separate continuous-output certificate.** Argmax mathematics cannot authorize
   the measured n96 `0.226` PoseNet share; compare its first six outputs and nonlinear pose-score debt.
10. **Exact spatially sparse frozen-teacher forward — `NO_GO` at the current formulation/instance.**
   Twenty-three global squeeze-excite blocks and a 685-pixel exact halo close any nonempty demand to
   the full frame, fixing the exact-forward arithmetic ceiling at `1.0×`. This does not kill sparse
   cotangents, custom adjoints, local students, or explicitly approximate cached-SE formulations.

The single highest-EV next measurement **overall** is the matched real-n600 CE versus existing typed
`margin_hinge` (`margin_target_end=0.0`) convergence A/B from identical EMA bytes. The single
highest-EV **throughput** measurement remains the real-n600 calibrated fixed-point SegNet forward
rung: aggregate and worst-pair flips, uncertified fraction, exact argmax digest, synchronized
integer-kernel latency, and proved GPU/ANE residency. Main's earlier n24 in-loop timing
(`~0.048 s` backward versus `~0.020 s` forward) described a fast MLX training slice and was explicitly
**RETRACTED as the throughput aim** after the n96 authority-verdict timer landed. It must not route
this campaign back toward backward-first work.

## Authority and recalled stores

This synthesis re-derived current state from these canonical surfaces rather than treating prompt
numbers as durable truth:

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`;
- v7.5 §8 and v8 vehicle SPECs;
- `reports/latest.md`, lane registry, subagent-progress registry, latest Codex/Claude memos;
- `.omx/research/pythagorean_exact_arithmetic_bitident_20260713.md` and its Metal receipt;
- `experiments/results/cheapen_real95_tilehalo_fp16_20260713/tile_halo_receipt.json`;
- `experiments/results/p0_sparse_adjoint_costate_vjp_20260713/measurement_receipt.json`;
- `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`;
- `src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py`;
- `src/tac/canonical_equations/segnet_state_local_bcr_adjoint_20260713.py`;
- `src/tac/canonical_equations/whole_teacher_distilled_student_20260713.py`;
- Task #494 authority-ladder implementation spec and both live inboxes.

All numerical claims below are labeled. NumPy-fp32 remains the local bit-equivalence authority;
macOS Metal is a non-promotable research signal; MPS is never score authority; exact contest CPU and
CUDA remain separate terminal axes.

## 1. Exact accumulation: the minimal algebra

### 1.1 Reorder-invariance theorem

Let `x_1,...,x_n` be operands mapped into a carrier `M`, and let `⊕` be the reduction law. Suppose:

1. `(M,⊕,0)` is a commutative monoid on **all reachable partial states**;
2. each intermediate state is represented exactly—no overflow, saturation, lossy normalization,
   undefined signed shift, or implementation-dependent division;
3. the final map `F:M→Y` is deterministic and is applied once.

Then for every permutation `π` and every parenthesization,

```text
F(x_1 ⊕ ... ⊕ x_n) = F(x_{π(1)} ⊕ ... ⊕ x_{π(n)}).
```

**DERIVED proof.** Associativity removes parenthesization dependence. Every permutation is a product
of adjacent transpositions; commutativity makes each transposition invariant. Exact representation
means the machine operation realizes `⊕` rather than a rounded surrogate. Applying the same
deterministic `F` to the unique monoid result gives identical output bytes. None of the three clauses
is cosmetic: ordinary fp32 addition violates associativity, fixed-width integer addition with
overflow no longer realizes addition over `Z`, and repeated intermediate dequantization reintroduces
rounded order dependence.

### 1.2 Reduction-cutset composition theorem

Consider a finite acyclic compute graph. Suppose its source bytes are fixed, every ordinary node is a
deterministic function of parent bytes, and every node whose schedule may reorder operands satisfies
the theorem above. Then every node is cross-process bit-identical.

**DERIVED proof.** Topologically order the graph. Sources are identical. If all parents of node `v`
are identical, an ordinary deterministic node emits identical bytes; a reordered reduction emits the
same monoid result and finalization by §1.1. Induction reaches every sink.

This is the **simple hack**: if #348 is correct that all 28 divergent tensors descend from one
duplicate-index atomic-scatter class, exactness need only replace that reduction cutset. There is no
mathematical requirement to quantize deterministic pointwise arithmetic that is not on the cutset.
The full-R n600 host receipt still owes the empirical premise that the cutset is complete.

### 1.3 Minimal signed width

If every term obeys `|q_i|≤A` and fan-in is at most `n`, then every partial sum obeys
`|Σ_I q_i|≤B=nA`. A signed `w`-bit two's-complement accumulator represents
`[-2^(w-1), 2^(w-1)-1]`, so exactness requires and is guaranteed by

```text
B ≤ 2^(w-1)-1,
w_min = 1 + bit_length(B)              for B>0,
w_min = 1                              for B=0.
```

This uses integer `bit_length`, not floating `ceil(log2(B+1))`, so the certificate remains literal
for bounds beyond binary64's exact integer range.

**MEASURED input:** `B=11,159,918` in the one-axis Q15 receipt.  
**DERIVED output:** `w_min=25`; int32 passes; headroom `(2^31-1)/B=192.428263989`.

### 1.4 Candidate number systems

| Candidate | Reorder invariant? | Exact structure | Minimal-structure verdict |
|---|---:|---|---|
| bounded fixed-point integer | yes, with range proof | `(Z,+,0)` on reachable states, one final rounding | **first build**; 25 bits on measured instance |
| Kulisch/binned superaccumulator | yes, if exact bins cover exponent range | exact fixed-point bins, canonical carry/final rounding | general-float fallback when one dyadic scale cannot certify |
| CRT / residue system | yes componentwise | product of modular commutative monoids | exact iff moduli are pairwise coprime and product `M>2B`; hardware value owed |
| posit without quire | no | rounded posit addition is non-associative | not an L70 proof; quire reduces to Kulisch-like exact accumulation |
| Kahan/Neumaier | no | sequential compensated state | improves error, does not erase schedule dependence |
| naive TwoSum/TwoProduct expansion | not by itself | exact transforms but order-shaped expansion | requires a canonical exact merge/normal form; otherwise not a proof |
| Gaussian integer | yes with two range proofs | two integer monoids | overhead for real scalar reduction; useful only if complex rotation geometry is shared |
| exact rational | mathematically yes | arbitrary-precision numerator after common-denominator normalization | bounded hardware width still needs overflow proof; denominator LCM can explode |
| tropical max-plus reduction | yes on its own graph | idempotent commutative `max`, associative `+` | decision-head/max-plus graph only; not a drop-in sum-product CNN |

For the concrete CRT construction `(4093,8191)`, the pairwise-coprime modulus product is
`33,525,763 > 22,319,836 = 2B`, so symmetric reconstruction is injective and its nominal information
width is again 25 bits. This is an algebraic existence proof, **not** a Metal/ANE speed claim. Native
int32 is likely simpler; only a measured residency/latency receipt may reverse that ordering.

Exact superaccumulation is established beyond Pact: Neal gives a practical exact summation/dot-product
design, and ReproBLAS gives order-independent reproducible reductions under explicit accumulator
contracts. These support the general-float fallback but do not override the narrower, cheaper fixed
point bound here. Sources: [Neal, 2015](https://arxiv.org/abs/1505.05571),
[Berkeley ReproBLAS report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.html).
The posit standard's quire is the same kind of exact-accumulator move rather than proof that ordinary
posit addition is associative: [Posit Standard 4.12](https://posithub.org/posit_standard4.12.pdf).

## 2. Tropical argmax: the exact result and the tempting false transfer

On `R∪{-∞}`, define

```text
a ⊕ b = max(a,b),       a ⊗ b = a+b.
```

Both laws are associative; `⊕` is commutative and idempotent. Therefore a max-reduction over
**already-identical candidate values** is order invariant. The class decision can also carry a
deterministic tie key, e.g. lexicographic `(score,-class_index)`, which remains a max over a totally
ordered carrier.

The false transfer is to say “argmax is tropical, therefore the CNN before argmax is deterministic.”
An ordinary convolution computes rounded sum-products. If Metal reorders those sums, the candidate
logits entering max may already differ. Tropical selection deterministically selects from two
different candidate sets; it does not make them equal. A full max-plus CNN is a different model
family and owes n600 decision/VJP fidelity and receiver-through-R gates. Relevant mathematical
context exists in tropical neural-network work, but it does not supply that missing equivalence:
[Tropical Geometry of Deep Neural Networks](https://arxiv.org/abs/2101.00717) and
[Tropicalizing Neural Networks](https://arxiv.org/abs/2102.06358).

### New synthesis: certified tropical stopping

The useful tropical object is a **stopping rule**, not a replacement network:

1. run a cheap low-precision or student path that emits sound intervals `[L_c,U_c]`;
2. select provisional class `a` by deterministic max;
3. stop if `L_a > max_{c≠a} U_c`;
4. otherwise refine the uncertain sample/pixel/layer or fall back to exact teacher arithmetic.

The final decision is exact wherever the interval test passes. Compute is spent only on the
uncertainty set. In a graph with global SE this does not automatically yield exact pixel-local
teacher FLOP savings; it can still gate whole-rung escalation, local students, cotangents, or a
different architecture whose dependencies remain local.

## 3. Certified argmax preservation

Let reference logits be `z`, reference winner `a=argmax_c z_c`, and sound classwise error bounds
`|z'_c-z_c|≤e_c`. Then

```text
z_a-e_a > max_{c≠a}(z_c+e_c)
```

is sufficient to preserve class `a`. It is also tight given only independent interval bounds: if
the intervals touch or overlap, admissible perturbations can create a tie or reversal. For one
uniform bound `e`, the condition reduces to

```text
margin = top1-top2 > 2e.
```

Equality is **uncertified**, not accepted by a convenient tie policy.

There are two different authorities:

- **prospective rigorous certificate:** layer/operator bounds are proved to enclose every admitted
  NumPy-fp32 result after the real R/QDQ path; this can govern unseen inputs inside its declared domain;
- **retrospective corpus certificate:** an observed maximum error on real n600 certifies only those
  measured pixels, pairs, and that exact rung/receipt. It cannot be promoted into a whole-domain IBP
  theorem.

The authority-ladder QDQ probe is designed to provide the second. True whole-network interval bounds
would need a sound IBP/linear-relaxation artifact; QA-IBP and auto_LiRPA show buildable methods, but
their existence is not evidence that Pact's bounds will be tight:
[QA-IBP](https://arxiv.org/abs/2211.16187),
[auto_LiRPA](https://proceedings.neurips.cc/paper_files/paper/2020/hash/0cbc5671ae26f67871cb914d81ef8fc1-Abstract.html).

## 4. Per-layer and per-region margin waterfill

Suppose a sound propagated logit-error bound has separable form

```text
E(b) = Σ_l a_l 2^(-b_l),
```

and a continuous cost proxy is `C(b)=Σ_l c_l b_l`. The relaxation

```text
minimize    Σ_l c_l b_l
subject to  Σ_l a_l 2^(-b_l) ≤ ε
```

has Lagrangian derivative

```text
c_l - λ ln(2) a_l 2^(-b_l) = 0.
```

For an interior solution,

```text
b_l = log2(λ ln(2) a_l/c_l)
    = log2(a_l Σ_j c_j / (ε c_l)).
```

Bit boxes require clipping and active-set re-solving. This **DERIVED KKT law** assigns more precision
where propagated sensitivity `a_l` is high and less where one bit is costly. It is not a hardware
latency verdict: bit costs are discrete, kernels have plateaus, and ANE/Metal residency can dominate
arithmetic counts. The executable allocator therefore enumerates measured `(bits,error_bound,cost)`
options per layer and retains the exact Pareto frontier; no tolerance merges near-equal certificate
states because a one-ulp lower error can be the only state inside a tight budget.

For pixel `p` with margin `m_p`, the exact uniform-error budget is `ε_p<m_p/2`. Therefore a rung
ladder has the monotone stopping rule

```text
r*(p) = lowest-cost rung r such that 2 ε_{p,r} < m_p.
```

This is the mathematical union of margin, Fisher concentration, precision allocation, and tropical
stopping. The margin field is used once to set a decision radius and again to prioritize computation.
Mixed-precision literature gives related inter-layer allocation strategies, but Pact's authority is
the exact discrete real-hardware receipt, not the paper's proxy:
[inter-layer mixed-precision optimization](https://arxiv.org/abs/2306.04879).

## 5. Separatrix sparsity after dependency closure

The deep geometry says `d_seg` changes at the codimension-one argmax separatrix, while class interiors
are decision-flat until a margin is exhausted. That does **not** imply an exact frozen-teacher sparse
forward. For demanded output support `A_L`, exact support propagates backward through the graph:

```text
A_{l-1} = Pred_l(A_l),
F(A_L) = Σ_l FLOPs_l(A_l).
```

Any nondegenerate global spatial reduction maps nonempty demand to full spatial support. The current
frozen SegNet has **MEASURED/source-inspected 23 squeeze-excite global reductions**, an exact safe
halo of **685 pixels**, receptive field **1311 pixels**, and exact source-area fraction `1.0`. Hence:

```text
exact sparse frozen-teacher forward ceiling = 1.0×.
```

The n600 receipt measures boundary area `0.047365976969` and boundary flip-mass share
`0.268038228210`. The frequently repeated “~4.7% area contains ~97% d_seg” came from an n16 advisory
subset, not a full-n600 verdict. It is forbidden here as an n600 headline.

This scoped negative leaves four live formulations:

1. **cotangent sparsity/custom sparse adjoint:** current ideal arithmetic ceiling is
   `2.208577×`, but the 4.7366%-area oracle mask has global relative-L2 input-costate error
   `0.363536` and cosine `0.954554`; dense kernels still realize `1×`;
2. **cached-SE approximate teacher:** admit only with explicit VJP/descent regret and wall-time gates;
3. **local/distilled student:** design dependencies to remain local, then use the certificate cascade;
4. **whole-rung precision escalation:** the full graph runs, but most samples may stop at a cheap rung.

`verdict_scope`: `INSTANCE × frozen tu-efficientnet_b2 SMP U-Net SegNet × exact finite crop/tile
forward`. It is not a FAMILY or PARADIGM negative.

## 6. Ordinal, Minkowski, and plasticity answer to unequal convergence

### 6.1 Recos gives a useful question, but full ranking is not the minimal certificate

Let `pi(z)` be the complete deterministic class ordering and `a(z)` its top class. Then

```text
pi(z) = pi(z')  =>  a(z) = a(z'),
a(z) = a(z')   =/=>  pi(z) = pi(z').
```

Therefore recos-style full ordinal concordance is **strictly stronger than SegNet argmax identity**.
For fixed-point admission it is not better than the strict interval winner certificate: preserving
loser-versus-loser order spends proof budget on distinctions that `d_seg` never observes. Verdict:
`INERT-CURIO` for the fixed-point certificate **FORMULATION**, while still a useful diagnostic.

For training, the minimal decision debt is instead

```text
ell_ord(z,y) = [0 - (z_y - max_{c!=y} z_c)]_+.
```

Zero is the **DERIVED** decision boundary, not a guessed robustness margin. This loss concentrates
gradient on target-versus-strongest-rival debt and implicitly mines undecided pixels. It can therefore
attack the exact-value diffusion mechanism hypothesized for rare classes, but it does not by itself
correct area imbalance or geometric thin-structure erasure. The existing typed
`--seg-loss margin_hinge --margin-target-end 0.0` surface supplies the first A/B; the control is CE,
both arms must load identical EMA bytes and differ in no other treatment. The probe reports all/hard/
easy Theil-Sen decline per update and per wall second for all five authority classes. No real-n600
receipt exists here, so causality is **OWED**, not inferred from the off-domain paper.

Source: [Ai, *Beyond Cosine Similarity*](https://arxiv.org/abs/2602.05266). Its experiments are
semantic-text similarity, so transfer to witness training remains a hypothesis.

### 6.2 Minkowski content identifies the geometric complement—and falsifies the Wulff shortcut

For a sufficiently regular planar body, the isotropic parallel-set expansion has the form

```text
Area(K + epsilon B2) = Area(K) + epsilon Per(K) + pi epsilon^2 chi(K)
```

under the usual topology/regularity conventions (for a convex body, `chi=1`). More generally the
tube coefficients are curvature integrals. This unifies perimeter and curvature geometry, but one
must not say the `epsilon^2` coefficient *is* mean-curvature-flow velocity: MCF follows from the
**first variation of perimeter**, `V_n = -mu kappa` under a sign/mobility convention. The adjacent
tube coefficient and the gradient-flow velocity share curvature; they are not literally identical.

Replacing `B2` by a convex body `W` yields anisotropic perimeter

```text
P_W(K) = integral_{boundary K} h_W(n(x)) dH^1(x),
```

where the support function depends on interface normal `n`. This exposes the critical implementation
fact: the existing `sigma_cc'` is a **scalar class-pair matrix** multiplying Euclidean interface
length. It balances interfaces by class pair and is perimeter-weighted rather than area-weighted,
but it has no spatial-normal dependence and is therefore isotropic, not a Wulff/Finsler law. A true
class-pair Wulff implementation needs `sigma_cc'(n)` or a convex body `W_cc'` for each pair.

For the usual multiphase pairwise-perimeter relaxation, surface tensions must obey every triangle
inequality

```text
sigma_ik <= sigma_ij + sigma_jk.
```

Otherwise a vanishing layer of phase `j` lowers the `i|k` interface cost (“wetting”), and the relaxed
cost is the shortest-path metric closure. The current fitted matrix fails exactly this static gate:

```text
sigma(Lane,MyCar) = 1.764344
sigma(Lane,Undrivable) + sigma(Undrivable,MyCar)
                      = 0.7381986449045815 + 1.0
                      = 1.7381986449045815.
```

Thus its Lane-MyCar edge relaxes downward by about `0.0261453551`. Verdict:
`BLOCKED_TRIANGLE_VIOLATION` for the **FITTED MATRIX INSTANCE**. Even a triangle-valid matrix is only
statically admissible; proving that Pact's discrete diffuse-interface/trainer energy Gamma-converges
also needs the actual potential, scaling, compactness, liminf, and recovery-sequence arguments. The
current scalar sigma is **not Gamma-limit-proven**. The matched n600 A/B must use an all-ones control
and a distinct preregistered metric-admissible treatment, then report per-class all/hard/easy rates.

Primary mathematical context: [Baldo, *Minimal interface criterion for phase transitions in mixtures of Cahn-Hilliard fluids*](https://ems.press/journals/aihpc/articles/4077580).

### 6.3 Weight decay is a plasticity hypothesis, not transferred evidence

Han, Bordt, Zhang, and Kakade report that stronger weight decay during LM pretraining can improve
subsequent learning, alongside more linearly separable representations, lower attention pseudo-rank,
and a smaller train/validation gap. Those are **MEASURED in their LM regime**; the proposed mechanisms
are correlational, and a single-video witness INR is a memorization/curriculum regime. Transfer is
therefore `SPECULATIVE × INSTANCE`, not a settled lever.

The decisive Pact A/B must pin two preregistered weight-decay values without guessing them here,
hold source/data/init/optimizer/curriculum bytes fixed, keep one effective-rank definition throughout,
and measure Lane/Movable versus common-class convergence plus exact archive bytes/SHA. The trainer has
a raw parser surface but no admitted typed DSL/resume policy for this experiment, so launch remains
**BLOCKED ON TYPED REGISTRATION**. Source: [Han et al., *Weight Decay Improves Language Model Plasticity*](https://arxiv.org/abs/2602.11137).

## 7. Deep-math object → throughput lever → confirming measurement

| Rank | Deep-math object | Throughput lever implied | Confirming measurement owed/held |
|---:|---|---|---|
| 1 | argmax interval radius `L_a>max U_c` | certificate-gated fixed-point SegNet forward on GPU/ANE; exact fallback only for the uncertified set | n600 heldout/full flip counts, classwise error bounds, uncertified fraction by rung, worst-pair, exact digest, synchronized integer-kernel wall time and residency |
| 2 | exact commutative reduction cutset | implement SegNet convolution/reduction accumulators in bounded integer/quire form and finalize once; retain deterministic nodes | per-op NumPy-int parity/range, full n600 logits/argmax, cross-process hashes, actual GPU/ANE placement; **HELD only for R-adjoint:** one-axis Metal 10 float hashes vs one integer hash |
| 3 | Fisher trace monotone in top-two margin (`ρ=0.978` n96 advisory) | use one margin field as decision radius and precision/acquisition priority | real n600 margin-stratified rung occupancy and charged latency; do not relabel cached margins as input-gradient saliency |
| 4 | whole-teacher decision quotient | distill the four-dimensional centered-logit quotient as a forward alternative; retain full input-VJP as a separate training gate | real n600 aggregate/worst-pair quotient and argmax fidelity, then separately VJP fidelity, anchor cadence, and fully charged timing |
| 5 | separatrix/dependency geometry | whole-rung precision escalation or local student—**not exact sparse teacher forward** | n600 certified rung occupancy; any approximate/local formulation separately gates argmax, VJP if trained, and wall time |
| 6 | PoseNet continuous receiver geometry | fixed-point/surrogate continuous forward with an interval or norm bound, not an argmax certificate | both frames, first-six output max-abs/MSE, `sqrt(10*d_pose)` debt, n600 worst pair, synchronized wall time; targets measured n96 `0.226` verdict share |
| 7 | state-local BCR conditional rank law | training-only state-conditioned compressed adjoint if far-block ranks remain bounded | heldout per-block spectra, drift radius, build amortization, NumPy-fp32 VJP fidelity and matched-device timing; not authority-verdict throughput |
| 8 | exact integer R-adjoint / deterministic GPU decode | training reproducibility and payload-TTO; not the measured forward-only authority-verdict bottleneck | full four-stage R, 1,200 frames, cross-process bytes, NumPy parity and latency; no score transfer from Metal |
| 9 | se(3) screw/covariance factorization through `(ξ,R)` | compute forward state once and warp/reuse only if equivariance holds | equivariance residual and occlusion/movable stratification; admit only if `warp_cost + refresh_cost/K < teacher_cost` |

### Important covariance caveat

Pose covariance does not make the frozen SegNet exactly SE(3)-equivariant. Occlusion, movable objects,
resampling, padding, and global SE can break compute-once-warp. The break-even formula is a cost
identity, not a fidelity proof. A failed warp at one cadence is `INSTANCE` or `FORMULATION`, never a
family kill.

## 8. Ranked execution queue

### Measure next — single highest EV within throughput

Run the real n600 calibrated fixed-point SegNet forward ladder. For every rung record:

```text
calibration / heldout / full-n600 pair custody
aggregate and worst-pair argmax flips
strict interval-certified and uncertified fractions
ordered argmax digest
maximum/RMSE logit error with bound_kind
synchronized integer-kernel latency and proved GPU/ANE residency
```

QDQ/fp32 accumulation answers feasibility only; the throughput verdict requires an actual integer-MAC
backend and placement receipt. The measured comparator is the one-thread CPU-torch n96 per-pair
SegNet wall; the 6.21-minute n600 forward-only authority wall is a declared linear projection.

### Direct score-relevant measurements before more MEANS work

1. **CE versus zero-margin winner/rival hinge:** identical EMA/seed/order/optimizer/curriculum/data;
   real n600 all/hard/easy per-class rates versus updates and wall time.
2. **All-ones versus distinct metric-admissible class-pair sigma:** do not use the fitted matrix
   unrelaxed; report triangle proof plus convergence and boundary-stratum effects.
3. **Weight decay plasticity:** only after typed DSL/preregistration; stable rank definition and exact
   archive SHA/bytes are mandatory.

### Throughput build/measure order

1. **Fixed-point/QDQ forward ladder plus interval postprocessor** — establishes the lowest argmax-safe
   precision but makes no integer speed claim.
2. **Actual fixed-point SegNet GPU/ANE backend** — exact accumulator parity, placement, and timing on
   the admitted rung; this is the throughput end-state.
3. **Tropical deterministic class head** — cheap exact max/tie selection on already-certified logits;
   it does not excuse upstream sum-product error.
4. **Exact-forward or distilled-forward comparator** — gate centered decision quotient, aggregate and
   worst-pair argmax, and charged n600 wall time. VJP fidelity is an additional training gate.
5. **PoseNet continuous fixed-point/surrogate ladder** — separately attack the measured n96 `0.226` share;
   argmax evidence cannot transfer.
6. **Discrete per-layer precision manifest** — each option must carry bit width, a sound or explicitly
   retrospective error bound, synchronized measured cost, substrate/residency, and authority scope.
7. **Full-R integer adjoint host receipt** — complete it for training reproducibility, but do not report
   it as an authority-verdict throughput win.

## 9. Built probes and resumability contract

`tools/probe_throughput_frontier_math.py` consumes sealed receipts and persists:

```text
stage_00_custody.json
stage_01_exact_number_system.json
stage_02_argmax_certificate.json
stage_03_discrete_waterfill.json
stage_04_support_closure.json
measurement_receipt.json
```

Every stage is atomically written, content-hashed, and bound to a canonical input/tool fingerprint.
Resume rejects fingerprint, stage, payload-hash, or deterministic-payload drift. The current-byte
static run landed under
`experiments/results/throughput_frontier_math_static_20260714_v2/c0438164e39c2805f456a77f70c7584564f3dff5c1fd2ada1a88d94e244ad3eb/`
with status `OWED_FIXEDPOINT_FORWARD_RECEIPT`, which is correct: both fresh `*_current.json` host
receipts are absent. Its byte-identical durable mirror is
`.omx/research/throughput_frontier_math_static_receipt_20260714.json`, SHA-256
`25282f9392e44bfa96ddbfb58f0bc62cf8bc37b4d15726e9a5a2eb7d29d546d0`. Synthetic NumPy cases verify
code only and are explicitly not scientific evidence.

The four zero-launch host commands are:

```bash
tools/run_throughput_frontier_math_host.command
tools/run_ordinal_perclass_convergence_host.command CE.json MARGIN.json OUT.json
tools/run_sigma_ccprime_gamma_limit_host.command
tools/run_weight_decay_plasticity_ab_host.command PREREG.json CONTROL.json TREATMENT.json OUT.json
```

Optional exact-path overrides are `PACT_FIXEDPOINT_RECEIPT`, `PACT_FULL_R_RECEIPT`, and
`PACT_THROUGHPUT_MATH_OUT`. Missing upstream receipts remain blocked/owed; the probe never substitutes
toy data, zeros, MPS, or inferred cross-axis equivalence.

## 10. Triality and apparatus wiring

- **Equations:** six isolated canonical equations register the reduction-cutset theorem, strict
  interval argmax certificate, precision waterfill, top-1 ordinal minimality, multiphase sigma metric
  closure, and exact dependency-closure FLOP ceiling.
- **DAG:** standalone FEED `throughput_frontier_math_DAG_FEED_20260714T015118Z.md`; main may merge it
  into the hot shared pursuit DAG after collision review.
- **DSL:** the ordinal A/B reuses the existing typed `seg_loss=margin_hinge` and zero endpoint. Sigma
  reuses the existing typed scalar-matrix surface but must refuse nonmetric treatments. Weight decay
  has no admitted typed experiment policy and remains blocked. The authority backend stays default-OFF.
- **Sensitivity map:** margin becomes the certificate radius/priority field; cached margin is not
  silently treated as gradient saliency.
- **Pareto constraint:** exact discrete `(error bound, measured cost)` frontier; no continuous-bit or
  FLOP proxy may claim hardware victory.
- **Bit allocator:** consumes admitted precision options and selects the least-cost certified row.
- **Autopilot:** cannot dispatch or promote while receipt status is incomplete; terminal CPU/CUDA
  fallback remains exact and axis-separated.
- **Continual learning:** current static receipt records missing host inputs and the scope-corrected,
  still-unconfirmed 95% attribution; host measurements append rather than overwrite.
- **Probe disambiguator:** rigorous interval bound and retrospective n600 bound are distinct modes,
  not one ambiguous “certificate.”

## 11. Round-1 adversarial review

The review re-derived every load-bearing number and found these prevented false claims:

1. Replaced floating `ceil(log2(B+1))` with integer `bit_length` so the minimal-width theorem is exact.
2. Removed tolerance from Pareto pruning; an ulp-smaller error state can be uniquely feasible.
3. Made argmax separation strict; ties are not certificates.
4. Rejected Kahan/Neumaier, naive EFT, posit-without-quire, and bounded rational arithmetic as automatic
   reorder-invariance proofs.
5. Scoped tropical max to already-identical candidates and true max-plus graphs.
6. Separated reduction bit-identity from NumPy-fp32 equivalence and argmax preservation.
7. Corrected the n16 `~97%` annulus shorthand rather than promoting it to n600.
8. Charged 23 global SE blocks and halo 685 before claiming spatial FLOP savings.
9. Consumed main's later timer and retracted backward-first routing: the authority bottleneck is
   forward-only; n96 timing is measured, while `372.6 s = 6.21 min` at n600 is explicitly derived
   by the canonical linear per-pair wall model.
10. Kept the local static receipt incomplete instead of manufacturing a Metal/fixedpoint verdict.
11. Proved full ordinal ranking overconstrains argmax by preserving irrelevant loser order; retained
    strict interval top-1 as the fixed-point certificate.
12. Refused to call scalar class-pair sigma Wulff anisotropy, found the fitted Lane-MyCar triangle
    violation, and separated static metric admissibility from an actual Gamma-limit proof.
13. Corrected the Steiner/MCF identification: curvature appears in adjacent geometry, while MCF is
    generated by the first variation of perimeter.
14. Scoped weight-decay transfer from LM pretraining to single-video INR as speculative and blocked
    launch until a typed preregistered treatment exists.

## Pointer delta honesty

Exactly zero. This apparatus matters only because it may make the score-moving V9·CGauge witness
runs fast enough to reach sub-0.15. It does not itself change `d_seg`, `d_pose`, archive bytes, or any
contest pointer.
