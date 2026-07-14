# POINTER STATUS — UNMOVED

- Submit-ready pointer: `0.19108` (`0.1910828242` full recalled pointer), unchanged.
- Defensive bank: `0.18804` (`0.1880443980` full recalled bank), unchanged.
- This unit produced no contest score, archive, CPU-Linux/CUDA evidence, promotion, GPU launch, or paid dispatch.
- Measurement axis: `[macOS CPU-torch through-R advisory; NumPy aggregation; non-promotable]`.

# V9 CGauge gauge / symmetry / homotopy warm-start — Codex findings

Date: 2026-07-14  
Lane: `warmstart_gauge_symmetry_homotopy`  
Receipt: `.omx/research/v9_cgauge_symmetry_homotopy_n600_receipt_20260714.json`  
Receipt SHA-256: `60dd6a4837706d100932416cf8fdf77fce0e7c171b1ef58fd3f1154021428308`  
Source-closed run: `experiments/results/v9_cgauge_symmetry_homotopy_n600_20260714/v9_ema_best_n600_threads6_batch32_r2`  
Checkpoint: `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz`

## Executive verdict

**D37 FIRED at the required V9 EMA-best n600 surface and the margin field is not a sufficient
statistic for the realized flip under this formulation.** On `2,551,382` unlike-class boundary
pixels, adding the directed class-edge refinement `C` to `(M,Qxi)` reduced held-out conditional
codelength by `467,373.9089` gross bits. After the preregistered `10,342 B` incremental table charge,
the net was `384,637.9089 bits`, with pair-bootstrap 95% interval
`[373,674.7586, 395,236.5487] bits`. Every one of the five outer folds selected
`margin_bins=16, xi_bins=2`.

The scoped conclusion is:

`RESIDUAL_NON_GAUGE_STRUCTURE_DETECTED__M_NOT_SUFFICIENT`

with `verdict_scope=FORMULATION x V9_EMA_BEST_N600_EMPIRICAL_SURFACE`. This is not an exact mutual
information identity, not a LieFlow-family verdict, and not yet a usable codec gain: `C` is an
ASSUMED directed GT boundary class pair, and its sequence is not charged. A live consumer must
derive `C` at the receiver or jointly code it before spending the measured conditional gain.

The phase-aware contrast remained positive but narrow after its larger table:
`12,241.2100 bits`, 95% interval `[957.5520, 23,180.7500]`, with `56,552 B` overhead. The
normalized-margin/velocity diagnostic had positive gross fit but negative net after its
`175,262 B` table, so that diagnostic does not reject margin sufficiency. It is explicitly a
diagnostic because its velocity is not a scorer Jacobian.

## Authority replay and holistic receiver measurements

The final post-review run reproduced the selector exactly over all `117,964,800` SegNet cells:

| Quantity | MEASURED value |
|---|---:|
| wrong pixels | `4,107,576` |
| total `d_seg` | `0.03482035319010417` |
| D37 boundary pixels | `2,551,382` |
| D37 boundary flips | `1,066,627` |
| boundary flip rate | `0.4180585267121897` |
| bounded pair-state bytes | `1,502,287` across 600 files; max file `2,629 B` |

Per-GT-class receiver decomposition from that same realized map:

| GT class | wrong / GT pixels | MEASURED within-class `d_seg` | share of flip mass |
|---|---:|---:|---:|
| Road | `3,032,636 / 27,407,046` | `0.11065169153946762` | `0.7383030770` |
| Lane | `265,741 / 690,639` | `0.38477554844137096` | `0.0646953337` |
| Undrivable | `773,770 / 58,413,281` | `0.013246473862682016` | `0.1883763076` |
| Movable | `9,910 / 1,460,325` | `0.006786160614931608` | `0.0024126151` |
| MyCar | `25,519 / 29,993,509` | `0.0008508174218628437` | `0.0062126665` |

The existing registered sampling-phase anchor `d_gauge=0.005318` was consumed, not remeasured.
Therefore `d_cov=max(0,d_total-d_gauge)=0.029502353190104167` is **DERIVED**. This is a
`SCALAR_ONLY_NOT_POINTWISE` compatibility decomposition; it does not identify which pixels are
covariant versus gauge debt.

## Per-paper deep warm-starts

### 1. LieFlow — discovering the empirical gauge refinement

**Deep-read theorem.** LieFlow models a distribution on a hypothesis Lie group by conditional flow
matching along the left-invariant interpolation `g_t=g exp(t log(g^{-1}g_1))`. Proposition 4.1 and
Theorem B.4 give subgroup-supported sampling only under ideal convergence/support-consistency and
global-generation hypotheses; Appendix A makes the infinitesimal-action identifiability conditional
on a trivial infinitesimal stabilizer (otherwise the action is identified only modulo the
stabilizer). Source: [LieFlow, arXiv:2512.20043](https://arxiv.org/html/2512.20043).

**Divergence fork.** V9 is not a globally free `SE(3)` action. Its frozen information space is
stratified by class/event cells, has nontrivial stabilizers, and includes discontinuous uint8/resize/
argmax crossings. A neural LieFlow fit is also not justified by the single-clip `$0` contract.

**Re-derived witness surface.** Replace asserted global group discovery with a finite support test:
within `(M,Qxi)` cells, ask whether the permutation group remains invariant after class refinement.
Pair-blocked conditional codelength supplies the observable quotient of the theorem's support claim.

**Built / measured.** `crossfit_conditional_codelength` in
`src/tac/boundary_math/gauge_symmetry_homotopy_20260714.py`; fired by the n600 probe. The positive net
interval supports only the refined product of within-`(M,Qxi,C)` fiber permutations. Verdict scope:
`FORMULATION x EMPIRICAL_SURFACE`; LieFlow family remains open.

### 2. FINO — metadata dependence as the anti-entanglement gate

**Deep-read method.** FINO uses a student/EMA-teacher objective: discrete metadata prototypes with
InfoNCE (Eq. 1) and EMA prototype updates (Eq. 2), continuous-metadata prediction (Eq. 3), and a
DINO+iBOT+SIGReg+metadata total objective (Eq. 4). Appendix A.3 warns that entangled metadata makes a
binary informative/spurious assignment unsafe and requires a dependence table first. Appendix A.4
balances branches with EMA gradient norms; its adversarial branch uses the DANN ramp
`2/(1+exp(-10s/S))-1`. Source: [FINO, arXiv:2606.05107](https://arxiv.org/abs/2606.05107).

**Divergence fork.** FINO adapts a vision backbone using auxiliary metadata; Pact freezes SegNet and
PoseNet. Updating the scorer or importing FINO's full loss would violate the information space.

**Re-derived witness surface.** Retain only FINO's dependence-table rule: test `C` as a positive
refinement of `(M,Qxi)` before suppressing or quotienting it. D37 is that frozen-scorer dependence
gate. **Measured verdict:** `C` is informative on the tested V9 surface, so a class-blind quotient is
rejected. No FINO training was performed; family verdict remains open.

### 3. Weyl symmetry / Noether current — exact group versus charged events

**Deep-read theorem.** For a frozen statistic `S`, the full exact symmetry is the fiber permutation
group `Aut_X(S)=prod_s Sym(S^{-1}(s))`; named geometric factors generally overlap and form a
stratified groupoid rather than a global direct product. Noether's first theorem yields a conserved
current only from an executable invariant action and its conjugate momentum, not from a zero mode or
an observed stable scalar. Source: [Weyl, *Symmetry*](https://www.nku.edu/~longa/classes/mat115/days/resources/docs/Symmetry.pdf).

**Divergence fork.** V9 declares a covariant trunk but has no executable affine-Legendre transform,
action momentum, transformed-pair equality test, or content-bound covariance receipt. The only clean
candidate current is the transport subaction, conditionally
`pi_b=W_b(D_t^xi phi_b-a_b)`, `Q_b=int Gamma_b pi_b ds`, with covariant conservation only between
typed events.

**Built / measured.** The new module includes an exact discrete continuity checker plus optional MLX
parity. Feeding per-class flip mass with zero flux produced `max_abs_residual=0.0852762858` over
`2,995` off-event cells. That is only `OBSERVABILITY_PROXY_NOT_NOETHER_CHARGE`; the verdict is
`NO_NOETHER_VERDICT_ACTION_MOMENTUM_UNTYPED`. V9 covariance status remains
`IMPLEMENTATION_CUSTODY_GAP_ONLY`, never a CGauge formulation/family negative.

### 4. Conditional probability / homotopy / Lie — D37

**Deep-read law.** The marked decomposition is
`H(X,W'|C)=H(X|C)+H(E|X,C)+H(Phi|E,X,C)+H(Delta^E|Phi,E,X,C)`; topology, chart/stabilizer changes,
and receiver/lattice crossings are separate event marks. An `SE(3)` label alone grants no zero-rate
coordinate.

**Divergence fork.** The predecessor D37 used an epoch-50 baseline and assumed contexts; it could not
transfer to V9. The warm-start bound the V9 selector, checkpoint, taper, IPE, R, scorer, GT cache,
thread law, and all 600 pairs.

**Fired result.** The base D37 result above rejects `M` sufficiency under the operational
`C=directed unlike-neighbour GT class pair`. The phase-aware result also remains barely positive
after overhead. The sequence-cost blocker remains explicit, so this is structural evidence for a
richer basis/conditional grammar, not a claimed archive saving.

### 5. Bousfield localization / D38 — local data do not automatically create a cheap section

**Deep-read theorems.** Barwick's Theorem 2.11 constructs left Bousfield localization only for a
left-proper combinatorial model category. Theorem 2.42 localizes right sections of an already-given
right Quillen presheaf so fibrant sections are homotopy cartesian. Theorem 3.36 constructs local
projective/injective enriched model structures after the site, presheaf and covers are supplied; it
does not infer those objects or a receiver. Source: [Barwick, arXiv:0708.2067](https://arxiv.org/pdf/0708.2067).

**Divergence fork.** Current V9/v8 artifacts do not type a model category, edge-tube/junction cover,
restriction functors, coefficient stack, or changing-isotropy band. Therefore no current Cech or
`H^2` obstruction and no numeric twist rate follows.

**Built / fired.** The finite checker validates pairwise restrictions and triple cocycles. On 600
realized flip maps, four exact quadrant restrictions agreed across `19,660,800` overlap points:
`EXACT_SECTION_GLUES`. Scope is only `INSTANCE x EXACT_ARRAY_RESTRICTION`; the locals were exact
restrictions of an already-global array. `GLOBAL_RATE_DESCENT_NOT_TYPED` remains the rate verdict.
The surviving conditional law is: first require nonempty
`G_U(x)=hofib_x(Desc_U(F)->prod_i F(U_i))`; only then can a chosen filler `Theta_U` carry rate
`H(Theta_U|q_H,A_U,U,x,public)`.

### 6. MuonH / Manifold Muon — preserve the functional polar factor

**Deep-read method.** Modular Manifolds assigns a manifold and norm per module; the Stiefel spectral
step solves the tangent-constrained linear minimization problem and retracts. Ambient Muon followed
by polar projection is not definitionally the same because the ambient direction has a normal
component. MuonH fixes a Frobenius radius and normalizes/reprojects the base update. Sources:
[Modular Manifolds](https://thinkingmachines.ai/blog/modular-manifolds/) and
[MuonH, arXiv:2606.16899](https://arxiv.org/abs/2606.16899).

**Divergence fork / measured predecessor.** Raw unit-Stiefel projection changes live `film.weight`
by about `0.8823 ||W||_F`; the function-preserving chart is `W=QH0`, freeze `H0`, optimize `Q`.
Generic full-matrix MuonH is a `FORMULATION` no-go while radius is functional. The existing manifold
lane owns the implementation and Hessian-head composition; this unit did not duplicate it or claim a
new n600 treatment result.

### 7. Natural geometry per slot / de Sitter

**Deep-read fork.** Geometry must follow each block's measured invariances, not one decorative
manifold: inverse-depth companding for the coordinate slot; locally flat/Fisher head geometry;
Newton-Cartan/advection geometry for transport; polar Stiefel geometry only for the FiLM factor.
Literal de Sitter constant-positive-curvature geometry has no measured premise.

**Measured predecessor consumed.** The bounded compander comparison reported inverse-depth JS
`0.069943`, versus log `0.160440` and uniform `0.248188`; it still lacks the ground class-pair and
receiver A/B. Curvature is not identifiable from 29 one-dimensional curves. Verdict: literal de
Sitter `FORMULATION` no-go; slot-specific geometry family open. No duplicate module was built.

### 8. Newstead infinite descent — quotient rate needs a complete paid fiber

**Deep-read theorem.** A semantic quotient supplies `D/~_U,D ~= U(D)` and an ideal semantic rate
`H(U(W))` only when the decoder/receiver fiber is complete. Infinite descent certifies termination
only over an explicit well-founded carrier; it is not an optimizer incantation. Source:
[Newstead, *An Infinite Descent into Pure Mathematics*, v0.7](https://cnewstead.codeberg.page/infdesc/infdesc_v0.7.pdf).

**Divergence fork.** V9's finite evaluator setoid does not automatically provide a legal receiver
section. An incomplete atlas leaves a real term `H(q_G(W)|U(W))`; class boundaries are cycles, not
independent flip confetti.

**Witness surface / verdict.** D37 identifies a nontrivial class refinement inside the margin fiber,
but `C` is not receiver-derived or charged. Thus the quotient is structurally informative and
operationally incomplete. Scope: `FORMULATION x CURRENT_SECTION`; no family negative and no bytes.

### 9. Garrett algebra — split extensions, orbits and strata

**Deep-read theorem.** A semidirect product is a split extension only after the action, normal
subgroup and section are typed. Orbit counting gives a combinatorial cardinality, not a receiver or
byte law. An arbitrary group action yields an order-reversing Galois connection; anti-isomorphism is
restricted to closure-fixed subgroups/subpayloads. Source:
[Garrett, *Abstract Algebra*](https://www-users.cse.umn.edu/~garrett/m/algebra/Whole.pdf).

**Divergence fork.** Current partition strata have no globally typed action/normality, and five
classes are multiplicities, not five proved covariance irreps. The section-invariant twist variable
is the choice class `Theta`, not a raw section-dependent cocycle.

**Witness surface / verdict.** D37 supplies the first finite evidence that the `(M,Qxi)` orbit is too
coarse; D38 supplies only tautological exact restrictions. The conditional rate candidate remains
`H(Theta|q_H,A,public)` after receiver typing. Global Galois anti-isomorphism is a `FORMULATION`
no-go; closure-fixed per-stratum machinery remains open.

### 10. Chiral tube algebra / defect network

**Deep-read method.** The source constructs chiral tube algebras for topological defect lines,
twisted modules and finite gauging; those CFT statements do not make image-boundary residuals into
topological sectors. Source: [Benjamin, Lam & Luo, arXiv:2607.07786](https://arxiv.org/abs/2607.07786).

**Divergence fork / measured predecessor.** The existing exact integer transform measured `PHAS1`
at `1,010,237 B` and `DTUB1` at `1,003,855 B`, but the mechanism streams worsened from `993,897 B`
to `996,246 B` (`+2,349 B`); the apparent `6,382 B` win came from generic header deduplication.
The `Z2` quotient was worse and zero modes covered `3.9844%`. Scope: defect-network formulation
no-go only; tube-algebra family open. The existing receiver path was inert, so no score or V9
promotion transfers here.

## Signal x EV ranking after the warm-start

1. **D37 class-refinement / LieFlow-FINO factor test — HIGH:** positive pair-blocked net interval on
   exact V9 EMA-best; next proof is receiver-derived or jointly charged `C`, then packed parse-back.
2. **Typed V9 gauge transform / Noether action custody — HIGH:** current declarative CGauge lacks the
   executable transform-pair equality receipt; this blocks any real covariance-current claim.
3. **D38 finite receiver complex — MEDIUM-HIGH:** type edge-tube/junction covers, restrictions,
   isotropy bands and filler bytes; current exact-restriction pass is intentionally tautological.
4. **Newstead/Garrett receiver section and twist choice — MEDIUM:** mathematically sharp rate target,
   but no legal paid section or archive A/B yet.
5. **Function-preserving polar Manifold-Muon — MEDIUM:** correct optimizer geometry exists in its
   owner lane; requires a treatment n600 rather than another derivation.
6. **Slot-specific compander/advection geometry — MEDIUM:** strong proxy ordering, receiver A/B owed.
7. **Tube/defect formulation — LOW now:** measured mechanism loss; reformulate only with a new
   receiver-active sector code rather than header effects.

## Round-1 adversarial review

1. The first inherited replay helper silently omitted the live run's `--dseg-aware-taper` and
   `--render-aa ipe`. The probe now reconstructs both from their hash-bound canonical modules before
   scoring. No result was admitted before this repair.
2. A one-thread batch-4 replay produced `4,107,574` wrong pixels and a one-thread batch-32 replay
   produced `4,107,579`; both were refused, with no tolerance. The selector run predates the
   2026-07-13 one-thread standard and used the contemporaneous six-thread default. Explicit
   six-thread/batch-32 custody reproduced exactly `4,107,576`, confirming the historical geometry.
3. The first exact six-thread pass exposed spurious macOS Accelerate floating-point flags in a
   finite PCA matrix product. Operands/results were finite and `einsum` agreed within `8.9e-16`, but
   the kernel was replaced with an explicit contraction and a strict `np.errstate(all="raise")`
   regression. The entire n600 receipt was rerun after the edit and repeated the scientific values.
4. The approved SSD was readable but sandbox-denied writes. The probe refused a local full-map
   cache and used a capped, bit-packed, resumable `1.50 MB` pair-state surface instead.
5. D37's `C`, `Phi`, `Qxi` quantization and velocity are explicitly ASSUMED operationalizations.
   Only `F`, `M`, n600 `d_seg`, and the receiver/scorer custody are measured. The conditional result
   may not be promoted to bytes until the class-edge context is receiver-derived or charged.

Round-1 code verification: Ruff clean, `py_compile` clean, `9 passed, 1 skipped` (MLX parity skips
without an exposed Metal device). Three clean post-fix passes and serializer outcome are recorded in
the session summary.

Artifact custody before serializer:

| New file | SHA-256 |
|---|---|
| `.omx/research/warmstart_gauge_symmetry_homotopy_20260714_BUILD_SPEC.md` | `0b39dbb28e1e2cafe77a6964bfe4d30fe32285e0b4d246746edd3841b99f0118` |
| `src/tac/boundary_math/gauge_symmetry_homotopy_20260714.py` | `7bfa4ec64eec74c0e4ce7e87a82882c6eb130bde3038c32bd3cb109252664e68` |
| `tools/probe_v9_cgauge_symmetry_homotopy_n600_20260714.py` | `2b211cb5a5868ad699916af1fb7f271b63cb98c70b01e44557a4c21b8241347b` |
| `tests/test_gauge_symmetry_homotopy_20260714.py` | `21f885a0790a5b24c8e560d69c15091137604b54548c70f12e425ae2ed85af01` |
| `.omx/research/v9_cgauge_symmetry_homotopy_n600_receipt_20260714.json` | `60dd6a4837706d100932416cf8fdf77fce0e7c171b1ef58fd3f1154021428308` |
| `.omx/research/warmstart_gauge_symmetry_homotopy_DAG_FEED_20260714.md` | `3fb7f029a7b73b6b0fb0dde0c1e6c912009bbbaaf7ef54783085f02731fc6c08` |

Serializer outcome: **BLOCKED by managed Git sandbox**. The canonical serializer's `git add`
failed with `rc=128`: `unable to create temporary file: Operation not permitted`; no direct-Git
fallback was attempted. The terminal handoff lists the final memo/session hashes in addition to the
implementation allowlist above.

## Held V9 integration request — exclusive provenance owner only

No hot DSL, equation, preflight, trainer, autoconfig, config, or provenance-bijection file was edited.
The exact requested integration is:

- factory: `GaugeSymmetryHomotopyProbePolicy.from_receipt(path, sha256)`;
- DSL: one default-OFF observational receipt path/hash lever, not a training-loss actuator;
- held LawRef: `v9_empirical_gauge_refinement_d37_v1`;
- required inputs: selector/checkpoint/GT/tool/module/R/scorer/thread hashes; exact n600 selector
  equality; pair-blocked estimator schema; charged table; explicit `C` custody; `score_claim=false`;
- consumer: the V9 asynchronous/local frozen-SegNet verdict-forward audit plus the provenance
  bijection; never the gradient backward path;
- output: `gauge_refinement_required=true`, supported group
  `prod Sym(fiber(M,Qxi,C))`, and a hard blocker until `C` is receiver-derived or jointly charged;
- acceptance: whole-V9 strict source closure must be green. Current fleet information says the
  exclusive provenance scientific-declaration seal is red, so status is
  `V9_INTEGRATION_BLOCKED_EXCLUSIVE_PROVENANCE_OWNER`.

Two further held equations are specifications, not landed claims:

1. `v9_gauge_covariance_pair_receipt_v1`: a typed transform pair on `(W,R,xi)` with exact pre/post
   action/divergence equality and content-bound chart custody. Until it exists, V9 covariance is
   `IMPLEMENTATION_CUSTODY_GAP_ONLY`.
2. `v9_receiver_descent_section_cost_v1`: require a typed cover/presheaf/restriction/isotropy band;
   empty homotopy fiber means infeasible, while a nonempty filler choice may carry
   `H(Theta_U|q_H,A_U,U,x,public)` plus exact packed bytes.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; the v7.5 and v8 canonical specs; operating manual and program; top-10
Claude memory entries; canonical lane/subagent/task/frontier/cost-band/probe/council surfaces; all
live inbox directives through the final checkpoint; prior `*.last.txt` and DAG FEEDs for Weyl,
conditional probability/homotopy/Lie, Bousfield, MuonH/Manifold-Muon, manifold slots, Newstead,
Garrett and defect/tube work; the predecessor D37 receipt; V9 selector/checkpoint/launch/log; frozen
n600 GT cache; canonical render/R/scorer/taper/IPE/thread-law sources; LieFlow, FINO, Weyl, Barwick,
Modular Manifolds, MuonH, Newstead, Garrett and Chiral Tube Algebras primary sources. No paid,
provider, GPU, MPS-score, live-run mutation, or upstream evaluator store was used.

## HISTORICAL_PROVENANCE

This memo is append-only. The epoch-50 D37 receipt remains historical and is not overwritten or
promoted. The two one-thread refusal receipts and the first exact six-thread receipt are preserved;
the delivered authority is the post-review `...threads6_batch32_r2` receipt above.
