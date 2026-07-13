# Barwick left Bousfield localization: deep adversarial read for the witness program

**Date:** 2026-07-13  
**Reader:** `bousfield_deep_reader`  
**Mode:** DESIGN / GROUNDING (MEANS), read-only analysis apart from this memo  
**Primary source:** Clark Barwick, *On (Enriched) Left Bousfield Localization of Model Categories*, arXiv:0708.2067v2 — [abstract](https://arxiv.org/abs/0708.2067), [40-page PDF](https://arxiv.org/pdf/0708.2067)  
**Source-reading note:** the arXiv abstract and the full 40-page PDF text were read. The arXiv HTML endpoint returns 404 for this 2007 paper; the PDF is the full-text authority.

## Executive verdict

> **Model-category hypotheses:** **BREAKS-at-argmax in the current exact formulation**. More precisely, argmax discontinuity does not by itself violate 2-out-of-3; the failure is that “witnesses” plus scorer-invisible perturbations have not been given a complete/cocomplete, locally presentable category with cofibrations, fibrations, and functorial factorizations. The obvious discrete and score-fiber groupoid candidates are not model categories. A left-proper combinatorial structure is recoverable only after replacing the witness set by a simplicial-presheaf envelope; a `(tau, epsilon)`-smoothed scorer gives one possible surrogate, but loses exact uint8/round/argmax authority.
>
> **Existence versus section:** **enriched-section CONSTRUCTIVE N**. Barwick supplies an abstract localized model structure and, through small-object/fibrant-replacement machinery, a category-theoretic replacement. He does not supply a split quotient section, a video decoder, a finite algorithm, or a 30-minute cost bound. The reflection unit has the wrong guarantee and generally the wrong direction for Rung-3's receiver section.
>
> **Descent -> D38:** **KILL for the current claimed derivation; conditional descent typing survives.** The v8 artifacts do not yet define a site/cover, restriction functors on overlaps, or a changing-isotropy coefficient stack. Therefore no current Cech or `H^2` obstruction, and no nonzero `R_twist^global`, is typed. If those data are built, the homotopy fiber of the Cech totalization gives a clean conditional rate variable. A nonzero obstruction means “no global section,” not “pay some additional bits.”
>
> **Overall scoped verdict:** **DESCENT-worth-a-dig**, but **GROUNDING-only today**. No score, archive, scorer row, or pointer changed.

Only a **MEASURED byte-closed exact row** through the frozen evaluator on exact archive bytes moves the pointer. This memo is at most a DERIVED typing candidate for v8 reconciliation cost; it is never a lever by itself. No numpy-fp32 or contest-axis measurement was run.

## Evidence labels and stores consulted

- **MEASURED:** none. No training, scorer execution, archive mutation, paid dispatch, or heavy launch occurred.
- **DERIVED:** mathematical consequences below that follow from Barwick's stated hypotheses plus the currently typed repo artifacts.
- **INFERRED:** proposed interpretations of v8 edge tubes as a cover and of reconciliation states as a stack. They are explicitly not current repo facts.
- **ASSUMED:** any future site, cover, coefficient band, good-cover condition, or receiver-computable choice of descent filler.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; top entries of the Claude project `MEMORY.md`; current directives; current sibling-ownership checkpoints; `.omx/research/bousfield_localization_dig_20260713.md`; `.omx/research/infdesc_foundations_dig_20260713.md`; `.omx/research/garrett_algebra_dig_20260713.md`; `.omx/research/weyl_symmetry_group_unification_20260713.md`; `.omx/research/condprob_homotopy_lie_dig_20260713.md`; `.omx/research/ladder_owed_measurables_20260713.md`; `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`; `src/tac/canonical_equations/rate_law_ladder_20260713.py`; `src/tac/canonical_equations/rate_law_ladder_measured_20260713.py`; Barwick arXiv:0708.2067v2 in full.

## What Barwick actually proves

The distinctions matter because the seed memo implicitly asks the localization theorem to create structure that it only consumes.

1. **Ordinary localization.** Given a model category `M` and a small set `H` of homotopy classes, an object `Z` is `H`-local when

   ```text
   RMap_M(B,Z) -> RMap_M(A,Z)
   ```

   is an isomorphism in `Ho(sSet)` for every `[A -> B] in H`. A map is an `H`-local equivalence when all local objects detect it as an equivalence. Theorem 2.11 constructs `L_H M` for a left-proper combinatorial `M`: the underlying category and cofibrations are unchanged, fibrant objects are the fibrant local objects, and weak equivalences are the local equivalences.

2. **Factorization and fibrations.** The accessible functorial factorizations used in the proof come from a transfinite small-object argument (Proposition 1.10). Barwick emphasizes in section 2.14 that one generally has very little control over the localized generating trivial cofibrations. Left localization can destroy right properness. Proposition 2.32 recognizes certain localized fibrations by a homotopy-pullback square after fibrant replacement when the codomain lies in a suitable local/right-proper subcategory; it does not give a pointwise codec formula.

3. **Right Quillen presheaves.** Theorem 2.42 localizes the projective model structure on right sections so that fibrant sections are homotopy cartesian. This models the homotopy limit of an already specified right Quillen presheaf. It does not infer the presheaf, its transition functors, or its sections from scalar observations.

4. **Postnikov towers.** Propositions 2.47--2.51 construct `n`-truncated model structures, a Postnikov-tower model structure, and hypercompletion for a left-proper combinatorial simplicial model category. The tower is defined by truncation of every derived mapping space, not by spatial frequency, optimizer stage, or annealing temperature.

5. **Enriched localization.** For a tractable symmetric-monoidal model category `V` and a left-proper tractable `V`-model category `C`, Theorem 3.18 constructs

   ```text
   L_(H/V) C = L_(I square S) C,
   ```

   where `I` is a generating set of cofibrations of `V`, `S` represents `H`, and `square` is the pushout product. Locality is tested by the derived `V`-mapping object. Enrichment preserves structured mapping objects; it does not add an optimizer or a computational section.

6. **Descent.** For an already specified small site `(C,tau)` and `V`-valued presheaf `F`, Definition 3.34 requires, for each covering sieve `R -> yX`,

   ```text
   F(X) -> holim_{Y in (C/R)^op} F(Y)
   ```

   to be an equivalence in `Ho(V)`. Theorem 3.36 obtains the local projective/injective model structures by enriched localization at `R . 1_V -> y_V X`. Section 3.35 explicitly warns that this is Cech descent, not hyperdescent in general; hypercovers require a further localization.

These are existence and recognition results **after** the categorical inputs are typed.

## Q1. Do the model-category hypotheses hold for the exact witness category?

### 1.1 Exact evaluator and the tempting weak equivalences

Let the receiver-realized witness space be

```text
W_8 = {0,...,255}^{T x H x W x 3},
```

and let the exact evaluator observable be

```text
E_exact(w) = (Argmax SegNet(R(w)), PoseNet_YUV6(R(w))),
```

where `R` includes the actual resize, rounding, and uint8 chain. A scorer-invisible deformation should preserve `E_exact`, not merely the final scalar score: equal scalar scores can hide different `(d_seg,d_pose,bytes)` components.

The seed's natural proposal is therefore

```text
w ~_E w'  iff  E_exact(w) = E_exact(w'),
```

possibly refined to membership in one path component of an evaluator cell.

**DERIVED:** endpoint equality under `E_exact` is an equivalence relation. If morphisms and compositions had already been defined compatibly, equality itself would satisfy 2-out-of-3. It would be false to blame a 2-out-of-3 failure directly on argmax.

The actual failure is earlier and more structural.

### 1.2 The obvious candidate categories fail

**Candidate A: the discrete category on exact witnesses.** Its only arrows are identities. With at least two witnesses `w != v`, there is no coproduct: a coproduct would need arrows from both `w` and `v` into one object. Hence it is not cocomplete and cannot be a model category.

**Candidate B: the thin groupoid of score/evaluator fibers.** Put one invertible arrow between any two witnesses in the same `~_E` class. Different evaluator fibers again have no coproduct or pushout. It is not cocomplete.

**Candidate C: the codiscrete category with an arrow between every pair.** Every arrow is then an isomorphism. A model structure must contain every isomorphism among its weak equivalences, so weak equivalences cannot be restricted to scorer-preserving arrows. This candidate collapses the evaluator distinction.

**Candidate D: paths that stay inside a hard evaluator cell.** Away from decision/rounding walls, hard cells can support path concatenation. But the repo has not defined the category of paths, its limits/colimits, cofibrations, lifting properties, or functorial factorizations. Exact `R` has rounding walls, and hard SegNet has tie walls

```text
z_c(R(w)) = z_d(R(w)),  c != d,
```

across which the argmax observable jumps. A generic interpolation or pushout need not remain within a cell. Thus the naive geometric operations required to argue left properness are not available.

**Verdict-scope `FORMULATION=current exact witnesses-as-objects + scorer-invisible perturbations-as-arrows`: BREAKS-at-argmax / UNESTABLISHED.** This does **not** prove that no model of exact scorer cells can ever exist. It refutes the stronger seed claim that the current witness program *already is* a left Bousfield localization.

No current artifact supplies:

- a complete and cocomplete underlying category;
- a locally presentable/combinatorial presentation;
- generating cofibrations and trivial cofibrations;
- functorial factorization;
- weak-equivalence stability under the relevant pushouts;
- left properness.

Therefore Theorem 2.11 cannot presently be invoked.

### 1.3 Two honest repairs, neither equal to the seed claim

#### Repair A: exact but enlarged stratified-presheaf envelope

**DERIVED possibility, not implemented.** Define a small category/site `C_exact` whose objects are explicitly enumerated receiver/evaluator strata or chart intersections and whose arrows are typed inclusions/specializations. Then take

```text
M_exact = sSet^{C_exact^op}
```

with the projective or injective model structure. Under the standard simplicial-set model structure, this presheaf category is combinatorial; properness is inherited objectwise in Barwick's Propositions 1.20--1.21. One may then localize at a small set of maps representing certified scorer-invisible chart deformations or covers.

Before localization, projective weak equivalences and fibrations are objectwise weak equivalences and Kan fibrations; projective cofibrations are determined by the lifting property. After localization, cofibrations stay fixed, fibrant objects are objectwise fibrant and local, and weak equivalences are detected by derived mapping spaces into local objects. Proposition 2.32 supplies only a conditional fibration recognition rule.

This construction can retain hard discontinuities by treating them as strata. But its objects are simplicial presheaves, **not raw witnesses**, and the current repo does not contain `C_exact`, its topology, or its generating maps. The number of exact uint8/evaluator cells may also make construction useless as a decoder.

#### Repair B: `(tau, epsilon)`-smoothed surrogate

For example, after clamping the continuous receiver value `u` to `[0,255]`, take the explicit soft quantizer

```text
q_epsilon(u)
  = sum_{k=0}^{255} k exp(-(u-k)^2/epsilon^2)
    / sum_{k=0}^{255} exp(-(u-k)^2/epsilon^2),
epsilon > 0.
```

Let `R_epsilon` replace the exact round/uint8 part of `R` by `q_epsilon`, and define

```text
p_tau(w) = softmax(z_seg(R_epsilon(w)) / tau),  tau > 0,
E_(tau,epsilon)(w) = (p_tau(w), PoseNet_YUV6(R_epsilon(w))).
```

Build a small chart site `C_(tau,epsilon)` in the continuous witness domain and use

```text
M_(tau,epsilon) = sSet^{C_(tau,epsilon)^op}.
```

Localize at a **chosen small set** `H_(tau,epsilon)` of chart deformation maps whose soft evaluator images are certified homotopically null. This gives a Barwick-compatible model category because the ambient presheaf structure, not equality of soft loss values, supplies the axioms.

**What is lost:**

- exact uint8/round boundaries;
- hard argmax cells and tie behavior;
- exact receiver parse-back authority;
- any guarantee that `tau,epsilon -> 0` commutes with localization;
- any guarantee of a Quillen equivalence between the surrogate and exact scorer geometry.

The level-set viscosity/softmax flow can motivate `H_(tau,epsilon)`, but it does not prove those limiting statements. This recovered model is a training/design surrogate only. It cannot authorize a score; numpy-fp32 exact-through-`R` remains the local verdict surface, and only exact contest CPU/CUDA rows can promote.

### Q1 answer

**Model-category-hypothesis verdict:** **BREAKS-at-argmax for the current exact witness category; RECOVERED-under-`(tau,epsilon)`-smoothing only after changing to a simplicial-presheaf surrogate.** An exact stratified-presheaf envelope is also mathematically possible but unbuilt. The discontinuity is not a universal no-go for category theory; it is a no-go for the current untyped “witnesses already form `M`” assertion.

## Q2. Does enriched localization construct the cheap receiver section?

No.

### 2.1 Reflection is not a split quotient

In homotopy-category language, localization gives a reflector-like functor and unit

```text
eta_X : X -> L_H X.
```

Rung-3 needs something structurally different. If

```text
q : D -> D / ~_E
```

is the quotient label, the codec needs a receiver-computable section

```text
s : D / ~_E -> D,
q o s = id,
```

together with a legal representation of `s` and a runtime below 30 minutes. A reflection unit does not split `q`; the universal property says maps from `X` to local objects factor essentially uniquely after localization. It does not select a representative in every equivalence class.

Even if one identifies `L_H X` with a task-sufficient statistic, `eta_X` is the map **to** the local object. The required inverse representative map is extra choice/data. It may not exist naturally, may not be computable, and may be expensive to encode.

### 2.2 What the proof can construct

Barwick's ordinary existence proof invokes Smith recognition and accessible functorial factorizations arising from the transfinite small-object argument. The enriched theorem replaces `H` by the pushout-product set `I square S` and invokes ordinary localization. Formally, after generators and all lifting problems are supplied, this can define a functorial fibrant/local replacement.

That is “constructive” only in a category-theoretic existence sense. It may attach cells for every lifting problem through transfinite stages. The paper supplies:

- no finite iteration bound;
- no arithmetic complexity bound;
- no memory bound;
- no receiver-compatible representation;
- no deterministic exact-video output;
- no 30-minute decoder proof.

Barwick's phrase “computed effectively” in the right-properness subsection concerns homotopy pullbacks after suitable factorization. It is not a wall-clock or codec-complexity claim. Section 2.14's warning that localized generating trivial cofibrations are generally poorly controlled cuts directly against a cheap decoder inference.

### 2.3 Enrichment does not fix the direction or cost

Enrichment replaces simplicial mapping spaces by derived mapping objects in a chosen symmetric-monoidal model category `V`. Nothing in Theorem 3.18 says that `V` is the evaluator metric, that the score geometry satisfies the pushout-product axiom, or that a local replacement is a canonical low-byte witness.

For the witness program, all of the following would still have to be built outside Barwick:

1. the enriched category and mapping object;
2. the localizing maps;
3. a finite local-replacement algorithm;
4. a deterministic representative-selection section;
5. a counted payload and receiver implementation;
6. parse-back and exact scorer verification.

### Q2 answer

**Existence-vs-constructive-section verdict:** **NON-CONSTRUCTIVE FOR THIS CODEC; enriched-section CONSTRUCTIVE N.** Barwick is redundant with the Rung-2/Rung-3 statement that a quotient/local object may exist while nearly all paid complexity sits in the section. It moves no pointer and supplies no decoder surprise.

## Q3. Can descent type D38 global gluing and derive `R_twist^global`?

### 3.1 The conditional site that would be needed

The following is an **INFERRED future typing**, not current v8 state.

Let `X` be a spacetime boundary complex. A plausible cover would contain:

- edge-tube opens `U_e` for each active unordered class adjacency `e={a,b}`;
- junction opens where three or more edge tubes meet;
- interior/background opens needed by the receiver reconstruction.

For every nonempty multi-index `I=(i_0,...,i_p)`, write

```text
U_I = U_i0 intersect ... intersect U_ip.
```

Let `F(U)` be the infinity-groupoid of receiver-valid local carrier/reconciliation states on `U`; morphisms are certified scorer-invisible gauge changes. The required restriction maps are

```text
rho_(V,U) : F(U) -> F(V),  V subset U,
```

with coherent composition. The Cech cosimplicial object is

```text
C^p(U,F) = product_{i_0<...<i_p} F(U_i0...ip),
```

with cofaces induced by restrictions and codegeneracies by repeated indices. Its descent object is

```text
Desc_U(F) = Tot C^bullet(U,F)
          = holim_{[p] in Delta} C^p(U,F).
```

The global-to-local comparison is

```text
rho_U : F(X) -> Desc_U(F).
```

Barwick's Theorem 3.36 says that, once `(C,tau)`, `F`, and the covering sieves are supplied, the local fibrant presheaves are those satisfying the analogous homotopy-limit condition. It does not create `X`, `U`, `F`, or `rho`.

All obstruction/rate statements below additionally assume that `F` is `tau`-local, so `rho_U` is an equivalence for this cover. Without that condition, a compatible object of `Desc_U(F)` need not come from `F(X)` at all; replacing `F` by its stackification may also change the carrier/codec semantics.

### 3.2 Explicit homotopy-coherent datum

An object of `Desc_U(F)` begins with

```text
x_i in F(U_i),
g_ij : x_j|U_ij -> x_i|U_ij,
a_ijk : g_ij o g_jk => g_ik on U_ijk,
```

followed by the tetrahedral coherence on quadruple overlaps and higher coherences when `F` is not 1-truncated.

For a 1-truncated gerbe with a **fixed abelian** automorphism sheaf `A`, the triple-overlap defect can be represented as

```text
c_ijk = g_ij g_jk g_ki in A(U_ijk),
delta c = 0,
[c] in Cech H^2(U;A).
```

Then `[c] != 0` obstructs a global object. If `[c]=0`, choosing a trivialization is still noncanonical; the choices form a torsor controlled by lower cohomology.

That `H^2` statement has strict hypotheses. For sheaves of sets, gluing is an equalizer condition, not automatically an `H^2` obstruction. For torsors the basic classification is nonabelian `H^1`. For a gerbe it is `H^2`. For a general infinity-stack, successive obstructions have the form

```text
o_(n+1) in Cech H^(n+1)(U; pi_n(F,x)),
```

with twisted local coefficients.

### 3.3 Changing isotropy is exactly the missing type

D38 fixes one regular stratum `sigma=(kappa,omega,a,r)` and obtains a strict local split

```text
E_(sigma,x) = K_(sigma,x) semidirect H_(cov,sigma,x),
s(h) = (1,h),
R_twist^ideal(local,sigma,x) = 0.
```

Across strata or objects, the groups `K_(sigma,x)` and stabilizers change. To form a coefficient sheaf/band, one needs overlap homomorphisms such as

```text
Ad(g_ij) : Aut(x_j)|U_ij -> Aut(x_i)|U_ij,
```

compatible on triple overlaps, at least up to specified inner automorphisms. Those maps are precisely what D38 marked as not typed.

With changing nonabelian isotropy, there is no single abelian group `A` in which `c_ijk` lives. At best there is a nonabelian band/gerbe and a pointed nonabelian cohomology class; more generally there is an obstruction tower with local coefficients. Writing plain `H^2(X;A)` now would be false precision.

### 3.4 The v8 “cover” is not yet a cover

The current v8 spec defines edge-indexed carriers and a global algorithm:

```text
merge -> frozen-SegNet diff -> chroma-first/luma-reserved correct -> iterate.
```

It does not define:

- opens `U_e` and their covering property;
- nonempty pair/triple overlaps at the receiver surface;
- restriction maps for carrier state;
- transition equivalences on overlaps;
- a stack of scorer-valid states;
- a band of changing isotropy groups.

Moreover, per-class carriers may be globally supported fields indexed by class adjacency rather than local sections of open subsets. Tropical merge couples them through a single global argmax. “Indexed family” is not the same object as “cover.”

**Verdict-scope `FORMULATION=current SPEC_v8 carriers + current D38 local split`: KILL the claim that Barwick already derives a global Cech/H^2 obstruction or `R_twist^global`.**

### 3.5 The clean conditional rate law that survives

Fix local states `x=(x_i)` and define the descent-filler space as the homotopy fiber

```text
G_U(x) = hofib_x(
    Desc_U(F) -> product_i F(U_i)
).
```

This space contains overlap transitions and all coherence fillers compatible with the fixed local carriers.

There are two qualitatively different cases.

1. **`G_U(x)` is empty.** Under the stated `tau`-locality assumption, the local carriers do not glue. This is a feasibility obstruction. No finite number of “twist bits” can make the same local datum descend; one must change the local states, refine the cover/model, or encode a defect sidecar that changes the problem.

2. **`G_U(x)` is nonempty.** Let

   ```text
   Theta_U = pi_0 G_U(x)
   ```

   denote the gauge class of a chosen descent filler. Conditional on a fully typed cover, stack, local carriers, and public receiver state, the ideal quotient-label term is

   ```text
   R_twist^global,ideal
     = H(Theta_U | q_H, A_U, U, x, public).
   ```

   This is a **DERIVED conditional form**, not a current numerical law. A real codec also needs a deterministic receiver-computable section

   ```text
   s_U : Theta_U -> G_U(x)
   ```

   and then a global reconstruction in `F(X)`. If no canonical public `s_U` exists, its representation/algorithm is again section cost. If `Theta_U` is public-deterministic, the entropy can be zero while compute time remains nonzero.

This sharpens D38: local strict splitting can force each local twist term to zero while global transition/trivialization data remain. But the obstruction class itself is not automatically a rate variable. Nonzero obstruction means no section; when the obstruction vanishes, the **choice of trivialization/descent filler** is the encodable random variable.

### Q3 answer

**Descent -> D38 verdict:** **KILL current derivation; retain the conditional `G_U(x)` law as DESCENT-worth-a-dig.** There is no admissible `R_twist^global` value, bit count, or canonical equation successor until the cover, restrictions, stack, and changing-isotropy band are typed and receiver-closed.

## Q4. Do Postnikov towers derive curriculum stage count or convergence?

No new measurable follows.

Barwick's `n`th stage is characterized by derived mapping spaces:

```text
RMap_M(Z,X(n)) is an n-type for every Z,
pi_j RMap_M(Z,X(n)) -> pi_j RMap_M(Z,X(n-1))
```

is an isomorphism for `j<n`. Convergence of the whole tower to `X` is the separate hypercompleteness condition

```text
X ~= holim_n X<n>.
```

The witness curriculum currently orders loss/temperature/boundary scales. It has not supplied:

- a simplicial model category of curriculum states;
- derived mapping spaces;
- truncation functors;
- homotopy groups killed or preserved by each stage;
- hypercompleteness;
- a finite homotopy-dimension bound.

Spatial scale, persistence threshold, and annealing temperature are not Postnikov degree without those identifications. In particular, the current D38/conditional-probability memo already records that varying `tau` at fixed logits does not move hard argmax topology. Barwick cannot turn that annealing schedule into a stage count.

**Q4 verdict:** **confirmation, not a lever.** No DERIVED stage count, convergence rate, byte delta, or stop rule.

## Triality and apparatus disposition

### Equations leg

The only candidate successor is the conditional equation

```text
G_U(x) = hofib_x(Desc_U(F) -> product_i F(U_i)),
R_twist^global,ideal = H(pi_0 G_U(x) | q_H,A_U,U,x,public),
```

with the fail-closed precondition `G_U(x) != empty` and an explicit receiver section `s_U`.

It is **not registered**. The current result kills the unqualified law, the necessary coefficient/cover types do not exist, and the shared rate-law/equation surfaces were owned by the D38 measurable sibling during this read. Registering a symbol-only successor would violate NO-FAKE and anti-collision.

### DAG leg

**DEFER to main.** A future FEED is justified only after an artifact defines:

1. the v8 receiver boundary complex and a genuine cover;
2. intersection/restriction maps;
3. changing-isotropy transition functors/band;
4. a finite Cech-totalization probe;
5. a deterministic receiver section and runtime/byte accounting.

At that point the DAG node should gate on `Desc_U(F)` nonemptiness before treating any transition choice as rate.

### DSL / actuator leg

None. This is GROUNDING, not an actuator or loss schedule. No `costate`, `witness_control`, curriculum DSL, live run, v9/#432 surface, or bit-allocator file was touched. No invented flag is proposed.

### Pointer delta

```text
pointer_before = unchanged
pointer_after  = unchanged
measured_exact_rows_added = 0
archives_created_or_mutated = 0
```

## Minimal next proof, if main elects to pursue descent

This is a bounded typing/probe task, not a training launch:

1. Materialize one small receiver-space boundary complex from an existing exact witness.
2. Define edge-tube and junction charts and prove they cover that complex.
3. Define restriction maps for a minimal carrier-state groupoid.
4. Enumerate pair/triple intersections and compute the finite Cech nerve.
5. Test whether transition data satisfy triple coherence; distinguish empty filler space from noncanonical filler choice.
6. Only if nonempty, measure the bytes and runtime of a deterministic filler section through receiver parse-back.

That probe could falsify or instantiate the conditional law. Until then, the law is DERIVED/conditional and not a score mover.

## One-line scoped verdict

**{BREAK-at-argmax for the current witness category; enriched-section CONSTRUCTIVE N; descent->D38 KILL now but conditional homotopy-fiber typing is DESCENT-worth-a-dig; Postnikov confirmation only; overall GROUNDING-only, pointer unchanged}.**
