# Local-to-global descent, boundary jets, and Gauss--Bonnet for the v8 witness

**Date:** 2026-07-13
**Lane:** `local_global_descent_dig`
**Role:** SOL ultra capstone deep-math dig
**Disposition:** `research_only=true` · MEANS/analysis · `$0` · no launch · no scorer run · no archive mutation · no pointer claim
**Authority:** existing n600 measurements are cited at their original advisory axes; all new mathematics below is `DERIVED`, `INFERRED`, or `ASSUMED` explicitly.
**Pointer:** UNMOVED. `measured_exact_rows_added=0`; `archives_created_or_mutated=0`.

## Answer first

> **{descent obstruction: OTHER -- the fixed-stratum D38 class vanishes and costs 0 ideal bits, while the global obstruction remains NOT-TYPED/UNMEASURED; it is NOT the MEASURED D36 `H(q_G|U_proxy)=147,616 bits = 18,452 B = 22.116745%` gap · 2-jet sufficient for the full boundary rate code: NO; only conditionally sufficient on each regular `C^3` arc when topology, junctions, phase, and a third-order remainder bound are supplied; predicted numerical saving = NONE ADMISSIBLE, with conditional `N_2(epsilon) >= L(M_3/(6 epsilon))^(1/3)` and current measured rate rows unchanged · Gauss--Bonnet storable invariant: YES as a per-class topology/checksum/event constraint, NO as a replacement for the boundary · REAL v8/rate lever: a finite receiver-space descent-filler measurement after typing the actual cover/restrictions; 2-jet is a bounded reformulation probe and Gauss--Bonnet-as-codec is beautiful-but-inert · overall verdict: LOCAL-TO-GLOBAL IS THE RIGHT ACCOUNTING FRAME, NOT A CURRENT RATE RESULT}.**

One-line verdicts:

1. **Descent/gluing:** `OTHER`, not `147,616 bits`: local D38 splits; global effectivity is untyped. A nonzero obstruction makes the fixed local datum infeasible, not finitely encodable; only a nonempty filler space can have a code length.
2. **2-jet/mod-`p^3`:** **NO** for the full partition/current coder. A continuously known curvature field plus one initial frame determines a regular curve, but finitely many 2-jets require a third-order remainder bound and separate topology/junction/phase data. No measured lower boundary rate follows.
3. **Gauss--Bonnet:** **YES** for a cheap invariant/checksum, **NO** for boundary reconstruction. It is useful for detecting topology events, not for locating the separatrix.

The only real campaign-facing output is the **descent probe specification**: type the receiver boundary complex, enumerate its Cech nerve, test filler nonemptiness, and only then measure the entropy/bytes of a deterministic filler. It can change v8 reconciliation accounting. The other two are constraints on such a code, not present byte wins.

---

## 0. Evidence ledger and snapshot correction

### 0.1 Labels

- **MEASURED:** only numbers already present in the named n600/n96 artifacts. No new empirical measurement was run in this arm.
- **DERIVED:** consequences of Cech descent, obstruction theory, Frenet/Taylor reconstruction, and planar Gauss--Bonnet under their stated hypotheses.
- **INFERRED:** interpreting future v8 edge tubes and junction charts as an actual cover/stack. The current v8 artifact does not yet instantiate these maps.
- **ASSUMED:** regularity (`C^3` where invoked), a receiver tolerance `epsilon`, a finite automorphism band, public decoder state, or any future good-cover/effectivity property.

### 0.2 The prompt's D36/D38 conflation is false in the live source of truth

**MEASURED, n600, `[macOS-CPU codelength advisory; score_claim=false]`:** D36 represents the shipped `(1200,32)` symmetric-int8 code table as `q_G` and evaluator-side label/pose features as `U_proxy`. Its cross-fitted residual stream is:

```text
H_operational(q_G | U_proxy) upper-bound code = 147,616 bits = 18,452 B
fraction of the 83,430 B archive              = 22.116744576%
fold-sensitivity 95% interval                 = [146,913.96, 148,318.04] bits
gross saving versus raw code                  = 1,903 B
conservative predictor charge                 = 15,256 B
net saving after model                        = -13,353 B
```

This is an **operational conditional codelength upper bound**, not the entropy of a Cech class. `U_proxy` is evaluator-side and not receiver-public.

**DERIVED, D38, `verdict_scope=FIXED-REGULAR-STRATUM x STRICT-SEMIDIRECT-FORMULATION`:** the typed local extension

```text
1 -> K_(sigma,x) -> K_(sigma,x) semidirect H_(cov,sigma,x)
  -> H_(cov,sigma,x) -> 1
```

has the homomorphic section `h -> (1,h)`. Its Schreier factor set is neutral, so

```text
R_twist^(local,ideal) = 0 bits.
```

**LIVE GLOBAL STATUS:** `NOT-TYPED / overlap-and-gluing audit owed`. The cover, restrictions, and changing-isotropy band are absent. Therefore the measured `147,616` bits belong to **D36**, while the open global-gluing question belongs to **D38**. No equality between them is defined, much less proved.

---

## 1. The exact local-to-global sequence for the witness

### 1.1 What must be typed before “local carriers glue” is a mathematical statement

Let `X` be the receiver-realized spacetime boundary complex, not the raw image rectangle alone. A prospective v8 cover is:

- edge-tube charts `U_e` for active unordered class adjacencies `e={a,b}`;
- junction charts `U_v` around triple/higher tie points, dash endpoints, births, deaths, and merges;
- interior/exterior charts required for a complete receiver reconstruction.

For every nonempty intersection `U_I=U_(i0) intersect ... intersect U_(ip)`, let `F(U_I)` be the groupoid of receiver-valid local carrier/reconciliation states, with morphisms the **certified** scorer-invisible gauge changes. Restrictions

```text
rho_(V,U): F(U) -> F(V),  V subset U
```

must compose coherently. These objects are **INFERRED FUTURE TYPES**. `SPEC_v8` currently supplies edge-indexed global fields plus `merge -> diff -> correct`; an indexed family is not yet an open cover or a presheaf.

The Cech cosimplicial object is

```text
C^p(U,F) = product_(i0<...<ip) F(U_(i0...ip)),
Desc_U(F) = Tot C^bullet(U,F) = holim_[p in Delta] C^p(U,F).
```

The global-to-local comparison is

```text
rho_U: F(X) -> Desc_U(F).
```

“Hasse holds for this cover” means `rho_U` is effective on the datum in question. It is **not** automatic for an arbitrary image cover; Hasse--Minkowski is a special arithmetic theorem for quadratic forms over global fields, not a generic gluing axiom.

### 1.2 The elementary Cech exact sequence: the Sha/Selmer analog

First assume a fixed automorphism sheaf/band `A` and local representatives `s_i`. On overlaps,

```text
s_i = g_ij . s_j,       g_ij in A(U_ij).
```

For an abelian band the Cech cochain complex begins

```text
0 -> C^0(U,A) --delta_0--> C^1(U,A) --delta_1--> C^2(U,A) -> ...,
H^1(U,A) = ker(delta_1) / im(delta_0).
```

For a nonabelian band, the corresponding pointed-set sequence is exact through `H^1`:

```text
1 -> A(X) -> C^0(U,A) --delta--> Z^1(U,A) -> Cech-H^1(U,A).
```

Here `delta(a)_ij=a_i a_j^(-1)` and the last arrow takes a transition cocycle to its class. Thus:

- `g_ij g_jk g_ki = 1` on every triple overlap says the transition data are coherent;
- `[g]=0 in Cech-H^1(U,A)` iff `g_ij=a_i a_j^(-1)` for local gauges `a_i`, hence the local representatives admit one global trivialization/section in the chosen receiver gauge.

The direct Tate--Shafarevich analog is the locally trivial kernel

```text
Sha^1_U(A) := ker( H^1(X,A) -> product_i H^1(U_i,A) ).
```

Its nonzero elements are globally nontrivial torsors that are trivial on every local chart. This is the exact “local-solvable-everywhere but not globally trivial” object. Calling the D36 code residual `Sha` without constructing this map would be analogy-hand-waving.

There is a distinct higher obstruction. If the overlap equivalences close only up to automorphisms, the triple defect

```text
c_ijk = g_ij g_jk g_ki
```

defines `[c] in Cech-H^2(U;Z(A))` only under a fixed abelian/central band. A nonzero class obstructs even a coherent torsor/object. With changing nonabelian isotropy, one needs a banded gerbe or an obstruction tower with twisted local coefficients; plain `H^2(X;A)` is false precision.

**DERIVED correction:** the gluing obstruction can live in `H^1`, `H^2`, or higher depending on whether the desired object is a trivialization, a gerbe object, or an infinity-stack section. There is no universal “the obstruction is H^2” statement for v8 until the carrier stack is typed.

### 1.3 The homotopy-coherent exact obstruction tower

For a pointed stack/groupoid `F`, the Bousfield--Kan descent spectral sequence has the conditional form

```text
E_2^(s,t) = Cech-H^s(U; pi_t(F,x))  =>  pi_(t-s) Tot(F(U_bullet)).
```

The successive effectivity obstructions are

```text
o_(r+1)(x) in Cech-H^(r+1)(U; pi_r(F,x)),
```

with local coefficients twisted by the transitions. This is the honest local-global sequence for changing isotropy.

Fix the local carrier objects `x=(x_i)` and define the descent-filler space

```text
G_U(x) = hofib_x( Desc_U(F) -> product_i F(U_i) ).
```

It contains overlap transitions and all higher coherence fillers compatible with those fixed locals. The local-global decision tree is exact:

```text
G_U(x) = empty
  => the fixed local datum is globally infeasible;
     change/refine the locals, cover, or problem. No finite “obstruction bits” fix it.

G_U(x) != empty
  => all existence obstructions vanish for this datum;
     Theta_U(x) := pi_0 G_U(x) is the noncanonical filler/gauge choice.
```

Only the second branch has a rate variable:

```text
R_glue^(ideal)(x,U)
  = H( Theta_U(x) | U, x, public receiver state ),       if G_U(x) != empty.
```

A real codec also needs a deterministic receiver-computable section

```text
s_U: Theta_U(x) -> G_U(x)
```

and must charge its payload, algorithm, runtime, and parse-back effects. If `Theta_U` is unique or public-deterministic, ideal extra gluing bits are zero even when the filler computation is nontrivial.

### 1.4 Descent-obstruction verdict and the 147,616-bit question

**VERDICT: `OTHER`.** More precisely:

- **DERIVED local D38:** obstruction neutral, `R_twist^(local,ideal)=0`.
- **GLOBAL D38:** obstruction status unknown because `X,U,F,rho`, and the isotropy band are not instantiated.
- **MEASURED D36:** `147,616 bits` is the conditional code length of an int8 atlas-label proxy given an evaluator proxy; it is not a gluing class or filler entropy.
- **NO EQUALITY:** no map `q_G -> Theta_U`, no sufficiency theorem, and no minimal-code proof exists. Even the units play different roles: D36 codes a symbol table; D38 first asks whether a global object exists.

If future typing finds a nonzero obstruction, the thesis “the obstruction is the payload cost” is **FALSE** at `FORMULATION=fixed-local-datum`: the same local datum does not glue at any finite rate. A defect sidecar that changes the local datum may have a rate, but that rate is not the obstruction class itself. If the obstruction vanishes, gluing overhead is near zero **only if** the filler is canonical/public; the local carrier payload and global section cost remain.

This conclusion composes with, rather than overturns, the rate-law ladder. Its Rung 1 object is the state-dependent stratified groupoid joined with `H_cov`; Rung 3 bounds the ideal quotient label by `64 bits` per frozen scoring axis because `|im S| <= 2^64`, so nearly all practical payload can indeed live in the receiver **section**. D36 measures one incomplete-atlas/section proxy; D38 asks whether locally chosen sections are globally effective. “Most payload is section cost” does not imply “every section cost is one obstruction entropy.”

It also sharpens the covariance audit's notation. The registered distortion split

```text
d_seg = d_cov + d_gauge
```

uses `d_gauge` for the **lattice-sampling phase mismatch** (the n600 spike channel, about `0.005318` on its named surface). That is a distortion term. A Cech `H^1/H^2` class is a topological effectivity obstruction, and `H(Theta_U|...)` is a rate term. The shared word “gauge” does not identify these three typed objects. Descent may organize how covariant edge charts and phase choices assemble, but it supplies no equality between `d_gauge`, the obstruction class, and the D36 bits.

**REAL MEASURABLE LEVER:** enumerate a finite Cech nerve over one receiver-closed v8 boundary complex, test `G_U(x)` nonemptiness, and, only on the nonempty branch, compare packed bytes of a deterministic filler with a global-carrier baseline. This can discover or eliminate reconciliation payload. The analogy alone cannot.

`verdict_scope` on the negative: **CURRENT SPEC_v8 + CURRENT D38 TYPES**. The descent family remains intact; the reactivation condition is the explicit cover/restriction/band implementation.

---

## 2. The 2-jet/mod-`p^3` claim as a boundary rate law

### 2.1 What a boundary 2-jet actually contains

On a regular class edge `Gamma_e`, use signed distance `phi_e`. At `p in Gamma_e`,

```text
j_p^2 phi_e = (phi_e(p)=0, grad phi_e(p)=n, Hess phi_e(p)).
```

The tangent is perpendicular to `n`; the tangent-tangent component of `Hess phi_e` gives signed curvature `kappa` up to orientation convention. Equivalently, an arc-length parametrized plane curve obeys the Frenet system

```text
r'(s)=t(s),
t'(s)=kappa(s)n(s),
n'(s)=-kappa(s)t(s).
```

**DERIVED:** one initial position/orientation plus the **entire function** `kappa(s)` determines a regular connected curve up to rigid motion. In that functional sense, curvature-order data are sufficient.

The classical surface statement is stronger than “keep each isolated 2-jet”: the first and second fundamental forms are **fields** that must satisfy the Gauss--Codazzi compatibility equations; only then does the fundamental theorem of surfaces reconstruct a local immersion up to rigid motion. That compatibility condition is itself a gluing/descent law. Our scorer partition lies in a flat image plane, so the ambient first form is fixed and the relevant moving object is its one-dimensional separatrix; the second-form content reduces to the curvature function on each regular edge. This is where the analogy is mathematically faithful, and also where the isolated-jet reading breaks.

But “the 2-jet at every point” is not a finite payload; it is another representation of the whole curve. A finite list of 2-jets is sufficient only after imposing a regularity/remainder model and separately supplying:

- component count and class-edge incidence;
- endpoints, corners, triple junctions, and orientation;
- birth/death/merge events across time;
- subpixel sampling phase needed through `R`;
- a bound on the omitted third-order term.

### 2.2 The clean conditional rate law

On one graph chart `y=f(x)`, store the quadratic Taylor model at anchors spaced by `h`:

```text
f(x_j+u) = f_j + f'_j u + (1/2)f''_j u^2 + R_3(u).
```

If `f in C^3` and `|f'''| <= M_3`, Taylor's theorem gives

```text
|R_3(u)| <= M_3 |u|^3 / 6.
```

To keep normal displacement below receiver tolerance `epsilon_e`, it is sufficient that

```text
h_e <= (6 epsilon_e / M_(3,e))^(1/3),
N_(2,e)(epsilon_e) >= ceil( L_e (M_(3,e)/(6 epsilon_e))^(1/3) ).
```

Therefore the conditional 2-jet boundary payload obeys the form

```text
B_2jet(epsilon)
  >= B_topology + B_junction + B_event + B_phase
     + sum_e [ B_initial,e
               + N_(2,e)(epsilon_e) (b_curvature,e + b_spacing,e) ].
```

For comparison, piecewise-linear/1-jet interpolation under `|f''|<=M_2` has error `O(M_2 h^2)` and sample count `N_1=O(epsilon^(-1/2))`; the quadratic/2-jet model has `N_2=O(epsilon^(-1/3))`. **This is a genuine asymptotic rate-exponent improvement on smooth `C^3` arcs.** It is not a numerical prediction until `M_3`, quantizer precision, `epsilon`, topology/event bytes, and parse-back survival are measured.

This is the precise, limited content of the “discard order >=3” analogy. The number-theoretic phrase “mod `p^3`” does not select a universal image-code truncation: a `p`-adic congruence and a real Taylor jet are different filtrations unless a formal model identifies their coefficient rings and valuation. Here the Taylor remainder, not the analogy, decides sufficiency.

### 2.3 Why 2-jets are not sufficient for the present v8 partition

The current corpus supplies three independent breakers.

1. **Corners, endpoints, and topology are not `C^2`.** The registered curvelet law is parabolic-optimal for a bounded-curvature `C^2` arc: `nu_parallel*=sqrt(nu_perp)`, so `sqrt(64)=8` matches the live allocation. The measured lane need is about `25`, a `3.125`--`3.2x` deficit caused by dash endpoints/gaps, i.e. codimension-2 singularities where the `C^2` hypothesis fails. A curvature jet on the smooth centerline does not say whether a dash exists.

2. **The lossless residual is not close to its generator.** **MEASURED n600:** the bit-exact curve-relative `(s,n)` coder loses to the absolute 2-D baseline: horizon `0.99x`; lane `0.90x`. Lane is only `60%` on-curve, has mean `|n|=96.4 px`, and its dominant Road/Lane stream remains `41,303 B`, rate `0.02750 S`. The problem is generator coverage, not omission of curvature. A second-order local model cannot code pixels that are not on the modeled branch.

3. **Sampling phase is independent state.** **MEASURED n600/source analysis:** subpixel advection through the fixed sampling comb creates `0.09--0.43 px` peak-to-peak boundary-position phase and a `0.005318` scored-frame spike rate. Curvature/topology can stay fixed while phase crosses a pixel/argmax cell. The phase trajectory must be transported/stored separately.

The smooth horizon is the favorable control, and it also supplies no new win. Its current dominant arc is already a degree-3 global polynomial: **MEASURED** `4,167 B`, `0.00277 S`, with residual sidecar explicitly owed. The separate macro-rate derivation already charges its moving intercept to the banked pose `xi`, predicting about `0.0003 S` for the frozen coefficients. A local 2-jet code is not proven smaller than that global cubic-plus-shared-`xi` form.

### 2.4 Sufficient-order and predicted-rate verdict

**2-jet sufficient?**

- **YES, conditional:** for one regular `C^3` edge, given initial frame, sampled curvature meeting the third-order error bound, and all topology/junction/phase data.
- **NO, requested full object:** not for the whole argmax partition, not for dash endpoints/island births/junctions, and not for current v8 receiver fidelity by itself.

**Predicted boundary rate:**

```text
conditional smooth-arc count:
N_(2,e)(epsilon) >= L_e (M_(3,e)/(6 epsilon_e))^(1/3)

admissible numeric delta versus current v8 coder:
UNKNOWN; banked delta = 0.
```

The existing measured rows remain the honest prediction baseline: geometric dominant-only `0.061 S`; Road/Lane `0.0275 S`; horizon `0.00277 S` before the separately derived shared-`xi` charge; curve-relative residual reformulation non-winning. Claiming a lower number from the jet analogy would be fake.

`verdict_scope` on “NO”: **FULL PARTITION x FINITE LOCAL-JET CODE x CURRENT GENERATORS**. The smooth-arc family remains alive behind a bounded `M_3/epsilon` measurement, but its likely consumer is a generator-coverage prior, not another residual coder.

---

## 3. Gauss--Bonnet as the assembly law

### 3.1 The exact planar partition formula

Let `Omega_c` be the region assigned to class `c`, with piecewise-`C^2` boundary oriented so the region lies to the left. For smooth edge segments `e` and corners/junction incidences `v`, planar Gauss--Bonnet gives

```text
sum_(e subset partial Omega_c) integral_e kappa_g ds
  + sum_(v subset partial Omega_c) alpha_(c,v)
  = 2 pi chi(Omega_c),
```

where `alpha_(c,v)` is the exterior turning angle and

```text
chi(Omega_c) = beta_0(Omega_c) - beta_1(Omega_c)
```

for planar regions. Image-domain boundary terms must be included when a class meets the frame edge.

This makes local-to-global assembly explicit:

- each local edge chart contributes an integrated curvature;
- each junction chart contributes a turning/incidence angle;
- their sum is the per-class topological invariant `2 pi chi`.

On a shared edge, the two class orientations are opposite, so their smooth curvature integrals cancel when summed over classes. The remaining global sum is controlled by outer-frame and junction/incidence terms. This is a strong **consistency law** for gluing local charts.

### 3.2 Why it cannot store the boundary

Every smooth simple closed plane curve has

```text
chi=1,     integral kappa ds = 2 pi,
```

regardless of its location, length, eccentricity, or high-frequency shape. Infinitely many scorer-distinct boundaries therefore share the same Gauss--Bonnet invariant. It cannot recover:

- where the boundary lies;
- its arc-length distribution or local curvature function;
- which pixels fall on which side;
- its sampling phase through `R`;
- the class paint needed to survive the frozen scorer.

The invariant costs little precisely because it discards the rate-bearing geometry.

### 3.3 Static core and island births

The repository's measured topology makes the scope concrete.

**MEASURED n600, frozen argmax advisory:**

```text
distinct class-adjacency graphs             11 / 600
components per frame                         35.5 +/- 4.9
Euler characteristic                         14.0 +/- 3.5
distinct full adjacency/component/Euler sig  573 / 600
small components per frame                    31.1
small-component pixel mass                    0.72%
d_seg from dropping fine islands               0.00705
```

The coarse topology is stable, but the fine island topology is volatile and scorer-relevant. Class 4/MyCar is the clean static-core case: `1.0 +/- 0.0` components in the n600 topology audit. The separate n96 hood artifact stores one majority-mask SDF in `56 B` with mean IoU `0.9944`, demonstrating that **static geometry**, not Euler characteristic alone, is what makes the core cheap.

For a simply connected class-`c` island born inside class `d`, the per-class Euler vector typically changes

```text
Delta chi_c = +1,     Delta chi_d = -1,
```

because the surrounding class gains a hole. The aggregate `sum_c chi(Omega_c)` can therefore remain unchanged: a scalar global Gauss--Bonnet checksum may be blind to the island birth. The **per-class Euler vector plus incidence/event mark** detects it, but still does not locate or shape it.

### 3.4 Storable-invariant verdict

- **YES:** store/derive per-class `chi`, component/hole counts, adjacency, and marked birth/death incidence as a tiny validation/event stream. Use Gauss--Bonnet to fail closed when local curvature/junction data do not assemble to those invariants.
- **NO:** do not replace the boundary, curvature function, event location, or phase stream with `chi`. The n600 `d_seg=0.00705` cost of dropping fine islands is the empirical warning.

**Campaign value:** a checksum and event gate, not a byte-reduction theorem. The static-core win is already explained by actual shared geometry; the island-birth problem requires event geometry/appearance, not merely the Euler delta.

`verdict_scope` on “NO”: **GAUSS--BONNET-INVARIANT-AS-STANDALONE-BOUNDARY-CODE**. Topology-aware carrier families remain intact.

---

## 4. What is a real v8/rate lever?

| sub-question | current result | campaign disposition |
|---|---|---|
| descent/gluing | exact conditional sequence; D36 not D38; global effectivity unmeasured | **REAL measurable accounting lever** after receiver cover/restrictions are typed; measure filler nonemptiness then packed filler bytes |
| 2-jet rate | conditional `epsilon^(-1/3)` smooth-arc sample law; no current numeric advantage | bounded reformulation probe only; measure `M_3`, receiver tolerance, singular/event mass, and compare exact packed bytes; existing negative prior is strong |
| Gauss--Bonnet | exact per-class assembly/checksum; no reconstruction | apparatus/checksum; **beautiful-but-inert as a codec** |

The recommended smallest proof is read-only/$0 on existing receiver artifacts:

1. Materialize one boundary CW complex and edge/junction cover from an existing exact argmax state.
2. Define actual restriction maps for a minimal carrier-state groupoid and enumerate pair/triple intersections.
3. Compute transition cocycles and the finite filler set `G_U(x)`.
4. If empty, record `INFEASIBLE` and identify which local carrier must change. Do not quote bits.
5. If nonempty, choose a deterministic filler, pack it, and compare exact receiver parse-back bytes with the global baseline.
6. In parallel only if useful, measure `M_3` and singular/event fraction; A/B a piecewise-quadratic arc stream against the already bit-exact absolute and curve-relative coders.
7. Validate every reconstructed class with the per-class Gauss--Bonnet checksum, but score the actual boundary/receiver output.

---

## 5. Triality, canonical-equation disposition, and apparatus

### DAG leg

Own-file FEED: `.omx/research/local_global_descent_dig_DAG_FEED_20260713.md`. The live canonical DAG was not edited because it is a sibling-hot surface.

### Equations leg

No new canonical equation is registered. That is deliberate, not an omission:

- the descent rate is conditional on a receiver cover/restriction/band that does not yet exist;
- the 2-jet rate needs measured `M_3`, quantizer, and receiver tolerance before it can predict bytes;
- Gauss--Bonnet is a classical checksum, not a new task-rate law.

The existing `rate_law_ladder_v2_measured` remains authoritative for D36/D38. Registering `H(q_G|U)=H(Theta_U|...)` would be false because no map between those variables is typed. The candidate piecewise rule

```text
R_glue(x,U) = INFEASIBLE,                         if G_U(x)=empty;
                  H(pi_0 G_U(x)|public),          otherwise
```

is held as `FORMALIZATION_PENDING` until the finite receiver-space probe instantiates every input.

### DSL leg

N/A. This is a proof/accounting pass, not a trainer actuator, loss weight, curriculum change, or launch configuration. No invented flag.

### Six-hook wire-in

1. **Sensitivity:** overlap/junction ambiguity and third-order boundary remainder become explicit sensitivity fields.
2. **Pareto:** charge actual packed filler/section bytes; a nonzero obstruction is infeasibility, not a favorable rate point.
3. **Bit allocator:** allocate only to noncanonical filler choices and actual boundary remainder, never to an abstract obstruction label by analogy.
4. **Autopilot:** fail closed on missing cover, restrictions, band, nonempty filler, deterministic section, or parse-back proof.
5. **Continual learning:** bank the D36-vs-D38 separation and the Gauss--Bonnet non-reconstruction negative in this dated memo/FEED.
6. **Probe disambiguator:** finite Cech-nerve probe arbitrates `empty` vs `nonempty/noncanonical` vs `canonical`; exact packed jet A/B arbitrates conditional rate benefit.

### Ownership and resumability

No run was launched. The shared lane/equation/DAG registries were already sibling-held; per operator scope this arm created only its own new research files and did not mutate those live surfaces. `lane_id=local_global_descent_dig` is recorded here and in the checkpoint stream; shared registry insertion is deferred to main review.

---

## 6. Hostile self-review

1. **Did this silently equate local carrier indices with a cover?** No. The memo labels the edge-tube cover `INFERRED FUTURE TYPING`; current v8 fields may be globally supported and therefore not local sections.
2. **Did it call a nonzero obstruction a bit cost?** No. Empty filler is infeasibility. Bits appear only after all obstructions vanish and a noncanonical filler remains.
3. **Could `147,616` still correlate with gluing complexity?** Possibly, but correlation is not equality. No `q_G -> Theta_U` map or sufficient-statistic proof exists; the current D36 receiver predictor already loses after charge.
4. **Is “2-jet sufficient” contradicted by Frenet reconstruction?** No. Frenet proves sufficiency of the *continuous curvature function plus initial data* on a regular component. The requested finite codec lacks that function unless sampled with a third-order remainder bound and topology/event/phase side information.
5. **Is the `epsilon^(-1/3)` law overclaimed?** It is a sufficient sample-count scaling on a `C^3` graph chart, not a universal entropy lower bound and not a measured archive prediction. The memo labels it conditional.
6. **Could Gauss--Bonnet plus all local curvature reconstruct the curve?** Local curvature as a function plus initial data can; Gauss--Bonnet only supplies its integral/compatibility condition. The rate lives in the function and initial/junction data, not the integral.
7. **Does static MyCar prove topology is enough?** No. The 56-B result stores the actual majority silhouette. `chi=1` alone would not reconstruct it.
8. **Empirical authority?** All empirical numbers retain original n600/n96 advisory scope. This arm adds no scorer or contest row and moves no pointer.

---

## 7. Stores consulted

`CLAUDE.md` (full) · `AGENTS.md` (full) · `docs/operating_manual_craft_handoff.md` (full) · top project `MEMORY.md` entries · current directives and sibling ownership/checkpoints · graph-memory recall for D38/descent and jet/topology · `SPEC_v75_optimal_single_trunk_20260708.md` · `SPEC_v8_perclass_decomposition_20260708.md` · `perclass_carriers_design_20260708.md` · `bousfield_localization_dig_20260713.md` · `bousfield_deep_read_20260713.md` · `ladder_owed_measurables_20260713.md` · `ladder_owed_measurables_DAG_FEED_20260713.md` · `rate_law_ladder_20260713.py` · `rate_law_ladder_measured_20260713.py` · `einstein_pass_covariance_laws_20260710.py` · `dpose_covariance_mirror_audit_20260711.md` · `residual_kit_deshare_curverel_build_20260709.md` · `curve_relative_offset_coder.py` · `v8_macro_rate_pass_20260710.md` · `v8_roadlane_ego_compensated_rate_20260709.md` · `road_undriv_bulk_field.py` · `v8_laguerre_generator_feasibility_and_perclass_hybrid_20260710.md` · `flicker_transform_geometry_term_design_20260710.md` · `cgauge_parametrization_optima_20260711.py` · `deepmath_lens_tropical_ot_powerdiagram_20260704.md` · `scaling_law_facet2_intrinsic_manifold_parametrization_20260704.md` · `frozen_partition_topology_ego_deformation_20260623.md` · `hood_static_component_20260627T071150Z.md` · latest sister Codex finding/session memo and latest available T3/design memo.

## Final scoped verdict

**Local-to-global descent is the right formal frame for v8 reconciliation, but the current global obstruction is neither measured nor equal to 147,616 bits. The 2-jet analogy yields a conditional smooth-arc sample law, not a present boundary-rate win. Gauss--Bonnet yields a cheap assembly checksum, not the separatrix. Overall: one real future measurement (descent filler), zero banked rate reduction, pointer unmoved.**
