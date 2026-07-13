# The rate-law ladder's owed measurables — D36–D39

**Date:** 2026-07-13
**Role:** SOL xhigh, operator-directed follow-up to the grokking/symmetry/algebra digs
**Checkpoint:** `ladder_measurables`
**Execution:** `$0` local CPU; n600; frozen inputs read-only; no render, training,
provider, GPU, live-run, cache, pointer, or shared deferral-ledger mutation
**State:** uncommitted by operator instruction
**Pointer:** UNMOVED; these are codelength/derivation/spec results, not an exact
contest score row

## Verdict first

| debt | result | error / scope | what changes |
|---|---|---|---|
| **D36 `fiber_completeness_gap_n600`** | **MEASURED operational codelength: 147,616 bits = 18,452 B = 22.116745% of the 83,430 B archive rate term** (`0.0122864294` score-rate units). It is 90.650946% of the current 20,355 B code section. | Five disjoint held-out folds, centered on the exact full Brotli stream: 95% fold-sensitivity interval **[146,913.96, 148,318.04] bits**. This is an upper bound on `H(q_G|U_proxy)`, not the unmaterialized exact canonical entropy. `U_proxy` is evaluator-side and not public to `inflate`. | The ideal conditional stream saves 1,903 B before the predictor, but the conservative predictor charge is 15,256 B and `U_proxy` signalling/derivation is uncharged. **No receiver lever is admitted.** A section predictor must be receiver-public/derivable and cost <1,903 B on this instance. |
| **D37 `I(F;C|M,xi)`** | **POSITIVE / CLASS-CONDITIONING-ADMITTED on the scoped surface.** `q00-q01` gross gain **401,322.03 bits** = **0.157296 bit/boundary-pixel**; after an explicit 10,342 B flat-table charge, net **318,586.03 bits**, 95% pair-bootstrap **[306,950.21, 329,649.88]**. | n600, 2,551,382 local boundary pixels, 1,050,200 receiver flips; real cached realized-through-`R` epoch-50 witness surface. `C`, `Qxi`, and `Phi` completions are `ASSUMED`; no inference to current mod32cap or contest CUDA/CPU. | Reject universal class-blind sufficiency on this surface. Queue one **coarse class-edge conditional grammar** exact packed A/B, but only when `C` is already decoded/derivable or jointly charged. The phase-aware flat table is not admitted: gross +407,978.06 bits but **net -44,437.94 bits** after its 56,552 B table. |
| **D38 `h2_obstruction_after_typing_Hcov`** | **LOCAL SPLIT / GLOBAL NOT-TYPED.** On a fixed regular isotropy chart, the explicitly typed strict extension is `K_sigma,x semidirect H_cov,sigma,x`; `h -> (1,h)` is a homomorphic section, the factor set is neutral, and `R_twist^ideal=0`. | `DERIVED`, `verdict_scope=STRATUM x FORMULATION`. Nontrivial action is not a non-split extension. Changing isotropy/topology and absent overlap maps prevent a single global obstruction class from being typed. | Allocate **zero local ideal twist bits** for the strict chart. Do not make action, chart, gluing, or statistical-independence bytes free. Replace the old obstruction question with a typed overlap/gluing audit. |
| **D39 `event_marks_telemetry`** | **SPECIFIED, BUILD-OWED.** Additive `event_mark` row under `pact.causal_manifest.v1`, with family, class edge, spacetime location, attachment/incidence, before/after stratum, receiver state, evidence, and stable resume identity. | Spec-only; `src/tac/causal_manifest.py` deliberately unchanged. | Implement [the marked-event increment](pact_causal_manifest_v1_event_marks_increment_spec_20260713.md) under [TICKET-D39](TICKET_D39_event_marks_telemetry_20260713.md); reject binary/count-only event telemetry. |

## 0. Registered estimators and stores consulted

The four rung memos were treated as the estimator registry, not as optional
background:

- `weyl_symmetry_group_unification_20260713.md`;
- `infdesc_foundations_dig_20260713.md`;
- `garrett_algebra_dig_20260713.md`;
- `condprob_homotopy_lie_dig_20260713.md`.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`;
`docs/operating_manual_craft_handoff.md`; current canonical frontier, lane,
subagent, equation, pool, deferral, and causal-manifest surfaces; the four rung
memos; `rate_law_ladder_v1`; the frozen n600 GT/scorer cache; the real n600
receiver argmax cache; the current mod32cap EMA-best checkpoint; and its exact
byte-close report. Sibling-owned caches and source files were read-only.

The reusable measurement is
`tools/measure_rate_law_ladder_owed.py`. Its outputs are:

- `experiments/results/ladder_owed_measurables_20260713/d36_fiber_completeness_gap_n600.json`;
- `experiments/results/ladder_owed_measurables_20260713/d37_flip_conditional_mi_n600.json`;
- `experiments/results/ladder_owed_measurables_20260713/receipt_manifest.json`.

The receipt hashes all four frozen custody inputs. The two large NPZ inputs are
memory-mapped as `ZIP_STORED` members and are never rewritten.

## 1. D36 — `H(q_G(W)|U(W))` by conditional codelength

### 1.1 Registered law

Rung 2 proves, for canonical statistic `U` and constructive orbit label `q_G`,

\[
H(q_G(W))-H(U(W))=H(q_G(W)\mid U(W))\ge0.
\]

The equality is exact only when `U` is a function of `q_G`. The repository does
not currently persist a finite canonical `q_G` label or a full finite `U` row for
each witness. The measurement therefore needs an operational completion.

### 1.2 Operational completion

The following judgments are **ASSUMED**, not silently promoted:

1. `q_G(W)` is represented by the **actual shipped symmetric-int8 per-frame code
   table** in the current mod32cap EMA-best checkpoint: shape `(1200,32)`, 38,400
   symbols, Brotli-q11 code section 20,355 B. It is a concrete constructive-atlas
   label proxy, not proof of orbit minimality.
2. `U_proxy(W)` is a lossy finite feature of the evaluator statistic: 4x4
   per-class frozen SegNet-label area fractions, directed horizontal/vertical
   label adjacencies, and the official cached PoseNet six-vector per pair—111
   predictors total.
3. `U_proxy` is **evaluator-side side information**, not public state available
   to `inflate.sh`. Its absence at the receiver is a fail-closed deployment
   blocker and its signalling cost is not hidden in the model charge.

This completion measures a reproducible **model-codelength upper bound** on the
registered gap. It does not claim to estimate the exact full-statistic conditional
entropy without bias.

### 1.3 Estimator

- outer folds: five deterministic pair blocks (`pair_id mod 5`);
- inner folds: three pair blocks, selecting ridge `alpha` from
  `{0.1,1,10,100,1000}` by held-out squared prediction error;
- prediction: round and clamp each held-out code symbol to int8;
- residual: `(actual-prediction) mod 256`, which is decoded bit-exactly;
- codelength: Brotli quality 11 on the complete cross-fitted residual stream;
- fallback: `min(raw code, conditional residual)` so an overfit model cannot
  manufacture a worse headline conditional code;
- model charge: explicit conservative int16 coefficient matrix plus scale/header,
  15,256 B. It excludes `U_proxy` carriage, which remains a named blocker.

### 1.4 Measurement and error accounting

```text
unconditional code stream       20,355 B = 162,840 bits
conditional residual stream     18,452 B = 147,616 bits
gross conditional saving         1,903 B =  15,224 bits
conservative predictor charge   15,256 B
net versus raw after predictor -13,353 B = -106,824 bits
U_proxy receiver cost           UNCHARGED / BLOCKER
```

The full archive is 83,430 B and its rate term is `0.055552612459`.
The measured gap proxy is therefore 22.116744576% of exact archive bytes/rate,
or `0.012286429403` rate-score units. It occupies 90.650945714% of the current
code section.

An iid row bootstrap is invalid for a dictionary coder because duplicated rows
create artificial Brotli matches. Error is instead the disjoint outer-fold
codelength spread, extrapolated to n600 and centered on the exact full-stream
length:

```text
centered fold bits: 148,360; 146,880; 147,480; 147,960; 147,400
SE: 252.856 bits
t(df=4) 95% fold-sensitivity interval: [146,913.96, 148,318.04] bits
```

This interval accounts for pair/fold heterogeneity and finite-stream ordering;
it does not remove the one-sided bias from lossy `U_proxy` or pay receiver side
information. The result is accordingly **MEASURED-SCOPED**, not an entropy
identity or legal archive saving.

### 1.5 Consumer decision

D36 changes the section-engineering budget, not the pointer. The present linear
section does not pay. A successor is admissible only if it:

1. derives `U` from public/already-decoded receiver state or charges it;
2. makes its predictor/grammar cost less than the measured 1,903 B gross saving;
3. decodes the same int8 symbols bit-exactly; and
4. improves exact archive bytes after ZIP/container overhead.

No pool arm is emitted for D36.

## 2. D37 — nested `I(F;C|M,Qxi)` conditional codelength

### 2.1 Registered estimator

Rung 4 registered two held-out contrasts:

\[
\widehat{\Delta L}_{MX}
=\sum_i[-\log_2q_{00}(F_i|M_i,X_i)
+\log_2q_{01}(F_i|M_i,X_i,C_i)],
\]

then the phase-aware `q10(F|M,X,Phi)` versus
`q11(F|M,X,Phi,C)`. Flexible calibrated models, pair-blocked nested
cross-fitting, pair bootstrap, per-class-pair calibration, parameter/table
bytes, and a normalized Jacobian/velocity diagnostic were mandatory.

### 2.2 Data and `ASSUMED` completion

- `F` (**MEASURED**): cached realized-through-`R` frozen SegNet argmax differs
  from frozen GT `lstar` on the epoch-50 `witness_perclass_baseline_n600` surface.
- sample support: every GT four-neighbour class boundary pixel, once; if several
  unlike neighbours exist, choose the one with the smallest stored margin.
- `M` (**MEASURED**): frozen top-1/top-2 margin.
- `C` (**ASSUMED**): directed `(GT winner, selected adjacent GT class)` edge.
  Runner-up logits were not retained, so this is a local active-edge proxy, not
  the exact logit runner-up. It is also not automatically receiver-public.
- `Qxi` (**ASSUMED**): train-fold PCA of the official cached PoseNet six-vector,
  quantile-binned in its first two coordinates.
- `Phi` (**ASSUMED**): `M_self/(M_self+M_adjacent)` as a local receiver tie
  coordinate.
- diagnostic (**ASSUMED proxy**): normalized margin `M/|grad M|` and
  `|pose_xy dot normalized grad M|`. This is not an exact scorer Jacobian.

The instance contains 2,551,382 boundary pixels and 1,050,200 receiver flips,
a boundary flip rate of 0.4116200553.

### 2.3 Model and nesting

The proper codelength models are beta-binomial context tables. Candidate context
grids are selected inside each outer training fold from:

```text
(margin bins, Qxi bins/PC, Phi bins) = (8,2,4), (16,2,8), (24,3,8)
```

All five outer folds selected `(16,2,8)`. The non-phase model omits `Phi` even
though the tuple records its candidate value. One pseudo-observation anchored
at the training base rate prevents zero-probability test symbols. No held-out
pair contributes to its probability table or quantizer fit.

The explicit model charge is **ASSUMED fixed serialization**:
`uint16 context_key + uint32 n0 + uint32 n1`, plus a 32 B header. This charges
tables but **not a class-context symbol sequence**. Therefore a codec may consume
the gain only if `C` is already decoded/derivable at that point; otherwise it
must jointly signal `C` and remeasure net bytes.

### 2.4 Result

| contrast | gross class gain | model charge | net class gain | 95% pair-bootstrap net CI | verdict |
|---|---:|---:|---:|---:|---|
| `q00-q01`: `M,Qxi` | 401,322.03 bits (50,165.25 B) | 10,342 B | **318,586.03 bits (39,823.25 B)** | **[306,950.21, 329,649.88] bits** | class conditioning admitted on scoped surface |
| `q10-q11`: `M,Qxi,Phi` | 407,978.06 bits (50,997.26 B) | 56,552 B | **-44,437.94 bits (-5,554.74 B)** | **[-55,850.72, -33,431.47] bits** | flat phase-aware class table does not pay |
| normalized scale/velocity diagnostic | 380,618.00 bits | 175,262 B | **-1,021,478.00 bits** | negative | class dependence persists grossly; naive high-dimensional table explodes |

The primary gross estimate is `0.1572959402 bit/boundary-pixel` or
`0.3821386693 bit/observed boundary flip`. Pair-bootstrap uses 10,000 resamples
of the 600 pair totals. Weighted absolute class-pair calibration residual drops
from `0.152270` class-blind to `0.000187` class-aware in the primary model.
That large reduction is consistent with the positive codelength verdict, but
the exact per-pair and per-class rows remain in the receipt rather than being
compressed into one headline.

### 2.5 Verdict and consumer decision

**D37 MI verdict:** `I(F;C|M,Qxi)` is positive on this completed real n600
surface; the lower confidence bound remains positive after the stated coarse
table charge. Universal class-blind sufficiency is rejected at
`FORMULATION x EMPIRICAL-SURFACE` scope.

This does **not** authorize the flat phase table, and it does not prove a legal
39,823 B archive saving. The emitted pool FEED admits only a coarse
class-edge-aware conditional contour grammar. Its next gate must preserve the
same innovation alphabet, charge/jointly code `C`, decode bit-exactly, survive
the real receiver, and beat the registered `0.65 B/surviving corrected flip`
gate. The pool proposal is isolated in
`ladder_owed_measurables_pool_FEED_20260713.jsonl`; the shared pool is left for
main to append.

## 3. D38 — type the extension, then derive split versus twist

### 3.1 Typed fixed-stratum object

Fix a regular stratum `sigma=(kappa,omega,a,r)` and an object `x` in that
stratum. Let:

- `K_{sigma,x}` be the isotropy group at `x` of the wide kernel subgroupoid
  generated by the rung-1 join of blind, resize-null, YUV-null, Pose-fiber,
  argmax-cell, and declared photometric arrows that preserve the chart;
- `H_{cov,sigma,x}` be the group of admissible covariance stabilizer arrows at
  `x` that remain inside `sigma`;
- `alpha_{sigma,x}: H_{cov,sigma,x} -> Aut(K_{sigma,x})` be conjugation,
  `alpha_h(k)=hkh^{-1}`.

**ASSUMED typing completion:** restrict `H_cov` to arrows for which conjugation
preserves `K_{sigma,x}` and composition/object compatibility holds. This is the
minimum closure condition required by rung 1; without it the `semidirect` symbol
is only suggestive notation.

Now define

\[
E_{\sigma,x}=K_{\sigma,x}\rtimes_{\alpha}H_{cov,\sigma,x},
\qquad
(k_1,h_1)(k_2,h_2)
=(k_1\alpha_{h_1}(k_2),h_1h_2).
\]

Projection and inclusion give the typed exact sequence

\[
1\to K_{\sigma,x}\xrightarrow{k\mapsto(k,1)}
E_{\sigma,x}\xrightarrow{(k,h)\mapsto h}
H_{cov,\sigma,x}\to1.
\]

The section `s(h)=(1,h)` satisfies

\[
q\circ s=id,\qquad s(h_1)s(h_2)=s(h_1h_2).
\]

Therefore the extension **splits by construction**. Its factor set is

\[
\omega(h_1,h_2)=s(h_1)s(h_2)s(h_1h_2)^{-1}=1.
\]

For abelian `K`, `[omega]=0` in the typed `H^2`; for the actual nonabelian joined
kernel, the full Schreier factor-system class is the pointed neutral element.
A nontrivial `alpha` makes the product non-direct, not non-split.

### 3.2 Local rate consequence and global boundary

On this strict fixed-stratum formulation,

\[
R_{twist}^{ideal}=H(Theta\mid q_H,\mathcal A,public)=0.
\]

This does not imply `H(K,H)=H(K)+H(H)`; the correct rate is still
`H(H)+H(K|H)`. Nor does it make action/atlas realization bytes zero.

The wide global object is a different problem. Topology and receiver-cell events
change objects, isotropy rank, and sometimes the kernel itself. The repository
has no typed overlap restriction maps, coefficient bundle, or transition factor
sets joining the fixed-stratum extensions. Consequently:

- local strict formulation: **DERIVED SPLIT**, neutral obstruction, zero ideal
  twist term;
- current global artifact: **NOT-TYPED / GLUING-AUDIT-OWED**;
- nonzero global `H^2`/Schreier obstruction: **NOT DERIVED**.

The bit allocator may remove a local ideal twist budget only for the strict
chart. It must retain chart, action, marked-event, overlap, and receiver-section
costs until measured or public-derived.

## 4. D39 — marked-event telemetry

The complete additive row contract is in
`pact_causal_manifest_v1_event_marks_increment_spec_20260713.md`; implementation
is isolated in `TICKET_D39_event_marks_telemetry_20260713.md`.

The proposed `event_mark` row carries:

1. the priority-partitioned family `topology | chart | receiver_lattice` and
   family-specific kind;
2. directed class edge or junction class set;
3. quantized spacetime location and support;
4. before/after component and junction incidence plus attachment rule;
5. before/after common-refinement stratum `(kappa,omega,a,r)`;
6. actual `R`, uint8, `Qxi`, phase, and prediction-chart identifiers;
7. evidence custody, receiver derivability, stable id, and resume key.

Strict validation rejects event counts/bits without those marks. The row remains
`[observability-only] NON-PROMOTABLE`. No manifest-module edit occurred in this
lane.

## 5. Proposed deferral updates — main flips the ledger

The shared deferral ledger was intentionally not edited.

| row | current debt | proposed main action | successor debt |
|---|---|---|---|
| D36 | measure `H(q_G|U)` | close as **MEASURED-SCOPED operational upper bound**, anchored to D36 receipt | receiver-public/derivable section predictor with total charged overhead <1,903 B; exact archive A/B |
| D37 | nested n600 conditional codelength | close as **MEASURED-SCOPED; class dependence positive** | exact packed class-aware contour A/B with `C` derived or jointly charged |
| D38 | type extension and decide split/twist | close fixed-stratum formulation as **DERIVED-SPLIT / neutral** | global overlap/coefficient-bundle/gluing audit; no inferred nonzero obstruction |
| D39 | marked-event telemetry | mark **SPEC-COMPLETE / BUILD-OWED** | execute TICKET-D39 and seal strict parser/writer/tests |

## 6. Equation, DAG, pool, and cathedral closure

- **Equations leg:** `rate_law_ladder_v2_measured` in
  `src/tac/canonical_equations/rate_law_ladder_measured_20260713.py` anchors the
  numeric gap/MI rows and scoped D38/D39 statuses without overwriting v1.
- **DAG leg:** `ladder_owed_measurables_DAG_FEED_20260713.md` makes custody,
  charged overhead, exact receiver A/B, gluing, and schema implementation
  explicit dependencies.
- **DSL leg:** D36/D37 are lossless grammar candidates, not training scalars;
  `dsl_na_reason` is explicit in the isolated pool row. D38/D39 are law/schema
  work, not levers.
- **Pool:** one row falls out—coarse class-pair conditional flip tables. D36 and
  both high-cardinality D37 variants fail charged-overhead admission.
- **Continual-learning outcomes:** canonical probe rows
  `ladder_d36_fiber_gap_n600_20260713` (`PARTIAL`, advisory) and
  `ladder_d37_flip_conditional_mi_n600_20260713` (`PROCEED`, advisory) preserve
  the scoped measurements and exact reactivation gates.
- **Pointer delta:** none.
- **Triality honesty:** equation and DAG are landed; no fake DSL lever was
  invented; D39 implementation and all exact packed archive gates remain owed.

`verdict_scope: D36 INSTANCE x MODEL-CODELENGTH; D37 FORMULATION x EMPIRICAL-SURFACE;
D38 STRATUM x FORMULATION; D39 SPECIFICATION. No family kill and no score promotion.`
