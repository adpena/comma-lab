# MDL of the frozen n600 digital argmax complex — premise-falsification measurement (2026-07-18)

`research_only=true`

`lane_id=lane_mdl_ms_complex_k_lower_bound_20260718`

`authority=[macOS-CPU advisory] NON-PROMOTABLE`

`score_claim=false` · `promotion_claim=false` · `execution_used=false`

`frontier_pointer_delta=NONE` · `sacred_c2_access=NONE`

## Verdict first

**FALSIFIED_AT_CLAIM_LEVEL [DERIVED]: a concrete MDL/code length is not a lower bound on
individual universal Kolmogorov complexity. It is a model- and decoder-relative constructive
upper bound.** Consequently, this cached-n600 pass does not establish the requested numeric
`MDL_bytes lower-bound-on-K`, and it cannot decide whether universal `K` is below or above the
strict sub-`0.15` rate ceiling.

The closest inherited project-model number is:

| quantity | Seg bytes | temporal-ξ bytes | total bytes | rate-only term | custody |
|---|---:|---:|---:|---:|---|
| optimistic digital-complex model | **228,779 [DERIVED]** | **7,195 [MEASURED]** | **235,974 [DERIVED]** | **0.157125400604051262 [DERIVED]** | `/2` shared-edge estimate + 15-byte palette assumption + real self-contained lossy-ξ section; not an emitted lossless complex code, not receiver-closed, not K |

The strict largest integer archive size with rate-only contribution `< 0.15` is **225,272 B
[DERIVED]**. The optimistic model is **10,702 B [DERIVED]** above that integer ceiling
(**10,701.066 B [DERIVED]** above the continuous crossing). This says only that this particular
optimistic model misses the ceiling. It does **not** imply that a shorter program cannot exist.

The decisive readout is therefore:

`NO_DECISIVE_K_THRESHOLD_VERDICT [DERIVED]`.

## 1. Why the requested inequality reverses

Let `T` be a target, `c_C(T)` a lossless code in a declared code family `C`, and `D_C` its fixed
decoder. A universal machine can run that decoder on the code, so

\[
K_U(T\mid D_C) \le |c_C(T)| + O_U(1),
\qquad
K_U(T) \le K_U(D_C)+|c_C(T)|+O_U(1).
\]

These are **DERIVED theorem relations**. Padding any valid code with ignored bytes is already a
counterexample to treating every description length as a lower bound. Declaring the generator
“free” changes the conditioning/decoder term; it does not reverse the inequality.

There is a valid lower-bound-shaped relation, but it contains an uncomputable quantity. Keep the
evaluator statistic distinct from a carrier: let `T_E=(S,P)`, where `S` is the SegNet argmax
partition and `P` is the frozen PoseNet output. If fixed evaluator `E` maps every exact witness `Y`
to `T_E`, then data processing gives

\[
K_U(T_E\mid E) \le K_U(Y\mid E)+O_U(1).
\]

Thus the true `K_U(T_E|E)` would lower-bound exact-witness complexity up to a machine constant.
No concrete code length measured here can replace it: the code proves an achiever from above.
Individual Kolmogorov complexity is uncomputable, and a nontrivial lower bound would require
excluding **every** shorter program, not merely measuring one grammar. This agrees with the prior
fresh-eyes correction that `K/H=0.47` was a codec-pair description ratio, not Kolmogorov complexity
or global optimality (`sol_ultra_v10_true_final_form_review_20260717.md`, lines 166–172), and with the
prior floor audit that every measured achiever is an upper bound
(`information_theoretic_floor_T_floor_20260610.md`, lines 180–195).

**Universal numeric lower bound [DERIVED]: `TRIVIAL_NONNEGATIVITY_ONLY`.** Standard `K` is expressed
in program bits and depends on the chosen universal machine up to an additive constant. A positive
archive-format minimum is a container constraint, not a measured universal-K byte bound.

## 2. What object was actually available

The frozen cache supplies a raster label field, not a continuous Morse function:

\[
S_i(x) = \operatorname*{argmax}_{k\in\{0,\ldots,4\}} z_{i,k}(x),
\qquad i=1,\ldots,600.
\]

From `S`, a fixed decoder may derive digital cell membership, crack edges, adjacency, connected
components, and multi-label raster junctions without duplicating payload. But `lstars` does not
contain a scalar potential, critical gradients/Hessians, transversality certificate, full
runner-up logits, or exact continuous tie loci. Therefore:

- `digital frozen-scorer argmax cell complex` — **CONFIRMED [MEASURED/DERIVED]**;
- `classical Morse–Smale complex` — **NOT ESTABLISHED [DERIVED]**;
- `global exact Laguerre/power diagram` — **NOT ESTABLISHED [DERIVED]**.

The latest scorer-pullback audit makes the same distinction: triple-label raster junctions become
classical Morse critical points only after the missing potential/nondegeneracy/transversality proof
(`collateral_coupling_geometry_and_film_flicker_sidecar_20260718.md`, lines 141–180). The #284/v8
probe measured that Laguerre weights materially help a finite fitted family, but its global diagram
still flattened near a `7e-3 [MEASURED]` label residual at `K=1024 [MEASURED]`; this supports a
coarse structured carrier, not an exact representation theorem.

## 3. Input and class-order custody

All inputs were read-only. No scorer, renderer, trainer, live run, or provider was invoked.

| field | value | label |
|---|---|---|
| cache | `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` | INPUT |
| cache bytes | 5,078,017,610 B | MEASURED |
| cache SHA-256 | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` | MEASURED |
| `lstars` shape / dtype / range | `(600,384,512)` / `int64` / `[0,4]` | MEASURED |
| semantic `uint8` class-label serialization SHA-256 | `f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557` | MEASURED; fail-closed tool contract |
| storage-preserving little-endian `int64` serialization SHA-256 | `bf1d0e5c7e2ef1b3c38ce6cd51ec827169ac13a02c675638b7bd97344a089ec4` | MEASURED |
| `gt_poses` shape / dtype | `(600,6)` / `float64` | MEASURED |
| canonical `gt_poses` serialization SHA-256 | `bee5821eeb892ad430878946dfcdf365e1a14202efb7cd1af2b019d75da0f481` | MEASURED |

The canonical class order was held fixed; no luma sort or spatial relabel was permitted:

| id [INPUT] | class [INPUT] | pixels [MEASURED] | fraction [DERIVED] |
|---:|---|---:|---:|
| 0 | Road | 27,407,046 | 0.23233240763346355 |
| 1 | Lane | 690,639 | 0.005854619344075521 |
| 2 | Undrivable | 58,413,281 | 0.49517551845974395 |
| 3 | Movable | 1,460,325 | 0.012379328409830729 |
| 4 | MyCar | 29,993,509 | 0.2542581261528863 |

Every count in that table is **MEASURED**; each fraction is **DERIVED** from that count and the
**117,964,800-pixel denominator [DERIVED from shape; confirmed by measured counts]**. This catches a material
legacy hazard: old prose sometimes called class `2` Road; that label is false under the canonical
comma10k ordering used here.

The durable machine receipt is
`.omx/research/mdl_ms_complex_K_lower_bound_20260718.json`, SHA-256
`5636137fa2ad57b222e076aa873b5cf6523ab51b25298a5fa0df11b6ce56d86b [MEASURED]`.
The measurement tool SHA-256 recorded by that receipt is
`5b63909992b2eaf6c5c822d9651066f30c735cba28f3e4bd61178cf89217e5be [MEASURED]`.
A second full invocation produced byte-identical receipt bytes **[MEASURED]**. The accepted full run
took `6.05 s [MEASURED wall-clock, local macOS CPU advisory]`; timing is operational evidence only.
`py_compile` passed **[MEASURED]**, Ruff passed **[MEASURED]**, and the existing context-codec suite
reported `18 passed [MEASURED]`.

## 4. Concrete description-length receipts

### 4.1 Real exact Seg partition code

The existing `context_partition_codec` emits one self-describing temporal-context range-code payload,
including its shared model and header. On all n600 cached labels:

| component | bytes | label |
|---|---:|---|
| transmitted context model | 442 | MEASURED |
| arithmetic stream | 254,824 | MEASURED |
| complete exact Seg payload | **255,288** | **MEASURED** |
| amortized | 425.48 B/frame | DERIVED |
| payload SHA-256 | `db0a925c666edc1e5124ee4dcf4fe2f0c66ee22140939bd78ebfac0410c9f4a4` | MEASURED |

A deterministic **2-frame [INPUT]** prefix encode/decode roundtrip passed bit-exactly **[MEASURED]**;
the codec's own
test suite guards the full format. This row is a **declared-code-family upper bound on the exact
partition**, conditional on the fixed decoder. It is not a lower bound and it is not yet an RGB
receiver archive.

### 4.2 The inherited “K-ladder” is not a complete code

The exact-geometry `eps=0 [INPUT/MEASURED]` calibration row records:

| item | bytes/value | label and scope |
|---|---:|---|
| polygon vertices | 928,868 | MEASURED |
| raw coordinate payload | 3,776,524 B | MEASURED inner stream |
| Brotli-q11 coordinate payload | **457,528 B** | MEASURED inner stream |
| post-Brotli `/2` value | **228,764 B** | DERIVED HEURISTIC ESTIMATE |
| geometric reconstruction distortion | 0.0 | MEASURED in-memory label reconstruction; not stream decode |
| realized frozen-SegNet distortion after palette/R | 0.04538490295410156 | MEASURED; nonzero |

The serializer in `tools/necessity_dseg_calibration.py` emits a flat sequence of first vertices,
polygon lengths, and deltas, with no demonstrated frame/class/component framing; it then divides the
nonlinear Brotli result by two. The quotient is not an emitted shared-edge code. The earlier
`143,552.5 B [DERIVED]` `eps=1` value is additionally lossy with
`dseg_geo=0.0026144917805989583 [MEASURED]`; it is excluded from the exact-corner accounting.

The necessity solver remains valuable for structure: **97.8% [MEASURED]** cell interiors,
**6,703 [MEASURED]** digital junctions, and **11.2/frame [DERIVED]** junction density; its own memo
already scopes the chain lengths as spatial tolerances rather than score claims
(`necessity_solver_inverse_factorization_20260715.md`, lines 34–57 and 85–105). It does not provide
a lower-bound proof.

### 4.3 Temporal-ξ custody is real but not an exact pose solution

The existing n600 byte-close receipt supplies:

| item | value | label |
|---|---:|---|
| entropy-coded ξ payload | 6,634 B | MEASURED |
| complete counted ξ section | **7,195 B** | MEASURED |
| raw-ξ reference | 7,232 B | MEASURED |
| quantizer levels | 4,096 | MEASURED configuration |
| realized `d_pose` | **0.0016095471538913576** | MEASURED `[macOS-CPU advisory]` through the real byte-closed decode |

The complete 7,195-byte section, rather than its 6,634-byte inner payload, is the honest composition
charge. It is quantized and has nonzero realized Pose distortion. It therefore cannot be used as
lossless Ξ custody for an exact `(d_seg,d_pose)=(0,0)` corner.

The cache's `gt_poses` is a frozen-PoseNet six-vector target. It is **not** the trained temporal twist
`ξ = xi_stored + dxi`; treating those arrays as interchangeable would be a coordinate-custody error.
The optional fixed XZ roundtrip of canonical cached `gt_poses` is explicitly
`NOT_RUN_OPTIONAL [MEASURED EXECUTION STATUS]` in the companion receipt: it would be only a diagnostic
declared-family upper bound, not temporal ξ and not a receiver, so it is excluded from this model total.

### 4.4 Honest split of the optimistic project model

Reusing the necessity model without silently upgrading it:

| inherited model section | charged bytes | label / assumption |
|---|---:|---|
| parsed edge-coordinate seed | 228,764 B | DERIVED post-Brotli `/2` estimate; not emitted |
| cell labels / five RGB palettes | 15 B | INFERRED project-model charge |
| adjacency graph | 0 B | INFERRED zero-marginal assumption: derive incidence from a fully parsed edge graph |
| digital junctions / raster tie-locus vertices | 0 B | INFERRED zero-marginal assumption: derive curve intersections; precision charged to edge seeds |
| generic graph/curve decoder and rasterizer | 0 counted B | INFERRED under RULE-118 CONTRACT ASSUMPTION |
| self-contained temporal-ξ section | 7,195 B | MEASURED |

The two zero-marginal structural assumptions are **not validated** for the current inner contour
stream, because that stream has no demonstrated framing or parse-back decoder. Exact continuous
scorer tie loci remain unavailable from `lstars`; only digital adjacency/junctions can be derived
from an exact raster or exact parsed boundaries.

\[
L_{seg}^{model}
=228{,}764\ [\mathrm{DERIVED\ /2\ estimate}]
+15\ [\mathrm{INFERRED\ five\ RGB\ palette\ bytes}]
=228{,}779\ \mathrm{B},
\]

\[
L_{joint}^{model}
=228{,}779\ [\mathrm{DERIVED}]
+7{,}195\ [\mathrm{MEASURED\ xi\ section}]
=235{,}974\ \mathrm{B}\ [\mathrm{DERIVED}].
\]

This is labeled `OPTIMISTIC_PROJECT_MODEL_DESCRIPTION_LENGTH`, not `MDL lower-bound-on-K`.
Its missing custody is cumulative: no self-delimiting shared-edge stream, no exact MS generator proof,
nonzero realized Seg distortion, nonzero realized Pose distortion, and no complete receiver/archive.

For a more concrete upper-bound comparison, exact Seg context code plus the actual ξ section costs
`255,288 + 7,195 = 262,483 B [DERIVED]`, with rate-only term
`0.174776655592366902 [DERIVED]`. That composition still is not an exact joint target because the ξ
section is lossy and the RGB receiver leg is absent.

## 5. Strict sub-0.15 calculation

At zero distortion, the rate-only condition is

\[
25B/37{,}545{,}489 < 0.15
\iff B < 225{,}272.934.
\]

All quantities in this display are **DERIVED** from the contest formula. Integer custody gives:

| bytes | rate-only term | threshold relation | label |
|---:|---:|---|---|
| 225,272 | 0.149999378087737784 | strictly below | DERIVED |
| 225,273 | 0.150000043946690906 | above | DERIVED |
| 235,974 optimistic model | 0.157125400604051262 | 10,702 B above integer ceiling | DERIVED |
| 262,483 exact-Seg + actual-ξ section | 0.174776655592366902 | 37,211 B above integer ceiling | DERIVED |

The prompt's `~225,273 B` is a rounded crossing, not the largest admissible integer. An above-threshold
description in either tested family only falsifies that family at that representation/custody scope.
It does not lower-bound the optimum and does not authorize the claim that reverse-waterfilling **must**
trade distortion. Conversely, a below-threshold upper bound would constructively prove a code-family
opportunity, not that an ideal lower bound “clears” anything.

## 6. Round-1 adversarial self-review

| audit question | verdict |
|---|---|
| Is a measured MDL a real lower bound on universal K? | **NO — FALSIFIED_AT_CLAIM_LEVEL [DERIVED]**; inequality points from K to code length. |
| Is the measured target a classical MS complex? | **NO-VERDICT [DERIVED]**; it is a digital argmax-cell complex without a potential/Hessian/transversality certificate. |
| Does the necessity `/2` quantity denote emitted bytes? | **NO [MEASURED/CODE-AUDITED]**; it is arithmetic after nonlinear compression and lacks a decoder/framing proof. |
| Is the class order correct? | **YES AS INPUT CONTRACT**; `[Road,Lane,Undrivable,Movable,MyCar]` is prescribed, while numeric IDs `0..4`, semantic-label hash, and exact counts are **MEASURED**. The hash does not independently infer class names. |
| Is ξ hand-waved? | **NO for its 7,195-byte section [MEASURED]**, but it is quantized, has nonzero `d_pose`, and is not cached `gt_poses`. |
| Is evaluator target `T_E=(S,P)` conflated with carrier object `(S,ξ)`? | **NO AFTER ROUND-1 FIX [DERIVED/CODE-AUDITED]**; equations use `P`, while the byte model separately charges quantized temporal ξ. |
| Can the tool place its durable receipt under system `/tmp`? | **NO AFTER ROUND-1 FIX [MEASURED negative-path test]**; it refuses before cache hashing. |
| Is `(0,0,K)` receiver custody present? | **NO [MEASURED/DERIVED]**; both realized-distortion and receiver-closure gates fail. |
| Does 235,974 B exceeding 225,272 B prove K exceeds it? | **NO [DERIVED]**; one code family missing a ceiling is inconclusive about a shorter program. |

**Independent ROUND-1 closure re-audit: CLEAN [MEASURED code/artifact review].** The re-audit
regenerated the full receipt byte-identically, rechecked all arithmetic and hashes, passed the
`/tmp` negative path before cache access, and found no remaining blocker. MAIN review is still mandatory.

No family is killed: the negative verdict scope is the **universal-lower-bound interpretation of a
concrete description length** and the current unclosed code instances. A model-restricted lower bound
would be possible only after defining a total bounded grammar and proving exhaustive minimality inside
that grammar; it would still not be universal K.

## 7. Triality, stores consulted, and next custody gate

- **Equation leg — CANDIDATE/DEBT:** replace the false `MDL(MS) <= K` bracket with the theorem-safe
  `K_U(T|D_C) <= L_C(T)+O(1)` and retain `K_U(T_E|E) <= K_U(Y|E)+O(1)`, with
  `T_E=(S,P)`, as the uncomputable target lower relation. Temporal ξ is a carrier variable, not
  evaluator output. No canonical equation is registered here; **MAIN mathematical review is mandatory**.
- **DAG leg — LANDED:**
  `.omx/research/mdl_ms_complex_K_lower_bound_DAG_FEED_20260718.md` routes this premise falsification,
  the exact partition upper bound, and receiver-closure debt to v10/#536/reverse-waterfill consumers.
- **Continual-learning pointer — LANDED:**
  `.omx/research/codex_premise_falsification_mdl_ms_complex_k_lower_bound_20260718T063906Z_codex.md`
  prevents a later session from silently restoring the reversed bracket.
- **DSL leg — N/A with rationale:** cached read-only measurement, no trainer/launcher lever.
- **Six-hook wire-in:** `research_only=true`; no sensitivity/Pareto/bit-allocator/autopilot mutation is
  authorized by an advisory unclosed model. The DAG feed is the safe consumer hook; posterior adoption
  remains blocked on an exact receiver-closed artifact.

**STORES CONSULTED:** #180 Morse–Smale polygon/temporal feasibility and its revised optimal-form FEED;
#284/#311 Laguerre/tropical carrier measurements; #369 separatrix/residual-sidecar lineage; necessity
solver + exact d_seg calibration; frozen-scorer factorization and intrinsic-coordinate design memory;
the latest collateral pullback correction; current frontier, lane, task, and inbox surfaces. Previously
settled numbers were reused and audited, not remeasured.

The next honest quantitative gate is an emitted, parse-back, self-delimiting cells/edge/junction
trajectory codec plus a receiver that reproduces the exact target statistic, with full Seg/Pose/rate
custody. Until then:

`REQUESTED_MDL_LOWER_RUNG = NOT_ESTABLISHED`

`MODEL_DESCRIPTION_LENGTH = 235974 B [DERIVED, NON-AUTHORIZING]`

`STRICT_RATE_CEILING = 225272 B [DERIVED]`

`UNIVERSAL_K_THRESHOLD_VERDICT = INCONCLUSIVE`

`POINTER = UNMOVED`

## MAIN landing review required

MAIN must review the branch diff and explicitly adjudicate:

1. the inequality correction and removal/non-reuse of `MDL <= K` as a universal claim;
2. the digital-cell-complex versus classical Morse–Smale scope;
3. the `/2` contour estimate and ξ/`gt_poses` custody distinctions;
4. the strict 225,272-byte integer ceiling;
5. whether the corrected equation candidate should be registered after the false premise is retired.

No score, launch, pointer move, or adoption authority follows from this memo.
