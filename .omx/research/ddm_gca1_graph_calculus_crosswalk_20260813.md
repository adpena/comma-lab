# DDM GCA1 — graph-calculus crosswalk

Date: 2026-08-13
Arm: `ddm_gca1_graph_calculus_crosswalk`
Authority: scorer-free literature derivation plus reductions of already-retained evidence
Source: Friedman and Tillich, [*Calculus on Graphs*](https://arxiv.org/abs/cs/0408028), arXiv:cs/0408028v1

## VERDICT FIRST

The prior is **partly confirmed**. The paper supplies a clean, usable calculus for Pact's
class-interface graph: oriented incidence, weak divergence, a positive graph Laplacian, and
coarea/cut identities. It also supplies the ingredients for a graph-energy proposal class on the
actual uint8 camera lattice. Those are two `ADOPT-CLASS` rows. Neither is an immediate score mover,
and neither authorizes a new explicit mask or payload.

The paper does **not** contain the three results that the charter most wanted to transfer directly:
its wave-equation, graph-distance/eigenvalue, and p-Laplacian results are only citations to separate
then-unpublished papers. Consequently:

- there is no paper-derived finite edit-propagation radius for QS2;
- there is no paper-derived equivalence between a p-Laplacian, an eikonal equation, and Pact's
  integer receiver lattice;
- diameter or a spectral gap cannot price a token's scorer blast radius.

No scorer was run, no payload was materialized, no archive was changed, and no pointer moved. The
live effective pointer remains **S = 0.16195513827824176 @ 186,252 B `[contest-CUDA T4, n600]`**.
The live own-vehicle frontier remains **S = 0.16959899569230852 @ 187,226 B
`[contest-CUDA T4, n600]`**.

## SOURCE CUSTODY AND READING BOUNDARY

The full 63-page arXiv HTML rendering was read, including definitions, propositions, proofs,
examples, and references. A direct binary-PDF download was attempted but the sandbox could not
resolve `arxiv.org`; therefore this arm has no local PDF bytes or PDF hash and makes no such custody
claim. The source boundary matters because the introduction explicitly routes wave equations,
p-Laplacians, and Laplacian-eigenvalue/distance results to the separate references `[FTa]`, `[FTb]`,
and `[FTc]` rather than proving them in this paper.

What this paper actually establishes and this memo uses is:

1. a metric-graph realization with vertex measure, edge measure, edge lengths, scalar functions,
   and vector fields;
2. weak divergence from integration by parts and the positive Laplacian
   `Delta = -div grad`;
3. reduction of that Laplacian to the usual weighted finite-graph operator for edgewise-linear
   functions;
4. coarea and Federer-Fleming identities connecting graph total variation to weighted cuts;
5. Cheeger/Dodziuk-style spectral inequalities;
6. the heat semigroup, positivity, contraction, heat-kernel bounds, and Sobolev/Nash consequences.

## RANKED CROSSWALK

| rank | paper object | Pact surface | disposition | conclusion | named consumer | follow-on disposition |
|---:|---|---|---|---|---|---|
| 1 | Oriented gradient, weak divergence, and positive Laplacian | Full-population directed class-confusion graph and per-edge residual decomposition | **ADOPT-CLASS** | Use one oriented class-interface graph as the accounting object. Directed edge flow explains node area bias exactly; undirected edge mass explains interface burden. This is a diagnostic and conditioning coordinate, not a replacement for the retained pixel field. | JS1 promoted-axis Stage 0; distortion-side learned implicit edge conditioning; RAG seg-core #52 | **QUEUED-WITH-A-FIRE-ORDER** as GCA1-P1 below. |
| 2 | Graph p-energy derived from the paper's gradient/divergence primitives | `rw2` / DK1 / finite-difference true-domain integer proposal family, content-mapped to the charter's “#974” | **ADOPT-CLASS** | Race integer-constrained edge energies as proposal generators on the camera lattice. Continuous p-Laplacian descent may rank proposals but never supplies acceptance authority; every accepted state remains exact uint8, receiver-realized, and scorer-gated. | `p0_true_domain_optimization_triple` content lineage; `rw2` DK1/FD successor | **QUEUED-WITH-A-FIRE-ORDER** as GCA1-P2 below. |
| 3 | Coarea and Federer-Fleming cut/TV identity | Connected interface support, g4 stationary edge events, and the successor to #941 grammar-v2 | **LESSON-ONLY** | Weighted boundary cost is a principled feature for comparing already-generated supports. It does not rescue explicit sparse Road/Lane token events: GV2 measured 0 positive, 1 neutral, and 253 harmful events among 254 target-reachable events `[macOS-CPU advisory, seeded stratified n32]`. | GCA1-P1 feature table; learned implicit edge-conditioning consumer, not a new explicit grammar | **FOLDED** into GCA1-P1; no duplicate grammar lane. |
| 4 | Heat semigroup and strictly positive heat kernel on a connected finite graph | QS2 edit propagation and the inherited 16.9312% realization efficiency | **LESSON-ONLY** | Heat flow is a possible attenuation null model, but it has no hard finite propagation radius: on a connected graph its kernel is positive everywhere for every `t > 0`. Pact's receiver-plus-SegNet map is nonlinear and thresholded, so the inherited efficiency is an empirical ratio, not a diffusion constant. | Existing QS2 per-pair postmortem / realization-lever harvest | **FOLDED** into the already-queued QS2 harvest as GCA1-P3; no new scorer or download lane. |
| 5 | Cheeger, diameter, and eigenvalue bounds | Token-versus-pixel blast-radius pricing; charter reference #896; #869 waterfill | **N-A** for pricing | These are global geometry or mixing bounds. They do not determine a local token's changed camera bytes, receiver impulse field, hard-margin crossings, or archive bytes. The task bridge describes #896 as “camera-grid rate lever closed (structurally dominated),” so GCA1 does not resurrect it. | #869 only through its existing measured per-token price and `S_R x margin` inputs | **FOLDED**; use measured impulse/reachability fields, not a spectral proxy. |
| 6 | Heat-kernel/Sobolev/Nash compression of global behavior | g4 spatial stationarity and context coding | **LESSON-ONLY** | Smoothness can motivate a context feature, but g4 already has stronger direct evidence: the same `(pixel, class-edge)` event accounts for 98.8063% of its 4,011,236-event field, while a generic boundary-distance context worsened the real stream from 490,794 B to 683,211 B `[macOS-CPU frozen-scorer advisory, n600]`. | g4/v13 context work | **FOLDED**; no boundary-distance retry on that vehicle without a new receiver-derived discriminator. |
| 7 | Wave equation and finite-speed propagation | Hard spatial bound for QS2 edits | **N-A** from this source | The paper does not present the wave equation; it cites `[FTc]`. Importing a finite-speed theorem without the operator, edge weights, and nonlinear threshold map would be a fake derivation. | QS2/hr1/rvs1 survival playbook | **FOLDED** until the separate source and an operator-faithful mapping are both in custody. |
| 8 | p-Laplacian and distance/eikonal relationship | “Native form” claim for #974 | **N-A** as an equivalence claim | The paper cites `[FTb]` for p-Laplacians and does not prove a p-Laplacian/eikonal equivalence. Even a valid real-valued p-energy remains a relaxation of Pact's uint8 feasible set. Row 2 adopts only the constrained proposal class, not the equivalence. | `rw2` successor | **FOLDED** into GCA1-P2 with the claim narrowed. |

There is no `ADOPT` row. Every surviving result is a class of coordinates or proposals whose Pact
value remains unmeasured.

## THE ADOPTED CLASS-INTERFACE CALCULUS

Let the five semantic classes be graph vertices. Give every observed unordered interface one
orientation, and let `B` be the vertex-by-edge incidence matrix. Let `M_V` contain node measures and
`M_E` contain nonnegative interface weights. For a node signal `f` and oriented edge field `X`, use

```text
grad f = B^T f
div X  = -M_V^-1 B M_E X
Delta f = -div grad f = M_V^-1 B M_E B^T f.
```

This convention gives the exact integration-by-parts identity

```text
<grad f, X>_E = -<f, div X>_V
```

and conservation `1^T M_V Delta f = 0`.

For Pact, distinguish two edge objects that must not be collapsed:

- **mass:** `A_ab = N(a->b) + N(b->a)`, the undirected burden on interface `{a,b}`;
- **flow:** `X_ab = N(a->b) - N(b->a)`, the signed directional imbalance.

The incidence divergence of the directed confusion flow equals each class's rendered-area bias.
That gives Stage 0 a cheap exact positive control: reconstruct the complete directed confusion
matrix, check that the five biases sum to zero, and preserve both `A` and `X`. Class marginals alone
lose the edge identity; undirected totals alone lose the over-paint/under-paint direction.

As a bounded check of the algebra, reducing the already-retained JS1 local-axis matrices gives
node-bias vectors `[1169, -1671, 171, 732, -401]` for CP135 and
`[1491, -1693, -710, 1205, -293]` for T1R1, each summing exactly to zero
`[macOS-CPU frozen-SegNet advisory, n600, 117,964,800 scorer pixels; DERIVED from retained confusion matrices]`.
This is not the missing promoted CUDA map and is not a score result.

### Why this changes the consumer

The old per-class view can charge the same separatrix twice. The graph view makes a single
Road-Lane interface the object while retaining both directions. It therefore supports:

1. exact conservation checks on any promoted Stage-0 decomposition;
2. per-edge conditioning derived from decoder state;
3. separate features for interface mass, directional imbalance, temporal persistence, and
   receiver reachability;
4. a no-double-counting allocation input to #869 after real byte prices exist.

It does **not** authorize a serialized adjacency mask. The live successor is distortion-side
learned implicit conditioning. SR1 closed rate-side implicit calibration at formulation scope, and
GV2 closed the unchanged-wire sparse token-event grammar at formulation scope.

## COAREA, CUTS, AND WHAT THEY DO NOT BUY

On a weighted spatial graph, define

```text
TV_G(f) = sum_{(u,v) in E} w_uv |f_u - f_v|.
```

For an indicator `f = 1_A`, `TV_G(f)` is exactly the weighted cut of `A`. The coarea identity says
the TV of a scalar field is the integral of the cut sizes of its superlevel sets. This gives a
principled way to report the boundary burden of a proposed connected support and to separate
“large interior, small cut” from fragmented high-cut support.

That identity is accounting, not an actuator. It predicts neither sign nor survival through
resize, uint8, SegNet receptive fields, and margin crossings. In particular, graph connectivity
cannot reverse GV2's measured harmful-event result. GCA1 therefore uses cut/TV only as a feature in
the retained-field P1 discriminator.

## TRUE-DOMAIN GRAPH ENERGY

The narrow live formulation is a constrained energy on the **camera-sample integer lattice**. For
an integer edit vector `z`, an oriented spatial incidence matrix `B_R`, and nonnegative weights
`W_R` derived from current receiver reachability and hard-margin evidence, define

```text
E_p(z) = (1/p) sum_e W_R[e] |(B_R^T z)[e]|^p,          p >= 1.
```

For `p > 1`, the real relaxation has the graph-gradient direction

```text
L_p(z) = B_R W_R (|B_R^T z|^(p-2) * B_R^T z).
```

This is a proposal-ranking device only. The feasible set remains integer uint8 values; sub-quantum
moves remain null; exact receiver bytes determine whether a proposal exists; and the normal
complete-score gate determines acceptance. The graph must be built in camera coordinates and its
weights must come from current `S_R x margin` evidence. A generic 4-neighbor grid with arbitrary
weights is not “receiver native.”

### Toy verification on a three-node path

Orient the unit path `0 -> 1 -> 2` and take

```text
B = [[-1,  0],
     [ 1, -1],
     [ 0,  1]].
```

Then

```text
L = B B^T = [[ 1, -1,  0],
             [-1,  2, -1],
             [ 0, -1,  1]].
```

For `f = [1,0,0]^T`, `B^T f = [-1,0]^T`, `L f = [1,-1,0]^T`, and the cut/TV of
`{0}` is one. For every `p >= 1`, the integer move `f -> [0,0,0]^T` lowers
`E_p` from `1/p` to zero. This verifies the incidence, conservation, cut, and integer-energy
statements. It does not model the contest scorer and therefore earns no empirical credit.

## WHY GRAPH HEAT DOES NOT EXPLAIN QS2'S EFFICIENCY

QS2's retained whole-candidate denominator is 189 changed hard pixels and 32 net flips, hence
`32/189 = 16.9312%` realization efficiency `[prior contest-CUDA T4 component evidence carried by
QS2; no new scorer measurement here]`. This ratio cannot be replaced by a paper constant.

A local linearization would have the form

```text
delta_margin[v] approximately sum_u K[v,u] delta_camera[u].
```

A hard class flip occurs only when this signed response crosses the pre-edit margin. Efficiency
therefore depends jointly on the directional kernel `K`, the margin distribution, edit signs,
receiver quantization, and off-target crossings.

The graph heat kernel `exp(-t Delta)` can be fitted as a null model for attenuation, but it cannot
supply a hard radius. The paper proves strict positivity of the heat kernel on a connected finite
graph for every positive time. On the three-node path, an impulse at node 0 reaches node 1 at first
order and node 2 at second order; it is not compactly supported for positive time. The absent wave
paper might provide finite speed for a different operator, but that result cannot be transferred to
the nonlinear receiver/scorer cascade without an operator-faithful proof.

GCA1-P3 therefore asks only whether graph distance or a fitted heat kernel predicts the **already
retained** per-pair response better than a margin-only baseline. It neither reruns SegNet nor claims
a propagation theorem.

## BLAST RADIUS AND WATERFILL

The quantity #869 needs is local and economic:

```text
token utility = complete realized score change / exact additional archive bytes.
```

A token's blast radius must be measured from its exact decoded camera-byte changes, the through-R
reachability field, and resulting hard-margin crossings. Graph diameter, a Cheeger constant, or a
spectral gap is a whole-graph quantity. None determines that token-specific numerator or the coder's
byte denominator. The usable routing is therefore:

```text
retained token impulse -> exact camera support -> S_R x margin -> per-edge hard response
                       -> exact payload bytes -> #869 waterfill.
```

This folds GCA1 into the existing RVS1 reachability-plus-margin treatment and #869. It does not
reopen #896, whose task-bridge subject already records the camera-grid rate lever as structurally
dominated.

## $0 PROBE SPECIFICATIONS AND FIRE ORDER

### GCA1-P1 — promoted per-edge incidence/context discriminator

- **Question:** does preserving oriented edge identity add predictive/coding value beyond class
  marginals and undirected edge mass on the current promoted field?
- **Inputs:** the already-queued JS1 promoted CUDA per-pair directed decomposition, exact input
  hashes, and decoder-derived state. Fail closed if the promoted field is absent.
- **Controls:** same event ordering, same train/holdout split, same selected events, same coder, and
  no explicit stored mask. Compare class-only, undirected-edge-only, and oriented-edge-plus-divergence
  contexts.
- **Positive controls:** exact 5x5 matrix reconstruction; all 20 directed cells; all 10 undirected
  interfaces; node divergence equals class area bias; total divergence is zero.
- **Decision:** keep the class only if held-out log loss or retained exact serialized bytes improve
  without changing decoded events. A feature-only win does not imply score improvement.
- **Retention:** keep every candidate payload, decoder repeat, hashes, and selected-input census in
  the existing promoted Stage-0 consumer store.
- **Fire order:** after JS1 has a promotion-admitted CUDA map, before any distortion-side learned
  edge-conditioning build.

### GCA1-P2 — integer graph-energy proposal discriminator

- **Question:** can a camera-lattice graph energy produce distinct receiver-nonnull integer
  proposals that the existing DK1/FD proposal order misses?
- **Inputs:** the retained `rw2` n1/n32 proposal surfaces and current receiver-reachability weights;
  no regenerated scorer fields.
- **Race:** bounded `p in {1,2,4}` against the existing DK1/FD order, with identical integer support
  and proposal budget. Continuous directions are snapped to measured whole-lattice quanta before
  any comparison.
- **Stage-A gate:** retain every proposal and require exact uint8-byte difference, deterministic
  receiver parse-back, and a distinct admissible ordering. If none survives, close only this
  weighted-energy instance.
- **Stage-B gate:** if Stage A survives, the true-domain owner may place the retained candidates in
  its normal same-axis complete-score queue. No local proxy selects a winner.
- **Retention:** preserve every materialized proposal/candidate payload with byte count and SHA-256;
  a scalar-only race is forbidden.
- **Fire order:** after the current `rw2` consumer accepts a current-vehicle `S_R x margin` map and
  before it spends a scorer call on another continuous-relaxation variant.

### GCA1-P3 — QS2 propagation null-model fit

- **Question:** do graph distance or a fitted heat kernel explain per-pair realized response beyond
  the pre-edit margin alone?
- **Inputs:** the exact QS2 candidate/base argmax fields and per-pair edit coordinates already
  queued for harvest; no re-run and no substitute axis.
- **Models:** margin-only baseline; distance plus margin; fitted heat-kernel feature plus margin.
  Split by pair, not by pixel, and report the complete denominator and selection mode.
- **Decision:** retain graph propagation only if it improves held-out pair-level prediction and its
  calibration survives the exact-step versus dead-zone split. It remains a prioritizer, not an
  acceptance oracle.
- **Fire order:** inside the existing QS2 `QUEUED-AT-HARVEST` postmortem, after the expected argmax
  SHA is locally verified. This is not a separate download, scorer, or Modal action.

## RECALL EVIDENCE

### Stores and queries

The recall covered the full required corpus rather than only the charter seeds:

- `.omx/research/` memos and receipts with content queries for `graph calculus`, `graph Laplacian`,
  `p-Laplacian`, `heat`, `Cheeger`, `coarea`, `edit propagation`, `blast radius`, `per-edge`,
  `Road Lane`, `true-domain`, `lattice`, `stationarity`, `reachability`, and `margin`;
- `.venv/bin/python tools/list_canonical_equations.py --json`, especially
  `receiver_lattice_leakage_exponent_v1`, `receiver_pose_semantic_preservation_ratio_v1`,
  `pose_stack_exact_budget_v1`, and `v8_geometric_rate_decomposition_v1`;
- `CANONICAL_RESEARCH_INDEX*`, the specialized canonical indexes, and `sub015_DAG_*` FEED blocks
  with queries for graph, edge, lattice, realization, blast, stationarity, p-Laplacian, and eikonal;
- `harness_tasklist_bridge_20260803.jsonl` and related task-status searches for #869, #896, #941,
  and #974;
- the live board, GDL1/TR2/NG1 crosswalks, QS2, JS1, RVS1, RW1/RW2, g4, PC2/m91, SX1, SR1, GV2,
  and the true-domain content mapping in PS135.

### Findings beyond the charter seeds and what changed

1. **JS1 already retains a complete local-axis directed 5x5 decomposition**, not just per-edge
   totals. This changed P1 from “invent a graph representation” to “apply exact incidence checks and
   race zero-payload contexts after promoted custody exists.” The local field is explicitly
   non-promotable because it misses the CUDA reference by 15,431 flips.
2. **SR1 already closed rate-side implicit edge conditioning at formulation scope**, while
   distortion-side decoder-derived conditioning remains open. This forbids routing GCA1 into a new
   probability-calibration arm.
3. **GV2 already closed the unchanged-wire sparse Road/Lane token-event formulation** on its stated
   vehicle/sampling scope. Coarea/connectivity therefore folds into the learned implicit consumer;
   it does not reopen a GV3 grammar.
4. **RVS1 already names `S_R x margin` and dynamic whole-quantum selection.** This changed the
   lattice graph weights from generic geometric weights to current receiver-derived weights and
   folded spectral blast-radius work into the existing reachability consumer.
5. **#896's bridge subject says the camera-grid rate lever is structurally dominated.** GCA1 does
   not create a successor under that ID.
6. **#974 was not found as a live canonical task or bridge row.** PS135 maps it by content to the
   true receiver-realized lattice doctrine. GCA1-P2 therefore names `rw2`/DK1/FD and the content
   lineage, not a phantom task owner.
7. **g4 already falsified generic boundary-distance context as a real coder win on its vehicle.**
   The graph paper supplies no evidence that overrides that measured result.

No current canonical equation supersedes these findings. The formulas in this memo are
literature-derived coordinates and exact algebraic identities, not a new calibrated empirical
law.

`# FORMALIZATION_PENDING: GCA1 introduces no new empirical model to register; incidence conservation and graph p-energy are literature-derived, while the only numeric reduction re-expresses an existing retained JS1 confusion matrix without a new measurement.`

## AUTHORITY, DENOMINATORS, AND BOUNDARIES

- **MEASURED elsewhere and reused with explicit scope:** JS1 local full-population directed
  decomposition `[macOS-CPU frozen-SegNet advisory, n600, 117,964,800 pixels]`; GV2 token-event
  result `[macOS-CPU advisory, seeded stratified n32, 254 target-reachable events]`; g4 stationarity
  and real-coder rows `[macOS-CPU frozen-scorer advisory, n600]`; QS2 inherited component/efficiency
  evidence as labeled in its receipt.
- **DERIVED here:** incidence/divergence identities; the two zero-sum node-bias reductions from the
  retained JS1 matrices; the three-node toy; the distinction between heat attenuation and hard
  finite propagation.
- **NOT MEASURED here:** any new SegNet/PoseNet output, any new decoded video, any payload bytes,
  any held-out context gain, any p-energy proposal gain, any complete score, any contest-CPU/CUDA
  result, or any frontier improvement.
- **Negative scopes:** the paper-source absences are source-scoped; GV2 and SR1 closures retain
  their original formulation scopes; g4's boundary-distance failure is vehicle/formulation scoped;
  no graph-calculus family is killed.
- **Mechanism-reduction gate:** not applicable. This arm neither materialized a candidate nor
  claimed a score-moving mechanism.
- **Goal status:** sub-0.15 remains unsatisfied. This research unit improved routing and formal
  accounting only; it did not move the exact pointer.

## NEXT_IF_RESUMED

- **BLOCKED-PENDING-GIT-WRITE — memo landing.** Owner: MAIN/operator in a Git-writable checkout. Consumer store: `.omx/research/ddm_gca1_graph_calculus_crosswalk_20260813.md`. Fire trigger: the repository index and object database permit writes; recompute the post-edit SHA-256 and rerun `tools/subagent_commit_serializer.py` for this file only. The required serializer was attempted here and failed before staging with `unable to create temporary file: Operation not permitted` / `failed to insert into database`.
- **QUEUED-WITH-A-FIRE-ORDER — GCA1-P1.** Owner: MAIN/js1 promoted-axis owner. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/`. Fire trigger: JS1 lands a promotion-admitted n600 CUDA directed decomposition with retained per-pair fields; then run the scorer-free class-only versus undirected-edge versus oriented-edge context race before building distortion-side implicit conditioning.
- **QUEUED-WITH-A-FIRE-ORDER — GCA1-P2.** Owner: `rw2` true-domain successor owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_rw2_graph_energy/retained/`, with the receipt consumed by `.omx/research/ddm_rw2_20260806/`. Fire trigger: a current-vehicle receiver-reachability-plus-margin map is accepted and the owner is about to spend another scorer call on a continuous-relaxation proposal; first run and retain the bounded integer `p in {1,2,4}` Stage-A candidates.
- **FOLDED / QUEUED-AT-HARVEST — GCA1-P3.** Owner: MAIN/QS2 postmortem owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/`. Fire trigger: the expected retained candidate argmax field is available locally with SHA-256 `ad1e3dcc0a57c53f0757773a018335924afc26992f398c23ec084eecace7ed20`; add the pair-split margin/distance/heat-kernel fit inside the existing harvest without rerunning SegNet.

## LIVE-HYPOTHESES

- Oriented class-edge flow will add useful conditioning beyond class marginals because interface
  mass and net class-area bias are different conserved objects, and the retained matrices already
  show large directional imbalances.
- A receiver-weighted `p=1` or `p=2` integer graph-energy order may find admissible whole-quantum
  proposals missed by the present DK1/FD order because it couples neighboring camera samples while
  respecting the exact lattice; this remains plausible only with current `S_R x margin` weights.
- Graph distance may improve QS2 response prediction after margin is included because resize and
  local receptive fields attenuate nearby perturbations, even though heat flow cannot provide a
  hard cutoff and the nonlinear threshold map may defeat the fit.
- Coarea/cut features may help the learned implicit edge consumer distinguish compact interface
  corrections from fragmented high-collateral supports without storing an explicit support mask.

## DEAD-ENDS

- A finite QS2 edit-propagation radius derived from *Calculus on Graphs* is closed at SOURCE scope:
  the wave equation is not in this paper, and its heat kernel has global positive support for every
  positive time on a connected graph.
- “p-Laplacian equals eikonal/viscosity on the uint8 receiver lattice” is closed as an EQUIVALENCE
  claim: the cited paper does not prove it, and a real-valued energy is still only a relaxation of
  the integer feasible set.
- Diameter, Cheeger, or spectral-gap pricing of token blast radius is closed as a FORMULATION:
  global graph bounds do not supply token-specific camera changes, hard-margin crossings, or exact
  archive bytes.
- Reopening #896 is closed by the live task-bridge disposition; its camera-grid rate lever is
  recorded as structurally dominated.
- A new explicit sparse Road/Lane grammar is closed on GV2's stated formulation scope: 253 of 254
  target-reachable token events were harmful and none was positive. Graph connectivity does not
  reverse that evidence.
- Generic boundary-distance context is closed on g4's measured vehicle/formulation scope because
  its real coded stream was worse despite an attractive ideal entropy estimate.
- Treating QS2's 16.9312% realization efficiency as a universal graph constant is closed: it is a
  candidate-specific ratio governed by receiver response, margins, signs, and off-target flips.
