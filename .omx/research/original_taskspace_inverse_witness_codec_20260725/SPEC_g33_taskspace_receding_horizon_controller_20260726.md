# G33 — Exact Whole-Object Receding-Horizon Controller

Date: 2026-07-26  
Lane: `lane_original_taskspace_inverse_witness_codec_capstone_20260726`  
Status: finite-horizon manifest/certificate controller implemented and locally verified; production actuation blocked on real public endpoints and authoritative evaluation  
Research scope: original Pact selected-solution codec; no public archive, weights, latents, selectors, or payload reused

## Purpose

The codec is a coupled controlled system, not a collection of independently acceptable Seg, Pose, and rate values. A local edit can change evaluator cells, archive entropy, future factorability, receiver behavior, and the reachable set of later repairs. Costates and local derivatives therefore propose actions; they do not choose actions.

G33 is the terminal decision surface. It compares proof-carrying finite-horizon terminal branches on one exact base and one dynamic pointer snapshot. `BASE_STOP` is always an exact incumbent. G33 commits at most the **first action** of one branch, and only when that branch's feasible terminal upper bound is strictly below both `BASE_STOP` and every competing branch's verified terminal lower bound. A commit invalidates local derivatives, but preserves nonselected exact branch evidence as historical signal before regenerating the universe from the new exact base.

Implementation:

- `src/tac/witness_control/taskspace_receding_horizon_controller_v1.py`
- `src/tac/witness_control/tests/test_taskspace_receding_horizon_controller_v1.py`
- G19 handoff: `src/tac/witness_control/taskspace_interaction_costate_bridge_v1.py`

## Governing state and action

For exact whole-object state `x_k`, define

`S(x_k) = 100 d_seg(x_k) + sqrt(10 d_pose(x_k)) + 25 B(x_k) / 37_545_489`.

`B` is the final complete `archive.zip` byte count, never raw tensor size, section size, pre-compression size, or a proxy. Decode runtime, memory, determinism, public video closure, recursive evaluator closure, absence of hidden data/network/source/scorer access, and full counting of video-specific state are hard constraints, not extra score terms.

The current action universe is

`U_k = {u_i : rebuild(x_k, u_i) -> x_{k+1,i}}`

over the closed families

`PRUNE, MERGE, MACRO_EVICT, MIGRATE, INVERSE_REPAIR, FIT_QUOTIENT, TRAIN_QUOTIENT, REQUANTIZE_STORAGE, JOINT_DESCENT`

and scales

`MICRO, MESO, MACRO, SYSTEM`.

The load-bearing generator manifest declares the exact full 9-family by 4-scale vocabulary, deterministic seed, parameter and chronology domains, horizon, maximum interaction order, generator/enumeration identities, and the exact content-addressed set of terminal leaves. Every leaf carries an ordered action path, its first action, parents, chronology context, and a canonical support hyperedge. Joint `PRUNE × INVERSE_REPAIR` or higher-order actions are therefore first-class leaves, not annotations on a scalar family cell.

Each materialized endpoint carries its exact generated-leaf descriptor, base identity, epoch, explicit authority mode and hardware axis, archive/output/runtime/placement/measurement/public-auth custody, realized `d_seg`, realized `d_pose`, final archive bytes, and public decode constraints. Evaluator-cell and continuation-equivalence identities each require exact proof dependencies. Every endpoint must match the base authority mode and axis; macOS advisory evidence cannot enter a contest-CPU or contest-CUDA universe by relabeling. Caller order is never a prior.

The base itself carries the same complete public-workflow constraint evidence and must pass it before it may serve as the incumbent for scoring or branch-and-bound. An infeasible or decode-only base cannot provide a low incumbent that falsely proves unexplored branches dominated.

The score implementation is the canonical `tac.contest_score` implementation, not a second hand-written formula. For every exact transition G33 also emits the joint finite component deltas, the local differential diagnostic, the nonlinear square-root remainder, and the complete conditional score-sublevel coordinates. These are coupled geometry diagnostics, not independent admission gates: there is no fixed per-endpoint Seg, Pose, or rate threshold.

Completeness does not require naively materializing every leaf. The exact manifest leaf set must equal a disjoint union of measured whole-object endpoints and subtrees carried by `ContinuationCertificateEvidenceBundleV1`. A bare `VerifiedContinuationCertificateV1` object has no authority at a G33 boundary. Each bundle carries that cached result together with the exact proof bytes, independently typed verification context, exact canonical generator-manifest payload bytes, and every dependency payload. G33 calls `reverify_continuation_certificate(...)` both when closing the universe partition and again before consuming any lower or upper value. Finite terminal evidence can therefore derive exact lower and feasible upper values. Primitive intervals remain structurally representable in G38 but deliberately fail closed until a typed interval-theorem verifier exists; production continuation admission likewise fails closed until its governed adapter exists. A random scalar plus a syntactically valid SHA, an in-process registry assertion, or a certificate object without durable bytes is rejected.

Family-by-scale coverage is a complete 36-cell **audit projection** of the manifest support hypergraph. It is never a universe-completeness proof. The load-bearing equality is:

`Leaves(M_k) = MeasuredLeaves ⊎ ProofByteReverifiedCertificateSubtrees`.

For horizon `H`, continuation class `q`, and first action `a`, define the reachable terminal branch value interval `[L_H(a,q), U_H(a,q)]`. Aggregate by first action:

`L_H(a) = min_q L_H(a,q)` and `U_H(a) = min_q U_H(a,q)`.

The decision contract is:

`commit(a*) iff U_H(a*) < S(BASE_STOP) and U_H(a*) < min_{b != a*} L_H(b)`.

Missing lower evidence blocks closure. Missing feasible upper evidence cannot win. If a lower-only branch remains below the base, G33 emits `BLOCKED_FEASIBLE_TERMINAL_UPPER_UNAVAILABLE`, never a fixed point. Equal terminal values across different continuations block unless the exact successor representation and proof dependencies establish equivalence; lexical action ID is not a value function.

After a commit:

`x_{k+1} <- rebuild(x_k, u*_k)`

and every old local marginal is stale. Reusing those derivatives is forbidden because the edit changes the coupled field and its continuation geometry. Nonselected exact terminal and certificate receipts are retained rather than discarded; only their old local derivatives lose decision relevance.

G33 recommends one search-state transition; it never authorizes archive promotion or pointer movement. Even a selected contest-axis endpoint remains owed same-archive contest-CPU and contest-CUDA replay before promotion custody can be considered.

## Identity-domain correction

The real G14 receipt falsified the assumption that one `measurement_id` must map to one proof-receipt hash. Ten semantic measurements have two proof receipts; one semantic state occurs 31 times. The only difference among those repeated realizations is proof custody.

The corrected domains are:

1. **Artifact identity** — the full occurrence, including its exact proof receipt.
2. **Semantic-control identity** — the realized measurement with proof-receipt identity projected out.
3. **Proof-dependency set** — every exact proof receipt supporting that semantic-control identity.

True semantic aliasing under one measurement ID still fails closed. Multiple proofs no longer multiply the controller state or destroy corroborating evidence.

## Functional quotient and the under-the-nose eureka

Present evaluator equality is necessary but insufficient for a safe quotient. Two representations can produce the same current evaluator cells and score while exposing different future factorization, repair, pruning, migration, or training action spaces.

G33 therefore quotients only by

`Q(x) = (evaluator_cell_identity(x), continuation_equivalence_identity(x), proof_dependency_identity(x))`.

This is a bounded control-bisimulation condition. Among states in the same `Q` class, the cheapest exact endpoint is the representative. States with equal current evaluator cells but different continuation identities or proof dependencies are preserved. More importantly, those preserved continuations are now assigned terminal lower/upper values and participate in strict first-action dominance. Identity preservation alone was insufficient: the prior implementation still selected by present score and lexical action ID, which G37 falsified with a dead-end-versus-rich-future construction.

Continuation identity must conservatively bind the future-relevant representation state: decoder ABI, factor graph, reservoir ownership, payload placement, reachable typed action generators, hard decode constraints, and any exact state needed to determine later endpoint availability. It is not permitted to contain scorer weights, source video, target cells, or hidden video-specific state in generic decoder code.

## Output-cell identity, sufficient statistics, and proof custody

Aggregate `d_seg`, aggregate `d_pose`, and final score are insufficient to identify an evaluator state. Distinct ordered pair populations can have equal aggregates while exposing different repair geometry and future actions. A production endpoint therefore needs an exact ordered evaluator-output-cell ledger, separately for contest-CPU and contest-CUDA:

- candidate SegNet argmax-cell identity and target/candidate mismatch count for every ordered pair;
- candidate official PoseNet first-six fp32-cell identity and exact pair MSE for every ordered pair;
- the corresponding target cells as authority evidence only, never as counted or uncounted decoder payload;
- exact binding to archive bytes, realized video bytes, scorer-input ledger, frozen evaluator/runtime ABI, hardware axis, and the unmodified official evaluation receipt.

Because the frozen evaluator has no output-capture hook, an instrumented scorer run cannot silently inherit official authority. The honest construction is an unmodified official run plus an observation-only mirror capture, byte/ABI-bound to that run, with a separately proved equivalence receipt. The semantic candidate-cell identity is distinct from the full output-ledger proof identity.

This yields five non-interchangeable domains that must remain separate in storage and APIs:

1. counted reconstruction/representation identity;
2. evaluator-cell semantic identity;
3. compact score sufficient statistics;
4. continuation/action-space identity;
5. proof-dependency identity.

This separation is the codec-level composition law: dense encoder-only evidence may certify a tiny counted sufficient statistic without being shipped, while future-action state is retained whenever it changes the reachable quotient even if present evaluator cells are equal.

## Real integrated receipt

Materialization:

`.omx/research/original_taskspace_inverse_witness_codec_20260725/taskspace_feedback_costate_materialization_n600_v5_20260726.json`

- File bytes: `4,043,545`
- File SHA-256: `9db91f681131c6cd1126a5dd8b2ee048b4d35b08967509ec07d240b016078338`
- G18 bytes / SHA-256: `3,874,482` / `dbbc59252e73b358a663972791c419f0ade40e4caf02c0aa3d346f385c0f83e0`
- G19 bytes / SHA-256: `167,111` / `b3feac0a86dc3db1c34fc129f662609fe87d1bed42a30eb4323817f6025b22b2`
- Semantic-control identities: `316`
- Multi-proof identities: `10`
- Maximum occurrence multiplicity: `31`
- Same-object G28 advisory score: `36.09269518733842`
- Dynamic target observed by that receipt: `0.172`
- Dominant debt and next phase: Pose; `MAXIMAL_INVERSE_POSE_PRESERVING_REPAIR`

This is research-only macOS/frozen-scorer advisory telemetry, not a contest score or candidate claim. It performs no scorer dispatch, pointer mutation, archive promotion, or authority upgrade.

## Exact composition sign reversals

The 240 measured G8-by-A four-cell interactions contain 12 exact sign reversals:

- 11 actions are harmful when evaluated locally on G0 but beneficial when composed with a G8 state;
- 1 action is beneficial on G0 but harmful when composed with G8;
- 140 interaction residuals are negative and 100 are positive.

The strongest reversal changes from `+0.005685046229473301` score units on G0 to `-0.03904014678044376` with its G8 context, an interaction residual of `-0.044725193009917064`. Therefore a controller that filters locally non-improving A actions before composing them would provably suppress useful signal already present in the measured archive population.

The strongest row uses a target-oracle diagnostic control and is not a legal deployable payload. It is evidence about the coupling law only. The same reversal class also appears in original class-bounded and class-shared medoid families. G33 consequently protects every sign-reversal hyperedge from local marginal pruning and requires paired whole-object endpoint enumeration before a verdict.

The four-cell residuals are measured second-order probes, not an assumption that the codec energy is pairwise. Three-way and higher interactions among topology, realization, temporal transport, factorization, entropy contexts, and decoder repair are absorbed automatically by rebuilding and scoring the complete endpoint. Pairwise costates guide acquisition; only a complete same-base endpoint or a proof-byte-reverified G38 continuation certificate decides.

The legacy G7 ordered allocator is consequently classified as a proposal generator only. Its caller order may be used as a search prior, but its greedy result has `global_optimality_claim=false` and `sign_reversal_safe=false`. It may never close a family, discard a locally harmful proposal, authorize a pointer move, or substitute for G33's exact manifest partition into same-base endpoints and proof-byte-reverified G38 certificate subtrees.

The older Consumer-15 per-pair ADMM/greedy treatment planner is now likewise machine-labeled `endpoint_proposal_generator_only=true`, `global_optimality_claim=false`, and `sign_reversal_safe=false`. Its recovery explicitly stops at locally non-improving marginals, and its historical plus-or-minus-five-percent interval is an uncalibrated symmetric heuristic rather than statistical coverage. The surface remains valuable for acquisition order and shadow-price telemetry, but only a whole-object rebuild can turn one of its proposals into G33 evidence.

## Decoder placement law

Generic deterministic entropy decode, factor expansion, synthesis, inverse solve, projection, iteration, repair, and postfilter machinery may live in `inflate.py` or `inflate.sh` at zero archive-rate cost, subject to runtime, memory, deterministic portability, contest legality, and public recursive-closure proof. The runtime constraint covers the complete official public workflow measured at the `evaluate.sh` boundary, not only decoder subprocess time.

All video-specific parameters, latents, selectors, atoms, exceptions, target-dependent branches, and learned residuals are counted payload. Decode may not access the source video, scorer, scorer weights, GT cells, teacher outputs, network, or uncounted video-specific state.

## Triality and six-hook wire-in

- DSL: typed `WholeObjectBaseV1`, `GeneratedTerminalLeafV1`, `GeneratorDomainManifestV1`, `WholeObjectActionEndpointV1`, `ContinuationCertificateEvidenceBundleV1` with exact G38 proof material, audit-only `ActionFamilyScaleCoverageV1`, and `CompleteActionUniverseV1`.
- DAG: exact base -> full-vocabulary finite generator manifest -> atomic and higher-order action paths -> measured terminals or reopened verifier certificates -> continuation values -> strict first-action dominance against `BASE_STOP` -> invalidate local derivatives but retain evidence -> regenerate.
- Equation: `commit(a*)` only when `U_H(a*) < S(BASE_STOP)` and `U_H(a*) < min_{b!=a*} L_H(b)`; independent Seg/Pose/rate thresholds and additive marginal ordering are forbidden.
- Sensitivity map: finite endpoint component deltas are proposal telemetry; no derivative authority is invented.
- Pareto surface: the nonlinear coupled score and hard decoder constraints are preserved exactly.
- Bit allocator: receives only complete rebuilt endpoints, never atomic marginal sums.
- Cathedral/autopilot: G19 exposes the exact G33 consumer and refuses partial-universe actuation.
- Continual learning: exact observations may update confidence only after same-base, same-population, public-output custody closes.
- Probe disambiguation: equal present cells with equal versus different continuation identity are both explicitly represented and regression-tested.

## Current blocker and next construction

G33 is callable, but the real chain does not yet provide a complete current-epoch public action manifest. G20/G22 close a private full-n600 same-state receiver replay; G25 improves conditional rate to 80,238 bytes but lacks an authority-bearing public endpoint; G28 supplies component routing but is advisory and Pose-catastrophic. G29 has now implemented the public inverse, custody schemas, exact candidate semantic-cell identity split, resumable preflight, and a real LVPG2-to-LVLS1 parse-back. Independent G31 review still classifies it `NO-AUTHORITY`: 0 closed, 9 partial, and 5 open authority gates. No governed Linux public evaluation, n600 mirror equality, full workflow timing, same-axis A/B, paired CPU/CUDA replay, or frontier score exists yet.

The next endpoint-generating sequence is:

1. use the G35-hardened R10 fitter only through its checkpointed/native-resolution/governed path; do not fire n600 until its state is bound to the real public action manifest;
2. complete the G23/G29-to-G33 adapter so exact public endpoints and certificate partitions share the same base, semantic cells, continuation state, and proof dependencies;
3. generate the smallest high-value matched inverse-repair, macro-eviction, merge/prune/factor, fit-only, and irreducible-training **terminal paths**, retaining locally harmful constituents inside joint hyperedges;
4. close two fresh bit-identical 1,200-frame decodes and the authoritative recursive evaluation graph for branches whose terminal value can beat `0.172`;
5. let G33 select one first action only after strict continuation dominance;
6. regenerate from the new exact base; reserve joint descent for the terminal whole-object link.

## Stores consulted

- `CLAUDE.md`
- `AGENTS.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- canonical frontier pointer and G14/G20/G22/G25/G28 receipts named by the materialization custody record
- latest original task-space selected-solution codec specifications and Codex findings in this research directory

## Pointer delta honesty

Pointer moved: **no**.  
Frontier score claim: **no**.  
Candidate archive produced: **no**.  
Concrete delta: identity-domain bug extincted; real receipt chain materialized; one-step-greedy false receding-horizon semantics removed; full-vocabulary hyperedge manifest equality, durable proof-byte-reverified G38 continuation certificates, exact `BASE_STOP`, and strict finite-horizon first-action dominance implemented and tested. No frontier candidate or score is claimed.
