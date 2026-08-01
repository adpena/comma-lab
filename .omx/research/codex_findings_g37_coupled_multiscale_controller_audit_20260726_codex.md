# G37 coupled multiscale controller audit — local-to-global signal suppression

Date: 2026-07-26  
Lane: `lane_g37_coupled_multiscale_controller_audit_20260726`  
Status: **complete adversarial review; NO-FIRE for G33 global/fixed-point semantics**  
Truth: `research_only=true`; `score_claim=false`; `candidate_claim=false`; `promotion_eligible=false`; `pointer_moved=false`; `dispatch_performed=false`  
Scope: read-only audit of root-owned G33 controller, G7 allocator, Consumer 15, focused tests, the real G14 sign-reversal receipt, and the G33 specification. No reviewed source or test was edited.

## Outcome first

The composition direction is correct, but the current controller is not yet the terminal global controller its output says it is. The deepest gap is not another local codec lever:

> G33 preserves a label for distinct future action spaces, but it optimizes only the present endpoint score. It is an exact one-step greedy selector with a continuation tag, not receding-horizon control.

Three P0 failures can suppress the useful joint signal already measured by G14:

1. **Continuation is preserved but not valued.** Equal-present/equal-score states with different future action spaces are tie-broken by `action_id`; the unselected richer continuation is then marked stale.
2. **Family-by-scale occupancy is called exhaustive enumeration.** The caller chooses a subset of required families/scales, and one supplied endpoint in each chosen cell makes `enumeration_complete=true`; neither missing actions nor missing higher-order hyperedges are detected.
3. **Branch-and-bound admissibility is caller-authored.** An arbitrary finite number plus an arbitrary syntactically valid SHA is accepted as an admissible lower-bound certificate and can close an unmeasured branch.

The exact nonlinear score arithmetic, exact-base/epoch check, explicit axis check, one-action commit discipline, post-commit marginal invalidation, and G7/Consumer-15 non-authority labels are useful and should be retained. They do not close the three P0s above.

## Authority cut and real coupling evidence

The live canonical pointer observed during the audit was the external official target `0.172`, artifact SHA-256 `2a61b052be496d3a9a1be1a9c230c8d179a788e61fd03472e50fc85832da94c6`, 17,319 bytes. This audit did not refresh or mutate it.

The real materialization

`.omx/research/original_taskspace_inverse_witness_codec_20260725/taskspace_feedback_costate_materialization_n600_v5_20260726.json`

is 4,043,545 bytes, SHA-256 `9db91f681131c6cd1126a5dd8b2ee048b4d35b08967509ec07d240b016078338`. Its `g19_controller.record.composition_sign_reversal_protection` contains 240 exact G8-by-A four-cell interactions and 12 sign reversals:

- 11 locally harmful actions become beneficial in a G8 context;
- 1 locally beneficial action becomes harmful;
- the strongest row moves from `+0.005685046229473301` alone to `-0.03904014678044376` in context;
- its interaction residual is `-0.044725193009917064`.

This is advisory n2 coupling evidence, not an n600 family verdict or contest score. It is nevertheless an exact falsifier of local marginal pruning within the measured population. The strongest row is an oracle diagnostic, while the reversal class also occurs in original class-bounded and class-shared medoid families, as recorded by the G33 spec.

## Findings

### G37-P0-1 — continuation identity does not make one-step selection receding-horizon

Code:

- `taskspace_receding_horizon_controller_v1.py:561-613` preserves evaluator-equal states with distinct `continuation_equivalence_sha256` values.
- `:660-662` nevertheless ranks representatives only by present exact score, archive bytes, and `action_id`.
- `:674-676` and `:737-742` mark every unselected action stale and require regeneration only from the selected base.

There is no horizon, reachable-set value, terminal value, continuation lower bound, viability kernel, or reversible beam. Therefore the continuation field changes quotienting but never changes the decision.

Exact adversarial reproduction on the audited code:

```text
same evaluator cells, same d_seg, same d_pose, same archive bytes
a.dead_end continuation != z.rich_future continuation

preserved_future_distinctions: one row with both continuation identities
selected_action_id: a.dead_end
stale_on_commit_action_ids: [macro.required, z.rich_future]
```

The lexical ID decides between equal present states even though the controller explicitly says their reachable future action spaces differ. More generally, it will choose `S(x_a) < S(x_b)` even when the best reachable descendant of `x_b` beats every descendant of `x_a`. Prune, merge, eviction, quantization, and factorization can be irreversible, so regenerate-after-commit does not repair the lost branch.

This contradicts the G33 spec's claim at lines 76-86 that the continuation quotient prevents deletion of a globally better continuation path. It prevents quotient deletion, but not commit deletion.

Required correction, one of:

1. **Real finite-horizon control:** for each first action `u`, close a depth-`H` action tree and select by terminal exact score
   `V_H(x,u) = min S(x_H)`
   over measured descendants plus verified admissible subtree bounds. Commit only the first action of the winning closed branch.
2. **Conservative search frontier:** retain a nondominated beam over `(present S, certified reachable lower bound, continuation class, hard constraints)`; stale only the old local derivatives, not alternate exact states whose continuation dominance is unproved.
3. Until either exists, rename and machine-label this surface `exact_one_step_endpoint_selector`, set `global_optimality_claim=false`, and replace `globally_lowest_exact_endpoint_selected` with `lowest_present_score_among_supplied_admissible_endpoints`.

Regression owed: two evaluator-equal/equal-score endpoints with distinct continuation certificates must not be resolved by `action_id`; selection must block or retain both unless one continuation is proved dominated.

### G37-P0-2 — Cartesian cell occupancy is not action-universe closure

Code:

- `CompleteActionUniverseV1.required_families` and `required_scales` are caller-supplied (`:397-400`).
- validation checks only that those caller-selected tuples contain known vocabulary entries (`:411-422`), not that they cover all nine families and four scales.
- each selected family-by-scale cell is considered closed when it contains at least one measured action ID, a caller-authored non-applicability row, or a dominance row (`:429-457`).
- `enumeration_complete` and `cross_scale_scan_complete` then return literal `True` (`:459-469`).
- each endpoint has exactly one `family` and one `scale`; `interaction_hypergraph_sha256` is only syntax-checked (`:263-323`) and is never interpreted.

Exact adversarial reproduction:

```text
required_families = [PRUNE]
required_scales = [MICRO]
one supplied PRUNE:MICRO endpoint

enumeration_complete: True
cross_scale_scan_complete: True
global_selection_closed: True
verdict: SELECT_ONE_ENDPOINT_THEN_REQUIRE_PAIRED_CONTEST_REPLAY
globally_lowest_exact_endpoint_selected: True
```

No merge, migration, inverse repair, fit, train, storage requantization, joint descent, meso, macro, or system action is represented. Even if all 36 single cells were occupied, a sign-reversing `PRUNE x INVERSE_REPAIR` bundle or a third-order `MERGE x REQUANTIZE_STORAGE x JOINT_DESCENT` action could still be absent. The G14 receipt proves that this is not hypothetical: cell-local action effects can reverse after composition.

The correct closure condition is not `every chosen cell has a row`; it is:

`generated finite action set = measured leaves union verifier-proved bounded subtrees union proved non-applicable domains`.

Required correction:

- Bind a typed `GeneratorDomainManifest` to the exact base. It must enumerate the full closed family/scale vocabulary, parameter domains, ordering/chronology domains, maximum interaction order or a closure construction, generator implementation hashes, deterministic seeds, and the exact Merkle root/cardinality of generated atomic and joint action IDs.
- Give an endpoint a support hyperedge, not one scalar cell: `support = ((family, scale), ...)`, `interaction_order`, parent action IDs, chronology/order, and context/base hash.
- Derive completion by reopening the manifest and proving every generated leaf measured or every omitted subtree closed by a verified certificate. A SHA string is a locator, not proof.
- If the generator domain is intentionally scoped, emit `scoped_selection_among_supplied_endpoints`; never `global_selection_closed`, `enumeration_complete`, or a fixed point.

Regression owed: omit one generated action while retaining a nonempty row in every family-by-scale cell; construction or selection must fail closed. Include the real G14 reversal hyperedge as an atomic endpoint fixture and prove no locally harmful constituent can be omitted.

### G37-P0-3 — branch-and-bound trusts an unverified scalar

Code:

- `ActionFamilyScaleCoverageV1` accepts `optimistic_score_lower_bound: float` and `evidence_sha256` (`:334-387`).
- the SHA is checked only for lowercase hex shape.
- `select_receding_horizon_action_v1` declares the branch proved dominated solely when that caller number is at least the exact incumbent (`:635-650`).

The inequality direction is mathematically correct *if* `L <= min_{leaf in subtree} S(leaf)` was independently proved. The implementation never proves that premise.

The focused test `test_certified_branch_and_bound_closes_unmeasured_dominated_family` demonstrates the hole: it supplies `_sha("merge-lower-bound-certificate")` with no certificate bytes or verifier and the number `0.30`; G33 reports both MERGE branches proved dominated and closes global selection against incumbent `0.23596485552462754`.

Required correction:

- Replace the scalar/hash pair with a typed proof object bound to exact base identity, generator subtree root, authority axis, decoder/runtime domain, continuation context, constraint constants, score implementation hash, derivation engine/version, and proof payload bytes/hash.
- Verify the bound before constructing a `CERTIFIED_DOMINATED` row. For analytic bounds, recompute from primitive intervals/lattice constraints. For finite subtrees, replay all leaves in tests. For nonconvex learned/repair domains without a valid relaxation, remain explicitly blocked.
- Never require callers to set `blocks_global_selection=false` for `CERTIFIED_DOMINATED`; the verifier should derive closed versus unresolved.

Regression owed: a random SHA plus a favorable scalar must fail; a certificate for another base, subtree, interaction context, or axis must fail; a valid bound below incumbent must remain open.

### G37-P1-4 — continuation and evaluator proof receipts are opaque assertions

The spec correctly says a caller-supplied identity hash alone has no authority. The code does not enforce that sentence.

`WholeObjectActionEndpointV1` merely shape-checks:

- evaluator-cell identity and ledger SHA;
- continuation-equivalence identity and receipt SHA;
- runtime and payload-placement SHA;
- measurement and public-auth SHA.

`_functional_quotient` keys only by evaluator-cell identity and continuation identity. It neither reopens nor validates either proof receipt and does not require equal proof dependencies, runtime, placement, generator ABI, or reachable-action manifest before declaring `quotient_safe=true`.

Exact adversarial reproduction accepted two endpoints with:

- equal evaluator-cell and continuation identity strings;
- different continuation proof-receipt hashes;
- different decoder runtime hashes;
- different payload-placement hashes.

G33 collapsed them, selected the cheaper one, and emitted `quotient_safe: true`.

Required correction: a trusted adapter must construct a parsed `ContinuationEquivalenceCertificate` whose hash is recomputed and whose payload binds all future-relevant state listed in G33 spec lines 86-87. G33 must verify the certificate or accept only a verified opaque capability type that untrusted callers cannot instantiate. Multiple proof receipts supporting the same semantic identity should be unioned into a proof-dependency set, not ignored.

### G37-P1-5 — official decode limits and dynamic target can be caller-laundered

`DecodeConstraintEvidenceV1` compares measured wall/RSS only with caller-supplied limits. It does not pin those limits to the contest contract or authority mode. Exact reproduction:

```text
measured_wall_seconds = 2000
wall_seconds_limit = 3600
blockers() = ()
```

The endpoint is called public-workflow feasible even though the binding project contract is 1,800 seconds. Peak memory has the same issue.

Likewise `WholeObjectBaseV1` stores a pointer artifact hash, bytes, and target score but does not reopen the artifact, prove the score comes from those bytes, or enforce freshness. This cannot corrupt base-relative one-step improvement, but it can falsify `dynamic_target_crossed` and frontier-candidate routing.

Required correction:

- derive hard limits from a pinned contest-contract receipt/constant keyed by authority axis; callers provide measurements, not limits;
- use the canonical dynamic-frontier snapshot type and verifier already used by G7, including artifact byte/hash binding and freshness/reopen checks;
- separate base score custody from target snapshot custody in the decision receipt.

### G37-P1-6 — Consumer 15 discards its coupled solve before returning the plan

The new machine labels are honest, but the useful coupled signal is orphaned:

- `_admm_solve` returns a plan, duals, and `nu_per_pair` (`master_gradient_consumers.py:4378-4514`).
- the public function passes that plan into `_greedy_primal_recovery` (`:4670-4696`).
- `_greedy_primal_recovery` copies the initial plan and then immediately resets every pair to `NONE` (`:4290-4321`). It rebuilds the returned plan from locally linearized marginal-score-per-byte ordering and stops at the first nonnegative marginal (`:4327-4345`).
- coupling affects `nu_per_pair` only inside the discarded solve. Returned `interaction_terms_with_pairs` are post-hoc labels; coupling is absent from returned `predicted_score_delta`.
- the reported KKT residual was computed on the pre-recovery plan and contains only normalized budget violation plus complementarity (`:4487-4503`), not stationarity and not residuals of the returned greedy plan.

Therefore Consumer 15 is not merely uncalibrated; its final proposal does not consume the coupled optimization it advertises. The focused interaction test computes `any_interaction` but never asserts it (`test_per_pair_optimal_treatment_plan.py:590-613`), so a no-effect coupling path passes.

Required correction for signal harvesting, not authority:

- Return the complete per-pair/treatment/theta candidate lattice, including locally non-improving candidates, the relaxed/dual proposal, the greedy proposal, and their distinct telemetry. Never discard candidates because G14 proves their joint sign may reverse.
- If greedy recovery remains, do not call its residual KKT and do not call the function optimal. Label it a local acquisition prior.
- Add a coupling-sensitivity test: changing only the coupling/hyperedge model must change the coupled proposal or explicitly prove why not.
- Materialize selected joint bundles through the exact whole-object endpoint generator before G33 sees them.

### G37-P1-7 — live and persisted Cathedral adapters disagree on safety

`optimal_plan_to_candidate_row` emits a feasible live proposal with no planning-only blocker (`master_gradient_consumers.py:4934-4945`), while `optimal_plan_payload_to_candidate_row` always adds `planning_only_master_gradient_optimal_plan_no_dispatch_packet` (`:5054-5077`). Both place the uncalibrated local `predicted_score_delta` in the ranker field and derive `expected_information_gain` from the arbitrary plus-or-minus-five-percent width (`:4913-4918`, `:5038-5044`).

The persisted-payload validator checks score/promotion/dispatch flags but does not require the new safety labels `endpoint_proposal_generator_only=true`, `global_optimality_claim=false`, `sign_reversal_safe=false`, `locally_nonimproving_candidates_pruned_by_recovery=true`, and `confidence_interval_calibrated=false` (`:4976-5015`). Thus older or incomplete sidecars can enter planning without proving the corrected semantics.

Required correction:

- give live and persisted adapters the same unconditional planning-only blocker;
- require every safety label at the persisted boundary;
- do not call the 5% width information gain or confidence; emit a separately named uncalibrated acquisition heuristic that cannot rank/kill or prune;
- preserve the full proposal ledger so ranker ordering cannot suppress nonlocal constituents.

## Scoped pass — what is coherent and should be kept

### G33

- Canonical nonlinear score functions are reused; no fixed Seg/Pose/rate box is used.
- Exact base identity, epoch, authority mode, and hardware axis are checked before comparison.
- Base infeasibility is rejected before using it as a branch-bound incumbent.
- Hard-infeasible endpoints are excluded rather than assigned soft score penalties.
- At most one present endpoint is selected, and old marginals are declared stale after a commit.
- Production commit, promotion, pointer movement, and score claims remain false.

These are scoped passes only; they do not make the supplied action set globally complete or provide future value.

### G7 whole-archive allocator

The current boundary is honest:

- it rebuilds and prices complete monolithic archive states;
- strict STORE/DEFLATE parse-back, double decode, receipt binding, exact nonlinear transition, and accepted-prefix remeasurement are tested;
- rejected trial states remain in `proposal_audits` rather than disappearing;
- truth is hard-coded research-only, proposal-generator-only, global-optimum-false, sign-reversal-unsafe, and caller-order-as-search-prior (`taskspace_whole_archive_allocator.py:366-386`);
- the docstring explicitly requires sign-reversing joint bundles to be rebuilt as same-base G33 endpoints (`:715-723`).

G7 remains unsafe as a selector because later trials run from the accepted greedy prefix and locally non-improving trials never enter that prefix (`:754-824`). It is acceptable only as acquisition telemetry. No G7 result may close a G33 cell or erase a rejected constituent.

### Consumer 15 labels

The new dataclass invariants correctly force proposal-only, non-global, sign-reversal-unsafe, whole-object-rebuild-required, and uncalibrated-interval labels. The bug is that downstream adapters and the internal plan construction do not yet fully honor those labels.

## Minimal coherent controller architecture

The capstone needs one finite proof-carrying search object, not another independent threshold layer:

```text
exact base x_k + fresh target snapshot
  -> verified finite generator-domain manifest G_k
  -> atomic and higher-order action hyperedges, all with exact support/context
  -> local costates rank acquisition only; full candidate ledger is retained
  -> each generated branch is either
       exact whole-object measured,
       verifier-proved bounded, or
       explicitly blocking
  -> quotient only with verified evaluator + continuation bisimulation proofs
  -> depth-H terminal-score/value closure or conservative multi-state beam
  -> commit one first action only when all competing continuations are dominated
  -> invalidate local derivatives, not unproved alternate states
  -> regenerate from new exact base
```

For a finite generator manifest `M_k`, the load-bearing invariant is:

`Leaves(M_k) = MeasuredLeaves union VerifiedBoundedSubtrees union ProvedNADomains`.

Family-by-scale occupancy is telemetry about `Leaves(M_k)`; it is not the equality proof.

## Patch-ready minimal exact first-action contract

The minimal correction does not require an infinite-horizon value network. It requires a finite, proof-carrying first-action tree with honest terminal intervals.

### 1. Branch objects and bounds

Fix exact base `x_k`, verified finite generator manifest `M_k`, and horizon `H >= 1`. Partition every reachable feasible terminal leaf by first action `a` and verified continuation class `q`:

`R_H(a,q) = {y : y is reached from x_k within H actions, first(y)=a, continuation(y)=q, hard_constraints(y)=true}`.

Every `(a,q)` branch must carry:

- `L[a,q]`, a **verified admissible terminal lower bound** satisfying
  `L[a,q] <= inf_{y in R_H(a,q)} S(y)`;
- an optional **feasible upper witness** `y_hat[a,q]` rebuilt and measured through the exact public workflow, with
  `U[a,q] = S(y_hat[a,q])`;
- exact first-action ID, action-support hyperedge, chronology/order, continuation certificate, reachable-subtree Merkle root/cardinality, remaining horizon, base/epoch/axis, score-implementation hash, hard-constraint-contract hash, and proof dependencies.

No feasible witness means `U[a,q] = +infinity`; it can never win, though its lower bound may still keep the branch open. No verified lower bound means `L[a,q] = -infinity`; the branch blocks commitment. Python/JSON need not serialize infinities: represent these as explicit `NO_FEASIBLE_UPPER_WITNESS` and `UNVERIFIED_LOWER_BOUND` states.

Aggregate by first action:

`L[a] = min_q L[a,q]`

`U[a] = min_q U[a,q]`, retaining the exact winning terminal witness and action sequence.

This is a terminal-score objective, not an additive sum of intermediate score deltas. Intermediate states need only satisfy the action/runtime contract required for the sequence; the committed first successor and terminal upper witness must satisfy all hard public constraints.

### 2. How the base/no-action branch enters

Add a controller-owned, non-caller-optional branch:

`a = BASE_STOP`, `R_H(BASE_STOP) = {x_k}`, `L[BASE_STOP] = U[BASE_STOP] = S(x_k)`.

Its witness is the already verified feasible base. This does two jobs:

1. supplies the initial feasible incumbent for branch-and-bound; and
2. prevents any non-improving action from being selected.

Do not conflate `BASE_STOP` with a defer/no-op action that preserves an opportunity to act later. If defer has distinct semantics, time/resource budget, or reachable set, it is an ordinary generated first-action branch with its own continuation certificate and bounds. In the contest objective, where waiting itself has no score value, a pure no-op with unchanged future action set is control-bisimilar to the current planning state and need not be duplicated.

### 3. Exact dominance and commitment rule

Maintain incumbent upper bound

`U_inc = min(S(x_k), min_a U[a])`.

A subtree may be pruned only after its typed proof verifies and its conservative lower bound satisfies

`L[subtree] >= U_inc`.

Select and commit non-base first action `a*` only if all of the following hold:

1. `U[a*]` has an exact feasible terminal witness whose recorded sequence begins with `a*`;
2. every generator-manifest leaf outside `a*` is measured or covered by a verified bound/non-applicability proof;
3. `U[a*] < S(x_k)`; and
4. `U[a*] < min_{b != a*} L[b]`.

The strict inequality is intentional. Equality between different continuation classes is not a safe lexical tie: retain/block both unless a verified control-bisimulation or continuation-dominance certificate proves either first action interchangeable. `action_id` may break ties only after exact successor-state and continuation equivalence has been proved.

After commit, all old **local derivatives/costates** become stale. Nonselected exact branch receipts remain historical search evidence; they are not called globally dead. Reopen the new exact base, rebuild `M_{k+1}`, and revalidate any reusable proof dependency before use.

The decision receipt must report `horizon`, every first action's `L/U` state, incumbent source, bound verifier result, terminal upper-witness identity, optimality gap, unresolved branches, and why strict dominance did or did not close. If the rule does not close, the verdict is `BLOCKED_CONTINUATION_DOMINANCE_UNPROVED`, not a fixed point.

### 4. Precise family-by-scale correction

`required_families` and `required_scales` must not be caller-selected closure claims.

- Remove them from public construction or require exact equality with version-pinned `ACTION_FAMILIES` and `ACTION_SCALES`.
- Require exactly one projection row for all `9 x 4 = 36` vocabulary cells. A cell may be `MEASURED`, `VERIFIED_BOUNDED`, `VERIFIED_NOT_APPLICABLE`, or `BLOCKED`; only a verifier may produce the two `VERIFIED_*` states.
- A measured cell is closed only when its manifest-derived generated-action root/cardinality equals its measured-action root/cardinality after canonical deduplication. Nonempty occupancy is insufficient.
- Family-by-scale rows are a projection audit, not the universe proof. The load-bearing manifest must also enumerate every support hyperedge and ordered action sequence through horizon `H`; a joint endpoint covers its full support set, not one arbitrarily chosen cell.
- The universe is closed only when the union of exact measured leaf IDs and verified closed subtree roots equals the exact leaf/subtree partition derived from `M_k`. Unknown extensions, generator failures, unbounded parameter domains, or an unverified N/A row are blocking.

Minimal adversarial tests:

1. caller supplies only `PRUNE x MICRO` -> constructor refuses;
2. all 36 cells occupied but one manifest leaf omitted -> closure false;
3. all singleton cells complete but one generated cross-family hyperedge omitted -> closure false;
4. N/A proof from another base/epoch/generator version -> refusal;
5. complete manifest plus exact measured leaves -> closure true independent of caller order.

### 5. Precise bound-proof correction

Replace `optimistic_score_lower_bound + evidence_sha256` with proof bytes that G33 itself reopens and verifies. A minimal `VerifiedSubtreeBoundV1` payload binds:

- schema/version and derivation method;
- exact base identity and universe epoch;
- generator program and domain-manifest SHA;
- subtree root SHA, leaf cardinality, first-action ID/support, continuation class, and horizon remaining;
- authority mode/axis and public hard-constraint-contract SHA;
- canonical score implementation/reference-byte denominator SHA;
- primitive conservative intervals or finite-leaf evidence from which the lower bound is recomputed;
- outward-rounding policy and derived lower-bound value;
- proof payload SHA and every dependency SHA.

The controller must recompute the payload hash, reopen dependencies, check every foreign key, and derive the bound. It must not accept a caller's `CERTIFIED_DOMINATED` status. Verification returns either a typed verified bound or a blocker; dominance is then derived from `L >= U_inc`.

For monotone independent primitive bounds, one valid form is

`L = 100*d_seg_lower + sqrt(10*d_pose_lower) + 25*bytes_lower/37_545_489`,

provided the verifier proves every leaf in the exact subtree satisfies all three lower bounds. Round `L` toward negative infinity before comparing so floating error cannot make pruning optimistic. For a finite subtree, replaying all leaf receipts is the reference verifier. For an arbitrary nonconvex learned/repair subtree without a proved relaxation, emit `UNVERIFIED_LOWER_BOUND` and block.

Minimal adversarial tests:

1. random SHA plus favorable scalar -> refusal;
2. valid proof bytes with one mutated bound primitive -> hash/derivation refusal;
3. proof bound to another base, axis, subtree root, horizon, or score implementation -> refusal;
4. recomputed `L < U_inc` -> branch remains open;
5. recomputed outward-rounded `L >= U_inc` -> subtree closes;
6. base `BASE_STOP` remains the incumbent when every action upper witness is non-improving.

## Six-hook consequences

1. **Sensitivity/costates:** preserve every atomic marginal, rejected proposal, and interaction acquisition reason; use them only to schedule exact descendants.
2. **Pareto/constraints:** track present score together with a verified continuation lower bound and hard feasibility; present-score dominance alone is insufficient.
3. **Bit allocator:** consume only exact archive endpoints or verified subtree bounds; never summed local byte/distortion predictions.
4. **Cathedral/autopilot:** proposal rows remain unconditionally blocked from rank/kill, dispatch, or family closure until a G33 endpoint adapter provides proof-carrying closure.
5. **Continual learning:** update costate/controller confidence only from same-base exact realized transitions; stale epochs and uncalibrated 5% widths cannot become posterior precision.
6. **Probe disambiguation:** spend measurements on branches whose interaction sign or continuation dominance is unresolved, especially constituents that are locally harmful but appear in sign-reversal hyperedges.

## Assumption challenge / unknown-unknown register

- **Assumption: the closed family vocabulary is complete.** Falsifier: a new representation exposes a new decoder-time repair or factorization action not expressible by the nine labels. Cure: versioned generator manifest with an explicit unknown/extension blocker.
- **Assumption: one family and one scale identify an action.** Falsifier: cross-family or cross-scale hyperedge. Cure: support sets and interaction order.
- **Assumption: pairwise interaction receipts cover the energy.** Falsifier: third-order sign change with all pairwise residuals small. Cure: endpoint rebuild and adaptive hyperedge acquisition; never pairwise pruning.
- **Assumption: a SHA proves the named theorem.** Falsifier: the focused synthetic B&B test already passes random label hashes. Cure: reopen and verify typed proof bytes.
- **Assumption: exact present score plus regeneration is globally safe.** Falsifier: irreversible action enters a worse continuation basin. Cure: horizon/value closure or retained beam.
- **Assumption: output-cell identity determines future behavior.** Falsifier: equal evaluator outputs with different decoder ABI, factor graph, reservoirs, or action generators. Cure: verified control-bisimulation certificate.
- **Assumption: old costates remain useful after an exact edit.** Current code correctly says no; the same rule must apply to Consumer-15 and G7 acquisition order after every committed state transition.
- **Assumption: an uncalibrated width is uncertainty.** Falsifier: it scales mechanically with predicted magnitude and contains no empirical coverage information. Cure: calibration receipts or a non-probabilistic heuristic label.

## Verification receipt

Audit-cut SHA-256 values:

```text
2149b81307059d62e8efc5825d8660275336edeff72467aa391de5d32784ef91  src/tac/witness_control/taskspace_receding_horizon_controller_v1.py
e32081c377d409433803ffb33420517c71c8b772244f691d99c595c2a7dbc98d  src/tac/witness_control/tests/test_taskspace_receding_horizon_controller_v1.py
c1048e4846f5c3e7079305ca1e401bd773fb21f664be36cd3a1f1c97a40e3eec  src/tac/witness_dsl/taskspace_whole_archive_allocator.py
110b97cb28d80291ee090933206ed910246d8cc102f10a6d419b517a353f7e6a  src/tac/witness_dsl/tests/test_taskspace_whole_archive_allocator.py
63ca144d5e6dd193cf86266e58a40f9b647894b8e48626cc5d97abcf94126f93  src/tac/master_gradient_consumers.py
b8b31e8acddfbe4258a2256f8409f8d208e1d2265a34b4ee10bb3a09a1229d6a  src/tac/tests/test_per_pair_optimal_treatment_plan.py
9108cabc3db604b3b28e13a6b0bb5a818f1c1328b6ab49d20efb57466a7c7213  SPEC_g14_taskspace_g8_a3_n2_allocator_runner_20260726.md
c63f9fd6ac1df2313d69bcccdc4152ac9a9cc8ee904bfb80da1e7e2c77118eae  SPEC_g33_taskspace_receding_horizon_controller_20260726.md
9db91f681131c6cd1126a5dd8b2ee048b4d35b08967509ec07d240b016078338  taskspace_feedback_costate_materialization_n600_v5_20260726.json
```

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  src/tac/witness_control/tests/test_taskspace_receding_horizon_controller_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_whole_archive_allocator.py \
  src/tac/tests/test_per_pair_optimal_treatment_plan.py
```

Result: `90 passed in 2.74s` on the final audit cut. These are implementation QA, not n600 or score evidence.

Lint/format check:

```bash
.venv/bin/ruff check <the six scoped Python files>
.venv/bin/ruff format --check <the six scoped Python files>
```

Result: not clean at the audit cut. Ruff reported five issues in root-owned `test_per_pair_optimal_treatment_plan.py`: unsorted imports, one unused import, and three unused locals; one unused local is the non-asserted `any_interaction`. Format-check reported that `master_gradient_consumers.py` and its focused test would be reformatted. G37 did not edit root-owned files.

Adversarial read-only scripts also reproduced:

- one-family/one-scale/one-endpoint false global closure;
- arbitrary numeric-plus-SHA B&B closure;
- equal present score with different continuation classes selected by lexical action ID and alternate continuation staled;
- quotienting of different runtime/placement/proof receipts behind one opaque continuation identity;
- 2,000-second workflow accepted under caller-supplied 3,600-second limit.

## Stores consulted

- full `CLAUDE.md` and byte-identical `AGENTS.md`, SHA-256 `47d4ac3a38f91a8b8e7dc3061131717d8122bd48ffb204ffb914eb58e687f0c9`;
- `PROGRAM.md`;
- top-10 project Claude `MEMORY.md` hooks;
- `/Users/adpena/.codex/skills/arbitrage/SKILL.md`;
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, lane maturity validation, and recent directive scan (`0` recent directives);
- G14 and G33 specifications named above;
- real G14/G18/G19 v5 materialization named above;
- all scoped source and focused tests.

No external paradigm research, score run, candidate construction, paid dispatch, lane dispatch, pointer mutation, commit, push, or public payload reuse occurred.

## Pointer delta honesty

Pointer moved: **false**.  
Frontier score lowered: **false**.  
Candidate archive produced: **false**.  
Exact contest evaluation performed: **false**.  
Concrete delta: three load-bearing controller P0s and four integration P1s are reproduced, localized, and given patch-ready invariants. Until P0-1 through P0-3 close, G33 may safely serve only as an exact present-endpoint comparison receipt, not as a globally complete receding-horizon decision organ.
