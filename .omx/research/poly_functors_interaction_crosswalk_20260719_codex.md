# Polynomial functors as interaction contracts: ranked Pact crosswalk

**Date:** 2026-07-19  
**Lane:** `poly_functors_interaction_crosswalk_20260719`  
**Mode:** `research_only=true`; local read-only analysis; no launch, dispatch, score, or pointer authority  
**Authority axis:** MEANS-only. The authority snapshot and `reports/latest.md` both name **0.1910828242 [contest-CPU]**; this lane neither measures nor mutates a pointer.  
**Primary source:** Nelson Niu and David I. Spivak, *Polynomial Functors: A Mathematical Theory of Interaction*, arXiv:2312.00990v2 (2024), 372 PDF pages, source PDF SHA-256 `02d87ffacc1a54b5cbff1d7b93aa96018afcbda0caa62af0e321bbf4be01c6a5`.

## One-line verdict

**DERIVED:** polynomial functors are useful here as a typed vocabulary for mode-dependent interfaces and backward reply routing, but they provide neither operational correctness nor arbitration authority; propose six narrow schemas across the four requested axes, use the lens/VJP analogy only as custody notation, and explicitly reject product/composition as an organ-authority resolver.

## Scope, labels, and stop conditions

- **MEASURED** means directly read from the current repository or exact source artifact. No new runtime measurement was performed.
- **DERIVED** means a conclusion follows from cited definitions plus an explicit repository invariant.
- **INFERRED** means a proposed transfer that is not a theorem of the monograph and must pass its named falsifier before adoption.
- Negative conclusions carry a `verdict_scope`; no result below kills the broader polynomial-functor family.
- The live artifact `experiments/results/levelset_n600_witness_20260717T113932Z/` was not opened, written, moved, or used as evidence.
- The bounded-probe discipline follows `docs/operating_manual_craft_handoff.md`: the consumer and stopping observation are named before any proposed probe. All adoption tests below are local and $0.

## Monograph facts that transfer—and facts that do not

For a polynomial

\[
P \cong \sum_{i\in P(1)} y^{P[i]},
\]

`i` is a **position** (the mode/status currently exposed) and `P[i]` is the set of **directions** (the replies admissible in that mode). This positions/directions reading is formalized in Definition 2.1 and Proposition 2.6/Definition 2.8 (monograph pp. 25–27; PDF pp. 37–39).

A lens `f : P -> Q` consists of:

1. a forward position map `f_1 : P(1) -> Q(1)`; and
2. at each internal position `i`, a backward direction map
   `f_i# : Q[f_1(i)] -> P[i]`.

This is the characterization in Proposition 3.6 (p. 43; PDF p. 55), with the component names fixed in Definition 3.9 (p. 44; PDF p. 56). Lens composition composes the visible maps forward and the direction maps backward (Exercise 3.49, restated by Proposition 3.50, p. 61; PDF p. 73). For `f : P -> Q` and `g : Q -> R`, the composite has `h_1 = g_1 ∘ f_1` and `h_i# = f_i# ∘ g_{f_1(i)}#`. Example 3.10 calls this an interaction protocol, but it is an interface shape—not a theorem about framing, persistence, atomicity, retry, liveness, or software correctness.

The following Chapter 6–8 facts are a separately verified extension beyond the Chapter 2–5 lens evidence pass. The two products relevant to the assignment are different:

- **Dirichlet/parallel product** `P ⊗ Q`: positions are pairs and directions are paired products (Definition 3.65, p. 67; PDF p. 79).
- **Composition product** `P ⊳ Q := P ∘ Q`: one polynomial is substituted into another and can model a nested interface after an operational interpretation (Definition 6.1, Proposition 6.2, and the notation following Corollary 6.5, pp. 178–179; PDF pp. 190–191). Example 6.38 (pp. 191–192; PDF pp. 203–204) is an especially useful warning: the composite position set can include misleading graftings, so substitution does not by itself encode which combinations are valid.

Polynomial comonoids have erasure and duplication maps satisfying counit and coassociativity (Definition 7.14, p. 235; PDF p. 247), and Theorem 7.28 gives a one-to-one, isomorphism-preserving correspondence between polynomial comonoids and small categories (p. 240; PDF p. 252). This object-level result alone is not an equivalence of the ordinary category of small categories with `Comon(Poly)`: the latter's morphisms correspond to retrofunctors. The book's current term for object-forward/arrow-backward structure is **retrofunctor** (Definition 7.55, p. 255; PDF p. 267); “cofunctor” is noted there as older terminology. State-category retrofunctors correspond to well-behaved lenses in Example 7.85 (pp. 265–267; PDF pp. 277–279).

**NO-TRANSFER verdicts:**

| Claim not licensed by the source | Verdict | `verdict_scope` |
|---|---|---|
| “A polynomial interface proves the implementation cannot drop or duplicate events.” | **REJECT** | Interface notation only; a cursor protocol with persistence tests remains viable. |
| “Lens composition is reverse-mode autodiff or proves VJP correctness.” | **REJECT** | Set-valued lenses only; a differentiable, dual-space instantiation remains viable if independently proved. |
| “Categorical associativity proves fp32/uint8/resize/parse-back commutation.” | **REJECT** | Numerical realization only; exact byte-level commutation can still be measured per factorization. |
| “Dirichlet or composition products choose the rightful organ actuator.” | **REJECT** | Authority selection only; a repository-defined wrapper can encode an already-chosen policy. |
| “A comonoid computes the right curriculum re-anchor.” | **REJECT** | Numeric re-anchoring only; category laws can still expose path-coherence obligations. |

## Ranked crosswalk

Ranking uses defect silence, blast radius, and ease of a decisive $0 falsifier—not expected score movement.

| Rank | Transfer | Current consumer | Adoption | Decisive test |
|---:|---|---|---|---|
| 1 | Landing disposition as a mode-dependent polynomial | `tools/codex_landing_review_gate.py` consuming `tools/codex_delegate.py` output | **PROPOSE after falsifier** | A tracked edit plus an untracked deliverable must not receive a green disposition without a complete base-to-head manifest. |
| 2 | Event-gated curriculum as a state category with explicit re-anchor scope | `src/tac/witness_dsl/curriculum_dsl.py`, `src/tac/witness_control/resume_registry.py`, witness trainer | **PROPOSE narrowly** | Early event, checkpoint, resume, and alternative parenthesizations must yield identical durable controller state and the declared re-anchor delta only. |
| 3 | Launcher-to-trainer resume as an explicit interaction polynomial | `tools/launch_witness_run.py` and `experiments/train_levelset_witness_realized_through_R_mlx.py` | **PROPOSE after continuity proof** | Continuous two-step run must equal one-step plus resume across live/EMA/optimizer/controller/render custody. |
| 4 | Monitor-to-event-log as a cursor-custodied polynomial | `src/tac/witness_run_monitor.py`; analogous fleet notifier in `tools/codex_delegate.py` | **PROPOSE after replay proof** | Prove lossless ordered replay across clean restart/rotation and explicitly close or scope the delivery-before-ACK crash window. |
| 5 | Lens-composition/VJP-order analogy as custody notation | `tools/measure_joint_seg_pose_rate.py` and the positive-band/preimage receipts | **PROPOSE notation only** | Candidate factorizations are admitted only when native fp32, uint8 bytes, parse-back, and scorer outputs agree—not when range values merely agree. |
| 6 | Explicit authority wrapper around regime dispatch | `src/tac/witness_control/regime_dispatch.py`; dashboard and digest consumers | **PROPOSE before any actuation bridge** | Conflicting global and per-state recommendations must leave trainer state unchanged and emit one scoped conflict/refusal record. |
| 7 | Products themselves arbitrate overlapping authority | none | **REJECT** | No test can derive a policy absent extra repository semantics; do not implement this premise. |

## 1. Landing arm -> landing gate disposition

### Current fact pattern

- **MEASURED:** `tools/codex_delegate.py` appends `DONE` and `LANDING-REVIEW-REQUIRED`, then writes a `.done` file. Its optional manifest path is described in the prompt but is not a typed field in the delegation ledger.
- **MEASURED:** `tools/codex_landing_review_gate.py` reasons over `.done`/`.last.txt` and can write a terminal disposition, but its current evidence model does not require a base SHA, head SHA, complete tracked path set, untracked path set, or diff digest.
- **DERIVED:** completion status and landing completeness are different modes. Flattening them into one “done” bit permits silent loss of an untracked deliverable.

### Polynomial sketch

Let

\[
P_{land}=\sum_{s\in S_{land}} y^{A_s}
\]

with positions

`RUNNING`, `DONE_NO_DIFF`, `DONE_TRACKED_ONLY`, `DONE_WITH_UNTRACKED`, `FAILED`, and `REVIEWED`.
The direction set depends on the position. In particular:

- `A_DONE_NO_DIFF = {REQUEST_MANIFEST, REFUSE}`;
- `A_DONE_TRACKED_ONLY = {REQUEST_UNTRACKED_SCAN, REFUSE}`;
- `A_DONE_WITH_UNTRACKED = {REQUEST_COVERAGE, REFUSE}`;
- only a manifest-complete position admits `WRITE_DISPOSITION`.

An **INFERRED** lens from the arm's internal state to the gate's visible state would expose a normalized position forward and translate the gate reply backward into a concrete arm/reviewer action. The lens does not prove that the manifest is truthful; the digest and test do.

Proposed typed receipt:

```text
LandingDiffManifest {
  base_sha, head_sha,
  changed_paths[], untracked_paths[],
  tracked_diff_sha256,
  per_path_sha256{}, untracked_tree_sha256,
  deliverable_paths[],
  generator
}
```

**Named consumer:** `tools/codex_landing_review_gate.py`; `WRITE_DISPOSITION` must fail closed if any field is absent or if a listed deliverable is outside `changed_paths union untracked_paths`.

**Falsifiable adoption test `landing_manifest_coverage`:** in a temporary git fixture, create one tracked modification and one untracked deliverable, write the current done marker, and invoke the gate with the presently required `--consumed-by` argument. The pre-adoption behavior is falsified if it can go green without both paths. The post-adoption acceptance criterion is refusal until a manifest names both paths, every recomputed per-path hash matches, the tracked diff hash matches, and the untracked tree hash binds the complete untracked set. Stop after this single counterexample or a green strict fixture; no repository-wide build is needed.

## 2. Event-gated curriculum -> state category and explicit re-anchoring

### Current fact pattern

- **MEASURED:** `Stage`, `StageSpec`, `EventTriggeredCurriculum`, and `CurriculumReanchorLevers` are typed in `src/tac/witness_dsl/curriculum_dsl.py`.
- **DERIVED:** the stage/event fields are sufficient to propose mode-dependent legal actions, but the current classes are not themselves a polynomial interface or event-composition API.
- **MEASURED:** the trainer restores event state, maps lever epoch through the fired tau, and consumes that state for lane, persistence, and seed behavior.
- **MEASURED:** `src/tac/witness_control/resume_registry.py` persists run-scoped controller state. `ResumeRegistry.restore()` refuses when a persisted manifest declares an event-active controller whose prefix has vanished; legacy no-manifest resumes retain warning/backward-compatible behavior.
- **MEASURED:** chroma absolute/re-anchor treatment is still named as a follow-on item in `src/tac/witness_autoconfig.py`; the transfer is therefore partial, not a claim of full re-anchor closure.

### Comonoid/category transfer

Model durable curriculum states and their allowed transitions as a small category `C_curr`:

- objects: `(stage, tau_fired, fired_epoch, controller_digest, reanchor_scope)`;
- arrows: typed events that are actually legal from that object;
- identity: “observe/no transition” leaves the durable digest unchanged;
- composition: sequential accepted events compose;
- associativity obligation: regrouping the same event path must produce the same final durable state.

Equivalently, the interface polynomial can be written

\[
P_{curr}=\sum_{m\in\{CE,tau,l7,Muon,halt\}} y^{E_m},
\]

where `E_m` contains only the events admissible in mode `m`. A retrofunctor is a useful **INFERRED** specification device: expose the internal durable state as an external mode, then lift an admitted external event backward to the internal transition selected by separately declared deterministic policy. It must preserve identity and composition. Neither uniqueness nor the numerical re-anchor follows from the retrofunctor laws.

Proposed explicit field:

```text
ReanchorScope = ABSOLUTE | FIRED_TAU_RELATIVE | STAGE_LOCAL
```

Each lever declares one scope. Existing lane/persistence/seed consumers that subtract the fired tau use `FIRED_TAU_RELATIVE`; explicitly absolute chroma schedules remain `ABSOLUTE`. Mixed implicit interpretation is a compile refusal.

**Named consumers:** `CurriculumReanchorLevers` compilation, trainer event-state restore, and `ResumeRegistry` persistence. The categorical event-composition harness proposed below does not yet exist.

**Falsifiable adoption test `curriculum_retrofunctor_resume`:** build a new tiny harness that groups and applies typed event paths; fire tau earlier than its cap, serialize at the next stage boundary, resume, and apply two subsequent admissible events. Assert:

1. `(e1 ; e2) ; e3` and `e1 ; (e2 ; e3)` have identical controller/state digests;
2. lane/persistence/seed epochs shift by exactly `fired_tau - cap_tau` only when their scope is relative;
3. an absolute chroma schedule does not shift;
4. deleting event state makes resume refuse.

Adopt only if all four are deterministic on the tiny fixture. `verdict_scope=curriculum transition/re-anchor contract`; this says nothing about whether the curriculum improves contest score.

## 3. Launcher -> trainer resume protocol

### Current fact pattern

- **MEASURED:** when explicitly selected, `tools/launch_witness_run.py --dry-start` performs an exact-config fresh boot followed by a resume round trip, then exits instead of entering the durable-spawn path. Dry-start is optional and default-off.
- **MEASURED:** the trainer restores live weights, EMA, optimizer, and epoch state; writes latest checkpoints atomically; preserves stage-encoded checkpoints; and emits transition/final checkpoints.
- **DERIVED:** “reloads and steps past the checkpoint” is weaker than “split execution equals uninterrupted execution.” The latter additionally requires RNG, controller sidecars, optimizer slots, EMA, and next-render equality.

### Polynomial sketch

Let positions be the custody state:

```text
NO_CHECKPOINT
VERIFIED_CHECKPOINT(checkpoint_sha, config_sha, stage, epoch, state_digest)
LEGACY_CHECKPOINT(reason)
REFUSED(reason)
```

Directions depend on the position:

- `NO_CHECKPOINT -> {START_FRESH, REFUSE}`;
- `VERIFIED_CHECKPOINT -> {DRY_ROUNDTRIP, RESUME_EXPECTING(next_state_digest), REFUSE}`;
- `LEGACY_CHECKPOINT -> {MIGRATE_WITH_RECEIPT, REFUSE}`;
- `REFUSED -> {RECORD_ONLY}`.

The launcher-to-trainer lens exposes custody forward and translates the launcher request backward into exact trainer argv plus the expected postcondition. If a separately implemented typed boundary rejects every input outside the declared direction set, it can make illegal replies unrepresentable at that boundary; the polynomial notation alone does not. The process-level continuity test remains the authority.

Proposed receipt:

```text
ResumeContinuityReceipt.v1 {
  seed, config_sha, checkpoint_sha,
  continuous_state_digest, resumed_state_digest,
  continuous_live_sha, resumed_live_sha,
  continuous_ema_sha, resumed_ema_sha,
  continuous_optimizer_sha, resumed_optimizer_sha,
  continuous_controller_sha, resumed_controller_sha,
  continuous_next_render_sha, resumed_next_render_sha,
  equality_verdict
}
```

**Named consumers:** the dry-start gate in `tools/launch_witness_run.py` and trainer resume loader.

**Falsifiable adoption test `resume_split_equivalence`:** on a deterministic two-epoch fixture, compare a continuous run with one epoch plus checkpoint plus resume. Require exact equality of live/EMA/optimizer/controller digests and the next realized render. Any mismatch rejects the contract or names the first divergent component. This is a continuity test, not a launch authorization.

## 4. Monitor -> event-log protocol

### Current fact pattern

- **MEASURED:** `src/tac/witness_run_monitor.py` defaults to `tail -n0 -F`; `from_start=True` is an all-history backfill.
- **MEASURED:** the fleet notifier in `tools/codex_delegate.py` uses the same new-events-only tail policy and appends terminal rows to a shared log.
- **DERIVED:** neither surface persists `(stream identity, inode, byte offset, last accepted record)`. A stopped monitor can miss records written while absent, while a full backfill can duplicate them.

### Polynomial sketch

```text
EventCursor {
  stream_id, inode, byte_offset,
  last_record_hash, sequence, delivery_id,
  state: OPEN | ROTATED | TRUNCATED | CORRUPT
}
```

For `OPEN`, legal directions are `{READ_NEXT, ACK_AND_PERSIST, STOP}`. For `ROTATED`, they are `{PROVE_SUCCESSOR, REPLAY_FROM_CURSOR, REFUSE}`. `TRUNCATED` and `CORRUPT` do not admit a silent reset. A monitor lens can expose a human-facing event category forward while translating acknowledgement backward into a cursor-update request. Atomic persistence is a separate implementation and transaction invariant.

**Named consumer:** `src/tac/witness_run_monitor.py`; the same schema can later be reused by the delegation notifier, but that reuse is outside this landing.

**Falsifiable adoption test `event_cursor_lossless_replay`:**

1. Clean path: append `E1`, consume and persist; stop; append `E2`; restart from the persisted cursor; rotate and append `E3`. Require ordered `E1,E2,E3`, with no clean-path loss or duplicate and the final cursor after `E3`.
2. Failure window: inject a crash after exposing `E2` but before cursor ACK persistence. On restart, require either (a) an idempotent consumer keyed by `delivery_id`, with its effect and cursor ACK atomically coupled, or (b) an explicit `AT_LEAST_ONCE` disposition that reports the replayed delivery. A cursor alone cannot prove exactly-once external effects.

Missing rows, silent duplicates, or an undeclared delivery guarantee reject adoption. `verdict_scope=one durable local consumer`; no distributed-delivery claim is made.

## 5. Lens-composition/VJP-order analogy -> custody, with a hard uint8 boundary

### What the analogy legitimately says

After separately specifying differentiable spaces and a dual-space interpretation, one may propose a lens-like interface whose forward component is `x -> F(x)` and whose backward component maps a cotangent by `v -> J_F(x)^T v`. For independently differentiable `G`, the backward-map order resembles the VJP chain-rule order:

\[
J_{G\circ F}(x)^T v = J_F(x)^T\bigl(J_G(F(x))^T v\bigr).
\]

The displayed equality follows from ordinary differential calculus alone. Its comparison with lens composition is an **INFERRED analogy**, not a correspondence proved by the monograph. The book works in `Set` and supplies no functor from differentiable maps/VJPs into `Poly`, tangent spaces, duals, differentiability, floating-point semantics, or autodiff implementation.

### Current custody surface

- **MEASURED:** `tools/produce_vjp_custody.py` is the VJP producer. `tools/measure_joint_seg_pose_rate.py` loads and validates custodied VJP manifests, invokes the hard oracle, and restricts admission to the declared Seg/Pose condition.
- **MEASURED:** `.omx/research/vjp_custody_positive_bands_20260719_codex.md` separates exact scorer-plane range equality from equality of preimages under native execution.
- **DERIVED:** a realization path can contain quantization and other non-injective or discontinuous steps. The source establishes no commutation result for resize, serialization, parse-back, or scorer execution, so no categorical reassociation or real-valued range equality proves that two implementations commute through the repository's actual path.

Proposed receipt extension:

```text
PreimageSelectionReceipt {
  policy_id, candidate_ids[],
  native_fp32_mismatches,
  uint8_frame_hashes[], parseback_hash,
  scorer_output_hash, selected_frame_hashes[]
}
```

**Named consumer:** the hard-oracle/admission path in `tools/measure_joint_seg_pose_rate.py`.

**Falsifiable adoption test `vjp_factorization_native_commutation`:** for each proposed VJP or renderer factorization, execute both forms through native fp32 and the actual rounding/uint8/resize path, then extend the test through archive parse-back and the frozen scorer once the owed receiver/archive gate exists. Call the measured portion of the diagram commutative only if exact uint8 frame hashes and scorer outputs agree on the preregistered boundary/tie lattice. A matching real-valued/intermediate field with differing preimages is a failure. Parse-back is an extension acceptance condition, not a claim that the current measurement tool owns receiver/archive closure. `verdict_scope=native realization of that factorization`; the underlying mathematical lens remains valid.

## 6–7. Dirichlet/composition products -> organ regime dispatch and authority

### Current fact pattern

- **MEASURED:** `src/tac/witness_control/regime_dispatch.py` documents a global-single-best baseline versus a per-state regime router, returns a typed recommendation, and hard-codes `actuation="NONE"`. Its baseline comparison is internal; there is no live adapter to `continual_costate.arbitrate_architecture`.
- **MEASURED:** advisory consumers include dashboard/digest surfaces, `lambda_net_backtest.py`, `costate_warmstart_cluster.py`, and `costate_agent_dsl.py`; the elevation backtest can declare the dispatcher inert for an always-on recommendation. None of these findings establishes an actuation path.
- **DERIVED:** an explicit global authority and a per-state router could disagree without either being malformed. If both later obtain actuation paths, product composition alone cannot decide which owns the trainer action.

### Correct product reading

- `P_global ⊗ P_regime` models the two recommendation interfaces as available in parallel. It faithfully preserves both replies; it does not rank them.
- `P_outer ⊳ P_inner` can model a nested interface structure after an operational interpretation. It makes the choice of outer/inner structure explicit; it neither guarantees runtime sequencing nor justifies that choice.
- Example 6.38's extra graftings are directly cautionary: a polynomial composition can express structurally possible combinations that the application must still forbid.

The missing ingredient is repository policy. Encode it explicitly rather than attributing it to the product:

\[
A=\sum_{s\in\{NO\_OWNER,ONE\_OWNER,CONFLICT,ADVISORY\}} y^{D_s}.
\]

Suggested receipt:

```text
AuthorityDecision {
  scope: FORECAST | TRAINER_ACTION,
  run_hash, source, selected_tool,
  authority_owner, execution_consumer,
  candidate_decision_hashes[],
  disposition: ADVISORY | ACCEPT | REFUSE_CONFLICT
}
```

At `CONFLICT`, the direction set is `{RECORD, REFUSE, REQUEST_EXPLICIT_POLICY}`—never `ACTUATE`. A repository-defined lens from paired recommendations into this wrapper is **INFERRED policy encoding**, not a monograph-derived arbitrator.

**Named consumer:** a future bridge into trainer actions; today `regime_dispatch.DispatchDecision` remains advisory, and its dashboard, digest, backtest, warm-start-cluster, and DSL consumers remain non-actuating.

**Falsifiable adoption test `regime_authority_conflict`:** build a new adapter/harness that supplies one explicit global-authority decision and one per-state decision, then construct a past-only fixture where they select different tools. Assert that (a) both recommendation hashes survive, (b) exactly one `AuthorityDecision(scope=FORECAST)` is recorded, (c) no `TRAINER_ACTION` owner exists, (d) trainer/config/checkpoint hashes remain unchanged, and (e) `actuation == "NONE"`. Any mutation is a P0 rejection. Before any future actuation bridge, additionally require a unique owner for `TRAINER_ACTION`; two owners must refuse.

**REJECTED alternative:** treating either `⊗` or `⊳` as the arbitration rule. `verdict_scope=authority selection`; both products remain useful for representing already-governed parallel or nested interaction.

## Proposed implementation order and concrete consumers

1. **First candidate:** a separately owned lane for the strict landing manifest, because the current failure is silent and can lose the arm's only deliverable.
2. **Second candidate:** explicit `ReanchorScope` compilation plus the existing resume registry on the associativity fixture.
3. **Third candidate:** strengthen dry-start from reload/forward-progress to split/continuous equivalence.
4. **Fourth candidate:** replace ephemeral tail position with a durable event cursor.
5. **Later, on demand:** extend the existing VJP receipt only when another preimage policy is ready for comparison.
6. **Before, not after, any separately authorized actuation:** route an authority-wrapper lane for #436. Do not change today's `actuation="NONE"` in this research lane.

No implementation is included here because the assignment is research/routing and each consumer has a distinct owner/hot-file surface. MAIN should route the first four as separate implementation lanes after reviewing the falsifiers and ownership map.

## Triality and unified-solver wire-in

| Leg/hook | This landing |
|---|---|
| DSL | Proposed typed position/direction schemas: `LandingDiffManifest`, `ReanchorScope`, `ResumeContinuityReceipt`, `EventCursor`, `PreimageSelectionReceipt`, `AuthorityDecision`. This research landing makes no compiler change; a future `ReanchorScope` implementation necessarily would. |
| DAG | `arm -> manifest -> landing gate`; `event -> typed transition -> checkpoint -> resume`; `launcher request -> trainer state -> continuity receipt`; `log append -> cursor -> monitor`; `candidate -> VJP -> native R -> parse-back -> hard oracle`; `recommendations -> authority wrapper -> advisory/refusal`. |
| Equations | Polynomial positions/directions, lens backward reply map, category identity/associativity, conditional VJP chain rule, and explicit non-commutation across uint8 above. |
| Sensitivity map | No new empirical derivative. The VJP proposal's consumer is the existing positive-band custody path; no sensitivity posterior is mutated. |
| Pareto / bit allocator | Non-binding: this memo changes interface custody only and claims no Seg/Pose/rate delta or byte allocation. |
| Autopilot / dispatch | No hook and no actuation. The #436 proposal is deliberately a refusal wrapper before any future bridge. |
| Continual learning | No empirical anchor was produced, so no posterior row is authorized. The six named falsifiers are the future probe-disambiguators. |

## Self-review ledger

The landing is capped at five rounds. The final count and corrections are recorded here before commit.

1. **Round 1 — source fidelity: COMPLETE/AMBER -> corrected.** Moved the lens characterization to Proposition 3.6, changed Example 3.49 to Exercise 3.49, stated Theorem 7.28 at object-correspondence strength, and used the book's `⊳` notation.
2. **Round 2 — repository fidelity: COMPLETE/AMBER -> corrected.** Fixed the stage/registry/autoconfig names, qualified legacy restore and optional dry-start, bound untracked bytes in the landing receipt, separated VJP producer from consumer, and made the regime conflict adapter explicitly new.
3. **Round 3 — authority/scope: COMPLETE/AMBER -> corrected.** Demoted lens/VJP to an inferred order analogy, separated polynomial notation from atomicity/uniqueness/runtime sequencing, and narrowed numerical/receiver claims.
4. **Round 4 — final integration: COMPLETE/GREEN after two fixes.** Replaced stale review placeholders and changed the cursor claim from unsupported exactly-once delivery to clean lossless replay plus an explicit delivery-before-ACK failure-window contract. No fifth round was needed.

## Stores consulted

- Authority prompt and its verified SHA-256.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and `docs/operating_manual_craft_handoff.md`.
- Niu–Spivak arXiv PDF v2, full extracted text, chapter summaries, definitions/results cited above, and rendered visual checks of the interaction-lens, Dirichlet, composition-product, and comonoid pages.
- `reports/latest.md` and `current_focus.md` for pointer separation only.
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, delegated/broadcast inboxes, and the latest available sister findings/session/design/council memos for ownership and scope.
- `docs/triality_dag_dsl_equations_deepmath.md` and `.omx/research/v10_flattened_lagrangian_kkt_derivation_20260719.md`.
- `src/tac/witness_dsl/curriculum_dsl.py`, `src/tac/witness_control/resume_registry.py`, `src/tac/witness_control/event_wirings.py`, `src/tac/witness_autoconfig.py`, and the witness trainer resume/event paths.
- `tools/launch_witness_run.py`, `src/tac/witness_run_monitor.py`, `tools/codex_delegate.py`, and `tools/codex_landing_review_gate.py`.
- `.omx/research/costate_controller_design_20260705.md`, `src/tac/witness_control/regime_dispatch.py`, `tools/render_levelset_dashboard.py`, `tools/costate_digest.py`, `tools/lambda_net_backtest.py`, `src/tac/witness_control/costate_warmstart_cluster.py`, `src/tac/witness_dsl/costate_agent_dsl.py`, and the organ elevation backtest.
- `.omx/research/vjp_custody_positive_bands_20260719_codex.md`, `tools/produce_vjp_custody.py`, and `tools/measure_joint_seg_pose_rate.py`.
- Prior read-only curriculum-audit memory was used only to identify the earlier `event_curriculum_inert_under_unify` scope; current source was re-read before making any present-tense structural claim.

## Pointer delta and custody

- **Report/authority pointer before:** 0.1910828242 [contest-CPU], restated solely to prove separation; not newly measured or promoted here.
- **Report/authority pointer after:** 0.1910828242 [contest-CPU], unchanged and not asserted as a new canonical measurement by this lane.
- **Delta:** exactly zero; no score evaluation, archive mutation, paid dispatch, GPU action, or launch occurred.
- **Sacred run:** `experiments/results/levelset_n600_witness_20260717T113932Z/` untouched.
- **Durable output:** this memo plus the L0 `research_only=true` lane registration. MAIN landing review is required before any merge or implementation routing.
