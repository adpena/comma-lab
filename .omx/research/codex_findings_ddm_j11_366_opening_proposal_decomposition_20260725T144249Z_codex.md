---
schema: codex_findings_v1
lane_id: lane_ddm_j11_366_opening_proposal_decomposition_20260725
authority_sha256: 36ec3ede75ac1bf2d9f6565d4e05ba7e4d2202efdc1174895e644586b88d7ec5
research_only: true
score_claim: false
pointer_moved: false
main_review_required: true
verdict: BLOCKED_J11_PROPOSAL_DECOMPOSITION_CUSTODY_PRECONDITION
verdict_scope: PRECONDITION_APPARATUS_NOT_FORMULATION
---

# DDM J11 #366 opening-proposal decomposition — findings

## Disposition

The requested decomposition cannot be lawfully materialized from the named inputs. The landed
proposal-source audit fails closed with:

`BLOCKED_J11_PROPOSAL_DECOMPOSITION_CUSTODY_PRECONDITION`

This is a **PRECONDITION/APPARATUS** result. It does not say that a correctly joined
pose-null/Seg-null decomposition is unproductive:

- **NO formulation-negative claim**
- **NO component score claim**
- **NO bounded scorer smoke or campaign**
- **NO `READY_TO_FIRE_UNDER_STANDING_GO`**
- pointer `0.1910828242 [contest-CPU]` **UNMOVED**

## Sealed J10 source and unchanged repricing

The exact source remains 138,813 bytes, SHA-256
`2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241`,
d_seg `0.06974277072482639`, d_pose `35.49982080959101`
(`[macOS-CPU frozen-scorer advisory]`).

The new audit re-derives the four original rows through
`tac.optimization.pure_priced_realized_objective.pure_priced_realized_delta`. It adds no
component gate and changes no weight or acceptance semantics:

| Proposal | d_seg | d_pose | joint delta_S | Disposition |
|---|---:|---:|---:|---|
| `worldsheet_joint_active_x_+1` | 0.06969220479329427 | 35.55635026903992 | +0.00994017010407013 | rejected |
| `worldsheet_joint_active_y_-1` | 0.06986434936523438 | 35.61925856781073 | +0.04382082650052955 | rejected |
| `local_exact_gradient` | 0.07025318569607204 | 35.55364773997821 | +0.06531830924357995 | rejected |
| `worldsheet_joint_active_x_-1` | 0.07033764309353299 | 36.000787594894206 | +0.1919619536286387 | rejected |

## Exact decomposition blockers

The MS4 bundle is genuinely `BUNDLE-COMPLETE`, but completeness is scoped to its declared
metric coordinates:

1. `POSE_RECEIVER_COORDINATE_JACOBIAN_AND_PROPOSAL_FOREIGN_KEY_ABSENT`: the exact Pose
   quadratic lives in PoseNet-output-6 coordinates. No receiver-coordinate Jacobian maps a J10
   proposal into that space.
2. `SEG_RANK4_RECEIVER_COORDINATE_INNER_JACOBIAN_AND_PROPOSAL_FOREIGN_KEY_ABSENT`: the Seg
   block explicitly declares `DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT`. The rank-4 quotient
   is not a proposal-space null projector.
3. `RANGE_A_PROJECTOR_IS_RESIZE_GAUGE_CANONICALIZER_NOT_SEG_NULL_PROJECTOR`: #580 preserves
   `A(PX)=A(X)` and removes resize-invisible energy; it supplies no SegNet-null equation.
4. `PC1_ACTIVE_ZERO_HOME_IS_NOT_SOURCE_PRESERVING_AT_WJOINT_STEP50`: PC2's active-zero home is
   139,547 bytes, d_seg `0.02491527133517795`, d_pose `163.04531226928225`. Relative to J10
   that is +734 bytes, delta_d_seg `-0.04482749938964844`, and delta_d_pose
   `+127.54549145969125`. Its measured ratio `14.023295441931698` is not transferable.

Therefore the required materialization counts are:

| Candidate class | Required | Materialized | Exact n600 priced |
|---|---:|---:|---:|
| single components (`pose-null_seg`, `seg-null_pose`) | 8 | 0 | 0 |
| `pose-null_seg + PC1 pose coordinate` composites | 4 | 0 | 0 |

Every field for archive bytes, d_seg, d_pose, and joint delta is explicitly `null`, never
zero. A numeric estimate would create false authority. Available RAM was independently
verified as 83.498 GiB, above the 20-GiB scorer gate, but no scorer invocation was lawful
because there was no receiver-realizable candidate.

## Landed strict proposal-source guard

The implementation is confined to the proposal-source layer:

- `tac.optimization.ddm_j11_opening_proposal_decomposition` validates all source hashes,
  strict bundle completeness, coordinate-domain declarations, #580's true invariant, the PC2
  home mismatch, and unchanged repricing.
- `DirectDescriptionJointDescent.j11_opening_proposal_decomposition_source(...)` exposes that
  audit without modifying acceptance, optimizer, or launcher semantics.
- `tools/audit_ddm_j11_opening_proposal_decomposition.py` regenerates the typed refusal receipt.
- The typed config binds every external artifact by SHA-256 and byte count.

Receipt:
`.omx/research/ddm_j11_366_opening_proposal_decomposition_refusal_20260725.json`,
SHA-256 `25f092d3499283a77dfcda274015af6826d1f3ce38ac91c26d4a01afa12ad7f4`.

The receipt reports `0/8` singles, `0/4` composites, `n600_scorer_invoked=false`,
`campaign_launched=false`, and
`tools/reseal_ddm_j7_366_ticket.py=NOT_RUN`. A READY/FIRE reseal without an admitted opening
step would be false authority.

## Verification

- Three consecutive focused passes: `74 passed in 93.90s`, `74 passed in 92.98s`, and
  `74 passed in 92.92s`.
- Ruff check and formatting checks pass on the changed/new Python surfaces.
- Fresh review-tracker marks were recorded after each pass: 16 entities in the audit module,
  82 in the proposal-source host, 7 in the tests, and 3 in the CLI.
- JSON parse and `git diff --check` pass.
- No source archive, scorer artifact, checkpoint, or live-run directory was modified.

## Reopener and MAIN landing review

The reopener is exact and intentionally narrow:

1. measure SHA-bound `J_pose` and rank-4-inner `J_seg` for each sealed J10 proposal in the
   exact receiver coordinate system;
2. land a PC1 adapter whose active-zero output equals the J10 source byte-for-byte;
3. project, integer-realize, receiver parse back, and exact-n600 price all 8 singles and 4
   composites under the unchanged rule;
4. only if at least one has `delta_S<0`, run the bounded smoke and reseal through the canonical
   resealer.

MAIN must review the eventual branch range and explicitly disposition it via
`tools/codex_landing_review_gate.py`, especially:

1. the distinction between metric completeness and receiver-coordinate decomposability;
2. the fail-closed null pricing and absence of synthesized zeros;
3. unchanged pure-pricing and proposal-source-only integration;
4. PC1 source-home non-equivalence;
5. the absence of scorer, launch, promotion, or READY/FIRE authority.

## Triality, system intelligence, and stores consulted

- **DSL:** typed config, strict audit module, proposal-source method, CLI, and tests.
- **DAG:** `FEED-603-j11`.
- **Equations:**
  `.omx/research/ddm_j11_opening_proposal_decomposition_canonical_equations_20260725.md`.
- **Sensitivity-map contribution:** the missing source/proposal-bound inner Jacobians are
  explicit required inputs; no surrogate is emitted.
- **Pareto/bit allocator:** no candidate price or value-per-byte exists until receiver-closed
  n600 measurement.
- **Cathedral/autopilot:** the receipt fails closed and emits no dispatch authority.
- **Continual learning:** the premise falsification is durable and scoped away from a
  formulation negative.
- **Probe disambiguator:** the future 8-single/4-composite exact receiver-closed table is the
  arbitration.
- **Stores consulted:** delegated authority; CLAUDE.md; AGENTS.md; craft operating manual;
  top-10 Claude memory; last-24-hour directives; canonical frontier, lane, ownership, probe,
  equation, and cost/posterior stores; J10 ticket/full receipt/proposal rows; MS4
  `BUNDLE-COMPLETE`; #580 implementation; PC2 exact receipt; latest relevant Codex/Claude
  memos.
- **Quarantine waiver:** `HARVEST-SIGNAL-ONLY`; no quarantined bytes, weights, scores, or
  commands were adopted.
- **Inbox/broadcast directives:** no task-window directive was present at checkpoints.
