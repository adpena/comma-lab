# G19 exact-interaction costate control bridge v1 — Codex findings (2026-07-26)

## Verdict

LANDED, implementation-complete, research-only apparatus.  The new pure adapter at
`src/tac/witness_control/taskspace_interaction_costate_bridge_v1.py` converts a frozen G18
feedback receipt plus an independently supplied current-context binding into:

1. exact-base-bound predicted-versus-realized transition calibration;
2. independently calibrated four-cell G8-by-A interaction residuals;
3. content-addressed, duplicate-safe advisory proposal-confidence state;
4. PowerPlay acquisition rows through the existing real control API;
5. `stage` telemetry rows consumable by the existing telemetry parser; and
6. atom/group/shard/global finite-action costate views across the complete factor-coordinate product;
7. a distortion-zero/high-rate full-lattice teacher endpoint with current-receiver replay gating;
8. rate-distortion homotopy plus generic-decoder/counted-payload placement telemetry;
9. one exact whole-object `REQUANTIZE_STORAGE` observation from the lossless xcodec receipt; and
10. typed blocked handoffs for every state-changing API that the advisory evidence cannot lawfully
    enter.

It launches nothing, invokes no scorer, reads/writes no pointer, appends no ledger, creates no
candidate, and claims no score or originality.  The unit did **not** lower an exact score and did
**not** move the frontier pointer.  It is means, not goal progress.

## Scope and ownership

- Lane: `lane_g19_costate_control_bridge_20260726`
- Class: `decision_apparatus`
- `research_only=true`
- Source checkout HEAD observed during landing: `0058123af31779d83d1fc10a728389b0ce7823ec`
- Frozen producer inspected but not edited:
  `src/tac/witness_dsl/taskspace_g8_a3_interaction_feedback.py`
- Frozen G18 spec/test inspected but not edited:
  `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g18_taskspace_g8_a3_interaction_feedback_20260726.md`
  and `src/tac/witness_dsl/tests/test_taskspace_g8_a3_interaction_feedback.py`
- G17, G14/G14a, and G18 owner surfaces were not edited.  The currently materializing G14a run
  directory was read only to confirm that no final receipt was yet available; this unit did not
  signal, launch, stop, or mutate it.
- No commit was made, per the parent task.

## Highest-impact missing bridge that was closed

G18 already preserved all exact n2 rows, exact finite whole-object transitions, and nonlinear
G8-by-A interactions, but its controller payload deliberately stopped at Tier-A observation:
`predicted_delta_adjustment=0`, no posterior update, no dispatch.  Existing costate surfaces then
had two bad options:

- ignore the new exact interaction evidence; or
- force it into an additive per-lever/costate row and erase the four-cell interaction.

G19 closes the read-only middle: compare a proposal's prediction with the exact realized effect,
calibrate confidence/acquisition state, and retain the interaction as one indivisible observation.
It does **not** create a new promotion or actuation path.

## Exact binding contract

`FeedbackExactBaseBindingV1` must match the G18 record byte-for-byte and field-for-field on:

- canonical G19 input feedback SHA-256;
- source and canonical G14 receipt SHA-256;
- source lane ID and the frozen `[macOS-CPU advisory]` axis;
- baseline bundle SHA-256;
- current baseline selected archive SHA-256 and byte count;
- selected-research archive SHA-256;
- frozen scorer SHA-256;
- pointer manifest SHA-256;
- pointer-start SHA-256 and target;
- pointer-latest SHA-256 and target;
- source pair IDs; and
- an external n600 population-manifest SHA-256 plus `population_pair_count=600`.

The adapter recomputes measurement score components, transition component/rate/total deltas,
finite byte ceilings, strict-integer byte ceilings, selected-archive encoding/hash/bytes, selected
minimum, scorer/target-forward custody, six-hook closure, and every G8-by-A interaction from its
four exact cells.  A caller cannot make a mutated G18 field admissible merely by rebinding the
outer feedback hash.

Population honesty is explicit: G18 proves only two source pair IDs.  The caller-provided n600
population hash is retained in every observation fingerprint, but G18 carries no membership proof
and G19 therefore emits `generalization_to_population=false`.  No n2 confidence becomes n600
authority.

## Nonadditive effect semantics

For a transition, the predicted component fields are endpoint deltas.  G19 reconstructs the
predicted endpoint and computes the nonlinear score difference exactly:

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

For an interaction, the realized component effects and score effect are separately reconstructed:

`I_x = x(G8,A) - x(G8,passA) - x(G0,A) + x(G0,passA)`

for `x in {d_seg, d_pose, archive_bytes, S}`.  The prediction must independently supply
`predicted_joint_score_effect`; G19 forbids deriving it by summing component residuals or applying
local score partials.  The transition and interaction remain separate acquisition/control units.

Agreement uses the fully declared symmetric relative form

`agreement(p,r) = 1` when `p=r=0`, otherwise
`max(0, 1 - |p-r|/(|p|+|r|))`.

The proposal's `advisory_confidence` is the arithmetic mean over its content-addressed joint-score
agreements.  It is a calibration score, not a probability, standard error, or authority upgrade.
Repeated ingestion of the identical bound observation is an exact no-op.

## Action semantics beyond ADD

The prediction type carries a closed action vocabulary:

| Action | Required reservoir shape | G19 treatment |
|---|---|---|
| `ADD` | target(s), no source | effect calibration only |
| `DELETE_PRUNE` | source(s), no target | effect calibration only; whole object must be remeasured |
| `MERGE_SHARE` | at least two sources plus target(s) | effect calibration only; never sum source marginals |
| `FACTOR` | source(s) plus at least two different targets | exact effect calibration only; factors do not become additive marginals |
| `REPLACE` | source(s) plus target(s) | effect calibration only |
| `MIGRATE` | different source/target sets | effect calibration only; reservoir movement preserved |
| `REQUANTIZE_STORAGE` | unchanged source/target reservoirs plus equivalence-receipt SHA | whole-object final-ZIP rate effect; never attributed to sections |
| `GAUGE_REPRESENTATIVE` | one unchanged reservoir plus equivalence-receipt SHA | hash preserved, gauge claim blocked |

Closed reservoirs are `base`, `population_code`, `semantic_grammar`, `realization`, `A`,
`learned_residual`, and `entropy_context`.

G18 does not encode operator/action semantics.  G19 therefore labels every action semantic as
declared-only (or declared with an unverified external evidence hash).  Exact outcome agreement
does not prove the claimed mechanism.  Evaluator-equivalent gauge selection remains blocked until
the equivalence receipt itself is content-validated through R and the representative is remeasured
as a whole object.

Historical V19c and the live DDM organ's fixed V19-family/additive rankings are explicitly
`HINT_ONLY_NOT_INGESTED`; only exact-base G18 prediction/realization pairs update G19 confidence.

## Selected solution, placement, and final-archive MDL

The controller record is centered on the selected complete evaluator-measured solution, not on a
verbose inventory of the search problem.  G19 independently retains its archive SHA, exact
selected `archive.zip` bytes, `d_seg`, `d_pose`, and recomputed global score.  The selection
objective is the nonlinear contest expression with **final post-compression archive bytes**.
Raw tensor bytes, member bytes, or the first feasible representation are never selection
objectives.  Factorability is priced through its realized final archive-byte consequence, not a
label-count proxy.

`DecoderPlacementV1` keeps placement explicit:

- generic `solve`, `repair`, `synthesize`, `project`, `factor_expand`, `gauge_select`, `iterate`,
  `entropy_decode`, and `procedural_basis` operations belong in `inflate.py`/`inflate.sh` and add
  zero contest-rate bytes;
- decoder source bytes, measured runtime, and deterministic-portability receipts remain explicit
  non-rate guardrails; and
- every video-specific selector, parameter, atom, latent, exception, or target-dependent branch is
  counted in the final archive.

Placement evidence hashes are preserved but not dereferenced by the pure adapter.  They therefore
cannot establish mechanism, portability, or evaluator equivalence on their own.

## Full-lattice teacher and hierarchical rate-distortion homotopy

`FullLatticeTeacherEndpointV1` represents a distortion-zero/high-rate feasible endpoint oracle.
An historical lattice with no replay has status
`HISTORICAL_HINT_ONLY_NOT_REPLAYED_ON_CURRENT_EXACT_RECEIVER`, contributes no numeric homotopy row,
and has exactly zero controller-confidence influence.  Numeric comparison is enabled only when a
`CurrentExactReceiverReplayV1` matches the current baseline bundle, frozen scorer, pointer,
population manifest, axis, and the same two source-pair IDs, proves zero `d_seg`/`d_pose`, and has a
measured archive larger than the compact base.  Even then it remains n2/macOS advisory.

Each exact G18 transition becomes a homotopy endpoint with before/after global score, distortion,
and final-archive MDL.  A four-cell interaction is explicitly a nonadditive interaction costate,
not a path endpoint.  Each effect is indexed—never summed—at atom, group, shard, and global scope
with factor coordinates for weights, codes, time, population, semantic, realization, pose,
entropy, and analytic-versus-learned placement.

## Exact lossless xcodec representation placement consumed

G19 strictly validates and consumes
`.omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_lossless_xcodec_recode_20260726/receipt.json`
(SHA-256 `02ccb8a6209c79651b64fa93b15aa1ed6155b03d9709f5f18b4ff98edfe25c8c`).
It records one indivisible `REQUANTIZE_STORAGE` whole-object action:

- source archive: 83,838 bytes;
- selected archive: 81,027 bytes;
- exact final-ZIP delta: −2,811 bytes;
- exact rate delta: −0.0018717295172264237 score units; and
- full quantized-state hashes equal, with bounded frozen-receiver uint8 equality.

The source receipt is L2/research-only, `not_a_candidate`, and still owes full-n600 frozen-receiver
replay plus same-byte contest-axis evaluation.  G19 does not update proposal confidence from it and
forbids additive attribution of the −2,811 bytes to weights, codes, entropy, or any other section.

## Existing real consumer APIs and blocked handoffs

Lawful read-only composition:

- `tac.witness_control.control_alphabet.powerplay_acquisition` is invoked on one unique key per
  transition/interaction effect.  Its surprise is predicted-versus-realized joint-score error.
- emitted rows carry `stage=taskspace_interaction_costate_calibration` and round-trip through
  `tac.witness_control.telemetry_binding.parse_log_lines` unchanged.

Typed no-call handoffs are emitted for:

- `tac.master_gradient.append_anchor_locked`: no per-byte tensor/sidecar, n600, runtime call, or
  contest-axis archive custody;
- `tac.witness_control.costate_organ_v3.append_realized_delta_row`: canonical v3 product factors
  are absent and the interaction cannot be collapsed into an additive benefit row;
- `tac.witness_control.costate_posterior.record_costate_observation`: no n600 independent costate
  estimate or standard error;
- `tac.witness_control.continual_costate.append_trajectory_record`: no campaign trajectory,
  architecture backtest, or prototype record;
- `tac.witness_dsl.taskspace_whole_archive_allocator.allocate_taskspace_whole_archive`: G18 has
  hashes/outcomes but no P/G/A/T packet bytes, transforms, or receiver/measurement callbacks;
- `tac.ddm_costate_organ.build_live_ddm_costate`: no typed external non-ADD action-observation API;
  historical V19-family rankings remain hints; and
- evaluator-equivalent gauge selection: no lawful consumer until equivalence bytes and through-R
  proof are validated;
- full-lattice teacher costate ingest: historical-only without replay, or n2 advisory after current
  replay; and
- whole-object placement ingest: xcodec full-n600 replay remains owed.

Each handoff has `write_performed=false` and concrete reactivation evidence.

## Micro-to-macro loop

1. G14 measures exact current-base n2 whole objects and four G8/A cells.
2. Frozen G18 validates/normalizes those rows and preserves transitions/interactions without
   dispatch.
3. A controller proposes one typed action/effect prediction bound to exact baseline archive,
   population, pointer, target, and effect ID.
4. G19 independently revalidates G18 and compares predicted versus realized component, byte, and
   joint-score effects.
5. The content-addressed observation updates advisory confidence and PowerPlay acquisition priority;
   duplicates do not double count.
6. Existing telemetry can display/route the surprise without granting authority.
7. Root review chooses a complete solution.  Any ADD/DELETE/MERGE/FACTOR/REPLACE/MIGRATE/
   REQUANTIZE/gauge bundle re-enters the whole-archive allocator with real packet bytes, receiver
   double-decode, and a new exact measurement against the then-current pointer.
8. The historical full lattice is only a feasible-endpoint teacher; current-receiver replay anchors
   homotopy, while the selected representative is chosen by global score and final archive MDL.
9. Only an n600 byte-closed contest-axis receipt can enter authoritative posterior/anchor/promotion
   stores or move the frontier.

This turns exact n2 interaction evidence into reusable controller memory while preserving the hard
boundary between advisory acquisition intelligence and a real score-moving candidate.

## Verification receipts

- Focused bridge suite:
  `19 passed in 1.47s`
- Full composed suite across G19, frozen G18, whole-archive allocator, costate-v3, master-gradient
  authority filters, and the costate/PowerPlay organ:
  `105 passed in 21.76s`
- Ruff:
  `All checks passed!`
- `python -m py_compile`: clean

Commands:

```text
.venv/bin/pytest -q src/tac/witness_control/tests/test_taskspace_interaction_costate_bridge_v1.py
.venv/bin/pytest -q src/tac/witness_control/tests/test_taskspace_interaction_costate_bridge_v1.py src/tac/witness_dsl/tests/test_taskspace_g8_a3_interaction_feedback.py src/tac/witness_dsl/tests/test_taskspace_whole_archive_allocator.py src/tac/tests/test_costate_organ_v3.py src/tac/tests/test_master_gradient_authoritative_axis_filter.py src/tac/tests/test_lambda_net_costate_organ.py
.venv/bin/ruff check src/tac/witness_control/taskspace_interaction_costate_bridge_v1.py src/tac/witness_control/tests/test_taskspace_interaction_costate_bridge_v1.py
.venv/bin/python -m py_compile src/tac/witness_control/taskspace_interaction_costate_bridge_v1.py src/tac/witness_control/tests/test_taskspace_interaction_costate_bridge_v1.py
```

## Pointer and goal honesty

At the final read-only check, `.omx/state/canonical_frontier_pointer.json` had SHA-256
`940491eaa6cd81fc66da5f93a497104e83426f306fd25a0d6c8f43bf2a311851` and reported:

- effective competitive target: official-leaderboard `0.172` (external custody);
- our local contest-CPU anchor: `0.1880443979880752`; and
- our local contest-CUDA anchor: `0.20533002902019143`.

G19 changed none of these.  It produced no new exact contest row, the pointer is unmoved, and the
recursive mission `score < live effective_frontier` remains unsatisfied.  The numeric pointer
values above are provenance-only observations and are not controller literals.

## Originality, payload, storage, and resumability

- New code/tests/memo are local original apparatus; no public archive payload, weights, latents,
  selector tables, or learned content were reused.
- The adapter intentionally emits `originality_claim=false`: apparatus authorship is not candidate
  payload originality.
- No large artifacts were created.  No cleanup/cold-store action was needed.
- No job or training run was launched, so resume/per-stage checkpoint requirements were not
  activated.  Controller state itself is deterministic, JSON-safe, content-addressed, and can be
  passed back as `prior_controller_state` without double counting.

## Artifact hashes

- `src/tac/witness_control/taskspace_interaction_costate_bridge_v1.py`:
  `52f3a8f54fe2ac9c2f56e415f337e8d5d83db52645b1a621448879b2aab8c7a6`
- `src/tac/witness_control/tests/test_taskspace_interaction_costate_bridge_v1.py`:
  `4e94c1ae33124108bee5926d0e2bd294c620464457a24ea6e591795c3d21141c`
