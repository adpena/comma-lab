# G3 label-local transport-dependency amendment

## Superseding verdict

This append-only amendment narrows one sentence in
`SPEC_g3_ep725_predictor_transport_seam_20260726.md`.  Topology events and
island shapes are not Pose6-dependent merely because they have a lifetime.
Their receiver coordinates depend on Pose6 if and only if at least one counted
transport gain is nonzero:

`requires_pose6(a) := (a.transport_gain_x_q4 != 0) or (a.transport_gain_y_q4 != 0)`.

For zero-gain event/island atoms, the displacement terms are identically zero
for every possible Pose6 trajectory.  Reading a transport array in that branch
was therefore a nominal-family dependency rather than a behavioral dependency.
It falsely refused an exact label-local residual program from ep725's truthful
`NoTransportV2` state.

## Exact receiver contract

One shared predicate, owned beside the finite event/island primitive types,
drives both V2 admission and rasterization.

- Zero-gain topology events and island shapes, for lifetime one or greater,
  are `LABEL_LOCAL`.  Their receiver must not access a Pose6 property or array.
- A nonzero x or y gain is `POSE6_ADVECTED`.  Passing no Pose6 array fails
  closed before coordinate evaluation.
- Worldsheets and knots remain conservatively `POSE6_ADVECTED`, regardless of
  their current parameter values.  Proving a label-local worldsheet subset is
  a separate formulation and is not admitted by this amendment.
- `SE3_XI` remains refused for every Pose6-advected family until an explicit
  parity adapter exists.
- Existing V9 Pose6 behavior and packet bytes do not change.

The bounded target-debt control gains a V2-specific entrypoint.  It accepts an
exact `TaskspacePredictorStateV2`, constructs only zero-gain row-span events,
and calls `compile_generative_taskspace_correction_v2`.  It never calls
`as_v1()` and never fabricates an all-zero Pose6 table.

## Compatibility and falsifiers

Golden V1 compatibility is pinned by the current representative packet:

- packet bytes: 246;
- packet SHA-256:
  `d146e5a19cba16f7ab2feff8661bc3192043fb953ef211c56a6edc2f0f17c935`;
- decoded-label SHA-256:
  `143cb722fe74e21ecd9e9f6468ce99855b82f566dd56cb9d2fb6a94bdadbc829`.

The one-pair event vector remains 141 bytes with SHA-256
`1fd833ee310800688c76a3bb7c1af8ea287d8d02fb1997c9d7f0e23c3f735230`.

Acceptance requires all of the following:

1. zero-gain lifetime-one and lifetime-two events and islands compile and
   double-decode under `NoTransportV2`;
2. their decoded labels are identical under distinct Pose6 trajectories;
3. nonzero-gain events and islands refuse `NONE` before receiver mutation;
4. direct mask calls refuse nonzero-gain plus absent Pose6;
5. worldsheets remain refused under `NONE`;
6. the golden V1 byte and label hashes remain exact.

Any divergence between the shared dependency predicate and receiver behavior
is a strict compatibility failure.

## Real bounded structural gate

The exact ep725 `NoTransportV2` prefix and frozen `gt_n600.npz::lstars[0:2]`
were reopened through their existing custody readers.  This is an n2
structural compatibility gate, never a score or n600 empirical verdict.

- predictor binding SHA-256:
  `a25ee50104fc887ba5ee5e92110caac736ee431338584a3ad6ac8b54ab7cfae9`;
- frozen target-label SHA-256:
  `6a9ee68a5d1ec8ec53653216d53b7406575530a2d1abf608d27547e779c6d474`;
- exact predictor-target debt: 60,217 semantic cells;
- zero-gain row-span events: 21,323 (12,259 births and 9,064 deaths);
- counted inner G packet control (not an outer archive): 341,316 bytes,
  SHA-256
  `9139f2a7744cc56f822f13dd0155c0e276e474d0c56481772cc089ac5835e0c1`;
- two fresh V2 decodes were label-identical and exactly reconstructed the
  frozen target slice;
- transport kind remained `NONE`; no scorer was invoked and candidate
  eligibility remained false.

The canonical parse-back receipt is
`ep725_n2_bounded_target_g_v2_receipt.json`.  The JSON file, including its
single POSIX terminal newline, has SHA-256
`854ffaf3d5102959edaab48f38fa574d4763f289b7b35026bcb9de7d9d1e916f`;
the canonical receipt bytes before that newline have SHA-256
`eb9ff88cc34ceba87c82e90c003077b50f4e5b84950823b53d93d6840b0bec8e`.
It is regenerated and compared field- and byte-exactly by:

`.venv/bin/pytest -q src/tac/witness_dsl/tests/test_bounded_target_g_encoder.py::test_real_ep725_n2_v2_control_exactly_reconstructs_frozen_target_without_transport`

No outer packet or ZIP was materialized in this gate, so 341,316 must never be
reported as an archive size or scored rate term.  Outer-container bundle
pricing remains a separate monolithic-composition measurement.

## Authority and remaining scope

This is a structural receiver correction and an exact semantic control, not a
score result.  It invokes no scorer, proves no through-R realization, carries
no rate-optimality claim, creates no candidate archive, and does not move the
canonical frontier pointer.  The exact row-span control remains
`research_only=true` and acquisition-lineage superseded; target labels stay
encoder-only.  Real n600 inverse selection, same-class realization repair,
standalone monolithic materialization, contest CPU/CUDA evaluation, and pointer
movement remain owed.

## Triality delta

- DSL: gain-derived `requires_pose6_transport` plus the exact V2 bounded-G
  entrypoint.
- DAG: `P labels + zero-gain G -> label-local decode`; the Pose6 edge exists
  only when a counted gain can affect coordinates.
- Equation: `dx = round(delta_pose[0] * gain_x / 16)` and
  `dy = round(delta_pose[1] * gain_y / 16)` collapse identically to zero when
  both gains are zero, so `delta_pose` is not an input to that branch.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, and `PROGRAM.md`;
- `SPEC_g3_ep725_predictor_transport_seam_20260726.md`;
- the finite event/island definitions, mask receivers, G program receiver, V2
  predictor state, V2 consumer seam, and bounded target-G encoder;
- canonical lane registry, subagent progress ledger, and current top-ten
  project memory anchors.

HISTORICAL_PROVENANCE: append-only correction of the nominal atom-family
transport overclassification; the original specification remains immutable.
