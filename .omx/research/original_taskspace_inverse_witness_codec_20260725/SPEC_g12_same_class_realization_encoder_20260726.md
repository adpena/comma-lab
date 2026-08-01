# G12 encoder-side same-class realization acquisition

Status: encoder API frozen 2026-07-26; `research_only=true`; lane L1
(`impl_complete` only).

Lane: `lane_g12_g8_realization_encoder_20260726`.

STORES CONSULTED: `CLAUDE.md` / `AGENTS.md` SHA-256
`47d4ac3a38f91a8b8e7dc3061131717d8122bd48ffb204ffb914eb58e687f0c9`;
`PROGRAM.md`; top Claude `MEMORY.md`; frozen G8 specification and receiver;
latest G1 exact-output-lineage findings/session summary; v7.5 operating contract;
v8 merge-diff-correct design; live lane registry and subagent-progress ownership;
the n2 frozen-scorer stage-ablation advisory receipt SHA-256
`2af9b50d70f342224aa438e95b4d53a05be3f253709c1fa1835da089f37e0f61`.

## Objective and exact boundary

G8 supplies a counted receiver coordinate but deliberately leaves selection to
an encoder-side inverse problem.  This unit supplies that missing acquisition
layer.  It consumes only typed, hashed encoder evidence and emits concrete
`SameClassRealizationRepairProgramV1` proposals for downstream G8 compilation
and G7 whole-object arbitration.

The exact acquisition debt is

`D = {i : current_semantic[i] == target_label[i] and realized_label[i] != target_label[i]}`.

For the exact semantic-G control, `current_semantic` is the decoded counted-G
label field.  For the PASS/predictor interpretation, it is the independently
hashed predictor-semantic topology.  Calling both fields `semantic_G` would
silently conflate the two bases, so the API does not do that.

Every emitted run cell must belong to `D`.  A current semantic error is not a
same-class realization error and must never enter this coordinate.  Dense
labels, scorer observations, and scorer-plane RGB arrays remain encoder-only;
only selected run coordinates and selected uint8 RGB triples can later enter a
counted G8 payload.

The acquisition also preserves the exhaustive four-way `(Z,T,H)` partition as
typed dense-free telemetry, where `Z=current_semantic`, `T=target`, and
`H=realized`: closed `(Z=T,H=T)`, realization debt `(Z=T,H!=T)`, topology debt
`(Z!=T,H!=T)`, and fortunate semantic mismatch `(Z!=T,H=T)`.  Each row carries
its exact boolean-mask SHA-256 plus global and per-target-class counts.  G8
acquisition is structurally row 2; the other three rows remain visible to the
coupled runner instead of becoming orphaned signal.

This is not a scorer, optimizer, candidate, or score.  It does not invoke the
frozen scorer, re-run through R, establish class realization, choose a winning
proposal, or move the frontier.  Every proposal remains `research_only=true`
until the parent composes the complete P/G/A object, replays its receiver, and
measures the nonlinear whole-object Seg/Pose/rate score.

## Typed input custody

`EncoderOnlySameClassRealizationEvidenceV1` binds:

- exact contiguous source-pair IDs;
- one explicit base interpretation: `EXACT_SEMANTIC_G_CONTROL_V1` or
  `PASS_PREDICTOR_SEMANTIC_TOPOLOGY_V1`;
- current base semantic labels, their semantic-binding foreign key, exact
  counted-P identity, and exact base camera-Y1 identity;
- a real counted-G identity for the exact-G branch, while the PASS branch
  requires `base_g_section_sha256=None` and rejects fake empty-G hashes;
- exact target labels;
- exact realized frozen-scorer labels;
- exact candidate and target scorer-plane uint8 RGB;
- a frozen-scorer identity plus candidate/target forward-receipt identities;
- the exact external `[macOS-CPU frozen-scorer advisory]` stage-ablation
  receipt SHA-256 `2af9b50d70f342224aa438e95b4d53a05be3f253709c1fa1835da089f37e0f61`,
  whose only authority is acquisition ordering, never score or closure;
- an independently supplied SHA-256 for every dense array, checked against its
  bytes before acquisition.

`SameClassRealizationCellPartitionTelemetryV1` is derived only after those
content hashes pass.  It serializes the three input hashes, four mask hashes,
and closing counts, never a dense mask.

The type has no serialization method.  Its binding digest contains hashes,
shapes, source IDs, and custody foreign keys, never dense evidence bytes.

Both interpretations run the same exact debt acquisition.  The frozen G8
compiler currently has a callable base path only for the branch with a real
semantic G section; this is a capability statement, not evidence that this
encoder invoked the compiler.  No proposal receipt claims compilation.
PASS proposals therefore carry the precise seam
`PASS_BASE_REQUIRES_NONEMPTY_COUNTED_G8_BASE_ENVELOPE_V1`; they are concrete
program evidence for the missing receiver branch, not falsely compiled G8
packets.  Once G10 supplies a real nonempty PASS-G receiver envelope, the same
program objects are directly consumable by the frozen G8 compiler.  The
branches never share a source binding and no empty G is invented.

## Proposal families and arbitration coordinates

No single palette size, cell threshold, or ranking interpretation is declared
universally optimal.  A typed plan supplies a sorted set of palette sizes, a
geometric prefix ratio, and both defensible prefix orders.  Acquisition emits:

1. `CLASS_SHARED_TARGET_MEDOID_V1`: one exact observed target-RGB medoid per
   semantic class;
2. `CLASS_BOUNDED_TARGET_MEDOIDS_V1`: a caller-enumerated bounded class-local
   palette, initialized deterministically and refined to observed RGB medoids;
3. `TARGET_PIXEL_RGB_ORACLE_CONTROL_V1`: the exact target scorer-plane RGB at
   every selected debt cell, a counted upper control rather than a ship claim.

Prefixes are the exact geometric integer sequence generated from one cell to
the complete debt set, with the complete-debt endpoint always included.  Both
canonical address order and descending exact candidate-to-target RGB numerator
SSE order are emitted.  G7, not this encoder, arbitrates the resulting complete
archive objects.

The proposal tuple is fidelity-first because the bound n2 advisory ablation
measured Y1 realization as the dominant current lever: full exact-target oracle
controls first, then bounded target-medoid approximations (larger/full prefixes
before smaller ones), then the class-shared compression control.  This ordering
only schedules measurements; it does not admit a proposal or transfer the n2
numbers to n600.

The complete proposal tuple retains every family/order interpretation.  Since
different interpretations can generate byte-identical G8 programs (especially
at the full prefix), `unique_program_proposals` returns the first
fidelity-ordered representative per program SHA-256.  G7/G10 should measure
that view so no duplicate exact-byte row is spent; receipts for the aliases
remain available for provenance.

Each selected cell is converted to a G8 run and adjacent horizontal cells are
merged if and only if pair, row, semantic class/role, and RGB agree.  The result
is constructed through `SameClassRealizationRepairProgramV1`, so overlap,
noncanonical order, and split-identical runs fail closed.

## Receipts and no-orphan handoff

Every proposal carries a canonical receipt with:

- input binding and all five dense content hashes;
- the complete four-way cell-partition telemetry, with proposal debt required
  to equal the realization-debt row globally and per target class;
- the explicit base interpretation, P/G/Y1 foreign keys, and exact frozen-API
  base-path/seam truth without claiming a compile invocation;
- the advisory ordering receipt SHA/axis/scope, explicitly non-authoritative;
- family, palette bound, prefix order, and requested/selected prefix size;
- exact total/covered/omitted debt counts globally and per class;
- program/run/cell counts and a canonical program digest;
- exact standalone G8 raw-wire estimate from the frozen G8 header/row ABI;
- exact selected target-RGB SSE before/after under the proposal prototype;
- explicit truth labels: no G8 compile/packet, scorer, allocator, eval, score,
  candidate, or promotion claim; through-R scorer admission still required;
  RGB SSE is encoder-coordinate evidence only; dense evidence serialized
  false; public/archive payload reused false; and selected video-derived
  RGB/coordinates counted if compiled true.

The acquisition result exposes concrete G8 programs and a byte-deduplicated
measurement view.  The downstream sequence is fixed: validate the proposal
against the same evidence, compile it through G8 using the exact P/G surface
and target-label custody, substitute the composite G section, condition A on
post-G8 Y1, and let G7 rebuild/replay/measure the whole archive.

## Canonical-vs-unique decision per layer

- Reuse the frozen G8 program/run/role ABI because it is the exact downstream
  receiver contract and prevents a second payload grammar.
- Keep acquisition in a new method-specific module because target/scorer
  evidence is encoder-only and must not leak into the frozen receiver module.
- Use exact integer RGB SSE and deterministic observed-colour medoids rather
  than a generic ML clustering dependency; this preserves bit determinism and
  n600 window resumability without inventing a proxy-score claim.
- Delegate complete-object admission to the existing G7 allocator; duplicating
  a component threshold here would recreate the independent-gate bug.

## Acceptance and falsifiers

Owned files only:

- `src/tac/witness_dsl/taskspace_same_class_realization_encoder.py`;
- `src/tac/witness_dsl/tests/test_taskspace_same_class_realization_encoder.py`;
- this specification.

Focused command:

`.venv/bin/pytest -q src/tac/witness_dsl/tests/test_taskspace_same_class_realization_encoder.py`

Static commands:

`.venv/bin/ruff check src/tac/witness_dsl/taskspace_same_class_realization_encoder.py src/tac/witness_dsl/tests/test_taskspace_same_class_realization_encoder.py`

`.venv/bin/ruff format --check src/tac/witness_dsl/taskspace_same_class_realization_encoder.py src/tac/witness_dsl/tests/test_taskspace_same_class_realization_encoder.py`

Required falsifiers cover: dense-content hash/custody mismatch; source-pair
mismatch; wrong-class admission; omission accounting for every prefix; exact
horizontal run canonicalization; deterministic proposal/receipt replay;
distinct bounded palette behavior on nondegenerate evidence; and the target
pixel oracle setting every selected cell to its exact target RGB.  The fixture
also exercises all four partition rows and verifies that row 2 is exactly the
encoder debt mask.  Synthetic fixtures prove code behavior only and carry no
empirical/score verdict.

All three generated families remain unadmitted until actual receiver-closed,
through-R frozen-scorer replay.  The exact-pixel oracle depends directly on
target scorer-plane RGB; the medoid families depend on statistics of that same
target RGB.  A zero encoder-coordinate RGB SSE is therefore not internal-label
closure and not scorer admission.  No family in this unit may be promoted on
its internal RGB numerator alone.

## Triality and pointer honesty

DSL leg: a typed encoder acquisition surface feeding the existing counted G8
receiver grammar.  DAG leg: G12 closes the missing `realized labels/RGB -> G8
program proposals -> G7 arbitration` edge.  Equation leg: the exact debt set
above plus conditional full-score admission owned downstream by G7.  No new
empirical law is registered.

Pointer delta from this unit is necessarily zero: it produces means, not an
authoritative n600 contest row.

## Frozen implementation receipt

- Encoder module SHA-256:
  `842035c29647f26aa620a1c19a26943bd8a307702563a283215afffcf234285a`.
- Focused test module SHA-256:
  `fab720b134fe950b19608b9c491096d64fbe4a6df086d5f2e5cc4ed8550d4489`.
- Consumed frozen G8 module SHA-256:
  `4e892c4aa0fab1dce65df54da96181511ecaff6dc7a84fff9876d8bd26060693`.
- Bound n2 advisory ordering receipt SHA-256:
  `2af9b50d70f342224aa438e95b4d53a05be3f253709c1fa1835da089f37e0f61`.
- Verification: focused encoder plus frozen G8 tests `27 passed in 2.44s`;
  Ruff check/format and Python byte compilation clean.  `ty` reports no
  owned-file diagnostic; its only diagnostic is the pre-existing unknown
  `possibly-unbound` rule in `pyproject.toml:305`.
