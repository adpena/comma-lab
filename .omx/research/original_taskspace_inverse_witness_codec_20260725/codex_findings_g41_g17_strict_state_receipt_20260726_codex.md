# G41 findings — strict reopenable G17 whole-object state receipt

Date: 2026-07-26  
Lane: `lane_g41_g17_strict_state_receipt_20260726`  
Scope: smallest rigorous G17 evidence receipt required by G36  
Authority: `research_only=true`; no public-RGB bridge, candidate, evaluator run, score claim, dispatch, or pointer mutation

## Outcome

`taskspace_selected_solution_compiler.py` now exports:

- `G17WholeObjectStateReceiptV1`;
- `build_g17_whole_object_state_receipt(state)`; and
- `parse_g17_whole_object_state_receipt(exact_bytes)`.

The builder accepts only an exact internally coherent
`G17WholeObjectStateV1` carrying research-advisory authority. The parser
reopens exact canonical bytes and independently reconstructs the physical ZIP
member map, typed compiler-placement manifest, logical ownership objects,
population coordinates, obligation coverage, pose ownership, optional R10
coordinates, decoder continuation basis, private decoded chronology,
candidate-forward receipt, score sufficient statistics, research authority,
proof-dependency sets, competitive target, and all composed identities.

The receipt retains exact archive/member/decoded/receiver/program/runtime/
authority/proof bytes as evidence. It does not put any of those bytes in a
candidate packet. `candidate_payload_allowed=false`,
`private_intermediate_only=true`, and `public_rgb_bridge_proven=false` are
literal fail-closed receipt fields. G17 chronology therefore cannot be
laundered into G29 public raw RGB; G36 still requires the separate typed raster
bridge.

## Exact arithmetic and anti-forgery closure

Observation aggregates are recomputed from typed sufficient statistics:
integer Seg mismatch/element counts and exact binary64 Pose squared-error sum/
element count. Score terms then recompute through `tac.contest_score` in pinned
upstream order:

`100*d_seg + sqrt(10*d_pose) + 25*(archive_nbytes/37_545_489)`.

The adversarial fixture deliberately selects a real ZIP length where the old
`(25*bytes)/N` expression differs by one ULP. A receipt resealed with that old
rate and total is refused. Canonically resealed archive-length, score, axis,
sufficient-statistic, or identity drift is also refused. Duplicate keys,
noncanonical JSON/base64/float encodings, bool-as-int drift, a mutated frozen
score receipt, and an `object.__new__` forged whole-state graph cannot pass.

No caller-provided hash establishes state. The strict state identity composes
independently recomputed archive representation, placement, ownership,
continuation basis, private decode, observation, authority, proof-dependency,
score, and competitive-target identities. Consumers must parse exact receipt
bytes; the Python instance alone is not an authority capability.

## Deliberate boundary

V1 supports `G17ResearchAuthorityEvidenceV1` only. Contest-CPU/CUDA receipt
construction fails closed until the already-owed governed exact-evaluator
adapter supplies strict public execution bytes. This is sufficient for G36's
current research-advisory state, and prevents a label from upgrading authority.

The continuation identity is a **basis identity**, not G38 action-space
closure: it preserves the exact population/ownership/placement/decoder state
that a subsequent `G17_ACTUATOR_IR_V1` must consume. It does not claim terminal
reachability, generator-leaf coverage, or a verified cost-to-go certificate.

## Validation

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/tac/witness_dsl/tests/test_taskspace_selected_solution_compiler.py
21 passed

.venv/bin/ruff check \
  src/tac/witness_dsl/taskspace_selected_solution_compiler.py \
  src/tac/witness_dsl/tests/test_taskspace_selected_solution_compiler.py
clean

.venv/bin/ruff format --check <same files>
clean
```

Repository search found no retained non-test construction of
`G17WholeObjectStateV1`; therefore no real low-distortion G17 state currently
exists to materialize through this API. MS1 remains an existence teacher, not
a current G17 represented state. G41 created no receipt artifact pretending
otherwise.

## Actuator-IR compatibility constraints

The frontier-critical next implementation must preserve:

1. the exact `G17PairPopulationV1` coordinate mapping;
2. one typed logical owner and one physical charged span for every counted
   actuator operand;
3. real receiver operation IDs and chronology dependencies in the continuation
   projection;
4. exact private decoded chronology as distinct from public RGB;
5. candidate-forward receipt and integer/binary64 sufficient statistics; and
6. research authority until the independent governed public adapter closes.

Adding an actuator operand without extending the existing placement,
obligation/pose/R10 continuation semantics must fail receipt construction; a
second parallel ontology would orphan the selected-solution signal.

## Triality and wire-in

- **DSL:** G17 receipt bytes are the durable strict form of the existing
  selected-solution state types.
- **DAG:** counted archive -> private receive -> frozen observation -> exact
  score -> strict state receipt -> G17/G29 raster bridge -> G36/G33.
- **Equation:** the receipt binds the complete coupled contest action and never
  introduces independent Seg/Pose/rate gates.

Sensitivity and bit-allocation consumers can join logical owner, charged span,
population support, before/after state, and exact score; Pareto admission stays
whole-object and nonlinear; cathedral/autopilot may schedule only a real
actuator endpoint; continual learning consumes parsed receipts, not hashes;
G17-private versus G29-public remains the explicit probe-disambiguator.

## Stores consulted

- byte-identical `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, and the arbitrage
  shared-worktree discipline;
- current frontier pointer (`0.172` external target), lane registry, subagent
  ownership, and top current project memory;
- G17 source/spec/tests and strict candidate-forward receipt parser;
- G29/G31 public closure findings, G36 exact-adapter spec, and G38 strict
  proof-byte findings; and
- G39 macro-crux finding selecting teacher-to-actuator IR as the next
  frontier-critical layer.

## Pointer honesty

Pointer moved: **no**.  
New candidate: **none**.  
Exact score row: **none**.  
Dispatch/evaluator run: **none**.

G41 is custody infrastructure. The forest-level next action remains the real
G17 teacher-to-existing-actuator IR that creates a different n600 state.
