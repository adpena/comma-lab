# G45 — ep725 V2 label-local G pairwise stream

Date: 2026-07-26  
Lane: `lane_g45_ep725_label_local_g_stream_20260726`  
Scope: production-shaped receiver seam; bounded mechanics only; `research_only=true`

## Objective

Execute exact counted per-pair G pages against the frozen ep725/LVLS1 predictor without
fabricating Pose6 or transport, without widening any bounded proof cap, and without retaining
an n600 realization in Python memory.  The executor must be valid for any ordered contiguous
population of at most 600 pairs and must write one chronological path-backed raw.

## Physical state transition

For each pair `p`, the exact counted P member and exact runtime produce camera `(Y0, Y1)` and
the frame-1 `phi.argmax` from the same `_render_pair` calculation.  G45 constructs
`TaskspacePredictorStateV2(..., source_pair_ids=(p,), labels=phi[None],
transport=NoTransportV2())`, parses the exact counted G page, calls
`require_g_transport_admission`, applies it through
`apply_generative_taskspace_correction_v2`, and realizes only the changed semantic support via
the unchanged `overlay_g_on_predictor_camera_y1` donor.  Y0 and every unowned Y1 camera value
must remain byte-identical.

## Runtime observation contract

The frozen runtime is loaded from descriptor-stable source bytes.  Its `_setup` and
`_render_pair` functions remain unmodified.  During one `_render_pair(p)` call, G45 temporarily
wraps `_outputs_from_h0` and captures exactly one `want_rgb=True` result whose `code_row` shares
memory with `code[2*p+1]`.  This is the actual frame-1 phi used by the runtime, not a second
renderer or proxy.  The wrapper is restored even on failure.  Runtime source, archive wrapper,
sole `0.bin` member, renderer identity, dimensions, population, fp64 mode, and page bytes are
all hash-bound and rechecked.

## Durability and scale

- Output is a preallocated `.partial` raw under the configured SSD tier, with explicit
  test-only local opt-in.
- One pair is resident at a time.  Every completed pair has an immutable canonical JSON
  checkpoint binding source/runtime/P/G/state/overlay/input/output hashes and exact range.
- G page references are retained, but page payloads are reopened, hash-verified, consumed, and
  released serially; the executor never retains the n600 G payload population in Python.
- A crash before a checkpoint causes that pair to be regenerated.  Resume rehashes every
  committed range and refuses gaps, overlaps, chronology drift, or source/page drift.
- Only after every pair checkpoint reopens does G45 hash the full raw and atomically promote
  `.partial` to the final output path.  A content-addressed pre-promotion execution receipt is
  fsynced before rename, so a crash between rename and the canonical receipt is verified and
  completed on the next invocation rather than becoming an ambiguous orphan.
- The module performs storage preflight but does not launch n600, run a scorer, build a
  candidate, or mutate the frontier pointer.

## Refusals

- V1/Pose6 projection or fabricated Pose6/SE3 transport.
- Transport-dependent event/island/worldsheet atoms under `NoTransportV2`.
- A G page whose pair window or predictor binding differs from the live per-pair state.
- Runtime frame-1 phi capture cardinality other than exactly one.
- Any Y0 or unowned Y1 change.
- Non-contiguous population, output/checkpoint gaps, path/hash/size drift, symlinks, or
  non-atomic artifact replacement.
- Any score, public-runtime, candidate, or authority claim.

## Acceptance

```bash
.venv/bin/python -m pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_ep725_label_local_g_stream.py
.venv/bin/ruff check \
  src/tac/witness_dsl/taskspace_ep725_label_local_g_stream.py \
  src/tac/witness_dsl/tests/test_taskspace_ep725_label_local_g_stream.py
```

Tests must exercise real donor behavior on full camera/scorer geometry, deterministic
path-backed output, interruption/resume with committed-range rehashing, transport refusal before
G mutation, source/page drift refusal, post-rename receipt recovery, and exact preservation of
Y0/unowned Y1.  The fixture is mechanical proof only and is never an n600 or score finding.

## Bounded G17 parity anchor

`taskspace_g17_actuator_ir_v1.py` is the closed n1/n2, in-memory, deterministic-double-replay
anchor for the same V2 parse/apply and predictor-preserving overlay donors.  G45 does not route
n600 through that bounded wrapper because it intentionally retains the full bounded chronology
and consumes packed G17 member spans.  Instead, G45 extends the identical typed donor physics to
path-backed chronological execution while preserving G17 as an independent bounded parity
surface.  A future selected-solution packer should translate its exact counted G17 spans into the
ordered `GPageRefV1` stream without changing either receiver's semantic physics.
