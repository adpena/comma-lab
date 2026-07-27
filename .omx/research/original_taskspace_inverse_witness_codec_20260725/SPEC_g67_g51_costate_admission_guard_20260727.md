# G67 specification — G51 costate admission guard

Date: 2026-07-27  
Lane: `lane_g67_g51_costate_admission_guard_20260727`  
Status: additive fail-closed implementation; no score, candidate, promotion,
dispatch, or pointer claim

## Purpose

G51 is a full-n600 encoder diagnostic. Its exact chunk-block measurements are
useful for representation design, but they are not scorer effects and are not
same-object archive ZIP marginals. Historical v1 names could invite unsafe
reinterpretation:

- `zlib9_marginal_bytes` was an independently reset pair stream;
- `per_pair_marginals` was encoder bookkeeping;
- `fits_*` and a "best" row compared chunk sums with planning headroom;
- declarative `READY_*` hook values did not wire a consumer; and
- the diagonal of final output residual energy was not a functional/scorer
  Gram.

G67 makes that interpretation executable. It does not edit G51, G64, G65, or
G66.

## Exact G51 boundary

`load_g51_encoder_diagnostic` accepts only the adversarial interpretation
receipt committed by `b84b4c6d948f24f3aa399c1774f557dcecfa3658`:

- schema:
  `tac.taskspace_conditional_quotient_profile_adversarial_interpretation.v1`;
- file SHA-256:
  `756dd421dd8b6ad21dd6d8b7ed271bdbac1ac84b036ab92bf36b40686a998a13`;
- population: 600 pairs; and
- all corrected authority fields remain false.

The return type retains only the receipt identity, source commit, population,
diagnostic status, and explicit blockers. It does not retain or expose any G51
numeric payload. Its only allowed use is `ENCODER_DIAGNOSTIC_ONLY`; its
actionable consumer set is empty.

## Integrity validation is not actionable admission

`inspect_g51_costate_evidence_candidate` requires two separate canonical
receipts and reopens their artifacts. Passing these checks establishes only
internal file integrity. It does not prove that the files were emitted by the
claimed receiver/`R`/scorer transition.

`admit_g51_actionable_costate_evidence` deliberately calls that inspector and
then refuses with
`G51_CANONICAL_RECEIVER_R_SCORER_EFFECT_MATERIALIZER_OWED`. The repository has
no canonical materializer that computes and seals all three effect surfaces
from one exact transition, so there is no positive production admission path.

### Scorer-effect evidence

Schema: `tac.taskspace_g51_untrusted_scorer_effect_candidate.v1`.

The receipt must bind one NPZ containing exactly:

1. ordered `int32` pair IDs `0..599`;
2. full-n600 `float32` scorer-term effect vectors with axes
   `[seg_score_term_delta, pose_score_term_delta]`;
3. full-n600 realized scorer JVP contractions on those axes; and
4. full-n600 realized scorer VJP contractions on those axes.

All arrays must be finite, shape `(600, 2)` for vectors, and nonzero. The
candidate receipt records claims about the public receiver,
realized-through-`R`, frozen CPU-torch SegNet and PoseNet, NumPy-fp32
reference, and false MPS/proxy authority. Those booleans are explicitly
self-attested claims, not proof. The inspector reopens the NPZ, recomputes
pair coverage, and retains only array-content SHA-256 identities; it exposes
no effect magnitudes to consumers.

### Same-object ZIP evidence

Schema: `tac.taskspace_g51_same_object_archive_zip_delta.v1`.

The receipt must bind:

- different baseline and candidate ZIP SHA-256 identities;
- one object ID and transition ID shared with the effect receipt;
- the same population identity, public receiver runtime, and evaluator source;
- baseline/candidate receiver-output SHA-256 identities;
- the exact effect-bundle SHA-256; and
- a measured outer-ZIP byte delta.

The guard reopens both regular non-symlink ZIP files, validates their member
streams, recomputes bytes/SHA-256 and the ZIP delta, and calls the canonical
`build_serialized_archive_delta_contract`. Chunk-block sums, pair resets,
static parameter counts, and claimed deltas cannot satisfy this gate.
However, arbitrary files named `inflate.py` and `evaluate.py` can still be
self-bound consistently; integrity does not establish canonical public-eval
lineage or causally relate the JVP/VJP arrays to the ZIP transition. That
provenance gap is exactly why action remains blocked.

## Historical-field extinction

Every evidence candidate is recursively scanned before parsing. The guard
rejects:

- exact legacy v1 fields and payload references;
- every `fits_*` key;
- "best archive/basis" keys;
- old ambient/functional Gram names; and
- any `status` value beginning with `READY`.

The returned `G51CostateEvidenceCandidateV1` retains only artifact identities,
array-content hashes, the recomputed ZIP delta contract, and false-authority
markers. It has no scorer-effect values, an empty actionable consumer set, and
an explicit blocked-consumer set. No raw G51 mapping or candidate array value
crosses the boundary.

## Real fail-closed integration point

`request_g51_costate_bit_allocation` is the production guard call:

1. it calls the strict admission guard;
2. the guard validates the candidate artifacts; and
3. it refuses before any sensitivity, allocator, costate, autopilot, or Pareto
   consumer call.

There is intentionally no import or invocation of the canonical allocator in
the G67 module. A regression monkeypatches that allocator and supplies
arbitrary fixture runtime/evaluator files plus nonzero JVP/VJP arrays that are
unrelated to the effect transition; the candidate remains diagnostic and the
allocator is not called. This closes the false-actionability path rather than
papering it over with stronger self-attestation.

## Unified consumer hooks

| Hook | G67 status |
|---|---|
| Sensitivity map | explicitly blocked pending canonical materializer |
| Pareto constraint | explicitly blocked pending canonical materializer |
| Bit allocator | real refusal guard wired; allocator is not invoked |
| Cathedral autopilot | explicitly blocked pending canonical materializer |
| Continual/costate controller and organ | explicitly blocked pending canonical materializer |
| Probe disambiguator | strict legacy-field and artifact-tamper tests active |

No hook is declared READY from G51, and no public actionable-admission
dataclass exists. Future work must first land a canonical materializer and a
new adversarial review before this guard may expose any consumer payload.

## Current verdict

`IMPLEMENTED_FAIL_CLOSED / G51_ENCODER_DIAGNOSTIC_ONLY /
CANONICAL_EFFECT_MATERIALIZER_OWED`.

The repository does not currently have a G51-specific full-n600
realized-through-`R` scorer-effect/JVP/VJP bundle coupled to a same-object
baseline/candidate archive ZIP delta, nor a canonical emitter that proves
those artifacts came from one transition. Therefore G51 remains inadmissible
for sensitivity, allocation, costate, autopilot, and Pareto action. This is a
frontier-protecting guard, not goal progress; the exact frontier pointer is
unmoved.

## Files and verification

- `src/tac/witness_control/taskspace_g51_costate_admission_guard_v1.py`
- `src/tac/witness_control/tests/test_taskspace_g51_costate_admission_guard_v1.py`
- this specification
- `g67_g51_costate_admission_guard_receipt_20260727.json`

Focused verification:

```bash
uv run pytest -q \
  src/tac/witness_control/tests/test_taskspace_g51_costate_admission_guard_v1.py

uv run ruff check \
  src/tac/witness_control/taskspace_g51_costate_admission_guard_v1.py \
  src/tac/witness_control/tests/test_taskspace_g51_costate_admission_guard_v1.py

uv run ruff format --check \
  src/tac/witness_control/taskspace_g51_costate_admission_guard_v1.py \
  src/tac/witness_control/tests/test_taskspace_g51_costate_admission_guard_v1.py

python3 -m py_compile \
  src/tac/witness_control/taskspace_g51_costate_admission_guard_v1.py \
  src/tac/witness_control/tests/test_taskspace_g51_costate_admission_guard_v1.py
```
