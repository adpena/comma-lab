# G35 findings — R10 bounded-inverse fitter P0 hardening

Date: 2026-07-26
Lane: `lane_g35_r10_fitter_p0_hardening_20260726`
Predecessors: G32 implementation and G34 adversarial review
Authority: local structural verification plus one bounded real-input n1 mechanism run
Execution exclusions: no n600, scorer, public mux, candidate, pointer mutation, commit, or push

## Verdict

G35 closes the local G34 P0 implementation defects needed before an n600 fire
can even be considered. The historical filesystem paths retain `maximum` only
for compatibility; the current Python API, schema, solver claims, residual
inventory, and receipt surface call the implementation what it is: a
**bounded finite R10 inverse fitter**.

This landing did not lower the exact score. Pointer delta is false, no score is
claimed, and the upstream-authority blocker remains:

`FULL_N600_PUBLIC_RECEIVER_SCORER_AND_COMPLETE_ZIP_PRICING_OWED`

## Closed defects

### 1. Immutable checkpoint custody

Stage, raw-range, and fit-range checkpoints now require all of:

- exact schema and key set;
- exact launch binding;
- canonical stage/range coordinates;
- filename suffix equal to SHA-256 of the retained record bytes;
- payload SHA-256 equal to canonical payload bytes;
- predecessor-record SHA-256 equal to the prior immutable record; and
- a single gap-free append-only prefix.

Mutation plus an updated internal payload hash while retaining the old filename
now refuses. Renaming a mutated record to its new content hash still refuses if
its predecessor or coordinates differ.

### 2. Real crash recovery

- A crash after `030_geometry` can restart XIP2 using the retained pitch.
- The selected runtime performs `_setup()` exactly once for the complete
  missing population tail, fsyncs each range, and publishes the range proof
  before continuing.
- XIP2, BASE_FEATURE, TEXTURE, SHOOTING_KNOT, STRATIFIED_FLOW, and the four
  joint-refit pair blocks persist content-addressed range state.
- A complete retained `110_joint_refit` synthesizes the exact range operands
  from its counted packet and trace, so resume verifies but does not re-solve
  completed pairs.
- Changed code/config/input/claim bindings refuse inherited state.

The bounded n1 run retained 15 stage checkpoints and 9 fit-range checkpoints.
Its complete-resume invocation returned in the `resumed_complete=true` path
after reopening the receipt, strict packet parse/re-emission, STORE member,
cleanup intent, cleanup certificate, and removed/preserved path claims.

### 3. Correct native geometry operation

The old proxy did:

`warp(new_geometry(sample(frame)))`

The fitted objective now does:

`sample(warp(native_geometry(frame)))`

These operators do not commute. Every pitch/XIP2 candidate now builds the
frozen geometry on `(H,W)`, executes the full-native warp, and samples only the
resulting integer residual. After XIP2 quantization, a full-resolution per-pair
zero-versus-fitted control zeros any fitted coordinate that worsens exact source
RGB SSE, requantizing until stable.

On the real n1 row, the quantized coordinate survived that control:

- zero full-resolution SSE: `2,453,412,736`;
- fitted full-resolution SSE: `2,427,651,739`;
- retained: `true`.

This is source-RGB encoder evidence, not scorer authority.

### 4. Governed n600 admission

The two caller-only n600 confirmation booleans are gone. Exactly n600 now
requires:

- `TAC_GOVERNED_ADMISSION=1`;
- exact governed claim job/platform arguments;
- a live nonterminal claim no older than 24 hours in the canonical ledger;
- the fixed G32 lane ID;
- the canonical claim-record SHA-256 bound into the stable launch binding; and
- live revalidation before/after materialization and before every fit range and
  stage publication.

The mutable whole-ledger SHA is recorded on observations, not used as the
stable binding: unrelated lane appends cannot brick an otherwise valid resume.
No live n600 claim was exercised in G35; only the fail-closed focused tests ran.

### 5. Claim and telemetry precision

- Public types/functions are `R10BoundedInverse*` and
  `compile_r10_bounded_inverse`.
- Schema is `taskspace_r10_n600_bounded_inverse_fitter.v2`.
- `bounded_action_family_exhausted=false`.
- Five unenumerated action families are explicit in the residual inventory.
- The inherited frozen-receiver `maximum_inverse_solve_hook` and
  `unbounded_encode_only...` labels are scoped at the G35 receipt boundary to a
  bounded-family hook; a receipt-level stale-label assertion enforces this.
- Positive `+32767` and negative `-32768` saturation are both counted.
- Whole-packet objective/runtime/change values are no longer copied into every
  section as fake marginal telemetry. Unmeasured section marginals are null;
  section-specific iterations and sufficient statistics remain.

### 6. Cleanup and completed-resume custody

Cleanup is now intent-before-delete:

1. hash every scratch object and write an immutable cleanup intent;
2. verify retained bytes against the intent;
3. delete or preserve as requested; and
4. write an immutable completion certificate linked to the intent hash.

A crash after some/all deletions but before certificate publication resumes
from the intent and cannot produce an empty cleanup proof. The intent also
binds rebuild argv/environment, source, selected archive, runtime, R10 receiver,
G22 receipt, launcher, fitter, and ffmpeg identities.

Completed resume reopens and verifies the receipt filename/hash, packet
filename/hash/bytes/strict parse, wrapper filename/hash/bytes/exact member,
cleanup intent/certificate linkage, and the truth of every removed/preserved
path assertion.

## Real bounded n1 mechanism receipt

Canonical durable root:

`/Volumes/VertigoDataTier/pact/g35_r10_fitter_p0_hardening_20260726/n1_real_v2`

Exact artifacts:

| artifact | bytes | SHA-256 |
|---|---:|---|
| counted R10 packet | 675 | `8c40c549c558cdaddfa9e17e1f62474b510a62f92b4f08f68a87f7b3a5b94987` |
| deterministic STORE wrapper | 793 | `6839f3d712e8b2c4ef51f4f6226bfbff05dc692211e0f0d50e2f8e8140415379` |
| full G35 receipt | — | `8796b8638dd05511b68e9f6f49548f5b7bc86352809c7034f59e93e826411420` |
| cleanup intent | — | `e26225f009dd0073b1668417a88457e33b6a29791d8bf06f9078716afede0c45` |
| cleanup certificate | — | `b4a58058abef9e593b291890ed249b99c0e92daf11ea50094e6a52bb27040527` |

Receiver evidence:

- output SHA-256:
  `7cb3e453a7b4e4b14ca6f0556dba9b4362642728cdfe1262d67207c65df54cb4`;
- deterministic double replay: `true`;
- selected-base SHA-256:
  `06722660ed18e3f60ae81beb90c35ecebdfe864295ffb673c9d1bd42a00a2467`;
- changed values: `5,732,274`;
- changed pixels: `1,985,060`;
- L1 RGB delta: `47,698,966`;
- max absolute RGB delta: `104`;
- pointer delta: `false`;
- score claim: `null`.

Recorded timing components:

- selected runtime `_setup`: `0.09653541608713567 s`;
- selected tail range: `5.200906333979219 s`;
- one-pass source extraction range: `0.21548329200595617 s`;
- custody-checkpoint to cleanup-certificate filesystem timestamps: `37 s`.

The exact end-to-end wall duration is not retained in the receipt. That is a
remaining telemetry debt; the 37-second timestamp delta is not promoted into a
more precise claim.

All six scratch objects were intent-certified and removed. The scratch tree has
zero retained files. No cleanup remains for v2. The earlier v1 mechanism run is
superseded because it exposed the stale inherited maximum label; its large
scratch was also certified and removed, so it is preserved evidence rather than
orphaned bulk.

## Verification

```text
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_r10_n600_maximum_inverse_fitter.py \
  tools/tests/test_fit_taskspace_r10_n600_maximum_inverse.py
33 passed in 0.62s

.venv/bin/python -m ruff check <four G35 Python files>
All checks passed!

.venv/bin/python -m ruff format --check <four G35 Python files>
4 files already formatted

.venv/bin/python -m py_compile <four G35 Python files>
PASS

.venv/bin/python tools/lane_maturity.py validate
OK — 2181 lane(s) validated cleanly.
```

Focused attacks cover stage/range content drift, predecessor drift, coordinate
drift, legal geometry-only restart, native-warp ordering, full-resolution XIP2
control, one selected setup and tail resume, per-fit-range no-resolve resume,
full-stage no-resolve resume, direct n600 refusal without governed admission,
cleanup crash recovery, completed artifact reopen, strict counted-state
coverage, receiver mutation canaries, and stale maximum/unbounded receipt
labels.

## Source identities at handoff

| file | bytes | SHA-256 |
|---|---:|---|
| fitter library | 114,822 | `8892b2c49569739d3b7809cd20b0781c1c88229e2b74741c3a641c716bbcc2f7` |
| launcher | 71,221 | `46f7e4b5c4543e1af6390f0773d29508f8eeba5c54b8a9591632be22c8c1efd4` |
| fitter tests | 15,212 | `0b6b9440b1af6852c6e8491374274c8841288a224eb7bdbc0d1f3fc4dd860a1d` |
| launcher tests | 19,147 | `480e2c66af4d20ca6793099628ecc765e8556f071743e2a01defa6dfaf469fd8` |
| G35 spec | 6,558 | `74b68baa1e31bbab78abb047234671a43e656d0600e23fa4bd213693762fdbe9` |

These are shared dirty-main untracked files. G35 did not commit or push.

## Triality

- DSL: the admitted object is a finite bounded source-RGB action family with
  counted R10 operands; learned/global/scorer-native families remain named and
  unenumerated.
- DAG: custody → selected/source materialization → immutable stage/range fit →
  strict packet → STORE wrapper → bounded receiver replay → receipt → cleanup
  intent/certificate → complete reopen.
- Equation: native candidate evaluation is
  `J(theta)=||S(R_native(x;theta)-y)||_2^2`, followed by the exact control
  `||R_native(x;theta_q)-y||_2^2 <= ||R_native(x;0)-y||_2^2`.
  `R_decimated(S(x);theta)` is not substituted for `S(R_native(x;theta))`.

## Remaining blockers and NO-FIRE status

Local G35 P0 defects are closed, but the capstone is still **NO-FIRE** until the
following external/action-space blockers are resolved:

1. No governed n600 G35 fit exists. Full-population packet/runtime/memory and
   fit telemetry therefore do not exist.
2. The bounded family is not the global R10 action universe and is not selected
   against exact scorer plus complete-ZIP Lagrangian value.
3. The G29/public-video endpoint, upstream recursive evaluator replay, and
   complete submission archive price are still outside this landing.
4. Continuation identity binds operands and receiver closure, but full public
   endpoint control-bisimulation/custody still needs G29/G31 composition.
5. The code-as-data audit remains lexical plus exact counted-span coverage; a
   stronger semantic/transitive proof remains open.
6. Terminal exact realized-score/ZIP joint descent is not admitted.
7. Exact total-wall timing is not persisted, and seed is recorded in the
   binding although this finite implementation uses no RNG.

The next authorized action is not an arbitrary n600 launch. It is to compose
the now-hardened bounded fitter with the current G29/G31 public-authority
closure, obtain an active governed claim for the fixed lane/job/platform, and
only then decide whether an n600 fit buys an imminent exact row. No old
`--confirm-full-n600` command is valid anymore.

## Stores consulted

- byte-identical `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, arbitrage skill, top
  project MEMORY, live lane/progress state, and last-24h directive search;
- G32 spec/findings and exact n1 custody;
- G34 adversarial review and executable attacks;
- canonical lane-claim tool and governed-execution precedent; and
- live G35 v2 stage/range/receipt/cleanup artifacts.
