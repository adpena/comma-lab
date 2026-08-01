# G34 adversarial review — G32 R10 n600 maximum-inverse fitter

Date: 2026-07-26

Review lane: `lane_g34_r10_fitter_adversarial_review_20260726`

Reviewed lane: `lane_g32_r10_n600_inverse_fitter_20260726`

Scope: review only; no G32 implementation, test, spec, receipt, scorer, public
runtime, candidate, pointer, or external payload was modified or executed

Authority: local structural and bounded adversarial verification only

Pointer delta: **false**

## Verdict

**NO-FIRE for the dormant n600 command.** G32 contains real and useful pieces:
it reopens the exact own-lineage G20/G22 base, produces a real canonical packet,
strictly replays the frozen G27 receiver at n1, wraps the packet in a real STORE
ZIP, constructs a byte-owning `G17PhysicalCodingGroupV1`, keeps scorer and score
fields null, and certifies the current n1 scratch deletion. Those facts survive
review.

The stronger claims do not. The current implementation is not crash-resumable
across every legal checkpoint boundary, its purported content-addressed chain
accepts in-place content drift, its geometry/XIP2 fit uses a non-commuting proxy
operation, several required receiver degrees of freedom are not solved, its
"nontraining options exhausted" bit is false, its controller identity is an
endpoint identity rather than continuation equivalence, and the selected-base
materializer still performs 300 selected-runtime setup cycles before fitting
begins. The initially reviewed source path also had approximately 150.5x
redundant decode work; root replaced that path with one linear stream during
this review and exact stride-controlled n1 replay closed determinism. Firing the
remaining stack would spend hours producing
a distortion-first proxy endpoint that is not yet a complete G33 action-universe
member and cannot enter the public G29/G23 path.

Blocker classification:

- implementation maturity: **L1 scaffold remains honest**;
- bounded n1 mechanism receipt: **ACCEPTED with corrected scope**;
- full n600 fit: **BLOCKED — correctness, resumability, governance, and
  wall-clock architecture**;
- public decoder/mux: **BLOCKED on G29/G23 closure**;
- scorer/score/promotion: **ABSENT, correctly null**;
- frontier movement: **none**.

## Reviewed exact objects

| object | bytes | SHA-256 |
|---|---:|---|
| fitter library | 84,554 | `7c30f49d27c63e54f7e13c5d8cc9872208285b846d532abd1bdc6bbfd2aae81f` |
| fitter library test | 10,023 | `1a50c123f4fa5e721da114d620e19688cfb52c62593432cbf28403aabef40438` |
| production launcher, after linear-source repair | 43,374 | `4893143d5946a595e8c1a118f12960213ae98cc7238ab0414238197756cb6339` |
| launcher test, after linear-source repair | 9,029 | `39bd77c127ce640dac4d0f574d436b0ec7bfdbcc6176df031e1a7d54c766a8f4` |
| G32 specification | 19,513 | `5449d64302bce4856436d9c13e069bfc0a243da436169ce563a129b300c8f217` |
| G32 findings | 14,254 | `42305903ee0f33925067a6a8d4448b7526f39f3485c1da9fe1287d1bcebb7ab3` |
| exact n1 receipt | — | `269f7f2368533013795c689260ba4fe492e8fda65e981bdde684fe706fc4cc90` |
| n1 packet | 675 | `3006db7af8122da54a4e03e546fbbe651aa648f9a12dc7e1e0a6a8413959d6f9` |
| n1 STORE wrapper | 793 | `006842e0fc4fd012ebb9bb112d3f454bf2fc9e983f4024ec2df92eef276869e9` |
| n1 cleanup certificate | — | `dd515db4d8360f3c862711eef548fafee422aa9c27fa219c8d4d9e576543bda6` |
| linear-source stride-16 n1 receipt | — | `a7574f95bb9c21a4ccf199dac8d55decf6549479298360dc9966ee81397b6244` |

The packet and wrapper currently match their filenames and receipt, the wrapper
reopens to exactly one `r10.packet` member, all nine section spans reopen, all 17
current stage/chunk checkpoint filenames currently match their bytes, and the
six certified scratch paths are absent. The vulnerabilities below are in what
the implementation will accept after drift or interruption; they are not a
claim that the retained n1 bytes are presently corrupt.

## P0 findings — block n600 fire

### P0.1 — The immutable checkpoint chain is mutable and accepts content drift

`ImmutableStageStore.publish()` names a checkpoint with the SHA-256 of its
record (`tools/fit_taskspace_r10_n600_maximum_inverse.py:302-320`), but
`load_prefix()` never recomputes that file hash or compares it with the digest
embedded in the filename (`:324-350`). It trusts a mutable record's internally
rewritable `payload_sha256`. `ChunkStore.lookup()` has the same omission and
also does not validate the record schema or its requested pair/range fields
(`:365-377`).

Adversarial execution modified a published `000_custody` record in place from
`{"exact":1}` to `{"exact":999}`, updated only the internal payload digest,
and retained the old filename. The actual file SHA became
`50cfed446f658a34d74a5e319dc5ba54c380dc344b84940f8cfb63b91ad175bc`,
while the retained filename still ended in
`71d9649ecf8dec8e413d966beb95cae83da7ad59de231fa2c9d6749ec5a205e8`.
`load_prefix()` accepted `{"exact":999}`.

This breaks immutable provenance, resume custody, and the premise that content
roots protect stage state. Before fire, every checkpoint load must verify the
filename content digest, an exact record schema/key set, predecessor-chain
digest, binding, stage/range coordinates, retained payload/range bytes, and
write-once filesystem identity.

### P0.2 — A legal crash between `030_geometry` and `040_xip2` permanently bricks the run

The fitter publishes geometry and XIP2 as two distinct immutable stages
(`taskspace_r10_n600_maximum_inverse_fitter.py:1147-1160`) but resumes them only
when both are present and refuses either one alone (`:1123-1143`). An executable
crash injection immediately after `030_geometry` retained the legal contiguous
prefix `020_pair_index,030_geometry`; restart failed with:

```text
R10MaximumInverseError: geometry and XIP2 resume checkpoints must be present together
```

Because `030_geometry` is immutable, the run cannot repair this state without
manual deletion, which the operating contract forbids. Publish this coupled
state as one atomic stage or make XIP2 continue from the retained pitch.

The broader n600 stages are also not chunk-resumable: `chunk_pairs` governs only
base/source materialization. The inverse fitter loops the entire pair population
and publishes one checkpoint only after each whole stage. A crash late in
BASE_FEATURE, TEXTURE/DASH1, knots, flow, or the joint pass loses that whole
stage. The already-retained `110_joint_refit` and `120_packet_adapter` stages are
ignored on resume and recomputed. This does not meet the mandatory per-stage
plus intra-stage preservation contract.

### P0.3 — Geometry/XIP2 optimization uses a proxy operation, not the exact frozen receiver operation

`_fit_pitch_and_xi()` first decimates the input and then constructs a new
`GroundHomographyGeom.eon(native_hw=base.shape[:2])` on the decimated dimensions
(`taskspace_r10_n600_maximum_inverse_fitter.py:475-486,493-503`). The frozen
receiver constructs geometry on the native output dimensions. Homography plus
bilinear sampling does not commute with decimation, so this is not "the same
G27 homography operation on decimated sufficient statistics."

An adversarial deterministic array measured:

```json
{"attack":"warp_then_decimate_vs_decimate_then_regeometry_warp","changed_values":219,"l1":4024,"max_abs":77,"shape":[8,10,3]}
```

The final full-resolution source-RGB objective only chooses between two corners
after pitch/XIP2 are frozen; it does not repair the proxy-selected geometry.
The exact cure is full-native receiver execution with sampled residual
accumulation, or an analytically proven intrinsics/resampling transform whose
integer output is demonstrated identical to full receiver then sample.

### P0.4 — “Maximum inverse” and “nontraining options exhausted” are false for the implemented action space

The result sets
`nontraining_options_exhausted_on_bounded_source_objective=True`
(`taskspace_r10_n600_maximum_inverse_fitter.py:1879-1886`), but the implemented
search leaves major declared receiver degrees of freedom and required spec work
untouched:

- geometry/XIP2 executes one plus/minus coordinate visit per coordinate and
  scale, not a fixed-point/convergence loop;
- GEOMETRY+XIP2 are not refit in the interaction pass; the trace merely says
  `revalidated_finite_solution=True` (`:1426-1435`);
- texture gain is hard-coded to 256; amplitude/gain factorization and
  compression alternatives are not searched (`:742-758`);
- DASH support is one fixed square around the largest frame-1 RGB residual per
  pair, not the connected residual-component/action universe (`:667-699`);
- SHOOTING_KNOT stores one knot per pair and fits only luma-bias delta; the other
  five receiver deltas, maximum-deviation refinement, and complete
  distortion/byte Pareto family required by SPEC G32 section 5.6 are absent
  (`:1317-1352,1517-1538`);
- PULLBACK_POLYGON is the same fixed box, not a selected connected pullback;
- STRATIFIED_FLOW searches only two translation coefficients over five values;
  the other four affine coefficients and the specified Lucas-Kanade normal
  equations are absent (`:785-810,851-887`).

This is a real bounded finite fitter for one narrow formulation, not a maximum
inverse solver and not evidence that training is the remaining irreducible
option. Rename/scope the current result or implement the complete typed action
universe; keep the exhaustion bit false until every receiver-realized analytic
and factorization action is either measured or has a typed blocker.

### P0.5 — Source decode amplification was fixed during review; selected runtime still repeats setup 300 times

The initially reviewed `--pair-count 600 --chunk-pairs 2` source materializer
started 300 fresh ffmpeg processes. Each command had no input seek and applied a
`select=between(n,first,last)` filter, so it decodes from frame zero through that
chunk's last frame (initial launcher lines 616-649).
The exact decoded-prefix total is:

```text
sum_{k=1..300} 4k = 180,600 decoded source frames
```

instead of 1,200 once: **150.5x frame-decode amplification**, plus 300 process
starts.

Root repaired the source half during this review. Current
`materialize_source_pairs()` uses one ffmpeg process for the complete ordered
population, streams bounded ranges directly into the preallocated raw, replays
and compares already-retained range hashes on resume, and keeps range
checkpoints (current launcher lines 603-735). A new stride-16 n1 run reproduced
the original exact selected/source range hashes, packet
`3006db7af8122da54a4e03e546fbbe651aa648f9a12dc7e1e0a6a8413959d6f9`,
wrapper `006842e0fc4fd012ebb9bb112d3f454bf2fc9e983f4024ec2df92eef276869e9`,
output identity, and continuation identity. Receipt:

`/Volumes/VertigoDataTier/pact/g32_r10_n600_maximum_inverse_fitter_20260726/n1_linear_source_stride16_v1/artifacts/receipt.a7574f95bb9c21a4ccf199dac8d55decf6549479298360dc9966ee81397b6244.json`

The earlier differing packet under the linear source path used
`sample_stride=8`; the original used 16. It was a config change, not
nondeterminism.

The selected-base half remains unfixed: `materialize_selected_base()` launches
a new Python process,
imports the runtime, and runs full `_setup()` once per two-pair chunk
(`:572-590,489-520`): 300 setup/reopen cycles. Setup time is excluded from its
reported `wall_seconds` because timing starts after `_setup()`.

The n1 fitter itself reports 27.67 seconds; a naive linear lower-bound
extrapolation is already about 4.61 hours before these repeated setup/decode
costs and before any public/scorer work. Encoder compute is legally unbounded,
but this command is not an honest shortest-wall-clock route to today's exact
row.

Keep the repaired single source decoder. Replace selected-base production with
one persistent selected-runtime setup, streaming directly into preallocated
ranges. Checkpoint and fsync each produced range. On crash, restart the producer
once, validate retained prefix hashes, and replay only to the first missing
range.

### P0.6 — Full fire is governed by caller-attested booleans, not canonical lane ownership

`--execute-reviewed`, `--confirm-full-n600`, and
`--confirm-no-live-heavy-owner` are booleans only (`tools/fit_taskspace_r10_n600_maximum_inverse.py:411-430`).
The launcher does not query the live lane registry, claim a dispatch lane,
consume a reviewed authorization receipt, or run through the governed launcher.
Any caller can assert the flags. Before n600, replace attestations with exact
canonical claim/authorization identities and revalidate them at launch and each
resumed heavy stage.

## P1 findings — block composition/promotion after the fitter runs

### P1.1 — Packet selection is a single RGB-distortion corner, not the coupled score/action universe

The compiler selects exactly one of `INITIAL_SEQUENTIAL` and
`POST_INTERACTION_REFIT` by full-resolution source RGB SSE and canonical operand
bytes (`taskspace_r10_n600_maximum_inverse_fitter.py:1591-1648`). It does not
jointly price d_seg, nonlinear d_pose, complete ZIP rate, decoder feasibility,
factorability, compressibility, or continuation value. It therefore repeats the
arbitrary single-coordinate failure the capstone is meant to remove.

Preserving earlier stage files is not an emitted complete action universe. G33
needs every nondominated exact packet endpoint (including delete/factorization
corners) with a same-base public endpoint, full byte delta, component debt, and
continuation identity. G32 should compile endpoints; the global receding-horizon
controller should select one whole-object action.

### P1.2 — The named continuation-equivalence identity is only deterministic endpoint identity

`R10ContinuationEquivalenceIdentityV1` binds source, pair population, base
realization, selected packet, G27 receiver source closure, output shape/dtype,
and pair order (`taskspace_r10_n600_maximum_inverse_fitter.py:326-342`). Those
fields are useful and should remain.

They do not bind the future transition/action set: alternate corners, solver and
checkpoint state, residual inventory, factorization/regeneration possibilities,
public mux/runtime capability, decoder resource feasibility, G23 product
placement, or terminal-joint-descent state. Two endpoints with the same current
packet/output but different preserved future moves collide. That violates the
control-bisimulation meaning of continuation equivalence discovered in G33.

Split the domains explicitly:

1. artifact identity — exact packet/wrapper/checkpoint/proof bytes;
2. deterministic endpoint identity — base + receiver + packet -> uint8 output;
3. semantic control/continuation identity — endpoint plus complete available
   action-universe digest and transition/resource/public-closure state;
4. proof-dependency set — all receipts and custody roots.

### P1.3 — The hidden code-as-data gate is lexical self-attestation

`audit_counted_state_manifest()` verifies packet span coverage well, but its
code-as-data test scans human-provided strings for a short blacklist
(`taskspace_r10_n600_maximum_inverse_fitter.py:943-981`). The manifest itself
hard-codes `forbidden_payloads_present=()` and static generic-mechanism prose.
It does not inspect shipped source/files, active decoder dataflow, imported
closure, syscalls, or hidden reads.

Executable adversarial mutation appended a semantically source-derived lookup
under the innocuous string `fixed lookup sequence rows=5,9 cols=5,9 value=0`.
The audit accepted it because no blacklisted token appeared. This does not taint
the current wrapper, which contains only the counted packet; it means the gate
cannot certify a future G29 inflate bundle. Public closure must inventory exact
shipped files, semantically classify constants/assets, prove decoder-input
reachability, and attach hidden-read/process/syscall evidence.

### P1.4 — Per-section telemetry repeats whole-packet values and estimated resources

Every `R10SectionFitV1` receives the same whole-packet `objective_before`,
`objective_after`, changed-value statistics, total elapsed time, and a formulaic
`peak_working_bytes` (`taskspace_r10_n600_maximum_inverse_fitter.py:1840-1863`).
GEOMETRY and XIP2 both receive the same combined iteration count; setup time,
per-stage wall/CPU time, actual RSS/native allocations, section deletion delta,
and interaction deltas are not measured. This suppresses exactly the marginal
signal the costate/allocator needs.

Emit separate before/after receiver realizations or streamed sufficient
statistics per section/action, measured stage timers and RSS/native peaks, and
conditional parent sets. Keep values null when not measured; do not duplicate a
whole-packet number into a per-section schema.

### P1.5 — Completed resume does not reopen artifact and cleanup custody against the receipt

The completed fast path reopens the receipt but returns whatever hashes the
current packet/wrapper paths have; it never compares them to the packet/wrapper
identities inside that receipt (`tools/fit_taskspace_r10_n600_maximum_inverse.py:775-800`).
It also returns the `140_cleanup_certificate` payload without reopening its
certificate path/hash. The pre-cleanup fast path has the same artifact gap.

There is also a two-phase cleanup crash window: the certificate is written,
scratch is deleted, and only afterward is stage 140 published (`:666-726,
945-952`). A crash after deletion but before stage publication resumes from
stage 130, finds no scratch, and can publish a new empty cleanup certificate,
orphaning the real deletion proof. Reopen every artifact/certificate on resume
and use a pending-intent -> verified-delete -> completed-certificate protocol.

## P2 findings — hardening and evidence precision

### P2.1 — Cleanup reproducibility omits implementation and transitive receiver closure

The cleanup rows preserve argv, thread environment, source, runtime, and selected
archive hashes, but not the exact fitter/tool source hashes, G27 transitive
closure, ffmpeg binary/build identity, or G22 proof root
(`tools/fit_taskspace_r10_n600_maximum_inverse.py:666-705`). These exist partly
elsewhere in the launch binding/receipt; the cleanup certificate should bind
them directly so it is independently sufficient.

### P2.2 — Claimed `git diff --check` did not inspect the untracked G32 files

All six reviewed G32 repo files are currently untracked. `git diff --check --
<files>` exits zero without examining untracked content, so that verification
line in the G32 findings is not evidence. Ruff, format, pycompile, and pytest are
real and passed; use an index-safe no-index check or serializer dry-run for
untracked whitespace validation.

### P2.3 — Small telemetry inaccuracies remain

`saturated_parameters` counts `abs(value)==32768`, which misses positive int16
saturation at 32767 (`taskspace_r10_n600_maximum_inverse_fitter.py:656`). The
seed is recorded but unused. These do not independently block the scaffold, but
they should be corrected before telemetry is used for allocation.

### P2.4 — The in-review source repair has not yet passed format verification or refreshed G32's own findings

After the linear-source repair, 23 focused tests and Ruff lint pass, but
`ruff format --check` reports that
`tools/fit_taskspace_r10_n600_maximum_inverse.py` would be reformatted. G32's
own findings still record the pre-repair launcher/test hashes and `21 passed`.
Refresh the durable evidence only after review fixes land; do not leave the old
hash table as current custody.

## Positive closure that survived review

- Exact source, selected archive/member, runtime, frozen G27 module, and G22
  receipt hashes are checked before a real run.
- The n1 packet is not a fixture: it changes real selected-base uint8 output,
  and frozen G27 double replay is deterministic.
- Strict packet parse/re-emission and all nine physical section spans pass.
- The deterministic STORE wrapper and `G17PhysicalCodingGroupV1` own real
  retained bytes; member and archive-relative spans are correct.
- The n1 receipt keeps d_seg, d_pose, complete candidate bytes, contest score,
  promotion, and pointer movement null/false. No authority laundering was
  found.
- No scorer, `upstream/evaluate.py`, public video mux, n600 run, external
  payload, commit, or push occurred in this review.
- The current cleanup certificate is real and all named scratch files are
  absent.

## Verification

Focused upstream tests and lint:

```text
PYTHONPATH=src .venv/bin/pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_r10_n600_maximum_inverse_fitter.py \
  tools/tests/test_fit_taskspace_r10_n600_maximum_inverse.py
23 passed in 0.45s

.venv/bin/ruff check <four G32 Python files>
All checks passed!

.venv/bin/ruff format --check <four G32 Python files>
Would reformat: tools/fit_taskspace_r10_n600_maximum_inverse.py
1 file would be reformatted, 3 files already formatted
```

Adversarial executions:

```text
checkpoint filename/content drift: ACCEPTED (must refuse)
crash after 030 before 040: resume REFUSED legal prefix (must continue)
full-warp->sample vs sample->new-geometry-warp: 219/240 values differ
semantic code-as-data without blacklist token: ACCEPTED (must refuse/leave unproved)
current n1 checkpoint filename hashes: 17/17 presently match
current n1 packet/wrapper/certificate hashes: match retained names and receipt
linear-source stride-16 replay: exact packet/wrapper/output identity reproduced
```

## Fire criteria

Do not launch n600 until all P0 findings are closed and re-reviewed. Minimum
criteria:

1. content-root and predecessor-chain verification plus crash injection at
   every stage boundary;
2. atomic geometry/XIP2 continuation and per-chunk fitter checkpoints;
3. exact full-native G27 operation in every selection objective;
4. complete receiver-DOF/action-universe solve with exhaustion false until
   proven;
5. persistent single-pass base/source materializers with measured wall/RSS;
6. canonical governed-launch and live-lane claim receipt;
7. G32 emits nondominated endpoints, while G33 alone selects the global action;
8. endpoint identity and control-continuation identity are separate;
9. G29/G23 public mux, exact 1,200-frame output, double decode, runtime/memory,
   and recursive evaluator closure are complete;
10. only then run full n600 scorer/complete-ZIP pricing and terminal joint
    descent.

## Stores consulted

- byte-identical `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, arbitrage skill, and
  top current Claude MEMORY;
- live lane registry, subagent ownership/checkpoints, and recent directives;
- G32 spec/findings/four-file implementation boundary and exact n1 receipt;
- frozen G27 receiver/adapter/source-closure code;
- G23 `G17PhysicalCodingGroupV1` byte-custody implementation;
- current G33 continuation-equivalence/controller findings supplied by the
  parent capstone state.
