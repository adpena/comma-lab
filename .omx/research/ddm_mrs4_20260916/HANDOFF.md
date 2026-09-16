# mrs4: byte-exact native port; timing gates not met

The six-file receiver is implemented and passes the ordered local identity and
bare-environment proofs. The matched corrector/mixer category improves only
**3.278703x**, below the required **10x**. All three
requested projection screens fail to establish a runtime at most **1,200 s**.
This is an **INSTANCE** timing-screen negative for these exact sources and
archive. It does not close native correctors or Lane implementations as families.
No new T4 timing or score was measured; the exact pointer is unmoved.

## Measured profile and the remaining seconds

All profile values below are **[macOS-CPU advisory]**, seed **20260916**, four
Torch CPU threads, Python 3.13.12, NumPy 1.26.4, Torch 2.12.1, macOS 26.4 arm64.
The same 48 SHA-bound pre-frame states are used: two seeded random pairs in
each of 24 consecutive 25-pair strata covering all 600 pairs. Both runs use the
same source and timers; the baseline omits only the corrector library, retaining
the range library. Each run matches all 48 token fields and raw pairs.

| Disjoint stage | Python baseline seconds | Native seconds |
|---|---:|---:|
| Arithmetic + frequency table | 0.649911 | 0.434651 |
| Adaptive corrector + Lane mixer | 82.542602 | 25.175383 |
| Prior network | 82.386276 | 77.272933 |
| Renderer | 25.590696 | 24.973090 |
| Pose carrier + selector | 0.511167 | 0.490182 |
| Other token work | 3.901517 | 3.386267 |
| Total selected-pair work | 195.582169 | 131.732505 |

The pair timer excludes setup, state loading and output writes. Runs were
sequential on a shared host, without a measured noise floor or an exclusive-host
guarantee. The unchanged prior and renderer differences are not attributed to
the port. The historical mrs3 53.53-second statistics measurement is not the
numerator of this matched speedup. The early two-pair diagnostic overlapped the
parity run and banks no timing verdict.

Nested attribution: corrector calls **62.372085 -> 7.878439 s**; native-run Lane
mixer calls **17.390962 s**, of which complete Lane geometry is **7.505668 s**.
Lane coding is **14.237208 s**, shared feature generation **4.574722 s**, and
shared end-frame bookkeeping **2.025408 s**. These rows overlap and include
different wrapper overhead; do not sum them into the disjoint category.
The broad category must reach **8.254260 s** to satisfy 10x; it still needs
**16.921122 s** removed across the 48 pairs. Eliminating the remaining corrector
alone would leave about 17.3 s of broad-category work, so it cannot meet 10x by
itself in this measured split. No third native file was added or proposed as an
authorized implementation.

## Ordered identity and smoke proofs

1. Differential outputs, persistent state, tokens and raw bytes: **48/48** real
   stratified pairs; **9,120** corrector groups; **3,792** persistent array checks
   (79 per frame), 48 previous-frame counters; **27,648** mixer function checks.
   The retained native/Python probability vectors were independently re-read and
   compared: **28,311,552 probability rows** across corrector and mixer outputs.
   Tokens total **9,437,184 B**; sampled raw bytes total **292,992,768 B** per run.
2. Actual cold public n600 parse-back in the bare APFS venv: **600/600** pair
   hashes and complete **3,662,409,600 B** raw match move 53. Full raw SHA-256:
   `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b`.
   Public subprocess runtime **1,474.011096 s [macOS-CPU advisory]**. It has no
   cold stage split; no causal I/O attribution or controlled historical slowdown
   claim is made.
3. Compiler-present bare smoke is the same actual full cold subprocess with
   both native libraries loaded. Compiler-absent bare smoke removes both injected
   stale libraries, observes both fallback diagnostics, and matches **48/48**
   tokens/raw pairs. It is not a second full n600 fallback decode.
4. Matched 48-pair profiles above ran only after both smokes completed.

`PARITY_SUMMARY.json`, `PUBLIC_SMOKES.json`, `INDEPENDENT_PAYLOAD_AUDIT.json`,
and `PROFILE_AND_PROJECTION.json` bind the authoritative retained receipts.
The parity runner predates the nested profiling timers; its exact original
source snapshot and hash survive in `RUNNER_SOURCE_CUSTODY.json`. The standalone
rehash audit is independent executable verification by this same root agent,
not a claimed independent reader. No GT decode or scorer was involved.

## Three T4 projections, not measurements

| Model | Projected seconds | Gate |
|---|---:|---|
| Cold host time x 1760.9 / 1390.8 | 1866.254055 | Above 1200 |
| Cold host time x 1185.899645171 / 936.6589231670368 | 1866.238812 | Above 1200 |
| Conditional stage-model upper envelope | 1644.800706 | Does not establish <=1200 |

The two ratios are rounded/full-precision forms of one transfer assumption,
not independent calibrations. They assume the old whole-process hardware ratio
transfers to this changed distribution of work, dependency versions and host.

The third model scales current serial token work (arithmetic + corrector/mixer +
other token work) by **600/48** and the full-precision ratio, giving
**458.901061 s**. It assumes the selected mean transfers
to n600 and applies the same ratio to C and Python CPU work. The original T4
receipt has **1,114.794542 s** for all token work, not an isolated GPU prior.
Using that whole token time as a deliberately loose GPU-prior upper envelope,
and transferring the original **71.105103 s** non-token remainder unchanged,
gives a conditional interval **[530.006164,
1644.800706] s**. The upper value double-allowances
old CPU token work within the GPU bound; it is conservative under the stated
transfer assumptions, not a measured split or a bound under arbitrary contention,
filesystem behavior or software changes. The lower value does not prove a pass.
Actual mrs4 T4 timing remains untested and unauthorized here.

## Source, provenance and reviewer limits

The C port derives from the read-only sealed `f26_corrector_native.c`, SHA-256
`3e2705f5505036121d85329958a4f23b5ea95e6d20d45ecf92901f2b65cca92a`.
It retains the ordered float64 algorithm and persistent state, drops unused
introspection/comments, reuses checked handle-owned scratch allocation, and adds
the existing Lane fixed-log2 and normalization arithmetic. Geometry retains its
original Python wide-integer semantics. The configuration/family guard prevents
a different Python generation from silently selecting this native corrector.
A failed live native operation raises; missing libraries or refused initialization
select Python with a clear stderr diagnostic. No learned content moved into code.

Both public libraries compile as plain C11; corrector flags include `-O2
-std=c11 -ffp-contract=off -fno-fast-math`. Strict pedantic syntax and warning
checks passed. No OpenMP, host-native instruction flag or compiler-wide integer
extension is introduced. This is a source-derived runtime port, not a new
compression method or originality claim.

| Public file | Physical LOC | Bytes |
|---|---:|---:|
| README.md | 24 | 1,687 |
| archive.zip | binary | 179,286 |
| corrector.c | 693 | 31,190 |
| inflate.py | 2831 | 144,586 |
| inflate.sh | 7 | 337 |
| range_decoder.c | 100 | 4,018 |

The archive is unchanged: SHA-256
`aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`.
There are exactly two one-line compile commands in the seven-line shell and no
compiled library or bytecode in the public delivery. The Python path remains.
`SOURCE_MANIFEST.json` and `PUBLIC_FILE_BUDGET.json` bind all six files.

Fresh-reader prompt: `.omx/research/ddm_mrs4_20260916/FRESH_READER_PROMPT.md`.
MAIN owns that review; none is claimed completed. `PR_BODY_DRAFT.md` is 20 lines,
quotes the archived receiver report, and leaves T4 time as a placeholder.
All five new Python files received two review-tracker passes without override;
`REVIEW_PASSES.json` states the two root-agent review scopes and exact hashes.

## Custody and boundaries

Bulk evidence is `/Volumes/APDataStore/pact/ddm_mrs4/`. All sampled payloads,
probability vectors, archive bytes and resumable pair/stage receipts are retained.
The full cold raw was certified against all 600 hashes before automatic cleanup
as reproducible duplicate scratch. The bare APFS venv is inventoried before
cleanup; wheels and rebuild commands survive. `CUSTODY_LOCATIONS.md` and the
external `RETENTION_SUMMARY.json` record the final retained bytes against 2 GiB.
All heavy launches used the canonical detached launcher with done receipts.

The source delivery is `/Volumes/APDataStore/pact/ddm_mrs4/mrs4_source_delivery.tar`;
external `SOURCE_DELIVERY.json` records the tar hash and member verification.
The serializer request uses post-edit SHA-256 per file, `--no-co-author`, and
`[no-triality] [p0-ledger-ok]`; its exact return code and landing state are in
`/Volumes/APDataStore/pact/ddm_mrs4/serializer/RESULT.json`. Source-file presence
is not a commit claim. rc 17/19 routes the verified source delivery to MAIN.
The ignored unchanged archive travels in the tar; no forced shared-index add.

**65 captured predecessor/sealed files remain hash-identical.** The shared index
is unchanged at the boundary check. No upstream, closed-PR, sealed-tree, mrs1–3,
or common-contract protected-file edits; no stash or direct index manipulation.
No Modal, fire, packet, PR, push, `authorize_*`, scorer, GT decode, MPS authority,
third C file, dependency pin, public environment switch, OpenMP/native flag,
co-author/AI trailer, review override or public generated file. No SSD permission
denial occurred. No scorer slot was claimed. Unrelated dirty work is preserved.
`BOUNDARY_VERIFICATION.json` distinguishes the hash-checked scope from command
history attestations. Research-only state is retained; no score promotion.

## RECALL EVIDENCE

`RECALL_EVIDENCE.md` contains full-corpus queries over research memo contents,
490 equation records, index/DAG, design/SPEC and task-ledger surfaces. Queries
include `corrector`, `float64`, `Lane.mixer`, `rlc1_geometry` and runtime/coding
variants. Beyond the charter seeds, fcd1's family-generation mismatch led to the
configuration guard and all-79-array comparison; rr8's invalid stage transfer led
to the explicitly conditional T4 model; tc3's inventory fixed the checkpoint
boundary. All six integration consumers are recorded there. No new equation or
frontier claim is manufactured from a receiver timing result.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/state/canonical_task_status.jsonl` task `ddm_mrs4_source_landing_20260916`; trigger: harvest and verify SOURCE_DELIVERY.json plus the serializer outcome.** Land the exact delivered source and evidence.
- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; same consumer, task `ddm_mrs4_fresh_reader_20260916`; trigger: exact sources under MAIN custody.** Run FRESH_READER_PROMPT.md and retain the response against the six hashes.
- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; same consumer, task `ddm_mrs4_timing_gate_resolution_20260916`; trigger: harvest the failed PROFILE_AND_PROJECTION.json gates.** Resolve the remaining stage cost within the two-C/700-line limits or explicitly revise the charter. Any new T4 measurement and implementation leads are FOLDED into this decision, not fired.

`MAIN_FIRE_ORDER.json` and `REGISTERED_FOLLOWONS.json` preserve the exact task
rows and triggers. Existing unrelated ledger validation warnings were not
silently repaired; the three new rows were written through the canonical API.

## LIVE-HYPOTHESES

- Lowering more Lane-mixer bookkeeping inside the existing native-file budget
  may help: Lane calls now dominate statistics, and geometry plus shared feature
  generation are measured costs. Exact wide-integer semantics and reviewer fit
  remain untested. FOLDED into MAIN's timing-resolution row.
- Reusing native context work or reducing Python/native boundary conversions may
  lower the remaining 7.88-second corrector cost. This is a plausible profiling
  lead, not a demonstrated optimization, and cannot alone meet the broad 10x
  gate in this split. FOLDED into the same row.
- Actual T4 runtime could differ from the conservative projections because the
  GPU prior share is unmeasured. A new measurement is conditional on MAIN's gate
  resolution and authorization; the projection is not evidence of a real T4 miss.

## DEAD-ENDS

- **INSTANCE:** treating this port as a completed 10x/1200-second timing cure.
  It delivers 3.278703x broad-stage speedup, with all projection screens unmet.
- **INSTANCE:** further corrector-only tuning as the complete 10x cure. Even
  zero corrector time leaves too much measured mixer work; both must be addressed.
- **SOURCE-MAPPING INSTANCE:** native HPAC as the statistics cure. It targets
  the prior network and the sealed receiver rejects that native path.
- **UNJUSTIFIED EQUIVALENCE:** replacing float64 correction and wide-integer
  geometry with naive int64. It does not preserve the reference arithmetic.
- **UNSUPPORTED CLAIM:** using the two ratios as independent evidence, treating
  the GPU prior bound as an isolated measurement, or transferring move53 score
  authority to this receiver without a new authority run.

composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)
