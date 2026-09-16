# mrs3: charter source mismatch proven; runtime cure not implemented

**PARTIAL / BLOCKED ON THE EXPLICIT SOURCE CONSTRAINT. No new receiver speedup or
score was measured.** The charter assigns the 45% statistics category to the
wrong native file. A new scorer-free 48-pair profile identifies the actual work,
and all 48 token fields and raw pairs match the predecessor byte-for-byte.
The requested scope correction was asked of the operator and had not been
received when this receipt was written. The five-file `submissions/mrs3/` tree
is an unchanged copy of landed mrs2, not a completed cure or a dispatch candidate.

The exact archive remains **179,286 B**, SHA-256
`aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`.
The exact pointer is unmoved by this work.

## The source constraint that prevents the default implementation

The charter requires Route A to derive `context_stats.c` from the sealed
`runtime/f26_hpac_native.c`. However, the actual mrs2 profiler assigns:

| Profiler category | Actual implementation |
|---|---|
| `prior_context_statistics` | `miss_FreeCorrector` and `geometry_mixer_LaneMixer` |
| `prior_network` | HPAC `prepare_frame_context` and sparse prior logits |

The first category maps to sealed `f26_corrector_native.c`, `rlc1_geometry.c`,
and the shared-mixer Python code. The nominated `f26_hpac_native.c` maps to the
second category. More decisively, sealed `runtime/f26_inflate.py:455-463`
**explicitly refuses native HPAC**, because it is unpatched and would decode a
different field. Thus the successful original move-53 T4 row did not use that
file as alleged in the charter. `SOURCE_STAGE_MAPPING.json` binds these sources.

The complete statistics category is also not integer-only: its corrector uses
ordered float64 products and square roots. The visible serial bigint loop is
Lane geometry, whose native precedent uses `__int128`. A direct torch int64
translation needs proved bounds or split arithmetic. This is a source/arithmetic
mismatch in the proposed implementation, not a FAMILY rejection of native C or
device-side integer work. No Route A or B ships; no third native file was added.

## Measured profile and exact identities

All seconds in this section are **[macOS-CPU advisory]**, seed **20260916**,
two random pairs in each of 24 consecutive 25-pair strata: **48/600 pairs**.
Setup, checkpoint loading and payload writes are excluded from pair timings.
There is one observation per selected pair and no measured noise floor.

| Stage | Historical mrs2 range-C profile | New unchanged-source diagnostic |
|---|---:|---:|
| Arithmetic decoder + frequency table | 0.371585 | 0.382488 |
| Adaptive corrector + Lane mixer, named statistics | 53.905538 | 53.530976 |
| Prior network | 42.730146 | 44.270533 |
| Renderer | 13.329563 | 14.092252 |
| Pose carrier + selector | 0.369977 | 0.378695 |
| Other token work | 1.617714 | 1.933911 |
| Total selected pair work | 112.324523 | 114.588856 |

**This is not a before/after cure comparison.** The source is unchanged and the
new diagnostic adds nested timers and explicitly enables deterministic CPU
algorithms. These instrument differences preclude a controlled speedup claim.
The original all-Python mrs2 statistics measurement was 52.385812 s; removing
range arithmetic had left that category essentially intact.

The new nested attribution finds:

| Actual surface | Seconds | Meaning |
|---|---:|---|
| Adaptive corrector, all outer calls | 40.732609 | About 76.1% of the broad statistics category |
| Corrector observe | 21.600729 | Included in the preceding row |
| Corrector coding row | 16.380959 | Included in the preceding row |
| Lane geometry context calculation | 4.288077 | 8.01% of broad statistics |
| Entire Lane geometry object | 4.703660 | 8.79%; init + contexts + observe |

Nested rows overlap with their parent timers and must not be summed. Even
granting **zero time to all Lane geometry**, the statistics category could
improve only **1.09633x** in this measured split, far below the charter's **10x**.
Removing only the corrector altogether has a **4.18264x** category ceiling;
meeting 10x requires work on the remaining mixer cost too. These are derived
within-run ceilings, not implementation or T4 measurements.

**Identity: 48/48 distinct expected pairs; 9,437,184 token bytes and 292,992,768
raw bytes identical.** Every retained token/raw payload and every pre-pair
checkpoint was independently rehashed after completion. The expected frame set,
state identity and oracle hashes were verified independently of resume receipt
reuse. Evidence: `DIAGNOSTIC_VERIFICATION.json` and
`/Volumes/APDataStore/pact/ddm_mrs3/stage_profile/RESULT.json`.

## Charter gates and timing projections

| Required gate | This arm's status |
|---|---|
| Cured-stage C/Python or torch/Python parity | NOT RUN: no cure implemented; diagnostic source-copy identity is 48/48 |
| Full cold n600 public parse-back | NOT RUN in mrs3; predecessor mrs2 has 600/600 and full raw SHA `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b` |
| Compiler-present bare-venv smoke | NOT RUN in mrs3; predecessor success is historical |
| Compiler-absent bare-venv smoke | NOT RUN in mrs3; predecessor 48/48 success is historical |
| Statistics at least 10x faster or moved onto device | NOT MET; no implementation measured |
| Both ratio projections at most 1,200 s | NOT MET for the unchanged mrs2 source |
| Stage-model projection at most 1,200 s | UNRESOLVED: isolated T4 prior-network share is absent |

The two existing whole-process projections remain **1,699.130670 s** using
`1760.9 / 1390.8`, and **1,699.116793 s** using
`1185.899645171 / 936.6589231670368`. Both use mrs2's historical **1,342.013139 s**
cold host process. They assume a whole-process transfer ratio survives the
hardware/work-distribution change. They are rounded/full-precision forms of
**one assumption**, not independent calibrations or a new mrs3 timing result.

The requested third model cannot be responsibly reduced to one measured-backed
number. Under the stated stratified scaling and mrs2 ratio, this diagnostic gives
`(arithmetic + statistics + other token work) * 600/48 * ratio = 883.851385 s`.
Adding the original T4 renderer's **41.454240 s** gives **925.305625 s**, **plus
unknown T4 prior-network time and setup/I/O/other work**. This is a conditional
partial model, not a passing projection or a hard bound on real T4 runtime.

The original T4 receipt measured combined token work **1,114.794542 s**, renderer
**41.454240 s**, selector/I/O **3.504575 s**, archive setup **8.727889 s**, and outer
inflate **1,185.899645 s**. It does not isolate the GPU prior-network share.
`/Volumes/APDataStore/pact/ddm_mrs3/ORIGINAL_T4_STAGE_EXTRACTION.json` binds the
actual receipt and JSON location. MAIN owns any new T4 timing and exact row.

## Reviewer budget and retained source

| Current public-copy file | Physical LOC | Bytes |
|---|---:|---:|
| inflate.sh | 6 | 197 |
| inflate.py | 2,715 | 138,152 |
| range_decoder.c | 100 | 4,018 |
| README.md | 24 | 1,559 |
| archive.zip | binary | 179,286 |

Five files are present, all byte-identical to mrs2. `context_stats.c` is absent;
new C/torch cure LOC = **0**, which is incompleteness, not a budget win.
`SOURCE_MANIFEST.json` carries all hashes. mrs2 provenance is landing `2171aaf9b`,
Python SHA `0689a77e061fe8a7d39953adb099c0503b8b9f2d983aacb7a98fc03d304e1425`;
move-53 pointer provenance is `a91a7dde3`. The nominated sealed C's exact SHA is
in `SOURCE_STAGE_MAPPING.json`.

Fresh-reader prompt for the actual current source copy:
`.omx/research/ddm_mrs3_20260916/FRESH_READER_PROMPT.md`. No reader of a cured
receiver was run. `PR_BODY_DRAFT.md` is 20 lines, carries the archived report,
and leaves `GPU: yes, N s on T4 (MAIN measurement pending)` explicitly unresolved.
It is not ready to publish.

The diagnostic runner and unchanged copied Python each received two review
tracker passes without an override. An independent source/harness review found
no blocker for the diagnostic, with the instrumentation caveat above and a
resume-receipt relabeling gap. The latter was excluded for this actual result by
the independent 48-frame/state/oracle audit; the harness is not claimed hardened
against deliberate receipt tampering. `REVIEW_PASSES.json` records that scope.

## Custody and boundaries

Bulk root: `/Volumes/APDataStore/pact/ddm_mrs3/`. Every materialized token/raw
payload remains retained, with per-file hashes. `RETENTION_SUMMARY.json` gives
the final byte count against the **2 GiB** cap. Source delivery is
`/Volumes/APDataStore/pact/ddm_mrs3/mrs3_source_delivery.tar`, with member hashes
and a verified external `SOURCE_DELIVERY.json`. The serializer's exact argv,
post-edit hashes, result code and landing state are in `serializer/RESULT.json`;
its result, not the presence of files, determines whether anything landed.
The charter's rc 17/19 source-tar fallback is honored.

The detached 48-pair run used `tools/launch_detached_process.py --done-receipt`;
all 48 pair stages completed and can be verified from disk. The done receipt,
source snapshots, library build command and immutable inputs are retained.
No APFS/ExFAT venv was created. The diagnostic library is reproducible build
output with a retained build certificate; it is outside the public file copy.

**60 immutable source/archive files were rehashed unchanged**, covering mrs1,
mrs2 and the sealed tree. The shared index is unchanged and the three common
contract protected paths are clean. `upstream/`, the closed PR tree and sealed
trees were not modified. No Modal, fire, packet, PR, push, `authorize_*`, scorer,
GT decode, MPS authority, stash, direct index manipulation, dependency pin,
environment switch in the public source, OpenMP/native flag, third C file,
co-author/AI trailer, or public `.pyc`/`.so` was added. No scorer slot was claimed,
no SSD permission denial occurred, and unrelated dirty work was preserved.

## RECALL EVIDENCE

`RECALL_EVIDENCE.md` records the parent and independent queries, full-corpus
scope, beyond-seed findings and their consequences. The 490-record equation
export and content search outputs are retained. The core changed premise is
the actual hot stage and its arithmetic, not a new compression law.
All six integration-hook dispositions are recorded there; this work is
research-only and produces no promotable receiver or score.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/research/ddm_mrs3_20260916/MAIN_FIRE_ORDER.json`; trigger: harvest the source map and completed 48-pair attribution.** Correct the explicit native-source/arithmetic scope, retaining the reviewer, identity and dispatch constraints.
- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; same consumer; trigger: harvest the verified source tar and serializer result.** Land this diagnostic evidence with its partial status; do not label the unchanged public copy a cure.
- **FOLDED into the source-scope correction; owner MAIN; same consumer; trigger: explicit authorization of the corrected scope.** Implement the actual hot-stage cure and complete its parity, cold n600, both smokes, three projections, reader and MAIN-owned T4 gates.

## LIVE-HYPOTHESES

- A bounded lowering of the adaptive corrector plus remaining mixer work could
  meet the timing target: those surfaces account for nearly all measured
  statistics time and have sealed reference implementations. Its performance,
  float/wide-integer parity and 600-line reviewer fit remain untested here.

## DEAD-ENDS

- **SOURCE-MAPPING INSTANCE:** porting `f26_hpac_native.c` as the 45% statistics
  cure. The profiler assigns it to the prior-network stage, and the successful
  sealed receiver explicitly refuses that native path.
- **THIS-INSTANCE TIMING:** Lane geometry alone as a 10x statistics cure. Even
  zero-time geometry permits only 1.09633x in the measured 48-pair split.
- **UNJUSTIFIED EQUIVALENCE CLAIM:** translating all statistics into plain int64.
  The corrector uses float64 and geometry needs wider products; a naive port
  does not preserve those semantics. The broader algorithm families remain open.
- **UNSUPPORTED PROJECTION CLAIM:** treating the two ratios as independent or
  inventing a measured GPU prior share. The original receipt does neither.

composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)
