Implemented and delivered; **identity passed, timing gates failed**. No new T4 measurement or score.

[Handoff](/Users/adpena/Projects/pact/.omx/research/ddm_mrs4_20260916/HANDOFF.md) · [Verified source tar](/Volumes/APDataStore/pact/ddm_mrs4/mrs4_source_delivery.tar) · [Fresh-reader prompt](/Users/adpena/Projects/pact/.omx/research/ddm_mrs4_20260916/FRESH_READER_PROMPT.md)

Matched 48-pair profile, **[macOS-CPU advisory]**, seconds:

| Stage | Python | Native |
|---|---:|---:|
| Arithmetic/frequency table | 0.650 | 0.435 |
| Corrector + mixer | 82.543 | 25.175 |
| Prior network | 82.386 | 77.273 |
| Renderer | 25.591 | 24.973 |
| Pose/selector | 0.511 | 0.490 |
| Other token work | 3.902 | 3.386 |
| Total | 195.582 | 131.733 |

Corrector/mixer improved **3.279×**, short of **10×**. Remaining nested costs include **17.391 s Lane mixer**, including **7.506 s geometry**, and **7.878 s corrector**. Nested timings overlap. Pair timings exclude setup, state loading and writes; these sequential shared-host runs have no measured noise floor.

Proofs passed:

- **48/48** stratified token/raw pairs; **9,120** corrector groups, **3,792** state-array comparisons and **27,648** mixer checks.
- Independent retained-vector audit: **28,311,552 probability rows**.
- Cold public decode: **600/600 raw pairs**, full-file hash identical to move 53; **1,474.011 s**.
- Compiler-present bare smoke passed; compiler-absent smoke removed stale libraries and matched **48/48** fallback pairs.

T4 projections are **1,866.254 s**, **1,866.239 s**, and a conditional stage upper estimate of **1,644.801 s**. None establishes the **1,200 s** gate. The first two assume the old host/T4 ratio transfers and are not independent calibrations. The stage model assumes the sampled mean scales to 600, applies that ratio to serial CPU work, transfers old non-token costs, and bounds GPU prior work by the original mixed token stage. Its conditional range is **530.006–1,644.801 s**; actual T4 runtime remains unknown.

Six public files: `inflate.sh` **7 LOC**, `inflate.py` **2,831**, `range_decoder.c` **100**, `corrector.c` **693**, `README.md` **24**, unchanged `archive.zip` **179,286 B**. C11 uses the required float flags and two compile commands. All five Python files received two recorded self-review passes.

Retained: **1,894,587,879 B**, allocated **2,080,112,640 B**, both below 2 GiB, with hashes. Certified cold-raw scratch and the inventoried APFS venv were cleaned up. Heavy steps used detached launches with receipts.

**Serializer rc 19; uncommitted.** Git object writes were denied and fallback storage failed its 40 GiB reserve. The verified tar contains **41 members**, including all six public files, for MAIN to land.

Boundaries held: 65 captured immutable files unchanged; upstream, sealed/closed-PR trees, mrs1–3 and protected contract files untouched; shared index preserved, no stash. No Modal, fire, packet, PR, push, `authorize_*`, scorer, GT decode, MPS, third C file, dependency pins, public environment switches, OpenMP/native flags, public generated files, review override or co-author trailer.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer `.omx/state/canonical_task_status.jsonl`, `ddm_mrs4_source_landing_20260916`; trigger: verified tar and serializer harvest.** Land the delivered sources and evidence.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; same consumer, `ddm_mrs4_fresh_reader_20260916`; trigger: exact sources under MAIN custody.** Execute the fresh-reader prompt.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; same consumer, `ddm_mrs4_timing_gate_resolution_20260916`; trigger: harvest failed timing gates.** Resolve remaining costs within the two-C/700-line limits or explicitly revise scope.

**LIVE-HYPOTHESES**

- More Lane-mixer lowering may help: it now dominates statistics time. Exact arithmetic and reviewer-budget fit remain untested.
- Reusing corrector context work may reduce its remaining cost. Both leads are folded into MAIN’s timing-resolution task.

**DEAD-ENDS**

- This port as a completed timing cure: only 3.279×, with all projection screens unmet.
- Corrector-only tuning as the complete 10× cure: even zero corrector time leaves too much mixer work.
- Native HPAC targets the wrong stage; naive int64 does not preserve the required arithmetic.
- The two ratios are not independent evidence, and local identity does not transfer contest-score authority.

composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)

verdict_scope: instance — the DEAD-ENDS above are mrs4 instance findings (corrector port alone 3.28× on its stage; host-ratio projections are not independent evidence and were measured on a loaded host). Identity 600/600 holds. No formulation closed; mrs5 adds the original geometry native file and measures the same-host cost ratio for the contract risk receipt.
