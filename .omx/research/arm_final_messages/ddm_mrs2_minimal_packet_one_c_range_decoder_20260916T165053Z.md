Implemented mrs2 and completed the local proofs. **Identity passes; the timing target fails.** Full details are in the [handoff](/Users/adpena/Projects/pact/.omx/research/ddm_mrs2_20260916/HANDOFF.md).

The measured 48-pair profile, `[macOS-CPU advisory]`:

| Stage | Python seconds | C-path seconds |
|---|---:|---:|
| Arithmetic + frequency table | 5.032 | 0.372 |
| Prior context statistics | 52.386 | 53.906 |
| Prior network | 42.773 | 42.730 |
| Renderer | 13.464 | 13.330 |
| Pose carrier + selector | 0.372 | 0.370 |
| Other token work | 1.599 | 1.618 |

Arithmetic represented only **4.35%** of baseline pair work; context statistics and the prior network represented **82.30%**. These are single observations, without a measured noise floor.

- **C/Python identity:** 48/48 stratified pairs, 9,437,184 tokens and 292,992,768 raw bytes match.
- **Cold public decode:** 600/600 raw pair hashes match move 53.
- **Bare smokes:** compiler-present n600 passed; compiler-hidden fallback passed 48/48 pairs, including stale-library removal and the clear fallback message.
- **Cold host time:** 1,342.013139 seconds `[macOS-CPU advisory]`.
- **Projected T4:** **1,699.130670 seconds** using `1760.9/1390.8`; **1,699.116793 seconds** using the original measured ratio. Both exceed 1,200 seconds and the 1,260-second risk ceiling. Neither is a T4 measurement.

The five public files are `inflate.py` **2,715 LOC**, `inflate.sh` **6 LOC**, `range_decoder.c` **100 LOC**, `README.md` **24 LOC**, and unchanged `archive.zip` **179,286 B**. C11 checks and both review passes per Python file passed. No public binaries, `.pyc`, internal names, environment switches, pins, OpenMP or native-architecture flags were added.

[Fresh-reader prompt](/Users/adpena/Projects/pact/.omx/research/ddm_mrs2_20260916/FRESH_READER_PROMPT.md) is ready for MAIN; no reader result is claimed. The inherited PR draft remains unchanged at 20 lines.

**Serializer rc 19:** Git object writes were denied; its bundle fallback failed the storage reserve. HEAD and index remained unchanged. The verified [source delivery tar](/Volumes/APDataStore/pact/ddm_mrs2/mrs2_source_delivery.tar) contains 45 members, including the archive and custody receipts. **No commit landed.** Retention totals **1,249,306,930 bytes**, including filesystem metadata, below 2 GiB. Payloads remain retained; full-raw and APFS-venv cleanup were certified.

mrs1, upstream, closed PR, sealed trees and protected paths remain unchanged. No Modal, fire, packet tool, PR, push, authorization call, scorer, GT decode, MPS authority, stash, direct index manipulation or attribution trailer was used. Heavy jobs used detached receipts. No CUDA/T4 execution, Python 3.11 execution or new score was measured.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer `MAIN_FIRE_ORDER.json` in the handoff directory; trigger: handoff harvest.** Verify the source tar and land the owned files.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; same consumer `MAIN_FIRE_ORDER.json`; trigger: source custody and identity-proof harvest.** Run the fresh-reader prompt and retain its reply.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; same consumer `MAIN_FIRE_ORDER.json`; trigger: failed timing-screen harvest.** Commission the bounded context-statistics successor.
- **FOLDED into that successor — owner MAIN; same consumer `MAIN_FIRE_ORDER.json`; trigger: both projections ≤1,200 seconds, identity and reader gates passed, exclusive lane and required authorization obtained.** Measure T4 timing and a fresh exact n600 row.

## LIVE-HYPOTHESES

- Optimizing prior-context computation could materially reduce runtime: it consumes 45.31% of baseline pair work. A byte-identical implementation and its T4 benefit remain untested.

## DEAD-ENDS

- **This range-decoder-only timing cure:** arithmetic became much faster, but both projections remain about 1,699 seconds.
- **Dispatching or transferring the old score to this receiver:** the timing screen fails, and raw identity does not replace a fresh exact evaluation.

composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)

verdict_scope: instance — the DEAD-ENDS above are mrs2 instance findings (a range-decoder-only C file does not cure the T4 projection: arithmetic was 4.35 % of pair work; the prior context statistics + prior network were 82.3 %). Identity 600/600 holds. No formulation closed; the context-statistics cure (mrs3) is the successor.
