# ddm_mrs2 — the minimal packet with ONE vendored C range decoder, to bring the T4 decode back under the 1,260 s ceiling (charter, MAIN 2026-09-16; codex astra xhigh; scorer-free)

## Why
mrs1 (landed c9b4b9b75; `submissions/mrs1/`; handoff `.omx/research/ddm_mrs1_20260916/HANDOFF.md`) produced the reviewable
packet: inflate.sh 3 lines, inflate.py 2,693 LOC pure Python, README 22 lines, archive unchanged (179,286 B, sha
aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957); raw identity 600/600 with move 53; bare-venv public smoke
passed; fresh reader: "readable but densely organized". Its decode is C-free: 1,390.8 s on the host [macOS-CPU advisory],
PROJECTED 1,760.9 s on T4 — 39 s under the contest's 1,800 s budget and ABOVE our own 1,260 s risk ceiling (the contract
refuses to authorize a fire above it; memory `receiver_change_needs_a_measured_decode_wall_clock_tc4_…`). The original
receiver decoded in 1,185.9 s with native C. mrs1's own live hypothesis: one native range decoder recovers the time. A
single vendored C file with a one-line compile is still reviewable (PR103 silver used a compiled range coder).

## Deliverable — `submissions/mrs2/` (start from a COPY of `submissions/mrs1/`; mrs1 stays as landed)
1. Profile the host decode of `submissions/mrs1/inflate.py` (cProfile or per-stage timers on ≥ 48 seeded pairs, retained)
   and report the time split: arithmetic decoding of the token field vs the prior's context statistics vs the renderer vs
   the pose carrier. MEASURED, not assumed.
2. Vendor exactly ONE C file, `range_decoder.c` (≤ 400 lines, plain C11, no OpenMP, no -march=native), implementing only
   the hot loop the profile names (expected: the five-symbol 63-bit arithmetic decoder + its frequency-table update),
   loaded via ctypes from inflate.py; keep the pure-Python path as the fallback when the compile fails (one clear line to
   stderr). inflate.sh becomes ≤ 6 lines: `cc -O2 -std=c11 -shared -fPIC range_decoder.c -o range_decoder.so || true`
   then exec python. README gets one sentence: what the C file is and that Python decodes identically without it.
3. Prove BIT-IDENTICAL decoding: C path vs Python path on ≥ 48 pairs, then the full cold n600 public parse-back with
   the C path → raws byte-identical to move 53's retained raws (600/600; shas in the pd6 memo / seal). Bare-venv smoke
   with a compiler present AND absent (the fallback must decode identically). Fresh-reader test again (write the prompt;
   MAIN spawns the reader). PR body draft ≤ 20 lines (carry mrs1's; add nothing).
4. Timing: host decode with the C path [advisory]; project T4 by the SAME ratio mrs1 used (1,390.8 → 1,760.9 s; state
   it) and by the original receiver's measured ratio; the projection must be ≤ 1,200 s to hand to MAIN for the T4
   measurement. Report both projections; if the C path does not reach it, say so and report where the time is.
5. Reviewer budget stays the gate: ≤ 5 files (inflate.sh, inflate.py, range_decoder.c, README.md, archive.zip), no
   .pyc, no internal names, no environment switches, no pins; tidy the duplicated import block at the bottom of mrs1's
   inflate.py while you are there (identical behaviour, proven by raw identity).

## Boundaries
As mrs1's charter: no Modal, fire, packet, PR, push, authorize_*; upstream/, the closed PR tree, sealed trees read-only;
work under `/Volumes/APDataStore/pact/ddm_mrs2/`; venvs for bare-environment smokes on an APFS path (`.omx/tmp/`), NEVER
on the ExFAT tier (mrs1's ensurepip SIGABRT); retain ≤ 2 GiB with sha256; heavy steps via
`tools/launch_detached_process.py --done-receipt`; serializer commits, two review passes per .py, `[no-triality]
[p0-ledger-ok]`, never a co-author trailer or AI attribution; rc 17/19 is NOT a stop — continue, bundle
(`/Volumes/APDataStore/pact/ddm_mrs2/mrs2_source_delivery.tar` if the git bundle needs space), MAIN lands.
Checkpoint `ddm_mrs2`; lane `ddm_mrs2_minimal_packet_one_c_range_decoder_20260916`.

## OPTIMAL FORM
Reference forms: mrs1's packet (behaviour), the sealed tree's `runtime/entropy/rc64_backend.c` (the original native
decoder — read it; reuse its arithmetic, not its build flags), PR103's compiled-coder precedent. Declared deltas: one C
file (the charter's point); nothing else changes. Provenance pins: mrs1 landing c9b4b9b75; mrs1 inflate.py sha
150f11328b7dfd2f; move 53 packet a91a7dde3; archive sha above.

## Prior negatives accounted (operator 2026-08-15)
PR #140 (over-engineering — hence ONE file, ≤ 400 lines, plain flags); mrs1 (C-free too slow for the ceiling); tc4 (the
T4 wall-clock must be MEASURED by MAIN — projections only rank); the ExFAT venv instance.

Final message: the profile split, the C file's LOC, bit-identity counts (pairs, then 600/600 raws), both smokes, both T4
projections, the fresh-reader prompt path, the file list with LOC, retained bytes, every boundary, serializer rc, ending
with `composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)` unchanged.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
