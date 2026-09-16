# ddm_mrs3 — the minimal packet's T4 timing cure: the prior's CONTEXT STATISTICS as the second (and last) native file, or a byte-identical vectorized torch path (charter, MAIN 2026-09-16; codex astra xhigh; scorer-free)

## Why (mrs2 handoff `.omx/research/ddm_mrs2_20260916/HANDOFF.md`, landed 2171aaf9b)
mrs2 measured the 48-pair host profile of the minimal receiver: arithmetic + frequency table 5.0 s → 0.37 s with the
vendored range decoder (4.35 % of pair work), prior CONTEXT STATISTICS 52.4 s (45.3 %), prior network 42.8 s (37.0 %),
renderer 13.5 s, pose carrier 0.4 s. Cold host decode 1,342 s; projected T4 1,699 s (both ratios) — still above our
1,260 s ceiling (the original receiver: 1,185.9 s on T4 with `f26_hpac_native.c`, 994 lines with OpenMP, doing exactly
this context-statistics loop natively). On T4 the prior network and renderer run on the GPU; the serial Python context
loop is the T4 wall clock. Identity is already proven (C/Python 48/48; raws 600/600 vs move 53). This arm cures the
one remaining stage under the reviewer budget.

## Deliverable — `submissions/mrs3/` (start from a COPY of `submissions/mrs2/`; mrs1/mrs2 stay as landed)
Route A (default): vendor the context-statistics loop as ONE more C file, `context_stats.c`, plain C11, no OpenMP, no
`-march=native`, ≤ 600 lines, derived from the sealed tree's `runtime/f26_hpac_native.c` arithmetic (read-only:
`/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime/runtime/f26_hpac_native.c`) with the SAME integer
semantics as the Python path; ctypes-loaded; Python fallback kept. inflate.sh compiles both C files on one line each.
Route B (run in the same arm if A's projection misses, or if the loop is naturally vectorizable): a byte-identical
torch integer implementation of the context statistics that runs on the evaluator's device (int64 ops; no float in
the statistics; determinism checked on CPU here and reasoned for CUDA — state the reasoning; the T4 measurement is
MAIN's). Report which route ships and why.
Proofs, in order: (1) per-pair bit-identity of the cured stage vs the Python path on ≥ 48 stratified pairs (tokens and
raw bytes); (2) the full cold n600 public parse-back → raws byte-identical to move 53 (600/600); (3) bare-venv smokes
with a compiler present and absent (fallback decodes identically); (4) the 48-pair profile again — the context stage
must fall by ≥ 10× on the host (C) or move onto the device (torch); (5) T4 projection by BOTH of mrs2's ratios AND by a
stage model: (T4 = measured serial-Python stages on host × the mrs2 ratio) + (GPU stages ≈ the original receiver's
measured share) — state every assumption; the handoff gate is projection ≤ 1,200 s. If it misses, report where the
seconds are; do not add a third native file without MAIN.
Reviewer budget stays the gate: ≤ 6 files (inflate.sh ≤ 8 lines, inflate.py, range_decoder.c, context_stats.c,
README.md ≤ 30 lines, archive.zip unchanged 179,286 B sha aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957);
no .pyc, no internal names, no environment switches, no pins, no OpenMP/native flags. Fresh-reader prompt written for
MAIN. PR draft ≤ 20 lines (carry mrs1's; the "GPU: yes, N s on T4" line stays a placeholder until MAIN measures).

## Boundaries
As mrs2's charter: no Modal, fire, packet, PR, push, authorize_*; upstream/, the closed PR tree, sealed trees, mrs1/mrs2
read-only; work under `/Volumes/APDataStore/pact/ddm_mrs3/`; scratch venvs on an APFS path (`.omx/tmp/`), never on
ExFAT; retain ≤ 2 GiB with sha256; heavy steps via `tools/launch_detached_process.py --done-receipt`; serializer
commits, two review passes per .py, `[no-triality] [p0-ledger-ok]`, never a co-author trailer or AI attribution; rc 17/19
is NOT a stop — continue, deliver `/Volumes/APDataStore/pact/ddm_mrs3/mrs3_source_delivery.tar`, MAIN lands.
Checkpoint `ddm_mrs3`; lane `ddm_mrs3_minimal_packet_context_statistics_cure_20260916`.

## OPTIMAL FORM
Reference forms: mrs2's packet (behaviour, identity proofs, smokes), the sealed tree's `f26_hpac_native.c` (the
arithmetic of the loop), mrs2's profiler. Declared deltas: one more native file OR a device-side integer path — the
charter's point; nothing else changes. Provenance pins: mrs2 landing 2171aaf9b; mrs2 inflate.py sha (record);
`f26_hpac_native.c` sha (record); move 53 packet a91a7dde3.

## Prior negatives accounted (operator 2026-08-15)
PR #140 (over-engineering: this is the LAST native file; plain flags; ≤ 600 lines); mrs2 (the range decoder alone does
not cure T4 — profile first, then vendor the stage the profile names); tc4 (T4 wall clock is MEASURED by MAIN);
the ExFAT venv instance; the pd6 rider-drop law (the archive is copied, never re-staged).

Final message: the profile before/after by stage, the C file's LOC (or the torch path's), bit-identity counts, 600/600
raws, both smokes, the three T4 projections with assumptions, the fresh-reader prompt path, the ≤ 6-file list with LOC,
retained bytes, every boundary, serializer rc, ending with
`composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)` unchanged.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
