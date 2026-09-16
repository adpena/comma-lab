# ddm_mrs4 — the minimal packet's T4 timing cure, corrected scope: the ADAPTIVE CORRECTOR + Lane-mixer statistics as the last native file, float64-exact (charter, MAIN 2026-09-16; codex astra xhigh; scorer-free)

## Why (mrs3 handoff `.omx/research/ddm_mrs3_20260916/HANDOFF.md`, landed 34bbf7842 — MAIN's mrs3 charter named the wrong source)
mrs3's 48-pair diagnostic [macOS-CPU advisory]: arithmetic 0.38 s; **corrector + mixer statistics 53.5 s** (adaptive
corrector 40.7 s; all Lane geometry 4.7 s); prior network 44.3 s; renderer 14.1 s; pose/selector 0.4 s. The stage that
is serial Python on T4 is the adaptive corrector, not the HPAC prior (which the sealed receiver explicitly refuses
natively). The original receiver shipped exactly this as `runtime/f26_corrector_native.c` (1,248 lines, float64,
compiled with `-ffp-contract=off -fno-fast-math`, Python fallback) and decoded in 1,185.9 s on T4. The minimal packet
projects 1,699 s without it. Identity of the current packet is proven (48/48; 600/600 raws vs move 53).

## Deliverable — `submissions/mrs4/` (start from a COPY of `submissions/mrs2/`; mrs1–mrs3 stay as landed)
1. Vendor ONE more C file, `corrector.c`, plain C11, ≤ 700 lines, derived from the sealed tree's
   `runtime/f26_corrector_native.c` (read-only: `/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime/runtime/f26_corrector_native.c`)
   covering the adaptive corrector and, if it shares the loop, the Lane-mixer statistics. Float64 semantics must be
   BIT-EXACT with the Python path: compile with `-O2 -std=c11 -ffp-contract=off -fno-fast-math` (no OpenMP, no
   `-march=native`); no reassociation of sums; the same evaluation order as the Python reference. ctypes-loaded; the
   Python path stays as the fallback with one clear stderr line. inflate.sh ≤ 8 lines (two one-line compiles).
2. Proofs, in order: (a) bit-identity of corrector outputs and tokens vs the Python path on ≥ 48 stratified pairs
   (tokens AND raw bytes); (b) the full cold n600 public parse-back → raws byte-identical to move 53 (600/600);
   (c) bare-venv smokes with a compiler present and absent (fallback identical); (d) the 48-pair profile again: the
   corrector + mixer stage must fall ≥ 10× on the host; (e) T4 projections by mrs2's two ratios AND the stage model
   (serial-Python stages × ratio + GPU stages bounded by the original 1,185.9 s receipt) — every assumption stated; the
   handoff gate is projection ≤ 1,200 s. If it misses, report where the seconds are; no third native file without MAIN.
3. Reviewer budget: ≤ 6 files (inflate.sh, inflate.py, range_decoder.c, corrector.c, README.md ≤ 30 lines, archive.zip
   unchanged 179,286 B sha aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957); no .pyc, no internal names,
   no environment switches, no pins, no OpenMP/native flags. Fresh-reader prompt for MAIN. PR draft ≤ 20 lines (T4 line
   stays a placeholder until MAIN measures).

## Boundaries
As mrs2/mrs3: no Modal, fire, packet, PR, push, authorize_*; upstream/, the closed PR tree, sealed trees, mrs1–mrs3
read-only; work under `/Volumes/APDataStore/pact/ddm_mrs4/`; scratch venvs on an APFS path (`.omx/tmp/`), never on
ExFAT; retain ≤ 2 GiB with sha256; heavy steps via `tools/launch_detached_process.py --done-receipt`; serializer commits,
two review passes per .py, `[no-triality] [p0-ledger-ok]`, never a co-author trailer or AI attribution; rc 17/19 is NOT a
stop — continue, deliver `/Volumes/APDataStore/pact/ddm_mrs4/mrs4_source_delivery.tar`, MAIN lands. Checkpoint
`ddm_mrs4`; lane `ddm_mrs4_minimal_packet_adaptive_corrector_native_20260916`.

## OPTIMAL FORM
Reference forms: mrs2's packet + proofs; the sealed `f26_corrector_native.c` (the arithmetic and its exact float64
evaluation order — reuse it, drop OpenMP and instrumentation); mrs2/mrs3's profiler. Declared deltas: the one native
file this charter names; nothing else. Provenance pins: mrs3 landing 34bbf7842; mrs2 landing 2171aaf9b;
`f26_corrector_native.c` sha 3e2705f550503612; move 53 packet a91a7dde3.

## Prior negatives accounted (operator 2026-08-15)
PR #140 (over-engineering — this is the LAST native file; ≤ 700 plain lines); mrs2 (range decoder alone: 4 % of the
work); mrs3 (MAIN named the wrong stage — hence the profile-named stage, by category and by the sealed source that
implemented it; naive int64 is wrong: the corrector is float64); tc4 (T4 wall clock measured by MAIN); the ExFAT venv.

Final message: profile before/after by stage, corrector.c LOC, bit-identity counts, 600/600 raws, both smokes, the
three T4 projections with assumptions, fresh-reader prompt path, the ≤ 6-file list with LOC, retained bytes, every
boundary, serializer rc, ending with `composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)`.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
