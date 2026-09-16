# ddm_mrs5 — the minimal packet at the ORIGINAL measured native set (+ geometry.c, 130 lines), a same-host interleaved cost ratio vs the sealed receiver, and the committed PRE-FIRE INTENT for one T4 measurement (charter, MAIN 2026-09-16; codex astra xhigh; scorer-free)

## Why (mrs4 handoff `.omx/research/ddm_mrs4_20260916/HANDOFF.md`, landed e621fd5a9)
mrs4's packet (`submissions/mrs4/`: sh 7 / py 2,831 / range_decoder.c 100 / corrector.c 693 / README 24 / archive unchanged)
is byte-identical to move 53 on all 600 pairs. Its T4 projections (1,866 s by host ratios; stage-model range 530–1,645 s)
are NOT evidence: every host timing ran while pd7 loads the same machine, and host→T4 ratios are not independent
calibrations. The decisive fact is in the sealed receiver's own `inflate.sh`: the T4 run that measured 1,185.9 s
(move 53's t4_direct leg) used exactly `rc64_backend.c` + `f26_corrector_native.c` + `rlc1_geometry.c` natively with
`F26_TOKEN_DECODER=python` (native HPAC OFF). mrs4 reproduces two of those three. Adding the 130-line geometry file
reproduces the MEASURED native configuration; the remaining question is the cost RATIO of the rewritten Python vs the
sealed Python on identical hardware, which the contract's timing-risk receipt is designed to carry (pr12:
`candidate_prefire_timing_risk.v1`, lineage to a completed t4_direct leg + the measured local cost fraction of the delta;
rlc1/rlc5 precedent: `.omx/research/ddm_rlc1_20260910/QUIESCED_TIMING_RECORD.json`).

## Deliverable
1. `submissions/mrs5/` = COPY of `submissions/mrs4/` + `geometry.c` (plain C11 port of the sealed
   `runtime/rlc1_geometry.c`, ≤ 150 lines, same arithmetic and evaluation order, compiled with the same float flags as
   corrector.c; ctypes; Python fallback kept). inflate.sh ≤ 9 lines (three one-line compiles). README ≤ 30 lines (one
   sentence per C file; the three together are the receiver's measured native set). Seven files; no .pyc, no internal
   names, no environment switches, no pins, no OpenMP/native flags. MAIN accepts the third native file because it is the
   measured configuration, not an addition.
2. Proofs: bit-identity vs the Python path on ≥ 48 stratified pairs (tokens + raws); the full cold n600 public
   parse-back → 600/600 raws identical to move 53; bare-venv smokes with a compiler present and absent.
3. **Same-host interleaved cost ratio (the instrument).** On the SAME 48 stratified pairs, alternate A = the sealed
   receiver (`/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime`, its own inflate.sh with its native builds,
   `RLC1_ADVISORY_CPU=1`) and B = mrs5, A/B/A/B (four runs each ≥ 48 pairs, cold per run), same threads, same device
   policy; record wall clock per run and the host load (`uptime` 1-min load before/after each run — pd7 is running; the
   interleaving makes the RATIO valid under stationary load, the absolute times are not evidence). Report
   ratio = median(B)/median(A) with its spread. Projected T4 = 1,185.9 s × ratio; state the assumption (serial-Python
   scaling; the GPU stages are unchanged code).
4. **The pre-fire intent** (pr12's contract as landed a47543199; read `src/tac/candidate_seal.py` ~1600–2560 and
   `tools/make_candidate_seal.py` first): build the timing-risk receipt `candidate_prefire_timing_risk.v1` with lineage to
   move 53's completed leg (`/Volumes/APDataStore/pact/ddm_pd6/SEAL_ddm_pd6_price_first_generator_contest_cuda.json.decode_wall_clock.json`,
   1,185.9 s, receiver behaviour 9f6e7168…), the delta = the receiver rewrite (list the seven files with sha256), the
   measured ratio from step 3, risk ceiling ≤ 1,260 s (if the projection exceeds it, STOP at this step and report — MAIN
   decides; do not fabricate a lower ratio); then emit and self-validate the intent with
   `tools/make_candidate_seal.py --first-fire-intent … --timing-risk-evidence …` on `submissions/mrs5` as the runtime dir
   (archive identical to the pointer: this intent is a RECEIVER-ONLY change at the pointer's bytes; every non-timing gate:
   archive bytes + digest, runtime + normalized receiver digests, manifest, current pointer move 53, smokes, retained
   payload paths, `score_claim=false`); prove the normal `validate_seal` and `--seal` path REFUSE it (record the typed
   refusal). If the intent producer refuses your real intent for a reason you believe is a contract defect, STOP with the
   exact refusal — that is the contract's real control; do not patch around it.
5. Fresh-reader prompt for MAIN; PR draft ≤ 20 lines (carry mrs1's; T4 line placeholder).

## Boundaries
As mrs4: no Modal, fire, packet, PR, push, `authorize_candidate_first_measurement.py`, first-measurement fire, completion
(all MAIN's); upstream/, the closed PR tree, sealed trees, mrs1–mrs4, contract code (`candidate_seal.py`,
`decode_wall_clock.py`) read-only; work under `/Volumes/APDataStore/pact/ddm_mrs5/`; scratch venvs on APFS
(`.omx/tmp/`); retain ≤ 2 GiB with sha256; heavy steps via `tools/launch_detached_process.py --done-receipt`; serializer
commits, two review passes per .py, `[no-triality] [p0-ledger-ok]`, never a co-author trailer or AI attribution; rc 17/19
is NOT a stop — continue, deliver `/Volumes/APDataStore/pact/ddm_mrs5/mrs5_source_delivery.tar`, MAIN lands and commits
the intent before authorization (that ordering is recorded). Checkpoint `ddm_mrs5`; lane
`ddm_mrs5_minimal_packet_geometry_native_prefire_intent_20260916`.

## OPTIMAL FORM
Reference forms: the sealed receiver's inflate.sh native set (the measured configuration), mrs4's packet and proofs,
rlc1/rlc5's same-host timing record and pr12's contract. Declared deltas: geometry.c (restores the measured set), the
interleaved A/B (the instrument), the intent (the contract's door). Provenance pins: mrs4 landing e621fd5a9; sealed
`rlc1_geometry.c` sha 414cc13dfdf597ca; move 53 packet a91a7dde3; contract a47543199 + `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`.

## Prior negatives accounted (operator 2026-08-15)
PR #140 (7 files / ~4.5k LOC is the measured minimum for this object; say so in the README in one sentence); mrs2/mrs3/mrs4
(single-stage cures and host-ratio projections are not evidence — hence the interleaved same-host ratio and the T4
measurement); tc4 / pr19 (a receiver change needs its own measured leg — the intent chain exists for exactly this);
rlc2/ffi1 (contract refusals are real controls); the ExFAT venv; pd6's rider-drop law.

Final message: the seven files with LOC, bit-identity counts, 600/600 raws, both smokes, the A/B/A/B table with load
readings and the ratio ± spread, the projected T4 time, the risk receipt path + sha, the intent path with file_sha256 /
bytes / canonical digest, the typed normal-validator refusal, the fresh-reader prompt path, every boundary, serializer rc,
ending with `composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)`.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
