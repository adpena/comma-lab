# Seal-contract gap: a SEAL_VALID candidate was undecodable by the contest's own inflate.sh (directive, 2026-09-05 18:58Z)

Tokens: `[no-triality] [p0-ledger-ok]`. Owner until claimed: MAIN. Priority: P0 apparatus (a paid row was lost to a gap the seal should have closed).

Incident (MEASURED): rc1's candidate (archive 1438049e…, 178,249 B) passed `make_candidate_seal.py` (SEAL_VALID: archive, runtime digest, receiver pins,
admit bar) and the fire tool's checks, then FAILED on T4 in 3.6 s: `runtime/f26_inflate.py:429 InflationError("F26 requires WANS1, SD1M, or SM3R semantic
weights")`. The arm had verified "receiver decode identity" through the receiver library path (`residual_archive.py` codec dispatch), which the public
`inflate.sh → inflate.py → f26_inflate.inflate_archive` path never reaches. Cost: one Modal call, one candidate round-trip, ~30 min.

## ITEM 1 — make_candidate_seal.py requires an inflate.sh smoke receipt through the PUBLIC entrypoint
The seal refuses unless it carries a receipt of `bash inflate.sh <extracted> <out> <file_list_first_2_pairs>` run on the staged tree (CPU is fine),
with the produced frame bytes sha-compared against the same 2 pairs produced by the CURRENT frontier tree the same way (identity for rate-only
candidates; for distortion-changing candidates, just the run succeeding + frame count). Acceptance: a seal built from a tree whose public path
refuses the archive cannot be SEAL_VALID; a test replays the rc1 incident and asserts refusal.

## ITEM 2 — fire_modal_auth_eval.py refuses a seal without the ITEM 1 receipt
Additive check beside SEAL PIN CONSISTENT; same-line waiver only for custody replays of an already-scored archive. Acceptance: dry-run on the rc1 seal
(pre-fix) prints the refusal with the rule chain.

## ITEM 3 — the arm-facing charter template names the public-entrypoint smoke as step zero of "receiver identity"
`tac.subagent_contract.standard_contract()` gains one sentence: "receiver identity means bash inflate.sh on the staged tree, not the library path".

**ITEM 1 refinement (rc1's measurement, 19:25Z):** the contest file_list selects VIDEOS (only "0" is accepted), so a "first 2 pairs" truncation is not
expressible, and `inflate.py:main` refuses to run without CUDA on this host (the submission's own gate) — so the local smoke receipt a seal must carry is
the PAIR rc1 realized in `stage_restage`: (a) `f26_inflate.inflate_archive` on the staged tree reaches the token decode within a time bound with no
exception (candidate AND the current frontier tree as control), and (b) `bash inflate.sh <dir> <out> <file_list>` runs the whole preamble (backend
build, Brotli gate, file-list dispatch, `_verify_input` against the re-pinned constants) and stops exactly at the CUDA gate. Both receipts, both trees.

## ITEM 4 — review_tracker `mark-file` marks without rescanning (vacuity class; MEASURED by md4, 2026-09-05 23:05Z)
`tools/review_tracker.py mark-file <f> --status reviewed` does not rescan the file: a function added after the last `scan` is marked reviewed VACUOUSLY (md4
measured 11 entities before `scan`, 12 after). Cure: `mark-file` rescans first (or refuses when the file's entity census differs from the last scan), so the
two-visible-passes policy binds the CURRENT entities. Acceptance: a test that adds a function and asserts `mark-file` either covers it or refuses.
