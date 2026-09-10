# Decode wall-clock seal review

Scope: `src/tac/decode_wall_clock.py`, `src/tac/candidate_seal.py`, and their two test modules. No score claim; synthetic contract fixtures are never timing measurements.

## RECALL EVIDENCE

Read charter plus MAIN addendum, common contract, PROGRAM, NO-FAKE canonical section, operating handoff, and hot state. Searched `.omx/research` content for `wall.clock|decode.*budget|1260`; additional seeds included `ddm_dc1_decode_budget_conditional_coding_20260816.md` references and `ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md`. Queried canonical-equations JSON for timing/decode terms; the first piped search ended early with a BrokenPipeError, so no completeness claim is made for that query. Root holds the complete equations recall artifact. No existing seal timing implementation was found in the candidate-seal source/test scope. The dual runtime-digest issue in the hot state changed the design: calibration reuses the worker's upload-projection algorithm rather than comparing unlike digests.

## Findings repaired before clean passes

- Archive bytes contaminated the proposed receiver identity. Receiver digest now excludes the counted `archive.zip` and normalizes only the AST literal value spans of the two top-level archive pins; it preserves all remaining shipped code bytes.
- Archive equality alone did not bind the T4 calibration receiver. The validator now computes the worker upload projection from the calibration's retained local runtime and compares it to the source remote runtime field.
- Unknown local concurrency could mask a measured remote overrun during inheritance. Source path discovery accepts unknown concurrency, then strict source validation propagates the actual T4 failure.
- A resumed suffix could be mislabeled as full decode wall time. Local receipts require `cold_start: true`, reject resumed flags and nonzero checkpoint start.
- tc4's failure must survive an optimistic tc3-based projection. Optional candidate T4 receipts validate raw completed/timeout evidence before considering projections. The actual retained tc4 timeout control was consumed successfully: lower bound 1800 seconds exceeds 1260.

- Root fresh eyes found host transfer was unbound: candidate local host and optional platform/hardware fingerprint now must exactly match the calibration local identity, with executed mismatch controls. This correction reset the clean-pass counter before both final passes below.

## Final clean review pass 1 — contract and provenance

Re-read final receipt construction/validation and source bindings. Verified v1 absent remains readable, v2 absent refuses, builder refuses missing timing, raw receipts are content-hashed, frames are consecutive integers in 0..599, exactly four threads are required, and finite positive numbers are checked before arithmetic. Recomputed ratio direction T4/local and the 0.7*1800 rule. Matched actual tc3/tc4 worker upload digests to their remote receipts. Inspected non-quiescent rejection and the absence of any unmeasured normalization factor. No further defect found.

Assumption challenged: code identity could imply timing identity for a new payload. It does not. The charter explicitly authorizes code-based inheritance; the receipt records this limitation, preserves the pointer archive, requires a valid direct measured source, and prohibits chaining or failed-source grandfathering.

## Final clean review pass 2 — adversarial consumer behavior

Re-read final seal integration, inheritance traversal, numeric/parser errors, timeout signatures, source drift, and fixture mutations. Confirmed timeout evidence binds command stage, configured timeout, archive, runtime, hardware, failure status, and retained bytes. Same-archive calibration cannot apply an old actual T4 time to changed candidate code. Rehashing invalid local fixture content also updates the calibration reference so malformed-field tests exercise fields rather than merely stale hashes. No further defect found.

Assumption challenged: a loaded local timing can safely pass because it is slower. The charter requires quiesced or measured-normalized input. This implementation only accepts quiesced input; unknown or competing process counts are retained as rejected evidence. Supporting normalization remains outside this implementation; no estimated divisor is accepted.

Validation: 93 tests passed across both modules; ruff passed on all four Python files. Tests are synthetic contract controls; empirical source validation is separately retained in `T4_RUNTIME_DIGEST_CROSSCHECK.json` and `TC4_TIMEOUT_VALIDATOR_CONTROL.json`.
