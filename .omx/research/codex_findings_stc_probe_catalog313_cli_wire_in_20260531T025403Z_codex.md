# Codex Findings - STC Probe Catalog #313 CLI Wire-In

timestamp_utc: 2026-05-31T02:54:03Z
agent: codex
scope:
  - tools/probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py
  - src/tac/codec/tests/test_probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py

## Reviewed Inputs

- Partner landing `d0ef13d13` added the STC clean-source mask-delta syndrome
  versus brotli disambiguator and an external `.omx/state/probe_outcomes.jsonl`
  row.
- Partner landing `a3690213c` correctly HALTed the UNIWARD wire-in on phantom
  APIs (`tac.bit_allocator.allocate_per_byte`, `tac.composition`) rather than
  inventing authority.
- `tools/lane_maturity.py validate` was clean before this patch; the remaining
  risk was not lane schema but orphaned probe signal.

## Finding

The STC probe CLI emitted a structured JSON verdict, but the CLI itself did not
own a canonical Catalog #313 registration path. That meant a future operator or
runner could re-run the probe and forget to append the outcome through
`tac.probe_outcomes_ledger.register_probe_outcome`, leaving the dispatch gate
dependent on a manually added row rather than the executable probe surface.

The existing empirical tests also encoded a stale fixed DEFER expectation and a
slow default run. The live lightweight syndrome-only rate model can return
PROCEED for small deterministic fixtures; that does not create score authority
because the probe remains `[macOS-CPU advisory]`, `research-signal`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Patch

- Added `catalog_313_probe_outcome_kwargs` to every probe verdict.
- Added `--register-probe-outcome`, `--probe-outcomes-ledger`, and
  `--probe-outcomes-lock` so the CLI can append through the canonical locked
  ledger, including custom ledger paths for tests.
- Kept default CLI execution non-mutating; ledger writes happen only when the
  registration flag is present.
- Updated tests to cover both PROCEED and DEFER branches explicitly, avoid
  timeout-prone large default fixtures, and assert that the registered row
  follows the measured verdict instead of hardcoding a stale one.

## Authority

No score claim was added. A PROCEED probe row is advisory only; a DEFER row is
blocking through Catalog #313. MLX/CPU research-signal remains local routing
signal and cannot promote, rank, kill, or dispatch exact eval without the
separate archive/runtime custody gates.

## Validation

- `.venv/bin/python -m ruff check tools/probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py src/tac/codec/tests/test_probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py`
- `.venv/bin/python -m pytest src/tac/codec/tests/test_probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py -q`

Result: 12 passed in 0.79s.

## Review Note

Subagent review was requested earlier, but the current thread had no free
subagent slots. I performed the local adversarial pass directly and preserved
the unresolved low-level semantic caveat: this probe is still rate-axis
research signal, not a byte-closed archive decoder proof.
