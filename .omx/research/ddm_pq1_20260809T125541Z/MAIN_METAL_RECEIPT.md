# PQ1 probe executed by MAIN — native sparse MPS at the PINNED torch: PASS

MAIN owns Metal; PQ1 built the probe but had no device. Executed 2026-08-09.

Receipt: `/Volumes/VertigoDataTier/pact/ddm_pq1_probe_20260809/probe_torch2100_pinned.json`
Runtime: `/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv`, built via
`UV_PROJECT_ENVIRONMENT=... uv sync --project upstream --frozen --group mps` (upstream tree untouched).

Authority `[macOS-Metal port verification]`, `score_claim=false`, `promotion_eligible=false`.
This verifies PORTABILITY of a training path. It is not a score and moves no pointer.

## The version gate did its job

First run used the repo venv (torch **2.12.1**) and the probe **REFUSED**:
`"worker torch version '2.12.1' is not the governed '2.10.0'"`, verdict FAIL.
That is correct: PP2's UNKNOWN rows 36/37 are VERSION-SPECIFIC (2.12.1 exposes
`SparseMPS` registrations that 2.10.0 may not), so measuring on 2.12.1 answers a
different question. The probe refused to produce a number about the wrong object.

## Result at the pinned version: PASS

| assertion | cpu | mps |
|---|---|---|
| torch_version | 2.10.0 | 2.10.0 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | 0 | 0 |
| stderr CPU-fallback text | none | none |
| `grad_is_sparse` | True | **True** |
| `grad_is_coalesced` | True | **True** |
| selected rows step 1 (ids `[5,2,5,19,2]`) | `[2,5,19]` | `[2,5,19]` |
| selected rows step 2 (ids `[19,7,19,5,7]`) | `[5,7,19]` | `[5,7,19]` |
| `untouched_rows_bit_identical` both steps | True | True |
| `row_step` (row-local clocks) | `[1,2,1,2]` | `[1,2,1,2]` |
| grad_norm before/after clip | 7.599459648132324 / 0.7500000596046448 | identical |

CPU/MPS state+update parity held within the PREDECLARED fp32 tolerance
(atol 2e-6, rtol 2e-5) — declared in the tool before the run, not fitted after.

## What this closes

1. **PP2 rows 36+37 (the real risk) are CLOSED with a real receipt**, not bypassed.
   Native sparse `nn.Embedding` backward and sparse-COO `coalesce/indices/values`
   feeding `RowLocalSparseAdam` run on MPS at the pinned torch with zero fallback.
   PQ1's dense-gradient adapter is a mechanism-preserving *fallback we do not need*;
   the REFERENCE form is available. Keep the adapter as insurance, run the reference.
2. **The row-local clock mechanism survives on MPS.** `row_step [1,2,1,2]` over
   nonzero_clock_rows `[2,5,7,19]`: rows 5 and 19 were selected in both steps (clock 2),
   rows 2 and 7 once each (clock 1). Repeated ids within a step increment the clock
   ONCE per coalesced row, not once per occurrence — the exact property that a stock
   dense-Adam substitution would have destroyed (PP2's named MECHANISM-reduction trap).
3. PP2's UNKNOWN denominator on the active path: **0 of 60** families now unmeasured-and-risky
   for the sparse question. Row 33 was removed structurally (CPU-first safetensors load).
   The other 57 remain PP2 static coverage, not PQ1 execution receipts.

## Consequence for the roadmap

PR130's pose leg can now be TRAINED locally on Metal at the pinned runtime. That was
the last named portability blocker on the pose leg of "reproduce PR130 here."
