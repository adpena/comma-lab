# PR130 REPRODUCED ON THIS MACHINE — byte-identical archive, full verify gate green

Operator roadmap (2026-08-09, restated): *"we're not really seeking to optimize our own
vehicle frontier now anymore. We're seeking to take PR one thirty, make sure we can run it
and reproduce here, and then iterate and optimize on that using everything we have that
they don't."* This is the "run it and reproduce here" milestone.

Date 2026-08-09. Intake tree READ-ONLY throughout; all outputs written outside it.

## The result

`scripts/reproduce.sh` rebuilds the archive and asserts byte-equality against the committed
canonical archive. It PASSED:

```
CPR1 byte-identical: 191052 0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd
```

`scripts/verify.sh` (the full gate: repo audit + compileall + reproduce + pytest) PASSED:
`24 passed in 1.04s` → `CPR1 repository verification passed`.

Output custody: `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/`
(archive.zip 191,052 B + cpr1.json + predecessor/). Run with
`PYTHON_BIN=/Users/adpena/Projects/pact/.venv/bin/python`.

## What this DOES and DOES NOT establish — read before citing

**DOES:** the ARCHIVE ASSEMBLY chain is byte-exact reproducible here. From the banked
artifacts (`artifacts/base/int5_delta_archive.zip`, `artifacts/hpac/*.bin.xz`, `*.tokens.bin`)
through `rebuild_submission_hpac.py` → `compress.sh` → the 191,052 B canonical archive, every
byte matches. Their packing, their coder settings, their ZIP determinism: all reproduce.

**DOES NOT:** re-derive the artifacts from the video. `reproduce.sh` consumes banked
TRAINED artifacts as inputs. Reproducing the TRAINING is a separate, per-leg question:

| leg | status 2026-08-09 |
|---|---|
| semantic renderer | REPRODUCED at inference on Metal: DALI-GT quantized `exact_seg` 0.0002857038709852431 = 0.998650× published Ada; AV-GT 0.0002764044867621528 = 1.000123× stage-08 recorded. 19s/n600. |
| pose carrier | PORT UNBLOCKED. Native sparse `nn.Embedding` + COO `coalesce` + `RowLocalSparseAdam` run on MPS at PINNED torch 2.10.0, zero CPU-fallback, row-local clocks preserved (`MAIN_METAL_RECEIPT.md`). Training run not yet done. |
| tokens / HPAC | hb1/hb2 line — HPAC trained on OUR labels; pack round-trip fixed. |
| score | NOT reproduced here. 0.172141297491896447 is `[contest-CUDA, DALI GT, n600]`; this machine has no CUDA. |

**A note the report volunteers, NOT a claim:** `cpr1.json` carries
`projection_from_displayed_metrics` = {ada 0.16984766243023947, a4500 0.17089548488809853},
both BELOW the 0.1721417 CPR1 figure. These are PROJECTIONS from ROUNDED displayed metrics,
not measurements. Do not treat them as a lower bar. The bar remains 0.172141297491896447.

## Archive anatomy (MEASURED, exact leave-one-out on these bytes)

Marginals are additive here (superadditivity gap **−20 B = 0.0267%**), so they can be budgeted:

| section | bytes | share | marginal S |
|---|---:|---:|---:|
| **tokens** | **116,980** | **61.23%** | **0.0778922** |
| semantic renderer | 36,580 | 19.15% | 0.0243571 |
| carrier (pose) | 23,384 | 12.24% | 0.0155704 |
| hpac | 15,092 | 7.90% | 0.0100491 |
| ZIP/header overhead | 104 | 0.05% | — |

Tokens are NOT in the joint LZMA stream. And the joint LZMA of the other three is **224 B
WORSE** than coding them separately — a free −224 B on any vehicle that copies this layout.

**Consequence for iteration: the rate axis is TOKENS (61.23%), not the renderer (19.15%).**

## Next on the roadmap

Iterate/optimize on THIS base with what we have and they don't. The reproduction is now a
fixed, byte-exact starting point, so every change is measurable as a delta against it.
