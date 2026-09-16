# Move-53 baseline coverage and raw identity

MEASURED `[macOS-CPU advisory]`: **4,860 distinct Python lines in 38 files**, from a cold n600 source decode. The covered files contain **11,785 physical source lines**; the whole sealed Python tree contains **43 files / 15,767 physical lines**. Physical lines include blanks, comments and definitions: these ratios are a size comparison, not an executable-statement coverage percentage. The charter's 35-file / 11,148-line static import closure is a different denominator; this audit does not re-derive that closure or equate it with the dynamic file count.

**600/600 raw identity:** 3,662,409,600 bytes, SHA-256 `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b`, identical to the sealed move-53 advisory raw receipt. The certificate retains 600 per-pair hashes. The measured run reports no render checkpoint reuse, token restart at frame 0, and token cache disabled. Archive remains 179,286 bytes, SHA-256 `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`. This is output identity, not a new score or T4 receiver admission.

Host decode/render took **1,170.415232 s**; traced call wall time **1,171.707572 s**. Native corrector, native range coder and native geometry were active. Python line coverage cannot cover their C internals, and import-time hits in alternate Python correctors do not establish fallback execution. The trace invokes `inflate_archive` directly; public shell/bootstrap smoke is separate. Seed 20260916 selected 32 pairs within the completed n600 run; no per-pair Python line attribution was collected.

## Custody and runner revisions

The exact v1 bytes identified by the parent were preserved before editing at `/Volumes/APDataStore/pact/ddm_mrs1/trace_runner_v1.py` (6,116 bytes, SHA-256 `6381e778764852c6adde96d98c58ca9f10e639b03c925df87918453d25b7f6b2`). The provenance record is `/Volumes/APDataStore/pact/ddm_mrs1/trace_runner_v1_provenance.json`. Limitation: v1 omitted its own launch-time hash; preservation happened after completion, before this first edit. The snapshot and parent execution custody bind v1; a retroactive launch hash is not claimed.

Original INPUTS, COVERAGE, DECODE_REPORT and IDENTITY files remain unchanged. The machine receipt alongside this file binds their hashes and every source file. The original runner committed the matching raw certificate before deleting raw; the raw is now absent. Its intermediate coverage files overstated planned completion. Only the final completed decode report plus identity certificate support n600; intermediate v1 wording is not evidence of completed n600.

The revised runner records its own source hash and new-run source/archive/library/runtime bindings, rejects mismatched reused coverage, observes completed token frames, validates completed certificates, and can finish certified raw cleanup after interruption. Completed v1 reuse validates preserved evidence without rewriting it as v2. Read-only helper validation against the actual completed evidence passed; AST compile passed. **No full v2 launch, v2 resume, crash injection or decode rerun was performed.** Two visible complete source reads and tracker passes `mrs1-trace-v2-pass1` / `mrs1-trace-v2-pass2` were completed. No submission files, archive, sealed source, scorer, T4 job, git index or commit were changed by this subtask.

## Source counts

| Sealed source | Live Python lines | Physical lines |
|---|---:|---:|
| `compress.py` | 0 | 2280 |
| `cpr1/carrier_codec.py` | 168 | 215 |
| `cpr1/ddm_mp2_semantic_receiver.py` | 157 | 341 |
| `cpr1/hpac_integer.py` | 265 | 492 |
| `cpr1/hpac_integer_sparse.py` | 88 | 229 |
| `cpr1/inflate.py` | 251 | 358 |
| `cpr1/integer_model_io.py` | 89 | 141 |
| `cpr1/rc1_adaptive_model_sections.py` | 55 | 563 |
| `inflate.py` | 0 | 78 |
| `runtime/__init__.py` | 1 | 1 |
| `runtime/baseline.py` | 43 | 73 |
| `runtime/bits.py` | 35 | 50 |
| `runtime/carrier_repack.py` | 158 | 234 |
| `runtime/compensation_overlay.py` | 38 | 164 |
| `runtime/ddm_wc1_advisory_runtime.py` | 0 | 681 |
| `runtime/dx2_cabac_coefficients.py` | 174 | 331 |
| `runtime/entropy/__init__.py` | 1 | 1 |
| `runtime/entropy/adaptive_ans.py` | 20 | 113 |
| `runtime/entropy/coefficient_ar1_codec.py` | 87 | 131 |
| `runtime/entropy/coefficient_predictor.py` | 62 | 107 |
| `runtime/entropy/rc64.py` | 56 | 78 |
| `runtime/entropy/renderer_weight_codec.py` | 25 | 233 |
| `runtime/f26_hpac_native.py` | 0 | 655 |
| `runtime/f26_inflate.py` | 382 | 728 |
| `runtime/frame0_selector.py` | 104 | 144 |
| `runtime/free_corrector.py` | 42 | 335 |
| `runtime/fx1_logistic_mixer_corrector.py` | 109 | 785 |
| `runtime/fx2_model_axis_corrector.py` | 76 | 730 |
| `runtime/hpac_inference.py` | 220 | 325 |
| `runtime/ihs2.py` | 140 | 365 |
| `runtime/ihs2_gate_a.py` | 0 | 288 |
| `runtime/native_free_corrector.py` | 217 | 415 |
| `runtime/rc1_adaptive_model_sections.py` | 151 | 563 |
| `runtime/rc2_hpac_semistatic_mixing.py` | 231 | 842 |
| `runtime/rc3_shared_mixer.py` | 121 | 182 |
| `runtime/residual_archive.py` | 439 | 822 |
| `runtime/rlc1_geometry.py` | 85 | 120 |
| `runtime/rlc1_mixer.py` | 93 | 135 |
| `runtime/rr4_free_corrector.py` | 57 | 336 |
| `runtime/rr5_arith_basis.py` | 244 | 522 |
| `runtime/sm1_semantic_mixer.py` | 143 | 223 |
| `runtime/tc1_receiver_checkpoint.py` | 97 | 162 |
| `runtime/tc1_shared_mixer.py` | 136 | 196 |

## RECALL EVIDENCE

Content queries `byte.identical|resume|raw identity` covered research index/DAG files, docs and the hot state; `raw|native|T4` covered the corrector-axis and rule-118 cure memos. The canonical equations registry was read with `tools/list_canonical_equations.py --json` and filtered for `raw identity`, `q16`, `checkpoint`, `reproducibility` and receiver refit wording. Beyond charter seeds, `ddm_cd1_corrector_shipping_axis_decomposition_20260820.md` records host/T4 timing inversion; `ddm_rlc1_rule118_cure_20260910.md` records actual checkpoint restart proof; `docs/canonical_long_training_infrastructure.md` requires substrate/curriculum resume binding. Registry receiver-refit domain explicitly excludes receiver changes from unchanged-receiver admission. These findings kept host timing advisory, motivated full resume bindings, and prevented treating unchanged archive bytes as new-receiver T4 proof. No additional baseline proof was inferred from that recall.
