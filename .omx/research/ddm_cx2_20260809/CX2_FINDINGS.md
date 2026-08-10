CX2 DID NOT BEAT the 0.172141297491896447 contest-CUDA bar: its receiver-closed archive measured `S=0.25517485122403977` on a non-comparable `[macOS-CPU advisory; upstream AV GT; immutable evaluate.py; n600]` axis, `score_claim=false`, and the own-vehicle frontier did not move.

# CX2 end-to-end composition findings

## Outcome and exact residual

The deliverable archive is `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/composed/archive.zip`: **186,698 B**, SHA-256 `2acd09e7a585c12403936d1e8a6dc70a9b35d826fe61ead7dea49ad470c4a996`. Its identical evaluator copy is under `submission/archive.zip`. It is 4,354 B smaller than the 191,052 B PR130 base, an exact rate improvement of `-0.0028991498818939504 S`.

The one full-n600 score was:

| term | measured value | S contribution | axis |
|---|---:|---:|---|
| SegNet distortion | 0.00042839895468205214 | 0.042839895468205214 | macOS CPU, upstream AV GT |
| PoseNet distortion | 0.0007747594499960542 | 0.08802042092583141 | macOS CPU, upstream AV GT |
| Rate | 186,698 / 37,545,489 | 0.12431453483000314 | exact bytes |
| **Total** | — | **0.25517485122403977** | local advisory, `score_claim=false` |

The numeric gap to 0.172141297491896447 is `+0.08303355373214333 S`, but this subtraction crosses CPU/PyAV and contest-CUDA/DALI axes and is not a promotion comparison. At the measured local distortions, crossing that numeric bar would require an archive of at most **61,996 B**, so this candidate is **124,702 B over**. Tokens are the largest section at 114,860 B and carry 92.1% of that fixed-distortion byte debt, but even making the entire token stream free would leave 9,842 B to remove. The cheapest named next measurement is therefore the paired uniform-q4 control on this exact CPU/AV runtime: it isolates whether the large Pose/Seg residual is caused by SD1M content or by the unpaired device/GT/runtime axis before more model work is routed.

## Composition and measured interactions

| stage | archive B | delta from previous |
|---|---:|---:|
| PR130 CPR1 base: q4 + joint XZ + Range | 191,052 | — |
| SD1 mixed semantic + joint XZ + Range | 190,204 | -848 |
| SD1 mixed + split Brotli + Range | 189,241 | -963 |
| SD1 mixed + CX2 reversible xcodec split Brotli + Range | 188,818 | -423 |
| SD1 mixed + CX2 xcodec split Brotli + retained ANS | **186,698** | **-2,120** |

Measured interaction terms, not arithmetic transfers:

- `MEASURED`: SD1 plus split Brotli landed 60 B below the additive prediction (`189,241` actual versus `189,301`).
- `MEASURED`: the CX2 reversible xcodec saved 423 B on the already-mixed split object.
- `MEASURED`: retained ANS saved exactly 2,120 B on the final model pack.
- `MEASURED`: the charter's 187,181 B arithmetic projection was not the object; the actual archive is 483 B smaller because the interaction and xcodec were measured on the composition.

## Final section table

| counted item | compressed B | parse-back object |
|---|---:|---|
| SD1M semantic Brotli stream | 33,714 | 39,090 B, SHA-256 `39002165c78ab707c15586110678671cd832101a970de5bd0f3b96824a2aa2cc` |
| pose carrier Brotli stream | 23,058 | 23,054 B, SHA-256 `a05d0985ca5a8d5110bd5bf5be39f238c6f89640b8a8bb888a3e1269bdf636e4` |
| HPAC-model Brotli stream | 14,950 | 20,179 B, SHA-256 `b07fff73fac41c5fec2d8acbfd7c43c518852696f18d95cf7465fc6ed7510b58` |
| retained ANS tokens | 114,860 | 117,964,800 decoded tokens, SHA-256 `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece` |
| split + outer headers | 16 | exact selector/length grammar |
| ZIP overhead | 100 | one stored, unencrypted member `p` |
| **archive.zip** | **186,698** | SHA-256 `2acd09e7a585c12403936d1e8a6dc70a9b35d826fe61ead7dea49ad470c4a996` |

The candidate mixed allocation is q3 only for `frame_embed.weight` and `blocks.{1,2,3}.film.weight`; all other semantic tensors remain q4. The candidate `models_raw` is 82,331 B, SHA-256 `4bae76d7878a753ef5a675c5fddafaff6d8987c4fe1611f54ebcd6f5d0fabf21`.

## New mechanism found by the dig

`MEASURED`: a G20/G25-style search over the **final deterministic ZIP**, not independent section minima, evaluated 6,534 complete archives: 33 semantic variants × 3 carrier variants × 33 HPAC variants × 2 ZIP policies. The winner applies signed zigzag, 4,096-byte blocking, and even/odd lane separation to semantic bytes before Brotli q10; keeps the carrier identity transform; and applies `xor80` to HPAC bytes before Brotli q10. Receiver selector 3 reverses each transform exactly without a new counted tag.

- Semantic: 34,125 B identity q11 → 33,714 B selected, `-411 B`.
- Carrier: q9/q10/q11 all emit the same 23,058-byte stream; q9 is only the deterministic representative.
- HPAC: 14,962 B identity q11 → 14,950 B selected, `-12 B`.
- Final net: `-423 B` versus the mixed split comparator.
- Six parameter rows tie at 186,698 B, representing two unique archive byte strings; three parameterizations are byte-identical to the selected archive. The selected parameterization and selected archive are not unique minima.

`MEASURED` dead search cells: outer ZIP DEFLATE produces 186,758 B, 60 B worse than stored. A preliminary 2-byte compact allocation-rank header made semantic Brotli 86 B worse than the retained 14-byte SD1M header; that run-local observation did not enter the retained 6,534-row search receipt.

`DERIVED`: the win comes from changing local byte coordinates seen by Brotli, not from lowering an entropy floor or inventing another memoryless coder. `CONJECTURE`: learned token context or a jointly trained cross-section model can still move the 114,860-byte dominant stream; the present search does not price that family.

## Parse-back, receiver, and evaluator closure

The parse-back receipt at `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/composed/parseback_receipt.json` proves all four raw sections are consumed and restored exactly, including SD1M q3/q4 allocation and legacy-q4 identity coverage. The scored ZIP has one stored member `p`; the literal receiver separately proved that `submission/archive/p` equals the bytes read from the scored ZIP.

The literal receiver receipt is `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/decode/literal_receiver_receipt.json`, SHA-256 `3f9b135ac6e07b90984a420b45d6a8164957612fad821644c1ca655438fc7d37`:

- current `inflate.sh` decoded all 117,964,800 ANS tokens and returned the explicit empty-final-state proof;
- the decoded-token SHA matches DT1: `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`;
- the atomically promoted raw is 3,662,409,600 B, SHA-256 `3319a2bddb98a93dc4552d1ccde8f404767bf3939985fae23c58098996ee541d`;
- wall time was 1,010.8084664579947 s, leaving 789.1915335420053 s under the 1,800 s inflate limit;
- constriction 0.5.0 and Brotli 1.2.0 were checked; the local run used the explicit Homebrew Brotli CLI because dependency-network access was unavailable, while the contest default remains the pinned wheel path.

The sole scorer receipt is `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/evaluation/evaluator_trace.json`, SHA-256 `39ab09e4d51365489310b1f6dbf631c5d58a2cfb7d09bb634ed95a1da72477e1`. It binds the pre/post-identical ZIP, raw, `evaluate.py`, `frame_utils.py`, `modules.py`, PoseNet and SegNet weights, video-name file, and original `0.mkv`; it enforces n600, CPU, AV, batch 16, seed 1234, two threads, prefetch depth 4, and recomputes full-precision S.

## Borrowed-substrate and custody accounting

- PR130 CPR1 neural state, carrier, HPAC model, and base runtime are borrowed off-the-shelf substrate, not claimed as original CX2 work.
- SD1's mixed semantic allocation is a landed predecessor result; CX2 added the public-runtime allocation parser, legacy-q4 identity, and its counted receiver grammar.
- DT1 supplied the retained n600 ANS payload and exact decode target. Its terminal receipt is SHA-256 `5c15f38ab68df68c09a5859d17d19e4247f90e76457282edccbc8a34d060916c`. DT1 source remains untracked and its source SHA is not self-recorded in that receipt; CX2 pins the observed receipt/payload/decoded result but cannot upgrade that missing provenance.
- CX2's original contribution is the reversible coordinate family, final-ZIP selection, composed builder, literal custody wrapper, evaluator tracer, and receiver/runtime integration—not a new learned vehicle.
- The intake clone and `upstream/` were read-only throughout. Protected common-contract paths were not edited.

## Verification and source landing status

`MEASURED`: 28 targeted tests passed, including the existing RC1 surface; the direct CX2 subset was 16/16. Ruff, Python compile, shell syntax, `git diff --check`, runtime-manifest hashes, and the pinned dependency smoke passed. Every changed Python file received two review marks. Final adversarial review found no remaining concrete P0; P1 remains because the literal wrapper's promotion/recovery and custody branches lack dedicated fault-injection unit tests.

The nine CX2 source/test/runtime files and these two terminal evidence files landed through the required serializer at commit `cf53216e3e856c15f849bcfe96a5dd4717da2d04`, with post-edit content hashes, `[no-triality] [p0-ledger-ok]`, and no attribution trailer. No fallback Git write or override was used, and `git diff --cached --name-only` was empty after landing.

The exact landed implementation allowlist is:

- `experiments/ddm_cx2_compose_end_to_end.py`
- `experiments/ddm_cx2_literal_receiver.py`
- `experiments/ddm_cx2_trace_evaluate.py`
- `experiments/tests/test_ddm_cx2_trace_evaluate.py`
- `src/tac/pr130_runtime/dv1_cpu_runtime/inflate.py`
- `src/tac/pr130_runtime/dv1_cpu_runtime/inflate.sh`
- `src/tac/pr130_runtime/dv1_cpu_runtime/receiver.py`
- `src/tac/pr130_runtime/dv1_cpu_runtime/runtime-dependencies.json`
- `src/tac/tests/test_ddm_cx2_compose_end_to_end.py`

## RECALL EVIDENCE

Sources and queries consulted beyond the charter seeds:

- Corpus query: `rg -n -i "lossless xcodec|whole-object|complete-final|G20|G25|PR130|split Brotli|ANS" .omx/research ...`. G20's complete-ZIP search measured 83,838 → 81,027 B (`SPEC_g20_ep725_lossless_xcodec_recode_20260726.md:36-71,91-111`); G25 made the complete archive the selection variable and measured 83,838 → 80,238 B with a `-274 B` coder/container interaction, independently reproduced (`SPEC_g25_population_global_same_solution_recode_20260726.md:9-21,66-114` and both v2 receipts). These are different vehicles, so only the search and interaction discipline transferred; their byte deltas did not.
- Canonical-equation query: `.venv/bin/python tools/list_canonical_equations.py --json | jq ... test("brotli|archive|codec|rate|zip|container")`. Relevant rows were `brotli_cascade_bounded_per_stream_v1`, `master_gradient_locality_violation_by_codec_v1`, and `cross_codec_super_additive_orthogonality_predictor_v1`. This changed the dig toward reversible decompressed-section coordinates followed by empirical full-ZIP selection, not raw compressed-byte gradients or predicted additivity.
- Index/DAG/ledger query: `rg -n -m 30 -i "G25|population-global|lossless_xcodec|whole-object" CANONICAL_RESEARCH_INDEX sub015_DAG ddm_deferral_queue_ledger`, plus `rg -n -i "ddm_cx2" codex_arm_queue*.jsonl`. No direct G20/G25 entry appeared in the index/ledger scope; the DAG did retain whole-object compactness and non-additive KKT guidance. Queue row 799 established `owns_scorer=true`, and the lane ledger enforced the one scorer pass.
- Negative-family recall: Clean60 is corrected to +185 B worse (`CORRECTION_clean60_lost_and_the_coder_gap.md:6-25`); temporal delta is 649,042 versus Brotli 429,676 B (`CODER_LINEAGE_VS_HPAC.md:21-30,63-74`); measured LZMA is 3.510× HPAC while the broader ~2.2–2.8× band is partly derived (`CODER_LINEAGE_VS_HPAC.md:48-61,76-87`); phase coset is +101,837 B / +38.298% (`ddm_cr1_20260808/CR1_FINDINGS.md:5-13,33-50,80-93`); boundary-distance context is +192,417 B (`codex_findings_ddm_lp1_layer_pricing_20260725_codex.md:43-54,72-81`); post-hoc shared-value codebooks are null at formulation × instance scope, while a paying joint chart saved 417 B (`witness_crosstensor_structure_rate_20260713.md:17-22,43-89`). These closures prevented re-running known dominated coder/representation cells and concentrated the only new search on exact reversible full-object interactions.

## What was not run

- Contest-CUDA/Linux DALI evaluation: no authorized contest hardware or operator GO; Modal was expressly forbidden. The measured CPU/AV row cannot be promoted or compared as the same axis.
- A paired uniform-q4 CPU/AV control: the common contract allowed one full-n600 scorer job, spent on the composed deliverable. Without that pair, SD1M's Pose and Seg deltas remain unisolated from runtime/device/GT differences.
- Linux runtime closure: the isolated shipping runtime was smoke-tested on macOS only.
- Metal/capacity work: Metal was unreachable and is a separate governed vehicle arm.
- A new token predictor, learned cross-section model, or rate-aware carrier QAT: those are mechanism families, not bounded additions to the required end-to-end row; no byte or score benefit is claimed for them here.

## Follow-on dispositions

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN scorer owner; consumer store: a paired q4/SD1M CPU-AV distortion receipt beside `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/evaluation/`; fire trigger: the n600 scorer lane is free and the exact same pinned raw-generation/evaluator axis can be used once for each bound archive.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN contest-row owner; consumer store: contest-CUDA/Linux receiver and evaluator receipts for archive `2acd09e7...`; fire trigger: Linux dependency/receiver closure passes, the lane is claimed, and operator authorization exists. This is lower priority than the paired q4 control because the local row is 0.0830 S numerically over the bar.

## LIVE-HYPOTHESES

- SD1M may be responsible for a large part of the local Pose/Seg residual because it is the only content-changing component; the paired q4 control is required because CPU rendering and AV-versus-DALI are also changed and can dominate.
- A better token model remains the largest byte lever because the exact ANS coder is already closed against its present probability model and tokens still occupy 114,860 B (61.52%). Cross-frame, ego-motion, or class-conditioned context is plausible uncoded structure.
- Cross-section learned conditioning may beat independent Brotli streams even though LZMA found near-independence; G25 and the measured -60 B CX2 interaction show that container/coordinate choices can expose object-level dependencies, but no learned row exists yet.

## DEAD-ENDS

- The 2-byte compact SD1M allocation header is closed for this formulation because its changed byte coordinates cost 86 B after Brotli despite saving 12 raw header bytes.
- Outer ZIP DEFLATE is closed on this exact object because it adds 60 B.
- The 187,181 B arithmetic composition is closed as a deliverable; the actual composed object is 186,698 B and interactions must be measured.
- Carrier Brotli q9 is not uniquely optimal; q9/q10/q11 are byte-identical and six parameter rows tie at the minimum.
- More memoryless coder swapping is closed on these sections by prior same-object measurements; the remaining token route requires a better model or representation.
- Shared FX1 cannot decode SD1M or satisfy the evaluator's three-argument contract unchanged; the isolated pinned runtime is the tested object.
- A local CPU/AV number cannot stand in for contest-CUDA/DALI, and no promotion claim is made.

Own-vehicle frontier: `S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600` — UNMOVED.
