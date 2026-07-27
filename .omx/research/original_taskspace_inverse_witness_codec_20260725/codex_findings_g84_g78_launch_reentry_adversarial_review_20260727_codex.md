# G84 adversarial launch re-entry review: repaired G78 batch-16 cache

Date: 2026-07-27  
Review lane: `lane_g81_g78_batch16_margin_base_scorer_cache_adversarial_review_20260727`  
Reviewed lane: `lane_g78_batch16_margin_base_scorer_cache_materializer_20260726`  
Authority: launch-readiness review only; no scorer forward, aggregate, score, candidate, promotion, or pointer claim

## Relationship to G81

The append-only G81 review remains the authoritative record of the defects
found in the earlier implementation:
`codex_findings_g81_g78_batch16_margin_base_scorer_cache_adversarial_launch_review_20260727_codex.md`,
SHA-256
`45070f74445962e38f4654f584ad34a071181d47dfede951d22a241200d1f451`.
This new memo records the post-fix re-entry verdict; it does not rewrite the
original refusal.

## Exact verdict

**READY FOR GOVERNED RESEARCH-ONLY G78 MATERIALIZATION.**

This verdict permits only the governed encoder-side n600 scorer-cache run.
Both named G72 custody blockers remain open until all 38 global batches and
five stages exist, the aggregate is sealed, and
`MarginBaseScorerCacheLoaderV1.open(...)` strictly reopens the exact aggregate.
The run cannot create a candidate, score, promotion, or pointer movement.

## Finding-to-fix closure

1. **G81 P0 — live V15 camera was recorded but not enforced.** The producer now
   renders every requested V15 camera batch before any forward, compares its
   SHA exactly with the owned 38-batch V15 identity, and binds the live-R
   scorer-input SHA. Resume repeats preparation and both comparisons before
   skipping a completed forward. Wrong camera and wrong live-R inputs refuse
   without invoking the scorer.
2. **G81 P0 — renderer dependency closure was incomplete.** Preflight now
   computes the transitive local runtime closure from the executed V15
   receiver. The r4 receipt seals 558 runtime files, including
   `src/tac/through_r/resolution_chain.py`; all are members of the 632-file
   sealed-input closure and are rehashed on reopen.
3. **G81 P1 — stages trusted self-reported fragments, G51 metadata, and dense
   bytes.** Strict reopen now rederives chronological cross-boundary fragments
   from the 38 validated batch shards, compares the exact preflight-owned G51
   stage binding, and streams the expected fragment composition for all three
   dense stage fields. Self-consistently resealed wrong-G51 and
   stage-divergent-from-batch attacks now refuse.
4. **G81 P1 — launch gate named superseded r2/r3.** The validated G78 lane now
   points only to the immutable r4 zero-forward preflight.
5. **G81 P2 — source indexing was implicit.** The production preparer checks
   both the exact source path and expected one-based batch index, including the
   final partial batch.
6. **G81 P2 — aggregate identity and duplicated claims were weak.** Downstream
   open requires an explicit aggregate SHA, recursively reopens the preflight
   and all batch/stage files, and rederives duplicated geometry, custody,
   coverage, blocker, and cleanup claims.
7. **G81 P2 — immutable publication had a concurrent-writer window.** The
   writer uses a no-replace publication path and verifies existing identical
   bytes rather than overwriting a raced destination.
8. **G81 P2 — receiver label named the legacy surface.** The current preflight
   records the executed
   `CarrierComposeReceiverV1.render_camera_pairs.v15` contract.

## Exact r4 custody

```text
path        /Volumes/VertigoDataTier/pact/taskspace_batch16_margin_base_scorer_cache_n600_20260726_r4/00_preflight_receipt.json
bytes       342543
file sha    3a65a8ce9cfe0eaa681b9abb875703aefb4f3b0adff7d65020153259e13134fe
self sha    d5a2b4d621af6b62d2fb06e0b051006e1acf02be94f5a158e68513cce2dd989e
inputs      632 sealed files
runtime     558 transitive source files
contents    preflight receipt only; zero scorer/dense outputs
```

Independent strict replay returned `PASS`. Storage admission requires
12,884,901,888 free bytes; r4 recorded 360,839,086,080 free bytes on
`/Volumes/VertigoDataTier/pact`.

## Exact implementation custody

```text
core        5d33fcac0463f0d309a96b5dd48924cb2d4959263a4ff35e0f74dd966accc59d
core tests  51b4993202efa17f1ad1b6b8636782d29e24dae517af13708af9676e8da56b00
CLI         056efa08ea16a99d2874776689c270f5eb6e605b476ffefb7e2e29b42a1f1c9d
CLI tests   16448dcb213fff7243316edeb2bbf61877819d27afac4aed1560df67b2d10187
config      6ccfee809ef7b642a3f8c6bf46ac80094d969c818a254a5cfa0044ab378ffe41
spec        363876314e3ae89dace227bcb8e758d625773926075ea17d42ab2a6c7eec5fe3
```

The landing receipt corrects a telemetry conflation: 13 focused tests plus
39 adjacent tests equal 52 total; it does not claim 52 adjacent tests.

## Verification

Focused:

```bash
.venv/bin/pytest -q \
  src/tac/witness_control/tests/test_taskspace_batch16_margin_base_scorer_cache_v1.py \
  tools/tests/test_materialize_taskspace_batch16_margin_base_scorer_cache_n600.py
```

Result: `13 passed`.

Adjacent:

```bash
.venv/bin/pytest -q \
  src/tac/witness_control/tests/test_taskspace_fresh_teacher_materializer_v1.py \
  src/tac/witness_control/tests/test_taskspace_fresh_scorer_plane_materializer_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_selected_preimage_program_v1.py
```

Result: `39 passed`.

Ruff, `py_compile`, source-hash replay, landing-receipt projected self-hash,
r4 strict replay, and the 2,222-lane registry validation passed.

## Governed command

```bash
.venv/bin/python tools/safe_run.py \
  --rss-mb 10240 --projected-gib 10.0 \
  --timeout 3600 --label g78_batch16_margin_base_n600 -- \
  .venv/bin/python \
    tools/materialize_taskspace_batch16_margin_base_scorer_cache_n600.py \
    .omx/research/configs/taskspace_batch16_margin_base_scorer_cache_n600_20260726.json \
    --materialize
```

The dispatch lane must be claimed immediately before launch and closed with a
terminal claim after success, refusal, or failure.

## Triality

DSL: exact typed preflight, immutable batch/stage receipts, and strict aggregate
loader.

DAG: owned source/G46/G51/V15/runtime custody -> fresh live batch preparation
-> exact batch-16 scorer forwards -> immutable global shards -> five
fragment-derived stages -> recursive aggregate reopen -> G72 compiler input.

Equations: `m = z_argmax - max(z_non-argmax) >= 0`; no per-component threshold,
proxy score, or candidate byte is introduced.

## STORES CONSULTED

- `CLAUDE.md` and `AGENTS.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/active_lane_dispatch_claims.md`
- exact G46, G51, fresh-V15, upstream scorer, source-video, r4 preflight, G78
  source/test/spec/receipt, and G81 refusal artifacts named above

## Pointer-delta honesty

Pointer delta is zero. The effective frontier remains 0.172. Readiness to
materialize encoder-side evidence is not a score result.
