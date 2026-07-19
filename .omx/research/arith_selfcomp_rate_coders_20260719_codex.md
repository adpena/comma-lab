# Arithmetic/Selfcomp rate coders: measured local audit

Date: 2026-07-19 UTC
Lane: `lane_arith_selfcomp_rate_coders_20260719`
Status: `research_only=true`; **PARTIAL / fail-closed blockers retained**
Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**
Authority: isolated local build and read-only measurement. No launch, paid
dispatch, contest score, promotion, submission, inflate dependency, or pointer
authority. MAIN landing review is required.

## Verdict

**MEASURED:** on the exact current EMA donor, general-purpose Brotli remains
the smallest complete int8 coder: `63,394 B` for 72,695 base-weight elements
and `20,518 B` for 38,400 pair-code elements, including per-tensor framing and
quantization metadata. True left/up same-channel context is not beneficial:
repository context/IID is `1.41719x` on base weights and `1.04010x` on pair
codes; constriction context/IID is `1.50960x` and `1.05585x` respectively.
This rejects these concrete spatial sign/magnitude models, not arithmetic or
context coding as a family.

**MEASURED:** the canonical PDW1 re-derives to exactly `338 B`, SHA-256
`84a49d802dc5bd9c416013fd71bc6f08655a2f3c23c249374469a4dc4d8ee275`.
Exact parse-back sizes are raw `338 B`, Brotli-Q11 `286 B`, LZMA `316 B`, and
zstd-19 CLI `271 B`. The 138-byte PDW2 remains **DERIVED ONLY** because no
strict PDW2 encoder/decoder exists; no entropy byte row is claimed.

**MEASURED rate-domain only:** a classical ternary shared-exponent block-FP
baseline plus LZMA lands at `1.0050497 bits/parameter` over all 111,095 donor
parameters (`block=32`, threshold `0.25`), within `0.0119503 bpp` of the
preregistered approximately `1.017 bpp` target. It is **not admissible**:
matched realized-through-R `d_seg` at real `n>=24` is owed. Per the 10:40:29Z
MAIN clarification, this landing owns only the classical baseline; it makes no
learned-method claim and leaves the learned arm to its owner.

**BLOCKED:** the settled n24 Seg-secant receipts retain stream hashes and exact
Brotli/zstd totals, but not the signed-int32 numerator payload bytes. Therefore
LZMA, constriction-spatial, and zigzag+RLE+range rows for those exact streams
were not fabricated. Exact rederivation is still required before novel-coder
ratios or a novel-coder waterfill can be reported. Among the two complete
settled coders, Brotli is smaller at all ten points; the existing Brotli
waterfill remains `MEASURED_SECANT_KKT_CANDIDATE`, scoped to measured adjacent
segments and conditional n600-equivalent range bytes.

## Durable machine receipt

The committed JSON receipt is
`.omx/research/arith_selfcomp_rate_coders_20260719_receipt.json`, `2,057,039 B`,
SHA-256 `8cd25a6bde36676285326ac10d49d041e9f58deb7ffb5f03e518a2185435490d`.
It contains the complete 18-tensor table: source/int8 shapes and dtypes,
per-tensor int8 hashes and scales, reconstruction MSE, framing bytes, every
coder result and parse-back state, explicit spatial transform, every block-FP
configuration, qint/exponent hashes, and the donor/source hashes. This is the
reusable response to the late MAIN directive; the Markdown table below is only
a summary.

The read-only donor is
`levelset_witness_ema_BEST.npz`, `458,622 B`, SHA-256
`6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca`.
The prior `61,842 B` base and `20,355 B` code rows are displayed in the JSON as
prior-receipt rows, not silently equated to this invocation's fully framed
per-tensor totals. The historical Selfcomp `52.6 KB / 6.54 bpp` row is also
kept separate because checkpoint equivalence is false.

| donor section | elements | Brotli | zstd | LZMA | repo IID | repo spatial | constriction spatial | zigzag/RLE/range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base weights | 72,695 | 63,394 | 64,196 | 65,024 | 66,322 | 93,991 | 100,120 | 120,279 |
| pair code | 38,400 | 20,518 | 23,594 | 20,671 | 35,989 | 37,432 | 37,999 | 67,863 |

All sizes count complete per-tensor frames, model tables, termination, tensor
names, and quantizer scale metadata. Constriction is encode-side research only
and needs `constriction` to decode; it is wire-incompatible with the repository
`RangeDecoder`. No production or `inflate.py` dependency was introduced.

## Settled n24 residual rows and exact blocker

These ratios are **MEASURED existing-coder rows** imported only after validating
both immutable chunk receipts by SHA-256. They cover 48 streams per point.

| point | Brotli B/pair | zstd B/pair | zstd/Brotli | `d_seg` | `d_pose` |
|---|---:|---:|---:|---:|---:|
| source | 2,222,946.21 | 2,502,807.96 | 1.125897 | 0 | 0 |
| margin 0.01 | 2,222,625.12 | 2,502,564.58 | 1.125950 | 1.86496e-5 | 1.79192e-8 |
| margin 0.03 | 2,222,173.38 | 2,501,966.79 | 1.125910 | 2.09808e-5 | 8.82571e-8 |
| margin 0.1 | 2,220,441.00 | 2,499,911.42 | 1.125863 | 2.56432e-5 | 3.09306e-7 |
| margin 0.3 | 2,214,597.46 | 2,493,139.21 | 1.125775 | 3.92066e-5 | 8.43934e-7 |
| precision 1 | 1,770,993.33 | 1,991,528.79 | 1.124526 | 1.62760e-4 | 3.86826e-5 |
| precision 2 | 1,313,066.92 | 1,480,862.88 | 1.127789 | 1.82470e-4 | 7.52753e-5 |
| precision 3 | 1,145,117.33 | 1,290,867.29 | 1.127279 | 1.73993e-4 | 1.41547e-4 |
| stride 8 | 1,139,842.04 | 1,281,615.46 | 1.124380 | 2.13619e-2 | 8.49763e-1 |
| stride 16 | 1,119,166.46 | 1,259,914.33 | 1.125761 | 7.54929e-3 | 1.02060 |

Exact blocker token:
`SETTLED_RECEIPTS_PRESERVE_STREAM_HASHES_AND_BROTLI_ZSTD_TOTALS_NOT_RESIDUAL_BYTES; LZMA_CONSTRICTION_ZIGZAG_RLE_REQUIRE_EXACT_REDERIVATION`.
The new runner accepts explicit `.npy`/`.npz` residual points and then measures
full exact parse-back ladders, but no absent bytes are inferred from a hash.

## Classical block-FP and allocator reconciliation

The baseline uses the existing classical `tac.block_fp_codec`: per-row shared
float32 exponent, ternary qint payload, exact header accounting, then a general-
purpose byte coder. It sweeps block sizes `{8,16,32}` and thresholds
`{0.25,0.30,0.35,0.40,0.50,0.75}`. The nearest-rate row is 13,957 fully framed
bytes over 111,095 parameters (`1.0050497 bpp`, LZMA). Lower-rate rows are not
evidence of quality; their weight-domain distortion rises and no scorer was
run.

Sensitivity allocation and block-FP are potentially composable only in this
order: the sensitivity allocator first assigns precision/block policy, then
block-FP encodes each assigned tensor. They are substitutes if both independently
quantize the same coefficient. No allocation was applied here, so the receipt
labels composition `UNMEASURED_NO_ALLOCATION_APPLIED` and matched real `d_seg`
`OWED_MATCHED_REALIZED_DSEG_N_GE_24`.

## Routing and #553 amendment

| payload | consumer | authority / route |
|---|---|---|
| PDW1 | #539 | Measured 338-byte frozen-head target/facet diagnostic; spatial pullback and receiver remain outside these bytes. |
| PDW2 | #553 | Derived 138-byte gauge-fixed construction only; no strict coded row. |
| donor weights | weight receiver | Current EMA per-tensor int8/block-FP table; no historical checkpoint equivalence. |
| `dxi` / pair code | pair-code receiver | Separate 38,400-element section, never merged into base weights. |
| residual | #386 / #536 | Existing per-class carrier/residual kit and measured waterfill; novel coders blocked on exact payload rederivation. |

Proposed #553 accounting amendment: any future PDW2 entropy row must count
strict framing, gauge/reference metadata, every model table, termination, and
decoder/source dependency byte; it must prove encode -> fresh decode -> encode
identity and the five-condition near-tie gate before changing
`DERIVED_ONLY_NO_STRICT_ENCODER`.

## Implementation, review, and triality

- `src/tac/optimization/arith_selfcomp_rate_coders.py` provides deterministic
  strict frames, IID and true spatial sign/magnitude models, constriction's
  separate wire format, zigzag/RLE/range, byte coders, and classical block-FP
  accounting.
- `tools/measure_arith_selfcomp_rate_coders.py` is write-once, read-only on
  sources, hashes every input, re-derives PDW1, emits per-tensor JSON, validates
  settled n24 receipt custody, and fails closed on missing residual bytes or
  matched scorer results.
- `src/tac/tests/test_arith_selfcomp_rate_coders.py` covers signed extrema,
  determinism, exact parse-back, trailer/truncation rejection, genuine
  left/up-same-channel contexts, lazy optional dependencies, byte accounting,
  authority labels, and settled-receipt hash refusal.

Triality: equations are the executable sign/magnitude frequency tables,
block-FP quantizer, ratios, and imported #536 secants; DAG/evidence is the
write-once receipt plus bound source hashes; DSL/control is explicitly absent
because this is research-only and adds no launch flag. No pointer or sacred
run byte changed.

Self-review: round 1 fixed the donor section classifier, complete matrix/vector
spatial transforms, and zstd CLI fallback. Round 2 added entropy coding to the
classical block-FP comparison, preserved the residual payload blocker, and
made the late MAIN classical-only boundary machine-readable. Round 3 validated
receipt custody and round 4 removed Markdown-only diff-check findings. The
maximum five rounds is respected.

## STORES CONSULTED

- delegated wrapped prompt and both Codex inboxes;
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and
  `docs/operating_manual_craft_handoff.md` (SHA-256
  `40d157a039d4dd242bfb189d53e6b82abcc5d037adceb0a52c9bb2956903f212`);
- canonical lane, subagent-progress, frontier, probe, posterior, and latest
  sister-agent research surfaces required by preflight;
- immutable Seg-secant v2 chunk receipts and composed n24 curve;
- exact gt_n600 cache, frozen SegNet head, and current EMA donor checkpoint.

`verdict_scope`: this landing settles the named coder implementations on the
measured PDW1 and current donor and preserves settled Brotli/zstd n24 evidence.
It does not settle other context models, reconstruct absent residual bytes,
establish matched block-FP quality, close a receiver archive, or authorize any
score/promotion claim.
