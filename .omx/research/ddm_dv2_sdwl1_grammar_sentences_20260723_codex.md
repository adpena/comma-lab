# Scorer-Derived Worldsheet Language v1 — n600 findings

date_utc: 2026-07-23
lane_id: lane_ddm_dv2_grammar_sentences_20260723
research_only: true
execution_allowed: false
axis: `[macOS-CPU frozen-scorer advisory]`
score_claim: false
promotion_eligible: false
pointer_moved: false
main_landing_review_required: true

## Outcome

The new typed grammar is **Scorer-Derived Worldsheet Language v1
(SDWL1)**. Its implementation is
`src/tac/optimization/ddm_dv2_sdwl1.py`; its bounded measurement entrypoint is
`tools/measure_ddm_dv2_sdwl1.py`; and its complete n600 receipt is
`.omx/research/ddm_dv2_sdwl1_n600_20260723/receipt.json`.

The selected measured representation is one typed-section, causal-delta
whole-clip sentence:

- complete outer-zlib9 bytes: **68,464**;
- complete outer payload SHA-256:
  `2b67caa997f353d1aee25b66737fcae1c0067deb92a0850401ce56f2f2537cab`;
- described records: 6,600 (11 per pair);
- described non-padding scalar facts: 45,600 (76 per pair);
- complete outer bytes per described scalar fact: 1.5014035087719297;
- exact semantic parse-back: true.

This is a language-rate measurement over the declared fact inventory. It is not
a pixel reconstruction, receiver-closed witness, candidate archive, evaluator
run, or contest score.

## Complete key rows

All values below are measured from complete framed objects after the repository
Catalog #557 left/up arithmetic coder and one Catalog #574 zlib level-9 outer
stream. The independent baseline resets the SDWL1 description and arithmetic
state at each pair, frames all 600 descriptions, then applies the one complete
outer stream required by the comparison.

| row | inner bytes | outer bytes | gain vs same-layout independent | outer SHA-256 |
|---|---:|---:|---:|---|
| independent monolithic absolute | 5,688,930 | 421,991 | — | `524af2f54eeeadeec903bbb70ecca907fc1e2dc7f2472aec141b70f2bf0506f1` |
| whole monolithic absolute | 128,072 | 116,423 | 305,568 | `f8498e26e6c679b0754a55ffd059f93a324d5aff50e4d9b42f1452ede3e0d576` |
| whole monolithic causal delta | 117,736 | 91,958 | 330,033 | `517d970f99dfaa4c69643281ac2dc50a71e6250ab72375717a61b5d8c2528bbd` |
| independent typed-section absolute | 6,781,910 | 521,139 | — | `5cf3ff1d5e2b13c768779ea2a6650dfc3d3973af0cea00d5cdc115601516fb39` |
| whole typed-section absolute | 106,277 | 91,903 | 429,236 | `541b55fd0bf1503a0da018336d89dfb98c0d0277f91efe6ca460d00fba1c59e0` |
| **whole typed-section causal delta** | **120,404** | **68,464** | **452,675** | `2b67caa997f353d1aee25b66737fcae1c0067deb92a0850401ce56f2f2537cab` |
| independent stratum-section absolute | 9,510,754 | 727,709 | — | `c82d974b12a051a9207eb0a77daff465036bb747cc83f33a5674f19a890af826` |
| whole stratum-section absolute | 115,354 | 92,457 | 635,252 | `43921a41545636fe1d919498a695a19fbb9a40164428d53c8cf58359eb5dde9a` |
| whole stratum-section causal delta | 182,831 | 75,197 | 652,512 | `8de6ea3fe3ca9598b7afd813247994e9f1b1427b27f7af93273b115fccca3d80` |

For the admitted typed layout, whole-sentence causal sharing removes 452,675
bytes (86.8626%) versus 600 independent typed descriptions. Causal deltas remove
23,439 bytes (25.5041%) versus the whole typed absolute sentence. The selected
row is also 353,527 bytes (83.7760%) below the smallest independent baseline,
which was monolithic.

## Exporter-realizable syntax

- Packet magic/version: `SDWL1PK\0`, version 1.
- Independent-collection magic/version: `SDWL1IC\0`, version 1.
- Every section is framed as a four-byte tag, little-endian unsigned 64-bit
  payload length, 32-byte payload SHA-256, and exact payload bytes.
- The lexicon and subject schema are sorted-key, compact, canonical UTF-8 JSON.
- Numeric sections are encoded only by
  `tac.optimization.arith_selfcomp_rate_coders.encode_spatial_context_arithmetic`
  and decoded by its matching strict decoder.
- The complete framed object is compressed with zlib level 9. Truncation,
  trailers, unknown/duplicate tags, noncanonical JSON/arithmetic, section/body
  hash drift, schema drift, nonzero padding, and semantic-hash drift fail closed.
- Pair zero is absolute. In the causal arm, later discrete facts are signed
  64-bit deltas and pose bit patterns are exact modulo-\(2^{64}\) deltas.
- Every admitted row was decompressed, strictly decoded, and compared with the
  exact 45,600-scalar semantic tensor.

## Measured production inventory

The n600 tensor produced 3,000 partition-cell, 3,000 separatrix, and 600
pair-screw subjects. Temporal inference produced 11 declarations, 4,928
deformations, 1,062 topology deltas (530 births and 532 deaths), 599
transports, and zero holds.

The following declared vocabulary had zero n600 use and is absent from the
base lexicon:

- subjects: Lane chart, resize-range atom;
- predicates: hold, omit-kernel, project-range;
- modifiers: chroma phase, ERF band, head normal, road frame, scale band.

Those types remain defined and provenance-bound in the grammar so a future
measured inventory cannot introduce anonymous structure, but they are not
charged in the selected sentence.

## MDL pruning and admitted dimensions

Only the typed-section layout and causal-delta temporal mode are admitted.
Best complete payloads for the other layouts were 91,958 bytes (monolithic)
and 75,197 bytes (stratum section), versus 68,464 bytes selected.

All four one-at-a-time, same-semantics syntax dimensions were measured and
pruned:

| counterfactual | complete outer bytes | delta vs selected control | verdict |
|---|---:|---:|---|
| explicit frame indices | 69,402 | +938 | PRUNE |
| repeated per-pair provenance digest | 68,690 | +226 | PRUNE |
| derived event masks | 68,621 | +157 | PRUNE |
| split topology birth/death vocabulary | 68,475 | +11 | PRUNE |

No unmeasured extra dimension is admitted. All 24 counterfactual payloads across
the three layouts and two temporal modes remain preserved in the receipt
directory.

## Custody and re-derivation

The read-only source was
`/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`,
5,078,017,610 bytes, SHA-256
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
The harness direct-mapped the three ZIP_STORED NPY members and did not copy or
mutate the source.

Exact command:

```bash
.venv/bin/python tools/measure_ddm_dv2_sdwl1.py \
  --source-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --output-dir .omx/research/ddm_dv2_sdwl1_n600_20260723 \
  --n-pairs 600
```

The same command with `--resume` rehashed the source, reparsed every preserved
payload, and reproduced the byte-identical final receipt SHA-256
`efc43fcda1f12f28df2b6059cd5e51e7ee2509a356d99b59e317b253927a709c`.

## Verdict scope

`CONFIRMED` only for exact syntax round-trip and complete outer-zlib byte rate
of this declared SDWL1 fact inventory on the frozen n600 cache. The result says
nothing about pixel reconstruction, receiver closure, frozen evaluator output,
contest CPU/CUDA parity, archive legality, or score movement. The canonical
frontier pointer is unchanged, and MAIN must review before landing.
