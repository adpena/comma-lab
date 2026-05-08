# AC bolt-on post-coarsening measured-config retirement - 2026-05-08

## Scope

This ledger preserves the reported AC bolt-on result as a scoped negative, not
as a family kill.

Lane: `ac_bolt_on_post_coarsening`

Measured config: AC on the PR101 Path-B step-6 K-coarsened stream with
`nsym=5`, `offset=2`, and the reported K-coarsened vector.

Evidence grade: `[CPU-prep proxy]`

## Reported result

- Brotli q11 joint payload: `737` bytes
- AC joint payload: `1347` bytes
- Delta: AC loses by `610` bytes at this config

Interpretation: for this heavily coarsened 5-symbol stream, brotli q11 appears
to be near the useful static-Huffman regime and the AC model/header overhead
dominates. This is a measured-config retirement only.

## Composition review

- Additive signal: the result identifies a post-coarsening alphabet regime
  where AC is unlikely to pay. Use it as a routing rule for entropy work:
  wide/nonstationary alphabets first, collapsed 5-symbol streams last.
- Antagonism: applying a model-heavy AC bolt-on after aggressive K-coarsening
  is antagonistic with byte savings because model/header cost is no longer
  amortized by coding gain.
- Orthogonality: this does not affect score-aware K allocation, Jacobian
  pullback, architecture shrink, sparsity retraining, or other distortion-side
  lanes. It only narrows one post-coarsening entropy subcase.
- VStack rescue path: AC/rANS/FSE/constriction should be tested earlier in the
  stack, before heavy coarsening collapses the alphabet, or with conditional
  models that exploit within-tensor distribution shifts.
- HStack path: joint encoding across all 28 tensors with shared headers remains
  open, especially if the tensor count/header overhead is the larger byte mass.

## Reactivation criteria

Reopen AC/range/ANS entropy coding when any of these are true:

1. Pre-coarsening raw int8 stream is the target, with a larger alphabet where
   adaptive or conditional models can beat static Huffman.
2. A lower-rms operating point keeps a wider alphabet than the retired
   5-symbol stream.
3. The engine is FSE-tANS, rANS, constriction, or context-mixing AC with
   non-static probability models and explicit header accounting.
4. Joint AC across all 28 tensors removes or amortizes the per-tensor header.
5. The result is evaluated in a byte-closed archive with deterministic decode,
   not only a standalone payload proxy.

Status: `measured-config retired`. Family status:
`DEFERRED-pending-research`. No kill language is justified.
