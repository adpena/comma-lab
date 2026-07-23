# SDWL1 round-one adversarial implementation review

Date: 2026-07-23

Scope: delegated v10 DDM language-layer work only. This is a rejection-and-repair
receipt for the first implementation pass; it is not a score claim or a witness
claim.

Evidence axis: `[macOS-CPU frozen-scorer advisory]`

Authority: `score_claim=false`

## Disposition

`REJECT_UNTIL_REPAIRED`

The typed vocabulary, exact fact inventory, strict framing, real Catalog #557
context-arithmetic streams, complete Catalog #574 outer-deflate measurement, and
three layout modes are structurally present. The following defects must be fixed
before n600:

1. The redundant event-mask counterfactual serializes an all-zero array. Replace
   it with a deterministic mask derived from the encoded causal stream, and make
   the decoder recompute and verify it. A fake empty sidecar is not admissible.
2. The repeated-provenance counterfactual duplicates provenance only once inside
   the lexicon. Add a strict framed section that repeats a canonical provenance
   digest once per described pair, with exact decoder verification. Add the
   section tag to the canonical ordering.
3. Distinguish 11 typed records per pair from 76 described scalar facts per pair
   (5 cell rows x 8, 5 separatrix rows x 6, 1 screw row x 6). Report both, and
   compute bytes-per-described-fact from the 76 non-padding scalar facts.
4. Add production counts derived from the semantic tensor. Predicates are
   deterministic grammar productions inferred from temporal values; they are not
   separately charged tokens. Do not claim `HOLD` usage without measuring it.
5. Verify modulo-2^64 causal differencing of arbitrary float64 pose bit patterns
   by exact round-trip tests.
6. Add the bounded measurement CLI with direct ZIP_STORED NPY memmap, strict
   source bytes/SHA-256 custody, deterministic/resumable stage receipts, storage
   preflight, atomic outputs, real n600 default, and no large local copies.
7. Add focused tests for all layouts, absolute/delta/independent parse-back,
   malformed/truncated/trailing rejection, counterfactual verification, outer
   deflate determinism, direct memmap, and derivation coverage.
8. `upstream/modules.py` and `upstream/frame_utils.py` are absent only in this
   isolated worktree; they are present in the canonical parent checkout. Keep
   the canonical repository-relative provenance names.

## Acceptance boundary

No n600 run in the repair pass. No launch, score, witness, promotion, or old
lineage work. The parent reviewer owns the real n600 measurement, durable
findings/equations/DAG receipts, serializer commit, and MAIN landing request.

## Verdict scope

This rejection applies only to the first SDWL1 implementation pass and the two
invalid same-semantics counterfactual constructions. It does not reject the
scorer-derived language family, the fact inventory, Catalog #557, Catalog #574,
or any witness backend.
