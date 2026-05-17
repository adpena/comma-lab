# Master-Gradient Partner WIP False-Authority Review - 2026-05-17

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch, lane claim, score claim, promotion claim, or method kill
is authorized by this review.

## Reviewed WIP

The following files are current dirty/partner WIP and were reviewed without
editing them:

- `src/tac/master_gradient.py`
- `tools/cathedral_autopilot_autonomous_loop.py`
- `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md`
- `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md`

This review builds on the already-landed routing ledgers:

- `.omx/research/master_gradient_raw_byte_finite_difference_adversarial_review_20260517_codex.md`
- `.omx/research/master_gradient_operator_response_plan_landed_20260517_codex.md`
- `.omx/research/omx_parent_markdown_and_fec6_selector_operator_followup_20260517_codex.md`

## Verdict

Composite **B + D** again:

- **B:** the WIP helper still carries a cargo-culted raw archive-byte
  coordinate system.
- **D:** the useful probe object remains a packet-valid operator response
  matrix, not an `(N_archive_bytes, 3)` tensor.

The master-gradient idea is still valuable if it is defined over valid codec
symbols or valid packet mutation operators. The current WIP should not land as
an authority-bearing helper until the false-authority surfaces below are
removed or explicitly demoted.

## Findings

### F1 - Raw archive-byte tensor reintroduced

`src/tac/master_gradient.py` defines `MasterGradient.load_gradient()` as shape
`(n_bytes, 3)` and describes `n_bytes` as archive-byte count. It also documents
`finite_difference_bit_flip` as an accepted `measurement_method`.

This contradicts the current routing rule: archive bytes are ZIP/container
bytes, compressed stream bytes, codec grammar bytes, and payload bytes mixed
together. The derivative is not well-defined at that grain even if the outer
ZIP metadata is rebuilt. A valid object must be one of:

- `logical_section_parameter` response rows,
- `grammar_aware_operator` response rows,
- codec-symbol gradients with a parser manifest and packet lowering proof, or
- a byte-different packet candidate with inflate and consumption proof.

### F2 - `predict_delta_s(byte_modifications)` is unsafe for packet bytes

The WIP helper accepts `{byte_idx: delta}` and projects through the loaded
gradient. That API makes raw archive-byte edits look mathematically meaningful.
For FEC6/PR101-style packets, a byte delta can corrupt:

- ZIP local headers,
- central directory fields,
- CRCs,
- compressed streams,
- fixed-Huffman selector state,
- FP4/codebook symbols with discrete nonlocal decode effects.

The API should be renamed or redesigned to accept `CandidateModificationSpec`
rows that are already packet-valid and tied to a parser-proven section.

### F3 - Autopilot "rerank" currently does not rerank by master gradient

The dirty `tools/cathedral_autopilot_autonomous_loop.py` diff adds
`rerank_candidates_via_master_gradient`, but when an anchor is present it keeps
`predicted_score_delta` unchanged because candidates do not yet carry byte or
operator modification specs.

This is acceptable only if named as an anchor-presence diagnostic. Calling it a
master-gradient reranker is false authority: it can change explanations while
leaving ordering untouched.

### F4 - Anchor lookup is too weak for score-lowering authority

The autopilot diff extracts the first 16+ hex token from `candidate.notes` or
`candidate_id` and treats it as an archive identifier. That can silently pick a
short SHA prefix or unrelated hex token. Since the ledger stores full
`archive_sha256`, exact lookup can also silently miss the intended anchor.

For authority-bearing routing, candidate rows need a structured
`archive_sha256` field or a typed source manifest reference. Notes parsing can
remain advisory only.

### F5 - Current campaign plan still contains stale score bands

The dirty campaign plan still advertises a `0.16-0.17 [contest-CPU]` trajectory
and an L5 Wyner-Ziv step at `-0.008` to `-0.015`. Current L5 rate-only review
says the closed-form pose-stream shrink band is about `-0.0019` to `-0.0032`
unless decoded-pose/frame component movement is proven.

Keep the ambitious campaign goal as a research target, but do not let those
numbers become dispatch or promotion priors.

## Required Unwind Before Landing The WIP Helper

1. Replace `(N_archive_bytes, 3)` with an explicit response-coordinate enum:
   `valid_mutation_operator`, `logical_section_parameter`, or
   `codec_symbol_parameter`.
2. Delete or fail-close `finite_difference_bit_flip` unless it is scoped to a
   parser-proven codec symbol and followed by packet rebuild, inflate proof,
   and byte-consumption proof.
3. Replace `predict_delta_s(byte_modifications)` with a projection over typed
   `CandidateModificationSpec` / operator rows.
4. Rename the autopilot hook to "anchor presence" or implement a real
   projection from candidate modification specs before calling it a reranker.
5. Replace notes-based SHA extraction with structured `archive_sha256` input.
6. Keep all outputs `score_claim=false`, `promotion_eligible=false`,
   `ready_for_provider_dispatch=false`, and
   `ready_for_exact_eval_dispatch=false` until an exact result-review packet
   lands.

## Guard Added

The focused regression test
`test_raw_archive_byte_remains_blocked_even_if_container_proofs_are_claimed`
was added to `src/tac/tests/test_master_gradient_feasibility.py`. It ensures
that raw archive-byte gradients stay blocked even when a caller claims ZIP
headers, CRCs, repacking, and inflate success. Those proofs are necessary for
operator-grain probes, but they do not make raw archive-byte derivatives valid.

## Next Valid Build Step

Do not land `src/tac/master_gradient.py` as currently written. The next valid
code artifact is either:

1. a `CandidateModificationSpec` schema consumed by the autopilot lens, or
2. a second concrete operator materializer after the FEC6 selector audit,
   producing one byte-different packet with inflate and byte-consumption proof.

Both routes preserve the master-gradient intuition while keeping the coordinate
system on the valid packet manifold.
