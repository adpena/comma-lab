# Bug-hunter adversarial review: session landings 2026-05-07

Scope: the codec_pipeline session landings 2026-05-07 (orchestrator + 5 ops +
training callback + Shannon H2 loss). Excludes pr101_split_brotli_codec.py /
pr103_arithmetic_codec.py / derivers (separately reviewed) and test files
(code-bug hunt only).

Reviewer: Claude (subagent), one round of the 3-clean-pass gate. Non-blocking
findings: subsequent rounds need separate dispatch.

## Summary

- 6 total findings: 1 CRITICAL, 3 MEDIUM, 2 LOW
- 2 fixes landed (CRITICAL #1, MEDIUM #3) with regression tests
- 4 findings deferred (rationale per finding)
- Baseline tests: 137 passed before fix; 140 passed after (3 new regression
  tests added). PR101/PR103 codec tests: 46 passed (unchanged).
- Out-of-scope note: `scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh`
  is staged-deleted (`git status: D`); review skipped.

## Per-file findings

### src/tac/codec_pipeline.py

#### Finding 1 (CRITICAL): `Op2_PR103ArithmeticCodec.encode` does not couple `latent_hi_symbols` array length to `n_latent_hi_symbols` drain count

- **What**: `Op2_PR103ArithmeticCodec` exposes `latent_hi_symbols: Any = None`
  (an embedded array) and `n_latent_hi_symbols: int = 0` (decoder drain
  count) as two independent dataclass fields. Pre-fix, `encode()` simply
  recorded `self.n_latent_hi_symbols` into `op_state` regardless of whether
  the array length matched. The PR103 encoder embeds every symbol of the
  supplied array into the merged AC stream; the decoder drains exactly
  `n_latent_hi_symbols` symbols from that stream after the weight payload.
  Mismatched values silently produce a corrupt state_dict roundtrip:
  - `latent_hi_symbols=arr (len N)` + default `n_latent_hi_symbols=0`:
    decoder leaves N×AC symbols undrained, the merged AC `RangeDecoder`
    stream desynchronises and downstream tensor reshape may succeed with
    garbage values OR throw a confusing constriction error.
  - `n_latent_hi_symbols=K` + `latent_hi_symbols=None`: decoder tries to
    drain K symbols that were never embedded; same desync class.

- **Why it's a bug**: Fridrich (inverse-steganalysis discipline) — the wire
  format must be self-consistent under all op_state values, but the two
  fields could disagree silently. Quantizr (adversarial RE) — a competitor
  reading the wrap would notice the missing coupling immediately. Yousfi
  (validate gate) — `validate()` does not check this either; the gate is
  decorative for this bug class.

- **Evidence**: `grep latent_hi_symbols src/tac/tests/test_codec_pipeline*.py
  → 0 matches` confirms the bug class is uncovered. The standalone codec
  unit tests in `test_pr103_arithmetic_codec.py` exercise both fields but
  always set them consistently — the wrap layer is the one that drops the
  invariant.

- **Fix landed** (commit pending via subagent serializer):
  `src/tac/codec_pipeline.py` `Op2_PR103ArithmeticCodec.encode` now:
  1. derives the effective drain count from `len(latent_hi_symbols)` when
     non-None,
  2. raises `ValueError` if `n_latent_hi_symbols` was set non-zero AND
     disagrees with the array length,
  3. raises `ValueError` if `n_latent_hi_symbols > 0` while
     `latent_hi_symbols is None`,
  4. records the effective count in `op_state["n_latent_hi_symbols"]`,
     not the raw configured field.

  Three regression tests added in `src/tac/tests/test_codec_pipeline.py`:
  `test_op2_rejects_latent_hi_count_mismatch`,
  `test_op2_rejects_n_hi_without_latent_array`,
  `test_op2_auto_derives_n_hi_when_default_zero`.

#### Finding 4 (LOW): `Op1_PR101SplitBrotli.encode` runs `auto_select_byte_maps` twice when `auto_select=True`

- **What**: lines 156-172 of `codec_pipeline.py`. When `explicit_overrides
  is None` and `auto_select=True`, `encode_decoder_compact` runs
  `auto_select_byte_maps` internally, then the wrap calls it AGAIN to
  capture the result for `op_state["effective_byte_maps"]`. Auto-select
  is deterministic so correctness is fine; this is a CPU-time waste only
  (~30 brotli evaluations per encode, ~tens of ms on PR106-shaped tensors).

- **Why it's a bug**: Hotz (engineering rigor) — redundant work in a
  function that may run on every training-callback epoch.

- **Evidence**: encoder source at `pr101_split_brotli_codec.py:399-402`
  derives `effective_byte_maps` internally; the wrap then re-derives
  starting at `codec_pipeline.py:168`.

- **Fix**: `[deferred — minor performance, not correctness]`. The cleanest
  long-term remedy is to have `encode_decoder_compact` accept a
  `return_layout=True` kwarg returning the effective overrides (mirroring
  PR103's pattern). Out of scope for this review pass.

#### Finding 5 (LOW): `Op2_PR103ArithmeticCodec` has docstring overclaim relative to validate signature

- **What**: docstring at line 446-449 says decoder requires
  `section_lengths` with keys `br, hists, merged_ac, hi_hist`, but the
  encoder also populates `ac_fallback` (per the per-tensor AC fallback
  landed 2026-05-07). The decoder accepts `ac_fallback` as optional
  (defaults to 0), so existing decoders still work — but the docstring
  is no longer accurate.

- **Why it's a bug**: Contrarian — docstrings rot when wire format
  evolves; the readers next time will be confused.

- **Fix**: `[deferred — pure docstring drift, no functional impact;
  catch in next routine docstring sweep]`.

### src/tac/codec_pipeline_apogee_int.py

#### Finding 2 (MEDIUM): `Op3_ApogeeIntN_Substrate.decode` silently drops tensors not in `tensor_names`

- **What**: `decode()` lines 197-209 only writes tensors named in
  `op_state["tensor_names"]`. The encode side (lines 157-166) iterates
  `FIXED_STATE_SCHEMA` and only emits tensors that ARE in the input
  state_dict. If a caller bypasses `validate()` via
  `pipeline.encode(state_dict, skip_validate=True)` and passes a
  partial state_dict (some FIXED_STATE_SCHEMA tensors missing), the
  decoded substrate is also partial. Op3 has `transforms_state_dict=True`,
  so the next op (Op1 / Op2) receives the partial dict and will crash
  inside `encode_decoder_compact` / `encode_decoder_ac` with an opaque
  KeyError-style message instead of a clean validate failure.

- **Why it's a bug**: Selfcomp (wire-format expertise) — the substrate
  transform should either round-trip ALL input tensors or refuse cleanly
  when invariants are broken at a substrate boundary, not punt the
  failure to the next op.

- **Evidence**: validate() requires `FIXED_STATE_SCHEMA` schema match, so
  the bug only surfaces when `skip_validate=True`. Existing tests don't
  exercise this combination on Op3.

- **Fix**: `[deferred — bounded by validate() in normal flow; cleanest
  remedy is for Op3 to assert on its OWN encoder-side that all
  FIXED_STATE_SCHEMA tensors are present even under skip_validate, but
  this duplicates validate logic. Recommendation: in a follow-up review
  round, change validate() invariants to be partially executed even on
  skip_validate, OR have decode() error if `len(tensor_names) <
  len(FIXED_STATE_SCHEMA)`.]`.

### src/tac/codec_pipeline_mask.py

No critical/medium findings. `Op_NerVMaskCodec.encode` correctly raises
`NotImplementedError` when `pretrained_nerv_codec` is missing (Yousfi gate
working as intended). `Op_AV1BaselineMask` uses `tempfile.TemporaryDirectory`
so the `/tmp` rule is preserved (in-memory pipe-through pattern, not a
persisted artifact path). `pick_smallest_mask_codec` correctly degrades to
`RuntimeError` when all candidates fail (no silent fallback).

### src/tac/codec_pipeline_sensitivity.py

No critical findings. β-identity short-circuit is correctly threaded through
`context` for the in-pipeline encode→decode flow. The documented degenerate
case (β-identity decoded without context after persistence) is correctly
dominated by downstream ops in `CodecPipeline.decode`'s last-op-wins
contract. Schema gate is rigorous (rejects unknown sensitivity_source eagerly
at `encode()` even when `skip_validate=True`).

The Council short-circuit is the right call to close the 274,411 B PR106
ballooning; the tests cover it.

### src/tac/codec_pipeline_joint_admm.py

No critical findings. `_per_tensor_scale` correctly handles the all-zero
tensor (returns 1.0 sentinel; no divide-by-zero). `_quantise_to_int8` clamps
explicitly to `[-127, 127]` so the symmetric-int8 invariant holds. JCSv1
container preserves stream order under `unpack_jcsp_container` — the strict
`zip` of `parsed_streams` and `meta_streams` is sound because both follow
the encoder's `sorted(state_dict.items())` order, and the wrap also
defensively cross-checks stream names.

The `KIND_BALLE_HYPERPRIOR` decode branch is dead-but-correct (the encoder
hard-codes `KIND_ARITHMETIC_STATIC` for every stream), and the
`KIND_RAW_PASSTHROUGH` branch raises a clean `ValueError` instead of
crashing in `decode_qints_arithmetic`. The `substrate_aware_init=True`
STUB path correctly logs the WARN and records the finding without
mutating byte counts (no dead-flag bug).

### src/tac/codec_pipeline_full_stack.py

No critical findings. `Op4_FullStackOrchestrator` correctly delegates
encode/decode to `Op1_PR101SplitBrotli` and only carries the matrix audit
in `op_state`; decoders ignore the audit. `pick_smallest_stack` correctly
breaks ties by `CANONICAL_STACK_NAMES` order (deterministic).
`_default_output_dir` writes under `experiments/results/`, not `/tmp`.

### src/tac/codec_pipeline_deltaepszeta_callback.py

No critical findings. `__post_init__` rejects `/tmp` paths explicitly per
CLAUDE.md transient-evidence rule. `add_to_loss` correctly preserves
torch.Tensor dtype/device via `loss * 0 + 0` arithmetic when in stub mode.
`report()` rejects `epoch < 0`.

The `lambda_penalty` math is non-differentiable by design (bytes are an
integer count of codec output) and the docstring correctly notes that —
no Hotz overclaim.

### src/tac/shannon_h2_loss.py

#### Finding 3 (MEDIUM): degenerate-input branch in `shannon_h2_loss` returns wrong units

- **What**: `shannon_h2_loss` line 199-207 (pre-fix). When `n < 3` (fewer
  than 3 weights, no trigram available), the function returned
  `shannon_h0_loss(weights, n_bits=reduced_bits, ...)`. This is in
  units of bits-per-REDUCED-symbol; the rest of the function returns
  bits-per-FULL-symbol via the `(n_bits / reduced_bits)` rescale.
  Result: `h2_h0_ratio()` on a 1-tensor weight would return a value
  with mixed units, which could confuse a δεζ training signal.

- **Why it's a bug**: MacKay (MDL discipline) — every entropy claim must
  carry consistent units; mixing reduced-alphabet and full-alphabet bits
  silently is exactly the unit-drift class MDL aims to prevent.
  Shannon (LEAD) — "every score-improvement claim must trace back to
  bits"; bits in different alphabets are different quantities.

- **Evidence**: source diff in `shannon_h2_loss.py` lines 199-207
  pre-fix returned `shannon_h0_loss(... n_bits=reduced_bits ...)` with
  no rescale; the main branch line 231 returns
  `h2_bits_per_reduced_symbol * (n_bits / max(1, reduced_bits))`.

- **Fix landed** (commit pending via subagent serializer):
  the degenerate branch now applies the same `(n_bits / max(1, reduced_bits))`
  rescale so units match. Inline comment cites this finding.

#### Finding 6 (LOW): docstring claims "minimum = 0 (delta)" for H₀ but the soft-assignment surrogate has finite-temperature positive bias

- **What**: `shannon_h0_loss` docstring line 117 states "minimum = 0
  (delta)". Strictly speaking this is true only as `temperature → 0`;
  at finite temperature the softmax over bin centers spreads probability
  across bins even for a delta-distributed input, so `H₀_surrogate > 0`.
  This is well-known for soft-histograms; the docstring should note the
  positive bias.

- **Why it's a bug**: Contrarian — docstring overclaim. Not a code bug;
  could lead a δεζ training loop's stopping criterion to fire later than
  intended.

- **Fix**: `[deferred — pure docstring; not a runtime issue.]`.

## Council positions

- **Shannon (LEAD)** — endorse fix #1 (CRITICAL): the wire-format must be
  self-consistent. endorse fix #3 (MEDIUM): unit consistency is foundational
  for any MDL-style training signal. dissent on closing without #6: H₀'s
  finite-temperature bias should be documented even if not fixed.
- **Yousfi** — endorse fix #1: the validate gate must reject mismatched
  drain counts. The current `validate()` is decorative for this bug class.
- **Fridrich** — endorse fix #1: encode/decode roundtrip is the most basic
  forensic invariant; silent corruption fails inverse steganalysis hygiene.
- **Contrarian** — challenge #4 LOW deferral: redundant `auto_select` calls
  are wasted GPU on every training callback, not just academic. Fix soon.
  Otherwise endorse the review pass.
- **Quantizr** — endorse fix #1: a competitor would spot this within minutes
  of reading the wrap. Endorse #2 deferral: bounded by `validate()` in
  normal flow.
- **Hotz** — endorse fix #1: silent corruption is the worst class of bug;
  the regression tests prevent recurrence. Endorse #4 LOW deferral.
- **Selfcomp** — endorse fix #1: wire-format coupling is the canonical
  thing the wrap layer must own. Endorse #2 deferral pending follow-up
  on Op3 substrate-transform invariant.
- **MacKay** — endorse fix #3: bits-per-reduced-symbol vs bits-per-full-symbol
  was a real unit drift; fixed cleanly with consistent scaling. Endorse #6
  deferral as a follow-up docstring sweep.

## Verdict

- 1 CRITICAL finding → fix LANDED with regression tests; ready for next
  council round.
- 3 MEDIUM findings → 1 fix LANDED (#3); 2 deferred (#2 with follow-up
  recommendation; commentary preserved for next round).
- 2 LOW findings → record only; no fix.

**Recommendation**: ready for next council round. The CRITICAL bug class
(Op2 latent_hi mismatch) is structurally extinct via the encode-side
guard + 3 regression tests; the MEDIUM unit-drift in Shannon H₂ is fixed
with comment cross-reference. The four deferred findings are documented
inline + here, and the next adversarial pass should consider them as
known-unknowns rather than re-derive them.

## Cross-references

- Composition contract: `.omx/research/four_way_stack_composition_contract_20260507_claude.md`
- Canonical orchestrator: `tac.codec_pipeline`
- PR103 codec source: `tac.pr103_arithmetic_codec` (`encode_decoder_ac`,
  `decode_decoder_ac`, `validate_ac_savings`)
- PR101 codec source: `tac.pr101_split_brotli_codec` (`auto_select_byte_maps`,
  `encode_decoder_compact`, `validate_byte_map_savings`)
- Substrate-aware-init STUB: `Op_GammaJointADMM.substrate_aware_init` (forensic
  record only; underlying API not landed per `BalleHyperpriorCodec` grep)
- Lane registry mapping: `lane_codec_pipeline_*` lanes each correspond to
  one of the modules reviewed here.
