# Bug-hunter v2 adversarial review: session landings 2026-05-07

Round 2 against the canonical CodecPipeline + cathedral refactor. Mandate
inversion: prior round deferred 4 findings; this round REQUIRES that every
finding (CRITICAL + MEDIUM + LOW) is fixed with a regression test, no
deferrals.

Reviewer: Claude (subagent), 8-perspective adversarial pass (Shannon LEAD,
Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay).

## Summary

- **6 total findings**: 1 prior MEDIUM (re-opened), 1 prior MEDIUM-docstring
  (re-opened), 2 prior LOW (re-opened), **2 NEW MEDIUM** discovered in this
  round.
- **6 fixes LANDED via 6 commits** through `tools/subagent_commit_serializer.py`.
  No deferrals.
- **Tests**: prior baseline 75 across the in-scope files; post-fix 90 (+15
  regression tests landed in this round; +parallel commits added the
  contest_rate_distortion_theorems suite separately).
- **Regression sweep**: 199/199 passing across all codec_pipeline +
  contest + shannon + build_deltaepszeta + per_tensor_shannon +
  deferred_dispatch_playbook test files.

## Per-file findings

### src/tac/codec_pipeline_apogee_int.py

#### Finding 1 (MEDIUM, re-opened): `Op3_ApogeeIntN_Substrate.decode` silently produces a partial state_dict when `skip_validate=True`

- **What**: pre-fix, `decode()` only wrote tensors named in
  `op_state["tensor_names"]`. The encode side iterates `FIXED_STATE_SCHEMA`
  and only emits tensors that ARE in the input state_dict. If a caller
  bypassed `validate()` via `pipeline.encode(state_dict, skip_validate=
  True)` and supplied a state_dict missing some FIXED_STATE_SCHEMA
  tensors, the decoded substrate was also partial. With
  `transforms_state_dict=True`, the partial dict then fed downstream
  Op1/Op2 encoders that crashed with opaque KeyError-style messages —
  silent substrate corruption punted to the next op.

- **Why it's a bug** (Selfcomp / Yousfi): the substrate transform must
  round-trip ALL input tensors or refuse cleanly when invariants are
  broken at THIS substrate boundary, not punt the failure to the next op
  with a confusing error.

- **Evidence**: re-derived prior bug-hunter finding #2 (`bug_hunter_review_session_landings_20260507_claude.md`).

- **Fix landed**: `b691d465`. `Op3_ApogeeIntN_Substrate.decode` now raises
  `ValueError("partial substrate refused — blob carries N tensors but
  FIXED_STATE_SCHEMA has M. Missing: [...]")` when `n_tens !=
  len(FIXED_STATE_SCHEMA)`, naming the missing schema tensors. Encode
  side unchanged; happy-path roundtrips remain bit-faithful.

- **Regression tests** (3, in
  `src/tac/tests/test_codec_pipeline_apogee_int.py`):
  - `test_op3_decode_refuses_partial_substrate_under_skip_validate`
  - `test_op3_pipeline_skip_validate_partial_state_dict_fails_at_decode`
  - `test_op3_full_state_dict_skip_validate_still_roundtrips`

### src/tac/codec_pipeline.py

#### Finding 2 (MEDIUM, re-opened): `Op2_PR103ArithmeticCodec` docstring lists 4 keys but encoder populates 5

- **What**: pre-fix docstring said decoder requires `section_lengths` with
  keys `br, hists, merged_ac, hi_hist`. The encoder also populates
  `ac_fallback` (per-tensor AC fallback landed 2026-05-07,
  `569e5ca8`). Decoder accepted `ac_fallback` as optional so callers
  worked, but the docstring was no longer accurate.

- **Why it's a bug** (Contrarian): docstrings rot when wire format
  evolves; the next reader will be confused about the contract.

- **Fix landed**: `eaea6c26`. Docstring now enumerates all 5 keys with
  what each one carries, plus the `ac_fallback_set` op_state field that
  the encoder records and the decoder reads.

- **Regression test** (1, in `src/tac/tests/test_codec_pipeline.py`):
  - `test_op2_docstring_lists_all_five_section_keys` — pins all 5
    section_lengths keys + `ac_fallback_set` mention in the class
    docstring; future drift will fail the test instead of silently
    rotting the wire-format contract.

#### Finding 3 (LOW, re-opened): `Op1_PR101SplitBrotli.encode` runs `auto_select_byte_maps` twice when `auto_select=True`

- **What**: pre-fix, when `auto_select=True` and `explicit_overrides is
  None`, `encode_decoder_compact(..., auto_select=True)` ran
  `auto_select_byte_maps` internally; then the wrap re-ran it to
  populate `op_state["effective_byte_maps"]`. Doubled per-encode CPU
  cost (~30 extra brotli evals per encode) for zero correctness
  benefit.

- **Why it's a bug** (Hotz / Contrarian challenge of prior deferral):
  redundant work in a function that may run on every training-callback
  epoch. Not academic — training-callback fires often.

- **Fix landed**: `d81709c0`. Wrap now computes `effective_byte_maps`
  once (from explicit_overrides, the auto_select path, or PR101
  defaults) and threads it explicitly into `encode_decoder_compact`
  with `auto_select=False` so the encoder doesn't re-derive. Byte
  output unchanged.

- **Regression tests** (2, in `src/tac/tests/test_codec_pipeline.py`):
  - `test_op1_auto_select_runs_exactly_once_per_encode` —
    monkeypatch-counts `auto_select_byte_maps` invocations per encode;
    asserts exactly 1.
  - `test_op1_auto_select_roundtrip_byte_faithful` — sanity that the
    one-call path still roundtrips through decode.

### src/tac/shannon_h2_loss.py

#### Finding 4 (LOW, re-opened): `shannon_h0_loss` docstring overclaims "minimum = 0 (delta)"

- **What**: pre-fix docstring said "Maximum value = n_bits (uniform
  distribution); minimum = 0 (delta)". For the finite-temperature
  soft-assignment surrogate, the minimum is **strictly positive** —
  softmax over bin centers spreads probability across multiple bins
  even for a delta input, producing residual entropy on the order of a
  few tenths of a bit at typical training temperature 1.0.

- **Why it's a bug** (Shannon LEAD / MacKay): training stopping criteria
  that rely on H₀ hitting exactly 0 will fire later than intended (or
  never).

- **Fix landed**: `7e101fd8`. Docstring now distinguishes:
  - **true H₀ (hard histogram)** in [0, n_bits]
  - **surrogate H₀ (finite-T softmax)** strictly > 0 on delta inputs;
    bias decreases as temperature → 0
  - guidance: treat surrogate as upper bound on H₀ for training-loss
    purposes; use `temperature` < 1 for a tighter surrogate.

- **Regression tests** (2, in `src/tac/tests/test_shannon_h2_loss.py`):
  - `test_h0_finite_temperature_bias_documented` — empirically asserts
    surrogate(delta) > 0 at T ∈ {1.0, 0.5, 0.1} and that the bias
    decreases monotonically as T decreases.
  - `test_h0_docstring_warns_about_finite_temperature_bias` — pins the
    docstring to mention "finite-temperature" + "bias" so future
    readers cannot accidentally re-introduce the overclaim.

### tools/build_deltaepszeta_training_targets.py

#### Finding 5 (MEDIUM, NEW): `_resolve_shannon_json` glob resolver picks "newest" by lex sort, not mtime

- **What**: pre-fix, `_resolve_shannon_json(pattern)` did
  `sorted(glob.glob(pattern))[-1]`, picking the lex-last entry. This
  works under the implicit "UTC-stamped" naming convention but fails
  subtly when:
  1. A file is renamed to `..._LATEST.json` (lex-late but stale, or
     lex-early-but-latest depending on sibling names).
  2. Two timestamp formats coexist (`2026-05-07T18` vs
     `20260507T180000Z`) — lex-order disagrees with mtime-order.
  3. A backup or hand-edit produces a file with a lex-late name but an
     earlier mtime.

- **Why it's a bug** (Selfcomp / Hotz): operator intent is "pick the
  newest analysis"; lex-order is a fragile proxy for "newest."

- **Fix landed**: `270f4893`. Resolver now sorts by `(mtime, lex)` so the
  most-recently-modified file wins, with lex-order as a deterministic
  tiebreaker for builds that produce multiple files in the same second.

- **Regression test** (1, in
  `src/tac/tests/test_build_deltaepszeta_training_targets.py`):
  - `test_resolve_shannon_json_mtime_beats_lex_order` — construct a
    sibling pair where the lex-late entry has the OLDER mtime; verify
    the resolver picks the lex-early-but-newer file. Pre-fix this test
    would have failed.
  - Existing `test_resolve_shannon_json_picks_newest` still passes
    (mtime agrees with lex when files are created in lex-ascending
    order).

### scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh

#### Finding 6 (MEDIUM, NEW): playbook warns but does not fail closed when lane claim is absent

- **What**: pre-fix, the script printed `WARNING: lane claim absent;
  re-claim before dispatch` but proceeded to `exec` the launcher
  anyway. Violates CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION
  non-negotiable": every dispatch must claim the lane via
  `tools/claim_lane_dispatch.py` BEFORE firing. Silent-warn-and-proceed
  is the coordination-failure pattern that burned $5-10 of duplicate
  GPU spend on 2026-05-01 (Claude H100 SXM vs codex Lightning).

- **Why it's a bug** (Quantizr / Selfcomp): real-provider dispatch with
  claim-missing-or-unknown must fail closed with a distinct exit code so
  cross-agent tools can detect the refusal. Both `claim_missing` (file
  exists but no row matches) and `claim_unknown` (file does not exist)
  are "no claim" conditions.

- **Fix landed**: `3f33c3c0`. Script changes:
  - New exit code 6 (distinct from 2/3/4/5/7/8) on real-provider
    dispatch when claim is missing OR unknown.
  - FATAL message names the exact `tools/claim_lane_dispatch.py`
    command operators should run, plus the CLAUDE.md non-negotiable
    reference.
  - Dry-run mode preserved as permissive (claim state still printed).

- **Regression tests** (6, in
  `src/tac/tests/test_deferred_dispatch_playbook_pr103_pr106.py`):
  - `test_playbook_dry_run_permissive_with_claim_missing` (exit 0)
  - `test_playbook_dry_run_permissive_with_claim_unknown` (exit 0)
  - `test_playbook_real_provider_fails_closed_when_claim_missing` (exit 6)
  - `test_playbook_real_provider_proceeds_when_claim_present` (≠ 6)
  - `test_playbook_archive_bytes_drift_exits_3` (regression)
  - `test_playbook_archive_missing_exits_2` (regression)

## Files NOT modified (8-perspective review surfaced no findings)

- `src/tac/contest_rate_distortion_system.py`: contest formula constants
  pinned to upstream (`CONTEST_SEG_WEIGHT=100`, `CONTEST_POSE_WEIGHT=10`,
  `CONTEST_RATE_WEIGHT=25`, `CONTEST_RAW_VIDEO_BYTES=37545489`).
  `clamp_min(1e-30)` on pose is sound: at exact pose=0, autograd
  gradient flows through the clamp (gradient 0 below 1e-30, which is
  the correct boundary behavior). `contest_score_decomposition` uses
  `max(float(pose), 1e-30)` consistent with the tensor-side clamp; both
  return the same `pose_term` at pose=0. Share-fractions guarded by
  `if total > 0` (defensive — `pose_term > 0` always due to clamp, so
  total is always > 0). No bug.
- `src/tac/tests/test_contest_rate_distortion_system.py`: 5 focused tests
  all pin documented behavior. No bug.

## Council positions

- **Shannon (LEAD)**: endorses all 6 fixes. Finding 4 (H₀ docstring) is
  foundational MDL discipline; the surrogate's positive bias on delta
  inputs is real and now documented honestly. Finding 5 (mtime vs
  lex-sort) is the right "follow operator intent, not the implicit
  naming convention" move. **No dissent.**
- **Yousfi**: endorses Finding 1 (Op3 substrate boundary) — the validate
  gate is decorative for the `skip_validate=True` case; the decode-side
  raise closes the bug class. Endorses Finding 6 (playbook fail-closed)
  — coordination-failure costs are forensic; the warning was a paper
  tiger.
- **Fridrich**: endorses Finding 1 — wire-format roundtrip integrity is
  the most basic forensic invariant; substrate-transform decode that
  silently emits a partial dict fails inverse-steganalysis hygiene.
- **Contrarian**: endorses ALL fixes (this round's mandate inversion
  vindicated their prior dissent on LOW#3 deferral). Notes that
  Finding 5 (mtime vs lex) was reachable in the prior round if the
  reviewer had asked "what's the failure mode of the implicit
  convention?"
- **Quantizr**: endorses Finding 6 strongly — a competitor reading the
  playbook would identify "warns but proceeds" as a coordination
  surface in seconds. Endorses Finding 5 — the
  `..._LATEST.json`/timestamp-format-coexistence failure modes are real
  in any long-lived ops directory.
- **Hotz**: endorses Finding 3 (LOW#3 LANDED at last) — training
  callbacks fire often; doubling the auto-select cost was real CPU
  waste. Endorses Finding 6 — exit codes must be distinct so cross-agent
  tools can react.
- **Selfcomp**: endorses Finding 1 — substrate-transform roundtrip must
  cover ALL input tensors or refuse cleanly. Endorses Finding 5 —
  glob-resolver hygiene is a wire-format-of-the-filesystem concern.
- **MacKay**: endorses Finding 4 (H₀ docstring) — every entropy claim
  must carry consistent caveats; finite-temperature bias is well-known
  in soft-histogram literature and now correctly noted. Endorses
  Finding 2 (Op2 5-key docstring) — keep documentation coupled to
  implementation.

## Verdict

- 6 findings → 6 fixes LANDED with regression tests. No deferrals.
- 1 commit per logical fix, all via
  `tools/subagent_commit_serializer.py`.
- 199/199 tests pass across the in-scope test surfaces.

**Recommendation**: ready for next council round. The four prior-round
deferrals are now structurally extinct, and two new MEDIUM findings
surfaced by the 8-perspective pass are also extinct. No blocking issues
require user input.

## Cross-references

- Prior bug-hunter pass: `.omx/research/bug_hunter_review_session_landings_20260507_claude.md`
- Composition contract:
  `.omx/research/four_way_stack_composition_contract_20260507_claude.md`
- Cathedral refactor: `c80ac1e4 Cathedral refactor:
  contest_rate_distortion_system canonical + minimal`
- Commits in this round (newest first):
  - `3f33c3c0` playbook: fail closed on missing lane claim (Finding 6)
  - `270f4893` build_deltaepszeta resolver: prefer mtime over lex sort (Finding 5)
  - `7e101fd8` shannon_h0 docstring: soften minimum=0 claim (Finding 4)
  - `d81709c0` op1 encode: dedup auto_select_byte_maps to one call (Finding 3)
  - `eaea6c26` op2 docstring: enumerate all 5 section_lengths keys (Finding 2)
  - `b691d465` op3 decode: refuse partial substrate (Finding 1)
- Lane registry mapping: every modified module has a corresponding
  `lane_codec_pipeline_*` lane and the cross-agent dispatch coordination
  fix touches `pr103_pr106_standalone` lane only.
