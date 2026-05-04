# Grand Council Lane C - PR91/HPM1 And PR85_STBM1BR/QFQ4 Unblock Decision

Recorded: 2026-05-04
Reviewer: codex
Scope: read-mostly adversarial review and small dispatch-readiness guard.
No remote dispatch, lane claim, scorer load, CUDA eval, archive promotion, or
score claim was performed.

## Source Material Inspected

- `.omx/research/recursive_adversarial_greenup_review_20260504_codex.md`
- `.omx/research/pr91_hpm1_parity_greenup_20260504_worker.md`
- `.omx/research/pr91_hpm1_parity_first_principles_20260504_codex.md`
- `.omx/research/pr91_hpm1_residual_lowering_and_submission_gate_20260504_codex.md`
- `.omx/research/pr85_stbm1br_model_recode_feasibility_20260504_codex.md`
- `.omx/research/pr85_qfq4_model_serializer_probe_worker_20260504.md`
- `.omx/research/pr85_nonmask_self_compression_audit_worker_20260504.md`
- `.omx/research/pr85_stbm1br_rmb1_exact_eval_readiness_review_20260504_codex.md`
- `.omx/research/contest_faithful_swarm_execution_20260502_codex.md`
- `src/tac/pr91_hpm1_codec.py`
- `src/tac/tests/test_pr91_hpm1_codec.py`
- `experiments/analyze_or_build_pr85_qfq4_model_serializer_candidate.py`
- `src/tac/tests/test_pr85_qfq4_model_serializer_probe.py`
- `experiments/profile_pr85_stbm1br_model_recode_feasibility.py`
- `src/tac/tests/test_profile_pr85_stbm1br_model_recode_feasibility.py`
- `experiments/preflight_candidate_manifest_dispatch_readiness.py`
- `src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py`

Literal `Hegel`/`Hypatia` scan result: only one `Hegel the 2nd` reference was
found, and it concerns an older C067 delta/overlay mask-topology worker, not
PR91/HPM1 or PR85_STBM1BR/QFQ4. No literal `Hypatia` finding was found in the
current searched surfaces. The current blocker evidence is therefore taken
from the PR91/PR85 worker and codex ledgers above.

## Decision Summary

PR91/HPM1 has the higher EV if local parity succeeds. The byte-faithful
PR85_STBM1BR to PR91 fusion is mask-only and saves `7352` archive bytes versus
the current STBM frontier, for a rate-only projection of about `-0.004895395`
score. That projection is prediction evidence only and cannot dispatch while
HPM1 entropy decode fails.

PR85_STBM1BR/QFQ4 model recode has lower upside and is currently blocked twice:
the best QFQ4-style model screen is formula-only `-659` model/archive bytes
and the decoded tensor parity fails for
`frame1_head.block1.film_proj.weight` (`4726 / 5376` changed elements,
max abs diff `6.103515625e-05`). The submitted replay/runtime surfaces also
lack QFQ4 model loader support, so no archive should be built from this screen.

## PR91/HPM1 Gates Before Any Archive Build

- Static custody must match the PR91 single-member `x` archive, HPM1 mask
  segment, token stream, and HPAC model hashes.
- Full HPM1 decode must pass for all 600 frames under the submitted runtime
  entropy/probability contract, not a prefix or alternate off-contract variant.
- Decode output must prove byte or reviewed semantic parity against the
  PR85_STBM mask tensor before using the `7352` byte saving.
- Full decode plus byte-exact re-encode, or an equivalent reviewed per-symbol
  entropy trace, must prove the submitted token stream contract.
- No uncharged fallback to STBM/QMA9 is allowed after HPM1 entropy failure.
  If fallback exists, the fallback payload bytes must be charged in the archive.
- The exact submitted `archive.zip -> inflate.sh -> upstream/evaluate.py`
  runtime path must load HPM1 without sidecars and record runtime tree SHA.
- A Level-2 dispatch claim is required before any remote exact eval.

Current fail-closed evidence: source-contract decode fails at frame `0`,
group `10`, symbol `191` after `5951` decoded symbols. All tested
probability variants and byte/word stream transforms fail closed. The reference
prefix probe does not prove PR85 mask identity.

## PR85_STBM1BR/QFQ4 Gates Before Any Archive Build

- A candidate archive must be emitted; formula-only byte deltas are not enough.
- The archive must change only the PR85 `model` segment; mask, pose, post,
  shift, frac, frac2, frac3, bias, region, and randmulti bytes must stay exact.
- Decoded model tensor parity must be bit-exact for every tensor.
- The submitted PR85/STBM replay runtime must include a no-sidecar QFQ4 model
  loader and single-`x` path support.
- Local runtime output parity must pass against the source archive before any
  exact eval claim.
- Exact runtime tree custody and Level-2 dispatch claim are still mandatory.

Current fail-closed evidence: QH0/QM0 passthrough has no byte win; QFQ4 has a
formula-only byte win but fails tensor parity and runtime loader support. No
candidate archive was built.

## Failure Classes Guarded

- Formula-only byte or score deltas being treated as dispatch evidence.
- Runtime-changing formats without exact submitted runtime contract.
- QFQ4 model recode with false runtime loader support.
- HPM1/HPAC readiness with false full decode or byte-parity fields.
- Nested failed/blocked readiness statuses hidden below builder-specific JSON
  schemas.
- PR91 score projections that assume component identity without decoded-mask
  parity.
- HPM1 fallback semantics that would consume uncharged STBM/QMA9 bytes.
- No-op or source-passthrough model recodes promoted as byte wins.

## Guard Greenup

Hardened `experiments/preflight_candidate_manifest_dispatch_readiness.py`:

- Added `qfq4` to runtime-changing marker detection.
- Added recursive fail-closed checks for formula-only evidence fields.
- Added recursive fail-closed checks for false runtime loader support fields.
- Added recursive fail-closed checks for false decode/exact-parity fields and
  nested failed/blocked status fields.

Focused tests added in
`src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py`:

- QFQ4 model recode readiness is blocked when it is formula-only, lacks runtime
  loader support, and has false decoded tensor parity.
- HPM1 projection readiness is blocked when decode parity failed and the only
  score movement is formula-only/prediction evidence.

Real-artifact preflight probes after the guard:

- `pr85_stbm1br_model_recode_feasibility_20260504_codex/candidate_summary.json`
  exits `2`, blocked on formula-only QFQ4/QH0 context, false QFQ4 parity, false
  QFQ4 runtime support, and missing exact runtime contract.
- `pr85_qfq4_model_serializer_probe_20260504_worker/candidate_summary.json`
  exits `2`, blocked on `dispatch_unlocked=false`, formula-only deltas, false
  tensor parity, false runtime support, and missing exact runtime contract.
- `pr91_hpm1_pr85_stbm_fusion_plan_20260504_codex.json` exits `2`, blocked on
  HPM1 replay failure, formula-only rate projection, and missing exact runtime
  contract.
- `pr91_hpm1_probability_variant_matrix_frame0_20260504_codex.json` exits `2`,
  blocked on false byte parity, failed variants, and missing exact runtime
  contract.

## Verification

```text
.venv/bin/python -m py_compile experiments/preflight_candidate_manifest_dispatch_readiness.py src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py
.venv/bin/python -m pytest src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py -q
# 8 passed in 0.08s

.venv/bin/python -m pytest src/tac/tests/test_pr85_qfq4_model_serializer_probe.py -q
# 2 passed in 0.68s

.venv/bin/python -m pytest src/tac/tests/test_profile_pr85_stbm1br_model_recode_feasibility.py -q
# 2 passed in 1.49s

.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py -q
# 20 passed in 116.92s
```

## Next Implementation Recommendation

Prioritize PR91/HPM1 only if the next slice obtains the missing entropy
contract evidence: original encoder/build recipe, a per-symbol trace around
frame `0` group `10` symbol `191`, or a native queue/probability trace that
reproduces the submitted token stream. Do not build an archive from HPM1 until
full decode and parity pass.

Keep PR85_STBM1BR/QFQ4 as a lower-upside disjoint lane. The next useful slice is
not archive building; it is either a bit-exact serializer for the special FiLM
row tensor or a no-sidecar QFQ4 loader plus runtime output parity harness. If
either parity or loader support remains false, the lane stays planning-only.
