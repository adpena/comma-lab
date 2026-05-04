# PR85 STBM1BR + PR92 RMB1 Exact-Eval Readiness Review - 2026-05-04

Reviewer: codex

Scope: independent local readiness review of the `PR85_STBM1BR + PR92 RMB1
randmulti` path. No remote GPU job, scorer load, training, exact eval, or
dispatch claim was executed.

## Candidate Under Review

- builder: `experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py`
- archive:
  `experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/archive.zip`
- manifest:
  `experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/manifest.json`
- preflight:
  `experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/preflight.json`
- archive bytes/SHA-256:
  `229480`,
  `f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774`
- source STBM bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- byte delta versus STBM: `-276`
- evidence grade: `empirical_local_lossless_recode_candidate`

## Readiness Findings

- Runtime RMB1 support: `submissions/robust_current/apply_qzs3_postprocess.py`
  contains `_decode_rmb1_randmulti_payload()` and routes `RMB1` through
  `_decode_randmulti()`.
- Reusable helper surface: `src/tac/pr85_bundle.py` now exposes
  `compare_pr85_randmulti_decoded_rows()` for decoded-row parity reports.
- Candidate archive is single-member `x`, ZIP_STORED, fixed timestamp
  `1980-01-01 00:00:00`, deterministic rewrite identical.
- Candidate changes only `randmulti` versus the STBM source. The STBM mask
  bytes are preserved:
  `1b1ec60b64e284aae11e838dc3d9996bce00125df5712a8ba9c3e8f739c9d313`.
- PR92 public archive member `a` / `RSB1` side-info is not carried into this
  candidate. The reviewed candidate is a pure in-bundle `RMB1` randmulti recode
  with no side-info members.
- Decoded randmulti rows match PR85/STBM:
  `87bcc720c1e80afb9adad5ee01477423ced526f31c54d461d69dbf26e08eecc9`.
- Local preflight status: `passed`; `ready_for_exact_eval_after_lane_claim=true`.
- Score claim: false. Dispatch performed: false.

## Dispatch Claim Template

Do not dispatch without first replacing placeholders and running the Level-2
claim command:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --claims-path .omx/state/active_lane_dispatch_claims.md \
  --lane-id pr85_stbm1br_pr92_rmb1_randmulti \
  --platform lightning \
  --instance-job-id exact_eval_pr85_stbm1br_plus_pr92_rmb1_randmulti_t4_${UTC_STAMP} \
  --agent ${AGENT_ID} \
  --predicted-eta-utc ${PREDICTED_ETA_UTC} \
  --status exact_eval_ready \
  --notes "archive_sha256=f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774 manifest=experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/manifest.json"
```

This command was not executed by this review.

## Verification

```bash
.venv/bin/python -m py_compile \
  experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  src/tac/pr85_bundle.py \
  src/tac/tests/test_pr85_bundle.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  src/tac/tests/test_pr85_bundle.py::test_current_pr92_rmb1_randmulti_is_decoded_row_parity_recode \
  src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py::test_rmb1_randmulti_decodes_to_headerless_sparse_rows \
  -q

.venv/bin/python experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py

.venv/bin/python experiments/profile_endgame_archive_decision.py \
  --candidate STBM1BR=experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip \
  --candidate PR92=experiments/results/public_pr92_intake_20260504_codex/archive.zip \
  --candidate STBM1BR_RMB1=experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/archive.zip \
  --frontier-label STBM1BR \
  --json-out /tmp/pr85_stbm1br_rmb1_endgame_profile.json \
  --markdown-out /tmp/pr85_stbm1br_rmb1_endgame_profile.md
```

Focused pytest result: `4 passed in 0.45s`.

Decision profile summary:

- `STBM1BR_RMB1`: strict ZIP true, decision-valid true, side bytes `0`.
- `STBM1BR_RMB1` versus `STBM1BR`: archive delta `-276`, changed segments
  only `randmulti`, runtime blockers `[]`.

## Verdict

Greenlight for exact CUDA auth-eval dispatch preparation after a fresh Level-2
lane claim. No exact-eval score evidence exists from this review, and no
dispatch was performed.
