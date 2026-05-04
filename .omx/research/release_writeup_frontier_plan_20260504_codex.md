# Release Writeup Frontier Plan - 2026-05-04 Codex

Scope: docs/report/site plan only. No scripts, runtime code, state ledgers,
experiment builders, GPU jobs, or score claims were changed.

## Current Frontier For Public Writeup

The current exact internal frontier is PR85+STBM1BR:

- evidence grade: `A++`
- score: `0.25369011029397787`
- archive bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- SegNet/PoseNet: `0.00057185` / `0.0001894`
- samples: `600`
- hardware: Tesla T4
- score artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.adjudicated.json`
- runtime tree SHA-256:
  `d195f4ecd0743cfd146efafee6729e96ee5428bfb28bbd0ca87cbad055494440`

Interpretation: PR85+STBM is a pure charged-rate improvement over PR85. It
preserves PR85 component distances at JSON precision and saves `6572` archive
bytes by replacing PR85's `QMA9` mask segment with the recovered lossless
`STBM1BR\0` representation.

## External And Fail-Closed Boundary

PR91/HPM1 self-reports `0.24879480490416128` at `222404` bytes, but local T4
and L40S replay failed before score in HPM1 entropy decode. The PR91 public
score is therefore external source context only.

PR91 source anatomy to carry in the report:

- single stored member `x`, `222304` bytes
- HPM1 mask segment `145087` bytes
- HPM1 token stream `116796` bytes
- HPAC model `28243` bytes
- model/pose/post/shift/frac/bias/region/randmulti side channels follow the
  PR85 v5 micro-bundle layout

Important runtime correction: PR91 `range_mask_codec.cpp` is live only for the
`QMA6`/`QMA7`/`QMA8`/`QMA9` branch. It is not the HPM1 decoder and it does not
fallback-rescue HPM1 entropy failures.

## Release Gate

The pre-submission compliance gate is now part of the writeup/release plan:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir <packet-dir> \
  --auth-eval-json <packet-dir>/contest_auth_eval.adjudicated.json \
  --require-auth-eval \
  --require-t4-equivalent \
  --expect-single-member x \
  --expected-archive-sha256 c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6 \
  --expected-archive-size-bytes 229756 \
  --expected-runtime-tree-sha256 d195f4ecd0743cfd146efafee6729e96ee5428bfb28bbd0ca87cbad055494440 \
  --public-pr-refs 85,90
```

The gate is provider-agnostic and non-scoring. A pass means the exact packet is
structurally ready for submission review; it does not create a new score.

## Edited Surfaces

- `reports/writeup_working.md`
- `docs/paper/04_results.md`
- `docs/submission_template.md`
- `reports/graphs/final_writeup_draft.md`
- `reports/graphs/final_submission_notes.md`
- `reports/graphs/release_checklist.md`
- `reports/graphs/evidence_index.md`

## Residual Gaps

- Build the final public packet directory and run the compliance gate against
  that exact directory.
- Finish PR91/HPM1 full decode/reencode parity before any PR91-derived score
  wording or dispatch.
- Run strict public release hygiene on the exact PR body, notebook, and site
  bundle before publishing URLs.
- Regenerate any derived static HTML/site copies only after the source Markdown
  is accepted, so generated output does not drift from the plan.
