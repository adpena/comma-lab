# Four-Hour Floor Push Stack Opportunity Review - 2026-05-04

Reviewer: codex

Scope: local systems/math adversarial review for the next four-hour floor push.
No remote dispatch, lane claim, scorer load, training, or exact eval was run in
this pass. Exact score truth remains `archive.zip -> inflate.sh ->
upstream/evaluate.py` through CUDA auth eval.

## Current Anchor

- Frontier archive: `PR85_STBM1BR`
- Exact evidence: `A++` T4 CUDA
- Exact artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.adjudicated.json`
- Score: `0.25369011029397787`
- Archive bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Components: SegNet `0.00057185`, PoseNet `0.0001894`, samples `600`
- Runtime tree SHA-256:
  `d195f4ecd0743cfd146efafee6729e96ee5428bfb28bbd0ca87cbad055494440`

## New Local Decision Artifact

Reran the byte-level endgame decision tool against `PR85`, `PR85_STBM1BR`,
`STBM1BR_RMB1`, `PR91_HPM1`, and `PR92_RSB1` with `PR85_STBM1BR` as the
frontier:

- JSON:
  `experiments/results/endgame_archive_decision_20260504_codex_four_hour_review/endgame_archive_decision_profile.json`
- Markdown:
  `experiments/results/endgame_archive_decision_20260504_codex_four_hour_review/endgame_archive_decision_profile.md`

This artifact is byte-level decision support only. It does not claim score.

Key ranked byte actions versus `PR85_STBM1BR`:

| Candidate | Surface | Est. bytes | Est. rate-score delta | Dispatch advice |
| --- | --- | ---: | ---: | --- |
| `PR91_HPM1` | mask | `-7352` | `-0.004895395023354203` | byte-positive but requires mask runtime/parity gates |
| `STBM1BR_RMB1` | randmulti | `-276` | `-0.0001837770710617193` | byte-positive but requires exact CUDA gate |
| `PR92_RSB1` | randmulti + side-info | `+188` | `+0.00012518148318696823` | do not dispatch rate-only |

## Top Five Four-Hour Opportunities

### 1. `STBM1BR_RMB1` exact-eval preparation

Expected movement: `-276` charged bytes, rate-only score delta
`-0.0001837770710617193` if exact components remain unchanged.

Why rank first by wall-clock: the candidate is already byte-closed,
deterministic, strict ZIP, single-member `x`, and local pre-submission
compliance passed. It is the only current positive stack candidate with no
known local implementation blocker.

Exact blockers:

- No exact CUDA result exists for
  `experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/archive.zip`.
- A Level-2 dispatch claim is mandatory before any remote/T4 eval.
- Score cannot be claimed until adjudicated CUDA JSON exists and the score is
  recomputed from components.

Concrete steps:

1. Recheck candidate archive SHA/bytes:
   `f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774`,
   `229480`.
2. Claim lane `pr85_stbm1br_pr92_rmb1_randmulti`.
3. Run canonical T4 exact eval using the fixed STBM runtime.
4. If components match the STBM anchor, promote as pure-rate frontier; if not,
   preserve component deltas as a narrow randmulti recode negative.

### 2. `PR91_HPM1` contract recovery, then `HPM1+RMB1`

Expected movement: `PR91_HPM1` alone is `-7352` bytes versus STBM
(`-0.004895395023354203` rate-score). If combined with `RMB1` after HPM1 is
recovered, the rate-only stack target is `-7628` bytes
(`-0.005079172094415922`).

Why not first: the upside is largest, but local PR91/HPM1 replay fails before
score and before full frame-0 decode.

Exact blockers:

- `HPM1` entropy/probability contract mismatch:
  `AssertionError: Tried to decode from compressed data that is invalid for the employed entropy model.`
- PR91 reuses the PR86 HPAC model but has a distinct token stream; no tested
  float/perfect/byte-order variant decodes frame 0.
- The PR91 prefix probe mismatches the PR85 token reference before the entropy
  failure, so PR85-mask identity is not locally proven.

Concrete steps:

1. Recover the actual PR86/PR91 HPAC token-generation trace or a byte-exact
   decode/reencode contract.
2. Prove full submitted-token decode and byte-exact reencode for `600x384x512`.
3. Build a deterministic STBM-family archive using `HPM1` mask semantics only
   after runtime parity is proven.
4. Add `RMB1` after HPM1 exact replay is valid; do not dispatch a derivative
   HPM1 stack before base HPM1 is replayable.

### 3. PR85/JFG model self-compression via QFQ4-style serializer

Expected movement: best local byte screen was `-659` model-segment bytes,
rate-only `-0.0004388010501075109`, if tensor/runtime output parity can be
made exact.

Why rank third: it is smaller than HPM1 but directly targets the PR85-family
JointFrameGenerator model payload and is independent of mask semantics.

Exact blockers:

- Tensor parity fails for at least `frame1_head.block1.film_proj.weight`.
- Current PR85 replay/robust runtime lacks the needed QFQ4 model-loader path
  for this archive family.
- QH0/QM0 exact serializer screens found no real byte win.

Concrete steps:

1. Decide whether the fp16 row reconstruction error can be made bit-exact, not
   just numerically close.
2. If bit-exact, add a narrow PR85-family runtime loader and output-parity
   preflight.
3. Build a deterministic archive only after tensor parity and runtime parity
   are both true.
4. Exact CUDA eval only after lane claim and archive custody.

### 4. PR92 `RSB1`/QRGB action stream as component-benefit research

Expected movement: rate-only `PR92_RSB1` transplant is not positive on STBM:
`RMB1` saves `276` bytes, but `RSB1` plus ZIP overhead costs `464` bytes, net
`+188` bytes. It needs at least `0.00012518148318696823` component-score gain
to break even.

Why rank fourth: it is close to the current JFG action surface and PR92 proves
the public direction, but existing PR85 QRGB singletons are exact negatives.

Exact blockers:

- Existing PR85 QRGB singleton T4 evals regress; prepared combo archives
  should stay blocked.
- `RSB1` is rate-negative on STBM without component benefit.
- No PR92 action-component response curve exists under our runtime.

Concrete steps:

1. Treat `RSB1` as an action vocabulary/proposal source, not a dispatch
   candidate.
2. Build a local action decoder/profile that maps `RSB1` actions to exact
   JFG/qpost surfaces and labels which outputs change.
3. Use exact-negative QRGB results as signed training labels.
4. Only build a score candidate when component-response evidence clears the
   `+188` byte break-even and archive byte closure is deterministic.

### 5. Yousfi-Fridrich field equations, multimask/multiresolution, ego/foveation

Expected movement: potentially large only if the atom field can replace or
repair mask semantics without PoseNet/SegNet collapse. Current direct CMG/
multimask routes are not dispatch candidates.

Why rank fifth: these tools are valuable for the next proposal distribution,
but they are below the four-hour exact-score path because existing mask grammar
evidence includes severe component collapse or rate-negative screens.

Exact blockers:

- Prior CMG3/CMG3A exact candidates are far outside the PR85/STBM basin.
- Multimask/multiresolution plans remain planning-only unless a concrete
  archive builder consumes all charged atoms and preserves runtime closure.
- Ego/foveation fields are proposal priors until their parameters are charged
  and consumed by inflate runtime.

Concrete steps:

1. Use field-equation and residual-density outputs only to rank HPM1/STBM mask
   atoms, not to dispatch raw CMG/row-run candidates.
2. Require decoded-mask parity for lossless variants or explicit component
   response for lossy variants.
3. Require archive manifests to record atom bytes, selector provenance,
   residual SHA, runtime consumption, and non-noop status.
4. Dispatch only a byte-closed archive with a geometry-escape proof and a fresh
   Level-2 claim.

## Commands Run

```bash
sed -n '1,1700p' AGENTS.md

.venv/bin/python experiments/profile_endgame_archive_decision.py \
  --candidate PR85=experiments/results/public_pr85_intake_20260503_codex/archive.zip \
  --candidate PR85_STBM1BR=experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip \
  --candidate STBM1BR_RMB1=experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/archive.zip \
  --candidate PR91_HPM1=experiments/results/public_pr91_intake_20260504_codex/archive.zip \
  --candidate PR92_RSB1=experiments/results/public_pr92_intake_20260504_codex/archive.zip \
  --frontier-label PR85_STBM1BR \
  --json-out experiments/results/endgame_archive_decision_20260504_codex_four_hour_review/endgame_archive_decision_profile.json \
  --markdown-out experiments/results/endgame_archive_decision_20260504_codex_four_hour_review/endgame_archive_decision_profile.md

.venv/bin/python -m pytest src/tac/tests/test_endgame_archive_decision.py -q
.venv/bin/python -m py_compile experiments/profile_endgame_archive_decision.py src/tac/endgame_archive_decision.py

jq '{status, archive, failed_checks, warning_checks, score_claim, public_hygiene}' \
  experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/pre_submission_compliance.json
```

Verification result: focused endgame tests `3 passed in 0.09s`; py_compile
passed. One exploratory `jq` filter against `pre_submission_compliance.json`
was malformed and failed locally; the corrected `jq` query above passed and
showed compliance `status="passed"` with no failed or warning checks.

## Changed Paths

- Added:
  `.omx/research/four_hour_floor_push_stack_opportunity_review_20260504_codex.md`
- Added:
  `experiments/results/endgame_archive_decision_20260504_codex_four_hour_review/endgame_archive_decision_profile.json`
- Added:
  `experiments/results/endgame_archive_decision_20260504_codex_four_hour_review/endgame_archive_decision_profile.md`

No source code was edited in this pass.
