# PR107 Post-Submission Hardening And PR100 Pending Replay Addendum - 2026-05-04

Scope: post-submission checklist and custody addendum only. No GPU job was
dispatched in this pass, and no exact artifacts or `.omx/state` files were
edited.

## PR107 Submitted Surface

- PR: <https://github.com/commaai/comma_video_compression_challenge/pull/107>
- title at audit time: `apogee submission (0.2293)`
- state at audit time: `OPEN`
- author: `adpena`
- head branch: `adpena:apogee-pr98-hnerv-adapter`
- head SHA at audit time:
  `20c411b58c9270dfb6ce19742e657fefd1f54829`
- release tag:
  <https://github.com/adpena/comma_video_compression_challenge/releases/tag/apogee-pr98-hnerv-adapter-20260504>
- release asset:
  <https://github.com/adpena/comma_video_compression_challenge/releases/download/apogee-pr98-hnerv-adapter-20260504/archive.zip>

## Release Asset Custody

GitHub release metadata for `apogee-pr98-hnerv-adapter-20260504` reports one
uploaded asset:

- name: `archive.zip`
- content type: `application/zip`
- size: `178392` bytes
- digest:
  `sha256:7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- release asset id: `411795115`
- created/updated: `2026-05-04T12:10:00Z`

The PR body's release URL, archive bytes, archive SHA-256, member name, member
SHA-256, sample count, T4 hardware claim, and runtime tree SHA-256 match the
local strict gate packet recorded below.

## Exact Local Custody Referenced By PR107

- score-bearing archive:
  `experiments/results/submission_packet_pr98_adapter_20260504/apogee_pr98_hnerv_adapter/archive.zip`
- archive bytes: `178392`
- archive SHA-256:
  `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- archive member: `0.bin`
- member bytes: `178284`
- member SHA-256:
  `fce200db2fe087cc6a051945b3fda2c37f5bbb3e19b8f20a1aea7201db0c9f5f`
- exact CUDA/T4 artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr98_hnerv_adapter_t4_20260504T0958Z/contest_auth_eval.adjudicated.json`
- recomputed score: `0.22933111465960354`
- samples: `600`
- runtime tree SHA-256:
  `0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0`

## Strict Pre-Submission Gate

Gate artifact:
`experiments/results/submission_packet_pr98_adapter_20260504/pre_submission_compliance.json`

- status: `passed`
- failed checks: `[]`
- warning checks: `[]`
- check count: `78`
- archive SHA/bytes matched the strict packet, GitHub release asset digest, and
  PR body.
- auth-eval linkage matched the exact CUDA/T4 artifact, `600` samples,
  `device=cuda`, `gpu_t4_match=true`, and the runtime tree SHA-256 above.

## PR Body Audit

The live PR body includes:

- submission name `apogee`
- release asset URL above
- CUDA report block with archive bytes `178392`
- exact local custody fields:
  `score_recomputed_from_components=0.22933111465960354`,
  archive SHA-256
  `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`,
  member `0.bin`, member SHA-256
  `fce200db2fe087cc6a051945b3fda2c37f5bbb3e19b8f20a1aea7201db0c9f5f`,
  `n_samples=600`, `eval_hardware=Tesla T4`, and runtime tree SHA-256
  `0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0`.
- statement that the canonical path was
  `archive.zip -> inflate.sh -> upstream/evaluate.py`.

Audit status: no mismatch found between the PR body, release asset metadata,
strict pre-submission gate, and exact local custody fields reviewed in this
pass.

## PR100 Pending Replay

Source ledger:
`.omx/research/public_pr100_hnerv_lc_v2_replay_20260504_codex.md`

- PR: <https://github.com/commaai/comma_video_compression_challenge/pull/100>
- title: `hnerv_lc_v2 submission (0.1954)`
- author: `BradyMeighan`
- public release asset:
  <https://github.com/BradyMeighan/comma_video_compression_challenge/releases/download/hnerv-lc-v2-archive/archive.zip>
- local intake archive:
  `experiments/results/public_pr100_intake_20260504_codex/archive.zip`
- archive bytes: `178981`
- archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- adapter runtime:
  `experiments/results/public_runtime_adapters_20260504_codex/pr100_runtime_adapter`
- primary replay lane:
  `public_pr100_hnerv_lc_v2_t4_adapter_replay`
- primary replay job:
  `exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z`
- latest checked state in this pass: no local PR100 exact-eval artifact
  directory was present; the replay remains pending/no-score evidence.
- duplicate hedge row observed in dispatch state:
  `exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_dup_20260504T1218Z`
  failed before Lightning job creation with `failed_presubmit_machine_alias`.
  This does not close the primary pending replay.

Do not dispatch or relaunch PR100 from this addendum. Wait for the already
claimed primary replay to produce a terminal artifact or be explicitly closed.

## If PR100 Validates

Only treat PR100 as superseding PR107 after all of these are true:

- a valid
  `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json`
  exists, or an explicitly superseding terminal replay artifact is recorded;
- the artifact reports exact CUDA/T4 or contest-equivalent hardware, `600`
  samples, canonical `archive.zip -> inflate.sh -> upstream/evaluate.py`, and
  archive SHA/bytes
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641` /
  `178981`;
- recomputed score from components agrees with the adjudicated JSON;
- component gates pass and no runtime-custody mismatch exists;
- a strict pre-submission style gate is run against the PR100 replay archive
  and adapter runtime tree with `status=passed`, `failed_checks=[]`, and no
  public-release hygiene violations.

If those gates pass and the score is below PR107's `0.22933111465960354`, then:

1. record a new dated ledger supersession note with PR100's exact score,
   components, archive SHA/bytes, runtime tree SHA-256, gate JSON path, and
   terminal dispatch row;
2. update public-facing Apogee/PR107 language to say PR107 remains valid exact
   custody but is no longer the frontier once PR100's exact replay validates;
3. do not reuse PR100 bytes or runtime as an Apogee-owned submission unless
   licensing, authorship, and contest rules explicitly permit it;
4. keep PR107 release asset custody immutable and distinguish it from any
   later replacement or withdrawal action.

Evidence grade for this addendum: custody/checklist. It records current
submission and pending-replay control decisions; it is not a new score claim.

## Verification

- `gh pr view 107 --repo commaai/comma_video_compression_challenge --json ...`
- `gh release view apogee-pr98-hnerv-adapter-20260504 --repo adpena/comma_video_compression_challenge --json ...`
- `jq` inspection of
  `experiments/results/submission_packet_pr98_adapter_20260504/pre_submission_compliance.json`
- local file presence check for PR100 exact-eval artifact directory

No dispatch commands were run.
