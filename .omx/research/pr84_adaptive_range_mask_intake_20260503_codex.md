# PR84 Adaptive Range Mask Intake - Codex

Timestamp: 2026-05-03T21:50:49Z

Evidence grade: A++ contest T4 after exact CUDA replay harvest.

## Objective

Preserve the PR84 public-submission intake, byte anatomy, and dispatch state
without turning public-reported or local-forensic facts into score claims.

## Current Exact Frontier Before PR84 Replay

- Internal A++ frontier: PR81/QMA9 replay.
- Score: `0.2812078926981712`.
- Archive bytes: `215960`.
- Archive SHA-256:
  `cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc`.
- Artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr81_qzs3_range_mask_t4_depsfix_20260503T195657Z/contest_auth_eval.adjudicated.json`.

## PR84 Intake

- Public PR: `https://github.com/commaai/comma_video_compression_challenge/pull/84`.
- Public-reported score: `0.2751402303839512`.
- Local downloaded archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr84/archive.zip`.
- Archive bytes: `215735`.
- Archive SHA-256:
  `a607a6c3ae9b610e6edfb546c3206004ae40fc348ecaef2446b7134a19b8e07f`.
- ZIP anatomy: one stored member `p`, payload bytes `215635`, ZIP overhead
  `100`.
- Public runtime source mirror:
  `experiments/results/top_submission_reverse_engineering_20260503_pr84/sources/`.

## No-Router Correction

Read-only deconstruction found PR84 and PR81 share the same QMA9 range-mask
stream:

- `range_mask.qma9` bytes: `159011`.
- `range_mask.qma9` SHA-256:
  `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`.

The PR84 archive-byte improvement over PR81 is therefore primarily a no-router
layout change, not a smaller QMA9 mask stream. The rate-only value of removing
`225` bytes is:

```text
225 * 25 / 37,545,489 = 0.00014981817160066974
```

This correction changes the next engineering priority: native no-router payload
support and finite QMA9 mode search are higher EV than assuming PR84 has a new
mask entropy breakthrough.

## PR84 Static Profile

Profile artifact:
`experiments/results/top_submission_reverse_engineering_20260503_pr84/pr84_qma9_semantic_range_mask_profile.json`.

Extracted segments:

- `range_mask.qma9`: offset `0`, bytes `159011`.
- `split_model_reordered.br_bundle`: offset `159011`, bytes `55725`.
- `optimized_poses.qp1.br`: offset `214736`, bytes `899`.

The PR84 no-router packed payload layout is:

```text
range_mask.qma9 | split_model_reordered.br_bundle | optimized_poses.qp1.br
```

This differs from PR81-style router payloads and must be represented explicitly
in manifests and robust runtime support.

## Running Exact CUDA Replay

- Job: `exact_eval_public_pr84_adaptive_range_mask_no_router_t4_20260503T214008Z`.
- Lane claim: `public_pr84_adaptive_range_mask_no_router_t4_replay`.
- State file:
  `.omx/state/public_pr84_adaptive_range_mask_no_router_t4_20260503T214008Z_batch_jobs.json`.
- Expected archive SHA-256:
  `a607a6c3ae9b610e6edfb546c3206004ae40fc348ecaef2446b7134a19b8e07f`.
- Expected archive bytes: `215735`.
- Status at 2026-05-03T21:49Z: Running on T4.
- Score claim: false until `contest_auth_eval.adjudicated.json` is harvested
  and recomputed.

## Exact CUDA Replay Harvest

Harvested at 2026-05-03T21:58Z through the state-derived Lightning SSH path.

- Artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr84_adaptive_range_mask_no_router_t4_20260503T214008Z/contest_auth_eval.adjudicated.json`.
- Adjudication provenance:
  `experiments/results/lightning_batch/exact_eval_public_pr84_adaptive_range_mask_no_router_t4_20260503T214008Z/adjudication_provenance.json`.
- Score: `0.2751401491321396`.
- Archive bytes: `215735`.
- Archive SHA-256:
  `a607a6c3ae9b610e6edfb546c3206004ae40fc348ecaef2446b7134a19b8e07f`.
- PoseNet: `0.00049341`.
- SegNet: `0.00061248`.
- Samples: `600`.
- Hardware: Tesla T4 CUDA.
- Promotion eligible: `true`.
- Delta versus PR81 exact frontier: `-0.006067743566031625`.

This promotes PR84 over PR81 as the current internal exact frontier. The
public-reported PR84 score differs only by rounding/precision from the
canonical contest JSON recomputation.

## Concurrent Stack Probes

Two PR81+PR82 all-72 QRM1 exact-eval probes were running or queued behind the
same custody standard:

- `exact_eval_pr81_pr82_controls_qrm1_all072_g4dn_t4_20260503T214202Z`.
- `exact_eval_pr81_pr82_qrm1_all072_randmulti_g4dn_t4_20260503T214202Z`.

At 2026-05-03T22:04Z both resolved to provider `Failed` with empty remote
artifact directories. They remain infrastructure negatives only: no
`contest_auth_eval.json`, no component evidence, and no method conclusion.

The valid PR84-native interaction tests queued after PR84 promotion are:

- `exact_eval_pr84_pr82_controls_all600_g4dn_t4_20260503T2205Z`, archive
  bytes `218873`, SHA
  `e8b4c250c0efc1250ef5a93ec3d875d950b00c457b8f55af9bc66a5532035b79`.
- `exact_eval_pr84_pr82_qrm1_all072_randmulti_g4dn_t4_20260503T2205Z`,
  archive bytes `232832`, SHA
  `25f37334a4460c5996882a4abf9e6860bd7500c1781db6db578079ccb494f421`.
- `exact_eval_pr84_pr82_controls_qrm1_all072_g4dn_t4_20260503T2205Z`,
  archive bytes `235363`, SHA
  `39d6b401fa4ee894fa7b305535dc1bf7d61ccb0923867f3336e9a355f80126c4`.

All three are `score_claim=false` until exact CUDA JSON is harvested. The
small controls-only sidecar has the best planning score if PR82 components
carry; the all-72 variants test richer interaction structure.

### Superseded Fixrestore Hedges

The earlier fixrestore hedge jobs resolved to `Failed` after a temporary
Pending/status-reconciliation anomaly:

- `exact_eval_pr81_pr82_controls_all600_g4dn_t4_fixrestore_20260503T211855Z`.
- `exact_eval_pr81_pr82_nm2_randmulti_g4dn_t4_fixrestore_20260503T211855Z`.
- `exact_eval_pr81_pr82_controls_qrm1_supported_g4dn_t4_fixrestore_20260503T211855Z`.

Remote artifact directories were empty at classification time. These failures
are infrastructure/dispatch negatives for the stale hedge wave, not scientific
evidence against the PR81+PR82 stack family. The corrected all-72 QRM1 reruns
are the valid interaction tests.

## Hardening Landed

The Lightning exact-eval submitter now distinguishes robust-current runtime
from external public replay runtime when deciding whether `config.env` is a
required staged file. This prevents public replay jobs from being blocked by a
contest-valid source tree that does not ship a config file, while preserving
the stricter robust-current requirement.

Focused verification:

```text
src/tac/tests/test_public_replay_exact_eval_hardening.py - 3 passed
src/tac/tests/test_profile_pr81_qma9_range_mask_contract.py - 5 passed
```

## Next Actions

1. Harvest PR84 T4 replay immediately when JSON lands.
2. If PR84 confirms, promote it over PR81 and close the dispatch claim with a
   terminal row.
3. Add native no-router support for our robust unpack/build path only after
   Pascal's PR84+PR82 builder lane returns or if that lane explicitly leaves
   the runtime support untouched.
4. Run deterministic local QMA9 finite byte-search. A candidate can dispatch
   only if it verifies full raw-mask parity, records charged bytes/SHA/mode,
   and beats the existing QMA9 stream locally.
5. Treat PR81+PR82 QRM1 stack probes as exact interaction tests, not as
   additive math.

## 2026-05-03T22:25Z Packed PR84+PR82 Wave

The first PR84+PR82 expanded-member T4 wave resolved as infrastructure-only
negative evidence:

- `exact_eval_pr84_pr82_controls_all600_g4dn_t4_20260503T2205Z`
- `exact_eval_pr84_pr82_qrm1_all072_randmulti_g4dn_t4_20260503T2205Z`
- `exact_eval_pr84_pr82_controls_qrm1_all072_g4dn_t4_20260503T2205Z`

All three accrued small T4 cost and then reported provider `Failed`, with empty
remote artifact directories and no `contest_auth_eval.json`. Claims were closed
as `failed_empty_artifact_dir_infra`. This is not score evidence and not method
evidence.

Godel's local packed-payload follow-up produced an opt-in
`public_payload_plus_qpost` archive layout that preserves PR84's public member
`p` byte-for-byte and adds only `qpost.bin`. The layout is 477 bytes smaller
than the expanded-member PR84+PR82 variants and remains `score_claim=false`
until exact CUDA auth eval.

Queued packed T4 exact-eval wave:

- `exact_eval_pr84_pr82_controls_all600_packedp_g4dn_t4_20260503T2224Z`,
  archive bytes `218396`, SHA
  `9fd8be8ba707e13eb5db06daff51ae8381e4e0f67f8a461883d4cd653ca7b82a`.
- `exact_eval_pr84_pr82_qrm1_all072_randmulti_packedp_g4dn_t4_20260503T2224Z`,
  archive bytes `232355`, SHA
  `d82bc774de93c073b31c1acc9dd24c57b28d14b554cee53c7b7be3404c8f2e2c`.
- `exact_eval_pr84_pr82_controls_qrm1_all072_packedp_g4dn_t4_20260503T2222Z`,
  archive bytes `234886`, SHA
  `3ccea52612fa038bd0d2ce9d0b6389e97ad03a41be5562610e55f54ce912786a`.

The source and artifact custody manifest for the two 22:24Z siblings is
`.omx/state/pr84_pr82_packedp_wave_g4dn_t4_20260503T2224Z_manifest.json`.
The controls+QRM1 22:22Z job used
`.omx/state/pr84_pr82_controls_qrm1_all072_packedp_g4dn_t4_20260503T2222Z_manifest.json`.
All three require harvest and adjudication before any score/rank statement.

Two attempted GCP T4 hedges for the `controls_all600_packedp` bytes were
blocked before remote job creation:

- Explicit `--cloud-account gcp-lightning-public-prod` failed the Lightning
  Studio cloud-account namespace guard.
- Omitting `--cloud-account` failed the local submit guard because GCP machine
  `n1-standard-8` requires an explicit cloud-account route.

Both were closed as predispatch/no-job failures with zero GPU spend. The valid
packed wave remains the g4dn/T4 queue above.

## 2026-05-03T22:29Z PR84 QMA9 Prefix Byte Screen

The full-stream pure-Python PR84 QMA9 byte search was interrupted after about
ten minutes without emitting a manifest. To avoid local CPU becoming the
bottleneck, it was replaced with a bounded 32-frame prefix screen:

- Artifact:
  `experiments/results/qma9_pr84_range_mask_byte_screen_20260503_codex_prefix32/qma9_pr84_range_mask_bitstream_profile.json`.
- Candidate count: `16`.
- Accepted local byte wins: `0`.
- QMF1 modes: disabled, because first-row specialization had already produced
  local negatives.
- QMB1 vertical block widths tested:
  `1,2,3,4,5,6,8,12,16,24,32,48,64,96,128`.

All QMB1 candidates preserved raw-mask parity on the prefix but lost bytes
against the PR84 QMA9 reference re-encode. The least-bad tested prefix result
was `qmb1_vertical_block_escape_bw128`, still `+1491` bytes versus the prefix
reference. This does not justify remote dispatch. Next QMA9 work should target
different model/context/backoff mechanics or a faster compiled full-stream
search, not the measured QMB1/QMF1 family.

## 2026-05-03T22:51Z Packed PR84+PR82 Runtime Fix1 Rerun

The first packed PR84+PR82 T4 wave did not produce score JSON. The artifact
dirs were not empty; each contained the exact submitted archive, preflight
artifacts, and `auth_eval.log`, but no `contest_auth_eval.json`. The log showed
a robust-current runtime bug before scoring:

```text
NameError: name 'pr81_router_actions' is not defined
```

Scope:

- `exact_eval_pr84_pr82_controls_all600_packedp_g4dn_t4_20260503T2224Z`,
  archive bytes `218396`, SHA
  `9fd8be8ba707e13eb5db06daff51ae8381e4e0f67f8a461883d4cd653ca7b82a`.
- `exact_eval_pr84_pr82_qrm1_all072_randmulti_packedp_g4dn_t4_20260503T2224Z`,
  archive bytes `232355`, SHA
  `d82bc774de93c073b31c1acc9dd24c57b28d14b554cee53c7b7be3404c8f2e2c`.
- `exact_eval_pr84_pr82_controls_qrm1_all072_packedp_g4dn_t4_20260503T2222Z`,
  archive bytes `234886`, SHA
  `3ccea52612fa038bd0d2ce9d0b6389e97ad03a41be5562610e55f54ce912786a`.

Evidence status: `invalid`, `score_claim=false`, `method_evidence=false`.
This is a harness/runtime bug in the PR84 no-router QPOST path, not evidence
against PR84+PR82 stacking.

Hardening landed:

- `src/tac/deploy/lightning/batch_jobs.py` now mirrors partial exact-eval
  artifacts when score JSON is missing and records
  `terminal_class=exact_eval_missing_score_json` instead of crashing on `scp`.
- `submissions/robust_current/inflate_renderer.py` now passes optional
  `pr81_router_actions` through `_generate_and_write`, defaulting to `None`
  for PR84 no-router archives.
- Regression tests cover both the partial-harvest failure class and joint
  generation without router actions.

Verification:

```text
.venv/bin/python -m py_compile submissions/robust_current/inflate_renderer.py src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py src/tac/deploy/lightning/batch_jobs.py src/tac/tests/test_lightning_batch_jobs.py
.venv/bin/python -m pytest src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py src/tac/tests/test_lightning_batch_jobs.py -q
121 passed, 1 warning
```

Fixed-source manifest:
`.omx/state/pr84_pr82_packedp_fix1_wave_g4dn_t4_20260503T2252Z_manifest.json`,
remote-verified with `1488` files, `26710156` bytes, manifest SHA
`3b68bf31d0f57e9b5a91e8003cc3c52779ca5d86d42ea727b5da4b7c343d8db6`.

Queued fixed-source T4 reruns:

- `exact_eval_pr84_pr82_controls_all600_packedp_fix1_g4dn_t4_20260503T2253Z`
- `exact_eval_pr84_pr82_qrm1_all072_randmulti_packedp_fix1_g4dn_t4_20260503T2253Z`
- `exact_eval_pr84_pr82_controls_qrm1_all072_packedp_fix1_g4dn_t4_20260503T2253Z`

All three use identical archive bytes to the failed pre-fix wave. Any score
claim still requires state-derived harvest of `contest_auth_eval.json`,
adjudication, component recomputation, and claim closure.
