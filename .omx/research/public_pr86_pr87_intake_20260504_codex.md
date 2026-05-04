# Public PR86/PR87 Intake - 2026-05-04 Codex

## Scope

This is a contest-compliance and implementation intake note for the newest
public GitHub PRs observed through the GitHub API on 2026-05-04 UTC.

Evidence grade: external/local forensic unless explicitly exact CUDA-evaluated.
No score claim is made here.

## PR87: `100_bytes`

Source: <https://github.com/commaai/comma_video_compression_challenge/pull/87>

Observed public metadata:

- title: `Add 100_bytes submission`
- head: `manthedan/comma_video_compression_challenge@fde5153dd8472734796a1abc68843c98387328c3`
- claimed archive size: `100` bytes
- claimed report score: `0.10`
- changed files: `submissions/100_bytes/inflate.py`, `inflate.sh`

Local intake:

- `inflate.py` size: `767055` bytes
- `inflate.py` SHA-256:
  `7fb6266b448b2a1f51f7b809e7fdf6ba4d37ec7e5f30fc0affe0d2ff6d9f267f`
- `inflate.sh` size: `418` bytes
- `inflate.sh` SHA-256:
  `6c75c76c628c1aa299d5f246c6b6b42129a6d934e7e7fd37bc5495cf9b5bad22`

Verdict for our system: invalid/external loophole, not a contest-faithful
candidate. The PR body itself says the approach is not in the spirit of the
competition. It moves score-affecting payload and correction streams into
runtime source while charging a 100-byte dummy archive. This may be useful as
harness-forensics, but it cannot promote, rank, or drive our archive claims.

Permanent guard added:

- `scripts/launch_lightning_batch_job.py` now blocks non-dry-run Studio
  exact-eval submit for external public-replay inflate runtimes that appear to
  embed large payload literals in source while charging a tiny archive.
- Break-glass requires
  `--allow-source-embedded-payload-runtime-reason` and should be used only for
  forensic quarantine, not score evidence.
- Regression test:
  `src/tac/tests/test_public_replay_exact_eval_hardening.py::test_public_replay_submit_blocks_source_embedded_payload_loophole`

## PR86: `jas0xf_adversarial_neural_representation`

Source: <https://github.com/commaai/comma_video_compression_challenge/pull/86>

Observed public metadata:

- title: `jas0xf_adversarial_neural_representation (0.27)`
- head: `jas0xf/comma_video_compression_challenge@b326decbdc4aefdca363e1c3fa770765a5bae250`
- claimed archive size: `207579` bytes
- changed files include `inflate.py`, `compress.py`, `training/hpac.py`,
  `training/archive.py`, `archive.zip`, `writeup.pdf`

Archive anatomy from local intake:

- `master.pt.gz`: `31144` bytes
- `slave.pt.gz`: `32287` bytes
- `hpac.pt.ppmd`: `28243` bytes
- `tokens.bin`: `113900` bytes
- `meta.pt`: `1499` bytes
- total ZIP member payload: `207073` bytes

Design signal:

- HPAC is a patch/group autoregressive token entropy model using
  `constriction`.
- The representation is not a direct PR85/QMA9 byte drop-in. The local
  PR86-HPAC-on-PR85-QMA9 parity probe failed with constriction's invalid
  entropy-model assertion, which means direct transfer is blocked until the
  exact PR86 token contract and dependency/runtime version contract are
  reproduced.
- The high-value reusable idea is still real: learned entropy coding over
  scorer-aligned token maps plus compact neural master/slave renderers.

Recorded local artifact:

- `experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_pr85_qma9_parity_probe.json`

Next action:

1. Reproduce PR86 inflate end-to-end in an isolated public replay dir.
2. Verify whether `tokens.bin` encodes raw token maps or residual tokens for
   the submitted archive; `training/hpac.py` describes residuals, while
   `training/archive.py::write_tokens` currently feeds raw token maps to
   `encode_frame`.
3. Only after PR86 decode parity is understood, test HPAC transfer to PR85/QMA9
   or train an Apogee-owned Rust/Python HPAC variant.

## PR85 Fixed-Runtime Bridge

Current status: PR85 has exact T4 public-replay evidence, but the fixed
`robust_current` runtime does not yet own the full PR85 v5 `x` bundle grammar.
It has QMA9 mask support, but PR85 also requires the exact micro-bundle
contract: v5 24-byte length header, fixed-length `bias` and `region`, and the
implicit `randmulti` tail.

Permanent bridge landed:

- `src/tac/pr85_bundle.py` provides the canonical fail-closed parser/serializer
  for PR85 v5 and explicit-30 local recode bundles.
- `src/tac/tests/test_pr85_bundle.py` verifies synthetic v5, explicit-30,
  fixed-length rejection, member-name safety, and real PR85 archive parse/pack
  byte identity when the intake archive is present.

Next fixed-runtime gate:

1. Move PR85 segment consumption behind `tac.pr85_bundle` instead of
   lane-local parsers.
2. Add a `robust_current` replay-readiness preflight that proves QMA9, QH0,
   P1D1, post, motion, bias/region, and randmulti decode contracts before any
   fixed-runtime exact eval.
3. Only then dispatch a no-op fixed-runtime PR85 exact eval with an active lane
   claim and runtime-tree hash custody.

## PR85 Randmulti Group Water-Fill: Bit-Level Candidate Build

New local tooling:

- `experiments/plan_pr85_randmulti_group_waterfill.py` decomposes the PR85
  `randmulti` stream into the 72 public replay groups and row atoms, then
  allocates exact negative CUDA component deltas across active sparse-row
  overlap. This remains `planning_only`.
- `experiments/build_pr85_randmulti_group_policy_candidates.py` consumes those
  policy rows and emits deterministic PR85 single-member `x` archive
  candidates. It only mutates the charged `randmulti` segment: selected groups
  preserve source rows; unselected groups become exact zero rows.
- `src/tac/tests/test_build_pr85_randmulti_group_policy_candidates.py`
  verifies selected-group preservation, zeroed-group semantics, duplicate
  group fail-closed behavior, deterministic ZIP member contract, and no score
  claim.

Real PR85 byte screen from
`experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex/`:

- `waterfill_top001`: `221736` bytes, SHA
  `2e671a7811261261cac99d571ec42f184e60418ee0e4dc1d7e84b525de6ee638`,
  preserves only group `0`, zeros `71` groups, rate delta `-0.009716213844`.
  The manifest now records per-group raw signal: source `randmulti` has
  `13496` nonzero choices and `27105` decoded sparse bytes; this candidate
  keeps `1194` nonzero choices and `2478` decoded sparse bytes.
- `waterfill_top002`: `222197` bytes, preserves groups `0,19`.
- `waterfill_top004`: `222964` bytes, preserves groups `0,19,20,63`.
- `waterfill_top008`: `224292` bytes, preserves groups
  `0,19,20,21,62,63,64,65`.
- `waterfill_top016`: `226766` bytes, preserves groups
  `0,19,20,21,22,26,27,28,31,33,61,62,63,64,65,66`.

The all-groups recompression policies are not dispatchable because they are
larger than source by `353` bytes; they are useful as a no-op/codec baseline.

Exact CUDA dispatch:

- Lane claim: `pr85_randmulti_top001_t4`
- Job: `exact_eval_pr85_randmulti_top001_t4_20260504T0047Z`
- Archive: `experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex/waterfill_top001/archive.zip`
- Status at submission: Pending on Lightning T4 (`g4dn.xlarge`)
- Purpose: cliff calibration for the entire randmulti water-fill curve.

Diagnostic hedge:

- Lane claim: `pr85_randmulti_top001_l40sdiag`
- Job: `exact_eval_pr85_randmulti_top001_l40sdiag_20260504T0049Z`
- Same archive SHA/bytes as the T4 job.
- Purpose: faster component-truth harvest only; not promotion evidence.

Conservative safety diagnostic:

- Lane claim: `pr85_randmulti_top016_l40sdiag`
- Job: `exact_eval_pr85_randmulti_top016_l40sdiag_20260504T0056Z`
- Archive: `experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex/waterfill_top016/archive.zip`
- Bytes/SHA: `226766`,
  `f791cdf11e7ea8161b3d60e157dd3dc3128bf102ea7214a111941b79b7251fed`
- Purpose: preserve `16` high-value groups (`5705` nonzero choices) while
  still saving `9562` bytes, giving a safer component-preservation point if
  `top001` over-prunes.

The first submit attempt was blocked correctly because the staged manifest did
not include the external public replay runtime closure. The workspace was
restaged with `README.md`, `inflate.py`, `inflate.sh`, and
`range_mask_codec.cpp`; the non-dry-run exact eval then submitted cleanly. This
is a self-protecting preflight success, not a bug to bypass.

PR85 fixed-runtime readiness preflight:

- Artifact:
  `experiments/results/public_pr85_intake_20260503_codex/pr85_fixed_runtime_readiness_preflight.json`
- Current result: `ready_for_fixed_runtime_exact_eval=false`.
- Remaining blockers:
  `pr85_single_member_x_dispatch`,
  `pr85_bundle_expands_to_runtime_members`,
  `qh0_model_loader_available`,
  `p1d1_pose_loader_available`,
  `pr85_sidechannels_exposed_to_qpost`,
  `pr85_randmulti_v5_consumption`.

Verification:

- `25 passed` for PR85 bundle, fixed-runtime preflight, PR86 HPAC parity,
  randmulti water-fill planner, randmulti policy builder, and source-embedded
  payload guard tests.
- `git diff --check` passed on the touched surfaces.

## 2026-05-04 Codex Exact Harvest And Execution Discipline Update

The randmulti water-fill exact probes turned the local byte screen into useful
negative component evidence:

- `waterfill_top001` T4 exact eval:
  `experiments/results/lightning_batch/exact_eval_pr85_randmulti_top001_t4_20260504T0047Z/contest_auth_eval.adjudicated.json`.
  Archive bytes `221736`, SHA
  `2e671a7811261261cac99d571ec42f184e60418ee0e4dc1d7e84b525de6ee638`,
  score `0.2807384658678841`, PoseNet `0.00053225`, SegNet `0.00060138`.
  This is an A++ exact negative for that measured policy: the rate gain does
  not pay for the pose loss.
- `waterfill_top016` L40S diagnostic exact eval:
  `experiments/results/lightning_batch/exact_eval_pr85_randmulti_top016_l40sdiag_20260504T0056Z/contest_auth_eval.adjudicated.json`.
  Archive bytes `226766`, SHA
  `f791cdf11e7ea8161b3d60e157dd3dc3128bf102ea7214a111941b79b7251fed`,
  score `0.3262003862199225`, PoseNet `0.00137101`, SegNet `0.00058116`.
  This is diagnostic-only because it ran on L40S, but it shows the top-16
  group policy is even farther outside the pose trust region.

Actionable conclusion: coarse randmulti group deletion is not the next
promotion path. Keep the active `top004` and `top008` T4 evals as already
queued calibration, but shift new local build work to finer post/motion atoms,
PR89 final-bias interaction tests, and PR86 HPAC transfer analysis.

The operating protocol was hardened in `AGENTS.md` with an execution
accountability rule: autonomous contest turns must advance concrete artifacts
or exact evidence, and narrative-only strategy loops are no longer acceptable
unless they directly change the next build, eval, guard, or dispatch.

## 2026-05-04 PR87 100-Byte Intake Classification

Fresh GitHub PR refresh artifact:
`experiments/results/public_pr_refresh_20260504_latest.json`.

PR87 (`Add 100_bytes submission`) was downloaded for source inspection under
`experiments/results/public_pr87_100_bytes_intake_20260504_codex/replay_submission/`.
Runtime hashes:

- `inflate.py` SHA
  `7fb6266b448b2a1f51f7b809e7fdf6ba4d37ec7e5f30fc0affe0d2ff6d9f267f`.
- `inflate.sh` SHA
  `6c75c76c628c1aa299d5f246c6b6b42129a6d934e7e7fd37bc5495cf9b5bad22`.

Classification: forensic/external only for this project. The runtime embeds
large score-affecting base85 payloads directly in `inflate.py` and documents a
dummy 100-byte compact archive path. That may be studied as a public-submission
exploit pattern, but it is not strict archive-byte custody under the current
Apogee contest-faithful rules. Do not copy this pattern into our scored archive
path unless the rule interpretation is explicitly changed and re-adjudicated.

Current practical transfer targets remain PR85/PR89/PR86:

- PR85 is the exact T4 floor anchor.
- PR89 final-bias is a charged `fb` atom and is already queued as an L40S
  interaction diagnostic on `waterfill_top004`.
- PR86 HPAC remains blocked for transfer until token semantics and decode
  parity are closed.

## 2026-05-04 Randmulti Top004/Top008 A++ Closure

The remaining already-queued randmulti water-fill T4 probes are now harvested
or locally adjudicated from mirrored exact artifacts:

- `waterfill_top004` T4 exact eval:
  `experiments/results/lightning_batch/exact_eval_pr85_randmulti_top004_t4_20260504T0106Z/contest_auth_eval.adjudicated.json`.
  Archive bytes `222964`, SHA
  `ef07307933a9458dc096ac40921fff1d203a7daeec0ad82fac8e074bd02f14e7`,
  score `0.3024608857881686`, PoseNet `0.00088594`, SegNet
  `0.00059874`. Evidence grade `A++ contest T4`, promotion eligible as an
  exact archive, but a measured exact negative versus PR85 because it is
  `+0.04439477549419074` worse.
- `waterfill_top008` T4 exact eval:
  `experiments/results/lightning_batch/exact_eval_pr85_randmulti_top008_t4_20260504T0106Z/contest_auth_eval.adjudicated.json`.
  Archive bytes `224292`, SHA
  `f7f0034d24f93581f67dd77429b8368df29728638368f4470c7fe4bf3a5afb03`,
  score `0.30536973018825064`, PoseNet `0.00094704`, SegNet
  `0.00058707`. The remote job wrote a complete adjudication log but did not
  mirror `contest_auth_eval.adjudicated.json`; the missing JSON was regenerated
  locally from the mirrored exact `contest_auth_eval.json`, `eval_provenance`,
  and archive bytes with `scripts/adjudicate_contest_auth_eval.py`. Evidence
  grade remains `A++ contest T4`, but it is `+0.04730361989427278` worse than
  PR85.

Decision: the coarse randmulti deletion bracket is now closed as a measured
implementation family for this cycle. `top001`, `top004`, `top008`, and
`top016` all trade rate for too much pose distortion. Do not dispatch more
coarse prefix/group deletion variants unless a new finer-grained atom, repair
stream, final-bias interaction, or runtime semantic change alters the
component economics.

## 2026-05-04 Harvest Harness Fixes And Post/Motion Diagnostic

Two harness fragilities were fixed while harvesting the next PR85/PR89
interaction wave:

- SSH harvest now recovers missing adjudication copy artifacts when the remote
  exact eval wrote machine-readable `contest_auth_eval.json`, metadata,
  provenance, and archive bytes but the persisted artifact mirror lagged the
  adjudication JSON copies. Successful re-harvest also clears stale
  `terminal_class` values from state records. Regression coverage:
  `src/tac/tests/test_lightning_batch_jobs.py::test_harvest_ssh_artifacts_recovers_missing_adjudication_copy_from_metadata`.
- `experiments/contest_auth_eval.py` now allows exact basename `fb` as a
  charged PR89-style final-bias atom while still rejecting arbitrary
  extensionless debug members. Regression coverage:
  `src/tac/tests/test_contest_auth_eval.py::test_archive_member_validator_accepts_charged_pr89_final_bias_member`
  and
  `src/tac/tests/test_contest_auth_eval.py::test_archive_member_validator_still_rejects_unknown_extensionless_debug_member`.

Post/motion atom diagnostic:

- `preserve_post23_motion` L40S exact diagnostic:
  `experiments/results/lightning_batch/exact_eval_pr85_post23_motion_l40sdiag_20260504T0118Z/contest_auth_eval.adjudicated.json`.
  Archive bytes `235924`, SHA
  `1448ce2436cf6870cf4671b45e090135b3d0bfbca685cec850b9ee02a0b56c06`,
  score `0.27449221257089806`, PoseNet `0.0003451`, SegNet
  `0.00058655`. Evidence grade `A score-grade` only because hardware was
  L40S. It is worse than PR85 by `+0.016426102276920207`, so do not promote
  this exact policy to T4. It is nevertheless useful: preserving post stages
  2/3 is far safer than coarse randmulti group deletion, so future post/motion
  work should use smaller learned/fitted atoms rather than whole-stage
  neutralization.

Final-bias interaction status:

- First `waterfill_top004_fb` diagnostic failed before score because
  `contest_auth_eval.py` rejected charged archive member `fb`. This is a
  harness allowlist bug, not method evidence.
- The same archive bytes were restaged after the allowlist fix with manifest
  `.omx/state/pr85_randmulti_top004_fb_allowfb_l40sdiag_20260504T0129Z_manifest.json`
  and submitted as
  `exact_eval_pr85_randmulti_top004_fb_allowfb_l40sdiag_20260504T0129Z`.
  Archive bytes remain `223344`, SHA
  `22d5a0a6919398ac6c91f6496b42654cd740df3b21d57d52094acbad1afbb3bb`.
  This is the active highest-signal interaction probe to harvest next.

## 2026-05-04 Final-Bias Negative And PR86 HPAC Replay Dispatch

The `waterfill_top004_fb` L40S diagnostic has been harvested:

- Artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_randmulti_top004_fb_allowfb_l40sdiag_20260504T0129Z/contest_auth_eval.adjudicated.json`.
- Archive bytes `223344`, SHA
  `22d5a0a6919398ac6c91f6496b42654cd740df3b21d57d52094acbad1afbb3bb`.
- Score `0.30525181986353395`, PoseNet `0.00093154`, SegNet
  `0.0006002`, samples `600`, hardware `NVIDIA L40S`.
- Evidence grade `A score-grade`; not promotion eligible because it is not
  T4/equivalent hardware.

Decision: PR89-style final-bias does not rescue coarse PR85 randmulti top004.
It is a measured interaction negative for this stacked policy. Keep PR89
final-bias as a possible tiny charged atom for safer PR85-family candidates,
but do not promote this exact stack to T4.

PR86 HPAC follow-up:

- Added a local public-replay wrapper:
  `experiments/results/public_pr86_intake_20260504_codex/inflate.sh`.
- Staged exact PR86 replay artifacts through manifest
  `.omx/state/public_pr86_hpac_l40sdiag_20260504T0139Z_manifest.json`.
- Queued forensic L40S exact replay:
  `exact_eval_public_pr86_hpac_l40sdiag_20260504T0139Z`.
- Archive bytes `207579`, SHA
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`.

This is not a PR85 transfer and not a promotion claim. It tests whether public
PR86's own archive can survive the contest CUDA path after dependency closure.
If it fails with the same HPAC entropy-decode assertion, HPAC remains a design
idea to reimplement under our deterministic contract, not a direct transplant.

Dependency hardening:

- `pyproject.toml` now declares HPAC runtime dependencies
  `constriction>=0.4,<0.5` and `pyppmd>=1.3,<2.0`.
- `uv.lock` was refreshed.
- The immediate Lightning Studio environment was repaired with
  `constriction==0.4.2` and `pyppmd==1.3.1` before queueing the PR86 forensic
  replay.

Verification:

- `7 passed` for PR86 HPAC replay parity plus PR89 final-bias allowlist tests.
- `py_compile` passed for `experiments/pr86_hpac_replay_parity.py` and
  `experiments/contest_auth_eval.py`.
- `git diff --check` passed for the touched HPAC dependency/replay surfaces.

Lightning namespace hardening:

- The first PR86 forensic submit attempts failed before job creation because
  Lightning SDK autodetection could not resolve `teamspace`/`studio`.
- `scripts/launch_lightning_batch_job.py` now fails locally before SDK submit
  unless a non-dry-run Lightning job has an explicit `--studio` or `--image`;
  Studio-backed jobs also require `--teamspace` and `--user`/`--org`.
- Regression coverage:
  `src/tac/tests/test_lightning_batch_jobs.py::test_non_dry_run_studio_submit_with_teamspace_requires_user_or_org`.

## 2026-05-04 PR86 T4 Hedge And Dispatch Fragility Closure

The first PR86 L40S diagnostic showed inconsistent provider telemetry:
`refresh-status` reported `REMOTE_STATUS_RECONCILIATION_REQUIRED`, remote
`Pending`, cost about `$0.2208`, and no artifact directory was visible under
`/teamspace/studios/this_studio/pact/experiments/results/lightning_batch/`.
Because PR86 is a public-neural-codec intake path and its claimed runtime is
T4, I queued one contest-equivalent T4 hedge instead of waiting behind the
ambiguous L40S state.

T4 hedge:

- Claim:
  `public_pr86_hpac_replay_t4_hedge`.
- Job:
  `exact_eval_public_pr86_hpac_t4_hedge_20260504T0152Z`.
- State:
  `.omx/state/public_pr86_hpac_t4_hedge_20260504T0152Z_batch_jobs.json`.
- Machine:
  `g4dn.xlarge` / Lightning `T4_SMALL`.
- Archive bytes `207579`, SHA
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`.
- Runtime env pins:
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`,
  `UV_INDEX_STRATEGY=unsafe-best-match`.
- Initial status after submit:
  `Pending`, cost `$0.0`.

Dispatch fragility found and fixed/recorded:

- Launcher CLI misuse: `--required-device` and `--required-samples` are
  adjudicator arguments emitted by the wrapper, not public
  `exact-eval` CLI flags. The bad command failed before job creation; future
  work must continue to use `--help`/argparse as the command surface.
- GCP T4 route: `n1-standard-8` is available only through
  `gcp-lightning-public-prod`, but the current Studio
  `lossy-compression-challenge` lives in a different cloud-account namespace.
  The launcher rejected this as
  `studio_cloud_account_namespace_mismatch`; do not retry that Studio-backed
  GCP route unless a matching Studio/env exists or the job is image-backed.
- AWS/Studio T4 route: `g4dn.xlarge` requires the explicit
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124` env so inflate-side Torch does not
  resolve a CUDA-13 wheel on an older T4 driver. With that pin, submit
  succeeded.

No score claim yet. The next action is state-derived harvest of whichever
PR86 job first emits `contest_auth_eval.json`, then immediate closure of the
other job's claim as duplicate/stopped if redundant.

## 2026-05-04T02:10Z Public PR Refresh

GitHub PR refresh:

- No PR `#90` exists (`gh pr view 90` returns no PullRequest).
- Newest visible PRs remain `#89` through `#85`.
- Open score-relevant public PRs:
  - `#86` `jas0xf_adversarial_neural_representation (0.27)`
  - `#85` `Adaptive masking joint frame model`
  - `#84` `Rang Mask Optimizations (0.275)`
  - `#81` `qzs3 range mask`
- `#89` `henosis_final_bias (0.28)` is closed and remains worse than PR85 under
  the local exact diagnostic/intake evidence.

Operational status:

- PR86 T4 hedge `exact_eval_public_pr86_hpac_t4_hedge_20260504T0152Z` is
  `Running`, cost about `$0.0143`.
- PR86 L40S diagnostic `exact_eval_public_pr86_hpac_l40sdiag_20260504T0139Z`
  is terminal-failed in the SDK, no artifact dir; the dispatch claim was closed
  as `failed_remote_failed_no_artifact_dir`.

Branch decision remains unchanged: harvest PR86 T4 for exact evidence, but do
not stall PR85 work behind it.  PR85 is the current exact best and the active
stacking substrate.

## 2026-05-04T02:17Z PR86 Whitelist Retry

The first PR86 T4 hedge failed before score JSON because the strict archive
validator rejected charged compressed model/state members:

- `master.pt.gz`
- `slave.pt.gz`
- `hpac.pt.ppmd`

This is a harness allowlist bug, not PR86 method evidence.  The terminal claim
for `exact_eval_public_pr86_hpac_t4_hedge_20260504T0152Z` was closed as
`failed_archive_whitelist_harness_bug`.

Permanent guard:

- `_KNOWN_ARCHIVE_SUFFIXES` now includes `.pt.gz` and `.pt.ppmd`.
- `test_archive_member_validator_accepts_charged_pr86_hpac_members` protects
  PR86-style HPAC archives.
- The PR85 `.qma9` archive-member bug was fixed in the same class of guard, so
  QMA9 and HPAC archive contracts are now both represented in
  `contest_auth_eval.py`.

Relaunch:

- Claim:
  `public_pr86_hpac_replay_t4_retry1`.
- Job:
  `exact_eval_public_pr86_hpac_t4_retry1_20260504T0213Z`.
- Archive bytes `207579`, SHA
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`.
- Source manifest:
  `.omx/state/public_pr86_hpac_t4_retry1_20260504T0213Z_manifest.json`.
- Initial status:
  `Pending`, cost `$0.0`.

No PR86 score claim exists until this retry emits `contest_auth_eval.json`.

## 2026-05-04T02:24Z HPAC/Token Anatomy Forensics

Local PR86 forensic tooling now records the archive/token contract without any
remote dispatch:

- Tool:
  `experiments/profile_pr86_hpac_token_anatomy.py`.
- Artifacts:
  `experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_token_anatomy_forensics.json`
  and `.md`.
- Verification:
  `10 passed` for
  `test_profile_pr86_hpac_token_anatomy.py` and
  `test_pr86_hpac_replay_parity.py`.

Findings:

- Archive identity matches the public PR86 intake: bytes `207579`, SHA
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`.
- Member set is exact and sidecar-clean:
  `master.pt.gz`, `slave.pt.gz`, `hpac.pt.ppmd`, `tokens.bin`, `meta.pt`.
- `tokens.bin` is the main byte surface: `113900` bytes, `28475` uint32
  words.
- The submitted archive is classified as `raw_tokens` even though training
  notes discuss residual-token training.  That means the PR85 transfer target
  is not a drop-in entropy wrapper.
- Gross PR85 mask opportunity is still large: replacing PR85's `159011`-byte
  QMA9 mask segment with PR86-style HPAC token+meta byte scale would save about
  `15369` bytes before runtime/contract costs, but it requires full PR86
  decode/re-encode parity and PR85 token parity first.

Decision:

Keep PR86 exact replay running as score truth.  In parallel, treat HPAC as a
contract-port design prior for a PR85-owned mask coder, not as an immediate
archive transplant.

## 2026-05-04T02:44Z PR86 HPAC Failed-Closed Contract Gate

The PR86 retry advanced past the archive allowlist bug, but failed before
score JSON on true T4 inside HPAC decode:

- Job:
  `exact_eval_public_pr86_hpac_t4_retry1_20260504T0213Z`.
- Archive bytes `207579`, SHA
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`.
- Failure:
  `AssertionError: Tried to decode from compressed data that is invalid for
  the employed entropy model`.
- No `contest_auth_eval.json`; no score claim.
- Dispatch claim closed as
  `failed_hpac_entropy_decode_before_score_json`.

Local parity gate:

- Tool:
  `experiments/pr86_hpac_replay_parity.py`.
- Artifact:
  `experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_full_decode_reencode_gate_20260504_codex.json`.
- Evidence grade:
  `planning_only_local_decode_reencode_gate`.
- Result:
  `failed_closed` during CPU full decode before byte-exact re-encode could
  complete.
- Location:
  frame `0`, group `10`, symbol `191`, after `5760` decoded symbols.
- Source `tokens.bin` SHA-256:
  `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`.
- Verification:
  `7 passed` for `src/tac/tests/test_pr86_hpac_replay_parity.py`.

The PR86-to-PR85 contract planner now defaults to the stricter full
decode/reencode gate when present, instead of silently reading the older
bounded-prefix diagnostic.  Regenerated plan:

- `experiments/results/pr86_hpac_pr85_contract_port_20260504_codex/plan.json`.
- `dispatchable=false`.
- `next_gate=pr86_full_decode_reencode_token_parity`.
- `status=failed_closed`.

Decision:

PR86 HPAC remains useful as byte-economics research, but it is not a
dispatchable transplant and not a PR85 entropy-code replacement until the HPAC
probability/token/device contract is reproduced through full decode/reencode
parity and PR85 raw-token parity.

## 2026-05-04T02:55Z PR85 Token-Source Gate Closed

The PR85-owned entropy-coder path now has an exact local token-source target:

- Tool:
  `experiments/profile_pr85_qma9_token_source.py`.
- Main artifact:
  `experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_token_source_profile.json`.
- Extracted raw tokens:
  `experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin`.
- Token tensor:
  shape `[600, 512, 384]`, dtype `uint8`, observed range `0..4`.
- Token SHA-256:
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`.
- Mask segment:
  `159011` bytes, SHA-256
  `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`.
- Verification:
  `3 passed` for `src/tac/tests/test_profile_pr85_qma9_token_source.py`.

The PR86-to-PR85 contract planner now consumes this token-source artifact:

- Regenerated plan:
  `experiments/results/pr86_hpac_pr85_contract_port_20260504_codex/plan.json`.
- `pr85_baseline_token_extraction` now passes as
  `passed_token_source_profiled`.
- Remaining blockers:
  PR86 exact score evidence, PR86 full decode/reencode parity, PR85 HPAC token
  parity, PR85 runtime output parity, and candidate archive byte closure.

Decision:

This closes the baseline-source gate only.  It does not unlock remote
dispatch.  The next entropy-code implementation must fit a PR85-owned model
against token SHA
`c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`, prove
decoded-token SHA parity, then prove runtime output parity before any exact
CUDA eval.

## 2026-05-04T03:10Z Public PR Refresh

GitHub API refresh over the upstream pull requests shows the current public
open frontier is still moving:

- PR #86:
  `jas0xf_adversarial_neural_representation (0.27)`, open, updated
  `2026-05-04T03:04:27Z`.
- PR #85:
  `Adaptive masking joint frame model`, open, updated
  `2026-05-04T02:47:17Z`.
- PR #84:
  `Rang Mask Optimizations (0.275)`, open, updated
  `2026-05-04T02:46:46Z`.
- PR #81:
  `qzs3 range mask`, open, updated `2026-05-04T02:43:42Z`.
- PR #87:
  `Add 100_bytes submission`, closed.
- PR #89:
  `henosis_final_bias (0.28)`, closed.

Local exact replay status remains stricter than PR title text:

- PR85 has exact local/T4 replay at score `0.25806611029397786` and remains the
  current measured anchor in this workspace.
- PR86 remains failed-closed locally before score JSON due HPAC entropy decode
  contract mismatch.  Its public title/body are external signal, not a local
  score claim.
- PR87 and PR78-style tiny archive loopholes remain non-faithful/forensic only
  unless upstream rules explicitly validate them through the normal contest
  path, which this workspace does not assume.

Decision:

Use PR85 as the measured contest-faithful anchor while continuing PR86
reverse-engineering as a compression-recipe source.  The immediate local
implementation target is not another title-score replay; it is extracting the
ANR/HPAC byte-economics and porting only the parts that can pass full
decode/reencode parity, runtime parity, and exact CUDA archive custody.

## 2026-05-04T03:56Z HPAC Probability-Contract Variant Probe

A cheap orchestrator-side replay probe tested whether the current PR86 HPAC
failure was caused by a single probability-model construction knob.  This was
local-only, CPU-only, and did not dispatch GPU work or claim score.

Command shape:

```bash
.venv/bin/python - <<'PY'
# import src.tac.pr86_hpac_codec, monkey-patch _categorical_from_probs,
# run decode_tokens_hpac(..., max_frames=1) for probability variants
PY
```

Source:

- PR86 archive SHA-256:
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`
- `tokens.bin` SHA-256:
  `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`

Variant results:

| variant | status | failure point |
|---|---|---|
| baseline | failed closed | frame `0`, group `10`, symbol `191`, after `5951` symbols |
| `perfect=True` | failed closed | frame `0`, group `15`, symbol `1534`, after `13822` symbols |
| epsilon `1e-9` | failed closed | frame `0`, group `10`, symbol `191`, after `5951` symbols |
| epsilon `1e-12` | failed closed | frame `0`, group `10`, symbol `191`, after `5951` symbols |
| float32 probabilities | failed closed | frame `0`, group `24`, symbol `561`, after `30513` symbols |
| rounded `16384` probability grid | failed closed | frame `0`, group `18`, symbol `448`, after `17728` symbols |

Interpretation:

The failure is not explained by the common one-knob suspects: `perfect`,
epsilon, float64-vs-float32 probability arrays, or a simple `1/16384`-style
probability grid.  Some variants move the failure later, which says probability
construction matters, but none reached one full frame.  The remaining likely
contract mismatches are deeper: exact tensor/state reconstruction, HPAC model
mode, encode/decode order, token order, or a non-obvious submitted-token
semantics difference.

Decision:

Keep PR86 HPAC failed-closed for dispatch.  The active HPAC worker should use
this probe as negative guidance and focus on contract-level differences rather
than scalar probability knobs alone.

## 2026-05-04T04:00Z PR86 Merged-Head Intake Refresh

GitHub API now reports PR86 as merged:

- PR:
  <https://github.com/commaai/comma_video_compression_challenge/pull/86>
- head SHA:
  `0eabe354f09b7490fd1cbb2b05a9102ab528d4d4`
- merge commit:
  `14bcede815306415a0005c3cd98804151bce4049`
- merged at:
  `2026-05-04T03:36:55Z`

The earlier cached `pr86_view.json` referenced the original head commit
`b326decbdc4aefdca363e1c3fa770765a5bae250`, so the source-context metadata
was stale.  A fresh merged-head intake was written to:

- `experiments/results/public_pr86_intake_20260504_merged_refresh/`
- intake summary:
  `experiments/results/public_pr86_intake_20260504_merged_refresh/intake_summary.json`
- source manifest:
  `experiments/results/public_pr86_intake_20260504_merged_refresh/source_manifest.json`

Critical custody finding:

- Fresh merged-head `archive.zip` bytes:
  `207579`
- Fresh merged-head `archive.zip` SHA-256:
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`
- This exactly matches the existing cached PR86 archive.

Decision:

The HPAC decode failure remains valid for the submitted archive bytes, but any
source-code-level reverse engineering should now use the merged-head files in
`public_pr86_intake_20260504_merged_refresh/`.  The HPAC worker has been
notified to distinguish old-artifact conclusions from current source-context
conclusions.

Follow-up source comparison:

- `inflate.py`: byte-identical between cached intake and merged-head refresh.
- `training/archive.py`: byte-identical.
- `training/hpac.py`: byte-identical.
- `training/README.md`: byte-identical.
- `training/master.py` and `training/slave.py` were newly materialized in the
  refresh directory for completeness.

This means the HPAC decode blocker is current for the submitted archive and
current HPAC source.  The stale surface was PR metadata/head tracking, not the
archive or HPAC implementation files.

Merged-head inflate smoke:

```bash
tmp=$(mktemp -d /tmp/pr86-inflate-smoke.XXXXXX)
unzip -q experiments/results/public_pr86_intake_20260504_merged_refresh/archive.zip -d "$tmp/data"
timeout 30 .venv/bin/python \
  experiments/results/public_pr86_intake_20260504_merged_refresh/inflate.py \
  "$tmp/data" "$tmp" "$tmp/out.raw"
```

Result:

- exit code: `1`
- failure:
  `AssertionError: Tried to decode from compressed data that is invalid for the employed entropy model`
- location:
  `inflate.py::decompress_tokens_hpac`, before any raw output is written.

Runtime observation:

`inflate.py` comments that HPAC decode should be forced onto CPU, but the call
passes the selected `device` variable into `decompress_tokens_hpac`.  That is a
hardware-fragility clue, but it does not explain the local CPU failure because
CPU replay also fails.

Decision:

PR86 remains external design signal, not local score evidence.  Any PR86-derived
submission must first fix/prove the HPAC token/runtime contract by producing raw
output and full decode/reencode parity on the exact archive bytes.
