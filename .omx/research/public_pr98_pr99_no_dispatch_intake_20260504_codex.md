# Public PR98/PR99 No-Dispatch Intake - 2026-05-04

Scope: PR98 `hnerv_muon_finetuned_from_pr95 (0.1963)` and PR99
`hnerv_muon_lc submission (0.20)` public frontier intake. This pass used
GitHub API/primary PR/release URLs only. No GPU job was dispatched in this pass.

## Source Custody

- PR98: <https://github.com/commaai/comma_video_compression_challenge/pull/98>
  - GitHub API: `experiments/results/leaderboard_intel_20260504_codex/pr98_api.json`
  - files: `pr98_files.json`; comments: `pr98_comments.json`,
    `pr98_review_comments.json`; commits: `pr98_commits.json`
  - release API: `pr98_release_api.json`
  - release archive URL:
    <https://github.com/EthanYangTW/comma_video_compression_challenge/releases/download/v1-hnerv-finetuned/archive.zip>
  - head tree API: `pr98_head_tree.json`
- PR99: <https://github.com/commaai/comma_video_compression_challenge/pull/99>
  - GitHub API: `experiments/results/leaderboard_intel_20260504_codex/pr99_api.json`
  - files: `pr99_files.json`; comments: `pr99_comments.json`,
    `pr99_review_comments.json`; commits: `pr99_commits.json`
  - release API: `pr99_release_api.json`
  - release archive URL:
    <https://github.com/BradyMeighan/comma_video_compression_challenge/releases/download/hnerv-muon-lc-archive/archive.zip>
  - author source URL from PR body:
    <https://github.com/BradyMeighan/comma_video_compression_challenge/tree/submission/hnerv_muon_lc>
  - source tree API: `pr99_author_source_tree.json`
  - author-linked CI URL metadata: `pr99_author_actions_run_25311463434.json`,
    `pr99_author_actions_run_25311463434_jobs.json`

## Static Score Claims

These are external claims only. This note does not promote, rank, or claim a
local score before exact CUDA replay.

| PR | External claim source | PoseNet | SegNet | Bytes | Recomputed score |
|---|---:|---:|---:|---:|---:|
| PR98 | PR title/body, CUDA report | `0.00003489` | `0.00058795` | `178392` | `0.19625777542725248` |
| PR99 | PR body CPU report; exact CI text `0.19667` | `0.00003349` | `0.00059494` | `178546` | `0.19668072586615531` |

## Archive And Runtime Custody

Machine-readable artifacts:

- `experiments/results/leaderboard_intel_20260504_codex/pr98_static_intake.json`
- `experiments/results/leaderboard_intel_20260504_codex/pr98_static_intake.md`
- `experiments/results/leaderboard_intel_20260504_codex/pr98_public_replay_intake_preflight.json`
- `experiments/results/leaderboard_intel_20260504_codex/pr99_static_intake.json`
- `experiments/results/leaderboard_intel_20260504_codex/pr99_static_intake.md`
- `experiments/results/leaderboard_intel_20260504_codex/pr99_public_replay_intake_preflight.json`
- `experiments/results/leaderboard_intel_20260504_codex/pr98_pr99_intake_summary.json`

PR98:

- archive: `experiments/results/leaderboard_intel_20260504_codex/pr98_archive.zip`
- archive bytes/SHA-256:
  `178392` / `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- release download matched the existing archive exactly.
- strict ZIP: passed; one stored member `0.bin`
- member bytes/SHA-256:
  `178284` / `fce200db2fe087cc6a051945b3fda2c37f5bbb3e19b8f20a1aea7201db0c9f5f`
- payload split:
  - meta brotli `80` bytes
  - compact `CD1` decoder brotli `162343` bytes, raw `229022`, scale bits `16`, tensors `28`
  - latents brotli `15849` bytes, raw `33720`, `600 x 28`
- runtime artifact:
  `experiments/results/leaderboard_intel_20260504_codex/pr98_runtime/`
- `contest_auth_eval` preflight runtime tree SHA-256:
  `4d71b5769e9c886e8a4e1be8997014ec47fe5d5ce5519619bf16bff0ae7f2738`
- static preflight: `ready_for_exact_eval_dispatch=true`, blockers `[]`

PR99:

- archive: `experiments/results/leaderboard_intel_20260504_codex/pr99_archive.zip`
- archive bytes/SHA-256:
  `178546` / `278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb`
- release download and committed PR archive matched the local archive exactly.
- strict ZIP: passed; one stored member `0.bin`
- member bytes/SHA-256:
  `178438` / `79b9f89e709c2892560822284a4465f17d5df8c6148b923e889683ab04bdb1d5`
- payload split:
  - decoder brotli `161883` bytes, raw `228958`
  - scales fp16 `56` bytes
  - latents brotli `15868` bytes, raw `33720`, `600 x 28`
  - correction sidecar brotli `615` bytes, raw `1202`, `598/600` non-noop pairs
- runtime artifact:
  `experiments/results/leaderboard_intel_20260504_codex/pr99_runtime/`
- `contest_auth_eval` preflight runtime tree SHA-256:
  `67fa8ef36f732be73d29053bc050a86a597b23d394ea07538451a8eb8303817f`
- static preflight: `ready_for_exact_eval_dispatch=true`, blockers `[]`

The raw PR submission files are preserved under `pr98_submission/` and
`pr99_submission/`. The `pr98_runtime/` and `pr99_runtime/` `inflate.sh`
wrappers are local exact-eval adapters that invoke the downloaded PR `inflate.py`
directly; payload-affecting Python code and model/source files are otherwise
from the PR file raw URLs.

## Compliance Risks

- PR98 and PR99 are external PR claims until our exact CUDA replay lands.
- Active dispatch claims already exist:
  - `public_pr98_hnerv_muon_finetuned_t4_replay`
  - `public_pr99_hnerv_muon_lc_t4_replay`
  Do not open duplicate claims unless those rows become terminal or stale.
- PR98 says GPU inflation is required and the runtime chooses CUDA when
  available. The report is CUDA but not local custody evidence.
- PR98 archive is a release asset rather than a committed PR file; release API
  and SHA custody are recorded.
- PR99 says GPU inflation is not required and the linked CI run is CPU, but the
  runtime still selects CUDA when available. CUDA replay can drift from CPU CI.
- PR99 original shell can install `brotli` if missing. Exact replay should stage
  dependency closure so inflate does not depend on network.
- PR99's latent-correction sidecar is charged inside `0.bin`; it is legal
  side information only if exact replay uses no external score-affecting files.
- PR99 author-linked CI run API reports head branch `master` and SHA
  `e84851da32108fcadf243c54d091cc71dc150c0e`, while the PR head is
  `6badd2506c1ce07751236fc299d0b14f2702ac29`; treat the CI link as context,
  not exact archive custody.

## Exact Next Commands

The current active claim rows already name these lane/job IDs. Do not run the
claim commands again unless the existing rows are terminal or stale.

PR98 local CUDA exact replay command:

```bash
.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/leaderboard_intel_20260504_codex/pr98_archive.zip --inflate-sh experiments/results/leaderboard_intel_20260504_codex/pr98_runtime/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir experiments/results/leaderboard_intel_20260504_codex/pr98_exact_eval_work
```

PR99 local CUDA exact replay command:

```bash
.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/leaderboard_intel_20260504_codex/pr99_archive.zip --inflate-sh experiments/results/leaderboard_intel_20260504_codex/pr99_runtime/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir experiments/results/leaderboard_intel_20260504_codex/pr99_exact_eval_work
```

If the current active claim rows need to be re-created after terminal/stale
closure, use the real helper surface:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id public_pr98_hnerv_muon_finetuned_t4_replay --platform lightning --instance-job-id exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z --agent codex:gpt-5.5 --predicted-eta-utc 2026-05-04T10:25Z --status active_exact_eval --notes "Exact T4 replay of PR98 public archive; bytes=178392 sha=7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb; static preflight ready."
```

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id public_pr99_hnerv_muon_lc_t4_replay --platform lightning --instance-job-id exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z --agent codex:gpt-5.5 --predicted-eta-utc 2026-05-04T10:25Z --status active_exact_eval --notes "Exact T4 replay of PR99 public archive; bytes=178546 sha=278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb; static preflight ready."
```

Lightning submit shape after source-manifest staging, while the matching claim
is active:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval --job-name exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z --archive /teamspace/studios/this_studio/pact/experiments/results/leaderboard_intel_20260504_codex/pr98_archive.zip --repo-dir /teamspace/studios/this_studio/pact --upstream-dir /teamspace/studios/this_studio/pact/upstream --inflate-sh /teamspace/studios/this_studio/pact/experiments/results/leaderboard_intel_20260504_codex/pr98_runtime/inflate.sh --machine T4 --studio lossy-compression-challenge --teamspace comma-lab --user adpena --python-bin .venv/bin/python --local-artifact-dir experiments/results/lightning_batch/exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z --source-manifest .omx/state/exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z_manifest.json --dispatch-lane-id public_pr98_hnerv_muon_finetuned_t4_replay --expected-archive-sha256 7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb --expected-archive-size-bytes 178392 --adjudicate --baseline-score 0.23089404465634825 --baseline-archive-bytes 178277 --predicted-band 0.18 0.24 --max-posenet-dist 0.01 --max-segnet-dist 0.01 --max-sane-score 1.0 --component-reference-label pr95_stemperm_a++ --delta-key score_delta_vs_pr95_stemperm
```

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval --job-name exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z --archive /teamspace/studios/this_studio/pact/experiments/results/leaderboard_intel_20260504_codex/pr99_archive.zip --repo-dir /teamspace/studios/this_studio/pact --upstream-dir /teamspace/studios/this_studio/pact/upstream --inflate-sh /teamspace/studios/this_studio/pact/experiments/results/leaderboard_intel_20260504_codex/pr99_runtime/inflate.sh --machine T4 --studio lossy-compression-challenge --teamspace comma-lab --user adpena --python-bin .venv/bin/python --local-artifact-dir experiments/results/lightning_batch/exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z --source-manifest .omx/state/exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z_manifest.json --dispatch-lane-id public_pr99_hnerv_muon_lc_t4_replay --expected-archive-sha256 278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb --expected-archive-size-bytes 178546 --adjudicate --baseline-score 0.23089404465634825 --baseline-archive-bytes 178277 --predicted-band 0.18 0.24 --max-posenet-dist 0.01 --max-segnet-dist 0.01 --max-sane-score 1.0 --component-reference-label pr95_stemperm_a++ --delta-key score_delta_vs_pr95_stemperm
```

## Verification

- GitHub API PR, files, comments, review comments, commits, release APIs, and
  source tree APIs were downloaded with `curl`.
- Release archive downloads matched local archive SHA-256 for PR98 and PR99.
- PR99 committed archive matched release/local archive SHA-256.
- Static archive/runtime preflight:
  - `.venv/bin/python experiments/preflight_public_replay_intake.py ... --fail-if-not-ready`
    passed for PR98 and PR99.
- Runtime syntax/import checks:
  - `bash -n` passed for both adapted `inflate.sh` wrappers.
  - `.venv/bin/python -m py_compile` passed for PR98/PR99 runtime Python files.
  - `.venv` imports `brotli`, `torch`, and `numpy`; local CUDA availability was
    false on this machine, so no local CUDA replay was attempted.
