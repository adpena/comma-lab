# Final PR Stack Synthesis - 2026-05-04

Scope: Grand Council final-stack synthesis over current public PRs and local
candidate families. This pass refreshed GitHub public PR state and wrote local
decision artifacts only. No remote job, lane claim, training, scorer load, or
exact eval was dispatched.

## Current Confirmed Frontier

The best confirmed internal score remains `PR85_STBM1BR`.

- Classification: `confirmed_A++`
- Exact artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.json`
- Score: `0.25369011029397787`
- Archive bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Samples/device: `600`, CUDA/T4
- Runtime tree SHA-256:
  `d195f4ecd0743cfd146efafee6729e96ee5428bfb28bbd0ca87cbad055494440`

This supersedes raw public PR85 for internal score custody. Public PR text,
MPS/CPU reports, local static intake, and failed replay logs remain external or
diagnostic until exact CUDA JSON exists for exact archive bytes and runtime
tree.

## GitHub Refresh

Live source:
`https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls?state=all&sort=updated&direction=desc&per_page=100`

Saved artifact:
`experiments/results/final_pr_stack_synthesis_20260504/github_pulls_latest_raw.json`

- SHA-256:
  `841319a4781be0b9156a1abc13fc89960a92f6e2a17b682673f8c2a7ab121046`
- bytes: `2089282`
- PR issue comments fetched for PRs `85`, `90`, `91`, `92`, `93`, `94`, `95`.

PR95 is new relative to the older local refresh. It is open and claims
`hnerv_muon submission (0.20)` from head
`9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9`.

## Public PR State

| PR | State | Head SHA | Title | Score Status | Evidence |
| ---: | --- | --- | --- | --- | --- |
| 85 | closed | `515abfbac1dc6ef42499d5b912e424319f982fb2` | Adaptive masking joint frame model | `0.25806611029397786`, `236328` bytes | GitHub Actions CUDA comment plus local A++ public replay |
| 86 | closed | `0eabe354f09b7490fd1cbb2b05a9102ab528d4d4` | jas0xf_adversarial_neural_representation (0.27) | recomputed text `0.2736358503762718`, `207579` bytes | external CUDA text; local replay invalid before score |
| 87 | closed | `fde5153dd8472734796a1abc68843c98387328c3` | Add 100_bytes submission | `0.10` text | invalid source-embedded payload loophole |
| 88 | closed | `b330dd75232f06ac541726fbb29de80074db8eb5` | qzs3 range mask gpu preflight | recomputed CPU text `0.2878245063016197`, `215960` bytes | CPU text, invalid for promotion |
| 89 | closed | `d970a0eaf5065807f75fc2f386bf5c21fd982ae1` | henosis_final_bias (0.28) | recomputed text `0.27511038756764417`, `236676` bytes | external CUDA text |
| 90 | open | `cce857392701e73861ad513d34906faba523f719` | add qrepro submission(0.28) | recomputed text `0.2788721801656914`, `218080` bytes | external text plus local partial CPU smoke |
| 91 | open | `77f958d24e55980d95e01e3e9767b5a94320ed43` | Hpac coder hybrid | exact text `0.24879480490416128`, `222404` bytes | external CUDA text; local HPM1 replay invalid before score |
| 92 | open | `95c711e8ec7a55e6cb066a0b4e20090391ccd2c2` | qzs3_range_joint_r258 (0.26) | recomputed text `0.2587078229986317`, `236516` bytes | external CUDA text; local static intake only |
| 93 | open | `887cccaccb376629829982660a1cf1ed06945bfc` | flatpup | promoted text `0.317848`, `284396` bytes | external CUDA text |
| 94 | open | `ae765553e9dc4bb12a93f4150a41f73ca3d9af16` | optimization_qpose_josema | recomputed MPS text `0.33425141289817706`, `277087` bytes | invalid MPS text; local static intake only |
| 95 | open | `9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9` | hnerv_muon submission (0.20) | recomputed body `0.1987048012202245`, `178417` bytes | external CPU-workflow text; local static ZIP intake only |

PR95 local static intake:

- Downloaded archive:
  `experiments/results/final_pr_stack_synthesis_20260504/pr95_archive.zip`
- Archive bytes/SHA-256:
  `178417`,
  `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- ZIP members: one stored member `0.bin`, `178309` bytes, SHA-256
  `4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4`
- Local action: static ZIP/file intake only. The HNeRV runtime was not executed.

## Stackability Matrix

| Surface | Class | Stack Verdict | Required Gates |
| --- | --- | --- | --- |
| `PR85_STBM1BR` mask recode | safe/lossless/pure-rate | Already confirmed A++; mask pixels match PR85 and components are flat at JSON precision | Keep runtime tree and archive SHA custody; any future comparison must use exact CUDA JSON |
| `STBM1BR_RMB1` randmulti recode | safe/lossless/pure-rate candidate | Best local PR85-family built stack: `229480` bytes, SHA `f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774`, `-276` bytes vs frontier | Fresh SHA check, pre-submission compliance, Level-2 claim, exact CUDA auth eval |
| PR92 `RMB1`/`RSB1` randmulti | contract-coupled | Raw PR92 is `+6760` bytes vs STBM because it keeps QMA9 mask; pure in-bundle RMB1 is useful only when side-info is not needed | Decode/reencode row parity, no hidden side-info dependency, runtime output parity, exact CUDA |
| PR91 `HPM1` mask | contract-coupled and currently invalid locally | Highest PR85-family rate signal, but local T4/L40S replay fails before score with HPAC entropy mismatch | Full submitted-token decode, byte-exact reencode or reviewed source-contract proof, runtime parity, exact CUDA |
| PR95 HNeRV | whole-stack external replacement | Not stream-stackable with PR85/STBM. It could become the public replacement anchor if exact CUDA replay reproduces the report | Strict archive/source custody, dependency closure, runtime tree hash, exact CUDA 600-sample JSON |
| PR85 non-mask pure repack | safe/lossless but exhausted | Local screens found no byte-negative candidate; strict one-member ZIP is already minimal | Reopen only with byte-positive decoded-semantic parity manifest |
| QH0/QFQ4 model recode | contract-coupled | Potential model bytes, but tensor parity currently fails; numeric closeness is not enough | Bit-exact tensor parity, model-loader runtime path, output parity, exact CUDA |
| QRGB/final-bias/post actions | component-benefit coupled | Current PR85 QRGB singletons are exact negatives; PR89/PR92 side channels need component benefit above byte cost | Non-noop action mapping, component-response evidence, exact stacked eval |
| Randmulti deletion/waterfill | exact-negative guardrail | `waterfill_top001` saved bytes but worsened score; deletion routes are not candidates | Do not dispatch measured deletion policies; use as signed training labels |
| PR94 qpose/tile actions | invalid/MPS-only and contract-coupled | MPS report is not promotion evidence; qpose velocity/tile actions are not isolated for PR85/STBM | CUDA replay only after archive/runtime custody, pose-output parity, and component break-even proof |
| PR87/PR70 tiny archive/source payload | invalid | Source-embedded payload loophole; not contest-faithful | Forensic guardrail only |

## Top Five Final Experiments

1. `canonical_cuda_replay_pr95_hnerv_muon`

   Expected if report reproduces: `0.1987048012202245`, delta
   `-0.05498530907375338` versus confirmed STBM.

   Runtime: PR95 `submissions/hnerv_muon` plus downloaded `0.bin` archive
   through canonical `experiments/contest_auth_eval.py --device cuda`.

   Gate: strict ZIP/source provenance, runtime tree SHA, Level-2 claim,
   exact CUDA JSON with `600` samples. Risk is high because the current score is
   an external CPU-workflow report, not our CUDA custody. Stop on any
   inflate/eval failure, non-600 sample count, source-side payload issue, or
   exact score not below `0.25369011029397787`.

2. `exact_cuda_pr85_stbm1br_rmb1`

   Expected if components stay flat: `0.25350633322291616`, delta
   `-0.0001837770710617193`.

   Runtime: STBM replay runtime with RMB1 randmulti decode. Gate: the local
   candidate/pre-submission gate is ready, but a fresh Level-2 claim and exact
   CUDA auth eval are still mandatory. Risk is low but real because randmulti is
   scorer-sensitive. Stop if components move or exact score is not lower.

3. `recover_pr91_hpm1_contract_then_hpm1_rmb1`

   Expected if components stay flat after recovery: `0.24861093819956195`,
   delta `-0.005079172094415922`.

   Runtime: PR91/PR86 HPAC HPM1 decoder plus native `hpac-codec` parity and a
   STBM-family archive runtime. Gate: full HPM1 submitted-token decode,
   byte-exact reencode or reviewed source-contract proof, runtime parity, then
   Level-2 claim and exact CUDA. Stop if frame-0/full-stream HPM1 cannot decode
   under pinned dependencies, prefix bytes miss break-even, or decoded-mask
   parity fails.

4. `pr85_jfg_qfq4_style_model_recode`

   Expected if bit-exact: `0.25325130924387035`, delta
   `-0.0004388010501075109`.

   Runtime: narrow PR85-family QFQ4/QH0 model loader with output parity
   preflight. Gate: bit-exact tensor parity, deterministic byte-positive
   archive, runtime output parity, Level-2 claim, exact CUDA. Stop if any model
   tensor cannot be bit-exact or model segment bytes are not positive.

5. `pr92_rsb1_qrgb_action_component_benefit_profile`

   Expected: no score target until component benefit exceeds the `+188` byte
   break-even, `0.00012518148318696823` score.

   Runtime: local RSB1/RMB1 action decoder and PR85/STBM action mapping. Gate:
   non-noop action mapping, exact source/candidate SHA custody, component
   response evidence, Level-2 claim, exact CUDA. Stop if no component-benefit
   path clears break-even or any action is not consumed by inflate.

## Score Bands

- Absolute best confirmed: `0.25369011029397787`
  (`confirmed_A++`, `PR85_STBM1BR`).
- Best plausible current public candidate if it passes canonical CUDA:
  `0.1987048012202245` (`external_prediction`, PR95 HNeRV).
- Best plausible local PR85-family built stack if it passes exact CUDA:
  `0.25350633322291616` (`prediction`, `PR85_STBM1BR_RMB1`).
- Aggressive theoretical band:
  - absolute public replacement band: `0.195-0.199` external/prediction if
    PR95 verifies and admits small entropy/payload refinements;
  - PR85-family stack band: `0.248-0.249` prediction if HPM1 contract, RMB1,
    and QFQ4-style model recode all pass.

No number in this section promotes a new score without exact CUDA evidence.

## Sources And Commands

Primary source artifacts:

- GitHub PR list:
  `experiments/results/final_pr_stack_synthesis_20260504/github_pulls_latest_raw.json`
- GitHub comments:
  `experiments/results/final_pr_stack_synthesis_20260504/pr{85,90,91,92,93,94,95}_issue_comments.json`
- PR95 static archive:
  `experiments/results/final_pr_stack_synthesis_20260504/pr95_archive.zip`
- Machine-readable synthesis:
  `experiments/results/final_pr_stack_synthesis_20260504/summary.json`
- Confirmed exact frontier:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.json`
- Local stack/readiness inputs:
  `experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/candidate_summary.json`,
  `experiments/results/public_pr92_intake_20260504_codex/public_frontier_intake_profile.json`,
  `experiments/results/public_pr94_qpose_intake_20260504_codex/profile_pr94_qpose_intake.json`,
  `experiments/results/pr85_full_stack_opportunity_matrix_20260504_orchestrator_final6/pr85_full_stack_opportunity_matrix.json`.

Commands run:

```bash
git status --short --branch
git remote -v
curl -fsSL 'https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls?state=all&sort=updated&direction=desc&per_page=100' -o experiments/results/final_pr_stack_synthesis_20260504/github_pulls_latest_raw.json
for n in 85 90 91 92 93 94 95; do curl -fsSL "https://api.github.com/repos/commaai/comma_video_compression_challenge/issues/$n/comments?per_page=100" -o "experiments/results/final_pr_stack_synthesis_20260504/pr${n}_issue_comments.json"; done
jq -r '.[] | [.number,.state,.title,.head.sha,.user.login,.updated_at] | @tsv' experiments/results/final_pr_stack_synthesis_20260504/github_pulls_latest_raw.json
curl -fL "$(cat experiments/results/final_pr_stack_synthesis_20260504/pr95_archive_url.txt)" -o experiments/results/final_pr_stack_synthesis_20260504/pr95_archive.zip
unzip -l experiments/results/final_pr_stack_synthesis_20260504/pr95_archive.zip
zipinfo -v experiments/results/final_pr_stack_synthesis_20260504/pr95_archive.zip
curl -fsSL 'https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls/95/files?per_page=100' -o experiments/results/final_pr_stack_synthesis_20260504/pr95_files.json
curl -fsSL 'https://github.com/commaai/comma_video_compression_challenge/pull/95.patch' -o experiments/results/final_pr_stack_synthesis_20260504/pr95.patch
```

Verification after writing this artifact:

```bash
jq empty experiments/results/final_pr_stack_synthesis_20260504/summary.json
git diff --check -- .omx/research/final_pr_stack_synthesis_20260504_codex.md experiments/results/final_pr_stack_synthesis_20260504/summary.json
```
