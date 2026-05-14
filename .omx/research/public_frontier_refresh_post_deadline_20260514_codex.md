# Public Frontier Refresh - Post Deadline - 2026-05-14

## Scope

Fresh public-frontier check for the comma video compression challenge after the
May 2026 deadline window. Evidence is external/public and is not an internal
score claim unless exact local replay lands separately.

Commands / sources:

```bash
gh pr list --repo commaai/comma_video_compression_challenge --state all --limit 30 \
  --json number,title,state,author,createdAt,updatedAt,url,headRefOid

gh pr view 101 --repo commaai/comma_video_compression_challenge \
  --json number,title,state,author,createdAt,updatedAt,comments,reviews,url,headRefName,headRepositoryOwner,headRefOid

gh pr view 103 --repo commaai/comma_video_compression_challenge \
  --json number,title,state,author,createdAt,updatedAt,comments,reviews,url,headRefName,headRepositoryOwner,headRefOid

gh pr view 107 --repo commaai/comma_video_compression_challenge \
  --json number,title,state,author,createdAt,updatedAt,comments,reviews,url,headRefName,headRepositoryOwner,headRefOid

gh pr view 108 --repo commaai/comma_video_compression_challenge \
  --json number,title,state,author,createdAt,updatedAt,comments,reviews,url,headRefName,headRepositoryOwner,headRefOid
```

Web sources:

- official leaderboard: `https://comma.ai/leaderboard`
- PR 101: `https://github.com/commaai/comma_video_compression_challenge/pull/101`
- PR 103: `https://github.com/commaai/comma_video_compression_challenge/pull/103`
- PR 107: `https://github.com/commaai/comma_video_compression_challenge/pull/107`
- PR 108: `https://github.com/commaai/comma_video_compression_challenge/pull/108`

## Leaderboard State

As fetched on 2026-05-14, the official leaderboard still ranks the HNeRV cluster
at the top:

1. `#101` / `hnerv_ft_microcodec`: `0.193`
2. `#103` / `hnerv_lc_ac`: `0.195`
3. `#102` / `hnerv_lc_v2_scale095_rplus1`: `0.195`
4. `#100` / `hnerv_lc_v2`: `0.195`
5. `#98` / `hnerv_muon_finetuned_from_pr95`: `0.197`
6. `#105` / `kitchen_sink`: `0.198`
7. `#95` / `hnerv_muon`: `0.199`
8. `#96` / `rem2_HNeRV`: `0.206`
9. `#106` / `belt_and_suspenders`: `0.209`
10. `#97` / `vibe_coder_final_boss`: `0.229`
11. `#107` / `apogee`: `0.229`

No public leaderboard movement below `0.193` was observed in this refresh.

## Post-Deadline PR Signal

### PR 108

PR `#108` (`andimin01`) was created 2026-05-05 and updated / closed
2026-05-11. It is not listed on the leaderboard as a competitive or new
approach result. The maintainer closure comment establishes the post-deadline
submission bar:

- a future submission should be competitive: better than current #1, or
- innovative: a novel leaderboard-relevant idea with potential.

Classification:

- `post_deadline_governance_signal`
- `not_frontier_score_signal`
- `established_tricks_are_insufficient_for_new_submission`

Operational impact:

- Any future submission packet should explicitly explain either:
  - why it beats the current #1 score on the chosen public axis, or
  - what novel idea it contributes beyond the existing leaderboard methods.
- Packaging should keep public repo links, archive SHA, runtime SHA, exact
  artifact paths, and reproducibility commands up front. PR `#107` already did
  this well and received normal leaderboard recognition.

### PR 107

PR `#107` (`apogee`) has no new post-May-5 technical comments beyond the
maintainer congratulatory note and job/internship contact instruction. The
official workflow result remains:

- axis: CUDA
- score: `0.23` rounded
- recomputed score from PR body: `0.22933111465960354`
- archive bytes: `178392`
- avg PoseNet distortion: `0.00017394`
- avg SegNet distortion: `0.00068841`
- archive SHA-256: `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- runtime tree SHA-256: `0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0`

Important packaging precedent from PR `#107`:

- include release/archive URL and immutable archive SHA
- include runtime entrypoint and archive manifest
- include exact score recomputation and T4 hardware evidence
- include public OSS/research repo links when appropriate
- explain deterministic reproducibility and production boundary

### PR 101

PR `#101` has both CUDA and CPU workflow results in public comments:

- CUDA: rounded `0.23`, avg pose `0.00017103`, avg seg `0.00066304`,
  bytes `178258`.
- CPU rerun: rounded `0.19`, avg pose `0.00003286`, avg seg `0.00056023`,
  bytes `178258`.

Maintainer comment awarded `#1` prize on 2026-05-05.

Classification:

- `cpu_cuda_axis_split_material`
- `public_cpu_axis_advantage_confirmed`
- `do_not_convert_cpu_to_cuda`

Operational impact:

- Continue to treat `[contest-CPU]` and `[contest-CUDA]` as separate evidence
  spaces.
- For HNeRV-family submissions, CPU evaluation can materially improve both
  PoseNet and SegNet components; if the submission does not require GPU for
  inflation, exact CPU replay is a first-class axis, not an afterthought.

### PR 103

PR `#103` also has both CUDA and CPU workflow results and an extended public
discussion of CPU-vs-GPU adjudication:

- CUDA: rounded `0.23`, avg pose `0.00017198`, avg seg `0.00067623`,
  bytes `178223`.
- CPU rerun: rounded `0.19`, avg pose `0.00003443`, avg seg `0.00057654`,
  bytes `178223`.

Maintainer explanation: same hardware makes submissions comparable; CPU/GPU
differences grew important as solutions became closer and more optimized. PR
author argued the originally posted CPU-vs-GPU selection rule should govern
submissions that do not require GPU. Maintainer later awarded `#2`.

Classification:

- `axis_policy_dispute_public_signal`
- `cpu_cuda_gap_is_submission_specific`
- `future_packet_must_state_inflate_gpu_requirement_precisely`

Operational impact:

- Submission reports must explicitly answer whether GPU is required for
  inflation, and local exact eval should cover both axes when feasible.
- For internal ranking, public CPU advantage is real, but promotion language
  must remain axis-labeled.

## George / geohot Search

Searches for George Hotz / geohot-specific comments in this repository and on
the web did not surface a relevant public PR comment in this refresh.

Commands / searches:

```bash
gh api 'search/issues?q=repo:commaai/comma_video_compression_challenge+geohot+OR+george+OR+hotz'
```

and web searches for:

- `geohot "comma_video_compression_challenge"`
- `George Hotz "video compression challenge" comma.ai`
- `site:github.com/commaai/comma_video_compression_challenge geohot`

## Next Actions

1. Keep exact CPU replay paths alive for all HNeRV-family packets that do not
   require GPU for inflation.
2. For any future PR/update, include:
   - public repo link(s)
   - immutable archive URL
   - archive bytes and SHA-256
   - runtime tree SHA-256
   - exact eval artifact paths
   - CPU/CUDA axis labels
   - deterministic reproduction commands
   - short novelty/competitiveness explanation under PR `#108` guidelines.
3. Deconstruct PR `#101` and PR `#103` under both CPU and CUDA axes if their
   original runtimes/archives are not already fully replayed in local custody.
4. Do not resubmit established AV1/ROI/sharpening-only techniques unless they
   are either strictly competitive with the current #1 or combined into a novel
   score-relevant mechanism.
