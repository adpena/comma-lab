# final writeup draft

## thesis

This writeup should present Apogee as a contest-faithful archive compiler for a
single fixed driving video and fixed neural evaluator. The score-bearing output
is not a proxy-score claim, public PR body claim, or leaderboard scrape. It is
exact CUDA custody for exact archive bytes through the canonical contest path.

We got nerd-sniped by the challenge in the best possible way: the problem kept
opening into deeper systems questions about what a video is when the consumer is
a fixed neural scorer, not a human viewer.

The current confirmed champion packet is the PR100 HNeRV-LC-v2 adapter replay,
built as an Apogee follow-up packet after Apogee PR #107. PR #107 remains the
earlier public Apogee submission; PR100 exact replay is now the lower local
A++ score-bearing packet.

The score-bearing claim is:

- `0.22826947142244708` exact Tesla T4 A++
- `178981` archive bytes
- archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- runtime tree SHA-256:
  `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- SegNet `0.00067623`
- PoseNet `0.00017198`
- score JSON:
  `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json`
- packet:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`
- public source PR:
  `https://github.com/commaai/comma_video_compression_challenge/pull/100`

Every public-facing section should make the same distinction: exact local CUDA
evidence can rank; public PR claims, comments, and static anatomy can motivate,
attribute, and explain the meta-game, but cannot rank until replayed.

## result arc

The narrative arc:

1. Early codec/postfilter and renderer work built the measurement discipline.
2. QZS3/QP1 and QMA9 public-floor work shifted the frontier to semantic
   sufficient statistics rather than perceptual reconstruction.
3. PR85 established a stronger public semantic-bundle anchor with exact T4
   score `0.25806611029397786`.
4. PR85+STBM/RMB1 pushed that lineage to `0.2535063602939779`.
5. The renderer era before public semantic bundles mattered: the 0.90 CUDA
   baseline, 1.15 Lane A pose-TTO control result, and 1.05 Lane G v3
   KL-distill/TTO result established scorer-aligned neural rendering and the
   exact-eval harness discipline.
6. Quantizr/JointFrameGenerator reproduction, QZS3/QP1 packer work, and PR67
   mask-source attribution produced the first sub-0.4 exact public-floor basin,
   including C-067 at `0.31561703078448233`.
7. PR95 HNeRV/Muon showed that route-specialized neural video representations
   had displaced hand-packed semantic streams as the live meta.
8. PR95 repacking established a local exact predecessor at
   `0.23089404465634825`.
9. PR98 exact adapter replay improved the SegNet basin; PR100 HNeRV-LC-v2 then
   became the current local exact A++ frontier at `0.22826947142244708`.

## contest meta

The challenge meta changed late. Public submissions were not a stable target;
they were part of a competitive information game. Quantizr anticipated this in
PR #53, noting that open submissions incentivize people to hold competitive
work until the end so others cannot cheaply tune past it. Quantizr's PR #55
then made two technically important public claims: sub-0.30 looked reachable by
architecture/compute search, and neural mask encoding was a plausible next
lever beyond ffmpeg-style mask video.

Those comments are external process evidence, not score authority. They are
useful because they explain why the final system had to run two coupled loops:

- monitor and deconstruct late public archives, wrappers, payloads, and PR
  prose;
- simultaneously run exact replay, adapter repair, repacking, ablation, and
  optimization searches under strict custody.

The writeup should be candid that this became a meta-optimization problem:
leaderboard intelligence changed the active representation family, while exact
CUDA replay decided which public claims were usable.

It should also be candid that the project state is distributed across git
history, `.omx/research` ledgers, exact-eval artifact directories, generated
site files, and public PR/release surfaces. The public release bundle should
therefore be a curated evidence index, not a raw dump of operator state.

## research process

Apogee was built over a month of AI-assisted research and engineering using
Claude Code, Codex, public PR intake, local artifact forensics, remote CUDA
evals, and a large internal ledger system. The "Grand Council" and
"Skunkworks Council" sessions were a deliberate prompting pattern: assign
different expert roles to pressure-test architecture, math, compliance,
steganalysis, openpilot priors, systems hardening, and contrarian failure
modes before spending GPU or promoting a claim.

That process did not replace evidence. It generated proposals, objections,
failure classifications, and next experiments. Exact CUDA auth eval and
deterministic archive custody remained the promotion gate. The online writeup
should include a separate tooling/process appendix about this AI-assisted
research workflow because it is part of what made the system reproducible under
deadline pressure.

## metalagrangian method

The metalagrangian view treats the whole submission as a constrained program:

```text
minimize    S(A, R)

S = 100 * D_seg(A, R)
  + sqrt(10 * D_pose(A, R))
  + 25 * |A| / 37,545,489
```

where `A` is the charged archive and `R` is the deterministic inflate runtime.
Every stream, payload, decoder weight, latent, side channel, postprocess,
wrapper, and packing rule is an optimization atom with a cost vector:

```text
atom_i -> (delta_bytes_i,
           delta_seg_i,
           delta_pose_i,
           delta_runtime_i,
           legality_i,
           reproducibility_i)
```

The practical objective is a constrained Lagrangian:

```text
L = 100 * D_seg
  + sqrt(10 * D_pose)
  + lambda_rate * bytes
  + lambda_time * inflate_time
  + lambda_custody * custody_risk
  + lambda_compliance * compliance_risk
```

The multipliers are not fixed constants; they are feedback variables. When the
frontier sits in a PoseNet cliff, `lambda_pose` effectively rises. When public
HNeRV archives reveal a smaller sufficient-statistic program, the representation
family changes and previous local mask-packer polish becomes lower EV. When a
runtime wrapper fails before score, `lambda_custody` becomes infinite until the
adapter contract is fixed.

This is why negative results mattered. A bad exact eval is a gradient-like
constraint sample: it tells the optimizer which basin, byte region, runtime
contract, or geometry transform is outside the trust region. The final Apogee
packet is one point in that constrained search, with exact CUDA eval as the
only optimizer check that can promote a point to evidence.

## method

The method is a typed archive compiler:

- Parse public or internal archives into typed streams: neural decoder,
  latents, masks, poses, correction payloads, layout, and action channels.
- Prove local decode/output parity before treating any recode as a candidate.
- Build deterministic archives with charged side information only.
- Evaluate only through `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- Preserve runtime tree custody because identical archive bytes can score
  differently under different inflate code.

The PR100 adapter result is not a pure-rate result. It preserves the public
PR100 HNeRV-LC-v2 archive bytes while repairing the contest wrapper contract
and validating the archive under exact Tesla T4 CUDA. The measured score is
lower than PR98 and PR95 stem-permutation because the decoded output moves to a
better SegNet/PoseNet tradeoff despite slightly higher charged bytes.

The harness is part of the method. Deterministic archive construction,
runtime-tree hashing, dispatch claims, exact JSON adjudication, wrapper
contract adapters, fail-closed archive validators, and public-release hygiene
checks are what made late public archive deconstruction useful instead of
unreproducible copying.

The public supplement should include the generated matplotlib/GIF visual
artifacts, especially `comma_comparison.gif` and `comma_comparison_full.gif`.
Those figures make the machine-perception nature of the result legible in the
same spirit as Quantizr's visual PR attachment, while the score table remains
the authority.

Unlimited-compute and inflate-time scorer-optimization experiments belong in
the paper as probes of the Yousfi-Fridrich landscape, not as contest-ranking
claims. They helped identify hard pairs, low-dimensional pose basins, and
correction atoms. They are contest-valid only after the learned artifact or
correction payload is charged inside `archive.zip`, inflate remains budgeted,
and exact CUDA auth eval passes on the final bytes.

## external context

PR100 publicly claims a lower HNeRV-LC-v2 score than PR98. Our exact T4 replay
validated the same archive bytes at `0.22826947142244708`, which supersedes the
PR98/PR107 packet locally even though the public PR body score itself remains
external context.

PR91 publicly self-reports `0.24879480490416128` at `222404` bytes. Local T4
and L40S replay failed before score in HPM1 entropy decode, so PR91 is an
external frontier hypothesis and source anatomy, not local exact evidence.

`range_mask_codec.cpp` in PR91 is live only for the QMA6-QMA9 fallback branch.
It does not decode HPM1 and does not fallback-rescue the observed HPM1 entropy
failure.

## compliance posture

The final release packet passed the strict pre-submission gate on:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter \
  --archive experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter/archive.zip \
  --auth-eval-json experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json \
  --contest-final \
  --expect-single-member 0.bin \
  --expected-archive-sha256 afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641 \
  --expected-archive-size-bytes 178981 \
  --expected-runtime-tree-sha256 ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832 \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id public_pr100_hnerv_lc_v2_t4_adapter_replay \
  --expected-job-id exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z \
  --source-prs PR100
```

This gate is a release-readiness check, not a scorer. It verifies byte closure,
deterministic ZIP custody, executable inflate, exact auth-eval match, runtime
tree match, public-source references, and sanitized public surfaces.

## paper claims

Allowed:

- "The PR100 Apogee follow-up packet is the current local exact A++ frontier at
  `0.22826947142244708`."
- "The score authority is the adjudicated exact auth-eval JSON on the submitted
  archive bytes and matching runtime tree."
- "Quantizr's PR #53/#55 comments correctly anticipated the late meta-game and
  the HNeRV/neural-mask direction, but they are external process context."
- "PR100 publicly claimed a lower score; local exact T4 replay validated the
  archive at `0.22826947142244708`."

Forbidden:

- "The PR98 public body score is the local exact score."
- "PR100 beats Apogee PR #107 because the public body score says so."
- "The PR91 public self-report is an exact score."
- "HPM1 is decoded by `range_mask_codec.cpp`."
- "Source-embedded payloads establish a valid compression floor."

## residual gaps

- PR100 follow-up PR/update using the strict packet if the upstream contest
  process accepts late follow-up submissions.
- Public release hygiene scan on the exact Cloudflare/site bundle and notebook.
- Final sanitized URLs for the public supplement.
- Consolidated OSS documentation for the minimal Apogee runtime and the larger
  `tac` research toolkit without publishing private custody state.
