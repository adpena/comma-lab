# judges one-pager

## headline (current frontier - PR100 Apogee follow-up)

- Best exact contest-CUDA score: **`0.22826947142244708`** `[A++]`
- Archive: `178981` bytes,
  SHA-256 `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- Packet:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`
- Runtime tree SHA-256:
  `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- Recipe: public PR100 HNeRV-LC-v2 source attribution plus contest-signature
  Apogee adapter runtime, evaluated on exact Tesla T4 CUDA custody
- PoseNet `0.00017198` / SegNet `0.00067623`
- Public source PR:
  `https://github.com/commaai/comma_video_compression_challenge/pull/100`

## historical arc

- We got nerd-sniped by this challenge. The work became a full-stack
  engineering/research sprint across compression, neural representations,
  scorer-aware optimization, archive forensics, and reproducibility tooling.
- Era 1 codec/postfilter floor: `1.73`
- Era 2 neural renderer controls: `0.90` CUDA baseline, `1.15` Lane A pose
  TTO, `1.05` Lane G v3 KL-distill/TTO
- Quantizr/QZS3 public-floor reproduction: C-067 exact T4
  `0.31561703078448233`
- Public semantic-bundle era: PR85 `0.25806611029397786`,
  PR85+STBM/RMB1 `0.2535063602939779`, PR95 stemperm
  `0.23089404465634825`, PR98/Apogee `0.22933111465960354`,
  PR100 follow-up `0.22826947142244708`

## why this is the score authority

- **exact packet**:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`
- **exact eval JSON**:
  `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json`
- **supersession**: PR99 exact adapter replay scored `0.2297226895103603`;
  PR95 stemperm scored `0.23089404465634825`
- **boundary**: PR96, public body scores, and PR91/HPM1 remain external until
  exact CUDA replay lands; PR100 has local exact T4 replay

## method contribution

- contest-CUDA `archive.zip -> inflate.sh -> upstream/evaluate.py` on exact
  archive bytes
- archive bytes/SHA and runtime tree SHA recorded with the score claim
- strict pre-submission compliance JSON retained beside the packet
- public-source attribution and score authority are separated
- no scorer patches, sidecars, hidden payloads, or script-side payload movement
- meta-Lagrangian atom pricing over bytes, SegNet, PoseNet, runtime, custody,
  and compliance risk
- deterministic packet generation, dispatch claims, exact JSON adjudication,
  wrapper-contract adapters, and public-release hygiene
- month-long research ledger plus AI-assisted Grand Council/Skunkworks Council
  adversarial review, used to generate and audit hypotheses before exact eval

## visual supplement

The public bundle includes generated comparison GIFs, including
`comma_comparison.gif` and `comma_comparison_full.gif`, as the visual companion
to the exact score tables. These are illustrative and judge-facing; they do not
create score authority.

## public context

- Quantizr PR #53/#55 anticipated the late meta-game and the sub-0.30
  architecture/compute direction; this is external context, not score evidence.
- PR100 public body score is external/static context; the ranked PR100 value is
  the local exact T4 replay at `0.22826947142244708`.
- PR91/HPM1 is source anatomy and failed before score in local canonical replay.
- PR87/PR70-style source-embedded payloads are compliance lessons, not valid
  floor evidence.
- Unlimited-compute or inflate-time optimization probes are research signals
  unless every score-affecting bit is charged and exact CUDA eval validates the
  final archive under the contest budget.
