# submission packet

## notebook surface

- interactive lab notebook: `reports/graphs/lab_notebook.md`
- methodology: `docs/lab_methodology.md`
- glossary: `reports/graphs/glossary.md`
- visual comparison GIF: `reports/graphs/site/comma_comparison.gif`
- full visual comparison GIF: `reports/graphs/site/comma_comparison_full.gif`

## score target

- Current exact contest-CUDA score: **`0.22826947142244708`** `[A++]`
- Archive bytes: `178981`
- Archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- Runtime tree SHA-256:
  `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- Submission packet:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`
- Promoted evidence:
  `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json`
- Public PR:
  `https://github.com/commaai/comma_video_compression_challenge/pull/107`
- Public release:
  `${APOGEE_RELEASE_MANIFEST}`

## evidence

- exact Tesla T4 CUDA auth eval on the final archive bytes
- `archive.zip -> inflate.sh -> upstream/evaluate.py` path preserved
- strict packet compliance JSON retained beside the packet
- PR100 public source attribution remains explicit
- PR98 public body score, PR95 body/CPU score, PR96, and PR91/HPM1 remain
  external until exact replay or parity exists

## path summary

- x265 honest floor reached `3.25`
- repaired AV1 path reached `2.20`
- one-axis AV1 tuning reached `2.18` then `2.12`
- encoder-side `sharpness=1` reached `2.08`
- a tiny learned int8 post-filter reached `2.05`
- longer-horizon QAT+EMA training improved that to `1.99`
- the wider h32 long-500 QAT+EMA branch reached `1.95`
- extending the h16 branch to 1000 epochs established `1.92`
- extending the h32 branch to 1000 epochs established `1.85`
- a bounded ensemble of the `1.85` floor and the best Monte Carlo refinement established `1.84`
- scaling the same long-horizon QAT+EMA recipe to h64 established the Era 1
  floor at `1.73`
- the first reproducible-from-saved-artifacts neural renderer baseline reached
  `0.90` on contest CUDA
- Lane A pose TTO from baseline poses reached `1.15`; this looks worse than
  `0.90` as a total score because it paid a large pose-payload rate cost, but
  it was a decisive PoseNet-basin control result
- Lane G v3 KL-distill weight `0.002` plus pose TTO retry reached `1.05`, with
  a Modal T4 repro around `1.04`
- Quantizr/JointFrameGenerator reproduction and QZS3/QP1 work produced the
  first sub-0.4 exact public-floor basin, including the C-067 PR67-mask fixed
  slice archive at `0.31561703078448233`
- PR85 semantic-bundle replay established `0.25806611029397786`
- PR85+STBM/RMB1 established `0.2535063602939779`
- PR95 HNeRV/Muon exact replay established the sub-`0.231` semantic-bundle
  anchor
- PR95 stem-permutation repack established `0.23089404465634825`
- PR100 adapter replay established the current exact frontier at
  **`0.22826947142244708`**

## meta-lagrangian summary

Apogee treats each archive component as a charged atom. The optimizer prices
that atom by bytes, SegNet movement, PoseNet movement, runtime, custody risk,
and compliance risk. Public PR anatomy and Quantizr's late-meta comments guide
the proposal distribution; exact CUDA eval is the only promotion rule.

The human story matters too: like Quantizr wrote in PR #55, this challenge
nerd-sniped us. The report should say that plainly while keeping every score
claim evidence-gated.

The research process used AI-assisted councils as adversarial review tools:
Grand Council and Skunkworks Council sessions assigned expert roles to surface
math, compression, steganalysis, hardware, openpilot, and compliance objections.
Those sessions produced hypotheses and reviews; exact CUDA artifacts decided
what entered the score ledger.

The same atom-pricing explains the nonmonotone trajectory: a candidate can
lower one distortion term while spending enough bytes to lose total score, and
it can still be scientifically valuable if it reveals a reusable basin or
constraint. The 0.90 -> 1.15 -> 1.05 renderer sequence is a concrete example:
pose optimization revealed control over PoseNet, KL-distill recovered part of
the SegNet/PoseNet tradeoff, and later HNeRV-style archives changed the
representation family entirely.

Unlimited-compute and inflate-time-optimization experiments are recorded as
research probes, not contest-valid score claims unless every score-affecting
bit is charged and the final `inflate.sh` path stays within the contest budget.
They still matter because they reveal gradients, hard pairs, low-dimensional
PoseNet subspaces, and useful correction atoms for the charged archive
compiler.

## active follow-on

- current upload packet is `apogee_pr100_hnerv_lc_v2_adapter`
- PR100 exact T4 replay has landed and is the current score authority; any
  follow-up should cite the adjudicated JSON and sanitized release manifest
- final public URLs stay placeholders until published through a sanitized
  release manifest
- no public PR, body score, or leaderboard title should rank unless exact CUDA
  replay lands
