# final submission notes

> Superseded archival note, 2026-05-10: this is a May 4 PR100 packet snapshot,
> not current submission authorization. Refresh archive/runtime custody,
> strict compliance, exact CUDA, exact CPU if rank/shipment language is used,
> and operator policy before treating any path below as current.

## submission posture

- Current score champion is the contest-faithful PR100 HNeRV-LC-v2 adapter exact
  replay.
- Submit the PR100 adapter packet from
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`.
- Cite `0.22826947142244708` as the exact local Tesla T4 A++ score, not the
  rounded `0.23` display.
- Attribute public-source material to PR100, while making clear that the score
  claim comes from local exact auth eval of the exact archive plus adapter
  runtime tree.
- Keep PR96, PR95 body/CPU score, and PR91/HPM1 in external context until
  local exact CUDA eval or replay parity exists for each exact archive. PR99 is
  exact A++ but superseded by PR100.

## score block

- archive bytes: `178981`
- archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- SegNet: `0.00067623`
- PoseNet: `0.00017198`
- recomputed score: `0.22826947142244708`
- samples: `600`
- hardware: Tesla T4
- runtime tree SHA-256:
  `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- score artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json`

## concise framing

The current score champion is the public PR100 HNeRV-LC-v2 archive evaluated
through a strict contest-signature adapter. The adapter preserves the exact
archive bytes and records runtime-tree custody; the score claim is for this
archive/runtime pair, not for the rounded public display score.

## public-context wording

PR100 is exact A++ only through the adapter adjudicated JSON listed above.
PR99 is independently exact A++ at `0.2297226895103603` but remains
superseded by PR100. PR98/PR107 is independently exact A++ at
`0.22933111465960354` and is also superseded. PR96 remains external until local
exact replay lands. PR95 stem-permutation is a superseded exact predecessor at
`0.23089404465634825`. PR91/HPM1 remains useful source anatomy, but local
canonical replay fails before score in the HPM1 entropy decoder.

## final gate

The strict gate has passed on the exact PR100 packet with no
failed or warning checks. Re-run strict public-release hygiene plus the
pre-submission compliance gate if any upload file, report text, runtime source,
or public-source reference changes before submission.
