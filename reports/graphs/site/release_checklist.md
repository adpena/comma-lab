# release checklist

## score lane

- [x] current exact frontier recorded: PR100 HNeRV-LC-v2 adapter replay,
      `0.22826947142244708`.
- [x] archive bytes/SHA recorded:
      `178981`,
      `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`.
- [x] exact Tesla T4 auth-eval JSON retained:
      `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json`.
- [x] runtime tree SHA recorded:
      `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`.
- [x] PR98 public archive/runtime attribution recorded separately from local
      exact score authority.
- [x] PR95 predecessor, PR99, PR100, PR91/HPM1, and public body scores
      classified by exact evidence status.
- [x] Built the final public packet directory with exactly the upload files:
      `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`.
- [x] Ran `scripts/pre_submission_compliance_check.py` on that exact packet.
- [x] Preserved the gate JSON/report beside the packet.

## writeup lane

- [x] Results section updated to PR100 adapter frontier.
- [x] Public PR claims separated from exact local evidence.
- [x] Quantizr PR #53/#55 meta comments incorporated as external process
      context.
- [x] Metalagrangian atom-pricing method documented.
- [x] Renderer-era scores in the 0.90/1.15/1.05 band restored to the trajectory
      as scientific context.
- [x] Harness engineering called out as a first-class contribution:
      deterministic packets, runtime tree hashes, dispatch claims, exact JSON
      adjudication, and release hygiene.
- [x] Visual supplement GIFs referenced for the public report.
- [x] Nerd-sniped challenge context included as non-score narrative.
- [x] PR91 anatomy and fail-closed status documented.
- [x] `range_mask_codec.cpp` scoped to QMA6-QMA9, not HPM1.
- [x] Pre-submission compliance gate documented.
- [x] Final pass over generated site bundle after PR100 packet handoff.
- [x] Strict public release hygiene scan on the exact PR body, notebook, and
      site bundle.

## public surface lane

- [ ] Replace placeholder URLs only from a sanitized release manifest.
- [ ] Keep private provider URLs, account metadata, raw state files, and local
      absolute paths out of GitHub/docs/site surfaces.
- [ ] Verify site links point to repo-relative artifacts or intentional public
      URLs only.
- [ ] Keep `tac` research ledgers and raw provider state out of the public
      release unless explicitly sanitized into a release artifact.
