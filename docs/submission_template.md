# Submission PR Template

Submit to: https://github.com/commaai/comma_video_compression_challenge/pulls

The upstream PR template currently asks for:

- submission name
- a download link to zipped `archive.zip`
- `report.txt`
- whether inflation requires GPU
- whether a compression script is included and should be merged
- additional comments

Use this Apogee template for the current contest-faithful packet. Replace only
the placeholder URL and refresh the score block from the final exact T4
`report.txt`. The current internal exact frontier is the PR100 HNeRV-LC-v2
adapter replay, prepared as an Apogee follow-up packet after PR #107:
`0.22826947142244708` recomputed score, `178981` archive bytes, archive
SHA-256 `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`.
Public PR100 source attribution is required. Public body scores remain
external unless local exact CUDA eval validates the exact archive bytes; the
PR100 archive now has that local exact T4 validation.

````markdown
# submission name: Apogee

# upload zipped `archive.zip`

Download: ${APOGEE_ARCHIVE_ZIP_URL}

# report.txt

```
PASTE FINAL EXACT T4/EQUIVALENT report.txt HERE.
```

# does your submission require gpu for evaluation (inflation)?

Yes. The submission is evaluated through the canonical
`archive.zip -> inflate.sh -> upstream/evaluate.py` path and should use a
T4-equivalent CUDA runner for the final score claim.

# did you include the compression script? and want it to be merged?

Compression/reproduction scripts are included as supporting code where practical.
The score-bearing archive is self-contained: every score-affecting runtime
artifact required by inflation is charged inside `archive.zip` or fixed contest
code. No scorer patches, host-local sidecars, network downloads, hidden payloads,
or script-side payload movement are used.

# additional comments

Apogee is a contest-faithful, scorer-aligned compression system. The submitted
archive records exact bytes, SHA-256, component distances, sample count,
recomputed score, CUDA/T4 hardware evidence, runtime custody, and attribution
for any external public payload segment used inside the charged archive.

Evidence boundary:

- Exact score evidence: local T4/equivalent CUDA auth eval on the submitted
  archive bytes through `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- External context: public PR titles, bodies, comments, and self-reported
  scores until replayed through the exact local custody path.
- Quantizr/late-meta note: public comments and PR bodies are useful process
  context for the challenge's shifting meta, but the score authority remains
  exact local CUDA replay on exact bytes.
- PR100 note: the local score claim is for the exact PR100 archive bytes plus
  the Apogee adapter runtime tree.
- PR96 note: this remains public/static context until exact T4 replay lands
  locally.

Public supplement:

- Lightning.ai notebook: ${LIGHTNING_SUPPLEMENT_URL}
- Cloudflare Pages writeup/site: ${CLOUDFLARE_PAGES_URL}
- Release manifest: ${APOGEE_RELEASE_MANIFEST}
````

## Final Gate

- [ ] `archive.zip` bytes and SHA match the exact auth-eval artifact.
- [ ] `report.txt` is copied from exact T4/equivalent CUDA evaluation.
- [ ] `scripts/pre_submission_compliance_check.py` passes on the exact packet
      with expected archive SHA/bytes, runtime tree SHA, auth-eval JSON, and
      public-source references.
- [ ] `inflate.sh` runs end-to-end on a fresh T4-equivalent machine in under
      30 minutes.
- [ ] Output frame shape/sample count matches the contest evaluator contract.
- [ ] All neural weights, masks, poses, tables, and score-affecting payloads are
      inside `archive.zip` or fixed contest code.
- [ ] No external downloads, network access, scorer patches, host-local files,
      hidden sidecars, malformed ZIP reliance, or script-embedded payload moves.
- [ ] Public supplement URLs are placeholders until intentionally published via
      a sanitized release manifest.
- [ ] Public release hygiene scan passes strict mode on the exact PR body,
      report docs, notebook, and Cloudflare site bundle.

Recommended gate command for the current frontier:

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
