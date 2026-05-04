# PR95 Submission Release Report Hardening - 2026-05-04

Scope: release/report/writeup hardening only. No exact-eval launcher, PR95/PR96
candidate builder, remote job, dispatch claim, scorer, or inflate runtime was
changed.

## Current Packet

Final packet:
`experiments/results/submission_packet_pr95_repack_20260504/apogee_pr95_repack`

Score-bearing archive:

- score: `0.23091954465634829`
- evidence: `A++` exact Tesla T4 CUDA, `600` samples
- archive bytes: `178321`
- archive SHA-256:
  `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`
- SegNet/PoseNet: `0.00070728` / `0.00017185`
- runtime tree SHA-256:
  `a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7`
- score artifact:
  `experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json`
- strict packet gate:
  `experiments/results/submission_packet_pr95_repack_20260504/pre_submission_compliance.json`
  with `status=passed`, `failed_checks=[]`, and `80` checks.

## Release Surface Fixes

The report/writeup surfaces still described the previous PR85/STBM frontier.
This pass updated the public-facing release surfaces to the PR95 conservative
repack, while preserving PR85/STBM, PR85, PR84, PR81, and C-067 as superseded
exact history.

Updated surfaces:

- `reports/latest.md`
- `reports/writeup_working.md`
- `docs/paper/04_results.md`
- `docs/submission_template.md`
- `reports/graphs/final_submission_notes.md`
- `reports/graphs/final_writeup_draft.md`
- `reports/graphs/release_checklist.md`
- `reports/graphs/evidence_index.md`

The release command snippets now use the real
`scripts/pre_submission_compliance_check.py` argparse flag `--source-prs PR95`
instead of the stale public-PR-refs spelling.

## Guardrail Added

Added a focused regression test in
`src/tac/tests/test_pre_submission_compliance_check.py` so release docs do not
reintroduce the stale public-PR-refs flag. This is an engineering guardrail,
not score evidence.

## Verification

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/submission_packet_pr95_repack_20260504/apogee_pr95_repack \
  --archive experiments/results/submission_packet_pr95_repack_20260504/apogee_pr95_repack/archive.zip \
  --auth-eval-json experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json \
  --contest-final --expect-single-member 0.bin \
  --expected-archive-sha256 2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b \
  --expected-archive-size-bytes 178321 --expected-samples 600 \
  --expected-runtime-tree-sha256 a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7 \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id pr95_hnerv_muon_repacked_t4_replay_fix2 \
  --expected-job-id exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z \
  --source-prs PR95
```

Result: `status=passed`, `failed_checks=[]`, `warning_checks=[]`, `80` checks.

```bash
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
```

Result: `22 passed in 0.63s`.

```bash
check_public_release_hygiene(strict=True, scan_paths=[...release docs...])
```

Result: `0` violations across `8` public files.

```bash
git diff --check -- docs/submission_template.md docs/paper/04_results.md \
  reports/latest.md reports/writeup_working.md reports/graphs/*.md \
  src/tac/tests/test_pre_submission_compliance_check.py \
  .omx/research/pr95_submission_release_report_hardening_20260504_codex.md
```

Result: passed for the scoped touched files.

## Residual Release Boundaries

- PR95 public body/CPU score remains external; exact T4 replay scored the public
  archive at `0.23098329465634826`.
- PR96 remains external/static until a local exact CUDA replay artifact exists.
- PR91/HPM1 remains invalid pre-score replay evidence until decode/reencode
  parity is fixed.
- Final public supplement URLs must remain placeholders until a sanitized
  release manifest intentionally publishes them.
