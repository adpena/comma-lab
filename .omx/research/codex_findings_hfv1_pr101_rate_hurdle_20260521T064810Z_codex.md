# Codex Findings: HFV1 PR101 Rate Hurdle

- timestamp_utc: 2026-05-21T06:48:10Z
- lane: hfv1_pr101_adapter_rate_hurdle
- status: LANDED_RATE_HURDLE_AUDITOR_AND_ARTIFACT
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## What landed

New tool:

- `tools/audit_hfv1_pr101_rate_hurdle.py`

The previous readiness packet proved that three HFV1 PR101 adapter archives are
exact-eval plan-ready. This rate-hurdle audit answers the next dispatch
question: how much SegNet/PoseNet component improvement must each larger HFV1
archive deliver before it can beat the current FEC6/PR110 CPU-axis score?

This is pure rate arithmetic. It does not run eval and does not claim a score.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_hfv1_pr101_rate_hurdle.py \
  --readiness-json experiments/results/hfv1_pr101_exact_eval_readiness_20260521T064232Z/hfv1_pr101_exact_eval_readiness.json \
  --output-dir experiments/results/hfv1_pr101_rate_hurdle_smoke_20260521T0655Z \
  > /tmp/hfv1_rate_hurdle_smoke.json

git diff --check -- tools/audit_hfv1_pr101_rate_hurdle.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_hfv1_pr101_rate_hurdle.py \
  --readiness-json experiments/results/hfv1_pr101_exact_eval_readiness_20260521T064232Z/hfv1_pr101_exact_eval_readiness.json \
  --output-dir experiments/results/hfv1_pr101_rate_hurdle_20260521T064740Z \
  > experiments/results/hfv1_pr101_rate_hurdle_20260521T064740Z/stdout.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/audit_hfv1_pr101_rate_hurdle.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check \
  tools/audit_hfv1_pr101_rate_hurdle.py
```

Review tracker result:

- `tools/audit_hfv1_pr101_rate_hurdle.py`: NORMAL
- 16 entities reviewed
- 0 policy violations

## Audit artifacts

- JSON: `experiments/results/hfv1_pr101_rate_hurdle_20260521T064740Z/hfv1_pr101_rate_hurdle.json`
- Markdown: `experiments/results/hfv1_pr101_rate_hurdle_20260521T064740Z/hfv1_pr101_rate_hurdle.md`
- stdout mirror: `experiments/results/hfv1_pr101_rate_hurdle_20260521T064740Z/stdout.json`

SHA-256:

```text
5026124166a2790b3a16aed487ebc484d3d8ac410d4984312df3568e576b1fca  hfv1_pr101_rate_hurdle.json
f8a239687761c915fcbd681658fc98d9918e91bfec3f9c4bdf93817ef377d102  hfv1_pr101_rate_hurdle.md
5026124166a2790b3a16aed487ebc484d3d8ac410d4984312df3568e576b1fca  stdout.json
```

Byte counts:

```text
6257  hfv1_pr101_rate_hurdle.json
1630  hfv1_pr101_rate_hurdle.md
6257  stdout.json
```

## Result

Baseline used for rate arithmetic:

- baseline archive: `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/archive.zip`
- baseline bytes: 178,517
- baseline score: `0.192051`
- baseline rate term: `0.11886714273451066251927095689178532206625408447870`
- baseline component term: `0.07318385726548933748072904310821467793374591552130`

Each current HFV1 candidate archive is 202,649 bytes, which is +24,132 bytes
over FEC6. That adds a rate penalty of:

```text
25 * 24132 / 37545489 = 0.01606850825674423896836181837983252795029517394220
```

Therefore a current HFV1 candidate must improve the non-rate component term by
at least `0.0160685082567` before it can tie FEC6/PR110 on the CPU-axis score.

| variant | current bytes | required component gain now | estimated floor bytes | required gain after estimated floor | priority |
|---|---:|---:|---:|---:|---|
| `seed_top16_component_hardpairs` | 202649 | `0.0160685082567` | 188668 | `0.00675913423314` | dispatch first if remote slot clean |
| `identity` | 202649 | `0.0160685082567` | 188342 | `0.00654206421443` | defer |
| `nonidentity` | 202649 | `0.0160685082567` | 188387 | `0.00657202786732` | defer |

## Interpretation

This changes dispatch priority. The readiness packet made all three candidates
look mechanically dispatchable, but rate arithmetic says identity and
nonidentity should not consume remote exact-eval slots first. They pay the same
large sidecar penalty without a plausible component-gain mechanism.

The only defensible first exact-eval candidate remains:

- `seed_top16_component_hardpairs`

It still needs a large component gain at current bytes, but it is the only row
whose frame selection is tied to component-hardpair information. If exact eval
is fired before sidecar recoding, this is the candidate to fire.

The better engineering move before broader dispatch is to shrink the HFV1
sidecar. Even using the rough estimated member floor, the seed row's required
component gain drops from `0.0160685` to `0.0067591`, which is a much more
reasonable empirical test.

## Recommended next action

Build the HFV1 sidecar recoder before launching identity/nonidentity exact eval.
If one remote exact-eval slot is available and the Modal ledger is clean, launch
only `seed_top16_component_hardpairs` first.
