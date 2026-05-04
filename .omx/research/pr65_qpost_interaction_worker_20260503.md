# PR65 qpost interaction worker - 2026-05-03

## Scope

Local-only builder for PR65 qpost bias atoms combined with existing C089/PR75
P6 action-runtime-compatible archives. No remote GPU, training, or exact-eval
job was dispatched.

Write-scope files:

- `experiments/build_pr65_qpost_interaction_candidates.py`
- `src/tac/tests/test_build_pr65_qpost_interaction_candidates.py`
- `experiments/results/pr65_qpost_interaction_worker_20260503/**`
- `.omx/research/pr65_qpost_interaction_worker_20260503.md`

## Command

```bash
.venv/bin/python experiments/build_pr65_qpost_interaction_candidates.py --output-dir experiments/results/pr65_qpost_interaction_worker_20260503
```

Stdout mirror:

- `experiments/results/pr65_qpost_interaction_worker_20260503/build_stdout.json`

Canonical summary:

- `experiments/results/pr65_qpost_interaction_worker_20260503/candidate_summary.json`

## Active-claim guard

Read `.omx/state/active_lane_dispatch_claims.md` and found 152 nonterminal
active claims. The builder skipped the C089/top40 P6 qpost top040 and top080
combinations because they matched active PR65 qpost T4 exact-eval claims:

- `exact_eval_pr65_qpost_v2_bias_poseadv_top040_t4_20260503T0805Z`
- `exact_eval_pr65_qpost_v2_bias_poseadv_top080_t4_20260503T0810Z`

The skipped records are preserved in
`candidate_summary.json.skipped_candidates`. No active duplicate archive was
built for those C089 source combinations.

## Best local candidate

Candidate:

- `ix_pr75_lagtop67_p6_bias_top080`

Archive:

- `experiments/results/pr65_qpost_interaction_worker_20260503/candidates/pr75_lagtop67_p6/ix_pr75_lagtop67_p6_bias_top080/archive.zip`

Byte/SHA:

- bytes: `276610`
- SHA-256: `b91f03162758329f97382b42a32f3937c1f7288b589a96fcf820761302a2e51b`

Source action base:

- `pr75_lagtop67_p6`
- source score: `0.3154979650614253` from A++ T4 no-frontier exact eval
- source archive SHA-256: `d73f0957cf2d8da0526c1786613443331ccb913f5648131cfaaac5e6f7eae972`
- runtime probe: single-member `p`, payload magic `P6`

Charged change:

- archive members: `p` and `qpost.bin`
- `qpost.bin` bytes: `164`
- selected active qpost atoms: `80`
- selected pair count: `80`
- score_claim: `false`
- dispatchable_later: `true`
- remote_dispatched: `false`

Trace/rate screen:

- public trace opportunity bound: `0.0023214431701688142`
- added rate score: `0.0001717916099055202`
- sub-0.314 break-even component gain: `0.0016697566713308196`
- trace slack vs target: `0.0006516864988379946`

Exact-eval command template, if later claimed and dispatched by an operator:

```bash
.venv/bin/python -u experiments/contest_auth_eval.py --archive /Users/adpena/Projects/pact/experiments/results/pr65_qpost_interaction_worker_20260503/candidates/pr75_lagtop67_p6/ix_pr75_lagtop67_p6_bias_top080/archive.zip --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir /Users/adpena/Projects/pact/experiments/results/pr65_qpost_interaction_worker_20260503/exact_eval_work/ix_pr75_lagtop67_p6_bias_top080
```

This command was not run.

## Other local candidates

All are planning-only, score_claim=false, remote_dispatched=false:

| candidate | bytes | sha256 | dispatchable_later | trace_slack_vs_target |
|---|---:|---|---|---:|
| `ix_pr75_lagtop67_p6_bias_top080` | 276610 | `b91f03162758329f97382b42a32f3937c1f7288b589a96fcf820761302a2e51b` | true | 0.0006516864988379946 |
| `ix_pr75_pose2_top67_p6_bias_top080` | 276596 | `999baf5ef307eab32a5cd02a0984a8d517eb9708ac4b69e669c88b0a6724c01b` | true | 0.0006269053384891756 |
| `ix_pr75_pose_safe_ampm1_p6_bias_top080` | 276575 | `fb2f136edfca5827560cd012b650b435e5c31c4367f59b193a4e9c3881727871` | true | 0.0005876839645555356 |
| `ix_pr75_lagtop67_p6_bias_top040` | 276571 | `d90912443bc9f60e972a0ac30f190ec1e552fa599a2c0162d8f0f1a99d3bcbd6` | true | 0.00029004802085068815 |
| `ix_pr75_pose2_top67_p6_bias_top040` | 276557 | `7e847f09288605eabd7373516e48e7b27da878c7892ec5b47306062e0beff01b` | true | 0.0002652668605018691 |
| `ix_pr75_pose_safe_ampm1_p6_bias_top040` | 276536 | `9f19b75ac6fdafa561b6f1a7a8f0695ded8409f33ccb11dcffa3abe17b5f78ad` | true | 0.0002260454865682291 |

## Verification

```bash
.venv/bin/python -m py_compile experiments/build_pr65_qpost_interaction_candidates.py src/tac/tests/test_build_pr65_qpost_interaction_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr65_qpost_interaction_candidates.py -q
```

Result: `2 passed in 0.10s`.

No remote jobs were dispatched and no AGENTS edits were made.
