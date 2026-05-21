# Codex Findings: HFV1 PR101 Exact-Eval Readiness

- timestamp_utc: 2026-05-21T06:42:57Z
- lane: hfv1_pr101_adapter_exact_eval_readiness
- status: LANDED_READINESS_ACTUATOR_AND_DISPATCH_PACKET
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## What landed

New tool:

- `tools/build_hfv1_pr101_exact_eval_readiness.py`

This consumes the archive-surface recode queue and materializes a paired
CPU/CUDA Modal exact-eval readiness packet for HFV1 PR101 adapter archives. It
does not dispatch remote work. It verifies:

- archive SHA-256 and byte count against the queue;
- ZIP member custody;
- membership in known HFV1 manifests;
- local runtime/report/manifest existence;
- local inflate/parity artifact reference;
- Modal uploaded runtime-tree hashes through the canonical paired dispatcher;
- lane-claim conflict status via `claim_lane_dispatch.py --dry-run`.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv1_pr101_exact_eval_readiness.py \
  --queue-json experiments/results/archive_surface_recode_queue_20260521T063508Z/archive_surface_recode_queue.json \
  --output-dir experiments/results/hfv1_pr101_exact_eval_readiness_smoke_20260521T0645Z \
  > /tmp/hfv1_readiness_smoke.json

git diff --check -- tools/build_hfv1_pr101_exact_eval_readiness.py

.venv/bin/python tools/claim_lane_dispatch.py summary

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv1_pr101_exact_eval_readiness.py \
  --queue-json experiments/results/archive_surface_recode_queue_20260521T063508Z/archive_surface_recode_queue.json \
  --output-dir experiments/results/hfv1_pr101_exact_eval_readiness_20260521T064232Z \
  > experiments/results/hfv1_pr101_exact_eval_readiness_20260521T064232Z/stdout.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_hfv1_pr101_exact_eval_readiness.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check \
  tools/build_hfv1_pr101_exact_eval_readiness.py
```

Review tracker result:

- `tools/build_hfv1_pr101_exact_eval_readiness.py`: NORMAL
- 26 entities reviewed
- 0 policy violations

## Readiness artifacts

- JSON: `experiments/results/hfv1_pr101_exact_eval_readiness_20260521T064232Z/hfv1_pr101_exact_eval_readiness.json`
- Markdown: `experiments/results/hfv1_pr101_exact_eval_readiness_20260521T064232Z/hfv1_pr101_exact_eval_readiness.md`
- stdout mirror: `experiments/results/hfv1_pr101_exact_eval_readiness_20260521T064232Z/stdout.json`

SHA-256:

```text
648f52f63d76bef17a02e1734d509a89d966be9934bc9db209eb4f54226aa11f  hfv1_pr101_exact_eval_readiness.json
49b287f550c8de3246d5c01854eb944a2e5942369872262a1120c56fb9e952d4  hfv1_pr101_exact_eval_readiness.md
648f52f63d76bef17a02e1734d509a89d966be9934bc9db209eb4f54226aa11f  stdout.json
```

Byte counts:

```text
43991  hfv1_pr101_exact_eval_readiness.json
 6217  hfv1_pr101_exact_eval_readiness.md
43991  stdout.json
```

## Dispatch-ready candidates

All three nontrivial HFV1 PR101 adapter candidates are plan-ready with zero
readiness blockers in this packet. `ready_for_exact_eval_dispatch` remains
false at packet level because the packet did not execute remote work and does
not contain `contest_auth_eval.json`.

| variant | archive bytes | archive SHA-256 | estimated recoverable bytes | dry-run claims |
|---|---:|---|---:|---|
| `identity` | 202649 | `6554f76fd506cda1066e9bd6672be840914b23e2aa626364a8158bc0a6444e6f` | 14307 | CUDA OK, CPU OK |
| `nonidentity` | 202649 | `172da5770da4f7de530741db58eb7b3e2bdba2352e47c8d6886e79065f8b382b` | 14262 | CUDA OK, CPU OK |
| `seed_top16_component_hardpairs` | 202649 | `72cbd8197a2a8064cb54e7e56e1a5b892a89251c28091f22eba6eef8edff3efb` | 13981 | CUDA OK, CPU OK |

Modal uploaded runtime-tree hashes are identical across the three archives
because they share the same `submission_dir` runtime:

- contest-CUDA runtime tree:
  `9bff8c0a4f1b543bd1f546a4af08a2eea6e3807514eb3931b3ca52fbcc9bfc1b`
- contest-CPU runtime tree:
  `e655b48b88a2a0ec70294b39364cf6a9b87c01f7127c4f68566dba467e376ca8`

## Why remote dispatch was not fired in this turn

`claim_lane_dispatch.py summary` reported two active DP1 Modal claims:

- `lane_dp1_original_baseline_first_paired_anchor_20260520`
- `lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520`

The tracked `.omx/state/modal_call_id_ledger.jsonl` was also already dirty from
partner dispatch activity. Firing three new paired HFV1 exact-eval jobs would
append new remote-dispatch state into that same dirty tracked ledger, mixing
ownership. The HFV1 lanes themselves dry-run conflict-free; the deferral is
ledger hygiene, not a technical blocker.

## Recommended next action

When the Modal ledger is clean or explicitly shared, execute the highest-signal
candidate first:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip --submission-dir experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir --inflate-sh inflate.sh --label hfv1_pr101_seed_top16_component_hardpairs --run-id hfv1_pr101_seed_top16_component_hardpairs_72cbd8197a2a --pair-group-id hfv1_pr101_seed_top16_component_hardpairs_72cbd8197a2a --lane-id-base hfv1_pr101_exact_eval_seed_top16_component_hardpairs_72cbd8197a2a --output-root experiments/results --expected-archive-sha256 72cbd8197a2a8064cb54e7e56e1a5b892a89251c28091f22eba6eef8edff3efb --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists --execute
```

The seed-top16 row is the best first dispatch because it is the only nontrivial
candidate whose selected frames were generated from component-hardpair
information rather than boundary/identity controls. Dispatch result should be
harvested through the existing paired Modal recovery path and then adjudicated
before any score claim.
