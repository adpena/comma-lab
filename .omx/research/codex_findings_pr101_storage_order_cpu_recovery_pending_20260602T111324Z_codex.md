# Codex Findings: PR101 Storage-Order CPU Recovery Still Pending

Created: 2026-06-02T11:13:24Z

Axis: `[contest-CPU]` recovery custody only. This is not a score claim, promotion claim, rank claim, or exact-eval completion.

## What Ran

Canonical recovery only:

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir /Users/adpena/Projects/pact/experiments/results/modal_auth_eval_cpu/pr101_storage_order_len24_cpu_20260601T1955Z \
  --call-id fc-01KT2BZT54G6CXPMD94SY43MMH \
  --timeout-s 0
```

The command returned `RECOVERY_RC=4`, which is the canonical pending path.

## Result

Recovery summary:

- `status=pending`
- `call_id=fc-01KT2BZT54G6CXPMD94SY43MMH`
- `score_claim=false`
- `score_claim_valid=false`
- `promotable=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `recovered_at_utc=2026-06-02T11:13:24Z`

The active dispatch ledger was refreshed with a new pending row:

- lane: `lane_pr101_storage_order_len24_exact_cpu_20260601`
- job: `pr101_storage_order_len24_exact_cpu_20260601`
- status: `active_modal_cpu_auth_eval_pending_recovery_poll`
- predicted next poll: `2026-06-02T14:13:24Z`

Machine-readable artifact:

- `.omx/research/pr101_cpu_recovery_poll_adjudication_20260602T111324Z.json`

## Verdict

NO-GO for any PR101 terminal-result claim: the CPU auth eval is still pending and no terminal artifact exists.

NO-GO for new exact, full-video, or CUDA launches while PR101, Z5, or current full600/full-video blockers remain nonterminal in the active claim ledger.

GO only for local, SSD-backed, false-authority-labeled SNeRV/HiNeRV stack work that does not consume the blocked exact/full-video/CUDA lane.

## Roadmap / Blockers

Immediate: poll PR101 CPU through `tools/recover_modal_auth_eval.py` only. Do not close the claim, score, promote, or dispatch paired CUDA until the canonical recovery path yields a terminal artifact.

SNeRV: continue pair-robust scorer-loop/NES decoder-QAT with PoseNet as a hard guard. Preserve mixed decoder modes, receiver-decoded byte accounting, and per-pair deltas. Treat scalar/closed-form HF sweeps as local controls, not promotable evidence.

HiNeRV: continue longer real-teacher SegNet/PoseNet training with coder-aware QAT, joint P18/P19 weighting, dense VJP L-infinity allocator, and MLX prefilter before any CPU replay or exact spend.

PR95: use upstream PR95 `hnerv_muon` as the baseline/control. Any claim to beat PR95 must replay the same archive/runtime/eval axis control, not compare across advisory or inferred axes.

Promotion: blocked until byte-closed full-600 receiver proof plus paired contest CPU/CUDA pass, with active-claim ledger closure and archive/runtime custody intact.
