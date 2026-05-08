# Lane 19 — Contest-CUDA Dispatch Plan (Phase F)

> Supersession note (2026-05-08 signal-loss review): this is a historical
> pre-Level-2 dispatch memo, not current launch authority. Do not dispatch from
> the commands below without revalidating against current `main`, opening a
> fresh `tools/claim_lane_dispatch.py claim ...` row, passing current
> preflight, and obtaining explicit operator authorization. The bands in this
> memo are predictions, not achieved score evidence.

**Date:** 2026-04-30
**Lane:** 19 (SegNet logit-margin boundary loss)
**Status:** Historical plan only; not ready-to-launch without the current gates above.

## Cost estimate

| Stage | GPU | $/hr | Wall clock | Cost |
|---|---|---|---|---|
| Stage 1 — training (1980 epochs) | RTX 4090 | $0.25 | ~5h | $1.25 |
| Stage 1b — FP4A export | (same instance) | — | <1min | included |
| Stage 2 — half-frame archive build | (same) | — | ~5min | included |
| Stage 3 — pose TTO 500 steps | (same) | — | ~30min | included |
| Stage 4 — contest_auth_eval | (same) | — | ~5min | included |
| **Total** | RTX 4090 | $0.25 | ~5.5h | **~$1.50** |

**Under $10 cap → no pre-approval needed per CLAUDE.md "GPU budget" section.**

## Launch invocation (Pattern A — nohup detach)

Per CLAUDE.md "Codex CLI invocation" Pattern A (background tasks must be detached, not via `Bash run_in_background:true`).

```bash
# 1. Provision Vast.ai 4090 (canonical)
.venv/bin/python -m tac.deploy.vastai.cli launch \
  --gpu RTX_4090 \
  --label lane_19_logit_margin \
  --script scripts/remote_lane_19_logit_margin.sh

# 2. Once SSH is up, the script runs Stages 0-4 in sequence on the remote.
# 3. Monitor via heartbeat: $LOG_DIR/heartbeat.log on the remote.
# 4. Final RESULT_JSON appears in $LOG_DIR/auth_eval.log on the remote.
# 5. Harvest result + tarball back to local.
```

**OR** (alternative — direct vastai CLI):

```bash
nohup bash -c '
  vastai create instance <offer-id> \
    --label lane_19_logit_margin \
    --image pytorch/pytorch:2.4.0-cuda12.1 \
    --disk 30 \
    --onstart-cmd "cd /workspace && bash scripts/remote_lane_19_logit_margin.sh"
' < /dev/null > /tmp/codex_runs/lane_19_dispatch.outer.log 2>&1 &
disown
```

## Predicted result band [prediction]

`[0.75, 1.05]` `[contest-CUDA]` standalone vs Lane G v3 1.05 anchor.

| Outcome | Score | Action |
|---|---|---|
| Floor — Lane 19 buys -3e-3 SegNet | 0.75 | PROMOTE Level 2 → Level 3 (mark real_archive_empirical + contest_cuda gates) |
| Mid — -1e-3 SegNet | 0.95 | PROMOTE; competitive sub-Lane-G-v3 result |
| Ceiling — no improvement vs CE | 1.05 | KILL: demote to Phase 3 deferred per profile comment |
| Worst — score regresses | >1.05 | INVESTIGATE: likely the detach() fix has a subtle bug under real-CUDA semantics (MPS/CPU smoke unanimously passed) |

## Post-result protocol

### If score < 1.05:
1. Mark gates: `tools/lane_maturity.py mark lane_19_segnet_logit_margin --gate real_archive_empirical --evidence reports/lane_19_logit_margin_cuda.json`
2. Mark gate: `--gate contest_cuda --evidence reports/lane_19_logit_margin_cuda.json`
3. Update memory `project_lane_19_logit_margin_landed_20260430.md` with `[contest-CUDA]` score.
4. Lane 19 graduates Level 1 → Level 3.

### If score >= 1.05:
1. Document the kill in memory + `.omx/state/findings.md`.
2. The 18 module + 11 preflight tests REMAIN — bug class is permanently extinct via Check 93 + the regression tests.
3. The Round 3 detach() fix is a permanent improvement to the loss formulation (would benefit any future logit-margin work).

## Risk mitigations

- **NVDEC bad-host:** Stage 0 NVDEC probe fires before any GPU spend. If it fails, instance destroys immediately.
- **Silent skip:** Stage 4 has RESULT_JSON guard. If auth eval crashes silently, exit 4 + log FATAL.
- **Detach regression:** test_fragility_weights_detached_in_loss_synthetic pins it; if Round 3 fix is reverted, the test fails before any GPU spend.
- **Instance orphan:** `--label lane_19_logit_margin` (CLAUDE.md non-negotiable Check A); tracker registers via Check B.

## Operator approval gate

Per CLAUDE.md project rule: "Cost ≥$10: pre-approval. Estimate $1.50 → proceed with plan memo." This memo IS the proceed-document.

**Standing instruction superseded:** no launch is authorized by this memo.
