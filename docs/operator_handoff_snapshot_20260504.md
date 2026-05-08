# Operator handoff snapshot (2026-05-04 22:30)

> Supersession note (2026-05-05): Apogee intN predicted bands below are
> historical byte-only planning artifacts. Exact T4 eval of
> `apogee_int4_postfix_sanity_20260505T172500Z` scored
> `1.4286639424744803`, so Apogee intN is now forensic/noncanonical until a
> SHA-tied distortion model, scorer-basin parity gate, or exact positive CUDA
> result exists. Current local A++ frontier remains PR106 x-repack at
> `0.20945123680571204`, which is a byte-identical custody/rate control rather
> than a new representation advance. This snapshot is historical handoff
> context only; do not dispatch from the commands below without a fresh
> `tools/claim_lane_dispatch.py claim ...` row, current preflight, and explicit
> operator authorization.

First operator briefing run with the fixed dashboard (commits dbb0032d +
b3e07b24). This memo captures the **actual terminal state** of the audit-
driven build phase. The next move is human-in-the-loop, not /loop-tick code.

## Phase 1: Pre-dispatch pipeline (READY, BLOCKED ON OPERATOR APPROVAL)

apogee_intN Pareto frontier — 4 launch-ready bits configs, all predict
beating PR106 baseline 0.20946:

| bits | bytes | rate Δ | risk | predicted band | dispatch cost |
|---:|---:|---:|---|---|---:|
| 4 | 109,996 | -0.0508 | HIGH | [0.155, 0.180] | $0.30 |
| **5** | **154,555** | **-0.0211** | **MEDIUM** | **[0.180, 0.196]** | **$0.30** ⭐ sweet spot |
| 6 | 170,450 | -0.0105 | LOW | [0.190, 0.204] | $0.30 |
| 8 | 187,731 | +0.0010 | ALMOST LOSSLESS | [0.196, 0.207] | $0.30 |

Plus 2 sidechannel lanes pre-registered at L1 (council-gated):
- `lane_pr106_latent_sidecar` (PR100 sidecar port, predicted -0.00218 Δ)
- `lane_pr106_yshift_sidechannel` (per-frame pixel translation, stacks)

## Phase 2: Public PR landscape (NOW VISIBLE in dashboard)

| rank | score | bytes | PR family |
|---:|---:|---:|---|
| 1 | **0.209451** | 186,231 | PR106 belt_and_suspenders xrepack |
| 2 | **0.209457** | 186,239 | PR106 belt_and_suspenders adapter ← canonical frontier |
| 3 | 0.226353 | 178,258 | PR101 hnerv_ft_microcodec |
| 4 | 0.227765 | 178,223 | PR103 hnerv_lc_ac |
| 5 | 0.228269 | 178,981 | PR100 hnerv_lc_v2 adapter |
| 6 | 0.229331 | 178,392 | PR98 hnerv adapter |
| 7 | 0.229723 | 178,546 | PR99 hnerv adapter |
| 8 | 0.230432 | 177,849 | PR105 kitchen_sink xrepack |
| 14 | 0.237678 | 186,631 | PR96 rem2_hnerv |
| 15 | 0.251633 | 197,160 | PR97 vibe_coder |

Patterns visible from the now-complete view:
- **PR106 is +0.017 ahead of PR101** — large architectural gap, not a codec polish gap
- **PR98-105 cluster tightly at 0.226-0.231** — same architecture family, different polish
- **PR95 (oldest) at 0.230894** — surprisingly competitive given age
- **PR97 outlier at 0.251** — vibe_coder family scores worse despite larger bytes (+10KB)

The cross-PR comparison framework has now empirically surfaced the
**rate-vs-distortion tradeoff**: smaller bytes don't always win (PR101's
178,258 beats PR105's 177,849 + 177,857 by ~0.004). Distortion matters as
much as rate; the architecture family is the primary lever.

## Phase 3: apogee_intN reconciliation (0/5 PENDING)

```
bits  predicted band     actual    in band?  beats PR106?   evidence
   4  [0.155, 0.180]    (pending)  —         —              (no contest_auth_eval.json yet)
   5  [0.180, 0.196]    (pending)  —         —              (no contest_auth_eval.json yet)
   6  [0.190, 0.204]    (pending)  —         —              (no contest_auth_eval.json yet)
   7  [0.198, 0.208]    (pending)  —         —              (no contest_auth_eval.json yet)
   8  [0.196, 0.207]    (pending)  —         —              (no contest_auth_eval.json yet)
```

**Blocked on operator GPU dispatch approval.** The /loop-tick build phase
has produced everything that doesn't require operator decision:

- 4 dispatch tools (Pareto / dashboard / reconciler / briefing)
- 2 dispatch dry-runs ($0 validation, both pass)
- 1 all-lanes preflight orchestrator
- 25 regression tests (all PASS)
- 2 sidechannel lanes pre-registered with proposals
- 9-memo paradigm thread + INDEX
- 4 honest-deferral memos with reactivation criteria
- Dashboard authority restored (357 scores visible, +5.6×)

## Operator handoff: 4 actionable choices

1. **Dispatch apogee_int5 (sweet spot, -$0.30, predicted [0.180, 0.196])**:
   ```
   APOGEE_INTN_BITS=5 .venv/bin/python scripts/launch_lane_on_vastai.py full \
     --lane-script scripts/remote_lane_apogee_intN.sh \
     --label lane_apogee_int5_pr106 \
     --predicted-band 0.18 0.196 \
     --estimated-cost 0.30 --council-priority 1 --max-dph 0.30
   ```

2. **Council-gate the sidechannel proposals** (5/8 council members must endorse
   before lane_pr106_latent_sidecar can dispatch). See
   `docs/pr100_latent_sidecar_porting_proposal_20260504.md`.

3. **Continue audit-driven research** (no GPU spend; /loop ticks produce more
   memos and pre-registered lanes). Diminishing returns at the ~60-commit mark.

4. **Stop and ship the current best** (PR106 0.209457 is already the best
   exact-CUDA score on disk per the dashboard; the current submission packet
   at `experiments/results/submission_packet_pr100_adapter_20260504/` may
   need refresh to PR106 if it isn't already pointing there).

## Recommendation (advisory only)

**Choice 1** (dispatch apogee_int5) is the highest-EV next move. It:
- Costs $0.30 (well within the $24 Vast.ai cap mentioned in CLAUDE.md)
- Has a tight predicted band (-0.013 to -0.029 score Δ vs PR106)
- Tests the apogee_intN paradigm empirically (turns the prediction into evidence)
- Unblocks Phase 3 of every future briefing run

If int5 lands within band, it becomes the new exact-CUDA frontier AND opens
the sidechannel-stacking pipeline (which is gated on PR106-stacking working
at all).

The continued audit-driven research (Choice 3) has reached saturation —
the next 5 ticks would produce more memos but the next significant signal
requires turning a PREDICTION into a MEASUREMENT, which requires a dispatch.

## Cumulative session deliverables (62 commits)

- 4 tools/ + 2 dry-runs + 1 orchestrator (the dispatch tooling stack)
- 25 regression tests covering every flag-emitting tool + byte-exact invariants
- 9-memo paradigm thread + INDEX (research catalogue)
- 4 honest-deferral memos with reactivation criteria
- 2 PR106-stacking lanes pre-registered at L1
- Dashboard log-parser fix → 5.6× score visibility
- 0 GPU dollars spent
- 0 speculative code shipped
