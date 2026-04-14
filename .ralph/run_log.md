# run log

## 2026-04-14T16:00:00Z — λ-SWEEP COMPLETE: 0.87 is architectural ceiling

λ_cap sweep from v5-best (auth=0.87): {500→0.90, 750→0.87, 1000→0.87, 1500→0.90, 2000→0.87}
Score plateaus at 0.87 across 3 different λ_cap values. Ceiling is architectural, not optimization.
Quantizr at 0.60 has 47x better PoseNet (0.00066 vs 0.031) — architecture gap, not tuning gap.
Council decision: kill λ-sweep path, pivot to constrained generation (joint frame pair output).
Kaggle still failing: dataset mount race condition persists even with 5-min retry. Need 30+ min wait.

## 2026-04-14T14:30:00Z — v5 CHECKPOINT LANDSCAPE SWEEP (9 auth evals)

Full drift curve from ep12500 to ep16999:
- ep12500: 0.93 | ep12600: **0.87** (BEST) | ep13000: 0.88 | ep13500: 0.96
- ep14000: 0.95 | ep14500: 0.90 | ep15000: 0.88 | ep15500: 1.26 (collapse starts)
- ep16000: 1.09 | ep16500: 1.15 | ep16999: 1.37

Sweet spot: ep12500-15000 (all < 1.00, beating v3). Two local minima at ep12600 (0.87) and ep13000/15000 (0.88).
Collapse onset: ep15500 (SegNet and PoseNet both degrade under weak Lagrangian).
Cost: 9 × $0.29 = $2.61. High information density — maps full Pareto drift trajectory.

## 2026-04-14T13:20:00Z — NEW BEST: asym_v5_lagrangian_fixed renderer_best auth=0.8700

- **auth=0.8700** — 13% improvement over v3 baseline (1.00 -> 0.87)
- Checkpoint: renderer_best.pt at ep12600 (only 200 epochs after resume from v3)
- PoseNet: 0.031 (35% better than v3's 0.048) — the big mover
- SegNet: 0.00217 (held flat vs v3's 0.00210)
- Rate: 0.00401 (identical — same model size)
- Key insight: R2 Lagrangian clamp (λ 10000→1000) on resume created a brief transient
  where the over-constrained v3 model found a better PoseNet basin before drifting
- Late checkpoint ep16999 REGRESSED to 1.37 — weaker constraints let model drift
- Council implication: short-horizon fine-tuning with reduced Lagrangian > long training
- v5 constraints_met ep16999: auth=1.37 (SegNet 0.004, PoseNet 0.075) — drift confirmed

## 2026-04-14T01:00:00Z — PRE-SUPERVISION BASELINE CONFIRMED: asym_v3_longer_tight auth=1.0000

- **auth=1.0000** — BEST RESULT TO DATE on asymmetric warp architecture
- seg=0.002104, pose=0.047917, rate=0.100355
- checkpoint: asym_v3_longer_tight/renderer_best.pt (pre-supervision, ~ep12400)
- CONFIRMED: supervision 7600 epochs starting from 1.0 regressed to 1.79–2.87
- Contrarian verdict UPHELD: supervision harmed the model
- Council diagnostic closed: pre-supervision was at 1.0, NOT broken before supervision
  → The 4 Lagrangian fixes are still valid but the failure is CONFIRMED supervision-caused
- IMMEDIATE ACTION REQUIRED: promote asym_v3_longer_tight/renderer_best.pt as current best
- NOTE: rate=0.100 vs dilated h64 rate=0.046, but total score 1.0 still beats 1.33
  → asym warp trades rate efficiency for much better seg (0.002) and pose (0.048)
- json: asym_v3_longer_tight/auth_eval_renderer_best.json (exists in volume, timestamp 2026-04-13T17:37:26Z)

## 2026-04-14T00:30:00Z — asym_v4_supervised best_proxy_constraints_met AUTH EVAL

- **auth=2.8700** (CONFIRMED: supervised training collapses PoseNet)
- seg=0.3002, pose=2.4707, rate=0.1004
- checkpoint: renderer_epoch16800_constraints_met.pt (selected by best_proxy_constraints_met strategy)
- proxy_score=0.7266 (score_projection, fallback — full_eval_score absent due to OOM)
- platform: Modal T4
- KEY FINDING: SegNet 0.003 = BEST EVER (equal to dilated h=64). PoseNet 2.47 = CATASTROPHIC (16× dilated).
- KEY FINDING: score_projection is NOT a reliable proxy for auth score when PoseNet collapses —
  it was tracking SegNet improvement (×33 better) while completely missing PoseNet collapse (×16 worse).
  score_projection as training proxy = useless for PoseNet-dominated regime.
- COMPARISON TABLE (asym_v4 supervised):
  ep16800 (best proxy): seg=0.003, pose=2.47  → total=2.87 ← best_proxy_constraints_met selected this
  ep19999 (periodic):   seg=0.566, pose=1.12  → total=1.79 ← randomly landed on more balanced epoch
  dilated_h64 baseline: seg=0.006, pose=0.060 → total=1.33 ← current best
- INVALIDATED: both supervised checkpoints are worse than dilated baseline; supervised approach FAILS
- AWAITING: raft_only (Kaggle v6) for valid A/B comparison on same architecture
- saved: /results/asym_v4_supervised/auth_eval_renderer_epoch16800_constraints_met.json

## 2026-04-13T23:00:00Z — Kaggle both kernels RUNNING (v6 datasets, canonical)

- asym_warp_raft_only: kernel v6 RUNNING, dataset v6 (all assets confirmed present)
- constrained_gen_smoke: kernel v5 RUNNING, tac-1.0.3 with P100 CPU fallback
- Dataset v6 = tac-1.0.3 + raft_flow.pt(900MB) + renderer_best_v3.pt + posenet_targets.bin + 0.mkv
- Root cause of prior failures: race condition — kernels ran while dataset was still processing
- Fix: push kernel AFTER dataset listing confirms all files present

## 2026-04-13T22:30:00Z — Kaggle kernel v4 + dataset v4 deployed

- asym_warp_raft_only: kernel v4 running, dataset v4 (tac-1.0.1 + raft_flow.pt + posenet_targets.bin + renderer_best_v3.pt)
- constrained_gen_smoke: kernel v3 running, video path fix in tac-1.0.1
- code_file pattern: train_renderer_fridrich.py IS the code_file — bootstrap preamble injects sys.argv before Click
- pyproject.toml bumped to v1.0.1 (commit 5c994bd1)

## 2026-04-13T21:25:00Z — asym_v4_supervised ep19999 AUTH EVAL

- **auth=1.7900** (REGRESSION from baseline ~1.0 at ep12400)
- seg=0.5664, pose=1.1188, rate=0.1004
- archive=150,715 bytes, 287,019 params
- checkpoint: renderer_epoch19999.pt (periodic, NOT best — best_score=0.6019 never updated = resume floor never beaten)
- platform: Modal T4
- key finding: 7600 epochs of PoseNet+RAFT supervision hurt the model. Both seg and pose regressed vs ep12400.
- hypothesis: supervision disruption phase extended past ep19999. Or supervision signal misaligned with scorer.
- status: NEGATIVE RESULT for supervised path at this epoch. Awaiting raft_only (Kaggle) for A/B comparison.
- saved: /results/asym_v4_supervised/auth_eval_renderer_epoch19999.json

## 2026-04-10T21:30:00-05:00 - promoted floor synchronized

- authoritative promoted floor: **1.33**
- variant: `dilated_h64`
- platform: `modal_a10g`
- evidence: `reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt`
- mirrors are now expected to be derived from canonical promoted_result.json

## 2026-04-11T23:46:58 — mask_renderer on modal_a10g ep70

- proxy=None
- Notes: L1 loss 0.031, Phase 1 pretrain, 31s/ep

## 2026-04-11T23:47:00 — dp_sims on modal_a10g ep80

- proxy=None
- Notes: L1 loss 0.018 LEADING, Phase 1, 17.7s/ep

## 2026-04-11T23:47:00 — wavelet on modal_a10g ep110

- proxy=8.2
- Notes: Phase 2 scorer training, 12.9s/ep

## 2026-04-11T23:47:00 — dilated_h64 on modal_a10g ep103

- proxy=1.407
- Notes: Steadily improving from 1.476

## 2026-04-12T00:25:49 — dp_sims on modal_a10g ep100

- proxy=4.54
- PoseNet: 0.75700000
- SegNet: 0.05100000
- Notes: First Phase 2 epoch. FP4 saved 2.3MB. Score will improve rapidly.

## 2026-04-12T00:25:50 — dilated_h64 on modal_a10g ep78

- proxy=1.423
- PoseNet: 0.06700000
- SegNet: 0.03400000
- Notes: Steady improvement from 1.476 start

## 2026-04-12T00:29:21 — dp_sims on modal_a10g ep100

- proxy=4.54
- PoseNet: 1.47200000
- SegNet: 0.00700000
- Rate: 2.262
- Notes: Only 1 Phase 2 epoch. Died at scorer start. SegNet excellent (0.007). Must resume P2 on Lightning.

## 2026-04-12T00:29:21 — dilated_h64 on modal_a10g ep84

- proxy=1.413
- PoseNet: 0.07200000
- SegNet: 0.00600000
- Rate: 0.046
- Notes: 84/2500 epochs. INT8 45KB saved. On track for sub-1.3.

## 2026-04-12T00:33:59 — dp_sims on modal_a10g ep109

- proxy=2.93
- PoseNet: 0.53200000
- SegNet: 0.00600000
- Rate: 2.262
- Notes: 9 Phase 2 epochs. SegNet 0.006 matches best! PoseNet needs training. Trajectory: sub-1.5 by ep 150.

## 2026-04-12T00:33:59 — dilated_h64 on modal_a10g ep99

- proxy=1.4
- PoseNet: 0.07000000
- SegNet: 0.00600000
- Rate: 0.046
- Notes: 99/2500 epochs. On track. Needs 800+ more.

## 2026-04-12T01:17:03 — dp_sims on modal_a10g ep189

- proxy=2.5
- PoseNet: 0.48200000
- SegNet: 0.00300000
- Rate: 2.262
- Notes: FINAL Modal run. SegNet 0.003 TIES Quantizr! PoseNet 480x gap + rate 5.7x gap = entire remaining problem.
