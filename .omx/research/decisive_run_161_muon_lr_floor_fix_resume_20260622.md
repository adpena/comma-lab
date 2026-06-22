# Mid-burn apparatus correction: stage-8 Muon LR-floor fix applied via guard-blessed resume (2026-06-22)

**Task #161** (recursive review + math optimization of the decisive LOCAL PR95 run BEFORE the burn completes).
Authority: `[contest-CPU advisory]` run; pointer UNMOVED 0.19110. This note is provenance, not a score claim.

## What was wrong
The decisive run (`yousfi_r3_taper_marginhinge_e5`, launched 2026-06-20) was started with
**`--no-muon-lr-floor-fix`** (BUG-B active). Recovered exact argv via psutil; the launcher's own help
(`launch_split_by_head_basin.py:84-94`) states *"pass `--muon-lr-floor-fix` for the decisive
full-curriculum run so the stage-8 d_seg-finishing polish is PR95-faithful"* — this run did not.

## The bug (derived from the scheduler, `driver.py:1827-1855`)
Without the fix, the stage-8 Muon optimizer SHARES AdamW's `lr_lambda`, whose cosine multiplier floors at
`eta_min_ratio = lr_floor_ratio/adamw_lr = 0.5`. Muon's peak LR is *larger* than AdamW's (orthogonalized
updates are normalized → bigger steps), so flooring Muon at 0.5× its own large peak leaves it bouncing at a
high absolute LR through the entire finisher — **it never anneals to the fine-polish regime**. That is
exactly the stage whose job is the d_seg κ-busting (the 2.6× needed to beat 0.19110). With the fix, Muon
gets its own floor `lr_floor_ratio/muon_lr` → anneals to the intended low LR → the polish lands.

## The remedy (safe + zero-loss)
The resume guard (`driver.py:3216-3231`) explicitly blesses toggling this flag at any pre-stage-8
(`has_muon=False`) checkpoint. The run was at **stage3_v332_smooth, epoch_in_stage 550, has_muon=False**.
Procedure: SIGTERM (clean exit 0.5s, no orphans) → verified the checkpoint loads (decoder + latents + EMA +
AdamW + Muon + schedulers + RNG intact) → relaunched with **one token changed**
(`--no-muon-lr-floor-fix` → `--muon-lr-floor-fix`), all else byte-identical, same `--out-dir` (auto-resume),
durable nohup daemon.

## Verification
Resumed process wrote a FRESH manifest (age 3.6s) at **epoch_in_stage 575, stage3, has_muon=False, and
`muon_lr_floor_fix = True`**. Trajectory/summary actively appending (n_records continued from ~9,200, NOT
reset). Zero epochs lost.

## State at correction
- Operating point (last advisory exact eval, ep ~8625): d_seg 0.00213, d_pose 0.000251 (pose term 0.050),
  rate 0.0584 → S 0.3212.
- Pointer-move threshold (beat borrowed 0.19110, using THIS run's measured pose+rate 0.108): **d_seg < 8.26e-4** (2.58× from current).
- Sub-0.15 (T_3): **d_seg < 4.16e-4** (5.1× from current).
- The d_seg finishers ahead: stage 5 (C1a-L7, 9000ep) + stage 8 (Muon, 5000ep, NOW with the correct floor).

Operator-approved (2026-06-22, "Restart now with the fix"). Sister anchor: the 2026-06-19 apparatus-audit
directive (verify the decisive run isn't measured on a broken/under-throughput stack; this is that, caught
in the act). `apparatus_audit_pr95_breakthrough_blocker_20260619T214001Z.md`.
