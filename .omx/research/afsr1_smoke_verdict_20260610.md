# AFSR-1 smoke verdict — KILLED-AT-IMPLEMENTATION (this aimed-retrain recipe does not descend)

UTC 2026-06-10 · `[macOS-MLX/CPU advisory]` / telemetry+exact-pair-scorer · pre-registered kill fired.
Run: afsr1_aimed_retrain_20260610 (smoke 11871, exited 05:52:54Z). Frontier baseline (48-pair eval
subset) d_seg=5.4158e-4.

## The trajectory (kill criterion: d_seg must descend; pre-registered)
| it | EMA d_seg | live d_seg | dS_dist | vs frontier |
|---|---|---|---|---|
| 100 | 5.859e-4 | 6.107e-4 | +4.995e-3 | ABOVE |
| 175 | 6.052e-4 | 6.229e-4 | +7.219e-3 | ABOVE |
| 275 | 6.168e-4 | 6.187e-4 | +8.576e-3 | ABOVE, MONOTONIC RISING |
d_seg rose MONOTONICALLY away from the frontier the entire run; d_pose also drifted up
(3.16e-5 → 3.50e-5). The aimed-retrain recipe as configured moves BOTH axes the wrong way.

## Verdict: KILLED-AT-IMPLEMENTATION (Catalog #307), NOT a paradigm kill
The AFSR-1 PARADIGM (reopen the frontier decoder as a trainable object, aim by the measured maps) is
NOT falsified — THIS SMOKE'S RECIPE is. The fixed frontier decoder is a tightly memorized point;
continuing to train it with the flip-targeted score-aware loss as configured DEGRADED it (consistent
with the decoder-QAT-recovery finding: a memorized single-video point has no slack to re-learn into
under naive continuation — gradient steps walk OFF the memorized optimum). The smoke correctly fired
the kill at $0 before any paid dispatch (MVP-first honored; no Modal spent).

## Why it degraded (the mechanism, for the next iteration)
Fine-tuning a converged memorized renderer with a NEW objective + LR pulls it off its sharp optimum
faster than the aimed term can recover the targeted flips — the same knife-edge the QAT-recovery lane
hit. The flip-targeting weighted the loss toward the 66k residual flips, but moving weights to fix
those flips perturbs the 3M correct pixels (receptive-field coupling), net d_seg UP.

## Reactivation criteria (pinned; the paradigm stays live)
1. LOWER LR + EMA-anchored trust region (tiny steps; the optimum is sharp — try LR/10, freeze most
   tensors, only adapt the heads + a small adapter).
2. Train-from-INIT at a smaller architecture (T11 channel-pruning lineage) rather than continuing the
   memorized point — a fresh basin has slack the memorized one lacks (the QAT-recovery lesson).
3. The null-space training constraint (T5) as the PRIMARY objective, not flip-targeting — put error
   into the certified-invisible DOF rather than chasing argmax flips that perturb correct pixels.
4. Multi-video / larger frame set so the decoder isn't a single-point memorization with zero slack.

## Routing
The frozen-frontier RATE axis is exhausted (T1/T8/S12 negative); the DISTORTION axis via THIS
continuation recipe is KILLED; the distortion axis via reactivation paths 1-4 (esp. train-from-init
smaller-arch + null-space-primary) remains the live campaign — but each is NEEDS-REAL-WORK, not a
cheap smoke. Frontier stands at 0.19109982 [contest-CPU] (recoded-R3); the immediate banked value is
the CUDA pairing (in flight) making that frontier submission-ready.
