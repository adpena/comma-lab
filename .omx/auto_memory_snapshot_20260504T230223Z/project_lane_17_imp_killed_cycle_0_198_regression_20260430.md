---
name: Lane 17 IMP cycle 0 = 1.98 [contest-CUDA] — KILL VERDICT WITHDRAWN (measurement bug, not science result)
description: 2026-04-30 ~22:55 UTC. Initial KILL verdict on Lane 17 (88K-param IMP) WITHDRAWN under adversarial grand council scrutiny. The 1.98 [contest-CUDA] is a real bytestream score, but the model that produced it had only ~3.5 seconds of "fine-tune" instead of the 200 epochs claimed — the dispatch script's "in-script lightweight loop" is a STUB that the comment promises gets "swapped for train_distill" but the swap never happened. The 1.98 reflects the post-prune-no-recovery state, NOT a converged sparse 88K-param model. Real verdict requires re-running with proper train_distill fine-tune (hours per cycle).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TL;DR

The earlier KILL verdict in this same memory file was WRONG. It is now WITHDRAWN by 8-of-10 inner council vote (Shannon, Dykstra, Contrarian, Hotz, Selfcomp, Quantizr, MacKay, Ballé in favor of withdrawal; Yousfi and Fridrich abstain pending re-measurement).

## The smoking gun

`/home/zeus/pact/lane_j_imp_results/cycle_0/stats.json`:
```json
{
  "epochs": 200,
  "fine_tune_steps": 200,
  "elapsed_sec": 3.47   // ← 200 epochs in 3.47 SECONDS = stub loop, not real training
}
```

`/home/zeus/pact/lane_j_imp_results/cycle_0/cycle.log`:
```
[lane-j-imp] fine-tune: 200 epochs @ lr=0.0001 (in-script lightweight loop; deploy script swaps in train_distill)
[lane-j-imp] EMA enabled (decay=0.997)
[lane-j-imp] saved cycle 0 artifacts → /home/zeus/pact/lane_j_imp_results/cycle_0
[lane-j-imp] DONE in 3.5s
```

The script comment promises a `train_distill` swap that the deploy script never performs. The "in-script lightweight loop" is the placeholder that ran — likely a few SGD steps on a tiny synthetic-pair batch. Real Lane G v3 training takes hours; real IMP cycle fine-tune at 200 epochs on the L40S would be 10-30 minutes minimum.

## What the 1.98 score actually measures

Lane G v3 weights with 20% magnitude-pruned sparsity applied, then ~3.5 seconds of stub-loop "fine-tune" (effectively no recovery), then exported to FP4A and auth-eval'd against the Lane G v3 anchor masks/poses.

Components:
- PoseNet distortion 0.120 (34.8× anchor 0.00345) — motion.head was magnitude-pruned but never re-trained, so the network can no longer compute pose-equivariant flows
- SegNet distortion 0.005 (1.25× anchor 0.00401) — seg.head appears to be more robust to 20% sparsity, but with no fine-tune we can't tell if this is real or asymmetric pruning bias
- Rate 0.0154 (sparsity helps; 16% smaller archive)

## Why the asymmetric regression supports "stub bug, not architectural ceiling"

If 88K params truly were at the architectural floor and 20% sparsity destroyed model quality globally, both PoseNet and SegNet would regress proportionally. Instead PoseNet went 34.8× worse while SegNet went only 1.25× worse — a 27.8× asymmetry. Two plausible explanations:
1. **Stub-bug hypothesis (council 8/10 favored)**: motion.head is more sensitive to weight perturbation than seg.head. With ~zero recovery from the stub loop, motion.head shows full damage; seg.head is naturally robust to small perturbations (averaging-over-pixels effect). Real fine-tune would recover motion.head.
2. **Architectural ceiling hypothesis**: motion.head IS the bottleneck at 88K params and 20% sparsity prunes its critical weights. Real fine-tune cannot recover.

Hypothesis 1 has support from Frankle 2019 (lottery-ticket subnetworks DO recover with proper fine-tune cycles). Hypothesis 2 has no supporting evidence — we don't have a converged sparse cycle-0 to test it.

## What needs to happen for a real verdict

1. **Wire the train_distill swap into `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh`**. The script currently runs `experiments/train_imp_cycle.py` which has the in-script stub. The fix: after the stub-loop completes, invoke `experiments/train_distill.py --resume <cycle_dir>/renderer.pt --epochs 1000 --lr 1e-4 --auth-eval-on-best` to produce real cycle weights.
2. **Re-run cycle 0 with proper fine-tune**. Estimated cost: ~2 hours on L40S × $0.60/hr = $1.20. If score still > 1.10 anchor threshold after real fine-tune, KILL is justified. If score ≤ anchor threshold, continue cycles 1-9.
3. **Add a STRICT preflight check** that train_imp_cycle.py's `_default_finetune_loop` is NEVER the production training path — must be EXPLICITLY swapped for train_distill in the dispatch script. The "in-script lightweight loop; deploy script swaps in train_distill" pattern is exactly the kind of thing that becomes a forgotten stub.

## Cost so far on Lane 17

- Lightning Studio L40S cycle 0 stub run + auth-smoke: ~$0.20 sunk
- Total Lane 17 spend tonight: ~$0.20 (well under $25 cap)
- Real cycle 0 (proper fine-tune): ~$1.20 estimated
- Full 10-cycle IMP (proper fine-tune): ~$12 estimated

## Lessons (3 distinct bug classes uncovered)

1. **Stub-loop pretending to be real training**: a placeholder labeled "in-script lightweight loop; deploy script swaps in train_distill" where the swap silently never happens. Need preflight check.
2. **Reporting inconsistency**: stats.json says "epochs: 200" while elapsed_sec says 3.47 — internal inconsistency that should be a script-time assertion (`assert elapsed_sec > epochs * 0.5` would have caught this).
3. **Adversarial grand council prevents premature kill**: the user's "extreme adversarial" challenge caught a kill that would have wasted a real lane. ALL future KILL verdicts must pass the adversarial council before being recorded as memory.

## Process protocol going forward

- **Every KILL verdict requires explicit "I checked X, Y, Z internal-consistency assertions" line in the memory file**, OR the verdict is conditional ("KILL pending: X, Y, Z").
- **No memory file marked `project_lane_*_killed_*` until the lane gets a real-recipe re-run**.
- **The grand council adversarial challenge is non-skippable** for any kill in a >$1M science-grade decision space. (We are at $0.20 sunk and a $25 budget — definitely within "kill scrutiny required" zone.)

## Cross-refs

- feedback_imp_dispatch_shape_mismatch_fix_20260430.md (the 3 bugs we DID fix on the dispatch path)
- feedback_imp_local_backport_landed_20260430.md (commit 9fdabc9e — the local backport of those 3 fixes)
- project_lane_g_v3_landed_1_05_20260428.md (the anchor we measured against)
- The user's challenge: "was the IMP results reliable and is that verdict actually hold up acording to etreme adversarail grand councill" (2026-04-30 ~22:50 UTC) — the right question that caught this.
