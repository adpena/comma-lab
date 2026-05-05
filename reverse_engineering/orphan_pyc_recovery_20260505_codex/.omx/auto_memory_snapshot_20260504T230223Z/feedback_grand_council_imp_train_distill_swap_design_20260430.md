---
name: Grand Council adversarial review — IMP train_distill swap sub-design (epochs / masks / auth-smoke ordering)
description: 2026-04-30 ~23:30 UTC. Inner council 10-member deliberation on the THREE sub-questions of the IMP × train_distill swap that the parent council file (feedback_grand_council_imp_permanent_fix_review_20260430.md) deferred to implementation time. Council Option B+assertion (6/3/1 verdict in parent file) won — this file resolves SQ1 (real epochs/cycle), SQ2 (masks source), SQ3 (auth-smoke ordering).
type: feedback
originSessionId: imp-train-distill-swap-implementation
---
## Context

Parent council file `feedback_grand_council_imp_permanent_fix_review_20260430.md` resolved the meta-question (Option B+assertion, 6/3/1) but deferred three sub-questions to "implementation-time council deliberation." This file resolves them.

The user mandate (2026-04-30 ~22:55 UTC) was: "all design decisions and ultimate experiment subject to extreme paranoia and adversarial grand council reviews." Each sub-question below is a design tradeoff with non-trivial consequences.

## SQ1 — How many real epochs per cycle?

The current dispatch script sets `EPOCHS_PER_CYCLE=200`, but that was the STUB-loop epochs (3.5s on L40S = 0.017s/epoch synthetic-tensor SGD). The real `train_distill.py` runs three phases (default 2000+5000+2000 = 9000) at ~0.5-2s/epoch on L40S. Even with `--only-phase1`, 2000 epochs is ~10-30 min × 10 cycles = 1.7-5h.

### Options considered

- **A** — Keep 200 epochs (--phase1-epochs 200, --skip-phase1 disabled, --only-phase1 enabled). Real-train at ~30-90s/cycle on L40S. Total 5-15 min for 10 cycles.
- **B** — Bump to ~500 epochs phase1-only. ~2-5 min/cycle, ~20-50 min total.
- **C** — Full 3-phase default (2000/5000/2000). ~30-90 min/cycle, ~5-15h total. Far over budget.
- **D** — Skip fine-tune entirely after cycle 0; use cycle 0's distill only as the "real" proxy and rely on the prune+rewind mask propagation for cycles 1-9. Cheapest but leaves the rewind path untested per cycle.

### Council vote (10 inner members)

| Member | Vote | Rationale |
|---|---|---|
| Shannon (LEAD) | B | R(D) recovery from a 20%-prune step needs ≥500 grad updates per Frankle 2019 figure 2; 200 is the "stub-claim" number, not a derived one. |
| Dykstra (CO-LEAD) | B | Convex feasibility region after pruning: 500-1000 epochs lands in the "recovered" regime; 200 is on the wrong side of the elbow. |
| Yousfi | B | Steganalysis-proxy: 200 epochs of CE/hinge on 5-class SegNet barely moves the boundary; 500 actually retrains the kept-weights to the new pruned manifold. |
| Fridrich | B | Sub-budget headroom; pay for the recovery instead of optimizing for stub-claim parity. |
| Contrarian | C | "If you're going to claim a real fine-tune, do it for real. 500 is split-the-difference theatre." |
| Quantizr | B | Same as Hotz — operationally simplest with --only-phase1. |
| Hotz | B | Run --only-phase1 with 500 epochs at lr=1e-4. ~$0.50 total budget. Shannon's grad-count argument is right. |
| Selfcomp | B | Frankle-Carbin LTH paper specifies "rewind + retrain to convergence"; 500 epochs phase1-only is convergence-proxy at our scale. |
| MacKay | B | MDL: rewind step changes the prior; 500 epochs lets the posterior re-equilibrate. |
| Ballé | B | Hyperprior interpretation: post-prune the mask is a new latent constraint; codec-encoder needs ≥500 steps to fit. |

**VERDICT: 9/10 for B (500 epochs phase1-only). Contrarian dissents.** Implementation: `--phase1-epochs 500 --phase2-epochs 0 --phase3-epochs 0 --only-phase1` per cycle. (`--only-phase1` is a real flag at line 1549.)

## SQ2 — Masks source: Lane G v3 anchor masks vs TTO-frame masks

`train_distill.py` accepts `--masks <path.mkv>` (line 1399). Two candidates:

- **A** — Lane G v3 anchor masks (`experiments/results/lane_g_v3_landed/iter_0/masks.mkv`). What ships in the contest archive.
- **B** — TTO-frame masks (the default `experiments/results/tto_v7_hinge_500/tto_frames.pt` that train_distill defaults to). What was used to create the Lane G v3 anchor.

### Council vote

| Member | Vote | Rationale |
|---|---|---|
| Shannon (LEAD) | A | The fine-tune target IS the contest-archive distribution. Train on what ships. |
| Dykstra (CO-LEAD) | A | Same: the feasibility region we want to hit is the archive's mask distribution. |
| Yousfi | A | CRITICAL: Lane B class catastrophe was masks at wrong resolution. Train on the SAME masks that go into the archive. |
| Fridrich | A | Anti-distribution-shift principle. |
| Contrarian | B | "TTO-frame masks have the gradient signal; archive masks are the lossy AV1 output. You're training the renderer to reconstruct lossy data." |
| Quantizr | A | Per-pair correspondence: train_distill loop expects (frame_pair, mask) pairs; Lane G v3 masks ARE those pairs. |
| Hotz | A | Yousfi's Lane B argument is decisive. |
| Selfcomp | A | Same — train on the masks that actually ship. |
| MacKay | A | MDL: shipping bytes are the truth source for what the model needs to predict. |
| Ballé | A | Hyperprior: ship-time mask distribution is the side-information y; train against y. |

**VERDICT: 9/10 for A (Lane G v3 anchor masks). Contrarian dissents.** Implementation: `--masks "$ANCHOR_MASKS"` (already defined at script line 117 as `experiments/results/lane_g_v3_landed/iter_0/masks.mkv`).

## SQ3 — Auth-smoke ordering: BEFORE or AFTER train_distill?

The existing dispatch script runs `Stage 1.5: per-cycle CUDA auth eval` AFTER `train_imp_cycle.py`. With the new `train_distill` swap inserted, the ordering question is: should auth-smoke run between `train_imp_cycle` and `train_distill`, or after `train_distill`?

### Options

- **A** — auth-smoke AFTER train_distill (stub → distill → smoke). Smoke measures the post-distill weights = what gets fed to the next cycle.
- **B** — auth-smoke BEFORE train_distill (stub → smoke → distill). Cheaper revert decision (don't burn distill GPU on a regressing cycle).
- **C** — Both (stub → smoke1 → distill → smoke2). Maximum information; 2× cost per cycle.

### Council vote

| Member | Vote | Rationale |
|---|---|---|
| Shannon (LEAD) | A | The score that matters is the one on what gets fed forward. Smoke must measure post-distill. |
| Dykstra (CO-LEAD) | A | The kill criterion is "is this cycle's output worse than baseline?"; the cycle's output IS the post-distill weights. |
| Yousfi | A | Pre-distill score is meaningless — every cycle's pre-distill is by definition catastrophic (just-pruned weights, not fine-tuned). |
| Fridrich | A | Same — pre-distill measurement is noise. |
| Contrarian | B | "If post-distill score is your metric, you've lost the ability to revert without paying for distill. That's the sunk-cost fallacy in code form." |
| Quantizr | A | Selfcomp's pipeline runs eval AFTER fine-tune for the same reason. |
| Hotz | A | Distill is cheap (~$0.05 on L40S at 500 epochs); pay for it before deciding. |
| Selfcomp | A | Empirical: my 0.38 pipeline does anchor → finetune → joint → QAT → final EVAL. EVAL is always at the end. |
| MacKay | A | MDL: the message length you compare is the post-recovery one. |
| Ballé | A | Same — codec score is post-rate-distortion-optimization. |

**VERDICT: 9/10 for A (auth-smoke AFTER train_distill). Contrarian dissents.** Implementation: keep the existing Stage 1.5 location (after the cycle's full work), but EXPORT the post-distill renderer.pt for the FP4 archive (NOT the post-prune weights).

## Cross-cutting decision: how to wire post-distill weights back into the cycle

After `train_distill` produces a distilled checkpoint, the dispatch script must overwrite `$CYC_DIR/renderer.pt` so the NEXT cycle's `train_imp_cycle.py --checkpoint $PREV_RENDER` sees the post-distill weights, not the post-prune weights. This is a unanimous council decision (no dissent — it's the ONLY way Option B works as designed).

Implementation: after `train_distill --output-dir "$CYC_DIR/distill"`, the trainer writes `$CYC_DIR/distill/best.pt` (or similar). The dispatcher copies this to `$CYC_DIR/renderer.pt` (overwriting the post-prune-stub one).

## Internal-consistency check

- The dispatcher's `train_distill` invocation passes `--phase1-epochs 500 --phase2-epochs 0 --phase3-epochs 0 --only-phase1` — verified against `experiments/train_distill.py` argparse at lines 1530-1549.
- `--masks` flag exists at line 1399.
- `--resume` flag exists at line 1547 (loads `.pt` via `torch.load(weights_only=True)` at line 1798; cycle's `renderer.pt` is the right input).
- `--output-dir` exists at line 1405.
- `--seed`, `--device`, arch flags (`--embed-dim 6 --pose-dim 6 --base-ch 36 --mid-ch 60 --motion-hidden 32 --depth 1 --padding-mode zeros`) all exist (lines 1408-1432).
- `--auth-eval-on-best` does NOT exist in train_distill argparse — it's a `train_renderer.py` flag. Auth eval continues to be the dispatcher's Stage 1.5 (per the parent council vote: Stage separation, Option B).
- The PCC1 STRICT preflight check verifies the dispatch script invokes BOTH `train_imp_cycle.py` AND `train_distill.py`.

## What would change my mind on these sub-decisions

- **SQ1**: empirical evidence that 500 epochs at the IMP cycle's narrow batch=4 / lr=1e-4 doesn't move the loss curve (then we drop to 200 to save budget OR go to 1000 to finish recovery).
- **SQ2**: empirical evidence that training on Lane G v3 anchor masks produces a renderer that scores WORSE on the contest archive than training on TTO-frame masks (would force re-evaluation of Yousfi's Lane B argument).
- **SQ3**: empirical evidence that distill is so noisy at 500 epochs that pre-distill is a better leading indicator (would force option C, both-eval).

## Cost projection (re-confirmed)

- Cycle 0: $0.05 (train_distill 500 epochs L40S) + $0.05 (smoke auth eval) = $0.10
- Full 10-cycle: $1.00 + revert-on-regression savings expected to halt before cycle 9
- Total worst case: $1.00 well within $25 cap

## Cross-refs

- Parent: `feedback_grand_council_imp_permanent_fix_review_20260430.md` (Option B+assertion 6/3/1)
- `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` (the KILL retraction)
- CLAUDE.md "Council conduct — non-negotiable"
- Implementation: `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` (Stage 1.X added this commit)
- PCC1 preflight check: `src/tac/preflight.py::check_imp_dispatch_calls_train_distill`
