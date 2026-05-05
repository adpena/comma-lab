---
name: Check 85 STRICT — training-script metric-key display bug class extinct (DARTS-S incident)
description: 2026-04-29 PM. 5h Vast.ai 4090 sweep (Lane DARTS-S V1, $1.41) appeared to show seg=nan pose=nan loss=277 for 400 epochs. Root cause: train_segmap.py:380 read epoch_metrics keys "seg"/"seg_loss"/"pose"/"pose_loss" but SegMapTrainer.train_epoch returns "seg_dist"/"pose_dist". Both keys missing → float("nan") fallback printed. Actual seg_dist=2.37, pose_dist=158.49 — those are also frozen (separate model-not-learning bug, not addressed by this check).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The bug class — "metric-dict key references diverge from trainer return keys"

When a printer/logger reads `epoch_metrics.get("KEY", float("nan"))`:
- If `"KEY"` exists in the trainer's return dict → printed correctly
- If `"KEY"` does NOT exist → silent NaN fallback → operator sees "training is broken" → wastes hours debugging the wrong layer

This is a META-BUG: the trainer is doing its job (returning real numbers), but the operator's view of training is corrupted by a key mismatch in the display layer. Hours of GPU compute can elapse before SSH-debug catches it.

## Today's incident (Lane DARTS-S V1 sweep)

- Dispatched scripts/remote_lane_darts_s_segmap_arch_sweep.sh on Vast.ai 4090 (5.46h ago, $1.41 spent)
- run.log shows: `[train_segmap] epoch=0 loss=277.018384 seg=nan pose=nan` … 400 epochs of nan
- SSH inspection of segmap_train.json revealed actual values: pose_dist=158.49, seg_dist=2.37, kl_aux=4.48, loss=277.02 — all FROZEN across 400 epochs (model not learning, separate bug)
- The display NaN was a printer bug — the trainer returns dict with keys {"loss", "pose_dist", "seg_dist", "kl_aux", "num_steps", "epoch"} but the printer at train_segmap.py:380 read "seg"/"seg_loss" (and "pose"/"pose_loss"). Both keys missing → `float("nan")` default → NaN printed.

The model-not-learning bug is REAL and separate (parameters constant across 400 epochs despite optimizer.step() called every epoch). Investigating that is the next step. But the printer bug is what made the failure mode INSCRUTABLE — it looked like NaN propagation when actually parameters were just stuck.

## Permanent fix landed (Check 85 STRICT)

`check_training_script_metric_keys_consistent` in src/tac/preflight.py (around line 6004+):
- Scans every `experiments/train_*.py` file for `<dict>.get("KEY", ...)` calls on metric-dict-shaped names (`epoch_metrics`, `metrics`).
- Cross-references against `TRAINER_RETURN_KEYS` registry:
  ```python
  TRAINER_RETURN_KEYS = {
      "SegMapTrainer.train_epoch": {"loss", "pose_dist", "seg_dist", "kl_aux", "num_steps", "epoch"},
      "DistillTrainer.step": {"seg_loss", "pose_loss", "pcgrad_conflict", "fridrich_loss",
                              "texture_loss", "linf_penalty", "markov_loss", "uncertainty_loss",
                              "loss", "epoch"},
  }
  ```
- Lands STRICT @ 0 violations after:
  1. Removing dead fallback chain in train_segmap.py:380-385 (kept only `seg_dist`/`pose_dist`)
  2. Registering DistillTrainer keys in TRAINER_RETURN_KEYS

Live STRICT preflight check count: **85**.

## Companion memory

- `feedback_modal_spawn_result_cache_pattern_20260429.md` — companion incident (5 OK runs almost lost to 24h GC)
- `feedback_concurrent_subagent_commit_message_swap_20260429.md` — companion incident (5 subagents committed in parallel, message attribution shuffled)
- This trio represents 2026-04-29 PM's three big DX hardening lessons. All three have memory + permanent prevention landed.

## Outstanding follow-ups

1. **The model-not-learning bug** — Lane DARTS-S V1 still produces frozen pose_dist=158/seg_dist=2.37 across 400 epochs. This is a separate investigation (likely render-output detached from model params, or SegMap's forward doesn't backprop through the affine canvas head). Track as task #249 follow-up.
2. **NaN-watchdog at script level** — even with display fixed, scripts should kill if ACTUAL values (not display) are NaN/Inf for >10 epochs. Lane scripts need a `nan_watchdog.py` sidecar.
3. **Frozen-loss watchdog** — if seg_dist/pose_dist deltas < 1e-6 for >50 epochs AND loss > 0, kill — model isn't learning. This would have caught Lane DARTS-S V1's frozen training in ~30 min instead of 5h.

## Cross-refs

- Commit landing Check 85 + train_segmap.py fix: forthcoming this session
- Vast.ai instance 35850015 (lane_darts_s_v1_a3) — still running config 1 of 3 in the sweep; user has not killed yet
- scripts/remote_lane_darts_s_segmap_arch_sweep.sh — the sweep script
- experiments/train_segmap.py:380-385 — the patched display
- src/tac/segmap_renderer.py:349-575 — SegMapTrainer.train_epoch (canonical return-dict source)
- experiments/train_distill.py:708, 1028-1030 — DistillTrainer keys (registered)
