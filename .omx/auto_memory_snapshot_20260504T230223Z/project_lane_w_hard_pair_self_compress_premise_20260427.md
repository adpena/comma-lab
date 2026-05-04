---
name: Lane W premise — hard-pair-weighted Self-Compression (memory synthesis 2026-04-27)
description: Council synthesis of `feedback_overfit_is_the_goal` + `feedback_posenet_tracking` + `feedback_curriculum_must_use_full_score`. Lane A's avg PoseNet=0.005 hides a heavy tail. Per-pair sensitivity profile + 5-10× weighting on top-K hardest pairs steers Lane S's per-channel learnable bit-depth to protect PoseNet-critical channels. Predicted band [0.85, 1.10] — could be first sub-1.0 score. Subagent dispatched 2026-04-27 to implement.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Premise:** Lane A scored 1.15 [contest-CUDA] with avg PoseNet=0.005 (near floor). But the contest scorer SUMS per-pair distortions; the heavy-tail pairs (probably 5-10% of 600 = 30-60 pairs, e.g., turns, lane changes, sun glare) likely dominate the score. Lane S Self-Compression in default mode would average bit-allocation across ALL pairs and may not preserve the channels critical to those hardest pairs.

**Three load-bearing memories**:
1. `feedback_overfit_is_the_goal` — single-video contest, memorization optimal, per-pair-specific optimization rewarded
2. `feedback_posenet_tracking` — heavy tail dominates; track per-pair not avg
3. `feedback_curriculum_must_use_full_score` + `project_council_shower_thoughts` — hard-frame curriculum using FULL score formula (100*seg + sqrt(10*pose) + 25*rate)

**Lane W = combination of the above with Lane S**:
- **Pre-pass**: profile per-pair sensitivity using Lane A's renderer + poses + masks. Compute per-pair contrib = 100*seg_i + sqrt(10*pose_i). Identify top-K hardest (default K=30).
- **Training**: weight those K pairs by 5-10× during Lane S Self-Compression training. Forces SC bit-allocation to spend bits where they matter most.
- **Effect**: SC layers protect channels critical to the WORST pairs, sacrificing precision on channels that only affect easy pairs. Net: lower auth score even at the same average bit-budget.

**Predicted EV**:
- Lane S alone: [0.95, 1.10] (structural rate win, may regress hard pairs)
- Lane W (Lane S + hard-pair): [0.85, 1.10] — could land sub-1.0
- If both fail: ruled out the per-channel SC + hard-pair combination empirically

**Council voices**:
- Yousfi: "The heavy-tail pairs are likely steganalysis-hard frames — high-frequency texture, low-quality JPEG-style artifacts, sudden camera motion. Sensitivity profiling should reveal them as a cluster."
- Fridrich: "Per-pair-specific budgets are the inverse-steganalysis principle applied to compression: 'spend bits where the detector cares most'."
- Hotz: "Existing per-channel SC + per-pair weighting is just two known mechanisms composed. No new training infra needed. Cheap to try."
- Quantizr: "I never considered per-pair-conditional QAT. If you find this works, it's a paper-worthy generalization of standard quantization."
- Contrarian: "If the hard-pair set is unstable across runs (different seeds → different K hardest), the weighting might just add noise. Profile multiple seeds first to verify stability before training."

**Key implementation knobs**:
- K (number of hardest pairs to weight): start at 30 (5%), adjust based on sensitivity profile
- weight multiplier: 5.0 (Hotz: "10 might dominate scorer signal; 5 is conservative")
- when to apply weight: scale per-pair losses BEFORE backward (so SC bit-depth gradient sees the upweighted signal)
- per-pair iteration order: must remain DETERMINISTIC; only the loss values change

**Predicted failure modes**:
1. **Heavy tail is noise, not structure** — random Lane A failures, not concentrated in specific pairs. Hard-pair weighting wouldn't help because there's no consistent target.
2. **Heavy tail correlates with already-protected layers** — the hard pairs happen because of FiLM / motion module / fuse_conv (all kept FP32 in Lane S anyway), not the bulk Conv2d that SC quantizes. Then SC has no bit-allocation to redirect.
3. **Bias toward overfit** — SC + hard-pair weights overfits to Lane A's specific failure modes; auth eval re-runs the same 600 pairs so this overfit IS the goal (per `feedback_overfit_is_the_goal`).

**Status**: Subagent dispatched 2026-04-27 to implement. Awaiting deliverables: profile_pair_sensitivity.py + train_renderer.py --pair-loss-weights flag + remote_lane_w_hard_pair_self_compress.sh + 3 tests.

**Related memories**:
- `feedback_overfit_is_the_goal` — single-video memorization
- `feedback_posenet_tracking` — heavy-tail score sensitivity
- `feedback_curriculum_must_use_full_score` — full-score loss weighting
- `project_council_shower_thoughts` — hard-frame curriculum proposal
- `feedback_no_signal_loss` — record per-pair stats for provenance
