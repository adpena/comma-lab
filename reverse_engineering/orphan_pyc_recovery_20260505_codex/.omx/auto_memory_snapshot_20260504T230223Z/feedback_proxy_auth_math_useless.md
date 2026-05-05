---
name: proxy-auth math is essentially useless
description: Proxy MSE / proxy distortion / proxy "loss" never predicts contest-CUDA auth. Even on CUDA-CUDA the gap can be 100-350x. MPS makes it worse (PoseNet specifically drifts 23x on MPS).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Binding rule:** the proxy loss (`extract_gt_pose_targets` MSE, `load_differentiable_scorers` smooth gradients, `--eval-roundtrip` simulation, `--posetto-noise-std=0.5`) is a TRAINING SIGNAL, not a measurement. It does not predict contest-CUDA auth distortion within any usable factor.

**Why:**
- LANE-B (2026-04-26): proxy PoseNet 0.0007 vs CUDA auth 0.246 — **350x gap**. Both on CUDA. `eval_roundtrip` + `noise_std=0.5` both wired and threaded.
- The proxy uses `load_differentiable_scorers` (smooth grads, FP16 quirks). Auth uses `upstream/evaluate.py` (no-grad, integer round-trips, exact sampler).
- MPS PoseNet drifts an additional 23x vs CUDA — multiplicative on top of any proxy-auth gap (memory: `feedback_mps_cuda_drift_critical`).
- Historical pose TTO proxy 0.00057 → auth 0.476 = 835x gap (memory: `feedback_proxy_auth_gap_835x`).

**How to apply:**
1. NEVER report a proxy score as a "result." Always tag `[proxy]` and treat as advisory only.
2. Before any pose TTO / training claim "improved score," run contest-CUDA auth on the EXACT archive bytes.
3. For pose TTO specifically: the optimization may not improve auth at all even when proxy converges. Compare to the baseline poses (`load_baseline_poses`) before adopting a TTO output.
4. Smoke pattern: run 50-pair TTO → build archive → CUDA auth eval → compare to baseline auth. If auth didn't improve, the TTO is destroying signal even if proxy looks great.
5. The **authoritative measurement loop is**: contest-CUDA inflate.sh → upstream/evaluate.py on EXACT archive bytes. Nothing else.
6. Stop trying to "close the gap." It's a structural artifact of differentiable-vs-exact. Build the authoritative measurement into the loop instead.

**LANE-B post-mortem in one line:** 6.5h GPU + $2 spent optimizing the wrong metric. The fresh TTO poses were 23x worse on PoseNet than the baseline poses they replaced. Should have run smoke auth at step 100 and caught it in 3 minutes.
