# Full-MPS base_ch=20 basin — LAUNCHED, guard CLEAN through ep20 (104× lever confirmed admissible)

> **What this unit did:** generalized the basin launcher to a `--split-by-head / --no-split-by-head` flag, launched the **full-MPS** (both scorer heads on MPS, the 104× lever) base_ch=20 n600 basin with an ACTIVE CPU-authority safety guard, and confirmed the guard is **CLEAN** through the first two authority evals (the d_pose transient at ep10 collapsed ~230× by ep20 — the exact CHAOS-clean signature the verdict memo predicted, NOT the n600 real-divergence signature). The split-by-head 7-11× fallback was **not needed**.

**Date:** 2026-06-12 (UTC) · **Subagent:** `full-mps-basin-launch-20260612` · **Authority:** `[macOS-CPU advisory]` NON-PROMOTABLE. The basin's d_seg/d_pose/score are torch-CPU exact on the EMA shadow; MPS is the GRADIENT backend only. **Frontier UNMOVED:** 177,169 B `[contest-CPU]` (S=0.19109982). This unit did NOT lower the exact score — it stands up the faster training backend that the optimizer-chaos verdict (`mps_pose_drift_patchable_verdict_20260612.md`, `79e975430`) unblocked. A sub-frontier basin GATES, never IS, a paired contest-CPU+CUDA exact eval.

---

## 1. Launcher change (commit `c4b33ff81`)

`experiments/launch_split_by_head_basin.py` previously hardcoded `split_by_head=True` at BOTH the `TorchVehicleConfig(...)` and `RealScorerContext(...)` calls — it could only launch the salvage path. Added a `--split-by-head / --no-split-by-head` flag (`argparse.BooleanOptionalAction`, **default True for safety**, so every legacy launch command is byte-for-byte unchanged) and threaded `args.split_by_head` into both call sites. `--no-split-by-head` selects the full-MPS basin (both heads on `train_device`).

5 NO-FAKE tests (`experiments/tests/test_launch_split_by_head_basin_flag.py`): legacy default unchanged (True), explicit `--split-by-head` True, `--no-split-by-head` False, full-MPS device threading (train=mps/device=cpu), and a source-level guard that the two hardcoded `split_by_head=True` forms are gone and `split_by_head=args.split_by_head` threads both call sites. All pass (0.26 s).

## 2. Launch command + out-dir

```bash
nohup .venv/bin/python -u experiments/launch_split_by_head_basin.py \
  --no-split-by-head --train-device mps --device cpu \
  --base-channels 20 --n-pairs 600 \
  --eval-every 10 --checkpoint-every-epochs 1 --ema-decay 0.999 \
  --out-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600 --seed 0 \
  < /dev/null > .../launch_nohup.log 2>&1 & disown
```

* **out-dir:** `experiments/results/torch_vehicle_full_mps_basin_bc20_n600` · **pid 59649** (durably detached, survives this session per `nohup … < /dev/null & disown`).
* Resumable per-epoch checkpoint (`torch_vehicle_checkpoint_state.pt`), best-by-canonical-score, full per-epoch torch-CPU exact telemetry JSONL (`torch_vehicle_trajectory.jsonl`), DONE-marker-on-exit. Reuses the byte-identical n600 GT target cache (`capstone_gt_targets_cache/gt_targets_n600.pt`, 943 MB) — skipped the ~2.5 h precompute.
* The byte-closed BEST archive is written to `best/best_archive.bin` (the artifact a future paired exact eval consumes).

## 3. Guard verdict — CLEAN (104× confirmed; split-by-head fallback NOT triggered)

The reported d_seg/d_pose are torch-CPU authority (`exact_eval` runs on `self.device=cpu`, NEVER MPS — verified in `driver.py:560-568`). The CHAOS verdict predicts d_seg descends AND d_pose stays low / descends; the n600 REAL-divergence signature was a monotonic pose climb 0.835 → 7 → 36 (`diverged=True`).

| eval | d_seg | d_pose | score | is_best | archive_bytes |
|---:|---:|---:|---:|:--:|---:|
| ep10 | 0.07172 | **12.9437** | 18.6138 | ✓ | — |
| ep20 | **0.01923** | **0.0559** | 2.7336 | ✓ | 94,532 |
| ep30 | **0.01091** | **0.0081** | 1.4366 | ✓ | — |

**Three consecutive CLEAN evals.** The d_pose transient at ep10 (12.94) collapsed monotonically to 0.056 (ep20) → 0.0081 (ep30) — the OPPOSITE of the n600 real-divergence signature (which RISES 0.835→7→36). d_pose at ep30 (0.0081) now MATCHES the trustworthy CPU daemon's ep10 authority pose (pid 42035: d_pose=0.0078), and d_seg (0.0109) is now BELOW the daemon's ep10 d_seg (0.0122) — i.e. the full-MPS basin tracks the trusted CPU authority trajectory, not a divergent one. **TRIP CONDITION NOT MET** (no monotonic d_pose rise to >>1; d_seg descended below 0.1 monotonically). Guard verdict: **CLEAN, confirmed across 3 authority evals**. The full-MPS basin is validated; the proven bit-identical split-by-head 7-11× fallback was NOT needed.

(Caveat on the daemon cross-check: pid 42035 uses `run_capstone_resumable_curriculum.py` with `pr95_adamw_then_muon` — a *different* curriculum/optimizer harness than the launcher's PR95 8-stage driver, so it is a ROUGH reference, not a bit-identical control. The load-bearing guard signal is the basin's OWN d_pose trend, which collapsed — that alone is the CHAOS-clean signature.)

## 4. Measured throughput + projected hours-to-basin

* **Training (MPS, the 104× lever): ~14.7 s/epoch steady-state** (the launcher PR95 8-stage curriculum, base_ch=20, 600 pairs). The trustworthy CPU daemon at the same operating point is ~1158 s/epoch → the MPS training lever delivers **~80× wall-clock speedup in practice** on this exact basin.
* **BUT the CPU authority eval dominates at eval epochs: ~699 s per eval** (wall_clock jumped 135 s → 834 s across the ep10 eval — the 600-pair torch-CPU exact_eval + byte-closed archive build). At `--eval-every 10` the effective rate is ≈ (10·14.7 + 699)/10 ≈ **84 s/epoch effective**.
* **Projected wall-clock (eval-every-10):** full faithful 29,650-epoch budget ≈ **697 h** (impractical — eval cadence, not MPS training, is the bottleneck). Proportional budgets: 2000 ep ≈ 47 h, 4000 ep ≈ 94 h. **Operator lever:** raise `--eval-every` (e.g. 50) to amortize the eval cost — at eval-every-50 the effective rate drops to **~29 s/epoch** (2000 ep ≈ 16 h, 4000 ep ≈ 32 h), and/or set `--total-epoch-budget` to a proportional 2000-4000. The basin is already at d_seg 0.011 / d_pose 0.008 / score 1.44 by ep30, so the basin floor is reachable well inside a proportional budget. The run is resumable, so the budget can be tuned by relaunching against the same out-dir (and a separate relaunch at higher `--eval-every` would inherit the checkpoint).

## 5. Authority caveat (the load-bearing honesty line)

Everything here is `[macOS-CPU advisory]`, NON-PROMOTABLE. The frontier is **UNMOVED** (177,169 B, S=0.19109982 `[contest-CPU]`). The full-MPS basin running cleanly LICENSES the 104× training backend; it does NOT itself lower the exact score. The pointer moves ONLY when `best/best_archive.bin` is run through `upstream/evaluate.py` on contest-CPU AND contest-CUDA (1:1 hardware). The basin GATES that paired exact eval; it never substitutes for it.

## 6. Chaos-control v2 corroboration (independent of this verdict)

`experiments/measure_torch_vehicle_chaos_control.py` was re-launched to a FRESH out-root (`experiments/results/torch_vehicle_chaos_control_ab_v2`, n48/30ep/single-stage-Muon, `--grad-noise-rel 2e-4`) — the prior runs (incl. the verdict memo's original) collided on the same dir under 3-way CPU contention (basin ep-eval + live daemon) and were killed mid-Arm-A. The final relaunch (pid 64433) runs at **`nice -n 15`** so it yields CPU to the basin's authority evals and does not get starved. It is CPU-bound and slow under contention (the verdict memo does NOT depend on it; it corroborates the *mechanism* that a 2e-4 perturbation alone reproduces the pose gap). Status: **IN FLIGHT** (arm_clean → arm_noise → gate; verdict.json pending). When it lands, its verdict.json fills the verdict memo's `<!-- CHAOS_CONTROL_RESULTS -->` placeholder with real numbers (the NO-FAKE fix — the memo's "reproduces" claim is then backed by data, not asserted ahead of it). **The full-MPS guard verdict (§3) is independent of this corroboration** — it is settled by 3 consecutive clean authority evals on the live basin.

## 7. 6-hook wire-in (Catalog #125)

1. **Sensitivity-map** — N/A (training-backend launch, no byte-allocation change).
2. **Pareto constraint** — N/A.
3. **Bit-allocator** — N/A.
4. **Cathedral autopilot dispatch** — the full-MPS basin is a local FREE actuator, now RUNNING; advisory until a byte-closed paired exact eval.
5. **Continual-learning posterior** — this unit confirms the prior verdict's prediction empirically: the full-MPS basin's authority d_pose collapsed 12.94→0.056 across ep10→ep20 (CHAOS-clean, not divergence), so the 104× lever trains cleanly with CPU-authority BEST-tracking.
6. **Probe-disambiguator** — the active CPU-authority guard (d_pose-trend watcher) IS the disambiguator; it RESOLVED transient-vs-divergence → TRANSIENT (clean).
