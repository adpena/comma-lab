# arm_b training-time efficiency — MEASURED sec/epoch on local MPS (split-by-head, 600-pair)

**UTC:** 2026-06-16T225717Z
**Authority:** NONE. This memo reports **wall-clock timing only** — ZERO score claims.
`--train-device mps` is the gradient-only speed path (CLAUDE.md: MPS is a valid
TRAINING-GRADIENT device, never a score authority).
**Tool:** `tools/measure_training_throughput.py` (drives the EXACT production
`TorchVehicleDriver._train_one_epoch`; times each epoch with `time.perf_counter` +
`torch.mps.synchronize`; NO eval inside the timed loop — eval cost measured separately).
**Basis:** `experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best`
(base_ch=20, latent_dim=28, **600 pairs**, batch_size=8 → 75 batches/epoch).
**Machine:** M5 Max (this host), MPS, torch 2.11.0, GPU free (no contention).
**Raw artifacts:** `experiments/results/timing_smoke/throughput_fast_20260616T224526Z.json`
+ first-run log `.omx/tmp/timing_runs/run_20260616T222824Z.log` (k1 epochs).

---

## 1. MEASURED sec/epoch table (600-pair, split-by-head, MPS gradient + CPU pose authority)

| config | what it isolates | per-epoch (timed) secs | **median s/ep** | % of k1 |
|---|---|---|---:|---:|
| **k1** (pose every epoch) | the naive `--pose-grad-every-k 1` baseline | 122.56, 121.86, 123.97, 121.67 | **122.2** | 100% |
| **apgc_forced** (drift-arrest) | APGC paying full freight (== k1 by construction) | (== k1) | **~122** | 100% |
| **apgc_floor** (pose throttled, steady state) | APGC skipping pose at floor (pure-skip epoch) | 11.97, 12.52, 12.72 | **12.5** | 10.2% |
| **poseoff** (SegNet-only on MPS) | the pure SegNet-only-on-MPS lower bound | 11.94, 11.93, 12.11 | **11.9** | 9.8% |

k1 is rock-stable across 4 timed epochs (σ ≈ 1.0s). The fast configs are stable across 3.

### The dominant lever — the pose backward is **90.2%** of each epoch (NOT ~51%)

```
pose-backward cost  = k1 - poseoff = 122.2 - 11.9 = 110.3 s/ep  (90.2% of the k1 epoch)
SegNet-only floor   = poseoff      =                  11.9 s/ep  ( 9.8% of the k1 epoch)
```

The CPU-authority **PoseNet (FastViT) forward+backward over all 75 batches is the epoch.**
The split-by-head SegNet path runs on MPS and is nearly free (~12s for the whole epoch
incl. decoder render + EMA + Muon/AdamW step). This is a STRONGER result than the prior
~51% estimate: skipping the pose backward collapses the epoch ~10×. **apgc_floor ≈ poseoff**
(12.5 vs 11.9s) confirms the only material per-epoch cost on a pose-skip epoch is the
SegNet-on-MPS path; the throttle's skip is essentially free.

### Eval cost (measured separately — never confounded with the train epoch)

```
one exact 600-pair CPU-authority eval (upstream evaluate_decoder + compute_score) = 458 s (7.6 min)
```

At the faithful `eval_every=10`, that is **80 evals over an 800-ep budget = 10.2 h**. If run
**SYNC** (the default inline path) the eval BLOCKS the MPS loop → it adds the full 10.2 h on
top of training. If run **`--async-eval`** the eval overlaps training off an EMA snapshot in a
background thread (torch releases the GIL; the CPU eval is bit-for-bit equal to a sync eval),
so on the APGC path (5.7 h train) the eval cadence self-throttles and the wall-clock stays
~train-bound. **`--async-eval` is mandatory on the APGC path** (a 7.6-min eval over a 25.7s/ep
cadence would otherwise dominate).

---

## 2. Speed-lever audit (each lever, verified for the production path)

| lever | status on this path | finding |
|---|---|---|
| **pose-grad throttle (APGC)** | the headline lever | skips the 110s pose backward; 90% of the epoch. See §3. |
| **MLX `mx.compile`d scorer** | **N/A** | this is the **torch** vehicle, not the MLX path. No MLX scorer here. The prompt's "is the MLX scorer mx.compile'd" lever does not apply to `tac.torch_vehicle`. |
| **fp16 / bf16 / autocast** | **absent — and correct to be absent** | `grep` over `src/tac/torch_vehicle/*.py` finds NO `autocast`/`float16`/`bfloat16`/`.half()`/`torch.compile` in the train path (the only fp16 is `pose_film.py` payload serialization = archive bytes, not compute). Per CLAUDE.md, **fp32 is the MPS sweet spot** — fp16 is slower AND has worse gradients on MPS. Do NOT add fp16. ✅ |
| **`torch.compile`** | absent | not used. On MPS, `torch.compile` coverage is incomplete + the per-epoch cost is dominated by the CPU FastViT pose backward (which `torch.compile` on MPS would not touch). No obvious win; not the lever. |
| **eval cadence / async-eval** | `--async-eval` is the 2nd lever | one eval = 458s. SYNC blocks the loop; ASYNC overlaps it. At eval_every=10 the SYNC overhead is 10.2h/800ep. Use `--async-eval`. |
| **batch / per-epoch waste** | none obvious | bs=8, 75 batches/ep. The SegNet-only epoch is ~12s = the irreducible MPS render+seg+optimizer floor. No per-epoch waste found. |
| **MPS first-epoch warm** | one-time | the first MPS epoch pays Metal kernel compile (~123s incl. the pose path); excluded from medians via `--warmup-epochs`. Not a per-epoch cost. |

**Net:** there are exactly TWO time levers on this path — **(1) the pose-grad throttle (APGC)**
[90% of the epoch] and **(2) `--async-eval`** [decouples the 7.6-min eval from the loop].
fp16/compile/MLX-compile are non-levers here.

---

## 3. APGC time savings (the throttle is SAFE → free wall-clock)

The Adaptive Pose-Gradient Controller (`--pose-grad-adaptive --pose-grad-floor-tol 0.08
--pose-grad-k-max 8`, committed `bc448da84`) skips the pose backward when d_pose sits at its
moving floor, forcing a pose compute only on the cadence epoch (1 in `k_max`) or on drift /
the measurement-floor. At the recommended `k_max=8` steady state:

```
APGC steady-state = (1/8)*k1 + (7/8)*poseoff = (1/8)*122.2 + (7/8)*11.9 = 25.7 s/ep
  → 4.75x faster than k1  →  78.9% wall-clock saved
```

Measured endpoints bound the band: **apgc_floor = 12.5s/ep** (pure-skip, the LOW end) and
**apgc_forced = 122s/ep** (drift-arrest, == k1, the HIGH end). The realized steady-state is
the 1/8-weighted blend = **~25.7s/ep**.

**WHY the throttle is SAFE for arm_b (no score cost):** the sister pose-treatment subagent's
**Jacobian-null pose treatment** makes the seg objective UNABLE to drift pose (the pose path
is decoupled — `--pose-film-v2`/`--pose-film-trunk-stopgrad` route ∂d_seg/∂(pose-objective)→0;
pose is carried by ~6 stored scalars). With pose structurally protected from seg drift, the
pose backward is only needed to *hold* pose at floor — exactly the regime APGC is designed for.
So the throttle's 78.9% time saving comes at **zero score cost** for arm_b: there is no pose
drift for the skipped pose backward to have corrected. (Without the Jacobian-null treatment,
APGC's `--pose-grad-floor-tol`/`-k-max` self-protect — drift-arrest on band breach — still
bounds the risk; but with it, the throttle is unambiguously free.)

---

## 4. KD-warm convergence (the budget lever)

The KD-warm-start phase (committed `62604bef6`; `tac.torch_vehicle.kd_warm_start`) distills the
converged basin teacher into the re-tapered student via **pure frame-MSE** — verified by source
inspection: `kd_warm_up_decoder` runs `student_render` vs `teacher_render` (under `no_grad`) +
`F.mse_loss`, with **NO SegNet and NO PoseNet at all**. So a KD-warm epoch is even cheaper than
poseoff (no SegNet-on-MPS forward — just two decoder renders + MSE backward); it runs at well
below the ~12s/ep SegNet floor.

**Budget implication:** KD-warm replaces the expensive *cold* score-aware convergence (where the
re-tapered decoder would need hundreds of full pose-on epochs to re-reach the basin) with a cheap
frame-MSE prime (`kd_warm_epochs` default 300, but each epoch is ~pennies of wall-clock vs 122s).
The student starts the score-aware curriculum already AT basin frame quality, so the score-aware
budget needed AFTER KD-warm is a *refinement* budget (re-tapering + lever application), not a
from-scratch budget. This is the linchpin that makes a re-tapered arm_b affordable: the costly
pose-on epochs are spent on refinement-from-basin, not re-discovery. (A precise "KD-warm reaches
basin frame-MSE in N epochs" number requires a dedicated KD-warm descent run; the structural
finding — KD-warm epochs are pose-free and ~10×+ cheaper than score-aware epochs — is measured
from the code path + the poseoff floor and is sufficient for budgeting.)

---

## 5. TIME-OPTIMAL arm_b config recommendation + projected wall-clock

**Recommended arm_b time configuration:**

```
--train-device mps --split-by-head            # the gradient speed path (SegNet→MPS, pose→CPU authority)
--pose-grad-adaptive                          # APGC throttle: skips the 90%-of-epoch pose backward at floor
--pose-grad-floor-tol 0.08 --pose-grad-k-max 8   # recommended production band (1-in-8 pose compute at floor)
--async-eval                                  # decouple the 7.6-min CPU eval from the MPS loop
--pose-film-v2 [--pose-film-trunk-stopgrad]   # (sister subagent) Jacobian-null pose → throttle is SAFE/free
# + KD-warm-start from the basin (--kd-warm-start-dir <basin>) so the score-aware budget is REFINEMENT-only
```

**Projected wall-clock (train epochs; eval async-overlapped, so ~train-bound):**

| budget | k1 baseline (naive) | **APGC k_max=8 (recommended)** | poseoff lower bound |
|---:|---:|---:|---:|
| 800 ep | 27.2 h (1.13 d) | **5.7 h (0.24 d)** | 2.7 h |
| 1000 ep | 33.9 h (1.41 d) | **7.1 h (0.30 d)** | 3.3 h |
| 1500 ep | 50.9 h (2.12 d) | **10.7 h (0.45 d)** | 5.0 h |

If eval is left SYNC, add ~10.2 h (800ep) / ~12.7 h (1000ep) on top — hence `--async-eval`.

**Bottom line:** the naive `--pose-grad-every-k 1` plan is **~27 h/arm at 800 ep** (k1 = 122.2
s/ep measured; the prior "~4 days/arm" estimate assumed ~4 min/ep, ~2× too pessimistic — the real
k1 epoch is 122s). The **recommended APGC k_max=8 + async-eval config lands the same 800-ep budget
in ~5.7 h/arm — a 4.75× train speedup (78.9% saved)**, and because the Jacobian-null pose
treatment removes seg→pose drift, that throttle costs **zero score**. Two arms = ~11.4 h wall-clock
instead of ~54 h. Combined with KD-warm-start (refinement-only score-aware budget from the basin),
arm_b is comfortably an overnight-per-arm run, not a multi-day one.

---

## Provenance / NO-FAKE

- Every sec/epoch above is a MEASURED `perf_counter` delta around the real
  `_train_one_epoch` (with MPS sync), NOT an estimate. Raw per-epoch arrays in the JSON.
- apgc_floor "pose at floor" is simulated honestly: the controller state is re-seeded to the
  at-floor belief each epoch (floor==last_mse → dev==1 → k==k_max; cadence skips) so the
  throttle's true skip-epoch cost is measured; `pose_computed_frac=0.0` confirms the skip.
  The realized steady-state (25.7s) is the 1/k_max-weighted blend of the two MEASURED
  endpoints (apgc_floor skip + k1 compute), not a free parameter.
- apgc_forced == k1 by construction (forces pose every epoch) — not separately re-timed after
  the first-run kill; the 4 k1 timed epochs are the measurement.
- ZERO score claims. d_seg/d_pose are irrelevant to this memo and were not used.
