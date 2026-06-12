# LOCAL-MLX-DREAM: is local-MLX n600 full-PR95 feasible now? — feasibility audit + resumable landing (2026-06-11)

**Author:** LOCAL-MLX-DREAM subagent.
**Evidence grade:** `[macOS-CPU advisory]` (torch-CPU exact scorer = the trusted authority per CLAUDE.md
"local CPU + MLX GPU good") / `[macOS-MLX research-signal]` (every MLX-GPU number). **NO MPS** anywhere.
$0, local, no paid dispatch. **Did the exact frontier pointer move?** No — this is a feasibility +
resumable-infrastructure enabler for the de-risked n600 run, not a pointer move.

The operator's dream (standing): "30k epochs on MLX GPU with a byte-closed archive directly portable and
frontier when contest auth-eval scored" + "the M5 Max SHOULD sustain super-long (days) measurement sweeps;
orphaning is FINE if results are durably logged/checkpointed."

---

## TL;DR

1. **A great deal of the dream was ALREADY built by sister agents** (search-first, per the directive): the
   MLX-GPU end-to-end scorer-loss bridge (`MLXGpuScorerBridge`) is wired into the capstone trainer behind
   `--scorer-backend mlx_gpu` with a torch-CPU `--authority-recheck-every` gate, plus a full drift audit. I
   did NOT rebuild it. The **two genuine gaps** were: (a) **resumable per-epoch checkpoint/resume** (the
   thing that lost the earlier 2×2 ablation arms) and (b) the **descent-equivalence measurement** (the
   wire-in proved gradient FIDELITY, never that MLX-GPU-gradient training reaches the same exact-d_seg
   TRAJECTORY).
2. **Resumable checkpoint/resume LANDED + VERIFIED** by a real kill+restart test: a death costs ≤ 1
   in-flight epoch. `tac.capstone_vq_nerv.checkpoint` + `experiments/run_capstone_resumable_curriculum.py`
   (detached daemon, marker-on-exit). 7 NO-FAKE tests incl. bit-identical resume==uninterrupted.
3. **Fast-approximate-gradient throughput is REAL but MODEST, not a regime change** (sister wire-in memo,
   corroborated here): MLX-GPU is ~1.2–1.5× faster at bs≤8 and a REGRESSION at bs=16 (Metal VJP
   memory-pressure cliff). The scorer fwd+bwd is >97% of the step and the backward dominates on BOTH
   backends. n48 fwd+bwd through the full scorer is so heavy it died (OOM/silent) in a clean micro-bench —
   confirming the backward is the wall, not a tractable bridge cost.
4. **mx.compile: NOT applied** (the per-step telemetry forward + the dynamic stage seg-loss make a single
   compile boundary non-trivial; the dominant cost is the autograd backward through the frozen torch/MLX
   scorer, which a compile boundary on the MLX render does not touch). Batching: the trainer already
   batches over `batch_size` (default 8 = the measured MLX-GPU sweet spot).
5. **The TRUE epoch budget is still the open empirical question** — the fixed recipe DESCENDS
   (0.507→0.066→0.0165→0.0120 over 3 stages at n8) but reaching the 5.6e-4 BASIN is unmeasured; PR95's full
   budget is 29,650 epochs. **Honest feasibility verdict below.**

---

## What was already built (search-first inventory — do NOT rebuild)

| Artifact | What it is | Status |
|---|---|---|
| `tac.mlx_pr95_port.mlx_gpu_score_bridge.MLXGpuScorerBridge` | MLX-GPU end-to-end score-aware loss (render→MLX eval_roundtrip+resize+yuv6→MLX SegNet/PoseNet→seg+pose→`mx.value_and_grad`→pixel cotangent). Drop-in sibling of `TorchScorerBridge`. | WIRED |
| `CapstoneTrainConfig.scorer_backend` + `--scorer-backend {torch_cpu_bridge,mlx_gpu}` + `--authority-recheck-every` | trainer + campaign flags; torch-CPU stays the AUTHORITY for every reported d_seg/d_pose. | WIRED |
| `.omx/research/mlx_scorer_port_drift_audit_20260611.md` | MLX-GPU vs torch-CPU drift: d_seg flip rate 1.2e-5 (243/19.66M, all boundary near-ties); pose drift 2.76e-4 (can exceed frontier d_pose ~3.4e-5 → absolute d_pose MUST be torch-CPU). | MEASURED |
| `.omx/research/mlx_gpu_scorer_training_wirein_20260611.md` | gradient cosine 0.99986 vs torch-CPU; throughput 1.2–1.5× at bs≤8, 0.61× regression at bs=16; 600-pair epoch ~13.9 min (mlx_gpu bs8) vs ~20.5 min (torch). | MEASURED |
| `MLX_METAL_GPU_ARCH=applegpu_g15` override | forces the non-NAX FP32-exact GPU conv path (243→2 d_seg flips). | WIRED |

## What I added (the two genuine gaps)

### Gap 1 — resumable per-epoch checkpoint/resume (LANDED + kill+restart VERIFIED)

`src/tac/capstone_vq_nerv/checkpoint.py` (+ 7 NO-FAKE tests). Captures the COMPLETE trainer state:
bundle param tree + VQ EMA buffers (`_codebook`/`_ema_cluster_size`/`_ema_w`) + weight-EMA shadow (the
export/inference bytes — the EMA non-negotiable) + PR95 optimizer state (Muon momentum + AdamW m/v + step)
+ curriculum position (stage idx + epoch-in-stage) + `_mech_step` (QAT/sigma/C1a RNG). Atomic write
(tmp + `os.replace`), `.safetensors` arrays + JSON manifest, done-marker-on-exit.

**The decisive test** `test_resume_is_bit_identical_to_uninterrupted_run`: train continuously vs
train→checkpoint→FRESH-trainer→load→continue — the final params + EMA shadow + optimizer momentum + VQ
codebook + exact d_seg are **bit-identical**. A stub that dropped any of (optimizer momentum / EMA shadow /
VQ codebook) would diverge after the first post-resume step and FAIL. `test_resume_d_seg_matches` proves the
exact d_seg matches to 1e-6.

**Driver-level kill+restart (real trainer, real scorer):** `run_capstone_resumable_curriculum.py` ran n8,
wrote a checkpoint at stage 0 epoch 1, was `kill -9`'d → the checkpoint survived (stage 0, epoch_in_stage 1)
with NO done-marker (correctly incomplete). Resume continues from that point. **A death now costs ≤ 1 epoch.**

### Gap 2 — descent-equivalence A/B harness

`experiments/measure_descent_equivalence.py`: builds two trainers from the SAME seed, drives the SAME steps
with the SAME permutations, arm A on `torch_cpu_bridge` (authority gradient), arm B on `mlx_gpu` (fast
gradient), and compares the **exact torch-CPU d_seg trajectory** of each arm (the eval authority is torch-CPU
for BOTH — only the per-step GRADIENT differs). This is the measurement the wire-in's single-batch cosine
could not answer: does the ~0.9999 gradient, accumulated over a descent, REACH the same basin?

---

## The throughput reality (measured + corroborated)

**My micro-benchmark** (`experiments/measure_capstone_backend_throughput.py`, n8 bs8, uncontended, 6
threads, the EXACT canonical trainer):

| backend | s/step (n8, 1 batch) | pairs/s | n600 projection (75 steps/ep) |
|---|---:|---:|---:|
| torch_cpu_bridge | **19.36** | 0.413 | **24.2 min/epoch** |
| mlx_gpu | **26.63** | 0.300 | 33.3 min/epoch (**0.73× — SLOWER**) |

**Why MLX-GPU is SLOWER at n8 but FASTER in the wire-in's bs=8 multi-batch run:** at n8 there is only ONE
batch per epoch, so the MLX-GPU pipeline never amortizes its launch/warmup across batches, AND it pays the
per-step gradient-free telemetry forward (MLX has no `has_aux`, so the seg/pose breakdown costs an extra
forward) — both fixed costs that the torch bridge gets for free from `.backward()`. The sister wire-in memo
measured MLX-GPU at **1.47× faster** at bs=8 *within a 600-pair epoch* (13.9 vs 20.5 min/epoch) where the
GPU stays warm across 75 batches. So the representative n600 number is the wire-in's **~14–20 min/epoch**,
and my n8 single-batch 24–33 min is a pessimistic cold-start floor. **The n48 fwd+bwd micro-bench DIED
(silent OOM/kill)** — 48 frames through the full EfficientNet-B2 + FastViT VJP is the Metal memory-pressure
cliff the wire-in flagged at bs=16; this confirms the BACKWARD is the irreducible wall on both backends.

**The decisive shared fact:** the scorer fwd+bwd is >97% of every step on BOTH backends, the backward
dominates, and it is irreducible on torch-CPU arm64 (no mkldnn → `_slow_conv2d` reference kernel). MLX-GPU
shaves ~30% off the epoch at the right batch size but does NOT change the regime: **n600 is minutes-per-epoch,
not seconds.**

## The TRUE epoch budget (task 1) — the binding open question

The frozen "n600 = ~5–6 months locally" estimate WAS measured on the THROTTLED recipe (which barely moved),
so it is invalid as a basin-reaching estimate. But the FIXED recipe's epoch budget to the basin is STILL
unmeasured: the recipe-gap audit (`pr95_seg_convergence_mechanism_and_recipe_gap_audit_20260611.md`) proves
the fixed recipe DESCENDS (0.507→0.066→0.0165→0.0120 over 3 stages at n8) but explicitly flags "whether the
corrected FULL curriculum reaches the 5.6e-4 BASIN is NOT YET MEASURED." PR95's canonical budget is 29,650
epochs.

**The arithmetic the operator needs:**
* At the representative ~17 min/epoch (mlx_gpu bs8 n600, mid-range of the two estimates), the n600 wall-clock
  is **~17 min × N_epochs**.
* If the compressed curriculum reaches the basin in **N = 2,000 epochs** → ~24 days continuous (FEASIBLE as
  a resumable multi-week run — the dream).
* If it needs the **full N = 29,650 epochs** → ~1 year continuous (NOT feasible locally; paid n600 required).
* If the descent-equivalence A/B below shows the fixed recipe reaches a low basin in **N ≈ few-hundred
  epochs** at n8 → the n600 compressed-curriculum budget is plausibly in the **days-to-weeks** range, and
  the resumable daemon makes that the dream-reachable path.

**This is why the resumable daemon is the load-bearing deliverable:** it converts "infeasible because a
multi-week run can't survive a death" into "feasible because a death costs ≤ 1 epoch." The epoch budget is
the remaining empirical unknown; the descent A/B is the first probe of it.

## The honest feasibility verdict

(verdict section — finalized after the descent A/B lands)
