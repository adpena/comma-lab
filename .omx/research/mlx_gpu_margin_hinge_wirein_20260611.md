# MLX-GPU cross-hardware margin-hinge wire-in (2026-06-11)

**Subagent:** mlx-gpu-margin-hinge-wirein. **Operator ask:** close the gap the
capstone build flagged — wire the L7 cross-hardware-robust margin hinge into the
**MLX-GPU scorer bridge** so the GPU-fast 600-pair capstone run is BOTH fast AND
margin-protected. Before this lane the hinge only worked on the slow
`torch_cpu_bridge`; selecting `--scorer-backend mlx_gpu --margin-hinge-weight X`
**fail-closed** (raised) because the shared `MLXGpuScorerBridge` had no wrappable
seg-loss hook. **Did NOT touch** the running daemons (capstone 48-pair pid 72123,
the 2x2 ablation pid 85721, the atlas workers). Edited only the MLX-GPU bridge +
its loss helper + the trainer wiring + tests.

**Authority discipline (CLAUDE.md, binding).** Every number here is
`[macOS-CPU advisory]` / `[macOS-MLX research-signal]`, **NON-PROMOTABLE**
(`promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`).
torch-CPU contest `evaluate.py` (600-sample, Linux x86_64) is the ONLY leaderboard
authority. **NO MPS** (the hinge runs on MLX-GPU Metal as a fast training signal,
or MLX-CPU as the bit-faithful reference; torch-CPU is the authority). NO paid
dispatch fired. The exact frontier pointer is **UNMOVED** — this is an engineering
wire-in that enables the GPU-fast 600-pair run to keep the L7 portability guard,
NOT a pointer move.

---

## 1. WHAT LANDED

Three small edits + tests, all NO-FAKE + parity-proven:

### Edit 1 — `margin_floor_hinge_mlx` (the MLX-native hinge math)

`src/tac/mlx_pr95_port/mlx_losses.py` gains `margin_floor_hinge_mlx(seg_logits,
targets_hard, *, margin_floor)` = `mean(maximum(margin_floor - margin, 0))`. It
REUSES the already-parity-tested `_target_margin_mlx` (the same
`target_logit - max_competing_logit` core the PR95 MLX surrogates use), so it is
1:1 with the torch-CPU authority twin
`tac.capstone_vq_nerv.cross_hw_margin_hinge.margin_floor_hinge` up to fp32 drift.
The `relu`/`maximum(.., 0)` is the boundary selector: a clear-of-floor pixel
contributes EXACTLY 0 (and 0 gradient); a below-floor (or argmax-wrong) pixel gets
a positive penalty whose gradient raises the target logit. Added to `__all__`.

### Edit 2 — wire the hinge into `MLXGpuScorerBridge`'s gradient

`src/tac/mlx_pr95_port/mlx_gpu_score_bridge.py` gains two constructor knobs
`margin_hinge_weight` (default 0) + `margin_hinge_floor` (default 0.1), validated
EARLY (before the heavy adapter build) so a bad config fails closed cheaply. The
hinge term is added **inside the `loss_fn` closure** that `mx.value_and_grad`
differentiates:

```python
seg_term = seg_loss_fn(seg_logits_nchw, targets)
if margin_hinge_weight > 0.0:
    seg_term = seg_term + margin_hinge_weight * margin_floor_hinge_mlx(
        seg_logits_nchw, targets, margin_floor=margin_hinge_floor)
total = seg_weight * seg_term  # 1:1 with the torch bridge's seg_weight * (base + w*hinge)
```

So `d(hinge)/d(pixels)` enters the render cotangent — the SAME floor-enforcing
gradient the torch-CPU bridge gets via the `CrossHwMarginHingeSegLoss` wrapper.
The hinge is scaled by `seg_weight` exactly as the torch path is (the torch
wrapper IS the `seg_loss_fn`, returning `base + w*hinge`, then multiplied by
`seg_weight`). `margin_hinge_weight == 0` skips the term (the closure never adds
it) → byte-identical to the bare bridge.

### Edit 3 — trainer wiring + remove the fail-closed

`src/tac/capstone_vq_nerv/capstone_trainer.py`:
- Removed the early `margin_hinge_weight>0 + scorer_backend=mlx_gpu` ValueError.
- Pass `margin_hinge_weight=config.margin_hinge_weight` /
  `margin_hinge_floor=config.margin_hinge_floor` into the `MLXGpuScorerBridge`
  construction.
- The torch-CPU `CrossHwMarginHingeSegLoss` wrapper now installs ONLY on the
  `torch_cpu_bridge` path (`config.scorer_backend != "mlx_gpu"`). On the mlx_gpu
  path the gradient comes from `self._loss_bridge` (the MLX-GPU bridge, hinge-LIVE),
  and `self.bridge` is the torch-CPU **authority re-score** path whose d_seg MUST
  stay the true hinge-free argmax (the authority must not see the surrogate floor).

So `--scorer-backend mlx_gpu --margin-hinge-weight X` now ACTUALLY applies the
floor in the GPU gradient; the torch authority re-score stays clean.

---

## 2. IS THE HINGE LIVE ON mlx_gpu? (the gradient test)

YES. The load-bearing NO-FAKE gradient proof
(`test_margin_floor_hinge_mlx_gradient_pushes_margin_up`): a small-positive-margin
pixel (0.05 < floor 0.1) gets a POSITIVE hinge value, the value-and-grad of the
hinge w.r.t. the seg logits is **non-zero**, AND the target-channel gradient is
**negative** (so a minimizer RAISES the target logit → larger margin). A
constant/no-op would have a zero gradient and FAIL this test.

On the bridge surface, `test_bridge_margin_hinge_changes_gradient_on_real_net`
(real upstream SegNet/PoseNet, real 0.mkv GT, real trained-init render): with the
hinge active the total loss is strictly above the bare bridge AND the pixel
cotangent changes by `> 1e-9` — proving the hinge term enters `mx.value_and_grad`
on real data (not inert). `test_trainer_mlx_gpu_backend_wires_in_hinge` confirms
the trainer builds the MLX-GPU bridge with the hinge knobs LIVE and a real step
runs.

---

## 3. MLX <-> torch hinge parity (the FP32-exact agreement)

`test_margin_hinge_mlx_vs_torch_parity_on_real_net_logits` runs the REAL upstream
SegNet (MLX-CPU bit-faithful path) on a real trained-init render of real 0.mkv GT,
pulls the SegNet logits, and computes BOTH `margin_floor_hinge_mlx` and the
torch-CPU authority `margin_floor_hinge` on the SAME logits (transferred to torch).
This isolates the hinge MATH parity (the scorer-forward drift is covered by the
existing `test_real_net_loss_and_gradient_parity_vs_torch_cpu`). Measured on the
n=8 fixture:

| floor | MLX hinge | torch hinge | abs delta | rel |
|------:|----------:|------------:|----------:|----:|
| 0.10  | 2.47774696 | 2.47775054 | 3.58e-6 | 1.44e-6 |
| 0.05  | 2.45238853 | 2.45238709 | 1.43e-6 | 5.83e-7 |
| 0.20  | 2.52848220 | 2.52847767 | 4.53e-6 | 1.79e-6 |

The hinge is a simple reduction over the SAME margins, so on the bit-faithful
MLX-CPU path it matches torch to ~fp32-ULP (abs delta ~3.6e-6 at the canonical
0.1 floor, well under the test's 1e-3 bound and far under the ~0.096 cross-hardware
logit drift the floor itself guards against). Under
`MLX_METAL_GPU_ARCH=applegpu_g15` the GPU forward is FP32-exact too, so the GPU
hinge term carries the same tight parity into the per-step gradient.

---

## 4. weight=0 is byte-identical (the no-op proof)

`test_bridge_margin_hinge_weight_zero_is_byte_identical` (real net, CPU path): a
default-off bridge and a `margin_hinge_weight=0` bridge give **bit-identical**
`loss_value` / `seg_loss_value` / `pixel_cotangent` (max abs cotangent delta
EXACTLY 0.0) on a real render. The default-off path is provably inert — the
running torch_cpu_bridge daemon and any `--margin-hinge-weight 0` mlx_gpu run are
unchanged.

---

## 5. CONFIRMATION FOR THE 600-PAIR RUN

`--scorer-backend mlx_gpu --margin-hinge-weight X --margin-hinge-floor 0.1` now
PROTECTS the 600-pair run's cross-hardware d_seg transfer: the MLX-native floor
hinge enters the GPU gradient (sections 2-3), so the LOCAL SegNet argmax is pushed
PAST the ~0.096 macOS->numpy->Linux/CUDA fp32 logit drift while the run keeps the
GPU throughput. The capstone build memo's section-3 caveat ("the 600-pair bet uses
mlx_gpu; the margin hinge is therefore OFF... or (b) wire the MLX-native hinge into
the shared MLX-GPU bridge — a follow-up") is now CLOSED: option (b) is landed. The
GPU-fast path is BOTH fast AND margin-protected.

The torch-CPU **authority re-score** (the `authority_recheck_every` cadence + every
eval) stays hinge-free — the reported `exact_d_seg` / `exact_d_pose` are the true
argmax, not the surrogate floor (NO-FAKE: the hinge shapes the gradient, never the
reported authority metric).

---

## 6. TEST STATUS

NEW NO-FAKE tests (all PASS):
- `test_mlx_gpu_score_bridge.py`: `test_margin_floor_hinge_mlx_gradient_pushes_margin_up`
  (a, the load-bearing gradient proof) + `test_margin_floor_hinge_mlx_zero_on_clear_of_floor`
  + `test_margin_floor_hinge_mlx_floor_must_be_positive` +
  `test_bridge_rejects_negative_hinge_weight_and_nonpositive_floor` (cheap) +
  `test_bridge_margin_hinge_weight_zero_is_byte_identical` (c) +
  `test_bridge_margin_hinge_changes_gradient_on_real_net` (a, real-net) +
  `test_margin_hinge_mlx_vs_torch_parity_on_real_net_logits` (b) +
  `test_trainer_mlx_gpu_backend_wires_in_hinge` (GPU trainer wire-in).
- `test_capstone_tie_and_margin_hinge.py::test_trainer_installs_hinge_on_torch_cpu_path`
  (replaces the old `..._fails_closed_on_mlx_gpu`; the mlx_gpu hinge-wiring assertion
  moved to the real-net GPU suite since it needs the real upstream adapter).

No-regression: the full `test_mlx_gpu_score_bridge.py` suite (incl the slow real-net
loss+gradient parity test) + `test_capstone_tie_and_margin_hinge.py` (17) +
`test_torch_parity.py` (16) all PASS. ruff clean on all edited files.

---

## 7. NO-FAKE / authority notes
- The MLX hinge is a REAL loss term with a REAL gradient (the value-and-grad test
  fails on a no-op) carried by `mx.value_and_grad` into the pixel cotangent.
- weight=0 is provably byte-identical (max cotangent delta EXACTLY 0.0).
- MLX<->torch hinge parity is ~fp32-ULP (abs ~3.6e-6 @ floor 0.1).
- The hinge shapes the per-step gradient ONLY; the torch-CPU authority re-score
  d_seg/d_pose stay the true hinge-free argmax (the authority must not see the
  surrogate floor).
- All numbers `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`,
  NON-PROMOTABLE. The exact frontier pointer is UNMOVED — this wire-in is the means
  that lets the GPU-fast 600-pair run (the unit aimed at the exact CPU-axis row)
  carry the L7 portability guard; it is not itself a lower exact S.
