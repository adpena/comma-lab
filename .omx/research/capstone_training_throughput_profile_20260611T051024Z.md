# Capstone training-throughput profile + optimization (2026-06-11)

**Source:** operator directive — PROFILE + OPTIMIZE the capstone training loop so carrier A/B
iterations are 2–5× faster. Lane: SCORER-BRIDGE + EVAL + throughput-instrumentation
(`src/tac/mlx_pr95_port/score_bridge.py`, `src/tac/score_aware_loop/targets.py`, a new profiling
tool). DID NOT touch the running `b20_n48` d_seg daemon (pid 34195) or the sister-held builder files
(`vq_nerv_bundle.py` / `export.py` / `numpy_reference.py` / `inflate.py` / `capstone_trainer.py`).

**Axis:** `[macOS-CPU advisory]` (M5 Max, torch 2.11 arm64, torch-CPU frozen scorer = trusted authority
per CLAUDE.md "local CPU + MLX GPU good"; MPS NEVER touched). $0, local. NOT a pointer move.

**Profile JSON:** `.omx/research/capstone_training_throughput_profile_20260611T051024Z.json`
**Tool:** `tools/profile_capstone_training_throughput.py` (reusable, real-scorer, no synthetic fixtures).

---

## 1. THE PROFILE BREAKDOWN (one training step, fwd+bwd, N=16 pairs, base_ch=20)

| Section | ms | % of step |
|---|---:|---:|
| **SegNet (EfficientNet-B2) forward** | 4,697 | 33% |
| **PoseNet (FastViT) forward** | 1,225 | 9% |
| **Backward (SegNet+PoseNet, frozen)** | 8,094 | **57%** |
| eval_roundtrip (bicubic-up 874×1164 + STE-uint8) | 197 | 1.4% |
| preprocess_input | 28 | 0.2% |
| MLX render + host sync | 40 | 0.3% |
| numpy copy render→leaf / permute.contiguous | ~0 | ~0% |
| **TOTAL step** | **14,280** | 100% |

**`scorer_fwd_bwd_fraction = 0.98` — the frozen torch-CPU scorer fwd+bwd is >97% of the wall-clock.**
The MLX render, the numpy copy, the eval_roundtrip resize, and the host syncs are <3% COMBINED. The
backward through SegNet is ~1.7× its forward (the dominant single sink). On the daemon config (N=48,
base_ch=20) the same shape holds: SegNet fwd ≈ 12.1s, PoseNet fwd ≈ 3.0s, backward ≈ 25.2s, total ≈ 41s.

### The root cause (the irreducibility finding — NO-FAKE)
`torch.profiler` on the SegNet forward: **`aten::_slow_conv2d_forward` = 85% of SegNet self-CPU**, with
**14,886 conv calls** for 16 frames. EfficientNet-B2 is depthwise-separable; **on Apple-Silicon arm64
torch has NO mkldnn and NO MKL** (`torch.backends.mkldnn.is_available()` and `mkl.is_available()` are
BOTH False), so the depthwise convs dispatch to the *naive reference kernel* (`_slow_conv2d`) — there is
no optimized grouped-conv path. `channels_last` does NOT help that path (verified: identical numerics,
identical wall-clock). **The scorer cost is therefore largely irreducible on torch-CPU.** A clean 2–5×
on the CURRENT daemon config cannot come from the bridge alone — that is the honest finding, not a
shortfall to paper over.

---

## 2. THE ONE LARGE NUMERICS-PRESERVING LEVER: scorer-batch amortization

The `_slow_conv2d` per-call FIXED cost amortizes over a larger scorer batch. Measured per-frame SegNet
forward:

| pairs / frames | SegNet fwd | **per-frame** |
|---:|---:|---:|
| 1 / 2 | 1,018 ms | **509 ms** |
| 4 / 8 | 2,625 ms | 328 ms |
| 8 / 16 | 3,789 ms | 237 ms |
| 16 / 32 | 4,718 ms | **147 ms** |

**3.5× per-frame speedup from batch=1 → batch=16.** End-to-end (full fwd+bwd step over a 48-pair epoch):
batch=8 → 60.9s/epoch, batch=16 → 46.3s (1.31×), batch=24 → 43.2s (1.41×). **This is the 2–5× lever** —
but it lives in the TRAINER STEP LANE (the `batch_size` knob changes the optimizer step granularity, so
it is an optimization-trajectory change, not a pure measurement). **The running `b20_n48` daemon ALREADY
uses `batch_size=n=48` (full-set single batch) — it is already at maximum amortization, which is WHY it
is slow per-step but optimal per-pair.** So for the current daemon the batch lever is exhausted; the
proposal below matters for runs that currently use a small `batch_size`.

Numerics check: d_seg (argmax-disagreement rate) is **bit-identical** across batch sizes; d_pose mean is
numerically equivalent to ~5 sig figs (float reassociation in the mean + bicubic only). The MLX render is
batch-composition-independent (max abs diff 0.0 for the same pair indices rendered in different batches).

---

## 3. WHAT I LANDED (bridge / targets / profiling lane — byte-identical numerics)

### (a) `tools/profile_capstone_training_throughput.py` — the reusable profiler
Real frozen scorer + real GT targets + real bundle/bridge over a seeded slice of the real video (NO
synthetic fixtures — the `_slow_conv2d` amortization degenerates on toy inputs). Emits: per-section step
breakdown, eval-pass cost (separate vs fused), SegNet batch-amortization curve, full-step torch-thread
sweep (spawns a fresh process per thread count so the setting takes before torch parallel init), and the
machine state (`mkldnn`/`mkl` availability). JSON out for the dashboard.

### (b) `TorchScorerBridge.configure_torch_cpu_threads(num_threads=None)` — measured thread pin
Default resolves to `min(perf_core_count, 8)` (the M5 Max has 6 perf cores; torch's default 6 is already
near-optimal). The full-step thread sweep shows wall-clock flat from ~6–10 threads and DEGRADING at ≥14
(cross-core cache thrash on the slow-conv path: 14–18 threads are SLOWER than 6). **Numerics-preserving:**
thread count changes only sub-ULP reduction order; d_seg is **bit-identical across thread counts**
(verified 2/4/6/8). A daemon should call this once at startup. NOTE: under contention (another daemon
holding ~2.5 cores) FEWER scorer threads win (less fighting); pin to perf-core count only when running
alone.

### (c) `TorchScorerBridge.fused_d_seg_d_pose(render, idx)` — one-preprocess eval
Runs SegNet AND PoseNet over ONE shared `_eval_preprocess` (resize → eval_roundtrip → NHWC → split),
avoiding the second render + preprocess the separate `exact_d_seg`/`exact_d_pose` calls pay. Returns
`(d_seg, d_pose)` **BIT-IDENTICAL** to the separate calls (proven: d_seg `0.5072727203369141` ==, d_pose
abs diff `0.0`). Uses `torch.inference_mode` (no autograd graph). Eval-pass speedup is small (~1.02×)
because the avoided preprocess is tiny relative to the SegNet forward — but it is real, free, and correct.

**Tests:** `src/tac/mlx_pr95_port/tests/test_score_bridge_throughput.py` — 11 NO-FAKE behavioral tests
(fused==separate bit-identical on d_seg/d_pose, with/without eval_roundtrip, partial-batch, fail-closed
when pose disabled, thread-count d_seg invariance, `_eval_preprocess` parity with the `exact_d_seg`
preamble). Full `mlx_pr95_port` suite (73 tests) green — no regressions.

**Measured speedup proof (byte-identical numerics):** the bridge additions do not change any scored value;
the speedup they directly deliver is the fused-eval ~1.02× + the thread pin (avoids the ≥14-thread
regression). The DECISIVE epochs/sec lever is the batch proposal in §4, which is the trainer lane.

---

## 4. PROPOSAL FOR THE TRAINER-STEP LANE (orchestrator to apply — a builder holds `capstone_trainer.py`)

1. **Decouple the EVAL batch from the TRAIN batch** (the [D1] cost). `CapstoneTrainer.exact_d_seg` /
   `mean_d_pose` loop `range(0, n_pairs, cfg.batch_size)`. When a run uses a small training `batch_size`,
   the eval pays the small-batch `_slow_conv2d` penalty (509 ms/frame at batch=1 vs 147 at batch=16). Add
   a `cfg.eval_batch_size` (default = `n_pairs`, i.e. one big batch) and use it in BOTH eval loops. d_seg
   is bit-identical; d_pose is numerically equivalent (~5 sig figs). **For a small-batch run this alone is
   a ~3× eval speedup.** (The `b20_n48` daemon uses batch=48 so eval is already maxed — this helps the
   small-batch carrier-A/B configs.)

2. **Use `bridge.fused_d_seg_d_pose` in the eval loops** instead of separate `exact_d_seg` + `mean_d_pose`
   passes. Today the trainer renders + preprocesses the whole set TWICE per eval (once for d_seg, once for
   d_pose). One fused pass halves the render + preprocess work (small but free). Combine with (1): one
   fused loop at `eval_batch_size = n_pairs`.

3. **Fix the [D1] per-stage re-eval** (8 extra full evals in the 8-stage curriculum): `run_stage_epochs`
   computes `d_seg_initial`/`d_pose_initial` over ALL pairs at the start of EVERY stage. That is 8 extra
   full SegNet+PoseNet passes per curriculum run. Pass the prior stage's final d_seg/d_pose forward as the
   next stage's `initial` (they are continuous across the stage boundary — the weights carry over) instead
   of re-measuring. Saves ~7 full evals per curriculum (each ~the cost of an epoch's eval).

4. **`call configure_torch_cpu_threads()` once at daemon startup** (in `run_capstone_campaign.py` `main()`
   before the first forward). Pins to perf-core count; prevents the ≥14-thread regression if a future host
   has more cores and torch picks a large default.

5. **Raise the default training `batch_size`** for small-N configs toward `n` (the daemon already does
   this). For configs that currently run `batch_size=8`, moving to 16–24 is a measured 1.3–1.4×
   epochs/sec at a changed-but-defensible optimization trajectory (larger batch = lower-variance gradient;
   PR95 itself batches the full set). This is the operator's "2–5× faster iterations" lever for any
   small-batch carrier-A/B run.

The hard ceiling stays: the SegNet+PoseNet fwd+bwd is >97% of the step and is irreducible on torch-CPU
arm64. To go materially past the batch+eval wins, the only paths are (a) a CUDA/T4 scorer (the contest
axis — the only pointer-moving step anyway), or (b) an MLX-GPU SegNet forward (blocked by the
second-order-autograd NaN trap that forced the learnable-head surrogate; out of scope here).

---

## 5. NO-FAKE / wire-in ledger

* NO scored value changes — fused==separate proven bit-identical; thread count proven d_seg-invariant;
  the profiler only MEASURES.
* eval_roundtrip non-negotiable preserved (the fused path applies the SAME bicubic-up/bilinear-down/STE
  ladder via the shared `_eval_preprocess`; `test_fused_matches_separate_without_eval_roundtrip` covers
  the clamp-only branch too).
* 6-hook wire-in: #1 sensitivity-map = N/A (throughput, not score-axis); #2 Pareto = N/A; #3 bit-allocator
  = N/A; #4 cathedral autopilot = N/A (advisory profiler); #5 continual-learning = the profile JSON is the
  reusable observability artifact (Max-observability non-negotiable); #6 probe-disambiguator = the profiler
  IS the throughput disambiguator (batch vs thread vs eval levers, each measured).
* DID NOT touch the running daemon (pid 34195) or the sister-held builder files. The serializer guards any
  accidental collision.
