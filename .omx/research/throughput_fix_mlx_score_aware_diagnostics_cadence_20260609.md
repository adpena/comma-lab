# MLX score-aware train_step throughput fix — diagnostics cadence + vectorized flood-fill

**Lane:** `lane_throughput_fix_mlx_score_aware_20260609`
**Agent:** THROUGHPUT-FIX (solo)
**Date:** 2026-06-09
**Authority:** `[macOS-MLX research-signal]` — `score_claim=false`, `promotion_eligible=false`, `promotable=false`. MLX timing is hardware-advisory ONLY; it is never a contest score. The contest score is exact-eval'd later on byte-closed `archive.zip` bytes via `upstream/evaluate.py`.

## Operator problem

The ~229K-param HiNeRV decoder trains pathologically slowly on local MLX (~8 s/epoch for a model that should be ~1-2 s/epoch). Operator verdict: "that time is insane... likely bugs or config... engineer to optimize." Task: make the inner training FAST **without changing the training MATH** (the adapter is shared by sister substrates z7/z8/dreamer/etc — loss + gradients must be preserved bit-for-bit so sisters are unaffected).

## Confirmed root cause (measured, not assumed)

The hot path is `MlxScoreAwareAdapter.train_step` in
`src/tac/substrates/_shared/mlx_score_aware/adapter.py` (three variants:
canonical `train_step`, `_train_step_pact_muon_adamw` (the DEFAULT optimizer
path), `_train_step_pr95_faithful_curriculum`).

`cProfile` of the DEFAULT path at the 229K config (`decoder_channels=(36,30,23,17,14,11,8)`, 228,903 params), batch 16, on REAL `upstream/videos/0.mkv` pairs:

| Cost | Per step | Share | Gateable? |
|---|---|---|---|
| `_score_aware_loss_part_metrics` (called **3×/step** — full score-aware loss RECOMPUTE for telemetry) | 0.468s | ~39% | **YES (observability-only)** |
| `_assert_mlx_loss_and_gradients_finite` (first `mx.eval` after value_and_grad — absorbs the fwd+bwd realization + a per-leaf isfinite sweep) | 0.385s | ~32% | NO (safety: refuses NaN-poisoned optimizer state) |
| `_train_student_heads` (a SECOND `mx.value_and_grad` that re-decodes the renderer forward + trains the student heads) | 0.194s | ~16% | NO (load-bearing distillation math) |
| `_capture_parameter_trace_leaves` (clones every param leaf each step) + group-norm/delta/weight traces + `_accumulate_decoder_weight_gradient_saliency` (per-group `mx.eval` sync) | rest | ~13% | **YES (observability-only)** |

Reference floor: the BARE renderer fwd+bwd (recon-only) is **48 ms/step** at batch 16; the full score-aware step is ~120 ms/step (gated) because the score-aware path inherently does the renderer forward ~2-3× (main loss + student head) plus the finite-guard sync. The pre-fix step paid an EXTRA ~3 full score-aware forward recomputes + a param-tree clone + a forced grad-tree sync **every step** purely for telemetry that does NOT feed the gradient.

**`mx.compile` count across the 9700-LOC adapter = 0** (no kernel fusion) — confirmed. On the bare renderer fwd+bwd, `mx.compile` gives only ~1.18x (48 → 40.5 ms); the renderer is already well-fused by MLX (conv-heavy). See "mx.compile / fp16 evaluation" below for why it was NOT wired.

The pure-Python per-pixel flood-fill (`_receiver_surface_worst_connected_region_margin_p50`, ~196K Python iterations per call at 384×512) is real but **behind `scorer_space_step_guard_enabled` which defaults to `False`**, so it is NOT on the default hot path. It was vectorized anyway (operator priority #3; bit-exact, see below).

## The fix (math-preserving)

### 1. Per-step diagnostics cadence gate (the dominant win)

New constructor kwarg `diagnostics_every_n_steps: int = 1`. At the default (1) the
adapter is **byte-identical** to the pre-fix version (every step runs the full
sampled diagnostics; no new metric keys leak into the returned dict). When set
`>1`, the sampled observability block (param-trace clone, the 3× score-aware
loss-part recompute, group-norm/delta/weight traces, decoder-weight gradient
saliency) runs only on a cadence (always on the first step + any
`request_diagnostics_flush()` boundary). The HOT step becomes value_and_grad →
finite guard → grad-clip → optimizer.update → student-head train → one `mx.eval`.

**Safety invariant (verified):** the gated diagnostics are observability-only on
the default path; they do NOT feed the gradient, the optimizer update, or the
returned `total` loss. When `scorer_space_step_guard_enabled=True`, the
guard-FEEDING diagnostics (pre/post score-aware loss-part metrics + receiver
surface snapshot + the pre-update param trace the guard restores from) are
load-bearing for the guard's reject/restore decision and therefore ALWAYS run
every step regardless of cadence (`need_guard_inputs = diag_active or guard_on`);
only the non-guard-feeding telemetry is sampled. The gate cannot change the
training math; it only samples observability less often.

Applied uniformly to all three `train_step` variants.

### 2. Vectorized worst-connected-region flood-fill (bit-exact)

`_receiver_surface_worst_connected_region_margin_p50` replaced its pure-Python
per-pixel double-loop flood-fill with a `scipy.ndimage.label` (4-connectivity =
von Neumann neighborhood, matching the reference up/down/left/right exactly)
pass + `scipy.ndimage.labeled_comprehension` applying the EXACT
`np.percentile(v, 50.0)` per component. The preserved pure-Python reference
oracle is `_receiver_surface_worst_connected_region_margin_p50_reference`. The
raster-order strict-`<` minimum tie-break (first-encountered wins on equal p50)
is reproduced exactly via the per-component first-pixel flat index.

This path is double-gated (guard-off by default + cadence-gated), so it rarely
runs; the vectorization eliminates the pathological 196K-iteration per-pixel
Python stall structurally and is bit-exact.

## Measured results — `b1_large_batch_timing_sweep.v1`

Tool: `tools/b1_large_batch_timing_sweep.py` (companion of
`tools/timing_smoke_hinerv_pr95_family.py`). Emits structured JSON to
`.omx/research/b1_large_batch_timing_sweep_<utc>.json`. BEFORE == cadence 1
(byte-identical to pre-fix); AFTER == cadence 50. 229K config, REAL contest
pairs, 5 epochs.

| batch_pairs | s/epoch BEFORE | s/epoch AFTER | speedup | math parity (max abs Δloss) | peak mem | seg proxy Δ | pose proxy Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 7.68 | 4.32 | **1.78×** | **0.0 (exact)** | 3.57 GB | -0.40 | -83.99 |
| 32 | 7.39 | 4.40 | **1.68×** | **0.0 (exact)** | 6.15 GB | -0.09 | -32.47 |
| 64 | 7.52 | 4.56 | **1.65×** | **0.0 (exact)** | 11.29 GB | -0.05 | -12.35 |

(JSON artifact: `.omx/research/b1_large_batch_timing_sweep_20260609T052231Z.json`)

### Math-parity evidence (the NO-FAKE proof)

For EVERY batch the per-step `total` loss trajectory under cadence 1 vs cadence
50 is **bit-identical (max abs diff = 0.0)**. The diagnostics gating provably
does not change the training math. This is enforced as a regression test
(`test_loss_trajectory_identical_*`, `test_guard_on_loss_trajectory_identical_across_cadences`).

### Honest speedup ceiling

The gating delivers a robust **~1.65-1.78×**, NOT the 4-8× the operator
hypothesized. The 4-8× target assumed the diagnostics were the WHOLE overhead;
measurement proved the remaining cost is the renderer fwd+bwd + the student-head
SECOND value_and_grad + the finite-guard sync — all load-bearing score-aware
math that cannot be gated without changing the result. The bare-renderer floor
is ~48 ms/step (~1.8 s/epoch at batch 16); the score-aware path is inherently
~2.5× that because it runs the renderer forward 2-3× (main loss + student head).
Reporting the REAL measured 1.7× per CLAUDE.md "NO FAKE IMPLEMENTATIONS".

### Recommended batch schedule (SPEED × PROXY-MOVEMENT, not speed alone)

The sweep's `recommended_batch_schedule`: **early_search_batch=16, qat_final_batch=64.**
Grounded in proxy-movement: batch 16 moves the proxy score MOST per epoch
(pose Δ=-84, seg Δ=-0.40 — more optimizer updates per epoch) → best for early
chamber search; batch 64 has the fewest updates per epoch (pose Δ=-12) → least
proxy movement, but is the right choice for stable full-batch QAT/final
continuation. This empirically confirms the operator's hypothesis (medium batch
for search, full/large for QAT/final).

NOTE: the operator's batch600=109.67 s/epoch catastrophe was not measured here
(batch 64 already hits 11.3 GB; batch 600 would approach the memory ceiling). At
batch 600 the diagnostics gate helps proportionally MORE (the 3 telemetry
recomputes are 3 full 600-pair forwards), but the dominant cost at that batch is
the single huge forward+backward, not the diagnostics. The recommended schedule
deliberately tops out at the largest non-OOM batch.

## mx.compile / fp16 evaluation (operator priorities #2 / #4) — measured, deferred with rationale

- **mx.compile**: on the bare renderer fwd+bwd it is ~1.18× (48 → 40.5 ms). The
  score-aware `value_and_grad` closes over the bundle, the gradient-free
  teacher caches, dynamic per-stage loss weights, and the student heads, with
  variable end-chunk batch shapes. Wiring `mx.compile` through that shared,
  9700-LOC, sister-shared loss path for a ~1.2× gain is HIGH-RISK for the
  byte-stability invariant the operator made non-negotiable. DEFERRED as a
  measured follow-up (the renderer-local forward is the only clean compile
  target; gain is small because MLX already fuses the conv-heavy renderer).
- **fp16/bf16**: NOT applied. The score-relevant outputs (SegNet argmax, PoseNet
  pose) are precision-sensitive; fp16 would risk silently moving the proxy
  parts. Per the operator ("measure, don't assume") this needs a paired
  proxy-parts ablation before adoption; DEFERRED.

## 0-regression proof

- New tests: `src/tac/substrates/_shared/mlx_score_aware/tests/test_diagnostics_cadence_throughput.py` — **20 tests pass** (loss-parity, guard-on safety, sampler-actually-samples, default byte-stability, flood-fill bit-identity across realistic/fragmented/multibatch/tie/even-median).
- Sister regression suites pass with **0 NEW failures**:
  - `test_mlx_score_aware.py` (49 passed), `test_wave_n11_stabilizer.py` + `test_scorer_binding.py` (49 passed).
  - `test_z7_real_hinton_teacher_pose_axis.py` + dreamer `test_dreamer_v3_real_hinton_teacher_pose_axis.py`: 27 passed, 2 failed — **both failures confirmed PRE-EXISTING on the baseline adapter** (a `total`-composition assertion + a `/tmp`-guard env issue), reproduced identically via `git stash` of the adapter change.
- `ruff check` clean on all 3 files.

## 6-hook wire-in (CLAUDE.md "Subagent coherence-by-default" / Catalog #125)

1. **Sensitivity-map** — N/A (no per-axis byte-savings; this is a training-throughput fix, not a score-byte lane).
2. **Pareto constraint** — N/A (non-binding; observability-cadence does not change archive bytes or the score polytope).
3. **Bit-allocator hook** — N/A (no per-tensor importance change).
4. **Cathedral autopilot dispatch hook** — N/A (not archive-deployable; advisory training-loop speed).
5. **Continual-learning posterior** — this memo + the `b1_large_batch_timing_sweep.v1` JSON are the empirical anchors; the `recommended_batch_schedule` is the consumable signal for the long-training harness batch policy.
6. **Probe-disambiguator** — N/A (no competing interpretations; math-parity is exactly 0.0, not a judgment call).

## Reactivation / follow-ups

1. `mx.compile` on the renderer-local forward only (clean closure, ~1.2× measured) if wall-clock becomes binding again.
2. fp16/bf16 renderer forward gated by a paired proxy-parts ablation (must not move SegNet argmax / PoseNet pose).
3. Reduce the student-head SECOND renderer forward by sharing the decoded frames with the main loss (load-bearing; needs careful gradient-graph design to stay math-exact).
4. Rust `runtime-rs` connected-components primitive per CLAUDE.md "Native eval-time runtime discipline" if the guard-on flood-fill ever becomes a default-path bottleneck (scipy is the faster-than-Python first step landed here).
