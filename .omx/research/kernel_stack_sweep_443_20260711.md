# Kernel-stack + training-hot-path optimization sweep (task #443) — MEASURED ledger, 2026-07-11

**Agent:** kernel_sweep_443 (subagent, Fable) · **Scope:** full MLX + Metal kernel stack AND (operator
scope-extension mid-task) everything that runs at training time — orchestration loop, telemetry, checkpoint
IO, verdict threading, caches — with Agner Fog's optimization discipline applied per candidate.
**Axis:** every number here `[macOS-CPU/GPU advisory · NON-PROMOTABLE]`, $0 local, bounded benches on an
idle machine (no live trainer; NO training launches; governor untouched). **Pointer 0.19108282
[contest-CPU] UNMOVED** — speed is MEANS (ΔS=0, lexicographic-secondary); no score claim anywhere below.

**STORES CONSULTED:** #356 `whole_step_megakernel_356_20260711.md` + law
`witness_fp_reorder_transform_bit_identity_wall_v1` (the binding wall — NOT re-litigated) · #306
`per_lever_compute_audit_20260705.md` (where the time goes) · #355 `v7_compute_exploitation_audit_20260708.md`
(lever coverage) · L70 (fused-R bit-identity) · L51 (memory envelope) · `spec_v9_cgauge` compiled TODAY
(launch-argv ground truth) · `metal_persistence_pool` / `persistence_topology_loss` /
`mlx_compile_step` / `scorer_throughput_gate` sources · the #205 OOM forbidden-pattern (verdict-batch)
· Agner Fog, *Optimizing software in C++* (operator-directed reference; principles cited per row).

---

## 0. Headline (honest)

**The stack is already at its measured bit-identical speed frontier.** Every BUILD-NOW candidate I
sized DIED under measurement before build (Fog rule #1 — measure before optimizing — applied to my own
candidate list). The V9·CGauge launch argv (compiled today) carries every surviving lever ON:
`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (~17×) + `TAC_MLX_CUSTOM_PERSISTENCE_POOL=1` (now BUILT, see §3.4)
+ `--fused-r-kernel` + `--cache-gt-skeleton` + `--async-verdict --verdict-batch 32 --verdict-pairs 0`.
The one order-of-magnitude lever remains the known, trajectory-affecting `--micro-batch-pairs`
(QUEUE, A/B-owed, binding blocker named in row 1). **Nothing was built this task — building any
measured-below-noise candidate would be orphan/goldplating theater** (deliverable-2 honesty clause).

## 1. THE RANKED LEDGER

Bit-identity classes: **BI** = exactly bit-identical · **LEVER** = changes step/verdict numerics
(trajectory-affecting; A/B-owed) · **SCHED** = score-neutral scheduling/IO (zero numeric change).
Win labels: **MEASURED** (benched today or prior anchor) / **DERIVED** / **SPECULATIVE**.

| # | site / chain | technique | class | projected win | memory | build cost | VERDICT |
|---|---|---|---|---|---|---|---|
| 1 | per-pair serial accum loop (600 value_and_grad/ep) | `--micro-batch-pairs` B∈{2,4} batched twin | **LEVER** (batched fp reduction; #410 class) | **MEASURED 2–4× step** (#293 anchors) ⇒ ~6.5d → ~2–3d — the ONLY order-of-magnitude lever | n6 +12 GiB@B=4; n600 RSS re-measure owed (waterfill) | routing build + n600 A/B | **QUEUE (A/B-owed).** V9-compat checked TODAY: `--seed-islands` routed (#293), wa-island routed (#313); **binding blocker = `--logit-adjust-loss-tau 1.0`** (`_validate_logit_adjust_compat` fail-closes; #355 lever 4). Named build item: route logit-adjust into the batched twin, then the n600 d_seg A/B under the RSS waterfill. Not mine to fire. |
| 2 | async-verdict worker (in-process thread: numpy render + CPU-torch scorers) | cap `torch.set_num_threads` (6→4 or 2) + pthread QoS UTILITY on the worker | nt-cap: **BI-verified at nt≥2** (MEASURED today, §3.1 — argmax/margin/pose hashes EQUAL at nt∈{2,4,6}; nt=1 margin DIVERGES) · QoS: **SCHED** | **DOWNGRADED by today's bench (§3.2): MLX GPU step is IMMUNE to CPU co-load (407→409–414 ms, ≤2%, noise)** ⇒ win 0–5 s/ep SPECULATIVE (the #306 "+12% verdict contention" attribution is NOT reproduced at the GPU-step level; residual suspect = GIL/host-side, unmeasurable without a live run) | 0 | ~100–200 LOC + DSL lever + startup parity gate | **QUEUE behind a live-run A/A cadence measurement.** Do NOT build blind — an unproven knob is an orphan. Fog: measure first; my own top candidate died here. |
| 3 | pred-side soft-skeleton (clDice, gradient path) — 17 min/max pools/pair pure-MLX 9-shift | fused 3×3 min/max Metal kernel + explicit-order gather-formulated VJP (no atomics; the fused-R pattern) | BI-candidate (fwd exact selection; VJP must match `mx.min/max(stack)` TIE semantics bit-for-bit — skeleton fields are full of exact 0.0 ties) | **MEASURED bound (§3.3): whole persistence loss fwd+bwd = 4.72 ms/pair ≈ 2.8 s/ep** at V9 shapes (K=1 class, sg cached) ⇒ kernel win ≤ ~2 s/ep (<1% of a 227 s/ep tau epoch) | ~0 | few hundred LOC + tie-parity gate | **REJECT as BUILD-NOW** (win <1%, parity risk nonzero). RE-OPEN only if `--persistence-classes` grows to ≥3–4 (win scales ~linearly in K). |
| 4 | GT-side persistence constants (recall weight `a`, `a·g`, `rden`, `present` — epoch-invariant, stop-grad) | per-pair cache (extend the #260 `CacheGtSkeleton` pattern) | **BI** (exact reuse) | DERIVED ~0.2 s/ep (density = 4 mean pools, already Metal-fused → µs-scale) | ~1.9 GB @ n600 K=1 | small | **REJECT (below noise).** Measured before built — correctly killed. |
| 5 | cf-feats cache (resident ~41–44 GiB fp32) | fp16 halving (#296) | **LEVER** (feature quantization) | 0 wall-clock (memory-save only; fp32 is the MLX sweet spot per #306 §4.5) | −22 GiB | bank-6 chain (build→d_seg-impact→memory→review) | **QUEUE** (#296 as charted). Headroom does not bind (run-1 RSS ~18 GB/128; L51 envelope ample) — memory-for-speed has no binding customer today. |
| 6 | `make_lane_band_compose_fn` — `mx.array(cov)` + `mx.array(margin)` host→GPU conversion PER CALL per pair per step | hoist invariant conversions to per-code cache (Fog: table lookup over recomputation; hoist loop invariants) | **BI** (same bits, same op order) | DERIVED ≤ tens of ms/ep ((384×512) fp32 ≈ 768 KB/upload, µs-scale vs the 250 ms/pair GPU step) | +0.5 GB | tiny | **REJECT (below noise at current shapes).** Recorded so nobody re-derives it. |
| 7 | accum loop — 2× `mx.eval` per pair (`eval(loss,grads)` + `eval(accum)`) | merge to one eval | BI (eval placement ≠ graph change) | DERIVED ≤0.5 s/ep (µs–ms per eval) | risk: UNBOUNDS the lazy-graph window | trivial | **REJECT — eval placement is LOAD-BEARING** (the #205 OOM cure bounds the lazy graph per pair). Do not touch. |
| 8 | Python orchestration per pair (tree_map lambdas, list builds, getattr, np.median spike bookkeeping) | micro-opts / preallocation (Fog: avoid allocation in hot loops) | SCHED | **MEASURED (§3.5): lazy-graph CONSTRUCTION = 0.12 ms/call = 0.6% of build+eval on the representative closure** ⇒ whole-loop Python overhead bounded ~1–4 s/ep worst case | 0 | — | **REJECT.** The step is compute-bound in the SegNet convs (#306); Python glue is noise. Fog's measure-first veto. |
| 9 | Muon optimizer — Newton-Schulz orthogonalization (a manifold-projection iteration) | fused explicit-order Metal NS kernel | BI-candidate | DERIVED negligible: witness matrices are 96²–256², ~75 opt steps/ep, 5 NS iters ⇒ ≲1 GFLOP/ep ⇒ sub-ms/ep on this GPU | 0 | few hundred LOC | **REJECT — tensor sizes 3+ orders below where a custom NS kernel pays.** (Answers the Lie-paper question, §5.) |
| 10 | frozen SegNet eval-mode BN | fold BN into conv weights (classic inference fusion) | **LEVER** (fp re-association of w·scale) | DERIVED few-% fwd | 0 | small | **REJECT by the #356 wall** — same fp-reorder class as compile; flips the uint8-STE argmax knife-edge. Not re-litigated, just mapped. |
| 11 | loss-term telemetry (`_lt_stride` default = 1 chunk/epoch, forward-only recompute of 8 pairs) | reduce cadence | SCHED | MEASURED-adjacent ~0.5 s/ep (~0.3%) | 0 | — | **NO ACTION** (observability non-negotiable; cost already floor-level). |
| 12 | checkpoint IO (`_atomic_savez` tmp+rename, BEST + resume sidecar @ eval-every 25) | batching/compression | SCHED | DERIVED negligible (MB-scale arrays, 1/25 ep cadence) | 0 | — | **NO ACTION.** Atomicity + per-stage preservation are non-negotiable; nothing to win. |
| 13 | verdict wall itself (mean 2189 s; numpy render of 1200 frames is single-thread serial Python-over-BLAS) | thread-pool across pairs (independent renders; per-pair BI) | BI-candidate (Accelerate GEMM concurrency-parity owed) | 0 today — the wall is FULLY async-hidden (window ≈ 2× wall, #306 §3) | 0 | medium | **QUEUE-conditional:** activates ONLY when the train window shrinks below the verdict wall (i.e. post-micro-batch). Pre-staged so the future owner doesn't re-derive. |
| 14 | RNG/reseed, spike-guard bookkeeping, closed-loop controller, resume-registry writes | — | SCHED | DERIVED ≈0 (np.median over a bounded window × 75 chunks/ep; controller decides at eval rows only) | 0 | — | **NO ACTION** (#306 already measured ~0; re-confirmed by inspection). |

**Verified-ON inventory (compiled `spec_v9_cgauge` today):** grouped-backward ~17× + persistence-pool
(env prefix) · `--fused-r-kernel` · `--cache-gt-skeleton` · `--async-verdict --verdict-batch 32
--verdict-pairs 0` · reorient amortization. No drift found between ledger-KEEP rows and the V9 argv
(the #306-era `cache-gt-skeleton` drift is cured).

## 2. What this does to the #306 attribution map (new MEASURED data)

- **The ep300 tau-stage +47 s/ep group:** persistence/clDice with sg-cache at V9 shapes (K=1) is only
  **2.8 s/ep MEASURED** ⇒ the tau-softplus form + lane-render-band own most of the remainder. The
  P8/P9 n24 probes (#306 §6 queue) remain the disambiguator — NOT run today (my charter forbids
  training launches; the probes train n24 for 4 ep).
- **The ~152→170 "+12% verdict contention" slice:** NOT reproduced at the GPU-step level (§3.2) —
  the MLX SegNet fwd+bwd is immune to a saturating CPU-torch scorer loop AND a saturating numpy
  Accelerate render loop (≤2%, within run-to-run noise). Residual suspects: GIL contention against the
  trainer's host-side glue, and the reorient/checkpoint slices. Only a live-run A/A can split it.

## 3. Bench receipts (all `[advisory NON-PROMOTABLE]`, scripts in session scratchpad `k443/`)

### 3.1 Verdict scorer bit-identity vs torch thread count (subprocess-isolated, seeded input)
| nt | SegNet argmax | SegNet margin | PoseNet raw |
|---|---|---|---|
| 6 (default on this host) | `71d3f520…` | `d9243ff3…` | `92a5c1c0…` |
| 4 | = | = | = |
| 2 | = | = | = |
| 1 | = | **DIVERGES** (`713729f3…`) | not probed |

⇒ nt∈{2,4,6} is bit-identical on these probes for BOTH scorers; nt=1 changes the margin bytes —
thread-count CAN move numerics, so any future nt-cap lever must carry a startup per-count parity gate
(the fused-R pattern). Single-input probes; a launch gate would verify on real pairs.

### 3.2 CPU-co-load immunity of the MLX GPU step (grouped-backward ON, B=8 SegNet fwd+bwd)
baseline 407–412 ms · under saturating torch nt=6: 409 · nt=2: 414 · nt=6+QoS-UTILITY: 414 ·
under saturating numpy-Accelerate GEMM chain: 411 (QoS makes no difference). **The GPU step does not
care about CPU co-load on this 18-core host.** (Kills row-2 as a build; the honest negative of this sweep.)

### 3.3 Persistence loss at V9 shapes (M=1 class × 1 frame, 384×512, iters 5, sg cached, GPU)
fwd 1.95 ms · **fwd+bwd 4.72 ms per pair ⇒ ×600 = 2.8 s/ep.** Pool micro-bench (1,384,512): metal
3 µs vs pure-MLX 17–18 µs (**5.6–5.9×**) — a real per-op speedup on ops that are µs-scale to begin with.

### 3.4 Persistence-pool Metal kernel — parity anchor (NEW)
`pool3x3_metal(x,"mean")` **== `_pool3x3_np` EXACTLY (max|Δ| = 0.0)** while the pure-MLX fallback
`_pool3x3_mlx(x,"mean")` differs from the numpy authority by **2.4e-7** (mx.mean reduction order).
min/max: metal == pure == numpy exactly. ⇒ with the flag ON (the V9 env), the density-weight forward is
MORE authority-faithful than the fallback, not just faster. The kernel (BUILT since #355, which listed it
signature-only) dispatches ONLY in the stop-grad density path; keep ON.

### 3.5 Lazy-graph-construction share (representative #356 closure, P=196608, GPU)
graph build 0.12 ms/call vs build+eval 21.8 ms/call = **0.6%**. Python graph construction — and by
extension the per-pair Python glue — is not a target (Fog: don't optimize what the profile exonerates).

## 4. Agner Fog discipline — mapping (operator-directed reference)

Applied per row above; the transferable summary for this stack: **measure-before-optimizing** killed 4 of
my 5 initial candidates (rows 2,3,4,6,8) before a line was written — that IS the Fog method working ·
**loop fusion done explicitly** = exactly the surviving explicit-order Metal family (#356 law; rows 3,9,10
are its boundary) · **table-lookup/caching over recomputation** = already systematically exploited
(cf_mx_cache, lstar_cache, sg cache, gt uint8, reorient amortization) — the remaining cacheables (rows 4,6)
are measured below noise · **memory locality / SoA** = the resident MLX caches are already
device-resident SoA; no AoS→SoA candidate found with a per-step duty cycle · **avoid hot-loop
allocation** = bounded at ≤0.6% (§3.5) · rows needing a C/Rust lowering to be real: none found — the
compute-bound core is already inside MLX/Metal kernels; `runtime-rs/` routing not required.

## 5. arXiv 2606.29636 (Lie-group diffusion for quantum circuit synthesis) — transfer assessment

Assessed, honest answer: **nothing transfers as a throughput/kernel lever.** (a) Its manifold-native
numerics (diffusion on SU(2)≅S³ with Cayley-style retractions) address SAMPLING on a compact group; our
`tac.lie` se(3)/SE(3) channel stores twists in the algebra and integrates by exact exp/log — there is no
sampling loop to accelerate. (b) The one genuine rhyme — Muon's Newton-Schulz as a manifold-projection
iteration that a fused explicit-order Metal kernel could serve — is DERIVED dead at our sizes (row 9:
96²–256² matrices × 75 steps/ep ⇒ sub-ms/ep; a custom kernel pays at ≥4k² transformer scales, 3+ orders
away). (c) Cayley log-eigen updates: convergence with WW-PGD #442 noted (that agent's scope; #442
itself returned LEVER NO-GO). (d) The "skeleton-selector hybrid" is circuit-skeleton search — no relation
to our clDice soft-skeleton beyond the word.

## 6. What I did NOT do (spec honesty — no silent caps)

- Did NOT build anything — every BUILD-NOW candidate was measured/derived below noise or carried an
  unproven win (rows 2,3,4,6); shipping one would be orphan theater. The DSL leg is N/A-with-reason.
- Did NOT run the #306 P8/P9/P_min n24 probes (they are 4-epoch training runs; my charter forbids
  training launches) — they remain the tau-group disambiguator.
- Did NOT measure the REAL trainer loop live (no live run existed; contention residual split is
  live-run-only) — rows 2 and 13 are gated on that measurement.
- Did NOT examine: the CUDA port (#438) side; the base trainer's non-levelset paths; startup-phase
  (first-epoch) costs; Accelerate GEMM concurrency-parity (row 13's owed gate); the tau-softplus form
  and lane-render-band per-lever n600 split.
- Did NOT re-litigate mx.compile/whole-step fusion (law `witness_fp_reorder_transform_bit_identity_wall_v1`).
- Flag for a future doc pass (not touched — outside a $0 sweep's mutation discipline):
  `make_persistence_topology_loss_mlx_compiled`'s docstring claims "fusion ≠ new numerics", which the
  #356 law contradicts; the factory is used only by a verify script (no hot-path exposure).

## Triality
- **DAG:** FEED-443 appended (same commit).
- **DSL:** **N/A-with-reason** — no lever ships (all BUILD-NOW candidates measured below noise or
  unproven; a default-off stub with no measured value is the config-orphan anti-pattern). The two
  surviving speed levers are already DSL-held + ON (`CacheGtSkeleton`, fused-R, perf-env prefix); row-1's
  build item belongs to the micro-batch owner (#313 family), row-5 to the #296 chain.
- **equations:** **N/A-with-reason** — no new law: the findings are host-conditional timing bounds +
  a parity ANCHOR on an existing built kernel (§3.4, recorded here + DAG); the governing law for this
  family (`witness_fp_reorder_transform_bit_identity_wall_v1`) already exists and today's rows 3/9/10
  sit inside its scope.

## Observability surface
Bench scripts + raw outputs: session scratchpad `k443/{nt_probe,nt_pose_probe,pool_bench,contention_bench,contention_bench2,graphbuild_bench}.py`
(scratch-only, non-evidence; every number reproduced in §3 with method). Verified-ON inventory
reproducible via `compile_v9_cgauge_432_launch_config().to_command()`. Diff-able vs #306/#355 (§2).

## 6-hook wire-in declaration (Catalog #125)
sensitivity-map: N/A (timing, not score axes) · Pareto: ACTIVE (row 1/13 feed the wall-clock runway +
waterfill notes) · bit-allocator: N/A · cathedral autopilot: N/A (advisory) · continual-learning: this
memo + DAG FEED-443 · probe-disambiguator: §2's live-run A/A + the #306 P8/P9 queue ARE the
disambiguators for the two unresolved attributions.
