# THE OPTIMAL FINAL ULTIMATE CAPSTONE VEHICLE — canonical buildable spec (2026-06-11)

**Subagent:** `capstone-spec-synth-20260611`. **Operator ask (2026-06-11):** the GPU-exploit verdict + every
converged session finding *"should inform our optimal final ultimate capstone vehicle"* — produce the ONE
canonical, buildable spec for vehicle #78 (our own learned basis, the pointer-mover toward sub-0.15), the
blueprint the GPU-fast 600-pair run is built from. **DESIGN MEMO ONLY** — I did NOT edit the running capstone
files, the daemons, or `mlx_scorer_adapters.py` / `capstone_trainer.py` (other lanes own those; this synthesis
feeds the build, it does not perform it).

**Authority discipline (CLAUDE.md, binding).** torch-CPU contest `evaluate.py` (600-sample, Linux x86_64) is
the ONLY leaderboard authority; NVIDIA T4 is the CUDA authority. Every local number cited below is
`[macOS-CPU advisory]` / `[macOS-MLX research-signal]` and **NON-PROMOTABLE** (`promotable=false`,
`score_claim=false`, `ready_for_exact_eval_dispatch=false`). Every predicted contest `S` is tagged
`[predicted]` with its source memo + the drift projection shown. NO MPS anywhere. NO paid dispatch fired.
The exact frontier pointer is **UNMOVED: 0.19109982 [contest-CPU], 177,169 B, sha `b46897267ded…`** (ABOVE
T_1 → GOAL UNSATISFIED). This memo emits NO archive — it is the spec + the build sequence + the pre-registered
prediction + the gated decisive next-run command.

NO FAKE: every architectural number traces to a cited measured artifact (the byte-budget JSON, the throughput
profile, the drift ladder, the scorer benchmark, the spectral atlas) or a closed-form arithmetic shown inline.
Where two session memos disagree (base_ch 20-vs-24-vs-36; the lever-map L1 recommendation), the contradiction
is named and reconciled explicitly in §0.1 — not papered over.

---

## 0. HEADLINE — the vehicle in one paragraph

The optimal capstone is **C1′ at frontier-class capacity: a fresh-init, score-aware-trained base_ch=24
HNeRV-class full-RGB decoder** (114,710 params; the LARGEST base_ch that byte-fits sub-0.19 at int8, and the
closest to the frontier's measured per-frame capacity that the d_seg-plateau verdict says is REQUIRED to reach
the seg floor) **+ the stored-float 28-d per-pair latent carrier** (the frontier's own proven pose-capable
carrier; drop the 8-bit VQ index that walled pose) **+ per-frame FiLM + the HiNeRV grid-PE** (low frequency
budget, `grid_pe_num_freqs=4`, validated by the spectral atlas) **+ the L1 weight-tied shared upsample blocks**
(the inflate-compute rate lever that brings base_ch=24's ~126 KB int8 archive down under the sub-0.19 budget).
It is **trained at COMPRESS time on the GPU** (the existing MLX-GPU fast-scorer wire-in, bs≤8, with torch-CPU
as the periodic authority gate and ExecuTorch/CoreML FP32-exact PoseNet) under the PR95 8-stage curriculum +
cosine LR + EMA-warmup + a **cross-hardware-robust margin hinge** (so the local d_seg argmax survives
macOS→numpy→Linux/CUDA). It **inflates on CPU** — a deterministic, scorer-free, numpy-portable forward inside
the 30-minute / 4-CPU budget. **The submit axis is the CPU leaderboard** (our HNeRV class is double-favorable:
the favorable axis AND the medal-deciding axis, per the simulator). **Pre-registered predicted contest S
(CPU axis): the gated decisive question is whether base_ch=24 reaches the frontier d_seg floor; at that floor
S ≈ 0.137–0.158 [predicted], conservatively drift-projecting sub-T_1 (sub-0.19) and edging sub-0.15.**

### 0.1 The contradictions this synthesis reconciles (named, not papered over)

**(A) base_ch = 20 vs 24 vs 36 — RECONCILED to base_ch=24.**
- The **dseg-crux memo** bet **base_ch=20** (smaller-is-better; "flat 88K–180K d_seg basin"). **REJECTED.** The
  dseg-plateau verdict (`dseg_plateau_data_vs_capacity_20260611.md`) proved that bet rested on a MIS-READ of the
  floor memo (its 5-PR cluster is ALL at ~178K params — one band, not a sweep) and on EMA-shadow-frozen d_seg.
  The EMA-fixed LIVE plateau floors at d_seg ≈ 0.0073–0.0097 at 48 pairs (13–17× above the 5.6e-4 frontier),
  and at 600 pairs base_ch=20 drops to **71 params/frame = 0.48× the frontier's per-frame capacity** — more
  pairs at fixed 85K makes the single-video fit HARDER. base_ch=20's sub-0.15 outcome requires the unmeasured
  "85K reaches what 178K needed" miracle. → base_ch=20 is a RESEARCH PROBE (the §6 ablation arm), not the bet.
- The **optimal-carrier memo** proposed **base_ch=36 shrunk to a 25–55 KB decoder** targeting sub-0.118 (lever
  C, the decoder-shrink class shift). **DEFERRED, not adopted as the first vehicle.** Its 25–55 KB target is a
  DERIVED conditional-floor BAND that "cannot prove without a run" (its own §5), and it is in direct tension
  with the dseg-plateau capacity physics (a 25–55 KB decoder is FAR below frontier per-frame capacity → the
  plateau verdict predicts it will NOT hold the seg floor). The decoder-shrink is the sub-0.118 LATER campaign,
  gated on base_ch=24 first PROVING a smaller-than-frontier decoder holds d_seg≈5.6e-4 (the plateau verdict's
  exact reactivation criterion). It is not the pointer-mover now.
- **base_ch=24 (114,710 params)** is the synthesis: the dseg-plateau verdict's PRIMARY recommendation
  (frontier-class capacity = the config with the best chance of reaching the seg floor: 95.6 params/frame vs the
  frontier's 148.6 — still below but 2× base_ch=20's), AND it byte-fits sub-0.19 at int8 (126,410 B, rate
  0.084) — and the L1 weight-tie + grid-PE bring it comfortably under. The optimal-carrier memo's strategic
  conclusion ("need frontier-class params") SURVIVES; only its specific 25–55 KB number is deferred.
  - **BANK fallback: base_ch=22** (99K params, rate 0.077, S≈0.150 at floor) if 126 KB feels tight pre-weight-tie.

**(B) The lever-map "L1 = byte-close lever-B + paired eval" recommendation — REJECTED (per the prompt).** Lever
B is the score-native palette/label-map carrier; its byte-closed candidates measure **S = 11.65 / 13.58**
(`lever_b_byte_close_exact_eval_readiness_20260611.md`) because a piecewise-constant palette frame1 is
**pose-blind** (d_pose 2.67–12.66 vs the tube's 2.4e-5) and its OWN rasterized d_seg rises to 0.064. Lever B is
NOT a sub-0.19 candidate and is NOT in this spec. (The seg-generator becomes a downstream zero-byte
distortion-repair tool ON the RGB decoder — §3 Phase 3 — never a standalone carrier.) The lever-map's GENUINE
contributions ARE adopted: L7 cross-hardware-robust margin (essential for numpy-portability), L3 boundary
byte-spend, and the D-LOCAL submit-decision rule.

**(C) GPU-only inflate — REJECTED/DEFERRED (per the GPU-exploit verdict).** `rate` is device-independent
(read from `archive.zip` bytes) and a GPU-required inflate forces `--device cuda` → the +0.033 CUDA tax. Any
byte-shrink that is CPU-feasible in 30 min belongs on CPU (tax-free). GPU power belongs at COMPRESS time. The
vehicle inflates on CPU; the GPU is exploited only for training/search. (Reactivation: a <50 KB archive whose
witness-decode is measured CPU-infeasible-in-30-min — not our regime.)

---

## 1. ARCHITECTURE (the decoder)

**base_ch=24 HNeRV-class full-RGB decoder**, fresh-init (NOT memorized-point continuation — that DEGRADES,
KILLED per the optimal-carrier memo), the shared PR95 `HNeRVDecoderMLX` backbone the capstone composes over.

| component | spec | source / rationale |
|---|---|---|
| base channels | **24** (`channels[0]=24`; taper `[24,24,24,18,14,12,12]`) | dseg-plateau verdict PRIMARY (frontier-class capacity; 95.6 params/frame) |
| params | **114,710** | byte-budget JSON `base_channels=24` row |
| upsample blocks | bilinear-skip + PixelShuffle(2) + sin, 6 stages (6×8 → 384×512) | ALREADY PRESENT in `HNeRVDecoderMLX` (HiNeRV audit: the bilinear-skip half is structural — do NOT rebuild) |
| **grid-PE** | `hinerv_grid_pe=True`, **`grid_pe_num_freqs=4`** (pe_dim=16) | HiNeRV delta (landed default-off); spectral atlas: scorer energy is LOW-freq + horizontal → a SMALL freq budget is principled, not arbitrary; high-freq budget is wasted. Deterministic grid = 0 archive bytes; only the `grid_pe_proj.proj.{weight,bias}` (24×16+24 = 408 params ≈ 0.4 KB) is stored |
| **weight-tie (L1 rate lever)** | tie the 6 upsample conv blocks → **2 shared blocks** + per-stage FiLM/scale modulation as the symmetry-breaker (keep stem + final distinct) | inflate-compute rate lever L1 (RANK 1); the per-stage FiLM keeps expressivity at ~0 byte cost; **NOT YET BUILT** (§6 build step) |
| per-stage FiLM | per-stage `(scale, shift)` modulation breaking the weight-tie symmetry | L1 mechanism; reuses the existing FiLM machinery (`_PoseFiLM` pattern) |
| activation | sin (NeRF-style; avoids dead-ReLU on single-video memorization) | PR95 L18 (already present) |
| dtype | **int8** (per-tensor fp16 scale + brotli q11) | byte-budget JSON; int8 ≈ 0.985 B/param vs fp16 1.678 |

**Param/byte budget (MEASURED, byte-budget JSON `base_channels=24`):**
- int8 decoder before weight-tie: 113,001 B + codebook 6,345 B + index/latent + pose 6,448 B → total **126,410 B**, rate **0.084**.
- After L1 weight-tie 6→2 (removes ~3–4 mid-block conv tensors, **−10 to −16 KB** `[predicted]`, inflate-compute memo L1) → decoder ~97–103 KB → **archive ~110–116 KB, rate ~0.073–0.077** → **byte-fits sub-0.19 with margin.**
- Target: **int8 archive < 115 KB for sub-0.19** (= rate < 0.077 = total < 115,640 B, dseg-plateau §4). The weight-tie is what brings base_ch=24's ~126 KB under this line. Aggressive 6→1 tie reaches −30 to −50 KB (rate to ~0.057) IF d_seg/d_pose hold — the falsifiable gate.

---

## 2. CARRIER (per-pair representation + pose)

**stored_latent + scalar pose-store + per-frame FiLM** — drop the 8-bit VQ index (it walled pose).

| section | spec | bytes (MEASURED / predicted) | source |
|---|---|---|---|
| **per-pair latent** | **stored 28-d float latent, temporal-delta + raw-LZMA coded** (PR95 L24/L25); carrier=`"stored_latent"` (the frontier's OWN proven carrier — reaches d_pose 2.9e-5) | ~12,000–15,387 B / 600 pairs | carrier-pivot memo (drop VQ: 8 bits < the 21 bits pose needs); optimal-carrier C1′ |
| **scalar pose-store** | OPTIONAL decoupled 6-d (effectively scalar) GT pose, temporal-delta coded; pose intrinsic dim = **1.00** (dim-0 = 99.80% of variance, ~21 bits/pair) — Quantizr's pose trick. FiLM-condition the moving frame on it | 0 (joint) **or** ~1,557 B (split fallback) | optimal-carrier §1.2 probe (`carrier_intrinsic_dim_probe`); byte-budget pose row 6,448 B is the un-delta'd upper bound |
| **per-frame FiLM** | `film_enabled=True`, `film_hidden=32`; separate film0/film1 modulating the decoder feature (the mechanism that held d_pose 2.7e-4) | rides the decoder section | `CapstoneVqNervConfig.film_enabled` (already present) |
| codebook | small, paid once ("free in decode") | ~6,345 B | byte-budget JSON |
| framing/sidecar | `capstone_config_v1` JSON sidecar (base_ch / pose_mean,std / film / num_pairs / dtype) | ~1,000 B | `_archive_with_config` (NO-FAKE: payload-only archive is NOT inflatable for a FiLM bundle) |

**Joint-vs-split decision (the crux):** keep the **joint stored-float 28-d latent as PRIMARY** (frontier-proven,
rate-efficient — 201 bits/pair joint is BELOW the 246-bit naive sum, exploiting seg↔pose cross-structure). The
**split scalar pose-store is the FALLBACK** if base_ch=24's joint latent can't hold the tube at the smaller
capacity — it converts "pose might wall" into "+1,557 B, pose guaranteed at the tube." Decision made at the §6
Phase-1 pose A/B (latent-vs-VQ already settled toward stored_latent; this is joint-latent-vs-split-pose-store).

---

## 3. TRAINING (COMPRESS time, GPU-max)

The compress-time loop has UNLIMITED GPU (M5 Max MLX) + NO tax + the scorer is legally available. This is where
all GPU power belongs (GPU-exploit verdict).

| element | spec | source |
|---|---|---|
| **curriculum** | **PR95 8-stage** (`--curriculum pr95_8stage`): CE → tau_softplus → smooth_disagreement → QAT → c1a-L7 → λ-sweep → σ-sweep → Muon-finetune (29,650-epoch-class schedule scaled to budget) | PR95 L14; `pr95_8stage_curriculum_mlx_port` |
| **optimizer schedule** | `--optimizer-schedule pr95_adamw_then_muon` (AdamW stages 1–7, Muon stage 8 final-stage-only) | PR95 L15 |
| **LR** | cosine LR per-stage-restart (the dominant d_seg-floor fix B1) | optimal-carrier Phase-2 recipe |
| **EMA** | weight-EMA 0.997 with **WARMUP decay** `min(decay,(1+t)/(10+t))`, export the SHADOW | EMA-shadow-lag fix `f771e6e00` (the 3rd measurement-artifact correction — constant decay froze the shadow ~init on short runs; warmup is non-negotiable AND a real export-correctness fix, not just telemetry) |
| **scorer backend (GPU-fast)** | **`--scorer-backend mlx_gpu`** at **bs≤8** (the sweet spot: 1.47× at bs=8; bs=16 REGRESSES to 0.61× via a Metal memory-pressure cliff in the VJP) — grad cosine 0.9999 vs torch-CPU, d_seg flip-delta 0 | mlx-gpu-scorer-training-wirein (commit 57d3a83ff) |
| **PoseNet authority** | torch-CPU `--authority-recheck-every 50` for the absolute d_pose (the MLX-GPU pose drift 2.76e-4 can EXCEED a frontier d_pose 3.4e-5 — NEVER trust MLX-GPU absolute pose near the frontier). ExecuTorch-MLX-GPU PoseNet (FP32-exact, rel_mse 5e-14, env `MLX_METAL_GPU_ARCH=applegpu_g15`) OR CoreML-FP32 are the zero-port FP32-exact GPU alternatives | scorer-backend benchmark + drift ladder RUNG A + executorch-segnet-fix |
| **SegNet (training gradients)** | our MLX-GPU SegNet (MLX-native VJP, flip-rate 1.24e-5 = boundary near-ties, negligible for training). CoreML-FP32 SegNet (0 flips) is the higher-fidelity gate alternative | scorer-backend benchmark §"SegNet" |
| **torch-CPU authority gate** | the periodic (every N epochs / pre-promotion) torch-CPU re-score of argmax-flip d_seg + first-6-dim d_pose gates ALL promotion. No score/kill from a GPU backend | scorer-backend benchmark authority invariant |
| **eval_roundtrip** | ON (bicubic up 874×1164, bilinear down 384×512, uint8 STE) — differentiable in the MLX path; diff-rgb_to_yuv6 + fail-closed assertion | CLAUDE.md eval_roundtrip non-negotiable; the MLX bridge applies `apply_eval_roundtrip_nhwc` |
| **cross-hardware-robust MARGIN hinge (L7)** | **NEW term** (NOT the existing `l7_softplus_seg_loss_mlx` weight-boost) — push boundary-pixel top1−top2 margin PAST the measured cross-hardware logit drift (≥ ~0.1, anchored on the MLX-GPU 0.096 max logit delta) so the LOCAL argmax SURVIVES macOS→numpy→Linux/CUDA. ESSENTIAL for numpy-portability: without it a local-sub-0.15 can EVAPORATE at the contest (the L9 risk) | lever-map L7 (the prompt names this ESSENTIAL); drift ladder §3.3 |
| **boundary byte-spend (L3)** | spend carrier/coder bits at the small-margin boundary set; 95% of pixels carry >2 logits of free room (the certified-stable interior is rate headroom) | lever-map L3 / margin field |
| pairs | **600** | the full contest video; do NOT under-train on 48 |

---

## 4. INFLATE (CPU, deterministic, numpy-portable, scorer-free)

The witness reconstruction runs in the contest's CPU instance (4 CPUs, 16 GB, 30-min budget — mostly unused;
the current capstone inflate is a single numpy forward measured in seconds).

| element | spec | source |
|---|---|---|
| substrate | **pure numpy + brotli/lzma** (NO torch, NO MLX, NO scorers) | substrate law; `numpy_reference.py` already proves conv2d/pixelshuffle/bilinear/bicubic/sin/FiLM in numpy |
| **weight-tied iterative decode** | the 2 shared upsample blocks applied across the 6 stages with per-stage FiLM (L1); K× the conv FLOPs — trivially ≤ budget | inflate-compute L1; the numpy block-loop edit |
| grid-PE | regenerated deterministically from `(base_h, base_w, num_freqs)` at decode (0 stored bytes); the numpy path MUST resolve `grid_pe_proj.proj.{weight,bias}` (the NO-FAKE key-mismatch bug the HiNeRV landing fixed — do NOT regress it) | HiNeRV grid-PE landing §4 |
| camera upscale | bicubic 384×512 → camera resolution | current inflate; render path unchanged |
| scorers | **NONE at inflate** (CLAUDE.md Strict scorer rule — SegNet/PoseNet ~73 MB + non-compliant) | hard constraint |
| L4 banker (optional, parallel) | context-adaptive arithmetic/cross-tensor coder on the decoder blob (`tac.pr103_arithmetic_codec`); −3 to −8 KB `[predicted]`, LOW risk, stacks on L1 | inflate-compute L4 |
| budget | full 30-min CPU; iterative decode + arithmetic unpack are seconds-to-low-minutes | upstream/README.md:114 |
| determinism | byte-stable across CPU/CUDA numerics (the eval roundtrip) | inflate-compute L1 compliance |

---

## 5. AXIS + PROJECTION (the submit rule)

| element | spec | source |
|---|---|---|
| **target axis** | **CPU leaderboard.** Our HNeRV-class capstone is in the CPU-favoring cluster (+0.033) — DOUBLE-favorable: the favorable axis AND the medal-deciding axis (the leaderboard ranks on CPU; PR102's third prize was awarded on its 0.19538 CPU score). The simulator's per-candidate verdict for OUR substrate = CPU | per-candidate simulator §4 |
| CPU-axis projection | `S_contest_CPU ≈ S_local_macOS_CPU − bias_B`; `bias_B = +1.05e-5` (SegNet-only, σ 8.3e-7), `guard_B = 3e-6`. The local advisory is a CONSERVATIVE UPPER BOUND (macOS reads slightly HIGH) | drift ladder RUNG B + simulator §3.1 |
| **drift-aware submit rule (D-LOCAL / L1)** | a local macOS-CPU advisory of **≤ 0.189987** conservatively projects below T_1 = 0.19 (the bias minus guard is +7e-6; required beat ~1.3e-5). Live trigger vs the current frontier: local < **0.191093** (= frontier 0.191099824 − 7e-6). When crossed → claim the lane, dispatch ONE `[contest-CPU]` exact eval | drift ladder §3.2 + lever-map L1 |
| **cross-hardware-robust requirement** | the L7 margin hinge (§3) is the structural guarantee that the local CPU advisory TRANSFERS — boundary argmax wins must live past the ~0.096 cross-hardware logit drift, never inside the 5.2e-5-margin tie band (the L9 failure mode) | lever-map L7/L9 |
| CUDA axis | NEVER projected from the CPU proxy. If a CUDA number is wanted, its own paired `[contest-CUDA]` eval; RUNG C says HNeRV-class CUDA ≈ CPU + 0.033 (informational only, class-bounded prior, never a claim) | drift ladder §1 + GPU-exploit |
| simulator validation | the §7 CPU leg (~$0.12) validates the simulator AND buys the exact-eval row simultaneously (means and end coincide) | simulator §5.1 |

### 5.1 Pre-registered predicted contest S (the falsifiable commitment)

Score model `S = 100·d_seg + √(10·d_pose) + 25·B/D`; pose held in the tube (≈2.9e-5) by the stored_latent
carrier (+ split pose-store fallback); rate from the weight-tied base_ch=24 int8 archive ~110–116 KB
(rate ~0.073–0.077). **Per-axis split, the gated decisive question = does base_ch=24 reach the d_seg floor:**

| d_seg outcome (the gate) | seg term | pose term | rate | **CPU-axis S `[predicted]`** | CUDA-axis (= CPU+0.033, informational) |
|---|---:|---:|---:|---:|---:|
| **frontier floor 5.6e-4** (the bet) | 0.056 | 0.017 | 0.075 | **0.148** ✓sub-0.15-edge | 0.181 |
| **capacity-scaled ~1.1e-3** (the bank) | 0.110 | 0.017 | 0.075 | **0.202** (just above T_1) | 0.235 |
| **48-pair plateau ~0.0085 holds** (the risk) | 0.85 | 0.017 | 0.075 | **0.94** (bust) | 0.97 |

**The pre-registered claim:** base_ch=24 @ 600 pairs reaches **d_seg ≤ ~1.1e-3** (banking sub-T_1) — the dseg-plateau
verdict's "best chance of reaching d_seg≈5.6e-4" config — at which point the CPU-axis S conservatively
drift-projects sub-0.19 (a local advisory ≤ 0.189987 → contest-CPU sub-0.19). The sub-0.15 edge (S≈0.148)
requires the frontier floor; the weight-tie's −0.005 to −0.011 rate ΔS is the margin that pulls it under 0.15.
**All figures `[predicted]`; the contest exact eval is the only arbiter.** This is the bet the §7 run measures.

---

## 6. BUILD SEQUENCE (ordered capstone-code changes, sequenced to avoid this-session's file collisions)

Each step is a concrete next change. **Collision discipline:** the daemons + `mlx_scorer_adapters.py` /
`capstone_trainer.py` are owned by live lanes — these steps EXTEND the capstone files, default-OFF/byte-identical
where possible (Catalog #290), and land via the serializer with post-edit-sha. Steps 1–2 are NEW builds; 3–5 are
already-landed dependencies to confirm; 6 is the cheap gate before the bet.

1. **[NEW BUILD] L1 weight-tie (`tie_depth`) on the decoder block loop.** Add `tie_depth` config to
   `CapstoneVqNervConfig` + the `HNeRVDecoderMLX`/`numpy_reference.numpy_decode_pair` block loop: share 1–2
   conv block(s) across the 6 upsample stages, with per-stage FiLM/scale as the symmetry-breaker. Thread through
   `export.py` (the tied weight dict) + `inflate.py` (BOTH stored_latent AND vq_index branches — the grid-PE
   landing's GAP lesson) + the `capstone_config_v1` sidecar. Default `tie_depth=6` (no tie = byte-identical).
   Parity test (numpy == MLX with a TRAINED tied block; a zero/no-op tie must FAIL the NO-FAKE test).
   *This is the rate lever that brings base_ch=24 under sub-0.19.*

2. **[NEW BUILD] Cross-hardware-robust margin hinge + decoupled scalar pose-store.**
   - **Margin hinge:** add a NEW loss term to `mlx_pr95_port.mlx_losses` (sister to `l7_softplus_seg_loss_mlx`):
     `mean(relu(margin_floor − margin_boundary))` with `margin_floor ≈ 0.1` (anchored on the 0.096 cross-hardware
     logit delta), applied at boundary pixels, wired into the curriculum's L7 stage. This is DISTINCT from the
     existing weight-boost L7 (it enforces a margin FLOOR, not a loss weight). It is the numpy-portability
     guarantee — the §5 transfer requirement.
   - **Scalar pose-store (FALLBACK):** add the decoupled 6-d temporal-delta pose-store + FiLM-on-moving-frame
     option to `vq_nerv_bundle.py` (does NOT exist yet) — the §2 pose-risk fallback. Default-OFF (joint latent
     primary); turned on only if the Phase-1 pose A/B shows the joint latent walls.

3. **[CONFIRM — LANDED] grid-PE CLI passthrough.** `hinerv_grid_pe`/`grid_pe_num_freqs` exist on the config
   (landed default-off, parity-proven) but are NOT yet exposed on `run_capstone_campaign.py`. Add
   `--hinerv-grid-pe` + `--grid-pe-num-freqs` (default 4) and thread into the `_archive_with_config` sidecar.
   *Small wiring; the mechanism + numpy port already landed (commit in `capstone_hinerv_skip_gridpe_upgrade`).*

4. **[CONFIRM — LANDED] MLX-GPU fast-scorer wire-in.** `--scorer-backend mlx_gpu` + `--authority-recheck-every`
   landed (commit 57d3a83ff). Confirm the bs≤8 sweet-spot is the trainer default (the campaign CLI does NOT
   expose `--batch-size`; default 8 is correct — do NOT raise to 16). *No new build; this is the GPU-fast
   dependency the bet rides on.*

5. **[CONFIRM — LANDED] stored_latent carrier + EMA-warmup.** `--carrier stored_latent` landed; the
   EMA-warmup fix landed (`f771e6e00`). *No new build.*

6. **[GATE — the cheap 2×2 capacity-confirm ablation] BEFORE the multi-day bet.** The dseg-plateau verdict's
   pre-registered 2×2: `{base_ch=20, base_ch=24} × {48 pairs, 192 pairs}`, CE-only, equal epochs-per-pair, LIVE
   `use_ema_for_eval=False`, marker-on-exit per daemon. The single decisive number: **sign(plateau(B=20@192) −
   plateau(A=20@48))** + whether plateau(C=24@48) < plateau(A). On the MLX-GPU loop this is ~hours, not the
   ~19h the CPU-scorer estimate implied. Confirms base_ch=24 is the right vehicle (or reverts to base_ch=20 if
   data-limited after all). *Gate: do NOT fire the 600-pair bet before this confirms capacity-limited.*

---

## 7. PRE-REGISTERED PREDICTION + THE DECISIVE NEXT-RUN COMMAND

**Pre-registered prediction (the falsifiable commitment, §5.1):** the GPU-fast base_ch=24 @ 600-pair run, under
the PR95 8-stage curriculum + cosine LR + EMA-warmup + the cross-hardware margin hinge + the weight-tie, reaches
**LIVE d_seg ≤ ~1.1e-3 and d_pose in the tube (≈2.9e-5)**, byte-closing to a ~110–116 KB int8 archive (rate
~0.073–0.077), for a **CPU-axis advisory S ≈ 0.148–0.20 `[predicted]`** — and the sub-T_1 outcome (advisory ≤
0.189987) conservatively drift-projects sub-0.19 on the contest CPU leaderboard. If LIVE d_seg holds the
48-pair plateau (~0.0085) instead, the run BUSTS (S ≈ 0.94) and the verdict reverts to the decoder-shrink /
larger-seg-decoder reactivation. The single number that decides it: **the LIVE d_seg the base_ch=24 600-pair
run reaches.**

**The decisive next-run command** (the GPU-fast base_ch=24 @ 600-pair run), GATED on (a) the §6 step-1 weight-tie
landing + (b) the §6 step-6 2×2 capacity-confirm ablation showing capacity-limited (base_ch=24 < base_ch=20 at
fixed pairs). Run as a detached nohup daemon with marker-on-exit (NEVER session-bound; the session-watcher trap
killed 3 daemons this session). The 600 GT targets must be precomputed first (cache currently n≤100):

```bash
# 1. lane-claim FIRST (non-negotiable cross-agent coordination)
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id lane_capstone_base_ch24_600pair_mlxgpu_20260611 --instance local_mlx_gpu \
    --status active --notes "optimal capstone vehicle: base_ch=24 + stored_latent + grid-PE + weight-tie + margin-hinge"

# 2. the decisive run (detached daemon; verify each NEW flag's argparse before firing)
nohup bash -c 'OMP_NUM_THREADS=6 .venv/bin/python experiments/run_capstone_campaign.py \
    --max-pairs 600 \
    --base-channels 24 \
    --carrier stored_latent \
    --decoder-dtype int8 \
    --hinerv-grid-pe --grid-pe-num-freqs 4 \
    --tie-depth 2 \
    --curriculum pr95_8stage \
    --optimizer-schedule pr95_adamw_then_muon \
    --curriculum-total-epochs 240 \
    --seg-weight 100.0 --pose-weight 1.0 \
    --scorer-backend mlx_gpu \
    --authority-recheck-every 50 \
    --eval-every 10 \
    --device cpu \
    --targets-cache experiments/results/capstone_gt_targets_cache \
    --out-dir experiments/results/capstone_base_ch24_600pair_mlxgpu_20260611; \
    echo "EXIT=$?" > experiments/results/capstone_base_ch24_600pair_mlxgpu_20260611/DONE.marker' \
    < /dev/null > experiments/results/capstone_base_ch24_600pair_mlxgpu_20260611.outer.log 2>&1 & disown

# 3. on completion: byte-close (advisory.score_reloaded_int8_archive) + apply the §5 submit rule;
#    if advisory S <= 0.189987 -> ONE paired contest-CPU (~$0.12) exact eval = the pointer-mover.
```

**Notes (NO-FAKE on the command):** `--device cpu` is the torch-CPU AUTHORITY device for eval re-scores (NEVER
mps). `--tie-depth`, `--hinerv-grid-pe`, `--grid-pe-num-freqs` are NEW flags that §6 steps 1+3 must add before
this fires (they do not exist yet — verify argparse, do not invent). `--scorer-backend mlx_gpu` + `--carrier
stored_latent` + the curriculum/optimizer flags ARE landed. bs=8 (trainer default, the MLX-GPU sweet spot) — do
not raise to 16. At ~14 min/epoch MLX-GPU bs=8, a 240-epoch curriculum ≈ 56 GPU-hours — confirm the epoch budget
against the §6 ablation before committing the multi-day run; if the margin hinge or weight-tie misbehaves, the
`--scorer-backend torch_cpu_bridge` fallback remains.

---

## 8. WIRE-IN (Catalog #125) + SCOREBOARD

1. **sensitivity-map — ACTIVE.** New top-level prior: the capstone is base_ch=24 (NOT 20, NOT 36) — the
   frontier-class capacity that the dseg-plateau physics says is required to REACH the seg floor, with the L1
   weight-tie + grid-PE as the byte levers that fit it under sub-0.19. Pose is a scalar (dim 1.00); the decoder
   is the byte budget to attack.
2. **Pareto — ACTIVE.** d_seg(base_ch) is a CAPACITY curve (not a flat basin); base_ch=24's rate cost (0.084
   pre-tie) is worth taking iff its d_seg floor ≤ ~1.1e-3, which the §6 ablation + the §7 run measure. Lever-B
   (palette carrier) is DOMINATED (pose-blind, S=11.65) — excluded from the feasible set.
3. **bit-allocator — ACTIVE.** {base_ch=24 decoder, weight-tied + grid-PE} + {stored 28-d latent 12–15 KB} +
   {scalar pose-store 0–1.6 KB fallback}; pose gets bits-as-precision, seg gets bits-as-refinement at the
   boundary set (L3); the L4 arithmetic banker codes whatever decoder bytes remain.
4. **cathedral-autopilot — gate-conditional.** §6 step-6 ablation → §7 600-pair daemon → (advisory ≤ 0.189987)
   → ONE paired exact CPU eval is the dispatch surface. Do NOT dispatch the 600-pair run before the ablation.
5. **continual-learning — ACTIVE.** Reseeds the V3 judge: (a) base_ch=24 is the synthesis of the
   dseg-plateau verdict (frontier-class capacity) — base_ch=20 is a probe, base_ch=36-shrink is the LATER
   sub-0.118 campaign; (b) the carrier is stored_latent (VQ index walled pose); (c) the submit axis is CPU
   (double-favorable per the simulator); (d) the EMA-warmup + margin-hinge are export-correctness + portability
   non-negotiables, not telemetry.
6. **probe-disambiguator — RESOLVED + ONE OPEN.** "base_ch 20/24/36?" → 24 (plateau verdict + byte-fit).
   "carrier?" → stored_latent. "submit axis?" → CPU. "GPU inflate?" → no, CPU inflate + compress-time GPU. The
   ONE OPEN probe is the §6 2×2 ablation whose `sign(plateau(B)−plateau(A))` is the empirical arbiter of the
   capacity-vs-data question before the multi-day bet.

**UPPER (vs T_1 sub-0.19):** unchanged — this is a design memo, no archive. Frontier holds 0.19110 [contest-CPU].
**LOWER (the floor):** the spec is the named vehicle to REACH d_seg≈5.6e-4 (the sub-0.15 distortion-at-budget
door, S≈0.148 at the floor); the sub-0.118 reach is the LATER base_ch=36-decoder-shrink campaign, gated on
base_ch=24 PROVING a smaller-than-frontier decoder holds the floor (the dseg-plateau reactivation criterion).

---

## 9. CROSS-REFERENCES (the consumed + reconciled session memos)

`dseg_plateau_data_vs_capacity_20260611.md` (CAPACITY: base_ch=24 PRIMARY; base_ch=20 REJECTED — the
contradiction §0.1-A reconciles) · `capstone_optimal_carrier_design_20260611T041937Z.md` (C1′ carrier; its
base_ch=36-shrink-to-25-55KB is DEFERRED to the sub-0.118 campaign per §0.1-A) ·
`capstone_carrier_pivot_vq_index_impoverishment_20260611T034500Z.md` (drop the 8-bit VQ → stored_latent) ·
`capstone_hinerv_skip_gridpe_upgrade_20260611.md` (grid-PE landed default-off; bilinear-skip already present;
the key-mismatch NO-FAKE bug) · `capstone_ema_shadow_lag_reverses_seg_wall_verdict_20260611T070000Z.md`
(EMA-warmup fix `f771e6e00` — export-correctness, not telemetry) ·
`scorer_backend_benchmark_ours_vs_others_20260611.md` (PoseNet=ExecuTorch/CoreML FP32-exact; SegNet=MLX-GPU +
CoreML gate; torch-CPU authority) · `executorch_mlx_delegate_segnet_fix_20260611.md` (BOTH nets FP32-exact GPU
via `MLX_METAL_GPU_ARCH=applegpu_g15`) · `mlx_gpu_scorer_training_wirein_20260611.md` (the GPU-fast dependency,
commit 57d3a83ff; bs≤8 sweet spot) · `local_to_contest_scorer_drift_ladder_and_correction_20260611.md` (RUNG B
+1.05e-5 CPU bias; RUNG C +0.033 CUDA tax; the ≤0.189987 submit rule) ·
`per_candidate_local_to_contest_score_simulator_20260611.md` (CPU axis double-favorable for HNeRV class — the
axis choice) · `inflate_compute_as_free_rate_lever_20260611.md` (L1 weight-tie RANK 1; L4 arithmetic banker) ·
`gpu_only_inflate_thought_experiment_20260611.md` (GPU-inflate DEFERRED; GPU at compress time; rate is
device-free) · `scorer_numerics_dynamics_lever_map_20260611.md` (L7 cross-hardware margin ESSENTIAL; L3 boundary
byte-spend; L1 submit rule — its "L1=byte-close lever-B" recommendation REJECTED per §0.1-B) ·
`lever_b_byte_close_exact_eval_readiness_20260611.md` (lever B = S 11.65/13.58, pose-blind — excluded) ·
`scorer_spectral_atlas_v2_runaway_killed_partial_signal_20260611.md` (grid_pe_num_freqs=4 principled: scorer
energy is LOW-freq + horizontal; SegNet broadly weak = boundary-flip) · `capstone_vq_nerv_byte_budget_20260610.json`
(MEASURED param↔byte: the base_ch=24 int8 row 126,410 B) · `capstone_training_throughput_profile_…json`
(14.28s/step CPU scorer; the ablation affordability) · `GOAL_standing_v3_20260610.md` (the sub-0.15 ladder;
levers A/B/C/E) · `upstream/{modules.py,evaluate.py,frame_utils.py,README.md}` (frozen authority: d_seg =
per-pixel argmax-flip; 30-min CPU budget; D=37,545,489).

## 10. NO-FAKE / authority notes
- Every architectural number traces to a cited MEASURED artifact or an inline closed-form arithmetic. Every
  predicted contest S is `[predicted]` with the drift projection shown; none is a measured exact-eval row.
- All local (macOS torch + MLX) numbers are `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`,
  NON-PROMOTABLE. Only Linux x86_64 = `[contest-CPU]` (leaderboard authority); only NVIDIA T4 = `[contest-CUDA]`.
  NO MPS anywhere.
- The +1.05e-5 (RUNG B) and +0.033 (RUNG C) offsets are HNeRV-medal-band-class-bounded projection priors / spend
  triggers, NEVER score claims, conversions, promotions, ranks, or kills. The contest exact eval is the only
  arbiter of any S.
- The three contradictions (base_ch 20-vs-24-vs-36; lever-map L1; GPU-inflate) are reconciled explicitly in
  §0.1, not papered over. The spec is buildable (§6 names the 2 new builds + 3 landed confirms), numpy-portable
  (§4), and scorer-free-at-inflate (§4); torch-CPU/contest is the authority (§3, §5).
- The exact frontier pointer is UNMOVED (0.19109982 [contest-CPU]). This memo is a DESIGN spec + a gated next-run
  command; it does NOT claim the pointer moved. The means (this spec) is not the end (a lower exact S) — the §7
  run, gated on the §6 ablation, is the unit aimed DIRECTLY at the exact CPU-axis row that crosses T_1.

---

## APPEND 2026-06-21 (additive, HISTORICAL_PROVENANCE — body above unchanged): the BATCH-SIZE axis

This canonical spec trains at **bs≤8** (§0). The 2026-06-21 throughput investigation MEASURED that bs=8 is a
hard **latency floor**: A10G ≈ T4 ≈ M5 Max MPS all ~11–13 s/ep (GPU-invariant) because the per-epoch is 75
serial 8-pair optimizer-per-batch steps, not arithmetic. Modal is therefore NO-GO at bs=8 (~0% faster). The
ONLY real un-CPU-bound lever is a **larger training batch**, which is score-LOCKED for the faithful PR95 run
(per-batch optimizer step → batch_size defines #steps/epoch + the gradient trajectory) but is a legitimate
**capstone re-solve** lever. The concrete first-order **B=64 rescaled schedule** (held epochs; AdamW η×√8,
EMA (1−ρ)×8 — note the DIFFERENT exponent, σ×√8, λ×√8, **Muon η ×1 / B-INVARIANT** — derived: its
orthogonalized update normalizes away the noise magnitude, so the noise-floor d_seg-polish step is η regardless
of B; compensate fewer steps with more stage-8 epochs, not higher η) + the full coupling derivation +
the ~2–4× speedup estimate + the empirical-re-solve caveats are in the dedicated spec:
**`capstone_batch_size_fixed_point_B64_launch_spec_20260621.md`**. Also keep: `defer_batch_sync` ON (proven
bit-identical +2%); pose stays fp32-exact (compile drifts PoseNet 22%, REJECTED); validate LOCAL first (B=64
may finally be GPU-bound, unlike bs=8 → then a paid GPU is worth re-pricing). The B=64 vehicle is a NEW-vehicle
fixed-point solve, gated behind the current faithful bs=8 decisive run's stage-5 verdict.
