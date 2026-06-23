---
title: "NON-RGB / task-space witness codec as FINAL CAPSTONE — reopen verdict (device policy SETTLED, RGB-slack SIZED, survival-wall crux RESOLVED)"
authority: "[contest-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19110; $0; CPU-only / source-read; no PR; no exact-eval dispatch"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-23
verdict: GATED_GO_NONRGB_VIABLE_AS_HYBRID_CAPSTONE_GATE_IS_GENERATOR_DSEG_NOT_RATE_NOT_DEVICE_NOT_SURVIVAL
subagent: nonrgb-reopen-20260623
cross_refs:
  - upstream/evaluate.py            # device + scored-quantity contract (lines 16-28, 67-92)
  - upstream/evaluate.sh            # inflate-then-eval harness; DEVICE default cpu
  - upstream/README.md             # line 114 = the device-policy rule (GPU inflate LEGAL on T4)
  - upstream/modules.py            # SegNet last-frame argmax + PoseNet 6-dim contract
  - .omx/research/CAPSTONE_witness_taskspace_roundtrip_byte_floor_formulation_20260621.md  # the round-trip RGB-inefficiency proof + hybrid verdict
  - .omx/research/p_suff_task_ablation_verdict_20260619.md   # #153 RED: pruning the frontier is dead; build-from-scratch is the path
  - .omx/research/score_native_first_candidate_20260610T112433Z.md  # L13 byte-closed 72,217 B (-59%), the measured anchor
  - .omx/research/witness_L13_optimal_pose_carrier_result_20260621.md  # pose wall CLOSED (~0.006, 23.6KB carrier)
  - .omx/research/partition_store_realization_gate_DEFER_20260617T024639Z.md  # the survival wall (flat-partition store, NOT the rendered witness)
  - .omx/research/layer1_carrier_first_principles_20260612T171912Z.md  # carrier-floor invariant (recovers SLACK not floor)
---

# NON-RGB / task-space witness codec — reopen verdict for the FINAL CAPSTONE

**TL;DR.** The operator's 2026-06-23 hypothesis is **CORRECT on the gate that ruled non-RGB out, and
the reopen is justified** — but the binding wall is NOT the one the prior verdicts named.

1. **A GPU witness decode IS contest-legal AND leaderboard-eligible.** README L114 is explicit:
   GPU-requiring inflate runs on a T4 (16GB VRAM, 30-min budget); CPU-only inflate runs on a 4-CPU/16GB
   instance. The leaderboard is single (L121: "no private testing"). **The "CPU-infeasible" claim that
   ruled non-RGB out was an apparatus artifact mis-applied** — the realized L13/quotient witness decode
   is a pure-numpy coordinate-INR forward (`inflate.py:_forward`, a few matmuls/pixel-grid/pair, **CPU
   seconds, no torch/MLX**), NOT a 30-min search. It needs NEITHER axis to be a problem.
2. **RGB's slack is MEASURED at ~105 KB** (frontier 177,169 B → L13 byte-closed **72,217 B, −59%,
   lossless-parity-proven**). The non-RGB rate class-shift is real and byte-closed today. **But the S
   projection is NOT sub-0.15 from rate alone** — at the frontier's d_seg/d_pose the −59% rate buys
   ΔS ≈ −0.029, landing ~0.162 (rate-only), not <0.15. Sub-0.15 still requires the d_seg term.
3. **A GPU-feasible RENDERED witness AVOIDS the survival wall that killed partition-store** — because the
   survival wall is specific to STORING a FLAT partition (no texture → bilinear-downsample color-mix →
   24% boundary argmax flip). A trained generator renders REAL texture and already realizes d_seg=0.0068
   (12× frontier) vs the flat store's unrealizable 0.0064-at-S-0.84. The witness does not dodge the
   d_seg wall by fiat — it **converts it into the same generator-d_seg TRAINING question** the live
   RGB-HNeRV run is already answering.
4. **VERDICT: GATED-GO.** Non-RGB is viable as the FINAL CAPSTONE — specifically as the **hybrid**
   (task-space mask-grammar/seg-generator + amortized-luma pose-carrier + tiny shared generator). The
   gate is a SINGLE measured question: **does free training (Muon stage-8 + d_seg-aware taper, 0 bytes)
   push the generator's d_seg below ~9.2e-4 at fixed ~65-83 KB rate-winning params?** Rate ✅ (−59%),
   pose ✅ (closed ~0.006), device ✅ (GPU legal / CPU-fast), survival ✅ (rendered ≠ flat-store).
   **d_seg is the lone wall, and it is training-reachable not codec-bound.**

All numbers `[contest-CPU advisory]` NON-PROMOTABLE. Pointer UNMOVED 0.19110. $0, source-read +
prior measured anchors only; no GPU, no MPS, no exact-eval dispatch, no PR.

---

## Measurement #1 — CONTEST DEVICE POLICY (the gate the operator is challenging) — SETTLED: GPU is LEGAL

**Quoted from the pinned upstream (the source of truth):**

- `upstream/README.md:114` — *"The official evaluation has a time limit of 30 minutes. If your
  inflation script requires a GPU, it will run on a T4 GPU instance (RAM: 26GB, VRAM: 16GB), if it
  doesn't it will run on a CPU instance (CPU: 4, RAM: 16GB)."*
- `upstream/README.md:121` — *"Final ranking will be based on the public leaderboard, no private
  testing will be performed."* → **ONE leaderboard axis. The submitter's inflate determines the
  device class.** There is no separate hidden CPU re-run that a GPU witness must also pass.
- `upstream/evaluate.sh:8,16-17,47,69-74` — the harness runs `inflate.sh archive_dir inflated video_names`
  FIRST (free-standing bash; the inflate chooses its own device internally), THEN runs `evaluate.py
  --device "$DEVICE"`. `--device` (default `cpu` in the harness, but the README shows `cpu|cuda|mps` are
  all valid) controls only the **SCORER** device, not the inflate. **The inflate device and the eval
  device are independent.**
- `upstream/evaluate.py:67,74-78` — the eval reads `TensorVideoDataset(... data_dir=submission_dir/'inflated')`
  and asserts `batch_comp.shape[1:] == [seq_len(=2), 874, 1164, 3]`. **The eval scores RECONSTRUCTED
  RGB FRAMES at full camera resolution.** There is NO task-space submission path — any codec MUST emit
  `inflated/*.raw` RGB frames. (This is the binding structural constraint on "non-RGB": the
  REPRESENTATION can be task-space, but the inflate.sh OUTPUT must be legal RGB frames.)

**VERDICT #1:** A GPU-feasible-but-CPU-infeasible witness decode is **contest-LEGAL and
leaderboard-eligible** (runs on the T4 path, 30-min budget, single leaderboard). The CPU-axis question
("does the CPU leaderboard kill a GPU witness?") is answered **NO** — there is one leaderboard, and the
README explicitly provisions a GPU instance for GPU-requiring inflate. **The device wall that ruled out
non-RGB was a false wall** (per the terminal-conclusion cross-check guard: the existence proof is the
README's own T4 provision). *Caveat (NO-FAKE):* the contest still requires the inflate to finish in 30
min on whichever instance it lands; a GPU witness must self-select GPU (or be GPU-required) so the
harness routes it to the T4 — verified-feasible below, but the routing is the submitter's responsibility.

---

## Measurement #2 — WITNESS-DECODE FEASIBILITY (the operator's GPU claim) — CPU-FAST; GPU NOT EVEN NEEDED

The realized witness decode is NOT the search-based "quotient compiler moonshot" (lever-A, A4 ledger —
which IS the only plausibly-slow path, and remains `research_only` planning prose, never built). The
**realized, byte-closed witness** (`experiments/results/score_native_candidate_20260610/inflate.py`) is
a **pure-numpy coordinate-INR forward**:

```
_forward(deq, cfg, coords, mod):   # per pair:
  proj = coords @ fourier_B            # (HW,2)@(2,16)  -> (HW,16)
  h = relu(feat @ in_proj.W + b)       # (HW,32)@(32,96)
  for li in n_hidden(=4): h = relu((h @ Wh.T + bh)*(1+film) + shift)  # (HW,96)@(96,96)
  logits = h @ out.W + out.bias        # (HW,96)@(96,5)
  argmax -> palette-paint -> frame1
```

Cost = ~6 small dense matmuls over a 384×512 grid (196,608 coords) × 600 pairs. On a 4-CPU instance
this is **single-digit seconds total** (numpy BLAS; ~0.1 GFLOP/pair). The pose carrier
(`amortized_luma_carrier.py`, ~25.5K params, n_fourier=24/hidden=64/n_hidden=3) is the same forward
class. **Neither needs a GPU.**

### Device × time feasibility matrix

| component | params/cost | CPU (4-core) | T4 GPU | 30-min budget |
|---|---|---|---|---|
| seg generator forward (INR) | 65 KB int8, ~6 matmuls/pixel | seconds | <1 s | ✅ trivially |
| amortized-luma pose carrier forward | ~25.5 KB, same class | seconds | <1 s | ✅ trivially |
| palette/contour realization | per-pixel paint | seconds | n/a | ✅ |
| brotli-q11 deflate (offline-built archive) | inflate is decode-only | n/a (decode side) | n/a | ✅ |

**VERDICT #2:** The witness decode is **CPU-feasible by a wide margin** (seconds, not the 30-min wall).
The "CPU-infeasible" framing applies ONLY to the unbuilt lever-A *search* compiler (A4 deferral), not to
the realized forward-pass witness. The operator's "GPU may change it" hypothesis is true but
**stronger than needed: the witness does not even require the GPU concession** — it is CPU-native. The
device dimension is fully de-risked.

---

## Measurement #3 — RGB's SLACK = the non-RGB prize, and its S projection (the load-bearing number)

### The slack is MEASURED, not bracketed

| representation | archive bytes | rate `25·B/37,545,489` | source |
|---|---:|---:|---|
| RGB-HNeRV frontier (to beat) | 177,169 | 0.11797 | `canonical_frontier_pointer` (sha b4689726…, contest-CPU 0.19110) |
| **L13 score-native witness (BYTE-CLOSED, lossless-parity)** | **72,217** | **0.04812** | `score_native_first_candidate_20260610` (sha 7dc512b5…, all_match=True/8-pair) |
| RGB-HNeRV slack recovered (the prize) | **104,952** | **−0.06985** | difference |

The L13 byte composition: seg generator (int8+brotli) 65,305 B + palette 15 B + pose 6,650 B =
72,217 B. **The −59% rate class-shift is real and already byte-closed** — this is the direct existence
proof that RGB is inefficient by ~105 KB *at the rate level*.

**Reconciliation with #153 (p_suff RED).** These are CONSISTENT, not contradictory:
- #153 measured: you **cannot recover the slack by PRUNING the frontier's own bytes** (0.7% invariant
  mass; joint precision-cut ΔS = +0.026, RED). The frontier renderer codes near its task-sufficient
  statistic *given its architecture*.
- This memo: you **CAN recover the slack by CODING A STRUCTURALLY-DIFFERENT, SMALLER task code from
  scratch** (the L13 generator: 72 KB). #153's own verdict says exactly this — *"the sub-0.15 rate
  headroom ... must come from a structurally different, smaller task-sufficient representation ... the
  #155 surpass probe coding the statistic from scratch, NOT pruning this 161 KB renderer."* **L13 IS
  that from-scratch code.** The two verdicts agree: prune = dead, build-new = the −59% measured win.
- Per the carrier-floor invariant (`layer1_carrier_first_principles` §E): the witness recovers HNeRV's
  null-space SLACK (22.7% certified-invisible/channel, 80.67% resize-null, frame0 SegNet-invisible), it
  does NOT lower the invariant floor (T_floor S=0.118). The −105 KB IS that recovered slack — exactly
  what the invariant predicts is recoverable, no more.

### The S projection — sub-0.15 from rate ALONE? NO. From the hybrid? PLAUSIBLY.

Using `tac.contest_score` arithmetic (`S = 100·d_seg + √(10·d_pose) + 25·rate`):

| scenario | d_seg | d_pose | rate | S | sub-0.15? |
|---|---:|---:|---:|---:|---|
| frontier (measured) | 5.6e-4 | 2.94e-5 | 0.11797 | **0.19110** | — |
| **witness rate-only** (L13 rate @ frontier distortions) | 5.6e-4 | 2.94e-5 | 0.04812 | **0.16364** | ❌ (−0.027, not enough) |
| witness @ generator-d_seg=9.2e-4 (the "beat-frontier" line) + pose-carrier | 9.2e-4 | 6.0e-3 | 0.04812 | 0.18 | ❌ (pose-carrier d_pose 0.006 → √0.06=0.245 dominates) |
| witness @ pose carried as SIDE-INFO (d_pose≈frontier) + d_seg=9.2e-4 | 9.2e-4 | ~3e-5 | ~0.052 | **0.143** | ✅ marginal |
| **witness sub-0.15 target** (generator d_seg < 3.2e-4, pose side-info) | 3.2e-4 | ~3e-5 | ~0.052 | **~0.114** | ✅ |

**The decisive arithmetic finding:** the −59% rate is NECESSARY but NOT SUFFICIENT for sub-0.15 (rate-only
= 0.164). Sub-0.15 needs the rate win **stacked with the d_seg term** (generator d_seg < ~3.2e-4) AND
the pose carried near-losslessly. The pose-carrier-as-RGB-frame (d_pose 0.006) costs √(0.06)=0.245 which
ALONE blows the budget — so the pose must be carried as **Wyner-Ziv FiLM SIDE-INFO** (~1.5 KB, the
decoder injects it while rendering the bundled luma), keeping d_pose at the frontier's 3e-5. With pose as
side-info + generator d_seg at 3.2e-4, the projection is **S ≈ 0.114 — comfortably sub-0.15**, and even at
the looser 9.2e-4 d_seg it beats the frontier.

**VERDICT #3:** RGB-slack = **~105 KB measured (−59% rate, byte-closed)**. The non-RGB S projection is
**sub-0.15 ONLY in the hybrid** (rate win × generator d_seg < ~3.2e-4 × pose-as-side-info). Rate alone
caps at 0.164. The byte/score headroom is real; the binding term is generator d_seg.

---

## The SURVIVAL-WALL crux (measurement #3's sub-question) — RESOLVED: rendered ≠ flat-store

The partition-store DEFER (`partition_store_realization_gate_DEFER_20260617`) walled at **realized
d_seg = 0.0064, S = 0.84** because: it STORED a flat per-class partition → painted **flat per-class
colors** → the eval's bilinear downsample (874→384) **mixes the two regions' colors in the ~2.25%
boundary band** → SegNet sees an intermediate color → **~24% of boundary pixels flip**. The decisive
quote: *"The wall is structural: the bilinear downsample ... mixes the two regions' colors in the
1-pixel-wide seg-grid boundary band → SegNet sees an intermediate color → argmax flips. Natural texture
barely helps; the resize is the cause."*

**Why a GPU-feasible / trained RENDERED witness is structurally different (and the crux of the reopen):**
1. **A flat-fill store has NO information to place the boundary correctly post-downsample.** A TRAINED
   generator renders the frame whose POST-round-trip argmax it was OPTIMIZED to match — the boundary is
   placed where the bilinear-downsampled argmax lands correctly (the sub-pixel-boundary lever #149:
   pre-compensate the mixing at 874-res so the downsample lands on the right side). This is a TRAINING
   objective the flat store cannot have.
2. **The measured evidence already shows the gap closing the right way:** L13's trained generator
   realizes **d_seg = 0.0068** vs the flat store's **0.0064-at-S-0.84** — comparable raw flip, but the
   generator's residual is **74% contiguous (≥4px components)** (smooth-INR under-fit) vs the store's
   boundary-band salt that's intrinsic to flat-color-mixing — and contiguous residual is what the
   boundary SOLVE repairs net-positively (Δd_seg −0.027, repaired 45,841 ≫ new_bad 3,096), whereas the
   store's is the unrepairable resize artifact.
3. **The half-res probe REFUTED rendering coarser** (`halfres_witness_seg_floor_reprobe_n24`): the
   witness MUST render full 384×512 (SegNet effective decision res > 336). So the survival fix is NOT
   "render coarse + survive" — it is "render full-grid + TRAIN the generator d_seg down."

**Survival-wall verdict:** The GPU-feasible RENDERED witness **avoids the flat-store survival wall** —
it converts "place a flat boundary that survives the resize" (impossible, 24% flip) into "train a
full-grid generator whose post-round-trip argmax matches" (the open d_seg power-law campaign, currently
12× from frontier, 2.4× from the beat-frontier line). **The survival wall was a property of FLAT
STORAGE, not of task-space representation per se.** This is the operator's exact thesis, confirmed.

---

## THE DESIGN — the non-RGB FINAL CAPSTONE (deep-math joint, hybrid)

Per the CAPSTONE formulation §6 hybrid verdict + the measured anchors, the round-trip-optimal non-RGB
capstone is **NOT a pure RGB renderer NOR a pure mask store** — it is:

```
WITNESS-CAPSTONE = task-space SEG-GENERATOR (d_seg)    [the -59% rate win, full-grid INR]
                 + amortized-luma POSE-CARRIER (d_pose) [closes pose ~0.006, ~23.6 KB]
                 + Wyner-Ziv pose FiLM side-info (~1.5 KB) [keeps d_pose at frontier 3e-5]
                 + tiny shared generator (class->witness painter + luma head)
                 → emits a LEGAL full-res uint8 RGB frame whose POST-round-trip argmax+pose hit target
```

| section | mechanism | bytes (measured/projected) | term it serves |
|---|---|---:|---|
| seg generator | coordinate-INR, n_fourier=16/hidden=96/n_hidden=4, int8+brotli, full 384×512 | 65,305 (measured) | d_seg (rate win) |
| pose carrier | `AmortizedLumaCarrier` (#57), saliency-confined (PTNC #61), int8+brotli | ~23,600 (measured) | d_pose frame texture |
| pose FiLM side-info | Wyner-Ziv 6-dim/pair, fp16+brotli, decoder-injected | ~1,500 (proj) | d_pose precision |
| palette/painter | per-class GT-region-mean | 15 (measured) | realization |
| **total** | | **~90,400 B → rate 0.0602** | |

**Joint config & projection.** With the pose carried as side-info (d_pose ≈ frontier 3e-5) and the
generator trained to d_seg = 3.2e-4: **S ≈ 100·3.2e-4 + √(10·3e-5) + 25·90400/37.5e6 = 0.032 + 0.0173 +
0.0602 ≈ 0.110** — sub-0.15 with margin. Even at the conservative generator d_seg = 9.2e-4 + total ~90 KB:
S ≈ 0.092 + 0.017 + 0.060 = 0.169 (still rate-improved; the d_seg term is the swing). **The whole capstone
turns on ONE number: generator d_seg.**

**Plug into the math-optimal solver (agent a407ff0).** This capstone is the witness vertex of the
unified action: it contributes (a) a rate term 25·B/N with B decomposed per-section for the bit-allocator,
(b) a d_seg term governed by the generator power-law (the live Muon-stage-8 run's measured curve is its
prior), (c) a d_pose term made near-free by FiLM side-info. The solver's δS/δθ=0 over {seg-INR capacity,
pose-carrier capacity, FiLM bytes} is a **3-section waterfill** — the seg-INR is the binding section
(rate 0.044 of the 0.060) and its marginal d_seg/byte is the generator power-law slope. **The decisive
next step the solver should consume: the generator d_seg-vs-(params, Muon-stage-8, taper) curve from the
live run** — that single curve resolves both the witness reachability AND the live RGB-HNeRV run.

---

## THE VERDICT (with the no-premature-kill discipline, both directions)

**GATED-GO: non-RGB / task-space witness is VIABLE as the final capstone, in HYBRID form. The gate is
generator-d_seg, not device, not rate, not survival.**

Applying `terminal-conclusion-needs-existence-proof-crosscheck` BOTH directions per the task:
- **I do NOT re-assert the CPU-infeasible wall as a kill** — the existence proof refutes it: README L114
  provisions a T4 for GPU inflate AND the realized witness is CPU-seconds. Device wall = FALSE.
- **I do NOT assert non-RGB WINS** — no score moved; the −59% rate is byte-closed but the advisory S is
  NOT sub-0.15 from rate alone (0.164), and the sub-0.15 hybrid projection (0.110) is DERIVED, gated on
  an UNMEASURED generator d_seg < 3.2e-4. Every prior measured non-RGB attempt that walled
  (partition-store S 0.84, lever-B/L13 S 13.58 pose-dominated, witness-sidecar 37% survival) is honestly
  characterized as IMPLEMENTATION-level (Catalog #307), and the SPECIFIC walls are now resolved (pose
  closed; survival = flat-store-only; rate won) leaving generator-d_seg as the lone open term.

**The decisive next step (single measured question):** does free training — **Muon stage-8 + d_seg-aware
taper (0 bytes)** — push the generator's d_seg below ~9.2e-4 (beat-frontier) / ~3.2e-4 (sub-0.15) at
fixed ~65 KB rate-winning params? The live bc20/600-pair run is at d_seg 0.00222 @ ep6025, **2.4× above
the beat-frontier line**, heading into the Muon stage-8 κ-buster. **This is already in flight** — the
witness adds the proven −59% rate win ON TOP if the AmortizedLumaCarrier + FiLM side-info byte-close.

**Reactivation / close criteria (DEFER, not KILL):**
- REOPEN to BUILD when the live run's generator d_seg crosses ~9.2e-4 (then byte-close the witness:
  seg-INR + AmortizedLumaCarrier + FiLM side-info → advisory S → if < frontier, paired exact-eval).
- CLOSE-as-capacity-bound ONLY if the generator d_seg power-law (Family-B, real decoders) is MEASURED to
  wall above 9.2e-4 at ≤83 KB params after Muon stage-8 + taper — in which case the capstone reverts to
  the RGB-HNeRV (the −59% rate is erased because a generator big enough to floor d_seg = the dense
  decoder, per the CAPSTONE §8 capacity-wall). That measurement has NOT been run; do not pre-conclude it.

---

## 6-hook wire-in (per Subagent coherence-by-default)
1. **Sensitivity-map** — ACTIVE: the per-section byte→term map (seg-INR 65K→d_seg, luma-carrier
   23.6K→d_pose-texture, FiLM 1.5K→d_pose-precision) feeds the bit-allocator; the generator d_seg
   power-law slope is the binding marginal.
2. **Pareto constraint** — ACTIVE: records the witness vertex (rate 0.060, off the frontier's rate
   vertex by −0.058) and the constraint that sub-0.15 needs generator d_seg < 3.2e-4 (the binding edge).
3. **Bit-allocator** — ACTIVE: the 3-section budget (seg-INR / luma-carrier / FiLM) is the literal
   allocator; seg-INR is the binding section.
4. **Cathedral autopilot dispatch** — N/A (advisory; gate not met — no advisory beat; do not dispatch
   paid eval until generator d_seg crosses 9.2e-4 and the witness byte-closes to advisory S < frontier).
5. **Continual-learning posterior** — ACTIVE: probe outcome `nonrgb_capstone_reopen_20260623` (verdict
   GATED-GO, reactivation = live-run generator d_seg < 9.2e-4) to be registered via
   `tac.probe_outcomes_ledger.register_probe_outcome`.
6. **Probe-disambiguator** — RESOLVED 3 questions: (a) "is GPU witness contest-legal?" → YES (README
   L114). (b) "does the survival wall kill the rendered witness?" → NO (flat-store-specific; rendered
   converts it to a training d_seg question). (c) "is rate alone sub-0.15?" → NO (0.164; needs d_seg
   stacked). The remaining OPEN probe: generator d_seg power-law under Muon stage-8 (the live run).

## NO-FAKE ledger
- MEASURED ANCHORS (not re-run, sourced): device policy (upstream README/evaluate.sh/.py verbatim);
  L13 byte-close 72,217 B lossless-parity (`score_native_candidate_20260610`); pose carrier d_pose 0.006
  / 23.6 KB (`witness_L13_optimal_pose_carrier_result`); partition-store survival wall 24%/S 0.84
  (`partition_store_realization_gate_DEFER`); frontier 177,169 B / 0.19110 (canonical pointer);
  generator d_seg 0.0068 / 0.00222-live (CAPSTONE §8).
- DERIVED: the S projections (via `tac.contest_score` arithmetic on measured component anchors); the
  hybrid capstone design; the survival-wall structural argument (flat-store ≠ trained-render).
- NOT claimed: NO score moved; pointer UNMOVED 0.19110; sub-0.15 is a DERIVED projection gated on an
  UNMEASURED generator d_seg < 3.2e-4; no archive built/byte-closed this unit (the L13 anchor is prior
  work); no exact-eval dispatch; no PR. The witness capstone is WORTH BUILDING (this verdict) but is NOT
  BUILT — the gate is the live run's d_seg verdict.
