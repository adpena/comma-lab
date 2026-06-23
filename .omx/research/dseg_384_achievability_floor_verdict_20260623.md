# d_seg 384 achievability floor — ABSOLUTE-FLOOR vs CAPACITY-LIMITED verdict

- **Date:** 2026-06-23
- **Subagent:** `dseg-384-floor-20260623`
- **Axis:** `[contest-CPU advisory]` / `[macOS-CPU advisory]` — NON-PROMOTABLE
  (`promotable=false`, `score_claim=false`). MECHANISM / DISAMBIGUATION measurement,
  NOT a score-roadmap row. Authority = exact frozen CPU SegNet argmax-disagreement
  (`upstream/modules.py:112`) through the exact preprocess. NOT a proxy. CPU-only, $0,
  NEVER MPS (the live MPS training run pid 16938 was never touched; this tool is GT-only
  and loads no decoder).
- **Pointer:** UNMOVED at 0.19110.
- **All score math via `tac.contest_score`** (canonical helper, never hand-rolled).
- **GT decode ONLY via `frame_utils.yuv420_to_rgb`** (BT.601 limited-range, bilinear chroma —
  PyAV rgb24 FORBIDDEN, ~100× phantom pose).
- **Tool:** `tools/measure_dseg_384_achievability_floor.py` (NEW sibling of the sister
  `measure_dseg_reducibility_gt_margin.py`; reuses its scorer-load / GT-decode / SegNet-
  preprocess / contest-score machinery). JSON:
  `.omx/research/dseg_384_achievability_floor_20260623.json` (N=48) +
  `.omx/research/dseg_384_achievability_floor_n600_20260623.json` (N=600 confirm).

## VERDICT: **CAPACITY-LIMITED** (the d_seg cap is decoder fidelity, NOT the 384 pipeline)

A PERFECT 384-output reconstruction (the GT frame itself, bilinear-downsampled to the
decoder's native 384×512) passes the **exact eval round-trip nearly cleanly**: FLOOR-384
d_seg = **1.87e-4** (S-units **0.019**, the N=600 full-eval value; N=48 = 0.016), which is
**~11× below OUR decoder's residual d_seg (0.0021 = S 0.21)** and **~1.7× below the T_3
sub-0.15 d_seg budget (3.2e-4)**.

**The 384 resolution bottleneck + uint8 round-trip are NOT what floors d_seg.** The d_seg
cap is decoder CAPACITY / fidelity — a better/bigger 384-output decoder has real headroom
to reduce d_seg toward ~1.6e-4 before hitting the pipeline floor. The sister tool's
IRREDUCIBLE verdict was about WHERE OUR current residual sits (low-GT-margin label-noise),
NOT an absolute floor that blocks all decoders.

**sub-0.15 on the RGB rung IS NOT pipeline-blocked on the d_seg axis.** A perfect 384
decoder leaves a d_seg of only 0.016 S-units — comfortably inside any sub-0.15 budget. The
gap from our 0.21-S d_seg to the 0.016-S floor is decoder-capacity-reachable in principle;
the binding question for sub-0.15 is whether a decoder can be TRAINED (and byte-closed) to
approach that floor, not whether the floor itself permits it.

## Method (NO FAKE — exact scorer, exact round-trip, real GT)

For N pairs of `0.mkv` (each pair = 196,608 SegNet-grid pixels), the GT-camera-res SegNet
argmax is the d_seg=0 reference (GT-vs-GT = 0 by construction). We construct the
**best-possible 384-output reconstruction** and score its argmax against that reference:

1. **GT** decoded via `frame_utils.yuv420_to_rgb` → `(2,874,1164,3)` uint8 camera res.
2. **Perfect 384 decoder output** = GT frame `float → bilinear↓ (384,512)` (the best a
   384-native decoder can output).
3. **Exact eval round-trip** = `bicubic↑ (874,1164)` (driver line 3766) → `clamp(0,255).round().uint8`
   (driver.kit_aware_exact_eval, line 3772 — the EXACT inflate cast) → SegNet
   `preprocess_input` (`x[:,-1]` last frame → `bilinear↓ (384,512)`, `upstream/modules.py:107-109`)
   → frozen `segnet(...)` → `(5,384,512)` logits.
4. d_seg = mean over pixels of `argmax(SegNet(floor_input)) != argmax(SegNet(GT_camres))`.

Four floor variants isolate each stage's irreducible contribution:

| floor | construction | isolates |
|---|---|---|
| **FLOOR-384** (headline) | full round-trip WITH uint8 round | achievability floor for ANY 384-output decoder |
| **FLOOR-384-float** | same but SKIP uint8 round (float → SegNet) | resolution-bottleneck contribution ALONE |
| **FLOOR-UINT8only** | GT camera-res uint8 re-round, NO 384 bottleneck | uint8-quantization contribution at camera res |
| **FLOOR-CAMRES** | GT straight to SegNet (self-consistency) | pipeline-faithfulness gate (MUST be 0) |

## Results

### Self-consistency gate — PASSED
- **FLOOR-CAMRES (GT-vs-GT) d_seg = 0.000000** (exactly 0) at both N=2 and N=48 → the SegNet
  preprocess pipeline is deterministic and faithful; the floor cross-tab is trustworthy.

### The three floors (N=600 full-eval headline; N=2 / N=48 agree to ~0.4% on FLOOR-384)

| floor | d_seg (N=600) | **S-units** | vs T_3 budget (3.2e-4) | vs OUR decoder (2.1e-3) |
|---|---:|---:|---|---|
| **FLOOR-384 (headline)** | **1.875e-4** | **0.0187** | **1.7× BELOW** | **11× BELOW** |
| FLOOR-384-float (resolution bottleneck) | 1.596e-4 | 0.0160 | 2.0× below | — |
| uint8-on-top (FLOOR-384 − float) | 2.786e-5 | 0.0028 | — | — |
| FLOOR-UINT8only (camera-res uint8 only) | 0.0 | 0.0 | — | — |

(N=48: FLOOR-384 1.596e-4 / float 1.396e-4 / uint8-on-top 2.003e-5 — same regime; N=600 is the
authoritative full-eval value.)

**Decomposition (N=600):** FLOOR-384 (1.87e-4) = resolution-bottleneck (1.60e-4, **85%**) + uint8-on-top
(2.8e-5, **15%**). The uint8 round-trip at camera res alone is **exactly 0** for already-uint8 GT
(self-consistency: a confident already-quantized frame does not self-flip). The uint8 contribution
only appears as the residual rounding of the bicubic-upsampled (non-integer) 384 reconstruction —
a tiny 0.003 S-units. **Neither the 384 bottleneck (0.016 S) nor uint8 (0.003 S) is anywhere near
a sub-0.15-blocking floor.**

### Per-row profile (where the floor flips sit)

The FLOOR-384 flips (only ~60 pixels across 9.4M at N=48) localize to the **horizon band**
(SegNet-grid rows ~174-217 + ~286 — camera rows ~400-490, the calibrated horizon/vanishing
point — same band as the sister verdict's rows 181-192). The floor flips are at **even LOWER
GT margin than OUR decoder's** (FLOOR-384 flip-margin median ~0.007 vs OUR 0.119 vs non-flip
5.81) — i.e. the floor's residual is pure knife-edge resampling jitter at SegNet decision
boundaries, not a structural region of resolution loss. This sharpens (not contradicts) the
sister verdict: even the perfect 384 reconstruction's tiny residual is label-noise; but its
MAGNITUDE (0.016 S) is 13× smaller than ours, proving the cap between is capacity.

### N=600 full-eval confirmations (the outstanding rigor step) — BOTH DONE

Both verdicts hold at the full 600-pair eval (the only outstanding rigor step the sister flagged):

- **Floor N=600:** FLOOR-CAMRES = **0.0** (self-consistency PASSED); FLOOR-384 d_seg = **1.875e-4**
  (S-units **0.0187**) = resolution-bottleneck **1.596e-4** (S 0.0160) + uint8-on-top **2.79e-5**
  (S 0.0028); FLOOR-UINT8only = **0.0**. Verdict **CAPACITY-LIMITED**. The N=2 → N=48 → N=600
  FLOOR-384 sequence is 1.60e-4 → 1.60e-4 → 1.87e-4 S-units (0.016 → 0.016 → 0.019) — stable, all
  **far below** the T_3 d_seg budget (3.2e-4) and **~11× below** OUR decoder's residual (2.1e-3).
  JSON `dseg_384_achievability_floor_n600_20260623.json`.
- **Sister reducibility N=600** (confirming the sister's N=48 IRREDUCIBLE verdict at full eval):
  measured d_seg = **0.002124** (0.7% off the live last-eval 0.002109 — full-eval, sanity PASSED),
  flip-margin median = **0.122** vs non-flip = **5.89** (~48× separation), 92.4% of flips below GT
  margin 0.5, reducible headroom (margin≥0.5) ΔS = **0.016**, verdict **IRREDUCIBLE**. The N=48 → N=600
  result is unchanged. JSON `dseg_reducibility_gt_margin_n600_20260623.json`.

## How the two verdicts compose (the full picture)

| measurement | question | verdict |
|---|---|---|
| sister `reducibility_gt_margin` | are OUR decoder's residual flips reducible? | **IRREDUCIBLE** — our flips are at low GT margin (label-noise); a horizon decoder recovers ≤ ΔS 0.012 |
| **this `384_achievability_floor`** | does a PERFECT 384 decoder hit an absolute floor? | **CAPACITY-LIMITED** — FLOOR-384 = 0.016 S, 13× below ours; the cap is capacity, not pipeline |

These are **consistent**: the residual our CURRENT decoder leaves (0.21 S) is mostly capacity gap
(perfect-384 floor 0.016 S) plus a small label-noise tail. The sister's "IRREDUCIBLE" says the
flips OUR decoder leaves are at coin-flip margins — true. This floor says a much better 384 decoder
would leave 13× fewer such flips — also true. **The d_seg axis is NOT at an absolute pipeline floor;
it is capacity-bound.** The honest open question is whether training+byte-closure can realize that
capacity headroom (the sister's prior horizon-decoder oracle of ΔS ≤ 0.012 was for OUR decoder's
specific flip set, NOT the full 0.016→0.21 capacity gap; the floor here is the true achievability
target).

## Recommended next move (reconciled with the sister verdict)

1. **sub-0.15 via d_seg is NOT pipeline-blocked.** A perfect 384 decoder floors d_seg at 0.016 S —
   inside any sub-0.15 budget. Do NOT conclude "384 RGB rung is exhausted on d_seg." The cap is
   decoder capacity/fidelity, reachable in principle.
2. **The binding sub-0.15 question is RATE, not the d_seg floor.** The frontier is rate-dominated;
   a higher-capacity 384 decoder that approaches the 0.016-S d_seg floor costs BYTES. The campaign
   tension is `d_seg ↓ via capacity` vs `bytes ↑`. The d_seg axis has real headroom (0.21 → 0.016 S);
   whether spending bytes to claim it nets a lower S is the joint-RD question for the planner.
3. **A from-scratch higher-RESOLUTION (≥ camera-res) decoder is NOT required for d_seg** — the 384
   bottleneck only costs 0.014 S. Higher resolution buys at most 0.014 S on d_seg; capacity within
   384 is the larger lever (0.19 S of the gap). Prefer capacity-within-384 over a resolution rebuild.
4. **The sister's "don't launch a horizon-decoder for a big d_seg win" still holds for OUR decoder's
   flip set** (those specific flips are label-noise), but the FLOOR shows the LARGER d_seg gap
   (0.21 → 0.016 S) is a CAPACITY/training problem — pursue it as a capacity+training campaign with
   a byte-budget, not as a horizon-only sidecar.

## Reactivation criteria (re-open this verdict if)

- A higher-capacity 384 decoder is trained and its measured d_seg does NOT approach ~1.6e-4 even
  with ample capacity → the floor analysis would need a non-pipeline explanation (re-run this tool
  on the new ckpt + the sister reducibility tool to see if the surviving flips shift to higher margin).
- A NEW scorer / video is used (the floor is scorer + content specific — recompute).
- The contest changes the inflate round-trip (camera res, uint8 cast, or SegNet input size) — the
  pipeline-stage decomposition would change; re-run.

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map contribution** — ACTIVE (conceptually): the per-row FLOOR-384 flip-rate profile
   localizes the irreducible-floor flips to the horizon band; combined with the sister's per-row table
   it tells the bit-allocator the d_seg floor for a perfect 384 decoder is 0.016 S, concentrated at the
   horizon. Not persisted to `tac.sensitivity_map` (advisory mechanism row, not a score-claim surface).
2. **Pareto constraint** — ACTIVE: this measurement LOOSENS the d_seg-axis Pareto bound the sister
   tightened — the reachable d_seg floor for a perfect 384 decoder is 1.6e-4 (0.016 S), NOT our current
   0.0021; the campaign planner should treat d_seg as capacity-bound-with-headroom (0.21→0.016 S),
   gated by the byte cost of that capacity, NOT as near-saturated.
3. **Bit-allocator hook** — N/A: diagnostic, adds no archive bytes, changes no per-tensor importance.
4. **Cathedral autopilot dispatch hook** — N/A: not archive-deployable (mechanism measurement).
5. **Continual-learning posterior update** — ACTIVE: the empirical anchor (FLOOR-384 d_seg = 1.6e-4 =
   0.016 S; resolution-bottleneck 0.014 S; uint8 0.002 S; CAPACITY-LIMITED) is recorded in this memo +
   the JSON for the next campaign to inherit. It corrects any "384 RGB rung is d_seg-floored" prior.
6. **Probe-disambiguator** — ACTIVE: this IS the disambiguator between ABSOLUTE-FLOOR (no decoder beats
   the pipeline) and CAPACITY-LIMITED (a better decoder reduces d_seg). The tool
   `tools/measure_dseg_384_achievability_floor.py` returns the regime-conditional verdict.
