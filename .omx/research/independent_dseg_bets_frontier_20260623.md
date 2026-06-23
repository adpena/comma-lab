# Independent $0 d_seg bets on the 0.19110 CPU frontier — measured verdicts

- **Date:** 2026-06-23
- **Subagent:** `indep_dseg_bets_20260623`
- **Axis:** `[contest-CPU advisory]` — single video `0.mkv` (1200 frames = 600 pairs) reproduces
  the contest 600-sample eval locally. Authority = exact frozen CPU SegNet argmax-disagreement
  through the exact preprocess. NOT a proxy.
- **Pointer:** UNMOVED at 0.19110. Any win here would still need byte-close + exact eval before a claim.

## Setup / authority validation (NO FAKE)

- Frontier archive: `experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip`
  sha256 `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e`, 177169 bytes,
  lane `lane_pr110_payload_entropy_recode_20260610` (matches `canonical_frontier_pointer.json`).
  report.txt: d_seg 0.00055978, d_pose 0.00002942, rate 0.00471878, score 0.19.
- Inflated to 1200 camera-res frames via the frontier's own `inflate.py` (deterministic, byte-closed).
- GT decoded via `upstream/frame_utils.yuv420_to_rgb` (BT.601 limited-range, bilinear chroma —
  **never** PyAV rgb24). SegNet = frozen `upstream/models/segnet.safetensors`, exact preprocess
  (last frame of pair → bilinear-resize to 512×384 → SegNet → argmax over 5 classes).
- **Harness validation:** measured 600-pair baseline `d_seg = 0.00055989` vs report `0.00055978`
  (Δ = 1e-7, 5-decimal match) → the harness is exact-scorer faithful.
- Tools: `experiments/probe_indep_dseg_bets_frontier.py` (hood/geo), 
  `experiments/probe_gate128_frame1_selector_headroom.py` (selector headroom).

## STRUCTURAL ROOT FINDING (explains all three NO-GOs)

The residual frontier d_seg is **scattered, content-dependent SegNet boundary noise at the
HORIZON band**, not a static/geometric region:

| SegNet-grid row band | region | share of total d_seg |
|---|---|---|
| rows 0–96 | sky/top | **0.0%** |
| rows 96–192 | upper (horizon) | 41.0% |
| rows 192–288 | mid (road-horizon) | 56.8% |
| rows 288–345 | lower/road-near | 2.2% |
| rows 345–384 | **hood** | **0.0%** |

- 97.8% of d_seg lives in rows 96–288 (the horizon/distant-scene band).
- **No static flip:** max per-pixel flip-frequency across 600 pairs = **5.8%**; 0 pixels flip in
  >20% of pairs; 88.7% of pixels NEVER flip. The disagreement is at dynamic distant-content
  semantic boundaries — there is no always-wrong pixel/region a 0-byte clamp can target.
- Per-pair d_seg is uniform (max 0.00106 vs min 0.00031, only 3.4×; top-10% of pairs carry just
  13.8% of total) — no concentrated bad pairs to exploit.
- Flips spread across all 5 classes (GT-class share: cls0 41.5%, cls1 29.2%, cls2 13.5%,
  cls3 12.4%, cls4 3.5%).

## Gate #139 — ego-hood static-region clamp → **NO-GO (measured)**

The hood is a fixed bottom band; GT SegNet argmax there is **100% class 4 (uniform)**. Measured
hood-band flip fraction across hood fracs 0.84–0.94 (bottom 6%–16% of grid), 600 pairs:

| hood row frac | band % of frame | flip frac in band | band share of d_seg | 0-byte clamp Δd_seg | oracle (force→GT) Δd_seg |
|---|---|---|---|---|---|
| 0.84 | 15.9% | **0.00000000** | 0.0000 | +0.00000000 | +0.00000000 |
| 0.90 | 9.9% | **0.00000000** | 0.0000 | +0.00000000 | +0.00000000 |
| 0.94 | 6.0% | **0.00000000** | 0.0000 | +0.00000000 | +0.00000000 |

**Verdict: NO-GO.** The hood has ZERO argmax-flips on all 600 pairs — the frontier already
reconstructs the static hood with zero SegNet disagreement. There is no d_seg to remove; a clamp
would save 0 and cost real bytes. Honest negative.

## Gate #138 — openpilot road/lane geometric prior → **NO-GO (measured)**

Ground-plane road trapezoid (horizon row 0.50·H → hood-top 0.90·H, widening), 22.6% of frame, 600 pairs:

- flip fraction in trapezoid = 0.000426; trapezoid carries **17.2%** of total d_seg
  (≈ proportional to its 22.6% area — NOT a hotspot).
- GT inside the trapezoid is **mixed-class** (dominant class only 55.6%), so a data-independent
  geometric prior is wrong ~half the time.
- **0-byte geometric prior** (force comp argmax → trapezoid-major class): **Δd_seg = +0.10019**
  (catastrophic — would raise score by ~+10).
- Even the **oracle upper bound** (force comp → GT inside trapezoid, which REQUIRES GT and is
  therefore NOT 0-byte): only **Δd_seg = −0.0000963**.

**Verdict: NO-GO.** The road region is not a flip hotspot, it is mixed-class so a geometric prior
hurts, and even an oracle (non-0-byte) override buys negligible d_seg. The IoU $0 gate fails:
the geometric region does not align with the argmax-flip set. Honest negative.

## Gate #128 — frontier sub-linear seg lever (FECa selector / DQS1) → **NO-GO (measured)**

Frontier's existing sub-linear levers, decoded from the archive:
- **FECa selector** (222 bytes): per-pair pick over a 31-mode palette, K=16 active. **Decoded the
  600 picks: ALL use palette idx 0–15 — every one is `none` or a `frame0_*` bias.** None touch
  frame_1 (the SegNet-scored last frame). The selector is tuning **d_pose** (frame_0 of the pair),
  leaving d_seg entirely to the base reconstruction.
- **DQS1** (42 bytes): per-pair decoder-q substitution, frame policy `segnet_last_frame_only`.

The unexploited sub-linear d_seg lever = the frame_1 palette modes (idx 22–30: luma ±1/±2, RGB
bias, blue-chroma). Decisive oracle headroom test — for each pair, apply each frame_1 mode to the
comp last-frame (the exact camera-res op the inflate applies before uint8 cast), re-run frozen
SegNet, take the per-pair d_seg-minimizing mode (oracle, uses GT):

| sample | baseline (none) | oracle best-frame1-mode | Δd_seg | rel reduction | modes chosen |
|---|---|---|---|---|---|
| 25-pair oracle | 0.00052572 | 0.00052470 | **−1.02e-6** | 0.19% | 22/25 keep `none` |

(A full-600 run was started but is purely confirmatory — the 25-pair **oracle** ceiling is already
decisive, so it was stopped to conserve CPU per the cheapest-decisive-first discipline. The oracle
cheats with GT to pick each pair's best mode; any real byte-encoded selector does strictly worse.)

**Verdict: NO-GO (oracle ceiling negligible).** The selector modes are ±1/±2/±4 pixel biases —
far too weak to flip SegNet argmax at the scattered horizon-band boundary pixels that drive d_seg.
Even the oracle per-pair pick (cheating with GT) buys ~0.2% relative (~1e-6 absolute), and the
bytes would be ADDITIONAL (the frame_0 selector tunes d_pose and cannot be reused). Pushing the
existing sub-linear lever harder on d_seg does not move the pointer.

## Ranked verdict

1. **#139 hood clamp — NO-GO.** Zero hood flips; nothing to clamp. (decisive, 0.0000)
2. **#138 road/lane geometric prior — NO-GO.** Not a hotspot, mixed-class, 0-byte prior +0.100. (decisive)
3. **#128 frontier selector seg lever — NO-GO.** Oracle frame_1-mode ceiling ≈ −1e-6 (0.2% rel);
   pixel biases can't flip horizon-boundary argmax; bytes are additive. (decisive)

**Single most promising $0 pointer-move path: NONE of the three.** The frontier's residual d_seg is
a diffuse, content-dependent SegNet boundary-disagreement floor at the horizon band, with no static
region, no geometric structure, and no per-pair categorical pixel-bias headroom. A real d_seg
reduction requires changing the **base reconstruction fidelity at the horizon band** (more decoder
capacity / a horizon-targeted residual / score-aware training) — i.e. a trainer-side lever, not a
$0 frontier-side transform. These are honest negatives that rule out three cheap paths.

## 6-hook wire-in (per Catalog #125)

- #1 sensitivity-map: ACTIVE — measured spatial d_seg distribution (horizon band 97.8%, hood 0%,
  per-pixel flip-freq map) is a reusable seg-sensitivity prior for the bit-allocator / trainer.
- #2 Pareto: N/A — no admitted candidate (all gates NO-GO).
- #3 bit-allocator: ACTIVE (advisory) — "d_seg lives at SegNet rows 96–288" routes any future
  horizon-targeted residual/capacity spend.
- #4 cathedral autopilot dispatch: N/A — advisory, non-promotable, no archive change.
- #5 continual-learning posterior: N/A — `[contest-CPU advisory]`, non-promotable per the MLX/CPU
  advisory rules; no exact-eval row produced.
- #6 probe-disambiguator: ACTIVE — `probe_indep_dseg_bets_frontier.py` +
  `probe_gate128_frame1_selector_headroom.py` are the disambiguators for these three paths.

Mission contribution: `frontier_protecting` (rules out three $0 paths as dead, conserving budget /
preventing a misleading "free d_seg" claim). Pointer UNMOVED 0.19110.
