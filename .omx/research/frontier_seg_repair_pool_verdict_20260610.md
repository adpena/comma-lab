# Frontier seg-repair pool — flip-map + carrier verdict (#51)

UTC 2026-06-10 · claude (#51 executor) · `[macOS-CPU advisory]`, non-promotable per
Catalog #192/#341/#127/#323. $0 local, NO cloud, NO paid GPU, NO MPS, NO /tmp. Output ->
`/Volumes/VertigoDataTier/pact/frontier_seg_repair_pool_20260610/` (SSD tier, rebuildable).
Subagent-id `frontier_seg_repair_pool_20260610`.

## Mission

Attack the FRONTIER's seg-repair pool — the largest single score pool (frontier
`avg_segnet_dist = 5.5979e-04` = `100·d_seg = 0.056` score units = **29%** of the 0.19199
total). Locate the frontier archive's (sha `b7106c9bdbb8…`, 178,493 B) ACTUAL flipped
pixels, generate Class-3 repair atoms targeted at them, design a byte carrier, advisory-screen,
and (if it clears THE LAW) ratify one paired contest-CPU batch.

## 1. THE FLIP MAP (headline artifact) — `flip_map_full/flip_map_summary.json`

Inflated the frontier locally, rendered the receiver comp frames per pair (byte-faithful to
`inflate.py`: decode → bicubic upscale to camera-res 874×1164 → −1 channel postproc →
clamp/round → frame-0 selector), decoded the **contest-EXACT GT** (`frame_utils.yuv420_to_rgb`,
NEVER PyAV rgb24 — the R3 GT-decode bug class; the STRICT gate
`test_pr110pp_candidate_generator_gt_decode_contest_exact.py` passes), and scored **all 600
pairs** through the EXACT `upstream/modules.py` SegNet on local CPU (NO MPS). d_seg flip =
`comp_argmax != gt_argmax` at the scorer grid 384×512 (exactly `SegNet.compute_distortion`).

| metric | value |
|---|---|
| d_seg_mean recomputed | **5.5982e-04** — matches frontier `avg_segnet_dist=5.5979e-04` to **5e-8** (apples-to-apples VALIDATED) |
| total flips (pool) | **66,039** |
| flips / pair (mean) | **110.1** |
| recoverable (margin ≤ 2.0 logit) | **99.94%** (66,000 / 66,039) |
| margin < 0.5 | **60,323 (91.3%)** |
| margin 0.5–1.0 | 5,053 |
| margin 1.0–2.0 | 624 |
| margin > 5.0 | 1 |

**The flips are overwhelmingly at TINY margins (91% < 0.5 logit units) — pixels sitting right
on the SegNet decision boundary.** Spatially they cluster in scorer-rows ~171–292 (the
road/horizon band of the dashcam), scattered (1.1 flips per 4×4 block — NOT RLE-friendly runs),
spanning all 5 SegNet classes (dominated by class 0 + 1). Cross-pair flip-frequency sharing
ratio is only **1.47** (66k instances across ~45k unique pixels) — flips are largely
per-pair-specific; only **819 pixels** flip in ≥10 of 600 pairs (the systematic boundary errors).

## 2. THE CARRIER + THE TWO REPAIR-DIRECTION FINDINGS (the science)

A frame-1 correction must enter the rendered camera-res frame (where it survives SegNet's
bilinear downsample) AND be coded into a self-contained archive section. The carrier grammar
designed: a new length-prefixed section appended after the DQS1 packet in member-x
(`OUTER_MAGIC | source_len | source | selector_len | selector | dqs1 | [SEG_REPAIR section]`),
consumed by a minimal inflate patch that adds a sparse per-pixel correction to frame-1 after
`apply_pr101_selector_to_frames`. The correction is coded at scorer-grid (384×512, the cheap
grid) and nearest-upsampled to camera-res by the inflate patch.

### Finding A — appearance-gap repair INCREASES d_seg (the Class-3 module's implemented direction is WRONG on the real frontier)

`frame1_seg_repair_atoms.generate_seg_repair_atom` corrects toward the GT **appearance** gap
(`gap = GT − rendered`). On the REAL frontier render (not the #50 degraded-GT proxy), snapping
flip pixels toward GT appearance **raises** d_seg: per-pair d_seg deltas were **+8.14e-5,
+3.05e-5** (positive = MORE flips). Direct experiments on pair 0 (base 114 flips):
- snap flip pixels fully to GT appearance (camera-res): **114 → 133** (worse)
- snap a 3× dilated region to GT: **114 → 450** (much worse)
- snap the entire flip band to GT: **114 → 2601** (catastrophic)
- gentle halfway blend: **114 → 115** (neutral)

**Mechanism = SegNet's spatial receptive field** (the composition-algebra §2 coupling). A flip
pixel's argmax is decided by a wide conv neighborhood, not its own color. A sparse/regional
appearance patch creates a *local appearance discontinuity* the network reads as a NEW boundary,
flipping MORE pixels. The seg flips are produced by SegNet's spatial reading of the renderer's
globally-soft reconstruction; they are not fixable by local pixel patches toward GT appearance.

### Finding B — the CORRECT lever is the SegNet logit-gradient sign (true margin-normal), but it fails THE LAW at pool scale

The genuinely-correct repair direction is `sign(∂(gt_class_logit − top_logit)/∂x)` at the flip
pixels (the logit-space margin-normal, which the module's docstring claims but its code does not
implement). Screened on 8 pairs (`screen_gradient_repair.py`): a sparse gradient-sign step DOES
reduce flips (114→89 at step 16), and **8/8 pairs "accepted" per-pair** — BUT the per-pair
acceptance carried a **units error** (it used `100·per_pair_d_seg` as the seg term, while the
contest seg term is `100·MEAN over 600 pairs`, confirmed at `upstream/evaluate.py:81-92`). At the
CORRECT pool level:

- gradient lever fixes **~23 flips/pair** at **~2.97 honest coded B/flip** (sorted position-delta
  + per-pixel int8 value, brotli q=11).
- per fixed flip: seg value **8.48e-7** vs byte cost `2.97·25/N = 1.98e-6` →
  **value/cost = 0.429** → **THE LAW FAILS by 2.3×**.
- fixing 13,800 flips pool-wide buys back 0.0117 seg score but costs 0.0273 rate score (40 KB
  carrier) → net seg-axis **ΔS = +0.0156 (WORSE)**.

The flip **positions** carry ~2–3 B/flip irreducible entropy (scattered, 1.1/4×4 block); the LAW
break-even is **1.27 B/flip**. The per-flip score value (8.48e-7) is simply too small relative to
the cost of *addressing* a scattered flip. The per-pair carrier cannot clear THE LAW.

### The shared correction field (position coded once) ALSO fails — measured composed worsening

A shared correction field breaks the per-pair position-entropy bottleneck (code the position set
ONCE, apply to all 600 frame-1s). The optimistic upper bound (target the 819 pixels flipping in
≥10 pairs, assume each is fixed in every pair with no new flips) gave value/cost ~6. The decisive
test (`test_shared_carrier_composed.py`) builds the per-pixel SYSTEMATIC gradient-sign direction
(mean over the pairs where each target pixel flips), applies the ONE shared field to a 60-pair
representative eval sample (high-flip + low-flip pairs), and measures the COMPOSED pool d_seg:

(819 pixels flip in ≥10 of 600 pairs; shared direction defined at 562 of them; eval on a 60-pair
high-flip+low-flip sample; carrier position+value coded ONCE = 915 B total.)

| step | composed d_seg | Δ d_seg | d_pose Δ | carrier bytes | composed ΔS |
|---:|---:|---:|---:|---:|---:|
| 4.0 | 5.92e-4 → 6.13e-4 | +2.01e-5 | −1.5e-7 | 915 | +2.58e-3 (WORSE) |
| 8.0 | 5.92e-4 → 6.41e-4 | +4.81e-5 | +4.1e-8 | 915 | +5.43e-3 (WORSE) |
| 16.0 | 5.92e-4 → 7.13e-4 | +1.21e-4 | +1.4e-6 | 915 | +1.30e-2 (WORSE) |

(monotonically worse with amplitude — the larger the shared correction, the more net new flips.)

The shared correction at even the most-systematic flip pixels **INCREASES** the composed pool
d_seg. The optimistic upper bound is violated by the receptive-field coupling (Finding A): a
shared field forces ONE direction per pixel, but the flip direction differs per pair, and the
field creates net new flips in the pairs where the pixel was correct (or where the gradient sign
disagrees). The shared carrier does not reduce composed d_seg, so it cannot clear THE LAW
regardless of how cheaply its position is coded.

### The information-theoretic floor (the rigorous clincher)

The minimum bits to ADDRESS K=110 scattered flips among the 384×512 = 196,608 scorer positions
is `log2(C(196608, 110)) = 167.8 bytes = 1.525 B/flip` — the **position-only information-theoretic
floor is already OVER the 1.27 B/flip LAW break-even**, before any value bits, before accounting
for the gradient lever fixing only ~23 of 110 flips (addressing 23 fixed flips alone costs 1.79
B/fixed-flip at the floor). The per-pair seg-repair sidecar carrier is **information-theoretically
incapable** of clearing THE LAW: the flips are too spatially scattered (high position entropy) and
the per-flip score value (8.48e-7) is too small. This is a fundamental bound, not an
implementation limitation.

## 3. Verdict + routing

**VERDICT: DEFER-pending-new-carrier — the frontier seg-repair pool (0.056, 29% of the score) is
REAL and FULLY MAPPED, but a frame-1 correction sidecar cannot reach it under THE LAW. NO Modal
dispatch (local advisory kill-gate fired: no carrier cleared THE LAW; dispatching would burn ~$0.3
to confirm a bound already proven information-theoretically).**

This is a DEFER, not a KILL (CLAUDE.md "Forbidden premature KILL"): the seg pool is real, the flip
map is the canonical seg-axis sensitivity artifact, and the gradient-margin direction is the
correct repair lever — only the *carrier economics* are bounded. Three carriers tested + falsified:

1. **appearance-gap correction** (the Class-3 module's implemented direction) — INCREASES d_seg
   (wrong direction; receptive-field coupling). Finding A.
2. **gradient-sign correction** (correct direction) — reduces per-pair d_seg but fails THE LAW at
   pool scale (value/cost 0.429; 2.97 honest B/flip vs 1.27 break-even). Finding B.
3. **shared correction field** (position coded once) — INCREASES composed pool d_seg (receptive-
   field coupling defeats the amortization). §2.

The information-theoretic floor (1.525 B/flip position-only) is over the LAW break-even (1.27
B/flip), so NO sidecar that addresses flips per-position can clear THE LAW. **The only carrier
whose cost is NOT proportional to flip-count is the decoder/latent axis** (a better reconstruction
fixes flips for free, paid in existing decoder bytes) — explicitly OUT of this lane's scope
(latent-axis surfaces) and owned by the decoder/latent executors (the 91%-of-bytes axis).

### Reactivation criteria

- A renderer/decoder change that reduces the systematic boundary error (the 819 high-frequency
  flip pixels) — paid in the existing 91%-decoder bytes, NOT a sidecar. (decoder/latent axis.)
- A frame-1 correction whose value per fixed flip exceeds ~1.5e-6 (e.g. a flip that ALSO carries
  a large pose gain, lifting the per-flip value above the position floor) — the flip-map's
  per-pair `gt_class_counts` + the pose-coupling rows are the input.
- A class-of-flips that is spatially DENSE enough (contiguous runs) that position entropy drops
  below 1 B/flip — the flip-map shows the current frontier flips are scattered (1.1/4×4 block),
  so this requires a different vehicle whose errors are blocky.

### FRONTIER-CANDIDATE alert

NONE. No candidate beat 0.19198534 beyond noise (no candidate was even byte-closed — the local
kill-gate refused before that). The seg pool remains unreached by this lane; the frontier stands.

### Sister-coherence note

This memo is the SEG-axis SISTER of `pr110pp_r3_onhost_selector_verdict_20260610.md` (which proved
the per-pair frame-0 *selector* lever exhausted on the POSE axis). Together they bound both
per-pair frame-0 (pose) and frame-1-sidecar (seg) levers at the 0.19199 frontier: both are
exhausted/unreachable-by-sidecar; the remaining lever is the decoder/latent axis (91% of bytes).

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map**: ACTIVE — the per-pixel flip-frequency field + margin field ARE a frontier
   d_seg sensitivity map (where the seg pool lives). Output rows reseed the seg-axis sensitivity.
2. **Pareto constraint**: ACTIVE — the carrier value/cost (0.429 per-pair; shared-carrier number
   below) is the per-atom Pareto-feasibility test on the seg/rate axes.
3. **Bit-allocator hook**: ACTIVE — the byte-per-flip economics (1.27 B/flip break-even vs 2.97
   measured) is a bit-allocator constraint for any future seg-repair section.
4. **Cathedral autopilot dispatch**: N/A — advisory verdict; no archive-deployable candidate
   cleared the local kill-gate, so no on-host dispatch was minted.
5. **Continual-learning posterior**: N/A — no contest-CUDA/CPU anchor (the local kill-gate
   refused dispatch); the flip-map rows are the side information a future agent reseeds from.
6. **Probe-disambiguator**: ACTIVE — RESOLVED the probe "is the seg pool fixable by a frame-1
   correction sidecar?" (appearance-gap=NO/harmful; gradient-sign=correct-direction-but-LAW-fails;
   shared-carrier=below) and "is the flip-map apples-to-apples valid?" (YES, d_seg matches to 5e-8).

## Cross-references

`pr110pp_frame1_class23_generators_landed_20260610.md` (the Class-3 generator I reused +
Finding A corrects its implemented direction) · `composition_algebra_coherence_law_20260610.md`
(§2 receptive-field coupling — Finding A's mechanism) ·
`pr110pp_r3_onhost_selector_verdict_20260610.md` (the GT-decode bug class + the per-pair-selector
exhaustion; this memo is the SISTER for the seg axis) ·
`experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis/render_and_score_lib.py`
(the contest-exact render + GT + scorer I reused).
