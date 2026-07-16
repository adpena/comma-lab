# Necessity §9 calibration — DP-ε seed → d_seg through the REAL scorer chain (2026-07-16)

**Source:** P0-TO-RULE-THEM-ALL crown-advance (operator 2026-07-15). The §9 calibration named in
`p0_UNIFICATION_projection_preimage_SUPREME_20260715`: convert the necessity solver's K-ladder
geometric seed into a REAL measured d_seg by *realizing* the witness from the (generator, seed)
design and pushing it through the real scorer preprocess + frozen SegNet. Pointer **0.19108 UNMOVED**
— everything here is MEANS; the exact `upstream/evaluate.py` CPU/CUDA row is the only authority.

**Axis / honesty:** `[macOS-CPU advisory]` — frozen CPU-torch fp32 SegNet on the bit-exact cached GT
argmax (`gt_n600.npz`, n600, ALL scored pairs). `score_claim=false; promotable=false; research_only`.
d_pose is the BANKED witness R1 dxi row (0.001610, 7.2 KB; L68), not re-measured. The score-authority
follow-on is `upstream/evaluate.py` on a byte-closed archive (operator-GO'd).

**Tool:** `tools/necessity_dseg_calibration.py` (committed `b625194009`; resumable, coder
bit-identical to the necessity-solver kladder stage — verified 756164/287105 @ε=1).
**Artifacts:** `experiments/results/necessity_dseg_calibration_20260715/{summary.json, rows_eps*.jsonl, palette.json}`.

---

## The realization (the (generator, seed) witness, per the necessity frame)

- **seed** [COUNTED]: DP-simplified per-class region contours at tolerance ε px @ (384,512), coded by
  the kladder coder (first vertex u16 + int16 deltas + brotli -q11). ε=0 = the exact lossless partition.
- **generator** [FREE, rule 118]: `fillPoly` even-odd raster (holes handled) + area paint-order
  overwrite + nearest-neighbor gap fill + per-class **median palette** fill (5×3 = 15 B, chosen over
  mean by a 6-frame smoke) + `INTER_NEAREST` upsample to camera res (874,1164).
- **static hood-tex seed** [COUNTED, +1759 B]: the necessity frame's "Road-MyCar → static one-time
  seed" element, extended to the MyCar **interior**: frame-0 GT texture, 16× downsampled, brotli.
- **chain**: uint8 camera frame → float (raw 0-255, no normalization) → bilinear (384,512) → frozen
  SegNet → argmax vs cached `lstars`. Exactly the upstream `SegnetWrapper` preprocess.

## The measured ε→S curve (n600, adjusted seed, d_pose banked; `S = 100·d_seg + √(10·d_pose) + 25·bytes/N`)

| arm | d_seg (real) | 100·d_seg | seed KB (adj) | rate | **S** |
|---|---:|---:|---:|---:|---:|
| ε=2 | 0.08938 | 8.938 | 97.9 | 0.070 | 9.135 |
| ε=1 | 0.07637 | 7.637 | 143.6 | 0.100 | 7.864 |
| ε=0.5 | 0.05099 | 5.099 | 232.6 | 0.160 | 5.385 |
| ε=0 (lossless) | 0.04538 | 4.538 | 228.8 | 0.157 | 4.823 |
| ε=1 + hood-tex | 0.01397 | 1.397 | 143.6 | 0.102 | 1.625 |
| **ε=0 + hood-tex** | **0.01328** | **1.328** | **228.8** | **0.158** | **1.613** ← min-S |

`dseg_geo` (reconstructed-partition vs GT-partition disagreement, geometry only): ε=0 → 0.0, ε=0.5 →
0.00039, ε=1 → 0.00261, ε=2 → 0.00484.

## The two decisive score-dynamics facts (per the min-S / waterfill steer)

1. **DP-ε geometry HURTS d_seg — ε* = 0 (lossless).** d_seg rises monotonically with ε (0.045 → 0.051
   → 0.076 → 0.089). The SegNet argmax **amplifies** the tiny geometric DP error ~14× (ε=0.5:
   geo Δ0.00039 → real Δ0.0056). Spending rate on *coarser* geometry buys NEGATIVE ΔS — anti-waterfill.
   The lossless partition is both the min-d_seg and (with hood folded) the min-S geometry choice.
2. **Static hood-tex is the single decisive waterfill buy.** At ε=0 it cuts d_seg 0.04538 → 0.01328
   (**−71%**) for **+1759 B** (+0.00117 rate). ΔS = **−3.21** (−3.211 d_seg, +0.0007 rate).
   ΔS/byte ≈ **−1.8e-3** — by far the highest-value buy on the board (a flip is worth ~1.3 bytes; the
   hood cures ~48k flips per the analysis for 1759 bytes). This confirms the coordinator's steer and
   the necessity frame's "Road-MyCar → static one-time seed" prediction, MEASURED and extended to the
   MyCar interior. **It is a standalone, vehicle-agnostic carrier** (static, one-time) that composes
   with any render.

## THE MIN-S OPERATING POINT (the waterfill knee) — the verdict

**ε=0 (lossless partition) + static hood-tex: S = 1.613, d_seg = 0.01328, rate = 0.158.**

**This is 8.4× ABOVE the 0.19108 pointer — the frame does NOT predict sub-0.19108 by this vehicle.**

Why the knee stops there (no further cheap buy): after the hood, the residual d_seg 0.01328 attributes
(% of disagreeing px) to **52% edges** (Road-Lane 368k, Road-MyCar 199k, Road-Undrivable 145k) +
**22% near-edge** + **24% Movable interior** + 0.8% saddle. Every remaining bucket is EITHER SegNet's
response to the sharp flat-palette **discontinuity** (edges/near-edge — needs a texture GRADIENT across
the boundary, not geometry) OR a **dynamic** object (Movable interiors move frame-to-frame — a static
seed cannot fix them; per-frame texture is neither cheap nor static). The next d_seg buy is not a cheap
seed — it is the trained render itself.

## HONEST VERDICT — FORMULATION-scoped (which stratum's generator is insufficient + the reformulation)

`verdict_scope: FORMULATION` — the negative is the **palette + geometry CELL-interior generator**, on
this vehicle, measured at optimal form (median palette beat mean; ε swept incl. lossless; static hood
folded; per-frame + noise palette variants smoked and REJECTED). It is NOT a paradigm kill of the
necessity/projection/Kolmogorov frame.

**What the necessity frame gets RIGHT (confirmed):**
- The **RATE floor is real and cheap**: edges 143 KB @ ε=1 (K/H = 0.47, from the solver). The
  (generator, seed) factorization is the correct *rate* accounting.
- The **static hood-tex carrier is a genuine, transferable, near-free d_seg lever** (−71% of the
  value-wall for 1.7 KB) — a per-stratum (generator, seed) win exactly as the frame predicted for the
  Road-MyCar / MyCar stratum. This composes with the witness.

**What is insufficient (the reformulation):**
- The **cell-interior generator must be the JOINT-trained render (the witness's actual output), NOT
  flat-palette value-fill.** Even at the EXACT lossless partition (geo error = 0) + static hood, SegNet's
  argmax on the flat-palette cartoon disagrees at d_seg **0.01328** — ~2.6–3.9× the current witness
  d_seg (~0.0034–0.005) and 8.4× the score target. This is the **photometric wall (L68) restated at the
  cell-interior surface**: a frame not JOINTLY shaped for the scorer does not carry argmax-faithful
  photometric signal; no post-hoc storage of a value palette crosses it — only the trained render does.
- **The trilemma is confirmed, not resolved, by geometry storage** (L12/L17): storing the partition
  geometrically is BOTH expensive (lossless = 0.157 rate, *above* the 0.118 frontier) AND d_seg-poor
  (0.013). Only ε≥1 gets under frontier rate, at catastrophic d_seg. The witness resolves the trilemma
  by AMORTIZING (a trained coord-INR is cheaper than storing the partition AND argmax-faithful because
  it is trained *through* the scorer). **The necessity frame is the RATE-FLOOR ANALYSIS + a per-stratum
  carrier catalog, not itself the d_seg vehicle.**

## Routing (V9·CGauge / the endgame)

1. **FOLD the static hood-tex seed by default** into the witness archive — 1.7 KB, static, one-time,
   −71% of the MyCar value-wall; near-free and vehicle-agnostic. Highest ΔS/byte carrier measured.
2. **Do NOT store the partition geometrically as the cell content** — it is expensive AND d_seg-poor.
   Cell interiors are the trained render's job (the witness). The necessity EDGE seed (143 KB) is the
   rate FLOOR the trained render is measured against, not a replacement for it.
3. **ε* = 0** wherever geometry IS stored (edges): DP simplification hurts d_seg via SegNet
   amplification; lossless contours are the min-d_seg geometry.
4. The binding d_seg term is **cell-interior value/texture realization + boundary texture-gradient**,
   NOT geometry — this is where the trained render earns its d_seg, and where the sub-pixel
   appearance-phase geometry (L71/L85/L86) owns the Lane realization floor.

## Triality
- **DAG** = `FEED-necessity-cal` (this measurement, the ε→S curve, the min-S knee, the formulation verdict).
- **equations** = `necessity_generator_seed_dseg_calibration_v1` (registered; the min-S-through-real-decode
  law + the two measured facts: DP-ε amplification and the hood-tex ΔS/byte carrier).
- **DSL** = [no-triality] — this arm is a *measurement + a negative verdict*, not a new config Lever. The
  hood-tex carrier, when DSL-wired into the witness archive builder, lands as a per-stratum seed Lever
  update at the archive-build surface (routing item 1); recorded as owed.

**Pointer 0.19108 UNMOVED — MEANS.** The verdict redirects the crown's d_seg realization back onto the
trained witness (cells) + the sub-pixel geometry (Lane floor), keeps the necessity frame as the rate-floor
+ per-stratum-carrier authority, and banks the static hood-tex as a folded near-free d_seg carrier.
