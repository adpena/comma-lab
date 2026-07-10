# PALETTE-ARTIFACT PROBE — is the naive-palette realized ceiling (F=0.0337) a PALETTE ARTIFACT? — 2026-07-10

**Operator (2026-07-09):** the C1 transfer-ceiling row realized the PERFECT direct partition (GT `L*`)
as per-class MEAN-RGB → R → frozen CPU-torch SegNet and got realized d_seg **F ≈ 0.0337** (16-strided
subset; `negaudit_retests_c1_e5_20260709.md`). *"That seems like a big number which should be smaller
indicating something we are overlooking."* Hypothesis under test: F is a **palette artifact** — R blends
two adjacent class colours into a THIRD class's RGB region — so a mixing-robust palette (a FREE design
variable) drops F to boundary noise.

**Axis:** `[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]`, **n600** (all 600
pairs, real `gt_n600.npz`), canonical `tac.through_r.measure_through_r`. **$0, CPU, no GPU. Pointer
contest-CPU 0.19110 UNMOVED — this is MEANS.** Result JSON:
`/scratchpad/palette_probe_result_600.json`; verdict + 42 MeasurementRows:
`.omx/research/palette_artifact_probe_verdict_20260710.json`; module `src/tac/through_r/palette_realization.py`
(+17 tests); equation `palette_realization_ceiling_context_dominated_v1`.

**STORES CONSULTED:** `negaudit_retests_c1_e5_20260709.md` (the C1 ceiling row + method) · `src/tac/through_r/`
(canonical R + frozen SegNet + compare) · #149 camera-res placement · #141 margin-saliency ·
`curvelet_directional_basis_dseg_reduction` / C1 register row · MEMORY L68/L17 (trained witness 0.0048) ·
`docs/operating_manual_craft_handoff.md` (§4 re-derive-from-primary · §5 label · §9 measured-runnability).

---

## THE RESULT — hypothesis REFUTED; the overlooked thing is the OPPOSITE of a fixable palette artifact

**U1 canary (n64):** real GT frame1 through R → SegNet gives d_seg **1.6e-7 (~0)** → **all of F is from
the painting**, not R. Confirmed as the premise stated.

### F per arm (n600, realized-through-R d_seg vs L*)

| arm | palette | realization | **F** | third-class % (manufactured/context) |
|---|---|---|---:|---:|
| 1 **baseline** | per-pair mean | render-grid, full R | **0.048323** | 85.1% |
| control | global mean | render-grid, full R | 0.070947 | 89.3% |
| 2a mixing-robust **abstract** | max-logit class colours | render-grid, full R | **0.504375** | 98.7% |
| 2b mixing-robust **scene-anchored** | global-mean + nudge | render-grid, full R | 0.070947 | 89.3% |
| 3 camera-res paint | global mean | camera-res (only DOWN mixes) | 0.072868 | 89.3% |
| 4 **boundary-snapped** | global mean | hard seg-grid, **ZERO R mixing** | 0.050274 | 85.5% |
| 4b **boundary-snapped** | **per-pair mean** | hard seg-grid, **ZERO R mixing** | **0.041622** | 83.0% |

(The n600 baseline is 0.0483; the operator's 0.0337 was a favourable 16-strided subset — both "big".)

### The decomposition IS the finding (two independent lenses AGREE)

- **Resolution mixing (R) contributes only 0.0067 = 14%** of the baseline 0.0483 (`F_naive − F_floor_perpair`
  = 0.048323 − 0.041622). **86% survives with zero mixing.** R is not the enemy (consistent with U1=0).
- **Flip decomposition** (a flip whose realized class is NOT in the local `L*` neighbourhood = manufactured/
  context; else = boundary jitter): baseline splits **85.1% third-class (0.0411) / 14.9% boundary (0.0072)**.
- **Cross-validation:** the decomp *boundary* component **0.0072 ≈** the resolution-mixing contribution
  **0.0067**; the decomp *third-class* component **0.0411 ≈** the zero-mixing floor **0.0416**. The two
  lenses independently converge: the "third-class" mass is a **flat-paint CONTEXT mis-read**, not R mixing.

### ROOT CAUSE — SegNet argmax is CONTEXT/TEXTURE-dominated, not colour-dominated (the decision geometry)

The 216 constant-colour tile probe (constant colour is R-invariant) decodes:

```
Undrivable 195 / 216   Movable 10   MyCar 10   Road 1   Lane 0
```

**Road and Lane NEVER win argmax on colour alone** — they are context-only classes. A flat colour with no
scene texture reads as SegNet's global prior (Undrivable/sky). So **no abstract palette can make a flat
region read as Road/Lane/Movable**; a flat per-class-mean partition strips the texture the argmax depends
on → ~0.04 d_seg irreducible **regardless of palette**. This is why the abstract mixing-robust palette is
CATASTROPHIC (0.504: every class except Undrivable flips ~100% — everything collapses to Undrivable), and
why the scene-anchored optimiser cannot improve on the scene mean (the mixing penalty is ~flat on a map
where everything is Undrivable, so it returns the global-mean seed unchanged).

### Per-class (baseline, F contribution / third-class share)

Flip mass is the thin/small classes: **Movable 0.367** (11% third → mostly real boundary jitter of cars),
**MyCar 0.147** (79% third → flat ego-hood mis-read), **Lane 0.138** (12% third → real thin-lane jitter),
Road 0.017 (93% third → flat road patch reads as non-adjacent), Undrivable 0.003. The zero-mixing floor
(4b) holds the SAME shape (Movable 0.366, Lane 0.140, MyCar 0.121) → the floor is flat-paint mis-read, not R.

---

## VERDICT (verdict_scope FORMULATION — direct-partition→palette realization, ALL palettes; NO new kill)

**REFUTED.** F is **not** a fixable palette artifact. (1) R resolution mixing is only **14%**; (2) the
palette is **not a free lever** — every recolour is worse-or-equal, the abstract mixing-robust palette is
catastrophic (0.504), the best palette is the per-pair scene MEAN (floor **0.0416**); (3) root cause is
SegNet **context/texture-domination** (constant colour → Undrivable 195/216).

**The C1 "mirage" caveat HARDENS, not softens.** The prompt's reopen condition ("if F drops to boundary
noise, REOPEN direct-partition→palette realization") is **NOT met**: F does not drop (floor 0.0416,
palette-irreducible). So the caveat sharpens from **naive-palette-only → ALL-palette**: direct-partition →
palette realization is **CONFIRMED SHUT**. The trained-through-R self-orient witness (realized d_seg
**0.0048**, MEMORY L68/L17) is **8.7× below** this palette floor and remains the **only** viable
realization regime — precisely because it renders real **textured** RGB, not a flat palette.

**v8 IMPLICATION (design correction).** v8's byte-close realization does **not** "inherit the optimal
palette for free" — there is no free optimal palette. A flat per-class colour codec (store 5 colours/pair)
floors at ~0.042 d_seg (dominated). **v8 must carry per-pixel texture/chroma** (or a texture residual
sidecar concentrated where flat-paint mis-reads: Movable/MyCar) to keep SegNet's context intact.

**Reformulation queue** (this is a FORMULATION negative, not a family kill): textured/chroma per-pixel
realization (the trained-witness regime, already at 0.0048) · per-pair palette + learned per-pixel texture
residual · palette + high-freq residual only on the mis-read classes.

## Triality legs
- **DAG:** FEED-palette (this landing).
- **Equations:** `palette_realization_ceiling_context_dominated_v1` (registered; VERIFIED_VIA_EMPIRICAL_ANCHOR;
  the measured law realized-ceiling vs palette + zero-mix floor + context-domination + cross-validation).
- **DSL/verdicts:** `.omx/research/palette_artifact_probe_verdict_20260710.json` (FORMULATION-scoped negative,
  42 MeasurementRows `[through-R]`, reformulation queue). No DSL lever (this is a realization-geometry
  characterisation, not a trainer knob).

**Pointer 0.19110 UNMOVED (means).**
