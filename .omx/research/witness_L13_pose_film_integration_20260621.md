# Witness L13 + Wyner-Ziv pose-FiLM integration — gap #3 (the pose-wall closure proof)

**Subagent:** `witness_L13_pose_film_integration_gap3`. **Date:** 2026-06-21.
**Authority:** `[contest-CPU advisory]` diagnostic, NON-PROMOTABLE. `score_claim=false`,
`promotable=false`, `ready_for_exact_eval_dispatch=false`. `$0` spend, no GPU, no paid dispatch,
**NO MPS** (the live MPS basin daemon pid 49375/49377 was never touched — read-only on all
checkpoints/artifacts). Frontier pointer **UNMOVED** (`.omx/state/canonical_frontier_pointer.json` →
contest-CPU `0.191099824`, archive `177169 B`, lane `pr110_payload_entropy_recode`).

> **THE VERDICT IN ONE LINE: the composition is ARCHITECTURALLY BLOCKED. L13 + pose-FiLM cannot be
> composed end-to-end because the pose-FiLM mechanism modulates a *neural HNeRV decoder feature map*
> and the L13 witness has *no neural feature map* — its frame1 is a piecewise-constant palette
> rasterization. This is reported as the honest blocker (the prompt's explicit fail-closed path),
> NOT a fabricated d_pose number. The pose wall on the L13 base is closeable, but by the
> `AmortizedLumaCarrier` coordinate-INR (`#57` build, already coded, NOT byte-closed into L13) — NOT
> by the pose-FiLM (which belongs to the HNeRV vehicle).**

---

## 1. What was asked (gap #3) and what was found

The probe asked: does composing the byte-closed L13 score-native witness (72,217 B, d_pose=12.66
palette-blind) with the Wyner-Ziv pose-FiLM side-info (~1.5 KB; the pose-FiLM disambiguator returned
GO) **CLOSE the pose wall** (d_pose 12.66 → ~frontier), decoupling the witness's two walls (pose
CLOSED by FiLM here, seg OPEN for gap #1)?

**Finding: the two artifacts are architecturally incompatible.** They are pose mechanisms for two
DIFFERENT vehicles:

| | L13 score-native witness | pose-FiLM (disambiguator's GO) |
|---|---|---|
| vehicle | `score_native_carrier.v1` (palette rasterizer) | `base_ch=20 HNeRV basin` (neural decoder) |
| frame1 render | `palette.colors[argmax_cam]` — piecewise-constant lookup | `sigmoid(rgb_1(neural feature map x)) * 255` |
| feature to modulate | **NONE** (flat per-class colors) | `x = stem(z).view(B, channels[0], base_h, base_w)` |
| pose mechanism | `AmortizedLumaCarrier` coord-INR (frame0/luma) — `#57` build | `PoseFiLMHNeRVWrapper` FiLM on the stem feature |
| render resolution | camera 874×1164 | SEG 384×512 |
| code | `src/tac/boundary_math/legal_frame_bridge.py` `rasterize_palette_frame` | `src/tac/torch_vehicle/pose_film.py` `PoseFiLMHNeRVWrapper` |

The composition premise ("inject the pose-FiLM into the L13 base") has no attachment point. This was
established by reading the two render paths end-to-end (no scorer forward needed — the blocker is
structural, in the code).

## 2. THE STRUCTURAL BLOCKER (the decisive code evidence)

### 2a. pose-FiLM REQUIRES a neural HNeRV decoder feature map

`PoseFiLMHNeRVWrapper._forward_with_film` (`src/tac/torch_vehicle/pose_film.py:197-223`) replicates
the vendored `HNeRVDecoder.forward` and injects FiLM ON THE STEM CHANNEL DIM:

```python
x = d.stem(z).view(B, d.channels[0], d.base_h, d.base_w)   # neural feature map
gamma, beta = self.pose_film(pose6)                         # (B, stem_channels)
x = gamma[:, :, None, None] * x + beta[:, :, None, None]    # <-- FiLM modulates the feature map
x = torch.sin(x)
... vendored block cascade (blocks/skips/ps/refine) ...
f1 = sigmoid(d.rgb_1(x)) * 255                              # pose shifts BECAUSE x propagated
```

The FiLM works because γ/β modulate a **rich continuous feature map** `x` that propagates through the
trained `sin`-cascade + `rgb_1` head; the rendered frame1's PoseNet readout shifts toward the stored
pose (Wyner-Ziv: hand the decoder the GT pose as side-info, the decoder paints the motion). The smoke
that returned GO (`experiments/smoke_pose_film_cpu_disambiguator.py:335`) instantiates exactly this:
`v.HNeRVDecoder(...)` loaded from the `basin_bc20_20260612T121523Z` forkpoint.

### 2b. L13 frame1 has NO feature map — it is a palette lookup

The L13 byte-closed inflate (`experiments/results/score_native_candidate_20260610/inflate.py:62-70`):

```python
def decode_frame1(member, pi):
    cfg, deq, pal, _ = parse(member)
    logits = _forward(deq, cfg, _coords(SEG_H, SEG_W), deq["mod"][pi])  # seg INR → LOGITS
    ag = logits.argmax(-1).reshape(SEG_H, SEG_W)                        # → ARGMAX label map
    am_cam = ag[ys][:, xs]                                              # nearest-upsample to camera
    return np.clip(np.round(pal[am_cam]), 0, 255).astype(np.uint8)     # palette lookup → flat colors
```

The render is `pal[am_cam]` — every pixel is its class's palette color. There is **no `z`, no stem,
no continuous feature map, no `sin`-cascade, no `rgb_1` head**. FiLM has nothing to modulate. And
even if you forced a γ/β onto the per-class colors, the output stays piecewise-constant (you'd
rescale/shift flat colors) — a flat-color region has **zero luma-gradient texture**, so PoseNet
(which reads 2-frame YUV6 motion) sees no motion structure regardless. The L13 inflate's own
docstring states it: *"this v1 emits palette frame1 only; pose collapses — research carrier."*

### 2c. The L13 candidate carries pose bytes it never renders from

The L13 archive parses a `pose` section (6,650 B fp16+brotli, `inflate.py:51-52`) but `decode_frame1`
ignores it (`_,` at line 63). The stored pose is dead weight in the current candidate — the §0 honest
miss the L13 memo already recorded. So even the L13 candidate's OWN pose intent is unrealized; the
pose-FiLM is not the wiring it's missing.

## 3. The two pose mechanisms are NOT interchangeable (why the disambiguator GO does not transfer)

The pose-FiLM disambiguator's GO is **valid for the HNeRV vehicle** and does not transfer to L13:

- **pose-FiLM (Wyner-Ziv side-info on a neural decoder):** stores 6 GT pose scalars/pair + a tiny
  FiLM MLP; the GT pose modulates the decoder's feature map so the *already-textured* neural render's
  PoseNet readout snaps toward the stored pose. It NEEDS the decoder to produce luma texture; it only
  *redirects* that texture's motion. On a palette frame there is no texture to redirect.
- **`AmortizedLumaCarrier` (the L13-correct pose carrier, `#57`):** a SCORER-FREE coordinate-INR
  (`src/tac/boundary_math/amortized_luma_carrier.py`) that GENERATES a frame0/luma appearance the
  PoseNet needs, amortized like the seg generator (NOT stored per-pair). This is the mechanism the
  L13 memo §4 reactivation-path #1 names and the `#56`-blocker doc closes. It is a different design
  (generate luma vs modulate-a-neural-feature) and it is **not byte-closed into the L13 candidate**.

So the L13 pose wall IS closeable — but by `AmortizedLumaCarrier`, not pose-FiLM. The probe's
composition target was the wrong mechanism for the L13 vehicle.

## 4. Deliverables (against the prompt's 5 asks)

### 4.1 d_pose BEFORE vs AFTER on the subset
- **BEFORE (L13 palette, durable artifact):** d_pose = **12.658** (8-pair,
  `score_native_first_candidate.json` `advisory_S.mean_d_pose`); GT-frame1-floor d_pose = **0.0**
  (proving the palette IS the entire pose problem).
- **AFTER (L13 + pose-FiLM):** **NOT MEASURABLE — composition architecturally blocked** (§2). No
  d_pose AFTER number exists because the pose-FiLM cannot attach to the palette rasterizer. Per the
  prompt's fail-closed instruction, this is reported as the blocker, not fabricated.

### 4.2 Integrated advisory S decomposition (from the durable L13 artifact, 8-pair)
The L13 candidate's OWN advisory S decomposition (no FiLM applied, because it cannot be):

| term | value | formula |
|---|---:|---|
| seg_term | **2.281** | `100 · d_seg = 100 · 0.022811` |
| pose_term | **11.252** | `sqrt(10 · d_pose) = sqrt(10 · 12.658)` |
| rate_term | **0.0481** | `25 · 72217 / 37_545_489` |
| **total S** | **13.58** | (matches the L13 memo §3) |

The pose_term dominates (11.25 of 13.58). The seg_term (2.281) is the next-binding wall once pose is
carried — exactly gap #1's territory. This decomposition is the L13 base AS-IS; the integration that
would lower pose_term is the `AmortizedLumaCarrier` build, not this probe.

### 4.3 Verdict: pose wall CLOSED or NOT-CLOSED on the L13 base
**NEITHER — the composition is BLOCKED (a third outcome the prompt anticipated).** The pose wall on
the L13 base is **not closed by pose-FiLM** (incompatible mechanism) and **not yet open-tested by the
correct mechanism** (`AmortizedLumaCarrier` is coded but not byte-closed into L13). The honest
finding: *the disambiguator's pose-FiLM GO is an HNeRV-vehicle result; the L13 witness needs its own
pose-carrier (`#57` amortized luma INR) to close its pose wall.*

### 4.4 Byte accounting
- L13 witness (current byte-closed): **72,217 B** → rate term **0.0481** (or 76,486 B / 0.0509 with
  the mdl_contour solver correction). −59% vs the 177,169 B RGB frontier (the rate class shift is
  real and byte-closed, lossless-parity-proven over 8 pairs, archive sha `1e851e69…`).
- pose-FiLM side-info (~1.5 KB) **does NOT apply** — it is the HNeRV vehicle's pose section. The
  L13-correct pose section is the `AmortizedLumaCarrier` weights+mod blob (a coordinate-INR, sized
  like the 65 KB seg generator's scale per the memo's amortization argument), **not measured here**
  (out of this probe's scope; it is the next build).

### 4.5 This memo — committed via serializer. Cross-links below.

## 5. What gap #3 actually proved (the reframe, stated plainly)

Gap #3's job was to prove the witness's pose wall is closeable so the two walls decouple. The probe
instead proved a **prerequisite finding**: *the pose-FiLM (the mechanism the disambiguator validated)
is bound to the HNeRV decoder and cannot be the L13 witness's pose-carrier.* The pose-wall-closure
proof for the L13 witness must run the **`AmortizedLumaCarrier`** (already coded, NO-FAKE-tested) and
byte-close it into the L13 candidate, then measure d_pose AFTER. That is the corrected gap-#3 build.

This is NOT a kill of either artifact (Catalog #307 IMPLEMENTATION-LEVEL, not paradigm):
- The pose-FiLM is INTACT for the HNeRV vehicle (its disambiguator GO stands for that base).
- The L13 witness is INTACT (rate class-shift real; seg-repairability proven; pose-carrier is the
  open `#57` piece, named by its own memo).
- The composition `L13 + pose-FiLM` is the FALSIFIED unit (wrong mechanism for the vehicle).

## 6. Honest expectation check (the prompt's pre-registration)
The prompt pre-registered that the integrated S would be ~0.73 (worse than the 0.19 frontier, fine —
the point was pose-closure not a winning S). The actual L13-base S is **13.58** (pose-dominated, no
FiLM applied), and the pose-closure could not be demonstrated because the mechanism is incompatible.
So the prompt's "~0.73" projection was itself predicated on the FiLM closing pose — which it cannot do
on this base. The corrected expectation: WITH `AmortizedLumaCarrier` carrying pose (the L13 memo's
factor-8-amortized projection: d_pose ~0.03 → pose_term ~0.55), the L13 base would sit near
S ≈ seg_term(2.28) + 0.55 + rate(~0.06) ≈ **2.9** — still far above frontier because the **seg_term
(2.28) is the true binding wall** once pose is carried (gap #1). That is the honest decoupling: pose
is cheap-CARRIABLE (by the right INR), seg is the expensive remaining wall.

## 7. Wire-in (Catalog #125, 6-hook)
1. **sensitivity-map** — N/A (this is an architecture-compatibility verdict, not a per-byte
   sensitivity producer).
2. **Pareto** — ACTIVE: records that the L13 vertex sits at {seg_term 2.28, pose_term 11.25 (palette),
   rate 0.048}; the Pareto-feasible pose move is `AmortizedLumaCarrier`, NOT pose-FiLM (the latter is
   off the L13 manifold entirely).
3. **bit-allocator** — DESIGN: the L13 pose section to allocate is the amortized-luma INR blob, not a
   FiLM side-info section.
4. **cathedral-autopilot** — gate NOT met (no archive-deployable composition; advisory + blocked).
5. **continual-learning** — ACTIVE, reseeds the planner with THE finding: *pose mechanisms are
   vehicle-bound* — pose-FiLM ↔ HNeRV (neural feature modulation); amortized-luma-INR ↔ score-native
   palette witness (scorer-free generation). A disambiguator GO on vehicle A does not transfer to
   vehicle B; the pose-carrier must match the witness's render substrate. Corrected gap-#3 build =
   byte-close `AmortizedLumaCarrier` into the L13 candidate.
6. **probe-disambiguator** — RESOLVED (with a third outcome): "does L13 + pose-FiLM close the pose
   wall?" → **BLOCKED (incompatible mechanism)**, not YES/NO. Next probe: "does `AmortizedLumaCarrier`
   close the L13 pose wall?" (the correct gap-#3 unit).

**Mission contribution:** `frontier_breaking_enabler` (a $0 architecture-compatibility verdict that
RE-ROUTES gap-#3's pose-closure build from the wrong mechanism to the right one, preventing a wasted
HNeRV-vehicle FiLM dispatch on the score-native witness). The END is a lower exact score; this is a
MEANS, stated plainly. **Frontier UNMOVED `0.191099824`.** No score asserted. No GPU/MPS used.

## 8. Cross-references
- `score_native_first_candidate_20260610T112433Z.md` — the L13 witness formulation + the §0 honest
  pose-collapse miss + §4 reactivation-path #1 (`AmortizedLumaCarrier`) this probe re-routes to.
- `pose_film_cpu_disambiguator_20260612.md` — the pose-FiLM GO (HNeRV `base_ch=20` vehicle; the
  result that does NOT transfer to L13).
- `score_native_pose_carrier_DESIGN_20260610T123000Z.md` + `score_native_pose_carrier_20260610T125000Z.md`
  — the score-native pose-carrier design line (the L13-correct lineage).
- Code: `src/tac/boundary_math/legal_frame_bridge.py` (`rasterize_palette_frame`, the L13 frame1 —
  no feature map) · `src/tac/boundary_math/amortized_luma_carrier.py` (`AmortizedLumaCarrier`, the
  L13-correct pose-carrier, coded + NO-FAKE-tested, NOT byte-closed into L13) ·
  `src/tac/torch_vehicle/pose_film.py` (`PoseFiLMHNeRVWrapper`, the HNeRV-bound FiLM) ·
  `experiments/smoke_pose_film_cpu_disambiguator.py:335` (the GO smoke wraps `v.HNeRVDecoder`).
- Durable artifact: `experiments/results/score_native_candidate_20260610/`
  (archive.zip 72,217 B sha `1e851e69b894…`, `score_native_first_candidate.json` with the BEFORE
  d_pose 12.658 / d_seg 0.02281 / GT-frame1-floor 0.0).
