# v8 INCREMENT-1 DESIGN DRAFT — the edge-centric Road+Undriv bulk-boundary field — 2026-07-09

**Author:** Opus (operator "build out a draft of our v8" while #205 chugs). **Axis:** all
`[macOS advisory · NON-PROMOTABLE]`. **Pointer contest-CPU 0.19110 UNMOVED — this is a DRAFT (design,
$0, NO launch); nothing here moves the pointer.** #205 (pid 45378) UNTOUCHED. This is NOT a
re-derivation — `SPEC_v8_perclass_decomposition_20260708.md` + `perclass_carriers_design_20260708`
(c8fe12d8f) hold the math; this turns SPEC §6 (increment-1) into a **buildable, substrate-composed,
gate-(c)-addressed** design. Labels are strict: **MEASURED / DERIVED / DESIGNED**.

STORES CONSULTED: SPEC_v8 (§1–8) · perclass_carriers_design_20260708 (the derivation) · the in-tree
carrier substrate (`src/tac/boundary_math/{road_horizon_component, hood_static_component,
lane_sdf_component, analytic_lane_render_band, laguerre_logit_offset, lever_b_levelset_generator}.py`) ·
#139 hood · #276 chroma-DOF · #73 Dykstra · #226 margin_conditional_residual · #308 grid-bulk+INR-annulus ·
the P0 forces (task #360, all four BUILT + default-off) · #205 run-1 per-class telemetry (Road = hub,
70% of flips) · SPEC_v75 §8 (OPERATING CONTRACT, binding here in full).

## 0. Thesis (one line)
Increment-1 = de-share the **Road separatrix** into ONE edge-centric bulk-boundary field carrying
{Road, Undrivable}, which **de-shares ≈99% of measured flip mass in ONE build** (P-A: Road is the hub;
every class flips only at its Road separatrix), **measures the decisive 20–50 KB unknown**, and **anchors
the two council-flagged equations** — while Movable / Lane / MyCar keep their existing carriers unchanged.

## 1. Carrier → substrate map (COMPOSE, do not reinvent — proactive-recall verified in-tree)

| class | carrier | module (STATUS) | bytes | source |
|---|---|---|---|---|
| MyCar (4) | static mask clamp | `hood_static_component.py` — **REUSE as-is** | ~0.1–0.5 KB | IoU 0.994 **MEASURED** #139 |
| Lane (1) | analytic ground-frame band | `analytic_lane_render_band.py` + `lane_sdf_component.py` — **REUSE** | ~1–2 KB | d_seg 0.00087 **MEASURED** (openpilot band) |
| Movable (3) | sparse islands + homotopy + area-Lagrange | v7.5 machinery (`--ladder-island-homotopy` + `area_constraint`) — **REUSE v7.5** | ~2–6 KB | **DERIVED** |
| **Road (0) + Undriv (2)** | **ONE edge-centric bulk-boundary field** | **NEW** — composes `road_horizon_component.py` + `lever_b_levelset_generator.py` (SDF) + `laguerre_logit_offset.py` (b_c) | **20–50 KB DERIVED = THE unknown to measure** | conservative-leaning (P-A: interiors near-flawless through R) |
| tie bias b_c | Road↔Lane calibration offset | `laguerre_logit_offset.py` — **REUSE** | ~0 bytes | 41% of Road's oracle flips **MEASURED** (P-A) |

Net stack ≈ **27–57 KB vs 114 KB incumbent ⇒ −50..75% ≈ 0.049 S rate-headroom DERIVED**, CONDITIONAL on
increment-1's byte MEASUREMENT. The *only* new module is the Road+Undriv field; four of five carriers
already exist.

## 2. The one new build — the Road+Undriv edge-centric bulk-boundary field
- **ONE field, not two** (risk-1 cure, edge-centric): a single SDF-gauged scalar φ_bulk over the Road/Undriv
  bulk; their shared boundary is ONE tie locus. Two region fields would pay for the same curve twice.
  `P(x)=argmax_c(φ_c(x)+b_c)`; separatrix = tie loci, DERIVED never represented (∂φ_c/∂θ_{c'}=0 ⇒ the
  measured cross-class theft — Lane 13.8× / Movable 4.6× stealing Road — is impossible by construction).
- **Grid-bulk + INR-annulus (#308):** the interior is near-flawless through R (**MEASURED** P-A: Road 0.17% /
  Undriv 0.03% within-class) ⇒ a COARSE grid for the bulk interior, byte budget spent on the
  SEPARATRIX/ANNULUS precision. This is the whole "interiors near-free" bet — gate on P-C (§6).
- **1-Lipschitz eikonal gauge** (pixel units) — reuse the trunk eikonal machinery. **Area-Lagrange per
  field** = exact dual of the mass-conservation identity — reuse v7.5 `area_constraint`.
- **Interface (DESIGNED):** `phi_bulk(coords, params) -> (H,W)` SDF; composed with the other carriers' fields
  by tropical argmax → the composite partition.

## 3. Staged training (risk-2 cure — NEVER end-to-end through the composite)
- **Stage A — fields vs EXACT SDF targets** (`signed_distance_fields`, argmax(sdf)==labels): each carrier
  learns its own partition; NO cross-class gradient path exists.
- **Stage B — paint solved SEPARATELY vs the frozen scorer** (the §4 loop).
- **FORBIDDEN:** end-to-end paint→R→SegNet training. It re-opens the theft channel in the score gradient
  (§8(3)) and would need its own measured justification. The whole point of the decoupling is that it never
  runs.

## 4. Reconciliation = MERGE → DIFF → CORRECT (the paint stage; SPEC §3, made concrete)
1. **MERGE:** tropical argmax composite → paint (RGB frame1).
2. **DIFF:** frozen SegNet argmax on `R(composite)` vs the intended partition — **frame1 ONLY** (SegNet reads
   `x[:,-1]`; frame0 is SegNet-free = pure pose territory).
3. **CORRECT (channel-routed by the modules.py asymmetry):** SegNet = full-RGB last-frame (chroma fully
   argmax-visible) vs PoseNet = YUV6×2 (4 luma + 2 *subsampled* chroma, pose is luma-dominated) ⇒ the
   correction Jacobian is near-**triangular** in (luma, chroma) ⇒ **CHROMA-FIRST seg repairs**
   (SegNet-strong, PoseNet-quiet; basis **MEASURED** #276), **LUMA RESERVED** for pose/warp coherence.
4. **Iterate to fixed point** = Dykstra alternating projections onto (argmax-cell ∩ pose-tube) in
   channel-split coordinates (**#73 reborn**). Unpaintable residual → counted sidecar (**#226**
   margin_conditional_residual; Lever-D b/flip economics, MEASURED at admit time).
   *Pose seams (risk-5) largely retire:* chroma-routed seams live where PoseNet barely looks; guard the LUMA
   seams (luma stays ONE coherent warp-structured field — the temporal-screw force is the necessary
   companion, already BUILT as P0 FORCE 1).

## 5. The 6 named risks — EXPLICIT address (gate (c) requirement, seeded here)
1. **Edge-duplication** → edge-centric single Road+Undriv field (§2).
2. **Theft migrates to the composite** → staged training, no end-to-end (§3).
3. **Mask-optimal ≠ score-optimal + oracle-paint gap** → **P-C probe** (flat/procedural fill) runs BEFORE the
   paint stage is designed (§6). This is the decisive "interiors near-free" go/no-go; P-A measured only the
   UPPER bound with real-frame texture, and video-derived texture params are COUNTED under rule 118.
4. **Tie-variance adds** (independent fields jitter) → temporal-screw companion (P0 FORCE 1) + the SDF gauge's
   tie conditioning.
5. **Pose seams** → chroma routing (§4) + MEASURE pose on composites, never assert.
6. **Apparatus ×5 + opportunity cost** → gating: **v7.5-first** (§8).

## 6. Pre-registered probes ($0, n600) — fire with OPERATOR-GO + memory headroom, NOT concurrent with #205
- **P-B:** decoupled-theft falsification — transplant the birth-stack losses onto parameter-independent toy
  fields; **prediction** part_frac ≈ 1.0× (instance-scope; the n600 verdict rides increment-1).
- **P-C:** flat/procedural-fill paint floor per class — risk-3's decisive measurement (gate b: runs *before*
  the paint stage is designed).
- **MEMORY GATE (binding):** both are heavy **n600 through-R SegNet forwards** — the exact full-P batched
  scorer path that OOM'd #205 at +66 GiB. With #205 peaking ~71 GiB / ~56 GiB free, they are **memory-unsafe
  to run concurrently**. Route through the governed launcher with the memory-preflight, and fire only when
  #205 yields the box OR headroom is measured-safe. **Not now.**

## 7. Triality + vehicle-OS discipline
- **DSL:** the Road+Undriv bulk-field lands as a `Lever` factory (the other four carriers reuse existing
  levers); `lever_registry.completeness()` must show 0 unmapped for its new flags (never-invent-flags).
- **equations:** the two council-flagged laws (tropical reconciliation; per-class carrier allocation) STAY
  FORMALIZATION_PENDING until P-B / P-C / increment-1 anchors land — registering a designed-but-unmeasured
  law is the exact anti-pattern the discipline forbids.
- **DAG:** append `FEED-v8-increment1-design`.
- **vehicle-OS:** EACH carrier independently **byte-closed + resumable + bit-exact-at-decode BEFORE
  composition** (the manifest-per-carrier discipline; §8(5)). Composition only after each passes its own
  reference contract.

## 8. Gates (binding) + what this draft is NOT
- **v7.5-FIRST:** if #205's v7 line reaches target on its measured trajectory, v8 increments **never fire**
  (opportunity cost, risk-6). This draft parks behind that verdict.
- This is **PRE-BUILD DESIGN**. Increment-1 BUILD additionally needs: P-C run (gate b), the full 6-risk design
  review (this doc is its seed, not its completion), and the full seal protocol (blind structural derivation,
  fix-all, verdict-scope, n600). Authority: only a byte-closed `upstream/evaluate.py` row judges any of it.

## Pointer
**0.19110 UNMOVED.** Draft = MEANS. The END is a byte-closed exact row from increment-1's measured A/B —
gated behind v7.5, and behind P-C's "interiors near-free" go/no-go.
