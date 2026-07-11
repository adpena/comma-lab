# Ground-plane d_seg: LABEL-NOISE vs RECONSTRUCTION — the reconciliation + verdict — 2026-07-10

**Dispatch question (strategy-deciding):** the just-landed Morse-Smale anisotropy map
(`partition_anisotropy_map_20260710.md`, commit 814fb1aac) frames Road↔Lane (0.503 share) +
Road↔Undrivable/HORIZON (0.179 share) = **~68% of the boundary mass** as a big lever via ONE unified
vanishing-point ground-frame **ξ RATE carrier**. But #141/#169 (June) concluded the horizon-band residual
is **LABEL-NOISE**, ΔS ceiling only **0.012–0.024** — which would CAP any lever there. Reconcile, and
decide: is the ground-plane d_seg RECONSTRUCTION-FIXABLE or LABEL-NOISE-CAPPED?

**Authority:** `[macOS advisory]`, $0, cached argmax + margin only (NO SegNet/PoseNet forward — scorer probe
pid 87834 + live run pid 88030 hold the memory budget). Pointer **0.19108282 UNMOVED** — this is a
strategy-deciding verdict, not a score-mover. Class order SELF-DETECTED (never luma-sorted); guard passed:
`0=Road(area 23.0%, row 239) 1=Lane(0.59%, row 226, IoU 0.263) 2=Undrivable(49.3%, row 95) 3=Movable(1.56%, row 194) 4=MyCar(25.6%, row 334)`.

---

## 1. The reconciliation — THREE objects were conflated under "horizon label-noise"

The contradiction dissolves once the three distinct things called "the horizon label-noise finding" are
separated. They are a FLOOR, a FORMULATION-negative, and a LOSS-lever ceiling — none of them bounds the
ξ-chart carrier.

| # | Object | What it actually is | Status (measured) |
|---|---|---|---|
| **A** | Absolute label-noise **FLOOR** (τ=0.137 proxy → d_seg 0.00123) | An estimated *irreducible floor* on the whole partition | **EXISTENCE-DISPROVEN** as a hard wall (DAG 25o/eu): frontier vehicle → d_seg **0.0003** globally (< the proxy floor); PR95 bc36 inflate → **6.02e-4**; 384+uint8 pipeline floor ~**1.6e-4**. Absolute d_floor ≲ **1.6–3e-4 ≪ need 0.00087.** |
| **B** | Horizon **"NO-GO ×3"** (DAG 25o, June) | (a) a **LINEAR store-the-flips SIDECAR** on the horizon band (full-rank in every linear basis → dead) + (b) a capacity-priority call ("bc20 generator should target the lane band, not horizon") | **FORMULATION-scoped negative** on the linear sidecar + a capacity-routing priority under an under-capacity vehicle. NOT a "boundary is irreducible" verdict. Its "flips not geometry-codeable" clause is **REFUTED** below. |
| **C** | **#169 horizon-weighted MARGIN term** (July, `horizon_weighted_margin_hinge_v1`) | A **0-byte train-time LOSS reweight** — a satisficing hinge `w·relu(m_target − m_wit)` on rows∈[96,288) × GT-margin∈[0.3,0.5) | The **0.012–0.024 is an ORACLE CEILING** (registry: *"the d_seg SIGN/MAGNITUDE is ASSUMED_AWAITING_VERIFICATION… the 0.012–0.024 is an oracle CEILING, not an achieved move"*; measurement_axis = `predicted`). It is the **max deliverable by that loss term**, and it is flagged **43.8% of remaining descent** in the costate duty-to-measure queue — a HIGH-value lever, the *opposite* of dismissed. |

The task's premise ("#169 concluded label-noise, ΔS ceiling only 0.012–0.024, which CAPS the lever") reads
**C's oracle CEILING as if it were A's irreducible FLOOR.** They are different objects. C is the *headroom
of one weak lever*; A is a *floor* and is disproven; B is a *dead sibling formulation*.

**The two levers are DIFFERENT (the operator's standing warning, honored):**
- **#169 = a LOSS reweight inside a FIXED representation.** It can only help if the decoder *already can*
  place the horizon boundary but isn't prioritizing it. Its ceiling 0.012–0.024 is the gain from
  re-prioritization *alone* — it adds **zero representational DOF**.
- **The anisotropy-map ξ-chart = a RATE carrier that CHANGES the representation** (adds an analytic
  `v_horizon(ξ)` line driven by the shared ego-screw). It attacks the reconstruction error directly.
- **⟹ #169's ceiling does NOT bound the ξ-chart carrier.** One formulation's ceiling ≠ the family's.

---

## 2. The label-noise-vs-reconstruction DECOMPOSITION per ground-plane edge (MEASURED)

**Discriminator (no scorer needed):** a boundary residual is RECONSTRUCTION-fixable if the boundary moves
as a **coherent** object (a witness with the right chart can track it) and LABEL-NOISE if it **flickers**
incoherently frame-to-frame (no witness can match a jittering target). Measured on the cached argmax by
decomposing the per-column horizon-interface temporal signal into **coherent MOTION** (median-3 smoothed)
vs **incoherent JITTER** (single-frame residual), + the annulus **margin** profile (low margin = closer to
the scorer's own decision boundary = label-noise frontier).

### 2a. Horizon (Road↔Undrivable) temporal coherence — n96 AND n600 agree

| scale | interface total std | coherent MOTION std | incoherent JITTER std | **coherent-fraction of variance** |
|---|---|---|---|---|
| n96  | 5.01 px | 4.93 px | 0.80 px | **0.975** |
| n600 | — | 7.55 px | 1.07 px | **0.980** |

**⟹ the horizon boundary is 98% coherent smooth motion, ~1 px incoherent jitter.** The boundary translates
with ego pitch (Road IoU 0.955, Undrivable IoU 0.995 — both classes barely change ⟹ their interface moves
as a coherent line). The **irreducible label-noise floor on the horizon is ~1 px of sub-line jitter** — tiny.
The 7.55 px coherent motion is exactly what a ξ-driven analytic line represents ⟹ **reconstruction-fixable.**

### 2b. Annulus margin — the horizon is NOT more label-noisy than the (settled-reconstruction) lane

| edge (canonical order) | pixel share | median margin | frac(margin < 0.137) = scorer-ambiguous |
|---|---|---|---|
| **Road–Undr (HORIZON)** | 0.42% | 0.449 | **0.157** (n96) / **0.177** (n600) |
| **Road–Lane (LANES)** | 0.99% | 0.479 | **0.152** |
| Road–MyCar (HOOD, static) | 0.51% | 0.458 | 0.150 |
| Undr–Mov (car tops) | 0.12% | 0.224 | **0.316** |
| Road–Mov (car bottoms) | 0.12% | 0.296 | **0.240** |

**⟹ the horizon's margin profile (med 0.449, ~16% low-margin) is statistically IDENTICAL to the lane's**
(0.479, 15%) and the hood's (0.458, 15%), and **cleaner** than the Movable edges (24–32% low-margin). The
~15–17% low-margin annulus is the **shared scorer-ambiguous frontier — the SAME on the lane, which is
already settled reconstruction-fixable** (8-dim nonlinear manifold; L80 "4.67% lane-edge residual GENUINE";
existence proof 0.0003 < need). If that shared 15% annulus does not make the *lane* label-noise-capped, it
cannot make the *horizon* label-noise-capped either. The "HORIZON = label-noise-like" claim (B) is refuted:
the horizon sits on the CLEAN end of the margin distribution, and its flips are 98% coherent motion.

### 2c. Line-fit richness (a caveat, still reconstruction not label-noise)

Per-frame straight-line fit of the horizon interface: **14.2 px RMS residual** (n96). So a *pure straight*
`v_horizon` under-captures ~part of the boundary (road crest / curvature) — matching the anisotropy map's
coverage 0.77 / residual 8.1 px. This is a **chart-RICHNESS** issue (needs the ground-plane homography / a
mildly-curved horizon model, not just a line), **still structured geometry, still reconstruction** — NOT
label-noise. The label-noise floor is the 1 px jitter, an order of magnitude below the 7.5–14 px coherent
structure the carrier can code.

---

## 3. Scope-ladder classification of #169's 0.012–0.024

| axis | classification | evidence |
|---|---|---|
| **Band scope** | HORIZON band ONLY (rows 96–288, GT margin 0.3–0.5) | registry `domain_of_validity`: `--seg-horizon-row-lo/-hi`, `--seg-horizon-margin-lo/-hi`. NOT the full ground-plane (excludes lanes, excludes Road↔Lane). |
| **Lever scope** | the MARGIN-REWEIGHTING LOSS TERM (0-byte, train-time, fixed representation) | registry: *"0-byte train-time loss reweighting on the reducible GT-margin band."* |
| **Value scope** | ORACLE CEILING, ASSUMED_AWAITING_VERIFICATION (predicted, not achieved) | registry note verbatim + `measurement_axis: [macOS-MLX, predicted]`. |
| **verdict_scope ladder** | **FORMULATION** (one lever, one band) | one loss-term formulation's ceiling. Does **NOT** bound the FAMILY of horizon levers, and specifically **does NOT bound the ξ-chart RATE carrier** (a representation-changing codec lever — a different formulation). Per the verdict-scope discipline: one failed/ceilinged formulation ≠ family dead. |

**#169's 0.012–0.024 is a LOWER bound on the horizon's reducible d_seg, not an upper bound** — because a
loss reweight cannot add the chart DOF the carrier adds. The carrier attacks the 7.55 px coherent-motion
error that the margin term can only indirectly nudge.

---

## 4. VERDICT

> **Ground-plane d_seg (Road/Lane/Undrivable, ~68% of boundary mass) is RECONSTRUCTION-FIXABLE, NOT
> label-noise-capped.** The unified vanishing-point ground-frame ξ chart is a REAL lever with headroom
> **bounded BELOW (not above) by #169's 0.012–0.024** — that number caps a *different, weaker* lever (a
> loss reweight inside a fixed representation), not the ξ-chart carrier.

Per-edge:
- **Horizon (Road↔Undrivable, 18% of mass): RECONSTRUCTION-FIXABLE.** 98% coherent motion / ~1 px jitter
  (n96 + n600 agree); margin profile identical to the lane; a clean directional line (d_H 4.45). Its
  label-noise floor is ~1 px — small, comparable to the lane, BELOW the need.
- **Lane (Road↔Lane, 50% of mass): RECONSTRUCTION-FIXABLE** (already settled: 8-dim nonlinear manifold,
  L80 GENUINE, existence proof 0.0003 < 0.00087). Confirmed here: same margin profile as the horizon.
- **Absolute label-noise floor:** existence-disproven; ≲ 1.6–3e-4 ≪ need 0.00087. The genuine irreducible
  component is the ~1 px per-edge jitter + the shared ~15% low-margin annulus — which caps the LAST
  fraction, not the bulk, and is identical on the lane we already treat as fixable.

**Strategic implication:** the ground-frame ξ chart earns its "68% of the cracks in one treatment" framing —
it attacks the coherent-motion reconstruction error on Road+Lane+Undrivable via ONE analytic
`v_horizon(ξ)` + lane-separatrix chart driven by the shared ego-screw. Fold the horizon into the lane's
ground-frame chart as an analytic line (the anisotropy map's §5.1 routing). This is THE ground-plane
d_seg lever; #169's ceiling does not cap it.

**Caveats (non-naive):**
1. The ξ-chart is a RATE carrier — its NET score win must be **byte-closed** (line-coord bytes vs d_seg
   reduction). This is `[macOS advisory]`; NO pointer move is claimed. The verdict says the lever is
   *reconstruction-attacking with headroom*, not that a win is banked.
2. Chart richness: a pure straight `v_horizon` leaves a 14 px crest/curvature residual → the carrier needs
   the ground-plane homography / mild curvature, not just a line. Still reconstruction.
3. The ~1 px jitter + ~15% low-margin annulus **IS** the genuine per-edge label-noise floor — small,
   comparable to the lane, below the need. It bounds the terminal fraction, not the strategy.
4. #169 (the margin term) remains independently worth firing (43.8% of remaining descent, oracle) — it is
   COMPLEMENTARY to the carrier (loss-side re-prioritization + representation-side chart), not redundant.

---

## Triality
- **DAG:** FEED-groundplane-recon (this memo).
- **DSL:** N/A — a reconciliation/verdict, no new lever wired (the ξ-chart carrier the sister lane-factorization
  agent owns is the DSL surface; this memo routes the horizon into it + clears the #169-caps-it misread).
- **equations:** consumes `horizon_weighted_margin_hinge_v1` (#169, re-scoped FORMULATION not floor) +
  the existence-proof floor anchors (25o/eu). Candidate law (MEASURED, not registered — needs a byte-closed
  ξ-chart carrier to confirm the win scales): `groundplane_boundary_is_coherent_motion_reconstruction_not_labelnoise`
  — horizon boundary temporal variance is 98% coherent (n96 0.975 / n600 0.980), jitter floor ~1 px,
  margin profile identical to the reconstruction-settled lane. Flagged, not asserted.

**Sisters (disjoint):** anisotropy-map memo (814fb1aac) + lane-factorization agent (afa9d885, owns the ξ
carrier). This memo reconciles the label-noise-vs-reconstruction tension + scope-classifies #169; it wires
nothing and touches no running agent's files. **Pointer 0.19108282 UNMOVED.**
