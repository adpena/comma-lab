# SPEC — v8 PER-CLASS DECOMPOSITION (edge-centric carriers + scorer-rule reconciliation) — 2026-07-08

STORES CONSULTED: perclass_carriers_design_20260708 (c8fe12d8f, the full derivation) ·
probe_PA_paintfloor_perclass (bf1ee1fa8, n600) · road_anomaly_probe · DAG FEEDs perclass/PA/
v8risks/mergediff/missingforces · #210 ideal-dash + #139 hood + #276 chroma-DOF + #73 Dykstra +
#226 margin_conditional_residual + #180 Morse-Smale codec + lane_sdf/hood_static components ·
upstream modules.py facts (CLAUDE.md §scorer architectures). Author: outgoing Fable session
(operator-directed handoff). Task #359. Pointer **0.19110 UNMOVED** — means. GATING: v8 builds
fire only per §6; v7.5 launches FIRST and its results gate increment-1.

## 1. THE ARCHITECTURE (operator riff, derived in c8fe12d8f)

Partition = **tropical argmax of decoupled SDF-gauged scalar fields**: P(x)=argmax_c(φ_c(x)+b_c).
Separatrix = tie loci, DERIVED never represented. ∂φ_c/∂θ_{c'}=0 ⇒ the measured cross-class theft
(run-1: Lane 13.8×/Movable 4.6× stealing Road) is IMPOSSIBLE for its gradient mechanism by
construction. SDF gauge (1-Lipschitz eikonal fields) puts all classes in pixel units; b_c = ~0-byte
per-class bias (the Road↔Lane tie calibration = 41% of Road's oracle flips — highest-leverage
~0-byte lever, P-A). Area-Lagrange per field = exact dual of the measured mass-conservation
identity. Per-class τ_c/β_c/completion annealing = five independent small flows (operator: the
level-set/Morse-Smale engineered annealing). d_seg factorizes over pairwise tie-locus displacements
on the region-adjacency graph — CONFIRMED by P-A's destination matrix (Road = hub; every class
flips ONLY at its Road separatrix; ZERO interior flips at the oracle floor).

**BINDING REFINEMENT (v8-risk-1 cure): the decomposition is EDGE-CENTRIC, not class-naive.**
Separatrix information is SHARED between adjacent classes; one field per adjacency-graph EDGE
(the design's Road+Undriv single bulk-boundary field is the pattern), never two region fields
paying for the same curve twice.

## 2. CARRIER TABLE (measured/derived; the heterogeneous win)

MyCar4 = static mask ~0.1-0.5 KB (IoU 0.994 MEASURED #139) · Lane1 = analytic ground-frame band
~1-2 KB (d_seg 0.00087 MEASURED, v7's band) · Road0+Undriv2 = ONE bulk-boundary field, grid-bulk +
INR-annulus per #308, **20-50 KB DERIVED = the decisive unknown, now CONSERVATIVE-leaning-confirmed
(P-A: bulk interiors near-flawless through R — Road 0.17%/Undriv 0.03% within-class; spend the
bytes on SEPARATRIX/ANNULUS PRECISION, not interior texture)** · Movable3 = sparse islands +
homotopy + area-Lagrange ~2-6 KB. Net stack ≈27-57 KB vs 114 KB incumbent ⇒ −50..75% ≈ **0.049 S
rate headroom, CONDITIONAL on increment-1's byte measurement.** 5-regime KKT waterfill generalizes
the two-regime law; flip-share weights from run-1 per-class data.

## 3. RECONCILIATION = MERGE → DIFF → CORRECT vs the frozen scorer (operator + modules.py)

(1) MERGE: tropical argmax composite → paint. (2) DIFF: frozen SegNet argmax on R(composite) vs
intended partition — **frame1 ONLY** (SegNet reads x[:,-1]; frame0 is SegNet-FREE = pure pose
territory). (3) CORRECT, channel-routed by the modules.py asymmetry: SegNet = full-RGB last-frame
(chroma fully argmax-visible) vs PoseNet = YUV6×2 = 4 luma + 2 SUBSAMPLED chroma (pose is
luma-dominated) ⇒ correction Jacobian near-TRIANGULAR in (luma, chroma): **CHROMA-FIRST seg
repairs (SegNet-strong, PoseNet-quiet; measured basis #276), LUMA RESERVED for pose/warp
coherence.** (4) Iterate to fixed point; unpaintable residual = counted sidecar (#226
margin_conditional_residual finds its consumer; Lever-D b/flip economics measured). FORMALLY:
Dykstra alternating projections onto (argmax-cell ∩ pose-tube) in channel-split coordinates (#73
reborn). Pose seams (risk-5) largely retire: chroma-routed seams live where PoseNet barely looks;
guard LUMA seams — luma stays ONE coherent warp-structured field (⇒ the temporal screw-consistency
force is a NECESSARY companion, risk-4).

## 4. NAMED RISKS (FEED-v8risks — binding on increment-1 review; all live at the SEAMS)

1. EDGE-DUPLICATION → edge-centric decomposition (cure adopted in §1). 2. THEFT MIGRATES TO THE
COMPOSITE (end-to-end training through paint→R→SegNet re-couples classes in the score gradient) →
STAGED training: fields vs exact mask targets (`signed_distance_fields` gives argmax(sdf)==labels);
paint solved separately vs the frozen scorer. 3. MASK-OPTIMAL ≠ SCORE-OPTIMAL + oracle-paint gap
(P-A measured the UPPER bound with real-frame texture; generated paint UNMEASURED; video-derived
texture params are COUNTED under rule 118) → **probe P-C (pre-registered, $0): re-run P-A with
flat/procedural fill** — the go/no-go on "interiors near-free". 4. TIE-VARIANCE ADDS (independent
fields jitter independently) → temporal screw-consistency companion + the SDF gauge's tie
conditioning. 5. POSE SEAMS → chroma routing (§3) + MEASURE pose on composites, never assert.
6. APPARATUS ×5 + OPPORTUNITY COST → gating (§6).

## 5. PRE-REGISTERED PROBES + COUNCIL-FLAGGED EQUATIONS

**P-B** ($0): decoupled-theft falsification — transplant the birth-stack losses onto
parameter-independent toy fields; prediction part_frac ≈ 1.0× (instance-scope; the n600 verdict
rides increment-1). **P-C** ($0): flat/procedural-fill paint floor per class (risk-3's decisive
measurement). Two candidate canonical equations (tropical reconciliation law; per-class carrier
allocation) are COUNCIL-FLAGGED, NOT registered — anchors owed to P-B/P-C/increment-1. Register
only with measured anchors (triality discipline).

## 6. INCREMENT-1 (the smallest decisive build) + GATING

Split Road/Undriv into the dedicated edge-centric bulk-boundary field (Movable keeps the v7.5
homotopy + area-Lagrange machinery; Lane keeps the band; MyCar keeps the clamp). One build
de-shares 99% of measured flip mass AND measures the 20-50 KB unknown AND anchors the flagged
equations. **GATES: (a) v7.5 launches first — if the v7 line's measured trajectory reaches target,
v8 increments never fire (opportunity cost, risk-6); (b) P-C runs before the paint stage is
designed; (c) increment-1's design review must address all 6 named risks explicitly; (d) full seal
protocol (blind structural derivation, fix-all, verdict-scope, n600).** Authority: only byte-closed
`upstream/evaluate.py` rows judge any of it.

## 7. RELATION TO v7.5 (composes-then-subsumes)

v7.5's area-Lagrange is REUSED as the per-field constraint; its completion event becomes per-field;
its FP-precision machinery becomes unnecessary for decoupled classes while remaining correct for
any still-shared head. The P0 in-trunk forces (task #360) transfer: temporal screw-consistency and
tie-locus displacement are MORE natural per-field; margin-band satisficing applies to the paint
stage; R-phase alignment applies to the composite. Pose is NEUTRAL-not-free (L68 unchanged:
d_pose OPEN+UNMEASURED on the witness; frame0 luma freedom + the ξ carrier is the path).
