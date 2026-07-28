# DAG FEED — ddm_ar1 archetype codec priced spec (2026-07-28)

**Node:** ddm_ar1 (codec-artist composition). **Base:** main@d41cba1b10. **Axis:** [macOS-CPU advisory].
**Pointer:** 0.19108 UNMOVED. score_claim=false.

## EDGE IN
fc1 (copy-base compose, frame_0 2.7 MB wall, scenario C 185 KB → 0.266 @ banked pose) + pt1 (synthetic
paint SegNet floor 0.0086 @ 30 B, PoseNet deferred) + dm4 (realized RGB 3.24 MB / 2065× exchange) +
PR130 intake (191,052 B @ 0.172141; learned int5 pose field 23 KB) + pose_plane_proximity_law +
6 operator corrections (pose-not-floored, e_p probe, terminal staging, hybrid, bit-depth, PDW2).

## MEASURED (this arm, $0)
- Banked PoseNet target matrix t_p (600×6): **rank-1 (SVD energy 0.998 in one comp; 3 comps → 0.99969),
  temporally smooth (lag-1 autocorr 0.55–0.98)**; int5-innovation proxy ~2.06 KB. ⇒ pose field AR-codes
  ≪ PR130's 23 KB; **pose is feasibility-bounded, not the binding constraint.**

## COMPOSED (arithmetic over MEASURED stream bytes)
- **Archetype byte FLOOR (fc1-C streams + corrected PR130 pose) = 0.154** — SUB the 0.172 bar (vs fc1's
  0.266 @ banked pose). With cheap pose field + contour support ≤157 KB → **sub-0.15**. Budget 212 KB is
  NOT the wall.
- **Realized regime:** described base 0.024 (2.42, dead) / classical correction 0.43 (rate-walled) /
  learned carrier ~0.14–0.15 (UNBUILT, PR130-class).

## BINDING UNKNOWN (ranked)
1. **SEG REALIZATION at target rate** — the crux (4th re-proof). Learned OUR-line carrier is the only
   measured escape (PR130 existence). Crux ≡ "communicate range(A) in uint8"; unifies with the bit-depth/
   gauge lever (store range(A), free ker(A) 80.67%, #580/#553). Probe: fit tiny grid+contour scene carrier,
   MEASURE realized d_seg + bytes. Falsifier: no ≤180 KB carrier realizes d_seg ≤ 1e-3 ⇒ byte-floor-only.
2. **Pose survival (e_p probe)** — PRE-REGISTERED, not fired. PoseNet(painted) − t_p rank/affine-R²/ξ-smooth.
   Falsifier: full-rank + ξ-rough (R² < 0.5) ⇒ from-scratch field (bounded 38.4 B/pair). Cheaper than #1;
   fire first with one n600 scorer slot (coordinate da1).
3. **Support bit-depth re-open** — int8-saturation diagnostic on smooth support strata; $0-measurable.

## EDGE OUT (routed to MAIN / build arm)
- BUILD #1: tiny OUR-line range(A)-only gauge-fixed coarse-quantized scene carrier + terminal pose solve.
- MEASURE #2: the e_p paint-survival probe (design in tool; one scorer slot).
- The archetype does NOT escape the realization crux; it improves the byte floor to sub-bar and proves the
  pose leg is cheap. The remaining physics: realize range(A) content in uint8 at a fraction of dm4's cost.

## ARTIFACTS
- `.omx/research/ddm_ar1_archetype_codec_priced_spec_20260728.md` (memo)
- `experiments/ddm_ar1_pose_target_structure_probe.py` (fired $0 probe + pre-registered e_p)
- `/Volumes/VertigoDataTier/pact/ddm_ar1_20260728/pose_target_structure_receipt.json`
