# ddm_js7 composed EC1 overlay — EXACT contest-CUDA row REFUTES the n32 projection (2026-08-12)

## The row [contest-CUDA T4, n600, locked env]
- Archive: 186,575 B, sha256 465d3c584c54c55d161c6411cfbd34c4bf3c4c546c189ddcbfbb1a2a941e9af4
- avg_segnet_dist 0.00029675 · avg_posenet_dist 9.06e-06 · gpu Tesla T4 (gpu_t4_match=True), validation_errors=[]
- S recomputed from components = **0.16342603740620176**
- Call fc-01KZVEFF3QRXHDMJYS4DZJ6JK4, modal_elapsed 409.1 s (~$0.10, #381 envelope)
- Row JSON: /Volumes/APDataStore/pact/ddm_js7_20260812/main_n600_and_exact/MODAL_CUDA_ROW.json

## Verdict: +0.00147 WORSE than cp135 (0.16195513827824176) vs projected −0.00058
Per-axis decomposition against cp135's components (seg 0.00029643 · pose 6.88e-6 · 186,252 B):
| Axis | Projected [n32 stratified advisory] | Realized [exact n600] | ΔS |
|---|---|---|---|
| seg | −1,133 robust flips | ≈ +63 net error pixels (d_seg +3.2e-7) | +0.000032 |
| pose | +1.38e-6 d_pose | +2.18e-6 d_pose | **+0.001224** |
| rate | +323 B | +323 B (exact) | +0.000215 |

## The two instrument findings (the durable value of this $0.10)
1. **The pose STACK gate was mis-calibrated by ~10× — derivable arithmetic we owed pre-dispatch.**
   At base d_pose 6.88e-6, the marginal is d(√(10p))/dp = 5/√(10p) ≈ **603 S per unit d_pose**.
   The js6/js7 per-proposal gate 2e-6 therefore permits +0.0012 S of pose damage — MORE than the
   entire projected seg gain (−0.0008 S). Correct stack budget: Δd_pose ≤ seg_gain_S/603 ≈ **1.3e-7**
   for this candidate (the realized +2.18e-6 is 17× over). Per-proposal gates were individually fine
   (admitted events ~1.7e-7); the STACK accumulated. LAW: acceptance gates must be priced in S units
   at the BASE operating point, and budgeted at STACK level, never per-proposal.
2. **n32-stratified robust-flip projection carries a SIGN ERROR at stack level** (−1,133 projected vs
   ≈ +63 realized): per-event receiver effects interact at n600 in ways the 32-pair stratified panel
   cannot see. Instance of m96/m94 (subset bias; a negative measures the instrument). The n32
   acceptance table remains valid for PER-EVENT triage; it is NOT a composition authority.

## What survives
The full chain ran end-to-end for the FIRST time on the seg leg: ec1 event alphabet → js6/js7
realized acceptance → joint compose → byte-close (+323 B, decode byte-identical, determinism repeat
exact) → Modal T4 exact row. The mechanism is real; this 44-event stack is dead. Rerun requires the
corrected stack pose budget (1.3e-7-class) + n600-advisory-calibrated flip projection.

Frontier UNMOVED: effective floor cp135 0.16195513827824176 @ 186,252 B [contest-CUDA T4 n600];
own-vehicle lc2 0.16959899569230852 @ 187,226 B.
