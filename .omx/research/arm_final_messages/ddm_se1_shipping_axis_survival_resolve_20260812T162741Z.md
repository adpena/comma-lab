F1 fired at FORMULATION scope; F2 did not. No admissible shipping-axis state, n600 extension, or exact recipe was earned.

The best numerical row was undrivable-only at magnitude 0.125: `ΔS=-0.003710012`, pose delta `-3.24e-05`, but 0/32 beneficial flips cleared `δ=0.0803604126`. The apparent gain is therefore below the measured CPU/CUDA disagreement margin and is rejected.

| Decoded class | 1.0 | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---:|---:|---:|---:|---:|
| Road | +0.241290 / 13.02% | +0.117931 / 2.04% | +0.051374 / 0% | +0.018766 / 0% | +0.007997 / 0% |
| Lane | +0.152456 / 7.03% | +0.072360 / 0% | +0.031476 / 0% | +0.012625 / 0% | +0.004924 / 0% |
| Undrivable | +0.096987 / 7.27% | +0.028124 / 0% | +0.001818 / 0% | **−0.003710 / 0%** | −0.000936 / 0% |
| Movable | +0.001048 / 6.67% | +0.000017 / 0% | −0.000231 / 0% | −0.000714 / 0% | −0.000558 / 0% |
| MyCar | +0.040979 / 0% | +0.015134 / 0% | +0.006237 / 0% | +0.000914 / 0% | +0.002693 / 0% |
| All classes | +0.325766 / 21.61% | +0.167762 / 5.46% | +0.081848 / 0% | +0.032640 / 0% | +0.018413 / 0% |

Each cell is `section-additive ΔS / δ-margin mass` on `[macOS-CPU advisory, stratified-random n32]`.

The best composed row used scales `[0.0078125, 0.0078125, 0.015625, 0.015625, 0.015625]` and measured `ΔS=-0.000142214`, pose delta `-5.45e-06`, but again 0/7 beneficial flips cleared δ. It was rejected.

All 35 parse-backed packets were 768 B and have byte-identical repeats. Every row retained its correction, camera bytes, logits, argmax, pose errors, and 32 pair receipts—8.9 GiB total. The projector and C1 teacher contribute zero shipped bytes.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_se1_shipping_axis_survival_resolve_20260812.md)
- [Final machine receipt](/Volumes/APDataStore/pact/ddm_se1_20260812/FINAL_RESULT.json)
- [Runner](/Users/adpena/Projects/pact/experiments/ddm_se1_shipping_axis_survival_resolve.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_se1_shipping_axis_survival_resolve.py)

Verification: 36 tests passed, Ruff clean/formatted, two recorded review passes per Python file. The implementation landed in `8030bf30a9`; scoped preflight evidence landed in `61c9961414`. The broader dirty-worktree preflight remains red on eight unrelated gates, with zero violations referencing either ddm_se1 file.

No n600 scorer, candidate archive, Modal dispatch, or exact evaluation was run. The effective frontier remains CP135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4 n600]`; the own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4 adjudicated n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN training-leg router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_ec1_20260812/realized_acceptance`; fire trigger: MAIN confirms the completed EC1 200-proposal bank is content-distinct, owns the sole scorer lane, and runs parse-backed realized acceptance through the same camera/uint8, δ-margin, joint-score, and pose-endpoint gates.

## LIVE-HYPOTHESES

- EC1 event coordinates remain plausible because they can address sparse C1 events directly, whereas the rejected packet moved every pixel of a decoded class together.
- Event-local deterministic rasterization may preserve robust Seg margin without the global photometric pose spill seen at larger class-wide magnitudes.
- Joint train-time pose shaping may close the post-uint8 nonlinear leakage that a fixed first-order null projector could not eliminate from the bare packet.

## DEAD-ENDS

- Literal C1-plane substitution remains closed: HC1 already measured `rho≈−1.223` and `S=0.4044688071` on contest CUDA.
- Another all-class JS5 amplitude replay is closed: robust Seg movement appears only with prohibitive pose damage.
- Negative local `ΔS` below δ is not survival: the best row had 0/32 robust beneficial flips.
- Five decoded-class scalar magnitudes on this hidden-4 conditioner are closed at FORMULATION scope: no single-class, all-class, or composed state passed all admission gates.
- Shipping the JS4 projector is closed: it is large scorer-derived training state, while the legal receiver packet must remain bare.