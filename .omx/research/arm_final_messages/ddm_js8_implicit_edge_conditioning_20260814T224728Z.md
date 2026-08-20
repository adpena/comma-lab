JS8 is complete and refused at INSTANCE / local-admission scope. The decoder-derived Road-hub gate recovered only 7 Seg flips while adding 1,749 bytes.

| n600 macOS-CPU advisory | MC36 base | JS8 scale 0.125 | Delta |
|---|---:|---:|---:|
| Seg flips | 50,388 | 50,381 | -7 |
| d_seg | 0.0004271444 | 0.0004270850 | -5.93e-8 |
| d_pose | 0.0001474662 | 0.0003114726 | +0.0001640064 |
| Archive bytes | 186,269 | 188,018 | +1,749 |
| Recomputed S | 0.2051446455 | 0.2237117046 | +0.0185670591 |

Seg improvement contributes `-0.000005934`, while rate adds `+0.001164587`: Seg plus rate is already `+0.001158653`. Paying the rate required 1,373.803 recovered flips. QS5 compensation was therefore folded: its proven role is restoring pose toward the source object, and it cannot alter frame-1 Seg or existing rate.

EC2’s terminal mechanism is clear: it fixed 12,075 errors but introduced 52,854, including 42,184 GT-Road errors. Which training stage caused that collateral is not measurable from retained custody because only the terminal stage-30 endpoint was scored.

Key artifacts:

- [Handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_js8_implicit_edge_conditioning_20260814.md)
- [Full n600 result](/Volumes/APDataStore/pact/pr135_joint_solve_20260810/edge_conditioned/js8/full_n600_v1/FULL_RESULT.json)
- [Typed adjudication](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/js8/ADJUDICATION.json)
- [Provenance receipt](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/js8/PROVENANCE.json)
- Commit: `c6cd5d6ecf4d6a0033e2851061af8fd3c1e0f457`

The real receiver decoded n600 in 806.059 seconds, and every archive, raw frame, scorer input, logit, argmax, pose vector, and interrupted partial was retained. Ruff, formatting, compilation, retention audit, two review passes, and policy checks passed; focused tests report `3 passed`.

Not measured: QS5-compensated JS8, JS8 on contest-CPU/CUDA, or `upstream/evaluate.py`. The CPU base field differs materially from retained T4 custody, so this refusal is not promoted into a T4 delta or family verdict.

No pointer moved; this session did not lower the exact score. Effective own-custody frontier remains MC36 Variant C: `S = 0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`. Own-vehicle LC2 remains `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.

## NEXT_IF_RESUMED

- `FIRED_EXISTING_OWNER_NO_DUPLICATE` — owner: trained-receiver #982 / RX2; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac`; fire trigger: the live governed training reaches its terminal QAT checkpoint; consume its export-fit-encode-receiver identity race.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN #978 scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1`; fire trigger: higher-priority exact rows are terminal, the T4 lane is free, and all sealed hashes match; fire only the MT1 sign gate and train only if `positive_t4_sign=true`.

## LIVE-HYPOTHESES

- Gate-aware joint training may preserve EC2’s corrected cells without its Road collateral. This is plausible because the refused adapter was trained for uniform application, never for the gated distribution.
- Joint multi-token conditioning may capture boundary geometry lost by a scalar class-pair gate. EC1 found decoded context highly predictive, while BG2 closed only the low-rank frame-state formulation.

## DEAD-ENDS

- Uniform terminal EC2 reuse: 12,075 errors fixed versus 52,854 introduced.
- Assigning EC2’s loss to a particular stage from current custody: stages 10 and 20 lack scored endpoints.
- Post-hoc Road-hub scalar gating on the local admission rail: 7 recovered flips versus 1,373.803 required.
- QS5 source-restoring compensation for this exact JS8 object: Seg plus rate is already positive.
- Generic bilinear gating on the existing 8-D frame state: held-out incremental R² is negative.
- Standalone implicit-context rate calibration and direct sparse Road-Lane events: respectively only 2 bytes saved and unacceptable collateral.