The named crux is closed: the best ξ-advected exact partition costs **453,449 B**, versus **356,636 B** intra. The real-code ratio is **1.271461658385581**, so F1 fires and full-pixel ξ/XOR transport is dead for this formulation on this clip.

## THE FLOOR LADDER

`N = 37,545,489` and `S = 100*d_seg + sqrt(10*d_pose) + 25*B/N`.

| Representation class | Counted bytes and status | Rate term | d_seg assumption or measurement | d_pose assumption or measurement | Floor / point in S | What the row actually establishes |
|---|---:|---:|---:|---:|---:|---|
| (a) PR135/cp135 dense semantic-token achiever | **186,252 B MEASURED complete archive** | **0.1240175617** | **0.00029643** measured, rounded | **6.88e-06** measured, rounded | **0.1619551383 exact row** | An achieved upper bound on the optimum and the current effective frontier, not a universal lower bound. Its lossless same-representation rate axis is closed on present evidence, but its distortion is not irreducible. |
| (a2) Exact dense intra partition plus the exact carried B1 pose/calibration description | **363,524 B MEASURED description** = 356,636 + 6,864 + 24 | **0.2420557101** | 0 only under an ideal evaluator-inverse realizer | 0 only under an ideal evaluator-inverse realizer | **0.2420557101 ideal-realizer point** | Best per-class mixture of the three tested real coders and five omitted-class choices. It is an exact semantic/pose description, not an RGB receiver or a Shannon lower bound. It is already dominated by cp135. |
| (b) Exact xi-advected partition plus the exact carried B1 pose/calibration description | **460,337 B MEASURED description** = 453,449 + 6,864 + 24 | **0.3065195129** | 0 only under an ideal evaluator-inverse realizer because every flicker event is retained | 0 only under an ideal evaluator-inverse realizer | **0.3065195129 ideal-realizer point** | F1 fires: 1.271462x intra. This exact-raster temporal class is dominated before RGB realization. |
| (b-F2) Required smooth-label/flicker diagnostic, using the later measured 875 B lossy pose point plus 24 B calibration | **454,348 B DERIVED cross-formulation diagnostic** | **0.3025316836** | **0.0053184** FL1 smooth-label floor | **0.001610** R1 lossy-pose point | **0.9612574590 diagnostic** | F2 fires overwhelmingly. This is deliberately not called an achievable codec row: adding FL1 to the exact innovation stream double-counts flicker, and the pose point is from another measured formulation. The diagnostic exists only to execute F2. |
| (c) Representation-free remote/joint rate-distortion optimum | **B_min unknown and not computable from present receipts** | `25*B_min/N` | no proven positive floor | no proven positive floor | **only `S > 0` is presently rigorous; no nontrivial numeric floor** | U2/Kolmogorov identifies the shortest legal witness program but cannot compute it. Joint/indirect RD identifies the optimization and sufficient statistic but has no calibrated empirical distribution/solver here. |
| Historical `T_floor` | **177,169 B historical achiever-class byte anchor** | **0.1179695649** | assumed 0 | assumed 0 | **0.1179695649 derived rate-only counterfactual** | Not a universal floor. It is the rate of an older complete achiever with both distortions set to zero by assumption. The live cp135 rate-only counterfactual is 0.1240175617 instead. |

The “124 KB” ambiguity is resolved: **0.1240175617 is cp135’s rate contribution in score units**, while its complete archive is 186,252 B. Likewise, `T_floor=0.11797` was `25*177,169/N`, not a representation-free proof.

Per-class ξ/intra real-code ratios:

| Road | Lane | Undrivable | Movable | MyCar |
|---:|---:|---:|---:|---:|
| 1.281310 | 1.270697 | 1.223610 | 1.394638 | 1.164540 |

Persistence also loses at 435,536 B, or **1.221234× intra**. Xi then adds another 4.11% over persistence.

The conditional BEYOND gate did not open, so TF1 has no honest first byte-closed candidate projection. The nearest sourced existing HY1 proxy remains **S=0.1475191686**, but it combines separately measured F26 `+11 B` arithmetic with 100% C1 realization and is not byte-closed against cp135. Modern work supports local motion context and task-space coding, but supplies no transferable contest floor: [MPEG-4 binary-shape CAE](https://www.tnt.uni-hannover.de/papers/data/368/368_1.pdf), [CVPR 2025 task-aware video compression](https://openaccess.thecvf.com/content/CVPR2025/html/Gadot_RL-RC-DoT_A_Block-level_RL_agent_for_Task-Aware_Video_Compression_CVPR_2025_paper.html), and [ICCV 2025 SweetTok](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_SweetTok_Semantic-Aware_Spatial-Temporal_Tokenizer_for_Compact_Video_Discretization_ICCV_2025_paper.html).

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_tf1_theoretical_floor_and_beyond_20260812.md)
- [Measurement implementation](/Users/adpena/Projects/pact/experiments/ddm_tf1_partition_innovation_floor.py)
- [Final retained receipt](/Volumes/VertigoDataTier/pact/ddm_tf1_20260812/final_v2/TF1_PARTITION_INNOVATION_RECEIPT.json), SHA-256 `6e8d338f…d13ea`
- 15/15 raw fields and 45/45 coder payloads matched across deterministic runs; 60/60 final containers and all 600 reconstructed partitions verified.
- Ruff, compilation, the 34 payload-retention tests, and both required review-tracker passes passed.
- Committed as `0e6aa91bb3` with `[no-triality] [p0-ledger-ok]`.

No scorer, RGB realization, R round trip, exact evaluator, or pointer movement occurred. Own-vehicle frontier remains **LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — #984/HY1 whole-container build.** Owner: HY1/js1 whole-container builder. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/`. Fire trigger: ps135 emits its terminal safe-run receipt and final archive; adapt its probability object to C1, recount the complete archive, and prove independent decoding before requesting a scorer lane.
- **QUEUED-WITH-A-FIRE-ORDER — js5 content-distinct extension.** Owner: MAIN training-leg router. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200`. Fire trigger: a C1/HY1 event-coordinate producer exists; stop at the first useful nonzero admission or 200 retained unique proposals.
- **QUEUED-WITH-A-FIRE-ORDER — cl1 event-coordinate HPAC prior.** Owner: ddm_cl1_capacity MAIN Metal executor/harvester. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/`. Fire trigger: repair causal-hash provenance and Gate-3 nesting, pack both lambda-1 terminals with exact equality, then fire lambda 0.5.
- **QUEUED-WITH-A-FIRE-ORDER — exact promotion.** Owner: MAIN. Consumer store: MAIN exact-evaluation receipt store. Fire trigger: a deterministic complete n600 candidate strictly improves local full score and passes custody/compliance.

## LIVE-HYPOTHESES

- Local curve/event coordinates may still beat dense intra because they avoid coding already-compressible cell interiors as fragmented XOR fields.
- C1/HY1 event coordinates may supply js5’s missing content-distinct proposals; js5 changed amplitude and width, but not representation.
- Joint HPAC/receiver training may exploit event coordinates where SR1’s frozen, post-hoc contexts could not.
- HY1 may approach its sourced 0.1475191686 proxy if receiver realization preserves enough C1 Seg gain without pose regression.

## DEAD-ENDS

- Full-resolution five-class ξ/XOR partition transport: 453,449 B versus 356,636 B intra.
- Identity persistence: 435,536 B, also worse than intra.
- `T_floor=0.11797` as a universal theoretical floor: it is a historical achiever-rate counterfactual.
- FL1 `d_seg=0.0053184` as a universal hard floor: it binds temporally smooth label witnesses only.
- The old “pose about 2 KB” estimate as measured entropy: current retained evidence is 6,864 B exact B1 carriage or a separate 875 B lossy point.
- Post-hoc frozen-HPAC edge/pose context: SR1 saved only 2 B on edge context and lost 43 B on pose context.