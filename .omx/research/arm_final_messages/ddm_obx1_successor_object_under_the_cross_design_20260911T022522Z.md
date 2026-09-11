Completed and landed `ddm_obx1` as commit `577f61d69`; serializer rc 0. The queue is marked `landed`.

The n600 falsifier rejects the direct Q8 Lane-edge splice: the correction worsened distortion and cost 4.45 MB. No training, Modal, paid dispatch, contest evaluation, or pointer movement occurred. All generated payloads were retained.

Full artifacts: [successor memo](/Users/adpena/Projects/pact/.omx/research/ddm_obx1_successor_object_under_the_cross_design_20260911.md), [falsifier receipt](/Users/adpena/Projects/pact/.omx/research/ddm_obx1_20260911/FALSIFIER_RECEIPT.json), [burn specification](/Users/adpena/Projects/pact/.omx/research/ddm_obx2_edge_local_implicit_correction_burn_spec_20260911.md), and [terminal result](/Volumes/VertigoDataTier/pact/ddm_obx1_successor_object/RESULT.json).

### Object table

Exact custody paths and SHA-256 values are recorded in the successor memo.

| Object | Counted bytes | Own distortion | Authority/scope |
|---|---:|---:|---|
| QBT2B r10 | 106,714 physical; 121,928 HT estimate | 8.6267007501 | macOS-CPU n600 |
| NG1 | 106,565 | 0.3728406704 | MPS n32/600 advisory |
| NG2 | 106,603 | 0.3500590776 | MPS n32/600 advisory |
| NG3 | 106,637 | 0.3208043130 | MPS n32/600 advisory |
| NG4 | 106,662 | 0.3538204559 | MPS n32/600 advisory |
| NG5 | 106,588 | 0.3138606514 | MPS n32/600 advisory |
| TB1 plain | 549,927 component estimate | unavailable; Seg-only | macOS-CPU/MLX n600 |
| TB1 LOTTO | 534,597 component estimate | unavailable; Seg-only | macOS-CPU/MLX n600 |
| BS3 | 101,150 | unavailable | scorer-free bytes |
| BS4Y | 180,369 | 3.5308745780 | macOS-CPU n20 |
| Move-44 pointer | 180,406 | 0.01711995387 | contest-CUDA T4 n600 |
| RI1 | 113,006 | 17.2310453104 | environment-mismatch n600 |
| NI1 | 122,250 | 27.7170345970 | macOS-CPU n600 |
| W96 | 179,290 | 0.4008661665 | contest-CUDA T4 n600 |
| FCD1 batch0/1/2 | 178,900 / 178,952 / 178,951 | unavailable | scorer-free re-encodes |
| FCD1 union | 176,436 | unavailable | scorer-free; n600 decode |
| LB1 | 2,832 carrier only | label-space 0.0012197 | n32; incomplete object |
| OBX1 fused | 4,555,301 research container | 14.5660713068 | macOS-CPU n600 |

### Three successor designs

- **A — edge-local implicit correction lattice:** replace QBF latents with a jointly trained signed-interface multiresolution field, anchored in [LIIF](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html) and [SHACIRA](https://openaccess.thecvf.com/content/ICCV2023/html/Girish_SHACIRA_Scalable_HAsh-grid_Compression_for_Implicit_Neural_Representations_ICCV_2023_paper.html). Selected and sealed pending GO.
- **B — sparse screened-Poisson fusion:** store sparse interface conditions and expand them with a deterministic [Poisson receiver](https://legacy.sites.fas.harvard.edu/~cs278/papers/poisson.pdf). Queued behind A.
- **C — temporal edge-state lattice:** predict a successful correction field through counted keyframes, motion, and innovations, anchored in [DVC](https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.html) and [Rippel et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Rippel_Learned_Video_Compression_ICCV_2019_paper.html). Queued behind A.

### Falsifier result

- Born QBT2B: `d_seg=0.06325726`, `d_pose=0.52944849`, distortion `8.62670075`.
- Fused Q8 splice: `d_seg=0.11243712`, `d_pose=1.10380735`, distortion `14.56607131`.
- Real carrier: 4,448,561 B.
- Research-only counted ZIP: 4,555,301 B, byte-identical repeat.
- `rate + distortion = 3.0331879550 + 14.5660713068 = 17.5992592619`.
- Shortfall versus 0.12: `+17.4792592619`.
- The inherited cross arithmetic was corrected: `121,928 B rate + pointer distortion = 0.0983068043`, not 0.109, but it remains a mixed-object counterfactual rather than an archive.

Validation passed: 5 tests, Ruff, compilation, JSON/diff checks, and two review passes with 100% Python entity coverage. The frontier remains **composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)**.

## NEXT_IF_RESUMED

- **SEALED-PENDING-EXPLICIT-OPERATOR-GO** — owner: MAIN/operator-designated QBF successor owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/`; fire trigger: explicit `GO ddm_obx2`, clean frozen pins, passing storage preflight, and unique training/scorer lane claims.
- **QUEUED-BEHIND-A** — owner: future OBX-B receiver owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_obx2_sparse_poisson_fusion/`; fire trigger: A is refused outside the direct-Q8 instance, or exposes a smooth residual fitting the 15,394 B additive ceiling, plus explicit operator GO.
- **QUEUED-BEHIND-A** — owner: future OBX-C temporal owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_obx2_temporal_edge_state/`; fire trigger: A first produces a parsed ≤122,000 B object with distortion `<0.04` and measurable real-coded temporal redundancy, plus explicit operator GO.

## LIVE-HYPOTHESES

- A co-trained local field may avoid the direct splice’s interpolation spill because its support, values, generator, and quantization adapt jointly through R.
- Sparse screened-Poisson expansion might produce a smoother correction, but the wrong-sign OBX1 result makes it secondary.
- Temporal prediction may compress a successful local field, but cannot rescue a spatial representation that has not first met the distortion gate.

## DEAD-ENDS

- The exact QBT2B r10 plus direct decoded Q8 Lane-edge correction instance is closed: 4,555,301 B and distortion 14.5660713068.
- Transferring QBT2B’s n32 HT distortion to n600 is closed; the complete-population distortion is 8.6267007501.
- Pointer token-tail models, explicit contour/run/endpoint carriers, categorical overrides, and unchanged post-hoc Lane bands remain closed at their measured scopes.
- The BS3/BS4Y HG1 plus inherited/fresh DX2 carrier and downstream one-hidden-layer screen must not be reopened.
- NG schedule, seed, transition, area-cap, tau-band, and carried-dual treatments do not provide the required object change.
- TB1’s component estimates remain incomplete: they are not byte-closed archives and have no Pose measurement.