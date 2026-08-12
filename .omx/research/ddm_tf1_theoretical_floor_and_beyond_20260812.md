# ddm_tf1 — theoretical floor ladder and BEYOND routing

**Date:** 2026-08-12  
**Measurement axis:** `[macOS-CPU scorer-free, cached frozen CPU-torch GT, n600]`  
**Score claim:** false  
**Promotion eligible:** false  
**Verdict scope:** the exact five-class 384x512 partition, coded as four one-hot binary planes, with either no predictor, label persistence, or the existing globally calibrated pose-conditioned screw warp on the 599 scored-pair transitions of this clip

The named crux is resolved. The best exact xi-advected partition container is **453,449 B**, while the best exact intra container is **356,636 B**. The measured real-code ratio is therefore **1.271461658385581**. F1 fires. Global xi transport is dead for this exact-raster formulation on this clip; it is not a family-level rejection of local motion contexts, curve coordinates, or learned representation-level temporal priors.

The effective frontier is unchanged: **cp135 `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`**, archive SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`. The own-vehicle row is unchanged: **LC2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`**, archive SHA-256 `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.

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

The table distinguishes achieved points, tested-code minima, and mathematical lower bounds. A real coder length is an achievable code length, not entropy itself. The true contest optimum is

`inf_Z [100*D_seg(Z) + sqrt(10*D_pose(Z)) + 25*R(Z)/N]`,

not merely `25*R_joint(0,0)/N`: the optimizer may accept distortion to save rate. Neither the canonical joint-RD equation nor U2 currently has the calibrated source distribution and legal-receiver computation needed to produce a nonzero numerical lower bound.

### Reconciliation of the “124 KB” / `T_floor` language

For cp135, **0.1240175617 is the rate contribution in score units, not 124 KB**. The complete archive is 186,252 B: 70,825 B model, 96 B compact residual, 115,231 B RC64 tokens, and 100 B ZIP framing. The output-driving semantic token plane contains 117,964,800 symbols and is reproduced exactly by the retained decoder. The historical `0.11797` was similarly `25*177,169/N`, not an information-theoretic proof that every legal witness needs 177,169 B.

Each ladder row breaks a different assumption:

- cp135 breaks the old 177,169-byte anchor because it is a newer representation and container with a different achieved rate/distortion point.
- Exact intra partition storage breaks neural amortization and pays for the whole evaluator partition explicitly.
- Exact xi innovation breaks intra independence, but on this clip the conditional residual is less compressible, not more compressible.
- Representation-free U2/remote-RD breaks every named representation restriction, at the cost of becoming noncomputable or presently uncalibrated.

## MEASURED INNOVATION-ENTROPY RECEIPT

The measurement stores exactly four of five class planes and infers the omitted class. It races Brotli q11, raw LZMA1, and the existing R7 SMEVR record coder, then races all five omitted-class choices. Frame 0 is intra; frames 1–599 are XOR residuals against the stated predictor. All coder streams round-trip exactly, and all 600 partitions reconstruct exactly for every predictor, coder, and omitted-class choice.

### Full-vehicle real-coder race

| Predictor | Brotli q11 best | LZMA1-raw best | SMEVR best | Per-class mixed-coder winner |
|---|---:|---:|---:|---:|
| intra | 368,195 B | 426,065 B | 363,454 B | **356,636 B**, omit Road, SHA `9ac5ee98...3320` |
| persistence | 503,737 B | 559,455 B | 437,454 B | **435,536 B**, omit Road, SHA `1e261270...92e9` |
| xi screw | 529,561 B | 581,026 B | 457,465 B | **453,449 B**, omit Road, SHA `d92e12fb...2e83` |

Persistence/intra is **1.2212339752576857**. Xi/persistence is **1.0411286323059403**. Thus most of the failure is not a bad calibration marginal: exact binary change fields are already more fragmented than the spatial masks, and the global screw warp adds another 4.11% over persistence.

### Per-class real-code ratios

These ratios use the minimum coder independently for each class. They are diagnostic strata, not additive full-vehicle bytes because a full container uses one coder and omits one class.

| Class | Intra best bytes | Xi best bytes | Xi / intra |
|---|---:|---:|---:|
| Road | 286,140 SMEVR | 366,634 SMEVR | **1.2813098483** |
| Lane | 187,427 SMEVR | 238,163 SMEVR | **1.2706973915** |
| Undrivable | 83,176 SMEVR | 101,775 SMEVR | **1.2236101760** |
| Movable | 57,929 Brotli | 80,790 Brotli | **1.3946382641** |
| MyCar | 28,060 SMEVR | 32,677 SMEVR | **1.1645402708** |

Every class loses. Movable loses most, MyCar least. There is no hidden winning Road/Lane stratum to compose.

### Payload custody

- Final run root: `/Volumes/VertigoDataTier/pact/ddm_tf1_20260812/final_v2/` (about 256 MiB at landing). The pre-optimal-form run remains retained separately under the parent; nothing was deleted.
- Final receipt: `TF1_PARTITION_INNOVATION_RECEIPT.json`, 45,129 B, SHA-256 `6e8d338f13a25e0bcd7b8db54784f04298539dc2896c401fb050a5f76f9d13ea`.
- Immutable state snapshot: `state.receipt_snapshot.json`, 46,264 B, SHA-256 `159a1f31137131d35f8a826b5e66ab70e6338d25df626d671c15320244cbf859`.
- Retained payloads: 15 raw packed fields, 45 real-coder streams, 60 complete partition containers including per-class mixtures, exact carried pose/calibration values, and both source versions used by the resumable run; **124 retained files, zero payload deletions**.
- Source custody: payload materialization source SHA `5a12365234ea67a14b7bce72851e18ba9857f150f70d56c75d69e77d28d686e9`; final source SHA `e584196538780ead3b7855a357e249f54df2c0c16b868541facb4e6ecb53fed8`. The only intervening change removed redundant container re-decodes after each underlying coder stream had already passed exact round-trip; encoding and container bytes were unchanged.
- Determinism repeat: the preserved first run and `final_v2` match **15/15 raw-field SHA-256 values and 45/45 real-coder payload SHA-256 values**.
- Actual carried B1 pose container: 6,864 B, SHA-256 `3121e6e5045d7e3167f385b0b1639327ff6dc52fc6015b2e64b7008dbb637af7`; coded pose substream 6,655 B, SHA-256 `69783b358f615b917f23bd6a3377f331b43c8e918a7b0d731034b1f0b0672e14`.
- GT cache: `gt_n600.npz`, 5,078,017,610 B, SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- cp135 custody pin: 186,252 B, SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
- Final command: `.venv/bin/python experiments/ddm_tf1_partition_innovation_floor.py --run-dir /Volumes/VertigoDataTier/pact/ddm_tf1_20260812/final_v2`.

The carried B1 float16 target values are consumed exactly. They differ from a fresh direct `gt_poses.astype(float16)` at 78/3,600 scalars, maximum absolute delta `6.103515625e-05`; the receipt records this lineage difference instead of asserting false byte identity. The three global warp scalars were previously fitted on the first 100 Road/Lane transitions, so the physical-SE(3) interpretation remains formulation-scoped.

## FALSIFIERS

- **F1 FIRED — FORMULATION CLOSED.** `453,449 / 356,636 = 1.271461658385581 >= 0.9`. Do not rebuild full-resolution one-hot XOR partition transport with this global xi predictor on this clip.
- **F2 FIRED — FORMULATION DOMINATED.** Xi innovation alone is 453,449 B, already 267,197 B larger than the complete 186,252-byte frontier archive. FL1's smooth-label `d_seg=0.0053184` contributes 0.53184 S, equivalent to 798,727.7 rate bytes. The required cross-formulation diagnostic is 0.9612574590 S. FL1 is not a universal flicker floor: it binds temporally smooth label-space witnesses only, and exact innovations preserve the flicker that FL1 removes.

## BEYOND DESIGN

The charter made the boundary-innovation build conditional on class (b) beating class (a). That condition is false. Therefore TF1 does **not** authorize a full-pixel boundary-innovation candidate, and there is no honest TF1 first byte-closed candidate score to report. Writing a numeric candidate would require inventing either a future payload length or a future realized flip count.

The live replacement is a representation-level event-coordinate codec inside the already queued HY1/HR1 composed campaign:

1. **Free generator/receiver:** retain the deterministic PR135-compatible renderer and receiver machinery; add only generic parsing and deterministic transforms to free code.
2. **Counted sufficient statistic:** use the C1/HY1 solved semantic-token/event object, not five full-resolution XOR planes. Keep all learned probabilities, weights, event coordinates, and payloads counted. Do not hide video-derived boundaries in code.
3. **Encode-side realization:** jointly fit token/event state and receiver output through camera uint8/resize/SegNet/PoseNet, with pose in the training loop and complete per-stage checkpoints.
4. **Temporal prior:** if a temporal feature is retried, condition HPAC or the learned proposal generator on local boundary birth/death/curve coordinates. Race the complete model-plus-token package against the no-context control. Do not add a post-hoc raw xi table.
5. **Promotion:** rebuild one complete archive, independently decode, retain a deterministic repeat, measure local full-n600 components, then let MAIN fire the exact row only if it improves.

The nearest sourced projection is the existing HY1 proxy, not a TF1 result: cp135 plus the separately measured F26 `+11 B` wire and 100% realization of the measured C1 Seg gain gives **projected `S = 0.1475191686` with pose unchanged**. HY1 already labels that arithmetic non-authoritative and non-additive across cp135/F26; the first real whole-container candidate therefore has **projected S = not yet computable** until its actual archive bytes and realized Seg/Pose components exist.

For any future representation-level proposal against cp135, the exact admission arithmetic is:

`S_projected = 0.16195513827824176 - 8.477105034722222e-7*F + 6.658589531221714e-7*delta_B + delta_pose_term`,

where `F` is robust full-n600 flips removed, `delta_B` is complete archive-byte growth, and `delta_pose_term` is the measured change in `sqrt(10*d_pose)`. With pose unchanged, break-even is **0.7854791823 robust flips per added byte**, or **1.2731082153 B/robust flip**. Sub-0.15 additionally requires the net improvement to exceed `0.01195513827824176`.

### Named consumer routing

- **#984 / HY1 whole-container campaign — QUEUED-WITH-A-FIRE-ORDER.** Owner: HY1/js1 whole-container builder. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/`. Fire trigger: ps135 emits its terminal safe-run receipt and final archive; bind that base, adapt its probability object to the retained C1 plane, recount the complete archive, and prove independent token decode before requesting a scorer lane. TF1 changes its temporal input from raw pixel XOR to the C1 event/token representation.
- **js5 content-distinct acceptance extension — QUEUED-WITH-A-FIRE-ORDER.** Owner: MAIN training-leg router. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200`. Fire trigger: the representation-level C1/HY1 event-coordinate producer exists; retain every realized payload and stop at the first nonzero useful bare admission or 200 unique proposals. A replay of scalar amplitude shrinks or raw TF1 XOR fields is forbidden.
- **cl1 HPAC prior — QUEUED-WITH-A-FIRE-ORDER under its existing gate.** Owner: ddm_cl1_capacity MAIN Metal executor/harvester. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/`. Fire trigger: fix the causal-hash provenance defect and Gate-3 admission-guard nesting, pack both lambda-1 terminals with exact decode equality, then fire lambda 0.5. TF1 permits only an in-representation local event-coordinate prior whose complete model-plus-token package beats the no-context control; SR1's post-hoc additive edge/pose contexts remain closed.
- **Full-pixel xi partition codec — FOLDED.** Owner: TF1. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_tf1_20260812/`. Fire trigger: none within this formulation; reopen only with a different representation and a preregistered complete-package test, not another global-raster XOR recode.

## LITERATURE CROSSWALK

- Ostermann's MPEG-4 binary-shape review describes the adopted context-based arithmetic encoder with motion compensation: inter CAE mixes causal current-shape pixels with a motion-compensated prior shape. That validates local INTER context as a serious codec family, but it does not predict a gain on this clip. TF1's global-raster implementation loses in every class, and SR1's post-hoc HPAC context saved only 2 B. Source: [Coding of Binary Shape in MPEG-4](https://www.tnt.uni-hannover.de/papers/data/368/368_1.pdf).
- RL-RC-DoT allocates block QP by downstream task reward rather than pixel fidelity. It supports scorer-aware allocation, not a numerical floor transfer to this contest. Source: [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Gadot_RL-RC-DoT_A_Block-level_RL_agent_for_Task-Aware_Video_Compression_CVPR_2025_paper.html).
- All-in-One task-aware coding uses shared and task-specific adaptors under a joint rate/task objective. This supports the proposed shared receiver plus event-specific conditioning split; its rates and task metrics do not transfer. Source: [arXiv:2504.12997](https://arxiv.org/abs/2504.12997).
- Compression Beyond Pixels codes semantic features instead of RGB. It supports the representation-class pivot away from exact raster innovations, not the claim that CLIP-feature rates apply to SegNet/PoseNet. Source: [arXiv:2509.05925](https://arxiv.org/abs/2509.05925).
- SweetTok separates appearance and motion token factors for compact video discretization. It makes a token-level motion/event prior plausible, but supplies no contest-specific lower bound. Source: [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_SweetTok_Semantic-Aware_Spatial-Temporal_Tokenizer_for_Compact_Video_Discretization_ICCV_2025_paper.html).

No literature number was imported into the floor ladder. The papers change the surviving mechanism class only: local/task-aware representation-level conditioning remains open after exact global-raster transport closes.

## RECALL EVIDENCE

Searched the full `.omx/research/`, state, task-status, DAG, and canonical-equation surfaces by content using these query families: `T_floor|0.11797|177169|Kolmogorov|U2|joint rate distortion|indirect RD|CEO`; `partition|lstars|temporal|warp|xi|screw|flicker|CAE`; `PR135|cp135|fd135|pi135|HPAC|token`; `QA39|PE1|SR1|JS5|CL1|HY1|HR1|#984`; and ran `tools/list_canonical_equations.py --json` filtered for the same concepts.

Beyond the charter seeds, recall found:

- the canonical `partition_temporal_transport_amortization_jitter_bound_v1`, whose July n600 zlib proxy had already predicted the same sign (846,116 B screw versus 601,931 B intra); TF1 upgraded that proxy to three retained real-coder races and exact containers;
- the later FL1 scope refinement explicitly forbidding use of `0.0053184` as a universal hard floor;
- the later 6,864 B exact B1 pose container and the R1 875 B / `d_pose=0.001610` lossy point, which replace the obsolete unsourced “pose about 2 KB” assumption;
- PE1's explicit-curve family (106,465 B at k16 but lossy and receiver-unsurvived), which keeps local curve/event coordinates live without making them a floor row;
- SR1's 2 B best additive edge-context saving and +43 B pose-context loss, which closes post-hoc context but not joint representation-level conditioning;
- HY1's current C1 solved-token head and HR1's staged realization gates, which prevent TF1 from inventing a duplicate BEYOND candidate;
- the live cp135 pointer, which supersedes the stale frontier text embedded in the common contract.

These findings changed the plan: the run consumed the exact carried B1 values instead of recoding fresh cache poses; FL1 is used only for the mandated F2 diagnostic; full-pixel xi is folded; and the surviving event-coordinate hypothesis is routed into existing HY1/js5/cl1 consumers rather than launched as an orphan.

## Boundaries

- No RGB frames, scorer forward, R round trip, archive receiver, `upstream/evaluate.py`, Modal job, or exact evaluation ran.
- The coder lengths are measured achievable lengths, not Shannon entropies and not archive scores.
- The xi predictor is the existing three-scalar globally calibrated formulation, not a locally fitted optical-flow field or MPEG-4 block motion search.
- Exact partition equality does not prove that a legal compact RGB receiver can realize the partition or pose targets.
- The result moved no exact score and did not move the pointer.

## LIVE-HYPOTHESES

- A local curve/event-coordinate prior may beat both dense intra and global-raster xi because MPEG-style local context does not force the already-compressible cell interiors through a fragmented XOR field, and PE1 measured a much smaller lossy curve description.
- C1/HY1 event coordinates may provide js5's missing content-distinct proposals because js5 varied amplitude and width but never changed the correction representation.
- Joint HPAC/receiver training on event coordinates may pay where SR1's post-hoc context did not because it can change the representation and probability object together instead of asking a frozen stream for additive savings.
- The existing HY1 whole-container head may reach the sourced 0.1475191686 proxy if it realizes at least the required C1 Seg fraction without worsening pose, but this remains unmeasured until the ps135-bound candidate is byte-closed and scored.

## DEAD-ENDS

- Full-resolution five-class one-hot xi XOR partition transport with the current global screw predictor: 453,449 B versus 356,636 B intra, ratio 1.271462; every class loses.
- Identity persistence as a cheaper substitute: 435,536 B, ratio 1.221234 versus intra.
- Treating `0.11797` as a universal information-theoretic floor: it is the rate term of a historical 177,169-byte achiever under assumed zero distortion.
- Treating FL1 `d_seg=0.0053184` as a universal hard floor: the canonical scope is temporally smooth label-space witnesses, while exact phase/flicker-carrying witnesses are outside it.
- Reusing the old “pose about 2 KB” estimate as measured entropy: newer retained streams show 6,864 B exact B1 carriage and a separate 875 B lossy R1 point; neither proves a universal pose floor.
- Post-hoc additive edge/pose context on the frozen HPAC representation: SR1 saved only 2 B for edge context and lost 43 B for pose context; any retry must change and jointly train the representation.
