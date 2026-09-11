# ddm_obx1 — successor object under the cross

Date: 2026-09-11  
Status: `COMPLETE — DESIGN SELECTED; DIRECT Q8 FALSIFIER REFUSED; BURN REQUIRES OPERATOR GO`  
Measurement axis: `[macOS-CPU advisory]`  
Score claim: false  
Promotion eligible: false  
Pointer moved: false

## Verdict first

The retained-byte falsifier rejects the direct construction. QBT2B r10 measures
`d_seg=0.06325725979275174`, `d_pose=0.52944848959797`, and distortion
`8.626700750101595` over all 600 pairs. Adding a decoded int8 RGB correction on the exact shipped
Lane-edge support makes both axes worse: `d_seg=0.11243711683485243`,
`d_pose=1.1038073466877865`, distortion `14.566071306836974`. Its real Brotli carrier is
4,448,561 B and its exact counted research archive is 4,555,301 B. Recomputed advisory
`rate + distortion = 3.03318795501638 + 14.566071306836974 = 17.599259261853355`.

This is an `INSTANCE` refusal, not a family claim. It kills an additive, post-hoc, dense-q8
Lane-edge correction on the unchanged r10 generator. It does not test a local implicit field that is
co-trained with the generator, reallocates the original latent budget, and changes its support under
the parsed forward pass.

The best successor design is therefore **A: a jointly trained multiresolution edge-local implicit
correction lattice**. It has no measured byte/distortion row yet. Its burn is sealed in
`.omx/research/ddm_obx2_edge_local_implicit_correction_burn_spec_20260911.md` and cannot start without
an explicit operator GO.

## The cross, corrected

The charter inherited `0.109` for the counterfactual combination. Recomputing from the current
numbers gives:

```text
25 * 121,928 / 37,545,489 = 0.08118685043628011
pointer distortion         = 0.01711995387438173
sum                        = 0.09830680431066184
```

So the two-number counterfactual is 0.0983068043, not 0.109. It remains **not an archive**:
121,928 B is QBT2B's n32 Horvitz-Thompson `B_hat`, while distortion 0.01711995 belongs to the
move-44 object. The new same-object n600 QBT2B measurement is distortion 8.62670075, which makes the
cross much less evidentially favorable than its separate-axis arithmetic suggests.

At a physical packet target of 122,000 B, the rate term is 0.0812347923 and a sub-0.12 object must
have distortion below 0.0387652077. If `d_pose<=1e-5`, that implies
`d_seg<0.0002876521`; at pointer-like `d_pose=4.59e-6`, it implies
`d_seg<0.0003199025`. These are admission thresholds, not predicted achievements.

## Object inventory — rate and own distortion

Distortion below means `100*d_seg + sqrt(10*d_pose)` and excludes rate. A dash means the retained
receipt did not measure a complete own-object distortion; it is not filled with a transferred
number. “Component estimate” is not a byte-closed archive.

| retained object | counted bytes | own distortion | axis / denominator | exact custody |
|---|---:|---:|---|---|
| QBT2B r10 physical archive | 106,714 archive; 121,928 `B_hat` | **8.6267007501**; older n32 HT estimate 0.3277115 | new `[macOS-CPU advisory]` n600; old fixed n32 HT over population 600 | `.../stage_05_end/reencode_payloads.tar::archive.zip`, sha `b26371e50696bdcdafdccbf4c629ef1119ae48aa1ac8765200a6ea2176f91830`; container `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r10/stage_05_same_budget_admission/reencoded/stage_05_end/reencode_payloads.tar`, 2,723,840 B, sha `18d69e4da2024d39ef13e73ef92623ca9857e67cc4f7b551f83d557f9880709d` |
| NG1 warm-transition terminal | 106,565 archive | 0.3728406704 | `[macOS-MPS n32 stratified advisory]`, selection 32 / population 600 | `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng1_warm_transition/runs/seed_20260902_warm_transition/milestones/step_005000/reencoded/reencode_payloads.tar::archive.zip`, sha `2056b13864617a2999df6ecd2d16e5c5f6f59d3cd9d5d35581695e134d5bac20` |
| NG2 area-cap terminal | 106,603 archive | 0.3500590776 | `[macOS-MPS n32 stratified advisory]`, 32 / 600 | `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng2_area_cap/runs/seed_20260902_area_cap_control_native100/milestones/step_005000/reencoded/reencode_payloads.tar::archive.zip`, sha `efdb04790345d142eea0756446057d975e60aaaf12d759782b8fc300ffe3d04c` |
| NG3 tau-band terminal | 106,637 archive | 0.3208043130 | `[macOS-MPS n32 stratified advisory]`, 32 / 600 | `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng3_tau_band/runs/seed_20260902_tau_band_control_native100/milestones/step_005000/reencoded/reencode_payloads.tar::archive.zip`, sha `aea9c624e6d3e26dd9e6d33e5eee9f727ffe2270b817e4341fc925023550b80a` |
| NG4 continuous-objective terminal | 106,662 archive | 0.3538204559 | `[macOS-MPS n32 stratified advisory]`, 32 / 600 | `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng4_continuous_objective/runs/seed_20260902_continuous_objective_control_native100/milestones/step_005000/reencoded/reencode_payloads.tar::archive.zip`, sha `9400a9bc57c1d544af02856ae1712b1b541781216199cd586202d45647df7eb5` |
| NG5 tau-band x carried-duals terminal | 106,588 archive | 0.3138606514 | `[macOS-MPS n32 stratified advisory]`, 32 / 600 | `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng5_tau_band_x_continuous_objective/runs/seed_20260902_tau_band_x_continuous_objective_control_native100/milestones/step_005000/reencoded/reencode_payloads.tar::archive.zip`, sha `a989485687c9065f815923441a0a6a03cf52b4adccd428c00c1d52c5eff2a9b8` |
| TB1 plain terminal checkpoint | 549,927 component estimate; **no archive** | —; `d_seg=0.014088`, Pose unmeasured | `[macOS-CPU/MLX advisory]` n600 Seg-only | `/Volumes/VertigoDataTier/pact/ddm_tb1_20260728/t2_n600_plain/checkpoints/stage_seg_trunk_tau_final.npz`, 14,959,145 B, sha `06b512972805bfe331fec078a2728031547106fb0dee83e2e175a27bbff48a76` |
| TB1 LOTTO terminal checkpoint | 534,597 component estimate; **no archive** | —; `d_seg=0.013833`, Pose unmeasured | `[macOS-CPU/MLX advisory]` n600 Seg-only | `/Volumes/VertigoDataTier/pact/ddm_tb1_20260728/t2_n600_lotto/checkpoints/stage_seg_trunk_tau_final.npz`, 14,963,191 B, sha `67454bc9ac30ea6f971c29c6c6cea7b8af8d98293ed77e44be3a14b8a8667bc2` |
| BS3 HG1 zero-residual body + inherited carrier | 101,150 archive | — | scorer-free exact bytes | `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/retained/body/born_small_inherited_carrier.zip`, sha `5743f0ac7e8881e970ef8ba53c4bee3fd2a7a6157d2a50d381fd609ae624fea6` |
| BS4Y HG1 body + fresh solved carrier | 180,369 archive | 3.5308745780 | `[macOS-CPU advisory]`, seeded random 20-pair subset of sealed n32 selection | `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/retained/bs4y/stage_30/archive.zip`, sha `60d4aaa0034f834c57a780c5af0254deebd8c33423cb0d116fdb49edabab6794` |
| move-44 pointer | **180,406 archive** | **0.01711995387** | `[contest-CUDA T4 n600]` authority | `/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime/archive.zip`, sha `04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`; decoded raw 3,662,409,600 B sha `2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc`; shipped field 117,964,800 B sha `a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8` |
| RI1 RC1 | 113,006 archive | 17.2310453104 | `[env-mismatch advisory]` n600 | `/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/build_r2/runtime/archive.zip`, sha `6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a` |
| NI1 NR1 K32 | 122,250 archive | 27.7170345970 | `[macOS-CPU frozen-scorer advisory]` n600 | `/Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/build_r4/runtime/archive.zip`, sha `fe7fe8058376543d5832912e691214969680fea5d85e125e861e9700c5ca534e` |
| SY2-recalled W96 single-FiLM object | 179,290 archive | 0.4008661665 | `[contest-CUDA T4 n600]` | `/Volumes/APDataStore/pact/ddm_rf1_renderer_film_rung/candidate_runtime_r1/archive.zip`, sha `34855e3c43e564d48adc492d919afa81662ebff847386d36bbf1a07304b26d21` |
| FCD1 batch0 | 178,900 archive | — | scorer-free exact re-encode | `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/reencode/retained/candidate_fcd1_batch0.zip`, sha `1ed41531edfecd88f6adb9dc8814d0d38cb6b1090b0b1ccd6dab194ca2e5b187` |
| FCD1 batch1 | 178,952 archive | — | scorer-free exact re-encode | `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/reencode/retained/candidate_fcd1_batch1.zip`, sha `d73dff6622312bd5bcba2718ebd89c6b311305e54b67db7003f88fdd8e15bc66` |
| FCD1 batch2 | 178,951 archive | — | scorer-free exact re-encode | `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/reencode/retained/candidate_fcd1_batch2.zip`, sha `85826a8d65682f591c86b39f25e81033025e8a2239eaeaa72d04c4393065fc14` |
| FCD1 union | **176,436 archive** | — | scorer-free exact re-encode; n600 public decode only | `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/reencode/retained/candidate_fcd1_union.zip`, sha `c45ab4e687d1a598b2c2191e5c4bf176bb1c12b24748795434cd109eb9a3aa6b`; retained raw 3,662,409,600 B sha `042fad94690563d774f9480a0fd136334f7c91e4867386587655d15dd04dff19` |
| LB1 analytic Lane band | 2,832 carrier only; **no complete archive** | label-space `d_seg=0.0012197`; Pose and through-R distortion unavailable | `[macOS-CPU advisory]` n32 nonconsecutive label-space ceiling | `/Volumes/APDataStore/pact/ddm_lb1_lane_band_ceiling/lane_band_lbnd2_dali.br`, sha `1828ce4f5e907ac1765712a602a063497f41782c91c8b3274aa4aaf4f096bbe2` |
| OBX1 direct Lane-edge q8 fused object | **4,555,301 research archive** | **14.5660713068** | `[macOS-CPU advisory]` n600; not a public contest runtime | `/Volumes/VertigoDataTier/pact/ddm_obx1_successor_object/candidate/archive.zip`, sha `a6456ccd2941c9ac2e5fa873e75cef22454593b997756a4c7bf71fdd0ae40a8a` |

SY2 itself produced no additional hybrid bytes; W96 is the exact retained object it used. FCD1's
union has a measured rate action but no scorer action, so it cannot supply a distortion number. LB1
is a carrier and label-space ceiling, not a complete object. These distinctions prevent a table of
real files from becoming a fake table of comparable scores.

## Three successor-object designs

| rank | design | one-line object definition | present disposition |
|---:|---|---|---|
| 1 | A — edge-local implicit correction lattice | Replace QBF's dynamic latent section with a co-trained multiresolution local feature field queried at parsed signed interfaces and emitting chroma-first RGB correction through R. | `SEALED-PENDING-OPERATOR-GO` |
| 2 | B — sparse screened-Poisson receiver fusion | Keep the QBF generator and count sparse interface-indexed RGB/chroma boundary values; a generic deterministic screened-Poisson receiver expands them into a smooth correction. | `QUEUED-BEHIND-A` |
| 3 | C — temporal edge-state predictive lattice | Replace per-pair QBF latents with counted edge-state keyframes, motion, and innovations that generate the correction field before R across the full sequence. | `QUEUED-BEHIND-A` |

### A — edge-local implicit correction lattice

**Mechanism and originality.** LIIF shows how an image function can query local latent features at
continuous coordinates, while SHACIRA shows multiresolution hash features with an explicit
compression model. OBX-A uses those ideas only as mechanism anchors. Its vehicle-specific object is
a QBF signed-interface-gated lattice trained jointly with the base generator, with distinct
frame-0/frame-1 and chroma/luma allocation and exact parsed-through-R scoring. No published weights,
architecture payload, or result is imported. Sources: [LIIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html)
and [SHACIRA, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Girish_SHACIRA_Scalable_HAsh-grid_Compression_for_Implicit_Neural_Representations_ICCV_2023_paper.html).

**Packet accounting.** The current QBF packet has 80,468 B outside its 26,138 B latent section.
OBX-A replaces that section. The entire new dynamic field, entropy metadata, and section framing must
fit within 41,532 B for a 122,000 B packet; all video-derived lattice values and learned weights are
counted. Generic coordinate lookup, interpolation, and fusion code is free under rule 118. No
121,928 B `B_hat` reserve is booked as physical headroom.

**Predicted path.** This is a target path, not an empirical forecast: distill the retained move-44
RGB as the compress-time teacher, then jointly descend the parsed candidate through the frozen
scorer and real R. Admission requires distortion `<0.04`; the sub-0.12 target at 122,000 B requires
`d_seg<2.8765207719e-4` if `d_pose<=1e-5`. The path is plausible because local coordinate features
can spend capacity only where the QBF field is structurally wrong, but the r10 n600 result proves
that no accuracy transfer may be assumed.

**Timing risk.** High until measured: QBT2B has no retained public n600 timing receipt. The proposed
low-resolution hash queries are linear in pixels and levels, but the complete public decode must be
measured twice and remain below 1,260 s. The burn spec targets at most 800 s for the base plus lattice
to leave receiver/IO reserve; that target is unmeasured.

**Different from prior closures.** It is not MD1–MD4 schedule/seed treatment, GDC2's categorical
Cool-Chic residual, GDC3/GDC4's explicit contour/run/endpoint description, LB1's analytic post-hoc
Lane authority, the OBX1 q8 splice, or BS3's one-hidden-layer regressor downstream of an HG1 exact
solve. The carrier participates in training and can reallocate latent capacity.

### B — sparse screened-Poisson receiver fusion

**Mechanism and originality.** Poisson image editing demonstrates deterministic gradient-domain
fusion from boundary/gradient constraints. OBX-B stores sparse, quantized RGB/chroma constraints
indexed along interfaces generated by the QBF base; a generic screened-Poisson multigrid solver
expands them into a camera-plane correction. This is an original receiver object for this vehicle,
not a copied Poisson-editing implementation. Source: [Pérez, Gangnet, and Blake, Poisson Image
Editing](https://legacy.sites.fas.harvard.edu/~cs278/papers/poisson.pdf).

**Packet accounting.** As an additive design on the 106,606 B physical packet, every sparse value,
interface ordinal, quantizer, and checksum must fit in at most 15,394 B before ZIP effects. The
solver and multigrid stencil are generic/free. If the carrier cannot meet this physical limit, the
design must become a replacement section and is no longer the registered B form.

**Predicted path.** Fit constraints against the move-44 teacher but accept only parsed-through-R
results. With pointer-like `d_pose=4.59e-6`, 122,000 B requires
`d_seg<3.1990253845e-4`; the first hard gate remains total distortion `<0.04`. The OBX1 q8 result
makes this hypothesis weak: simple local RGB replacement worsened both axes, so B must demonstrate
that the smooth solve prevents interpolation spill rather than assume it.

**Timing risk.** High. A fixed-iteration multigrid solve is predictable but adds full-frame passes.
The predeclared cap is 250 s of incremental compute and 1,260 s total public decode, both measured
twice. Failure closes the instance.

**Different from prior closures.** GDC4 stores a discrete field's endpoints/runs; B stores sparse
photometric conditions and generates the interior through a free generic PDE solver. It is also not
LB1's label authority or OBX1's direct q8 pasted values. Its risk is precisely that the receiver-side
post-hoc mechanism may share their wrong-sign realization behavior.

### C — temporal edge-state predictive lattice

**Mechanism and originality.** DVC encodes motion plus residuals, and Rippel et al. use recurrent
state and learned warping for video compression. OBX-C applies the predictive principle to the
scorer-visible edge correction state: counted keyframe lattice, ego/flow parameters, and innovations
generate local features before R. It does not predict the pointer token stream. Sources:
[DVC, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.html)
and [Rippel et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Rippel_Learned_Video_Compression_ICCV_2019_paper.html).

**Packet accounting.** C replaces the original 26,138 B latents. Keyframes, motion, innovations,
model values, entropy metadata, and framing share the same 41,532 B dynamic-section ceiling inside a
122,000 B packet. Generic warp and lattice query code is free; learned flow, per-frame state, and all
video-derived tables are counted.

**Predicted path.** First make A's static local field reach distortion `<0.04`, then replace repeated
per-pair field values with predictive state while preserving the exact parsed images. At 122,000 B,
the same `d_seg/d_pose` gates apply. No byte credit is predicted before a real sequential coder
produces it.

**Timing risk.** Medium-high. One stateful warp plus local queries per frame should be bounded, but
there is no receiver timing receipt. The design targets `<900 s` total and must remain deterministic
under restart; any state mismatch is a receiver failure.

**Different from prior closures.** BD1/MC1/tail-context work predicted token symbols or copied a
rigid previous field and was closed on the pointer token tail. C predicts a co-trained RGB correction
state before R and changes the object. It is nevertheless queued behind A because temporal coding
cannot rescue a spatial field that has not first reached the distortion regime.

## Canonical-vs-unique decision per layer

| layer | decision | treatment |
|---|---|---|
| contest objective and byte denominator | `ADOPT_CANONICAL` | Exact `100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489`; recompute from components. |
| QBF packet framing and deterministic repeat | `ADOPT_CANONICAL` | Reuse the proven typed section/CRC/archive parse-back discipline, with a new versioned section ID. |
| base coordinate generator | `ADOPT_CANONICAL` | Start from the QBF continuous RGB generator form, not its n32 accuracy number. |
| local implicit feature field | `FORK_EMPIRICAL` | New signed-interface-gated multires lattice, jointly trained and real-coded; LIIF/SHACIRA are anchors only. |
| photometric PDE fusion | `FORK_PRINCIPLED` | Generic screened-Poisson expansion from counted sparse conditions; no existing vehicle implementation is inherited. |
| temporal correction state | `FORK_EMPIRICAL` | State/innovation grammar is new to this object and must earn every byte through a real coder. |
| scorer loss | `ADOPT_CANONICAL` | Use the exact operating-point component weights and parsed-through-R validation. |
| quantization and entropy pressure | `FORK_EMPIRICAL` | Mixed precision and per-level entropy models are tuned to this field; physical bytes remain authority. |
| receiver output and rule-118 boundary | `ADOPT_CANONICAL` | Emit 1,200 RGB camera frames; generic algorithms free, every video-derived value counted. |
| public timing | `UNCLEAR_NEEDS_EMPIRICAL` | No QBT2B public n600 timing exists; each design must measure twice against 1,260 s. |

## 18-shared-assumption profile

The classifications apply to the proposed implementation, not to the measured OBX1 q8 instance.

| assumption | A | B | C | binding decision |
|---|---|---|---|---|
| SA01 two-frame pose-pair curriculum | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | Evaluator still consumes fixed two-frame pairs. |
| SA02 SegNet last frame only | `FORK_EMPIRICAL` | `ADOPT_CANONICAL` | `FORK_EMPIRICAL` | A/C allocate frame 0 primarily to Pose and frame 1 jointly; B leaves frame 0 unchanged. |
| SA03 stride-2 stem blindspot | `UNCLEAR_NEEDS_EMPIRICAL` | `UNCLEAR_NEEDS_EMPIRICAL` | `UNCLEAR_NEEDS_EMPIRICAL` | No invisibility credit; only parsed argmax can validate a lower internal resolution. |
| SA04 canonical ZIP grammar | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | Typed shards remain inside one deterministic ZIP. |
| SA05 camera-native inflate output | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | Compute internally at scorer scale, but deterministically upsample and emit the required camera shape. |
| SA06 no scorer at inflate | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | No scorer weights or distilled scorer are allowed in this burn. |
| SA07 no end-to-end compress/inflate training | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | Training differentiates through quantization/receiver behavior; runtime remains separate and byte-only. |
| SA08 pair independence | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | `FORK_EMPIRICAL` | C alone uses cross-pair compress/decode state. |
| SA09 canonical score-aware loss | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | Exact component formula and operating-point derivatives remain canonical. |
| SA10 uniform Tier-1 engineering | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | Optimize and checkpoint according to each receiver, not a shared HNeRV schedule. |
| SA11 $5–15 dispatch envelope | `UNCLEAR_NEEDS_EMPIRICAL` | `UNCLEAR_NEEDS_EMPIRICAL` | `UNCLEAR_NEEDS_EMPIRICAL` | OBX1 authorizes $0 only; any burn cost is an operator decision. |
| SA12 fixed 100/1000 epoch duration | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | Stage exits use parsed byte/distortion gates and bounded plateaus, not fixed epochs. |
| SA13 residual-on-A1 composition | `FORK_PRINCIPLED` | `FORK_PRINCIPLED` | `FORK_PRINCIPLED` | The base is QBF; A/C replace latents, B is an explicitly priced QBF fusion. |
| SA14 EMA 0.997 | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | Save and byte-close the EMA shadow at every stage. |
| SA15 Modal/Lightning/Vast-only dispatch | `UNCLEAR_NEEDS_EMPIRICAL` | `UNCLEAR_NEEDS_EMPIRICAL` | `UNCLEAR_NEEDS_EMPIRICAL` | No platform selected or authorized; reproducibility and contest match decide after GO. |
| SA16 fixed nonoverlapping pose pairs | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | `FORK_EMPIRICAL` | C may use overlapping compress-time context but must emit the fixed evaluator pairs. |
| SA17 RGB output | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | `ADOPT_CANONICAL` | Internal chroma allocation is allowed; public output remains RGB. |
| SA18 uniform per-channel quantizer | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | `FORK_EMPIRICAL` | Mixed precision must be selected by own-field score/byte action and real coder output. |

## `$0` falsifier result

The implementation is `experiments/ddm_obx1_successor_object_falsifier.py`. It:

- pins and parses the exact 106,606 B QBT2B r10 packet;
- renders the born object for all 600 pairs through its real camera round trip;
- derives the Lane-side four-neighbour boundary from the shipped `subset6.u8` field and dilates it by
  one four-neighbour cell;
- stores the rounded/clamped int8 difference between pointer and born frame-1 RGB at that support;
- encodes every 20-pair correction through a self-describing Brotli q11 payload, decodes it, applies
  it at the camera receiver, and scores born/fused frames in the same frozen CPU process;
- retains all renders, masks, corrections, coder payloads and repeats, logits, argmax, pose vectors,
  targets, per-pair rows, and stage checkpoints.

The exact denominator is 600 pairs, 117,964,800 Seg pixels, and 3,600 pose values. The support has
1,286,055 cells; 3,842,071 RGB correction values are nonzero and 330,096 saturate int8. Thirty
carrier chunks total 4,448,309 B; the catalog adds framing to 4,448,561 B. The complete research
container is 4,555,301 B and its independent repeat is byte-identical.

| term | cross requirement | measured fused object | shortfall |
|---|---:|---:|---:|
| complete packet/archive | about 122,000 B | 4,555,301 B | **+4,433,301 B** |
| additive carrier above 106,606 B packet | <=15,394 B before ZIP effects | 4,448,561 B | **+4,433,167 B** |
| distortion | <=0.04 design gate | 14.5660713068 | **+14.5260713068** |
| `rate + distortion` | <0.12 | 17.5992592619 | **+17.4792592619** |

The candidate's own-distortion strict byte cap is negative, so no nonnegative archive size can make
this exact fused object sub-0.12. The correction also raises distortion by 5.9393705567 before its
rate cost. The most plausible explanation is interpolation/spill from applying low-resolution q8
differences through the camera round trip, but that attribution is a hypothesis; the measured fact
is the wrong-sign end-to-end action.

Terminal receipt:
`/Volumes/VertigoDataTier/pact/ddm_obx1_successor_object/RESULT.json`, 24,379 B, SHA-256
`ae19bb553e6810b4492d2c8df8ba5cec4f319534bf7a559010faa46f9944b81e`.
The result records `score_claim=false`, `promotable=false`, zero Modal calls, zero contest-evaluator
invocations, and `pointer_moved=false`.

## RECALL EVIDENCE

The complete recall receipt is `.omx/research/ddm_obx1_20260911/RECALL_EVIDENCE.md`. Beyond the
charter seeds, the decisive additions were the BS3/HV3 closure of the earlier HG1 learned-carrier
screen, TB1's no-archive/no-Pose boundary, the canonical 41,526 B n600 LBND2 anchor, the canonical
procedural-residual rate-regression law, and the full 18-assumption matrix. These findings changed
the preferred object from a post-exact-solve carrier regressor to a jointly trained QBF image-space
field and forced all byte claims onto the candidate's own coder.

No standalone `md4` memo was found in the bounded corpus search. The MD4 conclusion is carried by
the GS3 addenda and the MD1–MD3/canonical trajectory records; it is not cited as an independently
hashed file.

## Custody and verification

The new retained root contains 135 files totaling 4,611,769,624 logical bytes. Nothing was deleted,
moved, or reduced to scalar-only evidence. Existing `/Volumes/...` sources were read-only. The
research container is not a public contest candidate: it has no public inflate runtime and supports
no score claim.

Implementation checks after the terminal run:

```text
5 pytest tests passed
ruff: All checks passed
py_compile: exit 0
review tracker: 28/28 runner entities and 5/5 test entities reviewed, 100%, two passes
```

Review receipt: `.omx/research/ddm_obx1_20260911/REVIEWS.md`. Compact falsifier receipt:
`.omx/research/ddm_obx1_20260911/FALSIFIER_RECEIPT.json`.

No training, Modal call, paid dispatch, or contest evaluation occurred. The one local n600 scorer
lane was claimed before launch and marked `completed_local_advisory` after the terminal receipt.

The frontier remains **composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]
(move 44)**.

## NEXT_IF_RESUMED

- **SEALED-PENDING-EXPLICIT-OPERATOR-GO** — implement and burn the A edge-local implicit correction
  object under `.omx/research/ddm_obx2_edge_local_implicit_correction_burn_spec_20260911.md`; owner:
  MAIN / operator-designated QBF successor owner; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/`; fire trigger: explicit
  `GO ddm_obx2`, clean frozen pins, passing two-run-plus-20GB storage preflight, and unique active
  training/scorer lanes.
- **QUEUED-BEHIND-A** — test B's sparse screened-Poisson fusion only as an own-coded parsed object;
  owner: future OBX-B receiver owner; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_obx2_sparse_poisson_fusion/`; fire trigger: A is refused outside
  the direct-q8 instance or A reaches distortion `<0.04` but exposes a smooth residual that fits the
  15,394 B additive ceiling, plus explicit operator GO.
- **QUEUED-BEHIND-A** — add C's temporal state/innovation grammar only after a spatial local field is
  viable; owner: future OBX-C temporal owner; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_obx2_temporal_edge_state/`; fire trigger: A produces a parsed
  `<=122,000 B`, distortion `<0.04` object and its per-pair lattice stream shows positive real-coded
  temporal redundancy, plus explicit operator GO.

## LIVE-HYPOTHESES

- A co-trained local feature field may avoid the wrong-sign interpolation spill of the direct q8
  splice because the base generator, support, values, and quantization adapt together through R. It
  is plausible from the local implicit/hash-grid literature and the MD finding that the generator
  form is wrong at persistent edge sites; it is unmeasured on this object.
- A sparse screened-Poisson expansion may turn a small set of boundary constraints into a smoother,
  cheaper correction than direct q8 values. It is plausible as a receiver mechanism, but OBX1's
  wrong-sign result makes it lower priority and no rate/distortion row exists.
- Temporal prediction may compress a successful edge-local state once that state exists. It is
  plausible from learned-video motion/residual coding, but token-tail temporal closures transfer no
  byte credit and the spatial prerequisite has not been met.

## DEAD-ENDS

- `INSTANCE: QBT2B r10 + direct decoded int8 Lane-edge q8 correction` is closed: 4,555,301 B,
  distortion 14.5660713068, advisory S 17.5992592619. Do not retry it with a different coder while
  leaving the support, values, generator, and receiver application unchanged.
- Transferring QBT2B r10's n32 HT distortion to n600 is closed. The exact physical packet measured
  distortion 8.6267007501 on the complete population; only its byte identity transfers.
- The pointer token-tail route is closed at the measured formulation scope by LS1/LS2 and the
  TC/EB/BD/MC/GDC chain. Do not repackage A, B, or C as another probability/context model on the
  unchanged shipped token field.
- Explicit contours, scanline runs, endpoints, categorical pixel residuals, and a stored override
  map remain closed by GDC2–GDC4 at their measured scopes and prices. The successor must change the
  generator-consumed object, not rename those payloads.
- LB1's analytic Lane band remains closed as a full-authority post-hoc mechanism: 162 configurations
  did not improve label-space d_seg, and its 2,832 B n32 carrier is not an n600 price.
- The BS3/BS4Y HG1 body plus inherited/fresh DX2 carrier and its downstream one-hidden-layer learned
  screen remain closed by their later custody adjudication. OBX2 must not reopen that exact parent or
  reuse its solved deltas.
- Warm transition, area cap, tau-band, carried-dual, initialization, schedule, and seed treatment do
  not supply the needed object change. Their NG rows remain research signals, not a path to transfer
  the born rate into pointer distortion.
- TB1's plain/LOTTO checkpoints are not byte-closed full objects and have no Pose term. Their counted
  component estimates and Seg-only measurements cannot be promoted into cross candidates.
