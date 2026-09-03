# ddm_ol1 online signal scan — priced public mechanisms after `afr1`

Date: 2026-09-03  
Arm: `ddm_ol1_online_signal_scan`  
Evidence class: source-inspected public research plus bounded local corpus recall; **no build, no scorer,
no archive mutation, no Modal/Metal, and no public interaction**  
Score claim: false

## Result

The outside scan found **no falsifier on the current object**. No public mechanism with a comparable
denominator prices to at least **10,504 B** (25% of the live 42,016 B demand) while holding the decoded
field and distortion fixed. The only outside mechanism large enough to be interesting is NerVast's
cross-chunk sharing, but that saving exists only after replacing the current single shared generator
with a multi-chunk generator. It is therefore a **different-object** lead, not a current-object rate
win, and it is already folded into `ddm_rn1`'s queued new-generator form.

The useful negative is sharper than “nothing online helps”:

- current-object lossless adaptation is already represented by the shipped HPAC/online corrector and
  by the local OPAL-class races; the public PR #138 headline projects to only **4,631 B** on today's
  113,411 B token section, and our own optimal-form geometric-mixer race realized only **560 B** on
  its then-live object;
- modern contour/chain coders can improve an obsolete contour baseline by tens of kilobytes, but even
  the paper's optimistic ratios leave that baseline **77,864–96,525 B larger** than today's token
  section;
- modern pose/time-series compressors price below **4,402 B** even before the severe denominator
  mismatch with our already-quantized 22,010 B carrier;
- the literature's strongest 2024–2026 direction is not another coder on the same bytes. It is a
  changed generator object: shared scene/GOP factors, static/dynamic separation, and sparse-coded
  weights/modulations.

No candidate was built or measured by this arm. Every number introduced below is **DERIVED** from a
source-reported ratio or a retained public claim, with its denominator stated. Local rows called
MEASURED retain their original axis and citation.

## Live arithmetic and authority boundary

The charter and common contract contain historical frontier statements. The live board supersedes
them. At scan time the owned and effective frontier is `afr1`:

`S = 0.14797617125559104 = 0.020139 + 0.007981227975693965 + 0.11985594327989708`

at **180,002 B**, with `d_seg = 0.00020139`, `d_pose = 0.00000637`, archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`,
`[contest-CUDA T4, n600]`. The sub-0.12 gap is 0.027976171255591042. At held distortion,
`42,016 B × 25 / 37,545,489 = 0.0279763`, so the target archive is at most 137,986 B. Rate exchange is
`25 / 37,545,489 = 6.658589531221714e-7 S/B`.

Live section denominators are token **113,411 B**, generator model **13,515 B**, renderer **30,856 B**,
pose carrier **22,010 B**, and container **217 B**. A field error costs 1.273 B-equivalent per argmax
site at the sharp optimum; a correction costs about 0.29 B/site. A headline byte saving that changes
the field is therefore not a held-distortion saving unless its correction burden is included.

Rule 118 boundary: a generic decoder algorithm may live free in `inflate.py`; learned weights or any
other video-derived content remain counted even if trained elsewhere. The 30-minute T4 budget has
**1,268 s arithmetic slack** after the current 532 s inflate, but every decode-fit statement below is
a source-based estimate, not a Pact runtime measurement.

## Contest surface, 2026-08-06 through 2026-09-03

The [official challenge repository](https://github.com/commaai/comma_video_compression_challenge)
still supplies the governing README, evaluator, 30-minute CPU/T4 envelope, rule-118 boundary, and MIT
repository licence. Search-engine copies of its leaderboard were stale at PR #130, so I did not use
them as current leaderboard authority. The bounded public/local surface resolved as follows:

- [PR #136 `hnerv_rc`](https://github.com/commaai/comma_video_compression_challenge/pull/136):
  author-reported 177,998 B and S 0.192584, no eval-bot authority. It is a PR95/HNeRV reskin plus a
  simple adaptive range coder. Relative to `afr1` it saves 2,004 B of rate (0.001334 S) while losing
  roughly 0.0446 S overall. The producer was also non-resumable/broken in the prior intake.
- [PR #137 `metric_shift_av1`](https://github.com/commaai/comma_video_compression_challenge/pull/137):
  retained public archive 866,558 B; author-reported S about 2.04, `d_seg = 0.00571624`,
  `d_pose = 0.07881037`, no eval-bot authority. AV1 segments, film grain, and a 1 B/frame luma
  correction are not competitive with the semantic packet: the archive is 686,556 B larger than
  `afr1` before its much worse distortion.
- [PR #138 `opal_v1`](https://github.com/commaai/comma_video_compression_challenge/pull/138):
  retained public archive 182,040 B, SHA-256
  `bd9a47149b52a8f4986758e9274e509836bfa9c89f9b5cb069e90837eeb18400`; author-reported
  S 0.1591495384, no eval-bot authority. Its isolated, decode-identical token move is
  114,706 → 110,022 B (−4,684 B): rank-one maximal/complement projection, 55 causal index
  expressions, engineered wavefront scan order, and 49.4 MB of decoder-regenerated adaptive state.
  The corpus already inspected and raced this family in optimal fixed-point/geometric form.
- PR #139: **did not find a usable source, archive, or technique claim** in the retained Aug-17 PR
  census, the bounded current repository/search cache, or local research corpus. This is scoped
  absence, not a claim that PR #139 never existed.
- [PR #140](https://github.com/commaai/comma_video_compression_challenge/pull/140): our posted `afr1`
  archive above. Local custody pins branch head `7f29354`, base `db52c5a9f0`, and exact T4 receipt.
  It is the current vehicle, not new outside signal.
- No newer PR with a source-verifiable mechanism or score was found in the bounded 2026-09-03 public
  surface. The public page cache was not fresh enough to support a global “none exists” claim.

## Ranked candidates

Rank means expected utility for the present campaign, not headline compression ratio. “Byte effect”
is a rate-only projection until a row says MEASURED. Negative is smaller. `ALREADY-MEASURED-HERE`
means the mechanism was already exercised locally or the public artifact was already intake-audited;
it does not promote the public author's score to authority.

| Rank | Mechanism and source | Object replaced | Priced byte effect and comparability | Distortion / receiver risk | Code, licence, and 30-minute fit | $0 first measurement | Verdict |
|---:|---|---|---|---|---|---|---|
| 1 | [NerVast, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Lee_NerVast_Improving_Implicit_Neural_Representation_for_Video_Compression_with_Neural_Network_WACV_2026_paper.html): share information across chunk INRs; paper reports **39.9% average parameter reduction** | A *multi-chunk* generator, not today's single/shared generator | Applied only as a changed-object bound to GF1's 47,603 B packet: `0.399 × 47,603 = 18,994 B`, `ΔS = −0.012647`; 45.2% of the 42,016 B demand, leaving 23,022 B. Transfer is not valid on `afr1` because there are no duplicated chunk models there. | Must reproduce the same realized field; any capacity loss enters the 1.273 B/site law. Paper quality gains do not prove argmax identity. | No official NerVast implementation/licence found on the bounded primary pages; base-model links are not this method. Paper reports >30 fps reconstruction, so decode plausibly fits, but Pact packet construction is unmeasured. | On RN1's retained generator instrument, split into ≥2 chunks, factor shared scene trunk vs chunk residuals, serialize every variant, and compare exact decoded mismatch plus packet bytes. | **CHARTER-WORTHY**, but **FOLDED** into RN1's already-queued new-generator form; do not open a duplicate arm. |
| 2 | Public PR #138 OPAL rank-one/projector mixer + causal scan order | Current 113,411 B token field | Public ratio: `4,684 / 114,706 = 4.0835%`; projection `0.040835 × 113,411 = 4,631 B`, `ΔS = −0.003084`, only 11.0% of demand. This projection is generous: local OPAL-class optimal-form race measured −560 B on its then-live object. | Decode-identical if arithmetic state remains synchronized. Public C uses `exp()`/libm across ~117.96M symbols; one ULP can avalanche the range decoder. | Public diff sits in MIT challenge repo, but direct copying remains provenance-sensitive and is rejected on exactness grounds. HX1 models ~2–15 min marginal CPU, so 532 s + that range plausibly fits but is unmeasured on T4. | Already done: ME1/FX1 raced fixed-point geometric variants over n600 and byte-closed the winner. Any revisit needs a genuinely new context family plus integer state-sync. | **ALREADY-MEASURED-HERE** — `ddm_hx1_pr_wave_harvest_20260817.md`, `ddm_me1_micro_edit_engine_20260817.md`, `ddm_fx1_fixed_point_logistic_mixer_20260817.md`. |
| 3 | [SINR sparse weight representation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jayasundara_SINR_Sparse_Weight_Representation_for_Implicit_Neural_Representation_CVPR_2025_paper.html) | A generator-weight packet such as GF1 | Paper/project comparison is about **15%** on a similar reconstruction point (INRIC ~2.0 bpp vs SINR ~1.7 bpp). Generous direct projection: `0.15 × 47,603 = 7,140 B`, `ΔS = −0.004755`, 17.0% of demand. Architecture and image-denominator transfer are weak; this is not a current-object effect. | Reconstructed weights must preserve the sharp argmax field; small weight error can be expensive. | [Project page](https://dsgrad.github.io/SINR/) is public; no bounded, licence-resolved vendoring path was established. Sparse reconstruction should fit the 1,268 s slack, but was not timed. | Serialize GF1 weights under the same tensor shapes with a deterministic sparse random-basis reconstruction; compare packet bytes and exact n600 field mismatches before any scorer. | **PRICED-OUT** for the live demand; the technique was already named in `residual_inr_overturn_oss_research_20260630T235712Z.md` and `goldmine_hunt_20260712.md`. |
| 4 | [CAECC semantic-map contour coding, arXiv 2603.03073](https://arxiv.org/abs/2603.03073): extended chain code + 3OT fallback + Markov arithmetic + shared-boundary skips | Local LTG1 contour stream, **not** HPAC tokens | Paper average is ~18% vs contour/generic baselines. Optimistic LTG1 projection: `0.18 × 233,262 = 41,987 B`; result 191,275 B, still **77,864 B larger** than the current 113,411 B token section. Applying 18% directly to HPAC would be denominator fraud. | Lossless label reconstruction avoids field distortion, but topology fragmentation changes which ECC/3OT mode wins. | [C++ source](https://github.com/InterDigitalInc/LosslessSegmentationMapCompression) is public. Licence text was not recoverable in the bounded page cache, so vendoring is blocked pending a licence read. Paper reports large decoder-time reductions vs generic baselines; likely fits, not Pact-measured. | None warranted. The ceiling is already below the incumbent on the comparable local contour baseline. | **PRICED-OUT**. |
| 5 | [CC-SMC, 2024](https://www.sciencedirect.com/science/article/pii/S1047320324001780): inter-frame chain-code differences for lossless semantic maps | LTG1 contour stream | Source reports >10% bit saving. Giving it exactly 10%: `0.10 × 233,262 = 23,326 B`; result 209,936 B, still **96,525 B larger** than current tokens. Temporal transfer is additionally overstated because current temporal structure is already shipped. | Lossless if decoded exactly; constant-velocity motion is not its claim, but local temporal context has little unused space. | [Public code](https://github.com/Yang-Runyu/CC-SMC) exists; licence was unresolved in the bounded cache. Reported >20% decoding-time saving suggests fit, not Pact-measured. | None warranted unless a source-derived ratio against a learned HPAC-class baseline appears. | **PRICED-OUT**; local temporal alternatives are also closed in `ddm_mc1_motion_compensated_previous_plane_20260903.md` and `ddm_tk1_temporal_context_arithmetic_20260808.md`. |
| 6 | [AFC lossless floating-point compressor, 2024](https://www.sciencedirect.com/science/article/pii/S0020025523014329) | 22,010 B pose carrier | Source says at least 20% compression-ratio improvement vs other float compressors. Impossible best-case transfer: `0.20 × 22,010 = 4,402 B`, `ΔS = −0.002931`, 10.5% of demand. Our carrier is quantized/int-coded rather than a raw float series, so real transfer should be lower. | Lossless would hold pose values, but packing/receiver exactness must survive. | No source/licence-resolved reference implementation found. At 600×6 values, any sane decoder fits; unavailable code blocks vendoring. | None: even the generous upper bound misses the falsifier and cannot close the demand. | **PRICED-OUT**. |
| 7 | [Rabbit lossless time-series codec, 2026](https://www.mdpi.com/2073-8994/18/4/558) | 22,010 B pose carrier | Source average improvement 4.15% over ACTF: `0.0415 × 22,010 = 913 B`, `ΔS = −0.000608`, 2.2% of demand. The reported “up to 43%” is not an admissible average or comparable denominator. | Lossless by paper claim; current carrier's custom integer law leaves little raw-float redundancy. | Article is public; no reusable implementation/licence found. Tiny data volume would fit if implemented. | None. | **PRICED-OUT**. |
| 8 | PR #136 per-tensor adaptive range coder / `min(brotli, RC)` lesson | Whole public archive / mixed sections | Public archive 177,998 B is 2,004 B smaller than `afr1`, rate benefit 0.001334 S, but author-reported S 0.192584 is ~0.044608 worse overall. The useful `min` selector is only order 10²–10³ B on mixed streams. | Frozen distortion is much worse; non-resumable producer and missing optimal section selection prevent reuse as a vehicle. | Small Python/constriction implementation in MIT parent repo; public exact runtime not established. | Already intake-audited; only per-section min-selection would merit a cheap existing-payload check. | **ALREADY-MEASURED-HERE** — `ddm_pi136_leaderboard_breadth_intake_20260810.md`, `ddm_hx1_pr_wave_harvest_20260817.md`. |
| 9 | PR #137 metric-shift AV1 + film grain + frame luma byte | Full archive | `866,558 − 180,002 = +686,556 B`; distortion is also far worse. No favorable denominator exists. | `d_pose = 0.07881037` dominates; correction is cosmetic rather than scorer-closed. | Public PR under MIT parent; decode apparently produced an author row, but no eval-bot/runtime authority. | None. | **ALREADY-MEASURED-HERE** — public artifact retained and audited in `ddm_hx1_pr_wave_harvest_20260817.md`. |
| 10 | [CALLIC/HPAC learned lossless image prior](https://arxiv.org/abs/2511.10991) plus online corrective law | Current token field | **0 new B** from “adopting” it: this is already the current 113,411 B token substrate. PPMd/ZPAQ replacements were about 3× worse; zero-stored causal escape members all grew the field, best +10,818 B on their measured object. | Decode-identical locally; new members still face state-sync and runtime. | Local receiver is already shipped and fits in the measured 532 s total. External learned weights would count; generic algorithm remains free. | No duplicate adoption. A new member must pre-price negative code length against retained tokens and keep its payload. | **ALREADY-MEASURED-HERE** — `ddm_ef1_token_entropy_floor_20260822.md`, `ddm_oe1_online_escape_member_20260822.md`, `ddm_gs3_gestalt_after_submission_20260903.md`. |

## Unpriced signals — not ranked

These papers are relevant design evidence but supplied no ratio that can honestly be transferred to a
live Pact section. None is a charter by itself.

| Signal | What is useful | Why UNPRICED / code boundary | Disposition |
|---|---|---|---|
| [T-NeRV, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Saethre_T-NeRV_An_Adaptive_Encoder_for_Implicit_Video_Representation_with_Transformer_CVPR_2024_paper.html) | Entropy-constrained frame and GOP embeddings; reported quality gains at matched bitrate | No same-field byte ratio; learned embeddings/weights count. Public method code/licence not established in the bounded primary page. | **FOLDED** into RN1's scene/GOP-factor generator form. |
| [DS-NeRV, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_DS-NeRV_Implicit_Neural_Video_Representation_with_Decomposed_Static_and_Dynamic_Codes_CVPR_2024_paper.html) | Separate static/dynamic codes and sampling | No transferable packet-byte ratio or argmax curve. Learned codes count. | **FOLDED** into RN1; do not open a second static/dynamic arm. |
| [C3, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_C3_High-performance_and_Low-complexity_Neural_Compression_from_a_Single_Image_or_CVPR_2024_paper.html) | Per-object overfit latent plus entropy model | Natural-image RD, not a frozen field; no live-section ratio. Model and latent count. | **FOLDED** as generator representation evidence, not a current candidate. |
| [RNeRV / INR design study, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Williams_How_to_Design_and_Train_Your_Implicit_Neural_Representation_for_WACV_2026_paper.html) | Architecture/training laws and public [INVRB code](https://github.com/mgwillia/invrb) | Reports quality/parameter tradeoffs rather than a same-quality byte reduction; licence must be read before reuse. | **FOLDED** into RN1; no numeric row. |
| [UniTAC, arXiv 2608.16696](https://arxiv.org/abs/2608.16696) | Integrated-gradient task map and one task-aware codec for multiple downstream tasks | Reports task accuracy/PSNR at stated bpp, not bytes at equal frozen-task error. Learned ViT weights and transmitted importance information count. No bounded public implementation/licence found. | **FOLDED** as independent support for task-saliency routing already native to Pact. |
| [RL-RC-DoT, CVPR 2025](https://research.nvidia.com/publication/2025-06_rl-rc-dot-reinforcement-learning-rate-controller-downstream-object-tracking-task) | RL chooses standard-codec macroblock QPs without task input at inference | No transferable rate ratio at equal Pact error; it remains RGB/video-codec shaped. Code/licence not established from the primary page. | **FOLDED** as confirmation of task-aware allocation; no AV1 revival. |
| [GIViC, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Gao_GIViC_Generative_Implicit_Video_Compression_ICCV_2025_paper.html) | Generative implicit decoder; paper reports BD-rate gains against visual codecs | BD-rate on RGB/YUV quality is not a byte effect on exact Seg/Pose cells. Diffusion weights count; primary page said code would be available, but no usable code/licence was found. | **PRICED-OUT-AS-UNCOMPARABLE**; not ranked. |
| [Generalized slimmable INR, 2026](https://www.mdpi.com/2079-9292/15/16/3609) | One checkpoint supports multiple rate points; reported 2.3–2.5× storage efficiency across four configurations | We ship one rate, so cross-checkpoint storage reduction has no single-packet denominator. Learned checkpoint counts. | **PRICED-OUT-AS-ZERO** for a one-rate archive. |
| [LosslessINR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Lossless_Compression_with_Implicit_Neural_Representations_CVPR_2025_paper.html) | INR-based exact residual coding | Already present in the corpus; image benchmarks do not price our partition/token section. | **FOLDED**; no duplicate charter. |
| [SIR/SNC, ICLR 2026](https://openreview.net/forum?id=QzPjq3R6ie) | Sparse implicit representation near SDF surfaces | 3D mesh/SDF denominator, not 2D argmax partitions; no matched byte/error curve. Public [code](https://github.com/yydlmzyz1/SIR-SNC) exists, licence unresolved here. | **PRICED-OUT-AS-UNCOMPARABLE**; direct transfer closed. |

## Prior-law verdict

**Prediction survives.** The threshold was 10,504 B on the current object with distortion held.

- OPAL is the strongest source-derived current-object claim: 4,631 B projected, only 44.1% of the
  falsifier threshold and 11.0% of the total demand. The local optimal-form family result is smaller.
- NerVast derives 18,994 B, but only by creating and then sharing a multi-chunk generator. That is the
  charter's predicted “anything larger must be a different object” case.
- CAECC derives 41,987 B only against LTG1's obsolete 233,262 B contour baseline; it remains 77,864 B
  behind the live token section. It cannot be counted as a 41,987 B improvement to the current object.
- No pose coder reaches 10,504 B even under an impossible generous transfer from raw-float sources.

There is therefore no source-backed reason to spend a scorer slot, paid dispatch, or build cycle on a
same-object online mechanism from this scan.

## GESTALT-DELTA

**No change to where sub-0.12 lives.** It still requires a changed generator/semantic-primary object,
not another lossless coder over today's bytes. The outside literature sharpens the preferred form:
share a static/scene trunk across chunks, give GOP/dynamic residuals their own small codes, and test a
sparse/modulated parameter representation under the exact decoded-field correction price. This is
already the RN1 queued direction. The scan adds no independent current-object door and does not move
the pointer.

## RECALL EVIDENCE

The required original recall was performed before adjudication:

- graph query:
  `.venv/bin/python tools/graph_memory_recall.py --json --max-seeds 16 --max-nodes 80 --max-depth 2
  "lossless segmentation label map contour temporal context online adaptive pose carrier generator form frozen downstream VCM"`;
- canonical-equation inventory:
  `.venv/bin/python tools/list_canonical_equations.py --json`, then content searches for token entropy,
  temporal prediction, partition/contour, online adaptation, pose, generator capacity, and sharp-optimum
  laws;
- corpus surfaces: `.omx/research/` content, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks,
  design/SPEC documents, current state/ledger surfaces, and retained public PR census JSON;
- online queries covered the official contest/PR surface and 2024–2026 primary papers for lossless
  semantic maps, chain/contour coding, learned discrete-field models, boundary/SDF INRs, implicit video
  representations, time-series/pose compression, frozen-network/task-aware compression, and VCM.

Beyond the charter's seed list, recall found four load-bearing closures:

1. `ddm_oe1_online_escape_member_20260822.md` already measured the zero-stored causal online escape
   family: every row grew, best +10,818 B. This removed “just add an online member” from the live list.
2. `ddm_fx1_fixed_point_logistic_mixer_20260817.md` already put the OPAL-class idea into deterministic
   optimal form and byte-closed −560 B. This changed PR #138 from a candidate charter to
   `ALREADY-MEASURED-HERE`.
3. `residual_inr_overturn_oss_research_20260630T235712Z.md` and
   `goldmine_hunt_20260712.md` already named SINR and LosslessINR. This removed duplicate novelty and
   forced their present-object price before any recommendation.
4. RN1 had already read NerVast, T-NeRV, DS-NeRV, C3, and RNeRV and already queued the one form their
   common signal supports. This turned the only charter-worthy outside lead into a `FOLDED` follow-on
   rather than orphan work.

Provenance pins from `git log -1 -- <memo>`:

- `ddm_gs3_gestalt_after_submission_20260903.md` — `b92a78512b4936511a6799b87d5f04bd986cf4a8`;
- `ddm_ww1_walls_that_werent_20260902.md` — `e79b5fef82b4e7c59976dcfb07323261944fa315`;
- `ddm_rn1_n600_reopen_sweep_20260903.md` — `90e560957973a58f8c08e11c0771023f4e4f2206`;
- `ddm_mc1_motion_compensated_previous_plane_20260903.md` — `f9aa0b348ec68ddf9e75b96f9102bd086800d775`;
- `ddm_ef1_token_entropy_floor_20260822.md` — `ac9804ee92b93d48a4f43327372bafd7bdef65e3`;
- `ddm_oe1_online_escape_member_20260822.md` — `e864cb4ab44ee32080e06601cc821eae2e4e7631`;
- `ddm_pi136_leaderboard_breadth_intake_20260810.md` — `b591fb1a4ec6719e1ea21ab703949446b7431eb0`;
- `ddm_hx1_pr_wave_harvest_20260817.md` — `6494d8065e1b0300386250977a6252a2d1d4feb6`;
- `ddm_map1_competition_design_space_20260808.md` — `66495c495ffcf41fed204f803a832cc8c128ad64`;
- `ddm_pi135_pr135_intake_20260810.md` — `244cc5082afce35b30d424cdb24b1c880e1dfc11`;
- `ddm_pr140_submission_posted_20260903.md` — `0da1ff27dd93d00f64d301396c63b12bfef208ac`.

The handoff follows `docs/operating_manual_craft_handoff.md`: typed evidence, scoped negatives,
explicit consumers, and no unowned “interesting” work.

## NEXT_IF_RESUMED

- **Disposition: FOLDED / QUEUED-WITH-A-FIRE-ORDER. Owner:** MAIN's next free scorer-free generator
  builder (the existing RN1 owner). **Consumer store:**
  `/Volumes/APDataStore/pact/ddm_rn1_n600_reopen_sweep/gf1_capacity/`.
  **Fire trigger:** a structurally richer shared-scene/GOP or static/dynamic generator with real
  capacity serializes to **≤71,404.5 B** and decodes to **≤46,804 full-n600 mismatches** under the
  physical residual; retain every packet and exact mismatch receipt. NerVast-style sharing and SINR-
  style sparse parameterization are variants inside this order, not new arms.

## LIVE-HYPOTHESES

- A multi-chunk generator may make cross-chunk scene sharing genuinely pay because NerVast reports a
  39.9% parameter reduction precisely where separate chunk INRs duplicate content; RN1's single/shared
  generator does not yet instantiate the denominator on which that saving exists.
- Static/dynamic or scene/GOP factorization may lower both model bytes and correction burden because
  dashcam backgrounds are stable while lane/actor boundaries carry concentrated change; DS-NeRV and
  T-NeRV independently choose this separation, and it matches the measured sparse hard-region shape.
- Sparse-coded generator weights may complement shared-scene factorization because SINR reports a
  smaller representation at a comparable image reconstruction point; it remains plausible only if
  exact decoded-field mismatches stay under RN1's physical-residual cap.

## DEAD-ENDS

- Another current-object online adaptive token member is closed at **FAMILY** scope for the tested
  zero-stored causal uniform-escape form: `ddm_oe1` measured all four rows larger, best +10,818 B.
- Repeating PR #138's OPAL architecture is closed as duplicate work: ME1/FX1 already inspected,
  determinized, raced, byte-closed, and bounded the family far below the 42,016 B demand.
- Direct contour/chain replacement is closed on the present denominator: CAECC and CC-SMC headline
  gains still leave the comparable LTG1 stream tens of kilobytes larger than current HPAC tokens.
- Constant-velocity decoder-side motion planes are closed at **FORMULATION** scope: `ddm_mc1` found
  every tested global/row-band/affine form worsened held-out code length. Non-constant flow is a
  different family, not permission to retry these planes.
- Generic PPMd/ZPAQ replacement is closed at **FORMULATION** scope: `ddm_ef1` measured roughly 3× the
  trained token stream.
- Raw floating-point pose compressors are closed as a direct transfer: even generous headline ratios
  save at most 4,402 B on the entire current carrier before accounting for the wrong raw-float
  denominator; the local exact recode headroom was only −18 B.
- PR #136/#137 and RGB generative/video-codec revival are closed as current vehicles: their public
  archives and task distortions are dominated, and visual BD-rate is not a frozen Seg/Pose price.
- Direct SDF/3D-surface compression transfer is closed as unpriced: mesh/CFD ratios do not establish
  bytes per exact 2D argmax partition and cannot be used as a current-field saving.

Own-vehicle frontier: **UNMOVED — S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600]**.
