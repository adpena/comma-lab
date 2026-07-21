---
schema: pact.autoencoder_describe_crosswalk.v1
utc: 2026-07-21T23:23:51Z
lane_id: lane_autoencoder_describe_crosswalk_20260721T231126Z
research_only: true
execution_authority: false
score_claim: false
promotion_eligible: false
axis: "[research-only; exact description-stream byte custody plus source/repository audit]"
pointer: "0.1910828242 [contest-CPU] UNMOVED"
pointer_delta: 0
paid_spend_usd: 0
seed: 1234
main_landing_review_required: true
---

# Autoencoders for the v10 direct-description line

## Ranked verdicts

| Rank | Candidate or control | Verdict | Exact reason and next consumer |
|---:|---|---|---|
| 1 | Direct exact coding controls for S2 and PDW2 | **ADOPT(#557 coder rung, measured $0 control rows)** | Fresh real-payload zlib-9 gives S2 `180,196 -> 152,160 B` and PDW2 `138 -> 111 B`, both bit-exact. PDW2 is 22 B below its measured 133 B Brotli-q11 control. These are stream controls, not archive deltas or score rows. |
| 2 | Tiny counted shared autoencoder over the S2 literal component stream | **N-A-WHY** | S2 has the largest live headroom. Against the new 152,160 B exact control, even an impossible zero-latent fp16 decoder must have at most 76,079 parameters. Exit: a seed-1234 n600 parser-consumed row reporting counted decoder, latent/residual bytes, exact-stream error, and through-scorer impact if lossy. `verdict_scope=FORMULATION x CURRENT-S2-LITERAL-PACKET`; stream-AE family open. Route measurement ownership to `direct_description_minimizer_builder`. |
| 3 | COIN++/VC-INR shared pretrained base decoder plus per-stream modulations | **ROUTE-TO-U4** | Under strict rule 118 the shared base-network weights count; under the alternative external-corpus generic-decoder reading only video-derived modulations/state count. This arm does not decide the boundary. Any free-decoder build remains `research_only` until #604 U4 rules. |
| 4 | Current `pose_from_embedding.py` model as an S1 replacement | **N-A-WHY** | Source-derived 4,782 parameters imply 9,564 raw fp16 weight bytes before headers, while the exact PNTG control is 6,791 B. Its previous holdout pose result was weak and no current serialized/through-R row exists. Exit: a total counted model plus latent/residual stream below 6,791 B and a real n600 frame-receiver scorer receipt. `verdict_scope=INSTANCE x CURRENT-POSE-MLP`; smaller/shared stream AEs open. |
| 5 | Deep Dict/DZip-style learned transform or neural context model over S1, PPCS, or future #574 residuals | **N-A-WHY** | The mechanisms are technically relevant, but no same-object candidate beats PNTG 6,791 B, PPCS zlib-9 78,969 B, or the #557/#558 direct coders. Exit: consume #574's custodied residuals rather than learning a second temporal predictor, then report the full six-field byte/error tuple and through-scorer loss. `verdict_scope=FORMULATION x NO-SAME-OBJECT-RECEIPT`; families open. |
| 6 | Autoencoder/codebook over standalone PDW2 | **ALREADY-HAVE-BETTER** | Exact zlib-9 is 111 B. With a zero-byte latent, a standalone raw-fp16 decoder could have at most 55 parameters; any actual latent lowers that ceiling. Exit: amortize an already-counted shared decoder over S2/PPCS and show incremental PDW2 bytes below 111 with RGB through-R custody. `verdict_scope=FORMULATION x STANDALONE-PDW2`; shared-decoder family open. |
| 7 | VQ-VAE, Ballé/Minnen/ELIC/CompressAI, Cool-Chic/C3, HNeRV/PR95, NVRC/NNCodec/DeepCABAC, REC/RECOMBINER/LotteryCodec | **ALREADY-HAVE-BETTER / ALREADY-HARVESTED** | Existing #97/#115/#152/#461/#557/#558 receipts already cover these mechanisms or a stronger in-tree control. Do not reopen frame-AE vehicles or repeat their settled measurements. |

No autoencoder candidate earns score promotion in this pass. The two `ADOPT` rows are exact,
lossless **control-rung** improvements only; neither is a receiver-closed archive marginal.

## Byte law and live-stream ceilings

Rule 118 makes all learned or video-derived decoder weights counted archive bytes. For a candidate
on the same exact stream,

`net_delta_bytes = decoder_counted_bytes + stream_bytes_after - best_exact_control_bytes`

and admission requires `net_delta_bytes < 0`. The byte score slope is
`lambda_B = 25 / 37,545,489 = 6.658589531221714e-7 score/B`. These ceilings assume raw fp16
decoder weights, zero latent/residual bytes, and zero container overhead, so they are deliberately
optimistic rather than candidate measurements.

| Stream | Live exact object | Best exact control | Maximum raw-fp16 decoder parameters at zero latent | Full control rate opportunity |
|---|---:|---:|---:|---:|
| S1 PNTG n600 | 6,791 B | 6,791 B PNTG fp16+zlib-9 | 3,395 | 0.004521848150653 |
| S2 R3 literal components | 180,196 B | **152,160 B fresh zlib-9** | 76,079 | 0.101317098307070 |
| PPCS B2 loose | 884,872 B | 78,969 B zlib-9 | 39,484 | 0.052582215669105 |
| PDW2 | 138 B | **111 B fresh zlib-9** | 55 | 0.000073910343797 |

The S2 raw per-class split remains useful for architecture sizing but is not a separable compressed
byte attribution: Road 131,565 B, Lane 8,162 B, Undrivable 39,679 B, MyCar 790 B, Movable 0 B.
Its raw zero-payload fp16 ceilings are respectively 65,782, 4,080, 19,839, 394, and 0 parameters.
A candidate must still beat the 152,160 B whole-object control and must not add the per-class raw
counts as if they were independent archive sections.

## A1 — tiny counted stream autoencoders

### S2 should be the first bounded AE probe

S2 is the only listed stream with enough current byte headroom to make a small decoder plausible
without assuming a free pretrained prior. The suitable shape is a shared tiny decoder over typed
component subsequences, with class/section conditioning and an explicit residual escape stream.
It must reconstruct the exact typed packet or decompose every lossy error through the real RGB
receiver and frozen scorer. A reconstruction MSE on numeric fields is not an admission metric.

The future probe must compare, on the identical S2 object, against zlib-9 152,160 B and whatever
#557 can produce with context arithmetic/block-FP. It must include decoder weights, codebook,
normalization constants, entropy state, latent table, section lengths, and parser overhead. A model
whose neural transform improves a proxy but loses total counted bytes is dominated.

### S1 and pose

The canonical PNTG stream is the exact serialized object: 6,791 B, stream SHA-256
`ccea4eb96d177a10e46ee56773290d6b158dd8fdec0e298560ca2f4454020014`, sourced from the n600
cache SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
It is frozen PoseNet six-vector target storage, not automatically a translation-first SE(3) xi
stream. The earlier “7.2 KB to 1–2 KB pose” shorthand is not a current receiver-closed row.

`PoseFromEmbeddingMLP` plus its mask feature extractor contains 4,782 source-derived parameters.
Raw fp16 weights are therefore 9,564 B, 2,773 B above the complete PNTG stream before any latent,
header, or residual. This costs 0.001846426877008 more rate score than PNTG even if the rest were
free. Compression/quantization of those weights could change the instance result, which is why the
negative is not a family kill. The measured n96 holdout result in the pose-path memo was also weak:
dimension-0 RMSE 1.067 against target standard deviation 1.019, with the realized frame path still
owed.

### PPCS and #574

PPCS is already 78,969 B under zlib-9 from 884,872 raw B. Its typed constraints dominate the raw
payload; a generic dense AE is likely to waste capacity on zeros and grammar. A useful candidate
would operate section-wise and preserve exact symbols, or would prove every quantization change
through the receiver. This arm polled branch `codexwt/xi_temporal_delta_coder_574_20260721T214553Z`
at commit `74324e81fa`; it contained no #574 measurement receipt. Therefore no temporal predictor
was duplicated. Any later xi AE row must consume #574's residual stream as its input.

### PDW2

The 138 B packet is too small for a standalone learned decoder. Fresh zlib-9 produces 111 B
bit-exact, smaller than the existing 133 B Brotli-q11 control. The packet still has no counted
spatial generator or scorer-free RGB pullback, so this is a packet control only. A shared decoder
could revisit PDW2 only after its weights are already counted by a larger S2/PPCS win.

## A2 — externally pretrained decoder boundary, deliberately unresolved

| Technical shape | Strict rule-118 reading | Alternative generic-decoder reading | Route |
|---|---|---|---|
| COIN++ meta-learned base coordinate network plus per-object modulation vector | Base weights, modulation, quantizer/entropy state, and syntax all count. | Base weights might be treated as external generic algorithm state; modulation and all contest-derived state still count. | #604 U4; free-reading experiments `research_only`. |
| VC-INR shared INR with latent soft gating/subnetwork selection | Shared INR/gating parameters and per-stream modulations count. | Only the externally trained fixed prior might be free; selected gates/modulations remain counted. | #604 U4. |
| External-corpus VAE/foundation codec decoder | Entire decoder asset counts unless U4 establishes a legal runtime boundary. | Fixed non-contest decoder could be free, but model identity/hash/runtime availability must be deterministic; every contest-derived latent, adaptation, codebook, normalization, and entropy table counts. | #604 U4. |
| Pre-fitted entropy prior such as FNLIC's lightweight autoregressive component | Prior bytes count under the strict reading; fitted per-object latent/probability state counts under both. | Fixed external prior may be free; fitted state never is. | #604 U4 plus #557 baseline race. |

Both readings must preserve exact deterministic decoder availability on the contest runtime. The
alternative reading is not an authorization to hide a video-derived model in `inflate.py`.

## A3 — research and OSS delta after existing #152/#461/#558 harvests

| Work | Mechanism relevant to description streams | Repository disposition |
|---|---|---|
| [COIN++](https://openreview.net/forum?id=NXB0rEM2Tq), [code](https://github.com/EmilienDupont/coinpp) | Meta-learned shared coordinate network; each datum is encoded by quantized, entropy-coded latent modulations. | New crosswalk delta, but A2-sensitive; ROUTE-TO-U4. Candidate application is low-dimensional typed sections, not frames. |
| [VC-INR](https://proceedings.mlr.press/v202/schwarz23a.html) | Shared INR plus input-dependent soft gating creates subnetworks; modulation vectors are rate-distortion coded. | New crosswalk delta, A2-sensitive. Strict accounting counts the shared network. |
| [Deep Dict](https://arxiv.org/abs/2401.10396) | Bernoulli-Transformer autoencoder latent, quantized residual, and entropy coding; quantization-entropy loss directly prices residual entropy. The paper explicitly includes the code, decoder, and encoded residual in storage and notes decoder overhead on small data. | Objective form worth retaining; no measured same-object win. N-A-WHY until an S2 or #574-residual receipt exists. |
| [DZip](https://arxiv.org/abs/1911.03572), [code](https://github.com/mohit1997/Dzip-torch) | Sequential neural prediction plus arithmetic coding, with adaptive or semi-adaptive training. | Mechanism is closer to #557 than to a transform AE. Model/state bytes and synchronized adaptation must be counted. ALREADY-HAVE-BETTER until it beats the same-object direct controls. |
| [Accelerating Relative Entropy Coding with Space Partitioning](https://openreview.net/forum?id=OuQYWNuNxm) | NeurIPS 2024 REC partitions the sampling space to make communication relative to a shared proposal more practical. | New-to-#558 crosswalk delta. It could encode fitted decoder/modulation samples, but shared-prior custody is A2-sensitive and no same-object bytes exist. N-A-WHY plus ROUTE-TO-U4. |
| [Universal Sample Coding](https://openreview.net/forum?id=qdV1vp1AtL) | NeurIPS 2024 channel simulation communicates samples relative to shared source-model access. | New-to-#558 theory delta. U4 must first rule shared-model availability; then a real S2/#574 sample-coding receipt must beat direct coding. |
| [Depth-limited A* channel-simulation analysis](https://openreview.net/forum?id=Hq07uannyG) | Fixed-runtime approximate A* coding trades proposal work against total-variation error. | New-to-#558 caution, not an adopt. Approximation error is a lossy stream change and therefore requires through-scorer decomposition. |
| [FNLIC](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Fitted_Neural_Lossless_Image_Compression_CVPR_2025_paper.html) | Per-image fitted latent probability model assisted by a pre-fitted lightweight autoregressive prior. | Already covered by #558; A2 boundary remains for the fixed prior. No duplicate build. |
| [LotteryCodec](https://proceedings.mlr.press/v267/wu25e.html) and REC/RECOMBINER-style sample communication | Communicate a compact selection/sample relative to a shared prior rather than literal learned weights. | Already harvested by #558. Shared-prior boundary goes to U4; no current description-stream receipt. |
| [Structured-data autoencoder theory](https://arxiv.org/abs/2402.05013) | Shows that shallow training can fail to exploit sparse structure and that nonlinearity/depth can matter. | Formulation caution only. It prevents treating the C6 instance miss as a family kill; it supplies no byte or scorer claim. |

The 2024–26 harvest is not empty: #558 already evaluated or classified NVRC/NVRC++, NeuroQuant,
NNCodec/DeepCABAC, COMBINER/RECOMBINER, LotteryCodec, C3/Cool-Chic, WaLLoC, FNLIC, and related
implicit/self-compression families. Its best receiver-complete measured DeepCABAC configuration was
85,274 B versus the canonical 83,838 B archive, a 1,436 B loss; that exact instance remains settled.

## Exact local control measurement

No model was trained. A read-only Python 3.14.6 / zlib 1.2.12 command loaded the three live payloads,
ran `zlib.compress(data, level=9)`, verified `zlib.decompress(output) == data`, and hashed both sides.
The receipt records the exact command semantics, paths, hashes, and structured nulls for missing
Brotli support (`ModuleNotFoundError`). This pass created no payload or scratch files.

## Consumer routing

| Consumer | Same-turn route |
|---|---|
| `direct_description_minimizer_builder` | Own the first bounded S2 shared-decoder probe. Gate on total counted bytes below the 152,160 B exact control and through-scorer decomposition for any lossy field. |
| `#557 coder rung` | Add the exact S2 zlib-9 152,160 B and PDW2 zlib-9 111 B controls; run context arithmetic/block-FP on the identical typed objects before admitting any AE. |
| `#604 U4 / einstein_kolmogorov` | Adjudicate the external-corpus generic-decoder boundary for COIN++/VC-INR/foundation-codec priors. This memo supplies both byte readings and makes no ruling. |
| `xi_temporal_delta_coder_574` | No duplicate predictor. Publish a custodied residual-stream receipt; later AE/coder rows consume that residual object. |
| `canonical_equations` | Candidate law only: `decoder_counted_bytes + stream_bytes_after < best_exact_control_bytes`, evaluator `net_delta_bytes < 0`. Do not register until a real candidate and parser/receiver receipt exist. |

**NOTE TO MAIN:** relay these exact rows and named debts to the five consumers above during branch
landing. This note routes evidence; it does not authorize a dispatch, build, or U4 decision.

## Triality and six hooks

- **DSL:** no Lever, flag, trainer, or curriculum change. A future codec must be a typed,
  parser-consumed description section with additive resume compatibility.
- **DAG:** live stream -> best exact direct coder -> counted AE fork -> exact parseback -> through-R
  scorer if lossy -> total-byte gate -> consumer. The U4 fork precedes any free-decoder branch.
- **Equations:** the admissibility inequality above is DERIVED and unregistered; all candidate
  performance fields remain null until measured.
- **Sensitivity:** reuse existing per-class/stratum allocation signals; no new sensitivity constant.
- **Pareto:** exact stream bytes first; if lossy, admit only with measured Seg/Pose score change and
  rate delta on the same receiver object.
- **Bit allocator:** optimistic parameter ceilings are preflight refusals, not budgets to fill.
- **Cathedral/autopilot:** no dispatch edge; the DAG only routes future measurement ownership.
- **Continual learning:** the two new lossless controls are durable evidence; no AE posterior claim.
- **Probe disambiguator:** U4 owns the strict-versus-generic decoder reading; ship neither as an
  implicit assumption.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and the direct-description PRIMARY spec.
- `reports/latest.md`; lane, subagent-progress, task, modal, cost-band, continual-learning,
  master-gradient, probe-outcome, and canonical-equation stores.
- `.omx/research/wrong_levels_describe_sweep_610_20260721T220916Z*` and live S1/S2 custody.
- `.omx/research/seed_compose_b2_measurements_20260721.json` and live PPCS B2 bytes.
- `.omx/research/pdw2_spatial_receiver_576_blocker_receipt_20260719.json` and live PDW2 bytes.
- `.omx/research/arith_selfcomp_rate_coders_20260719_codex*` (#557).
- `.omx/research/neural_selfcomp_sota_20260719_codex*` (#558).
- `.omx/research/levelset_pose_supervised_path_20260627T072947Z.md` and
  `src/tac/pose_from_embedding.py`.
- Settled C6 IBPS, Cool-Chic/C3, HNeRV/PR95, VQ-VAE/#461, and #152 rows by repository search.
- #574 branch/commit receipt poll; no measurement receipt was present.
- Primary paper/project pages linked in the A3 table.

## Authority boundary

Research only; $0; no training, scorer run, archive mutation, provider dispatch, equation
registration, or score promotion occurred. Retired archives were used only as settled harvest
signals. MAIN must review the complete branch diff before landing. Pointer **0.1910828242
[contest-CPU] UNMOVED**.
