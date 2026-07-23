# Codex findings — DDM SN1 telemetry and error-source asymmetry

Date: 2026-07-23  
Lane: `ddm_sn1_segnet_telemetry_asymmetry`  
Axis: `[macOS-CPU frozen-SegNet+PoseNet advisory]`  
Verdict: `MEASURED`, `research_only=true`, `score_claim=false`,
`promotion_eligible=false`, pointer unmoved  
Landing: isolated Codex worktree; MAIN review required

## Disposition

The delegated premise is confirmed in the measured scope: sided SegNet
decision distance is asymmetric, and the current v19c residual is not one
undifferentiated paint failure. The landing supplies an optional no-op
telemetry path, a strict sided-tolerance contract, a bounded three-segment
receiver inverse demonstration, an exhaustive n600 source tensor, and
scorer-native SegNet/PoseNet relay products.

Accepted error-tensor receipt:
`.omx/research/ddm_sn1_error_source_tensor_n600_20260723/ddm_sn1_error_source_tensor_receipt.json`,
SHA-256
`ecf9f015fa6999b9bb7602c93027da713bb278389b92d5d1bf0b95f4ced19faa`.
A finalize-only replay from the complete certified checkpoint tree reproduced
that file byte-for-byte.

## Stores consulted

All joins were consumed by content identity. The accepted receipt carries the
complete path/byte/SHA table; the principal decision inputs were:

| Store | SHA-256 | Authority used |
|---|---|---|
| v19c strict endpoint receipt | `506fb1dfed849beb06358d3a30d624fa8cbdad3c6e0da6cf1bf1ec14960472ae` | current 2,265,811-error state |
| DV1 receipt / selected payload | `e0c875346978f8768eedf96793bcff2fb472a37eba4ebbcec7254d107fc00333` / `03897224767612de0dc37c355710252a9eb418ff8a960c6cbd99276fcc4e76b3` | tested semantic extension and 1,610 bytes |
| G2 receipt / aggregate | `ada87717b39bc34ad67a3104d652574e544d82938fa3a1ea898acdf624c2bd67` / `061220fd8c1ca047b210841235fc805194a96175e933ee110ba4ac8bb2077d84` | class-level marginal only |
| G3 n600 atlas | `faaff7299d86aa49c97e25e9cce2eeb0201f64e919f110015d31708788bcec09` | pair/tail/event covariates |
| G4 recurrence arrays | `dbc85e7a4f593ab9b7a7f4ed017dbb63a064cb681df806d0bb93277ae8f42451` | historical recurrence only |
| v14 realization receipt | `82d3249908d42a86575c407ab3d7acdf9b3706b31225f2e46862b2472966e5a9` | named realization-leak cross-check |
| e1 runtime receipt | `69d19eee47c0435868d806947850d4f6099d9f2be1e35c399795816272fd54ba` | exporter/receiver-survival cross-check |
| DR2 receipt | `4e3cec44f9176342642b363d67e64b3ff92b010f01c879c8f4e44f00839f8ce7` | per-record constancy correction |
| official video / GT cache | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` / `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` | n600 frozen inputs |
| frozen SegNet / PoseNet | `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` / `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576` | scorer-native measurements |

## Measured findings

The exact 2,265,811-error v19c partition closes:

| Source | Errors | Share |
|---|---:|---:|
| `DESCRIBED_BUT_REALIZATION_LOST` | 892,710 | 39.3991% |
| `NEVER_DESCRIBED` | 738,090 | 32.5751% |
| `STRUCTURALLY_HARD_IRREDUCIBLE` | 635,011 | 28.0258% |

The third label is scoped to the current semantic program plus the one
SHA-pinned DV1 extension. It is not a family-impossibility claim.

Within realization loss, the observable partition is:
587,913 `COARSE_DESCRIPTION`, 208,623 `PAINT_FUNCTION`, and 96,174
`TEXTURE_PRIOR_REGION_ERF`. The highest-ranked solve-menu row is an
Undrivable→Road, no-continuous-Lane-curve, coarse-description, G3-tail
cluster. The menu contains 2,649 exhaustive rows.

The 738,090 errors newly reached by the shared 1,610-byte DV1 payload give
458.4409937888 semantic errors per shared byte. This is not receiver survival,
Pose value, exact archive value, or score-unit value per byte.

The current n600 #149 boundary wall is
1,613,214 / 4,684,236 = 0.34439212712596035. The historical mp128
three-frame contextual reference is 0.1605960279317129, a ratio of
2.1444622981111303. Different receivers, sample scopes, and evidence axes make
this context rather than a promotion comparison.

SN1 ordered-pair evidence is genuinely sided. Road→Lane q10 is
0.024886758 versus Lane→Road 0.019704731, a 1.26298 ratio. The strongest
observed reversal is Undrivable→Lane versus Lane→Undrivable, with absolute
q10 difference 0.0323151 and ratio 1.33648. MyCar↔Undrivable has no measured
boundary support and is typed as no support; no zero is invented.

Record-level constancy is resolved from the SHA-pinned DR2 exact record
inventory, not from recurring pixel coordinates. Each Road, Lane,
Undrivable, Movable, and MyCar cell record; each of the five corresponding
separatrix records; and the Pose pair-screw record has 600 unique states and
599 adjacent changes. Whole-record static coding is therefore inadmissible for
all eleven current records. G4's pixel recurrence remains only an upper bound
and a historical covariate; it does not contradict the record-level result or
rule out persistent primitives plus sparse innovations.

All three bounded inverse demonstrations realized the requested majority
transition, with collateral flips retained rather than hidden:
MyCar→Road 374, Road→MyCar 231, and Lane→Road 99.

## Scorer-native findings

SegNet divergence begins and peaks at `encoder.model.conv_stem`, gap RMS
111.3470599787. At that peak, geometry residual fraction is 0.9672467208 and
uniform-shift fraction is 0.0327532792. The strongest BN mean gap is
`encoder.model.bn1` at RMS 19.1553856306; the strongest SE gate gap is
`encoder.model.blocks.4.1.se.gate` at RMS 0.2049723522. The top advisory relay
is `decoder.blocks.4`.

PoseNet divergence begins at `vision.stem` and peaks at
`hydra.final_layer.pose`, gap RMS 9.1016619674. The strongest LayerScale gap
is `vision.stages.3.blocks.1.layer_scale` at RMS 0.8972970677; the strongest
BN gap is `summarizer.2.block_b.5.bn` at RMS 3.6903425396; the model's sole
SE gate is `vision.final_conv.se.gate`. The top advisory relay is
`vision.head`.

PoseNet's actual frozen amplitude inventory is 24 `LayerScale2d`,
8 `BatchNorm1d`, 19 `GELUTanh`, and one `SEModule`. The first-six-pair output
MSE is 163.0612073286 and cache replay max absolute error is
`1.1444091796875e-5`; both are advisory diagnostics, not official
\(d_{\rm pose}\).

Every-layer hooks execute in model topology order. The products retain channel
moments, 8×8 pooled native-space energy, within/across-frame and across-pair
contrasts, temporal spectra, spatial frequency bands, deterministic stable
rank, and directional secants. Raw full-video activation tensors are not
persisted. No intermediate Fisher pullback or full Jacobian spectrum is
claimed.

Frozen convolution DFTs and phase-indexed resize kernels are exact for their
recorded operators. A global bicubic-up → uint8 → bilinear-down scalar transfer
is refused because the chain is phase/border dependent and piecewise affine at
uint8.

## Adversarial defects found and extincted

1. A first scorer-native build produced stable-rank overflow warnings. The
   estimator now uses deterministic torch float64 power steps, normalizes at
   each half-step, and refuses non-finite intermediates.
2. The first build retained about 4.9 GiB of checkpoint bulk locally.
   Checkpoints are now SSD-first, losslessly certified, and source-linked.
3. Resume finalization initially trusted JSON dictionary order, which sorted
   layer names alphabetically. Finalization now reconstructs topology from the
   stored integer `order`, with a regression test.
4. `timm`'s gate class was not `torch.nn.Sigmoid`, so the sole PoseNet SE gate
   was initially missed. Class-name-aware gate detection and an exact inventory
   assertion prevent recurrence.
5. Measurement and finalizer implementation identities are recorded
   separately. Finalization refuses an incomplete measurement checkpoint tree.

## Exact limits and next action

No contest CPU/CUDA score was run. No archive was produced or mutated. No
remote/GPU job was launched. The exact next rung is a separately claimed,
receiver-closed probe against the highest-mass solve-menu cluster with same-
candidate frozen SegNet, official PoseNet, exact archive-byte, and parse-back
custody. This memo does not authorize that execution.
