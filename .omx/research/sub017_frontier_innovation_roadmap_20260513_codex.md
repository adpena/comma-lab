# Sub-0.17 frontier innovation roadmap (2026-05-13)

Status: planning + implementation-routing artifact
Score claim: false
Evidence axes: keep [contest-CUDA], [contest-CPU], and proxy/advisory signals separate.

## Operator correction

The frontier push cannot be reduced to more byte shaving. At the current HNeRV
CUDA anchor, one KiB is only about `0.00068184` score. Moving an HLM1-like
packet from `0.20638` to below `0.17` by rate alone would require about
`54.6 KiB` of charged archive savings at unchanged components, which is not a
credible short-term generic-compressor path. The right next move is model and
pipeline innovation first, with structured byte work amplifying a better
representation.

This roadmap ranks the non-local-minimum routes surfaced by the parallel
adversarial research/review passes. It is intentionally contest-constrained:
every path must end in a byte-closed archive/runtime packet, exact auth eval,
and preserved custody before it changes the frontier.

## Coverage matrix

This research pass did include the requested axes, but the implementation state
is uneven:

| Axis | Covered by this pass | Frontier implication | Immediate implementation gap |
| --- | --- | --- | --- |
| NeRV/HNeRV family | PR95/PR100/PR101/PR103/PR106 parity, HNeRV/HiNeRV/FFNeRV/E-NeRV, sane-HNeRV trainer discipline; see [NeRV-2021], [HNeRV-2023], [E-NeRV-2022] | highest near-term probability because public frontier already proved the family | recover real public training/export recipes and make exact-eval packet selection happen in-loop |
| Stacking/composition | HNeRV + LAPose, HNeRV + hyperprior, HNeRV + residual atom, PR106 sidecar stacks, A1 + wavelet/LAPose; see [Balle-2018], [CoolChic-2023], [CTW-1995] | sub-0.17 likely needs combined component and rate movement | typed stack contracts plus exact stacked archive eval; component positives are not composable until the stacked packet scores |
| Compiler/PacketIR | HDM5 q-streams, section grammars, PR101/PR103/PR106 PacketIR, arithmetic/range/ANS transforms; see [CTW-1995], [EZW-1993], [EBCOT-2000] | generic recompression is saturated; semantic payload lowering remains live | runtime decoder for best q-stream candidate and golden vectors tied to exact archive bytes |
| Bit-level first principles | byte anatomy, entropy floors, rate term arithmetic, per-section SHA/offset accounting; see [SlepianWolf-1973], [WynerZiv-1976], [MDL-1978] | rate-only from 0.20638 to sub-0.17 would need about 54.6 KiB, so bits must amplify a better model | convert entropy estimates into runtime-consumed packets, not standalone blobs |
| Mathematical score analysis | score formula, component marginals, CPU/CUDA axis separation, no-proxy authority | target must jointly move SegNet/PoseNet/rate; CPU wins do not rank CUDA | explicit per-candidate KKT/water-fill table tied to exact component traces |
| SIREN/INR | SIREN, FINER, WIRE, BACON, COIN++, NIF, Fourier-feature MLPs; see [SIREN-2020], [FINER-2024], [WIRE-2023], [BACON-2021], [COINPP-2022] | naked SIREN is lower EV than SIREN-like residual atoms over HNeRV, but SIREN-family replacement substrates remain high-upside if exact trainers mature | scorer-sensitivity atom selector and FINER/WIRE modes behind same sidecar schema |
| Non-NeRV | Ballé/CompressAI, Cool-Chic/C3, VQ-VAE, grayscale LUT, Self-Compress, procedural/ego-motion models; see [Balle-2018], [CompressAI-2020], [VQVAE-2017], [CoolChic-2023] | needed as escape routes if HNeRV saturates and as stack stages around HNeRV | real trainers, archive grammar, recipes, and exact packet paths before dispatch |
| Domain/hardware/scorer exploitation | dashcam ego-motion, road-plane/foveation, LAPose, SegNet stride-2, FastViT/PoseNet, CUDA/CPU drift | model should exploit the known video/scorer/hardware contract, not generic video compression | no-op/consumption proof for pose/foveation trailers and exact CUDA canaries |
| Proof/non-arbitrariness | kill criteria, no-op controls, exact-eval gates, probe-disambiguator pattern; GEPA/Muon can propose/train candidates but cannot claim score; see [Muon-Repo], [ModdedNanoGPT-Repo], [GEPA-2026], [GEPA-2025] | choices stay empirical and byte-closed instead of preference-driven | ship competing modes behind callable interfaces and let exact packets arbitrate |

The main undercovered item is not theory; it is forensic recovery of the public
HNeRV training pipeline. The research has enough breadth. The next score-lowering
work should turn that breadth into one of three concrete packets: a parity
HNeRV retrain/export, a runtime-consumed HDM5 PacketIR recode, or a scorer-aware
residual atom sidecar over the current HNeRV frontier.

## Paper and repo citation anchors

These anchors are execution constraints, not score claims. A linked paper or
repo can justify a model family, optimizer, or byte coder, but the Pact frontier
changes only after a byte-closed packet scores under the correct auth-eval axis.

### NeRV/HNeRV and replacement-substrate sources

- [NeRV-2021] Hao Chen et al., "NeRV: Neural Representations for Videos",
  NeurIPS 2021 / arXiv: https://arxiv.org/abs/2110.13903; repo:
  https://github.com/haochen-rye/NeRV
- [HNeRV-2023] Hao Chen et al., "HNeRV: A Hybrid Neural Representation for
  Videos", arXiv: https://arxiv.org/abs/2304.02633; project:
  https://haochen-rye.github.io/HNeRV; repo:
  https://github.com/haochen-rye/HNeRV
- [E-NeRV-2022] Zizhang Li et al., "E-NeRV: Expedite Neural Video
  Representation with Disentangled Spatial-Temporal Context", arXiv:
  https://arxiv.org/abs/2207.08132; repo:
  https://github.com/kyleleey/E-NeRV

### SIREN/INR-family sources

- [SIREN-2020] Vincent Sitzmann et al., "Implicit Neural Representations with
  Periodic Activation Functions", NeurIPS 2020 / arXiv:
  https://arxiv.org/abs/2006.09661; repo: https://github.com/vsitzmann/siren
- [FINER-2024] Zhen Liu et al., "FINER: Flexible Spectral-bias Tuning in
  Implicit Neural Representation by Variable-periodic Activation Functions",
  CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_FINER_Flexible_Spectral-bias_Tuning_in_Implicit_NEural_Representation_by_Variable-periodic_CVPR_2024_paper.pdf;
  repo: https://github.com/liuzhen0212/FINER
- [WIRE-2023] Vishwanath Saragadam et al., "WIRE: Wavelet Implicit Neural
  Representations", CVPR/arXiv: https://arxiv.org/abs/2301.05187; project:
  https://vishwa91.github.io/wire
- [BACON-2021] David B. Lindell et al., "BACON: Band-limited Coordinate
  Networks for Multiscale Scene Representation", arXiv:
  https://arxiv.org/abs/2112.04645
- [COINPP-2022] Emilien Dupont et al., "COIN++: Neural Compression Across
  Modalities", arXiv: https://arxiv.org/abs/2201.12904

### Learned compression and entropy-model sources

- [Balle-2018] Johannes Balle et al., "Variational image compression with a
  scale hyperprior", ICLR 2018 / arXiv: https://arxiv.org/abs/1802.01436
- [CompressAI-2020] Jean Begaint et al., "CompressAI: a PyTorch library and
  evaluation platform for end-to-end compression research", arXiv:
  https://arxiv.org/abs/2011.03029; docs:
  https://interdigitalinc.github.io/CompressAI/
- [VQVAE-2017] Aaron van den Oord et al., "Neural Discrete Representation
  Learning", NeurIPS 2017 / arXiv: https://arxiv.org/abs/1711.00937
- [CoolChic-2023] Theo Ladune et al., "COOL-CHIC: Coordinate-based Low
  Complexity Hierarchical Image Codec", ICCV 2023:
  https://openaccess.thecvf.com/content/ICCV2023/html/Ladune_COOL-CHIC_Coordinate-based_Low_Complexity_Hierarchical_Image_Codec_ICCV_2023_paper.html;
  related overfitted neural codec: https://arxiv.org/abs/2307.12706; project:
  https://orange-opensource.github.io/Cool-Chic/

### Classical source-coding and bitstream sources

- [SlepianWolf-1973] David Slepian and Jack K. Wolf, "Noiseless Coding of
  Correlated Information Sources", IEEE Transactions on Information Theory;
  reference copy: https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf
- [WynerZiv-1976] Aaron D. Wyner and Jacob Ziv, "The Rate-Distortion Function
  for Source Coding with Side Information at the Decoder", IEEE Transactions
  on Information Theory; reference copy:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
- [CTW-1995] Frans Willems, Yuri Shtarkov, and Tjalling Tjalkens, "The
  context-tree weighting method: basic properties", IEEE Transactions on
  Information Theory, DOI: https://doi.org/10.1109/18.382012
- [EZW-1993] Jerome M. Shapiro, "Embedded Image Coding Using Zerotrees of
  Wavelet Coefficients", IEEE Transactions on Signal Processing, DOI:
  https://doi.org/10.1109/78.258085
- [EBCOT-2000] David Taubman, "High performance scalable image compression
  with EBCOT", IEEE Transactions on Image Processing, DOI:
  https://doi.org/10.1109/83.847830
- [MDL-1978] Jorma Rissanen, "Modeling by shortest data description", Automatica
  1978, DOI: https://doi.org/10.1016/0005-1098(78)90005-5

### Optimizer/autoresearch sources

- [Muon-Repo] Keller Jordan, "Muon: An optimizer for the hidden layers of
  neural networks": https://github.com/KellerJordan/Muon
- [ModdedNanoGPT-Repo] Keller Jordan et al., "modded-nanogpt" speedrun stack:
  https://github.com/KellerJordan/modded-nanogpt
- [GEPA-2025] Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform
  Reinforcement Learning", arXiv: https://arxiv.org/abs/2507.19457
- [GEPA-2026] GEPA `optimize_anything` API docs:
  https://gepa-ai.github.io/gepa/api/optimize_anything/optimize_anything/

## Citation-to-packet mapping

- HNeRV parity and HNeRV replacement are separate lanes. [HNeRV-2023] explains
  why content-adaptive embeddings beat fixed-index INRs for video; [SIREN-2020],
  [FINER-2024], [WIRE-2023], [BACON-2021], and [COINPP-2022] define replacement
  substrate candidates that must earn their place through real trainers,
  archive grammar, and exact packets.
- Compiler/PacketIR byte work should borrow from [CTW-1995], [EZW-1993], and
  [EBCOT-2000] only when the decoder overhead and runtime-consumption proof are
  included in charged bytes.
- Ballé/CompressAI/Cool-Chic/VQ ideas become Pact-relevant only when the
  hyperprior/discrete-code/coordinate-decoder path is exported into the
  contest runtime without sidecar dependency leaks.
- Muon and GEPA are search/training accelerators, not evaluation authority:
  Muon can be tested as a hidden-layer optimizer for SIREN/FINER/HNeRV trainers;
  GEPA can propose text-serializable configs, PacketIR transforms, or atom
  schedules whose evaluator returns only byte-closed exact-eval or explicit
  no-claim diagnostics.

## Formal score arithmetic and decision calculus

Contest score:

```text
S = 100 * d_seg + sqrt(10 * d_pose) + 25 * B / 37,545,489
```

For a candidate packet `c` against baseline `b` on the same evidence axis:

```text
Delta S(c,b) =
    100 * (d_seg_c - d_seg_b)
  + (sqrt(10 * d_pose_c) - sqrt(10 * d_pose_b))
  + 25 * (B_c - B_b) / 37,545,489
```

Rate slope:

```text
25 / 37,545,489 = 6.658599e-7 score / byte
                 = 0.000681840 score / KiB
```

At the PR101/A1 operating point in the macOS advisory sweep
(`d_pose ~= 3.286e-5`, axis advisory only), the local pose derivative is:

```text
d/dp sqrt(10p) = 5 / sqrt(10p) ~= 275.8 score / pose-dist unit
```

This is the quantitative reason blind byte shaving is a local minimum. A
10 KiB byte win is only about `0.00682` score before component harm. Moving
from the current public frontier band (`~0.193`) to `0.170` by rate alone
requires about `33.7 KiB` of net charged-byte savings. Moving to `0.150`
requires about `63.0 KiB`. Those are not impossible, but they are too large
for saturated ZIP/Brotli cosmetics. Model/score-component movement must lead,
and PacketIR/arithmetic work should amplify it.

Non-arbitrary decision rule:

1. Prefer an action if it can create a byte-closed exact packet, exact replay,
   runtime-consumption proof, or public-training recovery artifact.
2. Reject an action if it improves a proxy while failing to define how bytes
   enter `inflate.sh archive_dir output_dir file_list`.
3. Split any contested design into callable modes and let exact packets or
   no-op-controlled component traces arbitrate.
4. Promote only same-axis evidence: `[contest-CUDA]` against `[contest-CUDA]`,
   `[contest-CPU]` against `[contest-CPU]`; advisory CPU is a routing signal
   only.

## Testable hypotheses

These are the paper/lab-grade units of work. Each must end in an artifact, not
only a conclusion.

| ID | Hypothesis | Mechanism | Exact artifact required | Falsification gate |
| --- | --- | --- | --- | --- |
| H1 | Public HNeRV won because content-adaptive embeddings plus export discipline match the one-video contest contract | Recover PR95/PR100/PR101/PR103 training/export schedules and score-domain checkpoint selection | trainer parity report, runtime-frame parity, exact packet, auth eval JSON | recovered trainer cannot reproduce a same-axis PR101/PR103 component neighborhood |
| H2 | HNeRV can be replaced, not only stacked | SIREN/FINER/WIRE/BACON/COIN++ full-renderer trainers with contest archive grammar | `train_substrate_*`, recipe, runtime decoder, archive manifest, exact smoke | full-renderer replacement stays above HNeRV comparator with no byte-floor advantage |
| H3 | SIREN-family bases are higher EV as sparse scorer-aware atoms than as naked full-video renderers | Select INR atoms by SegNet boundary and PoseNet hard-pair sensitivity per byte | no-op-controlled sidecar packet, raw-output hash delta, component deltas | predicted local component win disappears under exact CUDA |
| H4 | Semantic PacketIR recoding can beat generic compression | Typed q-streams, CTW/range/ANS coding, deterministic decoder overhead included | parse -> emit identity proof, section SHA/offset ledger, runtime-consumption mutation proof | decoder overhead or metadata consumes the entropy gain |
| H5 | Dashcam physics gives cheaper pose/scorer satisfaction than generic video reconstruction | FOE, horizon, road-plane, tau/time-to-contact, LAPose foveation/motion atoms | pose/foveation trailer consumed by runtime, component-response memo | no CUDA PoseNet/SegNet improvement after no-op-controlled consumption proof |
| H6 | Hyperprior/discrete-code stages are useful around an overfit renderer | Ballé/CompressAI/Cool-Chic/VQ style latent grammar around HNeRV/INR output | compressed latent packet with closed decoder and no external sidecar deps | float/proxy gain disappears after integer entropy path export |
| H7 | Muon improves sample efficiency enough to change dispatch economics | Orthogonalized updates for 2D hidden weights in HNeRV/SIREN/FINER trainers | paired seed/trainer run with identical archive export and exact smoke | same budget produces no score-domain validation gain or destabilizes QAT/export |
| H8 | GEPA/autoresearch can improve pipeline choices without becoming proxy authority | Text-serializable configs/transforms proposed by LLM search, evaluated by local exact/no-claim evaluator | candidate ledger with evaluator outputs and rejected unsafe proposals | proposed changes cannot pass preflight, exact packet construction, or no-claim discipline |

## Experimental design and artifact schema

Every experiment attached to this roadmap should emit a small manifest with:

- `lane_id`, `hypothesis_id`, git SHA, dirty diff SHA, author/source URL when
  public work is used, and license/compliance note;
- baseline archive path, bytes, SHA-256, runtime tree SHA, inflate command, and
  evaluator command;
- candidate archive path, bytes, SHA-256, runtime tree SHA, generated-file
  manifest, and exact packet construction command;
- evidence axis: one of `[contest-CUDA]`, `[contest-CPU]`,
  `[macOS-CPU advisory]`, or `proxy/no-score`;
- `score_claim`, `promotion_eligible`, `ready_for_exact_eval_dispatch`, and
  explicit reason when any is false;
- component fields: `seg_dist`, `pose_dist`, byte term, recomputed total, sample
  count, raw-output aggregate hash if available;
- no-op or mutation proof for every sidecar/PacketIR/residual path;
- reactivation criteria for every negative result.

The minimum journal-grade table for a result row is:

```text
method | baseline | axis | bytes | delta_bytes | seg | delta_seg |
pose | delta_pose | total | delta_total | packet_sha | runtime_sha |
claim_grade | failure_or_promotion_reason
```

The minimum lab-grade directory layout is:

```text
experiments/results/<lane_id>/
  manifest.json
  command.txt
  logs/
  archive.zip
  archive.sha256
  runtime_tree.sha256
  contest_auth_eval.json
  inflated_outputs_manifest.json
  no_op_or_mutation_proof.json
  component_delta_review.md
```

When an experiment is research-only, replace `contest_auth_eval.json` with a
`no_score_evaluator.json` that explicitly says why no score claim is allowed.

## Replacement-substrate protocol

The operator explicitly wants paths that can replace HNeRV. Treat these as
first-class lanes, not afterthoughts:

1. Full-renderer SIREN/FINER/WIRE/BACON/COIN++ replacement:
   coordinates include at least `(frame_index, x, y)`; any domain variables
   such as FOE, horizon, foveation weight, or pair hardness must be encoded in
   the charged packet or derived deterministically from runtime-available data.
2. Overfit neural-codec replacement:
   Ballé/CompressAI/Cool-Chic/VQ variants must export a closed decoder and
   entropy model. A library checkpoint is not an archive grammar.
3. Classical+learned replacement:
   motion-compensated prediction, wavelet/bitplane residuals, CTW/range coding,
   and tiny neural correction blocks must share one typed packet contract.
4. Domain-specific replacement:
   road-plane/ego-motion/foveation models must prove they change runtime RGB
   and lower actual components, not only explain the video semantically.

Replacement lanes must include a byte-floor estimate before dispatch:

```text
estimated_total_bytes =
    decoder_code_bytes
  + model_or_latent_bytes
  + entropy_metadata_bytes
  + per-frame_or_per-pair_side_info_bytes
  + ZIP/container overhead
```

If the estimated byte floor cannot plausibly fit below the HNeRV comparator
while improving components, the lane stays research-only until a new mechanism
changes the floor.

## Threats to validity

- Public frontier artifacts may encode undocumented training/export choices.
  Final archives alone cannot identify the causal mechanism.
- macOS CPU sweeps are routing signals only; ARM CPU, Linux CPU, T4 CUDA, and
  Modal A100 CUDA can differ materially.
- INR papers usually optimize PSNR/L2 or image fitting; Pact optimizes a known
  SegNet/PoseNet/rate formula. Literature rank is not Pact rank.
- Optimizer speedrun results are not direct evidence of contest score movement;
  Muon or nanoGPT-derived tricks must win inside the exact archive exporter.
- GEPA can overfit the evaluator wrapper or produce noncompliant shortcuts.
  Every proposal needs deterministic preflight and contest-compliance review.
- Byte coders can look excellent on isolated streams and lose once decoder
  overhead, framing, checksums, manifests, and runtime code are charged.
- Sidecar/residual work is especially vulnerable to no-op illusions. Mutation
  tests and raw-output hashes are mandatory.
- A component gain can be total-score negative because pose, segmentation, and
  rate trade at nonlinear local slopes.

## Review checklist before any sub-0.19 claim

- Same-axis baseline and candidate are named and replayable.
- Archive bytes and SHA-256 are measured from the scored archive, not a member.
- Formula score is recomputed from components and bytes.
- Runtime consumes every claimed payload section.
- `inflate.sh` does not load scorer models, network resources, or private paths.
- CPU/CUDA distinction appears next to every frontier, medal-band, or promotion
  phrase.
- Exact negative results are classified and preserved.
- The result changes a lane status only after adversarial engineering, math,
  scorer-geometry, and compliance review.

## Unknown-unknown risk register

These are the places where prior evidence can be locally true and still mislead
the solver:

1. Archive-selection mismatch: validation Lagrangian picks a checkpoint that
   loses after parse-pack-inflate-auth-eval. Probe: select candidates by parsed
   archive replay on a small schedule before full dispatch.
2. CPU/CUDA sign flip: a pose or interpolation change wins on CPU and loses on
   CUDA, or vice versa. Probe: keep paired raw-output hashes and never merge the
   axes in ranking logic.
3. Runtime-consumption illusion: a sidecar, residual, or recode is present in
   bytes but not consumed by `inflate.py`. Probe: no-op mutation tests plus
   old/new full-frame aggregate hashes.
4. Proxy-byte false floor: a standalone compressed blob beats entropy targets
   but the decoder, metadata, or grammar needed to consume it erases the gain.
   Probe: require runtime-consumed PacketIR candidates before score projection.
5. Component antagonism: a SegNet improvement harms PoseNet enough that total
   score rises because of the square-root pose term. Probe: every planner row
   carries component deltas, not just total or pixel loss.
6. Scorer-preprocess drift: training path and auth eval differ in resize,
   clamp, quantization, YUV6, or dtype. Probe: differentiable preprocessing
   guard plus exported archive replay samples.
7. Public-PR overfitting lesson misread: the public HNeRV jump may depend on
   an optimizer/curriculum/export detail rather than the visible model. Probe:
   recover training scripts and ablate one primitive at a time.
8. Byte-mass target wrong: the biggest section may not be the highest
   marginal section once component harm is included. Probe: water-fill over
   exact component traces and section-level byte anatomy together.
9. Hidden dependency/compliance trap: a great packet relies on non-closed
   packages, network install, scorer import at inflate, or device-specific
   hidden state. Probe: pre-submission compliance and no-scorer-in-inflate
   guards before exact spend.
10. Local-minimum distraction: a green guard or tiny byte win feels productive
    but does not change any path below 0.19. Probe: require every action to
    connect to a candidate packet, exact replay, high-byte semantic section,
    public-pipeline recovery, or scorer-targeted component delta.

## Underconsidered cross-domain literature backlog

These are not replacements for HNeRV parity. They are candidate sources of
missing primitives for the stack-of-stacks compiler.

1. Ecological optics and driving control: Gibson optic flow, Lee tau / time to
   collision, and visually guided braking. Pact translation: encode the video
   in driver-relevant invariants: focus of expansion, road-plane flow, horizon,
   time-to-contact, and foveated central roadway atoms. This belongs in LAPose
   and pose/foveation residual sidecars.
2. Classical optical flow and registration: Horn-Schunck, Lucas-Kanade,
   phase correlation, and structure-from-motion. Pact translation: cheap
   deterministic motion fields or per-pair alignment codes that explain most
   frame change before a neural residual. This is a non-NeRV escape hatch and
   a HNeRV sidecar candidate.
3. Photogrammetry / projective geometry: Ullman structure-from-motion,
   Marr-Poggio correspondence, Longuet-Higgins essential matrix. Pact
   translation: camera-motion and road-plane priors as a six-degree pose code
   rather than pixel residual bytes.
4. Universal coding and MDL/MML: Solomonoff, Kolmogorov, Rissanen MDL,
   Wallace-Boulton MML. Pact translation: treat the packet as a shortest
   program for the exact scorer/video pair; choose model families by total
   message length, not by architecture fashion.
5. Russian/Dutch universal source coding: Shtarkov normalized maximum
   likelihood and Willems/Shtarkov/Tjalkens context-tree weighting. Pact
   translation: CTW / context-tree switching over decoder byte maps, latent
   deltas, and PacketIR symbols before inventing bespoke neural entropy
   coders.
6. Distributed/source coding with side information: Slepian-Wolf and
   Wyner-Ziv. Pact translation: decoder has the previous frame and deterministic
   renderer state; sidecars should code residuals assuming decoder-side
   prediction, not as independent images.
7. Wavelet bitplane families: Shapiro EZW, Said-Pearlman SPIHT, Taubman EBCOT.
   Pact translation: scorer-aware embedded residual streams where truncation is
   byte-budget optimal and every prefix is a valid exact-eval candidate.
8. Trellis and vector quantization: Linde-Buzo-Gray VQ, Marcellin-Fischer
   trellis-coded quantization, finite-state VQ. Pact translation: train
   quantizers for decoder weights/latents under scorer-aware distortion, not
   post-hoc scalar rounding.
9. Fractal / iterated-function-system compression: Barnsley/Jacquin/PIFS.
   Pact translation: road/sky/lane regions may be self-similar under affine
   transforms; this is a possible procedural residual grammar with tiny decoder
   bytes, not a generic full-image codec.
10. Old low-bitrate telepresence/video coding: H.261/H.263 era
    motion-compensated DPCM/DCT and optical-flow motion compensation. Pact
    translation: the single-video dashcam stream may prefer an explicit
    motion-compensated residual compiler feeding HNeRV/C3, rather than a fully
    implicit neural renderer.

Immediate research-to-build conversion:

- Add a `cross_domain_primitives` queue where each row must name the exact
  archive stage it could improve: representation, prediction, quantization,
  hyperprior/context model, arithmetic, or pack.
- Prioritize primitives that can be tested as no-op-controllable sidecars over
  PR106/HLM1: phase-correlation shifts, tau/FOE pose features, CTW PacketIR
  symbol coding, and embedded wavelet residual prefixes.
- Treat non-English/older sources as implementation leads only after their
  primitive can be expressed as a byte-closed Python module with tests and
  exact-eval custody.

## Eureka-pattern hypotheses

These are the beautiful/simple algorithms that could plausibly beat another
incremental HNeRV refire if the contest video has the right hidden structure.
Each one must become a packet experiment or die.

1. Focus-of-expansion codec. A dashcam video is not arbitrary video: most
   motion radiates from a vanishing point plus ego-vehicle pitch/yaw. Encode
   FOE, horizon, road-plane scale, and a small residual instead of full latent
   vectors. Test as a LAPose/PR106 trailer with per-pair FOE parameters and
   exact CUDA PoseNet deltas.
2. Time-to-contact atoms. Lee tau says looming can be represented by a simple
   ratio rather than dense optical flow. Test whether object/road expansion
   fields explain PoseNet-sensitive frame differences with tens of bytes per
   segment.
3. Phase-correlation micro-shifts. Per-frame subpixel translation/scale may buy
   scorer movement for almost no bytes. Test a finite table of dy/dx/scale
   runtime constants against PR106/HLM1, with no-op and raw-output hashes.
4. Embedded scorer-bitstream. EZW/SPIHT/EBCOT style: emit residual bits in
   exact order of expected score gain. Every prefix is an exact-eval candidate.
   This is a better shape than arbitrary top-k residual sidecars.
5. Context-tree PacketIR coder. CTW is elegant because it mixes all suffix
   contexts without hand-picking one. Test CTW/range coding on PacketIR decoder
   symbols and latent deltas, including decoder overhead in the charged packet.
6. Program-plus-patches MDL. Treat the archive as the shortest Python program
   that fools SegNet/PoseNet on this video, then patches only the failure
   pairs. This reframes SIREN/HNeRV as subroutines in a minimum-message-length
   program.
7. Fractal road/sky/lane grammar. Road texture, lane markings, sky, and
   shadows may be affine/self-similar enough for PIFS-style block transforms.
   Test only as region-scoped residuals; full-frame fractal compression is too
   broad.
8. Cooperative correspondence cells. Marr-Poggio/Ullman/Longuet-Higgins style:
   encode sparse correspondences and recover the scene/renders by enforcing
   consistency, rather than storing dense pixels. Test for road boundaries,
   lane lines, horizon, and parked/moving vehicle contours.
9. Trellis quantized decoder weights. TCQ/VQ can exploit sequence constraints
   better than scalar quantization. Test trained trellis/rank-aware QAT for the
   HNeRV decoder, not post-hoc destructive rounding.
10. Two-speed model. Use a tiny analytic motion/geometry model for the bulk of
    frames and a neural residual only where the scorer sees novelty. Test via a
    hard-pair schedule from component traces: ordinary pairs get procedural
    bytes; hard pairs get neural atoms.
11. Boundary-only semantic renderer. SegNet loss is concentrated near class
    boundaries. Render only the structures that move argmax boundaries, let
    texture be cheap or wrong where the scorer is insensitive. Test with
    boundary marginals and exact SegNet component deltas.
12. CUDA-as-target compiler. Runtime dtype/interpolation/sigmoid/clamp choices
    are legal program constants. Treat CUDA math as the target ISA and compile
    the packet to that numerical behavior, with CPU as a separate diagnostic.

Eureka filter: a beautiful idea only matters here if it gives one of:

- fewer charged bytes for the same runtime frames;
- lower SegNet/PoseNet components at similar bytes;
- a stack primitive that composes with HNeRV without proxy authority;
- a clearer mathematical floor or kill proof.

## Priority 1: PR95/PR101/HNeRV parity retrain

Mechanism: recover and canonicalize the public HNeRV training discipline into a
reusable trainer: eval-roundtrip export, differentiable scorer preprocessing,
score-domain validation, EMA, QAT, and entropy-aware decoder/latent grammar.

Why it can move below 0.17: this attacks the dominant problem directly rather
than post-hoc packing. HNeRV-family public packets won because the model class
matched the one-video scorer/video contract: compact temporal representation,
small decoder, small latent stream, and acceptable SegNet/PoseNet distortion.
The missing leverage is not another blind refire; it is the exact training and
export loop that produced the public jump below 0.25.

First concrete work:
- complete forensic recovery of PR95/PR100/PR101/PR103/PR106 training scripts,
  optimizer schedules, loss weights, export code, and archive grammars;
- add parity guards that compare train-time decoded frames against
  runtime-consumed frames;
- require score-domain validation to select the runtime packet, not just a
  checkpoint proxy.

Kill condition: a byte-closed exact CUDA packet using the recovered discipline
cannot beat the HLM1/PR106 CUDA comparator after custody review.

## Priority 2: HDM5 / PacketIR structured decoder and latent grammar

Mechanism: replace high-byte opaque Brotli sections with typed q-streams,
shared codebooks, context models, and deterministic runtime decoders.

Why it matters: generic Brotli is close to saturation on the current bytestream,
but the payload still has semantic structure. This work should target decoder
weights and latent/sidecar grammar, not ZIP cosmetics.

First concrete work:
- implement runtime decode for the best non-self-describing q-stream candidate;
- preserve old/new section SHA, section offsets, exact charged bytes, and
  runtime consumption proof;
- keep all candidate packets `score_claim=false` until exact CUDA.

Kill condition: metadata/runtime overhead eats the entropy floor, or
full-frame/runtime parity breaks for a rate-only transform.

## Priority 3: scorer-aware residual atom compiler

Mechanism: use SIREN, FINER, WIRE, BACON, C3, wavelet, and related bases as
sparse residual atoms over the current frontier decode, selected by scorer
sensitivity per byte.

Why SIREN still matters: pure coordinate-SIREN-as-full-video is not the best
imminent spend. SIREN-style bases become higher value when used as targeted
score-aware residuals over PR106/HLM1: small payload, explicit no-op controls,
and selection against SegNet boundary or PoseNet hard-pair sensitivity rather
than L2 energy alone.

First concrete work:
- feed cached scorer sensitivity maps into
  `tools/materialize_siren_residual_pr106_sidecar.py`;
- compare SIREN/FINER/WIRE/wavelet atoms under one byte-closed manifest schema;
- exact-eval only candidates that show real component-targeted predicted
  delta per byte and pass no-op mutation proof.

Kill condition: L2/local ranking looks good but exact CUDA components regress
or the sidecar is not consumed by the runtime path.

## Priority 4: HiNeRV / FFNeRV / non-NeRV full-renderer fanout

Mechanism: test content-adaptive, frequency, flow, hierarchical, and codec
substrates as full RGB renderers with declared archive grammar before training.

Why it matters: the public frontier says the winning model class is not a
generic image/video compressor. The next step is to make alternate
representation classes exportable and exact-evaluable, not to dispatch L0
scaffolds.

First concrete work:
- wire real trainers, recipes, archive grammars, and exact-eval packet paths
  before any SIREN/Balle/Cool-Chic/VQ spend;
- prioritize HiNeRV/FFNeRV and Cool-Chic/C3 because they encode temporal and
  entropy structure, not just per-coordinate memorization.

Kill condition: exact CUDA anchor stays at or above the current HNeRV comparator
with no component win and no byte-floor advantage.

## Priority 5: dashcam-domain pose/geometry exploitation

Mechanism: integrate LAPose, ego-motion priors, road-plane/foveation atoms, and
legal runtime numerical choices into the exact-evaluable stack.

Why it matters: the video is one dashcam stream and the scorer is known.
Physical vehicle motion, forward flow, horizon/road-plane geometry, and
scorer-specific blind spots are part of the problem definition. They should
shape the model, not sit as disconnected tools.

First concrete work:
- build an A1/HLM1 plus LAPose no-op mutation proof and consumption proof;
- run a small exact CUDA canary only after the trailer changes RGB in the
  runtime path;
- keep CPU/CUDA drift as a measured axis, never as interchangeable evidence.

Kill condition: no CUDA PoseNet or SegNet improvement, or the pose trailer is
not consumed.

## Priority 6: HNeRV plus hyperprior / overfit codec stack

Mechanism: combine representation, prediction, quantization, hyperprior,
arithmetic coding, and pack passes as one stack, not isolated lanes.

Why it matters: sub-0.17 likely requires combined rate and component movement.
Ballé/CompressAI/Cool-Chic/C3 ideas should be treated as stack stages around a
contest-specific overfit renderer, not as generic library demos.

First concrete work:
- formalize typed contracts:
  `representation -> prediction -> quantization -> hyperprior -> arithmetic -> pack`;
- add small conformance vectors and no-scorer-in-inflate guards;
- only dispatch once runtime grammar and exact archive construction exist.

Kill condition: exported integer/entropy path destroys float gains or adds
contest-noncompliant dependencies.

## Priority 7: public-frontier forensic recovery

Mechanism: mine public PRs and author repositories for the training pipeline,
not just final inflate artifacts.

Why it matters: the archive tells us what ran; the training scripts explain why
it worked. Recovering optimizer, curriculum, score-loss, and export details is
higher information gain than another local micro-ablation.

First concrete work:
- queue PR95/PR100/PR101/PR103/PR106 source, writeup, and linked-repo intake;
- extract exact runnable primitives only after URL, commit, and license/status
  are recorded;
- keep detached clones out of the shared worktree.

Kill condition: only prose/proxy evidence exists and no runnable primitive or
exact replayable training artifact is recoverable.

## Anti-local-minimum guard

Before spending time on a small patch, answer yes to at least one:

- Does it make a candidate archive/runtime packet or exact replay more likely?
- Does it recover public-frontier training or archive mechanics?
- Does it reduce a known high-byte semantic section, not just container noise?
- Does it wire a substrate into the canonical exact-eval path?
- Does it expose a scorer-aware component target that can beat 0.19?

If all answers are no, it is probably local-minimum work.
