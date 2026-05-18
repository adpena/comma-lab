# RATE-ATTACK-NOVEL-VECTORS-DEEP-RESEARCH-20260518

Status: `research_only=true`; no score claim; no dispatch; no `.omx/state` mutation.
Supersedes/extends: `.omx/research/rate_attack_novel_vectors_design_memo_20260518.md`.
Operator scope: write only `.omx/research`; do not stage, commit, push, or mutate state.
Axis discipline: every band below is `prediction_only`; `[contest-CUDA]`, `[contest-CPU]`, and `[macOS-CPU advisory]` remain separate evidence spaces.

## Canonical preflight readback

- Branch/source of truth: current checkout is `main`; dirty partner work exists outside this write scope and was not touched.
- Frontier context from `reports/latest.md`: current preserved public/frontier surfaces are split by axis. `[contest-CPU GHA Linux x86_64]` best is PR101-derived; `[contest-CUDA T4]` best is PR106 format0-derived. No CPU/CUDA conversion is valid.
- Relevant prior rate memo: original 13 vectors A/B/C/META already normalized. This memo keeps those IDs and adds YUV category D `Y1-Y7` plus hardware-codec category E `H1-H9`.
- Relevant 20260518 constraints:
  - Q4/Wyner-Ziv-on-existing-archive-members is mostly saturated: validated contest archives show compression ratios near 1.0; post-hoc recompression is low-EV unless it changes pre-entropy representation.
  - Master-gradient evidence is high-signal but currently narrow: fec6-only advisory gradients cannot be generalized to every frontier archive without extraction.
  - YUV6/eval-roundtrip is a first-class constraint: PoseNet consumes two frames as 12 channels, i.e. two 6-channel BT.601-style YUV420 stacks. Upstream `rgb_to_yuv6` had a `no_grad` path; the differentiable scorer patch is the right training substrate.
  - Hardware paths are rate-relevant only when they shrink charged archive bytes. NVDEC, NVJPEG, DALI, NVENC, tensor cores, and sparse formats are otherwise wall-clock/search enablers, not score improvements.
- Pre-registration blocker: AGENTS.md requires L0 lane pre-registration, but the operator explicitly forbids `.omx/state` mutation. This memo records that blocker and remains `research_only=true` until a state-writing turn claims lanes.

## Primary sources and external references

- Challenge/context: [comma.ai leaderboard](https://comma.ai/leaderboard), [commaai/commavq](https://github.com/commaai/commavq).
- Wyner-Ziv source coding with decoder side information: [Wyner and Ziv, 1976 PDF](https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf).
- Practical embedding/channel coding: [Filler, Judas, Fridrich STC paper PDF](https://dde.binghamton.edu/filler/pdf/Fill10tifs-stc.pdf), [IEEE DOI](https://doi.org/10.1109/TIFS.2011.2134094).
- JPEG/DCT syntax and table overhead: [ITU-T T.81 / ISO/IEC 10918-1 PDF](https://www.w3.org/Graphics/JPEG/itu-t81.pdf).
- BT.601 luma/chroma basis used by the local YUV6 scorer path: [ITU-R BT.601 recommendation page](https://www.itu.int/rec/R-REC-BT.601/), [BT.601-7 PDF](https://www.itu.int/dms_pubrec/itu-r/rec/bt/r-rec-bt.601-7-201103-i%21%21pdf-e.pdf).
- NVIDIA hardware codec/runtime facts: [NVIDIA video encode/decode support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix), [Video Codec SDK NVENC guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.2/nvenc-video-encoder-api-prog-guide/index.html), [DALI video reader example](https://docs.nvidia.com/deeplearning/dali/archives/dali_1_44_0/user-guide/examples/sequence_processing/video/video_reader_simple_example.html), [nvJPEG docs](https://docs.nvidia.com/deeplearning/nvjpeg/index.html).
- Modern video grammar references: [AV1 bitstream specification PDF](https://aomediacodec.github.io/av1-spec/av1-spec.pdf), [AOM libaom](https://aomedia.googlesource.com/aom/), [VideoLAN dav1d](https://code.videolan.org/videolan/dav1d), [x265](https://bitbucket.org/multicoreware/x265_git), [Fraunhofer VVenC](https://github.com/fraunhoferhhi/vvenc).
- Adjacent steganographic context only, not contest authority: [Yassine Yousfi steganography notes](https://yassineyousfi.github.io/pages/steganography.html).

## Executive ranking

### TOP-5 rate vectors

1. `Y1+Y2+Y4+Y7`: YUV6-native scorer-domain representation and sublattice allocator.
   Rationale: PoseNet already consumes BT.601-style YUV420-derived channels; RGB-first rendering spends entropy on a representation the pose scorer immediately remaps. The route attacks pre-entropy representation instead of saturated ZIP/Brotli residue. `prediction_only DeltaS=[-0.016,-0.004]`, high evidence fit, medium implementation risk.

2. `A1+B3+M3`: stable-argmax orbit packet diet with STC/Brotli-aware allocation and Fisher/byte gradients.
   Rationale: master-gradient/xray/sensitive-byte surfaces say some bytes matter far more than others. Rate should be removed where scorer gradients and class-boundary risk are low, not uniformly. `prediction_only DeltaS=[-0.014,-0.003]`, high evidence fit, low-to-medium implementation risk after multi-archive gradient extraction.

3. `Y3+M1`: seed-matched chroma/per-class codebook with Wyner-Ziv framing.
   Rationale: decoder-side determinism is compliant when generated only from archive-shipped seeds or shipped weights; per-class chroma/palette material is a plausible 1-8 KB payload target. This is a targeted pre-entropy replacement, not generic recompression. `prediction_only DeltaS=[-0.008,-0.001]`, medium evidence fit, low implementation cost.

4. `M2+Y1+Y2`: tropical SegNet boundary grammar in YUV/luma subspaces.
   Rationale: SegNet argmax surfaces and luma boundary sublattices are the right place to encode masks if the packet can avoid RGB over-specification. The risk is brittle boundary flips and inflated decoder complexity. `prediction_only DeltaS=[-0.012,-0.004]`, medium evidence fit, medium risk.

5. `H1+H2+H9`: HEVC/H.264 deterministic bitstream payload probe with hardware determinism harness.
   Rationale: hardware-codec-as-byte-derivation is underexplored and can be a real rate substrate only if the charged archive stores a compact bitstream whose deterministic decode replaces a larger neural/residual payload. NVDEC/DALI/NVENC alone are not rate gains. `prediction_only DeltaS=[-0.010,-0.002]`, high upside, high compliance/runtime risk.

Close runner-up: `B1+B2` decoy/mosaic residual swarm remains high upside (`prediction_only DeltaS=[-0.018,-0.006]`) but is ranked below `H1/H2/H9` for this memo because its premise is less uniquely informed by the new YUV/hardware research mandate.

### TOP-3 routing candidates

1. `RATE-ROUTE-YUV6-SUBLATTICE-20260518`
   Route vectors: `Y1,Y2,Y4,Y7` plus `A1/M3` as allocator priors.
   Canonical consumer: cathedral autopilot should request a timing smoke that reports Y/U/V byte budgets, PoseNet deltas, SegNet deltas, luma-sublattice entropy, and full-frame output parity status.

2. `RATE-ROUTE-MG-STABLE-ORBIT-PACKET-DIET-20260518`
   Route vectors: `A1,B3,M3,C1,C2` plus master-gradient/xray/sensitive-byte intersections.
   Canonical consumer: cathedral autopilot should require multi-archive gradient extraction before any retirement/promotion.

3. `RATE-ROUTE-HEVC-NVDEC-BITSTREAM-PROBE-20260518`
   Route vectors: `H1,H2,H4,H7,H9` plus `Y5`.
   Canonical consumer: cathedral autopilot should treat this as a fail-closed probe: score-relevant only if archive bytes shrink and deterministic full-frame decode survives `[contest-CUDA]` runtime closure.

## 29-vector table

| ID | Vector | Mechanism | Highest-EV test | Dykstra feasibility intersection | `prediction_only` DeltaS band | Route |
|---|---|---|---|---|---:|---|
| A1 | SABOR / stable-argmax boundary orbit | Move pixels/latents along scorer-stable argmax orbits; spend bytes only near real boundaries. | Derive stable-orbit candidate from differentiable scorer, then mutate only low-risk bytes from sensitivity map. | `C_score_stability AND C_rate AND C_full_frame` | `[-0.010,-0.002]` | TOP-2 |
| A2 | S2SBS / high-frequency stride blindspots | Encode perceptual/high-frequency detail where scorer stride/downsample paths are least sensitive. | Pair xray stride blindspots with luma sublattice Y2. | `C_stride_alias AND C_pose_yuv AND C_seg` | `[-0.006,-0.001]` | support |
| A3 | Continuous-curvature operating-point sweep | Sweep local rate/distortion curvature instead of scalar quality knobs. | Build cheap local Taylor sweep around current archive packet families. | `C_local_taylor AND C_pareto AND C_allocator` | `[-0.005,-0.001]` | support |
| B1 | Decoy rendering | Render cheap decoy content where scorers are indifferent. | Test decoy planes under hard-pair and boundary xray masks. | `C_seg_boundary AND C_pose_indifference AND C_decode_budget` | `[-0.012,-0.003]` | runner-up |
| B2 | Mosaic encoder swarm | Tile/substrate-specific encoders chosen per region/frame/pair. | Autopilot routes hard pairs to heavy substrate, easy regions to cheap substrate. | `C_tile_pack AND C_header_overhead AND C_pareto` | `[-0.010,-0.002]` | runner-up |
| B3 | STC/Brotli-incompressible-aware allocation | Treat residual bytes as cover objects with per-byte distortion costs; use STC-like embedding/allocation. | Run STC-style allocator on sensitivity-ranked payload sections, count charged overhead. | `C_stc_cover AND C_brotli_entropy AND C_sensitive_bytes` | `[-0.009,-0.002]` | TOP-2 |
| B4 | Cheap-prototype discipline | Timing smoke before expensive campaigns; avoids overfitting slow lanes. | Require seconds/candidate and byte impact estimate for every route. | `C_wallclock AND C_cost AND C_autopilot` | `[0.000,0.000] process` | all |
| C1 | ACH-driven ranking | Rank by archive/change/harvest evidence, not prose plausibility. | Add ACH fields to route memos and autopilot intake. | `C_evidence AND C_rank AND C_custody` | `[0.000,0.000] process` | all |
| C2 | Key-assumption matrix | Make each premise falsifiable before implementation. | Per-route blocker table: rate, score, runtime, compliance. | `C_premise AND C_probe AND C_fail_closed` | `[0.000,0.000] process` | all |
| C3 | Rotating devil's advocate | Adversarial review of every claimed route. | Each directive includes a kill criterion and cargo-cult trap. | `C_review AND C_probe AND C_operator_budget` | `[0.000,0.000] process` | all |
| M1 | Brotli plus cooperative receiver dictionary | Generated/shipped dictionaries or deterministic codebooks reduce residual entropy. | Seed-derived codebook/chroma palette, not external dictionary. | `C_seeded_dict AND C_no_external_state AND C_rate` | `[-0.006,-0.001]` | TOP-3 support |
| M2 | Tropical SegNet argmax encoding | Encode mask boundaries/classes via max-plus/argmax grammar. | Boundary grammar in luma/YUV planes; compare exact SegNet deltas. | `C_argmax AND C_boundary AND C_decoder_simple` | `[-0.012,-0.004]` | TOP-4 |
| M3 | Per-bit Fisher-weighted FP4 lattice | Quantize bytes/tensors using scorer-gradient/Fisher weights. | Multi-archive master-gradient; allocate bits where gradient is steep. | `C_gradient AND C_quant_lattice AND C_archive_bytes` | `[-0.010,-0.002]` | TOP-2 |
| Y1 | YUV6-native scorer-domain representation | Train/encode in the exact PoseNet YUV6 channel space instead of RGB-first. | Emit YUV6-native latent/render pipeline with full-frame RGB/video output parity and exact scorer replay. | `C_pose_yuv6 AND C_full_frame AND C_rate` | `[-0.016,-0.004]` | TOP-1 |
| Y2 | Luma quartet sublattice grammar | Encode `y00,y10,y01,y11` sublattices as separate lower-entropy planes. | Compare entropy and scorer deltas for per-sublattice payloads; watch bilinear/resize aliasing. | `C_luma_sublattice AND C_stride_alias AND C_seg` | `[-0.010,-0.002]` | TOP-1 |
| Y3 | Chroma U/V seed palette | Replace shipped chroma/per-class tables with archive-seeded or weight-derived codebooks. | Seed-match current palette, then ablate pure uniform vs optimized seed. | `C_wyner_ziv_seed AND C_chroma_low_rank AND C_compliance` | `[-0.008,-0.001]` | TOP-3 support |
| Y4 | PoseNet YUV channel-weight allocator | Allocate bits by Y/U/V and frame-pair channel sensitivity. | Derive per-channel Fisher weights from differentiable scorer; feed allocator. | `C_pose_channel_fisher AND C_allocator AND C_pair` | `[-0.008,-0.002]` | TOP-1 |
| Y5 | YUV420 hardware bitstream bridge | Store compact 4:2:0 video bitstream; deterministic decode at inflate. | HEVC/H.264 bitstream replacing larger residual/neural payload; exact CUDA T4 runtime closure. | `C_codec_bitstream AND C_yuv420 AND C_hardware_determinism` | `[-0.010,-0.002]` | TOP-3 |
| Y6 | Pairwise YUV delta/residual | Encode inter-frame luma motion/residual plus slow chroma residual. | PoseNet two-frame path smoke: pair residual entropy vs pose delta. | `C_pair_motion AND C_pose_yuv6 AND C_entropy` | `[-0.007,-0.001]` | support |
| Y7 | YUV eval-roundtrip guard | Prevent zero-gradient/axis mismatch false negatives. | Enforce differentiable YUV6 in training and exact upstream YUV6 at eval. | `C_gradient_validity AND C_axis AND C_full_frame` | `[0.000,0.000] guard` | TOP-1 |
| H1 | HEVC/H.264 deterministic payload decode | Charged archive stores codec bitstream; inflate decodes deterministic frames. | Minimal I/P GOP HEVC/H.264 probe on T4, compare archive bytes and exact scorer deltas. | `C_hevc_h264 AND C_cuda_runtime AND C_archive_shrink` | `[-0.010,-0.002]` | TOP-3 |
| H2 | NVENC compress-time search | Use hardware encoder to search rate-control/ROI/emphasis maps; final packet may be software-decodable. | NVENC/QP/ROI timing smoke; final archive can use x265/ffmpeg if smaller and compliant. | `C_encoder_search AND C_rate_control AND C_runtime_independence` | `[-0.006,-0.001]` | TOP-3 support |
| H3 | NVJPEG/JPEG DCT carrier | Encode pose/mask detail in DCT coefficients or quantization/stego carrier. | JPEG table/header byte accounting plus STC carrier on DCT AC coefficients. | `C_dct AND C_header_charged AND C_stc_cover` | `[-0.006,-0.001]` | support |
| H4 | DALI/NVDEC loader acceleration | Faster GPU decode and axis xray, not direct rate. | Fill CPU/PyAV vs CUDA/DALI xray matrix; use only to accelerate search/eval. | `C_wallclock AND C_axis_xray AND C_no_score_claim` | `[0.000,0.000] speed` | TOP-3 support |
| H5 | Tensor-core / FP4 / NF4 / block quant | Smaller tensor payloads if serialized format changes; compute-only kernels do not count. | FP4/NF4/GGUF-like byte grammar with exact inflate reconstruction. | `C_quant_payload AND C_kernel_optional AND C_archive_bytes` | `[-0.009,-0.002]` | support |
| H6 | Sparse formats / 2:4 / CSR / BSR | Sparse grammar may save bytes; T4 has no Ampere 2:4 tensor-core advantage. | Sparse byte grammar on actual tensors; reject pure hardware-speed claims. | `C_sparse_payload AND C_rate AND C_t4_reality` | `[-0.005,-0.001]` | support |
| H7 | AV1/HEVC/VVC OBU/NAL microcontainer | Use standardized bitstream grammar as payload container or source of entropy coders. | AV1/HEVC/VVC parser footprint and runtime closure; T4 AV1/VVC decode not assumed. | `C_bitstream_grammar AND C_dependency_closure AND C_t4_support` | `[-0.006,-0.001]` | TOP-3 support |
| H8 | GPU deterministic seed expander | Archive stores seed; CUDA kernel expands codebook/residual deterministically. | Seed-to-codebook generator with CPU fallback; no hidden driver randomness. | `C_seed_generator AND C_determinism AND C_no_external_state` | `[-0.007,-0.001]` | support |
| H9 | Hardware determinism harness | Axis-proof wrapper for codec outputs and runtime versions. | Hash decoded frames across PyAV, DALI/NVDEC, ffmpeg, nvJPEG when applicable. | `C_axis_honesty AND C_runtime_sha AND C_full_frame` | `[0.000,0.000] guard` | TOP-3 |

## YUV synthesis

YUV is not just a color-space convenience here. The scorer contract makes it structural:

- PoseNet consumes two frames as 12 channels, i.e. two 6-channel YUV420-derived stacks. That makes RGB an intermediate carrier, not the pose scorer's native information geometry.
- BT.601-style luma carries most geometric signal; U/V are spatially subsampled and likely lower entropy. A YUV-native representation can allocate bits closer to the actual pose objective.
- The luma quartet `y00,y10,y01,y11` exposes subpixel/parity structure that RGB-first residual codecs hide. This intersects with stride blindspots, bilinear resize artifacts, and boundary stability.
- SegNet still consumes RGB-like frame inputs, so YUV-native work must be evaluated against both scorers. YUV savings that degrade SegNet are not promotable without a compensating mask/boundary grammar.
- Full-frame inflate parity remains mandatory. Decoded tensor or latent parity is parser evidence, not scorer or full-frame evidence.

Practical route: train/render in YUV6 or a YUV6-adjacent latent basis, but keep the contest-facing inflate output exactly compatible with the official evaluator. The top smoke should report per-channel entropy, per-channel sensitivity, exact bytes, and component deltas, never an unlabeled aggregate.

## Hardware-codec synthesis

Hardware can help in three distinct ways, only one of which is directly rate-relevant:

1. Rate substrate: the archive stores a compact HEVC/H.264/JPEG/AV1-like bitstream or compressed tensor grammar that replaces larger payload bytes. This can change score if runtime closure and deterministic decode hold.
2. Search accelerator: NVENC/NVDEC/DALI/nvJPEG/tensor cores make sweeps cheaper. This can discover better packets but is not itself a rate gain.
3. Axis xray: hardware paths expose CPU/CUDA decode/scorer differences. This prevents false negatives and false promotions.

The NVIDIA support matrix confirms Tesla T4 is a Turing part with H.264/HEVC encode/decode support but no AV1 encode in the T4 row and no assumption of VVC decode. Therefore:

- HEVC/H.264 are the only plausible T4-native hardware video codec probes for `[contest-CUDA]` inflate.
- AV1 and VVC are still useful grammar/OSS research paths, but should not be treated as T4-native decode candidates unless the runtime ships a compliant software decoder and passes time/dependency gates.
- NVJPEG/JPEG is attractive as a DCT/stego carrier, but JPEG syntax charges quantization/Huffman tables and marker overhead. Headers are not free.
- DALI/NVDEC can reduce wall-clock and fill the CPU/CUDA xray matrix. It should not be assigned a negative DeltaS band unless archive bytes shrink.

## Compliance envelope

Hard constraints for every route:

- `inflate.sh archive_dir output_dir file_list` must not import scorer weights, call the scorer, use the official labels, or query network/external state.
- Every byte needed for decode must be in the charged archive or deterministically derived from archive bytes/runtime code that is shipped inside the compliant packet.
- Hash-seeded generators are structurally compliant only when the seed and algorithm are archive/runtime-contained. Pre-baked dataset constants in `inflate.py` are non-compliant.
- Runtime dependency closure must be explicit: package versions, runtime tree SHA, decode binary/library origin, and driver/hardware assumptions.
- Hardware codec lanes need deterministic frame-output manifests. Driver-specific decode variance must be an axis blocker, not silently averaged away.
- Score labels must carry axis tags next to every readiness word: `[contest-CUDA]`, `[contest-CPU]`, `[macOS-CPU advisory]`, proxy, or `prediction_only`.
- Post-hoc compression of already-saturated archives is not a promotion path unless it produces a measurable byte reduction above header/custody overhead and survives exact replay.
- JPEG, AV1, HEVC, VVC, STC, or arithmetic-coder control data is charged. Tables, codebooks, seeds, section lengths, manifests, and decoder glue must all be counted.

## Dykstra-feasibility intersections

Use a Dykstra-style projection mental model: a route is feasible only at the intersection of independent convex-ish constraints. When projections cycle or one constraint expands another, the route needs a probe-disambiguator rather than a prose verdict.

| Constraint | Meaning | Routes most exposed |
|---|---|---|
| `C_rate` | Archive bytes shrink after all headers/codebooks/tables. | all rate lanes |
| `C_seg` | SegNet component remains within Pareto tolerance. | `M2,Y1,Y2,Y3,H1` |
| `C_pose_yuv6` | PoseNet YUV6/two-frame score remains stable or improves. | `Y1,Y2,Y4,Y6` |
| `C_full_frame` | Official inflate outputs full frames/videos, not only latents/tensors. | `Y1,Y5,H1,H3,H7` |
| `C_runtime` | Decode fits contest runtime/dependency closure. | hardware/codecs |
| `C_hardware_determinism` | CUDA/DALI/NVDEC/nvJPEG output is reproducible and hashed. | `H1,H3,H4,H9` |
| `C_custody` | Archive SHA, runtime tree SHA, manifest, and exact axis labels exist. | all |
| `C_autopilot` | Cathedral autopilot can consume the route mechanically. | all TOP routes |
| `C_probe` | Two defensible interpretations ship as callable modes. | `Y3,H1,H3,H7` |

Highest-value intersections:

- `Y1/Y2/Y4`: `C_pose_yuv6 AND C_rate AND C_seg AND C_full_frame`. This is the strongest frontier route because it attacks representation before entropy coding.
- `A1/B3/M3`: `C_gradient AND C_sensitive_bytes AND C_brotli_entropy AND C_score_stability`. This is the strongest byte-allocation route once master-gradient coverage extends beyond fec6.
- `Y3/M1`: `C_wyner_ziv_seed AND C_no_external_state AND C_chroma_low_rank AND C_rate`. This is the fastest narrow byte probe.
- `H1/Y5/H9`: `C_codec_bitstream AND C_archive_shrink AND C_hardware_determinism AND C_runtime`. This is high upside but fail-closed until deterministic decode and dependency closure are proven.
- `H3/B3`: `C_dct AND C_stc_cover AND C_header_charged AND C_full_frame`. This route is killed if JPEG marker/table overhead dominates the carried information.

## Cargo-cult audit

| Cargo-cult claim | Why false or incomplete | Correct action |
|---|---|---|
| Hardware acceleration lowers score. | Score uses archive bytes plus scorer deltas; speed alone has zero rate term. | Tag as speed/search unless archive bytes shrink. |
| YUV is automatically lower entropy and therefore better. | SegNet still matters; RGB output/eval parity must hold; chroma errors can move pose. | Run YUV6 channel sensitivity and full-frame eval. |
| JPEG quantization tables are free knobs. | ITU T.81 syntax charges DQT/DHT/marker overhead in the packet. | Count headers/tables before claiming rate. |
| NVDEC/DALI decode is deterministic enough. | Driver/library/hardware drift can change decoded pixels. | Hash decoded outputs per axis and runtime. |
| AV1/VVC are better codecs, so they are T4 routes. | T4 does not imply native AV1/VVC decode; software decode may break runtime budget. | Treat as grammar/CPU probe until runtime closure exists. |
| STC can hide bits without cost. | The cover object is the contest archive/output; distortion costs must be scorer-aware and headers count. | Use sensitivity-weighted STC with exact component replay. |
| Post-hoc recompression still has large gains. | Current Q4 audit found existing archives essentially compressed-saturated. | Prioritize pre-entropy representation shifts. |
| fec6 master-gradient is universal. | It is one archive/axis anchor. | Extend extractor to multiple frontier archives before route retirement. |
| Process guardrails deserve DeltaS bands. | Guards reduce false authority; they do not directly move score. | Mark `process`, `guard`, or `speed`, not negative score. |

## 6-hook wire-in declarations

These are declarations for the next implementation turn; this research turn does not edit code or state.

### `RATE-ROUTE-YUV6-SUBLATTICE-20260518`

1. Sensitivity map: add per-channel `Y00/Y10/Y01/Y11/U/V` sensitivity and entropy fields.
2. Pareto constraint: forbid component promotion unless SegNet and PoseNet deltas are both axis-tagged.
3. Bit allocator: allocate bits by YUV6 channel, luma sublattice, frame-pair role, and hard-pair class.
4. Cathedral autopilot hook: canonical consumer should rank YUV6 timing smoke above post-hoc recompression.
5. Continual learning posterior: append only after exact local smoke or exact eval artifact, not from this prediction memo.
6. Probe-disambiguator: ship RGB-first baseline and YUV6-native candidate as two callable modes.

### `RATE-ROUTE-MG-STABLE-ORBIT-PACKET-DIET-20260518`

1. Sensitivity map: consume multi-archive master-gradient, xray, hard-pair, and sensitive-byte overlays.
2. Pareto constraint: stable-orbit mutations must record class-boundary and pose deltas separately.
3. Bit allocator: STC/Fisher allocator receives per-byte cost vector and Brotli incompressibility mask.
4. Cathedral autopilot hook: route is blocked until at least two frontier-family archives have gradient extraction.
5. Continual learning posterior: update on exact archive mutation outcomes, including negative/no-op outcomes.
6. Probe-disambiguator: compare uniform byte shaving, gradient-weighted shaving, and STC-style allocation.

### `RATE-ROUTE-HEVC-NVDEC-BITSTREAM-PROBE-20260518`

1. Sensitivity map: connect codec ROI/QP map to boundary/xray hard-pair masks.
2. Pareto constraint: codec output must pass component replay; no aggregate-only claims.
3. Bit allocator: count bitstream bytes, headers, runtime glue, codebooks, and fallback code.
4. Cathedral autopilot hook: route is fail-closed until deterministic decode manifest exists.
5. Continual learning posterior: update only after decoded frame hashes, runtime SHA, and archive SHA are recorded.
6. Probe-disambiguator: compare H.264, HEVC, software x265/ffmpeg, and NVENC-produced bitstreams as separate modes.

## Cathedral autopilot contract

Cathedral autopilot is the canonical consumer for this memo. It should ingest route IDs, not parse prose. The next code-bearing turn should expose a compact route manifest with these fields:

- `route_id`
- `vectors`
- `axis_target`
- `prediction_only_band`
- `required_inputs`
- `timing_smoke_command`
- `byte_accounting_outputs`
- `component_delta_outputs`
- `kill_criteria`
- `promotion_blockers`
- `next_artifact_path`

Autopilot ordering for the next no-state-mutation-independent work:

1. YUV6 route timing smoke and entropy census.
2. Multi-archive master-gradient/sensitivity byte extraction.
3. Seed-chroma/Wyner-Ziv narrow packet probe.
4. HEVC/H.264 hardware-codec deterministic decode probe.
5. JPEG/STC carrier accounting probe.

## Explicit next artifacts

These are concrete artifacts the next implementation turn should create after lane registration is allowed:

- `experiments/results/rate_attack_yuv6_sublattice_entropy_20260518/manifest.json`
- `experiments/results/rate_attack_yuv6_sublattice_entropy_20260518/channel_entropy.csv`
- `experiments/results/rate_attack_yuv6_sublattice_entropy_20260518/component_deltas.json`
- `experiments/results/master_gradient_multi_archive_rate_attack_20260518/byte_costs.parquet`
- `experiments/results/seed_chroma_wz_packet_probe_20260518/packet_manifest.json`
- `experiments/results/hevc_nvdec_determinism_probe_20260518/decoded_frame_hashes.json`
- `.omx/research/codex_findings_rate_attack_routes_<utc>_codex.md` after adversarial review of any code landing.

## Blockers and kill criteria

- State blocker: lane pre-registration and dispatch claims cannot be performed in this turn because `.omx/state` writes are prohibited.
- Evidence blocker: master-gradient evidence must not be generalized from fec6 to all frontier archives.
- Hardware blocker: T4 hardware decode support must be verified in the target runtime, not inferred from desktop docs alone.
- Codec blocker: hardware bitstream routes are killed if charged headers/runtime glue exceed payload savings or if decoded full-frame hashes drift by axis.
- YUV blocker: YUV-native routes are killed if SegNet loss dominates PoseNet/rate savings.
- JPEG/STC blocker: carrier routes are killed if quantization/Huffman/marker overhead plus induced scorer loss beats carried payload savings.

## Recommended immediate action

After an operator permits state/code writes, register three L0 lanes and run the routing directives in this order:

1. `RATE-ROUTE-YUV6-SUBLATTICE-20260518`
2. `RATE-ROUTE-MG-STABLE-ORBIT-PACKET-DIET-20260518`
3. `RATE-ROUTE-HEVC-NVDEC-BITSTREAM-PROBE-20260518`

Do not run paid GPU or remote eval from this memo alone. This memo is a source-faithful research/routing artifact and all score movement remains `prediction_only`.
