# Public Top Submission Delta Anatomy vs C-089

Generated from refreshed public PR heads and downloaded public archive URLs. No remote GPU dispatch.

Adversarial source note: PR67 body still states SHA `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd` and `276,741` bytes, but the release URL downloaded in this pass is SHA `86c8694adf8bf53a09a2f2162285601be51ae3030572c73d97f85f3db04c85b8` and `276,620` bytes. Both are tracked separately.

## Score And Byte Gaps

| archive | zip bytes | bytes vs C-089 | rate score vs C-089 | score/body score | score vs C-089 |
|---|---:|---:|---:|---:|---:|
| pr55_quantizr | 299970 | 23628 | 0.015732915 | 0.3322719955107341 | 0.016801268 |
| pr63_qpose14 | 287573 | 11231 | 0.007478262 | 0.3249617830819938 | 0.009491056 |
| pr64_unified_brotli | 287165 | 10823 | 0.007206591 | 0.33612677067610813 | 0.020656043 |
| pr65_henosis | 284425 | 8083 | 0.005382138 | 0.3196824276891214 | 0.004211700 |
| pr67_release_current | 276620 | 278 | 0.000185109 | 0.31470817503416587 | -0.000762552 |
| pr67_body_stated_cached | 276741 | 399 | 0.000265678 | 0.31470817503416587 | -0.000762552 |
| c089_frontier | 276342 | 0 | 0.000000000 | 0.3154707273953505 | 0.000000000 |

## Logical Member Deltas vs C-089

| archive | member | candidate bytes | C-089 bytes | delta | same sha |
|---|---|---:|---:|---:|---|
| pr55_quantizr | masks.mkv | 219472 | 219472 | 0 | True |
| pr55_quantizr | optimized_poses.npy | 13185 | None | None | None |
| pr55_quantizr | optimized_poses.qp1 | None | 677 | None | None |
| pr55_quantizr | renderer.bin | 66841 | 55965 | 10876 | False |
| pr55_quantizr | seg_tile_actions.bin | None | 116 | None | None |
| pr63_qpose14 | masks.mkv | 219472 | 219472 | 0 | True |
| pr63_qpose14 | optimized_poses.bin | 1160 | None | None | None |
| pr63_qpose14 | optimized_poses.qp1 | None | 677 | None | None |
| pr63_qpose14 | renderer.bin | 66841 | 55965 | 10876 | False |
| pr63_qpose14 | seg_tile_actions.bin | None | 116 | None | None |
| pr64_unified_brotli | masks.mkv | 223385 | 219472 | 3913 | False |
| pr64_unified_brotli | optimized_poses.bin | 1200 | None | None | None |
| pr64_unified_brotli | optimized_poses.qp1 | None | 677 | None | None |
| pr64_unified_brotli | renderer.bin | 91582 | 55965 | 35617 | False |
| pr64_unified_brotli | seg_tile_actions.bin | None | 116 | None | None |
| pr65_henosis | bias | 223 | None | None | None |
| pr65_henosis | frac | 106 | None | None | None |
| pr65_henosis | frac2 | 149 | None | None | None |
| pr65_henosis | frac3 | 154 | None | None | None |
| pr65_henosis | masks.mkv | 219472 | 219472 | 0 | True |
| pr65_henosis | optimized_poses | 1487 | None | None | None |
| pr65_henosis | optimized_poses.qp1 | None | 677 | None | None |
| pr65_henosis | post | 1400 | None | None | None |
| pr65_henosis | randmulti | 3731 | None | None | None |
| pr65_henosis | region | 273 | None | None | None |
| pr65_henosis | renderer.bin | 57074 | 55965 | 1109 | False |
| pr65_henosis | seg_tile_actions.bin | None | 116 | None | None |
| pr65_henosis | shift | 226 | None | None | None |
| pr67_body_stated_cached | masks.mkv | 219472 | 219472 | 0 | True |
| pr67_body_stated_cached | optimized_poses.qp1 | 899 | 677 | 222 | False |
| pr67_body_stated_cached | renderer.bin | 56034 | 55965 | 69 | False |
| pr67_body_stated_cached | seg_tile_actions.bin | 236 | 116 | 120 | False |
| pr67_release_current | masks.mkv | 219472 | 219472 | 0 | True |
| pr67_release_current | optimized_poses.qp1 | 898 | 677 | 221 | False |
| pr67_release_current | renderer.bin | 55914 | 55965 | -51 | False |
| pr67_release_current | seg_tile_actions.bin | 236 | 116 | 120 | False |

## Top 5 Concrete Deltas

1. **Recover PR67/PR75 component edge with C-089 bytes: exact pose/seg regression audit between stated cc1dfa PR67 and C-089 P6 stack** Expected score gain `0.0007`-`0.002`. C-089 is already 399 bytes smaller than the stated PR67 archive but scores about 0.000763 worse from components; sub-0.314 needs about 0.001471 total, so component recovery plus another ~1KB is plausible. Gates: raw-output parity trace on C-089 vs stated PR67 archive, component trace delta localized by pair/frame, no release-SHA drift in source archive, T4 exact eval of any stack.
2. **PR75/P6 action stream dictionary-v2 or delta-varint grammar search over seg_tile_actions.bin** Expected score gain `0.0002`-`0.0018`. C-089 action stream is 160B vs public stated 268B and current release likely changed again; only 116-268B direct rate remains, but action choices move SegNet/PoseNet more than their byte cost. Gates: decode action records and prove non-noop runtime application, local raw parity/component proxy vs PR67 records, exact T4 eval for top 1-2 variants.
3. **Renderer QZS3 self-compression/learned bit-depth transplant without zeroing pose-sensitive frame1 head** Expected score gain `0.0015`-`0.01`. Renderer remains 59,288B encoded in C-089; public PR67/QZS3 model slices are around 56KB before decode, and self-compression/block-FP lanes target multi-KB savings. Need >2.2KB byte-equivalent for sub-0.314 if components hold. Gates: preflight_trained_renderer_transplant, preflight_renderer_transplant_pose_safety, runtime tree hash custody, T4 exact eval.
4. **Henosis postfilter atom transplant as charged, typed postprocess stream, but only after strict no-op and component gates** Expected score gain `0.0005`-`0.006`. PR65 has much better PoseNet body value (0.00035283) with worse SegNet/rate. Its post/shift/frac/bias/region/randmulti atoms are scorer-targeted pixel corrections; a small selected subset may improve pose enough to clear 0.314. Gates: port as opt-in runtime member with typed manifest, prove every selected atom changes raw output, pair/frame component trace, T4 exact eval.
5. **Pose-stream active-subspace transfer: PR65 pose benefit without PR65 mask/SegNet tax** Expected score gain `0.0005`-`0.004`. PR65 pose distance is dramatically lower than C-089, but its SegNet is worse. A bounded QP1/low-dimensional pose perturbation can be <1KB and high leverage if it preserves SegNet. Gates: decode and compare pose trajectories, bounded active-subspace builder, preflight pose loader magic, T4 exact eval.
