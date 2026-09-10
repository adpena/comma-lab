# ddm_gdc3 — next construction against the measured residual-geometry law — n600 receipt, 2026-09-10

## Verdict

**FORMULATION-NO-GO** for the fixed anisotropic key-row, zero-order-hold family tested here. The best of four full-n600 schedules was `uniform2`: counted program packet 227,551 B, 634,370 mismatches, real exact residual 225,612 B, and exact total 453,163 B. It exceeds the 94,010 B generator door by 359,153 B (4.820370x). The packet alone is 2.4205x the whole door.

This is a real byte-closed negative, not a score row. The run used no scorer, no training, no Modal, and no archive construction. The exact contest frontier did not move:

`composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`

The useful positive is the geometry law. In both predecessor fields, horizontal long-run, boundary-adjacent errors were the cheapest measured residual class. The key-row falsifier did what its geometry claimed—100% of its errors were boundary-adjacent—but the field description intercept dominated.

## Authority, source, and provenance

All geometry and candidate measurements are `[macOS-CPU byte-only n600]`; `score_claim=false` throughout.

- Move-44 field: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8`, 117,964,800 B, sha256 `78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6`. Move 44 inherits this field byte-identically. Pointer commit `99625f32f`; archive sha256 `04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`.
- gdc1 K=8 receiver render: `/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/k08/receiver_render.u8`, 117,964,800 B, sha256 `abd130921beec23255112e8b56148d55c49cf4fede03e99e0e99b55378b16ea2`.
- gdc2 selected receiver render: `/Volumes/VertigoDataTier/pact/ddm_gdc2_categorical_coolchic_k8_distill/governed_v1/exports/stageC_lam1_0.0003_final/render.u8`, 117,964,800 B, sha256 `62a25bc44658b3872df7e4913900e70441d4bc4a06251b00e50fa7db160890d7`.
- Governing memos: gdc1 sha256 `c924bb6a831e7a8fa1a48e5f8d8e2f5433ef64e95fea84cbf06bca7f6e67c439`; gdc2 sha256 `2f534b1d2ec91038c1909544193b674d153ee7c7ca5ce67cac90c576595688f0`; gs3 Addendum 26 carrier sha256 `6bfca8e0b775380ec5dc83763355ad6133baa15d24d045c8663f1e55a810d0d6`.
- Geometry receipt: `/Volumes/VertigoDataTier/pact/ddm_gdc3_next_construction_against_the_geometry_law/geometry_probe_v1/RESULT.json`, 626,836 B, sha256 `9513573ec8cb2eadb072d966e4217e80e3b5aa4970bf6dfc335339c56cab9476`; manifest 385,443 B, sha256 `94f2b17b67b30a36d57f29aa49a2cce9bab4dbd139ca92b44b224d88009dd322`.
- Falsifier receipt: `/Volumes/VertigoDataTier/pact/ddm_gdc3_next_construction_against_the_geometry_law/keyrow_falsifier_v1/RESULT.json`, 37,282 B, sha256 `a31f84ba2ff23d5cab31e5ef9c699ba511aad678a5ba5cd2695ed3e0e4c50d50`; manifest 18,449 B, sha256 `4ab48badeff0b2ce7b7e7a0a450af626ac9b4f65d2edb2b05853db684a6bfebb`.

The predecessors' SSD trees were read only. All new raw streams, every coder output and determinism repeat, every candidate packet and repeat, all four n600 renders, and both exact closure fields were retained under the two gdc3 receipt roots. An independent end-of-run audit rehashed all 1,131 manifest rows with zero missing or changed files.

## Geometry definition and equation

For each source field, let `E = (source != target)`. Horizontal maximal runs of `E` are classified as isolated (`length = 1`), short (`2 <= length <= 8`), or long (`length >= 9`). A target cell is boundary if any four-neighbor has a different class; boundary-adjacent means four-neighbor distance at most one from this two-sided boundary, and interior is its complement. The six geometry cells are exclusive and exhaustive. Class marginals use the target class, with Lane kept separate.

For source `G`, geometry cell `g`, order `o`, and selected real coder `C`, the measured marginal is

`rho(G,g,o) = bytes(C(o(delta_G restricted to g))) / count_nonzero(delta_G restricted to g)`.

The exact residual price for a candidate is not `M` times a borrowed constant. It is

`R_exact(candidate) = min_o bytes(C(o(candidate XOR target)))`,

with decode required to reproduce the target byte-identically. The geometry table is a design screen only; the candidate's own real residual decides the verdict.

## Measured geometry table

Each byte number is the best real-coded payload in that order. `B/M` is bytes per mismatch. Empty gdc1 isolated-interior has no rate.

| Geometry class | gdc1 K8 M | gdc1 tile64 B/M | gdc1 frame B/M | gdc2 M | gdc2 tile64 B/M | gdc2 frame B/M |
|---|---:|---:|---:|---:|---:|---:|
| isolated, boundary-adjacent | 8,826 | 2.028440 | 1.843757 | 56,993 | 1.554472 | 1.562876 |
| isolated, interior | 0 | n/a | n/a | 6,861 | 1.581548 | 1.799883 |
| short run, boundary-adjacent | 71,893 | 0.576579 | 0.616527 | 629,684 | 0.407811 | 0.501045 |
| short run, interior | 569 | 1.818981 | 1.762742 | 49,466 | 0.636841 | 0.747625 |
| long run, boundary-adjacent | 5,685 | 0.518206 | **0.494987** | 262,385 | 0.251020 | **0.246234** |
| long run, interior | 1,331 | 0.779865 | 0.740045 | 41,686 | 0.355131 | 0.364727 |
| full residual | 88,304 | 0.685360 | 0.726354 | 1,047,075 | 0.371721 | 0.461453 |

Long boundary-adjacent runs are the robust cheapest class: 0.494987 B/mismatch for gdc1 and 0.246234 B/mismatch for gdc2 after choosing the better order for each source. The rate is not a universal constant: the same named geometry still differs by about 2.01x between sources. The run populations also differ: gdc1 had 659 long-boundary runs (median 10, maximum 24); gdc2 had 23,518 (median 11, maximum 117).

### Per-target-class marginal table

This table answers which target classes are expensive before geometry conditioning. Each line selects the better order for that source/class; the complete 30-cell class-by-geometry-by-order table is in the retained geometry receipt.

| Target class | gdc1 K8: M; best order; bytes; B/M | gdc2: M; best order; bytes; B/M |
|---|---|---|
| Road | 7,914; frame; 8,326; 1.052060 | 207,376; tile64; 109,446; 0.527766 |
| Lane | 63,143; tile64; 37,072; 0.587112 | 557,983; tile64; 140,760; 0.252266 |
| Undrivable | 4,291; frame; 5,148; 1.199720 | 116,537; tile64; 53,936; 0.462823 |
| Movable | 12,530; frame; 8,340; 0.665603 | 95,019; tile64; 46,104; 0.485208 |
| MyCar | 426; frame; 871; 2.044601 | 70,160; frame; 22,848; 0.325656 |

Lane is the mismatch-count load in both fields—71.51% of gdc1 K8 errors and 53.29% of gdc2 errors—but its long boundary runs are cheap when the construction actually produces them. The design problem is therefore not “ignore Lane”; it is “represent Lane edges natively and leave coherent boundary runs rather than scattered cells.”

## Three construction screens

All predicted numbers below are explicitly DERIVED or HYPOTHESIS. Only construction A was built and measured.

### A. Fixed anisotropic key-row ribbons — built, FORMULATION-NO-GO

Store exact target rows on a fixed vertical schedule and repeat the last anchor row downward. Generic parsing and zero-order vertical fill are free; the counted packet contains schedule identity plus every video-derived anchor row. This was designed to turn vertical approximation error into horizontal boundary ribbons. It differs from gdc1 because it sparsifies whole rows rather than storing capped segments on every row, and from gc1 because it has no class-blind 2-D paint overlays. The mechanism is anchored by the progressive low-resolution idea in [ITU-T T.82](https://www.itu.int/rec/T-REC-T.82).

Four full-n600 schedules were retained:

| Variant | Anchor rows | Packet B | M | DERIVED residual B using its measured geometry table | DERIVED total B |
|---|---:|---:|---:|---:|---:|
| uniform2 | 192 | 227,551 | 634,370 | 279,273.80 | 506,824.80 |
| uniform4 | 96 | 120,731 | 1,695,914 | 602,934.93 | 723,665.93 |
| bands8_4_2 | 112 | 137,207 | 1,353,572 | 505,342.39 | 642,549.39 |
| bands16_8_4_2 | 90 | 105,851 | 1,924,388 | 673,562.81 | 779,413.81 |

`uniform2` was the least-bad design-point total and was taken through the real residual falsifier. The whole four-variant build, encode, exact-decode, and geometry analysis took 169.225 s on the declared host, below the 1,260 s receiver budget as a conservative upper bound.

### B. Class-protected anisotropic strip tree — FOLDED by recall, not built

Use `HORZ`, `VERT`, asymmetric and 4-way strip nodes, with class-protected occlusion and Lane-aware split costs; generic traversal/rasterization is free, while topology, labels, and video-derived boundary parameters are counted. The [AV1 specification](https://github.com/AOMediaCodec/av1-spec/blob/master/07.bitstream.semantics.md) anchors the anisotropic partition vocabulary. It differs from gc1's class-blind square overlays and from the global Laguerre/power-diagram family because leaves are occlusion-ordered strips.

The HYPOTHESIS design point was K=50,000 B and M=88,000, with errors deliberately converted to long boundary-adjacent runs. The M anchor is the measured 88,304-error gdc1 K=8 field rounded down as the required target, not an achieved fit. At the conservative gdc1 long-boundary rate, predicted residual is `88,000 * 2,814 / 5,685 = 43,558.84 B`; predicted total is 93,558.84 B, leaving only 451.16 B. Decode risk is low in principle—bounded tree traversal and raster fill—but was not measured.

Full-corpus recall changed this candidate's disposition. The prior v8 class-matched/geometric atom family already reduced its proxy substantially yet required 210,255 B for exact completion, and the exact Lane contour alone cost 130,960 B; epsilon simplification lost after buying errors back. Those are not identical strip-tree mechanisms, but they consume the same complete analytic-boundary representation budget and already exceed this whole door. This candidate is therefore FOLDED unless a future derivation proves a packet intercept below 50 KB before any build.

### C. Learned run-native endpoint generator — QUEUED-WITH-A-FIRE-ORDER; training required

Train a small shared categorical decoder to emit bounded `(x_stop, class)` sequences per row from counted per-frame latents. Endpoint traversal/rasterization is free; every learned weight, latent, class/run prior, and correction is counted. The design uses the same anisotropic [AV1](https://github.com/AOMediaCodec/av1-spec/blob/master/07.bitstream.semantics.md), progressive binary-context [T.82](https://www.itu.int/rec/T-REC-T.82), and run-native [COCO mask](https://github.com/cocodataset/cocoapi) mechanisms as anchors, but the packet ABI and model are original here.

It differs from gdc2 because its learned symbol space is runs rather than dense cells, from gdc1 because a shared learned decoder replaces explicit per-row endpoints, and from qma9 because no post-codec explicit run-escape flag family exists. The HYPOTHESIS design point is K=60,000 B, M=65,000, at least 90% long boundary-adjacent and at most 1% interior. The M threshold is derived from the door: `(94,010 - 60,000) / (2,814 / 5,685) = 68,708`, tightened to 65,000. Predicted residual at that required point is 32,174.14 B; predicted total is 92,174.14 B. Receiver risk is a bounded small integer endpoint decoder plus raster fill, with a mandatory measured wall below 1,260 s.

This candidate needs training. Per the charter, no training was launched. The governed launch and stop conditions are in `.omx/research/ddm_gdc4_run_native_endpoint_generator_charter_20260910.md`; MAIN GO is required.

## Real n600 falsifier result

The fixed key-row family was fitted directly from the exact target labels, so no training or scorer was involved. All four renders were 117,964,800 B and retained. The selected `uniform2` render sha256 is `5be16279339e56b25df69ae64038d256d9231b3d00ad9cde61f2613b0bcdc6a4`.

`uniform2` produced 634,370 mismatches:

- isolated boundary-adjacent: 51,231;
- short boundary-adjacent: 346,882;
- long boundary-adjacent: 236,257;
- every interior class: zero;
- Road 298,594; Lane 147,931; Undrivable 4,220; Movable 31,253; MyCar 152,372.

The geometry premise survived: 100% of errors were boundary-adjacent. The rate premise did not. The counted program packet was 227,551 B. The candidate's own exact residual was 225,612 B in `tile64_time` order using LZMA2, or 272,196 B in `frame_raster` order using Brotli q11. The selected 225,612 B residual is 0.355647 B/mismatch, not the borrowed screen value. Its decode plus the candidate reproduced the target sha256 `78e57545...` byte-identically. Exact total: `227,551 + 225,612 = 453,163 B`.

Verdict scope is exactly the four fixed anisotropic key-row zero-order-hold schedules listed above. It does not kill learned row predictors, endpoint-native models, adaptive content-selected rows, or interpolation generally. It does kill the claim that fixed row subsampling alone resolves the generator door.

## RECALL EVIDENCE

The recall search covered the full `.omx/research/` corpus by content, not only charter seeds; `.venv/bin/python tools/list_canonical_equations.py --json`; `CANONICAL_RESEARCH_INDEX*`; `sub015_DAG_*` FEED blocks; design/SPEC files; and state/task-ledger surfaces. Representative content queries were `run escape|horizontal run|rowspan|scanline|contour|SDF|atom bank|boundary-adjacent|residual geometry|program + residual`, `procedural_predictor_plus_residual`, and `gdc3|generator door|94,010|Lane contour`.

Findings beyond the charter seeds and their consequences:

- `.omx/research/ddm_vr1_v7_v11_signal_recall_20260903.md` records the prior v8 class-matched/geometric-dominant family: proxy 0.339 to 0.061, but exact completion 210,255 B versus 113,411 B shipped. This changed construction B from a build candidate to FOLDED absent a new sub-50 KB intercept proof.
- `.omx/research/ddm_gt2_gt_tongue_induction_20260803.md` prices a lossless Lane contour at 130,960 B and shows epsilon simplifications lose once errors are bought back. This removed direct Lane contour transmission from the candidate set.
- `.omx/research/qma9_range_mask_deconstruction_20260503_codex.md` records horizontal-run escape flags projecting worse. This removed “base codec plus explicit flags”; construction C instead makes endpoint runs the native learned output language and still prices a separate real exact residual.
- Canonical equation `procedural_predictor_plus_residual_correction_savings_v1` supplied the exact `K + R + H - N` accounting frame. Canonical `argmax_of_sdf_is_additively_weighted_power_diagram_v1` supports an SDF interpretation but does not erase the measured atom/contour packet intercept.
- Canonical `cls_lowres_downsample_policy_boundary_preservation_v1` warns that cell sampling policy changes boundaries, but its domain does not transfer to this vehicle. It changed fixed key rows into a falsifier with no transfer claim; it did not make their result authority for mode or interpolated row decoders.
- Task/lane searches found no duplicate active gdc3 owner before the run. During execution, cpx3 accidentally swept the completed geometry probe and test into commit `e39a88a9d`; `main_hot_state.md` records the incident. Their current bytes match that commit. This receipt does not rewrite or claim cpx3 work and lands only the remaining gdc3-owned artifacts.

I did not find, in those bounded corpus/index/DAG/task searches, a prior byte-closed endpoint-native learned generator against this move-44 token field. That scoped absence is why construction C remains a live hypothesis rather than a claimed original breakthrough.

## Verification and boundaries

- Full population: 600/600 frames for both geometry sources and all four candidate renders.
- Real coder race: Brotli q11, raw LZMA2 extreme, and zlib9; both `tile64_time` and `frame_raster`; repeat payloads byte-identical.
- Exact closures: all geometry-order reconstructions and both selected candidate residual orders reproduced the target bytes.
- Manifest audit: 1,070 geometry rows + 61 candidate rows, zero missing/hash/byte failures.
- Tests: 7 focused tests passed; ruff and py_compile passed.
- P0 retention detector: 2/2 gdc3 runner files examined, zero findings.
- Review: two explicit review-tracker passes per Python file. The implementation spec was written before code. The arbitrage skill's delegated `codex exec` route failed twice before sampling with `Operation not permitted`; the permitted local takeover produced the implementation.
- Not measured: d_seg, d_pose, contest score, candidate archive size, or contest receiver time. No scorer, burn, Modal, upstream edit, sealed-tree edit, or predecessor-tree write occurred.

## Next charter and disposition

`.omx/research/ddm_gdc4_run_native_endpoint_generator_charter_20260910.md` is **QUEUED-WITH-A-FIRE-ORDER**, owner **MAIN**, consumer `/Volumes/VertigoDataTier/pact/ddm_gdc4_run_native_endpoint_generator/`. Fire only after MAIN GO, unique lane claim, implementation landing, exact compiled configuration, storage preflight, deterministic integer/reference receiver, and resumable per-stage checkpoint gates. This gdc3 arm does not authorize the launch.

## LIVE-HYPOTHESES

- A learned endpoint-native decoder may lower the field-description intercept while structurally guaranteeing coherent horizontal boundary errors. It remains plausible because both measured sources price that geometry cheapest, and no prior byte-closed instance of this exact model/field was found in the searched corpus.
- Adaptive or interpolated key rows may outperform fixed zero-order hold, but only if their counted selector/interpolation packet is priced from the start. The fixed family proved boundary concentration is possible; it did not test content-adaptive anchors or endpoint interpolation.
- A class-protected anisotropic strip tree could reopen only with a pre-build proof of a sub-50 KB program intercept. Anisotropic leaves match the measured run geometry, but earlier atom and exact-contour prices make this low confidence.

## DEAD-ENDS

- Fixed anisotropic key rows with zero-order vertical hold: FORMULATION-NO-GO across `uniform2`, `uniform4`, `bands8_4_2`, and `bands16_8_4_2`; best exact total 453,163 B versus 94,010 B.
- Direct exact Lane contours or epsilon-simplified contours with correction buyback: prior measured packet 130,960 B and simplification losses already exceed the whole door.
- Dense categorical latent recoding: gdc2 is capacity-limited and its latents are incompressible beyond zeroth-order entropy; coder substitution does not create the missing structure.
- Base codec plus explicit horizontal run-escape flags: qma9 already projected this bolt-on family worse; future run work must make runs the native representation.
- Borrowing one residual B/mismatch constant across generators: the same long-boundary class differed about 2.01x between gdc1 and gdc2, and the key-row candidate's own real rate differed from its screen projection.
