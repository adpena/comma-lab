# SG2: PR130 segmentation-gap source audit

## Verdict

I did not find a post-render segmentation closer in the complete 49-stage PR130 repro graph at intake commit `e34f31bc4969042c0051ac81aa3c56884419a231`. The apparent 3.87x gap is a measurement-protocol mismatch: MAIN evaluated the floating master weights stored by a QAT checkpoint through `train_semantic_full.evaluate_all`, while the deployed archive runs the same semantic renderer after int4 pack/dequantization. The deployment-shaped metric is already recorded by stages 07 and 08.

The named mechanism is therefore **stage-07/08 int4 quantization-aware semantic rendering**, not a stage after the renderer. `train_semantic_quantized.py:30-64` fake-quantizes every semantic parameter and renders through the exact camera round trip; its `evaluate_all` calls that quantized path at `:90-108`, and it stores `quant_bits` plus `quantized_exact_seg` at `:356-376`. In contrast, `train_semantic_full.py:26-39,70-77` loads the same checkpoint state but calls the floating `render_for_seg` path and never consumes `quant_bits`. That is the protocol used for the 0.001105-0.001108 MAIN rows.

This is a scorer-free, static source-and-byte audit. I did not run SegNet, PoseNet, an evaluator, a trainer, an archive builder, CUDA, Metal, or MPS. Published official rows and checkpoint metrics below are pre-existing evidence, not new measurements.

## What the archive actually ships

Static parsing of the frozen `artifacts/final/archive.zip` and read-only invocation of the source packer established:

| Fact | Value | Axis |
|---|---:|---|
| Archive | 191,052 B; SHA-256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd` | measured bytes |
| Semantic renderer | 66,339 parameters | static model census |
| Raw int4 code floor | 33,170 B | derived as `ceil(66,339/2)` |
| Packed semantic section | 40,252 B | measured bytes |
| Rank-less-than-2 parameters and scale overhead | 7,082 B | derived as `40,252 - 33,170` |
| Semantic share of whole archive | 21.0686% | descriptive ratio, not compressed marginal cost |
| Stage-07 packed semantic SHA-256 | `81058169865ffc7d1a400feba7dbe174d3610b5d55af78d13aa595062ecc1ea9` | measured bytes; not the archive section |
| Stage-08 packed semantic SHA-256 | `9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99` | measured bytes; exactly the archive section |
| Stage-07 recorded quantized exact_seg | 0.0002972496880425347 | checkpoint record; original `/workspace/semantic-pose/gt_cache_600.pt` axis |
| Stage-08 recorded quantized exact_seg | 0.0002763705783420139 | checkpoint record; same cache axis |
| Tail delta | -0.000020879109700520817 (-7.0241%) | derived from checkpoint records |
| Published Ada official d_seg | 0.00028609 | pre-existing `[contest-CUDA]` report |
| Published A4500 official d_seg | 0.00029607 | pre-existing `[contest-CUDA]` report |

The final semantic bytes are produced by `pack_semantic_pose.py:97-123`, reconstructed by `inflate.py:171-215`, and rendered as the second/last frame at `inflate.py:610-635`. The pack/decode path therefore matches the quantized-forward representation, not the floating master. The exact archive marginal attributable to the semantic section is **not identifiable without a counterfactual rebuild**, because semantic, carrier, and HPAC bytes are jointly LZMA-compressed at `build_submission_archive.py:92-105`. The honest price is 40,252 raw packed bytes inside that joint stream; there are zero extra post-render-closer bytes.

The stage-08 tail is not merely a rate finish under the deployment-shaped evidence: its recorded quantized exact_seg is 7.0241% lower than stage 07 at the same 40,252-byte packed size, and its packed blob is exactly the one in the published archive. The difference between its cache-recorded 0.0002763706 and the published Ada 0.00028609 is 0.0000097194; this bounded static audit does not assign that residual to decoder, hardware, or cache-axis effects.

## Exhaustive 49-stage table

`$PR130` below means `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`. Inputs and outputs are taken from the authoritative `Pipeline._build` declarations in `$PR130/scripts/e2e.py`; line citations identify each declaration. `Touch` means whether the stage can alter the segmentation result in principle, not whether it explains MAIN's perfect-token float/quantized mismatch.

| # | Stage | Entry point | Inputs -> outputs | Touch | Source |
|---:|---|---|---|---|---|
| 01 | `01_targets` | `build_gt_cache_official.py` | challenge video/names/scorers -> official target cache + report | Reference/training-input only; no shipped frame | `e2e.py:296` |
| 02 | `02_semantic_seed_b2` | `semantic_renderer_oracle.py` | target cache -> b2 semantic checkpoint + report | Direct semantic frame-1 producer | `e2e.py:313` |
| 03 | `03_semantic_all600` | `train_semantic_full.py` | targets + stage 02 -> checkpoint + report | Direct semantic frame-1 producer | `e2e.py:331` |
| 04 | `04_semantic_exact_b2` | `train_semantic_full.py` | targets + stage 03 -> checkpoint + report | Direct semantic frame-1 producer | `e2e.py:348` |
| 05 | `05_semantic_expand_b4` | `expand_semantic_checkpoint.py` | stage 04 -> expanded checkpoint | Direct semantic model transform | `e2e.py:366` |
| 06 | `06_semantic_train_b4` | `train_semantic_full.py` | targets + stage 05 -> checkpoint + report | Direct semantic frame-1 producer | `e2e.py:378` |
| 07 | `07_semantic_qat12k` | `train_semantic_quantized.py` | targets + stage 06 -> QAT checkpoint + report | Direct; deployment-shaped int4 QAT | `e2e.py:396` |
| 08 | `08_semantic_tail6k` | `train_semantic_quantized.py` | targets + stage 07 -> final QAT checkpoint + report | Direct; final deployed semantic state | `e2e.py:414` |
| 09 | `09_pose_pilot4` | `learned_pose_carrier_oracle.py` | targets + stage 02 -> pose checkpoint + report | No: pose carrier becomes frame 0 | `e2e.py:435` |
| 10 | `10_pose_pilot12` | `learned_pose_carrier_oracle.py` | targets + stages 02/09 -> pose checkpoint + report | No: pose carrier becomes frame 0 | `e2e.py:463` |
| 11 | `11_pose_joint_qat` | `train_pose_carrier_full.py` | targets + stage 07 + stage 10 -> carrier checkpoint/latest + master cache + report | No: saves basis/coeff for frame 0 | `e2e.py:491` |
| 12 | `12_pose_raw7500` | `train_pose_carrier_full.py` | targets + stage 07 + stage 11/latest + master cache -> carrier checkpoint/latest + report | No: frame-0 carrier only | `e2e.py:521` |
| 13 | `13_pose_hard750` | `train_pose_carrier_full.py` | targets + stage 07 + stage 12/latest + master cache -> carrier checkpoint/latest + report | No: frame-0 carrier only | `e2e.py:549` |
| 14 | `14_pose_resident` | `train_pose_carrier_full.py` | targets + stage 07 + stage 13/latest + master cache -> carrier checkpoint/best + report | No: frame-0 carrier only | `e2e.py:580` |
| 15 | `15_pose_uniform` | `train_pose_carrier_full.py` | targets + stage 07 + stage 14/best + master cache -> carrier checkpoint/best + report | No: frame-0 carrier only | `e2e.py:610` |
| 16 | `16_pose_tail64` | `train_pose_carrier_full.py` | targets + stage 07 + stage 15/best + master cache -> carrier checkpoint + report | No: frame-0 carrier only | `e2e.py:639` |
| 17 | `17_pose_cpu_coeff100` | `train_pose_carrier_full.py` | targets + stage 07 + stage 16 + master cache -> carrier checkpoint + report | No: frame-0 carrier only | `e2e.py:669` |
| 18 | `18_pose_search32` | `search_pose_coeff_cpu.py` | targets + master cache + stage 17 -> carrier checkpoint + report | No: coefficient search for frame 0 | `e2e.py:698` |
| 19 | `19_pose_search256` | `search_pose_coeff_cpu.py` | targets + master cache + stage 18 -> carrier checkpoint + report | No: coefficient search for frame 0 | `e2e.py:713` |
| 20 | `20_pose_cpu_fullqat` | `train_pose_carrier_full.py` | targets + stage 07 + stage 19 + master cache -> carrier checkpoint + report | No: frame-0 carrier only | `e2e.py:728` |
| 21 | `21_pose_retarget_coeff1000` | `train_pose_carrier_full.py` | targets + stage 08 + stage 20 -> carrier checkpoint/latest + final master cache + report | No: reads final semantic, saves frame-0 carrier | `e2e.py:757` |
| 22 | `22_pose_basis_adapt250` | `train_pose_carrier_full.py` | targets + stage 08 + stage 21/latest + final master cache -> carrier checkpoint/best + report | No: frame-0 carrier only | `e2e.py:789` |
| 23 | `23_pose_basis_adapt3000` | `train_pose_carrier_full.py` | targets + stage 08 + stage 22/best + final master cache -> carrier checkpoint + report | No: frame-0 carrier only | `e2e.py:820` |
| 24 | `24_pose_final_cpu100` | `train_pose_carrier_full.py` | targets + stage 08 + stage 23 + final master cache -> carrier checkpoint/best + report | No: frame-0 carrier only | `e2e.py:849` |
| 25 | `25_pose_official_coeff` | `train_pose_carrier_full.py` | targets + stage 08 + stage 24/best + final master cache -> carrier checkpoint/best + report | No: frame-0 carrier only | `e2e.py:879` |
| 26 | `26_pose_grid128x3` | `search_pose_coeff_cpu.py` | targets + final master cache + stage 25/best -> carrier checkpoint + report | No: coefficient search for frame 0 | `e2e.py:907` |
| 27 | `27_pose_grid64x12` | `search_pose_coeff_cpu.py` | targets + final master cache + stage 26 -> carrier checkpoint + report | No: coefficient search for frame 0 | `e2e.py:922` |
| 28 | `28_pose_refine_pass1` | `refine_pose_coeff_codes.py` | targets + final master cache + stage 27 -> carrier checkpoint + report | No: coefficient refinement for frame 0 | `e2e.py:937` |
| 29 | `29_pose_refine_pass2` | `refine_pose_coeff_codes.py` | targets + final master cache + stage 28 -> carrier checkpoint + report | No: coefficient refinement for frame 0 | `e2e.py:954` |
| 30 | `30_pose_anchor` | `refine_pose_coeff_codes.py` | targets + final master cache + stage 29 -> carrier checkpoint + report | No: coefficient refinement for frame 0 | `e2e.py:971` |
| 31 | `31_pose_int6_stable8k` | `train_pose_carrier_full.py` | targets + stage 08 + stage 30 + final master cache -> carrier checkpoint + report | No: frame-0 carrier only | `e2e.py:988` |
| 32 | `32_pose_int6_coefftail4k` | `train_pose_carrier_full.py` | targets + stage 08 + stage 31 + final master cache -> final carrier checkpoint + report | No: frame-0 carrier only | `e2e.py:1018` |
| 33 | `33_hpac_smoke` | `train_hpac_integer.py` | targets -> HPAC checkpoint + report | Input-only: may corrupt semantic tokens, cannot improve perfect tokens | `e2e.py:1049` |
| 34 | `34_hpac_epoch60` | `train_hpac_integer.py` | targets + stage 33 -> HPAC checkpoint/latest + report | Input-only: may corrupt semantic tokens, cannot improve perfect tokens | `e2e.py:1064` |
| 35 | `35_hpac_raw_refine` | `train_hpac_integer.py` | targets + stage 34/latest -> HPAC checkpoint + report | Input-only: same bound | `e2e.py:1082` |
| 36 | `36_hpac_spm` | `train_hpac_integer.py` | targets + stage 35 -> HPAC checkpoint + report | Input-only: same bound | `e2e.py:1098` |
| 37 | `37_hpac_frame_scale` | `train_hpac_integer.py` | targets + stage 36 -> HPAC checkpoint + report | Input-only: same bound | `e2e.py:1116` |
| 38 | `38_hpac_halfstep_migrate` | `train_hpac_integer.py` | targets + stage 37 -> HPAC checkpoint + report | Input-only: same bound | `e2e.py:1136` |
| 39 | `39_hpac_halfstep_refine` | `train_hpac_integer.py` | targets + stage 38 -> HPAC checkpoint + report | Input-only: same bound | `e2e.py:1163` |
| 40 | `40_hpac_patch64` | `train_hpac_integer.py` | targets + stage 39 -> HPAC checkpoint + report | Input-only: same bound | `e2e.py:1189` |
| 41 | `41_hpac_selfcompress` | `train_hpac_self_compress.py` | targets + stage 40 -> HPAC checkpoint + report | Input-only: same bound | `e2e.py:1217` |
| 42 | `42_hpac_pack` | `pack_hpac_self_compress.py` | stage 41 -> packed HPAC + report | Input-only model pack; exact round-trip required | `e2e.py:1258` |
| 43 | `43_tokens_encode` | `codec_hpac_integer.py` | stage 41 + targets -> token stream + report | Input-only; encodes the exact target tokens | `e2e.py:1301` |
| 44 | `44_tokens_verify` | `codec_hpac_integer.py` | stage 41 + targets + token stream -> verification report | No mutation; requires exact token equality | `e2e.py:1314` |
| 45 | `45_predecessor` | `build_submission_archive.py` | stage 08 + stage 32 + packed HPAC + tokens -> predecessor payload/archive + report | Selection/pack only; no semantic correction | `e2e.py:1330` |
| 46 | `46_cpr1` | `repack_carrier.py` | predecessor archive -> final CPR1 archive + report | No: lossless carrier-only recode | `e2e.py:1353` |
| 47 | `47_submission` | `stage_submission.py` | CPR1 archive + runtime files -> staged submission + report | No: file staging only | `e2e.py:1368` |
| 48 | `48_official_evaluation` | `evaluate.sh` | staged submission + challenge evaluator -> report | No mutation; measures d_seg | `e2e.py:1388` |
| 49 | `49_official_report` | `verify_official_report.py` | report + archive -> metrics JSON | No mutation; parses and recomputes score | `e2e.py:1406` |

All 49 declarations were read. Stages 09-49 are the 41 stages downstream of the final semantic-training stage; stages 09-45 are the 37 strict pre-final-CPR1 candidates. No stage was reduced to its filename alone: the candidate families were traced to their output and runtime consumers.

## Downstream candidate elimination

1. **Pose stages 09-32:** `train_pose_carrier_full.py:237-264` loads and quantizes the semantic master only to render fixed masters, while `:400-462` persists only basis and coefficient tensors. At decode, those tensors write frame 0 at `inflate.py:637-653`; the semantic renderer writes frame 1 at `:610-635`. The official SegNet consumes only the last frame (`upstream/modules.py:103-113`). Therefore these 24 stages cannot change d_seg.
2. **HPAC/token stages 33-43:** HPAC can affect d_seg only by supplying different semantic tokens. `codec_hpac_integer.py:61-128` encodes/decodes the target map, and stage 44 requires exact equality at `:176-195`. Runtime then passes the decoded tokens directly to the semantic renderer (`inflate.py:559-598,621-630`). MAIN's renderer measurement already supplied those perfect target tokens, so an exact codec cannot make its input better; it can only preserve it or, if defective, worsen it.
3. **Stage 44:** verification only; no output payload mutation.
4. **Stage 45:** `build_submission_archive.py:33-61,87-107` selects stage 08, packs it, concatenates the independently packed carrier/HPAC/tokens, and writes a deterministic ZIP. It has no frame correction stream.
5. **Stage 46:** `repack_carrier.py:215-270` decodes/re-encodes only carrier symbols, asserts their equality, and copies the semantic, HPAC, and token bytes unchanged.
6. **Stages 47-49:** stage/copy, evaluate, and parse only.

## Ranked explanation of the apparent 3.87x gap

1. **Confirmed: MAIN used the wrong checkpoint evaluation path.** `train_semantic_full.evaluate_all` performs a floating forward even when its input checkpoint is a QAT checkpoint. The 0.001105-0.001108 rows therefore do not measure the representation shipped by the archive.
2. **Confirmed: stages 07/08 already record deployment-shaped d_seg near the published value.** Their int4-QAT records are 0.00029725 and 0.00027637, and stage 08 packs byte-identically to the final archive's semantic section.
3. **Bounded residual: cache/decoder/hardware axis.** The stage-08 checkpoint's recorded metric used `/workspace/semantic-pose/gt_cache_600.pt`, whereas the authoritative driver builds an official DALI cache and the published archive has distinct Ada/A4500 results. The evidence supports an axis difference but does not identify its exact share without a new scorer run, which SG2 forbids.
4. **Not supported in the audited intake:** a different hidden checkpoint, a post-render correction stream, a second renderer, or a mis-stated published d_seg. The archive contains exactly the named stage-08 semantic bytes, and two official GPU reports are recorded in `evidence/cpr1_verification.json`.

## Static Metal portability

- **Stages 07/08 mechanism:** statically portable with unexecuted risk. `train_semantic_quantized.py:30-64,90-108` uses PyTorch tensor operations, `torch.func.functional_call`, and bilinear interpolation. The only CUDA-specific TF32 calls are guarded by `device.type == "cuda"` at `:182-187`. SG2 did not execute Metal, so this is not a runtime parity verdict.
- **Target builder:** CUDA-bound as written; `build_gt_cache_official.py:28-38` rejects hosts without CUDA and binds the DALI dataset to CUDA.
- **Published decoder:** CUDA-bound as written; `inflate.py:657-690` rejects hosts without CUDA and hardcodes `torch.device("cuda")`. Removing that gate would be a code change requiring parity validation. Thus the QAT mechanism is statically Metal-portable, but the shipped archive runtime is not Metal-runnable unchanged.

## RECALL EVIDENCE

Before adjudication I searched:

- `.omx/research`, `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED surfaces, and `.omx/state/probe_outcomes.jsonl` for `PR130`, `CPR1`, `semantic renderer`, `HPAC`, `quantized_exact_seg`, `0.00028609`, `3.87`, and the archive SHA;
- the canonical-equations listing from `.venv/bin/python tools/list_canonical_equations.py --json` for PR130/semantic/HPAC-specific equations;
- the PR130 roadmap and OP1R archive-anatomy receipts, plus the operator and probe ledgers;
- the authoritative intake driver, every downstream entry point, the exact final archive, and both semantic checkpoints.

Beyond the charter seeds, OP1R/CPR1 evidence supplied the exact archive anatomy and exact carrier/token preservation claims, while the roadmap exposed the semantic/token boundary. The canonical equation registry did not add a PR130-specific equation in this searched scope. The recall changed the plan from hunting only downstream frame transforms to comparing the archive's embedded semantic blob against both checkpoint packings and auditing the float-versus-QAT evaluator split. That comparison resolved the premise without a scorer run.

## Disposition and frontier

The search for a post-render segmentation closer is **FOLDED** at the complete-intake/static-source scope. Reactivate only if the driver, final archive SHA-256, stage-08 checkpoint SHA-256, semantic packer, runtime unpacker, or SegNet last-frame contract changes. No SG2 follow-on remains queued.

This arm did not move the pointer. Own-vehicle frontier remains `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`.
