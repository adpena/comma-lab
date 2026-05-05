---
name: New tools landed 2026-04-29 — Selfcomp paradigm + Ara + EUREKA stack
description: Inventory of new src/tac/, experiments/, submissions/robust_current/, scripts/, tools/ files added today. Each tool with what-it-does + when-to-use. Total ~3500 LOC across 25 files + 3 inflate.sh dispatch arms.
type: reference
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## src/tac/ (Selfcomp paradigm core + EUREKA stacking)

- `segmap_renderer.py` — SegMap arch (94K params) + SegMapTrainer (eval_roundtrip + KL distill + EMA + roundtrip_noise_std=0.5). SegMapHomography 8-DOF subclass for Lane HM-S. pair_weights kwarg for Lane WC-S Curator.
- `mask_grayscale_lut.py` — encode_masks_grayscale (5 SegNet classes → grays {0,64,128,192,255}); create_gaussian_softmax_lut(sigma=15); decode_grayscale_to_classes (NN argmax). Used by Lane MM/SA/SC++/SO/AL.
- `block_fp_codec.py` — encode_conv_weight (per-channel int8 qint_max=7, ceil(log2)); HWOI permute; pack/unpack_payload_tar_xz; verify_roundtrip with NaN/Inf assertions. Selfcomp-paradigm weight format. per_channel_qint_max + per_key_qint_max for Lane FR-Ω.
- `arithmetic_qint_codec.py` — Witten/Neal/Cleary 1987 arithmetic coder (32-bit precision E1/E2/E3 scaling). Hits Shannon bound within 1%. Used by Lane SH (replaces tar.xz on qints).
- `segmap_film_canvas_renderer.py` — SegMap subclass + per-frame FiLM table on layer_in (init=0 → identical to vanilla at epoch 0). Used by Lane FC.
- `pose_delta_codec.py` — pose_delta_v1 sentinel (anchor + int8 deltas + per-channel scale). 600x6 → 5793B from 8745B fp16 (-34% pose rate). Wired through canonical load_optimized_poses.

## experiments/ (orchestration + training)

- `train_segmap.py` — SegMap training entrypoint with --variant {plain,kl_distill,hessian_quant} + --hidden/--block-hidden/--num-blocks/--epochs/--batch-size/--arch {segmap,segmap_homography}/--pair-weights/--device. **BUG: --batch-size flag dead inside SegMapTrainer.train_epoch (Subagent H fixing).**
- `train_segmap_film_canvas.py` — Lane FC training entrypoint.
- `init_segmap_from_posenet.py` — Lane PA helper. Runs frozen PoseNet on GT video → SegMap.frame_affine_embedding via arctanh inversion of tanh+scale.
- `build_lane_mm_archive.py` — Lane MM encoder. Re-encodes anchor archive masks as grayscale.mkv using Selfcomp class targets via ffmpeg subprocess `-pix_fmt gray`.

## submissions/robust_current/ (inflate paths)

- `inflate_renderer_grayscale.py` — Lane MM inflate (PYTHON_INFLATE=renderer_grayscale): grayscale.mkv → Gaussian LUT → argmax → re-encode legacy masks.mkv → existing inflate_renderer.py. NO scorer load.
- `inflate_segmap.py` — SegMap-paradigm inflate (PYTHON_INFLATE=segmap): grayscale.mkv → SegMap → bicubic upsample to 1164×874. SEGMAP_ARCH env var dispatches segmap vs segmap_homography. NO scorer load.
- `inflate_segmap_film_canvas.py` — Lane FC inflate (PYTHON_INFLATE=segmap_film_canvas).
- `inflate_segmap_arithmetic.py` — Lane SH inflate (PYTHON_INFLATE=segmap_arithmetic).
- `inflate.sh` — added 4 dispatch arms: renderer_grayscale, segmap, segmap_film_canvas, segmap_arithmetic.

## scripts/ (lane scripts, all use CONFIG_ENV_PATH override for PYTHON_INFLATE)

Selfcomp 4 (in flight or recently dispatched):
- `remote_lane_mm_grayscale_lut.sh` — Lane MM (encoder-only, 2h, $0.30)
- `remote_lane_sa_segmap_clone.sh` — Lane SA (12h, A10G)
- `remote_lane_sc_plus_plus_kl_distill.sh` — Lane SC++ (12h, A10G)
- `remote_lane_so_hessian_block_fp.sh` — Lane SO (KILLED per council)

5 sweep (deferred until SC++ control):
- `remote_lane_fr_omega_fridrich_block_fp.sh` — Lane FR-Ω
- `remote_lane_hm_s_segmap_homography.sh` — Lane HM-S
- `remote_lane_darts_s_segmap_arch_sweep.sh` — Lane DARTS-S
- `remote_lane_wc_s_curator_weighted.sh` — Lane WC-S
- `remote_lane_fr_mm_sigma_sweep.sh` — Lane FR-MM

5 EUREKA (deferred until SC++ control):
- `remote_lane_pa_pose_as_affine.sh` — Lane PA
- `remote_lane_fc_film_canvas.sh` — Lane FC
- `remote_lane_sh_shannon_arithmetic.sh` — Lane SH
- `remote_lane_tr_temporal_residual.sh` — Lane TR
- `remote_lane_pd_pose_deltas.sh` — Lane PD

## tools/ (Ara paradigm support)

- `ara_compile.py` — 505-line ARA Compiler (Seal Level 1). 4 stages: Semantic Deconstruction → Cognitive Mapping → Physical Stubbing → Exploration Graph Extraction. PRIVATE_TERMS redaction. Currently 0 errors / 9 warns on docs/paper/ara/.

## docs/paper/ara/ (Ara skeleton)

- `PAPER.md` — root manifest
- `RECOMMENDATION_20260429.md` — pivot recommendation + 6-pass hand-off
- `logic/` — 10 falsifiable claims bound to 10 experiments
- `src/index.md` — physical layer
- `trace/{exploration_tree.yaml, events.jsonl, seal_report.json}` — 3-era DAG + 408 events
- `evidence/` — 10 evidence files (2 real auth-eval JSON + 8 transcribed/placeholder)

## How to apply

- For each new lane: cite the file paths above when invoking; reuse rather than rebuild.
- For Lane AL (in implementation by Subagent I): will likely add experiments/optimize_grayscale_canvas.py + scripts/remote_lane_al_analog_latent.sh.
- Subagent H (lane failure hardening): may add new STRICT preflight check + modify SegMapTrainer.train_epoch.
- ARA skills (next session): install via `npx @orchestra-research/ara-skills install --all --local`; run /rigor-reviewer docs/paper/ara/.
