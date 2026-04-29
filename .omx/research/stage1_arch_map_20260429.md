# Stage 1 Architectural Map — sub-0.3 push

## 1. Submission pipeline

Current repo has two submission families:

**A. Current robust renderer archive.** Canonical archive contract is centralized in `tac.submission_archive`: archive construction is the single source of truth and ad hoc zips are explicitly banned (`src/tac/submission_archive.py:1`). The default renderer manifest is `renderer.bin + masks.mkv + optimized_poses.pt`; compact swaps `.pt` poses for `optimized_poses.bin` (`src/tac/submission_archive.py:201`, `src/tac/submission_archive.py:208`). `compress_archive.py` wraps that builder, supports half-frame masks, binary poses, and Brotli (`submissions/robust_current/compress_archive.py:2`, `submissions/robust_current/compress_archive.py:220`). Current local `archive.zip` contains `renderer.bin`, `masks.mkv`, and `optimized_poses.bin` with 725,030 raw member bytes.

**B. Selfcomp/SegMap archive.** `inflate_segmap.py` defines the newer layout: `segmap_weights.tar.xz`, `grayscale.mkv`, optional `optimized_poses.pt` (`submissions/robust_current/inflate_segmap.py:4`). Inflate path is: grayscale video -> LUT/classes -> one-hot -> SegMap -> bicubic camera upsample -> raw RGB (`submissions/robust_current/inflate_segmap.py:14`, `submissions/robust_current/inflate_segmap.py:180`). The shell dispatcher exposes `PYTHON_INFLATE=segmap`, `segmap_film_canvas`, and `segmap_arithmetic` (`submissions/robust_current/inflate.sh:323`, `submissions/robust_current/inflate.sh:334`, `submissions/robust_current/inflate.sh:345`). Stage-0 Brotli decompression is centralized before branch dispatch (`submissions/robust_current/inflate.sh:75`).

Encoder-side modules:

- Grayscale mask codec: class ids -> single 8-bit plane, then LUT back to probabilities/classes (`src/tac/mask_grayscale_lut.py:1`, `src/tac/mask_grayscale_lut.py:67`, `src/tac/mask_grayscale_lut.py:114`).
- SegMap renderer: 5 mask channels + 3 warped latent channels -> RGB (`src/tac/segmap_renderer.py:73`, `src/tac/segmap_renderer.py:157`).
- Block-FP payload: conv weights use qint + per-channel exponents; non-conv tensors use linear min/max quant; tar.xz member layout is explicit (`src/tac/block_fp_codec.py:420`, `src/tac/block_fp_codec.py:619`).
- Arithmetic payload: Lane SH replaces `segmap_weights.tar.xz` qint streams with `payload.bin` (`src/tac/arithmetic_qint_codec.py:317`, `submissions/robust_current/inflate_segmap_arithmetic.py:4`).

## 2. Rate levers

Highest EV is archive diet. Codex memory says 45KB saved is 0.03 score, and sub-0.30 at Quantizr size requires either non-rate <0.105 or shrink to ~240KB at Quantizr distortion (`project_codex_theoretical_floor_brutal_20260429.md:20`, `project_codex_theoretical_floor_brutal_20260429.md:24`).

Concrete levers:

- **Grayscale LUT masks:** Selfcomp’s one-channel gray mask is expected ~50% smaller than 5-channel one-hot AV1 (`src/tac/mask_grayscale_lut.py:11`, `src/tac/mask_grayscale_lut.py:13`). Lane MM and FR-MM ship this as encoder-only (`scripts/remote_lane_mm_grayscale_lut.sh:2`, `scripts/remote_lane_fr_mm_sigma_sweep.sh:2`).
- **Single-mask-per-pair / half-frame:** Quantizr-style builder supports keeping odd masks only (`submissions/robust_current/compress_archive.py:96`); Q-FAITHFUL explicitly targets single-mask + FiLM and half as many frames (`scripts/remote_lane_q_faithful_jointgen.sh:9`).
- **Binary / delta poses:** compact `.bin` saves ~8KB vs `.pt` (`src/tac/submission_archive.py:208`); Lane PD encodes anchor + int8 deltas and is decoded transparently by the canonical pose loader (`src/tac/pose_delta_codec.py:77`, `src/tac/submission_archive.py:517`).
- **Block-FP:** Selfcomp-like qint/exponent decode is `w ~= qint * 2**exp`; HWOI layout improves entropy (`src/tac/block_fp_codec.py:425`, `src/tac/block_fp_codec.py:511`).
- **Arithmetic qint:** Range coding stores qint streams near Shannon entropy (`src/tac/arithmetic_qint_codec.py:1`, `src/tac/arithmetic_qint_codec.py:232`), and Lane SH is a controlled archive-only repack (`scripts/remote_lane_sh_shannon_arithmetic.sh:4`).
- **Brotli:** archive builder and inflater support `.br`; quality 11 matches Quantizr (`src/tac/submission_archive.py:38`, `submissions/robust_current/inflate.sh:96`).

## 3. Distortion levers

SegMap architecture is the load-bearing shift. Memory identifies Selfcomp’s 94K-param SegMap as 5 mask channels plus 3 affine-warped latent channels, ResBlocks, sigmoid RGB (`project_selfcomp_reverse_engineered_20260429.md:16`). The local implementation has the same affine latent path: shared 3x30x40 latent, 6-D frame embedding, bounded zoom/aspect/shear/translation (`src/tac/segmap_renderer.py:94`, `src/tac/segmap_renderer.py:132`).

Pose/motion levers:

- Canonical SegMap uses analytical affine latent, not PoseNet at inflate (`src/tac/segmap_renderer.py:120`).
- HM-S adds 8-DOF homography embeddings with two perspective params at tiny rate cost (`src/tac/segmap_renderer.py:167`, `scripts/remote_lane_hm_s_segmap_homography.sh:2`).
- PA seeds affine from pose but script admits trainer does not fully honor `--init-from`; it restores only the embedding after training (`scripts/remote_lane_pa_pose_as_affine.sh:136`).

KL distill is contested. Current project rules say old KL loss mode is dead; `profiles.py` also marks `kl_distill` deprecated after 1.85/2.05 PoseNet collapse (`src/tac/profiles.py:14`). But SC++, SO, FR-Ω, FC, HM-S, DARTS-S, WC-S, and Q-FAITHFUL scripts still use `--variant kl_distill` or KL T=2.0 (`scripts/remote_lane_sc_plus_plus_kl_distill.sh:2`, `scripts/remote_lane_q_faithful_jointgen.sh:26`). Treat any KL lane as untrusted until authoritative score proves no PoseNet collapse.

## 4. Gaps

- **Quantizr-class clone is load-bearing but not yet proven locally:** memory says sub-0.30 probability collapses if q_faithful/SC++ do not produce Quantizr-class local score within 24h (`project_codex_theoretical_floor_brutal_20260429.md:9`, `project_codex_theoretical_floor_brutal_20260429.md:43`).
- **Archive diet not fully centralized for SegMap lanes:** Lane SH builds ad hoc zip with `payload.bin + grayscale.mkv + poses` (`scripts/remote_lane_sh_shannon_arithmetic.sh:130`) instead of `tac.submission_archive`’s strict manifest machinery.
- **Inflate parity bug risk:** standard SegMap uses bicubic upsample (`submissions/robust_current/inflate_segmap.py:191`), while arithmetic and FiLM-Canvas inflaters still use bilinear (`submissions/robust_current/inflate_segmap_arithmetic.py:173`, `submissions/robust_current/inflate_segmap_film_canvas.py:185`).
- **LUT divergence:** local LUT softmaxes log-distance, while comments say Selfcomp softmaxes `exp(-d^2/...)`; invisible only if argmax/one-hot is used (`src/tac/mask_grayscale_lut.py:138`).
- **Training entry mismatch:** AGENTS names `experiments/train_tac.py`, but this checkout has no such file; SegMap lanes use `experiments/train_segmap.py` scripts.

## 5. 14-lane portfolio status

Memory portfolio: MM running; SA, SC++, SO running; FR-MM, SH, TR, PD, PA, FR-Ω, HM-S, DARTS-S, WC-S, FC pending (`project_selfcomp_portfolio_tonight_20260429.md:16`, `project_selfcomp_portfolio_tonight_20260429.md:23`, `project_selfcomp_portfolio_tonight_20260429.md:28`). Repo has scripts for all 14: MM, FR-MM, SH, TR, PD, SA, SC++, PA, SO, FR-Ω, HM-S, DARTS-S, WC-S, FC. Highest-EV ordering should be: Q-FAITHFUL/SegMap score proof, then archive diet stack (SH/TR/PD/Brotli/block-FP), then only distortion bolt-ons that clear the 15KB => 0.01 score slope rule (`project_codex_theoretical_floor_brutal_20260429.md:22`, `project_codex_theoretical_floor_brutal_20260429.md:34`).
