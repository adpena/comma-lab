# ddm_hx1 — lessons-only sweep: PR 129 / 131 / 134 / 137
Date 2026-08-17. Lessons-only public intake. Read + static analysis. No launches, no scorer runs.
Custody: /Volumes/APDataStore/pact/ddm_hx1/intake/ (137 read from ddm_pq2 custody, not re-downloaded).

## PR 129 — qlp_exactgrid (ryanli0070, CLOSED, head 63f9bf7f)
Vehicle: frozen PR#95/#101 HNeRV decoder + PR#112 context range coder, unchanged; the ONLY
contribution is a quantization-aware polish of the 600x28 per-pair latents.
SCORE: **eval-bot CONFIRMED, device: cpu** (batch 16, threads 2, seed 1234) —
seg 0.00056051, pose 0.00002902, 176,337 B, rate 0.00469662, printed 0.19; author full precision
0.190503 (x86 CI) / 0.190506 (own CPU). BEATS rhnerv_comma #112 (0.191126) and #125 (0.190946).
No CUDA row exists.
"exactgrid" = **quantization grid alignment**, not a lattice reconstruction and not a grid search.
Receipt `submissions/qlp_exactgrid/pack_base.py:88-92`: the container packs latents with per-dim
**fp16** min/scale (`mins16=np.float16(mins)`, `scales16=np.float16(scales)`) then
`q=clip(rint((lat-mins16.astype(f32))/scales16.astype(f32)),...)`. The training-side straight-through
quantizer replicates that fp16 grid bit-for-bit, so what the optimizer sees IS what ships — zero
train/pack gap.
Second mechanism worth naming: the seg loss is a **boundary/margin** loss `sigmoid(-margin/tau)`
(smooth argmax-flip fraction, tau annealed), optimized through the EXACT inflate chain
(bicubic->874x1164, #98 channel biases, straight-through clamp/round). PR#101's 607-B sidecar is
dropped, absorbed into the latents.
VERDICT: SURPRISING (narrow). Independent, eval-confirmed evidence for two of our own levers:
(a) codim-1 margin/flip-fraction seg loss is what buys the last seg decimal at the frontier;
(b) STE must target the CONTAINER's grid, not a nominal quantizer.

## PR 131 — "Coolchic baseline" (Tibo-vagenBird, CLOSED, head 9266ca8a)
PR body is the UNFILLED template (empty report.txt block, no archive link). No eval-bot run ever.
Diff actually carries two unrelated things: (1) `experiments/coolchic_baseline/**` — a calibration
experiment, and (2) a full `submissions/rhnerv_latent_polish/**` (same family as #129: exact-score-
gated latent polish of #112, 1,802 +/-1/+/-2 grid-step changes, #101 sidecar folded in).
Cool-Chic: NOT a faithful contest submission — it is stock Cool-Chic 5.0 (Orange-OpenSource, MSE-
trained, full-res) run OUT OF TREE in a separate conda env, scored by an offline extrapolator
`score_coolchic.py` that codes N frames and multiplies rate to 1200. It NEVER exports an archive.zip
and has no inflate path: research-only by construction.
Only datum on disk: `runs/lmbda_0.02_n32/.../0000-results_encoder.tsv` — ONE frame (0000), intra,
psnr 36.13 dB, rate_bpp 0.013814 (nn 0.011479 + latent 0.002335), 673 s, 10000 iters. The README's
own stop-rule: ">1 at all reachable rates -> rate floor / MSE misallocation verdict, document and
stop." No score was recorded either way.
VERDICT: EMPTY. Nothing extractable, and it does not falsify Cool-Chic — it never got to an archive.
Honest read: the nn_bpp (0.0115) is 4.9x the latent_bpp (0.0023), i.e. at this rate region stock
Cool-Chic spends most of its budget on the SYNTHESIS/ARM weights, not the overfit latent. That is a
plausible reason the family stalls at contest rates, but it is a one-frame reading, not a verdict.

## PR 134 — metricwarp_av1 (bzlvkv, CLOSED, head f5267220)
Vehicle: full-resolution SVT-AV1 (preset 2, crf 56, 10-bit, tune=2, single keyframe, raw OBU,
456,276 B) + 7.3 KB of metric-guided side-channels. No neural weights in archive, CPU-only.
SCORE: **SELF-CLAIM ONLY, 0.93821 CPU** (seg 0.00531573, pose 0.00094302, 464,856 B). NO eval-bot
run. Maintainer YassineYousfi's only comment (2026-08-07) is a link to the repo's
coding-agents-and-LLMs policy — the body carries a "Generated with Claude Code" trailer. Closed.
THE mechanism: **exact-grid write into the metric's null space.** Receipt
`submissions/metricwarp_av1/inflate.py:21-27` (`_taps`) + `:89-99` (`gridplace_fullres`). Both
scorers bilinear-resize 1164x874 -> 512x384, align_corners=False, no antialias. Stride
1164/512 ~= 2.273 > 2, so the 2x2 tap blocks are PAIRWISE DISJOINT. `gridplace_fullres` writes the
same 512x384 uint8 value into all four taps (CY/CY1 x CX/CX1); the four bilinear weights sum to 1,
so the net's input pixel is set EXACTLY. Resample loss is exactly zero, and ~23% of full-res pixels
carry zero metric weight (constructive D-null-space).
Consequence they exploit: they never leave the 512x384 metric chart. Pose correction (dx,dy,rot,
zoom,bias,gain per pair, 1,686 B) is searched IN METRIC SPACE with the real PoseNet and
**uint8 rounding inside the loop** — README: floats-then-round costs 20x (pose MSE 1.23 -> 0.00094).
Seg fixes are greedy integer RGB nudges on 16x16 tiles of odd frames only (5,617 B, fixes 17% of
flipped pixels), uint8-exact so the decoder replays the search bit-for-bit. Per-pair layer mixing
(562/600 keep seg-fix) because seg-fix tile edges can hurt pose correctability.
CROSS-REF to our own ddm_pz1: we measured that a D-null-space field built in frame_1's lattice is
resampled by the pose warp onto a different lattice where D stops annihilating it (attenuation
1.662x). PR134 sidesteps that entirely by warping in the metric chart and grid-placing afterwards —
the lattice never changes. That is the transferable idea.
VERDICT: SURPRISING. Not for the score (0.94 is far off our lane and unconfirmed) but for the
construction: an explicit, verifiable basis for "which full-res pixels the scorer cannot see," plus
a measured 20x penalty for optimizing outside the uint8 grid.

## PR 137 — metric_shift_av1 (Amirjon06, OPEN, head 84962d57)
Vehicle: tuned multi-segment AV1 with film-grain synthesis + a ~1.2 KB per-frame luma side channel.
SCORE: **SELF-CLAIM ONLY, 2.04 CPU** (seg 0.00571624, pose 0.07881037, 866,558 B, rate 0.02308022).
No eval-bot run; only the auto-ack comment. Author's own framing is "better than baseline_fast"
(4.46 -> 2.04), not competitive with the leaderboard.
"metric shift" answer: **(b) a colorspace/normalization shift** — specifically a per-frame quantized
MEAN-LUMA correction, applied after the resize back to camera resolution. Receipts:
`submissions/metric_shift_av1/README.md:39-42` ("compares each frame to the source frame, and writes
a quantized mean-luma correction. During inflation, inflate.py applies that correction after
resizing"); `generate_sidechannel.py:570-573` (`correction_values` -> `inflate.luma_plane(diff).mean()
* gain`); `inflate.py:177-182` (`luma_plane` / `luma_plane_correction`) and `:501-515` (LUMA_GAIN /
LUMA_BIAS / RGB_BIAS post pass). It is NOT scorer-blind-spot targeting and NOT a bitrate ladder
retarget. A second, unrelated use of the word "shift" exists — `inflate.py:226-265`
(`estimate_global_shift` / `shift_rgb`, integer pixel shift for a pair-asymmetric blend) — but the
headline mechanism is the luma bias.
VERDICT: EXPECTED. Conventional AV1 tuning plus a scalar brightness-drift correction. Pose 0.0788 is
~2,700x worse than PR129's 0.0000290; the luma channel does not touch the geometry the scorer reads.
Nothing to take.

## Cross-PR pattern (tail of the leaderboard)
1. Clean split. The classical AV1 entries (134, 137) sit at 0.94 / 2.04 SELF-CLAIMED; the HNeRV
   neural lineage (129, and 131's rhnerv_latent_polish) sits at 0.19 CONFIRMED. Nothing classical is
   within 4x. Also: only PR129 has an eval-bot row at all — 131/134/137 are all unconfirmed.
2. The one convergent lesson across BOTH families is the same: put the real scorer, WITH uint8
   rounding, inside the search/optimization loop, and make the training-time quantizer bit-identical
   to the shipping container. PR129 gets it via an STE that replicates the fp16 pack grid; PR134
   measures a 20x pose penalty for rounding after instead of during. Every entry that actually moved
   its number did this; none that skipped it did.
3. The non-HNeRV neural lane is untested, not falsified. Cool-Chic (131) never produced an archive,
   and its single frame spends 4.9x more bits on synthesis/ARM weights than on the overfit latent —
   suggestive of why the family stalls at contest rates, but one frame is not a verdict.

## Custody
Downloaded this arm into /Volumes/APDataStore/pact/ddm_hx1/intake/ :
  pr129_meta.json 3,838 B ef6867574fbc | pr129_comments.json 1,738 B 6025a9044dd9
  pr129/README.md 4,404 B ce3c2187f4fc | pr129/pack_base.py 4,818 B 02c0f2ca7fd9
  pr129/inflate.py 3,701 B 06cdba16af94
  pr131_meta.json 1,459 B cecbe94fcf8c | pr131_comments.json 558 B 2faf76594fb9
  pr131/coolchic_baseline_README.md 2,064 B f1f65eba5937
  pr131/coolchic_baseline_score_coolchic.py 6,293 B 5c024163b120
  pr131/rhnerv_latent_polish_README.md 4,889 B 7a6d3771fb22
  pr131/work_0000-results_encoder.tsv 1,243 B d3e5ac66c2fc
  pr134_meta.json 2,959 B 941e19599f77 | pr134_comments.json 1,075 B f82e39d80429
  pr134/README.md 4,904 B 2326740fd2d4 | pr134/inflate.py 9,380 B 77fab33f4492
  pr134/tools_metric_lib.py 1,943 B 4e46f1647e5f | pr134/tools_correct2.py 5,942 B 53c46b0dae51
  pr137_src/* reconstructed locally from the EXISTING ddm_pq2 diff (no download):
    inflate.py 35,656 B 5898ea29cdec | generate_sidechannel.py 60,073 B 3cb5af2501c6
    compress.sh 16,538 B c7600ccddfe6 | README.md 3,714 B 2f49c29b8791
Read-only, not re-downloaded: /Volumes/APDataStore/pact/ddm_pq2/intake/pr137_meta.json,
pr137_full.diff, pr137/archive.zip (866,558 B, sha b396b26279c3 — matches the PR body's stated size).
Pre-existing in the intake dir from a prior arm, untouched by me: pr132*, pr133*, pr136*.
