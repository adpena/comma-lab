# Top PR Intelligence Pass - 2026-05-04

Scope: read-mostly PR95/top-PR intake for PR85, PR91, PR92, PR95, PR96.
No remote jobs or GPU dispatch were launched in this pass.

## Source Snapshot

- Official public leaderboard, refreshed 2026-05-04: https://comma.ai/leaderboard
- PR85: https://github.com/commaai/comma_video_compression_challenge/pull/85
- PR91: https://github.com/commaai/comma_video_compression_challenge/pull/91
- PR92: https://github.com/commaai/comma_video_compression_challenge/pull/92
- PR95: https://github.com/commaai/comma_video_compression_challenge/pull/95
- PR96: https://github.com/commaai/comma_video_compression_challenge/pull/96
- HNeRV external context: https://github.com/haochen-rye/HNeRV and
  https://arxiv.org/abs/2304.02633

Machine-readable intake:

- `experiments/results/leaderboard_intel_20260504_codex/leaderboard_intel_summary.json`
- `experiments/results/leaderboard_intel_20260504_codex/pr{85,91,92,95,96}_api.json`
- `experiments/results/leaderboard_intel_20260504_codex/pr{85,91,92,95,96}_files.json`
- `experiments/results/leaderboard_intel_20260504_codex/pr{85,91,92,95,96}_comments.json`
- `experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip`
- `experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/{inflate.py,inflate.sh}`
- `experiments/results/leaderboard_intel_hnerv_delta_20260504_agent/report.md`

## Current Score Targets

Official leaderboard is still PR85 at display score `0.26`; the page has not
yet incorporated the lower open PR claims. Open PR claims are external until
the organizer eval or our exact CUDA replay validates the exact archive bytes.

| Source | State | Evidence here | Claimed/recomputed score | Bytes | SegNet | PoseNet | Caveat |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Our PR85+STBM1BR | local A++ | exact T4 CUDA | `0.25369011029397787` | `229756` | `0.00057185` | `0.00018940` | Best confirmed local contest-grade archive. |
| PR95 `hnerv_muon` | open | external PR body + static archive anatomy | `0.1987048012202245` | `178417` | `0.00061212` | `0.00003494` | Not yet official leaderboard; exact replay pending/required. |
| PR96 `rem2_HNeRV` | open | external PR body + static archive anatomy | `0.20567121179282477` | `186631` | `0.00062231` | `0.00003675` | CPU-report claim; exact CUDA/T4 replay required. |
| PR91 `hpac_coder_hybrid` | open | external PR body + local static/prefix failure | `0.24879480490416128` | `222404` | `0.00057185` | `0.00018940` | Local HPM1 prefix decode currently fails; not faithful evidence here. |
| PR92 `qzs3_range_joint_r258` | open | external PR body + local byte intake | `0.2587078229986317` | `236516` | `0.00057675` | `0.00018963` | Worse than our A++; useful mainly for RMB1/randmulti byte idea. |
| PR85 `adaptive_masking_joint_frame_model` | leaderboard | official display + body exact | `0.25806622496743437` | `236328` | `0.00057185` | `0.00018940` | We already beat this with STBM1BR by `0.0043761146734565`. |

## Archive And Runtime Anatomy Deltas

Our confirmed A++ champion is PR85-family, still fundamentally a packed mask
grammar plus joint-frame renderer:

- Score: `0.25369011029397787`
- Archive: `229756` bytes,
  SHA-256 `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Components: SegNet term `0.057185`, PoseNet term `0.043520110293977884`,
  rate term `0.15298499999999998`
- Runtime custody: replay `inflate.sh` with `stbm1br_mask_codec.py`, T4 CUDA.

PR95 changes the problem geometry. It is not a better PR85 packer; it is a tiny
task-trained video program:

- Archive: one stored `0.bin`, `178309` payload bytes, `178417` zip bytes.
- Payload layout: brotli JSON metadata `80` bytes, decoder brotli `162349`
  bytes, latent brotli `15868` bytes.
- Model: HNeRV-style pair decoder, `latent_dim=28`, `base_channels=36`,
  `n_pairs=600`, `eval_size=(384,512)`, bicubic upsample to camera size.
- Training claim: 8-stage curriculum ending in Muon fine-tune and compression-
  shaped regularization. The PR95 `codec.py` notes a previously tested hybrid
  categorical coder saved about `217` bytes but was dropped for simplicity.
- Score geometry: rate term improves by about `0.03418444316080153` relative
  to our A++, and PoseNet term improves by about `0.02482786591295184`; SegNet
  term regresses by about `0.004027`. Net claimed win over us is about
  `0.05498530907375337`.

PR96 independently confirms the HNeRV direction:

- Archive: `186631` bytes, SHA-256
  `2ecbd2118bebdb5566f719ed538a89c4608ccab19c9edc7ae7a6de778bd42b46`.
- Members: `decoder.bin` deflated `169272` bytes, `latents.bin` deflated
  `16133` bytes, and an unused-looking stored member `p` of `930` bytes.
- Runtime: single `inflate.py` with inline decoder, constriction range decode
  for selected decoder tensors, brotli histogram codec in this archive
  (`comp_id=2`), and bicubic upsample.
- Claimed score is worse than PR95 but close enough to validate the architecture
  class and expose an alternate decoder entropy path.

PR91/PR92 remain PR85-family packer deltas:

- PR91 replaces the PR85 mask segment with HPM1/HPAC: mask segment `145087`
  bytes versus PR85 QMA9 `159011` and our STBM1BR `152439`. This is the only
  clear PR91 byte win. Local prefix decode still fails on the HPM1 entropy
  contract, so the archive is not validated by us.
- PR92 keeps PR85 QMA9 mask bytes and recodes `randmulti` from `16101` to
  `15825` bytes. It is a valid small pure-rate idea for our STBM archive if the
  runtime contract is correct, but it cannot approach PR95 by itself.

## Top 5 Contest-Compliant Moves To Beat PR95

1. **Make PR95 replay truth the anchor, then optimize only against exact bytes.**
   Promote no PR95-derived claim until exact CUDA/T4 replay validates the
   public archive. Once replay lands, use its exact component trace as the base
   for all deltas. This is mandatory because CPU/GPU bicubic and PyTorch
   version differences can alter raw frames.

2. **PR95 decoder self-compression under score-preserving parity.**
   Decoder bytes dominate PR95 (`162349` of `178417`). Immediate moves:
   exact-evaluate the existing deterministic PR95 repack candidate at
   `178321` bytes / SHA-256
   `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`
   if replay runtime is closed; restore the PR95-documented hybrid categorical
   coder (`~217` byte known opportunity); then search per-tensor codec
   assignment, mixed int7/int8, block-FP/int4 for low-sensitivity tensors, and
   PR96-style range coding. Dispatch only if local inflate parity or exact
   trace predicts no component movement; otherwise treat as a trained QAT lane.

3. **Latent stream predictive/range coding with ego-motion basis.**
   PR95 latents are only `15868` bytes but are cleanly separable. Replace the
   first-order delta+lo/hi brotli stream with deterministic per-dimension
   autoregressive/range-coded residuals, PCA/active-subspace ordering, or
   camera/ego-motion-conditioned low-rank bases. A `1-3 KB` latent reduction is
   worth `0.00067-0.00200` score with no distortion if decode parity holds.

4. **Rate-distortion retraining with packer in the loop.**
   PR95 is already scorer-aligned, but its score is not a proof of optimality.
   Run parallel H100 sweeps over `base_channels`, latent dim, C1a weight,
   Muon/AdamW schedule, sigma/lambda, hard-pair/SegNet boundary weights, and
   entropy proxy. Keep snapshots byte-closed so any partial winner can be
   packaged immediately. This is the highest-upside route to beat PR95 by more
   than rounding noise.

5. **Tiny charged postfilter or residual atom layer on PR95 hard pairs.**
   After exact replay, compute frame/pair/component residual density against
   SegNet/PoseNet. Add only charged atoms that beat their rate cost: low-rank
   correction head, sparse wavelet residuals, per-pair latent nudges, or a tiny
   learned correction table. This is where Fridrich/Lagrangian water-fill,
   active subspaces, foveation, and hard-pair profiling should directly choose
   bytes. Do not use PR85 as a second full representation; use it only as a
   diagnostic source for correction atoms because carrying its archive would
   erase PR95's rate advantage.

## Compliance Concerns

- PR95 and PR96 are open PR claims, not current official leaderboard entries as
  of this pass. They are external until exact replay/organizer eval.
- PR95 reports CPU verification in the PR body even though its runtime can use
  CUDA. Exact CUDA/T4 replay is required before ranking or stacking.
- PR95 training reproducibility is not the same thing as archive evidence. The
  submitted archive can be evaluated as fixed bytes; reproducing the same bytes
  from random init is a separate paper/OSS claim.
- PR96 uses `constriction` and has a code path for `zstandard`; this archive's
  decoder histogram `comp_id=2` uses brotli, but preflight should fail closed if
  future PR96-family archives require undeclared codecs. The current upstream
  `pyproject.toml` includes brotli and constriction.
- PR96 carries a `p` member that its visible runtime does not read. It is
  charged, not hidden side information, but it should be classified as an
  unused or compatibility payload before trusting a derivative.
- PR95's current deterministic repack candidate is byte-only evidence:
  `experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip`
  saves `96` bytes and round-trips decoded streams, but still requires exact
  CUDA replay because runtime custody and output bytes are score truth.
- PR91 is not replay-safe locally: HPM1 entropy decode fails in prefix smoke,
  the PR does not include a compressor/builder, and local fallback parity is
  unresolved. Treat it as external design signal only.
- PR92 is byte/anatomy-useful but score-negative relative to our A++ unless the
  isolated RMB1 randmulti recode is stacked with our STBM runtime and exact
  eval confirms unchanged components.

## Immediate Next Dispatch Decisions

No dispatch was performed in this pass. The next score-push tranche should be:

1. Harvest/finish exact CUDA replay for PR95 and PR85+STBM+RMB1 if already
   running.
2. If PR95 validates, switch the champion branch to PR95-family and run two
   parallel tracks: byte-preserving packer improvement and H100 retraining with
   entropy/scorer losses.
3. Keep PR85-family only as a fallback submission and as a correction-atom
   diagnostic source; its architecture is now dominated by HNeRV on claimed
   rate and PoseNet.

## Codex Worker Addendum: PR95 HNeRV Packing Lowering

Artifact: `experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/profile_pr95_hnerv_muon_packing.json`.

Scoped local-only tooling now performs bit/accounting and deterministic packing
search for the PR95 HNeRV Muon single-member archive. It parses the top-level
`0.bin` into meta, decoder, and latent streams; records per-decoder-tensor byte
counts, entropy, zero fraction, standalone Brotli size, and SHA-256; profiles
each latent dimension against the matching `stem.weight` column; and emits
explicit no-op detection. This is byte/provenance evidence only until exact
CUDA replay.

New candidates:

- Conservative byte-exact runtime-contract repack:
  `experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip`
  is `178321` bytes, SHA-256
  `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`.
  This preserves the earlier queued exact-eval custody path and saves `96`
  bytes from public PR95 by compacting equivalent metadata, deterministic
  decoder-record ordering, and Brotli parameter selection.
- Aggressive stem/latent permutation candidate:
  `experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked_stemperm.zip`
  is `178277` bytes, SHA-256
  `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`.
  It saves `140` bytes from public PR95 and `44` bytes from the conservative
  repack by permuting latent dimensions and the matching `stem.weight` input
  columns under the existing PR95 runtime. This is contest-compliant and
  no-requantization, but it is not claimed score-preserving: a local two-pair
  CPU probe observed max decoder-output difference `0.0009307861328125`,
  mean absolute difference `3.191186624462716e-05`, and rounded uint8
  non-equality, consistent with changed floating-point accumulation order.
  Exact CUDA auth eval is required before ranking or submission use.

If PR95 public replay validates exactly at its claimed score, pure rate math
would put the aggressive archive at `0.19861158096678738` before accounting for
any output perturbation. That number is a prediction, not evidence.
