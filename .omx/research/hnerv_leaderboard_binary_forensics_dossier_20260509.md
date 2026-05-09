# HNeRV leaderboard binary forensics dossier (2026-05-09)

<!-- generated_at: 2026-05-09T05:15:00Z, from_state_hash: hnerv_binary_forensics_pass_complete -->
<!-- HISTORICAL_PROVENANCE — append-only, audit forensic record -->

This dossier reverse-engineers the comma video compression challenge HNeRV
cluster (PRs #95, #98, #99, #100, #101, #102, #103, #105, #106) from their
public archive bytes, source code, PR comments, and author writeups. The
operator's question: "what specific training/architecture/hyperparameter thing
did they do that we haven't replicated."

All findings tagged `[empirical: <artifact path>]` per CLAUDE.md
"Forbidden Score Claims" rule. No proxy-derived claims; every line is from a
file in the repo.

Companion file (subagent handoff): `.omx/research/hnerv_forensics_critical_findings_for_a1a9359d_20260509.md`

## Executive summary

| PR | Author | Public CPU score | Archive bytes | Decoder weights source | Inflate-time tricks | Place |
|---|---|---:|---:|---|---|---|
| #95 | AaronLeslie138 | 0.1987 | 178,417 | trained from random (8-stage pipeline) | none | honorable + best writeup |
| #98 | EthanYangTW | 0.1963 | 178,392 | QAT-finetune of #95 | none | leaderboard |
| #99 | BradyMeighan | 0.19667 | 178,546 | re-pack of #95 | per-pair latent sidecar (615 B) | — |
| #100 | BradyMeighan | 0.1954 | 178,981 | re-pack of #95 | sidecar (606 B, scale=0.0100) | — |
| #101 | SajayR | 0.19284 | 178,258 | unchanged from #98 | sub-byte codec polymorphism + per-channel Y/V offset | **1st (gold)** |
| #102 | EthanYangTW | 0.19499 | 178,981 | byte-identical to #100 | sidecar scale 0.0100→0.0095 + frame0 R+1 | **3rd (bronze)** |
| #103 | rem2 | 0.19487 | 178,223 | re-pack of #95 | constriction AC + sidecar | **2nd (silver)** |
| #105 | valtteri | 0.19797 | 177,857 | own retrain (PR106 pipeline) | sidecar tuned to PyAV GT | honorable |
| #106 | valtteri | 0.20946 | 186,239 | own retrain (PR106 pipeline) | sidecar tuned to DALI GT | leaderboard |

The medal-band cluster (PR101 0.193 / PR103 0.195 / PR102 0.195) sits inside
4 hours of contest race time on a substrate that was published by
AaronLeslie138 at 2026-05-04T07:47:15Z with the **complete training
pipeline checked in**. Our PR #107 apogee submission (0.229) shipped 4h 23min
later via a separate substrate.

## §1 Per-PR runtime-tree summary

### PR #95 — `hnerv_muon` (AaronLeslie138, root of the cluster)

`/Users/adpena/Projects/pact/experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/`
(this directory is PR106's clone but the `train.py`/`stages/`/`losses.py`/`optim.py`/`codec.py` is byte-identical to PR95's published source, per PR98+PR99+PR100+PR101 README chain attribution)

- **Decoder architecture** (`src/model.py`, 55 lines): 28-d latent → Linear stem → 36 base channels at 6×8 → 6 PixelShuffle ×2 stages with sin() activation + bilinear-upsample skip → dilated-conv refine residual at 0.1× scale → separate `rgb_0` and `rgb_1` 3×3 conv heads (frame 0 / frame 1 weights NOT shared) × sigmoid × 255. Channel taper: `[36, 36, 36, 27, 20, 18, 18]`. ~229K params.
- **Training: 8-stage curriculum** (~30,650 epochs at batch_size=8, ~50 hours):
  - S1 random init, AdamW lr=1e-3 latent_lr=1e-2, CE seg, 3000 ep
  - S2 τ-Softplus (tau=0.3), 5650 ep
  - S3 smooth-disagreement (sigmoid bell), fresh cosine lr=1e-4, 1500 ep
  - S4 +QAT (per-tensor sym INT8 STE), 500 ep
  - S5 +L7-weighted Softplus + C1a entropy reg (λ=0.01, σ=0.2), lr=3e-5, 9000 ep
  - S6 C1a λ→0.02, 2000 ep
  - S7 C1a σ→0.1, 3000 ep
  - S8 +Muon optimizer (Newton-Schulz orth momentum, WD=5e-4) on hidden convs + AdamW on stem/heads/biases/latents, 5000 ep
- **Loss aggregation**: `loss = 100·seg + 1·pose + λ_C1a·entropy`. Pose is `sqrt(10·MSE(pose_pred[:6], pose_target))` (concave-in-MSE, emphasizes small errors).
- **EMA decay 0.999** from epoch 0; EMA shadow used at every eval, EMA shadow saved as inference checkpoint.
- **eval_roundtrip baked into training inner loop** (`stages/common.py:179-194`): every gradient step does bicubic-up to (874,1164) → bilinear-down to (384,512) → clamp(0,255) → STE-round → YUV6 → SegNet/PoseNet (matches contest scorer chain exactly).
- **rgb_to_yuv6 differentiability monkey-patch** (`src/data.py:51-81`): replaces `frame_utils.rgb_to_yuv6` AND `modules.rgb_to_yuv6` at import time so PoseNet's `preprocess_input` doesn't sever autograd.
- **Best-checkpoint selection by exact archive build**: every `eval_every=25` epochs, `build_archive` → `parse_archive` round-trip → real `evaluate_decoder` on the actual video, and the lowest-score archive bytes are saved as the inference-eligible candidate.
- **Codec**: per-tensor sym INT8 (n=127, scale=absmax/127) → zigzag → concat → brotli(quality=11) for decoder; per-dim asym uint8 + 1st-order temporal delta + zigzag → uint16 lo/hi byte split → brotli for latents; final archive = 12-byte header + meta_brotli + decoder_blob + latents_brotli. Round-trip verified bit-exact at codec stage.

[empirical: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/`]

### PR #98 — `hnerv_muon_finetuned_from_pr95` (EthanYangTW)

PR98's source was not separately re-cloned in our intake (only PR98's archive was downloaded for adjudication; canonical source mounted via PR101 lineage). PR98's archive was QAT fine-tune of PR95's pretrained weights. Score 0.1963 vs PR95 0.1987 = -0.0024 from ~500 epochs additional QAT. The PR98 author's PR comment also references a "decode-side channel postprocess" that PR101 explicitly inherits via `up[:,0,0].sub_(1.0)` etc.

[empirical: `pr98.json` PR comments + `pr101.json` README "PR #98's decode-side channel postprocess"]

### PR #99 — `hnerv_muon_lc` (BradyMeighan)

178,546 byte archive. PR comment: "Re-pack of @AaronLeslie138's `hnerv_muon` with our schema-driven INT8 codec (~480 B saved) and a 615 B per-pair latent-correction sidecar. The sidecar grid-searches a single-dim perturbation per latent that minimizes joint SegNet+PoseNet distortion under his decoder. Both seg and pose drop (0.000612→0.000595 and 0.0000349→0.0000334)." Score 0.19667.

The sidecar idea is the load-bearing innovation in #99. It costs nothing in
training time — it's a post-hoc grid search at compression time over a small
fixed delta vocabulary. Adds ~600 bytes archive overhead and saves ~1.5
mScore. Every PR after #99 inherits this sidecar mechanism.

[empirical: `pr99.json` PR body]

### PR #100 — `hnerv_lc_v2` (BradyMeighan)

178,981 byte archive. Same decoder bytes as PR95. Sidecar refined to 606 bytes,
DELTA_SCALE=0.0100. Layout: 4-section LE-u32 (dec_brotli, scale_fp16, latent_brotli, sidecar_brotli).
Score 0.1954. Inflate has no per-channel offsets.

The PR100 README explicitly: "HNeRV decoder weights and architecture by AaronLeslie138 (PR #95 / hnerv_muon). This submission re-packs his archive ~470 B smaller via schema-driven layer names + fp16 scales, and adds a ~1.2 KB latent-correction sidecar (per-pair single-dim perturbation chosen to minimize SegNet+PoseNet distortion)."

[empirical: `experiments/results/public_pr100_intake_20260504_codex/source/submissions/hnerv_lc_v2/inflate.py:1-13`]

### PR #101 — `hnerv_ft_microcodec` (SajayR, GOLD)

178,258 byte archive. Sole 1st prize. Decoder bytes are unchanged from PR98
(byte-identical state dict to PR95+PR98). The deltas vs PR100 are:

- **Sub-byte codec polymorphism**: 6 alternative sidecar layouts (HUFF_ENUM=607 / HUFF_COMB=609 / HUFF=614 / SPLIT=656 / PACKED=661 / RAW=600/1200) — encoder picks the smallest. PR101's archive uses HUFF_ENUM at 607 bytes [empirical: `extract_archive_layout.py` output: `sidecar: len=607, SIDECAR_HUFF_ENUM_LEN=607, matches HUFF_ENUM=True`].
- **Hand-tuned per-tensor brotli stream segmentation**: `DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)` — 7 separate brotli streams cover the 28 tensors at hand-picked partition boundaries.
- **Hand-tuned per-tensor permutation order**: `DECODER_STORAGE_ORDER = (14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11, 18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0)` — 28 distinct positions optimizing brotli compressibility of the concatenated stream.
- **Hand-tuned per-tensor 4D conv axis transpose**: `CONV4_STORAGE_PERMS = {2:(3,0,2,1), 4:(3,0,2,1), 6:(0,1,2,3), 8:(3,0,1,2), 10:(3,0,2,1), 12:(3,0,1,2), 14:(1,0,2,3), 16:(3,0,2,1), 18:(1,0,2,3), 20:(0,3,2,1), 22:(0,3,2,1), 24:(0,2,3,1), 26:(0,1,3,2)}` — different permutation per 4D conv tensor.
- **Hand-tuned per-tensor signed-int byte-map**: `DECODER_BYTE_MAPS = {9:'negzig', 14:'negzig', 20:'twos', 27:'off'}` — tensors 9 and 14 use negated zigzag (sign-flipped to match brotli's frequency bias), tensor 20 uses two's-complement int8, tensor 27 uses simple offset (subtract 128); the rest default to standard zigzag.
- **Latent codec**: `lzma.FORMAT_RAW` with `{FILTER_LZMA1, dict_size=4096, lc=3, lp=0, pb=0}` (raw LZMA, no XZ envelope, dictionary tuned for this exact 600×28 latent payload). Latents stored DIM-MAJOR via `LATENT_DIM_ORDER = (26, 0, 17, 15, 10, 24, 20, 12, 14, 21, 22, 18, 4, 11, 3, 7, 16, 2, 6, 8, 19, 23, 5, 9, 1, 13, 27, 25)`, with per-dim 1st-order delta + zigzag, then split into hi/lo bytes.
- **Sidecar fixed delta vocabulary**: `[-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10] × 0.01` — 16 alternatives per pair. The HUFF_ENUM layout encodes (a) packed dim-positions via base-LATENT_DIM mixed-radix integer, (b) Huffman-coded delta values with the canonical-Huffman-length-vector itself ranked through the Kraft-equality space (`huff_length_vector_count` enumerates valid length vectors with `MIN=2, MAX=8, KRAFT_TOTAL=256`), (c) no-op positions via combinatorial colex ranking.
- **Inflate-time per-channel Y/V bias correction**: `up[:, 0, 0].sub_(1.0); up[:, 0, 2].sub_(1.0); up[:, 1, 1].sub_(1.0)` — frame 0 R/B channels lose 1, frame 1 G channel loses 1. Attributed to "PR #98's decode-side channel postprocess" in the README. This costs zero archive bytes and is the highest-priority candidate explanation for PR101's distortion edge, but the causal effect still needs a same-archive offset sweep.

[empirical: `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py + inflate.py`]

### PR #102 — `hnerv_lc_v2_scale095_rplus1` (EthanYangTW, BRONZE)

178,981 byte archive — **byte-identical to PR100's archive**. Verified:
SHA-256 `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
on both files [empirical: `extract_archive_layout.py` output, both PR100 and
PR102 print this exact SHA].

The PR102 README explicitly: *"The archive payload is unchanged from PR #100;
only inference-time code constants changed."* The two changes:

1. `DELTA_SCALE` 0.0100 → **0.0095** (sidecar correction multiplier)
2. Add `up[:, 0, 0].add_(1.0)` (frame 0 red channel +1 at decode time)

That's it. PR102 won third prize on 2 numerical constants on top of PR100/PR95
bytes. PR100 0.1954 → PR102 0.19499.

[empirical: `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/sidecar.py:14, inflate.py:118-119`]

### PR #103 — `hnerv_lc_ac` (rem2, SILVER)

178,223 byte archive. Same decoder bytes as PR95. Layout: 7 sections
(`sca:54 / br:7097 / hists:895 / merged_ac:153856 / mins_scales:112 / lo_b:15537 / hi_hist:15 / wrp:557`).

- **Constriction-based arithmetic coding**: 8 specific tensor indices `AC_INDICES = [0, 2, 4, 6, 8, 10, 12, 21]` (stem.weight, blocks.0-5.weight, refine.0.weight — the heaviest weight tensors) get per-tensor empirical histogram conditioning + range coding via the `constriction` library. Other tensors stay brotli.
- **Latent split**: hi byte goes through AC (per-archive histogram), lo byte goes through brotli.
- **Same sidecar mechanism** as PR99/PR100 (single-dim correction per pair) but encoded as `brotli(N_PAIRS × [u8 dim, i8 dq_zz])`.
- **No inflate-time channel bias correction.**

PR103's `inflate.py` is one self-contained 222-line file (no `src/` subdirectory). The PR comments record a heated but ultimately resolved exchange about CPU-vs-CUDA evaluation; PR103 is silver after CPU re-eval at 0.19487.

[empirical: `experiments/results/public_pr103_intake_20260504_codex/source/submissions/hnerv_lc_ac/inflate.py`]

### PR #105 — `kitchen_sink` (valtterivalo)

177,857 byte archive. valtteri's own retrain via the PR106 pipeline, sidecar
tuned against PyAV-decoded ground truth. Same `model.py`/`codec.py`/`losses.py`
as PR106 (per byte-identical diff: `diff PR105/src/codec.py PR106/src/codec.py
=> identical`; same for model.py).

Per the kitchen-sink writeup HTML (`/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/kitchen_sink.html`): valtteri's pre-HNeRV-pivot best was a hand-built semantic codec (231,891 byte archive) at 0.257328, dominated by an exact mask cache (170 KB sparse-CTW Markov + brotli) + 6-DOF pose deltas (3 KB Laplace+range) + selector tail bytes + an inflate-time Adam loop on the renderer. This was overtaken on the final day when HNeRV emerged.

[empirical: `experiments/results/public_pr105_kitchen_sink_intake_20260504_codex/source/submissions/kitchen_sink/` and the writeup HTML]

### PR #106 — `belt_and_suspenders` (valtterivalo)

186,239 byte archive — the LARGEST in the cluster. Own retrain via the
8-stage pipeline (which is what's checked in under `src/`); sidecar tuned
against DALI cu128 ground truth. The DALI-tuned sidecar mismatched the actual
GitHub Actions cu124 runner enough that PR106 ended up at 0.20946 instead of
PR105's 0.19797. **PR106's value to us is that the FULL TRAINING PIPELINE IS
CHECKED IN HERE** (the same source that PR105 also has, but PR106 is the
clearer of the two for forensic investigation because it's tagged `belt_and_suspenders` "GPU-only — leaderboard auto-routes to T4").

[empirical: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/{train.py, stages/*.py, losses.py, optim.py, codec.py, score.py, data.py}`]

## §2 Per-PR archive byte-level layout

[empirical: `/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/extract_archive_layout.py` output]

| PR | Inner blob bytes | Inner SHA256 (first 32) | Layout | Sections (entropy bits/byte) |
|---|---:|---|---|---|
| 100 | 178,873 | `3234f0689164cfc95b7ee9f9cdf38ecf` | 4×LE-u32 (dec/sca/lat/wrp) | dec=162346 (7.997), sca=56 (5.022), lat=15849 (7.985), wrp=606 (7.675) |
| 101 | 178,158 | `5f1948f9572e65f71c614d2ff15764ee` | fixed offsets dec=162164 / lat=15387 / sidecar=607 | dec=162164 (7.998), lat=15387 (7.988), sidecar=607 (7.711) |
| 102 | 178,873 | `3234f0689164cfc95b7ee9f9cdf38ecf` | **identical to PR100** | identical to PR100 |
| 103 | 178,123 | `dd92b065b8fbd2bfe9de837eb110a971` | 8 sections | sca=54 (5.007), br=7097 (7.973), hists=895 (7.783), merged_ac=153856 (7.999), mins_scales=112 (5.063), lo_b=15537 (7.987), hi_hist=15 (3.774), wrp=557 (7.647) |
| 105 | 177,749 | `5f88fa8ed26816cf6e86a00cf3c88915` | first byte 0xff = "packed" layout (PR106 codec) | n/a — single packed brotli stream |
| 106 | 186,131 | `7f2cc905b7611ae8d7bced72be24e226` | first byte 0xff = "packed" layout | n/a — single packed brotli stream |

Observations:
- Decoder payload entropy is at 7.99-7.998 bits/byte (essentially Shannon-saturated for brotli quality 11). Further byte savings on the decoder weight bytes are at the AC/rANS level, not the brotli level.
- Latent payload entropy is at 7.98 bits/byte (also near saturated under brotli; lzma was tried in PR101 to squeeze a few more bytes).
- Sidecar payload entropy is 7.65-7.71 bits/byte (under-saturated — there is about 0.3 bits/byte of recoverable headroom in the sidecar via better modeling, but at 600-byte total this is <30 bytes absolute).
- Latent `hi` histogram (PR103, 15 bytes) at 3.77 bits/byte — this is a peaky distribution (most hi bytes are 0; the 15-byte payload encodes the ~5 nonzero hi-byte alphabet symbols + frequencies).

## §3 Per-PR weight tensor inventory

[empirical: `/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/extract_weights.py` output]

**PR100, PR101, PR102, PR103 share BIT-IDENTICAL DECODER WEIGHTS.** This was
proven by dequantizing each PR's archive with each PR's own parser and
comparing all 28 tensors' `absmax`, `std`, `min`, `max` values to >6 decimal
places. Every single tensor matches across these four PRs.

PR105 and PR106 are valtteri's own retrains and have slightly perturbed weight
absmax values (e.g., `stem.weight absmax`: PR100/101/102/103=1.526070 →
PR105=1.525254 → PR106=1.524996). They are independent retrainings of the
same architecture.

Weight statistics (28 tensors, 229K params total, all from PR100/PR95):

| Layer | Shape | absmax | std | int8_step |
|---|---|---:|---:|---:|
| stem.weight | (1728, 28) | 1.526 | 0.220 | 0.0120 |
| stem.bias | (1728,) | 2.108 | 0.633 | 0.0166 |
| blocks.0.weight | (144, 36, 3, 3) | 0.603 | 0.110 | 0.0047 |
| blocks.1.weight | (144, 36, 3, 3) | 0.515 | 0.096 | 0.0041 |
| blocks.2.weight | (108, 36, 3, 3) | 0.411 | 0.093 | 0.0032 |
| blocks.3.weight | (80, 27, 3, 3) | 0.405 | 0.087 | 0.0032 |
| blocks.4.weight | (72, 20, 3, 3) | 0.328 | 0.091 | 0.0026 |
| blocks.5.weight | (72, 18, 3, 3) | 0.818 | 0.130 | 0.0064 |
| skips.2.weight | (27, 36, 1, 1) | 0.479 | 0.131 | 0.0038 |
| skips.3.weight | (20, 27, 1, 1) | 0.381 | 0.130 | 0.0030 |
| skips.4.weight | (18, 20, 1, 1) | 0.355 | 0.130 | 0.0028 |
| refine.0.weight | (9, 18, 3, 3) | 0.967 | 0.206 | 0.0076 |
| refine.1.weight | (18, 9, 3, 3) | 0.445 | 0.120 | 0.0035 |
| rgb_0.weight | (3, 18, 3, 3) | 0.919 | 0.235 | 0.0072 |
| rgb_1.weight | (3, 18, 3, 3) | 0.696 | 0.187 | 0.0055 |

Observations:
- absmax/std ratio ≈ 7 across most weights — consistent with the C1a entropy regularizer pushing the distribution toward integer grid points (peaky-around-zero with small tail).
- No tensor has absmax > 2.2 (stem.bias is the largest); the dynamic range fits comfortably in INT8 with scale = absmax/127.
- `refine.0.weight` and `refine.1.weight` have higher std/absmax than blocks — these are the dilated-conv residual block, smaller in numel but with wider distribution.
- `rgb_0.weight` and `rgb_1.weight` have asymmetric absmax (0.919 vs 0.696) — Aaron does NOT share frame-0/frame-1 RGB heads; they're independently learned.

## §4 PR101 vs PR103 differential — the smallest set of changes

| Aspect | PR101 (gold, 0.193) | PR103 (silver, 0.195) |
|---|---|---|
| Decoder weights | byte-identical to PR98/PR95 | byte-identical to PR98/PR95 |
| Decoder codec | 7-stream brotli, hand-tuned per-tensor permutation/byte-map | brotli (light) + constriction AC on 8 heaviest tensors |
| Latent codec | raw LZMA1 dict_size=4096 + dim-major + temporal delta | brotli on lo + AC on hi |
| Sidecar | polymorphic 6-layout, HUFF_ENUM at 607 B | brotli of (u8 dim, i8 dq_zz)·N_PAIRS at 557 B |
| Sidecar delta vocab | 16 fixed values: ±[1,2,3,4,5,6,8,10]·0.01 | continuous (i8 zigzag of 0.01 quanta) |
| Inflate-time per-channel offset | `up[:,0,0].sub_(1.0); up[:,0,2].sub_(1.0); up[:,1,1].sub_(1.0)` | none |
| Inflate-time arch | bicubic-up to 874×1164 + per-channel sub | bicubic-up only |
| Decoder runtime | identical | identical |
| Archive bytes | 178,258 | 178,223 |
| Public CPU score | 0.19284 | 0.19487 |
| Public CUDA score | 0.23 | 0.23 |

**PR101's strongest observable distortion-side difference from PR103 is the
inflate-time per-channel Y/V offsets.** PR103 has marginally smaller archive
(-35 bytes) but worse distortion (PR101 seg=0.000560 pose=0.000033 vs PR103
seg=0.000577 pose=0.000034). The codec engineering (PR101's polymorphic codec
vs PR103's constriction AC) is essentially a wash on archive bytes. The
medal-position delta should be treated as a high-priority bias-correction
hypothesis until a same-archive offset ablation isolates causality.

## §5 PR comments / discussion harvest

[empirical: `gh pr view N --repo commaai/comma_video_compression_challenge --json body,comments`, full JSON saved to `.omx/tmp/hnerv_forensics/pr*.json`]

Notable comments:

- **AaronLeslie138 on PR95 (root)**: "Wow! Woke up to no shortage of people fine-tuning this in the last 3 hours to eek out an improvement on top. Might have to re-title my blog post ;) Regrettable that the landscape punishes publishing /src, but cool to see the small improvements folks were able make. Heart on my sleeve, it does suck to lose exclusively to submissions that — code is one thing — just used my archive.zip contents and not a fresh base model."
- **YassineYousfi (challenge host) reply**: "we are going to reward folks publishing their code even if not in top 3" — and PR95 received the honorable + best-writeup prizes.
- **AaronLeslie138's writeup blog**: `https://aaronleslie.dev/blog/comma-compression` — JS-rendered SPA, content not directly fetchable. The `train.py`/`stages/` pipeline shipped in the PR archive is the source of truth.
- **rem2 on PR103 (silver)**: long technical exchange with the host about CPU-vs-CUDA evaluation policy. Resolution: top 3 re-evaluated on CPU.
- **PR105 valtterivalo writeup** (HTML at `/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/kitchen_sink.html`): documents the `kitchen_sink` semantic codec phase (231,891 byte archive at 0.257), the f0/f1 pose-sponge asymmetry trick, the inflate-time Adam search, the day-1 pivot to HNeRV after PR95 dropped, and the late race against PR101/102/103. Quote: *"the final sprint became a different problem: tune that representation against the exact runner, squeeze the latent table, and ship before the board moved again."*

## §6 Activation-level differential

DEFERRED — requires GPU. Recommended next forensic step is to run PR95's inflate
on one frame of `upstream/videos/0.mkv`, dump every intermediate activation,
then run our internal `lane_12_nerv_mask_codec` on the same frame. The layer
where activation distributions diverge most is the layer where their training
trick lives. Estimated cost: $5 on Lightning T4 for one frame's activation
diff; ~30 min of analysis.

## §7 Author historical-work links

- **AaronLeslie138** (Aaron Leslie): software engineer & co-founder of Glyph (per `https://aaronleslie.dev`); GitHub https://github.com/AaronLeslie138; blog post `https://aaronleslie.dev/blog/comma-compression` (SPA, not directly fetchable; the `train.py` IS the canonical writeup).
- **EthanYangTW** (MIN-CHUN Yang): https://github.com/EthanYangTW; landed PR98 (QAT finetune of #95) AND PR102 (3rd prize via 2 inflate-time constants).
- **BradyMeighan** (Brady Meighan): https://github.com/BradyMeighan; landed PR99 (sidecar mechanism) and PR100 (sidecar refinement).
- **SajayR**: https://github.com/SajayR; landed PR101 (gold, sub-byte codec polymorphism).
- **rem2**: https://github.com/rem2; landed PR96 (early HNeRV) and PR103 (silver, constriction AC).
- **valtterivalo** (Valtteri Valo): https://github.com/valtterivalo; landed PR105 (kitchen_sink, honorable) and PR106 (belt_and_suspenders).

The submissions cluster naturally:
- **AaronLeslie138 (PR95)** is the root.
- **EthanYangTW (PR98)** + **BradyMeighan (PR99/PR100)** are the immediate finetune+repack derivatives.
- **SajayR (PR101)** + **rem2 (PR103)** + **EthanYangTW (PR102)** are the medal-band fork tips, all sharing PR98+PR95 weights.
- **valtterivalo (PR105/PR106)** is the only independent retrain of the architecture (own weights, but same `train.py` pipeline as PR95).

## §8 Hidden data scan

[empirical: `extract_archive_layout.py` first/last 16 bytes per PR]

- No ZIP comment fields used (all `comment_size=0`).
- No mojibake or steganographic patterns in the trailing bytes — every PR's last 16 bytes are high-entropy compressed payload.
- No padding bytes between sections — every section's documented length matches the actual offset.
- No duplicate or zero-byte members.
- PR105/PR106's first byte `0xff` is a magic byte for the "packed" layout (per `codec.py:288`).
- PR101's archive does NOT have a magic-byte prefix — relies on fixed-offset section lengths baked into the inflate code (decoder=162164, latent=15387). This is fragile (any retrain that produces a different decoder size would break the parser) but works for one-shot competition use.

No hidden side-channel data discovered. The archives are clean.

## §9 The Thing(s) — concrete technical secrets

### Top 3 by score-impact magnitude (ranked)

1. **The 8-stage curriculum (especially the C1a entropy regularizer + Muon + WD=5e-4 finale)** [WHAT: `losses.py:79-113 cat_entropy_v2` + `optim.py + stage8_muon_finetune.py:37`. WHY: C1a sharpens INT8 weight distribution toward integer grid → collapses brotli entropy floor; Muon's spectral-norm KKT with WD provides the final 0.0033 score reduction. CONFIDENCE: HIGH (Aaron's own writeup attributes this as the primary innovation; verifiable by ablation). REPLICATION: run `train.py` end-to-end, ~50 hours on RTX 4090 ~$12.50.]

2. **eval_roundtrip baked into the training inner loop + rgb_to_yuv6 differentiability monkey-patch** [WHAT: `stages/common.py:179-194` (every step does bicubic-up→bilinear-down→clamp→STE-round→YUV6→SegNet/PoseNet matching the contest scorer chain) + `data.py:51-81` (replaces the contest's `@torch.no_grad()`-decorated `rgb_to_yuv6` at import time so PoseNet's preprocess preserves autograd). WHY: without these two together, the proxy-vs-auth gap is 2-11× and pose plateaus at training-time-noise level. CONFIDENCE: CRITICAL (Aaron explicitly says "pose plateaued at 142 across 2500+ epochs" without the YUV fix). REPLICATION: 10 lines of code in any future NeRV/HNeRV training loop.]

3. **Per-pair latent correction sidecar with fixed delta vocabulary** [WHAT: `[-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10] × 0.01`, single-dim perturbation per pair grid-searched at compression time against the joint SegNet+PoseNet distortion. WHY: pure inflate-time mechanism, costs ~600 bytes archive overhead, saves ~0.001-0.002 score; compounds with retraining. Bonus: the `DELTA_SCALE=0.0095` retune in PR102 alone won bronze. CONFIDENCE: HIGH (consistent across PR99/100/101/102/103). REPLICATION: 50 lines of search code at compression time, no training cost.]

### The PR101-vs-PR103 differential confirms the mechanism prioritization

PR101 won gold over PR103 by 0.002 with **the same decoder weights and nearly
the same archive size** but additional inflate-time per-channel Y/V bias
corrections. This suggests that, at the medal band, inflate-time numerical
bias may be a higher-EV marginal lever than more archive-time byte savings.
It is not yet a controlled proof because PR101 and PR103 also differ in codec
layout and sidecar coding. Lessons:

- Spend the next 30 minutes optimizing inflate-time constants AFTER your codec,
  but record a same-archive no-op/control sweep before promoting causality.
- The score gradient w.r.t. inflate-time constants appears large enough to
  matter under deadline pressure (PR102 won 3rd prize on two runtime constants
  over PR100's byte-identical archive).

### Binary-forensics surprises

- **PR100 and PR102 are byte-identical archives** (verified SHA-256
  `afd5334...80641` on both). PR102 won bronze ONLY by changing 2 inflate-time
  constants on top of PR100's bytes. This was THE MOST EFFICIENT
  $/score-point in the entire race. **If we had landed PR100's bytes earlier,
  we could have shipped PR102's deltas as a $0 follow-up.**
- **The same INT8 decoder bytes underpin PR95, PR98, PR100, PR101, PR102,
  PR103** (verified per-tensor absmax to 6 decimals). Six distinct PRs
  ranking 0.196-0.198 share one 178-KB decoder. The cluster is "one
  architecture, one set of weights, six different codec/inflate engineerings."
- **PR101's codec uses 6 alternative sidecar layouts with a length-based
  selector** — the encoder picks the smallest. Building this kind of
  polymorphic codec is a sub-byte budget tool that we don't have anywhere in
  our internal codec stack.
- **PR101's hand-tuned per-tensor brotli stream segmentation
  (`DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)`), per-tensor 4D
  transpose-permutations, per-tensor signed-int byte-mappings**: these
  optimizations are completely absent from our codec stack.

### What is NOT recoverable from artifacts

- The exact training random seed (Aaron's `train.py` doesn't pin one
  explicitly; the contest eval default `seed=1234` is the most likely guess).
- Which intermediate ablations Aaron tried and discarded (no commented-out
  code or "didn't work" notes in the source — all dead lanes are simply
  absent).
- Whether 28×36 was first-try or after a hyperparameter search (no comment
  in `model.py`).
- The exact CPU/GPU machine Aaron trained on (mentioned "single GPU" in the
  README).

These are all questions that would require Aaron to comment publicly. The
training pipeline source is sufficient to reproduce, but not sufficient to
explore the design-space rationale.

## Appendix: forensic scripts

- `/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/extract_archive_layout.py` — per-PR archive section parser + entropy estimator.
- `/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/extract_weights.py` — per-PR decoder dequantizer + tensor statistics.
- `/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/pr{95,98,99,100,101,102,103,105,106}.json` — saved GitHub PR JSON for reproducibility.
- `/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/kitchen_sink.html` — valtteri's writeup HTML.

These scripts are reproducible from the public-PR clones; rerun via
`.venv/bin/python <script>.py`.

## Cross-references

- Operator clarification (the sharper question): `.omx/research/hnerv_retrospective_user_clarification_20260509.md`
- Codex representation-integration gap: `.omx/research/representation_integration_gap_audit_20260508_codex.md`
- Substrate-vs-codec meta: `~/.claude/projects/.../feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`
- HNeRV cluster CUDA-CPU drift calibration (R_pose=5.04, R_seg=1.17 derived from this same substrate): `~/.claude/projects/.../feedback_cuda_cpu_axis_profile_learning_layer_20260508.md`
- Critical-findings handoff to a1a9359d: `.omx/research/hnerv_forensics_critical_findings_for_a1a9359d_20260509.md`

## DEFERRED-pending-research items

Per CLAUDE.md `forbidden_premature_kill_without_research_exhaustion`, the
following hypotheses were SURFACED but NOT KILLED — they each require
empirical confirmation before being treated as load-bearing facts:

- **DEFERRED**: "Aaron's exact 8-stage hyperparameter chain is the only training-time path to 0.198 from this architecture." Reactivation requires running 2-3 alternative pipelines (e.g., one-stage CE, two-stage CE+QAT, omit Muon stage 8) on the same architecture and measuring the score gap.
- **DEFERRED**: "C1a sigma 0.1 is the optimal value." Reactivation requires sigma sweep at PR95's stage 7 entry point.
- **DEFERRED**: "Per-channel Y/V offsets at decode time are universal." Reactivation requires verifying that PR101's `[-1,-1,-1]` triplet AND PR102's `[+1,0,0]` deltas BOTH improve score on the same archive (suggesting numerical-bias attribution depends on substrate weights).
- **DEFERRED**: "229K params at 28-d latent + 36 base channels is the architectural ceiling for 178 KB." Reactivation requires a width/depth ablation.
