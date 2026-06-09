# SNeRV + all-vehicles fidelity review vs the frozen `evaluate.py`

Date: 2026-06-09
Author: Claude MAX-REASONING fidelity-audit subagent (READ-ONLY; no code edits, no training launched).
Sister of: `.omx/research/deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md` (the HiNeRV memo this
mirrors) + `.omx/research/b1_carrier_crux_decoder_hf_fidelity_not_latents_20260609.md` (the empirical chain
that localized the HiNeRV bug to decoder HF-fidelity).

Axis discipline: EVERY numeric here is tagged. The only score authority is exact `upstream/evaluate.py` on
`[contest-CPU]` (Linux x86_64) or `[contest-CUDA]` (T4). The SNeRV d_seg=0.711 / d_pose=163.2 trace cited
below is `[macOS-MLX research-signal]` (a live MLX render scored through the exact DistortionNet on macOS) —
it is a **mechanism diagnostic, NOT a score claim**. PSNR is `[advisory only]` and is never a score. This
memo is `research_only=true` / `mechanism_update_eligible`. Per CLAUDE.md "Forbidden premature KILL": nothing
here KILLS any paradigm; it falsifies the *current default carrier configurations* at the implementation
level (Catalog #307 IMPLEMENTATION-LEVEL) and names the cheap experiments that test the reactivation paths.

---

## 0. The headline (the cross-vehicle finding the operator asked for)

**The lab's entire NeRV-family fleet is, in its default/trained configuration, the SAME skip-free
PixelShuffle+SIREN(w=30) decoder wearing ~10 different names — and it fails the contest for the SAME two
reasons it fails for HiNeRV: (1) the decoder has no residual high-frequency path, so it mean-fields to a
blurry image that collapses SegNet's argmax to ~one class (d_seg ≈ 0.5–0.7); (2) the actually-trained path
(the shared MLX harness) optimizes a reconstruction-MSE base term, which *rewards* that mean-field.** This is
the 18-shared-assumption plateau the lab already documented, now manifesting concretely as the d_seg≈0.5 HF
ceiling. Two empirical receipts pin it:

- **HiNeRV** (sister memo): live MLX render d_seg = 0.5075, d_pose = 205.8 `[macOS-MLX research-signal]`,
  PSNR plateau ~21.7 dB `[advisory only]`.
- **SNeRV** (THIS memo, NEW): the official MFU/HFR/TUB renderer at **ep22399** (a very long run) scored
  `avg_segnet_dist = 0.711`, `avg_posenet_dist = 163.2`, nonrate ~111.8 `[macOS-MLX research-signal]`
  (`/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch22399_full_video_mlx_feedback_20260604T004900Z/nerv_full_video_mlx_scorer_feedback.json`,
  field `row.full_video_mlx_scorer_response.avg_segnet_dist`). The same run reports
  `observed_segnet_distillation_weight = None` — i.e. **it was trained recon-MSE-only**, scorer-blind.

The crucial nuance vs a naive "they're all the same fake": **SNeRV's architecture is genuinely faithful**
(real orthonormal DWT with exact adjoint, real official MFU/HFR/TUB conv blocks with LeakyReLU + skip-concat
carries, real Haar receiver). SNeRV did NOT fail because the architecture is fake. It failed because (a) it
was trained against recon-MSE with the scorer distillation weight OFF, and (b) its highest-resolution skip
can be (and in the channelmean/scalarmean runs WAS) collapsed to a frame-invariant mean, re-injecting the
mean-field at the finest scale. The per-vehicle UNIQUE missing tricks differ; the SHARED mistake (recon-MSE
base + no/weak HF path) spans the fleet.

---

## 1. Per-vehicle fidelity manifest (MATCH / APPROXIMATES / MISSING per defining technique)

### 1.1 SNeRV (`src/tac/substrates/snerv_inverse_steg_carrier/`) — DEEP

SNeRV = Spectra-preserving NeRV (arXiv 2501.01681): 2D DWT splits each frame into LF (coarse) + HF (detail);
**store LF, GENERATE HF** with MFU (multi-resolution fusion) + HFR (high-frequency restorer) + TUB (temporal
upsampling block). Its explicit thesis ("NNs learn HF slower than LF") is a *direct architectural answer to
the exact spectral-bias failure* we see across the fleet. Two carrier paths exist:

| # | SNeRV defining technique | Reference (paper) | Our impl | Verdict | Evidence (file:line) |
|---|---|---|---|---|---|
| DWT | orthonormal 2D DWT + exact synthesis adjoint | Mallat 1989 §7.5; Daubechies 1988 | pywt periodization db2 + NumPy Haar receiver; adjoint rel-residual 0.0 by dot-product test | **MATCH (genuine, not fake)** | dwt.py:196-279, 310-355; `_haar_dwt2_level` 408-422 |
| store-LF | store only LL approximation, generate HF | SNeRV §3 | `SnervFrameCode.lf_quant` stores LF only; HF regenerated at decode | **MATCH** | carrier.py:736-755, decode_frame 1340-1374 |
| HF-gen (path A, "spectra_preserving") | learned MFU/HFR | SNeRV §3 | **LINEAR ridge predictor**: detail = `einsum('...i,i->...', LF_patch_features, 9-tap kernel)` — NO nonlinearity between feature-extract and readout | **APPROXIMATES (degenerate: a linear map from LF context → HF)** | carrier.py:837-843 (`_apply_linear_decoder_features`), 1035-1096 (`generate_hf_from_lf`), fit 1099-1232 |
| HF-gen (path B, "official MFU/HFR/TUB") | conv MFU + conv HFR + temporal TUB | SNeRV official `model/*.py` | real `ConvTranspose2d → cat(skip_mid) → ResidualBlocksNoBN(LeakyReLU) → ConvTranspose2d → cat(skip_high) → ResidualBlocks`; HFR heads conv1→LeakyReLU(0.1)→conv2 | **MATCH (genuine conv U-Net w/ nonlinearity + skip-concat)** | official_mfu.py:295-344 (`ResidualBlockNoBN`: conv,LeakyReLU(0.1),conv,skip-add), 480-514 (MFU.forward); mlx_renderer.py:1306-1319 (`_hfr_head_forward` LeakyReLU), 1240-1249 (`_forward_trace`) |
| MFU/HFR/TUB enum | 3 distinct primitives | SNeRV | MFU (fusion), HFR (LH/HL/HH conv heads), TUB (temporal output-2 residual) are **structurally distinct** (not enum-padding) | **MATCH** | official_mfu.py / official_hfr.py / official_tub.py separate modules; `_forward_trace` 1246-1255 |
| **skip carries** | per-frame skip features | SNeRV | `skip_mid`/`skip_high` carried into MFU; **BUT `skip_high_mode ∈ {full, shared_mean, channel_mean, scalar_mean}`** — the finest skip can be collapsed to a frame-invariant mean | **MISSING-WHEN-NOT-`full` (the finest-scale mean-field trap)** | mlx_renderer.py:745-759 (mean-collapse), carrier.py:125-127 |
| LF-conditioned HF residual (sidecar) | generate HF | (lab prototype) | **STORES** int16-LZMA HF residual; explicitly `FALSE_AUTHORITY`; blocker "snerv_hf_residual_generator_receiver_payload_not_implemented" | **NOT a generator — a stored-residual prototype (Z8-disease shape)** | lf_conditioned_hf_residual.py:27-34, 60-137, 263-264 |
| loss = frozen-scorer-margin, NO recon base | SNeRV is recon-fidelity; OUR contest needs scorer-margin | PR95 | shared MLX harness: `recon MSE (base) + optional distill + optional direct-live`; **all scorer weights default 0.0**; the ep22399 run had `observed_segnet_distillation_weight=None` | **DIVERGES (recon-MSE base; scorer terms opt-in and were OFF)** | mlx_native_train_export.py:3348-3362 (defaults 0.0), 524-529 (recon is the fallback base), `_shared/mlx_score_aware/loss.py:4` + 511-523 |
| eval_roundtrip + differentiable YUV6 | NON-NEGOTIABLE | CLAUDE.md | direct-live SegNet/PoseNet terms exist (`segnet_direct_live_*`, `pose_direct_live_*`); they CAN backprop through frozen scorer | **AVAILABLE but opt-in** | mlx_native_train_export.py:497-616 (`_snerv_score_aware_checkpoint_selection_policy`) |
| byte-closed archive → inflate → parse-back | promotion requirement | CLAUDE.md L2/L9 | path B carries export blockers: "native MLX export NOT bound to official payload", "weight mapping missing", "source-forward replay missing" | **BLOCKED for path B (research-only); path A receiver is numpy-portable** | carrier.py:308-316 (`official_mfu_hfr_tub_export_blockers`) |

**SNeRV verdict:** the most architecturally-honest carrier in the fleet (real DWT, real conv MFU/HFR/TUB).
Its failure at ep22399 is **NOT architectural fakery** — it is (a) trained recon-MSE-only (scorer weight
None) and (b) the path-B skip_high collapsed to a per-frame-invariant statistic in the channelmean runs.
Path A's HF generator is a degenerate *linear* LF→HF map that cannot synthesize boundary HF (a linear
predictor of detail coefficients from a smoothed LF patch is approximately a sharpening filter — it adds a
little edge energy but cannot invent the argmax-flipping structure SegNet keys on).

### 1.2 pact_nerv_vq (`src/tac/substrates/pact_nerv_vq/`) — DEEP

VQ-VAE per van den Oord 1711.00937: per-pair latent → nearest codebook entry → straight-through estimator,
EMA codebook update, commitment loss.

| # | VQ defining technique | Reference | Our impl | Verdict | Evidence (file:line) |
|---|---|---|---|---|---|
| nearest-codebook lookup | argmin ‖z_e − e_k‖² | vdO §3.1 | real distance + `argmin` | **MATCH** | architecture.py:136-142 |
| straight-through estimator | `z_q = z_e + (z_q − z_e).detach()` | Bengio 2013 / vdO §3.1 | exact | **MATCH (genuine, not fake)** | architecture.py:147 |
| EMA codebook update | persistent N_c / m_c with Laplace smoothing | vdO §3.2 | real EMA + Laplace smoothing decay=0.99 | **MATCH** | architecture.py:150-166 |
| commitment loss | `β‖z_e − sg(z_q)‖²` | vdO §3.1, β=0.25 | real, returned for the Lagrangian | **MATCH** | architecture.py:144, score_aware_loss.py:108-109 |
| indices entropy-coded into archive | bytes must be charged | contest | uint16 indices + brotli decoder blob; declared archive grammar | **MATCH** | archive.py:13-21, 84 |
| inflate CONSUMES indices (no-op check) | bytes must change frames | Catalog #105 | inflate replaces per-pair latents with codebook[index] | **MATCH (indices are consumed)** | inflate.py:49-58 |
| loss = frozen-scorer, NO recon base | PR95 geometry | PR95 | `100·seg + γ·√pose + commitment` via `score_pair_components_dispatch`; eval_roundtrip mandatory; **NO recon-MSE base** | **MATCH (good objective shape)** | score_aware_loss.py:84-109, 72-76 |
| **decoder HF path** | skip / PE / refine | HiNeRV/PR95 | `_DsUpBlock = PixelShuffle(sin30(DepthSepConv))` — **NO skip, NO PE, NO refine** | **MISSING (the universal HF deficiency)** | architecture.py:70-78, 261-262 |

**pact_nerv_vq verdict:** the VQ machinery is GENUINE (not a fake codebook), the bytes are genuinely
consumed, and the PyTorch loss has the RIGHT objective shape (frozen-scorer, no recon base — better than the
shared MLX harness). Its single fatal gap is the **skip-free PixelShuffle+sin decoder**: it will mean-field
exactly like HiNeRV regardless of how good the VQ is, because the VQ compresses the *latent*, not the
*decoder's ability to synthesize HF*. Lane `lane_pact_nerv_vq_l0_scaffold_20260520` is `research_only=true`
L1; `lane_pact_nerv_vq_l1_long_run_mlx_local_20260528` exists — note its MLX path (mlx_renderer.py) would run
through the shared harness (recon-MSE base), re-introducing the objective problem the PyTorch loss avoids.

### 1.3 Triage table — the rest of the active NeRV family

Decoder structure verified by reading each `architecture.py` (and `mlx_renderer.py` where present). `loss_kind`
is the PyTorch `score_aware_loss.py` (all use `score_pair_components_dispatch` = frozen scorer, eval_roundtrip
mandatory, beta_seg=100, **no recon-MSE base** — GOOD shape; the recon-MSE problem is the SHARED MLX harness,
which only SNeRV/HiNeRV/pact_nerv_vq/nirvana_cascading actually train through at scale).

| vehicle | has_skip | has_PE/coord | loss_kind (pytorch) | has_refine | trainable path | will_mean_field? | evidence (file:line) |
|---|---|---|---|---|---|---|---|
| **ds_nerv** | NO | NO | frozen-scorer (good) | NO | pytorch-only L0/L1 | **YES** | architecture.py:107-118 (`_DsUpBlock`=PixelShuffle(sin(DepthSep))); loss 75-84 |
| **ff_nerv** | NO | NO (freq grid) | frozen-scorer (good) | NO | pytorch-only | **YES + band-limited by construction** | architecture.py:4-31 (predicts 64×64 DCT grid → IDCT → interp to 384×512: HF *structurally absent*), 115-121 |
| **tc_nerv** | NO | NO | frozen-scorer (good) | NO | pytorch-only | **YES** | architecture.py:110-126 (`_UpBlock`=Conv→sin→PixelShuffle), 213 (final bilinear resize only) |
| **block_nerv** | NO | NO | frozen-scorer (good) | NO (per-pair LoRA latent delta only) | pytorch-only | **YES** (LoRA is a *latent* residual, not a decoder HF path) | architecture.py:92-106, docstring 13-30 (LoRA on embed-grid, flagged "TOO BIG/wrong") |
| **coin_plus_plus** | NO | **YES (coord-MLP: x,y,t input)** | frozen-scorer (good) | NO | pytorch-only | **PARTIAL** — has coordinate conditioning (no H2 deficiency) but no residual; SIREN-MLP spectral bias still likely | architecture.py:18, 101-123 (FiLM-modulated SIREN coord-MLP), 134-135 |
| **sane_hnerv** | **CLAIMED, NOT IMPLEMENTED** | NO | frozen-scorer (good) | NO | pytorch-only L2 | **YES** | docstring 5,27 claims "bilinear-skip"; `_UpBlock.forward` 138-139 = `shuffle(act(conv(x)))` NO skip; main forward 236-237 bare loop NO skip accumulation |
| **boost_nerv** | NO (base) | NO | frozen-scorer (good) | **YES but gain-clamped ±0.1** | pytorch-only L1 | **LIKELY** — tiny clamped residual heads on a ds_nerv base can't escape a ±0.1 correction band | architecture.py:5-32, 26 (`residual gain clamped to [-0.1,0.1]`), 128-135 |
| **nirvana** | NO | NO | frozen-scorer (good) | NO (patch-stitched) | pytorch-only L1 | **YES** (patchwise ds_nerv: each 4×4 patch decoded by PixelShuffle+sin, stitched) | architecture.py:18-26, 123-130 |
| **nirvana_cascading_nerv** | residual cascade | NO | (MLX scaffold) | **STORED int8 residuals** | MLX scaffold (renderer class "lands Phase 2") | **N/A — Z8-disease shape** (HF residuals are STORED bytes, not generated) | mlx_renderer.py:2-7 (renderer class not landed), 130-131 ("residuals are STORED bytes, not learned per-pair") |
| **hi_nerv** (sister memo) | NO (default) | OFF (`use_hierarchical_feature_grid=False`) | shared MLX harness (recon base) | NO | MLX (trained) | **YES (empirically d_seg=0.5075)** | sister memo §1; architecture.py:136-140 |

---

## 2. Ranked root-cause hypotheses (per vehicle, with mechanism + math)

### 2.1 SNeRV

**S1 (TOP) — trained recon-MSE-only with the scorer distillation weight OFF; confidence HIGH.** The ep22399
official-renderer run reports `observed_segnet_distillation_weight = None` and `avg_segnet_dist = 0.711`
`[macOS-MLX research-signal]`. The shared MLX harness base loss is per-pixel reconstruction MSE
(`_shared/mlx_score_aware/loss.py:511-523`), whose minimizer is the conditional mean → blur. With scorer
terms at 0.0 (the default, `mlx_native_train_export.py:3348-3362`), SNeRV optimized pure reconstruction and
mean-fielded *even though its architecture is capable of HF*. This is the dominant cause and it is the SAME
objective-shape mistake (H3) the HiNeRV memo identified — shared between the two MLX-trained carriers by
construction (they share the harness).

**S2 — `skip_high` collapsed to a frame-invariant mean (`channel_mean`/`scalar_mean`); confidence HIGH for
those runs.** The official renderer can replace the highest-resolution skip with `mean(skip_high, axis=0)`
(shared across all frames) or even a single scalar (mlx_renderer.py:745-759). The finest skip carries the
sharpest detail; collapsing it to a per-frame-invariant statistic guarantees the finest scale is identical
across frames = a static blob at the resolution SegNet's stride-2 stem actually resolves. The run dir name
`snerv_channelmean_full600_*` confirms channelmean runs were the operating mode. This is a *byte-saving*
choice (one shared skip plane instead of 600) that directly trades away the boundary HF that d_seg needs —
the classic "optimize bytes as an independent knob" anti-pattern (CLAUDE.md "Results must become system
intelligence").

**S3 — path-A HF generator is a degenerate LINEAR LF→HF map; confidence HIGH (for path A only).** Path A
predicts each detail coefficient as `einsum('...i,i->...', LF_patch_features, kernel)` (carrier.py:837-843)
— a linear function of a (smoothed, upsampled) LF patch. A linear map from low-frequency context can apply a
fixed sharpening kernel but cannot synthesize the *new* high-frequency structure (occlusion edges, lane
markings, sign boundaries) that is, by definition, not linearly predictable from the LF approximation. This
is the SNeRV-specific analog of the "no nonlinear HF synthesis" deficiency. Path B (conv MFU/HFR with
LeakyReLU) does NOT have this problem — which is why path B is the right SNeRV vehicle, but path B is the one
trained recon-only (S1) and export-blocked.

**S4 — under-training is NOT the cause; confidence HIGH.** ep22399 is a very long run; d_seg got WORSE-than-
HiNeRV (0.71 vs 0.51), not better. More epochs against the recon-MSE objective sharpen reconstruction PSNR
but do not move the SegNet argmax (PSNR ≠ d_seg — the lab's documented lesson). S4 is a confound to control,
not the driver.

### 2.2 pact_nerv_vq

**V1 (TOP) — skip-free PixelShuffle+sin decoder mean-fields regardless of VQ quality; confidence HIGH.** The
VQ compresses the *latent code* (good byte story) but the *decoder* is `_DsUpBlock` (architecture.py:70-78)
with no skip/PE/refine. The mean-field local minimum is reached through the decoder, upstream of which the
VQ sits — so a perfect codebook still feeds a decoder that cannot synthesize HF. Mechanism identical to
HiNeRV H1.

**V2 — w=30 SIREN on coordinate-free feature maps; confidence MEDIUM.** Same H4 spectral-bias trap as HiNeRV
(`sin_frequency=30.0`, architecture.py:38, 56-57): high-ω sin on slowly-varying feature maps aliases into a
near-DC regime. HARD-EARNED for coordinate-MLP SIREN, CARGO-CULTED for a skip-free feature-map NeRV.

**V3 (latent, not yet a problem) — codebook collapse under byte pressure.** With codebook_size=512 and EMA,
if too few codes are used the per-pair latents collapse to a handful of vectors → all pairs render near-
identical frames → d_pose (two-frame motion) and per-pair d_seg both suffer. Not observed (no long run), but
the canonical VQ-VAE failure to watch (Laplace smoothing at 161 mitigates dead codes but not active collapse).

### 2.3 The PyTorch-only family (ds/ff/tc/block/sane_hnerv/boost/nirvana/coin++)

**Shared root cause — skip-free PixelShuffle+sin decoder (H1) + w=30 (H4).** Identical mechanism for all
except coin++ (which has coordinate input, so no H2, but still no residual) and ff_nerv (whose 64×64 DCT
grid is *band-limited by construction* — it physically cannot store the HF, a STRICTLY WORSE variant of the
same disease, root cause "intrinsic low-pass").

**sane_hnerv-specific — the bilinear-skip is documented but absent (Catalog #307 documentation-fake).** The
docstring (architecture.py:5,27) advertises "canonical HNeRV with bilinear-skip"; the code (138-139, 236-237)
has no skip. A reader trusting the docstring would believe this is the one fleet member with the HiNeRV-
reference HF path; it is not. This is the same "named X but isn't" pattern the HiNeRV memo found at §1.1 —
now confirmed to recur in a *second* substrate, which elevates it from a one-off to a fleet pattern.

---

## 3. THE CROSS-VEHICLE SHARED ENGINEERING-MISTAKE FINDING (the headline, expanded)

There are **two shared mistake classes**, orthogonal, both spanning the fleet:

### Shared mistake A — "skip-free NeRV decoder wearing N names" (architecture surface)

Every vehicle except SNeRV-path-B and coin++ uses the identical decoder atom: `PixelShuffle(2)(sin(w=30 ·
conv(x)))` with **no bilinear-skip, no positional/coordinate encoding, no terminal HF refine**. The PR95
reference decoder that WON (`model.py:46-51`, cited in the HiNeRV memo §1) is
`sin(PixelShuffle(conv(x)) + bilinear_up(x))` followed by `x + 0.1·sin(refine(x))`. The residual/skip path is
*precisely* the mechanism that lets the optimizer escape the blurry mean-field; without it the global MSE
minimizer (the DC/mean image) is the easy attractor, and SegNet's argmax collapses to one class → d_seg≈0.5.
This is the 18-shared-assumption plateau (CLAUDE.md) at the decoder-architecture layer: ~10 "different
substrates" are structurally the same skip-free implementation under different names. **The
documentation-fakes (sane_hnerv claiming a skip it lacks; HiNeRV named after a paper whose two defining ideas
are OFF by default) are the same disease manifesting at the docstring layer.**

### Shared mistake B — "recon-MSE base in the shared MLX harness" (objective surface)

The carriers that actually train at scale (HiNeRV, SNeRV, and pact_nerv_vq's MLX long-run lane) go through
`src/tac/substrates/_shared/mlx_score_aware/loss.py`, whose base term is per-pixel reconstruction MSE
(line 4, 511-523) with the SegNet/PoseNet distillation terms **opt-in and defaulting to 0.0**
(`snerv .../mlx_native_train_export.py:3348-3362`). MSE *rewards* the mean-field. The decisive receipt:
SNeRV ep22399 ran with `observed_segnet_distillation_weight = None` and landed d_seg=0.71. So even a
faithful architecture (SNeRV path B) is dragged to the mean-field by a recon-MSE objective. The PR95 winner
had NO recon term — its primary signal was a frozen-SegNet argmax-margin surrogate weighted 100×. **The
PyTorch-side `score_aware_loss.py` modules across the fleet ALREADY have the right shape (frozen-scorer, no
recon base) — but they are L0/L1 sketches that never trained at scale; the MLX harness that DID train is the
one carrying the recon-MSE base.** This is the orthogonal half of the plateau, at the objective layer.

### The single shared fix that lifts the whole family

`residual HF path (bilinear-skip + terminal refine) + coordinate/grid PE + frozen-scorer-margin loss with NO
recon-MSE base`. Concretely: (1) make the upsample atom `sin(PixelShuffle(conv(x)) + skip(bilinear_up(x)))`
+ add `x + 0.1·sin(refine(x))` before the heads (one shared `_UpBlock` edit replicated across the fleet, or
adopt SNeRV path B's conv-MFU which already has the skip-concat + nonlinearity); (2) inject a coordinate/grid
PE so the decoder has location-specific HF; (3) in the MLX harness, anneal the recon-MSE base weight toward 0
and make the frozen-SegNet direct-live margin the primary term (SNeRV's `segnet_direct_live_*` knobs already
exist — set them > 0 and the recon weight low). SNeRV path B is the closest to "free" because it already has
the residual conv blocks — its ONLY missing pieces are (a) the scorer weight (turn it on) and (b) keep
`skip_high_mode='full'` (don't mean-collapse the finest skip).

### Per-vehicle UNIQUE required trick our impl lacks (distinct from the shared fix)

- **SNeRV path A:** a *nonlinear* HF generator. The current linear LF→HF ridge map cannot synthesize new
  boundary HF (math: a linear function of the LF approximation is a fixed filter; boundary HF is not in the
  LF row-space). Replace with path B's conv-HFR (already exists) or a small nonlinear MLP per subband.
- **SNeRV both paths:** keep `skip_high_mode='full'` — the mean-collapse modes are a finest-scale mean-field
  switch disguised as a byte-saver.
- **pact_nerv_vq:** the VQ is fine; it needs the residual decoder (shared fix) AND a codebook-utilization
  monitor to pre-empt collapse (V3).
- **ff_nerv:** the 64×64 DCT band-limit must be widened or the IDCT replaced — the substrate currently cannot
  represent boundary HF *by construction* regardless of training.
- **nirvana_cascading:** it STORES int8 HF residuals (Z8-disease) — it must GENERATE them or it pays the
  byte cost the SNeRV paradigm was designed to avoid.
- **coin_plus_plus:** has coordinate PE (good); needs a residual/skip across MLP layers + likely a
  multi-frequency Fourier feature input to beat SIREN's single-ω spectral bias.
- **HiNeRV (sister memo):** turn ON the hierarchical grid PE (the literal HiNeRV defining feature, currently
  `use_hierarchical_feature_grid=False`) + add the bilinear-skip + refine.

---

## 4. Ranked cheapest-falsifiable fixes (per vehicle; all local MLX/CPU, `[macOS-MLX research-signal]`, $0)

Per "Forbidden premature KILL" these are research proposals; promotion requires byte-closed archive + paired
CPU/CUDA. Each states a falsifiable prediction in d_seg / PSNR terms.

### G1 (HIGHEST VALUE, CHEAPEST, fleet-wide) — Re-train SNeRV path B with scorer weight ON + `skip_high_mode='full'`
- **Change:** in the existing SNeRV official-MFU/HFR/TUB long-train config, set
  `segnet_direct_live_distillation_weight > 0` (and a class-balanced subcontrol, e.g.
  `segnet_direct_live_class_balanced_hinge_weight`), set `pose_direct_live_distillation_weight > 0`, anneal
  `recon_weight` toward a small anchor, and force `official_skip_high_mode='full'`. NO architecture change —
  SNeRV path B already has the residual conv blocks + nonlinearity.
- **Predicted mechanism (S1+S2):** removing the recon-MSE mean-field reward + keeping the finest skip per-
  frame lets the faithful conv decoder use its HF capacity on boundary structure. **Prediction: d_seg drops
  from 0.71 toward < 0.2 and avg_posenet_dist drops sharply within the same epoch budget, scored live through
  DistortionNet.** If d_seg stays ≈ 0.5–0.7 with the scorer weight on and skip_high='full', S1/S2 are
  falsified and the linear path-A-style HF deficiency or a deeper export-binding bug dominates.
- **Why first:** it is a *config-only* change on the most-faithful, already-built carrier, and it directly
  tests the dominant shared mistake (recon-MSE base) on a real architecture. Cheapest information per dollar.

### G2 — Add the PR95 bilinear-skip + refine to the shared `_UpBlock` and re-run ONE PyTorch carrier (ds_nerv or pact_nerv_vq)
- **Change:** `_DsUpBlock.forward → shuffle(act(conv(x))) + skip(bilinear_up(x))`; add `x + 0.1·sin(refine(x))`
  before the heads. ~15 LOC, replicated to whichever carrier you train. (The PyTorch loss is already the
  right shape — frozen-scorer, no recon base — so this isolates the *architecture* variable.)
- **Predicted mechanism (Shared mistake A / H1):** the identity path becomes free; the conv branch learns
  only HF corrections; PSNR breaks the ~21.7 dB plateau (expect ≥ 28 dB), SegNet stops collapsing.
  **Prediction: d_seg drops from ~0.5 toward < 0.1 and PSNR > 28 dB within the same budget.** If d_seg stays
  ~0.5 with the skip added, H1 is falsified and the objective (if MLX) or PE (H2) dominates → escalate to G3.
- **Note:** this is the SAME experiment the HiNeRV memo's F1 proposes; running it on a *second* carrier
  confirms whether the fix generalizes across the fleet (it should, since the mistake is shared).

### G3 — Coordinate/grid PE ablation (2×2: skip {off,on} × PE {off,on}) on one carrier
- **Change:** add a coordinate PE input (or turn on HiNeRV's `use_hierarchical_feature_grid`) and cross with
  the skip from G2. 4 short runs.
- **Predicted mechanism (H1 vs H2):** isolates whether residual *connectivity* or coordinate *conditioning*
  is the binding constraint. **Prediction: skip-on dominates; PE-on adds a further d_seg reduction
  (HiNeRV's own ablation = +2.5 dB PSNR).** coin_plus_plus is the natural PE-on/skip-off control (it already
  has coordinate input) — if coin++ ALSO mean-fields, that's evidence H1 (skip) is the binding term.

### G4 — SNeRV path-A nonlinearity probe (contract-free recon fit)
- **Change:** swap path-A's linear `einsum` HF readout for a 1-hidden-layer MLP-per-subband (or just route
  path A through path B's conv-HFR). Fit on real frames; measure recon PSNR of the *generated* HF subbands vs
  the *true* DWT detail.
- **Predicted mechanism (S3):** a nonlinear map can capture boundary HF a linear map cannot. **Prediction:
  generated-HF PSNR and overall recon PSNR rise materially vs the linear path; if they don't, the LF really
  doesn't carry the information and HF must be *stored* (Z8) or *seeded* rather than generated.**

### G5 — ff_nerv band-width / nirvana_cascading "generate not store" probes (lower priority)
- **ff_nerv:** widen the DCT grid (64→128) or replace IDCT with a learned upsampler; **prediction: d_seg
  improves monotonically with retained band until the PixelShuffle decoder's own HF ceiling (G2) binds.**
- **nirvana_cascading:** measure the stored-residual byte cost vs a generated-residual variant; **prediction:
  the stored residuals dominate the archive (Z8-disease) and a generated cascade is required for the byte
  term.**

### G6 — Control for under-training (diagnostic, ~$0) BEFORE any architecture verdict
- Read the recon-PSNR-vs-epoch slope for the existing SNeRV ep18199→ep22399 runs (both already on the SSD).
  d_seg got WORSE 0.68→0.71 across those two checkpoints `[macOS-MLX research-signal]`, which already
  indicates an objective/architecture ceiling, not a budget problem — but confirm the PSNR slope is flat
  before concluding.

---

## 5. Honest scope / limits

- **Verified by reading source (file:line cited):** SNeRV DWT adjoint exactness, the linear path-A HF
  generator, the conv path-B MFU/HFR/TUB with LeakyReLU + skip-concat, the `skip_high_mode` mean-collapse
  options, the shared MLX harness recon-MSE base + opt-in scorer terms (default 0.0), pact_nerv_vq's genuine
  VQ (STE + EMA + commitment) and genuine index consumption at inflate, and the skip-free PixelShuffle+sin
  decoder across ds/ff/tc/block/sane_hnerv/boost/nirvana. sane_hnerv's docstring-claimed-but-absent
  bilinear-skip is verified at the code level (forward methods read in full).
- **Verified by telemetry (`[macOS-MLX research-signal]`, NOT a score):** SNeRV ep22399 avg_segnet_dist=0.711,
  avg_posenet_dist=163.2, observed_segnet_distillation_weight=None
  (`snerv_epoch22399_full_video_mlx_feedback_20260604T004900Z/nerv_full_video_mlx_scorer_feedback.json`);
  ep18199 avg_segnet_dist=0.6796.
- **INFERRED (mechanism argument, not measured here):** that the proposed G1–G4 fixes will move d_seg
  (the d_seg/PSNR predictions are falsifiable hypotheses, not measurements); that the linear path-A map
  cannot synthesize boundary HF (linear-algebra argument); that coin++/ff_nerv behave as their structure
  implies (no long runs read).
- **NOT done (out of scope for a read-only audit):** no training, no GPU, no paid dispatch, no edits to any
  carrier source or `upstream/`. No byte-closed archive built. No `[contest-CPU]`/`[contest-CUDA]` score
  produced — every fidelity verdict is a *mechanism* verdict, not a score verdict.
- **Authority:** everything in this memo is `[macOS-CPU advisory]` / `mechanism_update_eligible`. It updates
  the next-experiment routing; it does NOT promote, rank, kill, or close any lane (per CLAUDE.md
  "Meta-Lagrangian/Pareto solver" + "Forbidden premature KILL").

### One-line per-vehicle verdicts

- **SNeRV:** faithful architecture (real DWT + conv MFU/HFR/TUB); failed at ep22399 because trained
  recon-MSE-only (scorer weight None) + finest skip mean-collapsed — fix is config-only (G1), not a rebuild.
- **pact_nerv_vq:** genuine VQ + genuine index consumption + correct PyTorch loss shape; one fatal gap = the
  skip-free decoder will mean-field (G2).
- **ds_nerv / tc_nerv / nirvana:** skip-free PixelShuffle+sin NeRV; will mean-field (G2).
- **ff_nerv:** band-limited 64×64 DCT grid CANNOT represent boundary HF by construction — strictly worse
  variant of the disease.
- **block_nerv:** ds_nerv decoder + per-pair LoRA *latent* delta (not a decoder HF path); will mean-field.
- **boost_nerv:** ds_nerv base + ±0.1 gain-clamped residual heads — clamp too tight to escape the mean-field.
- **sane_hnerv:** docstring claims a bilinear-skip it does NOT implement (Catalog #307 documentation-fake);
  will mean-field.
- **coin_plus_plus:** has coordinate PE (no H2), but no residual + single-ω SIREN — partial; the PE-on/skip-off
  control for G3.
- **nirvana_cascading_nerv:** STORES int8 HF residuals (Z8-disease) and the renderer class isn't landed —
  must generate, not store.
- **hi_nerv (sister memo):** skip-free + grid-PE OFF + recon-MSE base; empirically d_seg=0.5075.
