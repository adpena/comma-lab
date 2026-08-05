---
arm: ddm_ar11 (arm died at spawn — codex usage limit; this receipt is MAIN-AUTHORED per the
  operator complete-read directive 2026-08-05 "papers that are most on point should be read
  completely and harvested very deeply rather than just using the abstract")
paper: "arXiv 2608.01306 — SPAE: Spectrally Guided Autoencoder for Pretrained Visual Latents"
authors: "Huang, Hong, Li, Dai, Wang, Wang, Song, Wang, Sun, Xu, Zhu"
submitted: 2026-08-02 · cs.CV
utc: 2026-08-05
read_depth: FULL-TEXT (3 WebFetch passes over /html: method · spectral analysis · experiments)
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[paper-crosswalk scorer-free]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# AR11 Receipt — SPAE (arXiv 2608.01306), MAIN deep read

## Answer First

SPAE is a **latent-adaptation** paper: VFM (DINOv2/SigLIP2) latents are hard for a DiT to
generate because DiT-generated latents spectrally undershoot encoder latents from
mid-frequency up, and HF is **diffusely spread across channels, entangled with semantics**
(their Fig 2c: leading/intermediate/trailing channel groups have near-identical radial
power spectra). Cure: a **64-dim bottleneck** (1-block compressor → 6-block lifting →
decoder; frozen encoder) + **contiguous SUFFIX channel masking** during finetune
(k ∈ {0,12,16,24,32} w.p. {0.60,0.15,0.10,0.10,0.05}) that forces semantics into early
channels and detail into late ones. Measured: masking costs 0.68 PSNR but moves gFID
8.30→5.00 and IS 97.5→151.2; bottleneck-dim sweep shows generation degrades
monotonically 64→128 even as reconstruction improves (gFID 5.00→7.18 vs PSNR
30.28→31.35).

**Honest provenance note:** suffix masking is structurally **nested dropout**
(Rippel et al. 2014) applied to VFM-latent adaptation; the training trick is the
transferable value regardless of the paper's framing.

**Transfer boundary (the AR8 lesson, restated for this paper):** SPAE lives in a learned
latent space feeding a *generative* model; our surface is a *frozen discriminative*
scorer behind a fixed resize/uint8 operator. Their causal claim
(spectral alignment → gFID) does NOT transfer. What transfers: (1) the suffix-masking
TRAINING LEVER for prioritized channel ordering, (2) the per-channel-group radial-power
DIAGNOSTIC as a $0 instrument, (3) external confirmation that compact-semantic-code +
decoder-regenerated-HF (ship the description, not the solve) is the winning shape —
which our od9 ordering-law verdict already reached internally (SHIP-THE-SOLVE
RATE-DEAD at 1,214,007 B projected).

## Ranked Crosswalk (chartered consumers)

| rank | disposition | claim → Pact surface | named consumer | falsifier | cost |
|---:|---|---|---|---|---|
| 1 | **ADOPT-AS-LEVER (build, A/B-gated)** | Nested/suffix masking on the TR1 token-channel axis (and/or QA83 factorized-head rank components): train under randomized suffix masking → the token field acquires a TRAINED-IN graceful-truncation ordering → post-hoc drop becomes a CONTINUOUS rate knob instead of the fixed drop-level races. Composes directly with #869 token-by-token waterfill and the gr1 granularity re-race: today's drop knee is measured on an UNORDERED field; SPAE's Table 4b says ordering is trainable at ~0.7 PSNR-equivalent cost with a large downstream win. | TR1 burn arm-matrix (next window; DSL lever, default-off) · #869 · gr1 successor | Matched-window A/B: masked-trained truncation curve fails to beat the unmasked field's measured drop knee at equal bytes AND equal epochs → lever dead at INSTANCE scope. Second falsifier: seg-hold violated during masked training (lg1 guard fires) → needs class-protected masking or dies. | lever build + one matched A/B window; NOT a mid-burn change |
| 2 | **ADOPT-AS-INSTRUMENT ($0)** | Per-channel-group radial power spectrum of RENDERED frames vs C1 SOLVE frames (their Fig 2 protocol: FFT → radial average → compare, split by channel/class groups). Measures WHERE the renderer fails spectrally — the dw1 distill window's diagnostic gap (is the distill residual an HF deficit like DiT's, or class-structured?). | dw1 distill line · fl1 (optional per-class flicker-by-frequency decomposition) | Spectra of renders vs solve frames indistinguishable within noise → renderer failure is NOT spectral → the whole SPAE cure class is N-A for the distill line (itself a decisive routing fact). | $0 scorer-free (cached frames), one script |
| 3 | **ALREADY-EMBODIED, ours SHARPER** | "HF components are hard to model / suppressing them helps" — our #520/#839 version is a THEOREM about the fixed operator, not an empirical observation: HF dying in the composite-R downsample (RESIZE_KERNEL_NULLITY_DOF 80.67% + CERTIFIED_ZERO_WEIGHT_BLIND_MASK 22.70%) is PROVABLY rate waste. SPAE adds one empirical warning we lacked: HF is CHANNEL-DIFFUSE (their Fig 2c) — so any per-channel truncation scheme that assumes HF concentrates in identifiable channels is unsupported UNLESS the ordering is trained in (which is exactly rank-1's lever). | #520/#839 doctrine · rank-1's design rationale | A measured channel-group spectrum on OUR token field showing natural HF concentration (no training needed) → the trained-ordering lever is unnecessary; truncate natively. | folded into rank-2's instrument |
| 4 | **EXTERNAL CONFIRMATION (no new action)** | Compact-semantic-bottleneck + decoder-regenerated detail = the od8/od9 LARGE_DELTA_ENTROPY fork's live branch. Their dim sweep (gen quality monotonically worse 64→128 despite better reconstruction) is the same shape as our rate-dead ship-the-solve verdict: carrying MORE of the solve is worse once the receiver can regenerate. | od8 delta-entropy fork routing · pk1 composed packet | n/a (confirmation row, no claim of our own rides on it) | $0 |
| 5 | **ADOPT-AS-DISCIPLINE** | The #949 rate_crush family inherits SPAE's sweep protocol: any terminal-compression candidate must trace the (bottleneck-dim / drop-level) × (reconstruction, TASK metric) surface — their Table 4a is the canonical shape showing reconstruction and task metrics ANTI-correlate past the knee. For us the task metric is composed S through the real receiver, never PSNR. | #949 fire-order text (updated below) | n/a (protocol row) | $0 |
| 6 | **N-A** | Three-stage training schedule, alignment loss (they finetune a LEARNED encoder and must anchor it; our scorer is frozen — the entire drift problem doesn't exist), DiT/flow-matching specifics, GAN/LPIPS losses. #497 basis-cure: their result is channel-structure not spatial-basis; no consumer. | none | — | — |

## Key measured facts retained (for future recall, exact values)

- Architecture: frozen E_rep → 1-block compressor E_c → **Z_b ∈ ℝ^(H×W×64)** → 6-block
  lifting E_u → decoder D. Only E_c/E_u/D train (stages I/II/III = 16/2/10 epochs).
- Masking: m_i^(k) = 1{i ≤ 64−k}, k sampled per step from {0,12,16,24,32} w.p.
  {0.60,0.15,0.10,0.10,0.05}; applied Z̃_b = Z_b ⊙ m^(k) in stage III.
- Table 4b (the lever's price): Unmask PSNR 30.96/gFID 8.30/IS 97.5 → Channel-mask
  30.28/5.00/151.2. Token-mask (spatial) is the LOSING control: 30.33/7.54/102.9 —
  channel ordering, not spatial dropout, is what pays.
- Table 4a (dim sweep): 32→27.31 PSNR/4.82 gFID · 64→30.28/5.00 · 96→30.97/5.93 ·
  128→31.35/7.18.
- Linear probe: semantics survive the bottleneck (DINOv2 80.33→80.13 IN-1K).
- Fig 2b: RAE (high-dim) latents show generated-vs-encoder spectral gap from
  MID-frequency onward; VAE latents only minor HF deviation → dimensionality amplifies
  the modeling gap.
- Stated limitations: fixed training resolution (text-rich regions), fixed masking
  schedule (adaptive schedules named as open), no compute-cost comparison.

## RECALL EVIDENCE

- Charter: `.omx/tmp/codex_runs/ar11_prompt.md` (arm died rc=1 at spawn — codex usage
  limit until 2026-08-10; MAIN read owed per the operator complete-read directive).
- Consumers recalled from hot-state + ledger before ranking: od8/od9 LARGE_DELTA_ENTROPY
  fork + ordering-law receipts (#946) · dw1 distill window (#790) · #520/#839 HF/nullity
  canonical names (vs1) · TR1 head + QA83 factorized head (b2b) · #869 token waterfill ·
  gr1 granularity race (#778) · #949 rate_crush fire-order · #497 basis program ·
  fl1 flicker floors (#813) · lg1 seg-hold guard (#808) · AR8 transfer-validity clause
  (`.omx/research/ddm_ar8_20260805/AR8_RECEIPT.md`) reused as the pullback-precondition
  template. #918 (coder/basis closed) checked: rank-1 is NOT a coder/basis race — it is
  a training-time ordering lever on the token field, outside #918's closed scope.
- Scoped negative: no prior SPAE/2608.01306/nested-dropout receipt found under
  `.omx/research` or memory (queries: `SPAE`, `2608.01306`, `nested dropout`,
  `suffix mask`, `channel mask`, `spectral mismatch`).

## Boundaries

Full text read via 3 targeted passes over the arXiv /html surface; no code imported or
run; no scorer forward, no n600 job, no archive bytes, no launch, no lane claim, no
canonical equation registered (the lever registers at build time, not at read time).

## NEXT_IF_RESUMED

1. **AR11-P1 ($0, first):** the rank-2 spectral instrument on cached renders vs C1 solve
   frames — its outcome ALSO adjudicates rank-3's falsifier (natural HF concentration?).
2. **AR11-P2 (lever, gated):** DSL lever `tr1_token_suffix_mask` default-off, schedule
   {0,…} probabilities DERIVED not copied (their {0.60,…} is their-vehicle-tuned —
   constants-are-poison), + lg1-composed class protection; fires as a burn arm-matrix
   candidate at the next window boundary with the matched A/B falsifier pre-registered.
3. Fold rank-5's protocol sentence into #949's fire-order text.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
