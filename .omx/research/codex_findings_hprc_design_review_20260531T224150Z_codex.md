# Codex Findings: HPRC Design Review

UTC: 2026-05-31T22:41:50Z

Reviewed: `.omx/research/hprc_hierarchical_predictive_receiver_codec_design_20260531T223400Z_codex.md`

Authority: design review only. Score claim false. Promotion eligible false.

## Verdict

The HPRC design is pointed at the right object: stop transmitting an explicit
wavelet video and instead transmit a tiny deterministic receiver plus latent,
selector, and scorer-critical residual streams. The PR95 contrast makes this
non-negotiable: PR95 pays roughly 162 KB for decoder and 16 KB for latents,
while current Z8 pays megabytes for top-LL and detail wavelet payloads.

The repaired Z8 quantized-detail packet is now custody-valid, but it is still
not competitive: `archive.zip=10,289,674` bytes, rate term `6.8515`, with
`wavelet_blob=10,165,099` bytes (`99.71%` of inner packet). This is durable
negative evidence for explicit-Z8-as-payload and positive evidence for HPRC as
the next first-class lane.

## Bugs / Hardening Required

1. HPRC byte ceilings must include every charged byte: decoder weights,
   latents, residual streams, codebooks, entropy tables, config/header, runtime
   files, and ZIP overhead. Payload-only byte estimates are not authority.

2. Section integrity should store full SHA-256 for authority. `sha256_prefix`
   can be a display/index field, but exact custody and deterministic replay
   need full hashes.

3. Receiver-consumption proof cannot rely only on raw byte flips. Entropy-coded
   sections may parse-refuse random flips; use valid semantic mutations for
   decoder weights, latents, selectors, codebooks, and residual tokens, plus
   raw flip proof where appropriate.

4. The `q=0.25`/`28.7 KB` residual-sidecar number is advisory until it is inside
   an archive-bound HPRC packet, decoded by `inflate.sh`, and measured against
   full-video scorer-preserving replay.

5. Mamba/Dreamer/RAFT/C3/Cool-Chic/SIREN/CLADE/SPADE are admissible only when
   distilled into counted, deterministic bytes. Hidden runtime dependencies or
   per-pair hidden state recreate the explicit-video rate bug.

6. The first HPRC adapter should target the shared archive-bound candidate
   contract from day one. A candidate that cannot emit archive ZIP, runtime
   tree hash, receiver proof, byte profile, false-authority flags, and exact
   blocker should be migration work, not a score lane.

## Design Additions

- Add a byte-ledger gate before training: reject any HPRC candidate whose
  complete archive projection cannot plausibly fit below the sub-0.19 ceiling
  after counting runtime and tables.
- Treat Z8 wavelets as teacher/action surfaces: top-LL/detail residuals,
  P18/P19 gradients, pose-null masks, and class-boundary regions feed residual
  token selection, but do not become dense payload.
- Use float-compression codecs only as baselines/oracles. ZFP/SZ/MGARD/fpzip
  and exponent/mantissa separation answer "how much redundancy remains in
  floats"; the final contest path should be quantized integer symbols with
  EBCOT-like bitplane truncation, significance structure, and range/ANS/RLE
  coding selected by scorer-conditioned RD.
- Make the first runnable HPRC V0 small and falsifiable: PR95/PACT-NeRV-like
  base decoder plus a sparse Z8 residual token sidecar. If the base+sidecar
  cannot beat the repaired Z8 packet by orders of magnitude on archive bytes,
  demote quickly.

## Immediate Next Implementation

1. Package the HPRC grammar as a real `0.bin`/`hprc.bin` parser-packer with
   full section hashes and deterministic decode.
2. Add a minimal HPRC archive adapter that emits the shared archive-bound
   candidate package and receiver proof.
3. Train/export a compact MLX PR95/PACT-NeRV base, then compute Z8 residual
   tokens against that base rather than storing Z8 top-LL/detail fields.
4. Run full-video P18/P19 allocation over the residual token budget and profile
   complete `archive.zip` bytes after every candidate.

No exact score authority is implied until CPU/CUDA auth eval signs a
byte-closed archive/runtime pair.
