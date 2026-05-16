# Z3-G1 entropy-coded v2 full export research packet - 2026-05-16

## Scope

Owned landing: convert `experiments/train_substrate_z3_g1_entropy_coded_v2.py`
from parser-only smoke scaffold into a byte-closed, research-only export path.

This is not a score claim and not a dispatch claim. The packet remains blocked
from promotion until a learned compress-side class-index exporter, a verified
remote driver, and paired contest CPU/CUDA exact eval exist.

## Artifact

- Export command:

```bash
.venv/bin/python experiments/train_substrate_z3_g1_entropy_coded_v2.py \
  --a1-archive-path submissions/a1/archive.zip \
  --video-path upstream/videos/0.mkv \
  --upstream-dir upstream \
  --output-dir experiments/results/z3g1_entropy_coded_v2_full_export_codex_20260516T021938Z \
  --epochs 1 \
  --device cpu \
  --frame-proof-pairs 16
```

- Stats:
  `experiments/results/z3g1_entropy_coded_v2_full_export_codex_20260516T021938Z/stats.json`
- Frame proof:
  `experiments/results/z3g1_entropy_coded_v2_full_export_codex_20260516T021938Z/frame_mutation_proof.json`
- Runtime manifest:
  `experiments/results/z3g1_entropy_coded_v2_full_export_codex_20260516T021938Z/submission_dir/submission_runtime_manifest.json`

## Empirical Receipt

- `archive_zip_bytes`: 177816
- `z3g2_payload_bytes`: 177708
- `z3g2_payload_sha256`:
  `3f2c62fbae240d7a7cbb55535a9810c5085ac96bc74d5d7923f913cc2d2306d3`
- `archive_zip_sha256`:
  `b4153a23e7b605392a96fd4ff92f36568ae6b0b5c829bfefb17a780a6fff6c4a`
- `runtime_tree_sha256`:
  `af17e91ddb41117bab4f564547e6316588b4c52625040e82fa488d6c37fe67d6`
- `z3g2_section_bytes`: 14933
- A1 latent slot replaced: 15387 bytes
- Inner-payload byte savings: 454 bytes
- Rate-axis only delta estimate:
  `25 * 454 / 37545489 = 0.00030230610280124304`
- Distinguishing Z3G2 feature bytes shipped: 213
- Selected residual peak target: 80
- Selection rule: lowest mean absolute latent reconstruction error among
  byte-saving candidates.
- Latent reconstruction mean abs err: 0.008393692784011364
- Latent reconstruction max abs err: 0.030521124601364136

Bounded `inflate.sh` frame-output mutation proof:

- Scope: first 16 frame pairs only; this is not full exact eval.
- Verdict: pass
- Baseline raw SHA-256:
  `89ea9d3f4b3b383b9f96871ff324ddebfb058a22c18543850307bc8c1ed6d82e`
- Mutated raw SHA-256:
  `2278f614a3a67e455a18022861ef3602bc67702e7777aba2744d4bc73c278deb`
- Mutated blob: `sigma_table_blob`
- Payload byte offset: 162215

## Adversarial Notes

The first naive deterministic export was byte-negative:

- `z3g2_section_bytes`: 15903
- A1 latent slot: 15387
- Inner-payload byte regression: 516 bytes

That falsified the earlier synthetic-smoke extrapolation. The fix was not to
overclaim; the exporter now records a deterministic rate-distortion sweep over
residual peak targets and selects the best byte-saving candidate.

Paper/source-fidelity correction from the read-only citation pass: current
Z3G2 is not an arithmetic/range/ANS residual codec. It is a
constriction-Huffman class-index stream plus Brotli-compressed direct int8
residual bytes. Future range/ANS residual coding remains a valid follow-up, but
must land with measured histograms, CDF custody, roundtrip golden vectors, and
section-byte deltas before any source or score claim uses that language.

Relevant anchors to cite in follow-up ledgers:

- Ballé, J., Minnen, D., Singh, S., Hwang, S. J., & Johnston, N. (2018).
  *Variational Image Compression with a Scale Hyperprior*. ICLR.
  https://research.google/pubs/variational-image-compression-with-a-scale-hyperprior/
- Minnen, D., Ballé, J., & Toderici, G. D. (2018). *Joint Autoregressive and
  Hierarchical Priors for Learned Image Compression*. NeurIPS 31.
  https://papers.nips.cc/paper_files/paper/2018/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html
- Bamler, R. (2022). *Understanding Entropy Coding With Asymmetric Numeral
  Systems (ANS): a Statistician's Perspective*. arXiv:2201.01741.
  https://arxiv.org/abs/2201.01741
- constriction entropy-coding documentation:
  https://bamler-lab.github.io/constriction/

The bounded frame proof also exposed a hairline distinction: a one-pair proof
can fail even when parser/intermediate latent hashes change, because rounded
frame bytes may stay identical. The default proof window is therefore 16 pairs.

## Blockers

- Compress-side SegNet class export is not implemented.
- Deterministic modulo class indices are placeholders, not learned or
  score-conditioned.
- Bounded 16-pair frame mutation proof is not full 600-pair exact eval.
- No paired contest CPU/CUDA exact eval exists for this packet.
- No verified remote driver exists for this lane.
- Dispatch recipe remains `research_only=true` and `dispatch_enabled=false`.

## Next Work

1. Replace deterministic modulo class indices with the real compress-side
   SegNet class export path.
2. Run paired CPU/CUDA exact eval only after the remote driver and required
   input validation are strict.
3. Compare the learned packet against A1 and PR101/PR103 on the same evidence
   axis; do not infer CPU/CUDA transfer.
4. Keep the rate-distortion sweep in all future exports so synthetic-smoke byte
   savings cannot suppress real-packet evidence.
