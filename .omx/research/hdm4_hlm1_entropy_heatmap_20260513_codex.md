# HDM4/HLM1 Entropy Heatmap (2026-05-13)

Diagnostic command:

```bash
.venv/bin/python tools/xray_archive_section_entropy_heatmap.py \
  --archive experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/pr106_r2_lowlevel_hdm4_archive_candidate.zip \
  --label hdm4 \
  --archive experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip \
  --label hdm4_hlm1 \
  --output-dir experiments/results/xray_archive_section_entropy_heatmap_hdm4_hlm1_20260513_codex
```

Artifact:
`experiments/results/xray_archive_section_entropy_heatmap_hdm4_hlm1_20260513_codex/heatmap.json`

## Result

| Archive | bytes | sections | recoverable upper bound |
|---|---:|---:|---:|
| `hdm4` | `186492` | `1` | `32` bytes |
| `hdm4_hlm1` | `186423` | `1` | `32` bytes |

Both archives are single-member packets with payload entropy near `7.9986`
bits/byte and saturation ratio `0.9998`.

## Interpretation

This is a diagnostic-only byte-floor result: no score claim, no promotion
claim, and no lane retirement. It does sharpen the HDM5 byte-work target.

Generic recompression, wrapper changes, and entropy-code swaps are near
saturated at this transform. The remaining meaningful byte work must change the
payload grammar or representation:

- semantic recode of the latent/sidecar grammar;
- structured decoder-payload recode;
- PR93-style pose/latent delta-varint families where the data actually has
  low-order structure;
- transform-side changes before the near-uniform `0.bin` stream is emitted.

The HLM1 exact-CUDA gain was real but small because it changed the transform
payload by `69` archive bytes. The heatmap says the next score-lowering byte
gain cannot come from asking Brotli/zstd/range coding to squeeze the same
single near-uniform stream harder. HDM5 should target typed PacketIR sections
and structural priors, then rerun exact CUDA only for byte-closed packets.
