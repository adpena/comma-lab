# Staged SIREN residual byte-closed PR106 sidecar — ready to dispatch (operator-approval-required) — 2026-05-11

**Lane:** `lane_siren_residual_pr106_sidecar_dispatch_ready` (L1, 1/7 gates: impl_complete)

**Status:** SCAFFOLD-COMPLETE, NOT-YET-DISPATCHED.

## Family + canonical reference

- **Family:** SIREN (sinusoidal-activation coordinate-MLP)
- **Reference:** Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020). "Implicit Neural Representations with Periodic Activation Functions." NeurIPS 2020.
- **Residual encoding:** SPARSE 2D-FFT COEFFICIENTS. Each coef is a typed 9B tuple `(frame_idx u16, k_row i16, k_col i16, channel u8, real i8, imag i8)` + global float32 scale prefix. The inflate runtime places each coef into the 2D-FFT spectrum then inverse-FFTs per frame per channel.

This is the SMALLEST-BYTE SIREN-compatible residual: an MLP runtime would need
≥50KB for the weights; sparse FFT coefs at 64-256 coefs = 0.6–2.4 KB total.

## Predicted score delta

**`[predicted]` -0.0005 to -0.0020 over PR106 r2 (0.20664588545741508 [contest-CUDA T4]).**

Rationale (Bayesian): SIREN's frequency-domain prior captures smooth signal
components most efficiently. The sparse-FFT encoding bypasses the MLP weight
budget entirely while preserving the same frequency-domain signature. PR106
decoded RGB has known low-frequency residual energy from the 384→874
bicubic-upsample step — a 128-coef sparse FFT (1.16 KB residual) targeted at
low-frequency bands could recover ~10% of that energy.

`[predicted]` tag.

## Cost estimate per dispatch

- ~$0.20–$0.30 paired CPU+CUDA per family.

## Byte budget for the residual + total archive size

| Mode | residual_bytes | Archive total |
|---|---|---|
| `--residual-mode empty` (default) | 0 B | 178270 B |
| `--n-coefs 64 --residual-mode probe` | 6+64×9 = 582 B | 178852 B |
| `--n-coefs 256 --residual-mode probe` | 6+256×9 = 2310 B | 180580 B |
| score-aware sparse-FFT coefs | est. 1–4 KB | 179-183 KB (dispatch target) |

## Per-byte EV at PR106 r2's `2.40e-9 pose / 6.66e-9 seg` thresholds

A 1 KB sparse-FFT residual that moves pose by 2.4e-6 buys ≈ -0.00076.
The frequency-domain encoding is sister to the wavelet family (different basis,
same multi-resolution principle); EV is competitive with wavelet at much
smaller byte footprint.

## 8 archive-grammar fields cleared

| Field | Value |
|---|---|
| `archive_grammar` | pr106_plus_residual_sidecar_monolithic_v1 |
| `parser_section_manifest` | magic 0xFD + format_id 0x13 + pr106_len + pr106_bytes + residual_len + (4B scale + 2B n_coefs + n×9B coefs) |
| `inflate_runtime_loc_budget` | 171 LOC / 200 budget |
| `runtime_dep_closure` | numpy + torch + PR106 codec.py + PR106 model.py (uses np.fft.ifft2) |
| `export_format` | pr106_plus_residual_per_family_v1 |
| `score_aware_loss` | research_only_scaffold (L2: gradient-trained sparse FFT coefs) |
| `bolt_on_loc_budget` | 350 LOC / 350 budget |
| `no_op_detector_planned` | YES |

## Exact dispatch command (operator-approval-required)

```bash
.venv/bin/python tools/materialize_siren_residual_pr106_sidecar.py \
    --pr106-archive submissions/pr106_latent_sidecar_r2/archive.zip \
    --output-dir experiments/results/lane_siren_residual_pr106_sidecar_$(date -u +%Y%m%dT%H%M%SZ) \
    --n-coefs 0  # default empty residual; operator overrides post-approval
```

## Blockers remaining

1. **score_aware_loss** — gradient-trained sparse-FFT coef selection.
2. **Operator GPU spend approval**.

## 6-hook wire-in declaration

1. **Sensitivity-map**: per-band 2D-FFT-magnitude stats from `compute_siren_residual_stats()`. WIRED.
2. **Pareto constraint**: WIRED via manifest.
3. **Bit-allocator hook**: per-coef INT8 + global scale. WIRED.
4. **Cathedral autopilot dispatch hook**: WIRED.
5. **Continual-learning posterior update**: PENDING dispatch.
6. **Probe-disambiguator**: low/mid/high frequency band cutoffs (defaults in scaffold; selectable per-dispatch).

## Cross-references

- Scaffold landing: `feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md`
- Materializer: `tools/materialize_siren_residual_pr106_sidecar.py`
- Inflate runtime: `submissions/pr106_siren_residual_sidecar/`
