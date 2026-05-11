# Staged coordinate-MLP residual byte-closed PR106 sidecar — ready to dispatch (operator-approval-required) — 2026-05-11

**Lane:** `lane_coord_mlp_residual_pr106_sidecar_dispatch_ready` (L1, 1/7 gates: impl_complete)

**Status:** SCAFFOLD-COMPLETE, NOT-YET-DISPATCHED.

## Family + canonical reference

- **Family:** Coordinate-MLP (generic Fourier-features family — SIREN/NeRV/HNeRV/Cool-Chic/C3 all share the Laplacian-smoothness prior)
- **Reference:** Tancik, M., Srinivasan, P. P., Mildenhall, B., et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." NeurIPS 2020.
- **Residual encoding:** low-resolution residual at 1/8 scale (CAMERA_H/8 × CAMERA_W/8 = 109 × 145); INT8 + per-frame float32 scale; bicubic-upsample to camera resolution in the inflate runtime.

This is the FAMILY-AGNOSTIC SHARED PRIOR — any specific coordinate-MLP family
(SIREN/NeRV/etc.) competes against this baseline at L2.

## Predicted score delta

**`[predicted]` -0.0003 to -0.0015 over PR106 r2 (0.20664588545741508 [contest-CUDA T4]).**

Rationale (Bayesian): the low-resolution Laplacian-smoothness prior is the
SIMPLEST possible coordinate-MLP-compatible residual. It dominates SIREN's
sparse-FFT and Cool-Chic's pyramid at the lowest byte budgets but is
dominated by them at higher budgets. The expected EV at 1-2 KB is modest
but reliably positive when the dispatch reveals genuine PR106 residual
energy at low frequencies.

`[predicted]` tag — not empirical.

## Cost estimate per dispatch

- ~$0.20–$0.30 paired CPU+CUDA per family.

## Byte budget for the residual + total archive size

| Mode | residual_bytes | Archive total |
|---|---|---|
| `--residual-mode empty` (default) | 0 B | 178270 B |
| `--n-frames 1200 --residual-mode zero` | ~57 MB raw | ~57 MB (identity-bolt-on) |
| score-aware low-res residual + brotli | est. 0.5-2 KB | 179-181 KB (dispatch target) |

## Per-byte EV at PR106 r2's `2.40e-9 pose / 6.66e-9 seg` thresholds

A 1 KB low-res Laplacian residual that moves seg by 1.5e-6 + pose by 5e-7
buys ≈ -0.00045. This is the LOWEST predicted EV of the 5 families, which is
expected for the simplest representation; the value is as a BASELINE / DEFAULT
that more specific families must beat.

## 8 archive-grammar fields cleared

| Field | Value |
|---|---|
| `archive_grammar` | pr106_plus_residual_sidecar_monolithic_v1 |
| `parser_section_manifest` | magic 0xFD + format_id 0x14 + pr106_len + pr106_bytes + residual_len + (per-frame: 4B scale + LOW_H×LOW_W×3 int8) |
| `inflate_runtime_loc_budget` | 151 LOC / 200 budget |
| `runtime_dep_closure` | numpy + torch + PR106 codec.py + PR106 model.py |
| `export_format` | pr106_plus_residual_per_family_v1 |
| `score_aware_loss` | research_only_scaffold |
| `bolt_on_loc_budget` | 350 LOC / 350 budget |
| `no_op_detector_planned` | YES |

## Exact dispatch command (operator-approval-required)

```bash
.venv/bin/python tools/materialize_coord_mlp_residual_pr106_sidecar.py \
    --pr106-archive submissions/pr106_latent_sidecar_r2/archive.zip \
    --output-dir experiments/results/lane_coord_mlp_residual_pr106_sidecar_$(date -u +%Y%m%dT%H%M%SZ)
```

## Blockers remaining

1. **score_aware_loss** — gradient-trained low-res Laplacian residual.
2. **Operator GPU spend approval**.

## 6-hook wire-in declaration

1. **Sensitivity-map**: Laplacian-magnitude smoothness stats from `compute_coordinate_mlp_residual_stats()`. WIRED.
2. **Pareto constraint**: WIRED via manifest.
3. **Bit-allocator hook**: per-frame float32 scale. WIRED.
4. **Cathedral autopilot dispatch hook**: WIRED.
5. **Continual-learning posterior update**: PENDING dispatch.
6. **Probe-disambiguator**: family-agnostic vs family-specific (SIREN/NeRV/Cool-Chic/C3); this scaffold is the agnostic baseline.

## Cross-references

- Scaffold landing: `feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md`
- Materializer: `tools/materialize_coord_mlp_residual_pr106_sidecar.py`
- Inflate runtime: `submissions/pr106_coord_mlp_residual_sidecar/`
