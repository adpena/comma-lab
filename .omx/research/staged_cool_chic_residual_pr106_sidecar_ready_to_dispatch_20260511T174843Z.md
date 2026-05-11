# Staged Cool-Chic residual byte-closed PR106 sidecar — ready to dispatch (operator-approval-required) — 2026-05-11

**Lane:** `lane_cool_chic_residual_pr106_sidecar_dispatch_ready` (L1, 1/7 gates: impl_complete)

**Status:** SCAFFOLD-COMPLETE, NOT-YET-DISPATCHED. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + "Cross-agent dispatch coordination" + operator directive 2026-05-11 ("ready to dispatch in parallel as soon as we secure funding"): operator approval is REQUIRED before any exact T4 dispatch.

## Family + canonical reference

- **Family:** Cool-Chic (hierarchical pyramid of latent grids decoded by a tiny coordinate-MLP)
- **Reference:** Ladune, T., Philippe, P., Hamidouche, W., Henry, F., & Deforges, O. (2023). "Cool-Chic: Coordinate-based Low Complexity Hierarchical Image Codec." ICCV 2023.
- **Residual encoding:** upsample-cascade pyramid; each level L stores INT8 coefficients at H/2^L × W/2^L; bilinear-upsample + sum produces the residual added to PR106 decoded RGB.

## Predicted score delta

**`[predicted]` -0.0005 to -0.0025 over PR106 r2 (0.20664588545741508 [contest-CUDA T4]).**

Rationale (Bayesian): Cool-Chic's coordinate-MLP architecture is known to produce
sub-0.20 bpp parity on Kodak per the original paper; the hierarchical pyramid
captures multi-resolution scene structure orthogonal to PR106's HNeRV latent
encoding. At 3 levels with 1/4 / 1/8 / 1/16 resolution the per-level entropy
budget is ~2 KB total after INT8 + brotli. At PR106's operating point a 2 KB
residual that moves seg by 5e-6 + pose by 1e-6 buys ≈ -0.0008 score.

`[predicted]` tag — not empirical. No anchor until paired CPU+CUDA dispatch.

## Cost estimate per dispatch

- **Modal T4** ($0.59/hr): ~$0.30 paired CPU+CUDA
- **Vast.ai 4090** ($0.25/hr): ~$0.20 paired

Cumulative across all 5 families: $1.00–$2.50.

## Byte budget for the residual + total archive size

| Mode | residual_bytes | Archive total |
|---|---|---|
| `--residual-mode empty` (default) | 0 B | 178270 B (scaffold-readiness) |
| `--n-levels 3 --n-frames 1200 --residual-mode zero` | ~3.5 MB | ~3.7 MB (identity bolt-on) |
| score-aware residual (entropy-coded; L2 step) | est. 1.5-3 KB | 180-181 KB (operator-approved dispatch target) |

## Per-byte EV at PR106 r2's `2.40e-9 pose / 6.66e-9 seg` thresholds

A 2 KB residual that buys -0.0008 score corresponds to net EV = +0.0008 / 2048
= 3.9e-7 per byte (> break-even threshold of 6.66e-7 if seg-axis; > 2.40e-9
break-even if pose-axis — pose-axis is dominant).

## 8 archive-grammar fields cleared

| Field | Value |
|---|---|
| `archive_grammar` | pr106_plus_residual_sidecar_monolithic_v1 (single-file 0.bin) |
| `parser_section_manifest` | magic(1B)=0xFD + format_id(1B)=0x11 + pr106_len(4B LE) + pr106_bytes + residual_len(4B LE) + residual_bytes |
| `inflate_runtime_loc_budget` | 145 LOC / 200 budget |
| `runtime_dep_closure` | numpy + torch + PR106 codec.py + PR106 model.py |
| `export_format` | pr106_plus_residual_per_family_v1 |
| `score_aware_loss` | research_only_scaffold (L2: gradient-trained pyramid coefs) |
| `bolt_on_loc_budget` | 350 LOC / 350 budget |
| `no_op_detector_planned` | YES — byte-mutation smoke + e2e test |

## Exact dispatch command (operator-approval-required)

```bash
.venv/bin/python tools/materialize_cool_chic_residual_pr106_sidecar.py \
    --pr106-archive submissions/pr106_latent_sidecar_r2/archive.zip \
    --output-dir experiments/results/lane_cool_chic_residual_pr106_sidecar_$(date -u +%Y%m%dT%H%M%SZ)

# Claim, dispatch, harvest — same template as wavelet
```

## Blockers remaining

1. **score_aware_loss** — gradient-trained pyramid coefs (L2 step).
2. **Operator GPU spend approval**.
3. **Council-grade design review** of the per-level quantisation strategy.

## 6-hook wire-in declaration

1. **Sensitivity-map**: per-level pyramid sparsity stats from
   `compute_cool_chic_residual_stats()` are sensitivity priors. WIRED.
2. **Pareto constraint**: per-level byte cost vs. (d_seg, d_pose). WIRED via manifest.
3. **Bit-allocator hook**: per-level float32 scale. WIRED.
4. **Cathedral autopilot dispatch hook**: WIRED via lane registry.
5. **Continual-learning posterior update**: PENDING dispatch.
6. **Probe-disambiguator**: N/A at scaffold level.

## Cross-references

- Scaffold landing: `feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md`
- Materializer: `tools/materialize_cool_chic_residual_pr106_sidecar.py`
- Inflate runtime: `submissions/pr106_cool_chic_residual_sidecar/`
