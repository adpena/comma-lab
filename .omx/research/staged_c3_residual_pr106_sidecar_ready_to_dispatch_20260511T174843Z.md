# Staged C3 residual byte-closed PR106 sidecar — ready to dispatch (operator-approval-required) — 2026-05-11

**Lane:** `lane_c3_residual_pr106_sidecar_dispatch_ready` (L1, 1/7 gates: impl_complete)

**Status:** SCAFFOLD-COMPLETE, NOT-YET-DISPATCHED.

## Family + canonical reference

- **Family:** C3 (Compressed Conditional Content — Cool-Chic + conditional hyperprior)
- **Reference:** Kim, H., Bauer, M., Theis, L., Schwarz, J. R., & Dupont, E. (2024). "C3: High-performance and low-complexity neural compression from a single image." CVPR 2024.
- **Residual encoding:** frame-delta conditional residual at quarter resolution; INT8 + per-frame scale. The inflate runtime cumulatively integrates the deltas across time and bilinear-upsamples to camera resolution.

## Predicted score delta

**`[predicted]` -0.0008 to -0.0030 over PR106 r2 (0.20664588545741508 [contest-CUDA T4]).**

Rationale (Bayesian): C3 is the strongest single-image neural codec in the
~10K-parameter regime. Its frame-delta conditional residual is the natural
fit for the temporal-prediction component of PR106's per-pair latent decoding.
The delta-coding scheme is sister to PR93's delta-varint pose codec already
identified as #1 EV/byte primitive in `feedback_public_pr_nonhnerv_mechanism_backlog_landed_20260511.md`.

`[predicted]` tag — not empirical.

## Cost estimate per dispatch

- ~$0.20–$0.30 paired CPU+CUDA per family.

## Byte budget for the residual + total archive size

| Mode | residual_bytes | Archive total |
|---|---|---|
| `--residual-mode empty` (default) | 0 B | 178270 B |
| `--n-frames 1200 --residual-mode zero` | ~76 MB raw | ~76 MB (identity-bolt-on) |
| score-aware delta-coded + brotli | est. 3-8 KB | 181-186 KB (dispatch target) |

## Per-byte EV at PR106 r2's `2.40e-9 pose / 6.66e-9 seg` thresholds

Frame-delta residuals favor the POSE axis (motion / camera-motion-induced
content), which is 2.79× more marginally sensitive at PR106 r2 operating
point. A 4 KB delta-coded residual that moves pose by 4e-6 buys ≈ -0.0010.

## 8 archive-grammar fields cleared

| Field | Value |
|---|---|
| `archive_grammar` | pr106_plus_residual_sidecar_monolithic_v1 |
| `parser_section_manifest` | magic 0xFD + format_id 0x12 + pr106_len + pr106_bytes + residual_len + residual_bytes |
| `inflate_runtime_loc_budget` | 146 LOC / 200 budget |
| `runtime_dep_closure` | numpy + torch + PR106 codec.py + PR106 model.py |
| `export_format` | pr106_plus_residual_per_family_v1 |
| `score_aware_loss` | research_only_scaffold (L2: gradient-trained delta coefs) |
| `bolt_on_loc_budget` | 350 LOC / 350 budget |
| `no_op_detector_planned` | YES |

## Exact dispatch command (operator-approval-required)

```bash
.venv/bin/python tools/materialize_c3_residual_pr106_sidecar.py \
    --pr106-archive submissions/pr106_latent_sidecar_r2/archive.zip \
    --output-dir experiments/results/lane_c3_residual_pr106_sidecar_$(date -u +%Y%m%dT%H%M%SZ)
```

## Blockers remaining

1. **score_aware_loss** — gradient-trained delta coefs.
2. **Operator GPU spend approval**.

## 6-hook wire-in declaration

1. **Sensitivity-map**: frame-delta sparsity stats from `compute_c3_residual_stats()`. WIRED.
2. **Pareto constraint**: WIRED via manifest.
3. **Bit-allocator hook**: per-frame float32 scale. WIRED.
4. **Cathedral autopilot dispatch hook**: WIRED.
5. **Continual-learning posterior update**: PENDING dispatch.
6. **Probe-disambiguator**: frame_delta vs mean_baseline conditioning modes (existing in scaffold; selected per-dispatch).

## Cross-references

- Scaffold landing: `feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md`
- Materializer: `tools/materialize_c3_residual_pr106_sidecar.py`
- Inflate runtime: `submissions/pr106_c3_residual_sidecar/`
- Sister pose-axis primitive: PR93 delta-varint pose codec (top-EV mechanism row)
