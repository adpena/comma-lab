# Staged wavelet residual byte-closed PR106 sidecar — ready to dispatch (operator-approval-required) — 2026-05-11

**Lane:** `lane_wavelet_residual_pr106_sidecar_dispatch_ready` (L1, 1/7 gates: impl_complete)

**Status:** SCAFFOLD-COMPLETE, NOT-YET-DISPATCHED. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + "Cross-agent dispatch coordination" + operator directive 2026-05-11 ("ready to dispatch in parallel as soon as we secure funding"): operator approval is REQUIRED before any exact T4 dispatch.

## Family + canonical reference

- **Family:** wavelet (single-level 2D Haar over PR106 r2 decoded RGB residuals)
- **Reference:** Mallat, S. (1989). "A theory for multiresolution signal decomposition: the wavelet representation." IEEE PAMI 11(7): 674-693
- **Inflate runtime numpy port:** `tac.residual_basis.numpy_inverse_dwt` (clears the L1 PyWavelets-not-runtime-dep blocker per `feedback_numpy_inverse_dwt_landed_20260511.md`)

## Predicted score delta

**`[predicted]` -0.0005 to -0.003 over PR106 r2 (0.20664588545741508 [contest-CUDA T4]).**

Rationale (Bayesian; no exact eval): the wavelet residual basis captures
**sparse multi-resolution detail** that PR106's HNeRV decoder cannot represent at
its 384×512 bilinear-upsampled native resolution. The Mallat scattering-transform
invariance argument predicts residual energy concentrates in the HL/LH/HH bands
at the coarsest level (LL3 + 3 finer detail tuples in the multi-level variant);
INT8 quantisation + per-band scale prefix typically holds entropy within
4-12KB. At PR106's operating point (`dS/dB = 6.66e-7`) a 4KB residual that
moves seg by 1e-5 + pose by 1e-6 buys ≈ -0.0010 score.

**This is a prediction tag, NOT an empirical claim.** No [contest-CUDA] or
[contest-CPU] anchor exists for this archive until operator approves the
dispatch and a paired CPU+CUDA result lands.

## Cost estimate per dispatch

- **Vast.ai RTX 4090** ($0.25/hr): ~10 min training-free materialization + ~15 min
  inflate.sh + upstream/evaluate.py for the public 600-sample test set → ~$0.10
  CUDA + ~$0.06 CPU (Modal Linux x86_64). Total `≤$0.20` per family per axis pair.
- **Modal T4** ($0.59/hr): ~$0.30 if T4 is preferred for contest-CUDA parity.

The materializer itself runs in <30s on CPU and emits a deterministic archive
zip + manifest with `score_claim=False` / `ready_for_exact_eval_dispatch=False`
(strict-per-Catalog-#100 `check_gate2_no_naked_bytes`).

## Byte budget for the residual + total archive size

| Mode | residual_bytes | Archive total | When to use |
|---|---|---|---|
| `--residual-mode empty` (default) | 0 B | 178270 B = PR106 r2 (178256 B) + 14 B wrapper | scaffold-readiness archive (no score movement; wire-format closure only) |
| `--residual-mode zero --n-frames 1200` | 3,656,000 B | 3.83 MiB | per-frame zero band-coefs (identity bolt-on; no score movement; quantisation cliff test) |
| `--residual-mode probe --n-frames 1200` | 3,656,000 B | 3.83 MiB | inflate-runtime byte-mutation no-op smoke (researchers only) |
| `--residual-mode (custom)` — score-aware | est. 4-12 KB after entropy coding | 182-186 KB | operator-approved dispatch target |

Score-aware residual encoder (PR101 sidecar grammar via
`tac.packet_compiler.encode_centered_delta_uint8` + `split_brotli_self_delimiting`)
is the next L2 promotion step; this scaffold accepts the residual bytes via
the materializer's `--residual-mode` flag and a future encoder pass.

## Per-byte EV at PR106 r2's `2.40e-9 pose / 6.66e-9 seg` thresholds

A 4KB residual that moves pose by 9.6e-6 (=4096 × 2.40e-9 marginal at break-
even) or seg by 2.7e-5 is "break-even"; the wavelet sparsity prior predicts
detail-band energy contributes to pose-axis improvement (motion / boundary
edges live in HL/LH bands). Predicted EV is positive for a -0.0010 net
score delta @ 4KB cost.

## 8 archive-grammar fields cleared

Per HNeRV parity discipline lesson + STRICT preflight Catalog #124
`check_representation_lane_has_archive_grammar_at_design_time`:

| Field | Value |
|---|---|
| `archive_grammar` | pr106_plus_residual_sidecar_monolithic_v1 (single-file 0.bin) |
| `parser_section_manifest` | magic(1B)=0xFD + format_id(1B)=0x10 + pr106_len(4B LE) + pr106_bytes + residual_len(4B LE) + residual_bytes |
| `inflate_runtime_loc_budget` | 187 LOC actual / 200 budget (waiver: residual-decode helper) |
| `runtime_dep_closure` | numpy + torch + PR106 codec.py + PR106 model.py (no PyWavelets — numpy_inverse_dwt clears the blocker) |
| `export_format` | pr106_plus_residual_per_family_v1 (sister of PR100 hnerv_lc_v2 sidecar) |
| `score_aware_loss` | research_only_scaffold (operator-approved bolt-on for score-aware residual generation is the L2 step) |
| `bolt_on_loc_budget` | 350 LOC budget; current materializer ≈ 160 LOC |
| `no_op_detector_planned` | YES — `tac.residual_basis.pr106_materializer_helpers.run_no_op_detector_byte_mutation` + e2e byte-mutation test in test_materialize_residual_pr106_sidecars.py |

## Exact dispatch command (operator-approval-required)

```bash
# Step 1: materialize (CPU; ~30s)
.venv/bin/python tools/materialize_wavelet_residual_pr106_sidecar.py \
    --pr106-archive submissions/pr106_latent_sidecar_r2/archive.zip \
    --output-dir experiments/results/lane_wavelet_residual_pr106_sidecar_$(date -u +%Y%m%dT%H%M%SZ)

# Step 2: claim lane via canonical helper (REQUIRED per CLAUDE.md
# "Cross-agent dispatch coordination")
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id lane_wavelet_residual_pr106_sidecar_dispatch_ready \
    --provider modal --instance-or-job pending \
    --notes "operator-approved wavelet residual pre-stage dispatch"

# Step 3: dispatch (Vast.ai 4090 OR Modal T4 OR Lightning)
# (template; operator picks the provider)
.venv/bin/python scripts/launch_lane_on_vastai.py \
    --lane-id lane_wavelet_residual_pr106_sidecar_dispatch_ready \
    --archive experiments/results/lane_wavelet_residual_pr106_sidecar_<UTC>/wavelet_pr106_residual_sidecar_archive.zip \
    --runtime-tree submissions/pr106_wavelet_residual_sidecar \
    --device cuda --hard-cap 1.50

# Step 4: harvest result, update lane gates
.venv/bin/python tools/lane_maturity.py mark lane_wavelet_residual_pr106_sidecar_dispatch_ready \
    --gate contest_cuda --evidence "<score> [contest-CUDA] <reports/raw/...>"
```

## Blockers remaining

1. **score_aware_loss** — the current materializer accepts pre-computed residual
   bytes but does not yet contain an encoder that generates them from SegNet/
   PoseNet gradients on PR106 decoded outputs. The L2 promotion step is to
   wire the residual encoder to `tac.differentiable_eval_roundtrip` +
   `tac.score_gradient_param_saliency` on CUDA (operator-gated GPU dispatch).
2. **Operator GPU spend approval** ($0.20–$0.50 per dispatch; cumulative for
   all 5 families ~$1.00–$2.50).
3. **Council-grade design review** of the per-family residual encoder before
   GPU dispatch (per CLAUDE.md "Design decisions — non-negotiable").

## 6-hook wire-in declaration

1. **Sensitivity-map**: per-band sparsity statistics from
   `compute_wavelet_residual_stats()` are sensitivity priors for the bit
   allocator. WIRED via the residual scaffold result tuple.
2. **Pareto constraint**: each materialized variant is a new candidate
   `(byte_cost, predicted_d_seg, predicted_d_pose)` tuple. WIRED via
   `materialization_manifest.json`.
3. **Bit-allocator hook**: per-band quantisation scale prefix is the allocator's
   input. WIRED via the per-band 4×4B scales section.
4. **Cathedral autopilot dispatch hook**: lane is registered at L1; autopilot
   can route a dispatch when operator approves. WIRED.
5. **Continual-learning posterior update**: post-dispatch [contest-CUDA] +
   [contest-CPU] anchor → `tac.continual_learning.posterior_update_locked`.
   PENDING dispatch.
6. **Probe-disambiguator**: N/A — single canonical Haar single-level encoding;
   no design tension at this scaffold level.

## Cross-references

- Memo: `feedback_wavelet_residual_basis_pr106_scaffold_landed_20260511.md`
- numpy_inverse_dwt landing: `feedback_numpy_inverse_dwt_landed_20260511.md`
- Handoff: `~/Downloads/pact_score_lowering_handoff_2026-05-11.md` Bottom-line tranche item #6
- Operator-decision-required pin: `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md` (cumulative GPU spend pending)
- Materializer: `tools/materialize_wavelet_residual_pr106_sidecar.py`
- Inflate runtime: `submissions/pr106_wavelet_residual_sidecar/{inflate.py, inflate.sh, src/}`
- Wire format shared module: `tac.residual_basis.pr106_sidecar_packing`
- Materializer helper shared module: `tac.residual_basis.pr106_materializer_helpers`
