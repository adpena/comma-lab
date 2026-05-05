# PR106 Stacking — Decision Table (2026-05-04)

Consolidates all 8 codec-applicability investigations against the current public exact frontier (PR106 `belt_and_suspenders` by @valtterivalo at `0.20945673680571203` [contest-CUDA T4 A++], 186,239 bytes). One row per investigation: empirical preview, status, predicted score band, and one-liner dispatch.

## Summary

| Lane | Status | Predicted [contest-CUDA] | Stub-mode preview | Distortion risk | Dispatch cost |
|---|---|---|---|---|---|
| Lane Ω-W-V3 (water-fill v2) | ✅ Launch-ready | [0.194, 0.204] | −22,152 bytes / −11.9% archive | LOW (per-channel sensitivity controls) | ~$0.30 / 30min Vast.ai 4090 |
| Lane #04 int4 (uniform) | ✅ Launch-ready | [0.155, 0.180] | −76,258 bytes / −41.0% archive | HIGH (~7% rel err per weight) | ~$0.30 / 30min |
| Lane #04 int5 (sweet-spot) | ⏳ Sketch-only | [0.180, 0.196] | −34,720 bytes / −18.6% archive | MEDIUM (~3.3% rel err per weight) | needs adapter (1-tick build) |
| Lane #04 int6 (safe) | ⏳ Sketch-only | [0.190, 0.204] | −19,500 bytes / −10.5% archive | LOW (~1.6% rel err per weight) | needs adapter (1-tick build) |
| Lane SJ-KL C067 | ✅ Launch-ready (separate base) | (not yet predicted) | 942-byte sjkl.bin in CPU stub | LOW (additive residual) | ~$0.30 / 30min |
| Lane #02 arith_qint → PR106 latents | ❌ FALSIFIED | — | brotli already 24% below 0-th-order Shannon (3.76 vs 4.66 bits/byte) | — | — |
| Lane #03 QZS3 → PR106 decoder | ❌ FALSIFIED | — | hardcoded JointFrameGenerator schema; refuses HNeRV state dict | — | — |
| Lane #04 default block_fp (ternary) → PR106 | ❌ FALSIFIED | — | ternary {-1, 0, +1} only; destroys continuous distribution (max_err 0.5-1.0) | — | (revived as int4/5/6 above) |
| Lane #05 UNIWARD-delta → PR106 mask channel | ❌ FALSIFIED | — | PR106 has no separate mask channel (HNeRV implicit representation) | — | — |
| Lane #06 grayscale-LUT → PR106 mask channel | ❌ FALSIFIED | — | same root cause as #05 | — | — |

## Pareto frontier of int-N block-FP variants on PR106 Conv2d weights

Empirical from `experiments/block_fp_intN_codec_sketch.py` against PR106's 13 Conv2d tensors (block_size=128):

| bits | brotli bytes | rel_err vs PR106 max weight | rate Δ vs PR106 archive | Predicted score Δ |
|---:|---:|---:|---:|---:|
| 3 | 41,278 | 16.5% | −0.071 | (catastrophic distortion expected) |
| 4 | 58,721 | 7.1% | −0.051 | [0.155, 0.180] (HIGH risk) |
| **5** | **103,280** | **3.31%** | **−0.030** | **[0.180, 0.196] (MEDIUM risk — sweet spot)** |
| 6 | 119,175 | 1.55% | −0.020 | [0.190, 0.204] (LOW risk) |
| 7 | 153,883 | 0.79% | −0.011 | [0.198, 0.208] (LOW risk, modest savings) |
| 8 | 136,456 | 0.24% | −0.013 | [0.196, 0.207] (almost lossless, modest savings) |
| PR106 baseline (FP16+brotli) | 170,278 | 0.0% | 0 | 0.20946 |
| OWV2 stub (per-channel water-fill) | ~115,000 | sensitivity-weighted | −0.014 | [0.194, 0.204] |

**Key observation**: int5 is the unexplored sweet spot — it lands between Lane Ω-W-V3 (LOW-risk) and Lane #04 int4 (HIGH-risk) on both bytes and distortion. With `experiments/block_fp_intN_codec_sketch.py:encode_intN_block_fp(t, bits=5)` already implemented and tested, the only missing piece is a parameterized variant of `experiments/repack_pr106_with_int4_block_fp.py` that takes `--bits N` instead of hard-coded `bits=4`. The corresponding `submissions/apogee_intN/` adapter has the same shape as `submissions/apogee_int4/` but with a different magic byte per bit-width.

## Dispatch order recommendation

For minimum wall-clock to a contest-CUDA score that beats 0.20946:

1. **Lane Ω-W-V3 first** ($0.30, ~30min) — lowest-risk, council-approved 8/10. If it lands in [0.194, 0.204], we have a real beat.
2. **In parallel: Lane #04 int5 build + dispatch** (~$0.30 plus 1 tick adapter work) — if Ω-W-V3 doesn't land below 0.20946, int5's predicted band [0.180, 0.196] gives a deeper margin if distortion holds.
3. **Lane #04 int4 only if int5 distortion is acceptable** — int4 has 2x the distortion cost; gate on int5's empirical PoseNet+SegNet outcomes before committing $0.30 to int4.
4. **Lane SJ-KL C067 dispatchable independently** — different base archive (C067, not PR106); orthogonal axis to the int-N variants. Worth running in parallel with Lane Ω-W-V3 since it doesn't use the same renderer.

Total cost for the full dispatch matrix: ~$1.50 (5 jobs × $0.30 each).

## Falsification log (5 hidden-gem revival plans falsified at $0 cost)

The audit subagent that proposed PR106-stacking generated 8 candidates. 5 were falsified at zero GPU cost via codec-applicability stress tests (3 questions: codec accepts target's data shape? distribution matches? entropy ceiling below baseline?):

- **Lane #02 arith_qint → PR106 latents**: 0-th-order arithmetic coding cannot beat brotli that's already 24% below 0-th-order Shannon entropy. Memory: `feedback_arithmetic_qint_codec_pr106_latents_unviable_brotli_already_below_entropy_20260504.md`.
- **Lane #03 QZS3 → PR106 decoder**: Packer hardcoded for JointFrameGenerator schema; raises ValueError on HNeRV state dict. Memory: `feedback_qzs3_codec_pr106_unviable_jointframegenerator_locked_20260504.md`.
- **Lane #04 default block_fp → PR106 decoder**: Ternary-only ({-1, 0, +1}) destroys PR106's continuous weight distribution. Revived via int4 sketch (Lane #04 int4 above). Memory: `feedback_block_fp_codec_pr106_unviable_at_default_ternary_20260504.md`.
- **Lane #05 UNIWARD-delta → PR106 mask channel**: PR106 has no separate mask channel (HNeRV implicit representation; no SegNet-mask cover to perturb). Memory: `feedback_pr106_no_mask_channel_lanes_05_06_falsified_20260504.md`.
- **Lane #06 grayscale-LUT → PR106 mask channel**: Same root cause as #05.

The pattern: revival plans applied generic templates ({codec X} → {component Y}) without verifying PR106 actually has the relevant component. Lane Ω-W-V3 (water_filling_codec_v2) was the only schema-agnostic + distribution-handling lane that applied to PR106 unchanged.

Consolidated audit memo: `feedback_audit_pr106_revival_plans_consolidated_findings_20260504.md`.

## Custody pointers

| Asset | Path |
|---|---|
| PR106 source archive | `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip` |
| PR106 extracted state_dict | `experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt` (924KB) |
| PR106 extracted latents | `experiments/results/sensitivity_map_pr106_20260504_claude/latents.pt` (68KB) |
| PR106 sensitivity map (CPU stub) | `experiments/results/sensitivity_map_pr106_20260504_claude/sensitivity_map_stub.pt` (28KB) |
| Apogee_v2 (Lane Ω-W-V3) inflate adapter | `submissions/apogee_v2/inflate.{py,sh}` + `src/{model,codec}.py` |
| Lane Ω-W-V3 4-stage GPU dispatch wrapper | `scripts/remote_lane_omega_w_v3_pr106.sh` |
| Lane #04 int4 repack | `experiments/repack_pr106_with_int4_block_fp.py` |
| Lane SJ-KL C067 dispatch wrapper | `scripts/remote_lane_sjkl_c067.sh` |
| SJ-KL pipeline (5 modules) | `experiments/{prepare_sjkl_pair_tensors,build_sjkl_residual,build_sjkl_c067_archive}.py` + `src/tac/sjkl_basis.py` + `submissions/robust_current/unpack_renderer_payload.py` |

All SJ-KL modules + the canonical uv installer + dispatch ledger were rebuilt this session — see `docs/recovery_report_20260504.md` for the full lost-helper recovery log.
