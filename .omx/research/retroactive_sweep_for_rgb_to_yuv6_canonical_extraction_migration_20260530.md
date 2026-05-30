---
title: Retroactive Sweep for rgb_to_yuv6 Canonical Extraction Migration
date: 2026-05-30
catalog_ref: Catalog #348 sister discipline
lane_id: lane_rgb_to_yuv6_canonical_extraction_migration_20260530
migration_landing_memo: feedback_rgb_to_yuv6_canonical_extraction_migration_landed_20260530.md
---

# Retroactive sweep per Catalog #348 sister discipline

## Bug-class symptom signature

3 sister rgb_to_yuv6 implementations (`tac.constrained_gen.rgb_to_yuv6`,
`tac.saliency.rgb_to_yuv6`, `tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx`)
duplicated the canonical BT.601 4:2:0 chroma subsampling math
INDEPENDENTLY per audit inventory A.2.6. Each sister carried byte-stable
math but diverged in tensor layout (NCHW / HWC / NHWC) and dtype
(float32 / float64). A 4th sister
(`tac.composition.yuv6_chroma_subsampled_perturbation_operator.rgb_to_yuv6_numpy`)
retained PRINCIPLED FORK status per Catalog #290 falling-rule list
because the canonical helper is float32-native and the operator
downstream depends on float64 precision.

## Pre-fix window

Multiple sister landings predating the canonical extraction:
- `tac.constrained_gen.rgb_to_yuv6` — historical (sister of `tac.differentiable_eval_roundtrip` non-negotiable)
- `tac.saliency.rgb_to_yuv6` — historical (sister of upstream `frame_utils.rgb_to_yuv6`)
- `tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx` — sister of PR95-family MLX training stack
- `tac.composition.yuv6_chroma_subsampled_perturbation_operator.rgb_to_yuv6_numpy` — sister of Yousfi-blind-spot operator

## Historical KILL / DEFER / FALSIFY search results

`grep -rni "KILL\|FALSIFY\|DEFER" .omx/research/*.md ~/.claude/projects/-Users-adpena-Projects-pact/memory/*.md 2>/dev/null | grep -i "rgb_to_yuv6\|yuv6_to_rgb"`

Result: **0 historical KILL / DEFER / FALSIFY verdicts** affecting rgb_to_yuv6.

This canonical extraction migration is **purely additive** at the
canonicalization surface — no prior empirical falsification claims need
to be revisited.

## Per-finding RE-EVAL priority assignment

No historical findings affected by this migration. The migration's
structural value is **prospective**: future sister implementations of
rgb_to_yuv6 that emerge will inherit the canonical math via
`from tac.framework_agnostic.canonical_kernels import rgb_to_yuv6` —
preventing future math drift across the rgb_to_yuv6 / yuv6_to_rgb sister
surface per audit A.3 "0% → ~75% adoption rate".

## Verification

All 21 new dedicated migration parity tests + 125 sister test surfaces
pass with **0 regressions** detected. Byte-stable parity empirically
verified across:
- NCHW 4D / 5D / 6D leading-dim patterns (`constrained_gen`)
- HWC 3D / 4D / 5D leading-dim patterns (`saliency`)
- NHWC 4D / odd-crop / 5D leading-dim patterns (`pr95_hnerv_mlx`)
- Gradient propagation preserved (PyTorch sisters)
- Composition operator byte-identical to pre-migration form (PRINCIPLED FORK)
- Canonical helper divergence ~2.13e-5 documented as empirical anchor for the PRINCIPLED FORK rationale

## Conclusion

This canonical extraction migration is a **structural protection** that
prevents future math drift across the rgb_to_yuv6 / yuv6_to_rgb sister
surface. No historical verdicts are tainted; no retroactive RE-EVAL is
needed.
