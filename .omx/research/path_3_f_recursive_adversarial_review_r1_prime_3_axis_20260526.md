<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — review memo; do not mutate. -->
<!-- Catalog #229 PV closure: read landing memo + mlx_renderer.py + archive.py + inflate.py + tests/test_basic.py in full BEFORE any review claim. 16/16 tests verified passing. Empirical MLX↔PyTorch parity measurements run on Z8's _pixel_shuffle_2x_nhwc (max_abs=3.77) and _bilinear_resize_2x_nhwc (max_abs=1.51) per Axis 2. -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rao, Ballard, Mallat, Tishby-memorial, Wyner, Hafner-DreamerV3-author-cite, Carmack, Hotz, Quantizr, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Z8 inheriting A=DreamerV3 channel-LAST PixelShuffle + mx.repeat bilinear is correct because the landing pre-dates FIX-WAVE-R1 closing those bugs"
    classification: CARGO-CULTED
    rationale: "Empirically falsified by R1' measurement: Z8's `_pixel_shuffle_2x_nhwc` produces 3.77 max_abs vs PyTorch's `nn.PixelShuffle(2)` (sister D=Z6 canonical: 0.0 drift). Z8's `_bilinear_resize_2x_nhwc` (mx.repeat 2x) produces 1.51 max_abs vs PyTorch's `F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)`. These are TRAINING-INVALIDATING per R1 META finding #5: MLX trainer optimizes against MLX_buggy_decoder; PyTorch inflate uses CORRECT canonical primitives; state_dict-transferred frames at PyTorch-inflate DO NOT MATCH frames MLX trainer observed at convergence. The canonical fix EXISTS (PR95 helper landed during FIX-WAVE-R1); Z8 must adopt it."
  - assumption: "Z8's 16/16 PASS tests prove the MLX renderer is correct"
    classification: CARGO-CULTED
    rationale: "The 16 tests verify SHAPE correctness, archive byte-determinism, manifest observability, and SMOKE CONVERGENCE on SYNTHETIC RANDOM TARGETS. NONE of them test MLX↔PyTorch parity at the decoder forward boundary. Per the canonical R1 anchor on A=DreamerV3: the L0 smoke trainer 'loss decreased' on synthetic random targets does NOT reveal the bug because targets are noise. The bug surfaces structurally at L1+ score-aware-loss training. Tests pass != correctness when the test surface is the wrong shape."
  - assumption: "FIX-WAVE-R1' can land both Z8 MLX-primitive fixes in a single commit batch via canonical helper import"
    classification: HARD-EARNED
    rationale: "Z8's bugs are LINE-FOR-LINE the same as A=DreamerV3's pre-FIX-WAVE-R1 bugs. The fix is a near-mechanical port of A=DreamerV3's FIX-WAVE-R1 patches (commit `a23779a732e7bb056`): (1) rewrite `_pixel_shuffle_2x_nhwc` per channel-FIRST convention (matching D=Z6 / canonical PR95); (2) replace `_bilinear_resize_2x_nhwc` with import of canonical `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`. Touch surface: 1 file (`mlx_renderer.py`); ≤30 LOC of edits. R2' BLOCKED until FIX-WAVE-R1' lands."
council_decisions_recorded:
  - "R1' verdict: NOT CLEAN — counter resets to 0 for F per CLAUDE.md 'Recursive adversarial review protocol — close paths' item 3"
  - "FIX-WAVE-R1' required BEFORE R2' fires per protocol item 4"
  - "FIX-WAVE-R1' op-routables: F-OP1 (rewrite _pixel_shuffle_2x_nhwc per D=Z6 sister-canonical) + F-OP2 (replace _bilinear_resize_2x_nhwc with canonical PR95 helper import) + F-OP3 (add MLX↔PyTorch parity tests for both primitives mirroring A=DreamerV3 post-FIX-WAVE-R1 verification)"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526
  - path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526
  - path_3_a_recursive_adversarial_review_r1_3_axis_20260526
  - path_3_fix_wave_r1_close_findings_landed_20260526
  - path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526
canonical_equation_refs:
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
---

# Path 3 candidate F=Z8 — R1' 3-axis recursive adversarial review

**Verdict**: **PROCEED_WITH_REVISIONS — R1' NOT CLEAN for F=Z8** — counter resets to 0; FIX-WAVE-R1' successor subagent required BEFORE R2' fires.

**Commit under review**: `5ff5d2ab9` (`z8/path-3-f: land canonical-quadruple binding L0 SCAFFOLD (Catalog #312)`).

**Cost**: $0 GPU; ~60 min wall-clock (PV + 3-axis review + empirical MLX parity measurement + memo).

---

## Premise verification (Catalog #229)

| File | Purpose | LOC |
|---|---|---|
| `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md` | Landing memo | 199 |
| `.omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md` (cited; ~48 KB) | Design memo (Section 1-14) | — |
| `src/tac/substrates/z8_hierarchical_predictive_coding/__init__.py` | Catalog #124 8-field declaration + waiver + canonical equation refs | — |
| `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py` | MLX hierarchy + Mallat sum-pool proxy + DreamerV3 linear-gate proxy + decoder | 668 |
| `src/tac/substrates/z8_hierarchical_predictive_coding/archive.py` | Z8HPC1 grammar | 568 |
| `src/tac/substrates/z8_hierarchical_predictive_coding/inflate.py` | PyTorch inflate stub raising Z8L0ScaffoldNotImplementedError | — |
| `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py` | 16 tests | — |

**Empirical reproducer**: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/ -q` → **16 passed in 0.45s** (verified by R1').

---

## Axis 1 review: Math + scientific + engineering rigor

### Per-architectural-choice HARD-EARNED vs CARGO-CULTED classification

| Choice | Source | Classification | Rationale |
|---|---|---|---|
| **Canonical-quadruple binding (Rao-Ballard + Mallat + DreamerV3 + Wyner-Ziv)** | Landing memo §"What landed" + design memo Section 1-14 | HARD-EARNED | Citations to canonical papers (Rao-Ballard 1999; Mallat 1989; Hafner DreamerV3; Wyner-Ziv 1976). Catalog #312 satisfies canonical-quadruple requirement per CLAUDE.md non-negotiable. |
| **Multiplicative joint entropy reduction bound** | Landing memo lines 145-149 + design memo Section 4 | HARD-EARNED-PARTIAL | Multiplicative bound `(1 - r_RB)(1 - r_Mallat)(1 - r_DreamerV3)(1 - r_WZ)` is UPPER bound; Dykstra-feasibility per landing memo line 151 explicitly notes this is planning prior; "true achievable is the convex-intersection projection (subadditive)". Predicted band [0.05, 0.10] tagged `pending_post_training` per Catalog #324. |
| **3-level Rao-Ballard hierarchy default** | `mlx_renderer.py` line 64 (DEFAULT_NUM_LEVELS=3) | HARD-EARNED | Citation to "canonical Rao-Ballard visual cortex 3-level model". |
| **DEFAULT_NUM_GROUPS_PER_LEVEL=(24, 16, 8) descending** | `mlx_renderer.py` line 67 | HARD-EARNED | Rationale "Decreases at deeper levels per Mallat coarse-fine wavelet bound (fewer groups needed at coarser scales)". |
| **DEFAULT_NUM_CATEGORIES_PER_LEVEL=(256, 128, 64) descending** | `mlx_renderer.py` line 71 | HARD-EARNED | Same Mallat coarse-fine rationale. |
| **Mallat sum-pool proxy at L0 (full Daubechies-4 DWT deferred)** | `mlx_renderer.py` lines 230-256 | HARD-EARNED-PARTIAL | Honestly disclosed as "L0 scaffold proxy"; Phase 2 lands full DWT. Acceptable as scaffold-level placeholder. |
| **DreamerV3 linear-gate proxy at L0 (full GRUCell deferred)** | `mlx_renderer.py` lines 369-373 | HARD-EARNED-PARTIAL | Honestly disclosed; Phase 2 lands GRUCell. Acceptable as scaffold-level placeholder. |
| **Gumbel-Softmax STE reuse from A=DreamerV3 (canonical adoption)** | `mlx_renderer.py` lines 195-227 | HARD-EARNED | Sister A=DreamerV3 canonical implementation reused per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES. |
| **Decoder topology = canonical PR95 HNeRV** | `mlx_renderer.py` lines 380-410 + Landing memo line 116 | HARD-EARNED | Reuses sister A=DreamerV3 PixelShuffle block pattern; "empirically validated PR95/PR101/PR110 medal-class topology". |
| **Z8HPC1 byte-deterministic 62-byte header + 5 distinguishing sections** | Landing memo line 117 + archive.py | HARD-EARNED | All 5 sections (DECODER + INDICES + WAVELET + WYNER_ZIV + DREAMER_STATE) covered by Catalog #139 + #272 byte-mutation no_op_proof; verified PASS. |
| **18/18 PASS verdict table** | Landing memo lines 87-108 | HARD-EARNED | Empirical receipts cited per test name; reproducer commands documented. |
| **6-hook wire-in declaration** | Landing memo lines 165-172 | HARD-EARNED | All 6 hooks explicitly declared per Catalog #125 non-negotiable. |

**Net Axis 1 per-architectural-choice classification**: 8 HARD-EARNED + 4 HARD-EARNED-PARTIAL + **0 CARGO-CULTED at the architectural-decision level**.

### Findings (Axis 1)

**0 findings**. The architectural decisions are HARD-EARNED with explicit canonical-paper citations and Catalog #290 canonical-vs-unique decision per layer documented. The L0 SCAFFOLD discipline (proxy primitives at L0; full DWT + GRUCell deferred to Phase 2) is honest per Catalog #287 anti-overstatement.

**However**, Axis 2 below surfaces CRITICAL bugs at the MLX IMPLEMENTATION layer (not the architectural-decision layer). The CARGO-CULTED finding in the Assumption-Adversary verdict refers to Z8 silently inheriting A=DreamerV3's pre-FIX-WAVE-R1 MLX-primitive bugs, NOT to the architectural decisions themselves.

---

## Axis 2 review: MLX drift minimization (CRITICAL FINDINGS)

### Per-MLX-primitive empirical drift measurement

**Z8 ships 7 MLX-callable primitives**:
1. `_mallat_sum_pool_2x_nhwc` (lines 240-256)
2. `_pixel_shuffle_2x_nhwc` (lines 264-276) ← **EMPIRICALLY MEASURED BUG**
3. `_bilinear_resize_2x_nhwc` (lines 279-284) ← **EMPIRICALLY MEASURED BUG**
4. `gumbel_softmax_sample` (lines 195-227)
5. `_Z8UpsampleBlock.__call__` (lines 301-306; depends on #2 + #3 + Conv2d + sin)
6. `_decoder_forward` (lines 411-427; depends on #2 + #3 + #5 + stem Linear + refine0/1 Conv2d + rgb_0/1 Conv2d + sigmoid + sin + stack + reshape + transpose)
7. `Z8HierarchicalPredictiveCoderMLX.__call__` and `forward_training`/`forward_eval_from_indices` (the substrate's full forward; depends on #1-6)

### Per-primitive measurement results

**R1' empirical measurement (reproducer):**

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -c "
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import _pixel_shuffle_2x_nhwc as z8_pixshuf, _bilinear_resize_2x_nhwc as z8_bilin
from tac.substrates.time_traveler_l5_z6.mlx_renderer import _pixel_shuffle_2x_nhwc as z6_pixshuf

np.random.seed(0)
x_np = np.random.randn(1, 4, 4, 16).astype(np.float32)
x_mlx = mx.array(x_np)
x_torch = torch.from_numpy(x_np).permute(0, 3, 1, 2)
y_torch_nhwc = F.pixel_shuffle(x_torch, 2).permute(0, 2, 3, 1).numpy()
y_z8 = np.array(z8_pixshuf(x_mlx))
y_z6 = np.array(z6_pixshuf(x_mlx))
print(f'Z8 vs PyTorch max_abs: {np.abs(y_z8 - y_torch_nhwc).max():.6f}')
print(f'Z6 vs PyTorch max_abs: {np.abs(y_z6 - y_torch_nhwc).max():.6f}')

x2_np = np.random.randn(1, 8, 8, 4).astype(np.float32)
x2_mlx = mx.array(x2_np)
x2_torch = torch.from_numpy(x2_np).permute(0, 3, 1, 2)
y2_torch_nhwc = F.interpolate(x2_torch, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).numpy()
y2_z8 = np.array(z8_bilin(x2_mlx))
print(f'Z8 bilinear vs PyTorch max_abs: {np.abs(y2_z8 - y2_torch_nhwc).max():.6f}')
"
```

**Output (verified by R1' 2026-05-26T08:03Z)**:
```
Z8 vs PyTorch max_abs: 3.766418
Z6 vs PyTorch max_abs: 0.000000
Z8 bilinear vs PyTorch max_abs: 1.512860
```

### Per-primitive drift summary

| Primitive | Z8 max_abs vs PyTorch | Sister D=Z6 max_abs | Canonical PR95 helper max_abs | Verdict |
|---|---|---|---|---|
| `_pixel_shuffle_2x_nhwc` (Z8 lines 274-276 with transpose `(0, 1, 3, 2, 4, 5)` channel-LAST convention) | **3.77** | **0.00** (channel-FIRST `(0, 1, 4, 2, 5, 3)`) | **0.00** | **TRAINING-INVALIDATING** |
| `_bilinear_resize_2x_nhwc` (Z8 lines 282-284 via `mx.repeat`) | **1.51** | N/A (D=Z6 uses canonical helper) | **0.00** (canonical helper) | **TRAINING-INVALIDATING** |
| `_mallat_sum_pool_2x_nhwc` (Z8 lines 240-256) | Not measured (no PyTorch sister; L0 proxy) | N/A | N/A | OK at L0 (proxy; Phase 2 lands DWT) |
| `gumbel_softmax_sample` (Z8 lines 195-227) | Not measured (cannot meaningfully compare stochastic output across frameworks) | N/A | A=DreamerV3 sister (canonical adoption) | Inherited; same as sister A |
| `_Z8UpsampleBlock` (depends on PixelShuffle + bilinear) | **Inherits 3.77 + 1.51 drift compounded** | N/A | N/A | TRAINING-INVALIDATING |
| `_decoder_forward` (6 upsample blocks) | **Drift compounds 6× via sequential block stack** | N/A | N/A | TRAINING-INVALIDATING |

### Bug class anchor: identical to A=DreamerV3 pre-FIX-WAVE-R1

Z8's `_pixel_shuffle_2x_nhwc` at lines 274-276:
```python
y = mx.reshape(x, (batch, height, width, 2, 2, out_channels))
y = mx.transpose(y, (0, 1, 3, 2, 4, 5))
return mx.reshape(y, (batch, height * 2, width * 2, out_channels))
```

This is **LINE-FOR-LINE IDENTICAL** to A=DreamerV3's PRE-FIX-WAVE-R1 buggy implementation (the bug R1 identified as 2.4 absolute drift). FIX-WAVE-R1 (commit `a23779a732e7bb056`) replaced it with the channel-FIRST convention:
```python
y = mx.reshape(x, (batch, height, width, out_channels, 2, 2))
y = mx.transpose(y, (0, 1, 4, 2, 5, 3))
return mx.reshape(y, (batch, height * 2, width * 2, out_channels))
```

Sister D=Z6 (canonical sister-canonical reference per R1 META finding #3) uses the channel-FIRST convention. Canonical PR95 helper `tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc` uses the channel-FIRST convention.

Z8's `_bilinear_resize_2x_nhwc` at lines 282-284:
```python
y = mx.repeat(x, 2, axis=1)
y = mx.repeat(y, 2, axis=2)
return y
```

This is **LINE-FOR-LINE IDENTICAL** to A=DreamerV3's PRE-FIX-WAVE-R1 buggy implementation (24.34 max_abs drift bug). FIX-WAVE-R1 replaced it with a delegating import of `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`.

**Mechanism (per R1 META finding #5)**: Z8's MLX trainer optimizes against `MLX_buggy_decoder(weights) → MLX_frames`. The PyTorch inflate (`inflate.py` raises Z8L0ScaffoldNotImplementedError at L0; Phase 2 lands the actual inflate; but per design memo Phase 2 trainer uses canonical PyTorch `nn.PixelShuffle(2)` + `F.interpolate(mode='bilinear', align_corners=False)`). After MLX-training convergence + state_dict export, the rendered frames at PyTorch-inflate time DO NOT MATCH the frames the MLX trainer observed at convergence. The L0 smoke trainer's "2537.68 → 2503.03 → 2500.61 monotonic decrease" on SYNTHETIC random targets does NOT reveal this because targets are noise; the bug surfaces structurally at L1+ score-aware-loss training.

**Catalog #1265 implications**: the canonical MLX↔PyTorch contest-equivalence gate's threshold (`|S_MLX − S_PyTorch| ≤ 0.001 contest-units`) will FAIL on Z8 archives until FIX-WAVE-R1' lands. The sister gate planned per landing memo op-routable #1 (Z8HPC1 grammar gate extension) CANNOT PASS until the underlying decoder forward semantics match.

### Findings (Axis 2)

**F-OP1 (P0 / CRITICAL / TRAINING-INVALIDATING)**: rewrite `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py::_pixel_shuffle_2x_nhwc` from channel-LAST convention `(B, H, W, 2, 2, out_C)` + transpose `(0, 1, 3, 2, 4, 5)` to channel-FIRST convention `(B, H, W, out_C, 2, 2)` + transpose `(0, 1, 4, 2, 5, 3)` matching the SISTER-CANONICAL impl in D=Z6 + canonical PR95 helper + FIX-WAVE-R1 fixed A=DreamerV3 pattern. Empirically verified: D=Z6's convention is byte-stable vs PyTorch reference (0.0 drift).

**F-OP2 (P0 / CRITICAL / TRAINING-INVALIDATING)**: replace `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py::_bilinear_resize_2x_nhwc` (mx.repeat 2x nearest-neighbor) with import + usage of canonical `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`. Empirically verified: canonical helper is byte-stable vs PyTorch reference.

**F-OP3 (P1 / VERIFICATION)**: after F-OP1 + F-OP2 land, add test `test_z8_pixel_shuffle_matches_pytorch` + `test_z8_bilinear_resize_matches_pytorch` to `tests/test_basic.py` mirroring A=DreamerV3 post-FIX-WAVE-R1 verification pattern. Both tests should assert max_abs < 1e-5.

---

## Axis 3 review: Portability via numpy

### Per-MLX-primitive numpy reference status

| Primitive | numpy reference exists? | Sister substrate canonical? |
|---|---|---|
| `_mallat_sum_pool_2x_nhwc` | NO | NONE (Z8 is the first; should add numpy reference at L1 per G=NIRVANA sister pattern) |
| `_pixel_shuffle_2x_nhwc` | NO (in z8_hierarchical_predictive_coding/) | Canonical PR95 helper has MLX impl only; G=NIRVANA `numpy_reference.py` has no pixel_shuffle reference either; META-CONSOLIDATE-OP candidate |
| `_bilinear_resize_2x_nhwc` | NO (in z8_hierarchical_predictive_coding/) | G=NIRVANA `numpy_reference.bilinear_upsample_2x_nhwc` exists (correct align_corners=False reference); META-CONSOLIDATE-OP candidate |
| `gumbel_softmax_sample` | NO | A=DreamerV3 (no numpy reference); inherited gap |
| `_Z8UpsampleBlock` | NO | None |
| `_decoder_forward` | NO | None |

### CPU-only test rig operability

Z8's `mlx_renderer.py` has a `_require_mlx()` guard (lines 79-85); the test suite includes `test_z8_substrate_package_imports` (passes without MLX). However, the 16/16 tests include MLX-dependent tests (`test_z8_mlx_config_defaults_validate`, `test_z8_mlx_renderer_constructs_and_forward`, `test_z8_mlx_renderer_eval_from_indices_matches_shape`, `test_z8_mlx_architecture_manifest_observability_surface`, `test_z8_decoder_param_count_increases_with_levels`). These tests are SKIPPED on non-Apple-Silicon CI per the import guard.

### Findings (Axis 3)

**F-OP4 (P2 / ADVISORY; not blocking R1')**: when L1 lands the canonical META-CONSOLIDATE-OP (extract `_pixel_shuffle_2x_nhwc` + `_bilinear_resize_2x_nhwc` to canonical `tac.local_acceleration.pr95_hnerv_mlx`), the canonical helper SHOULD also ship a sister numpy reference at `tac.local_acceleration.numpy_reference` (or equivalent canonical location). This addresses both META-finding #1 from R1 aggregate (locally-invented primitives diverge) AND Axis 3 portability gap.

---

## R1' verdict for F=Z8

**Per-axis verdicts**:
- Axis 1 (math + sci + engineering rigor): **CLEAN** (0 findings)
- Axis 2 (MLX drift minimization): **NOT CLEAN** (2 CRITICAL TRAINING-INVALIDATING + 1 P1 verification)
- Axis 3 (numpy portability): **CLEAN-WITH-ADVISORY** (1 P2 advisory)

**Aggregate**: **PROCEED_WITH_REVISIONS — R1' NOT CLEAN**. Counter resets to **0/3** per CLAUDE.md protocol item 3.

**FIX-WAVE-R1' required BEFORE R2' fires** per protocol item 4.

---

## Counter state per CLAUDE.md "Recursive adversarial review protocol — close paths"

- **Before R1'**: counter = 0 (F is NEW landing post-R1; no prior cycle history)
- **R1' verdict**: NOT CLEAN → counter remains at **0/3** per protocol item 3 (any issue resets to 0)
- **R2' BLOCKED until FIX-WAVE-R1' lands**:
  - F-OP1 + F-OP2 (code fixes; ≤30 LOC across 1 file)
  - F-OP3 (verification tests; ≤30 LOC across 1 file)
- **R2' verification post-FIX-WAVE-R1'**:
  - Re-run 16 tests; verify all PASS
  - Add 2 new parity tests per F-OP3; verify both PASS with max_abs < 1e-5
  - Re-measure Z8 PixelShuffle + bilinear drift; verify both 0.0 max_abs vs PyTorch

---

## FIX-WAVE-R1' op-routables for F=Z8 (priority-ranked)

### P0 / CRITICAL / TRAINING-INVALIDATING

**F-OP1**: rewrite `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py::_pixel_shuffle_2x_nhwc` (lines 264-276):

```python
def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    """PixelShuffle 2x for NHWC tensors (canonical channel-FIRST convention).

    Per R1' Path 3 F review (2026-05-26): channel-FIRST convention matching
    sister D=Z6 + canonical PR95 helper + post-FIX-WAVE-R1 A=DreamerV3.
    The prior channel-LAST convention produced 3.77 absolute drift vs PyTorch
    nn.PixelShuffle(2); this convention is byte-stable (0.0 drift).
    """
    _require_mlx()
    batch, height, width, channels = (int(dim) for dim in x.shape)
    block = 4  # 2*2
    if channels % block != 0:
        raise ValueError(
            f"channels {channels} must be divisible by {block} for 2x pixel shuffle"
        )
    out_channels = channels // block
    y = mx.reshape(x, (batch, height, width, out_channels, 2, 2))
    y = mx.transpose(y, (0, 1, 4, 2, 5, 3))
    return mx.reshape(y, (batch, height * 2, width * 2, out_channels))
```

**F-OP2**: replace `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py::_bilinear_resize_2x_nhwc` (lines 279-284):

```python
def _bilinear_resize_2x_nhwc(x: Any) -> Any:
    """Bilinear upsample 2x for NHWC tensors via canonical PR95 helper.

    Per R1' Path 3 F review (2026-05-26): delegates to canonical
    `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`
    which is empirically PyTorch-byte-stable (0.0 drift) vs the prior
    mx.repeat 2x approximation that produced 1.51 absolute drift vs PyTorch
    F.interpolate(scale_factor=2, mode='bilinear', align_corners=False).
    """
    _require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx import (
        bilinear_resize2x_align_corners_false_nhwc,
    )
    return bilinear_resize2x_align_corners_false_nhwc(x)
```

### P1 / VERIFICATION

**F-OP3**: add to `tests/test_basic.py`:

```python
def test_z8_pixel_shuffle_matches_pytorch():
    """Z8 _pixel_shuffle_2x_nhwc must be byte-stable vs PyTorch nn.PixelShuffle(2)."""
    try:
        import mlx.core as mx
        import torch
        import torch.nn.functional as F
        from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import _pixel_shuffle_2x_nhwc
    except ImportError:
        pytest.skip("MLX or torch not available")
    np.random.seed(0)
    x_np = np.random.randn(1, 4, 4, 16).astype(np.float32)
    y_torch = F.pixel_shuffle(torch.from_numpy(x_np).permute(0, 3, 1, 2), 2)
    y_torch_nhwc = y_torch.permute(0, 2, 3, 1).numpy()
    y_mlx = np.array(_pixel_shuffle_2x_nhwc(mx.array(x_np)))
    max_abs = np.abs(y_mlx - y_torch_nhwc).max()
    assert max_abs < 1e-5, f"Z8 _pixel_shuffle_2x_nhwc drift {max_abs} > 1e-5 vs PyTorch"


def test_z8_bilinear_resize_matches_pytorch():
    """Z8 _bilinear_resize_2x_nhwc must be byte-stable vs PyTorch F.interpolate."""
    try:
        import mlx.core as mx
        import torch
        import torch.nn.functional as F
        from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import _bilinear_resize_2x_nhwc
    except ImportError:
        pytest.skip("MLX or torch not available")
    np.random.seed(0)
    x_np = np.random.randn(1, 8, 8, 4).astype(np.float32)
    y_torch = F.interpolate(
        torch.from_numpy(x_np).permute(0, 3, 1, 2),
        scale_factor=2, mode='bilinear', align_corners=False,
    )
    y_torch_nhwc = y_torch.permute(0, 2, 3, 1).numpy()
    y_mlx = np.array(_bilinear_resize_2x_nhwc(mx.array(x_np)))
    max_abs = np.abs(y_mlx - y_torch_nhwc).max()
    assert max_abs < 1e-5, f"Z8 _bilinear_resize_2x_nhwc drift {max_abs} > 1e-5 vs PyTorch"
```

### P2 / ADVISORY

**F-OP4**: L1+ META-CONSOLIDATE-OP for `_pixel_shuffle_2x_nhwc` canonical helper sister numpy reference (covered in R1 aggregate META-CONSOLIDATE-OP-1).

---

## Discipline applied

- **Catalog #229 PV**: landing memo + design memo + mlx_renderer.py + tests + sister A=DreamerV3 post-FIX-WAVE-R1 module.py + sister D=Z6 mlx_renderer.py read in full; 16 tests verified PASS; empirical MLX↔PyTorch parity measurement run
- **Catalog #110/#113 APPEND-ONLY**: NEW review memo; landing memo + design memo NEVER mutated
- **Catalog #287 placeholder-rationale rejection**: every assumption-adversary verdict carries non-placeholder rationale
- **Catalog #292 per-deliberation assumption surfacing**: per-axis council members declared; CARGO-CULTED assumption explicitly identified at the implementation-vs-architectural-decision boundary per Catalog #307 classification
- **Catalog #300 v2 frontmatter**: full T2 frontmatter
- **Catalog #340 sister-checkpoint guard**: PROCEED verdict
- **Catalog #307 paradigm-vs-implementation classification**: F=Z8's findings are IMPLEMENTATION-LEVEL (MLX primitive bugs), NOT PARADIGM-LEVEL refutations of the canonical-quadruple binding paradigm. Per CLAUDE.md "Forbidden premature KILL without research exhaustion": substrate paradigm INTACT; only L0 MLX primitives require FIX-WAVE-R1' patches
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8 honored
- **CLAUDE.md "Executing actions with care"**: review-only

---

## Cross-references

- Landing memo: `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md` (commit `5ff5d2ab9`)
- Design memo: `.omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md`
- Sister A=DreamerV3 R1 review (analogous CRITICAL bugs; FIXED in FIX-WAVE-R1): `.omx/research/path_3_a_recursive_adversarial_review_r1_3_axis_20260526.md`
- FIX-WAVE-R1 landing (canonical fix pattern for analogous A=DreamerV3 bugs): `.omx/research/path_3_fix_wave_r1_close_findings_landed_20260526.md` (commit `a23779a732e7bb056`)
- Canonical D=Z6 sister-canonical reference: `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc` (lines 361-372)
- Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`
- R1 aggregate META finding #1 (locally-invented MLX primitives diverge from sister-canonical): `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md` § META findings
- Lane: `lane_path_3_recursive_adversarial_review_r1_prime_3_axis_landings_b_c_f_g_20260526` L0
