# Lane Ω-W — Water-filling Lagrangian bit-budget allocator (design)

**Date**: 2026-04-29
**Author**: adversarial review council (Shannon LEAD + Dykstra CO-LEAD + Fridrich + Yousfi + Contrarian + Ballé + MacKay)
**Status**: BINDING design for Session-2 implementation. No vagueness. The implementor follows this doc verbatim.
**Predicted band**: **-0.02 to -0.04 score** vs Lane A 1.15 ([contest-CUDA] tag), i.e. ship 1.11-1.13.
**Lane code**: `Ω-W` (water-filling). Distinct from Lane Ω-Hessian (per-WEIGHT, deferred) and Lane SO (broken Hessian fallback, retired).

---

## 1. Mathematical formulation

### 1.1 Objective (Shannon)

We have a SegMap renderer with `C` total per-output-channel quantization slots distributed across `L` conv layers. Each slot `c` carries weight tensor `w_c ∈ R^{I·kH·kW}`. After block-FP encoding with `Q_c = qint_max[c]`, the integer codeword consumes ≈ `b_c = log2(2·Q_c + 1)` bits per element (signed integers in `[-Q_c, +Q_c]`).

The high-rate quantization-noise model (Gish & Pierce 1968; standard rate-distortion text) gives mean-squared reconstruction error per channel:

    ε_c(b_c) ≈ (σ_c² · 2^(-2·b_c)) / 12

where `σ_c²` is the per-channel weight variance (after the block-FP exponent `e_c` has factored out the dynamic-range term `2^(2·e_c)`). The **score-weighted** distortion uses Hessian curvature `H_c = E_pairs[(∂L_render/∂w_c)²]` summed across the channel's elements:

    D_c(b_c) = H_c · σ_c² · 2^(-2·b_c) / 12

Lagrangian (Dykstra convex feasibility on the bit-allocation simplex):

    minimize    Σ_c D_c(b_c)
    subject to  Σ_c b_c ≤ B,    b_c ≥ 0

### 1.2 KKT optimum (Shannon water-filling)

Stationarity: `∂D_c/∂b_c = -2 ln(2) · D_c(b_c) = -λ`, equating across active channels:

    H_c · σ_c² · 2^(-2·b_c) = λ / (2 ln 2)

Solve for `b_c`:

    b_c* = max(0, 0.5 · log2(H_c · σ_c² · 2 ln 2 / λ))

The constants `2 ln 2` get absorbed into `λ`. Define the **utility** `u_c = H_c · σ_c²` and the canonical water-fill form:

    b_c* = max(0, 0.5 · log2(u_c / λ))                                (Eq. 1)

This is the textbook reverse-water-filling for parallel Gaussian channels (Cover & Thomas Ch. 10, Ballé 2018 §3.2).

### 1.3 Discrete bit→Q mapping

The block-FP encoder accepts integer `qint_max ∈ {1,3,7,15,31}`. The mapping from continuous `b_c` to discrete `Q_c`:

| `b_c` continuous | `Q_c` discrete | bits/element (signed) | comment |
|---|---|---|---|
| `[0, 1.5)` | 1 | ~1.58 | ternary `{-1, 0, 1}`, encode as int8 (waste 6.4 bits raw, ARITHMETIC CODER recovers it; see §6) |
| `[1.5, 2.5)` | 3 | ~2.81 | 7-level `{-3..+3}` |
| `[2.5, 3.5)` | 7 | ~3.91 | 15-level `{-7..+7}` (current uniform default) |
| `[3.5, 4.5)` | 15 | ~4.95 | 31-level `{-15..+15}` |
| `[4.5, ∞)` | 31 | ~5.97 | 63-level `{-31..+31}` |

**MacKay note**: discrete rounding cost is bounded by 0.5 bits/channel in this scheme; with ~200 eligible channels in SegMap, total rounding waste < 100 bits ≪ 1% of B (B is on the order of 480_000 bits). Acceptable.

**Round-down rule (Contrarian-required)**: when continuous `b_c < 1` we ROUND UP to `Q_c = 1` (never `Q_c = 0` — a zero-bit channel cannot be reconstructed without side information we don't ship). This is the "min_qint = 1" floor.

### 1.4 Budget enforcement (Dykstra)

After mapping continuous `b_c*` to discrete `Q_c`, compute realised total bits `B' = Σ_c bits(Q_c) · n_c` where `n_c` is the per-channel element count `I · kH · kW`. If `B' > B`, peel the lowest-utility channels down one Q-level at a time until `B' ≤ B`. If `B' < B - tol`, optionally promote highest-utility channels up (rarely needed; binary search on λ already minimises slack). Tolerance: `±1%` of B.

### 1.5 Binary search on λ

`u_c` spans many orders of magnitude. Bracket `λ ∈ [u_min · 2^(-20), u_max · 2^(+10)]` and bisect for 50 iterations or until `|Σ_c b_c* − B| < 1`. Each iteration is `O(C)` and pure Python — sub-millisecond.

---

## 2. Hessian estimation

### 2.1 Algorithm

1. Load SC++ inference checkpoint from `lane_a_landed/iter_0/segmap_inference.pt` (or supplied path).
2. Restore SegMap (`hidden=24, block_hidden=24, num_blocks=8, max_frame_index=1200`).
3. Set `model.train()` and `requires_grad_(True)` for all parameters.
4. Decode anchor masks via `decode_masks_auto(masks_mkv)` → `(N, H, W)` long mask classes.
5. Build calibration batch: pick `K = min(64, N)` evenly-spaced frames; one-hot to `(K, 5, H, W)`.
6. **eval_roundtrip path** (MANDATORY per repository protocol): forward pass MUST go through the rendering path that includes `_eval_roundtrip_chain` — i.e. quantise to uint8, decode, recompute. Reuse `tac.segmap_renderer._eval_roundtrip_chain`.
7. Loss: `L_render = MSE(model_out, gt_target)` where `gt_target` is the GT video frames at calibration indices (decoded from `upstream/videos/0.mkv` via `tac.video.decode_frames` if available; fall back to mask-derived target — design accepts both, implementor picks).
8. `L_render.backward()` → per-parameter `.grad`.
9. For each conv weight `(O, I, kH, kW)`: aggregate per output channel:
       `H_c = (grad[c]).pow(2).sum().item()` for c in 0..O-1
   This produces a `Tensor[O]` per layer.
10. Skip protected layers via `iter_eligible_conv_names()` from `learnable_bit_quant`.
11. Validate: every `H_c` must be finite. If NaN/Inf encountered → raise `ValueError` (no silent zero — Lane SO failure mode).

### 2.2 Variance estimation

Cheap closed-form on the conv weights themselves (no forward pass):

    σ_c² = w_c.var().item()        (per output channel, over I·kH·kW elements)

Edge case: if `σ_c² == 0` → channel is dead, set `u_c = 0`, water-fill assigns `Q_c = 1` (the floor, NOT zero — see §1.3 round-down rule).

### 2.3 Device policy

CUDA-required by default. CPU opt-in via `--device cpu` with a `[CPU-FALLBACK] Hessian numerics will differ from production CUDA path` banner. **NO MPS** (repository non-negotiable).

---

## 3. Module structure (Session 2 implements)

### 3.1 `src/tac/water_filling_codec.py`

Public API:

```python
from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from pathlib import Path

# --- exceptions ---
class WaterFillError(ValueError):
    """Raised when Hessian is non-finite OR water-fill cannot satisfy budget."""

# --- discrete bit ladder (binding) ---
QINT_LEVELS: tuple[int, ...] = (1, 3, 7, 15, 31)  # the only allowed Q values
QINT_BITS:   tuple[float, ...] = tuple(
    __import__("math").log2(2 * q + 1) for q in QINT_LEVELS
)  # bits/element (signed) per Q level

def bits_for_qint(q: int) -> float:
    """Signed-integer bit-count for one element at qint_max=q."""

def qint_for_bits(b: float) -> int:
    """Continuous→discrete: bin centres at b ∈ {1.0, 2.0, 3.0, 4.0, 5.0+}.
    Round-down rule: any b < 1.5 maps to qint_max=1 (the floor).
    Any b >= 4.5 maps to qint_max=31 (the ceiling).
    """

# --- core estimators ---
def estimate_per_channel_variance(
    model: nn.Module,
    extra_protected_patterns: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Return {<conv_name>.weight: Tensor[O]} of per-output-channel weight variance."""

def estimate_per_channel_hessian(
    model: nn.Module,
    calibration_inputs: torch.Tensor,        # (K, 5, H, W) one-hot masks
    calibration_targets: torch.Tensor,       # (K, 3, H, W) GT frames
    calibration_frame_idx: torch.Tensor,     # (K,) long
    *,
    eval_roundtrip: bool = True,             # MANDATORY True; False raises
    device: str = "cuda",
    extra_protected_patterns: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Return {<conv_name>.weight: Tensor[O]} per-channel Hessian via 1-step
    gradient approximation. Raises WaterFillError on NaN/Inf or eval_roundtrip=False.
    """

# --- water-fill ---
def water_fill_bit_budget(
    hessians: dict[str, torch.Tensor],       # name -> Tensor[O]
    variances: dict[str, torch.Tensor],      # name -> Tensor[O]
    channel_element_counts: dict[str, list[int]],  # name -> per-channel I*kH*kW
    total_bits: int,
    *,
    bisect_iters: int = 80,
    budget_tol_frac: float = 0.01,
) -> dict[str, list[int]]:
    """Returns {name: [Q_c, ...]} suitable for pack_payload_tar_xz(per_key_qint_max=...).
    Implements §1.2-§1.4. Raises WaterFillError if budget infeasible
    (e.g. all-Q1 floor exceeds B).
    """

# --- orchestrator ---
def export_with_water_filling(
    model: nn.Module,
    calibration_inputs: torch.Tensor,
    calibration_targets: torch.Tensor,
    calibration_frame_idx: torch.Tensor,
    total_bits: int,
    output_path: str | Path,
    *,
    device: str = "cuda",
    eval_roundtrip: bool = True,
    extra_protected_patterns: tuple[str, ...] = (),
    verify_tol: float = 1e-3,
) -> dict[str, object]:
    """End-to-end: H + σ² → water-fill → pack_payload_tar_xz → verify_roundtrip.
    Returns dict with: 'qint_assignment', 'realised_bits', 'payload_bytes',
    'roundtrip_mse_max', 'allocations_per_layer'.
    """
```

### 3.2 `src/tac/tests/test_water_filling_codec.py` (≥12 tests)

Names exactly as below (Contrarian-binding):

```
test_water_fill_meets_budget
test_water_fill_high_hessian_gets_more_bits
test_water_fill_zero_hessian_gets_minimum_bits
test_water_fill_uniform_hessians_uniform_bits
test_water_fill_invariant_to_global_scaling
test_water_fill_respects_max_qint
test_water_fill_respects_min_qint
test_export_roundtrip_below_tolerance
test_export_byte_savings_vs_uniform_q7
test_hessian_estimation_finite
test_variance_estimation_nonneg
test_edge_case_single_channel_layer
test_edge_case_nan_hessian_raises
test_qint_for_bits_round_trip
test_export_eval_roundtrip_false_raises
```

(15 tests total — exceeds Contrarian's 12-bar; adds 3 Codex-paranoia tests.)

### 3.3 `experiments/lane_omega_w_water_filling.py` (CLI)

```
usage: lane_omega_w_water_filling.py [--checkpoint PATH] [--anchor-masks PATH]
                                     [--gt-video PATH] [--total-bits INT]
                                     [--output-payload PATH] [--device cuda|cpu]
                                     [--num-calib INT]
```

Loads SC++ checkpoint, builds calibration batch, calls `export_with_water_filling`,
prints JSON summary.

### 3.4 `scripts/remote_lane_omega_w_water_filling.sh` (5-stage)

Stage 0: NVDEC probe (`scripts/probe_nvdec.sh`; DALI must come from the
canonical environment bootstrap, not silent wrapper auto-install)
Stage 1: anchor file checks (`renderer.bin`, `masks.mkv`, `optimized_poses.pt`)
Stage 2: load `segmap_inference.pt` from `experiments/results/lane_a_landed/iter_0/`
Stage 3: water-filling export at THREE budgets `{360_000, 480_000, 600_000}` bits
Stage 4: contest_auth_eval [contest-CUDA] on each archive, pick lowest-score
Stage 5: provenance update + RESULT_JSON

Heartbeat every 300 s; provenance.json at start AND completion. Tarball-only parity (NO `git pull` / `git reset --hard`).

---

## 4. Compliance checklist (binding)

- [x] eval_roundtrip=True default; `--no-eval-roundtrip` flag NOT defined; in-code `False` raises `WaterFillError`.
- [x] CUDA-required default; `--device cpu` opt-in with banner.
- [x] No scorer load at inflate (water_fill is COMPRESS-time only; archive ships block-FP qints + exponents only).
- [x] No `FakeQuantFP4` (Check 40 — water-fill uses block-FP int8, not FP4).
- [x] `pack_payload_tar_xz` called WITHOUT `exponents=` kwarg (Round 1 lesson — that kwarg does not exist).
- [x] Lane script ends with `experiments/contest_auth_eval.py` (Check 7).
- [x] Provenance.json with required fields (Check L: lane_id, started_at_utc, completed_at_utc, git_hash, gpu_name, archive_bytes, lane_status).
- [x] Tarball-only parity (Check 66/67/68/69 — NO `git pull` / `git reset --hard` on remote).
- [x] All `.py` files marked reviewed via `tools/review_tracker.py mark-file --reviewer council` AND `--reviewer codex` (2-distinct-approver gate per repository protocol).
- [x] Lane script carries `# E2E_SMOKE_OPT_OUT:` marker IFF tests land in same commit (Check e2e-smoke-proof).

---

## 5. Predicted EV (Shannon + empirical)

### 5.1 Bit savings

Lane A's renderer.bin packs ~287 K conv weights at uniform Q=7 (4 bits/element effective). Empirical Shannon entropy of trained conv weight distributions per Lane S logs ≈ 2.4 bits/element. Water-fill should reach ~2.8 bits/element average (close to entropy floor; gap is the discrete-Q ladder waste). Expected payload reduction: `(4.0 - 2.8) / 4.0 = 30%` of bytes.

On the current 90 KB renderer payload, that's ~27 KB saved → archive bytes drop from ~338 KB to ~311 KB → rate term per Quantizr scoring `(25 · archive_bytes / 37.5e6)` falls by `25 · 27_000 / 37.5e6 = 0.018`.

### 5.2 Distortion impact

Water-fill PROTECTS critical channels. Predicted PoseNet/SegNet distortion ≤ uniform Q=7 baseline (since uniform Q=7 wastes bits on irrelevant channels and starves the critical few). Conservative model: distortion unchanged. Optimistic: distortion -0.005 to -0.015.

### 5.3 Total Δ score

- Conservative: `Δ = -0.018 (rate)` → ship 1.13.
- Central: `Δ = -0.025 (rate -0.018, distortion -0.007)` → ship 1.12.
- Optimistic: `Δ = -0.040 (rate -0.022, distortion -0.018)` → ship 1.11.

**Predicted band**: `[1.11, 1.13]` [contest-CUDA]. Below this band → archive likely too small (B = 360_000 bit ladder hurts critical channels). Above this band → Hessian estimation noisy or eval_roundtrip not actually firing.

---

## 6. Council sign-offs

- **Shannon (LEAD)**: rate-distortion floor justified. Water-fill is the optimum solution for parallel Gaussian channels under L2 distortion; Eq. 1 is canonical (Cover & Thomas 10.4.2).
- **Dykstra (CO-LEAD)**: bit-budget simplex is a closed convex set; reverse water-fill is a single Lagrange-multiplier projection. No iteration needed beyond λ-bisection. Convex-feasibility guaranteed.
- **Fridrich**: per-channel utility `H_c · σ_c²` is the natural inverse-steganalysis embedding cost at the layer-channel granularity. Direct analog to UNIWARD's per-coefficient Wavelet absolute residual.
- **Yousfi**: matches HAWQ (Dong 2019) and OBQ (Frantar 2022) literature, ADAPTED to the contest by using rendering-loss Hessian (not full-dataset task loss). The rendering loss IS the proxy for PoseNet/SegNet sensitivity — established in `feedback_proxy_auth_math_useless` as 100-350× noisy on absolute scale but ranking-correlated.
- **Contrarian**: 15-test bar (12 + 3 paranoia). NaN raises hard (no silent zero). budget_tol_frac=1% prevents Lane SO's "fall back to uniform" silent failure. Q-floor=1 (not 0) prevents ghost-channel inflate failures.
- **Ballé**: per-channel Q metadata is implicit hyperprior — already shipped in pack_payload_tar_xz's meta.json. No new arithmetic-coder side-info needed (orthogonal to Lane SH if it lands).
- **MacKay**: discrete-rounding loss < 1% of B (Σ < 100 bits over ~200 channels). Bit allocation respects the MDL principle — bits spent ⇄ rate-distortion gain.

---

## 7. Cross-references

- Memory: `project_research_bundle_self_compress_c3_water_bucket_20260429` (water-filling motivation)
- Memory: `project_lane_omega_bit_budget_hessian_aware_quantization` (per-WEIGHT Lane Ω, deferred)
- Memory: `project_codex_theoretical_floor_brutal_20260429` (Shannon floor 0.28 — water-fill is the operational tool to close the gap)
- Memory: `feedback_council_10_member_inner_grand_council_advisory_20260429` (council structure)
- Repository protocol: eval_roundtrip / CUDA-only / 2-approver / no-scorer-at-inflate non-negotiables
- Module: `src/tac/block_fp_codec.py` — encode_conv_weight / pack_payload_tar_xz / verify_roundtrip
- Module: `src/tac/learnable_bit_quant.py` — iter_eligible_conv_names + curvature pattern
- Module: `src/tac/segmap_renderer.py` — SegMap (the model)
- Script template: `scripts/remote_lane_so_hessian_block_fp.sh` (DO NOT touch — Lane SO is retired)

---

## 8. Out-of-scope (NOT in this design; Session 2 must NOT add these)

- ❌ QAT fine-tuning loop (this is POST-COMPRESS export-time only — no retraining)
- ❌ Per-WEIGHT bit allocation (that's Lane Ω-Hessian, separate design)
- ❌ Arithmetic coding on qints (that's Lane SH, separate design)
- ❌ Modal/Vast.ai dispatch (parent shell handles dispatch — lane script is what runs ON the remote)
- ❌ Touching `src/tac/block_fp_codec.py` (the existing `per_channel_qint_max` interface is already correct)
- ❌ Inflate-side changes (block-FP qints + exponents already round-trip via `unpack_payload_tar_xz`)

End of design.
