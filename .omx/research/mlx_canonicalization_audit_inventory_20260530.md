# MLX canonicalization audit inventory 2026-05-30

**Scope**: enumerate ALL MLX-bearing source files + detect canonical primitive duplication + recommend canonical-extraction migration plan per operator binding directive 2026-05-30 verbatim *"we have a lot of MLX code we want to ensure it is canonicalized and no duplicate code and compounding optimization and learning and coherent codebase, remember our tinygrad primitives work that is underway perhaps include that in the memo as well"* + *"all must be wired and integrated and tested and individually fractally optimized for extreme synergy and positive externalities"*.

**Per CLAUDE.md non-negotiables honored**: NO FAKE IMPLEMENTATIONS (substantive enumeration via grep + AST inspection; no fabricated counts) + UNIQUE-AND-COMPLETE-PER-METHOD (Catalog #290 falling-rule list applied per primitive) + Forbidden premature KILL (classifications are operator-routable remediation NOT KILL verdicts) + Apples-to-apples evidence discipline (per-primitive empirical duplication counts).

**Lane**: `lane_mlx_canonicalization_audit_plus_tinygrad_bridge_plus_6_pillar_discipline_20260530` L1 (impl_complete + canonical_helper + memory_entry).

## PHASE A — MLX canonicalization audit inventory

### A.1 Enumeration

Total MLX-bearing `.py` files (excluding `__pycache__` + `experiments/results/`):

- **155 total** (155 grep-matched `import mlx` / `import mlx.core`)
- **67 non-test** modules (production canonical surface)
- **17 substrate `mlx_renderer.py`** modules (substrate-specific MLX renderers)
- **3 canonical layers** identified:
  1. `tac.local_acceleration.pr95_hnerv_mlx` (3012 LOC; 83 functions; THE canonical PR95 HNeRV MLX core)
  2. `tac.framework_agnostic` (1331 LOC; backend dispatch + bridges + decorators; sister to `tac.local_acceleration` at the framework-selection surface)
  3. `tac.local_acceleration.tinygrad_bridge` (345 LOC; tinygrad → numpy inflate primitive bridge per 8th standing directive)

### A.2 Per-primitive canonical implementations

#### A.2.1 `pixel_shuffle_2x_nhwc` — CANONICAL @ `tac.local_acceleration.pr95_hnerv_mlx:937`
- Sister implementations: `tac.substrates._shared.numpy_portable_inflate:584` (numpy-reference)
- Substrate adopters (via `from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc`): atw_v2 / nirvana / pact_nerv_ia3 / pact_nerv_selector_v2 / pact_nerv_selector_v3 / pact_nerv_selector_v4 / pact_nerv_vq / time_traveler_l5_z5 / time_traveler_l5_z6 / z6_v2_cargo_cult_unwind / z8_hierarchical_predictive_coding (11 of 17 substrate renderers route through canonical = 65%)
- **DUPLICATE_IMPL_COUNT = 2** (canonical MLX + numpy reference); sister substrates correctly route via import
- Verdict: CANONICAL_ADOPTED_AT_65_PERCENT (remaining 35% need migration audit)

#### A.2.2 `bilinear_resize_nhwc` / `bilinear_resize2x_align_corners_false_nhwc` — CANONICAL @ `tac.local_acceleration.pr95_hnerv_mlx:1085` + `:1008`
- Sister implementations: `tac.substrates._shared.numpy_portable_inflate:514` (numpy-reference)
- Substrate adopters: same pattern as `pixel_shuffle_2x_nhwc` (sister import discipline)
- **DUPLICATE_IMPL_COUNT = 2** (canonical MLX + numpy reference)
- Verdict: CANONICAL_ADOPTED (mature)

#### A.2.3 `Conv2dMLX` / `_PR95Conv2dMLX` — CANONICAL @ `tac.local_acceleration.pr95_hnerv_mlx:1194`
- Sister implementations: `tac.local_acceleration.pr95_hnerv_numpy_reference:86` (numpy-reference `conv2d_nhwc`); `tac.substrates._shared.numpy_portable_inflate:407` (`conv2d_nhwc_oihw`)
- **DUPLICATE_IMPL_COUNT = 3** (canonical MLX + 2 sister numpy reference forks); the 2 numpy forks serve DIFFERENT contracts (different axis conventions) per UNIQUE-AND-COMPLETE-PER-METHOD per Catalog #290 (`numpy_portable_inflate` uses `_oihw` suffix to disambiguate; this is HARD-EARNED CANONICAL FORK)
- Verdict: CANONICAL_ADOPTED_WITH_PRINCIPLED_FORK

#### A.2.4 `KahanCompensatedPolyakEMAShadow` — CANONICAL @ `tac.training.long_training_canonical:1381`
- Sister implementations: `PolyakEMAShadow` parent class at same module
- **DUPLICATE_IMPL_COUNT = 1** (single canonical class with parent fallback per Slot 16 LANDED commit `65db9f570`)
- Verdict: CANONICAL_ADOPTED_AT_100_PERCENT

#### A.2.5 `gumbel_softmax_sample` — **DUPLICATION DETECTED**
- Sister implementations:
  - `tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer:207` (`gumbel_softmax_sample`)
  - `tac.substrates.dreamer_v3_rssm.module:199` (`gumbel_softmax_sample`)
  - `tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer:274` (`gumbel_softmax_sample_mlx`)
- **DUPLICATE_IMPL_COUNT = 3** (no canonical extracted yet)
- Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD per Catalog #290 falling-rule:
  - EMPIRICAL: no paired-comparison smoke between substrates → UNCLEAR
  - PRINCIPLED: each substrate has same canonical Gumbel-softmax mathematical form (per Hafner 2023 + sister DreamerV3 math fidelity audit at commit landed 2026-05-29) → OBVIOUS-FIT
- **Recommended canonical extraction**: lift to `tac.framework_agnostic.gumbel_softmax_sample(logits, *, temperature, unimix_alpha=0.01, backend=AUTO)` per Wave 3 DreamerV3 unimix landing 2026-05-29; threaded through Z8 / DreamerV3 / mdl_ibps_j Hafner-2023-canonical entry point
- Verdict: CANONICAL_EXTRACTION_RECOMMENDED (operator-routable migration; 3 substrate-side imports)

#### A.2.6 `rgb_to_yuv6` / `yuv6_to_rgb` — **DUPLICATION DETECTED**
- Sister implementations:
  - `tac.constrained_gen:97/144` (PyTorch primary)
  - `tac.composition.yuv6_chroma_subsampled_perturbation_operator.operator:198/243` (numpy reference)
  - `tac.local_acceleration.pr95_hnerv_mlx_training:106` (`rgb_to_yuv6_mlx`)
  - `tac.saliency:52` (PyTorch sister)
- **DUPLICATE_IMPL_COUNT = 4** (3 framework variants + 1 sister PyTorch with subtly different normalization)
- Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD per Catalog #290 falling-rule:
  - EMPIRICAL: `tac.saliency.rgb_to_yuv6` is the canonical contest-faithful version per CLAUDE.md "eval_roundtrip" non-negotiable; sister `tac.constrained_gen.rgb_to_yuv6` is the sister training-time differentiable variant per `tac.differentiable_eval_roundtrip` non-negotiable; FORK_BECAUSE_PRINCIPLED_MISMATCH
- **Recommended canonical extraction**: lift to `tac.framework_agnostic.rgb_to_yuv6(rgb_chw, *, backend=AUTO)` + sister `yuv6_to_rgb` per the canonical 4-backend pattern (NUMPY / PYTORCH / MLX / TINYGRAD); substrate trainers consume via backend dispatch
- Verdict: CANONICAL_EXTRACTION_RECOMMENDED (operator-routable; 4 implementations exist)

#### A.2.7 `softmax_with_epsilon` / `fp32_matmul_correction` — Per Slot 16 LANDED
- Sister implementations: per Slot 16 commit `65db9f570` (z6 MLX drift architecture-class dependent fix landed 2026-05-26)
- Per the canonical equation `mlx_pytorch_conv2d_fp64_accumulation_drift_reduction_v1` + sister equations
- **DUPLICATE_IMPL_COUNT = 1** (centralized in `pr95_hnerv_mlx` per Slot 16)
- Verdict: CANONICAL_ADOPTED_AT_100_PERCENT

#### A.2.8 `mlx_to_pytorch_state_dict_bridge` / `mlx_to_numpy_bridge` — CANONICAL @ `tac.framework_agnostic.helpers`
- Sister implementations:
  - `tac.framework_agnostic.helpers:mlx_state_dict_to_npz_bridge` (canonical npz bridge per 8th standing directive)
  - `tac.local_acceleration.pr95_hnerv_mlx:load_pytorch_state_dict_into_mlx` + sister `pytorch_state_dict_from_mlx`
  - Per-substrate export tools (`tools/export_pact_nerv_*_mlx_to_pytorch_state_dict.py` — 7 export tools)
- **DUPLICATE_IMPL_COUNT = 9** (1 canonical + 1 reverse + 7 per-substrate export tools)
- Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: 7 per-substrate export tools are CARGO-CULTED variations of the same canonical pattern; should consume `tac.framework_agnostic.helpers.mlx_state_dict_to_npz_bridge` + per-substrate metadata mapping
- **Recommended canonical extraction**: per-substrate export tools become thin wrappers (~30 LOC each) over the canonical bridge; current ~250 LOC × 7 = 1750 LOC → ~210 LOC total (~88% LOC reduction)
- Verdict: CANONICAL_EXTRACTION_RECOMMENDED (high-EV refactor; 7 sister tools)

### A.3 Cross-substrate canonical adoption summary

**Canonical adoption rate per primitive** (sister substrates importing canonical helper / total substrate MLX renderers = 17):

| Primitive | Adopters | Total Substrates | Adoption Rate |
|---|---|---|---|
| `pixel_shuffle_2x_nhwc` (canonical pr95_hnerv_mlx) | 11 | 17 | 65% |
| `bilinear_resize_nhwc` (canonical pr95_hnerv_mlx) | 11 | 17 | 65% |
| `pr95_mlx_*_conv2d_accumulation_*` (Slot 16 fixes) | 11 | 17 | 65% |
| `mlx_state_dict_to_npz_bridge` (canonical framework_agnostic) | 0 | 7 export tools | 0% (high-EV) |
| `gumbel_softmax_sample` (NO canonical yet) | 0 | 3 substrates | 0% (extraction needed) |
| `rgb_to_yuv6` (NO canonical yet) | 0 | 4 sister | 0% (extraction needed) |

**Operator-routable migration plan**:
1. **HIGH-EV** (1-2 hours; ~88% LOC reduction): refactor 7 per-substrate MLX→PyTorch export tools into thin wrappers over `tac.framework_agnostic.helpers.mlx_state_dict_to_npz_bridge`
2. **MEDIUM-EV** (1 hour; eliminates structural duplication of canonical Hafner 2023 + DreamerV3 unimix discipline): lift `gumbel_softmax_sample` to `tac.framework_agnostic.gumbel_softmax_sample`; migrate 3 substrate sister impls (DreamerV3 / Z8 / mdl_ibps_j)
3. **MEDIUM-EV** (1-2 hours): lift `rgb_to_yuv6` + `yuv6_to_rgb` to `tac.framework_agnostic` with 4-backend dispatch; migrate 4 sister impls preserving PRINCIPLED FORKS where contracts differ
4. **LOW-EV** (operator-routable; 6 substrates not yet routing): paired audit of 6 substrate MLX renderers that DON'T import `pr95_hnerv_mlx` canonical primitives; verify each has either substrate-optimal FORK (Catalog #290) or migration target

### A.4 Tinygrad primitives bridge status

**Current state** (per `tac.local_acceleration.tinygrad_bridge` 345 LOC):
- `is_tinygrad_available()` ✅ availability detection
- `TinygradBridgeManifest` ✅ canonical manifest dataclass
- `tinygrad_state_dict_to_zip_member_bytes()` ✅ canonical ZIP-member bridge
- `load_tinygrad_trained_weights_for_numpy_inflate()` ✅ inflate-side loader
- `build_tinygrad_bridge_manifest()` ✅ manifest builder
- `tinygrad_with_numpy_inflate_bridge()` ✅ decorator surface

**Tests**: 519 LOC at `src/tac/tests/test_tinygrad_portable_inflate_primitive_bridge.py` (10-section coverage per design memo)

**Missing canonical primitives for production-grade tinygrad deployment**:
- Tinygrad sister implementations of: `conv2d_nhwc` / `bilinear_resize_nhwc` / `pixel_shuffle_2x_nhwc` / `batchnorm2d` / `gumbel_softmax` / `rgb_to_yuv6`
- Cross-backend parity tests asserting MLX ↔ tinygrad ↔ PyTorch ↔ numpy byte-stable at fp32 (within Slot 16 numerical tolerance)
- Tinygrad inflate.py emission ≤200 LOC per HNeRV parity L4

**This subagent (Phase C) lands the tinygrad sister primitives + cross-backend parity test fixture + canonical equation registration per Catalog #344.**

---

## Sister landings TODAY (Wave 8 LANDED + 2 IN-FLIGHT)

Per parent prompt context:
- alaska canonical inverse-steganalysis patterns (commit `61a91a48e`)
- m9-v3 (commit `49f41e22c`)
- Yousfi-Tier-1 (commit `3d027ecf9`)
- Z7+Z8 mamba2_adapter (LANDED)
- MLX-LOCAL smoke (commit `98412f194`)
- deferred-items feeder (commit `46aa6ad86`)
- Fridrich-school extension (commit `396488202`)
- Z8 M12a pre-flight (commit `ef7fd29e3`)
- z6_v2 pre-flight (commit `7a8581424`)
- IN-FLIGHT: PR110-OPT-7 L1 promotion (`acd4123aaaba505a9`) + z6_v2 29,650ep MLX-LOCAL FULL RUN (`ae1c4683e73e39b7a`)

This subagent IS the canonical META-class wave at the MLX canonicalization + tinygrad bridge sub-surface; sister-DISJOINT vs in-flight subagents per Catalog #340 sister-checkpoint-guard.

---

## Apparatus mutation chain queued

1. NEW canonical helper `src/tac/local_acceleration/mlx_canonical_audit/` (~400-500 LOC) wraps the audit logic as machine-callable APIs
2. NEW operator-facing tool `tools/audit_mlx_canonicalization.py` (~200 LOC) emits canonical JSON consumable by cathedral autopilot consumer auto-discovery per Catalog #335
3. NEW STRICT preflight gate `check_mlx_primitives_route_through_canonical_helper` (Catalog #383) refuses NEW substrate MLX renderers that re-implement canonical primitives WITHOUT canonical-helper routing (WARN-ONLY at landing; live count likely 6 per A.3 high-EV migration plan)
4. NEW canonical equation `mlx_primitive_canonicalization_compounding_savings_v1` per Catalog #344 (registry 158 → 159)
5. NEW canonical equation `mlx_pytorch_tinygrad_cross_backend_byte_stable_v1` per Catalog #344 (registry 159 → 160)
6. NEW cathedral consumer `tac.cathedral_consumers.mlx_canonicalization_audit_consumer` per Catalog #335 auto-discovery
7. NEW tinygrad sister primitives `tac.framework_agnostic.canonical_kernels` + cross-backend parity tests
8. Probe outcome via Catalog #313 PROCEED 14-day advisory
9. Catalog #348 retroactive sweep memo `.omx/research/retroactive_sweep_for_catalog_383_20260530T*.md`
10. Landing memo `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_canonicalization_audit_plus_tinygrad_bridge_plus_6_pillar_discipline_landed_20260530.md`

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": classifications above are operator-routable migration paths NOT KILL verdicts. Each duplicate-implementation finding has reactivation criteria pinned (consume canonical helper OR document substrate-optimal FORK per Catalog #290).
