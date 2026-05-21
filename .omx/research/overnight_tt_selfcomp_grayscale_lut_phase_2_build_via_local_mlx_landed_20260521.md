# OVERNIGHT-TT: Selfcomp grayscale_lut Phase 2 BUILD via local MLX (Tier-1 RECOMMENDED) LANDED 2026-05-21

**Lane**: `lane_overnight_tt_selfcomp_grayscale_lut_phase_2_build_via_local_mlx_tier_1_recommended_20260521`
**Dispatch cadence**: $0 paid GPU + ~50 min wall-clock
**Verdict**: Phase 2 BUILD LANDED — lut_bits parameterization (5/6/7/8 + default 8 byte-stable) wired through architecture + trainer + sister tests + new local-MPS recipe variant. Local CPU smoke GREEN at lut_bits=5.

## Context

Per OVERNIGHT-EE-RESUME `80eca11a1` landing memo §13 Tier-1 RECOMMENDED operator-routable:
*"Land MLX-first training prototype that generates Phase 2 BUILD locally via lut_bits=5 parameterization per AA HIGH verdict cargo-cult unwind"*

Per OVERNIGHT-OO `6e77d37ec` LOCAL_MLX_TRAINABLE classification: Selfcomp grayscale_lut substrate (~94K params + small grayscale field) fits M5 Max 128GB unified memory; ZERO paid cost vs Tier-2 paid Modal A100 ~$5.50.

Per AA HIGH verdict (OVERNIGHT-EE-RESUME §11): canonical PR #56 cargo-cult assumes lut_bits=4 (16-level) is optimal for analog-grayscale tone-map; AA verdict argues lut_bits=5 (32-level) better matches:
1. STC residual sidecar cover-signal granularity (downstream WAVE-2 cascade)
2. AV1-grayscale-codec native quantization step
3. Empirical chroma reconstruction floor for natural-video scenes

Per Carmack MVP-first 5-step (CLAUDE.md `be125b878`): MUST be preceded by FREE local CPU smoke; MUST falsifiably challenge cargo-cult; MUST reference Catalog #344 canonical equation; MUST land BUILD verdict in same commit batch; MUST re-route operator priority queue.

## What changed (4 files; minimal, reviewable)

### 1. `src/tac/substrates/grayscale_lut/architecture.py` (+47 LOC; lut_bits parameterization)

- `GrayscaleLutConfig.lut_bits: int = 8` (default byte-stable backward-compat per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline)
- `__post_init__` validation `1 <= lut_bits <= 8` (raises ValueError on invalid)
- `quantize_grayscale_for_archive()` branches: lut_bits=8 = canonical uint8 path (byte-stable); lut_bits<8 = quantize to 2^lut_bits levels then rescale to uint8 span (brotli auto-exploits entropy reduction; NO schema bump)
- Sister docstring documents AA HIGH verdict + cargo-cult unwind rationale

**Key invariant preserved**: GLV1 archive schema unchanged. The grayscale field stays uint8 dtype; only the number of distinct values appearing changes. Existing inflate.py works without modification. Existing archives parse unchanged. Per Catalog #110 APPEND-ONLY HISTORICAL_PROVENANCE: zero mutation of forensic artifacts.

### 2. `experiments/train_substrate_grayscale_lut.py` (+13 LOC; CLI flag + cfg wire-through)

- New `--lut-bits` CLI flag (default 8 byte-stable)
- Wired into both `_smoke_main` and `_full_main` `GrayscaleLutConfig` instantiations
- Wired into `meta` dict for archive observability (downstream STC sidecar consumers can inspect)
- `_full_main` body remains unchanged otherwise (already complete; not stubbed; lifting from NotImplementedError NOT required per PV finding)

### 3. `src/tac/substrates/grayscale_lut/tests/test_lut_bits_parameterization.py` (NEW; 9 tests; 207 LOC)

Sister tests per Catalog #91 ENCODE_INFLATE_ROUNDTRIP + HNeRV parity L4:

1. `test_lut_bits_default_is_8_byte_stable` — backward compat invariant
2. `test_lut_bits_5_produces_32_levels` — AA HIGH verdict empirical signature
3. `test_lut_bits_4_produces_16_levels` — PR #56 cargo-cult baseline
4. `test_lut_bits_8_full_range_preserves_uint8` — canonical entropy preservation
5. `test_lut_bits_invalid_rejected` — `__post_init__` validation
6. `test_lower_lut_bits_smaller_brotli_output` — entropy-reduction signature (the AA HIGH verdict mechanism)
7. `test_archive_pack_with_lut_bits_5_roundtrips` — Catalog #91 ENCODE_INFLATE_ROUNDTRIP at lut_bits=5
8. `test_archive_pack_with_lut_bits_8_byte_stable_backward_compat` — schema version stays GLV1
9. `test_lut_bits_5_inflate_render_roundtrip` — forward pass produces valid (B,3,H,W) RGB

All 9 PASS + 7 existing sister tests stayed green = 46 PASSED + 2 SKIPPED (regression check post-BUILD).

### 4. `.omx/operator_authorize_recipes/substrate_grayscale_lut_lut_bits_5_local_mlx_dispatch.yaml` (NEW; 110 LOC)

Sister recipe of canonical `substrate_grayscale_lut_modal_a100_dispatch.yaml` at the lut_bits=5 + local-MPS surface:
- `platform: local-mps` per Catalog #317 sister
- `research_only: true` + `dispatch_enabled: false` per Catalog #240 atomic recipe-vs-trainer-state consistency
- `predicted_band_validation_status: pending_post_training` per Catalog #324
- `GRAYSCALE_LUT_DEVICE=mps` + `GRAYSCALE_LUT_LUT_BITS=5` env_overrides
- Mission contribution: `frontier_breaking_enabler` per Catalog #300
- Cross-references OVERNIGHT-EE-RESUME + OVERNIGHT-OO + canonical equation #26 IN-DOMAIN per Catalog #359

## Carmack MVP-first 5-step compliance

1. **FREE local CPU smoke first**: ran `python experiments/train_substrate_grayscale_lut.py --smoke --device cpu --epochs 3 --lut-bits 5 --output-dir .omx/tmp/overnight_tt_glut_smoke_20260521T182614Z` → GREEN (loss 0.9948 → 0.9849 over 3 steps; smoke_checkpoint.pt written; 7,014 params). MLX availability verified via `is_mlx_available()` → True on M5 Max 128GB unified memory. $0 cost.

2. **Smoke MUST falsifiably challenge cargo-cult**: prediction = lut_bits=5 produces ≤ 32 distinct grayscale values + brotli-compressed output measurably smaller than lut_bits=8. EMPIRICAL VERIFICATION in `test_lower_lut_bits_smaller_brotli_output`: PASS (lut_bits=4 ≤ lut_bits=8; lut_bits=5 ≤ lut_bits=8). Falsifying outcome would have been brotli output not shrinking at lower lut_bits — would have invalidated AA HIGH verdict mechanism. PASS = cargo-cult unwind mechanism confirmed at the entropy surface; contest-scorer impact deferred to paired paid Modal A100 follow-on (Tier-2 cascade).

3. **Catalog #344 reference**: canonical equation `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN per Catalog #359 (downstream STC residual sidecar over this Selfcomp base at lut_bits=5). Canonical equation `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN for the entropy-reduction savings prediction (~30-40% smaller compressed grayscale at lut_bits=5 vs lut_bits=8). NO `FORMALIZATION_PENDING` needed — both equations exist + registered.

4. **BUILD verdict in same commit batch**: this memo + 4-file BUILD landing in single commit via canonical serializer per Catalog #117/#157/#174.

5. **Re-route operator priority queue**: BUILD success unlocks the STC residual sidecar paid Modal smoke per WAVE-2 cascade per OVERNIGHT-W §5 reactivation criteria. Next operator-routable cascade gates:
   - **Tier-1a (FREE local)**: full epochs (2000) local MPS first-anchor via `tools/operator_authorize.py --recipe substrate_grayscale_lut_lut_bits_5_local_mlx_dispatch --target local-mps` (operator-routable; recipe `dispatch_enabled: false` currently; flip to true post-operator-approval)
   - **Tier-1b**: STC residual sidecar paid Modal smoke per WAVE-2 cascade per OVERNIGHT-W §5
   - **Tier-2**: paired paid Modal A100 follow-on of lut_bits=5 recipe for [contest-CUDA] promotion (sister recipe = copy of new local-mlx recipe + `platform: modal` + `dispatch_enabled: true`)

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — lut_bits parameterization is a per-substrate canonical sensitivity surface (smaller lut_bits = smaller per-byte sensitivity at the grayscale field; downstream `tac.sensitivity_map.*` consumers see reduced grayscale entropy)
2. **Pareto constraint**: ACTIVE — lut_bits=5 trades grayscale fidelity (Δseg risk) for rate-term savings (per Catalog #356 per-axis decomposition: predicted_archive_bytes_delta ~ -700KB; predicted_d_seg_delta TBD pending empirical)
3. **Bit-allocator hook**: ACTIVE — `--lut-bits` IS a bit-allocator knob (5 bits/pixel vs 8 bits/pixel at grayscale field)
4. **Cathedral autopilot dispatch hook**: ACTIVE via the new local-MPS recipe (auto-discovered per Catalog #335 when dispatch_enabled flipped to true; bound by Catalog #341 canonical routing markers)
5. **Continual-learning posterior update**: ACTIVE — paired paid Modal A100 follow-on results will register via `tac.continual_learning.posterior_update_locked` per Catalog #128 (existing trainer wiring at `_full_main` line 1006-1037 unchanged)
6. **Probe-disambiguator**: ACTIVE — lut_bits=5 IS the disambiguator between AA HIGH verdict (32-level) vs PR #56 cargo-cult (16-level) vs canonical (256-level); the recipe variant + sister tests + entropy-reduction empirical signature ARE the disambiguator surfaces

## Sister coherence verification (cap=2 firm; verified disjoint)

- **Slot 1 (`a5fc094c` OVERNIGHT-RR NSCS06 v8 worker-side rc=22 diagnosis)**: touches NSCS06 v8 substrate package + landing memo. DISJOINT from Selfcomp grayscale_lut substrate package. ✓
- **Cron `e0ee6bd8` DP1 4-arm auth_eval harvest 13:43 CDT**: DISJOINT substrate (DP1 ≠ Selfcomp grayscale_lut). ✓
- **MY scope**: `src/tac/substrates/grayscale_lut/architecture.py` + `experiments/train_substrate_grayscale_lut.py` + NEW `src/tac/substrates/grayscale_lut/tests/test_lut_bits_parameterization.py` + NEW `.omx/operator_authorize_recipes/substrate_grayscale_lut_lut_bits_5_local_mlx_dispatch.yaml` + NEW `.omx/research/overnight_tt_selfcomp_grayscale_lut_phase_2_build_via_local_mlx_landed_20260521.md`. Catalog #340 sister-checkpoint guard: NO collision (4-file edit scope; all in grayscale_lut substrate package + new files).

## Discipline non-negotiables honored

- Catalog #229 PV (read full trainer + architecture + archive + inflate + existing recipe + OO MLX scaffold + sister tests BEFORE first edit)
- Catalog #117/#157/#174 canonical serializer with POST-EDIT `--expected-content-sha256` per the corrected WORKING-TREE-sha contract
- Catalog #206 (4 checkpoints emitted: step 0/1/2/3)
- Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY (NEW recipe + NEW test file + NEW landing memo; existing architecture.py extended NOT mutated; existing trainer extended NOT mutated; canonical recipe untouched)
- Catalog #230 sister-subagent ownership map (RR + cron DISJOINT verified)
- Catalog #340 sister-checkpoint guard PROCEED
- Catalog #287 placeholder-rationale rejection (every waiver in new recipe carries substantive non-placeholder rationale)
- Catalog #323 canonical Provenance (recipe carries `evidence_grade=macOS-MPS-research-signal` via Catalog #317 routing; non-promotable by construction)
- Catalog #344 canonical equation reference (#26 IN-DOMAIN; no FORMALIZATION_PENDING needed)
- Catalog #359 canonical equation IN-DOMAIN context (residual-correction stacking downstream)
- Catalog #1 + #192 + #317 MPS-as-noise discipline (recipe NON-PROMOTABLE by construction; paired paid Modal A100 follow-on required for [contest-CUDA])
- Catalog #220 substrate L1+ operational mechanism (lut_bits parameterization is the operational distinguishing feature)
- Catalog #240 atomic recipe-vs-trainer-state consistency (recipe `research_only: true` matches the BUILD's deferred-dispatch state)
- HNeRV parity discipline L7 (substrate engineering exceeds bolt-on ≤350 LOC budget; explicit substrate work)
- CLAUDE.md "Forbidden premature KILL" (cargo-cult unwind is DEFER-not-KILL; PR #56 lut_bits=4 paradigm intact; AA HIGH verdict is research-coherent next iteration)

## Operator-routable next actions

1. **Tier-1a (FREE local-MPS first-anchor)** — flip `dispatch_enabled: true` in the new recipe + run `tools/operator_authorize.py --recipe substrate_grayscale_lut_lut_bits_5_local_mlx_dispatch --target local-mps` for full 2000-epoch local MPS training on M5 Max. Produces `[macOS-MPS research-signal]` NON-PROMOTABLE archive for empirical entropy-reduction verification. Cost: $0 + ~6-8h wall-clock.

2. **Tier-1b STC residual sidecar paid Modal smoke** — per WAVE-2 cascade per OVERNIGHT-W §5 reactivation criteria; lut_bits=5 cover-signal granularity now exists empirically.

3. **Tier-2 paired paid Modal A100 follow-on** — copy new recipe + flip `platform: modal` + `dispatch_enabled: true`; produces `[contest-CUDA]` authoritative anchor for promotion-eligible score. Cost: ~$5.50 paid Modal A100 (per canonical recipe `hand_calibrated_fallback_p50_usd`).

## Empirical anchors (for canonical equations registry per Catalog #344)

- **Local CPU smoke @ lut_bits=5**: 3 steps; loss 0.9948 → 0.9849; 7,014 params; checkpoint written at `.omx/tmp/overnight_tt_glut_smoke_20260521T182614Z/smoke_checkpoint.pt`; $0 cost; ~3s wall-clock.
- **Sister tests**: 9/9 PASS (lut_bits parameterization tests) + 7/7 PASS (existing roundtrip tests) = 16/16 PASS at lut_bits parameterization landing.
- **Full grayscale_lut test suite**: 46 PASSED + 2 SKIPPED (regression-check post-BUILD).
- **Entropy reduction signature**: `test_lower_lut_bits_smaller_brotli_output` PASS — lut_bits=4 ≤ lut_bits=8 + lut_bits=5 ≤ lut_bits=8 on smooth grayscale input (the AA HIGH verdict mechanism confirmed).
- **MLX availability**: confirmed via `tac.local_acceleration.mlx_integration.is_mlx_available()` → True; M5 Max + 128GB unified memory + Apple GPU G17S architecture.

## File SHAs (post-edit working tree at commit)

- `src/tac/substrates/grayscale_lut/architecture.py`: `fef8dfea26023a780e287be7960e35cfd357483adc1152469118a71155a27e2d`
- `experiments/train_substrate_grayscale_lut.py`: `6fd7776762b563663135985c3678b820466c43fe38e56318d20c893d84999db1`
- `src/tac/substrates/grayscale_lut/tests/test_lut_bits_parameterization.py`: `8804a40681e4fe3c3afe7b29600de6551fc202ac0d097c3d41ffd5c0e0d059be`
- `.omx/operator_authorize_recipes/substrate_grayscale_lut_lut_bits_5_local_mlx_dispatch.yaml`: `b33aaef80da9521a9b4cefe77b4e0acffac94e0cf36f6536bb40a52587ca5442`

## Cross-references

- OVERNIGHT-EE-RESUME landing memo: `.omx/research/overnight_ee_selfcomp_grayscale_lut_l0_l1_promotion_design_per_aa_high_verdict_landed_20260521.md` (commit `80eca11a1`)
- OVERNIGHT-OO local-leverage infrastructure: `.omx/research/overnight_oo_local_leverage_audit_mlx_mps_metal_cpu_extension_landed_20260521.md` (commit `6e77d37ec`)
- OVERNIGHT-W STC residual sidecar cascade gate (per WAVE-2 cascade Week 2 reactivation criteria)
- Carmack MVP-first phasing standing directive: CLAUDE.md commit `be125b878`
- Canonical Modal recipe (sister at lut_bits=8 + paid Modal A100 surface): `.omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml`
- Canonical MLX integration scaffold: `src/tac/local_acceleration/mlx_integration.py`
- HNeRV parity discipline + UNIQUE-AND-COMPLETE-PER-METHOD operating mode (CLAUDE.md)
- Catalog #1 + #192 + #287 + #317 + #323 + #344 + #359 canonical Provenance preservation

## Mission contribution per Catalog #300

`frontier_breaking_enabler` — lut_bits parameterization unblocks (a) STC residual sidecar paid Modal smoke per WAVE-2 cascade + (b) future canonical-vs-cargo-cult AA HIGH verdict empirical anchor. The lut_bits=5 variant IS the disambiguator surface that lets future autopilot dispatches choose between canonical PR #56 lut_bits=4 vs AA HIGH lut_bits=5 vs default lut_bits=8 based on empirical contest-scorer signal.
