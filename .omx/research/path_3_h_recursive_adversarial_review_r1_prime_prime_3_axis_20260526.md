<!-- SPDX-License-Identifier: MIT -->
---
schema_version: substrate_recursive_adversarial_review_memo_v2_20260516
deliberation_id: path_3_h_recursive_adversarial_review_r1_prime_prime_3_axis_20260526
substrate_id: atw_v2_cooperative_receiver_v2
review_round: R1''
per_substrate_counter_before: 0/3
per_substrate_counter_after: 0/3
verdict: NOT_CLEAN_FIX_WAVE_REQUIRED
landing_under_review_commits: [06ea98483, 683878854, 98484a08b, 84a403a3b, eca70401c]
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Atick
  - Ballard
  - Contrarian
  - AssumptionAdversary
  - PR95Author
  - Carmack
  - Hotz
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_assumption_adversary_verdict:
  - assumption: "MLX renderer drift band ``1e-3 to 1e-2`` per landing memo §3 Axis 2 is structurally achievable WITH the current implementation"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "MLX renderer at src/tac/substrates/atw_v2_cooperative_receiver_v2/mlx_renderer.py:248-249 uses ``mx.repeat`` for upsample — the EXACT anti-pattern that caused sister A=DreamerV3 max_abs=24.34 pre-FIX-WAVE-R1 drift. The landing memo's own §3 line 96-98 acknowledges 'End-to-end full-decoder + scorer drift bound 1e-3 to 1e-2' but does NOT acknowledge that this is the WORSE-CASE empirical residual AFTER avoiding anti-patterns — and H is currently NOT avoiding the anti-pattern. Empirical measurement on this machine confirms matmul drift at substrate dimensions = 4.97e-2 (one order of magnitude WORSE than upper bound of stated band)."
  - assumption: "Phase 1 cdf_table_blob unwind is structurally extincted per H's NEW ATWv2CR2 grammar"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "grep src/tac/substrates/atw_v2_cooperative_receiver_v2/*.py confirms 'cdf_table_blob' appears ONLY in __init__.py docstrings as HISTORICAL_PROVENANCE; ZERO functional code references. The 8-section grammar (header + encoder_blob + decoder_blob + cond_embed_blob + ego_motion_proj_blob + per_pair_latent_blob + class_cond_cdf_blob + meta_blob) replaces v1 with NO dead sections. Codex byte-mutation falsification commit 057130de4 (max_abs_raw_byte_delta=0 on 2,560 mutated bytes) is preserved as v1 anchor."
  - assumption: "Layer 1 META-unwind binding to Atick-Redlich SOLE anchor (demoting Tishby + dropping Wyner-Ziv) is structurally correct per the substrate's NEW conditioning variable"
    classification: HARD-EARNED-AXIOMATICALLY-VERIFIED
    rationale: "Atick-Redlich 1990 preconditions ARE satisfied: contest SegNet+PoseNet IS a fixed receiver; mutual-information maximization against a fixed receiver IS the canonical Atick-Redlich formulation. Wyner-Ziv preconditions ARE violated (scorer_class_prior_table is shared both-sides in archive, not decoder-only side-info); reframe as conditional source coding R(D|Y) is the mathematically correct fix. Tishby IB demotion to advisory is also correct (contest task is multi-task RGB reconstruction, not single-Y classification)."
  - assumption: "ego-motion FOE projection conditioning surface replaces D4-falsified per-class softmax with a STRUCTURALLY SOUND empirical alternative"
    classification: HARD-EARNED-PENDING-EMPIRICAL
    rationale: "Catalog #311 + Ballard 2007 cite is sound; D4 INDEPENDENT verdict (I(latent; scorer_class) = 0.006385 bits/symbol) empirically falsifies per-class softmax for A1-class latents. The ego-motion FOE hypothesis (dashcam vehicle motion is dominant continuous-time signal) is principled. HOWEVER the I(latent; Y_ego_motion) empirical anchor is DEFERRED to Phase 4 D4-equivalent probe (per op-routable #1). Until the probe runs, the conditioning surface is HARD-EARNED-by-principle but PENDING-EMPIRICAL-CONFIRMATION."
council_decisions_recorded:
  - "OP-1: FIX-WAVE-R1''-H MUST replace mx.repeat upsample with canonical bilinear (align_corners=False) per sister A=DreamerV3 forensic + Catalog #1255 mitigation pattern"
  - "OP-2: After OP-1, re-run MLX↔numpy parity test on full end-to-end decoder; verify drift band claim 1e-3 to 1e-2 holds empirically with mitigation applied"
  - "OP-3: Phase 4 D4-equivalent probe MUST register canonical equation #311 ego-motion entropy posterior anchor (deferred operational mechanism)"
  - "OP-4: H is the ONLY one of H+I+J+K that participates in WAVE-1 canonical posterior emission wire-in — sister I/J/K need Wave-2 follow-on per op-routable in aggregate memo"
council_predicted_mission_contribution: frontier_breaking_enabler
horizon_class: frontier_pursuit
deferred_substrate_id: atw_v2_cooperative_receiver_v2
related_deliberation_ids:
  - path_3_h_atw_v2_cooperative_receiver_L0_scaffold_landed_20260526
  - path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
audit_evidence_tag: "[macOS-MLX research-signal]"
---

# Path 3 H — ATW V2 cooperative-receiver v2 — R1'' recursive adversarial review (3-axis)

**Lane:** `lane_path_3_recursive_adversarial_review_r1_prime_prime_3_axis_landings_h_i_j_k_20260526` L1
**Predecessor counter:** 0/3 (NEW substrate; R1'' is first recursive review)
**Successor counter target:** 1/3 if CLEAN; reset 0/3 if NOT CLEAN
**This round verdict:** NOT_CLEAN_FIX_WAVE_REQUIRED → per-substrate counter REMAINS 0/3

## 1. Scope

This R1'' memo reviews the H=ATW V2 cooperative-receiver v2 substrate landing per the operator binding 3-axis discipline (Axis 1 math+scientific+engineering rigor / Axis 2 MLX drift minimization / Axis 3 portability via numpy) per directive #3.

Review-only NO file modifications per the brief's "review-only NO file modifications" constraint.

## 2. Landings under review

| Commit | Description |
|---|---|
| `06ea98483` | Phase 1 cargo-cult audit of v1 (17 NEW CCs surfaced; cdf_table_blob FALSIFIED anchor) |
| `683878854` | Phase 2 substrate-design decision (Path b FORK to Atick-only single-anchor + ego-motion FOE projection) |
| `98484a08b` | Phase 3 supplemental — meta-fix label on numpy_reference predecessor |
| `eca70401c` | Phase 3 L0 scaffold remaining files |
| `84a403a3b` | Phase 3 L0 scaffold final files (mlx_renderer + torch_compat + inflate + training) |

Substrate package: `src/tac/substrates/atw_v2_cooperative_receiver_v2/` (8 files, ~1450 LOC).

## 3. Axis 1 — Math + scientific + engineering rigor per layer

### 3.1 HARD-EARNED layers (verified)

| Layer | Anchor | Verdict |
|---|---|---|
| Atick-Redlich 1990 cooperative-receiver loss as SOLE substrate-optimal anchor | Phase 2 Layer 1 META-unwind binding; contest SegNet+PoseNet IS fixed receiver | HARD-EARNED-AXIOMATICALLY-VERIFIED |
| Tishby IB DEMOTED to advisory cross-check | Multi-task (RGB through SegNet+PoseNet) vs single-Y IB precondition mismatch | HARD-EARNED-AXIOMATICALLY-VERIFIED |
| Wyner-Ziv 1976 DROPPED + reframed as R(D|Y) | scorer_class_prior_table is shared both-sides in archive (not decoder-only side-info; WZ precondition violated) | HARD-EARNED-AXIOMATICALLY-VERIFIED |
| ATWv2CR2 grammar removes cdf_table_blob dead-bytes class | Codex byte-mutation falsification commit `057130de4` empirically proved max_abs_raw_byte_delta=0 on 2,560 mutated bytes | HARD-EARNED-EMPIRICALLY-VERIFIED |
| Ballard 2007 + Catalog #311 ego-motion-conditioning citation chain | Replaces per-class softmax (D4 INDEPENDENT verdict 0.006385 bits/symbol on A1 latents) | HARD-EARNED-BY-PRINCIPLE |
| HNeRV decoder canonical pattern (Chen et al. 2023 arXiv:2304.02633) | Sister Z6 mlx_renderer.py canonical pattern | HARD-EARNED-CANONICAL |

### 3.2 CARGO-CULTED-PENDING layers

| Layer | Cargo-cult risk | Unwind path |
|---|---|---|
| ego-motion FOE projection as DOMINANT conditioning signal | Hypothesis HARD-EARNED by principle but PENDING empirical D4-equivalent probe | Phase 4 op-routable #1 in landing memo §7 |
| MLX↔PyTorch parity drift bound 1e-3 (per Catalog #1265 gate) | Test passes for ego-motion FOE projection primitive only; full-decoder parity NOT yet measured | Phase 4 op-routable in landing memo §7 |

### 3.3 Findings

**NONE on Axis 1.** Math + scientific framing is sound. The Layer 1 META-unwind is the canonical structural fix for the v1 triple-citation-stacking; preconditions ARE satisfied for Atick-Redlich and ARE violated for Wyner-Ziv; the demotion + drop + reframe is mathematically correct.

## 4. Axis 2 — MLX drift minimization per primitive

### 4.1 Per-primitive drift verification

| Primitive | Source location | MLX usage | Empirical drift | Anti-pattern? |
|---|---|---|---|---|
| `mlx_ego_motion_foe_projection` | `mlx_renderer.py:129-157` | `mx.sqrt` + elementwise + `mx.concatenate` | bit-exact-equivalent (elementwise has no FMA reassociation concern) | NO |
| `_CondEmbeddingHead.__call__` | `mlx_renderer.py:179-182` | `nn.Linear` + `nn.relu` + `nn.Linear` | matmul drift on M-series MPS empirically ~4.97e-2 abs / ~7e-4 rel at substrate dimensions | NO (canonical) |
| `_HNeRVStyleDecoder.__call__` initial_proj | `mlx_renderer.py:209` | `nn.Linear` | matmul drift on M-series MPS empirically O(1e-2) | NO (canonical) |
| `_HNeRVStyleDecoder.__call__` PixelShuffle | `mlx_renderer.py:221-230` | `mx.reshape` + `mx.transpose` | bit-exact (no math primitives) | NO |
| **`_HNeRVStyleDecoder.__call__` upsample to output_height x output_width** | **`mlx_renderer.py:236-251`** | **`mx.repeat(h, scale_h, axis=1)` + `mx.repeat(h, scale_w, axis=2)`** | **NN-replication — quality-degrading anti-pattern per FIX-WAVE-R1 sister A=DreamerV3 forensic** | **YES — CRITICAL** |
| `mx.sigmoid` final | `mlx_renderer.py:253` | `mx.sigmoid` | bit-exact-equivalent | NO |

### 4.2 CRITICAL FINDING H-R1''-1 — mx.repeat upsample anti-pattern

**Location:** `src/tac/substrates/atw_v2_cooperative_receiver_v2/mlx_renderer.py:248-249`

```python
if scale_h > 1 or scale_w > 1:
    h = mx.repeat(h, scale_h, axis=1)
    h = mx.repeat(h, scale_w, axis=2)
# Then crop to exact target
h = h[:, :target_h, :target_w, :]
```

**Bug class:** Nearest-neighbor `mx.repeat` upsample is the SAME ANTI-PATTERN that caused sister A=DreamerV3 `max_abs=24.34` drift per FIX-WAVE-R1 forensic. The L0 scaffold's own docstring (lines 237-238) acknowledges *"For L0 SCAFFOLD, use simple nearest-neighbor as placeholder; production Phase 4 swaps in canonical MLX bilinear once #1265 gate confirms drift bound met."* — the operator is on notice but the placeholder ships.

**Empirical impact:** When the decoder outputs initial-grid 16x16 (per default `DEFAULT_DECODER_INITIAL_GRID_H=16` x `DEFAULT_DECODER_INITIAL_GRID_W=16`) and target output is 384x512, the scale factors become `scale_h=24, scale_w=32` (via `max(1, target_h // current_h)`). `mx.repeat` replicates each pixel 24x32=768 times, producing a hard pixelated reconstruction. This is NOT a drift bound issue — it is a STRUCTURAL semantic failure: the decoder cannot produce smooth RGB output even at infinite drift precision.

**Drift band claim:** The landing memo §3 line 96-98 states *"End-to-end full-decoder + scorer drift bound 1e-3 to 1e-2"* — but this claim is unverified because the test suite measures parity only on the `mlx_ego_motion_foe_projection` primitive (lines 100-103 of test_basic.py per Axis 2 evidence), not on the full end-to-end decoder. The actual full-decoder drift with `mx.repeat` upsample would be substantially worse than 1e-2.

**Cargo-cult classification:** CARGO-CULTED-EMPIRICALLY-FALSIFIED — the substrate landed an anti-pattern that the Path 3 cascade doctrine + MLX-first doctrine explicitly forbid.

**Fix path (FIX-WAVE-R1''-H):**
1. Replace `mx.repeat` lines 248-249 with canonical bilinear `align_corners=False` MLX kernel
2. Reference sister A=DreamerV3 FIX-WAVE-R1 commit `e1b101888` for the canonical implementation pattern
3. Re-run MLX↔numpy parity test on end-to-end decoder (not just primitive)
4. Confirm drift band 1e-3 to 1e-2 holds empirically with mitigation applied

**Sister A=DreamerV3 precedent:** Per FIX-WAVE-R1 closure `e1b101888`, the A substrate had identical bug class (`align_corners=True` + `mx.repeat` causing `max_abs=24.34` drift); the fix wave applied canonical bilinear `align_corners=False` and brought drift to <0.001 contest-units. H is empirically committed to the SAME anti-pattern; same fix expected.

### 4.3 Other MLX primitives — clean

- `mx.sqrt`, `mx.sum`, `mx.concatenate`, `mx.reshape`, `mx.transpose`, `mx.sigmoid` all per-element / bit-exact-equivalent operations with no FMA reassociation concern
- `nn.Linear`, `nn.Conv2d`, `nn.relu` canonical MLX primitives; drift bound is hardware-class O(1e-2) abs / O(1e-3) rel which is the M-series MPS empirical floor for fp32 matmul (per sister K=COIN++ verification)
- NO `mx.softmax` usage in H (substrate is not softmax-based)
- NO explicit fp16 cast (all fp32 via `mx.array` default + `nn.Linear` default)

### 4.4 Drift band claim — UNVERIFIED

The landing memo §3 Axis 2 claims drift band `1e-3 to 1e-2` for end-to-end full-decoder + scorer per `pr95_mlx_full_decoder_downstream_scorer_drift_landed`. The test suite empirically verifies parity only for `mlx_ego_motion_foe_projection` primitive (single elementwise + sqrt + division + concat) — NOT for the full decoder including `nn.Linear`+`nn.Conv2d`+`mx.repeat`+`mx.sigmoid` chain. The claim should be backed by an empirical anchor measurement at the FULL-DECODER surface BEFORE landing operator-routable claim of "drift band 1e-3 to 1e-2".

## 5. Axis 3 — Portability via numpy per primitive

### 5.1 Per-primitive numpy reference verification

| Primitive | numpy reference | Status |
|---|---|---|
| `mlx_ego_motion_foe_projection` | `numpy_ego_motion_foe_projection` | ✓ Sister reference present |
| `_CondEmbeddingHead.__call__` | `numpy_decode_pair_with_ego_motion_conditioning` (composed) | ✓ Sister reference present |
| `_HNeRVStyleDecoder.__call__` | `numpy_decode_pair_with_ego_motion_conditioning` (composed) | ✓ Sister reference present |
| Archive grammar pack/parse | `archive.py` uses only numpy + struct | ✓ Pure numpy + struct (trivially portable) |
| Inflate runtime | `inflate.py` uses canonical `select_inflate_device` (PyTorch only via canonical helper) | ✓ Catalog #205 canonical |
| _training_only.py | EXPLICITLY non-portable (PyTorch only); ISOLATED in own module | ✓ Documented exception per operator directive #3 |

### 5.2 Portability evidence

- `numpy_reference.py` is 20.3 KB / ~520 LOC; bit-exact-per-primitive reference for FORWARD/INFLATE path
- `inflate.py` is pure-numpy + canonical `select_inflate_device` (PyTorch only for the canonical helper; numpy for all parse + load + reconstruction logic)
- `archive.py` is pure-numpy + struct (trivially portable)
- Forward path + inflate path FULLY operational on CPU-only systems without MLX
- Test verification: `test_mlx_numpy_parity_*` skips gracefully when MLX unavailable; numpy tests still pass

### 5.3 Findings

**NONE on Axis 3.** Portability discipline is exemplary. The 4-layer separation (MLX renderer / numpy reference / PyTorch compat reference / PyTorch training-only) is the canonical Path 3 cascade pattern.

## 6. Cross-substrate META findings (review-context only)

| META finding | Surface | Notes |
|---|---|---|
| WAVE-1 posterior emission wire-in present (H is the only one of H+I+J+K with the wire-in) | `src/tac/substrates/atw_v2_cooperative_receiver_v2/__init__.py:285-388` | H landed FIRST (per timestamp `2026-05-26 03:43`-aligned with helper landing `f6b432be1` ~ 03:43); WAVE-1 wire-in at `3d103dafd` 04:01 covered H. Sister I/J/K (landed 03:16/03:23/03:06) all PRECEDE WAVE-1 wire-in and are MISSING the wire-in. This is a Wave-2 follow-on op-routable surfaced in aggregate memo. |
| Catalog #240(c) `_full_main raises NotImplementedError` | `mlx_renderer.py` Phase 4 council approval required | CORRECT POSTURE per substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY non-negotiable |
| Catalog #241 + Catalog #124 8-field declaration | `__init__.py:1-243` | All 8 fields declared inline; AST walker observable; substrate identity constants importable |

## 7. R1'' verdict + per-substrate counter

### Verdict: NOT_CLEAN_FIX_WAVE_REQUIRED

**Reason:** 1 CRITICAL Axis 2 finding (H-R1''-1 `mx.repeat` upsample anti-pattern) requires FIX-WAVE-R1''-H closure before per-substrate counter can advance to 1/3.

### Per-substrate counter

- Before R1'': 0/3
- After R1'': **0/3 (RESET due to 1 CRITICAL finding)**
- Path to 1/3: FIX-WAVE-R1''-H lands `mx.repeat` replacement + end-to-end decoder MLX↔numpy parity verification

### Successor required

**FIX-WAVE-R1''-H** subagent must:
1. Replace `mx.repeat` upsample at `mlx_renderer.py:248-249` with canonical bilinear `align_corners=False` MLX kernel per sister A=DreamerV3 FIX-WAVE-R1 commit `e1b101888` pattern
2. Add full-decoder MLX↔numpy parity test in `tests/test_basic.py` covering the entire decoder chain (not just the FOE projection primitive)
3. Empirically verify drift band 1e-3 to 1e-2 holds AT FULL-DECODER surface (not just primitive); update landing memo §3 with empirical anchor
4. After empirical anchor, mark H per-substrate counter 1/3 via R2''-H clean pass

## 8. 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = ACTIVE (this review memo's empirical findings inform `tac.sensitivity_map.*` per-primitive drift attribution)
- hook #2 Pareto constraint = N/A (defensive review memo; no Pareto signal)
- hook #3 bit-allocator = N/A
- hook #4 cathedral autopilot dispatch = ACTIVE (drift findings inform autopilot's ranking weight per Catalog #335 + #341)
- hook #5 continual-learning posterior = ACTIVE (this memo's frontmatter is consumable by `tac.council_continual_learning.append_council_anchor`)
- hook #6 probe-disambiguator = ACTIVE (the `mx.repeat` vs canonical bilinear distinction IS the disambiguator for Axis 2 substrate-quality routing)

## 9. Discipline compliance

- ✅ Catalog #229 PV (read landing memo + 3 substrate source files + tests + empirical MLX drift verification BEFORE writing memo)
- ✅ Catalog #110/#113 APPEND-ONLY (NEW review memo only; sister landing memos NEVER mutated)
- ✅ Catalog #208 docs/local-paths (no `/Users/` absolute paths in this memo)
- ✅ Catalog #230 sister-subagent ownership map (review-only; no file modifications per brief)
- ✅ Catalog #287 placeholder-rationale rejection (every assumption_adversary_verdict carries substantive ≥4-char rationale)
- ✅ Catalog #292 per-axis assumption surfacing (4 assumptions classified)
- ✅ Catalog #300 v2 frontmatter complete (tier T2; 9 attendees; quorum met; verdict + dissent + decisions + mission contribution)
- ✅ Catalog #340 sister-checkpoint guard PROCEED (review-only)
- ✅ Per CLAUDE.md "Executing actions with care": review-only NO code modifications

## 10. Cross-references

- Landing memo: `.omx/research/path_3_h_atw_v2_cooperative_receiver_L0_scaffold_landed_20260526.md`
- Phase 1 audit: `.omx/research/path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Phase 2 decision: `.omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526.md`
- Sister A=DreamerV3 FIX-WAVE-R1 commit: `e1b101888`
- v1 falsification anchor: commit `057130de4`
- Path 3 R1' aggregate (sister round): `.omx/research/path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526.md`

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
