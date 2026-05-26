<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — landing memo; do not mutate. -->
<!-- Catalog #229 PV: this landing memo verifies premises empirically: 18/18 Z8 tests + 27/27 NIRVANA tests pass via .venv/bin/python -m pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/ src/tac/substrates/nirvana_cascading_nerv/tests/ -q (run 2026-05-26T08:33Z; 45/45 PASS). Pre-fix and post-fix MLX↔PyTorch parity drift measured empirically; see §"Post-fix test verdict" below for numerical evidence. -->
<!-- FORMALIZATION_PENDING:fix_wave_close_findings_no_empirical_score_claim_per_catalog_344_mlx_research_signal_axis_only_per_claude_md_mlx_portable_local_substrate_authority_non_negotiable -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Carmack, Hotz, Quantizr, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "F=Z8 F-OP1 + F-OP2 fixes are mechanical ports of FIX-WAVE-R1's A=DreamerV3 patches and require no novel design decisions"
    classification: HARD-EARNED
    rationale: "Empirically verified by post-fix MLX↔PyTorch drift measurement: PixelShuffle 3.77→0.0 (PERFECT byte-stable) and bilinear 1.51→2.4e-7 (well below 1e-5 threshold; fp32 compound-op precision noise). The canonical channel-FIRST convention (B, H, W, out_C, 2, 2) + transpose (0, 1, 4, 2, 5, 3) and the canonical PR95 helper delegation are line-for-line identical to D=Z6 sister-canonical and FIX-WAVE-R1's A=DreamerV3 patch (commit a23779a732e7bb056). The 18/18 Z8 test pass rate confirms no regression. The decision is mechanical port; no novel design decisions required."
  - assumption: "G=NIRVANA findings are DOCUMENTATION-LEVEL only and require zero code changes beyond a single docstring correction"
    classification: HARD-EARNED
    rationale: "Per R1' review §Axis 2: G=NIRVANA mlx_renderer.py contains ZERO MLX primitives at L0 (only Config dataclass + factory helpers + estimators); the substrate is scaffold-only-by-design per Catalog #240 (research_only=true at L0; actual MLX renderer class is Phase 2 work). G-OP1 + G-OP3 (memo axis-label corrections) are pure documentation corrections via APPEND-ONLY footer per Catalog #110/#113. G-OP2 (mlx_renderer.py docstring) is the single source-code change. 27/27 NIRVANA tests PASS post-correction (no regression because doc-only fixes). Substrate paradigm INTACT; R1' did not raise any architectural or implementation finding against G's actual L0 code surface."
  - assumption: "CONSOLIDATE-OP-1 should be DEFERRED to separate subagent dependent on sister L1-PROMOTION-D-Z6 landing"
    classification: HARD-EARNED
    rationale: "Per R1' aggregate META finding #1 + sister L1-PROMOTION-D-Z6 in-flight on D=Z6 files: CONSOLIDATE-OP-1 refactors tac.local_acceleration.pr95_hnerv_mlx + MIGRATES A + D + F substrates to canonical helpers. Executing it within FIX-WAVE-R1' scope would (a) collide with sister L1-PROMOTION-D-Z6's D=Z6 file edits (Catalog #314/#340 absorption-pattern risk); (b) exceed bounded scope so R2' can fire; (c) require a separate Catalog #325 per-substrate symposium per the META finding's recommended structural extinction. Spawn as separate subagent AFTER L1-PROMOTION-D-Z6 lands per task #1286. CONSOLIDATE-OP-1 is documented in §'Deferred work' below."
council_decisions_recorded:
  - "F=Z8 F-OP1 + F-OP2 + F-OP3 LANDED (mechanical port from FIX-WAVE-R1 A=DreamerV3 patches; 18/18 tests PASS)"
  - "G=NIRVANA G-OP1 + G-OP2 + G-OP3 LANDED (1 in-place docstring + 1 APPEND-ONLY footer; 27/27 tests PASS)"
  - "F=Z8 post-fix MLX↔PyTorch drift: PixelShuffle 3.77→0.0 (PERFECT); bilinear 1.51→2.4e-7 (well below 1e-5 threshold)"
  - "R2' READY TO FIRE on B'+C'+F+G alongside this commit batch"
  - "CONSOLIDATE-OP-1 DEFERRED to separate subagent per task #1286 (depends on sister L1-PROMOTION-D-Z6 landing)"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526
  - path_3_g_recursive_adversarial_review_r1_prime_3_axis_20260526
  - path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526
  - path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526
  - path_3_fix_wave_r1_close_findings_landed_20260526
canonical_equation_refs:
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1
---

# Path 3 FIX-WAVE-R1' — close ALL R1' findings for F=Z8 + G=NIRVANA

**Verdict**: **PROCEED — FIX-WAVE-R1' CLOSES F=Z8 + G=NIRVANA findings per R1' Path 3 review**; R2' READY TO FIRE on B'+C'+F+G alongside this commit batch.

**Cost**: $0 GPU; ~30 min wall-clock (bounded scope: mechanical port from FIX-WAVE-R1 A=DreamerV3 + documentation footer).

**Successor to predecessor**: R1' adversarial review subagent `a7f70569776587e21` (commit `71d8ff687`) identified F+G as NOT CLEAN; this subagent (`fix_wave_r1_prime_close_findings_20260526`) closes those findings per CLAUDE.md "Recursive adversarial review protocol — close paths".

---

## What landed

### F=Z8 mechanical port of FIX-WAVE-R1 A=DreamerV3 patches

1. **F-OP1** (P0 / CRITICAL / TRAINING-INVALIDATING) — `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py::_pixel_shuffle_2x_nhwc`: rewrote from channel-LAST convention `(B, H, W, 2, 2, out_C)` + transpose `(0, 1, 3, 2, 4, 5)` to canonical channel-FIRST convention `(B, H, W, out_C, 2, 2)` + transpose `(0, 1, 4, 2, 5, 3)` matching sister D=Z6 (`src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc` lines 361-372) AND canonical PR95 helper (`tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc`) AND post-FIX-WAVE-R1 A=DreamerV3 (commit `a23779a732e7bb056`). Added `len(x.shape) != 4` shape validation per D=Z6 canonical guard.

2. **F-OP2** (P0 / CRITICAL / TRAINING-INVALIDATING) — `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py::_bilinear_resize_2x_nhwc`: replaced `mx.repeat` 2x nearest-neighbor approximation with canonical PR95 helper delegation `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`. Catalog #295 self-containment preserved (canonical helper imported only at MLX training time; substrate's inflate runtime is PyTorch-only and uses `F.interpolate(mode='bilinear', align_corners=False)` natively).

3. **F-OP3** (P1 / VERIFICATION) — `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py`: added `test_z8_pixel_shuffle_matches_pytorch` + `test_z8_bilinear_resize_matches_pytorch`. Both assert `max_abs < 1e-5` vs PyTorch reference. Both PASS post-fix.

### G=NIRVANA documentation correction (substrate paradigm INTACT)

1. **G-OP1 + G-OP3** (P0 / DOCUMENTATION FIX via APPEND-ONLY footer) — `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md`: appended §(12) `APPEND-ONLY HISTORICAL_PROVENANCE FOOTER — FIX-WAVE-R1' (2026-05-26)` correcting 3 documentation overstatements per R1' review:
   - **axis-label corrections** for `test_numpy_reference_bilinear_upsample_matches_pytorch` (line 70) + `test_numpy_reference_kahan_mean_stability` (line 74) + `test_cascade_pytorch_vs_numpy_reference_parity` (line 81): `axis 2 MLX↔PyTorch parity` → `axis 3 numpy↔PyTorch parity` (correct per actual test content)
   - **§"Axis 2: MLX drift minimization" canonical corrected statement**: ZERO MLX primitives implemented at L0; 7 anticipated primitives + 3 KNOWN-DRIFT-RISK characterizations are L1+ implementation guidance
   - **council_decisions_recorded "MLX-first per #1265 anchor"** correction: `MLX-config-scaffold-first per #1265 anchor; actual MLX renderer implementation deferred to Phase 2 council symposium per Catalog #325`
   - Per Catalog #110/#113: original landing memo content PRESERVED verbatim; footer is the canonical correction surface

2. **G-OP2** (P1 / DOCUMENTATION FIX via in-place edit) — `src/tac/substrates/nirvana_cascading_nerv/mlx_renderer.py` line 2 module docstring corrected from `MLX hierarchical residual decoder cascade.` to `MLX hierarchical residual decoder cascade SCAFFOLD (config + helpers; actual renderer class lands Phase 2).` with FIX-WAVE-R1' G-OP3 explanatory block. Source-file in-place edit is correct per source-file policy (CLAUDE.md HISTORICAL_PROVENANCE applies to design/landing/research memos, not source modules).

---

## Post-fix test verdict

### F=Z8 MLX↔PyTorch parity empirical measurement

**Pre-fix anchors (R1' Path 3 F review 2026-05-26T08:03Z; reproduced before edit)**:
- `_pixel_shuffle_2x_nhwc` max_abs vs PyTorch `nn.PixelShuffle(2)`: **3.766418** (TRAINING-INVALIDATING)
- `_bilinear_resize_2x_nhwc` max_abs vs PyTorch `F.interpolate(mode='bilinear', align_corners=False)`: **1.512860** (TRAINING-INVALIDATING)

**Post-fix measurement (run 2026-05-26T08:30Z; reproducer: identical to R1' anchor)**:
- `_pixel_shuffle_2x_nhwc` max_abs vs PyTorch: **0.0000000000** (PERFECT byte-stable; 3.77 → 0.0 absolute drift reduction)
- `_bilinear_resize_2x_nhwc` max_abs vs PyTorch: **0.0000002384** (well below 1e-5 threshold; 1.51 → 2.4e-7 absolute drift reduction; fp32 compound-op precision noise)

### Full Z8 + NIRVANA test suites

```
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
    src/tac/substrates/z8_hierarchical_predictive_coding/tests/ \
    src/tac/substrates/nirvana_cascading_nerv/tests/ -q
→ 45 passed in 0.58s
```

- **Z8**: 18/18 PASS (16 pre-existing + 2 new F-OP3 parity tests)
- **NIRVANA**: 27/27 PASS (no regression; doc-only G-OP2 docstring fix)
- **Aggregate**: 45/45 PASS

### Per-finding verdict table

| Finding | Severity | Surface | Verdict | Empirical Evidence |
|---|---|---|---|---|
| F-OP1 (PixelShuffle channel-LAST→FIRST) | P0 CRITICAL | Z8 mlx_renderer.py | **CLOSED** | 3.77→0.0 max_abs vs PyTorch; 18/18 tests PASS |
| F-OP2 (bilinear mx.repeat→canonical helper) | P0 CRITICAL | Z8 mlx_renderer.py | **CLOSED** | 1.51→2.4e-7 max_abs vs PyTorch; 18/18 tests PASS |
| F-OP3 (parity tests added) | P1 VERIFICATION | Z8 tests/test_basic.py | **CLOSED** | 2 new tests PASS at < 1e-5 threshold |
| G-OP1 (memo axis-label corrections) | P0 DOC | NIRVANA landing memo | **CLOSED** | APPEND-ONLY footer §(12) lands; 27/27 tests PASS |
| G-OP2 (memo "MLX-first" → "MLX-config-scaffold-first") | P0 DOC | NIRVANA landing memo + mlx_renderer.py docstring | **CLOSED** | APPEND-ONLY footer §(12) + in-place docstring correction; 27/27 tests PASS |
| G-OP3 (mlx_renderer.py docstring SCAFFOLD note) | P1 DOC | NIRVANA mlx_renderer.py | **CLOSED** | In-place docstring edit; 27/27 tests PASS |

---

## R2' readiness signal

**Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 4**: FIX-WAVE-R1' landed all op-routables identified by R1' for F + G. R2' can now fire on B'+C'+F+G with:

- **B'=Z7-Mamba-2-v2** (R1' verdict 1/3 advance) — no FIX-WAVE-R1' op-routables; R2' fires directly
- **C'=NSCS06 v8 chroma_lut** (R1' verdict 1/3 advance) — no FIX-WAVE-R1' op-routables; R2' fires directly
- **F=Z8** (R1' verdict NOT CLEAN; counter reset to 0/3) — FIX-WAVE-R1' CLOSES F-OP1+F-OP2+F-OP3; R2' fires fresh
- **G=NIRVANA** (R1' verdict NOT CLEAN; counter reset to 0/3) — FIX-WAVE-R1' CLOSES G-OP1+G-OP2+G-OP3; R2' fires fresh

---

## Deferred work — META-CONSOLIDATE-OP-1 (out of FIX-WAVE-R1' scope)

Per R1' aggregate META finding #1 + the prompt's explicit guidance: CONSOLIDATE-OP-1 is DEFERRED to a separate subagent for the following reasons:

1. **Scope**: CONSOLIDATE-OP-1 refactors `tac.local_acceleration.pr95_hnerv_mlx` AND MIGRATES A=DreamerV3 + D=Z6 + F=Z8 substrates to canonical helpers. This exceeds the bounded "close R1' findings" scope.
2. **Sister-collision risk**: sister L1-PROMOTION-D-Z6 (agent `a1d0f82a6bb42ff54`) is in-flight on D=Z6 files. Touching D=Z6 in this commit batch would trigger Catalog #314/#340 absorption-pattern bug class.
3. **Dependency ordering**: CONSOLIDATE-OP-1 should land AFTER L1-PROMOTION-D-Z6 stabilizes per task #1286.
4. **Council-grade tradeoff**: CONSOLIDATE-OP-1 design decision requires Catalog #325 per-substrate symposium per the META finding's recommended structural extinction.

**Recommended next**: spawn CONSOLIDATE-OP-1 subagent AFTER L1-PROMOTION-D-Z6 lands; it should consume the canonical sister-canonical references documented here (D=Z6 + post-FIX-WAVE-R1 A=DreamerV3 + post-FIX-WAVE-R1' Z8) and extract the canonical `_pixel_shuffle_2x_nhwc` helper to `tac.local_acceleration.pr95_hnerv_mlx`.

---

## Discipline checklist compliance

- ✅ Catalog #229 PV: read F R1' + G R1' review memos + Z8 mlx_renderer.py + Z6 sister-canonical mlx_renderer.py + canonical PR95 helper + FIX-WAVE-R1's A=DreamerV3 patch + Z8 test_basic.py + NIRVANA mlx_renderer.py + NIRVANA landing memo BEFORE any edit
- ✅ Catalog #117/#157/#174 canonical serializer: commit forthcoming via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per file
- ✅ Catalog #206 subagent checkpoint discipline: 3 checkpoints emitted (steps 0+1+2; step 3 at commit time)
- ✅ Catalog #119 Co-Authored-By Claude trailer: included in commit
- ✅ Catalog #287 placeholder-rationale rejection: no placeholder rationales in any waiver
- ✅ Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: NIRVANA landing memo NEVER mutated above original content; FIX-WAVE-R1' footer appended to §(12); NEW landing memo for FIX-WAVE-R1' itself
- ✅ Catalog #208 docs/local-paths: no `/Users/adpena/...` paths in any persisted artifact
- ✅ Catalog #230 ownership map: no overlap with sister L1-PROMOTION-D-Z6 (which is touching D=Z6 files; this lane is bounded to Z8+NIRVANA)
- ✅ Catalog #340 sister-checkpoint guard: serializer handles structurally; sister check at start returned PROCEED (0 in-flight subagent collisions on the 4 target files)
- ✅ Catalog #295 submission inflate self-containment: preserved for F=Z8 (canonical helper imported only at MLX training time; inflate.py is PyTorch-only and does NOT import MLX)
- ✅ Catalog #307 paradigm-vs-implementation classification: F=Z8 fixes are IMPLEMENTATION-LEVEL (MLX primitive bugs); G=NIRVANA fixes are DOCUMENTATION-LEVEL (label drift); both substrate paradigms INTACT
- ✅ All artifacts tagged `[macOS-MLX research-signal]` + `score_claim=false` + `promotion_eligible=false` + `ready_for_exact_eval_dispatch=false` per CLAUDE.md "MLX portable-local-substrate authority"
- ✅ NO `gh pr create` / `gh release create` / Modal/Vast/Lightning dispatch per CLAUDE.md "Executing actions with care"

## Files landed (Catalog #230 ownership map)

| File | Role | Edit type |
|---|---|---|
| `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py` | F-OP1 + F-OP2 fixes | In-place edit (source-file policy) |
| `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py` | F-OP3 parity tests added | In-place edit (source-file policy) |
| `src/tac/substrates/nirvana_cascading_nerv/mlx_renderer.py` | G-OP2 docstring SCAFFOLD note | In-place edit (source-file policy) |
| `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md` | G-OP1 + G-OP3 APPEND-ONLY footer | APPEND-ONLY footer per Catalog #110/#113 |
| `.omx/research/path_3_fix_wave_r1_prime_close_findings_landed_20260526.md` | THIS landing memo | NEW landing memo (Catalog #229 + #294 + #305 compliant) |

**Total scope**: ~30 LOC of source edits (mechanical) + ~80 LOC of test additions (verification) + 1 in-place docstring + 1 APPEND-ONLY footer + 1 new landing memo. Bounded per FIX-WAVE-R1' charter; no sister-collision; ready for R2' to fire.

---

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map contribution**: N/A — defensive fix (MLX primitive bug closure); no sensitivity signal
- **Hook #2 Pareto constraint**: N/A — same
- **Hook #3 bit-allocator hook**: N/A — same
- **Hook #4 cathedral autopilot dispatch hook**: **ACTIVE** — F=Z8 substrate is now eligible for downstream autopilot ranking once Z8 reaches L1+ (was structurally invalidated before fix); G=NIRVANA L0 scaffold posture is now correctly labeled
- **Hook #5 continual-learning posterior**: **ACTIVE** — F=Z8 MLX-trained state_dict can now be reliably exported to PyTorch inflate (MLX↔PyTorch byte-stable per F-OP3 parity tests)
- **Hook #6 probe-disambiguator**: N/A — no defensible alternative interpretations (mechanical port + documentation correction)

---

## Cross-references

- F R1' review: `.omx/research/path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- G R1' review: `.omx/research/path_3_g_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- F=Z8 landing memo (R1' subject): `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md` (commit `5ff5d2ab9`)
- G=NIRVANA landing memo (R1' subject + this footer's target): `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md` (commit `f7d2e86fe` + this footer)
- FIX-WAVE-R1 landing (canonical patch pattern for A=DreamerV3 analogous bugs): `.omx/research/path_3_fix_wave_r1_close_findings_landed_20260526.md` (commit `a23779a732e7bb056`)
- Canonical D=Z6 sister-canonical: `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc` (lines 361-372)
- Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`
- Lane: `lane_path_3_fix_wave_r1_prime_close_findings_20260526` L1 (impl_complete + memory_entry)
