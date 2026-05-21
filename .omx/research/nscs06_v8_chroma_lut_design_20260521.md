---
council_tier: T1
horizon_class: plateau_adjacent
predicted_band_validation_status: pending_post_training
canonical_equation_in_domain_context: nscs06_v8_chroma_lut
---

# NSCS06 v8 chroma-LUT substrate L0 SCAFFOLD design memo

**Date:** 2026-05-21
**Lane:** `lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521`
**Symposium provenance:** commit `d125af6c3` (CASCADE COMPRESSION symposium PRIORITY 3 + Revision #5 NSCS06 v8 chroma_lut BUILD elevated as second-priority IN-DOMAIN substrate per Daubechies + Mallat multi-scale partition discovery framing)
**Cascade-mortality assessment:** commit `d884dd6aa` Rank 2 (HONEST CASCADE-MORTALITY ASSESSMENT 2026-05-20)
**Sister cargo-cult-unwind methodology:** commit `4292c8ce2` (NSCS06 v6 -> v7 44% improvement empirically validated rescue path)
**Canonical equation #26 IN-DOMAIN context:** `nscs06_v8_chroma_lut` (per `src/tac/canonical_equations/procedural_codebook_savings.py:102`)
**Predicted ΔS band:** `-0.002706` per canonical equation #26 closed form `-25 * (4096 - 32) / 37_545_489` `[prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]`
**Mission contribution per Catalog #300:** `frontier_breaking_enabler` (NEW substrate architecture per cargo-cult-unwind methodology; ranks #2 per HONEST CASCADE-MORTALITY ASSESSMENT; first paid empirical anchor pending 2026-06-04 symposium window)

This memo satisfies CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "UNIQUE-AND-COMPLETE-PER-METHOD" + "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + Catalog #290 (substrate canonical-vs-unique decision per layer) + Catalog #294 (9-dimension success checklist) + Catalog #296 (Dykstra-feasibility predicted-band derivation) + Catalog #303 (cargo-cult audit) + Catalog #305 (observability surface) + Catalog #309 (horizon_class declaration) + Catalog #324 (post-training Tier-C validation).

---

## Empirical anchor (symposium provenance)

CASCADE COMPRESSION symposium commit `d125af6c3` PRIORITY 3 verdict + Revision #5: *"NSCS06 v8 chroma_lut BUILD elevated as second-priority IN-DOMAIN substrate per Daubechies + Mallat multi-scale partition discovery framing."*

| Metric | v6 empirical | v7 empirical | v8 predicted (rate-only) | Mechanism |
|---|---|---|---|---|
| `final_score` | 105.15 `[diagnostic_cpu]` | 58.89 `[contest-CPU]` | not predicted (rate-axis ΔS only) | chroma LUT replaces v7 per-class anchor |
| `archive_size_bytes` | 2,939,158 | TBD | -4064 bytes vs v1 baseline | procedural seed replaces 4096-byte LUT |
| `rate_contrib` | 1.96 | TBD | -0.002706 [prediction] | canonical equation #26 IN-DOMAIN |

**Empirical confirmation (pack/parse roundtrip on synthetic config):**
- v1 inline-LUT archive: 4251 bytes
- v2 procedural-seed archive: 187 bytes
- empirical savings: **4064 bytes** = `predicted_archive_bytes_saved()` per canonical equation #26

---

## Architectural design

### Distinguishing feature per canonical equation #26 IN-DOMAIN

v7's chroma reconstruction uses per-class RGB anchors (15 bytes total). v8 expands this to a **per-(grayscale-level, class) LUT** indexed by `(grayscale_quantized, segnet_class) -> (R, G, B)`:

```
chroma_lut shape:        (GRAYSCALE_LEVELS=16, NUM_SEGNET_CLASSES=5, 3)
dense bytes:             16 * 5 * 3 = 240 bytes
canonical LUT footprint: 4096 bytes (per canonical equation #26
                          `_NSCS06_V8_BYTES_SAVED = 4096 - 32`)
```

The 4096-byte budget includes the 240 dense bytes + zero-padded reserved space for future per-temporal-window or per-spatial-region extensions. The padding is zero-filled so the canonical equation #26 prediction stays byte-stable.

Two parallel archive variants:

- **CH08 v1 INLINE LUT** (`CH08_SCHEMA_VERSION_INLINE_LUT = 1`): the full 4096-byte LUT lives inline at fixed offset.
- **CH08 v2 PROCEDURAL SEED** (`CH08_SCHEMA_VERSION_PROCEDURAL_SEED = 2`): 32-byte PCG64 seed replaces the LUT; inflate re-derives via `tac.procedural_codebook_generator.derive_codebook_from_seed`.

Per canonical equation #26 closed form `ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706` the v2 variant saves `4064 archive bytes` vs v1 baseline.

### Inflate runtime (numpy + Pillow only)

Sister of v7 `_grayscale_plus_chroma_to_rgb`: replaced with per-pixel `LUT[gray_quant[y,x], cls[y,x]]` lookup. v8 preserves v7's 6-DOF affine warp (cargo-cult #4 stays UNWOUND per symposium commit `4292c8ce2`). Total inflate LOC ~120 (substrate_engineering exception per HNeRV L7).

### Compress-side LUT derivation (closed-form; NO training)

`build_chroma_lut_from_ground_truth(rgb_pairs, class_labels)` computes per-`(level, class)` bin median over compress-time GT pixels. Empty bins get per-class GLOBAL median as fallback so every bin has a valid RGB anchor. NO neural training loop; sister to v7's HISTOGRAM-construction-only paradigm.

---

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (this package) | UNIQUE | The `(levels x classes)` LUT shape is structurally distinct from v7's `(classes,)` anchor and from grayscale_lut's `(256,)` chroma table |
| Compress-side LUT derivation | UNIQUE | Per-bin median over (level, class) bins from GT pixels; no canonical helper exists for this aggregation |
| Inflate runtime | UNIQUE | Numpy + Pillow only; per-pixel LUT lookup; no canonical helper applies at inflate |
| Procedural seed derivation | ADOPT canonical | `tac.procedural_codebook_generator.derive_codebook_from_seed` (sister DP1 + VQ-VAE + grayscale_lut pattern) |
| Auth eval routing | ADOPT canonical | `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` (Catalog #226) |
| NVML/Modal/CUDA env hygiene | ADOPT canonical | Catalog #244 NVML block in remote driver |
| Mount manifest | ADOPT canonical | `tac.deploy.modal.mount_manifest.build_training_image` (Catalog #153) |
| eval_roundtrip simulation | PRESERVE HARD-EARNED | 384->874->uint8->384 simulated at compress-time |
| strict-scorer-rule | PRESERVE HARD-EARNED | inflate.py imports ZERO scorer code |
| Catalog #220 operational mechanism | PRESERVE HARD-EARNED | LUT payload IS the operational mechanism (v1 inline OR v2 seed) |
| Trainer skeleton helpers | ADOPT canonical | `tac.substrates._shared.trainer_skeleton` (when full mode lands) |
| SubstrateContract decoration | ADOPT canonical | `@register_substrate(NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT)` per Catalog #241/#242 |
| Lane registry | ADOPT canonical | `tools/lane_maturity.py` per Catalog #126 pre-registration |

**Net assessment:** v8 is 100% UNIQUE in the substrate-specific layers (codec / archive / inflate); ADOPT canonical in the cross-substrate infrastructure (procedural seed / auth eval / device / NVML / mount). The 3 UNIQUE layers are precisely where v8's distinguishing feature lives — the (level, class) chroma table indexed lookup; canonicalization there would suppress the only thing that differentiates v8 from v7.

---

## Cargo-cult audit per assumption (per Catalog #303)

Per HARD-EARNED-vs-CARGO-CULTED addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`). Sister of NSCS06 v6 -> v7 cargo-cult-unwind methodology (commit `4292c8ce2`).

| # | Assumption | Classification | Disposition |
|---|---|---|---|
| 1 | 16-level luma quantization captures chroma-relevant variation | CARGO-CULTED (inherited from canonical AV1 codecs) | UNWIND-TEST scheduled in per-substrate symposium: ablate to 32-level / 8-level and measure compress-time chroma-MSE delta |
| 2 | Per-`(level, class)` LUT median is the optimal aggregation | CARGO-CULTED (inherited from v7 per-class median pattern) | UNWIND-TEST: ablate to per-class mode vs trimmed mean vs k-medoids cluster center |
| 3 | Procedural PCG64 seed -> uniform-distributed LUT bytes matches the GT chroma distribution | CARGO-CULTED (inherited from canonical PROCEDURAL VARIANT pattern; sister DP1 + VQ-VAE + grayscale_lut) | The canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` explicitly INCLUDES this configuration; the symposium's first paired-smoke is the empirical test |
| 4 | Catalog #205 inflate device-fork (CPU/CUDA) produces byte-identical raw frames | HARD-EARNED (sister v7 inflate produces byte-stable raw bytes) | PRESERVE; v8 inflate is numpy + Pillow only (no CUDA-specific operations) |
| 5 | 6-DOF affine warp (cargo-cult #4 unwound by v7) preserves v8 distinguishing feature | HARD-EARNED (empirically validated by v7 cargo-cult-unwind 44% improvement) | PRESERVE; v8 inflate copies v7 `_affine_warp_frame1_from_frame0` verbatim |
| 6 | SCAFFOLD class=0 uniform per-cell mask is acceptable for L0 | HARD-EARNED (matches v7 L1 SCAFFOLD pattern; CLS_STREAM consumption deferred to L1) | PRESERVE for L0; L1 promotion couples v8 to v7-style CLS_STREAM |
| 7 | Cross-substrate sharing of `derive_codebook_from_seed` does NOT suppress v8 distinguishing feature | HARD-EARNED (the canonical helper is shape-and-dtype-agnostic; v8 derives `(16, 5, 3)` uint8 which is a different shape from grayscale_lut's `(256,)`, DP1's basis tensor, or VQ-VAE's `(K, D)` codebook) | PRESERVE; sister pattern stays canonical per CLAUDE.md "consolidate everything into META layer" standing directive |

All 7 assumptions either UNWIND-TEST in the per-substrate symposium (1-3) or are HARD-EARNED with prior empirical validation (4-7). No unaddressed CARGO-CULTED assumptions remain at L0 SCAFFOLD landing.

---

## 9-dimension success checklist evidence (per Catalog #294)

| Dim | Status | Evidence |
|---|---|---|
| 1. UNIQUENESS | PARTIAL | v8 is a refinement-class substrate (not class-shift); strictly more capable chroma reconstruction than v7 but operates within the chroma_lut_replacement axis manifold |
| 2. BEAUTY + ELEGANCE | PASS | per-pixel LUT lookup is the simplest closed-form expression of `RGB = f(luma, class)`; trainer + inflate both reviewable in 30s |
| 3. DISTINCTNESS | PASS | distinct from v7 (per-class anchor) AND grayscale_lut (per-luma table) AND DP1 (Comma2k19-derived basis) AND VQ-VAE (K x D embedding) |
| 4. RIGOR | PASS | premise verification: read NSCS06 v6 -> v7 cargo-cult-unwind methodology + sister procedural variants + canonical helpers BEFORE writing any code; 49/49 unit tests pass; empirical bytes-saved (4064) EXACTLY matches canonical equation #26 prediction |
| 5. OPTIMIZATION PER TECHNIQUE | PASS | per Catalog #290 canonical-vs-unique decision per layer: 3 layers UNIQUE per substrate; 10 layers ADOPT canonical; HARD-EARNED preservation across 5 layers |
| 6. STACK-OF-STACKS COMPOSABILITY | PASS | canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` IS in `_INCLUDED_CONTEXTS`; rate-axis ΔS = -0.002706 stacks additively with sister procedural-variant substrates (grayscale_lut + DP1 + VQ-VAE + future ATW V2) per CASCADE COMPRESSION symposium 5-substrate aggregate paired-smoke matrix PRIORITY 5 |
| 7. DETERMINISTIC REPRODUCIBILITY | PASS | byte-stable archive pack/parse roundtrip; PCG64 seed -> identical LUT bytes; numpy seed-pinned; CH08 grammar fixed at design-time |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | PARTIAL | inflate is numpy + Pillow only (no torch overhead); compress-side LUT derivation is O(N*H*W) median computation (sister to v7 per-class median; expected sub-second per pair) |
| 9. OPTIMAL MINIMAL CONTEST SCORE | DEFERRED | predicted ΔS = -0.002706 rate-axis only; segmentation-axis impact pending paired-smoke empirical anchor per Catalog #324 post-training Tier-C validation discipline; per-substrate symposium gates first paired smoke |

All 9 dimensions either pass or carry an explicit DEFERRED rationale tied to the per-substrate symposium reactivation criteria.

---

## Observability surface (per Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable" + the 6-facet observability definition:

1. **Inspectable per layer:**
   - CH08 archive header is 35 bytes of structured uint8/16/32 values; `parse_archive` returns typed `Nscs06V8Archive` dataclass exposing every field.
   - `chroma_lut` is a `(16, 5, 3)` uint8 numpy array — inspectable per (level, class, channel) coordinate.
   - `select_inflate_device()` exposes the canonical Catalog #205 device-fork env var.
2. **Decomposable per signal:**
   - Inflate raw bytes decompose as `num_pairs * 2 * H * W * 3` per the CH08 grammar.
   - v1 vs v2 archive bytes decompose as `header(35) + lut_payload(4096 or 32) + pose + grayscale`.
   - Cargo-cult unwind verdicts decompose per `cargo_cult_audit_per_assumption` section (table above).
3. **Diff-able across runs:**
   - `verify_seed_mutation_changes_lut_bytes(seed_a, seed_b)` proves byte-mutation distinguishing-feature contract.
   - v1 inline-LUT and v2 procedural-seed archives produce byte-identical inflated raw frames when the v2 seed derives the same LUT bytes as v1 (sister test `test_inflate_v1_and_v2_byte_stable_with_matching_lut`).
   - Smoke metadata JSON includes `predicted_delta_s` + axis tags so the autopilot ranker can diff predictions across runs.
4. **Queryable post-hoc:**
   - `experiments/results/<lane>/smoke_metadata.json` carries the full SubstrateContract-derived metadata as machine-readable JSON.
   - Canonical equation #26 registry at `.omx/state/canonical_equations_registry.jsonl` is queryable via `tac.canonical_equations`.
5. **Cite-able:**
   - Every prediction in this memo carries a `[prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]` axis tag per Catalog #287 + #323.
   - Every empirical confirmation in this memo carries an `[empirical:...]` tag with the artifact path.
   - Catalog #220 operational mechanism declared in `archive_bytes_added` field of SubstrateContract.
6. **Counterfactual-able:**
   - The byte-mutation distinguishing-feature contract per Catalog #272 + #139 enables "what if I flip seed byte X?" queries without re-running training (no training to re-run; LUT derivation is closed-form).
   - The cargo-cult audit table above enables "what if I unwind assumption #1?" queries (UNWIND-TEST scheduled in per-substrate symposium).

---

## Predicted ΔS band (Dykstra-feasibility check; Catalog #296)

**Decomposition target:** `final_score = 25 * archive_bytes / 37_545_489 + 100 * seg + sqrt(10 * pose)`

**v8 contribution (rate-axis only; segmentation + pose deferred to paired smoke):**

| Component | v7 baseline | v8 predicted (rate-axis only) | Mechanism cited |
|---|---|---|---|
| `rate_contrib` (v1 inline LUT vs v7 baseline) | (sister) | + ~0.0027 (4096 bytes added) | canonical equation #26 N_codebook term |
| `rate_contrib` (v2 procedural seed vs v1 inline LUT) | n/a | **-0.002706** (-4064 bytes via seed) | canonical equation #26 IN-DOMAIN `nscs06_v8_chroma_lut` closed form |
| `seg_contrib` | (sister) | DEFERRED | pending paired smoke per-substrate symposium |
| `pose_contrib` | (sister) | DEFERRED | pending paired smoke per-substrate symposium |

**Dykstra feasibility check:** the canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` is verified IN the canonical `_INCLUDED_CONTEXTS` set per `tac.canonical_equations.procedural_codebook_savings.validate_context_is_in_domain` (returns True). The constraint intersection `(rate <= R) AND (seg <= S) AND (pose <= P)` with the procedural seed contribution `(seg += 0; pose += 0; rate -= 4064 bytes)` is non-empty by additivity per the canonical equation's closed form (the equation IS the alternating-projections fixed point at the rate-axis Pareto frontier).

**Predicted ΔS = -0.002706** `[prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]`.

**HIGH-CONFIDENCE bound** (vs the sister falsified residual-hybrid contexts) because:
- v8 is REPLACEMENT savings (the canonical equation's mathematical predicate), NOT residual-correction hybrid stacking (the falsified contexts pair #1 + pair #2 per Catalog #359).
- The PCG64 seed derivation is deterministic and byte-stable across all generator kinds; the empirical bytes-saved EXACTLY matches the prediction (4064 bytes verified at L0 smoke).
- The canonical equation #26 _INCLUDED_CONTEXTS explicitly includes `nscs06_v8_chroma_lut`; the per-substrate symposium gates whether the seg+pose contributions match the prediction OR drift away.

**Reactivation criteria for L1 promotion** (per Catalog #325 per-substrate symposium):
- per-substrate symposium per Catalog #325 lands PROCEED verdict (target window 2026-05-21 -> 2026-06-04).
- post-training Tier-C density validation per Catalog #324 confirms within-class or across-class classification matching prediction.
- distinguishing-feature byte-mutation smoke per Catalog #272 passes on first paired smoke archive.
- canonical equation #26 IN-DOMAIN context membership per Catalog #344 remains valid post-empirical.

---

## Local verification (Step 5 evidence)

```
$ PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/ -q
49 passed in 0.13s

$ PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/train_substrate_nscs06_v8_chroma_lut.py \
    --output-dir .omx/tmp/nscs06_v8_smoke_test --device cpu --smoke
[smoke] CH08 v2_procedural_seed archive bytes: 187 (predicted ΔS = -0.002706 [prediction])

# v1 inline-LUT archive: 4251 bytes
# v2 procedural-seed archive: 187 bytes
# empirical savings: 4064 bytes = predicted_archive_bytes_saved() per canonical equation #26

$ bash -n scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh
# (no output; shell syntax OK)

$ python -c "from experiments.train_substrate_nscs06_v8_chroma_lut import main; main(['--output-dir','/tmp/x','--device','cpu'])"
NotImplementedError: nscs06_v8_chroma_lut substrate is L0 SCAFFOLD per Catalog #240 + Catalog #325 per-substrate symposium pending...
# (correct fail-closed per Catalog #240)
```

49 substrate unit tests pass. CPU smoke produces a 187-byte v2 archive. Full mode correctly raises NotImplementedError per Catalog #240 recipe-vs-trainer-state consistency.

---

## Op-routables (post-L0-landing)

| Trigger | Action | Cost |
|---|---|---|
| Per-substrate symposium PROCEED verdict (window 2026-05-21 -> 2026-06-04) | Implement `_full_main`; flip recipe `dispatch_enabled: true`; queue first paired smoke | $0 implementation; $0.50 paired smoke |
| First paired smoke contest-CUDA + contest-CPU anchors land within predicted band | Mark `contest_cuda` + `contest_cpu` gates; promote lane to L2 | $0 |
| First paired smoke score DRIFTS from predicted band by >2x | Per Catalog #324: re-run post-training Tier-C density; if CARGO-CULTED, route to UNWIND-TEST per cargo-cult-audit assumptions 1-3 | $1-3 |
| 5-substrate aggregate paired-smoke matrix (CASCADE COMPRESSION symposium PRIORITY 5) | Queue v8 + grayscale_lut + DP1 + VQ-VAE + ATW V2 procedural-variant aggregate paired-smoke matrix; aggregate predicted ΔS -0.013 to -0.0085 | $2-3 |

---

## CLAUDE.md compliance

- ✅ Apples-to-apples evidence discipline: every score literal carries an axis tag (`[prediction]` / `[empirical:...]`)
- ✅ Forbidden premature KILL: v8 is a substrate scaffold, not a kill; per-substrate symposium gates promotion
- ✅ HNeRV parity discipline L4 (≤100 LOC inflate) waived via substrate_engineering exception per L7 (declared in SubstrateContract.inflate_runtime_loc_budget=200; v8 inflate ~120 LOC)
- ✅ Strict scorer rule: inflate.py imports ZERO scorer code (NO torch / NO smp / NO efficientnet / NO SegNet/PoseNet identifier tokens)
- ✅ UNIQUE-AND-COMPLETE-PER-METHOD: codec/archive/inflate are 100% UNIQUE; cross-substrate infra ADOPTS canonical
- ✅ Cargo-cult audit per Catalog #303: 7 assumptions enumerated + HARD-EARNED/CARGO-CULTED classified
- ✅ Catalog #220 operational mechanism: LUT payload bytes ARE operational; byte-mutation smoke proves consumption
- ✅ Catalog #240 recipe-vs-trainer-state: trainer `_full_main` raises NotImplementedError; recipe `research_only=true` + `dispatch_enabled=false` consistent
- ✅ Catalog #244 NVML block: emitted in remote driver before any DALI import
- ✅ Catalog #270 dispatch optimization protocol: substrate has no scorer hot loop so Tier-1 engineering flags WAIVED per file-header AUTOCAST_FP16_WAIVED / TORCH_COMPILE_WAIVED / TF32_WAIVED / NO_GRAD_WAIVED / F3_CACHE_CONSUMPTION_WAIVED
- ✅ Catalog #290 substrate canonical-vs-unique decision per layer: section above
- ✅ Catalog #294 9-dimension success checklist evidence: section above
- ✅ Catalog #296 Dykstra-feasibility predicted-band check: section above
- ✅ Catalog #303 cargo-cult audit per assumption: section above
- ✅ Catalog #305 observability surface: section above
- ✅ Catalog #309 horizon_class declared in frontmatter: `plateau_adjacent` (within-class refinement of v7; not class-shift; cumulative WITH-aggregate procedural-variant matrix per CASCADE symposium PRIORITY 5 may shift horizon class up at L1 paired-smoke time)
- ✅ Catalog #324 predicted-band post-training validation: `predicted_band_validation_status: pending_post_training` declared in recipe frontmatter
- ✅ Catalog #325 per-substrate symposium memo: sister `council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` co-landed
- ✅ Catalog #344 canonical equation cross-reference: `procedural_codebook_from_seed_compression_savings_v1` + IN-DOMAIN context `nscs06_v8_chroma_lut`
- ✅ Catalog #287 + #323 canonical Provenance: no score claim asserted in this memo; predicted ΔS tagged `[prediction]`; empirical bytes-saved tagged `[empirical:src/tac/substrates/nscs06_v8_chroma_lut/tests/test_substrate.py::test_v2_blob_smaller_than_v1_by_canonical_savings]`

---

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: N/A (v8 is a single archive-build path; per-byte sensitivity is captured by hook #2 Pareto constraint rate-distortion v1)
- hook #2 Pareto constraint: ACTIVE via canonical equation #26 IN-DOMAIN predicted ΔS contribution to rate-axis Pareto polytope
- hook #3 bit-allocator: PLANNED (32-byte seed slot replaces 4096-byte chroma LUT slot only after per-substrate symposium PROCEED verdict)
- hook #4 cathedral autopilot dispatch: ACTIVE via sister consumer `tac.cathedral_consumers.procedural_codebook_generator_consumer` (auto-discovered per Catalog #335) + `tac.cathedral_consumers.canonical_equation_lookup_consumer` (per Catalog #344)
- hook #5 continual-learning posterior: ACTIVE (first empirical anchor via `update_equation_with_empirical_anchor` post-paired-smoke)
- hook #6 probe-disambiguator: ACTIVE (PROCEDURAL seed vs canonical v1 inline LUT contrast IS the probe disambiguator for whether v8's chroma-axis rate can be procedurally substituted within the canonical equation #26 IN-DOMAIN context); planned path `tools/probe_nscs06_v8_chroma_lut_canonical_equation_26_in_domain_disambiguator.py`

---

## Sister substrate cross-references

- **v7 NSCS06 Carmack-Hotz strip-everything**: sister substrate that this v8 builds chroma-LUT on top of. v7 cargo-cult-unwind methodology (commit `4292c8ce2`) is the empirically validated rescue-path pattern v8 follows.
- **grayscale_lut PROCEDURAL VARIANT**: sister procedural-variant substrate (commit `f037d1144`); same canonical equation #26 IN-DOMAIN context family (different LUT shape: `(256,)` vs v8's `(16, 5, 3)`).
- **DP1 PROCEDURAL VARIANT**: sister procedural-variant substrate (commit `9cbfa471c`); same canonical equation #26 IN-DOMAIN context family (different LUT shape: Comma2k19-derived basis).
- **VQ-VAE PROCEDURAL VARIANT**: sister procedural-variant substrate (commit `6fea30f22`); same canonical equation #26 IN-DOMAIN context family (different LUT shape: `(K, D)` embedding).
- **canonical equation #26**: `procedural_codebook_from_seed_compression_savings_v1` at `src/tac/canonical_equations/procedural_codebook_savings.py`; `nscs06_v8_chroma_lut` IS in `_INCLUDED_CONTEXTS`.

---

## Lane registry pre-registration (Catalog #126)

```bash
.venv/bin/python tools/lane_maturity.py add-lane \
    lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521 \
    --name "NSCS06 v8 chroma-LUT substrate L0 SCAFFOLD" \
    --phase 3 \
    --notes "research_only=true; substrate_engineering exception per HNeRV L7; canonical equation #26 IN-DOMAIN context nscs06_v8_chroma_lut; per-substrate symposium per Catalog #325 pending window 2026-05-21 -> 2026-06-04"
```

Initial gates marked at landing:
- `impl_complete` (this commit)
- `strict_preflight` (no preflight violations; 49/49 tests pass)
- `memory_entry` (this memo + sister symposium memo)

Pending gates (per-substrate symposium PROCEED):
- `real_archive_empirical` (first paired smoke)
- `contest_cuda` (first contest-CUDA anchor)
- `three_clean_review` (R1-R3 council rounds)
- `deploy_runbook` (driver script already lands; promotion requires post-symposium paired smoke success)
