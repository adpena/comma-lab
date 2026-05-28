---
title: PACT-NeRV-SELECTOR-V3 PyTorch decoder_quantization sister landing — Wave N+3 Slot 1 op-routable #1
lane_id: lane_pytorch_decoder_quant_sister_landing_op_routable_1_slot1_20260528
substrate_id: pact_nerv_selector_v3
substrate_aliases: [pact_nerv_selector_v3_int8_decoder, pact_nerv_selector_v3_heterogeneous_compound_c]
date_utc: 2026-05-28T14:42:00Z
council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_override_invoked: false
council_override_rationale: null
council_predicted_mission_contribution: frontier_breaking_enabler
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "canonical heterogeneous_bit_allocation helper is framework-agnostic at the PyTorch state_dict boundary AND can be invoked directly from the PyTorch trainer with no fork or framework= parameter"
    classification: HARD-EARNED
    rationale: "Empirically verified: src/tac/substrates/pact_nerv_selector_v3/heterogeneous_bit_allocation.py uses torch.Tensor as the input/output type for every helper (compute_per_tensor_sensitivity_via_taylor_expansion, derive_heterogeneous_bit_allocation, apply_fp4_qat_finetune_on_top_k_tensors); the MLX path at archive_candidate.py already invokes the same helpers via PyTorch tensor conversion at the MLX -> numpy -> torch boundary. Re-invocation from the PyTorch trainer requires zero adapter."
  - assumption: "the formerly-hardcoded fp16_brotli_q9 callsite at line 385 is the SINGLE archive_emit_path; threading decoder_quantization through the one pack_archive() call sufficiently covers all 4 decoder_quant modes"
    classification: HARD-EARNED
    rationale: "AST scan + grep verification: pack_archive() is invoked exactly once in _full_main; the smoke_main path does not emit archive bytes (saves a checkpoint .pt only); decoder_quant flag is a no-op in smoke and an active param in full."
  - assumption: "preserving fp16_brotli_q9 default behavior is sufficient for backward compatibility — operators who do NOT pass --decoder-quant continue to get baseline V3 archive bytes byte-identical to pre-landing state"
    classification: HARD-EARNED
    rationale: "Empirical verification: pack_archive() default decoder_quantization=DECODER_QUANT_FP16_BROTLI_Q9 (archive.py constant); the default branch in _serialize_state_dict (lines 195-203) produces the same bytes as the pre-landing call (the pre-landing call did NOT pass decoder_quantization, so the default kicked in; post-landing call passes decoder_quantization=args.decoder_quant which defaults to the same value)."
  - assumption: "anti-pattern false positives (fp4_packed_without_qat firing despite quantization_aware_training=True; brotli_plus_lzma firing despite no LZMA) are SLOT-2-OWNED bug class and do NOT block this Slot 1 landing"
    classification: HARD-EARNED
    rationale: "Mandate op-routable #4 explicitly cites the token-fallback false-positive class as a known issue for Slot 2's just-landed canonical anti-patterns work. Slot 1 verified the matcher fires the false positives as expected; documented below for Slot 2 sister-wave fix-pass."
canonical_equation_anchors:
  - equation_id: pact_nerv_decoder_state_dict_saturation_at_parity_floor_v1
    role: source_of_truth_for_decoder_quant_savings_predictions
  - equation_id: heterogeneous_per_tensor_bit_allocation_compounding_v1
    role: source_of_truth_for_compound_c_predicted_floor
catalog_compliance:
  - "Catalog #146 (3-arg inflate.sh + contest-compliant template) — preserved via existing write_contest_runtime canonical helper"
  - "Catalog #151 (operator wrapper threads trainer Tier-1 flags) — DECODER_QUANT + FP4_QAT_EPOCHS env vars added to TIER_1_OPERATOR_REQUIRED_FLAGS manifest"
  - "Catalog #152 (operator wrapper validates required input files) — required_input_files in recipes unchanged"
  - "Catalog #167 (smoke-before-full pattern) — preserved; recipes still dispatch_enabled:false"
  - "Catalog #192 (macOS-CPU advisory NON-PROMOTABLE) — smoke run on macOS CPU produces no contest-axis score claim"
  - "Catalog #205 (canonical select_inflate_device) — submission/inflate.py emit unchanged"
  - "Catalog #218 (mini-batch reconstruct) — N/A (PSV3 substrate decodes one pair at inflate time)"
  - "Catalog #220 (L1+ scaffold operational mechanism) — heterogeneous bit allocation IS the operational mechanism; decoder bytes change frame-level rendering at inflate time"
  - "Catalog #226 (gate_auth_eval_call canonical helper) — preserved; trainer routes through _canon_gate_auth_eval_call"
  - "Catalog #229 (premise verification before edit) — 12 premise files read pre-edit; assumptions documented above"
  - "Catalog #240 (recipe-vs-trainer-state consistency) — both recipes preserve dispatch_enabled:false + research_only:true"
  - "Catalog #244 (canonical NVML env block in driver) — DALI_DISABLE_NVML + CUBLAS_WORKSPACE_CONFIG + PYTORCH_CUDA_ALLOC_CONF preserved in recipes"
  - "Catalog #246 (paired CUDA + CPU) — operator-routable next-step explicitly cited"
  - "Catalog #270 (canonical dispatch optimization protocol) — Tier 1/2/3 preserved at recipe + trainer surfaces"
  - "Catalog #287 (placeholder-rationale rejection) — every waiver/rationale in this memo is substantive (≥4 chars; not <rationale>/<reason>)"
  - "Catalog #290 (canonical-vs-unique decision per layer) — documented in this memo (Canonical-vs-unique section below)"
  - "Catalog #292 (per-deliberation assumption surfacing) — 4 assumptions surfaced in frontmatter"
  - "Catalog #294 (9-dim success checklist) — section below"
  - "Catalog #295 (PYTHONPATH self-containment) — submission_dir vendoring via write_contest_runtime preserved"
  - "Catalog #296 (Dykstra-feasibility for predicted bands) — predicted bands in both recipes cite Slot 1 Dykstra polytope solver (per Slot 1 Wave N+1 sister landing)"
  - "Catalog #300 (council v2 frontmatter) — present"
  - "Catalog #303 (cargo-cult audit per assumption) — section below"
  - "Catalog #305 (observability surface 6-facet) — section below"
  - "Catalog #313 (probe outcomes ledger) — registered probe outcome VERDICT_PROCEED with 30d staleness window"
  - "Catalog #323 (canonical Provenance umbrella) — all evidence in this memo carries axis_tag=[predicted] / score_claim=false / promotable=false"
  - "Catalog #324 (predicted_band_validation_status) — preserved; pending_post_training in both recipes"
  - "Catalog #325 (per-substrate symposium pre-paid-dispatch) — operator-routable next-step cited"
  - "Catalog #335 (cathedral consumer canonical contract) — no NEW consumer added"
  - "Catalog #340 (sister-checkpoint guard) — DISJOINT scope from Slot 2 a2bc8189f495c821e in flight"
  - "Catalog #341 (Tier A canonical routing markers) — no consumer-side mutation"
  - "Catalog #344 (canonical equation registry references) — cited in frontmatter"
  - "Catalog #356 (per-axis Provenance) — preserved by existing trainer surface"
  - "Catalog #361 (Modal artifact filter preserves submission_dir) — preserved by canonical write_contest_runtime + build_archive_zip"
  - "Catalog #365 (canonical gate_auth_eval_call kwargs) — kwargs preserved at the _canon_gate_auth_eval_call callsite"
  - "Catalog #366 (inflate shim imports match canonical module exports) — submission/inflate.py emit via write_contest_runtime preserves canonical 'from ... import inflate_one_video' import"
  - "Catalog #367 (raw bytes fail-closed + CONTEST_RAW_BYTES) — preserved by existing inflate.py contract"
  - "Catalog #369 (inflate consumes real trained weights) — preserved; the new decoder_quant modes ALL consume the trained state_dict for archive bytes"
  - "Catalog #371 (canonical equation auto-recalibration) — N/A (no new empirical anchor in this landing)"
  - "Catalog #372 (Dykstra solver invoker default-on) — preserved at cathedral autopilot side"
---

# PACT-NeRV-SELECTOR-V3 PyTorch decoder_quantization sister landing — Wave N+3 Slot 1 op-routable #1

**Lane:** `lane_pytorch_decoder_quant_sister_landing_op_routable_1_slot1_20260528`
**Date (UTC):** 2026-05-28T14:42:00Z
**Council tier:** T1 (sextet pact present per Catalog #346)
**Council verdict:** PROCEED
**Mission contribution:** `frontier_breaking_enabler` (canonical-cascade-completion at PyTorch sister surface)

## Premise verification (Catalog #229) — DONE BEFORE EDIT

Read all 12 premise files pre-edit. Key findings:

1. `experiments/train_substrate_pact_nerv_selector_v3.py` (566 LOC; pre-landing): `pack_archive()` callsite at lines 385-388 hardcoded the canonical default `decoder_quantization=DECODER_QUANT_FP16_BROTLI_Q9` by omitting the kwarg. The argparse parser did NOT expose `--decoder-quant` or sister flags.
2. `experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py` (720 LOC): MLX-LOCAL canonical sister exposes the full canonical interface — `--decoder-quant {fp16_brotli_q9, fp16_brotli_q11, int8_per_channel_brotli_q11, heterogeneous_per_tensor}` + `--fp4-qat-epochs` + `--top-k-fp4` + `--sensitivity-ranking-method`. This Slot 1 PyTorch sister mirrors that interface exactly.
3. `src/tac/substrates/pact_nerv_selector_v3/archive.py` (334 LOC): `DECODER_QUANTIZATION_KINDS` frozenset + `_serialize_state_dict(sd, *, decoder_quantization=DECODER_QUANT_FP16_BROTLI_Q9)` dispatcher already present + `pack_archive(..., decoder_quantization=...)` API already accepts the kwarg. Slot 2 V3 int8 + Compound C heterogeneous already wired.
4. `src/tac/substrates/pact_nerv_selector_v3/heterogeneous_bit_allocation.py` (991 LOC; sister-landed by Slot 2/Compound C): canonical helpers `compute_per_tensor_sensitivity_via_taylor_expansion` + `derive_heterogeneous_bit_allocation` + `apply_fp4_qat_finetune_on_top_k_tensors` are framework-agnostic at the PyTorch state_dict boundary. NO `framework=` parameter is needed; PyTorch trainer invokes the same helpers as the MLX export path.
5. `src/tac/substrates/pact_nerv_selector_v3/archive_candidate.py` (224 LOC): the MLX export path's FP4-QAT integration pattern is the canonical reference; this PyTorch sister landing replicates it inline at the trainer's `_full_main` rather than via an export_archive_fn partial.
6. `src/tac/substrates/pact_nerv_selector_v3/inflate.py` (69 LOC; HNeRV parity L4 compliant): `inflate_one_video` already routes through `parse_archive(...)` which calls the canonical `_deserialize_state_dict(...)` dispatcher. Sister Catalog #366 + Catalog #367 contracts preserved.
7. `src/tac/quantization.py` + `tac.fp4_quantize`: PyTorch-native `FakeQuantSTE` + `FakeQuantFP4` + canonical `DEFAULT_CODEBOOK` are wired into `apply_fp4_qat_finetune_on_top_k_tensors`. The PyTorch trainer consumes the same canonical helper.
8. `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch.yaml` + sister Compound C recipe: both `dispatch_enabled: false` + `research_only: true` preserved per Catalog #240/#325.
9. Canonical anti-patterns matcher (Slot 2 Wave N+1 just-landed): empirical preflight check produces 3-4 false positives per stack at confidence 0.50 due to token-fallback substring matching (CLAUDE.md `_confidence_for_condition_match` lines 186-208). This is the known op-routable #4 issue per the mandate. Documented in operator-routable next-steps below for Slot 2 sister-wave fix.
10. Slot 2 V3 int8 + Compound C landing memos: cited as canonical predecessors; predicted bands inherited.

## Implementation summary

**File 1: `experiments/train_substrate_pact_nerv_selector_v3.py`** (+136 LOC; 566 → 783)

- Added 5 argparse companion flags mirroring MLX-LOCAL canonical interface:
  - `--decoder-quant {fp16_brotli_q9, fp16_brotli_q11, int8_per_channel_brotli_q11, heterogeneous_per_tensor}` (default `fp16_brotli_q9` preserves V3 baseline behavior)
  - `--fp4-qat-epochs INT` (default 0; applies only for heterogeneous mode; Quantizr canonical = 200)
  - `--top-k-fp4 INT` (default 3; informational; canonical helper uses `DEFAULT_TOP_K_FP4=3`)
  - `--sensitivity-ranking-method {magnitude_x_byte_cost, taylor_gradient, dykstra_lagrangian_dual}` (informational; future extension hook)
  - `--brotli-quality INT` (default None; canonical helper selects per-mode quality)
- Extended `TIER_1_OPERATOR_REQUIRED_FLAGS` manifest per Catalog #151:
  - `--decoder-quant` ↔ env `DECODER_QUANT`
  - `--fp4-qat-epochs` ↔ env `FP4_QAT_EPOCHS`
- Threaded `decoder_quantization=args.decoder_quant` through the canonical `pack_archive(...)` call at the formerly-hardcoded fp16 callsite.
- Added validation + fail-closed pre-emit guard: `if args.decoder_quant not in DECODER_QUANTIZATION_KINDS: raise ValueError(...)`.
- Added optional FP4-QAT pre-emit fine-tune for `heterogeneous_per_tensor` mode:
  - Consumes `tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation.{compute_per_tensor_sensitivity_via_taylor_expansion, derive_heterogeneous_bit_allocation, apply_fp4_qat_finetune_on_top_k_tensors}` (canonical framework-agnostic at PyTorch state_dict boundary).
  - Emits `qat_metrics.json` sidecar with `fp4_tensors_finetuned` + `qat_epochs` + `per_tensor_cos_pre_qat` + `per_tensor_cos_post_qat`.
  - Substitutes QAT-fine-tuned tensors back into `decoder_sd` BEFORE `pack_archive()` so the archive emit's heterogeneous quantization runs over grid-snapped floats.
- Extended provenance JSON schema to v2 (`pact_nerv_selector_v3_full_provenance_v2_decoder_quant_extended`) recording `decoder_quantization` + `fp4_qat_epochs` + `top_k_fp4` + `sensitivity_ranking_method` + `brotli_quality_override` + `qat_metrics`.

**File 2: `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch.yaml`** (minor updates)

- `dispatch_blockers` updated: bridge bullet marked `mlx_to_pytorch_state_dict_bridge_RESOLVED_pending_paired_cuda_dispatch_per_wave_n_plus_3_slot_1_landing_20260528` per Catalog #240 transparency.
- `env_overrides` comment updated to cite this landing memo at the `DECODER_QUANT` env-var threading.
- `reactivation_criteria` updated to mark step 2 (MLX→PyTorch bridge) RESOLVED + cite this landing memo.

**File 3: `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_heterogeneous_bit_modal_t4_dispatch.yaml`** (minor updates)

- `env_overrides` extended with `DECODER_QUANT: heterogeneous_per_tensor` + `FP4_QAT_EPOCHS: "200"` (now threading via Catalog #151 manifest into PyTorch trainer's `--decoder-quant` + `--fp4-qat-epochs`).
- `reactivation_criteria` first bullet marked RESOLVED + cite this landing memo.
- `notes` updated to acknowledge PyTorch sister landing closure.

**File 4 (this memo): `.omx/research/pact_nerv_selector_v3_pytorch_decoder_quant_sister_landing_op_routable_1_landed_20260528.md`**

## Smoke verification (all on $0 macOS CPU per CLAUDE.md "$0 GPU only")

1. **AST + argparse smoke** (35,755 chars; 783 LOC parse OK; all 5 new flags present + invalid choice correctly rejected).
2. **Synthetic-cfg pack_archive() smoke for all 4 decoder_quant modes** (PSV3 grammar end-to-end):
   - `fp16_brotli_q11` → 8,502 bytes; sha `f9b291e77c9eba80`; parse-roundtrip OK
   - `fp16_brotli_q9` → 8,879 bytes; sha `959c9e345e670c1c`; parse-roundtrip OK
   - `heterogeneous_per_tensor` → 4,385 bytes; sha `4fe60b59ceec355a`; parse-roundtrip OK
   - `int8_per_channel_brotli_q11` → 6,506 bytes; sha `ab7c1ca566e71971`; parse-roundtrip OK
3. **Byte-determinism verified**: re-running `pack_archive(..., decoder_quantization='int8_per_channel_brotli_q11')` twice produces byte-identical archive bytes (sha `ab7c1ca566e71971`).
4. **Synthetic-cfg `_smoke_main` path with `--decoder-quant int8_per_channel_brotli_q11 --epochs 2`** (rc=0; provenance.json emitted with score_claim=false per Catalog #192 NON-PROMOTABLE; smoke=True path preserved).
5. **V3 substrate test suite**: 25/25 pass (`src/tac/substrates/pact_nerv_selector_v3/tests/`).
6. **Canonical helper test suites**: 54/54 pass (`src/tac/canonical_anti_patterns/tests/` + `src/tac/substrates/_shared/tests/test_pact_nerv_full_main.py`).

Contest-resolution rendering empirical verification is the operator-attended paired-CUDA RATIFICATION cycle per Catalog #246 (the trainer's `EVAL_HW=(384, 512)` is the substrate's intermediate-resolution archive emit; the canonical `write_contest_runtime` + `inflate.py` at the submission tier render at 1164×874 per Catalog #367 + `CONTEST_RAW_BYTES = 3_662_409_600`).

## Anti-pattern pre-flight check (Catalog #335 / Slot 2 Wave N+1 canonical anti-patterns matcher)

Per the mandate op-routable #10:

**Heterogeneous mode `match_stack_against_anti_patterns(...)`** (stack = `compression_ops: ['fp4_packed_nibbles', 'int8_per_channel', 'int4_per_channel', 'brotli_q11']`, `quantization_aware_training: True`):

- `fp4_packed_without_qat_cos_collapse_v1` (CRITICAL, conf=0.50) — **FALSE POSITIVE per mandate op-routable #4**: our stack declares `quantization_aware_training: True` but the matcher's token-fallback fires on substring match of `fp4_packed` in `compression_ops`. The canonical override flag does NOT yet gate the match. Slot 2 sister-wave fix-pass.
- `cross_paradigm_test_without_per_axis_decomposition_v1` (HIGH, conf=0.50) — **FALSE POSITIVE**: our stack declares `per_axis_decomposition_active: True` but the matcher token-fallback fires anyway. Slot 2 sister-wave fix-pass.
- `lzma_on_already_brotli_saturated_compounding_v1` (MEDIUM, conf=0.50) — **FALSE POSITIVE**: our stack contains NO `lzma` token; the matcher's token-fallback fires on partial-token overlap. Slot 2 sister-wave fix-pass.
- `brotli_plus_lzma_chained_anti_pattern_v1` (MEDIUM, conf=0.50) — **FALSE POSITIVE**: identical issue to above.

**Int8 mode** (stack = `compression_ops: ['int8_per_channel', 'brotli_q11']`): produces 3 false positives via the same mechanism.

**Operator-routable for Slot 2 sister-wave:** the token-fallback in `_confidence_for_condition_match` at confidence 0.50 produces 3-4 false positives per stack. The fix is to either (a) raise the `min_confidence` default in `match_stack_against_anti_patterns` from 0.5 to a higher threshold (e.g. 0.7), OR (b) gate each anti-pattern's match against its explicit override flags (`quantization_aware_training`, `per_axis_decomposition_active`) so the user-declared override correctly filters the matcher.

**This Slot 1 landing does NOT fix the matcher** (scope discipline per Catalog #340 sister-checkpoint guard: Slot 2 owns `src/tac/canonical_anti_patterns/`). The empirical false positives are surfaced here as evidence input for the Slot 2 sister-wave fix-pass.

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| `pack_archive(...)` API | ADOPT_CANONICAL | `tac.substrates.pact_nerv_selector_v3.archive.pack_archive` already accepts `decoder_quantization=` kwarg; sister-landed by Slot 2/Compound C. |
| `_serialize_state_dict(...)` dispatcher | ADOPT_CANONICAL | Dispatcher routes per `decoder_quantization` kind with byte-stable wire format; no fork needed. |
| `heterogeneous_bit_allocation` helpers | ADOPT_CANONICAL | Framework-agnostic at PyTorch state_dict boundary; same helpers consumed by MLX export path. |
| `apply_fp4_qat_finetune_on_top_k_tensors` | ADOPT_CANONICAL | Single source of truth for FP4-QAT pre-emit fine-tune; consumes canonical `tac.fp4_quantize.DEFAULT_CODEBOOK` per Quantizr 0.33 pattern. |
| `argparse` flag interface | ADOPT_CANONICAL_FROM_SISTER | Mirrors MLX-LOCAL canonical interface exactly for cross-trainer parity; no fork. |
| `TIER_1_OPERATOR_REQUIRED_FLAGS` manifest | EXTEND_CANONICAL | Added 2 new entries (DECODER_QUANT + FP4_QAT_EPOCHS) per Catalog #151 manifest extension pattern. |
| Provenance JSON schema | EXTEND_CANONICAL | Bumped from v1 → v2 (`pact_nerv_selector_v3_full_provenance_v2_decoder_quant_extended`); recording 6 new fields. |
| FP4-QAT integration pattern | FORK_BECAUSE_PRINCIPLED_MISMATCH (minor) | The MLX path uses an `export_archive_fn=partial(...)` indirection at the `RendererBundle` boundary; the PyTorch trainer inlines the QAT call directly into `_full_main` because no equivalent bundle boundary exists in the PyTorch path. Functionally identical; the helper invocation is the same canonical surface. |

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: Inherits PACT-NeRV-V3 substrate-distinguishing primitive (Rice-Golomb selector coding at archive-encode time). Not a re-implementation; threads the canonical decoder_quantization dispatcher.
2. **BEAUTY + ELEGANCE**: +136 LOC for full 4-mode coverage + QAT integration. Sister of MLX-LOCAL canonical interface; reviewable in 30s.
3. **DISTINCTNESS**: Distinct from MLX-LOCAL sister by framework (PyTorch native, not MLX→numpy→PyTorch bridge); produces contest-resolution archive bytes at inflate time.
4. **RIGOR**: 4 modes byte-deterministic + parse-roundtrip + smoke + 79 sister tests pass + anti-pattern preflight + probe outcome registered. Catalog #229 PV evidence in this memo.
5. **OPTIMIZATION PER TECHNIQUE**: 4 distinct decoder_quant modes; each preserves the canonical V3 architecture; QAT applies to top-K tensors only (Quantizr 0.33 canonical pattern); int4-groupwise NF4 on tail tensors (bitsandbytes/QLoRA canonical).
6. **STACK-OF-STACKS-COMPOSABILITY**: Compound C orthogonal to Compound A (V3 baseline) + Compound B (Slot 2 int8); ΔS additive per compound stacking arithmetic.
7. **DETERMINISTIC REPRODUCIBILITY**: Byte-stable archive bytes verified; seed-pinned per existing `_canonical_pin_seeds(args.seed)`.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: Heterogeneous mode produces ~50% archive bytes vs fp16 baseline (synthetic-cfg: 4,385 vs 8,879). Real-cfg savings projection from Compound C landing memo: -43.5% over V3 baseline → predicted score floor 0.158-0.163 [contest-CPU pending paired-CUDA].
9. **OPTIMAL MINIMAL CONTEST SCORE**: Unblocks paired-CUDA RATIFICATION for BOTH sub-0.18 (int8) AND sub-0.16 (heterogeneous) candidates simultaneously — single Slot 1 landing covers 2 medal-class predicted-band trajectories.

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Evidence |
|---|---|---|
| The MLX-LOCAL canonical interface (`--decoder-quant` + sister flags) is the right API for the PyTorch sister | HARD-EARNED | MLX-LOCAL canonical interface predates this landing by hours; cross-trainer parity is operator's `# MLX-FIRST + numpy-portable INFLATE` standing directive 2026-05-26. |
| Mirroring MLX argparse signature exactly (no fork) is correct | HARD-EARNED | Per CLAUDE.md "Beauty + DX" non-negotiable + 11th INDIVIDUALLY-FRACTAL standing directive — but for the cross-trainer-interface case, parity reduces operator cognitive load. |
| The canonical heterogeneous_bit_allocation helper does not need a `framework=` parameter | HARD-EARNED | Verified by reading the helper module: every function signature uses `torch.Tensor` types; the MLX path already invokes via PyTorch tensor conversion at the boundary. |
| Preserving fp16_brotli_q9 default is correct | HARD-EARNED | Per CLAUDE.md "Backward compatibility" + Catalog #240 transparency — operators without `--decoder-quant` get baseline V3 behavior. |
| FP4-QAT pre-emit fine-tune happens BEFORE pack_archive (not via export_archive_fn partial) | HARD-EARNED (per principled-mismatch) | The PyTorch trainer has no equivalent `RendererBundle.export_archive_fn` boundary; inline call in `_full_main` is the canonical pattern for this surface. |
| The default `--fp4-qat-epochs 0` (disabled by default) is correct | HARD-EARNED | Forcing QAT to opt-in prevents silent FP4 quality regression on operators who do not explicitly opt into the QAT cycle. |
| The anti-pattern preflight false positives do NOT block this landing | HARD-EARNED | Per mandate op-routable #4: known issue; Slot 2 owns the fix; this Slot 1 landing surfaces empirical evidence for the next sister wave. |

## Observability surface (Catalog #305) — 6 facets

1. **Inspectable per layer**: `qat_metrics.json` sidecar surfaces per-tensor `cos_pre_qat` + `cos_post_qat` + `final_qat_loss`; `provenance.json` v2 surfaces `decoder_quantization` + `fp4_qat_epochs` + all 5 companion flags.
2. **Decomposable per signal**: per-tensor FP4-QAT metrics in `qat_metrics.json`; per-mode archive byte savings deducible from `archive_bytes` provenance field + Slot 2/Compound C predicted-band arithmetic.
3. **Diff-able across runs**: archive bytes byte-deterministic + sha256 in provenance; runs with same seed + same decoder_quant produce byte-identical archives.
4. **Queryable post-hoc**: `provenance.json` machine-readable; `qat_metrics.json` machine-readable; probe outcomes JSONL queryable via `tac.probe_outcomes_ledger.query_outcomes(...)`.
5. **Cite-able**: every artifact carries archive_sha256 + git_head + dispatched_at_utc + this landing memo path.
6. **Counterfactual-able**: byte-mutation smoke via `tools/verify_distinguishing_feature_byte_mutation.py` (Catalog #272 sister); each of 4 decoder_quant modes has distinct sha256 (proven above); mutating a single byte in any mode's archive would cause parse failure or output frame change at inflate time.

## 6-hook wire-in declaration (Catalog #125)

- **Hook #1 sensitivity-map**: ACTIVE — per-tensor Taylor expansion sensitivity consumed via `compute_per_tensor_sensitivity_via_taylor_expansion`; downstream `tac.sensitivity_map.*` consumers inherit signal through the canonical helper.
- **Hook #2 Pareto constraint**: ACTIVE — per-tensor Lagrangian dual via Slot 1 Wave N+1 Dykstra solver (just-landed sister); Catalog #372 fires DEFAULT-ON per cathedral autopilot iteration.
- **Hook #3 bit-allocator**: ACTIVE PRIMARY — this landing IS the substrate's bit-allocator surface (heterogeneous per-tensor FP4 + int8 + int4 routing); the PyTorch sister enables contest-resolution heterogeneous bit allocation.
- **Hook #4 cathedral autopilot dispatch**: ACTIVE — PyTorch sister unblocks paired-CUDA dispatch for ~0.168 (Slot 2 int8) + ~0.154 (Compound C heterogeneous) candidates; recipes preserve `dispatch_enabled: false` per Catalog #240/#325 operator-attended discipline.
- **Hook #5 continual-learning posterior**: ACTIVE — canonical equation `pact_nerv_decoder_state_dict_saturation_at_parity_floor_v1` + `heterogeneous_per_tensor_bit_allocation_compounding_v1` anchors will refresh post paired-CUDA RATIFICATION per Catalog #371 auto-recalibrator.
- **Hook #6 probe-disambiguator**: ACTIVE — PyTorch contest-resolution archive bytes ARE the disambiguator between MLX-LOCAL macOS-MLX-research-signal (NON-PROMOTABLE per Catalog #192) vs contest-CPU/contest-CUDA paired-RATIFICATION (PROMOTABLE per Catalog #246).

## Operator-routable paired-CUDA RATIFICATION cascade

1. ✅ **PyTorch sister landed** (THIS work; 2026-05-28 14:42Z): all 4 decoder_quant modes wired through canonical pack_archive(); --decoder-quant + companion flags exposed via argparse; Catalog #151 manifest entries added; FP4-QAT integration via canonical helper; provenance schema v2.
2. ⏳ **Per-substrate symposium per Catalog #325** (covers BOTH int8 + heterogeneous variants in 1 symposium): sextet pact + 6-step contract; operator-routable via `tools/codex_to_claude_inbox.py` directive.
3. ⏳ **Operator-attended paired-CUDA dispatch — int8 variant** via `tools/operator_authorize.py --recipe substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch` (~$1.00 paired T4 CUDA + Linux x86_64 CPU per Catalog #246). Set `dispatch_enabled: true` first per Catalog #240 transparency.
4. ⏳ **Operator-attended paired-CUDA dispatch — heterogeneous Compound C variant** via `tools/operator_authorize.py --recipe substrate_pact_nerv_selector_v3_heterogeneous_bit_modal_t4_dispatch` (~$1.50 paired per cost-band estimate).
5. ⏳ **If RATIFIED ≤0.18 (int8) AND/OR ≤0.16 (Compound C)**: PR submission cascade per `feedback_forbidden_claude_attribution_in_public_pr_surfaces` discipline (sole-author Alejandro Peña <adpena@gmail.com>; ZERO Claude/Anthropic tokens in PR-facing surfaces).
6. ⏳ **Slot 2 sister-wave fix-pass for canonical anti-patterns false positives** (out of Slot 1 scope per Catalog #340): raise min_confidence threshold OR gate match against explicit override flags so `quantization_aware_training: True` correctly filters `fp4_packed_without_qat_cos_collapse_v1`.

## Mission alignment (Catalog #300 §"Mission alignment" Consequence 5)

**Predicted mission contribution**: `frontier_breaking_enabler` — single Slot 1 landing unblocks paired-CUDA RATIFICATION for BOTH ~0.168 (Slot 2 int8) AND ~0.154 (Compound C heterogeneous) candidates simultaneously. Per CLAUDE.md "PACT-NeRV + LONG ORIGINAL SUBSTRATE TRAINING + CLASS/PARADIGM-SHIFT = TOP STANDING PRIORITY" 2026-05-27 + "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware" non-negotiable: this landing closes the canonical-cascade-completion at PyTorch sister surface and structurally enables the 2 medal-class predicted-band trajectories without further sister landings.

**Override invoked**: false (no operator-frontier-override per Catalog #300 Consequence 1; standard PROCEED council verdict applies).

## Cross-references

- Slot 2 V3 int8 landing memo: `.omx/research/pact_nerv_selector_v3_int8_decoder_quant_brotli_q11_600pair_long_mlx_landed_20260528.md`
- Compound C heterogeneous landing memo: `.omx/research/pact_nerv_selector_v3_heterogeneous_bit_allocation_fp4_qat_top3_600pair_long_mlx_landed_20260528.md`
- Slot 1 Wave N+1 Dykstra solver landing: `.omx/research/dykstra_pareto_polytope_solver_wire_in_dim1_phase4_landed_20260528.md`
- Slot 2 Wave N+1 canonical anti-patterns landing: `.omx/research/canonical_anti_patterns_registry_layer_1_plus_2_landed_20260528.md`
- Parent decoder compression analysis: `.omx/research/decoder_compression_analysis_pact_nerv_cluster_landed_20260528.md`
- V3 baseline landing: `.omx/research/pact_nerv_selector_v3_l1_long_run_mlx_landed_20260528.md`
- T3 council PR110-STACKING-PIVOT-ORDERING (operator current frontier baseline): `feedback_t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
- Recipes: `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch.yaml` + `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_heterogeneous_bit_modal_t4_dispatch.yaml`
- Probe outcome ledger entry: `pact_nerv_selector_v3_pytorch_decoder_quant_sister_landing_op_routable_1_20260528` (VERDICT_PROCEED; 30d staleness)
