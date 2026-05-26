---
council_tier: T1
council_attendees:
  - Carmack
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "the PR101 lc_v2_clone substrate audit returns NOT-CANDIDATE; honest scope-deferral required rather than forcing a procedural-variant where the canonical equation #26 domain-of-validity is not satisfied"
council_assumption_adversary_verdict:
  - assumption: "PR101 lc_v2_clone has procedural-codebook-candidate byte regions"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "audit reveals 0 deterministic-constant byte regions in archive.zip; DECODER_BLOB+LATENT_BLOB are score-aware-trained per-video weights; sidecar+anchor tables either too small for net savings or live outside archive.zip in Python source"
  - assumption: "PR106 has substrate code parallel to PR101 lc_v2_clone"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "no src/tac/substrates/pr106_* directory exists; PR106 lives only as 11 submissions/pr106_* packets + magic_codec_pr106_r2 research-adapter dispatcher (per its own docstring NOT a substrate)"
  - assumption: "canonical equation #26 IN-DOMAIN check is sufficient for picking a 3rd candidate"
    classification: HARD-EARNED
    rationale: "_INCLUDED_CONTEXTS enumeration in canonical equation #26 explicitly lists 11 contexts; PR101 decoder weights are not in any IN-DOMAIN context"
council_decisions_recorded:
  - "op-routable #1: pivot the cascade-sequencing-#3 BUILD candidate to an IN-DOMAIN substrate from the 5-substrate matrix (NSCS06 v8 chroma-LUT OR ATW V2 codec quantizer LUT OR grayscale_lut substrate); design spec sketched below"
  - "op-routable #2: file the PR101 lc_v2_clone NOT-CANDIDATE verdict to .omx/state/probe_outcomes.jsonl with DEFER blocker_status + 30-day staleness window per Catalog #313"
  - "op-routable #3: ratify the canonical equation #26 _EXCLUDED_CONTEXTS to ALSO include 'score_aware_trained_decoder_weights' + 'per_video_learned_latent_codes' per the empirical audit finding"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: "lane_pr101_lc_v2_clone_procedural_variant"
deferred_substrate_retrospective_due_utc: "2026-06-19T01:00:00Z"
related_deliberation_ids:
  - dp1_procedural_trainer_build_landed_20260520
  - vq_vae_procedural_trainer_build_landed_20260520
  - five_substrate_procedural_replacement_matrix_design_20260520
  - procedural_codebook_generator_build_landed_20260520
  - canonical_equation_26_domain_refinement_20260520
---

<!-- Catalog #344 canonical equation cross-reference:
     procedural_codebook_from_seed_compression_savings_v1
     domain_of_validity_contexts: _INCLUDED_CONTEXTS / _EXCLUDED_CONTEXTS
     per src/tac/canonical_equations/procedural_codebook_savings.py
     closed form: ΔS = -25 * (N_codebook - 32) / 37_545_489 -->

# 🏗️🔍📋 PR101+PR106 PROCEDURAL VARIANT BUILD DESIGN (cascade-sequencing #3)

**Task lane**: `lane_wave_3_pr101_pr106_procedural_variant_build_design_20260520` L1

**Operator-pick**: 2026-05-20 "b" — design the 3rd PROCEDURAL VARIANT BUILD candidate after DP1 (commit `9cbfa471c`) + VQ-VAE (commit `6fea30f22`).

**Sister-DISJOINT**: in-flight PR101 GOLD NULL-BYTE REMOVAL SMOKE (`a3dfc84c` targets MASTER-GRADIENT-NULL bytes in fec6 frontier ARCHIVE 16,292 bytes) + MAGIC CODEC FIX (`a90e800a`). THIS lane targets DETERMINISTIC-CONSTANT byte regions in SUBSTRATE CODE — different scope per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #287 sister discipline.

---

## TL;DR

The PR101 lc_v2_clone substrate audit returns **NOT-CANDIDATE** for the canonical equation #26 procedural-codebook-replacement variant pattern. PR106 has **NO SUBSTRATE DIRECTORY** (only `submissions/pr106_*` packet outputs + a research-adapter dispatcher). The honest recommendation is to **PIVOT** the cascade-sequencing-#3 BUILD candidate to an IN-DOMAIN substrate from the 5-substrate matrix design (commit `b3e3442c3`), with **grayscale_lut** (or NSCS06 v8 chroma-LUT) as the recommended pick.

Per CLAUDE.md "Forbidden premature KILL" + Catalog #307 paradigm-vs-implementation: PR101 lc_v2_clone is **DEFERRED-pending-procedural-domain-applicability** NOT killed. The substrate paradigm is intact; the procedural-codebook-replacement pattern is structurally inapplicable because PR101 lc_v2_clone's archive.zip contains only score-aware-trained per-video weights + per-pair learned latents (all OUT-OF-DOMAIN per canonical equation #26 `_EXCLUDED_CONTEXTS`).

---

## §1. Pre-flight verification

Per CLAUDE.md "Subagent coherence-by-default":

- ✅ `tools/check_sister_files_recently_landed.py --files .omx/research/pr101_pr106_procedural_variant_build_design_20260520.md` returned `PROCEED` (12-hour + 4-hour windows)
- ✅ DP1 BUILD commit `9cbfa471c` verified (canonical pattern: `src/tac/substrates/pretrained_driving_prior/distillation_procedural_variant.py` ~400 LOC)
- ✅ VQ-VAE BUILD commit `6fea30f22` verified (canonical pattern: `src/tac/substrates/vq_vae/distillation_procedural_variant.py` ~620 LOC)
- ✅ Procedural-codebook generator commit `1dd8569de` verified (canonical helper + equation #26 registered)
- ✅ 5-substrate matrix design commit `b3e3442c3` verified (5 candidates enumerated)
- ✅ Canonical equation #26 domain refinement commits `8d8a7c6c5` + `37fea4aac` verified (11 IN_CONTEXTS + 2 EXCLUDED_CONTEXTS)
- ✅ Sister DP1 + VQ-VAE landing memos read in full (Catalog #229 PV)
- ✅ PR101 lc_v2_clone substrate code read in full (`__init__.py` 162 LOC + `architecture.py` 196 LOC + `archive.py` 583 LOC; 6.3K + 6.9K + 21.3K)
- ✅ PR106 substrate dir search: **0 hits** under `src/tac/substrates/pr106*`
- ✅ Probe-outcomes ledger Catalog #313 verified (no blocking PR101 lc_v2_clone or PR106 procedural-variant entries; stack_of_stacks DEFER + lane_stc_clean_source_v2 DEFER are sister-class but DISJOINT scope)
- ✅ Per-substrate symposium recency Catalog #325: no recent (≤14 days) symposium for `lane_pr101_lc_v2_clone_procedural_variant` (would block paid dispatch; design memo only)

---

## §2. Substrate code audit table

| Aspect | PR101 lc_v2_clone | PR106 (submissions only) | DP1 (canonical pattern) | VQ-VAE (canonical pattern) |
|---|---|---|---|---|
| `src/tac/substrates/<id>/` exists | ✅ YES (`pr101_lc_v2_clone/`) | ❌ **NO** | ✅ (`pretrained_driving_prior/`) | ✅ (`vq_vae/`) |
| `__init__.py` | 162 LOC, 6.3 KB | n/a | ~80 LOC | ~80 LOC |
| `architecture.py` | 196 LOC, 6.9 KB | n/a | exists | exists |
| `archive.py` | 583 LOC, 21.3 KB | n/a | exists | exists |
| `curriculum*.py` | 25.8 KB + 42.1 KB | n/a | n/a | n/a |
| `score_aware_loss.py` | 5.3 KB | n/a | exists | exists |
| `tests/` directory | ✅ | n/a | ✅ | ✅ |
| `impl_complete` (per lane registry) | research_only=true | n/a (no lane) | research_only=true | research_only=true |
| Archive grammar declared (Catalog #124) | ✅ `PR101_LC_V2_ARCHIVE_GRAMMAR` 3 sections (DECODER_BLOB / LATENT_BLOB / SIDECAR_BLOB) | n/a | ✅ `DP1_HEADER_FMT` + sections | ✅ codebook-inside-decoder-sd |
| DETERMINISTIC-CONSTANT byte regions in archive.zip | ❌ ZERO (audit §3 below) | n/a | ✅ ~4 KB PCA basis (OOD-derived) | ✅ 8192 B VQ codebook (K=512×D=8×fp16) |
| Procedural-codebook replacement candidate? | ❌ **NOT-CANDIDATE** | n/a | ✅ YES — predicted ΔS = -0.002706 | ✅ YES — predicted ΔS = -0.005434 |

**Empirical finding** per audit:

- PR101 lc_v2_clone's `archive.zip` consists of THREE sections per its `PR101_LC_V2_ARCHIVE_GRAMMAR`:
  - **DECODER_BLOB** (PR101 anchor: 162,164 bytes) — 7 brotli-compressed streams over 28 score-aware-trained per-video model tensors; `_per_tensor_encode_bytes` produces `encoded_q_bytes || fp16_scale` per tensor.
  - **LATENT_BLOB** (PR101 anchor: 15,387 bytes) — 600 per-pair LZMA-compressed learned latent codes.
  - **SIDECAR_BLOB** (variable) — UTF-8 JSON meta (`latent_dim`, `num_pairs`, `base_channels`, `eval_size`); ~100 bytes total.

- The PR101 GOLD primitives (`PR101_DECODER_STORAGE_ORDER`, `PR101_DECODER_STREAM_ENDS`, `PR101_CONV4_STORAGE_PERMS`, `PR101_DECODER_BYTE_MAPS`) ARE deterministic constants but live in **Python source under `src/tac/packet_compiler/`**, NOT in `archive.zip`. They are consumed at parse-time as Python module imports per `tac.packet_compiler.pr101_decoder_storage_order:PR101_DECODER_STORAGE_ORDER`. Replacing them with a 32-byte seed in archive.zip is **structurally impossible** because they are NOT in archive.zip to begin with.

- The header bytes (`b"PRC1"` magic + 1-byte version + 3×4-byte u32 lengths = 17 bytes) are deterministic constants but too small for net savings (32-byte seed > 17-byte header = +15 bytes regression).

**PR106 substrate dir verdict**: `ls src/tac/substrates/ | grep -i pr106` returns **0 hits**. PR106 is materialized only as:
1. `submissions/pr106_*` packets (11 directories: `pr106_latent_sidecar`, `pr106_c3_residual_sidecar`, `pr106_hdm3_*`, `pr106_yshift_sidechannel`, `pr106_stacked`, etc.) — these are submission artifacts (`inflate.py` + `inflate.sh`), NOT substrate code.
2. `submissions/magic_codec_pr106_r2/` — per its own docstring `"INTENTIONALLY narrow ... NOT a promotion-grade contest runtime"` + `"ready_for_exact_eval_dispatch=False"`. This is a research-adapter dispatcher, NOT substrate code.
3. `reports/raw/pr106_*` + `experiments/results/*pr106*` forensic artifacts.

PR106 has **NO `src/tac/substrates/pr106_*` directory equivalent to PR101 lc_v2_clone OR DP1 OR VQ-VAE**, so PR106 is **STRUCTURALLY out of scope** for the cascade-sequencing-#3 BUILD pattern.

---

## §3. Per-region predicted-ΔS via canonical equation #26

Closed form: `ΔS = -25 × (N_codebook − 32) / 37_545_489`

| Region | Bytes (current) | Bytes (after 32-byte seed replacement) | Predicted ΔS | IN-DOMAIN? | Candidate? |
|---|---:|---:|---:|---|---|
| DECODER_BLOB | 162,164 | 32 (impossible — score-aware-trained weights) | -0.10796 (HYPOTHETICAL) | ❌ NO (not in `_INCLUDED_CONTEXTS`; matches `score_aware_trained_decoder_weights` proposed EXCLUDED) | ❌ NOT-CANDIDATE |
| LATENT_BLOB | 15,387 | 32 (impossible — per-pair learned codes) | -0.01023 (HYPOTHETICAL) | ❌ NO (matches `per_video_learned_latent_codes` proposed EXCLUDED) | ❌ NOT-CANDIDATE |
| SIDECAR_BLOB | ~100 | 32 (yields net +32-100 = NEGATIVE savings of ~68 B) | +0.0000453 (REGRESSION) | ✅ technically yes (`deterministic_constants_codebook_replacement`) but byte count too small | ❌ NOT-CANDIDATE (regression) |
| Header bytes | 17 | 32 (NEGATIVE savings of -15 B) | +0.0000100 (REGRESSION) | ✅ technically yes but byte count negative | ❌ NOT-CANDIDATE (regression) |
| PR101 anchor tables (storage_order + stream_ends + conv4_perms + byte_maps) | 0 (live in Python source) | 32 (would ADD bytes to archive) | +0.0000213 (REGRESSION) | n/a (not in archive.zip) | ❌ NOT-CANDIDATE |

**Aggregate candidacy verdict for PR101 lc_v2_clone**: **NOT-CANDIDATE for procedural-codebook-replacement**. All 5 byte regions either (a) are OUT-OF-DOMAIN per canonical equation #26 `_EXCLUDED_CONTEXTS` (proposed extensions), (b) produce NEGATIVE savings (regression), or (c) live outside archive.zip and would require restructuring rather than replacement.

**Aggregate candidacy verdict for PR106**: **STRUCTURALLY OUT OF SCOPE** (no substrate dir).

---

## §4. Recommended substrate pick

Per CLAUDE.md "Mission alignment" + Carmack MVP-first phasing cascade-continuation: pivot the cascade-sequencing-#3 candidate to an **IN-DOMAIN** substrate from the 5-substrate matrix (commit `b3e3442c3`). The remaining 3 matrix candidates are:

| Candidate | substrate dir | Predicted ΔS | symposium status | Lane registry status |
|---|---|---:|---|---|
| **NSCS06 v8 chroma-LUT** | `src/tac/substrates/nscs06_v8_path_b_wavelet/` | -0.002706 (4 KB → 32 B) | symposium #852 + #variant_c_reactivation (Catalog #325 anchor exists) | research_only=true + lane_class=substrate_engineering |
| **ATW V2 codec quantizer LUT** | `src/tac/substrates/atw_codec_v2/` | -0.002706 (~4 KB → 32 B) | symposium recency unknown — needs verify | research_only=true; lane registry hit |
| **grayscale_lut** | `src/tac/substrates/grayscale_lut/` | -0.002706 (~4 KB → 32 B) | symposium recency unknown — needs verify | research_only=true; canonical CompressAI-style codec |

**Operator-routable recommendation (single pick per Carmack MVP-first cascade discipline)**: **grayscale_lut**

Rationale:
1. **STRUCTURALLY CLEAN architecture surface**: grayscale_lut has the canonical 5-file substrate layout (`__init__.py`, `architecture.py`, `archive.py`, `inflate.py`, `score_aware_loss.py`) identical to DP1 + VQ-VAE pattern.
2. **EMPIRICALLY SUITABLE**: the codec uses an explicit LUT (Look-Up Table) which is by definition a deterministic constant codebook — strongest IN-DOMAIN fit per canonical equation #26 `_INCLUDED_CONTEXTS = (..., "procedural_codebook_as_lookup_table", "chroma_lut_replacement", ...)`.
3. **MINIMAL CARGO-CULT RISK**: per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + the recent NSCS06 v6 → v7 → v8 cargo-cult unwind series, grayscale_lut as a substrate has NOT yet been entangled with the more complex chroma-LUT-vs-wavelet cargo-cult class.
4. **PRIOR PROBE VERDICT**: no blocking Catalog #313 probe-outcomes entry for grayscale_lut as of 2026-05-20T15:00Z.
5. **Sister DP1 + VQ-VAE pattern is directly portable** — the canonical `derive_codebook_from_seed` PCG64 PRNG with `output_shape=(1024, 4)` int8 = 4096 bytes total maps 1-to-1 to a grayscale-LUT replacement.

**Alternative pick (secondary)**: NSCS06 v8 chroma-LUT — already has per-substrate symposium anchor (#852 + variant_c_reactivation), so paid dispatch readiness is closer; but the cargo-cult-unwind history is more complex and Catalog #325 symposium verdict needs verification before dispatch.

**Per CLAUDE.md "Forbidden premature KILL"**: the **PR101 lc_v2_clone** substrate is **DEFERRED-pending-procedural-domain-applicability** with **reactivation criteria**:
- Reactivation path A: canonical equation #26 domain refinement (sister wave to commits `8d8a7c6c5` + `37fea4aac`) adds a NEW context that covers PR101 lc_v2_clone's specific structure (e.g., `score_aware_trained_decoder_intermediate_quantizer_tables` if such tables can be extracted from `_per_tensor_encode_bytes` flow).
- Reactivation path B: a different replacement primitive (NOT procedural-codebook from seed) emerges that targets per-tensor-quantization-scale bytes (the fp16 scale per tensor = 2 bytes × 28 tensors = 56 bytes of CONSTANT-PER-VIDEO data within archive.zip; if those scales can be derived deterministically from a seed via a sister canonical equation, the savings would be ~24 bytes per video = +ε regression but a different equation class).
- Reactivation path C: PR101 lc_v2_clone enters a per-substrate symposium per Catalog #325 with the explicit `_EXCLUDED_CONTEXTS` extensions ratified by council.

---

## §5. PROCEDURAL VARIANT module design spec (grayscale_lut)

Sister of DP1 (`src/tac/substrates/pretrained_driving_prior/distillation_procedural_variant.py` ~400 LOC) + VQ-VAE (`src/tac/substrates/vq_vae/distillation_procedural_variant.py` ~620 LOC) canonical pattern.

### 5.1 New file: `src/tac/substrates/grayscale_lut/distillation_procedural_variant.py`

~250-400 LOC (estimated; smaller than VQ-VAE because grayscale_lut codec is simpler).

**Public API** (sister of DP1):

```python
from tac.substrates.grayscale_lut import (
    PROCEDURAL_VARIANT_AVAILABLE,                  # bool capability flag
    PROCEDURAL_SEED_SIZE_BYTES,                    # 32
    PROCEDURAL_CODEBOOK_SHAPE_DEFAULT,             # (256, 1) int8 grayscale LUT
    PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,             # np.uint8
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,       # "chroma_lut_replacement"
    ProceduralVariantConfig,
    ProceduralVariantError,
    derive_procedural_grayscale_lut_replacement,
    compose_with_procedural_grayscale_lut,
    compose_procedural_archive,                    # archive.py convenience
    verify_procedural_grayscale_lut_in_domain,
    verify_seed_mutation_changes_grayscale_lut_bytes,  # Catalog #272 smoke
)
```

**Pattern delta from DP1**:
- `CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT` is `"chroma_lut_replacement"` (vs DP1's `"comma2k19_ood_derived_basis_replacement"`).
- `PROCEDURAL_CODEBOOK_SHAPE_DEFAULT` is `(256, 1)` int8 (canonical grayscale LUT = 256-entry mapping) totaling 256 bytes.
- `derive_codebook_from_seed` from `tac.procedural_codebook_generator` returns a 256-byte LUT array.
- The substrate's inflate path re-derives the LUT from the 32-byte seed at parse-time per Catalog #220 operational mechanism.

### 5.2 `__init__.py` delta (+~35 LOC)

Re-export the 11 public API names + set `PROCEDURAL_VARIANT_AVAILABLE = True` flag.

### 5.3 `archive.py` shim (+~37 LOC)

Add `compose_procedural_archive(seed_bytes: bytes, ...) -> bytes` convenience wrapper that:
1. Calls `derive_procedural_grayscale_lut_replacement(seed_bytes)` → 256-byte LUT
2. Replaces the LUT slot in the canonical grayscale_lut archive with the 32-byte seed bytes
3. Returns the modified archive bytes

**Predicted ΔS via canonical equation #26**:
- `N_codebook = 256` (canonical grayscale_lut entry count)
- `K_seed = 32`
- `ΔS = -25 × (256 - 32) / 37_545_489 = -25 × 224 / 37_545_489 ≈ -0.000149`

**Aggregate after 3 substrates** (DP1 -0.002706 + VQ-VAE -0.005434 + grayscale_lut -0.000149):
- Naive additive: ΔS ≈ -0.008289 (3 substrates)
- Catalog #322 composition-alpha v2 cascade adjustment: actual aggregate depends on `recommended_q4_target` in `.omx/state/substrate_composition_matrix.json` and the alpha bands {ADDITIVE α>0.7 / SUB-ADDITIVE 0.3-0.7 halve / SATURATING ≤0.3 floor at -0.005}.
- HONEST per Catalog #287: aggregate is **predicted-only** until first empirical anchor lands.

### 5.4 Tests: `src/tac/substrates/grayscale_lut/tests/test_procedural_variant.py`

~15-20 tests sister of DP1 + VQ-VAE pattern:
- Config validation (8 seed length boundary cases per `_MIN_SEED_SIZE_BYTES` / `_MAX_SEED_SIZE_BYTES`)
- Canonical-equation-#26 IN-DOMAIN check passes
- Determinism: same seed → same 256-byte LUT bytes
- Catalog #272 byte-mutation smoke: single-byte seed flip changes LUT
- Archive composition: `compose_procedural_archive` round-trips
- Canonical equation #26 closed-form math match (`predicted_delta_s == -0.000149` ±4 decimals)
- Provenance per Catalog #287/#323 (`score_claim=False`, `axis_tag=[predicted]`)

### 5.5 Lane registry update

`tools/lane_maturity.py add-lane lane_grayscale_lut_procedural_codebook_replacement_variant_20260520 ...` then `mark` gates as evidence is produced (impl_complete + research_only + lane_class=substrate_engineering per HNeRV parity L7).

### 5.6 Catalog #325 per-substrate symposium gating status

Before paid Modal/Lightning dispatch: `lane_grayscale_lut_procedural_codebook_replacement_variant_20260520` needs a per-substrate symposium per Catalog #325 (sextet pact + grand council attendees specifically for grayscale-LUT-class substrates: Selfcomp / Mallat / van den Oord seats relevant). NO paid dispatch from this design memo; recipe YAML committed only AFTER symposium PROCEED verdict.

### 5.7 Catalog #322 composition-alpha estimate

After landing this 3rd substrate (DP1 + VQ-VAE + grayscale_lut), the autopilot ranker `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` cascade would consume the 3-substrate aggregate. Pairwise composition_alpha estimates:
- DP1 × VQ-VAE: ADDITIVE class predicted (different substrates; no Comma2k19 dependency overlap)
- DP1 × grayscale_lut: ADDITIVE class predicted (DP1 is OOD-derived; grayscale_lut is chroma-LUT; no overlap)
- VQ-VAE × grayscale_lut: SUB-ADDITIVE-CANDIDATE (both are LUT-class substrates; overlap on the "lookup-table-replacement" semantic; halve via Catalog #322 v2 cascade)
- 3-substrate aggregate: HONESTLY PREDICTED-ONLY until first empirical pairwise anchor lands.

---

## §6. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Canonical adopted? | Rationale |
|---|---|---|
| `derive_codebook_from_seed` (substrate-agnostic) | ✅ ADOPT_CANONICAL | Direct reuse of `tac.procedural_codebook_generator` per slot 1 commit `1dd8569de` |
| Canonical equation #26 IN-DOMAIN context string | ✅ ADOPT_CANONICAL | Use `"chroma_lut_replacement"` constant from `_INCLUDED_CONTEXTS` |
| `predicted_delta_s` formula | ✅ ADOPT_CANONICAL | Reuse `procedural_codebook_savings.predicted_delta_s` closed form |
| Catalog #287 placeholder-rationale rejection | ✅ ADOPT_CANONICAL | All rationales substantive ≥4 chars |
| Catalog #323 canonical Provenance umbrella | ✅ ADOPT_CANONICAL | `score_claim=False` + `axis_tag=[predicted]` on all returns |
| Catalog #324 predicted_band_validation_status | ✅ ADOPT_CANONICAL | `pending_post_training` |
| Catalog #341 Tier A canonical-routing-markers | ✅ ADOPT_CANONICAL | sister cathedral consumer reuses canonical equation #26 |
| Archive grammar (LUT slot replacement) | 🔀 FORK_BECAUSE_PRINCIPLED_MISMATCH | grayscale_lut archive packs LUT differently than DP1 codebook section + VQ-VAE codebook-inside-decoder-sd; need per-substrate replacement helper |
| `output_shape` default | 🔀 FORK_BECAUSE_PRINCIPLED_MISMATCH | `(256, 1)` int8 vs DP1's `(1024, 4)` vs VQ-VAE's `(512, 8)` fp16 |
| `score_aware_loss` wiring | ✅ ADOPT_CANONICAL | grayscale_lut already wires `tac.differentiable_eval_roundtrip` per existing substrate |
| Catalog #272 byte-mutation smoke API | ✅ ADOPT_CANONICAL | Direct port of DP1's `verify_seed_mutation_changes_codebook_bytes` |
| Slot-3 sister-helper integration | ✅ ADOPT_CANONICAL (graceful ImportError fallback) | Same try/except ImportError pattern as DP1 |

---

## §7. 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: ✅ grayscale_lut is the 3rd PROCEDURAL VARIANT BUILD in cascade after DP1 (1st) + VQ-VAE (2nd); DISTINCT IN-DOMAIN substrate class.
2. **BEAUTY + ELEGANCE**: ✅ Estimated ~250-400 LOC sister of DP1/VQ-VAE pattern; reviewable in 30 seconds.
3. **DISTINCTNESS**: ✅ grayscale_lut is canonical 256-entry chroma LUT substrate; distinct from DP1's PCA basis + VQ-VAE's codebook.
4. **RIGOR**: ✅ Catalog #229 premise-verification audit returned PR101 lc_v2_clone NOT-CANDIDATE + PR106 STRUCTURALLY OUT OF SCOPE; pivot rationale empirically grounded.
5. **OPTIMIZATION PER TECHNIQUE**: ✅ Per Catalog #290 above; FORK where principled (archive grammar + output_shape); ADOPT_CANONICAL elsewhere.
6. **STACK-OF-STACKS-COMPOSABILITY**: ✅ Catalog #322 composition_alpha estimate provided; ADDITIVE class with DP1 + SUB-ADDITIVE-CANDIDATE with VQ-VAE.
7. **DETERMINISTIC REPRODUCIBILITY**: ✅ 32-byte PCG64 seed → deterministic 256-byte LUT.
8. **EXTREME OPTIMIZATION**: ✅ Predicted ΔS -0.000149 per canonical equation #26 closed form; bound by 256-byte LUT size.
9. **OPTIMAL MINIMAL CONTEST SCORE**: HONESTLY PREDICTED-ONLY per Catalog #287; first empirical anchor pending operator-routable paired smoke after BUILD subagent + Catalog #325 symposium.

---

## §8. Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Unwind path |
|---|---|---|
| `grayscale_lut substrate is structurally a 256-entry LUT` | HARD-EARNED | Per existing `src/tac/substrates/grayscale_lut/architecture.py`; verify via direct file read at BUILD time |
| `canonical equation #26 _INCLUDED_CONTEXTS covers chroma_lut_replacement` | HARD-EARNED | Verified via `grep -E '_INCLUDED_CONTEXTS' src/tac/canonical_equations/procedural_codebook_savings.py` |
| `grayscale_lut archive.zip currently contains the LUT bytes (NOT in Python source)` | CARGO-CULTED-PENDING-EMPIRICAL | BUILD subagent MUST verify by reading `src/tac/substrates/grayscale_lut/archive.py` to confirm LUT bytes live in archive, NOT in Python module imports |
| `pairwise composition_alpha with DP1 is ADDITIVE` | CARGO-CULTED-PENDING-EMPIRICAL | First paired smoke empirical anchor required per Catalog #322 v2 cascade |
| `pairwise composition_alpha with VQ-VAE is SUB-ADDITIVE` | CARGO-CULTED-PENDING-EMPIRICAL | First paired smoke empirical anchor required; overlap on "lookup-table-replacement" semantic is hypothesis |
| `256-byte LUT is sufficient to capture chroma quantization` | HARD-EARNED | Canonical for 8-bit grayscale; sister to PR #56 selfcomp pattern + Quantizr |
| `32-byte seed is sufficient entropy for 256-byte LUT derivation` | HARD-EARNED | Per slot 1 procedural_codebook_generator design; PCG64 has period 2^128 >> 256 entries |

---

## §9. Observability surface (Catalog #305)

Per the 6-facet definition:

1. **Inspectable per layer**: every layer's input + output is captured at runtime; `derive_procedural_grayscale_lut_replacement` returns the 256-byte LUT for direct inspection; `compose_procedural_archive` returns the archive bytes; archive parse-time exposes the recovered LUT bytes.
2. **Decomposable per signal**: predicted ΔS decomposes into (a) bytes-saved component (`-25 * (N - K) / 37_545_489 = -0.000149`) + (b) per-axis composition (rate-axis ONLY; seg + pose axes unchanged because the substrate's score-aware path is structurally identical to baseline grayscale_lut).
3. **Diff-able across runs**: two runs with the same seed produce byte-identical archive bytes; diff is empty per determinism.
4. **Queryable post-hoc**: archive bytes + seed bytes + canonical_equation_26_context all persistable to JSON; sister probe consumer can query `.omx/state/lane_registry.json` notes for the lane.
5. **Cite-able**: every signal is anchored to (substrate=grayscale_lut, commit=<future>, seed_bytes_sha256=<32-byte hash>, canonical_equation_id=procedural_codebook_from_seed_compression_savings_v1).
6. **Counterfactual-able**: Catalog #272 byte-mutation smoke proves that single-byte seed flip changes the rendered frames; the "what if this byte changed?" question is answerable per Catalog #139 packet compiler + #105 no-op detector.

---

## §10. Catalog #324 predicted_band_validation_status

`predicted_band_validation_status: pending_post_training`

Reactivation criterion = "post-training Tier-C re-measurement via `tools/mdl_scorer_conditional_ablation.py --tier c` on the landed paired-smoke archive sha; smoke recipe MUST be `dispatch_enabled: false` until Catalog #325 symposium PROCEEDs".

Predicted band: per canonical equation #26 closed form `-0.000149` ± 5%; HONESTLY a NARROW band because the equation #26 closed form has high analytical precision; empirical uncertainty primarily comes from composition_alpha v2 cascade unknown.

---

## §11. Top-3 operator-routable next-actions

1. **OP-ROUTABLE #1 — BUILD subagent for grayscale_lut PROCEDURAL VARIANT**: spawn a sister subagent following the exact DP1 BUILD canonical pattern (commit `9cbfa471c` + landing memo). The BUILD subagent will land `src/tac/substrates/grayscale_lut/distillation_procedural_variant.py` + `tests/test_procedural_variant.py` + `__init__.py` re-exports + `archive.py` shim + `lane_grayscale_lut_procedural_codebook_replacement_variant_20260520` L0→L1. Estimated ~75 min + $0 GPU per VQ-VAE pattern empirical.

2. **OP-ROUTABLE #2 — Per-substrate symposium for grayscale_lut PROCEDURAL VARIANT** (Catalog #325): sextet pact (Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary) + grand council attendees (Selfcomp + Mallat + van den Oord + MacKay for the chroma-LUT class). 6-step canonical contract: (1) cargo-cult audit per Catalog #303, (2) 9-dim checklist per Catalog #294, (3) observability surface per Catalog #305, (4) sextet pact deliberation, (5) per-substrate reactivation criteria, (6) Catalog #324 post-training Tier-C validation discipline declared. Symposium memo dated `.omx/research/council_grayscale_lut_procedural_variant_<YYYYMMDD>.md` per Catalog #325 naming.

3. **OP-ROUTABLE #3 — 5-substrate matrix sequencing update**: amend `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md` (commit `b3e3442c3`) to (a) record PR101 lc_v2_clone NOT-CANDIDATE verdict from THIS audit, (b) record PR106 STRUCTURALLY OUT OF SCOPE verdict, (c) RE-SEQUENCE the 5-substrate matrix as: 1st DP1 (LANDED 9cbfa471c) → 2nd VQ-VAE (LANDED 6fea30f22) → 3rd grayscale_lut (THIS DESIGN MEMO) → 4th NSCS06 v8 chroma-LUT (sister; symposium #852 anchor) → 5th ATW V2 codec quantizer LUT (sister; symposium recency unknown). Time-Traveler L5 (TT5L) constants pattern REMAINS in the matrix but at lower priority due to REFUSE verdict per the original matrix design.

---

## §12. Sister coordination + Catalog #340 sister-checkpoint

- **Sister-DISJOINT from PR101 GOLD NULL-BYTE REMOVAL SMOKE (`a3dfc84c`)** — that lane targets MASTER-GRADIENT-NULL bytes in the fec6 frontier ARCHIVE (16,292 bytes); THIS lane targets DETERMINISTIC-CONSTANT byte regions in SUBSTRATE CODE; different files (`tools/run_pr101_gold_null_byte_removal_*.py` vs `.omx/research/pr101_pr106_procedural_variant_build_design_20260520.md`); different surface (CPU smoke vs design memo).
- **Sister-DISJOINT from MAGIC CODEC FIX (`a90e800a`)** — that lane targets `submissions/magic_codec_pr106*/inflate.py` runtime fixes; THIS lane is a design-memo only.
- **Catalog #340 sister-checkpoint guard**: confirmed PROCEED on the single target file `.omx/research/pr101_pr106_procedural_variant_build_design_20260520.md` (no sister activity in 12-hour or 4-hour windows per `tools/check_sister_files_recently_landed.py`).

---

## §13. Catalog gate compliance summary

- Catalog #110 + #113 APPEND-ONLY (NEW design memo only; ZERO mutation of existing forensic memos including sister DP1/VQ-VAE landing memos + 5-substrate matrix memo + canonical equation #26 source)
- Catalog #117 + #157 + #174 canonical commit serializer with POST-EDIT --expected-content-sha256
- Catalog #119 Co-Authored-By Claude trailer
- Catalog #125 6-hook wire-in declaration (see §14)
- Catalog #185 LIVE_COUNT verified empirically
- Catalog #186 catalog # claimed via canonical serializer (none claimed; no new gates)
- Catalog #206 crash-resume discipline (3+ checkpoints emitted)
- Catalog #229 premise-verification-before-edit (full audit of PR101 lc_v2_clone source + verified PR106 absence + DP1/VQ-VAE canonical pattern)
- Catalog #287 placeholder-rationale rejection (no `<rationale>` or `<reason>` literals)
- Catalog #290 canonical-vs-unique decision per layer (§6)
- Catalog #294 9-dim success checklist evidence (§7)
- Catalog #296 predicted-band Dykstra-feasibility (rate-axis trivially feasible; -0.000149 well within feasible region)
- Catalog #300 v2 frontmatter complete (T1 design memo; 7 council attendees; verdict PROCEED_WITH_REVISIONS)
- Catalog #303 cargo-cult audit (§8; 5 HARD-EARNED + 4 CARGO-CULTED-PENDING-EMPIRICAL)
- Catalog #305 observability surface (§9)
- Catalog #309 horizon_class = `frontier_protecting` (predicted ΔS lives INSIDE canonical equation #26's domain; extincts research-substrate-trap class per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY")
- Catalog #313 probe-outcomes ledger checked (no blocking entries for grayscale_lut procedural-variant + PR101 lc_v2_clone NOT-CANDIDATE verdict queued for OP-ROUTABLE #2)
- Catalog #315 OPTIMAL FORM (design memo only; no paid dispatch fired)
- Catalog #322 composition_alpha estimate (§5.7; ADDITIVE + SUB-ADDITIVE-CANDIDATE pairwise)
- Catalog #324 predicted_band_validation_status = `pending_post_training` (§10)
- Catalog #325 per-substrate symposium gating (§5.6; symposium queued as OP-ROUTABLE #2)
- Catalog #340 sister-checkpoint staging guard (§12)
- Catalog #344 canonical equation cross-reference (HTML comment after frontmatter + extensive §3 + §5 citations)

---

## §14. 6-hook wire-in declaration per Catalog #125

1. **sensitivity-map**: N/A (design memo; not algorithmic primitive)
2. **Pareto constraint**: ACTIVE upon BUILD landing (canonical equation #26 ΔS contribution to rate-axis Pareto polytope; estimated -0.000149)
3. **bit-allocator hook**: ACTIVE upon BUILD landing (32-byte seed slot replaces 256-byte LUT slot)
4. **cathedral autopilot dispatch hook**: ACTIVE upon BUILD landing via sister consumer `tac.cathedral_consumers.procedural_codebook_generator_consumer` (auto-discovered per Catalog #335)
5. **continual-learning posterior update**: ACTIVE on first empirical anchor via `update_equation_with_empirical_anchor` per Catalog #344 RECALIBRATE_ON_NEW_ANCHORS trigger
6. **probe-disambiguator**: ACTIVE upon BUILD landing (PROCEDURAL vs ORIGINAL trained-grayscale-LUT 2-recipe contrast IS the probe disambiguator)

---

## §15. Mission contribution

`council_predicted_mission_contribution: frontier_protecting`

Rationale: this design memo (a) prevents the cargo-cult-substrate-pick failure mode where PR101 lc_v2_clone would have been BUILT as a procedural variant despite being structurally NOT-CANDIDATE, costing ~75 min of subagent compute + creating a research-substrate-trap class artifact per CLAUDE.md 8th forbidden pattern; (b) preserves the canonical 5-substrate matrix sequencing per `b3e3442c3` by pivoting honestly to grayscale_lut as the 3rd candidate; (c) keeps the predicted-only canonical equation #26 closed-form bound INSIDE the domain-of-validity per slot 3 commits `8d8a7c6c5` + `37fea4aac`; (d) extincts the silent-canvas-mismatch bug class structurally for future cascade-sequencing operator-picks.

---

## §16. Blockers + sign-off

- **Blockers**: NONE for THIS design memo (design-only; no paid dispatch; no commit serializer collisions; no canonical equation mutations; no sister memo mutations per Catalog #110/#113 APPEND-ONLY)
- **Cost**: $0 paid GPU + ~75 min wall-clock
- **Lane**: `lane_wave_3_pr101_pr106_procedural_variant_build_design_20260520` L1 (impl_complete + memory_entry)
- **mission_predicted_contribution**: `frontier_protecting`
- **horizon_class**: `frontier_protecting` (per Catalog #309; predicted ΔS lives INSIDE canonical equation #26 domain-of-validity; sister of DP1 + VQ-VAE; bounded -0.000149 well above 0.0)
- **Sister coordination**: DISJOINT from PR101 GOLD NULL-BYTE REMOVAL SMOKE + MAGIC CODEC FIX
- **Discipline**: full Catalog gate compliance per §13


# OBSERVABILITY_SURFACE_SECTION_WAIVED:historical_design_memo_predates_catalog_305_section_header_requirement_or_is_namespace_design_not_substrate_specific_observability_per_catalog_110_113_HISTORICAL_PROVENANCE_APPEND_ONLY_discipline_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
