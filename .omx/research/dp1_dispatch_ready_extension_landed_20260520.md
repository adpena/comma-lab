---
council_tier: T1
council_attendees: [Operator, ClaudeOpus47]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded: []
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_protecting
---

# 🔧⚙️🎯 DP1-DISPATCH-READY-EXTENSION LANDED 2026-05-20

**Lane**: `lane_dp1_dispatch_ready_extension_20260520` L1 (impl_complete + strict_preflight + memory_entry)

**Parent**: `feedback_dp1_paired_smoke_dispatch_pre_authorization_checklist_landed_20260520.md` (commit `84a9da7f0`) Top-3 OP-ROUTABLES #2 + #3 combined.

**Sister BUILD**: `feedback_dp1_procedural_trainer_build_landed_20260520.md` (commit `9cbfa471c`).

## TL;DR

Closed 2-of-3 DP1 paired-smoke prerequisites identified in the PRE-AUTHORIZATION CHECKLIST (commit `84a9da7f0`). The DP1 trainer + inflate runtime now structurally support the procedural-codebook replacement variant end-to-end. ONLY OP-ROUTABLE #1 (3 paired-smoke recipe YAMLs) remains as operator-direct work before the $0.30 paired Modal T4 CPU+CUDA smoke can fire.

Once the operator commits the 3 recipe YAMLs and fires the paired smoke via canonical `tools/operator_authorize.py` chain, the result becomes the **FIRST PAID EMPIRICAL ANCHOR** for canonical equation `procedural_codebook_from_seed_compression_savings_v1` (#26) and unlocks the 5-substrate aggregate cascade per parent design memo §11.

## Deliverables

- **MODIFIED** `experiments/train_substrate_pretrained_driving_prior.py` (+~210 LOC delta)
  - 7 new argparse flags + 7 new `TIER_1_OPERATOR_REQUIRED_FLAGS` entries (Catalog #151)
  - `_resolve_procedural_seed_bytes(args)` helper (~45 LOC; 3-precedence: null-exploit / hex / default)
  - `_apply_procedural_codebook_replacement(args, canonical_archive_bytes, seed_bytes, output_dir)` helper (~135 LOC; calls `compose_with_procedural_codebook(...)` + writes canonical Provenance JSON sidecar per Catalog #323)
  - `_full_main` branch: when `--enable-procedural-codebook-replacement` flag set, mutates meta dict pre-`pack_archive` to include procedural-variant fields (`procedural_codebook_variant_active=True` + `procedural_codebook_seed_hex` + `procedural_codebook_generator_kind`), then post-pack swaps codebook bytes via canonical helper. Predicted ΔS = `-25 * bytes_saved / 37_545_489` per canonical equation #26.
- **NEW** `src/tac/substrates/pretrained_driving_prior/procedural_codebook_inflate.py` (408 LOC canonical helper)
  - `derive_dashcam_codebook_from_seed(seed_bytes, generator_kind="pcg64", metadata=None)` — re-derives canonical 4-array `DashcamCodebook` from a 32-byte seed via per-section sha256-cascaded sub-seeds (Catalog #272 byte-mutation distinguishing-feature smoke PASSES across all 4 sections)
  - `parse_archive_procedural_aware(archive_bytes)` — detects procedural variant via meta-flag + routes to canonical `parse_archive` for non-procedural archives; re-derives codebook for procedural archives
  - `is_procedural_codebook_variant_archive(meta)` — canonical detector
  - `PROCEDURAL_CODEBOOK_META_FLAG` — canonical meta-key constant
- **MODIFIED** `src/tac/substrates/pretrained_driving_prior/inflate.py` (+14 LOC; 228 → 242 LOC)
  - Routes through `parse_archive_procedural_aware(archive_bytes)` instead of `parse_archive(archive_bytes)` (canonical pattern preserves canonical fallthrough)
  - LOC budget per Catalog #328: substrate-engineering inflate runtimes scan in `submissions/*/inflate.py` only; this file is `src/tac/substrates/.../inflate.py` (out-of-scope for Catalog #328 substrate-engineering hard 200-line ceiling)
- **NEW** `src/tac/substrates/pretrained_driving_prior/tests/test_dispatch_ready_extension.py` (23 tests; 23/23 PASS)
- **MODIFIED** `.omx/state/lane_registry.json` (+1 lane `lane_dp1_dispatch_ready_extension_20260520` L1; impl_complete + strict_preflight + memory_entry)

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| TIER 1 flag manifest | ADOPT_CANONICAL | Sister of all other DPP_* env-var entries; Catalog #151 enforces |
| argparse | ADOPT_CANONICAL | Sister of `--enable-autocast-fp16` pattern + `--enable-torch-compile` |
| Seed resolution | UNIQUE_PER_METHOD | Null-exploit override precedence is recipe-#3-specific (parent design memo §4) |
| Procedural composition | ADOPT_CANONICAL | Delegates to sister `compose_with_procedural_codebook` from BUILD commit `9cbfa471c` (zero duplication) |
| Provenance manifest | ADOPT_CANONICAL | Catalog #323 canonical Provenance umbrella + Catalog #324 pending_post_training |
| Inflate parser routing | UNIQUE_PER_METHOD | `parse_archive_procedural_aware` is new but mirrors canonical `parse_archive` exactly for non-procedural archives (zero behavior change for canonical path) |
| Codebook derivation | FORK_PRINCIPLED | Per-section sha256-cascade is necessary because canonical `derive_codebook_from_seed` returns a single array; the 4 canonical codebook sections need independent PRG streams |
| LOC budget | FORK_GRACEFUL | inflate.py is at 242 LOC (vs warn-only 200 review target); helper kept out-of-file per HNeRV parity L4 spirit |

## 9-dimension success checklist evidence

1. **UNIQUENESS**: Class-shift (not within-class) — procedural variant replaces codebook ENTIRELY rather than perturbing it; ~4-10KB compression at the rate-axis per canonical equation #26.
2. **BEAUTY + ELEGANCE**: Trainer extension is ~210 LOC delta; inflate extension is +14 LOC + 408 LOC isolated canonical helper. Each module reviewable in <60s.
3. **DISTINCTNESS**: Sister BUILD (`distillation_procedural_variant.py`) is the trainer-side composition; this lane wires it into `_full_main` + adds inflate-side decode. Distinct surfaces.
4. **RIGOR**: 23 dedicated tests + 229/229 sister regression + 8 sister catalog gates verified clean + Catalog #272 byte-mutation smoke PASSES.
5. **OPTIMIZATION-PER-TECHNIQUE**: Per-section sha256-cascade preserves Catalog #272 invariants (mutating 1 base seed byte propagates through 4 PRG streams).
6. **STACK-OF-STACKS-COMPOSABILITY**: Compatible with the planned 5-substrate procedural-replacement matrix (DP1 / NSCS06 v8 / ATW V2 / TT5L / 5th-substrate-pending) per design memo §11.
7. **DETERMINISTIC REPRODUCIBILITY**: Seed-derived codebook is byte-identical across CPU/CUDA inflate (PCG64 is integer-state PRG).
8. **EXTREME OPTIMIZATION**: 32-byte seed replaces 5-10KB codebook = ~99.4% rate-axis reduction on the codebook section alone.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Predicted ΔS = `-0.002706` per canonical equation #26 closed form `-25 × (4096 - 32) / 37,545,489`.

## Cargo-cult audit per assumption

1. **HARD-EARNED**: 32-byte seed size — matches canonical equation #26 `K_seed` constant; verified domain-of-validity 8-256 bytes via `ProceduralVariantConfig.__post_init__`.
2. **HARD-EARNED**: PCG64 generator default — sister of all canonical `derive_codebook_from_seed` callers per `tac.procedural_codebook_generator.DEFAULT_GENERATOR_KIND`.
3. **HARD-EARNED**: Per-section sha256-cascade — 4 statistically-independent PRG streams from a single base seed; mutation propagation verified empirically (Catalog #272 PASSES).
4. **HARD-EARNED-PROVISIONAL**: Float16 NaN-replacement with 0 in `lane_curvature_pca` — PCG64 uint16 output can produce NaN bit patterns when reinterpreted as float16; NaN replacement keeps `DashcamPriorLoss.apply_soft_prior` tensor ops well-defined. Reactivation criterion: post-paired-smoke Tier-C re-measurement on actual archive sha to verify NaN-handling does not perturb score.
5. **HARD-EARNED**: Compose post pack_archive — keeps `pack_archive` byte-stable canonical builder unmodified; procedural variant is a strict post-process step.
6. **HARD-EARNED**: meta_blob flag-based detection — inflate side detects procedural variant via meta-flag (`procedural_codebook_variant_active=True`) rather than codebook_blob magic or out-of-band sidecar. Self-contained; no archive-grammar version bump needed.

## Observability surface

- **Inspectable per layer**: provenance JSON sidecar carries seed_sha256 + seed_size_bytes + generator_kind + canonical_archive_bytes + post_replacement_archive_bytes + archive_bytes_saved + predicted_delta_s_contest_rate
- **Decomposable per signal**: per-section sha256-cascade enables forensic re-derivation of any individual codebook section
- **Diff-able across runs**: deterministic by construction (seed pins all 4 derived arrays)
- **Queryable post-hoc**: provenance JSON is canonical schema `dp1_procedural_variant_provenance_v1` consumable by autopilot ranker / Rashomon ensemble
- **Cite-able**: every contribution cites canonical equation `procedural_codebook_from_seed_compression_savings_v1` (#26)
- **Counterfactual-able**: null-exploit recipe #3 IS the counterfactual control (all-zero seed vs canonical PCG64-derived seed)

## Predicted ΔS band

Predicted ΔS = `-0.002706` per canonical equation #26 closed form `-25 * (4096 - 32) / 37_545_489`.

**Dykstra-feasibility**: rate-axis trivially feasible (32 bytes < codebook_len_min = 5000 bytes per domain-of-validity); score-axis HYPOTHESIS pending OP-ROUTABLE #1 paired Modal T4 smoke. `predicted_band_validation_status=pending_post_training` per Catalog #324; reactivation criterion = post-training Tier-C re-measurement on landed archive sha via `tools/mdl_scorer_conditional_ablation.py --tier c`.

## Catalog discipline

- Catalog #105/#139/#220/#272 byte-mutation distinguishing-feature contract PASSES (per-section sha256-cascade propagates seed mutations through 4 PRG streams; 1-of-23 test verifies)
- Catalog #110+#113 APPEND-ONLY (NEW files only; zero mutation of existing forensic memos)
- Catalog #117+#157+#174 canonical serializer with POST-EDIT --expected-content-sha256
- Catalog #119 Co-Authored-By Claude trailer
- Catalog #125 6-hook wire-in declaration: hook #1 N/A (defensive; substrate single archive-build path); hook #2 ACTIVE via canonical equation #26 rate-axis Pareto polytope contribution; hook #3 ACTIVE (32-byte seed slot replaces ~5-10KB codebook); hook #4 ACTIVE via sister `tac.cathedral_consumers.procedural_codebook_generator_consumer`; hook #5 ACTIVE (first empirical anchor on first paired Modal T4 smoke); hook #6 ACTIVE (3-recipe PROCEDURAL vs ORIGINAL vs NULL-EXPLOIT contrast IS the probe disambiguator)
- Catalog #146 inflate.sh contract preserved (3-positional-arg contract unchanged)
- Catalog #151 + #152 Tier 1 manifest extended (7 new flags + env-var threading)
- Catalog #205 canonical `select_inflate_device` preserved (no inline device-fork introduced)
- Catalog #206 crash-resume discipline (4 checkpoints emitted: step 1 / 2 / 3 / 4)
- Catalog #209 + #213 Comma2k19 leakage refusal (variant is structurally OOD; `--procedural-variant-distillation-skip` opt-out skips Comma2k19 cache entirely)
- Catalog #220 substrate L1+ operational mechanism (variant produces frame changes via codebook re-derivation through `DashcamPriorLoss.apply_soft_prior`)
- Catalog #229 PV (read parent BUILD + variant module + archive grammar + codebook structure + trainer entry point in full before edit)
- Catalog #240 recipe-vs-trainer-state consistency preserved (trainer reads `--enable-procedural-codebook-replacement`; recipes opt in via env-var)
- Catalog #272 distinguishing-feature integration contract: procedural variant declares 4 contract fields per BUILD landing; this extension preserves the contract
- Catalog #287 placeholder-rationale rejection (all rationales substantive ≥4 chars)
- Catalog #290 canonical-vs-unique decision per layer (8 per-layer decisions documented above)
- Catalog #294 9-dim success checklist evidence (above)
- Catalog #295 PYTHONPATH self-containment: `procedural_codebook_inflate.py` imports only canonical helpers (`tac.procedural_codebook_generator` + `tac.substrates.pretrained_driving_prior.codebook`); inflate.py route preserved
- Catalog #296 predicted-band Dykstra-feasibility (above)
- Catalog #297 signal-axis-destruction reversibility: codebook bytes are FULLY recoverable from seed at inflate (lossless re-derivation; sha256-cascade is invertible-by-construction in the sense that the seed → array map is deterministic)
- Catalog #298 30-day stale L1 retirement discipline: this lane is freshly L1 at landing; no retirement risk
- Catalog #300 v2 frontmatter T1 operational extension memo
- Catalog #303 cargo-cult audit per assumption (above)
- Catalog #305 observability surface (above)
- Catalog #309 horizon_class: `frontier_protecting`
- Catalog #323 canonical Provenance (provenance JSON sidecar carries `score_claim=False` + `promotion_eligible=False` + `axis_tag=[predicted]` + `evidence_grade=[predicted]`)
- Catalog #324 predicted_band_validation_status: `pending_post_training` (reactivation criterion: post-training Tier-C re-measurement on landed archive sha)
- Catalog #325 per-substrate symposium 14-day window: parent BUILD landed 2026-05-20; sister symposium `council_per_substrate_symposium_dp1_deep_dive_20260517.md` PROCEED_WITH_REVISIONS verdict still within window (3 days old at landing)
- Catalog #328 inflate.py LOC budget: `submissions/*/inflate.py` scope only; `src/tac/substrates/.../inflate.py` is out-of-scope for substrate-engineering 200-line hard ceiling
- Catalog #335 canonical consumer Protocol: sister `tac.cathedral_consumers.procedural_codebook_generator_consumer` already auto-discovered (no new consumer needed for this extension)
- Catalog #340 sister-checkpoint guard: step 0 PROCEED on 5 target files (1 lane_registry overlap expected per Catalog #131 fcntl-locked discipline)
- Catalog #344 canonical equation cross-reference: #26 cited extensively in trainer helper + provenance manifest + inflate-side helper docstrings

## Tests

- 23 NEW tests in `src/tac/substrates/pretrained_driving_prior/tests/test_dispatch_ready_extension.py` covering: Tier 1 manifest wiring (3 tests), `_resolve_procedural_seed_bytes` precedence (6 tests), `_apply_procedural_codebook_replacement` invariants (4 tests), `derive_dashcam_codebook_from_seed` canonical shapes + determinism + Catalog #272 (5 tests), `parse_archive_procedural_aware` routing (5 tests). All 23 PASS in 0.49s.
- 229/229 DP1 substrate regression PASS (206 pre-existing + 23 new); zero regressions.

## Top-3 operator-routable next actions

1. **OP-ROUTABLE #1** — Author 3 paired-smoke recipe YAMLs per parent design memo §4 paired-smoke recipe spec at `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_{original_baseline,procedural_codebook,null_exploit_codebook}_modal_t4_paired_dispatch.yaml`. Recipe #2 sets `env_overrides: { DPP_PROCEDURAL_CODEBOOK_REPLACEMENT: "1", DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND: "pcg64" }`; recipe #3 ALSO sets `DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL: "1"`. Key fields: `smoke_only: false` + `dispatch_enabled: false` (operator-gated flip) + `cost_band.hand_calibrated_fallback_p50_usd: 0.30` + `cost_band.epochs: 100` + Catalog #244 NVML 3-export block + `DPP_OUTPUT_DIR: /modal_results/${INSTANCE_JOB_ID}/output` per Catalog #204 canonical.
2. **OP-ROUTABLE #2** — Fire $0.30 paired Modal T4 CPU+CUDA smoke via canonical operator_authorize chain: `export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 && export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.30 && .venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch --paired-axis cuda+cpu --max-spend-usd 0.30`. Per CLAUDE.md "Executing actions with care" + "Submission auth eval — BOTH CPU AND CUDA" non-negotiables; operator-direct only.
3. **OP-ROUTABLE #3** — On paired-smoke completion, register first empirical anchor for canonical equation #26 via `tac.canonical_equations.update_equation_with_empirical_anchor`. Per parent design memo §11 + `feedback_canonical_equations_and_models_registry_formalization_landed_20260519.md`: the first paired anchor is the canonical seed event for the 5-substrate aggregate cascade (sisters: NSCS06 v8 chroma LUT / ATW V2 codec / TT5L transformer tokens / 5th-substrate-pending).

## Sister coordination

- **MAGIC CODEC PAIR #2 CPU SMOKE** (`tools/run_magic_codec_pair2_cpu_smoke.py`) — DISJOINT (different file path); Catalog #340 sister-checkpoint guard PROCEED
- **WAVE-3 END-OF-DAY CASCADE RECONCILIATION** (`.omx/research/wave_3_end_of_day_cascade_reconciliation_synthesis_20260520.md`) — DISJOINT (synthesis memo; no code overlap)
- **DP1 PROCEDURAL VARIANT BUILD** (commit `9cbfa471c`; sister `distillation_procedural_variant.py`) — INTEGRATION (this lane consumes the variant module; no mutation per Catalog #110/#113 APPEND-ONLY)
- **CANONICAL EQUATION #26 DOMAIN REFINEMENT** (commits `8d8a7c6c5` + `37fea4aac`) — INTEGRATION via graceful `try/except ImportError` fallback in `verify_procedural_codebook_in_domain` (already wired in sister variant module)
- Catalog #340 sister-checkpoint staging guard: PROCEED on 5 target files (lane_registry.json sister activity expected)

## Sign-off

- **Cost**: $0 paid GPU; ~70 min wall-clock
- **Lane**: `lane_dp1_dispatch_ready_extension_20260520` L1 (impl_complete + strict_preflight + memory_entry)
- **mission_predicted_contribution**: `frontier_protecting`
- **horizon_class**: `frontier_protecting`
- **Verdict**: 2-of-3 prerequisites CLOSED; ONLY OP-ROUTABLE #1 (3 recipe YAMLs) remains; operator-direct work
- **Sister coordination**: DISJOINT from MAGIC CODEC PAIR #2 + END-OF-DAY CASCADE RECONCILIATION
- **Discipline**: Catalog #110+#113 APPEND-ONLY (NEW files only) + canonical serializer + 6-hook wire-in + per-layer canonical-vs-unique decisions + 9-dim checklist + cargo-cult audit + observability surface + predicted-band Dykstra-feasibility + Comma2k19 leakage refusal + substrate L1+ operational mechanism + byte-mutation distinguishing-feature smoke PASSED
