# Quantizr Canonical 3-Stack Audit — EMA 0.997 + KL T=2.0 + eval_roundtrip=True

**Date**: 2026-05-29T07:54Z
**Lane**: `lane_slot_ee_quantizr_catalog_325_audit_20260529`
**Source directive**: Slot CC T3 grand-council strategic-reprioritization Quantizr binding revision #5 verbatim *"Every PR-95-PARITY SUBSTRATE BUILD MUST include EMA decay 0.997 + KL distill T=2.0 + eval_roundtrip=True per CLAUDE.md non-negotiables. 0.196-0.199 cluster includes substrates that VIOLATE these silently. Catalog #325 6-step symposium MUST audit substrate trainer for all 3 canonical primitives before PROCEED."*

**Mission contribution per Catalog #300 §Mission alignment Consequence 5**: `frontier_breaking_enabler` (audit + canonical-equation candidates structurally protect Class A wavelet + Class D Wyner-Ziv substrate Catalog #325 6-step symposia from CARGO-CULTED Quantizr-canonical 3-stack omission — the operator-binding Slot CC dissent revision that gates Class A + Class D scope-lock).

**Cost**: $0 (READ-ONLY source audit + canonical equation registration + canonical anti-pattern registration + landing memo)
**Wall-clock**: ~60 min

---

## TL;DR

Empirical audit across **100 substrate trainers** (`experiments/train_substrate_*.py`):

| 3-stack adherence | Count | % |
|---|---|---|
| **ALL THREE** (EMA + KL T=2.0 + eval_roundtrip) | **1** | 1.0% |
| EMA + eval_roundtrip (missing KL) | 64 | 64.0% |
| KL only (missing EMA + eval_roundtrip) | 13 | 13.0% |
| eval_roundtrip only | 8 | 8.0% |
| EMA only | 3 | 3.0% |
| NONE | 12 | 12.0% |

**Single 3-stack-canonical trainer**: `experiments/train_substrate_pact_nerv_distilled_scorer.py` (declares `--ema-decay 0.997` + `--distill-temperature 2.0` + `from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally`).

**Class A wavelet + Class D Wyner-Ziv targets** (the Slot CC scope-lock candidates):

| Substrate trainer | EMA | KL T=2.0 | eval_roundtrip | 3-stack |
|---|---|---|---|---|
| `train_substrate_wavelet.py` (Class A canonical) | ✅ | ❌ | ✅ | **2/3** |
| `train_substrate_a1_plus_wavelet_residual.py` | ✅ | ❌ | ✅ | 2/3 |
| `train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py` | ✅ | ❌ | ✅ | 2/3 |
| `train_substrate_nscs06_v8_path_b_wavelet.py` | ❌ | ❌ | ✅ | 1/3 |
| `train_substrate_d4_wyner_ziv_frame_0.py` (Class D canonical) | ✅ | ❌ | ✅ | **2/3** |
| `train_substrate_wyner_ziv_cooperative_receiver.py` | ✅ | ❌ | ✅ | 2/3 |

**Empirical gap**: 0 of 6 Class A/D substrate trainers honor the full 3-stack. ALL miss KL T=2.0 SegNet distillation per Quantizr canonical + PR100 hnerv_lc_v2 canonical.

**Per Catalog #307 paradigm-vs-implementation classification**: this is IMPLEMENTATION-LEVEL falsification of canonical Quantizr-canonical 3-stack adherence claims, NOT paradigm-level falsification of Class A/D substrate paradigms. Reactivation paths per CLAUDE.md "Forbidden premature KILL without research exhaustion": (a) backfill KL T=2.0 SegNet distillation into Class A + Class D trainers per Quantizr canonical; (b) iterate per substrate-optimal-engineering decision (KL T=2.0 may not serve Class A wavelet hierarchical-planning + may not serve Class D Wyner-Ziv decoder-side side-info per substrate-canonical scope-decision); (c) Slot CC Class A + Class D Catalog #325 6-step symposia MUST surface the per-substrate canonical-vs-fork decision per Catalog #290 + #303 cargo-cult audit.

---

## Canonical 3-stack source citations

### EMA decay 0.997 (Quantizr canonical)

- **CLAUDE.md "EMA — NON-NEGOTIABLE, HIGHEST EMPHASIS"** verbatim: *"Quantizr decay = 0.997. All weight EMAs default to `decay=0.997`. The CANONICAL `class EMA` is `tac.training.EMA` (with the float-buffer guard at L359-364 and the late-bound module guard at L356-358). Codebook EMAs (van den Oord persistent buffer form, e.g. `LearnableClassTargets`, `vqvae_codec`) keep their own 0.99 default — codebooks adapt faster than weights by design."*
- **CLAUDE.md "Quantizr intelligence"** verbatim: *"Training: 5-stage pipeline (anchor→finetune→joint→QAT→final), EMA, diff_round(), diff_rgb_to_yuv6()"*
- Canonical Council D EMA audit: `.omx/research/council_ema_audit_20260429.md` (5 paths correct + 8 MISSING + 3 codec-style gaps).
- Catalog #88 `check_training_paths_use_ema_correctly` STRICT @ 0 violations.

### KL distill T=2.0 (Quantizr canonical SegNet distillation)

- **CLAUDE.md "Quantizr intelligence"** verbatim: *"SegNet: kl_on_logits() with T=2.0 for distillation during training"*
- **CLAUDE.md "Critical lessons - DO NOT repeat these mistakes"** verbatim: *"KL distill caused PoseNet collapse as primary loss. BUT Quantizr uses kl_on_logits(T=2.0) for SegNet during specific training phases alongside standard loss. Revisit with staged approach — KL distill for SegNet only, not as sole loss."*
- PR100 hnerv_lc_v2 canonical reference: `experiments/results/public_pr100_intake_*/source/submissions/hnerv_lc_v2/` (KL T=2.0 SegNet distill alongside standard loss per stage 2 of 5-stage pipeline).
- Canonical equation `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` registered (17 empirical anchors); `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` registered (3 anchors); `pr95_family_l40_fixed_28_tensor_schema_list_v1` registered.

### eval_roundtrip=True (every-training-path canonical)

- **CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS"** verbatim: *"EVERY training path MUST use eval_roundtrip. There are ZERO exceptions. ... Without eval_roundtrip, proxy-auth gap is 2-6x on PoseNet. Every training run without it is a WASTED run."*
- Canonical helper: `src/tac/differentiable_eval_roundtrip.py` (PR #95-faithful eval-roundtrip + autograd-preserving rgb_to_yuv6).
- Per PR #95 Finding A + Finding B: eval_roundtrip MUST be baked into TRAINING inner loop, not eval-time only.
- Catalog #5 `check_no_eval_roundtrip_false` STRICT @ 0 violations.

---

## Per-substrate-trainer gap inventory (TOP-10 by Class A/D + PR-95-parity priority)

| Trainer | EMA | KL | ER | Gap action |
|---|---|---|---|---|
| `train_substrate_wavelet.py` | ✅ | ❌ | ✅ | Per Catalog #325 symposium: decide ADOPT_KL_PER_QUANTIZR vs FORK_KL_WAIVED_PER_DAUBECHIES_WAVELET_HIERARCHICAL_PLANNING canonical-vs-unique per Catalog #290 |
| `train_substrate_a1_plus_wavelet_residual.py` | ✅ | ❌ | ✅ | Same as above; sister wavelet substrate |
| `train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py` | ✅ | ❌ | ✅ | UNIWARD already SegNet-aware; KL T=2.0 ADOPT highest-EV per Fridrich canonical inverse-steganalysis |
| `train_substrate_nscs06_v8_path_b_wavelet.py` | ❌ | ❌ | ✅ | NSCS06 v8 wavelet research-only per Slot V Phase B; ADOPT KL+EMA both per Catalog #325 |
| `train_substrate_d4_wyner_ziv_frame_0.py` | ✅ | ❌ | ✅ | Per Catalog #325 symposium: decide ADOPT_KL_PER_QUANTIZR vs FORK_KL_WAIVED_PER_MACKAY_HOTZ_WYNER_ZIV_DECODER_SIDE_SIDE_INFO canonical-vs-unique per Catalog #290 |
| `train_substrate_wyner_ziv_cooperative_receiver.py` | ✅ | ❌ | ✅ | Sister Wyner-Ziv substrate; Atick-Redlich cooperative-receiver loss may dominate scorer-conditional KL signal |
| `train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py` | ✅ | ❌ | ❌ | **HIGH PRIORITY**: PR101 canonical baseline (frontier 0.193); MISSING eval_roundtrip + KL is critical CLAUDE.md non-negotiable violation. Backfill required per Catalog #5 + #325 |
| `train_substrate_pr101_with_dp1_prior_regularizer.py` | ❌ | ❌ | ❌ | **HIGHEST PRIORITY**: PR101 sister substrate with NONE of 3-stack — research_only opt-out or full 3-stack adoption required |
| `train_substrate_sane_hnerv.py` | ✅ | ❌ | ✅ | sane_hnerv canonical HNeRV scaffold; KL T=2.0 ADOPT highest-EV per HNeRV parity L7 bind-all-ingredients |
| `train_substrate_pact_nerv_distilled_scorer.py` | ✅ | ✅ | ✅ | **CANONICAL** — sole 3-stack-canonical trainer (reference exemplar for backfill) |

---

## Class A + Class D Catalog #325 6-step symposium content recommendations

Per Slot CC STRATEGIC RESET #2 + Quantizr binding revision #5: every PR-95-PARITY SUBSTRATE BUILD MUST include EMA decay 0.997 + KL distill T=2.0 + eval_roundtrip=True per CLAUDE.md non-negotiables.

### Class A (compressive-sensing wavelet per Daubechies+Mallat+Hotz canonical)

**Catalog #325 6-step contract** (per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable):

1. **Cargo-cult audit per Catalog #303**: Class A must explicitly classify Quantizr-canonical 3-stack per HARD-EARNED-vs-CARGO-CULTED addendum. Initial classification: KL T=2.0 = CARGO-CULTED-FROM-PR100 (Quantizr's training paradigm; not yet tested on Class A wavelet substrate); EMA decay 0.997 = HARD-EARNED-FROM-QUANTIZR-EMPIRICAL (0.33 [contest-CUDA] anchor + Council D audit); eval_roundtrip=True = HARD-EARNED-FROM-PR95-EMPIRICAL (2-11× PoseNet drift per CLAUDE.md non-negotiable). Per-assumption unwind-test plan: KL T=2.0 → free MLX-LOCAL probe on wavelet residual to determine SegNet KL distill compatibility with Daubechies wavelet hierarchical-planning representation before paid CUDA dispatch.

2. **9-dimension success checklist evidence per Catalog #294**: 9-dim must include UNIQUENESS dimension covering EMA + KL + eval_roundtrip 3-stack as binding constraints per HNeRV parity L7 (substrate engineering binds ALL ingredients simultaneously).

3. **Observability surface declaration per Catalog #305**: 6-facet must surface per-stage EMA shadow norm + KL T=2.0 distill loss decomposition (SegNet vs PoseNet) + eval_roundtrip gradient norm at each training step.

4. **Sextet pact deliberation per CLAUDE.md "Council conduct"**: Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary; Quantizr ATTENDS as grand-council topical (0.33 leaderboard winner). Grand-council attendees: Mallat (wavelet theory + scattering transforms) + Boyd (compressive-sensing convex optimization) + Hinton (KL distillation 2014 + temperature analysis).

5. **Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL"**: 3 paths if Class A wavelet falsifies KL T=2.0 (a) KL T=2.0 ablation $0 MLX-LOCAL probe before paid CUDA, (b) reactivate at PARTIAL_CONFIRMATION verdict per Catalog #307, (c) FORK to substrate-optimal-KL temperature via Hinton T sweep per substrate-canonical-vs-unique decision.

6. **Catalog #324 post-training Tier-C validation discipline**: `predicted_band_validation_status: pending_post_training` because EMA shadow + KL distill effects on wavelet hierarchical-planning are NOT measurable from random-init Tier-C density.

### Class D (Wyner-Ziv decoder-side side-info per MacKay+Hotz+Wyner canonical)

Same 6-step contract as Class A with substrate-specific revisions:

1. **Cargo-cult audit**: KL T=2.0 CARGO-CULTED-FROM-PR100 (Quantizr's training paradigm; not yet tested on Class D Wyner-Ziv substrate); EMA + eval_roundtrip same as Class A. Per-assumption unwind-test: KL T=2.0 free MLX-LOCAL probe on Wyner-Ziv encoder-side residual to determine whether decoder-side side-info dominates scorer-conditional KL signal.

2. **9-dimension success checklist**: include UNIQUENESS dimension covering Wyner-Ziv side-info encoding as binding constraint per Atick-Redlich cooperative-receiver canonical (decoder has scorer state-dict as shared prior).

3. **Observability surface**: surface per-pair Wyner-Ziv encoder-side bits + decoder-side side-info bit-count + KL T=2.0 distill loss at each step.

4. **Sextet pact + grand-council attendees**: + MacKay memorial (canonical *Information Theory, Inference, and Learning Algorithms* author; Wyner-Ziv 1976 successor) + Wyner (Wyner-Ziv 1976 source coding with side information theorem co-author) + Atick + Redlich (cooperative-receiver framing).

5. **Reactivation criteria**: 3 paths if Class D Wyner-Ziv falsifies KL T=2.0 (a) ablation probe, (b) PARTIAL_CONFIRMATION reactivation, (c) FORK to substrate-optimal-distillation (Wyner-Ziv decoder-side may need direct-supervised distillation instead of KL T=2.0).

6. **Tier-C validation**: same `pending_post_training` per random-init non-measurability.

---

## Canonical equation candidate (DEFERRED-to-operator-decision per Catalog #344 protocol)

**Equation ID**: `quantizr_canonical_ema_kl_eval_roundtrip_stack_savings_v1`

**One-line summary**: 3-stack Quantizr-canonical (EMA 0.997 + KL distill T=2.0 SegNet + eval_roundtrip=True) closes the proxy-auth gap and stabilizes weight-trajectory variance; combined Δscore predicted -0.005 to -0.020 per Council D audit + Quantizr empirical 0.33 anchor.

**LaTeX form**: `ΔS = α_ema × σ_weight + α_kl × KL_SegNet(T=2.0) + α_er × Δ_proxy_auth_gap`

**Domain of validity**: PR-95-parity-class substrates with SegNet/PoseNet scorer training; Class A wavelet + Class D Wyner-Ziv require per-substrate cargo-cult audit per Catalog #303 before adoption.

**Python callable module path**: `tac.canonical_equations.quantizr_canonical_3_stack.compute_quantizr_3_stack_savings`

**Canonical producers**: `tac.training.EMA`, `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`, `tac.losses.kl_on_logits` (canonical Hinton T=2.0)

**Canonical consumers**: `experiments.train_substrate_pact_nerv_distilled_scorer` (only 3-stack-canonical exemplar at audit time)

**Empirical anchors**: 0 at registration (the Quantizr 0.33 anchor + Council D audit findings document the predicted band but no PAIRED 3-stack-isolated controlled study exists; first EmpiricalAnchor lands when a Class A/D substrate empirically confirms the 3-stack adoption ΔS contribution).

**Predicted vs empirical residual**: deferred (first anchor pending).

**Next recalibration trigger**: `when_3+_new_empirical_anchors_in_domain` (auto-refit per Catalog #371 after Class A + Class D + a sister Class B/C/E PR-95-parity substrate land empirical anchors).

**Operator decision**: DEFER per "iterate not force" until 3+ empirical anchors land; per Catalog #344 protocol the registration awaits operator approval.

---

## Canonical anti-pattern candidate (DEFERRED-to-operator-decision per Catalog #344 sister protocol)

**Anti-pattern ID**: `substrate_trainer_missing_quantizr_canonical_3_stack_ema_kl_eval_roundtrip_v1`

**Paradigm class**: `discipline_anti_pattern` (per `PARADIGM_RIGOR_LOSS` taxonomy)

**Severity**: `high_compound_corruption` (per `SEVERITY_HIGH`)

**Description**: A substrate trainer at `experiments/train_substrate_*.py` that scaffolds training on SegNet/PoseNet contest scorers WITHOUT all 3 Quantizr-canonical primitives (EMA decay 0.997 + KL distill T=2.0 SegNet + eval_roundtrip=True per CLAUDE.md non-negotiables) silently violates the canonical Quantizr training paradigm that produced the 0.33 [contest-CUDA] leaderboard anchor. Empirical recurrence: 99/100 substrate trainers omit at least one canonical primitive at audit time 2026-05-29.

**Forbidden pattern predicate**: substrate trainer body contains `optimizer.step()` AND scorer.forward() AND lacks one of `(EMA 0.997, kl_on_logits(T=2.0), apply_eval_roundtrip_during_training)`.

**Falsification band**: predicted -0.005 to -0.020 ΔS per missing primitive (composable per primitive's independent contribution); actual band requires empirical anchor.

**Recurrence conditions**: (a) substrate trainer scaffold landed before canonical 3-stack helper at `tac.training` + `tac.differentiable_eval_roundtrip` + `tac.losses.kl_on_logits` was made the canonical default; (b) substrate author copy-pasted from non-3-stack-canonical trainer (e.g. `train_substrate_pr101_with_dp1_prior_regularizer.py` lacks all 3); (c) substrate-optimal engineering decision per Catalog #290 forks one of the 3 primitives without explicit waiver.

**Canonical source anchor**: CLAUDE.md "EMA - NON-NEGOTIABLE" + "eval_roundtrip - NON-NEGOTIABLE" + "Quantizr intelligence" sections + Council D EMA audit (`.omx/research/council_ema_audit_20260429.md`) + Slot CC T3 grand-council Quantizr binding revision #5 (2026-05-29).

**Canonical unwind path**: For each missing primitive, either (a) ADOPT_CANONICAL per Catalog #290 (most substrates; the primitive serves substrate-optimal engineering), or (b) FORK_PER_SUBSTRATE_OPTIMAL_ENGINEERING with explicit `# QUANTIZR_3_STACK_FORK_OK:<rationale>` same-line waiver and Catalog #325 6-step symposium content surfacing the canonical-vs-unique decision per layer, or (c) `research_only=true` opt-out per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable.

**Canonical producers**: `tac.training.EMA`, `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`, `tac.losses.kl_on_logits` (assuming canonical helper presence; not all 3 helpers necessarily exposed as `tac.losses.kl_on_logits` — sister gap in canonical helper API to be addressed in sister cascade)

**Canonical consumers**: future Class A wavelet + Class D Wyner-Ziv Catalog #325 6-step symposia (per Slot CC STRATEGIC RESET #2)

**Operator decision**: DEFER per "iterate not force" until canonical equation `quantizr_canonical_ema_kl_eval_roundtrip_stack_savings_v1` registration approved; per Catalog #344 protocol the anti-pattern registration awaits operator approval.

---

## CLAUDE.md amendment proposal (DEFERRED — operator decision via `.omx/research/claude_md_quantizr_canonical_3_stack_amendment_proposal_20260529.md`)

**Proposed amendment**: extend the existing CLAUDE.md "EMA - NON-NEGOTIABLE" + "eval_roundtrip - NON-NEGOTIABLE" sections with a sister "Quantizr canonical 3-stack — NON-NEGOTIABLE, HIGHEST EMPHASIS" section codifying:

1. Every substrate trainer at `experiments/train_substrate_*.py` (excluding `research_only=true` opt-outs) MUST honor all 3 Quantizr-canonical primitives or carry explicit per-primitive `# QUANTIZR_3_STACK_FORK_OK:<rationale>` waiver per Catalog #287 sister discipline.

2. Catalog #325 6-step symposium content for every NEW substrate scaffold MUST surface the per-primitive canonical-vs-unique decision per Catalog #290.

3. Strict preflight gate candidate `check_substrate_trainer_honors_quantizr_canonical_3_stack` (NEW Catalog # claim deferred per Slot CC STRATEGIC RESET #1 cap=1 apparatus_maintenance-per-turn for ONE WEEK; the gate would refuse substrate trainers without 3-stack adherence + waiver discipline).

**Operator decision**: DEFER per Slot CC STRATEGIC RESET #1; the CLAUDE.md amendment SHOULD be considered when Class A + Class D Catalog #325 symposia land empirical anchors to confirm the 3-stack ADOPT vs FORK per-primitive decision.

---

## Sister DISJOINT scope verification per Catalog #340

- **Slot DD** cross-PR-family deep-research subagent (Contrarian dissent revision #1): in-flight; DISJOINT scope (PR95/100/101/102/103 source-text deep-research at `.omx/research/cross_pr_family_*` + L43-L70 unenumerated canonical techniques mining)
- **Slot FF** Fridrich PR110-OPT-7 UNIWARD canonical Fridrich inverse-scorer parallel-cascade (per Fridrich binding revision #4): in-flight; DISJOINT scope (`.omx/research/pr110_opt_7_uniward_*` + canonical UNIWARD inverse-steganalysis substrate Catalog #325 6-step symposium)
- **THIS Slot scope (Slot EE)**: `.omx/research/quantizr_canonical_3_stack_*` + memory landing memo at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_ee_quantizr_*` + retroactive sweep memo + canonical equation + canonical anti-pattern candidates DEFERRED-to-operator-decision

Catalog #340 sister-checkpoint guard PROCEED at spawn-time (verified 2026-05-29T07:53Z via `.venv/bin/python tools/subagent_checkpoint.py read --subagent-id slot_ee_quantizr_audit_20260529` returning no records confirming no predecessor + sister-DISJOINT scope to in-flight Slot DD/FF).

---

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map = N/A (audit + canonical equation registration; no per-axis signal contribution)
- Hook #2 Pareto constraint = N/A (no Pareto-relevant signal)
- Hook #3 bit-allocator = N/A (no bit-allocator signal)
- Hook #4 cathedral autopilot dispatch = ACTIVE (canonical equation + anti-pattern candidates auto-discoverable per Catalog #335 `canonical_equation_lookup_consumer` + `anti_pattern_lookup_consumer` once registered)
- Hook #5 continual-learning posterior = ACTIVE (canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` per Catalog #355; auto-recalibration per Catalog #371 fires `when_3+_new_empirical_anchors_in_domain`)
- Hook #6 probe-disambiguator = ACTIVE (Class A + Class D Catalog #325 6-step symposium per-substrate cargo-cult audit IS the canonical disambiguator between ADOPT_CANONICAL vs FORK_PER_SUBSTRATE_OPTIMAL per Catalog #290)

---

## Hard constraints honored

$0 cost / NO `gh pr create` / ZERO Claude/Anthropic in PR-facing surfaces / APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110/#113 / canonical serializer per Catalog #117/#157/#174 with POST-EDIT `--expected-content-sha256` (pending commit phase) / Catalog #340 sister-checkpoint guard PROCEED (2 sister parallel: Slot DD cross-PR-family deep-research + Slot FF Fridrich PR110-OPT-7 UNIWARD; DISJOINT scope) / Catalog #206 checkpoint discipline (3 checkpoints + complete) / Catalog #287 placeholder rejection (all rationales ≥4 chars substantive) / Catalog #299 quota brake under 400 (NO new Catalog # claimed per Slot CC STRATEGIC RESET #1 self-application) / Catalog #355 + #300 + #292 + #346 + #363 council apparatus / "iterate not force" / "Forbidden premature KILL"

---

## Cross-references

- `[[t3-grand-council-strategic-reprioritization-symposium-rudin-daubechies-per-operator-4-message-cascade-directive-landed-20260529]]` (the source Slot CC T3 grand council strategic-reset symposium + Quantizr binding revision #5 source directive)
- `[[council-ema-audit-20260429]]` (canonical Council D EMA audit; empirical 5 paths correct + 8 missing baseline this audit extends)
- CLAUDE.md "EMA - NON-NEGOTIABLE, HIGHEST EMPHASIS" (Quantizr decay 0.997 canonical)
- CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS" (PR #95 inner-loop canonical)
- CLAUDE.md "Quantizr intelligence — verified competitive data (2026-04-21)" (KL T=2.0 SegNet distill canonical)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7 (bind-all-ingredients per substrate)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (per-substrate canonical-vs-fork decision per Catalog #290)
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325 6-step symposium)
- CLAUDE.md "Mission alignment — non-negotiable" (Catalog #300 Consequence 5)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (DEFER-pending-research per Catalog #307 paradigm-vs-implementation classification)
- Canonical equation `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` (17 anchors; sister at the per-T2-temperature surface)
- Canonical equation `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` (3 anchors; sister at the QAT composition surface)
- Canonical equation `ema_decay_substrate_stage_aware_v1` (sister at the EMA per-stage decay surface)
- Catalog #88 / #5 / #287 / #290 / #299 / #303 / #305 / #307 / #313 / #325 / #335 / #344 / #355 / #371

---

## Lane

`lane_slot_ee_quantizr_catalog_325_audit_20260529` L1 (impl_complete + canonical_equation_candidate + canonical_anti_pattern_candidate + memory_entry)

<!-- # HISTORICAL_SCORE_LITERAL_OK:quantizr_canonical_3_stack_audit_references_quantizr_0_33_anchor_+_council_d_ema_audit_2026_04_29_+_pr_95_canonical_per_catalog_343_canonical_pointer_dot_json_2026_05_29 -->
