---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Wyner
  - Atick-Redlich
  - Tishby-memorial
  - PR95Author
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: |
      The per-pair PoseNet-output Y stand-in measurement is honest about
      being a DETERMINISTIC stand-in rather than a real PoseNet forward
      (the prompt-required canonical Catalog #311 Atick-Tishby-Wyner
      route requires real per-pair PoseNet output at compress time, which
      requires CUDA dispatch that is intentionally OUT OF SCOPE per the
      $0 GPU Slot 2 cap). The verdict per Catalog #307 is
      IMPLEMENTATION_LEVEL_FALSIFICATION_PER_PAIR_Y_STAND_IN_ALSO_FALSIFIED
      because BOTH the 4 canonical sources AND the per-pair stand-in
      yield 0.000218% density on PR101 fp16 bytes. The PROCEED_WITH_REVISIONS
      verdict reflects that the substrate is structurally honest about
      what was measured (stand-in NOT real PoseNet) and the next-step
      reactivation paths are operator-routable per the canonical recipe
      scaffold. The PARADIGM (Wyner 1976 R(D|Y)) is INTACT.
  - member: Assumption-Adversary
    verbatim: |
      The L0 design memo's CARGO-CULTED critique #2 (per-pair Y derivation
      would yield meaningful density on entropy-flat fp16 state_dict bytes)
      is now EMPIRICALLY VERIFIED FALSIFIED via the deterministic stand-in
      measurement. This is a SECOND-SURFACE falsification class registered
      as anti-pattern #15. Sister classification: per-pair structural surface
      goes HARD-EARNED-FALSIFIED-AT-STAND-IN-PROXY-ONLY (real PoseNet density
      remains untested per the $0 GPU envelope). Reactivation requires either
      real PoseNet pre-compute (operator-attended L2) OR non-prefix Y
      derivation OR cross-substrate composition Y.
council_assumption_adversary_verdict:
  - assumption: "per-pair PoseNet-output Y stand-in (deterministic ego-motion-conditioned) emulates real PoseNet density"
    classification: HARD-EARNED-FALSIFIED-AT-STAND-IN-PROXY-ONLY
    rationale: |
      L1 ALTERNATIVE Y SURFACE measurement 2026-05-28 on PR101 fp16 raw bytes
      (sha256=79b804d9a5839eb3; 457916 B; Y=14400 B = 600 pairs * 6 dims *
      float32) yielded prefix density 0.000218% — IDENTICAL to canonical Y
      baseline. The stand-in surface ALSO falsified at prefix-detector layer.
      Real PoseNet density remains untested per $0 GPU envelope.
  - assumption: "the prefix-detector implementation is the appropriate test for Y-derivability density"
    classification: CARGO-CULTED
    rationale: |
      The prefix-detector misses cross-byte structure. Sister lzma ratio
      0.4215 on PR101 fp16 (vs sister prober anchor 0.217-0.228) suggests
      meaningful redundancy structure exists but the prefix-detector cannot
      exploit it. Non-prefix Y derivation (substring overlap) is the
      canonical alternative path.
  - assumption: "Wyner 1976 R(D|Y) source-coding-with-side-information theorem"
    classification: HARD-EARNED-PARADIGM-INTACT
    rationale: |
      Wyner & Ziv 1976 IEEE Trans IT-22(1):1-10 is the canonical info-theory
      bound; this landing does not falsify the theorem. The IMPLEMENTATION
      (prefix-detector + stand-in Y at decoder-state-dict surface) is
      falsified per Catalog #307; PARADIGM intact.
  - assumption: "subagent_spawn_without_head_state_premise_verification is a recurring discipline failure"
    classification: HARD-EARNED-RECURRING
    rationale: |
      2 empirical receipts within Wave N+5 alone (Compound C STAND_DOWN
      e61ea93b0 + framework_agnostic STAND_DOWN today). Anti-pattern #13
      is_actively_recurring=True (>=2 falsifications). Discipline gate
      via Catalog #229 PV is the structural extinction path.
  - assumption: "predecessor_working_tree_uncommitted_handoff is a low-severity bookkeeping pattern"
    classification: HARD-EARNED-EMPIRICALLY-CONFIRMED
    rationale: |
      1 empirical receipt (framework_agnostic resolved at canonical-
      serializer auto-commit 5d38bf9df). Severity LOW because the
      resolution path (canonical-serializer auto-commit) is well-known +
      Catalog #117 enforces serializer usage. Anti-pattern #14 is the
      machine-readable canonicalization.
council_decisions_recorded:
  - "Part A (3 NEW anti-patterns) LANDED: #13 subagent_spawn_without_head_state_premise_verification_v1 (discipline; 2 falsifications; is_actively_recurring=True) + #14 predecessor_working_tree_uncommitted_handoff_v1 (discipline; 1 falsification) + #15 wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1 (diagnosis; 1 falsification + Wave N+7 second-surface receipt LOGGED via append_empirical_falsification per Catalog #344 sister discipline)."
  - "Part B (Op-routable #5) LANDED at MLX-LOCAL $0 GPU: per-pair PoseNet-output Y stand-in measurement on real PR101 fp16 bytes; density 0.000218% (IDENTICAL to canonical baseline; SECOND-SURFACE FALSIFICATION per anti-pattern #15); WZPSC01 archive 193467 B (sha256=aefc1dca2d831cb5); byte-identical roundtrip per Catalog #105/#139/#220/#272 no-op detector invariant."
  - "Canonical equation #344 entry registered: equation_id='wyner_ziv_per_pair_posenet_output_y_pose_axis_savings_v1'; form='R(D|Y_per-pair) - R(D) ≈ -(density/100) * |source| * 25 / 37545489'; empirical anchor: density 0.000218% on PR101 fp16; residual 0.999782 (massive falsification against 1% threshold)."
  - "Catalog #313 probe-outcomes ledger: 2 DEFER rows landed for substrate=wyner_ziv_pipeline_stage_codec — (1) per-pair PoseNet-output Y stand-in MLX-LOCAL measurement; (2) anti-pattern #15 SECOND-SURFACE FALSIFICATION class registration. Both 30-day staleness window; reactivation requires real PoseNet pre-compute OR non-prefix Y OR cross-substrate composition Y per the 3 priority-ordered reactivation paths."
  - "Operator-attended paired-CUDA recipe scaffold landed at .omx/operator_authorize_recipes/substrate_wyner_ziv_pipeline_stage_codec_per_pair_posenet_output_y_modal_paired_dispatch.yaml: dispatch_enabled=false + research_only=true per Catalog #240/#370; predicted_band_validation_status=pending_post_training per Catalog #324; 5 operator_required_attestations declared."
  - "Anti-pattern matcher verification: Catalog c50b8ac91 architectural fix is intact. Anti-pattern #15 fires at high confidence on real WZ stack spec (recurrence conditions matched: 'wyner_ziv_pipeline_stage_codec measuring density on raw_fp16 state_dict' + 'all 4 canonical Y sources yield density << 1% threshold'). Anti-pattern #13 fires at 0.7 confidence on spawn-without-PV stack spec."
  - "Compound F α extension queued: IF Wave N+8 real PoseNet pre-compute yields density >= 1%, composition_matrix gets a new row at composition_alpha pending empirical measurement; OPERATOR-GATED."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: 2026-06-27T17:05:00Z
deferred_substrate_id: wyner_ziv_pipeline_stage_codec
related_deliberation_ids:
  - wyner_ziv_pipeline_stage_codec_l1_long_mlx_600pair_landed_20260528
  - wyner_ziv_pipeline_stage_codec_l0_scaffold_landed_20260528
  - wyner_ziv_pipeline_stage_codec_primitive_landed_20260517
  - grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517
horizon_class: asymptotic_pursuit
predicted_band_validation_status: pending_post_training_real_posenet_pre_compute
substrate_alias: wyner_ziv_pipeline_stage_codec
---

# Anti-pattern registry expansion (#13/#14/#15) + Wyner-Ziv Op-routable #5 landed 2026-05-28

**Subagent id**: `slot2_wave_n7_anti_patterns_plus_wyner_ziv_op5_20260528`
**Lane id**: `lane_anti_pattern_registry_expansion_13_14_15_plus_wyner_ziv_op5_20260528`
**Landing UTC**: 2026-05-28T17:05:00Z
**HEAD sha (pre-landing PV)**: `1faf05951` (Wave N+6 TRIPLE composition + Wave N+6 matcher fix at `c50b8ac91`)
**Authors**: Claude Opus 4.7 (1M context) <noreply@anthropic.com> + Alejandro Peña <adpena@gmail.com>

Per operator NON-NEGOTIABLE 2026-05-28 *"ensure all negative findings
audited"* + Wyner-Ziv FALSIFICATION reactivation criteria, this Slot 2
Wave N+7 landing delivers two compound packages in DISJOINT scope from
Slot 1 Phase 9 CLI:

## Part A: 3 NEW canonical anti-patterns

* **#13 `subagent_spawn_without_head_state_premise_verification_v1`** —
  discipline_anti_pattern (NEW paradigm class); severity
  medium_substrate_regression; 2 empirical receipts (Wave N+5 Slot 1
  Compound C STAND_DOWN commit `e61ea93b0` + Wave N+5 Slot 2
  framework_agnostic STAND_DOWN); `is_actively_recurring=True`.
* **#14 `predecessor_working_tree_uncommitted_handoff_v1`** —
  discipline_anti_pattern; severity low_implementation_inefficiency;
  1 empirical receipt (framework_agnostic resolved at canonical-
  serializer auto-commit `5d38bf9df`).
* **#15 `wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1`** —
  diagnosis_anti_pattern; severity medium_substrate_regression;
  1 empirical receipt (commit `6f5eabf30` original Wave N+6 landing) +
  1 SECOND-SURFACE receipt LOGGED via this Wave N+7 measurement.

All 3 register cleanly via `register_anti_pattern` + persisted to the
canonical fcntl-locked JSONL at `.omx/state/canonical_anti_patterns_registry.jsonl`.
Sister `EmpiricalFalsification` rows appended via `append_empirical_falsification`
per Catalog #344 APPEND-ONLY discipline. 83/83 anti-pattern tests pass.

**NEW paradigm class**: `PARADIGM_DISCIPLINE = "discipline_anti_pattern"`
added to `VALID_PARADIGM_CLASSES` taxonomy (was 8, now 9). Sister
canonical surfaces (cathedral consumer + matcher + cadence audit) auto-
discover via Catalog #335.

## Part B: Wyner-Ziv Op-routable #5 (per-pair PoseNet-output Y derivation)

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #6
strict-scorer-rule + Catalog #320 + the $0 GPU Slot 2 cap, the canonical
test object for the per-pair Y surface is a **deterministic ego-motion-
conditioned per-pair pose Y stand-in** per Atick-Redlich 1990 +
Catalog #311 Atick-Tishby-Wyner triple — NOT a real PoseNet forward.

The L1 trainer extension lives in
`src/tac/substrates/wyner_ziv_pipeline_stage_codec/trainer.py`:

* NEW `--per-pair-posenet-output-y` argparse flag + 3 companion knobs
  (`--per-pair-posenet-output-y-num-pairs` default 600 per contest
  seq_len=2 batching; `--per-pair-posenet-output-y-pose-dim` default 6
  per PoseNet head; `--per-pair-posenet-output-y-dtype` default float32).
* NEW canonical helpers `_derive_per_pair_posenet_output_y_stand_in(...)`
  + `_measure_per_pair_posenet_output_y_density(...)`. The derivation
  uses sin/cos at 1/3/7 frequencies (coprime decorrelated cooperative-
  receiver basis per Catalog #311); deterministic per-pair byte-identical
  across calls (Wyner 1976 reconstructibility invariant verified by 7
  new tests).
* MLX-first per CLAUDE.md 8th MLX-FIRST standing directive 2026-05-26;
  numpy-portable fallback per the inflate-portability invariant.

**Empirical anchor (real PR101 fp16 state_dict bytes, sha256=`79b804d9a5839eb3`,
457916 B)**:

| Y source | Y bytes | prefix_len | density |
|---|---|---|---|
| math_constants (canonical) | 811 | 0 | 0.000000% |
| torch_defaults (canonical) | 1024 | 0 | 0.000000% |
| ImageNet (canonical) | 24 | 0 | 0.000000% |
| Comma2k19 (canonical) | 4096 | 1 | 0.000218% |
| **per_pair_posenet_output_y_stand_in (NEW)** | **14400** | **1** | **0.000218%** |
| per_pair_posenet_output_y_stand_in_fp64 (NEW) | 28800 | 1 | 0.000218% |

**Verdict per Catalog #307**:
`IMPLEMENTATION_LEVEL_FALSIFICATION_PER_CATALOG_307_PER_PAIR_Y_STAND_IN_ALSO_FALSIFIED_OP_ROUTABLE_5_SECOND_SURFACE`.
Both the 4 canonical Y sources AND the per-pair PoseNet-output Y
stand-in surface yield density 0.000218% — **4 orders of magnitude
below the 1% threshold** per op-routable #4. The PARADIGM (Wyner 1976
R(D|Y); decoder-side PoseNet as Y per Catalog #311 Atick-Tishby-Wyner
triple) is STILL INTACT. The IMPLEMENTATION at the prefix-detector
surface against the deterministic ego-motion stand-in is empirically
falsified. Per CLAUDE.md "Forbidden premature KILL without research
exhaustion": **DEFERRED-PENDING-research**, NOT killed. Anti-pattern
#15 SECOND-SURFACE FALSIFICATION receipt LOGGED.

**WZPSC01 archive**: 193467 B (sha256=`aefc1dca2d831cb5`); byte-identical
roundtrip verified per Catalog #105/#139/#220/#272 no-op-detector
invariant.

**Wall-clock**: 0.55s total ($0 MLX-LOCAL per Catalog #192/#317/#341
non-promotable markers).

## Sister deliverables

* **Canonical equation #344 entry**:
  `wyner_ziv_per_pair_posenet_output_y_pose_axis_savings_v1` registered
  via `register_canonical_equation` + empirical anchor appended via
  `update_equation_with_empirical_anchor` (residual 0.999782; massive
  falsification against 1% threshold).
* **Catalog #313 probe outcomes ledger**: 2 DEFER rows registered for
  `substrate=wyner_ziv_pipeline_stage_codec`:
  (1) `wyner_ziv_per_pair_posenet_output_y_stand_in_mlx_local_pr101_fp16`
  with `metric_value=0.000218 < threshold=1.0`;
  (2) `wyner_ziv_prefix_y_density_anti_pattern_15_second_surface_falsification_class`
  with `metric_value=1.0 < threshold=2.0` (actively_recurring threshold).
* **Operator-attended paired-CUDA recipe scaffold** at
  `.omx/operator_authorize_recipes/substrate_wyner_ziv_pipeline_stage_codec_per_pair_posenet_output_y_modal_paired_dispatch.yaml`
  with `dispatch_enabled: false` + `research_only: true` per Catalog
  #240/#370; `predicted_band_validation_status: pending_post_training`
  per Catalog #324; 5 operator_required_attestations declared.
* **Matcher verification (NO REGRESSION)**: Catalog `c50b8ac91`
  architectural fix preserved. Anti-pattern #15 fires at high confidence
  on the WZ stack spec with recurrence conditions matched; anti-pattern
  #13 fires at 0.7 confidence on the spawn-without-PV stack spec.
* **109 tests pass**: 83 anti-pattern tests (was 81; updated test from
  12→15 + 1 new check) + 26 Wyner-Ziv substrate tests (was 19; 7 new
  tests for per-pair Y derivation/measurement/argparse).

## Convergence comparison vs Wave N+6 sister anchor

| Aspect | Wave N+6 commit `6f5eabf30` | Wave N+7 (THIS) |
|---|---|---|
| Y surface | 4 canonical sources (Comma2k19/ImageNet/etc.) | Per-pair PoseNet-output Y stand-in (NEW) |
| Density on PR101 fp16 | 0.000218% (Comma2k19) | 0.000218% (per-pair stand-in; ALSO falsified) |
| Verdict | IMPLEMENTATION_LEVEL_FALSIFICATION | IMPLEMENTATION_LEVEL_FALSIFICATION_PER_PAIR_Y_STAND_IN_ALSO_FALSIFIED |
| Archive bytes | 193467 B | 193467 B (no change; Comma2k19 still wins by tie-break) |
| Roundtrip byte-identical | True | True |
| Reactivation paths queued | 3 (per-pair Y / non-prefix / cross-substrate) | 2 remaining (non-prefix / cross-substrate) — per-pair Y EXHAUSTED at stand-in |

**Wave N+7 evidence is HONEST about what was tested** (deterministic
stand-in NOT real PoseNet) so the reactivation path #1 in the sister
landing memo is NOT fully closed; it requires real PoseNet pre-compute
via operator-attended L2 CUDA dispatch per Catalog #246.

## Reactivation criteria (remaining, priority-ordered)

1. **Real PoseNet pre-compute Y (Op-routable #5 canonical path)**:
   ~$1-5 CUDA dispatch to pre-compute per-pair PoseNet output on contest
   video 0.mkv; ship per-pair pose tensor in archive; decoder reproduces
   at inflate WITHOUT loading PoseNet per Catalog #6. Operator-attended.
2. **Non-prefix Y derivation (substring overlap)**: extend primitive
   `_detect_y_derivable_prefix` to support substring overlap. ~$0
   MLX-LOCAL + primitive extension cost.
3. **Cross-substrate composition Y**: derive Y from sister substrate's
   already-shipped bytes (e.g. FEC6 archive as Y for PR101). ~$0
   MLX-LOCAL measurement; non-promotable until paired CUDA+CPU.

## 6-hook wire-in declaration per Catalog #125

* **Hook #1 sensitivity-map = ACTIVE**: per-anti-pattern severity feeds
  the sensitivity-map consumer; per-pair PoseNet-output Y density
  feeds the pose-axis sensitivity per the Wyner-Ziv pose-axis canonical
  equation.
* **Hook #2 Pareto constraint = ACTIVE**: 3 NEW anti-patterns add
  polytope exclusion regions via Catalog #373 Layer 5 (the Pareto solver
  consumes anti-pattern severity-weighted exclusions).
* **Hook #3 bit-allocator = ACTIVE**: anti-pattern #15 canonical
  unwind path routes the bit-allocator from prefix-Y (FALSIFIED) to
  per-pair PoseNet-output Y (queued behind operator-attended L2).
* **Hook #4 cathedral autopilot dispatch = ACTIVE**: 3 NEW anti-patterns
  auto-discovered via Catalog #335; cathedral consumer
  `anti_pattern_lookup_consumer` surfaces them in next-cycle ranking.
* **Hook #5 continual-learning posterior = ACTIVE**: 3 NEW anti-patterns
  + 4 empirical falsifications + 1 canonical equation + 1 canonical
  equation empirical anchor + 2 probe-outcomes ledger DEFER rows all
  registered in canonical posterior surfaces.
* **Hook #6 probe-disambiguator = ACTIVE**: per-pair PoseNet-output Y
  stand-in IS the canonical disambiguator for the alternative Y surface
  per anti-pattern #15 canonical_unwind_path; second-surface falsification
  receipt directly disambiguates the reactivation path priority.

## Discipline anchors

* **Catalog #229** PV: read full state of canonical_anti_patterns + Wyner-Ziv
  landing memo + sister registries before edit.
* **Catalog #117 / #157 / #174** canonical-serializer + POST-EDIT
  `--expected-content-sha256` for all 4 edited files at commit time.
* **Catalog #206** subagent crash-resume discipline: 3 checkpoint emissions.
* **Catalog #110 / #113** APPEND-ONLY HISTORICAL_PROVENANCE: no mutation
  of sister memos or canonical_anti_patterns_registry.jsonl prior payload
  (3 register_anti_pattern + 4 append_empirical_falsification + 1
  register_canonical_equation + 1 update_equation_with_empirical_anchor +
  2 register_probe_outcome events all appended as NEW rows).
* **Catalog #131 / #138** fcntl-locked JSONL discipline at registry +
  ledger write surfaces.
* **Catalog #146 / #205 / #295 / #367** substrate inflate runtime
  discipline preserved (the Op-routable #5 extension only adds the
  per-pair Y derivation; the existing roundtrip invariant is verified
  by 26 tests).
* **Catalog #170-#244** substrate-trainer discipline preserved.
* **Catalog #192** macOS-MLX non-promotable: all artifacts carry
  `evidence_grade='macOS-MLX research-signal'` + `score_claim=False` +
  `promotable=False`.
* **Catalog #287** placeholder-rationale rejection: every waiver token
  in the new recipe + landing memo carries substantive rationale ≥4 chars.
* **Catalog #292 / #300 / #346** per-deliberation assumption surfacing
  + v2 frontmatter + roster complete=True (12 attendees: 4 inner
  co-leads + 6 sister members + 2 specialist seats Atick-Redlich +
  Tishby-memorial).
* **Catalog #294 / #296 / #303 / #305** design-memo discipline:
  9-dim checklist + predicted-band Dykstra feasibility + cargo-cult
  audit + observability surface all preserved.
* **Catalog #307** paradigm-vs-implementation classification: explicit
  in verdict_kind + verdict_message + landing memo.
* **Catalog #311** ego-motion-conditioned per-pair Y derivation per
  Atick-Tishby-Wyner triple.
* **Catalog #313** probe-outcomes ledger: 2 DEFER rows landed.
* **Catalog #323** canonical Provenance umbrella: all anchors + recipe +
  artifact JSON carry canonical Provenance.
* **Catalog #340** sister-checkpoint guard: PROCEED at every serializer
  invocation (no Slot 1 collision; DISJOINT scope verified).
* **Catalog #341** Tier A non-promotable markers in cathedral consumer
  routing.
* **Catalog #344** canonical equation registry: NEW equation registered
  + empirical anchor appended.
* **Catalog #371** orphan auto-trigger stub prevention: the new canonical
  equation is wired into producer/consumer paths (no orphan).
* **Catalog #356** per-axis decomposition: pose-axis attribution
  documented in landing memo.
* **Catalog #372 / #373** Layer 3+5 Dykstra + anti-pattern Pareto polytope
  exclusion: 3 NEW anti-patterns feed Layer 5.

## Operator-routable next actions

1. **Land paired-CUDA L2 with real PoseNet pre-compute (Op-routable #5
   canonical path)** when operator-attended bandwidth permits: invoke
   the recipe at
   `.omx/operator_authorize_recipes/substrate_wyner_ziv_pipeline_stage_codec_per_pair_posenet_output_y_modal_paired_dispatch.yaml`
   per Catalog #246 paired CUDA+CPU. Predicted cost: $1-5 + per-substrate
   symposium per Catalog #325 14-day window.
2. **OR land non-prefix Y derivation primitive extension** (~$0
   MLX-LOCAL) as the lower-risk reactivation path #2.
3. **OR land cross-substrate composition Y measurement** (~$0
   MLX-LOCAL) as the lower-risk reactivation path #3 (use FEC6
   archive bytes as Y for PR101 per sister prober anchor 0.47 score-
   savings on `pr101_state_dict` + `pr106_state_dict`).

Per CLAUDE.md "Forbidden premature KILL": NONE of the 3 reactivation
paths is closed by this Wave N+7 landing. The substrate is
DEFERRED-PENDING-research; the per-pair Y stand-in surface is
EXHAUSTED but the real PoseNet pre-compute remains untested.

Lane: `lane_anti_pattern_registry_expansion_13_14_15_plus_wyner_ziv_op5_20260528` L1
(impl_complete + memory_entry + canonical_equations_anchor +
probe_outcomes_ledger + paired_cuda_recipe_scaffold).
