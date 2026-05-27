# BUILD-1 5D canvas canonical empirical populator LANDED 2026-05-26

**Lane**: `lane_build_1_populate_5d_canvas_20260526` L1
(impl_complete + memory_entry).

**Author**: Claude Opus 4.7 (1M context) per operator routing directive
2026-05-26 verbatim *"All operator approved"* blanket + DROP-MANY+REPLACE+
COMPOSITION APPARATUS STATE AUDIT memo (commit `1f62ac788`) operator-
routable #1 + 13th OPTIMAL-TRIO standing directive
(`feedback_automated_compounding_optimal_meta_principle_standing_directive_20260526.md`).

**Mission contribution per Catalog #300**: `apparatus_maintenance`
(foundation primitive; unblocks BUILD-2 + BUILD-3 + BUILD-4 sister
subagents + Phase 4 paired-axis dispatch wave per audit memo §Phase 4
PRIORITY 2-4).

**Spend**: $0 GPU + ~75 min wall-clock.

---

## TLDR

BUILD-1 sister-subagent op-routable from the DROP-MANY+REPLACE+
COMPOSITION APPARATUS STATE AUDIT memo (`1f62ac788`) Phase 4 PRIORITY 1
landed:

- Canonical empirical populator module
  `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py`
  (~700 LOC) reading `.omx/state/master_gradient_anchors.jsonl` via
  canonical `tac.master_gradient.load_anchors_strict` (Catalog #138
  fail-closed) and emitting `PairFrameScorerGeometryCell` instances per
  the canonical scaffold contract.
- Canonical sidecar persistence at
  `.omx/state/pair_frame_scorer_geometry_lattice/<sha[:12]>_<utc>.json`
  via fcntl-locked atomic write per Catalog #131/#138/#245 canonical
  4-layer ledger pattern.
- Canonical reader `load_empirical_lattice(archive_sha256, repo_root)`
  sister of the canonical writer.
- Operator-facing CLI `tools/populate_5d_canvas_cli.py` (~210 LOC) with
  3 modes: `--list-archives`, `--archive-sha256 <sha>`,
  `--latest`.
- Test suite `src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_populator.py`
  (~600 LOC; 43 tests; 100% pass) covering helper unit tests + end-to-end
  populator + sidecar roundtrip + CLI subprocess + Catalog discipline
  regression.

Empirical verification (live `.omx/state/master_gradient_anchors.jsonl`
with 11 rows across 6 distinct archives + 3 distinct measurement axes):

```text
$ .venv/bin/python tools/populate_5d_canvas_cli.py --latest --no-sidecar
========================================================================
5D canvas populator manifest (Tier A observability-only)
========================================================================
archive_sha256: 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
cells_populated: 3
anchors_consumed: 1
anchors_skipped_non_authoritative: 2
output_path: None
```

The LATEST archive carries 3 master gradient anchors (1 contest-CPU
authoritative + 2 advisory-axis); the populator emits 3 cells (one per
`ScorerAxis` for the authoritative anchor); the 2 advisory-axis anchors
are correctly skipped per CLAUDE.md "Submission auth eval — BOTH CPU
AND CUDA" non-negotiable + Catalog #192 (sister sidecar contains
`anchors_skipped_non_authoritative: 2`).

---

## Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": every
canonical helper / META layer field / engineering pattern adoption
decision per substrate documented with rationale per the falling-rule
acceptance cascade.

| Layer | Decision | Rationale |
|---|---|---|
| Master gradient anchor reader | ADOPT canonical `tac.master_gradient.load_anchors_strict` | Catalog #138 fail-closed; canonical 4-layer ledger reader; reuse > fork |
| Contest-axis custody | ADOPT canonical `tac.master_gradient.contest_axis_authority_violation_reason` | non-authoritative anchors flow into Provenance with explicit reason rather than silently leaking |
| Provenance contract | ADOPT canonical Provenance per Catalog #323 | every cell's `catalog_323_provenance` carries `tac.provenance.builders` output; canonical apparatus contract |
| fcntl-lock pattern | ADOPT canonical pattern per `tac.deploy.modal.call_id_ledger._ledger_lock` | Catalog #131 sister; transactional `.tmp.<uuid12>` + `os.replace` per Catalog #128 |
| 5D canvas schema | ADOPT scaffold contract verbatim | BUILD-1 scope is populate + persist + read; no schema extensions (sister-disjoint with BUILD-3 Catalog #356 wire-in) |
| BUILD-2 operation generators | FORK-OUT (sister-disjoint) | parallel BUILD-2+3 sister-subagent spawn; this module does NOT call any of the 4 canonical operation generators |
| BUILD-4 Tier B promotion | FORK-OUT (sister-disjoint) | scaffold's `CONSUMER_TIER = TIER_A_OBSERVABILITY_ONLY` preserved per Catalog #357 + #341 |

---

## Observability surface (Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable" 6-facet definition:

1. **Inspectable per layer**: every populated cell carries
   `catalog_323_provenance` with archive sha + measurement_axis +
   hardware_substrate + evidence_grade.
2. **Decomposable per signal**: canvas decomposable per (pair, frame,
   scorer_axis, receiver_runtime, cpu_cuda_axis) coordinate via
   `query_cell` instance method.
3. **Diff-able across runs**: deterministic archive_sha256 keying +
   `sort_keys=True` at writer = byte-stable output for the same input.
4. **Queryable post-hoc**: canonical sidecar at
   `.omx/state/pair_frame_scorer_geometry_lattice/<sha[:12]>_<utc>.json`
   is JSON-queryable via standard tools (jq / cat / Python json.loads).
5. **Cite-able**: every cell's `catalog_323_provenance.inputs_sha256`
   cites the canonical archive sha; manifest carries `ledger_path`
   field citing the master gradient ledger source.
6. **Counterfactual-able**: per-cell `receiver_feasibility` bool +
   per-cell `predicted_byte_cost` enable downstream consumers to
   compute counterfactual ΔS estimates without re-running.

---

## 9-dimension success checklist evidence (Catalog #294)

| Dim | Status | Evidence |
|---|---|---|
| 1. UNIQUENESS | PASS | FIRST canonical empirical populator of the 5D canvas; binds master gradient anchors → `PairFrameScorerGeometryCell` with canonical Provenance |
| 2. BEAUTY + ELEGANCE | PASS | 5 public functions + 1 canonical writer + 1 canonical reader; reuses canonical helpers; ~700 LOC |
| 3. DISTINCTNESS | PASS | sister-disjoint from BUILD-2 operation generators + BUILD-3 Catalog #356 wire-in + BUILD-4 Tier B promotion |
| 4. RIGOR | PASS | Catalog #138 fail-closed load + Catalog #131 fcntl + Catalog #323 Provenance + Catalog #192 advisory-axis skip |
| 5. OPTIMIZATION-PER-TECHNIQUE | PASS | per-anchor sparse representation; no 21.6M-cell dense allocation; only cells with finite empirical signal populated |
| 6. STACK-OF-STACKS-COMPOSABILITY | PASS | output sidecar IS the canonical composability primitive BUILD-2 + BUILD-3 + BUILD-4 consume; canvas-vs-algorithm separation per design memo |
| 7. DETERMINISTIC-REPRODUCIBILITY | PASS | archive_sha256-keyed; byte-stable output via `sort_keys=True`; same input → same output bytes |
| 8. EXTREME-OPTIMIZATION-PERFORMANCE | PASS | sparse representation; canvas population O(N_anchors) where N_anchors is current ledger row count |
| 9. OPTIMAL-MINIMAL-CONTEST-SCORE | DEFERRED-pending-BUILD-2+3+4 | BUILD-1 itself is foundation; the score-lowering value lands when BUILD-2 + BUILD-3 + BUILD-4 + paired-axis dispatch wave land downstream per audit memo Phase 4 |

---

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| `measurement_axis` → `CpuCudaAxis` mapping (`[contest-CPU]` → CONTEST_CPU; `[contest-CUDA]` → CONTEST_CUDA_T4; advisory axes → None) | HARD-EARNED | CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA" non-negotiables; canonical 1:1 contest-compliant axis discipline |
| `scorer_axis` per-anchor decomposition (d_seg + d_pose + rate) | HARD-EARNED | CLAUDE.md canonical contest formula `S = 100*d_seg + sqrt(10*d_pose) + 25*rate`; `operating_point` ledger field carries components verbatim per canonical equation #36 |
| `receiver_runtime = RAW_RESIDUAL` default for empirical anchors | HARD-EARNED | master gradient anchors are computed against AS-IS archive bytes; no per-receiver compensation applied at measurement point |
| Pair-aggregate decomposition (`pair_idx=0` + `frame_idx=0`) | CARGO-CULTED-PENDING-EMPIRICAL | master gradient ledger emits archive-level aggregate `operating_point`; per-pair decomposition requires `tac.master_gradient.predict_delta_s_per_pair` per `n_pairs_used` subset measurements. BUILD-1 populates archive-aggregate; future BUILD-1-PHASE-2 sister extends to per-pair |
| `latest-row-wins` on coordinate collision | HARD-EARNED | canonical 4-layer ledger pattern (Catalog #245 modal_call_id_ledger sister); chronological-order ledger means later anchor supersedes earlier per `effective_anchor_sort_key` discipline |
| Sidecar JSON schema (one schema per populator version) | HARD-EARNED | canonical fcntl-locked JSONL append-only sister discipline at Catalog #131/#138/#245; one-file-per-(archive, utc) sidecar pattern |

---

## Predicted ΔS band (Catalog #296)

**NOT a substrate dispatch proposal**. This is a canonical populator
module that emits empirical anchors AS-IS from
`master_gradient_anchors.jsonl`. No Dykstra-feasibility check needed
per Catalog #296 because the populator does NOT compose constraints;
it READS anchored measurements and persists them verbatim.

Downstream consumers (BUILD-2 operation generators, BUILD-3 per-axis
decomposition, BUILD-4 Tier B promotion) ARE Dykstra-feasibility-bound
per their respective design memos; BUILD-1's role is foundation.

`# PREDICTED_BAND_VIBES_OK:populator_is_foundation_no_band_emission_per_design_memo_section_8`

---

## Council attendees / verdict (Catalog #300 + #346)

**T1 working-group VERDICT PROCEED** (BUILD-1 sister subagent
op-routable per audit memo §Phase 4 Priority 1; no quorum required at
T1 per Catalog #300).

**Attendees (T1; quorum N/A)**:
- Shannon LEAD (info-theory anchor: per-axis component decomposition
  per canonical contest formula)
- Dykstra CO-LEAD (alternating-projections feasibility: receiver
  feasibility lookup is canvas-side; BUILD-1 emits the raw signal)
- Daubechies CO-LEAD (wavelet multi-scale: receiver runtime hierarchy
  is canvas-side)
- Rudin CO-LEAD (interpretable ML: per-cell `as_dict` JSON round-trip
  is operator-readable)
- Carmack (engineering shortcuts: ~700 LOC + reuses canonical helpers;
  no fork)
- Assumption-Adversary (HARD-EARNED-vs-CARGO-CULTED classifier: 5
  HARD-EARNED + 1 CARGO-CULTED-PENDING-EMPIRICAL surfaced above)

**Mission alignment** (Catalog #300 §"Mission alignment"):
`apparatus_maintenance` — foundation primitive; unblocks BUILD-2 +
BUILD-3 + BUILD-4 sister subagents + Phase 4 paired-axis dispatch wave.

`council_override_invoked: false`.

---

## 6-hook wire-in declaration (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125
non-negotiable:

| Hook | Status | Implementation |
|---|---|---|
| #1 sensitivity-map contribution | **ACTIVE** | every populated cell IS a sensitivity-map entry (per-axis component contribution from master gradient ledger) |
| #2 Pareto constraint | N/A — sister-disjoint | BUILD-3 Catalog #356 wire-in (sister-disjoint parallel spawn) lands the per-axis AxisDecomposition that feeds Pareto polytope; BUILD-1 emits the raw cells without Pareto composition |
| #3 bit-allocator hook | N/A — sister-disjoint | BUILD-2 operation generators (sister-disjoint parallel spawn) emit `ExecutableCandidate.predicted_byte_cost` that the bit-allocator consumes; BUILD-1 emits per-cell `predicted_byte_cost=0` (no operation applied at measurement) |
| #4 cathedral autopilot dispatch hook | N/A — sister-disjoint | BUILD-4 Tier B promotion (sister-disjoint parallel spawn) registers the canvas as a cathedral consumer per Catalog #335; BUILD-1 preserves scaffold's `CONSUMER_TIER = TIER_A_OBSERVABILITY_ONLY` |
| #5 continual-learning posterior update | **ACTIVE** | every populator invocation emits a canonical sidecar at `.omx/state/pair_frame_scorer_geometry_lattice/`; sister subagents read this as posterior input |
| #6 probe-disambiguator | **ACTIVE** | per-cell `is_authoritative_contest_axis` + `contest_axis_authority_violation_reason` IS the canonical disambiguator between authoritative + advisory anchors per Catalog #127 sister discipline |

---

## Horizon-class declaration (Catalog #309)

`horizon_class: apparatus_foundation` — this is META-layer apparatus
maintenance, not a substrate dispatch. The horizon-class taxonomy
(PLATEAU / FRONTIER / ASYMPTOTIC) applies to downstream BUILD-2+3+4
sister subagents whose canvas-consuming operations target specific
operating points.

`# HORIZON_CLASS_DECLARATION_OK:apparatus_foundation_not_a_substrate_dispatch_proposal`

---

## Files landed

| File | LOC (incl. blanks + comments) | Purpose |
|---|---|---|
| `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py` | 568 | Canonical populator module |
| `tools/populate_5d_canvas_cli.py` | 211 | Operator-facing CLI |
| `src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_populator.py` | 622 | Test suite (43 tests; 100% pass) |
| `.omx/research/build_1_populate_5d_canvas_canonical_helper_landed_20260526.md` | this file | Landing memo |

**Total LOC committed**: ~1400 (well within BUILD-1 scope budget
~400-800 LOC for populator + CLI; tests + memo are additive
infrastructure).

---

## Canonical equation candidate (Catalog #344 status)

**FORMALIZATION_PENDING**: the canonical equation candidate
`pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1` was
QUEUED by the audit memo for registration after 3+ empirical anchors
land per Catalog #344. BUILD-1 (this lane) does NOT register a NEW
canonical equation because:

1. BUILD-1 emits empirical anchors VERBATIM from the master gradient
   ledger; the populator IS a transparent reader+persister, not a
   predictor.
2. BUILD-2+3+4 sister subagents land the canonical operation
   generators + per-axis decomposition + Tier B promotion that
   produce the predictions the canonical equation would calibrate.
3. Paired CPU+CUDA dispatch wave (audit memo Phase 4 Priority 4)
   produces the 3+ empirical anchors required per Catalog #344.

`# FORMALIZATION_PENDING:foundation_lands_first_canonical_equation_registers_after_build_2_3_4_dispatch_wave_per_catalog_344_3plus_empirical_anchors_requirement`

---

## Sister coordination + Catalog #314/#340 absorption avoidance

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #340 sister-
checkpoint guard:

- **Sister-disjoint scope verified**: my edits are confined to NEW
  files (`pair_frame_scorer_geometry_lattice_5d_canvas_populator.py`
  + `populate_5d_canvas_cli.py` + dedicated test file + this landing
  memo); NO modification of sister-active files.
- **Sister BUILD-2+3 detected**: parallel-spawn sister subagent
  `build-2-3-operation-generators-full-drop-...` is mid-flight editing
  `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py`
  (the SCAFFOLD file); my BUILD-1 module does NOT modify that scaffold
  file (it ONLY imports from it). The sister's implementations of the
  4 canonical operation generators ARE detectable in the working tree
  but are NOT staged into my commit per Catalog #340 staging discipline.
- **Sister V14-V2 / V15 UNIWARD / Cascade C' WAVE-7 / Phase 4 builder
  detected**: all are sister-disjoint (different directories +
  different files); zero overlap.
- **Catalog #287 placeholder rejection**: all `<rationale>` /
  `<reason>` placeholder literals rejected in this memo.
- **Catalog #343 hardcoded score literal compliance**: no current-
  frontier score literals embedded; historical anchors per Catalog
  #110 carry `# HISTORICAL_SCORE_LITERAL_OK` waivers in source CLAUDE.md
  references (none in this memo body).

---

## Empirical receipt: live ledger population

```text
$ .venv/bin/python tools/populate_5d_canvas_cli.py --list-archives
Distinct archive sha256 values in ledger (6):
  6bae0201fb08...  (6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf)
  7ecb0df1c462...  (7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb)
  87ec7ca5f2f3...  (87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5)
  9cb989cef519...  (9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4)
  b83bf3488625...  (b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e)
  f174192aeadf...  (f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd)
```

```text
$ .venv/bin/python tools/populate_5d_canvas_cli.py --latest --no-sidecar
========================================================================
5D canvas populator manifest (Tier A observability-only)
========================================================================
archive_sha256: 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
cells_populated: 3
anchors_consumed: 1
anchors_skipped_non_authoritative: 2
output_path: None
```

```text
$ PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \\
    src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_populator.py
============================== 43 passed in 0.55s ==============================
```

---

## Operator-routable next steps

Per audit memo §Phase 4 prioritization (next steps consume the BUILD-1
foundation this lane lands):

### PRIORITY 2: BUILD-2 + BUILD-3 sister subagents (operation generators + Catalog #356 wire-in)

- **TECHNIQUE**: 4 canonical operation generators per design memo
  §DELIVERABLE 1 (`generate_full_drop_starts` / `generate_repair_starts`
  / `generate_masked_starts` / `generate_feathered_starts`) consume
  the populated canvas this BUILD-1 lands.
- **STATUS**: parallel-spawn sister subagent ALREADY active per
  `.omx/state/subagent_progress.jsonl` empirical receipt.
- **TIME**: PRE-DISPATCH; $0 paid GPU + ~4-8h wall-clock.

### PRIORITY 3: BUILD-4 sister subagent (Catalog #357 Tier B promotion)

- **TECHNIQUE**: package `src/tac/cathedral_consumers/pair_frame_scorer_geometry_lattice_consumer/__init__.py`
  per Catalog #335 + #341 + #356 + #357 canonical contract; consumes
  populated canvas via `load_empirical_lattice`.
- **TIME**: PRE-DISPATCH; $0 paid GPU + ~2-4h wall-clock.

### PRIORITY 4: PAIRED CPU+CUDA DISPATCH wave (canonical equation #344 anchors)

- **TECHNIQUE**: paid Modal dispatch of top-3 ExecutableCandidates per
  (operation, cpu_cuda_axis) = 4 × 3 × 2 = 24 dispatches; canonical
  4-arm paired auth_eval per CLAUDE.md "Submission auth eval — BOTH
  CPU AND CUDA" + per-substrate symposium per Catalog #325.
- **TIME**: DISPATCH; ~$4-15 total per audit memo Phase 4 Priority 4.

---

## Cross-references

- DROP-MANY+REPLACE+COMPOSITION APPARATUS STATE AUDIT (this lane's
  parent routing): `.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md`
  (commit `1f62ac788`)
- 5D canvas SCAFFOLD: `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py`
- 5D canvas design memo:
  `.omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md`
- Codex v1 row-based sister:
  `src/tac/optimization/pair_frame_scorer_geometry_lattice.py`
  (commit `4ed9eb905`)
- Sister META-LIFT family (cathedral consumers consuming the canvas):
  - META-LIFT-1 commit `60acdc2d2` cross-substrate master-gradient
    analyzer
  - META-LIFT-2 commit `da803dd30` Pareto polytope unified solver
  - META-LIFT-4 commit `6fbd7ec7f` UNIWARD invariant enumerator
- Just-saved 13th OPTIMAL-TRIO standing directive:
  `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_automated_compounding_optimal_meta_principle_standing_directive_20260526.md`
- 8th MLX-first standing directive:
  `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526.md`
- 10th apples-to-apples directive (sister)
- 11th ORDER directive (sister)
- 12th canonicalization directive (sister)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
- CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in
- CLAUDE.md "Max observability — non-negotiable" 6-facet definition
- Catalog #131 / #138 / #245 canonical 4-layer ledger pattern
- Catalog #192 advisory-axis non-promotion guard
- Catalog #287 placeholder-rationale rejection
- Catalog #290 / #294 / #296 / #300 / #303 / #305 / #309 design-memo
  discipline cluster
- Catalog #335 / #341 / #356 / #357 cathedral consumer canonical
  contract cluster
- Catalog #340 sister-checkpoint guard
- Catalog #343 frontier-pointer model
- Catalog #344 canonical equations + models registry
- Catalog #346 canonical council roster

Lane: `lane_build_1_populate_5d_canvas_20260526` L1
(impl_complete + memory_entry).
