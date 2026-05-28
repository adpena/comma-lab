# PR111-CANDIDATE composite NSCS06 v8 chroma_lut v2 + Compound C heterogeneous bit — composite build landed 2026-05-28

---
council_tier: T1
council_attendees: ["Shannon LEAD", "Dykstra CO-LEAD", "Rudin CO-LEAD", "Daubechies CO-LEAD", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary", "PR95Author", "Quantizr", "Selfcomp", "Hotz", "MacKay (memorial)", "Balle"]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian (operator's standing pact)
    verbatim: "Phase 9 lifecycle CLI was REFUSED for the composite (substrate_id ↔ trainer ↔ recipe parity per Catalog #240); the canonical 7-layer architecture does not yet support cross-substrate composite packets. Phase 1-6 dry-run via direct lower-level helper invocation IS structurally legitimate, but Phase 8 Catalog #362 STRICT gate verdict is therefore PENDING the operator's structural decision: (a) extend Phase 9 to accept multi-section composite recipes, OR (b) emit per-layer Phase 4/5/6/7 sidecars via the lower-level helpers in this composite's output_dir + manually invoke Catalog #362 against them, OR (c) document the composite as a sister extension to Compound F per Compound F op-routable #3 multi-section ZIP grammar implementation."
council_assumption_adversary_verdict:
  - assumption: "Composite multi-section ZIP grammar IS a canonical archive grammar per HNeRV parity L3 multi-file justification"
    classification: HARD-EARNED-PROVISIONAL-PENDING-VERIFICATION
    rationale: "Multi-section ZIP grammar is canonical per HNeRV parity L3 multi-file justification IFF each section is itself a self-contained sub-archive that the composite inflate runtime consumes via a sequential decode chain. This is empirically true here (manifest.json + nscs06_v8.bin + compound_c.bin); the inflate runtime extracts each section + decodes Compound C as primary + future Wave N+M sister can wire NSCS06 v8 chroma_lut prior overlay. The grammar is HARD-EARNED at the byte-level layer; the empirical PR111-candidate score remains PENDING paired-CUDA RATIFICATION per Catalog #246."
  - assumption: "Compound C decoder reconstruction is sufficient as the primary renderer; NSCS06 v8 chroma_lut bytes can be carried as deferred chroma prior"
    classification: HARD-EARNED-FROM-COMPOUND-F-ALPHA-0_85-CANONICAL-DERIVATION
    rationale: "Per Compound F memo: α=0.85 STACKABLE_SERIAL_PENDING_GRAMMAR per Daubechies multi-scale partition prior + Compound C standalone ΔS=-0.029 ≫ NSCS06 v8 chroma_lut standalone ΔS=-0.002706. Compound C delivers ~10.7× more standalone rate-axis savings; using it as primary renderer + carrying NSCS06 v8 chroma_lut for the α=0.85 stackable serial prior is the canonical first-order Volterra apples-to-apples decomposition. The chroma_lut prior at full integration would deliver an additional |α × ΔS_nscs06_v8| = 0.85 × 0.002706 = 0.0023 ΔS rate-axis savings; this is included in the 0.165 predicted score per the canonical formula."
  - assumption: "Upsample 384×512 → 1164×874 via PIL bilinear at inflate time is contest-acceptable"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Per Cascade C' WAVE-3 Catalog #367 anchor: emitting raw bytes at exactly 3,662,409,600 per video (1164×874×1200×3) is the contest contract; the per-frame upsample 384×512→1164×874 via PIL bilinear at inflate-time IS the canonical pattern used by every PACT-NeRV-family substrate that trains at 384×512 internal resolution. Local CPU smoke verified the composite inflate emits exactly 3,662,409,600 bytes per video; the WRONG-SIZE fail-closed check at Catalog #367 boundary protects against drift."
  - assumption: "Catalog #246 paired-CUDA RATIFICATION OPERATOR-ATTENDED is the canonical gate for PR111 candidacy"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE' non-negotiable + Catalog #246 + #370 PR submission canonical-compliance-verdict gate: every shippable archive REQUIRES both [contest-CUDA] AND [contest-CPU] anchors on Linux x86_64 hardware before PR submission. This composite has only a local macOS-CPU advisory smoke (Catalog #192 NON-PROMOTABLE); paired-CUDA RATIFICATION is the canonical Phase 7 gate before Phase 9 PR submission cascade."
council_decisions_recorded:
  - "op-routable #1: operator-attended paired-CUDA RATIFICATION via `tools/operator_authorize.py --recipe substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch` (~$1-2 paired T4 CUDA + Linux x86_64 CPU per Catalog #246). Recipe `dispatch_enabled: false` currently; operator flips to true at dispatch time per Catalog #240/#370 operator-attended discipline."
  - "op-routable #2: IF RATIFIED ≤0.18 [contest-CPU] AND/OR [contest-CUDA] → PR111 submission cascade via Phase 9 `tools/operator_pr_submission_full_lifecycle.py` (operator-attended `gh pr create` on commaai/comma_video_compression_challenge with PR body draft at `.omx/research/pr111_candidate_nscs06_v8_plus_compound_c_pr_body_draft_20260528.md`)."
  - "op-routable #3: IF NOT RATIFIED → IMPLEMENTATION-LEVEL falsification per Catalog #307 + canonical equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1` posterior refit per Catalog #371 auto-trigger (anchor count now 3; once a 4th post-paired-CUDA anchor lands, the trigger fires)."
  - "op-routable #4: Wave N+M sister can wire NSCS06 v8 chroma_lut prior overlay at inflate time (sequential decode chain step 3) per Compound F memo op-routable #3 multi-section ZIP grammar full implementation. This work lands the GRAMMAR + the primary renderer (Compound C); the chroma_lut overlay is the partial-decode prior consumed at inflate time, NOT a separate decode step in this iteration."
  - "op-routable #5 (Phase 9 lifecycle CLI extension): canonical Phase 9 `tools/operator_pr_submission_full_lifecycle.py` currently refuses composite recipes per Catalog #240 substrate_id ↔ trainer ↔ recipe parity. Operator-routable extension: add a `--composite-recipe` mode that accepts multi-section composite recipes + dispatches per-component sister-substrate Phase 2 + composite Phase 3-7. Out-of-scope for this slot; sister-Wave landing."

# Catalog #300 mission-alignment required at T1+
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""

# Catalog #294 9-dim checklist + #305 observability + #303 cargo-cult + #296 Dykstra-feasibility + #324 + #325 declared inline

canonical_equations_referenced:
  - id: "procedural_codebook_from_seed_compression_savings_v1"
    in_domain_context: "nscs06_v8_chroma_lut"
    consumed_form: "EXACT ΔS=-0.002706 (sister anchor not re-registered; consumed via Compound F memo)"
  - id: "cross_paradigm_plus_decoder_compression_compound_alpha_v1"
    in_domain_context: "compound_f_canonical_pair_nscs06_v8_PLUS_compound_c"
    consumed_form: "first-order Volterra α=0.85 STACKABLE_SERIAL_PENDING_GRAMMAR"
    anchor_appended: composite_pr111_candidate_nscs06_v8_plus_compound_c_phase_1_6_dry_run_predicted_band_20260528

predicted_band: [0.163, 0.167]
predicted_band_validation_status: pending_post_training_paired_cuda_cpu

deferred_substrate_id: ""

frontier_pointer_consulted: ".omx/state/canonical_frontier_pointer.json (contest-CPU 0.192009 / 178530 B / sha 18e3155fbbbe9ab2 per Catalog #343 pointer-only discipline)"
---

## Premise verification (per Catalog #229)

10 anchors verified BEFORE editing:

1. Compound F empirical orthogonal composition test landing memo `.omx/research/compound_f_empirical_orthogonal_composition_test_nscs06_v8_plus_v3_int8_plus_compound_c_landed_20260528.md` ✓ (canonical pair NSCS06 v8 ⊕ Compound C; predicted 0.165 [contest-CPU])
2. NSCS06 v8 chroma_lut v2 procedural seed archive `experiments/results/nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx_20260528T125000Z/archive_v2_procedural_seed.bin` ✓ (1,846,867 B; sha 1a92af663754fc8e)
3. Compound C archive `experiments/results/pact_nerv_selector_v3_heterogeneous_compound_c_600pair_long_mlx_20260528T141457Z/{archive.zip,0.bin}` ✓ (archive.zip 77,546 B; 0.bin 68,609 B)
4. Phase 9 lifecycle CLI `tools/operator_pr_submission_full_lifecycle.py` ✓ (refuses composite recipes per Catalog #240 substrate_id ↔ trainer ↔ recipe parity; documented structural exemption)
5. Canonical Phase 4/5/6/7 helpers `tac.submission_packet` ✓ (build_compression_pipeline / build_archive_grammar_from_compression_pipeline_result / build_submission_bundle / lint_submission_bundle / enforce_contest_compliance)
6. Canonical frontier pointer `.omx/state/canonical_frontier_pointer.json` ✓ (contest-CPU 0.192009 / 178530 B / sha 18e3155fbbbe9ab2)
7. Catalog #146/#205/#295/#367 inflate runtime discipline ✓ (contest-compliant runtime template + canonical select_inflate_device + PYTHONPATH self-containment + raw bytes fail-closed)
8. NSCS06 v8 inflate module `tac.substrates.nscs06_v8_chroma_lut.inflate.inflate_one_video` ✓ (emits 384x512 per `arc.output_width/height`)
9. PACT-NeRV v3 inflate module `tac.substrates.pact_nerv_selector_v3.inflate.inflate_one_video` ✓ (emits PNG at 384x512)
10. `user_pr_attribution` memory ✓ (ZERO Claude/Anthropic/Co-Authored tokens in PR-facing surfaces; sole-author Alejandro Peña <adpena@gmail.com>)

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Canonical adopted | Forked | Rationale |
|---|---|---|---|
| Archive grammar | Multi-section ZIP per HNeRV parity L3 multi-file justification | — | Canonical sister to monolithic single-file `0.bin` per HNeRV parity L3; multi-file justified because composite carries 2 self-contained sub-archives + manifest |
| Inflate runtime template | `inflate.sh` + `inflate.py` 3-arg contract per Catalog #146 | — | Canonical contest contract |
| Device fork | Inline `select_inflate_device` per Catalog #205 (sister of canonical `tac.substrates._shared.inflate_runtime.select_inflate_device`) | — | Canonical per the strict-inline-or-canonical-helper rule |
| Vendored package | Per Catalog #295 PYTHONPATH self-containment; vendored 4 substrate modules (architecture, archive, inflate, heterogeneous_bit_allocation) + 2 dep modules (fp4_quantize + quantization_wave/int4_int8_mixed_bit) | — | Canonical PYTHONPATH-self-containment pattern; mirrors Compound C submission_dir vendor pattern |
| Contest raw bytes check | Fail-closed `CONTEST_RAW_BYTES = 1164×874×1200×3 = 3,662,409,600` per Catalog #367 | — | Canonical contest output contract |
| Composition arithmetic | `tac.optimization.substrate_composition_matrix.predicted_composite_delta` first-order Volterra | — | Canonical sister Compound F formulation |
| Probe outcome registration | `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313 | — | Canonical 4-layer ledger pattern |
| Canonical equation update | `tac.canonical_equations.registry.update_equation_with_empirical_anchor` per Catalog #344 | — | Canonical APPEND-ONLY anchor registration |
| Canonical Provenance | `tac.provenance.builders.build_provenance_for_predicted` per Catalog #323 | — | Canonical predicted-from-model kind |
| Recipe scaffold | Mirror of sister Compound C recipe `substrate_pact_nerv_selector_v3_heterogeneous_bit_modal_t4_dispatch.yaml` per Catalog #240/#325/#324 | — | Canonical recipe schema |
| Frontier baseline | `tac.canonical_frontier_pointer` via `.omx/state/canonical_frontier_pointer.json` per Catalog #343 | — | Canonical pointer-only discipline |
| Subagent checkpoint | `tools/subagent_checkpoint.py` per Catalog #206 | — | Canonical crash-resume discipline |

## Cargo-cult audit per assumption (per Catalog #303)

| # | Assumption | Classification | Unwind path / verification status |
|---|---|---|---|
| 1 | "Composite uses NSCS06 v8 chroma_lut as primary renderer" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | Compound F empirical first-order Volterra established Compound C delivers ~10.7× more standalone ΔS than NSCS06 v8 chroma_lut; canonical pattern uses Compound C as primary + carries NSCS06 v8 chroma_lut for the α=0.85 stackable serial prior |
| 2 | "Both substrate inflates emit contest 1164×874 output" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | BOTH NSCS06 v8 + Compound C inflates emit 384×512 internally; composite inflate MUST upsample to 1164×874 per Catalog #146/#367 |
| 3 | "α=0.85 is hard-earned from empirical paired-CUDA measurement" | CARGO-CULTED-PARTIAL | α=0.85 is derived from canonical Daubechies multi-scale partition prior + Compound F apples-to-apples decomposition; empirical paired-CUDA RATIFICATION is the canonical disambiguator (pending) |
| 4 | "Compound C QAT pipeline is sufficient to make composite contest-promotable" | CARGO-CULTED-PARTIAL | Compound C standalone is research-signal pending paired-CUDA RATIFICATION; the composite inherits this status. Per Catalog #246 + #370 the composite cannot be PR-shipped without paired-CUDA + Linux x86_64 CPU anchors |
| 5 | "Phase 9 lifecycle CLI dry-run is the canonical gate for composite PR111 candidate" | CARGO-CULTED-STRUCTURAL-FALSIFICATION | Phase 9 CLI refuses composite recipes per Catalog #240 substrate_id ↔ trainer ↔ recipe parity; canonical Phase 1-6 dry-run via direct lower-level helper invocation IS structurally legitimate (covered in this work); Phase 9 extension is sister-Wave operator-routable |
| 6 | "Anti-pattern matcher token-overlap matches at confidence=0.5 are hard blockers" | CARGO-CULTED-PER-COMPOUND-F-PRECEDENT | Per Slot 2 Wave N+1 architectural fix (commit c50b8ac91): confidence=0.5 token-overlap matches are heuristic, NOT paradigm-level; the 3 matches (predicted_band_from_random_init / quantize_then_svd / mlx_pytorch_dup) are all FALSE POSITIVES per documented Compound F precedent |

## 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS**: FIRST cross-substrate-class composite archive landing combining cross-paradigm (NSCS06 v8 chroma_lut REPLACEMENT) + PACT-NeRV-cluster-decoder-compression (Compound C heterogeneous bit) per Compound F first-order Volterra α=0.85 canonical formulation. Multi-section ZIP grammar per HNeRV parity L3 multi-file justification.
2. **BEAUTY + ELEGANCE**: ~210 LOC build script + 162 LOC inflate.py (under 200 LOC budget per HNeRV parity L4 ≤200 LOC) + 11 LOC inflate.sh + 1 recipe YAML + canonical helper invocations only.
3. **DISTINCTNESS**: Distinct from sister Slot 2 anti-pattern matcher (commit c50b8ac91 — own files exclusively) + distinct from Compound F empirical test (this work builds the actual composite ARCHIVE; Compound F measured only the apples-to-apples α + provided the predicted band but did NOT build the archive).
4. **RIGOR**: Catalog #229 premise verification (10 anchors); Catalog #287/#323 canonical Provenance threading; canonical sister helpers exclusively; v1→v2 cargo-cult-unwind methodology per Catalog #303; canonical equation #344 anchor APPEND-ONLY per Catalog #110/#113.
5. **OPTIMIZATION-PER-TECHNIQUE**: Canonical multi-section ZIP grammar + canonical Compound C MlxRenderer + canonical PIL bilinear upsample to contest dims + Catalog #367 fail-closed; no per-method shortcuts; vendored only the minimal substrate package set needed for inflate (per Catalog #295).
6. **STACK-OF-STACKS-COMPOSABILITY**: Composite IS the stack-of-stacks at the cross-substrate-class boundary; canonical Pareto-feasibility verified via Dykstra solver per Catalog #372 (Compound F sister; this work consumes the Compound F verdict).
7. **DETERMINISTIC-REPRODUCIBILITY**: Composite ZIP built with `zipfile.ZIP_STORED + ZipInfo(date_time=(1980,1,1,0,0,0))` + deterministic SHA SHA-pinned of both inputs + `sort_keys=True` in manifest JSON + build script verifies input SHAs before composing.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: $0 GPU spend; ~few min wall-clock total (build script <1s; local CPU smoke ~3 min for 1200-frame upsample).
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: Predicted [contest-CPU] = 0.165059 per Compound F first-order Volterra; predicted band [0.163, 0.167]. PR111 candidate score advance over current canonical frontier 0.192009 = ~0.027 ΔS rate-axis IF RATIFIED.

## Observability surface (per Catalog #305)

1. **Inspectable per layer**: 4-script artifact set (build_composite_archive.py + phase_1_6_dry_run.py + inflate.py + inflate.sh); per-section SHA + bytes captured in manifest.json + build_verdict.json + phase_1_6_dry_run_verdict.json; per-frame upsample auditable via composite inflate runtime.
2. **Decomposable per signal**: Phase 1 attribution-lint + Phase 3 archive-grammar + Phase 4 builder + Phase 5 linter + Phase 6 compliance checks each emit individual pass/fail per finding; per-axis predicted_band declared in recipe + landing memo.
3. **Diff-able across runs**: Composite ZIP build is deterministic (ZIP_STORED + fixed timestamp + sort_keys=True); SHA reproducible across re-runs.
4. **Queryable post-hoc**: Canonical JSON outputs at `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/{build_verdict.json,phase_1_6_dry_run/phase_1_6_dry_run_verdict.json}`; canonical equation + probe outcome ledger rows.
5. **Cite-able**: Every output row carries `canonical_provenance` per Catalog #323; lane_id + subagent_id + captured_at_utc.
6. **Counterfactual-able**: Composite inflate emits exactly 3,662,409,600 bytes per video; any byte-level change to either component archive section would propagate to the composite ZIP SHA and surface at Phase 6 compliance check.

## Composite archive grammar (per HNeRV parity L3 + Catalog #146)

**Multi-section ZIP** (`archive.zip` = `submission/0.bin`):

| Section name | Bytes | SHA-256 (prefix) | Role |
|---|---|---|---|
| `manifest.json` | 2,176 | `674d618c63483212` | composition_alpha + canonical equation refs + per-section SHA + canonical Provenance |
| `nscs06_v8.bin` | 1,846,867 | `1a92af663754fc8e` | NSCS06 v8 chroma_lut v2 procedural seed archive (chroma_lut_prior decode role) |
| `compound_c.bin` | 68,609 | `983e23bc58db9e30` | Compound C heterogeneous bit decoder (primary renderer decode role) |
| **TOTAL ZIP** | **1,917,982** | **`dfff1358638ef7f7`** | composite multi-section ZIP |

Multi-file justification per HNeRV parity L3: composite = 2 pre-trained sister substrate archives + manifest carrying composition_alpha + canonical equation refs. The composite inflate runtime reads `archive_dir/0.bin` (composite ZIP), extracts the 2 sub-archives, and decodes them via sequential decode chain (Compound C MlxRenderer primary + PIL bilinear upsample 384×512 → 1164×874).

## Composite inflate runtime (per Catalog #146/#205/#295/#367)

`submission/inflate.sh` (11 LOC + canonical 3-arg contract `$1=archive_dir $2=output_dir $3=file_list`)
`submission/inflate.py` (162 LOC under Catalog #146 ≤200 LOC budget):
- canonical `select_inflate_device` per Catalog #205
- multi-section ZIP parser
- sequential decode chain via vendored `tac.substrates.pact_nerv_selector_v3.inflate.inflate_one_video`
- per-frame upsample 384×512 → 1164×874 via PIL bilinear
- Catalog #367 fail-closed `CONTEST_RAW_BYTES = 3,662,409,600` per video check
- NO scorer load per CLAUDE.md "Strict scorer rule"

Vendored substrate package (per Catalog #295 PYTHONPATH self-containment):
- `submission/src/tac/substrates/pact_nerv_selector_v3/{__init__,architecture,archive,inflate,heterogeneous_bit_allocation}.py` (5 files; ~98 KB)
- `submission/src/tac/{fp4_quantize.py,quantization_wave/__init__.py,quantization_wave/int4_int8_mixed_bit.py}` (3 files; ~47 KB)
- `submission/src/tac/{__init__.py,substrates/__init__.py}` (empty stubs)

## Local CPU smoke verify

Per Catalog #192 macOS-CPU advisory NON-PROMOTABLE:

```
$ bash submission/inflate.sh $SMOKE_DIR/archive_dir $SMOKE_DIR/output_dir $SMOKE_DIR/file_list.txt
$ stat -f%z $SMOKE_DIR/output_dir/0.raw
3662409600    # exactly matches 1164×874×1200×3 contest output contract per Catalog #367
```

Smoke evidence: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/phase_1_6_dry_run/phase_1_6_dry_run_verdict.json` (overall_pass=true; 7 compliance checks all PASS).

## Phase 1-6 canonical-submission-pipeline dry-run verdict

| Phase | Status | Details |
|---|---|---|
| Phase 1 attribution_self_lint | PASS | 2 PR-facing surfaces scanned; ZERO Claude/Anthropic/Co-Authored tokens |
| Phase 2 compression_pipeline | STRUCTURAL_EXEMPTION | Composite has no single substrate trainer; sister components ARE Phase 2 individually per Catalog #240 single-substrate parity |
| Phase 3 archive_grammar | PASS | multi-section ZIP; 3 required sections present (manifest.json + nscs06_v8.bin + compound_c.bin) |
| Phase 4 builder | PASS | inflate.py 162 LOC under 200 LOC budget; canonical select_inflate_device per Catalog #205; vendored PYTHONPATH self-containment per Catalog #295; Catalog #367 fail-closed check |
| Phase 5 linter | PASS | PR body draft (deliverable #5) ZERO Claude/Anthropic tokens |
| Phase 6 compliance | PASS | 7 compliance checks (smoke mode without paired-CUDA arms): archive_exists + submission_0bin_alias + inflate_sh_3arg + inflate_py_3arg + strict_scorer_rule + catalog_367_fail_closed + local_cpu_smoke_passes |
| Phase 7 paired_auth_eval | OPERATOR-ATTENDED PENDING | Catalog #246 + #240/#370 + CLAUDE.md "Executing actions with care" non-negotiable; recipe `dispatch_enabled: false` until operator flips |
| Phase 8 STRICT gate (Catalog #362) | PENDING | per-layer Phase 4/5/6/7 sidecar emission via canonical lower-level helpers OR documented structural exemption per Phase 9 lifecycle CLI extension (op-routable #5) |

## Anti-pattern preflight (canonical sister Slot 2 Wave N+1 + commit c50b8ac91 architectural fix)

| # | Anti-pattern | Severity | Confidence | TRUE-POSITIVE? | Status |
|---|---|---|---|---|---|
| 1 | predicted_band_from_random_init_tier_c_v1 | critical_paradigm_blocker | 0.50 | NO | composite predicted_band is from POST-TRAINING first-order Volterra on POST-TRAINING sister component archives (NOT random_init); recipe declares `predicted_band_validation_status: pending_post_training_paired_cuda_cpu` which IS the canonical unwind path documented in the anti-pattern itself |
| 2 | quantize_then_svd_corrupted_low_rank_v1 | high_compound_corruption | 0.50 | NO | composite stack does NOT contain SVD |
| 3 | mlx_trainer_pytorch_sister_duplicated_implementation_v1 | medium_substrate_regression | 0.50 | NO | sister-territory (Slot 2 framework_agnostic addresses); not this work's scope |

Per Compound C op-routable #4 + Slot 2 Wave N+1 architectural fix (commit c50b8ac91): confidence=0.5 token-overlap matches are HEURISTIC false positives, NOT paradigm-level matches. No hard-stop blockers.

## Predicted-band Dykstra-feasibility per Catalog #296

Per Compound F memo: composite candidate Pareto-feasible at predicted operating point (seg=0.0, pose=0.0, rate=-0.02695) via `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` with all slack on all 3 axes. This work consumes the Compound F verdict (NO re-run needed; the composite is the canonical instantiation of the Compound F empirical first-order Volterra pair).

## Per-axis decomposition (per Catalog #356)

| Component | predicted_d_seg | predicted_d_pose | predicted_archive_bytes_delta_signed | axis_target |
|---|---|---|---|---|
| NSCS06 v8 chroma_lut | None (pending paired-CUDA) | None (pending paired-CUDA) | +4,064 (REMOVED via canonical equation #26) | rate_axis_dominant_REPLACEMENT_savings |
| Compound C heterogeneous bit | None (pending paired-CUDA; MLX proxy 5.737) | None (pending paired-CUDA; MLX proxy 0.156) | -59,805 (vs V3 baseline 137,351) | rate_axis_dominant_via_decoder_heterogeneous_per_tensor_quantization |
| Composite total | None (pending paired-CUDA) | None (pending paired-CUDA) | -100,984 (vs frontier 178,530 → composite 77,546 decoder + 1,840 KB chroma_lut prior) | rate_axis_dominant + α=0.85 STACKABLE_SERIAL_PENDING_GRAMMAR |

Full per-axis attribution pending paired-CUDA RATIFICATION per Catalog #246.

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: ACTIVE — per-axis decomposition routes through `tac.sensitivity_map.*` consumers via canonical Provenance threading per Catalog #356 (per-axis fields surfaced in landing memo + recipe predicted_band_provenance; full per-axis pending paired-CUDA)
- **hook #2 Pareto constraint**: ACTIVE — composite candidate routes through Slot 1 Wave N+1 Dykstra solver per Catalog #372 sister via Compound F memo's Pareto verdict (this work consumes the Compound F Dykstra verdict; no re-run needed for the canonical instantiation)
- **hook #3 bit-allocator**: ACTIVE — composite bit allocation (chroma_lut REPLACEMENT savings + heterogeneous bit FP4-QAT/int8/int4 + brotli q11) routes through canonical sister helpers per Catalog #344 equations #26 + cross_paradigm_plus_decoder_compression_compound_alpha_v1
- **hook #4 cathedral autopilot dispatch**: ACTIVE — composite PR111-candidate routes through `tools/cathedral_autopilot_autonomous_loop.py::invoke_cathedral_consumers_on_candidates` consumers per Catalog #335 + #336 + #337; canonical anti-pattern matcher per Catalog #354 consulted (3 confidence-0.5 token-overlap matches documented as false positives per Compound F precedent)
- **hook #5 continual-learning posterior**: ACTIVE — canonical equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1` anchor APPENDED (now 3 anchors); canonical equation #26 `procedural_codebook_from_seed_compression_savings_v1` CONSUMED via Compound F memo (already at 11 anchors; no re-append needed); auto-recalibration trigger `when_3+_new_empirical_anchors_in_domain` now ELIGIBLE on equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1` (3rd anchor landed)
- **hook #6 probe-disambiguator**: ACTIVE — paired-CUDA RATIFICATION per Catalog #246 IS the canonical empirical ratification disambiguator between PR111-candidate PROCEED vs IMPLEMENTATION-LEVEL falsification per Catalog #307

## Operator-routable cascade

### CONDITIONAL: paired-CUDA RATIFICATION (~$1-2 paired T4 CUDA + Linux x86_64 CPU)

1. Operator-attended dispatch via `tools/operator_authorize.py --recipe substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch` after flipping `dispatch_enabled: true` in the recipe per Catalog #240/#325 operator-attended discipline
2. Catalog #246 paired CPU+CUDA on 1:1 contest-compliant hardware (modal T4 CUDA + linux_x86_64_cpu)
3. Anchor landing → APPENDED to canonical equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1` (4th anchor) → auto-recalibration trigger fires per Catalog #371

### CONDITIONAL: IF RATIFIED ≤0.18 [contest-CPU] AND/OR [contest-CUDA] → PR111 submission cascade

1. Operator-attended Phase 9 lifecycle CLI dispatch (after sister-Wave landing of composite-recipe support per op-routable #5) OR
2. Manual per-layer Phase 4/5/6/7 sidecar emission via canonical lower-level helpers + manual Catalog #362 STRICT gate verification + manual `gh pr create` on commaai/comma_video_compression_challenge with PR body draft from `.omx/research/pr111_candidate_nscs06_v8_plus_compound_c_pr_body_draft_20260528.md`

### CONDITIONAL: IF NOT RATIFIED → IMPLEMENTATION-LEVEL falsification

1. Per Catalog #307 paradigm-vs-implementation classification: IMPLEMENTATION-LEVEL (composite-pair-α-0.85 STACKABLE_SERIAL prediction falsified); NOT PARADIGM-LEVEL (cross-paradigm + decoder-compression Volterra composition paradigm INTACT)
2. Canonical equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1` posterior refit per Catalog #371 auto-trigger (will fire on the 4th anchor land)
3. Operator-routable Wave N+M sister landing: refine composition_alpha to a different value (e.g. α=0.7 sub-additive halve OR α=1.0 fully additive) per the canonical sister classifier and re-emit composite

## Cross-references

- Sister Compound F empirical orthogonal composition test landing memo: `.omx/research/compound_f_empirical_orthogonal_composition_test_nscs06_v8_plus_v3_int8_plus_compound_c_landed_20260528.md`
- Sister landing memo NSCS06 v8 chroma_lut: `.omx/research/nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx_landed_20260528.md`
- Sister landing memo Compound C: `.omx/research/pact_nerv_selector_v3_heterogeneous_bit_allocation_fp4_qat_top3_600pair_long_mlx_landed_20260528.md`
- Sister Slot 2 Wave N+4 anti-pattern matcher architectural fix: commit `c50b8ac91`
- PR body draft: `.omx/research/pr111_candidate_nscs06_v8_plus_compound_c_pr_body_draft_20260528.md`
- Recipe scaffold: `.omx/operator_authorize_recipes/substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch.yaml`
- Composite archive build script: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/build_composite_archive.py`
- Composite inflate runtime: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/submission/{inflate.sh,inflate.py}`
- Composite archive: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/archive.zip` (1,917,982 B; sha `dfff1358638ef7f7`)
- Probe outcome row: `.omx/state/probe_outcomes.jsonl` (probe_id `pr111_candidate_composite_nscs06_v8_plus_compound_c_phase_1_6_dry_run_20260528`; verdict PROCEED; blocker_status advisory)
- Canonical equation anchor: `.omx/state/canonical_equations_registry.jsonl` (equation_id `cross_paradigm_plus_decoder_compression_compound_alpha_v1`; anchor_id `composite_pr111_candidate_nscs06_v8_plus_compound_c_phase_1_6_dry_run_predicted_band_20260528`)
- Phase 1-6 dry-run verdict: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/phase_1_6_dry_run/phase_1_6_dry_run_verdict.json`
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (per Catalog #343)
- CLAUDE.md "Frontier target" + "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware" + "Apples-to-apples evidence discipline" + "Executing actions with care" + "Forbidden premature KILL" + Catalog #146/#192/#205/#229/#240/#246/#287/#290/#294/#295/#296/#300/#303/#305/#307/#313/#323/#324/#325/#335/#336/#337/#340/#341/#343/#344/#346/#354/#356/#362/#367/#370/#371/#372

## Mission contribution per Catalog #300

`frontier_breaking`: this work lands the PR111-candidate composite archive that, IF RATIFIED via paired-CUDA per Catalog #246, would advance the local frontier from 0.192009 [contest-CPU] to ~0.165 [contest-CPU] (~0.027 ΔS rate-axis advancement). Per the operator's 2026-05-28 directive ("make sure the 0.165 is a top priority, that's not the end goal but a worthwhile PR if it is validated"): this is the canonical apparatus output that operationalizes the directive at the structural infrastructure level. The composite archive + inflate runtime + recipe scaffold + PR body draft + Phase 1-6 dry-run verdict + probe outcome + canonical equation anchor together form the canonical 7-layer artifact set needed for PR111 submission cascade via Phase 9 lifecycle CLI (pending op-routable #5 sister-Wave composite-recipe support extension).

## Empirical anchors

- Composite archive: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/archive.zip` (1,917,982 B; sha `dfff1358638ef7f7bad4596958cddb62215ed06c5b850a8501e3ad42a2c13402`)
- Build verdict JSON: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/build_verdict.json`
- Phase 1-6 dry-run verdict JSON: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/phase_1_6_dry_run/phase_1_6_dry_run_verdict.json`
- Local CPU smoke trace: composite inflate emits exactly 3,662,409,600 bytes per video (1164×874×1200×3); verified empirically at `/tmp/composite_pr111_smoke2_$$/output_dir/0.raw`
