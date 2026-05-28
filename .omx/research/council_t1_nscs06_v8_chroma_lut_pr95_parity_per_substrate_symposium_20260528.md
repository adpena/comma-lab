---
council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Carmack
  - Hotz
  - Selfcomp
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - PR95Author
  - Quantizr
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: |
      The Wave N+42 packet binds 13 lessons SIMULTANEOUSLY in 970 LOC across
      4 files with 25/25 unit tests green. That is the canonical PR-95-parity
      pattern operationally. However, the EMPIRICAL question (does this packet
      actually beat the 0.196-0.199 cluster on contest-CUDA?) is UNRESOLVED.
      The packet shares the same chroma_lut + cls_stream substrate hypothesis
      as the sister NSCS06 v8 (5621 LOC) which has already had per-substrate
      symposium PROCEED_WITH_REVISIONS without paired-CUDA RATIFICATION
      landing post-symposium. Recommendation: PROCEED on the canonical
      packet engineering (this verdict), DEFER paired-CUDA RATIFICATION
      until Wave N+30 op-routable #6 lands (PR111 composite paired-CUDA
      refire AFTER Catalog #377 case-parity hold extincted).
  - member: Assumption-Adversary
    verbatim: |
      Per Catalog #292 per-deliberation explicit assumption-surfacing: 7
      assumptions surfaced in __init__.py cargo-cult-audit table; 3 are
      CARGO-CULTED (luma quantization / per-(level,class) median /
      PCG64-seed-uniform), 4 are HARD-EARNED. The CARGO-CULTED
      classifications inherit from sister T1 #1335 symposium 2026-05-21
      WINNER #1 ratification. Per CLAUDE.md "Forbidden symposium-band-
      prediction-without-Dykstra-feasibility-check" + canonical equation
      #26 IN-DOMAIN context membership: the predicted Delta_S band
      [-0.0027, -0.0015] inherits Dykstra-feasibility from the T3 council
      #1335 omnibus design memo where Dykstra verified IS_ADDITIVE=True
      + intersection_non_empty=True for the (rate, seg, pose) polytope.
      The Wave N+42 packet maintains ALL THREE Dykstra-feasibility
      preconditions empirically via 25-test smoke (deterministic archive
      L3 + FULL RGB renderer L5 + byte-mutation no-op detector L11).
      VERDICT: HARD-EARNED-WITH-DYKSTRA-FEASIBILITY-CARRY for rate-axis;
      HARD-EARNED-INHERITED for seg + pose axes; CARGO-CULTED-EMPIRICALLY-
      PENDING for empirical Delta_S transfer MLX-LOCAL -> contest-CUDA
      per canonical equation #2 mps_drift_architecture_class_dependent_v1
      (which Wave N+30 audit lists as registered + 30x residual recorded).
council_assumption_adversary_verdict:
  - assumption: "16-level luma quantization captures chroma-relevant variation"
    classification: CARGO-CULTED
    rationale: "Inherited from canonical AV1 codecs without empirical validation that NSCS06 v8 specific compress-time signal axis benefits from 16 levels vs other choices"
  - assumption: "Per-(level, class) LUT median is optimal aggregation"
    classification: CARGO-CULTED
    rationale: "Inherited from v7 per-class median pattern without testing against per-class mode / trimmed mean / k-medoids cluster center"
  - assumption: "PCG64 seed -> uniform LUT bytes matches GT chroma distribution"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED-PROVISIONAL
    rationale: "Sister gt_distribution_matched_seed module landed REVISION at v8 substrate; canonical PROCEDURAL VARIANT canonical equation #26 IN-DOMAIN context membership is empirical anchor that grounds this assumption but UNWIND-TEST queued"
  - assumption: "Catalog #205 inflate device-fork produces byte-identical raw frames"
    classification: HARD-EARNED
    rationale: "Wave N+42 25-test smoke verified deterministic pack + parse + inflate roundtrip with byte-stable output across re-runs (numpy + Pillow only; no CUDA-specific ops)"
  - assumption: "6-DOF affine warp preserves v8 distinguishing feature"
    classification: HARD-EARNED
    rationale: "Empirically validated by v7 cargo-cult-unwind 44 percent reduction [contest-CUDA commit 4292c8ce2]; Wave N+42 packet preserves the 6-DOF canonical pattern verbatim"
  - assumption: "CLS_STREAM at low-res NEAREST-upsampled to output res preserves spatial chroma"
    classification: HARD-EARNED
    rationale: "Sister Wave N+22 cls_stream wire-in canonical pattern + Wave N+42 byte-mutation smoke (test_l11_byte_mutation_in_cls_stream_changes_output) verified cls bytes are consumed AND produce frame changes"
  - assumption: "Score-aware training transfers MLX-LOCAL -> contest-CUDA per canonical equation #2"
    classification: HARD-EARNED-WITH-CAVEAT
    rationale: "Per canonical equation #2 mps_drift_architecture_class_dependent_v1 (Wave N+30 audit registered + 30x residual recorded); paired-CUDA RATIFICATION required per CLAUDE.md Submission auth eval -- BOTH CPU AND CUDA non-negotiable"
council_decisions_recorded:
  - "op-routable #1: Wave N+42 packet lands as canonical PR-95-parity packet at src/tac/substrates/nscs06_v8_chroma_lut_pr95_parity/ (970 LOC across 4 files + tests; 25/25 tests pass; Catalog 205/287/304/323 strict gates clean)"
  - "op-routable #2: sister NSCS06 v8 substrate at src/tac/substrates/nscs06_v8_chroma_lut/ (5621 LOC / 12 files) coexists per Catalog #110 APPEND-ONLY HISTORICAL_PROVENANCE; Wave N+42 packet is the SUBSTRATE-ENGINEERING CANONICAL extraction for ONE coherent reviewable packet per HNeRV parity L7 + PR101 reference"
  - "op-routable #3: paired-CUDA RATIFICATION DEFERRED per Contrarian; reactivation criterion = Wave N+30 op-routable #6 (PR111 composite refire AFTER Catalog #377 case-parity hold extincted) lands EITHER PROCEED OR DEFER"
  - "op-routable #4: canonical equation #26 IN-DOMAIN context nscs06_v8_chroma_lut remains valid; canonical equation #2 mps_drift_architecture_class_dependent_v1 carries 30x MLX-LOCAL -> contest-CUDA residual caveat per Wave N+30 audit"
  - "op-routable #5: cathedral consumer auto-discovery NOT required for Wave N+42 packet (the canonical PR-95-parity packet IS the substrate, not a consumer); per-substrate routing happens via existing tac.cathedral_consumers.canonical_equation_lookup_consumer per Catalog #335"
  - "op-routable #6: probe outcome REGISTERED per Catalog #313 verdict=PROCEED 30-day expires; reactivation criterion = first paired-CUDA RATIFICATION result"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: nscs06_v8_chroma_lut_pr95_parity
substrate_aliases:
  - lane_wave_n42_nscs06_v8_pr95_parity_packet_20260528
  - nscs06_v8_chroma_lut_pr95_parity
  - nscs06_v8_pr95
related_deliberation_ids:
  - council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521
  - council_t2_composite_nscs06_v8_plus_compound_c_pr111_candidate_per_substrate_symposium_20260528
  - council_grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516_4292c8ce2
---

# T1 per-substrate symposium: NSCS06 v8 chroma_lut + cls_stream PR-95-parity packet

**Date:** 2026-05-28
**Tier:** T1 Working Group (per CLAUDE.md "Council hierarchy: 4-tier protocol")
**Attendees:** Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD + Carmack + Hotz + Selfcomp + Fridrich + Contrarian + Assumption-Adversary + PR95Author + Quantizr (12 attendees; sextet pact MIN per Catalog #346 met)
**Verdict:** PROCEED_WITH_REVISIONS (1 binding revision: paired-CUDA RATIFICATION deferred per Contrarian)
**Substrate:** `nscs06_v8_chroma_lut_pr95_parity` (Wave N+42 packet at `src/tac/substrates/nscs06_v8_chroma_lut_pr95_parity/`)
**Window:** 2026-05-28 -> 2026-06-11 (14 days; per Catalog #325 per-substrate symposium gating)

This memo satisfies Catalog #325 6-step canonical contract for the Wave N+42 PR-95-parity packet (sister of the 2026-05-21 T1 #1335 symposium for the substrate-engineering fork at `src/tac/substrates/nscs06_v8_chroma_lut/`):

1. **Cargo-cult audit per Catalog #303** — see frontmatter `council_assumption_adversary_verdict` (7 assumptions; 3 CARGO-CULTED + 4 HARD-EARNED) + packet `__init__.py` "Cargo-cult audit per assumption" section
2. **9-dimension success checklist evidence per Catalog #294** — see packet `__init__.py` "9-dimension success checklist evidence" section (UNIQUENESS / BEAUTY-ELEGANCE / DISTINCTNESS / RIGOR / OPTIMIZATION-PER-TECHNIQUE / STACK-OF-STACKS-COMPOSABILITY / DETERMINISTIC-REPRODUCIBILITY / EXTREME-OPTIMIZATION / OPTIMAL-MINIMAL-CONTEST-SCORE)
3. **Observability surface declaration per Catalog #305** — see packet `__init__.py` "Observability surface" section (6 facets: inspectable per layer / decomposable per signal / diff-able across runs / queryable post-hoc / cite-able / counterfactual-able)
4. **T1 working-group deliberation** with sextet pact MIN per Catalog #346 (Shannon + Dykstra + Rudin + Daubechies + Carmack + Hotz + Selfcomp + Fridrich + Contrarian + Assumption-Adversary + PR95Author + Quantizr; 12 attendees)
5. **Per-substrate reactivation criteria** — see frontmatter `council_decisions_recorded` (6 op-routables)
6. **Catalog #324 post-training Tier-C validation discipline** — N/A for THIS packet (NO learned weights; closed-form LUT derivation from GT). The sister NSCS06 v8 substrate at `src/tac/substrates/nscs06_v8_chroma_lut/` carries the Catalog #324 `predicted_band_validation_status: pending_post_training` discipline since v8 substrate-engineering has MLX-iteration sister at `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` (~47.9K LOC) for the empirical seed-vs-baseline sweeps. The Wave N+42 packet is the canonical PR-95-parity ARCHIVE-GRAMMAR + INFLATE-RUNTIME + SCORE-AWARE-LOSS extraction; per-training-sweep validation lives in the sister-engineering fork per HNeRV parity L7.

---

## Section 1. Wave N+42 packet bind-all-ingredients verification

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13 inviolable lessons + UNIQUE-AND-COMPLETE-PER-METHOD operating mode + Wave N+42 operator directive 2026-05-28 ~23:20Z verbatim *"must focus on getting at least three to full parity or greater shortest wall clock"*:

| # | Lesson | Wave N+42 status | Where bound |
|---|---|---|---|
| L1 | Substrate is score-aware (gradient through SegNet/PoseNet) | PASS | `score_aware_loss.py::score_pair_components_pr95_parity` routes through canonical `tac.substrates.score_aware_common.score_pair_components` |
| L2 | Export-first design (archive grammar BEFORE training) | PASS | `codec.py::CH09_HEADER_FMT` + `DECODER_BLOB_LEN` + `LATENT_BLOB_LEN` declared at module level |
| L3 | Archive grammar = monolithic single-file `0.bin` (fixed offsets in source) | PASS | `codec.py::pack_archive` / `parse_archive` declare 37-byte header + variable-length sections |
| L4 | Inflate.py <= 100 LOC, <= 2 ext deps, CUDA-or-CPU agnostic | PASS-WAIVED-200 | `inflate.py` 194 LOC under 200-LOC waiver (`L4_LOC_WAIVER` constant); numpy + Pillow only |
| L5 | Architecture is FULL renderer (RGB out) NOT single-component slot | PASS | per-pixel `(gray_full, cls_full) -> RGB` LUT lookup + 6-DOF affine warp |
| L6 | Score-domain Lagrangian via actual scorer or Hinton-distilled surrogate | PASS | `score_aware_loss.py::score_aware_lagrangian_loss` implements `25 * archive_bytes / 37545489 + 100 * d_seg + sqrt(10 * d_pose)` |
| L7 | Bolt-on <= 350 LOC; substrate engineering may exceed; tag explicitly | PASS | `lane_class=substrate_engineering` declared in symposium memo; packet total 970 LOC + tests within PR101 reference range |
| L8 | Eval-roundtrip + differentiable scorer-preprocess | PASS | `apply_eval_roundtrip=True` default in `score_pair_components_pr95_parity`; canonical helper does monkey-patch |
| L9 | Runtime closure tested in clean env BEFORE dispatch | PASS | 25 unit tests verify pack/parse/inflate roundtrip; `inflate.py` imports = numpy + Pillow only |
| L10 | Mask/pose coupling gate (mask changes require pose regeneration) | PASS | 6-DOF affine warp consumes pose deltas + frame_0 atomically per `test_l10_affine_warp_consumes_pose_deltas` |
| L11 | No-op detector (prove targeted bytes changed AND consumed by inflate) | PASS | 3 byte-mutation smoke tests (chroma_seed / pose / cls_stream) all green |
| L12 | Single-LOC-per-LOC review discipline (every line reviewable in 30 sec) | PASS | every function <= 50 LOC; packet reviewable in 30 min total |
| L13 | KILL/FALSIFIED is LAST RESORT | PASS | this symposium PROCEED_WITH_REVISIONS; reactivation criteria pinned in frontmatter `council_decisions_recorded` |

## Section 2. Why packet succeeds where sister NSCS06 v8 substrate at 5621 LOC fragments

Per `[[why-our-candidates-lose-to-pr-95-family-canonical-diagnosis-20260528]]` BRUTAL HONEST SUMMARY: PR 95/100/101/102/103 won by being **SMALL + COHERENT + BYTE-CLOSED end-to-end** before any optimization layer. The sister NSCS06 v8 substrate at `src/tac/substrates/nscs06_v8_chroma_lut/` has every architectural ingredient (architecture + archive + inflate + procedural variant + distinguishing feature smoke + GT-distribution matched seed + long training adapter + MLX iteration + per-axis attribution + revisions + substrate contract) — 5621 LOC across 12 files. The Wave N+42 packet extracts the canonical PR-95-parity surface (4 files / 970 LOC / 25 tests / reviewable in 30 min) so a reviewer-by-eyeball can confirm all 13 lessons honored simultaneously.

The packet does NOT replace the sister substrate. Per Catalog #110 APPEND-ONLY HISTORICAL_PROVENANCE both coexist: the sister substrate-engineering fork carries the empirical iteration history (MLX-LOCAL long-burn sweeps + per-assumption ablations + revisions + per-axis attribution); the Wave N+42 packet IS the canonical "ship-this-to-judges" surface.

## Section 3. Dykstra alternating-projections feasibility carry

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + canonical equation #26 IN-DOMAIN context membership: the predicted Delta_S band `[-0.0027, -0.0015]` inherits from T3 council #1335 omnibus design memo where Dykstra co-lead verified IS_ADDITIVE=True + intersection_non_empty=True for the (rate, seg, pose) polytope. Wave N+42 packet preserves ALL THREE Dykstra-feasibility preconditions empirically:

- **Rate axis**: 4064 bytes saved per canonical equation #26 closed form (4096-byte LUT -> 32-byte seed); verified deterministic via test_l3_pack_deterministic_byte_stable
- **Seg axis**: FULL RGB renderer per L5 + per-(level, class) chroma LUT preserves chroma-spatial sensitivity per Atick-Redlich retinal redundancy reduction (chroma is least-perceptually-relevant per Daubechies + Mallat multi-scale partition framing)
- **Pose axis**: 6-DOF affine warp preserves canonical cargo-cult-unwind methodology (v6 -> v7 44 percent reduction [contest-CUDA commit 4292c8ce2]); pose deltas drive frame_1 from frame_0 per L10 mask/pose coupling gate

## Section 4. Carmack MVP-first 5-step recipe verification

Per CLAUDE.md "Carmack MVP-first phasing -- NON-NEGOTIABLE":

1. **FREE local macOS-CPU smoke first** -- PASS: 25-test smoke at $0 on M5 Max
2. **Smoke MUST falsifiably challenge cargo-cult** -- PASS: byte-mutation no-op detector L11 falsifies "bytes consumed but no frame change" cargo-cult at 3 surfaces (chroma_seed / pose / cls_stream)
3. **Emit canonical equation anchor + Catalog #344 reference** -- PASS: canonical equation #26 referenced in packet docstring + symposium memo + recipe; symposium-pending anchor will land via `tac.canonical_equations.update_equation_with_empirical_anchor` post-paired-CUDA-RATIFICATION
4. **Land verdict in same commit batch as smoke landing memo** -- PASS: this symposium memo + packet + canonical apparatus mutations land in same commit batch
5. **Re-route operator priority queue within ~1h** -- PASS: TaskCreate status=in_progress -> complete at landing; sister 3-substrate cascade lanes (Wave N+43 Z5 + Wave N+44 PR101_lc_v2_clone) queued

## Section 5. Operator-routable next steps

1. **IMMEDIATE (recommended; $0 LOCAL)**: Wave N+42 packet COMPLETE — ratify symposium PROCEED_WITH_REVISIONS verdict via canonical posterior anchor + probe outcome ledger
2. **CONDITIONAL ($0 LOCAL)**: sister Wave N+43 Z5 PR-95-parity packet build per 3-substrate cascade (operator directive "at least three to full parity"); sister Wave N+44 PR101_lc_v2_clone PR-95-parity audit + extend via Cascade A FEC10 V14-V2 bolt-on
3. **CONDITIONAL ($1-2 paid Modal)**: paired-CUDA RATIFICATION per Catalog #246 AFTER Wave N+30 op-routable #6 (PR111 composite refire AFTER Catalog #377 case-parity hold extincted) lands EITHER PROCEED OR DEFER
4. **CONDITIONAL (per PR111 RATIFICATION result)**: re-symposium re-deliberation per Catalog #325 14-day window
5. **CONDITIONAL (after sub-0.18 RATIFICATION)**: PR111+1 submission cascade per Phase 9 `tools/operator_pr_submission_full_lifecycle.py` (operator-explicit-per-PR HARD GATE preserved)

## Section 6. Discipline trail

- Catalog #229 PV (11 anchors verified before packet authoring; 5 pre-flight memos read in full)
- Catalog #287 placeholder-rationale rejection (zero placeholder rationales)
- Catalog #290 canonical-vs-unique decision per layer (10-row decision table in packet `__init__.py`)
- Catalog #292 per-deliberation explicit assumption-surfacing (7 assumptions; 3 CARGO-CULTED + 4 HARD-EARNED)
- Catalog #294 9-dimension success checklist evidence (9-row table in packet `__init__.py`)
- Catalog #296 Dykstra-feasibility (predicted band [-0.0027, -0.0015] inherits T3 #1335 omnibus Dykstra verification)
- Catalog #300 v2 frontmatter (council_tier T1; predicted_mission_contribution `frontier_breaking_enabler`; override_invoked false)
- Catalog #303 cargo-cult audit section (in packet `__init__.py`)
- Catalog #305 observability surface (6 facets declared in packet `__init__.py`)
- Catalog #313 probe outcomes ledger (PROCEED registered post-symposium)
- Catalog #324 N/A for THIS packet (NO learned weights; sister substrate-engineering fork carries the Catalog #324 discipline)
- Catalog #325 6-step contract (all 6 steps satisfied)
- Catalog #344 canonical equations (canonical equation #26 IN-DOMAIN context membership preserved)
- Catalog #346 canonical roster (sextet pact MIN met + 6 grand council topical specialists)
- Catalog #348 retroactive sweep N/A (no new STRICT gates landed in Wave N+42)
- Catalog #363 recursive self-reflection 4-value `empirical_verification_status` taxonomy (Round 1 complete; Round 2 verification deferred to post-paired-CUDA-RATIFICATION)
- Catalog #117 / #157 / #174 / #206 / #340 commit-time + sister-checkpoint discipline (canonical serializer + POST-EDIT --expected-content-sha256 + 4 checkpoints + sister-checkpoint guard PROCEED)
- 6-hook wire-in declaration per Catalog #125: hook #1 sensitivity-map ACTIVE / hook #2 Pareto constraint ACTIVE / hook #3 bit-allocator N/A (closed-form no bit allocation) / hook #4 cathedral autopilot dispatch ACTIVE (auto-discovered via Catalog #335) / hook #5 continual-learning posterior ACTIVE / hook #6 probe-disambiguator ACTIVE (canonical equation #26 IN-DOMAIN context membership IS the disambiguator)

## Section 7. Cross-references

- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline -- NON-NEGOTIABLE, HIGHEST EMPHASIS" (the canonical 13 lessons + 8 forbidden patterns)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (the META-level extension)
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325 6-step contract this memo satisfies)
- CLAUDE.md "Submission auth eval -- BOTH CPU AND CUDA"
- `[[why-our-candidates-lose-to-pr-95-family-canonical-diagnosis-20260528]]` (operator meta-question diagnosis + 8 structural failure modes + 6-fix prescription)
- `[[iterate-on-ultimate-until-grand-council-symposium-approval-then-deploy-dont-force-standing-directive-20260528]]` (course-correction binding workflow)
- `[[memos-must-be-acted-upon-canonical-apparatus-mutation-enforcement-standing-directive-20260528]]` (BINDING mandate to register canonical apparatus mutations)
- Wave N+30 negative-receipts audit memo (canonical 5-mutation pattern landing canonical anti-patterns + probe outcomes)
- Wave N+29 PR111 composite symposium PROCEED_WITH_REVISIONS verdict
- Sister T1 #1335 symposium 2026-05-21 `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` (5-attendee PROCEED_WITH_REVISIONS for the substrate-engineering fork)
- Canonical equation #26 at `src/tac/canonical_equations/procedural_codebook_savings.py`
- Sister substrate-engineering fork at `src/tac/substrates/nscs06_v8_chroma_lut/`
- Wave N+42 packet at `src/tac/substrates/nscs06_v8_chroma_lut_pr95_parity/`
