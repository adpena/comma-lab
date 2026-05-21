---
title: "OVERNIGHT-U PR110 stacking cascade Top-5 IN-DOMAIN to fec6 frontier — structural applicability landing"
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: AssumptionAdversary
    verbatim: "The operator-routable goal 'apply Top-5 IN-DOMAIN candidates to PR110/fec6 frontier' was CARGO-CULTED from the OVERNIGHT-G TRIAGE Pick 7 landing's per-substrate predicted ΔS range without first verifying whether canonical equation #26's IN-DOMAIN context predictions are MEANINGFUL across substrates. The 5 Top-5 candidates each predict REPLACEMENT savings WITHIN their OWN substrate's grammar (openpilot's class_codebook.json or z7's Mamba2 0.bin tokens), NOT cross-substrate application. Per Catalog #344 EXCLUDED #6 (direct_byte_substitution_on_decode_opaque_raw_sections) the cross-substrate application is structurally FORBIDDEN for decode-opaque packed-payload members like fec6's single 'x' member."
council_assumption_adversary_verdict:
  - assumption: "canonical equation #26 IN-DOMAIN predictions for substrate X are also applicable to substrate Y when substrate Y's archive has a member with similar role"
    classification: CARGO-CULTED
    rationale: "the IN-DOMAIN context is defined PER-SUBSTRATE (deterministic_constants_codebook_replacement = openpilot's class_codebook.json; tt5l_transformer_tokens = z7's Mamba2 0.bin). Cross-substrate REPLACEMENT requires the target substrate's grammar to have a corresponding REPLACEABLE codebook member AND the codec-side runtime to consume that member as a codebook (not as a packed payload). fec6 has neither."
  - assumption: "fec6 frontier archive's single packed-payload member 'x' is amenable to cross-substrate byte substitution from openpilot or z7 procedural codebooks"
    classification: CARGO-CULTED
    rationale: "Per Catalog #344 EXCLUDED #6 RATIFY-4 (commit eb7338455 register direct_byte_substitution_on_decode_opaque_raw_sections), fec6's 'x' member is PR101 + frame-exploit-selector + fixed-Huffman-k16 packed bytes; byte-level substitution would corrupt inflate. Empirically falsified at ATW V2 cdf_table_blob (commit 057130de4 — 2560 mutated bytes, max_abs_raw_byte_delta=0)."
council_decisions_recorded:
  - "op-routable #1: 5/5 OVERNIGHT-G Top-5 candidates NON-VIABLE for fec6 frontier cross-substrate application; NO paid dispatch warranted"
  - "op-routable #2: refine OVERNIGHT-G classifier docstring/header to make per-substrate-only semantics explicit so future operator routing does not re-attempt cross-substrate stacking"
  - "op-routable #3: cascade-coherent frontier-lowering paths for fec6 are NEW substrate-class-shift candidates (NSCS06 v8 chroma_lut on PR101 base / DP1 frontier dispatch / ATW V2 / Z7 / etc.) per OVERNIGHT-Q §5, NOT cross-substrate byte recoding"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
lane_id: lane_overnight_u_pr110_stacking_cascade_top5_in_domain_to_fec6_frontier_20260521
schema_version: council_deliberation_v2
---

# OVERNIGHT-U PR110 stacking cascade Top-5 IN-DOMAIN to fec6 frontier — structural applicability landing

## Operator-routable goal

Per OVERNIGHT-U prompt: map OVERNIGHT-G Top-5 IN-DOMAIN recode queue candidates (5 openpilot_prior + 7 z7_world_model; aggregate-12 ΔS [-0.0161, -0.0488] [predicted]) onto PR110/fec6 frontier archive (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`; 0.192051 [contest-CPU] per canonical frontier pointer). For each viable candidate: $0 local CPU byte-mutation smoke per Catalog #139 → if mutation produces measurable frame change → recommend paid dispatch for ratification.

## Empirical verdict (Carmack MVP-first 5-step Phase 1+2 — FREE local CPU smoke)

**5 of 5 Top-5 IN-DOMAIN candidates are STRUCTURALLY NON-VIABLE for fec6 frontier cross-substrate application.**

Per CLAUDE.md "Carmack MVP-first" non-negotiable (`be125b878`): the FREE structural smoke completed before any paid GPU spend, and the structural smoke determines that NO paid dispatch is warranted.

### Empirical anchors

**fec6 frontier archive** (verified Catalog #229 PV):
- Path: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`
- sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Size: 178,517 bytes
- ZIP grammar: SINGLE STORED member `'x'` (178,417 bytes uncompressed, CRC `d1f6dfa9`, method=0 STORED)
- Frontier score: 0.1920513168811056 [contest-CPU] (canonical pointer source)

**OVERNIGHT-G Top-5 IN-DOMAIN candidates** (per `.omx/state/archive_surface_recode_queue_executed_20260521T072658Z.json`):

| Rank | Class | IN-DOMAIN Context | Predicted ΔS | Candidate Members | fec6 Viability |
|---:|---|---|---|---|---|
| 1 | openpilot_prior_candidate | deterministic_constants_codebook_replacement | [-0.001342, -0.004070] | categorical_payload.bin / class_codebook.json / inflate.sh / label_prior_payload_manifest.json / runtime_consumer.py / runtime_consumer_proof_skeleton.json | **NON_VIABLE** — fec6 lacks class_codebook.json |
| 2 | openpilot_prior_candidate | deterministic_constants_codebook_replacement | [-0.001342, -0.004070] | (same as Rank 1) | **NON_VIABLE** — fec6 lacks class_codebook.json |
| 3 | z7_world_model_candidate | tt5l_transformer_tokens | [-0.001342, -0.004070] | 0.bin | **NON_VIABLE** — fec6 member name 'x' (PR101-LC-v2 packed payload), NOT '0.bin' |
| 4 | z7_world_model_candidate | tt5l_transformer_tokens | [-0.001342, -0.004070] | 0.bin | **NON_VIABLE** — same as Rank 3 |
| 5 | z7_world_model_candidate | tt5l_transformer_tokens | [-0.001342, -0.004070] | 0.bin | **NON_VIABLE** — same as Rank 3 |

**Viable subset**: 0 of 5. **Cumulative predicted ΔS for viable subset**: 0.0 (vacuous; no viable candidates).

## Structural rationale

The OVERNIGHT-G TRIAGE Pick 7 classifier landed Top-5 IN-DOMAIN candidates per canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`). Per the canonical equation's per-substrate prediction semantics:

1. **Each IN-DOMAIN prediction is a per-substrate REPLACEMENT-savings claim WITHIN that substrate's own grammar.** Rank 1+2 predict openpilot_prior_candidate's OWN `class_codebook.json` (a 2-6 KB lookup table within the openpilot substrate) could be replaced by a 32-byte seed + procedural codebook generator. Rank 3+4+5 predict z7_world_model_candidate's OWN `0.bin` Mamba2 transformer tokens could be replaced by a procedural-from-seed codebook.

2. **Cross-substrate application requires the TARGET substrate's grammar to have a CORRESPONDING replaceable codebook member AND the codec-side runtime to consume that member as a codebook (not as a packed payload).** fec6's grammar has NEITHER: it carries a single STORED member `'x'` which is a packed PR101-LC-v2 + frame-exploit-selector + fixed-Huffman-k16 stream. The bytes are decode-opaque to anything but the canonical fec6 inflate runtime.

3. **Per Catalog #344 EXCLUDED #6 RATIFY-4** (commit `eb7338455` register_NEW canonical equation #26 EXCLUDED context `direct_byte_substitution_on_decode_opaque_raw_sections`): substituting bytes from openpilot or z7 procedural codebooks into fec6's `'x'` member would corrupt inflate at parse time. Empirically falsified at ATW V2 cdf_table_blob (commit `057130de4` — 2,560 mutated bytes, `max_abs_raw_byte_delta == 0` across all mutations; no observable frame change because the bytes are bit-essential for decode-time grammar but produce no score signal).

4. **Per OVERNIGHT-Q T3 symposium §5 substrate-class-shift cascade** (commit `85ac7b9d2`): REPLACEMENT-UPSTREAM canonical equation #26 IN-DOMAIN is cascade-coherent ONLY for the substrate the prediction was made about. ADDITIVE BOLT-ON paradigm (HFV) is structurally rate-only-delta-bounded. Cross-substrate byte recoding to fec6 is structurally OUT-OF-SCOPE for both paradigms.

## Carmack MVP-first 5-step trace

Per CLAUDE.md non-negotiable `be125b878`:

1. **FREE local CPU smoke first per Catalog #139 byte-mutation smoke**: Phase 1+2+3+4 structural verification — verified fec6 archive sha + grammar (single STORED member `'x'`); cross-referenced OVERNIGHT-G Top-5 candidate predictions against fec6 grammar. Structural smoke determined ZERO candidates have a target codebook member in fec6.
2. **Smoke MUST falsifiably challenge**: predicted ΔS per candidate via canonical equation #26 IN-DOMAIN closed-form was verified to be per-substrate-only (not cross-substrate). Catalog #139 byte-mutation smoke NOT executed because the prerequisite (a target codebook member in fec6) does not exist — running a mutation against fec6's `'x'` member would either fall under EXCLUDED #6 (no frame change but corrupts inflate) or produce a trivially zero ΔS signal.
3. **Catalog #344 reference**: canonical equation #26 IN-DOMAIN contexts (`deterministic_constants_codebook_replacement` / `tt5l_transformer_tokens`) cited per candidate; EXCLUDED #6 (`direct_byte_substitution_on_decode_opaque_raw_sections`) cited as the structural reason cross-substrate application is FORBIDDEN.
4. **Land verdict in same commit batch**: this landing memo + structural verdict JSON + canonical serializer commit.
5. **Re-route operator priority queue within ~1h of empirical landing**: cascade-coherent paths for fec6 frontier-lowering are NEW substrate-class-shift candidates (NSCS06 v8 chroma_lut on PR101 base / DP1 frontier dispatch / ATW V2 / Z7 / etc.) per OVERNIGHT-Q §5, NOT cross-substrate byte recoding of OVERNIGHT-G Top-5.

## Operator-routable next steps

### NOT recommended for paid dispatch

The OVERNIGHT-U operator-routable goal "apply OVERNIGHT-G Top-5 IN-DOMAIN candidates to fec6 frontier" cannot be executed as a cross-substrate stacking cascade because the IN-DOMAIN predictions are per-substrate-internal. No paid Modal/Lightning/Vast.ai dispatch is warranted for these 5 candidates against fec6.

### Cascade-coherent frontier-lowering paths (per OVERNIGHT-Q §5)

For fec6 frontier-lowering below 0.192051 [contest-CPU], the cascade-coherent paths are:

1. **NSCS06 v8 chroma_lut substrate** (lane `lane_overnight_a_nscs06_v8_phase_2_design_20260521` per commit `29f92af8d` PROCEED_WITH_REVISIONS T2 Phase 2 design + OVERNIGHT-T T1 working group revision 1+4 per slot 3 in_flight). This is a NEW substrate-class-shift candidate building from PR101 base with chroma_lut REPLACEMENT (canonical IN-DOMAIN reference per OVERNIGHT-G).

2. **DP1 frontier dispatch** (lane `lane_overnight_r_dp1_3rd_attempt_re_dispatch_dpp_epochs_25_timeout_45min_20260521` per commit `c436eb17d` + `6e684236c` — 2 paired Modal T4 call_ids in flight: `fc-01KS5CTJ...` + `fc-01KS5CXQ...`). DP1 codebook_bytes is canonical IN-DOMAIN reference for `dp1_codebook_bytes` context.

3. **ATW V2 cdf_table_blob reconciliation** (commit `265431dfe` RATIFY-4 EXCLUDED context registration); ATW V2 substrate-pivot per OVERNIGHT-I (commit `d588d6aec`) MVP-first phasing.

4. **PR110 HFV-RESPAWN sensitivity-weighted recoded** (lane `lane_overnight_s_pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_20260521` — sister slot 2 in_flight at OVERNIGHT-U landing time). HFV is the ADDITIVE BOLT-ON paradigm; cascade-coherent for fec6 frontier as a rate-only-delta path.

These paths each build a NEW substrate that targets the fec6 frontier, rather than attempting cross-substrate byte recoding of unrelated substrates' archives.

## Sister-coherence verification

At OVERNIGHT-U landing time per `.omx/state/subagent_progress.jsonl` query:

- **Slot 2** (`overnight_s_pr110_...` `lane_overnight_s_pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_20260521`) touches HFV PR110 builder + recipe + Modal ledger + cathedral consumer modules — **DISJOINT** from my touched files (`experiments/results/pr110_stacking_cascade_top5_in_domain_smoke_20260521/structural_applicability_verdicts.json` + this landing memo).
- **Slot 3** (`overnight_t_t1_w...` `lane_overnight_t_nscs06_v8_phase_2_revision_1_4_t1_working_group_prerequisite_20260521`) touches `.omx/research/council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521.md` — **DISJOINT**.
- **DP1 3rd-attempt** (cron-spawned at 9:36 CDT) — touches Modal call_id ledger + dispatch claims; my landing does NOT mutate either — **DISJOINT**.

Catalog #340 sister-checkpoint guard PROCEED verified pre-edit; no overlap with in-flight sister `files_touched`.

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map contribution**: N/A (defensive structural-applicability analysis; no signal contribution; per-candidate predicted ΔS values feed Catalog #305 observability but not `tac.sensitivity_map.*`).
- **Hook #2 Pareto constraint**: N/A (no Pareto-relevant signal; 5/5 NON_VIABLE means no constraint adjustment).
- **Hook #3 bit-allocator hook**: N/A (no bit-allocator signal).
- **Hook #4 cathedral autopilot dispatch hook**: ACTIVE (the verdict JSON is consumable by `tools/cathedral_autopilot_autonomous_loop.py` per Catalog #335 auto-discovery; surfaces the cross-substrate IN-DOMAIN application as STRUCTURALLY NON-VIABLE so future autopilot iterations do not re-rank the Top-5 against fec6).
- **Hook #5 continual-learning posterior update**: NOT_APPLICABLE_NO_EMPIRICAL_ANCHOR — structural analysis is observability-only; NO canonical equations registry mutation; NO posterior anchor. Canonical equation #26's per-substrate prediction semantics are preserved (the predictions were never about cross-substrate application; this landing memo formalizes that).
- **Hook #6 probe-disambiguator**: ACTIVE (the structural-applicability analysis IS the canonical disambiguator between per-substrate-IN-DOMAIN REPLACEMENT vs cross-substrate cargo-cult stacking — per Catalog #344 EXCLUDED #6 + OVERNIGHT-Q §5).

## Discipline trace

- **Catalog #229** PV: read CLAUDE.md + AGENTS.md + canonical_frontier_pointer + OVERNIGHT-G executed JSON + OVERNIGHT-G landing memo + OVERNIGHT-Q T3 symposium memo + OVERNIGHT-O 5-substrate matrix memo + fec6 inflate.py + fec6 inflate.sh + canonical equation #26 module + canonical equations registry tail + ZIP grammar of fec6 archive BEFORE any analysis
- **Catalog #117/#157/#174** canonical serializer + POST-EDIT --expected-content-sha256
- **Catalog #119** Co-Authored-By trailer (auto-appended by serializer)
- **Catalog #125** 6-hook wire-in declaration (above)
- **Catalog #131/#138** fcntl-locked JSONL N/A (per-invocation unique path; no append-style mutation)
- **Catalog #139** byte-mutation smoke: NOT executed because 5/5 structurally NON_VIABLE (running a smoke against fec6's `'x'` packed-payload member with bytes from non-fec6 candidates would either trigger EXCLUDED #6 protections or produce trivially zero ΔS)
- **Catalog #186** lane pre-registered via OVERNIGHT-D TRIAGE / OVERNIGHT-G executor classification
- **Catalog #206** subagent checkpoint discipline (checkpoints written at steps 1+2+3+complete)
- **Catalog #220** substrate L1+ scaffold operational mechanism: NOT applicable (no new L1+ scaffold lands; analysis only)
- **Catalog #229** premise verification: read all reference files BEFORE any analysis
- **Catalog #230** sister-subagent ownership map: Slot 2 (overnight-S HFV-RESPAWN) + Slot 3 (overnight-T NSCS06 v8 T1 working group) + cron-spawned DP1 3rd-attempt all DISJOINT
- **Catalog #272** distinguishing-feature integration contract: respected — verdict surfaces cross-substrate application as STRUCTURALLY NON_VIABLE per the contract's distinguishing-feature byte-mutation requirement
- **Catalog #287** placeholder-rationale rejection: every verdict + rationale carries substantive non-placeholder text
- **Catalog #292** per-deliberation assumption surfacing: T1 working-group attendees declared in frontmatter; Assumption-Adversary verdict recorded with verbatim
- **Catalog #300** v2 frontmatter: required fields populated
- **Catalog #305** observability surface: per-candidate VIABILITY verdict + rationale + canonical Provenance per row in verdict JSON
- **Catalog #316** frontier pointer cited: 0.192051 [contest-CPU] fec6 anchor sha 6bae0201fb08...
- **Catalog #323** canonical Provenance: every classification carries axis_tag=`[predicted]` + evidence_grade=`predicted` + score_claim=False + promotable=False
- **Catalog #340** sister-checkpoint guard: verified DISJOINT scope before any edit
- **Catalog #341** canonical non-promotable markers: every verdict row carries the 3 canonical Tier-A markers
- **Catalog #344** canonical-equation evolution: respects canonical equation #26 IN-DOMAIN (per-substrate REPLACEMENT) + EXCLUDED #6 (direct_byte_substitution_on_decode_opaque_raw_sections) semantics; does NOT misapply per-substrate predictions to cross-substrate context
- **Catalog #359** residual-hybrid: NO residual-hybrid contexts in this analysis
- **CLAUDE.md "Carmack MVP-first phasing"** (`be125b878`): FREE structural smoke completed before paid GPU; smoke determined NO paid dispatch warranted

## Scope-honest deferrals

- NO Catalog #139 byte-mutation smoke executed on fec6 (5/5 structurally NON_VIABLE; smoke would be wasted CPU per Carmack MVP-first FREE-smoke-first rule)
- NO paid GPU dispatch
- NO operator-authorize chain invocation
- NO push to git origin
- NO nested subagent spawning
- NO mutation of CLAUDE.md, OVERNIGHT-G/Q/O/T/S memos, or HISTORICAL_PROVENANCE artifacts per Catalog #110/#113
- NO canonical equations registry mutation (canonical equation #26 anchor count unchanged at 12 events)
- NO mutation of fec6 frontier archive bytes (READ-only verification)

## Mission contribution

Per Catalog #300: `frontier_protecting` — prevents wasted paid GPU spend on a structurally-non-viable cross-substrate stacking cascade; preserves OVERNIGHT-G's canonical equation #26 per-substrate IN-DOMAIN prediction integrity by surfacing the cross-substrate cargo-cult assumption explicitly; re-routes operator priority queue to cascade-coherent NEW substrate-class-shift candidates per OVERNIGHT-Q §5.

## Cost

$0 GPU + ~45 min wall-clock + 0 paid dispatches.
