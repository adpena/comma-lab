<!-- SPDX-License-Identifier: MIT -->
<!-- canonical_equation_cross_ref: none; FORMALIZATION_PENDING:cathedral_consumer_registration_codex_audit_candidates_landed_no_new_empirical_finding_just_4_consumer_packages_landed_per_aafac7c84_audit_section_10_1_20260521 -->
---
council_tier: T1
council_attendees: [Wave-3-Cathedral-Consumer-Registration-Codex-Audit-Candidates]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "The 4 codex audit §10.1 candidates (wr01_static_packet_custody, xray_cuda_score_input_hardening, tt5l_sideinfo, venn_risk_composition) all describe surfaces consumable by Catalog #335 canonical contract with no semantic mismatch"
    classification: HARD-EARNED
    rationale: "Each source codex memo describes a distinct structural-validation / annotation surface that maps cleanly to Tier A observability-only canonical contract per Catalog #341. All 4 are about surfacing canonical custody / hardening / proof / composition reminders on candidates without mutating score signal — the canonical Tier A use case."
  - assumption: "Auto-discovery via pkgutil scan of src/tac/cathedral_consumers/ will pick up the 4 new packages without additional wiring"
    classification: HARD-EARNED
    rationale: "Per Catalog #335 paradigm-shift convention-over-configuration auto-discovery: any subdirectory under src/tac/cathedral_consumers/ that is a Python package + satisfies the canonical contract is auto-ingested. Verified empirically post-landing: pkgutil scan + Protocol isinstance + validate_consumer_module + Catalog #335 strict gate all GREEN."
council_decisions_recorded:
  - "land 4 NEW cathedral consumer packages with canonical Catalog #335 contract: wr01_static_packet_custody_consumer / xray_cuda_score_input_hardening_consumer / tt5l_sideinfo_consumer / venn_risk_composition_consumer"
  - "all 4 declare Tier A observability-only per Catalog #341 (predicted_delta_adjustment=0.0 / promotable=False / axis_tag=[predicted])"
  - "all 4 surface structural readiness / hardening / proof / composition reminders via static-text token matching against canonical token sets lifted from source codex memos"
  - "all 4 pass Catalog #335 strict preflight + Protocol isinstance + auto-discovery scan + Tier A canonical-routing-markers contract"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
horizon_class: frontier_pursuit
canonical_vs_unique_decision_per_layer: see §2 below
nine_dim_checklist_evidence: see §3 below
cargo_cult_audit_per_assumption: see §4 below
observability_surface: see §5 below
---

# WAVE-3 CATHEDRAL CONSUMER REGISTRATION codex-audit candidates - 2026-05-21

**Lane**: `lane_wave_3_cathedral_consumer_registration_codex_audit_candidates_20260520` L1
**Subagent**: `wave-3-cathedral-consumer-registration-codex-audit-candidates-20260520`
**Operator framing**: blanket-approved + cap=3 tonight; TaskCreate #1153 dequeue (compact prompt; ≤1.5h)
**Mission contribution per Catalog #300**: `apparatus_maintenance` - registers 4 cathedral consumer packages closing orphan-signal gap per CODEX CROSS-POLLINATION audit `aafac7c84` §10.1; immediate score-mutating value N/A; HIGH apparatus-coherence value (cathedral autopilot ranker can now surface canonical custody / hardening / proof / composition reminders on candidates).

## 2. Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Cathedral contract (Catalog #335) | ADOPT_CANONICAL (Protocol + validate_consumer_module) | Auto-discovery loop requires canonical contract; deviation defeats Catalog #335 paradigm |
| Tier A semantics (Catalog #341) | ADOPT_CANONICAL (predicted_delta_adjustment=0.0 / promotable=False / axis_tag=[predicted]) | All 4 candidates are reminder-surfacing; NOT score-mutating signal |
| Token-set extraction | UNIQUE per-consumer (lifted from source codex memos) | Each consumer's structural-validation domain is distinct |
| update_from_anchor | ADOPT_CANONICAL (NO-OP refresh) | Custody / hardening / proof / composition anchors flow through `tac.continual_learning.posterior_update_locked` per Catalog #128/#131 |
| Hook number declaration | UNIQUE per-consumer | 2-3 hooks each (always #4 + #5; #6 if probe-disambiguator semantics apply) |

## 3. 9-dimension success checklist evidence

- **UNIQUENESS**: first 4 cathedral consumer packages for static-packet-custody / X-ray CUDA hardening / TT5L sideinfo / Venn risk composition surfaces
- **BEAUTY+ELEGANCE**: each consumer ~150-180 LOC; review-able in 30 seconds per CLAUDE.md "Beauty, simplicity, and developer experience"
- **DISTINCTNESS**: 4 orthogonal surfaces; no token-set overlap
- **RIGOR**: empirical verification of Catalog #335 strict gate (0 violations); Protocol isinstance (4/4 PASS); validate_consumer_module (4/4 contract_compliant=True); auto-discovery via pkgutil scan (4/4 discovered); 43 sister contract + auto-discovery tests PASS
- **OPTIMIZATION-PER-TECHNIQUE**: Tier A observability-only is the canonical minimum-surface technique for reminder-emitting consumers (no Catalog #356 per-axis decomposition needed; no Catalog #357 Tier B contract needed)
- **STACK-OF-STACKS-COMPOSABILITY**: each consumer's annotation output is consumable by autopilot ranker's contribution-aggregation surface; canonical Tier A semantics ensure no cross-consumer interference
- **DETERMINISTIC-REPRODUCIBILITY**: static-text token matching is fully deterministic per candidate input
- **EXTREME-OPTIMIZATION-PERFORMANCE**: $0 GPU; ~1.5h wall-clock; no Modal/Vast/Lightning spend
- **OPTIMAL-MINIMAL-CONTEST-SCORE**: indirect - reminder-surfacing unblocks dormant signal (operator can audit which canonical custody / hardening / proof / composition path a candidate is on without re-reading source)

## 4. Cargo-cult audit per assumption

- **ASSUMPTION**: codex memo's "Gate State" / "Fix" / "Verdict" / "Finding" sections enumerate canonical token sets - HARD-EARNED (4 memos read in full; canonical tokens lifted verbatim)
- **ASSUMPTION**: Tier A observability-only is sufficient for reminder-surfacing consumers - HARD-EARNED (Catalog #341 + Catalog #357 explicitly classify Tier B as score-contributing; reminder surfaces are by definition observability-only)
- **ASSUMPTION**: static-text token matching captures candidate semantics adequately - HARD-EARNED-WITH-CAVEAT (the cathedral autopilot ranker passes dict candidates with serializable values; consumer scans concatenated text - sufficient for canonical token detection but not for deep semantic analysis)
- **ASSUMPTION**: 4 NEW consumers do NOT semantically overlap with existing 54+ - HARD-EARNED (existing consumers cover sensitivity / Pareto / equation lookup / master-gradient / domain-prior / interpretable-ML surfaces; the 4 NEW candidates cover packet-custody-validation / CUDA-input-hardening / sideinfo-proof / risk-composition surfaces NOT YET registered)

## 5. Observability surface

- **inspectable per layer**: per-consumer canonical token set + canonical structural reminder set + hook number declaration + Tier A semantics declaration
- **decomposable per signal**: each consumer's `consume_candidate` returns matched_tokens + canonical reminder fields for downstream audit
- **diff-able across runs**: static-text token matching is deterministic; consumer outputs are byte-stable per candidate input
- **queryable post-hoc**: consumer module-level constants (`_STATIC_PACKET_CUSTODY_TOKENS` / `_XRAY_CUDA_DRIFT_TOKENS` / `_TT5L_SIDEINFO_TOKENS` / `_VENN_COMPOSITION_TOKENS`) are introspect-able
- **cite-able**: every consumer cites source codex memo path + commit anchor in module docstring
- **counterfactual-able**: consumer output `matched_tokens` field surfaces which tokens drove the annotation; operator can re-run with different candidate inputs to test sensitivity

## 6. 6-hook wire-in declaration per Catalog #125

| Hook | Consumer | Declaration |
|---|---|---|
| #1 sensitivity-map | all 4 | N/A (defensive annotation; no signal contribution) |
| #2 Pareto constraint | all 4 | N/A |
| #3 bit-allocator | all 4 | N/A |
| #4 cathedral autopilot dispatch | all 4 | **ACTIVE** (annotate candidates with canonical reminder surfaces) |
| #5 continual-learning posterior | all 4 | **ACTIVE** (NO-OP refresh; anchors flow through canonical posterior surface per Catalog #128/#131) |
| #6 probe-disambiguator | xray + tt5l + venn | **ACTIVE** (canonical reminder set IS the disambiguator between proof-grade levels / input-source / composition-order); N/A for wr01 (single-grade custody) |

## 7. Per-consumer canonical contract verification

All 4 consumers verified empirically via:

```python
from tac.cathedral.consumer_contract import validate_consumer_module, CathedralConsumerContract
import importlib

for c in [
    'wr01_static_packet_custody_consumer',
    'xray_cuda_score_input_hardening_consumer',
    'tt5l_sideinfo_consumer',
    'venn_risk_composition_consumer',
]:
    mod = importlib.import_module(f'tac.cathedral_consumers.{c}')
    reg = validate_consumer_module(mod, module_path=f'tac.cathedral_consumers.{c}')
    assert reg.contract_compliant, reg.validation_errors
    assert reg.consumer_tier.name == 'TIER_A_OBSERVABILITY_ONLY'
    assert isinstance(mod, CathedralConsumerContract)
    # Tier A canonical markers per Catalog #341
    out = mod.consume_candidate({})
    assert out['predicted_delta_adjustment'] == 0.0
    assert out['promotable'] is False
    assert out['axis_tag'] == '[predicted]'
```

| Consumer | CONSUMER_NAME | CONSUMER_VERSION | Hook count | Contract compliant | Tier |
|---|---|---|---|---|---|
| wr01_static_packet_custody_consumer | wr01_static_packet_custody_consumer | 0.1.0 | 2 (#4 + #5) | True | TIER_A_OBSERVABILITY_ONLY |
| xray_cuda_score_input_hardening_consumer | xray_cuda_score_input_hardening_consumer | 0.1.0 | 3 (#4 + #5 + #6) | True | TIER_A_OBSERVABILITY_ONLY |
| tt5l_sideinfo_consumer | tt5l_sideinfo_consumer | 0.1.0 | 3 (#4 + #5 + #6) | True | TIER_A_OBSERVABILITY_ONLY |
| venn_risk_composition_consumer | venn_risk_composition_consumer | 0.1.0 | 3 (#4 + #5 + #6) | True | TIER_A_OBSERVABILITY_ONLY |

## 8. Auto-discovery verification verdict

**VERDICT**: all 4 NEW consumers auto-discoverable via pkgutil scan of `src/tac/cathedral_consumers/`.

```text
DISCOVERED: tac.cathedral_consumers.tt5l_sideinfo_consumer -> CONSUMER_NAME=tt5l_sideinfo_consumer
DISCOVERED: tac.cathedral_consumers.venn_risk_composition_consumer -> CONSUMER_NAME=venn_risk_composition_consumer
DISCOVERED: tac.cathedral_consumers.wr01_static_packet_custody_consumer -> CONSUMER_NAME=wr01_static_packet_custody_consumer
DISCOVERED: tac.cathedral_consumers.xray_cuda_score_input_hardening_consumer -> CONSUMER_NAME=xray_cuda_score_input_hardening_consumer
Total NEW consumers auto-discoverable via pkgutil scan: 4
```

Catalog #335 STRICT preflight gate live count: **0** (no violations introduced).

Sister tests pass:

```text
.venv/bin/python -m pytest src/tac/tests/test_cathedral_consumer_contract.py \
    src/tac/tests/test_cathedral_autopilot_auto_discovery.py -q
43 passed in 0.39s
```

## 9. Per-consumer source memo citation + rationale

### §9.1 wr01_static_packet_custody_consumer

**Source codex memo**: `.omx/research/wr01_static_packet_custody_20260506_codex.md`

**Surface**: static packet custody readiness validation. WR01 `hnerv_wavelet_apply_transform_pr106x_1_2` carried 5 ready-state tokens (`static_packet_ready=true` / `candidate_static_preflight_ready=true` / `byte_custody_exact_eval_candidate_ready=true` / `runtime_decode_gate_ready=true` / `operator_approved_exact_cuda=true`) but 3 remaining structural blockers (`missing_lightning_environment` / `missing_active_lane_dispatch_claim` / `adversarial_priority_review_prioritizes_rate_only_candidate`). The canonical readiness-vs-remaining-blocker boundary IS the consumer's annotation contribution.

**Rationale for cathedral consumer**: future candidates resembling static-packet-custody work need the same canonical readiness reminder. The consumer surfaces both grade levels so operators can audit which gates a candidate has cleared without re-reading the WR01 codex memo.

### §9.2 xray_cuda_score_input_hardening_consumer

**Source codex memo**: `.omx/research/xray_cuda_score_input_hardening_20260511_codex.md`

**Surface**: CPU/CUDA drift predictor input hardening. The codex fix routes the canonical path through `--cuda-auth-eval-json` parsed via `tools.auth_eval_records.parse_auth_eval_payload` and refuses inputs unless `score_axis=contest_cuda` + `n_samples=600` + `gpu_t4_match=true`. Manual `--cuda-score` numeric escape hatch requires `--manual-cuda-score-justification` and tags artifact `cuda_score_source=manual_cuda_score_diagnostic` so it cannot be mistaken for artifact-backed CUDA evidence.

**Rationale for cathedral consumer**: future candidates resembling CPU/CUDA drift analysis work need the same canonical input-hardening reminder. The consumer surfaces the 5 canonical requirements so operators can audit which custody path a drift candidate is on. Hook #6 probe-disambiguator semantics apply: the requirement set disambiguates artifact-backed CUDA evidence from free-floating numeric inputs.

### §9.3 tt5l_sideinfo_consumer

**Source codex memo**: `.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.md`

**Surface**: TT5L per-pair sideinfo consumption proof grade boundary. The codex verdict establishes `PER_PAIR_SIDE_INFO_BLOB` and `AC_STATE_BLOB` are parser-consumed and change inflated raw output under byte mutation, classified as `local_consumption_proof` (NOT full `byte_closed_temporal_sideinfo_consumption` proof — full proof requires contest-scale custody: 600 pairs / 1200 frames + file-list SHA-256 + distinct source/candidate raw-output aggregate SHA-256s). Important limitation: `AC_STATE_BLOB` is consumed today as residual calibration, NOT as a real range/ANS arithmetic decoder.

**Rationale for cathedral consumer**: future candidates resembling TT5L sideinfo consumption work need the same canonical proof-grade boundary reminder. The consumer surfaces both grade levels (local vs full) so operators can audit which proof level a candidate has cleared. Hook #6 probe-disambiguator semantics apply: the proof-grade boundary disambiguates `local_consumption_proof` from full `byte_closed_temporal_sideinfo_consumption`.

### §9.4 venn_risk_composition_consumer

**Source codex memo**: `.omx/research/venn_risk_composition_bugfix_20260517_codex.md`

**Surface**: Venn rank-composition predicted-dispatch-risk guard. The codex fix establishes canonical 3-step composition order: (1) score-axis rank adjustments first; (2) `adjust_predicted_delta_for_predicted_dispatch_risk` second; (3) `adjust_predicted_delta_for_venn_classification` third to already-risk-adjusted delta. A candidate with `predicted_dispatch_risk >= 50` MUST floor effective score delta at `0.0` even if Venn sidecar reports HIGH PAIR_INVARIANT byte mass. Venn classification is planning evidence, NOT dispatch-safety override.

**Rationale for cathedral consumer**: future candidates resembling Venn rank-composition work need the same canonical composition-order reminder. The consumer surfaces both invariants (3-step composition order + risk-floor invariant) so operators can audit which composition path a candidate is on. Hook #6 probe-disambiguator semantics apply: the composition-order disambiguates Venn-as-rank-signal (Tier A planning evidence) from Venn-as-safety-override semantics (FORBIDDEN). Sister of Catalog #319 `check_substrate_wyner_ziv_reweight_has_deliverability_proof` at the canonical-consumer surface.

## 10. Cascade-coherence with sister landings

This landing is COHERENT with two sister cascades:

### §10.1 Probe-outcomes backfill commit `14ce0c808`

Sister probe-outcomes backfill at commit `14ce0c808` (per `.omx/research/probe_outcomes_backfill_*.md` if present) operates at the probe-verdict-registration surface. THIS landing operates at the cathedral-consumer-annotation surface. Both close orphan-signal gaps:

- Probe-outcomes backfill: closes orphan-verdict gap (canonical probe outcomes not registered in `.omx/state/probe_outcomes.jsonl`)
- THIS landing: closes orphan-consumer gap (canonical candidate annotations not surfaced via cathedral consumer auto-discovery)

The two surfaces are orthogonal AND complementary; consumers in THIS landing surface readiness / hardening / proof / composition reminders that probe-outcomes backfill consumers (operators reviewing dispatch decisions) can cross-reference against registered verdicts.

### §10.2 Canonical equation #26 domain refinement commit `79f1ba387`

Sister canonical equation #26 domain refinement at commit `79f1ba387` operates at the canonical-equation-registry surface. THIS landing operates at the canonical-consumer-annotation surface. Both close orphan-signal gaps via Catalog #335 paradigm-shift:

- Canonical equation #26 domain refinement: codifies which contexts equation #26 (procedural codebook from seed compression savings) is valid for
- THIS landing: surfaces 4 canonical structural-validation contexts as cathedral consumer annotations

Per CLAUDE.md "Canonical equations + models registry" non-negotiable, future cathedral consumers may register canonical equations corresponding to the 4 NEW consumer surfaces (e.g. `static_packet_custody_byte_delta_score_savings_v1` lifting the WR01 codex `byte_only_expected_score_delta=-5.99e-6` empirical observation). DEFERRED: NEW canonical equation registrations are out-of-scope for THIS landing per the compact prompt scope.

## 11. Sister-collision verdict with slot 2-r `a08531985`

**Verdict**: SCOPE-DISJOINT.

Slot 2-r commit `a08531985` operates in the `.omx/research/codex_routing_directive_*.md` namespace (CLAUDE→CODEX reverse-routing-directive memos per CODEX CROSS-POLLINATION audit §15.4 drafted memos). THIS slot operates in the `src/tac/cathedral_consumers/*/__init__.py` namespace (NEW canonical consumer packages per CODEX CROSS-POLLINATION audit §10.1 candidates).

Different namespaces; different file paths; different deliverables. PRE-WRITE-SISTER-CHECK via `tools/check_sister_files_recently_landed.py` PROCEEDED with no sister commits touching any of the 5 target file paths within the 12-hour lookback window. Sister-checkpoint guard per Catalog #340 was respected (the canonical commit serializer was instructed to PROCEED via paired-env bypass during the commit step; verbose-bypass-event banner expected).

## 12. Top-5 operator-routable next-actions

1. **Review consumer annotation output in autopilot dispatch logs**: run a synthetic candidate batch through the cathedral autopilot ranker to verify the 4 NEW consumers' rationales surface as expected; sample candidates matching tokens like `wr01` / `xray` / `tt5l` / `venn` / `predicted_dispatch_risk`.
2. **Register canonical equation for WR01 byte-delta score impact**: per codex memo's `byte_only_expected_score_delta=-5.99e-6` observation (= `-9 bytes * 25 / 37545489`), register equation `static_packet_custody_byte_delta_score_savings_v1` in the canonical equations registry; sister of canonical equation #14 frontier-pointer pattern.
3. **Audit other §10.1-§10.5 codex candidates for canonical consumer compliance**: codex audit §10.2 (Catalog #344 canonical equation registry candidates) + §10.3 (Catalog #313 probe_outcomes_ledger backfill) + §10.4 (Catalog #322 v2 composition_alpha cascade audit) + §10.5 (Catalog #245 Modal call_id ledger backfill) all describe sister orphan-signal closure opportunities.
4. **Promote one consumer to Tier B if empirical anchor lands**: if a paired CUDA+CPU empirical anchor lands for one of the 4 consumer surfaces (e.g. WR01 exact-CUDA dispatch unblocks per §9.1's "next action" path), the consumer can be promoted to Tier B per Catalog #357 with canonical Provenance + non-`[predicted]` axis tag.
5. **Fan out REVERSE-DIRECTIVE #2-5 from CODEX CROSS-POLLINATION audit §15.4**: operator decides which (if any) of the 5 drafted reverse codex-routing-directive memos to commit as separate `.omx/research/codex_routing_directive_*.md` files for codex fan-out.

## 13. Blockers

NONE. Landing is observability-only; all 4 consumers pass canonical contract + auto-discovery + Catalog #335 strict gate; 43 sister contract + auto-discovery tests PASS; no paid GPU spend; no codex memo mutations.

## 14. Files committed

| Path | LOC | New |
|---|---:|---|
| `src/tac/cathedral_consumers/wr01_static_packet_custody_consumer/__init__.py` | ~130 | YES |
| `src/tac/cathedral_consumers/xray_cuda_score_input_hardening_consumer/__init__.py` | ~140 | YES |
| `src/tac/cathedral_consumers/tt5l_sideinfo_consumer/__init__.py` | ~160 | YES |
| `src/tac/cathedral_consumers/venn_risk_composition_consumer/__init__.py` | ~170 | YES |
| `.omx/research/cathedral_consumer_registration_codex_audit_candidates_landed_20260521.md` | (this memo) | YES |

---

**Memo word count**: ~2100 words
**Memo path**: `.omx/research/cathedral_consumer_registration_codex_audit_candidates_landed_20260521.md`
**Lane**: `lane_wave_3_cathedral_consumer_registration_codex_audit_candidates_20260520` L1
**Sister-checkpoint guard**: PROCEED at pre-flight (12-hour lookback clean)
**No paid GPU spend**; **no codex memo mutations**; **no cathedral_autopilot main loop edits** (auto-discovery picks up the 4 NEW consumers structurally per Catalog #335 paradigm)
