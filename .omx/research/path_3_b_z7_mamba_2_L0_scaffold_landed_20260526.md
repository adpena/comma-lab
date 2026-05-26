---
council_tier: T1
council_attendees: [Shannon, Dykstra, AssumptionAdversary, Contrarian, PR95Author]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "L0 SCAFFOLD landing is sufficient milestone given the 3-phase methodology"
    classification: HARD-EARNED
    rationale: "Per brief: Phase 1 audit + Phase 2 decision + Phase 3 design + skeleton + landing memo = canonical L0 SCAFFOLD deliverable; tests pass; sister coordination clean; $0 GPU."
council_decisions_recorded:
  - "op-routable #1: L1 follow-up subagent should implement Mamba2V2Cell + Mamba2TemporalDecoder + Z7MCM3 archive pack/unpack + MPS proxy probe (~$0)"
  - "op-routable #2: paired CUDA dispatch only AFTER MPS-Win on ≥1 axis per Phase 3 §6 probe-disambiguator"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: z7_mamba2_v2_fresh_substrate
related_deliberation_ids:
  - path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_b_z7_mamba_2_substrate_design_decision_20260526
  - path_3_b_z7_mamba_2_substrate_design_20260526
---

# Path 3 candidate B' — Z7-Mamba-2-v2 L0 SCAFFOLD LANDED 2026-05-26

**Lane:** `lane_path_3_b_prime_z7_mamba_2_cargo_cult_first_20260526` L0 → L1 (impl_complete + memory_entry)
**Subagent:** `path_3_b_prime_z7_mamba_2_cargo_cult_first_20260526`
**Cost:** $0 (no paid GPU; design + L0 SCAFFOLD skeleton + 40 tests)
**Wall-clock:** ~80 min across 3 phases

## What landed (the 3-phase deliverable)

1. **Phase 1 cargo-cult audit memo** — `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
   - 8 NEW CARGO-CULTED assumptions surfaced beyond existing CC-1..CC-10 from 2026-05-18
   - 2 NEW HARD-EARNED-PARTIAL assumptions (decompose-and-fork)
   - 4 orthogonal architectural axes (decoder / latent / training-pathway / grammar)
   - Predicted ΔS band P50 = -0.018 → score ~0.175 (frontier_pursuit upper-region)

2. **Phase 2 substrate-design decision memo** — `.omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md`
   - Path (c) FRESH SUBSTRATE DESIGN chosen per binding audit findings
   - 16-layer canonical-vs-unique decision table (8 UNIQUE-FORK + 6 CANONICAL-ADOPT + 1 UNIQUE-IMPL + 1 UNIQUE-DESIGN)
   - NEW substrate dir `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` (preserves existing v1 per Catalog #110/#113 APPEND-ONLY)

3. **Phase 3 L0 SCAFFOLD design memo** — `.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md`
   - All 7 required sections: Catalog #290 + #294 + #303 + #305 + #296 + #309 + Catalog #300 v2 frontmatter
   - Mamba-2 substrate math (selective SSM + SSD scan + temporal-conv decoder)
   - Z7MCM3 archive grammar specification with A_log procedural-regeneration (~5 KB savings vs Z7MCM2)
   - MLX-implementation roadmap per binding directive #1
   - 6-hook wire-in declaration per Catalog #125

4. **L0 SCAFFOLD skeleton** at `src/tac/substrates/z7_mamba2_v2_fresh_substrate/`:
   - `__init__.py` (~80 LOC) — SPDX + Catalog #241 legacy waiver + RESEARCH_ONLY + DISPATCH_ENABLED=False + cross-references
   - `architecture.py` (~245 LOC) — Z7Mamba2V2Config (cargo-cult-unwound defaults) + Z7Mamba2V2Substrate + Mamba2TemporalDecoder + Mamba2V2Cell (all raise NotImplementedError at L0)
   - `archive.py` (~175 LOC) — Z7MCM3 grammar constants + dataclass + pack/parse/replay stubs + A_log procedural-regeneration stub + byte-budget estimate
   - `inflate_runtime.py` (~95 LOC) — HNeRV parity L4 ≤200 LOC budget; canonical select_inflate_device contract; refuses at L0
   - `tests/__init__.py` + `tests/test_basic.py` (~250 LOC) — 40 tests covering RESEARCH_ONLY + DISPATCH_ENABLED + horizon_class + Config validation + all 3 skeleton classes refuse instantiation + Z7MCM3 grammar constants + byte budget self-consistency + inflate runtime arg validation

5. **THIS landing memo**

## Test verification

```
$ .venv/bin/python -m pytest src/tac/substrates/z7_mamba2_v2_fresh_substrate/tests/ -v
============================== 40 passed in 0.12s ==============================
```

## How the 3 phases satisfy the binding operator directives

- **Directive #1** (*"The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"*): the Phase 3 design memo §3 canonical-vs-unique decision per layer explicitly UNIQUE-FORKs 7 layers (decoder / latent_dim / ego_motion_dim / training_pathway / a_log_init / archive_grammar / ib_scale). The default training_backend is `mlx_native` per MLX-first scope. The substrate + curriculum + decoder + grammar + loss + training-pathway + scorer routing are ALL designed coherently around Mamba-2's selective-state-space math, NOT bolted onto Z7-LSTM/GRU's hidden_dim=128 sister-canonical.
- **Directive #2** (*"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*): Phase 1 is the rigorous adversarial pass (10 NEW CC items beyond CC-1..CC-10; 8 NEW CARGO-CULTED; Contrarian VETO satisfied with margin). Phase 2 explicitly REJECTS Path (a) and Path (b) based on the pass findings. Phase 3 designs Path (c) — fresh substrate dir, not extension.

## How predecessor's state_dict-key-parity work is used

Per brief: *"The state_dict key parity work is GENUINELY USEFUL RESEARCH INPUT but the extension approach itself was non-compliant."*

The predecessor (`ae2fa302fbbf5ffa4`) empirically verified Mamba-2 PyTorch↔MLX byte-stable state_dict keys. This is RESEARCH INPUT and is referenced in:
- Phase 1 audit memo §5 (informs the Dykstra-feasibility check that MLX-PyTorch math is portable at the cell layer)
- Phase 3 design memo §7.4 MLX-implementation roadmap (informs the decision to make L0 MLX-first while keeping reference_torch as stability-validation backend per CC-F unwind)

The predecessor's actual code (a bolt-on extension to `time_traveler_l5_z7_mamba2`) is NOT carried forward; the NEW substrate dir `z7_mamba2_v2_fresh_substrate/` is designed from first principles. The MATH the predecessor verified informs the design; the SCAFFOLD does not extend.

## Sister coordination per Catalog #230 (verified 0 file overlap)

- Sister A (`subagent_a_dreamer_v3_rssm_20260526T065116Z_10444`): `src/tac/substrates/dreamer_v3_rssm/` — DISJOINT
- Sister D (`lane_z6_predictive_coding_mlx_scaffold_20260526`): `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` — DISJOINT
- Sister E (`lane_path_3_e_boost_nerv_against_pr110_20260526`): `src/tac/substrates/boost_nerv_pr110_residual/` — DISJOINT
- Sister C' (`path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526`): `src/tac/substrates/nscs06_v8_chroma_lut/` — DISJOINT
- Sister `z7_mamba2_mlx_scaffold_ext_20260526`: `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py` — **SAME SUBSTRATE DIR but my output goes to NEW dir `z7_mamba2_v2_fresh_substrate/`; ZERO file overlap.** My Phase 1 audit memo serves as the cargo-cult prior that should have come BEFORE that sister's bolt-on work; the v1 sister's MLX work is preserved as historical research-signal per Catalog #110/#113.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: deferred to L1 trainer build (Mamba-2 selective-projection gradient norms registered at `tac.sensitivity_map.z7_mamba2_v2_fresh_substrate` at first dispatch)
2. **Pareto constraint**: ACTIVE — 4-axis polytope from Phase 1 §5 declared in Phase 3 §6
3. **Bit-allocator hook**: ACTIVE — Z7MCM3 grammar's A_log procedural regeneration saves ~4 KB; cosine quantization saves ~1 KB; declared in `tac.substrates.z7_mamba2_v2_fresh_substrate.archive.estimated_byte_budget`
4. **Cathedral autopilot dispatch**: deferred to L1 (recipe `.omx/operator_authorize_recipes/substrate_z7_mamba2_v2_fresh_substrate_modal_a100_dispatch.yaml` to be added at L1; `dispatch_enabled: false` + `research_only: true` baseline)
5. **Continual-learning posterior**: ACTIVE — landing memo + design memos can be persisted via `tac.council_continual_learning.append_council_anchor` if operator routes; MLX-research-signal advisory rows carry canonical Provenance non-promotable markers per Catalog #1/#192/#317
6. **Probe-disambiguator**: ACTIVE — 4 probe paths declared in Phase 3 §6 (decoder / latent / training-pathway / grammar disambiguators); implementation deferred to L1

## Catalog #229 PV closure

- ✓ PV-0: 5 canonical helpers importable (Phase 1 §1)
- ✓ PV-1: existing v1 scaffold integrity preserved (Phase 1 §1)
- ✓ PV-2: NEW substrate dir `z7_mamba2_v2_fresh_substrate/` did not exist pre-edit (Phase 3 §1)
- ✓ PV-3: 5 sister subagents in flight; all DISJOINT scope (Phase 1 §1 + Phase 3 §1)
- ✓ PV-4: probe outcome `z7_mamba2_canonical_scale_stability_20260518` BLOCKING applies to v1 only, not v2 (Phase 3 §1)
- ✓ PV-5: predecessor state_dict-key-parity work referenced as INPUT (Phase 1 §1 PV-6 + Phase 3 §7.4)

## 9-dimension success checklist evidence (Catalog #294)

Per Phase 3 design memo §4: all 9 dimensions carry evidence (UNIQUENESS / BEAUTY+ELEGANCE / DISTINCTNESS / RIGOR / OPTIMIZATION-PER-TECHNIQUE / STACK-OF-STACKS-COMPOSABILITY / DETERMINISTIC-REPRODUCIBILITY / EXTREME-OPTIMIZATION+PERFORMANCE / OPTIMAL-MINIMAL-CONTEST-SCORE).

## Atom emission per Catalog #245/#323

Atoms (3):
- `build_council_deliberation_atom(atom_id="path_3_b_z7_mamba_2_cargo_cult_audit_20260526", council_tier="T2", council_verdict="PROCEED", predicted_impact_lower=null, predicted_impact_upper=null, cost_envelope_usd=0.00)` — audit only
- `build_council_deliberation_atom(atom_id="path_3_b_z7_mamba_2_design_decision_20260526", council_tier="T1", council_verdict="PROCEED", predicted_impact_lower=null, predicted_impact_upper=null, cost_envelope_usd=0.00)` — decision only
- `build_council_deliberation_atom(atom_id="path_3_b_z7_mamba_2_L0_scaffold_20260526", council_tier="T1", council_verdict="PROCEED", predicted_impact_lower=-0.040, predicted_impact_upper=-0.005, cost_envelope_usd=0.00)` — L0 SCAFFOLD landed

## Exit criteria checklist

- ✓ Phase 1 cargo-cult audit memo (8 NEW CC; Contrarian VETO satisfied)
- ✓ Phase 2 design-decision memo (Path c FRESH SUBSTRATE explicitly chosen + justified)
- ✓ Phase 3 L0 SCAFFOLD design memo (all 7 sections; horizon_class declared)
- ✓ L0 SCAFFOLD skeleton (5 source files + 1 test file at NEW substrate dir)
- ✓ 40 tests pass
- ✓ Landing memo (this file)
- ✓ Sister coordination 0 file overlap
- ✓ 6-hook wire-in declaration
- ✓ Catalog #229 PV closure
- ✓ Catalog #294 9-dim evidence
- → Canonical serializer commit (next step; via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #117/#157/#174)
- → Lane registry `add-lane` + `mark` for `impl_complete` + `memory_entry` per Catalog #90 + lifecycle discipline

## Cross-references

- Phase 1: `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Phase 2: `.omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md`
- Phase 3: `.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md`
- 2026-05-18 design memo (v1 predecessor): `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- 2026-05-18 stability multi-week path forward (v1 deferral source): `.omx/research/z7_mamba_2_multi_week_path_forward_20260518.md`
- Canonical assumption classification addendum: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- CLAUDE.md non-negotiables: "META-ASSUMPTION ADVERSARIAL REVIEW" / "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" / "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" / "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" / "HNeRV / leaderboard-implementation parity discipline" L5 + L7 / "Forbidden premature KILL" / "Apples-to-apples evidence discipline" / "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
