# Wyner-Ziv optimal implementation queue (2026-05-17)

**Lane:** `lane_grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517`
**Sister memo:** `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md` (the T3 council verdict that produced this queue)
**Verdict:** PROCEED_WITH_REVISIONS — 8 op-routables encoded across 5 sequenced subagent dispatches.

## Sequencing rationale

The queue resolves the **autopilot fake-reward bug class** identified by the
Contrarian + Assumption-Adversary in 5 ordered subagent dispatches. Q1 is the
canonical helper that produces the deliverability_proof artifact; Q2 is the
STRICT preflight gate that enforces consultation of the artifact; Q3 is the
autopilot reweight v2 that consumes the artifact; Q4 is the first empirical
anchor (FEC6 Tier 2 Comma2k19 palette smoke); Q5 is the lane registry
integration.

The 5 subagents are sequentially dependent — Q2 needs Q1's schema; Q3 needs Q1
+ Q2; Q4 needs Q1 + Q2 + Q3 for the autopilot to actually consume the
empirical anchor when it lands; Q5 is the final attribution + integration
step.

The total cost is **~$0.70 GPU spend + ~10.25h editor time + ~520 LOC**.

## Q1 — `lane_wyner_ziv_deliverability_proof_builder_canonical_helper_20260517`

**Subagent briefing (when dispatched):**

Build canonical helper `tac.side_information.deliverability_proof_builder`
that consumes:

1. A `WynerZivSideInfoClassification` instance from `tac.master_gradient_consumers.wyner_ziv_side_info_covariance`.
2. The side-info baker registry from `tac.side_information.decorator._SIDE_INFO_BAKER_REGISTRY`.
3. A per-substrate inflate.py LOC budget (default 200 per HNeRV L4 waiver ceiling).

And produces a `WynerZivDeliverabilityProof` frozen dataclass with all 14
fields enumerated in the council memo Component 2 + persists to
`.omx/state/side_information_deliverability_proofs.jsonl` per Catalog
#128/#131/#138/#245 fcntl-locked JSONL append-only discipline.

**Surfaces:**

* `src/tac/side_information/deliverability_proof_builder.py` (~120 LOC + module docstring)
  * `WynerZivDeliverabilityProof` dataclass (~30 LOC frozen dataclass with `__post_init__` validation)
  * `build_deliverability_proof(substrate_id, classification, registry, inflate_loc_budget=200)` → proof
  * `_classify_byte_to_tier(byte_idx, registry, ...)` → SideInfoSourceTier (the per-byte tier assignment logic)
  * `append_proof_locked(proof)` → fcntl-locked JSONL append (mirror of `append_baker_outcome_locked`)
  * `load_proofs_strict(archive_sha256=None)` → list[WynerZivDeliverabilityProof] (mirror of `load_baker_outcomes_strict`)
  * `latest_proof_for_archive(archive_sha256)` → proof | None
* `src/tac/side_information/contract.py` — extend with `SideInfoSourceTier` IntEnum (~12 LOC delta)
* `src/tac/tests/test_deliverability_proof_builder.py` (~280 LOC, 18 tests)
  * Schema validation (proof shape + tier enum + per-byte mapping)
  * `build_deliverability_proof` happy-path Tier 1 / Tier 2 / Tier 3 / Tier 4 (4 tests)
  * Per-byte tier classification (3 tests covering boundary cases)
  * `append_proof_locked` + `load_proofs_strict` round-trip (3 tests)
  * 4-proc spawn-pool stress (1 test) per Catalog #131 sister
  * Quarantine on corrupt (1 test) per Catalog #138 sister
  * Empty / missing-file (2 tests)
  * `latest_proof_for_archive` filter (1 test)
  * Inflate-LOC-budget overrun rejection (1 test)
  * Hard-coded path string forbidden via Catalog #213 sister check (1 test for completeness)
* Premise verification per Catalog #229 at `.omx/tmp/wyner_ziv_deliverability_proof_builder_premise_verifier.txt` (~5 premises)

**Hooks per Catalog #125:**

1. Sensitivity-map contribution — N/A — this is the PROOF builder; consumes the existing sensitivity_map signal.
2. Pareto constraint — ACTIVE — the proof's `rate_only_score_delta_band_lo/_hi` IS a Pareto bound that `tac.optimization.pareto.add_per_pair_constraint` consumes.
3. Bit-allocator hook — N/A — proof is consumed by autopilot, not bit-allocator directly.
4. Cathedral autopilot dispatch hook — ACTIVE — Q3 wires this; Q1 only emits the proof.
5. Continual-learning posterior update — ACTIVE — `append_proof_locked` IS the canonical-helper append per Catalog #128/#131/#138.
6. Probe-disambiguator — N/A — tier classification is unambiguous given the registry.

**Budget:** ~120 LOC + 18 tests; ~2h editor + 0 GPU; $0.

## Q2 — `lane_catalog_318_venn_reweight_requires_deliverability_proof_strict_gate_20260517`

**Subagent briefing (when dispatched):**

Land new STRICT preflight gate Catalog #318
`check_venn_reweight_requires_deliverability_proof` that refuses any state
where `tools/cathedral_autopilot_autonomous_loop.py` applies the
HIGH_PAIR_INVARIANT reward branch without consulting a per-substrate
deliverability_proof field. Same-line waiver
`# VENN_REWEIGHT_DELIVERABILITY_OK:<rationale>` for the rare deliberate
operator-approved unconditional case (placeholder `<rationale>` literal
rejected per the canonical pattern).

**Surfaces:**

* Claim catalog #318 via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "venn-reweight-deliverability"` per Catalog #186.
* Implement `check_venn_reweight_requires_deliverability_proof` in `src/tac/preflight.py` (~80 LOC + 1 line in `preflight_all()` wire-in `strict=False` per "Strict-flip atomicity rule" until Q3 lands).
* Wire CLAUDE.md catalog row #318 per Catalog #176 META-meta-meta sister gate.
* `src/tac/tests/test_check_318_venn_reweight_requires_deliverability_proof.py` (~250 LOC, 15 tests)
  * Live-repo regression guard
  * Positive (bare HIGH_PAIR_INVARIANT reward without proof check flagged)
  * Negative (canonical proof-consulting pattern accepted)
  * Same-line waiver semantics (rationale accepted / placeholder rejected)
  * Self-exempt (canonical helper + preflight.py + tests)
  * Strict-mode raises with Catalog #318 message
  * Verbose output (clean + dirty)
  * Edge cases (no target file / corrupt source / missing function)
  * Multi-violation aggregation
  * Orchestrator-callsite warn-only wire-in regression guard

**Hooks per Catalog #125:**

1-6: N/A with rationale — this is a STRICT preflight gate self-protection landing.

**Budget:** ~80 LOC + 15 tests; ~1h editor; $0.

## Q3 — `lane_autopilot_venn_reweight_v2_tier_aware_20260517`

**Subagent briefing (when dispatched):**

Refactor `adjust_predicted_delta_for_venn_classification` in
`tools/cathedral_autopilot_autonomous_loop.py` to consume the per-substrate
`WynerZivDeliverabilityProof` artifact from Q1 and apply per-tier reward
factors per the council verdict Component 4 (Tier 1: 1.20× / Tier 2: 1.10×
/ Tier 3: 1.05× / Tier 4 or no-proof: 1.0×).

**Surfaces:**

* `tools/cathedral_autopilot_autonomous_loop.py` (~40 LOC delta)
  * Add `_TIER_REWARD_FACTORS` constant dict
  * Add `_load_deliverability_proof_for_archive(archive_sha256)` helper (delegates to `tac.side_information.deliverability_proof_builder.latest_proof_for_archive`)
  * Modify `adjust_predicted_delta_for_venn_classification` to compute reward factor from proof's tier classification
  * RETAIN HIGH_PAIR_SPECIFIC penalty (0.85×) as-is per council verdict (Wyner-Ziv 1976 correctly classifies pair-specific bytes as not-hoistable regardless of deliverability)
  * Strip the bare HIGH_PAIR_INVARIANT reward branch
* `src/tac/tests/test_autopilot_venn_reweight_v2_tier_aware.py` (~180 LOC, 12 tests)
  * Tier 1 proof → 1.20× reward
  * Tier 2 proof → 1.10× reward
  * Tier 3 proof → 1.05× reward
  * Tier 4 proof → 1.0× (no reward)
  * No proof → 1.0× (no reward)
  * Stale proof (>30 days per Catalog #298) → 1.0× (no reward)
  * HIGH_PAIR_SPECIFIC penalty preserved (0.85×) regardless of proof
  * Proof + HIGH_PAIR_SPECIFIC same archive → penalty wins
  * 4 tests for fixture / dataclass mocking / deterministic / observability

**Hooks per Catalog #125:**

1. Sensitivity-map contribution — ACTIVE — this is the autopilot consumer of the Wyner-Ziv sensitivity signal.
2. Pareto constraint — N/A.
3. Bit-allocator hook — N/A.
4. Cathedral autopilot dispatch hook — ACTIVE — this IS the autopilot ranker wire-in.
5. Continual-learning posterior update — N/A.
6. Probe-disambiguator — N/A.

**Budget:** ~40 LOC delta + 12 tests; ~45min editor; $0.

## Q4 — `lane_fec6_tier_2_comma2k19_palette_smoke_packet_first_empirical_anchor_20260517`

**Subagent briefing (when dispatched):**

Build a smoke packet for the FEC6-class archive (CPU frontier
`6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` per
canonical frontier scan) using a Tier 2 baker: Comma2k19 UV-palette as
shared prior + arithmetic residual encoder for the ~4800-byte poses.bin
section. Run inflate.sh on Modal CPU + paired CUDA T4 per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiable. Verify the score delta is in the L5 codex review's
rate-only band [−0.0019, −0.0032]. If delta exceeds the rate-only band,
falsify the "rate-only" framing per the L5 codex review's `What Would Make
The Larger Claim True` section.

**Surfaces:**

* New submission directory `submissions/fec6_tier_2_comma2k19_palette_wyner_ziv/`:
  * `inflate.py` (≤200 LOC waiver per HNeRV L4; baked Comma2k19 UV-palette constants + arithmetic residual decoder)
  * `inflate.sh` (canonical 3-arg signature `archive_dir output_dir file_list`)
  * `pack_archive.py` (~80 LOC; consumes FEC6 base archive + emits Wyner-Ziv hoist archive)
  * `README.md` (substrate provenance + Catalog #210 license tags + Catalog #213 Comma2k19 cite-chain)
* Modal dispatch recipe `.omx/operator_authorize_recipes/fec6_tier_2_comma2k19_palette_wyner_ziv_modal_cpu_dispatch.yaml` + sister CUDA T4 recipe per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
* `tools/build_fec6_tier_2_comma2k19_palette_smoke_packet.py` (~120 LOC; the encoder + packet builder)
* `src/tac/tests/test_fec6_tier_2_comma2k19_palette_smoke_packet.py` (~50 LOC, 6 tests; smoke roundtrip via test fixtures)
* Premise verification at `.omx/tmp/fec6_tier_2_comma2k19_palette_smoke_packet_premise_verifier.txt`

**Flow:**

1. Premise verify the FEC6 archive sha256 + the existing `tac.side_information.comma2k19_derived_prior_palette` baker contract.
2. Build the packet locally; run `bash submissions/fec6_tier_2_comma2k19_palette_wyner_ziv/inflate.sh <archive_dir> <output_dir> <file_list>` to verify runtime closure per HNeRV L9.
3. Build deliverability_proof artifact via Q1's canonical helper; verify tier_2_baked_constants_byte_count > 0.
4. Dispatch Modal CPU smoke per Catalog #243 + #271 (smoke-before-full discipline + codex pre-dispatch review).
5. Harvest results per Catalog #245 (Modal call_id ledger).
6. If CPU smoke green, dispatch paired CUDA T4 per CLAUDE.md dual-eval mandate.
7. Verify score delta in L5 codex band [−0.0019, −0.0032] for rate-only OR component-deltas if outside band.
8. Land empirical anchor in lane registry per Catalog #220 (substrate operational mechanism) + Catalog #233 (L1→L2 promotion canonical 4-gate).

**Hooks per Catalog #125:**

1. Sensitivity-map contribution — ACTIVE — consumes wire-in 2 sensitivity output.
2. Pareto constraint — ACTIVE — adds the empirical anchor to the Pareto frontier per CLAUDE.md "Meta-Lagrangian/Pareto solver".
3. Bit-allocator hook — ACTIVE — the arithmetic residual encoder consumes per-pair bit budgets.
4. Cathedral autopilot dispatch hook — ACTIVE — empirical anchor feeds back via the deliverability_proof artifact + autopilot reranking.
5. Continual-learning posterior update — ACTIVE — Modal call_id ledger + cost-band posterior + Wyner-Ziv deliverability proof posterior.
6. Probe-disambiguator — N/A — Comma2k19 palette is unambiguous given the cache content sha.

**Budget:** ~250 LOC + ~$0.30 Modal CPU + ~$0.40 Modal CUDA T4 + ~6h end-to-end (including verification + premise verifier + recipe + lane registry update).

## Q5 — `lane_pr101_fec6_deliverability_proof_artifact_integration_20260517`

**Subagent briefing (when dispatched):**

Update the existing `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515` lane registry entry to cite the Q4 empirical deliverability_proof anchor. This is the integration step that closes the loop: the FEC6 canonical CPU frontier substrate now has an empirically-verified Wyner-Ziv deliverability proof, which the autopilot ranker (Q3) consumes to apply the per-tier reward.

**Surfaces:**

* `tools/lane_maturity.py mark lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515 --gate <appropriate_gate> --evidence "<Q4 anchor citation>"` to land the proof attribution.
* Update `.omx/state/lane_registry.json` with the per-lane `deliverability_proof_artifact_id` field (matches the proof's `proof_id_for_autopilot_consumption` from Q1).
* Brief landing memo at `~/.claude/projects/.../memory/feedback_pr101_fec6_deliverability_proof_artifact_integration_landed_20260517.md`.
* No test landing; the test coverage is at Q4's level.

**Hooks per Catalog #125:** 1-6 inherited from Q4.

**Budget:** ~30 LOC + 8 tests (lane registry update + proof citation); ~30min editor; $0.

## Total

| Q | Subagent / lane | LOC | Tests | Wall-clock | GPU $ |
|---|---|---|---|---|---|
| Q1 | deliverability_proof_builder canonical helper | ~120 | 18 | ~2h | $0 |
| Q2 | Catalog #318 STRICT preflight gate | ~80 | 15 | ~1h | $0 |
| Q3 | autopilot reweight v2 tier-aware | ~40 | 12 | ~45min | $0 |
| Q4 | FEC6 Tier-2 Comma2k19 smoke packet (first empirical anchor) | ~250 | 6 | ~6h | ~$0.70 |
| Q5 | PR101 FEC6 lane registry integration | ~30 | 8 | ~30min | $0 |
| **Total** | **5 subagents** | **~520** | **59** | **~10.25h** | **~$0.70** |

**Returns:**

* The AUTOPILOT FAKE-REWARD BUG CLASS is EXTINCT (Q2 + Q3).
* The FIRST empirically-validated Wyner-Ziv hoist anchor lands on the canonical FEC6 CPU frontier substrate (Q4).
* The autopilot ranker now consumes EMPIRICAL deliverability proofs, not THEORETICAL Venn classification (Q3 + Q5).

## Blockers requiring operator decision

1. **Per-tier reward factor calibration** (Tier 1: 1.20× / Tier 2: 1.10× / Tier 3: 1.05×) are the council's PROPOSED values. The operator should confirm OR override per session preference; alternatively land them as kwarg-overridable module constants in Q3 + tune empirically post-Q4 anchor.

2. **Tier 3 operator-review threshold**: the council recommends OPERATOR REVIEW for Tier 3 (scorer-features baked constants) per frozen-weight attestation. The operator should specify the review channel (Codex pre-dispatch per Catalog #271 / standalone manual / autopilot pre-dispatch checkpoint per Catalog #243).

3. **Q4 dispatch budget approval**: ~$0.70 Modal CPU + CUDA T4 smoke for FEC6 Tier-2 first empirical anchor. Per CLAUDE.md "Long-burn score-lowering campaign default" — this is a SHORT-burn one-shot dispatch; no campaign queue required. Operator should approve the spend.

4. **HORIZON-CLASS scope**: the stacked predicted band [0.147, 0.167] reaches FRONTIER_PURSUIT class per CLAUDE.md HORIZON-CLASS standing directive. If the operator wants the Wyner-Ziv staircase to reach ASYMPTOTIC_PURSUIT [0.050, 0.120], additional Tier-2 + Tier-3 stacked hoists are required beyond Q4's first anchor (queued as follow-on op-routables Q6+).

## Mission-alignment summary

Per CLAUDE.md "Mission alignment — non-negotiable":

* `council_predicted_mission_contribution: frontier_protecting` — this council closes the structural drift between the autopilot ranker's THEORETICAL Wyner-Ziv potential signal and the EMPIRICAL contest frontier. The fix prevents the fake-reward bug class from biasing dispatch decisions toward unproven hoist substrates.
* No operator override invoked.
* Frontier opportunity preserved: the Q4 first empirical anchor unlocks the Tier-2 hoist class on the canonical CPU frontier substrate; the full Tier-1+2+3 stacked staircase has a [−0.025, −0.045] ΔS contest-CPU ceiling per the predicted-band Dykstra-feasibility analysis.

## End of implementation queue memo
