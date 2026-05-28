---
sweep_kind: retroactive_sweep_per_catalog_348
landing_utc: 2026-05-28T20:15:00Z
landing_lane: lane_wave_n12_canonical_apparatus_signal_preservation_20260528
landing_commit_ref: wave_n12_canonical_apparatus_signal_preservation_RESUME_extended_20260528
sweep_covers_6_artifacts: true
council_predicted_mission_contribution: apparatus_maintenance
---

# Retroactive sweep per Catalog #348 — Wave N+12 RESUME-EXTENDED 6 canonical apparatus artifacts

## Sweep scope

Per Catalog #348 4-field contract, this combined sweep covers 6 canonical artifacts landed in the
Wave N+12 RESUME-EXTENDED canonical apparatus signal preservation commit batch
(`lane_wave_n12_canonical_apparatus_signal_preservation_20260528`). The 6 artifacts decompose
into 4 categories per the upstream audit + symposium origins:

1. **Audit-origin (Slot 1 sextet pact adversarial audit `c9153273d` op-routables #2-#5)**:
   - **Artifact 1** — anti-pattern `wyner_ziv_y_derivable_from_x_at_byte_level_structural_ceiling_v1`
   - **Artifact 2** — equation `wyner_ziv_y_derivable_3_surface_convergence_density_ceiling_v1`
   - **Artifact 3** — anti-pattern `simultaneous_multi_subagent_spawn_rate_limit_cascade_v1`
   - **Artifact 4** — equation `api_rate_limit_burst_envelope_predicts_simultaneous_spawn_crash_v1`

2. **T4 symposium Wave N+13 Phase 4 origin (`f5d3c6835` §4.8 + §7.1 + §7.2)**:
   - **Artifact 5** — equation `mlx_pytorch_numerical_equivalence_within_tolerance_per_canonical_helper_v1`
   - **Artifact 6** — anti-pattern `substrate_trainer_uses_pytorch_default_without_mlx_first_consideration_v1`

Per the canonical Catalog #348 contract, each artifact requires per-finding fields:
**bug-class symptom**, **pre-fix window**, **historical KILL/DEFER/FALSIFY scan**, and
**per-finding RE-EVAL-priority assignment**. Below: each artifact is treated as a single
finding for the purpose of the sweep.

---

## Artifact 1: anti-pattern `wyner_ziv_y_derivable_from_x_at_byte_level_structural_ceiling_v1`

### Bug-class symptom signature

3 orthogonal Wyner-Ziv pose-axis Y-derivable-from-X surface measurements
(prefix-Y derived from X first 2 bytes / per-pair PoseNet-output-Y cross-substrate stand-in /
cross-substrate composition Y from FEC6/FECA for PR101) all converge at ~0.000291% mean density,
2290x-4585x BELOW the 1% threshold per audit memo. Empirical signature: PR101 decoder state_dict
bytes (X=~457KB fp16) structurally near-independent from ANY pose-axis-derivable Y at byte level.

### Pre-fix window

Pre-2026-05-28 (specifically before Wave N+5 commit `6f5eabf30` 2026-05-28 16:14Z first surface
landed). Window upper bound = audit landing `c9153273d` 2026-05-28T19:40Z. Pre-fix the apparatus
assumed "we just need the right Y surface" per Assumption-Adversary verdict; the 3-surface
convergence empirically falsifies that assumption at byte level.

### Historical KILL/DEFER/FALSIFY scan

Per `git log --oneline --all 2026-05-15..2026-05-28 | grep -i "wyner.*ziv\|y.derivable"`:
- Wave N+5 `6f5eabf30` first surface FALSIFICATION 2026-05-28
- Wave N+7 `49bdcd78f` second surface FALSIFICATION 2026-05-28
- Wave N+9 `2cedcee48` third surface FALSIFICATION 2026-05-28
- Wave N+10 Slot 2 `e4d3700b1` Yousfi override predicates extension closes audit gap
- Wave N+11 Slot 1 `c9153273d` sextet pact adversarial audit lands op-routable #1 + #2 (this sweep's artifacts)

**No KILL verdicts on Wyner-Ziv paradigm**; only PARADIGM-INTACT-BUT-Y-DERIVABLE-FROM-X-SUB-PARADIGM-STRUCTURALLY-BOUNDED
per Catalog #307 IMPLEMENTATION-LEVEL classification. Per CLAUDE.md "Forbidden premature KILL":
the canonical unwind path is pivot to non-Y-derivable side-info class (cooperative-receiver per
Atick-Redlich 1990 / Tishby IB / Rao-Ballard predictive-coding), not abandonment of the
Wyner-Ziv paradigm itself.

### Per-finding RE-EVAL-priority assignment

**PRIORITY: P1-MEDIUM** (paradigm-intact-but-structurally-bounded; reactivation via 4th orthogonal
Y derivation surface OR operator-invoked paradigm-class-shift pivot). Z6/Z7/Z8 substrate scaffolds
already encode the canonical unwind path per Catalog #310 + #311 + #312 sister gates. Future
substrate dispatches in pose-axis Wyner-Ziv class should default to cooperative-receiver or
predictive-coding architectures rather than Y-derivable-from-X variants.

---

## Artifact 2: equation `wyner_ziv_y_derivable_3_surface_convergence_density_ceiling_v1`

### Bug-class symptom signature

Per-equation predicate: `ρ_convergent = mean(ρ_i(prefix_detector(X, Y_i))) ≈ 0.000291% << 1%`
for any Y_i in the pose-axis derivable family. Empirical anchors: 3 surfaces measured at
0.000218% + 0.000218% + 0.000437% with residual factors 4585x / 4585x / 2290x below the 1%
threshold. The equation captures the structural ceiling as a queryable canonical artifact for
downstream cathedral consumer + canonical equation lookup consumer routing per Catalog #335.

### Pre-fix window

Same as Artifact 1: pre-2026-05-28 the apparatus had 3 separate canonical equations
(`wyner_ziv_decoder_side_information_conditional_entropy_savings_v1` +
`wyner_ziv_pipeline_stage_codec_decoder_side_canonical_y_savings_v1` +
`wyner_ziv_per_pair_posenet_output_y_pose_axis_savings_v1` +
`wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1`) but NO single META-equation
documenting the 3-surface convergence pattern. The META equation closes that gap.

### Historical KILL/DEFER/FALSIFY scan

Same as Artifact 1 (sister artifact at the equation surface; the anti-pattern is at the
forbidden-pattern surface). No KILL verdicts; only IMPLEMENTATION-LEVEL FALSIFICATIONS on the 3
specific surface implementations per Catalog #307.

### Per-finding RE-EVAL-priority assignment

**PRIORITY: P1-MEDIUM** sister to Artifact 1. The equation's `next_recalibration_trigger` is
`when_3+_new_empirical_anchors_in_domain`, so a 4th orthogonal Y derivation surface would
trigger auto-recalibration per the canonical Catalog #371 sister gate.

---

## Artifact 3: anti-pattern `simultaneous_multi_subagent_spawn_rate_limit_cascade_v1`

### Bug-class symptom signature

When N>=3 subagents are spawned concurrently within the same turn or in tight bursts
(<~60 sec window), Anthropic API server-side per-account rate-limiter fires and throttles or
crashes 1+ subagents at socket-closed errors. The 2026-05-28 session observed 6 distinct cascade
incidents across the day, each producing 1-3 simultaneous crashes.

### Pre-fix window

Pre-2026-05-28 the apparatus did not structurally model the per-account rate-limit envelope as
an orchestration constraint. Sister anchor memory file
`feedback_simultaneous_multi_subagent_spawn_rate_limit_cascade_anti_pattern_20260528` captured
the META insight 2026-05-28; this anti-pattern wires it into the canonical registry for
cathedral consumer auto-discovery.

### Historical KILL/DEFER/FALSIFY scan

Per memory file scan + session transcript:
- 6 distinct incidents 2026-05-28 each producing 1-3 simultaneous subagent crashes at socket-closed
- Predecessor `af44f5fcd869dbc80` Wave N+12 first attempt crashed at API socket-closed after 364s
  (transient class, NOT rate-limit; this RESUME demonstrates the recovery pattern)
- No structural KILL verdicts on subagent orchestration paradigm; the canonical unwind is
  single-spawn-per-turn cap=1-per-turn per CLAUDE.md "Subagent coherence-by-default"

### Per-finding RE-EVAL-priority assignment

**PRIORITY: P0-HIGH** (apparatus-wide session-disruption signature). Per the canonical unwind
path, future sessions MUST adopt single-spawn-per-turn cap=1-per-turn discipline at the parent
agent layer with notification-driven natural stagger producing ~30-60 sec inter-spawn windows
that stay within the per-account envelope.

---

## Artifact 4: equation `api_rate_limit_burst_envelope_predicts_simultaneous_spawn_crash_v1`

### Bug-class symptom signature

Predicate: `P(rate_limit_fire) = 1[N_concurrent_spawns >= N_burst_threshold AND
Δt_spawn_burst <= Δt_envelope]` with `N_burst_threshold ≈ 3` and `Δt_envelope ≈ 60s`. Empirical
anchors: 6 incidents 2026-05-28 with empirical rate_limit_fire_rate at N>=3 = 1.0 = predicted.
Residual = 0.0.

### Pre-fix window

Sister to Artifact 3 at equation surface; same pre-fix window. The equation provides the
queryable predictive surface for cathedral autopilot + parent agent orchestration consumers.

### Historical KILL/DEFER/FALSIFY scan

Same as Artifact 3. The canonical posterior anchor will accumulate any future incidents that
cross the burst envelope; the canonical Catalog #371 sister recalibrator auto-refits when 3+
new anchors land.

### Per-finding RE-EVAL-priority assignment

**PRIORITY: P0-HIGH** sister to Artifact 3. Apparatus-wide adoption of single-spawn-per-turn
discipline is the canonical unwind.

---

## Artifact 5: equation `mlx_pytorch_numerical_equivalence_within_tolerance_per_canonical_helper_v1`

### Bug-class symptom signature

MLX float32 vs PyTorch fp32 canonical numerical equivalence within element-wise tolerance
ε per canonical bridge helpers. Empirical: |S_MLX - S_PyTorch| <= ε with ε_canonical = 0.001
contest-units. PR95 hnerv_muon canonical archive empirical: |S_MLX - S_PyTorch| = 0.000011 =
90x margin over threshold; 72x SMALLER than PR110 vs PR101 frontier delta (0.000789).

### Pre-fix window

Pre-2026-05-26 the canonical 0.001 contest-units gate at `tools/gate_mlx_candidate_contest_equivalence.py`
existed per Catalog #1265 LANDED commit `9fdef4a04` but the canonical equation that documents
the equivalence-within-tolerance invariant was NOT registered. T4 symposium Wave N+13 Phase 4
centerpiece §4.8 + §7.1 PROPOSED this equation 2026-05-28 commit `f5d3c6835`; THIS artifact
registers it.

### Historical KILL/DEFER/FALSIFY scan

Per `git log --oneline | grep -i "mlx\|pytorch.*equivalence"`:
- Catalog #1265 LANDED 2026-05-26 `9fdef4a04` (canonical gate; empirical anchor)
- T4 symposium Wave N+13 `f5d3c6835` 2026-05-28 (canonical equation PROPOSED)
- No KILL verdicts on MLX-PyTorch equivalence; canonical empirical receipts all support
  equivalence-within-tolerance per the 3-anchor support set (PR95 hnerv_muon + bridge primitives + FiLM coord-MLP)

### Per-finding RE-EVAL-priority assignment

**PRIORITY: P1-MEDIUM-ENABLER** (the equation IS the canonical infrastructure that future
MLX-first substrate trainers cite per the canonical doctrine; routinely referenced from per-substrate
symposium ratifications). Recalibration trigger: `when_3+_new_empirical_anchors_in_domain`.

---

## Artifact 6: anti-pattern `substrate_trainer_uses_pytorch_default_without_mlx_first_consideration_v1`

### Bug-class symptom signature

Substrate trainer scaffolds that use PyTorch as the default `_full_main` execution path WITHOUT
explicit MLX-first consideration documented violate operator MLX-FIRST 8th standing directive
2026-05-26. Per T4 symposium Wave N+13 Phase 4 centerpiece: training MUST be MLX-first on M5
Max; PyTorch is canonical sister for paired-CUDA RATIFICATION ONLY per Catalog #246.

### Pre-fix window

Pre-2026-05-26 the operator directive did not yet exist (the directive was issued 2026-05-26).
Between 2026-05-26 and 2026-05-28 the directive was active but no structural canonical
anti-pattern enforced it at the substrate trainer scaffold surface. T4 symposium Wave N+13
Phase 4 centerpiece §7.2 PROPOSED the anti-pattern 2026-05-28; THIS artifact registers it.

### Historical KILL/DEFER/FALSIFY scan

Pre-existing PyTorch-default substrate trainers (majority of corpus per T4 symposium audit) are
PRESERVED per CLAUDE.md "Forbidden premature KILL" — the anti-pattern targets NEW substrate
trainer scaffolds going forward, not retroactive deletion of existing trainers. Existing trainers
can document an MLX-first consideration waiver per the canonical 5-step unwind path or simply
defer migration until per-substrate symposium ratification.

### Per-finding RE-EVAL-priority assignment

**PRIORITY: P0-HIGH-DOCTRINE-ENFORCEMENT** (operator MLX-FIRST 8th non-negotiable). Future
substrate trainer scaffolds MUST declare `_mlx_local_full_main` as canonical default OR carry
file-level `# MLX_FIRST_CONSIDERATION_WAIVED:<rationale>` waiver per T4 symposium Phase 4
doctrine + 5-step canonical unwind.

---

## Cross-artifact RE-EVAL summary

| Artifact | Priority | Active mitigation surface | Operator-routable next |
|---|---|---|---|
| 1 (anti-pattern WZ Y-derivable ceiling) | P1-MEDIUM | Z6/Z7/Z8 substrate scaffolds | pivot to cooperative-receiver / predictive-coding |
| 2 (equation WZ 3-surface convergence) | P1-MEDIUM | canonical equation registry + cathedral consumer | observe for 4th surface attempt before next dispatch |
| 3 (anti-pattern simultaneous spawn cascade) | P0-HIGH | parent agent orchestration | enforce single-spawn-per-turn cap=1-per-turn |
| 4 (equation API rate-limit burst envelope) | P0-HIGH | canonical equation + parent agent orchestration | use predicted P(rate_limit) to gate concurrent spawn count |
| 5 (equation MLX-PyTorch numerical equivalence) | P1-MEDIUM-ENABLER | MLX-first substrate trainers + canonical bridge helpers | cite per per-substrate symposium ratification |
| 6 (anti-pattern substrate PyTorch default w/o MLX-first) | P0-HIGH-DOCTRINE | substrate trainer scaffold lifecycle | enforce `_mlx_local_full_main` canonical default at landing |

---

## Closure

Per Catalog #348 contract: all 6 artifacts have bug-class symptom + pre-fix window + historical
KILL/DEFER/FALSIFY scan (none found across all 6; per CLAUDE.md "Forbidden premature KILL"
posture) + per-finding RE-EVAL-priority assignment. Sweep closes the canonical 4-field contract
in a single combined memo to reduce memo proliferation per CLAUDE.md "Memory file rotation
discipline".

Cross-references: Catalog #292 (per-deliberation assumption surfacing) + Catalog #294 (9-dim
checklist) + Catalog #296 (Dykstra-feasibility predicted-band) + Catalog #300 (council
deliberation v2 frontmatter) + Catalog #303 (cargo-cult audit) + Catalog #305 (observability
surface) + Catalog #125 (6-hook wire-in) + Catalog #346 (canonical council roster) + Catalog
#287 (placeholder-rationale rejection) + Catalog #176 (META-meta STRICT-callsite-has-CLAUDE.md-row).
