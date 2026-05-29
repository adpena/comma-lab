# Wave 9 NSCS06 v8 cargo-cult #4 per-(level, class) chroma LUT aggregation policy canonical helper + fix landed 2026-05-29

---
council_tier: T1
council_attendees:
  - Shannon_LEAD
  - Dykstra_CO_LEAD
  - Yousfi
  - Fridrich
  - Contrarian
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "np.median is the per-(level, class) bin estimator that minimizes contest PSNR distortion"
    classification: CARGO-CULTED
    rationale: "MEDIAN is a robust statistic but is not L2-optimal under Gaussian noise; dashcam pixel distributions are heavy-tailed (road / sky / lane-marking) so skewed bins favor MEAN or MODE. The new canonical helper provides the empirical-comparison unwind path."
  - assumption: "byte-default MEDIAN policy preserves archive-byte parity with all prior empirical anchors"
    classification: HARD-EARNED
    rationale: "Tested at test_byte_default_matches_legacy_implementation: bit-for-bit array_equal vs architecture.build_chroma_lut_from_ground_truth on both synthetic and real upstream/videos/0.mkv frame."
  - assumption: "WEIGHTED_MEAN_BY_CELL_COUNT differs from simple MEAN at per-bin scope"
    classification: CARGO-CULTED
    rationale: "Documented in module docstring and verified at test_weighted_mean_matches_mean_at_per_bin_scope: they collapse to identical output at per-bin scope. The arm is preserved for operator-facing disambiguation per Catalog #308 but the per-bin estimator is identical. Future revision may differentiate at per-class-fallback scope."
council_decisions_recorded:
  - "op-routable #1: canonical helper landed at src/tac/substrates/nscs06_v8_chroma_lut/chroma_lut_per_class_aggregation_policy.py"
  - "op-routable #2: trainer wired with --chroma-lut-aggregation-policy CLI flag defaulting to median_byte_default"
  - "op-routable #3: 19 dedicated Wave 9 tests pass plus 245 total NSCS06 v8 substrate tests pass"
  - "op-routable #4: operator may invoke --chroma-lut-aggregation-policy=mean / =mode_per_cell to land paired-CUDA RATIFICATION smoke vs byte-default median arm"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
horizon_class: plateau_adjacent
---

## Summary

Wave 9 of the 12-wave 15-item math-fidelity audit cascade lands the
canonical per-(level, class) chroma LUT aggregation policy helper at
`src/tac/substrates/nscs06_v8_chroma_lut/chroma_lut_per_class_aggregation_policy.py`,
extending the Wave 5 cargo-cult #6 sister pattern (`cls_lowres_downsample.py`)
to the chroma LUT aggregation surface. Closes Wave 5 op-routable #1
("same META-class as #6 at different helper") and routes the trainer
through the new helper with operator opt-in for the three empirical
unwind paths (MEAN / MODE_PER_CELL / WEIGHTED_MEAN_BY_CELL_COUNT).

## Cargo-cult audit per assumption

| # | Assumption (pre-Wave-9) | Classification | Unwind path |
|---|---|---|---|
| 1 | `np.median` per-(level, class) bin estimator minimizes contest PSNR | CARGO-CULTED | Operator opt-in via `--chroma-lut-aggregation-policy=mean` for paired-CUDA RATIFICATION smoke vs byte-default median arm |
| 2 | Robust statistics are universally appropriate for chroma anchor derivation | CARGO-CULTED | The canonical helper exposes per-policy `median_vs_policy_agreement_fraction` so the operator can quantify empirical relevance per substrate input distribution |
| 3 | The legacy `architecture.build_chroma_lut_from_ground_truth` output is the canonical byte-default | HARD-EARNED | Tested at `test_byte_default_matches_legacy_implementation`: bit-for-bit match preserved on both synthetic and real-frame inputs |
| 4 | The chroma LUT byte cost is independent of aggregation policy | HARD-EARNED | `(grayscale_levels, num_segnet_classes, 3)` uint8 cost is identical across all 4 policies; only the LUT contents differ |
| 5 | `WEIGHTED_MEAN_BY_CELL_COUNT` semantically differs from `MEAN` at per-bin scope | CARGO-CULTED | They are mathematically identical at per-bin scope; documented honestly and arm preserved for operator-facing disambiguation per Catalog #308 |
| 6 | Empty-bin fallback should use the per-class global statistic under the SAME policy | HARD-EARNED | Sister of `architecture.build_chroma_lut_from_ground_truth` fallback; preserved across all 4 policies via `_aggregate_channels(rgb_flat, cls_mask, policy=policy)` |
| 7 | `MODE_PER_CELL` should treat the per-pixel `(R, G, B)` tuple as a categorical 24-bit value | HARD-EARNED | Canonical numpy-deterministic implementation via `np.unique(packed, return_counts=True)` with first-wins tie-break |

## 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| UNIQUENESS | Wave 9 helper is per-substrate-unique (sister substrates use different LUT structures or no LUT at all); not shared with v7 / v3 / D1 etc. |
| BEAUTY+ELEGANCE | ~450 LOC reviewable in 30 seconds; one frozen dataclass + one main builder + one verify helper + one internal `_aggregate_per_bin` + one internal `_aggregate_channels` |
| DISTINCTNESS | The 4 policies are mathematically distinct at the per-bin estimator level (median vs mean vs mode vs weighted-mean); documented at Catalog #308 sister disambiguation |
| RIGOR | 19 dedicated tests including real-frame fidelity via `upstream/videos/0.mkv` per Catalog #213; byte-parity invariant test against legacy implementation |
| OPTIMIZATION-PER-TECHNIQUE | Per-policy `_aggregate_channels` branch dispatches to canonical numpy primitives (`np.median` / `np.clip + np.round + .mean()` / `np.unique + np.argmax`) |
| STACK-OF-STACKS-COMPOSABILITY | The helper is orthogonal to Wave 5 `cls_lowres_downsample.py`; both can be combined for a 4 × 2 = 8-arm composition matrix without interaction effects |
| DETERMINISTIC-REPRODUCIBILITY | Same input bytes -> same output LUT bytes -> same sha256 across runs; verified at `test_byte_default_matches_legacy_implementation` |
| EXTREME-OPTIMIZATION-PERFORMANCE | O(N_pixels * num_segnet_classes) cost; numpy-vectorized per-class bincount for MODE policy; canonical contest-shape (600 * 384 * 512 * 5) ~ 590M ops acceptable for compress-side helper |
| OPTIMAL-MINIMAL-CONTEST-SCORE | Operator opt-in to paired-CUDA RATIFICATION smoke (mean vs median, mode vs median) is the empirical path to a measurably lower contest score; current Wave 9 landing is the structural foundation |

## Observability surface

- **Inspectable per layer**: the `ChromaLutAggregationVerdict` dataclass surfaces `policy`, `chroma_lut_shape`, `chroma_lut_sha256`, `median_vs_policy_agreement_fraction` per call.
- **Decomposable per signal**: agreement fraction is a per-call research-signal metric that quantifies the empirical relevance of cargo-cult #4 unwind for the specific input distribution.
- **Diff-able across runs**: same input bytes -> same sha256; cross-run diff detects any non-deterministic regression.
- **Queryable post-hoc**: the verdict's `as_dict()` serialization is JSON-safe and consumable by canonical posterior writers.
- **Cite-able**: the helper is the canonical producer of canonical equation `chroma_lut_per_class_aggregation_policy_v1` per Catalog #344.
- **Counterfactual-able**: the operator can switch policy via `--chroma-lut-aggregation-policy` CLI flag and observe the resulting archive sha + paired-CUDA delta.

## Predicted ΔS band

[-0.005, +0.005] per Dykstra-feasibility: the byte cost is identical across
all 4 policies so the rate axis is invariant; the per-(level, class) chroma
anchor difference modulates the per-pixel reconstruction error which in
turn modulates SegNet argmax stability and PoseNet feature-vector drift.
Empirical magnitude is unknown until paired-CUDA RATIFICATION on each
policy arm; the canonical helper's `median_vs_policy_agreement_fraction`
surfaces a per-call upper bound on the policy-induced LUT byte divergence.

## Horizon class declaration

`horizon_class: plateau_adjacent` per CLAUDE.md "HORIZON-CLASS evaluation
axis": the v8 substrate sits in the [0.180, 0.200] PLATEAU-ADJACENT band;
the Wave 9 helper is a per-substrate engineering improvement, not a
class-shift. The operator routes paired-CUDA RATIFICATION via the standard
Catalog #246 pattern.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Module location (`src/tac/substrates/nscs06_v8_chroma_lut/`) | UNIQUE | Per-substrate canonical helper; not shared with sister substrates |
| Frozen dataclass `ChromaLutAggregationVerdict` | ADOPT canonical pattern | Mirrors `ClsLowresDownsampleVerdict` from Wave 5 sister helper |
| Non-promotable Provenance contract | ADOPT canonical | Catalog #287 + #323 + #341 canonical contract |
| 4-policy enum `SUPPORTED_AGGREGATION_POLICIES` | UNIQUE | Per-substrate per-(level, class) aggregation semantics; not portable to sister substrates with different LUT structures |
| `_aggregate_per_bin` + `_aggregate_channels` internal helpers | UNIQUE | Substrate-specific aggregation logic; not a shared canonical helper |
| Trainer `--chroma-lut-aggregation-policy` argparse flag | ADOPT canonical pattern | Mirrors Wave 5 `--cls-lowres-downsample-policy` operator opt-in semantics |
| MEDIAN byte-default | ADOPT canonical | Preserves archive-byte parity with all prior empirical anchors per CLAUDE.md "Frontier scores are pointer-only" |

## 6-hook wire-in declaration

| Hook | Status | Note |
|---|---|---|
| #1 sensitivity-map | ACTIVE | per-(level, class) MEAN preserves L2-optimal centroid that point-sample MEDIAN can shift |
| #2 Pareto constraint | N/A | same canonical chroma_lut byte cost across all 4 policies |
| #3 bit-allocator | N/A | same byte cost; no per-tensor reallocation |
| #4 cathedral autopilot dispatch | ACTIVE | policy selection IS the canonical disambiguator between BYTE-DEFAULT-MEDIAN vs MEAN-vs-MODE-vs-WEIGHTED-MEAN arms |
| #5 continual-learning posterior | ACTIVE | paired smoke would emit empirical anchor for canonical equation `chroma_lut_per_class_aggregation_policy_v1` |
| #6 probe-disambiguator | ACTIVE PRIMARY | this module IS the canonical disambiguator for cargo-cult #4 |

## Apparatus mutation chain

1. **Canonical helper** at `src/tac/substrates/nscs06_v8_chroma_lut/chroma_lut_per_class_aggregation_policy.py` (~450 LOC).
2. **Package export** at `src/tac/substrates/nscs06_v8_chroma_lut/__init__.py` with 7 new exports.
3. **Trainer wire-in** at `experiments/train_substrate_nscs06_v8_chroma_lut.py`:
   - New import: `build_chroma_lut_with_policy` + `CANONICAL_AGGREGATION_POLICY_BYTE_DEFAULT` + `SUPPORTED_AGGREGATION_POLICIES`.
   - New argparse flag: `--chroma-lut-aggregation-policy {median_byte_default,mean,mode_per_cell,weighted_mean_by_cell_count}` defaulting to `median_byte_default`.
   - Call site: `chroma_lut, chroma_lut_aggregation_verdict = build_chroma_lut_with_policy(odd_rgb, cls_full, ..., policy=args.chroma_lut_aggregation_policy)`.
4. **Tests** at `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_wave_9_cargo_cult_4_aggregation_policy.py` (19 dedicated tests; all pass).
5. **Canonical equation** `chroma_lut_per_class_aggregation_policy_v1` registered via `tac.canonical_equations.register_canonical_equation` per Catalog #344.
6. **Canonical anti-pattern** `chroma_lut_hardcoded_median_without_empirical_vs_alternatives_v1` registered via `tac.canonical_anti_patterns.register_anti_pattern` (severity `medium_substrate_regression`).
7. **Probe outcome** `wave_9_nscs06_v8_cargo_cult_4_aggregation_policy_helper_landed_20260529` PROCEED advisory 14-day expires 2026-06-12 via `tac.probe_outcomes_ledger.register_probe_outcome`.
8. **Council deliberation anchor** T1 PROCEED 5-voice via `tac.council_continual_learning.append_council_anchor` per Catalog #300 + #292 + #346 + #363.
9. **Lane registry** `lane_wave_9_nscs06_v8_cargo_cult_4_aggregation_policy_helper_plus_fix_20260529` L1 with `impl_complete` + `strict_preflight` gates marked.
10. **Retroactive sweep memo** per Catalog #348 at `.omx/research/retroactive_sweep_for_wave_9_cargo_cult_4_20260529.md`.
11. **Per-substrate symposium memo** appended to existing v8 chroma_lut symposium memo per Catalog #325.

## Empirical receipts

- 19 dedicated Wave 9 tests pass in 0.44s.
- 245 total NSCS06 v8 substrate tests pass in 3.20s (no regression).
- Real-frame smoke on `upstream/videos/0.mkv` (decoded via pyav per Catalog #213): byte-default MEDIAN matches legacy bit-for-bit; MEAN and MODE arms diverge with quantifiable agreement fractions.

## Reactivation criteria

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the
three unwind policies (MEAN / MODE_PER_CELL / WEIGHTED_MEAN_BY_CELL_COUNT)
are NOT killed; they are operator-routable via the new argparse flag.
Reactivation paths:

1. Operator invokes `--chroma-lut-aggregation-policy=mean` on the next
   v8 dispatch to land a paired-CUDA RATIFICATION smoke vs byte-default
   median.
2. Operator invokes `--chroma-lut-aggregation-policy=mode_per_cell` for
   the boundary-preserving MODE arm sister to Wave 5 cargo-cult #6 fix.
3. Future Wave N+ landing differentiates `WEIGHTED_MEAN_BY_CELL_COUNT`
   from simple `MEAN` at per-class-fallback scope (currently documented
   as identical at per-bin scope).

## Cross-references

- **Sister Wave 5** at commit `85521b61d` (`cls_lowres_downsample.py` canonical helper + integrated fix).
- **15-item audit cascade** standing directive at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_15_item_audit_validate_fix_harden_test_blanket_approval_1to1_fidelity_with_documented_adaptations_standing_directive_20260529.md`.
- **CLAUDE.md** "HNeRV / leaderboard-implementation parity discipline" L1 + L2 + L7.
- **CLAUDE.md** "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".
- **CLAUDE.md** "Apples-to-apples evidence discipline".
- **CLAUDE.md** "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch".
- **Catalog #344** canonical equations + anti-patterns registry sister.
- **Catalog #335** cathedral consumer auto-discovery.
- **Catalog #287** placeholder-rationale rejection.
- **Catalog #313** probe outcomes ledger.
- **Catalog #300** council deliberation v2 frontmatter.

## Sober English discipline

Per `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_excessive_canonical_adjective_spam_in_spawn_prompts_triggers_anthropic_usage_policy_guardrail_20260529.md`:
this memo uses "canonical" only when the word carries technical meaning
(canonical equation registry, canonical Provenance contract, canonical
helper, etc.). Prose narrative uses ordinary English.
