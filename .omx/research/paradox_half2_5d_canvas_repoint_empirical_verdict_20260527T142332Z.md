# PARADOX-CLOSER Half 2 — 5D canvas RE-POINT to MLX per-pair heuristic prior — EMPIRICAL VERDICT (2026-05-27)

- timestamp_utc: 2026-05-27T14:23:32Z
- agent: claude (`analysis_5d_canvas_repoint_20260527T141545Z`) — explicit ANALYSIS-sister hand-off from `.omx/research/mlx_per_pair_master_gradient_authoritative_artifacts_landed_20260527.md` §"HAND-OFF TO THE ANALYSIS SISTER"
- lane_id: `lane_paradox_half2_5d_canvas_repoint_per_pair_heuristic_prior_20260527`
- git_head_at_landing: `98339a839`
- scope: $0 — MLX-local / numpy only; NO paid GPU. The sweep verdict is a $0 advisory that GATES whether the paid contest-CUDA/CPU FIRE-phase (Catalog #246) is worth it. NOT a score/frontier/PR claim.
- authority: planning + observability ONLY; `axis_tag="[predicted]"`, `score_claim=false`, `promotable=false`, `evidence_grade=macOS-MLX research-signal` per Catalog #127/#192/#323.
- discipline anchors: Catalog #229 (premise verification before edit) + #192/#127/#323 (canonical Provenance + macOS-MLX non-promotable) + #296 (Dykstra feasibility for predicted ΔS) + #303 (cargo-cult audit flip) + #305 (observability) + #313 (probe-outcomes row LANDED) + #341 (canonical routing markers) + #125 (6-hook wire-in) + #314/#340 (sister-checkpoint guard).
- mission_contribution (Catalog #300): `frontier_protecting` + `apparatus_maintenance` — CLOSES the drop-one-frontier paradox Half 2 resolution gap (heuristic prior) and structurally confirms multi-op synergy is negligible at the frontier, preventing wasteful paid FIRE-phase dispatch on multi-op compositions.

## VERDICT

**`SINGLE_OP_PARETO_OPTIMAL_MULTIOP_SYNERGY_NEGLIGIBLE_RESOLUTION_GAP_NOW_CLOSED`** — with genuine per-pair structure (the resolution gap that GATED the prior Half-2 verdict is now CLOSED as a heuristic prior), the multi-op SYNERGY term remains ~7 orders of magnitude below the V14-V2 frontier crossing. **Single-op is Pareto-optimal at the frontier; multi-op composition adds negligibly beyond the single best operator.** DROP-MANY Hypothesis #2 is empirically VINDICATED — now with the resolution gap CLOSED rather than gated-as-unmeasurable.

### Apples-to-apples comparison (Catalog #229 control)

| quantity | archive-AGGREGATE path (prior, GATED) | per-pair HEURISTIC-PRIOR path (this re-point) |
|---|---|---|
| populator | `populate_5d_canvas_from_master_gradient_anchors` | `populate_per_pair_cells_from_gradient_array` (NEW) |
| source | `master_gradient_anchors.jsonl` (archive-aggregate, 1 coord) | `master_gradient_fec6_frontier_mlx_per_pair_64pair_20260527.npy` `(178517, 64, 3)` |
| canvas cell count | 3 (all at pair0/frame0) | **192** (64 pairs × 3 axes; 64 distinct pair coords) |
| productive operators (of 12) | **0** | **6** (replace_many / merge_pair / reorder_pair / motion_conditional / temporal_coherence + full_drop-via-RAW_RESIDUAL) |
| best single-op heuristic-prior magnitude | 0.0 (none composed) | −0.0154 (`replace_many`) |
| **predicted multi-op EXTRA synergy ΔS** | **−2.55e-19** | **−7.61e-13** |
| V14-V2 frontier crossing threshold | −7.66e-6 | −7.66e-6 |
| Dykstra feasibility | feasible (residual 3.39e-12, 1 iter; degenerate) | feasible (residual 5.43e-14, 118 iter; **near-collinear gradients**) |

Both paths agree on the genuine scientific result: **multi-op synergy is negligible.** The per-pair path moves the synergy term from −2.5e-19 (unmeasurable degenerate) to −7.6e-13 (measurable but still ~7 orders below the frontier-crossing threshold). The resolution gap is CLOSED; the answer is unchanged.

## CRITICAL HONESTY: the verdict-LABEL artifact vs the genuine result

The raw sweep emits `verdict="MULTIOP_BEATS_V14V2"`. **This label is an artifact of the sweep's comparison arithmetic and must NOT be read as "multi-op beats single-op at the frontier."** Per Catalog #229 premise verification + Catalog #323 canonical Provenance:

1. The sweep computes `predicted_multiop_delta = best_single_op_delta + predicted_multiop_extra_delta`, then compares that SUM against −7.66e-6. With the per-pair path, `best_single_op_delta = −0.0154` (the `replace_many` heuristic-prior magnitude), so the sum trips the `< threshold` branch. But the −0.0154 is **the SUM of per-pair finite-difference gradient MAGNITUDES** (a LEVERAGE indicator: `|d(score_component)/d(decoder_tensor)|` aggregated over bytes), NOT a realizable contest-CPU score delta. You do NOT realize a −0.0154 score improvement by applying `replace_many`.
2. The genuine multi-op-vs-single-op question is answered by `predicted_multiop_extra_delta` (the SYNERGY beyond the single best operator), which is **−7.61e-13** — still ~7 orders of magnitude below the V14-V2 frontier crossing. **Multi-op composition adds essentially nothing beyond the single best operator.**
3. The Dykstra polytope solver (118 iterations, residual 5.43e-14) confirms the operator gradients are **near-collinear** at the frontier: the synergy-axis volume of the feasible polytope is ~0. This is the same conclusion the degenerate aggregate path reached, now with genuine per-pair resolution rather than a single collapsed coordinate.

**The contest-faithful reading: single-op (drop-one / replace-one) remains Pareto-optimal; multi-op DISTORTION composition does NOT beat single-op rate-attack at the frontier.** The heuristic-prior leverage map says nothing about realizable score — only that paired contest-CUDA/CPU exact-eval (Catalog #246) on a SINGLE-OP candidate is the only path to a score claim, and Track-A class-shift is the only remaining lever for sub-single-op-Pareto gains.

## Re-point diff summary

### 1. `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py`

- **Cargo-cult audit assumption 4 FLIPPED** from `CARGO-CULTED-PENDING-EMPIRICAL` ("Pair-aggregate decomposition ... deferred to future per-pair gradient extraction wave") to `HEURISTIC-PRIOR-LANDED`. Documents that the per-pair path consumes the MLX `(N_bytes, N_pairs, 3)` artifact as a **macOS-MLX research-signal heuristic prior**, NOT a HARD-EARNED authority anchor; the PyTorch-autograd `tools/extract_master_gradient.py` remains the authority surface.
- **NEW function `populate_per_pair_cells_from_gradient_array`** (~210 LOC): reads the `.npy` + `.npy.meta.json` sidecar; fail-closed on wrong shape / corrupt sidecar / schema mismatch / missing sha (Catalog #138); aggregates per-byte sensitivity over the byte axis into per-pair `(d_seg, d_pose, rate)` magnitudes; builds ONE coordinate per distinct `pair_idx` at `(pair_idx, frame_idx=2·pair_idx, scorer_axis, RAW_RESIDUAL, CONTEST_CPU)`; threads canonical `build_provenance_for_macos_cpu_advisory` Provenance with diagnostic `heuristic_prior=true` + `is_authoritative_contest_axis=false` + the violation reason; emits Catalog #341 non-promotable sidecar.
- Added `CANONICAL_FRAME_COUNT` + `build_provenance_for_macos_cpu_advisory` imports; added function to `__all__`.

### 2. `tools/canvas_multiop_composition_closed_form_prediction_sweep.py`

- `main()` now accepts `argparse`: `--per-pair-gradient-npy <path>` (re-points the canvas population to the per-pair path) + `--max-pairs`. When omitted, the original archive-aggregate path runs (backward-compatible control). Payload carries `populator_source_kind` + `per_pair_gradient_npy_sha256` + `per_pair_pairs_populated` for cite-able provenance.

### 3. `src/tac/tests/test_5d_canvas_per_pair_populator.py` (NEW, 16 tests, all pass)

Shape contract fail-closed + per-pair cell mapping + Provenance non-promotable contract + sidecar roundtrip + max_pairs cap + schema-mismatch fail-closed + operator productivity (the ≥2-distinct-coordinate Half-2 unblock) + missing-artifact/absent-sha fail-closed + live-MLX-artifact regression (192 cells).

## Top-3 ranked operator compositions (heuristic-prior leverage; NON-PROMOTABLE)

| rank | operation | n_candidates | best heuristic-prior magnitude |
|---|---|---|---|
| 1 | `replace_many` | 8 | −0.0154 |
| 2 | `temporal_coherence` | 32 | −0.0108 |
| 3 | `reorder_pair` | 7 | −0.0090 |

These are LEVERAGE indicators (per-pair gradient magnitudes), NOT realizable score deltas. Their RANKING is the heuristic-prior signal; their MAGNITUDES are not contest-CPU score claims. Per Catalog #246 the FIRE-phase remains the only path to a realizable score.

## Catalog #313 probe-outcome row (LANDED)

`tac.probe_outcomes_ledger.register_probe_outcome`:

- `probe_id`: `canvas_multiop_composition_per_pair_heuristic_prior_repoint_sweep_20260527`
- `substrate`: `pair_frame_5d_canvas_12_operator_multiop_composition_per_pair`
- `verdict`: **DEFER** (blocker_status=advisory; 30-day staleness)
- `metric`: `predicted_multiop_extra_delta_s_beyond_single_op = −7.61e-13` vs threshold −7.66e-6
- This is the reactivation of the prior DEFER row `canvas_multiop_composition_closed_form_prediction_sweep_20260527` whose `next_action` was "land per-pair fp64 decomposition ledger then re-run sweep on multi-pair canvas". The MLX per-pair HEURISTIC-PRIOR artifact CLOSES the resolution gap as a heuristic prior. The verdict REMAINS DEFER (not PROCEED) because: (a) the multi-op synergy is empirically negligible, so the FIRE-phase is NOT worth paid spend on multi-op compositions; (b) the per-pair signal is a NON-PROMOTABLE heuristic prior, so it gates rather than authorizes. Honest: this is a heuristic-prior reactivation, not an overclaim of PROCEED.

## Cargo-cult audit per assumption (Catalog #303)

- **HARD-EARNED-EMPIRICALLY-VERIFIED (this re-point)**: with genuine per-pair resolution (192 cells, 64 distinct pair coords), 6/12 operators compose and the multi-op synergy term is −7.6e-13, ~7 orders below the V14-V2 frontier crossing. Verified by direct sweep run + apples-to-apples aggregate control.
- **HARD-EARNED**: the per-pair MLX artifact's byte-axis aggregation into per-pair `(d_seg, d_pose, rate)` magnitudes gives 64 distinct seg + 64 distinct pose coordinates (verified by direct numpy inspection: seg n_neg=45/n_pos=19; pose n_neg=36/n_pos=28).
- **HEURISTIC-PRIOR (NOT HARD-EARNED authority)**: the per-pair magnitudes are macOS-MLX research-signal per-tensor-FD-projected-per-byte values, NOT realizable contest-CPU score deltas. The producer memo's 3 anchor blockers (`source_runtime_full_frame_parity_missing` + `canonical_archive_byte_domain_mapping_missing` + `per_weight_or_per_byte_projector_missing`) keep this a probe-RANKING prior, not authority.
- **FALSIFIED-AS-LABEL-ARTIFACT**: the raw `MULTIOP_BEATS_V14V2` verdict label. The label trips because the sweep sums per-pair gradient MAGNITUDES into `best_single_op_delta` and treats that as a realizable score delta. The genuine multi-op-vs-single-op answer is the SYNERGY term (−7.6e-13 ≈ 0).

## Predicted ΔS band + Dykstra feasibility (Catalog #296)

**Predicted multi-op SYNERGY band**: `[−7.61e-13, 0.0]` (bounded by the near-collinear operator gradients at the frontier). The Dykstra polytope is feasible (residual 5.43e-14, 118 iterations) but its synergy-axis volume is ~0 because the per-pair operator gradients are near-collinear at the frontier operating point. This is the canonical Dykstra-feasibility intersection result that GATES the predicted band: feasible-but-near-empty on the synergy axis. The band does NOT widen with per-pair resolution — confirming the synergy is structurally negligible, not merely unmeasurable.

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: first per-pair-resolution closed-form multi-op-vs-single-op determination at the frontier (closes the Half-2 resolution gap that the prior aggregate sweep could only gate).
2. **BEAUTY + ELEGANCE**: ONE new populator function + ONE sweep `--per-pair-gradient-npy` flag; reuses the canonical 12 operators + Dykstra solver + canonical Provenance builders; no new solver math.
3. **DISTINCTNESS**: distinct from the prior aggregate sweep (gated) + the MLX producer (extracted the artifact); this re-point CONSUMES the per-pair artifact + emits the empirical verdict.
4. **RIGOR**: every API verified via Read/grep before edit; shape contract empirically verified; apples-to-apples aggregate control run; deterministic re-run; 16 dedicated tests + 139 existing canvas tests pass.
5. **OPTIMIZATION-PER-TECHNIQUE**: the Dykstra alternating-projection is the canonical Pareto-feasibility technique; reused not forked.
6. **STACK-OF-STACKS-COMPOSABILITY**: the per-pair populator is the canonical input the 12 operators + Dykstra solver compose over; consumes Catalog #356 AxisDecomposition surface; emits Catalog #341 routing markers.
7. **DETERMINISTIC-REPRODUCIBILITY**: re-run produces byte-identical verdict (deterministic numpy sum over the artifact); canonical JSON sort_keys=True.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: $0 + <2 min wall-clock; refuses paid dispatch.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: the verdict is that multi-op does NOT lower score beyond single-op at the frontier; the highest-EV path is a single-op (drop-one/replace-one) FIRE-phase candidate OR Track-A class-shift, NOT a multi-op composition.

## Observability surface (Catalog #305)

1. **Inspectable per layer**: per-pair populate → 12-operator sweep → problem-spec build → Dykstra solve, each in the sweep JSON payload.
2. **Decomposable per signal**: per-operator candidate counts + best ΔS in `all_operator_summaries`; per-pair per-axis cells queryable via `query_cell`.
3. **Diff-able across runs**: JSON sort_keys=True; deterministic.
4. **Queryable post-hoc**: `tools/canvas_multiop_composition_closed_form_prediction_sweep.py --per-pair-gradient-npy <path>` re-runnable any time.
5. **Cite-able**: frontier archive sha + per-pair artifact sha + Dykstra equation id + canonical equation id in payload.
6. **Counterfactual-able**: re-running with the full-600 artifact (when it lands) OR the PyTorch-autograd authority artifact will refine the per-pair magnitudes; the synergy verdict is the canonical disambiguator.

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map: ACTIVE — the per-pair `(N_bytes, N_pairs, 3)` artifact IS a per-pair per-axis sensitivity map; consumed by the populator into per-pair canvas cells.
- hook #2 Pareto constraint: ACTIVE — the Dykstra alternating-projection on the per-pair operator-gradient polytope IS the canonical Pareto-feasibility check (feasible, residual 5.43e-14, near-collinear → synergy-axis volume ~0).
- hook #3 bit-allocator: ACTIVE — the per-pair per-byte sensitivity is the bit-allocator's per-pair signal (consumed via the per-pair cells' `predicted_delta_score`).
- hook #4 cathedral autopilot dispatch: ACTIVE — the sweep verdict gates the autopilot ranker's FIRE-phase decision (currently a DEFER/non-promotable signal: do NOT fire multi-op).
- hook #5 continual-learning posterior: ACTIVE — the Catalog #313 probe-outcomes DEFER row LANDED; the prior `canvas_multiop_composition_closed_form_prediction_sweep_20260527` DEFER row's resolution criterion (per-pair canvas) is satisfied as heuristic prior.
- hook #6 probe-disambiguator: ACTIVE — the per-pair re-point IS the canonical disambiguator between the orthogonal-vs-synergistic interaction hypotheses; resolves empirically as ORTHOGONAL (near-collinear gradients → negligible synergy).

## Sister coordination (Catalog #314/#340)

Sister subagent `a299da55b207c9e64` (PACT-NeRV `_full_main` cluster) is STILL RUNNING and owns `experiments/train_substrate_pact_nerv_ia3.py`, `src/tac/substrates/pact_nerv_ia3/`, `src/tac/substrates/_shared/pact_nerv_full_main.py`, anything `*pact_nerv*`. My scope is disjoint: `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py` + the sweep tool + my new test file + the probe-outcomes row + this memo. I touched NONE of the sister's files.

## Operator-routable next steps (highest-EV first; all $0 advisory)

1. **Single-op FIRE-phase candidate (if paid spend authorized)**: the verdict confirms single-op is Pareto-optimal. The highest-EV paid candidate is a single-op drop-one / replace-one on the most-negative per-pair leverage coordinate (`replace_many` pair set's top member OR the existing DQS1 drop-one rank021 anchor). Paired contest-CUDA/CPU exact-eval per Catalog #246 is the ONLY path to a realizable score claim. **Multi-op compositions are NOT worth the FIRE-phase spend** (synergy −7.6e-13).
2. **Track-A class-shift** is CONFIRMED as the only remaining lever for sub-single-op-Pareto gains — multi-op DISTORTION composition at the frontier operating point is exhausted (orthogonal/near-collinear gradients). This vindicates routing future $0 design effort to class-shift substrates (predictive-receiver / cooperative-receiver / Wyner-Ziv / foveation) over more within-class multi-op composition exploration.
3. **Full-600 + PyTorch-autograd authority refinement** ($0, fidelity upgrade): re-run with the full-600 MLX artifact (RESUMABLE per the producer memo) OR the PyTorch-autograd authority artifact to refine the per-pair magnitudes. The synergy verdict is expected to hold (the 64-pair sample already shows near-collinear gradients); this is a fidelity confirmation, not a blocker.

## Discipline closure

- **Catalog #229 premise verification**: read the producer memo + prior verdict memo + populator + sweep tool + scaffold + MLX artifact shape/values BEFORE editing; confirmed the per-pair path did NOT exist (it was documented as deferred) and IMPLEMENTED it cleanly with tests rather than leaving a NotImplementedError.
- **Catalog #192/#127/#323 canonical Provenance**: every per-pair cell carries `build_provenance_for_macos_cpu_advisory` (`evidence_grade=macos_cpu_advisory`, `promotion_eligible=false`, `score_claim_valid=false`, `artifact_kind=advisory_non_promotable`) + diagnostic `heuristic_prior=true` + violation reason; sweep payload + probe-outcome row all `[predicted]` non-promotable.
- **Catalog #287 evidence tags**: every empirical claim cites the artifact sha / sweep payload / numpy inspection; no hardcoded scores (frontier read from the artifact meta + sweep constants).
- **Catalog #303 cargo-cult audit FLIP**: assumption 4 flipped from CARGO-CULTED-PENDING-EMPIRICAL to HEURISTIC-PRIOR-LANDED in the populator docstring.
- **Catalog #313 probe-outcomes**: DEFER verdict LANDED (heuristic-prior reactivation; advisory, 30-day staleness).
- **Catalog #110/#113 APPEND-ONLY**: NEW memo + NEW function + NEW test file + NEW sweep flag; NO mutation of the prior verdict memo, the producer memo, or sister artifacts.
- **Catalog #314/#340 sister-checkpoint guard**: scope disjoint from the PACT-NeRV sister; commit via canonical serializer with POST-EDIT `--expected-content-sha256`.
- **Catalog #206 checkpoint discipline**: 4 in-progress + 1 complete checkpoints via `tools/subagent_checkpoint.py`.

---

## APPEND-ONLY CORRECTION FOOTER (2026-05-27T15:40Z) — RANK-1 TAUTOLOGY DIAGNOSED + FIXED + RE-RUN

- agent: claude (`rank1_fix_5d_canvas_20260527`) — explicit operator-routed fix of the rank-1 problem-spec tautology the rigor review `.omx/research/master_gradient_analysis_rigor_signal_review_20260527T151020Z.md` (commit `21014faa7`) exposed.
- lane_id: `lane_rank1_multiop_problem_spec_full_rank_fix_20260527`
- discipline: Catalog #110/#113 APPEND-ONLY (the body above is UNMODIFIED; this footer is the only addition) + #307 (paradigm-vs-implementation) + #192/#127/#323 (canonical Provenance non-promotable macOS-MLX research-signal) + #287 (every claim has a NUMBER).
- scope: $0 — MLX-local / numpy / CPU only. NO paid dispatch.

### The diagnosis the body above MIS-ATTRIBUTED

The body's verdict (`multi-op synergy negligible`) was DIRECTIONALLY right but its STATED MECHANISM — "the Dykstra polytope solver confirms the operator gradients are **near-collinear** at the frontier" (body §"CRITICAL HONESTY" point 3) — was a **rank-1 tautology of `_build_multiop_problem_spec`, NOT an empirical property of the score surface.** Per the rigor review CRUX 3: every operator gradient was built as `(seg_aggregate, pose_aggregate, rate_aggregate) × scalar_leverage_i` — all 5 productive operators were scalar multiples of ONE shared vector → operator-gradient matrix **RANK 1**, max cosine-distance **0.0**. A rank-1 operator basis has **zero synergy-axis polytope volume FOR ANY INPUT** (per-pair or aggregate, MLX or PyTorch-authority), so the synergy term `-7.6e-13` was the arithmetic image of a rank-1 input, NOT a measurement.

### The fix

`tools/canvas_multiop_composition_closed_form_prediction_sweep.py::_build_multiop_problem_spec` now derives each operator's per-axis gradient `(|d_seg|, |d_pose|, |d_rate|)` from that operator's OWN per-pair footprint via its best candidate's Catalog #356 `AxisDecomposition` (NEW helper `_operator_axis_gradient_from_decomposition`). Different operators touch different pairs/bytes/regions, so their aggregate axis DIRECTIONS are genuinely distinct. The legacy rank-1 construction is preserved behind `rank1_legacy=True` (CLI `--rank1-legacy`) for the apples-to-apples before/after comparison ONLY. Per-operator fallback to the shared-aggregate × leverage value when an operator lacks a decomposition (so the spec never crashes). NEW helper `operator_gradient_matrix_rank(spec)` is the regression surface. NEW test `src/tac/tests/test_canvas_multiop_problem_spec_rank.py` (10 tests, all pass) ASSERTS the full-rank spec produces rank > 1 with distinct footprints — so the rank-1 tautology cannot silently return — and that the legacy path remains rank 1.

### The genuine re-run verdict (full-600 artifact, FULL-RANK spec)

| quantity | rank-1 LEGACY (the tautology) | FULL-RANK (the fix) |
|---|---|---|
| `problem_spec_kind` | `rank1_legacy_shared_aggregate_x_scalar` | `full_rank_per_operator_axis_decomposition` |
| **operator-gradient matrix rank** | **1** | **3** |
| max pairwise cosine-distance | 0.0 | ~0.9997 (`reorder_pair` rate-axis footprint orthogonal to seg/pose-only ops) |
| Dykstra feasible / residual / iters | True / 1.25e-13 / 159 | True / 0.0 / 2 |
| **`predicted_multiop_extra_delta` (SYNERGY)** | **−1.46e-12** | **−2.13e-07** |
| V14-V2 frontier crossing threshold | −7.66e-6 | −7.66e-6 |
| synergy vs threshold | ~6 orders below | **35.9× below (2.78% of threshold)** |

**The genuine number: full-rank multi-op SYNERGY = −2.13e-07.** That is **1.5×10⁵× larger** than the rank-1 tautology's −1.46e-12 — the rank-1 collapse was annihilating the genuine signal by ~5 orders of magnitude. BUT the genuine synergy is still **35.9× BELOW** the V14-V2 frontier-crossing threshold (−7.66e-6); it is **2.78%** of the threshold. The synergy is dominated by `reorder_pair`'s rate-axis −8-byte footprint being orthogonal to the seg/pose-only operators (`replace_many` / `merge_pair` / `motion_conditional` / `temporal_coherence`).

### Per Catalog #307 — corrected classification

- The body's `MULTIOP_BEATS_V14V2` raw verdict label remains a **label-artifact** (the `predicted_multiop_delta_score` sum is dominated by `best_single_op_delta = −0.0359`, a LEVERAGE-magnitude sum, NOT a realizable score). UNCHANGED.
- The synergy≈0 EVIDENCE is **re-classified from `near-collinear-operator-gradients-at-the-frontier` (IMPLEMENTATION-LEVEL rank-1 tautology) to a GENUINE FULL-RANK MEASUREMENT: synergy is REAL (−2.13e-07) but ~36× too small to cross V14-V2 on its own.**
- The strategic conclusion (`single-op rate-attack is Pareto-optimal; class-shift is the only sub-single-op-Pareto lever`) **now holds on a genuine measurement, not a tautology.** The multi-op DISTORTION route does **NOT** reopen: synergy is real but sub-threshold by ~36×, so paid FIRE-phase spend on a multi-op composition remains NOT worth it. A single-op (drop-one / replace-one) candidate per Catalog #246 paired contest-CUDA/CPU exact-eval remains the only path to a realizable score; Track-A class-shift remains the only remaining lever for genuinely-new gains.

### Provenance (Catalog #323)

Every number above is `evidence_grade=macOS-MLX research-signal` / `axis_tag=[predicted]` / `score_claim=false` / `promotable=false` / `hardware_substrate=darwin_arm64_m5_max_macos_mlx_advisory`. Sources: two live full-600 sweep runs (`--rank1-legacy` vs default) + the live-artifact regression test. NOT a contest score; Catalog #246 paired exact-eval remains the only score/frontier/PR path.

### Catalog #313 probe-outcome row

`canvas_multiop_composition_per_pair_heuristic_prior_repoint_sweep_20260527` — a NEW corrected `adjudicated` row appended (APPEND-ONLY; the prior rank-1-metric row is preserved): `verdict=DEFER`, `metric_value=−2.13e-07` (FULL_RANK), `operator_gradient_matrix_rank_full=3` / `_legacy=1`, `full_rank_synergy_below_threshold_factor=35.9`.
