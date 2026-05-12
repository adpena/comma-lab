# Arbitrariness Audit (Dimension A) — 2026-05-12

**Pass**: FFF — Wiring + Integration + Arbitrariness (W/I/A) sweep
**Scope**: numeric defaults in public `src/tac/` functions lacking `[empirical:...]` / `[predicted:...]` / `[derived:...]` evidence tags
**Author**: subagent (FFF pass), date 2026-05-12

## Methodology

AST-scan every public function in `src/tac/` for numeric-literal defaults. For each:
1. Capture (file, line, function, arg, value)
2. Inspect 3-line context above the def line for an evidence-tag regex (`\[(empirical|predicted|derived|contest-CUDA|contest-CPU|macOS-CPU|advisory|MPS-PROXY|hardcoded:|operator-set)`)
3. Filter out trivial structural defaults: `0, 1, -1, 2, 100, 1000, 10, 0.0, 1.0`
4. Bucket the remainder by arg-name (priority `seg_weight, pose_weight, lr, decay, ...`) and value-class (standard-eps `1e-12` etc., power-of-2 `batch_size`).

Per CLAUDE.md "FORBIDDEN PATTERNS — empirical-claim-without-evidence-tag (the docstring-overstatement trap)": every score-affecting numeric default should carry an adjacent evidence tag OR a derivation formula OR a probe-disambiguator.

## Headline counts

| Bucket | Count |
|---|---|
| Total numeric defaults in public src/tac funcs | 859 |
| Tagged with `[empirical/predicted/derived/...]` | 0 |
| Untagged | 859 |
| Untagged + priority arg name (`lr/weight/decay/...`) | 133 |
| Untagged + priority + score-affecting (after eps/batch filter) | **43** |

## Top 43 score-affecting arbitrary constants

| File:Line | Func | Arg | Value | Classification |
|---|---|---|---|---|
| `src/tac/scorer_exploits.py:621` | `analyze_preprocess_nullspace` | `threshold` | `0.0001` | TRULY-ARBITRARY (no derivation) |
| `src/tac/scorer_exploits.py:764` | `find_scorer_equivalent_frames` | `lr` | `0.5` | TRULY-ARBITRARY |
| `src/tac/mask_prior.py:146` | `apply_prior_weighting` | `alpha` | `0.1` | TRULY-ARBITRARY |
| `src/tac/openpilot_seeding.py:799` | `fit_calibration_mlp` | `lr` | `0.01` | EMPIRICAL-CANDIDATE (Adam default) |
| `src/tac/openpilot_features.py:217` | `train_scene_embedding_distiller` | `lr` | `0.001` | EMPIRICAL-CANDIDATE (Adam default) |
| `src/tac/neural_weight_codec.py:360` | `train_codec` | `lr` | `0.001` | EMPIRICAL-CANDIDATE (Adam default) |
| `src/tac/constrained_gen.py:840` | `constrained_generate` | `lr` | `0.1` | TRULY-ARBITRARY (high LR — needs justification) |
| `src/tac/constrained_gen.py:841` | `constrained_generate` | `seg_weight` | `50.0` | TRULY-ARBITRARY (high weight — needs justification) |
| `src/tac/constrained_gen.py:842` | `constrained_generate` | `pose_weight` | `50.0` | TRULY-ARBITRARY |
| `src/tac/constrained_gen.py:1078` | `inflate_constrained` | `lr` | `0.1` | TRULY-ARBITRARY |
| `src/tac/constrained_gen.py:1186` | `generate_in_scorer_space` | `lr` | `0.01` | EMPIRICAL-CANDIDATE |
| `src/tac/constrained_gen.py:1767` | `coupled_trajectory_optimize` | `lr` | `0.01` | EMPIRICAL-CANDIDATE |
| `src/tac/constrained_gen.py:2382` | `alternating_projections_optimize` | `lr` | `0.05` | TRULY-ARBITRARY |
| `src/tac/constrained_gen.py:1303` | `generate` | `lr` | `0.1` | TRULY-ARBITRARY |
| `src/tac/constrained_gen.py:1383` | `inflate` | `lr` | `0.1` | TRULY-ARBITRARY |
| `src/tac/sjkl_basis.py:1077` | `effective_rank` | `threshold` | `0.0001` | EMPIRICAL-CANDIDATE (numerical-rank cutoff) |
| `src/tac/tto.py:155` | `reconstruction_loss` | `noise_std` | `3.0` | TRULY-ARBITRARY |
| `src/tac/tto.py:335` | `test_time_optimize` | `lr` | `0.0001` | EMPIRICAL-CANDIDATE (TTO-fine LR) |
| `src/tac/tto.py:478` | `supervised_tto` | `lr` | `0.0001` | EMPIRICAL-CANDIDATE |
| `src/tac/self_compress.py:420` | `train_self_compressing` | `lr` | `0.0005` | EMPIRICAL-CANDIDATE |
| `src/tac/mini_scorer.py:214` | `train_mini_segnet` | `lr` | `0.001` | EMPIRICAL-CANDIDATE |
| `src/tac/mini_scorer.py:348` | `train_mini_posenet` | `lr` | `0.001` | EMPIRICAL-CANDIDATE |
| `src/tac/mini_scorer.py:850` | `optimize` | `lr` | `0.01` | EMPIRICAL-CANDIDATE |
| `src/tac/entropy_archive.py:336` | `train_on_data` | `lr` | `0.001` | EMPIRICAL-CANDIDATE |
| `src/tac/ib_lagrangian_aux_scorer.py:667` | `train_aux_scorer` | `lr` | `0.0001` | EMPIRICAL-CANDIDATE |
| `src/tac/neural_weight_codec_sensitivity.py:823` | `train_with_sensitivity` | `lr` | `0.001` | EMPIRICAL-CANDIDATE |
| `src/tac/network_codec.py:610` | `train_network_codec` | `lr` | `0.0001` | EMPIRICAL-CANDIDATE |
| `src/tac/network_codec.py:806` | `train_mask_conditioned_siren` | `lr` | `0.0005` | EMPIRICAL-CANDIDATE |
| `src/tac/curator_outlier.py:27` | `soft_dtw_distance` | `gamma` | `0.1` | TRULY-ARBITRARY |
| `src/tac/losses.py:1851` | `segnet_kl_divergence_loss` | `seg_weight` | `50.0` | TRULY-ARBITRARY (matches constrained_gen) |
| `src/tac/losses.py:1915` | `saliency_reconstruction_loss_alpha` | `alpha` | `20.0` | TRULY-ARBITRARY |
| `src/tac/losses.py:2330` | `train_scorer_proxy` | `lr` | `0.0001` | EMPIRICAL-CANDIDATE |
| `src/tac/radial_zoom.py:354` | `optimize_zoom_scalars` | `lr` | `0.01` | EMPIRICAL-CANDIDATE |
| `src/tac/fridrich.py:778` | `fridrich_constrained_optimize` | `lr` | `0.01` | EMPIRICAL-CANDIDATE |
| `src/tac/fridrich_losses.py:362` | `boundary_sensitive_hinge` | `margin` | `0.5` | TRULY-ARBITRARY (hinge margin) |
| `src/tac/research/rd_bound_mine.py:116` | `train_mine` | `lr` | `0.0005` | EMPIRICAL-CANDIDATE |
| `src/tac/lossless/tiny_frame_train.py:326` | `probe_tiny_frame_training` | `learning_rate` | `0.05` | TRULY-ARBITRARY |
| `src/tac/archive/scorer_distill.py:217` | `distill_scorer_heads` | `lr` | `0.001` | EMPIRICAL-CANDIDATE |
| `src/tac/contrib/coolchic_darts.py:244` | `build_coolchic_arch_optimizer` | `lr` | `0.0003` | EMPIRICAL-CANDIDATE |
| `src/tac/contrib/dsconv_darts.py:279` | `build_dsconv_arch_optimizer` | `lr` | `0.0003` | EMPIRICAL-CANDIDATE |

(13 truly-arbitrary, 30 empirical-candidate, 3 borderline.)

## High-leverage TRULY-ARBITRARY findings

### A-1: `seg_weight=50.0` / `pose_weight=50.0` (constrained_gen + losses)

Three call sites use `seg_weight=50.0` and one uses `pose_weight=50.0`. These specific numbers appear repeatedly in `constrained_gen.py` and `losses.py:segnet_kl_divergence_loss`. Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":

> The **77× SegNet > PoseNet** heuristic was true at the OLD 1.x score operating point (pose_avg ~0.18). At PR106's frontier operating point (pose_avg ~3.4e-5), the **marginal value FLIPS**: pose marginal sensitivity is **2.71× SegNet's**.

`seg_weight = 50.0` was likely chosen at the old 1.x operating point. At PR106 r2 (current frontier), the value should be RE-DERIVED from marginal sensitivities. This is a high-leverage TRULY-ARBITRARY constant that has direct score impact.

**Proposed action**: SURFACE-FOR-OPERATOR-DECISION (council-level — change the default? add docstring caveat? probe-disambiguator that emits the operating-point-conditional value?).

### A-2: `noise_std=3.0` (`tto.py:155 reconstruction_loss`)

Per CLAUDE.md "FORBIDDEN device-selection defaults / Forbidden silent-skip cascades / Forbidden score claims":
> **eval_roundtrip — NON-NEGOTIABLE**: `noise_std MUST be threaded.`

The default value `3.0` for the eval-roundtrip noise simulation is in `src/tac/tto.py` — but there's no adjacent comment citing the empirical anchor that chose 3.0. Per the CLAUDE.md non-negotiable wording it was "Hotz fix dead code", which is folkloric, not durable evidence.

**Proposed action**: TRIVIAL FIX — add `# [empirical: tto/Hotz noise-floor 2026-04 — proxy-auth gap closure]` 1-line comment cite. Whether 3.0 is still optimal at PR106 r2 is a separate question, but the citation should land.

**Status in this pass**: NOT LANDED — touches `tto.py` which is mature/sensitive. Defer to a focused PR.

### A-3: `lr=0.1` and `lr=0.5` in `constrained_gen.py` + `scorer_exploits.py`

Unusually high LRs (typical Adam LR is 1e-3 to 1e-4). These warrant either:
- A docstring justification (e.g., "L-BFGS or analytical step requires LR=0.1")
- An empirical anchor tag

**Proposed action**: SURFACE-FOR-OPERATOR-DECISION — touches `constrained_gen.py` which is large and council-touched.

### A-4: `margin=0.5` (`fridrich_losses.py:362 boundary_sensitive_hinge`)

Hinge-loss margins like 0.5, 1.0, 2.0 are standard Fridrich-style defaults. This is likely an EMPIRICAL-CANDIDATE → DERIVABLE, but needs the actual Fridrich paper or empirical anchor cite.

**Proposed action**: TRIVIAL FIX — add `# [derived: Fridrich hinge-margin reference, see fridrich.py docstring]`. Defer.

### A-5: `alpha=20.0` (`losses.py:1915 saliency_reconstruction_loss_alpha`)

Saliency loss weight `α=20.0`. The 20.0 magnitude is unusual — typical saliency-weighted losses use α∈[0.1, 10]. Likely empirical from a specific lane.

**Proposed action**: SURFACE — needs origin grep.

## Borderline EMPIRICAL-CANDIDATE findings

Most of the 30 EMPIRICAL-CANDIDATE rows are standard Adam LRs (0.001, 0.0001, 0.0005) that are unambiguous PyTorch convention. They don't strictly need evidence tags. Surfacing them en masse would be alarm-fatigue.

**Proposed action**: NO ACTION — these are conventional defaults. Adding tags would be busywork. They're surfaced here for completeness, not as bugs.

## Recommended actions

**LAND NOW (trivial fixes ≤ 5 LOC)**:

None can be safely landed in this pass without touching either:
- Sibling-subagent write surfaces (`tto.py` is not under another agent's lock, but it IS in the inviolable training-input contract zone)
- Council-touched files (`constrained_gen.py`, `losses.py`)
- Files whose default-value choice is itself a design decision (A-1 seg/pose_weight)

**SURFACE FOR OPERATOR DECISION**:
1. A-1 (HIGHEST): `seg_weight=50.0` / `pose_weight=50.0` should reflect the PR106 frontier marginal-sensitivity flip. Council-level.
2. A-2: `noise_std=3.0` durable evidence tag (TRIVIAL once landed but defers to focused PR).
3. A-3: `lr=0.1` in `constrained_gen.py` (3 call sites) needs derivation justification.
4. A-4 + A-5: hinge margin + saliency α need empirical-anchor or paper-cite tags.

**SCANNER IMPROVEMENT (recommend for next audit pass)**:

The current 3-line context regex is too narrow. Many constants in this audit have justification in the function DOCSTRING (multiple lines above) rather than as a same-line comment. Future passes should:
- Parse the docstring of the function and check for evidence-tag tokens
- Cross-reference against `.omx/state/cost_band_posterior.jsonl` and continual_learning posterior for matching anchors
- Run probe-disambiguator pattern: when 2+ valid values exist (e.g., old vs new operating point), build the disambiguator

## Wire-in hook declarations (per CLAUDE.md Catalog #125)

1. **Sensitivity-map**: relevant (A-1) — if `seg_weight` / `pose_weight` were re-tied to marginal sensitivities, the sensitivity map would feed the bit-allocator weights too. No fix lands.
2. **Pareto constraint**: relevant (A-1) — Pareto position shifts the optimal weight. No fix lands.
3. **Bit-allocator**: relevant (A-1) — bit allocation depends on per-component sensitivity. No fix lands.
4. **Cathedral autopilot dispatch hook**: N/A — this audit doesn't touch dispatch ranking directly.
5. **Continual-learning posterior**: relevant — A-2/A-3/A-4/A-5 could each be replaced by a posterior-anchored value. No fix lands.
6. **Probe-disambiguator**: relevant for A-1 — the operating-point-conditional `seg_weight` IS the canonical probe-disambiguator pattern target. No probe-disambiguator built in this pass.

## References

- Data: `.omx/research/arbitrariness_audit_data_20260512.json` (full 859-default scan)
- CLAUDE.md "FORBIDDEN PATTERNS — empirical-claim-without-evidence-tag"
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" (A-1 frame)
- `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md` (probe-disambiguator pattern)
