# Codex Routing Directive — TOP-1 Arbitrariness Extinction: λ_seg/λ_pose/λ_rate Lagrange Multipliers

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Captured at UTC**: `2026-05-18T21:15:00Z`
**Value ID**: `lambda_seg_pose_rate_multipliers_unprincipled`
**Resolution path**: `analytical_solve`
**Predicted ΔS**: [-0.012, -0.003] (per CLAUDE.md SegNet-vs-PoseNet operating-point-dependent rule)
**Cost envelope**: $0 (closed-form derivation)
**Rank score per dollar**: 12.0 (HIGHEST)

## Operator standing directive

> "the concept I see is identifying arbitrariness or less than optimal being applied across the board and using all techniques and exploits and contest rules and allowed and everything to either experimentally determine the proper solution or solve it and use the optimal solution or use a formula instead or learn and train against the values or use neural or self or some other alien tech or combination of teks"

— operator 2026-05-18

## Bug class

Across all ~30 substrate trainers, `λ_seg`, `λ_pose`, `λ_rate` multipliers in the score-aware loss are hand-tuned per substrate (e.g. nscs02 `args.seg_weight + args.pose_weight`, cool_chic `ar_rate_weight`) without principled basis. The CONTEST FORMULA fixes their relationship:

```
total_score = sqrt(10 * pose_avg) + 100 * seg_avg + 25 * archive_bytes / 37545489
```

So the **gradient-matching** multipliers at any operating point are CLOSED-FORM:

```
∂score/∂pose_avg = 5 / sqrt(10 * pose_avg)
∂score/∂seg_avg  = 100  (constant)
∂score/∂rate     = 25 / 37545489
```

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)": at PR106 frontier (pose_avg ~3.4e-5), the marginal ratio `d(pose)/d(seg) = 5 / (sqrt(10*3.4e-5) * 100) = 2.71`. So `λ_pose` should be ~**2.71× λ_seg** at frontier — INVERTED from the old 1.x operating point's 77× SegNet > PoseNet rule.

Current ad-hoc `λ` values almost certainly do NOT reflect this operating-point-dependent flip.

## 5-path analysis

1. **experimental** — sweep `λ_pose / λ_seg ∈ {0.1, 0.3, 1.0, 3.0, 10.0}` per substrate; pick best. Cost: $30+. **REJECTED**: closed-form available.
2. **analytical_solve** [RECOMMENDED] — derive `λ_axis = ∂score/∂axis_avg` at current operating point. Cost: $0.
3. **formula** — same as #2 but identifies the formula source (Lagrange multipliers for constrained R-D optimization per Boyd-Vandenberghe Ch.5).
4. **learned** — sister Catalog row `score_pair_components_weights_static` proposes uncertainty-weighted multi-task loss (Kendall et al 2018 arxiv:1705.07115). COMPLEMENTARY: derive analytic baseline; learn perturbations around it.
5. **self_alien_tech** — N/A (would be Volterra-series or higher-order interactions; out-of-scope until baseline closed-form lands).

## Concrete next step ($0)

Land canonical helper `tac.score_lagrangian`:

```python
def compute_marginal_multipliers(
    *,
    operating_point: ScorePoint,  # {pose_avg, seg_avg, archive_bytes}
    normalize_to: str = "seg",   # "seg" | "pose" | "rate"
) -> dict[str, float]:
    """Closed-form Lagrange multipliers for the contest score formula.

    Per CLAUDE.md "SegNet vs PoseNet — operating-point-dependent":
      d(score)/d(pose_avg) = 5 / sqrt(10 * pose_avg)
      d(score)/d(seg_avg)  = 100
      d(score)/d(archive_bytes) = 25 / 37545489

    Returns {lambda_pose, lambda_seg, lambda_rate} normalized to one axis.
    """
```

Wire into every substrate trainer's score-aware loss as the DEFAULT λ source.
Per-substrate overrides allowed via `--lambda-source manual` + explicit `--lambda-seg/pose/rate`.

## Sister coordination

- **Catalog #322** (composition α v2 cascade): the multipliers feed Cascade-2 reward decisions
- **Catalog #164** (canonical scorer-loss helper routing): wires into `tac.substrates._shared.score_aware_common.score_pair_components`
- **CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable**: this IS the canonical instance

## Exit criteria

1. `src/tac/score_lagrangian.py` canonical helper (~150 LOC + 20 tests)
2. `score_pair_components` defaults to canonical helper
3. Per-substrate trainers use `--lambda-source canonical` by default
4. Empirical anchor: paired smoke on smallest substrate confirms predicted ΔS lower bound

## Provenance

`evidence_grade=predicted` per Catalog #323. Anchor empirically before any score claim.
